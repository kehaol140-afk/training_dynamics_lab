#!/usr/bin/env python3
"""GTTR-CB1 runner —— preflight(CB0 离线) / smoke / main(CB1 重放)。

协议：FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md（2026-09-24）。

CB0（纯离线，读 R2B 36 轨迹）：
  - 从全精度 monitor 序列重建 OLD_GRID16_LAG16，核对旧 hit/阈值；
  - alive 加权和成本 vs direct 成本恒等式；
  - 三臂配对成本差分 seed/start/stream 列出；
  - landmark16/32 past-only gap/rho 与旧网格预定义预测。

CB1（GPU 确定性重放）：
  - 每 state 取数值 ID 较小 stream，3 臂共 18 条；从原 h1 full-state
    （theta/m/v/step/RNG）恢复，不重新施加动作；h2..T16 全部原生 AdamW k=2；
  - 4 步观测网格、16 步确认间隔；补齐 [8,T16] 全部非 16 倍数点；
    h16 与 T16 两处重叠复核（位级一致预期）；source=OLD_REUSED/
    REPLAY_NEW/REPLAY_OVERLAP；
  - 预算硬上限：replay updates<=4800；total fw<=125000；bw<=9600；
    monitor 整库 eval<=900；新增 q/HVP/probe=0；V_ctrl/V_final=0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_cb1_cost_measurement_v01"
RUNS = OUT / "runs"
MON_ROWS = OUT / "monitor_rows"
R2B = PROJECT_ROOT / "outputs" / "fr_gttr_r2b_solved_update_bridge_v01"
R1_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md"

SEEDS = (2030, 2031, 2032)
STARTS = (46, 82)
CELLS = [(s, t) for s in SEEDS for t in STARTS]
ARMS = ("BASE_K2", "GTTR_LR_ONLY", "GTTR_FULL")

CAP_REPLAY_UPDATES = 4800
CAP_TOTAL_FW = 125000
CAP_BW = 9600
CAP_MON_EVALS = 900

SLOTS = 1024

_C = {"replay_updates": 0, "train_fw": 0, "train_bw": 0, "monitor_fw": 0,
      "monitor_evals": 0, "q_hvp_probe": 0, "v_ctrl_calls": 0,
      "v_final_calls": 0, "failures": 0}
_T0 = None


def _total_fw():
    return _C["train_fw"] + _C["monitor_fw"]


def _replay_ledger():
    p = OUT / "resource_log.jsonl"
    if not p.exists():
        return
    last = None
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                continue
    if last:
        for k in _C:
            if k in last:
                _C[k] = int(last[k])


def _log(phase, note="", **kw):
    rec = {"ts": time.time(), "phase": phase, "note": note,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "total_fw": _total_fw(), **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(up_updates=0, up_fw=0, up_bw=0, up_mon=0):
    if _C["replay_updates"] + up_updates > CAP_REPLAY_UPDATES:
        raise SystemExit(f"BUDGET_BLOCKED: replay updates {CAP_REPLAY_UPDATES}")
    if _total_fw() + up_fw > CAP_TOTAL_FW:
        raise SystemExit(f"BUDGET_BLOCKED: total fw {CAP_TOTAL_FW}")
    if _C["train_bw"] + up_bw > CAP_BW:
        raise SystemExit(f"BUDGET_BLOCKED: bw {CAP_BW}")
    if _C["monitor_evals"] + up_mon > CAP_MON_EVALS:
        raise SystemExit(f"BUDGET_BLOCKED: monitor evals {CAP_MON_EVALS}")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# R2B 制品读取
# ---------------------------------------------------------------------------

def load_r2b_monitor():
    """{(seed,start,stream,arm): {h: (sum,count,mean,npz)}}（去重保留最后）。"""
    r2b_man = json.loads(
        (R2B / "state_stream_action_manifest.json").read_text())
    main_streams = {str(s) for c in r2b_man["state_stream_action"]
                    for s in c["stream_seeds"]}
    by = {}
    p = R2B / "monitor_metrics.csv"
    lines = p.read_text().splitlines()
    header = lines[0].split(",")
    for line in lines[1:]:
        if not line.strip():
            continue
        r = dict(zip(header, line.split(",")))
        if r["arm"] not in ARMS or r["stream"] not in main_streams:
            continue
        key = (r["seed"], r["start"], r["stream"], r["arm"])
        by.setdefault(key, {})[int(r["h"])] = (
            float(r["sum"]), int(r["count"]), float(r["mean"]), r["npz"])
    # 共享 h0
    for line in lines[1:]:
        if not line.strip():
            continue
        r = dict(zip(header, line.split(",")))
        if r["arm"] == "shared" and r["stream"] == "shared":
            for key in list(by):
                if key[0] == r["seed"] and key[1] == r["start"]:
                    by[key][0] = (float(r["sum"]), int(r["count"]),
                                  float(r["mean"]), r["npz"])
    return by


def load_r2b_detection():
    p = R2B / "target_detection.csv"
    lines = p.read_text().splitlines()
    header = lines[0].split(",")
    out = {}
    for line in lines[1:]:
        r = dict(zip(header, line.split(",")))
        hd = r["h_detect"]
        out[(r["seed"], r["start"], r["stream"], r["arm"])] = (
            r["status"], int(hd) if hd not in ("None", "NA", "") else None)
    return out


def load_r2b_valid_tokens():
    p = R2B / "token_counts.csv"
    lines = p.read_text().splitlines()
    header = lines[0].split(",")
    out = {}
    for line in lines[1:]:
        r = dict(zip(header, line.split(",")))
        v = r["valid_tokens_at_detect"]
        out[(r["seed"], r["start"], r["stream"], r["arm"])] = (
            int(v) if v not in ("NA", "") else None)
    return out


def load_ell():
    import csv as _csv
    with (PROJECT_ROOT / "outputs/fr_gttr_phase_a_native_calibration_v01/a1"
          / "target_calibration.csv").open() as f:
        rows = list(_csv.DictReader(f))
    return {int(r["start"]): float(r["ell_candidate"]) for r in rows}


# ---------------------------------------------------------------------------
# preflight = CB0 离线分析 + CB1 计划
# ---------------------------------------------------------------------------

def run_preflight():
    global _T0
    _T0 = time.perf_counter()
    from fr import cb1_cost_v01 as cb
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    MON_ROWS.mkdir(parents=True, exist_ok=True)

    # 0) CPU 测试
    test_mod = "tests.test_cb1_cost_v01"
    r = subprocess.run([sys.executable, "-m", "unittest", test_mod],
                       cwd=PROJECT_ROOT / "research", capture_output=True,
                       text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                            "HOME": "/home/b205"})
    out_txt = r.stdout + r.stderr
    passed = r.returncode == 0 and out_txt.strip().splitlines()[-1].strip() == "OK"
    (OUT / "implementation_checks.json").write_text(json.dumps({
        "unittest_module": test_mod, "all_pass": bool(passed),
        "tail": out_txt.strip().splitlines()[-1].strip()}, indent=1))
    assert passed, f"CPU tests failed:\n{out_txt[-1500:]}"

    ell = load_ell()
    assert abs(ell[46] - 7.020043501537879) < 1e-12
    assert abs(ell[82] - 6.979538126804536) < 1e-12
    mon = load_r2b_monitor()
    det_old = load_r2b_detection()
    vtok = load_r2b_valid_tokens()
    assert len(mon) == 36, f"expect 36 trajectories, got {len(mon)}"

    # 1) 旧 hit 重建 + 成本恒等式
    ident_rows = []
    max_resid = 0.0
    mismatch = 0
    for key in sorted(mon, key=lambda k: (int(k[0]), int(k[1]), int(k[2]),
                                          ARMS.index(k[3]))):
        seed, start, stream, arm = key
        e = ell[int(start)]
        means = {h: v[2] for h, v in mon[key].items()}
        grid16 = [h for h in range(0, 257, 16) if h in means]
        rec = cb.alive_capped_cost(means, e, grid16)
        old_status, old_hd = det_old[key]
        match = (rec["h_detect"] == old_hd)
        if not match:
            mismatch += 1
        max_resid = max(max_resid, abs(rec["residual"]))
        ident_rows.append((seed, start, stream, arm, repr(e), old_hd,
                           rec["h_detect"], match, rec["cost_alive"],
                           rec["cost_direct"], repr(rec["residual"]),
                           rec["censor"], rec["cost_alive"] / SLOTS))
    with (OUT / "old_cost_identity.csv").open("w") as f:
        f.write("seed,start,stream,arm,ell,old_h_detect,rebuilt_h_detect,"
                "match,cost_alive_slots,cost_direct_slots,residual,censor,"
                "cost_alive_steps\n")
        for r_ in ident_rows:
            f.write(",".join(str(x) for x in r_) + "\n")
    assert mismatch == 0, f"{mismatch} rebuilt hits differ from R2B records"

    # 2) 三臂配对成本差（分 seed/start/stream）
    cost = {k: r_[8] for k, r_ in zip(sorted(mon, key=lambda k: (
        int(k[0]), int(k[1]), int(k[2]), ARMS.index(k[3]))), ident_rows)}
    pair_rows = []
    pairs = sorted({(k[0], k[1], k[2]) for k in mon},
                   key=lambda p: (int(p[0]), int(p[1]), int(p[2])))
    for (seed, start, stream) in pairs:
        base = cost[(seed, start, stream, "BASE_K2")]
        lr = cost[(seed, start, stream, "GTTR_LR_ONLY")]
        fu = cost[(seed, start, stream, "GTTR_FULL")]
        vb = vtok[(seed, start, stream, "BASE_K2")]
        vl = vtok[(seed, start, stream, "GTTR_LR_ONLY")]
        vf = vtok[(seed, start, stream, "GTTR_FULL")]

        def lab(d):
            return "FULL_BETTER" if d < 0 else ("LR_BETTER" if d > 0 else "TIE")
        pair_rows.append((seed, start, stream,
                          fu - lr, lab(fu - lr), fu - base, lab(fu - base),
                          lr - base, lab(lr - base),
                          (vf - vl) if (vf is not None and vl is not None)
                          else "NA"))
    with (OUT / "paired_cost_effect.csv").open("w") as f:
        f.write("seed,start,stream,cost_FULLminusLR_slots,label_FL,"
                "cost_FULLminusBASE_slots,label_FB,cost_LRminusBASE_slots,"
                "label_LB,valid_tokens_FULLminusLR\n")
        for r_ in pair_rows:
            f.write(",".join(str(x) for x in r_) + "\n")

    # 3) past-only gap/rho 诊断（旧 16 步网格版，全部 36 条）
    diag_rows = []
    for key in sorted(mon, key=lambda k: (int(k[0]), int(k[1]), int(k[2]),
                                          ARMS.index(k[3]))):
        seed, start, stream, arm = key
        e = ell[int(start)]
        means = {h: v[2] for h, v in mon[key].items()}
        T16 = det_old[key][1]
        # 描述量：首达前各 16 步窗口进展的最小/最大
        win = [means[h - 16] - means[h] for h in range(16, 257, 16)
               if h in means and h - 16 in means and h <= T16]
        for l in (16, 32):
            if l not in means or (l - 16) not in means:
                continue
            confirmed = (T16 is not None and T16 <= l)
            g = cb.gap_rho(means[l - 16], means[l], e, l * SLOTS,
                           (l - 16) * SLOTS)
            if confirmed or g["gap"] <= 0:
                reason = ("ALREADY_CONFIRMED" if confirmed
                          else "NONPOSITIVE_GAP")
                diag_rows.append((seed, start, stream, arm, l, repr(g["gap"]),
                                  repr(g["rho"]), reason, "", "", "", "",
                                  "", ""))
                continue
            p16 = cb.predict_times(l, g["gap"], g["rho"], 16)
            if p16["abstain"]:
                diag_rows.append((seed, start, stream, arm, l,
                                  repr(g["gap"]), repr(g["rho"]),
                                  p16["abstain"], "", "", "", "", "", ""))
                continue
            actual_remaining = (T16 - l) * SLOTS
            pred_rem = p16["predicted_remaining_tokens"]
            diag_rows.append((
                seed, start, stream, arm, l, repr(g["gap"]), repr(g["rho"]),
                "OK", repr(p16["raw_cross_tokens_after_l"]),
                p16["t_pred_first"], p16["t_pred_confirm"],
                actual_remaining, pred_rem,
                abs(actual_remaining - pred_rem),
                p16["forecast_beyond_budget"],
                min(win) if win else "", max(win) if win else ""))
    with (OUT / "progress_cost_diagnostic.csv").open("w") as f:
        f.write("seed,start,stream,arm,landmark,gap,rho,status,"
                "raw_cross_tokens,t_pred_first16,t_pred_confirm16,"
                "actual_remaining_slots,pred_remaining_slots,abs_err,"
                "forecast_beyond_budget,min_window_progress,"
                "max_window_progress\n")
        for r_ in diag_rows:
            f.write(",".join(str(x) for x in r_) + "\n")

    # 4) CB1 选择与计划（以原 manifest 为准；数值 ID 较小 stream）
    r2b_man = json.loads(
        (R2B / "state_stream_action_manifest.json").read_text())
    sel = {}
    for c in r2b_man["state_stream_action"]:
        sel[(c["seed"], c["start"])] = cb.first_stream_of(c["stream_seeds"])
    sel_rows = []
    plan_updates = plan_evals = 0
    for (seed, start) in CELLS:
        stream = sel[(seed, start)]
        for arm in ARMS:
            key = (str(seed), str(start), str(stream), arm)
            T16 = det_old[key][1]
            assert T16 is not None
            new_pts = cb.new_points_to_measure(T16)
            n_ov = 2 if T16 != 16 else 1
            snap = R2B / f"snapshots/s{seed}_t{start}_st{stream}_{arm}_h001.pt"
            assert snap.exists(), snap
            plan_updates += T16 - 1
            plan_evals += len(new_pts) + n_ov
            sel_rows.append((seed, start, stream, arm, T16, T16 - 1,
                             len(new_pts), n_ov, str(snap),
                             _sha256(snap)))
    with (OUT / "selected_replays.csv").open("w") as f:
        f.write("seed,start,stream,arm,T16_old,replay_updates,n_new_points,"
                "n_overlap, snapshot_h1,snapshot_sha256\n")
        for r_ in sel_rows:
            f.write(",".join(str(x) for x in r_) + "\n")

    # 5) config + PROTOCOL_AND_SCOPE
    smoke_updates = 16
    smoke_evals = 3
    config = {
        "round": "fr_gttr_cb1_cost_measurement_v01",
        "protocol": "FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md",
        "protocol_sha256": _sha256(PROTOCOL),
        "selected_rule": "每 state 原两条 stream 中数值 ID 较小者（预算范围，"
                         "非择优）；另一路仍在 CB0 全表",
        "ell_by_start": {str(k): v for k, v in ell.items()},
        "detection": {"old": "OLD_GRID16_LAG16（保持原判定）",
                      "new": "REFINED_GRID4_LAG16（确认间隔仍 16 updates；"
                             "REPLAY_RESOLUTION_DIAGNOSTIC）",
                      "reuse": "h0/h4/16 倍数用旧同轴值；8,12,20,... 新测"},
        "replay": {"restore": "原 h1 full-state（theta/m/v/step/RNG），"
                              "restore_rng=True；不重新施加 R1 动作",
                   "rule": "h2..T16 全部原生 AdamW k=2",
                   "overlap_checks": "每条 h16 与 T16 两处（位级一致预期）",
                   "smoke_stream": "202609240300 之外不用；smoke 用首条选定"
                                   "BASE 的 h1 快照"},
        "budget": {"caps": {"replay_updates": CAP_REPLAY_UPDATES,
                            "total_fw": CAP_TOTAL_FW, "bw": CAP_BW,
                            "monitor_evals": CAP_MON_EVALS,
                            "q_hvp_probe": 0, "v_ctrl": 0, "v_final": 0},
                   "plan": {"main_updates": plan_updates,
                            "smoke_updates": smoke_updates,
                            "main_monitor_evals": plan_evals,
                            "smoke_monitor_evals": smoke_evals}},
        "sources": {"r2b_monitor": {"path": str(R2B / "monitor_metrics.csv"),
                                    "sha256": _sha256(
                                        R2B / "monitor_metrics.csv")},
                    "r2b_manifest": {"sha256": _sha256(
                        R2B / "state_stream_action_manifest.json")},
                    "r1_solutions": {"sha256": _sha256(
                        R1_OUT / "restricted_solutions.csv")}}}
    (OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (OUT / "PROTOCOL_AND_SCOPE.md").write_text(
        "# CB1 协议与范围\n\n本目录实现 FR_GTTR_CB1_Cost_Measurement_and_"
        "Progress_Link_v1.md：CB0 纯离线（36 条 R2B 旧轨迹的成本恒等式与"
        "past-only 进展诊断）＋CB1 固定首 stream 18 条的确定性重放（4 步观测"
        "网格、16 步确认间隔，REPLAY_RESOLUTION_DIAGNOSTIC）。\n\n"
        "- 不增加优化器/预测器；研究一次 R1 已求解首步改动后共同 AdamW 基线"
        "下的成本读出。\n"
        "- 训练 token 轴（轨迹 elapsed slots）与重放研究成本分开记账。\n"
        "- rho<=0 -> NO_POSITIVE_PAST_PROGRESS；预测不截断；FORECAST_BEYOND_"
        "BUDGET 保留。\n"
        "- 旧 16 步判定保持不变；新读数不回写 R2B 成功结论。\n")
    _log("preflight", "CB0 done", mismatch=0, max_resid=max_resid,
         plan_updates=plan_updates + smoke_updates,
         plan_evals=plan_evals + smoke_evals)
    print(json.dumps({"mismatch": mismatch, "max_residual": max_resid,
                      "n_identity_rows": len(ident_rows),
                      "n_pairs": len(pair_rows),
                      "n_diag_rows": len(diag_rows),
                      "selected": {f"{k[0]}/{k[1]}": v
                                   for k, v in sel.items()},
                      "plan": config["budget"]["plan"]}, indent=1))
    print("PREFLIGHT_OK")


# ---------------------------------------------------------------------------
# 重放内核（原生 AdamW k=2，无干预）
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    from fr import a0k_closed_loop_runner_v01 as a0k
    rt = a0k.build_live_runtime(seed, device)
    assert len(rt["optimizer"].param_groups) == 1
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    rt["eta"] = make_phase_a_get_lr(rt["get_lr"])
    return rt


def restore_h1(rt, snap_path):
    from fr.full_state import load_full_training_state
    load_full_training_state(snap_path, rt["model"], rt["optimizer"],
                             rt["scaler"], restore_rng=True)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)


def make_stream(rt, stream_seed, n_micro):
    import b0_run_v01 as b0
    return {"stream_seed": int(stream_seed),
            **b0.make_stream(rt, int(stream_seed), n_micro)}


def stack_batch_rt(rt, batch_indices):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], batch_indices, rt["device"])


def make_monitor_chunks(rt):
    import a0g_decision_utility_v01 as a0g
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    mon = PROJECT_ROOT / "outputs/fr_gttr_phase_a_native_calibration_v01" \
        "/a1/data/v_monitor_rows4096_8192.jsonl"
    ds = PretrainDataset(str(mon),
                         AutoTokenizer.from_pretrained(
                             str(PROJECT_ROOT / "minimind" / "model")),
                         max_length=int(rt["cfg"]["max_seq_len"]))
    chunks = [list(range(i, i + 32)) for i in range(0, 4096, 32)]
    return a0g._stack_all(ds, chunks, rt["device"])


def monitor_full(rt, stacked, tag, h, npz_name=None):
    import numpy as np
    import torch.nn.functional as F
    from fr.rng import capture_rng_state, restore_rng_state
    torch = rt["torch"]
    model = rt["model"]
    _check_caps(up_mon=1, up_fw=len(stacked))
    rng_snap = capture_rng_state()
    was_training = model.training
    total_sum = 0.0
    total_cnt = 0
    row_ids, row_sums, row_cnts = [], [], []
    t0 = time.perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for ci, (x, y) in enumerate(stacked):
                with rt["autocast_ctx"]:
                    res = model(x)
                logits = res.logits.float()
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                per_sum = ce.sum(dim=1)
                per_cnt = (tgt != -100).sum(dim=1)
                for r_ in range(tgt.shape[0]):
                    row_ids.append(4096 + ci * 32 + r_)
                    row_sums.append(float(per_sum[r_]))
                    row_cnts.append(int(per_cnt[r_]))
                total_sum += float(per_sum.sum())
                total_cnt += int(per_cnt.sum())
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_snap)
    _C["monitor_fw"] += len(stacked)
    _C["monitor_evals"] += 1
    name = npz_name or f"{tag}_h{h:03d}.npz"
    np.savez_compressed(MON_ROWS / name,
                        row_id=np.asarray(row_ids, dtype=np.int32),
                        nll_sum=np.asarray(row_sums, dtype=np.float64),
                        valid_count=np.asarray(row_cnts, dtype=np.int32))
    return {"h": h, "sum": total_sum, "count": total_cnt,
            "mean": total_sum / max(total_cnt, 1), "npz": name}


def replay_steps(rt, seed, start, stream, h_from, h_to, eval_at=None,
                 tag=""):
    """原生 AdamW k=2 前进 h_from+1..h_to；在 eval_at 的 h 上做 monitor。"""
    import torch
    from fr.native_schedule import set_native_micro_lr
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    gm0 = 2 * start
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    eval_at = eval_at or {}
    recs = []
    for h in range(h_from + 1, h_to + 1):
        _check_caps(up_updates=1, up_fw=accum, up_bw=accum)
        for a in range(accum):
            micro_idx = 2 * (h - 1) + a + 1
            lr_base = set_native_micro_lr(optimizer, rt["eta"],
                                          gm0 + micro_idx, 256,
                                          float(cfg["learning_rate"]))
            optimizer.param_groups[0]["lr"] = 2.0 * lr_base
            xb, yb = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
            with rt["autocast_ctx"]:
                res = model(xb, labels=yb)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            _C["train_fw"] += 1
            _C["train_bw"] += 1
            if not math.isfinite(float(res.loss)):
                raise RuntimeError(f"non-finite train loss at h={h}")
        scaler.unscale_(optimizer)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["grad_clip"])).detach().cpu())
        if not math.isfinite(preclip):
            raise RuntimeError(f"non-finite preclip at h={h}")
        scaler.step(optimizer)                    # 共同基线：无干预
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _C["replay_updates"] += 1
        if h in eval_at:
            stacked = eval_at[h]
            recs.append(monitor_full(rt, stacked, tag, h))
    return recs


def state_bitwise_sig(rt):
    sd = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    ost = rt["optimizer"].state_dict()
    return {"theta": sd,
            "opt_step": {str(k): int(v["step"])
                         for k, v in ost["state"].items()},
            "lr": [g["lr"] for g in ost["param_groups"]]}


def sig_equal(a, b):
    import torch
    for n in a["theta"]:
        if not torch.equal(a["theta"][n], b["theta"][n]):
            return False, f"theta:{n}"
    if a["opt_step"] != b["opt_step"]:
        return False, "opt_step"
    if a["lr"] != b["lr"]:
        return False, "lr"
    return True, ""


# ---------------------------------------------------------------------------
# smoke（16 updates + 3 整库 eval）
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr.rng import capture_rng_state
    mon = load_r2b_monitor()
    device = f"cuda:{gpu}"
    seed, start, stream, arm = 2030, 46, 202609240200, "BASE_K2"
    snap = R2B / f"snapshots/s{seed}_t{start}_st{stream}_{arm}_h001.pt"
    rt = build_rt(seed, device)
    stacked = make_monitor_chunks(rt)
    old_h1 = mon[(str(seed), str(start), str(stream), arm)][1]
    old_h4 = mon[(str(seed), str(start), str(stream), arm)][4]
    checks = {"cell": f"{seed}/{start}/{stream}/{arm}", "updates": 16,
              "monitor_evals": 3}

    # off：8 步无评价
    restore_h1(rt, snap)
    stream8 = make_stream(rt, stream, 2 * 9)
    replay_steps(rt, seed, start, stream8, 1, 9)
    sig_off = state_bitwise_sig(rt)
    rng_off = capture_rng_state()

    # on：h1 复核 -> h4 复核 -> h8 新测 -> h9
    restore_h1(rt, snap)
    evals = {}
    r1 = monitor_full(rt, stacked, "smoke_on", 1,
                      npz_name="smoke_on_h001.npz")
    evals[1] = r1
    replay_steps(rt, seed, start, stream8, 1, 4)
    r4 = monitor_full(rt, stacked, "smoke_on", 4,
                      npz_name="smoke_on_h004.npz")
    evals[4] = r4
    replay_steps(rt, seed, start, stream8, 4, 8)
    r8 = monitor_full(rt, stacked, "smoke_on", 8,
                      npz_name="smoke_on_h008.npz")
    evals[8] = r8
    replay_steps(rt, seed, start, stream8, 8, 9)
    sig_on = state_bitwise_sig(rt)
    rng_on = capture_rng_state()
    same, why = sig_equal(sig_off, sig_on)

    def rng_eq(a, b):
        import torch
        if a["python_random"] != b["python_random"]:
            return False
        if not torch.equal(a["torch_cpu"], b["torch_cpu"]):
            return False
        ca, cb = a.get("torch_cuda_all"), b.get("torch_cuda_all")
        if ca is None or cb is None:
            return ca is cb
        return all(torch.equal(x, y) for x, y in zip(ca, cb))

    checks.update({
        "off_on_bitwise_equal": bool(same), "mismatch": why,
        "rng_equal": bool(rng_eq(rng_off, rng_on)),
        "h1_overlap_bitwise": evals[1]["sum"] == old_h1[0],
        "h1_old_sum": old_h1[0], "h1_new_sum": evals[1]["sum"],
        "h4_overlap_bitwise": evals[4]["sum"] == old_h4[0],
        "h4_old_sum": old_h4[0], "h4_new_sum": evals[4]["sum"],
        "h8_new_mean": evals[8]["mean"],
        "opt_step_after": int(
            next(iter(rt["optimizer"].state.values()))["step"])})
    checks["verdict"] = "PASS" if (
        same and checks["rng_equal"] and checks["h1_overlap_bitwise"]
        and checks["h4_overlap_bitwise"]) else "FAIL"
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1))
    _log("smoke_done", checks["verdict"])
    print(json.dumps(checks, indent=1))
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main（18 条重放）
# ---------------------------------------------------------------------------

def run_main(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import cb1_cost_v01 as cb
    mon = load_r2b_monitor()
    det_old = load_r2b_detection()
    device = f"cuda:{gpu}"
    sel = [tuple(r[:4]) + (int(r[4]), r[8]) for r in (
        l.split(",") for l in
        (OUT / "selected_replays.csv").read_text().splitlines()[1:]
        if l.strip())]

    done = set()
    cells_path = RUNS / "replay_cells.jsonl"
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("completed"):
                        done.add((str(r["seed"]), str(r["start"]),
                                  str(r["stream"]), r["arm"]))
                except Exception:
                    continue

    dense_path = OUT / "dense_monitor.csv"
    if not dense_path.exists():
        dense_path.write_text(
            "seed,start,stream,arm,h,sum,count,mean,source,match_old\n")
    ident_path = OUT / "replay_identity.csv"
    if not ident_path.exists():
        ident_path.write_text(
            "seed,start,stream,arm,check_h,old_sum,new_sum,abs_diff,"
            "bitwise_equal,status\n")

    import torch
    for seed_i in SEEDS:
        rt = build_rt(seed_i, device)
        stacked = make_monitor_chunks(rt)
        for start_i in STARTS:
            for row in sel:
                seed, start, stream, arm, T16, snap = row
                if int(seed) != seed_i or int(start) != start_i:
                    continue
                key = (str(seed), str(start), str(stream), arm)
                if key in done:
                    print(f"SKIP done {key}")
                    continue
                old_means = {h: v for h, v in mon[key].items()}
                new_pts = cb.new_points_to_measure(T16)
                overlap_pts = [16] + ([T16] if T16 != 16 else [])
                plan_evals = len(new_pts) + len(overlap_pts)
                _check_caps(up_updates=T16 - 1,
                            up_fw=2 * (T16 - 1) + 128 * plan_evals,
                            up_bw=2 * (T16 - 1), up_mon=plan_evals)
                t0 = time.perf_counter()
                restore_h1(rt, Path(snap))
                # 快照身份核对
                step0 = int(next(iter(rt["optimizer"].state.values()))["step"])
                assert step0 == int(start) + 1, f"snapshot step {step0}"
                stream_full = make_stream(rt, int(stream), 2 * T16)
                tag = f"s{seed}_t{start}_st{stream}_{arm}"
                eval_at = {h: stacked for h in sorted(
                    set(new_pts) | set(overlap_pts))}
                # 先写 OLD_REUSED 行（0,4,16 倍数 <= T16）
                with dense_path.open("a") as f:
                    for h in sorted(old_means):
                        if h <= T16 and (h % 16 == 0 or h in (0, 1, 4)):
                            s_, c_, m_, _npz = old_means[h]
                            src = "OLD_REUSED"
                            f.write(",".join(str(x) for x in (
                                seed, start, stream, arm, h, repr(s_), c_,
                                repr(m_), src, "")) + "\n")
                status = "VERIFIED"
                blocked = False
                try:
                    recs = replay_steps(rt, seed, int(start), stream_full, 1,
                                        T16, eval_at=eval_at, tag=tag)
                except RuntimeError as exc:
                    _C["failures"] += 1
                    _log("main", f"{key} replay failure {exc!r}",
                         failures=1)
                    status = "REPLAY_FAILED"
                    blocked = True
                    recs = []
                rec_by_h = {r["h"]: r for r in recs}
                if not blocked:
                    with dense_path.open("a") as f, ident_path.open("a") as f2:
                        for h in sorted(eval_at):
                            r = rec_by_h.get(h)
                            if r is None:
                                continue
                            if h in overlap_pts:
                                old = old_means.get(h)
                                diff = abs(r["sum"] - old[0]) if old else None
                                bit = (diff == 0.0)
                                if not bit:
                                    status = "IDENTITY_MISMATCH"
                                    blocked = True
                                f2.write(",".join(str(x) for x in (
                                    seed, start, stream, arm, h,
                                    repr(old[0]) if old else "NA",
                                    repr(r["sum"]),
                                    repr(diff) if diff is not None else "NA",
                                    bit, status)) + "\n")
                                f.write(",".join(str(x) for x in (
                                    seed, start, stream, arm, h, repr(r["sum"]),
                                    r["count"], repr(r["mean"]),
                                    "REPLAY_OVERLAP",
                                    "bitwise" if bit else f"diff={diff!r}"))
                                    + "\n")
                            else:
                                f.write(",".join(str(x) for x in (
                                    seed, start, stream, arm, h,
                                    repr(r["sum"]), r["count"],
                                    repr(r["mean"]), "REPLAY_NEW", ""))
                                    + "\n")
                step_end = int(next(iter(rt["optimizer"].state.values()))["step"])
                with cells_path.open("a") as f:
                    f.write(json.dumps({
                        "seed": seed, "start": start, "stream": stream,
                        "arm": arm, "T16": T16, "status": status,
                        "completed": not blocked,
                        "final_step": step_end,
                        "expect_step": int(start) + T16,
                        "wall_s": time.perf_counter() - t0,
                        "ledger": dict(_C)}) + "\n")
                _log("cell_done", f"{key} status={status}")
                print(f"CELL_DONE {key} status={status}", flush=True)
        del rt, stacked
        torch.cuda.empty_cache()
    print("MAIN_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preflight", "smoke", "main", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "preflight":
        run_preflight()
    elif args.phase == "smoke":
        assert run_smoke(args.gpu)
    elif args.phase == "main":
        run_main(args.gpu)
    else:
        run_preflight()
        if run_smoke(args.gpu):
            run_main(args.gpu)


if __name__ == "__main__":
    main()
