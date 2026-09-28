#!/usr/bin/env python3
"""Phase A native calibration runner —— A0 一次性准备 + A1 native 目标校准。

只执行本文件与 FR_GTTR_PhaseA_Native_Target_Calibration_Claude_Code.md v1.0 /
FR_GTTR_Experiment_Roadmap_and_Next_Run_v1.md §3–§8 允许的 Phase A。
不进入 B/C/D/E；不改 optimizer；不训练 controller。

Schedule（用户明文冻结）：
    schedule_id = finite_cosine_then_hold_5e-5_v01
    label       = NATIVE_WITH_DECLARED_TAIL
    eta(gm) = old_get_lr(gm,256,5e-4)  (gm<=256) ; 5e-5 (gm>256)
只在新 PhaseA 入口包装（fr.phase_a_tail_schedule_v01）；旧 scheduler 与全部
历史输出保持不变。

子命令：
    preflight  现场核查 + 数据门 + schedule.csv + ema_timescales.csv（CPU）
    smoke      第一 cell 8 步 off + 8 步 on，共 16 真实 updates
    main       6 cells × 256 optimizer steps（smoke 通过后）
所有真实执行写入 resource_log.jsonl，caps 硬性检查。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
RUNS = OUT / "runs"
SNAPS = OUT / "snapshots"
DATA = OUT / "data"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
D0 = PROJECT_ROOT / "minimind" / "dataset" / "pretrain_t2t_mini.jsonl"
MONITOR_JSONL = DATA / "v_monitor_rows4096_8192.jsonl"

SCHEDULE_ID = "finite_cosine_then_hold_5e-5_v01"
SCHEDULE_LABEL = "NATIVE_WITH_DECLARED_TAIL"

SEEDS = (2030, 2031, 2032)
STARTS = (46, 82)
# stream IDs 按 (seed,start) 字典序分配；已核查既有注册无 97000..97005。
CELLS = [(s, t) for s in SEEDS for t in STARTS]
STREAM_IDS = {cell: 97000 + i for i, cell in enumerate(CELLS)}

H_MAIN = 256
MONITOR_POINTS = (0, 32, 64, 96, 128, 160, 192, 224, 256)
DIAG_POINTS = (1,) + tuple(range(16, 257, 16))
SNAP_POINTS = (128, 256)
EVAL_BATCH_ROWS = 32           # 纯评价设置；4096/32 = 128 forward batches/次
MONITOR_N_ROWS = 4096

# ---- 预算（协议 §8）-------------------------------------------------------
PLAN_MAIN_UPDATES = 6 * 256            # 1536
PLAN_SMOKE_UPDATES = 16
PLAN_UPDATES = PLAN_MAIN_UPDATES + PLAN_SMOKE_UPDATES   # 1552
CAP_UPDATES = 1664
CAP_MONITOR_FW = 7800
CAP_TOTAL_FW = 12000
CAP_WALL_S = 90 * 60

_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0

_C = {"optimizer_updates": 0, "train_fw": 0, "monitor_fw": 0,
      "monitor_evals": 0, "failures": 0, "snapshots": 0}
_T0 = None


def _log(phase, cell, note="", **kw):
    rec = {"ts": time.time(), "phase": phase,
           "cell": f"{cell[0]}/{cell[1]}" if cell else None,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "note": note, **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(phase, upcoming_updates=0, upcoming_monitor_fw=0,
                upcoming_train_fw=0):
    if _C["optimizer_updates"] + upcoming_updates > CAP_UPDATES:
        raise SystemExit("BUDGET_BLOCKED: updates cap 1664")
    if _C["monitor_fw"] + upcoming_monitor_fw > CAP_MONITOR_FW:
        raise SystemExit("BUDGET_BLOCKED: monitor forward cap 7800")
    if _C["train_fw"] + _C["monitor_fw"] + upcoming_monitor_fw + upcoming_train_fw \
            > CAP_TOTAL_FW:
        raise SystemExit("BUDGET_BLOCKED: total forward cap 12000")
    if time.perf_counter() - _T0 > CAP_WALL_S:
        raise SystemExit("BUDGET_BLOCKED: wall cap 5400s")


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ---------------------------------------------------------------------------
# preflight（CPU）
# ---------------------------------------------------------------------------

def run_preflight():
    import torch
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr

    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1) 数据身份
    n_lines = 0
    h = hashlib.sha256()
    with D0.open("rb") as f:
        for line in f:
            n_lines += 1
            h.update(line)
    d0_sha = h.hexdigest()
    subset = V07 / "e3c_subset.jsonl"
    assert n_lines == 1270238 and d0_sha.startswith("6dd6716c"), \
        f"D0 identity conflict: n={n_lines} sha={d0_sha[:16]}"
    rep["d0"] = {"path": str(D0), "lines": n_lines, "sha256": d0_sha}

    with D0.open("r", encoding="utf-8") as fi, MONITOR_JSONL.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if 4096 <= i < 8192:
                fo.write(line)
    rep["v_monitor"] = {"path": str(MONITOR_JSONL),
                        "rows": "[4096:8192)", "n": 4096,
                        "sha256": hashlib.sha256(MONITOR_JSONL.read_bytes()).hexdigest(),
                        "note": "D0 半开区间逐行复制，未改动原文件"}

    # 2) 分区/排除/重复（文本级 + 行 ID）
    part = json.loads((V07 / "partition_v07.json").read_text())
    tr, pr, ev = set(part["train_indices"]), set(part["probe_indices"]), set(part["eval_indices"])
    assert not (tr & pr) and not (tr & ev) and not (pr & ev)
    assert max(tr | pr | ev) < 4096
    assert set(range(0, 4096)) == tr | pr | ev
    rows_text = [json.loads(l)["text"] for l in D0.open(encoding="utf-8")]
    train_rows = [rows_text[i] for i in sorted(tr)]
    vm_rows = rows_text[4096:8192]
    vf_rows = rows_text[8192:12288]
    for name, seq in (("train", train_rows), ("v_monitor", vm_rows), ("v_final", vf_rows)):
        assert len(set(seq)) == len(seq), f"{name} exact-text duplicate"
    assert not (set(train_rows) & set(vm_rows))
    assert not (set(train_rows) & set(vf_rows))
    assert not (set(vm_rows) & set(vf_rows))
    rep["partition"] = {"train": len(tr), "probe": len(pr), "eval": len(ev),
                        "train_max_row": max(tr),
                        "monitor_row_range": [4096, 8192],
                        "final_row_range": [8192, 12288],
                        "train_excludes_monitor_final": True,
                        "exact_text_dup_within_or_across": 0,
                        "doc_id_field": None,
                        "document_independence": "DOCUMENT_INDEPENDENCE_UNVERIFIED"}

    # 3) bank 六 checkpoint 与真实 n0 / 单参数组语义
    bank = {}
    for seed, start in CELLS:
        p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
        assert p.exists(), f"missing {p}"
        st = torch.load(p, map_location="cpu", weights_only=False)
        md = st["metadata"]
        assert md["global_optimizer_step"] == start
        assert md["global_micro_step"] == 2 * start
        groups = md["optimizer_signature"]["param_groups"]
        assert len(groups) == 1 and list(groups[0]["betas"]) == [0.9, 0.999] \
            and groups[0]["weight_decay"] == 0.01 and groups[0]["eps"] == 1e-8
        k0 = next(iter(st["optimizer"]["state"]))
        assert float(st["optimizer"]["state"][k0]["step"]) == float(start)
        n_state = len(st["optimizer"]["state"])
        bank[f"{seed}/{start}"] = {
            "path": str(p), "n0": start, "global_micro": 2 * start,
            "model_tensors": len(st["model"]),
            "optimizer_state_entries": n_state,
            "moments_dtype": str(st["optimizer"]["state"][k0]["exp_avg"].dtype),
            "master_dtype": str(next(iter(st["model"].values())).dtype),
        }
    rep["bank"] = bank

    # 4) schedule：边界前后 + 各 cell LR 序列冻结表
    eta = make_phase_a_get_lr(_old_get_lr)
    assert abs(eta(256, 256, 5e-4) - 5e-5) < 1e-12, "tail boundary discontinuity"
    assert eta(257, 256, 5e-4) == 5e-5
    rows = ["seed,start,h,global_micro,lr_consumed"]
    for seed, start in CELLS:
        for h in (0, 1, 32, 64, 128, 192, 255, 256):
            gm = 2 * start + 2 * h
            rows.append(f"{seed},{start},{h},{gm},{eta(gm, 256, 5e-4):.10e}")
    (OUT / "schedule.csv").write_text("\n".join(rows) + "\n")
    rep["schedule"] = {"id": SCHEDULE_ID, "label": SCHEDULE_LABEL,
                       "formula": "old_get_lr(gm,256,5e-4) if gm<=256 else 5e-5",
                       "single_param_group": True, "group_tail_lr": 5e-5,
                       "warmup": "none-new", "cooldown": "none-new",
                       "moments_reset": False, "total_micro_rewrite": False,
                       "old_scheduler_untouched": True}

    # 5) stream IDs 冲突核查（97000..97005 未在任何已注册 stream_seed 使用）
    used = set()
    for p in PROJECT_ROOT.glob("outputs/**/*.json"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for tok in txt.split('"stream_seed":')[1:]:
            digits = tok.strip()[:20].split(",")[0].strip()
            if digits.isdigit():
                used.add(int(digits))
    conflicts = [sid for sid in (97000, 97005) if sid in used]
    assert not conflicts, f"stream id conflict {conflicts}"
    rep["streams"] = {f"{s}/{t}": STREAM_IDS[(s, t)] for s, t in CELLS}
    rep["stream_conflict_check"] = "no real-training registration at 97000..97005"

    # 6) EMA 尺度（纯滤波权重，不是响应寿命）
    ema = ["beta,half_steps,token_half,beta^64,beta^128,beta^256,"
           "w_old_corr_n0_46_H64,w_old_corr_n0_46_H128,w_old_corr_n0_46_H256,"
           "w_old_corr_n0_82_H64,w_old_corr_n0_82_H128,w_old_corr_n0_82_H256"]
    for beta in (0.9, 0.999):
        hh = math.log(0.5) / math.log(beta)
        row = [f"{beta}", f"{hh:.4f}", f"{hh * 1024:.1f}",
               f"{beta ** 64:.6e}", f"{beta ** 128:.6e}", f"{beta ** 256:.6e}"]
        for n0 in (46, 82):
            for H in (64, 128, 256):
                w = beta ** H * (1 - beta ** n0) / (1 - beta ** (n0 + H))
                row.append(f"{w:.6e}")
        ema.append(",".join(row))
    (OUT / "ema_timescales.csv").write_text("\n".join(ema) + "\n")
    rep["ema_note"] = "仅滤波权重；不是实际响应寿命或物理弛豫时间"

    # 7) 预算声明
    rep["budget"] = {"plan_main_updates": PLAN_MAIN_UPDATES,
                     "plan_smoke_updates": PLAN_SMOKE_UPDATES,
                     "plan_total": PLAN_UPDATES, "cap_updates": CAP_UPDATES,
                     "cap_monitor_fw": CAP_MONITOR_FW, "cap_total_fw": CAP_TOTAL_FW,
                     "cap_wall_s": CAP_WALL_S,
                     "new_gradient_calls": 0, "new_hvp_calls": 0,
                     "v_final_eval_calls": 0,
                     "train_backward": "present and logged"}

    (OUT / "config.json").write_text(json.dumps({
        "round": "fr_gttr_phase_a_native_calibration_v01/a1",
        "schedule_id": SCHEDULE_ID, "schedule_label": SCHEDULE_LABEL,
        "cells": [{"seed": s, "start": t, "stream_id": STREAM_IDS[(s, t)],
                   "n0": t} for s, t in CELLS],
        "monitor": {"rows": "[4096:8192)", "n_rows": 4096,
                    "eval_batch_rows": EVAL_BATCH_ROWS,
                    "metric": "sum(valid-target NLL)/count(valid targets) per row, "
                              "full-library aggregate",
                    "points": list(MONITOR_POINTS)},
        "train": {"optimizer": "torch.optim.AdamW", "betas": [_B1, _B2],
                  "eps": _EPS, "weight_decay": _WD, "clip_tau": _TAU,
                  "batch_size": 4, "accumulation_steps": 2,
                  "tokens_per_step_expected": 1024,
                  "dtype": "bfloat16 autocast, fp32 master weights/moments"},
        "budget": rep["budget"],
        "diagnostics": {"extra_gradient_calls": 0, "extra_hvp_calls": 0,
                        "layer_diag_points": list(DIAG_POINTS),
                        "innovation_label": "INNOVATION_PROXY"},
        "statuses_expected": {
            "ACTION_VALUE": "NOT_TESTED", "ADAPTIVE_GAIN": "NOT_TESTED",
            "FINAL_CONFIRMED_TARGET": "NOT_TESTED", "ENERGY_BENEFIT": "NOT_TESTED",
            "TRUE_TRACKING_ERROR": "NOT_MEASURED",
            "DRIFT_NOISE_SEPARATION": "NOT_MEASURED"},
    }, indent=1) + "\n")

    (OUT / "dataset_contract.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["bank"], indent=1))
    print("PREFLIGHT_OK", OUT)


# ---------------------------------------------------------------------------
# 运行时构造（复用 a0k 冻结路径）
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    from fr import a0k_closed_loop_runner_v01 as a0k
    rt = a0k.build_live_runtime(seed, device)
    assert len(rt["optimizer"].param_groups) == 1, "param group drift"
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    rt["eta"] = make_phase_a_get_lr(rt["get_lr"])
    return rt


def load_bank(rt, seed, start):
    from fr.full_state import load_full_training_state
    p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
    load_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    return p


def make_train_stream(rt, stream_id, n_micro):
    import b0_run_v01 as b0
    return {"stream_id": int(stream_id), **b0.make_stream(rt, int(stream_id), n_micro)}


def make_monitor_chunks(rt):
    """V_monitor 4096 行按固定 32 行 chunks 预堆叠（只做一次/cell）。"""
    import a0g_decision_utility_v01 as a0g
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(str(MONITOR_JSONL),
                         AutoTokenizer.from_pretrained(
                             str(PROJECT_ROOT / "minimind" / "model")),
                         max_length=int(rt["cfg"]["max_seq_len"]))
    chunks = [list(range(i, i + EVAL_BATCH_ROWS))
              for i in range(0, MONITOR_N_ROWS, EVAL_BATCH_ROWS)]
    assert len(chunks) == 128
    stacked = a0g._stack_all(ds, chunks, rt["device"])
    return ds, chunks, stacked


def monitor_full(rt, stacked, cell, h):
    """全库 4096 行 sum/count 评价。eval-mode + no_grad + RNG 包裹；
    不改训练状态/RNG/数据位置。返回聚合值并追加 per-row 记录。"""
    torch = rt["torch"]
    model = rt["model"]
    from fr.rng import capture_rng_state, restore_rng_state
    _check_caps("monitor", upcoming_monitor_fw=len(stacked))
    rng_snap = capture_rng_state()
    was_training = model.training
    total_sum = 0.0
    total_cnt = 0
    slots = 0
    nonpad = 0
    row_recs = []
    t0 = time.perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for ci, (x, y) in enumerate(stacked):
                with rt["autocast_ctx"]:
                    res = model(x)
                logits = res.logits.float()
                tgt = y[:, 1:]
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                per_sum = ce.sum(dim=1)
                per_cnt = (tgt != -100).sum(dim=1)
                per_nonpad = (y != -100).sum(dim=1)
                for r in range(tgt.shape[0]):
                    row_recs.append((4096 + ci * EVAL_BATCH_ROWS + r,
                                     float(per_sum[r]), int(per_cnt[r])))
                total_sum += float(per_sum.sum())
                total_cnt += int(per_cnt.sum())
                slots += int(x.numel())
                nonpad += int(per_nonpad.sum())
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_snap)
    _C["monitor_fw"] += len(stacked)
    _C["monitor_evals"] += 1
    wall = time.perf_counter() - t0
    _ensure_header("monitor_row_stats.csv")
    with (OUT / "monitor_row_stats.csv").open("a") as f:
        for rid, s, c in row_recs:
            f.write(f"{cell[0]},{cell[1]},{h},{rid},{s:.6f},{c}\n")
    _log("monitor", cell, f"h={h}", monitor_fw=len(stacked),
         sum=total_sum, count=total_cnt, wall=wall)
    return {"h": h, "sum": total_sum, "count": total_cnt,
            "mean": total_sum / max(total_cnt, 1),
            "input_slots": slots, "nonpad_tokens": nonpad,
            "wall_s": wall}


# ---------------------------------------------------------------------------
# 分层诊断（h ∈ {1,16,32,...,256}；零额外梯度）
# ---------------------------------------------------------------------------

def layer_diagnostics(rt, cell, h, theta_pre, m_pre, v_pre, c, theta_post,
                      true_step_pre, eta):
    """记录第 2.3 节统计。全部只读 + 克隆；tied 参数去重；零/小分母记 NA。"""
    torch = rt["torch"]
    rows = []
    opt_state = rt["optimizer"].state
    t_pre = true_step_pre
    t_post = true_step_pre + 1
    b1c, b2c = _B1, _B2
    seen = set()
    named = rt["named"]
    for name, p in named:
        if id(p) in seen:
            continue
        seen.add(id(p))
        W_pre = theta_pre[name]
        s = theta_post[name] - W_pre
        Wn = float(W_pre.norm())
        sn = float(s.norm())
        base = {"cell_seed": cell[0], "cell_start": cell[1], "h": h,
                "param": name, "kind": _kind_of(name, p),
                "rows": int(p.shape[0]), "cols": int(p.shape[1]) if p.dim() > 1 else 1}
        if Wn <= 0 or not math.isfinite(Wn):
            base.update(valid=False, note="zero_or_nonfinite_W_norm")
            rows.append(base)
            continue
        b_rad = float((W_pre * s).sum()) / (Wn ** 2)
        s_perp = s - b_rad * W_pre
        r_tangent = float(s_perp.norm()) / Wn
        r_total = sn / Wn
        ident_resid = abs(r_total ** 2 - (b_rad ** 2 + r_tangent ** 2))
        alpha_exact = float(
            (theta_post[name] / max(float(theta_post[name].norm()), 1e-30)
             - W_pre / Wn).norm())
        st = opt_state[p]
        m_post, v_post = st["exp_avg"], st["exp_avg_sq"]
        mhat_post = m_post / (1 - b1c ** t_post)
        vhat_post = v_post / (1 - b2c ** t_post)
        mhat_pre = m_pre[name] / (1 - b1c ** t_pre)
        vhat_pre = v_pre[name] / (1 - b2c ** t_pre)
        P_pre = 1.0 / (vhat_pre.sqrt() + _EPS)
        P_post = 1.0 / (vhat_post.sqrt() + _EPS)
        d_adaptive = -eta * mhat_post / (vhat_post.sqrt() + _EPS)
        s_decay = -eta * _WD * W_pre
        I_m = c[name] - m_pre[name]
        I_v = c[name] ** 2 - v_pre[name]
        align = -float((c[name] * s).sum()) / max(float(c[name].norm()) * sn, 1e-30)
        base.update(
            valid=True,
            W_norm=Wn, s_norm=sn,
            r_total=r_total, r_adaptive=float(d_adaptive.norm()) / Wn,
            b_rad=b_rad, r_tangent=r_tangent,
            radial_tangent_ident_resid=ident_resid, alpha_exact=alpha_exact,
            m_pre_norm=float(m_pre[name].norm()), v_pre_norm=float(v_pre[name].norm()),
            c_norm=float(c[name].norm()),
            m_post_norm=float(m_post.norm()), v_post_norm=float(v_post.norm()),
            mhat_post_norm=float(mhat_post.norm()), vhat_post_norm=float(vhat_post.norm()),
            P_pre_norm=float(P_pre.norm()), P_post_norm=float(P_post.norm()),
            P_rel_change=float((P_post - P_pre).norm()) / max(float(P_pre.norm()), 1e-30),
            d_adaptive_norm=float(d_adaptive.norm()), s_decay_norm=float(s_decay.norm()),
            I_m_norm=float(I_m.norm()), I_v_norm=float(I_v.norm()),
            I_label="INNOVATION_PROXY",
            train_batch_alignment=align,
            eta=eta, true_step_post=t_post)
        rows.append(base)
    return rows


def _kind_of(name, p):
    if p.dim() != 2:
        return "vector"
    if name == "model.embed_tokens.weight":
        return "embedding"
    if ".self_attn." in name:
        return "attn_matrix"
    if ".mlp." in name:
        return "mlp_matrix"
    return "matrix_other"


# ---------------------------------------------------------------------------
# 单条轨迹（smoke 与 main 共用内核）
# ---------------------------------------------------------------------------

def run_trajectory(rt, cell, stream, n_steps, monitor_points, diag_points,
                   monitor_stacked, phase, budget_tag,
                   snapshot_points=(), monitor_row_stats=True):
    """从已加载 bank 状态 + 已 seed stream 起跑 n_steps 个 optimizer steps。
    返回 dict: step_rows, monitor_records, snap_paths。"""
    torch = rt["torch"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    n0 = cell[1]
    gm0 = 2 * n0
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    named = rt["named"]

    step_rows = []
    monitor_records = []
    snap_paths = {}
    n_diag_rows = 0
    cum_input_tokens = 0
    cum_nonpad = 0
    cum_valid = 0

    def _snap(h):
        from fr.full_state import save_full_training_state
        path = SNAPS / f"s{cell[0]}_t{cell[1]}_h{h:03d}.pt"
        save_full_training_state(
            path, model, optimizer, scaler,
            metadata={"protocol": "FR_GTTR_PHASEA_A1_SNAPSHOT",
                      "schedule_id": SCHEDULE_ID,
                      "cell_seed": cell[0], "cell_start": cell[1],
                      "h": h, "true_step": n0 + h,
                      "global_micro": gm0 + 2 * h,
                      "stream_id": stream["stream_id"],
                      "micros_consumed": 2 * h,
                      "cum_valid_tokens": cum_valid})
        _C["snapshots"] += 1
        snap_paths[h] = str(path)
        _log(phase, cell, f"snapshot h={h}")

    for h in range(0, n_steps + 1):
        if h in monitor_points and monitor_stacked is not None and monitor_row_stats:
            rec = monitor_full(rt, monitor_stacked, cell, h)
            rec["cum_input_tokens"] = cum_input_tokens
            rec["cum_nonpad_tokens"] = cum_nonpad
            rec["cum_valid_tokens"] = cum_valid
            monitor_records.append(rec)
        if h == n_steps:
            break
        _check_caps(phase, upcoming_updates=1, upcoming_train_fw=accum)
        t_h0 = time.perf_counter()
        h_next = h + 1
        opt_step_row = {"h": h_next, "lr": None, "preclip_norm": None,
                        "clip_coeff": None, "clip_active": None,
                        "train_nll_sum": 0.0, "train_valid_cnt": 0,
                        "input_slots": 0, "nonpad_tokens": 0,
                        "update_norm": None, "finite": True,
                        "true_step": n0 + h_next, "wall_s": None}
        theta_pre = {n: p.detach().clone() for n, p in named}
        m_pre = {n: optimizer.state[p]["exp_avg"].detach().clone() for n, p in named}
        v_pre = {n: optimizer.state[p]["exp_avg_sq"].detach().clone() for n, p in named}
        for a in range(accum):
            micro_idx = 2 * h + a + 1
            lr = None
            from fr.native_schedule import set_native_micro_lr
            lr = set_native_micro_lr(optimizer, rt["eta"], gm0 + micro_idx,
                                     256, float(cfg["learning_rate"]))
            if a == accum - 1:
                opt_step_row["lr"] = float(lr)
            x, y = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
            with rt["autocast_ctx"]:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            _C["train_fw"] += 1
            valid_cnt = int((y[:, 1:] != -100).sum())
            opt_step_row["train_nll_sum"] += float(res.loss) * valid_cnt
            opt_step_row["train_valid_cnt"] += valid_cnt
            opt_step_row["input_slots"] += int(x.numel())
            opt_step_row["nonpad_tokens"] += int((y != -100).sum())
            if not math.isfinite(float(res.loss)):
                opt_step_row["finite"] = False
        cum_input_tokens += opt_step_row["input_slots"]
        cum_nonpad += opt_step_row["nonpad_tokens"]
        cum_valid += opt_step_row["train_valid_cnt"]
        scaler.unscale_(optimizer)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["grad_clip"])).detach().cpu())
        opt_step_row["preclip_norm"] = preclip
        opt_step_row["clip_coeff"] = (_TAU / preclip) if preclip > 0 else None
        opt_step_row["clip_active"] = bool(preclip > _TAU)
        if not math.isfinite(preclip):
            opt_step_row["finite"] = False
        c = {n: p.grad.detach().clone() for n, p in named}
        do_diag = h_next in diag_points
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _C["optimizer_updates"] += 1
        theta_post = {n: p.detach().clone() for n, p in named}
        upd_norm = float(torch.cat(
            [(theta_post[n] - theta_pre[n]).reshape(-1) for n, _ in named]
        ).norm())
        opt_step_row["update_norm"] = upd_norm
        if not math.isfinite(upd_norm):
            opt_step_row["finite"] = False
        opt_step_row["wall_s"] = time.perf_counter() - t_h0
        step_rows.append(opt_step_row)
        if do_diag:
            diag_rows = layer_diagnostics(
                rt, cell, h_next, {n: theta_pre[n] for n, _ in named},
                m_pre, v_pre, c,
                {n: theta_post[n] for n, _ in named}, n0 + h_next,
                float(opt_step_row["lr"]))
            _ensure_header("layer_diagnostics.csv")
            with (OUT / "layer_diagnostics.csv").open("a") as f:
                for r in diag_rows:
                    f.write(json.dumps(r) + "\n")
            n_diag_rows += len(diag_rows)
        _log(phase, cell, f"step h={h_next}", updates=1,
             finite=opt_step_row["finite"])
        if h_next in snapshot_points:
            _snap(h_next)
        if not opt_step_row["finite"]:
            _C["failures"] += 1
            _log(phase, cell, f"NON-FINITE at h={h_next}; trajectory aborted",
                 failures=1)
            break
    return {"step_rows": step_rows, "monitor_records": monitor_records,
            "snap_paths": snap_paths, "n_diag_rows": n_diag_rows,
            "cum": {"input_tokens": cum_input_tokens,
                    "nonpad_tokens": cum_nonpad,
                    "valid_tokens": cum_valid}}


_HEADERS = {
    "monitor_row_stats.csv":
        "seed,start,h,d0_row,nll_sum,valid_count\n",
    "step_metrics.csv":
        "seed,start,h,lr,preclip_norm,clip_coeff,clip_active,train_nll_sum,"
        "train_valid_cnt,input_slots,nonpad_tokens,update_norm,finite,"
        "true_step,wall_s\n",
}


def _ensure_header(name):
    p = OUT / name
    if name in _HEADERS and not p.exists():
        p.write_text(_HEADERS[name])
    elif name == "layer_diagnostics.csv" and not p.exists():
        p.write_text("")  # JSONL 行，键即 schema


def stack_batch_rt(rt, batch_indices):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], batch_indices, rt["device"])


def state_bitwise_sig(rt):
    torch = rt["torch"]
    sd = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    ost = rt["optimizer"].state_dict()
    sig = {"theta": sd, "opt_state": {
        str(k): {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                 for kk, vv in v.items()}
        for k, v in ost["state"].items()},
        "opt_step": {str(k): int(v["step"]) for k, v in ost["state"].items()},
        "param_groups": [{"lr": g["lr"], "betas": g["betas"],
                          "weight_decay": g["weight_decay"]} for g in ost["param_groups"]]}
    return sig


def sig_equal(a, b):
    import torch
    for n in a["theta"]:
        if not torch.equal(a["theta"][n], b["theta"][n]):
            return False, f"theta:{n}"
    for k in a["opt_state"]:
        for kk in a["opt_state"][k]:
            va, vb = a["opt_state"][k][kk], b["opt_state"][k][kk]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False, f"opt:{k}.{kk}"
            elif va != vb:
                return False, f"opt:{k}.{kk}"
    if a["opt_step"] != b["opt_step"]:
        return False, "opt_step"
    if a["param_groups"] != b["param_groups"]:
        return False, "param_groups"
    return True, ""


def rng_equal(a, b):
    import torch
    if a["python_random"] != b["python_random"]:
        return False, "python_random"
    na, nb = a.get("numpy_random"), b.get("numpy_random")
    if (na is None) != (nb is None):
        return False, "numpy_rng_presence"
    if na is not None and not (na[0] == nb[0] and (na[1] == nb[1]).all()):
        return False, "numpy_random"
    for k in ("torch_cpu",):
        if not torch.equal(a[k], b[k]):
            return False, k
    ca, cb = a.get("torch_cuda_all"), b.get("torch_cuda_all")
    if (ca is None) != (cb is None):
        return False, "cuda_rng_presence"
    if ca is not None:
        if len(ca) != len(cb):
            return False, "cuda_rng_count"
        for x, y in zip(ca, cb):
            if not torch.equal(x, y):
                return False, "torch_cuda"
    return True, ""


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _T0
    _T0 = time.perf_counter()
    RUNS.mkdir(parents=True, exist_ok=True)
    SNAPS.mkdir(parents=True, exist_ok=True)
    from fr.rng import capture_rng_state

    cell = CELLS[0]                    # (2030, 46)
    stream_id = STREAM_IDS[cell]
    rt = build_rt(cell[0], f"cuda:{gpu}")
    load_bank(rt, *cell)
    mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)

    checks = {"cell": f"{cell[0]}/{cell[1]}", "stream_id": stream_id,
              "updates_planned": 16, "identity_level": "bitwise(theta,m,v,step,lr) + RNG state equality"}

    # ---- arm OFF: 8 步，无 monitor、无诊断 --------------------------------
    from fr.stream_rng import seed_future_stream
    seed_future_stream(stream_id)
    stream = make_train_stream(rt, stream_id, 2 * 8)
    off = run_trajectory(rt, cell, stream, 8, monitor_points=(), diag_points=(),
                         monitor_stacked=None, phase="smoke_off", budget_tag="smoke")
    sig_off = state_bitwise_sig(rt)
    rng_off = capture_rng_state()
    cursor_off = 16

    # ---- 恢复初态，arm ON: h0/4/8 monitor + h=1 分层诊断 -------------------
    load_bank(rt, *cell)
    seed_future_stream(stream_id)
    stream_on = make_train_stream(rt, stream_id, 2 * 8)
    on = run_trajectory(rt, cell, stream_on, 8,
                        monitor_points=(0, 4, 8), diag_points=(1,),
                        monitor_stacked=mon_stacked, phase="smoke_on",
                        budget_tag="smoke")
    sig_on = state_bitwise_sig(rt)
    rng_on = capture_rng_state()
    cursor_on = 16

    same, why = sig_equal(sig_off, sig_on)
    checks["theta_m_v_step_lr_bitwise_equal"] = bool(same)
    checks["mismatch"] = why
    rsame, rwhy = rng_equal(rng_off, rng_on)
    checks["rng_state_equal"] = bool(rsame)
    checks["rng_mismatch"] = rwhy
    checks["data_cursor_equal"] = (cursor_off == cursor_on == 16)
    checks["monitor_records_on"] = [dict((k, m[k]) for k in
                                         ("h", "sum", "count", "mean", "wall_s"))
                                    for m in on["monitor_records"]]
    checks["diag_rows_on_h1"] = on["n_diag_rows"]
    checks["off_updates"] = len(off["step_rows"])
    checks["on_updates"] = len(on["step_rows"])
    checks["verdict"] = "PASS" if (same and rsame and cursor_off == cursor_on
                                   and len(off["step_rows"]) == 8
                                   and len(on["step_rows"]) == 8) else "FAIL"
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log("smoke_done", cell, checks["verdict"])
    print(json.dumps(checks, indent=1))

    # smoke 后 ETA
    wall = time.perf_counter() - _T0
    est = {
        "smoke_wall_s": wall,
        "per_step_s": wall / 16.0,
        "main_train_s": 1536 * (wall / 16.0) * 0.9,   # off 臂占比修正
        "monitor_per_eval_s": (sum(m["wall_s"] for m in on["monitor_records"])
                                / max(len(on["monitor_records"]), 1)),
        "main_monitor_s": 54 * (sum(m["wall_s"] for m in on["monitor_records"])
                                / max(len(on["monitor_records"]), 1)),
        "runtime_build_s_each": 30.0,
        "eta_main_s": (1536 * (wall / 16.0) * 0.9
                       + 54 * (sum(m["wall_s"] for m in on["monitor_records"])
                               / max(len(on["monitor_records"]), 1))
                       + 6 * 30.0),
        "cap_s": CAP_WALL_S}
    est["fits_cap"] = bool(est["eta_main_s"] + wall < CAP_WALL_S)
    (OUT / "eta_estimate.json").write_text(json.dumps(est, indent=1) + "\n")
    print("ETA", json.dumps(est, indent=1))
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_main(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    from fr.stream_rng import seed_future_stream
    t_main0 = time.perf_counter()
    results = {}
    for cell in CELLS:
        _check_caps("main", upcoming_updates=256, upcoming_train_fw=512,
                    upcoming_monitor_fw=9 * 128)
        seed, start = cell
        t_cell0 = time.perf_counter()
        rt = build_rt(seed, f"cuda:{gpu}")
        load_bank(rt, seed, start)
        mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)
        stream_id = STREAM_IDS[cell]
        seed_future_stream(stream_id)
        stream = make_train_stream(rt, stream_id, 2 * H_MAIN)
        out = run_trajectory(rt, cell, stream, H_MAIN,
                             monitor_points=MONITOR_POINTS,
                             diag_points=DIAG_POINTS,
                             monitor_stacked=mon_stacked, phase="main",
                             budget_tag="main",
                             snapshot_points=SNAP_POINTS)
        wall = time.perf_counter() - t_cell0
        results[f"{seed}/{start}"] = {
            "steps": len(out["step_rows"]),
            "completed": len(out["step_rows"]) == H_MAIN,
            "wall_s": wall,
            "monitors": out["monitor_records"],
            "cum": out["cum"],
            "snapshots": out["snap_paths"]}
        with (RUNS / "main_cells.jsonl").open("a") as f:
            f.write(json.dumps({"cell": f"{seed}/{start}",
                                "steps": len(out["step_rows"]),
                                "wall_s": wall,
                                "cum": out["cum"]}) + "\n")
        # 保存 step 级指标
        _ensure_header("step_metrics.csv")
        with (OUT / "step_metrics.csv").open("a") as f:
            for r in out["step_rows"]:
                f.write(f"{seed},{start}," + ",".join(
                    str(r[k]) for k in ("h", "lr", "preclip_norm", "clip_coeff",
                                        "clip_active", "train_nll_sum",
                                        "train_valid_cnt", "input_slots",
                                        "nonpad_tokens", "update_norm",
                                        "finite", "true_step", "wall_s")) + "\n")
        _log("cell_done", cell, f"steps={len(out['step_rows'])}")
        del rt, mon_stacked
    (RUNS / "main_summary.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n")
    print("MAIN_DONE", json.dumps({k: v["steps"] for k, v in results.items()}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preflight", "smoke", "main", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "preflight":
        run_preflight()
    elif args.phase == "smoke":
        run_smoke(args.gpu)
    elif args.phase == "main":
        run_main(args.gpu)
    else:
        run_preflight()
        if run_smoke(args.gpu):
            run_main(args.gpu)


if __name__ == "__main__":
    main()
