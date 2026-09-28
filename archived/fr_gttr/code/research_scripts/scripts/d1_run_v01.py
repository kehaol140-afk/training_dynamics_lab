#!/usr/bin/env python3
"""GTTR-D1 直接更新规则 pilot v1 runner —— preflight / smoke / main。

协议：FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md（D1 v1，2026-09-23）。
本轮只测一个候选读出（QH_RAW_A010 为同族读出对照）：

    r_A = P*mhat ; r_Q = (1-a)*r_A + a*(P*c) ; a=0.10
    r_N = ||r_A||_F * r_Q / ||r_Q||_F（逐唯一参数张量；zero/cancellation 回退）

五个固定臂：ADAMW_LR{050,100,200}（native，k=.5/1/2）、QH_RAW_A010、QH_NORM_A010。
真实 raw m/v、bias correction、P、clip、batch、schedule 全部沿用冻结 Phase A 配方；
候选只改变本步数据更新读出，不回写 moments，不新增全维持久状态。

子命令：
    preflight  只读核查 + config/manifest/implementation_checks（CPU）
    smoke      40 真实 updates + 6 全库 monitor（§8.2）
    main       15 条轨迹 × 256 步（§8.3，支持按 resource_log 续跑）
所有真实执行写 resource_log.jsonl；caps 为硬约束（§9）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_d1_direct_readout_v01"
RUNS = OUT / "runs"
SNAPS = OUT / "snapshots"
MON_ROWS = OUT / "monitor_rows"
FAILS = OUT / "failures"
A1 = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
D0 = PROJECT_ROOT / "minimind" / "dataset" / "pretrain_t2t_mini.jsonl"
MONITOR_JSONL = A1 / "data" / "v_monitor_rows4096_8192.jsonl"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md"

SCHEDULE_ID = "finite_cosine_then_hold_5e-5_v01"
SCHEDULE_LABEL = "NATIVE_WITH_DECLARED_TAIL"

SEEDS = (2030, 2031, 2032)
START = 46
CELLS = [(s, START) for s in SEEDS]

# 五个固定臂（协议 §2）。kind=native 走原 scaler.step 路径；kind=qh 走 QH 读出。
ARMS = {
    "ADAMW_LR050": {"kind": "native", "k": 0.5, "alpha": 0.0, "mode": None},
    "ADAMW_LR100": {"kind": "native", "k": 1.0, "alpha": 0.0, "mode": None},
    "ADAMW_LR200": {"kind": "native", "k": 2.0, "alpha": 0.0, "mode": None},
    "QH_RAW_A010": {"kind": "qh", "k": 1.0, "alpha": 0.10, "mode": "RAW"},
    "QH_NORM_A010": {"kind": "qh", "k": 1.0, "alpha": 0.10, "mode": "NORM"},
}
# 执行顺序（协议 §3.1，纯运维非择优）
ORDER = {
    2030: ["ADAMW_LR100", "QH_NORM_A010", "ADAMW_LR050", "QH_RAW_A010", "ADAMW_LR200"],
    2031: ["ADAMW_LR050", "QH_RAW_A010", "ADAMW_LR200", "ADAMW_LR100", "QH_NORM_A010"],
    2032: ["ADAMW_LR200", "ADAMW_LR100", "QH_NORM_A010", "ADAMW_LR050", "QH_RAW_A010"],
}
PROPOSED_STREAMS = {2030: 99000, 2031: 99001, 2032: 99002}
SMOKE_STREAM_PROPOSED = 99100

H_MAIN = 256
MONITOR_POINTS = tuple(range(0, H_MAIN + 1, 16))     # 17 点（协议 §4.1）
DIAG_POINTS = (1, 16, 64, 128, 256)                   # 协议 §6
SNAP_POINTS = (128, 256)
EVAL_BATCH_ROWS = 32
MONITOR_N_ROWS = 4096

# ---- 预算（协议 §9）---------------------------------------------------------
PLAN_MAIN_UPDATES = 3 * 5 * 256          # 3840
PLAN_SMOKE_UPDATES = 40
PLAN_UPDATES = PLAN_MAIN_UPDATES + PLAN_SMOKE_UPDATES   # 3880
CAP_UPDATES = 4096
CAP_TRAIN_BW = 8192
CAP_MONITOR_EVALS = 270
CAP_MONITOR_FW = 34560
CAP_TOTAL_FW = 45000
# plan 参照（报告用）：train_fw=7760, monitor_evals=249, monitor_fw=31872, total=39632

_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0

_C = {"optimizer_updates": 0, "train_fw": 0, "train_bw": 0, "monitor_fw": 0,
      "monitor_evals": 0, "failures": 0, "numerical_failures": 0,
      "snapshots": 0}
_T0 = None


def _replay_ledger():
    """跨进程续跑：resource_log.jsonl 最后一行即累计计数（§9 所有失败都计入）。"""
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


def _log(phase, cell, note="", **kw):
    rec = {"ts": time.time(), "phase": phase,
           "cell": f"{cell[0]}/{cell[1]}" if cell else None,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "note": note, **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(phase, upcoming_updates=0, upcoming_monitor_fw=0,
                upcoming_train_fw=0, upcoming_bw=0):
    if _C["optimizer_updates"] + upcoming_updates > CAP_UPDATES:
        raise SystemExit("BUDGET_BLOCKED: updates cap 4096")
    if _C["train_bw"] + upcoming_bw > CAP_TRAIN_BW:
        raise SystemExit("BUDGET_BLOCKED: backward microcall cap 8192")
    if _C["monitor_evals"] + (1 if upcoming_monitor_fw else 0) > CAP_MONITOR_EVALS:
        raise SystemExit("BUDGET_BLOCKED: monitor eval cap 270")
    if _C["monitor_fw"] + upcoming_monitor_fw > CAP_MONITOR_FW:
        raise SystemExit("BUDGET_BLOCKED: monitor forward cap 34560")
    if _C["train_fw"] + _C["monitor_fw"] + upcoming_monitor_fw + upcoming_train_fw \
            > CAP_TOTAL_FW:
        raise SystemExit("BUDGET_BLOCKED: total forward cap 45000")


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ---------------------------------------------------------------------------
# preflight（CPU，只读）
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def run_preflight():
    import torch
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr

    OUT.mkdir(parents=True, exist_ok=True)
    for d in (RUNS, SNAPS, MON_ROWS, FAILS):
        d.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1) D0 数据身份 + V_monitor 存在性（不重写 v_monitor，沿用 Phase A 原件）
    n_lines = 0
    h = hashlib.sha256()
    with D0.open("rb") as f:
        for line in f:
            n_lines += 1
            h.update(line)
    d0_sha = h.hexdigest()
    assert n_lines == 1270238 and d0_sha.startswith("6dd6716c"), \
        f"D0 identity conflict: n={n_lines} sha={d0_sha[:16]}"
    assert MONITOR_JSONL.exists(), f"missing {MONITOR_JSONL}"
    mon_lines = sum(1 for _ in MONITOR_JSONL.open("rb"))
    assert mon_lines == MONITOR_N_ROWS, f"v_monitor rows {mon_lines}"
    rep["d0"] = {"path": str(D0), "lines": n_lines, "sha256": d0_sha}
    rep["v_monitor"] = {"path": str(MONITOR_JSONL), "rows": "[4096:8192)",
                        "n": MONITOR_N_ROWS, "sha256": _sha256(MONITOR_JSONL),
                        "note": "Phase A 原件，未改动"}

    # 2) bank 三 checkpoint：真实 n0 / 单参数组语义
    bank = {}
    for seed, start in CELLS:
        p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
        assert p.exists(), f"missing {p}"
        st = torch.load(p, map_location="cpu", weights_only=False)
        md = st["metadata"]
        assert md["global_optimizer_step"] == start, f"n0 mismatch {seed}"
        assert md["global_micro_step"] == 2 * start
        groups = md["optimizer_signature"]["param_groups"]
        assert len(groups) == 1 and list(groups[0]["betas"]) == [0.9, 0.999] \
            and groups[0]["weight_decay"] == 0.01 and groups[0]["eps"] == 1e-8
        k0 = next(iter(st["optimizer"]["state"]))
        assert float(st["optimizer"]["state"][k0]["step"]) == float(start)
        bank[f"{seed}/{start}"] = {
            "path": str(p), "n0": start, "global_micro": 2 * start,
            "model_tensors": len(st["model"]),
            "optimizer_state_entries": len(st["optimizer"]["state"]),
            "moments_dtype": str(st["optimizer"]["state"][k0]["exp_avg"].dtype),
            "master_dtype": str(next(iter(st["model"].values())).dtype),
            "sha256": _sha256(p),
        }
    rep["bank"] = bank

    # 3) 配方核对（读取最近 Phase A config；冲突即报，不自行改）
    a1cfg = json.loads((A1 / "config.json").read_text())
    a1train = a1cfg["train"]
    recipe_expect = {"optimizer": "torch.optim.AdamW", "betas": [0.9, 0.999],
                     "eps": 1e-8, "weight_decay": 0.01, "clip_tau": 1.0,
                     "batch_size": 4, "accumulation_steps": 2}
    conflicts = {k: {"a1": a1train.get(k), "expected": v}
                 for k, v in recipe_expect.items() if a1train.get(k) != v}
    assert not conflicts, f"recipe conflict vs Phase A: {conflicts}"
    assert a1cfg["schedule_id"] == SCHEDULE_ID
    rep["recipe"] = {"from": str(A1 / "config.json"), "conflicts": conflicts,
                     "autocast": "bfloat16", "grad_scaler": "disabled",
                     "note": "实配方以 v07 runtime 为准，与 Phase A 一致"}

    # 4) ell：从 a1 全精度校准文件读取（不重新估计）
    ell = None
    with (A1 / "target_calibration.csv").open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            row = line.strip().split(",")
            if row[0] == str(START):
                ell = float(row[1])
                break
    assert ell is not None, "start46 ell not found"
    assert abs(ell - 7.020043501537879) < 1e-12, f"ell display mismatch {ell}"
    rep["ell_star"] = {"value": ell, "source": str(A1 / "target_calibration.csv"),
                       "start": START, "reestimated": False}

    # 5) schedule 边界
    eta = make_phase_a_get_lr(_old_get_lr)
    assert abs(eta(256, 256, 5e-4) - 5e-5) < 1e-12
    assert eta(257, 256, 5e-4) == 5e-5
    rep["schedule"] = {"id": SCHEDULE_ID, "label": SCHEDULE_LABEL,
                       "formula": "eta_arm(gm)=k_arm*(old_get_lr(gm,256,5e-4) if gm<=256 "
                                  "else 5e-5)",
                       "k_by_arm": {a: ARMS[a]["k"] for a in ARMS},
                       "tail_lr_by_arm": {a: 5e-5 * ARMS[a]["k"] for a in ARMS},
                       "warmup": "none-new", "cooldown": "none-new",
                       "moments_reset": False, "single_param_group": True}

    # 6) stream 冲突核查：若提议 ID 已有任何注册记录，整块 +1000（纯元数据）
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
    streams = dict(PROPOSED_STREAMS)
    smoke_stream = SMOKE_STREAM_PROPOSED
    while any(s in used for s in streams.values()) or smoke_stream in used:
        streams = {s: v + 1000 for s, v in streams.items()}
        smoke_stream += 1000
    rep["streams"] = {str(s): streams[s] for s in SEEDS}
    rep["smoke_stream"] = smoke_stream
    rep["stream_conflict_check"] = {
        "proposed_main": list(PROPOSED_STREAMS.values()),
        "proposed_smoke": SMOKE_STREAM_PROPOSED,
        "final_main": list(streams.values()), "final_smoke": smoke_stream,
        "n_registered_scanned": len(used),
        "shifted": streams != PROPOSED_STREAMS or smoke_stream != SMOKE_STREAM_PROPOSED,
        "metadata_only_decision": True}

    # 7) 八项 CPU 测试 -> implementation_checks.json
    import fr.d1_direct_readout_v01 as d1m
    test_mod = "tests.test_d1_direct_readout_v01"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "unittest", test_mod, "-v"],
                       cwd=PROJECT_ROOT / "research", capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin",
                            "PYTHONPATH": "src",
                            "HOME": "/home/b205"})
    out_txt = r.stdout + r.stderr
    passed = out_txt.strip().splitlines()[-1].strip() == "OK" if r.returncode == 0 else False
    impl = {
        "unittest_module": test_mod,
        "returncode": r.returncode,
        "tail_line": out_txt.strip().splitlines()[-1].strip() if out_txt.strip() else "",
        "all_pass": bool(passed),
        "wall_s": round(time.time() - t0, 3),
        "t1_to_t8": {
            "T1_alpha0_native_routing_and_bitwise": "covered_by_suite",
            "T2_moments_not_written_back": "covered_by_suite",
            "T3_raw_qh_independent_formula": "covered_by_suite",
            "T4_norm_matching_zero_cancel_shared": "covered_by_suite",
            "T5_alignment_not_descent_counterexample": "covered_by_suite",
            "T6_schedule_prefix_tail_k_scaling": "covered_by_suite",
            "T7_sumcount_detection_h0_share": "covered_by_suite",
            "T8_mutants_must_fail": "covered_by_suite"},
        "module": {"path": str(Path(d1m.__file__).resolve()),
                   "sha256": _sha256(Path(d1m.__file__).resolve())},
        "norm_definitions": "所有 ||.||_F = sqrt(sum of squares)，无额外 norm epsilon；"
                            "范数残差 = | ||r_N||_F - ||r_A||_F |",
    }
    (OUT / "implementation_checks.json").write_text(json.dumps(impl, indent=1) + "\n")
    rep["implementation"] = {"all_pass": bool(passed), "returncode": r.returncode}
    assert passed, f"CPU unittest not clean:\n{out_txt[-2000:]}"

    # 8) 预算与代码版本
    rep["budget"] = {
        "plan_main_updates": PLAN_MAIN_UPDATES, "plan_smoke_updates": PLAN_SMOKE_UPDATES,
        "plan_total_updates": PLAN_UPDATES, "cap_updates": CAP_UPDATES,
        "cap_train_bw": CAP_TRAIN_BW, "cap_monitor_evals": CAP_MONITOR_EVALS,
        "cap_monitor_fw": CAP_MONITOR_FW, "cap_total_fw": CAP_TOTAL_FW,
        "plan_train_fw": 7760, "plan_train_bw": 7760,
        "plan_monitor_evals": 249, "plan_monitor_fw": 31872,
        "plan_total_fw": 39632,
        "extra_gradient_calls": 0, "extra_hvp_calls": 0, "extra_fd_calls": 0,
        "candidate_rollout_for_selection": 0, "v_final_eval_calls": 0,
        "wall_clock": "OBSERVATIONAL_SHARED_GPU only"}
    rep["code"] = {"git_head": _git_head(),
                   "protocol_sha256": _sha256(PROTOCOL) if PROTOCOL.exists() else "MISSING",
                   "runner": str(Path(__file__).resolve()),
                   "runner_sha256": _sha256(Path(__file__).resolve())}

    (OUT / "config.json").write_text(json.dumps({
        "round": "fr_gttr_d1_direct_readout_v01",
        "protocol": "FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md",
        "arms": ARMS, "order_by_seed": ORDER,
        "cells": [{"seed": s, "start": START, "stream_seed": streams[s]} for s in SEEDS],
        "smoke_stream_seed": smoke_stream,
        "schedule_id": SCHEDULE_ID,
        "alpha": 0.10, "alpha_frozen": True,
        "lr_grid": {"ADAMW_LR050": 0.5, "ADAMW_LR100": 1.0, "ADAMW_LR200": 2.0},
        "lr_grid_note": "粗网格，不冒充充分调优；边界最优标 BASELINE_GRID_BOUNDARY",
        "candidate_vs_raw": "候选与 QH_RAW 同 schedule/eta/decay，只比较范数匹配",
        "monitor": {"points": list(MONITOR_POINTS), "n_points": len(MONITOR_POINTS),
                    "h0_shared_per_seed": True,
                    "rows": "[4096:8192)", "n_rows": MONITOR_N_ROWS,
                    "eval_batch_rows": EVAL_BATCH_ROWS,
                    "metric": "sum(valid-target NLL)/count(valid targets), full library",
                    "new_16_step_grid_note": "不同于旧 32 步检测，不能声称提速"},
        "detection": {"rule": "two consecutive monitor means <= ell_star",
                      "ell_star": ell,
                      "h_detect": "second point of first confirmed pair",
                      "not_completed_intervals_preserved": True},
        "diagnostics": {"points": list(DIAG_POINTS), "extra_gradient_calls": 0,
                        "extra_hvp_calls": 0,
                        "rA_on_norm_traj": "本状态 native-formula reference，"
                                           "不是另一条真实 native 轨迹的 r_A"},
        "train": {"optimizer": "torch.optim.AdamW", "betas": [_B1, _B2], "eps": _EPS,
                  "weight_decay": _WD, "clip_tau": _TAU, "batch_size": 4,
                  "accumulation_steps": 2, "tokens_per_step_expected": 1024,
                  "dtype": "bfloat16 autocast, fp32 master weights/moments",
                  "grad_scaler": "disabled",
                  "four_counters": ["input_slots", "nonpad_tokens",
                                    "valid_loss_tokens", "optimizer_updates"]},
        "budget": rep["budget"],
        "expected_statuses": {"IMPLEMENTATION": "TBD", "COVERAGE": "TBD",
                              "FINAL_BUDGET_NLL_EFFECT": "TBD",
                              "DETECTED_TARGET_TOKEN_EFFECT": "TBD",
                              "NORM_MATCHING_ABLATION": "TBD",
                              "FIXED_LR_COMPARISON": "TBD"},
    }, indent=1) + "\n")

    (OUT / "manifest.json").write_text(json.dumps({
        "state_manifest": bank,
        "data_manifest": {"d0": rep["d0"], "v_monitor": rep["v_monitor"],
                          "train_support": "冻结 v07 train_indices，不扩大",
                          "v_final": "封存，零模型调用"},
        "arm_manifest": {a: {**ARMS[a],
                             "role": {"ADAMW_LR050": "较小固定LR对照",
                                      "ADAMW_LR100": "native 基线",
                                      "ADAMW_LR200": "较大固定LR对照",
                                      "QH_RAW_A010": "QHAdamW 族读出对照(nu1=.9,nu2=1)",
                                      "QH_NORM_A010": "唯一主候选"}[a]}
                         for a in ARMS},
        "prior_art": {"QH": "Ma & Yarats ICLR 2019 (arXiv:1810.06801); "
                             "github.com/facebookresearch/qhoptim",
                      "grafting": "Agarwal et al. 2020 (arXiv:2002.11803)",
                      "novelty_claim": "无"},
        "stream_manifest": rep["streams"], "smoke_stream": smoke_stream,
        "schedule_manifest": rep["schedule"], "ell_manifest": rep["ell_star"],
        "execution_order": {str(s): ORDER[s] for s in SEEDS},
        "budget_manifest": rep["budget"], "code_manifest": rep["code"],
    }, indent=1) + "\n")

    print(json.dumps({"bank": rep["bank"], "ell": ell, "streams": rep["streams"],
                      "smoke_stream": smoke_stream, "unittest": impl["all_pass"]},
                     indent=1))
    print("PREFLIGHT_OK", OUT)


# ---------------------------------------------------------------------------
# 运行时构造（复用冻结路径，与 Phase A runner 相同）
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


def make_train_stream(rt, stream_seed, n_micro):
    import b0_run_v01 as b0
    return {"stream_seed": int(stream_seed),
            **b0.make_stream(rt, int(stream_seed), n_micro)}


def make_monitor_chunks(rt):
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


def stack_batch_rt(rt, batch_indices):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], batch_indices, rt["device"])


def monitor_full(rt, stacked, tag, h, npz_name=None):
    """全库 4096 行 sum/count 评价。eval-mode + no_grad + RNG 包裹；
    不改训练状态/RNG/数据位置。逐行结果写 NPZ。"""
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
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                per_sum = ce.sum(dim=1)
                per_cnt = (tgt != -100).sum(dim=1)
                per_nonpad = (y != -100).sum(dim=1)
                for r in range(tgt.shape[0]):
                    row_ids.append(4096 + ci * EVAL_BATCH_ROWS + r)
                    row_sums.append(float(per_sum[r]))
                    row_cnts.append(int(per_cnt[r]))
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
    name = npz_name or f"{tag}_h{h:03d}.npz"
    import numpy as np
    np.savez_compressed(MON_ROWS / name,
                        row_id=np.asarray(row_ids, dtype=np.int32),
                        nll_sum=np.asarray(row_sums, dtype=np.float64),
                        valid_count=np.asarray(row_cnts, dtype=np.int32))
    rec = {"h": h, "sum": total_sum, "count": total_cnt,
           "mean": total_sum / max(total_cnt, 1),
           "input_slots": slots, "nonpad_tokens": nonpad,
           "wall_s": wall, "npz": name, "billed_eval": True}
    _log("monitor", None, f"{tag} h={h}", monitor_fw=len(stacked),
         sum=total_sum, count=total_cnt, wall=wall)
    return rec


# ---------------------------------------------------------------------------
# 读出同状态参考诊断（h ∈ DIAG_POINTS；零额外梯度；只读本步已有量）
# ---------------------------------------------------------------------------

def readout_diag_rows(rt, seed, arm_name, arm, h, theta_pre, m_pre, v_pre, c,
                      theta_post, t_post, eta):
    from fr import d1_direct_readout_v01 as d1m
    torch = rt["torch"]
    rows = []
    seen = set()
    for name, p in rt["named"]:
        if id(p) in seen:
            continue
        seen.add(id(p))
        cc = c.get(name)
        if cc is None:
            continue
        rA, mhat, vhat, P = d1m.native_reference(
            cc, m_pre[name], v_pre[name], t_post, _B1, _B2, _EPS)
        Pc = P * cc
        rQ = d1m.r_q(rA, Pc, d1m.ALPHA)
        rN, fb = d1m.norm_match(rA, rQ, Pc, d1m.ALPHA)
        A = float(rA.norm())
        Q = float(rQ.norm())
        N = float(rN.norm())
        dot_an = float((rA * rN).sum())
        cos_an = dot_an / (A * N) if A > 0 and N > 0 else float("nan")
        applied = {"native": "rA", "RAW": "rQ", "NORM": "rN"}[
            "native" if arm["kind"] == "native" else arm["mode"]]
        r_applied = {"rA": rA, "rQ": rQ, "rN": rN}[applied]
        s = theta_post[name] - theta_pre[name]
        numel = int(p.numel())
        rows.append({
            "seed": seed, "arm": arm_name, "h": h, "param": name,
            "kind": _kind_of(name, p),
            "rA_norm": A, "rQ_norm": Q, "rN_norm": N,
            "norm_resid": abs(N - A),
            "dot_rA_rN": dot_an, "cos_rA_rN": cos_an,
            "c_dot_rA": float((cc * rA).sum()), "c_dot_rQ": float((cc * rQ).sum()),
            "c_dot_rN": float((cc * rN).sum()),
            "c_dot_diff": float((cc * rN).sum()) - float((cc * rA).sum()),
            "theta_norm": float(theta_pre[name].norm()),
            "adaptive_step_norm": eta * float(r_applied.norm()),
            "full_step_norm": float(s.norm()),
            "applied": applied,
            "fallback": fb if arm["kind"] == "qh" and arm["mode"] == "NORM" else "NA",
            "temp_bytes_fp32": 7 * numel * 4,
            "numel": numel, "eta": eta, "true_step": t_post})
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
# 单条轨迹（smoke 与 main 共用内核；Phase A run_trajectory 的最小注入版）
# ---------------------------------------------------------------------------

_HEADERS = {
    "step_metrics.csv":
        "seed,arm,h,lr,preclip_norm,clip_coeff,clip_active,train_nll_sum,"
        "train_valid_cnt,input_slots,nonpad_tokens,update_norm,finite,"
        "true_step,wall_s\n",
    "monitor_metrics.csv":
        "seed,arm,h,sum,count,mean,input_slots,nonpad_tokens,cum_input_tokens,"
        "cum_nonpad_tokens,cum_valid_tokens,wall_s,npz,shared_h0,billed_eval\n",
    "readout_diagnostics.csv":
        "seed,arm,h,param,kind,rA_norm,rQ_norm,rN_norm,norm_resid,dot_rA_rN,"
        "cos_rA_rN,c_dot_rA,c_dot_rQ,c_dot_rN,c_dot_diff,theta_norm,"
        "adaptive_step_norm,full_step_norm,applied,fallback,temp_bytes_fp32,"
        "numel,eta,true_step\n",
}


def _ensure_header(name):
    p = OUT / name
    if name in _HEADERS and not p.exists():
        p.write_text(_HEADERS[name])
    return p


def _save_failure_evidence(rt, seed, arm_name, h, theta_pre, m_pre, v_pre, c,
                           exc):
    """数值失败：保存失败前完整状态 + 证据，终止该轨迹（不换 seed 不重跑）。"""
    from fr.full_state import save_full_training_state
    FAILS.mkdir(parents=True, exist_ok=True)
    tag = f"s{seed}_{arm_name}_h{h:03d}"
    state_path = FAILS / f"{tag}_state.pt"
    save_full_training_state(
        state_path, rt["model"], rt["optimizer"], rt["scaler"],
        metadata={"protocol": "FR_GTTR_D1_FAILURE", "seed": seed,
                  "arm": arm_name, "h": h,
                  "exception": repr(exc), "when": time.time()})
    torch = rt["torch"]
    torch.save({
        "theta_pre": theta_pre, "m_pre": m_pre, "v_pre": v_pre,
        "m_pre_captured": m_pre is not None,
        "c": c,
        "grads": {i: p.grad.detach().cpu().clone() if p.grad is not None else None
                  for i, p in enumerate(rt["model"].parameters())},
        "exception": repr(exc),
    }, FAILS / f"{tag}_evidence.pt")
    return str(state_path)


def run_trajectory(rt, seed, arm_name, arm, stream, n_steps, monitor_points,
                   diag_points, monitor_stacked, phase, k_lr,
                   snapshot_points=(), shared_h0=None, tag=None):
    """从已加载 bank 状态 + 已 seed stream 起跑 n_steps 个 optimizer steps。

    与 Phase A runner 同算术；唯一注入：arm.kind==\"qh\" 时 scaler.step 位置改调
    qh_adamw_step（GradScaler disabled，二者等价位置）。返回 dict。"""
    torch = rt["torch"]
    from fr import d1_direct_readout_v01 as d1m
    from fr.native_schedule import set_native_micro_lr

    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    n0 = START
    gm0 = 2 * n0
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    named = rt["named"]
    tag = tag or f"s{seed}_{arm_name}"

    step_rows = []
    monitor_records = []
    diag_rows = []
    snap_paths = {}
    cum_input_tokens = 0
    cum_nonpad = 0
    cum_valid = 0
    numerical_failure = None

    def _snap(h, cum_valid):
        from fr.full_state import save_full_training_state
        path = SNAPS / f"{tag}_h{h:03d}.pt"
        save_full_training_state(
            path, model, optimizer, scaler,
            metadata={"protocol": "FR_GTTR_D1_SNAPSHOT",
                      "schedule_id": SCHEDULE_ID,
                      "seed": seed, "start": n0, "arm": arm_name, "h": h,
                      "true_step": n0 + h, "global_micro": gm0 + 2 * h,
                      "stream_seed": stream["stream_seed"],
                      "micros_consumed": 2 * h, "cum_valid_tokens": cum_valid})
        _C["snapshots"] += 1
        snap_paths[h] = str(path)
        _log(phase, (seed, n0), f"snapshot {tag} h={h}")

    aborted = False
    for h in range(0, n_steps + 1):
        if h in monitor_points and monitor_stacked is not None:
            if h == 0 and shared_h0 is not None:
                rec = dict(shared_h0)
                rec["shared_h0"] = True
                rec["billed_eval"] = False
            else:
                rec = monitor_full(rt, monitor_stacked, tag, h)
                rec["shared_h0"] = False
                rec["billed_eval"] = True
            rec["cum_input_tokens"] = cum_input_tokens
            rec["cum_nonpad_tokens"] = cum_nonpad
            rec["cum_valid_tokens"] = cum_valid
            monitor_records.append(rec)
        if h == n_steps:
            break
        _check_caps(phase, upcoming_updates=1, upcoming_train_fw=accum,
                    upcoming_bw=accum)
        t_h0 = time.perf_counter()
        h_next = h + 1
        row = {"seed": seed, "arm": arm_name, "h": h_next, "lr": None,
               "preclip_norm": None, "clip_coeff": None, "clip_active": None,
               "train_nll_sum": 0.0, "train_valid_cnt": 0, "input_slots": 0,
               "nonpad_tokens": 0, "update_norm": None, "finite": True,
               "true_step": n0 + h_next, "wall_s": None}
        theta_pre = {n: p.detach().clone() for n, p in named}
        do_diag = h_next in diag_points
        m_pre = v_pre = None
        c = None
        if do_diag:
            m_pre = {n: optimizer.state[p]["exp_avg"].detach().clone()
                     for n, p in named}
            v_pre = {n: optimizer.state[p]["exp_avg_sq"].detach().clone()
                     for n, p in named}
        try:
            for a in range(accum):
                micro_idx = 2 * h + a + 1
                lr = set_native_micro_lr(optimizer, rt["eta"], gm0 + micro_idx,
                                         256, float(cfg["learning_rate"]))
                if k_lr != 1.0:
                    optimizer.param_groups[0]["lr"] = lr * k_lr
                if a == accum - 1:
                    row["lr"] = float(optimizer.param_groups[0]["lr"])
                x, y = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
                with rt["autocast_ctx"]:
                    res = model(x, labels=y)
                    loss = (res.loss + res.aux_loss) / accum
                scaler.scale(loss).backward()
                _C["train_fw"] += 1
                _C["train_bw"] += 1
                valid_cnt = int((y[:, 1:] != -100).sum())
                row["train_nll_sum"] += float(res.loss) * valid_cnt
                row["train_valid_cnt"] += valid_cnt
                row["input_slots"] += int(x.numel())
                row["nonpad_tokens"] += int((y != -100).sum())
                if not math.isfinite(float(res.loss)):
                    row["finite"] = False
            cum_input_tokens += row["input_slots"]
            cum_nonpad += row["nonpad_tokens"]
            cum_valid += row["train_valid_cnt"]
            scaler.unscale_(optimizer)
            preclip = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])).detach().cpu())
            row["preclip_norm"] = preclip
            row["clip_coeff"] = (_TAU / preclip) if preclip > 0 else None
            row["clip_active"] = bool(preclip > _TAU)
            if not math.isfinite(preclip):
                row["finite"] = False
            c = {n: p.grad.detach().clone() for n, p in named}
            # ---- 更新步：唯一注入点 --------------------------------------
            if arm["kind"] == "native":
                scaler.step(optimizer)            # 原生路径，与 Phase A 逐位一致
            else:
                d1m.qh_adamw_step(optimizer, arm["alpha"], arm["mode"])
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            _C["optimizer_updates"] += 1
            theta_post = {n: p.detach().clone() for n, p in named}
            upd_norm = float(torch.cat(
                [(theta_post[n] - theta_pre[n]).reshape(-1) for n, _ in named]
            ).norm())
            row["update_norm"] = upd_norm
            if not math.isfinite(upd_norm):
                row["finite"] = False
            row["wall_s"] = time.perf_counter() - t_h0
            step_rows.append(row)
            if do_diag:
                diag_rows.extend(readout_diag_rows(
                    rt, seed, arm_name, arm, h_next, theta_pre, m_pre, v_pre, c,
                    theta_post, n0 + h_next, float(row["lr"])))
            _log(phase, (seed, n0), f"{tag} step h={h_next}",
                 updates=1, finite=row["finite"])
            if h_next in snapshot_points:
                _snap(h_next, cum_valid)
            if not row["finite"]:
                _C["failures"] += 1
                ev = _save_failure_evidence(
                    rt, seed, arm_name, h_next, theta_pre, m_pre, v_pre, c,
                    RuntimeError("NON-FINITE train/preclip/update"))
                _log(phase, (seed, n0), f"{tag} NON-FINITE at h={h_next}; aborted",
                     failures=1, evidence=ev)
                aborted = True
                break
        except d1m.NumericalFailure as exc:
            _C["failures"] += 1
            _C["numerical_failures"] += 1
            row["finite"] = False
            step_rows.append(row)
            ev = _save_failure_evidence(rt, seed, arm_name, h_next,
                                        theta_pre, m_pre, v_pre, c, exc)
            _log(phase, (seed, n0), f"{tag} NUMERICAL_FAILURE at h={h_next}; aborted",
                 failures=1, evidence=ev)
            numerical_failure = repr(exc)
            aborted = True
            break
    return {"step_rows": step_rows, "monitor_records": monitor_records,
            "diag_rows": diag_rows, "snap_paths": snap_paths,
            "cum": {"input_tokens": cum_input_tokens,
                    "nonpad_tokens": cum_nonpad, "valid_tokens": cum_valid},
            "aborted": aborted, "numerical_failure": numerical_failure}


def _append_step_metrics(rows):
    p = _ensure_header("step_metrics.csv")
    with p.open("a") as f:
        for r in rows:
            f.write(",".join(str(r[k]) for k in (
                "seed", "arm", "h", "lr", "preclip_norm", "clip_coeff",
                "clip_active", "train_nll_sum", "train_valid_cnt", "input_slots",
                "nonpad_tokens", "update_norm", "finite", "true_step",
                "wall_s")) + "\n")


def _append_monitor_metrics(seed, arm_name, recs):
    p = _ensure_header("monitor_metrics.csv")
    with p.open("a") as f:
        for m in recs:
            f.write(",".join(str(x) for x in (
                seed, arm_name, m["h"], m["sum"], m["count"], m["mean"],
                m["input_slots"], m["nonpad_tokens"], m["cum_input_tokens"],
                m["cum_nonpad_tokens"], m["cum_valid_tokens"], m["wall_s"],
                m["npz"], m["shared_h0"], m["billed_eval"])) + "\n")


def _append_diag_rows(rows):
    p = _ensure_header("readout_diagnostics.csv")
    with p.open("a") as f:
        for r in rows:
            f.write(",".join(str(r[k]) for k in (
                "seed", "arm", "h", "param", "kind", "rA_norm", "rQ_norm",
                "rN_norm", "norm_resid", "dot_rA_rN", "cos_rA_rN", "c_dot_rA",
                "c_dot_rQ", "c_dot_rN", "c_dot_diff", "theta_norm",
                "adaptive_step_norm", "full_step_norm", "applied", "fallback",
                "temp_bytes_fp32", "numel", "eta", "true_step")) + "\n")


# ---------------------------------------------------------------------------
# 状态/RNG 签名（smoke off/on 不变性检查；沿用 Phase A 机制）
# ---------------------------------------------------------------------------

def state_bitwise_sig(rt):
    torch = rt["torch"]
    sd = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    ost = rt["optimizer"].state_dict()
    return {"theta": sd, "opt_state": {
        str(k): {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                 for kk, vv in v.items()}
        for k, v in ost["state"].items()},
        "opt_step": {str(k): int(v["step"]) for k, v in ost["state"].items()},
        "param_groups": [{"lr": g["lr"], "betas": g["betas"],
                          "weight_decay": g["weight_decay"]}
                         for g in ost["param_groups"]]}


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
# smoke（§8.2：40 updates + 6 monitor evals）
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _T0
    _T0 = time.perf_counter()
    _replay_ledger()
    for d in (RUNS, SNAPS, MON_ROWS, FAILS):
        d.mkdir(parents=True, exist_ok=True)
    from fr.rng import capture_rng_state
    from fr.stream_rng import seed_future_stream
    cfg_all = json.loads((OUT / "config.json").read_text())
    smoke_stream = int(cfg_all["smoke_stream_seed"])
    seed = SEEDS[0]
    cell = (seed, START)

    checks = {"cell": f"{seed}/{START}", "smoke_stream": smoke_stream,
              "updates_planned": 40, "monitor_evals_planned": 6,
              "identity_level": "bitwise(theta,m,v,step,lr) + RNG equality, per-rule"}

    rt = build_rt(seed, f"cuda:{gpu}")
    load_bank(rt, *cell)
    mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)

    monitor_log = []

    def run_rule(arm_name, n_steps, mon_points, diag_points, expect_h0):
        arm = ARMS[arm_name]
        load_bank(rt, *cell)
        seed_future_stream(smoke_stream)
        stream = make_train_stream(rt, smoke_stream, 2 * n_steps)
        shared = checks.get("_h0_rec") if not expect_h0 else None
        out = run_trajectory(rt, seed, arm_name, arm, stream, n_steps,
                             monitor_points=mon_points, diag_points=diag_points,
                             monitor_stacked=mon_stacked, phase="smoke",
                             k_lr=arm["k"], shared_h0=shared,
                             tag=f"smoke_s{seed}_{arm_name}")
        if expect_h0:
            for m in out["monitor_records"]:
                if m["h"] == 0:
                    checks["_h0_rec"] = m
        return out

    # ---- ADAMW_LR100: off(h8 monitor) / on(h0+h8 monitor, h1 diag) --------
    off = run_rule("ADAMW_LR100", 8, (8,), (), expect_h0=False)
    sig_off = state_bitwise_sig(rt)
    rng_off = capture_rng_state()
    on = run_rule("ADAMW_LR100", 8, (0, 8), (1,), expect_h0=True)
    sig_on = state_bitwise_sig(rt)
    rng_on = capture_rng_state()
    same, why = sig_equal(sig_off, sig_on)
    rsame, rwhy = rng_equal(rng_off, rng_on)
    checks["native_off_on_bitwise_equal"] = bool(same)
    checks["native_rng_equal"] = bool(rsame)
    checks["native_mismatch"] = why or rwhy
    checks["native_off_updates"] = len(off["step_rows"])
    checks["native_on_updates"] = len(on["step_rows"])
    checks["native_on_diag_rows_h1"] = len(on["diag_rows"])
    monitor_log.extend(off["monitor_records"])
    monitor_log.extend(on["monitor_records"])

    # ---- QH_NORM_A010: off / on ------------------------------------------
    offn = run_rule("QH_NORM_A010", 8, (8,), (), expect_h0=False)
    sig_offn = state_bitwise_sig(rt)
    rng_offn = capture_rng_state()
    onn = run_rule("QH_NORM_A010", 8, (8,), (1,), expect_h0=False)
    sig_onn = state_bitwise_sig(rt)
    rng_onn = capture_rng_state()
    same_n, why_n = sig_equal(sig_offn, sig_onn)
    rsame_n, rwhy_n = rng_equal(rng_offn, rng_onn)
    checks["qh_norm_off_on_bitwise_equal"] = bool(same_n)
    checks["qh_norm_rng_equal"] = bool(rsame_n)
    checks["qh_norm_mismatch"] = why_n or rwhy_n
    checks["qh_norm_off_updates"] = len(offn["step_rows"])
    checks["qh_norm_on_updates"] = len(onn["step_rows"])
    checks["qh_norm_on_diag_rows_h1"] = len(onn["diag_rows"])
    monitor_log.extend(offn["monitor_records"])
    monitor_log.extend(onn["monitor_records"])

    # ---- QH_RAW_A010: 8 步接口/有限性 -------------------------------------
    raw = run_rule("QH_RAW_A010", 8, (8,), (1,), expect_h0=False)
    checks["qh_raw_updates"] = len(raw["step_rows"])
    checks["qh_raw_all_finite"] = bool(all(r["finite"] for r in raw["step_rows"]))
    checks["qh_raw_on_diag_rows_h1"] = len(raw["diag_rows"])
    monitor_log.extend(raw["monitor_records"])

    # ---- 公共 h0（在 native on-run 已测）；检查初始状态高于 ell -----------
    ell = float(cfg_all["detection"]["ell_star"])
    h0_rec = checks.get("_h0_rec")
    checks["h0_evaluated_once"] = h0_rec is not None
    if h0_rec:
        checks["start_above_ell"] = bool(h0_rec["mean"] > ell)
        checks["start_target_mismatch"] = bool(h0_rec["mean"] <= ell)
        checks["h0_mean"] = h0_rec["mean"]
    checks.pop("_h0_rec", None)
    checks["monitor_evals"] = [dict((k, m[k]) for k in
                                    ("h", "sum", "count", "mean", "wall_s"))
                               for m in monitor_log]
    checks["monitor_eval_count"] = len(monitor_log)
    checks["updates"] = (len(off["step_rows"]) + len(on["step_rows"])
                         + len(offn["step_rows"]) + len(onn["step_rows"])
                         + len(raw["step_rows"]))
    checks["ledger"] = dict(_C)
    checks["verdict"] = "PASS" if (
        same and rsame and same_n and rsame_n
        and checks["native_off_updates"] == 8 and checks["native_on_updates"] == 8
        and checks["qh_norm_off_updates"] == 8 and checks["qh_norm_on_updates"] == 8
        and checks["qh_raw_updates"] == 8 and checks["qh_raw_all_finite"]
        and len(monitor_log) == 6 and checks["start_above_ell"]) else "FAIL"
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log("smoke_done", cell, checks["verdict"])
    print(json.dumps(checks, indent=1))
    del rt, mon_stacked
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main（§8.3：15 条轨迹 × 256 步；共享 h0；按 resource_log 续跑）
# ---------------------------------------------------------------------------

def run_main(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr.stream_rng import seed_future_stream
    cfg_all = json.loads((OUT / "config.json").read_text())
    streams = {int(c["seed"]): int(c["stream_seed"]) for c in cfg_all["cells"]}

    done = set()
    cells_path = RUNS / "main_cells.jsonl"
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("completed"):
                        done.add((r["seed"], r["arm"]))
                except Exception:
                    continue

    snap_manifest = []
    if (SNAPS / "SNAPSHOT_MANIFEST.json").exists():
        snap_manifest = json.loads((SNAPS / "SNAPSHOT_MANIFEST.json").read_text())

    for seed in SEEDS:
        rt = build_rt(seed, f"cuda:{gpu}")
        load_bank(rt, seed, START)
        mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)
        h0_rec = None
        for arm_name in ORDER[seed]:
            if (seed, arm_name) in done:
                print(f"SKIP completed cell {seed}/{arm_name}")
                continue
            arm = ARMS[arm_name]
            _check_caps("main", upcoming_updates=256, upcoming_train_fw=512,
                        upcoming_bw=512, upcoming_monitor_fw=16 * 128)
            t_cell0 = time.perf_counter()
            load_bank(rt, seed, START)
            seed_future_stream(streams[seed])
            stream = make_train_stream(rt, streams[seed], 2 * H_MAIN)
            out = run_trajectory(rt, seed, arm_name, arm, stream, H_MAIN,
                                 monitor_points=MONITOR_POINTS,
                                 diag_points=DIAG_POINTS,
                                 monitor_stacked=mon_stacked, phase="main",
                                 k_lr=arm["k"], shared_h0=h0_rec,
                                 tag=f"s{seed}_{arm_name}",
                                 snapshot_points=SNAP_POINTS)
            if h0_rec is None:
                for m in out["monitor_records"]:
                    if m["h"] == 0 and m.get("billed_eval"):
                        h0_rec = m
                        break
            completed = (len(out["step_rows"]) == H_MAIN) and not out["aborted"]
            wall = time.perf_counter() - t_cell0
            _append_step_metrics(out["step_rows"])
            _append_monitor_metrics(seed, arm_name, out["monitor_records"])
            _append_diag_rows(out["diag_rows"])
            for h, p in out["snap_paths"].items():
                snap_manifest.append({
                    "seed": seed, "arm": arm_name, "h": h, "path": p,
                    "sha256": _sha256(Path(p)), "true_step": START + h,
                    "global_micro": 2 * START + 2 * h})
            with cells_path.open("a") as f:
                f.write(json.dumps({
                    "seed": seed, "arm": arm_name,
                    "steps": len(out["step_rows"]), "completed": completed,
                    "aborted": out["aborted"],
                    "numerical_failure": out["numerical_failure"],
                    "wall_s": wall, "cum": out["cum"],
                    "updates": len(out["step_rows"]),
                    "train_fw": 2 * len(out["step_rows"]),
                    "train_bw": 2 * len(out["step_rows"]),
                    "monitor_evals": sum(1 for m in out["monitor_records"]
                                         if m.get("billed_eval")),
                    "monitor_fw": 128 * sum(1 for m in out["monitor_records"]
                                            if m.get("billed_eval")),
                    "ledger": dict(_C)}) + "\n")
            _log("cell_done", (seed, START),
                 f"s{seed}_{arm_name} steps={len(out['step_rows'])} "
                 f"completed={completed}")
            print(f"CELL_DONE s{seed} {arm_name} steps={len(out['step_rows'])} "
                  f"completed={completed} wall={wall:.1f}s ledger={dict(_C)}")
        del rt, mon_stacked

    (SNAPS / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(snap_manifest, indent=1) + "\n")
    print("MAIN_DONE", OUT)


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
