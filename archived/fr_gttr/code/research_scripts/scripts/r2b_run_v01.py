#!/usr/bin/env python3
"""GTTR-R2B 桥接 runner —— preflight / smoke / main。

协议：FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md（2026-09-24）。
读取 R1 封存 primary r=.05 k=2 受限解，在每条轨迹的第一个 optimizer update
真实执行一次（BASE_K2 a=(0,0) 原生；LR_ONLY y=0 原生×(1+x)；FULL y>0 走
qh RAW 读出 eta*(1+x)、alpha=y/(1+x)），第 2..256 步全部回到共同 k=2 原生
AdamW schedule。6 state × 2 stream × 3 arm = 36 条 × 256 步。

V_monitor：h=0(每 state 共享),1,4,16,...,256；V_ctrl：h0 共享 + h1 每轨迹；
正式检测只用 h=0,16,...,256 连续两次 <=ell（PhaseA 原阈值按 start 全精度读）。
预算硬上限：updates<=9600；total fw<=110000；bw micro<=19200；
monitor eval<=700；ctrl eval<=60；新增 q/HVP/probe=0；V_final=0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r2b_solved_update_bridge_v01"
RUNS = OUT / "runs"
SNAPS = OUT / "snapshots"
MON_ROWS = OUT / "monitor_rows"
FAILS = OUT / "failures"
A1 = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
R1_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
R1_SOL = R1_OUT / "restricted_solutions.csv"
R1_MODEL = R1_OUT / "model_coefficients.csv"
R1_SEAL = R1_OUT / "prediction_seal.json"
R1_FITCSV = R1_OUT / "per_probe_coefficients.csv"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md"

SEEDS = (2030, 2031, 2032)
STARTS = (46, 82)
CELLS = [(s, t) for s in SEEDS for t in STARTS]          # S0..S5
ARMS = ("BASE_K2", "GTTR_LR_ONLY", "GTTR_FULL")
ORDER = {"BASE_K2": 0, "GTTR_LR_ONLY": 1, "GTTR_FULL": 2}

STREAM_BASE = 202609240200
SMOKE_STREAM_PROPOSED = 202609240300

H_MAIN = 256
MON_H = (1, 4) + tuple(range(16, H_MAIN + 1, 16))        # 每轨迹 18 点
SNAP_H = (1, 256)
CTRL_H = (1,)
EVAL_BATCH_ROWS = 32
MONITOR_N_ROWS = 4096

CAP_UPDATES = 9600
CAP_TOTAL_FW = 110000
CAP_BW = 19200
CAP_MON_EVALS = 700
CAP_CTRL_EVALS = 60

_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0

_C = {"optimizer_updates": 0, "train_fw": 0, "train_bw": 0,
      "monitor_fw": 0, "monitor_evals": 0, "ctrl_fw": 0, "ctrl_evals": 0,
      "snapshots": 0, "failures": 0, "q_hvp_probe_calls": 0, "v_final_calls": 0}
_T0 = None


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------

def _total_fw():
    return _C["train_fw"] + _C["monitor_fw"] + _C["ctrl_fw"]


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


def _log(phase, cell, note="", **kw):
    rec = {"ts": time.time(), "phase": phase,
           "cell": f"{cell[0]}/{cell[1]}" if cell else None,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "note": note, "total_fw": _total_fw(), **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(up_updates=0, up_fw=0, up_bw=0, up_mon=0, up_ctrl=0):
    if _C["optimizer_updates"] + up_updates > CAP_UPDATES:
        raise SystemExit(f"BUDGET_BLOCKED: updates cap {CAP_UPDATES}")
    if _total_fw() + up_fw > CAP_TOTAL_FW:
        raise SystemExit(f"BUDGET_BLOCKED: total fw cap {CAP_TOTAL_FW}")
    if _C["train_bw"] + up_bw > CAP_BW:
        raise SystemExit(f"BUDGET_BLOCKED: bw cap {CAP_BW}")
    if _C["monitor_evals"] + up_mon > CAP_MON_EVALS:
        raise SystemExit(f"BUDGET_BLOCKED: monitor eval cap {CAP_MON_EVALS}")
    if _C["ctrl_evals"] + up_ctrl > CAP_CTRL_EVALS:
        raise SystemExit(f"BUDGET_BLOCKED: ctrl eval cap {CAP_CTRL_EVALS}")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# 运行时（冻结路径，与 Phase A / D1 / R1 相同）
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
    from pathlib import Path as P
    mon = P(str(A1 / "data" / "v_monitor_rows4096_8192.jsonl"))
    ds = PretrainDataset(str(mon),
                         AutoTokenizer.from_pretrained(
                             str(PROJECT_ROOT / "minimind" / "model")),
                         max_length=int(rt["cfg"]["max_seq_len"]))
    chunks = [list(range(i, i + EVAL_BATCH_ROWS))
              for i in range(0, MONITOR_N_ROWS, EVAL_BATCH_ROWS)]
    assert len(chunks) == 128
    return ds, chunks, a0g._stack_all(ds, chunks, rt["device"])


# ---------------------------------------------------------------------------
# 评价（monitor=PhaseA 语义；ctrl=R1 FP32/MATH 语义；均 RNG 隔离）
# ---------------------------------------------------------------------------

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
                ce = F.cross_entropy(
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
    np.savez_compressed(MON_ROWS / name,
                        row_id=np.asarray(row_ids, dtype=np.int32),
                        nll_sum=np.asarray(row_sums, dtype=np.float64),
                        valid_count=np.asarray(row_cnts, dtype=np.int32))
    return {"h": h, "sum": total_sum, "count": total_cnt,
            "mean": total_sum / max(total_cnt, 1),
            "input_slots": slots, "nonpad_tokens": nonpad,
            "wall_s": wall, "npz": name, "billed_eval": True}


def ctrl_valid_count(rt):
    if "_ctrl_count" in rt:
        return rt["_ctrl_count"]
    n = 0
    for _, y in rt["val_score_tensors"]:
        n += int((y[:, 1:] != -100).sum())
    rt["_ctrl_count"] = n
    return n


def eval_ctrl_current(rt, tag, h):
    """V_ctrl（64 行 FP32/no-autocast/MATH/sum-count）当前参数求值。"""
    import torch.nn.functional as F
    from fr.rng import capture_rng_state, restore_rng_state
    import a0g_decision_utility_v01 as a0g
    torch = rt["torch"]
    model = rt["model"]
    _check_caps(up_ctrl=1, up_fw=len(rt["val_score_tensors"]))
    rng_snap = capture_rng_state()
    was_training = model.training
    total = 0.0
    t0 = time.perf_counter()
    model.eval()
    try:
        with torch.no_grad(), a0g._math_sdpa_context(torch):
            for x, y in rt["val_score_tensors"]:
                res = model(x)
                logits = res.logits
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                total += float(ce.sum())
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_snap)
    _C["ctrl_fw"] += len(rt["val_score_tensors"])
    _C["ctrl_evals"] += 1
    return {"tag": tag, "h": h, "sum": total, "count": ctrl_valid_count(rt),
            "mean": total / max(ctrl_valid_count(rt), 1),
            "wall_s": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# 状态/RNG 签名（smoke off/on 位级一致检查；D1 机制）
# ---------------------------------------------------------------------------

def state_bitwise_sig(rt):
    sd = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    ost = rt["optimizer"].state_dict()
    return {"theta": sd, "opt_state": ost["state"],
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
    if not torch.equal(a["torch_cpu"], b["torch_cpu"]):
        return False, "torch_cpu"
    ca, cb = a.get("torch_cuda_all"), b.get("torch_cuda_all")
    if (ca is None) != (cb is None):
        return False, "cuda_rng_presence"
    if ca is not None:
        for x, y in zip(ca, cb):
            if not torch.equal(x, y):
                return False, "torch_cuda"
    return True, ""


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight():
    import torch
    from fr import r2b_bridge_v01 as br
    from fr import r1_objective_identification_v01 as r1m

    for d in (OUT, RUNS, SNAPS, MON_ROWS, FAILS):
        d.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1) R1 seal 与主解
    ok, why = r1m.verify_seal(R1_SEAL, [R1_FITCSV, R1_MODEL, R1_SOL])
    assert ok, f"R1 seal broken: {why}"
    sols_all = br.read_primary_solutions(R1_SOL)
    sols = {k: v for k, v in sols_all.items() if k[2] == "2"}
    assert len(sols) == 6, f"expect 6 k2 states, got {len(sols)}"
    verify = br.verify_solutions_against_coefficients(sols, R1_MODEL)
    bad = {k: v for k, v in verify.items() if not v.get("ok")}
    assert not bad, f"solution recheck mismatch: {bad}"
    actions = {}
    for (seed, start, _k), v in sols.items():
        xLR = v["LR"][0]
        x2, y2 = v["2D"]
        eta_new, alpha_new, native = br.first_step_plan(x2, y2, 2 * int(start) + 2)
        actions[f"{seed}/{start}"] = {
            "x_LR": xLR, "x_2D": x2, "y_2D": y2,
            "eta_mult_LR": 1 + xLR, "eta_mult_2D": 1 + x2,
            "alpha_new_2D": y2 / (1 + x2), "full_uses_native": native}
    rep["r1_actions"] = actions
    rep["r1_provenance"] = {
        "restricted_solutions": {"path": str(R1_SOL), "sha256": _sha256(R1_SOL)},
        "model_coefficients": {"path": str(R1_MODEL),
                               "sha256": _sha256(R1_MODEL)},
        "prediction_seal": {"path": str(R1_SEAL),
                            "sha256": _sha256(R1_SEAL)},
        "consistency_recheck": "r1m 受限求解器对封存 b/K 复算与 CSV 解一致"
                               "（一致性与版本检查，不构成新拟合）"}

    # 2) bank
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
        bank[f"{seed}/{start}"] = {"path": str(p), "n0": start,
                                   "gm_first_update_last_micro": 2 * start + 2,
                                   "sha256": _sha256(p)}
    rep["bank"] = bank

    # 3) 首步 eta（真实 gm 调用位置）+ configured_peak 分列
    etas = {}
    for seed, start in CELLS:
        gm = 2 * start + 2
        etas[f"{seed}/{start}"] = {
            "n_before": start, "n_after_first_update": start + 1,
            "gm_used": gm, "configured_peak": 5e-4,
            "eta_base": br.eta_base(gm), "eta_ref": br.eta_ref(gm),
            "eta_ref_step2": br.eta_ref(gm + 2),
            "eta_ref_tail": 1e-4}
    rep["first_step_etas"] = etas

    # 4) 配方核对
    a1cfg = json.loads((A1 / "config.json").read_text())
    a1train = a1cfg["train"]
    recipe_expect = {"optimizer": "torch.optim.AdamW", "betas": [0.9, 0.999],
                     "eps": 1e-8, "weight_decay": 0.01, "clip_tau": 1.0,
                     "batch_size": 4, "accumulation_steps": 2}
    conflicts = {k: {"a1": a1train.get(k), "expected": v}
                 for k, v in recipe_expect.items() if a1train.get(k) != v}
    assert not conflicts, f"recipe conflict: {conflicts}"
    assert a1cfg["schedule_id"] == "finite_cosine_then_hold_5e-5_v01"
    rep["recipe_ok"] = True

    # 5) ell（PhaseA 原全精度，按 start 读，不手抄）
    ell = {}
    with (A1 / "target_calibration.csv").open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            row = line.strip().split(",")
            d = dict(zip(header, row))
            ell[int(d["start"])] = float(d["ell_candidate"])
    assert abs(ell[46] - 7.020043501537879) < 1e-12
    assert abs(ell[82] - 6.979538126804536) < 1e-12
    rep["ell"] = {str(k): v for k, v in ell.items()}

    # 6) 流种子冲突（整块 +1000）
    used = set()
    pat = re.compile(r"(?<!\d)(\d{12,13})(?!\d)")
    for p in PROJECT_ROOT.glob("outputs/**/*.json"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for mch in pat.finditer(txt):
            v = mch.group(1)
            if len(v) >= 12:
                used.add(int(v))
    base = STREAM_BASE
    smoke = SMOKE_STREAM_PROPOSED
    streams = {}
    while True:
        streams = {f"{s}/{t}": [base + 2 * i, base + 2 * i + 1]
                   for i, (s, t) in enumerate(CELLS)}
        if not any(v in used for vs in streams.values() for v in vs) \
                and smoke not in used:
            break
        base += 1000
        smoke += 1000
    rep["streams"] = {"base": base, "by_state": streams, "smoke": smoke,
                      "shifted_from_proposed": base != STREAM_BASE,
                      "metadata_only_decision": True}

    # 7) 磁盘（72 快照 ~4.8GB）
    import shutil
    free_gb = shutil.disk_usage(OUT).free / 1e9
    need_gb = 72 * 0.070
    rep["disk"] = {"free_gb": round(free_gb, 1), "need_gb_est": round(need_gb, 1),
                   "plan": "full h1+h256 snapshots" if free_gb > need_gb + 10
                   else "fallback: h1 full + h256 results"}
    assert free_gb > need_gb, "insufficient disk for snapshots"

    # 8) CPU 测试
    test_mod = "tests.test_r2b_bridge_v01"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "unittest", test_mod, "-v"],
                       cwd=PROJECT_ROOT / "research", capture_output=True,
                       text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                            "HOME": "/home/b205"})
    out_txt = r.stdout + r.stderr
    passed = r.returncode == 0 and out_txt.strip().splitlines()[-1].strip() == "OK"
    impl = {"unittest_module": test_mod, "returncode": r.returncode,
            "all_pass": bool(passed),
            "tail": out_txt.strip().splitlines()[-1].strip() if out_txt.strip() else "",
            "wall_s": round(time.time() - t0, 3),
            "coverage": {
                "C1_mapping_and_wd": "covered",
                "C2_native_routing_no_norm_matching": "covered",
                "C3_intervention_mask_first_update_only": "covered",
                "C4_first_step_shared_moments": "covered",
                "C5_read_primary_solutions_only": "covered",
                "C6_schedule_index_monotone_tail": "covered",
                "C7_detection_grid_semantics": "covered",
                "C8_isolation_unique_params_grad_none": "covered"}}
    (OUT / "implementation_checks.json").write_text(json.dumps(impl, indent=1))
    assert passed, f"CPU unittest failed:\n{out_txt[-2000:]}"
    rep["implementation"] = {"all_pass": True}

    # 9) config + manifest
    config = {
        "round": "fr_gttr_r2b_solved_update_bridge_v01",
        "protocol": "FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md",
        "protocol_sha256": _sha256(PROTOCOL),
        "cells": [{"seed": s, "start": t, "S": i,
                   "streams": streams[f"{s}/{t}"]}
                  for i, (s, t) in enumerate(CELLS)],
        "arms": {"BASE_K2": "a=(0,0) 原生 k=2",
                 "GTTR_LR_ONLY": "R1 a_LR(z,2)：首步原生 eta_ref*(1+x_LR)",
                 "GTTR_FULL": "R1 a_2D(z,2)：首步 qh RAW，eta_ref*(1+x_2D)，"
                              "alpha=y_2D/(1+x_2D)"},
        "intervention": "仅第一个 optimizer update；第 2..256 步全部原生 k=2",
        "schedule": {"id": "finite_cosine_then_hold_5e-5_v01", "k": 2.0,
                     "tail_ref": 1e-4, "configured_peak": 5e-4,
                     "first_update_gm": {f"{s}/{t}": 2 * t + 2
                                         for (s, t) in CELLS}},
        "monitor": {"points_per_traj": list(MON_H), "shared_h0": True,
                    "rows": "[4096:8192)", "eval_batch_rows": 32},
        "ctrl": {"h": [0, 1], "shared_h0": True, "no_gradients": True,
                 "semantics": "R1 FP32/MATH/sum-count 64 行"},
        "detection": {"grid": list(range(0, 257, 16)),
                      "rule": "连续两次 <= ell 的第二时点",
                      "ell_by_start": rep["ell"],
                      "h1_h4_excluded": True},
        "budget": {"caps": {"updates": CAP_UPDATES, "total_fw": CAP_TOTAL_FW,
                            "bw_micro": CAP_BW, "monitor_evals": CAP_MON_EVALS,
                            "ctrl_evals": CAP_CTRL_EVALS,
                            "q_hvp_probe": 0, "v_final": 0},
                   "plan": {"main_updates": 6 * 2 * 3 * 256,
                            "smoke_updates": 40, "monitor_evals": 660,
                            "ctrl_evals": 42}},
        "snapshots": {"h": list(SNAP_H), "n": 72,
                      "plan": rep["disk"]["plan"]},
        "code": {"git_head": _git_head(),
                 "runner_sha256": _sha256(Path(__file__).resolve())}}
    (OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    manifest = {
        "state_stream_action": [
            {"seed": s, "start": t, "S": i,
             "stream_seeds": streams[f"{s}/{t}"],
             "smoke_stream": smoke,
             "n0": t, "gm_first": 2 * t + 2,
             "action": actions[f"{s}/{t}"],
             "eta_ref_first": etas[f"{s}/{t}"]["eta_ref"],
             "ell": ell[t]}
            for i, (s, t) in enumerate(CELLS)],
        "r1_provenance": rep["r1_provenance"],
        "bank": bank,
        "sampler": "b0.make_stream（冻结 train pool；新流，不复用 R1 FIT/AUDIT）",
        "note": "不能将 36 轨迹称 36 独立模型；3 训练 seeds"}
    (OUT / "state_stream_action_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"actions": actions, "streams": streams,
                      "smoke": smoke, "ell": rep["ell"],
                      "disk": rep["disk"], "unittest": True}, indent=1))
    print("PREFLIGHT_OK")


# ---------------------------------------------------------------------------
# 轨迹内核
# ---------------------------------------------------------------------------

def run_trajectory(rt, seed, start, stream, arm_name, action, n_steps,
                   mon_points, ctrl_points, snap_points, phase, diag=False,
                   tag=None, mon_stacked=None, snap_manifest=None):
    """从已加载 bank 状态跑 n_steps 步。首步按 action 施加，其后原生 k=2。

    返回 dict：step_rows / monitor_records / ctrl_records / snap_paths /
    cum / aborted / first_step_diag。
    """
    import torch
    from fr import d1_direct_readout_v01 as d1m
    from fr import r2b_bridge_v01 as br
    from fr.native_schedule import set_native_micro_lr
    from fr.full_state import save_full_training_state

    torch_ = rt["torch"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    n0 = start
    gm0 = 2 * n0
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    named = rt["named"]
    tag = tag or f"s{seed}_t{start}_{arm_name}"
    x, y = action
    eta_new_first, alpha_new, full_native = br.first_step_plan(x, y, gm0 + 2)

    step_rows = []
    mon_recs = []
    ctrl_recs = []
    snap_paths = {}
    cum_input = cum_nonpad = cum_valid = 0
    aborted = False
    numerical_failure = None
    first_diag = {}

    for h in range(1, n_steps + 1):
        _check_caps(up_updates=1, up_fw=accum, up_bw=accum)
        intervention = (h == 1)
        row = {"seed": seed, "start": start, "stream": stream["stream_seed"],
               "arm": arm_name, "h": h, "gm": gm0 + 2 * h,
               "lr_base": None, "lr_ref": None, "lr_applied": None,
               "intervention": intervention, "alpha": None,
               "preclip_norm": None, "clip_active": None,
               "update_norm": None, "finite": True,
               "input_slots": 0, "nonpad_tokens": 0, "valid_tokens": 0,
               "cum_input": 0, "cum_nonpad": 0, "cum_valid": 0,
               "wall_s": None}
        t_h0 = time.perf_counter()
        theta_pre = {n_: p.detach().clone() for n_, p in named} \
            if (h == 1 and diag) else None
        try:
            for a in range(accum):
                micro_idx = 2 * (h - 1) + a + 1
                lr_base = set_native_micro_lr(optimizer, rt["eta"],
                                              gm0 + micro_idx, 256,
                                              float(cfg["learning_rate"]))
                lr_ref = 2.0 * lr_base
                if intervention:
                    lr_apply = eta_new_first if a == accum - 1 else lr_ref
                else:
                    lr_apply = lr_ref
                optimizer.param_groups[0]["lr"] = lr_apply
                if a == accum - 1:
                    row["lr_base"] = lr_base
                    row["lr_ref"] = lr_ref
                    row["lr_applied"] = lr_apply
                xb, yb = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
                with rt["autocast_ctx"]:
                    res = model(xb, labels=yb)
                    loss = (res.loss + res.aux_loss) / accum
                scaler.scale(loss).backward()
                _C["train_fw"] += 1
                _C["train_bw"] += 1
                row["input_slots"] += int(xb.numel())
                row["nonpad_tokens"] += int((yb != -100).sum())
                row["valid_tokens"] += int((yb[:, 1:] != -100).sum())
                if not math.isfinite(float(res.loss)):
                    row["finite"] = False
            cum_input += row["input_slots"]
            cum_nonpad += row["nonpad_tokens"]
            cum_valid += row["valid_tokens"]
            row["cum_input"] = cum_input
            row["cum_nonpad"] = cum_nonpad
            row["cum_valid"] = cum_valid
            row["alpha"] = alpha_new if intervention else 0.0
            scaler.unscale_(optimizer)
            preclip = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])).detach().cpu())
            row["preclip_norm"] = preclip
            row["clip_active"] = bool(preclip > _TAU)
            if not math.isfinite(preclip):
                row["finite"] = False
            # ---- 更新：仅首步施加动作 -----------------------------------
            if intervention and not full_native:
                d1m.qh_adamw_step(optimizer, alpha_new, "RAW")
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            _C["optimizer_updates"] += 1
            if h == 1 and diag:
                theta_post = {n_: p.detach().clone() for n_, p in named}
                row["update_norm"] = float(torch.cat(
                    [(theta_post[n_] - theta_pre[n_]).reshape(-1)
                     for n_, _ in named]).norm())
                first_diag = {"alpha_new": alpha_new,
                              "eta_new": eta_new_first,
                              "full_native": full_native,
                              "update_norm": row["update_norm"]}
            step_rows.append(row)
            if h in mon_points and mon_stacked is not None:
                mon_recs.append(monitor_full(rt, mon_stacked, tag, h))
            if h in ctrl_points:
                ctrl_recs.append(eval_ctrl_current(rt, tag, h))
            if h in snap_points:
                path = SNAPS / f"{tag}_h{h:03d}.pt"
                save_full_training_state(
                    path, model, optimizer, scaler,
                    metadata={"protocol": "FR_GTTR_R2B_SNAPSHOT",
                              "seed": seed, "start": start,
                              "stream_seed": stream["stream_seed"],
                              "arm": arm_name, "h": h,
                              "true_step": n0 + h,
                              "global_micro": gm0 + 2 * h})
                _C["snapshots"] += 1
                snap_paths[h] = str(path)
                if snap_manifest is not None:
                    snap_manifest.append({
                        "seed": seed, "start": start,
                        "stream": stream["stream_seed"], "arm": arm_name,
                        "h": h, "path": str(path), "sha256": _sha256(path),
                        "true_step": n0 + h})
            row["wall_s"] = time.perf_counter() - t_h0
            _log(phase, (seed, start), f"{tag} step {h}", updates=1,
                 finite=row["finite"])
            if not row["finite"]:
                _C["failures"] += 1
                aborted = True
                break
        except d1m.NumericalFailure as exc:
            _C["failures"] += 1
            row["finite"] = False
            step_rows.append(row)
            numerical_failure = repr(exc)
            _log(phase, (seed, start), f"{tag} NUMERICAL_FAILURE h={h}",
                 failures=1, err=repr(exc))
            aborted = True
            break
    return {"step_rows": step_rows, "monitor_records": mon_recs,
            "ctrl_records": ctrl_recs, "snap_paths": snap_paths,
            "cum": {"input": cum_input, "nonpad": cum_nonpad,
                    "valid": cum_valid},
            "aborted": aborted, "numerical_failure": numerical_failure,
            "first_diag": first_diag}


# ---------------------------------------------------------------------------
# smoke（§8：40 updates + 6 monitor evals）
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _T0
    _T0 = time.perf_counter()
    _replay_ledger()
    from fr import r2b_bridge_v01 as br
    from fr.rng import capture_rng_state
    from fr.stream_rng import seed_future_stream
    cfg_all = json.loads((OUT / "config.json").read_text())
    manifest = json.loads((OUT / "state_stream_action_manifest.json").read_text())
    smoke_stream = None
    for cell in manifest["state_stream_action"]:
        if cell["seed"] == 2030 and cell["start"] == 46:
            smoke_stream = cell["smoke_stream"]
    assert smoke_stream is not None
    seed, start = 2030, 46
    act = next(c["action"] for c in manifest["state_stream_action"]
               if c["seed"] == 2030 and c["start"] == 46)

    checks = {"cell": f"{seed}/{start}", "smoke_stream": smoke_stream,
              "updates_planned": 40, "monitor_evals_planned": 6,
              "identity_level": "bitwise(theta,m,v,step,lr)+RNG, per-rule"}
    rt = build_rt(seed, f"cuda:{gpu}")
    load_bank(rt, seed, start)
    _, _, mon_stacked = make_monitor_chunks(rt)
    actions = {"BASE_K2": (0.0, 0.0),
               "GTTR_LR_ONLY": (act["x_LR"], 0.0),
               "GTTR_FULL": (act["x_2D"], act["y_2D"])}
    monitor_log = []
    h0_rec = None

    def run8(arm_name, diag, expect_h0):
        nonlocal h0_rec
        action = actions[arm_name]
        load_bank(rt, seed, start)
        seed_future_stream(smoke_stream)
        stream = make_stream(rt, smoke_stream, 16)
        out = run_trajectory(
            rt, seed, start, stream, arm_name, action, 8,
            mon_points={8}, ctrl_points=(), snap_points=(), phase="smoke",
            diag=diag, tag=f"smoke_{arm_name}{'_diag' if diag else ''}",
            mon_stacked=mon_stacked)
        if expect_h0:
            pass
        return out

    # 公共 h0（bank theta 上测一次）
    h0_rec = monitor_full(rt, mon_stacked, "smoke_shared", 0,
                          npz_name="smoke_shared_h000.npz")
    monitor_log.append({**h0_rec, "shared_h0": True, "billed_eval": True,
                        "arm_tag": "shared"})

    off = run8("BASE_K2", diag=False, expect_h0=False)
    sig_off = state_bitwise_sig(rt)
    rng_off = capture_rng_state()
    on = run8("BASE_K2", diag=True, expect_h0=False)
    sig_on = state_bitwise_sig(rt)
    rng_on = capture_rng_state()
    same, why = sig_equal(sig_off, sig_on)
    rsame, _ = rng_equal(rng_off, rng_on)
    checks["base_off_on_bitwise_equal"] = bool(same)
    checks["base_rng_equal"] = bool(rsame)
    checks["base_mismatch"] = why

    offf = run8("GTTR_FULL", diag=False, expect_h0=False)
    sig_offf = state_bitwise_sig(rt)
    rng_offf = capture_rng_state()
    onf = run8("GTTR_FULL", diag=True, expect_h0=False)
    sig_onf = state_bitwise_sig(rt)
    rng_onf = capture_rng_state()
    samef, whyf = sig_equal(sig_offf, sig_onf)
    rsamef, _ = rng_equal(rng_offf, rng_onf)
    checks["full_off_on_bitwise_equal"] = bool(samef)
    checks["full_rng_equal"] = bool(rsamef)
    checks["full_mismatch"] = whyf

    lro = run8("GTTR_LR_ONLY", diag=True, expect_h0=False)
    for name, out in (("BASE_K2_off", off), ("BASE_K2_on", on),
                      ("GTTR_FULL_off", offf), ("GTTR_FULL_on", onf),
                      ("GTTR_LR_ONLY", lro)):
        for m in out["monitor_records"]:
            monitor_log.append({**m, "shared_h0": False, "billed_eval": True,
                                "arm_tag": name})
    checks.update({
        "base_off_updates": len(off["step_rows"]),
        "base_on_updates": len(on["step_rows"]),
        "full_off_updates": len(offf["step_rows"]),
        "full_on_updates": len(onf["step_rows"]),
        "lr_only_updates": len(lro["step_rows"]),
        "all_finite": all(r["finite"] for r in
                          off["step_rows"] + on["step_rows"]
                          + offf["step_rows"] + onf["step_rows"]
                          + lro["step_rows"]),
        "first_step_lr_applied": {
            "BASE": on["step_rows"][0]["lr_applied"],
            "LR_ONLY": lro["step_rows"][0]["lr_applied"],
            "FULL": onf["step_rows"][0]["lr_applied"]},
        "alpha_full_first_step": onf["step_rows"][0]["alpha"],
        "monitor_evals": len(monitor_log),
        "h0_mean": h0_rec["mean"],
        "ell_used": cfg_all["detection"]["ell_by_start"]["46"],
        "start_above_ell": bool(
            h0_rec["mean"] > float(cfg_all["detection"]["ell_by_start"]["46"]))})
    checks["verdict"] = "PASS" if (
        same and rsame and samef and rsamef
        and checks["base_off_updates"] == 8 and checks["base_on_updates"] == 8
        and checks["full_off_updates"] == 8 and checks["full_on_updates"] == 8
        and checks["lr_only_updates"] == 8 and checks["all_finite"]
        and len(monitor_log) == 6 and checks["start_above_ell"]) else "FAIL"
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _append(OUT / "monitor_metrics.csv", MON_HEADER, [(
        seed, start, smoke_stream, m.get("arm_tag", "shared"), m["h"],
        m["sum"], m["count"], m["mean"], m["npz"], m["shared_h0"],
        m["billed_eval"], m.get("wall_s", 0.0)) for m in monitor_log])
    _log("smoke_done", (seed, start), checks["verdict"])
    print(json.dumps(checks, indent=1))
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main（36 条 × 256 步）
# ---------------------------------------------------------------------------

STEP_HEADER = ("seed,start,stream,arm,h,gm,lr_base,lr_ref,lr_applied,"
               "intervention,alpha,preclip_norm,clip_active,update_norm,"
               "finite,input_slots,nonpad_tokens,valid_tokens,cum_input,"
               "cum_nonpad,cum_valid,wall_s\n")
MON_HEADER = ("seed,start,stream,arm,h,sum,count,mean,npz,shared_h0,"
              "billed_eval,wall_s\n")
CTRL_HEADER = "seed,start,stream,arm,h,sum,count,mean,wall_s\n"


def _append(path, header, rows):
    if not path.exists():
        path.write_text(header)
    with path.open("a") as f:
        for r in rows:
            f.write(",".join(str(v_) for v_ in r) + "\n")


def run_main(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr.stream_rng import seed_future_stream
    manifest = json.loads((OUT / "state_stream_action_manifest.json").read_text())
    cfg = json.loads((OUT / "config.json").read_text())
    cellmap = {f"{c['seed']}/{c['start']}": c for c in
               manifest["state_stream_action"]}
    ell = {int(k): v for k, v in cfg["detection"]["ell_by_start"].items()}

    done = set()
    cells_path = RUNS / "main_cells.jsonl"
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("completed"):
                        done.add((r["seed"], r["start"], r["stream"],
                                  r["arm"]))
                except Exception:
                    continue
    snap_manifest = []
    if (SNAPS / "SNAPSHOT_MANIFEST.json").exists():
        snap_manifest = json.loads(
            (SNAPS / "SNAPSHOT_MANIFEST.json").read_text())

    import torch
    for seed in SEEDS:
        rt = build_rt(seed, f"cuda:{gpu}")
        _, _, mon_stacked = make_monitor_chunks(rt)
        for start in STARTS:
            key = f"{seed}/{start}"
            cell = cellmap[key]
            act = cell["action"]
            actions = {"BASE_K2": (0.0, 0.0),
                       "GTTR_LR_ONLY": (act["x_LR"], 0.0),
                       "GTTR_FULL": (act["x_2D"], act["y_2D"])}
            load_bank(rt, seed, start)
            # 每 state 共享 h0（monitor + ctrl），bank theta
            h0m = monitor_full(rt, mon_stacked, f"s{seed}_t{start}_shared", 0,
                               npz_name=f"s{seed}_t{start}_shared_h000.npz")
            h0c = eval_ctrl_current(rt, f"s{seed}_t{start}", 0)
            if h0m["mean"] <= ell[start]:
                _log("main", (seed, start), "START_TARGET_MISMATCH",
                     h0_mean=h0m["mean"])
            _append(OUT / "monitor_metrics.csv", MON_HEADER, [(
                seed, start, "shared", "shared", 0, h0m["sum"], h0m["count"],
                h0m["mean"], h0m["npz"], True, True, h0m["wall_s"])])
            _append(OUT / "control_step1.csv", CTRL_HEADER, [(
                seed, start, "shared", "shared", 0, h0c["sum"],
                h0c["count"], h0c["mean"], h0c["wall_s"])])
            _log("main", (seed, start), "shared h0 evaluated",
                 monitor_fw=128, ctrl_fw=8)
            for stream_seed in cell["stream_seeds"]:
                for arm_name in ARMS:
                    if (seed, start, stream_seed, arm_name) in done:
                        print(f"SKIP done {key}/{stream_seed}/{arm_name}")
                        continue
                    _check_caps(up_updates=H_MAIN, up_fw=2 * H_MAIN + 128 * 18
                                + 8, up_bw=2 * H_MAIN, up_mon=18, up_ctrl=1)
                    t0 = time.perf_counter()
                    load_bank(rt, seed, start)
                    seed_future_stream(stream_seed)
                    stream = make_stream(rt, stream_seed, 2 * H_MAIN)
                    tag = f"s{seed}_t{start}_st{stream_seed}_{arm_name}"
                    out = run_trajectory(
                        rt, seed, start, stream, arm_name, actions[arm_name],
                        H_MAIN, mon_points=set(MON_H),
                        ctrl_points=set(CTRL_H), snap_points=set(SNAP_H),
                        phase="main", diag=True, tag=tag,
                        mon_stacked=mon_stacked, snap_manifest=snap_manifest)
                    completed = (len(out["step_rows"]) == H_MAIN
                                 and not out["aborted"])
                    _append(OUT / "step_metrics.csv", STEP_HEADER,
                            [(r["seed"], r["start"], r["stream"], r["arm"],
                              r["h"], r["gm"], r["lr_base"], r["lr_ref"],
                              r["lr_applied"], r["intervention"], r["alpha"],
                              r["preclip_norm"], r["clip_active"],
                              r["update_norm"], r["finite"],
                              r["input_slots"], r["nonpad_tokens"],
                              r["valid_tokens"], r["cum_input"],
                              r["cum_nonpad"], r["cum_valid"], r["wall_s"])
                             for r in out["step_rows"]])
                    _append(OUT / "monitor_metrics.csv", MON_HEADER,
                            [(seed, start, stream_seed, arm_name, m["h"],
                              m["sum"], m["count"], m["mean"], m["npz"],
                              False, True, m["wall_s"])
                             for m in out["monitor_records"]])
                    _append(OUT / "control_step1.csv", CTRL_HEADER,
                            [(seed, start, stream_seed, arm_name, c["h"],
                              c["sum"], c["count"], c["mean"], c["wall_s"])
                             for c in out["ctrl_records"]])
                    with cells_path.open("a") as f:
                        f.write(json.dumps({
                            "seed": seed, "start": start,
                            "stream": stream_seed, "arm": arm_name,
                            "steps": len(out["step_rows"]),
                            "completed": completed,
                            "aborted": out["aborted"],
                            "numerical_failure": out["numerical_failure"],
                            "wall_s": time.perf_counter() - t0,
                            "cum": out["cum"],
                            "snaps": {str(k_): v_ for k_, v_
                                      in out["snap_paths"].items()},
                            "first_diag": out["first_diag"],
                            "ledger": dict(_C)}) + "\n")
                    _log("cell_done", (seed, start),
                         f"{tag} steps={len(out['step_rows'])} "
                         f"completed={completed}")
                    print(f"CELL_DONE {tag} steps={len(out['step_rows'])} "
                          f"completed={completed}", flush=True)
        del rt, mon_stacked
        torch.cuda.empty_cache()
    (SNAPS / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(snap_manifest, indent=1) + "\n")
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
