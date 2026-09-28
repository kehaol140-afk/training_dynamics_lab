#!/usr/bin/env python3
"""GTTR-F1 实现闭环验证——三阶段（Target / State Sens / Hypergradient）。

Phase 1 (CPU)：目标 wrapper V=sum/count 的方向导数 vs FD。
Phase 2 (GPU)：phi 微扰 FD 状态差 vs 生产 S_theta/S_m/S_v（h=16/32/256）。
Phase 3 (GPU)：q^T S_theta vs FD(J)，保存 J+/J0/J- 绝对值。

使用修复后（D3+D4）生产代码；seed2030/start46/stream 202609250400；
phi0=(0,0)；delta∈{0.01, 0.005}（两坐标各一对）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_f1_implementation_closure_v01"
RUNS = OUT / "runs"
PROTOCOL = "FR_GTTR_F1_Implementation_Closure（用户指令，无独立文件）"

SEED, START, T_MAIN = 2030, 46, 256
READ_H = (16, 32, 256)
DELTAS = (0.01, 0.005)

RUN_ID = "F1-CLOSURE-1"
_C = {"traj_updates": 0, "train_fw": 0, "train_bw": 0, "hvp_dirs": 0,
      "hvp_micro": 0, "design_fw": 0, "design_evals": 0,
      "design_grad_evals": 0, "failures": 0}
_T0 = time.perf_counter()


def _log(stage, note="", **kw):
    rec = {"ts": time.time(), "run_id": RUN_ID, "stage": stage, "note": note,
           "wall_s": round(time.perf_counter() - _T0, 3), **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ===================================================================
# Phase 1：目标 wrapper V=sum/count 方向导数 vs FD（CPU 合成）
# ===================================================================

def phase1():
    """CPU：two-chunk 不同 valid count，方向导数 vs 中心差分。"""
    torch.manual_seed(42)
    DTYPE = torch.float64
    lin = torch.nn.Linear(6, 4, bias=False, dtype=DTYPE)
    # 两个 chunk，不同 valid token 数
    x1 = torch.randn(3, 6, dtype=DTYPE)
    # 3 个 token 位置、4 类 logits——直接用 3×4 形状
    y1 = torch.randint(0, 4, (3,))  # 3 个 token 的标签
    y1[0] = -100  # 1 个 ignore -> cnt1=2
    x2 = torch.randn(7, 6, dtype=DTYPE)
    y2 = torch.randint(0, 4, (7,))  # 7 个全有效 -> cnt2=7
    cnt1 = int((y1 != -100).sum())
    cnt2 = int((y2 != -100).sum())
    cnt = cnt1 + cnt2

    def V():
        """生产语义：全库 sum/count。"""
        lg1 = lin(x1)
        ce1 = torch.nn.functional.cross_entropy(
            lg1, y1, ignore_index=-100, reduction="none").sum()
        lg2 = lin(x2)
        ce2 = torch.nn.functional.cross_entropy(
            lg2, y2, reduction="none").sum()
        return (ce1 + ce2) / cnt

    # 随机方向
    torch.manual_seed(99)
    dirs = [torch.randn_like(lin.weight) for _ in range(3)]
    rows = []
    for i, d in enumerate(dirs):
        d = d / d.norm()
        # q = grad(V)（D3 修复后语义：grad(sum)/count）
        lin.zero_grad()
        v0 = V()
        v0.backward()
        q = lin.weight.grad.clone()
        # 方向导数
        dd = float((q * d).sum())
        # 中心差分
        for eps in (1e-3, 1e-4, 1e-5):
            with torch.no_grad():
                lin.weight.add_(d, alpha=eps)
            vp = V().item()
            with torch.no_grad():
                lin.weight.sub_(d, alpha=2 * eps)
            vm = V().item()
            with torch.no_grad():
                lin.weight.add_(d, alpha=eps)
            fd = (vp - vm) / (2 * eps)
            rows.append((f"dir{i}", eps, dd, fd, abs(dd - fd),
                         abs(dd - fd) / max(abs(dd), 1e-30)))
    with (OUT / "target_consistency.csv").open("w") as f:
        f.write("direction,eps,directional_derivative,FD,abs_diff,rel_diff\n")
        for r in rows:
            f.write(",".join(repr(x) for x in r) + "\n")
    max_rel = max(r[5] for r in rows if r[1] == 1e-4)
    print(f"Phase1: max rel_diff (eps=1e-4) = {max_rel:.2e} "
          f"{'PASS' if max_rel < 1e-6 else 'FAIL'}")
    return max_rel < 1e-6


# ===================================================================
# Phase 2+3：GPU rollouts（使用修复后生产代码）
# ===================================================================

def build_rt(seed, device):
    import f0_terminal_rule_run_v01 as base
    return base.build_rt(seed, device)


def rollout_fixed(rt, phi, T, sens=False, read_at=(), grad_at=()):
    """使用修复后 runner 的 REF32 rollout（D3+D4 已修）。"""
    import f0_terminal_rule_run_v01 as base
    import a0g_decision_utility_v01 as a0g
    from fr import f0_terminal_rule_v01 as f0
    from fr.stream_rng import seed_future_stream

    # D4 修复版 schedule：恒 T256 时钟
    w = f0.build_w(T_MAIN)
    psi = f0.build_psi(T_MAIN)
    eta, d_a, d_b = f0.rule_eta(phi[0], phi[1], w, psi)
    eta, d_a, d_b = eta[:T], d_a[:T], d_b[:T]

    base.load_bank(rt, SEED)
    theta = base._theta_state(rt)
    m, v = {}, {}
    for n_, p in rt["named"]:
        st = rt["optimizer"].state[p]
        m[n_] = st["exp_avg"].detach().clone()
        v[n_] = st["exp_avg_sq"].detach().clone()
    seed_future_stream(202609250400)
    stream = base.make_stream(rt, 202609250400, 2 * T)
    dirs = [("a", 0.0), ("b", 0.0)]
    sens_state = f0.init_sensitivity(theta, dirs) if sens else None
    names = [n_ for n_, _ in rt["named"]]
    params = [p for _, p in rt["named"]]
    V_reads = {}
    grads_out = {}
    state_snaps = {}
    clip_active = 0
    _C["traj_updates"] += T
    _C["train_fw"] += 2 * T
    _C["train_bw"] += 2 * T
    if sens:
        _C["hvp_dirs"] += 2 * T  # upper bound
        _C["hvp_micro"] += 2 * 2 * T
    read_set = set(read_at) | set(grad_at)
    _C["design_fw"] += 128 * len(read_set)
    _C["design_evals"] += len(read_set)
    for t in range(T):
        losses = []
        with a0g._math_sdpa_context(torch):
            for a_ in range(2):
                x, y = base.stack_batch_rt(rt, stream["batches"][2 * t + a_])
                res = rt["model"](x, labels=y)
                losses.append((res.loss + res.aux_loss) / 2)
            grads = torch.autograd.grad(losses[0] + losses[1], params,
                                        create_graph=True, allow_unused=True)
            g = {n_: (gr.detach() if gr is not None
                      else torch.zeros_like(theta[n_]))
                 for n_, gr in zip(names, grads)}
            if sens:
                hg = {}
                for d, _ in dirs:
                    if all(float(sens_state["S_theta"][n_][d]
                                 .abs().sum()) == 0.0 for n_ in names):
                        hg[d] = {n_: torch.zeros(theta[n_].shape,
                                                 dtype=torch.float64,
                                                 device=theta[n_].device)
                                 for n_ in names}
                        continue
                    dot = None
                    for n_, gr in zip(names, grads):
                        term = torch.sum(
                            gr * sens_state["S_theta"][n_][d].detach(),
                            dtype=torch.float64)
                        dot = term if dot is None else dot + term
                    vj = torch.autograd.grad(dot, params,
                                             retain_graph=True,
                                             allow_unused=True)
                    hg[d] = {n_: (z.detach().to(torch.float64) if z is not None
                                  else torch.zeros(theta[n_].shape,
                                                   dtype=torch.float64,
                                                   device=theta[n_].device))
                             for n_, z in zip(names, vj)}
                sens_state["Hg"] = hg
        sens_in = None
        if sens:
            sens_in = {"dirs": [("a", float(d_a[t])), ("b", float(d_b[t]))],
                       "S_theta": sens_state["S_theta"],
                       "S_m": sens_state["S_m"],
                       "S_v": sens_state["S_v"], "Hg": sens_state["Hg"]}
        theta, m, v, sens_out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, 0.9, 0.999, 1e-8, 0.01, float(eta[t]),
            START + t + 1, tau=1.0, sens=sens_in)
        if sens:
            sens_state = sens_out
        if info["clip_active"]:
            clip_active += 1
        base._apply_theta(rt, theta)
        h = t + 1
        if h in read_set:
            mean_v, cnt, q = base.eval_full_library(
                rt, base.design_tensors(rt), want_grad=(h in grad_at))
            V_reads[h] = mean_v
            state_snaps[h] = ({n_: tt.clone() for n_, tt in theta.items()},
                              {n_: tt.clone() for n_, tt in m.items()},
                              {n_: tt.clone() for n_, tt in v.items()})
            if h in grad_at:
                _C["design_grad_evals"] += 1
                ga = sum(float(torch.sum(
                    q[n_].to(torch.float64)
                    * sens_state["S_theta"][n_]["a"]).item())
                    for n_ in names)
                gb = sum(float(torch.sum(
                    q[n_].to(torch.float64)
                    * sens_state["S_theta"][n_]["b"]).item())
                    for n_ in names)
                grads_out[h] = (ga, gb)
    return {"V": V_reads, "grad": grads_out, "snaps": state_snaps,
            "clip_active": clip_active,
            "sens": sens_state if sens else None}


def phase23(gpu):
    device = f"cuda:{gpu}"
    rt = build_rt(SEED, device)
    results = {}
    # 1) SENS 基线 phi0=(0,0)
    print("Running SENS baseline phi0=(0,0)...", flush=True)
    r0 = rollout_fixed(rt, (0.0, 0.0), T_MAIN, sens=True,
                       read_at=READ_H, grad_at=READ_H)
    results["base"] = r0
    _log("sens_base_done", f"V256={r0['V'][256]:.6f} clip={r0['clip_active']}")

    # 2) FD 微扰（4 phi × 2 delta = 8 rollouts）
    for coord_idx, coord in enumerate(("a", "b")):
        for delta in DELTAS:
            for sign in (+1, -1):
                phi = [0.0, 0.0]
                phi[coord_idx] = sign * delta
                phi = tuple(phi)
                tag = f"{coord}{'+' if sign > 0 else '-'}{delta}"
                print(f"Running FD {tag}...", flush=True)
                r = rollout_fixed(rt, phi, T_MAIN, sens=False,
                                  read_at=READ_H)
                results[tag] = r
                _log("fd_done", f"{tag} V256={r['V'][256]:.6f}")

    # --- Phase 2: state sensitivity comparison ---
    ss_rows = []
    for h in READ_H:
        for coord in ("a", "b"):
            ci = 0 if coord == "a" else 1
            for delta in DELTAS:
                tp = results[f"{coord}+{delta}"]["snaps"][h]
                tm = results[f"{coord}-{delta}"]["snaps"][h]
                # FD sensitivity
                fd_theta = {n_: (tp[0][n_] - tm[0][n_]) / (2 * delta)
                            for n_ in tp[0]}
                fd_m = {n_: (tp[1][n_] - tm[1][n_]) / (2 * delta)
                        for n_ in tp[1]}
                fd_v = {n_: (tp[2][n_] - tm[2][n_]) / (2 * delta)
                        for n_ in tp[2]}
                # SENS
                s_theta = results["base"]["sens"]["S_theta"]
                s_m = results["base"]["sens"]["S_m"]
                s_v = results["base"]["sens"]["S_v"]
                # norms
                def norm64(d):
                    return math.sqrt(sum(float(
                        torch.sum(d[n_].to(torch.float64).square()).item())
                        for n_ in d))
                fd_th_norm = norm64(fd_theta)
                sens_th_norm = norm64({n_: s_theta[n_][coord] for n_ in fd_theta})
                diff_norm = norm64({n_: fd_theta[n_] - s_theta[n_][coord]
                                    .to(fd_theta[n_].dtype) for n_ in fd_theta})
                fd_m_norm = norm64(fd_m)
                sens_m_norm = norm64({n_: s_m[n_][coord] for n_ in fd_m})
                fd_v_norm = norm64(fd_v)
                sens_v_norm = norm64({n_: s_v[n_][coord] for n_ in fd_v})
                ss_rows.append((h, coord, delta, fd_th_norm, sens_th_norm,
                                diff_norm, diff_norm / max(fd_th_norm, 1e-30),
                                fd_m_norm, sens_m_norm,
                                fd_v_norm, sens_v_norm))
    with (OUT / "state_sensitivity.csv").open("w") as f:
        f.write("h,coord,delta,fd_S_theta_norm,sens_S_theta_norm,"
                "diff_norm,rel_diff,fd_S_m_norm,sens_S_m_norm,"
                "fd_S_v_norm,sens_S_v_norm\n")
        for r in ss_rows:
            f.write(",".join(repr(x) for x in r) + "\n")

    # --- Phase 3: hypergradient ---
    hg_rows = []
    for h in READ_H:
        for coord in ("a", "b"):
            ci = 0 if coord == "a" else 1
            sens_g = results["base"]["grad"][h][ci]
            for delta in DELTAS:
                vp = results[f"{coord}+{delta}"]["V"][h]
                vm = results[f"{coord}-{delta}"]["V"][h]
                v0 = results["base"]["V"][h]
                fd = (vp - vm) / (2 * delta)
                hg_rows.append((h, coord, delta, sens_g, fd, v0, vp, vm,
                                abs(sens_g - fd)))
    with (OUT / "hypergradient.csv").open("w") as f:
        f.write("h,coord,delta,SENS,FD,J_zero,J_plus,J_minus,abs_diff\n")
        for r in hg_rows:
            f.write(",".join(repr(x) for x in r) + "\n")

    # Summary
    summary = {}
    for h in READ_H:
        for coord in ("a", "b"):
            ci = 0 if coord == "a" else 1
            g = results["base"]["grad"][h][ci]
            for delta in DELTAS:
                fd = (results[f"{coord}+{delta}"]["V"][h]
                      - results[f"{coord}-{delta}"]["V"][h]) / (2 * delta)
                summary[f"h{h}_{coord}_d{delta}"] = {
                    "SENS": g, "FD": fd, "abs_diff": abs(g - fd)}
    (OUT / "closure_summary.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "source_sha_runner": _sha(PROJECT_ROOT / "research/scripts/"
                                  "f0_terminal_rule_run_v01.py"),
        "source_sha_kernel": _sha(PROJECT_ROOT / "research/src/fr/"
                                  "f0_terminal_rule_v01.py"),
        "dtype": "FP32 value path; FP32-graph HVP + FP64 dot/store/recursion",
        "backend": "MATH SDPA; TF32 off",
        "target": "V_design_monitor_FP32_MATH_sumcount_v1 "
                  "(4096 rows, sum/count, eval batch 32)",
        "schedule": "gttr_terminal_normalized_tilt_T256_v1 "
                    "(w=2*eta_base@gm, psi=2t/255-1, Z over 256)",
        "seed": SEED, "start": START, "stream": 202609250400,
        "clip_active_steps": results["base"]["clip_active"],
        "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    print("PHASE23_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase1", "phase23", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.phase in ("phase1", "all"):
        assert phase1()
    if args.phase in ("phase23", "all"):
        phase23(args.gpu)
    _log("done")


if __name__ == "__main__":
    main()
