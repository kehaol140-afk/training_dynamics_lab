#!/usr/bin/env python3
"""GTTR-F0 CAL-FIX 复测 runner —— smoke(8+8) + 10 条 T256 固定轨迹。

协议：FR_GTTR_F0_CALFIX_Revalidation_v1.md。修复版源码已冻结快照。
- 原 state/bank：seed2030/start46；原校准流（manifest 核对 202609250400）。
- 10 条：value 基线 (0,0)、修复 SENS 基线 (0,0)（a/b 双方向）、
  a/b 各 ±.01/±.005 共 8 条 FD；schedule 恒 T256 时钟。
- 每条在 h∈{16,32,256} 记 V_design 全库 sum/count/mean；SENS 基线三处
  同次评价取 q_H 并算 dot(q_H,S_H[j])；value 基线 h256 终点重复评价一次。
- value/SENS 基线在 h16/32/256 核对 theta/m/v（位级）与终态 RNG。
- 本轮 cap：updates 2816、fw 11000、bw 8000、HVP 方向 576、V_design 36、
  confirm/final 0；同时与 F0 原总 cap 联合约束（旧账不重置）。
- 精度命名如实：FP32 值路径；FP32 图内 HVP＋FP64 点积/存储/递推。
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_f0_calfix_revalidation_v01"
RUNS = OUT / "runs"
CKPT = OUT / "checkpoints"
F0_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_f0_terminal_rule_solver_v01"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_F0_CALFIX_Revalidation_v1.md"

SEED, START, T_MAIN = 2030, 46, 256
READ_H = (16, 32, 256)

CAP_UPDATES = 2816
CAP_FW = 11000
CAP_BW = 8000
CAP_HVP = 576
CAP_DESIGN = 36
F0_CAPS = {"traj_updates": 34000, "total_fw": 110000, "total_bw": 90000,
           "hvp_dirs": 7000, "design_evals": 250, "design_grad_evals": 24}

B1_, B2_, EPS_, WD_, TAU_ = 0.9, 0.999, 1e-8, 0.01, 1.0

RUN_ID = "CALFIX-1"
_C = {"traj_updates": 0, "train_fw": 0, "train_bw": 0, "hvp_dirs": 0,
      "hvp_micro": 0, "design_fw": 0, "design_evals": 0,
      "design_grad_evals": 0, "confirm_evals": 0, "v_final_calls": 0,
      "failures": 0}
_T0 = time.perf_counter()


def _total_fw():
    return _C["train_fw"] + _C["design_fw"]


def _total_bw():
    return _C["train_bw"] + _C["hvp_micro"] + _C["design_grad_evals"] * 128


def _log(stage, note="", **kw):
    rec = {"ts": time.time(), "run_id": RUN_ID, "stage": stage, "note": note,
           "wall_s": round(time.perf_counter() - _T0, 3),
           "total_fw": _total_fw(), "total_bw": _total_bw(), **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _f0_base():
    last = None
    p = F0_OUT / "resource_log.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last or {}


def _check_caps(up_up=0, up_fw=0, up_bw=0, up_dirs=0, up_design=0,
                up_grad=0):
    if _C["traj_updates"] + up_up > CAP_UPDATES:
        raise SystemExit(f"BUDGET_BLOCKED updates {CAP_UPDATES}")
    if _total_fw() + up_fw > CAP_FW:
        raise SystemExit(f"BUDGET_BLOCKED fw {CAP_FW}")
    if _total_bw() + up_bw > CAP_BW:
        raise SystemExit(f"BUDGET_BLOCKED bw {CAP_BW}")
    if _C["hvp_dirs"] + up_dirs > CAP_HVP:
        raise SystemExit(f"BUDGET_BLOCKED hvp {CAP_HVP}")
    if _C["design_evals"] + up_design > CAP_DESIGN:
        raise SystemExit(f"BUDGET_BLOCKED design {CAP_DESIGN}")
    base = _f0_base()
    if base.get("traj_updates", 0) + _C["traj_updates"] + up_up > \
            F0_CAPS["traj_updates"]:
        raise SystemExit("BUDGET_BLOCKED F0 total traj")
    if base.get("total_fw", 0) + _total_fw() + up_fw > F0_CAPS["total_fw"]:
        raise SystemExit("BUDGET_BLOCKED F0 total fw")
    if base.get("total_bw", 0) + _total_bw() + up_bw > F0_CAPS["total_bw"]:
        raise SystemExit("BUDGET_BLOCKED F0 total bw")
    if base.get("hvp_dirs", 0) + _C["hvp_dirs"] + up_dirs > \
            F0_CAPS["hvp_dirs"]:
        raise SystemExit("BUDGET_BLOCKED F0 total hvp")
    if base.get("design_evals", 0) + _C["design_evals"] + up_design > \
            F0_CAPS["design_evals"]:
        raise SystemExit("BUDGET_BLOCKED F0 total design")
    _ = up_grad


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 运行时（复用 F0 冻结接口）
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    import f0_terminal_rule_run_v01 as base
    rt = base.build_rt(seed, device)
    return rt


def load_bank(rt, seed):
    import f0_terminal_rule_run_v01 as base
    return base.load_bank(rt, seed)


def make_stream(rt, stream_seed, n_micro):
    import f0_terminal_rule_run_v01 as base
    return base.make_stream(rt, stream_seed, n_micro)


def stack_batch_rt(rt, idx):
    import f0_terminal_rule_run_v01 as base
    return base.stack_batch_rt(rt, idx)


def eval_design(rt, want_grad=False):
    import f0_terminal_rule_run_v01 as base
    return base.eval_full_library(rt, base.design_tensors(rt),
                                  want_grad=want_grad)


def _apply_theta(rt, theta):
    import f0_terminal_rule_run_v01 as base
    base._apply_theta(rt, theta)


def _theta_state(rt):
    import f0_terminal_rule_run_v01 as base
    return base._theta_state(rt)


def stream_id():
    cfg = json.loads((F0_OUT / "config.json").read_text())
    sid = int(cfg["streams"]["fit"]["2030"])
    assert sid == 202609250400, f"manifest stream {sid} != 202609250400"
    return sid


# ---------------------------------------------------------------------------
# rollout（与 F0 ref32 同算术；read_at/grad_at/norms）
# ---------------------------------------------------------------------------

def rollout(rt, phi, T, sens=False, read_at=(), grad_at=(), tag=""):
    """REF32 rollout。read_at: h 集合读 V_design；grad_at: 读+q+dot(q,S)。

    返回 {"V": {h: (sum,count,mean)}, "grad": {h: (ga,gb)},
    "snaps": {h: (theta,m,v)}, "rng": rng_state, "norms": [per-step rows],
    "clip_active_steps": n}。
    """
    import torch
    import a0g_decision_utility_v01 as a0g
    from fr import f0_terminal_rule_v01 as f0
    from fr.stream_rng import seed_future_stream
    from fr.rng import capture_rng_state

    w = f0.build_w(T_MAIN)
    psi = f0.build_psi(T_MAIN)
    eta, d_a, d_b = f0.rule_eta(phi[0], phi[1], w, psi)
    load_bank(rt, SEED)
    theta = _theta_state(rt)
    m, v = {}, {}
    for n_, p in rt["named"]:
        st = rt["optimizer"].state[p]
        m[n_] = st["exp_avg"].detach().clone()
        v[n_] = st["exp_avg_sq"].detach().clone()
    seed_future_stream(stream_id())
    stream = make_stream(rt, stream_id(), 2 * T)
    dirs = [("a", 0.0), ("b", 0.0)]
    sens_state = f0.init_sensitivity(theta, dirs) if sens else None
    names = [n_ for n_, _ in rt["named"]]
    params = [p for _, p in rt["named"]]
    read_at = set(read_at)
    grad_at = set(grad_at)
    V = {}
    grads_out = {}
    snaps = {}
    norms = []
    clip_active = 0
    _check_caps(up_up=T, up_fw=2 * T + 128 * len(read_at | grad_at),
                up_bw=2 * T, up_dirs=(2 * T if sens else 0),
                up_design=len(read_at | grad_at))
    for t in range(T):
        losses = []
        with a0g._math_sdpa_context(torch):
            for a_ in range(2):
                x, y = stack_batch_rt(rt, stream["batches"][2 * t + a_])
                res = rt["model"](x, labels=y)
                losses.append((res.loss + res.aux_loss) / 2)
            grads = torch.autograd.grad(losses[0] + losses[1], params,
                                        create_graph=True, allow_unused=True)
            g = {n_: (gr.detach() if gr is not None
                      else torch.zeros_like(theta[n_]))
                 for n_, gr in zip(names, grads)}
            if sens:
                hg = {}
                n_active_dirs = 0
                for d, _ in dirs:
                    zero = all(float(sens_state["S_theta"][n_][d]
                                     .abs().sum()) == 0.0 for n_ in names)
                    if zero:
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
                    hg[d] = {n_: (z.detach().to(torch.float64) if z
                                  is not None else
                                  torch.zeros(theta[n_].shape,
                                              dtype=torch.float64,
                                              device=theta[n_].device))
                             for n_, z in zip(names, vj)}
                    n_active_dirs += 1
                sens_state["Hg"] = hg
                _C["hvp_dirs"] += n_active_dirs
                _C["hvp_micro"] += 2 * n_active_dirs
        _C["train_fw"] += 2
        _C["train_bw"] += 2
        sens_in = None
        if sens:
            sens_in = {"dirs": [("a", float(d_a[t])), ("b", float(d_b[t]))],
                       "S_theta": sens_state["S_theta"],
                       "S_m": sens_state["S_m"],
                       "S_v": sens_state["S_v"], "Hg": sens_state["Hg"]}
        theta, m, v, sens_out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, B1_, B2_, EPS_, WD_, float(eta[t]),
            START + t + 1, tau=TAU_, sens=sens_in)
        if sens:
            sens_state = sens_out
            row = {"t": t + 1, "clip_active": bool(info["clip_active"])}
            for d, _ in dirs:
                for key in ("S_theta", "S_m", "S_v"):
                    s2 = sum(float(torch.sum(
                        sens_state[key][n_][d]
                        .to(torch.float64).square()).item())
                        for n_ in names)
                    row[f"{key}_{d}_l2"] = math.sqrt(s2)
                    if not math.isfinite(s2):
                        _C["failures"] += 1
                        raise RuntimeError(
                            f"non-finite sensitivity norm t={t+1} {key}/{d}")
            norms.append(row)
        if info["clip_active"]:
            clip_active += 1
        _apply_theta(rt, theta)
        _C["traj_updates"] += 1
        h = t + 1
        if h in read_at or h in grad_at:
            want_g = h in grad_at
            mean_v, cnt, q = eval_design(rt, want_grad=want_g)
            _C["design_fw"] += 128
            _C["design_evals"] += 1
            V[h] = (mean_v, cnt)
            snaps[h] = ({n_: tt.clone() for n_, tt in theta.items()},
                        {n_: tt.clone() for n_, tt in m.items()},
                        {n_: tt.clone() for n_, tt in v.items()})
            if want_g:
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
    return {"V": V, "grad": grads_out, "snaps": snaps,
            "rng": capture_rng_state(), "norms": norms,
            "clip_active_steps": clip_active,
            "final": (theta, m, v)}


# ---------------------------------------------------------------------------
# smoke（8+8；无目标评价）
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    import torch
    rt = build_rt(SEED, f"cuda:{gpu}")
    r_val = rollout(rt, (0.0, 0.0), 8, sens=False, tag="smoke_val")
    sig_v = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    rng_v = r_val["rng"]
    r_sens = rollout(rt, (0.0, 0.0), 8, sens=True, tag="smoke_sens")
    sig_s = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    rng_s = r_sens["rng"]
    theta_eq = all(torch.equal(sig_v[n], sig_s[n]) for n in sig_v)
    mv_eq = all(torch.equal(r_val["final"][1][n], r_sens["final"][1][n])
                and torch.equal(r_val["final"][2][n],
                                r_sens["final"][2][n]) for n in sig_v)
    rng_eq = torch.equal(rng_v["torch_cpu"], rng_s["torch_cpu"])
    checks = {"theta_bitwise": bool(theta_eq), "mv_bitwise": bool(mv_eq),
              "rng_equal": bool(rng_eq),
              "hvp_dirs_this_run": _C["hvp_dirs"],
              "design_evals_this_run": _C["design_evals"],
              "updates": _C["traj_updates"],
              "verdict": "PASS" if (theta_eq and mv_eq and rng_eq
                                    and _C["design_evals"] == 0
                                    and _C["traj_updates"] <= 16
                                    and _C["hvp_dirs"] <= 14) else "FAIL"}
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1))
    _log("smoke_done", checks["verdict"])
    print(json.dumps(checks, indent=1))
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main：10 条轨迹 + 重复评价 + 身份核对 + 快照保存
# ---------------------------------------------------------------------------

def run_main(gpu):
    import torch
    rt = build_rt(SEED, f"cuda:{gpu}")
    CKPT.mkdir(parents=True, exist_ok=True)
    tv_rows = []          # terminal_values.csv
    norms_path = OUT / "sensitivity_norms.csv"
    norms_path.write_text("t,clip_active,S_theta_a_l2,S_theta_b_l2,"
                          "S_m_a_l2,S_m_b_l2,S_v_a_l2,S_v_b_l2\n")

    def record(role, phi, h, mean_v, cnt, source, extra=""):
        tv_rows.append((role, f"({phi[0]:g},{phi[1]:g})", h,
                        repr(mean_v), cnt, mean_v / max(cnt, 1), source,
                        extra))

    # 1) value 基线
    rv = rollout(rt, (0.0, 0.0), T_MAIN, sens=False, read_at=READ_H,
                 tag="base_value")
    for h in READ_H:
        mean_v, cnt = rv["V"][h]
        record("BASE_VALUE", (0.0, 0.0), h, mean_v, cnt, "VALUE_ROLLOUT")
    torch.save({"theta": rv["final"][0], "m": rv["final"][1],
                "v": rv["final"][2], "role": "BASE_VALUE", "phi": [0, 0],
                "T": T_MAIN, "stream": stream_id()},
               CKPT / "BASE_VALUE_h256.pt")
    _log("traj_done", "BASE_VALUE")

    # 2) 修复 SENS 基线
    rs = rollout(rt, (0.0, 0.0), T_MAIN, sens=True, grad_at=READ_H,
                 tag="base_sens")
    for h in READ_H:
        mean_v, cnt = rs["V"][h]
        record("BASE_SENS", (0.0, 0.0), h, mean_v, cnt, "SENS_ROLLOUT",
               f"grad=({rs['grad'][h][0]:.9e},{rs['grad'][h][1]:.9e})")
    torch.save({"theta": rs["final"][0], "m": rs["final"][1],
                "v": rs["final"][2], "role": "BASE_SENS", "phi": [0, 0],
                "T": T_MAIN, "stream": stream_id(),
                "grads": {str(h): rs["grad"][h] for h in READ_H}},
               CKPT / "BASE_SENS_h256.pt")
    with norms_path.open("a") as f:
        for row in rs["norms"]:
            f.write(",".join(str(row[k]) for k in (
                "t", "clip_active", "S_theta_a_l2", "S_theta_b_l2",
                "S_m_a_l2", "S_m_b_l2", "S_v_a_l2", "S_v_b_l2")) + "\n")
    _log("traj_done", f"BASE_SENS clip_active={rs['clip_active_steps']}")

    # 身份核对（h16/32/256：theta/m/v 位级；终态 RNG）
    id_rows = []
    ok_all = True
    for h in READ_H:
        th_eq = all(torch.equal(rv["snaps"][h][0][n], rs["snaps"][h][0][n])
                    for n in rv["snaps"][h][0])
        m_eq = all(torch.equal(rv["snaps"][h][1][n], rs["snaps"][h][1][n])
                   for n in rv["snaps"][h][0])
        v_eq = all(torch.equal(rv["snaps"][h][2][n], rs["snaps"][h][2][n])
                   for n in rv["snaps"][h][0])
        ok = bool(th_eq and m_eq and v_eq)
        ok_all = ok_all and ok
        id_rows.append((h, ok, th_eq, m_eq, v_eq))
    rng_eq = torch.equal(rv["rng"]["torch_cpu"], rs["rng"]["torch_cpu"])
    with (OUT / "value_identity.csv").open("w") as f:
        f.write("check_h,identical,theta_bitwise,m_bitwise,v_bitwise\n")
        for r in id_rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.write(f"final_rng,{rng_eq},,,\n")
    _log("identity_checked", f"ok={ok_all} rng={rng_eq}")

    # 3) 8 条 FD
    fd_phis = [(0.01, 0.0), (-0.01, 0.0), (0.005, 0.0), (-0.005, 0.0),
               (0.0, 0.01), (0.0, -0.01), (0.0, 0.005), (0.0, -0.005)]
    fd_results = {}
    for phi in fd_phis:
        r = rollout(rt, phi, T_MAIN, sens=False, read_at=READ_H,
                    tag=f"fd_{phi[0]:g}_{phi[1]:g}")
        fd_results[phi] = r["V"]
        for h in READ_H:
            mean_v, cnt = r["V"][h]
            record("FD", phi, h, mean_v, cnt, "VALUE_ROLLOUT")
        torch.save({"theta": r["final"][0], "m": r["final"][1],
                    "v": r["final"][2], "role": "FD", "phi": list(phi),
                    "T": T_MAIN, "stream": stream_id()},
                   CKPT / f"FD_a{phi[0]:g}_b{phi[1]:g}_h256.pt")
        _log("traj_done", f"FD ({phi[0]:g},{phi[1]:g})")

    # 4) h256 基线重复评价
    blob = torch.load(CKPT / "BASE_VALUE_h256.pt", map_location="cpu",
                      weights_only=False)
    _apply_theta(rt, blob["theta"])
    mean_r, cnt_r, _ = eval_design(rt)
    _C["design_fw"] += 128
    _C["design_evals"] += 1
    record("BASE_REPEAT", (0.0, 0.0), 256, mean_r, cnt_r, "REPEAT_EVAL")
    _log("repeat_eval_done", f"V={mean_r}")

    with (OUT / "terminal_values.csv").open("w") as f:
        f.write("role,phi,h,sum,count,mean,source,extra\n")
        for r in tv_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # 5) 门计算与表
    analyze(rv, rs, fd_results, mean_r, blob, rt, ok_all and rng_eq)
    _log("main_done", "all trajectories complete")
    print("MAIN_DONE")


def analyze(rv, rs, fd_results, repeat_v, base_blob, rt, identity_ok):
    V = {"base": rv["V"], "sens": rs["V"]}
    eps_v = max(abs(repeat_v - V["base"][256][0]), 1e-7)
    floor = max(5e-4, 4 * eps_v / 0.005)

    def fd(coord, eps, h):
        pp = (eps if coord == "a" else 0.0, eps if coord == "b" else 0.0)
        pm = (-eps if coord == "a" else 0.0, -eps if coord == "b" else 0.0)
        return (fd_results[pp][h][0] - fd_results[pm][h][0]) / (2 * eps)

    table = {}
    sens_rows = []
    pair_rows = []
    for h in READ_H:
        for coord in ("a", "b"):
            fd01 = fd(coord, 0.01, h)
            fd005 = fd(coord, 0.005, h)
            sens = rs["grad"][h][0] if coord == "a" else rs["grad"][h][1]
            fd_diff = abs(fd01 - fd005)
            tol = floor + 0.20 * max(abs(sens), abs(fd01), abs(fd005))
            tol_fd = floor + 0.20 * max(abs(fd01), abs(fd005))
            low = (abs(sens) < floor and abs(fd01) < floor
                   and abs(fd005) < floor)
            table[(h, coord)] = {
                "SENS": sens, "FD01": fd01, "FD005": fd005,
                "fd_diff": fd_diff, "sens_diff": abs(sens - fd005),
                "tol": tol, "tol_fd": tol_fd, "low": low}
            sens_rows.append((h, coord, repr(sens), repr(fd01),
                              repr(fd005), repr(fd_diff),
                              repr(abs(sens - fd01)), repr(tol),
                              "LOW_SIGNAL" if low else "SIGNAL"))
            pair_rows.append((h, coord, repr(fd_diff), repr(tol_fd),
                              "FD_PAIR_OK" if fd_diff <= tol_fd
                              else "FD_PAIR_FAIL"))
    with (OUT / "sensitivity_check.csv").open("w") as f:
        f.write("h,coord,SENS,FD_eps0.01,FD_eps0.005,|FD01-FD005|,"
                "|SENS-FD01|,tol_original,signal_class\n")
        for r in sens_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with (OUT / "fd_pair_consistency.csv").open("w") as f:
        f.write("h,coord,fd_diff,tol_FD,verdict\n")
        for r in pair_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # 主门（h256）
    gate = {}
    for coord in ("a", "b"):
        d = table[(256, coord)]
        pair_ok = d["fd_diff"] <= d["tol_fd"]
        if not identity_ok:
            verdict = "REFERENCE_PATH_MISMATCH"
        elif d["low"]:
            verdict = "LOW_SIGNAL"
        elif d["fd_diff"] <= d["tol"] and d["sens_diff"] <= d["tol"] \
                and pair_ok:
            verdict = "CALIBRATED"
        elif not pair_ok or d["fd_diff"] > d["tol"]:
            verdict = "FD_INCONSISTENT"
        else:
            verdict = "SENS_MISMATCH"
        gate[coord] = {k: d[k] for k in
                       ("SENS", "FD01", "FD005", "fd_diff", "sens_diff",
                        "tol", "tol_fd", "low")}
        gate[coord]["verdict"] = verdict
    (OUT / "gate_h256.json").write_text(json.dumps(
        {"identity_ok": identity_ok, "eps_V": eps_v, "floor": floor,
         "gate": gate, "clip_active_steps": rs["clip_active_steps"],
         "table_all_h": {f"{h}_{c}": table[(h, c)]
                         for (h, c) in table}}, indent=1))
    print(json.dumps({"gate": {c: {k: gate[c][k] for k in
                                   ("SENS", "FD01", "FD005", "verdict")}
                              for c in gate}, "eps_V": eps_v}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "main", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        assert run_smoke(args.gpu)
    elif args.phase == "main":
        run_main(args.gpu)
    else:
        assert run_smoke(args.gpu)
        run_main(args.gpu)


if __name__ == "__main__":
    main()
