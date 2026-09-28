#!/usr/bin/env python3
"""F0-IC v1：薄适配器——真实生产 SENS 内核 × 附件独立参考九组夹具（CPU）。

- 直接 import 生产 fr.f0_terminal_rule_v01（记录 __file__ 与 SHA），
  每步调用生产 f0_adamw_sens_step（唯一被测内核；适配器只做打包/解包
  与 dtype 转换，不重写公式）。
- 参考侧：附件 verify_reference 的数据/初值/解析 g/H/显式递推（原样复制
  公式，作为独立参考，不调用生产函数）。
- 九组：tau∈{.02,.25,100}×h∈{16,32,256}；逐比较 theta/m/v/S_theta/S_m/S_v
  （两方向），记录首次分歧 t；终点 grad=dot(q,S) 对照。
- 容差（FP64 同对象）：err ≤ 1e-10 + 1e-8*max(norm_prod, norm_ref)。
- 附加：生产规则导数（rule_eta on build_w）vs autograd jacobian；
  D3/D4 缺陷 fixture 的 before 取证（--defects 模式）。
运行：CUDA_VISIBLE_DEVICES="" python f0_ic_adapter_v01.py [--defects]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_f0_implementation_consistency_v01"
DTYPE = torch.float64
T, D = 256, 4
B1, B2, WD = 0.9, 0.999, 0.01
EPS_ADAM, EPS_CLIP, RIDGE = 1e-8, 1e-6, 0.015
N0 = 46

from fr import f0_terminal_rule_v01 as prod  # noqa: E402

MODULE_FILE = prod.__file__
MODULE_SHA = hashlib.sha256(Path(MODULE_FILE).read_bytes()).hexdigest()


# ---------------- 参考侧（附件公式原样；不调用生产内核） ----------------

def data():
    gen = torch.Generator().manual_seed(250925)
    x = torch.randn(T, 12, D, dtype=DTYPE, generator=gen)
    x = x @ torch.tensor([[1., .3, .2, 0.], [0., .7, -.2, .3],
                          [.1, 0., 1.1, .2], [.2, -.2, .3, .9]], dtype=DTYPE)
    amp = torch.tensor([0.15, 0.7, 2.2, 1.0], dtype=DTYPE)
    x *= amp[torch.arange(T) % 4, None, None]
    teacher = torch.tensor([.7, -.4, .3, -.6], dtype=DTYPE)
    probs = torch.sigmoid(x @ teacher)
    y = (torch.rand(T, 12, dtype=DTYPE, generator=gen) < probs).to(DTYPE)
    qx = torch.randn(25, D, dtype=DTYPE, generator=gen)
    qy = (torch.rand(25, dtype=DTYPE, generator=gen)
          < torch.sigmoid(qx @ teacher)).to(DTYPE)
    return x, y, qx, qy


def initial():
    return (torch.tensor([.45, -.35, .18, .3], dtype=DTYPE),
            torch.tensor([.008, -.004, .006, -.003], dtype=DTYPE),
            torch.tensor([.025, .032, .02, .028], dtype=DTYPE))


def schedule_ref(phi, w):
    psi = torch.linspace(-1., 1., T, dtype=DTYPE)
    weights = w * torch.exp(phi[1] * psi)
    z = weights.sum() / w.sum()
    return w * torch.exp(phi[0] + phi[1] * psi) / z


def schedule_ref_grad(phi, w):
    psi = torch.linspace(-1., 1., T, dtype=DTYPE)
    eta = schedule_ref(phi, w)
    weights = w * torch.exp(phi[1] * psi)
    mu = (weights * psi).sum() / weights.sum()
    return eta, torch.stack([eta, eta * (psi - mu)], dim=1)


def explicit_ref(phi, w, tau, horizon, x, y, qx, qy):
    eta, deta = schedule_ref_grad(phi, w)
    th, m, v = initial()
    st = torch.zeros(D, 2, dtype=DTYPE)
    sm = torch.zeros_like(st)
    sv = torch.zeros_like(st)
    for t in range(horizon):
        prob = torch.sigmoid(x[t] @ th)
        g = x[t].T @ (prob - y[t]) / len(y[t]) + RIDGE * th
        hessian = x[t].T @ ((prob * (1 - prob))[:, None] * x[t]) / len(y[t]) \
            + RIDGE * torch.eye(D, dtype=DTYPE)
        hg = hessian @ st
        norm = torch.linalg.vector_norm(g)
        coef = min(1., float(tau / (norm + EPS_CLIP)))
        c = coef * g
        if coef < 1.:
            global_dot = (g[:, None] * hg).sum(dim=0)
            dc = coef * hg - tau * g[:, None] * global_dot[None, :] \
                / (norm * (norm + EPS_CLIP) ** 2)
        else:
            dc = hg
        m_next = B1 * m + (1 - B1) * c
        v_next = B2 * v + (1 - B2) * c.square()
        sm_next = B1 * sm + (1 - B1) * dc
        sv_next = B2 * sv + 2 * (1 - B2) * c[:, None] * dc
        bc1 = 1 - B1 ** (N0 + t + 1)
        bc2 = 1 - B2 ** (N0 + t + 1)
        mh, vh = m_next / bc1, v_next / bc2
        dmh, dvh = sm_next / bc1, sv_next / bc2
        root = torch.sqrt(vh)
        p = 1 / (root + EPS_ADAM)
        dp = -dvh / (2 * root[:, None] * (root[:, None] + EPS_ADAM) ** 2)
        r = p * mh + WD * th
        st_next = st - r[:, None] * deta[t][None, :] - eta[t] * (
            p[:, None] * dmh + mh[:, None] * dp + WD * st)
        th_next = th - eta[t] * r
        th, m, v, st, sm, sv = (th_next, m_next, v_next, st_next, sm_next,
                                sv_next)
    query_prob = torch.sigmoid(qx @ th)
    q = qx.T @ (query_prob - qy) / len(qy)
    return {"theta": th, "m": m, "v": v, "st": st, "sm": sm, "sv": sv,
            "grad": st.T @ q}


# ---------------- 生产内核适配（仅打包/解包 + dtype） ----------------

BLOCKS = {"p0": slice(0, 2), "p1": slice(2, 4)}


def to_blocks(vec4):
    return {k: vec4[s].clone() for k, s in BLOCKS.items()}


def from_blocks(dct):
    return torch.cat([dct["p0"], dct["p1"]])


def production_chain(phi, w, tau, horizon, x, y, qx, qy):
    eta, deta = schedule_ref_grad(phi, w)
    th4, m4, v4 = initial()
    theta = to_blocks(th4)
    m = to_blocks(m4)
    v = to_blocks(v4)
    dirs = [("a", 0.0), ("b", 0.0)]
    sens = prod.init_sensitivity(theta, dirs)
    first_div = None
    max_diff = {"theta": 0.0, "m": 0.0, "v": 0.0,
                "S_theta": 0.0, "S_m": 0.0, "S_v": 0.0}
    for t in range(horizon):
        # 参考侧同步推进（逐步对照，不止最终标量）
        ref = explicit_ref(phi, w, tau, t + 1, x, y, qx, qy)
        prob = torch.sigmoid(x[t] @ from_blocks(theta))
        g4 = x[t].T @ (prob - y[t]) / len(y[t]) + RIDGE * from_blocks(theta)
        hess = x[t].T @ ((prob * (1 - prob))[:, None] * x[t]) / len(y[t]) \
            + RIDGE * torch.eye(D, dtype=DTYPE)
        st4 = torch.stack([from_blocks({k: sens["S_theta"][k][d]
                                        for k in BLOCKS})
                           for d, _ in dirs], dim=1)
        hg4 = hess @ st4
        hg_blocks = [to_blocks(hg4[:, j]) for j in range(2)]
        sens["Hg"] = {d: hg_blocks[j] for j, (d, _) in enumerate(dirs)}
        sens["dirs"] = [("a", float(deta[t][0])), ("b", float(deta[t][1]))]
        theta, m, v, sens_out, info = prod.f0_adamw_sens_step(
            theta, m, v, to_blocks(g4), B1, B2, EPS_ADAM, WD,
            float(eta[t]), N0 + t + 1, tau=tau, sens=sens)
        sens = {"dirs": dirs, "S_theta": sens_out["S_theta"],
                "S_m": sens_out["S_m"], "S_v": sens_out["S_v"], "Hg": None}
        # 逐步比较
        pairs = {
            "theta": (from_blocks(theta), ref["theta"]),
            "m": (from_blocks(m), ref["m"]),
            "v": (from_blocks(v), ref["v"]),
            "S_theta": (torch.stack([from_blocks(
                {k: sens["S_theta"][k][d] for k in BLOCKS})
                for d, _ in dirs], dim=1), ref["st"]),
            "S_m": (torch.stack([from_blocks(
                {k: sens["S_m"][k][d] for k in BLOCKS})
                for d, _ in dirs], dim=1), ref["sm"]),
            "S_v": (torch.stack([from_blocks(
                {k: sens["S_v"][k][d] for k in BLOCKS})
                for d, _ in dirs], dim=1), ref["sv"]),
        }
        for key, (a, b) in pairs.items():
            err = float((a - b).abs().max())
            tol = 1e-10 + 1e-8 * max(float(a.norm()), float(b.norm()))
            max_diff[key] = max(max_diff[key], err)
            if err > tol and first_div is None:
                first_div = (t + 1, key, err, tol)
    qth = from_blocks(theta)
    query_prob = torch.sigmoid(qx @ qth)
    q = qx.T @ (query_prob - qy) / len(qy)
    st_final = torch.stack([from_blocks({k: sens["S_theta"][k][d]
                                         for k in BLOCKS})
                            for d, _ in dirs], dim=1)
    grad = st_final.T @ q
    ref_full = explicit_ref(phi, w, tau, horizon, x, y, qx, qy)
    grad_err = float((grad - ref_full["grad"]).abs().max())
    return {"max_diff": max_diff, "first_divergence": first_div,
            "grad": grad.tolist(), "grad_err": grad_err,
            "theta_err": float((qth - ref_full["theta"]).abs().max())}


def rule_derivative_check():
    w = prod.build_w(T)
    psi = prod.build_psi(T)
    phi = (0.11, -0.23)
    eta, d_a, d_b = prod.rule_eta(phi[0], phi[1], w, psi)
    wa = torch.tensor(w, dtype=DTYPE)
    pa = torch.tensor(phi, dtype=DTYPE, requires_grad=True)

    def f(p):
        psii = torch.tensor(psi, dtype=DTYPE)
        wgt = wa * torch.exp(p[1] * psii)
        z = wgt.sum() / wa.sum()
        return wa * torch.exp(p[0] + p[1] * psii) / z

    jac = torch.autograd.functional.jacobian(f, pa)
    err = max(float((torch.tensor(d_a) - jac[:, 0]).abs().max()),
              float((torch.tensor(d_b) - jac[:, 1]).abs().max()))
    sum_rel = abs(float(np_sum(eta) - math_exp(phi[0]) * np_sum(w))
                  / np_sum(w))
    return {"autograd_jac_err": err, "sum_eta_rel_err": sum_rel}


def np_sum(a):
    return float(a.sum())


def math_exp(x):
    import math
    return math.exp(x)


# ---------------- D3/D4 fixture（before 取证） ----------------

def defect_fixtures():
    import math
    ev = {}
    # D4：Z 按 T=16 vs T=256 切片（ref32/native rollout 现行行为）
    w16 = prod.build_w(16)
    psi16 = prod.build_psi(16)
    w256 = prod.build_w(256)
    psi256 = prod.build_psi(256)
    a0, b0 = 0.1, -0.2
    eta16, _, _ = prod.rule_eta(a0, b0, w16, psi16)
    eta256, _, _ = prod.rule_eta(a0, b0, w256, psi256)
    ev["D4"] = {
        "expected": "eta[:16] 应取自 T256 全规则（Z 对 256 项归一）",
        "failing_input": f"phi=({a0},{b0}), rollout T=16（现行 "
                         "ref32/native rollout 用 build_w(T)）",
        "before_output": {
            "eta16_first3": eta16[:3].tolist(),
            "eta256_first3": eta256[:3].tolist(),
            "max_abs_diff_first16": float(np.abs(eta16 - eta256[:16]).max()),
            "sum_eta16": float(eta16.sum()),
            "sum_eta256_prefix": float(eta256[:16].sum())},
        "independent_expected": "eta[:16]==eta256[:16]（差应=0）"}
    # D3：q 未除以 count（eval_full_library 现行行为）
    torch.manual_seed(7)
    lin = torch.nn.Linear(5, 3, bias=False)   # 3×5 权重作为被测参数
    x1 = torch.randn(4, 5)
    y1 = torch.randint(0, 3, (4,))
    x2 = torch.randn(6, 5)
    y2 = torch.randint(0, 3, (6,))
    # 两 chunk 不同有效 token 数：人为制造 ignore（用 y=-100 模拟掩码语义）
    logits = lin(x1)
    ce1 = torch.nn.functional.cross_entropy(
        logits, torch.where(y1 >= 0, y1, -100), reduction="none")
    s1 = logits.shape
    cnt1 = int((y1 >= 0).sum()) * 1  # 简化：全有效 4
    lg2 = lin(x2)
    ce2 = torch.nn.functional.cross_entropy(
        lg2, y2, reduction="none")
    cnt2 = 6
    total = ce1.sum() + ce2.sum()
    cnt = cnt1 + cnt2
    mean = total / cnt
    # 生产 pattern 先取（retain），再取独立 mean 梯度
    lin.zero_grad()
    (ce1.sum()).backward(retain_graph=True)
    (ce2.sum()).backward(retain_graph=True)
    q_prod_pattern = lin.weight.grad.clone()
    q_mean, = torch.autograd.grad(mean, lin.weight)
    ratio = float((q_prod_pattern / q_mean).flatten()[0])
    ev["D3"] = {
        "expected": "q = grad(V), V=sum/count -> q=grad(sum)/count",
        "failing_input": f"两 chunk 有效 token {cnt1}/{cnt2}（不同），"
                         f"count={cnt}",
        "before_output": {
            "pattern_q_vs_mean_grad_max_ratio": float(
                (q_prod_pattern / q_mean).abs().max()),
            "first_element_ratio": ratio,
            "count": cnt},
        "independent_expected": {"ratio_should_be": 1.0,
                                 "pattern_is_count_times": cnt}}
    return ev


import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defects", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.defects:
        ev = defect_fixtures()
        (OUT / "defect_fixtures_before.json").write_text(
            json.dumps(ev, indent=2))
        print(json.dumps(ev, indent=2))
        return
    x, y, qx, qy = data()
    grid = torch.linspace(0., 1., T, dtype=DTYPE)
    w = .008 * (1. + .25 * torch.cos(torch.pi * grid))
    phi = torch.tensor([.11, -.23], dtype=DTYPE)
    rows = []
    for tau in (.02, .25, 100.):
        for horizon in (16, 32, 256):
            res = production_chain(phi, w, tau, horizon, x, y, qx, qy)
            ref = explicit_ref(phi, w, tau, horizon, x, y, qx, qy)
            _ = ref
            rows.append({
                "tau": tau, "horizon": horizon,
                "module_file": MODULE_FILE, "module_sha": MODULE_SHA,
                "max_diff_theta": res["max_diff"]["theta"],
                "max_diff_m": res["max_diff"]["m"],
                "max_diff_v": res["max_diff"]["v"],
                "max_diff_S_theta": res["max_diff"]["S_theta"],
                "max_diff_S_m": res["max_diff"]["S_m"],
                "max_diff_S_v": res["max_diff"]["S_v"],
                "first_divergence": res["first_divergence"],
                "grad_prod": res["grad"],
                "grad_max_abs_err": res["grad_err"],
                "PASS": res["first_divergence"] is None
                        and res["grad_err"] < 1e-8})
            print(f"tau={tau} h={horizon}: PASS={rows[-1]['PASS']} "
                  f"grad_err={res['grad_err']:.3e} "
                  f"first_div={res['first_divergence']}", flush=True)
    rule = rule_derivative_check()
    summary = {"module_file": MODULE_FILE, "module_sha": MODULE_SHA,
               "cases": rows, "rule_derivative_check": rule,
               "all_pass": all(r["PASS"] for r in rows)
               and rule["autograd_jac_err"] < 1e-12
               and rule["sum_eta_rel_err"] < 1e-12}
    (OUT / "production_reference_compare.json").write_text(
        json.dumps(summary, indent=2))
    # CSV
    with (OUT / "production_reference_compare.csv").open("w") as f:
        f.write("tau,horizon,max_diff_theta,max_diff_m,max_diff_v,"
                "max_diff_S_theta,max_diff_S_m,max_diff_S_v,"
                "grad_max_abs_err,first_divergence,PASS\n")
        for r in rows:
            fd = r["first_divergence"]
            f.write(",".join(str(x_) for x_ in (
                r["tau"], r["horizon"], r["max_diff_theta"],
                r["max_diff_m"], r["max_diff_v"], r["max_diff_S_theta"],
                r["max_diff_S_m"], r["max_diff_S_v"], r["grad_max_abs_err"],
                f"{fd}" if fd else "NONE", r["PASS"])) + "\n")
    print(json.dumps({"all_pass": summary["all_pass"],
                      "rule_check": rule}, indent=1))


if __name__ == "__main__":
    main()
