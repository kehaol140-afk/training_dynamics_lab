#!/usr/bin/env python3
"""F0 诊断：小 T 上 SENS vs 未截断 autograd（同一固定线性终点）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import torch

import f0_terminal_rule_run_v01 as R
from fr import f0_terminal_rule_v01 as f0
from fr.stream_rng import seed_future_stream
import a0g_decision_utility_v01 as a0g


def _fixed_target(rt, seed_t=999):
    out = {}
    for i, (n, p) in enumerate(rt["named"]):
        g = torch.Generator().manual_seed(seed_t + i)
        cpu = torch.randn(p.shape, generator=g, dtype=torch.float32)
        out[n] = cpu.to(p.device).to(torch.float64)
    return out


def autograd_grad(rt, seed, stream_seed, phi, T, target):
    from torch.func import functional_call
    w = f0.build_w(T)
    psi = f0.build_psi(T)
    aL = torch.tensor(phi[0], dtype=torch.float64, requires_grad=True)
    bL = torch.tensor(phi[1], dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64)
    pt = torch.tensor(psi, dtype=torch.float64)
    R.load_bank(rt, seed)
    model = rt["model"]
    named_u = [(n, p) for n, p in rt["named"]]
    uniq = {n: p for n, p in named_u}
    theta = {n: p.detach().clone().requires_grad_(True) for n, p in named_u}
    m = {}
    v = {}
    for n, p in named_u:
        st = rt["optimizer"].state[p]
        m[n] = st["exp_avg"].detach().clone()
        v[n] = st["exp_avg_sq"].detach().clone()
    # functional_call 名字映射（tie：两个名字指向同一张量）
    full_map = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        for un, up in named_u:
            if p is up:
                full_map[n] = theta[un]
                break
    seed_future_stream(int(stream_seed))
    stream = R.make_stream(rt, stream_seed, 2 * T)
    model.train()
    for t in range(T):
        eb = torch.exp(bL * pt)
        Zb = torch.sum(wt * eb) / torch.sum(wt)
        eta_t = (wt[t] * torch.exp(aL) * torch.exp(bL * pt[t]) / Zb).to(
            torch.float32)
        losses = []
        with a0g._math_sdpa_context(torch):
            for a_ in range(2):
                x, y = R.stack_batch_rt(rt, stream["batches"][2 * t + a_])
                res = functional_call(model, full_map, (x,), {"labels": y})
                losses.append((res.loss + res.aux_loss) / 2)
            gs = torch.autograd.grad(losses[0] + losses[1],
                                     list(theta.values()),
                                     create_graph=True, allow_unused=True)
        g = {}
        for (n, _), g_ in zip(named_u, gs):
            if g_ is None:
                print(f"WARN unused param: {n}", flush=True)
                g_ = torch.zeros_like(theta[n])
            g[n] = g_
        norm = None
        for n in theta:
            s_ = torch.sum(g[n] * g[n])
            norm = s_ if norm is None else norm + s_
        coef = torch.clamp(1.0 / (torch.sqrt(norm) + 1e-6), max=1.0)
        c = {n: coef * g[n] for n in theta}
        bc1 = 1.0 - 0.9 ** (46 + t + 1)
        bc2 = 1.0 - 0.999 ** (46 + t + 1)
        new_theta = {}
        for n in theta:
            m[n] = 0.9 * m[n] + 0.1 * c[n]
            v[n] = 0.999 * v[n] + 0.001 * c[n] * c[n]
            mbar = m[n] / bc1
            vbar = v[n] / bc2
            new_theta[n] = theta[n] - eta_t * (
                mbar / (torch.sqrt(vbar) + 1e-8) + 0.01 * theta[n])
        theta = new_theta
    J = sum(torch.sum(theta[n].double() * target[n]) for n in theta)
    ga, gb = torch.autograd.grad(J, [aL, bL])
    return float(ga), float(gb)


def sens_grad(rt, seed, stream_seed, phi, T, target):
    out = R.ref32_rollout(rt, seed, stream_seed, phi, T, sens=True,
                          want_grad=False, tag=f"diag{T}")
    S = out["sens"]["S_theta"]
    ga = sum(float(torch.sum(target[n] * S[n]["a"]).item())
             for n in out["theta"])
    gb = sum(float(torch.sum(target[n] * S[n]["b"]).item())
             for n in out["theta"])
    return ga, gb, out


def main():
    R._load_stream_config()
    device = "cuda:0"
    seed = 2030
    rt = R.build_rt(seed, device)
    target = _fixed_target(rt)
    phi = (0.1, -0.2)
    results = {}
    for T in (2, 4, 8, 16):
        ga, gb = autograd_grad(rt, seed, R.FIT[seed], phi, T, target)
        sa, sb, out = sens_grad(rt, seed, R.FIT[seed], phi, T, target)
        na = abs(ga - sa) / max(abs(ga), abs(sa), 1e-30)
        nb = abs(gb - sb) / max(abs(gb), abs(sb), 1e-30)
        results[T] = {"autograd": [ga, gb], "sens": [sa, sb],
                      "rel_diff": [na, nb]}
        print(f"T={T}: autograd=({ga:.6e},{gb:.6e}) sens=({sa:.6e},"
              f"{sb:.6e}) rel=({na:.2e},{nb:.2e})", flush=True)
        _ = out
    (R.OUT / "runs" / "diag_small_T.json").write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=1))
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
