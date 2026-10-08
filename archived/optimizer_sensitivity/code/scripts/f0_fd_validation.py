#!/usr/bin/env python3
"""F0 Phase 5: finite-difference validation of the analytic one-step Jacobian
against the REAL torch.optim.AdamW stepping on bank state slices (READ-ONLY).

Anchors:
  A0: c0 = 0 (null gradient; matches Phase-4 exact diagonal; clip inactive).
  A1: c0 = exp_avg (bank-derived nonstationary anchor; activates m/dsqrt cross terms).

FD: [F(c0 + eps*q) - F(c0 - eps*q)]/(2 eps) on the update u(c) = theta - theta'(c),
compared against the vendored frozen adamw_tangent_step JVP at the same anchor.
eps grid spans both regimes of the v-floor nonlinearity: for coordinates with
v ~ 0 the Adam normalization makes the update ~sign(c) for sqrt(v1_hat) >> eps
and ~c/eps for sqrt(v1_hat) << eps, so FD approaches the analytic c=0 Jacobian
only for eps << eps_adam / sqrt((1-beta2)/bc2).

Precision variants: float64 (reference), float32 (native AdamW), bfloat16
(manual forward math; native bf16 CPU AdamW unsupported).

NaN/Inf adversarial probe on never/tail embedding rows in bf16 with full
required field logging.

Outputs: outputs/f0_existing_bank_audit/fd_jvp_comparison.csv,
         nan_inf_probes.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import opsens.f0_bank_audit_v01 as f0

CONFIG = json.loads((ROOT / "configs/f0_bank_audit_v01.json").read_text())
OUT = ROOT / "outputs/f0_existing_bank_audit"
OUT.mkdir(parents=True, exist_ok=True)
EMBED_NAME = "model.embed_tokens.weight"


def manual_adamw_update(theta, m, v, step_t, c, lr, b1, b2, eps, wd):
    """Forward AdamW update math (torch semantics) in the input dtypes."""
    t_after = step_t + 1.0
    m1 = b1 * m + (1.0 - b1) * c
    v1 = b2 * v + (1.0 - b2) * c.square()
    mhat = m1 / (1.0 - b1 ** t_after)
    vhat = v1 / (1.0 - b2 ** t_after)
    denom = vhat.sqrt() + eps
    return theta * (1.0 - lr * wd) - lr * mhat / denom


def native_adamw_step(theta, m, v, step_t, c, hp, dtype):
    p = torch.nn.Parameter(theta.detach().to(dtype).clone())
    opt = torch.optim.AdamW([p], lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]),
                            eps=hp["eps"], weight_decay=hp["weight_decay"])
    opt.state[p] = {
        "step": torch.tensor(float(step_t), dtype=torch.float32),
        "exp_avg": m.detach().to(dtype).clone(),
        "exp_avg_sq": v.detach().to(dtype).clone(),
    }
    p.grad = c.detach().to(dtype).clone()
    opt.step()
    return p.detach()


def analytic_jvp(theta, m, v, step_t, c0, q, hp, dtype):
    """Frozen tangent math on a single flat slice via vendored module."""
    from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step

    class _P:
        pass

    p = torch.nn.Parameter(theta.detach().to(dtype).clone())
    opt = _P()
    opt.param_groups = [{"params": [p], "lr": hp["lr"], "betas": (hp["beta1"], hp["beta2"]),
                         "eps": hp["eps"], "weight_decay": hp["weight_decay"],
                         "amsgrad": False, "maximize": False, "differentiable": False}]
    opt.state = {p: {"step": torch.tensor(float(step_t)),
                     "exp_avg": m.detach().to(dtype).clone(),
                     "exp_avg_sq": v.detach().to(dtype).clone()}}
    out = adamw_tangent_step(
        named_parameters=[("slice", p)], optimizer=opt,
        clipped_gradient_by_name={"slice": c0.detach().to(dtype).clone()},
        tangent_clipped_gradient_by_name={"slice": q.detach().to(dtype).clone()},
        tangent_theta_by_name={"slice": torch.zeros_like(p)},
        tangent_m_by_name={"slice": torch.zeros_like(p)},
        tangent_v_by_name={"slice": torch.zeros_like(p)},
    )
    return out["tangent_theta"]["slice"]


def _fd_row(scale, seed, step, slice_name, n_coords, anchor, precision, eps_abs, fd, an64):
    fd64 = fd.double()
    an = an64.double()
    fn = float(fd64.norm())
    an_ = float(an.norm())
    dot = float((fd64 * an).sum()) if fd64.shape == an.shape else float("nan")
    cos = dot / (fn * an_) if fn > 0 and an_ > 0 else float("nan")
    rel = abs(fn - an_) / an_ if an_ > 0 else (0.0 if fn == 0.0 else float("inf"))
    return {
        "scale": scale, "seed": seed, "step": step, "slice": slice_name,
        "n_coords": n_coords, "anchor": anchor, "precision": precision,
        "eps": eps_abs,
        "fd_norm": fn, "analytic_norm": an_,
        "rel_err_vs_fp64_analytic": rel, "cosine_vs_analytic": cos,
        "fd_finite": bool(torch.isfinite(fd).all().item()),
    }


def rel_err(fd, an):
    fn = float(fd.double().norm())
    an_ = float(an.double().norm())
    if an_ == 0.0:
        return 0.0 if fn == 0.0 else float("inf")
    return abs(fn - an_) / an_


def main() -> None:
    t0 = time.time()
    freq = f0.token_frequency_bins()
    gen = torch.Generator().manual_seed(20260926)
    rows = []
    probes = []

    for scale, (seed, step) in CONFIG["fd_states"].items():
        spec = CONFIG["banks"][scale]
        p = Path(spec["root"]) / spec["state_glob"].format(seed=seed, step=step)
        state = f0.load_bank_state(p)
        hp = f0.group_hyperparams(state)
        slot = next(s for n, i, s in f0.named_optimizer_entries(state) if n == EMBED_NAME)
        theta_e = state["model"][EMBED_NAME].detach().to(torch.float64)
        m_e = slot["exp_avg"].detach().to(torch.float64)
        v_e = slot["exp_avg_sq"].detach().to(torch.float64)
        step_t = float(slot["step"])

        def slice_rows(ids, n=64):
            ids = ids[:n]
            rows_idx = torch.tensor(ids, dtype=torch.long)
            return rows_idx

        bins = {b: torch.tensor(freq[b], dtype=torch.long) for b in ("head", "mid", "tail", "never")}
        slices = []
        for b in ("never", "tail", "head"):
            if bins[b].numel() == 0:
                continue
            ids = bins[b][:64]
            t = torch.zeros_like(theta_e)
            mm = torch.zeros_like(m_e)
            vv = torch.zeros_like(v_e)
            mask = torch.zeros_like(v_e)
            t[ids] = theta_e[ids]
            mm[ids] = m_e[ids]
            vv[ids] = v_e[ids]
            mask[ids] = 1.0
            slices.append((f"embed_{b}", t, mm, vv, mask, int(ids.numel())))
        mlp_name = "model.layers.0.mlp.gate_proj.weight"
        mlp_slot = next(s for n, i, s in f0.named_optimizer_entries(state) if n == mlp_name)
        flat_n = mlp_slot["exp_avg_sq"].numel()
        k = min(4096, flat_n)
        sel = torch.randperm(flat_n, generator=gen)[:k]
        mmask = torch.zeros(flat_n, dtype=torch.float64)
        mmask[sel] = 1.0
        slices.append(("mlp_gate_sample",
                       mlp_flat(state, mlp_name, "model"),
                       mlp_flat(state, mlp_name, "exp_avg", mlp_slot),
                       mlp_flat(state, mlp_name, "exp_avg_sq", mlp_slot), mmask, k))

        for slice_name, theta, m, v, mask, n_coords in slices:
            q = (torch.randn(v.shape, generator=gen, dtype=torch.float64) * mask)
            q = q / q.norm()
            for anchor_name, c0 in (("c0_zero", torch.zeros_like(v)), ("c0_mhat", m.clone())):
                an64 = analytic_jvp(theta, m, v, step_t, c0, q, hp, torch.float64)
                for eps_abs in CONFIG["fd_epsilons_abs"]["float64"]:
                    up = manual_adamw_update(theta, m, v, step_t, c0 + eps_abs * q, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"])
                    um = manual_adamw_update(theta, m, v, step_t, c0 - eps_abs * q, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"])
                    fd = (up - um) / (2.0 * eps_abs)
                    rows.append(_fd_row(scale, seed, step, slice_name, n_coords, anchor_name, "float64_manual_math", eps_abs, fd, an64))
                for eps_abs in CONFIG["fd_epsilons_abs"]["float32"]:
                    up = native_adamw_step(theta, m, v, step_t, (c0 + eps_abs * q), hp, torch.float32)
                    um = native_adamw_step(theta, m, v, step_t, (c0 - eps_abs * q), hp, torch.float32)
                    fd = (up - um) / (2.0 * eps_abs)
                    rows.append(_fd_row(scale, seed, step, slice_name, n_coords, anchor_name, "float32_native_adamw", eps_abs, fd, an64))
                bf16_gain_probe(scale, seed, step, slice_name, theta, m, v, step_t, q, hp, probes)
            print(f"[{scale}] slice {slice_name} done ({time.time()-t0:.0f}s)", flush=True)
        del state

    with open(OUT / "fd_jvp_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "nan_inf_probes.json").write_text(json.dumps(probes, indent=2))
    print(f"[done] {len(rows)} FD rows, {len(probes)} probes, wall={time.time()-t0:.0f}s", flush=True)


def mlp_flat(state, name, what, slot=None):
    if what == "model":
        return state["model"][name].detach().to(torch.float64).flatten().clone()
    if what == "exp_avg":
        return slot["exp_avg"].detach().to(torch.float64).flatten().clone()
    return slot["exp_avg_sq"].detach().to(torch.float64).flatten().clone()


def bf16_gain_probe(scale, seed, step, slice_name, theta, m, v, step_t, q, hp, probes):
    """Adversarial bf16 forward math on the slice; log NaN/Inf with context."""
    for eps_abs in (1e-6, 1e-4, 1e-2):
        th = theta.to(torch.bfloat16)
        mm = m.to(torch.bfloat16)
        vv = v.to(torch.bfloat16)
        cc = (eps_abs * q).to(torch.bfloat16)
        upd = manual_adamw_update(th, mm, vv, step_t, cc, hp["lr"], hp["beta1"], hp["beta2"],
                                  torch.tensor(hp["eps"], dtype=torch.bfloat16).item(), hp["weight_decay"])
        finite = bool(torch.isfinite(upd).all().item())
        denom = (vv * hp["beta2"] + (1.0 - hp["beta2"]) * cc.square().float()).sqrt() + hp["eps"]
        probes.append({
            "scale": scale, "seed": seed, "step": step, "slice": slice_name,
            "dtype": "bfloat16", "eps_fd": eps_abs,
            "update_all_finite": finite,
            "update_abs_max": float(upd.float().abs().max()),
            "n_inf": int(torch.isinf(upd.float()).sum()),
            "n_nan": int(torch.isnan(upd.float()).sum()),
            "sqrt_v1_max": float(denom.max()),
            "sqrt_v1_min_nonzero": float(denom[denom > 0].min()) if (denom > 0).any() else 0.0,
            "m_abs_max": float(mm.float().abs().max()),
            "v_abs_max": float(vv.float().abs().max()),
            "adam_eps": hp["eps"], "lr": hp["lr"],
            "clip_norm": None,
        })


if __name__ == "__main__":
    main()
