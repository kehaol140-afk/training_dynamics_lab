"""F2.5 trust-region library (v01) — frozen scope in docs/F2_5_TRUST_REGION_PREREG.md.

Oracle-intervened AdamW tangent step: per-parameter math is a line-by-line
replica of the vendored frozen adamw_tangent_step, with exactly two
backward-only (tangent-internal) intervention hooks:

  dv_mask   : per-coordinate 0/1 float tensor; zeros the v->theta' channel term
              (-mhat*dsqrt/denom^2) where mask==0. The m-channel term
              (dmhat/denom) is untouched.
  v_back_tau: replaces v by max(v, tau^2) throughout the tangent linearization
              (both M and Dv block denominators; F1-B semantics). The forward
              optimizer state is never modified.

Unit test contract: with no intervention the output must be BIT-EXACT equal to
the vendored frozen adamw_tangent_step (tests/test_f25_trust_region_v01.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .a0c_adamw_response_propagation_v01 import _finite

F25_EPS_METRIC = 1e-7
F25_EPS_ABS = 1e-6
F25_DELTA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1)
F25_H_READOUTS = (1, 2, 4, 8, 16, 32)
F25_R_THRESHOLD = 0.2


def tangent_step_oracle(named_parameters, optimizer, clipped_gradient_by_name,
                        tangent_clipped_gradient_by_name, tangent_theta_by_name,
                        tangent_m_by_name, tangent_v_by_name,
                        dv_mask_by_name: Optional[Dict] = None,
                        v_back_tau: Optional[float] = None):
    """Same per-parameter math as the vendored frozen function; see module doc.

    Returns dict with tangent_theta/m/v per name (JVP part only; the F2.5
    inhomogeneous u-term theta_next-theta_prev is added by the runner)."""
    named = list(named_parameters)
    names = [n for n, _ in named]
    expected = set(names)
    for label, mapping in (("clipped gradient", clipped_gradient_by_name),
                           ("clipped gradient tangent", tangent_clipped_gradient_by_name),
                           ("theta tangent", tangent_theta_by_name),
                           ("m tangent", tangent_m_by_name),
                           ("v tangent", tangent_v_by_name)):
        if set(mapping) != expected:
            raise ValueError(f"{label} parameter names do not match optimizer model")
    groups = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            groups[p] = group
    if set(groups) != {p for _, p in named}:
        raise ValueError("named parameters do not exactly match optimizer parameters")

    out_th, out_m, out_v = {}, {}, {}
    for name, parameter in named:
        group = groups[parameter]
        if bool(group.get("amsgrad", False)) or bool(group.get("maximize", False)):
            raise ValueError("registered tangent requires ordinary non-AMSGrad minimization")
        state = optimizer.state.get(parameter)
        if state is None or not all(k in state for k in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError(f"optimizer state missing m/v/step for {name}")
        beta1, beta2 = map(float, group["betas"])
        lr = _finite(group["lr"], f"lr[{name}]")
        adam_eps = _finite(group["eps"], f"eps[{name}]")
        weight_decay = _finite(group.get("weight_decay", 0.0), f"weight_decay[{name}]")
        step_before = _finite(state["step"], f"step[{name}]")
        step_after = step_before + 1.0
        bc1 = 1.0 - beta1 ** step_after
        bc2 = 1.0 - beta2 ** step_after
        if lr < 0.0 or adam_eps < 0.0 or bc1 <= 0.0 or bc2 <= 0.0:
            raise ValueError(f"invalid AdamW hyperparameters for {name}")

        m = state["exp_avg"].detach()
        v = state["exp_avg_sq"].detach()
        c = clipped_gradient_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        dc = tangent_clipped_gradient_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        dtheta = tangent_theta_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        dm = tangent_m_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        dv = tangent_v_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        if len({tuple(t.shape) for t in (m, v, c, dc, dtheta, dm, dv)}) != 1:
            raise ValueError(f"AdamW tangent tensor shape mismatch for {name}")
        if bool(torch.any(v < 0.0)):
            raise ValueError(f"negative exp_avg_sq for {name}")

        if v_back_tau is not None and v_back_tau > 0.0:
            v = torch.clamp(v, min=v_back_tau * v_back_tau)

        m1 = beta1 * m + (1.0 - beta1) * c
        v1 = beta2 * v + (1.0 - beta2) * c.square()
        dm1 = beta1 * dm + (1.0 - beta1) * dc
        dv1 = beta2 * dv + 2.0 * (1.0 - beta2) * c * dc
        mhat = m1 / bc1
        vhat = v1 / bc2
        dmhat = dm1 / bc1
        dvhat = dv1 / bc2
        sqrt_vhat = torch.sqrt(vhat)
        denominator = sqrt_vhat + adam_eps
        dsqrt = torch.zeros_like(dvhat)
        regular = sqrt_vhat > 0.0
        dsqrt[regular] = dvhat[regular] / (2.0 * sqrt_vhat[regular])
        if bool(torch.any((~regular) & (dvhat != 0.0))):
            raise ValueError(f"non-differentiable zero-vhat coordinate for {name}")
        v_channel = mhat * dsqrt / denominator.square()
        if dv_mask_by_name is not None and name in dv_mask_by_name:
            v_channel = v_channel * dv_mask_by_name[name].to(device=v_channel.device,
                                                              dtype=v_channel.dtype)
        adaptive_jvp = dmhat / denominator - v_channel
        out_th[name] = ((1.0 - lr * weight_decay) * dtheta - lr * adaptive_jvp).detach()
        out_m[name] = dm1.detach()
        out_v[name] = dv1.detach()
    return {"tangent_theta": out_th, "tangent_m": out_m, "tangent_v": out_v}


def rden_topk_masks(state, fraction):
    """Frozen rule (docs/F2_5_TRUST_REGION_PREREG.md §3): global top-f coordinates
    by R_den = 1/(sqrt(vhat1)+eps) computed at the BANK anchor state, via
    opsens.f0_bank_audit_v01.denominator_anchor. Returns per-name 0/1 float
    masks (1 = keep, 0 = knocked out) covering all parameters, plus meta."""
    import opsens.f0_bank_audit_v01 as f0
    cands = []
    shapes = {}
    total = 0
    hp = f0.group_hyperparams(state)
    for name, idx, slot in f0.named_optimizer_entries(state):
        denom = f0.denominator_anchor(slot["exp_avg_sq"], float(slot["step"]),
                                      hp["beta2"], hp["eps"])
        cands.append((name, 1.0 / denom.flatten()))  # R_den = 1/(sqrt(vhat1)+eps)
        shapes[name] = tuple(denom.shape)
        total += denom.numel()
    k = max(1, int(fraction * total))
    allR = torch.cat([c[1] for c in cands])
    thresh = torch.kthvalue(allR, total - k + 1).values  # k-th largest
    offsets = {}
    off = 0
    for name, r in cands:
        offsets[name] = (off, off + r.numel())
        off += r.numel()
    masks = {}
    for name, r in cands:
        lo, hi = offsets[name]
        sel = (allR[lo:hi] >= thresh)
        # reshape to the parameter's own shape: the tangent step multiplies
        # v_channel (parameter-shaped) by this mask elementwise.
        masks[name] = (~sel).reshape(shapes[name]).to(torch.float32)  # 1 = keep, 0 = knocked out
    return masks, {"k": k, "total": total, "thresh": float(thresh)}


def compute_R(dL, h, delta, eps_metric=F25_EPS_METRIC):
    pred = h * delta
    return abs(dL - pred) / (abs(pred) + eps_metric)


def usable(dL, h, delta, r_thr=F25_R_THRESHOLD, eps_metric=F25_EPS_METRIC):
    R = compute_R(dL, h, delta, eps_metric)
    return R <= r_thr and ((h * delta) > 0) == (dL > 0), R


def trust_radius(delta_grid, usable_flags, resolvable_flags):
    """max delta such that every smaller RESOLVABLE delta is also usable (and
    the delta itself is usable+resolvable); 0.0 when nothing qualifies."""
    order = sorted(zip(delta_grid, usable_flags, resolvable_flags))
    radius = 0.0
    for d, u, r in order:
        if not r:
            continue
        if u:
            radius = max(radius, d)
        else:
            return radius
    return radius
