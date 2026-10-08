"""Heterogeneous-group AdamW tangent step for FR_G1C_1B_FEASIBILITY_GATE_V01.

[FR-G1C-ADAPTER] The frozen ``fr.a0c_adamw_response_propagation_v01.
adamw_tangent_step`` computes the identical per-parameter tangent math but
aggregates ``single(lr/betas/eps/weight_decay)`` across ALL parameters — a
MiniMind single-group assumption (feasibility protocol section 6.8). The
official OLMo-2-0425-1B checkpoint has 2 param groups with different
weight_decay (0.1 / 0.0), which the frozen homogeneity check would reject.

This adapter is a LINE-BY-LINE copy of the frozen function with ONLY the
homogeneity aggregation changed: hyperparameters are returned per group
instead of a single global value. Every per-parameter tangent formula is
untouched. Tests (research/tests/test_g1c_1b_feasibility_v01.py) prove:
  1. on a homogeneous single-group optimizer, adapter == frozen function
     exactly (same tensors, same scalars);
  2. on heterogeneous groups, adapter == a naive independent per-parameter
     reference implementation.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch

from .a0c_adamw_response_propagation_v01 import _finite, _parameter_groups, _tensor_norm


def g1c_adamw_tangent_step(
    *, named_parameters: Sequence[tuple[str, object]], optimizer,
    clipped_gradient_by_name: Mapping[str, object],
    tangent_clipped_gradient_by_name: Mapping[str, object],
    tangent_theta_by_name: Mapping[str, object],
    tangent_m_by_name: Mapping[str, object],
    tangent_v_by_name: Mapping[str, object],
) -> Dict:
    """Differentiate one complete AdamW step before the native step executes.

    Identical math to the frozen implementation; hyperparameters are returned
    per group (lists aligned with ``param_groups`` order) to support
    heterogeneous optimizer groups.
    """
    torch = __import__("torch")
    named_parameters = list(named_parameters)
    names = [name for name, _ in named_parameters]
    expected = set(names)
    for label, mapping in (
        ("clipped gradient", clipped_gradient_by_name),
        ("clipped gradient tangent", tangent_clipped_gradient_by_name),
        ("theta tangent", tangent_theta_by_name),
        ("m tangent", tangent_m_by_name),
        ("v tangent", tangent_v_by_name),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{label} parameter names do not match optimizer model")
    groups = _parameter_groups(optimizer)
    if set(groups) != {parameter for _, parameter in named_parameters}:
        raise ValueError("named parameters do not exactly match optimizer parameters")

    next_theta = {}
    next_m = {}
    next_v = {}
    per_group = []
    for name, parameter in named_parameters:
        group = groups[parameter]
        if bool(group.get("amsgrad", False)) or bool(group.get("maximize", False)):
            raise ValueError("registered tangent requires ordinary non-AMSGrad minimization")
        if bool(group.get("differentiable", False)):
            raise ValueError("differentiable native optimizer is not the registered path")
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not all(
            key in state for key in ("step", "exp_avg", "exp_avg_sq")
        ):
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
        if len({tuple(value.shape) for value in (m, v, c, dc, dtheta, dm, dv)}) != 1:
            raise ValueError(f"AdamW tangent tensor shape mismatch for {name}")
        if bool(torch.any(v < 0.0)):
            raise ValueError(f"negative exp_avg_sq for {name}")

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
        adaptive_jvp = dmhat / denominator - mhat * dsqrt / denominator.square()
        next_theta[name] = (
            (1.0 - lr * weight_decay) * dtheta - lr * adaptive_jvp
        ).detach()
        next_m[name] = dm1.detach()
        next_v[name] = dv1.detach()
        per_group.append({
            "step_after": step_after, "lr": lr, "beta1": beta1, "beta2": beta2,
            "adam_eps": adam_eps, "weight_decay": weight_decay,
        })

    # Aggregate per group (replaces the frozen single() homogeneity check).
    by_group = {}
    idx = 0
    for gi, group in enumerate(optimizer.param_groups):
        rows = []
        params = list(group["params"])
        for _ in params:
            if idx >= len(per_group):
                break
            rows.append(per_group[idx])
            idx += 1
        if not rows:
            continue
        by_group[gi] = {
            "n_params": len(rows),
            "step_after_values": sorted(set(r["step_after"] for r in rows)),
            "lr_values": sorted(set(r["lr"] for r in rows)),
            "beta1_values": sorted(set(r["beta1"] for r in rows)),
            "beta2_values": sorted(set(r["beta2"] for r in rows)),
            "adam_eps_values": sorted(set(r["adam_eps"] for r in rows)),
            "weight_decay_values": sorted(set(r["weight_decay"] for r in rows)),
        }
    if idx != len(per_group):
        raise ValueError("parameter/group alignment failure in adapter aggregation")

    return {
        "tangent_theta": next_theta,
        "tangent_m": next_m,
        "tangent_v": next_v,
        "tangent_theta_norm": _tensor_norm(next_theta),
        "tangent_m_norm": _tensor_norm(next_m),
        "tangent_v_norm": _tensor_norm(next_v),
        "per_group": by_group,
    }
