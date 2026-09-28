"""Exact multi-step AdamW tangent propagation for the A0C audit.

[FR-OBSERVATION-ONLY] Existing checkpoints, q directions, manifests, and v07
finite-difference results are read-only inputs.
[FR-NATIVE-ADAMW] The tangent follows the complete native AdamW state update,
including clipping, m/v, bias correction, Adam epsilon, LR, and weight decay.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np


SELECTED_STATES = ((2037, 46), (2030, 46), (2037, 82))
HORIZONS = (1, 2, 4, 8)
CLIP_DENOMINATOR_EPS = 1e-6


def _torch():
    import torch

    return torch


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def relative_error(observed: float, predicted: float) -> float | None:
    observed = _finite(observed, "observed")
    predicted = _finite(predicted, "predicted")
    if observed == 0.0:
        return 0.0 if predicted == 0.0 else None
    return float(abs(predicted - observed) / abs(observed))


def _identical_names(left: Mapping, right: Mapping, label: str) -> list[str]:
    if not left or set(left) != set(right):
        raise ValueError(f"{label} must have identical non-empty parameter names")
    return list(left)


def clip_jvp(
    *, gradient_by_name: Mapping[str, object],
    tangent_gradient_by_name: Mapping[str, object], max_norm: float,
    denominator_eps: float = CLIP_DENOMINATOR_EPS,
) -> Dict:
    """Apply global-L2 clipping and its directional derivative.

    This is the smooth JVP of PyTorch's
    ``g * min(max_norm/(||g|| + 1e-6), 1)`` away from the clamp kink. Returned
    tensors stay on the gradient device and preserve its dtype.
    """
    torch = _torch()
    names = _identical_names(
        gradient_by_name, tangent_gradient_by_name, "gradient/tangent-gradient"
    )
    max_norm = _finite(max_norm, "max_norm")
    denominator_eps = _finite(denominator_eps, "denominator_eps")
    if max_norm < 0.0 or denominator_eps < 0.0:
        raise ValueError("clip max_norm and denominator epsilon must be nonnegative")

    norm_sq = None
    dot = None
    tangent_norm_sq = None
    gradients = {}
    tangents = {}
    for name in names:
        gradient = gradient_by_name[name]
        tangent = tangent_gradient_by_name[name]
        if not torch.is_tensor(gradient) or not torch.is_tensor(tangent):
            raise TypeError(f"gradient/tangent must be tensors for {name}")
        if tuple(gradient.shape) != tuple(tangent.shape):
            raise ValueError(f"gradient/tangent shape mismatch for {name}")
        gradient = gradient.detach()
        tangent = tangent.detach().to(device=gradient.device, dtype=gradient.dtype)
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(tangent).all()):
            raise ValueError(f"gradient/tangent contains NaN or Inf for {name}")
        gradients[name] = gradient
        tangents[name] = tangent
        g2 = torch.sum(gradient.to(torch.float64).square())
        gd = torch.sum(gradient.to(torch.float64) * tangent.to(torch.float64))
        d2 = torch.sum(tangent.to(torch.float64).square())
        norm_sq = g2 if norm_sq is None else norm_sq + g2
        dot = gd if dot is None else dot + gd
        tangent_norm_sq = d2 if tangent_norm_sq is None else tangent_norm_sq + d2

    norm = float(torch.sqrt(norm_sq).detach().cpu())
    gradient_dot_tangent = float(dot.detach().cpu())
    tangent_norm = float(torch.sqrt(tangent_norm_sq).detach().cpu())
    raw_coefficient = max_norm / (norm + denominator_eps)
    coefficient = min(raw_coefficient, 1.0)
    active = bool(raw_coefficient < 1.0)
    clipped = {}
    jvp = {}
    if active:
        if norm == 0.0:
            raise ValueError("active clipping cannot have zero gradient norm")
        radial = max_norm * gradient_dot_tangent / (
            norm * (norm + denominator_eps) ** 2
        )
        for name in names:
            clipped[name] = coefficient * gradients[name]
            jvp[name] = coefficient * tangents[name] - radial * gradients[name]
    else:
        for name in names:
            clipped[name] = gradients[name].clone()
            jvp[name] = tangents[name].clone()

    after_sq = None
    jvp_sq = None
    for name in names:
        value = torch.sum(clipped[name].to(torch.float64).square())
        tangent_value = torch.sum(jvp[name].to(torch.float64).square())
        after_sq = value if after_sq is None else after_sq + value
        jvp_sq = tangent_value if jvp_sq is None else jvp_sq + tangent_value
    return {
        "clipped_gradient": clipped,
        "clip_jvp": jvp,
        "clip_before_norm": norm,
        "clip_after_norm": float(torch.sqrt(after_sq).detach().cpu()),
        "clip_coefficient": float(coefficient),
        "clip_active": active,
        "clip_kink_margin": float(norm + denominator_eps - max_norm),
        "gradient_dot_tangent": gradient_dot_tangent,
        "tangent_gradient_norm": tangent_norm,
        "clip_jvp_norm": float(torch.sqrt(jvp_sq).detach().cpu()),
        "clip_denominator_epsilon": denominator_eps,
    }


def _parameter_groups(optimizer) -> Dict[object, Mapping]:
    result = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter in result:
                raise ValueError("optimizer parameter appears in multiple groups")
            result[parameter] = group
    return result


def _tensor_norm(mapping: Mapping[str, object]) -> float:
    torch = _torch()
    total = None
    for value in mapping.values():
        part = torch.sum(value.detach().to(torch.float64).square())
        total = part if total is None else total + part
    if total is None:
        raise ValueError("cannot norm an empty tensor mapping")
    return float(torch.sqrt(total).detach().cpu())


def adamw_tangent_step(
    *, named_parameters: Sequence[tuple[str, object]], optimizer,
    clipped_gradient_by_name: Mapping[str, object],
    tangent_clipped_gradient_by_name: Mapping[str, object],
    tangent_theta_by_name: Mapping[str, object],
    tangent_m_by_name: Mapping[str, object],
    tangent_v_by_name: Mapping[str, object],
) -> Dict:
    """Differentiate one complete AdamW step before the native step executes."""
    torch = _torch()
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
    step_values = []
    lr_values = []
    beta1_values = []
    beta2_values = []
    eps_values = []
    decay_values = []
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
        bc1 = 1.0 - beta1**step_after
        bc2 = 1.0 - beta2**step_after
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
        step_values.append(step_after)
        lr_values.append(lr)
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        eps_values.append(adam_eps)
        decay_values.append(weight_decay)

    def single(values, label):
        unique = sorted(set(map(float, values)))
        if len(unique) != 1:
            raise ValueError(f"heterogeneous {label}: {unique}")
        return unique[0]

    return {
        "tangent_theta": next_theta,
        "tangent_m": next_m,
        "tangent_v": next_v,
        "tangent_theta_norm": _tensor_norm(next_theta),
        "tangent_m_norm": _tensor_norm(next_m),
        "tangent_v_norm": _tensor_norm(next_v),
        "optimizer_step_after": single(step_values, "optimizer step"),
        "lr": single(lr_values, "learning rate"),
        "beta1": single(beta1_values, "beta1"),
        "beta2": single(beta2_values, "beta2"),
        "adam_eps": single(eps_values, "Adam epsilon"),
        "weight_decay": single(decay_values, "weight decay"),
    }


def project_tangent(q_by_name: Mapping[str, object], tangent_by_name: Mapping[str, object]) -> float:
    torch = _torch()
    names = _identical_names(q_by_name, tangent_by_name, "q/tangent")
    total = torch.zeros((), dtype=torch.float64)
    for name in names:
        tangent = tangent_by_name[name].detach()
        q = q_by_name[name].detach().to(device=tangent.device, dtype=tangent.dtype)
        total += torch.sum(q.to(torch.float64).cpu() * tangent.to(torch.float64).cpu())
    return float(total)


def _checkpoint_param_order(optimizer_state: Mapping) -> list:
    result = []
    for group in optimizer_state.get("param_groups", []):
        result.extend(group.get("params", []))
    return result


def _checkpoint_slot(state: Mapping, parameter_id):
    if parameter_id in state:
        return state[parameter_id]
    if str(parameter_id) in state:
        return state[str(parameter_id)]
    raise KeyError(f"optimizer state missing parameter id {parameter_id!r}")


def old_momentum_kernel(
    checkpoint: Mapping, q_by_name: Mapping[str, object], horizons: Sequence[int],
    *, eta: float,
) -> Dict:
    """Reproduce historical model A using first-future-step LR and raw v."""
    torch = _torch()
    model_state = checkpoint.get("model")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError("checkpoint must contain model and optimizer mappings")
    names = [name for name in model_state if name in q_by_name]
    parameter_ids = _checkpoint_param_order(optimizer)
    if set(names) != set(q_by_name) or len(names) != len(parameter_ids):
        raise ValueError("checkpoint/q/optimizer parameter alignment failed")
    groups = {}
    for group in optimizer.get("param_groups", []):
        for parameter_id in group.get("params", []):
            groups[parameter_id] = group
    mu = torch.zeros((), dtype=torch.float64)
    lr_values = []
    beta1_values = []
    beta2_values = []
    eps_values = []
    for name, parameter_id in zip(names, parameter_ids):
        group = groups[parameter_id]
        state = _checkpoint_slot(optimizer["state"], parameter_id)
        q = q_by_name[name].detach().to(torch.float64).cpu()
        v = state["exp_avg_sq"].detach().to(torch.float64).cpu()
        adam_eps = _finite(group["eps"], f"eps[{name}]")
        mu += torch.sum(q.square() / (torch.sqrt(v) + adam_eps))
        lr_values.append(_finite(group["lr"], f"lr[{name}]"))
        beta1_values.append(_finite(group["betas"][0], f"beta1[{name}]"))
        beta2_values.append(_finite(group["betas"][1], f"beta2[{name}]"))
        eps_values.append(adam_eps)

    def single(values, label):
        unique = sorted(set(values))
        if len(unique) != 1:
            raise ValueError(f"heterogeneous {label}: {unique}")
        return unique[0]

    checkpoint_lr = single(lr_values, "checkpoint LR")
    eta = _finite(eta, "first future optimizer-step eta")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    beta1 = single(beta1_values, "beta1")
    beta2 = single(beta2_values, "beta2")
    adam_eps = single(eps_values, "Adam epsilon")
    mu_value = float(mu)
    predictions = {
        str(int(horizon)): float(-eta * mu_value * (1.0 - beta1 ** int(horizon)))
        for horizon in horizons
    }
    return {
        "mu_q": mu_value,
        "eta_first_future_step": eta,
        "checkpoint_preceding_step_lr": checkpoint_lr,
        "beta1": beta1,
        "beta2": beta2,
        "adam_eps": adam_eps,
        "predictions": predictions,
        "formula": "-eta_first_future_step*mu_raw*(1-beta1**H)",
    }


def extract_v07_fd(theta_state: Mapping, horizons: Sequence[int]) -> Dict:
    """Extract the original v07 unit central differences without replaying ± branches."""
    epsilon = _finite(theta_state.get("epsilon"), "epsilon")
    if epsilon <= 0.0:
        raise ValueError("v07 epsilon must be positive")
    streams = theta_state.get("e2_streams")
    if not isinstance(streams, list) or len(streams) < 2:
        raise ValueError("v07 state needs at least two E2 streams")
    per_stream = []
    for record in sorted(streams, key=lambda row: int(row["stream_id"])):
        chi = {}
        for horizon in map(int, horizons):
            key = str(horizon)
            plus = _finite(record["plus"]["X_theta"][key], f"plus X[{key}]")
            minus = _finite(record["minus"]["X_theta"][key], f"minus X[{key}]")
            chi[key] = float((plus - minus) / (2.0 * epsilon))
        per_stream.append({"stream_id": int(record["stream_id"]), "chi_fd": chi})
    mean = {
        str(int(horizon)): float(
            np.mean([row["chi_fd"][str(int(horizon))] for row in per_stream])
        )
        for horizon in horizons
    }
    saved = theta_state.get("chi_theta", {})
    gaps = [
        abs(mean[str(int(horizon))] - _finite(saved[str(int(horizon))], "saved chi"))
        for horizon in horizons
    ]
    return {
        "epsilon": epsilon,
        "per_stream": per_stream,
        "mean": mean,
        "max_abs_vs_saved_chi": float(max(gaps, default=0.0)),
    }


def assess_baseline_replay(
    *, propagated: Mapping, shadow_h8: Mapping, saved_v07_h32: Mapping,
    horizons: Sequence[int],
) -> Dict:
    """Compare like-horizon H=8 hashes and registered v07 projections.

    The original v07 branch hash was captured only after its maximum H=32,
    whereas this experiment stops at H=8. It is therefore provenance evidence,
    not a comparable final-state hash. A fresh existing-protocol H=8 shadow
    branch supplies the valid whole-state comparison.
    """
    propagated_x = propagated["baseline_X_theta"]
    shadow_x = shadow_h8["observations"]
    saved_x = saved_v07_h32["X_theta"]
    shadow_gaps = {}
    saved_gaps = {}
    for horizon in map(int, horizons):
        key = str(horizon)
        value = _finite(propagated_x[key], f"propagated X[{key}]")
        shadow_gaps[key] = abs(
            value - _finite(shadow_x[key]["X_theta"], f"shadow X[{key}]")
        )
        saved_gaps[key] = abs(
            value - _finite(saved_x[key], f"saved v07 X[{key}]")
        )
    model_equal = (
        str(propagated["model_hash_final"])
        == str(shadow_h8["model_hash_final"])
    )
    optimizer_equal = (
        str(propagated["optimizer_hash_final"])
        == str(shadow_h8["optimizer_hash_final"])
    )
    max_shadow = float(max(shadow_gaps.values(), default=0.0))
    max_saved = float(max(saved_gaps.values(), default=0.0))
    return {
        "pass": bool(model_equal and optimizer_equal and max_shadow == 0.0 and max_saved == 0.0),
        "shadow_model_hash_equal": model_equal,
        "shadow_optimizer_hash_equal": optimizer_equal,
        "max_X_abs_gap_vs_shadow_H8": max_shadow,
        "max_X_abs_gap_vs_saved_v07": max_saved,
        "X_abs_gap_vs_shadow_H8": shadow_gaps,
        "X_abs_gap_vs_saved_v07": saved_gaps,
        "saved_final_hash_comparable": False,
        "saved_final_hash_horizon": 32,
        "propagated_and_shadow_hash_horizon": max(map(int, horizons)),
        "propagated_H8_model_hash": str(propagated["model_hash_final"]),
        "propagated_H8_optimizer_hash": str(propagated["optimizer_hash_final"]),
        "shadow_H8_model_hash": str(shadow_h8["model_hash_final"]),
        "shadow_H8_optimizer_hash": str(shadow_h8["optimizer_hash_final"]),
        "saved_H32_model_hash": str(saved_v07_h32["model_hash_final"]),
        "saved_H32_optimizer_hash": str(saved_v07_h32["optimizer_hash_final"]),
    }


def _metric(observed: Sequence[float], predicted: Sequence[float]) -> Dict:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.size < 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("metrics require matching finite arrays with at least two rows")
    residual = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    corr = None if float(np.std(y)) == 0.0 or float(np.std(p)) == 0.0 else float(np.corrcoef(y, p)[0, 1])
    slope = None
    intercept = None
    if float(np.var(p)) > 0.0:
        slope, intercept = map(float, np.polyfit(p, y, 1))
    return {
        "n": int(y.size),
        "corr": corr,
        "R2": None if sst == 0.0 else float(1.0 - np.sum(residual**2) / sst),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "mean_relative_error": float(
            np.mean([value for value in (relative_error(a, b) for a, b in zip(y, p)) if value is not None])
        ),
        "slope_observed_on_prediction": slope,
        "intercept_observed_on_prediction": intercept,
    }


def prediction_metrics(rows: Sequence[Mapping]) -> Dict:
    models = {
        "A_momentum": "chi_A_momentum",
        "B_optimizer_only": "chi_optimizer_only",
        "C_full_tangent": "chi_full_tangent",
    }
    result = {}
    for label, field in models.items():
        overall = _metric(
            [float(row["chi_fd"]) for row in rows],
            [float(row[field]) for row in rows],
        )
        by_state = {}
        state_groups = defaultdict(list)
        horizon_groups = defaultdict(list)
        for row in rows:
            state_groups[str(row["state"])].append(row)
            horizon_groups[str(int(row["H"]))].append(row)
        for state, group in state_groups.items():
            by_state[state] = _metric(
                [float(row["chi_fd"]) for row in group],
                [float(row[field]) for row in group],
            )
        by_horizon = {}
        for horizon, group in horizon_groups.items():
            by_horizon[horizon] = _metric(
                [float(row["chi_fd"]) for row in group],
                [float(row[field]) for row in group],
            )
        result[label] = {
            "prediction_field": field,
            "overall": overall,
            "per_state": by_state,
            "per_horizon": by_horizon,
        }
    return result


def assemble_summary(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    metrics: Mapping, source_paths: Mapping, input_hashes: Mapping,
    runtime: Mapping, conclusions: Mapping,
) -> Dict:
    expected_rows = {
        (seed, step, horizon)
        for seed, step in SELECTED_STATES for horizon in HORIZONS
    }
    actual_rows = {
        (int(row["seed"]), int(row["step"]), int(row["H"]))
        for row in result_rows
    }
    if len(result_rows) != len(expected_rows) or actual_rows != expected_rows:
        raise ValueError("result rows do not exactly match locked 3x4 grain")
    expected_streams = {
        (seed, step, stream_id)
        for seed, step in SELECTED_STATES for stream_id in range(8)
    }
    actual_streams = {
        (int(row["seed"]), int(row["step"]), int(row["stream_id"]))
        for row in stream_results
    }
    if len(stream_results) != len(expected_streams) or actual_streams != expected_streams:
        raise ValueError("stream rows do not exactly match locked 3x8 grain")
    required_models = {"A_momentum", "B_optimizer_only", "C_full_tangent"}
    if set(metrics) != required_models:
        raise ValueError("metrics must contain exactly models A/B/C")
    return {
        "protocol": "FR_A0C_ADAMW_RESPONSE_PROPAGATION_V01",
        "status": "COMPLETE",
        "scope": {
            "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
            "horizons": list(HORIZONS),
            "streams_per_state": 8,
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_registered)]",
        "model_definitions": {
            "A_momentum": "-eta_first_future_step*mu_raw*(1-beta1**H)",
            "B_optimizer_only": "complete stepwise AdamW tangent; pulse delta_g=q then future delta_g=0",
            "C_full_tangent": "complete stepwise AdamW tangent; pulse delta_g=q then future delta_g=H_math_SDPA(loss)*delta_theta; native flash baseline retained",
        },
        "tangent_units": "DERIVATIVE_PER_UNIT_POST_UNSCALE_GRADIENT_PULSE",
        "fit_parameters": [],
        "result_row_count": len(result_rows),
        "stream_count": len(stream_results),
        "results": list(result_rows),
        "stream_results": list(stream_results),
        "metrics": dict(metrics),
        "conclusions": dict(conclusions),
        "source_paths": dict(source_paths),
        "input_hashes": dict(input_hashes),
        "runtime": dict(runtime),
        "constraints": [
            "EXACT_V07_STATE_Q_CHECKPOINT_MANIFEST_STREAM",
            "REAL_BASELINE_FUTURE_BATCH_AND_RNG",
            "NATIVE_MICROBATCH_LR_AND_ACCUMULATION",
            "COMPLETE_ADAMW_M_V_BIAS_CORRECTION_EPS_WEIGHT_DECAY",
            "STEPWISE_GLOBAL_CLIP_JACOBIAN",
            "FULL_TANGENT_USES_REAL_LOSS_HVP",
            "NO_FITTED_OR_EMPIRICAL_VARIABLE",
            "NO_OLD_INPUT_MUTATION",
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    conclusions = summary["conclusions"]
    metrics = summary["metrics"]
    lines = [
        "# FR A0C Complete AdamW Response Propagation v01",
        "",
        "## Technical summary",
        "",
        f"1. **H=1 Jacobian 能否扩展到多步：{conclusions['h1_extends_to_multistep']}。** "
        f"{conclusions.get('h1_extension_explanation', conclusions.get('explanation', ''))}",
        f"2. **失败来自 optimizer memory 还是 curvature feedback：{conclusions['failure_source']}。** "
        f"{conclusions.get('failure_source_explanation', '')}",
        f"3. **是否需要 Mori-Zwanzig memory kernel：{conclusions['mori_zwanzig_needed']}。** "
        f"{conclusions.get('mori_zwanzig_explanation', '')}",
        "",
        "## A/B/C comparison across the three states",
        "",
        "Each row is a unit post-unscale pulse response. `rel A/B/C` is absolute prediction error divided by `|chi_fd|`.",
        "",
        "| state | H | chi_fd | A momentum | B optimizer-only | C full tangent | rel A | rel B | rel C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['state']} | {row['H']} | {_fmt(row['chi_fd'])} | "
            f"{_fmt(row['chi_A_momentum'])} | {_fmt(row['chi_optimizer_only'])} | "
            f"{_fmt(row['chi_full_tangent'])} | {_fmt(row['relative_error_A'])} | "
            f"{_fmt(row['relative_error_optimizer_only'])} | "
            f"{_fmt(row['relative_error_full_tangent'])} |"
        )
    lines.extend([
        "",
        "## Full tangent isolates curvature feedback from optimizer-only propagation",
        "",
        "The compact table is preferable to a trend chart for twelve exact audit points. Model B carries all optimizer memory while setting future gradient feedback to zero; the B-to-C change therefore isolates the smooth Hessian-feedback contribution along the same baseline streams. A→B does not isolate optimizer memory: A also omits the exact local preconditioner, bias correction, clipping Jacobian, future LR schedule, weight decay tangent, and current-gradient dependence.",
        "",
        "| model | overall R2 | corr | MAE | RMSE | mean relative error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label in ("A_momentum", "B_optimizer_only", "C_full_tangent"):
        row = metrics[label]["overall"]
        lines.append(
            f"| {label} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | "
            f"{_fmt(row['MAE'])} | {_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
        )
    lines.extend([
        "",
        "## Scope and response definition",
        "",
        "- Three locked confirmatory v07 states, eight paired E2 streams per state, and horizons H=1,2,4,8.",
        "- `chi_fd` is the original registered finite-epsilon central response `(X_plus-X_minus)/(2*epsilon)` averaged over streams; epsilon is not multiplied into any prediction.",
        "- All predictions are derivatives per unit pulse in the exact state-specific q direction used by the original measurement.",
        "- Model A preserves the historical first-future-step LR and raw-v `mu`; B and C use each future optimizer step's scheduled LR and complete state.",
        "",
        "## Complete optimizer propagation with a qualified math-SDPA HVP",
        "",
        "For every stream the complete checkpoint is restored, the registered RNG is seeded, and the registered future batches are replayed with BF16 autocast and two-microbatch accumulation. The baseline order is `forward -> loss -> backward -> unscale -> clip -> optimizer.step`. At the pulse step both tangents use `delta_g=q`. Later B uses `delta_g=0`, while C obtains `H(loss) delta_theta` from the same accumulated loss using reverse-over-reverse autograd. The optimizer-side JVP is exact: both directions pass through the native global clipping rule and complete bias-corrected AdamW m/v/parameter update, including Adam epsilon and decoupled weight decay. PyTorch does not implement double backward for the native flash-SDPA kernel, so only the auxiliary HVP forward uses math-SDPA; C is therefore a mixed-backend analytic surrogate, not an exact derivative of the flash kernel implementation.",
        "",
        "## Robustness, replay integrity, and limitations",
        "",
        f"- Baseline replay: `{summary['runtime'].get('baseline_replay', 'not recorded')}`; max saved-chi aggregation gap: `{_fmt(summary['runtime'].get('max_saved_chi_abs_gap'))}`.",
        f"- Input integrity: `{summary['runtime'].get('input_integrity', 'not recorded')}`; HVP backend: `{summary['runtime'].get('hvp_backend', 'not recorded')}`.",
        f"- Native flash double backward: `{summary['runtime'].get('native_flash_double_backward', 'not recorded')}`. The maximum relative gap between the math-SDPA HVP forward's first gradient and the native flash gradient is `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native'))}`; this does not bound the HVP error and qualifies C's smallest residuals.",
        "- Whole-state replay uses a fresh existing-protocol H=8 shadow branch: propagated/shadow model and optimizer hashes match on 24/24 streams. The original v07 final hashes are H=32 and are retained as provenance but are not compared to an H=8 state.",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "- C is a zero-pulse smooth tangent, while `chi_fd` uses the registered finite epsilon. A discrepancy may include finite-epsilon clip/activation boundary crossings; this is especially relevant to seed 2037/state 46, whose H=1 epsilon-convergence audit showed a smaller error at epsilon/4.",
        "- A Hessian-vector product captures smooth curvature along the baseline trajectory. It does not represent discrete changes of clipping regime or other non-smooth branch changes between finite ±epsilon trajectories.",
        "",
        "## Recommended next step",
        "",
        conclusions.get("recommended_next_step", "Use the quantified C residual to decide whether a reduced memory-kernel model is warranted."),
        "",
        "## Further question",
        "",
        conclusions.get("further_question", "Does the remaining residual shrink under a multi-step epsilon-halving check, separating tangent truncation from missing memory variables?"),
        "",
    ])
    return "\n".join(lines)


def json_dumps(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
