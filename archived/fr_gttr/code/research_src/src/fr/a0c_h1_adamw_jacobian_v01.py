"""Exact H=1 native-AdamW Jacobian helpers for the A0C audit.

[FR-OBSERVATION-ONLY] The functions consume saved state and captured
post-unscale gradients. They never mutate a checkpoint or q artifact.
[FR-NATIVE-ADAMW] Moments are the raw native ``exp_avg`` and ``exp_avg_sq``
slots; the derivative includes their next-step updates and bias corrections.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .batch_manifest import manifest_sha256


SELECTED_STATES = ((2037, 46), (2030, 46), (2037, 82))
CLIP_DENOMINATOR_EPS = 1e-6


def _torch():
    import torch

    return torch


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epsilon_levels(epsilon: float) -> list[float]:
    value = _finite(epsilon, "epsilon")
    if value <= 0.0:
        raise ValueError("epsilon must be positive")
    return [value, value / 2.0, value / 4.0]


def validate_e2_manifest(
    payload: Mapping, *, expected_seed: int, expected_step: int,
    expected_stream: int, expected_content_sha: str, accumulation_steps: int,
) -> Dict:
    """Validate one registered v07 E2 H1 batch stream before model use."""
    if payload.get("manifest_version") != "0.2":
        raise ValueError(f"unexpected manifest version: {payload.get('manifest_version')!r}")
    metadata = payload.get("metadata")
    batches = payload.get("batches")
    if not isinstance(metadata, Mapping) or not isinstance(batches, list):
        raise ValueError("manifest metadata/batches are missing")
    if metadata.get("protocol") != "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM":
        raise ValueError(f"unexpected manifest protocol: {metadata.get('protocol')!r}")
    actual = {
        "train_seed": int(metadata.get("train_seed", -1)),
        "state_step": int(metadata.get("state_step", -1)),
        "stream_id": int(metadata.get("stream_id", -1)),
    }
    expected = {
        "train_seed": int(expected_seed),
        "state_step": int(expected_step),
        "stream_id": int(expected_stream),
    }
    if actual != expected:
        raise ValueError(
            "manifest identity mismatch: "
            f"train_seed={actual['train_seed']}, state_step={actual['state_step']}, "
            f"stream_id={actual['stream_id']}; expected train_seed={expected['train_seed']}, "
            f"state_step={expected['state_step']}, stream_id={expected['stream_id']}"
        )
    content_sha = manifest_sha256(batches)
    if content_sha != str(payload.get("sha256")) or content_sha != str(expected_content_sha):
        raise ValueError("manifest batch-content sha256 mismatch")
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0 or len(batches) < accumulation_steps:
        raise ValueError("manifest does not contain the complete H1 accumulation window")
    for batch in batches[:accumulation_steps]:
        if not isinstance(batch, list) or not batch or not all(isinstance(x, int) for x in batch):
            raise ValueError("manifest contains an invalid H1 batch")
    for field in ("stream_seed", "batch_seed", "epsilon", "rho"):
        if field not in metadata:
            raise ValueError(f"manifest metadata missing {field}")
    return {
        "batches": [list(batch) for batch in batches],
        "metadata": dict(metadata),
        "content_sha256": content_sha,
    }


def relative_error(observed: float, expected: float) -> float | None:
    observed = _finite(observed, "observed")
    expected = _finite(expected, "expected")
    if expected == 0.0:
        return 0.0 if observed == 0.0 else None
    return float(abs(observed - expected) / abs(expected))


def clip_directional_derivative(
    *, q_by_name: Mapping[str, object], gradient_by_name: Mapping[str, object],
    max_norm: float, denominator_eps: float = CLIP_DENOMINATOR_EPS,
) -> Dict:
    """Return PyTorch global-L2 clipping and its directional derivative.

    PyTorch 2.11 uses ``min(max_norm/(total_norm+1e-6), 1)``. The result is
    the analytic JVP at the baseline gradient, away from the clamp kink.
    """
    torch = _torch()
    if set(q_by_name) != set(gradient_by_name) or not q_by_name:
        raise ValueError("q and gradient must have identical non-empty parameter names")
    max_norm = _finite(max_norm, "max_norm")
    denominator_eps = _finite(denominator_eps, "denominator_eps")
    if max_norm < 0.0 or denominator_eps < 0.0:
        raise ValueError("clip max_norm and denominator epsilon must be nonnegative")

    q64 = {}
    g64 = {}
    norm_sq = torch.zeros((), dtype=torch.float64)
    q_norm_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for name in q_by_name:
        q = q_by_name[name]
        g = gradient_by_name[name]
        if not torch.is_tensor(q) or not torch.is_tensor(g):
            raise TypeError(f"q/gradient must be tensors for {name}")
        qq = q.detach().to(device="cpu", dtype=torch.float64)
        gg = g.detach().to(device="cpu", dtype=torch.float64)
        if tuple(qq.shape) != tuple(gg.shape):
            raise ValueError(f"q/gradient shape mismatch for {name}")
        if not bool(torch.isfinite(qq).all()) or not bool(torch.isfinite(gg).all()):
            raise ValueError(f"q/gradient contains NaN or Inf for {name}")
        q64[name] = qq
        g64[name] = gg
        norm_sq += torch.sum(gg * gg)
        q_norm_sq += torch.sum(qq * qq)
        dot += torch.sum(gg * qq)

    norm = float(torch.sqrt(norm_sq).item())
    q_norm = float(torch.sqrt(q_norm_sq).item())
    g_dot_q = float(dot.item())
    raw_coefficient = max_norm / (norm + denominator_eps)
    coefficient = min(raw_coefficient, 1.0)
    active = bool(raw_coefficient < 1.0)
    clipped = {}
    jvp = {}
    if active:
        if norm == 0.0:
            raise ValueError("active clipping cannot have zero gradient norm")
        radial = max_norm * g_dot_q / (norm * (norm + denominator_eps) ** 2)
        for name in q64:
            clipped[name] = coefficient * g64[name]
            jvp[name] = coefficient * q64[name] - radial * g64[name]
    else:
        for name in q64:
            clipped[name] = g64[name].clone()
            jvp[name] = q64[name].clone()

    after_norm_sq = torch.zeros((), dtype=torch.float64)
    for value in clipped.values():
        after_norm_sq += torch.sum(value * value)
    after_norm = float(torch.sqrt(after_norm_sq).item())
    return {
        "clipped_gradient": clipped,
        "clip_jvp": jvp,
        "clip_before_norm": norm,
        "clip_after_norm": after_norm,
        "clip_coefficient": float(coefficient),
        "clip_active": active,
        "clip_kink_margin": float(norm + denominator_eps - max_norm),
        "gradient_dot_q": g_dot_q,
        "q_norm": q_norm,
        "clip_denominator_epsilon": denominator_eps,
    }


def _optimizer_param_order(optimizer: Mapping) -> list:
    order = []
    for group in optimizer.get("param_groups", []):
        order.extend(group.get("params", []))
    return order


def _groups_by_param(optimizer: Mapping) -> Dict[object, Mapping]:
    result = {}
    for group in optimizer.get("param_groups", []):
        for param_id in group.get("params", []):
            if param_id in result:
                raise ValueError(f"optimizer parameter in multiple groups: {param_id!r}")
            result[param_id] = group
    return result


def _slot(state: Mapping, param_id: object) -> Mapping:
    if param_id in state:
        return state[param_id]
    if str(param_id) in state:
        return state[str(param_id)]
    raise KeyError(f"optimizer state missing parameter id {param_id!r}")


def _single(values: Sequence[float], label: str) -> float:
    unique = sorted(set(map(float, values)))
    if len(unique) != 1:
        raise ValueError(f"heterogeneous {label} is not defined for scalar report: {unique}")
    return unique[0]


def adamw_h1_jacobian(
    *, checkpoint: Mapping, q_by_name: Mapping[str, object],
    gradient_by_name: Mapping[str, object], effective_lr: float, grad_clip: float,
) -> Dict:
    """Compute ``q^T G0 q`` for the complete next native AdamW step."""
    torch = _torch()
    effective_lr = _finite(effective_lr, "effective_lr")
    if effective_lr < 0.0:
        raise ValueError("effective_lr must be nonnegative")
    model_state = checkpoint.get("model")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError("checkpoint must contain model and optimizer mappings")
    if set(q_by_name) != set(gradient_by_name) or not q_by_name:
        raise ValueError("q/gradient names must match exactly")

    parameter_names = [name for name in model_state if name in q_by_name]
    if set(parameter_names) != set(q_by_name):
        raise ValueError("q names are not fully present in checkpoint model state")
    parameter_ids = _optimizer_param_order(optimizer)
    if len(parameter_names) != len(parameter_ids):
        raise ValueError(
            f"optimizer/model/q counts differ: {len(parameter_ids)} vs {len(parameter_names)}"
        )
    group_by_param = _groups_by_param(optimizer)
    state = optimizer.get("state", {})
    clip = clip_directional_derivative(
        q_by_name=q_by_name,
        gradient_by_name=gradient_by_name,
        max_norm=grad_clip,
    )

    chi_m = torch.zeros((), dtype=torch.float64)
    chi_v = torch.zeros((), dtype=torch.float64)
    mu_old = torch.zeros((), dtype=torch.float64)
    mu_old_bc = torch.zeros((), dtype=torch.float64)
    q_norm_sq = torch.zeros((), dtype=torch.float64)
    steps_before = []
    steps_after = []
    beta1_values = []
    beta2_values = []
    adam_eps_values = []
    weight_decay_values = []
    contributions = []

    for name, param_id in zip(parameter_names, parameter_ids):
        group = group_by_param[param_id]
        if bool(group.get("amsgrad", False)):
            raise ValueError("amsgrad=True is outside the registered native AdamW formula")
        if bool(group.get("maximize", False)):
            raise ValueError("maximize=True is outside the registered gradient-pulse sign")
        if bool(group.get("differentiable", False)):
            raise ValueError("differentiable=True is not the registered optimizer")
        if group.get("decoupled_weight_decay", True) is False:
            raise ValueError("checkpoint optimizer is not decoupled AdamW")
        betas = group.get("betas")
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError(f"AdamW betas missing for {name}")
        beta1 = _finite(betas[0], f"beta1[{name}]")
        beta2 = _finite(betas[1], f"beta2[{name}]")
        adam_eps = _finite(group.get("eps"), f"adam_eps[{name}]")
        weight_decay = _finite(group.get("weight_decay", 0.0), f"weight_decay[{name}]")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 or adam_eps < 0.0:
            raise ValueError(f"invalid AdamW hyperparameters for {name}")

        saved = _slot(state, param_id)
        if not all(key in saved for key in ("step", "exp_avg", "exp_avg_sq")):
            raise KeyError(f"optimizer m/v/step missing for {name}")
        step_before = _finite(saved["step"], f"step[{name}]")
        step_after = step_before + 1.0
        if step_before < 0.0:
            raise ValueError(f"negative optimizer step for {name}")
        m = saved["exp_avg"].detach().to(device="cpu", dtype=torch.float64)
        v = saved["exp_avg_sq"].detach().to(device="cpu", dtype=torch.float64)
        q = q_by_name[name].detach().to(device="cpu", dtype=torch.float64)
        c = clip["clipped_gradient"][name]
        cq = clip["clip_jvp"][name]
        if len({tuple(x.shape) for x in (m, v, q, c, cq)}) != 1:
            raise ValueError(f"q/c/Cq/m/v shape mismatch for {name}")
        if bool(torch.any(v < 0.0)):
            raise ValueError(f"negative exp_avg_sq for {name}")

        a = 1.0 - beta1
        b = 1.0 - beta2
        bc1_after = 1.0 - beta1**step_after
        bc2_after = 1.0 - beta2**step_after
        bc2_before = 1.0 - beta2**step_before
        if bc1_after <= 0.0 or bc2_after <= 0.0 or bc2_before <= 0.0:
            raise ValueError(f"invalid AdamW bias correction for {name}")

        m1 = beta1 * m + a * c
        v1 = beta2 * v + b * c.square()
        mhat = m1 / bc1_after
        vhat = v1 / bc2_after
        sqrt_vhat = torch.sqrt(vhat)
        denominator = sqrt_vhat + adam_eps

        m_direction = -effective_lr * (a / bc1_after) * cq / denominator
        v_numerator = effective_lr * (b / bc2_after) * mhat * c * cq
        v_direction = torch.zeros_like(v_numerator)
        regular = sqrt_vhat > 0.0
        v_direction[regular] = (
            v_numerator[regular]
            / (sqrt_vhat[regular] * denominator[regular].square())
        )
        if bool(torch.any((~regular) & (v_numerator != 0.0))):
            raise ValueError(f"non-differentiable zero-vhat coordinate for {name}")

        name_chi_m = torch.sum(q * m_direction)
        name_chi_v = torch.sum(q * v_direction)
        chi_m += name_chi_m
        chi_v += name_chi_v
        mu_old += torch.sum(q.square() / (torch.sqrt(v) + adam_eps))
        mu_old_bc += torch.sum(q.square() / (torch.sqrt(v / bc2_before) + adam_eps))
        q_norm_sq += torch.sum(q.square())
        contributions.append(
            {
                "name": name,
                "chi_jac_m": float(name_chi_m.item()),
                "chi_jac_v": float(name_chi_v.item()),
                "chi_jac": float((name_chi_m + name_chi_v).item()),
            }
        )
        steps_before.append(step_before)
        steps_after.append(step_after)
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        adam_eps_values.append(adam_eps)
        weight_decay_values.append(weight_decay)

    step_before = _single(steps_before, "optimizer step")
    step_after = _single(steps_after, "next optimizer step")
    beta1 = _single(beta1_values, "beta1")
    beta2 = _single(beta2_values, "beta2")
    adam_eps = _single(adam_eps_values, "Adam epsilon")
    weight_decay = _single(weight_decay_values, "weight decay")
    chi_m_value = float(chi_m.item())
    chi_v_value = float(chi_v.item())
    chi_value = chi_m_value + chi_v_value
    first_moment_scale = effective_lr * (1.0 - beta1) / (1.0 - beta1**step_after)
    if first_moment_scale == 0.0:
        raise ValueError("cannot define mu_q_full at zero first-moment scale")
    contributions.sort(key=lambda row: abs(row["chi_jac"]), reverse=True)
    return {
        "chi_jac": chi_value,
        "chi_jac_m": chi_m_value,
        "chi_jac_v": chi_v_value,
        "mu_q_old": float(mu_old.item()),
        "mu_q_old_bias_corrected": float(mu_old_bc.item()),
        "mu_q_full": float(-chi_value / first_moment_scale),
        "q_norm_sq": float(q_norm_sq.item()),
        "effective_lr": effective_lr,
        "optimizer_step_before": step_before,
        "optimizer_step_after": step_after,
        "beta1": beta1,
        "beta2": beta2,
        "adam_eps": adam_eps,
        "weight_decay": weight_decay,
        "clip_before_norm": clip["clip_before_norm"],
        "clip_after_norm": clip["clip_after_norm"],
        "clip_coefficient": clip["clip_coefficient"],
        "clip_active": clip["clip_active"],
        "clip_kink_margin": clip["clip_kink_margin"],
        "gradient_dot_q": clip["gradient_dot_q"],
        "parameter_tensor_count": len(parameter_names),
        "top_parameter_contributions": contributions[:10],
        "formula": "q^T[(1-beta1)M+2(1-beta2)V*Diag(c)]*C*q",
        "injection_jacobian": "IDENTITY_POST_UNSCALE_GRADIENT_PULSE",
        "moment_semantics": "UPDATED_M_AND_V_WITH_STEP_PLUS_ONE_BIAS_CORRECTION",
        "weight_decay_derivative": "ZERO_FOR_GRADIENT_PULSE_AT_FIXED_INPUT_THETA",
    }


def summarize_epsilon_convergence(rows: Sequence[Mapping]) -> Dict:
    if len(rows) != 3:
        raise ValueError(f"expected three epsilon rows, got {len(rows)}")
    ordered = sorted(rows, key=lambda row: float(row["epsilon"]), reverse=True)
    errors = [float(row["relative_error"]) for row in ordered]
    return {
        "epsilon_descending": [float(row["epsilon"]) for row in ordered],
        "relative_error_descending": errors,
        "convergence_monotonic": bool(errors[1] <= errors[0] and errors[2] <= errors[1]),
        "full_to_quarter_error_ratio": None if errors[2] == 0.0 else float(errors[0] / errors[2]),
        "smallest_epsilon_relative_error": errors[2],
    }


def assemble_summary(
    *, state_results: Sequence[Mapping], epsilon_rows: Sequence[Mapping],
    stream_results: Sequence[Mapping], source_paths: Mapping, input_hashes: Mapping,
    runtime: Mapping, conclusions: Mapping,
) -> Dict:
    required_states = set(SELECTED_STATES)
    observed_states = {(int(row["seed"]), int(row["step"])) for row in state_results}
    if observed_states != required_states:
        raise ValueError(f"state set mismatch: {sorted(observed_states)}")
    if len(state_results) != 3 or len(epsilon_rows) != 9 or len(stream_results) != 24:
        raise ValueError("expected 3 states, 9 epsilon rows, and 24 streams")
    return {
        "protocol": "FR_A0C_H1_ADAMW_JACOBIAN_V01",
        "status": "COMPLETE",
        "scope": {
            "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
            "horizon": 1,
            "streams_per_state": 8,
            "epsilon_multipliers": [1.0, 0.5, 0.25],
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_level)]",
        "jacobian_formula": "G0=[aM+2bV*Diag(c)]*J*C; J=I for post-unscale pulse",
        "mu_definitions": {
            "mu_q_old": "sum(q^2/(sqrt(raw_exp_avg_sq)+adam_eps))",
            "mu_q_old_bias_corrected": "sum(q^2/(sqrt(raw_v/(1-beta2^step))+adam_eps))",
            "mu_q_full": "-chi_jac/[lr*(1-beta1)/(1-beta1^(step+1))]",
        },
        "fit_parameters": [],
        "state_count": len(state_results),
        "stream_count": len(stream_results),
        "epsilon_row_count": len(epsilon_rows),
        "states": list(state_results),
        "epsilon_results": list(epsilon_rows),
        "stream_results": list(stream_results),
        "conclusions": dict(conclusions),
        "source_paths": dict(source_paths),
        "input_hashes": dict(input_hashes),
        "runtime": dict(runtime),
        "constraints": [
            "EXACT_V07_STATE_Q",
            "EXACT_V07_E2_MANIFESTS_AND_STREAM_RNG",
            "FORWARD_BACKWARD_UNSCALE_PULSE_CLIP_STEP",
            "NATIVE_ADAMW_COMPLETE_M_V_UPDATE",
            "BIAS_CORRECTION_AND_ADAM_EPS",
            "NO_FROZEN_V_APPROXIMATION",
            "NO_BETA_HORIZON_APPROXIMATION",
            "NO_FITTED_OR_EMPIRICAL_SCALE",
            "OLD_INPUTS_READ_ONLY",
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    conclusions = summary["conclusions"]
    lines = [
        "# FR A0C H=1 Complete AdamW Jacobian v01",
        "",
        "## Technical summary",
        "",
        f"1. **H1 是否能解释真实响应：{conclusions['h1_explains_real_response']}。** "
        f"{conclusions.get('h1_explanation', '')}",
        f"2. **nu 异常是否来自冻结 v 近似：{conclusions['nu_anomaly_from_frozen_v']}。** "
        f"{conclusions.get('nu_explanation', '')}",
        f"3. **是否需要 Hessian feedback：{conclusions['hessian_feedback_next']}。** "
        f"{conclusions.get('hessian_explanation', '')}",
        "",
        "## Three-state result",
        "",
        "The headline row for each state uses epsilon/4; all three epsilon levels follow below.",
        "",
        "| state | epsilon | chi_fd | chi_jac | relative error | mu_q_old | mu_q_full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for state in summary["states"]:
        lines.append(
            f"| {state['state']} | {_fmt(state['epsilon'])} | {_fmt(state['chi_fd'])} | "
            f"{_fmt(state['chi_jac'])} | {_fmt(state['relative_error'])} | "
            f"{_fmt(state['mu_q_old'])} | {_fmt(state['mu_q_full'])} |"
        )
    lines.extend(
        [
            "",
            "## Epsilon convergence",
            "",
            "Exact lookup values are shown as a table because the experiment has only nine points; a chart would add no resolution.",
            "",
            "| state | epsilon level | epsilon | chi_fd | chi_jac | relative error | clip active (+/base/-) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["epsilon_results"]:
        active = row.get("clip_active", {})
        lines.append(
            f"| {row['state']} | {row.get('epsilon_level', 'n/a')} | {_fmt(row['epsilon'])} | "
            f"{_fmt(row['chi_fd'])} | {_fmt(row['chi_jac'])} | "
            f"{_fmt(row['relative_error'])} | "
            f"{active.get('plus_rate', 'n/a')}/{active.get('baseline_rate', 'n/a')}/"
            f"{active.get('minus_rate', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Scope and definitions",
            "",
            "- H=1 only; three preregistered v07 states; eight registered paired E2 streams per state.",
            "- `chi_fd` keeps the original unit-response definition `(X_plus-X_minus)/(2*epsilon)` and is averaged over streams.",
            "- `chi_jac` is the stream-mean exact local native-AdamW derivative, including clipping, updated m/v, step+1 bias correction, Adam epsilon, and actual LR.",
            "- `mu_q_old` is the preceding raw-v projection. `mu_q_full` is the exact Jacobian projected into the same first-moment scale; it is diagnostic, not fitted.",
            "",
            "## Complete single-step specification",
            "",
            "The replay order is `forward -> loss -> backward -> unscale -> pulse -> clip -> optimizer.step`. The clipping JVP uses PyTorch's `max_norm/(norm+1e-6)` rule. The formula is `G0=[aM+2bV Diag(c)]JC`; here `J=I` because the pulse is injected directly in gradient coordinates after unscale. Native decoupled weight decay is executed in finite differences, while its derivative with respect to the pulse is zero at fixed input theta.",
            "",
            "## Validation and limitations",
            "",
            f"- Input state/q/manifest integrity: `{summary['runtime'].get('input_integrity', 'not recorded')}`.",
            f"- Full-epsilon reproduction of saved v07 H1 branches: `{summary['runtime'].get('v07_replay_reproduction', 'not recorded')}`.",
            f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
            "- A local H1 match tests the optimizer-input Jacobian only. It does not establish H>1 dynamics, where parameter displacement changes later gradients and may cross activation or clipping boundaries.",
            "",
            "## Recommended next step",
            "",
            f"{conclusions.get('recommended_next_step', 'Use the H1 result to decide whether to propagate the perturbation through future-gradient Hessian feedback.')}",
            "",
            "## Further question",
            "",
            "For H>1, how much of the remaining response comes from smooth Hessian-vector propagation versus discrete clipping/activation boundary changes?",
            "",
        ]
    )
    return "\n".join(lines)


def json_dumps(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
