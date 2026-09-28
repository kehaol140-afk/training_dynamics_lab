"""Analysis primitives for the FR A0D finite-horizon epsilon audit.

[FR-OBSERVATION-ONLY] Existing checkpoints, q directions, manifests, A0C
measurements, and A0C tangent predictions are immutable inputs.
[FR-PERTURBATION-CRITICAL] Every response is divided by the epsilon actually
used by that paired branch.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np


SELECTED_STATES = ((2037, 46), (2030, 46), (2037, 82))
HORIZONS = (2, 4, 8)
EPSILON_LEVELS = ("epsilon", "epsilon/2", "epsilon/4")
EPSILON_MULTIPLIERS = (1.0, 0.5, 0.25)


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def central_response(*, x_plus: float, x_minus: float, epsilon: float) -> float:
    """Return the unit central response at the supplied pulse amplitude."""
    plus = _finite(x_plus, "x_plus")
    minus = _finite(x_minus, "x_minus")
    level = _finite(epsilon, "epsilon")
    if level <= 0.0:
        raise ValueError("epsilon must be positive")
    return float((plus - minus) / (2.0 * level))


def relative_error(observed: float, predicted: float) -> float | None:
    observed = _finite(observed, "observed")
    predicted = _finite(predicted, "predicted")
    if observed == 0.0:
        return 0.0 if predicted == 0.0 else None
    return float(abs(predicted - observed) / abs(observed))


def _ordered_epsilon_rows(rows: Sequence[Mapping]) -> list[Mapping]:
    if len(rows) != 3:
        raise ValueError(f"expected exactly three epsilon rows, got {len(rows)}")
    by_level = {str(row["epsilon_level"]): row for row in rows}
    if set(by_level) != set(EPSILON_LEVELS):
        raise ValueError("epsilon rows must contain epsilon, epsilon/2, epsilon/4")
    ordered = [by_level[level] for level in EPSILON_LEVELS]
    epsilon = [_finite(row["epsilon"], "epsilon") for row in ordered]
    if not (
        math.isclose(epsilon[1], epsilon[0] / 2.0, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(epsilon[2], epsilon[0] / 4.0, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError("epsilon levels are not exact registered halvings")
    tangent = [_finite(row["chi_tangent"], "chi_tangent") for row in ordered]
    if max(tangent) != min(tangent):
        raise ValueError("zero-pulse tangent prediction must not depend on epsilon")
    return ordered


def epsilon_convergence(rows: Sequence[Mapping]) -> Dict:
    """Describe convergence without introducing a fitted threshold."""
    ordered = _ordered_epsilon_rows(rows)
    finite_difference = [_finite(row["chi_fd"], "chi_fd") for row in ordered]
    tangent = _finite(ordered[0]["chi_tangent"], "chi_tangent")
    absolute = [abs(value - tangent) for value in finite_difference]
    relative = [relative_error(value, tangent) for value in finite_difference]
    first_change = abs(finite_difference[0] - finite_difference[1])
    second_change = abs(finite_difference[1] - finite_difference[2])
    ratio = None if first_change == 0.0 else float(second_change / first_change)
    order = None
    if first_change > 0.0 and second_change > 0.0:
        order = float(math.log2(first_change / second_change))
    return {
        "epsilon_descending": [_finite(row["epsilon"], "epsilon") for row in ordered],
        "chi_fd_descending": finite_difference,
        "chi_tangent": tangent,
        "absolute_error_descending": absolute,
        "relative_error_descending": relative,
        "fd_change_full_to_half": float(first_change),
        "fd_change_half_to_quarter": float(second_change),
        "fd_change_contraction_ratio": ratio,
        "observed_central_difference_order": order,
        "fd_changes_contract": bool(second_change <= first_change),
        "absolute_error_monotonic": bool(absolute[2] <= absolute[1] <= absolute[0]),
        "relative_error_monotonic": bool(
            all(value is not None for value in relative)
            and relative[2] <= relative[1] <= relative[0]
        ),
        "absolute_error_full_to_quarter_reduction": float(absolute[0] - absolute[2]),
        "relative_error_full_to_quarter_reduction": (
            None if relative[0] is None or relative[2] is None
            else float(relative[0] - relative[2])
        ),
    }


def h8_conclusion(
    convergence_rows: Sequence[Mapping], *, full_metrics: Mapping,
    quarter_metrics: Mapping,
) -> Dict:
    """Separate endpoint improvement from strict three-level monotonicity."""
    rows = list(convergence_rows)
    if len(rows) != 3:
        raise ValueError("H=8 conclusion requires exactly three state rows")
    endpoint_improvement = 0
    monotonic = 0
    for row in rows:
        errors = list(row["relative_error_descending"])
        if len(errors) != 3 or any(value is None for value in errors):
            raise ValueError("H=8 conclusion requires three finite relative errors")
        endpoint_improvement += bool(float(errors[2]) < float(errors[0]))
        monotonic += bool(row["relative_error_monotonic"])
    pooled_improves = float(quarter_metrics["mean_relative_error"]) < float(
        full_metrics["mean_relative_error"]
    )
    if endpoint_improvement == 3 and pooled_improves:
        answer = (
            "RETAINS_AND_IMPROVES_MONOTONICALLY_ALL_3"
            if monotonic == 3
            else "YES_AT_EPSILON_QUARTER_ALL_3_WITH_NONMONOTONIC_INTERMEDIATE"
        )
    else:
        answer = f"PARTIAL_ENDPOINT_IMPROVEMENT_{endpoint_improvement}_OF_3"
    return {
        "answer": answer,
        "endpoint_improvement_count": int(endpoint_improvement),
        "monotonic_count": int(monotonic),
        "pooled_endpoint_improves": bool(pooled_improves),
    }


def epsilon8_candidates(convergence_rows: Sequence[Mapping]) -> list[Dict]:
    """Select unresolved cells without confusing FD contraction with tangent fit."""
    result = []
    for row in convergence_rows:
        noncontracting = not bool(row["fd_changes_contract"])
        nonmonotonic = not bool(row["relative_error_monotonic"])
        if not (noncontracting or nonmonotonic):
            continue
        if noncontracting and nonmonotonic:
            reason = "FD_NONCONTRACTING_AND_TANGENT_ERROR_NONMONOTONIC"
        elif noncontracting:
            reason = "FD_NONCONTRACTING"
        else:
            reason = "TANGENT_ERROR_NONMONOTONIC"
        result.append(
            {"state": str(row["state"]), "H": int(row["H"]), "reason": reason}
        )
    return result


def _metric(rows: Sequence[Mapping]) -> Dict:
    if not rows:
        raise ValueError("metrics require at least one row")
    observed = np.asarray([float(row["chi_fd"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row["chi_tangent"]) for row in rows], dtype=np.float64)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite observations and predictions")
    residual = predicted - observed
    sst = float(np.sum((observed - observed.mean()) ** 2))
    correlation = None
    slope = None
    intercept = None
    if observed.size >= 2 and float(np.std(observed)) > 0.0 and float(np.std(predicted)) > 0.0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
        slope, intercept = map(float, np.polyfit(predicted, observed, 1))
    rel = [relative_error(a, b) for a, b in zip(observed, predicted)]
    rel = [value for value in rel if value is not None]
    return {
        "n": int(observed.size),
        "corr": correlation,
        "R2": None if observed.size < 2 or sst == 0.0 else float(
            1.0 - np.sum(np.square(residual)) / sst
        ),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_relative_error": None if not rel else float(np.mean(rel)),
        "slope_observed_on_prediction": slope,
        "intercept_observed_on_prediction": intercept,
    }


def prediction_metrics(rows: Sequence[Mapping]) -> Dict:
    """Return pooled and locked-grain tangent-vs-FD metrics."""
    rows = list(rows)
    epsilon_groups = defaultdict(list)
    horizon_groups = defaultdict(list)
    state_groups = defaultdict(list)
    horizon_epsilon_groups = defaultdict(lambda: defaultdict(list))
    state_epsilon_groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        level = str(row["epsilon_level"])
        horizon = str(int(row["H"]))
        state = str(row["state"])
        epsilon_groups[level].append(row)
        horizon_groups[horizon].append(row)
        state_groups[state].append(row)
        horizon_epsilon_groups[horizon][level].append(row)
        state_epsilon_groups[state][level].append(row)
    return {
        "overall": _metric(rows),
        "per_epsilon": {
            level: _metric(epsilon_groups[level]) for level in EPSILON_LEVELS
        },
        "per_horizon": {
            horizon: _metric(horizon_groups[horizon]) for horizon in map(str, HORIZONS)
        },
        "per_state": {state: _metric(group) for state, group in state_groups.items()},
        "per_horizon_epsilon": {
            horizon: {
                level: _metric(horizon_epsilon_groups[horizon][level])
                for level in EPSILON_LEVELS
            }
            for horizon in map(str, HORIZONS)
        },
        "per_state_epsilon": {
            state: {
                level: _metric(state_epsilon_groups[state][level])
                for level in EPSILON_LEVELS
            }
            for state in state_groups
        },
    }


def assemble_summary(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    convergence_results: Sequence[Mapping], metrics: Mapping,
    source_paths: Mapping, input_hashes: Mapping, source_config: Mapping,
    audit_config: Mapping, audit_config_sha256: str, runtime: Mapping,
    conclusions: Mapping,
) -> Dict:
    expected_rows = {
        (seed, step, horizon, level)
        for seed, step in SELECTED_STATES
        for horizon in HORIZONS
        for level in EPSILON_LEVELS
    }
    actual_rows = {
        (int(row["seed"]), int(row["step"]), int(row["H"]), str(row["epsilon_level"]))
        for row in result_rows
    }
    if len(result_rows) != 27 or actual_rows != expected_rows:
        raise ValueError("result rows do not exactly match locked 3x3x3 grain")
    expected_streams = {
        (seed, step, stream_id)
        for seed, step in SELECTED_STATES for stream_id in range(8)
    }
    actual_streams = {
        (int(row["seed"]), int(row["step"]), int(row["stream_id"]))
        for row in stream_results
    }
    if len(stream_results) != 24 or actual_streams != expected_streams:
        raise ValueError("stream results do not exactly match locked 3x8 grain")
    for row in stream_results:
        levels = [str(item["epsilon_level"]) for item in row.get("epsilon_results", [])]
        if levels != list(EPSILON_LEVELS):
            raise ValueError("each stream must contain the three ordered epsilon levels")
    expected_convergence = {
        (seed, step, horizon)
        for seed, step in SELECTED_STATES for horizon in HORIZONS
    }
    actual_convergence = {
        (int(row["seed"]), int(row["step"]), int(row["H"]))
        for row in convergence_results
    }
    if len(convergence_results) != 9 or actual_convergence != expected_convergence:
        raise ValueError("convergence results do not exactly match locked 3x3 grain")
    if not isinstance(source_config, Mapping) or not source_config:
        raise ValueError("complete source config is required")
    if not isinstance(audit_config, Mapping) or not audit_config:
        raise ValueError("complete audit config is required")
    if not audit_config_sha256:
        raise ValueError("audit config hash is required")
    return {
        "protocol": "FR_A0D_EPSILON_CONVERGENCE_V01",
        "status": "COMPLETE",
        "scope": {
            "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
            "horizons": list(HORIZONS),
            "epsilon_levels": list(EPSILON_LEVELS),
            "epsilon_multipliers": list(EPSILON_MULTIPLIERS),
            "streams_per_state": 8,
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_level)]",
        "tangent_definition": "saved A0C v01 full tangent derivative per unit post-unscale gradient pulse; independent of finite-difference epsilon",
        "fit_parameters": [],
        "result_row_count": len(result_rows),
        "stream_count": len(stream_results),
        "epsilon_stream_count": sum(len(row["epsilon_results"]) for row in stream_results),
        "results": list(result_rows),
        "stream_results": list(stream_results),
        "convergence_results": list(convergence_results),
        "metrics": dict(metrics),
        "conclusions": dict(conclusions),
        "source_config": dict(source_config),
        "audit_config": dict(audit_config),
        "audit_config_sha256": str(audit_config_sha256),
        "source_paths": dict(source_paths),
        "input_hashes": dict(input_hashes),
        "runtime": dict(runtime),
        "constraints": [
            "READ_ONLY_V07_CHECKPOINTS_Q_MANIFESTS_AND_RAW_RESULTS",
            "READ_ONLY_A0C_V01_TANGENT_RESULTS",
            "EXACT_STATE_SPECIFIC_Q_AND_STREAM_MANIFEST",
            "SAME_STREAM_RNG_AND_FUTURE_BATCHES_ACROSS_PAIRED_BRANCHES",
            "NATIVE_BF16_ACCUMULATION_CLIPPING_AND_ADAMW_STEP",
            "NO_FITTED_OR_EMPIRICAL_VARIABLE",
            "NO_OLD_RESULT_OVERWRITE",
        ],
    }


def _fmt(value) -> str:
    return "undefined" if value is None else f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    conclusions = summary["conclusions"]
    metrics = summary["metrics"]
    lines = [
        "# FR A0D Finite-Horizon Epsilon Convergence Audit",
        "",
        "## Technical summary",
        "",
        f"1. **Finite differences enter the linear region: {conclusions['linear_region']}.** {conclusions.get('linear_region_explanation', '')}",
        f"2. **Full-tangent error as epsilon decreases: {conclusions['tangent_error']}.** {conclusions.get('tangent_error_explanation', '')}",
        f"3. **H=8 retains the full-tangent advantage: {conclusions['h8']}.** {conclusions.get('h8_explanation', '')}",
        "",
        "## Smaller pulses test the same zero-pulse tangent",
        "",
        "The table is used instead of a trend chart because each curve has only three registered epsilon anchors. Errors compare the stream-mean finite difference with the unchanged A0C full-tangent prediction.",
        "",
        "| state | H | epsilon level | epsilon | chi_fd | chi_tangent | absolute error | relative error |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['state']} | {row['H']} | {row['epsilon_level']} | {_fmt(row['epsilon'])} | "
            f"{_fmt(row['chi_fd'])} | {_fmt(row['chi_tangent'])} | {_fmt(row['absolute_error'])} | "
            f"{_fmt(row['relative_error'])} |"
        )
    lines.extend([
        "",
        "## Predictive agreement by epsilon level",
        "",
        "Each row pools the three states and H=2,4,8 (nine observations). The pooled endpoint improvement is consistent with reduced finite-amplitude bias, but only 4/9 state-horizon cells improve monotonically; it does not distinguish finite-amplitude truncation from tangent-backend or omitted-dynamics error.",
        "",
        "| epsilon level | R2 | correlation | MAE | RMSE | mean relative error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for level in EPSILON_LEVELS:
        row = metrics["per_epsilon"][level]
        lines.append(
            f"| {level} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | {_fmt(row['MAE'])} | "
            f"{_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
        )
    lines.extend([
        "",
        "## Successive-halving diagnostics",
        "",
        "`fd contraction` compares |chi(epsilon/2)-chi(epsilon/4)| with |chi(epsilon)-chi(epsilon/2)|. `p` is the descriptive log2 ratio of those changes; no convergence threshold or fitted variable is introduced.",
        "",
        "| state | H | fd change full→half | fd change half→quarter | contraction ratio | p | error monotonic |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary["convergence_results"]:
        lines.append(
            f"| {row['state']} | {row['H']} | {_fmt(row['fd_change_full_to_half'])} | "
            f"{_fmt(row['fd_change_half_to_quarter'])} | {_fmt(row['fd_change_contraction_ratio'])} | "
            f"{_fmt(row['observed_central_difference_order'])} | {row['relative_error_monotonic']} |"
        )
    lines.extend([
        "",
        "## Scope, definitions, and replay integrity",
        "",
        "- Scope is the three locked A0C states, eight E2 streams per state, H=2,4,8, and epsilon multipliers 1, 1/2, 1/4.",
        "- `chi_fd=(X_plus-X_minus)/(2*epsilon_level)` is a derivative per unit post-unscale gradient pulse. The denominator changes with each epsilon level; epsilon is never multiplied into `chi_tangent`.",
        "- Full-epsilon branches are the immutable confirmatory v07 measurements. Half- and quarter-epsilon branches are new strict replays using the same checkpoint, state-specific q, manifest batches, stream RNG, BF16 autocast, two-microbatch accumulation, clipping, LR schedule, and native AdamW update.",
        "- `chi_tangent` is read unchanged from the A0C v01 mixed-backend full-tangent artifact and is identical across epsilon levels. Native flash-SDPA does not expose double backward, so its auxiliary HVP used math-SDPA; the recorded first-gradient backend gap remains a limitation, not a fitted correction.",
        f"- Input integrity: `{summary['runtime'].get('input_integrity', 'not recorded')}`; replay provenance: `{summary['runtime'].get('replay_provenance', 'not recorded')}`; full-epsilon source check: `{summary['runtime'].get('full_epsilon_source_check', 'not recorded')}`; new replay branch count: `{summary['runtime'].get('new_branch_count', 'not recorded')}`.",
        f"- The inherited math/native first-gradient gap is `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native_inherited_from_v01'))}`, comparable to the pooled epsilon/4 mean relative error. It is not an HVP-error bound, so the endpoint improvement cannot by itself rule out tangent-backend error.",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "",
        "## Limitations and robustness",
        "",
        "The audit varies only pulse amplitude while holding the registered states, directions, batches, RNG, and horizons fixed. With eight streams and three states, per-H correlation/R2 has n=3 and should be read as a consistency diagnostic, not a population estimate. Clipping-regime crossings are part of the finite-epsilon response and may make convergence non-monotonic for individual streams even when the state mean contracts.",
        "",
        "## Recommended next step",
        "",
        conclusions.get("recommended_next_step", "Use the quarter-epsilon residual as the finite-horizon local-linearity bound."),
        "",
        "## Further question",
        "",
        conclusions.get("further_question", "Do any state-stream pairs retain a non-contracting residual after epsilon/4?"),
        "",
    ])
    return "\n".join(lines)


def json_dumps(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
