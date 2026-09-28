"""Analysis primitives for the FR A0E curvature contribution decomposition.

[FR-OBSERVATION-ONLY] Existing checkpoints, q directions, manifests, v07
finite-difference results, and the frozen FR_A0C summary are read-only inputs.
[FR-NATIVE-ADAMW] All tangents follow the complete native AdamW state update,
exactly as in FR_A0C.

Model ladder (FR_A0C definitions kept frozen):

- ``A_momentum``      historical kernel ``-eta*mu_raw*(1-beta1**H)`` (FR_A0C).
- ``B_optimizer_only`` complete stepwise AdamW tangent with the pulse
  ``delta_g=q`` and zero future gradient tangent (FR_A0C implementation kept).
- ``BH_curvature_feedback`` Model B plus the single term
  ``delta_g = H(loss) delta_theta`` at every future step. No other extra
  higher-order term is added: the raw HVP enters the gradient-tangent channel
  directly, i.e. the FR_A0C clip-JVP radial correction is *not* applied.
  Wherever clipping is inactive this coincides with C; wherever clipping is
  active the BH-to-C gap measures the clipping Jacobian contribution.
- ``C_full_tangent``  FR_A0C full tangent (clip-JVP applied to the HVP
  feedback; math-SDPA HVP surrogate).

Metrics are not redefined here: ``relative_error`` and ``_metric`` are
imported from the frozen FR_A0C analysis module and used verbatim.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence

# Metric definitions are reused from the frozen FR_A0C module on purpose; A0E
# is forbidden from redefining R2/corr/MAE/RMSE/mean-relative-error semantics.
from fr.a0c_adamw_response_propagation_v01 import (
    SELECTED_STATES as A0C_SELECTED_STATES,
    _metric,
    json_dumps,
    relative_error,
)


SELECTED_STATES = tuple(A0C_SELECTED_STATES)
HORIZONS_PROP = (1, 2, 4, 8)
HORIZONS_TEST = (2, 4, 8)

PREDICTION_FIELDS = {
    "A_momentum": "chi_A_momentum",
    "B_optimizer_only": "chi_optimizer_only",
    "BH_curvature_feedback": "chi_BH_curvature",
    "C_full_tangent": "chi_full_tangent",
}

ERROR_FIELDS = {
    "A_momentum": "relative_error_A",
    "B_optimizer_only": "relative_error_optimizer_only",
    "BH_curvature_feedback": "relative_error_BH_curvature",
    "C_full_tangent": "relative_error_full_tangent",
}


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def prediction_metrics(rows: Sequence[Mapping]) -> Dict:
    """FR_A0C-style metrics for the four-model ladder.

    Uses the imported frozen ``_metric`` verbatim; aggregation layout matches
    FR_A0C (overall / per_state / per_horizon).
    """
    result = {}
    for label, field in PREDICTION_FIELDS.items():
        overall = _metric(
            [float(row["chi_fd"]) for row in rows],
            [float(row[field]) for row in rows],
        )
        state_groups = defaultdict(list)
        horizon_groups = defaultdict(list)
        for row in rows:
            state_groups[str(row["state"])].append(row)
            horizon_groups[str(int(row["H"]))].append(row)
        by_state = {
            state: _metric(
                [float(row["chi_fd"]) for row in group],
                [float(row[field]) for row in group],
            )
            for state, group in state_groups.items()
        }
        by_horizon = {
            horizon: _metric(
                [float(row["chi_fd"]) for row in group],
                [float(row[field]) for row in group],
            )
            for horizon, group in horizon_groups.items()
        }
        result[label] = {
            "prediction_field": field,
            "overall": overall,
            "per_state": by_state,
            "per_horizon": by_horizon,
        }
    return result


def row_gains(row: Mapping) -> Dict:
    """Per state-horizon decomposition gains (relative and absolute error).

    gain_optimizer = error_A - error_B      (optimizer memory contribution)
    gain_H         = error_B - error_BH     (curvature feedback contribution)
    clip_correction = error_BH - error_C    (remaining clip-JVP term; the
                                             difference between the raw-HVP BH
                                             model and the FR_A0C full tangent)
    """
    rel_a = _finite(row["relative_error_A"], "relative_error_A")
    rel_b = _finite(row["relative_error_optimizer_only"], "relative_error_B")
    rel_bh = _finite(row["relative_error_BH_curvature"], "relative_error_BH")
    rel_c = _finite(row["relative_error_full_tangent"], "relative_error_C")
    abs_a = _finite(row["absolute_error_A"], "absolute_error_A")
    abs_b = _finite(row["absolute_error_optimizer_only"], "absolute_error_B")
    abs_bh = _finite(row["absolute_error_BH_curvature"], "absolute_error_BH")
    abs_c = _finite(row["absolute_error_full_tangent"], "absolute_error_C")
    return {
        "gain_optimizer_rel": rel_a - rel_b,
        "gain_H_rel": rel_b - rel_bh,
        "clip_correction_rel": rel_bh - rel_c,
        "gain_optimizer_abs": abs_a - abs_b,
        "gain_H_abs": abs_b - abs_bh,
        "clip_correction_abs": abs_bh - abs_c,
    }


def aggregate_gains(rows: Sequence[Mapping]) -> Dict:
    """Aggregate decomposition gains over the locked H in {2,4,8} rows."""
    test_rows = [row for row in rows if int(row["H"]) in HORIZONS_TEST]
    if len(test_rows) != len(SELECTED_STATES) * len(HORIZONS_TEST):
        raise ValueError("gain aggregation requires the locked 3x3 test grain")

    def mean(values) -> float:
        values = [float(value) for value in values]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("cannot average empty or non-finite gains")
        return float(sum(values) / len(values))

    per_row = []
    for row in test_rows:
        gains = row_gains(row)
        per_row.append(
            {
                "state": str(row["state"]),
                "H": int(row["H"]),
                **gains,
            }
        )
    by_horizon = {}
    for horizon in HORIZONS_TEST:
        group = [entry for entry in per_row if entry["H"] == horizon]
        by_horizon[str(horizon)] = {
            "gain_optimizer_rel": mean(e["gain_optimizer_rel"] for e in group),
            "gain_H_rel": mean(e["gain_H_rel"] for e in group),
            "clip_correction_rel": mean(e["clip_correction_rel"] for e in group),
            "gain_optimizer_abs": mean(e["gain_optimizer_abs"] for e in group),
            "gain_H_abs": mean(e["gain_H_abs"] for e in group),
            "clip_correction_abs": mean(e["clip_correction_abs"] for e in group),
        }
    return {
        "per_row": per_row,
        "by_horizon": by_horizon,
        "overall": {
            "gain_optimizer_rel": mean(e["gain_optimizer_rel"] for e in per_row),
            "gain_H_rel": mean(e["gain_H_rel"] for e in per_row),
            "clip_correction_rel": mean(e["clip_correction_rel"] for e in per_row),
            "gain_optimizer_abs": mean(e["gain_optimizer_abs"] for e in per_row),
            "gain_H_abs": mean(e["gain_H_abs"] for e in per_row),
            "clip_correction_abs": mean(e["clip_correction_abs"] for e in per_row),
            "total_rel_decomposition_A_to_C": mean(
                e["gain_optimizer_rel"] + e["gain_H_rel"] + e["clip_correction_rel"]
                for e in per_row
            ),
        },
    }


def validate_against_a0c(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    a0c_summary: Mapping, tolerance: float = 1e-9,
) -> Dict:
    """Assert the reproduced A/B/C values match the frozen FR_A0C artifact."""
    a0c_rows = {
        (int(r["seed"]), int(r["step"]), int(r["H"])): r
        for r in a0c_summary["results"]
    }
    a0c_streams = {
        (int(r["seed"]), int(r["step"]), int(r["stream_id"])): r
        for r in a0c_summary["stream_results"]
    }
    max_gaps = defaultdict(float)
    checked = {"rows": 0, "streams": 0}
    for row in result_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        old = a0c_rows.get(key)
        if old is None:
            raise ValueError(f"A0C artifact missing row {key}")
        for field in ("chi_A_momentum", "chi_optimizer_only", "chi_full_tangent"):
            gap = abs(float(row[field]) - float(old[field]))
            max_gaps[field] = max(max_gaps[field], gap)
            if gap > tolerance * max(1.0, abs(float(old[field]))):
                raise ValueError(
                    f"A0C reproduction mismatch {key} {field}: {row[field]} vs {old[field]}"
                )
        checked["rows"] += 1
    for stream in stream_results:
        key = (int(stream["seed"]), int(stream["step"]), int(stream["stream_id"]))
        old = a0c_streams.get(key)
        if old is None:
            raise ValueError(f"A0C artifact missing stream {key}")
        for horizon in HORIZONS_PROP:
            hkey = str(horizon)
            for field in ("chi_optimizer_only", "chi_full_tangent"):
                gap = abs(
                    float(stream[field][hkey]) - float(old[field][hkey])
                )
                max_gaps[f"stream:{field}"] = max(max_gaps[f"stream:{field}"], gap)
                if gap > tolerance * max(1.0, abs(float(old[field][hkey]))):
                    raise ValueError(
                        f"A0C stream reproduction mismatch {key} H={horizon} {field}"
                    )
        checked["streams"] += 1
    return {
        "pass": True,
        "tolerance": float(tolerance),
        "max_abs_gap_by_field": dict(max_gaps),
        "rows_checked": int(checked["rows"]),
        "streams_checked": int(checked["streams"]),
    }


def assemble_summary(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    metrics: Mapping, gains: Mapping, a0c_crosscheck: Mapping,
    source_paths: Mapping, input_hashes: Mapping, runtime: Mapping,
    conclusions: Mapping,
) -> Dict:
    expected_rows = {
        (seed, step, horizon)
        for seed, step in SELECTED_STATES for horizon in HORIZONS_PROP
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
    required_models = set(PREDICTION_FIELDS)
    if set(metrics) != required_models:
        raise ValueError("metrics must contain exactly models A/B/BH/C")
    return {
        "protocol": "FR_A0E_CURVATURE_DECOMPOSITION_V01",
        "status": "COMPLETE",
        "scope": {
            "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
            "horizons_propagated": list(HORIZONS_PROP),
            "horizons_tested": list(HORIZONS_TEST),
            "streams_per_state": 8,
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_registered)]",
        "model_definitions": {
            "A_momentum": "-eta_first_future_step*mu_raw*(1-beta1**H) [FR_A0C]",
            "B_optimizer_only": "complete stepwise AdamW tangent; pulse delta_g=q then future delta_g=0 [FR_A0C]",
            "BH_curvature_feedback": (
                "Model B plus the single future term delta_g=H_math_SDPA(loss)*delta_theta; "
                "raw HVP enters the gradient-tangent channel directly without the clip-JVP "
                "radial correction and without any other higher-order term"
            ),
            "C_full_tangent": "complete stepwise AdamW tangent; future delta_g=clip_jvp(H(loss)*delta_theta) [FR_A0C]",
        },
        "gain_definitions": {
            "gain_optimizer": "error_A - error_B",
            "gain_H": "error_B - error_BH",
            "clip_correction": "error_BH - error_C (diagnostic; not part of the requested gains)",
        },
        "tangent_units": "DERIVATIVE_PER_UNIT_POST_UNSCALE_GRADIENT_PULSE",
        "fit_parameters": [],
        "result_row_count": len(result_rows),
        "stream_count": len(stream_results),
        "results": list(result_rows),
        "stream_results": list(stream_results),
        "metrics": dict(metrics),
        "gains": dict(gains),
        "a0c_crosscheck": dict(a0c_crosscheck),
        "conclusions": dict(conclusions),
        "source_paths": dict(source_paths),
        "input_hashes": dict(input_hashes),
        "runtime": dict(runtime),
        "constraints": [
            "EXACT_V07_STATE_Q_CHECKPOINT_MANIFEST_STREAM",
            "REAL_BASELINE_FUTURE_BATCH_AND_RNG",
            "NATIVE_MICROBATCH_LR_AND_ACCUMULATION",
            "COMPLETE_ADAMW_M_V_BIAS_CORRECTION_EPS_WEIGHT_DECAY",
            "STEPWISE_GLOBAL_CLIP_JACOBIAN_FOR_C",
            "BH_USES_RAW_HVP_FEEDBACK_WITHOUT_CLIP_JVP",
            "FULL_TANGENT_USES_REAL_LOSS_HVP",
            "NO_FITTED_OR_EMPIRICAL_VARIABLE",
            "NO_OLD_INPUT_MUTATION",
            "METRICS_REUSED_FROM_FR_A0C_MODULE",
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    conclusions = summary["conclusions"]
    metrics = summary["metrics"]
    gains = summary["gains"]
    lines = [
        "# FR A0E Curvature Contribution Decomposition v01",
        "",
        "## Technical summary",
        "",
        f"1. **Curvature feedback contribution (gain_H = error_B - error_BH)："
        f"{conclusions['curvature_contribution']}。** "
        f"{conclusions.get('curvature_explanation', '')}",
        f"2. **Optimizer memory contribution (gain_optimizer = error_A - error_B)："
        f"{conclusions['optimizer_contribution']}。** "
        f"{conclusions.get('optimizer_explanation', '')}",
        f"3. **BH 与 FR_A0C full tangent 的差距（clip-JVP 修正）："
        f"{conclusions['bh_vs_c_gap']}。** "
        f"{conclusions.get('bh_vs_c_explanation', '')}",
        "",
        "## Model ladder",
        "",
        "- **A_momentum** — historical kernel `-eta*mu_raw*(1-beta1**H)` (FR_A0C, unchanged).",
        "- **B_optimizer_only** — FR_A0C complete stepwise AdamW tangent; pulse `delta_g=q`, future `delta_g=0`.",
        "- **BH_curvature_feedback** — Model B plus the single future term `delta_g = H(loss) delta_theta` (raw math-SDPA HVP; no clip-JVP radial correction, no other higher-order term).",
        "- **C_full_tangent** — FR_A0C full tangent: the HVP feedback passes through the exact global-clip Jacobian.",
        "",
        "Wherever clipping is inactive BH and C coincide by construction; the BH-to-C gap therefore isolates the clipping Jacobian contribution.",
        "",
        "## Per state-horizon error table (three decomposition models + A reference)",
        "",
        "`rel X` is absolute prediction error divided by `|chi_fd|`. `gain_optimizer = relA - relB`, `gain_H = relB - relBH`.",
        "",
        "| state | H | chi_fd | A | B | BH | C | rel A | rel B | rel BH | rel C | gain_optimizer | gain_H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        if int(row["H"]) not in HORIZONS_TEST:
            continue
        gain = next(
            entry for entry in gains["per_row"]
            if entry["state"] == row["state"] and entry["H"] == int(row["H"])
        )
        lines.append(
            f"| {row['state']} | {row['H']} | {_fmt(row['chi_fd'])} | "
            f"{_fmt(row['chi_A_momentum'])} | {_fmt(row['chi_optimizer_only'])} | "
            f"{_fmt(row['chi_BH_curvature'])} | {_fmt(row['chi_full_tangent'])} | "
            f"{_fmt(row['relative_error_A'])} | {_fmt(row['relative_error_optimizer_only'])} | "
            f"{_fmt(row['relative_error_BH_curvature'])} | "
            f"{_fmt(row['relative_error_full_tangent'])} | "
            f"{_fmt(gain['gain_optimizer_rel'])} | {_fmt(gain['gain_H_rel'])} |"
        )
    lines.extend([
        "",
        "## Aggregate metrics over the nine H in {2,4,8} rows",
        "",
        "Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim); layout matches FR_A0C.",
        "",
        "| model | overall R2 | corr | MAE | RMSE | mean relative error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label in PREDICTION_FIELDS:
        row = metrics[label]["overall"]
        lines.append(
            f"| {label} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | "
            f"{_fmt(row['MAE'])} | {_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
        )
    lines.extend([
        "",
        "## Contribution decomposition",
        "",
        "| quantity | mean over H in {2,4,8} | H=2 | H=4 | H=8 |",
        "|---|---:|---:|---:|---:|",
    ])
    overall = gains["overall"]
    for key, label in (
        ("gain_optimizer_rel", "gain_optimizer = relA - relB"),
        ("gain_H_rel", "gain_H = relB - relBH"),
        ("clip_correction_rel", "clip_correction = relBH - relC (diagnostic)"),
    ):
        lines.append(
            f"| {label} | {_fmt(overall[key])} | "
            f"{_fmt(gains['by_horizon']['2'][key])} | "
            f"{_fmt(gains['by_horizon']['4'][key])} | "
            f"{_fmt(gains['by_horizon']['8'][key])} |"
        )
    lines.append(
        f"| total A-to-C mean relative error reduction | "
        f"{_fmt(overall['total_rel_decomposition_A_to_C'])} | | | |"
    )
    lines.extend([
        "",
        "A positive gain means the later model has smaller error. The three relative-error gains sum exactly to the A-to-C mean relative-error reduction, so the decomposition is complete by construction.",
        "",
        "## Absolute-error decomposition",
        "",
        "| quantity | mean over H in {2,4,8} |",
        "|---|---:|",
    ])
    for key, label in (
        ("gain_optimizer_abs", "gain_optimizer (abs error)"),
        ("gain_H_abs", "gain_H (abs error)"),
        ("clip_correction_abs", "clip_correction (abs error)"),
    ):
        lines.append(f"| {label} | {_fmt(overall[key])} |")
    lines.extend([
        "",
        "## Consistency with the frozen FR_A0C artifact",
        "",
        f"- Reproduced A/B/C row values match `response_propagation_v01/summary.json`: "
        f"`{summary['a0c_crosscheck']['pass']}`; rows checked: "
        f"{summary['a0c_crosscheck']['rows_checked']}, streams checked: "
        f"{summary['a0c_crosscheck']['streams_checked']}.",
        f"- Max absolute reproduction gap by field: "
        + ", ".join(
            f"{field}={_fmt(gap)}"
            for field, gap in sorted(summary["a0c_crosscheck"]["max_abs_gap_by_field"].items())
        )
        + ".",
        "",
        "## Scope and method",
        "",
        "- Same three locked v07 states, eight paired E2 streams per state, and propagation horizons H=1,2,4,8 as FR_A0C; the requested test set is H in {2,4,8} (nine state-horizon rows).",
        "- BH shares the complete AdamW m/v/bias-correction/weight-decay tangent with B; its only addition is the future gradient feedback `delta_g = H(loss) delta_theta` evaluated on the same accumulated loss as C (reverse-over-reverse autograd, math-SDPA HVP forward, RNG snapshot/restore around the auxiliary replay).",
        "- At H=1 all tangent models reduce to the same pulse Jacobian, so H=1 is excluded from the gain aggregates (it is retained in `results` for continuity with FR_A0C).",
        "",
        "## Robustness, replay integrity, and limitations",
        "",
        f"- Baseline replay: `{summary['runtime'].get('baseline_replay', 'not recorded')}`; max saved-chi aggregation gap: `{_fmt(summary['runtime'].get('max_saved_chi_abs_gap'))}`.",
        f"- Input integrity: `{summary['runtime'].get('input_integrity', 'not recorded')}`; HVP backend: `{summary['runtime'].get('hvp_backend', 'not recorded')}`.",
        f"- Max math-SDPA HVP-forward gradient relative gap vs native flash: `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native'))}`; this does not bound the HVP error and qualifies the smallest residuals, as in FR_A0C.",
        f"- Clipping was active on {summary['runtime'].get('clip_active_steps_h_gt_1', 'unknown')}/{summary['runtime'].get('total_steps_h_gt_1', 'unknown')} future steps (H>1); on all inactive steps BH and C coincide exactly.",
        f"- Max |BH - C| response gap at H=8 across streams: `{_fmt(summary['runtime'].get('max_abs_BH_minus_C_H8'))}`.",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "- As in FR_A0C, all tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; remaining residuals may include finite-epsilon clip/activation boundary crossings.",
        "",
        "## Recommended next step",
        "",
        conclusions.get("recommended_next_step", "Project the BH and C residuals against candidate omitted state variables before fitting any memory kernel."),
        "",
        "## Further question",
        "",
        conclusions.get("further_question", "Does the BH-to-C clipping gap grow with horizon when clipping is active at several consecutive future steps?"),
    ])
    return "\n".join(lines)
