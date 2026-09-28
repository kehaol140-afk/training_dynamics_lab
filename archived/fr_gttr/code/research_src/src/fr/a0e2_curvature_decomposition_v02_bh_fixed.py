"""Analysis primitives for FR A0E2 — BH-fixed curvature decomposition.

[FR-OBSERVATION-ONLY] Existing checkpoints, q directions, manifests, v07
finite-difference results, and the frozen FR_A0C / FR_A0E summaries are
read-only inputs.

Background (independent audit of FR_A0E): the old ``BH_curvature_feedback``
implementation fed the *raw* pulse ``q`` into the AdamW tangent at the
initial step (``bh_gradient_tangent = full_gradient_tangent`` with
``full_gradient_tangent = q_device`` at ``horizon == 1``), bypassing the
global-clip Jacobian ``J_clip(g_t) q`` that models B and C apply to the
pulse.  At H=1 there is no future HVP, so theory requires
``B(H=1) = BH(H=1) = C(H=1)``; the old artifact instead shows BH diverging
from B/C exactly on streams whose initial step activates clipping
(e.g. seed_2037/step_046 stream 2: initial preclip norm 18.27,
B=C=-0.00369 vs BH=-0.07191).

FR_A0E2 carries a corrected ladder:

- ``A_momentum``        historical kernel ``-eta*mu_raw*(1-beta1**H)`` [FR_A0C]
- ``B_optimizer_only``  complete stepwise AdamW tangent, pulse ``delta_g=q``
                        through ``J_clip(g_t)``, future ``delta_g=0`` [FR_A0C]
- ``BH_old``            frozen FR_A0E model, reproduced here bit-exactly for
                        comparison; its initial pulse bypasses ``J_clip``
- ``BH_curvature_fixed`` identical to B at the initial step (pulse through
                        the same ``J_clip(g_t)``); at future steps the raw
                        HVP ``H(loss) delta_theta`` enters *without* the
                        clip-JVP radial correction.  Differs from C only by
                        the future clip-JVP.
- ``C_full_tangent``    frozen FR_A0C full tangent (future HVP through
                        ``J_clip(g_k)``) — unchanged.

Metrics are not redefined here: ``relative_error`` and ``_metric`` are
imported from the frozen FR_A0C analysis module and used verbatim.

Two aggregation scopes are reported, each explicitly labeled:

- ``all_12``    3 states x H in {1,2,4,8}   (12 rows)
- ``h_gt_1_9``  3 states x H in {2,4,8}     (9 rows)
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping, Sequence

# Metric definitions are reused from the frozen FR_A0C module on purpose;
# A0E2 is forbidden from redefining R2/corr/MAE/RMSE/mean-relative-error.
from fr.a0c_adamw_response_propagation_v01 import (
    SELECTED_STATES as A0C_SELECTED_STATES,
    _metric,
    json_dumps,
    relative_error,
)


SELECTED_STATES = tuple(A0C_SELECTED_STATES)
HORIZONS_PROP = (1, 2, 4, 8)
HORIZONS_TEST = (2, 4, 8)

SCOPES = {
    "all_12": set(HORIZONS_PROP),
    "h_gt_1_9": set(HORIZONS_TEST),
}

PREDICTION_FIELDS = {
    "A_momentum": "chi_A_momentum",
    "B_optimizer_only": "chi_optimizer_only",
    "BH_old_curvature_feedback": "chi_BH_old_curvature",
    "BH_curvature_fixed": "chi_BH_fixed_curvature",
    "C_full_tangent": "chi_full_tangent",
}

ERROR_FIELDS = {
    "A_momentum": "relative_error_A",
    "B_optimizer_only": "relative_error_optimizer_only",
    "BH_old_curvature_feedback": "relative_error_BH_old_curvature",
    "BH_curvature_fixed": "relative_error_BH_fixed_curvature",
    "C_full_tangent": "relative_error_full_tangent",
}


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def rows_in_scope(rows: Sequence[Mapping], scope: str) -> list:
    """Return the rows of one explicitly labeled aggregation scope."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; must be one of {sorted(SCOPES)}")
    allowed = SCOPES[scope]
    selected = [row for row in rows if int(row["H"]) in allowed]
    expected = len(SELECTED_STATES) * len(allowed)
    if len(selected) != expected:
        raise ValueError(
            f"scope {scope} requires exactly {expected} rows, got {len(selected)}"
        )
    return selected


def prediction_metrics(rows: Sequence[Mapping], scope: str) -> Dict:
    """FR_A0C-style metrics for the five-model ladder in one labeled scope.

    Uses the imported frozen ``_metric`` verbatim; aggregation layout matches
    FR_A0C (overall / per_state / per_horizon).
    """
    scoped_rows = rows_in_scope(rows, scope)
    result = {}
    for label, field in PREDICTION_FIELDS.items():
        overall = _metric(
            [float(row["chi_fd"]) for row in scoped_rows],
            [float(row[field]) for row in scoped_rows],
        )
        state_groups = defaultdict(list)
        horizon_groups = defaultdict(list)
        for row in scoped_rows:
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

    gain_optimizer            = error_A - error_B
    gain_raw_curvature_fixed  = error_B - error_BH_fixed
    gain_clip_JVP             = error_BH_fixed - error_C
    total                     = error_A - error_C  (identity check target)
    """
    rel_a = _finite(row["relative_error_A"], "relative_error_A")
    rel_b = _finite(row["relative_error_optimizer_only"], "relative_error_B")
    rel_bhf = _finite(
        row["relative_error_BH_fixed_curvature"], "relative_error_BH_fixed"
    )
    rel_c = _finite(row["relative_error_full_tangent"], "relative_error_C")
    abs_a = _finite(row["absolute_error_A"], "absolute_error_A")
    abs_b = _finite(row["absolute_error_optimizer_only"], "absolute_error_B")
    abs_bhf = _finite(
        row["absolute_error_BH_fixed_curvature"], "absolute_error_BH_fixed"
    )
    abs_c = _finite(row["absolute_error_full_tangent"], "absolute_error_C")
    return {
        "gain_optimizer_rel": rel_a - rel_b,
        "gain_raw_curvature_fixed_rel": rel_b - rel_bhf,
        "gain_clip_JVP_rel": rel_bhf - rel_c,
        "total_rel_A_to_C": rel_a - rel_c,
        "gain_optimizer_abs": abs_a - abs_b,
        "gain_raw_curvature_fixed_abs": abs_b - abs_bhf,
        "gain_clip_JVP_abs": abs_bhf - abs_c,
        "total_abs_A_to_C": abs_a - abs_c,
    }


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("cannot average empty or non-finite gains")
    return float(sum(values) / len(values))


def aggregate_gains(rows: Sequence[Mapping], scope: str) -> Dict:
    """Aggregate the BH-fixed decomposition gains in one labeled scope."""
    scoped_rows = rows_in_scope(rows, scope)
    per_row = []
    for row in scoped_rows:
        gains = row_gains(row)
        per_row.append(
            {
                "state": str(row["state"]),
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "H": int(row["H"]),
                **gains,
            }
        )
    by_horizon = {}
    for horizon in sorted({entry["H"] for entry in per_row}):
        group = [entry for entry in per_row if entry["H"] == horizon]
        by_horizon[str(horizon)] = {
            key: _mean(entry[key] for entry in group)
            for key in (
                "gain_optimizer_rel",
                "gain_raw_curvature_fixed_rel",
                "gain_clip_JVP_rel",
                "total_rel_A_to_C",
                "gain_optimizer_abs",
                "gain_raw_curvature_fixed_abs",
                "gain_clip_JVP_abs",
                "total_abs_A_to_C",
            )
        }
    overall = {
        key: _mean(entry[key] for entry in per_row)
        for key in (
            "gain_optimizer_rel",
            "gain_raw_curvature_fixed_rel",
            "gain_clip_JVP_rel",
            "total_rel_A_to_C",
            "gain_optimizer_abs",
            "gain_raw_curvature_fixed_abs",
            "gain_clip_JVP_abs",
            "total_abs_A_to_C",
        )
    }
    # Exact-addition identity: the three gains must sum to the total A->C
    # reduction (float associativity may leave a last-ulp deviation).
    max_identity_deviation = 0.0
    for entry in per_row:
        summed = (
            entry["gain_optimizer_rel"]
            + entry["gain_raw_curvature_fixed_rel"]
            + entry["gain_clip_JVP_rel"]
        )
        max_identity_deviation = max(
            max_identity_deviation, abs(summed - entry["total_rel_A_to_C"])
        )
    return {
        "scope": scope,
        "row_count": len(per_row),
        "per_row": per_row,
        "by_horizon": by_horizon,
        "overall": overall,
        "identity_gain_sum_equals_total": bool(max_identity_deviation <= 1e-12),
        "max_identity_deviation": float(max_identity_deviation),
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


def validate_against_a0e(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    a0e_summary: Mapping, tolerance: float = 1e-9,
) -> Dict:
    """Assert the reproduced BH_old values match the frozen FR_A0E artifact.

    BH_old is carried in A0E2 with the exact frozen code path
    (``bh_gradient_tangent = full_gradient_tangent`` at every horizon); this
    check proves the new runner reproduces the old artifact bit-exactly.
    """
    a0e_rows = {
        (int(r["seed"]), int(r["step"]), int(r["H"])): r
        for r in a0e_summary["results"]
    }
    a0e_streams = {
        (int(r["seed"]), int(r["step"]), int(r["stream_id"])): r
        for r in a0e_summary["stream_results"]
    }
    max_gaps = defaultdict(float)
    checked = {"rows": 0, "streams": 0}
    for row in result_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        old = a0e_rows.get(key)
        if old is None:
            raise ValueError(f"A0E artifact missing row {key}")
        gap = abs(float(row["chi_BH_old_curvature"]) - float(old["chi_BH_curvature"]))
        max_gaps["chi_BH_curvature"] = max(max_gaps["chi_BH_curvature"], gap)
        if gap > tolerance * max(1.0, abs(float(old["chi_BH_curvature"]))):
            raise ValueError(
                f"A0E BH_old reproduction mismatch {key}: {row['chi_BH_old_curvature']} vs {old['chi_BH_curvature']}"
            )
        checked["rows"] += 1
    for stream in stream_results:
        key = (int(stream["seed"]), int(stream["step"]), int(stream["stream_id"]))
        old = a0e_streams.get(key)
        if old is None:
            raise ValueError(f"A0E artifact missing stream {key}")
        for horizon in HORIZONS_PROP:
            hkey = str(horizon)
            gap = abs(
                float(stream["chi_BH_old_curvature"][hkey])
                - float(old["chi_BH_curvature"][hkey])
            )
            max_gaps["stream:chi_BH_curvature"] = max(
                max_gaps["stream:chi_BH_curvature"], gap
            )
            if gap > tolerance * max(1.0, abs(float(old["chi_BH_curvature"][hkey]))):
                raise ValueError(
                    f"A0E BH_old stream reproduction mismatch {key} H={horizon}"
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
    metrics_by_scope: Mapping, gains_by_scope: Mapping,
    a0c_crosscheck: Mapping, a0e_crosscheck: Mapping,
    h1_identity: Mapping, source_paths: Mapping, input_hashes: Mapping,
    runtime: Mapping, diagnostics: Mapping, a0f_a0f2_audit: Mapping,
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
    for scope, block in metrics_by_scope.items():
        if set(block) != required_models:
            raise ValueError(f"metrics scope {scope} must contain exactly models {sorted(required_models)}")
    return {
        "protocol": "FR_A0E2_CURVATURE_DECOMPOSITION_V02_BH_FIXED",
        "status": "COMPLETE",
        "scope": {
            "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
            "horizons_propagated": list(HORIZONS_PROP),
            "horizons_tested": list(HORIZONS_TEST),
            "streams_per_state": 8,
            "aggregation_scopes": {
                key: {
                    "rows": len(SELECTED_STATES) * len(allowed),
                    "horizons": sorted(allowed),
                }
                for key, allowed in SCOPES.items()
            },
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_registered)]",
        "model_definitions": {
            "A_momentum": "-eta_first_future_step*mu_raw*(1-beta1**H) [FR_A0C]",
            "B_optimizer_only": "complete stepwise AdamW tangent; pulse delta_g=q through J_clip(g_t), future delta_g=0 [FR_A0C]",
            "BH_old_curvature_feedback": (
                "FROZEN FR_A0E model reproduced bit-exactly: raw HVP feedback at future steps "
                "AND raw pulse q at the initial step (initial pulse bypasses J_clip(g_t))"
            ),
            "BH_curvature_fixed": (
                "complete stepwise AdamW tangent; initial pulse identical to B/C "
                "(delta_g=q through J_clip(g_t)); at future steps delta_g=H(loss)*delta_theta "
                "raw HVP WITHOUT the clip-JVP radial correction; differs from C only by the "
                "future clip-JVP"
            ),
            "C_full_tangent": "complete stepwise AdamW tangent; future delta_g=clip_jvp(H(loss)*delta_theta) [FR_A0C, unchanged]",
        },
        "gain_definitions": {
            "gain_optimizer": "error_A - error_B",
            "gain_raw_curvature_fixed": "error_B - error_BH_fixed",
            "gain_clip_JVP": "error_BH_fixed - error_C",
            "total": "error_A - error_C (identity: the three gains sum to total)",
        },
        "tangent_units": "DERIVATIVE_PER_UNIT_POST_UNSCALE_GRADIENT_PULSE",
        "fit_parameters": [],
        "result_row_count": len(result_rows),
        "stream_count": len(stream_results),
        "results": list(result_rows),
        "stream_results": list(stream_results),
        "metrics_by_scope": dict(metrics_by_scope),
        "gains_by_scope": dict(gains_by_scope),
        "a0c_crosscheck": dict(a0c_crosscheck),
        "a0e_bh_old_reproduction": dict(a0e_crosscheck),
        "h1_identity_check": dict(h1_identity),
        "diagnostics": dict(diagnostics),
        "a0f_a0f2_non_interference_audit": dict(a0f_a0f2_audit),
        "conclusions": dict(conclusions),
        "source_paths": dict(source_paths),
        "input_hashes": dict(input_hashes),
        "runtime": dict(runtime),
        "constraints": [
            "EXACT_V07_STATE_Q_CHECKPOINT_MANIFEST_STREAM",
            "REAL_BASELINE_FUTURE_BATCH_AND_RNG",
            "NATIVE_MICROBATCH_LR_AND_ACCUMULATION",
            "COMPLETE_ADAMW_M_V_BIAS_CORRECTION_EPS_WEIGHT_DECAY",
            "STEPWISE_GLOBAL_CLIP_JACOBIAN_FOR_B_C_AND_BH_FIXED_PULSE",
            "BH_OLD_REPRODUCED_WITH_FROZEN_CODE_PATH",
            "BH_FIXED_PULSE_IDENTICAL_TO_B_AND_C_AT_H1",
            "BH_FIXED_FUTURE_RAW_HVP_WITHOUT_CLIP_JVP",
            "C_FULL_TANGENT_FROZEN_DEFINITION_UNCHANGED",
            "NO_FITTED_OR_EMPIRICAL_VARIABLE",
            "NO_OLD_INPUT_MUTATION",
            "METRICS_REUSED_FROM_FR_A0C_MODULE",
            "A0F_A0F2_FROZEN_ARTIFACTS_HASH_AUDITED",
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    conclusions = summary["conclusions"]
    metrics = summary["metrics_by_scope"]
    gains = summary["gains_by_scope"]
    h1 = summary["h1_identity_check"]
    diag = summary["diagnostics"]
    lines = [
        "# FR A0E2 — BH 对照修正与曲率 / clipping 机制分解 v02",
        "",
        "## Verdict 1 — 旧 BH 的 H=1 不一致根因",
        "",
        conclusions["verdict1"],
        "",
        "## Verdict 2 — 修正后 H=1 恒等性",
        "",
        conclusions["verdict2"],
        "",
        f"- max |B(H1) - BH_fixed(H1)| over 24 streams: `{_fmt(h1['max_abs_B_minus_BH_fixed_H1'])}`",
        f"- max |B(H1) - C(H1)| over 24 streams: `{_fmt(h1['max_abs_B_minus_C_H1'])}`",
        f"- max |BH_fixed(H1) - C(H1)| over 24 streams: `{_fmt(h1['max_abs_BH_fixed_minus_C_H1'])}`",
        f"- streams where BH_fixed and C tangents stay bit-identical through all 8 steps: "
        f"{diag['streams_bh_fixed_c_bit_identical_all_steps']}/24 "
        f"(first divergence step reported per stream in per_stream_diagnostics.csv).",
        "",
        "## Verdict 3 — 严格隔离 future raw curvature 后的 B → BH_fixed 方向",
        "",
        conclusions["verdict3"],
        "",
        "## Verdict 4 — A0F/A0F2 主结论不受影响",
        "",
        conclusions["verdict4"],
        "",
        "## Model ladder",
        "",
        "- **A_momentum** — historical kernel `-eta*mu_raw*(1-beta1**H)` (FR_A0C, unchanged).",
        "- **B_optimizer_only** — FR_A0C complete stepwise AdamW tangent; pulse `delta_g=q` through `J_clip(g_t)`, future `delta_g=0`.",
        "- **BH_old** — frozen FR_A0E model, reproduced bit-exactly here; its *initial pulse bypasses* `J_clip(g_t)` (raw `q` enters the AdamW tangent).",
        "- **BH_fixed** — initial pulse identical to B/C (`J_clip(g_t) q`); future steps raw HVP `H(loss) delta_theta` *without* the clip-JVP radial correction. Differs from C only by the future clip-JVP.",
        "- **C_full_tangent** — FR_A0C full tangent, unchanged.",
        "",
        "Wherever clipping is inactive at every future step, BH_fixed and C coincide bit-exactly by construction.",
        "",
        "## Per state-horizon comparison (12 rows)",
        "",
        "`rel X` = |prediction - chi_fd| / |chi_fd|. Each scope is labeled by its actual row count.",
        "",
        "| state | H | chi_fd | A | B | BH_old | BH_fixed | C | rel A | rel B | rel BH_old | rel BH_fixed | rel C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['state']} | {row['H']} | {_fmt(row['chi_fd'])} | "
            f"{_fmt(row['chi_A_momentum'])} | {_fmt(row['chi_optimizer_only'])} | "
            f"{_fmt(row['chi_BH_old_curvature'])} | {_fmt(row['chi_BH_fixed_curvature'])} | "
            f"{_fmt(row['chi_full_tangent'])} | "
            f"{_fmt(row['relative_error_A'])} | {_fmt(row['relative_error_optimizer_only'])} | "
            f"{_fmt(row['relative_error_BH_old_curvature'])} | "
            f"{_fmt(row['relative_error_BH_fixed_curvature'])} | "
            f"{_fmt(row['relative_error_full_tangent'])} |"
        )
    for scope, label in (("all_12", "ALL 12 ROWS (3 states x H in {1,2,4,8})"),
                         ("h_gt_1_9", "H>1 ONLY, 9 ROWS (3 states x H in {2,4,8})")):
        lines.extend([
            "",
            f"## Aggregate metrics — {label}",
            "",
            "Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim).",
            "",
            "| model | overall R2 | corr | MAE | RMSE | mean relative error |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for model_label in PREDICTION_FIELDS:
            row = metrics[scope][model_label]["overall"]
            lines.append(
                f"| {model_label} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | "
                f"{_fmt(row['MAE'])} | {_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
            )
        lines.extend([
            "",
            f"### Contribution decomposition — {label}",
            "",
            "| quantity | mean | " + " | ".join(
                f"H={h}" for h in sorted(gains[scope]["by_horizon"])
            ) + " |",
            "|---|---:|" + "---:|" * len(gains[scope]["by_horizon"]),
        ])
        overall = gains[scope]["overall"]
        for key, text in (
            ("gain_optimizer_rel", "gain_optimizer = relA - relB"),
            ("gain_raw_curvature_fixed_rel", "gain_raw_curvature_fixed = relB - relBH_fixed"),
            ("gain_clip_JVP_rel", "gain_clip_JVP = relBH_fixed - relC"),
            ("total_rel_A_to_C", "total = relA - relC"),
        ):
            lines.append(
                f"| {text} | {_fmt(overall[key])} | "
                + " | ".join(
                    _fmt(gains[scope]["by_horizon"][h][key])
                    for h in sorted(gains[scope]["by_horizon"])
                )
                + " |"
            )
        lines.append(
            f"| identity check: gain sum == total | "
            f"`{gains[scope]['identity_gain_sum_equals_total']}` "
            f"(max deviation {_fmt(gains[scope]['max_identity_deviation'])}) | | | |"
        )
    lines.extend([
        "",
        "## Diagnostics (24 streams)",
        "",
        f"- Streams with initial clip active: {diag['streams_initial_clip_active']}/24.",
        f"- Future clip-active steps (H>1): {diag['future_clip_active_steps']}/{diag['future_total_steps']}.",
        f"- BH_old(H1) - B(H1) nonzero streams: {diag['streams_bh_old_differs_from_B_at_H1']}/24 "
        f"(max |gap| {_fmt(diag['max_abs_BH_old_minus_B_H1'])}).",
        f"- BH_fixed(H1) - B(H1) nonzero streams: {diag['streams_bh_fixed_differs_from_B_at_H1']}/24 "
        f"(max |gap| {_fmt(diag['max_abs_BH_fixed_minus_B_H1'])}).",
        f"- max |BH_fixed - C| response gap over streams x H: {_fmt(diag['max_abs_BH_fixed_minus_C'])}.",
        f"- seed_2037/step_046 explanation: {diag['seed2037_step046_explanation']}",
        "",
        "## Consistency with frozen artifacts",
        "",
        f"- FR_A0C reproduction (A/B/C): `{summary['a0c_crosscheck']['pass']}`; max abs gap by field: "
        + ", ".join(
            f"{field}={_fmt(gap)}"
            for field, gap in sorted(summary["a0c_crosscheck"]["max_abs_gap_by_field"].items())
        ) + ".",
        f"- FR_A0E BH_old reproduction: `{summary['a0e_bh_old_reproduction']['pass']}`; max abs gap: "
        + ", ".join(
            f"{field}={_fmt(gap)}"
            for field, gap in sorted(summary["a0e_bh_old_reproduction"]["max_abs_gap_by_field"].items())
        ) + ".",
        f"- H1 propagation vs frozen H1-Jacobian artifact: max abs gap "
        f"`{_fmt(summary['runtime'].get('h1_cross_artifact_max_abs_gap'))}`.",
        "",
        "## A0F/A0F2 non-interference (read-only audit)",
        "",
        f"- Code dependency audit: A0F/A0F2 src+runner modules contain zero references to BH "
        f"(grep evidence recorded in SOURCE_FIX_AUDIT.md).",
        f"- Frozen artifact SHA256 recorded at A0E2 runtime: "
        + ", ".join(
            f"{name}={digest[:16]}…"
            for name, digest in sorted(
                summary['a0f_a0f2_non_interference_audit']['sha256_at_a0e2_runtime'].items()
            )
        )
        + f"; before/after comparison against pre-run hashes: "
        f"`{summary['a0f_a0f2_non_interference_audit']['hashes_unchanged']}` "
        f"(see SOURCE_FIX_AUDIT.md).",
        f"- A0F frozen verdict: {summary['a0f_a0f2_non_interference_audit']['a0f_frozen_wins_ties_losses'][0]} wins / "
        f"{summary['a0f_a0f2_non_interference_audit']['a0f_frozen_wins_ties_losses'][1]} H1 ties / "
        f"{summary['a0f_a0f2_non_interference_audit']['a0f_frozen_wins_ties_losses'][2]} losses over 128 rows, mean improvement "
        f"+{_fmt(summary['a0f_a0f2_non_interference_audit']['a0f_mean_improvement'])}.",
        f"- A0F2 frozen verdict: {summary['a0f_a0f2_non_interference_audit']['a0f2_frozen_wins_ties_losses'][0]} wins / "
        f"{summary['a0f_a0f2_non_interference_audit']['a0f2_frozen_wins_ties_losses'][1]} H1 ties / "
        f"{summary['a0f_a0f2_non_interference_audit']['a0f2_frozen_wins_ties_losses'][2]} losses over 224 rows, mean improvement "
        f"+{_fmt(summary['a0f_a0f2_non_interference_audit']['a0f2_mean_improvement'])}.",
        "",
        "## Scope and method",
        "",
        "- Same three locked v07 states, eight paired E2 streams per state, and propagation horizons H=1,2,4,8 as FR_A0C/FR_A0E; the test scopes are all-12-rows and the locked H>1 nine rows, each explicitly labeled.",
        "- BH_fixed shares the complete AdamW m/v/bias-correction/weight-decay tangent with B and C; its only difference from C is the absence of the clip-JVP radial correction on the future HVP feedback.",
        "- BH_old is carried with the exact frozen FR_A0E code path so the old artifact is reproduced bit-exactly inside the same run.",
        "",
        "## Robustness, replay integrity, and limitations",
        "",
        f"- Baseline replay: `{summary['runtime'].get('baseline_replay', 'not recorded')}`; max saved-chi aggregation gap: `{_fmt(summary['runtime'].get('max_saved_chi_abs_gap'))}`.",
        f"- Input integrity: `{summary['runtime'].get('input_integrity', 'not recorded')}`; HVP backend: `{summary['runtime'].get('hvp_backend', 'not recorded')}`.",
        f"- Max math-SDPA HVP-forward gradient relative gap vs native flash: `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native'))}`; as in FR_A0C this does not bound the HVP error and qualifies the smallest residuals.",
        f"- Clipping was active on {summary['runtime'].get('clip_active_steps_h_gt_1', 'unknown')}/{summary['runtime'].get('total_steps_h_gt_1', 'unknown')} future steps (H>1).",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "- As in FR_A0C/A0E, all tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; remaining residuals may include finite-epsilon clip/activation boundary crossings.",
    ])
    return "\n".join(lines)
