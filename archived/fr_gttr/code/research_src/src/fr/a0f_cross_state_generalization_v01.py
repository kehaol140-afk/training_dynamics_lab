"""Analysis primitives for the FR A0F cross-state generalization audit.

[FR-OBSERVATION-ONLY] The v07 multiseed state bank (8 seeds x 7 steps), its
checkpoints, q directions, manifests, and finite-difference theta responses
are read-only inputs. Model definitions are exactly FR_A0C's A/B/C; this
experiment changes only the set of states being tested.

Sampling is deterministic: for each seed, an independent RNG seeded with
``SAMPLE_SEED + seed`` draws ``STATES_PER_SEED`` training steps, so the exact
state list is reproducible without storing generated data.

Metrics are not redefined here: ``relative_error`` and ``_metric`` are
imported from the frozen FR_A0C analysis module and used verbatim.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import numpy as np

# Metric definitions are reused from the frozen FR_A0C module on purpose; A0F
# is forbidden from redefining R2/corr/MAE/RMSE/mean-relative-error semantics.
from fr.a0c_adamw_response_propagation_v01 import (
    _metric,
    json_dumps,
    relative_error,
)


SEEDS = (2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037)
STEPS = (46, 52, 58, 64, 70, 76, 82)
SAMPLE_SEED = 20260911
STATES_PER_SEED = 4
HORIZONS = (1, 2, 4, 8)

PREDICTION_FIELDS = {
    "A_momentum": "chi_A_momentum",
    "B_optimizer_only": "chi_optimizer_only",
    "C_full_tangent": "chi_full_tangent",
}


def sample_states(
    *, states_per_seed: int = STATES_PER_SEED, sample_seed: int = SAMPLE_SEED,
) -> tuple[tuple[int, int], ...]:
    """Deterministically sample states covering every seed and stage range."""
    if not 1 <= states_per_seed <= len(STEPS):
        raise ValueError("states_per_seed must be within the available steps")
    selected = {}
    for seed in SEEDS:
        rng = random.Random(int(sample_seed) + int(seed))
        selected[seed] = sorted(rng.sample(list(STEPS), int(states_per_seed)))
    return tuple((seed, step) for seed in SEEDS for step in selected[seed])


def distribution_stats(values: Sequence[float]) -> Dict:
    """Mean / std (ddof=1) / median plus extremes; no fitted thresholds."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("distribution stats need at least two finite values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _require_error(value, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is undefined (zero observed response)")
    return _finite(value, label)


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def prediction_metrics(rows: Sequence[Mapping]) -> Dict:
    """FR_A0C-style metrics for A/B/C (imported frozen ``_metric`` verbatim)."""
    result = {}
    for label, field in PREDICTION_FIELDS.items():
        overall = _metric(
            [float(row["chi_fd"]) for row in rows],
            [float(row[field]) for row in rows],
        )
        horizon_groups = defaultdict(list)
        for row in rows:
            horizon_groups[str(int(row["H"]))].append(row)
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
            "per_horizon": by_horizon,
        }
    return result


def error_distributions(rows: Sequence[Mapping]) -> Dict:
    """Per model per horizon distribution of per-state errors."""
    result = {}
    for model_label, rel_field, abs_field in (
        ("A_momentum", "relative_error_A", "absolute_error_A"),
        ("B_optimizer_only", "relative_error_optimizer_only", "absolute_error_optimizer_only"),
        ("C_full_tangent", "relative_error_full_tangent", "absolute_error_full_tangent"),
    ):
        by_horizon = {}
        for horizon in HORIZONS:
            group = [row for row in rows if int(row["H"]) == horizon]
            if len(group) < 2:
                raise ValueError("need at least two states per horizon")
            by_horizon[str(horizon)] = {
                "relative_error": distribution_stats(
                    _require_error(row[rel_field], rel_field) for row in group
                ),
                "absolute_error": distribution_stats(float(row[abs_field]) for row in group),
            }
        result[model_label] = by_horizon
    return result


def paired_b_c_comparison(rows: Sequence[Mapping]) -> Dict:
    """Paired per-state B-vs-C comparison at every horizon and overall.

    improvement = error_B - error_C; positive means the full tangent wins.
    """
    def summarize(pairs) -> Dict:
        improvements = [float(pair[0] - pair[1]) for pair in pairs]
        wins = sum(value > 0.0 for value in improvements)
        losses = sum(value < 0.0 for value in improvements)
        return {
            "distribution": distribution_stats(improvements),
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(len(improvements) - wins - losses),
            "win_rate": float(wins / len(improvements)),
            "worst_regression": float(min(improvements)),
            "best_improvement": float(max(improvements)),
        }

    def rel(row, field):
        return _require_error(row[field], field)

    per_horizon = {}
    for horizon in HORIZONS:
        group = [row for row in rows if int(row["H"]) == horizon]
        per_horizon[str(horizon)] = {
            "relative_error": summarize([
                (rel(row, "relative_error_optimizer_only"), rel(row, "relative_error_full_tangent"))
                for row in group
            ]),
            "absolute_error": summarize([
                (float(row["absolute_error_optimizer_only"]), float(row["absolute_error_full_tangent"]))
                for row in group
            ]),
        }
    overall = {
        "relative_error": summarize([
            (rel(row, "relative_error_optimizer_only"), rel(row, "relative_error_full_tangent"))
            for row in rows
        ]),
        "absolute_error": summarize([
            (float(row["absolute_error_optimizer_only"]), float(row["absolute_error_full_tangent"]))
            for row in rows
        ]),
    }
    by_seed = {}
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        group = [row for row in rows if int(row["seed"]) == seed]
        by_seed[str(seed)] = {
            "mean_relative_error_B": float(np.mean([rel(row, "relative_error_optimizer_only") for row in group])),
            "mean_relative_error_C": float(np.mean([rel(row, "relative_error_full_tangent") for row in group])),
            "mean_relative_improvement_B_to_C": float(np.mean([
                rel(row, "relative_error_optimizer_only") - rel(row, "relative_error_full_tangent")
                for row in group
            ])),
            "C_beats_B_rows": int(sum(
                rel(row, "relative_error_full_tangent") < rel(row, "relative_error_optimizer_only")
                for row in group
            )),
            "row_count": int(len(group)),
        }
    return {"per_horizon": per_horizon, "overall": overall, "by_seed": by_seed}


def assemble_summary(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    states: Sequence[tuple[int, int]], metrics: Mapping, distributions: Mapping,
    paired: Mapping, source_paths: Mapping, input_hashes: Mapping,
    runtime: Mapping, conclusions: Mapping,
) -> Dict:
    expected_rows = {
        (seed, step, horizon)
        for seed, step in states for horizon in HORIZONS
    }
    actual_rows = {
        (int(row["seed"]), int(row["step"]), int(row["H"]))
        for row in result_rows
    }
    if len(result_rows) != len(expected_rows) or actual_rows != expected_rows:
        raise ValueError("result rows do not exactly match the sampled state grain")
    expected_streams = {
        (seed, step, stream_id)
        for seed, step in states for stream_id in range(8)
    }
    actual_streams = {
        (int(row["seed"]), int(row["step"]), int(row["stream_id"]))
        for row in stream_results
    }
    if len(stream_results) != len(expected_streams) or actual_streams != expected_streams:
        raise ValueError("stream rows do not exactly match the sampled state grain")
    required_models = set(PREDICTION_FIELDS)
    if set(metrics) != required_models:
        raise ValueError("metrics must contain exactly models A/B/C")
    return {
        "protocol": "FR_A0F_CROSS_STATE_GENERALIZATION_V01",
        "status": "COMPLETE",
        "scope": {
            "seeds": sorted({seed for seed, _ in states}),
            "available_steps": list(STEPS),
            "states": [{"seed": seed, "step": step} for seed, step in states],
            "horizons": list(HORIZONS),
            "streams_per_state": 8,
            "sampling": {
                "scheme": "per_seed_independent_rng",
                "sample_seed": int(SAMPLE_SEED),
                "states_per_seed": int(STATES_PER_SEED),
            },
        },
        "chi_definition": "mean_stream[(X_plus-X_minus)/(2*epsilon_registered)]",
        "model_definitions": {
            "A_momentum": "-eta_first_future_step*mu_raw*(1-beta1**H) [FR_A0C]",
            "B_optimizer_only": "complete stepwise AdamW tangent; pulse delta_g=q then future delta_g=0 [FR_A0C]",
            "C_full_tangent": "complete stepwise AdamW tangent; future delta_g=clip_jvp(H(loss)*delta_theta) [FR_A0C]",
        },
        "tangent_units": "DERIVATIVE_PER_UNIT_POST_UNSCALE_GRADIENT_PULSE",
        "fit_parameters": [],
        "result_row_count": len(result_rows),
        "stream_count": len(stream_results),
        "results": list(result_rows),
        "stream_results": list(stream_results),
        "metrics": dict(metrics),
        "error_distributions": dict(distributions),
        "paired_b_c": dict(paired),
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
    dist = summary["error_distributions"]
    paired = summary["paired_b_c"]
    lines = [
        "# FR A0F Cross-State Generalization v01",
        "",
        "## Technical summary",
        "",
        f"1. **Full tangent 是否稳定优于 optimizer-only：{conclusions['c_vs_b_stability']}。** "
        f"{conclusions.get('c_vs_b_explanation', '')}",
        f"2. **优势是否跨 seed、跨训练阶段一致：{conclusions['consistency_across_states']}。** "
        f"{conclusions.get('consistency_explanation', '')}",
        f"3. **A/B/C 三层模型的总体排序：{conclusions['model_ordering']}。** "
        f"{conclusions.get('ordering_explanation', '')}",
        "",
        "## Sampled states",
        "",
        "Deterministic sampling: per-seed independent RNG seeded with "
        f"`{summary['scope']['sampling']['sample_seed']} + seed`, "
        f"{summary['scope']['sampling']['states_per_seed']} of "
        f"{len(summary['scope']['available_steps'])} steps per seed; "
        f"{len(summary['scope']['states'])} states total, "
        f"{len(summary['scope']['seeds'])} seeds.",
        "",
        "| seed | sampled steps |",
        "|---:|---|",
    ]
    by_seed = defaultdict(list)
    for state in summary["scope"]["states"]:
        by_seed[int(state["seed"])].append(int(state["step"]))
    for seed in sorted(by_seed):
        lines.append(
            f"| {seed} | {', '.join(str(step) for step in sorted(by_seed[seed]))} |"
        )
    lines.extend([
        "",
        "## Per-horizon metrics over all sampled states",
        "",
        "Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim). "
        "Each horizon pools the per-state stream-mean responses of all sampled states.",
        "",
        "| horizon | model | R2 | corr | MAE | RMSE | mean relative error |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for horizon in summary["scope"]["horizons"]:
        for label in PREDICTION_FIELDS:
            row = metrics[label]["per_horizon"][str(horizon)]
            lines.append(
                f"| {horizon} | {label} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | "
                f"{_fmt(row['MAE'])} | {_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
            )
    lines.extend([
        "",
        "## Per-state relative error distribution (mean / std / median across states)",
        "",
        "| horizon | model | mean | std | median | min | max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for horizon in summary["scope"]["horizons"]:
        for label in PREDICTION_FIELDS:
            stats = dist[label][str(horizon)]["relative_error"]
            lines.append(
                f"| {horizon} | {label} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
                f"{_fmt(stats['median'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
            )
    lines.extend([
        "",
        "## Paired B-vs-C comparison per state (improvement = relErr_B - relErr_C)",
        "",
        "A positive improvement means the full tangent C predicts that state's response better than optimizer-only B. H=1 has B == C by construction (no future step, hence no curvature feedback), so it serves as a zero-control.",
        "",
        "| horizon | mean improvement | std | median | wins | ties | losses | win rate | worst regression | best improvement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in summary["scope"]["horizons"]:
        entry = paired["per_horizon"][str(horizon)]["relative_error"]
        stats = entry["distribution"]
        lines.append(
            f"| {horizon} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | {_fmt(stats['median'])} | "
            f"{entry['wins']} | {entry['ties']} | {entry['losses']} | {_fmt(entry['win_rate'])} | "
            f"{_fmt(entry['worst_regression'])} | {_fmt(entry['best_improvement'])} |"
        )
    overall = paired["overall"]["relative_error"]
    ostats = overall["distribution"]
    lines.append(
        f"| **all** | {_fmt(ostats['mean'])} | {_fmt(ostats['std'])} | {_fmt(ostats['median'])} | "
        f"{overall['wins']} | {overall['ties']} | {overall['losses']} | {_fmt(overall['win_rate'])} | "
        f"{_fmt(overall['worst_regression'])} | {_fmt(overall['best_improvement'])} |"
    )
    lines.extend([
        "",
        "## Per-seed summary (mean relative error, paired B-to-C)",
        "",
        "| seed | mean rel err B | mean rel err C | mean improvement | C-beats-B rows | rows |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in sorted(paired["by_seed"], key=int):
        entry = paired["by_seed"][seed]
        lines.append(
            f"| {seed} | {_fmt(entry['mean_relative_error_B'])} | "
            f"{_fmt(entry['mean_relative_error_C'])} | "
            f"{_fmt(entry['mean_relative_improvement_B_to_C'])} | "
            f"{entry['C_beats_B_rows']} | {entry['row_count']} |"
        )
    lines.extend([
        "",
        "## Scope and method",
        "",
        "- Same replay protocol, horizons H=1,2,4,8, eight paired E2 streams per state, and frozen FR_A0C A/B/C definitions.",
        "- Model A uses each state's registered first-future-step LR (`eta`) from the frozen `mv_theory_consistent` artifact.",
        "- Every stream restores the exact checkpoint, registered RNG, and registered future batches; whole-state hashes of the propagated baseline are compared against a fresh existing-protocol H=8 shadow branch.",
        "",
        "## Robustness, replay integrity, and limitations",
        "",
        f"- Baseline replay: `{summary['runtime'].get('baseline_replay', 'not recorded')}` over {summary['runtime'].get('stream_count', 'unknown')} streams.",
        f"- Max math-SDPA HVP-forward gradient relative gap vs native flash: `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native'))}` (does not bound the HVP error; qualifies C's smallest residuals, as in FR_A0C).",
        f"- Clipping was active on {summary['runtime'].get('clip_active_steps_h_gt_1', 'unknown')}/{summary['runtime'].get('total_steps_h_gt_1', 'unknown')} future steps (H>1).",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "- All tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; per-state residuals may include finite-epsilon clip/activation boundary crossings, so the stability claim is about the tangent model class, not about exactness at any single state.",
        "",
        "## Recommended next step",
        "",
        conclusions.get("recommended_next_step", "Extend to the full 56-state bank if the sampled audit is uniformly favorable."),
        "",
        "## Further question",
        "",
        conclusions.get("further_question", "Is the minority of B-beating-C states explained by clipping-regime crossings at the registered epsilon?"),
    ])
    return "\n".join(lines)
