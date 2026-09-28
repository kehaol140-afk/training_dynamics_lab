"""Analysis primitives for the FR A0F2 cross-partition extension (v07 full bank).

[FR-OBSERVATION-ONLY] The v07 multiseed state bank, its checkpoints, q
directions, manifests, and finite-difference theta responses are read-only
inputs. Model definitions are exactly FR_A0C's A/B/C and the analysis
functions (metrics, error distributions, paired B-vs-C comparison) are
imported from the FR_A0F module — they are not redefined here.

Scope: the full v07 bank is 8 seeds x 7 steps = 56 states (a census, not a
sample). The 32-state FR_A0F random sample is a strict subset; the runner
cross-checks every shared state against the frozen FR_A0F artifact.

The v06 replication bank cannot host this comparison without generating new
observations: its theta mappings carry no per-state q files, no E2 streams,
and no batch manifests, and its locked horizon set is {1,8,32}, not
{1,2,4,8}. Extending to v06 would be a new E2/E3-grade experiment, not a
replay extension, so it is out of scope for A0F2.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence

# Analysis functions reused from FR_A0F (which itself reuses the frozen
# FR_A0C metric code); nothing is redefined here.
from fr.a0f_cross_state_generalization_v01 import (
    HORIZONS,
    SEEDS,
    STEPS,
    distribution_stats,
    error_distributions,
    paired_b_c_comparison,
    prediction_metrics,
)


def full_states() -> tuple[tuple[int, int], ...]:
    """The complete v07 census: every seed at every locked training step."""
    return tuple((seed, step) for seed in SEEDS for step in STEPS)


def validate_against_a0f(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    a0f_summary: Mapping, tolerance: float = 1e-9,
) -> Dict:
    """Assert the 32 FR_A0F sample states are reproduced bit-exactly.

    Stream-level B/C predictions and row-level A/B/C values for every state
    that appears in the frozen FR_A0F artifact must match.
    """
    a0f_rows = {
        (int(r["seed"]), int(r["step"]), int(r["H"])): r
        for r in a0f_summary["results"]
    }
    a0f_streams = {
        (int(r["seed"]), int(r["step"]), int(r["stream_id"])): r
        for r in a0f_summary["stream_results"]
    }
    max_gaps = defaultdict(float)
    checked = {"rows": 0, "streams": 0}
    for row in result_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        old = a0f_rows.get(key)
        if old is None:
            continue
        for field in ("chi_A_momentum", "chi_optimizer_only", "chi_full_tangent"):
            gap = abs(float(row[field]) - float(old[field]))
            max_gaps[f"row:{field}"] = max(max_gaps[f"row:{field}"], gap)
            if gap > tolerance * max(1.0, abs(float(old[field]))):
                raise ValueError(
                    f"A0F reproduction mismatch {key} {field}: {row[field]} vs {old[field]}"
                )
        checked["rows"] += 1
    for stream in stream_results:
        key = (int(stream["seed"]), int(stream["step"]), int(stream["stream_id"]))
        old = a0f_streams.get(key)
        if old is None:
            continue
        for horizon in HORIZONS:
            hkey = str(horizon)
            for field in ("chi_optimizer_only", "chi_full_tangent"):
                gap = abs(float(stream[field][hkey]) - float(old[field][hkey]))
                max_gaps[f"stream:{field}"] = max(max_gaps[f"stream:{field}"], gap)
                if gap > tolerance * max(1.0, abs(float(old[field][hkey]))):
                    raise ValueError(
                        f"A0F stream reproduction mismatch {key} H={horizon} {field}"
                    )
        checked["streams"] += 1
    if checked["rows"] == 0 or checked["streams"] == 0:
        raise ValueError("no shared states found between A0F2 and the frozen A0F artifact")
    return {
        "pass": True,
        "tolerance": float(tolerance),
        "max_abs_gap_by_field": dict(max_gaps),
        "rows_checked": int(checked["rows"]),
        "streams_checked": int(checked["streams"]),
    }


def assemble_summary(
    *, result_rows: Sequence[Mapping], stream_results: Sequence[Mapping],
    metrics: Mapping, distributions: Mapping, paired: Mapping,
    a0f_crosscheck: Mapping, source_paths: Mapping, input_hashes: Mapping,
    runtime: Mapping, conclusions: Mapping,
) -> Dict:
    states = full_states()
    expected_rows = {
        (seed, step, horizon)
        for seed, step in states for horizon in HORIZONS
    }
    actual_rows = {
        (int(row["seed"]), int(row["step"]), int(row["H"]))
        for row in result_rows
    }
    if len(result_rows) != len(expected_rows) or actual_rows != expected_rows:
        raise ValueError("result rows do not exactly match the 56-state census grain")
    expected_streams = {
        (seed, step, stream_id)
        for seed, step in states for stream_id in range(8)
    }
    actual_streams = {
        (int(row["seed"]), int(row["step"]), int(row["stream_id"]))
        for row in stream_results
    }
    if len(stream_results) != len(expected_streams) or actual_streams != expected_streams:
        raise ValueError("stream rows do not exactly match the 56-state census grain")
    return {
        "protocol": "FR_A0F2_CROSS_PARTITION_EXTENSION_V01",
        "status": "COMPLETE",
        "scope": {
            "seeds": list(SEEDS),
            "steps": list(STEPS),
            "states": [{"seed": seed, "step": step} for seed, step in states],
            "horizons": list(HORIZONS),
            "streams_per_state": 8,
            "sampling": {
                "scheme": "full_census_of_v07_bank",
                "extends": "FR_A0F_CROSS_STATE_GENERALIZATION_V01",
            },
            "v06_note": (
                "v06 replication bank lacks per-state q, E2 streams, and manifests "
                "(locked horizons {1,8,32}); a v06 comparison requires generating new "
                "E2/E3 observations and is out of scope for this replay extension."
            ),
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
        "a0f_crosscheck": dict(a0f_crosscheck),
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
            "ANALYSIS_REUSED_FROM_FR_A0F_MODULE",
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
        "# FR A0F2 Cross-Partition Extension v01 (Full v07 Bank)",
        "",
        "## Technical summary",
        "",
        f"1. **Full tangent 在全库 56 状态上是否稳定优于 optimizer-only：{conclusions['c_vs_b_stability']}。** "
        f"{conclusions.get('c_vs_b_explanation', '')}",
        f"2. **与 FR_A0F 32 状态随机抽样结果的一致性：{conclusions['a0f_consistency']}。** "
        f"{conclusions.get('a0f_consistency_explanation', '')}",
        f"3. **跨训练阶段的一致性：{conclusions['stage_consistency']}。** "
        f"{conclusions.get('stage_explanation', '')}",
        "",
        "## Coverage",
        "",
        "Full census of the v07 confirmatory bank: 8 seeds x 7 steps (46-82) = 56 states, "
        "8 paired E2 streams per state (448 streams total). The FR_A0F 32-state random "
        "sample is a strict subset and is reproduced and cross-checked inside this run.",
        "",
        "| seed | steps |",
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
        "The v06 replication bank is not included: its theta mappings carry no per-state q "
        "files, no E2 streams, and no batch manifests, and its locked horizon set is "
        "{1,8,32}; hosting the A/B/C comparison there requires generating new E2/E3 "
        "observations rather than replaying registered ones.",
        "",
        "## Per-horizon metrics over all 56 states",
        "",
        "Metrics are computed by the frozen FR_A0C code (imported verbatim). Each horizon "
        "pools the per-state stream-mean responses of all 56 states.",
        "",
        "| horizon | model | R2 | corr | MAE | RMSE | mean relative error |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for horizon in summary["scope"]["horizons"]:
        for label in ("A_momentum", "B_optimizer_only", "C_full_tangent"):
            row = metrics[label]["per_horizon"][str(horizon)]
            lines.append(
                f"| {horizon} | {label} | {_fmt(row['R2'])} | {_fmt(row['corr'])} | "
                f"{_fmt(row['MAE'])} | {_fmt(row['RMSE'])} | {_fmt(row['mean_relative_error'])} |"
            )
    lines.extend([
        "",
        "## Per-state relative error distribution across 56 states (mean / std / median)",
        "",
        "| horizon | model | mean | std | median | min | max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for horizon in summary["scope"]["horizons"]:
        for label in ("A_momentum", "B_optimizer_only", "C_full_tangent"):
            stats = dist[label][str(horizon)]["relative_error"]
            lines.append(
                f"| {horizon} | {label} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
                f"{_fmt(stats['median'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
            )
    lines.extend([
        "",
        "## Paired B-vs-C comparison per state (improvement = relErr_B - relErr_C)",
        "",
        "A positive improvement means the full tangent C predicts that state's response "
        "better than optimizer-only B. H=1 has B == C by construction (zero-control).",
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
        "## Consistency with the frozen FR_A0F artifact",
        "",
        f"- Shared states reproduced: `{summary['a0f_crosscheck']['pass']}`; rows checked: "
        f"{summary['a0f_crosscheck']['rows_checked']}, streams checked: "
        f"{summary['a0f_crosscheck']['streams_checked']}.",
        f"- Max absolute reproduction gap by field: "
        + ", ".join(
            f"{field}={_fmt(gap)}"
            for field, gap in sorted(summary["a0f_crosscheck"]["max_abs_gap_by_field"].items())
        )
        + ".",
        "",
        "## Scope and method",
        "",
        "- Same replay protocol and frozen FR_A0C A/B/C definitions as FR_A0F; only the state set changes (census instead of random sample).",
        "- Model A uses each state's registered first-future-step LR (`eta`) from the frozen `mv_theory_consistent` artifact (covers all 56 states).",
        "",
        "## Robustness, replay integrity, and limitations",
        "",
        f"- Baseline replay: `{summary['runtime'].get('baseline_replay', 'not recorded')}` over {summary['runtime'].get('stream_count', 'unknown')} streams.",
        f"- Max math-SDPA HVP-forward gradient relative gap vs native flash: `{_fmt(summary['runtime'].get('max_hvp_math_gradient_relative_gap_vs_native'))}`.",
        f"- Clipping was active on {summary['runtime'].get('clip_active_steps_h_gt_1', 'unknown')}/{summary['runtime'].get('total_steps_h_gt_1', 'unknown')} future steps (H>1).",
        f"- Runtime: `{summary['runtime'].get('device', 'unknown')}`, AMP `{summary['runtime'].get('amp_dtype', 'unknown')}`, PyTorch `{summary['runtime'].get('torch_version', 'unknown')}`.",
        "- Tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; per-state residuals may include finite-epsilon clip/activation boundary crossings.",
        "",
        "## Recommended next step",
        "",
        conclusions.get("recommended_next_step", "Generate a v06-compatible observation protocol only if cross-partition replication is scientifically required."),
        "",
        "## Further question",
        "",
        conclusions.get("further_question", "Does the C advantage persist on a different model scale or optimizer configuration?"),
    ])
    return "\n".join(lines)
