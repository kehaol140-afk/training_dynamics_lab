"""Strictly aligned A0C AdamW m+v response analysis.

[FR-OBSERVATION-ONLY] Reads existing A0C, v07 q/checkpoint, and saved replay
gradient artifacts. It performs no forward/backward pass and never mutates an
optimizer or checkpoint.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_mv_theory import build_metric_breakdowns, compute_mu_nu, file_sha256
from .direction import load_direction


EXPECTED_SEEDS = tuple(range(2030, 2038))
EXPECTED_HORIZONS = (1, 2, 4, 8, 12, 16, 24, 32)
EXPECTED_GRADIENT_AGGREGATION = (
    "ARITHMETIC_MEAN_OF_8_REAL_E2_PRECLIP_UNSCALED_GRADIENTS"
)


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def compute_consistent_state_features(
    *,
    q_by_name: Mapping[str, object],
    q_metadata: Mapping,
    q_sha256: str,
    expected_q_sha256: str,
    gradient_record: Mapping,
    checkpoint: Mapping,
    seed: int,
    state_step: int,
) -> Dict:
    """Validate exact state alignment and compute raw-m/raw-v projections."""
    seed = int(seed)
    state_step = int(state_step)
    if int(q_metadata.get("train_seed", -1)) != seed:
        raise ValueError(
            f"q metadata train_seed={q_metadata.get('train_seed')}, expected {seed}"
        )
    if int(q_metadata.get("state_step", -1)) != state_step:
        raise ValueError(
            f"q metadata state_step={q_metadata.get('state_step')}, expected {state_step}"
        )
    if str(q_sha256) != str(expected_q_sha256):
        raise ValueError(
            f"q direction sha256={q_sha256!r}, expected {expected_q_sha256!r}"
        )

    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("checkpoint metadata missing")
    if int(checkpoint_metadata.get("train_seed", -1)) != seed:
        raise ValueError("checkpoint train_seed does not match requested state")
    if int(checkpoint_metadata.get("state_optimizer_step", -1)) != state_step:
        raise ValueError("checkpoint state_optimizer_step does not match requested state")

    if int(gradient_record.get("seed", -1)) != seed:
        raise ValueError("gradient seed does not match requested state")
    if int(gradient_record.get("state_step", -1)) != state_step:
        raise ValueError("gradient state_step does not match requested state")
    if gradient_record.get("aggregation") != EXPECTED_GRADIENT_AGGREGATION:
        raise ValueError(f"unexpected gradient aggregation: {gradient_record.get('aggregation')!r}")
    gradient = gradient_record.get("gradient")
    if not isinstance(gradient, Mapping):
        raise ValueError("gradient record missing parameter mapping")

    streams = gradient_record.get("stream_audit")
    if not isinstance(streams, list) or len(streams) != 8:
        raise ValueError("gradient record must contain exactly eight E2 stream audits")
    eta_values = [_finite(stream.get("first_step_lr"), "first_step_lr") for stream in streams]
    if len(set(eta_values)) != 1:
        raise ValueError(f"first-step eta differs across E2 streams: {sorted(set(eta_values))}")

    features = compute_mu_nu(q_by_name, gradient, checkpoint)
    if features["beta1"] != 0.9 or features["beta2"] != 0.999:
        raise ValueError(
            f"unexpected AdamW betas: ({features['beta1']}, {features['beta2']})"
        )
    return {
        **features,
        "seed": seed,
        "state_step": state_step,
        "eta": eta_values[0],
        "q_sha256": str(q_sha256),
        "gradient_stream_count": len(streams),
    }


def predict_unit_response(
    *, eta: float, beta1: float, beta2: float, horizon: int,
    mu_q: float, nu_q: float,
) -> Dict[str, float]:
    """Predict the epsilon-normalized A0C response without another epsilon."""
    eta = _finite(eta, "eta")
    beta1 = _finite(beta1, "beta1")
    beta2 = _finite(beta2, "beta2")
    mu_q = _finite(mu_q, "mu_q")
    nu_q = _finite(nu_q, "nu_q")
    horizon = int(horizon)
    if eta < 0 or horizon <= 0:
        raise ValueError("eta must be nonnegative and horizon must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("beta1/beta2 must lie in [0, 1)")
    chi_m = -eta * mu_q * (1.0 - beta1**horizon)
    chi_v = eta * nu_q * (1.0 - beta2**horizon)
    return {
        "chi_m": float(chi_m),
        "chi_v": float(chi_v),
        "chi_mv": float(chi_m + chi_v),
    }


def assemble_consistent_summary(
    prediction_rows: Sequence[Mapping],
    *, state_results: Sequence[Mapping], source_paths: Mapping,
    previous_inconsistent_summary: Mapping | None = None,
    global_q_unit_rows: Sequence[Mapping] | None = None,
) -> Dict:
    metrics = build_metric_breakdowns(prediction_rows)
    momentum = metrics["chi_m"]["overall"]
    mv = metrics["chi_mv"]["overall"]
    gain = float(mv["R2"] - momentum["R2"])
    residuals = np.asarray(
        [float(row["chi"]) - float(row["chi_mv"]) for row in prediction_rows],
        dtype=np.float64,
    )
    seed_gains = {
        seed: float(metrics["chi_mv"]["per_seed"][seed]["R2"] - value["R2"])
        for seed, value in metrics["chi_m"]["per_seed"].items()
    }
    horizon_gains = {
        horizon: float(
            metrics["chi_mv"]["per_horizon"][horizon]["R2"] - value["R2"]
        )
        for horizon, value in metrics["chi_m"]["per_horizon"].items()
    }
    previous = None
    alignment_conclusion_changed = None
    if previous_inconsistent_summary is not None:
        previous_gain = float(previous_inconsistent_summary["comparison"]["R2_improvement"])
        previous = {
            "path": previous_inconsistent_summary.get("_source_path"),
            "R2_improvement": previous_gain,
            "v_path_improved": bool(previous_gain > 0),
        }
        alignment_conclusion_changed = bool((gain > 0) != (previous_gain > 0))
    global_q_control = None
    q_conclusion_changed = None
    if global_q_unit_rows is not None:
        global_q_metrics = build_metric_breakdowns(global_q_unit_rows)
        global_q_gain = float(
            global_q_metrics["chi_mv"]["overall"]["R2"]
            - global_q_metrics["chi_m"]["overall"]["R2"]
        )
        global_q_control = {
            "role": "ISOLATE_Q_EFFECT_AT_UNIT_RESPONSE_SCALE",
            "formula": "same chi_m/chi_mv equations without perturbation epsilon",
            "metrics": global_q_metrics,
            "R2_improvement": global_q_gain,
            "v_path_improves_R2": bool(global_q_gain > 0),
        }
        q_conclusion_changed = bool((gain > 0) != (global_q_gain > 0))
    return {
        "protocol": "FR_A0C_ADAMW_MV_THEORY_STRICT_CONSISTENCY_V01",
        "status": "COMPLETE",
        "formula": {
            "mu_q": "sum_i(q_i^2/(sqrt(exp_avg_sq_i)+adam_eps))",
            "nu_q": (
                "sum_i(q_i^2*exp_avg_i*g_t_i/"
                "(sqrt(exp_avg_sq_i)*(sqrt(exp_avg_sq_i)+adam_eps)^2))"
            ),
            "chi_m": "-eta*mu_q*(1-beta1^H)",
            "chi_mv": "-eta*mu_q*(1-beta1^H)+eta*nu_q*(1-beta2^H)",
        },
        "target_definition": "chi=mean_stream[(X_plus-X_minus)/(2*epsilon)]",
        "prediction_scale": "UNIT_PERTURBATION_RESPONSE_NO_RHO_SIGMA_Q_FACTOR",
        "q_semantics": "EXACT_V07_STATE_SPECIFIC_Q_USED_FOR_CHI_MEASUREMENT",
        "gradient_semantics": (
            "STATE_MEAN_OF_8_E2_GRADIENTS_AFTER_UNSCALE_BEFORE_CLIP_AND_STEP"
        ),
        "fit_parameters": [],
        "metrics": metrics,
        "comparison": {
            "R2_improvement": gain,
            "v_path_improves_R2": bool(gain > 0),
            "remaining_unexplained_fraction": float(1.0 - mv["R2"]),
            "remaining_residual_MAE": mv["MAE"],
            "remaining_residual_SSE": float(residuals @ residuals),
            "seed_R2_improvement": seed_gains,
            "horizon_R2_improvement": horizon_gains,
            "seeds_improved": int(sum(value > 0 for value in seed_gains.values())),
            "horizons_improved": int(sum(value > 0 for value in horizon_gains.values())),
        },
        "previous_inconsistent_result": previous,
        "strict_alignment_changes_v_path_conclusion": alignment_conclusion_changed,
        "global_q_unit_scale_control": global_q_control,
        "state_specific_q_changes_v_path_conclusion": q_conclusion_changed,
        "matched_state_count": len(state_results),
        "prediction_row_count": len(prediction_rows),
        "source_paths": dict(source_paths),
        "state_results": list(state_results),
        "prediction_rows": list(prediction_rows),
        "constraints": [
            "STATE_SPECIFIC_Q_EXACT_MATCH",
            "NO_PERTURBATION_EPSILON_SCALING",
            "NO_REPLAY_OR_TRAINING",
            "RAW_EXP_AVG_AND_EXP_AVG_SQ",
            "NO_BIAS_CORRECTION",
            "NO_EMPIRICAL_OR_FITTED_SCALE",
        ],
    }


def _fmt(value) -> str:
    return "undefined" if value is None else f"{float(value):.12g}"


def render_consistent_report(summary: Mapping) -> str:
    metrics = summary["metrics"]
    momentum = metrics["chi_m"]["overall"]
    mv = metrics["chi_mv"]["overall"]
    comparison = summary["comparison"]
    improves = bool(comparison["v_path_improves_R2"])
    previous = summary.get("previous_inconsistent_result")
    q_control = summary.get("global_q_unit_scale_control")
    q_changed = summary.get("state_specific_q_changes_v_path_conclusion")
    if q_control is None:
        q_answer = "Not evaluated against a global-q unit-scale control in this fixture."
    elif q_changed:
        q_answer = (
            "Yes—the sign of the v-path R2 change differs from the global-q control "
            "evaluated at the same unit-response scale."
        )
    else:
        q_answer = (
            "No—the v-path improvement/non-improvement conclusion matches the global-q "
            "control evaluated at the same unit-response scale."
        )
    sufficient = mv["R2"] is not None and float(mv["R2"]) > 0
    q_control_lines = []
    if q_control is not None:
        global_m = q_control["metrics"]["chi_m"]["overall"]
        global_mv = q_control["metrics"]["chi_mv"]["overall"]
        q_control_lines = [
            "## State-specific q control",
            "",
            "Both rows below use the corrected unit-response scale. This isolates the q "
            "alignment change from the removed perturbation-scale factor.",
            "",
            "| q source | chi_m R2 | chi_mv R2 | v-path delta R2 |",
            "|---|---:|---:|---:|",
            f"| exact state-specific q | {_fmt(momentum['R2'])} | {_fmt(mv['R2'])} | "
            f"{_fmt(comparison['R2_improvement'])} |",
            f"| prior global q features | {_fmt(global_m['R2'])} | "
            f"{_fmt(global_mv['R2'])} | {_fmt(q_control['R2_improvement'])} |",
            "",
        ]
    numerical_lines = []
    feature_rows = [
        row for row in summary.get("state_results", []) if "nu_q" in row
    ]
    prediction_rows = [
        row for row in summary.get("prediction_rows", []) if "chi_v" in row
    ]
    if feature_rows and prediction_rows:
        largest_nu = max(feature_rows, key=lambda row: abs(float(row["nu_q"])))
        largest_v = max(prediction_rows, key=lambda row: abs(float(row["chi_v"])))
        numerical_lines = [
            "## Numerical behavior",
            "",
            f"- Largest `|nu_q|`: seed `{largest_nu['seed']}`, state "
            f"`{largest_nu['state_step']}`, `nu_q={_fmt(largest_nu['nu_q'])}`.",
            f"- Largest `|chi_v|`: seed `{largest_v['seed']}`, state "
            f"`{largest_v['state_step']}`, H=`{largest_v['horizon']}`, "
            f"`chi_v={_fmt(largest_v['chi_v'])}`.",
            f"- v-path improves `{comparison['seeds_improved']}/"
            f"{len(comparison['seed_R2_improvement'])}` seeds and "
            f"`{comparison['horizons_improved']}/"
            f"{len(comparison['horizon_R2_improvement'])}` horizons.",
            "",
        ]
    lines = [
        "# A0C AdamW m+v Strict-Consistency Report",
        "",
        "## Answers",
        "",
        f"1. Does v-path improve after fixing scale? **{'Yes' if improves else 'No'}**. "
        f"R2 change: `{_fmt(comparison['R2_improvement'])}`.",
        f"2. Does state-specific q change the conclusion? **{q_answer}**",
        f"3. Is AdamW m+v sufficient? **{'Yes' if sufficient else 'No'}**. "
        f"Direct R2: `{_fmt(mv['R2'])}`; residual `SSE/SST = 1-R2`: "
        f"`{_fmt(comparison['remaining_unexplained_fraction'])}`.",
        "",
        "The q conclusion above uses a saved global-q feature control recomputed without "
        "epsilon, so both sides use the same unit-response scale. The comparison is diagnostic "
        "and introduces no fitted parameter.",
        "",
        "## Overall metrics",
        "",
        "| prediction | n | corr | R2 | MAE |",
        "|---|---:|---:|---:|---:|",
        f"| chi_m | {momentum['n']} | {_fmt(momentum['corr'])} | "
        f"{_fmt(momentum['R2'])} | {_fmt(momentum['MAE'])} |",
        f"| chi_mv | {mv['n']} | {_fmt(mv['corr'])} | "
        f"{_fmt(mv['R2'])} | {_fmt(mv['MAE'])} |",
        "",
        *q_control_lines,
        *numerical_lines,
        "## Alignment audit",
        "",
        f"- Matched states: `{summary['matched_state_count']}`",
        f"- Prediction rows: `{summary['prediction_row_count']}`",
        "- q: exact v07 seed/state direction loaded with internal tensor-hash validation.",
        "- g: existing state-level mean of eight matched E2 pre-clip unscaled gradients.",
        "- Scale: no `rho*sigma_q` factor; target and prediction are unit responses.",
        "- No forward, backward, clipping, optimizer step, fit, or calibration was run.",
        "",
        "## Per-seed metrics",
        "",
        "```json",
        json.dumps({key: value["per_seed"] for key, value in metrics.items()}, indent=2),
        "```",
        "",
        "## Per-horizon metrics",
        "",
        "```json",
        json.dumps({key: value["per_horizon"] for key, value in metrics.items()}, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def _load_json(path: Path) -> Dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_a0c_rows(path: Path) -> list[Dict]:
    required = {"seed", "state_step", "horizon", "chi"}
    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"A0C CSV missing required columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            row = dict(raw)
            row.update(
                seed=int(raw["seed"]),
                state_step=int(raw["state_step"]),
                horizon=int(raw["horizon"]),
                chi=_finite(raw["chi"], f"chi line {line_number}"),
            )
            key = (row["seed"], row["state_step"], row["horizon"])
            if key in seen:
                raise ValueError(f"duplicate A0C row: {key}")
            seen.add(key)
            rows.append(row)
    if tuple(sorted({row["seed"] for row in rows})) != EXPECTED_SEEDS:
        raise ValueError("A0C seeds do not match 2030-2037")
    if tuple(sorted({row["horizon"] for row in rows})) != EXPECTED_HORIZONS:
        raise ValueError("A0C horizons do not match the registered horizons")
    if len(rows) != 448:
        raise ValueError(f"expected 448 A0C rows, got {len(rows)}")
    return rows


def run_consistent_experiment(project_root: str | Path) -> Dict:
    """Compute strict state-q/unit-scale results from saved replay gradients."""
    import torch

    root = Path(project_root).resolve()
    a0c_path = root / "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv"
    state_root = root / "outputs/e3_confirmatory/v07"
    gradient_path = root / "outputs/a0_fr_audit/mv_theory_final/gradient_g.pt"
    previous_path = root / "outputs/a0_fr_audit/mv_theory_final/summary.json"
    for path in (a0c_path, gradient_path, previous_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = _read_a0c_rows(a0c_path)
    rows_by_state = defaultdict(list)
    for row in rows:
        rows_by_state[(row["seed"], row["state_step"])].append(row)
    requested_states = sorted(rows_by_state)
    if len(requested_states) != 56:
        raise ValueError(f"expected 56 seed/state pairs, got {len(requested_states)}")

    previous = _load_json(previous_path)
    previous["_source_path"] = str(previous_path)
    if previous.get("status") != "COMPLETE":
        raise ValueError("previous gradient-producing experiment is not COMPLETE")
    actual_gradient_sha = file_sha256(gradient_path)
    if actual_gradient_sha != previous["gradient_artifact"]["sha256"]:
        raise ValueError("gradient_g.pt sha256 differs from its producing summary")
    gradient_payload = torch.load(gradient_path, map_location="cpu", weights_only=False)
    if gradient_payload.get("state_count") != 56 or gradient_payload.get("stream_count") != 448:
        raise ValueError("gradient artifact does not contain 56 states/448 source streams")
    if gradient_payload.get("definition") != "AFTER_SCALER_UNSCALE_BEFORE_CLIP_BEFORE_OPTIMIZER_STEP":
        raise ValueError("gradient artifact capture point differs from required definition")
    gradient_states = gradient_payload.get("states")
    if not isinstance(gradient_states, Mapping) or len(gradient_states) != 56:
        raise ValueError("gradient artifact state mapping is incomplete")

    theta_by_state = {}
    theta_paths = {}
    for seed in EXPECTED_SEEDS:
        theta_path = state_root / f"seed_{seed}/theta_mapping_raw.json"
        theta = _load_json(theta_path)
        if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
            raise ValueError(f"invalid theta mapping for seed {seed}")
        theta_paths[str(seed)] = str(theta_path)
        for state in theta.get("states", []):
            key = (seed, int(state["step"]))
            if key in theta_by_state:
                raise ValueError(f"duplicate theta state: {key}")
            theta_by_state[key] = state
    if set(theta_by_state) != set(requested_states):
        raise ValueError("theta state set does not exactly match A0C state set")

    previous_features = {
        (int(state["seed"]), int(state["state_step"])): state
        for state in previous["state_results"]
    }
    if set(previous_features) != set(requested_states):
        raise ValueError("previous global-q feature states do not exactly match A0C")

    state_results = []
    prediction_rows = []
    global_q_unit_rows = []
    for index, (seed, state_step) in enumerate(requested_states, start=1):
        theta_state = theta_by_state[(seed, state_step)]
        checkpoint_path = state_root / f"seed_{seed}/states/state_step_{state_step:03d}.pt"
        expected_q_path = state_root / f"seed_{seed}/q/q_state_step_{state_step:03d}.pt"
        recorded_q_path = Path(theta_state["q_path"]).resolve()
        if recorded_q_path != expected_q_path.resolve():
            raise ValueError(
                f"recorded q path mismatch for seed={seed}, state={state_step}: {recorded_q_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        q_by_name, q_metadata, q_sha = load_direction(expected_q_path)
        if Path(q_metadata.get("state_path", "")).resolve() != checkpoint_path.resolve():
            raise ValueError(f"q checkpoint provenance mismatch for {expected_q_path}")

        gradient_key = f"seed_{seed}/state_step_{state_step:03d}"
        if gradient_key not in gradient_states:
            raise ValueError(f"gradient state missing: {gradient_key}")
        features = compute_consistent_state_features(
            q_by_name=q_by_name,
            q_metadata=q_metadata,
            q_sha256=q_sha,
            expected_q_sha256=theta_state["q_sha256"],
            gradient_record=gradient_states[gradient_key],
            checkpoint=checkpoint,
            seed=seed,
            state_step=state_step,
        )
        state_result = {
            "seed": seed,
            "state_step": state_step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "q_path": str(expected_q_path),
            "q_file_sha256": file_sha256(expected_q_path),
            "q_direction_sha256": q_sha,
            "q_norm": float(theta_state["q_norm"]),
            "gradient_key": gradient_key,
            "gradient_stream_count": features["gradient_stream_count"],
            "mu_q": features["mu_q"],
            "nu_q": features["nu_q"],
            "eta": features["eta"],
            "beta1": features["beta1"],
            "beta2": features["beta2"],
            "adam_eps": features["adam_eps"],
            "n_parameter_tensors": features["n_parameter_tensors"],
        }
        state_results.append(state_result)

        previous_state = previous_features[(seed, state_step)]
        for row in sorted(rows_by_state[(seed, state_step)], key=lambda item: item["horizon"]):
            strict_prediction = predict_unit_response(
                eta=features["eta"],
                beta1=features["beta1"],
                beta2=features["beta2"],
                horizon=row["horizon"],
                mu_q=features["mu_q"],
                nu_q=features["nu_q"],
            )
            prediction_rows.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "horizon": row["horizon"],
                    "chi": row["chi"],
                    **strict_prediction,
                }
            )
            global_prediction = predict_unit_response(
                eta=float(previous_state["eta_first_step"]),
                beta1=float(previous_state["beta1"]),
                beta2=float(previous_state["beta2"]),
                horizon=row["horizon"],
                mu_q=float(previous_state["mu_q"]),
                nu_q=float(previous_state["nu_q"]),
            )
            global_q_unit_rows.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "horizon": row["horizon"],
                    "chi": row["chi"],
                    **global_prediction,
                }
            )
        print(
            f"MV_CONSISTENT {index}/56 seed={seed} state={state_step} "
            f"mu_q={features['mu_q']:.8g} nu_q={features['nu_q']:.8g}",
            flush=True,
        )
        del checkpoint, q_by_name

    if len(state_results) != 56 or len(prediction_rows) != 448:
        raise RuntimeError("strict-consistency computation is incomplete")
    source_paths = {
        "project_root": str(root),
        "a0c_state_horizon_csv": str(a0c_path),
        "checkpoint_root": str(state_root),
        "state_specific_q_source": "theta_mapping_raw.json states[*].q_path",
        "theta_mapping_by_seed": theta_paths,
        "gradient_g": str(gradient_path),
        "previous_inconsistent_summary": str(previous_path),
    }
    summary = assemble_consistent_summary(
        prediction_rows,
        state_results=state_results,
        source_paths=source_paths,
        previous_inconsistent_summary=previous,
        global_q_unit_rows=global_q_unit_rows,
    )
    summary["input_hashes"] = {
        "a0c_state_horizon_csv": file_sha256(a0c_path),
        "gradient_g": actual_gradient_sha,
        "previous_inconsistent_summary": file_sha256(previous_path),
    }
    summary["matched_seeds"] = list(EXPECTED_SEEDS)
    summary["matched_horizons"] = list(EXPECTED_HORIZONS)
    summary["matched_q_count"] = len(state_results)
    summary["missing_states"] = []
    return summary
