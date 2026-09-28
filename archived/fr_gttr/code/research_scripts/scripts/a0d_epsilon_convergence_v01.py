#!/usr/bin/env python3
"""Run the FR A0D multi-step epsilon convergence audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_h1_adamw_jacobian_v01 import file_sha256, validate_e2_manifest
from fr.a0d_epsilon_convergence_v01 import (
    EPSILON_LEVELS,
    EPSILON_MULTIPLIERS,
    HORIZONS,
    SELECTED_STATES,
    assemble_summary,
    central_response,
    epsilon_convergence,
    epsilon8_candidates,
    h8_conclusion,
    json_dumps,
    prediction_metrics,
    relative_error,
    render_report,
)
from fr.direction import load_direction
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.theta_protocol import run_theta_branch


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot average empty or non-finite values")
    return float(array.mean())


def _std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("standard deviation requires at least two finite values")
    return float(array.std(ddof=1))


def _assert_finite_tree(value, label="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite result at {label}: {value}")


def _canonical_sha256(payload: dict) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _state_path(root: Path, seed: int, step: int) -> Path:
    return root / f"outputs/e3_confirmatory/v07/seed_{seed}/states/state_step_{step:03d}.pt"


def _branch_x(branch: dict, horizon: int) -> float:
    return float(branch["observations"][str(int(horizon))]["X_theta"])


def _paired_last_lr(plus: dict, minus: dict) -> float:
    """Validate the LR field populated by ordinary, non-capture branches."""
    plus_lr = float(plus["last_lr"])
    minus_lr = float(minus["last_lr"])
    if plus_lr != minus_lr:
        raise ValueError("paired branch final LR mismatch")
    return plus_lr


def _source_branch_x(stream_record: dict, sign: str, horizon: int) -> float:
    return float(stream_record[sign]["X_theta"][str(int(horizon))])


def _build_audit_config(source_config: dict) -> dict:
    return {
        "protocol": "FR_A0D_EPSILON_CONVERGENCE_V01",
        "states": [{"seed": seed, "step": step} for seed, step in SELECTED_STATES],
        "horizons": list(HORIZONS),
        "epsilon_levels": list(EPSILON_LEVELS),
        "epsilon_multipliers": list(EPSILON_MULTIPLIERS),
        "streams_per_state": 8,
        "full_epsilon_source": "IMMUTABLE_E3_CONFIRMATORY_V07_PAIRED_BRANCHES",
        "half_and_quarter_source": "NEW_RUN_THETA_BRANCH_STRICT_REPLAY",
        "tangent_source": "IMMUTABLE_A0C_RESPONSE_PROPAGATION_V01_C_FULL_TANGENT",
        "chi_formula": "(X_plus-X_minus)/(2*epsilon_level)",
        "branch_order": "restore checkpoint -> restore stream RNG -> future batches -> forward -> accumulated backward -> unscale -> pulse -> clip -> native AdamW step",
        "dtype": source_config["dtype"],
        "accumulation_steps": int(source_config["accumulation_steps"]),
        "grad_clip": float(source_config["grad_clip"]),
        "batch_size": int(source_config["batch_size"]),
        "base_learning_rate": float(source_config["learning_rate"]),
        "no_fitted_variables": True,
    }


def run_experiment(project_root: str | Path, device: str) -> dict:
    started = time.time()
    root = Path(project_root).resolve()
    config_path = root / "research/configs/e3c_confirmatory_v07.json"
    source_config = _read_json(config_path)
    if source_config.get("dtype") != "bfloat16":
        raise ValueError("v07 dtype changed")
    if int(source_config.get("accumulation_steps", -1)) != 2:
        raise ValueError("v07 accumulation_steps changed")
    if float(source_config.get("grad_clip", -1.0)) != 1.0:
        raise ValueError("v07 grad_clip changed")
    if not set(HORIZONS).issubset(set(map(int, source_config.get("horizons", [])))):
        raise ValueError("v07 config does not contain H=2,4,8")

    state_root = root / source_config["output_dir"]
    data_path = state_root / "e3c_subset.jsonl"
    partition_path = state_root / "partition_v07.json"
    v01_path = root / "outputs/a0_fr_audit/response_propagation_v01/summary.json"
    for path in (data_path, partition_path, v01_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    v01 = _read_json(v01_path)
    if v01.get("status") != "COMPLETE" or v01.get("protocol") != "FR_A0C_ADAMW_RESPONSE_PROPAGATION_V01":
        raise ValueError("A0C v01 propagation artifact is not complete")
    v01_streams = {
        (int(row["seed"]), int(row["step"]), int(row["stream_id"])): row
        for row in v01["stream_results"]
    }
    v01_results = {
        (int(row["seed"]), int(row["step"]), int(row["H"])): row
        for row in v01["results"]
    }

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("strict v07 BF16 replay requires an available CUDA device")
    model_config = MiniMindConfig(
        hidden_size=int(source_config["hidden_size"]),
        num_hidden_layers=int(source_config["num_hidden_layers"]),
        max_seq_len=int(source_config["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        model_config,
        from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=device,
    )
    dataset = PretrainDataset(
        str(data_path), tokenizer, max_length=int(source_config["max_seq_len"])
    )
    autocast_ctx, scaler, device_type = make_amp(
        torch, device, source_config["dtype"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(source_config["learning_rate"])
    )
    assert_native_adamw(optimizer)
    named = [
        (name, parameter)
        for name, parameter in raw_model(model).named_parameters()
        if parameter.requires_grad
    ]
    named_names = {name for name, _ in named}

    audit_config = _build_audit_config(source_config)
    audit_config_sha256 = _canonical_sha256(audit_config)
    output_dir = root / "outputs/a0_fr_audit/response_propagation_v02_epsilon"
    config_artifact_path = output_dir / "experiment_config.json"
    config_artifact_payload = {
        "source_v07_config": source_config,
        "audit_config": audit_config,
        "audit_config_sha256": audit_config_sha256,
    }
    config_artifact_bytes = json_dumps(config_artifact_payload).encode("utf-8")

    source_paths = {
        "project_root": str(root),
        "v07_config": str(config_path),
        "checkpoint_root": str(state_root),
        "data": str(data_path),
        "partition": str(partition_path),
        "v01_summary": str(v01_path),
        "v01_analysis_module": str(root / "research/src/fr/a0c_adamw_response_propagation_v01.py"),
        "v01_runner": str(root / "research/scripts/a0c_adamw_response_propagation_v01.py"),
        "replay_primitives": str(root / "research/src/fr/theta_protocol.py"),
        "replay_full_state": str(root / "research/src/fr/full_state.py"),
        "replay_stream_rng": str(root / "research/src/fr/stream_rng.py"),
        "replay_native_schedule": str(root / "research/src/fr/native_schedule.py"),
        "replay_perturbation": str(root / "research/src/fr/perturbation.py"),
        "replay_direction": str(root / "research/src/fr/direction.py"),
        "replay_hashing": str(root / "research/src/fr/hashing.py"),
        "runtime_minimind_v03": str(root / "research/src/fr/minimind_v03_runtime.py"),
        "manifest_validator": str(root / "research/src/fr/a0c_h1_adamw_jacobian_v01.py"),
        "native_adamw_guard": str(root / "research/src/fr/native_adamw_guard.py"),
        "model_minimind": str(root / "minimind/model/model_minimind.py"),
        "pretrain_dataset": str(root / "minimind/dataset/lm_dataset.py"),
        "trainer_utils": str(root / "minimind/trainer/trainer_utils.py"),
        "tokenizer_json": str(root / "minimind/model/tokenizer.json"),
        "tokenizer_config": str(root / "minimind/model/tokenizer_config.json"),
        "analysis_module": str(root / "research/src/fr/a0d_epsilon_convergence_v01.py"),
        "runner": str(Path(__file__).resolve()),
        "experiment_config": str(config_artifact_path),
    }
    hash_source_keys = (
        "v07_config", "data", "partition", "v01_summary", "v01_analysis_module",
        "v01_runner", "replay_primitives", "replay_full_state",
        "replay_stream_rng", "replay_native_schedule", "replay_perturbation",
        "replay_direction", "replay_hashing", "runtime_minimind_v03",
        "manifest_validator", "native_adamw_guard", "model_minimind",
        "pretrain_dataset", "trainer_utils", "tokenizer_json",
        "tokenizer_config", "analysis_module", "runner",
    )
    input_hashes = {
        key: file_sha256(source_paths[key]) for key in hash_source_keys
    }

    result_rows = []
    stream_results = []
    convergence_results = []
    full_epsilon_source_gaps = []
    tangent_aggregate_gaps = []
    new_branch_count = 0

    for state_number, (seed, step) in enumerate(SELECTED_STATES, start=1):
        theta_path = state_root / f"seed_{seed}/theta_mapping_raw.json"
        theta = _read_json(theta_path)
        if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
            raise ValueError(f"invalid v07 theta mapping for seed {seed}")
        theta_state = next(
            (row for row in theta["states"] if int(row["step"]) == step), None
        )
        if theta_state is None or len(theta_state.get("e2_streams", [])) != 8:
            raise ValueError(f"incomplete v07 state seed={seed} step={step}")
        checkpoint_path = _state_path(root, seed, step)
        q_path = _resolve(root, theta_state["q_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        if int(metadata.get("train_seed", -1)) != seed or int(metadata.get("state_optimizer_step", -1)) != step:
            raise ValueError("checkpoint seed/state mismatch")
        q_by_name, q_metadata, q_direction_sha = load_direction(q_path)
        if q_direction_sha != theta_state["q_sha256"]:
            raise ValueError("state-specific q direction hash mismatch")
        if int(q_metadata.get("train_seed", -1)) != seed or int(q_metadata.get("state_step", -1)) != step:
            raise ValueError("q seed/state metadata mismatch")
        if _resolve(root, q_metadata["state_path"]) != checkpoint_path.resolve():
            raise ValueError("q checkpoint provenance mismatch")
        if set(q_by_name) != named_names:
            raise ValueError("q/model parameter-name mismatch")

        state_label = f"seed_{seed}/state_step_{step:03d}"
        state_key = f"{seed}:{step}"
        for key, path in (
            (f"checkpoint:{state_key}", checkpoint_path),
            (f"q:{state_key}", q_path),
            (f"theta:{seed}", theta_path),
        ):
            source_paths[key] = str(path)
            input_hashes[key] = file_sha256(path)
        registered_epsilon = float(theta_state["epsilon"])
        epsilon_values = [
            registered_epsilon * multiplier for multiplier in EPSILON_MULTIPLIERS
        ]
        reference_theta = checkpoint["model"]
        state_stream_results = []

        # [FR-BATCH-STREAM] Every epsilon uses the exact saved E2 stream manifest.
        for stream_record in sorted(
            theta_state["e2_streams"], key=lambda row: int(row["stream_id"])
        ):
            stream_id = int(stream_record["stream_id"])
            manifest_path = _resolve(root, stream_record["batch_manifest_path"])
            manifest_payload = _read_json(manifest_path)
            manifest = validate_e2_manifest(
                manifest_payload,
                expected_seed=seed,
                expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=stream_record["batch_manifest_sha256"],
                accumulation_steps=int(source_config["accumulation_steps"]),
            )
            if int(manifest["metadata"]["stream_seed"]) != int(stream_record["stream_seed"]):
                raise ValueError("manifest/theta stream_seed mismatch")
            if int(manifest["metadata"]["batch_seed"]) != int(stream_record["batch_seed"]):
                raise ValueError("manifest/theta batch_seed mismatch")
            if len(manifest["batches"]) < max(HORIZONS) * int(source_config["accumulation_steps"]):
                raise ValueError("manifest has insufficient future batches")
            manifest_key = f"manifest:{seed}:{step}:{stream_id}"
            source_paths[manifest_key] = str(manifest_path)
            input_hashes[manifest_key] = file_sha256(manifest_path)

            prior = v01_streams.get((seed, step, stream_id))
            if prior is None:
                raise ValueError("missing A0C v01 stream tangent")
            if Path(prior["manifest_path"]).resolve() != manifest_path.resolve():
                raise ValueError("A0C tangent manifest path mismatch")
            if prior["manifest_content_sha256"] != manifest["content_sha256"]:
                raise ValueError("A0C tangent manifest content mismatch")
            if float(prior["epsilon_registered"]) != registered_epsilon:
                raise ValueError("A0C tangent registered epsilon mismatch")

            common = dict(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                ds=dataset,
                state_path=checkpoint_path,
                reference_theta=reference_theta,
                q=q_by_name,
                named=named,
                batches=manifest["batches"],
                stream_seed=int(stream_record["stream_seed"]),
                horizons=HORIZONS,
                accumulation_steps=int(source_config["accumulation_steps"]),
                batch_size=int(source_config["batch_size"]),
                device=device,
                grad_clip=float(source_config["grad_clip"]),
                base_lr=float(source_config["learning_rate"]),
                global_micro_step=int(metadata["global_micro_step"]),
                total_micro_steps=int(metadata["total_micro_steps"]),
                get_lr=get_lr,
                stack_batch=stack_batch,
                measure_s_first=False,
            )

            epsilon_results = []
            for level_index, (level, level_epsilon) in enumerate(
                zip(EPSILON_LEVELS, epsilon_values)
            ):
                if level_index == 0:
                    x_plus = {
                        str(horizon): _source_branch_x(stream_record, "plus", horizon)
                        for horizon in HORIZONS
                    }
                    x_minus = {
                        str(horizon): _source_branch_x(stream_record, "minus", horizon)
                        for horizon in HORIZONS
                    }
                    branch_source = "SAVED_V07"
                    plus_hash = str(stream_record["plus"]["model_hash_final"])
                    minus_hash = str(stream_record["minus"]["model_hash_final"])
                    plus_optimizer_hash = str(
                        stream_record["plus"]["optimizer_hash_final"]
                    )
                    minus_optimizer_hash = str(
                        stream_record["minus"]["optimizer_hash_final"]
                    )
                    first_preclip = {
                        "plus": float(stream_record["plus"]["first_preclip_grad_norm"]),
                        "minus": float(stream_record["minus"]["first_preclip_grad_norm"]),
                    }
                    first_clipped = {
                        "plus": bool(stream_record["plus"]["first_clipped"]),
                        "minus": bool(stream_record["minus"]["first_clipped"]),
                    }
                    saved_hash_horizon = 32
                else:
                    # [FR-REPLAY-CRITICAL] run_theta_branch restores checkpoint/RNG.
                    # [FR-PERTURBATION-CRITICAL] Pulse is injected post-unscale.
                    plus = run_theta_branch(sign=1, epsilon=level_epsilon, **common)
                    minus = run_theta_branch(sign=-1, epsilon=level_epsilon, **common)
                    new_branch_count += 2
                    for branch in (plus, minus):
                        if branch["batch_manifest_hash"] != stream_record["batch_manifest_hash"]:
                            raise ValueError("new branch batch-manifest hash mismatch")
                    paired_last_lr = _paired_last_lr(plus, minus)
                    x_plus = {
                        str(horizon): _branch_x(plus, horizon) for horizon in HORIZONS
                    }
                    x_minus = {
                        str(horizon): _branch_x(minus, horizon) for horizon in HORIZONS
                    }
                    branch_source = "NEW_STRICT_REPLAY"
                    plus_hash = str(plus["model_hash_final"])
                    minus_hash = str(minus["model_hash_final"])
                    plus_optimizer_hash = str(plus["optimizer_hash_final"])
                    minus_optimizer_hash = str(minus["optimizer_hash_final"])
                    first_preclip = {
                        "plus": float(plus["first_preclip_grad_norm"]),
                        "minus": float(minus["first_preclip_grad_norm"]),
                    }
                    first_clipped = {
                        "plus": bool(plus["first_clipped"]),
                        "minus": bool(minus["first_clipped"]),
                    }
                    saved_hash_horizon = max(HORIZONS)

                chi_fd = {
                    str(horizon): central_response(
                        x_plus=x_plus[str(horizon)],
                        x_minus=x_minus[str(horizon)],
                        epsilon=level_epsilon,
                    )
                    for horizon in HORIZONS
                }
                tangent = {
                    str(horizon): float(prior["chi_full_tangent"][str(horizon)])
                    for horizon in HORIZONS
                }
                if level_index == 0:
                    for horizon in HORIZONS:
                        full_epsilon_source_gaps.append(
                            abs(chi_fd[str(horizon)] - float(prior["chi_fd"][str(horizon)]))
                        )
                epsilon_results.append(
                    {
                        "epsilon_level": level,
                        "epsilon_multiplier": EPSILON_MULTIPLIERS[level_index],
                        "epsilon": float(level_epsilon),
                        "branch_source": branch_source,
                        "X_plus": x_plus,
                        "X_minus": x_minus,
                        "chi_fd": chi_fd,
                        "chi_tangent": tangent,
                        "absolute_error": {
                            key: abs(tangent[key] - chi_fd[key]) for key in tangent
                        },
                        "relative_error": {
                            key: relative_error(chi_fd[key], tangent[key]) for key in tangent
                        },
                        "plus_model_hash_final": plus_hash,
                        "minus_model_hash_final": minus_hash,
                        "plus_optimizer_hash_final": plus_optimizer_hash,
                        "minus_optimizer_hash_final": minus_optimizer_hash,
                        "first_preclip_grad_norm": first_preclip,
                        "first_clipped": first_clipped,
                        "branch_hash_horizon": saved_hash_horizon,
                        "last_lr": (
                            float(stream_record["plus"]["last_lr"])
                            if level_index == 0 else paired_last_lr
                        ),
                    }
                )
                print(
                    f"A0D state={state_label} stream={stream_id}/7 level={level} "
                    f"H8_fd={chi_fd['8']:.9g} H8_tangent={tangent['8']:.9g}",
                    flush=True,
                )

            stream_result = {
                "state": state_label,
                "seed": seed,
                "step": step,
                "stream_id": stream_id,
                "stream_seed": int(stream_record["stream_seed"]),
                "batch_seed": int(stream_record["batch_seed"]),
                "manifest_path": str(manifest_path),
                "manifest_content_sha256": manifest["content_sha256"],
                "registered_epsilon": registered_epsilon,
                "q_path": str(q_path),
                "q_direction_sha256": q_direction_sha,
                "checkpoint_path": str(checkpoint_path),
                "epsilon_results": epsilon_results,
            }
            state_stream_results.append(stream_result)
            stream_results.append(stream_result)

        for horizon in HORIZONS:
            key = str(horizon)
            horizon_rows = []
            for level_index, (level, level_epsilon) in enumerate(
                zip(EPSILON_LEVELS, epsilon_values)
            ):
                cells = [row["epsilon_results"][level_index] for row in state_stream_results]
                fd_values = [float(cell["chi_fd"][key]) for cell in cells]
                tangent_values = [float(cell["chi_tangent"][key]) for cell in cells]
                chi_fd = _mean(fd_values)
                chi_tangent = _mean(tangent_values)
                prior_aggregate = float(v01_results[(seed, step, horizon)]["chi_full_tangent"])
                tangent_aggregate_gaps.append(abs(chi_tangent - prior_aggregate))
                result = {
                    "state": state_label,
                    "seed": seed,
                    "step": step,
                    "H": horizon,
                    "epsilon_level": level,
                    "epsilon_multiplier": EPSILON_MULTIPLIERS[level_index],
                    "epsilon": float(level_epsilon),
                    "chi_fd": chi_fd,
                    "chi_fd_stream_std": _std(fd_values),
                    "chi_tangent": chi_tangent,
                    "chi_tangent_stream_std": _std(tangent_values),
                    "absolute_error": abs(chi_tangent - chi_fd),
                    "relative_error": relative_error(chi_fd, chi_tangent),
                    "stream_count": len(cells),
                }
                result_rows.append(result)
                horizon_rows.append(result)
            convergence_results.append(
                {
                    "state": state_label,
                    "seed": seed,
                    "step": step,
                    "H": horizon,
                    **epsilon_convergence(horizon_rows),
                }
            )
        print(f"A0D_STATE {state_number}/3 {state_label} COMPLETE", flush=True)
        del checkpoint, q_by_name, state_stream_results
        torch.cuda.empty_cache()

    if max(full_epsilon_source_gaps, default=0.0) != 0.0:
        raise RuntimeError("saved v07 full-epsilon finite differences diverge from A0C v01")
    if max(tangent_aggregate_gaps, default=0.0) > 1e-15:
        raise RuntimeError("stream-mean tangent diverges from A0C v01 aggregate")
    if new_branch_count != 96:
        raise RuntimeError(f"expected 96 new paired branches, got {new_branch_count}")

    metrics = prediction_metrics(result_rows)
    contraction_count = sum(bool(row["fd_changes_contract"]) for row in convergence_results)
    absolute_monotonic_count = sum(
        bool(row["absolute_error_monotonic"]) for row in convergence_results
    )
    relative_monotonic_count = sum(
        bool(row["relative_error_monotonic"]) for row in convergence_results
    )
    h8_convergence = [row for row in convergence_results if int(row["H"]) == 8]
    full_metrics = metrics["per_epsilon"]["epsilon"]
    quarter_metrics = metrics["per_epsilon"]["epsilon/4"]
    h8_full = metrics["per_horizon_epsilon"]["8"]["epsilon"]
    h8_quarter = metrics["per_horizon_epsilon"]["8"]["epsilon/4"]
    h8_assessment = h8_conclusion(
        h8_convergence,
        full_metrics=h8_full,
        quarter_metrics=h8_quarter,
    )
    epsilon8_cells = epsilon8_candidates(convergence_results)

    linear_answer = (
        "SUPPORTED_ALL_9_STATE_HORIZONS"
        if contraction_count == 9
        else f"PARTIAL_{contraction_count}_OF_9_STATE_HORIZONS"
    )
    error_answer = (
        "MONOTONIC_DECREASE_ALL_9_STATE_HORIZONS"
        if relative_monotonic_count == 9
        else f"PARTIAL_{relative_monotonic_count}_OF_9_STATE_HORIZONS"
    )
    h8_answer = h8_assessment["answer"]
    conclusions = {
        "linear_region": linear_answer,
        "linear_region_explanation": (
            f"Successive finite-difference changes contract in {contraction_count}/9 "
            "state-horizon cells under exact epsilon halving."
        ),
        "tangent_error": error_answer,
        "tangent_error_explanation": (
            f"Relative error decreases monotonically in {relative_monotonic_count}/9 "
            f"cells (absolute error: {absolute_monotonic_count}/9). Pooled mean relative "
            f"error changes from {full_metrics['mean_relative_error']:.6g} at epsilon "
            f"to {quarter_metrics['mean_relative_error']:.6g} at epsilon/4."
        ),
        "h8": h8_answer,
        "h8_explanation": (
            f"H=8 epsilon/4 error is lower than full-epsilon error in "
            f"{h8_assessment['endpoint_improvement_count']}/3 states, while strict "
            f"three-level monotonicity holds in {h8_assessment['monotonic_count']}/3. Its pooled "
            f"mean relative error changes from {h8_full['mean_relative_error']:.6g} "
            f"to {h8_quarter['mean_relative_error']:.6g}, with epsilon/4 R2="
            f"{h8_quarter['R2']:.6g} and corr={h8_quarter['corr']:.6g}."
        ),
        "recommended_next_step": (
            f"Treat the epsilon/4 discrepancy as the observed residual, not a bound on "
            f"tangent-model error. Run a fixed epsilon/8 diagnostic for all "
            f"{len(epsilon8_cells)} cells with non-contracting finite differences or "
            "nonmonotonic tangent error, including all H=8 cells; do not fit a new memory "
            "term from the present finite-amplitude residuals."
        ),
        "further_question": (
            "For any non-contracting state-horizon cell, does epsilon/8 restore central-"
            "difference contraction without changing clipping regime?"
        ),
        "diagnostics": {
            "fd_contraction_count": contraction_count,
            "absolute_error_monotonic_count": absolute_monotonic_count,
            "relative_error_monotonic_count": relative_monotonic_count,
            "state_horizon_count": len(convergence_results),
            "h8_endpoint_improvement_count": h8_assessment["endpoint_improvement_count"],
            "h8_relative_error_monotonic_count": h8_assessment["monotonic_count"],
            "h8_state_count": len(h8_convergence),
            "epsilon8_candidate_count": len(epsilon8_cells),
            "epsilon8_candidates": epsilon8_cells,
            "pooled_R2_full": full_metrics["R2"],
            "pooled_R2_quarter": quarter_metrics["R2"],
            "pooled_corr_full": full_metrics["corr"],
            "pooled_corr_quarter": quarter_metrics["corr"],
            "pooled_MAE_full": full_metrics["MAE"],
            "pooled_MAE_quarter": quarter_metrics["MAE"],
        },
    }
    runtime = {
        "device": str(device),
        "device_type": device_type,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "amp_dtype": source_config["dtype"],
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "optimizer_signature": optimizer_signature(optimizer),
        "input_integrity": "PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION",
        "replay_provenance": "PASS_TRANSITIVE_CODE_AND_BRANCH_STATE_HASHES",
        "full_epsilon_source_check": "PASS_EXACT_V07_VS_A0C_V01",
        "max_full_epsilon_source_abs_gap": float(max(full_epsilon_source_gaps)),
        "max_tangent_aggregate_abs_gap_vs_v01": float(max(tangent_aggregate_gaps)),
        "new_branch_count": new_branch_count,
        "new_branch_max_h": max(HORIZONS),
        "full_epsilon_branch_source": "SAVED_V07_H32_BRANCH_WITH_H2_H4_H8_OBSERVATIONS",
        "half_quarter_branch_source": "NEW_NATIVE_H8_REPLAY",
        "hvp_backend_inherited_from_v01": v01["runtime"]["hvp_backend"],
        "max_hvp_math_gradient_relative_gap_vs_native_inherited_from_v01": v01["runtime"][
            "max_hvp_math_gradient_relative_gap_vs_native"
        ],
        "elapsed_seconds": float(time.time() - started),
    }
    summary = assemble_summary(
        result_rows=result_rows,
        stream_results=stream_results,
        convergence_results=convergence_results,
        metrics=metrics,
        source_paths=source_paths,
        input_hashes=input_hashes,
        source_config=source_config,
        audit_config=audit_config,
        audit_config_sha256=audit_config_sha256,
        runtime=runtime,
        conclusions=conclusions,
    )
    summary["config_artifact"] = {
        "path": str(config_artifact_path),
        "sha256": hashlib.sha256(config_artifact_bytes).hexdigest(),
    }
    summary["config_artifact_payload"] = config_artifact_payload
    _assert_finite_tree(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output_dir = root / "outputs/a0_fr_audit/response_propagation_v02_epsilon"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_experiment(root, args.device)

    config_path = output_dir / "experiment_config.json"
    config_tmp = output_dir / "experiment_config.json.tmp"
    config_tmp.write_text(json_dumps(summary["config_artifact_payload"]), encoding="utf-8")
    config_tmp.replace(config_path)
    if file_sha256(config_path) != summary["config_artifact"]["sha256"]:
        raise RuntimeError("written experiment config hash mismatch")
    del summary["config_artifact_payload"]

    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(json_dumps(summary), encoding="utf-8")
    summary_tmp.replace(summary_path)
    report_path = output_dir / "EPSILON_CONVERGENCE_REPORT.md"
    report_tmp = output_dir / "EPSILON_CONVERGENCE_REPORT.md.tmp"
    report_tmp.write_text(render_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)
    print("STATUS = COMPLETE")
    print(f"CONFIG = {config_path}")
    print(f"SUMMARY = {summary_path}")
    print(f"REPORT = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
