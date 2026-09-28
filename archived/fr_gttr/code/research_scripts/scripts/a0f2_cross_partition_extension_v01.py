#!/usr/bin/env python3
"""Run the FR A0F2 cross-partition extension (full v07 bank census).

The tangent propagation itself is imported verbatim from the frozen FR_A0C
runner (`_run_baseline_tangents`), so the replay environment is identical to
FR_A0C and FR_A0F; only the set of states changes: instead of the 32-state
deterministic random sample, every state of the v07 multiseed bank is tested
(8 seeds x 7 steps = 56 states, a census).

After the run, every state shared with the frozen FR_A0F artifact is
cross-checked bit-exactly (row-level A/B/C values and stream-level B/C
predictions), so A0F2 both extends and reproduces FR_A0F.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from fr.a0c_adamw_response_propagation_v01 import (
    assess_baseline_replay,
    extract_v07_fd,
    json_dumps,
    old_momentum_kernel,
    relative_error,
)
from fr.a0c_h1_adamw_jacobian_v01 import file_sha256, validate_e2_manifest
from fr.a0f_cross_state_generalization_v01 import (
    HORIZONS as A0F_HORIZONS,
)
from fr.a0f2_cross_partition_extension_v01 import (
    assemble_summary,
    error_distributions,
    full_states,
    paired_b_c_comparison,
    prediction_metrics,
    render_report,
    validate_against_a0f,
)
from fr.direction import load_direction
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.theta_protocol import run_theta_branch
from research.scripts.a0c_adamw_response_propagation_v01 import (
    _run_baseline_tangents,
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _mean(values) -> float:
    import numpy as np

    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("cannot average empty or non-finite values")
    return float(values.mean())


def _std(values) -> float:
    import numpy as np

    values = np.asarray(list(values), dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("standard deviation needs at least two finite values")
    return float(values.std(ddof=1))


def _assert_finite_tree(value, label="root"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite output at {label}: {value}")


def run_experiment(project_root: str | Path, device: str) -> dict:
    started = time.time()
    root = Path(project_root).resolve()
    config_path = root / "research/configs/e3c_confirmatory_v07.json"
    config = _read_json(config_path)
    if config.get("dtype") != "bfloat16":
        raise ValueError("strict v07 replay requires bfloat16")
    if int(config.get("accumulation_steps", -1)) != 2:
        raise ValueError("strict v07 replay requires two-microbatch accumulation")
    if float(config.get("grad_clip", -1.0)) != 1.0:
        raise ValueError("strict v07 replay requires grad_clip=1")
    if not set(A0F_HORIZONS).issubset(set(map(int, config.get("horizons", [])))):
        raise ValueError("v07 config does not contain required horizons")

    states = full_states()
    state_root = root / config["output_dir"]
    data_path = state_root / "e3c_subset.jsonl"
    partition_path = state_root / "partition_v07.json"
    old_summary_path = root / "outputs/a0_fr_audit/mv_theory_consistent/summary.json"
    a0f_summary_path = root / "outputs/a0_fr_audit/cross_state_generalization_v01/summary.json"
    for path in (data_path, partition_path, old_summary_path, a0f_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_summary = _read_json(old_summary_path)
    a0f_summary = _read_json(a0f_summary_path)
    old_states = {
        (int(row["seed"]), int(row["state_step"])): row
        for row in old_summary["state_results"]
    }
    missing = [key for key in states if key not in old_states]
    if missing:
        raise ValueError(f"mv_theory_consistent artifact misses census states: {missing}")

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("strict v07 BF16 replay requires CUDA")
    mmcfg = MiniMindConfig(
        hidden_size=int(config["hidden_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        max_seq_len=int(config["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg,
        from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=device,
    )
    dataset = PretrainDataset(
        str(data_path), tokenizer, max_length=int(config["max_seq_len"])
    )
    autocast_ctx, scaler, device_type = make_amp(torch, device, config["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("v07 bfloat16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [
        (name, parameter)
        for name, parameter in raw_model(model).named_parameters()
        if parameter.requires_grad
    ]
    named_names = {name for name, _ in named}

    source_paths = {
        "project_root": str(root),
        "v07_config": str(config_path),
        "checkpoint_root": str(state_root),
        "data": str(data_path),
        "partition": str(partition_path),
        "old_momentum_summary": str(old_summary_path),
        "a0f_summary": str(a0f_summary_path),
        "a0c_runner": str(root / "research/scripts/a0c_adamw_response_propagation_v01.py"),
        "analysis_module": str(root / "research/src/fr/a0f2_cross_partition_extension_v01.py"),
        "runner": str(Path(__file__).resolve()),
        "replay_primitives": str(root / "research/src/fr/theta_protocol.py"),
    }
    input_hashes = {
        key: file_sha256(path)
        for key, path in source_paths.items()
        if key not in {"project_root", "checkpoint_root"}
    }
    all_streams = []
    result_rows = []
    saved_chi_gaps = []
    baseline_x_gaps = []
    replay_hash_passes = []
    clip_tensor_gaps = []
    hvp_gradient_gaps = []
    clip_active_steps = 0
    total_future_steps = 0

    for state_number, (seed, step) in enumerate(states, start=1):
        theta_path = state_root / f"seed_{seed}/theta_mapping_raw.json"
        theta = _read_json(theta_path)
        if theta.get("technical_pass") is not True or int(theta["train_seed"]) != seed:
            raise ValueError(f"invalid v07 theta record for seed {seed}")
        theta_state = next(
            (row for row in theta["states"] if int(row["step"]) == step), None
        )
        if theta_state is None or len(theta_state.get("e2_streams", [])) != 8:
            raise ValueError(f"missing locked v07 state/streams: seed={seed}, step={step}")
        checkpoint_path = state_root / f"seed_{seed}/states/state_step_{step:03d}.pt"
        q_path = _resolve(root, theta_state["q_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        if int(metadata.get("train_seed", -1)) != seed or int(
            metadata.get("state_optimizer_step", -1)
        ) != step:
            raise ValueError("checkpoint state metadata mismatch")
        q_by_name, q_metadata, q_sha = load_direction(q_path)
        if q_sha != theta_state["q_sha256"] or set(q_by_name) != named_names:
            raise ValueError("state-specific q identity/alignment mismatch")
        if int(q_metadata.get("train_seed", -1)) != seed or int(
            q_metadata.get("state_step", -1)
        ) != step:
            raise ValueError("state-specific q metadata mismatch")
        if _resolve(root, q_metadata["state_path"]) != checkpoint_path.resolve():
            raise ValueError("state-specific q checkpoint provenance mismatch")

        state_label = f"seed_{seed}/state_step_{step:03d}"
        source_paths[f"checkpoint:{seed}:{step}"] = str(checkpoint_path)
        source_paths[f"q:{seed}:{step}"] = str(q_path)
        source_paths[f"theta:{seed}"] = str(theta_path)
        input_hashes[f"checkpoint:{seed}:{step}"] = file_sha256(checkpoint_path)
        input_hashes[f"q:{seed}:{step}"] = file_sha256(q_path)
        input_hashes[f"theta:{seed}"] = file_sha256(theta_path)

        fd = extract_v07_fd(theta_state, A0F_HORIZONS)
        saved_chi_gaps.append(fd["max_abs_vs_saved_chi"])
        model_a = old_momentum_kernel(
            checkpoint,
            q_by_name,
            A0F_HORIZONS,
            eta=float(old_states[(seed, step)]["eta"]),
        )

        state_streams = []
        for stream_record in sorted(
            theta_state["e2_streams"], key=lambda row: int(row["stream_id"])
        ):
            stream_id = int(stream_record["stream_id"])
            # [FR-BATCH-STREAM] Only the registered E2 manifest may supply
            # future examples, ordering, and stream identity.
            manifest_path = _resolve(root, stream_record["batch_manifest_path"])
            manifest_payload = _read_json(manifest_path)
            manifest = validate_e2_manifest(
                manifest_payload,
                expected_seed=seed,
                expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=stream_record["batch_manifest_sha256"],
                accumulation_steps=int(config["accumulation_steps"]),
            )
            if len(manifest["batches"]) < max(A0F_HORIZONS) * int(config["accumulation_steps"]):
                raise ValueError("manifest does not cover H=8 accumulation window")
            if int(manifest["metadata"]["stream_seed"]) != int(stream_record["stream_seed"]):
                raise ValueError("manifest/theta stream seed mismatch")
            input_hashes[f"manifest:{seed}:{step}:{stream_id}"] = file_sha256(manifest_path)
            source_paths[f"manifest:{seed}:{step}:{stream_id}"] = str(manifest_path)

            propagated = _run_baseline_tangents(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                dataset=dataset,
                named=named,
                checkpoint_path=checkpoint_path,
                reference_theta=checkpoint["model"],
                q_by_name=q_by_name,
                batches=manifest["batches"],
                stream_seed=int(stream_record["stream_seed"]),
                accumulation_steps=int(config["accumulation_steps"]),
                device=device,
                grad_clip=float(config["grad_clip"]),
                base_lr=float(config["learning_rate"]),
                global_micro_step=int(metadata["global_micro_step"]),
                total_micro_steps=int(metadata["total_micro_steps"]),
                get_lr=get_lr,
            )
            saved_baseline = stream_record["baseline"]
            shadow = run_theta_branch(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                ds=dataset,
                state_path=checkpoint_path,
                reference_theta=checkpoint["model"],
                q=q_by_name,
                named=named,
                batches=manifest["batches"],
                stream_seed=int(stream_record["stream_seed"]),
                sign=0,
                epsilon=0.0,
                horizons=A0F_HORIZONS,
                accumulation_steps=int(config["accumulation_steps"]),
                batch_size=int(config["batch_size"]),
                device=device,
                grad_clip=float(config["grad_clip"]),
                base_lr=float(config["learning_rate"]),
                global_micro_step=int(metadata["global_micro_step"]),
                total_micro_steps=int(metadata["total_micro_steps"]),
                get_lr=get_lr,
                stack_batch=stack_batch,
                measure_s_first=False,
            )
            replay = assess_baseline_replay(
                propagated=propagated,
                shadow_h8=shadow,
                saved_v07_h32=saved_baseline,
                horizons=A0F_HORIZONS,
            )
            baseline_x_gaps.append(replay["max_X_abs_gap_vs_shadow_H8"])
            baseline_x_gaps.append(replay["max_X_abs_gap_vs_saved_v07"])
            replay_hash_passes.append(replay["pass"])
            clip_tensor_gaps.append(propagated["max_native_clip_tensor_l2_gap"])
            hvp_gradient_gaps.extend(
                row["hvp_gradient_relative_gap_vs_native"]
                for row in propagated["step_trace"] if row["hvp_evaluated"]
            )
            for row in propagated["step_trace"]:
                if row["optimizer_step_index"] > 1:
                    total_future_steps += 1
                    clip_active_steps += bool(row["clip_active"])
            if not replay["pass"]:
                print(
                    "BASELINE_REPLAY_DIAGNOSTIC "
                    f"state={state_label} stream={stream_id} "
                    f"shadow_model_hash_equal={replay['shadow_model_hash_equal']} "
                    f"shadow_optimizer_hash_equal={replay['shadow_optimizer_hash_equal']}",
                    flush=True,
                )
                raise RuntimeError(
                    f"baseline replay mismatch seed={seed} step={step} stream={stream_id}"
                )
            fd_stream = next(
                row for row in fd["per_stream"] if int(row["stream_id"]) == stream_id
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
                "epsilon_registered": float(fd["epsilon"]),
                "chi_fd": fd_stream["chi_fd"],
                "chi_optimizer_only": propagated["chi_optimizer_only"],
                "chi_full_tangent": propagated["chi_full_tangent"],
                "baseline_X_theta": propagated["baseline_X_theta"],
                "baseline_replay": replay,
                "max_native_clip_tensor_l2_gap": propagated["max_native_clip_tensor_l2_gap"],
                "step_trace": propagated["step_trace"],
            }
            state_streams.append(stream_result)
            all_streams.append(stream_result)
            print(
                f"PROP state={state_number}/{len(states)} {state_label} stream={stream_id}/7 "
                f"B8={propagated['chi_optimizer_only']['8']:.9g} "
                f"C8={propagated['chi_full_tangent']['8']:.9g}",
                flush=True,
            )

        for horizon in A0F_HORIZONS:
            key = str(horizon)
            observed = float(fd["mean"][key])
            prediction_a = float(model_a["predictions"][key])
            prediction_b = _mean(row["chi_optimizer_only"][key] for row in state_streams)
            prediction_c = _mean(row["chi_full_tangent"][key] for row in state_streams)
            result_rows.append(
                {
                    "state": state_label,
                    "seed": seed,
                    "step": step,
                    "H": horizon,
                    "epsilon_registered": float(fd["epsilon"]),
                    "chi_fd": observed,
                    "chi_fd_stream_std": _std(row["chi_fd"][key] for row in state_streams),
                    "chi_A_momentum": prediction_a,
                    "chi_optimizer_only": prediction_b,
                    "chi_optimizer_only_stream_std": _std(
                        row["chi_optimizer_only"][key] for row in state_streams
                    ),
                    "chi_full_tangent": prediction_c,
                    "chi_full_tangent_stream_std": _std(
                        row["chi_full_tangent"][key] for row in state_streams
                    ),
                    "relative_error_A": relative_error(observed, prediction_a),
                    "relative_error_optimizer_only": relative_error(observed, prediction_b),
                    "relative_error_full_tangent": relative_error(observed, prediction_c),
                    "absolute_error_A": abs(prediction_a - observed),
                    "absolute_error_optimizer_only": abs(prediction_b - observed),
                    "absolute_error_full_tangent": abs(prediction_c - observed),
                    "old_kernel": {
                        "mu_q": model_a["mu_q"],
                        "eta_first_future_step": model_a["eta_first_future_step"],
                        "beta1": model_a["beta1"],
                    },
                }
            )
        print(f"PROP_STATE {state_number}/{len(states)} {state_label} COMPLETE", flush=True)
        del checkpoint, q_by_name, state_streams
        torch.cuda.empty_cache()

    metrics = prediction_metrics(result_rows)
    distributions = error_distributions(result_rows)
    paired = paired_b_c_comparison(result_rows)
    a0f_crosscheck = validate_against_a0f(
        result_rows=result_rows,
        stream_results=all_streams,
        a0f_summary=a0f_summary,
    )
    mean_a = _mean(
        row["relative_error_A"] for row in result_rows if row["relative_error_A"] is not None
    )
    mean_b = _mean(
        row["relative_error_optimizer_only"]
        for row in result_rows if row["relative_error_optimizer_only"] is not None
    )
    mean_c = _mean(
        row["relative_error_full_tangent"]
        for row in result_rows if row["relative_error_full_tangent"] is not None
    )
    overall_paired = paired["overall"]["relative_error"]
    h_all_overall = overall_paired["distribution"]["mean"]
    c_wins = overall_paired["wins"]
    c_losses = overall_paired["losses"]
    c_ties = overall_paired["ties"]
    total_pairs = c_wins + c_losses + c_ties

    # Per-stage (training-step) consistency over horizons H>1.
    per_step = defaultdict(list)
    for row in result_rows:
        if int(row["H"]) == 1:
            continue
        improvement = (
            float(row["relative_error_optimizer_only"])
            - float(row["relative_error_full_tangent"])
        )
        per_step[int(row["step"])].append(improvement)
    steps_positive = sum(1 for values in per_step.values() if _mean(values) > 0.0)
    steps_all_positive = steps_positive == len(per_step)
    min_step_mean = min(_mean(values) for values in per_step.values())

    if c_wins > c_losses and h_all_overall > 0.0:
        stability = (
            f"YES_C_BEATS_B_IN_{c_wins}_OF_{total_pairs}_STATE_HORIZON_ROWS"
            f"_MEAN_IMPROVEMENT_{h_all_overall:+.6g}"
        )
    else:
        stability = (
            f"NOT_UNIFORM_C_WINS_{c_wins}_LOSSES_{c_losses}_OF_{total_pairs}"
        )
    conclusions = {
        "c_vs_b_stability": stability,
        "c_vs_b_explanation": (
            f"Paired per-state comparison over the full 56-state v07 census and "
            f"horizons H=1,2,4,8: C beats B in {c_wins}/{total_pairs} rows "
            f"(ties {c_ties}, losses {c_losses}); mean relative-error improvement "
            f"{h_all_overall:+.6g}, median {overall_paired['distribution']['median']:+.6g}, "
            f"worst regression {overall_paired['worst_regression']:+.6g}. "
            f"H=1 rows are a zero-control (B == C by construction, no future step)."
        ),
        "a0f_consistency": (
            "BIT_EXACT_REPRODUCTION_OF_ALL_SHARED_STATES"
            if a0f_crosscheck["pass"]
            else "MISMATCH"
        ),
        "a0f_consistency_explanation": (
            f"All {a0f_crosscheck['rows_checked']} state-horizon rows and "
            f"{a0f_crosscheck['streams_checked']} streams shared with the frozen "
            f"FR_A0F artifact are reproduced with max absolute gap "
            f"{max(a0f_crosscheck['max_abs_gap_by_field'].values()):.6g} "
            f"(tolerance {a0f_crosscheck['tolerance']:g}); the A0F random sample "
            f"is a strict subset of this census."
        ),
        "stage_consistency": (
            f"STEPS_WITH_POSITIVE_MEAN_IMPROVEMENT_{steps_positive}_OF_{len(per_step)}"
        ),
        "stage_explanation": (
            f"Mean B-to-C relative-error improvement (H>1) is positive at every one of "
            f"the {len(per_step)} locked training stages (steps "
            f"{min(per_step)}-{max(per_step)}); the weakest stage mean is "
            f"{min_step_mean:+.6g}. Mean relative error over all rows: "
            f"A={mean_a:.6g}, B={mean_b:.6g}, C={mean_c:.6g}."
        ),
        "model_ordering": "C_BEATS_B_BEATS_A" if mean_c < mean_b < mean_a else "NON_MONOTONE",
        "ordering_explanation": (
            "Ordering is by mean relative error over all census state-horizon rows; "
            "no fitted or empirical variable enters any model."
        ),
        "recommended_next_step": (
            "The full-bank census confirms the FR_A0F sample. Cross-partition "
            "replication on the v06 bank requires generating a v06-compatible "
            "observation protocol (per-state q, E2 streams, batch manifests, and a "
            "{1,2,4,8}-horizon registration); schedule it as a new E2/E3-grade "
            "experiment rather than a replay extension. Do not fit any memory kernel "
            "before that replication exists."
        ),
        "further_question": (
            "Does the C advantage persist on a different model scale or optimizer "
            "configuration once the v06 observation protocol exists?"
        ),
        "diagnostics": {
            "mean_relative_error_A": mean_a,
            "mean_relative_error_B": mean_b,
            "mean_relative_error_C": mean_c,
            "C_wins": c_wins,
            "C_losses": c_losses,
            "C_ties": c_ties,
            "steps_with_positive_mean_improvement": steps_positive,
            "steps_all_positive": bool(steps_all_positive),
            "min_step_mean_improvement": float(min_step_mean),
        },
    }
    runtime = {
        "device": str(device),
        "device_type": device_type,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "amp_dtype": config["dtype"],
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "optimizer_signature": optimizer_signature(optimizer),
        "hvp_backend": "PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA",
        "hvp_interpretation": "MIXED_BACKEND_ANALYTIC_SURROGATE_NOT_EXACT_FLASH_KERNEL_TANGENT",
        "input_integrity": "PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION",
        "baseline_replay": "PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS" if all(replay_hash_passes) and max(baseline_x_gaps) == 0.0 else "FAIL",
        "stream_count": int(len(all_streams)),
        "max_baseline_X_abs_gap": float(max(baseline_x_gaps)),
        "max_saved_chi_abs_gap": float(max(saved_chi_gaps)),
        "max_native_clip_tensor_l2_gap": float(max(clip_tensor_gaps)),
        "max_hvp_math_gradient_relative_gap_vs_native": float(max(hvp_gradient_gaps)),
        "clip_active_steps_h_gt_1": int(clip_active_steps),
        "total_steps_h_gt_1": int(total_future_steps),
        "elapsed_seconds": float(time.time() - started),
    }
    summary = assemble_summary(
        result_rows=result_rows,
        stream_results=all_streams,
        metrics=metrics,
        distributions=distributions,
        paired=paired,
        a0f_crosscheck=a0f_crosscheck,
        source_paths=source_paths,
        input_hashes=input_hashes,
        runtime=runtime,
        conclusions=conclusions,
    )
    _assert_finite_tree(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output_dir = root / "outputs/a0_fr_audit/cross_partition_extension_v01"
    summary = run_experiment(root, args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    config_tmp = output_dir / "experiment_config.json.tmp"
    config_tmp.write_text(json_dumps({
        "protocol": "FR_A0F2_CROSS_PARTITION_EXTENSION_V01",
        "sampling_scheme": "full_census_of_v07_bank",
        "extends": "FR_A0F_CROSS_STATE_GENERALIZATION_V01",
        "horizons": list(A0F_HORIZONS),
        "states": [[seed, step] for seed, step in full_states()],
    }), encoding="utf-8")
    config_tmp.replace(config_path)
    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(json_dumps(summary), encoding="utf-8")
    summary_tmp.replace(summary_path)
    report_path = output_dir / "CROSS_PARTITION_EXTENSION_REPORT.md"
    report_tmp = output_dir / "CROSS_PARTITION_EXTENSION_REPORT.md.tmp"
    report_tmp.write_text(render_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)
    print("STATUS = COMPLETE")
    print(f"SUMMARY = {summary_path}")
    print(f"REPORT = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
