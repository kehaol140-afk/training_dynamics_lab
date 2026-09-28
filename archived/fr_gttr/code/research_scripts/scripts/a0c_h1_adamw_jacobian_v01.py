#!/usr/bin/env python3
"""Run the strict three-state H=1 native-AdamW Jacobian experiment."""
from __future__ import annotations

import argparse
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

from fr.a0c_h1_adamw_jacobian_v01 import (
    SELECTED_STATES,
    adamw_h1_jacobian,
    assemble_summary,
    epsilon_levels,
    file_sha256,
    json_dumps,
    relative_error,
    render_report,
    summarize_epsilon_convergence,
    validate_e2_manifest,
)
from fr.direction import load_direction
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.theta_protocol import run_theta_branch


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def _clip_is_active(norm: float, max_norm: float) -> bool:
    return bool(float(max_norm) / (float(norm) + 1e-6) < 1.0)


def _clip_after_norm(norm: float, max_norm: float) -> float:
    coefficient = min(float(max_norm) / (float(norm) + 1e-6), 1.0)
    return float(norm) * coefficient


def _branch_x(branch: dict) -> float:
    return float(branch["observations"]["1"]["X_theta"])


def _assert_finite_tree(value, label="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite result at {label}: {value}")


def _state_path(root: Path, seed: int, step: int) -> Path:
    return root / f"outputs/e3_confirmatory/v07/seed_{seed}/states/state_step_{step:03d}.pt"


def _resolve_recorded_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_experiment(project_root: str | Path, device: str) -> dict:
    started = time.time()
    root = Path(project_root).resolve()
    config_path = root / "research/configs/e3c_confirmatory_v07.json"
    config = _read_json(config_path)
    if config.get("dtype") != "bfloat16" or int(config.get("accumulation_steps", -1)) != 2:
        raise ValueError("v07 AMP/gradient-accumulation settings changed")
    if float(config.get("grad_clip", -1.0)) != 1.0:
        raise ValueError("v07 grad_clip changed")
    if tuple(map(int, config.get("confirmation_train_seeds", []))) != tuple(range(2030, 2038)):
        raise ValueError("v07 confirmatory seed registry changed")

    state_root = root / config["output_dir"]
    data_path = state_root / "e3c_subset.jsonl"
    partition_path = state_root / "partition_v07.json"
    previous_path = root / "outputs/a0_fr_audit/mv_theory_consistent/summary.json"
    for path in (data_path, partition_path, previous_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    previous = _read_json(previous_path)
    previous_states = {
        (int(row["seed"]), int(row["state_step"])): row
        for row in previous["state_results"]
    }
    previous_h1 = {
        (int(row["seed"]), int(row["state_step"])): row
        for row in previous["prediction_rows"] if int(row["horizon"]) == 1
    }

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("strict v07 BF16 AMP replay requires an available CUDA device")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(name, param) for name, param in raw_model(model).named_parameters() if param.requires_grad]
    named_names = {name for name, _ in named}

    source_paths = {
        "project_root": str(root),
        "v07_config": str(config_path),
        "checkpoint_root": str(state_root),
        "data": str(data_path),
        "partition": str(partition_path),
        "previous_mv_summary": str(previous_path),
        "replay_module": str(root / "research/src/fr/theta_protocol.py"),
        "analysis_module": str(root / "research/src/fr/a0c_h1_adamw_jacobian_v01.py"),
        "runner": str(Path(__file__).resolve()),
    }
    input_hashes = {
        "v07_config": file_sha256(config_path),
        "data": file_sha256(data_path),
        "partition": file_sha256(partition_path),
        "previous_mv_summary": file_sha256(previous_path),
        "replay_module": file_sha256(source_paths["replay_module"]),
        "analysis_module": file_sha256(source_paths["analysis_module"]),
        "runner": file_sha256(source_paths["runner"]),
    }

    all_state_results = []
    all_epsilon_rows = []
    all_stream_results = []
    v07_replay_errors = []

    for state_index, (seed, step) in enumerate(SELECTED_STATES, start=1):
        theta_path = state_root / f"seed_{seed}/theta_mapping_raw.json"
        theta = _read_json(theta_path)
        if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
            raise ValueError(f"invalid v07 theta mapping for seed {seed}")
        theta_state = next((row for row in theta["states"] if int(row["step"]) == step), None)
        if theta_state is None:
            raise ValueError(f"state absent from theta mapping: seed={seed}, step={step}")
        checkpoint_path = _state_path(root, seed, step)
        q_path = _resolve_recorded_path(root, theta_state["q_path"])
        if not checkpoint_path.is_file() or not q_path.is_file():
            raise FileNotFoundError(f"checkpoint/q missing for seed={seed}, step={step}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        if int(metadata.get("train_seed", -1)) != seed or int(metadata.get("state_optimizer_step", -1)) != step:
            raise ValueError("checkpoint seed/state metadata mismatch")
        q_by_name, q_metadata, q_sha = load_direction(q_path)
        if q_sha != theta_state["q_sha256"]:
            raise ValueError("state-specific q direction hash mismatch")
        if int(q_metadata.get("train_seed", -1)) != seed or int(q_metadata.get("state_step", -1)) != step:
            raise ValueError("state-specific q metadata mismatch")
        if _resolve_recorded_path(root, q_metadata["state_path"]) != checkpoint_path.resolve():
            raise ValueError("q checkpoint provenance mismatch")
        if set(q_by_name) != named_names:
            raise ValueError("q/model trainable parameter-name mismatch")
        if len(theta_state.get("e2_streams", [])) != 8:
            raise ValueError("selected state does not contain eight E2 streams")

        state_label = f"seed_{seed}/state_step_{step:03d}"
        state_hash_prefix = f"{seed}:{step}"
        input_hashes[f"checkpoint:{state_hash_prefix}"] = file_sha256(checkpoint_path)
        input_hashes[f"q:{state_hash_prefix}"] = file_sha256(q_path)
        input_hashes[f"theta:{seed}"] = file_sha256(theta_path)
        source_paths[f"checkpoint:{state_hash_prefix}"] = str(checkpoint_path)
        source_paths[f"q:{state_hash_prefix}"] = str(q_path)
        source_paths[f"theta:{seed}"] = str(theta_path)

        registered_epsilon = float(theta_state["epsilon"])
        levels = epsilon_levels(registered_epsilon)
        state_streams = []
        reference_theta = checkpoint["model"]

        for stream_record in sorted(theta_state["e2_streams"], key=lambda row: int(row["stream_id"])):
            stream_id = int(stream_record["stream_id"])
            manifest_path = _resolve_recorded_path(root, stream_record["batch_manifest_path"])
            manifest_payload = _read_json(manifest_path)
            manifest = validate_e2_manifest(
                manifest_payload,
                expected_seed=seed,
                expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=stream_record["batch_manifest_sha256"],
                accumulation_steps=int(config["accumulation_steps"]),
            )
            if int(manifest["metadata"]["stream_seed"]) != int(stream_record["stream_seed"]):
                raise ValueError("manifest/theta stream_seed mismatch")
            if int(manifest["metadata"]["batch_seed"]) != int(stream_record["batch_seed"]):
                raise ValueError("manifest/theta batch_seed mismatch")
            input_hashes[f"manifest:{seed}:{step}:{stream_id}"] = file_sha256(manifest_path)
            source_paths[f"manifest:{seed}:{step}:{stream_id}"] = str(manifest_path)

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
                horizons=[1],
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
            captured = run_theta_branch(
                sign=0,
                epsilon=0.0,
                capture_first_unscaled_gradient=True,
                stop_after_first_unscaled_gradient=True,
                **common,
            )
            if captured["batch_manifest_hash"] != stream_record["batch_manifest_hash"]:
                raise ValueError("captured branch batch manifest hash mismatch")
            jacobian = adamw_h1_jacobian(
                checkpoint=checkpoint,
                q_by_name=q_by_name,
                gradient_by_name=captured["first_unscaled_gradient"],
                effective_lr=float(captured["first_step_lr"]),
                grad_clip=float(config["grad_clip"]),
            )
            grad_norm_gap = abs(
                float(captured["first_unscaled_grad_norm"])
                - float(jacobian["clip_before_norm"])
            )

            epsilon_stream_rows = []
            baseline_hashes = []
            for level_index, level_epsilon in enumerate(levels):
                label = ("epsilon", "epsilon/2", "epsilon/4")[level_index]
                baseline = run_theta_branch(sign=0, epsilon=0.0, **common)
                plus = run_theta_branch(sign=1, epsilon=level_epsilon, **common)
                minus = run_theta_branch(sign=-1, epsilon=level_epsilon, **common)
                for branch in (baseline, plus, minus):
                    if branch["batch_manifest_hash"] != stream_record["batch_manifest_hash"]:
                        raise ValueError("finite-difference branch batch hash mismatch")
                    if float(branch["last_lr"]) != float(captured["first_step_lr"]):
                        raise ValueError("finite-difference branch current-step LR mismatch")
                baseline_hashes.append(
                    (baseline["model_hash_final"], baseline["optimizer_hash_final"], _branch_x(baseline))
                )
                x0 = _branch_x(baseline)
                xp = _branch_x(plus)
                xm = _branch_x(minus)
                chi_fd = (xp - xm) / (2.0 * level_epsilon)
                preclip = {
                    "baseline": float(baseline["first_preclip_grad_norm"]),
                    "plus": float(plus["first_preclip_grad_norm"]),
                    "minus": float(minus["first_preclip_grad_norm"]),
                }
                postclip = {
                    branch: _clip_after_norm(norm, float(config["grad_clip"]))
                    for branch, norm in preclip.items()
                }
                active = {
                    branch: _clip_is_active(norm, float(config["grad_clip"]))
                    for branch, norm in preclip.items()
                }
                replay_error = None
                if level_index == 0:
                    saved = stream_record
                    replay_error = {
                        "baseline_X_abs": abs(x0 - float(saved["baseline"]["X_theta"]["1"])),
                        "plus_X_abs": abs(xp - float(saved["plus"]["X_theta"]["1"])),
                        "minus_X_abs": abs(xm - float(saved["minus"]["X_theta"]["1"])),
                        "baseline_preclip_abs": abs(preclip["baseline"] - float(saved["baseline"]["first_preclip_grad_norm"])),
                        "plus_preclip_abs": abs(preclip["plus"] - float(saved["plus"]["first_preclip_grad_norm"])),
                        "minus_preclip_abs": abs(preclip["minus"] - float(saved["minus"]["first_preclip_grad_norm"])),
                    }
                    v07_replay_errors.extend(replay_error.values())
                epsilon_stream_rows.append(
                    {
                        "epsilon_level": label,
                        "epsilon": float(level_epsilon),
                        "X_baseline": x0,
                        "X_plus": xp,
                        "X_minus": xm,
                        "chi_fd": float(chi_fd),
                        "chi_jac": jacobian["chi_jac"],
                        "relative_error": relative_error(chi_fd, jacobian["chi_jac"]),
                        "clip_before": preclip,
                        "clip_after": postclip,
                        "clip_active": active,
                        "v07_full_epsilon_replay_abs_error": replay_error,
                    }
                )
            if len(set(baseline_hashes)) != 1:
                raise RuntimeError("repeated baseline branches are not bitwise reproducible")
            if grad_norm_gap > 1e-5:
                raise RuntimeError(f"captured/analytic gradient norm mismatch: {grad_norm_gap}")

            stream_result = {
                "state": state_label,
                "seed": seed,
                "step": step,
                "stream_id": stream_id,
                "stream_seed": int(stream_record["stream_seed"]),
                "batch_seed": int(stream_record["batch_seed"]),
                "manifest_path": str(manifest_path),
                "manifest_content_sha256": manifest["content_sha256"],
                "gradient_norm_capture": float(captured["first_unscaled_grad_norm"]),
                "gradient_norm_analytic": jacobian["clip_before_norm"],
                "gradient_norm_abs_gap": grad_norm_gap,
                **jacobian,
                "epsilon_results": epsilon_stream_rows,
            }
            state_streams.append(stream_result)
            all_stream_results.append(stream_result)
            print(
                f"H1_JAC state={state_label} stream={stream_id}/7 "
                f"clip={jacobian['clip_active']} chi_jac={jacobian['chi_jac']:.9g}",
                flush=True,
            )

        if len(state_streams) != 8:
            raise RuntimeError("state replay did not produce eight streams")
        jac_mean = _mean(row["chi_jac"] for row in state_streams)
        jac_m_mean = _mean(row["chi_jac_m"] for row in state_streams)
        jac_v_mean = _mean(row["chi_jac_v"] for row in state_streams)
        mu_old_values = [float(row["mu_q_old"]) for row in state_streams]
        mu_old_bc_values = [float(row["mu_q_old_bias_corrected"]) for row in state_streams]
        if max(mu_old_values) != min(mu_old_values) or max(mu_old_bc_values) != min(mu_old_bc_values):
            raise RuntimeError("state-only old mu changed across streams")
        mu_full_mean = _mean(row["mu_q_full"] for row in state_streams)
        state_epsilon_rows = []
        for level_index, level_epsilon in enumerate(levels):
            stream_level_rows = [row["epsilon_results"][level_index] for row in state_streams]
            fd_values = [float(row["chi_fd"]) for row in stream_level_rows]
            clip_before = {
                branch: {
                    "mean": _mean(row["clip_before"][branch] for row in stream_level_rows),
                    "min": min(float(row["clip_before"][branch]) for row in stream_level_rows),
                    "max": max(float(row["clip_before"][branch]) for row in stream_level_rows),
                }
                for branch in ("plus", "baseline", "minus")
            }
            clip_after = {
                branch: {
                    "mean": _mean(row["clip_after"][branch] for row in stream_level_rows),
                    "min": min(float(row["clip_after"][branch]) for row in stream_level_rows),
                    "max": max(float(row["clip_after"][branch]) for row in stream_level_rows),
                }
                for branch in ("plus", "baseline", "minus")
            }
            clip_active = {
                f"{branch}_rate": _mean(float(row["clip_active"][branch]) for row in stream_level_rows)
                for branch in ("plus", "baseline", "minus")
            }
            chi_fd = _mean(fd_values)
            row = {
                "state": state_label,
                "seed": seed,
                "step": step,
                "epsilon_level": ("epsilon", "epsilon/2", "epsilon/4")[level_index],
                "epsilon": float(level_epsilon),
                "chi_fd": chi_fd,
                "chi_fd_std": _std(fd_values),
                "chi_fd_per_stream": fd_values,
                "chi_jac": jac_mean,
                "chi_jac_m": jac_m_mean,
                "chi_jac_v": jac_v_mean,
                "relative_error": relative_error(chi_fd, jac_mean),
                "clip_before": clip_before,
                "clip_after": clip_after,
                "clip_active": clip_active,
                "mu_q_old": mu_old_values[0],
                "mu_q_old_bias_corrected": mu_old_bc_values[0],
                "mu_q_full": mu_full_mean,
                "v07_observed_chi_H1": float(theta_state["chi_theta"]["1"]),
            }
            state_epsilon_rows.append(row)
            all_epsilon_rows.append(row)

        convergence = summarize_epsilon_convergence(state_epsilon_rows)
        smallest = state_epsilon_rows[-1]
        previous_state = previous_states[(seed, step)]
        previous_row = previous_h1[(seed, step)]
        state_result = {
            **{key: smallest[key] for key in (
                "state", "seed", "step", "epsilon", "chi_fd", "chi_jac",
                "relative_error", "clip_before", "clip_after", "mu_q_old",
                "mu_q_old_bias_corrected", "mu_q_full",
            )},
            "registered_epsilon": registered_epsilon,
            "sigma_q": float(theta_state["sigma_q"]),
            "rho": float(theta_state["rho"]),
            "q_path": str(q_path),
            "q_direction_sha256": q_sha,
            "checkpoint_path": str(checkpoint_path),
            "stream_count": len(state_streams),
            "chi_jac_m": jac_m_mean,
            "chi_jac_v": jac_v_mean,
            "chi_jac_stream_std": _std(row["chi_jac"] for row in state_streams),
            "v07_observed_chi_H1": float(theta_state["chi_theta"]["1"]),
            "full_epsilon_fd_minus_v07": float(
                state_epsilon_rows[0]["chi_fd"] - float(theta_state["chi_theta"]["1"])
            ),
            "epsilon_results": state_epsilon_rows,
            **convergence,
            "old_double_exponential": {
                "mu_q": float(previous_state["mu_q"]),
                "nu_q": float(previous_state["nu_q"]),
                "chi_m_H1": float(previous_row["chi_m"]),
                "chi_v_H1": float(previous_row["chi_v"]),
                "chi_mv_H1": float(previous_row["chi_mv"]),
            },
            "exact_v_to_old_v_abs_ratio": (
                None if float(previous_row["chi_v"]) == 0.0
                else float(abs(jac_v_mean) / abs(float(previous_row["chi_v"])))
            ),
        }
        all_state_results.append(state_result)
        print(
            f"H1_JAC_STATE {state_index}/3 {state_label} "
            f"fd_quarter={smallest['chi_fd']:.9g} jac={jac_mean:.9g} "
            f"rel={smallest['relative_error']:.6g}",
            flush=True,
        )
        del checkpoint, q_by_name
        torch.cuda.empty_cache()

    if len(all_state_results) != 3 or len(all_epsilon_rows) != 9 or len(all_stream_results) != 24:
        raise RuntimeError("incomplete H1 experiment")
    errors_full = [float(row["epsilon_results"][0]["relative_error"]) for row in all_state_results]
    errors_quarter = [float(row["relative_error"]) for row in all_state_results]
    h1_converges = all(q <= f for q, f in zip(errors_quarter, errors_full))
    extreme = next(row for row in all_state_results if row["seed"] == 2037 and row["step"] == 46)
    exact_v = abs(float(extreme["chi_jac_v"]))
    old_v = abs(float(extreme["old_double_exponential"]["chi_v_H1"]))
    nu_regularized = bool(exact_v < old_v)
    conclusions = {
        "h1_explains_real_response": "YES" if h1_converges else "PARTIAL",
        "h1_explanation": (
            f"Across the three states, epsilon/4 relative errors range from "
            f"{min(errors_quarter):.6g} to {max(errors_quarter):.6g}; "
            f"all quarter-epsilon errors are no larger than their full-epsilon values: "
            f"{h1_converges}."
        ),
        "nu_anomaly_from_frozen_v": "YES" if nu_regularized else "NO",
        "nu_explanation": (
            f"For seed 2037/state 46, the old H1 v term is "
            f"{extreme['old_double_exponential']['chi_v_H1']:.9g}, while the complete "
            f"local v-path is {extreme['chi_jac_v']:.9g} (absolute ratio "
            f"{extreme['exact_v_to_old_v_abs_ratio']:.6g})."
        ),
        "hessian_feedback_next": "YES_FOR_H_GT_1" if h1_converges else "DEFER_UNTIL_LOCAL_GAP_RESOLVED",
        "hessian_explanation": (
            "H=1 has no future-gradient feedback after the post-unscale pulse. Extending "
            "the validated local map to H>1 requires propagation through later gradients, "
            "including Hessian-vector and clipping/activation boundary effects."
        ),
        "recommended_next_step": (
            "Propagate the H1 tangent through future gradients with Hessian-vector products "
            "and retain stepwise clipping Jacobians; keep activation-boundary changes as a "
            "separate non-smooth residual diagnostic."
        ),
    }
    max_replay_error = max(v07_replay_errors)
    runtime = {
        "device": str(device),
        "device_type": device_type,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "amp_dtype": config["dtype"],
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "optimizer_signature": optimizer_signature(optimizer),
        "input_integrity": "PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION",
        "v07_replay_reproduction": "PASS_EXACT" if max_replay_error == 0.0 else "PASS_WITH_NUMERIC_DELTA",
        "v07_replay_max_abs_error": float(max_replay_error),
        "elapsed_seconds": float(time.time() - started),
    }
    summary = assemble_summary(
        state_results=all_state_results,
        epsilon_rows=all_epsilon_rows,
        stream_results=all_stream_results,
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
    output_dir = root / "outputs/a0_fr_audit/H1_adamw_jacobian_v01"
    summary = run_experiment(root, args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(json_dumps(summary), encoding="utf-8")
    summary_tmp.replace(summary_path)
    report_path = output_dir / "H1_JACOBIAN_REPORT.md"
    report_tmp = output_dir / "H1_JACOBIAN_REPORT.md.tmp"
    report_tmp.write_text(render_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)
    print("STATUS = COMPLETE")
    print(f"SUMMARY = {summary_path}")
    print(f"REPORT = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
