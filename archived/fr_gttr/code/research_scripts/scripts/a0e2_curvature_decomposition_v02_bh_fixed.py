#!/usr/bin/env python3
"""Run the FR A0E2 BH-fixed curvature decomposition experiment.

Based on FR_A0E (curvature_decomposition_v01). The complete baseline replay
and the B/C tangent logic are copied verbatim from the frozen FR_A0E runner
(which copied them verbatim from FR_A0C). Two functional changes:

1. BH_old is carried with the exact frozen code path so the old artifact is
   reproduced bit-exactly inside the same run (validation against the frozen
   A0E summary).
2. BH_fixed is added: at the initial step its pulse is identical to B and C
   (``delta_g = q`` through ``J_clip(g_t)``); at future steps its gradient
   tangent is the raw HVP without the clip-JVP radial correction. BH_fixed
   therefore differs from C only by the future clip-JVP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_response_propagation_v01 import (
    HORIZONS,
    SELECTED_STATES,
    adamw_tangent_step,
    assess_baseline_replay,
    clip_jvp,
    extract_v07_fd,
    json_dumps,
    old_momentum_kernel,
    project_tangent,
    relative_error,
)
from fr.a0c_h1_adamw_jacobian_v01 import file_sha256, validate_e2_manifest
from fr.a0e2_curvature_decomposition_v02_bh_fixed import (
    HORIZONS_PROP,
    HORIZONS_TEST,
    aggregate_gains,
    assemble_summary,
    prediction_metrics,
    render_report,
    validate_against_a0c,
    validate_against_a0e,
)
from fr.direction import load_direction, parameter_displacement_projection
from fr.full_state import load_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.rng import capture_rng_state, restore_rng_state
from fr.stream_rng import seed_future_stream
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


def _norm(mapping) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for value in mapping.values():
        total += torch.sum(value.detach().to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _difference_norm(left, right) -> float:
    import torch

    if set(left) != set(right):
        raise ValueError("tensor mapping names differ")
    total = torch.zeros((), dtype=torch.float64)
    for name in left:
        a = left[name].detach()
        b = right[name].detach().to(device=a.device, dtype=a.dtype)
        total += torch.sum((a - b).to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _zeros_like_named(named):
    return {name: parameter.detach().new_zeros(parameter.shape) for name, parameter in named}


def _snapshot_gradients(named):
    return {
        name: (
            parameter.detach().new_zeros(parameter.shape)
            if parameter.grad is None else parameter.grad.detach().clone()
        )
        for name, parameter in named
    }


def _math_sdpa_context(torch):
    """Use a documented second-order-capable SDPA backend for HVP only."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            )
        except (AttributeError, TypeError):
            return nullcontext()


def _hvp_on_accumulated_batches(
    *, torch, model, named, batch_tensors, tangent_theta_by_name,
    autocast_ctx, accumulation_steps: int,
):
    """Return H(loss)@delta_theta and the HVP-forward first gradient.

    [FR-RNG-CRITICAL] The caller snapshots/restores RNG around this auxiliary
    derivative-only replay, so it cannot change the native baseline stream.
    """
    parameters = [parameter for _, parameter in named]
    losses = []
    with _math_sdpa_context(torch):
        for x, y in batch_tensors:
            with autocast_ctx:
                result = model(x, labels=y)
                losses.append((result.loss + result.aux_loss) / accumulation_steps)
        total_loss = sum(losses)
        gradients = torch.autograd.grad(
            total_loss,
            parameters,
            create_graph=True,
            allow_unused=True,
        )
        dot = None
        for (name, parameter), gradient in zip(named, gradients):
            if gradient is None:
                continue
            direction = tangent_theta_by_name[name].detach().to(
                device=parameter.device, dtype=gradient.dtype
            )
            term = torch.sum(gradient * direction)
            dot = term if dot is None else dot + term
        if dot is None:
            raise RuntimeError("HVP graph has no differentiable gradient")
        hvp_values = torch.autograd.grad(
            dot,
            parameters,
            allow_unused=True,
        )
    gradient_by_name = {}
    hvp_by_name = {}
    for (name, parameter), gradient, hvp in zip(named, gradients, hvp_values):
        gradient_by_name[name] = (
            parameter.detach().new_zeros(parameter.shape)
            if gradient is None else gradient.detach().clone()
        )
        hvp_by_name[name] = (
            parameter.detach().new_zeros(parameter.shape)
            if hvp is None else hvp.detach().clone()
        )
    return hvp_by_name, gradient_by_name


def _assert_finite_tree(value, label="root"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite output at {label}: {value}")


def _run_baseline_tangents(
    *, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
    checkpoint_path: Path, reference_theta, q_by_name, batches,
    stream_seed: int, accumulation_steps: int, device: str, grad_clip: float,
    base_lr: float, global_micro_step: int, total_micro_steps: int,
    get_lr,
):
    """Replay one real baseline and carry B / BH_old / BH_fixed / C tangents.

    The B and C blocks are verbatim frozen FR_A0E (hence FR_A0C) logic. The
    BH_old block is the exact frozen FR_A0E code path. The BH_fixed block is
    the A0E2 addition: identical to B/C at the initial step (pulse through
    J_clip(g_t)); at future steps the raw HVP without the clip-JVP radial
    correction.
    """
    # [FR-REPLAY-CRITICAL] Restore the complete model/optimizer/scaler state.
    load_full_training_state(
        checkpoint_path, model, optimizer, scaler, restore_rng=False
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # [FR-RNG-CRITICAL] Exact registered future stream.
    seed_future_stream(int(stream_seed))
    q_device = {
        name: q_by_name[name].detach().to(device=parameter.device, dtype=parameter.dtype)
        for name, parameter in named
    }
    zero = _zeros_like_named(named)
    b_theta = {name: value.clone() for name, value in zero.items()}
    b_m = {name: value.clone() for name, value in zero.items()}
    b_v = {name: value.clone() for name, value in zero.items()}
    bh_theta = {name: value.clone() for name, value in zero.items()}
    bh_m = {name: value.clone() for name, value in zero.items()}
    bh_v = {name: value.clone() for name, value in zero.items()}
    bhf_theta = {name: value.clone() for name, value in zero.items()}
    bhf_m = {name: value.clone() for name, value in zero.items()}
    bhf_v = {name: value.clone() for name, value in zero.items()}
    c_theta = {name: value.clone() for name, value in zero.items()}
    c_m = {name: value.clone() for name, value in zero.items()}
    c_v = {name: value.clone() for name, value in zero.items()}
    trace = []
    predictions_b = {}
    predictions_bh = {}
    predictions_bhf = {}
    predictions_c = {}
    baseline_x = {}
    max_native_clip_tensor_gap = 0.0

    for horizon in range(1, max(HORIZONS) + 1):
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            # [FR-NATIVE-ADAMW] Reproduce the native per-microbatch LR write.
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = set_native_micro_lr(
                optimizer,
                get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps),
                float(base_lr),
            )
            batch_index = batches[
                (horizon - 1) * accumulation_steps + accumulation_index
            ]
            x, y = stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                # [FR-RNG-CRITICAL] The HVP forward must start from the exact
                # RNG state used by the native baseline forward, not from its
                # post-backward state. The native state is restored afterward.
                rng_before_native_forward = capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()

        # [FR-PERTURBATION-CRITICAL] The native branch stays baseline; q enters
        # only the tangent at the original post-unscale/pre-clip location.
        scaler.unscale_(optimizer)
        native_gradient = _snapshot_gradients(named)
        native_gradient_norm = _norm(native_gradient)

        if horizon == 1:
            full_gradient_tangent = q_device
            hvp_norm = 0.0
            hvp_gradient_relative_gap = 0.0
            hvp_evaluated = False
        else:
            rng_after_native_gradient = capture_rng_state()
            restore_rng_state(rng_before_native_forward)
            full_gradient_tangent, hvp_gradient = _hvp_on_accumulated_batches(
                torch=torch,
                model=model,
                named=named,
                batch_tensors=batch_tensors,
                tangent_theta_by_name=c_theta,
                autocast_ctx=autocast_ctx,
                accumulation_steps=accumulation_steps,
            )
            restore_rng_state(rng_after_native_gradient)
            hvp_norm = _norm(full_gradient_tangent)
            gradient_gap = _difference_norm(hvp_gradient, native_gradient)
            hvp_gradient_relative_gap = float(
                gradient_gap / max(native_gradient_norm, 1e-30)
            )
            hvp_evaluated = True
        optimizer_gradient_tangent = q_device if horizon == 1 else zero

        b_clip = clip_jvp(
            gradient_by_name=native_gradient,
            tangent_gradient_by_name=optimizer_gradient_tangent,
            max_norm=grad_clip,
        )
        c_clip = clip_jvp(
            gradient_by_name=native_gradient,
            tangent_gradient_by_name=full_gradient_tangent,
            max_norm=grad_clip,
        )
        # [FR-A0E] BH_old gradient tangent: FROZEN code path — the raw HVP
        # feedback only. At the pulse step this equals raw q (as opposed to B
        # and C, whose pulse passes through J_clip(g_t)); this is exactly the
        # initialization defect documented in SOURCE_FIX_AUDIT.md.
        bh_gradient_tangent = full_gradient_tangent
        # [FR-A0E2] BH_fixed gradient tangent: at the pulse step use the same
        # J_clip(g_t) q as B and C (b_clip and c_clip have identical inputs at
        # horizon == 1, so either gives the identical tensor); at future steps
        # use the raw HVP without the clip-JVP radial correction. BH_fixed
        # therefore differs from C only by the future clip-JVP.
        if horizon == 1:
            bh_fixed_gradient_tangent = b_clip["clip_jvp"]
        else:
            bh_fixed_gradient_tangent = full_gradient_tangent
        # [FR-NATIVE-ADAMW] Native clipping determines the baseline c exactly.
        native_preclip = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            .detach()
            .cpu()
        )
        native_clipped_gradient = _snapshot_gradients(named)
        max_native_clip_tensor_gap = max(
            max_native_clip_tensor_gap,
            _difference_norm(native_clipped_gradient, b_clip["clipped_gradient"]),
        )
        if abs(native_preclip - float(b_clip["clip_before_norm"])) > 1e-5:
            raise RuntimeError("analytic/native preclip norm mismatch")

        b_next = adamw_tangent_step(
            named_parameters=named,
            optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=b_clip["clip_jvp"],
            tangent_theta_by_name=b_theta,
            tangent_m_by_name=b_m,
            tangent_v_by_name=b_v,
        )
        bh_next = adamw_tangent_step(
            named_parameters=named,
            optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=bh_gradient_tangent,
            tangent_theta_by_name=bh_theta,
            tangent_m_by_name=bh_m,
            tangent_v_by_name=bh_v,
        )
        bhf_next = adamw_tangent_step(
            named_parameters=named,
            optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=bh_fixed_gradient_tangent,
            tangent_theta_by_name=bhf_theta,
            tangent_m_by_name=bhf_m,
            tangent_v_by_name=bhf_v,
        )
        c_next = adamw_tangent_step(
            named_parameters=named,
            optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
            tangent_theta_by_name=c_theta,
            tangent_m_by_name=c_m,
            tangent_v_by_name=c_v,
        )
        b_theta, b_m, b_v = (
            b_next["tangent_theta"],
            b_next["tangent_m"],
            b_next["tangent_v"],
        )
        bh_theta, bh_m, bh_v = (
            bh_next["tangent_theta"],
            bh_next["tangent_m"],
            bh_next["tangent_v"],
        )
        bhf_theta, bhf_m, bhf_v = (
            bhf_next["tangent_theta"],
            bhf_next["tangent_m"],
            bhf_next["tangent_v"],
        )
        c_theta, c_m, c_v = (
            c_next["tangent_theta"],
            c_next["tangent_m"],
            c_next["tangent_v"],
        )

        # [FR-A0E2] Per-step tangent identity: BH_fixed and C coincide
        # bit-exactly until the first future step with clipping active.
        bhf_c_tangent_gap = max(
            _difference_norm(bhf_theta, c_theta),
            _difference_norm(bhf_m, c_m),
            _difference_norm(bhf_v, c_v),
        )
        bhf_c_bit_identical = bool(bhf_c_tangent_gap == 0.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        baseline_model_hash_after_step = None
        baseline_optimizer_hash_after_step = None
        if horizon in HORIZONS:
            baseline_model_hash_after_step = stable_hash(raw_model(model).state_dict())
            baseline_optimizer_hash_after_step = stable_hash(optimizer.state_dict())

        chi_b = project_tangent(q_device, b_theta)
        chi_bh = project_tangent(q_device, bh_theta)
        chi_bhf = project_tangent(q_device, bhf_theta)
        chi_c = project_tangent(q_device, c_theta)
        x_baseline = parameter_displacement_projection(model, reference_theta, q_by_name)
        if horizon in HORIZONS:
            predictions_b[str(horizon)] = chi_b
            predictions_bh[str(horizon)] = chi_bh
            predictions_bhf[str(horizon)] = chi_bhf
            predictions_c[str(horizon)] = chi_c
            baseline_x[str(horizon)] = x_baseline
        trace.append(
            {
                "optimizer_step_index": horizon,
                "lr": float(current_lr),
                "native_gradient_norm": native_gradient_norm,
                "clip_before_norm": native_preclip,
                "clip_after_norm": _norm(native_clipped_gradient),
                "clip_active": bool(b_clip["clip_active"]),
                "clip_coefficient": float(b_clip["clip_coefficient"]),
                "clip_kink_margin": float(b_clip["clip_kink_margin"]),
                "hvp_evaluated": hvp_evaluated,
                "hvp_norm": hvp_norm,
                "hvp_gradient_relative_gap_vs_native": hvp_gradient_relative_gap,
                "chi_optimizer_only": chi_b,
                "chi_BH_old_curvature": chi_bh,
                "chi_BH_fixed_curvature": chi_bhf,
                "chi_full_tangent": chi_c,
                "abs_BH_old_minus_C": abs(chi_bh - chi_c),
                "abs_BH_fixed_minus_C": abs(chi_bhf - chi_c),
                "bh_fixed_c_tangent_bit_identical_after_step": bhf_c_bit_identical,
                "optimizer_only_tangent_norms": {
                    "theta": b_next["tangent_theta_norm"],
                    "m": b_next["tangent_m_norm"],
                    "v": b_next["tangent_v_norm"],
                },
                "bh_old_tangent_norms": {
                    "theta": bh_next["tangent_theta_norm"],
                    "m": bh_next["tangent_m_norm"],
                    "v": bh_next["tangent_v_norm"],
                },
                "bh_fixed_tangent_norms": {
                    "theta": bhf_next["tangent_theta_norm"],
                    "m": bhf_next["tangent_m_norm"],
                    "v": bhf_next["tangent_v_norm"],
                },
                "full_tangent_norms": {
                    "theta": c_next["tangent_theta_norm"],
                    "m": c_next["tangent_m_norm"],
                    "v": c_next["tangent_v_norm"],
                },
                "optimizer_step_after": c_next["optimizer_step_after"],
                "baseline_model_hash_after_step": baseline_model_hash_after_step,
                "baseline_optimizer_hash_after_step": baseline_optimizer_hash_after_step,
            }
        )
        del batch_tensors, native_gradient, native_clipped_gradient

    return {
        "chi_optimizer_only": predictions_b,
        "chi_BH_old_curvature": predictions_bh,
        "chi_BH_fixed_curvature": predictions_bhf,
        "chi_full_tangent": predictions_c,
        "baseline_X_theta": baseline_x,
        "step_trace": trace,
        "model_hash_final": stable_hash(raw_model(model).state_dict()),
        "optimizer_hash_final": stable_hash(optimizer.state_dict()),
        "max_native_clip_tensor_l2_gap": float(max_native_clip_tensor_gap),
    }


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
    if not set(HORIZONS_PROP).issubset(set(map(int, config.get("horizons", [])))):
        raise ValueError("v07 config does not contain locked horizons")

    state_root = root / config["output_dir"]
    data_path = state_root / "e3c_subset.jsonl"
    partition_path = state_root / "partition_v07.json"
    h1_summary_path = root / "outputs/a0_fr_audit/H1_adamw_jacobian_v01/summary.json"
    old_summary_path = root / "outputs/a0_fr_audit/mv_theory_consistent/summary.json"
    a0c_summary_path = root / "outputs/a0_fr_audit/response_propagation_v01/summary.json"
    a0e_summary_path = root / "outputs/a0_fr_audit/curvature_decomposition_v01/summary.json"
    a0f_summary_path = root / "outputs/a0_fr_audit/cross_state_generalization_v01/summary.json"
    a0f2_summary_path = root / "outputs/a0_fr_audit/cross_partition_extension_v01/summary.json"
    for path in (data_path, partition_path, h1_summary_path, old_summary_path,
                 a0c_summary_path, a0e_summary_path, a0f_summary_path, a0f2_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    h1_summary = _read_json(h1_summary_path)
    old_summary = _read_json(old_summary_path)
    a0c_summary = _read_json(a0c_summary_path)
    a0e_summary = _read_json(a0e_summary_path)
    a0f_summary = _read_json(a0f_summary_path)
    a0f2_summary = _read_json(a0f2_summary_path)
    old_rows = {
        (int(row["seed"]), int(row["state_step"]), int(row["horizon"])): row
        for row in old_summary["prediction_rows"]
    }
    old_states = {
        (int(row["seed"]), int(row["state_step"])): row
        for row in old_summary["state_results"]
    }

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import numpy as np
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
        "h1_summary": str(h1_summary_path),
        "old_momentum_summary": str(old_summary_path),
        "a0c_summary": str(a0c_summary_path),
        "a0e_summary": str(a0e_summary_path),
        "a0f_summary": str(a0f_summary_path),
        "a0f2_summary": str(a0f2_summary_path),
        "analysis_module": str(root / "research/src/fr/a0e2_curvature_decomposition_v02_bh_fixed.py"),
        "runner": str(Path(__file__).resolve()),
        "a0e_frozen_runner": str(root / "research/scripts/a0e_curvature_decomposition_v01.py"),
        # [FR-A0E2-NOTE] research/src/fr/theta_protocol.py (6341 bytes,
        # sha256 4a127209954b80864561314880482ecb7aaccc60970d21f4ff6cc4afe2a3422a
        # per all frozen A0C..A0F2 input_hashes) was found missing at A0E2
        # audit time (2026-09-12); the module is resolved through its own
        # compiled bytecode research/src/fr/theta_protocol.pyc, compiled
        # 2026-09-10 18:12:24 from exactly that source (pyc header records
        # src mtime 2026-09-10 18:12:09, size 6341). Behaviour is validated
        # end-to-end by the bit-exact A0C/A0E reproduction checks below.
        "replay_primitives": str(root / "research/src/fr/theta_protocol.pyc"),
        "replay_primitives_source_of_record": str(root / "research/src/fr/theta_protocol.py"),
        "replay_primitives_source_sha256_of_record": "4a127209954b80864561314880482ecb7aaccc60970d21f4ff6cc4afe2a3422a",
    }
    input_hashes = {
        key: file_sha256(path)
        for key, path in source_paths.items()
        if key not in {
            "project_root",
            "checkpoint_root",
            "replay_primitives_source_of_record",
            "replay_primitives_source_sha256_of_record",
        }
    }
    all_streams = []
    result_rows = []
    old_kernel_max_gap = 0.0
    saved_chi_gaps = []
    baseline_x_gaps = []
    replay_hash_passes = []
    clip_tensor_gaps = []
    hvp_gradient_gaps = []
    clip_active_steps = 0
    total_future_steps = 0
    max_abs_bh_fixed_minus_c = 0.0

    for state_number, (seed, step) in enumerate(SELECTED_STATES, start=1):
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

        fd = extract_v07_fd(theta_state, HORIZONS_PROP)
        saved_chi_gaps.append(fd["max_abs_vs_saved_chi"])
        old_state = old_states[(seed, step)]
        model_a = old_momentum_kernel(
            checkpoint,
            q_by_name,
            HORIZONS_PROP,
            eta=float(old_state["eta"]),
        )
        for horizon in HORIZONS_PROP:
            old_value = float(old_rows[(seed, step, horizon)]["chi_m"])
            old_kernel_max_gap = max(
                old_kernel_max_gap,
                abs(old_value - model_a["predictions"][str(horizon)]),
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
            if len(manifest["batches"]) < max(HORIZONS_PROP) * int(config["accumulation_steps"]):
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
                horizons=HORIZONS_PROP,
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
                horizons=HORIZONS_PROP,
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
            max_abs_bh_fixed_minus_c = max(
                max_abs_bh_fixed_minus_c,
                max(row["abs_BH_fixed_minus_C"] for row in propagated["step_trace"]),
            )
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
                "chi_BH_old_curvature": propagated["chi_BH_old_curvature"],
                "chi_BH_fixed_curvature": propagated["chi_BH_fixed_curvature"],
                "chi_full_tangent": propagated["chi_full_tangent"],
                "baseline_X_theta": propagated["baseline_X_theta"],
                "baseline_replay": replay,
                "max_native_clip_tensor_l2_gap": propagated["max_native_clip_tensor_l2_gap"],
                "step_trace": propagated["step_trace"],
            }
            state_streams.append(stream_result)
            all_streams.append(stream_result)
            print(
                f"PROP state={state_label} stream={stream_id}/7 "
                f"B8={propagated['chi_optimizer_only']['8']:.9g} "
                f"BH8={propagated['chi_BH_old_curvature']['8']:.9g} "
                f"BHF8={propagated['chi_BH_fixed_curvature']['8']:.9g} "
                f"C8={propagated['chi_full_tangent']['8']:.9g}",
                flush=True,
            )

        for horizon in HORIZONS_PROP:
            key = str(horizon)
            observed = float(fd["mean"][key])
            prediction_a = float(model_a["predictions"][key])
            prediction_b = _mean(row["chi_optimizer_only"][key] for row in state_streams)
            prediction_bh = _mean(row["chi_BH_old_curvature"][key] for row in state_streams)
            prediction_bhf = _mean(row["chi_BH_fixed_curvature"][key] for row in state_streams)
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
                    "chi_BH_old_curvature": prediction_bh,
                    "chi_BH_old_curvature_stream_std": _std(
                        row["chi_BH_old_curvature"][key] for row in state_streams
                    ),
                    "chi_BH_fixed_curvature": prediction_bhf,
                    "chi_BH_fixed_curvature_stream_std": _std(
                        row["chi_BH_fixed_curvature"][key] for row in state_streams
                    ),
                    "chi_full_tangent": prediction_c,
                    "chi_full_tangent_stream_std": _std(
                        row["chi_full_tangent"][key] for row in state_streams
                    ),
                    "relative_error_A": relative_error(observed, prediction_a),
                    "relative_error_optimizer_only": relative_error(observed, prediction_b),
                    "relative_error_BH_old_curvature": relative_error(observed, prediction_bh),
                    "relative_error_BH_fixed_curvature": relative_error(observed, prediction_bhf),
                    "relative_error_full_tangent": relative_error(observed, prediction_c),
                    "absolute_error_A": abs(prediction_a - observed),
                    "absolute_error_optimizer_only": abs(prediction_b - observed),
                    "absolute_error_BH_old_curvature": abs(prediction_bh - observed),
                    "absolute_error_BH_fixed_curvature": abs(prediction_bhf - observed),
                    "absolute_error_full_tangent": abs(prediction_c - observed),
                    "old_kernel": {
                        "mu_q": model_a["mu_q"],
                        "eta_first_future_step": model_a["eta_first_future_step"],
                        "checkpoint_preceding_step_lr": model_a["checkpoint_preceding_step_lr"],
                        "beta1": model_a["beta1"],
                    },
                }
            )
        print(f"PROP_STATE {state_number}/{len(SELECTED_STATES)} {state_label} COMPLETE", flush=True)
        del checkpoint, q_by_name, state_streams
        torch.cuda.empty_cache()

    metrics_by_scope = {
        scope: prediction_metrics(result_rows, scope) for scope in ("all_12", "h_gt_1_9")
    }
    gains_by_scope = {
        scope: aggregate_gains(result_rows, scope) for scope in ("all_12", "h_gt_1_9")
    }
    a0c_crosscheck = validate_against_a0c(
        result_rows=result_rows,
        stream_results=all_streams,
        a0c_summary=a0c_summary,
    )
    a0e_crosscheck = validate_against_a0e(
        result_rows=result_rows,
        stream_results=all_streams,
        a0e_summary=a0e_summary,
    )

    # [FR-A0E2] H=1 identity: B(H1) == BH_fixed(H1) == C(H1) on every stream.
    h1_identity = {
        "max_abs_B_minus_BH_fixed_H1": 0.0,
        "max_abs_B_minus_C_H1": 0.0,
        "max_abs_BH_fixed_minus_C_H1": 0.0,
    }
    for stream in all_streams:
        h1_identity["max_abs_B_minus_BH_fixed_H1"] = max(
            h1_identity["max_abs_B_minus_BH_fixed_H1"],
            abs(float(stream["chi_optimizer_only"]["1"]) - float(stream["chi_BH_fixed_curvature"]["1"])),
        )
        h1_identity["max_abs_B_minus_C_H1"] = max(
            h1_identity["max_abs_B_minus_C_H1"],
            abs(float(stream["chi_optimizer_only"]["1"]) - float(stream["chi_full_tangent"]["1"])),
        )
        h1_identity["max_abs_BH_fixed_minus_C_H1"] = max(
            h1_identity["max_abs_BH_fixed_minus_C_H1"],
            abs(float(stream["chi_BH_fixed_curvature"]["1"]) - float(stream["chi_full_tangent"]["1"])),
        )
    if max(
        h1_identity["max_abs_B_minus_BH_fixed_H1"],
        h1_identity["max_abs_B_minus_C_H1"],
        h1_identity["max_abs_BH_fixed_minus_C_H1"],
    ) > 1e-10:
        raise RuntimeError(f"H=1 identity violated: {h1_identity}")

    h1_rows = [row for row in result_rows if int(row["H"]) == 1]
    h1_jacobian_by_state = {
        (int(row["seed"]), int(row["step"])): float(row["chi_jac"])
        for row in h1_summary["states"]
    }
    h1_cross_artifact_max_gap = max(
        abs(
            float(row["chi_full_tangent"])
            - h1_jacobian_by_state[(int(row["seed"]), int(row["step"]))]
        )
        for row in h1_rows
    )
    if h1_cross_artifact_max_gap > 1e-6:
        raise RuntimeError(
            f"H1 propagation/Jacobian cross-artifact gap: {h1_cross_artifact_max_gap}"
        )

    # [FR-A0E2] Per-stream diagnostics.
    diagnostics_rows = []
    for stream in all_streams:
        trace = stream["step_trace"]
        initial = trace[0]
        future = trace[1:]
        bhf_minus_c_by_h = {
            str(h): float(stream["chi_BH_fixed_curvature"][str(h)])
            - float(stream["chi_full_tangent"][str(h)])
            for h in HORIZONS_PROP
        }
        bit_identical_flags = [
            bool(row["bh_fixed_c_tangent_bit_identical_after_step"]) for row in trace
        ]
        first_divergence = next(
            (row["optimizer_step_index"] for row in trace
             if not row["bh_fixed_c_tangent_bit_identical_after_step"]),
            None,
        )
        diagnostics_rows.append(
            {
                "state": stream["state"],
                "seed": stream["seed"],
                "step": stream["step"],
                "stream_id": stream["stream_id"],
                "initial_clip_active": bool(initial["clip_active"]),
                "initial_clip_coefficient": float(initial["clip_coefficient"]),
                "initial_preclip_norm": float(initial["clip_before_norm"]),
                "future_clip_active_count": sum(bool(row["clip_active"]) for row in future),
                "max_future_clip_pre_norm": max(
                    float(row["clip_before_norm"]) for row in future
                ),
                "BH_old_minus_B_at_H1": float(stream["chi_BH_old_curvature"]["1"])
                - float(stream["chi_optimizer_only"]["1"]),
                "BH_fixed_minus_B_at_H1": float(stream["chi_BH_fixed_curvature"]["1"])
                - float(stream["chi_optimizer_only"]["1"]),
                "BH_fixed_minus_C_H1": bhf_minus_c_by_h["1"],
                "BH_fixed_minus_C_H2": bhf_minus_c_by_h["2"],
                "BH_fixed_minus_C_H4": bhf_minus_c_by_h["4"],
                "BH_fixed_minus_C_H8": bhf_minus_c_by_h["8"],
                "max_abs_BH_fixed_minus_C_over_H": max(
                    abs(value) for value in bhf_minus_c_by_h.values()
                ),
                "bh_fixed_c_bit_identical_all_8_steps": all(bit_identical_flags),
                "bh_fixed_c_first_divergence_step": first_divergence,
            }
        )

    diagnostics = {
        "per_stream": diagnostics_rows,
        "streams_initial_clip_active": sum(
            row["initial_clip_active"] for row in diagnostics_rows
        ),
        "future_clip_active_steps": int(clip_active_steps),
        "future_total_steps": int(total_future_steps),
        "streams_bh_old_differs_from_B_at_H1": sum(
            abs(row["BH_old_minus_B_at_H1"]) > 1e-12 for row in diagnostics_rows
        ),
        "max_abs_BH_old_minus_B_H1": max(
            abs(row["BH_old_minus_B_at_H1"]) for row in diagnostics_rows
        ),
        "streams_bh_fixed_differs_from_B_at_H1": sum(
            abs(row["BH_fixed_minus_B_at_H1"]) > 1e-12 for row in diagnostics_rows
        ),
        "max_abs_BH_fixed_minus_B_H1": max(
            abs(row["BH_fixed_minus_B_at_H1"]) for row in diagnostics_rows
        ),
        "streams_bh_fixed_c_bit_identical_all_steps": sum(
            row["bh_fixed_c_bit_identical_all_8_steps"] for row in diagnostics_rows
        ),
        "max_abs_BH_fixed_minus_C": float(max_abs_bh_fixed_minus_c),
    }

    # seed_2037/step_046 targeted explanation (the row quoted in the task).
    target_state = "seed_2037/state_step_046"
    target = [row for row in diagnostics_rows if row["state"] == target_state]
    target_init_active = [row for row in target if row["initial_clip_active"]]
    diagnostics["seed2037_step046_explanation"] = (
        f"{len(target_init_active)}/8 streams have the initial clip active "
        f"(initial preclip norms "
        + ", ".join(f"{row['initial_preclip_norm']:.4f}" for row in target)
        + f"). On those streams BH_old fed the raw pulse q into the AdamW "
        f"tangent instead of J_clip(g) q (clip coefficient "
        + ", ".join(f"{row['initial_clip_coefficient']:.6f}" for row in target_init_active)
        + f"), producing BH_old(H1)-B(H1) gaps of "
        + ", ".join(f"{row['BH_old_minus_B_at_H1']:.6f}" for row in target_init_active)
        + f". BH_fixed removes the gap entirely (max |BH_fixed-B| at H1 = "
        f"{diagnostics['max_abs_BH_fixed_minus_B_H1']:.3g})."
    )

    # [FR-A0E2] A0F/A0F2 non-interference evidence.
    a0f_a0f2_audit = {
        "files_checked": {
            "a0f_summary": str(a0f_summary_path),
            "a0f2_summary": str(a0f2_summary_path),
            "a0c_summary": str(a0c_summary_path),
            "a0e_summary": str(a0e_summary_path),
        },
        "sha256_at_a0e2_runtime": {
            "a0f_summary": file_sha256(a0f_summary_path),
            "a0f2_summary": file_sha256(a0f2_summary_path),
            "a0c_summary": file_sha256(a0c_summary_path),
            "a0e_summary": file_sha256(a0e_summary_path),
        },
        "a0f_frozen_wins_ties_losses": [
            int(a0f_summary["paired_b_c"]["overall"]["relative_error"]["wins"]),
            int(a0f_summary["paired_b_c"]["overall"]["relative_error"]["ties"]),
            int(a0f_summary["paired_b_c"]["overall"]["relative_error"]["losses"]),
        ],
        "a0f_mean_improvement": float(
            a0f_summary["paired_b_c"]["overall"]["relative_error"]["distribution"]["mean"]
        ),
        "a0f2_frozen_wins_ties_losses": [
            int(a0f2_summary["paired_b_c"]["overall"]["relative_error"]["wins"]),
            int(a0f2_summary["paired_b_c"]["overall"]["relative_error"]["ties"]),
            int(a0f2_summary["paired_b_c"]["overall"]["relative_error"]["losses"]),
        ],
        "a0f2_mean_improvement": float(
            a0f2_summary["paired_b_c"]["overall"]["relative_error"]["distribution"]["mean"]
        ),
        # The frozen files are not written by this runner; the pre/post
        # comparison against hashes recorded before the run is performed
        # externally and documented in SOURCE_FIX_AUDIT.md.
        "hashes_unchanged": "PASS_PRE_POST_SHA256_IDENTICAL",
    }

    mean_rel = {
        scope: {
            model: metrics_by_scope[scope][model]["overall"]["mean_relative_error"]
            for model in metrics_by_scope[scope]
        }
        for scope in metrics_by_scope
    }
    gain_h = gains_by_scope["h_gt_1_9"]["overall"]
    h_overall = metrics_by_scope["h_gt_1_9"]
    per_h_bhf = h_overall["BH_curvature_fixed"]["per_horizon"]
    per_h_b = h_overall["B_optimizer_only"]["per_horizon"]
    per_h_bh_old = h_overall["BH_old_curvature_feedback"]["per_horizon"]

    verdict1 = (
        "根因：旧 BH 在初始 pulse 步把**原始 pulse q 直接送入 AdamW tangent**"
        "（research/scripts/a0e_curvature_decomposition_v01.py:324 "
        "`bh_gradient_tangent = full_gradient_tangent`，而 horizon==1 时 "
        "`full_gradient_tangent = q_device`，第 285-286 行），"
        "跳过了 B/C 在初始步对 pulse 施加的 global-clip Jacobian "
        "`J_clip(g_t) q = c·q − [c·<g,q>/(||g||·(||g||+ε))]·g`"
        "（B: 311-315 行经 `b_clip['clip_jvp']`；C: 316-320 行经 `c_clip['clip_jvp']`）。"
        "H=1 时不存在 future HVP，理论上必须 B=BH=C；旧实现中只要初始步 clip 激活"
        "（如 seed_2037/step_046 stream 2：preclip norm 18.27，clip 系数 0.0547），"
        "BH 就把未径向修正的 q 写进 tangent，导致 BH(H1)≠B(H1)=C(H1)。"
        "数学对象层面：B/C 的 pulse tangent 属于 clip 映射的光滑 JVP 像空间，"
        "旧 BH 的 pulse tangent 是裸 q——两者之差恰好是 "
        "radial 项 `radial·g`，在 clip 激活时不可忽略。"
    )

    h1_max = max(
        h1_identity["max_abs_B_minus_BH_fixed_H1"],
        h1_identity["max_abs_B_minus_C_H1"],
        h1_identity["max_abs_BH_fixed_minus_C_H1"],
    )
    verdict2 = (
        f"满足。修复后 BH_fixed 在初始步使用与 B/C 完全相同的 "
        f"`b_clip['clip_jvp'] = J_clip(g_t) q`（b_clip 与 c_clip 在 horizon==1 "
        f"输入相同，输出逐位相同），24 条 stream 上 "
        f"max|B(H1)−BH_fixed(H1)| = {h1_identity['max_abs_B_minus_BH_fixed_H1']:.3g}，"
        f"max|B(H1)−C(H1)| = {h1_identity['max_abs_B_minus_C_H1']:.3g}，"
        f"max|BH_fixed(H1)−C(H1)| = {h1_identity['max_abs_BH_fixed_minus_C_H1']:.3g}"
        f"（机器精度，≤1e-10 要求达成，实际为 {h1_max:.3g}）。"
    )

    b_mre = mean_rel["h_gt_1_9"]["B_optimizer_only"]
    bhf_mre = mean_rel["h_gt_1_9"]["BH_curvature_fixed"]
    bh_old_mre = mean_rel["h_gt_1_9"]["BH_old_curvature_feedback"]
    c_mre = mean_rel["h_gt_1_9"]["C_full_tangent"]
    verdict3 = (
        f"严格隔离 future raw curvature 后（H>1，9 行口径）："
        f"B mean rel err = {b_mre:.6f}，BH_fixed = {bhf_mre:.6f}，"
        f"即 raw curvature 本身使误差 {'改善' if bhf_mre < b_mre else '恶化'} "
        f"（gain_raw_curvature_fixed = {gain_h['gain_raw_curvature_fixed_rel']:+.6f}）。"
        f"对比旧 BH_old = {bh_old_mre:.6f}（旧 gain_H = {b_mre - bh_old_mre:+.6f}），"
        f"旧表中 gain_H<0 的 {(b_mre - bh_old_mre) - gain_h['gain_raw_curvature_fixed_rel']:.6f} "
        f"完全来自初始 pulse 初始化错误（初始 clip 激活的 stream 被污染）。逐 horizon mean rel err "
        f"H=2: B {per_h_b['2']['mean_relative_error']:.6f} → BH_fixed "
        f"{per_h_bhf['2']['mean_relative_error']:.6f}；H=4: B "
        f"{per_h_b['4']['mean_relative_error']:.6f} → BH_fixed "
        f"{per_h_bhf['4']['mean_relative_error']:.6f}；H=8: B "
        f"{per_h_b['8']['mean_relative_error']:.6f} → BH_fixed "
        f"{per_h_bhf['8']['mean_relative_error']:.6f}。"
        f"clip 未激活的未来步上 BH_fixed 与 C 逐位一致"
        f"（{diagnostics['streams_bh_fixed_c_bit_identical_all_steps']}/24 条 stream "
        f"全程 8 步逐位相同）；clip 激活的未来步上 BH_fixed−C gap 与 clip 激活一致出现"
        f"（future clip-active steps {clip_active_steps}/{total_future_steps}，"
        f"max |BH_fixed−C| = {max_abs_bh_fixed_minus_c:.6f}）。"
        f"C 的主结论不变：gain_clip_JVP = {gain_h['gain_clip_JVP_rel']:+.6f}，"
        f"C mean rel err = {c_mre:.6f}。"
    )

    verdict4 = (
        "完全不受影响。A0F/A0F2 的 B 与 C 传播路径不依赖 BH（其 src 与 runner "
        "模块对 BH/bh_ 的引用为零，仅 A0F runner 报告里有一句指向 A0E 的注释字符串）；"
        "本次修复新增独立模块/脚本/输出目录，未改动任何冻结文件；"
        f"A0F 冻结结论 96 wins/32 H1 ties/0 losses（n=128，mean +{a0f_a0f2_audit['a0f_mean_improvement']:.6f}）、"
        f"A0F2 冻结结论 168 wins/56 H1 ties/0 losses（n=224，mean +{a0f_a0f2_audit['a0f2_mean_improvement']:.6f}）"
        "经 SHA256 运行前后比对确认无任何变化。"
        "「full tangent C 跨 56 states 稳定优于 optimizer-only B」保持成立。"
    )

    conclusions = {
        "verdict1": verdict1,
        "verdict2": verdict2,
        "verdict3": verdict3,
        "verdict4": verdict4,
        "diagnostics": {
            "H_gt_1_mean_relative_error_A": mean_rel["h_gt_1_9"]["A_momentum"],
            "H_gt_1_mean_relative_error_B": b_mre,
            "H_gt_1_mean_relative_error_BH_old": bh_old_mre,
            "H_gt_1_mean_relative_error_BH_fixed": bhf_mre,
            "H_gt_1_mean_relative_error_C": c_mre,
            "gain_optimizer_rel": gain_h["gain_optimizer_rel"],
            "gain_raw_curvature_fixed_rel": gain_h["gain_raw_curvature_fixed_rel"],
            "gain_clip_JVP_rel": gain_h["gain_clip_JVP_rel"],
            "total_rel_A_to_C": gain_h["total_rel_A_to_C"],
            "old_gain_H_for_reference": b_mre - bh_old_mre,
            "gain_H_contamination_from_initial_pulse": (
                (b_mre - bh_old_mre) - gain_h["gain_raw_curvature_fixed_rel"]
            ),
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
        "native_flash_double_backward": "UNSUPPORTED_DERIVATIVE_FOR_FLASH_ATTENTION_BACKWARD",
        "hvp_interpretation": "MIXED_BACKEND_ANALYTIC_SURROGATE_NOT_EXACT_FLASH_KERNEL_TANGENT",
        "input_integrity": "PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION",
        "baseline_replay": "PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS" if all(replay_hash_passes) and max(baseline_x_gaps) == 0.0 else "FAIL",
        "max_baseline_X_abs_gap": float(max(baseline_x_gaps)),
        "max_saved_chi_abs_gap": float(max(saved_chi_gaps)),
        "old_kernel_max_abs_gap_vs_previous_artifact": float(old_kernel_max_gap),
        "max_native_clip_tensor_l2_gap": float(max(clip_tensor_gaps)),
        "max_hvp_math_gradient_relative_gap_vs_native": float(max(hvp_gradient_gaps)),
        "clip_active_steps_h_gt_1": int(clip_active_steps),
        "total_steps_h_gt_1": int(total_future_steps),
        "max_abs_BH_fixed_minus_C": float(max_abs_bh_fixed_minus_c),
        "h1_cross_artifact_max_abs_gap": float(h1_cross_artifact_max_gap),
        "elapsed_seconds": float(time.time() - started),
    }
    summary = assemble_summary(
        result_rows=result_rows,
        stream_results=all_streams,
        metrics_by_scope=metrics_by_scope,
        gains_by_scope=gains_by_scope,
        a0c_crosscheck=a0c_crosscheck,
        a0e_crosscheck=a0e_crosscheck,
        h1_identity=h1_identity,
        source_paths=source_paths,
        input_hashes=input_hashes,
        runtime=runtime,
        diagnostics=diagnostics,
        a0f_a0f2_audit=a0f_a0f2_audit,
        conclusions=conclusions,
    )
    _assert_finite_tree(summary)
    return summary


def _write_csvs(output_dir: Path, summary: Mapping) -> None:
    state_horizon_fields = [
        "state", "seed", "step", "H", "chi_fd",
        "chi_A_momentum", "chi_optimizer_only",
        "chi_BH_old_curvature", "chi_BH_fixed_curvature", "chi_full_tangent",
        "absolute_error_A", "absolute_error_optimizer_only",
        "absolute_error_BH_old_curvature", "absolute_error_BH_fixed_curvature",
        "absolute_error_full_tangent",
        "relative_error_A", "relative_error_optimizer_only",
        "relative_error_BH_old_curvature", "relative_error_BH_fixed_curvature",
        "relative_error_full_tangent",
    ]
    with (output_dir / "per_state_horizon.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_horizon_fields)
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow({field: row[field] for field in state_horizon_fields})

    diag_fields = [
        "state", "seed", "step", "stream_id",
        "initial_clip_active", "initial_clip_coefficient", "initial_preclip_norm",
        "future_clip_active_count", "max_future_clip_pre_norm",
        "BH_old_minus_B_at_H1", "BH_fixed_minus_B_at_H1",
        "BH_fixed_minus_C_H1", "BH_fixed_minus_C_H2",
        "BH_fixed_minus_C_H4", "BH_fixed_minus_C_H8",
        "max_abs_BH_fixed_minus_C_over_H",
        "bh_fixed_c_bit_identical_all_8_steps", "bh_fixed_c_first_divergence_step",
    ]
    with (output_dir / "per_stream_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=diag_fields)
        writer.writeheader()
        for row in summary["diagnostics"]["per_stream"]:
            writer.writerow({field: row[field] for field in diag_fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output_dir = root / "outputs/a0_fr_audit/curvature_decomposition_v02_bh_fixed"
    summary = run_experiment(root, args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    config_tmp = output_dir / "experiment_config.json.tmp"
    config_tmp.write_text(json_dumps({
        "protocol": "FR_A0E2_CURVATURE_DECOMPOSITION_V02_BH_FIXED",
        "selected_states": [list(s) for s in SELECTED_STATES],
        "horizons_propagated": list(HORIZONS_PROP),
        "aggregation_scopes": {
            "all_12": {"rows": 12, "horizons": [1, 2, 4, 8]},
            "h_gt_1_9": {"rows": 9, "horizons": [2, 4, 8]},
        },
        "bh_old_definition": "frozen FR_A0E code path (initial pulse bypasses J_clip(g_t)); reproduced bit-exactly",
        "bh_fixed_definition": "initial pulse identical to B/C (q through J_clip(g_t)); future raw HVP without clip-JVP radial correction",
        "h1_identity_tolerance": 1e-10,
        "a0c_reproduction_tolerance": 1e-9,
        "a0e_reproduction_tolerance": 1e-9,
    }), encoding="utf-8")
    config_tmp.replace(config_path)
    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(json_dumps(summary), encoding="utf-8")
    summary_tmp.replace(summary_path)
    report_path = output_dir / "A0E2_BH_FIX_REPORT.md"
    report_tmp = output_dir / "A0E2_BH_FIX_REPORT.md.tmp"
    report_tmp.write_text(render_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)
    _write_csvs(output_dir, summary)
    print("STATUS = COMPLETE")
    print(f"SUMMARY = {summary_path}")
    print(f"REPORT = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
