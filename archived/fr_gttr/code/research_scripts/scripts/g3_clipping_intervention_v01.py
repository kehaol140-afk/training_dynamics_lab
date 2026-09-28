#!/usr/bin/env python3
"""G3 — clipping intervention on the frozen 26M FD/B/BH_fixed/C mechanism.

Subcommands (run in order):
  preregister      write experiment_config/run_manifest/hashes/environment
  g3a_operator     operator-level control on the frozen G1a clip=1.0 bank
  train            26M no-clip state bank (seeds 2042/2043/2044, clip disabled)
  theta            q + E1 + frozen manifests + E2 FD branches (no-clip sentinel)
  smoke            no-clip numerical/implementation gate (seed 2042, step 26)
  clipref          BH_fixed for G1a clip=1.0 seeds 2043/2044 (paired interaction)
  tangents         B/BH_fixed/C replay on all 9 no-clip states x 4 streams
  analyze          interaction statistics, frozen-threshold verdict, report
  audit            SHA256SUMS + audit bundle (excludes large checkpoints)

[FR-OBSERVATION-ONLY] v07 bank, frozen a0* modules/outputs, and the frozen
G1a/G1b artifact trees are read-only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Mapping

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from fr.a0c_h1_adamw_jacobian_v01 import file_sha256
from fr.direction import (
    direction_norm,
    load_direction,
    normalize_direction,
    parameter_displacement_projection,
    save_direction,
)
from fr.full_state import load_full_training_state, save_full_training_state
from fr.g1a_26m_scale_generalization_v01 import (
    E1_BASE_SEED,
    E1_STREAMS,
    E2_BASE_SEED,
    E2_STREAMS,
    HORIZONS_OFFICIAL,
    RHO,
    STAGE_FRACTIONS,
    STAGE_STEPS,
    STAGE_TOKENS,
    STAGES,
    TOTAL_OPT_STEPS,
    TRAIN_SEEDS,
    batch_seed_from_stream_seed,
    stage_of_step,
    two_level_seed,
    validate_g1a_e2_manifest,
)
from fr.g3_clipping_intervention_v01 import (
    CLIP_DISABLED_MAX_NORM,
    E2_MANIFEST_PROTOCOL,
    H_GT_1,
    NOCLIP_STATE_PROTOCOL,
    PROTOCOL,
    Q_PROTOCOL,
    VERDICT_THRESHOLDS,
    aggregate_noclip_metrics,
    cell_relative_errors_bhc,
    cluster_bootstrap_interaction,
    decide_mechanism_verdict,
    json_dumps,
    render_report,
    validate_g3_e2_manifest,
)
from fr.batch_manifest import save_manifest
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import load_partition
from fr.response_stats import paired_linearity_summary
from fr.rng import capture_rng_state, restore_rng_state
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import seed_future_stream
from fr.theta_protocol import run_theta_branch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/generalization_g3_clipping_intervention_v01"
BANK = OUT / "bank"
THETA = OUT / "theta"
OPERATOR = OUT / "operator_control"
NOCLIP = OUT / "no_clip_regime"
CLIPREF = OUT / "clip_reference"
RESULTS = OUT / "results"
CONFIG_PATH = ROOT / "research/configs/g3_clipping_intervention_v01.json"
V07_DIR = ROOT / "outputs/e3_confirmatory/v07"
G1A_OUT = ROOT / "outputs/generalization_g1a_26m_v01"
COMMANDS_LOG = OUT / "COMMANDS.log"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log_command(argv) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with COMMANDS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(map(str, argv))}\n")


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _frozen_runner_import(script_name: str, import_as: str):
    path = RESEARCH_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(import_as, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimind_setup(device: str):
    minimind = ROOT / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    cfg = _read_json(CONFIG_PATH)
    mcfg = cfg["model"]
    mmcfg = MiniMindConfig(
        hidden_size=int(mcfg["hidden_size"]),
        num_hidden_layers=int(mcfg["num_hidden_layers"]),
        max_seq_len=int(mcfg["max_seq_len"]),
        num_attention_heads=int(mcfg["num_attention_heads"]),
        num_key_value_heads=int(mcfg["num_key_value_heads"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg,
        from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=device,
    )
    return torch, model, tokenizer, get_lr, setup_seed


def _count_params(model) -> dict:
    raw = raw_model(model)
    unique = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    elements = sum(t.numel() for t in raw.state_dict().values())
    return {"unique_params": int(unique), "state_dict_elements": int(elements)}


def _assert_finite_tree(value, label="root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite output at {label}: {value}")


def _state_paths(base: Path = BANK):
    for seed in TRAIN_SEEDS:
        for stage in STAGES:
            step = STAGE_STEPS[stage]
            yield seed, stage, step, base / f"seed_{seed}" / "states" / f"state_step_{step:03d}.pt"


def _g1a_state_paths():
    for seed in TRAIN_SEEDS:
        for stage in STAGES:
            step = STAGE_STEPS[stage]
            yield seed, stage, step, G1A_OUT / "bank" / f"seed_{seed}" / "states" / f"state_step_{step:03d}.pt"


# ---------------------------------------------------------------------------
# preregister
# ---------------------------------------------------------------------------

def cmd_preregister(_args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _read_json(CONFIG_PATH)
    shutil.copyfile(CONFIG_PATH, OUT / "experiment_config.json")

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_env": "fr_minimind",
        "torch_version": None,
        "cuda_version": None,
        "gpu": None,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        env["torch_error"] = str(exc)
    _write_json(OUT / "ENVIRONMENT.json", env)

    sources = {
        "g3_config": CONFIG_PATH,
        "g3_protocol_md": OUT / "G3_PROTOCOL.md",
        "g3_module": SRC_ROOT / "fr/g3_clipping_intervention_v01.py",
        "g3_runner": RESEARCH_ROOT / "scripts/g3_clipping_intervention_v01.py",
        "g1a_config": ROOT / "research/configs/g1a_26m_v01.json",
        "frozen_a0c_module": SRC_ROOT / "fr/a0c_adamw_response_propagation_v01.py",
        "frozen_a0c_runner": RESEARCH_ROOT / "scripts/a0c_adamw_response_propagation_v01.py",
        "frozen_a0e2_module": SRC_ROOT / "fr/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "frozen_a0e2_runner": RESEARCH_ROOT / "scripts/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "theta_protocol_pyc": SRC_ROOT / "fr/theta_protocol.pyc",
        "minimind_model": ROOT / "minimind/model/model_minimind.py",
        "minimind_trainer_utils": ROOT / "minimind/trainer/trainer_utils.py",
        "v07_data_d0": V07_DIR / "e3c_subset.jsonl",
        "v07_partition": V07_DIR / "partition_v07.json",
        "g1a_summary": G1A_OUT / "results/summary.json",
        "g1a_per_stream": G1A_OUT / "results/per_stream.json",
        "g1a_mechanism_session": G1A_OUT / "mechanism/_session.json",
    }
    lines = ["# SOURCE_AND_INPUT_HASHES (FR_G3_CLIPPING_INTERVENTION_V01)", ""]
    hashes = {}
    for key, path in sources.items():
        digest = file_sha256(path) if Path(path).is_file() else "MISSING"
        hashes[key] = digest
        lines.append(f"{digest}  {key}  {path}")
    (OUT / "SOURCE_AND_INPUT_HASHES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "protocol": PROTOCOL,
        "created_at_utc": env["recorded_at_utc"],
        "experiment_config": "experiment_config.json",
        "phases": ["preregister", "g3a_operator", "train", "theta", "smoke",
                   "clipref", "tangents", "analyze", "audit"],
        "phase_status": {p: "NOT_STARTED" for p in
                         ["g3a_operator", "train", "theta", "smoke", "clipref",
                          "tangents", "analyze", "audit"]},
        "source_hashes": hashes,
        "output_subdirs": ["operator_control", "bank", "theta", "clip_reference",
                           "no_clip_regime", "results"],
    }
    _write_json(OUT / "run_manifest.json", manifest)
    print("PREREGISTER_DONE")
    return 0


def _manifest_touch(phase: str, status: str, extra: dict | None = None) -> None:
    path = OUT / "run_manifest.json"
    manifest = _read_json(path)
    manifest["phase_status"][phase] = status
    if extra:
        manifest.setdefault("phase_details", {})[phase] = extra
    _write_json(path, manifest)


# ---------------------------------------------------------------------------
# shared tangent-session driver
# ---------------------------------------------------------------------------

_TANGENT_MODEL = None
_TANGENT_OPTIMIZER = None
_TANGENT_SCALER = None
_TANGENT_DEVICE = None


def _setup_tangent_session(device: str):
    global _TANGENT_MODEL, _TANGENT_OPTIMIZER, _TANGENT_SCALER, _TANGENT_DEVICE
    torch, model, tokenizer, get_lr, _ = _minimind_setup(device)
    cfg = _read_json(CONFIG_PATH)
    autocast_ctx, scaler, _ = make_amp(torch, device, cfg["training"]["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    assert_native_adamw(optimizer)
    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(
        str(V07_DIR / "e3c_subset.jsonl"), tokenizer,
        max_length=int(cfg["model"]["max_seq_len"]),
    )
    _TANGENT_MODEL = model
    _TANGENT_OPTIMIZER = optimizer
    _TANGENT_SCALER = scaler
    _TANGENT_DEVICE = device
    return torch, model, optimizer, scaler, autocast_ctx, ds, get_lr


# ---------------------------------------------------------------------------
# G3a ablation driver: C with future-step clip-JVP replaced by identity.
# Faithful copy of the frozen a0e2 runner's _run_baseline_tangents with a
# single flagged difference (marked [FR-G3A]); everything else identical.
# ---------------------------------------------------------------------------

def _run_baseline_tangents_c_identity(
    *, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
    checkpoint_path: Path, reference_theta, q_by_name, batches,
    stream_seed: int, accumulation_steps: int, device: str, grad_clip: float,
    base_lr: float, global_micro_step: int, total_micro_steps: int,
    get_lr, helpers,
):
    from fr.a0c_adamw_response_propagation_v01 import (
        HORIZONS,
        adamw_tangent_step,
        clip_jvp,
        project_tangent,
    )

    _zeros_like_named = helpers._zeros_like_named
    _snapshot_gradients = helpers._snapshot_gradients
    _hvp_on_accumulated_batches = helpers._hvp_on_accumulated_batches
    _norm = helpers._norm
    _difference_norm = helpers._difference_norm

    load_full_training_state(checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))
    q_device = {
        name: q_by_name[name].detach().to(device=parameter.device, dtype=parameter.dtype)
        for name, parameter in named
    }
    zero = _zeros_like_named(named)
    b_theta = {name: value.clone() for name, value in zero.items()}
    b_m = {name: value.clone() for name, value in zero.items()}
    b_v = {name: value.clone() for name, value in zero.items()}
    bhf_theta = {name: value.clone() for name, value in zero.items()}
    bhf_m = {name: value.clone() for name, value in zero.items()}
    bhf_v = {name: value.clone() for name, value in zero.items()}
    c_theta = {name: value.clone() for name, value in zero.items()}
    c_m = {name: value.clone() for name, value in zero.items()}
    c_v = {name: value.clone() for name, value in zero.items()}
    trace = []
    predictions_b = {}
    predictions_bhf = {}
    predictions_c_identity = {}
    baseline_x = {}
    max_native_clip_tensor_gap = 0.0

    for horizon in range(1, max(HORIZONS) + 1):
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = set_native_micro_lr(
                optimizer, get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr),
            )
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                rng_before_native_forward = capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()

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
                torch=torch, model=model, named=named, batch_tensors=batch_tensors,
                tangent_theta_by_name=c_theta, autocast_ctx=autocast_ctx,
                accumulation_steps=accumulation_steps,
            )
            restore_rng_state(rng_after_native_gradient)
            hvp_norm = _norm(full_gradient_tangent)
            gradient_gap = _difference_norm(hvp_gradient, native_gradient)
            hvp_gradient_relative_gap = float(gradient_gap / max(native_gradient_norm, 1e-30))
            hvp_evaluated = True
        optimizer_gradient_tangent = q_device if horizon == 1 else zero

        b_clip = clip_jvp(
            gradient_by_name=native_gradient,
            tangent_gradient_by_name=optimizer_gradient_tangent,
            max_norm=grad_clip,
        )
        if horizon == 1:
            # pulse step unchanged: real J_clip(g_t) q for both B and C_identity
            c_clip_jvp = clip_jvp(
                gradient_by_name=native_gradient,
                tangent_gradient_by_name=full_gradient_tangent,
                max_norm=grad_clip,
            )["clip_jvp"]
        else:
            # [FR-G3A] operator control: future-step clip-JVP replaced by identity
            c_clip_jvp = full_gradient_tangent
        bh_fixed_gradient_tangent = (
            b_clip["clip_jvp"] if horizon == 1 else full_gradient_tangent
        )
        native_preclip = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            .detach().cpu()
        )
        native_clipped_gradient = _snapshot_gradients(named)
        max_native_clip_tensor_gap = max(
            max_native_clip_tensor_gap,
            _difference_norm(native_clipped_gradient, b_clip["clipped_gradient"]),
        )
        if abs(native_preclip - float(b_clip["clip_before_norm"])) > 1e-5:
            raise RuntimeError("analytic/native preclip norm mismatch")

        b_next = adamw_tangent_step(
            named_parameters=named, optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=b_clip["clip_jvp"],
            tangent_theta_by_name=b_theta, tangent_m_by_name=b_m, tangent_v_by_name=b_v,
        )
        bhf_next = adamw_tangent_step(
            named_parameters=named, optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=bh_fixed_gradient_tangent,
            tangent_theta_by_name=bhf_theta, tangent_m_by_name=bhf_m, tangent_v_by_name=bhf_v,
        )
        c_next = adamw_tangent_step(
            named_parameters=named, optimizer=optimizer,
            clipped_gradient_by_name=native_clipped_gradient,
            tangent_clipped_gradient_by_name=c_clip_jvp,
            tangent_theta_by_name=c_theta, tangent_m_by_name=c_m, tangent_v_by_name=c_v,
        )
        b_theta, b_m, b_v = b_next["tangent_theta"], b_next["tangent_m"], b_next["tangent_v"]
        bhf_theta, bhf_m, bhf_v = bhf_next["tangent_theta"], bhf_next["tangent_m"], bhf_next["tangent_v"]
        c_theta, c_m, c_v = c_next["tangent_theta"], c_next["tangent_m"], c_next["tangent_v"]

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        chi_b = project_tangent(q_device, b_theta)
        chi_bhf = project_tangent(q_device, bhf_theta)
        chi_c_ident = project_tangent(q_device, c_theta)
        x_baseline = parameter_displacement_projection(model, reference_theta, q_by_name)
        if horizon in HORIZONS:
            predictions_b[str(horizon)] = chi_b
            predictions_bhf[str(horizon)] = chi_bhf
            predictions_c_identity[str(horizon)] = chi_c_ident
            baseline_x[str(horizon)] = x_baseline
        bhf_c_gap = max(
            _difference_norm(bhf_theta, c_theta),
            _difference_norm(bhf_m, c_m),
            _difference_norm(bhf_v, c_v),
        )
        trace.append({
            "optimizer_step_index": horizon,
            "clip_active": bool(b_clip["clip_active"]),
            "clip_coefficient": float(b_clip["clip_coefficient"]),
            "hvp_evaluated": hvp_evaluated,
            "hvp_norm": hvp_norm,
            "hvp_gradient_relative_gap_vs_native": hvp_gradient_relative_gap,
            "chi_optimizer_only": chi_b,
            "chi_BH_fixed_curvature": chi_bhf,
            "chi_C_identity": chi_c_ident,
            "abs_BH_fixed_minus_C_identity": abs(chi_bhf - chi_c_ident),
            "bh_fixed_c_identity_bit_identical_after_step": bool(bhf_c_gap == 0.0),
            "optimizer_step_after": c_next["optimizer_step_after"],
        })

    return {
        "chi_optimizer_only": predictions_b,
        "chi_BH_fixed_curvature": predictions_bhf,
        "chi_C_identity": predictions_c_identity,
        "baseline_X_theta": baseline_x,
        "step_trace": trace,
        "model_hash_final": stable_hash(raw_model(model).state_dict()),
        "optimizer_hash_final": stable_hash(optimizer.state_dict()),
        "max_native_clip_tensor_l2_gap": float(max_native_clip_tensor_gap),
    }


# ---------------------------------------------------------------------------
# G3a operator control
# ---------------------------------------------------------------------------

def _load_g1a_theta_state(seed: int, step: int):
    theta = _read_json(G1A_OUT / f"theta/seed_{seed}/theta_mapping_raw.json")
    if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
        raise ValueError(f"invalid frozen G1a theta record for seed {seed}")
    state = next((r for r in theta["states"] if int(r["step"]) == step), None)
    if state is None or len(state.get("e2_streams", [])) != E2_STREAMS:
        raise ValueError(f"missing frozen G1a state/streams: seed={seed} step={step}")
    return state


def cmd_g3a_operator(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import("a0e2_curvature_decomposition_v02_bh_fixed.py", "g3_frozen_a0e2_runner")
    accum = int(cfg["training"]["accumulation_steps"])
    grad_clip = 1.0  # G1a frozen clip regime
    OPERATOR.mkdir(parents=True, exist_ok=True)

    rows = []
    max_abs_gap = 0.0
    sum_abs_gap = 0.0
    max_rel_gap = 0.0
    frozen_cross_max = 0.0
    abl_vs_real_max = 0.0
    n_cells = 0
    bit_identical_cells = 0

    for seed, stage, step, state_path in _g1a_state_paths():
        if seed != TRAIN_SEEDS[0]:
            continue
        if not state_path.is_file():
            raise FileNotFoundError(f"missing frozen G1a checkpoint {state_path}")
        theta_state = _load_g1a_theta_state(seed, step)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        q_by_name, _q_meta, q_sha = load_direction(theta_state["q_path"])
        if q_sha != theta_state["q_sha256"]:
            raise ValueError("q identity mismatch vs frozen G1a record")
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            stream_id = int(rec["stream_id"])
            manifest = validate_g1a_e2_manifest(
                _read_json(Path(rec["batch_manifest_path"])),
                expected_seed=seed, expected_step=step, expected_stream=stream_id,
                expected_content_sha=rec["batch_manifest_sha256"], accumulation_steps=accum,
            )
            meta = load_full_training_state(state_path, model, optimizer, scaler, restore_rng=False)
            common = dict(
                checkpoint_path=state_path, reference_theta=payload["model"],
                q_by_name=q_by_name, batches=manifest["batches"],
                stream_seed=int(rec["stream_seed"]), accumulation_steps=accum,
                device=device, grad_clip=grad_clip,
                base_lr=float(cfg["training"]["learning_rate"]),
                global_micro_step=int(meta["global_micro_step"]),
                total_micro_steps=int(meta["total_micro_steps"]),
                get_lr=get_lr,
            )
            real = a0e2_runner._run_baseline_tangents(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named, **common,
            )
            abl = _run_baseline_tangents_c_identity(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named,
                **common, helpers=a0e2_runner,
            )
            frozen_mech = _read_json(G1A_OUT / "mechanism" / f"stream_seed{seed}_step{step:03d}_s{stream_id}.json")
            for h in HORIZONS_OFFICIAL:
                hk = str(h)
                chi_bhf_abl = float(abl["chi_BH_fixed_curvature"][hk])
                chi_c_ident = float(abl["chi_C_identity"][hk])
                chi_bhf_real = float(real["chi_BH_fixed_curvature"][hk])
                chi_c_real = float(real["chi_full_tangent"][hk])
                chi_bhf_frozen = float(frozen_mech["chi_BH_fixed"][hk])
                chi_c_frozen = float(frozen_mech["chi_C"][hk])
                gap = abs(chi_c_ident - chi_bhf_abl)
                rel = gap / max(abs(chi_bhf_abl), 1e-30)
                # Cross-check is only meaningful for the UNMODIFIED frozen
                # runner re-run vs the frozen G1a JSONs.  The ablation run's
                # BH_fixed is NOT expected to match across runs: the frozen
                # HVP leg is taken along the C tangent, which the identity
                # override deliberately alters from horizon 3 onward.
                cross = max(
                    abs(chi_bhf_real - chi_bhf_frozen),
                    abs(chi_c_real - chi_c_frozen),
                )
                abl_vs_real = abs(chi_bhf_abl - chi_bhf_real)
                max_abs_gap = max(max_abs_gap, gap)
                sum_abs_gap += gap
                max_rel_gap = max(max_rel_gap, rel)
                frozen_cross_max = max(frozen_cross_max, cross)
                abl_vs_real_max = max(abl_vs_real_max, abl_vs_real)
                n_cells += 1
                if gap == 0.0:
                    bit_identical_cells += 1
                rows.append({
                    "seed": seed, "stage": stage, "step": step, "stream_id": stream_id, "H": h,
                    "chi_BH_fixed_ablation": chi_bhf_abl,
                    "chi_C_identity": chi_c_ident,
                    "abs_gap_C_identity_vs_BH_fixed": gap,
                    "rel_gap_C_identity_vs_BH_fixed": rel,
                    "chi_BH_fixed_frozen_run": chi_bhf_real,
                    "chi_C_frozen_run": chi_c_real,
                    "chi_BH_fixed_frozen_g1a_json": chi_bhf_frozen,
                    "chi_C_frozen_g1a_json": chi_c_frozen,
                    "frozen_cross_check_abs_gap": cross,
                    "ablation_BH_fixed_vs_real_abs_gap": abl_vs_real,
                    "bit_identical": bool(gap == 0.0),
                })
            print(f"G3A seed={seed} step={step} stream={stream_id} DONE", flush=True)
        torch.cuda.empty_cache()

    passed = bool(max_abs_gap <= VERDICT_THRESHOLDS["g3a_max_abs_gap_tol"])
    summary = {
        "protocol": PROTOCOL,
        "subphase": "G3a_operator_control",
        "pass": passed,
        "tolerance_abs": VERDICT_THRESHOLDS["g3a_max_abs_gap_tol"],
        "tolerance_rel": VERDICT_THRESHOLDS["g3a_max_rel_gap_tol"],
        "n_cells": n_cells,
        "bit_identical_cells": bit_identical_cells,
        "max_abs_gap": float(max_abs_gap),
        "mean_abs_gap": float(sum_abs_gap / max(n_cells, 1)),
        "max_rel_gap": float(max_rel_gap),
        "frozen_cross_check_max_abs_gap": float(frozen_cross_max),
        "ablation_BH_fixed_vs_unmodified_real_max_abs_gap": float(abl_vs_real_max),
        "ablation_divergence_note": (
            "expected and NOT an implementation inconsistency: the frozen HVP leg "
            "is taken along the C tangent, which the identity override alters from "
            "horizon 3 onward; within-run C_identity == BH_fixed bit-exact is the "
            "operator-control result"
        ),
        "baseline_regime": "clip=1.0 (G1a frozen bank, read-only)",
        "meaning": "C_identity == BH_fixed attributes the BH_fixed-vs-C difference to the clipping Jacobian operator; NOT a no-clip training-regime claim",
    }
    _write_json(OPERATOR / "summary.json", summary)
    with (OPERATOR / "per_stream.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _manifest_touch("g3a_operator", "COMPLETE" if passed else "FAIL", summary)
    print(json_dumps(summary))
    if not passed:
        raise RuntimeError("G3A_OPERATOR_CONTROL_FAILED")
    print("G3A_OPERATOR_PASS")
    return 0


# ---------------------------------------------------------------------------
# train (no-clip)
# ---------------------------------------------------------------------------

def _reinit_model(device, train_seed):
    minimind = ROOT / "minimind"
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import init_model, setup_seed
    cfg = _read_json(CONFIG_PATH)["model"]
    setup_seed(train_seed)
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none",
        tokenizer_path=str(minimind / "model"), device=device,
    )
    model.train()
    return model, tokenizer


def cmd_train(args) -> int:
    device = args.device
    torch, model, tokenizer, get_lr, setup_seed = _minimind_setup(device)
    cfg = _read_json(CONFIG_PATH)
    if cfg["training"].get("grad_clip_enabled") is not False:
        raise RuntimeError("G3 train requires grad_clip_enabled=false in config")
    partition = load_partition(V07_DIR / "partition_v07.json")
    data_path = V07_DIR / "e3c_subset.jsonl"
    from dataset.lm_dataset import PretrainDataset
    ds_pre = PretrainDataset(
        str(data_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"])
    )
    autocast_ctx, scaler, device_type = make_amp(torch, device, cfg["training"]["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("G3 bf16 requires GradScaler disabled")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["learning_rate"])
    )
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    accum = int(cfg["training"]["accumulation_steps"])
    batch_size = int(cfg["training"]["batch_size"])
    base_lr = float(cfg["training"]["learning_rate"])
    total_opt = TOTAL_OPT_STEPS
    total_micro = total_opt * accum
    max_step = max(STAGE_STEPS.values())
    max_micro = max_step * accum
    clip_threshold_reference = 1.0  # counterfactual indicator only
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(TRAIN_SEEDS)

    param_counts = _count_params(model)
    if param_counts["unique_params"] != int(cfg["model"]["expected_unique_params"]):
        raise RuntimeError(
            f"param count drift: expected {cfg['model']['expected_unique_params']}, "
            f"got {param_counts['unique_params']}"
        )
    print(f"PARAM_COUNT unique={param_counts['unique_params']} state_dict_elements={param_counts['state_dict_elements']}")

    for train_seed in seeds:
        seed_dir = BANK / f"seed_{train_seed}"
        states_dir = seed_dir / "states"
        report_path = seed_dir / "state_bank_report.json"
        if report_path.exists() and not args.force:
            old = _read_json(report_path)
            if old.get("technical_pass") and int(old.get("train_seed", -1)) == train_seed:
                print(f"SEED_{train_seed}_BANK_SKIP_EXISTING_PASS")
                continue
        states_dir.mkdir(parents=True, exist_ok=True)
        setup_seed(train_seed)
        model, tokenizer = _reinit_model(device, train_seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        assert_native_adamw(optimizer)
        optimizer.zero_grad(set_to_none=True)
        named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

        train_batch_seed = train_seed + int(cfg["training"]["train_batch_seed_offset"])
        batches = batches_from_train_pool(
            partition["train_indices"], batch_size, train_batch_seed, max_micro
        )
        first_micro_batch = list(map(int, batches[0]))
        probe_set = set(partition["probe_indices"])
        eval_set = set(partition["eval_indices"])
        if set(first_micro_batch) & probe_set or set(first_micro_batch) & eval_set:
            raise RuntimeError("trajectory step0 micro-batch overlaps probe/eval")

        records = []
        dynamics = []
        loss_accum = 0.0
        used = set()
        clip_calls = 0
        for micro_idx, batch_idx in enumerate(batches, start=1):
            used.update(batch_idx)
            lr = set_native_micro_lr(optimizer, get_lr, micro_idx, total_micro, base_lr)
            x, y = stack_batch(ds_pre, batch_idx, device)
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            loss_accum += float(loss.detach().float().cpu())

            if micro_idx % accum == 0:
                scaler.unscale_(optimizer)
                # [FR-G3] genuine no-clip: clip_grad_norm_ is NEVER called.
                with torch.no_grad():
                    preclip = float(torch.sqrt(sum(
                        (p.grad.detach().float() ** 2).sum()
                        for p in model.parameters() if p.grad is not None
                    )).cpu())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step = micro_idx // accum
                with torch.no_grad():
                    pnorm = float(torch.sqrt(sum(
                        p.detach().float().pow(2).sum() for _, p in named
                    )).cpu())
                dynamics.append({
                    "optimizer_step": opt_step,
                    "global_micro_step": micro_idx,
                    "lr": float(lr),
                    "train_loss_avg": loss_accum / accum,
                    "preclip_grad_norm": preclip,
                    "clip_grad_norm_called": False,
                    "clip_active": False,
                    "counterfactual_would_clip_threshold_1": bool(preclip > clip_threshold_reference),
                    "param_norm": pnorm,
                })
                loss_accum = 0.0
                if opt_step in set(STAGE_STEPS.values()):
                    path = states_dir / f"state_step_{opt_step:03d}.pt"
                    stage = stage_of_step(opt_step)
                    st = save_full_training_state(
                        path, model, optimizer, scaler,
                        metadata={
                            "protocol": NOCLIP_STATE_PROTOCOL,
                            "train_seed": train_seed,
                            "state_optimizer_step": opt_step,
                            "state_stage": stage,
                            "state_fraction": STAGE_FRACTIONS[stage],
                            "cumulative_tokens": STAGE_TOKENS[stage],
                            "global_optimizer_step": opt_step,
                            "global_micro_step": micro_idx,
                            "planned_total_optimizer_steps": total_opt,
                            "total_micro_steps": total_micro,
                            "partition_sha256": partition["sha256"],
                            "train_batch_seed": train_batch_seed,
                            "grad_clip_enabled": False,
                            "clip_grad_norm_called_steps": clip_calls,
                            "optimizer_signature": optimizer_signature(optimizer),
                            "config_sha256": file_sha256(CONFIG_PATH),
                        },
                    )
                    records.append({
                        "step": opt_step,
                        "stage": stage,
                        "path": str(path),
                        "model_hash": stable_hash(st["model"]),
                        "optimizer_hash": stable_hash(st["optimizer"]),
                    })
                    print(f"SEED_{train_seed}_STATE_SAVED step={opt_step} stage={stage}")

        if set(used) & probe_set or set(used) & eval_set:
            raise RuntimeError(f"seed={train_seed} trajectory used probe/eval")
        report = {
            "protocol": NOCLIP_STATE_PROTOCOL,
            "technical_pass": len(records) == len(STAGES),
            "train_seed": train_seed,
            "train_batch_seed": train_batch_seed,
            "model_params": param_counts,
            "state_steps": [r["step"] for r in records],
            "states": records,
            "dynamics": dynamics,
            "grad_clip_enabled": False,
            "clip_grad_norm_called_total": clip_calls,
            "config_sha256": file_sha256(CONFIG_PATH),
        }
        _write_json(report_path, report)
        _write_json(seed_dir / "training_dynamics.json", {"dynamics": dynamics})
        print(f"SEED_{train_seed}_BANK_PASS states={len(records)} clip_calls={clip_calls}")
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    training_manifest = {
        "protocol": NOCLIP_STATE_PROTOCOL,
        "seeds": seeds,
        "stages": {s: {"step": STAGE_STEPS[s], "fraction": STAGE_FRACTIONS[s],
                       "cumulative_tokens": STAGE_TOKENS[s]} for s in STAGES},
        "total_planned_optimizer_steps": TOTAL_OPT_STEPS,
        "tokens_per_optimizer_step": 1024,
        "training_terminated_at_step": max(STAGE_STEPS.values()),
        "grad_clip_enabled": False,
        "clip_implementation": "clip_grad_norm_ never called; see experiment_config.json",
        "param_counts": param_counts,
        "seed_reports": [str(BANK / f"seed_{s}/state_bank_report.json") for s in seeds],
    }
    _write_json(OUT / "training_manifest.json", training_manifest)

    checkpoint_manifest = []
    for seed, stage, step, path in _state_paths():
        checkpoint_manifest.append({
            "seed": seed, "stage": stage, "step": step,
            "path": str(path.relative_to(OUT)),
            "sha256": file_sha256(path) if path.is_file() else "MISSING",
        })
    _write_json(OUT / "checkpoint_manifest.json", {"checkpoints": checkpoint_manifest})
    _manifest_touch("train", "COMPLETE", {"seeds": seeds, "param_counts": param_counts})
    print("TRAIN_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# theta (no-clip q + E1 + manifests + E2 FD)
# ---------------------------------------------------------------------------

def _branch_audit(branch):
    return {
        "first_preclip_grad_norm": branch["first_preclip_grad_norm"],
        "first_clipped": branch["first_clipped"],
        "last_lr": branch["last_lr"],
        "batch_manifest_hash": branch["batch_manifest_hash"],
        "model_hash_final": branch["model_hash_final"],
        "optimizer_hash_final": branch["optimizer_hash_final"],
        "X_theta": {str(h): float(obs["X_theta"]) for h, obs in branch["observations"].items()},
    }


def _rho0_noop_result(baseline, noop):
    return {
        "pass": bool(
            baseline["model_hash_final"] == noop["model_hash_final"]
            and baseline["optimizer_hash_final"] == noop["optimizer_hash_final"]
            and baseline["observations"] == noop["observations"]
            and baseline["batch_manifest_hash"] == noop["batch_manifest_hash"]
        ),
    }


def cmd_theta(args) -> int:
    device = args.device
    torch, model, tokenizer, get_lr, _setup_seed = _minimind_setup(device)
    cfg = _read_json(CONFIG_PATH)
    partition = load_partition(V07_DIR / "partition_v07.json")
    data_path = V07_DIR / "e3c_subset.jsonl"
    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, device, cfg["training"]["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    accum = int(cfg["training"]["accumulation_steps"])
    batch_size = int(cfg["training"]["batch_size"])
    base_lr = float(cfg["training"]["learning_rate"])
    branch_grad_clip = CLIP_DISABLED_MAX_NORM
    max_h = max(HORIZONS_OFFICIAL)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(TRAIN_SEEDS)

    state_ctx = {}
    for seed, stage, step, state_path in _state_paths():
        if seed not in seeds:
            continue
        if not state_path.is_file():
            raise FileNotFoundError(f"missing no-clip checkpoint {state_path}; run train first")
        seed_dir = THETA / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        meta = load_full_training_state(state_path, model, optimizer, scaler)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        reference_theta = payload["model"]

        optimizer.zero_grad(set_to_none=True)
        model.eval()
        xp, yp = stack_batch(ds, partition["probe_indices"], device)
        with autocast_ctx:
            rr = model(xp, labels=yp)
            probe_loss = rr.loss + rr.aux_loss
        params = [p for _, p in named]
        grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
        q_raw = {
            n: (torch.zeros_like(p.detach()) if g is None else g.detach().clone())
            for (n, p), g in zip(named, grads)
        }
        q, probe_grad_norm = normalize_direction(q_raw)
        q_norm = direction_norm(q)
        q_path = seed_dir / "q" / f"q_state_step_{step:03d}.pt"
        q_sha = save_direction(
            q_path, q,
            metadata={
                "protocol": Q_PROTOCOL,
                "train_seed": seed,
                "state_step": step,
                "state_stage": stage,
                "state_path": str(state_path),
                "partition_sha256": partition["sha256"],
                "probe_indices": partition["probe_indices"],
                "probe_loss": float(probe_loss.detach().float().cpu()),
                "probe_grad_norm_before_normalize": probe_grad_norm,
                "q_norm": q_norm,
                "config_sha256": file_sha256(CONFIG_PATH),
            },
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()

        s_values = []
        e1_records = []
        for r in range(E1_STREAMS):
            stream_seed = two_level_seed(E1_BASE_SEED, seed, step, r)
            batch_seed = batch_seed_from_stream_seed(stream_seed)
            batches = batches_from_train_pool(
                partition["train_indices"], batch_size, batch_seed, max_h * accum
            )
            manifest_path = seed_dir / "e1_stream_manifests" / f"state_step_{step:03d}" / f"e1_stream_{r:03d}_batches.json"
            manifest_sha = save_manifest(
                manifest_path, batches,
                metadata={
                    "protocol": "FR_G3_NOCLIP_E1_STREAM",
                    "train_seed": seed,
                    "state_step": step,
                    "stream_id": r,
                    "stream_seed": stream_seed,
                    "batch_seed": batch_seed,
                },
            )
            br = run_theta_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                reference_theta=reference_theta, q=q, named=named, batches=batches,
                stream_seed=stream_seed, sign=0, epsilon=0.0,
                horizons=HORIZONS_OFFICIAL, accumulation_steps=accum,
                batch_size=batch_size, device=device, grad_clip=branch_grad_clip,
                base_lr=base_lr, global_micro_step=int(meta["global_micro_step"]),
                total_micro_steps=int(meta["total_micro_steps"]), get_lr=get_lr,
                stack_batch=stack_batch, measure_s_first=True,
            )
            s_values.append(float(br["s"]))
            e1_records.append({
                "stream_id": r, "s_r": float(br["s"]), "stream_seed": stream_seed,
                "batch_seed": batch_seed,
                "batch_manifest_path": str(manifest_path),
                "batch_manifest_sha256": manifest_sha,
                "batch_manifest_hash": br["batch_manifest_hash"],
                "first_preclip_grad_norm": br["first_preclip_grad_norm"],
                "first_clipped": br["first_clipped"],
            })
        sigma_q = float(np.std(np.asarray(s_values, dtype=np.float64), ddof=1))
        if not math.isfinite(sigma_q) or sigma_q <= 0:
            raise RuntimeError(f"seed={seed} step={step} sigma_q invalid: {sigma_q}")
        epsilon = RHO * sigma_q

        e2_manifest_records = []
        for r in range(E2_STREAMS):
            stream_seed = two_level_seed(E2_BASE_SEED, seed, step, r)
            batch_seed = batch_seed_from_stream_seed(stream_seed)
            batches = batches_from_train_pool(
                partition["train_indices"], batch_size, batch_seed, max_h * accum
            )
            manifest_path = seed_dir / "e2_stream_manifests" / f"state_step_{step:03d}" / f"e2_stream_{r:03d}_batches.json"
            manifest_sha = save_manifest(
                manifest_path, batches,
                metadata={
                    "protocol": E2_MANIFEST_PROTOCOL,
                    "train_seed": seed,
                    "state_step": step,
                    "stream_id": r,
                    "stream_seed": stream_seed,
                    "batch_seed": batch_seed,
                    "rho": RHO,
                    "epsilon": epsilon,
                },
            )
            e2_manifest_records.append({
                "stream_id": r,
                "stream_seed": stream_seed,
                "batch_seed": batch_seed,
                "batch_manifest_path": str(manifest_path),
                "batch_manifest_sha256": manifest_sha,
            })
        state_ctx[(seed, step)] = {
            "meta": meta, "reference_theta": reference_theta,
            "q_path": str(q_path), "q_sha256": q_sha, "q_norm": q_norm,
            "probe_grad_norm": probe_grad_norm, "sigma_q": sigma_q,
            "epsilon": epsilon, "e1_records": e1_records,
            "e2_manifest_records": e2_manifest_records,
        }
        print(f"THETA_STAGE1 seed={seed} step={step} sigma_q={sigma_q:.6g} epsilon={epsilon:.6g}")

    future_stream_manifest = {
        "protocol": PROTOCOL,
        "ordering_rule": "all 36 manifests generated and written before any E2 response branch",
        "e2_base_seed": E2_BASE_SEED,
        "paired_with": "outputs/generalization_g1a_26m_v01 (identical stream seeds)",
        "streams": [
            {
                "train_seed": seed, "state_step": step, "stage": stage_of_step(step),
                "epsilon": state_ctx[(seed, step)]["epsilon"],
                "rho": RHO,
                "manifests": state_ctx[(seed, step)]["e2_manifest_records"],
            }
            for seed, stage, step, _ in _state_paths() if seed in seeds
        ],
    }
    _write_json(OUT / "future_stream_manifest.json", future_stream_manifest)

    for seed in seeds:
        seed_dir = THETA / f"seed_{seed}"
        result_path = seed_dir / "theta_mapping_raw.json"
        state_results = []
        rho0_noop_all_pass = True
        for _seed, stage, step, state_path in _state_paths():
            if _seed != seed:
                continue
            ctx = state_ctx[(seed, step)]
            meta = ctx["meta"]
            reference_theta = ctx["reference_theta"]
            q, q_meta, q_sha = load_direction(ctx["q_path"])
            if q_sha != ctx["q_sha256"]:
                raise RuntimeError("q sha drift between stage1 and stage2")
            e2_rows = []
            e2_stream_records = []
            for rec in ctx["e2_manifest_records"]:
                stream_id = int(rec["stream_id"])
                manifest_path = Path(rec["batch_manifest_path"])
                manifest = validate_g3_e2_manifest(
                    _read_json(manifest_path),
                    expected_seed=seed, expected_step=step,
                    expected_stream=stream_id,
                    expected_content_sha=rec["batch_manifest_sha256"],
                    accumulation_steps=accum,
                )
                if len(manifest["batches"]) < max_h * accum:
                    raise ValueError("manifest does not cover the H=8 window")
                if int(manifest["metadata"]["stream_seed"]) != int(rec["stream_seed"]):
                    raise ValueError("manifest/record stream seed mismatch")
                if not math.isclose(
                    float(manifest["metadata"]["epsilon"]), float(ctx["epsilon"]), rel_tol=0, abs_tol=0
                ):
                    raise ValueError("manifest epsilon mismatch")
                common = dict(
                    torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                    autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                    reference_theta=reference_theta, q=q, named=named,
                    batches=manifest["batches"], stream_seed=int(rec["stream_seed"]),
                    horizons=HORIZONS_OFFICIAL, accumulation_steps=accum,
                    batch_size=batch_size, device=device, grad_clip=branch_grad_clip,
                    base_lr=base_lr,
                    global_micro_step=int(meta["global_micro_step"]),
                    total_micro_steps=int(meta["total_micro_steps"]),
                    get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
                )
                b0 = run_theta_branch(sign=0, epsilon=0.0, **common)
                bp = run_theta_branch(sign=+1, epsilon=float(ctx["epsilon"]), **common)
                bm = run_theta_branch(sign=-1, epsilon=float(ctx["epsilon"]), **common)
                bz = run_theta_branch(sign=+1, epsilon=0.0, **common)
                if not (b0["batch_manifest_hash"] == bp["batch_manifest_hash"] == bm["batch_manifest_hash"]):
                    raise RuntimeError(f"paired batch manifest mismatch seed={seed} step={step} stream={stream_id}")
                if not (b0["last_lr"] == bp["last_lr"] == bm["last_lr"]):
                    raise RuntimeError(f"paired LR path mismatch seed={seed} step={step} stream={stream_id}")
                noop = _rho0_noop_result(b0, bz)
                if not noop["pass"]:
                    rho0_noop_all_pass = False
                    raise RuntimeError(f"rho0 no-op failure seed={seed} step={step} stream={stream_id}")
                if b0["first_clipped"] or bp["first_clipped"] or bm["first_clipped"]:
                    raise RuntimeError(f"unexpected clipping active in no-clip branch seed={seed} step={step} stream={stream_id}")
                e2_rows.append({"baseline": b0, "plus": bp, "minus": bm})
                e2_stream_records.append({
                    "stream_id": stream_id,
                    "stream_seed": int(rec["stream_seed"]),
                    "batch_seed": int(rec["batch_seed"]),
                    "batch_manifest_path": str(manifest_path),
                    "batch_manifest_sha256": rec["batch_manifest_sha256"],
                    "batch_manifest_hash": b0["batch_manifest_hash"],
                    "paired_batch_manifest_hash_equal": True,
                    "paired_lr_path_equal": True,
                    "baseline": _branch_audit(b0),
                    "plus": _branch_audit(bp),
                    "minus": _branch_audit(bm),
                    "rho0_noop": noop,
                })
                print(f"E2 seed={seed} step={step} stream={stream_id} DONE", flush=True)

            chi = {}
            odd = {}
            for h in HORIZONS_OFFICIAL:
                hk = str(h)
                sm = paired_linearity_summary(
                    [x["baseline"]["observations"][hk]["X_theta"] for x in e2_rows],
                    [x["plus"]["observations"][hk]["X_theta"] for x in e2_rows],
                    [x["minus"]["observations"][hk]["X_theta"] for x in e2_rows],
                    float(ctx["epsilon"]),
                )
                chi[hk] = sm["chi"]
                odd[hk] = sm["odd_error"]
            clip = {
                "baseline_rate": sum(x["baseline"]["first_clipped"] for x in e2_rows) / E2_STREAMS,
                "plus_rate": sum(x["plus"]["first_clipped"] for x in e2_rows) / E2_STREAMS,
                "minus_rate": sum(x["minus"]["first_clipped"] for x in e2_rows) / E2_STREAMS,
            }
            state_results.append({
                "train_seed": seed,
                "step": step,
                "stage": stage,
                "state_fraction": STAGE_FRACTIONS[stage],
                "q_norm": ctx["q_norm"],
                "q_path": ctx["q_path"],
                "q_sha256": ctx["q_sha256"],
                "probe_grad_norm": ctx["probe_grad_norm"],
                "sigma_q": ctx["sigma_q"],
                "epsilon": ctx["epsilon"],
                "rho": RHO,
                "chi_theta": chi,
                "odd_error_theta": odd,
                "clipping": clip,
                "e1_streams": ctx["e1_records"],
                "e2_streams": e2_stream_records,
            })
            print(f"THETA_STAGE2 seed={seed} step={step} chi={chi}")

        result = {
            "protocol": "FR_G3_NOCLIP_THETA_MAPPING_V01",
            "technical_pass": True,
            "rho0_noop_pass": rho0_noop_all_pass,
            "train_seed": seed,
            "primary_rho": RHO,
            "horizons": list(HORIZONS_OFFICIAL),
            "grad_clip_sentinel": CLIP_DISABLED_MAX_NORM,
            "config_sha256": file_sha256(CONFIG_PATH),
            "states": state_results,
        }
        _write_json(result_path, result)
        print(f"SEED_{seed}_THETA_MAPPING_PASS")

    _manifest_touch("theta", "COMPLETE", {
        "states": len(seeds) * len(STAGES),
        "e2_streams": len(seeds) * len(STAGES) * E2_STREAMS,
    })
    print("THETA_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# smoke (no-clip gate)
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    device = args.device
    seed, step = TRAIN_SEEDS[0], STAGE_STEPS["EARLY"]
    state_path = BANK / f"seed_{seed}/states/state_step_{step:03d}.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path}; run train first")
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import("a0e2_curvature_decomposition_v02_bh_fixed.py", "g3_frozen_a0e2_runner")
    accum = int(cfg["training"]["accumulation_steps"])
    branch_grad_clip = CLIP_DISABLED_MAX_NORM
    report = {"protocol": PROTOCOL, "state": {"seed": seed, "step": step}, "checks": {}}

    meta = load_full_training_state(state_path, model, optimizer, scaler)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    m_v_ok = True
    for pname, param in named:
        st = optimizer.state.get(param)
        if (
            st is None
            or st["exp_avg"].shape != param.shape
            or st["exp_avg_sq"].shape != param.shape
            or int(st["step"]) < 1
        ):
            m_v_ok = False
    report["checks"]["optimizer_state_dims_names"] = bool(m_v_ok)
    report["checks"]["checkpoint_restore"] = True

    theta = _read_json(THETA / f"seed_{seed}/theta_mapping_raw.json")
    theta_state = next((r for r in theta["states"] if int(r["step"]) == step), None)
    if theta_state is None or len(theta_state.get("e2_streams", [])) != E2_STREAMS:
        raise ValueError("missing no-clip theta state/streams")
    q_by_name, _qm, q_sha = load_direction(theta_state["q_path"])
    q_ok = set(q_by_name) == {n for n, _ in named} and all(
        tuple(q_by_name[n].shape) == tuple(p.shape) for n, p in named
    )
    report["checks"]["q_names_shapes"] = bool(q_ok and q_sha == theta_state["q_sha256"])

    rec = sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"]))[0]
    manifest = validate_g3_e2_manifest(
        _read_json(Path(rec["batch_manifest_path"])),
        expected_seed=seed, expected_step=step, expected_stream=int(rec["stream_id"]),
        expected_content_sha=rec["batch_manifest_sha256"], accumulation_steps=accum,
    )
    common = dict(
        checkpoint_path=state_path, reference_theta=payload["model"],
        q_by_name=q_by_name, batches=manifest["batches"],
        stream_seed=int(rec["stream_seed"]), accumulation_steps=accum,
        device=device, grad_clip=branch_grad_clip,
        base_lr=float(cfg["training"]["learning_rate"]),
        global_micro_step=int(meta["global_micro_step"]),
        total_micro_steps=int(meta["total_micro_steps"]), get_lr=get_lr,
    )

    def branch(sign: int, epsilon: float):
        return run_theta_branch(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
            reference_theta=common["reference_theta"],
            q=common["q_by_name"], named=named,
            batches=common["batches"], stream_seed=common["stream_seed"],
            sign=sign, epsilon=epsilon, horizons=HORIZONS_OFFICIAL,
            accumulation_steps=common["accumulation_steps"],
            batch_size=int(cfg["training"]["batch_size"]), device=device,
            grad_clip=common["grad_clip"], base_lr=common["base_lr"],
            global_micro_step=common["global_micro_step"],
            total_micro_steps=common["total_micro_steps"],
            get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
        )

    b0 = branch(sign=0, epsilon=0.0)
    bp = branch(sign=+1, epsilon=float(theta_state["epsilon"]))
    bm = branch(sign=-1, epsilon=float(theta_state["epsilon"]))
    bz = branch(sign=+1, epsilon=0.0)
    crn_ok = (
        b0["batch_manifest_hash"] == bp["batch_manifest_hash"] == bm["batch_manifest_hash"]
        and b0["last_lr"] == bp["last_lr"] == bm["last_lr"]
    )
    noop_ok = _rho0_noop_result(b0, bz)["pass"]
    finite_ok = all(
        math.isfinite(float(obs["X_theta"]))
        for br in (b0, bp, bm)
        for obs in br["observations"].values()
    )
    report["checks"]["fd_crn_consistent"] = bool(crn_ok)
    report["checks"]["rho0_noop"] = bool(noop_ok)
    report["checks"]["fd_outputs_finite"] = bool(finite_ok)
    report["checks"]["no_clip_never_active_branches"] = bool(
        not (b0["first_clipped"] or bp["first_clipped"] or bm["first_clipped"])
    )

    propagated = a0e2_runner._run_baseline_tangents(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, dataset=ds, named=named, **common,
    )
    chi_fd_h1 = float(
        (rec["plus"]["X_theta"]["1"] - rec["minus"]["X_theta"]["1"])
        / (2.0 * float(theta_state["epsilon"]))
    )
    h1_rel_err = abs(propagated["chi_optimizer_only"]["1"] - chi_fd_h1) / max(abs(chi_fd_h1), 1e-30)
    report["checks"]["h1_analytic_vs_fd_rel_err"] = float(h1_rel_err)
    report["checks"]["h1_consistency"] = bool(h1_rel_err < 0.3)
    h1_b = float(propagated["chi_optimizer_only"]["1"])
    h1_bhf = float(propagated["chi_BH_fixed_curvature"]["1"])
    h1_c = float(propagated["chi_full_tangent"]["1"])
    report["checks"]["h1_B_equals_BH_fixed_equals_C"] = bool(h1_b == h1_bhf == h1_c)
    bhc_gaps = {
        str(h): abs(float(propagated["chi_BH_fixed_curvature"][str(h)]) - float(propagated["chi_full_tangent"][str(h)]))
        for h in HORIZONS_OFFICIAL
    }
    report["checks"]["noclip_BH_fixed_equals_C_max_abs_gap"] = float(max(bhc_gaps.values()))
    report["checks"]["noclip_BH_fixed_equals_C"] = bool(max(bhc_gaps.values()) == 0.0)
    trace_clip_active = [bool(row["clip_active"]) for row in propagated["step_trace"]]
    report["checks"]["noclip_no_clip_active_in_trace"] = bool(not any(trace_clip_active))
    shadow = branch(sign=0, epsilon=0.0)
    report["checks"]["shadow_hash_match"] = bool(
        str(propagated["model_hash_final"]) == str(shadow["model_hash_final"])
        and str(propagated["optimizer_hash_final"]) == str(shadow["optimizer_hash_final"])
    )
    hvp_flags = [row["hvp_evaluated"] for row in propagated["step_trace"]]
    hvp_norms = [row["hvp_norm"] for row in propagated["step_trace"] if row["hvp_evaluated"]]
    report["checks"]["hvp_executed"] = bool(any(hvp_flags))
    report["checks"]["hvp_finite"] = bool(all(math.isfinite(v) for v in hvp_norms))
    report["checks"]["no_unexpected_optimizer_steps"] = bool(
        propagated["step_trace"][-1]["optimizer_step_after"]
        == float(meta["global_optimizer_step"] + max(HORIZONS_OFFICIAL))
    )
    bank_report = _read_json(BANK / f"seed_{seed}/state_bank_report.json")
    all_uncalled = all(
        d["clip_grad_norm_called"] is False and d["clip_active"] is False
        for d in bank_report["dynamics"]
    )
    report["checks"]["training_never_clipped"] = bool(all_uncalled)
    _assert_finite_tree(
        {k: v for k, v in propagated.items() if k != "step_trace"}, "smoke"
    )
    report["checks"]["all_outputs_finite"] = True

    checks = report["checks"]
    report["overall_pass"] = bool(
        checks["optimizer_state_dims_names"] and checks["q_names_shapes"]
        and checks["fd_crn_consistent"] and checks["rho0_noop"]
        and checks["fd_outputs_finite"] and checks["no_clip_never_active_branches"]
        and checks["h1_consistency"] and checks["h1_B_equals_BH_fixed_equals_C"]
        and checks["noclip_BH_fixed_equals_C"] and checks["noclip_no_clip_active_in_trace"]
        and checks["shadow_hash_match"] and checks["hvp_executed"] and checks["hvp_finite"]
        and checks["no_unexpected_optimizer_steps"] and checks["training_never_clipped"]
        and checks["all_outputs_finite"]
    )
    _write_json(OUT / "smoke_report.json", report)
    print(json_dumps(report))
    if not report["overall_pass"]:
        raise RuntimeError("SMOKE_GATE_FAILED")
    _manifest_touch("smoke", "PASS", {"seed": seed, "step": step})
    print("SMOKE_PASS")
    return 0


# ---------------------------------------------------------------------------
# clip reference: BH_fixed for frozen G1a seeds 2043/2044 (mechanism subset
# only covered 2042).  Frozen G1a bank read-only; outputs land in G3 dir only.
# ---------------------------------------------------------------------------

def cmd_clipref(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import("a0e2_curvature_decomposition_v02_bh_fixed.py", "g3_frozen_a0e2_runner")
    accum = int(cfg["training"]["accumulation_steps"])
    grad_clip = 1.0
    CLIPREF.mkdir(parents=True, exist_ok=True)
    g1a_cfg = _read_json(ROOT / "research/configs/g1a_26m_v01.json")

    rows = []
    cross_check_max = 0.0
    for seed, stage, step, state_path in _g1a_state_paths():
        if seed == TRAIN_SEEDS[0]:
            continue  # seed 2042: frozen G1a mechanism JSONs used directly
        theta_state = _load_g1a_theta_state(seed, step)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        q_by_name, _qm, q_sha = load_direction(theta_state["q_path"])
        if q_sha != theta_state["q_sha256"]:
            raise ValueError("q identity mismatch vs frozen G1a record")
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            stream_id = int(rec["stream_id"])
            manifest = validate_g1a_e2_manifest(
                _read_json(Path(rec["batch_manifest_path"])),
                expected_seed=seed, expected_step=step, expected_stream=stream_id,
                expected_content_sha=rec["batch_manifest_sha256"], accumulation_steps=accum,
            )
            meta = load_full_training_state(state_path, model, optimizer, scaler, restore_rng=False)
            propagated = a0e2_runner._run_baseline_tangents(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named,
                checkpoint_path=state_path, reference_theta=payload["model"],
                q_by_name=q_by_name, batches=manifest["batches"],
                stream_seed=int(rec["stream_seed"]), accumulation_steps=accum,
                device=device, grad_clip=grad_clip,
                base_lr=float(g1a_cfg["training"]["learning_rate"]),
                global_micro_step=int(meta["global_micro_step"]),
                total_micro_steps=int(meta["total_micro_steps"]), get_lr=get_lr,
            )
            for h in HORIZONS_OFFICIAL:
                hk = str(h)
                rows.append({
                    "seed": seed, "stage": stage, "step": step, "stream_id": stream_id, "H": h,
                    "chi_B": float(propagated["chi_optimizer_only"][hk]),
                    "chi_BH_fixed": float(propagated["chi_BH_fixed_curvature"][hk]),
                    "chi_C": float(propagated["chi_full_tangent"][hk]),
                })
            print(f"CLIPREF seed={seed} step={step} stream={stream_id} DONE", flush=True)
        torch.cuda.empty_cache()

    # integrity cross-check: recompute one frozen seed-2042 stream and compare
    seed, stage, step, state_path = next(iter(list(_g1a_state_paths())[:1]))
    theta_state = _load_g1a_theta_state(seed, step)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    q_by_name, _qm, q_sha = load_direction(theta_state["q_path"])
    rec = sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"]))[0]
    manifest = validate_g1a_e2_manifest(
        _read_json(Path(rec["batch_manifest_path"])),
        expected_seed=seed, expected_step=step, expected_stream=int(rec["stream_id"]),
        expected_content_sha=rec["batch_manifest_sha256"], accumulation_steps=accum,
    )
    meta = load_full_training_state(state_path, model, optimizer, scaler, restore_rng=False)
    propagated = a0e2_runner._run_baseline_tangents(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, dataset=ds, named=named,
        checkpoint_path=state_path, reference_theta=payload["model"],
        q_by_name=q_by_name, batches=manifest["batches"],
        stream_seed=int(rec["stream_seed"]), accumulation_steps=accum,
        device=device, grad_clip=grad_clip,
        base_lr=float(g1a_cfg["training"]["learning_rate"]),
        global_micro_step=int(meta["global_micro_step"]),
        total_micro_steps=int(meta["total_micro_steps"]), get_lr=get_lr,
    )
    frozen_mech = _read_json(G1A_OUT / "mechanism" / f"stream_seed{seed}_step{step:03d}_s{int(rec['stream_id'])}.json")
    for h in HORIZONS_OFFICIAL:
        cross_check_max = max(
            cross_check_max,
            abs(float(propagated["chi_BH_fixed_curvature"][str(h)]) - float(frozen_mech["chi_BH_fixed"][str(h)])),
            abs(float(propagated["chi_full_tangent"][str(h)]) - float(frozen_mech["chi_C"][str(h)])),
        )

    _write_json(CLIPREF / "summary.json", {
        "protocol": PROTOCOL,
        "subphase": "clip_reference_bh_fixed_seeds_2043_2044",
        "note": "computed from frozen read-only G1a states to complete the paired interaction matrix; frozen G1a outputs untouched; seed 2042 from frozen mechanism JSONs",
        "cross_check_seed2042_max_abs_gap_vs_frozen": float(cross_check_max),
        "n_rows": len(rows),
    })
    with (CLIPREF / "per_stream_bh_fixed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _manifest_touch("clipref", "COMPLETE", {"rows": len(rows), "cross_check_max_gap": float(cross_check_max)})
    print("CLIPREF_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# tangents (no-clip B / BH_fixed / C on all states)
# ---------------------------------------------------------------------------

def _load_noclip_theta_state(seed: int, step: int):
    theta = _read_json(THETA / f"seed_{seed}/theta_mapping_raw.json")
    if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
        raise ValueError(f"invalid no-clip theta record for seed {seed}")
    state = next((r for r in theta["states"] if int(r["step"]) == step), None)
    if state is None or len(state.get("e2_streams", [])) != E2_STREAMS:
        raise ValueError(f"missing no-clip state/streams: seed={seed} step={step}")
    return state


def cmd_tangents(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import("a0e2_curvature_decomposition_v02_bh_fixed.py", "g3_frozen_a0e2_runner")
    accum = int(cfg["training"]["accumulation_steps"])
    branch_grad_clip = CLIP_DISABLED_MAX_NORM
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(TRAIN_SEEDS)
    NOCLIP.mkdir(parents=True, exist_ok=True)

    replay_passes = []
    clip_gaps = []
    hvp_gaps = []
    per_stream_rows = []
    stream_records = []

    for seed, stage, step, state_path in _state_paths():
        if seed not in seeds:
            continue
        theta_state = _load_noclip_theta_state(seed, step)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            stream_id = int(rec["stream_id"])
            manifest = validate_g3_e2_manifest(
                _read_json(Path(rec["batch_manifest_path"])),
                expected_seed=seed, expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=rec["batch_manifest_sha256"],
                accumulation_steps=accum,
            )
            common = dict(
                checkpoint_path=state_path, reference_theta=payload["model"],
                q_by_name=None, batches=manifest["batches"],
                stream_seed=int(rec["stream_seed"]), accumulation_steps=accum,
                device=device, grad_clip=branch_grad_clip,
                base_lr=float(cfg["training"]["learning_rate"]),
                global_micro_step=None,
                total_micro_steps=None, get_lr=get_lr,
            )
            meta = load_full_training_state(state_path, model, optimizer, scaler, restore_rng=False)
            q_by_name, _qm, q_sha = load_direction(theta_state["q_path"])
            if q_sha != theta_state["q_sha256"]:
                raise ValueError("q identity mismatch")
            common["q_by_name"] = q_by_name
            common["global_micro_step"] = int(meta["global_micro_step"])
            common["total_micro_steps"] = int(meta["total_micro_steps"])

            propagated = a0e2_runner._run_baseline_tangents(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named,
                **{k: v for k, v in common.items() if k != "get_lr"},
                get_lr=get_lr,
            )
            shadow = run_theta_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds,
                state_path=common["checkpoint_path"],
                reference_theta=common["reference_theta"],
                q=common["q_by_name"], named=named,
                batches=common["batches"], stream_seed=common["stream_seed"],
                sign=0, epsilon=0.0, horizons=HORIZONS_OFFICIAL,
                accumulation_steps=accum,
                batch_size=int(cfg["training"]["batch_size"]), device=device,
                grad_clip=branch_grad_clip,
                base_lr=float(cfg["training"]["learning_rate"]),
                global_micro_step=common["global_micro_step"],
                total_micro_steps=common["total_micro_steps"],
                get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
            )
            gaps = {
                str(h): abs(
                    float(propagated["baseline_X_theta"][str(h)])
                    - float(shadow["observations"][str(h)]["X_theta"])
                )
                for h in HORIZONS_OFFICIAL
            }
            model_equal = str(propagated["model_hash_final"]) == str(shadow["model_hash_final"])
            optimizer_equal = str(propagated["optimizer_hash_final"]) == str(shadow["optimizer_hash_final"])
            replay_pass = bool(model_equal and optimizer_equal and max(gaps.values()) == 0.0)
            replay_passes.append(replay_pass)
            clip_gaps.append(propagated["max_native_clip_tensor_l2_gap"])
            hvp_gaps.extend(
                row["hvp_gradient_relative_gap_vs_native"]
                for row in propagated["step_trace"] if row["hvp_evaluated"]
            )
            trace_clip = [bool(row["clip_active"]) for row in propagated["step_trace"]]
            if any(trace_clip):
                raise RuntimeError(f"clip active in no-clip tangent seed={seed} step={step} stream={stream_id}")
            bhc_gap = max(
                abs(float(propagated["chi_BH_fixed_curvature"][str(h)]) - float(propagated["chi_full_tangent"][str(h)]))
                for h in HORIZONS_OFFICIAL
            )
            if bhc_gap != 0.0:
                raise RuntimeError(f"no-clip BH_fixed != C seed={seed} step={step} stream={stream_id} gap={bhc_gap}")
            stream_result = {
                "seed": seed, "stage": stage, "step": step, "stream_id": stream_id,
                "stream_seed": int(rec["stream_seed"]),
                "epsilon": float(theta_state["epsilon"]),
                "chi_fd": {
                    str(h): float(
                        (rec["plus"]["X_theta"][str(h)] - rec["minus"]["X_theta"][str(h)])
                        / (2.0 * float(theta_state["epsilon"]))
                    )
                    for h in HORIZONS_OFFICIAL
                },
                "chi_B": {k: float(v) for k, v in propagated["chi_optimizer_only"].items()},
                "chi_BH_fixed": {k: float(v) for k, v in propagated["chi_BH_fixed_curvature"].items()},
                "chi_C": {k: float(v) for k, v in propagated["chi_full_tangent"].items()},
                "baseline_X_theta": {k: float(v) for k, v in propagated["baseline_X_theta"].items()},
                "step_trace": propagated["step_trace"],
                "replay": {
                    "pass": replay_pass,
                    "model_hash_equal": model_equal,
                    "optimizer_hash_equal": optimizer_equal,
                    "X_abs_gap_vs_shadow": gaps,
                },
                "model_hash_final": str(propagated["model_hash_final"]),
                "optimizer_hash_final": str(propagated["optimizer_hash_final"]),
            }
            stream_records.append(stream_result)
            _write_json(
                NOCLIP / "streams" / f"stream_seed{seed}_step{step:03d}_s{stream_id}.json",
                stream_result,
            )
            for h in HORIZONS_OFFICIAL:
                per_stream_rows.append({
                    "seed": seed, "stage": stage, "step": step,
                    "stream_id": stream_id, "H": h,
                    "chi_fd": stream_result["chi_fd"][str(h)],
                    "chi_B": stream_result["chi_B"][str(h)],
                    "chi_BH_fixed": stream_result["chi_BH_fixed"][str(h)],
                    "chi_C": stream_result["chi_C"][str(h)],
                })
            print(f"TAN seed={seed} step={step} stream={stream_id} DONE", flush=True)
        del payload
        torch.cuda.empty_cache()

    # official per-state-horizon rows
    grouped = {}
    for row in per_stream_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        grouped.setdefault(key, {"chi_fd": [], "chi_B": [], "chi_BH_fixed": [], "chi_C": []})
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            grouped[key][m].append(float(row[m]))
    official_rows = []
    for (seed, step, h), vals in sorted(grouped.items()):
        entry = {"seed": seed, "stage": stage_of_step(step), "step": step, "H": h,
                 "n_streams": len(vals["chi_fd"])}
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            entry[m] = float(np.mean(vals[m]))
            entry[f"{m}_stream_std"] = float(np.std(vals[m], ddof=1))
        official_rows.append(entry)
    errors = cell_relative_errors_bhc(official_rows)
    for row in official_rows:
        row.update(errors[(int(row["seed"]), int(row["step"]), int(row["H"]))])

    _write_json(NOCLIP / "summary.json", {
        "protocol": PROTOCOL,
        "subphase": "no_clip_formal_matrix",
        "replay_pass_all": all(replay_passes),
        "max_native_clip_tensor_l2_gap": float(max(clip_gaps)),
        "max_hvp_math_gradient_relative_gap_vs_native": float(max(hvp_gaps)),
        "bh_fixed_equals_C_all_streams": True,
        "stream_count": len(stream_records),
    })
    with (NOCLIP / "per_stream.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_stream_rows[0]))
        writer.writeheader()
        writer.writerows(per_stream_rows)
    with (NOCLIP / "per_state_horizon.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(official_rows[0]))
        writer.writeheader()
        writer.writerows(official_rows)
    with (NOCLIP / "mechanism_decomposition.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_stream_rows[0]))
        writer.writeheader()
        writer.writerows(per_stream_rows)
    _manifest_touch("tangents", "COMPLETE", {
        "streams": len(stream_records),
        "replay_pass_all": all(replay_passes),
        "bh_fixed_equals_C": True,
    })
    print("TANGENTS_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _rel(observed: float, pred: float) -> float:
    from fr.a0c_adamw_response_propagation_v01 import relative_error
    r = relative_error(observed, pred)
    return float(r) if r is not None else math.nan


def cmd_analyze(_args) -> int:
    cfg = _read_json(CONFIG_PATH)
    g3a = _read_json(OPERATOR / "summary.json")
    noclip_session = _read_json(NOCLIP / "summary.json")
    smoke = _read_json(OUT / "smoke_report.json")

    # ---------- no-clip official stats --------------------------------------
    mech_rows = list(csv.DictReader((NOCLIP / "mechanism_decomposition.csv").open("r", encoding="utf-8")))
    official = []
    grouped = {}
    for row in mech_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        grouped.setdefault(key, {"chi_fd": [], "chi_B": [], "chi_BH_fixed": [], "chi_C": []})
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            grouped[key][m].append(float(row[m]))
    for (seed, step, h), vals in sorted(grouped.items()):
        entry = {"seed": seed, "stage": stage_of_step(step), "step": step, "H": h,
                 "n_streams": len(vals["chi_fd"])}
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            entry[m] = float(np.mean(vals[m]))
        official.append(entry)
    noclip_stats = aggregate_noclip_metrics(official)

    # ---------- clip=1.0 reference (frozen G1a) ------------------------------
    g1a_per_stream = _read_json(G1A_OUT / "results/per_stream.json")["rows"]
    clip_stream_cells = {}  # (seed, step, stream, H) -> dict
    for row in g1a_per_stream:
        key = (int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))
        clip_stream_cells[key] = {
            "chi_fd": float(row["chi_fd"]), "chi_B": float(row["chi_B"]), "chi_C": float(row["chi_C"]),
        }
    # BH_fixed for clip condition: seed 2042 from frozen mechanism JSONs,
    # seeds 2043/2044 from this experiment's clipref computations
    for mech_path in sorted((G1A_OUT / "mechanism").glob("stream_seed*_step*_s*.json")):
        rec = _read_json(mech_path)
        seed = int(rec["seed"]); step = int(rec["step"]); sid = int(rec["stream_id"])
        for h in HORIZONS_OFFICIAL:
            key = (seed, step, sid, h)
            if key in clip_stream_cells:
                clip_stream_cells[key]["chi_BH_fixed"] = float(rec["chi_BH_fixed"][str(h)])
    for row in csv.DictReader((CLIPREF / "per_stream_bh_fixed.csv").open("r", encoding="utf-8")):
        key = (int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))
        if key in clip_stream_cells:
            clip_stream_cells[key]["chi_BH_fixed"] = float(row["chi_BH_fixed"])
    missing = [k for k, v in clip_stream_cells.items() if "chi_BH_fixed" not in v]
    if missing:
        raise RuntimeError(f"clip reference missing BH_fixed for {len(missing)} stream-cells")

    # clip-condition official cells
    clip_grouped = {}
    for (seed, step, sid, h), vals in clip_stream_cells.items():
        key = (seed, step, h)
        clip_grouped.setdefault(key, {"chi_fd": [], "chi_B": [], "chi_BH_fixed": [], "chi_C": []})
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            clip_grouped[key][m].append(vals[m])
    clip_official = []
    for (seed, step, h), vals in sorted(clip_grouped.items()):
        entry = {"seed": seed, "stage": stage_of_step(step), "step": step, "H": h}
        for m in ("chi_fd", "chi_B", "chi_BH_fixed", "chi_C"):
            entry[m] = float(np.mean(vals[m]))
        clip_official.append(entry)
    clip_errors = cell_relative_errors_bhc(clip_official)
    clip_paired = []
    for row in clip_official:
        if int(row["H"]) not in H_GT_1:
            continue
        e = clip_errors[(int(row["seed"]), int(row["step"]), int(row["H"]))]
        clip_paired.append({
            "seed": int(row["seed"]), "stage": row["stage"], "step": int(row["step"]), "H": int(row["H"]),
            "gain_curvature": e["rel_err_B"] - e["rel_err_BH_fixed"],
            "gain_clip": e["rel_err_BH_fixed"] - e["rel_err_C"],
        })
    clip_gains = {
        "gain_curvature_mean": float(np.mean([p["gain_curvature"] for p in clip_paired])),
        "gain_clip_mean": float(np.mean([p["gain_clip"] for p in clip_paired])),
        "paired_rows": clip_paired,
    }

    # no-clip paired gains (official cells)
    noclip_paired = noclip_stats["paired_h_gt_1"]["rows"]
    noclip_gains = noclip_stats["pooled_gains_h_gt_1"]

    # ---------- interaction (official-cell level) ----------------------------
    noclip_cell_gain = {(int(r["seed"]), int(r["step"]), int(r["H"])): r for r in noclip_paired}
    paired_interaction = []
    for cp in clip_paired:
        key = (int(cp["seed"]), int(cp["step"]), int(cp["H"]))
        nc = noclip_cell_gain.get(key)
        if nc is None:
            continue
        paired_interaction.append({
            "seed": key[0], "stage": cp["stage"], "step": key[1], "H": key[2],
            "gain_clip_clip": float(cp["gain_clip"]),
            "gain_clip_noclip": float(nc["gain_clip"]),
            "delta_clip": float(cp["gain_clip"]) - float(nc["gain_clip"]),
            "gain_curvature_clip": float(cp["gain_curvature"]),
            "gain_curvature_noclip": float(nc["gain_curvature"]),
        })
    mean_delta = float(np.mean([p["delta_clip"] for p in paired_interaction]))
    by_group = {}
    for label, keyfn in (
        ("H4", lambda p: p["H"] == 4), ("H8", lambda p: p["H"] == 8),
        ("seed_2042", lambda p: p["seed"] == 2042),
        ("seed_2043", lambda p: p["seed"] == 2043),
        ("seed_2044", lambda p: p["seed"] == 2044),
        ("EARLY", lambda p: p["stage"] == "EARLY"),
        ("MIDDLE", lambda p: p["stage"] == "MIDDLE"),
        ("LATE", lambda p: p["stage"] == "LATE"),
    ):
        vals = [p["delta_clip"] for p in paired_interaction if keyfn(p)]
        by_group[label] = float(np.mean(vals)) if vals else math.nan

    stream_paired = []
    noclip_stream = {}
    for row in mech_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))
        noclip_stream[key] = row
    for (seed, step, sid, h), cv in clip_stream_cells.items():
        nv = noclip_stream.get((seed, step, sid, h))
        if nv is None or h not in H_GT_1:
            continue
        g_clip = _rel(cv["chi_fd"], cv["chi_BH_fixed"]) - _rel(cv["chi_fd"], cv["chi_C"])
        g_nc = _rel(float(nv["chi_fd"]), float(nv["chi_BH_fixed"])) - _rel(float(nv["chi_fd"]), float(nv["chi_C"]))
        stream_paired.append({
            "seed": seed, "stage": stage_of_step(step), "step": step,
            "stream_id": sid, "H": h,
            "gain_clip_clip": g_clip, "gain_clip_noclip": g_nc,
            "delta_clip": g_clip - g_nc,
        })
    boot = cluster_bootstrap_interaction(stream_paired)

    # ---------- dynamics context --------------------------------------------
    bank_first = _read_json(BANK / f"seed_{TRAIN_SEEDS[0]}/state_bank_report.json")
    g1a_bank_first = _read_json(G1A_OUT / "bank" / f"seed_{TRAIN_SEEDS[0]}/state_bank_report.json")
    noclip_dyn = bank_first["dynamics"]
    clip_dyn = g1a_bank_first["dynamics"]

    def _trace_stats(glob_pattern: str):
        first_clips, all_clips, norms, hvps = [], [], [], []
        for path in sorted(glob_pattern):
            if path.name.startswith("_"):
                continue
            rec = _read_json(path)
            trace = rec.get("step_trace", [])
            if not trace:
                continue
            first_clips.append(bool(trace[0]["clip_active"]))
            for row in trace:
                all_clips.append(bool(row["clip_active"]))
                norms.append(float(row["native_gradient_norm"]))
                if row["hvp_evaluated"]:
                    hvps.append(float(row["hvp_norm"]))
        return {
            "first_clip_rate": float(np.mean(first_clips)) if first_clips else math.nan,
            "all_clip_rate": float(np.mean(all_clips)) if all_clips else math.nan,
            "grad_norm_mean": float(np.mean(norms)) if norms else math.nan,
            "hvp_norm_mean": float(np.mean(hvps)) if hvps else math.nan,
        }

    noclip_tr = _trace_stats(list((NOCLIP / "streams").glob("stream_seed*_step*_s*.json")))
    clip_tr = _trace_stats(list((G1A_OUT / "tangents").glob("stream_seed*_step*_s*.json")))

    dynamics_table = {
        "training clip rate": (
            f"{float(np.mean([d['clip_active'] for d in clip_dyn])):.4f}",
            f"{float(np.mean([d['clip_active'] for d in noclip_dyn])):.4f} (by definition)",
        ),
        "training preclip norm mean": (
            f"{float(np.mean([d['preclip_grad_norm'] for d in clip_dyn])):.4f}",
            f"{float(np.mean([d['preclip_grad_norm'] for d in noclip_dyn])):.4f}",
        ),
        "counterfactual would-clip rate (preclip>1)": (
            "n/a",
            f"{float(np.mean([d['counterfactual_would_clip_threshold_1'] for d in noclip_dyn])):.4f}",
        ),
        "training loss mean": (
            f"{float(np.mean([d['train_loss_avg'] for d in clip_dyn])):.4f}",
            f"{float(np.mean([d['train_loss_avg'] for d in noclip_dyn])):.4f}",
        ),
        "future first-step clip rate": (
            f"{clip_tr['first_clip_rate']:.4f}",
            f"{noclip_tr['first_clip_rate']:.4f}",
        ),
        "future all-step clip rate": (
            f"{clip_tr['all_clip_rate']:.4f}",
            f"{noclip_tr['all_clip_rate']:.4f}",
        ),
        "future grad norm mean": (
            f"{clip_tr['grad_norm_mean']:.4f}",
            f"{noclip_tr['grad_norm_mean']:.4f}",
        ),
        "future HVP norm mean": (
            f"{clip_tr['hvp_norm_mean']:.4f}",
            f"{noclip_tr['hvp_norm_mean']:.4f}",
        ),
    }

    # ---------- verdict ------------------------------------------------------
    interaction_summary = {
        "mean_delta_clip": mean_delta,
        "median_delta_clip": float(np.median([p["delta_clip"] for p in paired_interaction])),
        "n_official_cells": len(paired_interaction),
        "n_stream_cells": len(stream_paired),
        "by_group": by_group,
        "bootstrap": boot["bootstrap"],
    }
    verdict = decide_mechanism_verdict(
        g3a_pass=bool(g3a["pass"]),
        g3a_max_abs_gap=float(g3a["max_abs_gap"]),
        clip_gains=clip_gains,
        noclip_gains=noclip_gains,
        noclip_stats=noclip_stats,
        delta_clip=mean_delta,
        contract_mismatch=not bool(smoke["overall_pass"]),
    )

    # ---------- outputs --------------------------------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "clipping_interaction.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(paired_interaction[0]))
        writer.writeheader()
        writer.writerows(paired_interaction)

    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE",
        "verdict": verdict,
        "g3a_operator_control": g3a,
        "clip_condition_gains": {
            "gain_curvature_mean": clip_gains["gain_curvature_mean"],
            "gain_clip_mean": clip_gains["gain_clip_mean"],
            "source": "frozen G1a per_stream.json + mechanism JSONs (seed 2042) + clipref (seeds 2043/2044)",
        },
        "noclip_condition_gains": noclip_gains,
        "noclip_statistics": noclip_stats,
        "interaction": interaction_summary,
        "dynamics_context": dynamics_table,
        "runtime": {
            "hvp_backend": "PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA",
            "baseline_replay": "PASS_EXACT" if noclip_session["replay_pass_all"] else "FAIL",
            "smoke_gate": "PASS" if smoke["overall_pass"] else "FAIL",
            "max_hvp_math_gradient_relative_gap_vs_native": noclip_session["max_hvp_math_gradient_relative_gap_vs_native"],
        },
        "scope_statement": (
            "G3a is an operator control, not causal regime evidence; the causal claim "
            "rests on G3b no-clip trajectories. Curvature feedback is not claimed to "
            "vanish. No-clip does not imply C == FD (finite-epsilon FD nonlinearity)."
        ),
        "next_step": "G2 (26M data-distribution generalization) is a user decision; not started.",
    }
    _assert_finite_tree({k: v for k, v in summary.items()}, "summary")
    _write_json(RESULTS / "summary.json", summary)

    context = {
        "model": dict(cfg["model"]) | {"actual_unique_params": 26681728},
        "g3a": g3a,
        "training": {
            "seeds": list(TRAIN_SEEDS),
            "states": [STAGE_STEPS[s] for s in STAGES],
            "clip_grad_norm_called": False,
            "counterfactual_would_clip_rate": float(np.mean([d["counterfactual_would_clip_threshold_1"] for d in noclip_dyn])),
            "training_preclip_norm_mean": float(np.mean([d["preclip_grad_norm"] for d in noclip_dyn])),
            "g1a_training_preclip_norm_mean": float(np.mean([d["preclip_grad_norm"] for d in clip_dyn])),
        },
        "noclip_stats": noclip_stats,
        "gains": {"clip": clip_gains, "noclip": noclip_gains},
        "interaction": {
            "mean_delta_clip": interaction_summary["mean_delta_clip"],
            "bootstrap": interaction_summary["bootstrap"],
            "by_group": interaction_summary["by_group"],
        },
        "verdict": verdict,
        "dynamics_table": dynamics_table,
        "incidents": [
            f"smoke gate: overall_pass={smoke['overall_pass']}, h1_analytic_vs_fd_rel_err={smoke['checks']['h1_analytic_vs_fd_rel_err']:.6g}",
            f"GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log",
            "no OOM; no retry; no RESOURCE_LIMIT",
        ],
    }
    (OUT / "G3_CLIPPING_INTERVENTION_REPORT.md").write_text(render_report(context), encoding="utf-8")
    _manifest_touch("analyze", "COMPLETE", {"verdict": verdict["verdict"]})
    print(json_dumps(verdict))
    print(f"SUMMARY = {RESULTS / 'summary.json'}")
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(_args) -> int:
    def _is_large_tensor(rel: str) -> bool:
        return rel.startswith("bank/") or "/q/" in rel

    included, excluded = [], []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(OUT).as_posix()
        if rel in {"SHA256SUMS.txt", "AUDIT_BUNDLE_SHA256.txt"} or rel.endswith(".tmp"):
            continue
        entry = (rel, file_sha256(path))
        if _is_large_tensor(rel):
            excluded.append(entry)
        else:
            included.append(entry)
    lines = [
        "# SHA256SUMS (FR_G3_CLIPPING_INTERVENTION_V01)",
        "# Section 1: files included in g3_clipping_intervention_audit_package.tar.gz",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in included]
    lines += [
        "",
        "# Section 2: large tensor payloads EXCLUDED from the audit bundle "
        "(hashes recorded; reproducible from frozen protocol inputs):",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in excluded]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = OUT / "g3_clipping_intervention_audit_package.tar.gz"
    if bundle.exists():
        bundle.unlink()
    with tarfile.open(bundle, "w:gz") as tar:
        for rel, _digest in included:
            tar.add(OUT / rel, arcname=f"g3_clipping_intervention_v01/{rel}")
        tar.add(OUT / "SHA256SUMS.txt", arcname="g3_clipping_intervention_v01/SHA256SUMS.txt")
    bundle_sha = file_sha256(bundle)
    (OUT / "AUDIT_BUNDLE_SHA256.txt").write_text(
        f"{bundle_sha}  g3_clipping_intervention_audit_package.tar.gz\n", encoding="utf-8"
    )
    with tarfile.open(bundle, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            while fh.read(1 << 20):
                pass
    print(f"AUDIT_BUNDLE = {bundle}")
    print(f"AUDIT_BUNDLE_SHA256 = {bundle_sha}")
    print(f"included={len(included)} excluded_large={len(excluded)}")
    return 0


# ---------------------------------------------------------------------------

def cmd_officialboot(_args) -> int:
    """[FR-G3-PHASE-A] Official-cell hierarchical bootstrap for Delta_clip.

    Uses exactly the frozen official-cell estimator (per (seed, stage, H)
    cell: mean of the 4 stream chi -> cell-level relative error vs
    cell-mean chi_FD -> per-cell gain_clip = rel_err_BH_fixed - rel_err_C;
    H in {4,8}; Delta_clip = equal-weight mean over the 18 paired cells),
    resampling seed -> stage -> H-cell with replacement at every level
    (H kept whole within stage).  20000 resamples, rng 20260916.
    Read-only over frozen G1a/G3 artifacts; writes only new files under
    results/.  Direction consistency with the frozen stream-level
    bootstrap is checked; disagreement returns exit code 2
    (STATISTICAL_MISMATCH).
    """
    g1a_per_stream = _read_json(G1A_OUT / "results" / "per_stream.json")["rows"]
    clip_stream = {}
    for row in g1a_per_stream:
        key = (int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))
        clip_stream[key] = {"chi_fd": float(row["chi_fd"]), "chi_BH_fixed": math.nan, "chi_C": float(row["chi_C"])}
    for mech_path in sorted((G1A_OUT / "mechanism").glob("stream_seed*_step*_s*.json")):
        rec = _read_json(mech_path)
        for h in HORIZONS_OFFICIAL:
            key = (int(rec["seed"]), int(rec["step"]), int(rec["stream_id"]), h)
            if key in clip_stream:
                clip_stream[key]["chi_BH_fixed"] = float(rec["chi_BH_fixed"][str(h)])
    for row in csv.DictReader((CLIPREF / "per_stream_bh_fixed.csv").open("r", encoding="utf-8")):
        key = (int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))
        if key in clip_stream:
            clip_stream[key]["chi_BH_fixed"] = float(row["chi_BH_fixed"])
    noclip_stream = {}
    for row in csv.DictReader((NOCLIP / "mechanism_decomposition.csv").open("r", encoding="utf-8")):
        noclip_stream[(int(row["seed"]), int(row["step"]), int(row["stream_id"]), int(row["H"]))] = {
            "chi_fd": float(row["chi_fd"]),
            "chi_BH_fixed": float(row["chi_BH_fixed"]),
            "chi_C": float(row["chi_C"]),
        }

    def _official_gain(stream_map):
        grouped = {}
        for (seed, step, _sid, h), v in stream_map.items():
            grouped.setdefault((seed, step, h), []).append(v)
        out = {}
        for (seed, step, h), rows in grouped.items():
            fd = float(np.mean([r["chi_fd"] for r in rows]))
            bh = float(np.mean([r["chi_BH_fixed"] for r in rows]))
            c = float(np.mean([r["chi_C"] for r in rows]))
            out[(seed, step, h)] = _rel(fd, bh) - _rel(fd, c)
        return out

    clip_gain = _official_gain(clip_stream)
    noclip_gain = _official_gain(noclip_stream)
    paired_cells = []
    for key in sorted(clip_gain):
        seed, step, h = key
        if h not in H_GT_1 or key not in noclip_gain:
            continue
        paired_cells.append({
            "seed": seed, "stage": stage_of_step(step), "step": step, "H": h,
            "gain_clip_clip": float(clip_gain[key]),
            "gain_clip_noclip": float(noclip_gain[key]),
            "delta_clip": float(clip_gain[key] - noclip_gain[key]),
        })
    if len(paired_cells) != 18:
        raise RuntimeError(f"official-cell pairing must yield 18 cells, got {len(paired_cells)}")
    boot = cluster_bootstrap_interaction(paired_cells, n_resamples=20000, seed=20260916)
    stream_boot = _read_json(RESULTS / "summary.json")["interaction"]["bootstrap"]
    direction_consistent = bool((boot["observed_mean_delta_clip"] > 0) and (stream_boot["mean"] > 0))
    payload = {
        "protocol": PROTOCOL,
        "subphase": "official_cell_bootstrap_phase_a",
        "estimand": ("Delta_clip = equal-weight mean over the 18 official (seed,stage,H) cells of "
                     "[gain_clip(clip=1.0) - gain_clip(no-clip)]; official cell = mean of the 4 "
                     "stream chi; cell-level relative error vs cell-mean chi_FD "
                     "(frozen a0c relative_error); gain_clip(cell) = rel_err_BH_fixed - rel_err_C; H in {4,8}"),
        "aggregation": "official-cell; identical to results/summary.json interaction.mean_delta_clip",
        "resampling_hierarchy": "seed -> stage -> H-cell, with replacement at every level; H kept whole within stage",
        "n_resamples": 20000,
        "rng_seed": 20260916,
        "observed_delta_clip": float(boot["observed_mean_delta_clip"]),
        "bootstrap": {
            "mean": float(boot["bootstrap"]["mean"]),
            "median": float(boot["bootstrap"]["median"]),
            "pi95_low": float(boot["bootstrap"]["pi95_low"]),
            "pi95_high": float(boot["bootstrap"]["pi95_high"]),
            "fraction_resamples_delta_positive": float(boot["bootstrap"]["p_positive"]),
        },
        "interpretation_note": ("fraction_resamples_delta_positive is the fraction of bootstrap "
                                "resamples in which the difference is positive; it is NOT a p-value, "
                                "a posterior probability, or a significance probability."),
        "stream_level_bootstrap_frozen_reference": stream_boot,
        "direction_consistent_with_stream_level_bootstrap": direction_consistent,
        "statistical_mismatch": bool(not direction_consistent),
        "paired_cells": paired_cells,
    }
    (RESULTS / "official_cell_bootstrap.json").write_text(json_dumps(payload), encoding="utf-8")
    with (RESULTS / "official_cell_bootstrap_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "stage", "step", "H", "gain_clip_clip", "gain_clip_noclip", "delta_clip"])
        writer.writeheader()
        writer.writerows(paired_cells)
    print(json_dumps({
        "observed_delta_clip": payload["observed_delta_clip"],
        "bootstrap": payload["bootstrap"],
        "direction_consistent_with_stream_level_bootstrap": direction_consistent,
    }))
    if not direction_consistent:
        print("STATISTICAL_MISMATCH: official-cell vs stream-level bootstrap directions disagree", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preregister")
    p.set_defaults(func=cmd_preregister)

    p = sub.add_parser("g3a_operator")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_g3a_operator)

    p = sub.add_parser("train")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("theta")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="")
    p.set_defaults(func=cmd_theta)

    p = sub.add_parser("smoke")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("clipref")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_clipref)

    p = sub.add_parser("tangents")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="")
    p.set_defaults(func=cmd_tangents)

    p = sub.add_parser("analyze")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("officialboot")
    p.set_defaults(func=cmd_officialboot)

    p = sub.add_parser("audit")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    _log_command([Path(sys.argv[0]).name] + sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
