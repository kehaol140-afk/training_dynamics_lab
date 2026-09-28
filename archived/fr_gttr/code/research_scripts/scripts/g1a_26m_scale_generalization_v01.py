#!/usr/bin/env python3
"""G1a — 26M scale-only generalization of the frozen FD/B/BH_fixed/C mechanism.

Subcommands (run in order):
  preregister  write experiment_config/run_manifest/hashes/environment (no GPU)
  train        build the 3-seed x 3-stage 26M state bank with dynamics recording
  theta        q construction + E1 + frozen stream manifests + E2 FD branches
  smoke        single-state numerical/implementation gate (seed 2042, step 64)
  tangents     chi_B / chi_C replay on all 9 states x 4 streams (reuses frozen a0c runner)
  mechanism    B / BH_fixed / C / FD on seed 2042 x 3 stages (reuses frozen a0e2 runner)
  analyze      statistics, frozen-threshold verdict, report, CSVs
  audit        SHA256SUMS + audit bundle (excludes large checkpoints)

[FR-OBSERVATION-ONLY] v07 bank artifacts and all frozen a0* modules/outputs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import shutil
import subprocess
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
    save_direction,
)
from fr.full_state import load_full_training_state, save_full_training_state
from fr.g1a_26m_scale_generalization_v01 import (
    E1_BASE_SEED,
    E1_STREAMS,
    E2_BASE_SEED,
    E2_MANIFEST_PROTOCOL,
    E2_STREAMS,
    HORIZONS_OFFICIAL,
    PROTOCOL,
    Q_PROTOCOL,
    RHO,
    STAGE_FRACTIONS,
    STAGE_STEPS,
    STAGE_TOKENS,
    STAGES,
    TOTAL_OPT_STEPS,
    TRAIN_SEEDS,
    aggregate_metrics,
    batch_seed_from_stream_seed,
    cell_relative_errors,
    decide_verdict,
    json_dumps,
    render_report,
    stage_of_step,
    two_level_seed,
    validate_g1a_e2_manifest,
)
from fr.batch_manifest import save_manifest
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import load_partition
from fr.response_stats import observational_slope, paired_linearity_summary
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed
from fr.theta_protocol import run_theta_branch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/generalization_g1a_26m_v01"
BANK = OUT / "bank"
THETA = OUT / "theta"
TANGENTS = OUT / "tangents"
MECH = OUT / "mechanism"
RESULTS = OUT / "results"
CONFIG_PATH = ROOT / "research/configs/g1a_26m_v01.json"
V07_DIR = ROOT / "outputs/e3_confirmatory/v07"
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
    """Import a frozen runner script by file path (never edit it)."""
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


def _state_paths():
    for seed in TRAIN_SEEDS:
        for stage in STAGES:
            step = STAGE_STEPS[stage]
            yield seed, stage, step, BANK / f"seed_{seed}" / "states" / f"state_step_{step:03d}.pt"


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
        "g1a_config": CONFIG_PATH,
        "g1a_protocol_md": OUT / "G1A_PROTOCOL.md",
        "g1a_module": SRC_ROOT / "fr/g1a_26m_scale_generalization_v01.py",
        "g1a_runner": RESEARCH_ROOT / "scripts/g1a_26m_scale_generalization_v01.py",
        "frozen_a0c_module": SRC_ROOT / "fr/a0c_adamw_response_propagation_v01.py",
        "frozen_a0c_runner": RESEARCH_ROOT / "scripts/a0c_adamw_response_propagation_v01.py",
        "frozen_a0e2_module": SRC_ROOT / "fr/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "frozen_a0e2_runner": RESEARCH_ROOT / "scripts/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "theta_protocol_pyc": SRC_ROOT / "fr/theta_protocol.pyc",
        "minimind_model": ROOT / "minimind/model/model_minimind.py",
        "minimind_trainer_utils": ROOT / "minimind/trainer/trainer_utils.py",
        "v07_data_d0": V07_DIR / "e3c_subset.jsonl",
        "v07_partition": V07_DIR / "partition_v07.json",
        "v07_config": ROOT / "research/configs/e3c_confirmatory_v07.json",
        "full_state_module": SRC_ROOT / "fr/full_state.py",
        "direction_module": SRC_ROOT / "fr/direction.py",
        "stream_rng_module": SRC_ROOT / "fr/stream_rng.py",
        "batch_manifest_module": SRC_ROOT / "fr/batch_manifest.py",
        "stage_c_runtime_module": SRC_ROOT / "fr/stage_c_runtime.py",
        "native_schedule_module": SRC_ROOT / "fr/native_schedule.py",
        "native_adamw_guard_module": SRC_ROOT / "fr/native_adamw_guard.py",
        "response_stats_module": SRC_ROOT / "fr/response_stats.py",
        "pilot_partition_module": SRC_ROOT / "fr/pilot_partition.py",
        "hashing_module": SRC_ROOT / "fr/hashing.py",
    }
    lines = ["# SOURCE_AND_INPUT_HASHES (FR_G1A_26M_SCALE_GENERALIZATION_V01)", ""]
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
        "phases": ["preregister", "train", "theta", "smoke", "tangents", "mechanism", "analyze", "audit"],
        "phase_status": {p: "NOT_STARTED" for p in
                         ["train", "theta", "smoke", "tangents", "mechanism", "analyze", "audit"]},
        "source_hashes": hashes,
        "output_subdirs": ["bank", "theta", "tangents", "mechanism", "results"],
    }
    _write_json(OUT / "run_manifest.json", manifest)
    print("PREREGISTER_DONE")
    print(f"  protocol   = {OUT / 'G1A_PROTOCOL.md'}")
    print(f"  config     = {OUT / 'experiment_config.json'}")
    print(f"  hashes     = {OUT / 'SOURCE_AND_INPUT_HASHES.txt'}")
    print(f"  env        = {OUT / 'ENVIRONMENT.json'}")
    return 0


def _manifest_touch(phase: str, status: str, extra: dict | None = None) -> None:
    path = OUT / "run_manifest.json"
    manifest = _read_json(path)
    manifest["phase_status"][phase] = status
    if extra:
        manifest.setdefault("phase_details", {})[phase] = extra
    _write_json(path, manifest)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    device = args.device
    torch, model, tokenizer, get_lr, setup_seed = _minimind_setup(device)
    cfg = _read_json(CONFIG_PATH)
    partition = load_partition(V07_DIR / "partition_v07.json")
    data_path = V07_DIR / "e3c_subset.jsonl"
    ds_pre = __import__("dataset.lm_dataset", fromlist=["PretrainDataset"]).PretrainDataset(
        str(data_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"])
    )
    autocast_ctx, scaler, device_type = make_amp(torch, device, cfg["training"]["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("G1a bf16 requires GradScaler disabled")
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
    grad_clip = float(cfg["training"]["grad_clip"])
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
        # re-init model per seed for a clean trajectory
        model, tokenizer = _reinit_model(device, train_seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        assert_native_adamw(optimizer)
        optimizer.zero_grad(set_to_none=True)

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
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    .detach().cpu()
                )
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
                    "clip_active": bool(preclip > grad_clip),
                    "param_norm": pnorm,
                })
                loss_accum = 0.0
                if opt_step in set(STAGE_STEPS.values()):
                    path = states_dir / f"state_step_{opt_step:03d}.pt"
                    stage = stage_of_step(opt_step)
                    st = save_full_training_state(
                        path, model, optimizer, scaler,
                        metadata={
                            "protocol": "G1A_26M_STATE_BANK_V01",
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
            "protocol": "G1A_26M_STATE_BANK_V01",
            "technical_pass": len(records) == len(STAGES),
            "train_seed": train_seed,
            "train_batch_seed": train_batch_seed,
            "model_params": param_counts,
            "state_steps": [r["step"] for r in records],
            "states": records,
            "dynamics": dynamics,
            "config_sha256": file_sha256(CONFIG_PATH),
        }
        _write_json(report_path, report)
        _write_json(seed_dir / "training_dynamics.json", {"dynamics": dynamics})
        print(f"SEED_{train_seed}_BANK_PASS states={len(records)}")
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    training_manifest = {
        "protocol": "G1A_26M_STATE_BANK_V01",
        "seeds": seeds,
        "stages": {s: {"step": STAGE_STEPS[s], "fraction": STAGE_FRACTIONS[s],
                       "cumulative_tokens": STAGE_TOKENS[s]} for s in STAGES},
        "total_planned_optimizer_steps": TOTAL_OPT_STEPS,
        "tokens_per_optimizer_step": 1024,
        "training_terminated_at_step": max(STAGE_STEPS.values()),
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


# ---------------------------------------------------------------------------
# theta (q + E1 + manifests + E2 FD)
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
        "model_hash_equal": baseline["model_hash_final"] == noop["model_hash_final"],
        "optimizer_hash_equal": baseline["optimizer_hash_final"] == noop["optimizer_hash_final"],
        "observations_equal": baseline["observations"] == noop["observations"],
        "batch_manifest_hash_equal": baseline["batch_manifest_hash"] == noop["batch_manifest_hash"],
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
    grad_clip = float(cfg["training"]["grad_clip"])
    max_h = max(HORIZONS_OFFICIAL)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(TRAIN_SEEDS)

    # ---------------- phase 1: q + E1 + epsilon + ALL manifests --------------
    state_ctx = {}
    for seed, stage, step, state_path in _state_paths():
        if seed not in seeds:
            continue
        if not state_path.is_file():
            raise FileNotFoundError(f"missing checkpoint {state_path}; run train first")
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
                    "protocol": "FR_G1A_26M_E1_STREAM",
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
                batch_size=batch_size, device=device, grad_clip=grad_clip,
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
            })
        sigma_q = float(np.std(np.asarray(s_values, dtype=np.float64), ddof=1))
        if not math.isfinite(sigma_q) or sigma_q <= 0:
            raise RuntimeError(f"seed={seed} step={step} sigma_q invalid: {sigma_q}")
        epsilon = RHO * sigma_q
        b_slopes = {}
        for h in HORIZONS_OFFICIAL:
            hk = str(h)
            b_slopes[hk] = None  # filled after E2 in v07; here E1-only slope optional

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

    # freeze ALL future-stream manifests before any E2 branch executes
    future_stream_manifest = {
        "protocol": PROTOCOL,
        "ordering_rule": "all 36 manifests generated and written before any E2 response branch",
        "e2_base_seed": E2_BASE_SEED,
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

    # ---------------- phase 2: E2 paired branches ----------------------------
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
                manifest_payload = _read_json(manifest_path)
                manifest = validate_g1a_e2_manifest(
                    manifest_payload,
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
                    batch_size=batch_size, device=device, grad_clip=grad_clip,
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
            "protocol": "FR_G1A_26M_THETA_MAPPING_V01",
            "technical_pass": True,
            "rho0_noop_pass": rho0_noop_all_pass,
            "train_seed": seed,
            "primary_rho": RHO,
            "horizons": list(HORIZONS_OFFICIAL),
            "config_sha256": file_sha256(CONFIG_PATH),
            "states": state_results,
        }
        _write_json(result_path, result)
        print(f"SEED_{seed}_THETA_MAPPING_PASS")

    _manifest_touch("theta", "COMPLETE", {
        "states": len([1 for s in seeds for _ in STAGES]),
        "e2_streams": len(seeds) * len(STAGES) * E2_STREAMS,
    })
    print("THETA_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# shared tangent-stream driver
# ---------------------------------------------------------------------------

def _load_theta_state(seed: int, step: int):
    theta_path = THETA / f"seed_{seed}/theta_mapping_raw.json"
    theta = _read_json(theta_path)
    if theta.get("technical_pass") is not True or int(theta.get("train_seed", -1)) != seed:
        raise ValueError(f"invalid G1a theta record for seed {seed}")
    state = next((r for r in theta["states"] if int(r["step"]) == step), None)
    if state is None or len(state.get("e2_streams", [])) != E2_STREAMS:
        raise ValueError(f"missing G1a state/streams: seed={seed} step={step}")
    return theta, state


def _iter_stream_jobs(seeds):
    for seed in seeds:
        for _seed, stage, step, state_path in _state_paths():
            if _seed == seed:
                yield seed, stage, step, state_path


def _tangent_common(cfg, seed, step, state_path, theta_state, manifest, rec):
    torch = __import__("torch")
    meta = load_full_training_state(state_path, _TANGENT_MODEL, _TANGENT_OPTIMIZER, _TANGENT_SCALER, restore_rng=False)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    q_by_name, q_meta, q_sha = load_direction(theta_state["q_path"])
    if q_sha != theta_state["q_sha256"]:
        raise ValueError("q identity mismatch")
    return {
        "checkpoint_path": state_path,
        "reference_theta": payload["model"],
        "q_by_name": q_by_name,
        "batches": manifest["batches"],
        "stream_seed": int(rec["stream_seed"]),
        "accumulation_steps": int(cfg["training"]["accumulation_steps"]),
        "device": _TANGENT_DEVICE,
        "grad_clip": float(cfg["training"]["grad_clip"]),
        "base_lr": float(cfg["training"]["learning_rate"]),
        "global_micro_step": int(meta["global_micro_step"]),
        "total_micro_steps": int(meta["total_micro_steps"]),
    }


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
# tangents (B / C on all states)
# ---------------------------------------------------------------------------

def cmd_tangents(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0c_runner = _frozen_runner_import(
        "a0c_adamw_response_propagation_v01.py", "g1a_frozen_a0c_runner"
    )
    accum = int(cfg["training"]["accumulation_steps"])
    max_h = max(HORIZONS_OFFICIAL)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(TRAIN_SEEDS)
    TANGENTS.mkdir(parents=True, exist_ok=True)

    replay_passes = []
    clip_gaps = []
    hvp_gaps = []
    per_stream_rows = []
    stream_records = []

    for seed, stage, step, state_path in _iter_stream_jobs(seeds):
        _theta, theta_state = _load_theta_state(seed, step)
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            stream_id = int(rec["stream_id"])
            manifest_path = Path(rec["batch_manifest_path"])
            manifest = validate_g1a_e2_manifest(
                _read_json(manifest_path),
                expected_seed=seed, expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=rec["batch_manifest_sha256"],
                accumulation_steps=accum,
            )
            common = _tangent_common(cfg, seed, step, state_path, theta_state, manifest, rec)
            propagated = a0c_runner._run_baseline_tangents(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named, **common,
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
                grad_clip=float(cfg["training"]["grad_clip"]),
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
            if not replay_pass:
                raise RuntimeError(
                    f"baseline replay mismatch seed={seed} step={step} stream={stream_id}: "
                    f"model_equal={model_equal} optimizer_equal={optimizer_equal} gaps={gaps}"
                )
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
            out_path = TANGENTS / f"stream_seed{seed}_step{step:03d}_s{stream_id}.json"
            _write_json(out_path, stream_result)
            for h in HORIZONS_OFFICIAL:
                per_stream_rows.append({
                    "seed": seed, "stage": stage, "step": step,
                    "stream_id": stream_id, "H": h,
                    "chi_fd": stream_result["chi_fd"][str(h)],
                    "chi_B": stream_result["chi_B"][str(h)],
                    "chi_C": stream_result["chi_C"][str(h)],
                })
            print(
                f"TAN seed={seed} step={step} stream={stream_id} "
                f"B8={stream_result['chi_B']['8']:.9g} C8={stream_result['chi_C']['8']:.9g}",
                flush=True,
            )
        del checkpoint
        torch.cuda.empty_cache()

    _write_json(TANGENTS / "_session.json", {
        "replay_pass_all": all(replay_passes),
        "max_native_clip_tensor_l2_gap": float(max(clip_gaps)),
        "max_hvp_math_gradient_relative_gap_vs_native": float(max(hvp_gaps)),
        "stream_count": len(stream_records),
        "hvp_backend": "PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA",
        "native_flash_double_backward": "UNSUPPORTED_DERIVATIVE_FOR_FLASH_ATTENTION_BACKWARD",
    })
    _write_json(RESULTS / "per_stream.json", {"rows": per_stream_rows})
    with (RESULTS / "per_stream.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_stream_rows[0]))
        writer.writeheader()
        writer.writerows(per_stream_rows)
    _manifest_touch("tangents", "COMPLETE", {
        "streams": len(stream_records),
        "replay_pass_all": all(replay_passes),
        "max_native_clip_tensor_l2_gap": float(max(clip_gaps)),
    })
    print("TANGENTS_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# mechanism (B / BH_fixed / C on seed 2042 x 3 stages)
# ---------------------------------------------------------------------------

def cmd_mechanism(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import(
        "a0e2_curvature_decomposition_v02_bh_fixed.py", "g1a_frozen_a0e2_runner"
    )
    accum = int(cfg["training"]["accumulation_steps"])
    mech_seed = TRAIN_SEEDS[0]
    MECH.mkdir(parents=True, exist_ok=True)

    rows = []
    h1_identity_gaps = []
    for seed, stage, step, state_path in _iter_stream_jobs([mech_seed]):
        _theta, theta_state = _load_theta_state(seed, step)
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            stream_id = int(rec["stream_id"])
            manifest = validate_g1a_e2_manifest(
                _read_json(Path(rec["batch_manifest_path"])),
                expected_seed=seed, expected_step=step,
                expected_stream=stream_id,
                expected_content_sha=rec["batch_manifest_sha256"],
                accumulation_steps=accum,
            )
            common = _tangent_common(cfg, seed, step, state_path, theta_state, manifest, rec)
            propagated = a0e2_runner._run_baseline_tangents(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=ds, named=named, **common,
                get_lr=get_lr,
            )
            h1_identity_gaps.append(
                max(
                    abs(float(propagated["chi_optimizer_only"]["1"]) - float(propagated["chi_BH_fixed_curvature"]["1"])),
                    abs(float(propagated["chi_optimizer_only"]["1"]) - float(propagated["chi_full_tangent"]["1"])),
                )
            )
            stream_result = {
                "seed": seed, "stage": stage, "step": step, "stream_id": stream_id,
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
                "chi_BH_old_byproduct": {k: float(v) for k, v in propagated["chi_BH_old_curvature"].items()},
                "step_trace": propagated["step_trace"],
            }
            _write_json(MECH / f"stream_seed{seed}_step{step:03d}_s{stream_id}.json", stream_result)
            for h in HORIZONS_OFFICIAL:
                rows.append({
                    "seed": seed, "stage": stage, "step": step,
                    "stream_id": stream_id, "H": h,
                    "chi_fd": stream_result["chi_fd"][str(h)],
                    "chi_B": stream_result["chi_B"][str(h)],
                    "chi_BH_fixed": stream_result["chi_BH_fixed"][str(h)],
                    "chi_C": stream_result["chi_C"][str(h)],
                })
            print(f"MECH seed={seed} step={step} stream={stream_id} DONE", flush=True)
        torch.cuda.empty_cache()

    _write_json(MECH / "_session.json", {
        "seed": mech_seed,
        "h1_identity_max_abs_gap": float(max(h1_identity_gaps)),
        "bh_old_note": "chi_BH_old is a deprecated frozen-runner byproduct; excluded from official statistics",
    })
    with (RESULTS / "mechanism_decomposition.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _manifest_touch("mechanism", "COMPLETE", {
        "seed": mech_seed,
        "h1_identity_max_abs_gap": float(max(h1_identity_gaps)),
    })
    print("MECHANISM_COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# smoke gate
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    device = args.device
    seed, step = TRAIN_SEEDS[0], STAGE_STEPS["MIDDLE"]
    state_path = BANK / f"seed_{seed}/states/state_step_{step:03d}.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path}; run train first")
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0c_runner = _frozen_runner_import(
        "a0c_adamw_response_propagation_v01.py", "g1a_frozen_a0c_runner"
    )
    accum = int(cfg["training"]["accumulation_steps"])
    report = {"protocol": PROTOCOL, "state": {"seed": seed, "step": step}, "checks": {}}

    # 1) checkpoint full restore + optimizer state dims/names
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
    report["checks"]["rng_restored"] = True  # restore_rng=True above

    # 2) q names/shapes
    _theta, theta_state = _load_theta_state(seed, step)
    q_by_name, q_meta, q_sha = load_direction(theta_state["q_path"])
    q_ok = set(q_by_name) == {n for n, _ in named} and all(
        tuple(q_by_name[n].shape) == tuple(p.shape) for n, p in named
    )
    report["checks"]["q_names_shapes"] = bool(q_ok and q_sha == theta_state["q_sha256"])

    # 3) one E2 stream: FD +/- CRN + rho0 noop
    rec = sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"]))[0]
    manifest = validate_g1a_e2_manifest(
        _read_json(Path(rec["batch_manifest_path"])),
        expected_seed=seed, expected_step=step, expected_stream=int(rec["stream_id"]),
        expected_content_sha=rec["batch_manifest_sha256"], accumulation_steps=accum,
    )
    common = _tangent_common(cfg, seed, step, state_path, theta_state, manifest, rec)

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
        for branch in (b0, bp, bm)
        for obs in branch["observations"].values()
    )
    report["checks"]["fd_crn_consistent"] = bool(crn_ok)
    report["checks"]["rho0_noop"] = bool(noop_ok)
    report["checks"]["fd_outputs_finite"] = bool(finite_ok)

    # 4) B/C tangents + H=1 consistency + shadow hash + HVP
    propagated = a0c_runner._run_baseline_tangents(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, dataset=ds, named=named, **common, get_lr=get_lr,
    )
    chi_fd_h1 = float(
        (rec["plus"]["X_theta"]["1"] - rec["minus"]["X_theta"]["1"])
        / (2.0 * float(theta_state["epsilon"]))
    )
    h1_rel_err = abs(propagated["chi_optimizer_only"]["1"] - chi_fd_h1) / max(abs(chi_fd_h1), 1e-30)
    report["checks"]["h1_analytic_vs_fd_rel_err"] = float(h1_rel_err)
    report["checks"]["h1_consistency"] = bool(h1_rel_err < 0.3)
    report["checks"]["h1_B_equals_C"] = bool(
        propagated["chi_optimizer_only"]["1"] == propagated["chi_full_tangent"]["1"]
    )
    shadow = run_theta_branch(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
        reference_theta=payload["model"], q=q_by_name, named=named,
        batches=manifest["batches"], stream_seed=int(rec["stream_seed"]),
        sign=0, epsilon=0.0, horizons=HORIZONS_OFFICIAL, accumulation_steps=accum,
        batch_size=int(cfg["training"]["batch_size"]), device=device,
        grad_clip=float(cfg["training"]["grad_clip"]),
        base_lr=float(cfg["training"]["learning_rate"]),
        global_micro_step=int(meta["global_micro_step"]),
        total_micro_steps=int(meta["total_micro_steps"]),
        get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
    )
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
    _assert_finite_tree(
        {k: v for k, v in propagated.items() if k != "step_trace"}, "smoke"
    )
    report["checks"]["all_outputs_finite"] = True

    checks = report["checks"]
    report["overall_pass"] = bool(
        checks["optimizer_state_dims_names"] and checks["q_names_shapes"]
        and checks["fd_crn_consistent"] and checks["rho0_noop"]
        and checks["fd_outputs_finite"] and checks["h1_consistency"]
        and checks["h1_B_equals_C"] and checks["shadow_hash_match"]
        and checks["hvp_executed"] and checks["hvp_finite"]
        and checks["no_unexpected_optimizer_steps"] and checks["all_outputs_finite"]
    )
    _write_json(OUT / "smoke_report.json", report)
    print(json_dumps(report))
    if not report["overall_pass"]:
        raise RuntimeError("SMOKE_GATE_FAILED")
    _manifest_touch("smoke", "PASS", {"seed": seed, "step": step})
    print("SMOKE_PASS")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def cmd_analyze(_args) -> int:
    from fr.g1a_26m_scale_generalization_v01 import relative_error as _rel_err

    cfg = _read_json(CONFIG_PATH)
    per_stream = _read_json(RESULTS / "per_stream.json")["rows"]

    # official per-state-horizon rows: means over 4 streams
    grouped = {}
    for row in per_stream:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        grouped.setdefault(key, {"chi_fd": [], "chi_B": [], "chi_C": []})
        grouped[key]["chi_fd"].append(float(row["chi_fd"]))
        grouped[key]["chi_B"].append(float(row["chi_B"]))
        grouped[key]["chi_C"].append(float(row["chi_C"]))
    official_rows = []
    for (seed, step, h), vals in sorted(grouped.items()):
        chi_fd = float(np.mean(vals["chi_fd"]))
        chi_b = float(np.mean(vals["chi_B"]))
        chi_c = float(np.mean(vals["chi_C"]))
        official_rows.append({
            "seed": seed, "stage": stage_of_step(step), "step": step, "H": h,
            "chi_fd": chi_fd,
            "chi_fd_stream_std": float(np.std(vals["chi_fd"], ddof=1)),
            "chi_B": chi_b,
            "chi_B_stream_std": float(np.std(vals["chi_B"], ddof=1)),
            "chi_C": chi_c,
            "chi_C_stream_std": float(np.std(vals["chi_C"], ddof=1)),
            "n_streams": len(vals["chi_fd"]),
        })
    errors = cell_relative_errors(official_rows)
    for row in official_rows:
        errs = errors[(int(row["seed"]), int(row["step"]), int(row["H"]))]
        row.update(errs)
    stats = aggregate_metrics(official_rows)
    verdict = decide_verdict(stats)

    with (RESULTS / "per_state_horizon.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(official_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(official_rows)

    # mechanism aggregation
    mech_session = _read_json(MECH / "_session.json")
    mech_rows = list(csv.DictReader((RESULTS / "mechanism_decomposition.csv").open("r", encoding="utf-8")))
    mech_grouped = {}
    for row in mech_rows:
        key = (row["stage"], int(row["H"]))
        mech_grouped.setdefault(key, {"B": [], "BH": [], "C": []})
        mech_grouped[key]["B"].append(float(row["chi_B"]))
        mech_grouped[key]["BH"].append(float(row["chi_BH_fixed"]))
        mech_grouped[key]["C"].append(float(row["chi_C"]))
    mech_fd = {}
    for row in mech_rows:
        key = (row["stage"], int(row["H"]))
        mech_fd.setdefault(key, []).append(float(row["chi_fd"]))

    def _mean_rel(preds, obs):
        vals = [
            _rel_err(o, p) for p, o in zip(preds, obs)
        ]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def _gain(a, b):
        if a is None or b is None:
            return None
        return a - b

    mech_by_stage = {}
    gains_curv = []
    gains_clip = []
    for stage in STAGES:
        mech_by_stage[stage] = {}
        for h in HORIZONS_OFFICIAL:
            key = (stage, h)
            fd = mech_fd[key]
            mb = _mean_rel(mech_grouped[key]["B"], fd)
            mbh = _mean_rel(mech_grouped[key]["BH"], fd)
            mc = _mean_rel(mech_grouped[key]["C"], fd)
            mech_by_stage[stage][str(h)] = {
                "mean_rel_err_B": mb,
                "mean_rel_err_BH_fixed": mbh,
                "mean_rel_err_C": mc,
                "gain_curvature": _gain(mb, mbh),
                "gain_clip": _gain(mbh, mc),
            }
            if h > 1:
                gc = _gain(mb, mbh)
                gl = _gain(mbh, mc)
                if gc is not None:
                    gains_curv.append(gc)
                if gl is not None:
                    gains_clip.append(gl)
    mechanism = {
        "by_stage": mech_by_stage,
        "h1_identity_max_abs_gap": float(mech_session["h1_identity_max_abs_gap"]),
        "pooled_h_gt_1": {
            "gain_curvature_mean": float(np.mean(gains_curv)) if gains_curv else None,
            "gain_clip_mean": float(np.mean(gains_clip)) if gains_clip else None,
        },
    }

    # epsilon per state
    epsilon_by_state = {}
    for seed in TRAIN_SEEDS:
        theta = _read_json(THETA / f"seed_{seed}/theta_mapping_raw.json")
        for st in theta["states"]:
            epsilon_by_state[f"{seed}:{st['step']}"] = float(st["epsilon"])

    # dynamics context: G1a vs 5M anchor
    dyn_g1a_future_clip = []
    dyn_g1a_first_norms = []
    for path in sorted(TANGENTS.glob("stream_seed*_step*.json")):
        rec = _read_json(path)
        trace = rec["step_trace"]
        dyn_g1a_future_clip.append(bool(trace[0]["clip_active"]))
        dyn_g1a_first_norms.append(float(trace[0]["native_gradient_norm"]))
    bank_report = _read_json(BANK / f"seed_{TRAIN_SEEDS[0]}/state_bank_report.json")
    training_clip = [d["clip_active"] for d in bank_report["dynamics"]]
    training_norms = [d["preclip_grad_norm"] for d in bank_report["dynamics"]]

    anchor_clip = []
    anchor_norms = []
    for seed, step in ((2037, 46), (2030, 46), (2037, 82)):
        theta = _read_json(V07_DIR / f"seed_{seed}/theta_mapping_raw.json")
        state = next(r for r in theta["states"] if int(r["step"]) == step)
        for rec in state["e2_streams"]:
            anchor_clip.append(bool(rec["baseline"]["first_clipped"]))
            anchor_norms.append(float(rec["baseline"]["first_preclip_grad_norm"]))

    dynamics = {
        "g1a": {
            "future_clip_rate_step1": float(np.mean(dyn_g1a_future_clip)),
            "first_preclip_norm_mean": float(np.mean(dyn_g1a_first_norms)),
            "first_preclip_norm_median": float(np.median(dyn_g1a_first_norms)),
            "training_clip_rate": float(np.mean(training_clip)),
            "training_preclip_norm_mean": float(np.mean(training_norms)),
        },
        "anchor5m": {
            "future_clip_rate_step1": float(np.mean(anchor_clip)),
            "first_preclip_norm_mean": float(np.mean(anchor_norms)),
            "first_preclip_norm_median": float(np.median(anchor_norms)),
            "training_clip_rate": None,
            "training_preclip_norm_mean": None,
        },
    }

    tan_session = _read_json(TANGENTS / "_session.json")
    smoke = _read_json(OUT / "smoke_report.json")
    incidents = []
    if not tan_session["replay_pass_all"]:
        incidents.append("baseline replay mismatch detected (would have raised during tangents)")
    incidents.append(
        f"smoke gate: overall_pass={smoke['overall_pass']}, "
        f"h1_analytic_vs_fd_rel_err={smoke['checks']['h1_analytic_vs_fd_rel_err']:.6g}"
    )
    incidents.append(
        f"GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log"
    )

    context = {
        "model": dict(cfg["model"]) | {
            "actual_unique_params": bank_report["model_params"]["unique_params"],
            "actual_state_dict_elements": bank_report["model_params"]["state_dict_elements"],
        },
        "actual_unique_params": bank_report["model_params"]["unique_params"],
        "actual_state_dict_elements": bank_report["model_params"]["state_dict_elements"],
        "epsilon_by_state": epsilon_by_state,
        "official_rows": official_rows,
        "cell_errors": errors,
        "stats": stats,
        "verdict": verdict,
        "mechanism": mechanism,
        "dynamics": dynamics,
        "incidents": incidents,
        "next_step": "G1b (~100M) is a user decision; this experiment does not start it.",
    }
    report_md = render_report(context)
    (OUT / "G1A_26M_SCALE_GENERALIZATION_REPORT.md").write_text(report_md, encoding="utf-8")

    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE",
        "verdict": verdict,
        "scope": {
            "seeds": list(TRAIN_SEEDS),
            "stages": {s: STAGE_STEPS[s] for s in STAGES},
            "horizons_official": list(HORIZONS_OFFICIAL),
            "states": 9,
            "streams_per_state": E2_STREAMS,
            "official_state_horizon_cells": len(official_rows),
        },
        "model": dict(cfg["model"]) | {
            "actual_unique_params": bank_report["model_params"]["unique_params"],
            "actual_state_dict_elements": bank_report["model_params"]["state_dict_elements"],
        },
        "results_per_state_horizon": official_rows,
        "statistics": stats,
        "mechanism_decomposition": mechanism,
        "dynamics_context": dynamics,
        "runtime": {
            "hvp_backend": tan_session["hvp_backend"],
            "native_flash_double_backward": tan_session["native_flash_double_backward"],
            "hvp_interpretation": "MIXED_BACKEND_ANALYTIC_SURROGATE_NOT_EXACT_FLASH_KERNEL_TANGENT",
            "max_native_clip_tensor_l2_gap": tan_session["max_native_clip_tensor_l2_gap"],
            "max_hvp_math_gradient_relative_gap_vs_native": tan_session["max_hvp_math_gradient_relative_gap_vs_native"],
            "baseline_replay": "PASS_EXACT_H8_SHADOW_HASHES_AND_X" if tan_session["replay_pass_all"] else "FAIL",
            "smoke_gate": "PASS" if smoke["overall_pass"] else "FAIL",
        },
        "scope_statement": (
            "G1a is model-scale generalization only; it does not replace A0F2 and does not "
            "by itself establish 100M/1B generalization."
        ),
    }
    _assert_finite_tree(
        {k: v for k, v in summary.items() if k not in {"results_per_state_horizon"}},
        "summary",
    )
    _write_json(RESULTS / "summary.json", summary)
    _manifest_touch("analyze", "COMPLETE", {"verdict": verdict["verdict"]})
    print(json_dumps(verdict))
    print(f"SUMMARY = {RESULTS / 'summary.json'}")
    print(f"REPORT  = {OUT / 'G1A_26M_SCALE_GENERALIZATION_REPORT.md'}")
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(_args) -> int:
    # Large tensors are hashed but NOT bundled (same policy as bank/):
    # they are reproducible from the frozen protocol inputs (state checkpoint
    # + partition + q construction rule), and bundling ~3.8GB is impractical.
    excluded_prefixes = (
        BANK.relative_to(OUT).as_posix() + "/",
    )

    def _is_large_tensor(rel: str) -> bool:
        return rel.startswith(excluded_prefixes) or (
            rel.startswith("theta/") and "/q/" in rel
        )

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
        "# SHA256SUMS (FR_G1A_26M_SCALE_GENERALIZATION_V01)",
        "# Section 1: files included in g1a_26m_audit_package.tar.gz",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in included]
    lines += [
        "",
        "# Section 2: large tensor payloads EXCLUDED from the audit bundle "
        "(hashes recorded; reproducible from frozen protocol inputs):",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in excluded]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = OUT / "g1a_26m_audit_package.tar.gz"
    if bundle.exists():
        bundle.unlink()
    with tarfile.open(bundle, "w:gz") as tar:
        for rel, _digest in included:
            tar.add(OUT / rel, arcname=f"g1a_26m_v01/{rel}")
        tar.add(OUT / "SHA256SUMS.txt", arcname="g1a_26m_v01/SHA256SUMS.txt")
    bundle_sha = file_sha256(bundle)
    (OUT / "AUDIT_BUNDLE_SHA256.txt").write_text(
        f"{bundle_sha}  g1a_26m_audit_package.tar.gz\n", encoding="utf-8"
    )
    # full read-back integrity verification
    with tarfile.open(bundle, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            while fh.read(1 << 20):
                pass
    print(f"AUDIT_BUNDLE = {bundle}")
    print(f"AUDIT_BUNDLE_SHA256 = {bundle_sha}")
    print(f"SHA256SUMS = {OUT / 'SHA256SUMS.txt'}")
    print(f"included={len(included)} excluded_large={len(excluded)}")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preregister")
    p.set_defaults(func=cmd_preregister)

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

    p = sub.add_parser("tangents")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="")
    p.set_defaults(func=cmd_tangents)

    p = sub.add_parser("mechanism")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_mechanism)

    p = sub.add_parser("analyze")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("audit")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    _log_command([Path(sys.argv[0]).name] + sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
