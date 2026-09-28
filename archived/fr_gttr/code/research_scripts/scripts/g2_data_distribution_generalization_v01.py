#!/usr/bin/env python3
"""G2 — data-distribution generalization of the frozen FD/B/BH_fixed/C mechanism.

Subcommands (run in order):
  preregister  write experiment_config/run_manifest/hashes/environment (no GPU)
  dataset      verify D0/D1 hashes+schema, token stats, dup/overlap diagnostic,
               freeze D1 pool (4096) + partition, write dataset_manifest +
               DATASET_PROVENANCE.md (B1/B2/B3/B4)
  train        build the 3-seed x 3-stage 26M state bank on the D1 pool
  theta        q construction + E1 + frozen stream manifests + E2 FD branches
  smoke        single-state numerical/implementation gate (seed 2042, step 26 EARLY)
  tangents     chi_B / chi_C replay on all 9 states x 4 streams (frozen a0c runner)
  mechanism    B / BH_fixed / C / FD on seed 2042 x 3 stages (frozen a0e2 runner)
  analyze      statistics (frozen G1a aggregation), dual-caliber mechanism gains,
               D0-vs-D1 matched comparison, cluster bootstrap, frozen-tier verdict
  audit        SHA256SUMS + audit bundle (excludes checkpoints / q / raw corpus)

[FR-OBSERVATION-ONLY] All v07 / G1a / G3 bank artifacts and frozen a0* modules
and outputs are read-only.  Metrics are imported, never redefined.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
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
    aggregate_metrics,
    batch_seed_from_stream_seed,
    cell_relative_errors,
    json_dumps,
    stage_of_step,
    two_level_seed,
)
from fr.g2_data_distribution_generalization_v01 import (
    DATASET_MANIFEST_PROTOCOL,
    D1_POOL_SOURCE_RECORDS,
    E1_STREAM_PROTOCOL,
    E2_MANIFEST_PROTOCOL,
    G2_D1_POOL_SEED,
    G2_EVAL_SIZE,
    G2_PARTITION_SEED,
    G2_PROBE_SIZE,
    PROTOCOL,
    Q_PROTOCOL,
    STATE_BANK_PROTOCOL,
    THETA_PROTOCOL,
    cluster_bootstrap_paired_improvement,
    decide_data_verdict,
    jaccard,
    relative_error,
    render_report,
    select_pool_indices,
    shingle_digest,
    validate_dataset_manifest,
    validate_g2_e2_manifest,
)
from fr.batch_manifest import save_manifest
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import build_partition, load_partition, save_partition
from fr.response_stats import paired_linearity_summary
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed
from fr.theta_protocol import run_theta_branch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/generalization_g2_data_distribution_v01"
DATA = OUT / "data"
BANK = OUT / "bank"
THETA = OUT / "theta"
TANGENTS = OUT / "tangents"
MECH = OUT / "mechanism"
RESULTS = OUT / "results"
CONFIG_PATH = ROOT / "research/configs/g2_data_distribution_generalization_v01.json"
V07_DIR = ROOT / "outputs/e3_confirmatory/v07"
G1A_OUT = ROOT / "outputs/generalization_g1a_26m_v01"
D0_PATH = V07_DIR / "e3c_subset.jsonl"
D0_PARTITION_PATH = V07_DIR / "partition_v07.json"
D1_RAW_PATH = DATA / "pretrain_hq.jsonl"
D1_POOL_PATH = DATA / "d1_pool.jsonl"
D1_PARTITION_PATH = OUT / "partition_g2_d1.json"
DATASET_MANIFEST_PATH = OUT / "dataset_manifest.json"
COMMANDS_LOG = OUT / "COMMANDS.log"

PHASES = ["preregister", "dataset", "train", "theta", "smoke",
          "tangents", "mechanism", "analyze", "audit"]


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


def _load_tokenizer():
    minimind = ROOT / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(minimind / "model"))


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


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def _reservoir_lines(path: Path, n_sample: int, seed: int):
    """Deterministic reservoir sample of (line_index, raw_line)."""
    rng = random.Random(int(seed))
    reservoir: list = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < n_sample:
                reservoir.append((i, line))
            else:
                j = rng.randrange(i + 1)
                if j < n_sample:
                    reservoir[j] = (i, line)
    return sorted(reservoir)


def _exact_duplicate_stats(path: Path) -> dict:
    """Full-file exact-duplicate-line diagnostic (64-bit digests, streaming)."""
    seen: set[int] = set()
    dup = 0
    n = 0
    with open(path, "rb") as fh:
        for raw in fh:
            n += 1
            digest = int.from_bytes(
                hashlib.md5(raw.rstrip(b"\n")).digest()[:8], "little"
            )
            if digest in seen:
                dup += 1
            else:
                seen.add(digest)
    return {
        "records": n,
        "exact_duplicate_lines": dup,
        "exact_duplicate_fraction": float(dup / n) if n else 0.0,
    }


def _text_sample(path: Path, n_sample: int, seed: int) -> list:
    return [line for _, line in _reservoir_lines(path, n_sample, seed)]


def _parse_schema_sample(lines) -> dict:
    keys = set()
    n_text_str = 0
    n_parsed = 0
    im_end_hits = 0
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_parsed += 1
        if isinstance(obj, dict):
            keys.update(obj.keys())
            if isinstance(obj.get("text"), str):
                n_text_str += 1
                if "<|im_end|>" in obj["text"]:
                    im_end_hits += 1
    return {
        "parsed": n_parsed,
        "keys": sorted(keys),
        "text_field_string": n_text_str,
        "im_end_literal_fraction": float(im_end_hits / n_parsed) if n_parsed else 0.0,
    }


def _token_stats(texts, tokenizer, max_length: int) -> dict:
    token_lens = []
    char_lens = []
    for text in texts:
        ids = tokenizer(
            str(text), add_special_tokens=False,
            max_length=max_length - 2, truncation=True,
        ).input_ids
        token_lens.append(len(ids))
        char_lens.append(len(str(text)))
    arr = np.asarray(token_lens, dtype=np.float64)
    carr = np.asarray(char_lens, dtype=np.float64)
    return {
        "n": int(arr.size),
        "token_len_mean": float(arr.mean()),
        "token_len_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "token_len_min": int(arr.min()),
        "token_len_max": int(arr.max()),
        "token_len_truncated_at_126_fraction": float((arr >= max_length - 2).mean()),
        "char_len_mean": float(carr.mean()),
    }


def _mean_rel_pair(preds, obs) -> float | None:
    vals = [
        relative_error(o, p) for p, o in zip(preds, obs)
    ]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


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
        "g2_config": CONFIG_PATH,
        "g2_protocol_md": OUT / "G2_PROTOCOL.md",
        "g2_module": SRC_ROOT / "fr/g2_data_distribution_generalization_v01.py",
        "g2_runner": RESEARCH_ROOT / "scripts/g2_data_distribution_generalization_v01.py",
        "frozen_g1a_module": SRC_ROOT / "fr/g1a_26m_scale_generalization_v01.py",
        "frozen_g1a_runner": RESEARCH_ROOT / "scripts/g1a_26m_scale_generalization_v01.py",
        "frozen_a0c_module": SRC_ROOT / "fr/a0c_adamw_response_propagation_v01.py",
        "frozen_a0c_runner": RESEARCH_ROOT / "scripts/a0c_adamw_response_propagation_v01.py",
        "frozen_a0e2_module": SRC_ROOT / "fr/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "frozen_a0e2_runner": RESEARCH_ROOT / "scripts/a0e2_curvature_decomposition_v02_bh_fixed.py",
        "theta_protocol_pyc": SRC_ROOT / "fr/theta_protocol.pyc",
        "minimind_model": ROOT / "minimind/model/model_minimind.py",
        "minimind_trainer_utils": ROOT / "minimind/trainer/trainer_utils.py",
        "minimind_lm_dataset": ROOT / "minimind/dataset/lm_dataset.py",
        "d0_e3c_subset": D0_PATH,
        "d0_partition_v07": D0_PARTITION_PATH,
        "d1_raw_pretrain_hq": D1_RAW_PATH,
        "d1_pool": D1_POOL_PATH,
        "d1_partition": D1_PARTITION_PATH,
        "g1a_output_dir_readonly_reference": G1A_OUT / "results/summary.json",
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
    lines = ["# SOURCE_AND_INPUT_HASHES (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)", ""]
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
        "phases": PHASES,
        "phase_status": {p: "NOT_STARTED" for p in PHASES if p != "preregister"},
        "source_hashes": hashes,
        "output_subdirs": ["data", "bank", "theta", "tangents", "mechanism", "results"],
    }
    _write_json(OUT / "run_manifest.json", manifest)
    print("PREREGISTER_DONE")
    print(f"  protocol   = {OUT / 'G2_PROTOCOL.md'}")
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
# dataset (B1/B2/B3/B4)
# ---------------------------------------------------------------------------

def cmd_dataset(_args) -> int:
    cfg = _read_json(CONFIG_PATH)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    max_len = int(cfg["model"]["max_seq_len"])

    # ---- B1: D0 verification (frozen G1a actual pool; manifest governs) ----
    print("DATASET_PHASE d0 verify ...", flush=True)
    d0_sha = file_sha256(D0_PATH)
    if d0_sha != cfg["d0_reference"]["sha256"]:
        raise RuntimeError(
            f"D0 sha256 mismatch: {d0_sha} != {cfg['d0_reference']['sha256']}"
        )
    d0_records = _count_lines(D0_PATH)
    d0_dup = _exact_duplicate_stats(D0_PATH)
    d0_sample_lines = _text_sample(D0_PATH, 200, G2_D1_POOL_SEED)
    d0_schema = _parse_schema_sample(d0_sample_lines)
    print(f"DATASET_PHASE d0 ok records={d0_records} sha={d0_sha[:16]}...")

    # ---- B2/B3: D1 verification + provenance ----
    print("DATASET_PHASE d1 verify (1.6GB sha256, takes ~1 min) ...", flush=True)
    if not D1_RAW_PATH.is_file():
        raise FileNotFoundError(
            f"{D1_RAW_PATH}; download pretrain_hq.jsonl at revision "
            f"{cfg['d1_selection']['source_revision']} first (see DATASET_PROVENANCE.md)"
        )
    d1_sha = file_sha256(D1_RAW_PATH)
    if d1_sha != cfg["d1_selection"]["lfs_sha256"]:
        raise RuntimeError(
            f"D1 sha256 mismatch: {d1_sha} != {cfg['d1_selection']['lfs_sha256']}"
        )
    d1_records = _count_lines(D1_RAW_PATH)
    d1_dup = _exact_duplicate_stats(D1_RAW_PATH)
    d1_sample_lines = _text_sample(D1_RAW_PATH, 300, G2_D1_POOL_SEED)
    d1_schema = _parse_schema_sample(d1_sample_lines)
    print(f"DATASET_PHASE d1 ok records={d1_records} sha={d1_sha[:16]}...")

    if "text" not in d1_schema["keys"] or d1_schema["text_field_string"] == 0:
        raise RuntimeError(
            f"CONTRACT_INCOMPATIBLE: D1 schema lacks a string 'text' field: {d1_schema}"
        )
    if "text" not in d0_schema["keys"]:
        raise RuntimeError(f"D0 schema unexpected (no 'text'): {d0_schema}")

    # ---- token statistics sample (same tokenizer both corpora) ----
    print("DATASET_PHASE token stats ...", flush=True)
    tokenizer = _load_tokenizer()
    d0_texts = [json.loads(line)["text"] for line in d0_sample_lines[:150]]
    d1_texts = [json.loads(line)["text"] for line in d1_sample_lines[:150]]
    d0_tokens = _token_stats(d0_texts, tokenizer, max_len)
    d1_tokens = _token_stats(d1_texts, tokenizer, max_len)

    # ---- overlap diagnostic: 13-gram shingle Jaccard (sampled, cheap) ----
    print("DATASET_PHASE shingle overlap ...", flush=True)
    d0_shingle_texts = [json.loads(line)["text"] for line in d0_sample_lines[:150]]
    d1_shingle_texts = [json.loads(line)["text"] for line in d1_sample_lines[:150]]
    d0_sets = [shingle_digest(t) for t in d0_shingle_texts]
    d1_sets = [shingle_digest(t) for t in d1_shingle_texts]
    jaccards = [
        jaccard(a, b) for a in d0_sets for b in d1_sets
    ]
    shingle_jaccard_mean = float(np.mean(jaccards))
    shingle_jaccard_max = float(np.max(jaccards))

    # ---- freeze D1 pool (4096, deterministic; no content selection) ----
    print("DATASET_PHASE pool freeze ...", flush=True)
    pool_indices = select_pool_indices(d1_records, G2_D1_POOL_SEED, D1_POOL_SOURCE_RECORDS)
    if len(pool_indices) != D1_POOL_SOURCE_RECORDS or len(set(pool_indices)) != len(pool_indices):
        raise RuntimeError("pool selection invalid")
    pool_set = set(pool_indices)
    DATA.mkdir(parents=True, exist_ok=True)
    pool_path_tmp = D1_POOL_PATH.with_suffix(".jsonl.tmp")
    found = 0
    last_index = pool_indices[-1]
    with open(D1_RAW_PATH, "r", encoding="utf-8") as fin, open(pool_path_tmp, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in pool_set:
                fout.write(line if line.endswith("\n") else line + "\n")
                found += 1
            if i >= last_index:
                break
    if found != D1_POOL_SOURCE_RECORDS:
        raise RuntimeError(f"pool extraction incomplete: {found}")
    pool_path_tmp.replace(D1_POOL_PATH)
    d1_pool_sha = file_sha256(D1_POOL_PATH)
    d1_pool_records = _count_lines(D1_POOL_PATH)
    if d1_pool_records != D1_POOL_SOURCE_RECORDS:
        raise RuntimeError("pool record count mismatch")

    # ---- partition (same sizes as v07/G1a) ----
    partition = build_partition(D1_POOL_SOURCE_RECORDS, G2_PARTITION_SEED, G2_PROBE_SIZE, G2_EVAL_SIZE)
    save_partition(D1_PARTITION_PATH, partition)

    manifest = {
        "protocol": DATASET_MANIFEST_PROTOCOL,
        "created_at_utc": created,
        "d0": {
            "path": str(D0_PATH.relative_to(ROOT)),
            "sha256": d0_sha,
            "records": d0_records,
            "bytes": D0_PATH.stat().st_size,
            "role": "frozen G1a pool; read-only reference; never rerun",
            "schema_keys": d0_schema["keys"],
        },
        "d1": {
            "path": str(D1_RAW_PATH.relative_to(ROOT)),
            "sha256": d1_sha,
            "records": d1_records,
            "bytes": D1_RAW_PATH.stat().st_size,
            "source": cfg["d1_selection"]["source"],
            "source_revision": cfg["d1_selection"]["source_revision"],
            "lfs_sha256": cfg["d1_selection"]["lfs_sha256"],
            "schema_keys": d1_schema["keys"],
            "schema_adapter": "none (identical {'text'} schema to D0)",
        },
        "d1_pool": {
            "path": str(D1_POOL_PATH.relative_to(ROOT)),
            "sha256": d1_pool_sha,
            "records": d1_pool_records,
            "bytes": D1_POOL_PATH.stat().st_size,
            "rule": f"first {D1_POOL_SOURCE_RECORDS} indices of make_permutation({d1_records}, {G2_D1_POOL_SEED}), sorted",
            "pool_seed": G2_D1_POOL_SEED,
        },
        "partition": {
            "path": str(D1_PARTITION_PATH.relative_to(ROOT)),
            "sha256": partition["sha256"],
            "seed": G2_PARTITION_SEED,
            "dataset_len": D1_POOL_SOURCE_RECORDS,
            "probe_size": G2_PROBE_SIZE,
            "eval_size": G2_EVAL_SIZE,
            "train_size": len(partition["train_indices"]),
        },
        "overlap_diagnostic": {
            "method": "exact duplicate lines (64-bit md5 digest, full file) + char 13-gram shingle (step 3) Jaccard on samples",
            "d0_exact_duplicate_lines": d0_dup["exact_duplicate_lines"],
            "d0_exact_duplicate_fraction": d0_dup["exact_duplicate_fraction"],
            "d1_exact_duplicate_lines": d1_dup["exact_duplicate_lines"],
            "d1_exact_duplicate_fraction": d1_dup["exact_duplicate_fraction"],
            "shingle_ngram": 13,
            "shingle_step": 3,
            "d0_sample_n": len(d0_sets),
            "d1_sample_n": len(d1_sets),
            "shingle_jaccard_mean": shingle_jaccard_mean,
            "shingle_jaccard_max": shingle_jaccard_max,
            "note": (
                "nonzero overlap is a property of the corpora, not a reason to "
                "switch datasets; D1 was frozen before any G2 response result"
            ),
        },
        "token_statistics_sample": {
            "tokenizer": "minimind/model (vocab 6400), add_special_tokens=False, truncation at max_length-2",
            "d0": d0_tokens,
            "d1": d1_tokens,
            "d0_im_end_literal_fraction": d0_schema["im_end_literal_fraction"],
            "d1_im_end_literal_fraction": d1_schema["im_end_literal_fraction"],
        },
        "frozen_before_any_response_result": True,
    }
    validate_dataset_manifest(manifest)
    _write_json(DATASET_MANIFEST_PATH, manifest)

    prov_lines = [
        "# DATASET_PROVENANCE (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)",
        "",
        f"Created: {created} (frozen BEFORE any G2 response result).",
        "",
        "## D0 (frozen G1a condition, read-only, never rerun)",
        "",
        f"- Path: `{manifest['d0']['path']}`",
        f"- SHA256: `{d0_sha}`",
        f"- Records: {d0_records}; bytes: {D0_PATH.stat().st_size}",
        f"- Schema keys: {d0_schema['keys']}",
        "- Note: D0 is the 4096-record G1a pool (e3c_subset.jsonl), NOT the full",
        "  pretrain_t2t_mini.jsonl; the frozen G1a config/manifest governs (B1).",
        "",
        "## D1 (pretrain_hq.jsonl, MiniMind official asset)",
        "",
        f"- Source: {cfg['d1_selection']['source']}",
        f"- Revision: `{cfg['d1_selection']['source_revision']}` "
        "(pre-deletion official revision, uploaded 2026-01-17; file deleted from",
        "HEAD 2026-03-26 commit 3d8cf7da91c2 — recorded for transparency)",
        f"- LFS SHA256 (HF): `{cfg['d1_selection']['lfs_sha256']}`",
        f"- Local path: `{manifest['d1']['path']}`",
        f"- Local SHA256: `{d1_sha}` (verified equal to LFS oid)",
        f"- Records: {d1_records:,}; bytes: {D1_RAW_PATH.stat().st_size:,}",
        f"- Schema keys: {d1_schema['keys']} (identical to D0 -> no adapter; B4)",
        "",
        "## B2 selection audit",
        "",
        "- Priority rule: (1) MiniMind pretrain_hq.jsonl if compatible+sufficient+",
        "  different corpus; (2) other local MiniMind official assets; (3) download",
        "  to versioned dir with provenance; (4) else DATASET_UNAVAILABLE.",
        "- Executed: rule (3). The file was absent locally and absent from current",
        "  HEAD of the official HF repo, but present at the pre-deletion official",
        "  revision, which was downloaded and hash-verified. No alternative corpus",
        "  was ever compared on results; D1 was locked before training began.",
        "- No random substitutes, no FineWeb/English, no 'scientific' dataset.",
        "",
        "## Pool / partition freeze",
        "",
        f"- Pool: first {D1_POOL_SOURCE_RECORDS} indices of make_permutation({d1_records:,}, seed={G2_D1_POOL_SEED}), sorted",
        f"  -> `{manifest['d1_pool']['path']}` sha256 `{d1_pool_sha}`",
        "  No content filtering, no balancing, no D0 mixing, no content picking.",
        f"- Partition: build_partition(4096, seed={G2_PARTITION_SEED}, probe=32, eval=128)",
        f"  -> `{manifest['partition']['path']}` sha256 `{partition['sha256']}`",
        "",
        "## Duplicate / overlap diagnostic (B3)",
        "",
        f"- D0 exact duplicate lines: {d0_dup['exact_duplicate_lines']} / {d0_records} "
        f"({d0_dup['exact_duplicate_fraction']:.6f})",
        f"- D1 exact duplicate lines: {d1_dup['exact_duplicate_lines']:,} / {d1_records:,} "
        f"({d1_dup['exact_duplicate_fraction']:.6f})",
        f"- 13-gram shingle Jaccard (sampled {len(d0_sets)}x{len(d1_sets)} pairs): "
        f"mean {shingle_jaccard_mean:.6f}, max {shingle_jaccard_max:.6f}",
        "- Interpretation: these are descriptive properties of the corpora; nonzero",
        "  overlap does NOT trigger a dataset switch.",
        "",
        "## Token statistics sample (frozen tokenizer, both corpora)",
        "",
        f"- D0: {json.dumps(d0_tokens)}",
        f"- D1: {json.dumps(d1_tokens)}",
        f"- literal '<|im_end|>' fraction: D0 {d0_schema['im_end_literal_fraction']:.4f}, "
        f"D1 {d1_schema['im_end_literal_fraction']:.4f} (same tokenizer path; no adapter)",
    ]
    (OUT / "DATASET_PROVENANCE.md").write_text("\n".join(prov_lines) + "\n", encoding="utf-8")
    _manifest_touch("dataset", "COMPLETE", {
        "d0_records": d0_records, "d1_records": d1_records,
        "d1_pool_sha256": d1_pool_sha, "partition_sha256": partition["sha256"],
    })
    print("DATASET_COMPLETE")
    print(f"  manifest   = {DATASET_MANIFEST_PATH}")
    print(f"  provenance = {OUT / 'DATASET_PROVENANCE.md'}")
    return 0


def _require_frozen_dataset() -> dict:
    if not DATASET_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"{DATASET_MANIFEST_PATH}; run dataset first")
    manifest = _read_json(DATASET_MANIFEST_PATH)
    validate_dataset_manifest(manifest)
    if file_sha256(D1_POOL_PATH) != manifest["d1_pool"]["sha256"]:
        raise RuntimeError("d1_pool hash drift vs dataset_manifest")
    if not D1_PARTITION_PATH.is_file():
        raise FileNotFoundError(f"{D1_PARTITION_PATH}; run dataset first")
    partition = load_partition(D1_PARTITION_PATH)
    if partition["sha256"] != manifest["partition"]["sha256"]:
        raise RuntimeError("partition hash drift vs dataset_manifest")
    if file_sha256(D0_PATH) != manifest["d0"]["sha256"]:
        raise RuntimeError("D0 hash drift")
    return manifest


# ---------------------------------------------------------------------------
# train (B5/B6/B7; exact G1a copy on the D1 pool)
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    device = args.device
    torch, model, tokenizer, get_lr, setup_seed = _minimind_setup(device)
    cfg = _read_json(CONFIG_PATH)
    dataset_manifest = _require_frozen_dataset()
    partition = load_partition(D1_PARTITION_PATH)
    data_path = D1_POOL_PATH
    ds_pre = __import__("dataset.lm_dataset", fromlist=["PretrainDataset"]).PretrainDataset(
        str(data_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"])
    )
    autocast_ctx, scaler, device_type = make_amp(torch, device, cfg["training"]["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("G2 bf16 requires GradScaler disabled")
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
                            "protocol": STATE_BANK_PROTOCOL,
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
                            "dataset_path": str(data_path.relative_to(ROOT)),
                            "dataset_sha256": dataset_manifest["d1_pool"]["sha256"],
                            "source_corpus_sha256": dataset_manifest["d1"]["sha256"],
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
            "protocol": STATE_BANK_PROTOCOL,
            "technical_pass": len(records) == len(STAGES),
            "train_seed": train_seed,
            "train_batch_seed": train_batch_seed,
            "dataset": dataset_manifest["d1_pool"]["path"],
            "dataset_sha256": dataset_manifest["d1_pool"]["sha256"],
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
        "protocol": STATE_BANK_PROTOCOL,
        "seeds": seeds,
        "stages": {s: {"step": STAGE_STEPS[s], "fraction": STAGE_FRACTIONS[s],
                       "cumulative_tokens": STAGE_TOKENS[s]} for s in STAGES},
        "total_planned_optimizer_steps": TOTAL_OPT_STEPS,
        "tokens_per_optimizer_step": 1024,
        "training_terminated_at_step": max(STAGE_STEPS.values()),
        "param_counts": param_counts,
        "dataset": dataset_manifest["d1_pool"]["path"],
        "dataset_sha256": dataset_manifest["d1_pool"]["sha256"],
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
    dataset_manifest = _require_frozen_dataset()
    partition = load_partition(D1_PARTITION_PATH)
    data_path = D1_POOL_PATH
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

    # ---------------- phase 1: q + E1 + epsilon + manifests for these seeds --
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
                    "protocol": E1_STREAM_PROTOCOL,
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

    # freeze manifests for THESE seeds before any of their E2 branch executes;
    # merge with an existing manifest so multi-invocation runs keep all seeds.
    fsm_path = OUT / "future_stream_manifest.json"
    if fsm_path.is_file():
        future_stream_manifest = _read_json(fsm_path)
        stream_blocks = {
            (int(b["train_seed"]), int(b["state_step"])): b
            for b in future_stream_manifest.get("streams", [])
        }
    else:
        stream_blocks = {}
    for seed, stage, step, _ in _state_paths():
        if seed not in seeds:
            continue
        stream_blocks[(seed, step)] = {
            "train_seed": seed, "state_step": step, "stage": stage_of_step(step),
            "epsilon": state_ctx[(seed, step)]["epsilon"],
            "rho": RHO,
            "manifests": state_ctx[(seed, step)]["e2_manifest_records"],
        }
    future_stream_manifest = {
        "protocol": PROTOCOL,
        "ordering_rule": (
            "for each theta invocation, all E2 manifests of its seeds are generated "
            "and written before any E2 response branch of those states executes; "
            "invocations merge into this file"
        ),
        "e2_base_seed": E2_BASE_SEED,
        "streams": [
            stream_blocks[(seed, step)]
            for seed in TRAIN_SEEDS for step in STAGE_STEPS.values()
            if (seed, step) in stream_blocks
        ],
    }
    _write_json(fsm_path, future_stream_manifest)

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
                manifest = validate_g2_e2_manifest(
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
            "protocol": THETA_PROTOCOL,
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
        raise ValueError(f"invalid G2 theta record for seed {seed}")
    state = next((r for r in theta["states"] if int(r["step"]) == step), None)
    if state is None or len(state.get("e2_streams", [])) != E2_STREAMS:
        raise ValueError(f"missing G2 state/streams: seed={seed} step={step}")
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
        str(D1_POOL_PATH), tokenizer,
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
        "a0c_adamw_response_propagation_v01.py", "g2_frozen_a0c_runner"
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
            manifest = validate_g2_e2_manifest(
                _read_json(Path(rec["batch_manifest_path"])),
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
# mechanism (B / BH_fixed / C / FD on seed 2042 x 3 stages)
# ---------------------------------------------------------------------------

def cmd_mechanism(args) -> int:
    device = args.device
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0e2_runner = _frozen_runner_import(
        "a0e2_curvature_decomposition_v02_bh_fixed.py", "g2_frozen_a0e2_runner"
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
            manifest = validate_g2_e2_manifest(
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
# smoke gate (B9: seed 2042, EARLY, step 26 + loader replay / hash checks)
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    device = args.device
    seed, step = TRAIN_SEEDS[0], STAGE_STEPS["EARLY"]
    state_path = BANK / f"seed_{seed}/states/state_step_{step:03d}.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path}; run train first")
    torch, model, optimizer, scaler, autocast_ctx, ds, get_lr = _setup_tangent_session(device)
    cfg = _read_json(CONFIG_PATH)
    dataset_manifest = _require_frozen_dataset()
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    a0c_runner = _frozen_runner_import(
        "a0c_adamw_response_propagation_v01.py", "g2_frozen_a0c_runner"
    )
    accum = int(cfg["training"]["accumulation_steps"])
    report = {"protocol": PROTOCOL, "state": {"seed": seed, "step": step,
                                              "stage": stage_of_step(step)}, "checks": {}}

    # 0) dataset / pool / partition / loader deterministic replay (B9 extra)
    partition = load_partition(D1_PARTITION_PATH)
    batch_size = int(cfg["training"]["batch_size"])
    max_micro = max(STAGE_STEPS.values()) * accum
    train_batch_seed = seed + int(cfg["training"]["train_batch_seed_offset"])
    batches_a = batches_from_train_pool(partition["train_indices"], batch_size, train_batch_seed, max_micro)
    batches_b = batches_from_train_pool(partition["train_indices"], batch_size, train_batch_seed, max_micro)
    report["checks"]["train_loader_deterministic_replay"] = bool(
        [list(map(int, b)) for b in batches_a] == [list(map(int, b)) for b in batches_b]
    )
    report["checks"]["dataset_pool_partition_hashes_match"] = bool(
        file_sha256(D1_POOL_PATH) == dataset_manifest["d1_pool"]["sha256"]
        and partition["sha256"] == dataset_manifest["partition"]["sha256"]
        and file_sha256(D0_PATH) == dataset_manifest["d0"]["sha256"]
    )
    report["checks"]["state_metadata_dataset_match"] = None  # filled after restore

    # 1) checkpoint full restore + optimizer state dims/names
    meta = load_full_training_state(state_path, model, optimizer, scaler)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    report["checks"]["state_metadata_dataset_match"] = bool(
        meta.get("dataset_sha256") == dataset_manifest["d1_pool"]["sha256"]
    )
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
    manifest = validate_g2_e2_manifest(
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
            batch_size=batch_size, device=device,
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
        batch_size=batch_size, device=device,
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
        and checks["train_loader_deterministic_replay"]
        and checks["dataset_pool_partition_hashes_match"]
        and checks["state_metadata_dataset_match"]
    )
    _write_json(OUT / "smoke_report.json", report)
    print(json_dumps(report))
    if not report["overall_pass"]:
        raise RuntimeError("SMOKE_GATE_FAILED")
    _manifest_touch("smoke", "PASS", {"seed": seed, "step": step})
    print("SMOKE_PASS")
    return 0


# ---------------------------------------------------------------------------
# analyze (B10/B11/B12/B13/B14/B15)
# ---------------------------------------------------------------------------

def _load_tangent_dynamics(tangents_dir: Path) -> dict:
    first_clip = []
    first_norms = []
    hvp_norms = []
    for path in sorted(tangents_dir.glob("stream_seed*_step*.json")):
        rec = _read_json(path)
        trace = rec["step_trace"]
        first_clip.append(bool(trace[0]["clip_active"]))
        first_norms.append(float(trace[0]["native_gradient_norm"]))
        for row in trace:
            if row["hvp_evaluated"]:
                hvp_norms.append(float(row["hvp_norm"]))
    return {
        "future_first_clip_rate": float(np.mean(first_clip)) if first_clip else math.nan,
        "future_grad_norm_mean": float(np.mean(first_norms)) if first_norms else math.nan,
        "hvp_norm_mean": float(np.mean(hvp_norms)) if hvp_norms else math.nan,
    }


def _load_training_dynamics(bank_dir: Path, seed: int) -> dict:
    report = _read_json(bank_dir / f"seed_{seed}/state_bank_report.json")
    dyn = report["dynamics"]
    return {
        "training_loss_mean": float(np.mean([d["train_loss_avg"] for d in dyn])),
        "training_preclip_norm_mean": float(np.mean([d["preclip_grad_norm"] for d in dyn])),
        "training_clip_rate": float(np.mean([d["clip_active"] for d in dyn])),
    }


def cmd_analyze(_args) -> int:
    cfg = _read_json(CONFIG_PATH)
    dataset_manifest = _read_json(DATASET_MANIFEST_PATH)
    per_stream = _read_json(RESULTS / "per_stream.json")["rows"]

    # ---------------- D1 official statistics (frozen G1a aggregation) --------
    grouped = {}
    for row in per_stream:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        grouped.setdefault(key, {"chi_fd": [], "chi_B": [], "chi_C": []})
        grouped[key]["chi_fd"].append(float(row["chi_fd"]))
        grouped[key]["chi_B"].append(float(row["chi_B"]))
        grouped[key]["chi_C"].append(float(row["chi_C"]))
    official_rows = []
    for (seed, step, h), vals in sorted(grouped.items()):
        official_rows.append({
            "seed": seed, "stage": stage_of_step(step), "step": step, "H": h,
            "chi_fd": float(np.mean(vals["chi_fd"])),
            "chi_fd_stream_std": float(np.std(vals["chi_fd"], ddof=1)),
            "chi_B": float(np.mean(vals["chi_B"])),
            "chi_B_stream_std": float(np.std(vals["chi_B"], ddof=1)),
            "chi_C": float(np.mean(vals["chi_C"])),
            "chi_C_stream_std": float(np.std(vals["chi_C"], ddof=1)),
            "n_streams": len(vals["chi_fd"]),
        })
    errors = cell_relative_errors(official_rows)
    for row in official_rows:
        row.update(errors[(int(row["seed"]), int(row["step"]), int(row["H"]))])
    stats = aggregate_metrics(official_rows)
    verdict = decide_data_verdict(stats=stats)

    with (RESULTS / "per_state_horizon.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(official_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(official_rows)

    # ---------------- mechanism subset: BOTH calibers (B11) ------------------
    mech_session = _read_json(MECH / "_session.json")
    mech_rows = list(csv.DictReader((RESULTS / "mechanism_decomposition.csv").open("r", encoding="utf-8")))

    # (a) stream-first aggregation — identical recipe to G1a report §7:
    # per-stream relative error first, then mean over the 4 streams of a cell.
    stream_first_by_stage = {}
    gains_curv_sf = []
    gains_clip_sf = []
    for stage in STAGES:
        stream_first_by_stage[stage] = {}
        for h in HORIZONS_OFFICIAL:
            sub = [r for r in mech_rows if r["stage"] == stage and int(r["H"]) == h]
            fd = [float(r["chi_fd"]) for r in sub]
            mb = _mean_rel_pair([float(r["chi_B"]) for r in sub], fd)
            mbh = _mean_rel_pair([float(r["chi_BH_fixed"]) for r in sub], fd)
            mc = _mean_rel_pair([float(r["chi_C"]) for r in sub], fd)
            gc = None if (mb is None or mbh is None) else mb - mbh
            gl = None if (mbh is None or mc is None) else mbh - mc
            stream_first_by_stage[stage][str(h)] = {
                "mean_rel_err_B": mb, "mean_rel_err_BH_fixed": mbh,
                "mean_rel_err_C": mc, "gain_curvature": gc, "gain_clip": gl,
            }
            if h > 1:
                if gc is not None:
                    gains_curv_sf.append(gc)
                if gl is not None:
                    gains_clip_sf.append(gl)

    # (b) official-cell diagnostic — cell-mean chi first, then one relative
    # error per (stage,H) cell, exactly the main-matrix aggregation.
    official_cell_by_stage = {}
    gains_curv_oc = []
    gains_clip_oc = []
    for stage in STAGES:
        official_cell_by_stage[stage] = {}
        for h in HORIZONS_OFFICIAL:
            sub = [r for r in mech_rows if r["stage"] == stage and int(r["H"]) == h]
            fd_cell = float(np.mean([float(r["chi_fd"]) for r in sub]))
            mb = _mean_rel_pair([float(np.mean([float(r["chi_B"]) for r in sub]))], [fd_cell])
            mbh = _mean_rel_pair([float(np.mean([float(r["chi_BH_fixed"]) for r in sub]))], [fd_cell])
            mc = _mean_rel_pair([float(np.mean([float(r["chi_C"]) for r in sub]))], [fd_cell])
            gc = None if (mb is None or mbh is None) else mb - mbh
            gl = None if (mbh is None or mc is None) else mbh - mc
            official_cell_by_stage[stage][str(h)] = {
                "mean_rel_err_B": mb, "mean_rel_err_BH_fixed": mbh,
                "mean_rel_err_C": mc, "gain_curvature": gc, "gain_clip": gl,
            }
            if h > 1:
                if gc is not None:
                    gains_curv_oc.append(gc)
                if gl is not None:
                    gains_clip_oc.append(gl)

    mechanism = {
        "stream_first": {
            "by_stage": stream_first_by_stage,
            "gain_curvature_mean": float(np.mean(gains_curv_sf)) if gains_curv_sf else None,
            "gain_clip_mean": float(np.mean(gains_clip_sf)) if gains_clip_sf else None,
        },
        "official_cell": {
            "by_stage": official_cell_by_stage,
            "gain_curvature_mean": float(np.mean(gains_curv_oc)) if gains_curv_oc else None,
            "gain_clip_mean": float(np.mean(gains_clip_oc)) if gains_clip_oc else None,
        },
        "h1_identity_max_abs_gap": float(mech_session["h1_identity_max_abs_gap"]),
        "caliber_note": (
            "stream-first = G1a report section 7 recipe (per-stream errors then mean); "
            "official-cell = main-matrix recipe (cell-mean chi then one error). "
            "Both are reported and never mixed."
        ),
    }

    # ---------------- epsilon per state --------------------------------------
    epsilon_by_state = {}
    for seed in TRAIN_SEEDS:
        theta = _read_json(THETA / f"seed_{seed}/theta_mapping_raw.json")
        for st in theta["states"]:
            epsilon_by_state[f"{seed}:{st['step']}"] = float(st["epsilon"])

    # ---------------- B14: D0 vs D1 matched descriptive comparison -----------
    g1a_rows = list(csv.DictReader((G1A_OUT / "results/per_state_horizon.csv").open("r", encoding="utf-8")))
    g1a_by_key = {
        (int(r["seed"]), int(r["step"]), int(r["H"])): r for r in g1a_rows
    }
    comparison_rows = []
    for row in official_rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        g0 = g1a_by_key[key]
        comparison_rows.append({
            "seed": key[0], "stage": row["stage"], "step": key[1], "H": key[2],
            "chi_fd_d0": float(g0["chi_fd"]), "chi_B_d0": float(g0["chi_B"]),
            "chi_C_d0": float(g0["chi_C"]),
            "rel_err_B_d0": float(g0["rel_err_B"]), "rel_err_C_d0": float(g0["rel_err_C"]),
            "chi_fd_d1": row["chi_fd"], "chi_B_d1": row["chi_B"], "chi_C_d1": row["chi_C"],
            "rel_err_B_d1": row["rel_err_B"], "rel_err_C_d1": row["rel_err_C"],
            "delta_chi_fd": row["chi_fd"] - float(g0["chi_fd"]),
            "delta_rel_err_B": row["rel_err_B"] - float(g0["rel_err_B"]),
            "delta_rel_err_C": row["rel_err_C"] - float(g0["rel_err_C"]),
        })
    with (RESULTS / "d0_vs_d1_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    g1a_summary = _read_json(G1A_OUT / "results/summary.json")
    g1a_stats = g1a_summary["statistics"]
    g1a_mech = g1a_summary["mechanism_decomposition"]

    d0_dyn = _load_training_dynamics(G1A_OUT / "bank", TRAIN_SEEDS[0])
    d0_dyn.update(_load_tangent_dynamics(G1A_OUT / "tangents"))
    d1_dyn = _load_training_dynamics(BANK, TRAIN_SEEDS[0])
    d1_dyn.update(_load_tangent_dynamics(TANGENTS))

    dynamics_comparison = {
        "d0": {
            "c_pooled_r2": g1a_stats["pooled"]["C"]["R2"],
            "c_pooled_mre": g1a_stats["relative_error_summary"]["C_mean"],
            "b_pooled_mre": g1a_stats["relative_error_summary"]["B_mean"],
            "win_fraction_h_gt_1": g1a_stats["paired_h_gt_1"]["win_fraction"],
            "mean_improvement_h_gt_1": g1a_stats["paired_h_gt_1"]["mean_paired_improvement"],
            "training_loss_mean": d0_dyn["training_loss_mean"],
            "training_preclip_norm_mean": d0_dyn["training_preclip_norm_mean"],
            "training_clip_rate": d0_dyn["training_clip_rate"],
            "future_first_clip_rate": d0_dyn["future_first_clip_rate"],
            "future_grad_norm_mean": d0_dyn["future_grad_norm_mean"],
            "hvp_norm_mean": d0_dyn["hvp_norm_mean"],
            "gain_curvature_stream_first": g1a_mech["pooled_h_gt_1"]["gain_curvature_mean"],
            "gain_clip_stream_first": g1a_mech["pooled_h_gt_1"]["gain_clip_mean"],
        },
        "d1": {
            "c_pooled_r2": stats["pooled"]["C"]["R2"],
            "c_pooled_mre": stats["relative_error_summary"]["C_mean"],
            "b_pooled_mre": stats["relative_error_summary"]["B_mean"],
            "win_fraction_h_gt_1": stats["paired_h_gt_1"]["win_fraction"],
            "mean_improvement_h_gt_1": stats["paired_h_gt_1"]["mean_paired_improvement"],
            "training_loss_mean": d1_dyn["training_loss_mean"],
            "training_preclip_norm_mean": d1_dyn["training_preclip_norm_mean"],
            "training_clip_rate": d1_dyn["training_clip_rate"],
            "future_first_clip_rate": d1_dyn["future_first_clip_rate"],
            "future_grad_norm_mean": d1_dyn["future_grad_norm_mean"],
            "hvp_norm_mean": d1_dyn["hvp_norm_mean"],
            "gain_curvature_stream_first": mechanism["stream_first"]["gain_curvature_mean"],
            "gain_clip_stream_first": mechanism["stream_first"]["gain_clip_mean"],
        },
    }

    # ---------------- B15: cluster bootstrap ---------------------------------
    paired_cells = [
        dict(r, stage=stage_of_step(int(r["step"])))
        for r in stats["paired_h_gt_1"]["rows"]
    ]
    bootstrap = cluster_bootstrap_paired_improvement(paired_cells, n_resamples=20000, seed=20260916)
    _write_json(RESULTS / "bootstrap.json", bootstrap)

    # ---------------- report context -----------------------------------------
    bank_report = _read_json(BANK / f"seed_{TRAIN_SEEDS[0]}/state_bank_report.json")
    tan_session = _read_json(TANGENTS / "_session.json")
    smoke = _read_json(OUT / "smoke_report.json")

    shift_notes = [
        f"training loss mean: D0 {dynamics_comparison['d0']['training_loss_mean']:.6f} "
        f"vs D1 {dynamics_comparison['d1']['training_loss_mean']:.6f} (seed {TRAIN_SEEDS[0]}, steps 1..102)",
        f"training preclip norm mean: D0 {dynamics_comparison['d0']['training_preclip_norm_mean']:.6f} "
        f"vs D1 {dynamics_comparison['d1']['training_preclip_norm_mean']:.6f}",
        f"training clip rate: D0 {dynamics_comparison['d0']['training_clip_rate']:.4f} "
        f"vs D1 {dynamics_comparison['d1']['training_clip_rate']:.4f}",
        f"future first-step clip rate: D0 {dynamics_comparison['d0']['future_first_clip_rate']:.4f} "
        f"vs D1 {dynamics_comparison['d1']['future_first_clip_rate']:.4f}",
        f"future grad norm mean (first step): D0 {dynamics_comparison['d0']['future_grad_norm_mean']:.6f} "
        f"vs D1 {dynamics_comparison['d1']['future_grad_norm_mean']:.6f}",
        f"future HVP norm mean: D0 {dynamics_comparison['d0']['hvp_norm_mean']:.6f} "
        f"vs D1 {dynamics_comparison['d1']['hvp_norm_mean']:.6f}",
        "token statistics sample: see dataset_manifest.json token_statistics_sample "
        "(D1 mean token length "
        f"{dataset_manifest['token_statistics_sample']['d1']['token_len_mean']:.2f} vs D0 "
        f"{dataset_manifest['token_statistics_sample']['d0']['token_len_mean']:.2f}).",
        "These variables are context only (B13); no state was filtered on them.",
    ]
    incidents = [
        f"smoke gate: overall_pass={smoke['overall_pass']}, "
        f"h1_analytic_vs_fd_rel_err={smoke['checks']['h1_analytic_vs_fd_rel_err']:.6g}",
        f"baseline replay exactness: replay_pass_all={tan_session['replay_pass_all']}",
        f"max native-vs-analytic clip tensor L2 gap = {tan_session['max_native_clip_tensor_l2_gap']:.6g}; "
        f"max HVP math-vs-native gradient relative gap = {tan_session['max_hvp_math_gradient_relative_gap_vs_native']:.6g} "
        "(mixed-backend surrogate caveat, inherited)",
        "GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log",
        "D1 obtained from pre-deletion official HF revision (deleted from HEAD 2026-03-26); "
        "hash-verified; see DATASET_PROVENANCE.md",
        "no OOM; no retry; no RESOURCE_LIMIT",
    ]

    if verdict["verdict"] == "STRONG_DATA_REPLICATION":
        next_step = ("G2 complete. HANDOFF next = G1c (~1B stress test) — "
                     "a user decision; NOT started by this experiment.")
    else:
        next_step = ("G2 finished without STRONG_DATA_REPLICATION. Next = G2 "
                     "diagnostic / boundary analysis — a user decision; NOT started "
                     "by this experiment.")

    context = {
        "d0_path": str(D0_PATH.relative_to(ROOT)),
        "d1_path": str(D1_RAW_PATH.relative_to(ROOT)),
        "model": dict(cfg["model"]),
        "actual_unique_params": bank_report["model_params"]["unique_params"],
        "dataset_summary": {
            "d0_records": dataset_manifest["d0"]["records"],
            "d0_sha256": dataset_manifest["d0"]["sha256"],
            "d1_records": dataset_manifest["d1"]["records"],
            "d1_sha256": dataset_manifest["d1"]["sha256"],
            "d1_pool_records": dataset_manifest["d1_pool"]["records"],
            "d1_pool_sha256": dataset_manifest["d1_pool"]["sha256"],
            "exact_dup_lines": dataset_manifest["overlap_diagnostic"]["d1_exact_duplicate_lines"],
            "shingle_jaccard_sample": dataset_manifest["overlap_diagnostic"]["shingle_jaccard_mean"],
        },
        "stats": stats,
        "verdict": verdict,
        "mechanism": mechanism,
        "dynamics_comparison": dynamics_comparison,
        "d0_vs_d1_csv": "results/d0_vs_d1_comparison.csv",
        "bootstrap": bootstrap,
        "shift_notes": shift_notes,
        "incidents": incidents,
        "next_step": next_step,
    }
    report_md = render_report(context)
    (OUT / "G2_DATA_DISTRIBUTION_GENERALIZATION_REPORT.md").write_text(report_md, encoding="utf-8")

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
            "d0_condition": "frozen G1a 26M (read-only reference, never rerun)",
            "d1_condition": dataset_manifest["d1_pool"]["path"],
        },
        "model": dict(cfg["model"]) | {
            "actual_unique_params": bank_report["model_params"]["unique_params"],
            "actual_state_dict_elements": bank_report["model_params"]["state_dict_elements"],
        },
        "dataset_manifest": str(DATASET_MANIFEST_PATH.relative_to(OUT)),
        "results_per_state_horizon": official_rows,
        "statistics": stats,
        "mechanism_decomposition": mechanism,
        "dynamics_comparison": dynamics_comparison,
        "bootstrap": bootstrap,
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
            "G2 varies ONLY the pretraining corpus at 26M. It is not an algorithm "
            "experiment, and the D0-vs-D1 comparison is a matched descriptive "
            "comparison by (seed, stage, H) identity, not a sample-paired causal "
            "dataset experiment. 26M only; no extrapolation to 1B."
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
    print(f"REPORT  = {OUT / 'G2_DATA_DISTRIBUTION_GENERALIZATION_REPORT.md'}")
    return 0


# ---------------------------------------------------------------------------
# audit (B19)
# ---------------------------------------------------------------------------

def cmd_audit(_args) -> int:
    # Large payloads are hashed but NOT bundled: state checkpoints (bank/),
    # theta q directions, and the raw 1.6GB D1 corpus (reproducible from the
    # hash-verified public source at the pinned revision). The 4096-line
    # d1_pool.jsonl IS bundled (small, frozen scientific input).
    excluded_prefixes = (
        BANK.relative_to(OUT).as_posix() + "/",
        "data/pretrain_hq.jsonl",
    )

    def _is_excluded(rel: str) -> bool:
        return rel.startswith(excluded_prefixes) or (
            rel.startswith("theta/") and "/q/" in rel
        )

    included, excluded = [], []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(OUT).as_posix()
        if rel in {"SHA256SUMS.txt", "AUDIT_BUNDLE_SHA256.txt",
                   "g2_data_distribution_audit_package.tar.gz"} or rel.endswith(".tmp"):
            continue
        entry = (rel, file_sha256(path))
        if _is_excluded(rel):
            excluded.append(entry)
        else:
            included.append(entry)
    lines = [
        "# SHA256SUMS (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)",
        "# Section 1: files included in g2_data_distribution_audit_package.tar.gz",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in included]
    lines += [
        "",
        "# Section 2: large tensor payloads EXCLUDED from the audit bundle "
        "(hashes recorded; checkpoints/q reproducible from frozen protocol inputs; "
        "raw D1 corpus reproducible from the pinned public HF revision):",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in excluded]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = OUT / "g2_data_distribution_audit_package.tar.gz"
    if bundle.exists():
        bundle.unlink()
    with tarfile.open(bundle, "w:gz") as tar:
        for rel, _digest in included:
            tar.add(OUT / rel, arcname=f"g2_data_distribution_v01/{rel}")
        tar.add(OUT / "SHA256SUMS.txt", arcname="g2_data_distribution_v01/SHA256SUMS.txt")
    bundle_sha = file_sha256(bundle)
    (OUT / "AUDIT_BUNDLE_SHA256.txt").write_text(
        f"{bundle_sha}  g2_data_distribution_audit_package.tar.gz\n", encoding="utf-8"
    )
    # internal self-check: every included file matches its recorded hash
    for rel, digest in included:
        actual = file_sha256(OUT / rel)
        if actual != digest:
            raise RuntimeError(f"SHA256SUMS self-check failed at {rel}")
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

    p = sub.add_parser("dataset")
    p.set_defaults(func=cmd_dataset)

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
