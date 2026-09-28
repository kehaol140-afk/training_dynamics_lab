"""FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01 — D0 vs D1 data-distribution generalization.

科学问题：模型规模/架构/tokenizer/训练目标/AdamW/clipping/训练预算/因果响应
协议全部不变，仅更换预训练语料分布（D0 -> D1）时，完整切线 C 是否仍能准确
预测有限差分响应并在 H>1 系统性优于 optimizer-only B。

主统计口径严格复用冻结的 G1a/G1b aggregation（`cell_relative_errors` /
`aggregate_metrics` / `relative_error` / `_metric`，一律 import，禁止重定义）。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_response_propagation_v01 import _metric, relative_error
from .g1a_26m_scale_generalization_v01 import (
    HORIZONS_OFFICIAL,
    STAGES,
    TRAIN_SEEDS,
    aggregate_metrics,
    cell_relative_errors,
    json_dumps,
    stage_of_step,
)

PROTOCOL = "FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01"
STATE_BANK_PROTOCOL = "G2_26M_STATE_BANK_V01"
THETA_PROTOCOL = "FR_G2_26M_THETA_MAPPING_V01"
DATASET_MANIFEST_PROTOCOL = "FR_G2_DATASET_MANIFEST_V01"
Q_PROTOCOL = "FR_G2_26M_THETA_Q_V01"
E1_STREAM_PROTOCOL = "FR_G2_26M_E1_STREAM"
E2_MANIFEST_PROTOCOL = "FR_G2_26M_E2_PAIRED_STREAM"

# D1 pool / partition 冻结规则（在任何 G2 响应结果打开前固定）
D1_POOL_SOURCE_RECORDS = 4096          # 与 D0 (e3c_subset) 同规模 pool
G2_D1_POOL_SEED = 60202601             # pool 采样排列种子（确定性 make_permutation）
G2_PARTITION_SEED = 60202603           # train/probe/eval 划分种子
G2_PROBE_SIZE = 32                     # 与 v07/G1a 相同 probe 规模
G2_EVAL_SIZE = 128                     # 与 v07/G1a 相同 eval 规模

# B12 预注册判定阈值（结果打开前冻结，不得根据结果修改）
VERDICT_THRESHOLDS = {
    "c_pooled_r2_min": 0.90,
    "h_gt_1_win_fraction_min": 0.80,
    "h8_r2_min": 0.50,
    "seed_positive_count_min": 2,
    "tie_tol": 1e-12,
}


# ---------------------------------------------------------------------------
# B12 frozen verdict tiers
# ---------------------------------------------------------------------------

def decide_data_verdict(
    *,
    stats: Mapping,
    resource_limit: bool = False,
    dataset_unavailable: bool = False,
    contract_incompatible: bool = False,
) -> Dict:
    """Frozen G2 verdict tiers (B12).  Thresholds fixed above; never tuned."""
    th = VERDICT_THRESHOLDS
    if contract_incompatible:
        return {"verdict": "CONTRACT_INCOMPATIBLE", "conditions": {}, "pass": None}
    if dataset_unavailable:
        return {"verdict": "DATASET_UNAVAILABLE", "conditions": {}, "pass": None}
    if resource_limit:
        return {"verdict": "RESOURCE_LIMIT", "conditions": {}, "pass": None}

    cond1 = float(stats["pooled"]["C"]["R2"]) >= th["c_pooled_r2_min"]
    paired = stats["paired_h_gt_1"]
    cond2 = paired["n"] > 0 and (paired["wins"] / paired["n"]) >= th["h_gt_1_win_fraction_min"]
    h8 = stats["by_horizon"].get("8", {}).get("C", {})
    h8_rows = [p for p in paired["rows"] if int(p["H"]) == 8]
    cond3 = (
        len(h8_rows) > 0
        and float(np.mean([p["err_C"] for p in h8_rows]))
        < float(np.mean([p["err_B"] for p in h8_rows]))
        and h8.get("R2") is not None
        and float(h8["R2"]) >= th["h8_r2_min"]
        and h8.get("corr") is not None
        and float(h8["corr"]) > 0.0
    )
    seed_pos = sum(1 for v in stats["seed_mean_paired_improvement"].values() if v > 0.0)
    cond4 = seed_pos >= th["seed_positive_count_min"]
    c_better_overall = (
        stats["relative_error_summary"]["C_mean_h_gt_1"]
        < stats["relative_error_summary"]["B_mean_h_gt_1"]
    )
    if cond1 and cond2 and cond3 and cond4:
        verdict = "STRONG_DATA_REPLICATION"
    elif c_better_overall:
        verdict = "PARTIAL_DATA_REPLICATION"
    else:
        verdict = "NO_DATA_REPLICATION"
    return {
        "verdict": verdict,
        "pass": bool(cond1 and cond2 and cond3 and cond4),
        "conditions": {
            "cond1_c_pooled_r2_ge_0.90": {"pass": bool(cond1), "value": float(stats["pooled"]["C"]["R2"])},
            "cond2_h_gt_1_win_fraction_ge_0.80": {
                "pass": bool(cond2), "wins": int(paired["wins"]), "n": int(paired["n"]),
            },
            "cond3_h8_paired_and_predictive": {
                "pass": bool(cond3),
                "mean_err_C_h8": float(np.mean([p["err_C"] for p in h8_rows])),
                "mean_err_B_h8": float(np.mean([p["err_B"] for p in h8_rows])),
                "R2_C_h8": h8.get("R2"),
                "corr_C_h8": h8.get("corr"),
            },
            "cond4_seed_positive_ge_2": {
                "pass": bool(cond4),
                "seed_means": {k: float(v) for k, v in stats["seed_mean_paired_improvement"].items()},
            },
        },
        "c_better_overall": bool(c_better_overall),
        "thresholds": dict(th),
    }


# ---------------------------------------------------------------------------
# B15 cluster-aware bootstrap for paired B/C improvement (D1, H>1)
# ---------------------------------------------------------------------------

def cluster_bootstrap_paired_improvement(
    paired_cells: Sequence[Mapping],
    n_resamples: int = 20000,
    seed: int = 20260916,
) -> Dict:
    """Hierarchical cluster bootstrap over seeds -> stages -> official cells.

    Each entry must carry ``err_B`` and ``err_C`` (cell-level relative
    errors) plus ``seed``/``stage``.  The estimand is the equal-weight mean
    paired improvement  mean(err_B - err_C)  over the H>1 official cells;
    every resample recomputes exactly this estimand on the resampled
    clusters.  Resampling: seeds with replacement, then stages within seed,
    then cells within stage.
    """
    rng = np.random.default_rng(int(seed))
    by_seed: Dict[int, Dict[str, list]] = {}
    for cell in paired_cells:
        by_seed.setdefault(int(cell["seed"]), {}).setdefault(str(cell["stage"]), []).append(cell)
    if not by_seed:
        return {"n_cells": 0, "mean": math.nan}
    observed = float(np.mean([
        float(c["err_B"]) - float(c["err_C"]) for c in paired_cells
    ]))
    seeds = sorted(by_seed)
    stats = []
    for _ in range(int(n_resamples)):
        picked = []
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            stages_d = by_seed[s]
            names = sorted(stages_d)
            for t in rng.choice(names, size=len(names), replace=True):
                cells = stages_d[t]
                idx = rng.integers(0, len(cells), size=len(cells))
                for i in idx:
                    picked.append(cells[int(i)])
        if not picked:
            continue
        stats.append(float(np.mean([
            float(c["err_B"]) - float(c["err_C"]) for c in picked
        ])))
    arr = np.asarray(stats, dtype=np.float64)
    return {
        "n_cells": len(paired_cells),
        "n_resamples": int(n_resamples),
        "rng_seed": int(seed),
        "observed_mean_improvement": observed,
        "bootstrap": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "pi95_low": float(np.percentile(arr, 2.5)),
            "pi95_high": float(np.percentile(arr, 97.5)),
            "fraction_resamples_improvement_positive": float((arr > 0).mean()),
        },
        "interpretation_note": (
            "fraction_resamples_improvement_positive is the fraction of bootstrap "
            "resamples in which the paired improvement is positive; it is NOT a "
            "p-value, a posterior probability, or a significance probability. "
            "With only 3 train seeds the interval is exploratory uncertainty "
            "under a small number of independent training runs; stream/cell "
            "counts must not be mistaken for independent runs."
        ),
    }


# ---------------------------------------------------------------------------
# dataset helpers (pure; used by runner cmd_dataset and tests)
# ---------------------------------------------------------------------------

def validate_g2_e2_manifest(
    payload: Mapping, *, expected_seed: int, expected_step: int,
    expected_stream: int, expected_content_sha: str, accumulation_steps: int,
) -> Dict:
    """Validate one registered G2 E2 stream manifest.

    Mirrors the frozen ``validate_g1a_e2_manifest`` checks exactly (manifest
    version 0.2, content sha256, identity fields, accumulation window), with
    the G2 protocol tag.  Reusing the frozen validator with the G1a tag would
    mislabel G2 manifests, so this twin exists; no check is loosened.
    """
    from .batch_manifest import manifest_sha256

    if payload.get("manifest_version") != "0.2":
        raise ValueError(f"unexpected manifest version: {payload.get('manifest_version')!r}")
    metadata = payload.get("metadata")
    batches = payload.get("batches")
    if not isinstance(metadata, Mapping) or not isinstance(batches, list):
        raise ValueError("manifest metadata/batches are missing")
    if metadata.get("protocol") != E2_MANIFEST_PROTOCOL:
        raise ValueError(f"unexpected manifest protocol: {metadata.get('protocol')!r}")
    actual = {
        "train_seed": int(metadata.get("train_seed", -1)),
        "state_step": int(metadata.get("state_step", -1)),
        "stream_id": int(metadata.get("stream_id", -1)),
    }
    expected = {
        "train_seed": int(expected_seed),
        "state_step": int(expected_step),
        "stream_id": int(expected_stream),
    }
    if actual != expected:
        raise ValueError(f"manifest identity mismatch: {actual} vs {expected}")
    content_sha = manifest_sha256(batches)
    if content_sha != str(payload.get("sha256")) or content_sha != str(expected_content_sha):
        raise ValueError("manifest batch-content sha256 mismatch")
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0 or len(batches) < accumulation_steps:
        raise ValueError("manifest does not contain the complete accumulation window")
    for batch in batches[: max(accumulation_steps, max(HORIZONS_OFFICIAL) * accumulation_steps)]:
        if not isinstance(batch, list) or not batch or not all(isinstance(x, int) for x in batch):
            raise ValueError("manifest contains an invalid batch")
    for field in ("stream_seed", "batch_seed", "epsilon", "rho"):
        if field not in metadata:
            raise ValueError(f"manifest metadata missing {field}")
    return {
        "batches": [list(batch) for batch in batches],
        "metadata": dict(metadata),
        "content_sha256": content_sha,
    }


def select_pool_indices(source_n_records: int, pool_seed: int, pool_size: int) -> list:
    """Deterministic D1 pool selection: first ``pool_size`` entries of a
    frozen permutation of the source corpus indices, sorted ascending."""
    from .batch_manifest import make_permutation

    if source_n_records < pool_size:
        raise ValueError("source corpus smaller than requested pool")
    perm = make_permutation(int(source_n_records), int(pool_seed))
    return sorted(int(i) for i in perm[: int(pool_size)])


def shingle_digest(text: str, n: int = 13, mod: int = 1 << 24) -> set:
    """Char-level n-gram shingle hash set (cheap overlap diagnostic)."""
    t = "".join(str(text).split())
    if len(t) < n:
        return {hashlib.md5(t.encode("utf-8")).hexdigest()}
    return {
        hashlib.md5(t[i:i + n].encode("utf-8")).hexdigest()
        for i in range(0, len(t) - n + 1, 3)
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def validate_dataset_manifest(payload: Mapping) -> None:
    """Structural validation of dataset_manifest.json (frozen schema)."""
    required = [
        "protocol", "d0", "d1", "d1_pool", "partition", "overlap_diagnostic",
        "token_statistics_sample", "created_at_utc",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"dataset_manifest missing {key}")
    if payload["protocol"] != DATASET_MANIFEST_PROTOCOL:
        raise ValueError("dataset_manifest protocol mismatch")
    for block in ("d0", "d1", "d1_pool"):
        for field in ("path", "sha256", "records", "bytes"):
            if field not in payload[block]:
                raise ValueError(f"dataset_manifest.{block} missing {field}")
    if int(payload["d1_pool"]["records"]) != D1_POOL_SOURCE_RECORDS:
        raise ValueError("d1_pool record count mismatch")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_report(context: Mapping) -> str:
    c = context
    lines: list[str] = []
    a = lines.append
    a("# G2 — Data Distribution Generalization Report (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)")
    a("")
    a("## 1. Question and design")
    a("")
    a("- Sole question: with model scale / architecture / tokenizer / objective /")
    a("  AdamW / clipping / token budget / response protocol held fixed, does the")
    a("  full tangent C still predict the finite-difference response and beat")
    a("  optimizer-only B at H>1 when only the pretraining corpus changes (D0 -> D1)?")
    a(f"- D0 = frozen G1a pool `{c['d0_path']}` (read directly; never rerun).")
    a(f"- D1 = `{c['d1_path']}` selected and frozen before any G2 response result;")
    a("  pool/partition rules fixed in G2_PROTOCOL.md (B2/B3/B4).")
    a("- This is NOT an algorithm-generalization experiment; D0 vs D1 cells are a")
    a("  matched descriptive comparison by (seed, stage, H) identity, NOT a")
    a("  sample-paired causal dataset experiment (training data differed).")
    a("")
    a("## 2. Model / training configuration (fixed, identical to G1a)")
    a("")
    m = c["model"]
    a(f"- hidden={m['hidden_size']} layers={m['num_hidden_layers']} heads={m['num_attention_heads']} "
      f"kv={m['num_key_value_heads']} head_dim={m['head_dim']} intermediate={m['intermediate_size']} "
      f"vocab={m['vocab_size']} seq={m['max_seq_len']}")
    a(f"- unique params = {c['actual_unique_params']} (must equal 26,681,728)")
    a("- AdamW betas=(0.9,0.999) eps=1e-8 wd=0.01; clip=1.0; bf16 GradScaler disabled;")
    a("  global batch 8 seqs x 128 tokens = 1024 tokens/optimizer-step; cosine LR;")
    a("  128 planned optimizer steps; states at steps 26/64/102 (~20.3/50/79.7%).")
    a("")
    a("## 3. Dataset provenance summary")
    a("")
    d = c["dataset_summary"]
    a(f"- D0: {d['d0_records']} records, sha256 `{d['d0_sha256'][:16]}...`")
    a(f"- D1: {d['d1_records']} records, sha256 `{d['d1_sha256'][:16]}...`")
    a(f"- D1 pool: {d['d1_pool_records']} records (same size as D0 pool), sha256 `{d['d1_pool_sha256'][:16]}...`")
    a(f"- overlap diagnostic: exact duplicate lines = {d['exact_dup_lines']}; "
      f"sampled 13-gram shingle Jaccard = {d['shingle_jaccard_sample']:.4f}")
    a("")
    a("## 4. D1 official results (27 cells: 3 seeds x 3 stages x H={1,4,8})")
    a("")
    a("| model | pooled R2 | corr | MAE | mean rel |")
    a("|---|---:|---:|---:|---:|")
    s = c["stats"]
    for model in ("B", "C"):
        st = s["pooled"][model]
        a(f"| {model} | {st['R2']:.10f} | {st['corr']:.10f} | {st['MAE']:.10f} | {st['mean_relative_error']:.10f} |")
    a("")
    a("### per-horizon C")
    a("")
    a("| H | C R2 | C mean rel | B mean rel |")
    a("|---|---:|---:|---:|")
    for h in HORIZONS_OFFICIAL:
        ch = s["by_horizon"][str(h)]
        a(f"| {h} | {ch['C']['R2']:.10f} | {ch['C']['mean_relative_error']:.10f} | {ch['B']['mean_relative_error']:.10f} |")
    a("")
    a("## 5. Verdict (frozen B12 thresholds)")
    a("")
    v = c["verdict"]
    a(f"**{v['verdict']}**")
    a("")
    a("| condition | pass | value |")
    a("|---|---:|---|")
    for name, payload in v["conditions"].items():
        a(f"| {name} | {payload['pass']} | {json.dumps(payload.get('value', payload), default=str)[:120]} |")
    a("")
    a("## 6. D1 mechanism subset (seed 2042; B -> BH_fixed -> C, FD reference)")
    a("")
    mech = c["mechanism"]
    a("### 6a. stream-first aggregation (identical caliber to G1a report §7)")
    a("")
    a(f"- gain_curvature = {mech['stream_first']['gain_curvature_mean']:.10f}")
    a(f"- gain_clip      = {mech['stream_first']['gain_clip_mean']:.10f}")
    a("")
    a("### 6b. official-cell diagnostic (same aggregation as §4)")
    a("")
    a(f"- gain_curvature = {mech['official_cell']['gain_curvature_mean']:.10f}")
    a(f"- gain_clip      = {mech['official_cell']['gain_clip_mean']:.10f}")
    a("- The two calibers differ only in the order of stream-mean and relative-")
    a("  error; both are reported and never mixed.")
    a("")
    a("## 7. D0 vs D1 matched descriptive comparison (seed/stage/H identity)")
    a("")
    cmp_ = c["dynamics_comparison"]
    a("| quantity | D0 (G1a frozen) | D1 (G2) |")
    a("|---|---:|---:|")
    for key, label in (
        ("c_pooled_r2", "C pooled R2"),
        ("c_pooled_mre", "C pooled mean rel err"),
        ("b_pooled_mre", "B pooled mean rel err"),
        ("win_fraction_h_gt_1", "C<B win fraction (H>1)"),
        ("mean_improvement_h_gt_1", "mean(err_B - err_C) H>1"),
        ("training_loss_mean", "training loss mean"),
        ("training_preclip_norm_mean", "training preclip norm mean"),
        ("training_clip_rate", "training clip rate"),
        ("future_first_clip_rate", "future first-step clip rate"),
        ("future_grad_norm_mean", "future grad norm mean"),
        ("hvp_norm_mean", "future HVP norm mean"),
        ("gain_curvature_stream_first", "mech gain_curvature (stream-first)"),
        ("gain_clip_stream_first", "mech gain_clip (stream-first)"),
    ):
        d0v = cmp_["d0"].get(key)
        d1v = cmp_["d1"].get(key)
        fmt = lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)
        a(f"| {label} | {fmt(d0v)} | {fmt(d1v)} |")
    a("")
    a(f"Full per-cell table: `{c['d0_vs_d1_csv']}`")
    a("")
    a("## 8. Bootstrap uncertainty (B15; D1 H>1 paired B/C improvement)")
    a("")
    b = c["bootstrap"]
    a(f"- observed mean paired improvement = {b['observed_mean_improvement']:.10f}")
    bb = b["bootstrap"]
    a(f"- cluster bootstrap (seed>stage>cell, {b['n_resamples']} resamples, rng {b['rng_seed']}): "
      f"mean {bb['mean']:.10f}, median {bb['median']:.10f}, "
      f"95% PI [{bb['pi95_low']:.10f}, {bb['pi95_high']:.10f}]")
    a(f"- fraction of bootstrap resamples with improvement > 0 = "
      f"{bb['fraction_resamples_improvement_positive']:.5f} "
      f"(NOT a p-value / posterior / significance probability)")
    a("")
    a("## 9. Dynamics / distribution-shift diagnostics (B13; explanation only)")
    a("")
    for line in c["shift_notes"]:
        a(f"- {line}")
    a("")
    a("## 10. Anomalies / incidents")
    a("")
    for line in c["incidents"]:
        a(f"- {line}")
    a("")
    a("## 11. Scope statement")
    a("")
    a("- G2 varies ONLY the pretraining corpus; it is not an algorithm experiment.")
    a("- D0/D1 comparison is matched descriptive by construction, not a")
    a("  sample-level paired causal dataset experiment.")
    a("- Mixed-backend HVP caveat inherited unchanged (MATH SDPA surrogate).")
    a("- 26M only; no extrapolation to 1B.")
    a("")
    a("## 12. Next step")
    a("")
    a(c["next_step"])
    a("")
    return "\n".join(lines)


__all__ = [
    "PROTOCOL", "STATE_BANK_PROTOCOL", "THETA_PROTOCOL",
    "DATASET_MANIFEST_PROTOCOL", "Q_PROTOCOL", "E1_STREAM_PROTOCOL",
    "E2_MANIFEST_PROTOCOL", "D1_POOL_SOURCE_RECORDS",
    "G2_D1_POOL_SEED", "G2_PARTITION_SEED", "G2_PROBE_SIZE", "G2_EVAL_SIZE",
    "VERDICT_THRESHOLDS",
    "decide_data_verdict", "cluster_bootstrap_paired_improvement",
    "select_pool_indices", "shingle_digest", "jaccard", "validate_dataset_manifest",
    "validate_g2_e2_manifest",
    "render_report",
    "aggregate_metrics", "cell_relative_errors", "relative_error", "_metric",
    "stage_of_step", "json_dumps",
    "HORIZONS_OFFICIAL", "STAGES", "TRAIN_SEEDS",
]
