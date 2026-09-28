"""G3 clipping-intervention analysis helpers (FR_G3_CLIPPING_INTERVENTION_V01).

Scientific goal: test whether the clipping-Jacobian mechanism identified by A0E2
has intervenable causal evidence.  G3a = operator-level control on the frozen
G1a 26M clip=1.0 bank (C with future clip-JVP replaced by identity must equal
BH_fixed).  G3b = true no-clip training intervention (26M, clip disabled),
FD/B/BH_fixed/C formal matrix, paired against the frozen G1a clip=1.0 condition.

[FR-OBSERVATION-ONLY] All frozen a0* / G1a / G1b artifacts are read-only.
Metrics are imported from the frozen a0c module -- redefining them is forbidden.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_response_propagation_v01 import _metric, relative_error
from .g1a_26m_scale_generalization_v01 import (
    E1_BASE_SEED,
    E2_BASE_SEED,
    HORIZONS_OFFICIAL,
    RHO,
    STAGES,
    STAGE_STEPS,
    TRAIN_SEEDS,
    batch_seed_from_stream_seed,
    stage_of_step,
    two_level_seed,
)

PROTOCOL = "FR_G3_CLIPPING_INTERVENTION_V01"
E2_MANIFEST_PROTOCOL = "FR_G3_NOCLIP_E2_PAIRED_STREAM"
Q_PROTOCOL = "FR_G3_NOCLIP_THETA_Q_V01"
NOCLIP_STATE_PROTOCOL = "G3_NOCLIP_STATE_BANK_V01"
E1_STREAMS = 16
E2_STREAMS = 4
H_GT_1 = (4, 8)

# Paired-stream identity with G1a: identical (base, train_seed, state_step,
# stream_id) derivation => identical future batches/RNG across conditions.
PAIRED_WITH = "outputs/generalization_g1a_26m_v01"

# Identity operator under the frozen clip framework.  The frozen primitives
# (clip_jvp and torch.nn.utils.clip_grad_norm_) implement
#   g * min(max_norm / (||g|| + 1e-6), 1.0)
# With max_norm = 1e30 the clamp is exactly 1.0 for any finite gradient norm,
# i.e. the operator is the mathematical identity (gradients bit-unchanged).
# Used ONLY for the frozen response primitives in the no-clip condition; the
# G3 training loop genuinely never calls clip_grad_norm_ (see runner train).
CLIP_DISABLED_MAX_NORM = 1e30

VERDICT_THRESHOLDS = {
    "g3a_max_abs_gap_tol": 1e-12,
    "g3a_max_rel_gap_tol": 1e-9,
    "noclip_gain_clip_shrink_factor": 0.2,
    "partial_shrink_factor": 0.5,
    "noclip_reasonable_r2_min": 0.5,
    "noclip_reasonable_mre_max": 0.5,
    "noclip_bhc_identity_rel_tol": 1e-6,
    "tie_tol": 1e-12,
}


def validate_g3_e2_manifest(
    payload: Mapping, *, expected_seed: int, expected_step: int,
    expected_stream: int, expected_content_sha: str, accumulation_steps: int,
) -> Dict:
    """Validate one registered G3 E2 stream manifest (same checks as G1a,
    G3 protocol tag)."""
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


def cell_relative_errors_bhc(rows: Sequence[Mapping]) -> Dict:
    """Per-cell absolute / relative / signed errors for B / BH_fixed / C."""
    out = {}
    for row in rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        observed = float(row["chi_fd"])
        result = {}
        for model in ("B", "BH_fixed", "C"):
            pred = float(row[f"chi_{model}"])
            rel = relative_error(observed, pred)
            result[f"abs_err_{model}"] = abs(pred - observed)
            result[f"rel_err_{model}"] = float(rel) if rel is not None else math.nan
            result[f"signed_err_{model}"] = pred - observed
        out[key] = result
    return out


def aggregate_noclip_metrics(rows: Sequence[Mapping]) -> Dict:
    """Aggregations over the 27 official no-clip cells for B / BH_fixed / C."""
    rows = [row for row in rows if int(row["H"]) in HORIZONS_OFFICIAL]
    if len(rows) != len(TRAIN_SEEDS) * len(STAGES) * len(HORIZONS_OFFICIAL):
        raise ValueError(f"official grid must have 27 rows, got {len(rows)}")
    models = ("B", "BH_fixed", "C")

    by_horizon = {
        str(h): {
            model: _metric(
                [float(r["chi_fd"]) for r in rows if int(r["H"]) == h],
                [float(r[f"chi_{model}"]) for r in rows if int(r["H"]) == h],
            )
            for model in models
        }
        for h in HORIZONS_OFFICIAL
    }
    by_seed = {}
    for seed in TRAIN_SEEDS:
        sub = [r for r in rows if int(r["seed"]) == seed]
        by_seed[str(seed)] = {
            model: _metric([float(r["chi_fd"]) for r in sub], [float(r[f"chi_{model}"]) for r in sub])
            for model in models
        }
    by_stage = {}
    for stage in STAGES:
        sub = [r for r in rows if stage_of_step(int(r["step"])) == stage]
        by_stage[stage] = {
            model: _metric([float(r["chi_fd"]) for r in sub], [float(r[f"chi_{model}"]) for r in sub])
            for model in models
        }

    errors = cell_relative_errors_bhc(rows)

    def pooled(model: str):
        out = _metric(
            [float(r["chi_fd"]) for r in rows],
            [float(r[f"chi_{model}"]) for r in rows],
        )
        rels = [
            errors[(int(r["seed"]), int(r["step"]), int(r["H"]))][f"rel_err_{model}"]
            for r in rows
        ]
        rels = [v for v in rels if not math.isnan(v)]
        if rels:
            out["median_relative_error"] = float(np.median(rels))
        return out
    paired = []
    for row in rows:
        if int(row["H"]) not in H_GT_1:
            continue
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        e = errors[key]
        if any(math.isnan(e[f"rel_err_{m}"]) for m in models):
            continue
        paired.append({
            "seed": key[0], "step": key[1], "H": key[2],
            "err_B": e["rel_err_B"], "err_BH_fixed": e["rel_err_BH_fixed"], "err_C": e["rel_err_C"],
            "gain_curvature": e["rel_err_B"] - e["rel_err_BH_fixed"],
            "gain_clip": e["rel_err_BH_fixed"] - e["rel_err_C"],
        })

    def _mean(key: str) -> float:
        vals = [p[key] for p in paired]
        return float(np.mean(vals)) if vals else math.nan

    # BH_fixed vs C collapse metric (mechanism identity under no-clip)
    bh_c_abs_gaps = []
    bh_c_rel_gaps = []
    for row in rows:
        d = abs(float(row["chi_BH_fixed"]) - float(row["chi_C"]))
        bh_c_abs_gaps.append(d)
        denom = max(abs(float(row["chi_fd"])), 1e-30)
        bh_c_rel_gaps.append(d / denom)

    return {
        "pooled": {model: pooled(model) for model in models},
        "by_horizon": by_horizon,
        "by_seed": by_seed,
        "by_stage": by_stage,
        "paired_h_gt_1": {"rows": paired, "n": len(paired)},
        "pooled_gains_h_gt_1": {
            "gain_curvature_mean": _mean("gain_curvature"),
            "gain_clip_mean": _mean("gain_clip"),
        },
        "bh_fixed_vs_c": {
            "max_abs_gap": float(max(bh_c_abs_gaps)) if bh_c_abs_gaps else math.nan,
            "mean_abs_gap": float(np.mean(bh_c_abs_gaps)) if bh_c_abs_gaps else math.nan,
            "max_rel_gap_vs_fd": float(max(bh_c_rel_gaps)) if bh_c_rel_gaps else math.nan,
            "mean_rel_gap_vs_fd": float(np.mean(bh_c_rel_gaps)) if bh_c_rel_gaps else math.nan,
        },
    }


def cluster_bootstrap_interaction(
    paired_cells: Sequence[Mapping], n_resamples: int = 20000, seed: int = 20260916,
) -> Dict:
    """Hierarchical cluster bootstrap over seeds -> stages -> stream-cells.

    Each entry of ``paired_cells`` must carry gain_clip under both conditions:
    ``gain_clip_clip`` and ``gain_clip_noclip`` plus ``seed``/``stage``.  The
    bootstrap resamples seeds with replacement, then stages within seed, then
    stream-cells within stage, and recomputes the mean interaction
    Delta_clip = mean(gain_clip_clip) - mean(gain_clip_noclip).
    """
    rng = np.random.default_rng(int(seed))
    by_seed: Dict[int, Dict[str, list]] = {}
    for cell in paired_cells:
        by_seed.setdefault(int(cell["seed"]), {}).setdefault(str(cell["stage"]), []).append(cell)
    if not by_seed:
        return {"n_cells": 0, "mean": math.nan}
    observed = float(np.mean([
        float(c["gain_clip_clip"]) - float(c["gain_clip_noclip"]) for c in paired_cells
    ]))
    seeds = sorted(by_seed)
    stats = []
    for _ in range(int(n_resamples)):
        picked = []
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            stages = by_seed[s]
            names = sorted(stages)
            for t in rng.choice(names, size=len(names), replace=True):
                cells = stages[t]
                idx = rng.integers(0, len(cells), size=len(cells))
                for i in idx:
                    picked.append(cells[int(i)])
        if not picked:
            continue
        stats.append(float(np.mean([
            float(c["gain_clip_clip"]) - float(c["gain_clip_noclip"]) for c in picked
        ])))
    arr = np.asarray(stats, dtype=np.float64)
    return {
        "n_cells": len(paired_cells),
        "n_resamples": int(n_resamples),
        "observed_mean_delta_clip": observed,
        "bootstrap": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "pi95_low": float(np.percentile(arr, 2.5)),
            "pi95_high": float(np.percentile(arr, 97.5)),
            "p_positive": float((arr > 0).mean()),
        },
    }


def decide_mechanism_verdict(
    *, g3a_pass: bool, g3a_max_abs_gap: float,
    clip_gains: Mapping, noclip_gains: Mapping,
    noclip_stats: Mapping, delta_clip: float,
    contract_mismatch: bool = False,
    resource_limit: bool = False,
) -> Dict:
    """Frozen G3 mechanism verdict (thresholds fixed in G3_PROTOCOL.md B10)."""
    th = VERDICT_THRESHOLDS
    if resource_limit:
        return {"verdict": "RESOURCE_LIMIT", "conditions": {}, "thresholds": dict(th)}
    if contract_mismatch:
        return {"verdict": "CONTRACT_MISMATCH", "conditions": {}, "thresholds": dict(th)}

    gain_clip_clip1 = float(clip_gains["gain_clip_mean"])
    gain_clip_noclip = float(noclip_gains["gain_clip_mean"])
    shrink_ratio = (
        abs(gain_clip_noclip) / abs(gain_clip_clip1)
        if abs(gain_clip_clip1) > 0 else math.inf
    )
    bh_c = noclip_stats["bh_fixed_vs_c"]
    pooled = noclip_stats["pooled"]
    cond1 = bool(g3a_pass) and abs(float(g3a_max_abs_gap)) <= th["g3a_max_abs_gap_tol"]
    cond2 = bool(shrink_ratio <= th["noclip_gain_clip_shrink_factor"])
    cond3 = bool(
        float(pooled["BH_fixed"]["R2"]) >= th["noclip_reasonable_r2_min"]
        and float(pooled["C"]["R2"]) >= th["noclip_reasonable_r2_min"]
        and float(pooled["BH_fixed"]["mean_relative_error"]) <= th["noclip_reasonable_mre_max"]
        and float(pooled["C"]["mean_relative_error"]) <= th["noclip_reasonable_mre_max"]
        and float(bh_c["max_rel_gap_vs_fd"]) <= th["noclip_bhc_identity_rel_tol"]
    )
    cond4 = bool(delta_clip > 0.0)
    if cond1 and cond2 and cond3 and cond4:
        verdict = "MECHANISM_CONFIRMED"
    elif cond1 and shrink_ratio <= th["partial_shrink_factor"]:
        verdict = "MECHANISM_PARTIAL"
    else:
        verdict = "MECHANISM_NOT_CONFIRMED"
    return {
        "verdict": verdict,
        "conditions": {
            "cond1_operator_control": {
                "pass": bool(cond1), "g3a_max_abs_gap": float(g3a_max_abs_gap),
                "tol": th["g3a_max_abs_gap_tol"],
            },
            "cond2_gain_clip_collapse_le_20pct": {
                "pass": bool(cond2),
                "gain_clip_clip1": gain_clip_clip1,
                "gain_clip_noclip": gain_clip_noclip,
                "shrink_ratio": float(shrink_ratio),
                "threshold": th["noclip_gain_clip_shrink_factor"],
            },
            "cond3_noclip_bh_c_identity_and_predictive": {
                "pass": bool(cond3),
                "pooled_R2_BH_fixed": float(pooled["BH_fixed"]["R2"]),
                "pooled_R2_C": float(pooled["C"]["R2"]),
                "pooled_mre_BH_fixed": float(pooled["BH_fixed"]["mean_relative_error"]),
                "pooled_mre_C": float(pooled["C"]["mean_relative_error"]),
                "bh_c_max_rel_gap_vs_fd": float(bh_c["max_rel_gap_vs_fd"]),
            },
            "cond4_delta_clip_positive": {
                "pass": bool(cond4), "delta_clip": float(delta_clip),
            },
        },
        "thresholds": dict(th),
    }


def json_dumps(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def render_report(context: Mapping) -> str:
    """Render G3_CLIPPING_INTERVENTION_REPORT.md from an analyze context."""
    model = context["model"]
    lines = [
        "# G3 — Clipping Intervention Report (FR_G3_CLIPPING_INTERVENTION_V01)",
        "",
        "## 1. Question and design",
        "",
        "- G3a operator control: on the frozen G1a 26M clip=1.0 bank (seed 2042 x",
        "  EARLY/MIDDLE/LATE, 4 streams/state, H={1,4,8}), C_identity (future",
        "  clip-JVP replaced by identity, everything else unchanged) must equal",
        "  BH_fixed.  Not a causal regime claim -- the baseline trajectory is",
        "  still clip=1.0-trained.",
        "- G3b true intervention: 26M retrained with clipping genuinely disabled",
        "  (clip_grad_norm_ never called), seeds 2042/2043/2044, states",
        "  26/64/102; formal FD/B/BH_fixed/C matrix paired against the frozen",
        "  G1a clip=1.0 condition (identical stream seeds).",
        "- Core prediction: gain_clip > 0 under clip=1.0; gain_clip ~ 0 under",
        "  no-clip; Delta_clip = mean(gain_clip_clip1) - mean(gain_clip_noclip) > 0.",
        "",
        "## 2. Model configuration (26M, fixed by protocol)",
        "",
        f"- hidden={model['hidden_size']}, layers={model['num_hidden_layers']}, "
        f"heads={model['num_attention_heads']}, kv={model['num_key_value_heads']}, "
        f"head_dim={model['head_dim']}, intermediate={model['intermediate_size']}",
        f"- vocab={model['vocab_size']}, seq={model['max_seq_len']}; "
        f"unique params = {model['actual_unique_params']:,}",
        "- Everything identical to G1a except clipping disabled in training.",
        "",
        "## 3. G3a operator control",
        "",
    ]
    g3a = context["g3a"]
    lines += [
        f"- pass = **{g3a['pass']}**; n stream-H cells = {g3a['n_cells']}",
        f"- max |C_identity - BH_fixed| = {g3a['max_abs_gap']:.6g}",
        f"- mean |C_identity - BH_fixed| = {g3a['mean_abs_gap']:.6g}",
        f"- max relative gap = {g3a['max_rel_gap']:.6g}",
        f"- frozen-G1a cross-check max gap = {g3a['frozen_cross_check_max_abs_gap']:.6g}",
        "",
        "## 4. G3b no-clip training",
        "",
    ]
    tr = context["training"]
    lines += [
        f"- seeds {tr['seeds']}; states {tr['states']}; clip_grad_norm_ called: "
        f"**{tr['clip_grad_norm_called']}** (manifest-recorded)",
        f"- counterfactual would-clip rate (preclip norm > 1.0): "
        f"{tr['counterfactual_would_clip_rate']:.4f}",
        f"- training preclip norm mean (no-clip) = {tr['training_preclip_norm_mean']:.4f} "
        f"vs G1a clip=1.0 = {tr['g1a_training_preclip_norm_mean']:.4f}",
        "",
        "## 5. No-clip formal results (27 official cells)",
        "",
        "| model | pooled R2 | corr | MAE | mean rel | median rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in ("B", "BH_fixed", "C"):
        s = context["noclip_stats"]["pooled"][m]
        lines.append(
            f"| {m} | {s['R2']:.10f} | {s['corr']:.10f} | {s['MAE']:.10f} | "
            f"{s['mean_relative_error']:.10f} | {s.get('median_relative_error', float('nan')):.10f} |"
        )
    bhc = context["noclip_stats"]["bh_fixed_vs_c"]
    lines += [
        "",
        f"- BH_fixed vs C collapse: max abs gap = {bhc['max_abs_gap']:.6g}, "
        f"max rel gap vs FD = {bhc['max_rel_gap_vs_fd']:.6g} (identity expected).",
        "",
        "## 6. Clip vs no-clip mechanism gains (pooled H>1)",
        "",
        "| condition | gain_curvature | gain_clip |",
        "|---|---:|---:|",
        f"| clip=1.0 (G1a frozen) | {context['gains']['clip']['gain_curvature_mean']:+.8f} | "
        f"{context['gains']['clip']['gain_clip_mean']:+.8f} |",
        f"| no-clip (G3b) | {context['gains']['noclip']['gain_curvature_mean']:+.8f} | "
        f"{context['gains']['noclip']['gain_clip_mean']:+.8f} |",
        "",
        "## 7. Delta_clip interaction",
        "",
        f"- pooled mean Delta_clip = {context['interaction']['mean_delta_clip']:+.8f}",
    ]
    bs = context["interaction"]["bootstrap"]
    lines += [
        f"- cluster bootstrap (seed>stage>stream): mean {bs['mean']:+.8f}, "
        f"95% PI [{bs['pi95_low']:+.8f}, {bs['pi95_high']:+.8f}], "
        f"P(Delta>0) = {bs['p_positive']:.4f}",
        "",
        "### per-H / per-seed / per-stage interaction",
        "",
        "| group | mean Delta_clip |",
        "|---|---:|",
    ]
    for key, val in context["interaction"]["by_group"].items():
        lines.append(f"| {key} | {val:+.8f} |")
    lines += [
        "",
        "## 8. Verdict (frozen thresholds)",
        "",
        f"**{context['verdict']['verdict']}**",
        "",
        "| condition | pass | value |",
        "|---|---:|---|",
    ]
    for name, cond in context["verdict"]["conditions"].items():
        lines.append(f"| {name} | {cond.get('pass')} | {json.dumps(cond, default=str)} |")
    lines += [
        "",
        "## 9. Dynamics context (explanation only, no post-hoc filtering)",
        "",
        "| quantity | clip=1.0 (G1a) | no-clip (G3b) |",
        "|---|---:|---:|",
    ]
    for key, pair in context["dynamics_table"].items():
        lines.append(f"| {key} | {pair[0]} | {pair[1]} |")
    lines += [
        "",
        "## 10. Anomalies / incidents",
        "",
    ]
    lines += [f"- {item}" for item in context["incidents"]]
    lines += [
        "",
        "## 11. Scope statement",
        "",
        "- G3a is an operator control, NOT causal training-regime evidence.",
        "- The causal claim rests on G3b no-clip training trajectories.",
        "- Curvature feedback is not claimed to vanish; only the clip-JVP leg is",
        "  intervened on.  No-clip does not imply C == FD (finite-epsilon FD",
        "  nonlinearity remains).",
        "- Clipping-regime changes across scales remain observations; no",
        "  empirical law is fitted and no causal attribution beyond the G3b",
        "  intervention is made.",
        "",
        "## 12. Next step",
        "",
        "G2 (26M data-distribution generalization) is a user decision; it is NOT",
        "started by this experiment.",
        "",
    ]
    return "\n".join(lines)
