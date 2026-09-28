"""G1a 26M scale-generalization analysis helpers (FR_G1A_26M_SCALE_GENERALIZATION_V01).

[FR-OBSERVATION-ONLY] All v07 bank artifacts, frozen a0* modules and their outputs
are read-only inputs. Metrics are imported from the frozen a0c module — redefining
them is forbidden by project protocol.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_response_propagation_v01 import _metric, relative_error
from .a0c_h1_adamw_jacobian_v01 import file_sha256
from .batch_manifest import manifest_sha256
from .stream_rng import derive_stream_seed

PROTOCOL = "FR_G1A_26M_SCALE_GENERALIZATION_V01"
E2_MANIFEST_PROTOCOL = "FR_G1A_26M_E2_PAIRED_STREAM"
Q_PROTOCOL = "FR_G1A_26M_THETA_Q_V01"

TRAIN_SEEDS = (2042, 2043, 2044)
STAGES = ("EARLY", "MIDDLE", "LATE")
# frozen in research/configs/g1a_26m_v01.json (state_step_rule = round(fraction*128))
STAGE_STEPS = {"EARLY": 26, "MIDDLE": 64, "LATE": 102}
STAGE_FRACTIONS = {"EARLY": 26 / 128, "MIDDLE": 64 / 128, "LATE": 102 / 128}
STAGE_TOKENS = {"EARLY": 26624, "MIDDLE": 65536, "LATE": 104448}
HORIZONS_OFFICIAL = (1, 4, 8)
H_GT_1 = (4, 8)
TOTAL_OPT_STEPS = 128
TOKENS_PER_STEP = 1024
E1_STREAMS = 16
E2_STREAMS = 4
RHO = 0.25
E1_BASE_SEED = 408202601
E2_BASE_SEED = 409202601
BATCH_SEED_SALT = 404
TIE_TOL = 1e-12

VERDICT_THRESHOLDS = {
    "c_pooled_r2_min": 0.90,
    "h_gt_1_win_fraction_min": 0.80,
    "h8_r2_min": 0.50,
    "seed_positive_count_min": 2,
    "tie_tol": TIE_TOL,
}


def two_level_seed(base: int, train_seed: int, state_step: int, stream_id: int) -> int:
    """Identical derivation rule to v07 E3C: fold train_seed/step/stream_id into base."""
    x = derive_stream_seed(int(base), int(train_seed))
    x = derive_stream_seed(x, int(state_step))
    return derive_stream_seed(x, int(stream_id))


def batch_seed_from_stream_seed(stream_seed: int) -> int:
    return derive_stream_seed(int(stream_seed), BATCH_SEED_SALT)


def stage_of_step(step: int) -> str:
    for stage, value in STAGE_STEPS.items():
        if int(step) == value:
            return stage
    raise ValueError(f"unregistered state step: {step}")


def validate_g1a_e2_manifest(
    payload: Mapping, *, expected_seed: int, expected_step: int,
    expected_stream: int, expected_content_sha: str, accumulation_steps: int,
) -> Dict:
    """Validate one registered G1a E2 stream manifest (modeled on the frozen v07 validator).

    The frozen ``validate_e2_manifest`` in a0c_h1 is hard-tagged to the v07 protocol
    string and is therefore not reusable for G1a manifests; this validator enforces
    the same checks with the G1a protocol tag.
    """
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


def cell_relative_errors(rows: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
    """Per-cell absolute / relative / signed errors for B and C against chi_FD."""
    out = {}
    for row in rows:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        observed = float(row["chi_fd"])
        result = {}
        for model in ("B", "C"):
            pred = float(row[f"chi_{model}"])
            rel = relative_error(observed, pred)
            result[f"abs_err_{model}"] = abs(pred - observed)
            result[f"rel_err_{model}"] = float(rel) if rel is not None else math.nan
            result[f"signed_err_{model}"] = pred - observed
        out[key] = result
    return out


def aggregate_metrics(rows: Sequence[Mapping]) -> Dict:
    """A/B/C/D/E aggregations over official-grid rows using the frozen _metric."""
    rows = [row for row in rows if int(row["H"]) in HORIZONS_OFFICIAL]
    if len(rows) != len(TRAIN_SEEDS) * len(STAGES) * len(HORIZONS_OFFICIAL):
        raise ValueError(f"official grid must have 27 rows, got {len(rows)}")

    def pooled(model: str):
        return _metric(
            [float(r["chi_fd"]) for r in rows],
            [float(r[f"chi_{model}"]) for r in rows],
        )

    by_horizon = {
        str(h): {
            model: _metric(
                [float(r["chi_fd"]) for r in rows if int(r["H"]) == h],
                [float(r[f"chi_{model}"]) for r in rows if int(r["H"]) == h],
            )
            for model in ("B", "C")
        }
        for h in HORIZONS_OFFICIAL
    }
    by_seed = {}
    for seed in TRAIN_SEEDS:
        sub = [r for r in rows if int(r["seed"]) == seed]
        by_seed[str(seed)] = {
            model: _metric([float(r["chi_fd"]) for r in sub], [float(r[f"chi_{model}"]) for r in sub])
            for model in ("B", "C")
        }
    by_stage = {}
    for stage in STAGES:
        sub = [r for r in rows if stage_of_step(int(r["step"])) == stage]
        by_stage[stage] = {
            model: _metric([float(r["chi_fd"]) for r in sub], [float(r[f"chi_{model}"]) for r in sub])
            for model in ("B", "C")
        }

    errors = cell_relative_errors(rows)
    paired = []
    for row in rows:
        if int(row["H"]) not in H_GT_1:
            continue
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        err_b = errors[key]["rel_err_B"]
        err_c = errors[key]["rel_err_C"]
        if math.isnan(err_b) or math.isnan(err_c):
            continue
        diff = err_b - err_c
        if abs(diff) <= TIE_TOL:
            outcome = "tie"
        elif diff > 0:
            outcome = "C_better"
        else:
            outcome = "C_worse"
        paired.append({
            "seed": key[0], "step": key[1], "H": key[2],
            "err_B": err_b, "err_C": err_c,
            "paired_improvement": diff, "outcome": outcome,
        })
    wins = sum(1 for p in paired if p["outcome"] == "C_better")
    ties = sum(1 for p in paired if p["outcome"] == "tie")
    losses = sum(1 for p in paired if p["outcome"] == "C_worse")
    improvements = np.asarray([p["paired_improvement"] for p in paired], dtype=np.float64)

    seed_means = {}
    for seed in TRAIN_SEEDS:
        vals = [p["paired_improvement"] for p in paired if p["seed"] == seed]
        seed_means[str(seed)] = float(np.mean(vals)) if vals else math.nan

    rel_b = np.asarray([e["rel_err_B"] for e in errors.values() if not math.isnan(e["rel_err_B"])], dtype=np.float64)
    rel_c = np.asarray([e["rel_err_C"] for e in errors.values() if not math.isnan(e["rel_err_C"])], dtype=np.float64)
    rel_b_hgt1 = np.asarray([p["err_B"] for p in paired], dtype=np.float64)
    rel_c_hgt1 = np.asarray([p["err_C"] for p in paired], dtype=np.float64)

    return {
        "pooled": {model: pooled(model) for model in ("B", "C")},
        "by_horizon": by_horizon,
        "by_seed": by_seed,
        "by_stage": by_stage,
        "paired_h_gt_1": {
            "rows": paired,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n": len(paired),
            "win_fraction": float(wins / len(paired)) if paired else math.nan,
            "mean_paired_improvement": float(improvements.mean()) if improvements.size else math.nan,
            "median_paired_improvement": float(np.median(improvements)) if improvements.size else math.nan,
        },
        "relative_error_summary": {
            "B_mean": float(rel_b.mean()), "B_median": float(np.median(rel_b)),
            "C_mean": float(rel_c.mean()), "C_median": float(np.median(rel_c)),
            "B_mean_h_gt_1": float(rel_b_hgt1.mean()), "C_mean_h_gt_1": float(rel_c_hgt1.mean()),
            "B_median_h_gt_1": float(np.median(rel_b_hgt1)), "C_median_h_gt_1": float(np.median(rel_c_hgt1)),
        },
        "seed_mean_paired_improvement": seed_means,
    }


def decide_verdict(stats: Mapping) -> Dict:
    """Frozen three-tier verdict (thresholds fixed in G1A_PROTOCOL.md §10)."""
    th = VERDICT_THRESHOLDS
    c_pooled_r2 = stats["pooled"]["C"]["R2"]
    paired = stats["paired_h_gt_1"]
    wins = paired["wins"]
    n = paired["n"]
    cond1 = c_pooled_r2 is not None and float(c_pooled_r2) >= th["c_pooled_r2_min"]
    cond2 = n > 0 and (wins / n) >= th["h_gt_1_win_fraction_min"]
    h8 = stats["by_horizon"].get("8", {}).get("C", {})
    h8b = stats["by_horizon"].get("8", {}).get("B", {})
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
        verdict = "STRONG_SCALE_REPLICATION"
    elif c_better_overall:
        verdict = "PARTIAL_SCALE_REPLICATION"
    else:
        verdict = "NO_SCALE_REPLICATION"
    return {
        "verdict": verdict,
        "conditions": {
            "cond1_c_pooled_r2_ge_0.90": {"pass": bool(cond1), "value": c_pooled_r2},
            "cond2_h_gt_1_win_fraction_ge_0.80": {
                "pass": bool(cond2), "value": paired["win_fraction"],
                "wins": wins, "n": n,
            },
            "cond3_h8_paired_and_predictive": {
                "pass": bool(cond3),
                "mean_err_C_h8": float(np.mean([p["err_C"] for p in h8_rows])) if h8_rows else math.nan,
                "mean_err_B_h8": float(np.mean([p["err_B"] for p in h8_rows])) if h8_rows else math.nan,
                "R2_C_h8": h8.get("R2"), "corr_C_h8": h8.get("corr"),
            },
            "cond4_seed_positive_ge_2": {
                "pass": bool(cond4),
                "seed_means": stats["seed_mean_paired_improvement"],
            },
        },
        "c_better_overall_h_gt_1_mean_rel": bool(c_better_overall),
        "thresholds": dict(th),
    }


def _finite_or_nan(value):
    return value if (value is None or (isinstance(value, float) and math.isnan(value))) else float(value)


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.10g}"


def render_report(context: Mapping) -> str:
    """Render G1A_26M_SCALE_GENERALIZATION_REPORT.md from an analyze context dict."""
    stats = context["stats"]
    verdict = context["verdict"]
    paired = stats["paired_h_gt_1"]
    rel = stats["relative_error_summary"]
    lines = [
        "# G1a — 26M Scale Generalization Report (FR_G1A_26M_SCALE_GENERALIZATION_V01)",
        "",
        "## 1. Model configuration (scale-only)",
        "",
        f"- Codebase: `{context['model']['codebase']}` (frozen v07 generation, unmodified).",
        f"- hidden={context['model']['hidden_size']}, layers={context['model']['num_hidden_layers']}, "
        f"heads={context['model']['num_attention_heads']} (kv={context['model']['num_key_value_heads']}, "
        f"head_dim={context['model']['head_dim']}), intermediate={context['model']['intermediate_size']} "
        f"(frozen default formula), vocab={context['model']['vocab_size']}, seq_len={context['model']['max_seq_len']}.",
        f"- Unique parameters: **{context['model']['actual_unique_params']:,}** "
        f"(anchor 4,983,168; ratio {context['model']['actual_unique_params'] / 4983168:.3f}x). "
        "Checkpoint tensor elements: "
        f"{context['model']['actual_state_dict_elements']:,}.",
        "- Tokenizer / position encoding / normalization / activation / attention implementation / loss: unchanged.",
        "",
        "## 2. Training",
        "",
        f"- Seeds: {', '.join(map(str, TRAIN_SEEDS))} (preregistered; no identity conflict).",
        f"- Schedule: 128 planned optimizer steps (256 micro), microbatch 4 x accum 2 = 1024 tokens/step, "
        f"lr 5e-4 cosine, AdamW(0.9/0.999, eps 1e-8, wd 0.01), clip 1.0, bf16 autocast, GradScaler disabled.",
        "- Stages (normalized progress, not 46/64/82): "
        + "; ".join(
            f"{stage}=step {STAGE_STEPS[stage]} ({STAGE_FRACTIONS[stage] * 100:.4g}%, {STAGE_TOKENS[stage]} tokens)"
            for stage in STAGES
        ) + ".",
        f"- Training terminated at LATE step {STAGE_STEPS['LATE']} (no later registered state).",
        "",
        "## 3. Official matrix",
        "",
        f"- {len(TRAIN_SEEDS)} seeds x {len(STAGES)} stages = 9 states; 4 future streams per state = 36 baseline streams; "
        f"H = {list(HORIZONS_OFFICIAL)} -> 27 official state-horizon cells.",
        "- All 36 stream manifests were generated and frozen before any E2 response branch executed.",
        f"- Epsilon rule: rho={RHO} x sigma_q(16 E1 streams), per state: "
        + ", ".join(
            f"seed{seed}/step{step}={context['epsilon_by_state'][f'{seed}:{step}']:.6g}"
            for seed in TRAIN_SEEDS for step in STAGE_STEPS.values()
        ) + ".",
        "",
        "## 4. FD / B / C pooled results (27 official rows)",
        "",
        "| model | R2 | corr | MAE | RMSE | mean rel err | median rel err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B", "C"):
        m = stats["pooled"][model]
        med = rel[f"{model}_median"]
        lines.append(
            f"| {model} | {_fmt(m['R2'])} | {_fmt(m['corr'])} | {_fmt(m['MAE'])} | "
            f"{_fmt(m['RMSE'])} | {_fmt(rel[model + '_mean'])} | {_fmt(med)} |"
        )
    lines += [
        "",
        "## 5. By horizon",
        "",
        "| H | R2 B | R2 C | corr B | corr C | MAE B | MAE C | mean rel B | mean rel C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    errors_by_h: Dict[int, Dict[str, list]] = {
        h: {"B": [], "C": []} for h in HORIZONS_OFFICIAL
    }
    for key, errs in context["cell_errors"].items():
        h = key[2]
        if h in errors_by_h and not math.isnan(errs["rel_err_B"]):
            errors_by_h[h]["B"].append(errs["rel_err_B"])
            errors_by_h[h]["C"].append(errs["rel_err_C"])
    for h in HORIZONS_OFFICIAL:
        b = stats["by_horizon"][str(h)]["B"]
        c = stats["by_horizon"][str(h)]["C"]
        mb = float(np.mean(errors_by_h[h]["B"])) if errors_by_h[h]["B"] else math.nan
        mc = float(np.mean(errors_by_h[h]["C"])) if errors_by_h[h]["C"] else math.nan
        lines.append(
            f"| {h} | {_fmt(b['R2'])} | {_fmt(c['R2'])} | {_fmt(b['corr'])} | {_fmt(c['corr'])} | "
            f"{_fmt(b['MAE'])} | {_fmt(c['MAE'])} | {_fmt(mb)} | {_fmt(mc)} |"
        )
    lines += [
        "",
        "## 6. B vs C (H>1 paired, 18 rows)",
        "",
        f"- C beats B: **{paired['wins']}**, ties: **{paired['ties']}**, C worse than B: **{paired['losses']}** "
        f"(win fraction {paired['win_fraction']:.4g}).",
        f"- Mean paired improvement err_B - err_C: **{paired['mean_paired_improvement']:+.6g}** "
        f"(median {paired['median_paired_improvement']:+.6g}).",
        "- Per-seed mean paired improvement (H>1 rows): "
        + ", ".join(f"seed {k}: {v:+.6g}" for k, v in stats["seed_mean_paired_improvement"].items()) + ".",
        "- Per-stage mean paired improvement (H>1 rows): "
        + ", ".join(
            f"{stage}: {float(np.mean([p['paired_improvement'] for p in paired['rows'] if stage_of_step(p['step']) == stage])):+.6g}"
            for stage in STAGES
        ) + ".",
        "",
        "### Per-cell table (27 official rows)",
        "",
        "| seed | stage | step | H | chi_FD | chi_B | chi_C | rel err B | rel err C |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in context["official_rows"]:
        key = (int(row["seed"]), int(row["step"]), int(row["H"]))
        errs = context["cell_errors"][key]
        lines.append(
            f"| {row['seed']} | {stage_of_step(int(row['step']))} | {row['step']} | {row['H']} | "
            f"{_fmt(row['chi_fd'])} | {_fmt(row['chi_B'])} | {_fmt(row['chi_C'])} | "
            f"{_fmt(errs['rel_err_B'])} | {_fmt(errs['rel_err_C'])} |"
        )
    mech = context.get("mechanism")
    lines += [
        "",
        "## 7. Mechanism subset (seed 2042, B -> BH_fixed -> C, FD reference)",
        "",
    ]
    if mech is None:
        lines.append("Mechanism subset: NOT_RUN.")
    else:
        lines += [
            "| stage | H | mean rel err B | mean rel err BH_fixed | mean rel err C | gain_curvature (B-BHf) | gain_clip (BHf-C) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for stage in STAGES:
            block = mech["by_stage"][stage]
            for h in HORIZONS_OFFICIAL:
                row = block[str(h)]
                lines.append(
                    f"| {stage} | {h} | {_fmt(row['mean_rel_err_B'])} | {_fmt(row['mean_rel_err_BH_fixed'])} | "
                    f"{_fmt(row['mean_rel_err_C'])} | {_fmt(row['gain_curvature'])} | {_fmt(row['gain_clip'])} |"
                )
        lines += [
            "",
            f"- H=1 structural identity max |BH_fixed - B| = {mech['h1_identity_max_abs_gap']:.6g} "
            f"(requirement: B = BH_fixed = C at H=1).",
            "- Pooled over 3 stages x H>1: gain_curvature = "
            + (
                f"{mech['pooled_h_gt_1']['gain_curvature_mean']:+.6g} (mean), gain_clip = "
                f"{mech['pooled_h_gt_1']['gain_clip_mean']:+.6g} (mean)."
                if mech['pooled_h_gt_1']['gain_curvature_mean'] is not None
                else "not defined (all reference cells degenerate)."
            ),
            f"- BH_old deprecated byproduct values were computed by the frozen runner but are excluded "
            "from all official G1a statistics.",
        ]
    dyn = context.get("dynamics")
    lines += ["", "## 8. Training dynamics context vs 5M anchor", ""]
    if dyn:
        lines += [
            f"- G1a (26M) future-step clip activation rate (step 1 of 36 baseline streams): "
            f"{dyn['g1a']['future_clip_rate_step1']:.4g}; 5M anchor (A0C 3 states, 8 streams each): "
            f"{dyn['anchor5m']['future_clip_rate_step1']:.4g}.",
            f"- G1a first-step preclip grad norm mean over streams: {dyn['g1a']['first_preclip_norm_mean']:.6g} "
            f"(median {dyn['g1a']['first_preclip_norm_median']:.6g}); anchor 5M: "
            f"{dyn['anchor5m']['first_preclip_norm_mean']:.6g} (median {dyn['anchor5m']['first_preclip_norm_median']:.6g}).",
            f"- G1a training-phase clip activation (bank steps): {dyn['g1a']['training_clip_rate']:.4g}; "
            f"mean preclip norm: {dyn['g1a']['training_preclip_norm_mean']:.6g}.",
            "- These variables are context only; no state was filtered on them.",
        ]
    lines += [
        "",
        "## 9. Verdict (frozen thresholds, G1A_PROTOCOL.md §10)",
        "",
        f"**{verdict['verdict']}**",
        "",
        "| condition | pass | value |",
        "|---|---:|---|",
    ]
    for name, cond in verdict["conditions"].items():
        val = cond.get("value")
        extra = {k: v for k, v in cond.items() if k not in {"pass", "value"}}
        lines.append(f"| {name} | {cond['pass']} | {_fmt(val) if val is not None else ''} {extra if extra else ''} |")
    lines += [
        "",
        "## 10. Anomalies, OOM, retries, caveats",
        "",
    ]
    for item in context.get("incidents", []):
        lines.append(f"- {item}")
    lines += [
        "- HVP uses the math-SDPA reverse-over-reverse surrogate (native flash-SDPA has no double "
        "backward): C/BH_fixed are mixed-backend analytic surrogates, same frozen caveat as A0C/A0E2.",
        "- chi_FD uses the registered finite epsilon per state; C is a zero-pulse smooth tangent — "
        "the same finite-epsilon interpretation caveat as A0C applies.",
    ]
    if context.get("resource_limit"):
        lines += ["", "## RESOURCE_LIMIT", "", context["resource_limit"]]
    lines += [
        "",
        "## 11. Scope statement",
        "",
        "G1a is model-scale generalization only. It does NOT replace A0F2 (56-state census at 5M) and "
        "does NOT by itself establish generalization at 100M/1B or beyond.",
        "",
        "## 12. Next step",
        "",
        context.get("next_step", "Decision on ~100M G1b is deferred to the user; it is NOT started by this experiment."),
    ]
    return "\n".join(lines) + "\n"


def json_dumps(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def file_sha256_or_missing(path) -> str:
    p = Path(path)
    return file_sha256(p) if p.is_file() else "MISSING"
