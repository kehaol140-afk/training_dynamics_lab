#!/usr/bin/env python3
"""v0.7 confirmatory FR 的 CPU-only 预注册分析。

Primary endpoint 固定为 H=8。Discovery seeds 只允许进入末尾比较段。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from fr.confirmatory_stats import analyze_confirmatory_horizons
from scientific_preflight_v07 import assert_prereg_lock_valid, file_sha256


def load_required_seed_results(out: Path, seeds):
    raw = []
    for seed in map(int, seeds):
        p = out / f"seed_{seed}/theta_mapping_raw.json"
        if not p.exists():
            raise SystemExit(f"缺少 confirmation seed {seed} theta_mapping_raw.json")
        x = json.loads(p.read_text(encoding="utf-8"))
        if x.get("protocol") != "E3C_CONFIRMATORY_V07_MULTISEED_THETA":
            raise SystemExit(f"seed {seed} protocol 不是 v0.7 confirmatory")
        if not x.get("technical_pass") or not x.get("rho0_noop_pass"):
            raise SystemExit(f"seed {seed} technical/no-op 未 PASS")
        if int(x.get("train_seed", -1)) != seed:
            raise SystemExit(f"seed {seed} result train_seed 不匹配")
        raw.append(x)
    return raw


def load_optional_discovery_results(out: Path, seeds):
    raw = []
    for seed in map(int, seeds):
        p = out / f"seed_{seed}/theta_mapping_raw.json"
        if not p.exists():
            return None
        x = json.loads(p.read_text(encoding="utf-8"))
        if not x.get("technical_pass") or not x.get("rho0_noop_pass"):
            return None
        raw.append(x)
    return raw


def flatten_by_horizon(raw_results, horizons, *, source_role: str):
    rows_by_h = {str(int(h)): [] for h in horizons}
    for result in raw_results:
        seed = int(result["train_seed"])
        for state in result["states"]:
            for h in horizons:
                hk = str(int(h))
                if hk not in state["B_theta"] or hk not in state["chi_theta"]:
                    continue
                rows_by_h[hk].append({
                    "source_role": source_role,
                    "train_seed": seed,
                    "step": int(state["step"]),
                    "state_fraction": float(state["state_fraction"]),
                    "B": float(state["B_theta"][hk]),
                    "chi": float(state["chi_theta"][hk]),
                    "odd_error": float(state["odd_error_theta"][hk]),
                    "clipping": state.get("clipping", {}),
                })
    return rows_by_h


def sign_label(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def discovery_vs_confirmation_compare(*, confirmation, discovery):
    common = sorted(set(confirmation["analysis"]) & set(discovery["analysis"]), key=int)
    by_horizon = {}
    for h in common:
        ca = confirmation["analysis"][h]
        da = discovery["analysis"][h]
        by_horizon[h] = {
            "confirmation_K": ca["seed_fe_adjusted"]["K"],
            "discovery_K": da["seed_fe_adjusted"]["K"],
            "confirmation_K_direction": sign_label(ca["seed_fe_adjusted"]["K"]),
            "discovery_K_direction": sign_label(da["seed_fe_adjusted"]["K"]),
            "confirmation_loso_mean_gain": ca["loso"]["mean_relative_mae_improvement"],
            "discovery_loso_mean_gain": da["loso"]["mean_relative_mae_improvement"],
            "confirmation_loso_folds_improved": ca["loso"]["folds_improved"],
            "discovery_loso_folds_improved": da["loso"]["folds_improved"],
        }
    return {
        "section": "DISCOVERY_VS_CONFIRMATION_COMPARE",
        "boundary": "Discovery seeds 只用于最终比较，不进入 v0.7 primary confirmatory statistics。",
        "common_horizons": common,
        "confirmation_horizon_curve_K": {
            h: confirmation["analysis"][h]["seed_fe_adjusted"]["K"]
            for h in sorted(confirmation["analysis"], key=int)
        },
        "discovery_horizon_curve_K": {
            h: discovery["analysis"][h]["seed_fe_adjusted"]["K"]
            for h in sorted(discovery["analysis"], key=int)
        },
        "by_horizon": by_horizon,
    }


def fmt(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(x) else f"{x:.6g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e3c_confirmatory_v07.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "research/src"))
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    out = root / cfg["output_dir"]
    lock = assert_prereg_lock_valid(root, out)

    confirmation_seeds = list(map(int, cfg["confirmation_train_seeds"]))
    discovery_seeds = list(map(int, cfg["discovery_train_seeds"]))
    horizons = list(map(int, cfg["horizons"]))
    primary_horizon = int(cfg["primary_horizon"])

    confirmation_raw = load_required_seed_results(out, confirmation_seeds)
    rows_by_h = flatten_by_horizon(
        confirmation_raw,
        horizons,
        source_role="CONFIRMATION_PRIMARY_ELIGIBLE",
    )
    for h in horizons:
        n = len(rows_by_h[str(h)])
        expected = len(confirmation_seeds) * len(cfg["e3_state_steps"])
        if n != expected:
            raise SystemExit(f"H={h} confirmation rows={n}，预期={expected}")

    confirmation_analysis = analyze_confirmatory_horizons(
        rows_by_h,
        primary_horizon=primary_horizon,
        bootstrap_reps=int(cfg["confirmatory_bootstrap_reps"]),
        permutation_reps=int(cfg["confirmatory_permutation_reps"]),
        random_seed=int(cfg["confirmatory_random_seed"]),
    )
    primary = confirmation_analysis["analysis"][str(primary_horizon)]["primary_confirmation"]

    discovery_compare = None
    discovery_out = root / cfg["discovery_output_dir"]
    discovery_raw = load_optional_discovery_results(discovery_out, discovery_seeds)
    if discovery_raw is not None:
        discovery_horizons = sorted({
            int(h)
            for result in discovery_raw
            for state in result["states"]
            for h in state.get("B_theta", {})
        })
        discovery_rows_by_h = flatten_by_horizon(
            discovery_raw,
            discovery_horizons,
            source_role="DISCOVERY_ONLY",
        )
        discovery_analysis = analyze_confirmatory_horizons(
            discovery_rows_by_h,
            primary_horizon=-1,
            bootstrap_reps=int(cfg["confirmatory_bootstrap_reps"]),
            permutation_reps=int(cfg["confirmatory_permutation_reps"]),
            random_seed=int(cfg["confirmatory_random_seed"]) + 909000,
        )
        discovery_compare = discovery_vs_confirmation_compare(
            confirmation=confirmation_analysis,
            discovery=discovery_analysis,
        )

    report = {
        "protocol": "E3C_CONFIRMATORY_V07_ANALYSIS",
        "technical_pass": True,
        "H8_CONFIRMATION": primary["H8_CONFIRMATION"],
        "primary_confirmation": primary,
        "confirmation_seeds": confirmation_seeds,
        "discovery_seeds": discovery_seeds,
        "discovery_boundary": "Discovery seeds 不进入 primary fitting/p-value/bootstrap CI/LOSO。",
        "primary_horizon": primary_horizon,
        "horizons": horizons,
        "analysis": confirmation_analysis["analysis"],
        "secondary_holm": confirmation_analysis["secondary_holm"],
        "DISCOVERY_VS_CONFIRMATION_COMPARE": discovery_compare,
        "prereg_lock": lock,
        "prereg_lock_sha256": file_sha256(out / "PREREG_LOCK.json"),
    }
    (out / "e3c_confirmatory_analysis_v07.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# v0.7 Confirmatory FR Analysis",
        "",
        f"- H8_CONFIRMATION: **{primary['H8_CONFIRMATION']}**",
        f"- confirmation seeds: `{confirmation_seeds}`",
        f"- primary horizon: `H={primary_horizon}`",
        "- discovery seeds: `DISCOVERY ONLY`，不进入 primary statistics",
        "",
        "## Primary H=8",
    ]
    obs = primary["observed"]
    cond = primary["conditions"]
    lines += [
        f"- K_8: `{fmt(obs['K'])}`",
        f"- cluster CI95 lower: `{fmt(obs['cluster_ci95_lower'])}`",
        f"- detrended Pearson: `{fmt(obs['within_seed_detrended_pearson'])}`",
        f"- circular-shift p(one-sided positive): `{fmt(obs['circular_shift_p_one_sided_positive'])}`",
        f"- LOSO mean relative MAE improvement: `{fmt(obs['loso_mean_relative_mae_improvement'])}`",
        f"- LOSO folds improved: `{obs['loso_folds_improved']}/{obs['loso_total_folds']}`",
    ]
    for key, value in cond.items():
        lines.append(f"- {key}: `{value}`")

    lines += ["", "## Secondary Horizon Curve"]
    for h in horizons:
        a = confirmation_analysis["analysis"][str(h)]
        holm = a.get("secondary_holm")
        lines += [
            "",
            f"### H={h}",
            f"- endpoint_role: `{a['endpoint_role']}`",
            f"- seed-FE K_H: `{fmt(a['seed_fe_adjusted']['K'])}`",
            f"- cluster CI95(K_H): `[{fmt(a['cluster_bootstrap_ci_K']['ci95'][0])}, {fmt(a['cluster_bootstrap_ci_K']['ci95'][1])}]`",
            f"- detrended Pearson: `{fmt(a['within_seed_detrended']['pearson'])}`",
            f"- circular-shift p: `{fmt(a['circular_shift_permutation']['p_one_sided_positive'])}`",
            f"- first-difference Pearson: `{fmt(a['first_difference']['pearson'])}`",
            f"- LOSO mean MAE improvement: `{fmt(a['loso']['mean_relative_mae_improvement'])}`",
            f"- LOSO folds improved: `{a['loso']['folds_improved']}/{a['loso']['total_folds']}`",
            f"- clipping baseline/+/-: `{fmt(a['clipping_rates']['baseline_rate'])}` / `{fmt(a['clipping_rates']['plus_rate'])}` / `{fmt(a['clipping_rates']['minus_rate'])}`",
        ]
        if holm is not None:
            lines.append(f"- secondary Holm p: `{fmt(holm['holm_p'])}` significant=`{holm['significant_holm_0_05']}`")
        for seed in confirmation_seeds:
            m = a["per_seed"][str(seed)]
            lines.append(
                f"- seed {seed}: Pearson=`{fmt(m['pearson'])}`, Spearman=`{fmt(m['spearman'])}`"
            )

    if discovery_compare is not None:
        lines += ["", "## DISCOVERY_VS_CONFIRMATION_COMPARE", "", discovery_compare["boundary"]]
        for h, item in discovery_compare["by_horizon"].items():
            lines.append(
                f"- H={h}: K direction discovery=`{item['discovery_K_direction']}`, "
                f"confirmation=`{item['confirmation_K_direction']}`, "
                f"LOSO gain discovery=`{fmt(item['discovery_loso_mean_gain'])}`, "
                f"confirmation=`{fmt(item['confirmation_loso_mean_gain'])}`"
            )

    md = out / "e3c_confirmatory_analysis_v07.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("E3C_V07_CONFIRMATORY_ANALYSIS_DONE")
    print("H8_CONFIRMATION=", primary["H8_CONFIRMATION"])
    print("report=", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
