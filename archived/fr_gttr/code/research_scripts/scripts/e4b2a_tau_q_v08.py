#!/usr/bin/env python3
"""E4B-2A：q-dynamics 单时间尺度+plateau 拟合与 curve-collapse 探索诊断。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fr.e4b_dynamic_timescale import (
    atomic_json,
    atomic_text,
    classify_fit_quality,
    classify_screen,
    curve_collapse_diagnostic,
    distribution_summary,
    fit_tau_plateau,
    load_q_rotation_rows,
    loso_single_candidate,
    sha256_file,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e4b_dynamic_timescale_v08.json")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = root / cfg["tau_output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    _, state_rows, source_files = load_q_rotation_rows(root, cfg)
    horizons = list(map(int, cfg["horizons"]))
    primary_h = int(cfg["primary_exploratory_horizon"])
    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    states = list(map(int, cfg["state_steps"]))

    by_state = {(int(r["seed"]), int(r["state_step"])): [] for r in state_rows}
    for r in state_rows:
        by_state[(int(r["seed"]), int(r["state_step"]))].append(r)

    fits = []
    for seed in seeds:
        for step in states:
            rows = sorted(by_state[(seed, step)], key=lambda r: int(r["horizon"]))
            if [int(r["horizon"]) for r in rows] != horizons:
                raise RuntimeError(f"horizon grid mismatch seed={seed}, state={step}")
            obs = [float(r["mean_Rq"]) for r in rows]
            fit = fit_tau_plateau(horizons, obs, cfg["tau_fit"])
            rec = {
                "seed": seed,
                "state_step": step,
                "state_fraction": float(rows[0]["state_fraction"]),
                "tau_q": float(fit["tau_q"]),
                "c": float(fit["c"]),
                "RMSE": float(fit["rmse"]),
                "R2": float(fit["r2"]),
                "fit_success": bool(fit["fit_success"]),
                "boundary_hit_tau": bool(fit["boundary_hit_tau"]),
                "boundary_hit_c": bool(fit["boundary_hit_c"]),
            }
            for h, y, yp in zip(horizons, fit["observed_Rq"], fit["fitted_Rq"]):
                rec[f"observed_Rq_H{h}"] = float(y)
                rec[f"fitted_Rq_H{h}"] = float(yp)
            fits.append(rec)

    if len(fits) != len(seeds) * len(states):
        raise RuntimeError("tau_q fit count != 56")

    with (out / "e4b2a_tau_q_state_fits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
        w.writeheader(); w.writerows(fits)
    atomic_json(out / "e4b2a_tau_q_state_fits.json", fits)

    quality = classify_fit_quality([
        {
            "r2": r["R2"], "fit_success": r["fit_success"],
            "boundary_hit_tau": r["boundary_hit_tau"], "boundary_hit_c": r["boundary_hit_c"],
        } for r in fits
    ], cfg["tau_fit"])

    tau_summary = distribution_summary([r["tau_q"] for r in fits])
    c_summary = distribution_summary([r["c"] for r in fits])
    r2_summary = distribution_summary([r["R2"] for r in fits])

    # seed/state variation：描述性 ANOVA-like 方差分解，不做显著性宣称。
    seed_tau_means = {seed: float(np.mean([r["tau_q"] for r in fits if int(r["seed"]) == seed])) for seed in seeds}
    step_tau_means = {step: float(np.mean([r["tau_q"] for r in fits if int(r["state_step"]) == step])) for step in states}
    variation = {
        "tau_q_total_std": float(np.std([r["tau_q"] for r in fits], ddof=1)),
        "seed_mean_std": float(np.std(list(seed_tau_means.values()), ddof=1)),
        "state_step_mean_std": float(np.std(list(step_tau_means.values()), ddof=1)),
        "seed_means": seed_tau_means,
        "state_step_means": step_tau_means,
    }

    fit_map = {(int(r["seed"]), int(r["state_step"])): r for r in fits}
    h8_rows = []
    all_rows = []
    for r in state_rows:
        fr = fit_map[(int(r["seed"]), int(r["state_step"]))]
        rr = dict(r)
        rr["tau_q"] = float(fr["tau_q"])
        rr["c"] = float(fr["c"])
        rr["normalized_horizon"] = float(r["horizon"]) / float(fr["tau_q"])
        all_rows.append(rr)
        if int(r["horizon"]) == primary_h:
            h8_rows.append(rr)

    tau_screen = loso_single_candidate(h8_rows, "tau_q")
    c_screen = loso_single_candidate(h8_rows, "c")
    tau_status = classify_screen(tau_screen, cfg["screening"])
    c_status = classify_screen(c_screen, cfg["screening"])

    collapse = curve_collapse_diagnostic(all_rows, cfg["curve_collapse"])
    binned_rows = collapse.pop("binned_rows")

    # 输出用于复查的 normalized-horizon 表。
    collapse_csv_rows = []
    for r in binned_rows:
        collapse_csv_rows.append({
            "seed": int(r["seed"]), "state_step": int(r["state_step"]),
            "state_fraction": float(r["state_fraction"]), "horizon": int(r["horizon"]),
            "tau_q": float(r["tau_q"]), "normalized_horizon": float(r["normalized_horizon"]),
            "x_bin": int(r["x_bin"]), "x_bin_left": float(r["x_bin_left"]), "x_bin_right": float(r["x_bin_right"]),
            "B": float(r["B"]), "chi": float(r["chi"]),
        })
    with (out / "e4b2a_curve_collapse_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(collapse_csv_rows[0].keys()))
        w.writeheader(); w.writerows(collapse_csv_rows)

    worst = sorted(fits, key=lambda r: float(r["R2"]))[:5]
    analysis = {
        "protocol": cfg["protocol_version"] + "_E4B2A",
        "study_role": cfg["study_role"],
        "data_integrity": True,
        "fit_model": cfg["tau_fit"]["model"],
        "fit_quality": quality,
        "tau_q_summary": tau_summary,
        "c_summary": c_summary,
        "R2_summary": r2_summary,
        "variation": variation,
        "worst_5_states_by_R2": worst,
        "H8_tau_screen": tau_screen,
        "H8_tau_status": tau_status,
        "H8_c_screen": c_screen,
        "H8_c_status": c_status,
        "curve_collapse": collapse,
        "boundary": [
            "v0.7 H8_CONFIRMATION=FAIL remains unchanged.",
            "tau_q/c are fitted from future baseline q dynamics, so they are mechanism diagnostics, not online controller states.",
            "Curve-collapse is descriptive and uses no chi/B ratio.",
            "No multi-variable combination search was performed.",
        ],
    }
    atomic_json(out / "e4b2a_tau_q_analysis.json", analysis)

    lines = [
        "# E4B-2A v0.8 q-Dynamics Low-dimensional Timescale",
        "",
        "本报告为 post-confirmatory exploratory mechanism study；不改写 v0.7 confirmatory FAIL。",
        "",
        "## Data Integrity",
        "",
        f"- source formal q-rotation state files: `{len(source_files)}`",
        f"- fitted seed×state curves: `{len(fits)}` (expected 56)",
        "- formal B/chi: `NOT_RECOMPUTED`",
        "- GPU replay: `NO`",
        "",
        "## Frozen Fit",
        "",
        "`Rbar_q(H) = c + (1-c) * exp(-H/tau_q)`",
        "",
        f"- tau range: `[{cfg['tau_fit']['tau_min']}, {cfg['tau_fit']['tau_max']}]` optimizer steps",
        f"- coarse log grid: `{cfg['tau_fit']['coarse_log_grid_points']}`",
        f"- refine log grid: `{cfg['tau_fit']['refine_log_grid_points']}` × `{cfg['tau_fit']['refine_rounds']}` rounds",
        "- scipy: `NOT_REQUIRED`",
        "- deterministic numpy bounded fit: `YES`",
        "",
        "## Fit Quality",
        "",
        f"- E4B2A_TIMESCALE_FIT_STATUS: `{quality['status']}`",
        f"- median R2: `{quality['median_r2']:.6g}`",
        f"- fit success fraction: `{quality['success_fraction']:.4%}`",
        f"- any-boundary-hit fraction: `{quality['boundary_hit_fraction']:.4%}`",
        f"- tau_q median: `{tau_summary['median']:.6g}`",
        f"- tau_q IQR: `[{tau_summary['q25']:.6g}, {tau_summary['q75']:.6g}]`",
        f"- tau_q range: `[{tau_summary['min']:.6g}, {tau_summary['max']:.6g}]`",
        f"- c median: `{c_summary['median']:.6g}`",
        f"- c IQR: `[{c_summary['q25']:.6g}, {c_summary['q75']:.6g}]`",
        f"- c range: `[{c_summary['min']:.6g}, {c_summary['max']:.6g}]`",
        "",
        "### tau_q variation",
        "",
        f"- total state-level std: `{variation['tau_q_total_std']:.6g}`",
        f"- std of seed means: `{variation['seed_mean_std']:.6g}`",
        f"- std of state-step means: `{variation['state_step_mean_std']:.6g}`",
        "",
        "### Worst 5 fits",
        "",
        "| seed | state | tau_q | c | R2 | RMSE | boundary |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in worst:
        lines.append(
            f"| {r['seed']} | {r['state_step']} | {r['tau_q']:.6g} | {r['c']:.6g} | {r['R2']:.6g} | {r['RMSE']:.6g} | "
            f"tau={r['boundary_hit_tau']}, c={r['boundary_hit_c']} |"
        )

    lines += [
        "",
        "## H8 Single-state Screens",
        "",
        "| candidate | status | mean gain | median gain | folds improved | interaction sign stability |",
        "|---|---|---:|---:|---:|---:|",
        f"| tau_q | {tau_status} | {tau_screen['mean_relative_mae_improvement']:.4%} | {tau_screen['median_relative_mae_improvement']:.4%} | {tau_screen['folds_improved']}/8 | {tau_screen['delta_same_sign_folds']}/8 ({tau_screen['delta_majority_sign']}) |",
        f"| c | {c_status} | {c_screen['mean_relative_mae_improvement']:.4%} | {c_screen['median_relative_mae_improvement']:.4%} | {c_screen['folds_improved']}/8 | {c_screen['delta_same_sign_folds']}/8 ({c_screen['delta_majority_sign']}) |",
        "",
        "## Curve-collapse Diagnostic",
        "",
        "主要统计不使用 chi/B；而是在 raw-H 组或固定 H/tau_q bin 内拟合 `chi ~ seed_FE + state_fraction + B`，比较 pooled residual RMSE。",
        "",
        f"- raw-H pooled RMSE: `{collapse['raw_horizon']['pooled_rmse']:.6g}`",
        f"- H/tau_q pooled RMSE: `{collapse['normalized_horizon']['pooled_rmse']:.6g}`",
        f"- relative RMSE improvement: `{collapse['relative_pooled_rmse_improvement']:.4%}`",
        f"- normalized used fraction: `{collapse['normalized_used_fraction']:.4%}`",
        f"- E4B2A_CURVE_COLLAPSE: `{collapse['status']}`",
        "",
        "### Fixed normalized-horizon bins",
        "",
        "| bin | n | used | RMSE |",
        "|---:|---:|---|---:|",
    ]
    for g in collapse["normalized_horizon"]["groups"]:
        rmse = "NA" if g["rmse"] is None else f"{g['rmse']:.6g}"
        lines.append(f"| {g['group']} | {g['n']} | {g['used']} | {rmse} |")

    lines += [
        "",
        "## Boundary",
        "",
        "- [FR-OBSERVATION-ONLY] tau_q/c 只由 future baseline q dynamics 拟合。",
        "- [FR-METRIC] H8 tau_q 与 c 分开筛选，不做 tau_q+c 或其它多变量组合。",
        "- [FR-METRIC] curve-collapse 是描述性诊断，不是新的 confirmation。",
        "- [FR-IO] 只写 tau_q/ 新目录。",
        "- 不进入 controller，不重写 v0.7。",
        "",
        "## Final",
        "",
        "E4B2A_DATA_INTEGRITY = PASS",
        f"E4B2A_TIMESCALE_FIT_STATUS = {quality['status']}",
        f"E4B2A_MEDIAN_TAU_Q = {tau_summary['median']:.10g}",
        f"E4B2A_TAU_Q_IQR = [{tau_summary['q25']:.10g}, {tau_summary['q75']:.10g}]",
        f"E4B2A_MEDIAN_R2 = {quality['median_r2']:.10g}",
        f"E4B2A_BOUNDARY_HIT_FRACTION = {quality['boundary_hit_fraction']:.10g}",
        f"E4B2A_TAU_Q_H8_STATUS = {tau_status}",
        f"E4B2A_C_H8_STATUS = {c_status}",
        f"E4B2A_CURVE_COLLAPSE = {collapse['status']}",
    ]

    # 可选画图；缺 matplotlib 不影响分析。
    plot_status = make_plot_if_available(out, all_rows, fits)
    lines.append(f"E4B2A_PLOT = {plot_status}")
    atomic_text(out / "e4b2a_tau_q_analysis.md", "\n".join(lines) + "\n")

    audit = [
        "# E4B2A_AUDIT",
        "",
        "E4B2A_DATA_INTEGRITY = PASS",
        f"SOURCE_Q_ROTATION_STATE_FILE_COUNT = {len(source_files)}",
        f"CONFIG_SHA256 = {sha256_file(cfg_path)}",
        "FORMAL_B_CHI_RECOMPUTED = NO",
        "SCIPY_INSTALLED_OR_REQUIRED = NO",
        "TAU_FIT_DETERMINISTIC = YES",
        "GPU_RERUN = NO",
        "V07_LOCKED_FILES_WRITTEN = NO",
        "V07_RAW_FILES_WRITTEN = NO",
        "E4A_E4B1_RAW_FILES_WRITTEN = NO",
        "MULTIVARIABLE_SEARCH = NO",
        f"OUTPUT_SCOPE = {cfg['tau_output_dir']}",
        "CURVE_COLLAPSE_CONFIRMATORY = NO",
    ]
    atomic_text(out / "E4B2A_AUDIT.md", "\n".join(audit) + "\n")

    next_rec = decide_next(status_tau=tau_status, status_c=c_status, collapse=collapse["status"], fit_quality=quality["status"])
    print("E4B2A_DATA_INTEGRITY = PASS")
    print("E4B2A_TIMESCALE_FIT_STATUS =", quality["status"])
    print("E4B2A_MEDIAN_TAU_Q =", tau_summary["median"])
    print("E4B2A_TAU_Q_IQR =", [tau_summary["q25"], tau_summary["q75"]])
    print("E4B2A_MEDIAN_R2 =", quality["median_r2"])
    print("E4B2A_BOUNDARY_HIT_FRACTION =", quality["boundary_hit_fraction"])
    print("E4B2A_TAU_Q_H8_STATUS =", tau_status)
    print("E4B2A_C_H8_STATUS =", c_status)
    print("E4B2A_CURVE_COLLAPSE =", collapse["status"])
    print("NEXT_RECOMMENDATION =", next_rec)
    print("REPORT =", out / "e4b2a_tau_q_analysis.md")
    return 0


def make_plot_if_available(out: Path, rows, fits) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "SKIPPED_MATPLOTLIB_UNAVAILABLE"
    # [FR-PLOT] 仅描述性图；不参与 status 判定。
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    for seed in sorted({int(r["seed"]) for r in rows}):
        pts = [r for r in rows if int(r["seed"]) == seed]
        pts = sorted(pts, key=lambda r: float(r["normalized_horizon"]))
        ax.scatter([r["normalized_horizon"] for r in pts], [r["chi"] for r in pts], s=8, alpha=0.45, label=str(seed))
    ax.set_xscale("log")
    ax.set_xlabel("H / tau_q")
    ax.set_ylabel("chi")
    ax.set_title("E4B-2A descriptive normalized-horizon view")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    p = out / "e4b2a_curve_collapse.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def decide_next(*, status_tau: str, status_c: str, collapse: str, fit_quality: str) -> str:
    if fit_quality in {"GOOD", "MIXED"} and (status_tau == "PROMISING" or status_c == "PROMISING" or collapse == "SUPPORTED"):
        return "E4C_LOW_DIM_STATE"
    if fit_quality == "POOR" or (status_tau == "UNSUPPORTED" and status_c == "UNSUPPORTED" and collapse == "UNSUPPORTED"):
        return "E4B2B_GRADIENT_TIMESCALE"
    return "REASSESS_DYNAMIC_STATE_HYPOTHESIS"


if __name__ == "__main__":
    raise SystemExit(main())
