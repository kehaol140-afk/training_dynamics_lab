#!/usr/bin/env python3
"""A0U analyzer: Stage 1 phase analysis + Stage 2/3 schedule comparison.

Offline (CPU): reads frozen A0O stage1 chi trajectories + the driver
rollout records; writes response_phase_analysis.csv, schedule_comparison.csv,
analysis.json, plots, and A0U_REPORT.md. No GPU, no recomputation of chi.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0u_response_schedule_discovery_v01"
RUNS = OUT / "runs"
ANALYSIS = OUT / "analysis"
REPORTS = OUT / "reports"
PLOTS = OUT / "analysis" / "plots"
A0O_CELLS = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01/runs/stage1_cells.jsonl"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
import fr.a0u_response_schedule_discovery_v01 as a0u  # noqa: E402

KNOTS = a0u.KNOTS


def write_csv(path, rows, header):
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in header))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    t_start = time.perf_counter()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    trajs = a0u.load_trajectories(A0O_CELLS)
    records = [json.loads(l) for l in (RUNS / "cells.jsonl").read_text().splitlines()]
    assert len(records) == 12, f"expected 12 cells, got {len(records)}"
    assert all(r["crn_ok"] for r in records), "CRN failure in rollout records"

    # ---------------- Stage 1: phase analysis ----------------
    phase_rows = []
    for (seed, stream), chi_by_step in sorted(trajs.items()):
        st = a0u.trajectory_stats(chi_by_step)
        row = {"seed": seed, "stream": stream,
               **{f"chi_{s}": chi_by_step[s] for s in KNOTS},
               "mean": st["mean"], "var": st["var"],
               "sign_consistency": st["sign_consistency"],
               "all_negative": st["all_negative"]}
        for ph, d in st["phases"].items():
            row[f"{ph}_mean_abs"] = d["mean_abs"]
            row[f"{ph}_mean_sign"] = d["mean_sign"]
            row[f"{ph}_all_negative"] = d["all_negative"]
        phase_rows.append(row)
    write_csv(ANALYSIS / "response_phase_analysis.csv", phase_rows,
              ["seed", "stream", "chi_46", "chi_52", "chi_64", "chi_82",
               "mean", "var", "sign_consistency", "all_negative",
               "early_mean_abs", "early_mean_sign", "early_all_negative",
               "middle_mean_abs", "middle_mean_sign", "middle_all_negative",
               "late_mean_abs", "late_mean_sign", "late_all_negative"])
    css = a0u.cross_seed_similarity(trajs)

    phase_summary = {
        "n_trajectories": len(phase_rows),
        "frac_all_negative": float(np.mean([r["all_negative"] for r in phase_rows])),
        "median_sign_consistency": float(np.median([r["sign_consistency"] for r in phase_rows])),
        "sign_consistency_distribution": sorted({r["sign_consistency"] for r in phase_rows}),
        "phase_mean_abs": {ph: {"median": float(np.median([r[f"{ph}_mean_abs"] for r in phase_rows])),
                                  "min": float(np.min([r[f"{ph}_mean_abs"] for r in phase_rows])),
                                  "max": float(np.max([r[f"{ph}_mean_abs"] for r in phase_rows]))}
                           for ph in ("early", "middle", "late")},
        "does_response_define_phases": {
            "note": "descriptive: |chi| median per phase + sign consistency across knots",
            "all_negative_count": int(sum(1 for r in phase_rows if r["all_negative"])),
            "late_weaker_than_early_frac":
                float(np.mean([r["late_mean_abs"] < r["early_mean_abs"] for r in phase_rows])),
        },
        "cross_seed_similarity": css,
    }

    # ---------------- Stage 2/3: schedule comparison ----------------
    comp_rows = []
    metrics = {}
    for rec in records:
        key = (rec["seed"], rec["stream_seed"])
        metrics[key] = {arm: a0u.curve_metrics(rec["losses"][arm])
                        for arm in ("A", "B", "C")}
        for arm in ("A", "B", "C"):
            m = metrics[key][arm]
            row = {"seed": rec["seed"], "stream": rec["stream_seed"], "arm": arm,
                   "final_loss": m["final_loss"], "auc": m["auc"],
                   "early_improvement_vs_A":
                       a0u.early_improvement(rec["losses"]["A"], rec["losses"][arm])
                       if arm != "A" else 0.0,
                   "late_std": m["late_std"], "late_slope": m["late_slope"],
                   "cell_time_s": rec["cell_time_s"]}
            if arm == "C":
                row["n_lr_changes"] = rec["schedule_props"]["n_lr_changes"]
                row["n_up_segments"] = rec["schedule_props"]["n_up_segments"]
                row["n_down_segments"] = rec["schedule_props"]["n_down_segments"]
                row["multipliers"] = "|".join(f"{m_:.6f}" for m_ in rec["schedule_props"]["multipliers"])
            comp_rows.append(row)
    write_csv(ANALYSIS / "schedule_comparison.csv", comp_rows,
              ["seed", "stream", "arm", "final_loss", "auc",
               "early_improvement_vs_A", "late_std", "late_slope",
               "n_lr_changes", "n_up_segments", "n_down_segments",
               "multipliers", "cell_time_s"])

    def metric(rec, arm, name):
        return metrics[(rec["seed"], rec["stream_seed"])][arm][name]

    final_blocks = {f"{x}_minus_{y}": a0u.paired_block(
        records, lambda r, a=x: metric(r, a, "final_loss"),
        lambda r, b=y: metric(r, b, "final_loss"))
        for (x, y) in (("A", "B"), ("A", "C"), ("B", "C"))}
    auc_blocks = {f"{x}_minus_{y}": a0u.paired_block(
        records, lambda r, a=x: metric(r, a, "auc"),
        lambda r, b=y: metric(r, b, "auc"))
        for (x, y) in (("A", "B"), ("A", "C"), ("B", "C"))}
    # early improvement: paired_block computes first - second; feed 0 vs
    # -improvement so the reported median is the improvement itself.
    early_blocks = {x: a0u.paired_block(
        records, lambda r: 0.0,
        lambda r, x=x: -a0u.early_improvement(r["losses"]["A"], r["losses"][x]))
        for x in ("B", "C")}

    verdict = a0u.verdict_case(final_blocks["A_minus_C"], final_blocks["B_minus_C"],
                               final_blocks["A_minus_B"])

    sched_props = [r["schedule_props"] for r in records]
    schedule_summary = {
        "n_cells": len(records),
        "n_lr_changes_distribution": sorted(p["n_lr_changes"] for p in sched_props),
        "n_up_segments_distribution": sorted(p["n_up_segments"] for p in sched_props),
        "n_down_segments_distribution": sorted(p["n_down_segments"] for p in sched_props),
        "multiplier_vectors": [p["multipliers"] for p in sched_props],
        "identical_to_B_frac": float(np.mean([
            all(abs(m - a0u.B_MULT) < 1e-9 for m in p["multipliers"]) for p in sched_props])),
        "mean_n_lr_changes": float(np.mean([p["n_lr_changes"] for p in sched_props])),
    }

    budget = json.loads((OUT / "gpu_budget_log.json").read_text())
    cost = {
        "schedule_generation_cost_s": round(time.perf_counter() - t_start, 3),
        "training_cost_gpu_s": budget["TOTALS"]["cumulative_wall_s"],
        "budget_s": budget["TOTALS"]["budget_s"],
        "per_arm_gpu_s_est": round(budget["TOTALS"]["cumulative_wall_s"] / 4.0, 1),
    }

    analysis = {
        "stage1_phase_analysis": phase_summary,
        "stage3_blocks_final_loss": final_blocks,
        "stage3_blocks_auc": auc_blocks,
        "stage3_blocks_early_improvement": early_blocks,
        "schedule_summary": schedule_summary,
        "verdict": verdict,
        "cost": cost,
        "crn_all": all(r["crn_ok"] for r in records),
        "n_cells": len(records),
    }
    (ANALYSIS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for (seed, stream), d in sorted(trajs.items()):
        ax.plot(KNOTS, [d[s] for s in KNOTS], marker="o", ms=3, alpha=0.6,
                label=f"s{seed}-x{stream}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Stage 1: chi_lr trajectories (frozen A0O bank)")
    ax.set_xlabel("bank step")
    ax.set_ylabel("chi_lr")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    cps = a0u.CHECKPOINTS
    for arm, color in (("A", "k"), ("B", "tab:blue"), ("C", "tab:red")):
        curves = np.array([[rec["losses"][arm][str(o - a0u.START_STEP)] for o in cps]
                           for rec in records])
        ax.plot(cps, curves.mean(axis=0), marker="o", color=color, label=f"arm {arm}")
        ax.fill_between(cps, curves.mean(axis=0) - curves.std(axis=0),
                        curves.mean(axis=0) + curves.std(axis=0), color=color, alpha=0.15)
    ax.set_title("Stage 3: val-loss curves (12 cells, mean±std)")
    ax.set_xlabel("bank step")
    ax.set_ylabel("val loss")
    ax.legend()

    ax = axes[2]
    d_ca = np.array([metric(r, "A", "final_loss") - metric(r, "C", "final_loss")
                     for r in records])
    d_cb = np.array([metric(r, "B", "final_loss") - metric(r, "C", "final_loss")
                     for r in records])
    d_ba = np.array([metric(r, "A", "final_loss") - metric(r, "B", "final_loss")
                     for r in records])
    bp = ax.boxplot([d_ba, d_ca, d_cb], tick_labels=["A-B", "A-C", "B-C"],
                    showmeans=True)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Paired final-loss differences (12 cells)")
    ax.set_ylabel("loss difference (positive = right arm better)")
    fig.tight_layout()
    fig.savefig(PLOTS / "a0u_summary.png", dpi=130)
    plt.close(fig)

    print(json.dumps({"verdict": verdict, "cost": cost,
                      "schedule_changes": schedule_summary["n_lr_changes_distribution"],
                      "identical_to_B_frac": schedule_summary["identical_to_B_frac"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
