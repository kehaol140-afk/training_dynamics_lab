#!/usr/bin/env python3
"""FR B0.5 mechanism identification analyzer (v01).

Reads the frozen config + raw tables produced by b05_run_v01.py and produces:
  - h1_alignment.csv / h2_timescales.csv / h3_channels.csv (per-cell tables)
  - summary.json (verdicts per the pre-registered decision rules)
  - figures/f1..f6
  - B05_mechanism_report.md

All analysis math lives in fr.b05_mechanism_v01 (CPU-tested).
Terminology rules from config.json apply to every string written here.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b05_mechanism_identification_v01"
RUNS = OUT / "runs"
FIGS = OUT / "figures"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
import fr.b05_mechanism_v01 as b5  # noqa: E402

CONFIG = json.loads((OUT / "config.json").read_text())
E = math.e


def read_csv(name):
    with open(RUNS / name) as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

steps = read_csv("step_records.csv")
vals = read_csv("val_curves.csv")
h2 = read_csv("h2_curves.csv")

cells = sorted({(int(r["seed"]), int(r["step"])) for r in steps})
ARMS = ["native", "persistent", "pulse4_release", "D_transplant"]
ARM_KEY = {"native": "A", "persistent": "B", "pulse4_release": "C", "D_transplant": "D"}

step_by = {(int(r["seed"]), int(r["step"]), r["arm"]): [] for r in steps}
for r in steps:
    step_by[(int(r["seed"]), int(r["step"]), r["arm"])].append(r)
for k in step_by:
    step_by[k].sort(key=lambda r: int(r["h"]))

val_by = defaultdict(dict)
for r in vals:
    val_by[(int(r["seed"]), int(r["step"]), r["arm"])][int(r["h"])] = {
        "v_eval": f(r["v_eval"]), "v_score": f(r["v_score"])}

h2_by = defaultdict(list)
for r in h2:
    h2_by[(int(r["seed"]), int(r["step"]))].append(r)
for k in h2_by:
    h2_by[k].sort(key=lambda r: int(r["lag"]))

# ---------------------------------------------------------------------------
# H1: time reparameterization
# ---------------------------------------------------------------------------

h1_rows = []
h1_detail = {}
for (seed, step) in cells:
    rowsA = step_by[(seed, step, "native")]
    t_grid = [int(r["h"]) for r in rowsA]
    lrA = [f(r["lr_native"]) for r in rowsA]
    sA = b5.cumulative_lr_time(lrA, [1.0] * len(lrA))
    L_A = {h: val_by[(seed, step, "native")][h]["v_score"] for h in
           sorted(val_by[(seed, step, "native")])}
    h_grid = sorted(L_A)
    L_A_vec = [L_A[h] for h in h_grid]
    sA_grid = [sA[h - 1] for h in h_grid]
    rec = {"cell": f"s{seed}_t{step}"}
    for arm in ["persistent", "pulse4_release", "D_transplant"]:
        rowsX = step_by[(seed, step, arm)]
        multX = [f(r["multiplier"]) for r in rowsX]
        sX_full = b5.cumulative_lr_time(lrA, multX)
        sX_grid = [sX_full[h - 1] for h in h_grid]
        L_X = {h: val_by[(seed, step, arm)][h]["v_score"] for h in h_grid}
        L_X_vec = [L_X[h] for h in h_grid]
        L_tilde, clamped = b5.align_loss_by_s(h_grid, sA_grid, L_A_vec, sX_grid)
        rel = b5.masked_rel_err(L_X_vec, L_tilde, L_A_vec, clamped)
        n_clamped = sum(clamped)
        row = {"cell": rec["cell"], "arm": arm, "alignment_rel_err": rel,
               "n_clamped": n_clamped, "n_points": len(h_grid)}
        if arm == "pulse4_release":
            lag = b5.implied_lag(sA, t_grid, sX_full)
            post_lag = lag[4:]  # h>=5; tail clamp included in descriptive stats
            xs = [x for x in post_lag if x is not None]
            row["implied_lag_post_mean"] = sum(xs) / len(xs)
            row["implied_lag_post_std"] = (max(xs) - min(xs))
            delta = [L_X[h] - L_A[h] for h in h_grid]
            row["delta_h8"] = delta[1] if len(delta) > 1 else None
            row["delta_h64"] = delta[-1]
            row["delta_ratio_64_8"] = (delta[-1] / delta[1]
                                       if len(delta) > 1 and abs(delta[1]) > 1e-12
                                       else None)
            rec.setdefault("pulse", row)
        rec[ARM_KEY[arm]] = rel
        h1_rows.append(row)
    h1_detail[rec["cell"]] = rec

with open(OUT / "h1_alignment.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["cell", "arm", "alignment_rel_err",
                                       "n_clamped", "n_points",
                                       "implied_lag_post_mean",
                                       "implied_lag_post_std", "delta_h8",
                                       "delta_h64", "delta_ratio_64_8"])
    w.writeheader()
    for r in h1_rows:
        w.writerow({k: r.get(k, "") for k in w.fieldnames})

# decision rule (config): SUPPORTED if alignment_rel_err(B, H=64 window) < 0.5
# in >=5/6 cells AND pulse frozen-delta test holds in >=5/6 cells.
# Pulse test operationalization: (i) implied lag post-release approximately
# constant (std < 0.5 steps) AND (ii) |delta_h64| not decayed toward 0
# relative to |delta_h8| (ratio > 0.5) — both descriptive, within-sample.
b_ok = sum(1 for r in h1_rows if r["arm"] == "persistent"
           and r["alignment_rel_err"] < 0.5)
pulse_rows = [r for r in h1_rows if r["arm"] == "pulse4_release"]
p_lag_ok = sum(1 for r in pulse_rows
               if r["implied_lag_post_std"] is not None
               and r["implied_lag_post_std"] < 0.5)
p_delta_ok = sum(1 for r in pulse_rows
                 if r["delta_ratio_64_8"] is not None and r["delta_ratio_64_8"] > 0.5)
h1_verdict = ("SUPPORTED" if (b_ok >= 5 and p_lag_ok >= 5 and p_delta_ok >= 5)
              else "NOT_SUPPORTED" if (b_ok <= 1 or p_delta_ok <= 1)
              else "MIXED")

# ---------------------------------------------------------------------------
# H2: slow-fast separation (native arm only)
# ---------------------------------------------------------------------------

h2_rows = []
for (seed, step) in cells:
    rows = h2_by[(seed, step)]
    c_raw = [f(r["c_v_raw"]) for r in rows]
    c_res = [f(r["c_v_resid"]) for r in rows]
    pers = [f(r["dir_persist"]) for r in rows]
    tau_fast = b5.tau_from_curve(c_res)
    tau_slow = b5.tau_from_curve(pers)
    tau_slow_lb = tau_slow if tau_slow is not None else 17  # lower-bound marker
    h2_rows.append({"cell": f"s{seed}_t{step}", "tau_fast": tau_fast,
                    "tau_slow": tau_slow, "tau_slow_lower_bound": tau_slow is None,
                    "c_res_lag1": c_res[0], "c_res_lag4": c_res[3],
                    "c_res_lag8": c_res[7], "c_res_lag16": c_res[15],
                    "pers_lag1": pers[0], "pers_lag8": pers[7],
                    "pers_lag16": pers[15]})

with open(OUT / "h2_timescales.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(h2_rows[0].keys()))
    w.writeheader()
    for r in h2_rows:
        w.writerow(r)

n_fast_observed = sum(1 for r in h2_rows if r["tau_fast"] is not None)
def _tau_fast_lt_slow4(r):
    if r["tau_fast"] is None:
        return False
    slow = 17 if r["tau_slow_lower_bound"] else r["tau_slow"]
    return r["tau_fast"] < slow / 4.0
n_sep = sum(1 for r in h2_rows if _tau_fast_lt_slow4(r))
n_fast_big = sum(1 for r in h2_rows
                 if r["tau_fast"] is not None and not _tau_fast_lt_slow4(r))
h2_verdict = ("SUPPORTED" if (n_fast_observed >= 5 and n_sep >= 5)
              else "NOT_SUPPORTED" if (n_fast_observed <= 2 or n_fast_big >= 4)
              else "UNRESOLVED")

# ---------------------------------------------------------------------------
# H3: optimizer memory after-effect (channel decomposition)
# ---------------------------------------------------------------------------

h3_rows = []
for (seed, step) in cells:
    row = {"cell": f"s{seed}_t{step}"}
    for H in (16, 64):
        LA = val_by[(seed, step, "native")][H]["v_score"]
        GB = LA - val_by[(seed, step, "persistent")][H]["v_score"]
        GC = LA - val_by[(seed, step, "pulse4_release")][H]["v_score"]
        GD = LA - val_by[(seed, step, "D_transplant")][H]["v_score"]
        row[f"G_B_{H}"] = GB
        row[f"G_C_{H}"] = GC
        row[f"G_D_{H}"] = GD
        row[f"rho_C_{H}"] = GC / GB if abs(GB) > 1e-12 else None
        row[f"rho_D_{H}"] = GD / GB if abs(GB) > 1e-12 else None
        row[f"mv_share_{H}"] = b5.mv_channel_share(GC, GD)
    # distance trajectories at grid steps
    for H in (8, 16, 32, 64):
        for arm in ["persistent", "pulse4_release", "D_transplant"]:
            r = step_by[(seed, step, arm)][H - 1]
            row[f"d_theta_{ARM_KEY[arm]}_{H}"] = f(r["d_theta"])
            row[f"d_m_{ARM_KEY[arm]}_{H}"] = f(r["d_m"])
            row[f"d_v_{ARM_KEY[arm]}_{H}"] = f(r["d_v"])
    h3_rows.append(row)

with open(OUT / "h3_channels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(h3_rows[0].keys()))
    w.writeheader()
    for r in h3_rows:
        w.writerow(r)

shares64 = [r["mv_share_64"] for r in h3_rows if r["mv_share_64"] is not None]
n_share_gt05 = sum(1 for s in shares64 if s > 0.5)
n_share_lt02 = sum(1 for s in shares64 if s < 0.2)
h3_verdict = ("SUPPORTED" if n_share_gt05 >= 5
              else "NOT_SUPPORTED" if n_share_lt02 >= 5
              else "MIXED")
sign_ok = sum(1 for r in h3_rows
              if r["G_C_64"] * r["G_D_64"] > 0 or abs(r["G_D_64"]) < 1e-12)

# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------

summary = {
    "config_frozen": "outputs/fr_b05_mechanism_identification_v01/config.json",
    "cells": [f"s{s}_t{t}" for (s, t) in cells],
    "n_cells": len(cells),
    "H1_time_reparameterization": {
        "verdict": h1_verdict,
        "persistent_alignment_rel_err_lt_0.5_cells": f"{b_ok}/6",
        "pulse_implied_lag_approx_const_cells": f"{p_lag_ok}/6",
        "pulse_delta_not_decayed_cells": f"{p_delta_ok}/6",
        "per_cell": h1_detail,
        "label": "within-sample descriptive alignment; not a deployable predictor",
    },
    "H2_slow_fast": {
        "verdict": h2_verdict,
        "tau_fast_observed_cells": f"{n_fast_observed}/6",
        "tau_fast_lt_tau_slow_over_4_cells": f"{n_sep}/6",
        "per_cell": h2_rows,
        "claim_boundary": "only tau_fast << tau_slow (or its absence) is claimed; "
                          "no stationary-state claim of any kind",
    },
    "H3_optimizer_memory": {
        "verdict": h3_verdict,
        "mv_share_64_cells_gt_0.5": f"{n_share_gt05}/6",
        "mv_share_64_cells_lt_0.2": f"{n_share_lt02}/6",
        "gain_sign_consistency_C_vs_D_cells": f"{sign_ok}/6",
        "per_cell": h3_rows,
        "arm_D_label": "intervention arm (m/v transplant at h=4); not a training method",
    },
    "statistics": CONFIG["statistics"],
    "A0P_caveat": "all gains are short-window rollout gains on mid-training segments; "
                  "no final-training claim",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k]["verdict"] for k in
                  ["H1_time_reparameterization", "H2_slow_fast",
                   "H3_optimizer_memory"]}, indent=1))

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELLS_LBL = [f"s{s}_t{t}" for (s, t) in cells]
COL = {"A": "#222222", "B": "#c0392b", "C": "#2980b9", "D": "#27ae60"}

# f1: val curves per cell
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, (seed, step) in zip(axes.flat, cells):
    for arm, key in [("native", "A"), ("persistent", "B"),
                     ("pulse4_release", "C"), ("D_transplant", "D")]:
        hs = sorted(val_by[(seed, step, arm)])
        L = [val_by[(seed, step, arm)][h]["v_score"] for h in hs]
        ax.plot(hs, L, marker=".", ms=3, color=COL[key], label=key)
    ax.set_title(f"s{seed}_t{step}")
    ax.set_xlabel("optimizer step h")
    ax.set_ylabel("validation loss (v_score)")
axes.flat[0].legend(fontsize=8)
fig.suptitle("B0.5 f1: validation loss trajectories, all arms (u=+0.05)")
fig.tight_layout()
fig.savefig(FIGS / "f1_val_curves.png", dpi=130)
plt.close(fig)

# f2: H1 alignment for persistent arm (representative cell = first)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
seed, step = cells[0]
rowsA = step_by[(seed, step, "native")]
t_grid = [int(r["h"]) for r in rowsA]
lrA = [f(r["lr_native"]) for r in rowsA]
sA = b5.cumulative_lr_time(lrA, [1.0] * len(lrA))
h_grid = sorted(val_by[(seed, step, "native")])
sA_grid = [sA[h - 1] for h in h_grid]
L_A_vec = [val_by[(seed, step, "native")][h]["v_score"] for h in h_grid]
rowsB = step_by[(seed, step, "persistent")]
sB = b5.cumulative_lr_time(lrA, [f(r["multiplier"]) for r in rowsB])
L_B_vec = [val_by[(seed, step, "persistent")][h]["v_score"] for h in h_grid]
L_tilde, clamped = b5.align_loss_by_s(h_grid, sA_grid, L_A_vec,
                                      [sB[h - 1] for h in h_grid])
axes[0].plot(h_grid, L_B_vec, "o-", color=COL["B"], label="L_B (persistent)")
axes[0].plot(h_grid, L_A_vec, "k-", label="L_A (native)")
axes[0].plot([h for h, c in zip(h_grid, clamped) if not c],
             [x for x, c in zip(L_tilde, clamped) if not c],
             "x--", color="#8e44ad", label="L_A(s^-1(s_B(t)))")
axes[0].set_title(f"H1 alignment, {CELLS_LBL[0]}")
axes[0].set_xlabel("optimizer step h")
axes[0].legend(fontsize=8)
# pooled relative errors
relB = [r["alignment_rel_err"] for r in h1_rows if r["arm"] == "persistent"]
relC = [r["alignment_rel_err"] for r in h1_rows if r["arm"] == "pulse4_release"]
relD = [r["alignment_rel_err"] for r in h1_rows if r["arm"] == "D_transplant"]
xs = range(len(CELLS_LBL))
w = 0.27
axes[1].bar([x - w for x in xs], relB, width=w, color=COL["B"], label="B persistent")
axes[1].bar([x for x in xs], relC, width=w, color=COL["C"], label="C pulse4_release")
axes[1].bar([x + w for x in xs], relD, width=w, color=COL["D"], label="D transplant")
axes[1].axhline(1.0, color="k", ls="--", lw=1)
axes[1].axhline(0.5, color="gray", ls=":", lw=1)
axes[1].set_xticks(list(xs))
axes[1].set_xticklabels(CELLS_LBL, rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("alignment relative error (lower = better H1 fit)")
axes[1].set_title("H1 alignment error by cell (dashed=1: same-time baseline)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "f2_h1_alignment.png", dpi=130)
plt.close(fig)

# f3: distances vs native
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
hs = list(range(1, 65))
for arm, key in [("persistent", "B"), ("pulse4_release", "C"), ("D_transplant", "D")]:
    for metric, ax in zip(["d_theta", "d_m", "d_v"], axes):
        for (seed, step) in cells:
            ys = [f(r[metric]) for r in step_by[(seed, step, arm)]]
            ax.plot(hs, ys, color=COL[key], alpha=0.25, lw=0.8)
        med = [sorted(f(step_by[(s, t, arm)][i][metric]) for (s, t) in cells)[len(cells) // 2]
               for i in range(64)]
        ax.plot(hs, med, color=COL[key], lw=2.2, label=key)
for ax, metric in zip(axes, ["d_theta", "d_m", "d_v"]):
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step h")
    ax.set_ylabel(f"||{metric} - {metric}^A||_F")
    ax.set_title(metric)
    ax.legend(fontsize=8)
fig.suptitle("B0.5 f3: state separation from native (faint=cells, solid=median)")
fig.tight_layout()
fig.savefig(FIGS / "f3_distances.png", dpi=130)
plt.close(fig)

# f4: H2 curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
lags = list(range(1, 17))
for metric, ax in zip(["c_v_raw", "c_v_resid", "dir_persist"], axes):
    for (seed, step) in cells:
        ys = [f(r[metric]) for r in h2_by[(seed, step)]]
        ax.plot(lags, ys, color="#7f8c8d", alpha=0.4, lw=0.9)
    # median across cells per lag
    med = [sorted(f(h2_by[c][i][metric]) for c in cells)[len(cells) // 2]
           for i in range(16)]
    ax.plot(lags, med, "k-", lw=2.2)
    ax.axhline(1.0 / E, color="r", ls=":", lw=1)
    ax.set_xlabel("lag D (optimizer steps)")
    ax.set_title(metric)
    ax.set_ylim(-0.4, 1.05)
fig.suptitle("B0.5 f4: native-trajectory velocity structure (red dotted = 1/e)")
fig.tight_layout()
fig.savefig(FIGS / "f4_h2_curves.png", dpi=130)
plt.close(fig)

# f5: H3 channel decomposition
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
xs = range(len(CELLS_LBL))
w = 0.27
for i, H in enumerate((16, 64)):
    GB = [r[f"G_B_{H}"] for r in h3_rows]
    GC = [r[f"G_C_{H}"] for r in h3_rows]
    GD = [r[f"G_D_{H}"] for r in h3_rows]
    axes[i].bar([x - w for x in xs], GB, width=w, color=COL["B"], label="G_B persistent")
    axes[i].bar([x for x in xs], GC, width=w, color=COL["C"], label="G_C pulse4_release")
    axes[i].bar([x + w for x in xs], GD, width=w, color=COL["D"], label="G_D transplant")
    axes[i].set_xticks(list(xs))
    axes[i].set_xticklabels(CELLS_LBL, rotation=45, ha="right", fontsize=7)
    axes[i].set_title(f"gain vs native at h={H} (v_score)")
    axes[i].legend(fontsize=8)
fig.suptitle("B0.5 f5: H3 channel gains — G_C−G_D is the m/v-memory-channel gain")
fig.tight_layout()
fig.savefig(FIGS / "f5_h3_gains.png", dpi=130)
plt.close(fig)

# f6: cos_sep
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for arm, key in [("persistent", "B"), ("pulse4_release", "C"), ("D_transplant", "D")]:
    for (seed, step) in cells:
        ys = [f(r["cos_sep"]) if r["cos_sep"] not in ("", "None") else float("nan")
              for r in step_by[(seed, step, arm)]]
        ax.plot(hs, ys, color=COL[key], alpha=0.25, lw=0.8)
    med = [sorted(float(step_by[(s, t, arm)][i]["cos_sep"]) for (s, t) in cells)[len(cells) // 2]
           for i in range(64)]
    ax.plot(hs, med, color=COL[key], lw=2.2, label=key)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("optimizer step h")
ax.set_ylabel("cos(theta_arm - theta_A, native update)")
ax.set_title("B0.5 f6: direction of separation from native trajectory")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "f6_cos_sep.png", dpi=130)
plt.close(fig)

print("[analyzer] tables + figures written to", OUT)
