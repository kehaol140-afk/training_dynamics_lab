#!/usr/bin/env python3
"""GTTR-R2B 图（<=4 张；只读已保存 CSV）。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/b205/projects/fluctuation_response/outputs/"
           "fr_gttr_r2b_solved_update_bridge_v01")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

with (OUT / "gains_vs_horizon.csv").open() as f:
    gains = list(csv.DictReader(f))
with (OUT / "token_counts.csv").open() as f:
    toks = list(csv.DictReader(f))
with (OUT / "control_step1.csv").open() as f:
    ctrl = list(csv.DictReader(f))

pairs = sorted({(r["seed"], r["start"], r["stream"]) for r in gains})
lab = {p: f"{p[0]}/{p[1]}/st{p[2][-2:]}" for p in pairs}

# fig1: G_struct(h) 曲线（逐配对 + 均值）
fig, ax = plt.subplots(figsize=(7.2, 4.4))
hs = sorted({int(r["h"]) for r in gains})
for p in pairs:
    vs = [(int(r["h"]), float(r["G_struct_LRminusFULL"])) for r in gains
          if (r["seed"], r["start"], r["stream"]) == p]
    vs.sort()
    ax.plot([v[0] for v in vs], [v[1] for v in vs], lw=0.6, alpha=0.45)
mean_by_h = [np.mean([float(r["G_struct_LRminusFULL"]) for r in gains
                      if int(r["h"]) == h]) for h in hs]
ax.plot(hs, mean_by_h, "k-", lw=2.2, label="mean over 12 pairs")
ax.axhline(0, color="r", ls="--", lw=0.8)
ax.set_xlabel("h (optimizer updates)")
ax.set_ylabel("G_struct(h) = V_LR - V_FULL")
ax.set_xscale("log")
ax.legend()
ax.set_title("Structure gain vs horizon (one-shot R1 solution)")
fig.tight_layout()
fig.savefig(FIG / "fig1_gain_vs_horizon.png", dpi=140)
plt.close(fig)

# fig2: h256 G_struct 逐配对条形
g256 = [(lab[p], float(r["G_struct_LRminusFULL"])) for r in gains
        for p in [(r["seed"], r["start"], r["stream"])]
        if int(r["h"]) == 256]
g256.sort()
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.bar([g[0] for g in g256], [g[1] for g in g256],
       color=["tab:green" if g[1] > 0 else "tab:red" for g in g256])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("G_struct(256)")
ax.set_xticklabels([g[0] for g in g256], rotation=60, fontsize=7)
ax.set_title("Terminal structure gain per pair")
fig.tight_layout()
fig.savefig(FIG / "fig2_terminal_gain.png", dpi=140)
plt.close(fig)

# fig3: h1 ctrl FULL-LR 与 h1 monitor G_struct
c1 = defaultdict(dict)
for r in ctrl:
    if r["h"] == "1":
        c1[(r["seed"], r["start"], r["stream"])][r["arm"]] = float(r["mean"])
xs = np.arange(len(pairs))
fml = [c1[p]["GTTR_FULL"] - c1[p]["GTTR_LR_ONLY"] for p in pairs]
g1 = [float(r["G_struct_LRminusFULL"]) for r in gains if r["h"] == "1"]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.bar(xs - 0.18, fml, 0.36, label="ctrl h1: FULL-LR")
ax.bar(xs + 0.18, g1, 0.36, label="monitor h1: LR-FULL")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([lab[p] for p in pairs], rotation=60, fontsize=7)
ax.legend()
ax.set_title("Immediate one-step effect (negative ctrl / positive monitor"
             " favor FULL)")
fig.tight_layout()
fig.savefig(FIG / "fig3_immediate_h1.png", dpi=140)
plt.close(fig)

# fig4: 检测 h_detect 三臂
fig, ax = plt.subplots(figsize=(7.2, 4.0))
arms = ("BASE_K2", "GTTR_LR_ONLY", "GTTR_FULL")
for i, arm in enumerate(arms):
    hd = [int(r["h_detect"]) for r in toks
          if r["arm"] == arm and r["h_detect"] != "NA"]
    ax.scatter([i] * len(hd), hd, s=28, alpha=0.75)
    ax.annotate(f"median {np.median(hd):.0f}", (i, np.median(hd)),
                fontsize=8, ha="center",
                xytext=(i + 0.18, np.median(hd)))
ax.set_xticks(range(3))
ax.set_xticklabels(arms)
ax.set_ylabel("h_detect (16-step grid)")
ax.set_title("Detection step by arm (12 pairs each)")
fig.tight_layout()
fig.savefig(FIG / "fig4_detection.png", dpi=140)
plt.close(fig)
print("FIGURES_DONE")
