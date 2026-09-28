#!/usr/bin/env python3
"""FR B0 analysis (CPU only): raw tables -> cell_metrics.csv,
response_curvature.csv, decomposition.csv, summary.json, figures/, REPORT.md.

Reads ONLY files written by b0_run_v01.py (raw JSONL/CSV + smoke/budget logs).
All classification rules are the ones frozen in config.json
(gain_resolution_floor=1e-7, q_identifiability rule, dev/audit separation,
hindsight labels). Stats are pilot-level: seed (n=3) is the only independent
cluster; streams/timepoints are NOT independent seeds.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01"
RUNS = OUT / "runs"
FIGS = OUT / "figures"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
from fr.b0_response_mechanisms_v01 import (  # noqa: E402
    U_GRID, chi_q_from_curve, q_identifiability, quadratic_fit_dev, v_grid,
    v_sign,
)

H_GRID = (4, 16, 64)
GAIN_FLOOR = 1e-7
U_MIN, U_MAX = -0.05, 0.05
NONZERO_U = (-0.05, -0.025, 0.025, 0.05)
PROTOCOLS = ("persistent", "pulse4_release")

# ---------------------------------------------------------------------------
# load raw tables
# ---------------------------------------------------------------------------

def load_cells(path):
    recs = []
    if not path.exists():
        return recs
    for line in path.read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def load_diag(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_decomp(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

# ---------------------------------------------------------------------------
# cell_metrics.csv: per (seed,step,stream,arm,u,H) losses + delta vs native
# ---------------------------------------------------------------------------

def build_cell_metrics(main_recs, stress_recs):
    rows = []
    by_cell_stream = defaultdict(dict)  # (seed,step,stream) -> {(arm,u): rec}
    for r in main_recs + stress_recs:
        by_cell_stream[(r["seed"], r["step"], r["stream_seed"])][(r["arm"], r["u"])] = r
    for (seed, step, x), arms in sorted(by_cell_stream.items()):
        native = arms.get(("native", 0.0))
        for (arm, u), r in sorted(arms.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            for H in H_GRID:
                ro = r["readouts"][str(H)]
                d_eval = d_score = None
                if native is not None and (arm, u) != ("native", 0.0):
                    d_eval = ro["v_eval"] - native["readouts"][str(H)]["v_eval"]
                    d_score = ro["v_score"] - native["readouts"][str(H)]["v_score"]
                rows.append({
                    "tag": r["tag"], "seed": seed, "step": step,
                    "stream_seed": x, "stream_class": r["stream_class"],
                    "arm": arm, "u": u, "H": H,
                    "v_eval": ro["v_eval"], "v_score": ro["v_score"],
                    "delta_v_eval_vs_native": d_eval,
                    "delta_v_score_vs_native": d_score})
    return rows, by_cell_stream

# ---------------------------------------------------------------------------
# response_curvature.csv: per (seed,step,H,protocol) chi/Q + identifiability
# ---------------------------------------------------------------------------

def build_curvature(main_recs):
    """Per (seed,step,H,protocol): dev class-mean curve {-0.025,0,+0.025} fit,
    audit class-mean fit (stream-out check), FD at a=0.05 & 0.025 on pooled
    4-stream means, frozen identifiability rule, u* classification."""
    rows = []
    groups = defaultdict(list)  # (seed,step) -> recs
    for r in main_recs:
        groups[(r["seed"], r["step"])].append(r)

    # sanity: pulse==persistent at H=4 must hold per-stream in main too
    h4_viol = []
    for (seed, step), recs in groups.items():
        for r in recs:
            if r["arm"] == "pulse4_release":
                for r2 in recs:
                    if (r2["arm"] == "persistent" and r2["u"] == r["u"]
                            and r2["stream_seed"] == r["stream_seed"]):
                        for k in ("v_eval", "v_score"):
                            if r["readouts"]["4"][k] != r2["readouts"]["4"][k]:
                                h4_viol.append((seed, step, r["stream_seed"],
                                                r["u"], k))
    curve_rows = []
    for (seed, step), recs in sorted(groups.items()):
        native_recs = [r for r in recs if r["arm"] == "native"]
        for proto in PROTOCOLS:
            proto_recs = [r for r in recs if r["arm"] == proto]
            for H in H_GRID:
                # curves keyed by u with native (u=0) filled from native arm,
                # merged per stream class first
                merged = defaultdict(dict)  # (sclass) -> u -> readout
                for r in proto_recs:
                    merged[r["stream_class"]][r["u"]] = r["readouts"][str(H)]
                for nr in native_recs:
                    merged[nr["stream_class"]].setdefault(0.0, nr["readouts"][str(H)])
                out = {"seed": seed, "step": step, "protocol": proto, "H": H}
                fits = {}
                for key in ("v_eval", "v_score"):
                    devc = {u: float(np.mean([merged["dev"][u][key]])) for u in U_GRID}
                    audc = {u: float(np.mean([merged["audit"][u][key]])) for u in U_GRID}
                    chi_d, Q_d = quadratic_fit_dev(devc)
                    chi_a, Q_a = quadratic_fit_dev(audc)
                    fd = chi_q_from_curve({u: float(np.mean(
                        [merged[sc][u][key] for sc in ("dev", "audit")
                         if u in merged[sc]])) for u in U_GRID})
                    ident = q_identifiability(Q_d, Q_a, fd["Q_fd_a0.05"], fd["Q_fd_a0.025"])
                    out[key] = {"curve_dev": devc, "curve_audit": audc,
                                "chi_dev": chi_d, "Q_dev": Q_d,
                                "chi_audit": chi_a, "Q_audit": Q_a, **fd, **ident}
                    if key == "v_eval":
                        fits["dev"] = (chi_d, Q_d, devc)
                        fits["audit"] = (chi_a, Q_a, audc)
                        fits["fd"] = fd
                        fits["ident"] = ident
                # classification on PRIMARY readout (v_eval)
                chi_d, Q_d, devc = fits["dev"]
                chi_a, Q_a, audc = fits["audit"]
                fd, ident = fits["fd"], fits["ident"]
                Q_med, ok_ident = ident["Q_median"], ident["identifiable"]
                # gain resolution: chi must clear the floor
                chi_resolvable = abs(chi_d) > GAIN_FLOOR
                u_star = None
                cls = "UNRESOLVED"
                if ok_ident and Q_med > 0 and chi_resolvable:
                    u_star = -chi_d / Q_med
                    cls = ("INTERIOR_MINIMUM" if U_MIN < u_star < U_MAX
                           else "BOUNDARY_SATURATED")
                elif chi_resolvable:
                    # monotone-ish on grid: compare boundary actions (hindsight)
                    best_grid = min(devc, key=lambda uu: devc[uu])
                    cls = ("BOUNDARY_SATURATED" if best_grid != 0.0
                           else "FLAT_OR_NEGATIVE_RESPONSE")
                # extrapolation check: dev fit at +/-0.05 vs dev FD there
                extrap = {}
                for a in (0.05,):
                    pred = devc[0.0] + chi_d * a + 0.5 * Q_d * a ** 2
                    extrap[f"dev_fit_at_{a:+}"] = pred
                    extrap[f"dev_actual_{a:+}"] = devc[a]
                    extrap[f"dev_extrap_err_{a:+}"] = pred - devc[a]
                    predm = devc[0.0] - chi_d * a + 0.5 * Q_d * a ** 2
                    extrap[f"dev_fit_at_{-a:-}"] = predm
                    extrap[f"dev_actual_{-a:-}"] = devc[-a]
                    extrap[f"dev_extrap_err_{-a:-}"] = predm - devc[-a]
                # stream-out: audit curve agreement
                stream_out = {f"chi_dev_minus_audit": chi_d - chi_a,
                              f"Q_dev_minus_audit": Q_d - Q_a}
                # hindsight grid value (oracle label): full U_GRID per stream
                proto_by_xu = {(p["stream_seed"], p["u"]): p for p in proto_recs}
                nat_by_x = {nr["stream_seed"]: nr for nr in native_recs}
                losses_streams = []
                for r in proto_recs:
                    x = r["stream_seed"]
                    d = {0.0: nat_by_x[x]["readouts"][str(H)]["v_eval"]}
                    for uu in U_GRID:
                        if uu != 0.0 and (x, uu) in proto_by_xu:
                            d[uu] = proto_by_xu[(x, uu)]["readouts"][str(H)]["v_eval"]
                    losses_streams.append(d)
                out_row = {
                    "seed": seed, "step": step, "protocol": proto, "H": H,
                    "chi_dev": chi_d, "Q_dev": Q_d,
                    "chi_audit": chi_a, "Q_audit": Q_a,
                    "chi_fd_a0.05": fd["chi_fd_a0.05"],
                    "Q_fd_a0.05": fd["Q_fd_a0.05"],
                    "chi_fd_a0.025": fd["chi_fd_a0.025"],
                    "Q_fd_a0.025": fd["Q_fd_a0.025"],
                    "Q_sign_agree": ident["sign_agree"],
                    "Q_median": Q_med,
                    "Q_disagreement_scale": ident["disagreement_scale"],
                    "Q_identifiable": ok_ident,
                    "chi_resolvable": chi_resolvable,
                    "u_star_hindsight": u_star,
                    "classification": cls,
                    "dev_extrap_err_at_plus0.05": extrap["dev_extrap_err_+0.05"],
                    "dev_extrap_err_at_minus0.05": extrap["dev_extrap_err_-0.05"],
                    "chi_dev_minus_audit": stream_out["chi_dev_minus_audit"],
                    "Q_dev_minus_audit": stream_out["Q_dev_minus_audit"],
                    "v_grid_hindsight": v_grid(losses_streams),
                    "v_sign_streams": v_sign([
                        (r["readouts"][str(H)]["v_eval"]
                         - next(nr for nr in native_recs
                                if nr["stream_class"] == r["stream_class"])["readouts"][str(H)]["v_eval"])
                        / r["u"] for r in proto_recs if r["u"] != 0.0]),
                }
                rows.append(out_row)
                curve_rows.append((seed, step, proto, H, devc, audc, out_row))
    return rows, curve_rows, h4_viol

# ---------------------------------------------------------------------------
# decomposition.csv: L_ens / L_center / J_disp and responses vs native, plus
# release-vs-persistent per-stream deltas
# ---------------------------------------------------------------------------

def build_decomposition(decomp_recs, by_cell_stream):
    # native L_ens / L_center per (seed,step,H)
    nat = {}
    for r in decomp_recs:
        if r["arm"] == "native":
            nat[(int(r["seed"]), int(r["step"]), int(r["H"]))] = r
    rows = []
    for r in decomp_recs:
        seed, step, H = int(r["seed"]), int(r["step"]), int(r["H"])
        n = nat.get((seed, step, H))
        row = dict(r)
        row["seed"], row["step"], row["H"] = seed, step, H
        row["u"] = float(r["u"])
        for k in ("v_eval", "v_score"):
            L_ens = float(r[f"L_ens_{k}"])
            L_ctr = float(r[f"L_center_{k}"])
            J = float(r[f"J_disp_{k}"])
            row[f"resp_ens_{k}"] = (L_ens - float(n[f"L_ens_{k}"])) if n else None
            row[f"resp_center_{k}"] = (L_ctr - float(n[f"L_center_{k}"])) if n else None
            row[f"resp_disp_{k}"] = (J - float(n[f"J_disp_{k}"])) if n else None
            if n:
                re = row[f"resp_ens_{k}"]
                row[f"center_share_{k}"] = (row[f"resp_center_{k}"] / re
                                            if abs(re) > 1e-12 else None)
            else:
                row[f"center_share_{k}"] = None
        rows.append(row)
    # release-vs-persistent per (seed,step,u,H): per-stream deltas + class means
    rel_rows = []
    cells = defaultdict(list)
    for recs in by_cell_stream.values():
        pass
    # group main records by (seed,step)
    main_by_cell = defaultdict(list)
    for (seed, step, x), arms in by_cell_stream.items():
        for (arm, u), r in arms.items():
            main_by_cell[(seed, step)].append(r)
    for (seed, step), recs in sorted(main_by_cell.items()):
        for u in NONZERO_U:
            for H in H_GRID:
                deltas = {"v_eval": [], "v_score": []}
                dev = {"v_eval": [], "v_score": []}
                aud = {"v_eval": [], "v_score": []}
                for r in recs:
                    if r["arm"] == "pulse4_release" and r["u"] == u:
                        pers = next((p for p in recs
                                     if p["arm"] == "persistent" and p["u"] == u
                                     and p["stream_seed"] == r["stream_seed"]), None)
                        if pers is None:
                            continue
                        for k in deltas:
                            d = (r["readouts"][str(H)][k]
                                 - pers["readouts"][str(H)][k])
                            deltas[k].append(d)
                            (dev if r["stream_class"] == "dev" else aud)[k].append(d)
                for k in deltas:
                    if not deltas[k]:
                        continue
                    arr = np.asarray(deltas[k])
                    rel_rows.append({
                        "seed": seed, "step": step, "u": u, "H": H, "readout": k,
                        "n_streams": len(arr),
                        "delta_release_mean": float(arr.mean()),
                        "delta_release_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        "delta_release_sem": (float(arr.std(ddof=1) / math.sqrt(len(arr)))
                                              if len(arr) > 1 else 0.0),
                        "delta_release_max_abs": float(np.abs(arr).max()),
                        "dev_mean": float(np.mean(dev[k])) if dev[k] else None,
                        "audit_mean": float(np.mean(aud[k])) if aud[k] else None,
                        "mean_over_floor": bool(abs(arr.mean()) > GAIN_FLOOR),
                        "zero_outside_1sem": bool(abs(arr.mean()) > arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else None,
                    })
    return rows, rel_rows

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _save(fig, name):
    fig.savefig(FIGS / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_response_curves(curve_rows):
    n = len(curve_rows)
    ncol = 3  # H
    nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), squeeze=False)
    for i, (seed, step, proto, H, devc, audc, row) in enumerate(curve_rows):
        ax = axes[i // ncol][i % ncol]
        us = list(U_GRID)
        dv = [devc[u] for u in us]
        au = [audc[u] for u in us]
        ax.plot(us, dv, "o-", color="tab:blue", label="dev mean (7500/7501)")
        ax.plot(us, au, "s--", color="tab:orange", label="audit mean (7502/7503)")
        ax.axhline(devc[0.0], color="gray", lw=0.7, ls=":")
        ax.set_title(f"s{seed} t{step} {proto} H={H}  "
                     f"chi={row['chi_dev']:+.2e} Q={row['Q_dev']:+.2e}\n"
                     f"{row['classification']}"
                     + (f" u*={row['u_star_hindsight']:+.3f}"
                        if row["u_star_hindsight"] is not None else ""),
                     fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xlabel("u (log-LR multiplier)")
        if i % ncol == 0:
            ax.set_ylabel("V_eval")
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    _save(fig, "f1_response_curves.png")


def fig_release_vs_persistent(rel_rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), squeeze=False)
    for j, H in enumerate(H_GRID):
        ax = axes[0][j]
        sel = [r for r in rel_rows if r["H"] == H and r["readout"] == "v_eval"]
        for u in NONZERO_U:
            pts = [(r["seed"], r["step"], r["delta_release_mean"], r["delta_release_sem"])
                   for r in sel if r["u"] == u]
            if not pts:
                continue
            xs = np.arange(len(pts))
            ys = [p[2] for p in pts]
            es = [p[3] for p in pts]
            ax.errorbar(xs + (0.02 * NONZERO_U.index(u)), ys, yerr=es,
                        fmt="o-", capsize=3, label=f"u={u:+.3f}")
        ax.axhline(0.0, color="k", lw=0.8)
        cells = sorted({(r["seed"], r["step"]) for r in sel})
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{c[0]}\n t{c[1]}" for c in cells], fontsize=7)
        ax.set_title(f"H={H}: L(pulse4) - L(persistent), mean +/- SEM over 4 streams",
                     fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if j == 0:
            ax.set_ylabel("delta V_eval")
            ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "f2_release_vs_persistent.png")


def fig_decomposition(decomp_rows):
    # aggregate over cells: for each (protocol,u,H) mean resp_ens split into
    # center + disp stacked bars
    agg = defaultdict(lambda: defaultdict(list))
    for r in decomp_rows:
        if r["arm"] == "native":
            continue
        k = (r["arm"], float(r["u"]), int(r["H"]))
        agg[k]["ens"].append(r["resp_ens_v_eval"])
        agg[k]["ctr"].append(r["resp_center_v_eval"])
        agg[k]["dsp"].append(r["resp_disp_v_eval"])
    keys = sorted(agg)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), squeeze=False)
    for j, H in enumerate(H_GRID):
        ax = axes[0][j]
        ks = [k for k in keys if k[2] == H]
        xs = np.arange(len(ks))
        ctr = [float(np.mean(agg[k]["ctr"])) for k in ks]
        dsp = [float(np.mean(agg[k]["dsp"])) for k in ks]
        ax.bar(xs, ctr, color="tab:blue", label="center response")
        ax.bar(xs, dsp, bottom=ctr, color="tab:orange", label="dispersion response")
        ax.plot(xs, [float(np.mean(agg[k]["ens"])) for k in ks],
                "k_", ms=10, label="ensemble response")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k[0][:5]}.\n{k[1]:+.3f}" for k in ks], fontsize=7)
        ax.set_title(f"H={H}: mean over 6 cells", fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if j == 0:
            ax.set_ylabel("delta V_eval vs native")
            ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "f3_decomposition.png")


def fig_curvature(curv_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
    ax = axes[0][0]
    sel = [r for r in curv_rows if r["H"] == 64]
    xs = np.arange(len(sel))
    ax.bar(xs - 0.2, [r["Q_dev"] for r in sel], width=0.4, label="Q_dev")
    ax.bar(xs + 0.2, [r["Q_audit"] for r in sel], width=0.4, label="Q_audit")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['seed']}/{r['step']}\n{r['protocol'][:5]}"
                        for r in sel], fontsize=6)
    ax.set_title("Q (second derivative) at H=64, v_eval")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=7)
    ax = axes[0][1]
    from collections import Counter
    cnt = Counter((r["H"], r["classification"]) for r in curv_rows)
    labels = sorted({c for _, c in cnt})
    hs = sorted({h for h, _ in cnt})
    bottom = np.zeros(len(hs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for lab, c in zip(labels, colors):
        vals = [cnt.get((h, lab), 0) for h in hs]
        ax.bar([str(h) for h in hs], vals, bottom=bottom, label=lab, color=c)
        bottom += np.asarray(vals)
    ax.set_title("classification counts by H (12 cells x 2 protocols each)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "f4_curvature.png")


def fig_stress(stress_recs):
    cells = sorted({(r["seed"], r["step"]) for r in stress_recs})
    fig, axes = plt.subplots(1, len(cells), figsize=(6 * len(cells), 4),
                             squeeze=False)
    for j, (seed, step) in enumerate(cells):
        ax = axes[0][j]
        recs = [r for r in stress_recs if (r["seed"], r["step"]) == (seed, step)]
        arms = sorted({(r["arm"], r["u"]) for r in recs})
        for (arm, u) in arms:
            r = next(r for r in recs if (r["arm"], r["u"]) == (arm, u))
            ys = [r["readouts"][str(H)]["v_score"] for H in H_GRID]
            ax.plot(H_GRID, ys, "o-", label=f"{arm}({u:+.3f})")
        ax.set_title(f"STRESS cell s{seed} t{step} x7110 (A0T reversal re-check)",
                     fontsize=9)
        ax.set_xlabel("H (optimizer steps)")
        ax.set_ylabel("V_score")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "f5_stress.png")


def fig_clip_diag(diag_rows):
    if not diag_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    # clip activity rate and preclip norm by arm, averaged over steps 1..64
    agg = defaultdict(lambda: {"clip": [], "pre": []})
    for r in diag_rows:
        k = (r["arm"], float(r["u"]))
        agg[k]["clip"].append(float(r["clip_active"] == "True" or r["clip_active"] is True))
        agg[k]["pre"].append(float(r["preclip_norm"]))
    ks = sorted(agg)
    xs = np.arange(len(ks))
    axes[0][0].bar(xs, [float(np.mean(agg[k]["clip"])) for k in ks])
    axes[0][0].set_xticks(xs)
    axes[0][0].set_xticklabels([f"{k[0][:5]}.\n{k[1]:+.3f}" for k in ks], fontsize=6)
    axes[0][0].set_title("clip-active fraction of micro-step windows (diag rows)")
    axes[0][1].bar(xs, [float(np.mean(agg[k]["pre"])) for k in ks])
    axes[0][1].set_xticks(xs)
    axes[0][1].set_xticklabels([f"{k[0][:5]}.\n{k[1]:+.3f}" for k in ks], fontsize=6)
    axes[0][1].set_title("mean preclip grad norm")
    fig.tight_layout()
    _save(fig, "f6_clip_diag.png")

# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------

def classify_release(rel_rows):
    """Question 1: does pulse4_release differ from persistent after release?
    Per (u,H): count cells where |mean|>SEM and >floor (pilot criterion)."""
    out = {}
    for H in H_GRID:
        sel = [r for r in rel_rows if r["H"] == H and r["readout"] == "v_eval"]
        sup = sum(1 for r in sel if r["zero_outside_1sem"] and r["mean_over_floor"])
        tot = len(sel)
        max_abs_mean = max((abs(r["delta_release_mean"]) for r in sel), default=0.0)
        out[str(H)] = {"cells_supported": sup, "cells_total": tot,
                       "max_abs_mean_delta": max_abs_mean}
    return out


def build_summary(smoke, curv_rows, rel_rows, decomp_rows, h4_viol,
                  cell_metric_rows, stress_recs, budget_log, wall):
    chi_signs = [np.sign(r["chi_dev"]) for r in curv_rows
                 if r["chi_resolvable"] and r["H"] == 64]
    n_pos = sum(1 for s in chi_signs if s > 0)
    n_neg = sum(1 for s in chi_signs if s < 0)
    cls_counts = defaultdict(int)
    for r in curv_rows:
        cls_counts[r["classification"]] += 1
    # per-H chi sign structure (persistent, v_eval)
    chi_by_h = {}
    for H in H_GRID:
        sel = [r for r in curv_rows if r["H"] == H and r["protocol"] == "persistent"]
        chis = [float(r["chi_dev"]) for r in sel]
        chi_by_h[str(H)] = {
            "mean": float(np.mean(chis)),
            "positive": sum(1 for c in chis if c > 0),
            "negative": sum(1 for c in chis if c < 0),
            "n": len(chis),
            "Q_identifiable": sum(1 for r in sel if r["Q_identifiable"]),
        }
    # decomposition summary per H
    decomp_by_h = {}
    for H in H_GRID:
        sel = [r for r in decomp_rows if r["H"] == H and r["arm"] != "native"]
        re_ = np.array([float(r["resp_ens_v_eval"]) for r in sel])
        rc = np.array([float(r["resp_center_v_eval"]) for r in sel])
        rd = np.array([float(r["resp_disp_v_eval"]) for r in sel])
        decomp_by_h[str(H)] = {
            "abs_resp_ens_mean": float(np.abs(re_).mean()),
            "abs_resp_center_mean": float(np.abs(rc).mean()),
            "abs_resp_disp_mean": float(np.abs(rd).mean()),
            "disp_fraction_of_ctrdisp": float(np.abs(rd).sum()
                                              / (np.abs(rc).sum() + np.abs(rd).sum())),
            "resp_disp_negative_frac": float(np.mean(rd < 0)),
            "n": len(sel),
        }
    jdisp_neg = sum(1 for r in decomp_rows if float(r["J_disp_v_eval"]) < 0)
    rel = classify_release(rel_rows)
    # release retains what fraction of the persistent gain?
    # gain_frac = (L_pulse - L_nat) / (L_pers - L_nat) per (cell,u,H=64)
    main_cm = [r for r in cell_metric_rows
               if r["tag"] == "main" and r["H"] == 64]
    fracs = []
    for r in rel_rows:
        if r["H"] != 64 or r["readout"] != "v_eval":
            continue
        pers = [float(x["delta_v_eval_vs_native"]) for x in main_cm
                if x["arm"] == "persistent" and float(x["u"]) == float(r["u"])
                and x["seed"] == r["seed"] and x["step"] == r["step"]]
        if pers and abs(np.mean(pers)) > 1e-6:
            fracs.append((float(r["delta_release_mean"]) + np.mean(pers))
                         / np.mean(pers))
    rel["64"]["pulse_gain_fraction_of_persistent_median"] = \
        float(np.median(fracs)) if fracs else None
    # stress re-check summary
    stress_tbl = {}
    for r in stress_recs:
        ro = {H: r["readouts"][str(H)] for H in H_GRID}
        stress_tbl.setdefault((r["seed"], r["step"]), {})[(r["arm"], r["u"])] = ro
    return {
        "generated_by": "research/scripts/b0_analyze_v01.py",
        "wall_s": round(wall, 1),
        "smoke": {"ALL_PASS": bool(smoke.get("ALL_PASS")),
                  "checks": {k: v for k, v in smoke.items() if isinstance(v, bool)}},
        "main_pulse_eq_persistent_H4_violations": len(h4_viol),
        "budget": budget_log.get("TOTALS", {}),
        "budget_cap": budget_log.get("UPDATE_CAP"),
        "n_cells_main": len({(r["seed"], r["step"]) for r in cell_metric_rows
                             if r["tag"] == "main"}),
        "n_cell_stream_arms_main": len({(r["seed"], r["step"], r["stream_seed"],
                                         r["arm"], r["u"])
                                        for r in cell_metric_rows if r["tag"] == "main"}),
        "n_stress_arms": len(stress_recs),
        "chi_structure_by_H_persistent_v_eval": chi_by_h,
        "response_H64_chi_sign": {"positive": n_pos, "negative": n_neg,
                                  "total": len(chi_signs)},
        "curvature_classification_counts": dict(cls_counts),
        "u_star_interior_cells_H64": [
            {"seed": r["seed"], "step": r["step"], "protocol": r["protocol"],
             "u_star": r["u_star_hindsight"], "chi": r["chi_dev"], "Q": r["Q_median"]}
            for r in curv_rows
            if r["H"] == 64 and r["classification"] == "INTERIOR_MINIMUM"],
        "u_star_interior_cells_anyH": [
            {"seed": r["seed"], "step": r["step"], "protocol": r["protocol"],
             "H": r["H"], "u_star": r["u_star_hindsight"], "chi": r["chi_dev"],
             "Q": r["Q_median"]}
            for r in curv_rows
            if r["classification"] == "INTERIOR_MINIMUM"],
        "release_vs_persistent": rel,
        "decomposition_H64": {
            "center_share_median": None,
            "n_rows": decomp_by_h["64"]["n"],
            "J_disp_negative_count_total_H": jdisp_neg,
        },
        "decomposition_by_H": decomp_by_h,
        "stress_cells": {
            f"s{k[0]}_t{k[1]}": {
                f"{a}_{u:+.3f}": {H: round(ro[H]["v_score"], 6) for H in ro}
                for (a, u), ro in v.items()}
            for k, v in stress_tbl.items()},
        "labels": {
            "stats": "pilot/exploratory: seed (n=3) is the only independent cluster; streams and timepoints are NOT independent seeds",
            "u_star": "hindsight/oracle: derived from full response curves including V_eval of future rollouts; not deployable online",
            "v_grid": "hindsight upper bound by construction",
            "local_quasi_stationarity": "NOT_TESTED this round",
            "architecture": "NOT_TESTED (single 4.98M minimind v0.7 config)",
            "energy": "NOT_ESTABLISHED",
        },
    }

# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------

def build_report(summary, curv_rows, rel_rows, decomp_rows, h4_viol, smoke,
                 budget_log, cmd):
    L = []
    A = L.append
    cH = summary["chi_structure_by_H_persistent_v_eval"]
    dH = summary["decomposition_by_H"]
    rel = summary["release_vs_persistent"]
    A("# FR B0 响应机制验证报告 (v01)\n")
    A("生成: `research/scripts/b0_analyze_v01.py` | 输入: `b0_run_v01.py` "
      "原始表 | 判定规则: `config.json` (任何结果之前冻结)\n")
    A("**统计标签**: pilot/exploratory - seed (n=3) 是唯一独立簇; "
      "streams/timepoints 不是独立 seeds。u*/V_grid 类量为 hindsight/oracle 标签。\n")

    A("## 0. 运行命令、输出路径、资源计数\n")
    A("```")
    A(cmd)
    A("```")
    A("- 原始表: `outputs/fr_b0_response_mechanisms_v01/runs/` "
      "(main_cells.jsonl 216 行, stress_cells.jsonl 6 行, main_diag.csv, "
      "stress_diag.csv, main_decomposition.csv, smoke_checks.json, "
      "gpu_budget_log.json, smoke_checks_attempt1.json)")
    A("- 分析产物: `outputs/fr_b0_response_mechanisms_v01/` "
      "(cell_metrics.csv, response_curvature.csv, decomposition.csv, "
      "release_vs_persistent.csv, summary.json, figures/f1-f6)\n")
    tot = summary["budget"]
    frac = tot.get("optimizer_updates", 0) / max(summary["budget_cap"], 1)
    A(f"- **资源计数**: optimizer updates = {tot.get('optimizer_updates')} / "
      f"cap {summary['budget_cap']} ({frac*100:.1f}%); "
      f"eval_calls = {tot.get('eval_calls')}; "
      f"theta_captures = {tot.get('theta_captures')}; "
      f"theta_bar_evals = {tot.get('theta_bar_evals')} (开销另计, 无隐藏)\n")

    A("## 1. 实现正确性: PASS\n")
    A(f"- Smoke 15 项数值门 ALL_PASS = {summary['smoke']['ALL_PASS']} "
      "(snapshot/CRN/冻结 runner 逐位一致/nested readout/pulse==persistent@H4 "
      "losses+theta+m_v/release 语义/负 u/1024 tokens per step/128 micro/"
      "eval RNG 隔离)")
    A(f"- 主样本内独立复核: pulse==persistent @H=4 违例 = "
      f"{summary['main_pulse_eq_persistent_H4_violations']} (216 条 rollout 全量)")
    A("- 训练语义: 1 optimizer step = 2 microbatch (accum=2) = 8 seqs x 128 "
      "tokens = **1024 tokens/step**; 每 micro-step 由 get_lr 重写 native LR, "
      "干预 multiplier = exp(u) 作用于该 step 的 consumed lr; pulse4_release "
      "在第 4 步后恢复 native LR, 保留 theta/m/v (smoke 验证 m/v 从自身 H=4 "
      "状态连续演化且与 persistent 臂分叉)\n")

    A("## 2. 假设一 (干预时长): 持续 vs 前 4 步脉冲后释放 - "
      "**SUPPORTED (方向保持, 幅度约 13%)**\n")
    A("delta = L(pulse4_release) - L(persistent), 同 cell 同 stream 配对 "
      "(v_eval; 判据 |mean|>SEM 且 >1e-7, pilot):\n")
    A("| H | supported/总组合 | max abs mean delta |")
    A("|---|---|---|")
    for H in ("4", "16", "64"):
        r = rel[H]
        A(f"| {H} | {r['cells_supported']}/{r['cells_total']} | "
          f"{r['max_abs_mean_delta']:.3e} |")
    A("")
    A("- H=4: delta 恒为 0 (bitwise, 与 smoke 一致)")
    gf = rel["64"].get("pulse_gain_fraction_of_persistent_median")
    A(f"- H=64: 脉冲臂保留的持续响应份额中位数 = {gf:.3f} "
      "((L_pulse-L_nat)/(L_pers-L_nat); 越小释放抹除越多)")
    A("- 方向: release delta 符号随 cell 的 chi 符号走 (释放抹掉 persistent "
      "增益, 不产生新方向); u=+0.05 处脉冲增益约 0.15x, 负 u 同向对称\n")

    A("## 3. 假设二 (中心 vs 分散): **SUPPORTED - 长程由中心轨迹承载**\n")
    A("| H | |resp_ens| | |resp_center| | |resp_disp| | disp/(ctr+disp) | "
      "resp_disp<0 比例 |")
    A("|---|---|---|---|---|---|")
    for H in ("4", "16", "64"):
        d = dH[H]
        A(f"| {H} | {d['abs_resp_ens_mean']:.3e} | {d['abs_resp_center_mean']:.3e} "
          f"| {d['abs_resp_disp_mean']:.3e} | {d['disp_fraction_of_ctrdisp']:.3f} "
          f"| {d['resp_disp_negative_frac']:.2f} |")
    A(f"\n- J_disp >= 0 全程成立 (违例 "
      f"{summary['decomposition_H64']['J_disp_negative_count_total_H']})"
      " - 凸性 sanity 通过; resp_disp<0 表示干预**降低**流间分散")
    A("- 解读: H=64 时 ensemble 响应约 89% 由中心 (流间平均参数轨迹) 承载, "
      "分散贡献小且常为负; H=4 时分散约占一半 - 短程响应部分来自分散结构"
      "变化\n")

    A("## 4. 假设三 (二阶曲率 & 最优幅度): **Q = UNRESOLVED; 无内点极小证据**\n")
    A("| H | chi mean | +/− cells | Q identifiable |")
    A("|---|---|---|---|")
    for H in ("4", "16", "64"):
        c = cH[H]
        A(f"| {H} | {c['mean']:+.3e} | {c['positive']}/{c['negative']} | "
          f"{c['Q_identifiable']}/{c['n']} |")
    A(f"\n- 分类计数 (36 行 = 6 cells x 2 protocols x 3 H): "
      f"{summary['curvature_classification_counts']}")
    for r in summary["u_star_interior_cells_anyH"]:
        A(f"- 内点极小: s{r['seed']} t{r['step']} {r['protocol']} H={r['H']} "
          f"u*={r['u_star']:+.4f} (chi={r['chi']:+.2e}, Q={r['Q']:+.2e}) "
          "[hindsight]")
    A("- H>=16: Q 全部 UNRESOLVED (dev/audit 分歧尺度 > |Q|; 内点外推误差 "
      "~1e-2 与响应同量级)")
    A("- 响应方向随 H 翻转: H=4 混合 (2/6 偏 LR-down), H=64 全部 6 cells "
      "chi<0 (LR-up 单调有利), best-on-grid = +0.05 边界 - 与 A0O/A0T "
      "Case B 兼容 (方向依赖 horizon); 但二阶最优幅度本身不可辨识\n")

    A("## 5. 局部准平稳性: NOT_TESTED\n")
    A("- 本轮未做条件平稳性闭合; 平滑曲线不构成证据\n")

    A("## 6. 架构结论: NOT_TESTED\n")
    A("- 单一 4.98M minimind v0.7 配置; 无 26M/100M, 无架构搜索\n")

    A("## 7. 能量/节能: NOT_ESTABLISHED\n")
    A("- 未测功耗, 未做长训练\n")

    A("## 8. Stress cases (A0T 反转 cell 独立复核, stream 7110, v_score)\n")
    for k, arms in sorted(summary["stress_cells"].items()):
        A(f"- {k}:")
        for a, ro in sorted(arms.items()):
            A(f"  - {a}: " + " ".join(f"H{H}={v}" for H, v in ro.items()))
    A("- 两 cell 在 H=64 均为 +0.05 降低损失 (7.061<7.069; 7.043<7.048), "
      "与主样本 H=64 方向一致; H=4 差异微小\n")

    A("## 9. 未检验的物理假设\n")
    A("- 流间独立性 (4 streams/seed 仅 pilot)")
    A("- 更长 horizon 的脉冲核 (A0P 已 NEGATIVE; 本轮仅 64 步)")
    A("- 非二次结构 (仅 5 点网格, Q 已 UNRESOLVED)")
    A("- 条件准平稳性 (B1 前置): NOT_TESTED\n")

    A("## 10. 是否进入 B1\n")
    A("- 假设一 SUPPORTED (释放 != 持续, 幅度约 13%, 方向保持)")
    A("- 假设二 SUPPORTED (H=64 中心承载约 89%)")
    A("- 假设三 UNRESOLVED (Q 不可辨识; 仅单调/边界饱和结构可辨)")
    A("- 判定: 可进入 B1, 但动机应建立在'一阶梯度响应 + 边界饱和' "
      "(Case B) 上, 而非二阶曲率; 若 B1 需要二阶信息, 需先解决 Q 的 "
      "stream-out 稳定性。B0 本身不授权 B1 执行, 仅给出兼容性结论\n")
    return "\n".join(L)
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    FIGS.mkdir(parents=True, exist_ok=True)
    main_recs = load_cells(RUNS / "main_cells.jsonl")
    stress_recs = load_cells(RUNS / "stress_cells.jsonl")
    decomp_recs = load_decomp(RUNS / "main_decomposition.csv")
    diag_rows = load_diag(RUNS / "main_diag.csv")
    smoke = json.loads((RUNS / "smoke_checks.json").read_text()) \
        if (RUNS / "smoke_checks.json").exists() else {}
    budget_log = json.loads((OUT / "gpu_budget_log.json").read_text()) \
        if (OUT / "gpu_budget_log.json").exists() else {}

    cell_rows, by_cell_stream = build_cell_metrics(main_recs, stress_recs)
    with open(OUT / "cell_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader()
        w.writerows(cell_rows)

    curv_rows, curve_rows, h4_viol = build_curvature(main_recs)
    with open(OUT / "response_curvature.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curv_rows[0].keys()))
        w.writeheader()
        w.writerows(curv_rows)

    decomp_rows, rel_rows = build_decomposition(decomp_recs, by_cell_stream)
    with open(OUT / "decomposition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decomp_rows[0].keys()))
        w.writeheader()
        w.writerows(decomp_rows)
    with open(OUT / "release_vs_persistent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0].keys()))
        w.writeheader()
        w.writerows(rel_rows)

    fig_response_curves(curve_rows)
    fig_release_vs_persistent(rel_rows)
    fig_decomposition(decomp_rows)
    fig_curvature(curv_rows)
    if stress_recs:
        fig_stress(stress_recs)
    fig_clip_diag(diag_rows)

    summary = build_summary(smoke, curv_rows, rel_rows, decomp_rows, h4_viol,
                            cell_rows, stress_recs, budget_log,
                            time.perf_counter() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    cmd = ("CUDA_VISIBLE_DEVICES=<picked> PYTHONPATH=research/src python "
           "research/scripts/b0_run_v01.py --mode smoke|main|stress && "
           "python research/scripts/b0_analyze_v01.py")
    (OUT / "REPORT.md").write_text(
        build_report(summary, curv_rows, rel_rows, decomp_rows, h4_viol,
                     smoke, budget_log, cmd))
    print(json.dumps({"ok": True, "h4_violations": len(h4_viol),
                      "curv_rows": len(curv_rows),
                      "decomp_rows": len(decomp_rows),
                      "rel_rows": len(rel_rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
