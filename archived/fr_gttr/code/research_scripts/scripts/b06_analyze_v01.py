#!/usr/bin/env python3
"""FR B0.6 analyzer: Q3a (PRIMARY) switching linear response, Q1 (supporting)
regime structure vs phase+drift nulls, Q2 chi-by-regime, Q3b occupancy-mixture
chi, retained-gain descriptive leg, theory tests T1-T6 (full settings).

Inputs (all frozen before results):
  runs/gate.json              — offline CRN gate, must be ALL_PASS
  runs/psi_main.csv           — 24 native trajectories x 64 steps (recorder)
  runs/psi_stress.csv         — 2 stress trajectories (robustness only)
  frozen B0 response_curvature.csv  — chi_dev/chi_audit per cell/protocol/H
  frozen B0 main_diag.csv           — native + pulse4_release(u=+0.05) losses

Outputs: analysis.json, theory_tests.json, figures/fig*.png, summary.json
Verdict composition follows config.json thresholds_frozen_before_results,
including the headline_rule (Q3a headline; if Q1 NOT_SUPPORTED -> Q3 NOT_TESTED).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b06_regime_dynamics_v01"
RUNS = OUT / "runs"
FIGS = OUT / "figures"
FROZEN_B0 = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import fr.b06_regime_dynamics_v01 as b06  # noqa: E402

SEED = 20260921
GAIN_RESOLUTION_FLOOR = 1e-7  # frozen in config.json

Q1_RULES = {"min_mean_run_length": 8}
Q2_RULES = {"alpha": 0.05, "min_seed_consistent": 2, "n_perm": 20000,
            "seed": SEED}
Q3A_RULES = {"across_weight": 1.0}
Q3B_RULES = {"min_seed_folds": 2, "resolution": 1e-12}


# ---------------------------------------------------------------------------
# JSON sanitizing (numpy arrays in switching-map regime parameters)
# ---------------------------------------------------------------------------

def _safe(v):
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, (np.ndarray,)):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _f(x):
    return None if x == "" else float(x)


def load_psi_by_traj(path):
    """CSV -> {(seed, step, stream): [per-step dicts ordered by h]}.

    upd_cos: recorder CSV has update_cos_prev empty at h=1; the channel
    convention (frozen, same as g_cos) is 0.0 when no previous vector exists.
    """
    by_traj = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), int(row["step"]), int(row["stream_seed"]))
            by_traj.setdefault(key, []).append({
                "h": int(row["h"]),
                "grad_norm": float(row["grad_norm"]),
                "update_norm": float(row["update_norm"]),
                "train_loss": float(row["train_loss"]),
                "preclip_norm": float(row["preclip_norm"]),
                "lr_native": float(row["lr_native"]),
                "clip_active": row["clip_active"] == "True",
                "m_norm": float(row["m_norm"]),
                "v_norm": float(row["v_norm"]),
                "g_cos": float(row["g_cos"]),
                "upd_cos": _f(row["update_cos_prev"]) or 0.0,
                "disp": float(row["disp"]),
            })
    for rows in by_traj.values():
        rows.sort(key=lambda r: r["h"])
    return by_traj


def load_chi_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"seed": int(r["seed"]), "step": int(r["step"]),
                         "protocol": r["protocol"], "H": int(r["H"]),
                         "chi_dev": float(r["chi_dev"]),
                         "chi_audit": float(r["chi_audit"])})
    return rows


def load_frozen_diag_losses(path):
    """Frozen B0 main_diag -> per-arm per-step train_loss dicts.

    Returns {(seed, step, stream, u): [rows by h]} for arm in
    {native, pulse4_release}."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] not in ("native", "pulse4_release"):
                continue
            key = (int(r["seed"]), int(r["step"]), int(r["stream_seed"]),
                   float(r["u"]))
            out.setdefault(key, []).append({"h": int(r["h"]),
                                            "train_loss": float(r["train_loss"])})
    for rows in out.values():
        rows.sort(key=lambda r: r["h"])
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figs(figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = plt.figure

    # fig1: psi channels, representative trajectory (2030, 46, 7500)
    ch = figs["channels_example"]
    names = list(b06.CLUSTER_CHANNELS)
    f, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    h = np.arange(1, len(ch["log_g"]) + 1)
    for ax, name in zip(axes.flat, names):
        ax.plot(h, ch[name], lw=0.9)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 1].set_xlabel("h")
    f.suptitle("psi_t channels — native replay (2030, 46, stream 7500)")
    f.tight_layout(rect=[0, 0, 1, 0.96])
    f.savefig(FIGS / "fig1_psi_channels.png", dpi=150)
    plt.close(f)

    # fig2: HMM k-grid test loglik + null bands
    q1 = figs["q1"]
    ks = [r["k"] for r in q1["hmm_table"]]
    ll = [r["test_ll"] for r in q1["hmm_table"]]
    f, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ll, "o-", label="HMM held-out test LL")
    ax.axhline(q1["phase_null"]["test_ll"]["p95"], color="r", ls="--",
               label="phase-null p95")
    ax.axhline(q1["drift_null"]["test_ll"]["p95"], color="g", ls="--",
               label="drift-null p95")
    ax.set_xlabel("K"); ax.set_ylabel("test loglik")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Q1: HMM k-grid vs null bands")
    f.tight_layout(); f.savefig(FIGS / "fig2_hmm_k_grid.png", dpi=150)
    plt.close(f)

    # fig3: Viterbi paths, all main trajectories
    f, ax = plt.subplots(figsize=(11, 5))
    for i, (tid, path) in enumerate(sorted(q1["paths"].items())):
        ax.step(range(1, len(path) + 1), [p + 1.05 * i for p in path],
                where="post", lw=0.9)
        ax.text(65.5, 1.05 * i + 0.5, f"{tid[0]}/{tid[1]}/{tid[2]}",
                fontsize=6, va="center")
    ax.set_xlabel("h"); ax.set_yticks([])
    ax.set_title("Q1: Viterbi regime paths (main, offset per trajectory)")
    f.tight_layout(); f.savefig(FIGS / "fig3_viterbi_paths.png", dpi=150)
    plt.close(f)

    # fig4: persistence vs nulls
    pers = q1["persistence"]
    f, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(["data", "phase-null\n(med)", "drift-null\n(med)"],
                [pers["mean_run_length"],
                 q1["phase_null"]["mean_run_length"]["median"],
                 q1["drift_null"]["mean_run_length"]["median"]],
                color=["C0", "C1", "C2"])
    axes[0].set_ylabel("mean run length"); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(["data", "phase-null\n(med)", "drift-null\n(med)"],
                [pers["switch_rate"],
                 q1["phase_null"]["switch_rate"]["median"],
                 q1["drift_null"]["switch_rate"]["median"]],
                color=["C0", "C1", "C2"])
    axes[1].set_ylabel("switch rate"); axes[1].grid(alpha=0.3, axis="y")
    f.suptitle("Q1: persistence vs nulls")
    f.tight_layout(rect=[0, 0, 1, 0.94])
    f.savefig(FIGS / "fig4_persistence_nulls.png", dpi=150)
    plt.close(f)

    # fig5: Q3a held-out one-step MSE
    m = figs["q3a"]["test_mse"]
    f, ax = plt.subplots(figsize=(6, 4))
    labels = ["local\nwithin", "local\nacross", "global\nwithin", "global\nacross"]
    vals = [m["local_within"], m["local_across"], m["global_within"],
            m["global_across"]]
    vals = [np.nan if v is None else v for v in vals]
    ax.bar(labels, vals, color=["C0", "C3", "C1", "C4"])
    ax.set_ylabel("held-out one-step MSE")
    ax.set_title(f"Q3a: local vs global map (best_k={q1['best_k']})")
    ax.grid(alpha=0.3, axis="y")
    f.tight_layout(); f.savefig(FIGS / "fig5_q3a_maps.png", dpi=150)
    plt.close(f)

    # fig6: retained gain decay + half-life vs first transition
    rg = figs["retained_gain"]
    f, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for row in rg["rows"][:24]:
        pass
    if rg["curves"]:
        for key, g in list(rg["curves"].items())[:12]:
            axes[0].plot(sorted(g), [g[h] for h in sorted(g)], lw=0.8)
    axes[0].set_xlabel("h"); axes[0].set_ylabel("g(h) = loss_pulse - loss_native")
    axes[0].set_title("pulse4_release (+0.05) retained gain")
    axes[0].grid(alpha=0.3)
    if rg["corr"]["spearman_rho"] is not None:
        hl = [r["half_life"] for r in rg["rows"]
              if abs(r["g5"]) >= GAIN_RESOLUTION_FLOOR]
        ft = [r["first_transition"] for r in rg["rows"]
              if abs(r["g5"]) >= GAIN_RESOLUTION_FLOOR]
        axes[1].plot(ft, hl, "o")
        axes[1].set_xlabel("first Viterbi transition h")
        axes[1].set_ylabel("gain half-life h")
        axes[1].set_title(
            f"Spearman rho={rg['corr']['spearman_rho']:.3f} "
            f"(p={rg['corr']['spearman_p']:.3g}, n={rg['corr']['n']}, descriptive)")
        axes[1].grid(alpha=0.3)
    f.tight_layout(); f.savefig(FIGS / "fig6_retained_gain.png", dpi=150)
    plt.close(f)

    # fig7: Q2/Q3b — chi by dominant regime + LOCO errors
    q2 = figs["q2"]; q3b = figs["q3b"]
    f, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for row in q2["rows"]:
        if "perm_p" in row and row["H"] == 16:
            axes[0].bar([f"A(reg {row['group_A']})", f"B(reg {row['group_B']})"],
                        [row["mean_A"], row["mean_B"]])
            axes[0].set_title(
                f"Q2 chi by dominant regime (H=16, {row['protocol']})\n"
                f"perm p={row['perm_p']:.3g}")
            break
    axes[0].set_ylabel("mean chi"); axes[0].grid(alpha=0.3, axis="y")
    wa = q3b.get("within_across_median_abs_err", {})
    if wa:
        cats = ["within", "across"]
        loc = [wa["within_cluster"]["local"], wa["across_transition"]["local"]]
        glo = [wa["within_cluster"]["global"], wa["across_transition"]["global"]]
        x = np.arange(2)
        axes[1].bar(x - 0.15, [np.nan if v is None else v for v in loc], 0.3,
                    label="local (occupancy mixture)")
        axes[1].bar(x + 0.15, [np.nan if v is None else v for v in glo], 0.3,
                    label="global (constant)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylabel("median |err| (LOCO-CV)")
    axes[1].set_title("Q3b: local vs global chi prediction")
    f.tight_layout(); f.savefig(FIGS / "fig7_q2_q3b_chi.png", dpi=150)
    plt.close(f)


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.perf_counter()
    FIGS.mkdir(parents=True, exist_ok=True)

    gate = json.loads((RUNS / "gate.json").read_text())
    if not gate.get("ALL_PASS"):
        print("GATE not PASS — analysis refused on ungated data.", flush=True)
        return 2

    psi_main = load_psi_by_traj(RUNS / "psi_main.csv")
    psi_stress = load_psi_by_traj(RUNS / "psi_stress.csv")
    assert len(psi_main) == 24, f"expected 24 main trajectories, got {len(psi_main)}"
    assert len(psi_stress) == 2

    channels_by_traj = {k: b06.psi_channels_from_rows(v)
                        for k, v in psi_main.items()}
    channels_stress = {k: b06.psi_channels_from_rows(v)
                       for k, v in psi_stress.items()}

    # ---- Q1 (supporting): regime structure vs BOTH nulls ------------------
    print("[1/6] Q1 regime analysis (HMM k-grid + phase/drift nulls)...",
          flush=True)
    q1 = b06.q1_regime_analysis(channels_by_traj, seed=SEED)
    v1 = b06.q1_verdict(q1, Q1_RULES)

    # standardized mats identical to q1's internal scaling (deterministic)
    ids = sorted(channels_by_traj.keys())
    mats = [b06.channels_to_matrix(channels_by_traj[i]) for i in ids]
    scaler = b06.Standardizer().fit(mats)
    z_mats = [scaler.transform(m) for m in mats]
    paths_list = [np.asarray(q1["paths"][i]) for i in ids]

    # stress robustness: transform with MAIN scaler, decode with MAIN HMM
    stress_mats = [b06.channels_to_matrix(channels_stress[k])
                   for k in sorted(channels_stress)]
    z_stress = [scaler.transform(m) for m in stress_mats]
    full_hmm = b06.GaussianHMM(q1["best_k"], seed=SEED + 1000 * q1["best_k"]).fit(z_mats)
    stress_paths = {k: full_hmm.viterbi(X)[0].tolist()
                    for k, X in zip(sorted(channels_stress), z_stress)}
    stress_pers = b06.persistence_stats(list(stress_paths.values()))

    # ---- Q3a (PRIMARY): switching linear response vs global map -----------
    print("[2/6] Q3a switching linear maps (PRIMARY)...", flush=True)
    if q1["best_k"] >= 2:
        sw = b06.fit_switching_linear_maps(z_mats, paths_list, q1["best_k"])
        v3a = b06.q3a_verdict(sw, Q3A_RULES)
        # robustness: within-cell standardization variant
        wc_mats = []
        for m in mats:
            mu, sd = m.mean(axis=0), m.std(axis=0)
            sd[sd < 1e-12] = 1.0
            wc_mats.append((m - mu) / sd)
        sw_wc = b06.fit_switching_linear_maps(wc_mats, paths_list, q1["best_k"])
        v3a_wc = b06.q3a_verdict(sw_wc, Q3A_RULES)
    else:
        sw, v3a = None, {"verdict": "NOT_TESTED",
                         "reason": "best_k<2 (headline_rule)"}
        sw_wc, v3a_wc = None, {"verdict": "NOT_TESTED"}
    q3a = {"switch_res": sw, "verdict": v3a,
           "within_cell_variant": {"verdict": v3a_wc,
                                   "test_mse": None if sw_wc is None
                                   else sw_wc["test_mse"]}}

    # headline composition per frozen headline_rule
    headline = q3a["verdict"]["verdict"]
    if v1["verdict"] == "NOT_SUPPORTED" and headline != "NOT_TESTED":
        headline = "NOT_TESTED"
        q3a["headline_note"] = ("Q1 NOT_SUPPORTED -> Q3a relabeled NOT_TESTED "
                                "per frozen headline_rule")

    # ---- Q2 (supporting): chi by regime ------------------------------------
    print("[3/6] Q2 chi by regime (permutation)...", flush=True)
    chi_rows = load_chi_rows(FROZEN_B0 / "response_curvature.csv")
    occ_by_cell = {}
    for (seed, step, stream) in ids:
        cell = (seed, step)
        occ_by_cell.setdefault(cell, []).append(paths_list[ids.index((seed, step, stream))])
    occ_by_cell = {c: b06.occupancy_from_paths(ps, q1["best_k"])
                   for c, ps in occ_by_cell.items()}
    q2 = b06.q2_chi_by_regime(chi_rows, occ_by_cell, Q2_RULES)

    # ---- Q3b: occupancy-mixture vs constant chi ----------------------------
    print("[4/6] Q3b local vs global chi (LOCO-CV, frozen a0c _metric)...",
          flush=True)
    q3b = b06.q3b_local_vs_global_chi(chi_rows, occ_by_cell,
                                      lambda c: c[0], Q3B_RULES)
    if v1["verdict"] == "NOT_SUPPORTED" and q3b["verdict"] != "NOT_TESTED":
        q3b["verdict_relabeled"] = "NOT_TESTED"
        q3b["verdict"] = "NOT_TESTED"

    # ---- retained-gain descriptive leg -------------------------------------
    print("[5/6] retained-gain descriptive leg...", flush=True)
    diag = load_frozen_diag_losses(FROZEN_B0 / "runs" / "main_diag.csv")
    pulse_rows = {k: v for k, v in diag.items()
                  if abs(k[3] - 0.05) < 1e-12}
    native_rows = {}
    for (s, t, x, _u) in pulse_rows:
        native_rows[(s, t, x, 0.05)] = diag[(s, t, x, 0.0)]
    tpoints = {k: b06.transition_points(q1["paths"][k]) for k in ids}
    rg = b06.retained_gain_decay(pulse_rows, native_rows, tpoints)
    below = [r for r in rg["rows"] if abs(r["g5"]) < GAIN_RESOLUTION_FLOOR]
    rg["below_resolution_floor"] = {"floor": GAIN_RESOLUTION_FLOOR, "n": len(below),
                                    "rows": below}
    rg["curves"] = {f"{k[0]}/{k[1]}/{k[2]}": {h: loss for h, loss in
                   [(r["h"], r["train_loss"]) for r in v]}
                   for k, v in pulse_rows.items()}
    for r in rg["rows"]:
        r["below_resolution"] = bool(abs(r["g5"]) < GAIN_RESOLUTION_FLOOR)

    # ---- theory tests T1-T6 (full settings) --------------------------------
    print("[6/6] theory tests T1-T6 (full settings)...", flush=True)
    theory = b06.run_all_theory_tests()
    (RUNS / "theory_tests.json").write_text(json.dumps(_safe(theory), indent=1))

    # ---- assemble + figures -------------------------------------------------
    analysis = _safe({
        "seed": SEED,
        "n_main_trajectories": len(ids),
        "traj_ids": [list(i) for i in ids],
        "q1": q1, "q1_verdict": v1,
        "q3a_primary": q3a, "headline_verdict": headline,
        "occupancy_by_cell": {f"{c[0]}/{c[1]}": occ for c, occ in occ_by_cell.items()},
        "q2": q2,
        "q3b": {"verdict": q3b["verdict"], "metric": q3b["metric"],
                "seed_cmp": q3b["seed_cmp"],
                "n_seed_favor_local": q3b["n_seed_favor_local"],
                "n_seed_total": q3b["n_seed_total"],
                "within_across_median_abs_err": q3b["within_across_median_abs_err"],
                "fold_rows": q3b["fold_rows"]},
        "stress_robustness": {"paths": stress_paths,
                              "persistence": stress_pers},
        "retained_gain": {k: v for k, v in rg.items() if k != "curves"},
        "theory_tests_ok": bool(theory["ALL_OK"]),
        "wall_s": round(time.perf_counter() - t_start, 1),
    })
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    make_figs({"channels_example": channels_by_traj[(2030, 46, 7500)],
               "q1": q1, "q3a": {"test_mse": (q3a["switch_res"] or {}).get(
                   "test_mse", {})},
               "retained_gain": rg, "q2": q2, "q3b": q3b})

    summary = {"headline_verdict": headline,
               "q1_verdict": v1["verdict"],
               "q2_verdict": q2["verdict"],
               "q3a_verdict": q3a["verdict"]["verdict"],
               "q3b_verdict": q3b["verdict"],
               "best_k": q1["best_k"],
               "theory_tests_ok": bool(theory["ALL_OK"]),
               "wall_s": analysis["wall_s"]}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
