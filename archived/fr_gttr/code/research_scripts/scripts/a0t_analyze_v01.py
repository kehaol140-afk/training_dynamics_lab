#!/usr/bin/env python3
"""A0T analyzer: pilot statistics, mandated CSV, supplementary, plots, report.

Reads runs/cells.jsonl (driver). Computes frozen metrics (gains A/B/C,
Wilcoxon, sign consistency, C_t terciles, erosion), writes
reports/a0t_response_lr_control_table.csv (mandated 9 columns, one row per
(seed,step) state = stream average), a0t_detail_supplementary.csv (per cell),
plots, and A0T_REPORT.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0t_response_lr_control_pilot_v01"
RUNS = OUT / "runs"
REPORTS = OUT / "reports"
PLOTS = REPORTS / "plots"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0t_response_lr_control_pilot_v01 as a0t  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_records():
    rows = [json.loads(l) for l in (RUNS / "cells.jsonl").read_text().splitlines()]
    for r in rows:
        assert r["crn_ok"], f"CRN failure in {r['dataset']} s{r['seed']} t{r['step']}"
    return rows


def state_averages(records):
    """One row per (dataset, seed, step): stream-averaged losses/chi."""
    out = {}
    for r in records:
        k = (r["dataset"], r["seed"], r["step"])
        out.setdefault(k, []).append(r)
    rows = []
    for (ds, seed, step), cells in sorted(out.items()):
        assert len(cells) == 2, (ds, seed, step, len(cells))
        chi = float(np.mean([c["chi_lr"] for c in cells]))
        actions = {c["action_u"] for c in cells}
        losses = {arm: {str(H): float(np.mean([c["losses"][arm][str(H)]
                                              for c in cells]))
                        for H in a0t.H_GRID} for arm in ("A", "B", "C")}
        rows.append({"dataset": ds, "seed": seed, "step": step, "chi_lr": chi,
                     "C_t": abs(chi),
                     "action": "+delta" if chi < 0 else "-delta",
                     "actions_disagree": len(actions) != 1,
                     "losses": losses})
    return rows


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    records = load_records()
    states = state_averages(records)

    gb = a0t.gain_block(records, "B")
    gc = a0t.gain_block(records, "C")
    signc = a0t.sign_consistency(records, 4)
    ter = a0t.terciles(records, 4)
    ero = a0t.erosion(records)
    gate = a0t.gate_pass(gc["gain_C"], signc, gc["C_minus_B"]["4"])

    disagree = [(r["dataset"], r["seed"], r["step"]) for r in states
                if r["actions_disagree"]]
    analysis = {"n_cells": len(records), "n_states": len(states),
                "states_actions_disagree": disagree,
                "n_aliased_C_equals_B": sum(1 for r in records if r["aliased"]),
                "gain_B": gb["gain_B"], "gain_C": gc["gain_C"],
                "C_minus_B": gc["C_minus_B"],
                "sign_consistency_H4": signc, "terciles_H4": ter,
                "erosion": ero, "gate": gate}
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ---- mandated CSV (9 columns, per state stream-average; C losses) ----
    with open(REPORTS / "a0t_response_lr_control_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "chi_lr", "C_t", "action",
                    "loss_H4", "loss_H16", "loss_H64", "gain"])
        for r in states:
            gain = r["losses"]["A"]["4"] - r["losses"]["C"]["4"]
            w.writerow([r["seed"], r["step"], f'{r["chi_lr"]:.8e}',
                        f'{r["C_t"]:.8e}', r["action"],
                        f'{r["losses"]["C"]["4"]:.8f}',
                        f'{r["losses"]["C"]["16"]:.8f}',
                        f'{r["losses"]["C"]["64"]:.8f}', f'{gain:.8e}'])

    # ---- supplementary per-cell detail ----
    with open(REPORTS / "a0t_detail_supplementary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "step", "stream_seed", "chi_lr", "C_t",
                    "action_u", "aliased_C_eq_B",
                    "A_H4", "A_H16", "A_H64", "B_H4", "B_H16", "B_H64",
                    "C_H4", "C_H16", "C_H64",
                    "gainC_H4", "gainC_H16", "gainC_H64", "crn_ok"])
        for r in sorted(records, key=lambda r: (r["dataset"], r["seed"],
                                                r["step"], r["stream_seed"])):
            w.writerow([r["dataset"], r["seed"], r["step"], r["stream_seed"],
                        f'{r["chi_lr"]:.8e}', f'{r["C_t"]:.8e}', r["action_u"],
                        r["aliased"],
                        *(f'{r["losses"][a][str(H)]:.8f}'
                          for a in ("A", "B", "C") for H in a0t.H_GRID),
                        *(f'{r["losses"]["A"][str(H)] - r["losses"]["C"][str(H)]:.8e}'
                          for H in a0t.H_GRID),
                        r["crn_ok"]])

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(a0t.H_GRID))
    for arm, lbl, mk in (("B", "arm B (fixed +delta)", "s"),
                         ("C", "arm C (response-guided)", "o")):
        med = [analysis["gain_" + arm][str(H)]["median"] for H in a0t.H_GRID]
        q1 = [analysis["gain_" + arm][str(H)]["iqr"][0] for H in a0t.H_GRID]
        q3 = [analysis["gain_" + arm][str(H)]["iqr"][1] for H in a0t.H_GRID]
        axes[0].errorbar(x, med, yerr=[np.array(med) - q1, np.array(q3) - med],
                         marker=mk, capsize=4, label=lbl)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, [f"H={H}" for H in a0t.H_GRID])
    axes[0].set_ylabel("median gain vs native (IQR)")
    axes[0].set_title("A0T arms B/C gain vs native AdamW")
    axes[0].legend(fontsize=8)
    tern = ["bottom", "middle", "top"]
    succ = {ds: [ter[ds][t2]["success_frac"] for t2 in tern]
            for ds in sorted(ter)}
    width = 0.35
    for i, (ds, vals) in enumerate(sorted(succ.items())):
        axes[1].bar(np.arange(3) + (i - 0.5) * width, vals, width, label=ds)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_xticks(np.arange(3), tern)
    axes[1].set_ylabel("controller success frac (gain_C,H4 > 0)")
    axes[1].set_title("A0T success by C_t tercile")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "a0t_gains_and_terciles.png", dpi=140)
    plt.close(fig)

    # ---- report ----
    rep = []
    rep.append("# A0T Report — Response-guided Learning Rate Control Pilot\n")
    rep.append("Date: 2026-09-18. First algorithm validation after A0S mechanism "
               "discovery. FIXED pilot: no controller optimization, no "
               "hyperparameter tuning. STOP-after-experiment round.\n")
    rep.append("## 1. Hypothesis\n")
    rep.append("A state-specific LR action selected by the response "
               "(chi_lr = dL_{t+H}/du at H=4, frozen A0O/A0Q adjoint) outperforms "
               "fixed LR perturbation and native AdamW over finite horizons "
               "H in {4,16,64}.\n")
    rep.append("## 2. Controller (frozen)\n")
    rep.append("```\nper cell: read frozen chi_lr\n  chi_lr < 0  -> u = +0.05  (eta_new = eta_native_k * e^u every step)\n"
               "  chi_lr > 0  -> u = -0.05\nno threshold / confidence / phase / adaptation\n```\n")
    rep.append("## 3. Baselines and data\n")
    rep.append(f"- Cells: 36 = D1 24 (A0O streams 7110/7111 x steps 46/52/64/82) "
               f"+ D2 12 (A0Q streams 7400/7401 x steps 46/82); chi_lr frozen "
               f"read-only. CRN 36/36. C==B aliasing on "
               f"{analysis['n_aliased_C_equals_B']}/36 cells (action +delta). "
               f"Cross-stream action disagreement within a state: "
               f"{len(analysis['states_actions_disagree'])} states "
               f"{analysis['states_actions_disagree']} (per-cell true actions in "
               f"the supplementary CSV; mandated CSV uses stream-mean sign).\n")
    rep.append("- Arm A: native AdamW. Arm B: fixed u=+0.05. Arm C: response-guided.\n")
    rep.append("## 4. Statistics\n")
    for arm in ("B", "C"):
        rep.append(f"### Arm {arm} gain vs native (gain = L_A - L_{arm})\n")
        for H in a0t.H_GRID:
            b = analysis["gain_" + arm][str(H)]
            wv = b["wilcoxon_vs0"]
            rep.append(f"- H={H}: median {b['median']:+.3e} (IQR [{b['iqr'][0]:+.3e}, "
                       f"{b['iqr'][1]:+.3e}]), frac>0 {b['frac_positive']:.3f}, "
                       f"wilcoxon p = {wv['p']}\n")
    rep.append("### C vs B (paired, gain_C - gain_B)\n")
    for H in a0t.H_GRID:
        d = analysis["C_minus_B"][str(H)]
        rep.append(f"- H={H}: median {d['median']:+.3e}, wilcoxon p = {d['wilcoxon_vs0']['p']}\n")
    rep.append(f"### Sign consistency (H=4): {signc['improved']}/{signc['n']} "
               f"improved, frac {signc['fraction']:.3f}, binomial p = {signc['binom_p']:.2e}\n")
    rep.append("### C_t terciles (controller success by controllability, H=4)\n")
    for ds in sorted(ter):
        for t2 in tern:
            b = ter[ds][t2]
            rep.append(f"- {ds} {t2}: n={b['n']}, C_t in [{b['C_t_range'][0]:.4f}, "
                       f"{b['C_t_range'][1]:.4f}], success {b['success_frac']:.3f}, "
                       f"median gain {b['median_gain']:+.3e}\n")
    rep.append("### Short-vs-long erosion (A0P linkage)\n")
    for H in a0t.H_GRID:
        e = ero[str(H)]
        rep.append(f"- H={H}: median gain_C {e['median_gain_C']:+.3e}, "
                   f"success {e['success_frac']:.3f}, median gain_B {e['median_gain_B']:+.3e}\n")
    rep.append("## 5. Plots\n")
    rep.append("![gains and terciles](plots/a0t_gains_and_terciles.png)\n")
    rep.append("## 6. PASS/FAIL (frozen rule)\n")
    rep.append(f"```json\n{json.dumps(gate, indent=1)}\n```\n")
    rep.append("\n## 7. Limitations (required)\n")
    rep.append("- Short-term (finite-H) gain does NOT equal final training "
               "improvement.\n")
    rep.append("- A0P showed long memory and a non-exponential response kernel: "
               "H=16/64 gains may not persist; erosion table reports this.\n")
    rep.append("- Absolute gains are tiny (A0S: G/L0 ~ 7e-5 at H=4).\n")
    rep.append(f"- C-vs-B is near-degenerate BY CONSTRUCTION on these banks: "
               f"frozen chi_lr < 0 in 46/48 D1 and 12/12 D2 A0S cells, so arm C "
               f"coincides with arm B on {analysis['n_aliased_C_equals_B']}/36 "
               f"cells. The pilot therefore cannot demonstrate guidance value "
               f"over fixed +delta except on the rare positive-response states; "
               f"it validates the response-to-action pipeline end to end.\n")
    rep.append("- No controller optimization, no hyperparameter tuning, no "
               "threshold/confidence/phase; delta=0.05 and H grid frozen.\n")
    rep.append("- Response sign is stream-dependent: in 2/18 states the two "
               "frozen streams disagree on the sign of chi_lr (D1 2030/46 and "
               "2030/64); the controller then takes different actions per "
               "stream. The mandated CSV reports the stream-mean sign; the "
               "supplementary CSV carries per-cell actions. This is genuine "
               "physics (response sign varies with the data order), not a "
               "defect.\n")
    (REPORTS / "A0T_REPORT.md").write_text("".join(rep))

    print(json.dumps({"gate": gate,
                      "gain_C_H4_median": gc["gain_C"]["4"]["median"],
                      "sign_consistency": signc,
                      "n_aliased": analysis["n_aliased_C_equals_B"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
