#!/usr/bin/env python3
"""A0S analyzer: controllability identification statistics.

D1 (offline): frozen A0O stage1 cells, H=4 (n=48).
D2: frozen A0Q chi values joined with a0s_run_v01 rollout losses (n=12).
Writes: runs/analysis.json, reports/a0s_controllability_table.csv (mandated
columns), d1_supplementary.csv, d2_detail.csv, plots PNG, A0S_REPORT.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0s_controllability_identification_v01"
RUNS = OUT / "runs"
REPORTS = OUT / "reports"
PLOTS = REPORTS / "plots"
A0O_RUNS = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01/runs"
A0Q_RUNS = PROJECT_ROOT / "outputs/a0q_response_control_map_v01/runs"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0s_controllability_identification_v01 as a0s  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_d1():
    rows = [json.loads(l) for l in (A0O_RUNS / "stage1_cells.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        if int(r["H"]) != a0s.H_FIXED:
            continue
        g = a0s.gain_row(r["h_eta"], r["L0"], r["L_plus"], r["L_minus"])
        out.append({"seed": r["seed"], "step": r["step"],
                    "stream_seed": r["stream_seed"], "chi_lr": r["h_eta"],
                    "chi_clip": 0.0, "chi_wd": 0.0,
                    "L0": r["L0"], "L_plus": r["L_plus"], "L_minus": r["L_minus"],
                    **g})
    return out


def load_d2():
    chi = {(r["seed"], r["step"], r["stream_seed"]): r
           for r in (json.loads(l) for l in (A0Q_RUNS / "cells.jsonl").read_text().splitlines())}
    out = []
    for r in (json.loads(l) for l in (RUNS / "d2_cells.jsonl").read_text().splitlines()):
        key = (r["seed"], r["step"], r["stream_seed"])
        assert key in chi, f"D2 cell {key} missing from frozen A0Q cells.jsonl"
        c = chi[key]
        assert c["crn_ok"] and r["crn_ok"], f"CRN failure in cell {key}"
        g = a0s.gain_row(c["h_lr"], r["L0"], r["L_plus"], r["L_minus"],
                         c["h_clip"], c["h_wd"])
        out.append({"seed": r["seed"], "step": r["step"],
                    "stream_seed": r["stream_seed"],
                    "chi_lr": c["h_lr"], "chi_clip": c["h_clip"],
                    "chi_wd": c["h_wd"],
                    "L0": r["L0"], "L_plus": r["L_plus"], "L_minus": r["L_minus"],
                    **g})
    return out


def a0s1_block(rows, label):
    C = [r["C_lr"] for r in rows]
    G = [r["G"] for r in rows]
    groups = [r["seed"] for r in rows]
    corr = a0s.correlations(C, G)
    fit = a0s.ols(C, G)
    loso = a0s.loso_ols(groups, C, G)
    return {"dataset": label, "n": len(rows), "corr": corr, "ols": fit,
            "loso": loso, "gate": a0s.gate_a0s1(corr, loso)}


def a0s2_block(rows, label):
    d = a0s.direction_stats([r["predicted_action"] for r in rows],
                            [r["best_action"] for r in rows])
    return {"dataset": label, **d, "gate": a0s.gate_a0s2(d)}


def scatter(ax, rows, title):
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        ax.scatter([r["C_lr"] for r in sub], [r["G"] for r in sub],
                   label=f"seed {seed}", s=40)
    C = np.array([r["C_lr"] for r in rows])
    G = np.array([r["G"] for r in rows])
    a, b = np.polyfit(C, G, 1)
    xs = np.linspace(C.min(), C.max(), 50)
    ax.plot(xs, a * xs + b, "k--", lw=1, label=f"OLS slope={b:.3f}")
    ax.set_xlabel("C = |chi_LR|")
    ax.set_ylabel("G = L0 - min(L+, L-)")
    ax.set_title(title)
    ax.legend(fontsize=8)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    d1 = load_d1()
    d2 = load_d2()
    assert len(d1) == 48 and len(d2) == 12, (len(d1), len(d2))

    # ---------------- A0S-1 ----------------
    a0s1_d1 = a0s1_block(d1, "D1_A0O_H4_bank_n48")
    a0s1_d2 = a0s1_block(d2, "D2_A0Q_bank_n12")

    # ---------------- A0S-2 ----------------
    a0s2_d1 = a0s2_block(d1, "D1_A0O_H4_bank_n48")
    a0s2_d2 = a0s2_block(d2, "D2_A0Q_bank_n12")

    # ---------------- A0S-3 ----------------
    shares = a0s.cj_shares(d2)
    model_ab = a0s.model_ab_loso([r["seed"] for r in d2], d2)
    clip_improves = bool(model_ab["model_B_all_channels"]["mse"]
                         < model_ab["model_A_lr_only"]["mse"])

    gates = {"A0S-1": a0s1_d1["gate"], "A0S-2_D1": a0s2_d1["gate"],
             "A0S-2_D2": a0s2_d2["gate"],
             "A0S-3_clip_improves_over_lr_only": clip_improves}
    verdict_pass = bool(a0s1_d1["gate"] and a0s2_d1["gate"] and a0s2_d2["gate"])
    verdict = {
        "PASS": verdict_pass,
        "fail_case": (None if verdict_pass else
                      ("FAIL_case_1_response_magnitude_useless"
                       if not a0s1_d1["gate"] else
                       "FAIL_case_direction_prediction")),
        "fail_case_2_note": ("A0S measures finite-H=4 controllability only; "
                             "A0P showed no stable temporal decay kernel, so "
                             "short-term G must NOT be read as long-term "
                             "training improvement."),
        "gates": gates,
    }

    analysis = {"a0s1": {"D1": a0s1_d1, "D2": a0s1_d2},
                "a0s2": {"D1": a0s2_d1, "D2": a0s2_d2},
                "a0s3": {"cj_shares": shares, "model_ab_loso": model_ab,
                         "clip_improves_prediction": clip_improves},
                "verdict": verdict,
                "n": {"D1": len(d1), "D2": len(d2)}}
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ---------------- mandated CSV (D2, exact column order) ----------------
    with open(REPORTS / "a0s_controllability_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "chi_lr", "chi_clip", "chi_wd", "J_norm",
                    "baseline_loss", "best_lr_loss", "gain", "best_action"])
        for r in sorted(d2, key=lambda r: (r["seed"], r["step"])):
            best_loss = r["L_plus"] if r["best_action"] == "+delta" else r["L_minus"]
            w.writerow([r["seed"], r["step"], f'{r["chi_lr"]:.8e}',
                        f'{r["chi_clip"]:.8e}', f'{r["chi_wd"]:.8e}',
                        f'{r["J_norm"]:.8e}', f'{r["L0"]:.8f}',
                        f'{best_loss:.8f}', f'{r["G"]:.8e}', r["best_action"]])

    # ---------------- supplementary tables ----------------
    with open(REPORTS / "d1_supplementary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "stream_seed", "h_eta", "C", "L0",
                    "L_plus", "L_minus", "G", "best_action", "predicted_action"])
        for r in sorted(d1, key=lambda r: (r["seed"], r["step"], r["stream_seed"])):
            w.writerow([r["seed"], r["step"], r["stream_seed"],
                        f'{r["chi_lr"]:.8e}', f'{r["C_lr"]:.8e}',
                        f'{r["L0"]:.8f}', f'{r["L_plus"]:.8f}',
                        f'{r["L_minus"]:.8f}', f'{r["G"]:.8e}',
                        r["best_action"], r["predicted_action"]])
    with open(REPORTS / "d2_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "stream_seed", "chi_lr", "chi_clip", "chi_wd",
                    "J_norm", "C_J", "L0", "L_plus", "L_minus", "G",
                    "best_action", "predicted_action"])
        for r in sorted(d2, key=lambda r: (r["seed"], r["step"], r["stream_seed"])):
            w.writerow([r["seed"], r["step"], r["stream_seed"],
                        f'{r["chi_lr"]:.8e}', f'{r["chi_clip"]:.8e}',
                        f'{r["chi_wd"]:.8e}', f'{r["J_norm"]:.8e}',
                        f'{r["C_J"]:.8e}', f'{r["L0"]:.8f}', f'{r["L_plus"]:.8f}',
                        f'{r["L_minus"]:.8f}', f'{r["G"]:.8e}',
                        r["best_action"], r["predicted_action"]])

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    scatter(axes[0], d1, "A0S-1  D1 (A0O H=4 bank, n=48)")
    scatter(axes[1], d2, "A0S-1  D2 (A0Q bank, n=12)")
    fig.tight_layout()
    fig.savefig(PLOTS / "a0s1_gain_vs_response.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    chans = ["lr", "clip", "wd"]
    axes[0].bar(chans, [shares[c]["median_share"] for c in chans])
    axes[0].set_title("A0S-3  median share of C_J = J^T J (D2)")
    axes[0].set_ylabel("median share")
    models = ["Model A (|chi_lr|)", "Model B (all 3)"]
    axes[1].bar(models, [model_ab["model_A_lr_only"]["mse"],
                         model_ab["model_B_all_channels"]["mse"]])
    axes[1].set_title("A0S-3  LOSO CV MSE (lower=better)")
    axes[1].set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(PLOTS / "a0s3_matrix_analysis.png", dpi=140)
    plt.close(fig)

    # ---------------- report ----------------
    def fmt_ci(ci):
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    rep = []
    rep.append("# A0S Report — Response-Based Controllability Identification\n")
    rep.append("Date: 2026-09-18. Mechanism validation experiment (NOT an "
               "optimization algorithm). STOP-after-A0S round.\n")
    rep.append("## 1. Hypothesis\n")
    rep.append("Finite-time controllability proxy $C_t = \\|J_u(t)\\|$ (response "
               "magnitude) contains usable controllability information: states "
               "with larger $C_t$ achieve larger best-intervention gain "
               "$G_t = L_{baseline} - \\min(L_{LR+}, L_{LR-})$, and the response "
               "sign predicts the useful action direction.\n")
    rep.append("## 2. Equations\n")
    rep.append("```\n"
               "u = log(eta/eta0), delta = 0.05 (frozen a0o), H = 4\n"
               "chi_LR = dL_score(Z_{t+H}) / du          (A0O/A0Q analytic adjoint)\n"
               "C_t = |chi_LR|                            (A0S-1)\n"
               "C_J = chi_lr^2 + chi_clip^2 + chi_wd^2    (A0S-3 Gramian proxy)\n"
               "G_t = L0 - min(L_plus, L_minus)\n"
               "best_action   = +delta if L_plus < L_minus else -delta\n"
               "predicted     = +delta if chi_LR < 0 else -delta\n"
               "Model A: G = alpha + beta |chi_lr|\n"
               "Model B: G = alpha + b1|chi_lr| + b2|chi_clip| + b3|chi_wd|\n"
               "```\n")
    rep.append("## 3. Data\n")
    rep.append(f"- **D1** (A0S-1 primary): frozen A0O stage1 cells, H=4, n={len(d1)} "
               "(seeds 2030/2031/2037 x steps 46/52/64/82 x streams 7110-7113). "
               "Analytic h_eta and rollout losses from the same record.\n")
    rep.append(f"- **D2** (A0S-2/A0S-3 + CSV): A0Q 12 cells (seeds 2030/2031/2037 x "
               "steps 46/82 x streams 7400/7401); chi from frozen A0Q records, "
               "rollout losses from a0s_run_v01 (CRN "
               f"{sum(1 for r in d2 if True)}/12).\n")
    rep.append("## 4. Statistics\n")
    for lbl, blk in (("D1", a0s1_d1), ("D2", a0s1_d2)):
        rep.append(f"### A0S-1 magnitude vs gain ({lbl}, n={blk['n']})\n")
        rep.append(f"- Pearson r = {blk['corr']['pearson']:+.4f} "
                   f"(p = {blk['corr']['pearson_p']:.2e})\n")
        rep.append(f"- Spearman rho = {blk['corr']['spearman']:+.4f} "
                   f"(p = {blk['corr']['spearman_p']:.2e})\n")
        rep.append(f"- OLS: beta = {blk['ols']['beta']:.4f}, "
                   f"95% CI {fmt_ci(blk['ols']['beta_ci95'])}, "
                   f"p = {blk['ols']['p']:.2e}, R^2 = {blk['ols']['r2']:.4f}\n")
        rep.append(f"- LOSO: pooled Pearson = {blk['loso']['pooled_pearson']}, "
                   f"pooled R^2 = {blk['loso']['pooled_r2']}, "
                   f"pooled MAE = {blk['loso']['pooled_mae']:.5f}\n")
        for f in blk["loso"]["folds"]:
            rep.append(f"  - fold held-out seed {f['held_out']}: "
                       f"beta={f['beta']:.4f}, pearson={f['pearson']}, "
                       f"mae={f['mae']:.5f}\n")
        rep.append(f"- **gate {'PASS' if blk['gate'] else 'FAIL'}**\n")
    for lbl, blk in (("D1", a0s2_d1), ("D2", a0s2_d2)):
        rep.append(f"### A0S-2 direction prediction ({lbl}, n={blk['n']})\n")
        rep.append(f"- accuracy = {blk['accuracy']:.4f} (chance 0.5), "
                   f"binomial p = {blk['binom_p']:.2e}\n")
        rep.append(f"- confusion: {blk['confusion']}\n")
        rep.append(f"- **gate {'PASS' if blk['gate'] else 'FAIL'}**\n")
    rep.append("### A0S-3 controllability matrix (D2, n=12)\n")
    rep.append(f"- median share of C_J: "
               + ", ".join(f"{c}={shares[c]['median_share']:.4f}" for c in chans)
               + "\n")
    rep.append(f"- LOSO Model A (|chi_lr|): MSE={model_ab['model_A_lr_only']['mse']:.3e}, "
               f"R^2={model_ab['model_A_lr_only']['r2']}\n")
    rep.append(f"- LOSO Model B (all 3):    MSE={model_ab['model_B_all_channels']['mse']:.3e}, "
               f"R^2={model_ab['model_B_all_channels']['r2']}\n")
    rep.append(f"- adding clip/wd improves prediction: **{clip_improves}**\n")
    rep.append("## 5. Plots\n")
    rep.append("![A0S-1](plots/a0s1_gain_vs_response.png)\n")
    rep.append("![A0S-3](plots/a0s3_matrix_analysis.png)\n")
    rep.append("## 6. PASS/FAIL decision\n")
    rep.append(f"```json\n{json.dumps(verdict, indent=1)}\n```\n")
    if verdict_pass:
        rep.append("\n**Verdict: PASS** — finite-H=4 response controllability "
                   "exists: J_u carries usable information about achievable "
                   "control gain, out-of-seed. Follow-on (explicit "
                   "authorization only, equal-wall-clock): response-guided "
                   "controller.\n")
    else:
        rep.append(f"\n**Verdict: FAIL** ({verdict['fail_case']}) — see rules.\n")
    rep.append("\n### FAIL case 2 statement (always reported)\n")
    rep.append(verdict["fail_case_2_note"] + "\n")
    rep.append("\n## 7. Restrictions honored\n")
    rep.append("No controller designed; no LR schedule tuned; no neural "
               "predictor; optimizer/lambda/training dynamics untouched; "
               "delta=0.05 and H=4 frozen from a0o/a0q; all frozen artifacts "
               "read-only; report covers all seeds and streams.\n")
    (REPORTS / "A0S_REPORT.md").write_text("".join(rep))

    print(json.dumps({"verdict": verdict,
                      "a0s1_D1": {k: a0s1_d1[k] for k in ("corr", "gate")},
                      "gpu": "see gpu_budget_log.json (driver)"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
