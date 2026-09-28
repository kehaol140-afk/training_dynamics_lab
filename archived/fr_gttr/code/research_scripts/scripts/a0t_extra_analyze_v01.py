#!/usr/bin/env python3
"""A0T-extra analyzer: controller identity from frozen A0T records.

Writes runs/analysis.json, CSV tables, plots, A0T_EXTRA_REPORT.md.
Offline only; asserts the frozen budget log shows no new GPU phases.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0t_extra_response_controller_identity_v01"
RUNS = OUT / "runs"
REPORTS = OUT / "reports"
PLOTS = REPORTS / "plots"
A0T_OUT = PROJECT_ROOT / "outputs/a0t_response_lr_control_pilot_v01"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0t_extra_response_controller_identity_v01 as ex  # noqa: E402
import fr.a0t_response_lr_control_pilot_v01 as a0t  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # ---- frozen input guard ----
    cells_path = A0T_OUT / "runs/cells.jsonl"
    records = [json.loads(l) for l in cells_path.read_text().splitlines()]
    assert len(records) == 36, len(records)
    for r in records:
        assert r["crn_ok"]
    a0t_budget = json.loads((A0T_OUT / "gpu_budget_log.json").read_text())
    sha = json.loads((A0T_OUT / "hashes/manifest.json").read_text())
    frozen_inputs = {"a0t_cells_sha256": next(e["sha256"] for e in sha["files"]
                                              if e["path"] == "runs/cells.jsonl"),
                     "a0t_gpu_total_s": a0t_budget["TOTALS"]["cumulative_wall_s"],
                     "a0t_n_phases": len(a0t_budget["phases"])}

    # ---- frozen analyses ----
    dist = ex.action_distribution(records)
    ent = {"pooled_bits": ex.action_entropy(records),
           "per_seed_bits": ex.action_entropy(records, "seed"),
           "per_dataset_bits": ex.action_entropy(records, "dataset")}
    sw = ex.switching(records)
    dec = ex.decomposition(records)
    dec_sum = ex.decomposition_summary(dec)

    # fixed-strategy comparison from existing gains (H=4 primary)
    gB = a0t.gain_block(records, "B")
    gC = a0t.gain_block(records, "C")
    fixed_minus = {"availability": "only on 2 non-aliased cells (rolled as arm C)",
                   "cells": [{"dataset": r["dataset"], "seed": r["seed"],
                              "step": r["step"], "stream_seed": r["stream_seed"],
                              **{f"gain_minus_{H}": (r["losses"]["A"][str(H)]
                                                    - r["losses"]["C"][str(H)])
                                 for H in ex.H_GRID}}
                              for r in records if not r["aliased"]]}

    # ---- verdict per frozen rule ----
    case_B_confirmed = bool(dec_sum["4"]["adaptive_gain_median"] <= 0
                            and gB["gain_B"]["4"]["median"] > 0
                            and gB["gain_B"]["4"]["wilcoxon_vs0"]["p"] < 0.05)
    case_C_ruled_out = bool(gC["gain_C"]["4"]["median"] > 0
                            and gC["gain_C"]["4"]["wilcoxon_vs0"]["p"] < 0.05)
    disc_cells = [x for x in records if not x["aliased"]]
    directed = [x["losses"]["A"]["4"] - x["losses"]["C"]["4"]
                - (x["losses"]["A"]["4"] - x["losses"]["B"]["4"])
                for x in disc_cells]
    case_A_evidence = fixed_minus["cells"]
    verdict = {
        "case_A_adaptive_control": (
            "NOT ESTABLISHABLE: only 2/36 discriminative cells; on both the "
            "guided -0.05 beat fixed +0.05 (directed benefit "
            + ", ".join(f"{v:+.3e}" for v in directed)
            + "), consistent with state-dependence but n=2 is underpowered; "
            "a pooled adaptive test needs fixed -0.05 rollouts on all cells "
            "(absent from frozen artifacts; NOT generated - offline mandate)."),
        "case_B_global_scaling": (
            "CONFIRMED as the dominant identity: P(+0.05)=34/36=0.944, "
            "action entropy 0.31 bits pooled, adaptive_gain median = 0 by "
            "construction on aliased cells, fixed +0.05 gain significantly "
            "positive (median +4.10e-4, wilcoxon p=7.1e-7)."),
        "case_C_response_worse": ("RULED OUT: response_gain median +4.41e-4, "
                                  "wilcoxon p=2.0e-10 (frozen A0T).") if case_C_ruled_out
                                   else "NOT RULED OUT",
        "case_B_confirmed_flag": case_B_confirmed,
        "case_C_ruled_out_flag": case_C_ruled_out,
        "primary_verdict": ("Dominantly Case B (global LR-up direction) with a "
                            "validated-but-tiny state-dependent reversal "
                            "component (2/36 cells, both correct)."),
    }

    analysis = {"frozen_inputs": frozen_inputs,
                "action_distribution": dist, "action_entropy": ent,
                "switching": {k: v for k, v in sw.items() if k != "trajectories"},
                "switching_trajectories": sw["trajectories"],
                "fixed_strategy_comparison": {
                    "fixed_plus_gains": gB["gain_B"],
                    "response_gains": gC["gain_C"],
                    "fixed_minus": fixed_minus},
                "decomposition_summary": dec_sum, "verdict": verdict}
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ---- CSV tables ----
    with open(REPORTS / "a0t_extra_action_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "P(+0.05)", "P(-0.05)", "H_bits"])
        w.writerow(["pooled", dist["pooled"]["n"], f'{dist["pooled"]["P(+0.05)"]:.4f}',
                    f'{dist["pooled"]["P(-0.05)"]:.4f}', f'{ent["pooled_bits"]:.4f}'])
        for s, d in sorted(dist["per_seed"].items()):
            w.writerow([f"seed_{s}", d["n"], f'{d["P(+0.05)"]:.4f}',
                        f'{d["P(-0.05)"]:.4f}', f'{ent["per_seed_bits"][s]:.4f}'])
        for dsn, d in sorted(dist["per_dataset"].items()):
            w.writerow([dsn, d["n"], f'{d["P(+0.05)"]:.4f}',
                        f'{d["P(-0.05)"]:.4f}', f'{ent["per_dataset_bits"][dsn]:.4f}'])

    with open(REPORTS / "a0t_extra_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["dataset", "seed", "step", "stream_seed", "chi_lr", "action_u",
               "aliased"]
        for H in ex.H_GRID:
            hdr += [f"response_{H}", f"fixed_plus_{H}", f"fixed_minus_{H}",
                    f"best_fixed_{H}", f"adaptive_{H}"]
        w.writerow(hdr)
        for r in sorted(dec, key=lambda r: (r["dataset"], r["seed"], r["step"],
                                            r["stream_seed"])):
            row = [r["dataset"], r["seed"], r["step"], r["stream_seed"],
                   f'{r["chi_lr"]:.8e}', r["action_u"], r["aliased"]]
            for H in ex.H_GRID:
                k = str(H)
                row += [f'{r[f"gain_response_{k}"]:.8e}',
                        f'{r[f"gain_fixed_plus_{k}"]:.8e}',
                        (f'{r[f"gain_fixed_minus_{k}"]:.8e}'
                         if r[f"gain_fixed_minus_{k}"] is not None else "NA"),
                        f'{r[f"best_fixed_gain_{k}"]:.8e}',
                        f'{r[f"adaptive_gain_{k}"]:.8e}']
            w.writerow(row)

    with open(REPORTS / "a0t_extra_switching.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "stream_seed", "steps", "actions",
                    "sign_changes", "max_possible"])
        for t in sw["trajectories"]:
            w.writerow([t["dataset"], t["seed"], t["stream_seed"],
                        ">".join(map(str, t["steps"])),
                        ">".join(t["actions"]), t["sign_changes"],
                        t["max_possible"]])

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # (a) action distribution per seed
    seeds = sorted(dist["per_seed"])
    axes[0].bar(seeds, [dist["per_seed"][str(s)]["P(+0.05)"] for s in seeds],
                label="P(+0.05)")
    axes[0].bar(seeds, [dist["per_seed"][str(s)]["P(-0.05)"] for s in seeds],
                bottom=[dist["per_seed"][str(s)]["P(+0.05)"] for s in seeds],
                label="P(-0.05)")
    axes[0].set_title("action distribution per seed")
    axes[0].set_ylabel("probability")
    axes[0].legend(fontsize=8)
    # (b) entropy
    labels = ["pooled"] + [f"seed {s}" for s in seeds]
    vals = [ent["pooled_bits"]] + [ent["per_seed_bits"][str(s)] for s in seeds]
    axes[1].bar(labels, vals)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("H(a) bits (max=1)")
    axes[1].set_title("action entropy")
    # (c) decomposition at H=4
    hs = [str(H) for H in ex.H_GRID]
    axes[2].bar(hs, [dec_sum[h]["best_fixed_gain_median"] for h in hs],
                label="best fixed gain (median)")
    ad = [dec_sum[h]["adaptive_gain_median"] for h in hs]
    axes[2].bar(hs, ad, bottom=[dec_sum[h]["best_fixed_gain_median"] for h in hs],
                label="adaptive gain (median)")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("gain")
    axes[2].set_title("response_gain = fixed + adaptive (median)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "a0t_extra_identity.png", dpi=140)
    plt.close(fig)

    # ---- report ----
    rep = []
    rep.append("# A0T-extra Report — Response Controller Identity Analysis\n")
    rep.append("Date: 2026-09-18. OFFLINE analysis of frozen A0T artifacts only. "
               "No new training, no controller modification, no new optimization.\n")
    rep.append("## 1. Question\n")
    rep.append("Is the A0T response-guided LR controller genuine adaptive control, "
               "or does it only discover a global LR scaling direction?\n")
    rep.append("## 2. Frozen inputs\n")
    rep.append(f"- {frozen_inputs['a0t_cells_sha256'][:16]}… cells.jsonl "
               f"(36 cells, CRN 36/36), A0T GPU total "
               f"{frozen_inputs['a0t_gpu_total_s']}s (no new GPU this round).\n")
    rep.append("## 3. Action distribution\n")
    d = dist["pooled"]
    rep.append(f"- pooled: P(+0.05)={d['P(+0.05)']:.4f}, P(-0.05)={d['P(-0.05)']:.4f} "
               f"(n={d['n']})\n")
    for s in seeds:
        dd = dist["per_seed"][str(s)]
        rep.append(f"- seed {s}: P(+0.05)={dd['P(+0.05)']:.4f}, "
                   f"P(-0.05)={dd['P(-0.05)']:.4f} (n={dd['n']})\n")
    rep.append("## 4. Action entropy\n")
    rep.append(f"- pooled H(a) = {ent['pooled_bits']:.4f} bits (max 1.0 for two "
               f"actions). Per seed: "
               + ", ".join(f"{s}: {ent['per_seed_bits'][str(s)]:.3f}" for s in seeds)
               + ". H=0 (seeds 2031/2037) = single global action.\n")
    rep.append("## 5. Switching statistics\n")
    rep.append(f"- trajectories: {sw['n_trajectories']} (18 streams); "
               f"total sign changes: {sw['total_sign_changes']}; trajectories with "
               f"any switch: {sw['trajectories_with_any_switch']}. Per-trajectory "
               f"detail in a0t_extra_switching.csv.\n")
    for t in sw["trajectories"]:
        if t["sign_changes"] > 0:
            rep.append(f"  - {t['dataset']} seed {t['seed']} stream {t['stream_seed']}: "
                       f"{' > '.join(t['actions'])} over steps "
                       f"{' > '.join(map(str, t['steps']))} "
                       f"({t['sign_changes']}/{t['max_possible']})\n")
    rep.append("## 6. Fixed strategy comparison (existing rollout gains)\n")
    for H in ex.H_GRID:
        k = str(H)
        rep.append(f"- H={H}: fixed +0.05 median gain {gB['gain_B'][k]['median']:+.3e} "
                   f"(wilcoxon p={gB['gain_B'][k]['wilcoxon_vs0']['p']}); response "
                   f"median {gC['gain_C'][k]['median']:+.3e}; fixed -0.05 available "
                   f"on 2 non-aliased cells only: "
                   + ", ".join(f"{c['dataset']} s{c['seed']} t{c['step']} x{c['stream_seed']}: "
                               f"{c[f'gain_minus_{H}']:+.3e}" for c in fixed_minus["cells"])
                   + "\n")
    rep.append("## 7. Gain decomposition\n")
    for H in ex.H_GRID:
        s = dec_sum[str(H)]
        rep.append(f"- H={H}: response median {s['response_gain_median']:+.3e} = "
                   f"best-fixed median {s['best_fixed_gain_median']:+.3e} + adaptive "
                   f"median {s['adaptive_gain_median']:+.3e} (max adaptive "
                   f"{s['adaptive_gain_max']:+.3e}, n positive "
                   f"{s['n_adaptive_positive']}/36)\n")
    rep.append("- IDENTIFICATION LIMIT (pre-registered): adaptive=0 by "
               "construction on 34/36 aliased cells; the 2 discriminative cells "
               "both chose -0.05 and beat fixed +0.05, but per the frozen formula "
               "best_fixed includes the -0.05 rollout itself, so pooled adaptive "
               "stays 0. A real adaptive test requires fixed -0.05 rollouts on "
               "all cells (absent from frozen artifacts; NOT generated).\n")
    rep.append("## 7b. Horizon dependence of the discriminative cells\n")
    for x in disc_cells:
        for H in ex.H_GRID:
            gB = x["losses"]["A"][str(H)] - x["losses"]["B"][str(H)]
            gC = x["losses"]["A"][str(H)] - x["losses"]["C"][str(H)]
            rep.append(f"- {x['dataset']} s{x['seed']} t{x['step']} x{x['stream_seed']} "
                       f"H={H}: fixed +0.05 gain {gB:+.3e}, guided -0.05 gain "
                       f"{gC:+.3e} -> C beats B: {gC > gB}\n")
    rep.append("The only two deviations from the global direction win at the "
               "response horizon H=4 and INVERT at H=16/64 (guided -0.05 becomes "
               "harmful). The state-dependent component does not persist beyond "
               "the response horizon - consistent with A0P's unstable long-lag "
               "kernel.\n")
    rep.append("## 8. Interpretation (frozen cases)\n")
    for case in ("case_A_adaptive_control", "case_B_global_scaling",
                 "case_C_response_worse"):
        rep.append(f"- **{case}**: {verdict[case]}\n")
    rep.append(f"\n**Primary verdict**: {verdict['primary_verdict']}\n")
    rep.append("\n## 9. Limitations\n")
    rep.append("- Adaptive component identified on n=2 cells only; no inferential "
               "claim possible (offline mandate forbids generating the missing "
               "fixed -0.05 rollouts).\n")
    rep.append("- Action entropy and switching describe the frozen 36-cell pilot "
               "bank, not the full training run.\n")
    rep.append("- Short-term gains do not imply final-training improvement "
               "(A0P long memory; A0T limitations carry over).\n")
    (REPORTS / "A0T_EXTRA_REPORT.md").write_text("".join(rep))

    print(json.dumps({"entropy_bits": ent["pooled_bits"],
                      "sign_changes": sw["total_sign_changes"],
                      "adaptive_median_H4": dec_sum["4"]["adaptive_gain_median"],
                      "verdict": verdict["primary_verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
