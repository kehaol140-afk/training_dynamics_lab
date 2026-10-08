#!/usr/bin/env python3
"""F3 post-processing: raw/ -> summary_final.json + the 8 frozen CSVs.

Reads outputs/f3_curvature_mechanism/raw/{rq1,rq2,rq3,rq4}_{cell}.json and
writes outputs/f3_curvature_mechanism/{summary_final.json, alpha_sweep.csv,
forced_growth.csv, impulse_growth.csv, module_ablation.csv, module_only.csv,
spectrum.csv, finite_time_growth.csv, curvature_ratio_angle.csv}.
(The 2026-09-27 BLOCKED summary.json is preserved as history; the final
post-run aggregate is written to summary_final.json.)

Verdict procedures are exactly prereg docs/F3_CURVATURE_MECHANISM_PREREG.md
§8.1/§8.2. Monotonicity operationalization (analysis choice, reported in
summary.json): a cell's alpha-sweep is "monotone" iff G_forced(H, alpha) is
non-decreasing across the frozen 7-point ALPHA_GRID at every frozen readout
H in F3_H_READOUTS.

Run after all RQ raws are complete:
    python scripts/f3_postprocess.py
"""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "f3_curvature_mechanism"
RAW = OUT / "raw"

CELLS = [("2042", "26"), ("2043", "26"), ("2042", "64")]
EARLY = [("2042", "26"), ("2043", "26")]
ALPHAS = ("0.0", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0")
H_READOUTS = (1, 2, 4, 8, 16, 32)
PROBES = ("theta_only", "m_only", "v_only", "full_random", "attn_q_localized",
          "q0_lrsource")
OFFSETS = (0, 1, 3, 7, 15)
ABLAT_MAIN = ("FULL", "NO_ATTN_Q", "NO_ATTN_K", "NO_ATTN_V", "NO_ATTN_O",
              "NO_MLP", "NO_EMBED", "NO_NORM")
ABLAT_ONLY = ("ONLY_ATTN_Q", "ONLY_ATTN_K", "ONLY_MLP")
ABLAT_SUB = ("NO_MLP_UP", "NO_MLP_GATE", "NO_MLP_DOWN")

# frozen thresholds (prereg §8)
RHO_MARGIN = 1.05
TRANSIENT_G16 = 10.0
CONTROL_G16 = 3.0
NONNORMAL_PROXY = 2.0
LARGE_EFFECT = 10.0       # H=32 tangent norm reduction vs FULL (descriptive)
ONLY_RETAIN = 0.5         # ONLY_* retains >=50% of FULL growth
SWEEP_RATIO = 10.0        # G_forced(H=32, alpha=1) / G_forced(H=32, alpha=0)


def load(name):
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(p)
    return json.load(p.open())


def cell_str(s, st):
    return f"{s}:{st}"


def g_forced(steps, alpha, H):
    """G_forced(H, alpha) from rq1 steps (norms dict keyed by str alpha)."""
    n1 = nH = None
    for r in steps:
        a = r["arms"].get(alpha)
        if a is None or a.get("stopped"):
            continue
        if r["h"] == 1:
            n1 = a["dtheta_norm"]
        if r["h"] == H:
            nH = a["dtheta_norm"]
    if n1 is None or nH is None:
        return None
    return nH / max(n1, 1e-30)


def monotone_sweep(steps):
    """Per readout H: True if G(H, a_i+1) >= G(H, a_i) for all consecutive."""
    per_h = {}
    for H in H_READOUTS:
        gs = [g_forced(steps, a, H) for a in ALPHAS]
        if any(g is None for g in gs):
            per_h[H] = None
        else:
            per_h[H] = all(gs[i + 1] >= gs[i] - 1e-12 * max(1.0, gs[i])
                            for i in range(len(gs) - 1))
    return per_h


def impulse(steps, key, H):
    n0 = nH = None
    for r in steps:
        a = r["arms"].get(key)
        if a is None or a.get("stopped"):
            continue
        if r["h"] == 1:
            n0 = a["dz_norm"]
        if r["h"] == H:
            nH = a["dz_norm"]
    if n0 is None or nH is None:
        return None
    return nH / max(n0, 1e-30)


def rq3_g(steps, arm, H, field="dtheta_norm"):
    n1 = nH = None
    for r in steps:
        a = r["arms"].get(arm)
        if a is None or a.get("stopped"):
            continue
        if r["h"] == 1:
            n1 = a[field]
        if r["h"] == H:
            nH = a[field]
    if n1 is None or nH is None:
        return None
    return nH / max(n1, 1e-30)


def rq3_final(steps, arm, field="dtheta_norm"):
    for r in reversed(steps):
        a = r["arms"].get(arm)
        if a is not None and not a.get("stopped"):
            return a[field], r["h"]
    return None, None


def main():
    rq1 = {cell_str(s, st): load(f"rq1_{s}_{st}.json") for s, st in CELLS}
    rq2 = {cell_str(s, st): load(f"rq2_{s}_{st}.json") for s, st in CELLS}
    rq3 = {cell_str(s, st): load(f"rq3_{s}_{st}.json") for s, st in CELLS}
    rq4 = {cell_str(s, st): load(f"rq4_{s}_{st}.json") for s, st in CELLS}

    summary = {"cells": {}, "verdict": {}, "frozen_thresholds": {
        "RHO_MARGIN": RHO_MARGIN, "TRANSIENT_G16": TRANSIENT_G16,
        "CONTROL_G16": CONTROL_G16, "NONNORMAL_PROXY": NONNORMAL_PROXY,
        "LARGE_EFFECT": LARGE_EFFECT, "ONLY_RETAIN": ONLY_RETAIN,
        "SWEEP_RATIO": SWEEP_RATIO}}

    # ---------------- alpha_sweep.csv + forced_growth.csv -------------------
    with (OUT / "alpha_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "alpha", "h", "loss", "clip_coef", "grad_norm",
                    "dtheta_norm", "dm_norm", "dv_norm", "dz_norm",
                    "hvp_norm", "top_module", "h_eta"])
        for cell, raw in rq1.items():
            for r in raw["steps"]:
                for al in ALPHAS:
                    a = r["arms"].get(al)
                    if a is None:
                        continue
                    w.writerow([cell, al, r["h"], r["loss"], r["clip_coef"],
                                r["grad_norm"], a["dtheta_norm"], a["dm_norm"],
                                a["dv_norm"], a["dz_norm"], a["hvp_norm"],
                                a.get("top_module", ""), a.get("h_eta", "")])
    with (OUT / "forced_growth.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "alpha", "H", "G_forced"])
        fgrowth = {}
        for cell, raw in rq1.items():
            fgrowth[cell] = {}
            for al in ALPHAS:
                for H in H_READOUTS:
                    g = g_forced(raw["steps"], al, H)
                    fgrowth[cell][(al, H)] = g
                    w.writerow([cell, al, H, g if g is not None else ""])

    # ---------------- impulse_growth.csv ------------------------------------
    with (OUT / "impulse_growth.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "alpha", "probe", "H", "G_impulse", "lambda_H"])
        igrowth = {}
        for cell, raw in rq2.items():
            igrowth[cell] = {}
            for al in ("0.0", "1.0"):
                for p in PROBES:
                    key = f"a{al}|{p}"
                    for H in H_READOUTS:
                        g = impulse(raw["steps"], key, H)
                        igrowth[cell][(al, p, H)] = g
                        w.writerow([cell, al, p, H, g if g is not None else "",
                                    (math.log(max(g, 1e-30)) / H)
                                    if g is not None else ""])

    # ---------------- module_ablation.csv / module_only.csv -----------------
    def write_ablation(fname, arms):
        with (OUT / fname).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "arm", "G_forced_H16", "G_forced_H32",
                        "so_G16", "so_G32", "final_dtheta_H32", "h_eta_H16",
                        "LARGE_EFFECT_vs_FULL"])
            out = {}
            for cell, raw in rq3.items():
                g32_full, _ = rq3_final(raw["steps"], "FULL")
                rows = {}
                for arm in arms:
                    g16 = rq3_g(raw["steps"], arm, 16)
                    g32 = rq3_g(raw["steps"], arm, 32)
                    so16 = rq3_g(raw["steps"], f"SO|{arm}", 16, "dz_norm")
                    so32 = rq3_g(raw["steps"], f"SO|{arm}", 32, "dz_norm")
                    fin, _ = rq3_final(raw["steps"], arm)
                    heta = None
                    for r in raw["steps"]:
                        if r["h"] == 16:
                            a = r["arms"].get(arm) or {}
                            heta = a.get("h_eta")
                    large = (g32_full is not None and fin is not None
                             and g32_full / max(fin, 1e-30) >= LARGE_EFFECT)
                    rows[arm] = {"G16": g16, "G32": g32, "so16": so16,
                                 "so32": so32, "final": fin, "h_eta": heta,
                                 "large": large}
                    w.writerow([cell, arm, g16 if g16 is not None else "",
                                g32 if g32 is not None else "",
                                so16 if so16 is not None else "",
                                so32 if so32 is not None else "",
                                fin if fin is not None else "",
                                heta if heta is not None else "", large])
                out[cell] = rows
            return out

    ab_main = write_ablation("module_ablation.csv", ABLAT_MAIN)
    ab_only = write_ablation("module_only.csv", ABLAT_ONLY)
    ab_sub = write_ablation("module_ablation_secondary.csv", ABLAT_SUB)

    # ---------------- spectrum.csv ------------------------------------------
    with (OUT / "spectrum.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "offset", "alpha", "sigma_max", "pi_iters",
                    "rho", "rho_reliable", "rho_residual", "arnoldi_m",
                    "nonnormal_proxy", "clip_active", "clip_coef", "loss"])
        spec = {}
        for cell, raw in rq4.items():
            spec[cell] = {}
            for off in OFFSETS:
                rec = raw["offsets"].get(str(off))
                if rec is None:
                    continue
                for al in ("0.0", "1.0"):
                    d = rec[f"alpha{al}"]
                    spec[cell][(off, al)] = d
                    w.writerow([cell, off, al, d["sigma_max"], d["pi_iters"],
                                d["rho"] if d["rho"] is not None else "",
                                d["rho_reliable"], d["rho_residual"],
                                d["arnoldi_m"],
                                d["nonnormal_proxy"] if d["nonnormal_proxy"] is not None else "",
                                rec["clip_active"], rec["clip_coef"], rec["loss"]])

    # ---------------- RQ5: finite_time_growth.csv + curvature_ratio_angle.csv
    with (OUT / "finite_time_growth.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "h", "alpha", "norm_A", "norm_C",
                    "norm_AplusC", "G_forced"])
        for cell, raw in rq1.items():
            for r in raw["steps"]:
                dec = r.get("decomposition", {}).get("1.0")
                if dec:
                    w.writerow([cell, r["h"], "1.0", dec["norm_A"],
                                dec["norm_C"], dec["norm_AplusC"],
                                fgrowth[cell][("1.0", r["h"])]
                                if r["h"] in H_READOUTS else ""])
    with (OUT / "curvature_ratio_angle.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "h", "normC_over_normA", "cos_angle_A_C",
                    "inner_A_C"])
        for cell, raw in rq1.items():
            for r in raw["steps"]:
                dec = r.get("decomposition", {}).get("1.0")
                if dec:
                    ratio = dec["norm_C"] / max(dec["norm_A"], 1e-30)
                    w.writerow([cell, r["h"], ratio, dec["cos"],
                                dec["inner_A_C"]])

    # ---------------- §8.1 RQ4 mechanism classification ----------------------
    def classify(cell):
        g16_a1 = igrowth[cell].get(("1.0", "q0_lrsource", 16))
        g16_a0 = igrowth[cell].get(("0.0", "q0_lrsource", 16))
        offsets_ok = []
        nn_evidence = spectral_evidence = 0
        for off in OFFSETS:
            d = spec[cell].get((off, "1.0"))
            if d is None:
                continue
            if d["rho_reliable"] and d["rho"] is not None and d["rho"] > RHO_MARGIN:
                offsets_ok.append(off)
            if d["rho_reliable"] and d["rho"] is not None:
                if d["nonnormal_proxy"] is not None and \
                        d["nonnormal_proxy"] > NONNORMAL_PROXY and d["rho"] <= RHO_MARGIN:
                    nn_evidence += 1
        spectral_evidence = len(offsets_ok)
        transient = (g16_a1 is not None and g16_a1 >= TRANSIENT_G16)
        control = (g16_a0 is not None and g16_a0 < CONTROL_G16)
        if spectral_evidence >= 2 and transient:
            label = "SPECTRAL_UNSTABLE"
        elif nn_evidence >= 2 and transient and control:
            label = "NONNORMAL_TRANSIENT"
        elif transient:
            label = "TIME_VARYING_AMPLIFICATION"
        else:
            label = "UNRESOLVED"
        return {"label": label, "offsets_rho_gt_margin": offsets_ok,
                "transient_g16_a1": g16_a1, "control_g16_a0": g16_a0,
                "nn_offsets": nn_evidence}

    mech = {cell: classify(cell) for cell in rq1}

    # aggregation (§8.1): >=2/3 cells and >=2 EARLY cells
    labels = [mech[c]["label"] for c in mech]
    from collections import Counter
    cnt = Counter(labels)
    top_label, top_n = cnt.most_common(1)[0]
    early_labels = [mech[c]["label"] for c in ("2042:26", "2043:26")]
    if top_n >= 2 and early_labels.count(top_label) >= 2 \
            and top_label != "UNRESOLVED":
        mech_agg = top_label
    elif len(set(early_labels)) > 1:
        mech_agg = "MIXED" if top_n >= 2 else "UNRESOLVED"
    else:
        mech_agg = "UNRESOLVED"

    # ---------------- §8.2 F3 verdict ----------------------------------------
    mono = {cell: monotone_sweep(raw["steps"]) for cell, raw in rq1.items()}
    sweep_ok_cells = []
    for cell in rq1:
        m = mono[cell]
        mono_all = all(m[H] for H in H_READOUTS if m[H] is not None) and \
            all(m[H] is not None for H in H_READOUTS)
        g32_a1 = fgrowth[cell][("1.0", 32)]
        g32_a0 = fgrowth[cell][("0.0", 32)]
        ratio = (g32_a1 / max(g32_a0, 1e-30)
                 if g32_a1 is not None and g32_a0 is not None else None)
        if mono_all and ratio is not None and ratio >= SWEEP_RATIO:
            sweep_ok_cells.append(cell)
    early_sweep_ok = [c for c in sweep_ok_cells if c in ("2042:26", "2043:26")]
    g16_a1 = {c: igrowth[c].get(("1.0", "q0_lrsource", 16)) for c in rq1}
    impulse_ok = [c for c in rq1
                  if g16_a1[c] is not None and g16_a1[c] >= TRANSIENT_G16]

    causal_cells = [c for c in sweep_ok_cells if c in impulse_ok]
    causal_early = [c for c in causal_cells if c in ("2042:26", "2043:26")]
    v1 = len(causal_cells) >= 2 and len(causal_early) >= 2

    v2 = False
    if v1:
        e1 = e2 = 0
        for c in ("2042:26", "2043:26"):
            full32 = ab_main[c]["FULL"]["final"]
            noq32 = ab_main[c]["NO_ATTN_Q"]["final"]
            onlyq_g32 = ab_only[c]["ONLY_ATTN_Q"]["G32"]
            full_g32 = ab_main[c]["FULL"]["G32"]
            if full32 is not None and noq32 is not None and \
                    full32 / max(noq32, 1e-30) >= LARGE_EFFECT:
                e1 += 1
            if onlyq_g32 is not None and full_g32 is not None and \
                    onlyq_g32 / max(full_g32, 1e-30) >= ONLY_RETAIN:
                e2 += 1
        v2 = (e1 >= 2 and e2 >= 2)

    # assemble per-cell summary
    for cell in rq1:
        summary["cells"][cell] = {
            "monotone_per_H": {str(H): mono[cell][H] for H in H_READOUTS},
            "G_forced": {al: {str(H): fgrowth[cell][(al, H)]
                              for H in H_READOUTS} for al in ALPHAS},
            "G_impulse_q0": {"alpha1_H16": igrowth[cell].get(("1.0", "q0_lrsource", 16)),
                             "alpha0_H16": igrowth[cell].get(("0.0", "q0_lrsource", 16)),
                             "alpha1_H32": igrowth[cell].get(("1.0", "q0_lrsource", 32))},
            "module_ablation_G32": {arm: ab_main[cell][arm]["G32"]
                                    for arm in ABLAT_MAIN},
            "module_only_G32": {arm: ab_only[cell][arm]["G32"]
                                for arm in ABLAT_ONLY},
            "mechanism": mech[cell],
        }
    summary["checks"] = {
        "sweep_ok_cells": sweep_ok_cells,
        "sweep_G32_ratio": {c: (fgrowth[c][("1.0", 32)] / max(fgrowth[c][("0.0", 32)], 1e-30))
                            for c in rq1},
        "impulse_ok_cells": impulse_ok,
        "causal_cells": causal_cells,
        "v1_CURVATURE_FEEDBACK_CAUSAL": v1,
        "v2_ATTN_Q_BOTTLENECK": v2,
        "mechanism_aggregation": mech_agg,
        "monotone_operationalization":
            "G_forced(H,alpha) non-decreasing across frozen ALPHA_GRID at all "
            "frozen readouts H (tolerance 1e-12 relative)",
    }

    if v1 and v2:
        verdict = "CURVATURE_FEEDBACK_CAUSAL + ATTN_Q_CURVATURE_BOTTLENECK"
    elif v1:
        verdict = "CURVATURE_FEEDBACK_CAUSAL"
    elif mech_agg in ("SPECTRAL_UNSTABLE", "NONNORMAL_TRANSIENT",
                      "TIME_VARYING_AMPLIFICATION", "MIXED"):
        verdict = mech_agg
    else:
        # distinguish CORRELATED_ONLY / NO_CURVATURE / UNRESOLVED
        alpha_effect = []
        for c in rq1:
            g0 = fgrowth[c][("0.0", 32)]
            g1 = fgrowth[c][("1.0", 32)]
            if g0 is not None and g1 is not None and g1 > g0:
                alpha_effect.append(c)
        if len(alpha_effect) >= 2:
            verdict = "CURVATURE_CORRELATED_ONLY"
        elif len(alpha_effect) == 0:
            verdict = "NO_CURVATURE_MECHANISM"
        else:
            verdict = "UNRESOLVED"
    summary["verdict"] = {"label": verdict, "main": verdict}

    with (OUT / "summary_final.json").open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"verdict": verdict, "v1": v1, "v2": v2,
                      "mech_agg": mech_agg}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
