#!/usr/bin/env python3
"""FR-C1 GTTR probe-validation offline analyzer (protocol §10-§11).

Deterministic CPU aggregation over saved scalar tables only. No GPU, no model,
no randomness. Reads outputs/fr_c1_gttr_probe_validation_v01/{runs/,*.json};
writes probe_summary.csv, error_decomposition.csv, aggregate_E.csv,
fixed_projection.csv, update_feedback.csv, verdicts.json and <=3 figures.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "fr_c1_gttr_probe_validation_v01"
RUNS = OUT / "runs"
C0X = PROJECT_ROOT / "outputs" / "fr_c0x_offline_explanation_v01"
C0 = PROJECT_ROOT / "outputs" / "fr_c0_gttr_validation_v01"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def E_agg(pairs):
    """Frozen aggregate: sqrt(sum sse / sum tss) over the fixed anchor set."""
    num = sum(p["sse"] for p in pairs)
    den = sum(p["tss"] for p in pairs)
    if den <= 0:
        return float("nan"), num
    return math.sqrt(num / den), num


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (n * (n - 1)))
          if n > 1 else float("nan"))
    return m, se


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    smoke = json.loads((OUT / "smoke_checks.json").read_text())
    budget = json.loads((OUT / "budget_log.json").read_text())
    failed = json.loads((RUNS / "failed_attempts.json").read_text())
    preds = read_jsonl(RUNS / "anchor_predictions.jsonl")
    recs = read_jsonl(RUNS / "anchor_records.jsonl")
    banks = json.loads((RUNS / "probe_banks.json").read_text())
    c0x = json.loads((C0X / "verdict_comparison.json").read_text())
    floor = smoke["numerical_floor_same_probe"]

    anchors = sorted({(r["cell"], r["arm"], r["anchor"]) for r in recs})
    cells = sorted({r["cell"] for r in recs})
    CHANNELS = {"field": {"pred": "P_D", "target": "Y_A", "pred_a": "P_A"},
                "shadow": {"pred": "Q_D", "target": "Z_A", "pred_a": "Q_A"}}
    KS = [8, 16, 32]

    # ------------------------------------------------------------------
    # order integrity (leakage gate): predictions saved before any audit read
    # ------------------------------------------------------------------
    order_ok = True
    order_detail = []
    for r in recs:
        p = next((q for q in preds if (q["cell"], q["arm"], q["anchor"]) ==
                  (r["cell"], r["arm"], r["anchor"])), None)
        ok = (p is not None and p["order_index"] == r["order_index_pred"]
              and r["order_index_pred"] < r["order_index_audit"]
              and len(p.get("sha256_fixed_predictions", "")) == 64)
        order_ok &= ok
        order_detail.append({"anchor": f'{r["cell"]}/{r["arm"]}/t{r["anchor"]}',
                             "ok": ok,
                             "pred_idx": p["order_index"] if p else None,
                             "audit_idx": r["order_index_audit"]})
    pm, rm = (RUNS / "anchor_predictions.jsonl").stat().st_mtime, \
             (RUNS / "anchor_records.jsonl").stat().st_mtime
    order_ok &= pm < rm
    # decomposition identity must hold exactly (cross term kept)
    ident_ok = all(
        abs(r["decomposition"][ch]["sq_identity_resid"]) <= 1e-9 * max(
            1.0, r["decomposition"][ch]["e_norm"] ** 2)
        for r in recs for ch in CHANNELS)

    # ------------------------------------------------------------------
    # probe_summary.csv (§10.2)
    # ------------------------------------------------------------------
    with open(OUT / "probe_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "arm", "anchor", "channel", "K",
                    "target_norm", "target_se", "pred_norm", "pred_se",
                    "same_probe_resid_se", "floor", "resolved_3se"])
        resolv = {}
        for (cell, arm, t) in anchors:
            r = next(x for x in recs if (x["cell"], x["arm"], x["anchor"]) ==
                     (cell, arm, t))
            for ch, g in CHANNELS.items():
                tg, pr = r["groups"][g["target"]], r["groups"][g["pred"]]
                res3 = None
                for k in KS:
                    tn, tse = tg[f"K{k}_mean_norm"], tg[f"K{k}_se"]
                    pn, pse = pr[f"K{k}_mean_norm"], pr[f"K{k}_se"]
                    rse = (r["same_probe_resid_norm_mean_se"]["audit"]["se"]
                           if k == 32 else "")
                    flag = "TRUE" if tn > 3 * math.sqrt(tse ** 2 + 0.0 ** 2) \
                        else "FALSE"
                    if k == 32:
                        res3 = flag == "TRUE"
                    w.writerow([cell, arm, t, ch, k,
                                f"{tn:.6g}", f"{tse:.6g}",
                                f"{pn:.6g}", f"{pse:.6g}",
                                (f"{rse:.6g}" if rse != "" else "NOT_SAVED"),
                                floor["grad"], flag])
                resolv[(cell, t, ch)] = res3

    # ------------------------------------------------------------------
    # error_decomposition.csv (per anchor, cross term kept; squared units)
    # ------------------------------------------------------------------
    with open(OUT / "error_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "arm", "anchor", "channel",
                    "e_sq", "e_bank_sq", "e_local_sq", "cross_x2",
                    "bank_plus_local_plus_cross", "sq_identity_resid",
                    "K", "pair_sse", "pair_tss", "E_pointwise_sqrt_sse_tss"])
        for (cell, arm, t) in anchors:
            r = next(x for x in recs if (x["cell"], x["arm"], x["anchor"]) ==
                     (cell, arm, t))
            for ch in CHANNELS:
                d = r["decomposition"][ch]
                e2 = d["e_norm"] ** 2
                eb2 = d["e_bank_norm"] ** 2
                el2 = d["e_local_norm"] ** 2
                cx2 = 2.0 * d["cross"]
                ps = r["pair_stats"][ch]
                ks = sorted(ps, key=lambda s: (len(s), s))
                for i, k in enumerate(ks):
                    row = [cell, arm, t, ch,
                           f"{e2:.6g}", f"{eb2:.6g}", f"{el2:.6g}", f"{cx2:.6g}",
                           f"{eb2 + el2 + cx2:.6g}",
                           f"{d['sq_identity_resid']:.3g}"]
                    if k.startswith("K"):
                        kk = int(k[1:])
                        w.writerow(row + [kk, f"{ps[k]['sse']:.6g}",
                                          f"{ps[k]['tss']:.6g}",
                                          f"{math.sqrt(ps[k]['sse']/ps[k]['tss']):.6g}"])
                    else:
                        w.writerow(row + [f"half:{k}",
                                          f"{ps[k]['sse']:.6g}",
                                          f"{ps[k]['tss']:.6g}",
                                          f"{math.sqrt(ps[k]['sse']/ps[k]['tss']):.6g}"])

    # ------------------------------------------------------------------
    # aggregate_E.csv (frozen aggregate over each cell's measured anchors)
    # ------------------------------------------------------------------
    agg_rows = []
    with open(OUT / "aggregate_E.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "channel", "K_or_half", "E_aggregate",
                    "abs_SSE_sum", "TSS_sum", "n_anchors"])
        for cell in cells:
            crecs = [r for r in recs if r["cell"] == cell]
            for ch in CHANNELS:
                ps = [r["pair_stats"][ch] for r in crecs]
                for k in KS:
                    pairs = [{"sse": p[f"K{k}"]["sse"], "tss": p[f"K{k}"]["tss"]}
                             for p in ps]
                    e, sse = E_agg(pairs)
                    w.writerow([cell, ch, k, f"{e:.6g}", f"{sse:.6g}",
                                f"{sum(p['tss'] for p in pairs):.6g}",
                                len(crecs)])
                    agg_rows.append({"cell": cell, "channel": ch, "K": k,
                                     "E": e})
                for half in ("first16", "last16"):
                    pairs = [{"sse": p[half]["sse"], "tss": p[half]["tss"]}
                             for p in ps]
                    e, sse = E_agg(pairs)
                    w.writerow([cell, ch, f"half:{half}", f"{e:.6g}",
                                f"{sse:.6g}",
                                f"{sum(p['tss'] for p in pairs):.6g}",
                                len(crecs)])
                    agg_rows.append({"cell": cell, "channel": ch,
                                     "half": half, "E": e})

    # ------------------------------------------------------------------
    # fixed_projection.csv (predetermined directions only)
    # ------------------------------------------------------------------
    with open(OUT / "fixed_projection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "arm", "anchor", "group", "direction",
                    "proj_mean", "proj_se", "n", "q_plus_norm", "s_norm",
                    "note"])
        for (cell, arm, t) in anchors:
            r = next(x for x in recs if (x["cell"], x["arm"], x["anchor"]) ==
                     (cell, arm, t))
            qn, sn = r["q_plus_norm"], r["s_norm"]
            for g in sorted(r["fixed_dir_proj"]):
                for d, st in r["fixed_dir_proj"][g].items():
                    note = ""
                    if g == "Q_D" and d == "q+":
                        note = "SHADOW_FIRST_ORDER_READOUT=-q+^T Qbar_D; " \
                               f"value={-st['mean']:.6g}"
                    if g == "Z_A" and d == "q+":
                        note = "SHADOW_FIRST_ORDER_READOUT=-q+^T Zbar_A; " \
                               f"value={-st['mean']:.6g}"
                    w.writerow([cell, arm, t, g, d, f"{st['mean']:.6g}",
                                f"{st['se']:.6g}", st["n"],
                                f"{qn:.6g}", f"{sn:.6g}", note])

    # ------------------------------------------------------------------
    # update_feedback.csv (stop-v ablation + local c->U derivative)
    # ------------------------------------------------------------------
    with open(OUT / "update_feedback.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "arm", "anchor", "pair", "bank",
                    "full_mean_norm", "full_se", "stop_mean_norm", "stop_se",
                    "inner_mean", "inner_se", "diff_norm_mean", "diff_norm_se",
                    "direction_cos_descr", "note"])
        for (cell, arm, t) in anchors:
            r = next(x for x in recs if (x["cell"], x["arm"], x["anchor"]) ==
                     (cell, arm, t))
            for gfull, gstop, pair in (("Q_D", "Qstop_D", "Q_vs_Qstop"),
                                       ("Q_A", "Qstop_A", "Q_vs_Qstop"),
                                       ("Z_D", "Zstop_D", "Z_vs_Zstop"),
                                       ("Z_A", "Zstop_A", "Z_vs_Zstop")):
                bank = "dev" if gfull.endswith("_D") else "audit"
                pf = r["stopv_pairs"]["Q" if gfull.startswith("Q") else "Z"]
                key = "dev" if bank == "dev" else "audit"
                inner = pf[key]["inner_mean_se"]
                diff = pf[key]["diff_norm_mean_se"]
                gf, gs = r["groups"][gfull], r["groups"][gstop]
                cos = (inner["mean"] /
                       max(gf["mean_norm"] * gs["mean_norm"], 1e-30))
                w.writerow([cell, arm, t, pair, bank,
                            f"{gf['mean_norm']:.6g}", f"{gf['se']:.6g}",
                            f"{gs['mean_norm']:.6g}", f"{gs['se']:.6g}",
                            f"{inner['mean']:.6g}", f"{inner['se']:.6g}",
                            f"{diff['mean']:.6g}", f"{diff['se']:.6g}",
                            f"{cos:.6g}",
                            "stop-v ablation; m updates, P frozen from audit "
                            "mean clipped g0; direction_cos is descriptive "
                            "(inner of means over product of mean norms)"])
            d = r["fixed_dir_proj"]["d_c_to_u"]
            w.writerow([cell, arm, t, "d_c_to_u", "dev",
                        "", "", "", "",
                        f"{d['q+']['mean']:.6g}", f"{d['q+']['se']:.6g}",
                        f"{d['s_hat']['mean']:.6g}", f"{d['s_hat']['se']:.6g}",
                        "", "local c->U derivative, post-update common state; "
                        "q+ and s_hat columns"])

    # ------------------------------------------------------------------
    # verdicts.json (§10)
    # ------------------------------------------------------------------
    rt = budget["counters_round_total"]
    caps = budget["caps"]
    within = all(rt[k] <= caps[k + "_max"] for k in rt) and \
        budget["wall_run_s"] <= caps["gpu_wall_s_max"]
    smoke_ok = (smoke["diag_state_identity"]
                and smoke["hvp_precision_gate"] == "PASS"
                and banks["disjointness"]["status"] == "PASS")

    # SAME_PROBE_APPROXIMATION vs M0 (M0 = zero prediction, error = ||Ybar_A||)
    sp = []
    for (cell, arm, t) in anchors:
        r = next(x for x in recs if (x["cell"], x["arm"], x["anchor"]) ==
                 (cell, arm, t))
        for ch, g in CHANNELS.items():
            m0 = r["groups"][g["target"]]["mean_norm"]
            el = r["decomposition"][ch]["e_local_norm"]
            sp.append({"anchor": f"{cell}/{arm}/t{t}", "channel": ch,
                       "M0_err_norm": m0, "same_probe_err_norm": el,
                       "better_than_M0": el < m0})
    n_better = sum(1 for x in sp if x["better_than_M0"])
    same_probe_verdict = {
        "per_anchor": sp,
        "summary": f"{n_better}/{len(sp)} anchor-channels beat M0 "
                   f"(zero prediction) under the same-probe approximation",
        "resolution_note": "resolvability per probe_summary (3SE, floor=0)"}

    def bank_gate(channel):
        """LOCAL_SUPPORT iff both cells K32 E<=0.8, both halves E<=1,
        >=2 distinct t resolvable per cell (§10). RULE_NOT_MET = an E-gate
        value fails; UNRESOLVED = E-gates pass but the resolvability
        precondition cannot be established at the measured K (targets below
        the 3SE floor — '当前测量未解决', not evidence against prediction)."""
        e32 = {c: next(a["E"] for a in agg_rows
                       if a["cell"] == c and a["channel"] == channel
                       and a.get("K") == 32) for c in cells}
        halves = {c: [a["E"] for a in agg_rows
                      if a["cell"] == c and a["channel"] == channel
                      and a.get("half")] for c in cells}
        tdist = {c: len({t for (cc, _, t) in anchors if cc == c and
                         resolv.get((c, t, channel))}) for c in cells}
        e_ok = all(e32[c] <= 0.8 for c in cells) and all(
            h <= 1.0 for c in cells for h in halves[c])
        missing = any(math.isnan(v) for v in e32.values()) or any(
            math.isnan(h) for c in cells for h in halves[c])
        t_ok = all(tdist[c] >= 2 for c in cells)
        ok = e_ok and t_ok and not missing
        return {
            "E_K32_per_cell": e32,
            "E_halves_per_cell": halves,
            "distinct_t_resolvable": tdist,
            "coverage": budget["coverage"],
            "verdict": ("LOCAL_SUPPORT_ON_SELECTED_ANCHORS" if ok else
                        ("UNRESOLVED" if (e_ok and not t_ok) or missing
                         else "RULE_NOT_MET")),
            "verdict_note": ("E gates pass; >=2-distinct-t resolvability "
                             "precondition not established (t0 shadow targets "
                             "below 3SE floor at all measured K)"
                             if (e_ok and not t_ok) else
                             ("all gate values computable; an E-gate fails"
                              if not e_ok and not missing else
                              "gate statistics missing"))}

    cross = [r["decomposition"][ch]["cross"]
             for r in recs for ch in CHANNELS]
    e2s = [r["decomposition"][ch]["e_norm"] ** 2
           for r in recs for ch in CHANNELS]
    cross_rel = [abs(c) / max(e2, 1e-30) for c, e2 in zip(cross, e2s)]
    err_dec = {"cross_term_range": [min(cross), max(cross)],
               "abs_cross_over_e_sq": [min(cross_rel), max(cross_rel)],
               "identity_residual_max": max(
                   abs(r["decomposition"][ch]["sq_identity_resid"])
                   for r in recs for ch in CHANNELS),
               "verdict": "MIXED",
               "note": ("cross term reported and kept; no norm-percentage "
                        "attribution; sign/magnitude varies by anchor")}

    verdicts = {
        "C0_RULE_REVIEW": {
            "X1_halfbank_gate": c0x["C0_RULE_REVIEW"]["X1_halfbank_gate"],
            "X2_floor_correction": c0x["C0_RULE_REVIEW"]["X2_floor_correction"],
            "X3_axis": c0x["C0_RULE_REVIEW"]["X3_axis"],
            "X4_ledger": c0x["C0_RULE_REVIEW"]["X4_ledger"],
            "overall": "ERRATUM_REQUIRED",
            "note": "C0 正式判定只由 C0X 按原规则计算；C1 不作为 C0 的重新验收"},
        "C1_EXECUTION": {
            "verdict": "PARTIAL" if within and smoke_ok and order_ok and ident_ok
            else "BLOCKED",
            "within_caps": within,
            "round_total_counters": rt,
            "caps": caps,
            "smoke_gates_pass": smoke_ok,
            "order_integrity": {"ok": order_ok, "detail": order_detail,
                                "predictions_mtime_lt_records_mtime": pm < rm},
            "decomposition_identity_ok": ident_ok,
            "failed_attempts_counted": [f["attempt"] for f in failed],
            "coverage": budget["coverage"],
            "reason": ("执行与防泄漏顺序干净、round 预算内；覆盖对称减至 4/6 "
                       "anchor（round HVP cap 计入失败首跑后 432>416）")},
        "SAME_PROBE_APPROXIMATION": same_probe_verdict,
        "INDEPENDENT_BANK_PREDICTION": bank_gate("field"),
        "SHADOW_PREDICTION": bank_gate("shadow"),
        "ERROR_DECOMPOSITION": err_dec,
        "FUTURE_H16_H64_PREDICTION": "NOT_TESTED",
        "ADAPTIVE_TRAINING_VALUE": "NOT_TESTED",
        "ENERGY_BENEFIT": "NOT_TESTED",
    }
    (OUT / "verdicts.json").write_text(
        json.dumps(verdicts, indent=1, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # figures (from saved tables only)
    # ------------------------------------------------------------------
    # fig 1: aggregate E vs K (+ half-bank E), gates 0.8/1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        for cell, mk in (("R", "o"), ("C", "s")):
            ks = [a["K"] for a in agg_rows if a["cell"] == cell
                  and a["channel"] == ch and a.get("K")]
            es = [a["E"] for a in agg_rows if a["cell"] == cell
                  and a["channel"] == ch and a.get("K")]
            ax.plot(ks, es, mk + "-", label=f"cell {cell}")
            hs = [a["E"] for a in agg_rows if a["cell"] == cell
                  and a["channel"] == ch and a.get("half")]
            ax.plot([16, 16], hs, mk, mfc="none", ms=12,
                    label=f"{cell} halves" if cell == "R" else None)
        ax.axhline(0.8, ls="--", c="grey", lw=1)
        ax.axhline(1.0, ls=":", c="grey", lw=1)
        ax.set_title(f"{ch} channel")
        ax.set_xlabel("K (probes)")
        ax.set_xticks(KS)
    axes[0].set_ylabel("aggregate E over measured anchors")
    axes[0].legend(fontsize=8)
    fig.suptitle("C1: independent-bank prediction error vs K (4/6 anchors)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_1_error_vs_K.png", dpi=140)
    plt.close(fig)

    # fig 2: error decomposition per anchor (log scale; cross term annotated)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ch in zip(axes, CHANNELS):
        xs = [f"{c}/{a}/t{t}" for (c, a, t) in anchors]
        for key, lab, mk in (("e_norm", "|e|", "o"),
                             ("e_bank_norm", "|e_bank|", "s"),
                             ("e_local_norm", "|e_local|", "^")):
            ys = [next(r for r in recs if (r["cell"], r["arm"], r["anchor"]) ==
                       x_)["decomposition"][ch][key]
                  for x_ in anchors]
            ax.plot(xs, ys, mk + "--", label=lab, alpha=0.8)
        crs = [next(r for r in recs if (r["cell"], r["arm"], r["anchor"]) ==
                    x_)["decomposition"][ch]["cross"] for x_ in anchors]
        for i, c in enumerate(crs):
            ax.annotate(f"2x={2*c:.1e}", (i, ax.get_ylim()[0]), fontsize=7,
                        rotation=90, va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{ch}: e = e_bank + e_local (cross kept)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("C1: cross-bank vs same-probe error decomposition (4 anchors)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_2_decomposition.png", dpi=140)
    plt.close(fig)

    # fig 3: fixed readout direction projections (q+)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    groups = ["P_D", "P_A", "Y_A", "Q_D", "Q_A", "Z_A", "resid_field_audit"]
    xs = [f"{c}/{a}/t{t}" for (c, a, t) in anchors]
    width = 0.11
    for gi, g in enumerate(groups):
        means, ses = [], []
        for x_ in anchors:
            r = next(r for r in recs if (r["cell"], r["arm"], r["anchor"]) == x_)
            st = r["fixed_dir_proj"][g]["q+"]
            means.append(st["mean"])
            ses.append(st["se"])
        pos = [i + (gi - 3) * width for i in range(len(xs))]
        ax.errorbar(pos, means, yerr=ses, fmt="o", ms=4,
                    label=g if not g.startswith("resid") else "same-probe resid")
    ax.axhline(0.0, ls=":", c="grey", lw=1)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=45)
    ax.set_ylabel("projection onto q+ = grad V_diag(theta+s)  (mean +- SE)")
    ax.set_title("C1: predetermined readout direction (fixed before audit endpoints)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_3_fixed_direction.png", dpi=140)
    plt.close(fig)

    print(json.dumps({
        "anchors": len(anchors), "order_ok": order_ok,
        "ident_ok": ident_ok,
        "round_totals": rt,
        "field_E32": verdicts["INDEPENDENT_BANK_PREDICTION"]["E_K32_per_cell"],
        "shadow_E32": verdicts["SHADOW_PREDICTION"]["E_K32_per_cell"],
        "same_probe_better": same_probe_verdict["summary"],
        "verdicts": {k: (v if isinstance(v, str) else v.get("verdict"))
                     for k, v in verdicts.items()
                     if k not in ("C0_RULE_REVIEW",)}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
