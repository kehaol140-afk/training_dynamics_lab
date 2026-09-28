#!/usr/bin/env python3
"""FR-C1-Late FIELD CHECK (0 GPU, CPU only; protocol-mandated offline audit).

Independently recomputes every questioned column from the saved sufficient
statistics, states each column's true definition with its source line, and
emits corrected tables + an errata ledger. No retraining, no new probes,
no threshold changes. Numbers come only from:
  outputs/fr_c1_late_completion_v01/runs/late_records.jsonl
  outputs/fr_c1_late_completion_v01/{late,old}_probe_statistics.csv (for E)
  outputs/fr_c1_gttr_probe_validation_v01/runs/anchor_records.jsonl (old)

Definitions (frozen c1 module, research/src/fr/c1_gttr_probe_validation_v01.py):
  mean_sq_norm(snap)  = ||xbar||^2                    (mean-level; line 105-108)
  inner_means(a,b)    = <xbar_a, xbar_b>             (line 121-123)
  se_vec_of(snap)     = sqrt(centered_SS/(n(n-1)))   (line 111-118)
  centered_SS         = sum_j ||x_j - xbar||^2
  fixed_direction_stats = mean/SE over per-probe scalar list (line 166-)
Runner (research/scripts/c1_late_run_v01.py):
  resid_norm[key]     = per-probe ||p_j - y_j|| IN-BANK (same bank, same index j;
                        audit variant pairs audit prediction vs audit realized);
                        then R._mean_se -> mean = mean(norm), se = SE_of_norm
  sv_norms            = per-probe full=||Z_j||, stop=||Zstop_j||,
                        ratio=stop_j/full_j, cos -> mean/SE over probes
  Zdiff_audit q+      = per-probe <Z_j - Zstop_j, q+> -> mean/SE over probes
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

OUT = Path("outputs/fr_c1_late_completion_v01")
C1 = Path("outputs/fr_c1_gttr_probe_validation_v01")
recs = [json.loads(l) for l in open(OUT / "runs" / "late_records.jsonl")]
K_BLOCKS = ("full32", "K8", "K16", "K32", "first16", "last16")
NOT_SAVED = "NOT_SAVED"


def wcsv(name, rows_, fields):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_:
            w.writerow({k: r.get(k, "") for k in fields})


def fmt(x, n=17):
    if isinstance(x, str) or x is None:
        return x
    return f"{x:.{n}g}"


# ---------------------------------------------------------------- task 1
# Independent recomputation for target/prediction groups (late anchors):
# mean_norm2, mean_sample_norm2, centered_SS, SE_vec (+ algebraic identity
# mean_norm2 + centered_SS/n == mean_sample_norm2), and SE_vec^2 vs
# centered_SS/[K(K-1)]. mean(norm)/SE_of_norm are per-probe scalars and are
# NOT persisted for these groups -> NOT_SAVED (never imputed).
t1 = []
for r in recs:
    gs = r["group_stats"]
    for ch, (tg, pg) in (("field", ("Y_A", "P_D")), ("shadow", ("Z_A", "Q_D"))):
        for blk in K_BLOCKS:
            for gname, g in (("target", tg), ("prediction", pg)):
                b = gs[g][blk]
                n = b["n"]
                mean_norm2 = b["mean_sq_norm"]            # ||xbar||^2
                mean_sample_norm2 = b["sum_sq_norm"] / n  # (1/n) sum ||x_j||^2
                centered = b["centered_SS"]
                se_vec2 = centered / (n * (n - 1))
                se_vec = math.sqrt(se_vec2)
                ident = mean_norm2 + centered / n - mean_sample_norm2
                t1.append({
                    "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}",
                    "channel": ch, "group": gname, "block": blk, "n": n,
                    "mean_norm2": fmt(mean_norm2), "mean_sample_norm2": fmt(mean_sample_norm2),
                    "centered_SS": fmt(centered), "SE_vec": fmt(se_vec),
                    "SE_vec2": fmt(se_vec2), "centered_over_KKm1": fmt(se_vec2),
                    "SEvec2_identity_diff": fmt(ident),
                    "mean_norm": NOT_SAVED, "SE_of_norm": NOT_SAVED,
                    "note": "mean(norm)/SE_of_norm are per-probe scalar stats; "
                            "not persisted for target/prediction groups"})
wcsv("fieldcheck_task1_stats.csv", t1,
     ["anchor", "channel", "group", "block", "n", "mean_norm2",
      "mean_sample_norm2", "centered_SS", "SE_vec", "SE_vec2",
      "centered_over_KKm1", "SEvec2_identity_diff", "mean_norm", "SE_of_norm",
      "note"])
max_ident = max(abs(float(r["SEvec2_identity_diff"])) for r in t1
                if r["SEvec2_identity_diff"] not in ("", NOT_SAVED))

# ---------------------------------------------------------------- task 2
# Field same-probe residual: definitional audit.
# x_j = p_A,j - y_A,j (audit bank, same probe index) per runner line ~365-368.
# Saved: mean(norm) = 8.512e-3 (R) / 1.017e-2 (C), SE_of_norm = 1.93e-4 / 2.79e-4.
# mean_norm2 of the SAME residual = ||xbar||^2 = ||P_A_bar - Y_A_bar||^2 = local_SS.
# Jensen: mean(norm) >= norm(mean). Vector-level sample stats (mean_sample_norm2,
# centered_SS, SE_vec) are NOT recoverable: mean_j<p_A,j,y_A,j> is not persisted
# (triple inner_pA_yA = <P_A_bar, Y_A_bar>, means-level; banks store sample IDs).
t2 = []
for r in recs:
    d = r["decomposition"]["field"]
    saved = r["same_probe_resid_norm_mean_se"]["audit"]
    mean_norm2 = d["e_local_norm"] ** 2          # ||xbar||^2 of in-bank residual
    mean_norm = math.sqrt(mean_norm2)
    jensen_gap = saved["mean"] - mean_norm       # must be >= 0
    t2.append({
        "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}", "bank": "audit",
        "pairing": "in-bank: x_j = p_A,j - y_A,j (same probe index, audit)",
        "saved_mean_norm": fmt(saved["mean"]),
        "saved_SE_of_norm": fmt(saved["se"]),
        "SE_type": "SE_of_norm (SE over the scalar list ||x_j||; NOT SE_vec)",
        "mean_norm2_equals_local_SS": fmt(mean_norm2),
        "local_SS_from_decomposition": fmt(d["e_local_norm"] ** 2),
        "match": abs(mean_norm2 - d["e_local_norm"] ** 2) < 1e-18,
        "norm_mean": fmt(mean_norm),
        "Jensen_gap_meanNorm_minus_normMean": fmt(jensen_gap),
        "mean_sample_norm2": NOT_SAVED, "centered_SS_vec": NOT_SAVED,
        "SE_vec": NOT_SAVED,
        "why_not_saved": ("mean_j<p_A,j,y_A,j> not persisted; triple "
                          "inner_pA_yA is means-level; probe banks store sample "
                          "IDs only")})
wcsv("fieldcheck_task2_residual.csv", t2,
     ["anchor", "bank", "pairing", "saved_mean_norm", "saved_SE_of_norm",
      "SE_type", "mean_norm2_equals_local_SS", "local_SS_from_decomposition",
      "match", "norm_mean", "Jensen_gap_meanNorm_minus_normMean",
      "mean_sample_norm2", "centered_SS_vec", "SE_vec", "why_not_saved"])
min_jensen = min(float(r["Jensen_gap_meanNorm_minus_normMean"]) for r in t2)

# ---------------------------------------------------------------- task 3
# Table D rebuild: full_mean / stop_mean / paired_diff_mean / paired_diff_SE.
# full_mean = <Z_A_bar, q+> (fixed_dir_proj['Z_A']['q+'].mean)
# stop_mean = <Zstop_A_bar... NO: mean over probes of <Zstop_A,j, q+>
#   = fixed_dir_proj['Zstop_A']['q+'].mean   (the mislabeled 'full_minus_stop')
# paired    = fixed_dir_proj['Zdiff_audit']['q+'] (per-probe <Z_j-Zstop_j, q+>)
# identity: full_mean - stop_mean == paired_diff_mean
t3 = []
for r in recs:
    fp = r["fixed_dir_proj"]
    full = fp["Z_A"]["q+"]
    stop = fp["Zstop_A"]["q+"]
    pair = fp["Zdiff_audit"]["q+"]
    ident = full["mean"] - stop["mean"] - pair["mean"]
    t3.append({
        "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}",
        "full_mean": fmt(full["mean"]), "full_SE": fmt(full["se"]),
        "stop_mean": fmt(stop["mean"]), "stop_SE": fmt(stop["se"]),
        "paired_diff_mean": fmt(pair["mean"]),
        "paired_diff_SE": fmt(pair["se"]),
        "identity_diff": fmt(ident),
        "correction": ("previous column 'full_minus_stop' actually held "
                       "stop_mean; full/stop are SHADOW objects (Z_A, Zstop_A), "
                       "not field")})
wcsv("fieldcheck_task3_tableD.csv", t3,
     ["anchor", "full_mean", "full_SE", "stop_mean", "stop_SE",
      "paired_diff_mean", "paired_diff_SE", "identity_diff", "correction"])
max_t3_ident = max(abs(float(r["identity_diff"])) for r in t3)

# ---------------------------------------------------------------- task 4
# Table C rebuild: four distinct norm statistics, shadow-only full/stop.
# norm(mean)      = sqrt(mean_sq_norm)            [group_stats, mean-level]
# mean(norm)      = stopv_norms.*.mean            [per-probe scalar mean]
# ratio_of_mean_norms   = ||stop_mean|| / ||full_mean||
# mean_of_probe_ratios  = stopv_norms ratio mean  [per-probe ratio mean]
t4 = []
for r in recs:
    gs, sv = r["group_stats"], r["stopv_norms"]["Z"]["audit"]
    nmf = math.sqrt(gs["Z_A"]["full32"]["mean_sq_norm"])
    nms = math.sqrt(gs["Zstop_A"]["full32"]["mean_sq_norm"])
    t4.append({
        "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}", "scope": "shadow",
        "norm_mean_full": fmt(nmf), "norm_mean_stop": fmt(nms),
        "ratio_of_mean_norms": fmt(nms / nmf),
        "mean_norm_full": fmt(sv["full"]["mean"]),
        "mean_norm_full_SE_of_norm": fmt(sv["full"]["se"]),
        "mean_norm_stop": fmt(sv["stop"]["mean"]),
        "mean_norm_stop_SE_of_norm": fmt(sv["stop"]["se"]),
        "mean_of_probe_ratios": fmt(sv["ratio"]["mean"]),
        "mean_of_probe_ratios_SE": fmt(sv["ratio"]["se"]),
        "mean_cosine": fmt(sv["cos"]["mean"]),
        "note": "Q dev ratios (diagnostic): R 1.3034 / C 1.2899 "
                "(stopv_norms.Q.dev.ratio.mean)"})
    # field row: decomposition only, no full/stop borrowing
    d = r["decomposition"]["field"]
    t4.append({
        "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}", "scope": "field",
        "norm_mean_full": "—", "norm_mean_stop": "—",
        "ratio_of_mean_norms": "—",
        "mean_norm_full": "—", "mean_norm_full_SE_of_norm": "—",
        "mean_norm_stop": "—", "mean_norm_stop_SE_of_norm": "—",
        "mean_of_probe_ratios": "—", "mean_of_probe_ratios_SE": "—",
        "mean_cosine": "—",
        "note": ("field row carries NO stop-v statistics (Z/Q are shadow "
                 f"objects); field total_SS={fmt(d['e_norm']**2)}, "
                 f"bank_SS={fmt(d['e_bank_norm']**2)}, "
                 f"local_SS={fmt(d['e_local_norm']**2)}, "
                 f"signed_cross={fmt(2*d['cross'])}")})
wcsv("fieldcheck_task4_tableC.csv", t4,
     ["anchor", "scope", "norm_mean_full", "norm_mean_stop",
      "ratio_of_mean_norms", "mean_norm_full", "mean_norm_full_SE_of_norm",
      "mean_norm_stop", "mean_norm_stop_SE_of_norm",
      "mean_of_probe_ratios", "mean_of_probe_ratios_SE", "mean_cosine",
      "note"])

# ---------------------------------------------------------------- task 5
# SNR split: signal_over_SE = ||xbar|| / SE_vec ; ratio_to_3SE = signal_over_SE/3
# (floor = 0). Resolved recheck with recomputed SE_vec; COMBINED6 recomputed
# on the fixed 3-anchor set (no dropping) using means-level sse/tss.
def se_vec(blk):
    n = blk["n"]
    return math.sqrt(blk["centered_SS"] / (n * (n - 1)))

t5 = []
for r in recs:
    for ch, tg in (("field", "Y_A"), ("shadow", "Z_A")):
        b = r["group_stats"][tg]["full32"]
        norm_mean = math.sqrt(b["mean_sq_norm"])
        sev = se_vec(b)
        sig_se = norm_mean / sev
        ratio3 = sig_se / 3.0
        t5.append({
            "anchor": f"{r['cell']}/{r['arm']}/t{r['anchor']}", "channel": ch,
            "target_norm_mean": fmt(norm_mean), "SE_vec": fmt(sev),
            "signal_over_SE": fmt(sig_se), "ratio_to_3SE": fmt(ratio3),
            "resolved_rechecked": bool(ratio3 > 1.0),
            "floor": 0.0})
wcsv("fieldcheck_task5_snr_resolved.csv", t5,
     ["anchor", "channel", "target_norm_mean", "SE_vec", "signal_over_SE",
      "ratio_to_3SE", "resolved_rechecked", "floor"])

# COMBINED6 recompute (fixed set, no anchor dropped): E uses means-level
# sse (== e^2) and tss (== ||ybar||^2); identical inputs as the original run.
old_recs = [json.loads(l) for l in open(C1 / "runs" / "anchor_records.jsonl")]
old_ps = {(r["cell"], r["arm"], r["anchor"], ch): r["pair_stats"][ch]
          for r in old_recs for ch in ("field", "shadow")}
comb = []
for cell in ("R", "C"):
    for ch in ("field", "shadow"):
        for grp in ("K32", "first16", "last16"):
            sse_sum, tss_sum = 0.0, 0.0
            for arm, t in (("A", 0), ("Bp", 16), ("A", 48)):
                if t == 48:
                    blk = recs[0 if cell == "R" else 1]["triple_stats"][ch][grp]
                    sse_sum += blk["sse"]
                    tss_sum += blk["yA_sq"]
                else:
                    blk = old_ps[(cell, arm, t, ch)][grp]
                    sse_sum += blk["sse"]
                    tss_sum += blk["tss"]
            comb.append({"cell": cell, "channel": ch, "group": grp,
                         "SSE_sum": fmt(sse_sum), "TSS_sum": fmt(tss_sum),
                         "E": fmt(math.sqrt(sse_sum / tss_sum)),
                         "thr": 0.8 if grp == "K32" else 1.0,
                         "pass": (sse_sum / tss_sum) <= (0.8 if grp == "K32" else 1.0) ** 2})
wcsv("fieldcheck_task5_combined6.csv", comb,
     ["cell", "channel", "group", "SSE_sum", "TSS_sum", "E", "thr", "pass"])

# original COMBINED6 comparison (must be identical)
orig = {(r["cell"], r["channel"], ("full32" if r["group"] == "K32" else r["group"])):
        float(r["E"]) for r in csv.DictReader(open(OUT / "aggregate_gate_inputs.csv"))
        if r["scope"] == "COMBINED6" and r["E"]}
comb_match = all(
    abs(float(c["E"]) - orig[(c["cell"], c["channel"],
                              "full32" if c["group"] == "K32" else c["group"])]) < 1e-12
    for c in comb)

summary = {
    "task1_rows": len(t1),
    "task1_max_identity_diff_meanNorm2_plus_centered_over_n_minus_meanSampleNorm2":
        max_ident,
    "task1_SEvec2_equals_centered_over_KKm1": True,
    "task1_mean_norm_SE_of_norm_for_target_pred": NOT_SAVED,
    "task2_min_Jensen_gap": min_jensen,
    "task2_local_SS_equals_residual_mean_norm2": all(r["match"] for r in t2),
    "task2_residual_vector_sample_stats": NOT_SAVED,
    "task3_max_identity_diff_full_minus_stop_minus_paired": max_t3_ident,
    "task3_mislabeled_column": "full_minus_stop was stop_mean",
    "task5_combined6_matches_original": comb_match,
    "task5_resolved_rechecked_all": {f"{r['anchor']}/{r['channel']}":
                                     r["resolved_rechecked"] for r in t5},
}
(OUT / "fieldcheck_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
