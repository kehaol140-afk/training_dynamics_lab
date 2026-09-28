#!/usr/bin/env python3
"""FR-C1-Late offline analysis (0 GPU): independent recheck of C0X/C1 key data
(O1/O2/O3), late-anchor tables (A-D), COMBINED6 gate, criterion trace, and the
mandated artifact set (protocol FR_C1L §3/§7.2/§8/§9).

Independence rule (§O1/§11): the OLD-data recheck below uses ONLY json/csv
reading + small local math functions defined here; it does NOT import any
production criterion/aggregation helper.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C0 = ROOT / "outputs" / "fr_c0_gttr_validation_v01"
C0X = ROOT / "outputs" / "fr_c0x_offline_explanation_v01"
C1 = ROOT / "outputs" / "fr_c1_gttr_probe_validation_v01"
OUT = ROOT / "outputs" / "fr_c1_late_completion_v01"
RUNS = OUT / "runs"

FLOOR = 0.0  # same-probe floor from C1/C1L smoke (identical evaluations)


# --- tiny independent reference math (no production helpers) -----------------

def ref_E(sse_sum, tss_sum):
    return math.sqrt(sse_sum / tss_sum) if tss_sum > 0 else float("nan")


def ref_se(n, centered_ss):
    return math.sqrt(max(centered_ss, 0.0) / (n * (n - 1))) if n >= 2 else float("nan")


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def jload(p):
    return json.loads(Path(p).read_text())


old_recs = read_jsonl(C1 / "runs" / "anchor_records.jsonl")
old_preds = read_jsonl(C1 / "runs" / "anchor_predictions.jsonl")
late_recs = read_jsonl(RUNS / "late_records.jsonl")
late_preds = read_jsonl(RUNS / "late_predictions.jsonl")
old_summary_rows = list(csv.DictReader(open(C1 / "probe_summary.csv")))
aggregate_E_rows = list(csv.DictReader(open(C1 / "aggregate_E.csv")))
c0x_verdict = jload(C0X / "verdict_comparison.json")
c0_cfg = jload(C0 / "config.json")
res_log = jload(OUT / "resource_log.json")
smoke = jload(OUT / "smoke_checks.json")
c1_budget = jload(C1 / "budget_log.json")
c1_config = jload(C1 / "config.json")

# ---------------------------------------------------------------------------
# O1: independent recheck of OLD 4 anchors from raw records
# ---------------------------------------------------------------------------
CH = {"field": ("P_D", "Y_A", "P_A"), "shadow": ("Q_D", "Z_A", "Q_A")}

old_by_key = {}
for r in old_recs:
    for ch in ("field", "shadow"):
        ps = r["pair_stats"][ch]
        g = r["groups"]
        yg = g["Y_A" if ch == "field" else "Z_A"]
        pg = g["P_D" if ch == "field" else "Q_D"]
        n32 = yg["K32_mean_norm"]
        se32 = yg["K32_se"]
        row = {
            "cell": r["cell"], "arm": r["arm"], "anchor": r["anchor"],
            "channel": ch, "h": r["h"],
            "t_after": r["t_after"],
            "K": 32,
            "target_norm": n32, "target_se": se32,
            "pred_norm": pg["K32_mean_norm"], "pred_se": pg["K32_se"],
            "floor": FLOOR,
            "signal_ratio": (n32 / (3 * math.sqrt(se32 ** 2 + FLOOR ** 2))
                             if se32 > 0 else float("nan")),
            "resolved": bool(n32 > 3 * math.sqrt(se32 ** 2 + FLOOR ** 2)),
            "residual_se": (r["same_probe_resid_norm_mean_se"]["audit"]["se"]
                            if ch == "field" else None),
            "residual_se_note": ("" if ch == "field"
                                 else "NOT_SAVED (old C1 shadow per-probe "
                                      "residual SE never saved; the K32 value "
                                      "printed in probe_summary.csv shadow rows "
                                      "is the FIELD value - analyzer artifact, "
                                      "not used)"),
            "sse": ps["K32"]["sse"], "tss": ps["K32"]["tss"],
            "E": ref_E(ps["K32"]["sse"], ps["K32"]["tss"]),
            "perK": {k: {"sse": ps[k]["sse"], "tss": ps[k]["tss"],
                         "E": ref_E(ps[k]["sse"], ps[k]["tss"])}
                     for k in ("K8", "K16", "K32")},
            "halves": {h: {"sse": ps[h]["sse"], "tss": ps[h]["tss"],
                           "E": ref_E(ps[h]["sse"], ps[h]["tss"])}
                       for h in ("first16", "last16")},
        }
        old_by_key[(r["cell"], r["arm"], r["anchor"], ch)] = row

# cross-check recomputed OLD4 aggregates against the published aggregate_E.csv
agg_check = []
for scope_cell in ("R", "C"):
    for ch in ("field", "shadow"):
        keys = [(scope_cell, "A", 0, ch), (scope_cell, "Bp", 16, ch)]
        for grp, get in (("full32", lambda d: d["perK"]["K32"]),
                         ("first16", lambda d: d["halves"]["first16"]),
                         ("last16", lambda d: d["halves"]["last16"])):
            sses = [get(old_by_key[k])["sse"] for k in keys]
            tsss = [get(old_by_key[k])["tss"] for k in keys]
            E = ref_E(sum(sses), sum(tsss))
            label = grp if grp == "full32" else f"half:{grp}"
            pub = [r for r in aggregate_E_rows
                   if r["cell"] == scope_cell and r["channel"] == ch
                   and r["K_or_half"] == (label if grp == "full32" else grp)]
            pubE = float(pub[0]["E_aggregate"]) if pub else None
            agg_check.append({"cell": scope_cell, "channel": ch, "group": label,
                              "SSE_sum": sum(sses), "TSS_sum": sum(tsss),
                              "E_recomputed": E, "E_published": pubE,
                              "match": (pubE is None or abs(E - pubE) < 1e-9)})

# ---------------------------------------------------------------------------
# LATE anchors: derived table-A rows (full precision from saved stats)
# ---------------------------------------------------------------------------

def late_anchor_rows(r):
    gs = r["group_stats"]
    out = {}
    for ch in ("field", "shadow"):
        yname = "Y_A" if ch == "field" else "Z_A"
        pname = "P_D" if ch == "field" else "Q_D"
        res = r["same_probe_resid_norm_mean_se" if ch == "field"
                else "same_probe_resid_shadow_norm_mean_se"]["audit"]
        yf, pf = gs[yname]["full32"], gs[pname]["full32"]
        n32 = math.sqrt(yf["mean_sq_norm"])
        se32 = ref_se(yf["n"], yf["centered_SS"])
        k32 = r["triple_stats"][ch]["K32"]
        # frozen convention (verified vs old records): tss = mean_j ||y_j||^2
        # (= yA_sq); sse = mean_j ||p_j - y_j||^2
        def _blk(d):
            return {"sse": d["sse"], "tss": d["yA_sq"], "E": ref_E(d["sse"], d["yA_sq"])}
        out[ch] = {
            "cell": r["cell"], "arm": r["arm"], "anchor": r["anchor"],
            "channel": ch, "h": r["h"], "t_after": r["t_after"], "K": 32,
            "target_norm": n32, "target_se": se32,
            "pred_norm": math.sqrt(pf["mean_sq_norm"]),
            "pred_se": ref_se(pf["n"], pf["centered_SS"]),
            "floor": FLOOR,
            "signal_ratio": (n32 / (3 * math.sqrt(se32 ** 2 + FLOOR ** 2))
                             if se32 > 0 else float("nan")),
            "resolved": bool(n32 > 3 * math.sqrt(se32 ** 2 + FLOOR ** 2)),
            "residual_se": res["se"],
            "residual_se_note": ("SE of per-probe paired-residual NORMS "
                                 "(same convention as C1)"),
            "sse": k32["sse"], "tss": k32["yA_sq"],
            "E": ref_E(k32["sse"], k32["yA_sq"]),
            "perK": {k: _blk(r["triple_stats"][ch][k])
                     for k in ("K8", "K16", "K32")},
            "halves": {h: _blk(r["triple_stats"][ch][h])
                       for h in ("first16", "last16")},
        }
    return out


late_by_key = {}
for r in late_recs:
    for ch, row in late_anchor_rows(r).items():
        late_by_key[(r["cell"], r["arm"], r["anchor"], ch)] = row

ALL_OLD = [("A", 0), ("Bp", 16)]
ALL_NEW = [("A", 48)]
ALL6 = [("A", 0), ("Bp", 16), ("A", 48)]


def scope_agg(cell, ch, arm_t_list, grp):
    """Aggregate over a FIXED anchor set; grp in {K8,K16,K32,first16,last16}."""
    sses, tsss, ts_resolved = [], [], set()
    grp_key = "K32" if grp == "full32" else grp
    missing = []
    for arm, t in arm_t_list:
        key = (cell, arm, t, ch)
        src = old_by_key.get(key) if t != 48 else late_by_key.get(key)
        if src is None:
            missing.append(f"{cell}/{arm}/t{t}/{ch}")
            continue
        blk = src["perK"][grp_key] if grp_key.startswith("K") else src["halves"][grp_key]
        sses.append(blk["sse"])
        tsss.append(blk["tss"])
        if src["resolved"]:
            ts_resolved.add(t)
    if missing:
        return {"missing": missing}
    return {"SSE_sum": sum(sses), "TSS_sum": sum(tsss),
            "E": ref_E(sum(sses), sum(tsss)),
            "n_distinct_t_resolved": len(ts_resolved),
            "distinct_t": sorted(ts_resolved)}


# Table B rows
tableB = []
for cell in ("R", "C"):
    for ch in ("field", "shadow"):
        for scope, alist in (("OLD4", ALL_OLD), ("LATE2", ALL_NEW),
                             ("COMBINED6", ALL6)):
            ids = ";".join(f"{a}/t{t}" for a, t in alist)
            for grp, thr in (("full32", 0.8), ("first16", 1.0),
                             ("last16", 1.0)):
                g = scope_agg(cell, ch, alist, grp)
                if "missing" in g:
                    tableB.append({"scope": scope, "cell": cell, "channel": ch,
                                   "group": grp, "anchor_IDs": ids,
                                   "n_anchor": len(alist),
                                   "n_distinct_t_resolved": None,
                                   "SSE_sum": None, "TSS_sum": None, "E": None,
                                   "threshold": thr,
                                   "criterion": "COMBINED_GATE_DATA_MISSING",
                                   "detail": json.dumps(g["missing"])})
                    continue
                thr_ok = g["E"] <= thr
                row = {"scope": scope, "cell": cell, "channel": ch, "group": grp,
                       "anchor_IDs": ids, "n_anchor": len(alist),
                       "n_distinct_t_resolved": g["n_distinct_t_resolved"],
                       "SSE_sum": g["SSE_sum"], "TSS_sum": g["TSS_sum"],
                       "E": g["E"], "threshold": thr,
                       "criterion": "TRUE" if thr_ok else "FALSE"}
                tableB.append(row)

# full-gate booleans (per scope/cell/channel): E gates + distinct-t (full32 set)
gate_trace = []
for cell in ("R", "C"):
    for ch in ("field", "shadow"):
        for scope, alist in (("OLD4", ALL_OLD), ("LATE2", ALL_NEW),
                             ("COMBINED6", ALL6)):
            f = scope_agg(cell, ch, alist, "K32")
            h1 = scope_agg(cell, ch, alist, "first16")
            h2 = scope_agg(cell, ch, alist, "last16")
            if "missing" in f:
                gate_trace.append({"scope": scope, "cell": cell, "channel": ch,
                                   "status": "COMBINED_GATE_DATA_MISSING",
                                   "missing": f["missing"]})
                continue
            b_k32 = bool(f["E"] <= 0.8)
            b_h1 = bool(h1["E"] <= 1.0)
            b_h2 = bool(h2["E"] <= 1.0)
            b_t = bool(f["n_distinct_t_resolved"] >= 2)
            status = ("PASS" if (b_k32 and b_h1 and b_h2 and b_t)
                      else "NUMERICAL_UNRESOLVED" if not b_t
                      else "RULE_NOT_MET")
            gate_trace.append({
                "scope": scope, "cell": cell, "channel": ch, "status": status,
                "E_K32": f["E"], "bool_K32_E_le_0.8": b_k32,
                "E_first16": h1["E"], "bool_first16_E_le_1": b_h1,
                "E_last16": h2["E"], "bool_last16_E_le_1": b_h2,
                "n_distinct_t_resolved": f["n_distinct_t_resolved"],
                "distinct_t": f["distinct_t"], "bool_distinct_t_ge_2": b_t,
                "SSE_sum_K32": f["SSE_sum"], "TSS_sum_K32": f["TSS_sum"]})

# late per-anchor verdicts (scope-labeled, NOT a unified headline)
late_verdicts = []
for cell in ("R", "C"):
    for ch in ("field", "shadow"):
        k = (cell, "A", 48, ch)
        row = late_by_key[k]
        resolved = row["resolved"]
        e32, h1 = row["perK"]["K32"]["E"], row["halves"]["first16"]["E"]
        h2 = row["halves"]["last16"]["E"]
        if not resolved:
            v = "UNRESOLVED"
        elif e32 <= 0.8 and h1 <= 1.0 and h2 <= 1.0:
            v = "LATE_ANCHOR_RESOLVED_AND_IMPROVED"
        else:
            v = "RULE_NOT_MET"
        late_verdicts.append({"cell": cell, "arm": "A", "anchor": 48,
                              "channel": ch, "E_K32": e32,
                              "E_first16": h1, "E_last16": h2,
                              "resolved": resolved,
                              "signal_ratio": row["signal_ratio"], "verdict": v})

# ---------------------------------------------------------------------------
# write artifacts
# ---------------------------------------------------------------------------

def wcsv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# late_probe_statistics.csv (Table A for the two late anchors + perK/halves)
ta_rows = []
for key in [(c, "A", 48, ch) for c in ("R", "C") for ch in ("field", "shadow")]:
    row = late_by_key[key]
    base = {k: row[k] for k in
            ("cell", "arm", "anchor", "channel", "h", "t_after", "K",
             "target_norm", "target_se", "floor", "signal_ratio", "resolved",
             "pred_norm", "pred_se", "residual_se", "residual_se_note",
             "sse", "tss", "E")}
    ta_rows.append(base)
    for kk, blk in row["perK"].items():
        ta_rows.append({**{k: base[k] for k in
                           ("cell", "arm", "anchor", "channel")},
                        "K": kk, "sse": blk["sse"],
                        "tss": blk["tss"], "E": blk["E"]})
    for hn, blk in row["halves"].items():
        ta_rows.append({**{k: base[k] for k in
                           ("cell", "arm", "anchor", "channel")},
                        "K": hn, "sse": blk["sse"], "tss": blk["tss"],
                        "E": blk["E"]})
wcsv(OUT / "late_probe_statistics.csv", ta_rows,
     ["cell", "arm", "anchor", "channel", "h", "t_after", "K",
      "target_norm", "target_se", "floor", "signal_ratio", "resolved",
      "pred_norm", "pred_se", "residual_se", "residual_se_note",
      "sse", "tss", "E"])

# old-anchor Table A rows (K32; O1 independent recheck at anchor level)
oa_rows = []
for key, row in sorted(old_by_key.items()):
    oa_rows.append({k: row[k] for k in
                    ("cell", "arm", "anchor", "channel", "h", "t_after", "K",
                     "target_norm", "target_se", "floor", "signal_ratio",
                     "resolved", "pred_norm", "pred_se", "residual_se",
                     "residual_se_note", "sse", "tss", "E")})
wcsv(OUT / "old_probe_statistics.csv", oa_rows,
     ["cell", "arm", "anchor", "channel", "h", "t_after", "K",
      "target_norm", "target_se", "floor", "signal_ratio", "resolved",
      "pred_norm", "pred_se", "residual_se", "residual_se_note",
      "sse", "tss", "E"])

# old recheck CSV (O1): per-anchor/channel/K + halves + aggregate cross-check
o1_rows = []
for key, row in old_by_key.items():
    for kk, blk in row["perK"].items():
        o1_rows.append({"cell": row["cell"], "arm": row["arm"],
                        "anchor": row["anchor"], "channel": row["channel"],
                        "group": kk, "sse": blk["sse"], "tss": blk["tss"],
                        "E": blk["E"]})
    for hn, blk in row["halves"].items():
        o1_rows.append({"cell": row["cell"], "arm": row["arm"],
                        "anchor": row["anchor"], "channel": row["channel"],
                        "group": hn, "sse": blk["sse"], "tss": blk["tss"],
                        "E": blk["E"]})
wcsv(OUT / "old_anchor_recheck.csv", o1_rows,
     ["cell", "arm", "anchor", "channel", "group", "sse", "tss", "E"])
wcsv(OUT / "old_aggregate_recheck.csv", agg_check,
     ["cell", "channel", "group", "SSE_sum", "TSS_sum", "E_recomputed",
      "E_published", "match"])

# Table B + gate trace + late verdicts
wcsv(OUT / "aggregate_gate_inputs.csv", tableB,
     ["scope", "cell", "channel", "group", "anchor_IDs", "n_anchor",
      "n_distinct_t_resolved", "SSE_sum", "TSS_sum", "E", "threshold",
      "criterion", "detail"])
(OUT / "criterion_trace.json").write_text(
    json.dumps({"gate_trace": gate_trace, "late_verdicts": late_verdicts,
                "old_aggregate_recheck_match":
                    all(a["match"] for a in agg_check)}, indent=1) + "\n")

# Table C: late error decomposition + stop-v feedback
tc_rows = []
for r in late_recs:
    for ch in ("field", "shadow"):
        d = r["decomposition"][ch]
        z = r["stopv_norms"]["Z"]["audit"]
        q = r["stopv_norms"]["Q"]["audit"]
        zd = r["stopv_pairs"]["Z"]["audit"]["diff_norm_mean_se"]
        tc_rows.append({
            "cell": r["cell"], "arm": r["arm"], "anchor": r["anchor"],
            "channel": ch,
            "total_SS": d["e_norm"] ** 2,
            "bank_SS": d["e_bank_norm"] ** 2,
            "local_SS": d["e_local_norm"] ** 2,
            "signed_cross": 2.0 * d["cross"],
            "identity_residual": d["sq_identity_resid"],
            "shadow_full_signal_norm": z["full"]["mean"],
            "shadow_full_SE": z["full"]["se"],
            "shadow_stop_signal_norm": z["stop"]["mean"],
            "shadow_stop_SE": z["stop"]["se"],
            "shadow_norm_ratio": z["ratio"]["mean"],
            "shadow_mean_cosine": z["cos"]["mean"],
            "shadow_paired_diff_mean": zd["mean"],
            "shadow_paired_diff_SE": zd["se"],
            "Qstop_norm_ratio": q["ratio"]["mean"],
            "Qstop_mean_cosine": q["cos"]["mean"]})
wcsv(OUT / "late_error_decomposition.csv", tc_rows,
     ["cell", "arm", "anchor", "channel", "total_SS", "bank_SS", "local_SS",
      "signed_cross", "identity_residual",
      "shadow_full_signal_norm", "shadow_full_SE",
      "shadow_stop_signal_norm", "shadow_stop_SE",
      "shadow_norm_ratio", "shadow_mean_cosine",
      "shadow_paired_diff_mean", "shadow_paired_diff_SE",
      "Qstop_norm_ratio", "Qstop_mean_cosine"])

# Table D: fixed q+ readout (raw <x, q+> projections; sign convention note)
td_rows = []
for r in late_recs:
    fp = r["fixed_dir_proj"]
    for ch, (pn, yn) in (("field", ("P_D", "Y_A")), ("shadow", ("Q_D", "Z_A"))):
        res = fp["resid_field_audit" if ch == "field"
                 else "resid_shadow_audit"]["q+"]
        td_rows.append({
            "cell": r["cell"], "channel": ch, "qplus_norm": r["q_plus_norm"],
            "pred_mean": fp[pn]["q+"]["mean"], "pred_SE": fp[pn]["q+"]["se"],
            "target_mean": fp[yn]["q+"]["mean"],
            "target_SE": fp[yn]["q+"]["se"],
            "same_probe_residual_mean": res["mean"],
            "residual_SE": res["se"],
            "full_minus_stop_mean": fp["Zstop_A"]["q+"]["mean"]
            if ch == "shadow" else "",
            "full_minus_stop_SE": fp["Zstop_A"]["q+"]["se"]
            if ch == "shadow" else "",
            "paired_full_minus_stop_mean": fp["Zdiff_audit"]["q+"]["mean"]
            if ch == "shadow" else "",
            "paired_full_minus_stop_SE": fp["Zdiff_audit"]["q+"]["se"]
            if ch == "shadow" else "",
            "d_c_to_u_qplus_mean": fp["d_c_to_u"]["q+"]["mean"],
            "sign_convention": ("raw <x, q+>; SHADOW_FIRST_ORDER_READOUT "
                                "= -q+^T x_bar (not a loss)")})
wcsv(OUT / "late_fixed_projection.csv", td_rows,
     ["cell", "channel", "qplus_norm", "pred_mean", "pred_SE",
      "target_mean", "target_SE", "same_probe_residual_mean", "residual_SE",
      "full_minus_stop_mean", "full_minus_stop_SE",
      "paired_full_minus_stop_mean", "paired_full_minus_stop_SE",
      "d_c_to_u_qplus_mean", "sign_convention"])

# update feedback (stop-v paired detail) = subset reuse of late_error_decomp
# (kept as separate mandated file)
uf_rows = []
for r in late_recs:
    for g in ("Q", "Z"):
        for b in ("dev", "audit"):
            sv = r["stopv_norms"][g][b]
            pr = r["stopv_pairs"][g][b]
            uf_rows.append({"cell": r["cell"], "anchor": r["anchor"],
                            "group": f"{g}_{b}",
                            "full_norm_mean": sv["full"]["mean"],
                            "full_norm_SE": sv["full"]["se"],
                            "stop_norm_mean": sv["stop"]["mean"],
                            "stop_norm_SE": sv["stop"]["se"],
                            "norm_ratio_mean": sv["ratio"]["mean"],
                            "norm_ratio_SE": sv["ratio"]["se"],
                            "mean_cosine": sv["cos"]["mean"],
                            "cosine_SE": sv["cos"]["se"],
                            "inner_mean": pr["inner_mean_se"]["mean"],
                            "paired_diff_norm_mean": pr["diff_norm_mean_se"]["mean"],
                            "paired_diff_norm_SE": pr["diff_norm_mean_se"]["se"]})
wcsv(OUT / "late_update_feedback.csv", uf_rows,
     ["cell", "anchor", "group", "full_norm_mean", "full_norm_SE",
      "stop_norm_mean", "stop_norm_SE", "norm_ratio_mean", "norm_ratio_SE",
      "mean_cosine", "cosine_SE", "inner_mean",
      "paired_diff_norm_mean", "paired_diff_norm_SE"])

# old_data_missing.csv
odm = [
    {"item": "C0 half-bank aggregate gate (g3)",
     "status": "DATA_MISSING",
     "note": ("C0 never saved half-bank SSE/TSS; aggregate per-point recheck "
              "impossible. C0X verdict unchanged (erratum, original preserved). "
              "New late data does NOT repair this historical gap.")},
    {"item": "OLD C1 shadow per-probe same-probe residual SE",
     "status": "NOT_SAVED",
     "note": ("Old runner saved field residual norms only. probe_summary.csv "
              "K32 shadow rows carry the FIELD residual value (analyzer fill "
              "artifact) - excluded from any shadow inference.")},
    {"item": "OLD C1 stop-v per-probe norms/ratio/cosine",
     "status": "PARTIAL",
     "note": ("Only mean-level inner and diff-norm saved for old anchors; "
              "per-probe norm ratio and cosine not recoverable offline.")},
    {"item": "A0T - C0 v_eval cross-axis absolute comparison",
     "status": "AXIS_MISMATCH",
     "note": ("v_score full axis vs independent 64-sample v_eval; no shared "
              "axis; constant-offset alignment forbidden. Unchanged.")},
    {"item": "C0 readout ledger h=65 row",
     "status": "NOT_PRESENT (index definition, not imputed)",
     "note": ("63 ledger rows per cell/arm, h in 2..64 (updates 1..63); "
              "intervals [0,4)=[1..4], [4,16)=[5..16], [16,63]=[17..63]; "
              "no interpolation/zero-fill; no new GPU spent on history.")},
    {"item": "OLD C1 per-group sumsq/centered_SS",
     "status": "NOT_SAVED",
     "note": ("Old records saved mean_sq_norm and SE only; per-probe "
              "dispersion (centered_SS) not recomputable beyond the saved SE. "
              "NEW late records save full sufficient statistics.")},
]
wcsv(OUT / "old_data_missing.csv", odm, ["item", "status", "note"])

print("ANALYZE_STAGE1_OK")
print(json.dumps({"late_verdicts": late_verdicts,
                  "gates": gate_trace}, indent=1))
