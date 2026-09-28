#!/usr/bin/env python
"""FR-C0X offline rule recheck. Zero GPU.

Reads C0 frozen outputs and recomputes the verdicts under the strictly-applied
frozen rule (protocol FR_C0_Geometry_Tracking_Validation_Claude_Code.md §10),
without modifying any C0 file. Writes outputs/fr_c0x_offline_explanation_v01/.

X1: half-bank gate is an AGGREGATE over the same resolved anchor set A
    (§10.2 line 366: "使用全库已确定的同一anchor集合"); C0's analyzer applied it
    PER-ANCHOR -> erratum. Aggregate full-bank E is recomputable from saved
    (E_a, y_norm_a); aggregate half-bank E is DATA_MISSING (per-half SSE/TSS
    were never saved; only per-anchor E ratios).
X2: signal resolution table under corrected floor2 (=0.0, same-probe pairing).
X3: same-axis (V_diag) gain / cumulative W / K by interval; V_eval separate;
    A0T axis map (AXIS_MISMATCH kept).
X4: ledger recheck (417 approved, reconciled 17,994; cap overage kept disclosed).
"""
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C0 = os.path.join(REPO, "outputs", "fr_c0_gttr_validation_v01")
OUT = os.path.join(REPO, "outputs", "fr_c0x_offline_explanation_v01")
os.makedirs(OUT, exist_ok=True)

ARMS = ["A", "Bp", "Bm"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- inputs ----------------
floor2 = json.load(open(os.path.join(C0, "smoke_checks_floor2.json")))
floor = {"grad": 0.0, "hvp": 0.0, "v": 0.0, "q": 0.0}
assert abs(floor2["numerical_floor_corrected"]["grad"] - 0.0) < 1e-15
floor_note = ("floor2 (=0.0 on every quantity, same-probe pairing, 4 repeats) supersedes "
              "smoke_checks.json floor (1.65, cross-probe pooling). Same-probe floor=0 measures "
              "bitwise repeatability only; it does NOT bound systematic error (HVP-proxy truncation, "
              "bf16 master weights, Adam eps denominators, 8-probe sampling bias, axis mismatch).")

records = [json.loads(l) for l in open(os.path.join(C0, "runs", "anchor_records.jsonl"))]
summary = json.load(open(os.path.join(C0, "summary.json")))
steps = [json.loads(l) for l in open(os.path.join(C0, "runs", "step_records.jsonl"))]
cell_metrics = list(csv.DictReader(open(os.path.join(C0, "cell_metrics.csv"))))

# ---------------- X1: strict frozen-rule recheck ----------------
def resolved(ym):
    return ym["y_norm"] > 3.0 * math.sqrt(ym["se"] ** 2 + floor["grad"] ** 2)

def aggregate_E(rows, e_key, y_key="y_norm"):
    """Frozen §10.2: E_M = sqrt( sum||p-ybar||^2 / sum||ybar||^2 ) over set A.
    Recomputable from saved per-anchor (E_a, y_a): SSE_a = E_a^2 * y_a^2.
    `rows` are raw anchor records; y lives under the nested y-block."""
    yn = "y_mu" if e_key.endswith("_mu") else "y_U"
    num = sum(r[e_key] ** 2 * r[yn]["y_norm"] ** 2 for r in rows)
    den = sum(r[yn]["y_norm"] ** 2 for r in rows)
    return math.sqrt(num / den), num, den

rule_rows = []
A_by_cell = defaultdict(list)
AU_by_cell = defaultdict(list)
for a in sorted(records, key=lambda r: (r["cell"], ARMS.index(r["arm"]), r["anchor"])):
    ym, yu = a["y_mu"], a["y_U"]
    hb = a["halfbank"]
    r_f = resolved(ym)
    r_u = resolved(yu)
    if r_f:
        A_by_cell[a["cell"]].append(a)
    if r_u:
        AU_by_cell[a["cell"]].append(a)
    rule_rows.append({
        "cell": a["cell"], "arm": a["arm"], "anchor_t": a["anchor"],
        "in_A_field": r_f, "in_A_shadow": r_u,
        "y_norm": ym["y_norm"], "se": ym["se"], "floor_grad": floor["grad"],
        "resolved_gate_value": 3.0 * math.sqrt(ym["se"] ** 2 + floor["grad"] ** 2),
        "E_M0_mu": a["E_M0_mu"], "E_Mgeom_mu": a["E_Mgeom_mu"],
        "sse_mu": a["E_Mgeom_mu"] ** 2 * ym["y_norm"] ** 2,
        "tss_mu": ym["y_norm"] ** 2,
        "halfbank_first4_E_mu": hb["first4"]["E_Mgeom_mu"],
        "halfbank_last4_E_mu": hb["last4"]["E_Mgeom_mu"],
        "halfbank_agg_E_mu": "DATA_MISSING(per-half SSE/TSS not saved; ratios alone cannot be aggregated)",
        "y_U_norm": yu["y_norm"], "y_U_se": yu["se"],
        "E_Mgeom_U": a["E_Mgeom_U"],
        "halfbank_first4_E_U": hb["first4"]["E_Mgeom_U"],
        "halfbank_last4_E_U": hb["last4"]["E_Mgeom_U"],
        "halfbank_agg_E_U": "DATA_MISSING(per-half SSE/TSS not saved)",
    })

agg = {}
for cell in ("R", "C"):
    Af = A_by_cell[cell]
    Au = AU_by_cell[cell]
    e_f, sse_f, tss_f = aggregate_E(Af, "E_Mgeom_mu")
    e_u, sse_u, tss_u = aggregate_E(Au, "E_Mgeom_U", "y_U_norm")
    agg[cell] = {
        "field": {"n": len(Af),
                  "anchors": [f"{a['arm']}@t{a['anchor']}" for a in Af],
                  "arms": sorted({a["arm"] for a in Af}),
                  "distinct_t": sorted({a["anchor"] for a in Af}),
                  "agg_E_Mgeom": e_f, "agg_SSE": sse_f, "agg_TSS": tss_f,
                  "mean_E_arith": sum(a['E_Mgeom_mu'] for a in Af) / len(Af)},
        "shadow": {"n": len(Au),
                   "anchors": [f"{a['arm']}@t{a['anchor']}" for a in Au],
                   "distinct_t": sorted({a["anchor"] for a in Au}),
                   "agg_E_Mgeom_U": e_u, "agg_SSE": sse_u, "agg_TSS": tss_u},
    }

# frozen decision ladder, field channel
gate = {}
for cell in ("R", "C"):
    f = agg[cell]["field"]
    gate[cell] = {
        "g1_signal": (f["n"] >= 6 and len(f["arms"]) >= 2 and len(f["distinct_t"]) >= 2),
        "g2_agg_E_le_0.8": f["agg_E_Mgeom"] <= 0.8,
        "g3_halfbank_agg_E_le_1": "DATA_MISSING",
    }

errata = [{
    "erratum_id": "C0X-X1-01",
    "status": "ERRATUM_REQUIRED",
    "where": "research/scripts/c0_analyze_v01.py geom_verdict() ~L294-306 "
             "(half_ok = all per-anchor halfbank E <= 1 over resolved anchors)",
    "original_verdict_preserved": summary["verdicts"]["local_training_field_prediction"],
    "frozen_rule": ("FR_C0 protocol §10.2 (line 366): 按固定ID将dev和audit各分成前4/后4，"
                    "使用全库已确定的同一anchor集合；两个半库中Mgeom均不劣于各自M0（E_Mgeom<=1），"
                    "不重新选anchor。E_M 的定义（§10.2 主预测指标）是同一 resolved anchor 集合 A 上的"
                    "聚合 SSE 比率 sqrt(sum||p-ybar||^2/sum||ybar||^2)，不是逐点指标。"),
    "misapplication": ("C0 analyzer 对每个 resolved anchor 分别要求 halfbank f4/l4 E<=1（逐点门），"
                       "并把 C/Bp/t16 last4 单点 E=1.166 当作整 cell 半库门失败依据。"),
    "fact_check": ("1.166 是 C/Bp/t16 单点 last-4 比值，不是聚合指标；该点在修正 floor 下属于 A。"),
    "correction": ("按冻结规则，半库门 = A 集合上的聚合 E<=1（f4 与 l4 各一个数 per cell per channel）。"
                   "C0 保存工件只有逐点半库 E 比值，未保存半库 SSE/targetSS，"
                   "聚合半库 E 无法从保存数据精确复算 -> 该门数值为 DATA_MISSING。"
                   "单点 1.166 不构成冻结门失败证据。"),
    "recalculated_field_verdict": "LOCAL_GEOMETRY_UNRESOLVED",
    "recalc_reason": ("冻结门无法完整求值：g1 signal PASS、g2 聚合 E<=0.8 PASS（复算 R=%.4f C=%.4f），"
                      "g3 半库聚合 E DATA_MISSING。按阶梯不能记 SUPPORTED；原 NOT_SUPPORTED 的依据是规则误用。"
                      "UNRESOLVED 的语义此处=门值统计量缺失，非 signal 不足。") % (
                          agg["R"]["field"]["agg_E_Mgeom"], agg["C"]["field"]["agg_E_Mgeom"]),
    "sensitivity_disclosure": ("C/Bp/t16 的 resolved 判定是刀锋：y_norm=0.09191 > 3*SE=0.08458 仅余量 ~7.3e-3。"
                               "该 anchor 占 C cell target SS 的 26.3%%；若将其剔除 A，"
                               "C cell 聚合 E 变为 %.4f（仍 <=0.8）。不做任何重选。") % (
        aggregate_E([a for a in A_by_cell["C"] if not (a["arm"] == "Bp" and a["anchor"] == 16)],
                    "E_Mgeom_mu")[0],),
}]

verdict_comparison = {
    "C0_RULE_REVIEW": {
        "X1_halfbank_gate": "ERRATUM_REQUIRED",
        "X2_floor_correction": "CONSISTENT_RECOMPUTED (floor2 same-probe; all flags recomputed offline)",
        "X3_axis": "UNCHANGED (A0T vs C0-v_eval AXIS_MISMATCH stands)",
        "X4_ledger": "UNCHANGED (417 approved; reconciled 17,994/18,000)",
    },
    "original_verdicts_preserved": summary["verdicts"],
    "recalculated_verdicts": {
        "formula_implementation": summary["verdicts"]["formula_implementation"],
        "local_training_field_prediction": "LOCAL_GEOMETRY_UNRESOLVED",
        "local_training_field_prediction_note": ("erratum C0X-X1-01：原 NOT_SUPPORTED_IN_PILOT 的依据为半库逐点误用；"
                                                 "冻结聚合半库门值 DATA_MISSING，无法确认亦无法证伪。"
                                                 "原判定保留于 C0 summary.json，不覆盖。"),
        "local_shadow_update_prediction": summary["verdicts"]["local_shadow_update_prediction"],
        "local_shadow_update_prediction_note": ("A_U 仅 t48 三点 per cell（1 个 distinct t），"
                                                "聚合 E_U 复算 R=%.4f C=%.4f；门 g1 计数失败 -> UNRESOLVED 不变。") % (
                                                    agg["R"]["shadow"]["agg_E_Mgeom_U"],
                                                    agg["C"]["shadow"]["agg_E_Mgeom_U"]),
        "tracking_signal_resolution": summary["verdicts"]["tracking_signal_resolution"],
        "readout_accounting": summary["verdicts"]["readout_accounting"],
        "future_H16_H64_prediction": "NOT_TESTED",
        "adaptive_training_value": "NOT_TESTED",
        "energy_benefit": "NOT_TESTED",
    },
    "errata": errata,
    "frozen_rule_quotes": {
        "E_definition": "对每个cell，单独计算signal-resolved anchors上的：E_M = sqrt[ sum ||prediction_M - ybar||^2 / sum ||ybar||^2 ]。",
        "halfbank_gate": "按固定ID将dev和audit各分成前4/后4，使用全库已确定的同一anchor集合；两个半库中Mgeom均不劣于各自M0（E_Mgeom<=1），不重新选anchor。",
        "amendment_check": ("live_manifest.json budget_events 仅 CONFIG_FROZEN / THEORY_TESTS_ALL_OK / "
                            "IMPLEMENTATION_FROZEN / BUDGET_ADJUSTMENT_USER_APPROVED 四项，"
                            "CONFIG_FROZEN 之后无规则 amendment；规则优先级 = 冻结 config + 协议原文。"),
    },
}

with open(os.path.join(OUT, "rule_recalculation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rule_rows[0].keys()))
    w.writeheader()
    w.writerows(rule_rows)
json.dump(verdict_comparison, open(os.path.join(OUT, "verdict_comparison.json"), "w"),
          indent=1, ensure_ascii=False)

# ---------------- X2: signal resolution ----------------
geo = {(r["cell"], r["arm"], int(r["anchor_t"])): r for r in
       csv.DictReader(open(os.path.join(C0, "local_geometry_predictions.csv")))}
sho = {(r["cell"], r["arm"], int(r["anchor_t"])): r for r in
       csv.DictReader(open(os.path.join(C0, "shadow_update_predictions.csv")))}
sig_rows = []
for cell in ("R", "C"):
    for arm in ARMS:
        for t in (0, 4, 16, 48):
            g, s = geo[(cell, arm, t)], sho[(cell, arm, t)]
            ym = next(a["y_mu"] for a in records if a["cell"] == cell and a["arm"] == arm and a["anchor"] == t)
            yu = next(a["y_U"] for a in records if a["cell"] == cell and a["arm"] == arm and a["anchor"] == t)
            sig_rows.append({
                "cell": cell, "arm": arm, "anchor_t": t,
                "dev_pred_mean_norm_p_mu": s["p_mu_norm"],
                "audit_target_mean_norm_y_mu": g["y_norm"],
                "se_vec_mu": g["se"],
                "resolved_mu": bool(g["resolved_mu"]),
                "pred_sampling_flag": "NOT_COMPUTED_IN_C0(PREDICTION_SAMPLING_UNRESOLVED never triggered; dev SE over predictions not saved)",
                "shadow_pred_mean_norm_p_U": s["p_U_norm"],
                "shadow_target_mean_norm_y_U": s["y_U_norm"],
                "se_vec_U": s["y_U_se"],
                "resolved_U": bool(s["resolved_U"]),
                "stopv_pred_norm": s["p_stopv_norm"],
                "stopv_target_norm": s["y_stopv_norm"],
                "E_Mstopv": s["E_Mstopv"],
                "clip_active_frac": g["mgeom_clip_active_frac"],
                "kink_margin_min": g["mgeom_kink_margin_min"],
                "hvp_first_grad_rel_gap_max": g["hvp_first_grad_rel_gap_max"],
                "halfbank_mean_diff_norm": "DATA_MISSING(C0 runner saved only per-anchor halfbank E ratios, not half-mean differences)",
                "scalar_projection_err_fixed_dir": "DATA_MISSING(no per-probe projections saved in C0)",
                "U_vs_stopv_direction_dot": "DATA_MISSING(per-probe vectors not saved; only norms)",
            })
with open(os.path.join(OUT, "signal_resolution.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
    w.writeheader()
    w.writerows(sig_rows)

# ---------------- X3: same-axis gain & cumulative W/K ----------------
# step_records: row h = PRE-update state of step h (after h-1 updates), h=1..64.
# W_total_prev(h) = -q_{h-1} . s_{h-1} (step h-1 realized readout), K_V(h) telescopes.
# step-indexed: W_s(t)=W_total_prev(row t+1), K_s(t)=K_V(row t+1); v(t+1)-v(t)=K_s(t)-W_s(t).
# row h=65 does not exist -> state after 64 updates NOT recorded; last interval ends t=63. No interpolation.
by = defaultdict(dict)
for r in steps:
    by[(r["cell"], r["arm"])][int(r["h"])] = r
INTERVALS = [("[0,4)", 1, 4), ("[4,16)", 5, 16), ("[16,63]", 17, 63)]
H_POINTS = [4, 16, 63]

def v_at(cell, arm, t_after):
    return by[(cell, arm)][t_after + 1]["v_diag"]

gb_rows = []
for cell in ("R", "C"):
    for pair in (("Bp", "A"), ("Bm", "A"), ("Bp", "Bm")):
        a, nat = pair
        for (lab, t0, t1) in INTERVALS:
            dW = sum(by[(cell, a)][t + 1]["W_total_prev"] - by[(cell, nat)][t + 1]["W_total_prev"]
                     for t in range(t0, t1 + 1))
            dK = sum(by[(cell, a)][t + 1]["K_V"] - by[(cell, nat)][t + 1]["K_V"]
                     for t in range(t0, t1 + 1))
            gain_wk = dW - dK
            dv = (v_at(cell, nat, t1) - v_at(cell, nat, t0 - 1)) - (v_at(cell, a, t1) - v_at(cell, a, t0 - 1))
            gb_rows.append({"cell": cell, "comparison": f"{a}_minus_{nat}", "axis": "V_diag",
                            "interval": lab, "steps": f"{t0}..{t1}",
                            "sum_dW": dW, "sum_dK": dK, "gain_from_WK": gain_wk,
                            "dV_direct_native_minus_a": dv, "identity_residual": dv - gain_wk})
        for H in H_POINTS:
            cumW = sum(by[(cell, a)][t + 1]["W_total_prev"] - by[(cell, nat)][t + 1]["W_total_prev"]
                       for t in range(1, H + 1))
            cumK = sum(by[(cell, a)][t + 1]["K_V"] - by[(cell, nat)][t + 1]["K_V"]
                       for t in range(1, H + 1))
            gb_rows.append({"cell": cell, "comparison": f"{a}_minus_{nat}", "axis": "V_diag",
                            "interval": f"cumulative_1..{H}", "steps": f"1..{H}",
                            "sum_dW": cumW, "sum_dK": cumK, "gain_from_WK": cumW - cumK,
                            "dV_direct_native_minus_a": v_at(cell, nat, H) - v_at(cell, a, H),
                            "identity_residual": (v_at(cell, nat, H) - v_at(cell, a, H)) - (cumW - cumK)})
cm = {(r["cell"], r["arm"]): r for r in cell_metrics}
for cell in ("R", "C"):
    for pair in (("Bp", "A"), ("Bm", "A"), ("Bp", "Bm")):
        a, nat = pair
        for h in ("4", "16", "64"):
            gb_rows.append({"cell": cell, "comparison": f"{a}_minus_{nat}", "axis": "V_eval",
                            "interval": f"point_h{h}", "steps": "post-update readout",
                            "sum_dW": "", "sum_dK": "",
                            "gain_from_WK": "",
                            "dV_direct_native_minus_a": float(cm[(cell, nat)][f"gain_vs_armA_v_eval_h{h}"])
                            - float(cm[(cell, a)][f"gain_vs_armA_v_eval_h{h}"]),
                            "identity_residual": "NOT_APPLICABLE(no per-step W/K recorded on V_eval axis)"})
with open(os.path.join(OUT, "gain_balance_by_interval.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gb_rows[0].keys()))
    w.writeheader()
    w.writerows(gb_rows)

axis_rows = [
    {"axis": "A0T_cells_axis", "owner": "A0T runs/cells.jsonl",
     "sample_source": "validation_split v_score FULL (a0g._val_loss_only on val_score_tensors)",
     "n_samples": 64, "chunking": "EVAL_BATCH_SIZE=8", "reduction": "mean token CE loss",
     "dtype": "fp32 eval", "used_by": "historical_a0t cross-check in C0 config",
     "note": "same sample POOL as C0 v_diag but different aggregation; absolute offset vs v_eval"},
    {"axis": "C0_v_diag", "owner": "C0 runs/step_records.jsonl",
     "sample_source": "first 8 sample IDs of v_score (=1 chunk of 8, 1024 tokens)",
     "n_samples": 8, "chunking": "1 chunk", "reduction": "mean token CE loss (fp64 accumulate)",
     "dtype": "bf16 model, fp64 inner products", "used_by": "C0 step gains W/K bookkeeping",
     "note": "same-trajectory axis for X3 gain accounting"},
    {"axis": "C0_v_eval", "owner": "C0 cell_metrics.csv",
     "sample_source": "frozen 64-sample v_eval split (disjoint from v_score)",
     "n_samples": 64, "chunking": "8 chunks", "reduction": "mean token CE loss",
     "dtype": "bf16 model", "used_by": "read-only cross-check at h=4/16/64",
     "note": "separate axis; never mixed with v_diag in C0 accounting"},
    {"axis": "B0_stress_native", "owner": "B0 stress runs",
     "sample_source": "training-stream batches", "n_samples": "n/a",
     "chunking": "train micros", "reduction": "train loss", "dtype": "n/a",
     "used_by": "external bitwise identity gate only",
     "note": "not a validation axis; excluded from all loss comparisons"},
    {"axis": "OFFSET_MEASUREMENT", "owner": "C0 REPORT §cross-check",
     "sample_source": "A0T(arm,h) - C0_v_eval(same cell,arm,h)", "n_samples": 15,
     "chunking": "-", "reduction": "-", "dtype": "-", "used_by": "X1 axis audit",
     "note": ("all 15 comparisons differ by ~+0.106 (constant); v_score-64 vs v_eval-64 are DISJOINT "
              "sample sets so the non-constant part is NOT computable offline -> AXIS_MISMATCH stands; "
              "this is an axis definition difference, not trajectory divergence (trajectory identity "
              "independently established by B0 bitwise gate)")},
]
with open(os.path.join(OUT, "target_axis_map.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(axis_rows[0].keys()))
    w.writeheader()
    w.writerows(axis_rows)

# ---------------- X4: ledger recheck ----------------
caps = {"optimizer_updates": 408, "grad_evals": 1600, "hvp_calls": 400, "forward_calls": 2400}
tot = defaultdict(float)
approvals = []
for l in open(os.path.join(C0, "budget_log.jsonl")):
    r = json.loads(l)
    if str(r.get("note", "")).startswith("RECONCILIATION"):
        continue
    for k, v in r.get("counters", {}).items():
        if isinstance(v, (int, float)):
            tot[k] += v
    if r.get("event") == "BUDGET_ADJUSTMENT_USER_APPROVED":
        approvals.append(r)
ledger = {
    "old_cap_keeps_historical_meaning": True,
    "round_approved_total_updates": 417,
    "approvals": approvals,
    "reconciled_c0_totals_skip_reconciliation_rows": dict(tot),
    "project_cumulative": "17,994 / 18,000 (17,577 pre-C0 + 417 this round, disclosed in C0 REPORT)",
    "cap_note": ("per-process counters were within caps; ROUND-TOTAL logged forward_calls 2,428 exceeds the "
                 "2,400 cap by 28 (two zero-update salvage floor phases) — kept disclosed, not hidden by "
                 "per-process accounting"),
    "c0x_gpu_cost": {"optimizer_updates": 0, "grad_evals": 0, "hvp_calls": 0, "forward_calls": 0,
                     "wall_seconds": 0, "note": "C0X is fully offline (this script)"},
    "c1_budget_separate_ledger": {
        "optimizer_updates_cap": 192, "planned": "132 main + 8 smoke + up to 52 reserve",
        "grad_evals_cap": 1400, "hvp_calls_cap": 416, "micro_forward_calls_cap": 3200,
        "gpu_wall_seconds_cap": 5400,
        "rule": ("C1 costs listed separately; never divided by 18,000; '0 optimizer updates' cannot hide "
                 "diagnostic cost; all failures/rechecks/salvage counted")},
}
json.dump(ledger, open(os.path.join(OUT, "ledger_recheck.json"), "w"), indent=1, ensure_ascii=False)

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.8))
xs, ys, cols, mks = [], [], [], []
labels = []
for cell in ("R", "C"):
    for a in A_by_cell[cell] + [x for x in records if x["cell"] == cell and not resolved(x["y_mu"])]:
        pass
# simpler: plot all field anchors; mark membership in A
idx = 0
for cell in ("R", "C"):
    for arm in ARMS:
        for t in (0, 4, 16, 48):
            r = next(x for x in rule_rows if x["cell"] == cell and x["arm"] == arm and x["anchor_t"] == t)
            idx += 1
            inA = r["in_A_field"]
            ax.scatter(idx, r["halfbank_first4_E_mu"], marker="o", c="#1f77b4" if inA else "#aec7e8")
            ax.scatter(idx, r["halfbank_last4_E_mu"], marker="s", c="#d62728" if inA else "#f4b6b6")
            if r["halfbank_last4_E_mu"] > 1.0 or r["halfbank_first4_E_mu"] > 1.0:
                ax.annotate(f"{cell}/{arm}/t{t}\n{'A' if inA else 'not A'}",
                            (idx, max(r['halfbank_first4_E_mu'], r['halfbank_last4_E_mu'])),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.axhline(1.0, ls="--", c="k", lw=1)
ax.set_xlabel("anchor (R: A/Bp/Bm x t0/4/16/48, then C)")
ax.set_ylabel("per-anchor half-bank E_Mgeom (f4 o / l4 s)")
ax.set_title("C0X-X1: frozen half-bank gate is AGGREGATE over A -> aggregate value DATA_MISSING;\n"
             "points shown are per-anchor diagnostics, NOT the gate. Dark = in resolved set A.")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_c0x1_halfbank_gate.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for j, cell in enumerate(("R", "C")):
    ax = axes[j]
    for i, pair in enumerate((("Bp", "A"), ("Bm", "A"), ("Bp", "Bm"))):
        rows = [r for r in gb_rows if r["cell"] == cell and r["comparison"] == f"{pair[0]}_minus_{pair[1]}"
                and r["axis"] == "V_diag" and r["interval"].startswith("[")]
        labs = [r["interval"] for r in rows]
        dW = [r["sum_dW"] for r in rows]
        g = [r["gain_from_WK"] for r in rows]
        ax.bar([x + (i - 1) * 0.26 for x in range(3)], dW, width=0.24,
               label=f"{pair[0]}-{pair[1]} dW", alpha=0.75)
        ax.plot([x + (i - 1) * 0.26 for x in range(3)], g, "ko-", ms=4,
                label=f"{pair[0]}-{pair[1]} gain")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_title(f"cell {cell} (V_diag)")
    if j == 0:
        ax.set_ylabel("native_minus_arm (positive = native worse)")
    ax.legend(fontsize=6.5, loc="best")
fig.suptitle("C0X-X3: cumulative dW and realized gain by interval on V_diag (dK bars omitted for scale; "
             "see gain_balance_by_interval.csv)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_c0x2_gain_intervals.png"), dpi=130)
plt.close(fig)

# ---------------- config + audit ----------------
inputs = {p: sha256(os.path.join(C0, p)) for p in
          ["config.json", "summary.json", "local_geometry_predictions.csv",
           "shadow_update_predictions.csv", "cell_metrics.csv", "readout_balance.csv",
           "smoke_checks.json", "smoke_checks_floor2.json", "budget_log.jsonl"]}
inputs["runs/anchor_records.jsonl"] = sha256(os.path.join(C0, "runs/anchor_records.jsonl"))
inputs["runs/step_records.jsonl"] = sha256(os.path.join(C0, "runs/step_records.jsonl"))
json.dump({
    "experiment_id": "FR_C0X_OFFLINE_EXPLANATION_V01",
    "protocol": "FR_C0X_C1_GTTR_Offline_and_Probe_Validation_Claude_Code.md (C0X part)",
    "gpu": False,
    "floor": floor,
    "floor_note": floor_note,
    "inputs_c0_sha256": inputs,
    "anchor_set_A": {c: agg[c]["field"]["anchors"] for c in ("R", "C")},
    "aggregate_E_recalc": {c: {"field": agg[c]["field"]["agg_E_Mgeom"],
                               "shadow_over_AU": agg[c]["shadow"]["agg_E_Mgeom_U"]}
                           for c in ("R", "C")},
    "halfbank_aggregate": "DATA_MISSING (per-half SSE/TSS not saved in C0 artifacts)",
}, open(os.path.join(OUT, "config.json"), "w"), indent=1, ensure_ascii=False)

print("C0X analysis written to", OUT)
print("aggregate E field  R=%.4f C=%.4f" % (agg["R"]["field"]["agg_E_Mgeom"], agg["C"]["field"]["agg_E_Mgeom"]))
print("aggregate E shadow(AU) R=%.4f C=%.4f" % (agg["R"]["shadow"]["agg_E_Mgeom_U"], agg["C"]["shadow"]["agg_E_Mgeom_U"]))
for cell in ("R", "C"):
    print(cell, "gates:", gate[cell])
print("resolved A:", {c: len(A_by_cell[c]) for c in ("R", "C")},
      "resolved AU:", {c: len(AU_by_cell[c]) for c in ("R", "C")})
