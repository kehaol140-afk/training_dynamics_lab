#!/usr/bin/env python3
"""A0I-A: frozen-label offline adaptivity diagnostics (CPU only, no GPU).

Reads A0H20 frozen artifacts (predictions / labels / analysis) and produces
per-unit / per-state / per-seed adaptivity tables, action frequencies, regret
decomposition, margin diagnostics and calibration metrics. NEVER rewrites any
frozen label; this script only reads. Controller = fp32hvp_adjoint argmin
(production shared-adjoint scorer). All derived quantities recomputed from the
frozen CSV/JSON-level artifacts (analysis.json + labels + predictions).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0h_corrected_accept_v01"
A_OUT = PROJECT_ROOT / "outputs/a0i_directional_scoring_v01/a0i_a"
LAMBDA = 0.05


def load_json(p: Path):
    return json.loads(p.read_text())


def argmin_tie_low(values: dict) -> int:
    """Frozen A0G selector: min by (value, action_id) with exact comparison."""
    return min((int(k) for k in values), key=lambda j: (values[str(j)], j))


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"slope": float("nan"), "intercept": float("nan")}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return {"slope": slope, "intercept": my - slope * mx}


def reg_metrics(xs, ys):
    n = len(xs)
    m = ols(xs, ys)
    slope, intercept = m["slope"], m["intercept"]
    preds = [slope * x + intercept for x in xs]
    resid = [y - p for y, p in zip(ys, preds)]
    mae = sum(abs(r) for r in resid) / n
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    my = sum(ys) / n
    tss = sum((y - my) ** 2 for y in ys)
    rss = sum(r * r for r in resid)
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {"n": n, "pearson": pearson(xs, ys), "mae": mae, "rmse": rmse,
            "r2": r2, "ols_slope": slope, "ols_intercept": intercept}


def main() -> int:
    analysis = load_json(OUT / "analysis/analysis.json")
    per_unit = analysis["per_unit"]

    units = {}
    for uid, rec in per_unit.items():
        pred = load_json(OUT / f"predictions/predictions_{uid}.json")
        s_lab = load_json(OUT / f"labels/score/score_labels_{uid}.json")
        e_lab = load_json(OUT / f"labels/eval/eval_labels_{uid}.json")
        units[uid] = {"pred": pred, "s_lab": s_lab, "e_lab": e_lab, "a": rec}

    A_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- per-unit adaptivity table ----------------
    unit_rows = []
    for uid in sorted(units):
        u = units[uid]
        pred, s_lab, e_lab, a = u["pred"], u["s_lab"], u["e_lab"], u["a"]
        dls = s_lab["delta_L_score"]          # str keys 0..4
        dle = e_lab["delta_L_eval"]
        gsc = s_lab["gain_vs_noop_score"]
        gev = e_lab["gain_vs_noop_eval"]
        hhat = pred["scorers_H4"]["hhat"]["fp32hvp_adjoint"]
        pred_delta = {str(j): LAMBDA * hhat[str(j)] for j in range(1, 5)}
        pred_delta["0"] = 0.0

        ctrl = a["fp32adj_selected"]
        ctrl_choices = a["choices"]
        eval_oracle = argmin_tie_low(dle)
        score_oracle = a["choices"]["replay_score_oracle"]

        # margins from frozen predictions (fp32adj argmin block)
        marg = pred["choices_pre_eval"]["fp32adj_argmin"]["margin"]
        pred_top1top2 = marg["margin"]
        # best nonzero vs no-op predicted margin (signed, pred_delta scale)
        best_nonzero_j = min(range(1, 5), key=lambda j: (pred_delta[str(j)], j))
        gamma_noop = pred_delta[str(best_nonzero_j)] - 0.0
        # actual top1-top2 gaps on score/eval deltas
        s_sorted = sorted(float(dls[str(j)]) for j in range(5))
        e_sorted = sorted(float(dle[str(j)]) for j in range(5))
        score_top1top2 = s_sorted[1] - s_sorted[0]
        eval_top1top2 = e_sorted[1] - e_sorted[0]

        unit_rows.append({
            "unit_id": uid,
            "train_seed": pred["state"]["train_seed"],
            "step": pred["state"]["step"],
            "stream_id": pred["state"]["stream_id"],
            "controller_selected_id": ctrl,
            "bf16C_selected_id": ctrl_choices["bf16C_argmin"],
            "fp32fwd_selected_id": ctrl_choices["fp32fwd_argmin"],
            "replay_score_oracle_id": score_oracle,
            "eval_oracle_id": eval_oracle,
            "dev_fixed_id": ctrl_choices["dev_fixed"],
            "noop_id": 0,
            "gain_score_selected": gsc[str(ctrl)],
            "gain_eval_selected": gev[str(ctrl)],
            "gain_eval_dev_fixed": a["gain_eval_by_strategy"]["dev_fixed"],
            "gain_eval_random_expectation": a["random_expectation_gain_eval"],
            "gain_eval_score_oracle": a["gain_eval_by_strategy"]["replay_score_oracle"],
            "gain_eval_eval_oracle": gev[str(eval_oracle)],
            "regret_eval": a["regret_eval"],
            "harmful": a["fp32adj_harmful_eval"],
            "noop_selected": a["fp32adj_noop"],
            "oracle_hit_exact": a["oracle_hit_exact"],
            "pred_top1_top2_margin": pred_top1top2,
            "gamma_noop_best_nonzero_pred_delta": gamma_noop,
            "actual_score_top1_top2_gap": score_top1top2,
            "actual_eval_top1_top2_gap": eval_top1top2,
            "delta_gain_vs_dev_fixed": gev[str(ctrl)] - a["gain_eval_by_strategy"]["dev_fixed"],
            "delta_gain_vs_noop": gev[str(ctrl)] - 0.0,
        })

    with open(A_OUT / "adaptivity_unit_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unit_rows[0].keys()))
        w.writeheader()
        w.writerows(unit_rows)

    # ---------------- state / seed aggregation ----------------
    def agg(rows, key):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        out = {}
        for k, rs in groups.items():
            dg_fix = [r["delta_gain_vs_dev_fixed"] for r in rs]
            dg_noop = [r["delta_gain_vs_noop"] for r in rs]
            g_ctrl = [r["gain_eval_selected"] for r in rs]
            g_fix = [r["gain_eval_dev_fixed"] for r in rs]
            out[k] = {
                "n_units": len(rs),
                "controller_gain_eval_mean": sum(g_ctrl) / len(rs),
                "dev_fixed_gain_eval_mean": sum(g_fix) / len(rs),
                "delta_gain_adaptive_vs_dev_fixed_mean": sum(dg_fix) / len(dg_fix),
                "delta_gain_adaptive_vs_dev_fixed_min": min(dg_fix),
                "delta_gain_adaptive_vs_dev_fixed_max": max(dg_fix),
                "delta_gain_adaptive_vs_noop_mean": sum(dg_noop) / len(dg_noop),
                "n_controller_positive": sum(1 for r in rs if r["gain_eval_selected"] > 0),
                "n_harmful": sum(1 for r in rs if r["harmful"]),
            }
        return out

    state_rows = []
    for k, v in sorted(agg(unit_rows, "state_key" if "state_key" in unit_rows[0] else "unit_id").items()):
        pass  # placeholder (unit_id aggregation unused)
    # state key = seed/step (stream averaged)
    for r in unit_rows:
        r["state_key"] = f"{r['train_seed']}_{r['step']:03d}"
    state_agg = agg(unit_rows, "state_key")
    seed_agg = agg(unit_rows, "train_seed")

    with open(A_OUT / "adaptivity_state_table.csv", "w", newline="") as f:
        keys = ["state_key", "n_units", "controller_gain_eval_mean",
                "dev_fixed_gain_eval_mean", "delta_gain_adaptive_vs_dev_fixed_mean",
                "delta_gain_adaptive_vs_dev_fixed_min", "delta_gain_adaptive_vs_dev_fixed_max",
                "delta_gain_adaptive_vs_noop_mean", "n_controller_positive", "n_harmful"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for k in sorted(state_agg):
            row = {"state_key": k}
            row.update(state_agg[k])
            w.writerow(row)

    with open(A_OUT / "adaptivity_seed_table.csv", "w", newline="") as f:
        keys = ["train_seed", "n_units", "controller_gain_eval_mean",
                "dev_fixed_gain_eval_mean", "delta_gain_adaptive_vs_dev_fixed_mean",
                "delta_gain_adaptive_vs_dev_fixed_min", "delta_gain_adaptive_vs_dev_fixed_max",
                "delta_gain_adaptive_vs_noop_mean", "n_controller_positive", "n_harmful"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for k in sorted(seed_agg, key=int):
            row = {"train_seed": k}
            row.update(seed_agg[k])
            w.writerow(row)

    # ---------------- action frequency ----------------
    freq = {}
    freq_seed = {}
    for r in unit_rows:
        freq[r["controller_selected_id"]] = freq.get(r["controller_selected_id"], 0) + 1
        d = freq_seed.setdefault(r["train_seed"], {})
        d[r["controller_selected_id"]] = d.get(r["controller_selected_id"], 0) + 1
    with open(A_OUT / "action_frequency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "action_id", "count", "fraction"])
        n = len(unit_rows)
        for j in sorted(freq):
            w.writerow(["all_12_units", j, freq[j], freq[j] / n])
        for s in sorted(freq_seed, key=int):
            tot = sum(freq_seed[s].values())
            for j in sorted(freq_seed[s]):
                w.writerow([f"seed_{s}", j, freq_seed[s][j], freq_seed[s][j] / tot])
        w.writerow(["all_12_units", "dev_fixed_reference(j=1)", 12, 1.0])

    # ---------------- retrospective best-fixed POSTHOC diagnostic ----------------
    # best fixed action by mean eval gain across the 12 units (uses labels -> POSTHOC)
    mean_gain_by_action = {}
    for j in range(1, 5):
        gs = []
        for uid in sorted(units):
            e_lab = units[uid]["e_lab"]
            gs.append(e_lab["gain_vs_noop_eval"][str(j)])
        mean_gain_by_action[j] = sum(gs) / len(gs)
    best_fixed = max(range(1, 5), key=lambda j: (mean_gain_by_action[j], -j))
    posthoc_rows = []
    for uid in sorted(units):
        a = units[uid]["a"]
        e_lab = units[uid]["e_lab"]
        ctrl = a["fp32adj_selected"]
        posthoc_rows.append({
            "unit_id": uid,
            "controller_gain_eval": e_lab["gain_vs_noop_eval"][str(ctrl)],
            "best_fixed_gain_eval": e_lab["gain_vs_noop_eval"][str(best_fixed)],
            "delta_gain_controller_vs_best_fixed":
                e_lab["gain_vs_noop_eval"][str(ctrl)] - e_lab["gain_vs_noop_eval"][str(best_fixed)],
        })
    posthoc_mean = sum(r["delta_gain_controller_vs_best_fixed"] for r in posthoc_rows) / len(posthoc_rows)

    # ---------------- regret decomposition ----------------
    dec_rows = []
    for uid in sorted(units):
        a = units[uid]["a"]
        e_lab = units[uid]["e_lab"]
        dle = e_lab["delta_L_eval"]
        adj_id = a["fp32adj_selected"]
        rep_id = a["choices"]["replay_score_oracle"]
        oracle_id = argmin_tie_low(dle)
        regret = float(dle[str(adj_id)]) - min(float(dle[str(j)]) for j in range(5))
        term1 = float(dle[str(adj_id)]) - float(dle[str(rep_id)])
        term2 = float(dle[str(rep_id)]) - float(dle[str(oracle_id)])
        dec_rows.append({
            "unit_id": uid,
            "adj_id": adj_id,
            "replay_score_oracle_id": rep_id,
            "eval_oracle_id": oracle_id,
            "regret_eval": regret,
            "term1_adj_minus_replay": term1,
            "term2_replay_minus_oracle": term2,
            "closure_residual": regret - (term1 + term2),
        })
    with open(A_OUT / "regret_decomposition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dec_rows[0].keys()))
        w.writeheader()
        w.writerows(dec_rows)

    # ---------------- margin diagnostics ----------------
    with open(A_OUT / "margin_diagnostics.csv", "w", newline="") as f:
        keys = ["unit_id", "pred_top1_top2_margin", "gamma_noop_best_nonzero_pred_delta",
                "actual_score_top1_top2_gap", "actual_eval_top1_top2_gap",
                "top1_near_tie_flag_analysis"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in unit_rows:
            w.writerow({
                "unit_id": r["unit_id"],
                "pred_top1_top2_margin": r["pred_top1_top2_margin"],
                "gamma_noop_best_nonzero_pred_delta": r["gamma_noop_best_nonzero_pred_delta"],
                "actual_score_top1_top2_gap": r["actual_score_top1_top2_gap"],
                "actual_eval_top1_top2_gap": r["actual_eval_top1_top2_gap"],
                "top1_near_tie_flag_analysis": units[r["unit_id"]]["a"]["top1_near_tie"],
            })

    # ---------------- calibration ----------------
    # (a) selected-action level: x = lambda*ĥ_selected (pred), y = delta_L_score actual
    xs_sel, ys_sel = [], []
    for uid in sorted(units):
        a = units[uid]["a"]
        s_lab = units[uid]["s_lab"]
        ctrl = a["fp32adj_selected"]
        xs_sel.append(a["pred_delta_score_selected"])
        ys_sel.append(float(s_lab["delta_L_score"][str(ctrl)]))
    # (b) all-candidate level: 12 units x 5 actions (no-op included, x=0,y=0)
    xs_all, ys_all = [], []
    for uid in sorted(units):
        pred = units[uid]["pred"]
        s_lab = units[uid]["s_lab"]
        hhat = pred["scorers_H4"]["hhat"]["fp32hvp_adjoint"]
        dls = s_lab["delta_L_score"]
        for j in range(1, 5):
            xs_all.append(LAMBDA * hhat[str(j)])
            ys_all.append(float(dls[str(j)]))
        xs_all.append(0.0)
        ys_all.append(float(dls["0"]))
    # (c) score -> eval transfer on selected action (observational only)
    xs_tr, ys_tr = [], []
    for uid in sorted(units):
        a = units[uid]["a"]
        xs_tr.append(a["delta_L_score_selected"])
        ys_tr.append(a["delta_L_eval_selected"])

    calibration = {
        "pooled_conventions": {
            "selected_level": "12 points (controller-selected action per unit); the single no-op-selected unit contributes (0,0) and is included; unit-weighted (each unit one point)",
            "all_candidate_level": "60 points (12 units x 5 actions); no-op included as (predicted 0, actual 0); unit-weighted, candidates within a unit implicitly equally weighted",
            "transfer_level": "12 points selected-action only; OBSERVATIONAL ONLY, no causal mechanism claim",
        },
        "x_pred_delta_score_vs_y_delta_L_score": {
            "selected_level": reg_metrics(xs_sel, ys_sel),
            "all_candidate_level": reg_metrics(xs_all, ys_all),
        },
        "x_delta_L_score_vs_y_delta_L_eval_selected": reg_metrics(xs_tr, ys_tr),
        "frozen_reference_note": "A0H20 analysis pearson (selected) 0.6348 / (score->eval) 0.4293 - recomputed here from raw labels; small differences are rounding-only",
    }
    (A_OUT / "calibration.json").write_text(json.dumps(calibration, indent=1))

    # ---------------- case study: harmful / noop / max-regret ----------------
    harmful_uids = [r["unit_id"] for r in unit_rows if r["harmful"]]
    noop_uids = [r["unit_id"] for r in unit_rows if r["noop_selected"]]
    maxreg_uid = max(unit_rows, key=lambda r: r["regret_eval"])["unit_id"]
    case_study = {"harmful_units": harmful_uids, "noop_units": noop_uids,
                  "max_regret_unit": maxreg_uid,
                  "coincide": {"harmful_is_noop": bool(set(harmful_uids) & set(noop_uids)),
                               "harmful_is_maxregret": bool(set(harmful_uids) & {maxreg_uid}),
                               "noop_is_maxregret": bool(set(noop_uids) & {maxreg_uid})},
                  "full_five_action": {}}
    for uid in sorted(set(harmful_uids + noop_uids + [maxreg_uid])):
        u = units[uid]
        hhat = u["pred"]["scorers_H4"]["hhat"]["fp32hvp_adjoint"]
        dls = u["s_lab"]["delta_L_score"]
        dle = u["e_lab"]["delta_L_eval"]
        pd_ = {str(j): LAMBDA * hhat[str(j)] for j in range(1, 5)}
        pd_["0"] = 0.0
        rank_pred = sorted(range(5), key=lambda j: (pd_[str(j)], j))
        rank_score = sorted(range(5), key=lambda j: (float(dls[str(j)]), j))
        rank_eval = sorted(range(5), key=lambda j: (float(dle[str(j)]), j))
        case_study["full_five_action"][uid] = {
            "hhat": {str(j): (0.0 if j == 0 else hhat[str(j)]) for j in range(5)},
            "predicted_delta": pd_,
            "delta_L_score": {str(j): float(dls[str(j)]) for j in range(5)},
            "delta_L_eval": {str(j): float(dle[str(j)]) for j in range(5)},
            "ranking_pred": rank_pred,
            "ranking_score": rank_score,
            "ranking_eval": rank_eval,
        }
    (A_OUT / "case_study.json").write_text(json.dumps(case_study, indent=1))

    # ---------------- summary ----------------
    n = len(unit_rows)
    summary = {
        "n_units": n,
        "controller_action_frequency": {str(j): freq.get(j, 0) for j in range(5)},
        "controller_action_frequency_per_seed": {str(s): freq_seed[s] for s in sorted(freq_seed, key=int)},
        "delta_gain_adaptive_vs_dev_fixed": {
            "per_unit": {r["unit_id"]: r["delta_gain_vs_dev_fixed"] for r in unit_rows},
            "per_seed_mean": {str(s): seed_agg[s]["delta_gain_adaptive_vs_dev_fixed_mean"] for s in sorted(seed_agg, key=int)},
            "overall_mean_paired": sum(r["delta_gain_vs_dev_fixed"] for r in unit_rows) / n,
            "state_means": {k: state_agg[k]["delta_gain_adaptive_vs_dev_fixed_mean"] for k in sorted(state_agg)},
        },
        "delta_gain_adaptive_vs_noop": {
            "per_unit": {r["unit_id"]: r["delta_gain_vs_noop"] for r in unit_rows},
            "overall_mean": sum(r["delta_gain_vs_noop"] for r in unit_rows) / n,
        },
        "retrospective_best_fixed_POSTHOC_DIAGNOSTIC_ONLY": {
            "mean_gain_by_action": {str(j): mean_gain_by_action[j] for j in range(1, 5)},
            "best_fixed_action": best_fixed,
            "delta_gain_controller_vs_best_fixed_mean": posthoc_mean,
            "per_unit": posthoc_rows,
        },
        "regret_decomposition": {
            "mean_regret": sum(r["regret_eval"] for r in dec_rows) / n,
            "mean_term1_adj_minus_replay": sum(r["term1_adj_minus_replay"] for r in dec_rows) / n,
            "mean_term2_replay_minus_oracle": sum(r["term2_replay_minus_oracle"] for r in dec_rows) / n,
            "max_abs_closure_residual": max(abs(r["closure_residual"]) for r in dec_rows),
        },
        "case_study": {"harmful": harmful_uids, "noop": noop_uids, "max_regret": maxreg_uid,
                       "coincide": case_study["coincide"]},
        "oracle": {
            "oracle_hit_exact_rate": sum(1 for r in unit_rows if r["oracle_hit_exact"]) / n,
            "mean_regret": sum(r["regret_eval"] for r in unit_rows) / n,
        },
        "A0I_A_DIAGNOSTIC_COMPLETE": "YES",
    }
    (A_OUT / "a0i_a_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
