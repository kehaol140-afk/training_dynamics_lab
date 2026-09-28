#!/usr/bin/env python3
"""A0H-2.0 corrected-acceptance pilot: P4 analysis (reads eval ONLY after all
choices are locked in P1/P3; separate status fields per contract).

Inputs (all under outputs/a0h_corrected_accept_v01/):
  predictions/predictions_<uid>.json   (P1, score-side)
  gate.json                            (P2)
  labels/score/score_labels_<uid>.json (P3 score labels + replay-score oracle)
  labels/eval/eval_labels_<uid>.json   (P3 eval labels, hash-locked to score)
  acceptance_manifest.json

Outputs: analysis/analysis.json, analysis/summary.json
Status fields are SEPARATE (never a composite PASS).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0h_corrected_accept_v01"
ANALYSIS = OUT / "analysis"

H_DEC = 4
LAM = 0.05
BOOT_N = 10000
BOOT_RNG = 260916
NEAR_TIE_FRACTION = 0.05


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_all(manifest):
    units = manifest["units"]
    preds, score_labels, eval_labels = {}, {}, {}
    for u in units:
        uid = u["unit_id"]
        p = OUT / "predictions" / f"predictions_{uid}.json"
        s = OUT / "labels" / "score" / f"score_labels_{uid}.json"
        e = OUT / "labels" / "eval" / f"eval_labels_{uid}.json"
        if p.exists():
            preds[uid] = json.loads(p.read_text())
        if s.exists():
            score_labels[uid] = json.loads(s.read_text())
        if e.exists():
            eval_labels[uid] = json.loads(e.read_text())
    gate = json.loads((OUT / "gate.json").read_text()) if (OUT / "gate.json").exists() else None
    return units, preds, score_labels, eval_labels, gate


def frozen_decision(hhat_by_j: dict, lam: float) -> int:
    from fr.a0g_decision_utility_v01 import select_action as fsa
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"NON_FINITE j={j}")
        predicted[int(j)] = float(lam) * v
    return int(fsa(predicted))


def five_rank(hhat_by_j: dict, lam: float) -> list:
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        predicted[int(j)] = float(lam) * float(v)
    return [int(j) for j in sorted(predicted, key=lambda j: (predicted[j], j))]


def strategy_choices(uid, preds, score_labels, dev_fixed):
    pr = preds[uid]
    sc = score_labels[uid]
    ch = {
        "noop": 0,
        "bf16C_argmin": int(pr["choices_pre_eval"]["bf16C_argmin"]["chosen_id"]),
        "fp32fwd_argmin": int(pr["choices_pre_eval"]["fp32fwd_argmin"]["chosen_id"]),
        "fp32adj_argmin": int(pr["choices_pre_eval"]["fp32adj_argmin"]["chosen_id"]),
        "dev_fixed": int(dev_fixed),
    }
    gains = {int(j): v for j, v in sc["gain_vs_noop_score"].items()}
    best = max(gains, key=lambda j: (gains[j], -j))
    ch["replay_score_oracle"] = 0 if gains[best] <= 0.0 else int(best)
    return ch


def hierarchical_bootstrap(unit_values: dict, n=BOOT_N, rng_seed=BOOT_RNG):
    """unit_values: uid -> value. seed->state->stream hierarchy from the manifest
    (6 seeds x 1 state x 2 streams). Equal weight over seeds."""
    by_seed = {}
    for uid, v in unit_values.items():
        seed = uid.split("_")[0]
        by_seed.setdefault(seed, []).append(v)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(rng_seed)
    stats = []
    for _ in range(n):
        acc = []
        for s in seeds:
            pool = by_seed[s]
            idx = rng.integers(0, len(pool), size=len(pool))
            acc.append(float(np.mean([pool[i] for i in idx])))
        stats.append(float(np.mean(acc)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(np.mean(list(unit_values.values()))),
            "boot_mean": float(np.mean(stats)),
            "pi95": [float(lo), float(hi)], "n_boot": n, "rng_seed": rng_seed,
            "design": "seed->state->stream resample, equal-weight seed mean",
            "interpretation": "exploratory uncertainty only; NOT a p-value"}


def main():
    manifest = json.loads((OUT / "acceptance_manifest.json").read_text())
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    units, preds, score_labels, eval_labels, gate = load_all(manifest)
    dev_fixed = int(manifest["dev_fixed"]["action_id"])
    uids = [u["unit_id"] for u in units]

    # ---------- completeness / PROTOCOL_VALID ----------
    missing = []
    for uid in uids:
        if uid not in preds:
            missing.append(("predictions", uid))
        if uid not in score_labels:
            missing.append(("score_labels", uid))
        if uid not in eval_labels:
            missing.append(("eval_labels", uid))
    complete = not missing

    pv_checks = {"all_units_present": complete, "n_units": len(uids),
                 "n_predictions": len(preds), "n_score_labels": len(score_labels),
                 "n_eval_labels": len(eval_labels)}
    recomp_ok = True
    eval_lock_ok = True
    five_ok = True
    finite_ok = True
    for uid in uids:
        if uid not in preds or uid not in score_labels or uid not in eval_labels:
            continue
        pr, sc, ev = preds[uid], score_labels[uid], eval_labels[uid]
        hhat = {int(j): v for j, v in pr["scorers_H4"]["hhat"]["fp32hvp_adjoint"].items()}
        if frozen_decision(hhat, LAM) != int(pr["choices_pre_eval"]["fp32adj_argmin"]["chosen_id"]):
            recomp_ok = False
        if sorted(sc["v_score"]) != ["0", "1", "2", "3", "4"] or \
           sorted(ev["v_eval"]) != ["0", "1", "2", "3", "4"]:
            five_ok = False
        if sha256_file(OUT / "labels" / "score" / f"score_labels_{uid}.json") \
                != ev["score_labels_sha256_at_lock"]:
            eval_lock_ok = False
        for coll in (sc["v_score"], sc["v_eval"] if "v_eval" in sc else {},
                     ev["v_eval"], pr["scorers_H4"]["hhat"]["fp32hvp_adjoint"]):
            for v in coll.values():
                if not math.isfinite(float(v)):
                    finite_ok = False
    pv_checks.update({
        "choices_recomputable_from_predictions": recomp_ok,
        "five_action_labels_complete": five_ok,
        "eval_lock_hash_matches_score_file": eval_lock_ok,
        "all_reported_values_finite": finite_ok,
        "gate_thresholds_parseable": gate is not None and all(
            math.isfinite(float(g["threshold"])) and float(g["threshold"]) >= 0
            for u in gate["units"].values() for g in u["per_action"].values()) if gate else False,
        "split_pairing": "validation_split v_score/v_eval frozen a0g split; "
                         "score labels computed in-trajectory before eval labels (hash lock)",
        "selection_by_identity_only": "see exposure_ledger.json",
    })
    PROTOCOL_VALID = "PASS" if all([complete, recomp_ok, five_ok, eval_lock_ok, finite_ok,
                                    pv_checks["gate_thresholds_parseable"]]) else "FAIL"

    # ---------- NUMERICAL_GATE ----------
    NUMERICAL_GATE = gate["status"] if gate else "NOT_EVALUABLE"
    gate_pass = bool(gate and gate["all_pass"])

    # ---------- COMPLETION_STATUS ----------
    if not complete:
        COMPLETION = "INCOMPLETE"
    elif not gate_pass:
        COMPLETION = "STOPPED_AT_NUMERICAL_GATE"
    else:
        COMPLETION = "COMPLETE"

    # ---------- DECISION_PRESERVATION ----------
    def preservation(a_name, b_name):
        rec = {"a": a_name, "b": b_name, "units": {}}
        n_rank = n_choice = n_pos = 0
        for uid in uids:
            if uid not in preds:
                continue
            ha = {int(j): float(v) for j, v in
                  preds[uid]["scorers_H4"]["hhat"][a_name].items()}
            hb = {int(j): float(v) for j, v in
                  preds[uid]["scorers_H4"]["hhat"][b_name].items()}
            ra, rb = five_rank(ha, LAM), five_rank(hb, LAM)
            ca, cb = frozen_decision(ha, LAM), frozen_decision(hb, LAM)
            pos_a, pos_b = ra.index(0), rb.index(0)
            diffs = {str(j): hb[j] - ha[j] for j in sorted(ha)}
            rec["units"][uid] = {
                "ranking_a": ra, "ranking_b": rb, "rank_match": ra == rb,
                "chosen_a": ca, "chosen_b": cb, "chosen_match": ca == cb,
                "noop_position_a": pos_a, "noop_position_b": pos_b,
                "noop_position_match": pos_a == pos_b,
                "score_diffs_b_minus_a": diffs,
                "exact_ties_a": len(set(float(LAM * v) for v in ha.values()) | {0.0})
                                < 5,
            }
            n_rank += int(ra == rb)
            n_choice += int(ca == cb)
            n_pos += int(pos_a == pos_b)
        n = len(rec["units"])
        rec["agreement"] = {"rank_match": f"{n_rank}/{n}", "chosen_id_match": f"{n_choice}/{n}",
                            "noop_position_match": f"{n_pos}/{n}"}
        return rec

    pres_same = preservation("fp32hvp_fwd", "fp32hvp_adjoint")
    pres_cross = preservation("bf16_fwd_C", "fp32hvp_adjoint")
    DECISION_PRESERVATION = {
        "same_config_fp32fwd_vs_fp32adj": pres_same,
        "cross_config_bf16C_vs_fp32adj": pres_cross,
        "note": ("near-tie margins reported in predictions (diagnostic only); "
                 "exact ties resolved by frozen lowest-id rule; cross-precision "
                 "differences keep each method's own V_eval utility"),
    }

    # ---------- utility analysis (only if complete + gate pass) ----------
    UTILITY = "NOT_COVERED"
    per_unit = {}
    main_endpoint = None
    if complete and gate_pass:
        rows = {}
        for uid in uids:
            sc, ev = score_labels[uid], eval_labels[uid]
            choices = strategy_choices(uid, preds, score_labels, dev_fixed)
            ge = {int(j): float(v) for j, v in ev["gain_vs_noop_eval"].items()}
            gs = {int(j): float(v) for j, v in sc["gain_vs_noop_score"].items()}
            ve = {int(j): float(v) for j, v in ev["v_eval"].items()}
            dl_eval = {int(j): float(v) for j, v in ev["delta_L_eval"].items()}
            dl_score = {int(j): float(v) for j, v in sc["delta_L_score"].items()}
            row = {"choices": choices,
                   "gain_eval_by_strategy": {name: ge[j] for name, j in choices.items()},
                   "random_expectation_gain_eval": float(np.mean([ge[j] for j in sorted(ge)]))}
            sel = choices["fp32adj_argmin"]
            best_eval = min(ve.values())
            tol = max(1e-12, NEAR_TIE_FRACTION * (max(ve.values()) - best_eval))
            tied = sorted(j for j in ve if ve[j] - best_eval <= tol)
            hhat = {int(j): float(v) for j, v in
                    preds[uid]["scorers_H4"]["hhat"]["fp32hvp_adjoint"].items()}
            pred_delta_sel = LAM * hhat.get(sel, 0.0) if sel >= 1 else 0.0
            row.update({
                "fp32adj_selected": sel,
                "fp32adj_gain_eval": ge[sel],
                "fp32adj_harmful_eval": bool(ge[sel] < 0.0),
                "fp32adj_noop": bool(sel == 0),
                "regret_eval": dl_eval[sel] - min(dl_eval.values()),
                "oracle_hit_exact": bool(sel == min(ve, key=lambda j: (ve[j], j))),
                "top1_near_tie": bool(sel in tied), "tied_optimal_eval": tied,
                "within_state_eval_rank": sorted(dl_eval, key=lambda j: (dl_eval[j], j)).index(sel) + 1,
                "finite_action_prediction_error": pred_delta_sel - dl_score.get(sel, 0.0),
                "pred_delta_score_selected": pred_delta_sel,
                "delta_L_score_selected": dl_score.get(sel, 0.0),
                "delta_L_eval_selected": dl_eval[sel],
            })
            per_unit[uid] = row

        main_values = {uid: per_unit[uid]["fp32adj_gain_eval"] for uid in per_unit}
        main_endpoint = hierarchical_bootstrap(main_values)
        paired = {}
        for strat in ("noop", "bf16C_argmin", "fp32fwd_argmin", "replay_score_oracle",
                      "dev_fixed"):
            dv = {uid: per_unit[uid]["fp32adj_gain_eval"]
                       - per_unit[uid]["gain_eval_by_strategy"][strat] for uid in per_unit}
            paired[strat] = hierarchical_bootstrap(dv)
        dv_rand = {uid: per_unit[uid]["fp32adj_gain_eval"]
                        - per_unit[uid]["random_expectation_gain_eval"] for uid in per_unit}
        paired["random_expectation"] = hierarchical_bootstrap(dv_rand)

        gains = np.array([per_unit[uid]["fp32adj_gain_eval"] for uid in per_unit])
        UTILITY = ("POSITIVE_POINT_ESTIMATE" if float(gains.mean()) > 0
                   else "NEGATIVE_POINT_ESTIMATE" if float(gains.mean()) < 0
                   else "ZERO")

        rows = {
            "main_endpoint_fp32adj_gain_eval": main_endpoint,
            "paired_gain_diffs_vs_baselines": paired,
            "harmful_rate": float(np.mean([per_unit[u]["fp32adj_harmful_eval"] for u in per_unit])),
            "noop_fraction_fp32adj": float(np.mean([per_unit[u]["fp32adj_noop"] for u in per_unit])),
            "regret_eval_mean": float(np.mean([per_unit[u]["regret_eval"] for u in per_unit])),
            "regret_eval_max": float(np.max([per_unit[u]["regret_eval"] for u in per_unit])),
            "oracle_hit_rate_exact": float(np.mean([per_unit[u]["oracle_hit_exact"] for u in per_unit])),
            "top1_near_tie_rate": float(np.mean([per_unit[u]["top1_near_tie"] for u in per_unit])),
            "finite_action_prediction_error_mean": float(np.mean(
                [per_unit[u]["finite_action_prediction_error"] for u in per_unit])),
            "finite_action_prediction_error_absmean": float(np.mean(
                [abs(per_unit[u]["finite_action_prediction_error"]) for u in per_unit])),
            "per_seed": {},
            "per_state": {},
        }
        for uid in per_unit:
            seed = uid.split("_")[0]
            rows["per_seed"].setdefault(seed, []).append(per_unit[uid]["fp32adj_gain_eval"])
            rows["per_state"][uid] = per_unit[uid]["fp32adj_gain_eval"]
        for seed in rows["per_seed"]:
            rows["per_seed"][seed] = {"mean": float(np.mean(rows["per_seed"][seed])),
                                      "n_streams": len(rows["per_seed"][seed])}

        # observational localization ONLY (no causal claims)
        pds = np.array([per_unit[u]["pred_delta_score_selected"] for u in per_unit])
        dls = np.array([per_unit[u]["delta_L_score_selected"] for u in per_unit])
        dle = np.array([per_unit[u]["delta_L_eval_selected"] for u in per_unit])
        loc = {"observational_only": True,
               "no_claim_without_magnitude_or_native_probe": True}
        if len(set(dls)) > 1 and len(set(pds)) > 1:
            loc["pearson_pred_delta_score_vs_delta_L_score"] = float(np.corrcoef(pds, dls)[0, 1])
        if len(set(dls)) > 1 and len(set(dle)) > 1:
            loc["pearson_delta_L_score_vs_delta_L_eval"] = float(np.corrcoef(dls, dle)[0, 1])
        rows["localization"] = loc
        analysis_rows = rows
    else:
        analysis_rows = None

    analysis = {
        "run_id": manifest["run_id"],
        "n_units_planned": len(uids),
        "COMPLETION_STATUS": COMPLETION,
        "PROTOCOL_VALID": PROTOCOL_VALID,
        "PROTOCOL_VALID_checks": pv_checks,
        "NUMERICAL_GATE": NUMERICAL_GATE,
        "gate_detail": gate,
        "DECISION_PRESERVATION": DECISION_PRESERVATION,
        "UTILITY_PILOT_RESULT": UTILITY,
        "status_note": ("fields are separate by contract; a STOPPED/NOT_COVERED value is "
                        "not a protocol violation"),
        "missing": [f"{k}:{u}" for k, u in missing],
        "per_unit": per_unit,
        "utility": analysis_rows,
    }
    (ANALYSIS / "analysis.json").write_text(json.dumps(analysis, indent=1))
    summary = {k: analysis[k] for k in ("COMPLETION_STATUS", "PROTOCOL_VALID",
                                        "NUMERICAL_GATE", "UTILITY_PILOT_RESULT",
                                        "n_units_planned", "missing")}
    if analysis_rows:
        summary["main_endpoint_fp32adj_gain_eval"] = analysis_rows["main_endpoint_fp32adj_gain_eval"]
        summary["harmful_rate"] = analysis_rows["harmful_rate"]
        summary["noop_fraction_fp32adj"] = analysis_rows["noop_fraction_fp32adj"]
        summary["per_seed"] = analysis_rows["per_seed"]
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
