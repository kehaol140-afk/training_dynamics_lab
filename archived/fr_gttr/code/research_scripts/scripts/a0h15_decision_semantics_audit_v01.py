#!/usr/bin/env python3
"""A0H-1.5 round-3 decision-semantics & five-action utility audit (OFFLINE).

Read-only re-analysis of the saved round-2 A' artifacts. No GPU, no new
experiments. Answers: was the A' L2 negative result a selector-semantics
violation (Branch A), a reporting name error (B), an ID/label pairing error
(C), or a genuine finite-action utility failure (D)?

Reads:  outputs/a0h15_vjp_shared_v01/accept_aprime/*.json
        outputs/a0h15_vjp_shared_v01/production/timing_d1_h4_m4.json
        outputs/a0h15_vjp_shared_v01/production/timing_d1_matrix_ext.json
Writes: outputs/a0h15_vjp_shared_v01/decision_semantics_audit/
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APRIME = ROOT / "outputs/a0h15_vjp_shared_v01/accept_aprime"
PROD = ROOT / "outputs/a0h15_vjp_shared_v01/production"
OUT = ROOT / "outputs/a0h15_vjp_shared_v01/decision_semantics_audit"
LAM = 0.05  # FROZEN_LAMBDA (A0G Rule B global lambda, read-only)
UNITS = ("2030_052", "2033_064", "2037_082")
SEED_STEP = {"2030_052": (2030, 52), "2033_064": (2033, 64), "2037_082": (2037, 82)}
HORIZONS = ("1", "4", "8")
MISSING = "MISSING"


def frozen_decision(hhat_by_j):
    """FROZEN A0G selector: predicted Delta L_j = LAM*hhat_j, j=0 -> 0.0,
    argmin with tie-break to lowest j (fr/a0g_decision_utility_v01.py select_action)."""
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        predicted[int(j)] = LAM * float(v)
    return int(min(predicted, key=lambda j: (predicted[j], j)))


def raw_argmax(hhat_by_j):
    """The round-2 program's actual selection (raw-hhat argmax, tie -> lowest j)."""
    return int(max(hhat_by_j, key=lambda j: (hhat_by_j[j], -j)))


def load():
    data = {}
    for label in UNITS:
        data[label] = {
            "diag_bf16": json.loads((APRIME / f"diag_bf16_{label}_s0.json").read_text()),
            "diag_fp32": json.loads((APRIME / f"diag_fp32hvp_{label}_s0.json").read_text()),
            "replay": json.loads((APRIME / f"replay_{label}_s0.json").read_text()),
            "verdict": json.loads((APRIME / f"verdict_{label}_s0.json").read_text()),
        }
    return data


def five_action_rows(data):
    """one row per (unit, H, j) with j=0..4; labels only exist at H=4."""
    rows = []
    for label in UNITS:
        seed, step = SEED_STEP[label]
        v = data[label]["verdict"]
        d16 = data[label]["diag_bf16"]
        d32 = data[label]["diag_fp32"]
        rep = data[label]["replay"]
        stream = int(v["stream_id"])
        chosen_old = int(v["chosen_action_fp32_adj"])
        for hk in HORIZONS:
            h = int(hk)
            bf = {int(j): float(x) for j, x in d16["hhat_fwd_C"][hk].items()}
            f32 = {int(j): float(x) for j, x in d32["hhat_fwd_C"][hk].items()}
            adj = {int(j): float(x) for j, x in d32["vjp_shared_hhat_adj"][hk].items()}
            labels_here = (h == int(v["H_decision"]))
            for j in (0, 1, 2, 3, 4):
                hhat_bf = bf.get(j)
                hhat_f = f32.get(j)
                hhat_a = adj.get(j)
                pred_dl = 0.0 if j == 0 else (LAM * hhat_a if hhat_a is not None else None)
                pred_gain = (0.0 if j == 0 else (-pred_dl if pred_dl is not None else None))
                if labels_here and str(j) in rep["v_score"]:
                    v_base = float(rep["v_score"]["0"])
                    v_act = float(rep["v_score"][str(j)])
                    gain_s = float(rep["gain_vs_noop"][str(j)])
                    d_l = v_act - v_base
                else:
                    v_base = v_act = d_l = gain_s = MISSING
                # V_eval labels: chosen action + no-op only (from verdict JSON)
                if labels_here and j == 0:
                    ve_base = float(v["L2_veval_secondary"]["noop_veval"])
                    ve_act = ve_base
                    gain_e = 0.0
                    d_e = 0.0
                elif labels_here and j == chosen_old:
                    ve_base = float(v["L2_veval_secondary"]["noop_veval"])
                    ve_act = float(v["L2_veval_secondary"]["chosen_action_veval"])
                    gain_e = ve_base - ve_act
                    d_e = ve_act - ve_base
                else:
                    ve_base = ve_act = d_e = gain_e = MISSING
                rows.append({
                    "seed": seed, "step": step, "stream": stream, "H": h,
                    "lambda": LAM, "action_id": j,
                    "candidate_key": ("no_op" if j == 0 else f"group_{j-1}_of_4"),
                    "candidate_order": ("no-op (no candidate batches)" if j == 0
                                        else f"j={j} -> data group g={j-1} (0-based), "
                                             f"Philox candidate_batches_for(stream_seed, g=j-1)"),
                    "noop_score": 0.0,
                    "hhat_frozen_bf16_raw": hhat_bf if hhat_bf is not None else MISSING,
                    "hhat_fp32_forward_raw": hhat_f if hhat_f is not None else MISSING,
                    "hhat_fp32_adjoint_raw": hhat_a if hhat_a is not None else MISSING,
                    "raw_score_semantics": ("dL_score/dlambda at theta_{t+H} (per unit lambda); "
                                            "finite prediction = lambda*hhat (A0G frozen convention)" if j else "no-op: no response readout"),
                    "raw_score_sign": ("+" if (hhat_a or 0) > 0 else "-" if (hhat_a or 0) < 0 else "0") if j else "n/a",
                    "predicted_delta_loss": pred_dl if pred_dl is not None else MISSING,
                    "predicted_gain": pred_gain if pred_gain is not None else MISSING,
                    "original_selector_mode": "raw hhat ARGMAX over j=1..4 (ANTI-selector; round-2 driver)",
                    "original_program_chosen_id": chosen_old,
                    "frozen_A0G_selector_mode": "argmin predicted Delta L=lambda*hhat over j=0..4, tie->lowest j",
                    "frozen_rule_chosen_id": frozen_decision(adj) if h == int(v["H_decision"]) else frozen_decision({jj: adj[jj] for jj in adj}),
                    "bf16_forward_chosen_id": frozen_decision(bf),
                    "fp32_forward_chosen_id": frozen_decision(f32),
                    "fp32_adjoint_chosen_id": frozen_decision(adj),
                    "choose_noop_original": bool(rep["choose_noop"]) if labels_here else MISSING,
                    "choose_noop_frozen_rule": frozen_decision(adj) == 0,
                    "best_active_score": max(adj.values()) if j == 0 else MISSING,
                    "margin_with_noop": (min(LAM * x for x in adj.values()) if j == 0 else MISSING),
                    "tie_flag": any(abs(LAM * adj[a] - LAM * adj[b]) < 1e-15
                                    for a in adj for b in adj if a < b) if j == 0 else MISSING,
                    "tie_break_rule": "min(predicted, key=(value, j)) -> lowest j (frozen A0G)",
                    "V_score_baseline_loss": v_base,
                    "V_score_action_loss": v_act,
                    "delta_L_score": d_l,
                    "Gain_score": gain_s,
                    "V_eval_baseline_loss": ve_base,
                    "V_eval_action_loss": ve_act,
                    "delta_L_eval": d_e,
                    "Gain_eval": gain_e,
                })
    return rows


def ident_checks(rows):
    """mandate §6 identities on rows that carry real labels."""
    out = []
    groups = {}
    for r in rows:
        if r["Gain_score"] != MISSING:
            groups.setdefault((r["seed"], r["step"], r["H"]), []).append(r)
    for key, rs in sorted(groups.items()):
        base = [r for r in rs if r["action_id"] == 0][0]["V_score_baseline_loss"]
        for r in rs:
            if r["action_id"] == 0:
                ok_gain0 = (r["Gain_score"] == 0.0 and r["delta_L_score"] == 0.0)
            else:
                ok_gain0 = None
            recomputed = base - r["V_score_action_loss"]
            ok_id = abs(recomputed - r["Gain_score"]) <= 1e-12
            out.append({"check": "Gain=L_base-L_action", "unit": f"{key[0]}/{key[1]}",
                        "H": key[2], "j": r["action_id"], "ok": ok_id,
                        "detail": f"{recomputed:.12f} vs {r['Gain_score']:.12f}"})
            if ok_gain0 is not None:
                out.append({"check": "no-op Gain=0", "unit": f"{key[0]}/{key[1]}",
                            "H": key[2], "j": 0, "ok": ok_gain0, "detail": "gain=0,delta=0"})
        gains = [r["Gain_score"] for r in rs]
        out.append({"check": "oracle max Gain >= 0", "unit": f"{key[0]}/{key[1]}",
                    "H": key[2], "j": "*", "ok": max(gains) >= 0.0,
                    "detail": f"max={max(gains):.8f}"})
        for r in rs:
            regret = max(gains) - r["Gain_score"]
            out.append({"check": "Regret>=0", "unit": f"{key[0]}/{key[1]}",
                        "H": key[2], "j": r["action_id"], "ok": regret >= -1e-12,
                        "detail": f"regret={regret:.12f}"})
    return out


def rank_matches(data):
    """mandate §7: re-interpret the '9/9 ranking agreement'."""
    out = []
    for label in UNITS:
        seed, step = SEED_STEP[label]
        v = data[label]["verdict"]
        rep = data[label]["replay"]
        d32 = data[label]["diag_fp32"]
        for hk in HORIZONS:
            bf = {int(j): float(x) for j, x in data[label]["diag_bf16"]["hhat_fwd_C"][hk].items()}
            adj = {int(j): float(x) for j, x in d32["vjp_shared_hhat_adj"][hk].items()}
            active_bf = sorted(bf, key=lambda j: (-bf[j], j))
            active_adj = sorted(adj, key=lambda j: (-adj[j], j))
            active_match = active_bf == active_adj
            five_bf = sorted({**bf, 0: 0.0}, key=lambda j: (-({**bf, 0: 0.0}[j]), j))
            five_adj = sorted({**adj, 0: 0.0}, key=lambda j: (-({**adj, 0: 0.0}[j]), j))
            five_match = five_bf == five_adj
            chosen_bf = frozen_decision(bf)
            chosen_adj = frozen_decision(adj)
            rec = {
                "seed": seed, "step": step, "H": int(hk),
                "active_only_rank_match": active_match,
                "five_action_rank_match_raw_score_order": five_match,
                "chosen_id_match_bf16_vs_fp32adj_frozen_rule": chosen_bf == chosen_adj,
                "choose_noop_match_bf16_vs_fp32adj": (chosen_bf == 0) == (chosen_adj == 0),
                "frozen_chosen_bf16": chosen_bf, "frozen_chosen_fp32_adj": chosen_adj,
                "no_op_position_bf16": five_bf.index(0) + 1, "no_op_position_fp32_adj": five_adj.index(0) + 1,
                "raw_argmax_bf16": raw_argmax(bf), "raw_argmax_fp32_adj": raw_argmax(adj),
            }
            if hk == str(v["H_decision"]):
                rec["replay_oracle_chosen_id"] = int(rep["chosen_id"])
                rec["frozen_rule_equals_replay_oracle"] = chosen_adj == int(rep["chosen_id"])
                rec["old_program_chosen_id"] = int(v["chosen_action_fp32_adj"])
                rec["old_program_equals_oracle"] = int(v["chosen_action_fp32_adj"]) == int(rep["chosen_id"])
            else:
                for k in ("replay_oracle_chosen_id", "frozen_rule_equals_replay_oracle",
                          "old_program_chosen_id", "old_program_equals_oracle"):
                    rec[k] = MISSING
            out.append(rec)
    return out


def reanalysis(data):
    """Branch A offline re-scoring with existing labels."""
    out = []
    for label in UNITS:
        seed, step = SEED_STEP[label]
        v = data[label]["verdict"]
        rep = data[label]["replay"]
        d32 = data[label]["diag_fp32"]
        adj = {int(j): float(x) for j, x in d32["vjp_shared_hhat_adj"]["4"].items()}
        bf = {int(j): float(x) for j, x in data[label]["diag_bf16"]["hhat_fwd_C"]["4"].items()}
        corrected = frozen_decision(adj)
        corrected_bf16 = frozen_decision(bf)
        oracle = int(rep["chosen_id"])
        old = int(v["chosen_action_fp32_adj"])
        gains = {int(j): float(g) for j, g in rep["gain_vs_noop"].items()}
        signs = {int(j): (adj[j] < 0) == (gains[j] > 0) for j in adj}
        out.append({
            "unit": f"{seed}/{step}", "stream": 0, "H": 4, "lambda": LAM,
            "old_program_choice": old, "old_gain_V_score": gains[old],
            "corrected_frozen_rule_choice": corrected,
            "corrected_gain_V_score": gains[corrected],
            "corrected_choice_from_bf16_forward": corrected_bf16,
            "replay_oracle_choice": oracle, "oracle_gain": gains[oracle],
            "frozen_rule_matches_oracle": corrected == oracle,
            "proxy_sign_accuracy_H4": f"{sum(signs.values())}/4",
            "V_eval_gain_old_choice": float(v["L2_veval_secondary"]["veval_gain"]),
            "V_eval_gain_corrected_choice": MISSING,
            "note_V_eval": "per-candidate V_eval labels exist only for the old chosen action + no-op",
        })
    return out


def utility_decomposition(data):
    """Branch D partial decomposition (only terms existing records support)."""
    out = []
    for label in UNITS:
        seed, step = SEED_STEP[label]
        rep = data[label]["replay"]
        d32 = data[label]["diag_fp32"]
        adj = {int(j): float(x) for j, x in d32["vjp_shared_hhat_adj"]["4"].items()}
        gains = {int(j): float(g) for j, g in rep["gain_vs_noop"].items()}
        for j in (1, 2, 3, 4):
            pred_gain = -LAM * adj[j]          # term 1: predicted utility
            resid = gains[j] - pred_gain       # terms 2+3 (+transfer N/A for V_score)
            out.append({
                "unit": f"{seed}/{step}", "H": 4, "j": j, "lambda": LAM,
                "term1_predicted_gain_-lambda*hhat": pred_gain,
                "realized_Gain_score": gains[j],
                "total_residual": resid,
                "term2_proxy_vs_native_derivative": "DERIVATIVE_VS_NONLINEARITY_NOT_IDENTIFIABLE",
                "term3_finite_action_remainder": "DERIVATIVE_VS_NONLINEARITY_NOT_IDENTIFIABLE",
                "term4_Vscore_to_Veval_transfer_tau_j": "NOT_RECONSTRUCTABLE (per-candidate V_eval labels missing; "
                                                        "only +lambda branch recorded, no -lambda, no smaller-lambda scan)",
                "identifiability_note": "single +lambda replay only: cannot separate local-derivative error from "
                                        "curvature remainder; A0G lambda scan exists only on A0G formal units, not A' units",
            })
    return out


def timing_recheck():
    t = json.loads((PROD / "timing_d1_h4_m4.json").read_text())
    by = {}
    for r in t["records"]:
        by.setdefault(r["method"], []).append(r["wall_s"])
    order = t["order"]
    med = {m: st.median(w) for m, w in by.items()}
    rows = []
    for m in order:
        w = by[m]
        rows.append({"method": m, "n": len(w), "runs_wall_s": " | ".join(f"{x:.6f}" for x in w),
                     "median_s": f"{st.median(w):.6f}", "mean_s": f"{st.mean(w):.6f}",
                     "std_s": f"{st.stdev(w):.6f}", "min_s": f"{min(w):.6f}", "max_s": f"{max(w):.6f}",
                     "fixed_order_in_run": " > ".join(order),
                     "combos": "H=4,M=4", "repeats": t["repeats"]})
    ratios = {
        "R_structure_T_fp32_forward_over_T_fp32_adjoint": med["fp32_forward"] / med["fp32_adjoint"],
        "R_replay_T_native_replay_over_T_fp32_adjoint": med["replay"] / med["fp32_adjoint"],
        "R_frozen_T_bf16_forward_over_T_fp32_adjoint": med["bf16_forward"] / med["fp32_adjoint"],
        "time_reduction_vs_replay_1_minus_Tadj_over_Treplay": 1.0 - med["fp32_adjoint"] / med["replay"],
        "note": "speedup ~= 1.17x AND time reduction ~= 14.6% are DIFFERENT statements; report both separately",
    }
    me = json.loads((PROD / "timing_d1_matrix_ext.json").read_text())
    mrows = []
    mb = {}
    for r in me["records"]:
        mb.setdefault((r["H"], r["M"], r["method"]), []).append(r["wall_s"])
    for (h, m, meth) in sorted(mb):
        w = mb[(h, m, meth)]
        mrows.append({"H": h, "M": m, "method": meth, "n": len(w),
                      "median_s": f"{st.median(w):.6f}", "runs": " | ".join(f"{x:.4f}" for x in w)})
    gpu_log = (PROD / "gpu1_util_during_timing.log").read_text().strip().splitlines()
    utils = [float(l.split()[-1]) for l in gpu_log if l.split()]
    ratios["gpu_condition"] = (f"GPU1 sampled during H4/M4 window ({len(utils)} samples): "
                               f"avg util {st.mean(utils):.1f}%, max {max(utils)}%; foreign job on GPU0 "
                               f"untouched (empty 0MiB context on GPU1) -> labeled DEV_ENGINEERING_TIMING "
                               f"exclusive-in-practice; NOT a formal deployment performance claim")
    return rows, ratios, mrows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()

    rows = five_action_rows(data)
    with open(OUT / "five_action_decision_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "five_action_decision_audit.json").write_text(json.dumps(rows, indent=1))

    checks = ident_checks(rows)
    n_ok = sum(1 for c in checks if c["ok"])
    (OUT / "identity_checks.json").write_text(json.dumps(
        {"n_checks": len(checks), "n_pass": n_ok,
         "all_pass": n_ok == len(checks), "checks": checks}, indent=1))

    rm = rank_matches(data)
    with open(OUT / "rank_match_reanalysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rm[0].keys()))
        w.writeheader()
        w.writerows(rm)

    ra = reanalysis(data)
    with open(OUT / "aprime_reanalysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ra[0].keys()))
        w.writeheader()
        w.writerows(ra)

    ud = utility_decomposition(data)
    with open(OUT / "utility_error_decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ud[0].keys()))
        w.writeheader()
        w.writerows(ud)

    trows, ratios, mrows = timing_recheck()
    with open(OUT / "timing_round2_recheck.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)
    with open(OUT / "timing_matrix_ext_recheck.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrows[0].keys()))
        w.writeheader()
        w.writerows(mrows)
    (OUT / "timing_ratios.json").write_text(json.dumps(ratios, indent=1))

    contract = {
        "hhat_math_semantics": "hhat_j = d L_score(theta_{t+H}) / d lambda (per unit lambda, Case 1 of round-3 mandate)",
        "action": "g_action = g + sign*lambda*(g_j - g), post-unscale/pre-clip, single optimizer step at h=1; sign=+1 is +lambda",
        "predicted_delta_loss_j": "lambda * hhat_j; j=0 no-op: 0.0 (in candidate set)",
        "predicted_gain_j": "-lambda * hhat_j; no-op gain 0.0",
        "frozen_selector": "argmin_j predicted_delta_loss over j=0..4, tie-break lowest j",
        "equivalences": "argmin predicted_delta_loss == argmin hhat == argmax predicted_gain (lambda>0)",
        "sources": {
            "a0g_action_def": "research/scripts/a0g_decision_utility_v01.py L7-15 (docstring), L576-625 (_run_action_branch)",
            "a0g_hhat_def": "research/scripts/a0g_decision_utility_v01.py L540-548 (hhat = dL(theta_{t+H})/dλ; project_tangent(v_h, r_H))",
            "a0g_selector": "research/src/fr/a0g_decision_utility_v01.py L225-229 select_action; L232-262 level3_decision",
            "a0g_frozen_data_crosscheck": "outputs/fr_decision_utility_v01/level2_action_predictions.csv + level3_decision_metrics.csv: j_hat == argmin pred_delta_C over j=0..4 in 360/360; no-op chosen 6/360",
            "a0h15_forward_scorer": "research/scripts/a0h15_vjp_diagnostics_v01.py L608-616 hhat_fwd_C = <v_h, r_H_C> (same convention); L559 hhat_adj = <w_0, a_j>",
            "replay_gain_def": "research/scripts/a0h15_production_entry_v01.py L434-449 gain_vs_noop = v_noop - v_action; chosen_id = argmax gain (tie -> lowest j); choose_noop iff best gain <= 0",
        },
        "round2_violation": {
            "file": "research/scripts/a0h15_accept_aprime_v01.py (pre-fix hash 937798f6...417fe10, saved in accept_aprime/driver_sha256_before_semantics_fix.txt)",
            "what": "raw-hhat ARGMAX over j=1..4 (ranking() descending + max(key=(score,-j))); no-op excluded from ranking; L2 judged on that action",
            "why_wrong": "hhat = dL/dλ => argmax hhat = max predicted Delta L = worst action; frozen rule is argmin",
        },
        "fixed_driver": "research/scripts/a0h15_accept_aprime_v01.py now uses frozen_decision() = fr.a0g_decision_utility_v01.select_action on {0:0.0, j:LAM*hhat_j}; A′ demoted to dev-investigation per frozen retry rule 2; no A″ this round",
    }
    (OUT / "selector_contract.json").write_text(json.dumps(contract, indent=1, ensure_ascii=False))

    print("wrote audit tables to", OUT)
    print(json.dumps({"identity_checks": f"{n_ok}/{len(checks)}",
                      "ratios": {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in ratios.items() if k != "gpu_condition"}},
                     indent=1))
    print("reanalysis:")
    for r in ra:
        print(" ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
