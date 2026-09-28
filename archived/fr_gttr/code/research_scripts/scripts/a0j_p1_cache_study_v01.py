#!/usr/bin/env python3
"""A0J-P1 measurement: three-arm exact/cache/stale comparison on frozen dev units.

Units: 3 anchors x 2 streams from the FROZEN cache_dev_manifest.json (selected
before any cache result was viewed). Per unit: anchor pass (age 0) captures
p_tau / w_tau; aged passes at ages {1,2,4} inject cached p / stale w with fresh
G_t. Writes cached_scores / gap_diagnostics / decision_preservation /
cache_vector_diagnostics (secondary) / action_vjp_gate / cache_timing_* /
cache_memory.json, then freezes decisions_frozen.json + .sha256 BEFORE replay.

Frozen tolerance (A0H20/A0I, never adjusted after viewing):
  elementwise |new - ref| <= atol32 + rtol32 * |ref|, atol32=1e-5, rtol32=1e-4.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"
ATOL32 = 1e-5
RTOL32 = 1e-4
AGES = (1, 2, 4)

import a0h15_production_entry_v01 as pe  # noqa: E402
import a0g_decision_utility_v01 as a0g  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import (  # noqa: E402
    aged_decision_pass, apply_Gt_T, choose, clone_p, cosine_maps, cosine_p,
    gap_table, map_bytes, max_abs_score_diff, norm_map, norm_p, p_bytes,
    rank_agreement, ranking, score_actions, top_margin,
)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _budget(entries, phase, t0, kind="METERED"):
    entries.append({"phase": phase, "seconds": time.perf_counter() - t0, "kind": kind})


def _worst_tol_violation(torch, ref_map, new_map) -> float:
    """max over elements of |new-ref| - (atol + rtol*|ref|); <= 0 means PASS."""
    worst = -float("inf")
    for name, ref in ref_map.items():
        r = ref.detach().to(torch.float64)
        n = new_map[name].detach().to(torch.float64)
        viol = float(torch.max(torch.abs(n - r) - (ATOL32 + RTOL32 * torch.abs(r))))
        worst = max(worst, viol)
    return float(worst)


def _map_maxabs(torch, m) -> float:
    return float(max(float(torch.max(torch.abs(v.detach().to(torch.float64)))) for v in m.values()))


def _tick_block(torch, fn, repeats=5):
    outs = []
    for rep in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        outs.append((time.perf_counter() - t0, out))
    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--timing-repeats", type=int, default=5)
    args = ap.parse_args()

    manifest = json.loads((OUT / "cache_dev_manifest.json").read_text())
    session_t0 = time.perf_counter()
    budget: list = []

    rt = pe.build_runtime(args.device)
    torch = rt["torch"]
    _budget(budget, "preload_runtime", session_t0, kind="NON_METERED_PRELOAD_GPU_HELD")
    grad_clip = float(rt["cfg"]["grad_clip"])

    score_rows, gap_rows, pres_rows, vec_rows, gate_rows = [], [], [], [], []
    timing_rows, decisions = [], {}

    for anchor in manifest["anchors"]:
        key = f"{anchor['train_seed']}/{anchor['state_step']}"
        ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == key)
        for srec in anchor["streams"]:
            sid = int(srec["stream_id"])
            stream = next(s for s in ctx.streams if int(s["stream_id"]) == sid)
            unit = f"{key}/s{sid}"
            u_t0 = time.perf_counter()

            # ---------------- anchor pass (age 0): capture p_tau, w_tau ----------
            anchor_pass = aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                                             want_stage_times=True)
            p_tau = clone_p(anchor_pass["p_t"])
            w_tau = {k: v.detach().clone() for k, v in anchor_pass["w_exact"].items()}
            g_anchor = anchor_pass["gate"]
            h_ex = anchor_pass["scores"]["exact"]
            w_ref = _map_maxabs(torch, anchor_pass["w_exact"])

            # production reproduction: frozen run_adjoint_only on the SAME unit
            prod = pe.run_adjoint_only(rt, ctx, stream, args.H, args.M, True,
                                       pe._new_counters())
            h_prod = {0: 0.0}
            for j, v in prod["scores_hhat_adj"].items():
                h_prod[int(j)] = float(v)
            prod_chosen = choose(h_prod)
            gate_rows.append({
                "unit": unit, "age": 0, "kind": "gate_apply_Gt_T_vs_inline",
                "max_abs_w_diff": g_anchor["max_abs_w_diff"], "w_ref_maxabs": w_ref,
                "within_tolerance": bool(g_anchor["max_abs_w_diff"] <= ATOL32 + RTOL32 * w_ref),
                "max_abs_score_diff": g_anchor["max_abs_score_diff"],
                "rank_agreement": g_anchor["rank_agreement"],
                "chosen_id_match": g_anchor["chosen_id_match"]})
            score_diff_prod = max_abs_score_diff(h_ex, h_prod)
            score_ref = max(abs(v) for v in h_ex.values())
            gate_rows.append({
                "unit": unit, "age": 0, "kind": "production_reproduction_run_adjoint_only",
                "max_abs_w_diff": None, "w_ref_maxabs": prod["w_0_norm"],
                "within_tolerance": bool(score_diff_prod <= ATOL32 + RTOL32 * score_ref),
                "max_abs_score_diff": score_diff_prod,
                "rank_agreement": rank_agreement(ranking(h_ex), ranking(h_prod)),
                "chosen_id_match": int(choose(h_ex)) == int(prod_chosen)})

            decisions[unit] = {"anchor": key, "stream_id": sid,
                               "ages": {"0": {"chosen": anchor_pass["chosen"],
                                              "h": {k: v for k, v in anchor_pass["scores"].items()}}},
                               "bytes": {"p_tau": p_bytes(p_tau), "w_tau": map_bytes(w_tau)}}
            _record_pass(anchor_pass, unit, 0, score_rows, gap_rows, pres_rows, vec_rows,
                         p_tau, anchor_pass["p_t"], torch)

            # ---------------- aged passes --------------------------------------
            for age in AGES:
                ap_t0 = time.perf_counter()
                aged = aged_decision_pass(pe, rt, ctx, stream, age=age, H=args.H, M=args.M,
                                          cached_p=p_tau, cached_w=w_tau,
                                          want_stage_times=True)
                _record_pass(aged, unit, age, score_rows, gap_rows, pres_rows, vec_rows,
                             p_tau, aged["p_t"], torch)
                g = aged["gate"]
                gate_rows.append({
                    "unit": unit, "age": age, "kind": "gate_apply_Gt_T_vs_inline",
                    "max_abs_w_diff": g["max_abs_w_diff"],
                    "w_ref_maxabs": _map_maxabs(torch, aged["w_exact"]),
                    "within_tolerance": bool(g["max_abs_w_diff"] <= ATOL32 + RTOL32 *
                                             _map_maxabs(torch, aged["w_exact"])),
                    "max_abs_score_diff": g["max_abs_score_diff"],
                    "rank_agreement": g["rank_agreement"],
                    "chosen_id_match": g["chosen_id_match"]})
                decisions[unit]["ages"][str(age)] = {"chosen": aged["chosen"],
                                                     "h": {k: v for k, v in aged["scores"].items()}}
                # timing decomposition from this pass
                t = aged["times"]; wt = aged["walk_times"]
                ff = sum(v for k, v in t.items() if k.startswith("ff_step_"))
                walk = sum(v for k, v in wt.items() if k.startswith("e1_") or
                           k.startswith("e2_") or k.startswith("e3_"))
                timing_rows.append({
                    "unit": unit, "age": age,
                    "T_refresh_s": ff + t.get("pending_fwd_bwd", 0.0) +
                        t.get("rollout_rest", 0.0) + t.get("endpoint_grad", 0.0) + walk,
                    "T_GTp_cached_s": t.get("gtTp_cached"),
                    "T_candidate_score_s": t.get("candidate_grads", 0.0) + t.get("scoring_dots", 0.0),
                    "T_native_step_s": None,
                    "T_selected_action_execution_s": None})
                _budget(budget, f"aged_pass_{unit}_a{age}", ap_t0)
            _budget(budget, f"unit_{unit}", u_t0)

    # ---------------- interleaved cost block (one representative unit) --------
    # Honest reuse accounting: candidate gradient of the selected action is
    # already computed by the scorer -> execution adds only the injection copy.
    c_t0 = time.perf_counter()
    anchor0 = manifest["anchors"][0]
    key0 = f"{anchor0['train_seed']}/{anchor0['state_step']}"
    ctx0 = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == key0)
    stream0 = next(s for s in ctx0.streams if int(s["stream_id"]) == 0)
    unit0 = f"{key0}/s0"
    AGE0 = 2
    pe._restore_and_seed(rt, ctx0, stream0)
    cand0 = pe._candidates(ctx0, stream0, args.M)
    for h in range(1, AGE0 + 1):
        ng, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], h, want_snapshot=False)
        pe._clip_only(rt, ng, None)
        pe._step_once(rt)
    ng, _, snap0 = pe._baseline_no_step(rt, ctx0, stream0["batches"], AGE0 + 1,
                                        want_snapshot=True)
    pe._clip_only(rt, ng, snap0)
    # recompute the unit's cached p_tau honestly: one full anchor pass
    ap0 = aged_decision_pass(pe, rt, ctx0, stream0, age=0, H=args.H, M=args.M,
                             want_stage_times=False)
    p_tau0 = clone_p(ap0["p_t"])
    # restore back to the pending snapshot state for the timing block
    pe._restore_and_seed(rt, ctx0, stream0)
    for h in range(1, AGE0 + 1):
        ng, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], h, want_snapshot=False)
        pe._clip_only(rt, ng, None)
        pe._step_once(rt)
    ng, _, snap0 = pe._baseline_no_step(rt, ctx0, stream0["batches"], AGE0 + 1,
                                        want_snapshot=True)
    snap0["native_gradient"] = ng
    pe._clip_only(rt, ng, snap0)

    def _cand_score_block():
        g_by_j = pe._candidate_grads(rt, cand0)
        a_by_j = {int(j): {name: g_by_j[j][name] - ng[name] for name, _ in rt["named"]}
                  for j in cand0}
        w = apply_Gt_T(named=rt["named"], optimizer=rt["optimizer"], snapshot=snap0,
                       p=p_tau0, grad_clip=grad_clip)["w"]
        return score_actions(w, a_by_j)
    cands = _tick_block(torch, _cand_score_block, repeats=args.timing_repeats)
    gtp = _tick_block(torch, lambda: apply_Gt_T(named=rt["named"], optimizer=rt["optimizer"],
                                                snapshot=snap0, p=p_tau0,
                                                grad_clip=grad_clip),
                      repeats=args.timing_repeats)

    inj = []
    for rep in range(args.timing_repeats):
        pe._restore_and_seed(rt, ctx0, stream0)
        for h in range(1, AGE0 + 1):
            ngx, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], h,
                                             want_snapshot=False)
            pe._clip_only(rt, ngx, None)
            pe._step_once(rt)
        ngx, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], AGE0 + 1,
                                         want_snapshot=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _injected_step_inner(rt, ngx, cand0)
        torch.cuda.synchronize()
        inj.append(time.perf_counter() - t0)

    nat = []
    for rep in range(args.timing_repeats):
        pe._restore_and_seed(rt, ctx0, stream0)
        for h in range(1, AGE0 + 1):
            ngx, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], h,
                                             want_snapshot=False)
            pe._clip_only(rt, ngx, None)
            pe._step_once(rt)
        ngx, _, _ = pe._baseline_no_step(rt, ctx0, stream0["batches"], AGE0 + 1,
                                         want_snapshot=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pe._clip_only(rt, ngx, None)
        pe._step_once(rt)
        torch.cuda.synchronize()
        nat.append(time.perf_counter() - t0)

    interleaved = {
        "unit": unit0, "age": AGE0, "repeats": args.timing_repeats,
        "T_candidate_score_median_s": statistics.median([x[0] for x in cands]),
        "T_GTp_cached_median_s": statistics.median([x[0] for x in gtp]),
        "T_selected_action_execution_median_s": statistics.median(inj),
        "T_native_step_median_s": statistics.median(nat),
        "note": ("T_selected_action_execution honestly re-runs the candidate scorer "
                 "(full-grad scorer already holds the selected gradient in production; "
                 "here the injection copy is measured WITH candidate recomputation as "
                 "the conservative upper bound)"),
    }
    for row in timing_rows:
        if row["unit"] == unit0 and row["age"] == AGE0:
            row["T_native_step_s"] = interleaved["T_native_step_median_s"]
            row["T_selected_action_execution_s"] = interleaved["T_selected_action_execution_median_s"]
    _budget(budget, "interleaved_cost_block", c_t0)

    # ---------------- memory ------------------------------------------------
    mem = {"tag": "A0J_CACHE_MEMORY", "units": {}}
    P_bytes = None
    for unit, d in decisions.items():
        pb = d["bytes"]["p_tau"]
        wb = d["bytes"]["w_tau"]
        if P_bytes is None:
            # p channels are (theta, m, v) over named params -> theta == P
            P_bytes = pb // 3
        mem["units"][unit] = {
            "p_theta_p_m_p_v_bytes": pb, "w_bytes": wb, "P_bytes": P_bytes,
            "p_over_P": pb / P_bytes if P_bytes else None,
            "w_over_P": wb / P_bytes if P_bytes else None}
    (OUT / "cache_memory.json").write_text(json.dumps(mem, indent=1))

    # ---------------- write CSVs --------------------------------------------
    _write_csv("cached_scores.csv", score_rows)
    _write_csv("gap_diagnostics.csv", gap_rows)
    _write_csv("decision_preservation.csv", pres_rows)
    _write_csv("cache_vector_diagnostics.csv", vec_rows)
    _write_csv("action_vjp_gate.csv", gate_rows)
    _write_csv("cache_timing_raw.csv", timing_rows)
    (OUT / "cache_timing_summary.json").write_text(json.dumps({
        "tag": "A0J_CACHE_TIMING", "interleaved_block": interleaved,
        "per_unit_from_passes": timing_rows,
        "definitions": {
            "T_refresh": "ff steps + pending fwd/bwd + rollout + endpoint grad + reverse walk (one-shot, A0I-comparable)",
            "T_GTp_cached": "apply_Gt_T(snapshot_pending, p_tau) alone",
            "T_candidate_score": "candidate gradients + score dots + selector",
            "T_selected_action_execution": "injection copy + clip + step at pending horizon (candidate recomputed = conservative upper bound)",
            "T_native_step": "plain clip + step at pending horizon"}} , indent=1))

    # ---------------- freeze decisions BEFORE replay ------------------------
    frozen_path = OUT / "decisions_frozen.json"
    frozen_path.write_text(json.dumps({
        "tag": "A0J_DECISIONS_FROZEN_BEFORE_REPLAY",
        "frozen_tolerance": {"atol32": ATOL32, "rtol32": RTOL32},
        "decisions": decisions}, indent=1))
    frozen_sha = _sha256_file(frozen_path)
    (OUT / "decisions_frozen.sha256").write_text(frozen_sha + "  decisions_frozen.json\n")

    gate_all = all(bool(r["within_tolerance"]) and bool(r["chosen_id_match"])
                   for r in gate_rows)
    total_s = time.perf_counter() - session_t0
    budget.append({"phase": "TOTAL_SESSION", "seconds": total_s, "kind": "GPU_HELD_WALL"})
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + budget
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries if e["phase"] != "TOTAL_SESSION")},
        indent=1))

    status = {"CACHE_IMPLEMENTATION_GATE": "PASS" if gate_all else "FAIL",
              "n_gate_rows": len(gate_rows),
              "frozen_sha256": frozen_sha,
              "total_session_s": total_s}
    (OUT / "p1_status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))
    return 0 if gate_all else 2


def _injected_step_inner(rt, native_gradient, cand):
    """Injection copy + clip + step (candidate gradient passed in by caller)."""
    g_by_j = pe._candidate_grads(rt, cand)
    j_sel = sorted(int(c) for c in cand)[0]
    for name, p in rt["named"]:
        p.grad.copy_(native_gradient[name] + 0.05 * (g_by_j[j_sel][name] - native_gradient[name]))
    pe._clip_only(rt, native_gradient, None)
    pe._step_once(rt)


def _record_pass(p, unit, age, score_rows, gap_rows, pres_rows, vec_rows,
                 p_tau, p_t, torch):
    h_ex, h_ca, h_st = p["scores"]["exact"], p["scores"]["cache"], p["scores"]["stale"]
    r_ex, r_ca, r_st = p["ranks"]["exact"], p["ranks"]["cache"], p["ranks"]["stale"]
    ids = sorted(h_ex)
    for j in ids:
        score_rows.append({
            "unit": unit, "age": age, "action_j": j,
            "h_exact": h_ex[j], "h_cache": (None if h_ca is None else h_ca[j]),
            "h_stale": (None if h_st is None else h_st[j]),
            "pred_exact": 0.05 * h_ex[j],
            "pred_cache": (None if h_ca is None else 0.05 * h_ca[j]),
            "pred_stale": (None if h_st is None else 0.05 * h_st[j]),
            "rank_exact": r_ex.index(j), "rank_cache": (None if r_ca is None else r_ca.index(j)),
            "rank_stale": (None if r_st is None else r_st.index(j))})
    for row in gap_table(h_ex, h_ca, h_st):
        row2 = {"unit": unit, "age": age}
        row2.update(row)
        gap_rows.append(row2)
    pres_rows.append({
        "unit": unit, "age": age,
        "chosen_exact": p["chosen"]["exact"], "chosen_cache": p["chosen"]["cache"],
        "chosen_stale": p["chosen"]["stale"],
        "rank_agreement_cache": bool(r_ex == r_ca), "rank_agreement_stale": bool(r_ex == r_st),
        "top1_exact": r_ex[0], "top2_exact": r_ex[1],
        "margin_pred_exact": top_margin(h_ex)["margin_pred"],
        "max_abs_score_err_cache": (None if h_ca is None else max_abs_score_diff(h_ex, h_ca)),
        "max_abs_score_err_stale": (None if h_st is None else max_abs_score_diff(h_ex, h_st))})
    vec_rows.append({
        "unit": unit, "age": age,
        "cos_p_t_ptau": cosine_p(p_t, p_tau),
        "cos_w_exact_cache": cosine_maps(p["w_exact"], p["w_cache"]),
        "cos_w_exact_stale": cosine_maps(p["w_exact"], p["w_stale"]),
        "norm_p_t": norm_p(p_t), "norm_p_tau": norm_p(p_tau),
        "norm_w_exact": norm_map(p["w_exact"]),
        "norm_w_cache": norm_map(p["w_cache"]), "norm_w_stale": norm_map(p["w_stale"]),
        "SECONDARY_DIAGNOSTIC_ONLY": True})


def _write_csv(name, rows):
    if not rows:
        return
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
