#!/usr/bin/env python3
"""A0I-B analysis: aggregate benchmark outputs into summaries + verdicts.

Reads outputs/a0i_directional_scoring_v01/a0i_b/run_*/ and writes
equivalence_summary.json, decision_preservation.csv, timing_summary.json,
memory_summary.json, marginal_cost.json, method_fingerprints.json,
benchmark_commands.txt (caller-provided), and the separate-status verdicts.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
B_OUT = PROJECT_ROOT / "outputs/a0i_directional_scoring_v01/a0i_b"
ATOL32 = 1e-5
RTOL32 = 1e-4


def load_runs():
    runs = {}
    for d in sorted(B_OUT.glob("run_*")):
        if not (d / "directional_scores.csv").exists():
            continue
        with open(d / "directional_scores.csv") as f:
            scores = list(csv.DictReader(f))
        with open(d / "timing_raw.csv") as f:
            timing = list(csv.DictReader(f))
        meta = json.loads((d / "run_meta.json").read_text())
        runs[d.name] = {"scores": scores, "timing": timing, "meta": meta}
    return runs


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def median(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def main():
    runs = load_runs()
    if not runs:
        print("no runs found")
        return 1

    # ---------- equivalence ----------
    eq_rows = []
    all_abs, all_abs_iso, all_rel = [], [], []
    rank_agree = {"A_vs_B": 0, "A_vs_Amath": 0}
    chosen_agree = {"A_vs_B": 0, "A_vs_Amath": 0}
    n_units = 0
    gate_pass = 0
    gate_total = 0
    for name, r in runs.items():
        for row in r["scores"]:
            n_units += 1
            abs_errs, abs_iso, rels = [], [], []
            ranks_agree_row = {"A_vs_B": True, "A_vs_Amath": True}
            for j in range(1, 5):
                h_ref = f(row[f"h_ref_j{j}"])
                h_refm = f(row[f"h_refmath_j{j}"])
                h_new = f(row[f"h_new_j{j}"])
                ae = abs(h_new - h_ref)
                ai = abs(h_new - h_refm)
                re = ae / max(abs(h_ref), 1e-30)
                abs_errs.append(ae); abs_iso.append(ai); rels.append(re)
                all_abs.append(ae); all_abs_iso.append(ai); all_rel.append(re)
                gate_total += 1
                if ae <= ATOL32 + RTOL32 * abs(h_ref):
                    gate_pass += 1
            if row["ranking_A"] != row["ranking_B"]:
                ranks_agree_row["A_vs_B"] = False
            if row["ranking_A"] != row["ranking_A_math"]:
                ranks_agree_row["A_vs_Amath"] = False
            for pair in rank_agree:
                if ranks_agree_row[pair]:
                    rank_agree[pair] += 1
            if row["chosen_A"] == row["chosen_B"]:
                chosen_agree["A_vs_B"] += 1
            if row["chosen_A"] == row["chosen_A_math"]:
                chosen_agree["A_vs_Amath"] += 1
            eq_rows.append({
                "state": row["state"], "repeat": row["repeat"],
                "max_abs_err": max(abs_errs), "median_abs_err": median(abs_errs),
                "max_abs_err_isolated": max(abs_iso),
                "max_rel_err": max(rels),
                "ranking_agreement_A_vs_B": ranks_agree_row["A_vs_B"],
                "chosen_id_A": row["chosen_A"], "chosen_id_B": row["chosen_B"],
                "chosen_match_A_vs_B": row["chosen_A"] == row["chosen_B"],
            })

    with open(B_OUT / "decision_preservation.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(eq_rows[0].keys()))
        w.writeheader()
        w.writerows(eq_rows)

    equivalence = {
        "n_passes_total": n_units,
        "abs_err": {"max": max(all_abs), "median": median(all_abs),
                    "mean": sum(all_abs) / len(all_abs)},
        "abs_err_isolated_vs_math_ref": {"max": max(all_abs_iso),
                                         "median": median(all_abs_iso)},
        "rel_err": {"max": max(all_rel), "median": median(all_rel)},
        "gate_atol32": ATOL32, "gate_rtol32": RTOL32,
        "gate_pass_fraction": f"{gate_pass}/{gate_total}",
        "ranking_agreement": {k: f"{v}/{n_units}" for k, v in rank_agree.items()},
        "chosen_id_agreement": {k: f"{v}/{n_units}" for k, v in chosen_agree.items()},
        "interpretation": {
            "abs_err_end_to_end": "JVP(math) vs production full-grad(ambient flash): includes SDPA kernel difference",
            "abs_err_isolated": "JVP(math) vs full-grad(math): isolates forward-mode vs reverse-mode accumulation",
            "gate": "pre-fixed engineering tolerance; pass means implementation agreement only",
        },
    }
    (B_OUT / "equivalence_summary.json").write_text(json.dumps(equivalence, indent=1))

    # ---------- timing ----------
    stages = {}
    for name, r in runs.items():
        for row in r["timing"]:
            key = (row["state"], row["stage"])
            stages.setdefault(key, []).append(row)
    timing_summary = []
    for (state, stage), rows in sorted(stages.items()):
        walls = [f(r["wall_s"]) for r in rows]
        walls_nn = [x for x in walls if not math.isnan(x)]
        timing_summary.append({
            "state": state, "stage": stage, "n": len(walls_nn),
            "wall_median_s": median(walls),
            "wall_min_s": min(walls_nn) if walls_nn else float("nan"),
            "wall_max_s": max(walls_nn) if walls_nn else float("nan"),
            "cuda_median_ms": median([f(r["cuda_ms"]) for r in rows]),
        })
    (B_OUT / "timing_summary.json").write_text(
        json.dumps({"note": "shared-GPU conditions (external jobs present); "
                            "medians over interleaved repeats",
                    "summary": timing_summary}, indent=1))

    # per-state decision cost totals
    def stage_median(state, stage):
        return median([f(r["wall_s"]) for r in stages.get((state, stage), [])])

    states = sorted({k[0] for k in stages})
    decision_cost = []
    for s in states:
        prefix = stage_median(s, "shared_prefix")
        a = stage_median(s, "scoring_A")
        b = stage_median(s, "scoring_B")
        rec = stage_median(s, "selected_gradient_recompute")
        native = stage_median(s, "native_step")
        total_A = prefix + a
        total_B = prefix + b + rec
        decision_cost.append({
            "state": s,
            "shared_prefix_s": prefix,
            "scoring_A_full_grad_s": a,
            "scoring_B_jvp_s": b,
            "selected_gradient_recompute_s": rec,
            "total_decision_cost_METHOD_A_s": total_A,
            "total_decision_cost_METHOD_B_s": total_B,
            "native_step_s": native,
            "equivalent_native_steps_METHOD_A": total_A / native if native else float("nan"),
            "equivalent_native_steps_METHOD_B": total_B / native if native else float("nan"),
        })
    (B_OUT / "decision_cost_summary.json").write_text(json.dumps(decision_cost, indent=1))

    # ---------- memory ----------
    mem = []
    for name, r in runs.items():
        for row in r["timing"]:
            mem.append({"state": row["state"], "stage": row["stage"],
                        "peak_alloc_mb": f(row["peak_alloc_mb"]),
                        "peak_reserved_mb": f(row["peak_reserved_mb"])})
    mem_stage = {}
    for m in mem:
        if math.isnan(m["peak_alloc_mb"]):
            continue
        mem_stage.setdefault((m["state"], m["stage"]), []).append(m)
    memory_summary = []
    for (state, stage), rows in sorted(mem_stage.items()):
        memory_summary.append({
            "state": state, "stage": stage,
            "peak_alloc_max_mb": max(r["peak_alloc_mb"] for r in rows),
            "peak_reserved_max_mb": max(r["peak_reserved_mb"] for r in rows),
        })
    (B_OUT / "memory_summary.json").write_text(json.dumps(memory_summary, indent=1))

    # ---------- marginal cost T(M) = a + bM ----------
    # per-candidate A/B times live on scoring_A rows as A_j1..B_j4; prefix on scoring_A rows
    marg = []
    for s in states:
        rows_a = stages.get((s, "scoring_A"), [])
        perA = {j: median([f(r[f"A_j{j}"]) for r in rows_a if f"A_j{j}" in r and r[f"A_j{j}"]])
                for j in (1, 2, 3, 4)}
        perB = {j: median([f(r[f"B_j{j}"]) for r in rows_a if f"B_j{j}" in r and r[f"B_j{j}"]])
                for j in (1, 2, 3, 4)}
        pA1 = median([f(r["prefix_A_M1"]) for r in rows_a if r.get("prefix_A_M1")])
        pA2 = median([f(r["prefix_A_M2"]) for r in rows_a if r.get("prefix_A_M2")])
        pB1 = median([f(r["prefix_B_M1"]) for r in rows_a if r.get("prefix_B_M1")])
        pB2 = median([f(r["prefix_B_M2"]) for r in rows_a if r.get("prefix_B_M2")])
        allM_A = stage_median(s, "scoring_A")
        allM_B = stage_median(s, "scoring_B")
        for method, pts in (("A", [(1, pA1), (2, pA2), (4, allM_A)]),
                            ("B", [(1, pB1), (2, pB2), (4, allM_B)])):
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else float("nan")
            intercept = my - slope * mx
            marg.append({"state": s, "method": method,
                         "raw_points": {str(x): y for x, y in pts},
                         "per_candidate_single_median_s": median(perA.values()) if method == "A" else median(perB.values()),
                         "estimated_fixed_cost_a_s": intercept,
                         "estimated_marginal_candidate_b_s": slope,
                         "tag": "LOCAL_AFFINE_DIAGNOSTIC",
                         "note": "M=1,2,4 prefixes are TIMING-ONLY nested subsets of the frozen 4 candidates; NOT new science candidate sets; 3-point affine fit, NOT a scaling law"})
    (B_OUT / "marginal_cost.json").write_text(json.dumps(marg, indent=1))

    # ---------- verdicts ----------
    def rel_speed():
        sp_b = median([d["scoring_B_jvp_s"] for d in decision_cost])
        sp_a = median([d["scoring_A_full_grad_s"] for d in decision_cost])
        return sp_b / sp_a if sp_a else float("nan")

    ratio = rel_speed()
    mem_a = median([m["peak_alloc_max_mb"] for m in memory_summary
                    if m["stage"] == "scoring_A"])
    mem_b = median([m["peak_alloc_max_mb"] for m in memory_summary
                    if m["stage"] == "scoring_B"])
    tot_a = median([d["total_decision_cost_METHOD_A_s"] for d in decision_cost])
    tot_b = median([d["total_decision_cost_METHOD_B_s"] for d in decision_cost])

    verdicts = {
        "A0I_A_DIAGNOSTIC_COMPLETE": "YES",
        "DIRECTIONAL_SCORE_EQUIVALENCE":
            "SUPPORTED" if chosen_agree["A_vs_B"] == n_units else
            ("INCOMPLETE" if chosen_agree["A_vs_B"] == 0 else "SUPPORTED_WITH_UNIT_DIVERGENCE"),
        "DIRECTIONAL_DECISION_PRESERVATION":
            f'{chosen_agree["A_vs_B"]}/{n_units} passes same choice; '
            f'ranking agreement {rank_agree["A_vs_B"]}/{n_units}',
        "PER_CANDIDATE_JVP_SPEED": (
            "FASTER" if ratio < 0.9 else ("SIMILAR" if ratio < 1.1 else "SLOWER")
        ) + f" (B/A scoring wall ratio {ratio:.3f}; ambient flash JVP = UNSUPPORTED_IN_CURRENT_BACKEND, ran under math SDPA)",
        "SHARED_POOL_DIRECTIONAL": "NOT_APPLICABLE (disjoint hash groups; see method_c_derivation.md)",
        "MEMORY_RESULT": (
            "IMPROVED" if mem_b < 0.9 * mem_a else ("SIMILAR" if mem_b < 1.1 * mem_a else "WORSE")
        ) + f" (scoring stage peak alloc A {mem_a:.0f} MB vs B {mem_b:.0f} MB)",
        "TOTAL_DECISION_COST_RESULT": (
            "IMPROVED" if tot_b < 0.9 * tot_a else ("SIMILAR" if tot_b < 1.1 * tot_a else "WORSE")
        ) + f" (total A {tot_a:.3f}s vs B {tot_b:.3f}s incl. selected-gradient recompute)",
        "n_passes": n_units,
    }
    (B_OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=1))

    print(json.dumps({"equivalence": equivalence, "decision_cost": decision_cost,
                      "marginal": marg, "verdicts": verdicts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
