#!/usr/bin/env python3
"""A0N analyzer: dashboard + mandated metrics + frozen Case A/B/C verdict.

Offline. Reads runs/seed_*/{arm}/summary.json + C events.jsonl +
timing_meta.json + gpu_budget_log.json.
Case rules (frozen in contract.json):
  CASE_A: median(L_B - L_C) > 0 AND >=2/3 seeds positive
  CASE_C: median(L_A - L_C) < 0 AND median(L_B - L_C) < 0
  CASE_B: otherwise
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0n_phase_scheduler_pilot_v01"
RUNS = OUT / "runs"
DASH = OUT / "dashboard"
AN = OUT / "analysis"
REP = OUT / "reports"
HASHES = OUT / "hashes"
ARMS = ["A", "B", "C", "A_wc", "B_wc"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main() -> int:
    for d in (DASH, AN, REP, HASHES):
        d.mkdir(parents=True, exist_ok=True)

    seeds = sorted({int(p.name.split("_")[1]) for p in RUNS.glob("seed_*")})
    S = {}
    for seed in seeds:
        for arm in ARMS:
            p = RUNS / f"seed_{seed}" / arm / "summary.json"
            if p.exists():
                S[(seed, arm)] = json.loads(p.read_text())
        tm = RUNS / f"seed_{seed}/timing_meta.json"
        if tm.exists():
            S[(seed, "timing")] = json.loads(tm.read_text())

    # ---- rho(t) / phase trajectory / action events (mandated saves) ----
    rho_rows = []
    for seed in seeds:
        evp = RUNS / f"seed_{seed}/C/events.jsonl"
        if not evp.exists():
            continue
        for line in evp.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            rho_rows.append({"seed": seed, "step": e["step"], "rho": e["rho"],
                             "phase": e["phase"],
                             "chosen_action": e["chosen_action_id"],
                             "a0k_frozen_choice": e["a0k_frozen_choice"],
                             "rollout_time": e["rollout_time"],
                             "controller_compute_time": e["controller_compute_time"]})
    rho_traj = sorted(rho_rows, key=lambda r: (r["seed"], r["step"]))
    import csv
    with open(AN / "rho_trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rho_traj[0].keys()))
        w.writeheader()
        for r in rho_traj:
            w.writerow(r)

    rows = []
    for seed in seeds:
        if not all((seed, a) in S for a in ARMS):
            continue
        L = {a: S[(seed, a)]["final_v_eval"] for a in ARMS}
        tm = S[(seed, "timing")]
        occ = S[(seed, "C")]["phase_occupancy"]
        ev_rows = [r for r in rho_traj if r["seed"] == seed]
        ctrl_s = sum(r["controller_compute_time"] for r in ev_rows)
        rows.append({
            "seed": seed,
            "L_A": L["A"], "L_B": L["B"], "L_C": L["C"],
            "gain_vs_adamw": L["A"] - L["C"],
            "gain_vs_fixed_j1": L["B"] - L["C"],
            "L_A_wc": L["A_wc"], "gain_time_vs_adamw": L["A_wc"] - L["C"],
            "L_B_wc": L["B_wc"], "gain_time_vs_fixed_j1": L["B_wc"] - L["C"],
            "steps_A_wc": S[(seed, "A_wc")]["steps_completed"],
            "steps_B_wc": S[(seed, "B_wc")]["steps_completed"],
            "T_C": tm["T_C"], "T_A": tm["T_A"], "T_extra": tm["T_extra"],
            "n_events": S[(seed, "C")]["n_control_events"],
            "n_injections": S[(seed, "C")]["n_injections"],
            "occ_high": occ.get("high", 0), "occ_medium": occ.get("medium", 0),
            "occ_low": occ.get("low", 0),
            "controller_overhead_s": ctrl_s,
            "controller_overhead_frac": ctrl_s / tm["T_C"],
        })

    fields = list(rows[0].keys())
    with open(DASH / "dashboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def col(k):
        return [r[k] for r in rows]

    g_a, g_b = med(col("gain_vs_adamw")), med(col("gain_vs_fixed_j1"))
    gt_a, gt_b = med(col("gain_time_vs_adamw")), med(col("gain_time_vs_fixed_j1"))
    n_pos_b = sum(1 for r in rows if r["gain_vs_fixed_j1"] > 0)
    scale_ok = all(r["steps_A_wc"] == 128 and r["steps_B_wc"] == 128 for r in rows)
    occ_med = med(col("occ_medium"))
    inj_med = med(col("n_injections"))
    rho_med = med([r["rho"] for r in rho_traj])

    if g_b is not None and g_b > 0 and n_pos_b >= math.ceil(2 * len(rows) / 3):
        case = "A"
    elif g_a is not None and g_a < 0 and g_b is not None and g_b < 0:
        case = "C"
    else:
        case = "B"
    cases = {
        "A": "C优于B——phase scheduling 有效",
        "B": "C≈B——response 主要发现固定策略",
        "C": "C<A/B——停止 phase 路线",
    }

    budget = json.loads((OUT / "gpu_budget_log.json").read_text())
    metrics = {
        "gain_vs_adamw": {"median": g_a,
                          "per_seed": {str(r["seed"]): r["gain_vs_adamw"] for r in rows}},
        "gain_vs_fixed_j1": {"median": g_b,
                             "per_seed": {str(r["seed"]): r["gain_vs_fixed_j1"] for r in rows}},
        "equal_wall_clock": {"vs_adamw_median": gt_a, "vs_fixed_j1_median": gt_b,
                             "scale_condition": scale_ok},
        "controller_overhead": {"T_extra_median": med(col("T_extra")),
                                "overhead_frac_median": med(col("controller_overhead_frac"))},
        "phase_occupancy_median": {"high": med(col("occ_high")),
                                   "medium": occ_med, "low": med(col("occ_low"))},
        "rho_median": rho_med,
        "n_injections_median": inj_med,
        "case": case, "case_text": cases[case],
        "gpu_budget_totals": budget.get("TOTALS"),
    }
    (DASH / "dashboard.json").write_text(json.dumps({"rows": rows, "metrics": metrics},
                                                    indent=1, default=float))

    fps = {}
    for rel in ["contract/contract.json", "runs/all_arm_summaries.json",
                "dashboard/dashboard.csv", "gpu_budget_log.json",
                "analysis/rho_trajectory.csv"]:
        p = OUT / rel
        if p.exists():
            fps[rel] = sha(p)
    for rel in ["research/src/fr/a0n_phase_scheduler_runner_v01.py",
                "research/scripts/a0n_run_v01.py",
                "research/scripts/a0n_analyze_v01.py"]:
        p = PROJECT_ROOT / rel
        if p.exists():
            fps[rel] = sha(p)
    (HASHES / "method_fingerprints.json").write_text(json.dumps(fps, indent=1))

    def fmt(x, nd=6):
        return "n/a" if x is None else f"{x:.{nd}g}"

    s1 = json.loads((RUNS / "smoke_S1/result.json").read_text())
    s2 = json.loads((RUNS / "smoke_S2/result.json").read_text())
    pm = json.loads((RUNS / "post_main_consistency.json").read_text())

    L = ["# A0N Response Phase-aware Pilot — REPORT\n",
         "dashboard (final v_eval @128, equal steps; gains = loss reduction, positive = better):\n",
         "| seed | L_A | L_B | L_C | gain vs AdamW | gain vs fixed-j1 | gain_time vs AdamW | gain_time vs fixed-j1 | inj | occ H/M/L | rho median |",
         "---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['seed']} | {fmt(r['L_A'])} | {fmt(r['L_B'])} | {fmt(r['L_C'])} "
                 f"| {fmt(r['gain_vs_adamw'])} | {fmt(r['gain_vs_fixed_j1'])} "
                 f"| {fmt(r['gain_time_vs_adamw'])} | {fmt(r['gain_time_vs_fixed_j1'])} "
                 f"| {r['n_injections']}/12 | {r['occ_high']}/{r['occ_medium']}/{r['occ_low']} "
                 f"| {fmt(med([x['rho'] for x in rho_traj if x['seed']==r['seed']]), 4)} |")
    L += ["",
          "## 1. Protocol\n",
          "Frozen by contract.json before any GPU. Phase metric rho = |mean(h_1..4)|/"
          "(std(h_1..4)+1e-12) (population std). Policy: rho>2 -> no-op; 1<=rho<=2 -> inject "
          "argmin hhat_j (g+0.05*(g_j-g), tie->smallest id); rho<1 -> no-op. Response formula, "
          "optimizer, lambda, clipping untouched. K=10 events (steps 10..120) reused from A0K "
          "frozen manifest; seeds 2030-2032; T=128. Arms: A native, B fixed-j1 blind injection "
          "every event, C phase-aware, A_wc/B_wc at T_C.",
          "",
          "## 2. Gates\n",
          f"- S1 arm A(2031) bitwise == v0.7 bank @46/52: **{s1['ok']}**",
          f"- S2 ghost-C(2030) forced no-op == arm A tail: **{s2['ok']}**",
          f"- POST-MAIN A/B final_v_eval bitwise == frozen A0K K10: **{pm['ok']}**",
          "",
          "## 3. Mandated metrics\n",
          f"1. Gain vs AdamW (equal steps): median **{fmt(g_a)}** {json.dumps({k: round(v,6) for k,v in metrics['gain_vs_adamw']['per_seed'].items()})}",
          f"2. Gain vs fixed-j1 (equal steps): median **{fmt(g_b)}** {json.dumps({k: round(v,6) for k,v in metrics['gain_vs_fixed_j1']['per_seed'].items()})} ({n_pos_b}/{len(rows)} seeds positive)",
          f"3. Equal wall-clock: vs AdamW median {fmt(gt_a)}, vs fixed-j1 median {fmt(gt_b)} "
          f"(scale condition {'HOLDS' if scale_ok else 'BROKEN'}: A_wc/B_wc steps "
          f"{[r['steps_A_wc'] for r in rows]}/{[r['steps_B_wc'] for r in rows]})",
          f"4. Controller overhead: T_extra median {fmt(med(col('T_extra')), 4)}s, "
          f"frac of T_C median {fmt(med(col('controller_overhead_frac')), 4)}",
          f"5. Phase occupancy (median events/12): high {fmt(med(col('occ_high')),3)} / "
          f"medium {fmt(occ_med,3)} / low {fmt(med(col('occ_low')),3)}; injections median {fmt(inj_med,3)}/12; "
          f"rho median {fmt(rho_med,4)}",
          "",
          "## 4. rho(t) trajectory",
          "per analysis/rho_trajectory.csv (seed, step, rho, phase, chosen action).",
          "",
          "## 5. Verdict\n",
          f"Frozen rules -> **CASE {case}**: {cases[case]}",
          "",
          "## 6. Budget\n",
          f"gpu_budget_log TOTALS: {json.dumps(budget.get('TOTALS'))}",
          "",
          "## 7. Conclusion\n",
          "Pilot only, per mandate: no optimizer/lambda changes, no expansion, no new rounds. "
          "A0K/A0L/A0M conclusions unchanged.",
          ]
    (REP / "A0N_REPORT.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"case": case, "rows": len(rows), "g_a": g_a, "g_b": g_b,
                      "n_pos_b": n_pos_b, "scale_ok": scale_ok,
                      "occ": metrics["phase_occupancy_median"],
                      "budget": budget.get("TOTALS")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
