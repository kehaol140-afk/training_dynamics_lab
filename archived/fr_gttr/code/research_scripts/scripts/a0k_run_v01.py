#!/usr/bin/env python3
"""A0K driver: engineering smokes + main runs under the 1800s GPU budget.

Priority (frozen in contract):
  1. complete K=20 protocol (A,B,C equal-step + equal-wall-clock, 3 seeds)
  2. >=3 seeds guaranteed
  3. then K=10 sensitivity (3 seeds)
  4. only if budget remains: 6-seed expansion for K=20

Smokes S1/S2/S3 run BEFORE any scientific run. Budget accounting starts at
first GPU touch and is appended to gpu_budget_log.json after every phase.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
PY = "/home/b205/miniforge3/envs/fr_minimind/bin/python"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402

GPU = "cuda:1"  # A0J used GPU1; GPU0/GPU1 both busy externally, we share quietly


def nvidia_summary():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()
    except Exception as e:  # pragma: no cover
        return [f"nvidia-smi failed: {e}"]


def log_phase(name, t0, t1, detail, crashed=False):
    entry = {
        "phase": name, "t_start": t0, "t_end": t1,
        "wall_s": round(t1 - t0, 3), "crashed": bool(crashed),
        "gpu": GPU, "detail": detail,
        "occupancy": nvidia_summary(),
    }
    log = json.loads(BUDGET_LOG.read_text()) if BUDGET_LOG.exists() else {"phases": []}
    log["phases"].append(entry)
    total = sum(p["wall_s"] for p in log["phases"])
    log["TOTALS"] = {"cumulative_wall_s": round(total, 3),
                     "budget_s": 1800, "remaining_s": round(1800 - total, 3),
                     "n_crashes": sum(1 for p in log["phases"] if p["crashed"])}
    BUDGET_LOG.write_text(json.dumps(log, indent=1))
    print(f"[budget] {name}: {entry['wall_s']:.1f}s | cum {total:.1f}/1800s")
    return total


def run_arm_suite(seed, K, manifest, wc_dir, t_budget_remaining):
    """A, B, C equal-step + A_wc/B_wc at T_C. Returns (rows, wall_used)."""
    phase_rows = []
    t_suite0 = time.perf_counter()
    rtC = a0k.build_live_runtime(seed, GPU)
    sC, stepsC, evC = a0k.train_arm(seed, "C", K, rtC, manifest,
                                    RUNS / f"K{K}/seed_{seed}/C")
    T_C = sC["wall_s"]
    del rtC
    phase_rows.append({"seed": seed, "K": K, "arm": "C", **sC})

    # equal-step A and B in fresh runtimes (identical init discipline)
    rtA = a0k.build_live_runtime(seed, GPU)
    sA, stepsA, _ = a0k.train_arm(seed, "A", K, rtA, manifest,
                                  RUNS / f"K{K}/seed_{seed}/A")
    del rtA
    phase_rows.append({"seed": seed, "K": K, "arm": "A", **sA})

    rtB = a0k.build_live_runtime(seed, GPU)
    sB, stepsB, evB = a0k.train_arm(seed, "B", K, rtB, manifest,
                                    RUNS / f"K{K}/seed_{seed}/B")
    del rtB
    phase_rows.append({"seed": seed, "K": K, "arm": "B", **sB})

    # equal-wall-clock: A and B consume exactly T_C
    rtAw = a0k.build_live_runtime(seed, GPU)
    sAw, stepsAw, _ = a0k.train_arm(seed, "A_wc", K, rtAw, manifest,
                                    RUNS / f"K{K}/seed_{seed}/A_wc",
                                    wc_budget_s=T_C)
    del rtAw
    phase_rows.append({"seed": seed, "K": K, "arm": "A_wc", **sAw})

    rtBw = a0k.build_live_runtime(seed, GPU)
    sBw, stepsBw, _ = a0k.train_arm(seed, "B_wc", K, rtBw, manifest,
                                    RUNS / f"K{K}/seed_{seed}/B_wc",
                                    wc_budget_s=T_C)
    del rtBw
    phase_rows.append({"seed": seed, "K": K, "arm": "B_wc", **sBw})

    # persist per-run artifacts
    base = RUNS / f"K{K}/seed_{seed}"
    for tag, steps, evs in (("C", stepsC, evC), ("A", stepsA, []),
                            ("B", stepsB, evB), ("A_wc", stepsAw, []),
                            ("B_wc", stepsBw, [])):
        d = base / tag
        (d / "step_log.json").write_text(json.dumps(steps, indent=1))
        if evs:
            with open(d / "events.jsonl", "w") as f:
                for e in evs:
                    f.write(json.dumps(e) + "\n")
    meta = {"seed": seed, "K": K, "T_C": T_C,
            "T_A128": sA["wall_s"],
            "T_extra": T_C - sA["wall_s"]}
    (base / "timing_meta.json").write_text(json.dumps(meta, indent=1))
    t_suite1 = time.perf_counter()
    return phase_rows, t_suite1 - t_suite0


def main() -> int:
    manifest = json.loads((OUT / "contract/live_manifest.json").read_text())
    seeds_primary = [2030, 2031, 2032]
    seeds_expansion = [2033, 2034, 2035]

    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ---- engineering smokes (S1, S2, S3) -------------------------------
    t0 = time.perf_counter()
    rt_smoke = a0k.build_live_runtime(2031, GPU)
    s_smoke, steps_smoke, _ = a0k.train_arm(
        2031, "A", 20, rt_smoke, manifest, RUNS / "smoke_S1/arm_A_2031")
    del rt_smoke
    bank = json.loads((PROJECT_ROOT / "outputs/e3_confirmatory/v07/seed_2031/state_bank_report.json").read_text())
    bank_hashes = {int(s["step"]): s["model_hash"] for s in bank["states"]}
    live_hashes = {int(r["step"]): r["model_hash"] for r in steps_smoke
                   if r.get("model_hash")}
    mism = {s: (live_hashes.get(s), bank_hashes.get(s))
            for s in (46, 52) if live_hashes.get(s) != bank_hashes.get(s)}
    s1 = {"ok": not mism, "mismatches": mism,
          "live": {s: live_hashes.get(s) for s in (46, 52)},
          "bank": {s: bank_hashes.get(s) for s in (46, 52)}}
    (RUNS / "smoke_S1/result.json").write_text(json.dumps(s1, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S1_armA_bitwise_bank", t0, t1, s1, crashed=bool(mism))
    if mism:
        print("S1 FAILED — arm A does not bitwise-reproduce the v0.7 bank. STOP.")
        return 2

    t0 = time.perf_counter()
    rt_g = a0k.build_live_runtime(2031, GPU)
    s_g, steps_g, ev_g = a0k.train_arm(
        2031, "C", 20, rt_g, manifest, RUNS / "smoke_S2/ghost_C_2031",
        ghost_noop=True)
    del rt_g
    live_g = {int(r["step"]): r["model_hash"] for r in steps_g if r.get("model_hash")}
    mism_g = {s: (live_g.get(s), bank_hashes.get(s))
              for s in (46, 52) if live_g.get(s) != bank_hashes.get(s)}
    tail_match = (s_g["trajectory_tail"] == s_smoke["trajectory_tail"])
    s2 = {"ok": not mism_g and tail_match, "mismatches": mism_g,
          "tail_match": tail_match,
          "n_events": len(ev_g),
          "event_costs": [e["controller_compute_time"] for e in ev_g]}
    (RUNS / "smoke_S2/result.json").write_text(json.dumps(s2, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S2_ghost_noop_invisible", t0, t1, s2, crashed=bool(mism_g))
    if not s2["ok"]:
        print("S2 FAILED — ghost no-op event is trajectory-visible. STOP.")
        return 2

    # S3: one real control event cost summary from S2 machinery timing
    t0 = time.perf_counter()
    costs = s2["event_costs"]
    s3 = {"n_events": len(costs),
          "mean_controller_compute_time": sum(costs) / max(1, len(costs)),
          "max_controller_compute_time": max(costs) if costs else None}
    s3_dir = RUNS / "smoke_S3"
    s3_dir.mkdir(parents=True, exist_ok=True)
    (s3_dir / "result.json").write_text(json.dumps(s3, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S3_event_cost", t0, t1, s3)

    # ---- main runs ------------------------------------------------------
    all_rows = []
    budget = json.loads(BUDGET_LOG.read_text())["TOTALS"]["cumulative_wall_s"]

    # priority 1+2: K=20 x 3 seeds (A,B,C + A_wc,B_wc)
    for seed in seeds_primary:
        t0 = time.perf_counter()
        rows, used = run_arm_suite(seed, 20, manifest, RUNS, None)
        all_rows.extend(rows)
        t1 = time.perf_counter()
        budget = log_phase(f"main_K20_seed_{seed}", t0, t1,
                           {"arms": [r["arm"] for r in rows]})
        (RUNS / "phase_rows.jsonl").open("a").write(
            "".join(json.dumps(r) + "\n" for r in rows))
        if budget > 1500:
            print("budget guard: <300s left, stopping before K=10")
            break

    # priority 3: K=10 x 3 seeds
    if budget <= 1500:
        for seed in seeds_primary:
            t0 = time.perf_counter()
            rows, used = run_arm_suite(seed, 10, manifest, RUNS, None)
            all_rows.extend(rows)
            t1 = time.perf_counter()
            budget = log_phase(f"main_K10_seed_{seed}", t0, t1,
                               {"arms": [r["arm"] for r in rows]})
            (RUNS / "phase_rows.jsonl").open("a").write(
                "".join(json.dumps(r) + "\n" for r in rows))
            if budget > 1500:
                print("budget guard: <300s left, stopping expansion")
                break

    # priority 4: 6-seed expansion for K=20
    if budget <= 1200:
        for seed in seeds_expansion:
            t0 = time.perf_counter()
            rows, used = run_arm_suite(seed, 20, manifest, RUNS, None)
            all_rows.extend(rows)
            t1 = time.perf_counter()
            budget = log_phase(f"main_K20_seed_{seed}", t0, t1,
                               {"arms": [r["arm"] for r in rows], "expansion": True})
            (RUNS / "phase_rows.jsonl").open("a").write(
                "".join(json.dumps(r) + "\n" for r in rows))
            if budget > 1500:
                break

    (RUNS / "all_arm_summaries.json").write_text(json.dumps(all_rows, indent=1))
    print(json.dumps({"done": True, "n_arm_runs": len(all_rows),
                      "budget": BUDGET_LOG.read_text()[-200:]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
