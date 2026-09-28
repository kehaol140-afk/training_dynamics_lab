#!/usr/bin/env python3
"""A0L driver: engineering smokes + pilot runs under the 1800s GPU budget.

Pilot scope (frozen): seeds 2030/2031/2032 only; arms A,B,C + A_wc,B_wc per
seed. No expansion, no cadence/alpha/gamma changes, regardless of results.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0l_response_mixture_feasibility_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0l_mixture_runner_v01 as a0l  # noqa: E402

GPU = "cuda:1"


def nvidia_summary():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()
    except Exception as e:  # pragma: no cover
        return [f"nvidia-smi failed: {e}"]


def log_phase(name, t0, t1, detail, crashed=False):
    entry = {"phase": name, "t_start": t0, "t_end": t1,
             "wall_s": round(t1 - t0, 3), "crashed": bool(crashed),
             "gpu": GPU, "detail": detail, "occupancy": nvidia_summary()}
    log = json.loads(BUDGET_LOG.read_text()) if BUDGET_LOG.exists() else {"phases": []}
    log["phases"].append(entry)
    total = sum(p["wall_s"] for p in log["phases"])
    log["TOTALS"] = {"cumulative_wall_s": round(total, 3), "budget_s": 1800,
                     "remaining_s": round(1800 - total, 3),
                     "n_crashes": sum(1 for p in log["phases"] if p["crashed"])}
    BUDGET_LOG.write_text(json.dumps(log, indent=1))
    print(f"[budget] {name}: {entry['wall_s']:.1f}s | cum {total:.1f}/1800s")
    return total


def persist_run(seed, arm, run_dir, summary, steps, events):
    d = RUNS / f"seed_{seed}" / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    (d / "step_log.json").write_text(json.dumps(steps, indent=1))
    if events:
        with open(d / "events.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")


def run_seed(seed, manifest, rows):
    """A, B, C equal-step + A_wc/B_wc at T_C, same session."""
    t0 = time.perf_counter()
    rtC = a0l.build_live_runtime(seed, GPU)
    sC, stC, evC = a0l.train_arm("C", seed, rtC, manifest, RUNS / f"seed_{seed}/C")
    T_C = sC["wall_s"]
    del rtC
    persist_run(seed, "C", RUNS / f"seed_{seed}/C", sC, stC, evC)
    timing = {"seed": seed, "T_C": T_C}

    out = [sC]
    for arm, budget in (("A", None), ("B", None),
                        ("A_wc", T_C), ("B_wc", T_C)):
        rt = a0l.build_live_runtime(seed, GPU)
        s, st, ev = a0l.train_arm(arm, seed, rt, manifest,
                                  RUNS / f"seed_{seed}/{arm}", wc_budget_s=budget)
        del rt
        persist_run(seed, arm, RUNS / f"seed_{seed}/{arm}", s, st, ev)
        out.append(s)
    timing["T_A128"] = next(s["wall_s"] for s in out if s["arm"] == "A")
    timing["T_extra"] = T_C - timing["T_A128"]
    (RUNS / f"seed_{seed}/timing_meta.json").write_text(json.dumps(timing, indent=1))
    t1 = time.perf_counter()
    log_phase(f"main_seed_{seed}", t0, t1,
              {"arms": [s["arm"] for s in out], "T_C": T_C})
    rows.extend(out)


def main() -> int:
    manifest = json.loads((OUT / "contract/a0l_events_manifest.json").read_text())
    seeds = [2030, 2031, 2032]
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # S1: arm A(2031) bitwise == v0.7 bank at 46/52
    t0 = time.perf_counter()
    rt = a0l.build_live_runtime(2031, GPU)
    sA, stA, _ = a0l.train_arm("A", 2031, rt, manifest, RUNS / "smoke_S1/arm_A_2031")
    del rt
    bank = json.loads((PROJECT_ROOT / "outputs/e3_confirmatory/v07/seed_2031/state_bank_report.json").read_text())
    bank_hashes = {int(s["step"]): s["model_hash"] for s in bank["states"]}
    live = {int(r["step"]): r["model_hash"] for r in stA if r.get("model_hash")}
    mism = {s: (live.get(s), bank_hashes.get(s)) for s in (46, 52)
            if live.get(s) != bank_hashes.get(s)}
    s1 = {"ok": not mism, "mismatches": mism}
    (RUNS / "smoke_S1").mkdir(parents=True, exist_ok=True)
    (RUNS / "smoke_S1/result.json").write_text(json.dumps(s1, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S1_armA_bitwise_bank", t0, t1, s1, crashed=bool(mism))
    if mism:
        print("S1 FAILED — native path drift. STOP.")
        return 2

    # S2: arm B(2030) determinism — run twice, identical tail
    t0 = time.perf_counter()
    rt = a0l.build_live_runtime(2030, GPU)
    sB1, _, _ = a0l.train_arm("B", 2030, rt, manifest, RUNS / "smoke_S2/B_run1")
    del rt
    rt = a0l.build_live_runtime(2030, GPU)
    sB2, _, _ = a0l.train_arm("B", 2030, rt, manifest, RUNS / "smoke_S2/B_run2")
    det = sB1["trajectory_tail"] == sB2["trajectory_tail"]
    s2 = {"ok": bool(det), "tail1": sB1["trajectory_tail"], "tail2": sB2["trajectory_tail"],
          "final_match": sB1["final_v_eval"] == sB2["final_v_eval"]}
    (RUNS / "smoke_S2").mkdir(parents=True, exist_ok=True)
    (RUNS / "smoke_S2/result.json").write_text(json.dumps(s2, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S2_B_determinism", t0, t1, s2, crashed=not det)
    if not det:
        print("S2 FAILED — arm B not deterministic. STOP.")
        return 2

    for seed in seeds:
        run_seed(seed, manifest, rows)

    (RUNS / "all_arm_summaries.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({"done": True, "n_arm_runs": len(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
