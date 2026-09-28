#!/usr/bin/env python3
"""A0N driver: identity smokes + pilot runs under the 1800s GPU budget.

Pilot scope (frozen): seeds 2030/2031/2032, K=10 cadence, arms A/B/C +
A_wc/B_wc. Arms A/B/A_wc/B_wc run via the untouched A0K train_arm; arm C runs
the frozen phase-aware policy (a0n_phase_scheduler_runner_v01).

Gates:
  S1        arm A(2031) model_hash bitwise == v0.7 bank at steps 46/52
  S2        ghost-C(2030) (full phase machinery, forced action=0) tail == arm A(2030) tail
  POST_MAIN A/B final_v_eval bitwise == frozen A0K K10 summaries (all 3 seeds)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0n_phase_scheduler_pilot_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0n_phase_scheduler_runner_v01 as a0n  # noqa: E402

GPU = "cuda:1"
K = 10
SEEDS = [2030, 2031, 2032]


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
    print(f"[budget] {name}: {entry['wall_s']:.1f}s | cum {total:.1f}/1800s", flush=True)
    return total


def persist_run(seed, arm, summary, steps, events):
    d = RUNS / f"seed_{seed}" / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    (d / "step_log.json").write_text(json.dumps(steps, indent=1))
    if events:
        with open(d / "events.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")


def run_a0k_arm(seed, arm, manifest, wc_budget_s=None):
    rt = a0k.build_live_runtime(seed, GPU)
    s, st, ev = a0k.train_arm(seed, arm, K, rt, manifest,
                              RUNS / f"seed_{seed}/{arm}", wc_budget_s=wc_budget_s)
    del rt
    persist_run(seed, arm, s, st, ev)
    return s


def main() -> int:
    manifest = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ---- S1: native path identity vs v0.7 bank ----
    t0 = time.perf_counter()
    sA31 = run_a0k_arm(2031, "A", manifest)
    bank = json.loads((PROJECT_ROOT / "outputs/e3_confirmatory/v07/seed_2031/state_bank_report.json").read_text())
    bank_hashes = {int(s["step"]): s["model_hash"] for s in bank["states"]}
    live = {int(r["step"]): r["model_hash"] for r in json.loads(
        (RUNS / "seed_2031/A/step_log.json").read_text()) if r.get("model_hash")}
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

    # ---- S2: ghost-C(2030) trajectory invisibility ----
    t0 = time.perf_counter()
    sA30 = run_a0k_arm(2030, "A", manifest)
    rt = a0k.build_live_runtime(2030, GPU)
    sG, stG, evG = a0n.train_arm_C_phase(2030, K, rt, manifest,
                                         RUNS / "smoke_S2/ghost_C_2030",
                                         ghost_noop=True)
    del rt
    (RUNS / "smoke_S2").mkdir(parents=True, exist_ok=True)
    (RUNS / "smoke_S2/ghost_summary.json").write_text(json.dumps(sG, indent=1))
    s2 = {"ok": sG["trajectory_tail"] == sA30["trajectory_tail"],
          "ghost_tail": sG["trajectory_tail"], "armA_tail": sA30["trajectory_tail"],
          "ghost_events": len(evG)}
    (RUNS / "smoke_S2/result.json").write_text(json.dumps(s2, indent=1))
    t1 = time.perf_counter()
    log_phase("smoke_S2_ghost_phase_invisibility", t0, t1, s2, crashed=not s2["ok"])
    if not s2["ok"]:
        print("S2 FAILED — phase machinery not invisible under forced no-op. STOP.")
        return 2

    # ---- main: per seed C first (T_C), then remaining arms ----
    done = {(2031, "A"), (2030, "A")}
    summaries = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        rt = a0k.build_live_runtime(seed, GPU)
        sC, stC, evC = a0n.train_arm_C_phase(seed, K, rt, manifest,
                                             RUNS / f"seed_{seed}/C")
        del rt
        persist_run(seed, "C", sC, stC, evC)
        T_C = sC["wall_s"]
        timing = {"seed": seed, "T_C": T_C}

        for arm, budget in (("A", None), ("B", None), ("A_wc", T_C), ("B_wc", T_C)):
            if (seed, arm) in done:
                s = json.loads((RUNS / f"seed_{seed}/{arm}/summary.json").read_text())
            else:
                s = run_a0k_arm(seed, arm, manifest, wc_budget_s=budget)
            summaries.append(s)
        timing["T_A"] = json.loads(
            (RUNS / f"seed_{seed}/A/summary.json").read_text())["wall_s"]
        timing["T_extra"] = T_C - timing["T_A"]
        (RUNS / f"seed_{seed}/timing_meta.json").write_text(json.dumps(timing, indent=1))
        t1 = time.perf_counter()
        log_phase(f"main_seed_{seed}", t0, t1, {"T_C": T_C})

    (RUNS / "all_arm_summaries.json").write_text(json.dumps(summaries, indent=1))

    # ---- POST_MAIN: bitwise consistency vs frozen A0K K10 A/B ----
    frozen = { (r["seed"], r["arm"]): r["final_v_eval"]
               for r in json.loads((A0K_OUT / "runs/all_arm_summaries.json").read_text())
               if r.get("K") == 10 and r["arm"] in ("A", "B") }
    live = {}
    for seed in SEEDS:
        for arm in ("A", "B"):
            live[(seed, arm)] = json.loads(
                (RUNS / f"seed_{seed}/{arm}/summary.json").read_text())["final_v_eval"]
    mism = {f"{k[0]}/{k[1]}": (live[k], frozen.get(k)) for k in live
            if frozen.get(k) != live[k]}
    gate = {"ok": not mism, "mismatches": mism}
    (RUNS / "post_main_consistency.json").write_text(json.dumps(gate, indent=1))
    log_phase("post_main_consistency", time.perf_counter(), time.perf_counter(),
              gate, crashed=bool(mism))
    if mism:
        print("POST-MAIN consistency FAILED vs frozen A0K K10. STOP.")
        return 3

    print(json.dumps({"done": True, "n_new_arm_runs": len(summaries)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
