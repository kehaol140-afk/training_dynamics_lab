#!/usr/bin/env python3
"""A0K B_wc re-run: the initial B_wc runs wrongly excluded fixed-j1 events
(engineering defect found on inspection). Re-runs arm B_wc per (seed,K) with
the frozen T_C budget from timing_meta.json, then replaces the B_wc rows in
phase_rows.jsonl. Sequential, after replay, to keep wall times clean.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402

GPU = "cuda:1"


def log_phase(name, t0, t1, detail):
    entry = {"phase": name, "t_start": t0, "t_end": t1,
             "wall_s": round(t1 - t0, 3), "crashed": False, "gpu": GPU,
             "detail": detail}
    log = json.loads(BUDGET_LOG.read_text()) if BUDGET_LOG.exists() else {"phases": []}
    log["phases"].append(entry)
    total = sum(p["wall_s"] for p in log["phases"])
    log["TOTALS"] = {"cumulative_wall_s": round(total, 3), "budget_s": 1800,
                     "remaining_s": round(1800 - total, 3),
                     "n_crashes": sum(1 for p in log["phases"] if p["crashed"])}
    BUDGET_LOG.write_text(json.dumps(log, indent=1))
    print(f"[budget] {name}: {entry['wall_s']:.1f}s | cum {total:.1f}/1800s")


def main() -> int:
    manifest = json.loads((OUT / "contract/live_manifest.json").read_text())
    rows = [json.loads(l) for l in (RUNS / "phase_rows.jsonl").read_text().splitlines() if l.strip()]
    pairs = sorted({(r["seed"], r["K"]) for r in rows if r["arm"] == "B_wc"})
    new_rows = []
    for seed, K in pairs:
        tm = json.loads((RUNS / f"K{K}/seed_{seed}/timing_meta.json").read_text())
        T_C = tm["T_C"]
        t0 = time.perf_counter()
        rt = a0k.build_live_runtime(seed, GPU)
        s, steps, evs = a0k.train_arm(seed, "B_wc", K, rt, manifest,
                                      RUNS / f"K{K}/seed_{seed}/B_wc",
                                      wc_budget_s=T_C)
        del rt
        t1 = time.perf_counter()
        log_phase(f"rerun_Bwc_seed_{seed}_K{K}", t0, t1,
                  {"T_C": T_C, "steps": s["steps_completed"]})
        d = RUNS / f"K{K}/seed_{seed}/B_wc"
        (d / "step_log.json").write_text(json.dumps(steps, indent=1))
        if evs:
            with open(d / "events.jsonl", "w") as fh:
                for e in evs:
                    fh.write(json.dumps(e) + "\n")
        new_rows.append({"seed": seed, "K": K, "arm": "B_wc", **s})
    rows = [r for r in rows if r["arm"] != "B_wc"] + new_rows
    rows.sort(key=lambda r: (r["K"], r["seed"], r["arm"]))
    (RUNS / "phase_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"rerun": len(new_rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
