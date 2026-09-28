#!/usr/bin/env python3
"""A0K offline counterfactual replay (per control event, C arm).

Frozen A0H20 semantics via a0j.replay_five_actions (age=0, H=4): per event
snapshot, branches j=0..4 = restore -> seed -> native horizons 1..4 with
injection at horizon 1 (j>0) -> dual V_score/V_eval endpoint labels.

Gains are vs the no-op branch: gain = L(noop) - L(j); positive = beneficial.
Writes events_replayed.jsonl next to each C run's events.jsonl and one
aggregated counterfactual_replay.csv. GPU time is appended to
gpu_budget_log.json (counts against the 1800s round budget).
"""
from __future__ import annotations

import csv
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
from fr.a0j_future_sensitivity_cache_v01 import replay_five_actions  # noqa: E402

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
    t_all0 = time.perf_counter()

    # discover C-event logs: runs/K<K>/seed_<s>/C/events.jsonl
    event_logs = sorted(RUNS.glob("K*/seed_*/C/events.jsonl"))
    if not event_logs:
        print("no C event logs found; run a0k_run_v01.py first")
        return 1

    rows = []
    rt_by_seed = {}
    for elog in event_logs:
        K = int(elog.parts[-4][1:])
        seed = int(elog.parts[-3].split("_")[1])
        run_dir = elog.parent
        events = [json.loads(l) for l in elog.read_text().splitlines() if l.strip()]
        if seed not in rt_by_seed:
            rt_by_seed[seed] = a0k.build_live_runtime(seed, GPU)
        rt = rt_by_seed[seed]
        import a0h15_production_entry_v01 as pe
        out_lines = []
        for e in events:
            step = int(e["step"])
            mev = manifest["events"][f"{seed}/{step}"]
            snap = run_dir / "events" / e["control_id"] / f"event_{seed}_{step}.pt"
            assert snap.exists(), f"missing snapshot {snap}"
            ctx = a0k.make_event_ctx(rt, mev, seed, step, snap)
            stream = {"stream_id": 0, "stream_seed": int(mev["stream_seed"]),
                      "batches": mev["lookahead_batches"]}
            cand = {int(j): mev["candidate_batches"][str(j)]["batches"]
                    for j in range(1, 5)}
            t_r0 = time.perf_counter()
            labels = replay_five_actions(pe, rt, ctx, stream, age=0, H=4, cand=cand)
            t_r1 = time.perf_counter()
            chosen = int(e["chosen_action_id"])
            lab0 = labels[0]
            rec = {
                "seed": seed, "K": K, "step": step, "control_id": e["control_id"],
                "chosen_action_id": chosen,
                "hhat": e["hhat"], "predicted_delta_loss": e["predicted_delta_loss"],
                "offline_reconstruct_ok": (
                    chosen == min(range(5),
                                  key=lambda j: (float(e["predicted_delta_loss"][str(j)]), j))),
                "replay_time_s": round(t_r1 - t_r0, 3),
                "labels": {str(j): {"v_score": labels[j]["v_score"],
                                    "v_eval": labels[j]["v_eval"]}
                           for j in range(5)},
            }
            for j in (1, 2, 3, 4):
                rec[f"gain_v_score_j{j}"] = lab0["v_score"] - labels[j]["v_score"]
                rec[f"gain_v_eval_j{j}"] = lab0["v_eval"] - labels[j]["v_eval"]
            rec["actual_action_gain_v_score"] = rec[f"gain_v_score_j{chosen}"] if chosen else 0.0
            rec["actual_action_gain_v_eval"] = rec[f"gain_v_eval_j{chosen}"] if chosen else 0.0
            rec["counterfactual_no_op_gain_v_eval"] = 0.0
            rec["counterfactual_fixed_reference_gain_v_eval"] = rec["gain_v_eval_j1"]
            rec["counterfactual_fixed_reference_gain_v_score"] = rec["gain_v_score_j1"]
            rows.append(rec)
            out_lines.append(json.dumps(rec))
        (run_dir / "events_replayed.jsonl").write_text("\n".join(out_lines) + "\n")
        print(f"replayed seed={seed} K={K}: {len(events)} events")

    fields = ["seed", "K", "step", "control_id", "chosen_action_id",
              "offline_reconstruct_ok", "replay_time_s",
              "actual_action_gain_v_score", "actual_action_gain_v_eval",
              "counterfactual_fixed_reference_gain_v_score",
              "counterfactual_fixed_reference_gain_v_eval",
              "gain_v_score_j1", "gain_v_score_j2", "gain_v_score_j3", "gain_v_score_j4",
              "gain_v_eval_j1", "gain_v_eval_j2", "gain_v_eval_j3", "gain_v_eval_j4"]
    with open(RUNS / "counterfactual_replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    t_all1 = time.perf_counter()
    log_phase("counterfactual_replay", t_all0, t_all1,
              {"n_events": len(rows), "seeds": sorted(rt_by_seed)})
    print(json.dumps({"n_events_replayed": len(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
