#!/usr/bin/env python3
"""A0U driver: response-guided schedule discovery pilot rollouts.

12 cells = 3 seeds x 4 streams (A0O stage1 D1 bank), all starting at bank
state step 46. Per cell: frozen schedule built OFFLINE from the frozen
4-knot chi trajectory, then open-loop rollout comparison:
  A native (m=1), B constant +5% (m=1.05), C response schedule (m per segment).
W=64 optimizer steps (bank 47..110), val-loss checkpoints at bank steps
(58, 70, 82, 94, 110). CRN: arm A twice per cell (bitwise all checkpoints).
Hard budget stop at 1650s cumulative.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0u_response_schedule_discovery_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"
A0O_CELLS = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01/runs/stage1_cells.jsonl"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0u_response_schedule_discovery_v01 as a0u  # noqa: E402

GPU = "cuda:0"  # 2026-09-18: GPU 1 occupied by foreign processes (30.6 GiB); GPU 0 free. Pre-results fix, manifest seq 2.
SEEDS = a0u.SEEDS
STREAMS = a0u.STREAMS
W = a0u.W
START = a0u.START_STEP
OFFSET_CPS = tuple(o - START for o in a0u.CHECKPOINTS)  # (12,24,36,48,64)


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


def stream_event_key(seed, step):
    src_seed = seed if seed in (2030, 2031) else 2032
    return f"{src_seed}/{(int(step) // 10) * 10}"


def cell(rt, seed, stream_seed, chi_by_step):
    """One (state, stream) cell: schedule built offline, arms A/A/B/C rolled."""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, START)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0U uses frozen streams only")

    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)  # A0K recipe, frozen
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, W * accum)
    assert len(lookahead) == W * accum

    rng_live = capture_rng_state()
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, START, rt["a0u_snap"])
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(rt["a0u_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    segments = a0u.build_schedule(chi_by_step)
    props = a0u.schedule_properties(segments)

    def roll(mult_fn):
        c2 = pe._new_counters()
        v = a0u.run_schedule_rollout(rt, ctx, stream, W, OFFSET_CPS,
                                     mult_fn, c2)
        reset()
        return v

    t0 = time.perf_counter()
    a1 = roll(lambda h: 1.0)
    a2 = roll(lambda h: 1.0)
    crn_ok = bool(a1 == a2)
    losses = {"A": a1, "B": roll(lambda h: a0u.B_MULT),
              "C": roll(lambda h: a0u.multiplier_at_offset(h, segments))}
    rec = {"seed": seed, "stream_seed": stream_seed,
           "chi": {str(s): chi_by_step[s] for s in a0u.KNOTS},
           "schedule_segments": segments, "schedule_props": props,
           "losses": losses, "crn_ok": crn_ok,
           "cell_time_s": time.perf_counter() - t0}
    return rec


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    cells_path = RUNS / "cells.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    from fr.full_state import load_full_training_state, save_full_training_state

    trajs = a0u.load_trajectories(A0O_CELLS)
    assert len(trajs) == 12, f"expected 12 trajectories, got {len(trajs)}"

    rows = []
    t_start = time.perf_counter()
    for seed in SEEDS:
        cum = (json.loads(BUDGET_LOG.read_text())["TOTALS"]["cumulative_wall_s"]
               if BUDGET_LOG.exists() else 0.0)
        if cum > 1650:
            print("BUDGET GUARD: stopping before new state", flush=True)
            break
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{START:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        rt["a0u_snap"] = RUNS / f"_cell_snap_s{seed}_t{START}.pt"
        save_full_training_state(rt["a0u_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"],
                                 metadata={"protocol": "A0U_SCHEDULE_DISCOVERY",
                                           "train_seed": int(seed),
                                           "event_step": int(START)})
        for stream_seed in STREAMS:
            t0 = time.perf_counter()
            rec = cell(rt, seed, stream_seed, trajs[(seed, stream_seed)])
            rows.append(rec)
            with open(cells_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            cm = a0u.curve_metrics(rec["losses"]["C"])
            am = a0u.curve_metrics(rec["losses"]["A"])
            print(f"cell s{seed} x{stream_seed}: ms={rec['schedule_props']['multipliers']} "
                  f"final A={am['final_loss']:.5f} C={cm['final_loss']:.5f} "
                  f"crn={rec['crn_ok']} ({rec['cell_time_s']:.1f}s)", flush=True)
            log_phase(f"cell_s{seed}_x{stream_seed}", t0, time.perf_counter(),
                      {"cells_done": len(rows)})
        rt["a0u_snap"].unlink()
        del rt

    summary = {"done": len(rows) == 12, "n_cells": len(rows),
               "crn_all": all(r["crn_ok"] for r in rows),
               "total_s": round(time.perf_counter() - t_start, 1)}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))
    log_phase("summary", t_start, time.perf_counter(), summary)
    print(json.dumps(summary, indent=1))
    return 0 if summary["done"] else 3


if __name__ == "__main__":
    sys.exit(main())
