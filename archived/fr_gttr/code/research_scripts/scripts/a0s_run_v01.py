#!/usr/bin/env python3
"""A0S driver: D2 rollout losses for controllability identification.

Per cell (seed x step x stream, SAME 12 cells and frozen streams as A0Q):
  - baseline u=0 twice (CRN bitwise check)
  - LR +/- delta persistent H=4 rollouts (FROZEN import-only
    a0q.run_channel_rollout; delta imported from frozen a0o via a0q.DELTA)

Analytic chi values are NOT recomputed: they are read from the frozen A0Q
cells.jsonl by the analyzer and joined on (seed, step, stream_seed).
This driver only supplies the rollout losses A0Q did not persist.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0s_controllability_identification_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0q_response_control_map_v01 as a0q  # noqa: E402

GPU = "cuda:1"
DELTA = a0q.DELTA  # frozen a0o value, import-only
H = 4
SEEDS = (2030, 2031, 2037)
STEPS = (46, 82)
STREAM_SEEDS = (7400, 7401)  # frozen A0Q streams
STATES = [(s, t) for s in SEEDS for t in STEPS]


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


def cell(rt, seed, step, stream_seed):
    """Rollout losses only: u=0 x2 (CRN) + lr +/- delta, persistent H=4."""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, step)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0S uses frozen streams only")

    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)  # A0K recipe, frozen
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, H * accum)
    assert len(lookahead) == H * accum

    rng_live = capture_rng_state()
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, step, rt["a0s_snap"])
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(rt["a0s_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    t0 = time.perf_counter()
    outs = {}
    runs0 = []
    for _ in range(2):  # CRN: identical u=0 rollouts
        c2 = pe._new_counters()
        runs0.append(a0q.run_channel_rollout(rt, ctx, stream, H, "lr", 0.0, c2))
        reset()
    outs[0.0] = runs0[0]
    crn_ok = bool(runs0[0] == runs0[1])
    for u in (DELTA, -DELTA):
        c2 = pe._new_counters()
        outs[u] = a0q.run_channel_rollout(rt, ctx, stream, H, "lr", u, c2)
        reset()

    return {"seed": seed, "step": step, "stream_seed": stream_seed,
            "L0": outs[0.0], "L_plus": outs[DELTA], "L_minus": outs[-DELTA],
            "crn_ok": crn_ok,
            "cell_time_s": time.perf_counter() - t0}


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    cells_path = RUNS / "d2_cells.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    from fr.full_state import load_full_training_state, save_full_training_state

    rows = []
    for (seed, step) in STATES:
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        rt["a0s_snap"] = RUNS / f"_cell_snap_s{seed}_t{step}.pt"
        save_full_training_state(rt["a0s_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"],
                                 metadata={"protocol": "A0S_CONTROLLABILITY",
                                           "train_seed": int(seed), "event_step": int(step)})
        for stream_seed in STREAM_SEEDS:
            t0 = time.perf_counter()
            rec = cell(rt, seed, step, stream_seed)
            rows.append(rec)
            with open(cells_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"cell s{seed} t{step} x{stream_seed}: "
                  f"L0={rec['L0']:.6f} L+={rec['L_plus']:.6f} L-={rec['L_minus']:.6f} "
                  f"crn={rec['crn_ok']} ({rec['cell_time_s']:.1f}s)", flush=True)
            log_phase(f"cell_s{seed}_t{step}_x{stream_seed}", t0, time.perf_counter(),
                      {"cells_done": len(rows)})
        rt["a0s_snap"].unlink()
        del rt

    summary = {"done": True, "n_cells": len(rows),
               "crn_all": all(r["crn_ok"] for r in rows)}
    (RUNS / "d2_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
