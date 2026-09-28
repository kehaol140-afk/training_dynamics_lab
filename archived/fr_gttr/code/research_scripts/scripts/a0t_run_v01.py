#!/usr/bin/env python3
"""A0T driver: response-guided LR control pilot rollouts (arms A/B/C).

Per cell (36 cells: D1 24 + D2 12): read FROZEN chi_lr (A0O stage1 H=4 /
A0Q cells.jsonl), build ONE 128-micro lookahead (nested H windows),
roll arms A (native), B (fixed u=+delta), C (response-guided u=action).
C == B bitwise on cells where action == +delta -> run once, alias.
CRN: arm A twice at H=4. Hard budget stop at 1650s cumulative.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0t_response_lr_control_pilot_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"
A0O_CELLS = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01/runs/stage1_cells.jsonl"
A0Q_CELLS = PROJECT_ROOT / "outputs/a0q_response_control_map_v01/runs/cells.jsonl"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0q_response_control_map_v01 as a0q  # noqa: E402
import fr.a0t_response_lr_control_pilot_v01 as a0t  # noqa: E402

GPU = "cuda:1"
DELTA = a0q.DELTA  # frozen a0o value, import-only
H_GRID = a0t.H_GRID
N_MICRO = max(H_GRID) * 2  # 128 micro-batches (accum=2)

D1 = [("D1", s, t, x) for s in (2030, 2031, 2037) for t in (46, 52, 64, 82)
      for x in (7110, 7111)]
D2 = [("D2", s, t, x) for s in (2030, 2031, 2037) for t in (46, 82)
      for x in (7400, 7401)]
CELLS = D1 + D2


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


def load_frozen_chi():
    chi = {}
    for r in (json.loads(l) for l in A0O_CELLS.read_text().splitlines()):
        if int(r["H"]) == 4:
            chi[("D1", r["seed"], r["step"], r["stream_seed"])] = r["h_eta"]
    for r in (json.loads(l) for l in A0Q_CELLS.read_text().splitlines()):
        chi[("D2", r["seed"], r["step"], r["stream_seed"])] = r["h_lr"]
    return chi


def stream_event_key(seed, step):
    src_seed = seed if seed in (2030, 2031) else 2032
    return f"{src_seed}/{(int(step) // 10) * 10}"


def cell(rt, dataset, seed, step, stream_seed, chi):
    """Rollout losses for arms A/B/C at H in {4,16,64}; C aliases B when equal."""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, step)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0T uses frozen streams only")

    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    assert accum * max(H_GRID) == N_MICRO
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)  # A0K recipe, frozen
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, N_MICRO)
    assert len(lookahead) == N_MICRO

    rng_live = capture_rng_state()
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, step, rt["a0t_snap"])
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(rt["a0t_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    u_c = a0t.action_u(chi)
    aliases = bool(abs(u_c - DELTA) < 1e-15)

    def roll(u, H):
        c2 = pe._new_counters()
        v = a0q.run_channel_rollout(rt, ctx, stream, H, "lr", u, c2)
        reset()
        return v

    t0 = time.perf_counter()
    losses = {"A": {}, "B": {}, "C": {}}
    # CRN: A twice at H=4
    a4 = [roll(0.0, 4), roll(0.0, 4)]
    losses["A"]["4"] = a4[0]
    crn_ok = bool(a4[0] == a4[1])
    for H in (16, 64):
        losses["A"][str(H)] = roll(0.0, H)
    for H in H_GRID:
        losses["B"][str(H)] = roll(DELTA, H)
        losses["C"][str(H)] = (losses["B"][str(H)] if aliases
                               else roll(u_c, H))
    return {"dataset": dataset, "seed": seed, "step": step,
            "stream_seed": stream_seed, "chi_lr": chi, "C_t": abs(chi),
            "action_u": u_c, "aliased": aliases, "crn_ok": crn_ok,
            "losses": losses,
            "cell_time_s": time.perf_counter() - t0}


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    cells_path = RUNS / "cells.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    from fr.full_state import load_full_training_state, save_full_training_state

    chi = load_frozen_chi()
    assert len(chi) == 60, f'frozen chi map has {len(chi)} entries (expect 60)'
    missing = [c for c in CELLS if (c[0], c[1], c[2], c[3]) not in chi]
    assert not missing, f"frozen chi missing for {missing}"

    rows = []
    by_state = {}
    for c in CELLS:
        by_state.setdefault((c[1], c[2]), []).append(c)
    t_start = time.perf_counter()
    partial = False
    for (seed, step), cells in sorted(by_state.items()):
        cum = (json.loads(BUDGET_LOG.read_text())["TOTALS"]["cumulative_wall_s"]
               if BUDGET_LOG.exists() else 0.0)
        if cum > 1650:
            partial = True
            print("BUDGUARD: approaching budget, stopping new states", flush=True)
            break
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        rt["a0t_snap"] = RUNS / f"_cell_snap_s{seed}_t{step}.pt"
        save_full_training_state(rt["a0t_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"],
                                 metadata={"protocol": "A0T_LR_CONTROL_PILOT",
                                           "train_seed": int(seed), "event_step": int(step)})
        for (dataset, s, t, stream_seed) in cells:
            t0 = time.perf_counter()
            rec = cell(rt, dataset, s, t, stream_seed,
                       chi[(dataset, s, t, stream_seed)])
            rows.append(rec)
            with open(cells_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            g4 = rec["losses"]["A"]["4"] - rec["losses"]["C"]["4"]
            print(f"cell {dataset} s{s} t{t} x{stream_seed}: chi={rec['chi_lr']:+.4e} "
                  f"act={'+' if rec['action_u']>0 else '-'} alias={rec['aliased']} "
                  f"gainC_H4={g4:+.2e} crn={rec['crn_ok']} ({rec['cell_time_s']:.1f}s)",
                  flush=True)
            log_phase(f"cell_{dataset}_s{s}_t{t}_x{stream_seed}", t0,
                      time.perf_counter(), {"cells_done": len(rows)})
        rt["a0t_snap"].unlink()
        del rt

    summary = {"done": not partial, "partial": partial, "n_cells": len(rows),
               "crn_all": all(r["crn_ok"] for r in rows),
               "n_aliased": sum(1 for r in rows if r["aliased"]),
               "total_s": round(time.perf_counter() - t_start, 1)}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if not partial else 3


if __name__ == "__main__":
    sys.exit(main())
