#!/usr/bin/env python3
"""A0Q driver: response-control map over 3 channels x 12 cells.

Per cell (seed x step x stream): one shared analytic pass (h_lr, h_clip,
h_wd) + rollouts: u=0 twice (CRN), then each channel at +/- delta.
Per-channel gate = A0O thresholds on the 12 analytic-vs-FD pairs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0q_response_control_map_v01"
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
DELTA = a0q.DELTA
H = 4
CHANNELS = a0q.CHANNELS
SEEDS = (2030, 2031, 2037)
STEPS = (46, 82)
STREAM_SEEDS = (7400, 7401)
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
    """One (state, stream) cell: analytic map + FD rollouts for all channels."""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, step)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0Q uses frozen streams only")

    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, H * accum)
    assert len(lookahead) == H * accum

    rng_live = capture_rng_state()
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, step, rt["a0q_snap"])
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(rt["a0q_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    t0 = time.perf_counter()
    counters = pe._new_counters()
    res = a0q.run_response_map(rt, ctx, stream, H, True, counters)
    reset()

    runs0 = []
    for _ in range(2):  # CRN: identical u=0 rollouts
        c2 = pe._new_counters()
        runs0.append(a0q.run_channel_rollout(rt, ctx, stream, H, "lr", 0.0, c2))
        reset()
    crn_ok = bool(runs0[0] == runs0[1])
    L0 = runs0[0]

    fd = {}
    for ch in CHANNELS:
        outs = {}
        for u in (DELTA, -DELTA):
            c2 = pe._new_counters()
            outs[u] = a0q.run_channel_rollout(rt, ctx, stream, H, ch, u, c2)
            reset()
        fd[ch] = (outs[DELTA] - outs[-DELTA]) / (2.0 * DELTA)

    rec = {"seed": seed, "step": step, "stream_seed": stream_seed,
           "h_lr": res["h_lr"], "h_clip": res["h_clip"], "h_wd": res["h_wd"],
           "fd_lr": fd["lr"], "fd_clip": fd["clip"], "fd_wd": fd["wd"],
           "L0": L0, "crn_ok": crn_ok, "v_H_norm": res["v_H_norm"],
           "per_step": res["per_step"],
           "cell_time_s": time.perf_counter() - t0}
    return rec


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    cells_path = RUNS / "cells.jsonl"
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
        rt["a0q_snap"] = RUNS / f"_cell_snap_s{seed}_t{step}.pt"
        save_full_training_state(rt["a0q_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"],
                                 metadata={"protocol": "A0Q_CONTROL_MAP",
                                           "train_seed": int(seed), "event_step": int(step)})
        for stream_seed in STREAM_SEEDS:
            t0 = time.perf_counter()
            rec = cell(rt, seed, step, stream_seed)
            rows.append(rec)
            with open(cells_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            nclip = sum(1 for p in rec["per_step"] if p["clip_active"])
            print(f"cell s{seed} t{step} x{stream_seed}: "
                  f"lr={rec['h_lr']:+.3e}/{rec['fd_lr']:+.3e} "
                  f"clip={rec['h_clip']:+.3e}/{rec['fd_clip']:+.3e} "
                  f"wd={rec['h_wd']:+.3e}/{rec['fd_wd']:+.3e} "
                  f"clip_active={nclip}/{H} ({rec['cell_time_s']:.1f}s)", flush=True)
            log_phase(f"cell_s{seed}_t{step}_x{stream_seed}", t0, time.perf_counter(),
                      {"cells_done": len(rows)})
        rt["a0q_snap"].unlink()
        del rt

    # ---- per-channel gate (A0O thresholds, frozen) ----
    def gate_block(hs, fs):
        hs = np.asarray(hs, dtype=np.float64)
        fs = np.asarray(fs, dtype=np.float64)
        pearson = None
        if hs.std() > 0 and fs.std() > 0:
            pearson = float(np.corrcoef(hs, fs)[0, 1])
        return {"n": len(hs), "pearson": pearson,
                "sign_accuracy": float(np.mean(np.sign(hs) == np.sign(fs))),
                "rel_err_median": float(np.median(np.abs(hs - fs) / (np.abs(fs) + 1e-12)))}

    gates = {}
    for ch in CHANNELS:
        g = gate_block([r[f"h_{ch}"] for r in rows], [r[f"fd_{ch}"] for r in rows])
        g["PASS"] = bool(g["pearson"] is not None and abs(g["pearson"]) >= 0.8
                         and g["sign_accuracy"] >= 0.9)
        gates[ch] = g
    (RUNS / "gates.json").write_text(json.dumps(gates, indent=1))
    log_phase("gates", time.perf_counter(), time.perf_counter(), gates,
              crashed=not all(g["PASS"] for g in gates.values()))
    print(json.dumps(gates, indent=1), flush=True)

    # ---- evaluation (pre-registered) ----
    ev = {}
    for ch in CHANNELS:
        hs = np.array([r[f"h_{ch}"] for r in rows], dtype=np.float64)
        signs = np.sign(hs)
        majority = 1.0 if (signs > 0).sum() >= (signs < 0).sum() else -1.0
        per_seed = {}
        for s in SEEDS:
            sub = np.array([r[f"h_{ch}"] for r in rows if r["seed"] == s])
            per_seed[str(s)] = {"median": float(np.median(sub)),
                                "sign": float(np.sign(np.median(sub)))}
        seed_signs = {v["sign"] for v in per_seed.values()}
        ev[ch] = {
            "median_abs_h": float(np.median(np.abs(hs))),
            "sign_stability": float(np.mean(signs == majority)),
            "majority_sign": float(majority),
            "per_seed_median": per_seed,
            "cross_seed_sign_agreement": f"{len(seed_signs)}/3 unique signs",
            "cv_abs_h": float(np.abs(hs).std() / np.abs(hs).mean()),
            "clip_steps_active_total": sum(sum(1 for p in r["per_step"] if p["clip_active"])
                                           for r in rows) if ch == "clip" else None,
        }
    mag_lr = ev["lr"]["median_abs_h"]
    for ch in CHANNELS:
        ev[ch]["magnitude_vs_lr"] = (ev[ch]["median_abs_h"] / mag_lr
                                     if mag_lr > 0 else None)
    (RUNS / "evaluation.json").write_text(json.dumps(ev, indent=1))

    # ---- stable-channel rule (pre-registered) ----
    stable = {ch: gates[ch]["PASS"] for ch in CHANNELS}
    candidates = [ch for ch in CHANNELS if stable[ch]
                  and ev[ch]["sign_stability"] >= 2 / 3
                  and len({v["sign"] for v in ev[ch]["per_seed_median"].values()}) == 1]
    ranking = sorted(candidates,
                     key=lambda ch: ev[ch]["median_abs_h"], reverse=True)
    verdict = {"stable_channels": stable,
               "control_variable_candidates_in_rule_order": ranking,
               "note": "rule from contract: gate PASS AND sign_stability>=2/3 AND "
                       "3/3 seed sign agreement, ranked by median |h|"}
    (RUNS / "verdict.json").write_text(json.dumps(verdict, indent=1))
    log_phase("evaluation", time.perf_counter(), time.perf_counter(), verdict)
    print(json.dumps(verdict, indent=1), flush=True)

    print(json.dumps({"done": True, "n_cells": len(rows),
                      "crn_all": all(r["crn_ok"] for r in rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
