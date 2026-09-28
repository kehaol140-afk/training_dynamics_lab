#!/usr/bin/env python3
"""A0O driver: Stage-1 numerical validation -> (gate) -> Stage-2 prediction.

Inputs (read-only): v0.7 bank checkpoints (seed_2030/2031/2037 x steps
46/52/64/82), A0K frozen live_manifest events (lookahead streams), frozen
validation split via build_live_runtime.

Stage 1: per (seed, step): H in {1,4}; analytic h_eta (a0o module) and
         FD (L(+delta)-L(-delta))/(2 delta), delta=0.05 frozen; plus u=0
         branch for finite-diff curvature sanity and CRN check.
Gate: pooled Pearson |r| >= 0.8 AND sign accuracy >= 0.9 -> Stage 2.
Stage 2: states (seed2030, steps 46/64/82) x 8 streams; real H=4 losses
         L(-delta), L(0), L(+delta) per stream; direction accuracy of
         argmin h_eta, regret, calibration; null = constant-zero response.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0o_lr_response_v01 as a0o  # noqa: E402

GPU = "cuda:1"
DELTA = a0o.DELTA
STAGE1_STATES = [(2030, s) for s in (46, 52, 64, 82)] + \
                [(2031, s) for s in (46, 52, 64, 82)] + \
                [(2037, s) for s in (46, 52, 64, 82)]
STAGE1_STREAM_SEEDS = [7110, 7111, 7112, 7113]
STAGE2_STATES = [(2030, s) for s in (46, 64, 82)]
STAGE2_STREAM_SEEDS = [7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207]


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
    """Amendment A1 (contract): map A0O bank state -> frozen A0K event whose
    lookahead_batches serve as the replay future stream. 2037 has no A0K events
    -> use seed 2032. Step maps to the largest multiple of 10 <= step."""
    src_seed = seed if seed in (2030, 2031) else 2032
    return f"{src_seed}/{(int(step) // 10) * 10}"


def cell(rt, seed, step, stream_seed, H, stage_dir):
    """One (seed, step, stream, H) cell: analytic h_eta + FD + u0 (x2, CRN).

    Amendment A2: the future stream CONTENT depends on stream_seed —
    batch_seed = derive_stream_seed(stream_seed, 404) (A0K recipe),
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, 8*accum).
    (Pre-A2 the A0K event lookahead was reused for all streams, making the
    4 frozen stream seeds identical replicas; no valid results were produced
    under that bug and none are reused.)"""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state, save_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    ev_dir = RUNS / stage_dir / f"seed_{seed}_step{step}" / f"stream_{stream_seed}"
    ev_dir.mkdir(parents=True, exist_ok=True)
    snap = ev_dir / "snapshot.pt"
    rng_live = capture_rng_state()
    save_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                             metadata={"protocol": "A0O_LR_RESPONSE",
                                       "train_seed": int(seed), "event_step": int(step),
                                       "stream_seed": int(stream_seed)})
    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, step)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0O uses frozen streams only")
    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)   # A0K recipe, frozen
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, 8 * accum)
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, step, snap)
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                                 restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    t0 = time.perf_counter()
    counters = pe._new_counters()
    res = a0o.run_lr_response(rt, ctx, stream, H, True, counters)
    h_eta = res["h_eta"]
    t_ana = time.perf_counter() - t0
    reset()

    losses = {}
    L0_first = None
    t_roll = None
    for u in (DELTA, 0.0, -DELTA, 0.0):     # u0 twice for CRN bitwise check
        c2 = pe._new_counters()
        t1 = time.perf_counter()
        val = a0o.run_lr_rollout(rt, ctx, stream, H, u, c2)
        losses[u] = val
        if u == 0.0:
            if L0_first is None:
                L0_first = val
            else:
                losses["L0_repeat"] = val
        if t_roll is None:
            t_roll = time.perf_counter() - t1
        reset()
    fd = (losses[DELTA] - losses[-DELTA]) / (2.0 * DELTA)
    crn_ok = (losses["L0_repeat"] == L0_first)  # second u0 run bitwise == first
    rec = {"seed": seed, "step": step, "stream_seed": stream_seed, "H": H,
           "h_eta": h_eta, "fd": fd, "L_plus": losses[DELTA], "L_minus": losses[-DELTA],
           "L0": losses[0.0], "analytic_time_s": t_ana, "rollout_time_s": t_roll,
           "v_H_norm": res["v_H_norm"], "crn_ok": bool(crn_ok)}
    return rec


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())

    # ---------------- Stage 1 ----------------
    t_stage = time.perf_counter()
    cells = []
    from fr.full_state import load_full_training_state
    for (seed, step) in STAGE1_STATES:
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        for stream_seed in STAGE1_STREAM_SEEDS:
            for H in (1, 4):
                t0 = time.perf_counter()
                rec = cell(rt, seed, step, stream_seed, H, "stage1")
                rec["cell_time_s"] = time.perf_counter() - t0
                cells.append(rec)
                print(f"  cell s{seed} t{step} x{stream_seed} H{H}: "
                      f"h={rec['h_eta']:+.5e} fd={rec['fd']:+.5e} "
                      f"({rec['cell_time_s']:.1f}s)", flush=True)
        del rt
        log_phase(f"stage1_seed_{seed}", t_stage, time.perf_counter(),
                  {"cells_done": len(cells)})
    with open(RUNS / "stage1_cells.jsonl", "w") as f:
        for r in cells:
            f.write(json.dumps(r) + "\n")

    # gate metrics (pooled across states/streams/H)
    def pooled(H_filter=None):
        sel = [c for c in cells if H_filter is None or c["H"] == H_filter]
        h = np.array([c["h_eta"] for c in sel])
        fd = np.array([c["fd"] for c in sel])
        r = float(np.corrcoef(h, fd)[0, 1])
        sign_acc = float(np.mean(np.sign(h) == np.sign(fd)))
        rel = np.abs(h - fd) / (np.abs(fd) + 1e-12)
        return {"n": len(sel), "pearson": r, "sign_accuracy": sign_acc,
                "rel_err_median": float(np.median(rel)),
                "rel_err_p90": float(np.quantile(rel, 0.9))}

    gate = {"all": pooled(), "H1": pooled(1), "H4": pooled(4)}
    gate["PASS"] = bool(abs(gate["all"]["pearson"]) >= 0.8 and
                        gate["all"]["sign_accuracy"] >= 0.9)
    (RUNS / "stage1_gate.json").write_text(json.dumps(gate, indent=1))
    log_phase("stage1_gate", time.perf_counter(), time.perf_counter(), gate,
              crashed=not gate["PASS"])
    print(json.dumps(gate, indent=1), flush=True)
    if not gate["PASS"]:
        print("STAGE 1 FAILED — LR response not validated. STOP per mandate.")
        return 2

    # ---------------- Stage 2 ----------------
    rows = []
    for (seed, step) in STAGE2_STATES:
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        for stream_seed in STAGE2_STREAM_SEEDS:
            t0 = time.perf_counter()
            rec = cell(rt, seed, step, stream_seed, 4, "stage2")
            rec["cell_time_s"] = time.perf_counter() - t0
            rows.append(rec)
            print(f"  s2 s{seed} t{step} x{stream_seed}: h={rec['h_eta']:+.5e} "
                  f"L-={rec['L_minus']:.6f} L0={rec['L0']:.6f} L+={rec['L_plus']:.6f}",
                  flush=True)
        del rt
        log_phase(f"stage2_seed_{seed}", t_stage, time.perf_counter(),
                  {"cells_done": len(rows)})
    with open(RUNS / "stage2_cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # metrics
    def metrics(key):
        hs = np.array([r["h_eta"] for r in rows])
        lms = np.array([r["L_minus"] for r in rows])
        l0s = np.array([r["L0"] for r in rows])
        lps = np.array([r["L_plus"] for r in rows])
        pred_action = np.where(hs > 0, -1, np.where(hs < 0, 1, 0))  # +1=increase LR
        true_best = np.argmin(np.stack([lms, l0s, lps], axis=1), axis=1) - 1
        # true deltas: L(+d)-L(0) vs L(-d)-L(0): best action sign
        d_plus = lps - l0s
        d_minus = lms - l0s
        true_action = np.where((d_plus < d_minus) & (d_plus < 0), 1,
                       np.where(d_minus < 0, -1, 0))
        dir_acc = float(np.mean(pred_action == true_action))
        # regret of the analytic choice vs optimal realized loss
        chosen_loss = np.where(pred_action == 1, lps,
                       np.where(pred_action == -1, lms, l0s))
        opt_loss = np.minimum(np.minimum(lms, l0s), lps)
        regret = float(np.median(chosen_loss - opt_loss))
        # calibration: predicted dL = h_eta * u vs realized directional derivative
        # along the CHOSEN action u_chosen = -delta*sign(h) (h>0 -> decrease LR,
        # so the realized branch is L(-delta); h<0 -> increase LR, branch L(+delta)).
        u_chosen = np.where(hs > 0, -DELTA, DELTA)
        d_chosen = np.where(hs > 0, d_minus, d_plus)
        slope = np.where(np.abs(hs) > 0, d_chosen / u_chosen, 0.0)
        calib = float(np.corrcoef(hs, slope)[0, 1])
        # null: constant-zero response -> action always 0 (baseline)
        null_chosen = l0s
        null_regret = float(np.median(null_chosen - opt_loss))
        null_dir = float(np.mean(true_action == 0))
        return {"n": len(rows), "direction_accuracy": dir_acc,
                "regret_median": regret, "null_regret_median": null_regret,
                "calibration_pearson": calib, "null_direction_accuracy": null_dir,
                "better_than_null_regret": regret < null_regret}

    m = metrics(None)
    (RUNS / "stage2_metrics.json").write_text(json.dumps(m, indent=1))
    log_phase("stage2_metrics", time.perf_counter(), time.perf_counter(), m)
    print(json.dumps(m, indent=1), flush=True)

    print(json.dumps({"done": True, "stage1_pass": True,
                      "n_stage1_cells": len(cells), "n_stage2_cells": len(rows)},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
