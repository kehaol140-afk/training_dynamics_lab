#!/usr/bin/env python3
"""FR B0.6 recorder: native-only replay of the frozen B0 cell sample with the
full psi_t channel set.

Semantics (frozen in outputs/fr_b06_regime_dynamics_v01/config.json):
  native LR at every optimizer step (multiplier 1.0; recorder NEVER touches
  param_groups lr), T=64, NO eval calls and NO theta captures (zero RNG
  perturbation relative to the frozen B0 training sequence).
  The step loop mirrors b0_run_v01.run_arm EXACTLY (frozen primitives
  pe._baseline_no_step + pe._clip_and_step, same _TrainLossProbe, same
  stream recipe derive_stream_seed(x,404) + batches_from_train_pool), and adds
  recorder-only channels m_norm / v_norm / g_cos / disp.
  Every shared field is gated BITWISE against frozen B0 diag rows
  (smoke h=1..8 inline; full 26x64 offline via b06_gate_v01.py).

Budget: smoke 16 + main 1,536 + stress 128 = 1,680 optimizer updates;
project base at B0.6 start 15,817/18,000 (B0 14,208 + B0.5 1,609);
guard stops before the cap.

B0.5 lessons applied: (1) all norms/cosines use the named trainable subset
(the raw state_dict carries a 1.64M frozen rotary buffer); (2) CSV writers are
flushed before any reread; (3) gpu_budget_log TOTALS are merged across
processes (each launch adds its own consumption to the carried TOTALS).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b06_regime_dynamics_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"
FROZEN_B0 = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01/runs"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
from b0_run_v01 import (  # noqa: E402  (frozen helpers, reused verbatim)
    _TrainLossProbe, cell_snapshot, make_stream)

T_STEPS = 64
SMOKE_T = 8
DEV_STREAMS = (7500, 7501)
AUDIT_STREAMS = (7502, 7503)
MAIN_SEEDS = (2030, 2031, 2032)
MAIN_STEPS = (46, 82)
STRESS_CELLS = ((2030, 46, 7110), (2030, 64, 7110))
UPDATE_CAP = 18000
CONSUMED_BEFORE_B06 = 15817  # B0 14,208 + B0.5 1,609 (frozen, see config)

BUDGET = {"optimizer_updates": 0, "eval_calls": 0, "theta_bar_evals": 0,
          "theta_captures": 0}

# one id per process: phases of one launch share it, so cross-process TOTALS
# merge can take each launch's final cumulative once (B0.5 lesson 3, corrected
# after the seq-6 merge bug: adding carried TOTALS at EVERY phase of a
# multi-phase launch double-counts that launch's cumulative budget)
LAUNCH_ID = f"{os.getpid()}_{time.time():.0f}"

REC_FIELDS = ["h", "lr_native", "lr_applied", "multiplier", "preclip_norm",
              "clip_active", "grad_norm", "update_norm", "update_cos_prev",
              "m_norm", "v_norm", "g_cos", "disp", "train_loss", "tokens",
              "wall_s"]
PSI_FIELDS = ["tag", "seed", "step", "stream_seed"] + REC_FIELDS

# fields that must match frozen B0 diag BITWISE (offline gate + smoke h1..8)
SHARED_FIELDS = ["grad_norm", "update_norm", "update_cos_prev", "train_loss",
                 "preclip_norm", "lr_native", "lr_applied", "multiplier",
                 "clip_active"]


def pick_gpu() -> str:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    best, best_free = None, -1
    for line in out.stdout.strip().splitlines():
        idx, used, total = [x.strip() for x in line.split(",")]
        free = float(total[:-3]) - float(used[:-3])
        if free > best_free:
            best, best_free = int(idx), free
    dev = f"cuda:{best}"
    print(f"[gpu] selected {dev} (free {best_free:.1f} MiB)", flush=True)
    return dev


def log_phase(name, t0, t1, detail, gpu):
    """Append phase; merge TOTALS across LAUNCHES (B0.5 lesson 3, corrected):
    each launch's contribution is its FINAL cumulative budget (max within the
    launch's phases), summed over launches. Sequential launches by convention;
    a launch crash before its first log_phase is accounted manually."""
    entry = {"phase": name, "launch_id": LAUNCH_ID,
             "t_start": t0, "t_end": t1,
             "wall_s": round(t1 - t0, 3), "gpu": gpu, "detail": detail,
             **{k: v for k, v in BUDGET.items()}}
    log = json.loads(BUDGET_LOG.read_text()) if BUDGET_LOG.exists() \
        else {"phases": []}
    log["phases"].append(entry)
    contrib = {}
    for ph in log["phases"]:
        lid = ph.get("launch_id") or f"legacy_{ph['phase']}"
        cur = contrib.setdefault(lid, {k: 0 for k in BUDGET})
        for k in BUDGET:
            cur[k] = max(cur[k], int(ph.get(k, 0)))
    totals = {k: sum(c[k] for c in contrib.values()) for k in BUDGET}
    log["TOTALS"] = totals
    log["UPDATE_CAP"] = UPDATE_CAP
    log["consumed_before_b06"] = CONSUMED_BEFORE_B06
    log["project_cumulative"] = CONSUMED_BEFORE_B06 + totals["optimizer_updates"]
    BUDGET_LOG.write_text(json.dumps(log, indent=1))


def record_psi_arm(rt, ctx, stream, T):
    """One native T-step rollout with the recorder's extra psi_t channels.

    Step semantics identical to b0_run_v01.run_arm at multiplier 1.0
    (no lr mutation, no eval calls, no theta captures). Extra channels:
      m_norm, v_norm : L2 norm of Adam exp_avg / exp_avg_sq over trainable
                       params (optimizer state has no frozen-buffer entries)
      g_cos          : cos(g_h, g_{h-1}) over the named trainable gradient;
                       0.0 at h=1 by convention (no previous gradient)
      disp           : ||theta_h - theta_0|| over the named trainable subset,
                       theta_0 = trajectory start (h=1 theta_start)
    """
    import a0h15_production_entry_v01 as pe

    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    tau0 = float(cfg["grad_clip"])

    probe = _TrainLossProbe(rt["model"])
    pe._restore_and_seed(rt, ctx, stream)
    rows = []
    prev_update = None
    prev_grad = None
    ref_theta = None
    for h in range(1, int(T) + 1):
        t_h0 = time.perf_counter()
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        BUDGET["optimizer_updates"] += 1
        preclip, _clipped = pe._clip_and_step(rt, native_gradient, snap)

        upd = torch.cat([(p.detach() - snap["theta_start"][name]).reshape(-1).float()
                         for name, p in named])
        upd_norm = float(torch.linalg.vector_norm(upd))
        cos_prev = (float(torch.dot(prev_update, upd)
                          / (torch.linalg.vector_norm(prev_update) * upd_norm))
                    if prev_update is not None else None)
        prev_update = upd
        grad_norm = math.sqrt(sum(
            float((g.float() ** 2).sum()) for g in native_gradient.values()))
        grad_flat = torch.cat([native_gradient[name].reshape(-1).float()
                               for name, _ in named])
        grad_flat_norm = float(torch.linalg.vector_norm(grad_flat))
        g_cos = (float(torch.dot(prev_grad, grad_flat)
                       / (float(torch.linalg.vector_norm(prev_grad)) * grad_flat_norm))
                 if prev_grad is not None else 0.0)
        prev_grad = grad_flat
        # Adam state norms over trainable params only (rotary buffer has no
        # optimizer state entries; B0.5 lesson 1).
        st = rt["optimizer"].state_dict()["state"]
        m_norm = math.sqrt(sum(float((s["exp_avg"].float() ** 2).sum())
                               for s in st.values()))
        v_norm = math.sqrt(sum(float((s["exp_avg_sq"].float() ** 2).sum())
                               for s in st.values()))
        if ref_theta is None:
            ref_theta = {name: snap["theta_start"][name] for name, _ in named}
        disp = float(torch.linalg.vector_norm(torch.cat(
            [(p.detach() - ref_theta[name]).reshape(-1).float()
             for name, p in named])))
        tokens = sum(int(x.numel()) for x, _y in snap["batch_tensors"])
        rows.append({"h": h,
                     "lr_native": float(snap["lr"]),
                     "lr_applied": float(rt["optimizer"].param_groups[0]["lr"]),
                     "multiplier": float(rt["optimizer"].param_groups[0]["lr"]) / float(snap["lr"]),
                     "preclip_norm": float(preclip),
                     "clip_active": bool(preclip > tau0),
                     "grad_norm": grad_norm,
                     "update_norm": upd_norm,
                     "update_cos_prev": cos_prev,
                     "m_norm": m_norm,
                     "v_norm": v_norm,
                     "g_cos": g_cos,
                     "disp": disp,
                     "train_loss": probe.step_loss(accum),
                     "tokens": tokens,
                     "wall_s": time.perf_counter() - t_h0,
                     "_n_state_entries": len(st),
                     "_grad_flat_norm": grad_flat_norm})
    probe.close()
    return rows


def rows_bitwise_equal(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for k, v in ra.items():
            if k.startswith("_") or k == "wall_s":
                continue  # internal/diagnostic fields; wall_s is timing only
            if k not in rb or not _scalar_equal(v, rb[k]):
                return False
    return True


def _scalar_equal(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, bool) or isinstance(y, bool):
        return bool(x) == bool(y)
    return float(x) == float(y)


def load_frozen_b0_diag(path):
    """Frozen B0 diag rows keyed by (seed, step, stream_seed) -> {h: row}."""
    by_traj = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["arm"] != "native" or float(row["u"]) != 0.0:
                continue
            key = (int(row["seed"]), int(row["step"]), int(row["stream_seed"]))
            by_traj.setdefault(key, {})[int(row["h"])] = row
    return by_traj


def check_vs_frozen(rows, frozen_rows, hs):
    """Bitwise comparison of recorder rows against frozen B0 diag rows."""
    detail = {}
    ok_all = True
    for r in rows:
        h = int(r["h"])
        if h not in hs:
            continue
        fr = frozen_rows.get(h)
        if fr is None:
            ok_all = False
            detail[f"h{h}"] = "missing frozen row"
            continue
        for k in SHARED_FIELDS:
            fv = fr[k]
            rv = r[k]
            if rv is None:
                ok = fv == ""
            elif k == "clip_active":
                ok = str(bool(rv)) == fv
            else:
                ok = float(fv) == float(rv)
            if not ok:
                ok_all = False
                detail[f"h{h}.{k}"] = {"recorder": rv, "frozen": fv}
    return ok_all, detail


# ---------------------------------------------------------------------------
# smoke: 16 optimizer updates (2 x 8 steps), 7 checks + budget record
# ---------------------------------------------------------------------------

def smoke(gpu):
    from fr.full_state import load_full_training_state

    checks = {}
    seed, step = MAIN_SEEDS[0], MAIN_STEPS[0]
    rt = a0k.build_live_runtime(seed, gpu)
    bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
    load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    snap = cell_snapshot(rt, seed, step)
    ctx = a0k.make_event_ctx(rt, {"stream_seed": DEV_STREAMS[0],
                                  "lookahead_batches": [], "lookahead_sha256": None},
                             seed, step, snap)

    h_model_before = a0k.model_hash(rt)
    h_opt_before = a0k.opt_hash(rt)
    load_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    checks["snapshot_identity"] = bool(
        a0k.model_hash(rt) == h_model_before and a0k.opt_hash(rt) == h_opt_before)

    stream = make_stream(rt, DEV_STREAMS[0], 2 * T_STEPS)
    rows1 = record_psi_arm(rt, ctx, stream, SMOKE_T)
    rows2 = record_psi_arm(rt, ctx, stream, SMOKE_T)
    checks["crn_native_repeat"] = bool(rows_bitwise_equal(rows1, rows2))

    checks["tokens_per_step_1024"] = bool(
        all(int(r["tokens"]) == 1024 for r in rows1))
    checks["m_v_finite_positive"] = bool(all(
        math.isfinite(r["m_norm"]) and r["m_norm"] > 0.0
        and math.isfinite(r["v_norm"]) and r["v_norm"] > 0.0 for r in rows1))
    checks["cos_bounds"] = bool(all(
        -1.0 - 1e-6 <= r["g_cos"] <= 1.0 + 1e-6
        and (r["update_cos_prev"] is None
             or -1.0 - 1e-6 <= r["update_cos_prev"] <= 1.0 + 1e-6)
        for r in rows1))
    disp_seq = [r["disp"] for r in rows1]
    checks["disp_monotone_nonneg"] = bool(
        all(d >= 0.0 for d in disp_seq)
        and all(disp_seq[i + 1] >= disp_seq[i] for i in range(len(disp_seq) - 1))
        and abs(disp_seq[0] - rows1[0]["update_norm"]) <= 1e-12 * max(1.0, disp_seq[0]))
    checks["state_entries_eq_named"] = bool(
        all(int(r["_n_state_entries"]) == len(rt["named"]) for r in rows1))
    # grad flat-norm over the named subset must reproduce grad_norm up to
    # fp32 summation-order noise (per-tensor sum vs cat-then-norm)
    checks["grad_flat_norm_consistency"] = bool(all(
        abs(r["_grad_flat_norm"] - r["grad_norm"]) <= 1e-6 * max(1.0, r["grad_norm"])
        for r in rows1))

    frozen = load_frozen_b0_diag(FROZEN_B0 / "main_diag.csv")
    key = (seed, step, DEV_STREAMS[0])
    ok, detail = check_vs_frozen(rows1, frozen.get(key, {}), set(range(1, SMOKE_T + 1)))
    checks["bitwise_vs_frozen_b0_diag_h1_8"] = bool(ok)
    if not ok:
        checks["bitwise_detail"] = detail

    checks["budget_after_smoke"] = {"smoke_optimizer_updates":
                                    BUDGET["optimizer_updates"]}
    checks["ALL_PASS"] = bool(all(v for k, v in checks.items() if isinstance(v, bool)))
    (RUNS / "smoke_checks.json").write_text(json.dumps(checks, indent=1))
    log_phase("smoke", time.perf_counter(), time.perf_counter(), checks, gpu)
    print(json.dumps(checks, indent=1), flush=True)
    del rt
    return checks["ALL_PASS"]


# ---------------------------------------------------------------------------
# main + stress
# ---------------------------------------------------------------------------

def run_block(rt, ctx, seed, step, stream_seeds, tag, writer, fdiag):
    for x in stream_seeds:
        stream = make_stream(rt, x, 2 * T_STEPS)
        t0 = time.perf_counter()
        rows = record_psi_arm(rt, ctx, stream, T_STEPS)
        for r in rows:
            row = {k: r[k] for k in REC_FIELDS}
            row.update({"tag": tag, "seed": seed, "step": step, "stream_seed": x})
            writer.writerow(row)
        fdiag.flush()  # B0.5 lesson 2: flush before any reread
        print(f"  [{tag}] s{seed} t{step} x{x}: h1 g={rows[0]['grad_norm']:.6f} "
              f"h64 g={rows[-1]['grad_norm']:.6f} "
              f"({time.perf_counter() - t0:.1f}s) "
              f"budget={BUDGET['optimizer_updates']}", flush=True)
        if CONSUMED_BEFORE_B06 + BUDGET["optimizer_updates"] + T_STEPS > UPDATE_CAP:
            print("BUDGET GUARD: project cap would be exceeded; stopping.",
                  flush=True)
            return False
    return True


def main_cells(gpu):
    from fr.full_state import load_full_training_state
    ok = True
    with open(RUNS / "psi_main.csv", "w", newline="") as fdiag:
        writer = csv.DictWriter(fdiag, fieldnames=PSI_FIELDS)
        writer.writeheader()
        for (seed, step) in [(s, t) for s in MAIN_SEEDS for t in MAIN_STEPS]:
            t0 = time.perf_counter()
            rt = a0k.build_live_runtime(seed, gpu)
            bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
            load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                     rt["scaler"], restore_rng=False)
            rt["model"].train()
            rt["optimizer"].zero_grad(set_to_none=True)
            snap = cell_snapshot(rt, seed, step)
            ctx = a0k.make_event_ctx(rt, {"stream_seed": DEV_STREAMS[0],
                                          "lookahead_batches": [],
                                          "lookahead_sha256": None},
                                     seed, step, snap)
            ok = run_block(rt, ctx, seed, step, DEV_STREAMS + AUDIT_STREAMS,
                           "main", writer, fdiag)
            snap.unlink()
            log_phase(f"main_s{seed}_t{step}", t0, time.perf_counter(),
                      {"optimizer_updates": BUDGET["optimizer_updates"]}, gpu)
            del rt
            if not ok:
                break
    return ok


def stress_cells(gpu):
    from fr.full_state import load_full_training_state
    with open(RUNS / "psi_stress.csv", "w", newline="") as fdiag:
        writer = csv.DictWriter(fdiag, fieldnames=PSI_FIELDS)
        writer.writeheader()
        for (seed, step, x) in STRESS_CELLS:
            rt = a0k.build_live_runtime(seed, gpu)
            bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
            load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                     rt["scaler"], restore_rng=False)
            rt["model"].train()
            rt["optimizer"].zero_grad(set_to_none=True)
            snap = cell_snapshot(rt, seed, step)
            ctx = a0k.make_event_ctx(rt, {"stream_seed": x,
                                          "lookahead_batches": [],
                                          "lookahead_sha256": None},
                                     seed, step, snap)
            run_block(rt, ctx, seed, step, (x,), "stress", writer, fdiag)
            snap.unlink()
            del rt
    log_phase("stress", time.perf_counter(), time.perf_counter(),
              {"optimizer_updates": BUDGET["optimizer_updates"]}, gpu)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main", "stress", "all"],
                    required=True)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    gpu = pick_gpu()

    if args.mode in ("smoke", "all"):
        ok = smoke(gpu)
        if not ok:
            print("SMOKE FAILED — gates not met; stop per protocol.", flush=True)
            return 2
        if args.mode == "smoke":
            return 0
    if args.mode in ("main", "all"):
        ok = main_cells(gpu)
        if not ok:
            print("MAIN incomplete: budget guard triggered.", flush=True)
            return 3
    if args.mode in ("stress", "all"):
        stress_cells(gpu)
    summary = {"done": True, "budget": dict(BUDGET),
               "consumed_before_b06": CONSUMED_BEFORE_B06,
               "project_cumulative": CONSUMED_BEFORE_B06 + BUDGET["optimizer_updates"],
               "budget_frac_of_cap": (CONSUMED_BEFORE_B06 + BUDGET["optimizer_updates"]) / UPDATE_CAP}
    (RUNS / "driver_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
