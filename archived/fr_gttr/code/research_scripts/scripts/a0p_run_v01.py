#!/usr/bin/env python3
"""A0P driver: impulse-kernel Stage-1 validation -> (gate) -> Stage-2 temporal law.

Inputs (read-only): v0.7 bank checkpoints (seed 2030/2031/2037 x steps 46/82),
A0K frozen live_manifest (stream provenance gate only), frozen validation
split via build_live_runtime.

Stage 1: per cell (6 states x 2 streams): analytic R(tau) (impulse adjoint)
and FD R_fd(tau) = (L(+d)-L(-d))/(2d) at every tau in the frozen grid
{1,2,4,8,16,32,64}; u=0 branch twice for CRN.
Gate (A0O thresholds, frozen): pooled |Pearson| >= 0.8 AND sign accuracy >= 0.9
over all 12 cells x 7 taus (n=84) -> Stage 2. FAIL -> STOP.

Stage 2: per-cell kernel fits (exponential / power law, scipy curve_fit,
frozen p0), memory metrics (integrated |R|, interpolated half-life,
sign stability), and pre-registered Q1/Q2/Q3 verdicts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0p_lr_temporal_kernel_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
A0K_OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.a0p_lr_temporal_kernel_v01 as a0p  # noqa: E402

GPU = "cuda:1"
DELTA = a0p.DELTA
GRID = list(a0p.TAU_GRID)
SEEDS = (2030, 2031, 2037)
STEPS = (46, 82)
STREAM_SEEDS = (7300, 7301)
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
    """A0O amendment-A1 pattern (provenance gate only): map the A0P bank state
    to the frozen A0K event key whose existence is required. Content comes from
    the contract recipe (derive_stream_seed(stream_seed, 404) + train pool)."""
    src_seed = seed if seed in (2030, 2031) else 2032
    return f"{src_seed}/{(int(step) // 10) * 10}"


def cell(rt, seed, step, stream_seed):
    """One (state, stream) cell: full kernel, analytic + FD, CRN check."""
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state, save_full_training_state
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed

    man = json.loads((A0K_OUT / "contract/live_manifest.json").read_text())
    key = stream_event_key(seed, step)
    if key not in man["events"]:
        raise SystemExit(f"A0K manifest has no event {key}; A0P uses frozen streams only")

    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_indices = list(map(int, partition["train_indices"]))
    batch_seed = derive_stream_seed(int(stream_seed), 404)  # A0K recipe, frozen
    n_micro = a0p.TAU_MAX * accum  # 64 optimizer steps x 2 micro-batches
    lookahead = batches_from_train_pool(train_indices, bs, batch_seed, n_micro)
    assert len(lookahead) == n_micro

    rng_live = capture_rng_state()
    ctx = a0k.make_event_ctx(rt, {"stream_seed": int(stream_seed),
                                  "lookahead_batches": lookahead,
                                  "lookahead_sha256": None},
                             seed, step, rt["a0p_snap"])
    stream = {"stream_id": 0, "stream_seed": int(stream_seed), "batches": lookahead}

    def reset():
        load_full_training_state(rt["a0p_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        restore_rng_state(rng_live)

    # ---- analytic impulse kernel: one adjoint run per grid tau ----
    R_ana = {}
    for tau in GRID:
        counters = pe._new_counters()
        t0 = time.perf_counter()
        res = a0p.run_lr_impulse_response(rt, ctx, stream, tau, True, counters)
        R_ana[tau] = res["R"]
        reset()
        if tau == a0p.TAU_MAX:
            v_H_norm = res["v_H_norm"]
        print(f"  ana s{seed} t{step} x{stream_seed} tau{tau}: "
              f"R={res['R']:+.5e} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- FD rollouts: +delta, 0, -delta, 0(repeat for CRN) ----
    runs = []
    for u in (DELTA, 0.0, -DELTA, 0.0):
        counters = pe._new_counters()
        out = a0p.run_lr_impulse_rollout(rt, ctx, stream, u, set(GRID), counters)
        runs.append(out)
        reset()
    l_plus, l0_first, l_minus, l0_repeat = runs
    # CRN: second u=0 rollout must reproduce the first bitwise at every grid tau
    crn_ok = bool(all(l0_first[t] == l0_repeat[t] for t in GRID))
    if not crn_ok:
        print(f"  WARN CRN mismatch s{seed} t{step} x{stream_seed}", flush=True)

    rec = {"seed": seed, "step": step, "stream_seed": stream_seed,
           "v_H_norm": v_H_norm, "crn_ok": crn_ok}
    for tau in GRID:
        lp = l_plus[tau]
        l0 = l0_first[tau]
        lm = l_minus[tau]
        rec[f"tau{tau}"] = {"R_analytic": R_ana[tau],
                            "R_fd": (lp - lm) / (2.0 * DELTA),
                            "L_plus": lp, "L0": l0, "L_minus": lm}
    return rec


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    BUDGET_LOG.parent.mkdir(parents=True, exist_ok=True)
    for stale in ("stage1_cells.jsonl", "stage2_kernels.jsonl"):
        p = RUNS / stale
        if p.exists():
            p.unlink()  # fresh run: never mix cells across launches
    from fr.full_state import load_full_training_state, save_full_training_state
    from fr.rng import capture_rng_state

    t_start = time.perf_counter()
    rows = []
    for (seed, step) in STATES:
        rt = a0k.build_live_runtime(seed, GPU)
        bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        rt["model"].train()
        rt["optimizer"].zero_grad(set_to_none=True)
        rt["a0p_snap"] = RUNS / f"_cell_snap_s{seed}_t{step}.pt"
        save_full_training_state(rt["a0p_snap"], rt["model"], rt["optimizer"],
                                 rt["scaler"],
                                 metadata={"protocol": "A0P_LR_TEMPORAL_KERNEL",
                                           "train_seed": int(seed), "event_step": int(step)})
        for stream_seed in STREAM_SEEDS:
            t0 = time.perf_counter()
            rec = cell(rt, seed, step, stream_seed)
            rec["cell_time_s"] = time.perf_counter() - t0
            rows.append(rec)
            with open(RUNS / "stage1_cells.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            r64 = rec[f"tau{a0p.TAU_MAX}"]
            print(f"cell s{seed} t{step} x{stream_seed}: "
                  f"R64={r64['R_analytic']:+.5e} fd={r64['R_fd']:+.5e} "
                  f"({rec['cell_time_s']:.1f}s)", flush=True)
            log_phase(f"cell_s{seed}_t{step}_x{stream_seed}", t0, time.perf_counter(),
                      {"cells_done": len(rows)})
        rt["a0p_snap"].unlink()
        del rt

    # ---------------- Stage 1 gate ----------------
    ha, hf = [], []
    per_tau = {tau: {"h": [], "fd": []} for tau in GRID}
    for rec in rows:
        for tau in GRID:
            ha.append(rec[f"tau{tau}"]["R_analytic"])
            hf.append(rec[f"tau{tau}"]["R_fd"])
            per_tau[tau]["h"].append(rec[f"tau{tau}"]["R_analytic"])
            per_tau[tau]["fd"].append(rec[f"tau{tau}"]["R_fd"])

    def safe_pearson(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def gate_block(a, f):
        a = np.asarray(a, dtype=np.float64)
        f = np.asarray(f, dtype=np.float64)
        return {"n": len(a),
                "pearson": safe_pearson(a, f),
                "sign_accuracy": float(np.mean(np.sign(a) == np.sign(f))),
                "rel_err_median": float(np.median(np.abs(a - f) / (np.abs(f) + 1e-12)))}

    gate = {"all": gate_block(ha, hf)}
    for tau in GRID:
        gate[f"tau{tau}"] = gate_block(per_tau[tau]["h"], per_tau[tau]["fd"])
    gate["PASS"] = bool(gate["all"]["pearson"] is not None and
                        abs(gate["all"]["pearson"]) >= 0.8 and
                        gate["all"]["sign_accuracy"] >= 0.9)
    (RUNS / "stage1_gate.json").write_text(json.dumps(gate, indent=1))
    log_phase("stage1_gate", time.perf_counter(), time.perf_counter(),
              gate, crashed=not gate["PASS"])
    print(json.dumps(gate["all"], indent=1), flush=True)
    if not gate["PASS"]:
        print("STAGE 1 FAILED — impulse kernel not validated. STOP per mandate.")
        return 2

    # ---------------- Stage 2: temporal law ----------------
    from scipy.optimize import curve_fit

    def _r2(R, pred):
        sst = float(np.sum((R - R.mean()) ** 2))
        if sst <= 1e-30:  # flat kernel: no decay to fit -> r2 = 0 by fiat
            return 0.0
        return float(1 - np.sum((R - pred) ** 2) / sst)

    def fit_exp(tau, R):
        f = lambda t, A, tc: A * np.exp(-t / tc)
        p, _ = curve_fit(f, tau, R, p0=[R[0], 8.0], maxfev=20000)
        pred = f(tau, *p)
        return {"A": float(p[0]), "tau_c": float(p[1]),
                "sse": float(np.sum((R - pred) ** 2)), "r2": _r2(R, pred)}

    def fit_pow(tau, R):
        f = lambda t, A, al: A * t ** (-al)
        p, _ = curve_fit(f, tau, R, p0=[R[0], 1.0], maxfev=20000)
        pred = f(tau, *p)
        return {"A": float(p[0]), "alpha": float(p[1]),
                "sse": float(np.sum((R - pred) ** 2)), "r2": _r2(R, pred)}

    def half_life(tau, R):
        thr = abs(R[0]) / 2.0
        if abs(R[0]) == 0.0:
            return None
        for i in range(len(tau)):
            if abs(R[i]) <= thr:
                if i == 0:
                    return float(tau[0])
                t0, t1 = tau[i - 1], tau[i]
                r0, r1 = abs(R[i - 1]), abs(R[i])
                if r0 == r1:
                    return float(t1)
                return float(t0 + (t1 - t0) * (r0 - thr) / (r0 - r1))
        return None  # not_reached on the frozen grid

    tau_arr = np.array(GRID, dtype=np.float64)
    kernels = []
    for rec in rows:
        R = np.array([rec[f"tau{t}"]["R_analytic"] for t in GRID], dtype=np.float64)
        kern = {"seed": rec["seed"], "step": rec["step"],
                "stream_seed": rec["stream_seed"], "R_by_tau": {str(t): float(r) for t, r in zip(GRID, R)}}
        for name, fn in (("exp", fit_exp), ("pow", fit_pow)):
            try:
                kern[name] = {"ok": True, **fn(tau_arr, R)}
            except Exception as e:
                kern[name] = {"ok": False, "error": str(e)[:200]}
        kern["integrated_abs"] = float(np.sum(np.abs(R)))
        kern["half_life"] = half_life(GRID, R)
        kern["sign_stability"] = float(np.mean(np.sign(R) == np.sign(R[0])))
        kern["S4"] = float(R[:3].sum())    # dyadic window {1,2,4}
        kern["S64"] = float(R.sum())       # full frozen grid
        kernels.append(kern)
        with open(RUNS / "stage2_kernels.jsonl", "a") as f:
            f.write(json.dumps(kern) + "\n")

    # ---- Q1/Q2/Q3 (pre-registered rules from contract) ----
    valid_tc = [k["exp"]["tau_c"] for k in kernels if k["exp"]["ok"] and k["exp"]["tau_c"] > 0]
    tc_arr = np.array(valid_tc, dtype=np.float64)
    med_sign = float(np.median([k["sign_stability"] for k in kernels]))
    med_r2 = float(np.median([max(k["exp"]["r2"], k["pow"]["r2"]) for k in kernels
                              if k["exp"]["ok"] and k["pow"]["ok"]]))
    cv_tc = float(tc_arr.std() / tc_arr.mean()) if len(tc_arr) > 1 else None
    q1 = {"median_sign_stability": med_sign, "median_best_r2": med_r2,
          "cv_tau_c": cv_tc, "n_tau_c": len(tc_arr),
          "verdict": bool(med_sign >= 5 / 7 and med_r2 >= 0.8 and cv_tc is not None and cv_tc < 0.5)}

    S4 = np.array([k["S4"] for k in kernels])
    S64 = np.array([k["S64"] for k in kernels])
    s4_s64_pearson = safe_pearson(S4, S64)
    q2 = {"sign_agreement": float(np.mean(np.sign(S4) == np.sign(S64))),
          "pearson": s4_s64_pearson,
          "verdict": bool(np.mean(np.sign(S4) == np.sign(S64)) >= 10 / 12
                          and s4_s64_pearson is not None and s4_s64_pearson > 0)}

    by_seed, by_step = {}, {}
    for k in kernels:
        if k["exp"]["ok"] and k["exp"]["tau_c"] > 0:
            by_seed.setdefault(k["seed"], []).append(k["exp"]["tau_c"])
            by_step.setdefault(k["step"], []).append(k["exp"]["tau_c"])
    shapes = []
    for k in kernels:
        R = np.array([k["R_by_tau"][str(t)] for t in GRID], dtype=np.float64)
        if abs(R[0]) > 0:
            shapes.append(R / R[0])
    pair = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i].std() == 0 or shapes[j].std() == 0:
                continue
            c = float(np.corrcoef(shapes[i], shapes[j])[0, 1])
            if np.isfinite(c):
                pair.append(c)
    pooled_med = float(np.median(tc_arr)) if len(tc_arr) else None
    group_dev = {g: float(np.median(v)) for g, v in by_seed.items()}
    group_dev.update({f"step{g}": float(np.median(v)) for g, v in by_step.items()})
    max_dev = max((abs(m - pooled_med) / pooled_med) for m in group_dev.values()) \
        if pooled_med else None
    q3 = {"cv_tau_c": cv_tc, "tau_c_median": pooled_med,
          "group_medians": group_dev, "max_group_median_deviation": max_dev,
          "normalized_shape_pairwise_pearson_median": float(np.median(pair)) if pair else None,
          "verdict": bool(cv_tc is not None and cv_tc < 0.5
                          and (max_dev is None or max_dev <= 2.0))}

    q = {"q1_stable_timescale": q1, "q2_h4_long_term_direction": q2,
         "q3_shared_timescale": q3, "n_cells": len(kernels)}
    (RUNS / "stage2_q.json").write_text(json.dumps(q, indent=1))
    log_phase("stage2_metrics", time.perf_counter(), time.perf_counter(), q)
    print(json.dumps(q, indent=1), flush=True)

    print(json.dumps({"done": True, "stage1_pass": True, "n_cells": len(rows),
                      "total_s": round(time.perf_counter() - t_start, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
