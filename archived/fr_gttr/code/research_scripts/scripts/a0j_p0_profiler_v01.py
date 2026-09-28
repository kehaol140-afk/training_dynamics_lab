#!/usr/bin/env python3
"""A0J-P0: bounded shared-prefix profiler + production cost accounting.

Attribution: one torch.profiler pass over an exact shared-adjoint decision
(age 0) on one anchor-stream -> chrome trace + op-level summary.
Production numbers: no-profiler CUDA-synchronized stage timing (warmup 1,
>=5 timed repeats, same GPU/state/stream) -> prefix_components.csv +
wallclock_*.json. Computes the pre-registered branch gate
S_upper = T_total_exact / (T_total_exact - T_recomputable_prefix) (both the
contract definition incl. pending fwd/bwd and the marginal closed-loop
definition reported; the GATE uses the conservative loose prefix).

Profiler wall time is NEVER used as production timing (contract p0_profiler_spec).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"

import a0h15_production_entry_v01 as pe  # noqa: E402
import a0g_decision_utility_v01 as a0g  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import aged_decision_pass  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fingerprints() -> dict:
    files = {
        "a0j_module": "research/src/fr/a0j_future_sensitivity_cache_v01.py",
        "a0j_p0_runner": "research/scripts/a0j_p0_profiler_v01.py",
        "production_entry": "research/scripts/a0h15_production_entry_v01.py",
        "a0h1_runner": "research/scripts/a0h_adjoint_dev_check_v01.py",
        "a0h_adjoint": "research/src/fr/a0h_adjoint_v01.py",
        "a0c_module": "research/src/fr/a0c_adamw_response_propagation_v01.py",
        "a0g_script": "research/scripts/a0g_decision_utility_v01.py",
        "a0g_module": "research/src/fr/a0g_decision_utility_v01.py",
    }
    return {k: {"path": v, "sha256": _sha256_file(PROJECT_ROOT / v)} for k, v in files.items()}


def _gpu_occupancy() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def _timeit(torch, fn, *, repeats: int, warmup: int, label: str, records: list):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for rep in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        records.append({"label": label, "repeat": rep, "wall_s": t1 - t0,
                        "cuda_ms": float(s.elapsed_time(e)),
                        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2**20)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--anchor", default="2031/52")
    ap.add_argument("--stream-id", type=int, default=0)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    budget_entries = []
    session_t0 = time.perf_counter()

    rt = pe.build_runtime(args.device)
    torch = rt["torch"]
    preload_s = time.perf_counter() - session_t0
    budget_entries.append({"phase": "preload_runtime", "seconds": preload_s,
                           "kind": "NON_METERED_PRELOAD_GPU_HELD",
                           "note": "model init + dataset + val tensors + contexts on GPU"})

    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == args.anchor)
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == args.stream_id)
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])

    (OUT / "method_fingerprints.json").write_text(json.dumps({
        "tag": "A0J_METHOD_FINGERPRINTS", "recorded_at_first_gpu_run": True,
        "anchor": args.anchor, "stream_id": args.stream_id,
        "H": args.H, "M": args.M, "fingerprints": _fingerprints(),
        "math_sdpa_hvp_surrogate": True,
        "hvp_config": "fp32 (production candidate of the shared adjoint)",
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": args.device, "gpu_occupancy_before": _gpu_occupancy()}},
        indent=1))

    # ------------------------------------------------------------------
    # Phase 1: torch.profiler attribution pass (bounded, one decision pass)
    # ------------------------------------------------------------------
    t_prof0 = time.perf_counter()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    pe._restore_and_seed(rt, ctx, stream)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                           want_stage_times=False)
        torch.cuda.synchronize()
    prof_s = time.perf_counter() - t_prof0
    budget_entries.append({"phase": "p0_profiler_pass", "seconds": prof_s,
                           "kind": "METERED_ATTRIBUTION_ONLY"})
    trace_path = OUT / "profiler_trace" / f"trace_a0j_{args.anchor.replace('/', '_')}_s{args.stream_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    key_avgs = prof.key_averages()
    top_cuda = sorted(key_avgs, key=lambda e: -e.device_time_total)[:40]
    top_cpu = sorted(key_avgs, key=lambda e: -e.cpu_time_total)[:40]
    profiler_summary = {
        "tag": "A0J_P0_PROFILER_ATTRIBUTION",
        "attribution_only_not_production_timing": True,
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "anchor": args.anchor, "stream_id": args.stream_id,
        "wall_s_under_profiler": prof_s,
        "top_by_cuda_time_total_us": [
            {"key": e.key, "count": e.count, "device_time_total_us": e.device_time_total,
             "cpu_time_total_us": e.cpu_time_total} for e in top_cuda],
        "top_by_cpu_time_total_us": [
            {"key": e.key, "count": e.count, "cpu_time_total_us": e.cpu_time_total,
             "device_time_total_us": e.device_time_total} for e in top_cpu],
    }

    # ------------------------------------------------------------------
    # Phase 2: no-profiler production stage timing
    # ------------------------------------------------------------------
    records: list = []
    t_timing0 = time.perf_counter()

    def _restore_probe():
        pe._restore_and_seed(rt, ctx, stream)
    _timeit(torch, _restore_probe, repeats=args.repeats, warmup=0,
            label="A_restore_seed", records=records)

    last_pass = {}
    def _exact_pass():
        last_pass.clear()
        last_pass.update(aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                                            want_stage_times=True))
    _timeit(torch, _exact_pass, repeats=args.repeats, warmup=args.warmup,
            label="full_exact_decision_pass", records=records)

    def _native_step():
        pe._restore_and_seed(rt, ctx, stream)
        native_gradient, _, _ = pe._baseline_no_step(rt, ctx, stream["batches"], 1,
                                                     want_snapshot=False)
        pe._clip_only(rt, native_gradient, None)
        pe._step_once(rt)
    _timeit(torch, _native_step, repeats=args.repeats, warmup=args.warmup,
            label="native_step_reference", records=records)

    def _endpoint_fwd_probe():
        rng = capture_rng_state()
        model = rt["model"]
        model.train()
        with rt["autocast_ctx"]:
            for x, y in rt["val_score_tensors"]:
                model(x, labels=y)
        restore_rng_state(rng)
    _timeit(torch, _endpoint_fwd_probe, repeats=args.repeats, warmup=1,
            label="C_endpoint_Vscore_forward_probe_trainmode", records=records)

    timing_s = time.perf_counter() - t_timing0
    budget_entries.append({"phase": "p0_stage_timing", "seconds": timing_s, "kind": "METERED"})

    # E4 attribution probe: one HVP under bf16 autocast vs fp32 (bounded)
    t_e40 = time.perf_counter()
    pe._restore_and_seed(rt, ctx, stream)
    pe._baseline_no_step(rt, ctx, stream["batches"], 1, want_snapshot=False)
    native_gradient2, _, snap2 = pe._baseline_no_step(rt, ctx, stream["batches"], 2,
                                                      want_snapshot=True)
    pe._clip_only(rt, native_gradient2, snap2)
    from fr.minimind_v03_runtime import raw_model
    tangent = snap2["clipped_gradient"]
    e4 = {}
    for name, ctx_cls in (("bf16_autocast", rt["autocast_ctx"]), ("fp32_no_autocast", nullcontext())):
        ts = []
        for _ in range(3):
            rng_now = capture_rng_state()
            restore_rng_state(snap2["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(snap2["theta_start"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            a0g._hvp(torch, rt["model"], rt["named"], snap2["batch_tensors"], tangent,
                     ctx_cls, accum)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            restore_rng_state(rng_now)
        e4[name] = {"median_s": statistics.median(ts), "min_s": min(ts)}
    budget_entries.append({"phase": "p0_e4_hvp_precision_probe", "seconds": time.perf_counter() - t_e40,
                           "kind": "METERED"})

    # ------------------------------------------------------------------
    # Phase 3: aggregate components + branch gate
    # ------------------------------------------------------------------
    stage_keys = sorted(set(last_pass["times"]) | set(last_pass["walk_times"]))
    # Per-repeat stage times captured by re-running with want_stage_times:
    per_rep_times: list = []
    per_rep_walk: list = []
    def _exact_pass_capture():
        out = aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                                 want_stage_times=True)
        per_rep_times.append(dict(out["times"]))
        per_rep_walk.append(dict(out["walk_times"]))
    t_agg0 = time.perf_counter()
    _timeit(torch, _exact_pass_capture, repeats=args.repeats, warmup=0,
            label="full_exact_decision_pass_perstage", records=records)
    budget_entries.append({"phase": "p0_perstage_aggregation", "seconds": time.perf_counter() - t_agg0,
                           "kind": "METERED"})

    def _med(key, src):
        vals = [d.get(key) for d in src if d.get(key) is not None]
        return (statistics.median(vals) if vals else None)

    times, walk = per_rep_times, per_rep_walk
    a_recs = [r["wall_s"] for r in records if r["label"] == "A_restore_seed"]
    A = statistics.median(a_recs) if a_recs else None
    pending_fwd = _med("pending_fwd_bwd", times)
    pending_clip = _med("pending_clip", times)
    cand = _med("candidate_grads", times)
    rollout = _med("rollout_rest", times)
    endpoint_grad = _med("endpoint_grad", times)
    c_fwd_recs = [r["wall_s"] for r in records
                  if r["label"] == "C_endpoint_Vscore_forward_probe_trainmode"]
    C_fwd = statistics.median(c_fwd_recs) if c_fwd_recs else None
    e1 = sum((_med(k, walk) or 0.0) for k in stage_keys if k.startswith("e1_adj_h"))
    e2_future = sum((_med(k, walk) or 0.0) for k in stage_keys
                    if k.startswith("e2_clipT_h") and not k.endswith("_h1"))
    e3 = sum((_med(k, walk) or 0.0) for k in stage_keys if k.startswith("e3_hvp_h"))
    F = _med("e2_clipT_h1", walk)  # w_0 transpose at the pending step
    gate_t = _med("gate_GtTp_fresh", times)
    score_t = _med("scoring_dots", times)

    B_loose = (pending_fwd or 0.0) + (rollout or 0.0)          # full lookahead incl pending fwd/bwd
    B_strict = (rollout or 0.0)                                # marginal closed-loop skippable
    E = e1 + e2_future + e3
    # Contract-literal gate prefix = B + C + D + E (endpoint forward counted via
    # the C probe; the forward fused inside D is reported separately as
    # attribution and NOT added again -> prefix_gate is conservative but not
    # double-counted; prefix_strict is the executed marginal closed-loop view).
    prefix_gate = B_loose + (C_fwd or 0.0) + (endpoint_grad or 0.0) + E
    prefix_strict = B_strict + (endpoint_grad or 0.0) + E

    T_total = (A or 0.0) + (pending_fwd or 0.0) + (pending_clip or 0.0) + (cand or 0.0) \
        + (rollout or 0.0) + (endpoint_grad or 0.0) + E + (F or 0.0) + (gate_t or 0.0) + (score_t or 0.0)
    native_recs = [r["wall_s"] for r in records if r["label"] == "native_step_reference"]
    T_native = statistics.median(native_recs) if native_recs else None
    full_pass_recs = [r["wall_s"] for r in records if r["label"] == "full_exact_decision_pass"]
    T_total_measured = statistics.median(full_pass_recs) if full_pass_recs else None

    S_upper_gate = T_total_measured / (T_total_measured - prefix_gate) if T_total_measured else None
    S_upper_strict = T_total_measured / (T_total_measured - prefix_strict) if T_total_measured else None
    gate_verdict = "P1_FULL" if (S_upper_gate is not None and S_upper_gate >= 1.25) \
        else "CACHE_ROUTE_LOW_ENGINEERING_LEVERAGE"

    components = [
        {"component": "A_state_snapshot_clone_restore", "median_s": A, "definition": "load_full_training_state + zero_grad + seed_future_stream"},
        {"component": "B_lookahead_pending_fwd_bwd", "median_s": pending_fwd, "definition": "pending horizon fwd/bwd (+snapshot)"},
        {"component": "B_lookahead_rollout_rest", "median_s": rollout, "definition": "native rollout horizons 2..H (fwd/bwd+clip+step, snapshots)"},
        {"component": "B_clip_pending", "median_s": pending_clip, "definition": "clip_grad_norm_ at pending horizon"},
        {"component": "C_endpoint_Vscore_forward_probe", "median_s": C_fwd, "definition": "ATTRIBUTION PROBE only (train-mode forward, no backward); not separable in executed path"},
        {"component": "D_endpoint_Vscore_gradient", "median_s": endpoint_grad, "definition": "executed endpoint gradient on val_score split"},
        {"component": "E1_optimizer_local_analytic_VJP", "median_s": e1, "definition": "sum over walk steps of adamw_tangent_step_adjoint"},
        {"component": "E2_clipping_VJP_future_steps", "median_s": e2_future, "definition": "sum over walk steps h>=2 of clip transpose"},
        {"component": "E3_curvature_HVP", "median_s": e3, "definition": "sum over walk steps h>=2 of math-SDPA HVP (bf16 autocast)"},
        {"component": "E4_precision_backend_conversion_probe", "median_s": None, "definition": f"bf16 HVP median {e4['bf16_autocast']['median_s']:.4f}s vs fp32 {e4['fp32_no_autocast']['median_s']:.4f}s (probe)"},
        {"component": "F_action_step_GtTp_w0", "median_s": F, "definition": "clip transpose at pending step producing w_0"},
        {"component": "G_candidate_gradient_generation", "median_s": cand, "definition": "4 candidate full gradients (RNG-wrapped)"},
        {"component": "H_score_dot_products_selector", "median_s": score_t, "definition": "4 dot products + argmin"},
        {"component": "GATE_GtTp_fresh_selfcheck", "median_s": gate_t, "definition": "apply_Gt_T(p_fresh) self-consistency (gate, not production)"},
        {"component": "T_total_exact_one_shot", "median_s": T_total_measured, "definition": "median whole-pass wall (measured)"},
        {"component": "T_total_exact_one_shot_sum_of_stages", "median_s": T_total, "definition": "sum of stage medians (sanity check)"},
        {"component": "T_native_step", "median_s": T_native, "definition": "one native optimizer step (fwd/bwd+clip+step)"},
        {"component": "T_recomputable_prefix_GATE_BplusCplusDplusE", "median_s": prefix_gate, "definition": "B(lookahead incl pending fwd/bwd) + C(endpoint fwd probe) + D(endpoint grad) + E(reverse adjoint)"},
        {"component": "T_recomputable_prefix_strict_marginal", "median_s": prefix_strict, "definition": "rollout beyond pending + endpoint grad + E (marginal closed-loop view)"},
        {"component": "S_upper_GATE", "median_s": S_upper_gate, "definition": "T_total/(T_total - prefix_gate)"},
        {"component": "S_upper_strict", "median_s": S_upper_strict, "definition": "T_total/(T_total - prefix_strict)"},
    ]
    with open(OUT / "prefix_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "median_s", "definition"])
        w.writeheader()
        for row in components:
            w.writerow(row)

    with open(OUT / "wallclock_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "repeat", "wall_s", "cuda_ms", "peak_mem_mb"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    wallclock_summary = {
        "tag": "A0J_P0_PRODUCTION_WALLCLOCK",
        "timing_method": "CUDA events + perf_counter + synchronize, no profiler, warmup %d, %d repeats, same GPU/state/stream" % (args.warmup, args.repeats),
        "shared_gpu": True, "gpu_occupancy_after": _gpu_occupancy(),
        "medians": {c["component"]: c["median_s"] for c in components},
        "T_total_exact_median_s": T_total_measured,
        "T_native_step_median_s": T_native,
        "T_recomputable_prefix_gate_s": prefix_gate,
        "T_recomputable_prefix_strict_s": prefix_strict,
        "S_upper_gate": S_upper_gate,
        "S_upper_strict": S_upper_strict,
        "branch_gate": {"rule": "S_upper_gate >= 1.25 -> P1_FULL else CACHE_ROUTE_LOW_ENGINEERING_LEVERAGE",
                        "verdict": gate_verdict},
        "native_step_equivalents": {
            "T_total_exact": (T_total_measured / T_native if T_native else None),
            "T_prefix_gate": (prefix_gate / T_native if T_native else None)},
    }
    (OUT / "wallclock_summary.json").write_text(json.dumps(wallclock_summary, indent=1))
    profiler_summary["branch_gate"] = wallclock_summary["branch_gate"]
    profiler_summary["S_upper_gate"] = S_upper_gate
    (OUT / "profiler_summary.json").write_text(json.dumps(profiler_summary, indent=1))

    total_s = time.perf_counter() - session_t0
    budget_entries.append({"phase": "TOTAL_SESSION", "seconds": total_s,
                           "kind": "GPU_HELD_WALL"})
    log_path = OUT / "gpu_budget_log.json"
    if log_path.exists():
        old = json.loads(log_path.read_text())
        entries = old.get("entries", []) + budget_entries
    else:
        entries = budget_entries
    (log_path).write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800,
        "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries if e["phase"] != "TOTAL_SESSION")},
        indent=1))
    print(json.dumps({"gate_verdict": gate_verdict, "S_upper_gate": S_upper_gate,
                      "T_total_exact": T_total_measured, "prefix_gate": prefix_gate,
                      "T_native": T_native, "total_session_s": total_s}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
