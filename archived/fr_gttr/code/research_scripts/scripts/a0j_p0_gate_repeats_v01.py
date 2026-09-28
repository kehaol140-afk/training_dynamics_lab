#!/usr/bin/env python3
"""A0J-P0 supplement: per-repeat branch-gate timing with full per-repeat dumps.

The original P0 mixed full-pass medians from one contention regime with stage
medians from another (external jobs on the shared GPU inflate bursts 2-30x).
This script re-measures with EVERY repeat self-contained: per repeat r,
interleaved on the same GPU envelope:
  [full exact pass (stage-timed)] then [native step]
and dumps per-repeat stage times to wallclock_per_repeat.jsonl. The gate
statistic is the MEDIAN of per-repeat S_upper values (each ratio internally
consistent), with min/max reported. Same manifest / same algorithm as P0.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"

import a0h15_production_entry_v01 as pe  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import aged_decision_pass  # noqa: E402


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

    session_t0 = time.perf_counter()
    rt = pe.build_runtime(args.device)
    torch = rt["torch"]
    preload_s = time.perf_counter() - session_t0
    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == args.anchor)
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == args.stream_id)

    # warmup (untimed)
    for _ in range(args.warmup):
        aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                           want_stage_times=False)
    torch.cuda.synchronize()

    rows = []
    t_timing0 = time.perf_counter()
    for rep in range(args.repeats):
        # interleaved pair on the same contention envelope
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = aged_decision_pass(pe, rt, ctx, stream, age=0, H=args.H, M=args.M,
                               want_stage_times=True)
        torch.cuda.synchronize()
        pass_wall = time.perf_counter() - t0

        pe._restore_and_seed(rt, ctx, stream)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ng, _, _ = pe._baseline_no_step(rt, ctx, stream["batches"], 1, want_snapshot=False)
        pe._clip_only(rt, ng, None)
        pe._step_once(rt)
        torch.cuda.synchronize()
        native_wall = time.perf_counter() - t1

        t, wt = p["times"], p["walk_times"]
        ff = sum(v for k, v in t.items() if k.startswith("ff_step_"))
        pending_fwd = t.get("pending_fwd_bwd", 0.0)
        rollout = t.get("rollout_rest", 0.0)
        endpoint = t.get("endpoint_grad", 0.0)
        e1 = sum(v for k, v in wt.items() if k.startswith("e1_"))
        e2f = sum(v for k, v in wt.items() if k.startswith("e2_") and not k.endswith("_h1"))
        e3 = sum(v for k, v in wt.items() if k.startswith("e3_"))
        E = e1 + e2f + e3
        B = pending_fwd + rollout
        prefix = B + endpoint + E   # strict executed view (C fused in endpoint)
        T_total = pass_wall
        s_upper = T_total / (T_total - prefix) if T_total > prefix else None
        rows.append({
            "repeat": rep, "T_total_exact_s": T_total, "T_native_step_s": native_wall,
            "A_restore_s": None,
            "B_lookahead_s": B, "pending_fwd_bwd_s": pending_fwd,
            "rollout_rest_s": rollout, "D_endpoint_grad_s": endpoint,
            "E1_s": e1, "E2_future_s": e2f, "E3_hvp_s": e3, "E_reverse_s": E,
            "F_w0_s": wt.get("e2_clipT_h1"), "G_candidates_s": t.get("candidate_grads"),
            "H_scoring_s": t.get("scoring_dots"), "gate_selfcheck_s": t.get("gate_GtTp_fresh"),
            "T_recomputable_prefix_s": prefix,
            "S_upper_repeat": s_upper,
            "native_step_equivalents_total": T_total / native_wall if native_wall else None,
        })
    timing_s = time.perf_counter() - t_timing0

    s_vals = [r["S_upper_repeat"] for r in rows if r["S_upper_repeat"] is not None]
    total_vals = [r["T_total_exact_s"] for r in rows]
    native_vals = [r["T_native_step_s"] for r in rows]
    summary = {
        "tag": "A0J_P0_GATE_PER_REPEAT",
        "anchor": args.anchor, "stream_id": args.stream_id,
        "repeats": args.repeats, "interleaved": "full pass + native step per repeat",
        "shared_gpu_contention_note": ("external jobs on GPU1 inflate individual repeats "
                                       "2-30x; per-repeat S_upper is a ratio within one "
                                       "contention envelope; median is the gate statistic"),
        "S_upper_median": statistics.median(s_vals),
        "S_upper_min": min(s_vals), "S_upper_max": max(s_vals),
        "S_upper_all_repeats": s_vals,
        "T_total_exact_median_s": statistics.median(total_vals),
        "T_total_exact_min_s": min(total_vals), "T_total_exact_max_s": max(total_vals),
        "T_native_step_median_s": statistics.median(native_vals),
        "T_recomputable_prefix_median_s": statistics.median(
            [r["T_recomputable_prefix_s"] for r in rows]),
        "gate": {"rule": "median S_upper >= 1.25 -> P1_FULL",
                 "verdict": "P1_FULL" if statistics.median(s_vals) >= 1.25
                 else "CACHE_ROUTE_LOW_ENGINEERING_LEVERAGE"},
    }
    (OUT / "wallclock_per_repeat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (OUT / "wallclock_gate_summary.json").write_text(json.dumps(summary, indent=1))

    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + [
        {"phase": "preload_runtime_gate_repeats", "seconds": preload_s,
         "kind": "NON_METERED_PRELOAD_GPU_HELD"},
        {"phase": "p0_gate_per_repeat_timing", "seconds": timing_s, "kind": "METERED"}]
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries
                            if e["phase"] != "TOTAL_SESSION")}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
