#!/usr/bin/env python3
"""A0I-B: scalar directional scoring feasibility on pre-existing dev states.

Question: can the frozen candidate score h_j = <w_0, g_j - g> be obtained as
h_j = D_theta L_j[w_0] - <w_0, g> via per-candidate scalar JVP (torch.func),
without materializing each full candidate gradient?

METHOD A (reference, production semantics): frozen _grad_on_batches full grads
+ project_tangent(w_0, g_j - g), ambient backend.
METHOD B (candidate): per-micro-batch torch.func.jvp over the SAME loss graph
semantics (sum of (loss+aux)/accum), tangent = w_0 fixed.
METHOD C: structurally NOT_APPLICABLE (verified offline: 4 hash groups are
pairwise disjoint -> no reusable shared pool -> no amortization).

Stages are timed separately (baseline rollout, endpoint grad, shared reverse
adjoint w_0, per-candidate and all-M scoring for A/B, selected-gradient
recompute, native-step reference) with CUDA events + peak-memory stats.
M=1/2/4 scoring prefixes are TIMING-ONLY (not new science candidates).

Numerical gate thresholds are PRE-FIXED in outputs/a0i_directional_scoring_v01/
contract.json (atol32=1e-5, rtol32=1e-4). Raw errors reported regardless.
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

OUT = PROJECT_ROOT / "outputs/a0i_directional_scoring_v01"
B_OUT = OUT / "a0i_b"

import a0h15_production_entry_v01 as pe  # noqa: E402

from fr.a0c_adamw_response_propagation_v01 import project_tangent  # noqa: E402
from fr.minimind_v03_runtime import raw_model  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402

import a0g_decision_utility_v01 as a0g  # noqa: E402

LAMBDA = 0.05
ATOL32 = 1e-5
RTOL32 = 1e-4


# ---------------------------------------------------------------------------
# timing helper
# ---------------------------------------------------------------------------
def _timed(rt, fn):
    torch = rt["torch"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_ev.record()
    out = fn()
    end_ev.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {"wall_s": t1 - t0, "cuda_ms": start_ev.elapsed_time(end_ev),
            "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "result": out}


def _untimed(fn):
    return {"wall_s": float("nan"), "cuda_ms": float("nan"),
            "peak_alloc_mb": float("nan"), "peak_reserved_mb": float("nan"),
            "result": fn()}


# ---------------------------------------------------------------------------
# shared prefix: baseline rollout + endpoint grad + shared reverse adjoint
# (mirrors pe.run_adjoint_only exactly, but without candidate work and with
# stage-level timing)
# ---------------------------------------------------------------------------
def shared_prefix(rt, ctx, stream, H, counters):
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    hvp_ctx = None  # fp32-hvp adjoint config (production candidate)
    from contextlib import nullcontext
    hvp_ctx = nullcontext()

    pe._restore_and_seed(rt, ctx, stream)
    snapshots = {}
    native_g = None
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        counters["baseline_steps"] += 1
        snapshots[horizon] = snap
        if horizon == 1:
            native_g = {k: v.detach().clone() for k, v in native_gradient.items()}
        pe._clip_only(rt, native_gradient, snap)
        pe._step_once(rt)

    v_H = pe._endpoint_grad(rt, counters)

    from fr.a0h_adjoint_v01 import adamw_tangent_step_adjoint
    a0h1 = pe._a0h1

    def clip_transpose(b_map, snap):
        counters["clip_transpose_calls"] += 1
        return a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_map,
            grad_clip=float(cfg["grad_clip"]), preclip_norm=snap["preclip_norm"])

    H = int(H)
    p_th = {k: v.detach().clone() for k, v in v_H.items()}
    p_m = a0g._zeros_like_named(named)
    p_v = a0g._zeros_like_named(named)
    w_by_k = {}
    b_last = None
    for k in range(H, 0, -1):
        snap = snapshots[k]
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=snap,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        counters["reverse_adjoint_steps"] += 1
        b_k = adj["cotangent_dc"]
        w_k = clip_transpose(b_k, snap)
        if k >= 2:
            w_by_k[k] = w_k
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        if k >= 2:
            rng_now = capture_rng_state()
            restore_rng_state(snap["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(snap["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"],
                                w_k, hvp_ctx, int(cfg["accumulation_steps"]))
            counters["hvp_calls"] += 1
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
        b_last = b_k
    w_0 = clip_transpose(b_last, snapshots[1])
    return {"snapshots": snapshots, "native_g": native_g, "v_H": v_H,
            "w_0": w_0, "counters": counters}


# ---------------------------------------------------------------------------
# METHOD A: full-grad scoring (production semantics, ambient backend)
# ---------------------------------------------------------------------------
def score_method_A(rt, cand_tensors_by_j, native_g, w_0, counters):
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    scores = {}
    per_cand = {}
    for j, tensors in cand_tensors_by_j.items():
        rng_snap = capture_rng_state()
        g_j = a0g._grad_on_batches(torch, rt["model"], named, tensors,
                                   rt["autocast_ctx"], int(cfg["accumulation_steps"]))
        restore_rng_state(rng_snap)
        counters["candidate_grad_calls"] += 1
        a_j = {name: g_j[name] - native_g[name] for name, _ in named}
        scores[str(j)] = project_tangent(w_0, a_j)
        counters["scoring_dot_calls"] += 1
        per_cand[str(j)] = g_j
    return scores, per_cand


def score_method_A_math(rt, cand_tensors_by_j, native_g, w_0, counters):
    """Kernel control: same full-grad semantics under forced math SDPA."""
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    scores = {}
    with a0g._math_sdpa_context(torch):
        for j, tensors in cand_tensors_by_j.items():
            rng_snap = capture_rng_state()
            g_j = a0g._grad_on_batches(torch, rt["model"], named, tensors,
                                       rt["autocast_ctx"], int(cfg["accumulation_steps"]))
            restore_rng_state(rng_snap)
            counters["candidate_grad_calls"] += 1
            a_j = {name: g_j[name] - native_g[name] for name, _ in named}
            scores[str(j)] = project_tangent(w_0, a_j)
            counters["scoring_dot_calls"] += 1
    return scores


# ---------------------------------------------------------------------------
# METHOD B: per-candidate scalar JVP  D_theta L_j[w_0] - <w_0, g>
# ---------------------------------------------------------------------------
def _jvp_backend_context(torch, mode):
    if mode == "math":
        return a0g._math_sdpa_context(torch)
    from contextlib import nullcontext
    return nullcontext()


def score_method_B(rt, cand_tensors_by_j, w_0, w0_dot_g, counters, backend):
    torch = rt["torch"]
    cfg = rt["cfg"]
    from torch.func import functional_call, jvp
    rm = raw_model(rt["model"])
    params = {k: p.detach() for k, p in rm.named_parameters()}
    buffers = {k: b.detach() for k, b in rm.named_buffers()}
    t_p = {k: w_0[k].detach().to(params[k].dtype) for k in params}
    t_b = {k: torch.zeros_like(b) for k, b in buffers.items()}
    accum = int(cfg["accumulation_steps"])
    scores = {}
    with _jvp_backend_context(torch, backend):
        for j, tensors in cand_tensors_by_j.items():
            rng_snap = capture_rng_state()

            def fn(p, b, _tensors=tensors):
                total = None
                for (x, y) in _tensors:
                    out = functional_call(rm, (p, b), (x,), {"labels": y})
                    term = (out.loss + out.aux_loss) / accum
                    total = term if total is None else total + term
                return total

            with rt["autocast_ctx"]:
                _, d = jvp(fn, (params, buffers), (t_p, t_b))
            restore_rng_state(rng_snap)
            counters["scoring_dot_calls"] += 1  # jvp scoring acts in place of dot
            scores[str(j)] = float(d) - w0_dot_g
    return scores


# ---------------------------------------------------------------------------
# native optimizer step reference (fresh restore)
# ---------------------------------------------------------------------------
def native_step_time(rt, ctx, stream):
    def fn():
        native_gradient, _, _ = pe._forward_backward(
            rt, ctx, stream["batches"], 1, want_snapshot=False)
        pe._clip_and_step(rt, native_gradient, None)
    return fn


# ---------------------------------------------------------------------------
# probe: does JVP run through the model (ambient vs math)?
# ---------------------------------------------------------------------------
def cmd_probe(rt, ctx, stream, H, M, out_path):
    torch = rt["torch"]
    counters = pe._new_counters()
    prefix = shared_prefix(rt, ctx, stream, H, counters)
    w_0 = prefix["w_0"]
    native_g = prefix["native_g"]
    w0_dot_g = project_tangent(w_0, native_g)
    # restore theta_t for scoring
    raw_model(rt["model"]).load_state_dict(prefix["snapshots"][1]["theta_start"])

    cand = pe._candidates(ctx, stream, M)
    cand_tensors = {j: a0g._stack_all(rt["dataset"], cand[j], rt["device"]) for j in cand}

    # reference full grad on candidate 1 (ambient)
    g1 = a0g._grad_on_batches(torch, rt["model"], rt["named"], cand_tensors[1],
                              rt["autocast_ctx"], int(rt["cfg"]["accumulation_steps"]))
    a1 = {n: g1[n] - native_g[n] for n, _ in rt["named"]}
    h_ref = project_tangent(w_0, a1)

    result = {"state": f"{ctx.seed}/{ctx.step}", "stream": stream["stream_id"],
              "h_ref_ambient": h_ref, "w0_dot_g": w0_dot_g, "attempts": {}}
    for backend in ("ambient", "math"):
        try:
            s = score_method_B(rt, {1: cand_tensors[1]}, w_0, w0_dot_g,
                               pe._new_counters(), backend)
            result["attempts"][backend] = {"ok": True, "h_B": s["1"],
                                           "abs_err": abs(s["1"] - h_ref)}
        except Exception as e:  # noqa: BLE001 - probe records the raw failure
            result["attempts"][backend] = {"ok": False,
                                           "error_type": type(e).__name__,
                                           "error": str(e)[:2000]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


# ---------------------------------------------------------------------------
# main benchmark run
# ---------------------------------------------------------------------------
def cmd_run(rt, ctx, stream, H, M, repeats, warmup, out_dir, jvp_backend):
    torch = rt["torch"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand = pe._candidates(ctx, stream, M)

    timing_rows = []
    score_rows = []
    gpu_t0 = time.perf_counter()

    def one_pass(rep, timed):
        counters = pe._new_counters()
        stages = {}
        # ---- shared prefix
        t = _timed(rt, lambda: shared_prefix(rt, ctx, stream, H, counters)) if timed \
            else {"wall_s": float("nan"), "cuda_ms": float("nan"),
                  "peak_alloc_mb": float("nan"), "peak_reserved_mb": float("nan"),
                  "result": shared_prefix(rt, ctx, stream, H, counters)}
        sp = t["result"]
        stages["shared_prefix"] = {k: t[k] for k in ("wall_s", "cuda_ms",
                                                     "peak_alloc_mb", "peak_reserved_mb")}
        # split prefix into rollout / endpoint+walk is intrusive; report combined
        # plus a separate endpoint+walk timing on a fresh identical prefix is
        # skipped to save budget: combined is what both methods share.
        w_0 = sp["w_0"]
        native_g = sp["native_g"]
        w0_dot_g = project_tangent(w_0, native_g)

        # restore theta_t + candidate tensors
        raw_model(rt["model"]).load_state_dict(sp["snapshots"][1]["theta_start"])
        cand_tensors = {j: a0g._stack_all(rt["dataset"], cand[j], rt["device"]) for j in cand}

        def run_A():
            c = pe._new_counters()
            s, _ = score_method_A(rt, cand_tensors, native_g, w_0, c)
            return s
        def run_Amath():
            c = pe._new_counters()
            return score_method_A_math(rt, cand_tensors, native_g, w_0, c)
        def run_B():
            c = pe._new_counters()
            return score_method_B(rt, cand_tensors, w_0, w0_dot_g, c, jvp_backend)

        order = [("A", run_A), ("B", run_B)] if rep % 2 == 0 else [("B", run_B), ("A", run_A)]
        scores = {}
        for name, fn in order:
            t = _timed(rt, fn) if timed else _untimed(fn)
            scores[name] = t["result"]
            stages[f"scoring_{name}"] = {k: t[k] for k in ("wall_s", "cuda_ms",
                                                           "peak_alloc_mb", "peak_reserved_mb")}
        # per-candidate timing (subset of the M=4 pass, timed separately, same restore)
        per_cand = {}
        for j in cand:
            sub = {j: cand_tensors[j]}
            def one(_sub=sub, _j=j):
                c = pe._new_counters()
                s, _ = score_method_A(rt, _sub, native_g, w_0, c)
                return s
            t = _timed(rt, one) if timed else _untimed(one)
            per_cand[f"A_j{j}"] = t["wall_s"]
        for j in cand:
            sub = {j: cand_tensors[j]}
            def oneB(_sub=sub, _j=j):
                c = pe._new_counters()
                return score_method_B(rt, _sub, w_0, w0_dot_g, c, jvp_backend)
            t = _timed(rt, oneB) if timed else _untimed(oneB)
            per_cand[f"B_j{j}"] = t["wall_s"]
        # M=1,2 prefix timing (TIMING ONLY)
        prefix_times = {}
        for m in (1, 2):
            keys = sorted(cand, key=int)[:m]
            sub = {j: cand_tensors[j] for j in keys}
            def mA(_sub=sub):
                c = pe._new_counters()
                s, _ = score_method_A(rt, _sub, native_g, w_0, c)
                return s
            def mB(_sub=sub):
                c = pe._new_counters()
                return score_method_B(rt, _sub, w_0, w0_dot_g, c, jvp_backend)
            tA = _timed(rt, mA) if timed else _untimed(mA)
            tB = _timed(rt, mB) if timed else _untimed(mB)
            prefix_times[f"A_M{m}"] = tA["wall_s"]
            prefix_times[f"B_M{m}"] = tB["wall_s"]
        # kernel control (A under math SDPA), timed once per pass
        t = _timed(rt, run_Amath) if timed else _untimed(run_Amath)
        scores["A_math"] = t["result"]
        stages["scoring_A_math"] = {k: t[k] for k in ("wall_s", "cuda_ms",
                                                      "peak_alloc_mb", "peak_reserved_mb")}

        # ---- selected-gradient recompute (production decision = argmin of A)
        pred = {str(j): LAMBDA * scores["A"][str(j)] for j in cand}
        pred["0"] = 0.0
        chosen = min(range(0, int(M) + 1), key=lambda j: (pred[str(j)], j))
        def recompute():
            c = pe._new_counters()
            if chosen == 0:
                return None
            rng_snap = capture_rng_state()
            g = a0g._grad_on_batches(torch, rt["model"], rt["named"], cand_tensors[chosen],
                                     rt["autocast_ctx"], int(rt["cfg"]["accumulation_steps"]))
            restore_rng_state(rng_snap)
            return g
        t = _timed(rt, recompute) if timed else _untimed(recompute)
        stages["selected_gradient_recompute"] = {
            k: t[k] for k in ("wall_s", "cuda_ms", "peak_alloc_mb", "peak_reserved_mb")}
        stages["selected_gradient_recompute"]["chosen_id"] = chosen

        # ---- native step reference
        t = _timed(rt, native_step_time(rt, ctx, stream)) if timed else \
            _untimed(native_step_time(rt, ctx, stream))
        stages["native_step"] = {k: t[k] for k in ("wall_s", "cuda_ms",
                                                   "peak_alloc_mb", "peak_reserved_mb")}

        # ---- numerics
        rec = {"repeat": rep, "state": f"{ctx.seed}/{ctx.step}",
               "stream": stream["stream_id"], "chosen_id": chosen}
        for name in ("A", "A_math", "B"):
            ranks = sorted(range(0, int(M) + 1),
                           key=lambda j: (pred[str(j)] if name == "A"
                                          else ((0.0 if j == 0 else LAMBDA * scores[name][str(j)]), j)))
            rec[f"ranking_{name}"] = ranks
            rec[f"chosen_{name}"] = ranks[0]
        for j in cand:
            h_ref = scores["A"][str(j)]
            h_refm = scores["A_math"][str(j)]
            h_new = scores["B"][str(j)]
            rec[f"h_ref_j{j}"] = h_ref
            rec[f"h_refmath_j{j}"] = h_refm
            rec[f"h_new_j{j}"] = h_new
            rec[f"abs_err_j{j}"] = abs(h_new - h_ref)
            rec[f"abs_err_isolated_j{j}"] = abs(h_new - h_refm)
            rec[f"rel_err_j{j}"] = abs(h_new - h_ref) / max(abs(h_ref), 1e-30)
        score_rows.append(rec)
        for stage, v in stages.items():
            row = {"repeat": rep, "state": rec["state"], "stream": stream["stream_id"],
                   "stage": stage}
            row.update(v)
            row.update(per_cand if stage == "scoring_A" else {})
            if stage == "scoring_A":
                for k2, v2 in prefix_times.items():
                    row[f"prefix_{k2}"] = v2
            timing_rows.append(row)
        return scores

    # warmup (untimed) + timed repeats
    for _ in range(warmup):
        one_pass(rep=-1, timed=False)
    for rep in range(repeats):
        one_pass(rep=rep, timed=True)
        print(f"[{ctx.seed}/{ctx.step}] repeat {rep} done", flush=True)

    gpu_seconds = time.perf_counter() - gpu_t0

    # ---- persist
    import csv as _csv
    fieldnames = list(dict.fromkeys(k for row in timing_rows for k in row.keys()))
    with open(out / "timing_raw.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(timing_rows)
    with open(out / "directional_scores.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)

    meta = {"state": f"{ctx.seed}/{ctx.step}", "stream": stream["stream_id"],
            "H": int(H), "M": int(M), "lambda": LAMBDA,
            "repeats": repeats, "warmup": warmup, "jvp_backend": jvp_backend,
            "gpu_seconds_for_state": gpu_seconds,
            "counters_last_pass": pe._new_counters()}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cmd", choices=["probe", "run"], required=True)
    ap.add_argument("--state", required=True, help="dev SEED/STEP, e.g. 2031/52")
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--jvp-backend", choices=["ambient", "math"], default="ambient")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rt = pe.build_runtime(args.device)
    preload = time.perf_counter() - t0
    seed_s, step_s = args.state.split("/")
    ctx = next(c for c in rt["ctxs"] if str(c.seed) == seed_s and str(c.step) == step_s)
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == args.stream)

    B_OUT.mkdir(parents=True, exist_ok=True)
    (B_OUT / "preload_seconds.json").write_text(json.dumps(
        {"state": args.state, "preload_s": preload}, indent=1))

    if args.cmd == "probe":
        return cmd_probe(rt, ctx, stream, args.horizon, args.m, Path(args.out))
    return cmd_run(rt, ctx, stream, args.horizon, args.m, args.repeats, args.warmup,
                   args.out, args.jvp_backend)


if __name__ == "__main__":
    sys.exit(main())
