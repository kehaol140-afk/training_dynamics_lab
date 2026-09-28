#!/usr/bin/env python3
"""A0J per-call trace probe: wrap pe primitives with RNG+model+optimizer hashing.

Traces two executions — (1) verbatim production rollout+endpoint, (2) module
aged_decision_pass(age=0) — and reports the FIRST call whose before/after state
hashes diverge. Read-only w.r.t. all frozen files.
"""
from __future__ import annotations

import json
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
import a0g_decision_utility_v01 as a0g  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import aged_decision_pass  # noqa: E402
from fr.minimind_v03_runtime import raw_model  # noqa: E402
from fr.rng import capture_rng_state  # noqa: E402

TRACE = {"tag": None, "events": []}


def _rng_hash():
    st = capture_rng_state()
    import torch, hashlib
    h = hashlib.sha256()
    h.update(st["torch_cpu"].numpy().tobytes())
    for s in st["torch_cuda_all"]:
        h.update(s.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _model_hash(rt):
    return a0g.stable_hash(raw_model(rt["model"]).state_dict())[:16]


def _opt_hash(rt):
    import hashlib, torch
    h = hashlib.sha256()
    for p in rt["optimizer"].param_groups[0]["params"]:
        st = rt["optimizer"].state.get(p)
        if st is None:
            continue
        h.update(str(float(st["step"])).encode())
        h.update(st["exp_avg"].detach().cpu().numpy().tobytes())
        h.update(st["exp_avg_sq"].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _wrap(rt):
    torch = rt["torch"]
    fns = {}


def install(rt):
    import torch

    def wrap(name):
        orig = getattr(pe, name)

        def wrapped(*args, **kwargs):
            pre = (_rng_hash(), _model_hash(rt), _opt_hash(rt))
            out = orig(*args, **kwargs)
            torch.cuda.synchronize()
            post = (_rng_hash(), _model_hash(rt), _opt_hash(rt))
            TRACE["events"].append({"call": len(TRACE["events"]), "fn": name,
                                    "pre_rng": pre[0], "post_rng": post[0],
                                    "pre_model": pre[1], "post_model": post[1],
                                    "pre_opt": pre[2], "post_opt": post[2]})
            return out
        setattr(pe, name, wrapped)
        return orig
    originals = {name: wrap(name) for name in
                 ["_restore_and_seed", "_baseline_no_step", "_candidate_grads",
                  "_clip_only", "_step_once", "_endpoint_grad"]}
    return originals


def uninstall(originals):
    for name, orig in originals.items():
        setattr(pe, name, orig)


def run_verbatim(rt, ctx, stream):
    TRACE["tag"] = "verbatim"
    named = rt["named"]
    pe._restore_and_seed(rt, ctx, stream)
    cand = pe._candidates(ctx, stream, 4)
    for horizon in range(1, 5):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        if horizon == 1:
            g_by_j = pe._candidate_grads(rt, cand)
        pe._clip_only(rt, native_gradient, snap)
        pe._step_once(rt)
    pe._endpoint_grad(rt, pe._new_counters())


def main() -> int:
    session_t0 = time.perf_counter()
    rt = pe.build_runtime("cuda:0")
    preload_s = time.perf_counter() - session_t0
    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == "2031/52")
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)

    t0 = time.perf_counter()
    originals = install(rt)
    run_verbatim(rt, ctx, stream)
    trace_v = list(TRACE["events"])
    TRACE["events"] = []
    m = aged_decision_pass(pe, rt, ctx, stream, age=0, H=4, M=4, want_stage_times=False)
    trace_m = list(TRACE["events"])
    uninstall(originals)

    # module also calls apply_Gt_T internals AFTER endpoint; truncate both traces
    # at the endpoint for the rollout comparison, then compare endpoint output.
    def endpoint_idx(tr):
        for i, e in enumerate(tr):
            if e["fn"] == "_endpoint_grad":
                return i
        return None

    ev, em = endpoint_idx(trace_v), endpoint_idx(trace_m)
    v_roll, m_roll = trace_v[:ev + 1], trace_m[:em + 1]

    def compare(a, b):
        first = None
        for ea, eb in zip(a, b):
            if ea["fn"] != eb["fn"]:
                return {"kind": "call_sequence", "a": ea, "b": eb}
            for k in ("pre_rng", "post_rng", "pre_model", "post_model", "pre_opt", "post_opt"):
                if ea[k] != eb[k]:
                    first = {"kind": "state", "call": ea["fn"], "field": k,
                             "verbatim": ea[k], "module": eb[k],
                             "call_index": ea["call"]}
                    return first
        if len(a) != len(b):
            return {"kind": "length", "a": len(a), "b": len(b)}
        return None

    import hashlib, torch
    result = {
        "tag": "A0J_PER_CALL_TRACE",
        "n_calls_verbatim": len(trace_v), "n_calls_module": len(trace_m),
        "rollout_divergence": compare(v_roll, m_roll),
        "verbatim_endpoint_post_model": v_roll[-1]["post_model"],
        "module_endpoint_post_model": m_roll[-1]["post_model"],
        "vH_match": m["v_H"] is not None,
        "preload_s": preload_s, "probe_s": time.perf_counter() - t0,
        "module_trace_tail": trace_m[em:em + 6],
    }
    (OUT / "per_call_trace.json").write_text(json.dumps(result, indent=1))
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + [
        {"phase": "preload_trace_probe", "seconds": preload_s,
         "kind": "NON_METERED_PRELOAD_GPU_HELD"},
        {"phase": "per_call_trace_probe", "seconds": result["probe_s"],
         "kind": "METERED"}]
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries
                            if e["phase"] != "TOTAL_SESSION")}, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
