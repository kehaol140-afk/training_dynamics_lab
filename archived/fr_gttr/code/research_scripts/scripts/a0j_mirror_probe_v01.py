#!/usr/bin/env python3
"""A0J mirror-divergence localization probe (part of the FAILED-PROBE deliverable).

Replicates the production rollout + reverse walk and the A0J mirrored rollout +
reverse walk in one process, hashing every intermediate (snapshots, v_H,
per-step cotangents b_k / w_k / hvp_u, final w, scores). Both pipelines are
deterministic; the first hash mismatch localizes the semantic divergence.

No production files are modified; the walk codes are copied here verbatim.
"""
from __future__ import annotations

import json
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

import a0g_decision_utility_v01 as a0g  # noqa: E402
import a0h15_production_entry_v01 as pe  # noqa: E402
import a0h_adjoint_dev_check_v01 as a0h1  # noqa: E402
from fr.a0c_adamw_response_propagation_v01 import project_tangent  # noqa: E402
from fr.a0h_adjoint_v01 import adamw_tangent_step_adjoint  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import clip_transpose_w  # noqa: E402
from fr.minimind_v03_runtime import raw_model  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402


def hmap(m):
    return a0g.stable_hash({k: v for k, v in m.items()})


def rollout_with_hashes(pe, rt, ctx, stream, H, M):
    """Production rollout verbatim + hashing."""
    torch = rt["torch"]
    named = rt["named"]
    pe._restore_and_seed(rt, ctx, stream)
    cand = pe._candidates(ctx, stream, M)
    snapshots = {}
    a_by_j = {}
    hashes = {"horizons": {}}
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        snapshots[horizon] = snap
        if horizon == 1:
            g_by_j = pe._candidate_grads(rt, cand)
            for j in cand:
                a_by_j[j] = {name: g_by_j[j][name] - native_gradient[name] for name, _ in named}
        pe._clip_only(rt, native_gradient, snap)
        pe._step_once(rt)
        hashes["horizons"][str(horizon)] = {
            "theta_start": a0g.stable_hash(snap["theta_start"]),
            "exp_avg": hmap(snap["exp_avg"]),
            "lr": snap["lr"], "step_after": snap["step_after"],
            "preclip_norm": snap["preclip_norm"],
            "native_gradient": hmap(snap["native_gradient"]),
        }
    v_H = pe._endpoint_grad(rt, pe._new_counters())
    hashes["v_H"] = hmap(v_H)
    hashes["a_by_j"] = {str(j): hmap(a) for j, a in a_by_j.items()}
    return snapshots, a_by_j, v_H, hashes


def walk_production(pe, rt, snapshots, v_H, H):
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    hvp_ctx = nullcontext()
    accum = int(cfg["accumulation_steps"])
    p_th = {k: v.detach().clone() for k, v in v_H.items()}
    p_m = {k: v.detach().clone() for k, v in pe._zeros_like(named).items()}
    p_v = {k: v.detach().clone() for k, v in pe._zeros_like(named).items()}
    hashes = {"steps": {}}
    b_last = None
    for k in range(int(H), 0, -1):
        snap = snapshots[k]
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=snap,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        b_k = adj["cotangent_dc"]
        w_k = a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_k,
            grad_clip=float(cfg["grad_clip"]), preclip_norm=snap["preclip_norm"])
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        step_hash = {"b_k": hmap(b_k), "w_k": hmap(w_k),
                     "p_in_th": hmap(adj["cotangent_theta"])}
        if k >= 2:
            rng_now = capture_rng_state()
            restore_rng_state(snap["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(snap["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"], w_k,
                                hvp_ctx, accum)
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
            step_hash["hvp_u"] = hmap(hvp_u)
        hashes["steps"][str(k)] = step_hash
        b_last = b_k
    w_0 = a0h1._clip_transpose(
        native_gradient=snapshots[1]["native_gradient"], b=b_last,
        grad_clip=float(cfg["grad_clip"]), preclip_norm=snapshots[1]["preclip_norm"])
    hashes["w_0"] = hmap(w_0)
    return w_0, hashes


def walk_mirror(pe, rt, snapshots, v_H, H, age=0):
    """A0J mirrored walk verbatim (module lines) + hashing."""
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    hvp_ctx = nullcontext()
    accum = int(cfg["accumulation_steps"])
    p_th = {k: v.detach().clone() for k, v in v_H.items()}
    zero = pe._zeros_like(named)
    p_m = {k: v.detach().clone() for k, v in zero.items()}
    p_v = {k: v.detach().clone() for k, v in zero.items()}
    hashes = {"steps": {}}
    b_last = None
    for k in range(int(age) + int(H), int(age), -1):
        s = snapshots[k]
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=s,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        b_k = adj["cotangent_dc"]
        w_k = clip_transpose_w(native_gradient=s["native_gradient"], b=b_k,
                               grad_clip=float(cfg["grad_clip"]),
                               preclip_norm=s["preclip_norm"])
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        step_hash = {"b_k": hmap(b_k), "w_k": hmap(w_k)}
        if k >= int(age) + 2:
            rng_now = capture_rng_state()
            restore_rng_state(s["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(s["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, s["batch_tensors"], w_k,
                                hvp_ctx, accum)
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
            step_hash["hvp_u"] = hmap(hvp_u)
        hashes["steps"][str(k)] = step_hash
        b_last = b_k
    w_exact = clip_transpose_w(native_gradient=snapshots[int(age) + 1]["native_gradient"],
                               b=b_last, grad_clip=float(cfg["grad_clip"]),
                               preclip_norm=snapshots[int(age) + 1]["preclip_norm"])
    hashes["w_0"] = hmap(w_exact)
    return w_exact, hashes


def main() -> int:
    session_t0 = time.perf_counter()
    rt = pe.build_runtime("cuda:0")
    torch = rt["torch"]
    preload_s = time.perf_counter() - session_t0
    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == "2031/52")
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
    named = rt["named"]

    t0 = time.perf_counter()
    # production pipeline end-to-end
    snap_p, a_p, vH_p, hash_roll_p = rollout_with_hashes(pe, rt, ctx, stream, 4, 4)
    w_p, hash_walk_p = walk_production(pe, rt, snap_p, vH_p, 4)
    scores_p = {str(j): float(project_tangent(w_p, a_p[j])) for j in a_p}

    # mirror pipeline end-to-end (fresh restore)
    snap_m, a_m, vH_m, hash_roll_m = rollout_with_hashes(pe, rt, ctx, stream, 4, 4)
    w_m, hash_walk_m = walk_mirror(pe, rt, snap_m, vH_m, 4, age=0)
    scores_m = {str(j): float(project_tangent(w_m, a_m[j])) for j in a_m}

    # cross-check: production walk on the MIRROR rollout snapshots
    w_x, hash_walk_x = walk_production(pe, rt, snap_m, vH_m, 4)

    def first_diff(a, b, path=""):
        diffs = []
        for k in a:
            if isinstance(a[k], dict):
                diffs += first_diff(a[k], b.get(k, {}), f"{path}{k}.")
            elif a[k] != b.get(k):
                diffs.append(f"{path}{k}: {a[k]} != {b.get(k)}")
        return diffs

    result = {
        "tag": "A0J_MIRROR_DIVERGENCE_LOCALIZATION",
        "unit": "2031/52/s0",
        "rollout_diffs": first_diff(hash_roll_p, hash_roll_m),
        "walk_prod_vs_mirror_walk_same_rollout_diffs": first_diff(hash_walk_x, hash_walk_m),
        "walk_prod_vs_walk_prod_diffs": first_diff(hash_walk_p, hash_walk_x),
        "scores_prod": scores_p, "scores_mirror": scores_m,
        "w_prod_vs_w_mirror_maxdiff": max(
            float(torch.max(torch.abs(w_p[n].detach().to(torch.float64)
                                      - w_m[n].detach().to(torch.float64))))
            for n, _ in named),
        "preload_s": preload_s, "probe_s": time.perf_counter() - t0,
    }
    (OUT / "mirror_divergence_localization.json").write_text(json.dumps(result, indent=1))
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + [
        {"phase": "preload_mirror_probe", "seconds": preload_s,
         "kind": "NON_METERED_PRELOAD_GPU_HELD"},
        {"phase": "mirror_divergence_localization_probe", "seconds": result["probe_s"],
         "kind": "METERED"}]
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries
                            if e["phase"] != "TOTAL_SESSION")}, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
