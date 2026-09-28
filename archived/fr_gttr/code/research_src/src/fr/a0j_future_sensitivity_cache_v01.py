"""A0J: future-sensitivity reuse — cached p_tau + fresh action-injection G_t (CACHE_P).

Dev-only algorithm study (contract: outputs/a0j_cached_future_sensitivity_v01/
contract.json). This module introduces NO new forward model and NO new science
semantics; it reuses the frozen primitives:

* ``fr.a0h_adjoint_v01.adamw_tangent_step_adjoint`` / ``clip_jvp_transpose``
  (the exact transposes frozen in A0H/A0H-1.5),
* ``research/scripts/a0h15_production_entry_v01.py`` (production entry:
  baseline fwd/bwd, clip, step, candidate grads, endpoint grad — mirrored
  call-for-call by the orchestration below),
* frozen A0G selector (``fr.a0g_decision_utility_v01.select_action``).

Three arms per aged decision unit (anchor tau, age k, t = tau+k):

* EXACT   w_exact = G_t^T p_t        (fresh H=4 shared-adjoint walk)
* CACHE_P w_cache = G_t^T p_tau      (anchor-cached cotangent, CURRENT snapshot)
* STALE_W w_stale = w_tau            (anchor's full w reused verbatim)

All arms score the SAME a_j(t) = g_j(t) - g_t computed once per unit with the
frozen FULL_GRAD_DOT scorer (project_tangent). Vector cosines are secondary
diagnostics only; the primary accuracy metrics are chosen-id preservation and
pairwise action-gap error (see contract).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Mapping, Optional, Sequence

from .a0c_adamw_response_propagation_v01 import project_tangent
from .a0h_adjoint_v01 import adamw_tangent_step_adjoint, clip_jvp_transpose
from .a0g_decision_utility_v01 import select_action as frozen_select_action

LAMBDA = 0.05


# ---------------------------------------------------------------------------
# action-step VJP primitive (the extracted, reusable G_t^T)
# ---------------------------------------------------------------------------
def clip_transpose_w(*, native_gradient: Mapping[str, object],
                     b: Mapping[str, object], grad_clip: float,
                     preclip_norm: float) -> Dict[str, object]:
    """Production-equivalent clip transpose (mirror of the frozen
    ``a0h_adjoint_dev_check_v01._clip_transpose`` wrapper)."""
    return clip_jvp_transpose(
        gradient_by_name=native_gradient,
        cotangent_by_name=b,
        max_norm=float(grad_clip),
        active=bool(preclip_norm > float(grad_clip)),
        clip_before_norm=float(preclip_norm),
    )["clip_jvp_transpose"]


def apply_Gt_T(*, named: Sequence[tuple[str, object]], optimizer,
               snapshot: Mapping, p: Mapping[str, Mapping[str, object]],
               grad_clip: float) -> Dict[str, object]:
    """w = G_t^T p for an arbitrary 3-channel cotangent p = (theta, m, v).

    This is the CACHE_P primitive: one optimizer-local analytic adjoint step
    (no future HVP, no model forward) + the clip transpose, using the CURRENT
    step snapshot (fresh raw gradient / m / v / optimizer internal step / LR /
    bias correction / clipping state). With p = p_tau (anchor cache) it yields
    w_cache(t); with p = p_t (fresh) it must reproduce the production inline
    w_0 exactly (implementation gate, contract implementation_gate_prefixed).
    """
    adj = adamw_tangent_step_adjoint(
        named_parameters=named, optimizer=optimizer, snapshot=snapshot,
        cotangent_theta_by_name=p["theta"], cotangent_m_by_name=p["m"],
        cotangent_v_by_name=p["v"], feedback_cotangent_by_name=None)
    w = clip_transpose_w(native_gradient=snapshot["native_gradient"],
                         b=adj["cotangent_dc"], grad_clip=float(grad_clip),
                         preclip_norm=snapshot["preclip_norm"])
    return {"w": w, "cotangent_dc": adj["cotangent_dc"],
            "adjoint_meta": {"d": adj["d"], "adjoint_step_after": adj["adjoint_step_after"]}}


def clone_p(p: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    return {ch: {k: v.detach().clone() for k, v in p[ch].items()} for ch in ("theta", "m", "v")}


def p_bytes(p: Mapping[str, Mapping[str, object]]) -> int:
    return int(sum(int(v.numel()) * int(v.element_size()) for ch in ("theta", "m", "v")
                   for v in p[ch].values()))


def map_bytes(m: Mapping[str, object]) -> int:
    return int(sum(int(v.numel()) * int(v.element_size()) for v in m.values()))


# ---------------------------------------------------------------------------
# frozen selector / scoring helpers (single definition point)
# ---------------------------------------------------------------------------
def score_actions(w: Optional[Mapping[str, object]],
                  a_by_j: Mapping[int, Mapping[str, object]]) -> Optional[Dict[int, float]]:
    """h_j = <w, a_j> for j=1..M; h_0 = 0 exactly. Returns None if w is None."""
    if w is None:
        return None
    h: Dict[int, float] = {0: 0.0}
    for j, a in a_by_j.items():
        h[int(j)] = float(project_tangent(w, a))
    return h


def predicted(h: Mapping[int, float]) -> Dict[int, float]:
    return {int(j): float(LAMBDA) * float(v) for j, v in h.items()}


def choose(h: Mapping[int, float]) -> int:
    """Frozen selector: argmin (lambda*h_j, action_id), j=0..M, tie -> lowest id.
    NON_FINITE scores are blocked before selection (frozen A0H20 guard)."""
    for j, v in h.items():
        if not math.isfinite(float(v)):
            raise ValueError(f"NON_FINITE score blocked from selection: j={j} value={v!r}")
    return int(frozen_select_action(predicted(h)))


def ranking(h: Mapping[int, float]) -> list[int]:
    for j, v in h.items():
        if not math.isfinite(float(v)):
            raise ValueError(f"NON_FINITE score blocked from ranking: j={j} value={v!r}")
    pred = predicted(h)
    return [int(j) for j in sorted(pred, key=lambda j: (pred[j], j))]


def max_abs_score_diff(h_a: Mapping[int, float], h_b: Mapping[int, float]) -> float:
    keys = sorted(set(h_a) | set(h_b))
    return max((abs(float(h_a[k]) - float(h_b[k])) for k in keys), default=0.0)


def rank_agreement(ra: Sequence[int], rb: Sequence[int]) -> bool:
    return list(ra) == list(rb)


# ---------------------------------------------------------------------------
# vector diagnostics (SECONDARY only — never a cache-acceptance gate)
# ---------------------------------------------------------------------------
def _flat_f64(m: Mapping[str, object]) -> "object":
    import torch
    parts = [v.detach().to(torch.float64).reshape(-1) for v in m.values()]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def cosine_maps(a: Optional[Mapping[str, object]], b: Optional[Mapping[str, object]]) -> Optional[float]:
    import torch
    if a is None or b is None:
        return None
    va = _flat_f64(a); vb = _flat_f64(b)
    na = float(torch.norm(va)); nb = float(torch.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(va, vb) / (na * nb))


def norm_map(a: Optional[Mapping[str, object]]) -> Optional[float]:
    import torch
    if a is None:
        return None
    return float(torch.norm(_flat_f64(a)))


def cosine_p(a: Optional[Mapping[str, Mapping[str, object]]],
             b: Optional[Mapping[str, Mapping[str, object]]]) -> Optional[float]:
    if a is None or b is None:
        return None
    import torch
    va = torch.cat([_flat_f64(a[ch]) for ch in ("theta", "m", "v")])
    vb = torch.cat([_flat_f64(b[ch]) for ch in ("theta", "m", "v")])
    na = float(torch.norm(va)); nb = float(torch.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(va, vb) / (na * nb))


def norm_p(a: Optional[Mapping[str, Mapping[str, object]]]) -> Optional[float]:
    if a is None:
        return None
    import torch
    va = torch.cat([_flat_f64(a[ch]) for ch in ("theta", "m", "v")])
    return float(torch.norm(va))


# ---------------------------------------------------------------------------
# pairwise action-gap diagnostics (PRIMARY accuracy metric family)
# ---------------------------------------------------------------------------
def gap_table(h_exact: Mapping[int, float], h_cache: Optional[Mapping[int, float]],
              h_stale: Optional[Mapping[int, float]]) -> list[dict]:
    """All pairs (j,l), j<l over the full 5-action set incl. no-op (0)."""
    ids = sorted(h_exact)
    rows = []
    pred_ex = predicted(h_exact)
    order = ranking(h_exact)
    top1, top2 = order[0], order[1]
    nonzero = [j for j in ids if j != 0]
    best_nonzero = min(nonzero, key=lambda j: (pred_ex[j], j))
    for i, j in enumerate(ids):
        for l in ids[i + 1:]:
            rows.append({
                "j": int(j), "l": int(l),
                "gap_exact": float(h_exact[j] - h_exact[l]),
                "gap_cache": (None if h_cache is None else float(h_cache[j] - h_cache[l])),
                "gap_stale": (None if h_stale is None else float(h_stale[j] - h_stale[l])),
                "gap_err_cache": (None if h_cache is None else float(
                    (h_cache[j] - h_cache[l]) - (h_exact[j] - h_exact[l]))),
                "gap_err_stale": (None if h_stale is None else float(
                    (h_stale[j] - h_stale[l]) - (h_exact[j] - h_exact[l]))),
                "is_exact_top1top2_pair": bool((j == top1 and l == top2) or (j == top2 and l == top1)),
                "is_bestnonzero_noop_pair": bool((j == best_nonzero and l == 0) or (j == 0 and l == best_nonzero)),
            })
    return rows


def top_margin(h: Mapping[int, float]) -> dict:
    pred = predicted(h)
    order = sorted(pred, key=lambda j: (pred[j], j))
    return {"top1": int(order[0]), "top2": int(order[1]),
            "margin_pred": float(pred[order[1]] - pred[order[0]])}


# ---------------------------------------------------------------------------
# decision preservation counting (pure)
# ---------------------------------------------------------------------------
def preservation(rows: Sequence[Mapping], *, method: str, ages: Sequence[int] = (1, 2, 4)) -> Dict[int, dict]:
    """rows: decision_preservation records. method in {cache, stale}."""
    out: Dict[int, dict] = {}
    for age in ages:
        sub = [r for r in rows if int(r["age"]) == int(age)]
        n = len(sub)
        chosen_match = sum(1 for r in sub if int(r[f"chosen_{method}"]) == int(r["chosen_exact"]))
        rank_match = sum(1 for r in sub if bool(r[f"rank_agreement_{method}"]))
        out[int(age)] = {"n": n, "chosen_id_preservation": f"{chosen_match}/{n}",
                         "chosen_id_fraction": (chosen_match / n if n else None),
                         "full_ranking_agreement": f"{rank_match}/{n}"}
    return out


# ---------------------------------------------------------------------------
# Kr / Ks cost projection (pure arithmetic — COST_PROJECTION_ONLY)
# ---------------------------------------------------------------------------
def kr_ks_projection(*, T_refresh: float, T_score: float, T_execute: float,
                     T_GTp: float, T_native: float,
                     grid: Sequence[tuple[int, int]] = ((1, 1), (2, 1), (4, 1), (2, 2),
                                                        (4, 2), (4, 4), (8, 4), (8, 8)),
                     utility_ages_supported: Sequence[int] = (1, 2, 4)) -> list[dict]:
    """extra_native_step_equiv = (T_refresh/K_r + (T_score+T_execute+T_GTp)/K_s)/T_native.

    Every row is COST_ONLY / UTILITY_NOT_VALIDATED: cost arithmetic, never
    closed-loop utility. Rows whose K_r exceeds the supported cache-age evidence
    are additionally flagged beyond_supported_age."""
    rows = []
    supported_max = max(utility_ages_supported) if utility_ages_supported else 0
    for (kr, ks) in grid:
        extra = (float(T_refresh) / int(kr) +
                 (float(T_score) + float(T_execute) + float(T_GTp)) / int(ks)) / float(T_native)
        rows.append({
            "K_r": int(kr), "K_s": int(ks),
            "extra_native_step_equiv": float(extra),
            "refresh_share_equiv": float(T_refresh) / int(kr) / float(T_native),
            "score_share_equiv": (float(T_score) + float(T_execute) + float(T_GTp)) / int(ks) / float(T_native),
            "beyond_measured_max_age": bool(int(kr) > int(supported_max)),
            "claim": "COST_ONLY / UTILITY_NOT_VALIDATED",
        })
    return rows


# ---------------------------------------------------------------------------
# GPU orchestration (imports production entry primitives; not unit-tested)
# ---------------------------------------------------------------------------
def aged_decision_pass(pe, rt, ctx, stream, *, age: int, H: int, M: int,
                       cached_p: Optional[Mapping[str, Mapping[str, object]]] = None,
                       cached_w: Optional[Mapping[str, object]] = None,
                       want_stage_times: bool = False,
                       counters: Optional[dict] = None) -> dict:
    """One complete three-arm decision at t = tau+age on the frozen stream.

    Order of operations (mirror of ``pe.run_adjoint_only`` with the pending
    horizon shifted to ``age+1``):
      restore checkpoint -> seed future stream -> fast-forward `age` native
      steps (stream horizons 1..age) -> pending horizon (age+1): fwd/bwd with
      snapshot -> candidate grads (RNG-wrapped, before clip) -> clip_only ->
      native rollout horizons age+2..age+H with per-horizon snapshots ->
      endpoint V_score grad -> shared reverse adjoint walk ->
      w_exact (inline) + p_t (fresh cotangent at the pending step) ->
      gate: apply_Gt_T(snap_pending, p_t) vs inline w_exact ->
      CACHE_P arm: apply_Gt_T(snap_pending, cached_p) ->
      STALE_W arm: cached_w verbatim.
    Scores for every arm use the SAME a_j(t). Returns maps + per-stage times.
    """
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    grad_clip = float(cfg["grad_clip"])
    accum = int(cfg["accumulation_steps"])
    hvp_ctx = nullcontext_if_fp32(torch, rt, True)
    cnt = counters if counters is not None else pe._new_counters()
    times: Dict[str, float] = {}
    mem: Dict[str, float] = {}

    def _tick(name: str, fn):
        if not want_stage_times:
            return fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - t0
        mem[name] = float(torch.cuda.max_memory_allocated() / 2**20)
        return out

    pe._restore_and_seed(rt, ctx, stream)
    cand = pe._candidates(ctx, stream, M)

    # fast-forward `age` native steps (stream horizons 1..age)
    for h in range(1, int(age) + 1):
        def _ff(hh=h):
            native_gradient, _, _ = pe._baseline_no_step(rt, ctx, stream["batches"], hh,
                                                         want_snapshot=False)
            pe._clip_only(rt, native_gradient, None)
            pe._step_once(rt)
            cnt["baseline_steps"] += 1
        _tick(f"ff_step_h{h}", _ff)

    # pending horizon (age+1): snapshot + candidates + clip (production order)
    snap: dict = {}
    def _pending_fwd():
        native_gradient, _, s = pe._baseline_no_step(rt, ctx, stream["batches"], int(age) + 1,
                                                     want_snapshot=True)
        snap.update(s)
        snap["native_gradient"] = native_gradient
    _tick("pending_fwd_bwd", _pending_fwd)

    def _cand():
        g_by_j = pe._candidate_grads(rt, cand)
        cnt["candidate_grad_calls"] += len(cand)
        snap["g_by_j"] = g_by_j
    _tick("candidate_grads", _cand)

    def _clip():
        pe._clip_only(rt, snap["native_gradient"], snap)
        snap["a_by_j"] = {int(j): {name: snap["g_by_j"][j][name] - snap["native_gradient"][name]
                                   for name, _ in named}
                          for j in cand}
        # the pending native step EXECUTES here (production order: clip -> step),
        # after the pre-step snapshot/candidates/clip are captured above
        pe._step_once(rt)
    _tick("pending_clip", _clip)

    g_t = snap["native_gradient"]
    a_by_j = snap["a_by_j"]

    # native rollout horizons age+2..age+H with snapshots
    snapshots = {int(age) + 1: snap}
    def _rollout():
        for h in range(int(age) + 2, int(age) + int(H) + 1):
            native_gradient, _, s = pe._baseline_no_step(rt, ctx, stream["batches"], h,
                                                         want_snapshot=True)
            snapshots[h] = s
            pe._clip_only(rt, native_gradient, s)
            pe._step_once(rt)
            cnt["baseline_steps"] += 1
    _tick("rollout_rest", _rollout)

    v_H = _tick("endpoint_grad", lambda: pe._endpoint_grad(rt, cnt))

    # shared reverse adjoint walk (mirror of pe.run_adjoint_only loop)
    from fr.minimind_v03_runtime import raw_model
    from fr.rng import capture_rng_state, restore_rng_state
    import a0g_decision_utility_v01 as a0g

    p_th = {k: v.detach().clone() for k, v in v_H.items()}
    zero = pe._zeros_like(named)
    p_m = {k: v.detach().clone() for k, v in zero.items()}
    p_v = {k: v.detach().clone() for k, v in zero.items()}
    w_exact: Optional[Dict[str, object]] = None
    p_t: Optional[Dict[str, Dict[str, object]]] = None
    walk_times: Dict[str, float] = {}
    b_last = None
    for k in range(int(age) + int(H), int(age), -1):
        s = snapshots[k]
        if want_stage_times:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if k == int(age) + 1:
            p_t = {"theta": {n: t.detach().clone() for n, t in p_th.items()},
                   "m": {n: t.detach().clone() for n, t in p_m.items()},
                   "v": {n: t.detach().clone() for n, t in p_v.items()}}
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=s,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        cnt["reverse_adjoint_steps"] += 1
        b_k = adj["cotangent_dc"]
        if want_stage_times:
            torch.cuda.synchronize()
            walk_times[f"e1_adj_h{k}"] = time.perf_counter() - t0
            t1 = time.perf_counter()
        w_k = clip_transpose_w(native_gradient=s["native_gradient"], b=b_k,
                               grad_clip=grad_clip, preclip_norm=s["preclip_norm"])
        cnt["clip_transpose_calls"] += 1
        if want_stage_times:
            torch.cuda.synchronize()
            walk_times[f"e2_clipT_h{k}"] = time.perf_counter() - t1
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        if k >= int(age) + 2:
            if want_stage_times:
                torch.cuda.synchronize()
                t2 = time.perf_counter()
            rng_now = capture_rng_state()
            restore_rng_state(s["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(s["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, s["batch_tensors"], w_k,
                                hvp_ctx, accum)
            cnt["hvp_calls"] += 1
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
            if want_stage_times:
                torch.cuda.synchronize()
                walk_times[f"e3_hvp_h{k}"] = time.perf_counter() - t2
        b_last = b_k
    # inline w_0 for the pending step (production semantics)
    w_exact = clip_transpose_w(native_gradient=snap["native_gradient"], b=b_last,
                               grad_clip=grad_clip, preclip_norm=snap["preclip_norm"])
    cnt["clip_transpose_calls"] += 1

    # implementation gate: extracted apply_Gt_T must reproduce inline w_exact
    gate = None
    w_gate = None
    if p_t is not None:
        def _gate():
            return apply_Gt_T(named=named, optimizer=rt["optimizer"], snapshot=snap,
                              p=p_t, grad_clip=grad_clip)
        gate_out = _tick("gate_GtTp_fresh", _gate)
        w_gate = gate_out["w"]
        h_gate = score_actions(w_gate, a_by_j)
        h_exact = score_actions(w_exact, a_by_j)
        gate = {"max_abs_w_diff": _max_abs_map_diff(w_exact, w_gate),
                "max_abs_score_diff": max_abs_score_diff(h_exact, h_gate),
                "rank_agreement": rank_agreement(ranking(h_exact), ranking(h_gate)),
                "chosen_id_match": int(choose(h_exact)) == int(choose(h_gate))}

    # CACHE_P arm
    w_cache = None
    if cached_p is not None:
        def _cache():
            return apply_Gt_T(named=named, optimizer=rt["optimizer"], snapshot=snap,
                              p=cached_p, grad_clip=grad_clip)
        w_cache = _tick("gtTp_cached", _cache)["w"]

    # STALE_W arm (verbatim map)
    w_stale = cached_w

    # scoring (same a_j for every arm)
    def _score_all():
        return {"exact": score_actions(w_exact, a_by_j),
                "cache": score_actions(w_cache, a_by_j),
                "stale": score_actions(w_stale, a_by_j)}
    scores = _tick("scoring_dots", _score_all)
    for _ in scores.values():
        if _ is not None:
            cnt["scoring_dot_calls"] += len(a_by_j)

    return {"age": int(age), "g_t": g_t, "a_by_j": a_by_j, "snap_pending": snap,
            "w_exact": w_exact, "w_cache": w_cache, "w_stale": w_stale,
            "p_t": p_t, "v_H": v_H, "scores": scores,
            "chosen": {arm: (None if scores[arm] is None else int(choose(scores[arm])))
                       for arm in scores},
            "ranks": {arm: (None if scores[arm] is None else ranking(scores[arm]))
                      for arm in scores},
            "gate": gate, "times": times, "walk_times": walk_times, "mem": mem,
            "counters": cnt}


def _max_abs_map_diff(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    import torch
    total = 0.0
    for name in a:
        va = a[name].detach().to(torch.float64)
        vb = b[name].detach().to(torch.float64)
        total = max(total, float(torch.max(torch.abs(va - vb))) if va.numel() else 0.0)
    return float(total)


def nullcontext_if_fp32(torch, rt, fp32: bool):
    from contextlib import nullcontext
    return nullcontext() if fp32 else rt["autocast_ctx"]


# ---------------------------------------------------------------------------
# five-action dual-label replay for utility retention (A0G frozen semantics)
# ---------------------------------------------------------------------------
def replay_five_actions(pe, rt, ctx, stream, *, age: int, H: int, cand,
                        lambda_: float = LAMBDA) -> Dict[int, dict]:
    """Replay j=0..4 from z_(tau+age): restore -> seed -> ff `age` native steps
    -> inject g + lambda*(g_j - g) pre-clip at pending horizon age+1 (j>0) ->
    native rollout horizons age+2..age+H -> dual endpoint V_score + V_eval.
    Each action branch is an independent restore (production discipline)."""
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    results: Dict[int, dict] = {}
    for j in [0] + sorted(int(c) for c in cand):
        pe._restore_and_seed_pre(rt, ctx, stream["stream_seed"])
        for h in range(1, int(age) + 1):
            native_gradient, _, _ = pe._forward_backward(rt, ctx, stream["batches"], h,
                                                         want_snapshot=False)
            pe._clip_and_step(rt, native_gradient, None)
        for h in range(int(age) + 1, int(age) + int(H) + 1):
            native_gradient, _, _ = pe._forward_backward(rt, ctx, stream["batches"], h,
                                                         want_snapshot=False)
            if h == int(age) + 1 and j != 0:
                cand_tensors = a0g_stack_all(rt, cand[j])
                from fr.rng import capture_rng_state, restore_rng_state
                rng_snap = capture_rng_state()
                g_j = a0g_grad_on_batches(torch, rt, cand_tensors, accum)
                restore_rng_state(rng_snap)
                del cand_tensors
                for name, p in named:
                    p.grad.copy_(native_gradient[name]
                                 + float(lambda_) * (g_j[name] - native_gradient[name]))
            pe._clip_and_step(rt, native_gradient, None)
        from fr.rng import capture_rng_state, restore_rng_state
        rng_snap = capture_rng_state()
        v_score = float(a0g_val_loss_only(torch, rt, rt["val_score_tensors"]))
        v_eval = float(a0g_val_loss_only(torch, rt, rt["val_eval_tensors"]))
        restore_rng_state(rng_snap)
        from fr.minimind_v03_runtime import raw_model
        from .hashing import stable_hash
        results[int(j)] = {"v_score": v_score, "v_eval": v_eval,
                           "endpoint_model_hash": stable_hash(raw_model(rt["model"]).state_dict())}
    return results


def _a0g():
    import a0g_decision_utility_v01 as a0g
    return a0g


def a0g_stack_all(rt, batches):
    return _a0g()._stack_all(rt["dataset"], batches, rt["device"])


def a0g_grad_on_batches(torch, rt, tensors, accum):
    return _a0g()._grad_on_batches(torch, rt["model"], rt["named"], tensors,
                                   rt["autocast_ctx"], accum)


def a0g_val_loss_only(torch, rt, tensors):
    return _a0g()._val_loss_only(torch, rt["model"], tensors, rt["autocast_ctx"])
