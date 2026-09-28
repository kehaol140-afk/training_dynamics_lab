"""A0H-1.5 GPU diagnostics: telescope closure, linearity, self-adjoint vs independent-VJP.

Frozen contract: docs/FR_A0H15_PROTOCOL.md (hashes in outputs/a0h15_vjp_shared_v01/contract/).
Read-only wrt all A0G/A0H-1 frozen artifacts. One joint pass per dev unit reuses the frozen
forward machinery from a0h_adjoint_dev_check_v01._run_joint_pass via import; this script
adds ONLY new diagnostics (no re-implementation of the frozen forward):

  CHK-TELE  telescope identity closure per candidate (BF16 ops; diagnostic dots FP64)
  CHK-LIN   repeatability / linearity / K(0) at each future-step HVP point (C model)
  CHK-SA    self-adjoint defect  <Kx,y> - <Hx,y>      (y = adjoint walk direction w_k)
  CHK-IVJP  independent-VJP defect <Kx,y> - <x, VJP_K(anchor,y)> via a tangent-input-only
            wrapper around the frozen HVP call (anchor = actual trajectory point)

Anchors: zero-direction anchor (base rollout, direction 0) + actual candidate tangent
anchors j=1..4 at each future step. Output: one JSON per dev unit in
outputs/a0h15_vjp_shared_v01/diagnostics/.

GPU budget: --budget-seconds caps this script's share of the 30-min round cap.
"""
from __future__ import annotations

import argparse
import importlib.util
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

from fr.a0c_adamw_response_propagation_v01 import (  # noqa: E402
    adamw_tangent_step, clip_jvp, project_tangent,
)
from fr.full_state import load_full_training_state  # noqa: E402
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch  # noqa: E402
from fr.native_adamw_guard import assert_native_adamw  # noqa: E402
from fr.native_schedule import set_native_micro_lr  # noqa: E402
from fr.pilot_partition import load_partition  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.stream_rng import seed_future_stream  # noqa: E402

import a0g_decision_utility_v01 as a0g  # noqa: E402

# Reuse the A0H-1 joint pass verbatim (frozen code, read-only import).
_spec = importlib.util.spec_from_file_location(
    "a0h_adjoint_dev_check_v01",
    PROJECT_ROOT / "research" / "scripts" / "a0h_adjoint_dev_check_v01.py",
)
_a0h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a0h1)

OUTPUT_DIR = "outputs/a0h15_vjp_shared_v01"
REPEATS = 3  # CHK-LIN REP: >=3 repeats of same direction (frozen contract §4)

# Pre-registered diagnostic tolerances (FR_A0H15_PROTOCOL.md §3/§4). These are
# DIAGNOSTIC only; the original BF16 elementwise gate (atol=1e-6, rtol=1e-4) is
# read in unchanged from the A0H-1 contract and NOT re-applied here.
ATOL_TELE = 1e-3
RTOL_TELE = 1e-2
LIN_REL_TOL = 1e-2


def _require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def _inner64(mapping_a, mapping_b) -> float:
    import torch
    total = 0.0
    for name in mapping_a:
        a = mapping_a[name].detach().to(device="cpu", dtype=torch.float64)
        b = mapping_b[name].detach().to(device="cpu", dtype=torch.float64)
        total += float(torch.sum(a * b))
    return total


def _norm64(mapping) -> float:
    import torch
    total = 0.0
    for v in mapping.values():
        vv = v.detach().to(device="cpu", dtype=torch.float64)
        total += float(torch.sum(vv.square()))
    return float(total ** 0.5)


def _sub(a, b):
    return {k: a[k] - b[k] for k in a}


def _add(a, b):
    return {k: a[k] + b[k] for k in a}


def _scale(a, s):
    return {k: a[k] * s for k in a}


def _zeros_like(named):
    return {name: p.detach().clone().zero_() for name, p in named}


def _hvp_direction_vjp(torch, model, named, batch_tensors, direction_leaf, y_map,
                       autocast_ctx, accumulation_steps):
    """Frozen-replica HVP (math-SDPA, autocast, same loss normalization) computed
    with create_graph=True at BOTH differentiation orders, returning
    (hvp detached, direction-vjp detached).

    ``direction_leaf`` must carry requires_grad (tangent-input-only wrapper,
    contract §2). The batch/model/RNG are restored by the caller and are constants
    here. No silent zero-fill: a None HVP/VJP entry raises. Third-order chain:
    g1 = grad(loss, p, create_graph); dot = <g1, x>; hvp = grad(dot, p,
    create_graph=True); s = <hvp, y>; vjp = grad(s, x).
    """
    _math_sdpa_context = a0g._math_sdpa_context
    parameters = [p for _, p in named]
    with _math_sdpa_context(torch):
        losses = []
        with autocast_ctx:
            for x, y in batch_tensors:
                result = model(x, labels=y)
                losses.append((result.loss + result.aux_loss) / accumulation_steps)
        total_loss = sum(losses)
        gradients = torch.autograd.grad(total_loss, parameters, create_graph=True,
                                        allow_unused=True)
        dot = None
        for (name, parameter), gradient in zip(named, gradients):
            if gradient is None:
                continue
            direction = direction_leaf[name].to(device=parameter.device, dtype=gradient.dtype)
            term = torch.sum(gradient * direction)
            dot = term if dot is None else dot + term
        if dot is None:
            raise RuntimeError("HVP graph has no differentiable gradient")
        hvp_values = torch.autograd.grad(dot, parameters, create_graph=True,
                                         allow_unused=True)
        s = None
        for (name, parameter), hvp in zip(named, hvp_values):
            if hvp is None:
                continue
            term = torch.sum(hvp.to(torch.float32) * y_map[name].to(torch.float32))
            s = term if s is None else s + term
        if s is None or not torch.is_tensor(s) or not s.requires_grad:
            raise RuntimeError(
                "direction-VJP scalar has no grad_fn (nested AD unsupported here)")
        vjp_values = torch.autograd.grad(
            s, [direction_leaf[n] for n in direction_leaf], allow_unused=True)
    hvp_out = {}
    vjp_out = {}
    for (name, parameter), hvp, vjp in zip(named, hvp_values, vjp_values):
        if hvp is None or vjp is None:
            raise RuntimeError(
                f"_hvp_direction_vjp: None grad for {name} (connectivity break; "
                "silent zero-fill refused per contract §2)")
        hvp_out[name] = hvp.detach()
        vjp_out[name] = vjp.detach()
    return hvp_out, vjp_out


def _clone_map(m):
    return {k: v.detach().clone() for k, v in m.items()}


def _first_order_graph_status(torch, model, parameters, batch_tensors, autocast_ctx,
                              accumulation_steps):
    """Diagnostic: does the first-order gradient carry a graph (create_graph) at
    the current model state? Returns counts; never raises."""
    _math_sdpa_context = a0g._math_sdpa_context
    try:
        with _math_sdpa_context(torch):
            losses = []
            with autocast_ctx:
                for x, y in batch_tensors:
                    result = model(x, labels=y)
                    losses.append((result.loss + result.aux_loss) / accumulation_steps)
            total_loss = sum(losses)
            if not torch.is_tensor(total_loss) or not total_loss.requires_grad:
                return {"loss_requires_grad": False, "n_requires_grad": 0,
                        "total": len(parameters)}
            grads = torch.autograd.grad(total_loss, parameters, create_graph=True,
                                        allow_unused=True)
            n = sum(1 for g in grads if g is not None and g.requires_grad)
            return {"loss_requires_grad": True, "n_requires_grad": n,
                    "total": len(parameters)}
    except RuntimeError as exc:
        return {"loss_requires_grad": "error", "n_requires_grad": -1,
                "total": len(parameters), "error": str(exc)[:200]}


def _run_diagnostics_pass(*, torch, model, optimizer, scaler, autocast_ctx, dataset,
                          named, checkpoint_path, batches, stream_seed,
                          cand_batches_by_j, val_score_tensors, accumulation_steps,
                          device, grad_clip, base_lr, global_micro_step,
                          total_micro_steps, get_lr, timers, hvp_fp32=False,
                          full_diagnostics=True, collect=None):
    """Mirror of _a0h1._run_joint_pass PLUS per-step tangent/cotangent storage and
    in-loop CHK-LIN/CHK-SA/CHK-IVJP probes. The forward segments are line-identical
    to the frozen joint pass (same call order, same RNG discipline); deviations would
    be caught by the frozen-vs-joint determinism check in A0H-1 (already PASS).

    full_diagnostics=False (acceptance mode) skips the CHK-LIN/CHK-SA/CHK-IVJP
    probes and keeps only the shared-adjoint walk + telescope bookkeeping.

    collect (optional dict, bookkeeping mode): when passed, the pass additionally
    records RAW per-step vectors into it WITHOUT altering any computation:
      collect["x_all"][(h, j)]  = pre-step joint tangent triple at every step h
      collect["p_store"][H][k]  = adjoint-input triple at step k of the H-walk
      collect["w_0"][H], collect["a_by_j"][j], collect["v_score_grad"][h],
      collect["r_H_C"][h][j]    = post-step theta tangent endpoint
    All values are detached clones; FP64 pairing happens in the collector."""
    hvp_ctx = nullcontext() if hvp_fp32 else autocast_ctx
    load_full_training_state(checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))

    zero = _zeros_like(named)
    c_theta = {j: _clone_map(zero) for j in cand_batches_by_j}
    c_m = {j: _clone_map(zero) for j in cand_batches_by_j}
    c_v = {j: _clone_map(zero) for j in cand_batches_by_j}

    max_h = max(a0g.HORIZONS)
    g_by_j = {}
    a_by_j = {}
    v_score_grad = {}
    r_H_C = {}
    snapshots = {}
    hvp_dir_by_j = {}
    x_by_j_h = {}   # (h, j) -> joint tangent INPUT at step h (pre-step; pairs with p_store[h])
    clip_trace = []

    pg0 = optimizer.param_groups[0]
    hyper = {
        "weight_decay": float(pg0["weight_decay"]),
        "beta1": float(pg0["betas"][0]),
        "beta2": float(pg0["betas"][1]),
        "adam_eps": float(pg0["eps"]),
    }

    diag = {"lin": [], "sa": [], "ivjp": [], "anchors": []}
    hvp_by_j = {}  # per-horizon {j: HVP mapping}; reset each horizon step

    # Nested-AD pre-check runs on the FIRST future step (before any in-place
    # optimizer write of that step), mirroring contract §2 ordering discipline.
    _precheck_done = False

    for horizon in range(1, max_h + 1):
        theta_start = {k: v.detach().clone() for k, v in raw_model(model).state_dict().items()}
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = set_native_micro_lr(
                optimizer, get_lr, int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr))
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                rng_before_native_forward = capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        native_gradient = {name: p.grad.detach().clone() for name, p in named}

        if horizon == 1:
            for j, cand_batches in cand_batches_by_j.items():
                cand_tensors = a0g._stack_all(dataset, cand_batches, device)
                rng_snap = capture_rng_state()
                g_by_j[j] = a0g._grad_on_batches(
                    torch, model, named, cand_tensors, autocast_ctx, accumulation_steps)
                restore_rng_state(rng_snap)
                del cand_tensors
            for j in cand_batches_by_j:
                a_by_j[j] = {name: g_by_j[j][name] - native_gradient[name] for name, _ in named}

        native_preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(grad_clip)).detach().cpu())
        native_clipped_gradient = {name: p.grad.detach().clone() for name, p in named}

        hvp_dir_by_j[horizon] = {}
        if horizon > 1:
            # ---- IVJP nested-AD pre-check (once): third-order chain viability ----
            if full_diagnostics and not _precheck_done:
                _precheck_done = True
                probe = _clone_map(c_theta[1])
                probe_leaf = {name: v.detach().clone().requires_grad_(True)
                              for name, v in probe.items()}
                y_c = {name: torch.randn(p.shape, generator=torch.Generator("cpu").manual_seed(1234)).to(device=p.device, dtype=p.dtype)
                       for name, p in probe.items()}
                restore_rng_state(rng_before_native_forward)
                raw_model(model).load_state_dict(theta_start)
                try:
                    kx_probe, vjp_probe = _hvp_direction_vjp(
                        torch, model, named, batch_tensors, probe_leaf, y_c, hvp_ctx,
                        accumulation_steps)
                    ref_probe, _ = a0g._hvp(torch, model, named, batch_tensors,
                                            probe, hvp_ctx, accumulation_steps)
                    diag["ivjp_precheck"] = {
                        "ok": True,
                        "wrapper_vs_frozen_gap": _norm64(_sub(kx_probe, ref_probe)),
                    }
                except RuntimeError as exc:
                    diag["ivjp_precheck"] = {"ok": False, "error": str(exc)[:400]}
                restore_rng_state(rng_before_native_forward)

            # ---- CHK-LIN/SA/IVJP probes (skipped in acceptance mode) ----
            rng_now = capture_rng_state()
            if full_diagnostics:
                x_dir = _clone_map(c_theta[1])
                restore_rng_state(rng_before_native_forward)
                raw_model(model).load_state_dict(theta_start)

                # REP: >=3 repeats of the same direction
                kx_reps = []
                for _ in range(REPEATS):
                    restore_rng_state(rng_before_native_forward)
                    kx_r, _ = a0g._hvp(torch, model, named, batch_tensors, x_dir, hvp_ctx,
                                       accumulation_steps)
                    kx_reps.append(_clone_map(kx_r))
                kx = kx_reps[0]
                rep_gaps = [_norm64(_sub(kx_r, kx)) for kx_r in kx_reps[1:]]
                rep_mean = sum(rep_gaps) / max(len(rep_gaps), 1)
                rep_var = sum((g - rep_mean) ** 2 for g in rep_gaps) / max(len(rep_gaps), 1)

                # K(0)
                restore_rng_state(rng_before_native_forward)
                k0, _ = a0g._hvp(torch, model, named, batch_tensors, zero, hvp_ctx,
                                 accumulation_steps)
                # K(-x)
                restore_rng_state(rng_before_native_forward)
                k_neg, _ = a0g._hvp(torch, model, named, batch_tensors, _scale(x_dir, -1.0),
                                    hvp_ctx, accumulation_steps)
                # K(alpha x), alpha in {0.5, 2, 1.5} (pre-fixed in contract)
                lin_alpha = {}
                for alpha in (0.5, 2.0, 1.5):
                    restore_rng_state(rng_before_native_forward)
                    ka, _ = a0g._hvp(torch, model, named, batch_tensors, _scale(x_dir, alpha),
                                     hvp_ctx, accumulation_steps)
                    lin_alpha[str(alpha)] = _norm64(_sub(ka, _scale(kx, alpha)))
                # K(x+y) - K(x) - K(y), y = orthogonal-ish perturbation of x
                g_seed = (int(stream_seed) + horizon * 97) % (2 ** 31)
                gen = torch.Generator(device="cpu").manual_seed(g_seed)
                y_dir = {name: torch.randn(p.shape, generator=gen).to(device=p.device, dtype=p.dtype)
                         for name, p in x_dir.items()}
                restore_rng_state(rng_before_native_forward)
                kxy, _ = a0g._hvp(torch, model, named, batch_tensors, _add(x_dir, y_dir),
                                  hvp_ctx, accumulation_steps)
                restore_rng_state(rng_before_native_forward)
                ky, _ = a0g._hvp(torch, model, named, batch_tensors, y_dir, hvp_ctx,
                                 accumulation_steps)
                lin_add = _norm64(_sub(kxy, _add(kx, ky)))
                lin_neg = _norm64(_add(k_neg, kx))
                k0_rel = _norm64(k0) / max(_norm64(kx), 1e-30)
                lin_alpha_rel = {a: v / max(_norm64(_scale(kx, float(a))), 1e-30)
                                 for a, v in lin_alpha.items()}
                lin_add_rel = lin_add / max(_norm64(_add(kx, ky)), 1e-30)
                lin_neg_rel = lin_neg / max(_norm64(kx), 1e-30)
                diag["lin"].append({
                    "horizon": horizon,
                    "rep_abs_gaps": rep_gaps,
                    "rep_mean_abs_gap": rep_mean,
                    "rep_var": rep_var,
                    "k0_norm": _norm64(k0),
                    "k0_rel_to_kx": k0_rel,
                    "lin_neg_rel": lin_neg_rel,
                    "lin_alpha_rel": lin_alpha_rel,
                    "lin_add_rel": lin_add_rel,
                })

            # ---- per-candidate HVP (frozen forward discipline; ALWAYS needed) ----
            hvp_by_j = {}
            for j in cand_batches_by_j:
                hvp_dir_by_j[horizon][j] = _clone_map(c_theta[j])
                rng_after = capture_rng_state()
                restore_rng_state(rng_before_native_forward)
                hvp_j, _ = a0g._hvp(torch, model, named, batch_tensors, c_theta[j],
                                    hvp_ctx, accumulation_steps)
                restore_rng_state(rng_after)
                hvp_by_j[j] = hvp_j

            # ---- CHK-SA + CHK-IVJP at multiple anchors (diagnostics only) ----
            if full_diagnostics:
                gen_y = torch.Generator(device="cpu").manual_seed(
                    (int(stream_seed) + horizon * 97 + 1) % (2 ** 31))
                y_probe = {name: torch.randn(p.shape, generator=gen_y).to(
                    device=p.device, dtype=p.dtype) for name, p in c_theta[1].items()}
                wrapper_check = None
                for j in (1,) + tuple(cand_batches_by_j.keys()):
                    anchor_dir = hvp_dir_by_j[horizon][j]
                    restore_rng_state(rng_before_native_forward)
                    raw_model(model).load_state_dict(theta_start)
                    k_anchor, _ = a0g._hvp(torch, model, named, batch_tensors,
                                           anchor_dir, hvp_ctx, accumulation_steps)
                    restore_rng_state(rng_before_native_forward)
                    k_anchor_y, _ = a0g._hvp(torch, model, named, batch_tensors,
                                             y_probe, hvp_ctx, accumulation_steps)
                    lhs = _inner64(anchor_dir, k_anchor_y)
                    rhs = _inner64(k_anchor, y_probe)
                    sa_abs = abs(lhs - rhs)
                    sa_rel = sa_abs / max(abs(lhs), abs(rhs), 1e-30)
                    diag["sa"].append({
                        "horizon": horizon, "anchor_j": j,
                        "lhs": lhs, "rhs": rhs, "abs_gap": sa_abs, "rel": sa_rel,
                    })
                    dir_leaf = {name: v.detach().clone().requires_grad_(True)
                                for name, v in anchor_dir.items()}
                    ivjp_lhs = _inner64(k_anchor, y_probe)
                    try:
                        restore_rng_state(rng_before_native_forward)
                        raw_model(model).load_state_dict(theta_start)
                        kx_graph, ivjp_map = _hvp_direction_vjp(
                            torch, model, named, batch_tensors, dir_leaf, y_probe,
                            hvp_ctx, accumulation_steps)
                        if wrapper_check is None:
                            ref_check, _ = a0g._hvp(
                                torch, model, named, batch_tensors, anchor_dir,
                                hvp_ctx, accumulation_steps)
                            wrapper_check = _norm64(_sub(kx_graph, ref_check))
                        ivjp_rhs = _inner64(anchor_dir, ivjp_map)
                        ivjp_abs = abs(ivjp_lhs - ivjp_rhs)
                        ivjp_rel = ivjp_abs / max(abs(ivjp_lhs), abs(ivjp_rhs), 1e-30)
                        diag["ivjp"].append({
                            "horizon": horizon, "anchor_j": j,
                            "lhs": ivjp_lhs, "rhs": ivjp_rhs, "abs_gap": ivjp_abs,
                            "rel": ivjp_rel, "ivjp_map_norm": _norm64(ivjp_map),
                            "status": "OK",
                        })
                    except RuntimeError as exc:
                        diag["ivjp"].append({
                            "horizon": horizon, "anchor_j": j,
                            "lhs": ivjp_lhs, "abs_gap": None, "rel": None,
                            "status": "NOT_SUPPORTED",
                            "error": str(exc)[:300],
                        })
                if wrapper_check is not None:
                    diag["lin"].append({"horizon": horizon,
                                        "wrapper_vs_frozen_gap": wrapper_check})
            restore_rng_state(rng_now)

        # ---- tangent steps (frozen forward) ----
        for j in cand_batches_by_j:
            # x_k = joint tangent INPUT at step k (pre-step), matching p_store[k]
            # (adjoint input at step k). Step k advances z_k -> z_{k+1}.
            if horizon in a0g.HORIZONS:
                x_by_j_h[(horizon, j)] = (
                    _clone_map(c_theta[j]), _clone_map(c_m[j]), _clone_map(c_v[j]))
            if collect is not None:
                collect.setdefault("x_all", {})[(horizon, j)] = (
                    _clone_map(c_theta[j]), _clone_map(c_m[j]), _clone_map(c_v[j]))
            tC = a_by_j[j] if horizon == 1 else hvp_by_j[j]
            c_clip = clip_jvp(
                gradient_by_name=native_gradient, tangent_gradient_by_name=tC,
                max_norm=float(grad_clip))
            c_next = adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
                tangent_theta_by_name=c_theta[j],
                tangent_m_by_name=c_m[j],
                tangent_v_by_name=c_v[j])
            c_theta[j], c_m[j], c_v[j] = (
                c_next["tangent_theta"], c_next["tangent_m"], c_next["tangent_v"])
            if horizon == 1 and j == 1:
                step_after_1 = float(c_next["optimizer_step_after"])

        exp_avg = {p: optimizer.state[p]["exp_avg"].detach().clone() for _, p in named}
        exp_avg_sq = {p: optimizer.state[p]["exp_avg_sq"].detach().clone() for _, p in named}
        snapshots[horizon] = {
            "exp_avg": {name: exp_avg[p] for name, p in named},
            "exp_avg_sq": {name: exp_avg_sq[p] for name, p in named},
            "clipped_gradient": _clone_map(native_clipped_gradient),
            "native_gradient": _clone_map(native_gradient),
            "preclip_norm": float(native_preclip),
            "lr": float(current_lr),
            "step_after": float(step_after_1) if horizon == 1 else float(
                snapshots[horizon - 1]["step_after"]) + 1.0,
            "batch_tensors": batch_tensors,
            "rng_before_native_forward": rng_before_native_forward,
            "theta_start": theta_start,
            **hyper,
        }
        clip_trace.append({
            "horizon": horizon,
            "clip_active": bool(native_preclip > float(grad_clip)),
            "preclip_norm": float(native_preclip),
            "lr": float(current_lr),
        })

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if horizon in a0g.HORIZONS:
            rng_snap = capture_rng_state()
            v_score_grad[str(horizon)] = a0g._grad_on_batches(
                torch, model, named, val_score_tensors, autocast_ctx,
                len(val_score_tensors))
            restore_rng_state(rng_snap)
            # post-step theta tangent at this horizon (endpoint for hhat_fwd_C)
            r_H_C[str(horizon)] = {j: _clone_map(c_theta[j]) for j in cand_batches_by_j}
            if collect is not None:
                collect.setdefault("v_score_grad", {})[str(horizon)] = _clone_map(
                    v_score_grad[str(horizon)])
                collect.setdefault("r_H_C", {})[str(horizon)] = {
                    str(j): _clone_map(c_theta[j]) for j in cand_batches_by_j}

    # ---- adjoint walk per H with telescope bookkeeping (reusing A0H-1 walk logic) ----
    from fr.a0h_adjoint_v01 import adamw_tangent_step_adjoint

    def clip_transpose(b_map, snap):
        return _a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_map,
            grad_clip=grad_clip, preclip_norm=snap["preclip_norm"])

    tele = {}
    vjp_shared = {}
    for horizon in a0g.HORIZONS:
        p_th = _clone_map(v_score_grad[str(horizon)])
        p_m = _clone_map(zero)
        p_v = _clone_map(zero)
        # Single reverse walk with cotangent storage: p_store[k] is the adjoint
        # INPUT at step k (p_store[H] = ell_H; after step k, p feeds step k-1).
        p_store = {horizon: (_clone_map(p_th), _clone_map(p_m), _clone_map(p_v))}
        w_by_k = {}
        b_last = None
        snap1 = snapshots[1]
        for k in range(horizon, 0, -1):
            snap = snapshots[k]
            adj = adamw_tangent_step_adjoint(
                named_parameters=named, optimizer=optimizer, snapshot=snap,
                cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
                cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
            b_k = adj["cotangent_dc"]
            w_k = clip_transpose(b_k, snap)
            if k >= 2:
                w_by_k[k] = _clone_map(w_k)
            p_th = adj["cotangent_theta"]
            p_m = adj["cotangent_m"]
            p_v = adj["cotangent_v"]
            if k >= 2:
                rng_now = capture_rng_state()
                restore_rng_state(snap["rng_before_native_forward"])
                raw_model(model).load_state_dict(snap["theta_start"])
                hvp_u, _ = a0g._hvp(torch, model, named, snap["batch_tensors"], w_k,
                                    hvp_ctx, accumulation_steps)
                restore_rng_state(rng_now)
                p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
                p_store[k - 1] = (_clone_map(p_th), _clone_map(p_m), _clone_map(p_v))
            b_last = b_k
        w_0 = clip_transpose(b_last, snap1)
        hhat_adj = {j: project_tangent(w_0, a_by_j[j]) for j in cand_batches_by_j}
        if collect is not None:
            collect.setdefault("p_store", {})[str(horizon)] = {
                k: (_clone_map(t[0]), _clone_map(t[1]), _clone_map(t[2]))
                for k, t in p_store.items()}
            collect.setdefault("w_0", {})[str(horizon)] = _clone_map(w_0)
            collect.setdefault("a_by_j", {})[str(horizon)] = {
                str(j): _clone_map(a_by_j[j]) for j in cand_batches_by_j}

        # ---- CHK-TELE: telescope closure per candidate ----
        # T_j = (p1.x1 - w0.a_j) + sum_{k=1}^{H-1} (p_{k+1}.x_{k+1} - p_k.x_k)
        # (single walk already stored p_store; x_k stored post-step during forward)
        per_j = {}
        for j in cand_batches_by_j:
            terms = []
            # k=1 term uses p_1 (input to step 1 adjoint = v_H back-propagated to k=1 input)
            x1 = x_by_j_h.get((1, j))
            p1 = p_store[1]
            if x1 is not None:
                t1 = (_inner64(p1[0], x1[0]) + _inner64(p1[1], x1[1])
                      + _inner64(p1[2], x1[2]) - _inner64(w_0, a_by_j[j]))
                terms.append(t1)
            for k in range(1, horizon):
                xk1 = x_by_j_h.get((k + 1, j))
                xk = x_by_j_h.get((k, j))
                pk1 = p_store[k + 1]
                pk = p_store[k]
                if xk1 is None or xk is None:
                    continue
                terms.append(
                    _inner64(pk1[0], xk1[0]) + _inner64(pk1[1], xk1[1])
                    + _inner64(pk1[2], xk1[2])
                    - _inner64(pk[0], xk[0])
                    - _inner64(pk[1], xk[1])
                    - _inner64(pk[2], xk[2]))
            max_abs = max(abs(t) for t in terms) if terms else 0.0
            scale = max(abs(t1), 1e-30)
            per_j[j] = {
                "terms": terms,
                "max_abs_term": max_abs,
                "rel_to_first": max_abs / scale,
                "pass_tele": bool(max_abs <= ATOL_TELE + RTOL_TELE * scale),
            }
        tele[str(horizon)] = per_j
        vjp_shared[str(horizon)] = {
            "hhat_adj": {str(j): hhat_adj[j] for j in hhat_adj},
            "w_0_norm": _norm64(w_0),
        }

    # ---- frozen forward scores hhat_fwd_C[j][h] = <v_h, theta tangent at h> ----
    hhat_fwd_C = {}
    for horizon in a0g.HORIZONS:
        hk = str(horizon)
        v_h = v_score_grad[hk]
        per_j = {}
        for j in cand_batches_by_j:
            per_j[j] = _inner64(v_h, r_H_C[hk][j])
        hhat_fwd_C[hk] = per_j

    return {
        "v_score_grad": {k: _clone_map(v) for k, v in v_score_grad.items()},
        "a_by_j": {j: _clone_map(a_by_j[j]) for j in a_by_j},
        "snapshots": snapshots,
        "tele": tele,
        "diag": diag,
        "vjp_shared": vjp_shared,
        "clip_trace": clip_trace,
        "hhat_fwd_C": hhat_fwd_C,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--states", default="",
                    help="comma list like '2031/52,2034/64'; default = all 3 dev states")
    ap.add_argument("--budget-seconds", type=float, default=1200.0)
    ap.add_argument("--hvp-fp32", action="store_true",
                    help="Branch B1: run the 7 _hvp call points under nullcontext "
                         "(FP32-HVP different-config candidate)")
    ap.add_argument("--lean", action="store_true",
                    help="full_diagnostics=False (skip CHK-LIN/SA/IVJP probes)")
    args = ap.parse_args()

    wall_start = time.time()
    root = Path(args.project_root).resolve()
    out_dir = root / OUTPUT_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "research/configs/e3c_confirmatory_v07.json").read_text())
    _require(cfg.get("dtype") == "bfloat16" and int(cfg.get("accumulation_steps")) == 2
             and float(cfg.get("grad_clip")) == 1.0, "config drift")
    v07_out = root / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"
    a0g_out = root / a0g.OUTPUT_DIR
    data_group_definition = json.loads(
        (a0g_out / "data_group_definition.json").read_text(encoding="utf-8"))
    validation_split = json.loads(
        (a0g_out / "validation_split.json").read_text(encoding="utf-8"))

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False)
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device)
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + a0g.EVAL_BATCH_SIZE] for i in range(0, len(idx), a0g.EVAL_BATCH_SIZE)]

    val_score_tensors = a0g._stack_all(dataset, _chunk(validation_split["v_score"]), args.device)

    timers = a0g._Timers(torch)
    dev_ctxs = a0g._load_v07_state_contexts(root, cfg, partition, data_group_definition)
    if args.states:
        wanted = set(tuple(s.split("/")) for s in args.states.split(","))
        dev_ctxs = [c for c in dev_ctxs if (str(c.seed), str(c.step)) in wanted]

    for ctx in dev_ctxs:
        elapsed = time.time() - wall_start
        if elapsed > args.budget_seconds:
            print("BUDGET_EXHAUSTED; stopping before next unit", flush=True)
            break
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        state_label = f"dev:{ctx.seed}/{ctx.step}"
        print(f"[{state_label}] diagnostics pass ...", flush=True)
        t0 = time.perf_counter()
        v_t = a0g._compute_v_t(torch, model, optimizer, scaler, ctx, val_score_tensors,
                               autocast_ctx, named)
        cand = ctx.candidate_batches(stream["stream_seed"])
        res = _run_diagnostics_pass(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            checkpoint_path=ctx.checkpoint_path,
            batches=stream["batches"], stream_seed=stream["stream_seed"],
            cand_batches_by_j=cand, val_score_tensors=val_score_tensors,
            accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=get_lr, timers=timers, hvp_fp32=bool(args.hvp_fp32),
            full_diagnostics=not bool(args.lean))
        wall = time.perf_counter() - t0

        # strip heavy fields before JSON
        report = {
            "state": state_label,
            "stream_id": 0,
            "wall_seconds": wall,
            "budget_remaining": args.budget_seconds - (time.time() - wall_start),
            "tele": res["tele"],
            "diag": res["diag"],
            "vjp_shared": res["vjp_shared"],
            "hhat_fwd_C": res["hhat_fwd_C"],
            "clip_trace": res["clip_trace"],
            "hvp_fp32": bool(args.hvp_fp32),
            "tolerances": {"atol_tele": ATOL_TELE, "rtol_tele": RTOL_TELE,
                           "lin_rel_tol": LIN_REL_TOL},
        }
        suffix = "_fp32hvp" if args.hvp_fp32 else ""
        out_path = out_dir / f"diag_{ctx.seed}_{ctx.step}{suffix}_s0.json"
        out_path.write_text(json.dumps(report, indent=1, default=float))
        print(f"[{state_label}] wrote {out_path.name} ({wall:.1f}s)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
