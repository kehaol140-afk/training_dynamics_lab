"""A0Q response-control map (core module).

Builds the training-dynamics response matrix J = [h_lr, h_clip, h_wd] over
three optimizer/external control inputs, all as PERSISTENT multiplicative
log-perturbations over the A0O-validated H=4 window (u applied at EVERY
optimizer step, delta = 0.05 frozen, imported from a0o):

    lr:   u = log(eta/eta0)          [A0O-validated channel, physics reused]
    clip: u = log(tau/tau0), tau0=1.0 (frozen cfg grad_clip)
    wd:   u = log(lambda/lambda0), lambda0 = 0.01 (native AdamW default in
          the v0.2 bank state)

Analytic sources (first-order, exact tangent of the frozen system):

  LR  (A0O amendment-A2, pure theta-channel push):
      h_lr = sum_k <p_th^{(k)}, lr_k * e_lr(k)>,
      e_lr = -adaptive(c_k) - wd*theta_k   (frozen a0o.lr_pulse, import-only)

  WD  (pure theta channel; wd does not touch m/v in AdamW):
      theta_{k+1} = (1 - lr*wd(u))*theta_k - lr*adaptive(c_k)
      d(theta_{k+1})/du = -lr_k * lambda0 * theta_k     [d wd/du = lambda0]

  CLIP (dc channel; clipped = g*min(1, tau/||g||)):
      d(clipped_k)/d tau = ghat_k := g_k/||g_k||  when preclip_norm_k > tau0
                           0 otherwise ;   d tau/du = tau0
      h_clip = sum_k tau0 * <b_k, ghat_k> * 1[preclip_k > tau0]
      where b_k = cotangent_dc ENTERING the frozen clip transpose at iter k.

All three h_i are accumulated in ONE shared forward pass + ONE rewind through
the frozen A0H15 reverse adjoint (the rewind is verbatim A0O machinery; the
only new code is the per-channel source contractions above).

FD reference per channel: (L(+delta) - L(-delta)) / (2*delta), L = V_score at
theta_{t+H}(u) with the channel's scalar perturbed at every step; u=0 shared
across channels (run twice by the driver, CRN bitwise check).
"""
from __future__ import annotations

from contextlib import nullcontext

import torch as _torch

from .a0o_lr_response_v01 import DELTA, lr_pulse  # frozen A0O: import-only
from .minimind_v03_runtime import raw_model
from .rng import capture_rng_state, restore_rng_state

CHANNELS = ("lr", "clip", "wd")
TAU0 = 1.0  # frozen cfg grad_clip (runtime asserts grad_clip == 1.0)


def run_response_map(rt, ctx, stream, H: int, hvp_fp32: bool, counters):
    """One shared forward pass; rewind once; return all three channel h_i."""
    import a0g_decision_utility_v01 as a0g
    from a0h15_production_entry_v01 import (_a0h1, _baseline_no_step, _clip_only,
                                            _clone_map, _endpoint_grad,
                                            _restore_and_seed, _step_once,
                                            _zeros_like)
    from .a0h_adjoint_v01 import adamw_tangent_step_adjoint

    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    hvp_ctx = nullcontext() if hvp_fp32 else rt["autocast_ctx"]
    assert hvp_fp32, "A0Q production candidate is fp32-hvp only"
    assert float(cfg["grad_clip"]) == TAU0, "clip channel assumes frozen tau0"

    _restore_and_seed(rt, ctx, stream)
    snapshots = {}
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = _baseline_no_step(rt, ctx, stream["batches"],
                                                     horizon, want_snapshot=True)
        counters["baseline_steps"] += 1
        snapshots[horizon] = snap
        _clip_only(rt, native_gradient, snap)
        _step_once(rt)
    v_H = _endpoint_grad(rt, counters)

    p_th = _clone_map(v_H)
    p_m = _clone_map(_zeros_like(named))
    p_v = _clone_map(_zeros_like(named))
    h = {"lr": 0.0, "clip": 0.0, "wd": 0.0}
    per_step = []
    H = int(H)
    for k in range(H, 0, -1):
        snap = snapshots[k]
        # --- sources contracted with p_th ENTERING reverse iter k ---
        e = lr_pulse(snap)
        lr_k = float(snap["lr"])
        lam0 = float(snap["weight_decay"])
        th = snap["theta_start"]
        h["lr"] += sum(float((p_th[n] * (lr_k * e[n])).sum()) for n, _ in named)
        h["wd"] += sum(float((p_th[n] * (-lr_k * lam0 * th[n])).sum())
                       for n, _ in named)
        # --- rewind step k (verbatim A0O inner loop) ---
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=snap,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        counters["reverse_adjoint_steps"] += 1
        b_k = adj["cotangent_dc"]
        preclip = float(snap["preclip_norm"])
        if preclip > TAU0:  # clip active at step k
            ghat = {n: snap["native_gradient"][n] / preclip for n, _ in named}
            h["clip"] += TAU0 * sum(float((b_k[n] * ghat[n]).sum())
                                    for n, _ in named)
        w_k = _a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_k,
            grad_clip=float(cfg["grad_clip"]), preclip_norm=preclip)
        counters["clip_transpose_calls"] += 1
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
        per_step.append({"k": k, "lr": lr_k, "wd": lam0, "preclip_norm": preclip,
                         "clip_active": bool(preclip > TAU0)})
    per_step.reverse()
    return {"h_lr": float(h["lr"]), "h_clip": float(h["clip"]),
            "h_wd": float(h["wd"]), "v_H_norm": float(a0g._norm(v_H)),
            "per_step": per_step, "mode": "control_map_adjoint", "H": H,
            "counters": counters}


def run_channel_rollout(rt, ctx, stream, H: int, channel: str, u: float,
                        counters) -> float:
    """FD branch: H native steps with ONE channel's scalar perturbed at every
    step (persistent, A0O semantics). u == 0 never touches anything -> bitwise
    baseline (identical for all channels; driver runs it twice for CRN).

    lr:   g['lr'] = lr_k * e^u each step (native lr rewrites per micro-step)
    wd:   g['weight_decay'] = lambda0 * e^u once (persists across steps)
    clip: clip_grad_norm_(..., tau0 * e^u) at each step
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe
    torch = rt["torch"]
    cfg = rt["cfg"]

    pe._restore_and_seed(rt, ctx, stream)
    lam0 = float(rt["optimizer"].param_groups[0]["weight_decay"])
    if channel == "wd" and u != 0.0:
        for g in rt["optimizer"].param_groups:
            g["weight_decay"] = lam0 * (_torch.e ** u)
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        counters["baseline_steps"] += 1
        if channel == "lr" and u != 0.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * (_torch.e ** u)
        if channel == "clip" and u != 0.0:
            torch.nn.utils.clip_grad_norm_(rt["model"].parameters(),
                                           TAU0 * (_torch.e ** u))
            pe._step_once(rt)
        else:
            pe._clip_and_step(rt, native_gradient, snap)
    rng_now = capture_rng_state()
    val = float(a0g._val_loss_only(rt["torch"], rt["model"],
                                   rt["val_score_tensors"], rt["autocast_ctx"]))
    restore_rng_state(rng_now)
    return val
