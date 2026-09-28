"""A0P learning-rate response temporal kernel validation (core module).

Extends the A0O-validated optimizer-input response from a PERSISTENT schedule
perturbation to an IMPULSE kernel with the frozen log-LR channel

    eta_1(u) = eta_1 * exp(u) at optimizer step t ONLY (u = log(eta/eta0));
    steps t+1..t+64 run the frozen native AdamW schedule (set_native_micro_lr
    rewrites lr every micro-step, so the pulse reverts natively at step 2).

Impulse response kernel (mandate definition):

    R(tau) = d L(t+tau) / du |_{u=0},   tau in {1,2,4,8,16,32,64}

Analytic (A0O physics, source re-placed): the pulse at step 1 is the pure
theta-channel push d(theta_2)/du = lr_1 * e_lr(1) with
e_lr = -adaptive(c_1) - wd*theta_1 (m1/v1 lr-independent; Amendment-A2
theory of A0O, reused via the frozen `lr_pulse`). Under linearity of the
frozen tangent system,

    R(tau) = < p_th ENTERING reverse iter 1 , lr_1 * e_lr(1) >

where p_th is the endpoint cotangent dL/d(theta_{t+tau}) rewound through the
A0H15 production reverse adjoint (adamw_tangent_step_adjoint + clip transpose
+ fp32 future HVP at k>=2) over steps k=tau..2. The ONLY delta vs the A0O
production loop is source placement: A0O contracts e_lr at every k (persistent
input); A0P contracts once, at k=1 (impulse). The rewind itself is verbatim
A0O/A0H15 machinery.

FD reference per grid point:
    R_fd(tau) = ( L(tau; u=+delta) - L(tau; u=-delta) ) / (2*delta)
with L evaluated along a single 64-step rollout per branch (val loss at every
grid checkpoint, RNG-wrapped so the training trajectory is CRN-identical to
the u=0 baseline; u=0 run twice for a bitwise CRN check).

delta = 0.05 imported from the frozen A0O module (never redefined here).
"""
from __future__ import annotations

from contextlib import nullcontext

import torch as _torch

from .a0h_adjoint_v01 import adamw_tangent_step_adjoint
from .a0o_lr_response_v01 import DELTA, lr_pulse  # frozen A0O: import-only
from .minimind_v03_runtime import raw_model
from .rng import capture_rng_state, restore_rng_state

TAU_GRID = (1, 2, 4, 8, 16, 32, 64)  # frozen in contract (no result-driven selection)
TAU_MAX = max(TAU_GRID)


def run_lr_impulse_response(rt, ctx, stream, tau: int, hvp_fp32: bool, counters):
    """Analytic impulse kernel R(tau) on a tau-step shared (u=0) forward pass.

    Mirrors A0O run_lr_response exactly except the source contraction is
    restricted to k=1 (impulse, not persistent). State left MUTATED (caller
    restores). Returns {"R": float, "v_H_norm": float, ...}.
    """
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
    assert hvp_fp32, "A0P production candidate is fp32-hvp only"

    _restore_and_seed(rt, ctx, stream)
    snapshots = {}
    for horizon in range(1, int(tau) + 1):
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
    # Rewind steps k=tau..2 (verbatim A0O inner loop, source terms removed):
    # at every k>=2 the fp32 future HVP carries gradient-frame feedback of the
    # post-pulse trajectory, exactly as in the A0O-validated persistent case.
    for k in range(int(tau), 1, -1):
        snap = snapshots[k]
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=snap,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        counters["reverse_adjoint_steps"] += 1
        b_k = adj["cotangent_dc"]
        w_k = _a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_k,
            grad_clip=float(cfg["grad_clip"]), preclip_norm=snap["preclip_norm"])
        counters["clip_transpose_calls"] += 1
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        rng_now = capture_rng_state()
        restore_rng_state(snap["rng_before_native_forward"])
        raw_model(rt["model"]).load_state_dict(snap["theta_start"])
        hvp_u, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"],
                            w_k, hvp_ctx, int(cfg["accumulation_steps"]))
        counters["hvp_calls"] += 1
        restore_rng_state(rng_now)
        p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
        del snapshots[k]  # free as we consume (peak memory = full forward pass)

    # Impulse contraction at k=1: p_th now enters reverse iter 1 (= dL/d(theta_2),
    # post-step-1 state). Source: pure theta-channel push at the pulse step.
    e = lr_pulse(snapshots[1])
    lr_1 = float(snapshots[1]["lr"])
    R = sum(float((p_th[n] * (lr_1 * e[n])).sum()) for n, _ in named)
    return {"R": float(R), "v_H_norm": float(a0g._norm(v_H)),
            "mode": "lr_impulse_adjoint", "tau": int(tau), "counters": counters}


def run_lr_impulse_rollout(rt, ctx, stream, u: float, grid, counters) -> dict:
    """FD branch: 64 native steps with the LR impulse at step 1 ONLY.

    u == 0 never touches param_group lr (bitwise baseline; also run twice by
    the driver for the CRN check). For u != 0 the lr is scaled once, before
    the step-1 clip_and_step; set_native_micro_lr rewrites lr at every later
    micro-step so the pulse reverts natively from step 2 on. Returns
    {tau: val_loss} for every tau in grid.
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe

    pe._restore_and_seed(rt, ctx, stream)
    losses = {}
    for horizon in range(1, TAU_MAX + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=(horizon == 1))
        counters["baseline_steps"] += 1
        if horizon == 1 and u != 0.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * (_torch.e ** u)
        pe._clip_and_step(rt, native_gradient, snap)
        if horizon in grid:
            rng_now = capture_rng_state()
            losses[int(horizon)] = float(
                a0g._val_loss_only(rt["torch"], rt["model"],
                                   rt["val_score_tensors"], rt["autocast_ctx"]))
            restore_rng_state(rng_now)
    return losses
