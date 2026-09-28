"""A0O learning-rate input response validation (core module).

Validates whether the training-dynamics response framework extends from DATA
perturbation (A0G-A0K) to an OPTIMIZER-INPUT perturbation channel: the
multiplicative log-LR of the frozen AdamW schedule,

    eta_k(u) = eta_k * exp(u),   u in {-delta, 0, +delta},  delta = 0.05 frozen.

Theory (decoupled weight decay included, per mandate):

    theta+ = d*theta - lr*adaptive(c) ,  d = 1 - lr*wd
    d theta+ / du = (d theta+ / d lr) * lr      [chain rule: d lr/du = lr at u=0]

  lr-channel tangent:  e_lr = -adaptive(c) - wd*theta
  u-channel tangent:   e_u  = lr * e_lr

Analytic response:  h_eta = sum_k <p_th^{(k)}, e_u(k)>
  where p_th^{(k)} is the theta-cotangent ENTERING reverse iteration k
  (= dL/d(theta_{k+1}), post-step-k state) produced by the frozen reverse
  adjoint (adamw_tangent_step_adjoint + clipping transpose + fp32 future HVP,
  identical to A0H15 production). The LR pulse is a PURE THETA-CHANNEL push:
  m1/v1 are lr-independent (the consumed gradient c_k does not depend on u,
  so dm1 = dv1 = 0 at the pulse step); all gradient/m/v-channel effects of
  LATER steps are carried by the frozen adjoint chain. This is exact in u
  to first order (tangent); O(u^2) terms are excluded by the FD reference.

FD reference:  (L(+delta) - L(-delta)) / (2*delta),  L = V_score at endpoint
theta_{t+H}(u), u applied at EVERY optimizer step k=1..H (the LR is a
persistent schedule input, not a one-shot pulse).

RNG hygiene: the branch interposes no RNG consumers; it is CRN-identical to
the u=0 baseline on the shared H-step forward pass.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch as _torch

from .a0c_adamw_response_propagation_v01 import project_tangent
from .a0h_adjoint_v01 import adamw_tangent_step_adjoint
from .minimind_v03_runtime import raw_model
from .rng import capture_rng_state, restore_rng_state

DELTA = 0.05  # frozen before any GPU: u = log(eta/eta0) perturbation amplitude


def lr_pulse(snap) -> dict:
    """Gradient-frame shift s_k = u * e_lr(k); returns the UNIT e_lr map.

    e_lr = -adaptive(c) - wd*theta  (decoupled weight decay included).
    """
    scalars = {k: float(snap[k]) for k in
               ("lr", "weight_decay", "beta1", "beta2", "adam_eps", "step_after")}
    lr, wd, b1, b2, eps = (scalars["lr"], scalars["weight_decay"],
                           scalars["beta1"], scalars["beta2"], scalars["adam_eps"])
    bc1 = 1.0 - b1 ** scalars["step_after"]
    bc2 = 1.0 - b2 ** scalars["step_after"]
    out = {}
    for name, c in snap["clipped_gradient"].items():
        c = c.detach()
        m = snap["exp_avg"][name].detach()
        v = snap["exp_avg_sq"][name].detach()
        theta = snap["theta_start"][name].detach()
        m1 = b1 * m + (1.0 - b1) * c
        v1 = b2 * v + (1.0 - b2) * c * c
        adaptive = (m1 / bc1) / (_torch.sqrt(v1 / bc2) + eps)
        out[name] = (-adaptive - wd * theta).to(dtype=c.dtype)
    return out


def run_lr_response(rt, ctx, stream, H: int, hvp_fp32: bool, counters):
    """Analytic LR-input response on the shared (u=0) H-step forward pass.

    Mirrors run_adjoint_only (A0H15) exactly except: candidate phase omitted
    (LR input carries no data candidate) and w_0 is replaced by the
    accumulated LR-pulse cotangent. State is left MUTATED (caller restores).
    """
    import a0g_decision_utility_v01 as a0g
    from a0h15_production_entry_v01 import (_a0h1, _baseline_no_step, _clip_only,
                                            _clone_map, _endpoint_grad,
                                            _restore_and_seed, _step_once,
                                            _zeros_like)

    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    hvp_ctx = nullcontext() if hvp_fp32 else rt["autocast_ctx"]
    accum = int(cfg["accumulation_steps"])
    assert hvp_fp32, "A0O production candidate is fp32-hvp only"

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
    h_eta = 0.0
    H = int(H)
    for k in range(H, 0, -1):
        snap = snapshots[k]
        # LR pulse at step k is a pure THETA-channel push:
        #   theta_{k+1}(u) = (1 - lr e^u wd) theta_k - lr e^u adaptive(c_k)
        #   d(theta_{k+1})/du = lr_k * e_lr(k),  e_lr = -adaptive(c_k) - wd*theta_k
        # m1/v1 are lr-independent (the consumed gradient c_k does not depend on
        # u), so dm1 = dv1 = 0 at the pulse step. p_th ENTERING iteration k is
        # dL/d(theta_{k+1}) (post-step-k state), hence the pulse contracts
        # directly with p_th before the adjoint rewinds step k.
        e = lr_pulse(snap)
        lr_k = float(snap["lr"])
        h_eta += sum(float((p_th[n] * (lr_k * e[n])).sum()) for n, _ in named)
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
        if k >= 2:
            rng_now = capture_rng_state()
            restore_rng_state(snap["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(snap["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"],
                                w_k, hvp_ctx, accum)
            counters["hvp_calls"] += 1
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}

    return {"h_eta": float(h_eta), "v_H_norm": float(a0g._norm(v_H)),
            "mode": "lr_input_adjoint", "H": H, "counters": counters}


def run_lr_rollout(rt, ctx, stream, H: int, u: float, counters) -> float:
    """FD branch: H native steps with eta_k(u) = eta_k*exp(u) at EVERY step.

    u == 0 reproduces the frozen baseline bitwise. CRN: identical RNG to the
    u=0 shared pass. Returns V_score at endpoint theta_{t+H}(u).
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])

    pe._restore_and_seed(rt, ctx, stream)
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        counters["baseline_steps"] += 1
        # perturb the schedule input for THIS step only
        for g in rt["optimizer"].param_groups:
            g["lr"] = float(snap["lr"]) * (_torch.e ** u)
        pe._clip_and_step(rt, native_gradient, snap)
    rng_now = capture_rng_state()
    val = float(a0g._val_loss_only(rt["torch"], rt["model"],
                                   rt["val_score_tensors"], rt["autocast_ctx"]))
    restore_rng_state(rng_now)
    return val
