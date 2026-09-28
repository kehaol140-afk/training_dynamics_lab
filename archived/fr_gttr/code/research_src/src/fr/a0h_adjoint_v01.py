"""Reverse adjoint of the frozen A0C/A0G forward tangent operators.

[FR-OBSERVATION-ONLY] This module implements only the *transpose* of the exact
linear operators already frozen in ``a0c_adamw_response_propagation_v01``
(``clip_jvp``, ``adamw_tangent_step``). It introduces no new forward model.

Math contract (frozen in docs/FR_A0H_PROTOCOL.md, section 1):

* Forward joint tangent ``z = (dtheta, dm, dv)`` per parameter tensor:
    dm1  = beta1*dm + (1-beta1)*dc
    dv1  = beta2*dv + 2*(1-beta2)*c*dc
    dth+ = d*dtheta + D_m*dm1 + D_v*dv1 ,  d = 1 - lr*wd
    D_m  = -lr / (bc1*(sqrt_vhat+eps))
    D_v  = +lr*mhat / (bc2*2*sqrt_vhat*(sqrt_vhat+eps)**2)
  with c the clipped gradient of the step and (mhat, vhat) the bias-corrected
  step-post moments m1/bc1, v1/bc2 (m1 = beta1*m + (1-beta1)*c,
  v1 = beta2*v + (1-beta2)*c^2), bc1/bc2 at step_after. The snapshot stores the
  step-pre optimizer moments; this module reconstructs the step-post moments
  exactly as the frozen forward does internally.

* Reverse adjoint step ``p = (p_th, p_m, p_v)`` -> ``A^T p``:
    u_m = p_m + D_m*p_th
    u_v = p_v + D_v*p_th
    b   = (1-beta1)*u_m + 2*(1-beta2)*c*u_v
    p_theta' = d*p_th (+ K^T J^T b for model-C future steps)
    p_m' = beta1*u_m ,  p_v' = beta2*u_v

* Transposes:
    - ``J^T`` (clip) implemented directly (``clip_jvp_transpose``), verified by
      the inner-product identity, not assumed.
    - ``K^T`` uses the *same* registered math-SDPA HVP callable as the forward
      (scalar-loss Hessian is symmetric); the residual of proxy symmetry is
      reported per state as a diagnostic.

* Readout: w_0 = J_1^T b_1 ; hhat_j = <w_0, a_j>. No feedback HVP at the start
  state (the action is an exogenous gradient perturbation).
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

from . import a0c_adamw_response_propagation_v01 as _a0c

CLIP_DENOMINATOR_EPS = _a0c.CLIP_DENOMINATOR_EPS


def _torch():
    import torch

    return torch


def clip_jvp_transpose(
    *, gradient_by_name: Mapping[str, object],
    cotangent_by_name: Mapping[str, object], max_norm: float,
    active: bool, clip_before_norm: float,
    denominator_eps: float = CLIP_DENOMINATOR_EPS,
) -> Dict:
    """Transpose (VJP) of the frozen ``clip_jvp`` linear map, same signature data.

    Active:  w_out = coef*w_in - kappa*g*<g, w_in>
    Inactive: identity.
    """
    torch = _torch()
    names = _a0c._identical_names(gradient_by_name, cotangent_by_name, "gradient/cotangent")
    max_norm = _a0c._finite(max_norm, "max_norm")
    denom_norm = _a0c._finite(clip_before_norm, "clip_before_norm")
    out = {}
    if not active:
        for name in names:
            w = cotangent_by_name[name].detach()
            out[name] = w.clone()
        return {"clip_jvp_transpose": out}
    if denom_norm == 0.0:
        raise ValueError("active clipping cannot have zero gradient norm")
    # global contraction <g, w> accumulated in float64
    dot = None
    for name in names:
        g = gradient_by_name[name].detach()
        w = cotangent_by_name[name].detach().to(device=g.device, dtype=g.dtype)
        part = torch.sum(g.to(torch.float64) * w.to(torch.float64))
        dot = part if dot is None else dot + part
    dot = float(dot.detach().cpu())
    coef = max_norm / (denom_norm + denominator_eps)
    kappa = max_norm * dot / (denom_norm * (denom_norm + denominator_eps) ** 2)
    for name in names:
        g = gradient_by_name[name].detach()
        w = cotangent_by_name[name].detach().to(device=g.device, dtype=g.dtype)
        out[name] = (coef * w - kappa * g).detach()
    return {"clip_jvp_transpose": out}


def adamw_tangent_step_adjoint(
    *, named_parameters: Sequence[tuple[str, object]], optimizer,
    snapshot: Mapping,
    cotangent_theta_by_name: Mapping[str, object],
    cotangent_m_by_name: Mapping[str, object],
    cotangent_v_by_name: Mapping[str, object],
    feedback_cotangent_by_name: Mapping[str, object] | None = None,
) -> Dict:
    """One reverse AdamW tangent step ``A^T p`` from frozen native quantities.

    ``snapshot`` carries the per-step native data the forward operator used:
    ``clipped_gradient`` (c), ``exp_avg`` (m), ``exp_avg_sq`` (v),
    ``step_after`` (float), plus hyperparameters ``lr, weight_decay, beta1,
    beta2, adam_eps`` (already validated homogeneous by the forward pass).
    ``feedback_cotangent_by_name`` is ``K^T J^T b`` for model-C future steps;
    pass ``None`` for model B.
    """
    torch = _torch()
    named_parameters = list(named_parameters)
    names = [name for name, _ in named_parameters]
    expected = set(names)
    for label, mapping in (
        ("cotangent theta", cotangent_theta_by_name),
        ("cotangent m", cotangent_m_by_name),
        ("cotangent v", cotangent_v_by_name),
        ("feedback cotangent", feedback_cotangent_by_name),
    ):
        if mapping is not None and set(mapping) != expected:
            raise ValueError(f"{label} parameter names do not match optimizer model")
    groups = _a0c._parameter_groups(optimizer)

    scalar_keys = ("lr", "weight_decay", "beta1", "beta2", "adam_eps", "step_after")
    scalars = {key: _a0c._finite(snapshot[key], key) for key in scalar_keys}
    d = 1.0 - scalars["lr"] * scalars["weight_decay"]
    bc1 = 1.0 - scalars["beta1"] ** scalars["step_after"]
    bc2 = 1.0 - scalars["beta2"] ** scalars["step_after"]

    out_theta: Dict[str, object] = {}
    out_m: Dict[str, object] = {}
    out_v: Dict[str, object] = {}
    out_dc: Dict[str, object] = {}
    for name, parameter in named_parameters:
        if parameter not in groups:
            raise ValueError(f"parameter {name} not registered in optimizer")
        m = snapshot["exp_avg"][name].detach()      # step-pre first moment
        v = snapshot["exp_avg_sq"][name].detach()   # step-pre second moment
        c = snapshot["clipped_gradient"][name].detach().to(device=m.device, dtype=m.dtype)
        if tuple(m.shape) != tuple(v.shape) or tuple(m.shape) != tuple(c.shape):
            raise ValueError(f"AdamW adjoint tensor shape mismatch for {name}")
        p_th = cotangent_theta_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        p_m = cotangent_m_by_name[name].detach().to(device=m.device, dtype=m.dtype)
        p_v = cotangent_v_by_name[name].detach().to(device=m.device, dtype=m.dtype)

        # Reconstruct the step-post moments exactly as the frozen forward does:
        #   m1 = beta1*m + (1-beta1)*c ; v1 = beta2*v + (1-beta2)*c^2
        beta1 = scalars["beta1"]
        beta2 = scalars["beta2"]
        m1 = beta1 * m + (1.0 - beta1) * c
        v1 = beta2 * v + (1.0 - beta2) * c.square()
        mhat = m1 / bc1
        vhat = v1 / bc2
        sqrt_vhat = torch.sqrt(vhat)
        denominator = sqrt_vhat + scalars["adam_eps"]

        # Coefficients of the frozen forward linear map
        #   dtheta+ = d*dtheta + D_m*dm1 + D_v*dv1
        # verified numerically against the frozen implementation to machine
        # precision (the dv -> dtheta coefficient is POSITIVE: the forward
        # applies -lr*adaptive_jvp and the dv path of adaptive_jvp carries a
        # minus sign, giving -lr*(-mhat*dsqrt/den^2) with dsqrt > 0).
        D_m = -scalars["lr"] / (bc1 * denominator)
        denom2 = denominator.square()
        D_v = torch.zeros_like(vhat)
        regular = sqrt_vhat > 0.0
        D_v[regular] = (
            scalars["lr"] * mhat[regular]
            / (bc2 * 2.0 * sqrt_vhat[regular] * denom2[regular])
        )

        u_m = p_m + D_m * p_th
        u_v = p_v + D_v * p_th
        b = (1.0 - scalars["beta1"]) * u_m + 2.0 * (1.0 - scalars["beta2"]) * c * u_v

        p_theta_next = d * p_th
        if feedback_cotangent_by_name is not None:
            p_theta_next = p_theta_next + feedback_cotangent_by_name[name].detach().to(
                device=m.device, dtype=m.dtype
            )
        out_theta[name] = p_theta_next.detach()
        out_m[name] = (scalars["beta1"] * u_m).detach()
        out_v[name] = (scalars["beta2"] * u_v).detach()
        # dc-channel cotangent: the forward maps dc into (dm1, dv1) only, so
        # its VJP is exactly b (needed to contract against J^T at the boundary
        # and to verify the per-channel inner-product identity)
        out_dc[name] = b.detach()

    return {
        "cotangent_theta": out_theta,
        "cotangent_m": out_m,
        "cotangent_v": out_v,
        "cotangent_dc": out_dc,
        "adjoint_step_after": scalars["step_after"],
        "d": d,
    }


def _inner(mapping_a: Mapping[str, object], mapping_b: Mapping[str, object]) -> float:
    torch = _torch()
    names = _a0c._identical_names(mapping_a, mapping_b, "inner-product")
    total = torch.zeros((), dtype=torch.float64)
    for name in names:
        a = mapping_a[name].detach()
        b = mapping_b[name].detach().to(device=a.device, dtype=a.dtype)
        total += torch.sum(a.to(torch.float64) * b.to(torch.float64))
    return float(total.detach().cpu())


def verify_clip_transpose(
    *, gradient_by_name: Mapping[str, object],
    tangent_by_name: Mapping[str, object],
    cotangent_by_name: Mapping[str, object], max_norm: float,
) -> Dict:
    """Numerically verify <J a, w> == <a, J^T w> with the frozen operators."""
    fwd = _a0c.clip_jvp(
        gradient_by_name=gradient_by_name,
        tangent_gradient_by_name=tangent_by_name,
        max_norm=max_norm,
    )
    adj = clip_jvp_transpose(
        gradient_by_name=gradient_by_name,
        cotangent_by_name=cotangent_by_name,
        max_norm=max_norm,
        active=fwd["clip_active"],
        clip_before_norm=fwd["clip_before_norm"],
    )
    lhs = _inner(fwd["clip_jvp"], cotangent_by_name)
    rhs = _inner(tangent_by_name, adj["clip_jvp_transpose"])
    return {
        "clip_active": fwd["clip_active"],
        "lhs": lhs,
        "rhs": rhs,
        "abs_gap": abs(lhs - rhs),
    }


def hvp_proxy_symmetry_residual(
    *, hvp: Callable[[Mapping[str, object]], Mapping[str, object]],
    x_by_name: Mapping[str, object], y_by_name: Mapping[str, object],
) -> Dict:
    """Diagnostic: max |<x, H y> - <H x, y>| for the registered HVP proxy."""
    hx = hvp(x_by_name)
    hy = hvp(y_by_name)
    lhs = _inner(x_by_name, hy)
    rhs = _inner(hx, y_by_name)
    return {"lhs": lhs, "rhs": rhs, "abs_gap": abs(lhs - rhs)}
