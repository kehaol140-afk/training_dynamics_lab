"""C0-GTTR validation: pure tensor-dict algebra for Geometry/Tracking/Transport/Readout.

Offline (CPU-testable, FP64) companions to the frozen GPU primitives. Every
function here is PURE: inputs are never mutated, outputs are fresh dicts, all
reductions are accumulated in float64. No module-level torch state.

Conventions frozen by C0 config.json / audit.md:
  * clip semantics = PyTorch ``g * min(tau/(||g|| + 1e-6), 1)`` over the GLOBAL
    L2 norm of the concatenated parameter gradient (one clip per optimizer
    step / per probe, never merged probes).
  * step functions take the CLIPPED gradient c and operate in bias-corrected
    coordinates: bar_m (== native m_hat), bar_v (== native v_hat), with the
    real cumulative step t_prev BEFORE the update.
  * the log-LR input channel is b_eta = [s_t, 0, 0]: s_t is the realized
    displacement which already contains -eta; nothing here multiplies by eta
    twice.
  * wd is decoupled: s = -lr * r - lr * wd * theta.

Tracking / readout (V_diag axis only; never mixed with V_eval):
  * hat_mu = mean audit clipped gradients at anchor theta_t; hat_nu = mean of
    their squares; e_m = bar_m - hat_mu; e_v = bar_v - hat_nu.
  * W_total = -q^T s (realized, fp64), decomposed as
    W_field + W_track + W_wd with W_field = eta q^T P mu, W_track = eta q^T P e_m,
    W_wd = eta * lambda * q^T theta.
  * K_V = Delta V + W_total (exact readout remainder; NOT guaranteed positive).
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

CLIP_DENOMINATOR_EPS = 1e-6  # matches torch.nn.utils.clip_grad_norm_

TensorDict = Dict[str, "object"]  # torch tensors keyed by parameter name


def _torch():
    import torch
    return torch


def _names(a: Mapping[str, object], b: Mapping[str, object], label: str):
    if not a or set(a) != set(b):
        raise ValueError(f"{label} must have identical non-empty names")
    return list(a)


def clone_td(a: Mapping[str, object]) -> TensorDict:
    return {k: v.detach().clone() for k, v in a.items()}


def global_norm(g: Mapping[str, object]) -> float:
    """Global L2 norm across all tensors, accumulated in fp64."""
    torch = _torch()
    total = None
    for v in g.values():
        s = torch.sum(v.detach().to(torch.float64).square())
        total = s if total is None else total + s
    return float(torch.sqrt(total).detach().cpu())


def dot_td(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    torch = _torch()
    names = _names(a, b, "dot_td")
    total = None
    for n in names:
        x = a[n].detach().to(torch.float64)
        y = b[n].detach().to(torch.float64)
        s = torch.sum(x * y)
        total = s if total is None else total + s
    return float(total.detach().cpu())


def norm_td(a: Mapping[str, object]) -> float:
    return math.sqrt(max(dot_td(a, a), 0.0))


def add_td(a, b) -> TensorDict:
    names = _names(a, b, "add_td")
    return {n: a[n].detach() + b[n].detach() for n in names}


def sub_td(a, b) -> TensorDict:
    names = _names(a, b, "sub_td")
    return {n: a[n].detach() - b[n].detach() for n in names}


def scale_td(a, alpha: float) -> TensorDict:
    return {n: alpha * v.detach() for n, v in a.items()}


def clip_coefficient(norm_value: float, tau: float) -> float:
    return min(tau / (norm_value + CLIP_DENOMINATOR_EPS), 1.0)


def clip_active(norm_value: float, tau: float) -> bool:
    return (tau / (norm_value + CLIP_DENOMINATOR_EPS)) < 1.0


def apply_clip(g: Mapping[str, object], tau: float) -> TensorDict:
    """Exact PyTorch clip_grad_norm_ semantics on detached copies (no mutation).

    Global norm over ALL tensors jointly; each tensor scaled by the same
    coefficient min(tau/(||g||+1e-6), 1).
    """
    coeff = clip_coefficient(global_norm(g), float(tau))
    return scale_td(g, coeff)


def ema_coeffs(beta: float, t_prev: int) -> Tuple[float, float]:
    """Coeffs taking bias-corrected moments from step t_prev to t_prev+1.

    bar_{t+1} = a * bar_t + b * c  with  bar_t == native bias-corrected moment.
    Derivation: native m_{t+1} = beta m_t + (1-beta) c and
    m_hat_{t+1} = m_{t+1}/(1-beta^{t+1}); substituting m_hat_t = m_t/(1-beta^t)
    gives a = beta (1-beta^{t_prev}) / (1-beta^{t_prev+1}),
    b = (1-beta) / (1-beta^{t_prev+1}).
    t_prev = 0 gives (a=0, b=1): no division by zero for beta < 1.
    """
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0,1)")
    t_prev = int(t_prev)
    if t_prev < 0:
        raise ValueError("t_prev must be >= 0")
    num = 1.0 - beta ** t_prev
    den = 1.0 - beta ** (t_prev + 1)
    a = beta * num / den
    b = (1.0 - beta) / den
    return a, b


def _wd_value(wd, name) -> float:
    return float(wd[name]) if isinstance(wd, Mapping) else float(wd)


def adamw_rewrite_step(*, m_hat, v_hat, c, beta1, beta2, t_prev, eps, lr, wd, theta):
    """One full AdamW step in bias-corrected coordinates, pure.

    Inputs: m_hat/v_hat (bias-corrected moments, pre-update), c (CLIPPED grad),
    theta (pre-update params). wd: float or per-name mapping (decoupled mask).
    Returns dict with fresh tensors: m_hat, v_hat (post-update), r = adaptive
    direction, s = realized displacement = -lr*r - lr*wd*theta.
    t_prev is the REAL cumulative optimizer step count before this update.
    Moment arithmetic is dtype-native (matches native AdamW bitwise for a
    given dtype; T1 checks fp64).
    """
    names = _names(m_hat, v_hat, "m_hat/v_hat")
    _names(m_hat, c, "c")
    _names(m_hat, theta, "theta")
    a1, b1 = ema_coeffs(beta1, t_prev)
    a2, b2 = ema_coeffs(beta2, t_prev)
    out_m, out_v, out_r, out_s = {}, {}, {}, {}
    for n in names:
        cm = c[n].detach()
        m1 = a1 * m_hat[n].detach() + b1 * cm
        v1 = a2 * v_hat[n].detach() + b2 * cm * cm  # dtype-native, like native AdamW
        r = m1 / (_torch_sqrt(v1) + float(eps))
        w = _wd_value(wd, n)
        s = -float(lr) * r - float(lr) * w * theta[n].detach()
        out_m[n], out_v[n], out_r[n], out_s[n] = m1, v1, r, s
    return {"m_hat": out_m, "v_hat": out_v, "r": out_r, "s": out_s}


def _torch_sqrt(v):
    import torch
    return torch.sqrt(v)


def realized_carry_over(*, m_hat, v_hat, c, beta1, beta2, t_prev, eps, lr, wd, theta):
    """Realized displacement s_t for anchor bookkeeping (== adamw_rewrite_step['s']).

    The carry-over identity under test (T2): s_t depends on P_t = 1/(sqrt(v_t)+eps)
    THROUGH this update's v_hat only; an implementation substituting the previous
    step's P, or multiplying eta twice, deviates and must fail T2.
    """
    return adamw_rewrite_step(
        m_hat=m_hat, v_hat=v_hat, c=c, beta1=beta1, beta2=beta2, t_prev=t_prev,
        eps=eps, lr=lr, wd=wd, theta=theta)["s"]


def u_map(g_raw, *, m_hat, v_hat, beta1, beta2, t_prev, eps, lr, wd, theta, tau):
    """Shadow update U(g; common_state): real clip -> moments (next cumulative
    step) -> adaptive direction -> displacement. Pure algebra on clones; never
    touches a real optimizer. Returns the shadow displacement s'."""
    c = apply_clip(g_raw, tau)
    return adamw_rewrite_step(
        m_hat=m_hat, v_hat=v_hat, c=c, beta1=beta1, beta2=beta2, t_prev=t_prev,
        eps=eps, lr=lr, wd=wd, theta=theta)["s"]


def u_map_stop_v(g_raw, *, m_hat, v_hat, beta1, beta2, t_prev, eps, lr, wd, theta,
                 tau, P_fixed):
    """Stop-v ablation: m updates with the new gradient, P frozen from a
    pair-baseline (artificial diagnostic; never an optimizer candidate)."""
    c = apply_clip(g_raw, tau)
    a1, b1 = ema_coeffs(beta1, t_prev)
    out_s = {}
    for n in m_hat:
        cm = c[n].detach()
        m1 = a1 * m_hat[n].detach() + b1 * cm
        r = m1 * P_fixed[n].detach()
        w = _wd_value(wd, n)
        out_s[n] = -float(lr) * r - float(lr) * w * theta[n].detach()
    return out_s


def predict_geom(g, hvp, s):
    """Mgeom primary prediction of the displaced gradient: g + H s (pre-clip).

    The caller then applies the REAL clip C before forming predictions
    (Mgeom = C(g + H s) - c). M0 is the zero map; MnoJ is H s without re-clip.
    """
    return add_td(g, hvp)


def tracking_e_m(e_m_prev, c, mu, a1):
    """e_m recursion: e_m,t = a1 e_m,t-1 + (1-a1)(c_t - mu_t)."""
    return add_td(scale_td(e_m_prev, a1), scale_td(sub_td(c, mu), 1.0 - a1))


def tracking_e_v(e_v_prev, c_sq, nu, a2):
    return add_td(scale_td(e_v_prev, a2), scale_td(sub_td(c_sq, nu), 1.0 - a2))


def p_diag(v_hat, eps):
    """Elementwise adaptive preconditioner P = 1/(sqrt(v_hat)+eps)."""
    return {n: 1.0 / (_torch_sqrt(v.detach()) + float(eps)) for n, v in v_hat.items()}


def readout_terms(*, q, s, P, mu, e_m, wd, theta, eta):
    """Per-step readout accounting (V_diag axis). W_total is computed
    INDEPENDENTLY as -q^T s; the identity W_field+W_track+W_wd == W_total is a
    testable property (T4/T9), not enforced by construction."""
    wd_scalar = None if isinstance(wd, Mapping) else float(wd)
    q_dot_theta = dot_td(q, theta)
    W_field = float(eta) * dot_td(q, {n: P[n] * mu[n] for n in mu})
    W_track = float(eta) * dot_td(q, {n: P[n] * e_m[n] for n in e_m})
    if wd_scalar is not None:
        W_wd = float(eta) * wd_scalar * q_dot_theta
    else:
        W_wd = float(eta) * dot_td(q, {n: float(wd[n]) * theta[n] for n in theta})
    W_total = -dot_td(q, s)
    return {"W_field": W_field, "W_track": W_track, "W_wd": W_wd,
            "W_total": W_total, "q_dot_theta": q_dot_theta}


def k_v(v_new: float, v_old: float, W_total: float) -> float:
    """K_V = Delta V + W_total. Exact readout remainder; NOT guaranteed positive."""
    return float(v_new) - float(v_old) + float(W_total)


def ybar(Y) -> TensorDict:
    """Mean of a list of same-named tensor dicts (per-probe outcomes), fp64."""
    if not Y:
        raise ValueError("empty probe list")
    torch = _torch()
    names = list(Y[0])
    out = {}
    for n in names:
        acc = None
        for y in Y:
            v = y[n].detach().to(torch.float64)
            acc = v if acc is None else acc + v
        out[n] = acc / float(len(Y))
    return out


def se_vec(Y) -> float:
    """sqrt(sum ||y_j - ybar||^2 / (K*(K-1))) -- standard error of the mean
    vector, accumulated in fp64."""
    torch = _torch()
    K = len(Y)
    if K < 2:
        raise ValueError("need >= 2 probes for SE")
    mean = ybar(Y)
    acc = None
    for y in Y:
        d = sub_td(y, mean)
        s = None
        for v in d.values():
            t = torch.sum(v.to(torch.float64).square())
            s = t if s is None else s + t
        acc = s if acc is None else acc + s
    return float(torch.sqrt(acc / (K * (K - 1))).detach().cpu())


def resolved(ymean_norm: float, se: float, numerical_floor: float) -> bool:
    """3x engineering resolution gate (NOT significance)."""
    return float(ymean_norm) > 3.0 * math.sqrt(float(se) ** 2 + float(numerical_floor) ** 2)


def E_model(pred, ymean) -> float:
    """E_M = sqrt(sum ||pred - ybar||^2 / sum ||ybar||^2); M0 == 1 by definition."""
    num = norm_td(sub_td(pred, ymean)) ** 2
    den = norm_td(ymean) ** 2
    if den == 0.0:
        return float("nan")
    return math.sqrt(num / den)


def quadratic_pulse_response(lams, x0s, eta, p, H, wrt="eta"):
    """Analytic dL_H/d(pulse) for the fixed quadratic valley L=sum lam_i x_i^2/2
    under a p-step pulse at the start of H gradient steps (T7). Returns
    -p * sum lam_i^2 x0_i^2 (1-eta*lam_i)^(2H-1)  for wrt='eta', and times eta
    for wrt='u' (log-LR pulse, chain rule d/du = eta d/d eta at 0)."""
    if wrt not in ("eta", "u"):
        raise ValueError("wrt must be 'eta' or 'u'")
    total = 0.0
    for lam, x0 in zip(lams, x0s):
        total += (float(lam) ** 2) * (float(x0) ** 2) * (1.0 - float(eta) * float(lam)) ** (2 * H - 1)
    val = -float(p) * total
    return val * float(eta) if wrt == "u" else val


def adamw_step_tangent(*, d_theta, m_hat, v_hat, d_m_hat, d_v_hat, c, d_c,
                       theta, beta1, beta2, t_prev, eps, lr, wd,
                       lr_u_tangent=False):
    """Tangent (d_theta', d_m_hat', d_v_hat') of one AdamW step.

    d_c is the tangent of the clipped gradient induced by the perturbation
    (e.g. J_clip H d_theta for the geometry channel; 0 for a pure LR-input
    pulse when d_theta=0 at the current point). lr_u_tangent=True adds the
    direct d(-lr(u) r - lr(u) wd theta)/du terms for the log-LR input
    channel (d lr/du = lr at u=0). All internal math fp64.
    """
    torch = _torch()
    names = _names(m_hat, v_hat, "m_hat/v_hat")
    a1, b1 = ema_coeffs(beta1, t_prev)
    a2, b2 = ema_coeffs(beta2, t_prev)
    d_th, d_m, d_v = {}, {}, {}
    for n in names:
        cm = c[n].detach().to(torch.float64)
        dcm = d_c[n].detach().to(torch.float64)
        m0 = m_hat[n].detach().to(torch.float64)
        v0 = v_hat[n].detach().to(torch.float64)
        dth = d_theta[n].detach().to(torch.float64)
        dm0 = d_m_hat[n].detach().to(torch.float64)
        dv0 = d_v_hat[n].detach().to(torch.float64)
        m1 = a1 * m0 + b1 * cm
        v1 = a2 * v0 + b2 * cm * cm
        d_m1 = a1 * dm0 + b1 * dcm
        d_v1 = a2 * dv0 + b2 * 2.0 * cm * dcm
        D = torch.sqrt(v1) + float(eps)
        r1 = m1 / D
        d_r = d_m1 / D - m1 * d_v1 / (2.0 * D ** 3)
        w = _wd_value(wd, n)
        d_s = -float(lr) * d_r - float(lr) * w * dth
        if lr_u_tangent:
            # log-LR input eta(u) = eta e^u at u=0: direct terms
            # d(-eta r - eta wd theta_pre)/du = -eta r1 - eta wd theta_pre.
            th_pre = theta[n].detach().to(torch.float64)
            d_s = d_s - float(lr) * r1 - float(lr) * w * th_pre
        d_th[n] = dth + d_s
        d_m[n] = d_m1
        d_v[n] = d_v1
    return {"d_theta": d_th, "d_m_hat": d_m, "d_v_hat": d_v}
