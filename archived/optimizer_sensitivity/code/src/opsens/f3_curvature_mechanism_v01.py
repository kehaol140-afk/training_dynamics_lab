"""F3 curvature-feedback mechanism library (v01) — frozen scope in
docs/F3_CURVATURE_MECHANISM_PREREG.md (2026-09-27, commits f51ce82 + 658301c).

Per-arm curvature-feedback tangent with frozen alpha scaling and module
receiver masks; impulse (source-free) propagation; matrix-free spectrum
diagnostics (power iteration + Arnoldi); curvature gain decomposition.

Red lines: no optimizer/stabilizer/controller development; the vendored
frozen adamw_tangent_step is never modified; module masks only zero the
curvature-feedback vector dc AFTER clip_jvp and BEFORE it enters the
tangent (forward rollout untouched).
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from . import f0_bank_audit_v01 as f0
from .a0c_adamw_response_propagation_v01 import adamw_tangent_step

# ---- frozen grids / constants (prereg §1, §3-§6) --------------------------
F3_ALPHA_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
F3_H_READOUTS = (1, 2, 4, 8, 16, 32)
F3_H_MAX = 32
F3_EPS_METRIC = 1e-30
F3_STOP_TANGENT_NORM = 1e30

F3_PROBE_SEEDS = {
    "theta_only": 91001,
    "m_only": 91002,
    "v_only": 91003,
    "full_random": 91004,
    "attn_q_localized": 91005,
}

F3_OFFSETS = (0, 1, 3, 7, 15)
F3_PI_ITERS = 20
F3_PI_TOL = 1e-6
F3_PI_SEED = 91011
F3_ARNOLDI_M = 16
F3_ARNOLDI_RES_TOL = 0.05

F3_RHO_MARGIN = 1.05
F3_TRANSIENT_THRESHOLD = 10.0
F3_ALPHA0_CONTROL = 3.0
F3_NONNORMAL_PROXY = 2.0
F3_LARGE_EFFECT = 10.0

# ---- RQ3 module receiver masks (prereg §5) --------------------------------
# arm -> (mode, groups) with mode "remove" (groups zeroed) or "keep" (only
# groups kept). Groups are F0 classify_group labels (frozen taxonomy).
F3_ARMS: Mapping[str, Tuple[str, Tuple[str, ...]]] = {
    "FULL": ("keep", ("embed_tied", "attn_q", "attn_k", "attn_v", "attn_o",
                      "attn_qk_norm", "rmsnorm", "mlp_gate", "mlp_up", "mlp_down")),
    "NO_ATTN_Q": ("remove", ("attn_q",)),
    "NO_ATTN_K": ("remove", ("attn_k",)),
    "NO_ATTN_V": ("remove", ("attn_v",)),
    "NO_ATTN_O": ("remove", ("attn_o",)),
    "NO_MLP": ("remove", ("mlp_gate", "mlp_up", "mlp_down")),
    "NO_EMBED": ("remove", ("embed_tied",)),
    "NO_NORM": ("remove", ("attn_qk_norm", "rmsnorm")),
    "ONLY_ATTN_Q": ("keep", ("attn_q",)),
    "ONLY_ATTN_K": ("keep", ("attn_k",)),
    "ONLY_MLP": ("keep", ("mlp_gate", "mlp_up", "mlp_down")),
    "NO_MLP_UP": ("remove", ("mlp_up",)),
    "NO_MLP_GATE": ("remove", ("mlp_gate",)),
    "NO_MLP_DOWN": ("remove", ("mlp_down",)),
}


def module_mask(named, arm: str) -> Dict[str, torch.Tensor]:
    """Per-name 0/1 float32 mask for the curvature-feedback vector dc.
    FULL -> all ones. Bit-exact against no-mask when arm == FULL."""
    if arm not in F3_ARMS:
        raise ValueError(f"unknown F3 arm {arm}")
    mode, groups = F3_ARMS[arm]
    gset = set(groups)
    mask = {}
    for name, _ in named:
        g = f0.classify_group(name)
        keep = (g in gset) if mode == "keep" else (g not in gset)
        ref = next(p for n, p in named if n == name)
        mask[name] = torch.full(ref.shape, 1.0 if keep else 0.0,
                                dtype=torch.float32)
    return mask


def apply_mask(dc_by_name: Mapping[str, torch.Tensor],
               mask_by_name: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name, dc in dc_by_name.items():
        m = mask_by_name[name].to(device=dc.device, dtype=dc.dtype)
        out[name] = dc * m
    return out


# ---- state helpers (dz = {'th','m','v'} : name -> fp64 tensor) -------------
def state_norm(dz) -> float:
    total = 0.0
    for ch in ("th", "m", "v"):
        for t in dz[ch].values():
            total += float(torch.sum(t.double().square()).item())
    return math.sqrt(total)


def state_inner(a, b) -> float:
    total = 0.0
    for ch in ("th", "m", "v"):
        for n in a[ch]:
            total += float(torch.sum(a[ch][n].double() * b[ch][n].double()).item())
    return total


def state_scale(a, s: float):
    return {ch: {n: t * s for n, t in a[ch].items()} for ch in ("th", "m", "v")}


def state_add(a, b, sa: float = 1.0, sb: float = 1.0):
    return {ch: {n: sa * a[ch][n] + sb * b[ch][n] for n in a[ch]}
            for ch in ("th", "m", "v")}


def zeros_like_state(named, dtype=torch.float64):
    z = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    return {"th": dict(z), "m": dict(z), "v": dict(z)}


# ---- RQ2 probes ------------------------------------------------------------
def make_probe(named, which: str, seed: int):
    gen = torch.Generator().manual_seed(seed)
    dz = zeros_like_state(named)
    if which == "theta_only":
        chans = ("th",)
    elif which == "m_only":
        chans = ("m",)
    elif which == "v_only":
        chans = ("v",)
    elif which == "full_random":
        chans = ("th", "m", "v")
    elif which == "attn_q_localized":
        chans = ("th",)
    else:
        raise ValueError(which)
    for ch in chans:
        for name, p in named:
            if which == "attn_q_localized" and f0.classify_group(name) != "attn_q":
                continue
            dz[ch][name] = torch.randn(p.shape, generator=gen, dtype=torch.float64)
    n = state_norm(dz)
    if n == 0.0:
        raise ValueError("zero probe")
    return state_scale(dz, 1.0 / n)


def lr_source_q0(named, theta_prev, theta_next):
    """Frozen q0: theta-only unit tangent along the h=1 LR source direction."""
    dz = zeros_like_state(named)
    for name, p in named:
        dz["th"][name] = (theta_next[name].double() - theta_prev[name].double())
    n = state_norm(dz)
    return state_scale(dz, 1.0 / max(n, F3_EPS_METRIC))


# ---- tangent step wrappers --------------------------------------------------
def hom_step(named, opt, clipped_by_name, dc_by_name, dz):
    """Homogeneous part of one AdamW tangent step (vendored, frozen)."""
    out = adamw_tangent_step(
        named_parameters=named, optimizer=opt,
        clipped_gradient_by_name=clipped_by_name,
        tangent_clipped_gradient_by_name=dc_by_name,
        tangent_theta_by_name=dz["th"], tangent_m_by_name=dz["m"],
        tangent_v_by_name=dz["v"])
    return {"th": out["tangent_theta"], "m": out["tangent_m"],
            "v": out["tangent_v"]}


# ---- RQ5 curvature gain decomposition ---------------------------------------
def curvature_decomposition(named, opt, clipped_by_name, dc_by_name, dz):
    """Along the alpha=1 trajectory: A dz = homogeneous step with dc=0;
    (A+C) dz = homogeneous step with dc; C dz = difference (affine map).
    Returns (norm_AdC, norm_A, norm_C, inner_AdC_A, cos)."""
    zero = {n: torch.zeros_like(t) for n, t in clipped_by_name.items()}
    out_ad = hom_step(named, opt, clipped_by_name, dc_by_name, dz)
    out_a = hom_step(named, opt, clipped_by_name, zero, dz)
    c = state_add(out_ad, out_a, sa=1.0, sb=-1.0)
    n_adc, n_a, n_c = state_norm(out_ad), state_norm(out_a), state_norm(c)
    inner = state_inner(out_a, c)
    cos = inner / max(n_a * n_c, F3_EPS_METRIC)
    return {"norm_AplusC": n_adc, "norm_A": n_a, "norm_C": n_c,
            "inner_A_C": inner, "cos": cos}


# ---- metrics (frozen formulas) ----------------------------------------------
def G_forced(dtheta_H_norm: float, dtheta_1_norm: float) -> float:
    return dtheta_H_norm / max(dtheta_1_norm, F3_EPS_METRIC)


def G_impulse(dz_H_norm: float, dz_0_norm: float) -> float:
    return dz_H_norm / max(dz_0_norm, F3_EPS_METRIC)


def lambda_H(G: float, H: int) -> float:
    return math.log(max(G, F3_EPS_METRIC)) / H


# ---- RQ4: J / J^T around a step ---------------------------------------------
class StepOperator:
    """Matrix-free linear operator of one homogeneous tangent step.

    JVP:  dz -> adamw_tangent_step(dz; dc = K(dth))   (K = clip_jvp o HVP)
    VJP:  adjoint, with K^T(w) = coef*HVP(w) - k*<g,w>*HVP(g)  (clip active)
          or HVP(w) when clipping is inactive. HVP(g) is cached per step.

    hvp_fn: name->fp64-CPU tensor dict -> name->fp64-CPU tensor dict,
            applying the Gauss-Newton HVP at the step's batch (GPU inside).
    """

    def __init__(self, named, optimizer, clipped_by_name, coef: float,
                 active: bool, k: float, hvp_fn, enabled: bool = True):
        self.named = list(named)
        self.opt = optimizer
        self.clipped = {n: t.detach().clone() for n, t in clipped_by_name.items()}
        self.coef = float(coef)
        self.active = bool(active)
        self.k = float(k)
        self.hvp = hvp_fn
        self.enabled = bool(enabled)
        g = {n: t.double() for n, t in self.clipped.items()}
        self.hvp_g = self.hvp(g) if (self.active and self.enabled) else None

    def _K(self, dth):
        if not self.enabled:
            return {n: torch.zeros_like(t, dtype=torch.float64)
                    for n, t in dth.items()}
        hvp_x = self.hvp({n: t.double() for n, t in dth.items()})
        if not self.active:
            return hvp_x
        dot = sum(float(torch.sum(self.clipped[n].double() * hvp_x[n]).item())
                  for n in hvp_x)
        out = {}
        for n in hvp_x:
            out[n] = self.coef * hvp_x[n] - (self.k * dot) * self.clipped[n].double()
        return out

    def _KT(self, w):
        if not self.enabled:
            return {n: torch.zeros_like(t, dtype=torch.float64)
                    for n, t in w.items()}
        if not self.active:
            return self.hvp({n: t.double() for n, t in w.items()})
        hvp_w = self.hvp({n: t.double() for n, t in w.items()})
        dot = sum(float(torch.sum(self.clipped[n].double() * w[n]).item())
                  for n in w)
        out = {}
        for n in hvp_w:
            out[n] = self.coef * hvp_w[n] - (self.k * dot) * self.hvp_g[n]
        return out

    def jvp(self, dz):
        dc = self._K({n: t for n, t in dz["th"].items()})
        return hom_step(self.named, self.opt, self.clipped, dc, dz)

    def vjp(self, w):
        # adjoint of the AdamW tangent recursion (see module doc / prereg §6).
        # Pass 1: per-parameter adjoint accumulation WITHOUT any HVP.
        a_dc = {}
        out_th, out_m, out_v = {}, {}, {}
        groups = {}
        for grp in self.opt.param_groups:
            for p in grp["params"]:
                groups[p] = grp
        for name, p in self.named:
            grp = groups[p]
            beta1, beta2 = map(float, grp["betas"])
            lr = float(grp["lr"])
            eps = float(grp["eps"])
            wd = float(grp.get("weight_decay", 0.0))
            state = self.opt.state[p]
            step_t = float(state["step"])
            bc1 = 1.0 - beta1 ** (step_t + 1.0)
            bc2 = 1.0 - beta2 ** (step_t + 1.0)
            m = state["exp_avg"].detach().double()
            v = state["exp_avg_sq"].detach().double()
            c = self.clipped[name].double()
            m1 = beta1 * m + (1.0 - beta1) * c
            v1 = beta2 * v + (1.0 - beta2) * c.square()
            mhat = m1 / bc1
            vhat = v1 / bc2
            sqrt_v = torch.sqrt(vhat)
            denom = sqrt_v + eps
            regular = sqrt_v > 0.0
            w_th = w["th"][name].double()
            w_m = w["m"][name].double()
            w_v = w["v"][name].double()
            # dth' = (1-lr*wd)*dth - lr*dmhat/denom + lr*mhat*dsqrt/denom^2
            a_dmhat = (-lr / denom) * w_th
            a_dsqrt = (lr * mhat / denom.square()) * w_th
            a_dvhat = torch.zeros_like(a_dsqrt)
            a_dvhat[regular] = a_dsqrt[regular] / (2.0 * sqrt_v[regular])
            a_dv1 = w_v + a_dvhat / bc2
            a_dm1 = w_m + a_dmhat / bc1
            a_dc[name] = (1.0 - beta1) * a_dm1 + 2.0 * (1.0 - beta2) * c * a_dv1
            out_m[name] = (beta1 * a_dm1).clone()
            out_v[name] = (beta2 * a_dv1).clone()
            out_th[name] = ((1.0 - lr * wd) * w_th).clone()
        # Pass 2: ONE transposed curvature application for all parameters.
        kt = self._KT(a_dc)
        for name, _ in self.named:
            out_th[name] += kt[name]
        return {"th": out_th, "m": out_m, "v": out_v}

    def matvec(self, dz):
        return self.jvp(dz)


def clip_linear_coefficients(clip_info: Mapping, max_norm: float):
    """From clip_jvp output: (coef, active, k) with dc = coef*HVP(dth)
    - k*<g,HVP(dth)>*g when active; dc = HVP(dth) when inactive."""
    active = bool(clip_info["clip_active"])
    coef = float(clip_info["clip_coefficient"])
    if not active:
        return coef, False, 0.0
    norm = float(clip_info["clip_before_norm"])
    eps_c = float(clip_info["clip_denominator_epsilon"])
    k = max_norm / (norm * (norm + eps_c) ** 2)
    return coef, True, k


# ---- RQ4: power iteration + Arnoldi -----------------------------------------
def _random_unit_state(named, seed: int):
    gen = torch.Generator().manual_seed(seed)
    dz = zeros_like_state(named)
    for ch in ("th", "m", "v"):
        for name, p in named:
            dz[ch][name] = torch.randn(p.shape, generator=gen, dtype=torch.float64)
    return state_scale(dz, 1.0 / state_norm(dz))


def power_iteration_sigma_max(op: StepOperator, named, iters=F3_PI_ITERS,
                              tol=F3_PI_TOL, seed=F3_PI_SEED):
    """sigma_max(J) = sqrt(lambda_max(J^T J)) via matrix-free PI."""
    q = _random_unit_state(named, seed)
    sigma_prev = None
    sigma = 0.0
    done = 0
    for i in range(1, iters + 1):
        v = op.jvp(q)
        u = op.vjp(v)
        n2 = state_inner(u, u)
        sigma = math.sqrt(max(n2, 0.0))
        if sigma <= 0.0:
            return 0.0, i
        q = state_scale(u, 1.0 / sigma)
        done = i
        if sigma_prev is not None and abs(sigma - sigma_prev) <= tol * max(sigma, F3_EPS_METRIC):
            break
        sigma_prev = sigma
    return sigma, done


def arnoldi_rho(op: StepOperator, named, m=F3_ARNOLDI_M, seed=F3_PI_SEED,
                res_tol=F3_ARNOLDI_RES_TOL):
    """Matrix-free Arnoldi; returns (rho, reliable, residual_bound, m).

    rho = |largest-magnitude eigenvalue of Hessenberg H_m|;
    residual bound = h_{m+1,m} * |s_m| / |theta| with s the Ritz vector.
    """
    import numpy as np

    basis = [_random_unit_state(named, seed)]
    H = torch.zeros((m + 1, m), dtype=torch.float64)
    m_eff = m
    for j in range(m):
        w = op.jvp(basis[j])
        # modified Gram-Schmidt with one reorthogonalization pass
        for _ in range(2):
            for i in range(j + 1):
                hij = state_inner(w, basis[i])
                H[i, j] += hij
                w = state_add(w, basis[i], sa=1.0, sb=-hij)
        h_next = state_norm(w)
        H[j + 1, j] = h_next
        if h_next <= 0.0:
            m_eff = j + 1
            break
        if j + 1 <= m - 1:
            basis.append(state_scale(w, 1.0 / h_next))
    Hm = H[:m_eff, :m_eff].numpy()
    vals, vecs = np.linalg.eig(Hm)
    idx = int(np.argmax(np.abs(vals)))
    theta = vals[idx]
    s = vecs[:, idx]
    resid = abs(H[m_eff, m_eff - 1] * s[m_eff - 1]) / max(abs(theta), 1e-300)
    return abs(theta), bool(resid <= res_tol), float(resid), m_eff


# ---- RQ4 mechanism classification (frozen, prereg §8.1) ----------------------
def classify_cell(rho_points, sigma_points, G16_alpha1, G16_alpha0):
    """rho_points: list of (offset, rho or None[NOT_ESTIMATED], reliable).
    Returns one of SPECTRAL_UNSTABLE / NONNORMAL_TRANSIENT /
    TIME_VARYING_AMPLIFICATION / UNRESOLVED (MIXED decided by caller)."""
    early = [(o, r, rel) for (o, r, rel) in rho_points]
    reliable_gt = sum(1 for (_, r, rel) in early if rel and r > F3_RHO_MARGIN)
    reliable_le = sum(1 for (_, r, rel) in early
                      if rel and r <= F3_RHO_MARGIN)
    any_reliable = any(rel for (_, _, rel) in early)
    transient = G16_alpha1 >= F3_TRANSIENT_THRESHOLD
    control_ok = G16_alpha0 < F3_ALPHA0_CONTROL
    nonnormal_proxy = False
    if any_reliable:
        for (o, r, rel), (_, s) in zip(rho_points, sigma_points):
            if rel and s > F3_NONNORMAL_PROXY * max(r, F3_EPS_METRIC):
                nonnormal_proxy = True
    spectral = reliable_gt >= 2 and transient
    nonnormal = (reliable_le >= 2 and nonnormal_proxy and transient and control_ok)
    # time-varying is the residual class: transient growth that is neither
    # spectrally unstable nor matching the non-normal pattern.
    timevarying = transient and (not spectral) and (not nonnormal)
    if spectral:
        return "SPECTRAL_UNSTABLE"
    if nonnormal:
        return "NONNORMAL_TRANSIENT"
    if timevarying:
        return "TIME_VARYING_AMPLIFICATION"
    return "UNRESOLVED"
