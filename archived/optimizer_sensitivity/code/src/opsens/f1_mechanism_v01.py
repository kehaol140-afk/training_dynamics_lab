"""F1 mechanism-closure library (v01) — frozen scope in configs/f1_mechanism_v01.json.

All bank access READ-ONLY (legacy v07/G1a/G1b banks via opsens.f0_bank_audit_v01).
Anchor: c=0 (null next-step gradient), identical to F0. At this anchor the
full-state one-step AdamW map z=(theta,m,v) -> z' has the exact per-coordinate
block structure (all diagonal; MiniMind banks contain no exact-zero v, F0-verified):

  dtheta' = (1-lr*wd)*dtheta - lr*[ dm1/bc1 - m1_hat*dsqrt ] / (sqrt(v1_hat)+eps)^2
          = (1-lr*wd)*dtheta - M*dm + Dv*dv
  dm'     = beta1*dm
  dv'     = beta2*dv

with the gradient channel G_i = lr*(1-beta1)/((1-beta1^(t+1))*(sqrt(v1_hat_i)+eps))
[F0], the m-STATE channel diagonal

  M_i = lr*beta1/(bc1*(sqrt(v1_hat_i)+eps)) = (beta1/(1-beta1))*G_i

(NOT beta1*G: F0's informal "m-channel = beta1*G" note conflated the momentum
recurrence with the state-channel derivative; F0's measured artifacts used the
gradient channel only and are unaffected), m1_hat = beta1*m/bc1,
v1_hat = beta2*v/bc2, and the analytic v-channel diagonal

  Dv_i = lr*m1_hat_i*(beta2/(2*bc2*sqrt(v1_hat_i)))/(sqrt(v1_hat_i)+eps)^2.

All amplifications are computed with the vendored frozen
adamw_tangent_step (identical to F0's pipeline), cross-checked against these
closed forms by unit tests.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import f0_bank_audit_v01 as f0
from .a0c_adamw_response_propagation_v01 import adamw_tangent_step

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "f1_mechanism"

GEN_SEED = 20260926
RANDOM_SEEDS = (11, 22, 33)
TAU_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)
KNOCKOUT_FRACTIONS = (1e-4, 1e-3, 1e-2, 5e-2)
KILLCURVE_FRACTIONS = tuple([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2,
                             0.1, 0.2, 0.5, 1.0])


def gradient_channel_diagonal(v: torch.Tensor, step_t: float, lr: float,
                              beta1: float, beta2: float, eps: float,
                              dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """F0 gradient-channel diagonal with the arithmetic forced into `dtype`
    (bank v is fp32 storage; fp64 analysis requires the cast up front)."""
    return f0.diagonal_gain(v.to(dtype), step_t, lr, beta1, beta2, eps)


def m_channel_diagonal(v: torch.Tensor, step_t: float, lr: float, beta1: float,
                       beta2: float, eps: float,
                       dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Analytic |d theta'/d m| diagonal at the c=0 anchor:
    lr*beta1/(bc1*(sqrt(v1_hat)+eps)) = (beta1/(1-beta1))*G."""
    g = gradient_channel_diagonal(v, step_t, lr, beta1, beta2, eps, dtype)
    return g * (beta1 / (1.0 - beta1))


def v_channel_diagonal(m: torch.Tensor, v: torch.Tensor, step_t: float, lr: float,
                       beta1: float, beta2: float, eps: float,
                       compute_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Analytic |d theta'/d v| diagonal at the c=0 anchor (config R_der basis)."""
    m = m.to(compute_dtype)
    v = v.to(compute_dtype)
    t_after = step_t + 1.0
    bc1 = 1.0 - beta1 ** t_after
    bc2 = 1.0 - beta2 ** t_after
    vhat1 = beta2 * v / bc2
    m1_hat = beta1 * m / bc1
    sqrt_v = torch.sqrt(vhat1)
    denom = sqrt_v + eps
    return lr * m1_hat * (beta2 / (2.0 * bc2 * sqrt_v)) / denom.square()


def theta_channel_factor(lr: float, wd: float) -> float:
    return 1.0 - lr * wd


def run_tangent_state_channels(named, optimizer, *, q_theta=None, q_m=None, q_v=None):
    """Vendored frozen tangent at the c=0 anchor with arbitrary per-channel
    tangent inputs (any subset may be None -> zeros). Returns the raw output
    dict (tangent_theta/m/v) in the optimizer's dtype."""
    zero = {name: torch.zeros_like(p.detach()) for name, p in named}
    dtheta = zero if q_theta is None else q_theta
    dm = zero if q_m is None else q_m
    dv = zero if q_v is None else q_v
    return adamw_tangent_step(
        named_parameters=named,
        optimizer=optimizer,
        clipped_gradient_by_name=zero,
        tangent_clipped_gradient_by_name=zero,
        tangent_theta_by_name=dtheta,
        tangent_m_by_name=dm,
        tangent_v_by_name=dv,
    )


def channel_directions(named, seed: int):
    """Unit-L2 iid N(0,1) directions per channel, fp64, frozen seed."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for ch in ("theta", "m", "v"):
        q, sq = {}, 0.0
        for name, p in named:
            t = torch.randn(p.shape, generator=gen, dtype=torch.float64)
            sq += float(torch.sum(t * t).item())
            q[name] = t
        n = math.sqrt(sq)
        out[ch] = {k: v / n for k, v in q.items()}
    return out


def unit_mhat_direction(named, opt_state) -> Dict[str, torch.Tensor]:
    q, sq = {}, 0.0
    for name, p in named:
        t = opt_state[p]["exp_avg"].detach().to(torch.float64)
        sq += float(torch.sum(t * t).item())
        q[name] = t
    n = math.sqrt(sq)
    return {k: v / n for k, v in q.items()}


def fake_optimizer(state: Dict, dtype: torch.dtype, v_override=None):
    """F0-style fake optimizer carrying bank m/v/step; v_override substitutes
    counterfactual v tensors (F1-B)."""
    hp = f0.group_hyperparams(state)
    named = []
    opt_state = {}
    for name, idx, slot in f0.named_optimizer_entries(state):
        p = torch.nn.Parameter(state["model"][name].detach().to(dtype).clone())
        named.append((name, p))
        v_src = slot["exp_avg_sq"] if v_override is None else v_override[name]
        opt_state[p] = {
            "step": torch.tensor(float(slot["step"])),
            "exp_avg": slot["exp_avg"].detach().to(dtype).clone(),
            "exp_avg_sq": v_src.detach().to(dtype).clone(),
        }

    class _Opt:
        pass

    opt = _Opt()
    opt.param_groups = [{"params": [p for _, p in named], "lr": hp["lr"],
                         "betas": (hp["beta1"], hp["beta2"]), "eps": hp["eps"],
                         "weight_decay": hp["weight_decay"], "amsgrad": False,
                         "maximize": False, "differentiable": False}]
    opt.state = opt_state
    return named, opt


def analytic_channel_diagonals(state: Dict, dtype: torch.dtype = torch.float64):
    """Per-parameter analytic diagonals (G, Dv, theta factor) at the anchor."""
    hp = f0.group_hyperparams(state)
    out = {}
    for name, idx, slot in f0.named_optimizer_entries(state):
        step_t = float(slot["step"])
        v = slot["exp_avg_sq"]
        m = slot["exp_avg"]
        g = f0.diagonal_gain(v, step_t, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"],
                             compute_dtype=dtype)
        dv = v_channel_diagonal(m, v, step_t, hp["lr"], hp["beta1"], hp["beta2"],
                                hp["eps"], compute_dtype=dtype)
        out[name] = {"G": g, "Dv": dv, "step": step_t}
    return out, hp


def anchor_update(v: torch.Tensor, m: torch.Tensor, step_t: float, lr: float,
                  beta1: float, beta2: float, eps: float, wd: float,
                  theta: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """c=0-anchor forward update u = theta'-theta (torch-semantics math)."""
    t_after = step_t + 1.0
    bc1 = 1.0 - beta1 ** t_after
    bc2 = 1.0 - beta2 ** t_after
    v64, m64 = v.to(dtype), m.to(dtype)
    m1 = beta1 * m64
    v1 = beta2 * v64
    mhat = m1 / bc1
    vhat = v1 / bc2
    upd = mhat / (vhat.sqrt() + eps)
    return -lr * (upd + wd * theta.to(dtype))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    den = n * (n - 1) / 2.0
    return (conc - disc) / den if den else float("nan")


def bank_state_path(scale: str, seed: int, step: int) -> Path:
    spec = f0.LAB_ROOT / "legacy_fluctuation_response"
    roots = {
        "v07_5m": spec / "outputs/e3_confirmatory/v07",
        "g1a_26m": spec / "outputs/generalization_g1a_26m_v01/bank",
        "g1b_100m": spec / "outputs/generalization_g1b_100m_v01/bank",
    }
    return roots[scale] / f"seed_{seed}/states/state_step_{step:03d}.pt"


def modulewise_shares(named, tensor_by_name) -> Dict[str, float]:
    """Per-module share of the squared norm of tensor_by_name (F0 taxonomy;
    MiniMind layout has unfused params, so classify_group maps whole tensors)."""
    acc = {}
    total = 0.0
    for name, _ in named:
        t = tensor_by_name[name].detach().to(torch.float64)
        g = f0.classify_group(name)
        s = float(torch.sum(t * t).item())
        acc[g] = acc.get(g, 0.0) + s
        total += s
    return {k: v / total for k, v in acc.items()} if total > 0 else acc
