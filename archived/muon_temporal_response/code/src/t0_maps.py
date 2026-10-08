"""T0 maps: Nesterov recursion, ideal polar, installed-semantics finite NS (FP64), tangents.

finite_ns_fp64 mirrors the installed torch/optim/_muon.py math (SHA 38e95918...) exactly:
  - transpose if rows>cols (not triggered for 2x2; recorded)
  - X = H / clamp(||H||_F, min=eps)
  - 5 iterations: A = X X^T; B = b A + c A@A; X = a X + B X
  - no bf16 cast, no shape scaling (A=B for 2x2 -> factor 1 anyway)
polar_fp64 is the ideal full-rank polar factor.
"""
from __future__ import annotations

import numpy as np

from .t0_construct import S, T, BETA

NS_A, NS_B, NS_C = 3.4445, -4.775, 2.0315
NS_STEPS = 5
EPS_NORM = 1e-7


# ---------- Nesterov momentum (installed EMA convention) ----------

def nesterov_H_direct(g: np.ndarray, m0: np.ndarray, beta: float = BETA) -> np.ndarray:
    """H_t via the installed code path: buf = beta*buf + (1-beta)*G; H = (1-beta)*G + beta*buf."""
    buf = m0.copy()
    out = np.empty_like(g)
    for t in range(g.shape[0]):
        buf = beta * buf + (1.0 - beta) * g[t]
        out[t] = (1.0 - beta) * g[t] + beta * buf
    return out


def nesterov_H_window(g: np.ndarray, m0: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Finite-window expansion: H_t = beta^(t+1) M0 + sum_{j=0}^{t-1} w_j G_{t-j}."""
    out = np.empty_like(g)
    for t in range(1, g.shape[0] + 1):
        acc = beta ** (t + 1) * m0
        for j in range(0, t):
            w = (1.0 - beta ** 2) if j == 0 else (1.0 - beta) * beta ** (j + 1)
            acc = acc + w * g[t - 1 - j]
        out[t - 1] = acc
    return out


def kernel_weights(beta: float = BETA, max_len: int = T + 1) -> np.ndarray:
    w = np.empty(max_len)
    w[0] = 1.0 - beta ** 2
    for j in range(1, max_len):
        w[j] = (1.0 - beta) * beta ** (j + 1)
    return w


def filter_covariance(c_xi: np.ndarray, beta: float = BETA) -> np.ndarray:
    """C_H(t) = sum_{i,j<t} w_i w_j C_xi((i-j) mod L), t=1..T. c_xi: (L,4,4)."""
    L = c_xi.shape[0]
    w = kernel_weights(beta, L + 1)
    out = np.empty((L, 4, 4))
    for t in range(1, L + 1):
        acc = np.zeros((4, 4))
        for i in range(t):
            for j in range(t):
                acc += w[i] * w[j] * c_xi[(i - j) % L]
        out[t - 1] = acc
    return out


# ---------- ideal polar ----------

def polar_fp64(h: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(h)
    return u @ vt


def polar_tangent_at_S(e: np.ndarray) -> np.ndarray:
    """Frechet derivative of polar at S (S posdef -> Q=I): S*Omega+Omega*S = E - E^T."""
    c = e - e.T
    sig = np.diag(S)
    omega = np.empty_like(e)
    for i in range(2):
        for j in range(2):
            omega[i, j] = c[i, j] / (sig[i] + sig[j])
    return omega  # Q=I, (I-QQ^T)=0 for square full-rank


# ---------- installed-semantics finite NS (FP64 proxy) ----------

def finite_ns_fp64(h: np.ndarray, ns_steps: int = NS_STEPS, eps: float = EPS_NORM,
                   coeffs=(NS_A, NS_B, NS_C)) -> np.ndarray:
    a, b, c = coeffs
    x = h.copy()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    n = max(float(np.linalg.norm(x, "fro")), eps)  # norm().clamp(min=eps)
    x = x / n
    for _ in range(ns_steps):
        g_ = x @ x.T
        bu = b * g_ + c * (g_ @ g_)
        x = a * x + bu @ x
    if transposed:
        x = x.T
    return x


def finite_ns_tangent_at_S(e: np.ndarray, ns_steps: int = NS_STEPS, eps: float = EPS_NORM,
                           coeffs=(NS_A, NS_B, NS_C)) -> np.ndarray:
    """JVP of finite_ns_fp64 at H=S along E: normalization branch + E2 recursion.

    Normalization (smooth branch n=||S||_F>eps): dX0 = E/n - S<S,E>_F/n^3."""
    a, b, c = coeffs
    n = float(np.linalg.norm(S, "fro"))
    inner = float(np.sum(S * e))
    x = S / n
    dx = e / n - S * inner / n ** 3
    for _ in range(ns_steps):
        g_ = x @ x.T
        dg = dx @ x.T + x @ dx.T
        dbu = b * dg + c * (dg @ g_ + g_ @ dg)
        dx = a * dx + dbu @ x + (b * g_ + c * (g_ @ g_)) @ dx
        x = a * x + (b * g_ + c * (g_ @ g_)) @ x
    return dx


def finite_ns_tangent_at_H(h: np.ndarray, e: np.ndarray, ns_steps: int = NS_STEPS,
                           eps: float = EPS_NORM, coeffs=(NS_A, NS_B, NS_C)) -> np.ndarray:
    """JVP of finite_ns_fp64 at general (full-rank, n>eps) H along E."""
    a, b, c = coeffs
    transposed = h.shape[0] > h.shape[1]
    hh = h.T if transposed else h
    ee = e.T if transposed else e
    n = float(np.linalg.norm(hh, "fro"))
    if n <= eps:  # clamp branch: derivative of X/n w.r.t. H is E/eps
        x = hh / eps
        dx = ee / eps
    else:
        inner = float(np.sum(hh * ee))
        x = hh / n
        dx = ee / n - hh * inner / n ** 3
    for _ in range(ns_steps):
        g_ = x @ x.T
        bu = b * g_ + c * (g_ @ g_)
        dg = dx @ x.T + x @ dx.T
        dbu = b * dg + c * (dg @ g_ + g_ @ dg)
        dx = a * dx + dbu @ x + bu @ dx
        x = a * x + bu @ x
    if transposed:
        dx = dx.T
    return dx


# ---------- map registry ----------

def apply_map(map_name: str, h: np.ndarray) -> np.ndarray:
    if map_name == "polar_fp64":
        return polar_fp64(h)
    if map_name == "finite_ns_fp64":
        return finite_ns_fp64(h)
    raise ValueError(map_name)


def tangent_at_S(map_name: str, e: np.ndarray) -> np.ndarray:
    if map_name == "polar_fp64":
        return polar_tangent_at_S(e)
    if map_name == "finite_ns_fp64":
        return finite_ns_tangent_at_S(e)
    raise ValueError(map_name)


def tangent_matrix(map_name: str) -> np.ndarray:
    """4x4 Jacobian at H=S in row-major vec convention."""
    basis = [np.zeros((2, 2)) for _ in range(4)]
    Jm = np.empty((4, 4))
    for k in range(4):
        basis[k][k // 2, k % 2] = 1.0
        dk = tangent_at_S(map_name, basis[k])
        Jm[:, k] = dk.reshape(4)
    return Jm


# ---------- shuffle covariance formula B2 (audit) ----------

def b2_covariance(vectors: list[np.ndarray], a_coef: np.ndarray) -> np.ndarray:
    """Exact covariance of sum_i a_i g_{pi(i)} under uniform random permutation (B2)."""
    n = len(vectors)
    g = np.stack([v.reshape(-1) for v in vectors])
    gbar = g.mean(axis=0)
    cen = g - gbar
    sigma = (cen.T @ cen) / n
    return (n / (n - 1) * float(np.sum(a_coef ** 2))
            - float(np.sum(a_coef)) ** 2 / (n - 1)) * sigma
