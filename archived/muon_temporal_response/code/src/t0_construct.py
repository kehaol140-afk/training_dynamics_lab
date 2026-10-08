"""T0 round-1 fixed-multiset construction (execution plan section 4).

All arrays FP64 NumPy. Zero-based sequence indices; optimizer step t is 1-based.
"""
from __future__ import annotations

import hashlib

import numpy as np

S = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
BS = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
BA = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64) / np.sqrt(2.0)
T = 64
BETA = 0.95
S_NORM_F = float(np.linalg.norm(S, "fro"))  # sqrt(5)


def x_sequence() -> np.ndarray:
    x = np.ones(T, dtype=np.float64)
    x[32:] = -1.0
    return x


def y_sequence() -> np.ndarray:
    i = np.arange(T)
    return (-1.0) ** i


def xi_sequence(arm: str) -> np.ndarray:
    """Direction-process xi_t (4-vector per step, vec row-major) for one full period, index 0..63.

    arm1: xi[i] = x[i]*Bs + y[i]*Ba
    arm2: xi[i] = y[i]*Bs + x[i]*Ba
    """
    x, y = x_sequence(), y_sequence()
    out = np.zeros((T, 2, 2), dtype=np.float64)
    for i in range(T):
        a, b = (x[i], y[i]) if arm == "arm1" else (y[i], x[i])
        out[i] = a * BS + b * BA
    return out


def arm_inputs(arm: str, phase: int, epsilon: float) -> np.ndarray:
    """G(q, t) for t=1..64 -> array (T,2,2), i=(t-1+q)%64."""
    xi = xi_sequence(arm)
    g = np.empty((T, 2, 2), dtype=np.float64)
    for t in range(1, T + 1):
        i = (t - 1 + phase) % T
        g[t - 1] = S + epsilon * xi[i]
    return g


def multiset_sha(arm: str, phase: int, epsilon: float) -> str:
    """Order-independent multiset hash of one arm/phase input set (exact symbol labels)."""
    g = arm_inputs(arm, phase, epsilon)
    if epsilon == 0.0:
        # all matrices identical S; hash bytes of S once repeated T times deterministically
        blob = b"".join(g[i].tobytes() for i in range(T))
    else:
        xi = xi_sequence(arm)
        # symbol multiset: the xi sign-pair labels; order-independent
        labels = []
        for t in range(1, T + 1):
            i = (t - 1 + phase) % T
            labels.append(xi[i].tobytes())
        blob = b"".join(sorted(labels))
    return hashlib.sha256(blob).hexdigest()


def symbol_counts(arm: str, phase: int) -> dict:
    """Count of the four (sign_Bs, sign_Ba) combinations in one arm/phase sequence."""
    xi = xi_sequence(arm)
    counts = {}
    for t in range(1, T + 1):
        i = (t - 1 + phase) % T
        a = 1 if xi[i][0, 0] > 0 else -1  # coefficient of Bs
        b = 1 if xi[i][1, 0] > 0 else -1  # coefficient of Ba (entry [1,0])
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def scalar_autocorr(arm: str) -> np.ndarray:
    """C_scalar(k) = mean_t <xi_t, xi_{t-k}>_F over the period (circular index), k=0..63."""
    xi = xi_sequence(arm).reshape(T, 4)
    nrm = np.array([np.sum(xi[t] * xi[t]) for t in range(T)])
    out = np.zeros(T)
    for k in range(T):
        vals = [np.sum(xi[t] * xi[(t - k) % T]) for t in range(T)]
        out[k] = float(np.mean(vals))
    return out


def direction_lag_covariance(arm: str) -> np.ndarray:
    """C_xi(k) = mean_t vec(xi_t) vec(xi_{t-k})^T (4x4 per lag), circular index."""
    xi = xi_sequence(arm).reshape(T, 4)
    out = np.zeros((T, 4, 4))
    for k in range(T):
        acc = np.zeros((4, 4))
        for t in range(T):
            acc += np.outer(xi[t], xi[(t - k) % T])
        out[k] = acc / T
    return out
