"""T0 metrics: phase statistics, model predictions, pulse bookkeeping."""
from __future__ import annotations

import numpy as np

from .t0_construct import T
from .t0_maps import filter_covariance, kernel_weights


def phase_stats(d: np.ndarray) -> dict:
    """d: (n_phase, T, 2, 2). Population statistics over the finite phase enumeration.

    ERRATUM r1: covariance now divides by n_phase (complete finite population), matching
    the A definition; the original round-1 code divided by n_phase-1 (see
    docs/ROUND1_ERRATUM.md). A == mean_t trace(cov_t) now holds exactly.
    """
    n_phase = d.shape[0]
    vec = d.reshape(n_phase, T, 4)
    dbar = vec.mean(axis=0)  # (T,4)
    cen = vec - dbar[None]
    a_total = float(np.mean(np.sum(cen ** 2, axis=2)))  # mean_t mean_phase ||.||^2
    cov_t = np.einsum("qti,qtj->tij", cen, cen) / n_phase  # population cross-phase cov
    cov_mean = cov_t.mean(axis=0)
    return {"dbar": dbar.reshape(T, 2, 2), "A": a_total,
            "cov_t": cov_t, "cov_mean": cov_mean}


def bias_decomposition(d: np.ndarray, d0: np.ndarray) -> dict:
    """Mean-variance decomposition against a same-path zero-perturbation baseline D0.

    B2 = mean_t ||mean_phase(D_t) - D0||_F^2 ; total_mse = A + B2 exactly.
    d: (n_phase, T, 2, 2); d0: (T, 2, 2) baseline trajectory or its phase mean is NOT used —
    d0 must be the per-time baseline map output on the constant input (here time-invariant).
    """
    st = phase_stats(d)
    d0v = np.broadcast_to(np.asarray(d0, dtype=np.float64).reshape(2, 2), (T, 2, 2))
    diff = st["dbar"] - d0v  # (T,2,2)
    b2 = float(np.mean(np.sum(diff.reshape(T, 4) ** 2, axis=1)))
    n_phase = d.shape[0]
    vec = d.reshape(n_phase, T, 4)
    d0_rep = np.broadcast_to(d0v.reshape(1, T, 4), (n_phase, T, 4))
    total = float(np.mean(np.sum((vec - d0_rep) ** 2, axis=2)))
    b1 = float(np.mean(np.linalg.norm(diff.reshape(T, 4), axis=1)))  # mean_t ||bias_t|| (norm)
    return {"A": st["A"], "B2": b2, "total_mse": total, "B1_mean_norm": b1,
            "identity_ok": abs(st["A"] + b2 - total) <= 1e-9 * max(total, 1e-30) + 1e-14}


def same_H_error_decomposition(d_ref: np.ndarray, d_native: np.ndarray) -> dict:
    """Decompose native same-H output variance: D_native = D_ref + q.

    A_native = A_ref + A_q + 2*cross (all population, same phase/time centering);
    mean_q reported separately. Identity holds exactly by construction.
    """
    n_phase = d_ref.shape[0]
    vr = d_ref.reshape(n_phase, T, 4)
    vn = d_native.reshape(n_phase, T, 4)
    q = vn - vr
    cr = vr - vr.mean(axis=0)
    cq = q - q.mean(axis=0)
    a_r = float(np.mean(np.sum(cr ** 2, axis=2)))
    a_q = float(np.mean(np.sum(cq ** 2, axis=2)))
    cross2 = 2.0 * float(np.mean(np.sum(cr * cq, axis=2)))
    a_n = float(np.mean(np.sum((cr + cq) ** 2, axis=2)))
    return {"A_ref": a_r, "A_q": a_q, "cross2": cross2, "A_native": a_n,
            "mean_q": float(q.mean()),
            "identity_ok": abs(a_r + a_q + cross2 - a_n) <= 1e-9 * max(a_n, 1e-30) + 1e-14}


def decompose_pair(d_ref: np.ndarray, d_native: np.ndarray) -> dict:
    """Generic population same-input decomposition for arbitrary (n, ...) arrays.

    Same formulas and conventions as same_H_error_decomposition (shared statistics
    entry for FP64 and native data); n is the sample axis (e.g. batches).
    """
    vr = d_ref.reshape(d_ref.shape[0], -1)
    vn = d_native.reshape(d_native.shape[0], -1)
    q = vn - vr
    cr = vr - vr.mean(axis=0)
    cq = q - q.mean(axis=0)
    a_r = float(np.mean(np.sum(cr ** 2, axis=1)))
    a_q = float(np.mean(np.sum(cq ** 2, axis=1)))
    cross2 = 2.0 * float(np.mean(np.sum(cr * cq, axis=1)))
    a_n = float(np.mean(np.sum((cr + cq) ** 2, axis=1)))
    return {"A_ref": a_r, "A_q": a_q, "cross2": cross2, "A_native": a_n,
            "mean_q": float(q.mean()),
            "identity_ok": abs(a_r + a_q + cross2 - a_n) <= 1e-9 * max(a_n, 1e-30) + 1e-14}


def empirical_C_G(arm_xi: np.ndarray) -> np.ndarray:
    """Cross-phase two-time covariance of the direction process at each lag (periodic).

    For each time t, xi at position i=(t-1+q)%64 over q gives the empirical (population)
    one-time covariance; two-time C(t,s) depends only on (t-s) mod 64. We return
    c[k] = mean_t Cov_phase(vec(G_t), vec(G_{t-k})) (population, divide by n_phase).
    """
    n_phase, L = T, T
    xi = arm_xi.reshape(L, 4)
    # position-marginal mean is the same for every t (phase enumeration covers all i)
    mu = np.zeros(4)  # symbols are +/- pairs, mean over full period is zero
    c = np.zeros((L, 4, 4))
    for k in range(L):
        acc = np.zeros((4, 4))
        # For phase-enumerated population: Cov(vec xi_t, vec xi_{t-k}) with t fixed,
        # q varying: entries i_t=(t-1+q)%64, i_s=(t-k-1+q)%64, difference fixed = k.
        # So C(t, t-k) over phases equals outer products over positions j of xi_j vs xi_{j-k}.
        for j in range(L):
            acc += np.outer(xi[j] - mu, xi[(j - k) % L] - mu)
        c[k] = acc / L
    return c


def direction_model_A(c_xi: np.ndarray, J: np.ndarray, beta: float) -> dict:
    """M_direction: A_pred = mean_t tr(J C_H(t) J^T), plus per-time C_D_pred."""
    c_H = filter_covariance(c_xi, beta)
    cD = np.einsum("ik,tkj,lj->til", J, c_H, J)
    a_pred = float(np.mean(np.trace(cD, axis1=1, axis2=2)))
    return {"A_pred": a_pred, "cD_t": cD}


def scalar_model_A(c_xi: np.ndarray, J: np.ndarray, beta: float) -> dict:
    """M_scalar: r_ts = tr(C_G(t,s))/tr(Sigma0); C_G_scalar(t,s)=r_ts*Sigma0; same filter+J."""
    L = c_xi.shape[0]
    Sigma0 = c_xi[0]
    tr0 = float(np.trace(Sigma0))
    w = kernel_weights(beta, L + 1)
    # scalar-lag correlations: rho_k = tr(c[k])/tr0 (c[k] symmetric-consistent population cov)
    rho = np.array([float(np.trace(c_xi[k])) / tr0 for k in range(L)])
    # scalar-filtered Sigma: sum_{i,j<t} w_i w_j rho_{(i-j)%L} * Sigma0
    factors = np.empty(L)
    for t in range(1, L + 1):
        acc = 0.0
        for i in range(t):
            for j in range(t):
                acc += w[i] * w[j] * rho[(i - j) % L]
        factors[t - 1] = acc
    cD = factors[:, None, None] * (J @ Sigma0 @ J.T)[None, :, :]
    a_pred = float(np.mean(np.trace(cD, axis1=1, axis2=2)))
    return {"A_pred": a_pred, "cD_t": cD, "rho": rho}


def mean_shift_stats(d: np.ndarray, map_ref: np.ndarray) -> dict:
    """B_j(t) = Dbar_t - map(S); returns max/mean Frobenius norm."""
    st = phase_stats(d)
    diff = st["dbar"] - map_ref[None]
    norms = np.linalg.norm(diff.reshape(T, 4), axis=1)
    return {"B_mean": float(norms.mean()), "B_max": float(norms.max()),
            "dbar_first": st["dbar"][0].tolist(), "dbar_last": st["dbar"][-1].tolist()}
