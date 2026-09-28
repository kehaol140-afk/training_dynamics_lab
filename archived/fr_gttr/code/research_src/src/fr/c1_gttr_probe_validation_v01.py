"""FR-C1 GTTR probe validation: pure streaming statistics + error decomposition.

Zero GPU, no model state. Vector algebra reuses fr.c0_gttr_validation_v01 (frozen
in C0). Everything here is deterministic CPU math on tensor-dicts.

Key structures:
- GroupAcc: streaming per-probe group statistics (count, sum vector, sum of
  squared norms) with fixed-prefix snapshots (K=8/16/32) and fixed-ID half-bank
  accumulators (first16/last16). From these, ALL predetermined K/half-bank/
  aggregate statistics and error cross terms are recoverable WITHOUT storing
  per-probe vectors (protocol §11b/§11d).
- decompose: exact field/shadow error split e = e_bank + e_local with the
  cross term kept (protocol §7.2).
- d_c_to_u: local c->U derivative D_i (protocol §9), evaluated with the same
  shadow state semantics as c0.u_map.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

import torch

import fr.c0_gttr_validation_v01 as c0

TensorDict = Dict[str, object]

K_PREFIXES = (8, 16, 32)
HALF_IDS = ("first16", "last16")

_HOST_FP64 = {"device": "cpu", "dtype": torch.float64}


class GroupAcc:
    """Streaming accumulator for a bank of per-probe vector-dicts.

    add(x, j): x is one probe's tensor-dict; j its fixed bank ID (0-based).
    All state lives on HOST fp64 regardless of x's device (GPU-memory
    discipline: the runner offloads everything long-lived to CPU RAM).
    Prefix snapshots at K in K_PREFIXES are taken when the K-th add lands.
    Half-bank accumulators split on fixed ID j<half_split vs j>=half_split
    (never re-selected).
    """

    def __init__(self, k_prefixes=K_PREFIXES, half_split=16):
        self.k_prefixes = tuple(k_prefixes)
        self.half_split = int(half_split)
        self.n = 0
        self.sum = None          # tensor-dict of running sums (host fp64)
        self.sumsq = 0.0         # sum of squared norms
        self._prefix_snaps = {}
        self._halves = {h: {"n": 0, "sum": None, "sumsq": 0.0} for h in HALF_IDS}

    def add(self, x: TensorDict, j: int):
        if self.sum is None:
            self.sum = {n: v.detach().to(**_HOST_FP64).clone() for n, v in x.items()}
        else:
            for n, v in x.items():
                self.sum[n] += v.detach().to(**_HOST_FP64)
        self.sumsq += float(sum((v.detach().to(**_HOST_FP64) ** 2).sum()
                                for v in x.values()))
        self.n += 1
        if self.n in self.k_prefixes:
            self._prefix_snaps[self.n] = self._snap()
        h = HALF_IDS[0] if int(j) < self.half_split else HALF_IDS[1]
        hb = self._halves[h]
        if hb["sum"] is None:
            hb["sum"] = {n: v.detach().to(**_HOST_FP64).clone() for n, v in x.items()}
        else:
            for n, v in x.items():
                hb["sum"][n] += v.detach().to(**_HOST_FP64)
        hb["sumsq"] += float(sum((v.detach().to(**_HOST_FP64) ** 2).sum()
                                 for v in x.values()))
        hb["n"] += 1

    def _snap(self):
        return {"n": self.n,
                "sum": {n: v.clone() for n, v in self.sum.items()},
                "sumsq": self.sumsq}

    def snapshot(self):
        return self._snap()

    def prefixes(self):
        return {k: self._prefix_snaps[k] for k in self.k_prefixes
                if k in self._prefix_snaps}

    def halves(self):
        return {h: {"n": self._halves[h]["n"],
                    "sum": (None if self._halves[h]["sum"] is None
                            else {n: v.clone() for n, v in self._halves[h]["sum"].items()}),
                    "sumsq": self._halves[h]["sumsq"]}
                for h in HALF_IDS}


def mean_of(snap) -> TensorDict:
    return c0.scale_td(snap["sum"], 1.0 / snap["n"])


def mean_norm(snap) -> float:
    """||xbar|| for the group."""
    return c0.norm_td(mean_of(snap))


def mean_sq_norm(snap) -> float:
    """||xbar||^2 = ||sum||^2 / n^2."""
    s = snap["sum"]
    return c0.dot_td(s, s) / (snap["n"] ** 2)


def se_vec_of(snap) -> float:
    """SE_vec = sqrt( (sum||x||^2 - ||sum||^2/n) / (n(n-1)) )."""
    n = snap["n"]
    if n < 2:
        return float("nan")
    ssq_centered = snap["sumsq"] - c0.dot_td(snap["sum"], snap["sum"]) / n
    ssq_centered = max(ssq_centered, 0.0)
    return math.sqrt(ssq_centered / (n * (n - 1)))


def inner_means(snap_a, snap_b) -> float:
    """<xbar, ybar> = <S_x, S_y> / (n_x n_y) without storing means."""
    return c0.dot_td(snap_a["sum"], snap_b["sum"]) / (snap_a["n"] * snap_b["n"])


def sse_error(snap_p, snap_y) -> float:
    """||pbar - ybar||^2 expanded through group sums (no mean formed)."""
    return (mean_sq_norm(snap_p) + mean_sq_norm(snap_y)
            - 2.0 * inner_means(snap_p, snap_y))


def E_over_anchors(per_anchor) -> float:
    """Frozen aggregate E over a fixed anchor set:
    sqrt( sum_a SSE_a / sum_a TSS_a )."""
    num = sum(a["sse"] for a in per_anchor)
    den = sum(a["tss"] for a in per_anchor)
    if den <= 0:
        return float("nan")
    return math.sqrt(num / den)


def decompose(p_d_bar, y_a_bar, p_a_bar):
    """Exact field/shadow decomposition (protocol §7.2).

    e      = P_D_bar - Y_A_bar      (independent-bank prediction error)
    e_bank = P_D_bar - P_A_bar      (cross-bank difference of predictions)
    e_local= P_A_bar - Y_A_bar      (same-probe approximation defect)
    Returns dict with vectors + norms + cross term + identity residuals.
    """
    e = c0.sub_td(p_d_bar, y_a_bar)
    e_bank = c0.sub_td(p_d_bar, p_a_bar)
    e_local = c0.sub_td(p_a_bar, y_a_bar)
    out = {"e": e, "e_bank": e_bank, "e_local": e_local,
           "e_norm": c0.norm_td(e), "e_bank_norm": c0.norm_td(e_bank),
           "e_local_norm": c0.norm_td(e_local),
           "cross": c0.dot_td(e_bank, e_local)}
    resid = c0.sub_td(e, c0.add_td(e_bank, e_local))
    out["add_identity_resid"] = c0.norm_td(resid)
    sq_resid = (out["e_norm"] ** 2
                - (out["e_bank_norm"] ** 2 + out["e_local_norm"] ** 2
                   + 2.0 * out["cross"]))
    out["sq_identity_resid"] = sq_resid
    return out


def fixed_direction_stats(projections):
    """projections: list of per-probe scalars <x_j, d>. Returns n, mean, SE."""
    n = len(projections)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    mean = sum(projections) / n
    if n < 2:
        return {"n": n, "mean": mean, "se": float("nan")}
    var = sum((p - mean) ** 2 for p in projections) / (n * (n - 1))
    return {"n": n, "mean": mean, "se": math.sqrt(var)}


def d_c_to_u(*, c, m_hat, v_hat, beta1, beta2, t_prev, eps, lr):
    """Local c->U derivative at fixed common state (protocol §9), per name:

        D_i = -eta * [ (1-a) P_i  -  (1-b) m_i c_i / ( sqrt(v_i) (sqrt(v_i)+eps)^2 ) ]

    with a/b the NEXT-step bias-correction coefficients (c0.ema_coeffs) and
    m_i/v_i/P_i the POST-update shadow quantities of this probe's baseline U:
        m1 = a m + (1-a) c ; v1 = b_v-form ; P = 1/(sqrt(v1)+eps).
    Valid where v1_i > 0 (else coordinate marked NaN, not smoothed).
    """
    torch = c0._torch()
    a1, b1 = c0.ema_coeffs(beta1, t_prev)
    a2, b2 = c0.ema_coeffs(beta2, t_prev)
    out = {}
    for n in m_hat:
        cm = c[n].detach().to(torch.float64)
        m1 = a1 * m_hat[n].detach().to(torch.float64) + b1 * cm
        v1 = a2 * v_hat[n].detach().to(torch.float64) + b2 * cm * cm
        sqrt_v = torch.sqrt(v1)
        P = 1.0 / (sqrt_v + float(eps))
        denom = sqrt_v * (sqrt_v + float(eps)) ** 2
        safe = denom > 0
        term = torch.where(
            safe,
            (b1 * P - b2 * m1 * cm / torch.where(safe, denom, torch.ones_like(denom))),
            torch.full_like(denom, float("nan")))
        out[n] = -float(lr) * term
    return out


def shadow_first_order_readout(q_plus, x_bar):
    """-q+^T x_bar (SHADOW_FIRST_ORDER_READOUT; not a loss, no curvature)."""
    return -c0.dot_td(q_plus, x_bar)


def w_k_gain_identity(dW_list, dK_list, v_native_change, v_arm_change):
    """Sign-convention check: gain = sum(dW - dK) must equal
    (V_native_change - V_arm_change)."""
    gain = sum(dW_list) - sum(dK_list)
    direct = v_native_change - v_arm_change
    return gain, direct, gain - direct
