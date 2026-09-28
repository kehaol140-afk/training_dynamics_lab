"""FR B0.5 mechanism identification: pure analysis math (offline, no GPU).

Single source of truth for the H1/H2/H3 constructions in
outputs/fr_b05_mechanism_identification_v01/config.json.
Runner (b05_run_v01.py) and analyzer (b05_analyze_v01.py) both import from here.

Terminology discipline (config.json): controlled non-equilibrium dynamical
system; quasi-steady state only if numbers warrant; memory kernel / response
function language for H3. No banned terms in any output string produced here.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# H1: cumulative LR time s(t) and alignment
# ---------------------------------------------------------------------------

def cumulative_lr_time(lr_native, multipliers):
    """s(t) = sum_{k<=t} multipliers[k]*lr_native[k], t = 1..T (list index 0..T-1)."""
    assert len(lr_native) == len(multipliers)
    out, acc = [], 0.0
    for lr, m in zip(lr_native, multipliers):
        acc += float(m) * float(lr)
        out.append(acc)
    return out


def invert_monotone(s_ref, t_grid, s_query):
    """t̃ with s_ref(t̃) = s_query by linear interpolation on (t_grid, s_ref).

    s_ref must be strictly increasing. s_query outside [s_ref[0], s_ref[-1]]
    is clamped to the edge and flagged via the returned dict's 'clamped' field.
    Returns dict(t=float, clamped=bool).
    """
    n = len(s_ref)
    assert n == len(t_grid) >= 2
    for i in range(n - 1):
        assert s_ref[i + 1] > s_ref[i], "s_ref must be strictly increasing"
    sq = float(s_query)
    if sq <= s_ref[0]:
        return {"t": float(t_grid[0]), "clamped": sq < s_ref[0]}
    if sq >= s_ref[-1]:
        return {"t": float(t_grid[-1]), "clamped": sq > s_ref[-1]}
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s_ref[mid] <= sq:
            lo = mid
        else:
            hi = mid
    frac = (sq - s_ref[lo]) / (s_ref[hi] - s_ref[lo])
    t = float(t_grid[lo]) + frac * (float(t_grid[hi]) - float(t_grid[lo]))
    return {"t": t, "clamped": False}


def align_loss_by_s(t_grid, s_A, L_A, s_X):
    """Predict arm-X loss by reparameterizing the native curve: for each t,
    t̃ = s_A^{-1}(s_X(t)); L̃_X(t) = interp(L_A, t̃). Returns (L_tilde, clamped_mask)
    where clamped_mask[i] is True where s_X[i] fell outside [s_A[0], s_A[-1]]
    and the prediction is an edge clamp (callers must exclude these indices
    from alignment metrics)."""
    assert len(t_grid) == len(s_A) == len(L_A) == len(s_X)
    clamped = []
    L_tilde = []
    for t_i, sq in enumerate(s_X):
        inv = invert_monotone(s_A, t_grid, sq)
        clamped.append(inv["clamped"])
        tq = inv["t"]
        # interpolate L_A at tq (t_grid equally spaced by construction but do general)
        if tq <= t_grid[0]:
            L_tilde.append(float(L_A[0]))
        elif tq >= t_grid[-1]:
            L_tilde.append(float(L_A[-1]))
        else:
            j = 0
            while t_grid[j + 1] < tq:
                j += 1
            frac = (tq - t_grid[j]) / (t_grid[j + 1] - t_grid[j])
            L_tilde.append(float(L_A[j]) + frac * (float(L_A[j + 1]) - float(L_A[j])))
    return L_tilde, clamped


def masked_rel_err(L_X, L_tilde, L_A, mask):
    """alignment_rel_err over indices where mask is False (non-clamped)."""
    xs = [x for x, m in zip(L_X, mask) if not m]
    ts = [x for x, m in zip(L_tilde, mask) if not m]
    bs = [x for x, m in zip(L_A, mask) if not m]
    return alignment_rel_err(xs, ts, bs)


def alignment_rel_err(L_X, L_tilde, L_A):
    """||L_X - L_tilde|| / ||L_X - L_A|| (same-time baseline in denominator)."""
    num = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(L_X, L_tilde)))
    den = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(L_X, L_A)))
    if den == 0.0:
        return float("nan")
    return num / den


def implied_lag(s_A, t_grid, s_X):
    """l(t) = s_A^{-1}(s_X(t)) - t. Constant post-release under a frozen-excess-LR-time arm."""
    return [invert_monotone(s_A, t_grid, sq)["t"] - float(t_grid[i])
            for i, sq in enumerate(s_X)]


# ---------------------------------------------------------------------------
# H2: slow-fast split of the native trajectory
# ---------------------------------------------------------------------------

def smooth_component(theta_seq, w):
    """Centered moving average of a per-step sequence (each element any nested
    numeric sequence flattened by the caller OR a 1-D list of scalars).
    Window shrinks at the edges (divide by actual count). w must be odd.
    Returns list same length as input."""
    assert w % 2 == 1 and w >= 3
    n = len(theta_seq)
    half = w // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        acc = [0.0] * len(theta_seq[0])
        for j in range(lo, hi):
            for d, x in enumerate(theta_seq[j]):
                acc[d] += float(x)
        cnt = hi - lo
        out.append([a / cnt for a in acc])
    return out


def velocities(theta_seq):
    """v_t = theta_t - theta_{t-1}, t=1..n-1 (list index 0..n-2)."""
    return [[float(b) - float(a) for a, b in zip(theta_seq[t], theta_seq[t + 1])]
            for t in range(len(theta_seq) - 1)]


def normalized_autocorr(vels, max_lag):
    """C_v(D) = sum_t <v_t, v_{t+D}> / sum_t <v_t, v_t>, D=1..max_lag."""
    n = len(vels)
    denom = sum(sum(x * x for x in v) for v in vels)
    out = []
    for d in range(1, max_lag + 1):
        if n - d < 2 or denom == 0.0:
            out.append(float("nan"))
            continue
        num = 0.0
        for t in range(n - d):
            num += sum(a * b for a, b in zip(vels[t], vels[t + d]))
        out.append(num / denom)
    return out


def direction_persistence(smooth_vels, max_lag):
    """mean_t cos(s_t, s_{t+D}) over valid t, D=1..max_lag (nan if undefined)."""
    n = len(smooth_vels)
    out = []
    for d in range(1, max_lag + 1):
        if n - d < 1:
            out.append(float("nan"))
            continue
        vals = []
        for t in range(n - d):
            a, b = smooth_vels[t], smooth_vels[t + d]
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na > 0.0 and nb > 0.0:
                vals.append(sum(x * y for x, y in zip(a, b)) / (na * nb))
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def tau_from_curve(curve, threshold=1.0 / math.e):
    """First lag (1-indexed) where curve < threshold; None if no crossing.
    NaN entries are skipped (treated as not-crossed)."""
    for i, c in enumerate(curve):
        if not (c == c):  # NaN
            continue
        if c < threshold:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# H3: channel decomposition bookkeeping
# ---------------------------------------------------------------------------

def mv_channel_share(G_C, G_D):
    """m/v-memory-channel share of retained pulse gain:
    share = (G_C - G_D) / G_C with G_X = L_A - L_X. Guards: if |G_C| below
    resolution, returns None (UNRESOLVED)."""
    if abs(G_C) < 1e-12:
        return None
    return (G_C - G_D) / G_C


# ---------------------------------------------------------------------------
# Distance / cosine helpers (fp32 accumulation semantics match the runner)
# ---------------------------------------------------------------------------

def flat_sq_norm(vec):
    return sum(float(x) * float(x) for x in vec)


def cosine(a, b):
    na = math.sqrt(flat_sq_norm(a))
    nb = math.sqrt(flat_sq_norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)
