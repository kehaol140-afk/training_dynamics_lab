"""FR B0 mechanism validation (core pure module: theory tests + statistics).

No GPU code here. Everything the driver needs that is pure math lives here:

Theory tests (protocol Stage B, CPU, FP64):
  B1  slow-progress + transverse-fluctuation toy model: deterministic q recursion
      + central FD vs analytic chi_H; catches exp(u)->1+u, q0(u) drift,
      pulse4/persistent confusion, H off-by-one, Gain sign errors.
  B2  steady state != detailed balance: 2-D Gaussian AR(1), A_lag = M Sigma - Sigma M^T.
  B3  decision value under linear response (V_sign) and quadratic (u_star clip).
  B4  finite-ensemble decomposition identity L_ens = L_center + J_disp and its FD.

Analysis (protocol Stages E/F): chi/Q from class-mean losses, dev/audit split,
pulse-vs-persistent duration contrast, V_sign, V_grid hindsight bound,
identifiability classification. All functions are deterministic given records.
"""
from __future__ import annotations

import numpy as np

U_GRID = (-0.05, -0.025, 0.0, 0.025, 0.05)
H_GRID = (4, 16, 64)
NONZERO_U = (-0.05, -0.025, 0.025, 0.05)


# ---------------------------------------------------------------------------
# B1 toy model
# ---------------------------------------------------------------------------

def toy_q_recursion(eta0: float, u: float, kappa: float, sigma: float,
                    H: int, q0: float, pulse_steps: int | None = None) -> float:
    """q_H from q_{k+1}=(1-eta*kappa)^2 q_k + eta^2 sigma^2, FP64.

    pulse_steps=None -> persistent (eta=eta0*exp(u) at every step).
    pulse_steps=p -> eta=eta0*exp(u) only for steps 1..p, native afterwards.
    q0 is FIXED (never re-derived under u) per protocol.
    """
    eta_native = float(eta0)
    eta = float(eta0) * np.exp(float(u))
    q = float(q0)
    for k in range(1, int(H) + 1):
        e = eta if (pulse_steps is None or k <= pulse_steps) else eta_native
        q = (1.0 - e * kappa) ** 2 * q + e ** 2 * sigma ** 2
    return q


def toy_x_H(x0: float, eta0: float, u: float, a: float, H: int,
            pulse_steps: int | None = None) -> float:
    """x_H = x0 + sum_k eta_k * a (deterministic slow progress)."""
    eta_native = float(eta0)
    eta = float(eta0) * np.exp(float(u))
    x = float(x0)
    for k in range(1, int(H) + 1):
        e = eta if (pulse_steps is None or k <= pulse_steps) else eta_native
        x += e * a
    return x


def toy_L(x0, y0_unused, q0, eta0, u, a, kappa, sigma, H, L_star,
          pulse_steps=None):
    """E[L_H(u)] = L_star - a*x_H + (kappa/2)*q_H  (E[y]=0 by symmetry; the
    deterministic q recursion IS the variance of y)."""
    x_H = toy_x_H(x0, eta0, u, a, H, pulse_steps)
    q_H = toy_q_recursion(eta0, u, kappa, sigma, H, q0, pulse_steps)
    return L_star - a * x_H + 0.5 * kappa * q_H


def analytic_chi(H: int, eta0: float, a: float, kappa: float, sigma: float) -> float:
    """chi_H = -H*eta0*a^2 + eta0*sigma^2/(2-eta0*kappa)^2 * [1-(1-eta0*kappa)^(2H)]."""
    H = int(H)
    r = 1.0 - eta0 * kappa
    return (-H * eta0 * a ** 2
            + eta0 * sigma ** 2 / (2.0 - eta0 * kappa) ** 2 * (1.0 - r ** (2 * H)))


def b1_toy_battery() -> dict:
    """B1 checks. Deterministic q recursion + central FD vs analytic."""
    eta0, a, kappa, sigma = 0.1, 0.15, 1.0, 1.0
    q0 = eta0 * sigma ** 2 / (kappa * (2.0 - eta0 * kappa))
    x0, L_star = 0.0, 2.0
    fd_a = 1e-6  # FD probe amplitude for the TOY (analytic check), not the GPU delta
    checks = []
    for H in (4, 16, 64):
        chi_ana = analytic_chi(H, eta0, a, kappa, sigma)
        lp = toy_L(x0, 0.0, q0, eta0, +fd_a, a, kappa, sigma, H, L_star)
        lm = toy_L(x0, 0.0, q0, eta0, -fd_a, a, kappa, sigma, H, L_star)
        chi_fd = (lp - lm) / (2 * fd_a)
        checks.append({
            "H": H, "chi_analytic": float(chi_ana), "chi_fd": float(chi_fd),
            "abs_err": float(abs(chi_ana - chi_fd)),
            "sign": int(np.sign(chi_ana)),
        })
    signs = {c["H"]: c["sign"] for c in checks}
    # sigma=0: only the negative progress term remains
    chi_s0 = analytic_chi(16, eta0, a, kappa, 0.0)
    # Gain sign: Gain(u) = L(0)-L(u); for u>0 with chi>0 at H=4, Gain>0
    L0 = toy_L(x0, 0.0, q0, eta0, 0.0, a, kappa, sigma, 4, L_star)
    Lp = toy_L(x0, 0.0, q0, eta0, +0.01, a, kappa, sigma, 4, L_star)
    gain = L0 - Lp
    gain_sign_ok = bool(np.sign(gain) == np.sign(-analytic_chi(4, eta0, a, kappa, sigma) * 0.01))
    # exp(u) vs 1+u trap: recompute FD under the WRONG 1+u law, must mismatch
    def toy_L_wrong(x0, q0, eta0, u, a, kappa, sigma, H):
        x, q = x0, q0
        for k in range(1, H + 1):
            e = eta0 * (1.0 + u)
            x += e * a
            q = (1.0 - e * kappa) ** 2 * q + e ** 2 * sigma ** 2
        return L_star - a * x + 0.5 * kappa * q
    # use the production-scale probe (0.05): at 1e-6 the two laws are
    # indistinguishable, which is exactly why the trap is dangerous
    pw = 0.05
    chi_ana16 = analytic_chi(16, eta0, a, kappa, sigma)
    lp_w = toy_L_wrong(x0, q0, eta0, +pw, a, kappa, sigma, 16)
    lm_w = toy_L_wrong(x0, q0, eta0, -pw, a, kappa, sigma, 16)
    lp_c = toy_L(x0, 0.0, q0, eta0, +pw, a, kappa, sigma, 16, L_star)
    lm_c = toy_L(x0, 0.0, q0, eta0, -pw, a, kappa, sigma, 16, L_star)
    chi_wrong = (lp_w - lm_w) / (2 * pw)
    chi_correct = (lp_c - lm_c) / (2 * pw)
    # the 1+u law agrees with exp(u) to FIRST order at u=0 (both d eta/du =
    # eta0), so small-probe FD cannot see it; it inflates the finite-amplitude
    # FD bias instead. Detector: wrong law's FD bias vs analytic exceeds the
    # correct law's FD bias (pure O(a^2) truncation) by a clear factor.
    bias_c = abs(chi_correct - chi_ana16)
    bias_w = abs(chi_wrong - chi_ana16)
    wrong_law_detected = bool(bias_w > 1.5 * bias_c + 1e-9)
    # q0(u) drift trap: re-derive q0 under eta(u) -> response contaminated
    def q0_of(eta):
        return eta * sigma ** 2 / (kappa * (2.0 - eta * kappa))
    lp_q = toy_L(x0, 0.0, q0_of(eta0 * np.exp(+fd_a)), eta0, +fd_a, a, kappa, sigma, 16, L_star)
    lm_q = toy_L(x0, 0.0, q0_of(eta0 * np.exp(-fd_a)), eta0, -fd_a, a, kappa, sigma, 16, L_star)
    chi_qdrift = (lp_q - lm_q) / (2 * fd_a)
    qdrift_detected = bool(abs(chi_qdrift - analytic_chi(16, eta0, a, kappa, sigma)) > 1e-4)
    # pulse4 vs persistent confusion: at H=4 they coincide, at H=64 differ
    p4_h4 = toy_L(x0, 0.0, q0, eta0, 0.02, a, kappa, sigma, 4, L_star, pulse_steps=4)
    pe_h4 = toy_L(x0, 0.0, q0, eta0, 0.02, a, kappa, sigma, 4, L_star)
    p4_h64 = toy_L(x0, 0.0, q0, eta0, 0.02, a, kappa, sigma, 64, L_star, pulse_steps=4)
    pe_h64 = toy_L(x0, 0.0, q0, eta0, 0.02, a, kappa, sigma, 64, L_star)
    pulse_coincide_h4 = bool(abs(p4_h4 - pe_h4) < 1e-12)
    pulse_differ_h64 = bool(abs(p4_h64 - pe_h64) > 1e-6)
    # H off-by-one trap: chi at "H" computed with H-1 steps must mismatch
    lp_m1 = toy_L(x0, 0.0, q0, eta0, +fd_a, a, kappa, sigma, 15, L_star)
    lm_m1 = toy_L(x0, 0.0, q0, eta0, -fd_a, a, kappa, sigma, 15, L_star)
    offbyone_detected = bool(abs((lp_m1 - lm_m1) / (2 * fd_a) - analytic_chi(16, eta0, a, kappa, sigma)) > 1e-3)
    ok = (all(c["abs_err"] < 1e-8 for c in checks)
          and signs[4] > 0 and signs[16] < 0 and signs[64] < 0
          and chi_s0 < 0 and gain_sign_ok and wrong_law_detected
          and qdrift_detected and pulse_coincide_h4 and pulse_differ_h64
          and offbyone_detected)
    return {"name": "B1_toy_progress_transverse", "ok": bool(ok),
            "params": {"eta0": eta0, "a": a, "kappa": kappa, "sigma": sigma,
                       "q0": q0, "fd_probe": fd_a},
            "chi_by_H": checks,
            "H4_sign": signs[4], "H16_sign": signs[16], "H64_sign": signs[64],
            "sigma0_chi_H16": float(chi_s0),
            "gain_sign_ok": gain_sign_ok,
            "wrong_1plusu_law_detected": wrong_law_detected,
            "wrong_1plusu_fd_bias": float(bias_w),
            "correct_exp_fd_bias": float(bias_c),
            "q0_u_drift_detected": qdrift_detected,
            "pulse4_eq_persistent_at_H4": pulse_coincide_h4,
            "pulse4_differs_persistent_at_H64": pulse_differ_h64,
            "offbyone_detected": offbyone_detected}


# ---------------------------------------------------------------------------
# B2 steady state vs detailed balance
# ---------------------------------------------------------------------------

def b2_lag_antisymmetry() -> dict:
    Xi = 0.36 * np.eye(2)
    Sigma = np.eye(2)
    rot = np.array([[np.cos(0.35), -np.sin(0.35)],
                    [np.sin(0.35), np.cos(0.35)]])
    mats = {"0.8I": 0.8 * np.eye(2),
            "diag(0.8,-0.8)": np.diag([0.8, -0.8]),
            "0.8Rot(0.35)": 0.8 * rot}
    out = {}
    ok_all = True
    for name, M in mats.items():
        lhs = M @ Sigma @ M.T + Xi
        lyapunov_holds = bool(np.max(np.abs(lhs - Sigma)) < 1e-12)
        A_lag = M @ Sigma - Sigma @ M.T
        a_norm = float(np.max(np.abs(A_lag)))
        out[name] = {"sigma_fixed_point_ok": lyapunov_holds,
                     "A_lag_maxabs": a_norm,
                     "A_lag": A_lag.tolist()}
        ok_all = ok_all and lyapunov_holds
    ok = bool(ok_all
              and out["0.8I"]["A_lag_maxabs"] < 1e-12
              and out["diag(0.8,-0.8)"]["A_lag_maxabs"] < 1e-12
              and out["0.8Rot(0.35)"]["A_lag_maxabs"] > 1e-3)
    out["name"] = "B2_steady_state_vs_detailed_balance"
    out["scope_note"] = ("Gaussian Markov, single time-reversal rule for all "
                         "coordinates; does NOT extend to full AdamW entropy production")
    out["ok"] = ok
    return out


# ---------------------------------------------------------------------------
# B3 decision value and boundary saturation
# ---------------------------------------------------------------------------

def b3_decision_value() -> dict:
    delta = 0.05
    rng = np.random.default_rng(20260920)
    # all same sign -> V_sign = 0
    chi_same = np.array([-0.01, -0.02, -0.015, -0.008])
    v_same = delta * (np.mean(np.abs(chi_same)) - abs(np.mean(chi_same)))
    # mixed sign with same magnitudes -> V_sign > 0
    chi_mixed = np.array([-0.01, +0.02, -0.015, +0.008])
    v_mixed = delta * (np.mean(np.abs(chi_mixed)) - abs(np.mean(chi_mixed)))
    v_sign_ok = bool(abs(v_same) < 1e-15 and v_mixed > 0)

    # quadratic interior optimum, Q>0
    Q = 0.4
    chi1 = -0.01   # u* = 0.025 interior
    u1 = np.clip(-chi1 / Q, -delta, delta)
    chi2 = -0.05   # u* = 0.125 -> clip to boundary
    u2 = np.clip(-chi2 / Q, -delta, delta)
    chi3 = +0.001  # u* = -0.0025 interior (LR-down side)
    u3 = np.clip(-chi3 / Q, -delta, delta)
    interior_ok = bool(abs(u1 - 0.025) < 1e-12 and abs(u2 - delta) < 1e-12
                       and abs(u3 - (-0.0025)) < 1e-12)
    # loss check: quadratic achieves min at u*
    loss = lambda c, u: c * u + 0.5 * Q * u ** 2
    best_u = min((u for u in np.linspace(-delta, delta, 4001)),
                 key=lambda u: loss(chi1, u))
    interior_is_min = bool(abs(best_u - u1) < 5e-4)
    # same-sign chi does NOT exclude amplitude adaptivity:
    chi_a, chi_b = -0.002, -0.02   # both negative, different |u*|
    ua = float(np.clip(-chi_a / Q, -delta, delta))
    ub = float(np.clip(-chi_b / Q, -delta, delta))
    same_sign_diff_amp = bool(abs(ua - ub) > 1e-3)
    # Q<=0: no interior min; boundary vs zero comparison
    Qneg = -0.3
    u_star_formula_invalid = True  # must not apply clip(-chi/Q): -chi/Q is a MAXIMUM
    loss_neg = lambda c, u: c * u + 0.5 * Qneg * u ** 2
    for c in (-0.02, 0.02):
        cand = {u: loss_neg(c, u) for u in (-delta, 0.0, delta)}
        best = min(cand, key=cand.get)
        if abs(best - np.clip(-c / Qneg, -delta, delta)) < 1e-12:
            u_star_formula_invalid = False
    ok = bool(v_sign_ok and interior_ok and interior_is_min
              and same_sign_diff_amp and u_star_formula_invalid)
    return {"name": "B3_decision_value_boundary", "ok": ok,
            "V_sign_same_sign": float(v_same), "V_sign_mixed_sign": float(v_mixed),
            "u_star_interior": float(u1), "u_star_boundary": float(u2),
            "u_star_negative_chi": float(u3),
            "same_sign_different_amplitude": same_sign_diff_amp,
            "Qneg_formula_not_applied": u_star_formula_invalid,
            "rng_seed": 20260920}


# ---------------------------------------------------------------------------
# B4 finite-ensemble decomposition identity
# ---------------------------------------------------------------------------

def b4_decomposition_identity() -> dict:
    rng = np.random.default_rng(20260921)
    S = 4
    dim = 6
    A = rng.normal(size=(dim, dim))
    A = 0.5 * (A + A.T)          # symmetric, indefinite on purpose
    b = rng.normal(size=dim)

    def L(th):
        th = np.asarray(th)
        return float(0.5 * th @ A @ th + b @ th)

    base = rng.normal(size=(S, dim))
    dirs = rng.normal(size=(S, dim))
    u_probe = 0.05

    def curves(u):
        return np.array([L(base[s] + u * dirs[s]) for s in range(S)])

    def block(u):
        ths = np.stack([base[s] + u * dirs[s] for s in range(S)])
        th_bar = ths.mean(axis=0)
        L_ens = float(np.mean([L(ths[s]) for s in range(S)]))
        L_center = L(th_bar)
        J_disp = L_ens - L_center
        return L_ens, L_center, J_disp

    L0_ens, L0_center, J0 = block(0.0)
    ident_ok = bool(abs(L0_ens - (L0_center + J0)) < 1e-12)
    # central FD of each quantity
    def fd_of(fn):
        return (fn(+u_probe) - fn(-u_probe)) / (2 * u_probe)

    chi_ens = fd_of(lambda u: block(u)[0])
    chi_center = fd_of(lambda u: block(u)[1])
    chi_disp = fd_of(lambda u: block(u)[2])
    fd_ident_ok = bool(abs(chi_ens - (chi_center + chi_disp)) < 1e-10)
    # J_disp not forced non-negative: count sign over random blocks
    signs = []
    for _ in range(200):
        bb = rng.normal(size=(S, dim))
        th_bar = bb.mean(axis=0)
        e = float(np.mean([L(x) for x in bb]))
        signs.append(np.sign(e - L(th_bar)))
    frac_nonneg = float(np.mean([s >= 0 for s in signs]))
    ok = bool(ident_ok and fd_ident_ok and 0.0 < frac_nonneg < 1.0)
    return {"name": "B4_decomposition_identity", "ok": ok,
            "S": S, "dim": dim,
            "L_ens_eq_L_center_plus_Jdisp": ident_ok,
            "chi_ens_eq_chi_center_plus_chi_disp": fd_ident_ok,
            "chi_ens": float(chi_ens), "chi_center": float(chi_center),
            "chi_disp": float(chi_disp),
            "Jdisp_nonneg_fraction_random": frac_nonneg,
            "note": "J_disp sign-free (quadratic, indefinite A)"}


def run_all_theory_tests() -> dict:
    b1 = b1_toy_battery()
    b2 = b2_lag_antisymmetry()
    b3 = b3_decision_value()
    b4 = b4_decomposition_identity()
    return {"B1": b1, "B2": b2, "B3": b3, "B4": b4,
            "ALL_OK": bool(b1["ok"] and b2["ok"] and b3["ok"] and b4["ok"])}


# ---------------------------------------------------------------------------
# Analysis helpers (Stage E/F)
# ---------------------------------------------------------------------------

def class_mean_losses(records, key, stream_class):
    """records: iterable of per-stream loss dicts for one (cell, protocol, H).
    key in {"v_eval","v_score"}. Returns {u: mean over streams in class}."""
    sel = [r for r in records if r["stream_class"] == stream_class]
    out = {}
    for u in U_GRID:
        vals = [r["losses"][u][key] for r in sel]
        out[u] = float(np.mean(vals)) if vals else None
    return out


def chi_q_from_curve(Lmap):
    """Lmap: {u: L}. Central FD at a=0.05 and 0.025; Q via second difference.
    Returns dict with chi/Q at both a and consistency flags."""
    def get(u):
        v = Lmap[u]
        if v is None:
            raise ValueError("missing u")
        return float(v)

    out = {}
    for a in (0.05, 0.025):
        chi = (get(+a) - get(-a)) / (2 * a)
        Q = (get(+a) + get(-a) - 2 * get(0.0)) / a ** 2
        out[f"chi_fd_a{a}"] = float(chi)
        out[f"Q_fd_a{a}"] = float(Q)
    out["Q_sign_agree"] = bool(np.sign(out["Q_fd_a0.05"]) == np.sign(out["Q_fd_a0.025"]))
    return out


def quadratic_fit_dev(Ldev):
    """Least-squares quadratic on {-0.025, 0, +0.025} of dev class means.
    Returns (chi, Q) with L(u) ~ L0 + chi*u + 0.5*Q*u^2."""
    us = np.array([-0.025, 0.0, 0.025])
    ls = np.array([Ldev[u] for u in us])
    # exact fit of 3 points to a quadratic
    A = np.stack([np.ones_like(us), us, 0.5 * us ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, ls, rcond=None)
    return float(coef[1]), float(coef[2])


def q_identifiability(Q_dev, Q_audit, Q_a05, Q_a025):
    """Frozen rule from config: same sign AND |Q| > disagreement scale."""
    sign_agree = bool(np.sign(Q_a05) == np.sign(Q_a025))
    disagree = max(abs(Q_dev - Q_audit), abs(Q_a05 - Q_a025))
    Q = float(np.median([Q_dev, Q_audit, Q_a05, Q_a025]))
    identifiable = bool(sign_agree and abs(Q) > disagree)
    return {"Q_median": Q, "sign_agree": sign_agree,
            "disagreement_scale": float(disagree),
            "identifiable": identifiable}


def v_sign(chi_per_stream, delta=0.05):
    c = np.asarray(chi_per_stream, dtype=np.float64)
    return float(delta * (np.mean(np.abs(c)) - abs(np.mean(c))))


def v_grid(losses_per_stream):
    """losses_per_stream: list of {u: L}. Hindsight upper bound:
    V_grid = min_j mean_i L_i(u_j) - mean_i min_j L_i(u_j) >= 0."""
    us = U_GRID
    mat = np.array([[l[u] for u in us] for l in losses_per_stream])
    return float(np.min(mat.mean(axis=0)) - np.mean(np.min(mat, axis=1)))
