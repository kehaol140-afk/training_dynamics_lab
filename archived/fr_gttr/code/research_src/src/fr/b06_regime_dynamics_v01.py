"""FR B0.6 metastable regime dynamics (pure module: features, GMM/HMM, statistics).

No GPU code here. Everything the drivers need that is pure math lives here.

PRIMARY QUESTION (Q3): switching linear response model vs a single global map.
  Local model  (R = C Phi_i B): psi_{t+1} = A_{z_t} psi_t + b_{z_t} + eps with
    per-regime linear maps (A_i, b_i, residual covariance); z_t = metastable
    regime index from a Gaussian HMM over psi_t.
  Global model (single Phi):   psi_{t+1} = A psi_t + b + eps, one map.
  Measured: one-step-ahead held-out prediction error, split by
    within-cluster steps (z_t == z_{t+1}) vs across-transition steps
    (z_t != z_{t+1}); transition points; cluster persistence time; plus the
    cross-sectional analog (occupancy-mixture vs constant prediction of the
    frozen response chi_H).

SUPPORTING QUESTIONS:
  Q1  Do psi_t trajectories form metastable regimes at all? GMM grid
      (held-out LL) + numpy full-covariance Gaussian HMM (EM + Viterbi)
      against TWO nulls: (a) multivariate phase-randomized surrogates
      (stationary linear process, same second-order structure); (b)
      continuous-drift null (smooth per-trajectory drift transplanted across
      trajectories + stationary residuals) — regimes must beat BOTH smooth
      drift and stationary linearity to be called metastable switching.
  Q2  Does the response chi_H differ between regimes? (permutation test on
      seed-cluster means).

State construction (psi_t), per optimizer step per native trajectory:
  log gradient norm, log update norm, log optimizer-memory norms (m, v),
  cos(g_t, g_{t-1}), cos of successive updates, log parameter displacement
  from trajectory start, train-loss slope, log pre-clip norm, clip-activity
  flag. Clustering uses a frozen channel subset (CLUSTER_CHANNELS);
  lr_native and train_loss level are recorded but NOT clustered.

Terminology discipline (config.json): controlled non-equilibrium dynamical
system; metastable regime / switching dynamics / non-stationary reduced
model. Banned in any output string produced here: temperature, free energy,
entropy production, equilibrium, detailed balance, field theory, phase
transition (the ban list itself may be referenced).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_triangular

# ---------------------------------------------------------------------------
# Constants (frozen in config.json; mirrored here for the module's tests)
# ---------------------------------------------------------------------------

CLUSTER_CHANNELS = (
    "log_g", "log_upd", "log_m", "log_v",
    "g_cos", "upd_cos", "log_disp", "loss_slope", "log_preclip",
)
RECORDED_CHANNELS = CLUSTER_CHANNELS + ("clip_flag", "lr_native", "train_loss", "h")

HMM_K_GRID = (1, 2, 3, 4)
GMM_K_GRID = (1, 2, 3, 4)
DEFAULT_SEED = 20260921
DEFAULT_N_SURROGATES = 39
DEFAULT_TRAIN_FRAC = 0.75
DEFAULT_N_PERM = 20000
PURITY_THRESHOLD = 0.8
DRIFT_WINDOW = 9  # odd, for the continuous-drift null's smooth component


# ---------------------------------------------------------------------------
# psi_t feature construction
# ---------------------------------------------------------------------------

def psi_channels_from_rows(rows):
    """rows: per-step dicts (recorder output), ordered by h, one trajectory.

    Required keys: grad_norm, update_norm, train_loss, preclip_norm,
    lr_native, clip_active, h. Optional: m_norm, v_norm, g_cos, upd_cos,
    disp (exact displacement from trajectory start). Missing optional fields
    are filled with NaN (offline B0-diag rows lack them).

    Returns dict channel -> np.ndarray (len T) with the channels in
    RECORDED_CHANNELS.
    """
    T = len(rows)
    g = np.array([float(r["grad_norm"]) for r in rows], dtype=np.float64)
    upd = np.array([float(r["update_norm"]) for r in rows], dtype=np.float64)
    tl = np.array([float(r["train_loss"]) for r in rows], dtype=np.float64)
    pc = np.array([float(r["preclip_norm"]) for r in rows], dtype=np.float64)

    def opt(name):
        if rows[0].get(name) is None:
            return np.full(T, np.nan)
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    m = opt("m_norm")
    v = opt("v_norm")
    gcos = opt("g_cos")
    ucos = opt("upd_cos")
    disp = opt("disp")

    slope = np.zeros(T)
    slope[1:] = tl[1:] - tl[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "log_g": np.log(g),
            "log_upd": np.log(upd),
            "log_m": np.log(m),
            "log_v": np.log(v),
            "g_cos": gcos,
            "upd_cos": ucos,
            "log_disp": np.log(disp),
            "loss_slope": slope,
            "log_preclip": np.log(pc),
            "clip_flag": np.array([bool(r["clip_active"]) for r in rows],
                                  dtype=np.float64),
            "lr_native": np.array([float(r["lr_native"]) for r in rows]),
            "train_loss": tl,
            "h": np.array([int(r["h"]) for r in rows], dtype=np.float64),
        }
    return out


def channels_to_matrix(channels, channel_names=CLUSTER_CHANNELS):
    """Stack selected channels into a (T, D) matrix."""
    return np.stack([np.asarray(channels[c], dtype=np.float64)
                     for c in channel_names], axis=1)


# ---------------------------------------------------------------------------
# Standardization (train-window fit; no test leakage)
# ---------------------------------------------------------------------------

class Standardizer:
    """Pooled z-scoring: mean/std fit on train windows only."""

    def __init__(self, channel_names=CLUSTER_CHANNELS):
        self.channel_names = tuple(channel_names)
        self.mean_ = None
        self.std_ = None

    def fit(self, mats):
        pooled = np.concatenate(mats, axis=0)
        self.mean_ = pooled.mean(axis=0)
        self.std_ = pooled.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, mat):
        return (mat - self.mean_) / self.std_


def split_train_test(mats, train_frac=DEFAULT_TRAIN_FRAC):
    """Split each (T, D) trajectory matrix into train/test windows."""
    train, test = [], []
    for X in mats:
        n_tr = max(2, int(round(len(X) * train_frac)))
        n_tr = min(n_tr, len(X) - 1)
        train.append(X[:n_tr])
        test.append(X[n_tr:])
    return train, test


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

def multivariate_phase_surrogate(X, rng):
    """Phase-randomized surrogate preserving per-channel spectra AND
    cross-channel linear correlations (one shared random phase per
    frequency), destroying regime structure. Stationary linear Gaussian null.
    """
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    F = np.fft.rfft(X, axis=0)
    n_freq = F.shape[0]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    rot = np.exp(1j * phases)[:, None]
    Y = np.fft.irfft(F * rot, n=T, axis=0)
    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0)
    sd_x[sd_x < 1e-12] = 1.0
    mu_y = Y.mean(axis=0)
    sd_y = Y.std(axis=0)
    sd_y[sd_y < 1e-12] = 1.0
    return (Y - mu_y) / sd_y * sd_x + mu_x


def smooth_drift(X, w=DRIFT_WINDOW):
    """Centered moving-average smooth component of a (T, D) series."""
    T, D = X.shape
    assert w % 2 == 1 and w >= 3
    half = w // 2
    out = np.zeros_like(X)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out[t] = X[lo:hi].mean(axis=0)
    return out


def drift_null_series(mats, rng):
    """Continuous-drift null series.

    Each trajectory is decomposed into a smooth drift + residual. The null
    recombines a DONOR drift (another trajectory, possibly time-reversed)
    with a phase-randomized version of the residual. The result has smooth
    drift and stationary fluctuations but NO discrete regime switching:
    apparent 'regimes' under this null would be an artifact of continuous
    drift plus local fluctuations.
    """
    n = len(mats)
    out = []
    for c, X in enumerate(mats):
        drift = smooth_drift(X)
        resid = X - drift
        donor = int(rng.integers(n))
        rev = bool(rng.integers(2))
        d = mats[donor]
        d_drift = smooth_drift(d)[::-1] if rev else smooth_drift(d)
        if len(d_drift) != len(drift):
            d_drift = drift  # length mismatch guard (should not happen)
        null = d_drift + multivariate_phase_surrogate(resid, rng)
        out.append(null)
    return out


# ---------------------------------------------------------------------------
# GMM (sklearn wrapper, seeded)
# ---------------------------------------------------------------------------

def fit_gmm_grid(mats_train, mats_test, ks=GMM_K_GRID, seed=DEFAULT_SEED):
    """Fit GaussianMixture per k on pooled train; score train and test."""
    from sklearn.mixture import GaussianMixture

    X_tr = np.concatenate(mats_train, axis=0)
    X_te = np.concatenate(mats_test, axis=0)
    out = []
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, max_iter=200, reg_covar=1e-6,
                             random_state=int(seed))
        gm.fit(X_tr)
        out.append({
            "k": int(k),
            "train_ll": float(gm.score(X_tr)),
            "test_ll": float(gm.score(X_te)),
            "aic": float(gm.aic(X_tr)),
            "bic": float(gm.bic(X_tr)),
            "converged": bool(gm.converged_),
        })
    return out


# ---------------------------------------------------------------------------
# Gaussian HMM (numpy EM + Viterbi), full covariance
# ---------------------------------------------------------------------------

def _log_gauss(X, mean, cov):
    """log N(X; mean, cov) for X (T, D). Cholesky-based."""
    D = X.shape[1]
    cov = cov + 1e-4 * np.eye(D)
    L = np.linalg.cholesky(cov)
    diff = (X - mean).T
    sol = solve_triangular(L, diff, lower=True)
    maha = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet + maha)


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        m = float(np.max(a))
        return m + math.log(float(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class GaussianHMM:
    """Full-covariance Gaussian HMM fit by EM over a list of sequences.

    Deterministic given (K, seed): means initialized at K distinct pooled
    train rows chosen by a seeded generator; pi uniform; A sticky
    (0.9 diagonal). Convergence: relative LL improvement < tol.
    """

    def __init__(self, K, seed=DEFAULT_SEED, tol=1e-6, max_iter=200):
        self.K = int(K)
        self.seed = int(seed)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.logpi = None
        self.logA = None
        self.means = None
        self.covs = None
        self.n_iter_ = 0
        self.loglik_history_ = []

    def _emit_logb(self, X):
        return np.stack([_log_gauss(X, self.means[i], self.covs[i])
                         for i in range(self.K)], axis=1)

    def _forward_scaled(self, logB):
        """Scaled forward pass: normalized alphas, total loglik, and the
        per-step normalization constants needed by the scaled backward pass."""
        T, K = logB.shape
        alpha = np.zeros((T, K))
        cs = np.zeros(T)
        a = self.logpi + logB[0]
        cs[0] = _logsumexp(a)
        alpha[0] = a - cs[0]
        for t in range(1, T):
            a = _logsumexp(alpha[t - 1][:, None] + self.logA, axis=0) + logB[t]
            cs[t] = _logsumexp(a)
            alpha[t] = a - cs[t]
        return alpha, float(np.sum(cs)), cs

    def _backward_scaled(self, logB, cs):
        T, K = logB.shape
        beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = _logsumexp(self.logA + logB[t + 1][None, :]
                                 + beta[t + 1][None, :], axis=1) - cs[t + 1]
        return beta

    def fit(self, seqs):
        Xp = np.concatenate(seqs, axis=0)
        Ttot, D = Xp.shape
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(Ttot, size=self.K, replace=False)
        self.means = Xp[idx].copy()
        gcov = np.cov(Xp.T) + 1e-4 * np.eye(D)
        self.covs = np.stack([gcov.copy() for _ in range(self.K)])
        self.logpi = np.full(self.K, -math.log(self.K))
        off = 0.1 / max(self.K - 1, 1)
        A = np.full((self.K, self.K), off)
        np.fill_diagonal(A, 0.9)
        if self.K == 1:
            A = np.ones((1, 1))
        self.logA = np.log(A)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            gamma_sum = np.zeros(self.K)
            xx_sum = np.zeros((self.K, D))
            xxT_sum = np.zeros((self.K, D, D))
            xi_sum = np.zeros((self.K, self.K))
            pi_sum = np.zeros(self.K)
            ll_tot = 0.0
            for X in seqs:
                logB = self._emit_logb(X)
                alpha, ll, cs = self._forward_scaled(logB)
                beta = self._backward_scaled(logB, cs)
                gamma = alpha + beta
                gamma -= _logsumexp(gamma, axis=1)[:, None]
                g = np.exp(gamma)
                pi_sum += g[0]
                gamma_sum += g.sum(axis=0)
                xx_sum += g.T @ X
                # weighted per-state covariance, vectorized over states:
                # Dd[t, i, :] = X[t] - mean_i ; xxT_i = sum_t g[t,i] Dd Dd^T
                Dd = X[:, None, :] - self.means[None, :, :]
                xxT_sum += np.einsum("tk,tki,tkj->kij", g, Dd, Dd)
                # xi over all t at once: (T-1, K, K)
                xi = (alpha[:-1][:, :, None] + self.logA[None, :, :]
                      + logB[1:][:, None, :] + beta[1:][:, None, :])
                xi -= _logsumexp(xi.reshape(len(xi), -1), axis=1)[:, None, None]
                xi_sum += np.exp(xi).sum(axis=0)
                ll_tot += ll
            self.loglik_history_.append(float(ll_tot))
            eps = 1e-9
            self.logpi = np.log(np.maximum(pi_sum, eps)) - np.log(max(pi_sum.sum(), eps))
            A = np.maximum(xi_sum, eps)
            A = A / A.sum(axis=1, keepdims=True)
            if self.K == 1:
                A = np.ones((1, 1))
            self.logA = np.log(A)
            for i in range(self.K):
                cnt = max(gamma_sum[i], eps)
                self.means[i] = xx_sum[i] / cnt
                self.covs[i] = xxT_sum[i] / cnt + 1e-4 * np.eye(D)
            self.n_iter_ = it + 1
            if prev_ll > -np.inf and abs(ll_tot - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll_tot
        return self

    def loglik(self, seqs):
        ll = 0.0
        for X in seqs:
            _, l, _ = self._forward_scaled(self._emit_logb(X))
            ll += l
        return float(ll)

    def viterbi(self, X):
        T = len(X)
        logB = self._emit_logb(X)
        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = self.logpi + logB[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + self.logA
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(self.K)] + logB[t]
        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path, float(np.max(delta[T - 1]))


def fit_hmm_grid(mats_train, mats_test, ks=HMM_K_GRID, seed=DEFAULT_SEED):
    """Fit GaussianHMM per k on train windows; score train/test."""
    out = []
    models = {}
    for k in ks:
        hmm = GaussianHMM(k, seed=seed + 1000 * k).fit(mats_train)
        out.append({
            "k": int(k),
            "train_ll": hmm.loglik(mats_train),
            "test_ll": hmm.loglik(mats_test),
            "n_iter": hmm.n_iter_,
        })
        models[k] = hmm
    return out, models


# ---------------------------------------------------------------------------
# Regime summaries: run lengths, transitions, persistence
# ---------------------------------------------------------------------------

def run_lengths(states):
    """Run-length encoding of an integer state sequence."""
    runs = []
    for s in states:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([int(s), 1])
    return [(int(a), int(b)) for a, b in runs]


def transition_points(states):
    """1-based step indices at which a new Viterbi regime becomes active
    (a change between steps h and h+1 is reported as h+1)."""
    st = np.asarray(states)
    if len(st) < 2:
        return []
    return [int(t + 2) for t in np.where(st[1:] != st[:-1])[0]]


def persistence_stats(list_of_states):
    """Pooled over trajectories: mean/median run length, switch rate,
    occupancy per state."""
    total_runs = []
    n_switches = 0
    n_steps = 0
    occupancy = {}
    for states in list_of_states:
        rl = run_lengths(states)
        total_runs.extend([L for _, L in rl])
        n_switches += max(len(rl) - 1, 0)
        n_steps += max(len(states) - 1, 0)
        for s in states:
            occupancy[int(s)] = occupancy.get(int(s), 0) + 1
    tot = sum(occupancy.values())
    occupancy = {k: v / tot for k, v in sorted(occupancy.items())}
    return {
        "mean_run_length": float(np.mean(total_runs)) if total_runs else 0.0,
        "median_run_length": float(np.median(total_runs)) if total_runs else 0.0,
        "min_run_length": int(min(total_runs)) if total_runs else 0,
        "max_run_length": int(max(total_runs)) if total_runs else 0,
        "n_runs": int(len(total_runs)),
        "switch_rate": float(n_switches / n_steps) if n_steps else 0.0,
        "occupancy": occupancy,
    }


def transition_matrix(list_of_states, K):
    """Pooled empirical transition matrix (row-normalized counts)."""
    C = np.zeros((K, K))
    for states in list_of_states:
        st = np.asarray(states)
        for a, b in zip(st[:-1], st[1:]):
            C[int(a), int(b)] += 1.0
    rows = C.sum(axis=1, keepdims=True)
    rows[rows == 0.0] = 1.0
    return C / rows


# ---------------------------------------------------------------------------
# Q3a (PRIMARY): switching linear response model vs single global map
# ---------------------------------------------------------------------------

def fit_switching_linear_maps(mats, paths, K, train_frac=DEFAULT_TRAIN_FRAC):
    """Fit per-regime linear one-step maps  psi_{t+1} = A_i psi_t + b_i + eps
    (the local model R = C Phi_i B reduced to the observed state) and the
    single global map (A, b). Least squares on train windows only.

    Returns dict with per-regime parameters (A, b, residual covariance,
    spectral radius, train count), global parameters, and held-out one-step
    prediction errors split by within-cluster steps (z_t == z_{t+1}) vs
    across-transition steps (z_t != z_{t+1}) for both models.
    """
    D = mats[0].shape[1]
    tr_len = max(2, int(round(len(mats[0]) * train_frac)))
    tr_len = min(tr_len, len(mats[0]) - 1)

    def design(X, t0, t1):
        Xs, ys = [], []
        for X_ in mats:
            Xs.append(X_[t0:t1])
            ys.append(X_[t0 + 1:t1 + 1])
        return Xs, ys

    # collect train pairs per regime and globally
    feats = {i: [] for i in range(K)}
    feats["global"] = []
    targets = {i: [] for i in range(K)}
    targets["global"] = []
    for X, z in zip(mats, paths):
        for t in range(tr_len):
            x = np.concatenate([X[t], [1.0]])
            feats["global"].append(x)
            targets["global"].append(X[t + 1])
            feats[int(z[t])].append(x)
            targets[int(z[t])].append(X[t + 1])

    def lstsq(feat_list, tgt_list):
        if len(feat_list) < D + 2:
            return None
        Fm = np.stack(feat_list)
        Tm = np.stack(tgt_list)
        W, *_ = np.linalg.lstsq(Fm, Tm, rcond=None)   # (D+1, D)
        resid = Tm - Fm @ W
        cov = resid.T @ resid / max(len(feat_list) - (D + 1), 1)
        return W, cov

    reg = {}
    for key in list(feats.keys()):
        out = lstsq(feats[key], targets[key])
        if out is None:
            reg[key] = None
            continue
        W, cov = out
        A = W[:-1, :].T          # (D, D) mapping psi_t -> psi_{t+1}
        b = W[-1, :]
        ev = np.linalg.eigvals(A)
        reg[key] = {
            "W": W,
            "A": A,
            "b": b,
            "resid_cov": cov,
            "resid_var_trace": float(np.trace(cov)),
            "spectral_radius": float(np.max(np.abs(ev))),
            "n_train": int(len(feats[key])),
        }

    # held-out one-step prediction on test windows
    errs = {("local", "within"): [], ("local", "across"): [],
            ("global", "within"): [], ("global", "across"): []}
    for X, z in zip(mats, paths):
        for t in range(tr_len, len(X) - 1):
            x = np.concatenate([X[t], [1.0]])
            y = X[t + 1]
            transition = int(z[t]) != int(z[t + 1])
            slot = "across" if transition else "within"
            rl = reg.get(int(z[t]))
            if rl is not None:
                pred = x @ rl["W"]
                errs[("local", slot)].append(float(np.mean((pred - y) ** 2)))
            rg = reg.get("global")
            if rg is not None:
                pred = x @ rg["W"]
                errs[("global", slot)].append(float(np.mean((pred - y) ** 2)))

    def mse(v):
        return float(np.mean(v)) if v else None

    return {
        "regimes": reg,
        "test_mse": {
            "local_within": mse(errs[("local", "within")]),
            "local_across": mse(errs[("local", "across")]),
            "global_within": mse(errs[("global", "within")]),
            "global_across": mse(errs[("global", "across")]),
            "n_local_within": len(errs[("local", "within")]),
            "n_local_across": len(errs[("local", "across")]),
            "n_global_within": len(errs[("global", "within")]),
            "n_global_across": len(errs[("global", "across")]),
        },
        "train_len": int(tr_len),
        "D": int(D),
    }


def q3a_verdict(switch_res, rules):
    """Frozen decision rule for the PRIMARY question: the switching linear
    response model is supported if (i) the local map beats the global map on
    held-out one-step error AND (ii) its deficit concentrates at transitions
    (across-transition error > within-cluster error) — i.e. the single-map
    failure is a switching-across-regimes failure."""
    m = switch_res["test_mse"]
    if m["local_within"] is None or m["global_within"] is None:
        return {"verdict": "NOT_TESTED"}
    cond_local_better = (m["local_within"] + (m["local_across"] or 0.0)
                         * rules["across_weight"]
                         < m["global_within"] + (m["global_across"] or 0.0)
                         * rules["across_weight"])
    across_gap_local = ((m["local_across"] or 0.0) / max(m["local_within"], 1e-12))
    across_gap_global = ((m["global_across"] or 0.0) / max(m["global_within"], 1e-12))
    cond_transition_gap = (m["local_across"] is not None
                           and m["local_across"] > m["local_within"])
    cond_local_worse_across = across_gap_local > across_gap_global
    ok = bool(cond_local_better and cond_transition_gap)
    verdict = ("SUPPORTED" if ok else
               "NOT_SUPPORTED" if not cond_local_better else "UNRESOLVED")
    return {"verdict": verdict,
            "cond_local_better_overall": bool(cond_local_better),
            "cond_local_across_gt_within": bool(cond_transition_gap),
            "across_within_gap_local": float(across_gap_local),
            "across_within_gap_global": float(across_gap_global),
            "local_gap_exceeds_global_gap": bool(cond_local_worse_across)}


# ---------------------------------------------------------------------------
# Q1 pipeline (supporting: does regime structure exist at all?)
# ---------------------------------------------------------------------------

def _band(values):
    v = np.asarray(list(values), dtype=np.float64)
    return {"min": float(v.min()), "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def q1_regime_analysis(channels_by_traj, ks=HMM_K_GRID, seed=DEFAULT_SEED,
                       n_surrogates=DEFAULT_N_SURROGATES,
                       train_frac=DEFAULT_TRAIN_FRAC):
    """Regime-structure analysis with TWO nulls.

    Null (a): phase-randomized surrogates (stationary linear Gaussian).
    Null (b): continuous-drift surrogates (smooth transplanted drift +
    stationary residuals).
    Regime structure must beat both to be called metastable switching.
    """
    ids = sorted(channels_by_traj.keys())
    mats = [channels_to_matrix(channels_by_traj[i]) for i in ids]
    scaler = Standardizer().fit(mats)
    z_mats = [scaler.transform(m) for m in mats]
    z_train, z_test = split_train_test(z_mats, train_frac)

    gmm_table = fit_gmm_grid(z_train, z_test, ks=ks, seed=seed)
    hmm_table, _ = fit_hmm_grid(z_train, z_test, ks=ks, seed=seed)

    best_k = max(hmm_table, key=lambda r: r["test_ll"])["k"]

    full_hmm = GaussianHMM(best_k, seed=seed + 1000 * best_k).fit(z_mats)
    paths = {tid: full_hmm.viterbi(X)[0] for tid, X in zip(ids, z_mats)}
    pers = persistence_stats([paths[i] for i in ids])
    tmat = transition_matrix([paths[i] for i in ids], best_k)
    tpoints = {i: transition_points(paths[i]) for i in ids}

    rng = np.random.default_rng(seed + 7)

    def _synth_stats(synth_mats):
        s_train, s_test = split_train_test(synth_mats, train_frac)
        s_hmm = GaussianHMM(best_k, seed=seed + 1000 * best_k).fit(s_train)
        s_full = GaussianHMM(best_k, seed=seed + 1000 * best_k).fit(synth_mats)
        s_paths = [s_full.viterbi(X)[0] for X in synth_mats]
        s_pers = persistence_stats(s_paths)
        return {
            "test_ll": s_hmm.loglik(s_test),
            "mean_run_length": s_pers["mean_run_length"],
            "switch_rate": s_pers["switch_rate"],
        }

    phase_surr = [_synth_stats([multivariate_phase_surrogate(X, rng) for X in z_mats])
                  for _ in range(n_surrogates)]
    drift_surr = [_synth_stats(drift_null_series(z_mats, rng))
                  for _ in range(n_surrogates)]

    def _summ(ss):
        return {"test_ll": _band(s["test_ll"] for s in ss),
                "mean_run_length": _band(s["mean_run_length"] for s in ss),
                "switch_rate": _band(s["switch_rate"] for s in ss),
                "n": int(len(ss))}

    return {
        "traj_ids": ids,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.std_.tolist(),
        "gmm_table": gmm_table,
        "hmm_table": hmm_table,
        "best_k": int(best_k),
        "paths": {i: [int(s) for s in paths[i]] for i in ids},
        "persistence": pers,
        "transition_matrix": tmat.tolist(),
        "transition_points": tpoints,
        "phase_null": _summ(phase_surr),
        "drift_null": _summ(drift_surr),
    }


def q1_verdict(res, rules):
    """Frozen decision rule for Q1 (supporting)."""
    if res["best_k"] < 2:
        return {"verdict": "NOT_SUPPORTED",
                "reason": "best_k<2: no multimodal structure in held-out likelihood"}
    hmm_by_k = {r["k"]: r for r in res["hmm_table"]}
    cond_multimodal = hmm_by_k[2]["test_ll"] > hmm_by_k[1]["test_ll"]
    cond_ll = max(r["test_ll"] for r in res["hmm_table"]) > res["phase_null"]["test_ll"]["p95"]
    cond_persist = (res["persistence"]["mean_run_length"] >= rules["min_mean_run_length"]
                    and res["persistence"]["switch_rate"] <= res["phase_null"]["switch_rate"]["p95"])
    cond_drift = (res["persistence"]["mean_run_length"]
                  > res["drift_null"]["mean_run_length"]["p95"])
    ok = bool(cond_multimodal and cond_ll and cond_persist and cond_drift)
    not_ok = bool(not cond_multimodal or not cond_ll)
    verdict = "SUPPORTED" if ok else ("NOT_SUPPORTED" if not_ok else "UNRESOLVED")
    return {"verdict": verdict,
            "cond_multimodal": bool(cond_multimodal),
            "cond_hmm_beats_phase_null_ll": bool(cond_ll),
            "cond_persistence": bool(cond_persist),
            "cond_beats_drift_null": bool(cond_drift),
            "min_mean_run_length": rules["min_mean_run_length"]}


# ---------------------------------------------------------------------------
# Q2 (supporting): response chi by regime
# ---------------------------------------------------------------------------

def occupancy_from_paths(paths, K):
    """Fraction of time in each state over the given paths."""
    counts = np.zeros(K)
    total = 0
    for p in paths:
        counts += np.bincount(np.asarray(p, dtype=int), minlength=K)
        total += len(p)
    occ = counts / max(total, 1)
    return {int(i): float(occ[i]) for i in range(K)}


def q2_chi_by_regime(chi_rows, occupancy_by_cell, rules):
    """chi_rows: (seed, step, protocol, H, chi_dev, chi_audit).
    occupancy_by_cell: {(seed, step): {state: frac}}.

    Per (protocol, H): group cells by dominant regime; statistic =
    |mean(group A) - mean(group B)|; seeded permutation p; seed-cluster
    sign consistency.
    """
    cells = sorted(occupancy_by_cell.keys())
    dom = {c: max(occupancy_by_cell[c], key=occupancy_by_cell[c].get) for c in cells}
    states = sorted({dom[c] for c in cells})
    top2 = states[:2] if len(states) >= 2 else states + [None]
    rng = np.random.default_rng(rules.get("seed", DEFAULT_SEED))

    results = []
    protos = sorted({r["protocol"] for r in chi_rows})
    Hs = sorted({int(r["H"]) for r in chi_rows})
    for proto in protos:
        for H in Hs:
            sel = [r for r in chi_rows
                   if r["protocol"] == proto and int(r["H"]) == H]
            if len(sel) < 4:
                continue
            vals, labs, meta = [], [], []
            for r in sel:
                key = (int(r["seed"]), int(r["step"]))
                if key not in dom or dom[key] not in top2 or top2[1] is None:
                    continue
                for cls in ("chi_dev", "chi_audit"):
                    vals.append(float(r[cls]))
                    labs.append(dom[key])
                    meta.append(key)
            if len(set(labs)) < 2 or len(vals) < 6:
                results.append({"protocol": proto, "H": int(H), "n": len(vals),
                                "verdict_detail": "insufficient group diversity"})
                continue
            vals = np.asarray(vals)
            labs = np.asarray(labs)
            obs = abs(vals[labs == top2[0]].mean() - vals[labs == top2[1]].mean())
            n_perm = int(rules.get("n_perm", DEFAULT_N_PERM))
            cnt = 0
            for _ in range(n_perm):
                pl = rng.permutation(labs)
                stat = abs(vals[pl == top2[0]].mean() - vals[pl == top2[1]].mean())
                if stat >= obs:
                    cnt += 1
            p = (cnt + 1) / (n_perm + 1)
            diff = float(vals[labs == top2[0]].mean() - vals[labs == top2[1]].mean())
            seed_signs = {}
            for s in sorted({m[0] for m in meta}):
                idx = [i for i, m in enumerate(meta) if m[0] == s]
                if len(set(labs[idx])) < 2:
                    seed_signs[s] = None
                    continue
                d = float(vals[[i for i in idx if labs[i] == top2[0]]].mean()
                          - vals[[i for i in idx if labs[i] == top2[1]]].mean())
                seed_signs[s] = int(np.sign(d)) if d != 0.0 else 0
            consistent = sum(1 for v in seed_signs.values() if v == int(np.sign(diff)))
            results.append({
                "protocol": proto, "H": int(H), "n": int(len(vals)),
                "group_A": int(top2[0]), "group_B": int(top2[1]),
                "mean_A": float(vals[labs == top2[0]].mean()),
                "mean_B": float(vals[labs == top2[1]].mean()),
                "diff_A_minus_B": diff,
                "perm_p": float(p),
                "seed_signs": {str(k): v for k, v in seed_signs.items()},
                "n_seed_consistent": int(consistent),
                "n_seed_total": int(len(seed_signs)),
            })
    tested = [r for r in results if "perm_p" in r]
    if not tested:
        verdict = "NOT_TESTED"
    else:
        pos = [r for r in tested if r["perm_p"] < rules["alpha"]
               and r["n_seed_consistent"] >= rules["min_seed_consistent"]]
        verdict = "SUPPORTED" if len(pos) >= math.ceil(0.5 * len(tested)) else \
                  ("NOT_SUPPORTED" if not pos else "UNRESOLVED")
    return {"rows": results, "verdict": verdict, "alpha": rules["alpha"],
            "min_seed_consistent": rules["min_seed_consistent"],
            "n_tested": len(tested)}


# ---------------------------------------------------------------------------
# Q3b: cross-sectional local (occupancy mixture) vs global chi prediction
# ---------------------------------------------------------------------------

def q3b_local_vs_global_chi(chi_rows, occupancy_by_cell, seed_of_cell, rules):
    """Occupancy-mixture (local) vs pooled-constant (global) prediction of
    chi under leave-one-cell-out CV; errors via the frozen a0c _metric."""
    from fr.a0c_adamw_response_propagation_v01 import _metric

    cells = sorted(occupancy_by_cell.keys())
    K = max(max(occupancy_by_cell[c]) for c in cells) + 1

    protos = sorted({r["protocol"] for r in chi_rows})
    Hs = sorted({int(r["H"]) for r in chi_rows})

    fold_rows = []
    for proto in protos:
        for H in Hs:
            sel = [r for r in chi_rows
                   if r["protocol"] == proto and int(r["H"]) == H]
            sub = {}
            for r in sel:
                key = (int(r["seed"]), int(r["step"]))
                if key in occupancy_by_cell:
                    sub.setdefault(key, {})["dev"] = float(r["chi_dev"])
                    sub.setdefault(key, {})["audit"] = float(r["chi_audit"])
            keys = sorted(sub.keys())
            if len(keys) < 4:
                continue
            occ = np.array([[occupancy_by_cell[c].get(i, 0.0) for i in range(K)]
                            for c in keys])
            for j, c in enumerate(keys):
                tr = [i for i in range(len(keys)) if i != j]
                tr_occ = occ[tr]
                tr_dev = np.array([sub[keys[i]]["dev"] for i in tr])
                tr_aud = np.array([sub[keys[i]]["audit"] for i in tr])
                loc_dev = np.array([np.average(tr_dev, weights=np.maximum(tr_occ[:, i], 1e-9))
                                    for i in range(K)])
                loc_aud = np.array([np.average(tr_aud, weights=np.maximum(tr_occ[:, i], 1e-9))
                                    for i in range(K)])
                glob_dev = float(tr_dev.mean())
                glob_aud = float(tr_aud.mean())
                for cls, loc_pred, glob_pred in (("dev", loc_dev, glob_dev),
                                                 ("audit", loc_aud, glob_aud)):
                    obs = sub[c][cls]
                    fold_rows.append({
                        "protocol": proto, "H": int(H), "cell": c,
                        "seed": int(seed_of_cell(c)), "cls": cls,
                        "observed": obs, "pred_local": float(occ[j] @ loc_pred),
                        "pred_global": float(glob_pred),
                        "abs_err_local": abs(float(occ[j] @ loc_pred) - obs),
                        "abs_err_global": abs(float(glob_pred) - obs),
                        "purity": float(occ[j].max()),
                        "split": ("within_cluster" if occ[j].max() >= PURITY_THRESHOLD
                                  else "across_transition"),
                    })
    if not fold_rows:
        return {"fold_rows": [], "verdict": "NOT_TESTED"}
    metric_rows = {}
    for model in ("local", "global"):
        metric_rows[model] = _metric([r["observed"] for r in fold_rows],
                                     [r[f"pred_{model}"] for r in fold_rows])

    seeds = sorted({r["seed"] for r in fold_rows})
    # ties at numerical resolution count as ties, not wins
    resolution = float(rules.get("resolution", 1e-12))
    seed_cmp = []
    for s in seeds:
        rows = [r for r in fold_rows if r["seed"] == s]
        d = [r["abs_err_local"] - r["abs_err_global"] for r in rows]
        seed_cmp.append({"seed": int(s), "n": len(d),
                         "median_diff": float(np.median(d)),
                         "local_wins": int(sum(1 for x in d if x < -resolution)),
                         "local_losses": int(sum(1 for x in d if x > resolution)),
                         "ties": int(sum(1 for x in d if abs(x) <= resolution))})
    n_favor = sum(1 for c in seed_cmp if c["median_diff"] < -resolution)

    def _med(split, model):
        errs = [r[f"abs_err_{model}"] for r in fold_rows if r["split"] == split]
        return float(np.median(errs)) if errs else None

    verdict = ("SUPPORTED" if n_favor >= rules["min_seed_folds"]
               else "NOT_SUPPORTED" if n_favor == 0 else "UNRESOLVED")
    return {
        "fold_rows": fold_rows,
        "metric": metric_rows,
        "seed_cmp": seed_cmp,
        "n_seed_favor_local": int(n_favor),
        "n_seed_total": int(len(seed_cmp)),
        "within_across_median_abs_err": {
            "within_cluster": {"local": _med("within_cluster", "local"),
                               "global": _med("within_cluster", "global")},
            "across_transition": {"local": _med("across_transition", "local"),
                                  "global": _med("across_transition", "global")},
        },
        "verdict": verdict,
        "min_seed_folds": rules["min_seed_folds"],
    }


# ---------------------------------------------------------------------------
# Descriptive companion: pulse4 retained-gain decay vs transition locations
# ---------------------------------------------------------------------------

def retained_gain_decay(pulse_rows, native_rows, tpoints_by_traj):
    """pulse/native rows: per-step train_loss dicts. g(t) = loss_pulse(t) -
    loss_native(t), t>=5. Decay half-life = first t with |g(t)| <= 0.5|g(5)|
    (censored at T). Spearman(half_life, first transition point) —
    hindsight/descriptive only."""
    from scipy.stats import spearmanr

    def curve(rows):
        return {int(r["h"]): float(r["train_loss"]) for r in rows}

    gains = {}
    for key, prows in pulse_rows.items():
        if key not in native_rows:
            continue
        pc, nc = curve(prows), curve(native_rows[key])
        g = {h: pc[h] - nc[h] for h in sorted(pc) if h in nc and h >= 5}
        if len(g) < 4 or g[5] == 0.0:
            continue
        gains[key] = g

    rows = []
    for key, g in gains.items():
        thr = 0.5 * abs(g[5])
        half = next((h for h in sorted(g) if abs(g[h]) <= thr), None)
        cell_key = key[:2]
        stream = key[2]
        tps = tpoints_by_traj.get((int(cell_key[0]), int(cell_key[1]), int(stream)), [])
        rows.append({
            "seed": int(cell_key[0]), "step": int(cell_key[1]),
            "stream_seed": int(stream), "u": float(key[3]),
            "g5": float(g[5]), "g16": float(g.get(16, float("nan"))),
            "g64": float(g.get(64, float("nan"))),
            "half_life": int(half if half is not None else 64),
            "censored": bool(half is None),
            "first_transition": int(tps[0] if tps else 65),
            "n_transitions": len(tps),
        })
    if len(rows) >= 8:
        rho, p = spearmanr([r["half_life"] for r in rows],
                           [r["first_transition"] for r in rows])
        corr = {"spearman_rho": float(rho), "spearman_p": float(p),
                "n": int(len(rows))}
    else:
        corr = {"spearman_rho": None, "spearman_p": None, "n": int(len(rows))}
    return {"rows": rows, "corr": corr}


# ---------------------------------------------------------------------------
# Theory tests T1-T6 (CPU, deterministic)
# ---------------------------------------------------------------------------

def _sample_slds(rng, n_seq, T, D, K=2):
    """Switching linear dynamical system with sticky transitions and well
    separated regime means; returns sequences, states, true A."""
    means = np.stack([(-1.0) ** k * (0.8 + 0.1 * k) * np.ones(D)
                      for k in range(K)])
    A_true = np.full((K, K), 0.04)
    np.fill_diagonal(A_true, 0.96)
    seqs, states = [], []
    for _ in range(n_seq):
        z = np.zeros(T, dtype=int)
        z[0] = rng.integers(K)
        for t in range(1, T):
            z[t] = rng.choice(K, p=A_true[z[t - 1]])
        X = np.zeros((T, D))
        for t in range(T):
            X[t] = means[z[t]] + 0.35 * rng.standard_normal(D)
        seqs.append(X)
        states.append(z)
    return seqs, states, A_true


def t1_slds_recovery(seed=DEFAULT_SEED):
    """HMM recovers planted regimes in a synthetic switching system."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    seqs, states, _ = _sample_slds(rng, n_seq=24, T=64, D=9, K=2)
    tr, te = split_train_test(seqs, DEFAULT_TRAIN_FRAC)
    table, _ = fit_hmm_grid(tr, te, ks=(1, 2), seed=seed)
    hmm2 = GaussianHMM(2, seed=seed + 2000).fit(seqs)
    aris = [adjusted_rand_score(z, hmm2.viterbi(X)[0])
            for X, z in zip(seqs, states)]
    pers = persistence_stats([hmm2.viterbi(X)[0] for X in seqs])
    true_pers = persistence_stats(states)
    ratio = pers["mean_run_length"] / max(true_pers["mean_run_length"], 1e-9)
    ok = bool(table[1]["test_ll"] > table[0]["test_ll"]
              and float(np.mean(aris)) >= 0.85
              and 0.5 <= ratio <= 2.0)
    return {"name": "T1_slds_recovery", "ok": bool(ok),
            "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
            "hmm2_beats_hmm1_ll": bool(table[1]["test_ll"] > table[0]["test_ll"]),
            "recovered_mean_run": pers["mean_run_length"],
            "true_mean_run": true_pers["mean_run_length"]}


def _stationary_ar1(rng, n_seq, T, D):
    M = np.eye(D) * 0.85 + rng.normal(size=(D, D)) * 0.05
    M *= 0.9 / max(np.abs(np.linalg.eigvals(M)))
    C = rng.normal(size=(D, D))
    cov = C @ C.T / D + 0.1 * np.eye(D)
    Lc = np.linalg.cholesky(cov)
    seqs = []
    for _ in range(n_seq):
        X = np.zeros((T, D))
        X[0] = Lc @ rng.standard_normal(D)
        for t in range(1, T):
            X[t] = M @ X[t - 1] + Lc @ rng.standard_normal(D) * 0.5
        seqs.append(X)
    return seqs


def _null_fp_rate(maker, null_gen, seed, n_rep, n_surrogates, max_iter,
                  fp_allowance):
    """Shared null-calibration loop: fit k=2 HMM on data and on null
    surrogates, flag if data mean run length exceeds the null max."""
    fp = 0
    details = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + 31 * rep)
        seqs = maker(rng)
        hmm = GaussianHMM(2, seed=seed + 2000, max_iter=max_iter).fit(seqs)
        data_mrl = persistence_stats([hmm.viterbi(X)[0] for X in seqs])["mean_run_length"]
        surr_mrl = []
        for _ in range(n_surrogates):
            sseqs = null_gen(seqs, rng)
            sh = GaussianHMM(2, seed=seed + 2000, max_iter=max_iter).fit(sseqs)
            surr_mrl.append(persistence_stats([sh.viterbi(X)[0] for X in sseqs])["mean_run_length"])
        outside = data_mrl > max(surr_mrl)
        fp += int(outside)
        details.append({"rep": rep, "data_mrl": float(data_mrl),
                        "surr_max": float(max(surr_mrl)), "outside": bool(outside)})
    return fp, details, fp_allowance


def t2_null_calibration(seed=DEFAULT_SEED, n_rep=20, n_surrogates=DEFAULT_N_SURROGATES,
                        max_iter=200):
    """Stationary AR(1): the phase-null band must contain the data
    statistic in (nearly) all repetitions — the pipeline must not invent
    regimes from a stationary linear process."""
    def maker(rng):
        return _stationary_ar1(rng, 12, 64, 9)

    fp, details, allow = _null_fp_rate(
        maker,
        lambda seqs, rng: [multivariate_phase_surrogate(X, rng) for X in seqs],
        seed, n_rep, n_surrogates, max_iter, fp_allowance=max(1, n_rep // 10))
    ok = bool(fp <= allow)
    return {"name": "T2_null_calibration", "ok": ok,
            "false_positives": int(fp), "n_rep": int(n_rep),
            "allowance": int(allow), "details": details}


def t3_drift_null_calibration(seed=DEFAULT_SEED, n_rep=10, n_surrogates=DEFAULT_N_SURROGATES,
                              max_iter=200):
    """Pure continuous drift + stationary noise (NO discrete regimes): the
    drift-null band must contain the data run length in (nearly) all
    repetitions."""
    def maker(rng):
        D = 9
        seqs = []
        for c in range(12):
            t = np.linspace(0, 1, 64)
            drift = np.stack([np.sin(2 * np.pi * (t + 0.13 * c) + d) * 0.8
                              for d in range(D)], axis=1)
            seqs.append(drift + 0.4 * rng.standard_normal((64, D)))
        return seqs

    fp, details, allow = _null_fp_rate(
        maker, drift_null_series,
        seed, n_rep, n_surrogates, max_iter, fp_allowance=max(1, n_rep // 5))
    ok = bool(fp <= allow)
    return {"name": "T3_drift_null_calibration", "ok": ok,
            "false_positives": int(fp), "n_rep": int(n_rep),
            "allowance": int(allow), "details": details}


def t4_switching_map_recovery(seed=DEFAULT_SEED):
    """Planted per-regime linear maps with regime-dependent response: the
    switching-map fit recovers the maps, beats the global map out of sample,
    and shows larger error at across-transition steps."""
    rng = np.random.default_rng(seed)
    D = 6
    K = 2
    A0 = np.eye(D) * 0.7
    A1 = np.eye(D) * 0.2 + np.diag([0.3] * (D - 1), k=1)
    b0 = 0.1 * np.ones(D)
    b1 = -0.2 * np.ones(D)
    seqs, states, _ = _sample_slds(rng, n_seq=24, T=64, D=D, K=K)
    mats = []
    for X, z in zip(seqs, states):
        Y = np.zeros_like(X)
        Y[0] = X[0]
        for t in range(1, len(X)):
            A = A0 if z[t - 1] == 0 else A1
            b = b0 if z[t - 1] == 0 else b1
            Y[t] = A @ X[t - 1] + b + 0.05 * rng.standard_normal(D)
            # after a switch, the state jumps toward the new regime mean
            if z[t] != z[t - 1]:
                Y[t] += (0.8 if z[t] == 1 else -0.8) * np.ones(D)
        mats.append(Y)
    res = fit_switching_linear_maps(mats, states, K)
    m = res["test_mse"]
    ok = bool(m["local_within"] is not None
              and m["local_across"] is not None
              and (m["local_within"] + m["local_across"])
              < (m["global_within"] + m["global_across"])
              and m["local_across"] > m["local_within"])
    return {"name": "T4_switching_map_recovery", "ok": bool(ok),
            "test_mse": m,
            "regime0_spectral_radius": res["regimes"][0]["spectral_radius"],
            "regime1_spectral_radius": res["regimes"][1]["spectral_radius"]}


def t5_local_global_chi_calibration(seed=DEFAULT_SEED, n_rep=10):
    """LOCO-CV chi machinery, both directions, repeated:
    (a) chi truly occupancy-dependent -> local model favored in most seeds
        (most reps);
    (b) chi exactly global-constant -> strict ties (no spurious local wins).
    """
    chi_regime = {0: -0.012, 1: -0.004}
    dep_favors, indep_favors = [], []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + 101 * rep)
        cells = [(2030, 46), (2030, 82), (2031, 46), (2031, 82),
                 (2032, 46), (2032, 82)]
        occ = {c: {0: float(rng.uniform(0.05, 0.6)),
                   1: float(rng.uniform(0.05, 0.6))} for c in cells}
        for c in cells:
            s = sum(occ[c].values())
            occ[c] = {k: v / s for k, v in occ[c].items()}

        def make_rows(chi_of, noisy):
            rows = []
            for c in cells:
                base = chi_of(c)
                for H in (4, 16, 64):
                    dev = base + (rng.normal(0, 2e-4) if noisy else 0.0)
                    rows.append({"seed": c[0], "step": c[1], "protocol": "persistent",
                                 "H": H, "chi_dev": dev,
                                 "chi_audit": base + (rng.normal(0, 2e-4) if noisy else 0.0)})
            return rows

        dep = make_rows(lambda c: occ[c][0] * chi_regime[0]
                        + occ[c][1] * chi_regime[1], noisy=True)
        indep = make_rows(lambda c: -0.008, noisy=False)
        rules = {"min_seed_folds": 2}
        seed_of = lambda c: c[0]  # noqa: E731
        dep_favors.append(q3b_local_vs_global_chi(dep, occ, seed_of, rules)["n_seed_favor_local"])
        indep_favors.append(q3b_local_vs_global_chi(indep, occ, seed_of, rules)["n_seed_favor_local"])
    dep_mean = float(np.mean(dep_favors))
    indep_max = int(max(indep_favors))
    ok = bool(dep_mean >= 2.0 and indep_max == 0)
    return {"name": "T5_local_global_chi_calibration", "ok": bool(ok),
            "dependent_favor_local_seeds_mean": dep_mean,
            "dependent_favor_local_seeds": [int(x) for x in dep_favors],
            "independent_favor_local_seeds_max": indep_max}


def t6_metric_gate():
    """Response-prediction errors computed with the frozen a0c metric."""
    from fr.a0c_adamw_response_propagation_v01 import _metric

    exact = _metric([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    shifted = _metric([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    keys = {"n", "corr", "R2", "MAE", "RMSE", "mean_relative_error",
            "slope_observed_on_prediction", "intercept_observed_on_prediction"}
    ok = bool(keys.issubset(set(exact))
              and exact["MAE"] == 0.0 and exact["RMSE"] == 0.0 and exact["R2"] == 1.0
              and abs(shifted["MAE"] - 1.0) < 1e-12)
    return {"name": "T6_metric_gate", "ok": bool(ok),
            "exact_mae": exact["MAE"], "shifted_mae": shifted["MAE"]}


def run_all_theory_tests():
    t1 = t1_slds_recovery()
    t2 = t2_null_calibration()
    t3 = t3_drift_null_calibration()
    t4 = t4_switching_map_recovery()
    t5 = t5_local_global_chi_calibration()
    t6 = t6_metric_gate()
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6,
            "ALL_OK": bool(all(t["ok"] for t in (t1, t2, t3, t4, t5, t6)))}
