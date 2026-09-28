"""A0S response-based controllability identification (core module).

Question (mechanism validation, NOT an optimization algorithm): does the
finite-time response magnitude C_t = ||J_u(t)|| contain usable controllability
information about the achievable best-intervention gain
G_t = L_baseline - min(L_LR+, L_LR-)?

All inputs are read from FROZEN artifacts:
  - D1: A0O stage1 cells (H=4) with analytic h_eta and rollout losses L0/L+/L-
  - D2: A0Q cells (chi_lr/chi_clip/chi_wd) + A0S rollout losses (same cells)

Statistics are pure functions of those records (OLS with t-CI, Spearman,
leave-one-seed-out validation, 2x2 confusion + binomial test, Model A/B
LOSO comparison). No GPU code lives here.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats

DELTA = 0.05  # frozen a0o value (import-identical; kept scalar for labeling)
H_FIXED = 4   # frozen finite horizon


def gain_row(chi_lr: float, L0: float, L_plus: float, L_minus: float,
             chi_clip: float = 0.0, chi_wd: float = 0.0):
    """Build one controllability record from response + rollout losses."""
    best_plus = bool(L_plus < L_minus)
    return {
        "C_lr": abs(float(chi_lr)),
        "G": float(L0) - min(float(L_plus), float(L_minus)),
        "best_action": "+delta" if best_plus else "-delta",
        "predicted_action": "+delta" if chi_lr < 0 else "-delta",
        "J_norm": float(np.sqrt(chi_lr ** 2 + chi_clip ** 2 + chi_wd ** 2)),
        "C_J": float(chi_lr ** 2 + chi_clip ** 2 + chi_wd ** 2),
    }


def ols(x, y):
    """OLS y = alpha + beta x with 95% t-CI and p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    res = _stats.linregress(x, y)
    tcrit = _stats.t.ppf(0.975, n - 2)
    return {"n": n, "alpha": float(res.intercept), "beta": float(res.slope),
            "beta_ci95": [float(res.slope - tcrit * res.stderr),
                          float(res.slope + tcrit * res.stderr)],
            "beta_se": float(res.stderr),
            "p": float(res.pvalue), "r": float(res.rvalue),
            "r2": float(res.rvalue ** 2)}


def correlations(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pr = _stats.pearsonr(x, y)
    sr = _stats.spearmanr(x, y)
    return {"pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def loso_ols(groups, x, y):
    """LOSO for 1-D OLS; reports pooled held-out Pearson vs actual y."""

    def fit(xf, yf):
        a, b = np.polyfit(xf, yf, 1)
        return lambda xn: a * np.asarray(xn) + b

    groups = np.asarray(groups)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = []
    yh_l, y_l = [], []
    for g in sorted(set(groups.tolist())):
        tr, te = groups != g, groups == g
        if tr.sum() < 3 or te.sum() < 1:
            continue
        a, b = np.polyfit(x[tr], y[tr], 1)
        yh = a * x[te] + b
        folds.append({"held_out": g, "n_train": int(tr.sum()),
                      "n_test": int(te.sum()),
                      "beta": float(b), "alpha": float(a),
                      "pearson": (float(_stats.pearsonr(y[te], yh).statistic)
                                  if np.std(yh) > 0 else None),
                      "mae": float(np.mean(np.abs(y[te] - yh)))})
        yh_l.extend(yh.tolist())
        y_l.extend(y[te].tolist())
    yh = np.asarray(yh_l)
    yv = np.asarray(y_l)
    ss_res = float(np.sum((yv - yh) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    pooled_pearson = (float(_stats.pearsonr(yv, yh).statistic)
                      if np.std(yh) > 0 else None)
    return {"folds": folds,
            "pooled_r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "pooled_mae": float(np.mean(np.abs(yv - yh))),
            "pooled_pearson": pooled_pearson}


def direction_stats(pred_actions, actual_actions):
    """2x2 confusion + accuracy + binomial test vs chance 0.5."""
    pred = list(pred_actions)
    act = list(actual_actions)
    n = len(pred)
    tp = sum(1 for p, a in zip(pred, act) if p == "+delta" and a == "+delta")
    fn = sum(1 for p, a in zip(pred, act) if p == "-delta" and a == "+delta")
    fp = sum(1 for p, a in zip(pred, act) if p == "+delta" and a == "-delta")
    tn = sum(1 for p, a in zip(pred, act) if p == "-delta" and a == "-delta")
    acc = (tp + tn) / n
    bt = _stats.binomtest(tp + tn, n, 0.5)
    return {"n": n, "accuracy": float(acc),
            "confusion": {"TP(+delta/+delta)": tp, "FN(-delta/+delta)": fn,
                          "FP(+delta/-delta)": fp, "TN(-delta/-delta)": tn},
            "chance": 0.5, "binom_p": float(bt.pvalue)}


def cj_shares(rows):
    """Median per-channel share of C_J across records."""
    out = {}
    for ch in ("lr", "clip", "wd"):
        key = {"lr": "chi_lr", "clip": "chi_clip", "wd": "chi_wd"}[ch]
        num = np.array([abs(r[key]) ** 2 for r in rows], dtype=np.float64)
        den = np.array([r["C_J"] for r in rows], dtype=np.float64)
        shares = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        out[ch] = {"median_share": float(np.median(shares)),
                   "mean_share": float(np.mean(shares))}
    return out


def model_ab_loso(groups, rows):
    """Model A: G ~ |chi_lr|; Model B: G ~ [|chi_lr|, |chi_clip|, |chi_wd|].

    Leave-one-seed-out (3 folds); returns per-model LOSO MSE/R2."""
    groups = np.asarray(groups)
    y = np.asarray([r["G"] for r in rows], dtype=np.float64)
    Xa = np.asarray([[abs(r["chi_lr"])] for r in rows], dtype=np.float64)
    Xb = np.asarray([[abs(r["chi_lr"]), abs(r["chi_clip"]), abs(r["chi_wd"])]
                     for r in rows], dtype=np.float64)

    def loso(X):
        yh_l, y_l = [], []
        for g in sorted(set(groups.tolist())):
            tr, te = groups != g, groups == g
            Xd = np.column_stack([np.ones(int(tr.sum())), X[tr]])
            coef, *_ = np.linalg.lstsq(Xd, y[tr], rcond=None)
            yh = np.column_stack([np.ones(int(te.sum())), X[te]]) @ coef
            yh_l.extend(yh.tolist())
            y_l.extend(y[te].tolist())
        yh = np.asarray(yh_l)
        yv = np.asarray(y_l)
        ss_res = float(np.sum((yv - yh) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        return {"mse": float(np.mean((yv - yh) ** 2)),
                "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None}

    return {"model_A_lr_only": loso(Xa), "model_B_all_channels": loso(Xb)}


def gate_a0s1(corr, loso):
    return bool(corr["spearman"] > 0 and corr["spearman_p"] < 0.05
                and loso["pooled_pearson"] is not None
                and loso["pooled_pearson"] > 0)


def gate_a0s2(dstat):
    return bool(dstat["accuracy"] > 0.5 and dstat["binom_p"] < 0.05)
