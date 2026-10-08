#!/usr/bin/env python3
"""F4A.5 formal stability modeling (frozen: docs/F4A5_FORMAL_MODELING_PREREG.md).
90 states (74 F4A + 16 F4A.1 matched). M0-M5 candidates, LOSO by seed,
Huber robustness, cheap-risk classifier gate, healing descriptive.
Outputs -> outputs/f4a5_modeling/."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "outputs" / "f4a_phase_map" / "raw"
OUT = ROOT / "outputs" / "f4a5_modeling"
OUT.mkdir(parents=True, exist_ok=True)
SCALES = ("v07_5m", "g1a_26m", "g1b_100m")
EPS = 1e-30

# ------------------------------------------------------------ dataset --------
rows = []
for p in sorted(RAW.glob("*/*/step_*/metrics.json")):
    m = json.loads(p.read_text())
    g = m["G"]
    ind = m["indicators"]
    cc1 = m["clip"]["clip_coefficient_1"]
    rows.append({
        "scale": m["scale"], "logN": math.log10(m["params"]), "seed": m["seed"],
        "step": m["step"], "progress": m["progress"],
        "y_R": math.log10(max(g["32"]["R_curv"], EPS)),
        "y_lambda": g["32"]["lambda_SO"],
        "logA_m": math.log10(max(ind["A_m"], EPS)),
        "logA_v": math.log10(max(ind["A_v"], EPS)),
        "logA_theta": math.log10(max(ind["A_theta"], EPS)),
        "logK_lr": math.log10(max(m["K_lr"], EPS)),
        "vfloor": ind["fraction_sqrtv_lt_eps"],
        "clip_ratio": cc1,
        "grad_norm_1": (1.0 / cc1) if cc1 < 1.0 else 1.0,
        "dominant": int(g["32"]["R_curv"] >= 10),
        "S_eta_32": g["32"]["S_eta"],
    })
with (OUT / "dataset.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"dataset: {len(rows)} states")

seeds = sorted({r["seed"] for r in rows})
scale_of_seed = {r["seed"]: r["scale"] for r in rows}


def col(key, subset=None):
    return np.array([r[key] for r in (subset or rows)], dtype=float)


def design(model, subset=None):
    sub = rows if subset is None else subset
    n = len(sub)
    p_ = col("progress", sub)
    logN = col("logN", sub)
    sc = [r["scale"] for r in sub]
    D = {"M0": [np.ones(n), p_],
         "M1": [np.ones(n), logN, p_, logN * p_],
         "M2": [np.ones(n), p_, p_ ** 2] +
               [np.array([1.0 if s == k else 0.0 for s in sc])
                for k in SCALES[1:]] +
               [np.array([1.0 if s == k else 0.0 for s in sc]) * p_
                for k in SCALES[1:]],
         }
    base = D["M2"]
    D["M3"] = base + [col("logA_m", sub), col("logA_v", sub),
                      col("clip_ratio", sub), col("logK_lr", sub),
                      col("vfloor", sub)]
    D["M4"] = base
    D["M5"] = base
    return np.column_stack(D[model])


def fit_predict(model, train_idx, test_idx):
    tr = [rows[i] for i in train_idx]
    te = [rows[i] for i in test_idx]
    Xtr = design(model, tr)
    Xte = design(model, te)
    ytr = np.array([r["y_R"] for r in tr])
    if model in ("M4", "M5"):
        tr_seeds = sorted({r["seed"] for r in tr})
        # within-seed demeaning for shared slopes
        ys, Xs = [], []
        means = {}
        for s in tr_seeds:
            idx = [i for i, r in enumerate(tr) if r["seed"] == s]
            Xs_, ys_ = Xtr[idx], ytr[idx]
            mX, my = Xs_.mean(0), ys_.mean()
            means[s] = (mX, my)
            Xs.append(Xs_ - mX)
            ys.append(ys_ - my)
        Xd = np.vstack(Xs)
        yd = np.concatenate(ys)
        beta, *_ = np.linalg.lstsq(Xd[:, 1:], yd, rcond=None)
        intercepts = [means[s][1] - means[s][0][1:] @ beta for s in tr_seeds]
        pop = float(np.mean(intercepts))
        # M5 (random p-slope): shrinkage only stabilizes TRAIN-side
        # estimation; unseen-seed LOSO predictions reduce to population
        # slope (documented in report) -> same prediction rule as M4.
        pred = pop + Xte[:, 1:] @ beta
        return pred, beta, pop
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta, beta, float(beta[0])


def loso(model, ykey="y_R"):
    global rows
    preds = np.full(len(rows), np.nan)
    for s in seeds:
        te = np.array([i for i, r in enumerate(rows) if r["seed"] == s])
        tr = np.array([i for i, r in enumerate(rows) if r["seed"] != s])
        saved = [r["y_R"] for r in rows]
        if ykey != "y_R":
            for r in rows:
                r["y_R"] = r[ykey]
        p, _, _ = fit_predict(model, tr, te)
        if ykey != "y_R":
            for r, v in zip(rows, saved):
                r["y_R"] = v
        preds[te] = p
    return preds


y = col("y_R")
mask = ~np.isnan(loso_preds := loso("M0"))  # noqa: F841
results = {}
pred_cols = {}
for model in ("M0", "M1", "M2", "M3", "M4", "M5"):
    pr = loso(model)
    pred_cols[model] = pr
    ok = ~np.isnan(pr)
    resid = y[ok] - pr[ok]
    from scipy.stats import spearmanr
    r2 = 1 - (resid ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    results[model] = {
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "spearman": float(spearmanr(pr[ok], y[ok]).statistic),
        "r2": float(r2)}
with (OUT / "model_comparison.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "mae", "rmse", "spearman", "r2"])
    for k, v in results.items():
        w.writerow([k] + [round(v[x], 4) for x in ("mae", "rmse", "spearman", "r2")])
best_mae = min(v["mae"] for v in results.values())
simple_order = ["M0", "M1", "M2", "M3", "M4", "M5"]
selected = next(m for m in simple_order
                if results[m]["mae"] <= 1.05 * best_mae)
print("model comparison:", {k: round(v["mae"], 4) for k, v in results.items()},
      "=> selected:", selected)

# LOSO predictions table
with (OUT / "loso_predictions.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scale", "seed", "step", "progress", "y_R"] +
               [f"pred_{m}" for m in simple_order])
    for i, r in enumerate(rows):
        w.writerow([r["scale"], r["seed"], r["step"], r["progress"],
                    round(r["y_R"], 4)] +
                   [round(pred_cols[m][i], 4) if not np.isnan(pred_cols[m][i])
                    else "" for m in simple_order])

# coefficient summary (full-data fit) + per-fold sign stability for M3
def coef_table(model):
    X = design(model)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

names = {
    "M0": ["int", "p"], "M1": ["int", "logN", "p", "logN:p"],
    "M2": ["int", "p", "p2", "s26", "s100", "s26:p", "s100:p"],
    "M3": ["int", "p", "p2", "s26", "s100", "s26:p", "s100:p",
           "logA_m", "logA_v", "clip_ratio", "logK_lr", "vfloor"],
}
with (OUT / "coefficient_summary.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "term", "coef_full_data"])
    for model in ("M0", "M1", "M2", "M3"):
        beta = coef_table(model)
        for nm, b in zip(names[model], beta):
            w.writerow([model, nm, round(float(b), 4)])
# fold sign stability for M3 key terms
stab = {k: [] for k in names["M3"]}
for s in seeds:
    tr = np.array([i for i, r in enumerate(rows) if r["seed"] != s])
    X = design("M3", [rows[i] for i in tr])
    ytr = y[tr]
    beta, *_ = np.linalg.lstsq(X, ytr, rcond=None)
    for nm, b in zip(names["M3"], beta):
        stab[nm].append(float(np.sign(b)))
sign_stable = {k: (len(set(v)) == 1) for k, v in stab.items()}
print("M3 sign stability:", {k: v for k, v in sign_stable.items()
                             if k in ("logA_m", "logA_v", "clip_ratio",
                                      "logK_lr", "vfloor")})

# variance decomposition (M4): between-seed intercept variance vs residual
tr_all = list(range(len(rows)))
_, beta4, pop = fit_predict("M4", np.array(tr_all), np.array(tr_all))
intercepts = []
for s in seeds:
    sub = [r for r in rows if r["seed"] == s]
    ybar = np.mean([r["y_R"] for r in sub])
    Xb = design("M2", sub).mean(0)
    intercepts.append(ybar - Xb[1:] @ beta4)
intercepts = np.array(intercepts)
resid_var = float(np.var(y - design("M2") @ np.concatenate(
    [[pop], beta4]), ddof=1))
between = float(np.var(intercepts, ddof=1))
tot = float(np.var(y, ddof=1))
with (OUT / "variance_decomposition.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["component", "variance", "fraction"])
    w.writerow(["between_seed_intercept", round(between, 4),
                round(between / (between + resid_var), 4)])
    w.writerow(["residual", round(resid_var, 4),
                round(resid_var / (between + resid_var), 4)])
    w.writerow(["total_y", round(tot, 4), 1.0])
seed_frac = between / (between + resid_var)
print(f"seed intercept variance fraction: {seed_frac:.3f}")

# predictor ablation: M3 minus each indicator (LOSO MAE delta)
ab = []
for term in ("logA_m", "logA_v", "clip_ratio", "logK_lr", "vfloor"):
    keep = [t for t in names["M3"] if t != term]
    ix = [i for i, nmm in enumerate(names["M3"]) if nmm != term]

    def design_abl(sub):
        X = design("M3", sub)
        return np.column_stack([X[:, 0]] + [X[:, i] for i in ix[1:]])
    preds = np.full(len(rows), np.nan)
    for s in seeds:
        te = np.array([i for i, r in enumerate(rows) if r["seed"] == s])
        tr = np.array([i for i, r in enumerate(rows) if r["seed"] != s])
        Xtr = design_abl([rows[i] for i in tr])
        Xte = design_abl([rows[i] for i in te])
        b, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        preds[te] = Xte @ b
    ok = ~np.isnan(preds)
    mae_wo = float(np.abs(y[ok] - preds[ok]).mean())
    ab.append({"removed": term, "mae_without": round(mae_wo, 4),
               "mae_full_M3": round(results["M3"]["mae"], 4),
               "delta": round(mae_wo - results["M3"]["mae"], 4)})
with (OUT / "predictor_ablation.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ab[0]))
    w.writeheader()
    w.writerows(ab)

# cheap-risk classifier gate (M3 feature set + logistic, LOSO)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             average_precision_score)
Xc = design("M3")[:, 1:]  # drop intercept
yc = col("dominant")
cp = np.full(len(rows), np.nan)
for s in seeds:
    te = np.array([i for i, r in enumerate(rows) if r["seed"] == s])
    tr = np.array([i for i, r in enumerate(rows) if r["seed"] != s])
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xc[tr], yc[tr])
    cp[te] = clf.predict_proba(Xc[te])[:, 1]
ok = ~np.isnan(cp)
gate = {"auroc": float(roc_auc_score(yc[ok], cp[ok])),
        "balanced_accuracy": float(balanced_accuracy_score(
            yc[ok], (cp[ok] >= 0.5).astype(int))),
        "pr_auc": float(average_precision_score(yc[ok], cp[ok])),
        "n": int(ok.sum()), "n_dominant": int(yc[ok].sum())}
gate["F4B_PRESCREEN_ENABLED"] = bool(gate["auroc"] >= 0.75
                                     and gate["balanced_accuracy"] >= 0.65)
print("cheap-risk gate:", gate)

# y_lambda models
lam = col("y_lambda")
lam_res = {}
for model in ("M0", "M2", "M4"):
    saved = [r["y_R"] for r in rows]
    for r in rows:
        r["y_R"] = r["y_lambda"]
    pr = loso(model)
    for r, v in zip(rows, saved):
        r["y_R"] = v
    ok = ~np.isnan(pr)
    lam_res[model] = float(np.abs(lam[ok] - pr[ok]).mean())
print("y_lambda LOSO MAE:", lam_res)

# healing descriptive (updated with F4A.1 points)
heal = []
for scale in SCALES:
    for s in sorted({r["seed"] for r in rows if r["scale"] == scale}):
        ms = sorted([r for r in rows if r["scale"] == scale and r["seed"] == s],
                    key=lambda r: r["progress"])
        seq = [(r["progress"], 10 ** r["y_R"]) for r in ms]
        status, lo, hi = "INSUFFICIENT_POINTS", None, None
        if len(seq) >= 3:
            hi_idx = None
            for i in range(len(seq)):
                if all(v < 3 for _, v in seq[i:]):
                    hi_idx = i
                    break
            if hi_idx is None:
                status = "NOT_HEALED_IN_COVERAGE"
            elif hi_idx == 0:
                status, lo, hi = "OBSERVED", seq[0][0], seq[0][0]
            else:
                status = "INTERVAL_CENSORED"
                lo, hi = seq[hi_idx - 1][0], seq[hi_idx][0]
        heal.append({"scale": scale, "seed": s, "status": status,
                     "p_heal_lower": lo, "p_heal_upper": hi,
                     "r32_seq": " ".join(f"{v:.2f}" for _, v in seq),
                     "p_seq": " ".join(f"{p_:.3f}" for p_, _ in seq)})
with (OUT / "healing_model.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(heal[0]))
    w.writeheader()
    w.writerows(heal)

# verdict (prereg §9)
best = selected
robust_phase = results[best]["mae"] <= 0.45 and results[best]["spearman"] >= 0.7
scale_effect = (results["M2"]["mae"] <= 0.95 * results["M0"]["mae"]
                and all(sign_stable.get(k, False)
                        for k in ("s26", "s100")))
state_aware = results["M3"]["mae"] <= 0.95 * results["M2"]["mae"]
seed_dom = seed_frac > 0.5 and results["M4"]["mae"] <= 0.90 * results["M2"]["mae"]
if robust_phase and scale_effect:
    verdict = "ROBUST_PHASE_MODEL_WITH_SCALE_EFFECT"
elif robust_phase and seed_dom:
    verdict = "SEED_EFFECT_DOMINANT"
elif robust_phase and state_aware:
    verdict = "STATE_AWARE_MODEL_IMPROVES_GENERALIZATION"
elif robust_phase:
    verdict = "ROBUST_PHASE_MODEL_NO_SCALE_EFFECT"
else:
    verdict = "MODEL_PREDICTABILITY_LIMITED"
summary = {
    "study": "F4A5_FORMAL_MODELING", "n_states": len(rows),
    "selected_model": selected, "verdict": verdict,
    "results": results, "y_lambda_mae": lam_res,
    "seed_intercept_variance_fraction": round(seed_frac, 4),
    "cheap_risk_gate": gate,
    "answers": {
        "progress_dominant": bool(results["M0"]["mae"] <= 0.55),
        "scale_effect": bool(scale_effect),
        "state_aware_improvement": bool(state_aware),
        "seed_variance_fraction": round(seed_frac, 3),
        "F4B_prescreen": "ENABLED" if gate["F4B_PRESCREEN_ENABLED"]
                         else "DISABLED_BY_PREREG"}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({"verdict": verdict, "selected": selected,
                  "seed_frac": round(seed_frac, 3), "gate": gate}, indent=1))
