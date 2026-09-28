"""v0.7 confirmatory FR 统计模块。

本模块只消费已经生成的 B/chi 审计结果，不读取或修改训练状态。
确认性协议边界：
- [FR-OBSERVATION-ONLY] 不改变科学数据，只做预注册统计；
- training seed 是 cluster/bootstrap/LOSO 的独立单位；
- H=8 是唯一 primary endpoint；
- secondary horizon 的 multiplicity correction 不纳入 H=8。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_BOOTSTRAP_REPS = 10000
DEFAULT_PERMUTATION_REPS = 20000
DEFAULT_RANDOM_SEED = 77070001


def _np():
    import numpy as np

    return np


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    np = _np()
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.shape != yy.shape or xx.size < 2:
        return math.nan
    xc = xx - xx.mean()
    yc = yy - yy.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom <= 0:
        return math.nan
    return float(np.sum(xc * yc) / denom)


def _rankdata(values: Sequence[float]):
    np = _np()
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _ols_beta(X, y):
    np = _np()
    XX = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    return beta


def _ols_predict(train_X, train_y, test_X):
    beta = _ols_beta(train_X, train_y)
    np = _np()
    return np.asarray(test_X, dtype=np.float64) @ beta, beta


def _as_rows(rows: Iterable[Mapping]) -> List[Dict]:
    out = []
    for row in rows:
        out.append({
            "train_seed": int(row["train_seed"]),
            "step": int(row.get("step", 0)),
            "state_fraction": float(row["state_fraction"]),
            "B": float(row["B"]),
            "chi": float(row["chi"]),
            "clipping": dict(row.get("clipping", {})),
        })
    if not out:
        raise ValueError("confirmatory rows 不能为空")
    return out


def _seed_order(rows: Sequence[Mapping]) -> List[int]:
    return sorted({int(row["train_seed"]) for row in rows})


def _rows_by_seed(rows: Sequence[Mapping]) -> Dict[int, List[Dict]]:
    by_seed: Dict[int, List[Dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["train_seed"]), []).append(dict(row))
    for seed in by_seed:
        by_seed[seed] = sorted(by_seed[seed], key=lambda x: (int(x["step"]), float(x["state_fraction"])))
    return by_seed


def _seed_fe_design(rows: Sequence[Mapping], *, include_B: bool):
    np = _np()
    seeds = _seed_order(rows)
    if len(seeds) < 2:
        raise ValueError("seed fixed effect 模型至少需要 2 个 training seed")
    seed_to_col = {seed: idx for idx, seed in enumerate(seeds[1:])}
    columns = ["intercept"] + [f"seed_fe_{seed}" for seed in seeds[1:]] + ["state_fraction"]
    if include_B:
        columns.append("B")
    X = []
    y = []
    for row in rows:
        seed = int(row["train_seed"])
        x = [1.0] + [0.0] * (len(seeds) - 1) + [float(row["state_fraction"])]
        if seed in seed_to_col:
            x[1 + seed_to_col[seed]] = 1.0
        if include_B:
            x.append(float(row["B"]))
        X.append(x)
        y.append(float(row["chi"]))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), columns, seeds


def seed_fe_adjusted_K(rows: Iterable[Mapping]) -> Dict:
    """拟合 chi = alpha_seed + c*state_fraction + K*B。

    [FR-OBSERVATION-ONLY] 这里只做 OLS 估计；不做阈值选择。
    """
    data = _as_rows(rows)
    X, y, columns, seeds = _seed_fe_design(data, include_B=True)
    beta = _ols_beta(X, y)
    pred = X @ beta
    resid = y - pred
    sse = float((resid * resid).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "model": "chi = alpha_seed + c*state_fraction + K*B",
        "K": float(beta[-1]),
        "state_fraction_coef": float(beta[-2]),
        "intercept": float(beta[0]),
        "columns": columns,
        "n_rows": len(data),
        "n_training_seeds": len(seeds),
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
    }


def cluster_bootstrap_ci_K(
    rows: Iterable[Mapping],
    *,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    alpha: float = 0.05,
) -> Dict:
    """以完整 training seed 为 cluster 的 K bootstrap CI。"""
    if reps <= 0:
        raise ValueError("bootstrap reps 必须为正")
    data = _as_rows(rows)
    np = _np()
    by_seed = _rows_by_seed(data)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("cluster bootstrap 至少需要 2 个 training seed")

    rng = np.random.default_rng(int(random_seed))
    vals = []
    for _ in range(int(reps)):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample_rows = []
        for seed in sampled:
            sample_rows.extend(by_seed[int(seed)])
        vals.append(seed_fe_adjusted_K(sample_rows)["K"])

    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return {
        "cluster_unit": "training_seed",
        "method": "cluster_bootstrap_percentile",
        "random_seed": int(random_seed),
        "reps": int(reps),
        "n_clusters": len(seeds),
        "ci95": [lo, hi],
        "bootstrap_mean_K": float(arr.mean()),
        "bootstrap_std_K": float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
    }


def residualize_within_seed(rows: Iterable[Mapping], value_key: str) -> List[float]:
    """每个 seed 内线性去除 state_fraction 趋势。"""
    data = _as_rows(rows)
    np = _np()
    residuals = [0.0] * len(data)
    by_seed: Dict[int, List[tuple[int, Dict]]] = {}
    for idx, row in enumerate(data):
        by_seed.setdefault(int(row["train_seed"]), []).append((idx, row))
    for items in by_seed.values():
        x = np.asarray([float(row["state_fraction"]) for _, row in items], dtype=np.float64)
        y = np.asarray([float(row[value_key]) for _, row in items], dtype=np.float64)
        X = np.column_stack([np.ones_like(x), x])
        beta = _ols_beta(X, y)
        res = y - X @ beta
        for (idx, _), val in zip(items, res):
            residuals[idx] = float(val)
    return residuals


def within_seed_detrended_association(rows: Iterable[Mapping]) -> Dict:
    data = _as_rows(rows)
    rb = residualize_within_seed(data, "B")
    rc = residualize_within_seed(data, "chi")
    return {
        "method": "within_seed_linear_detrend",
        "pearson": _pearson(rb, rc),
        "spearman": _spearman(rb, rc),
        "n_points": len(data),
    }


def first_difference_association(rows: Iterable[Mapping]) -> Dict:
    data = _as_rows(rows)
    db = []
    dc = []
    for items in _rows_by_seed(data).values():
        for a, b in zip(items[:-1], items[1:]):
            db.append(float(b["B"] - a["B"]))
            dc.append(float(b["chi"] - a["chi"]))
    return {
        "method": "within_seed_first_difference",
        "pearson": _pearson(db, dc),
        "spearman": _spearman(db, dc),
        "n_pairs": len(db),
    }


def circular_shift_permutation(
    rows: Iterable[Mapping],
    *,
    reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict:
    """within-seed circular-shift permutation，方向预注册为 positive。"""
    if reps <= 0:
        raise ValueError("permutation reps 必须为正")
    data = _as_rows(rows)
    np = _np()
    rb = np.asarray(residualize_within_seed(data, "B"), dtype=np.float64)
    rc = np.asarray(residualize_within_seed(data, "chi"), dtype=np.float64)
    observed = _pearson(rb, rc)
    if not math.isfinite(observed):
        return {
            "method": "within_seed_circular_shift",
            "random_seed": int(random_seed),
            "reps": int(reps),
            "observed_pearson": observed,
            "p_one_sided_positive": math.nan,
        }

    seed_indices = {}
    for idx, row in enumerate(data):
        seed_indices.setdefault(int(row["train_seed"]), []).append(idx)
    for seed, indices in seed_indices.items():
        seed_indices[seed] = sorted(indices, key=lambda idx: (int(data[idx]["step"]), float(data[idx]["state_fraction"])))

    rng = np.random.default_rng(int(random_seed))
    ge_count = 0
    for _ in range(int(reps)):
        perm = rc.copy()
        for indices in seed_indices.values():
            shift = int(rng.integers(0, len(indices)))
            vals = rc[indices]
            perm[indices] = np.roll(vals, shift)
        stat = _pearson(rb, perm)
        if math.isfinite(stat) and stat >= observed:
            ge_count += 1
    return {
        "method": "within_seed_circular_shift",
        "direction": "positive",
        "random_seed": int(random_seed),
        "reps": int(reps),
        "observed_pearson": float(observed),
        "p_one_sided_positive": float((ge_count + 1) / (int(reps) + 1)),
    }


def leave_one_training_seed_out(rows: Iterable[Mapping]) -> Dict:
    """8-fold LOSO：M0=state_fraction；M1=state_fraction+B。"""
    data = _as_rows(rows)
    np = _np()
    by_seed = _rows_by_seed(data)
    folds = []
    for held in sorted(by_seed):
        train = [row for seed, rows_ in by_seed.items() if seed != held for row in rows_]
        test = by_seed[held]
        ytr = np.asarray([row["chi"] for row in train], dtype=np.float64)
        yte = np.asarray([row["chi"] for row in test], dtype=np.float64)
        sf_tr = np.asarray([row["state_fraction"] for row in train], dtype=np.float64)
        sf_te = np.asarray([row["state_fraction"] for row in test], dtype=np.float64)
        b_tr = np.asarray([row["B"] for row in train], dtype=np.float64)
        b_te = np.asarray([row["B"] for row in test], dtype=np.float64)

        X0_tr = np.column_stack([np.ones_like(sf_tr), sf_tr])
        X0_te = np.column_stack([np.ones_like(sf_te), sf_te])
        X1_tr = np.column_stack([np.ones_like(sf_tr), sf_tr, b_tr])
        X1_te = np.column_stack([np.ones_like(sf_te), sf_te, b_te])

        p0, beta0 = _ols_predict(X0_tr, ytr, X0_te)
        p1, beta1 = _ols_predict(X1_tr, ytr, X1_te)
        mae0 = float(np.mean(np.abs(p0 - yte)))
        mae1 = float(np.mean(np.abs(p1 - yte)))
        rel = float((mae0 - mae1) / mae0) if mae0 > 0 else math.nan
        folds.append({
            "held_out_seed": int(held),
            "mae_M0_state_fraction": mae0,
            "mae_M1_state_fraction_plus_B": mae1,
            "relative_mae_improvement": rel,
            "M0_coef_state_fraction": float(beta0[1]),
            "M1_coef_state_fraction": float(beta1[1]),
            "M1_coef_B_K": float(beta1[2]),
            "improved": bool(math.isfinite(rel) and rel > 0.0),
        })

    improvements = np.asarray([fold["relative_mae_improvement"] for fold in folds], dtype=np.float64)
    finite = improvements[np.isfinite(improvements)]
    return {
        "method": "leave_one_training_seed_out",
        "folds": folds,
        "mean_relative_mae_improvement": float(finite.mean()) if finite.size else math.nan,
        "median_relative_mae_improvement": float(np.median(finite)) if finite.size else math.nan,
        "folds_improved": int(sum(fold["improved"] for fold in folds)),
        "total_folds": len(folds),
    }


def per_seed_association(rows: Iterable[Mapping]) -> Dict[str, Dict]:
    data = _as_rows(rows)
    out = {}
    for seed, items in _rows_by_seed(data).items():
        out[str(seed)] = {
            "n": len(items),
            "pearson": _pearson([row["B"] for row in items], [row["chi"] for row in items]),
            "spearman": _spearman([row["B"] for row in items], [row["chi"] for row in items]),
        }
    return out


def clipping_rates(rows: Iterable[Mapping]) -> Dict:
    data = _as_rows(rows)
    keys = ["baseline_rate", "plus_rate", "minus_rate"]
    out = {}
    for key in keys:
        vals = [float(row["clipping"][key]) for row in data if key in row.get("clipping", {})]
        out[key] = float(sum(vals) / len(vals)) if vals else math.nan
    out["n_state_records"] = len(data)
    return out


def primary_confirmation_status(analysis: Mapping, *, alpha: float = 0.05) -> Dict:
    """H=8 primary confirmation 的预注册三条件门禁。"""
    k = float(analysis["seed_fe_adjusted"]["K"])
    ci_low = float(analysis["cluster_bootstrap_ci_K"]["ci95"][0])
    residual_r = float(analysis["within_seed_detrended"]["pearson"])
    pval = float(analysis["circular_shift_permutation"]["p_one_sided_positive"])
    loso = analysis["loso"]
    mean_improvement = float(loso["mean_relative_mae_improvement"])
    folds_improved = int(loso["folds_improved"])
    total_folds = int(loso["total_folds"])

    condition_a = math.isfinite(k) and math.isfinite(ci_low) and k > 0.0 and ci_low > 0.0
    condition_b = (
        math.isfinite(residual_r)
        and math.isfinite(pval)
        and residual_r > 0.0
        and pval < float(alpha)
    )
    condition_c = (
        math.isfinite(mean_improvement)
        and mean_improvement > 0.0
        and folds_improved >= 6
        and total_folds == 8
    )

    if any(not math.isfinite(x) for x in [k, ci_low, residual_r, pval, mean_improvement]):
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if condition_a and condition_b and condition_c else "FAIL"

    return {
        "status": status,
        "H8_CONFIRMATION": "PASS" if status == "PASS" else status,
        "alpha": float(alpha),
        "conditions": {
            "A_K8_positive_cluster_ci_lower_gt_0": condition_a,
            "B_detrended_circular_shift_one_sided_p_lt_0_05": condition_b,
            "C_LOSO_mean_gain_gt_0_and_at_least_6_of_8_folds": condition_c,
        },
        "observed": {
            "K": k,
            "cluster_ci95_lower": ci_low,
            "within_seed_detrended_pearson": residual_r,
            "circular_shift_p_one_sided_positive": pval,
            "loso_mean_relative_mae_improvement": mean_improvement,
            "loso_folds_improved": folds_improved,
            "loso_total_folds": total_folds,
        },
    }


def analyze_confirmatory_horizon(
    rows: Iterable[Mapping],
    *,
    horizon: int,
    primary_horizon: int = 8,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict:
    """输出单个 horizon 的 v0.7 confirmatory 统计。"""
    data = _as_rows(rows)
    role = "PRIMARY" if int(horizon) == int(primary_horizon) else "SECONDARY"
    seed_fe = seed_fe_adjusted_K(data)
    boot = cluster_bootstrap_ci_K(
        data, reps=int(bootstrap_reps), random_seed=int(random_seed)
    )
    detrended = within_seed_detrended_association(data)
    perm = circular_shift_permutation(
        data, reps=int(permutation_reps), random_seed=int(random_seed) + int(horizon)
    )
    first_diff = first_difference_association(data)
    loso = leave_one_training_seed_out(data)
    analysis = {
        "horizon": int(horizon),
        "endpoint_role": role,
        "n_rows": len(data),
        "n_training_seeds": len(_seed_order(data)),
        "per_seed": per_seed_association(data),
        "seed_fe_adjusted": seed_fe,
        "cluster_bootstrap_ci_K": boot,
        "within_seed_detrended": detrended,
        "circular_shift_permutation": perm,
        "first_difference": first_diff,
        "loso": loso,
        "clipping_rates": clipping_rates(data),
    }
    if role == "PRIMARY":
        analysis["primary_confirmation"] = primary_confirmation_status(analysis)
    else:
        analysis["primary_confirmation"] = None
    return analysis


def holm_secondary_correction(
    p_by_horizon: Mapping[str, float],
    *,
    primary_horizon: int = 8,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Holm correction；H=8 primary endpoint 不纳入 secondary multiplicity。"""
    primary_key = str(int(primary_horizon))
    items = [
        (str(h), float(p))
        for h, p in p_by_horizon.items()
        if str(h) != primary_key and math.isfinite(float(p))
    ]
    items.sort(key=lambda item: item[1])
    m = len(items)
    corrected = {}
    running = 0.0
    for rank, (h, p) in enumerate(items, start=1):
        raw_adj = min(1.0, (m - rank + 1) * p)
        running = max(running, raw_adj)
        corrected[h] = {
            "raw_p": p,
            "holm_p": min(1.0, running),
            "significant_holm_0_05": bool(min(1.0, running) < float(alpha)),
        }
    return corrected


def analyze_confirmatory_horizons(
    rows_by_horizon: Mapping[str, Iterable[Mapping]],
    *,
    primary_horizon: int = 8,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict:
    """批量分析所有 horizon，并附加 secondary Holm correction。"""
    analyses = {}
    for h in sorted((int(k) for k in rows_by_horizon), key=int):
        analyses[str(h)] = analyze_confirmatory_horizon(
            rows_by_horizon[str(h)],
            horizon=h,
            primary_horizon=primary_horizon,
            bootstrap_reps=bootstrap_reps,
            permutation_reps=permutation_reps,
            random_seed=random_seed + h * 1000,
        )
    p_by_h = {
        h: a["circular_shift_permutation"]["p_one_sided_positive"]
        for h, a in analyses.items()
    }
    secondary_holm = holm_secondary_correction(
        p_by_h, primary_horizon=primary_horizon
    )
    for h, value in secondary_holm.items():
        analyses[h]["secondary_holm"] = value
    return {
        "primary_horizon": int(primary_horizon),
        "analysis": analyses,
        "secondary_holm": secondary_holm,
    }
