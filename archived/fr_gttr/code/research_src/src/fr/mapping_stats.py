"""E3 的 state-level B -> chi 映射统计。

注意：这里的科学样本单位是 state，不是 state 内的 random stream。
"""
from __future__ import annotations
import math
from typing import Sequence, Dict


def _rankdata(values):
    """不依赖 scipy 的平均秩实现。"""
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _pearson(x, y):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean(); yc = y - y.mean()
    den = float(np.sqrt(np.sum(xc*xc) * np.sum(yc*yc)))
    if den == 0:
        return math.nan
    return float(np.sum(xc*yc) / den)


def mapping_summary(B: Sequence[float], chi: Sequence[float]) -> Dict:
    """对一个固定 H 的多个 state 做 chi=a+K*B 映射。"""
    import numpy as np
    x = np.asarray(B, dtype=np.float64)
    y = np.asarray(chi, dtype=np.float64)
    if x.shape != y.shape or x.size < 4:
        raise ValueError("E3 mapping 至少需要 4 个 state")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("B/chi 存在 NaN/Inf")

    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    sse = float(np.sum((y-pred)**2))
    sst = float(np.sum((y-y.mean())**2))
    r2 = float(1.0 - sse/sst) if sst > 0 else math.nan

    pearson = _pearson(x, y)
    spearman = _pearson(_rankdata(x), _rankdata(y))

    nz = (np.abs(x) > 1e-30) & (np.abs(y) > 1e-30)
    sign_agreement = float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if nz.any() else math.nan
    ratios = y[np.abs(x) > 1e-30] / x[np.abs(x) > 1e-30]

    # Leave-one-state-out：仅作稳定性诊断，不把同一轨迹 state 当成独立 seed。
    loo_abs = []
    loo_pred = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool); mask[i] = False
        Xi = np.column_stack([np.ones(mask.sum()), x[mask]])
        bi, *_ = np.linalg.lstsq(Xi, y[mask], rcond=None)
        pi = float(bi[0] + bi[1]*x[i])
        loo_pred.append(pi)
        loo_abs.append(abs(pi-y[i]))

    return {
        'n_states': int(len(x)),
        'intercept_a': float(beta[0]),
        'slope_K': float(beta[1]),
        'pearson_r': pearson,
        'spearman_rho': spearman,
        'r2': r2,
        'sign_agreement': sign_agreement,
        'ratio_chi_over_B_median': float(np.median(ratios)),
        'ratio_chi_over_B_mad': float(np.median(np.abs(ratios-np.median(ratios)))),
        'ratio_chi_over_B_min': float(np.min(ratios)),
        'ratio_chi_over_B_max': float(np.max(ratios)),
        'loo_mae': float(np.mean(loo_abs)),
        'loo_predictions': [float(v) for v in loo_pred],
        'B_min': float(x.min()),
        'B_max': float(x.max()),
        'chi_min': float(y.min()),
        'chi_max': float(y.max()),
    }
