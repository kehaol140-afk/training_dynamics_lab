"""E1/E2 响应统计。只做统计，不修改训练状态。"""
from __future__ import annotations
import math
from typing import Sequence, Dict


def distribution_summary(values: Sequence[float]) -> Dict:
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("分布样本不足或存在 NaN/Inf")
    mean = float(x.mean())
    centered = x - mean
    sigma = float(x.std(ddof=1))
    mad = float(np.median(np.abs(x - np.median(x))))
    if sigma > 0:
        z = centered / sigma
        skew = float(np.mean(z ** 3))
        excess_kurtosis = float(np.mean(z ** 4) - 3.0)
    else:
        skew = math.nan
        excess_kurtosis = math.nan
    return {
        "n": int(x.size),
        "mean": mean,
        "std": sigma,
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
        "mad": mad,
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "distinct_count": int(np.unique(x).size),
    }


def observational_slope(s_values: Sequence[float], y_values: Sequence[float], eps: float = 1e-30) -> Dict:
    """B = Cov(delta s, Y) / Var(delta s)。"""
    import numpy as np
    s = np.asarray(s_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if s.shape != y.shape or s.size < 2:
        raise ValueError("s/y 尺寸不一致或样本不足")
    ds = s - s.mean()
    denom = float(np.mean(ds * ds))
    cov = float(np.mean(ds * (y - y.mean())))
    return {
        "B": float(cov / (denom + eps)),
        "cov": cov,
        "var_s": denom,
    }


def paired_linearity_summary(y0, yp, ym, epsilon: float) -> Dict:
    """汇总 baseline/+/- 的中心差分与奇对称误差。"""
    import numpy as np
    a = np.asarray(y0, dtype=np.float64)
    p = np.asarray(yp, dtype=np.float64)
    m = np.asarray(ym, dtype=np.float64)
    if a.shape != p.shape or a.shape != m.shape or a.size < 2:
        raise ValueError("paired 分支尺寸不一致或样本不足")
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正")

    dp = p - a
    dm = m - a
    mean_dp = float(dp.mean())
    mean_dm = float(dm.mean())
    numerator = abs(mean_dp + mean_dm)
    denominator = abs(mean_dp - mean_dm) + 1e-30
    chi_per_stream = (p - m) / (2.0 * float(epsilon))

    return {
        "mean_delta_plus": mean_dp,
        "mean_delta_minus": mean_dm,
        "odd_error": float(numerator / denominator),
        "chi": float(chi_per_stream.mean()),
        "chi_std": float(chi_per_stream.std(ddof=1)),
        "chi_per_stream": [float(v) for v in chi_per_stream],
    }
