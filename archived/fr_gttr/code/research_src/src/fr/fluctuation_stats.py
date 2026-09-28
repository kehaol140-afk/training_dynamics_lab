"""E1 自然涨落统计工具。只做统计，不修改训练状态。"""
from __future__ import annotations

from typing import Iterable, Dict


def summarize_projections(values: Iterable[float], grad_norms: Iterable[float]) -> Dict[str, float | int | bool | list]:
    """汇总 s_r=<q,g_r>，给出 sigma_q 与下一阶段 epsilon 候选。"""
    import math
    import numpy as np

    x = np.asarray(list(values), dtype=np.float64)
    gn = np.asarray(list(grad_norms), dtype=np.float64)
    if x.size < 2:
        raise ValueError("至少需要 2 条随机流才能估计方差")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(gn)):
        raise ValueError("projection 或 grad_norm 存在 NaN/Inf")

    mean = float(x.mean())
    sigma = float(x.std(ddof=1))
    centered = x - mean
    mean_gn = float(gn.mean())
    rel = sigma / mean_gn if mean_gn > 0 else math.nan
    distinct = int(np.unique(x).size)

    return {
        "R": int(x.size),
        "mean_s": mean,
        "sigma_q": sigma,
        "min_s": float(x.min()),
        "max_s": float(x.max()),
        "median_s": float(np.median(x)),
        "mean_grad_norm": mean_gn,
        "sigma_over_mean_grad_norm": float(rel),
        "distinct_projection_count": distinct,
        "finite": True,
        "nonzero_variance": bool(sigma > 0.0 and distinct >= 2),
        "centered_projections": [float(v) for v in centered],
        "epsilon_candidates": {
            "rho_0.25": float(0.25 * sigma),
            "rho_0.5": float(0.5 * sigma),
            "rho_1.0": float(1.0 * sigma),
        },
    }
