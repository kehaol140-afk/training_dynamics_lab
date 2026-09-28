"""A0T response-guided LR control pilot (core module, pure statistics).

Frozen pilot question: does the A0S-discovered controllability information
convert into a useful FIXED control rule? Arms: A native (u=0), B fixed
u=+delta, C response-guided u=sign(chi_lr)-dependent. Metrics, sign
consistency, C_t-terciles, and short-vs-long erosion are pure functions of
the rollout records produced by a0t_run_v01.py. No GPU code here.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats

DELTA = 0.05          # frozen a0o value (labeling only; rollouts import a0q)
H_GRID = (4, 16, 64)  # frozen contract horizon grid


def action_u(chi_lr: float) -> float:
    """Frozen controller: +delta if chi_lr < 0 else -delta."""
    return DELTA if chi_lr < 0 else -DELTA


def wilcoxon_safe(diff):
    """Signed-rank vs 0; returns None when too few non-zero differences."""
    d = np.asarray(diff, dtype=np.float64)
    nz = d[d != 0.0]
    if len(nz) < 5:
        return {"n_nonzero": int(len(nz)), "statistic": None, "p": None,
                "note": "fewer than 5 non-zero differences"}
    res = _stats.wilcoxon(d)
    return {"n_nonzero": int(len(nz)), "statistic": float(res.statistic),
            "p": float(res.pvalue)}


def gain_block(records, arm: str):
    """Per-H gain statistics for arm X in {B,C}: gain = L_A - L_X."""
    out = {}
    for H in H_GRID:
        g = np.array([r["losses"]["A"][str(H)] - r["losses"][arm][str(H)]
                      for r in records], dtype=np.float64)
        diff_cb = None
        out[str(H)] = {
            "n": len(g),
            "median": float(np.median(g)),
            "iqr": [float(np.percentile(g, 25)), float(np.percentile(g, 75))],
            "mean": float(np.mean(g)),
            "frac_positive": float(np.mean(g > 0)),
            "wilcoxon_vs0": wilcoxon_safe(g),
        }
    # C vs B paired difference at each H
    ccb = {}
    for H in H_GRID:
        d = np.array([r["losses"]["B"][str(H)] - r["losses"]["C"][str(H)]
                      for r in records], dtype=np.float64)
        ccb[str(H)] = {"median": float(np.median(d)),
                       "wilcoxon_vs0": wilcoxon_safe(d)}
    return {"gain_" + arm: out, "C_minus_B": ccb}


def sign_consistency(records, H: int = 4):
    """Fraction of cells where the response-guided arm improves on native."""
    pos = sum(1 for r in records
              if r["losses"]["A"][str(H)] > r["losses"]["C"][str(H)])
    n = len(records)
    bt = _stats.binomtest(pos, n, 0.5)
    return {"H": H, "n": n, "improved": pos,
            "fraction": float(pos / n), "binom_p": float(bt.pvalue)}


def terciles(records, H: int = 4):
    """Split each dataset's cells by C_t into top/mid/bottom terciles.

    Controller success per tercile: fraction with gain_C,H > 0 and median
    gain. Question: does controllability magnitude predict success?"""
    out = {}
    for ds in sorted({r["dataset"] for r in records}):
        sub = sorted([r for r in records if r["dataset"] == ds],
                     key=lambda r: r["C_t"])
        n = len(sub)
        k = n // 3
        parts = {"bottom": sub[:k], "middle": sub[k:n - k], "top": sub[n - k:]}
        out[ds] = {}
        for name, cells in parts.items():
            gains = np.array([r["losses"]["A"][str(H)] - r["losses"]["C"][str(H)]
                              for r in cells], dtype=np.float64)
            out[ds][name] = {
                "n": len(cells),
                "C_t_range": [float(min(r["C_t"] for r in cells)),
                              float(max(r["C_t"] for r in cells))],
                "success_frac": float(np.mean(gains > 0)),
                "median_gain": float(np.median(gains)),
            }
    return out


def erosion(records):
    """Short-vs-long: median gain_C and success fraction across H grid."""
    out = {}
    for H in H_GRID:
        gains = np.array([r["losses"]["A"][str(H)] - r["losses"]["C"][str(H)]
                          for r in records], dtype=np.float64)
        out[str(H)] = {"median_gain_C": float(np.median(gains)),
                       "success_frac": float(np.mean(gains > 0)),
                       "median_gain_B": float(np.median(
                           [r["losses"]["A"][str(H)] - r["losses"]["B"][str(H)]
                            for r in records]))}
    return out


def gate_pass(gains_c, signc, ccb_h4):
    """Frozen PASS rule from contract."""
    i = (gains_c["4"]["median"] > 0
         and gains_c["4"]["wilcoxon_vs0"]["p"] is not None
         and gains_c["4"]["wilcoxon_vs0"]["p"] < 0.05)
    ii = (signc["fraction"] > 0.5 and signc["binom_p"] < 0.05)
    iii = ccb_h4["median"] >= 0
    return {"(i)_gain_significant": bool(i), "(ii)_direction": bool(ii),
            "(iii)_not_worse_than_B": bool(iii), "PASS": bool(i and ii and iii)}
