"""E4B-1b / E4B-2A：动态随机分叉与低维 q 时间尺度工具。

科学边界：
- [FR-OBSERVATION-ONLY] 只读取已经完成 exact replay 的 E4B-1 q-rotation 输出；
- [FR-METRIC] 不重算 v0.7 formal B/chi；直接读取 E4B-1 state JSON 中冻结转存的 B_theta/chi_theta；
- 不运行 GPU，不修改 v0.7 locked/raw，不进入 controller。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def atomic_json(path: str | Path, obj) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(map(float, values)), dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("distribution_summary 输入为空或含非有限值")
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def load_q_rotation_rows(
    project_root: str | Path,
    cfg: Mapping,
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """读取 formal E4B-1 state JSON，构造 stream-level 与 state×H-level 表。

    返回：stream_rows, state_h_rows, source_files。
    state_h_rows 中 B/chi 直接来自 E4B-1 转存的 formal v0.7 B_theta/chi_theta。
    """
    root = Path(project_root).resolve()
    src = root / str(cfg["source_q_rotation_output"])
    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    states = list(map(int, cfg["state_steps"]))
    horizons = list(map(int, cfg["horizons"]))
    R = int(cfg["streams_per_state"])
    ddof = int(cfg["branching_metric"]["ddof"])

    stream_rows: List[Dict] = []
    state_rows: List[Dict] = []
    files: List[Path] = []

    for seed in seeds:
        for step in states:
            p = src / f"seed_{seed}" / f"state_step_{step:03d}.json"
            if not p.exists():
                raise FileNotFoundError(f"缺少 E4B-1 formal state output: {p}")
            obj = json.loads(p.read_text(encoding="utf-8"))
            files.append(p)
            streams = obj.get("streams", [])
            if not bool(obj.get("technical_pass")):
                raise RuntimeError(f"E4B-1 state technical_pass != True: {p}")
            if len(streams) != R:
                raise RuntimeError(f"E4B-1 stream count != {R}: {p}")
            if "B_theta" not in obj or "chi_theta" not in obj:
                raise RuntimeError(f"E4B-1 state 未携带 formal B_theta/chi_theta: {p}")

            by_h: Dict[int, List[float]] = {h: [] for h in horizons}
            seen_streams = set()
            for s in streams:
                sid = int(s["stream_id"])
                if sid in seen_streams:
                    raise RuntimeError(f"重复 stream_id: seed={seed}, state={step}, stream={sid}")
                seen_streams.add(sid)
                if not bool(s.get("replay_exact", {}).get("pass")):
                    raise RuntimeError(f"exact replay 未 PASS: seed={seed}, state={step}, stream={sid}")
                seen_h = set()
                for o in s.get("observations", []):
                    h = int(o["horizon"])
                    if h not in by_h:
                        continue
                    if h in seen_h:
                        raise RuntimeError(f"同一 stream horizon 重复: seed={seed}, state={step}, stream={sid}, H={h}")
                    seen_h.add(h)
                    rq = float(o["R_q"])
                    if not math.isfinite(rq):
                        raise RuntimeError("R_q 出现非有限值")
                    by_h[h].append(rq)
                    stream_rows.append({
                        "seed": seed,
                        "state_step": step,
                        "state_fraction": float(obj["state_fraction"]),
                        "stream_id": sid,
                        "horizon": h,
                        "R_q": rq,
                    })
                if seen_h != set(horizons):
                    missing = sorted(set(horizons) - seen_h)
                    raise RuntimeError(f"stream observation horizons 不完整: seed={seed},state={step},stream={sid}, missing={missing}")

            if seen_streams != set(range(R)):
                raise RuntimeError(f"stream ids 不是 0..{R-1}: seed={seed}, state={step}")

            for h in horizons:
                vals = np.asarray(by_h[h], dtype=np.float64)
                if vals.size != R:
                    raise RuntimeError(f"Rq count != {R}: seed={seed}, state={step}, H={h}")
                sm = distribution_summary(vals)
                mad = float(np.median(np.abs(vals - np.median(vals))))
                state_rows.append({
                    "seed": seed,
                    "state_step": step,
                    "state_fraction": float(obj["state_fraction"]),
                    "horizon": h,
                    "mean_Rq": sm["mean"],
                    "std_Rq": float(vals.std(ddof=ddof)),
                    "median_Rq": sm["median"],
                    "q25_Rq": sm["q25"],
                    "q75_Rq": sm["q75"],
                    "iqr_Rq": sm["q75"] - sm["q25"],
                    "mad_Rq": mad,
                    "B": float(obj["B_theta"][str(h)]),
                    "chi": float(obj["chi_theta"][str(h)]),
                })

    expected_stream = len(seeds) * len(states) * R * len(horizons)
    expected_state_h = len(seeds) * len(states) * len(horizons)
    if len(stream_rows) != expected_stream or len(state_rows) != expected_state_h:
        raise RuntimeError(
            f"数据规模不匹配 stream={len(stream_rows)}/{expected_stream}, "
            f"state_h={len(state_rows)}/{expected_state_h}"
        )
    return stream_rows, state_rows, files


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)
    return coef


def _candidate_design(
    rows: Sequence[Mapping],
    *,
    train_seed_levels: Sequence[int],
    candidate_key: str | None,
    stats: Mapping | None,
) -> Tuple[np.ndarray, Dict]:
    """构造 seed_FE + state_fraction + B (+ z + B*z)。

    [FR-METRIC] 标准化只允许由 LOSO training fold 估计。
    """
    keys = ["state_fraction", "B"] + ([candidate_key] if candidate_key else [])
    if stats is None:
        stats = {}
        for k in keys:
            v = np.asarray([float(r[k]) for r in rows], dtype=np.float64)
            if not np.isfinite(v).all():
                raise ValueError(f"{k} 含非有限值")
            sd = float(v.std(ddof=0))
            stats[k] = {"mean": float(v.mean()), "std": sd if sd > 1e-12 else 1.0}

    X = []
    for r in rows:
        sf = (float(r["state_fraction"]) - stats["state_fraction"]["mean"]) / stats["state_fraction"]["std"]
        b = (float(r["B"]) - stats["B"]["mean"]) / stats["B"]["std"]
        row = [1.0]
        for s in train_seed_levels[1:]:
            row.append(1.0 if int(r["seed"]) == int(s) else 0.0)
        row.extend([sf, b])
        if candidate_key:
            z = (float(r[candidate_key]) - stats[candidate_key]["mean"]) / stats[candidate_key]["std"]
            row.extend([z, b * z])
        X.append(row)
    return np.asarray(X, dtype=np.float64), dict(stats)


def loso_single_candidate(rows: Sequence[Mapping], candidate_key: str) -> Dict:
    """严格 leave-one-training-seed-out：M0 vs M0+z+B*z。"""
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(seeds) < 3:
        raise ValueError("LOSO 至少需要 3 个 training seed")
    folds = []
    deltas = []
    for held in seeds:
        train = [r for r in rows if int(r["seed"]) != held]
        test = [r for r in rows if int(r["seed"]) == held]
        levels = sorted({int(r["seed"]) for r in train})
        X0_tr, stats0 = _candidate_design(train, train_seed_levels=levels, candidate_key=None, stats=None)
        X0_te, _ = _candidate_design(test, train_seed_levels=levels, candidate_key=None, stats=stats0)
        X1_tr, stats1 = _candidate_design(train, train_seed_levels=levels, candidate_key=candidate_key, stats=None)
        X1_te, _ = _candidate_design(test, train_seed_levels=levels, candidate_key=candidate_key, stats=stats1)
        ytr = np.asarray([float(r["chi"]) for r in train], dtype=np.float64)
        yte = np.asarray([float(r["chi"]) for r in test], dtype=np.float64)
        c0 = _fit_ols(X0_tr, ytr)
        c1 = _fit_ols(X1_tr, ytr)
        p0 = X0_te @ c0
        p1 = X1_te @ c1
        mae0 = float(np.mean(np.abs(yte - p0)))
        mae1 = float(np.mean(np.abs(yte - p1)))
        gain = float((mae0 - mae1) / mae0) if mae0 > 0 else 0.0
        delta = float(c1[-1])
        deltas.append(delta)
        folds.append({
            "held_out_seed": held,
            "baseline_mae": mae0,
            "candidate_mae": mae1,
            "relative_mae_improvement": gain,
            "delta_interaction": delta,
            "delta_sign": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"),
        })
    gains = np.asarray([f["relative_mae_improvement"] for f in folds], dtype=np.float64)
    pos = sum(d > 0 for d in deltas)
    neg = sum(d < 0 for d in deltas)
    majority = "positive" if pos > neg else ("negative" if neg > pos else "tie")
    return {
        "candidate_key": candidate_key,
        "folds": folds,
        "mean_relative_mae_improvement": float(gains.mean()),
        "median_relative_mae_improvement": float(np.median(gains)),
        "folds_improved": int((gains > 0).sum()),
        "n_folds": len(folds),
        "delta_same_sign_folds": int(max(pos, neg)),
        "delta_majority_sign": majority,
    }


def classify_screen(screen: Mapping, screening_cfg: Mapping) -> str:
    if float(screen["mean_relative_mae_improvement"]) <= float(screening_cfg["promising_mean_relative_mae_gain_gt"]):
        return "UNSUPPORTED"
    if (
        int(screen["folds_improved"]) >= int(screening_cfg["promising_folds_improved_gte"])
        and int(screen["delta_same_sign_folds"]) >= int(screening_cfg["promising_interaction_same_sign_folds_gte"])
    ):
        return "PROMISING"
    return "WEAK"


def fit_tau_plateau(
    horizons: Sequence[float],
    rq_mean: Sequence[float],
    fit_cfg: Mapping,
) -> Dict:
    """确定性有界拟合：R(H)=c+(1-c)exp(-H/tau)。

    对固定 tau，c 是线性参数，可解析最小二乘并裁剪到 [c_min,c_max]；
    tau 采用固定 log grid + 邻域多轮细化，不需要 scipy，不使用随机初始化。
    """
    h = np.asarray(horizons, dtype=np.float64)
    y = np.asarray(rq_mean, dtype=np.float64)
    if h.ndim != 1 or y.ndim != 1 or h.size != y.size or h.size < 3:
        raise ValueError("horizons/rq_mean 形状错误")
    if not (np.isfinite(h).all() and np.isfinite(y).all() and np.all(h > 0)):
        raise ValueError("fit 输入含非有限值或非正 horizon")

    tau_min = float(fit_cfg["tau_min"])
    tau_max = float(fit_cfg["tau_max"])
    c_min = float(fit_cfg["c_min"])
    c_max = float(fit_cfg["c_max"])
    n_coarse = int(fit_cfg["coarse_log_grid_points"])
    n_refine = int(fit_cfg["refine_log_grid_points"])
    rounds = int(fit_cfg["refine_rounds"])
    if not (0 < tau_min < tau_max and c_min < c_max):
        raise ValueError("tau/c bounds 非法")

    def eval_grid(log_tau_grid: np.ndarray):
        tau = np.exp(log_tau_grid)
        a = np.exp(-h[:, None] / tau[None, :])
        w = 1.0 - a
        denom = np.sum(w * w, axis=0)
        numer = np.sum(w * (y[:, None] - a), axis=0)
        c = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-18)
        c = np.clip(c, c_min, c_max)
        pred = a + w * c[None, :]
        sse = np.sum((y[:, None] - pred) ** 2, axis=0)
        idx = int(np.argmin(sse))
        return idx, float(tau[idx]), float(c[idx]), float(sse[idx])

    lo = math.log(tau_min)
    hi = math.log(tau_max)
    grid = np.linspace(lo, hi, n_coarse, dtype=np.float64)
    idx, best_tau, best_c, best_sse = eval_grid(grid)

    # 在 log-space 中围绕当前最优点确定性细化。
    for _ in range(rounds):
        j0 = max(0, idx - 2)
        j1 = min(grid.size - 1, idx + 2)
        lo2 = float(grid[j0])
        hi2 = float(grid[j1])
        if hi2 <= lo2 + 1e-15:
            break
        grid = np.linspace(lo2, hi2, n_refine, dtype=np.float64)
        idx, best_tau, best_c, best_sse = eval_grid(grid)

    pred = best_c + (1.0 - best_c) * np.exp(-h / best_tau)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / sst) if sst > 1e-18 else (1.0 if best_sse <= 1e-18 else float("nan"))

    tol_frac = float(fit_cfg.get("boundary_tolerance_fraction", 0.001))
    log_range = math.log(tau_max) - math.log(tau_min)
    lt = math.log(best_tau)
    boundary_tau = (lt - math.log(tau_min) <= tol_frac * log_range) or (math.log(tau_max) - lt <= tol_frac * log_range)
    c_range = c_max - c_min
    boundary_c = (best_c - c_min <= tol_frac * c_range) or (c_max - best_c <= tol_frac * c_range)

    success = bool(math.isfinite(best_tau) and math.isfinite(best_c) and math.isfinite(rmse) and math.isfinite(r2))
    return {
        "tau_q": best_tau,
        "c": best_c,
        "rmse": rmse,
        "r2": r2,
        "fit_success": success,
        "boundary_hit_tau": bool(boundary_tau),
        "boundary_hit_c": bool(boundary_c),
        "observed_Rq": [float(v) for v in y],
        "fitted_Rq": [float(v) for v in pred],
    }


def classify_fit_quality(fits: Sequence[Mapping], fit_cfg: Mapping) -> Dict:
    r2 = np.asarray([float(f["r2"]) for f in fits if math.isfinite(float(f["r2"]))], dtype=np.float64)
    success_fraction = float(np.mean([bool(f["fit_success"]) for f in fits])) if fits else 0.0
    boundary_fraction = float(np.mean([bool(f["boundary_hit_tau"]) or bool(f["boundary_hit_c"]) for f in fits])) if fits else 1.0
    median_r2 = float(np.median(r2)) if r2.size else float("nan")
    if (
        median_r2 >= float(fit_cfg["fit_status_good_median_r2_gte"])
        and success_fraction >= float(fit_cfg["fit_status_good_success_fraction_gte"])
        and boundary_fraction <= float(fit_cfg["fit_status_good_boundary_fraction_lte"])
    ):
        status = "GOOD"
    elif (
        median_r2 >= float(fit_cfg["fit_status_mixed_median_r2_gte"])
        and success_fraction >= float(fit_cfg["fit_status_mixed_success_fraction_gte"])
    ):
        status = "MIXED"
    else:
        status = "POOR"
    return {
        "status": status,
        "median_r2": median_r2,
        "r2_summary": distribution_summary(r2) if r2.size else None,
        "success_fraction": success_fraction,
        "boundary_hit_fraction": boundary_fraction,
    }


def _group_residual_rmse(rows: Sequence[Mapping], group_key: str, min_group_n: int) -> Dict:
    """描述性 curve-collapse 指标：组内拟合 chi ~ seed_FE + state_fraction + B 后汇总 residual RMSE。"""
    groups: Dict[str, List[Mapping]] = {}
    for r in rows:
        groups.setdefault(str(r[group_key]), []).append(r)
    group_stats = []
    total_sse = 0.0
    total_n = 0
    for g in sorted(groups, key=lambda x: (float(x) if _is_float(x) else x)):
        rr = groups[g]
        if len(rr) < min_group_n:
            group_stats.append({"group": g, "n": len(rr), "used": False, "rmse": None})
            continue
        seeds = sorted({int(r["seed"]) for r in rr})
        X = []
        y = []
        for r in rr:
            row = [1.0]
            for s in seeds[1:]:
                row.append(1.0 if int(r["seed"]) == s else 0.0)
            row.extend([float(r["state_fraction"]), float(r["B"])])
            X.append(row)
            y.append(float(r["chi"]))
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        coef = _fit_ols(X, y)
        resid = y - X @ coef
        sse = float(np.sum(resid ** 2))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        total_sse += sse
        total_n += len(rr)
        group_stats.append({"group": g, "n": len(rr), "used": True, "rmse": rmse})
    pooled_rmse = float(np.sqrt(total_sse / total_n)) if total_n else float("nan")
    return {"pooled_rmse": pooled_rmse, "used_n": total_n, "groups": group_stats}


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def assign_fixed_x_bins(rows: Sequence[Mapping], edges: Sequence[float]) -> List[Dict]:
    e = np.asarray(list(map(float, edges)), dtype=np.float64)
    if e.ndim != 1 or e.size < 3 or not np.all(np.diff(e) > 0):
        raise ValueError("x_bin_edges 必须严格递增")
    out = []
    for r in rows:
        x = float(r["normalized_horizon"])
        idx = int(np.searchsorted(e, x, side="right") - 1)
        idx = max(0, min(idx, e.size - 2))
        rr = dict(r)
        rr["x_bin"] = idx
        rr["x_bin_left"] = float(e[idx])
        rr["x_bin_right"] = float(e[idx + 1])
        out.append(rr)
    return out


def curve_collapse_diagnostic(rows: Sequence[Mapping], collapse_cfg: Mapping) -> Dict:
    """比较 raw H 分组与固定 H/tau 分箱后的组内 FR residual dispersion。

    这是 post-confirmatory descriptive diagnostic，不是 confirmatory test。
    主要统计不使用 chi/B，避免 B≈0 时的数值不稳定。
    """
    min_n = int(collapse_cfg["minimum_group_n"])
    raw = _group_residual_rmse(rows, "horizon", min_n)
    binned_rows = assign_fixed_x_bins(rows, collapse_cfg["x_bin_edges"])
    norm = _group_residual_rmse(binned_rows, "x_bin", min_n)
    total_n = len(rows)
    used_fraction = float(norm["used_n"] / total_n) if total_n else 0.0
    raw_rmse = float(raw["pooled_rmse"])
    norm_rmse = float(norm["pooled_rmse"])
    gain = float((raw_rmse - norm_rmse) / raw_rmse) if raw_rmse > 0 and math.isfinite(norm_rmse) else float("nan")
    if (
        math.isfinite(gain)
        and gain >= float(collapse_cfg["supported_relative_rmse_gain_gte"])
        and used_fraction >= float(collapse_cfg["minimum_used_fraction_supported"])
    ):
        status = "SUPPORTED"
    elif (
        math.isfinite(gain)
        and gain > float(collapse_cfg["mixed_relative_rmse_gain_gt"])
        and used_fraction >= float(collapse_cfg["minimum_used_fraction_mixed"])
    ):
        status = "MIXED"
    else:
        status = "UNSUPPORTED"
    return {
        "status": status,
        "raw_horizon": raw,
        "normalized_horizon": norm,
        "relative_pooled_rmse_improvement": gain,
        "normalized_used_fraction": used_fraction,
        "x_bin_edges": list(map(float, collapse_cfg["x_bin_edges"])),
        "boundary": "Exploratory descriptive curve-collapse diagnostic; no chi/B ratio; tau_q is fitted from the same q-dynamics data.",
        "binned_rows": binned_rows,
    }
