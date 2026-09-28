"""E4B-1：动态方向持续性 q-rotation 的小型工具集。

本模块只提供 observation-only 的方向测量和分析辅助函数。
它不改变 v0.7 B/chi，也不施加任何人工 perturbation。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_and_verify_manifest(path: str | Path, expected_sha256: str | None = None):
    """读取 v0.7 batch manifest，并校验其 batches SHA256。"""
    from .batch_manifest import manifest_sha256

    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    batches = [list(map(int, b)) for b in obj["batches"]]
    sha = manifest_sha256(batches)
    if sha != obj.get("sha256"):
        raise RuntimeError(f"manifest 内部 SHA256 校验失败: {p}")
    if expected_sha256 and sha != str(expected_sha256):
        raise RuntimeError(f"manifest 与 v0.7 raw 记录 SHA256 不一致: {p}")
    return batches, obj, sha


def resolve_recorded_path(project_root: str | Path, recorded: str | Path) -> Path:
    """解析 raw JSON 中记录的路径；优先使用原绝对路径，再尝试项目根目录相对路径。"""
    root = Path(project_root).resolve()
    p = Path(recorded)
    if p.exists():
        return p.resolve()
    if not p.is_absolute():
        q = (root / p).resolve()
        if q.exists():
            return q
    # 允许项目被移动后仍按 outputs/research/minimind 后缀恢复。
    parts = list(p.parts)
    for anchor in ("outputs", "research", "minimind"):
        if anchor in parts:
            q = root.joinpath(*parts[parts.index(anchor):])
            if q.exists():
                return q.resolve()
    raise FileNotFoundError(f"无法解析记录路径: {recorded}")


def direction_cosine(q_a: Dict[str, "object"], q_b: Dict[str, "object"]) -> float:
    """计算两个按参数名保存的全局方向 cosine。"""
    import torch

    names_a = set(q_a)
    names_b = set(q_b)
    if names_a != names_b:
        miss_a = sorted(names_b - names_a)[:5]
        miss_b = sorted(names_a - names_b)[:5]
        raise KeyError(f"方向参数名不一致, only_b={miss_a}, only_a={miss_b}")

    dot = torch.zeros((), dtype=torch.float64)
    aa = torch.zeros((), dtype=torch.float64)
    bb = torch.zeros((), dtype=torch.float64)
    for name in sorted(names_a):
        a = q_a[name].detach().double().cpu().reshape(-1)
        b = q_b[name].detach().double().cpu().reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"{name} 方向形状不一致: {a.shape} vs {b.shape}")
        dot += (a * b).sum()
        aa += (a * a).sum()
        bb += (b * b).sum()
    denom = torch.sqrt(aa * bb)
    if not torch.isfinite(denom) or float(denom) <= 0:
        raise RuntimeError("方向 cosine 分母无效")
    out = float((dot / denom).cpu())
    if out < -1.000001 or out > 1.000001:
        raise RuntimeError(f"方向 cosine 越界: {out}")
    return max(-1.0, min(1.0, out))


def probe_direction_observation(
    *, torch, model, optimizer, autocast_ctx, ds, probe_indices,
    device: str, named, stack_batch,
):
    """在固定 probe 上计算 normalized q，但不改变未来训练状态。

    [FR-OBSERVATION-ONLY] 捕获并恢复 RNG；仅用 autograd.grad 读取 probe gradient。
    [FR-RNG-CRITICAL] probe 前后的 Python/NumPy/Torch CPU/CUDA RNG 保持一致。
    """
    from .direction import direction_sha256, normalize_direction
    from .rng import capture_rng_state, restore_rng_state

    rng_before = capture_rng_state()
    was_training = bool(model.training)
    try:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        xp, yp = stack_batch(ds, probe_indices, device)
        with autocast_ctx:
            rr = model(xp, labels=yp)
            probe_loss = rr.loss + rr.aux_loss
        params = [p for _, p in named]
        grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
        q_raw = {
            n: (torch.zeros_like(p.detach()) if g is None else g.detach().clone())
            for (n, p), g in zip(named, grads)
        }
        q, grad_norm = normalize_direction(q_raw)
        sha = direction_sha256(q)
        loss = float(probe_loss.detach().float().cpu())
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
        restore_rng_state(rng_before)
    return q, {
        "q_sha256": sha,
        "probe_loss": loss,
        "probe_grad_norm_before_normalize": float(grad_norm),
    }


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    """输出 mean/std/median/q25/q75/min/max。"""
    import numpy as np

    x = np.asarray(list(map(float, values)), dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("distribution_summary 输入为空或包含非有限值")
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


def _fit_ols(X, y):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _design(rows, *, train_seed_levels, include_rq: bool, stats=None):
    """构建设计矩阵。标准化统计量必须只由 LOSO training fold 提供。"""
    import numpy as np

    if stats is None:
        arr = {
            k: np.asarray([float(r[k]) for r in rows], dtype=np.float64)
            for k in ("state_fraction", "B", "mean_Rq")
        }
        stats = {}
        for k, v in arr.items():
            sd = float(v.std(ddof=0))
            stats[k] = {"mean": float(v.mean()), "std": sd if sd > 1e-12 else 1.0}

    X = []
    for r in rows:
        sf = (float(r["state_fraction"]) - stats["state_fraction"]["mean"]) / stats["state_fraction"]["std"]
        b = (float(r["B"]) - stats["B"]["mean"]) / stats["B"]["std"]
        rq = (float(r["mean_Rq"]) - stats["mean_Rq"]["mean"]) / stats["mean_Rq"]["std"]
        row = [1.0]
        # seed FE: 训练 seed 中去掉第一个作 reference；held-out seed 的 dummy 全为0。
        for s in train_seed_levels[1:]:
            row.append(1.0 if int(r["seed"]) == int(s) else 0.0)
        row.extend([sf, b])
        if include_rq:
            row.extend([rq, b * rq])
        X.append(row)
    return np.asarray(X, dtype=np.float64), stats


def loso_conditional_screen(rows: Sequence[Dict]) -> Dict:
    """H 固定后，比较 M0 与 M_Rq 的 8-fold leave-one-training-seed-out MAE。"""
    import numpy as np

    seeds = sorted({int(r["seed"]) for r in rows})
    if len(seeds) < 3:
        raise ValueError("LOSO 至少需要 3 个 training seed")
    folds = []
    deltas = []
    for held in seeds:
        train = [r for r in rows if int(r["seed"]) != held]
        test = [r for r in rows if int(r["seed"]) == held]
        levels = sorted({int(r["seed"]) for r in train})
        X0_tr, stats = _design(train, train_seed_levels=levels, include_rq=False, stats=None)
        X0_te, _ = _design(test, train_seed_levels=levels, include_rq=False, stats=stats)
        X1_tr, _ = _design(train, train_seed_levels=levels, include_rq=True, stats=stats)
        X1_te, _ = _design(test, train_seed_levels=levels, include_rq=True, stats=stats)
        ytr = np.asarray([float(r["chi"]) for r in train], dtype=np.float64)
        yte = np.asarray([float(r["chi"]) for r in test], dtype=np.float64)
        c0 = _fit_ols(X0_tr, ytr)
        c1 = _fit_ols(X1_tr, ytr)
        p0 = X0_te @ c0
        p1 = X1_te @ c1
        mae0 = float(np.mean(np.abs(yte - p0)))
        mae1 = float(np.mean(np.abs(yte - p1)))
        gain = float((mae0 - mae1) / mae0) if mae0 > 0 else 0.0
        delta = float(c1[-1])  # standardized B*Rq interaction
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
    stable = max(pos, neg)
    return {
        "folds": folds,
        "mean_relative_mae_improvement": float(gains.mean()),
        "median_relative_mae_improvement": float(np.median(gains)),
        "folds_improved": int((gains > 0).sum()),
        "n_folds": len(folds),
        "delta_same_sign_folds": int(stable),
        "delta_majority_sign": majority,
    }
