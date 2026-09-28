"""E4A v0.8 低维 AdamW 状态筛选工具。

本模块只消费 v0.7 已生成的 full state、q 与 B/chi 审计产物，
不训练、不扰动、不重算 formal mapping。

[FR-OBSERVATION-ONLY] 所有函数只读输入对象并返回低维标量。
[FR-NATIVE-ADAMW] AdamW 一阶/二阶矩使用 PyTorch state 中已有 exp_avg/exp_avg_sq，
只按 AdamW 的 bias correction 读取 m_hat/v_hat。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


CANDIDATES = [
    "mu_q",
    "momentum_alignment",
    "update_alignment",
    "sigma_q",
    "NSR_q",
]


def _np():
    import numpy as np

    return np


def _torch():
    import torch

    return torch


def _as_float(value) -> float:
    torch = _torch()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("AdamW step 必须是标量")
        return float(value.detach().double().cpu().item())
    return float(value)


def _finite_dict(values: Mapping[str, object], keys: Sequence[str]) -> bool:
    for key in keys:
        try:
            if not math.isfinite(float(values[key])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _safe_cos(dot: float, norm_a_sq: float, norm_b_sq: float) -> float:
    denom = math.sqrt(float(norm_a_sq) * float(norm_b_sq))
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    return float(dot / denom)


def _optimizer_param_order(optimizer_state: Mapping) -> List[object]:
    params = []
    for group in optimizer_state.get("param_groups", []):
        params.extend(list(group.get("params", [])))
    return params


def _group_by_param(optimizer_state: Mapping) -> Dict[object, Mapping]:
    out = {}
    for group in optimizer_state.get("param_groups", []):
        for param_id in group.get("params", []):
            out[param_id] = group
    return out


def adamw_lowdim_from_optimizer_state(q_by_name: Mapping[str, object], optimizer_state: Mapping) -> Dict:
    """从 q 与 AdamW optimizer.state_dict() 提取 E4A 低维标量。

    q 与 optimizer param id 的对应沿用 v0.7 保存顺序：q 文件由
    named_parameters 顺序写入，AdamW param_groups 保存同一 trainable 参数顺序。
    输出不包含任何高维 m/v/q。
    """
    torch = _torch()
    q_names = list(q_by_name.keys())
    param_order = _optimizer_param_order(optimizer_state)
    if len(q_names) != len(param_order):
        raise ValueError(f"q 张量数={len(q_names)} 与 AdamW 参数数={len(param_order)} 不一致")

    state = optimizer_state.get("state", {})
    group_by_param = _group_by_param(optimizer_state)
    q_norm_sq = 0.0
    m_norm_sq = 0.0
    update_norm_sq = 0.0
    dot_q_m = 0.0
    dot_q_update = 0.0
    mu_q = 0.0
    lrs = []
    eps_values = []
    beta1_values = []
    beta2_values = []
    steps = []

    # [FR-OBSERVATION-ONLY] 每个 tensor 只在 CPU 上临时读取，绝不写回 state。
    for name, param_id in zip(q_names, param_order):
        if param_id not in state:
            raise KeyError(f"AdamW state 缺少参数 id: {param_id!r}")
        if param_id not in group_by_param:
            raise KeyError(f"AdamW param_group 缺少参数 id: {param_id!r}")
        slot = state[param_id]
        group = group_by_param[param_id]
        if "exp_avg" not in slot or "exp_avg_sq" not in slot:
            raise KeyError(f"AdamW state[{param_id!r}] 缺少 exp_avg/exp_avg_sq")

        beta1, beta2 = group.get("betas", (0.9, 0.999))
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(group.get("eps", 1e-8))
        step = _as_float(slot.get("step", 0.0))
        if step <= 0.0:
            raise ValueError(f"AdamW state[{param_id!r}] step 非正：{step}")
        b1_corr = 1.0 - beta1**step
        b2_corr = 1.0 - beta2**step
        if b1_corr <= 0.0 or b2_corr <= 0.0:
            raise ValueError(f"AdamW state[{param_id!r}] bias correction 无效")

        q = q_by_name[name].detach().to(dtype=torch.float64, device="cpu")
        m_hat = slot["exp_avg"].detach().to(dtype=torch.float64, device="cpu") / b1_corr
        v_hat = slot["exp_avg_sq"].detach().to(dtype=torch.float64, device="cpu") / b2_corr
        if q.shape != m_hat.shape or q.shape != v_hat.shape:
            raise ValueError(
                f"{name} 的 q/m/v 形状不一致：{tuple(q.shape)}, {tuple(m_hat.shape)}, {tuple(v_hat.shape)}"
            )

        denom = torch.sqrt(v_hat) + eps
        update = m_hat / denom
        q_sq_tensor = q * q

        q_norm_sq += float(q_sq_tensor.sum().item())
        m_norm_sq += float((m_hat * m_hat).sum().item())
        update_norm_sq += float((update * update).sum().item())
        dot_q_m += float((q * m_hat).sum().item())
        dot_q_update += float((q * update).sum().item())
        mu_q += float((q_sq_tensor / denom).sum().item())
        lrs.append(float(group.get("lr", math.nan)))
        eps_values.append(eps)
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        steps.append(step)

    out = {
        "mu_q": float(mu_q),
        "momentum_alignment": _safe_cos(dot_q_m, q_norm_sq, m_norm_sq),
        "update_alignment": _safe_cos(dot_q_update, q_norm_sq, update_norm_sq),
        "q_norm_from_tensor": math.sqrt(q_norm_sq) if q_norm_sq >= 0.0 else math.nan,
        "lr": float(lrs[0]) if lrs else math.nan,
        "lr_min": float(min(lrs)) if lrs else math.nan,
        "lr_max": float(max(lrs)) if lrs else math.nan,
        "adamw_eps": float(eps_values[0]) if eps_values else math.nan,
        "adamw_beta1": float(beta1_values[0]) if beta1_values else math.nan,
        "adamw_beta2": float(beta2_values[0]) if beta2_values else math.nan,
        "adamw_step_min": float(min(steps)) if steps else math.nan,
        "adamw_step_max": float(max(steps)) if steps else math.nan,
        "n_parameter_tensors": len(q_names),
    }
    out["finite"] = _finite_dict(
        out,
        [
            "mu_q",
            "momentum_alignment",
            "update_alignment",
            "q_norm_from_tensor",
            "lr",
            "lr_min",
            "lr_max",
            "adamw_eps",
            "adamw_beta1",
            "adamw_beta2",
            "adamw_step_min",
            "adamw_step_max",
        ],
    )
    return out


def e1_e2_observation_features(
    state_record: Mapping,
    *,
    nsr_eps: float = 1e-12,
    grad_clip: float | None = None,
) -> Dict:
    """从已有 E1/E2 审计产物读取 sigma、mean_s_q 与 clipping 标量。

    [FR-BATCH-STREAM] 这里只消费已保存 stream 结果，不重新生成 batch/RNG。
    """
    np = _np()
    e1_streams = list(state_record.get("e1_streams", []))
    e2_streams = list(state_record.get("e2_streams", []))
    if not e1_streams:
        raise ValueError("E4A 需要已有 E1 stream 审计产物")
    s_vals = np.asarray([float(x["s_r"]) for x in e1_streams], dtype=np.float64)
    e1_pre = np.asarray([float(x["first_preclip_grad_norm"]) for x in e1_streams], dtype=np.float64)
    sigma_q = float(state_record["sigma_q"])
    mean_s_q = float(np.mean(s_vals))
    nsr_q = float(sigma_q / (abs(mean_s_q) + float(nsr_eps)))

    def _stream_branch_values(branch: str, field: str):
        vals = []
        for stream in e2_streams:
            branch_payload = stream.get(branch, {})
            if field in branch_payload:
                vals.append(branch_payload[field])
        return vals

    baseline_pre = [float(x) for x in _stream_branch_values("baseline", "first_preclip_grad_norm")]
    plus_pre = [float(x) for x in _stream_branch_values("plus", "first_preclip_grad_norm")]
    minus_pre = [float(x) for x in _stream_branch_values("minus", "first_preclip_grad_norm")]
    all_e2_pre = baseline_pre + plus_pre + minus_pre
    clipped_flags = []
    for branch in ("baseline", "plus", "minus"):
        clipped_flags.extend([bool(x) for x in _stream_branch_values(branch, "first_clipped")])

    clipping = dict(state_record.get("clipping", {}))
    clip_indicator = bool(
        any(clipped_flags)
        or any(float(clipping.get(k, 0.0)) > 0.0 for k in ("baseline_rate", "plus_rate", "minus_rate"))
    )
    preclip_norm = float(np.mean(baseline_pre)) if baseline_pre else float(np.mean(e1_pre))
    if grad_clip is not None and math.isfinite(float(grad_clip)):
        clip_margin = float(float(grad_clip) - preclip_norm)
    else:
        clip_margin = math.nan

    out = {
        "sigma_q": sigma_q,
        "mean_s_q": mean_s_q,
        "NSR_q": nsr_q,
        "e1_stream_count": int(len(e1_streams)),
        "e1_first_preclip_norm_mean": float(np.mean(e1_pre)),
        "e1_first_preclip_norm_std": float(np.std(e1_pre, ddof=1)) if e1_pre.size > 1 else math.nan,
        "preclip_norm": preclip_norm,
        "e2_stream_count": int(len(e2_streams)),
        "e2_baseline_first_preclip_norm_mean": float(np.mean(baseline_pre)) if baseline_pre else math.nan,
        "e2_plus_first_preclip_norm_mean": float(np.mean(plus_pre)) if plus_pre else math.nan,
        "e2_minus_first_preclip_norm_mean": float(np.mean(minus_pre)) if minus_pre else math.nan,
        "baseline_clip_rate": float(clipping.get("baseline_rate", math.nan)),
        "plus_clip_rate": float(clipping.get("plus_rate", math.nan)),
        "minus_clip_rate": float(clipping.get("minus_rate", math.nan)),
        "clip_indicator": clip_indicator,
        "clip_margin": clip_margin,
    }
    out["finite"] = _finite_dict(
        out,
        [
            "sigma_q",
            "mean_s_q",
            "NSR_q",
            "e1_first_preclip_norm_mean",
            "preclip_norm",
            "baseline_clip_rate",
            "plus_clip_rate",
            "minus_clip_rate",
        ],
    )
    return out


def _ols_beta(X, y):
    np = _np()
    beta, *_ = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)
    return beta


def _rows_for_horizon(rows: Iterable[Mapping], horizon: int) -> List[Dict]:
    b_key = f"B_H{int(horizon)}"
    chi_key = f"chi_H{int(horizon)}"
    out = []
    for row in rows:
        if b_key not in row or chi_key not in row:
            raise KeyError(f"row 缺少 {b_key}/{chi_key}")
        out.append({
            **dict(row),
            "train_seed": int(row["train_seed"]),
            "step": int(row["step"]),
            "state_fraction": float(row["state_fraction"]),
            "B": float(row[b_key]),
            "chi": float(row[chi_key]),
        })
    if not out:
        raise ValueError("E4A LOSO rows 不能为空")
    return sorted(out, key=lambda r: (int(r["train_seed"]), int(r["step"])))


def _rows_by_seed(rows: Sequence[Mapping]) -> Dict[int, List[Dict]]:
    by_seed: Dict[int, List[Dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["train_seed"]), []).append(dict(row))
    for seed in by_seed:
        by_seed[seed] = sorted(by_seed[seed], key=lambda r: (int(r["step"]), float(r["state_fraction"])))
    return by_seed


def _design(rows: Sequence[Mapping], train_seeds: Sequence[int], *, candidate: str | None = None):
    np = _np()
    seed_to_col = {int(seed): idx for idx, seed in enumerate(list(train_seeds)[1:])}
    columns = ["intercept"] + [f"seed_fe_{int(seed)}" for seed in list(train_seeds)[1:]] + [
        "state_fraction",
        "B",
    ]
    if candidate is not None:
        columns.extend([candidate, f"B_x_{candidate}"])
    X = []
    for row in rows:
        seed = int(row["train_seed"])
        x = [1.0] + [0.0] * (len(train_seeds) - 1) + [float(row["state_fraction"]), float(row["B"])]
        if seed in seed_to_col:
            x[1 + seed_to_col[seed]] = 1.0
        if candidate is not None:
            z = float(row[f"{candidate}__z"])
            x.extend([z, float(row["B"]) * z])
        X.append(x)
    return np.asarray(X, dtype=np.float64), columns


def _mae(pred, y) -> float:
    np = _np()
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


def _fold_summary(folds: Sequence[Mapping]) -> Dict:
    np = _np()
    gains = np.asarray([float(f["relative_mae_improvement"]) for f in folds], dtype=np.float64)
    finite = gains[np.isfinite(gains)]
    return {
        "mean_relative_mae_improvement": float(finite.mean()) if finite.size else math.nan,
        "median_relative_mae_improvement": float(np.median(finite)) if finite.size else math.nan,
        "folds_improved": int(sum(bool(f.get("improved", False)) for f in folds)),
        "total_folds": int(len(folds)),
    }


def baseline_loso(rows: Iterable[Mapping], *, horizon: int) -> Dict:
    """M0: chi = seed_FE + c*state_fraction + beta*B 的 8-fold LOSO。"""
    np = _np()
    data = _rows_for_horizon(rows, horizon)
    by_seed = _rows_by_seed(data)
    if len(by_seed) != 8:
        raise ValueError("E4A LOSO 需要 8 个 confirmation training seed")
    folds = []
    for held in sorted(by_seed):
        train_seeds = [seed for seed in sorted(by_seed) if seed != held]
        train = [row for seed in train_seeds for row in by_seed[seed]]
        test = by_seed[held]
        X_train, columns = _design(train, train_seeds)
        X_test, _ = _design(test, train_seeds)
        y_train = np.asarray([float(row["chi"]) for row in train], dtype=np.float64)
        y_test = np.asarray([float(row["chi"]) for row in test], dtype=np.float64)
        beta = _ols_beta(X_train, y_train)
        pred = X_test @ beta
        folds.append({
            "held_out_seed": int(held),
            "mae_M0": _mae(pred, y_test),
            "M0_columns": columns,
            "M0_coef_B": float(beta[-1]),
            "M0_coef_state_fraction": float(beta[-2]),
        })
    maes = np.asarray([float(f["mae_M0"]) for f in folds], dtype=np.float64)
    return {
        "method": "leave_one_training_seed_out",
        "model": "M0: chi = seed_FE + c*state_fraction + beta*B",
        "horizon": int(horizon),
        "folds": folds,
        "mean_mae_M0": float(maes.mean()),
        "median_mae_M0": float(np.median(maes)),
        "total_folds": int(len(folds)),
    }


def single_factor_loso(rows: Iterable[Mapping], *, candidate: str, horizon: int) -> Dict:
    """单候选 Mz 与 M0 的 LOSO 比较。

    [FR-OBSERVATION-ONLY] z 标准化只用当前训练 fold，held-out seed 不参与。
    """
    if candidate not in CANDIDATES:
        raise ValueError(f"未知 E4A candidate: {candidate}")
    np = _np()
    data = _rows_for_horizon(rows, horizon)
    by_seed = _rows_by_seed(data)
    if len(by_seed) != 8:
        raise ValueError("E4A LOSO 需要 8 个 confirmation training seed")

    folds = []
    for held in sorted(by_seed):
        train_seeds = [seed for seed in sorted(by_seed) if seed != held]
        train_base = [dict(row) for seed in train_seeds for row in by_seed[seed]]
        test_base = [dict(row) for row in by_seed[held]]
        z_train_raw = np.asarray([float(row[candidate]) for row in train_base], dtype=np.float64)
        z_mean = float(z_train_raw.mean())
        z_std_raw = float(z_train_raw.std(ddof=0))
        z_std = z_std_raw if math.isfinite(z_std_raw) and z_std_raw > 0.0 else 1.0

        for row in train_base:
            row[f"{candidate}__z"] = (float(row[candidate]) - z_mean) / z_std
        for row in test_base:
            row[f"{candidate}__z"] = (float(row[candidate]) - z_mean) / z_std

        X0_train, columns0 = _design(train_base, train_seeds)
        X0_test, _ = _design(test_base, train_seeds)
        Xz_train, columns_z = _design(train_base, train_seeds, candidate=candidate)
        Xz_test, _ = _design(test_base, train_seeds, candidate=candidate)
        y_train = np.asarray([float(row["chi"]) for row in train_base], dtype=np.float64)
        y_test = np.asarray([float(row["chi"]) for row in test_base], dtype=np.float64)

        beta0 = _ols_beta(X0_train, y_train)
        betaz = _ols_beta(Xz_train, y_train)
        pred0 = X0_test @ beta0
        predz = Xz_test @ betaz
        mae0 = _mae(pred0, y_test)
        maez = _mae(predz, y_test)
        rel = float((mae0 - maez) / mae0) if mae0 > 0.0 else math.nan
        delta = float(betaz[-1])
        if math.isfinite(delta) and delta > 0.0:
            sign = "positive"
        elif math.isfinite(delta) and delta < 0.0:
            sign = "negative"
        elif math.isfinite(delta):
            sign = "zero"
        else:
            sign = "nan"
        folds.append({
            "held_out_seed": int(held),
            "mae_M0": mae0,
            "mae_Mz": maez,
            "relative_mae_improvement": rel,
            "improved": bool(math.isfinite(rel) and rel > 0.0),
            "delta_interaction": delta,
            "delta_sign": sign,
            "gamma_candidate_main": float(betaz[-2]),
            "M0_columns": columns0,
            "Mz_columns": columns_z,
            "train_z_mean": z_mean,
            "train_z_std": z_std,
            "train_z_std_raw": z_std_raw,
            "z_standardized_degenerate": bool(z_std != z_std_raw),
        })

    summary = _fold_summary(folds)
    pos = sum(1 for f in folds if f["delta_sign"] == "positive")
    neg = sum(1 for f in folds if f["delta_sign"] == "negative")
    if pos == 0 and neg == 0:
        majority = "none"
    else:
        majority = "positive" if pos >= neg else "negative"
    summary.update({
        "method": "leave_one_training_seed_out_single_factor",
        "feature_standardization_scope": "LOSO_TRAINING_FOLD_ONLY",
        "horizon": int(horizon),
        "candidate": candidate,
        "model_M0": "chi = seed_FE + c*state_fraction + beta*B",
        "model_Mz": "chi = seed_FE + c*state_fraction + beta*B + gamma*z + delta*(B*z)",
        "folds": folds,
        "delta_positive_folds": int(pos),
        "delta_negative_folds": int(neg),
        "delta_sign_stability_count": int(max(pos, neg)),
        "delta_majority_sign": majority,
    })
    summary["exploratory_status"] = classify_candidate(summary)
    return summary


def classify_candidate(result: Mapping) -> str:
    """E4A exploratory 分类；绝不输出 confirmed。"""
    mean_gain = float(result.get("mean_relative_mae_improvement", math.nan))
    if not math.isfinite(mean_gain) or mean_gain <= 0.0:
        return "UNSUPPORTED"
    if (
        int(result.get("folds_improved", 0)) >= 6
        and int(result.get("delta_sign_stability_count", 0)) >= 6
    ):
        return "PROMISING"
    return "WEAK"


def rank_candidates(results_by_candidate: Mapping[str, Mapping]) -> List[Dict]:
    """只按 H=8 LOSO 排名候选变量。"""
    status_rank = {"PROMISING": 2, "WEAK": 1, "UNSUPPORTED": 0}
    rows = []
    for candidate, result in results_by_candidate.items():
        rows.append({
            "candidate": candidate,
            "exploratory_status": result.get("exploratory_status", classify_candidate(result)),
            "mean_relative_mae_improvement": float(result.get("mean_relative_mae_improvement", math.nan)),
            "median_relative_mae_improvement": float(result.get("median_relative_mae_improvement", math.nan)),
            "folds_improved": int(result.get("folds_improved", 0)),
            "delta_sign_stability_count": int(result.get("delta_sign_stability_count", 0)),
            "delta_majority_sign": result.get("delta_majority_sign", "none"),
        })

    def _key(row):
        gain = row["mean_relative_mae_improvement"]
        return (
            status_rank.get(row["exploratory_status"], -1),
            gain if math.isfinite(gain) else -math.inf,
            row["folds_improved"],
            row["delta_sign_stability_count"],
            row["candidate"],
        )

    return sorted(rows, key=_key, reverse=True)
