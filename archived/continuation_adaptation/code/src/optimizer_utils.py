"""optimizer_utils — 原生 torch.optim.AdamW 的只读观测与 state 操作工具。

本模块不实现任何新优化器；所有函数只对 `torch.optim.AdamW` 的
param_groups / state 做显式、可验证的读写：

- set_betas：在 `load_state_dict` 之后修改 param_groups 的 betas
  （load 会用 checkpoint 的 param_groups 覆盖新 optimizer 的构造参数，
   因此改 β 必须发生在 load 之后——见 tests/test_beta_modification.py）；
- reset_optimizer_moments：按通道清零 exp_avg / exp_avg_sq / step；
- optimizer_state_statistics：逐步只读统计（loss 曲线诊断用）。

有效学习率口径（docs/EXPERIMENT_PROTOCOL.md §4 冻结）：
  v̂_i = v_i / (1 - beta2^t)          （t 为该参数的 optimizer step 计数）
  denom_i = sqrt(v̂_i) + eps
  efflr_i = lr / denom_i
  gain_i  = lr * (1 - beta1) / ((1 - beta1^t) * denom_i)   （F0 审计的 G_i 口径）
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def set_betas(optimizer, beta1: float, beta2: float) -> Tuple[float, float]:
    """把所有 param_groups 的 betas 显式设为 (beta1, beta2)，并回读验证。"""
    for group in optimizer.param_groups:
        group["betas"] = (float(beta1), float(beta2))
    # 回读验证（防止任何静默失败）
    for group in optimizer.param_groups:
        got = tuple(float(x) for x in group["betas"])
        if got != (float(beta1), float(beta2)):
            raise RuntimeError(f"set_betas 未生效: 期望 {(beta1, beta2)} 实得 {got}")
    return float(beta1), float(beta2)


def _fill_step_zero(entry: Dict) -> None:
    """把单个 state entry 的 step 计数归零（兼容 tensor / float 两种存储形态）。"""
    import torch

    step = entry["step"]
    if torch.is_tensor(step):
        step.fill_(0.0)
    else:
        entry["step"] = 0.0


def reset_optimizer_moments(optimizer, *, reset_m: bool, reset_v: bool, reset_step: bool) -> Dict[str, int]:
    """按通道原位清零 AdamW moment / step 计数，返回清零的通道与条目数。"""
    n = 0
    for entry in optimizer.state.values():
        if not entry:
            continue
        n += 1
        if reset_m:
            entry["exp_avg"].zero_()
        if reset_v:
            entry["exp_avg_sq"].zero_()
        if reset_step:
            _fill_step_zero(entry)
    reset_channels = [c for c, flag in (("m", reset_m), ("v", reset_v), ("step", reset_step)) if flag]
    return {"reset_channels": reset_channels, "n_entries": n}


def build_param_name_groups(model) -> Dict[int, Tuple[str, str]]:
    """id(param) -> (参数名, 诊断分组)。分组口径：embed / attn / mlp / other。"""
    mapping: Dict[int, Tuple[str, str]] = {}
    for name, p in model.named_parameters():
        if "embed_tokens" in name:
            group = "embed"
        elif ".self_attn." in name:
            group = "attn"
        elif ".mlp." in name:
            group = "mlp"
        else:
            group = "other"
        mapping[id(p)] = (name, group)
    return mapping


def _group_summary(m_list, v_list, denom_means, efflr_parts, gain_parts, t_min, t_max) -> Dict:
    import torch

    out: Dict = {}
    if m_list:
        m = torch.cat(m_list)
        v = torch.cat(v_list)
        out["n_coords"] = int(m.numel())
        out["m_norm"] = float(torch.linalg.vector_norm(m.double()))
        out["v_norm"] = float(torch.linalg.vector_norm(v.double()))
        out["sqrt_vhat_mean"] = float(sum(denom_means) / len(denom_means)) if denom_means else None
    else:
        out.update({"n_coords": 0, "m_norm": 0.0, "v_norm": 0.0, "sqrt_vhat_mean": None})
    out["step_min"] = t_min if t_min is not None else None
    out["step_max"] = t_max if t_max is not None else None
    if efflr_parts:
        eff = torch.cat(efflr_parts).float()
        out["efflr_mean"] = float(eff.mean())
        out["efflr_median"] = float(torch.quantile(eff, 0.5))
        out["efflr_p10"] = float(torch.quantile(eff, 0.10))
        out["efflr_p90"] = float(torch.quantile(eff, 0.90))
    else:
        for k in ("efflr_mean", "efflr_median", "efflr_p10", "efflr_p90"):
            out[k] = None
    if gain_parts:
        out["gain_mean"] = float(torch.cat(gain_parts).float().mean())
    else:
        out["gain_mean"] = None
    return out


def optimizer_state_statistics(
    optimizer,
    param_name_groups: Dict[int, Tuple[str, str]],
    *,
    lr: Optional[float],
    beta1: float,
    beta2: float,
    eps: float,
) -> Dict:
    """只读统计当前 AdamW state：全局 + 分参数组的 m/v 范数与有效学习率分布。

    t == 0 的 entry（如 arm D 适配后尚未 step）不参与 efflr/gain 统计
    （bias correction 分母为 0），只计入范数。
    """
    import torch

    buckets: Dict[str, Dict] = {}
    t_min, t_max = None, None

    for p, entry in optimizer.state.items():
        if not entry:
            continue
        name, grp = param_name_groups.get(id(p), (getattr(p, "_name", "?"), "other"))
        t = float(entry["step"])
        t_min = t if t_min is None else min(t_min, t)
        t_max = t if t_max is None else max(t_max, t)

        m = entry["exp_avg"].detach()
        v = entry["exp_avg_sq"].detach()

        b = buckets.setdefault(grp, {"m": [], "v": [], "denom_mean": [], "efflr": [], "gain": []})
        b["m"].append(m.reshape(-1))
        b["v"].append(v.reshape(-1))

        if t > 0:
            vhat = v / (1.0 - beta2 ** t)
            denom = vhat.sqrt() + eps
            b["denom_mean"].append(float(denom.double().mean()))
            if lr is not None and lr > 0:
                b["efflr"].append((lr / denom).reshape(-1))
                gain = lr * (1.0 - beta1) / ((1.0 - beta1 ** t) * denom)
                b["gain"].append(gain.reshape(-1))
        else:
            b["denom_mean"].append(float(v.sqrt().double().mean()))

    groups = {g: _group_summary(b["m"], b["v"], b["denom_mean"], b["efflr"], b["gain"], t_min, t_max)
              for g, b in buckets.items()}

    all_m = [t for b in buckets.values() for t in b["m"]]
    all_v = [t for b in buckets.values() for t in b["v"]]
    all_dm = [x for b in buckets.values() for x in b["denom_mean"]]
    all_eff = [t for b in buckets.values() for t in b["efflr"]]
    all_gain = [t for b in buckets.values() for t in b["gain"]]
    overall = _group_summary(all_m, all_v, all_dm, all_eff, all_gain, t_min, t_max)
    overall["groups"] = groups
    return overall
