"""尚未接入 MiniMind 的梯度脉冲基础工具。

用途
----
未来在 TASK-00 / E0 完成后，实现一次性的：

- baseline：`grad <- grad`
- plus：`grad <- grad + epsilon*q`
- minus：`grad <- grad - epsilon*q`

当前状态
--------
TASK-00 已确认当前 MiniMind 的候选因果插入位置：完成梯度累积并执行 AMP unscale 之后、
gradient clipping 之前。v0.3 仍然**不调用**本模块；真正 ±epsilon 实验留到 E2。
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


def normalize_direction(q_by_name: Dict[str, "object"], eps: float = 1e-12):
    """把按参数名存储的方向 q 做全局 L2 归一化。"""
    import torch
    sq = None
    for q in q_by_name.values():
        val = q.detach().float().pow(2).sum()
        sq = val if sq is None else sq + val
    if sq is None:
        raise ValueError("q_by_name 不能为空")
    norm = torch.sqrt(sq)
    if not torch.isfinite(norm) or float(norm) <= eps:
        raise ValueError(f"q 的范数无效：{float(norm)}")
    return {k: v / norm.to(device=v.device, dtype=v.dtype) for k, v in q_by_name.items()}, float(norm)


def apply_signed_gradient_impulse_(
    named_parameters: Iterable[Tuple[str, "object"]],
    q_by_name: Dict[str, "object"],
    epsilon: float,
    sign: int,
) -> None:
    """对当前 `.grad` 原地施加带符号的一次梯度脉冲。

    注意：这个函数本身不决定“什么时候调用”。真实插入位置必须由 TASK-00 审计
    决定。
    """
    # ============================================================
    # [FR-PERTURBATION-CRITICAL] 人工扰动关键代码
    #
    # 这是未来配对因果实验中唯一允许的科学差异：
    #   baseline: grad <- grad
    #   plus:     grad <- grad + epsilon * q
    #   minus:    grad <- grad - epsilon * q
    #
    # 同一 paired stream 内，本函数不得通过任何其它路径改变：
    # - learning rate；
    # - optimizer 设置；
    # - batch 顺序；
    # - RNG；
    # - 学习率调度路径；
    # - 模型其它状态。
    #
    # TASK-00 已确认：未来 E2 仅允许在
    # gradient accumulation 完成 + AMP unscale 之后、gradient clip 之前调用。
    # v0.3（E0/E1）禁止调用本函数进入真实训练路径。
    # ============================================================
    if sign not in (-1, 0, 1):
        raise ValueError("sign 只能是 -1、0 或 +1")
    if epsilon < 0:
        raise ValueError("epsilon 必须非负")
    if sign == 0 or epsilon == 0:
        return

    for name, p in named_parameters:
        if p.grad is None:
            continue
        if name not in q_by_name:
            raise KeyError(f"存在梯度但 q 中缺少对应参数：{name}")
        q = q_by_name[name]
        if tuple(q.shape) != tuple(p.grad.shape):
            raise ValueError(f"参数 {name} 形状不匹配：grad={p.grad.shape}, q={q.shape}")
        p.grad.add_(q.to(device=p.grad.device, dtype=p.grad.dtype), alpha=float(sign) * float(epsilon))
