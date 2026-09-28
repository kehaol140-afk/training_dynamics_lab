"""MiniMind 原生 micro-batch 学习率时序。

为什么单独做这个模块？
----------------------
当前 MiniMind 预训练循环会在每个 micro-batch forward 之前：

    lr = get_lr(global_micro_step, total_micro_steps, base_lr)
    optimizer.param_groups[*]['lr'] = lr

即使 accumulation_steps>1，AdamW 真正消费的是 accumulation 边界最后一个
micro-batch 写入 optimizer 的 lr。

v0.5 所有 future branch 统一复刻这个时序。
"""
from __future__ import annotations


def set_native_micro_lr(optimizer, get_lr, global_micro_step: int, total_micro_steps: int, base_lr: float) -> float:
    """在当前 micro-batch 开始前写入 MiniMind 原生 learning rate。"""
    if global_micro_step <= 0:
        raise ValueError("global_micro_step 必须从 1 开始")
    if total_micro_steps <= 0:
        raise ValueError("total_micro_steps 必须为正")
    lr = float(get_lr(int(global_micro_step), int(total_micro_steps), float(base_lr)))
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr
