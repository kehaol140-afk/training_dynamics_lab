"""v0.4 Stage-C 共用运行辅助函数。"""
from __future__ import annotations
from typing import Sequence, Dict, List

from .batch_manifest import make_permutation, batchify_indices


def batches_from_train_pool(train_indices: Sequence[int], batch_size: int, seed: int, needed_micro_batches: int) -> List[List[int]]:
    """在固定 train pool 内生成未来 batch；不会碰 probe/eval。"""
    if needed_micro_batches <= 0:
        return []
    pos_perm = make_permutation(len(train_indices), int(seed))
    ordered = [int(train_indices[i]) for i in pos_perm]
    batches = batchify_indices(ordered, int(batch_size))
    if len(batches) < needed_micro_batches:
        raise RuntimeError("train pool 不足以生成所需 future batches")
    return batches[:needed_micro_batches]


def get_continuation_lr(get_lr, global_micro_step: int, total_micro_steps: int, base_lr: float) -> float:
    """沿同一预注册 pilot 学习率时间轴继续训练。"""
    return float(get_lr(int(global_micro_step), int(total_micro_steps), float(base_lr)))
