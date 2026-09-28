"""evaluation — validation NLL 评价的薄封装。

primary 指标：eval pool（partition.eval_indices，从未参与 C0 训练）上的
逐 token cross-entropy 均值（vendored eval_observables，含 RNG 保护与
train/eval 模式恢复）。本模块不引入新的评价口径。
"""
from __future__ import annotations

from typing import Dict, List

from vendored_fr.eval_observables import evaluate_token_losses
from vendored_fr.pilot_partition import load_partition

EVAL_KEYS = ("token_count", "mean", "p95", "p99", "cvar99")


def load_partition_checked(path, expected_sha256: str) -> Dict:
    """加载 partition（内建 SHA256 校验）并核对 C0 metadata 的期望值。"""
    partition = load_partition(path)
    if partition["sha256"] != expected_sha256:
        raise RuntimeError(
            f"partition SHA256 与 C0 metadata 不一致: partition={partition['sha256']} expected={expected_sha256}"
        )
    return partition


def run_eval_nll(model, ds, eval_indices: List[int], batch_size: int, device: str, autocast_ctx) -> Dict:
    """在 eval pool 上计算逐 token NLL 统计（无副作用）。"""
    result = evaluate_token_losses(model, ds, eval_indices, batch_size, device, autocast_ctx)
    missing = [k for k in EVAL_KEYS if k not in result]
    if missing:
        raise RuntimeError(f"eval 结果缺字段: {missing}")
    if int(result["token_count"]) <= 0:
        raise RuntimeError("eval token_count 非正")
    for k in ("mean", "p95", "p99", "cvar99"):
        v = float(result[k])
        if not (v == v) or v in (float("inf"), float("-inf")):  # NaN/Inf 检查
            raise RuntimeError(f"eval 指标 {k} 非有限: {v}")
    return {k: (int(result[k]) if k == "token_count" else float(result[k])) for k in EVAL_KEYS}


def assert_eval_identical(a: Dict, b: Dict, label: str) -> None:
    """两个 eval 结果必须完全一致（同一 C0 模型 → 逐位相同）。"""
    for k in EVAL_KEYS:
        if a[k] != b[k]:
            raise RuntimeError(f"{label}: eval 字段 {k} 不一致 ({a[k]} vs {b[k]})")
