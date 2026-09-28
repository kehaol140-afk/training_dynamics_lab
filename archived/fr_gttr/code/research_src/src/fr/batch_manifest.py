"""显式 batch index manifest 工具。

用途
----
把“未来会用哪些样本、按什么顺序组成 batch”保存成明确的数据结构。
这样 paired branches 不需要依赖“我重新 seed 了，所以应该一样”的隐式假设。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def make_permutation(dataset_len: int, seed: int) -> List[int]:
    """使用独立 CPU Generator 生成确定性的样本排列，不污染全局 Torch RNG。"""
    import torch

    if dataset_len <= 0:
        raise ValueError("dataset_len 必须为正数")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(dataset_len, generator=g).tolist()


def batchify_indices(
    indices: Sequence[int],
    batch_size: int,
    *,
    skip_batches: int = 0,
) -> List[List[int]]:
    """按照 MiniMind SkipBatchSampler 的语义组成 batch 并跳过前若干 batch。"""
    # ============================================================
    # [FR-BATCH-STREAM] batch manifest 关键代码
    #
    # skip_batches 的单位是“完整 batch 数”，不是样本数。
    # 最后不足一个 batch 的尾批仍然保留，与 MiniMind SkipBatchSampler 一致。
    # ============================================================
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    if skip_batches < 0:
        raise ValueError("skip_batches 不能为负数")

    batches = [list(indices[i:i + batch_size]) for i in range(0, len(indices), batch_size)]
    return batches[skip_batches:]


def build_native_single_gpu_batches(
    dataset_len: int,
    epoch: int,
    batch_size: int,
    *,
    skip_batches: int = 0,
) -> List[List[int]]:
    """复现 MiniMind 单 GPU 非 DDP 的 epoch 顺序：seed=42+epoch -> randperm。"""
    # [FR-BATCH-STREAM] 当前 MiniMind 单 GPU原生路径的显式版本。
    return batchify_indices(
        make_permutation(dataset_len, 42 + int(epoch)),
        batch_size,
        skip_batches=skip_batches,
    )


def manifest_sha256(batches: Sequence[Sequence[int]]) -> str:
    """计算 batch manifest 的稳定 SHA256。"""
    payload = json.dumps(list(map(list, batches)), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_manifest(path: str | Path, batches: Sequence[Sequence[int]], metadata: Dict | None = None) -> str:
    """保存 manifest，并返回 SHA256。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = manifest_sha256(batches)
    obj = {
        "manifest_version": "0.2",
        "sha256": sha,
        "metadata": metadata or {},
        "batches": list(map(list, batches)),
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return sha
