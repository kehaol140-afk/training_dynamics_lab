"""Stage-C pilot 的 train / train-probe / eval-probe 固定划分。

科学目的
--------
train-probe 用于构造 q；
eval-probe 用于 mean / tail 响应；
二者都必须从 MID checkpoint 的训练 pool 中排除。

这样避免“probe 名义上 held-out，但其实 warmup 训练曾经见过”的歧义。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

from .batch_manifest import make_permutation
from .hashing import stable_hash


def build_partition(dataset_len: int, seed: int, probe_size: int, eval_size: int) -> Dict:
    """构造三块互斥 partition。"""
    if dataset_len <= probe_size + eval_size:
        raise ValueError("数据量不足以划分 train/probe/eval")
    if probe_size <= 0 or eval_size <= 0:
        raise ValueError("probe_size 和 eval_size 必须为正")

    perm = make_permutation(dataset_len, int(seed))
    probe = perm[:probe_size]
    eval_idx = perm[probe_size:probe_size + eval_size]
    train = perm[probe_size + eval_size:]

    obj = {
        "partition_version": "0.4",
        "dataset_len": int(dataset_len),
        "seed": int(seed),
        "probe_indices": list(map(int, probe)),
        "eval_indices": list(map(int, eval_idx)),
        "train_indices": list(map(int, train)),
    }

    # [FR-BATCH-STREAM] 显式验证三块没有重叠。
    a, b, c = set(train), set(probe), set(eval_idx)
    if a & b or a & c or b & c:
        raise RuntimeError("train/probe/eval partition 出现重叠")
    obj["sha256"] = stable_hash({
        "train": obj["train_indices"],
        "probe": obj["probe_indices"],
        "eval": obj["eval_indices"],
    })
    return obj


def save_partition(path: str | Path, obj: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_partition(path: str | Path) -> Dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = stable_hash({
        "train": obj["train_indices"],
        "probe": obj["probe_indices"],
        "eval": obj["eval_indices"],
    })
    if expected != obj.get("sha256"):
        raise RuntimeError("partition SHA256 校验失败")
    return obj
