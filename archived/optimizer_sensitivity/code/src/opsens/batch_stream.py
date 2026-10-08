"""确定性 stream_id / seed 映射工具。

用途
----
为每个 `stream_id` 生成一组可重复的 Python / NumPy / Torch / DataLoader seed。

重要边界
--------
本模块目前**不会修改 MiniMind DataLoader 行为**。真正的 sampler / worker / batch
manifest 如何锁定，必须在 TASK-00 后依据 MiniMind 当前实现接线。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class StreamSeeds:
    """一个随机流对应的种子集合。"""
    stream_id: int
    python_seed: int
    numpy_seed: int
    torch_seed: int
    dataloader_seed: int


def derive_stream_seeds(base_seed: int, stream_id: int) -> StreamSeeds:
    """从 base_seed 和 stream_id 确定性派生各随机源 seed。"""
    # ============================================================
    # [FR-BATCH-STREAM] batch 随机流关键代码
    #
    # 不同 stream_id：随机流应该不同，用于采样自然涨落。
    # 同一个 stream_id：baseline / plus / minus 必须复用完全相同的 seed，
    # 并在 TASK-00 后进一步复用相同 sampler / worker / batch manifest。
    #
    # TODO_AFTER_TASK00_AUDIT：
    # 把 dataloader_seed 真正绑定到 MiniMind 当前的数据采样机制。
    # ============================================================
    def derive(label: str) -> int:
        raw = f"FR|{base_seed}|{stream_id}|{label}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)

    return StreamSeeds(
        stream_id=stream_id,
        python_seed=derive("python"),
        numpy_seed=derive("numpy"),
        torch_seed=derive("torch"),
        dataloader_seed=derive("dataloader"),
    )
