"""FR 随机流工具：为不同 stream_id 生成可复现、彼此独立的未来随机流。

这个模块做什么
--------------
E1/E2/E3 中，我们需要满足：
- 同一个 stream_id 内，baseline / plus / minus 的未来随机性完全一致；
- 不同 stream_id 之间，未来随机性主动放开。

这里把“流编号 -> 随机种子”做成显式映射，并统一设置 Python、NumPy、Torch CPU、
Torch CUDA RNG。它不会改变模型、优化器或 batch manifest。
"""
from __future__ import annotations

import random
from typing import Dict


def derive_stream_seed(base_seed: int, stream_id: int) -> int:
    """从 base_seed 和 stream_id 确定性派生一个 63-bit 正整数种子。"""
    if stream_id < 0:
        raise ValueError("stream_id 不能为负数")
    # 使用两个大奇数混合，避免简单相邻 stream 只差 1。
    x = (int(base_seed) * 6364136223846793005 + (int(stream_id) + 1) * 1442695040888963407)
    return int(x % (2**63 - 1))


def seed_future_stream(seed: int) -> Dict[str, int]:
    """显式设置一个未来随机流，并返回记录用 metadata。"""
    # ============================================================
    # [FR-RNG-CRITICAL] 不同 stream 通过这个函数获得不同未来随机性。
    # 同一 stream 的 paired branches 必须使用完全相同的 seed。
    # ============================================================
    seed = int(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return {"stream_seed": seed}
