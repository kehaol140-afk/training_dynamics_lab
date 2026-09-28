"""配对因果重放所需的随机数状态（RNG）保存/恢复工具。

用途
----
同一个 stream_id 内，baseline / plus / minus 必须拥有完全一致的随机数状态。
本模块负责 Python、NumPy、Torch CPU 与 Torch CUDA RNG 的 capture / restore。

重要边界
--------
它**不能单独保证完整重放**。DataLoader / Sampler / worker 的随机状态仍必须
在 TASK-00 中单独审计。
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict


def capture_rng_state() -> Dict[str, Any]:
    """捕获当前可访问的随机数生成器状态。"""
    # ============================================================
    # [FR-RNG-CRITICAL] 随机数关键代码
    #
    # 同一个 paired stream 内，baseline / plus / minus 在恢复后
    # 必须拥有完全相同的 stochastic state。
    #
    # 注意：
    # 本函数本身不足以保证 DataLoader/Sampler/worker 重放；
    # 那些状态必须在 TASK-00 后单独处理。
    # ============================================================
    state: Dict[str, Any] = {"python_random": random.getstate()}

    try:
        import numpy as np
        state["numpy_random"] = copy.deepcopy(np.random.get_state())
    except Exception:
        state["numpy_random"] = None

    try:
        import torch
        state["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = [x.clone() for x in torch.cuda.get_rng_state_all()]
        else:
            state["torch_cuda_all"] = None
    except Exception:
        state["torch_cpu"] = None
        state["torch_cuda_all"] = None

    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """恢复 capture_rng_state 保存的 RNG 状态。"""
    # ============================================================
    # [FR-RNG-CRITICAL] 随机数关键代码
    #
    # 这里只恢复上面明确捕获到的 RNG。
    # 不会静默重新 seed DataLoader 或 Sampler。
    # ============================================================
    random.setstate(state["python_random"])

    if state.get("numpy_random") is not None:
        import numpy as np
        np.random.set_state(state["numpy_random"])

    if state.get("torch_cpu") is not None:
        import torch
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = state.get("torch_cuda_all")
        if cuda_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("保存状态中包含 CUDA RNG，但当前恢复环境没有可用 CUDA。")
            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA RNG 设备数量不匹配：保存时={len(cuda_states)}，"
                    f"当前={torch.cuda.device_count()}"
                )
            torch.cuda.set_rng_state_all(cuda_states)
