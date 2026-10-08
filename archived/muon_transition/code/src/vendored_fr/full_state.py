"""E0 精确重放所需的完整训练状态保存/恢复工具。

这个模块是干什么的？
--------------------
MiniMind 自带 checkpoint 面向普通续训，会把模型权重转成 half 保存，而且不保存
完整 RNG。E0 的目标更严格：同一状态恢复两次后必须尽可能得到同一未来轨迹。

因此本模块单独保存：
- 模型原始 dtype state_dict；
- AdamW optimizer state（含 m/v/step）；
- GradScaler state；
- Python / NumPy / Torch CPU / Torch CUDA RNG；
- 由调用方提供的训练位置 metadata。

本模块不会修改 AdamW 算法，也不会修改 MiniMind 源码。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from .rng import capture_rng_state, restore_rng_state


def _unwrap_model(model: Any) -> Any:
    """去掉常见 DDP / torch.compile 外壳，仅取得真实模型对象。"""
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    return raw_model


def _clone_to_cpu(obj: Any) -> Any:
    """递归克隆状态到 CPU；Tensor 保留原 dtype，不做 half/bfloat16 强制转换。"""
    try:
        import torch
        if torch.is_tensor(obj):
            return obj.detach().clone().cpu()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def capture_full_training_state(
    model: Any,
    optimizer: Any,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """在内存中捕获一份完整训练状态。"""
    # ============================================================
    # [FR-REPLAY-CRITICAL] 完整状态关键代码
    #
    # 与 MiniMind 原生 resume 的主要区别：
    # 1) 模型权重保留原始 dtype，不降为 half；
    # 2) optimizer state 完整保存；
    # 3) RNG 完整保存；
    # 4) metadata 显式记录训练位置和 batch manifest 信息。
    # ============================================================
    raw_model = _unwrap_model(model)
    state: Dict[str, Any] = {
        "fr_state_version": "0.2",
        "model": _clone_to_cpu(raw_model.state_dict()),
        "optimizer": _clone_to_cpu(optimizer.state_dict()),
        "scaler": _clone_to_cpu(scaler.state_dict()) if scaler is not None else None,
        "metadata": copy.deepcopy(metadata or {}),
    }

    # ============================================================
    # [FR-RNG-CRITICAL] RNG 必须与模型/optimizer 属于同一个逻辑时间点。
    # ============================================================
    state["rng"] = capture_rng_state()
    return state


def save_full_training_state(
    path: str | Path,
    model: Any,
    optimizer: Any,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """捕获并把完整训练状态保存到磁盘。"""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = capture_full_training_state(model, optimizer, scaler, metadata)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return state


def load_full_training_state(
    path: str | Path,
    model: Any,
    optimizer: Any,
    scaler: Optional[Any] = None,
    *,
    restore_rng: bool = True,
    strict_model: bool = True,
) -> Dict[str, Any]:
    """从磁盘恢复完整训练状态，并返回保存时的 metadata。"""
    import torch

    # ============================================================
    # [FR-REPLAY-CRITICAL] 恢复顺序关键代码
    #
    # 先恢复 model / optimizer / scaler，最后再恢复 RNG。
    # 原因：即使未来某个 load 操作内部意外消耗 RNG，最后恢复 RNG 仍能保证
    # branch 真正起跑前处于保存时的随机状态。
    # ============================================================
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("fr_state_version") != "0.2":
        raise RuntimeError(f"不支持的 FR state 版本：{state.get('fr_state_version')}")

    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(state["model"], strict=strict_model)
    optimizer.load_state_dict(state["optimizer"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if restore_rng:
        # ========================================================
        # [FR-RNG-CRITICAL] 必须放在所有 load_state_dict 之后。
        # ========================================================
        restore_rng_state(state["rng"])

    return copy.deepcopy(state.get("metadata", {}))
