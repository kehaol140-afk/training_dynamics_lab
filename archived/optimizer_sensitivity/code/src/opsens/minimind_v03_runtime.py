"""v0.3 中 E0/E1 共用的 MiniMind 运行辅助函数。

所有函数均以“尽量贴近当前 MiniMind 原生预训练路径”为原则；不会改 minimind/。
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence


def copy_jsonl_prefix(src: Path, dst: Path, n: int) -> None:
    """复制前 n 行作为 pilot 子集。n<=0 时不应调用本函数。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def make_amp(torch, device: str, dtype_name: str):
    """构造与 MiniMind 当前语义等价的 AMP 上下文和 GradScaler。"""
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if device_type == "cpu":
        return nullcontext(), torch.amp.GradScaler("cuda", enabled=False), device_type
    # PyTorch 2.11 推荐新接口，避免旧 torch.cuda.amp 的 FutureWarning。
    return (
        torch.amp.autocast("cuda", dtype=amp_dtype),
        torch.amp.GradScaler("cuda", enabled=(dtype_name == "float16")),
        device_type,
    )


def stack_batch(ds, batch_indices: Sequence[int], device: str):
    """按显式 index manifest 取样并堆叠成一个 batch。"""
    import torch
    xs, ys = zip(*(ds[int(i)] for i in batch_indices))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def raw_model(model):
    """去掉 DDP / compile 外壳。"""
    x = getattr(model, "module", model)
    return getattr(x, "_orig_mod", x)
