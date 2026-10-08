"""用于 E0 重放审计的稳定状态哈希工具。

用途
----
把嵌套的模型/优化器/普通 Python 状态转换成 SHA256，用来比较两条重放轨迹
是否真的一致。

科学安全性
--------
`[FR-OBSERVATION-ONLY]`：本模块只能读取状态，不能修改状态。
"""
from __future__ import annotations

import hashlib
from typing import Any


def _update_hash(h: "hashlib._Hash", obj: Any) -> None:
    # [FR-OBSERVATION-ONLY] 只读取对象内容，禁止原地修改输入对象。
    if obj is None:
        h.update(b"N")
        return
    if isinstance(obj, (str, int, float, bool)):
        h.update(type(obj).__name__.encode())
        h.update(repr(obj).encode())
        return
    if isinstance(obj, bytes):
        h.update(b"B" + obj)
        return
    if isinstance(obj, dict):
        h.update(b"D")
        # dict 按 key 的稳定表示排序，避免插入顺序影响 hash。
        for key in sorted(obj.keys(), key=lambda x: repr(x)):
            _update_hash(h, key)
            _update_hash(h, obj[key])
        return
    if isinstance(obj, (list, tuple)):
        h.update(b"L" if isinstance(obj, list) else b"T")
        for item in obj:
            _update_hash(h, item)
        return

    try:
        import torch
        if torch.is_tensor(obj):
            # detach + contiguous + CPU，只为了稳定读取字节，不做任何训练修改。
            t = obj.detach().contiguous().cpu()
            h.update(b"TORCH")
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            # 统一按 uint8 字节视图哈希，避免部分 torch dtype 无法直接转 NumPy。
            h.update(t.view(torch.uint8).numpy().tobytes())
            return
    except Exception:
        pass

    # 其它对象退化为“类型名 + repr”表示。
    h.update(type(obj).__name__.encode())
    h.update(repr(obj).encode())


def stable_hash(obj: Any) -> str:
    """返回嵌套重放状态的 SHA256 摘要。"""
    h = hashlib.sha256()
    _update_hash(h, obj)
    return h.hexdigest()
