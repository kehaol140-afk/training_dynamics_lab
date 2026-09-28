"""原生 AdamW 身份与超参数只读检查工具。

用途
----
记录 optimizer 的 class、参数组和关键超参数，并检查它仍然是
`torch.optim.AdamW`。

它不是新优化器，也不会替换或修改 optimizer。
"""
from __future__ import annotations

from typing import Any, Dict


def optimizer_signature(optimizer: Any) -> Dict[str, Any]:
    """只读提取 optimizer 身份和关键参数组配置。"""
    # ============================================================
    # [FR-NATIVE-ADAMW] 原生 AdamW 保护
    # [FR-OBSERVATION-ONLY] 只读观测
    #
    # 这里只记录，不修改 optimizer。
    # ============================================================
    sig: Dict[str, Any] = {
        "class_module": optimizer.__class__.__module__,
        "class_name": optimizer.__class__.__name__,
        "param_groups": [],
    }
    for group in optimizer.param_groups:
        sig["param_groups"].append({
            "lr": group.get("lr"),
            "betas": group.get("betas"),
            "eps": group.get("eps"),
            "weight_decay": group.get("weight_decay"),
            "amsgrad": group.get("amsgrad"),
            "maximize": group.get("maximize"),
            "capturable": group.get("capturable"),
            "differentiable": group.get("differentiable"),
            "fused": group.get("fused"),
        })
    return sig


def assert_native_adamw(optimizer: Any) -> None:
    """如果 optimizer 不是原生 torch.optim.AdamW，则直接报错。"""
    # [FR-NATIVE-ADAMW] 原生 AdamW 保护
    module = optimizer.__class__.__module__
    name = optimizer.__class__.__name__
    if name != "AdamW" or not module.startswith("torch.optim"):
        raise RuntimeError(f"预期原生 torch.optim.AdamW，实际得到 {module}.{name}")
