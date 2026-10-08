"""MiniMind 与 FR v0.2 的最小适配层。

这个文件不复制 MiniMind 的完整训练器，也不修改 `minimind/`。
它只负责：
1. 检查当前 MiniMind 源码是否仍满足我们已经审计过的 source contract；
2. 调用 FR full-state 保存/恢复；
3. 构造显式 batch manifest。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .batch_manifest import build_native_single_gpu_batches
from .full_state import load_full_training_state, save_full_training_state


class MiniMindReplayAdapterV02:
    """E0 使用的最小 MiniMind 适配器。"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.minimind_root = self.project_root / "minimind"

    def verify_source_contract(self) -> Dict[str, bool]:
        """只读检查核心训练顺序，没有通过就禁止继续 E0。"""
        # ============================================================
        # [FR-OBSERVATION-ONLY] 只读取源码文本。
        # [FR-NATIVE-ADAMW] 确认当前路径仍使用 torch.optim.AdamW。
        # ============================================================
        p = self.minimind_root / "trainer" / "train_pretrain.py"
        txt = p.read_text(encoding="utf-8")
        required = {
            "backward": "scaler.scale(loss).backward()" in txt,
            "unscale": "scaler.unscale_(optimizer)" in txt,
            "clip": "clip_grad_norm_" in txt,
            "adamw_step": "scaler.step(optimizer)" in txt,
            "zero_grad": "optimizer.zero_grad(set_to_none=True)" in txt,
            "native_adamw": "optim.AdamW(model.parameters(), lr=args.learning_rate)" in txt,
            "accumulation": "step % args.accumulation_steps == 0" in txt,
        }
        if not all(required.values()):
            bad = [k for k, v in required.items() if not v]
            raise RuntimeError(f"MiniMind source contract 不匹配，禁止继续 E0：{bad}")
        return required

    def save_full_state(self, path: str, model: Any, optimizer: Any, scaler: Any, metadata: Dict):
        """保存 FR 无损完整状态。"""
        # [FR-REPLAY-CRITICAL]
        return save_full_training_state(path, model, optimizer, scaler, metadata)

    def load_full_state(self, path: str, model: Any, optimizer: Any, scaler: Any) -> Dict:
        """恢复 FR 无损完整状态。"""
        # [FR-REPLAY-CRITICAL] [FR-RNG-CRITICAL]
        return load_full_training_state(path, model, optimizer, scaler)

    def build_native_batches(self, dataset_len: int, epoch: int, batch_size: int, skip_batches: int = 0):
        """构造 MiniMind 单 GPU非 DDP 的显式 batch manifest。"""
        # [FR-BATCH-STREAM]
        return build_native_single_gpu_batches(dataset_len, epoch, batch_size, skip_batches=skip_batches)
