"""MiniMind 重放适配器模板。

这个文件是干什么的？
--------------------
它是未来把 `research/` 实验框架接到真实 MiniMind 训练代码的唯一适配层。

当前为什么不实现？
----------------
因为 E0 要求完全复现原生训练语义。如果没有先审计当前服务器版本的
`train_pretrain.py`、checkpoint、DataLoader、AMP、梯度裁剪和 AdamW 流程，
直接凭经验实现会污染科学实验。

当前状态：模板；禁止正式 E0。
"""
from __future__ import annotations


class MiniMindReplayAdapter:
    """E0 精确重放所需接口。

    所有方法在 TASK-00 审计完成前故意不实现。
    这样可以防止我们对 checkpoint、DataLoader、AMP、clip 或 optimizer
    的真实语义做未经证实的假设。
    """

    def __init__(self, project_root: str):
        # 项目根目录，例如 /home/b205/projects/fluctuation_response
        self.project_root = project_root

    def audit_binding(self):
        # ============================================================
        # [FR-REPLAY-CRITICAL] 重放关键代码
        #
        # TODO_AFTER_TASK00_AUDIT：
        # 在真正接线时，必须逐项记录：
        # - 对应 MiniMind 源文件；
        # - 函数名；
        # - 关键代码位置；
        # - 为什么这样接不会改变原生训练语义。
        # ============================================================
        raise NotImplementedError("必须先完成 TASK-00 审计，才能绑定 MiniMind")

    def save_full_state(self, path: str):
        # ============================================================
        # [FR-REPLAY-CRITICAL] 重放关键代码
        # [FR-RNG-CRITICAL] 随机数关键代码
        #
        # TODO_AFTER_TASK00_AUDIT：
        # 完整状态至少要考虑：
        # theta、AdamW m/v/step、学习率/训练位置、global step、
        # RNG，以及 MiniMind 实际使用的数据/采样器状态。
        # ============================================================
        raise NotImplementedError

    def load_full_state(self, path: str):
        # ============================================================
        # [FR-REPLAY-CRITICAL] 重放关键代码
        # [FR-RNG-CRITICAL] 随机数关键代码
        #
        # 必须与 save_full_state 一一对应恢复，不能只恢复模型权重。
        # ============================================================
        raise NotImplementedError

    def build_locked_stream(self, stream_id: int):
        # ============================================================
        # [FR-BATCH-STREAM] batch 随机流关键代码
        #
        # TODO_AFTER_TASK00_AUDIT：
        # 必须证明同一个 stream_id 下的 baseline / plus / minus
        # 消耗完全相同的未来 batch。
        # ============================================================
        raise NotImplementedError

    def run_native_rollout(self, steps: int, stream_id: int):
        # ============================================================
        # [FR-NATIVE-ADAMW] 原生 AdamW 保护
        #
        # E0 阶段禁止任何人工扰动。这里只允许原生 AdamW rollout。
        # ============================================================
        raise NotImplementedError
