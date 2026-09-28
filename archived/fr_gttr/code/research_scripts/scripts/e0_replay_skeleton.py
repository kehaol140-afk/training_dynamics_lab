#!/usr/bin/env python3
"""E0 精确重放入口骨架。

这个脚本是干什么的？
------------------
E0 最终要验证：从完全相同的完整状态和随机流出发，两条没有任何干预的
原生 AdamW 轨迹是否能够精确重放。

为什么现在不直接运行？
------------------
MiniMind adapter 尚未根据 TASK-00 审计结果实现。因此当前脚本会故意输出
`E0_BLOCKED_BY_DESIGN`，防止在 checkpoint / RNG / DataLoader 语义未明确时
误跑实验。
"""
from pathlib import Path
import sys


def main():
    # ============================================================
    # [FR-REPLAY-CRITICAL] 重放关键代码
    #
    # E0 必须比较两条“无人工干预”的原生 AdamW rollout：
    # - 起点完整状态完全相同；
    # - stochastic stream 完全锁定；
    # - 不允许 +epsilon / -epsilon。
    # ============================================================
    adapter = Path(__file__).resolve().parents[1] / "adapters/minimind_adapter_template.py"
    txt = adapter.read_text(encoding="utf-8")
    if "TODO_AFTER_TASK00_AUDIT" in txt or "raise NotImplementedError" in txt:
        print("E0_BLOCKED_BY_DESIGN")
        print("原因：TASK-00 审计后 MiniMind adapter 仍未实现。")
        print("请先阅读：docs/TASK00_IMPLEMENTATION_AUDIT_PROMPT.md")
        return 2
    print("检测到 adapter 已实现。请运行经过人工审查的正式 E0 实现，而不是这个骨架脚本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
