#!/usr/bin/env python3
"""检查实验框架本身是否安装完整。

用途
----
- 检查关键文档、配置和 adapter 模板是否存在；
- 检查关键 `[FR-...]` 审计标签有没有被意外删除；
- 统计尚未完成的 `TODO_AFTER_TASK00_AUDIT` 数量。

注意
----
这个脚本通过，只表示 scaffold 文件结构正常，**绝不等价于 E0 科学重放通过**。
"""
from pathlib import Path
import sys

# 这些英文标签必须原样保留，供 grep / AI / 自动检查使用。
CRITICAL_TAGS = [
    "[FR-REPLAY-CRITICAL]",
    "[FR-RNG-CRITICAL]",
    "[FR-BATCH-STREAM]",
    "[FR-PERTURBATION-CRITICAL]",
    "[FR-NATIVE-ADAMW]",
]


def main():
    script = Path(__file__).resolve()
    research = script.parents[1]
    root = research.parent

    # 必须存在的核心文件。
    expected = [
        root / "docs/Transformer_AdamW_Fluctuation_Response_ResearchPlan_AI_v2.md",
        research / "CHANGE_AUDIT.md",
        research / "configs/e0_pilot.json",
        research / "adapters/minimind_adapter_template.py",
    ]
    missing = [str(x) for x in expected if not x.exists()]
    if missing:
        print("SCAFFOLD_FAIL 缺少以下文件：")
        print("\n".join(missing))
        return 2

    # 汇总全部 Python 源码，检查高风险审计标签是否存在。
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in research.rglob("*.py")
    )
    missing_tags = [t for t in CRITICAL_TAGS if t not in text]
    if missing_tags:
        print("SCAFFOLD_FAIL 缺少关键审计标签：", missing_tags)
        return 3

    print("SCAFFOLD_PASS")
    print("说明：这里只代表实验框架结构检查通过，不代表 E0 科学重放通过。")
    print("等待 TASK-00 审计后处理的 TODO 数量：", text.count("TODO_AFTER_TASK00_AUDIT"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
