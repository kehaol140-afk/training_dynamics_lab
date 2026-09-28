#!/usr/bin/env python3
"""v0.3 静态完整性检查。

用途：GPU 运行前先确认 v0.3 必需脚本/模块/文档已经完整安装。
本脚本不加载模型、不修改数据、不占用 GPU。
"""
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "MANUAL_REVIEW_V03.md",
        root / "RUN_ORDER_V03.md",
        root / "docs/E0_SMOKE_PASS_20260902.md",
        root / "research/scripts/e0_extended_validate.py",
        root / "research/scripts/e1_natural_fluctuation.py",
        root / "research/src/fr/direction.py",
        root / "research/src/fr/stream_rng.py",
        root / "research/src/fr/fluctuation_stats.py",
        root / "research/src/fr/minimind_v03_runtime.py",
        root / "research/configs/e0_extended_v03.json",
        root / "research/configs/e1_pilot_v03.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("V03_CHECK_FAIL")
        for p in missing:
            print("缺少:", p)
        return 2
    print("V03_CHECK_PASS")
    print("说明：这只是文件/静态检查，不等于 E0/E1 科学结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
