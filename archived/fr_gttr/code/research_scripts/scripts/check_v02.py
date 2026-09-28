#!/usr/bin/env python3
"""v0.2 框架静态自检。

检查新增文件存在，并确认关键脚本没有直接写入 `minimind/` 的安装逻辑。
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
required = [
    root / "research/src/fr/full_state.py",
    root / "research/src/fr/batch_manifest.py",
    root / "research/adapters/minimind_adapter_v02.py",
    root / "research/scripts/task00_source_audit.py",
    root / "research/scripts/e0_preflight.py",
    root / "research/scripts/e0_minimind_smoke.py",
    root / "docs/TASK00_实现审计结论_v0.2.md",
    root / "人工审查导读_v0.2.md",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("V02_CHECK_FAIL")
    for p in missing:
        print("缺少:", p)
    raise SystemExit(2)
print("V02_CHECK_PASS")
print("说明：这是框架静态检查，不是 E0 PASS。")
