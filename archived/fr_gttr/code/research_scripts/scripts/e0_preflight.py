#!/usr/bin/env python3
"""E0 运行前检查。

作用：确认 TASK-00 报告、关键源码、FR full-state 与适配器都已到位。
不运行训练。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="E0 preflight")
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    required = [
        root / "research/reports/TASK00_SOURCE_AUDIT.json",
        root / "research/src/fr/full_state.py",
        root / "research/src/fr/batch_manifest.py",
        root / "research/adapters/minimind_adapter_v02.py",
        root / "minimind/trainer/train_pretrain.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("E0_PREFLIGHT_FAIL: 缺少文件")
        for p in missing:
            print(" -", p)
        return 2

    audit = json.loads(required[0].read_text(encoding="utf-8"))
    if not audit.get("fr_v02_e0_smoke_ready"):
        print("E0_PREFLIGHT_FAIL: TASK-00 未允许进入 E0 smoke")
        return 2

    print("E0_PREFLIGHT_PASS")
    print("说明：允许运行 E0 smoke；这不等于正式 E0 已通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
