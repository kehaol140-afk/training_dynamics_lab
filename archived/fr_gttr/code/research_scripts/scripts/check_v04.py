#!/usr/bin/env python3
"""v0.4 GPU 前静态完整性检查。"""
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "MANUAL_REVIEW_V04.md",
        root / "RUN_ORDER_V04.md",
        root / "research/configs/stage_c_v04.json",
        root / "research/scripts/prepare_mid_checkpoint_v04.py",
        root / "research/scripts/e1_formal_v04.py",
        root / "research/scripts/e2_preflight_v04.py",
        root / "research/scripts/e2_epsilon_linearity_v04.py",
        root / "research/src/fr/pilot_partition.py",
        root / "research/src/fr/eval_observables.py",
        root / "research/src/fr/response_stats.py",
        root / "research/src/fr/stage_c_runtime.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("V04_CHECK_FAIL")
        for p in missing:
            print("缺少:", p)
        return 2
    print("V04_CHECK_PASS")
    print("说明：静态文件完整，不代表 E1/E2 科学结论。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
