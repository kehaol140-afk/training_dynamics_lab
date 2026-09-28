#!/usr/bin/env python3
"""v0.5 GPU 前静态检查。"""
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    required = [
        root/'MANUAL_REVIEW_V05.md', root/'RUN_ORDER_V05.md',
        root/'research/configs/e3_theta_v05.json',
        root/'research/scripts/e3_prepare_state_bank_v05.py',
        root/'research/scripts/e3_theta_mapping_v05.py',
        root/'research/src/fr/native_schedule.py',
        root/'research/src/fr/mapping_stats.py',
        root/'research/src/fr/theta_protocol.py',
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print('V05_CHECK_FAIL')
        for p in missing: print('缺少:', p)
        return 2
    print('V05_CHECK_PASS')
    print('说明：这里只证明 v0.5 文件完整，不代表 E3 映射成立。')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
