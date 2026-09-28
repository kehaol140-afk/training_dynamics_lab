#!/usr/bin/env python3
"""v0.6 文件完整性检查。"""
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]
    required=[
        root/'CODEX_AUDIT_PROMPT_V06.md',
        root/'RUN_ORDER_V06.md',
        root/'research/configs/e3r_cross_seed_v06.json',
        root/'research/src/fr/cross_seed_stats.py',
        root/'research/scripts/scientific_code_preflight_v06.py',
        root/'research/scripts/e3r_prepare_multiseed_state_bank_v06.py',
        root/'research/scripts/e3r_multiseed_theta_v06.py',
        root/'research/scripts/e3r_analyze_v06.py',
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print('V06_CHECK_FAIL')
        for p in missing: print('缺少:',p)
        return 2
    print('V06_CHECK_PASS')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
