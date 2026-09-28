#!/usr/bin/env python3
"""E2 运行前静态/结果门禁。不占 GPU。"""
from pathlib import Path
import json, sys


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/stage_c_v04"
    required = [
        out / "mid_state_v04.pt",
        out / "partition_v04.json",
        out / "full_model_q_v04.pt",
        out / "e1_formal_v04_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("E2_PREFLIGHT_FAIL")
        for p in missing:
            print("缺少:", p)
        return 2

    report = json.loads((out / "e1_formal_v04_report.json").read_text(encoding="utf-8"))
    if not report.get("technical_pass"):
        print("E2_PREFLIGHT_FAIL")
        print("原因：E1 v0.4 technical_pass != True")
        return 3
    if int(report.get("R", 0)) != 16:
        print("E2_PREFLIGHT_FAIL")
        print("原因：正式 E1 R 不是 16")
        return 4
    if float(report.get("sigma_q", 0.0)) <= 0:
        print("E2_PREFLIGHT_FAIL")
        print("原因：sigma_q 无效")
        return 5

    print("E2_PREFLIGHT_PASS")
    print("sigma_q=", report["sigma_q"])
    print("epsilon_candidates=", report["epsilon_candidates"])
    print("说明：这里只允许进入 E2 技术运行，不代表局部线性已成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
