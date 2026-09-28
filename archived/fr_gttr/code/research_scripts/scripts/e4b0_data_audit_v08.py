#!/usr/bin/env python3
"""E4B-0：只读审计 q-rotation 所需数据是否已存在。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fr.e4b_data_audit import build_audit, render_markdown


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--seeds", default="2030,2031,2032,2033,2034,2035,2036,2037")
    ap.add_argument("--states", default="46,52,58,64,70,76,82")
    ap.add_argument("--horizons", default="1,2,4,8,12,16,24,32")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    report = build_audit(root, parse_int_list(args.seeds), parse_int_list(args.states), parse_int_list(args.horizons))

    out = root / "outputs" / "e4b_dynamic_state" / "v08"
    out.mkdir(parents=True, exist_ok=True)  # [FR-IO] 只创建 E4B 独立输出目录。
    jpath = out / "e4b0_data_audit.json"
    mpath = out / "e4b0_data_audit.md"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mpath.write_text(render_markdown(report), encoding="utf-8")

    for k, v in report["decision"].items():
        print(f"{k} = {v}")
    print(f"JSON = {jpath}")
    print(f"MD = {mpath}")
    return 0 if report["decision"]["E4B0_READY"] == "YES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
