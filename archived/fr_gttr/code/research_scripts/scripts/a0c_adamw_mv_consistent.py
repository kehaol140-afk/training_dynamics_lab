#!/usr/bin/env python3
"""Run the strict q/scale-consistent A0C AdamW m+v analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_mv_consistent import render_consistent_report, run_consistent_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output_dir = root / "outputs/a0_fr_audit/mv_theory_consistent"

    summary = run_consistent_experiment(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)

    report_path = output_dir / "A0C_MV_CONSISTENT_REPORT.md"
    report_tmp = output_dir / "A0C_MV_CONSISTENT_REPORT.md.tmp"
    report_tmp.write_text(render_consistent_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)

    print("STATUS = COMPLETE")
    print(f"SUMMARY = {summary_path}")
    print(f"REPORT = {report_path}")
    print(
        "R2 =",
        {
            "chi_m": summary["metrics"]["chi_m"]["overall"]["R2"],
            "chi_mv": summary["metrics"]["chi_mv"]["overall"]["R2"],
            "improvement": summary["comparison"]["R2_improvement"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
