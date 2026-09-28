#!/usr/bin/env python3
"""Run the read-only FR A0C Momentum Theory v02 audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Make the repository script runnable without relying on a caller-set PYTHONPATH.
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_momentum_theory_v02 import build_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--output",
        help=(
            "Optional summary path. Defaults to "
            "outputs/a0_fr_audit/momentum_theory_v02_v07/summary.json under project root."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else project_root
        / "outputs/a0_fr_audit/momentum_theory_v02_v07/summary.json"
    )
    summary = build_summary(project_root)

    # [FR-OBSERVATION-ONLY] Only the dedicated v02 summary is written.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"STATUS = {summary['status']}")
    print(f"SUMMARY = {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
