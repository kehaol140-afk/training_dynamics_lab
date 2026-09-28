#!/usr/bin/env python3
"""Run the complete read-only A0C AdamW m+v response experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_mv_theory import file_sha256, render_report, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = root / "outputs/a0_fr_audit/mv_theory_final"
    summary, gradient_payload = run_experiment(root, device=args.device)

    # [FR-OBSERVATION-ONLY] Only the new audit directory is written.
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / "gradient_g.pt"
    gradient_tmp = output_dir / "gradient_g.pt.tmp"
    import torch

    torch.save(gradient_payload, gradient_tmp)
    gradient_tmp.replace(gradient_path)
    summary["gradient_artifact"] = {
        "path": str(gradient_path),
        "sha256": file_sha256(gradient_path),
        "state_count": gradient_payload["state_count"],
        "stream_count": gradient_payload["stream_count"],
        "aggregation": "STATE_LEVEL_ARITHMETIC_MEAN_OF_8_CAPTURED_E2_GRADIENTS",
    }

    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / "summary.json.tmp"
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)

    report_path = output_dir / "A0C_MV_THEORY_REPORT.md"
    report_tmp = output_dir / "A0C_MV_THEORY_REPORT.md.tmp"
    report_tmp.write_text(render_report(summary), encoding="utf-8")
    report_tmp.replace(report_path)

    print("STATUS = COMPLETE")
    print(f"SUMMARY = {summary_path}")
    print(f"GRADIENT = {gradient_path}")
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
