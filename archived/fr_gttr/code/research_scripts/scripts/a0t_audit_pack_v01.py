#!/usr/bin/env python3
"""A0T audit pack: sha256 manifest of all artifacts + tar bundle."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0t_response_lr_control_pilot_v01"
HASHES = OUT / "hashes"
AUDIT = OUT / "audit"

CODE_FILES = [
    "research/src/fr/a0t_response_lr_control_pilot_v01.py",
    "research/scripts/a0t_run_v01.py",
    "research/scripts/a0t_analyze_v01.py",
    "research/scripts/a0t_audit_pack_v01.py",
]
OUT_FILES = [
    "contract/contract.json",
    "contract/live_manifest.json",
    "runs/cells.jsonl",
    "runs/summary.json",
    "runs/analysis.json",
    "gpu_budget_log.json",
    "driver.log",
    "reports/A0T_REPORT.md",
    "reports/a0t_response_lr_control_table.csv",
    "reports/a0t_detail_supplementary.csv",
    "reports/plots/a0t_gains_and_terciles.png",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    HASHES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel in CODE_FILES:
        entries.append({"path": rel, "root": "repo", "sha256": sha256(PROJECT_ROOT / rel)})
    for rel in OUT_FILES:
        entries.append({"path": rel, "root": "output", "sha256": sha256(OUT / rel)})
    manifest = {"audit_id": "A0T_RESPONSE_LR_CONTROL_PILOT_V01_AUDIT",
                "date": "2026-09-18", "files": entries}
    (HASHES / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (HASHES / "SHA256SUMS.txt").write_text(
        "\n".join(f"{e['sha256']}  {e['path']}" for e in entries) + "\n")
    tar = AUDIT / "a0t_audit_pack_v01.tar.gz"
    if tar.exists():
        tar.unlink()
    paths = [PROJECT_ROOT / f for f in CODE_FILES] + [OUT / f for f in OUT_FILES]
    paths += [HASHES / "manifest.json", HASHES / "SHA256SUMS.txt"]
    subprocess.run(["tar", "-czf", str(tar), "-C", str(PROJECT_ROOT)]
                   + [str(p.relative_to(PROJECT_ROOT)) for p in paths], check=True)
    tar_sha = sha256(tar)
    (AUDIT / "TAR_SHA256.txt").write_text(tar_sha + "\n")
    print(json.dumps({"n_files": len(entries), "tar_sha256": tar_sha}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
