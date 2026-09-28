#!/usr/bin/env python3
"""A0H-2.0 corrected-acceptance pilot: audit package builder.

Small source/config/CSV/JSON/docs only (NO weights/gradients/checkpoints).
Per-file SHA256 manifest (excluding itself), self-verify, then tar + separate
tar SHA256; verifies listable/unpackable/file count. Read-only wrt inputs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0h_corrected_accept_v01"

# small text artifacts only; explicitly exclude any .pt / tensors
INCLUDE = [
    "contract/contract.json", "contract/PROTOCOL_AMENDMENT_A0H20.md",
    "contract/contract.sha256", "contract/contract_crosscheck.json",
    "exposure_ledger.json", "acceptance_manifest.json",
    "gpu_budget_log.json",
    "dev_rehearsal/dev_rehearsal_report.json",
    "gate.json", "COMPLETION_STATUS.json",
    "UTILITY_PILOT_REPORT.md", "run_batch.log", "env_commands.json",
]
GLOBS = ["predictions/*.json", "labels/score/*.json", "labels/eval/*.json",
         "analysis/*.json"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_files():
    files = []
    for rel in INCLUDE:
        p = OUT / rel
        if p.exists():
            files.append(p)
    for g in GLOBS:
        files.extend(sorted(OUT.glob(g)))
    for f in files:
        if f.suffix in (".pt", ".pth", ".bin"):
            raise SystemExit(f"REFUSED: binary artifact in package set: {f}")
    return sorted(set(files))


def main():
    files = collect_files()
    print(f"packaging {len(files)} files")
    manifest_lines = []
    for f in files:
        rel = f.relative_to(OUT)
        manifest_lines.append(f"{sha256_file(f)}  {rel}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n")

    # self-verify
    bad = []
    for line in manifest_lines:
        h, rel = line.split("  ")
        if sha256_file(OUT / rel) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    print(f"self-verify OK ({len(manifest_lines)} files)")

    tar_path = OUT / "a0h20_corrected_accept_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files + [OUT / "SHA256SUMS.txt"]:
            tar.add(f, arcname=str(f.relative_to(OUT)))
    tar_sha = sha256_file(tar_path)
    (OUT / "a0h20_corrected_accept_audit.tar.gz.sha256").write_text(
        f"{tar_sha}  a0h20_corrected_accept_audit.tar.gz\n")

    # verify listable/unpackable/file count
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
            tar.extractall(td)
        unpacked = list(Path(td).rglob("*"))
        n_files = len([p for p in unpacked if p.is_file()])
    info = {"tar": str(tar_path), "tar_bytes": tar_path.stat().st_size,
            "tar_sha256": tar_sha, "entries": len(names), "files_unpacked": n_files,
            "manifest_files": len(manifest_lines), "verify": "listable+unpackable OK"}
    (OUT / "audit_package_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
