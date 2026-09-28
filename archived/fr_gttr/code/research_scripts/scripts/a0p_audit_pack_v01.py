#!/usr/bin/env python3
"""A0P audit pack: manifest of hashes + tar + sha256 + self-verify (a0o pattern)."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0p_lr_temporal_kernel_v01"
AUDIT = OUT / "audit"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    code_files = [
        "research/src/fr/a0p_lr_temporal_kernel_v01.py",
        "research/scripts/a0p_run_v01.py",
        "research/scripts/a0p_analyze_v01.py",
        "research/scripts/a0p_audit_pack_v01.py",
    ]
    out_files = [
        "contract/contract.json",
        "contract/live_manifest.json",
        "runs/stage1_cells.jsonl",
        "runs/stage1_gate.json",
        "runs/stage2_kernels.jsonl",
        "runs/stage2_q.json",
        "analysis/stage2_q_offline.json",
        "reports/A0P_REPORT.md",
        "hashes/method_fingerprints.json",
        "gpu_budget_log.json",
        "driver.log",
    ]
    entries = []
    for rel in code_files + out_files:
        src = (PROJECT_ROOT / rel) if rel.startswith("research/") else (OUT / rel)
        if not src.exists():
            print(f"WARN missing: {rel}")
            continue
        if src.suffix in (".pt", ".pth", ".bin"):
            raise SystemExit(f"REFUSE to pack weight file: {rel}")
        entries.append((rel, src))

    manifest_lines = []
    tar_path = AUDIT / "a0p_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel, src in entries:
            data = src.read_bytes()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
            manifest_lines.append(f"{sha_bytes(data)}  {rel}")
    (AUDIT / "manifest_sha256.txt").write_text("\n".join(manifest_lines) + "\n")
    tar_sha = sha_file(tar_path)
    (AUDIT / "a0p_audit.tar.gz.sha256").write_text(f"{tar_sha}  a0p_audit.tar.gz\n")

    bad = []
    for line in manifest_lines:
        h, rel = line.split("  ", 1)
        src = (PROJECT_ROOT / rel) if rel.startswith("research/") else (OUT / rel)
        if not src.exists() or sha_file(src) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    info = {"tag": "A0P_AUDIT", "n_files": len(entries),
            "tar_bytes": tar_path.stat().st_size, "tar_sha256": tar_sha,
            "verify": "all manifest hashes match disk"}
    (AUDIT / "audit_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
