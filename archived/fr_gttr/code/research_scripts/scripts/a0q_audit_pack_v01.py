#!/usr/bin/env python3
"""A0Q audit pack: manifest of hashes + tar + sha256 + self-verify (a0o pattern)."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0q_response_control_map_v01"
AUDIT = OUT / "audit"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    code_files = [
        "research/src/fr/a0q_response_control_map_v01.py",
        "research/scripts/a0q_run_v01.py",
        "research/scripts/a0q_analyze_v01.py",
        "research/scripts/a0q_audit_pack_v01.py",
    ]
    out_files = [
        "contract/contract.json",
        "contract/live_manifest.json",
        "runs/cells.jsonl",
        "runs/gates.json",
        "runs/evaluation.json",
        "runs/verdict.json",
        "analysis/q_offline.json",
        "reports/A0Q_REPORT.md",
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
    tar_path = AUDIT / "a0q_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel, src in entries:
            data = src.read_bytes()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
            manifest_lines.append(f"{sha_bytes(data)}  {rel}")
    (AUDIT / "manifest_sha256.txt").write_text("\n".join(manifest_lines) + "\n")
    tar_sha = sha_file(tar_path)
    (AUDIT / "a0q_audit.tar.gz.sha256").write_text(f"{tar_sha}  a0q_audit.tar.gz\n")

    bad = []
    for line in manifest_lines:
        h, rel = line.split("  ", 1)
        src = (PROJECT_ROOT / rel) if rel.startswith("research/") else (OUT / rel)
        if not src.exists() or sha_file(src) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    info = {"tag": "A0Q_AUDIT", "n_files": len(entries),
            "tar_bytes": tar_path.stat().st_size, "tar_sha256": tar_sha,
            "verify": "all manifest hashes match disk"}
    (AUDIT / "audit_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
