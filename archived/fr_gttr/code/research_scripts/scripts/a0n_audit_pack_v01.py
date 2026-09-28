#!/usr/bin/env python3
"""A0N audit pack: manifest of hashes + tar + sha256 + self-verify."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0n_phase_scheduler_pilot_v01"
AUDIT = OUT / "audit"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    code_files = [
        "research/src/fr/a0n_phase_scheduler_runner_v01.py",
        "research/scripts/a0n_run_v01.py",
        "research/scripts/a0n_analyze_v01.py",
        "research/scripts/a0n_audit_pack_v01.py",
        "research/src/fr/a0k_closed_loop_runner_v01.py",
    ]
    out_files = [
        "contract/contract.json",
        "runs/all_arm_summaries.json",
        "runs/post_main_consistency.json",
        "dashboard/dashboard.csv",
        "dashboard/dashboard.json",
        "analysis/rho_trajectory.csv",
        "reports/A0N_REPORT.md",
        "hashes/method_fingerprints.json",
        "gpu_budget_log.json",
    ]
    run_files = sorted(
        str(p.relative_to(OUT))
        for pat in ("runs/seed_*/*/summary.json", "runs/seed_*/timing_meta.json",
                    "runs/seed_*/C/events.jsonl",
                    "runs/smoke_S1/result.json", "runs/smoke_S2/result.json")
        for p in OUT.glob(pat)
    )
    entries = []
    for rel in code_files + out_files + run_files:
        src = (PROJECT_ROOT / rel) if rel.startswith("research/") else (OUT / rel)
        if not src.exists():
            print(f"WARN missing: {rel}")
            continue
        if src.suffix in (".pt", ".pth", ".bin"):
            raise SystemExit(f"REFUSE to pack weight file: {rel}")
        entries.append((rel, src))

    manifest_lines = []
    tar_path = AUDIT / "a0n_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel, src in entries:
            data = src.read_bytes()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
            manifest_lines.append(f"{sha_bytes(data)}  {rel}")
    (AUDIT / "manifest_sha256.txt").write_text("\n".join(manifest_lines) + "\n")
    tar_sha = sha_file(tar_path)
    (AUDIT / "a0n_audit.tar.gz.sha256").write_text(f"{tar_sha}  a0n_audit.tar.gz\n")

    bad = []
    for line in manifest_lines:
        h, rel = line.split("  ", 1)
        src = (PROJECT_ROOT / rel) if rel.startswith("research/") else (OUT / rel)
        if not src.exists() or sha_file(src) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    info = {"tag": "A0N_AUDIT", "n_files": len(entries),
            "tar_bytes": tar_path.stat().st_size, "tar_sha256": tar_sha,
            "verify": "all manifest hashes match disk"}
    (AUDIT / "audit_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
