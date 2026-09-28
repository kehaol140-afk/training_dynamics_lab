#!/usr/bin/env python3
"""A0K audit package builder (follows a0j_audit_pack_v01 pattern).

Small text artifacts only (NO weights/gradients/checkpoints/.pt). Per-file
SHA256 manifest (excluding itself), self-verify, then tar + separate tar
SHA256; verifies listable/unpackable/file count. Read-only wrt inputs.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"

INCLUDE = [
    "contract/contract.json",
    "contract/live_manifest.json",
    "gpu_budget_log.json",
    "runs/phase_rows.jsonl",
    "runs/all_arm_summaries.json",
    "runs/smoke_S1/result.json",
    "runs/smoke_S2/result.json",
    "runs/smoke_S3/result.json",
    "runs/counterfactual_replay.csv",
    "dashboard/dashboard.csv",
    "dashboard/dashboard.json",
    "reports/A0K_REPORT.md",
    "hashes/method_fingerprints.json",
    "research/scripts/a0k_build_live_manifest_v01.py",
    "research/scripts/a0k_run_v01.py",
    "research/scripts/a0k_counterfactual_replay_v01.py",
    "research/scripts/a0k_rerun_bwc_v01.py",
    "research/scripts/a0k_analyze_v01.py",
    "research/src/fr/a0k_closed_loop_runner_v01.py",
]
GLOBS = [
    "runs/K*/seed_*/timing_meta.json",
    "runs/K*/seed_*/*/step_log.json",
    "runs/K*/seed_*/C/events.jsonl",
    "runs/K*/seed_*/C/events_replayed.jsonl",
    "runs/K*/seed_*/B/events.jsonl",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arc_for(f: Path) -> Path:
    parts = f.parts
    if "src" in parts and "fr" in parts:
        return Path("research_src_fr") / f.name
    if "tests" in parts:
        return Path("research_tests") / f.name
    return Path("research_scripts") / f.name


def collect_files():
    files = []
    for rel in INCLUDE:
        p = OUT / rel
        if p.exists():
            files.append(p)
        else:
            alt = PROJECT_ROOT / rel
            if alt.exists():
                files.append(alt)
            else:
                print(f"warn: missing {rel}")
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
        try:
            rel = f.relative_to(OUT)
        except ValueError:
            rel = arc_for(f)
        manifest_lines.append(f"{sha256_file(f)}  {rel}")
    (OUT / "manifest_sha256.txt").write_text("\n".join(manifest_lines) + "\n")

    bad = []
    for line in manifest_lines:
        h, rel = line.split("  ")
        p = OUT / rel
        if not p.exists():
            p = PROJECT_ROOT / rel
        if not p.exists() and Path(rel).parts[0] == "research_scripts":
            p = PROJECT_ROOT / "research/scripts" / Path(rel).name
        if not p.exists() and Path(rel).parts[0] == "research_src_fr":
            p = PROJECT_ROOT / "research/src/fr" / Path(rel).name
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    print(f"self-verify OK ({len(manifest_lines)} files)")

    tar_path = OUT / "a0k_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files + [OUT / "manifest_sha256.txt"]:
            try:
                arc = f.relative_to(OUT)
            except ValueError:
                arc = arc_for(f)
            tar.add(f, arcname=str(arc))
    tar_sha = sha256_file(tar_path)
    (OUT / "a0k_audit.tar.gz.sha256").write_text(f"{tar_sha}  a0k_audit.tar.gz\n")

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
            tar.extractall(td)
        unpacked = list(Path(td).rglob("*"))
        n_files = len([p for p in unpacked if p.is_file()])
    info = {"tar": str(tar_path), "tar_bytes": tar_path.stat().st_size,
            "tar_sha256": tar_sha, "entries": len(names),
            "files_unpacked": n_files, "manifest_files": len(manifest_lines),
            "verify": "listable+unpackable OK"}
    (OUT / "audit_package_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
