#!/usr/bin/env python3
"""A0I audit package builder (follows a0h20_audit_pack_v01 pattern).

Small text artifacts only (NO weights/gradients/checkpoints/.pt).
Per-file SHA256 manifest (excluding itself), self-verify, then tar +
separate tar SHA256; verifies listable/unpackable/file count. Read-only
wrt inputs.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0i_directional_scoring_v01"

INCLUDE = [
    "contract.json",
    "ALGORITHM_HEALTH_DASHBOARD.json",
    "REPORT.md",
    "a0i_a/a0i_a_summary.json", "a0i_a/calibration.json", "a0i_a/case_study.json",
    "a0i_b/environment.json", "a0i_b/gpu_budget_log.json",
    "a0i_b/method_fingerprints.json", "a0i_b/method_c_derivation.md",
    "a0i_b/equivalence_summary.json", "a0i_b/decision_preservation.csv",
    "a0i_b/decision_cost_summary.json", "a0i_b/memory_summary.json",
    "a0i_b/marginal_cost.json", "a0i_b/verdicts.json",
    "a0i_b/benchmark_commands.txt",
    "a0i_b/probe_2031_52.json", "a0i_b/preload_seconds.json",
    "research/scripts/a0i_a_offline_diagnostics_v01.py",
    "research/scripts/a0i_b_directional_scoring_v01.py",
    "research/scripts/a0i_b_analyze_v01.py",
]
GLOBS = [
    "a0i_a/*.csv",
    "a0i_b/timing_summary.json",
    "a0i_b/run_*/directional_scores.csv",
    "a0i_b/run_*/timing_raw.csv",
    "a0i_b/run_*/run_meta.json",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
            rel = Path("research_scripts") / f.name
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
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    print(f"self-verify OK ({len(manifest_lines)} files)")

    tar_path = OUT / "a0i_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files + [OUT / "manifest_sha256.txt"]:
            try:
                arc = f.relative_to(OUT)
            except ValueError:
                arc = Path("research_scripts") / f.name
            tar.add(f, arcname=str(arc))
    tar_sha = sha256_file(tar_path)
    (OUT / "a0i_audit.tar.gz.sha256").write_text(
        f"{tar_sha}  a0i_audit.tar.gz\n")

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
