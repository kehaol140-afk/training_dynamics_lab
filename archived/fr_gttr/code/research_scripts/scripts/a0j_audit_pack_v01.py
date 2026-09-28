#!/usr/bin/env python3
"""A0J audit package builder (follows a0i_audit_pack_v01 pattern).

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
OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"

INCLUDE = [
    "contract.json",
    "cache_dev_manifest.json",
    "ALGORITHM_HEALTH_DASHBOARD.json",
    "REPORT.md",
    "commands.txt",
    "method_fingerprints.json",
    "p1_status.json",
    "decisions_frozen.json", "decisions_frozen.sha256",
    "prefix_components.csv", "wallclock_raw.csv", "wallclock_summary.json",
    "wallclock_per_repeat.jsonl", "wallclock_gate_summary.json",
    "profiler_summary.json",
    "action_vjp_gate.csv",
    "cached_scores.csv", "gap_diagnostics.csv", "decision_preservation.csv",
    "cache_vector_diagnostics.csv",
    "cache_timing_raw.csv", "cache_timing_summary.json", "cache_memory.json",
    "five_action_replay_labels.csv", "utility_retention.csv",
    "utility_retention_summary.json",
    "kr_ks_cost_projection.csv", "kr_ks_cost_projection.meta.json",
    "gpu_budget_log.json",
    "gate_failure_diagnosis_probe.json", "mirror_divergence_localization.json",
    "module_vs_verbatim_bisect.json", "per_call_trace.json",
    "logs_p0_stdout.txt", "logs_gate_repeats_stdout.txt",
    "logs_p1_stdout.txt", "logs_replay_stdout.txt",
    "research/scripts/a0j_p0_profiler_v01.py",
    "research/scripts/a0j_p0_gate_repeats_v01.py",
    "research/scripts/a0j_p1_cache_study_v01.py",
    "research/scripts/a0j_p1_replay_utility_v01.py",
    "research/scripts/a0j_kr_ks_cost_v01.py",
    "research/scripts/a0j_analyze_v01.py",
    "research/scripts/a0j_gate_diagnosis_probe_v01.py",
    "research/scripts/a0j_mirror_probe_v01.py",
    "research/scripts/a0j_bisect_probe_v01.py",
    "research/scripts/a0j_trace_probe_v01.py",
    "research/src/fr/a0j_future_sensitivity_cache_v01.py",
    "research/tests/test_a0j_cache_v01.py",
]
GLOBS = [
    "profiler_trace/*.json",
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
        if not p.exists() and Path(rel).parts[0] == "research_tests":
            p = PROJECT_ROOT / "research/tests" / Path(rel).name
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"self-verify FAILED: {bad}")
    print(f"self-verify OK ({len(manifest_lines)} files)")

    tar_path = OUT / "a0j_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files + [OUT / "manifest_sha256.txt"]:
            try:
                arc = f.relative_to(OUT)
            except ValueError:
                arc = arc_for(f)
            tar.add(f, arcname=str(arc))
    tar_sha = sha256_file(tar_path)
    (OUT / "a0j_audit.tar.gz.sha256").write_text(
        f"{tar_sha}  a0j_audit.tar.gz\n")

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
