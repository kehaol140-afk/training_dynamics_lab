"""Run the full M0 validation suite: unit tests then gate scripts."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

TESTS = [
    "tests/test_routing.py",
    "tests/test_muon_reference.py",
    "tests/test_adamw_aux.py",
    "tests/test_eval_sum_count.py",
    "tests/test_observation_hooks.py",
    "tests/test_repro_and_checkpoint.py",
]

SCRIPTS = [
    "scripts/m0_gate1_freeze.py",
    "scripts/m0_gate2_routing.py",
    "scripts/m0_gate3_data_audit.py",
    "scripts/m0_gate3_gradient_probe.py",
    "scripts/m0_gate4_pipeline_audit.py",
]


def run(cmd):
    print(f"\n=== {cmd} ===")
    t0 = time.time()
    r = subprocess.run([PY, str(ROOT / cmd)], cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    tail = "\n".join(r.stdout.strip().splitlines()[-8:])
    print(f"[{dt:.1f}s] exit={r.returncode}")
    if tail:
        print(tail)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {cmd}")
    return r


def main():
    for t in TESTS + SCRIPTS:
        run(t)
    print("\nM0 SUITE: ALL PASSED")


if __name__ == "__main__":
    main()
