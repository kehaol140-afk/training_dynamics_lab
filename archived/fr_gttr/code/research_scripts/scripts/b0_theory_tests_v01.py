#!/usr/bin/env python3
"""FR B0 theory tests (CPU, FP64) -> theory_tests.json.

Runs the B1-B4 battery from fr.b0_response_mechanisms_v01 and writes real
results. Exit code 0 iff ALL_OK. No GPU, no network.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr.b0_response_mechanisms_v01 import run_all_theory_tests  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    res = run_all_theory_tests()
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    res["device"] = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "theory_tests.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ALL_OK"] else 1


if __name__ == "__main__":
    sys.exit(main())
