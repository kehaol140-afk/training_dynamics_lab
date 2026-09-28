#!/usr/bin/env python3
"""A0J Kr/Ks cost projection (OFFLINE arithmetic — COST_PROJECTION_ONLY).

Reads measured medians from cache_timing_summary.json / wallclock_gate_summary.json
and writes kr_ks_cost_projection.csv. Every row is labeled
COST_ONLY / UTILITY_NOT_VALIDATED: this is cost arithmetic, never a claim of
closed-loop utility at any (K_r, K_s).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"
if str(PROJECT_ROOT / "research" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr.a0j_future_sensitivity_cache_v01 import kr_ks_projection  # noqa: E402


def main() -> int:
    timing = json.loads((OUT / "cache_timing_summary.json").read_text())
    inter = timing["interleaved_block"]
    T_refresh = None
    # one-shot refresh cost from per-unit passes (median over units/ages)
    vals = [r["T_refresh_s"] for r in timing["per_unit_from_passes"] if r.get("T_refresh_s")]
    if vals:
        vals = sorted(vals)
        T_refresh = vals[len(vals) // 2]
    rows = kr_ks_projection(
        T_refresh=float(T_refresh if T_refresh else inter["T_GTp_cached_median_s"]),
        T_score=float(inter["T_candidate_score_median_s"]),
        T_execute=float(inter["T_selected_action_execution_median_s"]),
        T_GTp=float(inter["T_GTp_cached_median_s"]),
        T_native=float(inter["T_native_step_median_s"]),
        utility_ages_supported=(1, 2, 4))
    with open(OUT / "kr_ks_cost_projection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (OUT / "kr_ks_cost_projection.meta.json").write_text(json.dumps({
        "tag": "A0J_KR_KS_COST_PROJECTION",
        "inputs": {"T_refresh_median_s": T_refresh,
                   "T_candidate_score_median_s": inter["T_candidate_score_median_s"],
                   "T_selected_action_execution_median_s": inter["T_selected_action_execution_median_s"],
                   "T_GTp_cached_median_s": inter["T_GTp_cached_median_s"],
                   "T_native_step_median_s": inter["T_native_step_median_s"]},
        "shared_gpu_contention_note": ("absolute seconds are contention-inflated on the shared "
                                       "GPU1; native-step-equivalent ratios are the comparable "
                                       "quantity"),
        "claim": "COST_ONLY / UTILITY_NOT_VALIDATED"}, indent=1))
    print(json.dumps({"rows": len(rows), "claim": "COST_ONLY / UTILITY_NOT_VALIDATED"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
