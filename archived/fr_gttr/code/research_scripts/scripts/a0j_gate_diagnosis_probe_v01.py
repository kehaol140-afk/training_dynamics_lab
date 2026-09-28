#!/usr/bin/env python3
"""A0J gate-failure diagnosis probe (part of the FAILED-PROBE deliverable).

Determines whether the production_reproduction score mismatch is (a) a mirror
bug in the A0J walk or (b) run-to-run nondeterminism of the frozen production
path itself on this GPU (no torch determinism config; bf16 backward kernels).

Runs on ONE unit (2031/52 stream 0):
  1. production run_adjoint_only twice -> production-vs-production score diff
  2. A0J anchor pass twice -> mirror-vs-mirror score diff and bitwise w check
  3. production-vs-mirror score diff (the failing comparison, re-measured)

No semantic changes; no cache results are viewed to alter any rule.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"

import a0h15_production_entry_v01 as pe  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import (  # noqa: E402
    aged_decision_pass, max_abs_score_diff, score_actions,
)


def main() -> int:
    device = "cuda:0"
    session_t0 = time.perf_counter()
    rt = pe.build_runtime(device)
    preload_s = time.perf_counter() - session_t0
    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == "2031/52")
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)

    t0 = time.perf_counter()
    prod1 = pe.run_adjoint_only(rt, ctx, stream, 4, 4, True, pe._new_counters())
    prod2 = pe.run_adjoint_only(rt, ctx, stream, 4, 4, True, pe._new_counters())
    m1 = aged_decision_pass(pe, rt, ctx, stream, age=0, H=4, M=4, want_stage_times=False)
    m2 = aged_decision_pass(pe, rt, ctx, stream, age=0, H=4, M=4, want_stage_times=False)
    probe_s = time.perf_counter() - t0

    def h_prod(p):
        return {0: 0.0} | {int(j): float(v) for j, v in p["scores_hhat_adj"].items()}

    def h_mirr(m):
        return m["scores"]["exact"]

    def w_maxdiff(a, b):
        import torch
        worst = 0.0
        for name in a:
            va = a[name].detach().to(torch.float64)
            vb = b[name].detach().to(torch.float64)
            worst = max(worst, float(torch.max(torch.abs(va - vb))) if va.numel() else 0.0)
        return worst

    result = {
        "tag": "A0J_GATE_FAILURE_DIAGNOSIS_PROBE",
        "unit": "2031/52/s0",
        "prod_vs_prod_max_abs_score_diff": max_abs_score_diff(h_prod(prod1), h_prod(prod2)),
        "mirror_vs_mirror_max_abs_score_diff": max_abs_score_diff(h_mirr(m1), h_mirr(m2)),
        "mirror_vs_mirror_max_abs_w_diff": w_maxdiff(m1["w_exact"], m2["w_exact"]),
        "prod_vs_mirror_run1_max_abs_score_diff": max_abs_score_diff(h_prod(prod1), h_mirr(m1)),
        "prod_vs_mirror_run2_max_abs_score_diff": max_abs_score_diff(h_prod(prod2), h_mirr(m2)),
        "frozen_tolerance_reference": {"atol32": 1e-5, "rtol32": 1e-4},
        "no_torch_determinism_config": True,
        "h_scale": {"max_abs_h_prod": max(abs(v) for v in h_prod(prod1).values())},
        "preload_s": preload_s, "probe_s": probe_s,
    }
    (OUT / "gate_failure_diagnosis_probe.json").write_text(json.dumps(result, indent=1))
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + [
        {"phase": "preload_gate_diagnosis_probe", "seconds": preload_s,
         "kind": "NON_METERED_PRELOAD_GPU_HELD"},
        {"phase": "gate_failure_diagnosis_probe", "seconds": probe_s, "kind": "METERED"}]
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries
                            if e["phase"] != "TOTAL_SESSION")}, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
