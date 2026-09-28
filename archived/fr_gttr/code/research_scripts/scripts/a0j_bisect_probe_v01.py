#!/usr/bin/env python3
"""A0J bisect: module aged_decision_pass vs probe-verbatim pipeline, staged.

Stage 1: does the module's v_H match the verbatim rollout's v_H?
Stage 2: does the module's w_exact match the verbatim walk's w_0?
Stage 3: does the module's exact scores match production scores?
Runs the module pass with want_stage_times True and False separately.
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

import a0g_decision_utility_v01 as a0g  # noqa: E402
import a0h15_production_entry_v01 as pe  # noqa: E402
from fr.a0j_future_sensitivity_cache_v01 import aged_decision_pass  # noqa: E402

import importlib
mp = importlib.import_module("a0j_mirror_probe_v01")
hmap = mp.hmap


def main() -> int:
    session_t0 = time.perf_counter()
    rt = pe.build_runtime("cuda:0")
    torch = rt["torch"]
    preload_s = time.perf_counter() - session_t0
    ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == "2031/52")
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)

    t0 = time.perf_counter()
    snap_p, a_p, vH_p, _ = mp.rollout_with_hashes(pe, rt, ctx, stream, 4, 4)
    w_p, _ = mp.walk_production(pe, rt, snap_p, vH_p, 4)
    scores_p = {0: 0.0} | {int(j): float(__import__(
        "fr.a0c_adamw_response_propagation_v01", fromlist=["project_tangent"]
    ).project_tangent(w_p, a_p[j])) for j in a_p}

    m_t = aged_decision_pass(pe, rt, ctx, stream, age=0, H=4, M=4, want_stage_times=True)
    m_f = aged_decision_pass(pe, rt, ctx, stream, age=0, H=4, M=4, want_stage_times=False)
    prod = pe.run_adjoint_only(rt, ctx, stream, 4, 4, True, pe._new_counters())
    scores_prod = {0: 0.0} | {int(j): float(v) for j, v in prod["scores_hhat_adj"].items()}

    def sd(a, b):
        return max(abs(float(a[k]) - float(b[k])) for k in set(a) | set(b))

    result = {
        "tag": "A0J_MODULE_VS_VERBATIM_BISECT",
        "unit": "2031/52/s0",
        "stage1_vH_hash_match": {
            "verbatim": hmap(vH_p), "module_timed": hmap(m_t["v_H"]),
            "module_untimed": hmap(m_f["v_H"]),
            "timed_matches_verbatim": hmap(vH_p) == hmap(m_t["v_H"]),
            "untimed_matches_verbatim": hmap(vH_p) == hmap(m_f["v_H"]),
        },
        "stage2_w_hash_match": {
            "verbatim": hmap(w_p), "module_timed": hmap(m_t["w_exact"]),
            "module_untimed": hmap(m_f["w_exact"]),
            "timed_matches_verbatim": hmap(w_p) == hmap(m_t["w_exact"]),
            "untimed_matches_verbatim": hmap(w_p) == hmap(m_f["w_exact"]),
        },
        "stage3_scores": {
            "verbatim_vs_prod_maxdiff": sd(scores_p, scores_prod),
            "module_timed_vs_prod_maxdiff": sd(m_t["scores"]["exact"], scores_prod),
            "module_untimed_vs_prod_maxdiff": sd(m_f["scores"]["exact"], scores_prod),
            "module_timed_vs_verbatim_maxdiff": sd(m_t["scores"]["exact"], scores_p),
            "module_untimed_vs_verbatim_maxdiff": sd(m_f["scores"]["exact"], scores_p),
            "module_timed_vs_untimed_maxdiff": sd(m_t["scores"]["exact"], m_f["scores"]["exact"]),
        },
        "preload_s": preload_s, "probe_s": time.perf_counter() - t0,
    }
    (OUT / "module_vs_verbatim_bisect.json").write_text(json.dumps(result, indent=1))
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + [
        {"phase": "preload_bisect_probe", "seconds": preload_s,
         "kind": "NON_METERED_PRELOAD_GPU_HELD"},
        {"phase": "module_vs_verbatim_bisect", "seconds": result["probe_s"],
         "kind": "METERED"}]
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries
                            if e["phase"] != "TOTAL_SESSION")}, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
