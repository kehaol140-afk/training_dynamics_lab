#!/usr/bin/env python3
"""A0J-P1 replay utility: dual-label 5-action replay on aged dev units.

Runs ONLY after decisions are frozen (decisions_frozen.json + .sha256, written
by a0j_p1_cache_study_v01 BEFORE this script) and only if the implementation
gate passed. Replays j=0..4 once per aged unit (ages {1,2,4}) with the action
injected at the pending stream horizon (A0G frozen semantics), dual V_score /
V_eval endpoint labels in the same pass. Computes Gain_eval / Regret per
selector arm and DeltaGain(arm - exact) per age. V_eval never alters choices;
no mismatches dropped; no-op and harmful actions included.

This is dev approximation utility retention evidence, NOT independent
validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0j_cached_future_sensitivity_v01"
AGES = (1, 2, 4)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    args = ap.parse_args()

    status = json.loads((OUT / "p1_status.json").read_text())
    if status.get("CACHE_IMPLEMENTATION_GATE") != "PASS":
        print(json.dumps({"skipped": "implementation gate not PASS", "status": status}))
        return 2
    frozen_path = OUT / "decisions_frozen.json"
    frozen_sha = _sha256_file(frozen_path)
    recorded = (OUT / "decisions_frozen.sha256").read_text().split()[0]
    if frozen_sha != recorded:
        raise SystemExit(f"decisions hash mismatch: {frozen_sha} != {recorded}")
    decisions = json.loads(frozen_path.read_text())["decisions"]

    import a0h15_production_entry_v01 as pe  # noqa: E402
    from fr.a0j_future_sensitivity_cache_v01 import replay_five_actions  # noqa: E402

    session_t0 = time.perf_counter()
    budget = []
    rt = pe.build_runtime(args.device)
    budget.append({"phase": "preload_runtime_replay", "seconds": time.perf_counter() - session_t0,
                   "kind": "NON_METERED_PRELOAD_GPU_HELD"})

    label_rows, util_rows = [], []
    for unit, d in decisions.items():
        key = d["anchor"]
        ctx = next(c for c in rt["ctxs"] if f"{c.seed}/{c.step}" == key)
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == int(d["stream_id"]))
        cand = pe._candidates(ctx, stream, args.M)
        for age in AGES:
            t0 = time.perf_counter()
            labels = replay_five_actions(pe, rt, ctx, stream, age=age, H=args.H, cand=cand)
            budget.append({"phase": f"replay_{unit}_a{age}", "seconds": time.perf_counter() - t0,
                           "kind": "METERED"})
            for j, lab in sorted(labels.items()):
                label_rows.append({
                    "unit": unit, "age": age, "action_j": j,
                    "v_score": lab["v_score"], "v_eval": lab["v_eval"],
                    "endpoint_model_hash": lab["endpoint_model_hash"]})
            chosen = d["ages"][str(age)]["chosen"]
            v0 = labels[0]["v_eval"]
            best_j = min(labels, key=lambda j: labels[j]["v_eval"])
            row = {"unit": unit, "age": age,
                   "chosen_exact": chosen["exact"], "chosen_cache": chosen["cache"],
                   "chosen_stale": chosen["stale"], "oracle_j_eval": best_j,
                   "v_eval_noop": v0}
            for arm in ("exact", "cache", "stale"):
                cj = int(chosen[arm])
                row[f"gain_eval_{arm}"] = v0 - labels[cj]["v_eval"]
                row[f"regret_eval_{arm}"] = labels[cj]["v_eval"] - labels[best_j]["v_eval"]
            row["dGain_cache_minus_exact"] = row["gain_eval_cache"] - row["gain_eval_exact"]
            row["dGain_stale_minus_exact"] = row["gain_eval_stale"] - row["gain_eval_exact"]
            util_rows.append(row)

    with open(OUT / "five_action_replay_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_rows[0]))
        w.writeheader()
        for r in label_rows:
            w.writerow(r)
    with open(OUT / "utility_retention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(util_rows[0]))
        w.writeheader()
        for r in util_rows:
            w.writerow(r)

    summary = {"tag": "A0J_UTILITY_RETENTION", "ages": {}, "n_units": len(decisions)}
    for age in AGES:
        sub = [r for r in util_rows if int(r["age"]) == int(age)]
        blk = {"n": len(sub)}
        for arm in ("exact", "cache", "stale"):
            blk[f"mean_gain_eval_{arm}"] = statistics.mean(r[f"gain_eval_{arm}"] for r in sub)
            blk[f"mean_regret_eval_{arm}"] = statistics.mean(r[f"regret_eval_{arm}"] for r in sub)
            blk[f"chosen_id_match_vs_exact_{arm}"] = sum(
                1 for r in sub if int(r[f"chosen_{arm}"]) == int(r["chosen_exact"]))
        blk["mean_dGain_cache_minus_exact"] = statistics.mean(
            r["dGain_cache_minus_exact"] for r in sub)
        blk["mean_dGain_stale_minus_exact"] = statistics.mean(
            r["dGain_stale_minus_exact"] for r in sub)
        blk["v_eval_units"] = "same frozen validation_split.json v_eval split as A0H20/A0I"
        summary["ages"][str(age)] = blk
    (OUT / "utility_retention_summary.json").write_text(json.dumps(summary, indent=1))

    total_s = time.perf_counter() - session_t0
    budget.append({"phase": "TOTAL_SESSION", "seconds": total_s, "kind": "GPU_HELD_WALL"})
    log_path = OUT / "gpu_budget_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"entries": []}
    entries = old.get("entries", []) + budget
    log_path.write_text(json.dumps({
        "tag": "A0J_GPU_BUDGET_LOG", "budget_seconds": 1800, "entries": entries,
        "cumulative_s": sum(e["seconds"] for e in entries if e["phase"] != "TOTAL_SESSION")},
        indent=1))
    print(json.dumps({"replay_branches": len(label_rows),
                      "decisions_sha256_verified": frozen_sha,
                      "total_session_s": total_s}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
