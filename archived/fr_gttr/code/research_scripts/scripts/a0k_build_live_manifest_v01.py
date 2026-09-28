#!/usr/bin/env python3
"""A0K live manifest builder (OFFLINE, no GPU).

Freezes, per (seed, control step): E2-recipe lookahead stream seed + batch
content hashes, and candidate batch content hashes (j=1..4). All identity
inputs hashed. Run once before any GPU work; output is frozen.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"

from fr.stage_c_runtime import batches_from_train_pool  # noqa: E402
from fr.stream_rng import derive_stream_seed  # noqa: E402
from a0g_decision_utility_v01 import (  # noqa: E402
    N_DATA_GROUPS, candidate_batches_for)

E2_STREAM_BASE_SEED = 403202610
K20 = [20, 40, 60, 80, 100, 120]
K10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
SEEDS = [2030, 2031, 2032, 2033, 2034, 2035]


def two_level_seed(base, train_seed, state_step, stream_id):
    x = derive_stream_seed(int(base), int(train_seed))
    x = derive_stream_seed(x, int(state_step))
    return derive_stream_seed(x, int(stream_id))


def h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = json.loads((PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json").read_text())
    v07 = PROJECT_ROOT / cfg["output_dir"]
    partition = json.loads((v07 / "partition_v07.json").read_text())
    dg_path = PROJECT_ROOT / "outputs/fr_decision_utility_v01/data_group_definition.json"
    dg = json.loads(dg_path.read_text())
    vs_path = PROJECT_ROOT / "outputs/fr_decision_utility_v01/validation_split.json"
    vs = json.loads(vs_path.read_text())
    train_indices = list(map(int, partition["train_indices"]))
    bs = int(cfg["batch_size"])
    accum = int(cfg["accumulation_steps"])

    events = {}
    for seed in SEEDS:
        for step in sorted(set(K20 + K10)):
            stream_seed = two_level_seed(E2_STREAM_BASE_SEED, seed, step, 0)
            batch_seed = derive_stream_seed(stream_seed, 404)
            lookahead = batches_from_train_pool(train_indices, bs, batch_seed, 8 * accum)
            cand = {}
            for g in range(N_DATA_GROUPS):
                batches = candidate_batches_for(stream_seed, g, dg["members"][str(g)], bs, accum)
                cand[str(g + 1)] = {"batches": batches, "sha256": h(batches)}
            key = f"{seed}/{step}"
            events[key] = {
                "stream_id": 0,
                "stream_seed": stream_seed,
                "batch_seed": batch_seed,
                "lookahead_batches": lookahead,
                "lookahead_sha256": h(lookahead),
                "candidate_batches": cand,
            }

    manifest = {
        "tag": "A0K_LIVE_MANIFEST",
        "frozen_before_any_gpu": True,
        "config_sha256": file_sha(PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json"),
        "partition_sha256": file_sha(v07 / "partition_v07.json"),
        "data_group_definition_sha256": file_sha(dg_path),
        "validation_split_sha256": file_sha(vs_path),
        "group_definition_definition_sha256": dg["definition_sha256"],
        "validation_identity": {
            "v_score_n": len(vs["v_score"]), "v_eval_n": len(vs["v_eval"]),
            "eval_batch_size": 8, "eval_points": list(range(10, 129, 10)),
        },
        "cadence": {"K20_control_steps": K20, "K10_control_steps": K10,
                    "T_total": 128},
        "seeds": SEEDS,
        "events": events,
        "overlap_note": ("lookahead/candidate batches are train-pool samples drawn by the "
                         "frozen E2/Philox recipes; the real trajectory sequentially consumes "
                         "train_indices via batches_from_train_pool(train_indices,4,seed+1704,256); "
                         "overlap between controller simulation batches and future real batches "
                         "follows the same pattern as all prior dev rounds and is permitted; "
                         "candidates never overlap probe/eval pools (data groups are subsets of "
                         "train_indices; validation split is a subset of eval_indices)"),
    }
    (OUT / "contract/live_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"events": len(events), "sha": h(events)[:16]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
