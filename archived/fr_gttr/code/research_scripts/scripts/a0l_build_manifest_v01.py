#!/usr/bin/env python3
"""A0L events manifest builder (OFFLINE, no GPU).

Freezes the 6 control events (3 seeds x steps {50,100}): E2-recipe lookahead
streams + candidate batches, reusing the exact frozen stream seeds already
locked in A0K's live_manifest (read-only reference), with content re-hashed
here so A0L is self-contained and independently verifiable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0l_response_mixture_feasibility_v01"
A0K_MANIFEST = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01/contract/live_manifest.json"

from a0g_decision_utility_v01 import N_DATA_GROUPS, candidate_batches_for  # noqa: E402

SEEDS = [2030, 2031, 2032]
STEPS = [50, 100]
PI0 = [1038 / 3936, 973 / 3936, 987 / 3936, 938 / 3936]


def h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = json.loads((PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json").read_text())
    v07 = PROJECT_ROOT / cfg["output_dir"]
    dg_path = PROJECT_ROOT / "outputs/fr_decision_utility_v01/data_group_definition.json"
    dg = json.loads(dg_path.read_text())
    vs_path = PROJECT_ROOT / "outputs/fr_decision_utility_v01/validation_split.json"
    a0k = json.loads(A0K_MANIFEST.read_text())

    events = {}
    mismatch = []
    for seed in SEEDS:
        for step in STEPS:
            src = a0k["events"][f"{seed}/{step}"]
            stream_seed = int(src["stream_seed"])
            cand = {}
            for g in range(N_DATA_GROUPS):
                batches = candidate_batches_for(stream_seed, g, dg["members"][str(g)],
                                                int(cfg["batch_size"]), int(cfg["accumulation_steps"]))
                if batches != src["candidate_batches"][str(g + 1)]["batches"]:
                    mismatch.append(f"{seed}/{step}/g{g}")
                cand[str(g + 1)] = {"batches": batches, "sha256": h(batches)}
            lookahead = src["lookahead_batches"]
            events[f"{seed}/{step}"] = {
                "stream_id": 0,
                "stream_seed": stream_seed,
                "lookahead_batches": lookahead,
                "lookahead_sha256": h(lookahead),
                "candidate_batches": cand,
                "source": "A0K live_manifest events (frozen, read-only); content re-hashed here",
            }
    if mismatch:
        raise SystemExit(f"candidate batches mismatch vs A0K manifest: {mismatch}")

    manifest = {
        "tag": "A0L_EVENTS_MANIFEST",
        "frozen_before_any_gpu": True,
        "K": 50,
        "control_steps": STEPS,
        "T_total": 128,
        "pi_0_empirical": PI0,
        "pi_0_note": "groups partition the 3936-sample train pool; pi_0 = original distribution",
        "alpha": 0.1,
        "gamma_form": "gamma_t = 1/(std_j(h)+1e-12); q=softmax(-gamma_t*h); guard std<1e-12 -> uniform",
        "config_sha256": file_sha(PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json"),
        "partition_sha256": file_sha(v07 / "partition_v07.json"),
        "data_group_definition_sha256": file_sha(dg_path),
        "data_group_definition_definition_sha256": dg["definition_sha256"],
        "validation_split_sha256": file_sha(vs_path),
        "a0k_live_manifest_sha256": file_sha(A0K_MANIFEST),
        "validation_identity": a0k["validation_identity"],
        "seeds": SEEDS,
        "events": events,
    }
    (OUT / "contract/a0l_events_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"events": len(events), "events_sha": h(events)[:16],
                      "pi0": [round(p, 6) for p in PI0]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
