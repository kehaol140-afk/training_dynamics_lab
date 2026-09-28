#!/usr/bin/env python3
"""FR B0.6 offline CRN gate: recorder psi CSVs vs frozen B0 diag (bitwise).

For ALL 26 trajectories (main 6 cells x 4 streams + stress 2 cells) x 64
steps, every shared field recorded by b06_record_psi_v01 must be BITWISE equal
to the frozen B0 native rows (main_diag.csv / stress_diag.csv). Exact float
equality after CSV round-trip (repr round-trip is exact); clip_active compared
as bool strings; empty update_cos_prev at h=1 compared as None/"".

Fail => exit 2 and stop per protocol (no analysis on ungated data).
Pass => writes runs/gate.json; analysis may proceed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b06_regime_dynamics_v01"
RUNS = OUT / "runs"
FROZEN_B0 = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01/runs"

SHARED_FIELDS = ["grad_norm", "update_norm", "update_cos_prev", "train_loss",
                 "preclip_norm", "lr_native", "lr_applied", "multiplier",
                 "clip_active"]
T_STEPS = 64
EXPECTED_TRAJECTORIES = {"main": 24, "stress": 2}


def load_psi(path):
    by_traj = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), int(row["step"]), int(row["stream_seed"]))
            by_traj.setdefault(key, {})[int(row["h"])] = row
    return by_traj


def load_frozen(path):
    by_traj = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["arm"] != "native" or float(row["u"]) != 0.0:
                continue
            key = (int(row["seed"]), int(row["step"]), int(row["stream_seed"]))
            by_traj.setdefault(key, {})[int(row["h"])] = row
    return by_traj


def field_equal(k, rec_v, fro_v):
    if k == "clip_active":
        return str(rec_v) == fro_v
    if rec_v == "" or fro_v == "":
        return rec_v == fro_v
    return float(rec_v) == float(fro_v)


def gate_tag(tag, psi_path, frozen_path):
    psi = load_psi(psi_path)
    frozen = load_frozen(frozen_path)
    per_traj = {}
    n_bad = 0
    for key, rows in sorted(psi.items()):
        fro_rows = frozen.get(key, {})
        bad = []
        if set(fro_rows) != set(rows):
            bad.append(f"row set mismatch: recorder {sorted(set(rows))} vs "
                       f"frozen {sorted(set(fro_rows))}")
        for h, r in sorted(rows.items()):
            fr = fro_rows.get(h)
            if fr is None:
                continue
            for k in SHARED_FIELDS:
                if not field_equal(k, r[k], fr[k]):
                    bad.append(f"h{h}.{k}: recorder={r[k]} frozen={fr[k]}")
        per_traj[{"main": "m", "stress": "s"}[tag] + f"{key}"] = {
            "n_steps": len(rows), "n_mismatch": len(bad),
            "mismatches": bad[:5]}
        n_bad += len(bad)
    return {"n_trajectories": len(psi), "expected": EXPECTED_TRAJECTORIES[tag],
            "n_mismatch_total": n_bad, "per_trajectory": per_traj,
            "PASS": len(psi) == EXPECTED_TRAJECTORIES[tag] and n_bad == 0}


def main() -> int:
    result = {"main": gate_tag("main", RUNS / "psi_main.csv",
                               FROZEN_B0 / "main_diag.csv"),
              "stress": gate_tag("stress", RUNS / "psi_stress.csv",
                                 FROZEN_B0 / "stress_diag.csv")}
    result["ALL_PASS"] = bool(result["main"]["PASS"] and result["stress"]["PASS"])
    (RUNS / "gate.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_trajectory"}
                          if isinstance(v, dict) else v)
                      for k, v in result.items()}, indent=1), flush=True)
    if not result["ALL_PASS"]:
        print("GATE FAILED — recorder trajectories not bitwise equal to frozen "
              "B0 diag; stop per protocol.", flush=True)
        return 2
    print("GATE PASS — all 26 trajectories x 64 steps bitwise equal to frozen "
          "B0 diag native rows.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
