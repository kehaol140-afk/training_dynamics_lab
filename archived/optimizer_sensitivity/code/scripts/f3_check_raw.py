#!/usr/bin/env python3
"""Integrity check between F3 RQ stages (prereg §10: 完整性检查).

Usage: python scripts/f3_check_raw.py {rq1|rq2|rq3|rq4} SEED STEP

Checks the just-written raw/{rq}_{seed}_{step}.json:
- expected arm keys present at h=32 (rq1/rq2/rq3) or all offsets present
  (rq4);
- per-step primal losses bitwise-equal the current-environment frozen
  reference (outputs/f3_compatibility/frozen_reference/baseline_summary.json
  losses[cell]) -- the same-environment primal must be reproduced exactly;
- no unexpected arm stops at h=32 (rq1/rq2/rq3).

Exit 0 = PASS, 1 = FAIL (stop the driver; do not proceed to next RQ).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "f3_curvature_mechanism" / "raw"
FROZEN = ROOT / "outputs" / "f3_compatibility" / "frozen_reference" / "baseline_summary.json"

RQ1_ARMS = {"0.0", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0"}
RQ2_ARMS = {f"a{a}|{p}" for a in ("0.0", "1.0") for p in
            ("theta_only", "m_only", "v_only", "full_random", "attn_q_localized",
             "q0_lrsource")}
RQ3_ARMS = {"FULL", "NO_ATTN_Q", "NO_ATTN_K", "NO_ATTN_V", "NO_ATTN_O",
            "NO_MLP", "NO_EMBED", "NO_NORM", "ONLY_ATTN_Q", "ONLY_ATTN_K",
            "ONLY_MLP", "NO_MLP_UP", "NO_MLP_GATE", "NO_MLP_DOWN"}


def fail(msg):
    print(f"CHECK FAIL: {msg}")
    return 1


def main():
    rq, seed, step = sys.argv[1], sys.argv[2], sys.argv[3]
    cell = f"{seed}:{step}"
    raw = json.load((RAW / f"{rq}_{seed}_{step}.json").open())
    ref = json.load(FROZEN.open())
    ref_losses = ref["losses"][cell]

    if rq in ("rq1", "rq2", "rq3"):
        steps = raw.get("steps", [])
        if len(steps) != 32:
            return fail(f"{rq} {cell}: {len(steps)} steps != 32")
        losses = [r["loss"] for r in steps]
        if losses != list(ref_losses):
            for h, (a, b) in enumerate(zip(losses, ref_losses), 1):
                if a != b:
                    return fail(f"{rq} {cell}: loss diverges from frozen ref at h={h} "
                                f"({a!r} vs {b!r})")
            return fail(f"{rq} {cell}: loss length mismatch vs frozen ref")
        want = {"rq1": RQ1_ARMS, "rq2": RQ2_ARMS,
                "rq3": RQ3_ARMS | {f"SO|{a}" for a in RQ3_ARMS}}[rq]
        last = steps[-1]
        missing = want - set(last["arms"])
        if missing:
            return fail(f"{rq} {cell}: arms missing at h=32: {sorted(missing)}")
        stopped = [k for k, v in last["arms"].items() if v.get("stopped")]
        if stopped:
            return fail(f"{rq} {cell}: arms stopped by h=32: {stopped}")
    elif rq == "rq4":
        offs = raw.get("offsets", {})
        for off in ("0", "1", "3", "7", "15"):
            rec = offs.get(off)
            if rec is None:
                return fail(f"rq4 {cell}: offset {off} missing")
            for al in ("alpha0.0", "alpha1.0"):
                if al not in rec:
                    return fail(f"rq4 {cell}: offset {off} {al} missing")
                d = rec[al]
                for field in ("sigma_max", "pi_iters", "rho_reliable",
                              "rho_residual", "arnoldi_m", "nonnormal_proxy"):
                    if field not in d:
                        return fail(f"rq4 {cell}: offset {off} {al} field {field} missing")
    else:
        return fail(f"unknown rq {rq}")
    print(f"CHECK PASS: {rq} {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
