#!/usr/bin/env python3
"""A0H-2.0 corrected-acceptance pilot: exposure ledger + acceptance manifest.

Runs BEFORE any A'' GPU exposure. Selection is by identity/availability/
exposure ONLY (never by hhat/gain/loss). Everything written here is frozen;
later phases may only append status, never rewrite selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "outputs/e3_confirmatory/v07"
APRIME = ROOT / "outputs/a0h15_vjp_shared_v01/accept_aprime"
OUT = ROOT / "outputs/a0h_corrected_accept_v01"
OUT.mkdir(parents=True, exist_ok=True)

H = 4
LAM = 0.05
SEEDS = list(range(2030, 2038))
STEPS = [46, 52, 58, 64, 70, 76, 82]

# ---- exposure records (states; unit-level = state + stream) ----------------
D_STATES = [(2031, 52), (2034, 64), (2036, 76)]      # A0H-1 adjoint dev (diag stream0 viewed)
DEMOTED_A = [(2032, 58), (2035, 70), (2037, 46)]     # A0H-1.5 A batch dev-validation
APRIME_STATES = [(2030, 52), (2033, 64), (2037, 82)]  # A0H-1.5 A' acceptance
A0C_LOCKED = [(2030, 46), (2037, 46), (2037, 82)]    # A0C mechanism-only (historical)
DEV_REHEARSAL = (2031, 52)                            # this round's dev unit (also D)

EXCLUDED = {
    **{s: "A0H-1 diagnostics dev unit (D): scorer/adjoint implementation debugging on stream0 outputs; utility labels viewed" for s in D_STATES},
    **{s: "A0H-1.5 A batch (demoted per read-once rule): dev-validation decision comparison viewed" for s in DEMOTED_A},
    **{s: "A0H-1.5 A' acceptance unit: hhat/ranking/gain viewed incl. corrected re-scoring" for s in APRIME_STATES},
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank_states():
    states = []
    for seed in SEEDS:
        tm = json.loads((BANK / f"seed_{seed}/theta_mapping_raw.json").read_text())
        assert tm["train_seed"] == seed
        for rec in tm["states"]:
            stp = int(rec["step"])
            streams = {}
            for e2 in rec["e2_streams"]:
                streams[int(e2["stream_id"])] = {
                    "stream_seed": e2["stream_seed"],
                    "batch_manifest_sha256": e2["batch_manifest_sha256"],
                    "batch_manifest_path": e2["batch_manifest_path"],
                }
            states.append({
                "train_seed": seed, "step": stp,
                "checkpoint_path": str(BANK / f"seed_{seed}/states/state_step_{stp:03d}.pt"),
                "q_sha256": rec["q_sha256"],
                "q_path": rec["q_path"],
                "streams": streams,
            })
    return states


def select_batch(eligible, n_states=6, n_streams=2):
    """Round-robin across seeds over (seed, step, ckpt)-sorted states."""
    by_seed = {}
    for s in sorted(eligible, key=lambda r: (r["train_seed"], r["step"], r["checkpoint_path"])):
        by_seed.setdefault(s["train_seed"], []).append(s)
    order = []
    rnd = 0
    while len(order) < n_states:
        progressed = False
        for seed in sorted(by_seed):
            if len(by_seed[seed]) > rnd:
                order.append(by_seed[seed][rnd]); progressed = True
        if not progressed:
            break
        rnd += 1
    picked = order[:n_states]
    units = []
    for s in picked:
        for sid in sorted(s["streams"])[:n_streams]:
            units.append({
                "unit_id": f"{s['train_seed']}_{s['step']:03d}_s{sid}",
                "train_seed": s["train_seed"], "step": s["step"],
                "checkpoint_path": s["checkpoint_path"], "q_sha256": s["q_sha256"],
                "stream_id": sid,
                "stream_seed": s["streams"][sid]["stream_seed"],
                "batch_manifest_sha256": s["streams"][sid]["batch_manifest_sha256"],
                "H": H, "lambda": LAM,
            })
    return picked, units


def dev_fixed_action():
    """Freeze ONE fixed action from OLD A' replay V_score gains (pre-exposure).

    Rule: per-state mean Gain_score (single stream each), equal weight across
    the 3 A' states including no-op; argmax, tie -> lowest action_id.
    """
    gains = {}
    for seed, stp in APRIME_STATES:
        rp = json.loads((APRIME / f"replay_{seed}_{stp:03d}_s0.json").read_text())
        assert int(rp["H"]) == H and abs(float(rp["lambda"]) - LAM) < 1e-12
        for j, g in rp["gain_vs_noop"].items():
            gains.setdefault(int(j), []).append(float(g))
    per_action = {j: sum(v) / len(v) for j, v in sorted(gains.items())}
    best = min(per_action, key=lambda j: (-per_action[j], j))
    return per_action, best


def main():
    states = load_bank_states()
    assert len(states) == 56, len(states)

    eligible, excluded = [], []
    for s in states:
        key = (s["train_seed"], s["step"])
        if key in EXCLUDED:
            excluded.append({**{k: s[k] for k in ("train_seed", "step", "checkpoint_path", "q_sha256")},
                             "reason": EXCLUDED[key]})
        else:
            eligible.append(s)
    assert len(eligible) == 47 and len(excluded) == 9

    picked, units = select_batch(eligible)
    assert len(picked) == 6 and len(units) == 12
    assert len({u["train_seed"] for u in units}) == 6

    per_action, dev_fixed = dev_fixed_action()

    # checkpoint identity hashes for the 6 picked states only (64MB each)
    ckpt_hashes = {}
    for s in picked:
        ckpt_hashes[f"{s['train_seed']}_{s['step']:03d}"] = sha256_file(Path(s["checkpoint_path"]))

    contract_dir = OUT / "contract"
    contract_sha = sha256_file(contract_dir / "contract.json")

    srcs = [
        "research/src/fr/a0g_decision_utility_v01.py",
        "research/scripts/a0g_decision_utility_v01.py",
        "research/scripts/a0h15_vjp_diagnostics_v01.py",
        "research/scripts/a0h15_production_entry_v01.py",
        "research/scripts/a0h15_accept_aprime_v01.py",
        "research/scripts/a0h20_contract_v01.py",
        "research/scripts/a0h20_manifest_v01.py",
        "research/scripts/a0h20_corrected_accept_v01.py",
        "research/tests/test_a0h20_acceptance_v01.py",
        "outputs/fr_decision_utility_v01/validation_split.json",
        "outputs/e3_confirmatory/v07/PREREG_LOCK.json",
    ]
    src_hashes = {p: sha256_file(ROOT / p) for p in srcs}

    ledger = {
        "ledger_id": "A0H20_EXPOSURE_LEDGER_V01",
        "date": "2026-09-16",
        "frozen_before_A_double_prime_exposure": True,
        "state_key_definition": "train_seed + checkpoint identity (state_step_SSS.pt path + sha256) + true optimizer step",
        "unit_key_definition": ("state key + future stream identity (stream_id + stream_seed + batch_manifest_sha256) "
                                "+ H + split/candidate identity"),
        "unseen_definition": ("unit not used in A0H implementation/selector-rule debugging AND its task-utility "
                              "labels (gain/ranking) never viewed"),
        "global_historical_exposure": {
            "scope": "ALL 56 bank states",
            "source": "A0C/A0D/A0E2/A0F/A0F2 response-propagation (mechanism-only: chi/tangent model fits)",
            "disposition": "historical exposure, NOT auto-excluded (mechanism-only, no selector/utility viewing)",
        },
        "excluded_states": sorted(excluded, key=lambda r: (r["train_seed"], r["step"])),
        "historical_mechanism_only_states": [
            {"train_seed": s, "step": t, "note": "A0C locked mechanism state (H in {1,2,4,8} x 8 e2 streams); mechanism-only; eligible with record" if (s, t) not in EXCLUDED else "A0C locked mechanism state; already excluded on other grounds"}
            for s, t in A0C_LOCKED
        ],
        "dev_rehearsal_unit": {"train_seed": DEV_REHEARSAL[0], "step": DEV_REHEARSAL[1], "stream_id": 0,
                               "note": "dev-only; D state (already excluded); never enters acceptance analysis"},
        "unknown_exposure_policy": "exclude when unclear (宁缺勿补)",
        "counts": {"bank_states": 56, "excluded": 9, "eligible": 47, "selected_states": 6, "selected_units": 12,
                   "selected_seeds": 6},
    }

    manifest = {
        "manifest_id": "A0H20_ACCEPTANCE_MANIFEST_V01",
        "run_id": "A0H20_CORRECTED_ACCEPTANCE_V01",
        "date": "2026-09-16",
        "frozen_before_A_double_prime_exposure": True,
        "contract_sha256": contract_sha,
        "semantics": {
            "action_ids": [0, 1, 2, 3, 4],
            "H": H, "lambda": LAM,
            "selected_id": "min(action_ids, key=(predicted_delta_loss_j=lambda*hhat_j, action_id)); no-op predicted 0.0; exact tie lowest id",
        },
        "dev_fixed": {
            "action_id": dev_fixed,
            "frozen_before_exposure": True,
            "source": "old A' 3-state replay V_score gain_vs_noop (H=4, lambda=0.05, stream0)",
            "per_action_equal_weight_mean_gain": per_action,
            "rule": "equal-weight per-state mean incl. no-op; argmax; tie lowest action_id",
        },
        "units": units,
        "unit_fixed_order": [u["unit_id"] for u in units],
        "dev_unit": {"unit_id": "dev_2031_052_s0", "train_seed": 2031, "step": 52, "stream_id": 0, "H": H, "lambda": LAM},
        "checkpoint_identity_sha256": ckpt_hashes,
        "exclusion_count": 9,
        "source_hashes": src_hashes,
        "aggregation_rule": ("per-unit Gain/delta -> per-state mean over its 2 streams -> equal weight over 6 states; "
                             "ALL units incl. no-op/harmful; no post-hoc denoising/thresholds"),
        "bootstrap": {"design": "seed->state->stream hierarchical paired", "n": 10000, "rng_seed": 260916,
                      "interval": "95% percentile", "note": "exploratory only; not a p-value"},
        "commands": {
            "env": "conda env fr_minimind; PYTHONPATH=research/src; CUDA_VISIBLE_DEVICES=1",
            "dev_rehearsal": ("python research/scripts/a0h20_corrected_accept_v01.py --phase dev "
                              "--seed 2031 --step 52 --stream 0"),
            "batch": ("python research/scripts/a0h20_corrected_accept_v01.py --phase batch "
                      "--manifest outputs/a0h_corrected_accept_v01/acceptance_manifest.json"),
            "cpu_tests": "python -m unittest research.tests.test_a0h20_acceptance_v01 -v",
        },
        "fixed_execution_order": [
            "1. CPU unit tests (selector 7 + interface checks)",
            "2. dev rehearsal on dev_2031_052_s0 (full chain, timed, GPU budget log start)",
            "3. P1 scorers for all 12 units -> predictions/ + choices_pre_eval (locked, hashed; no-op explicit)",
            "4. P2 same-config FP32 numerical gate -> gate.json (any fail -> STOPPED_AT_NUMERICAL_GATE)",
            "5. P3 per-unit 5-action dual-label replay -> labels/ (score labels first, lock replay-score choice, then eval labels)",
            "6. P4 analysis reads eval only after all choices locked",
        ],
        "gpu_budget": {"cap_seconds": 1800, "scope": "dev rehearsal + all batch GPU work, summed",
                       "log": "outputs/a0h_corrected_accept_v01/gpu_budget_log.json",
                       "waiting_total_max_minutes": 10,
                       "no_external_kill": True},
        "failure_retry_rules": {
            "infra_interruption": "resume same fingerprint/manifest/fixed order; never overwrite successful records",
            "numeric_selection_label_change_informed_by_scores": "whole batch demoted; independent acceptance ends",
            "serialization_only_fix": "requires proof of no numeric/choice change else demote",
            "gate_failure": "NUMERICAL_GATE=FAIL -> COMPLETION_STATUS=STOPPED_AT_NUMERICAL_GATE, UTILITY_PILOT_RESULT=NOT_COVERED; keep failures; no passing-subset continuation; no re-batch",
            "reduced_batch": "3-5 eligible states -> register reduced batch BEFORE opening results; <3 states or <2 legal streams/state -> NO_ELIGIBLE_BATCH (ledger+manifest only)",
        },
        "eval_isolation": ("V_eval 5-action labels written separately; selectors/thresholds/sample selection never read eval; "
                           "analysis reads eval only after ALL choices locked; interface tests verify"),
        "implementation_notes": [
            {"date": "2026-09-16", "note": ("P1 scorers use production-entry forward_only/adjoint_only modes at H_dec=4 "
                                            "(round-2 selfcheck: bit-identical to frozen diag pass at same config). "
                                            "Multi-H diag pass exercised once in dev rehearsal; H=1/H=8 diffs are "
                                            "diagnostic-only per contract; main analysis is H=4-only. Rationale: 1800s "
                                            "GPU budget. No score read; no numeric/selection/label logic change.")},
            {"date": "2026-09-16", "note": ("dev phase budget logging added retroactively (serialization-only; "
                                            "wall reconstructed from preload-end to report mtime; no numeric/choice effect)")},
        ],
    }

    (OUT / "exposure_ledger.json").write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "acceptance_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("eligible:", len(eligible), "picked:", [(s['train_seed'], s['step']) for s in picked])
    print("units:", [u["unit_id"] for u in units])
    print("dev_fixed:", dev_fixed, per_action)
    print("contract_sha256:", contract_sha)


if __name__ == "__main__":
    main()
