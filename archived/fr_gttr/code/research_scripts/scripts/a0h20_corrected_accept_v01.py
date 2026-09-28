#!/usr/bin/env python3
"""A0H-2.0 corrected-acceptance pilot driver (A'' batch, run_id A0H20_CORRECTED_ACCEPTANCE_V01).

Executes the FROZEN contract in outputs/a0h_corrected_accept_v01/contract/contract.json
(written and hash-frozen BEFORE any A'' exposure) on the 12 pre-registered units in
acceptance_manifest.json (6 states x 2 streams, exposure-ledger selected).

Thin adapter: reuses the FIXED production implementation —
  * scorers   : a0h15_vjp_diagnostics_v01._run_diagnostics_pass (lean), two passes per
                unit (BF16 fwd C config; FP32-HVP fwd + shared adjoint config).
  * selector  : fr.a0g_decision_utility_v01.select_action (FROZEN A0G semantics,
                argmin predicted Delta L over j=0..4 with no-op, exact tie lowest j)
                via frozen_decision(); hhat = dL/dlambda, predicted = lambda*hhat.
  * replay    : entry._forward_backward/_clip_and_step discipline; one dual-label
                branch per action (j=0 baseline + j=1..4 single-step injection at
                h=1 post-unscale/pre-clip), V_score AND V_eval at H in the SAME
                trajectory; selectors never see eval.

Phase-gated leakage control (mandate §三):
  P1  scorers for ALL units -> predictions/ + choices_pre_eval.json (locked+hashed)
  P2  same-config FP32 numerical gate -> gate.json (any fail -> STOPPED_AT_NUMERICAL_GATE)
  P3  per-unit 5-action dual-label replay -> labels/ (score labels first, replay-score
      choice locked from score only, then eval labels to a SEPARATE module)
  P4  analysis (a0h20_analyze_v01.py) reads eval only after ALL choices are locked.

Resume: never overwrite existing successful per-unit records; a numeric/selection/
label-logic change informed by read scores demotes the whole batch per manifest rules.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs/a0h_corrected_accept_v01"
ATOL32, RTOL32 = 1e-5, 1e-4

_spec = importlib.util.spec_from_file_location(
    "a0h15_entry", PROJECT_ROOT / "research/scripts/a0h15_production_entry_v01.py")
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)

_spec2 = importlib.util.spec_from_file_location(
    "a0h15_accept_aprime_v01", PROJECT_ROOT / "research/scripts/a0h15_accept_aprime_v01.py")
aprime_driver = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(aprime_driver)

import a0g_decision_utility_v01 as a0g  # noqa: E402
import a0h15_vjp_diagnostics_v01 as diag  # noqa: E402

from fr.a0g_decision_utility_v01 import (  # noqa: E402  FROZEN selector, read-only
    select_action as frozen_select_action,
)
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402


# ---------------------------------------------------------------------------
# frozen decision semantics (single definition point for driver AND tests)
# ---------------------------------------------------------------------------
def frozen_decision(hhat_by_j: dict, lam: float) -> int:
    """FROZEN A0G semantics: hhat = dL/dlambda; predicted Delta L_j = lam*hhat_j;
    j=0 no-op predicted 0.0 IS in the candidate set; argmin, exact tie lowest j.
    NaN/Inf scores are NON_FINITE and are rejected before selection (contract
    finite-check; an input guard, not a ranking rule)."""
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"NON_FINITE score blocked from selection: j={j} value={v!r}")
        predicted[int(j)] = float(lam) * v
    return int(frozen_select_action(predicted))


def five_action_ranking(hhat_by_j: dict, lam: float) -> list[int]:
    """Ascending predicted Delta L over the FULL 5-action set (no-op included).
    Ties: lowest action_id first (exact frozen comparison)."""
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"NON_FINITE score blocked from ranking: j={j} value={v!r}")
        predicted[int(j)] = float(lam) * v
    return [int(j) for j in sorted(predicted, key=lambda j: (predicted[j], j))]


def prediction_margin(hhat_by_j: dict, lam: float) -> dict:
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        predicted[int(j)] = float(lam) * float(v)
    srt = sorted(predicted, key=lambda j: (predicted[j], j))
    return {"best_j": int(srt[0]), "best_pred_delta": predicted[srt[0]],
            "second_j": int(srt[1]), "second_pred_delta": predicted[srt[1]],
            "margin": predicted[srt[1]] - predicted[srt[0]]}


# ---------------------------------------------------------------------------
# P3: dual-label 5-action replay (trajectory identical to entry._replay_branch;
# endpoint evaluates V_score then V_eval in the SAME pass, separate files)
# ---------------------------------------------------------------------------
def _dual_label_replay_branch(rt, ctx, batches, stream_seed, H, cand_batches, action_j=None,
                              lam=None):
    """One real replay branch mirroring entry._replay_branch call-for-call; at the
    endpoint computes V_score and V_eval (both in-trajectory, same weights)."""
    if lam is None:
        lam = entry.FROZEN_LAMBDA
    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    entry._restore_and_seed_pre(rt, ctx, stream_seed)
    for horizon in range(1, int(H) + 1):
        native_gradient, _, _ = entry._forward_backward(rt, ctx, batches, horizon,
                                                        want_snapshot=False)
        if horizon == 1 and action_j is not None:
            cand_tensors = a0g._stack_all(rt["dataset"], cand_batches[action_j], rt["device"])
            rng_snap = capture_rng_state()
            g_j = a0g._grad_on_batches(torch, rt["model"], named, cand_tensors,
                                       rt["autocast_ctx"], accum)
            restore_rng_state(rng_snap)
            del cand_tensors
            for name, p in named:
                p.grad.copy_(native_gradient[name]
                             + float(lam) * (g_j[name] - native_gradient[name]))
        entry._clip_and_step(rt, native_gradient, None)
    rng_snap = capture_rng_state()
    v_score = float(a0g._val_loss_only(torch, rt["model"], rt["val_score_tensors"],
                                       rt["autocast_ctx"]))
    v_eval = float(a0g._val_loss_only(torch, rt["model"], rt["val_eval_tensors"],
                                      rt["autocast_ctx"]))
    restore_rng_state(rng_snap)
    return {"v_score": v_score, "v_eval": v_eval,
            "endpoint_model_hash": a0g.stable_hash(
                __import__("fr.minimind_v03_runtime", fromlist=["raw_model"]).raw_model(
                    rt["model"]).state_dict())}


# ---------------------------------------------------------------------------
# phase helpers
# ---------------------------------------------------------------------------
def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _budget_log(rec: dict):
    p = OUT / "gpu_budget_log.json"
    logs = []
    if p.exists():
        logs = json.loads(p.read_text())
    logs.append(rec)
    p.write_text(json.dumps(logs, indent=1))


def _load_manifest():
    return json.loads((OUT / "acceptance_manifest.json").read_text())


def _unit_ctx_stream(rt, unit):
    root = PROJECT_ROOT
    cfg = rt["cfg"]
    from fr.pilot_partition import load_partition
    partition = load_partition(root / cfg["output_dir"] / "partition_v07.json")
    group_definition = json.loads((root / a0g.OUTPUT_DIR / "data_group_definition.json").read_text())
    ctx = aprime_driver.build_context(root, cfg, partition, group_definition,
                                      int(unit["train_seed"]), int(unit["step"]))
    stream = next(s for s in ctx.streams if int(s["stream_id"]) == int(unit["stream_id"]))
    return ctx, stream


def _slim_scores(res):
    out = {"hhat_fwd_C": {hk: {str(j): float(v) for j, v in per.items()}
                          for hk, per in res["hhat_fwd_C"].items()}}
    out["hhat_adj"] = {hk: {str(j): float(v) for j, v in per["hhat_adj"].items()}
                       for hk, per in res["vjp_shared"].items()}
    out["clip_trace"] = res["clip_trace"]
    return out


def phase_p1(rt, manifest, only_unit=None):
    """Scorers for all planned units -> predictions/ (locked, no-op explicit).

    Uses the PRODUCTION ENTRY modes (forward_only/adjoint_only, hvp fp32/bf16) at
    the frozen decision horizon H=4 only — the same implementation validated in
    round-2 (entry fwd/adj bit-identical to the frozen diag pass at same config,
    selfcheck gap=0). The full multi-H diagnostics pass was exercised once in the
    dev rehearsal; per contract the H=1/H=8 diffs are diagnostic and this round's
    main analysis is H=4-only, so P1 scores at H_dec only (budget cap 1800s).
    """
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    lam = float(manifest["semantics"]["lambda"])
    H_dec = int(manifest["semantics"]["H"])
    M = len(manifest["semantics"]["action_ids"]) - 1
    done, skipped = [], []
    for unit in manifest["units"]:
        uid = unit["unit_id"]
        if only_unit and uid != only_unit:
            continue
        out_path = pred_dir / f"predictions_{uid}.json"
        if out_path.exists():
            skipped.append(uid)
            continue
        t0 = time.perf_counter()
        ctx, stream = _unit_ctx_stream(rt, unit)
        r_bf16 = entry.run_forward_only(rt, ctx, stream, H_dec, M, False, entry._new_counters())
        r_f32f = entry.run_forward_only(rt, ctx, stream, H_dec, M, True, entry._new_counters())
        r_adj = entry.run_adjoint_only(rt, ctx, stream, H_dec, M, True, entry._new_counters())
        wall = time.perf_counter() - t0

        h4_bf16 = {int(j): float(v) for j, v in r_bf16["scores_hhat_fwd"].items()}
        h4_f32f = {int(j): float(v) for j, v in r_f32f["scores_hhat_fwd"].items()}
        h4_adj = {int(j): float(v) for j, v in r_adj["scores_hhat_adj"].items()}
        for tag, sc in (("bf16", h4_bf16), ("fp32fwd", h4_f32f), ("fp32adj", h4_adj)):
            for j, v in sc.items():
                if not math.isfinite(v):
                    raise ValueError(f"NON_FINITE scorer output {tag} j={j}: {v!r}")

        rec = {
            "unit_id": uid, "H_decision": H_dec, "lambda": lam,
            "scorer_mode": "production_entry forward_only/adjoint_only at H_dec (round-2 selfcheck: bit-identical to frozen diag pass at same config)",
            "state": {"train_seed": unit["train_seed"], "step": unit["step"],
                      "q_sha256": unit["q_sha256"], "stream_id": unit["stream_id"],
                      "stream_seed": unit["stream_seed"],
                      "batch_manifest_sha256": unit["batch_manifest_sha256"]},
            "scorers_H4": {
                "hhat": {"bf16_fwd_C": {str(j): h4_bf16[j] for j in sorted(h4_bf16)},
                         "fp32hvp_fwd": {str(j): h4_f32f[j] for j in sorted(h4_f32f)},
                         "fp32hvp_adjoint": {str(j): h4_adj[j] for j in sorted(h4_adj)}},
                "units": {"hhat": "dL_score/dlambda (loss per unit lambda)",
                          "predicted_delta_loss": "lambda*hhat (loss)",
                          "predicted_gain": "-lambda*hhat (loss)"},
            },
            "mode_fingerprints": {
                "bf16_fwd": {"hvp_fp32": r_bf16["hvp_fp32"], "mode": r_bf16["mode"]},
                "fp32_fwd": {"hvp_fp32": r_f32f["hvp_fp32"], "mode": r_f32f["mode"]},
                "fp32_adj": {"hvp_fp32": r_adj["hvp_fp32"], "mode": r_adj["mode"],
                             "w_0_norm": r_adj["w_0_norm"]},
                "adjoint_constraint": ("adjoint_only ran without candidate future tangents; "
                                       "forward references are separate runs (audit cost)"),
            },
            "counters": {"bf16_fwd": r_bf16["counters"], "fp32_fwd": r_f32f["counters"],
                         "fp32_adj": r_adj["counters"]},
            "choices_pre_eval": {
                name: {"chosen_id": frozen_decision(sc, lam),
                       "five_action_ranking": five_action_ranking(sc, lam),
                       "margin": prediction_margin(sc, lam)}
                for name, sc in (("bf16C_argmin", h4_bf16),
                                 ("fp32fwd_argmin", h4_f32f),
                                 ("fp32adj_argmin", h4_adj))},
            "wall_seconds": wall,
        }
        out_path.write_text(json.dumps(rec, indent=1))
        _budget_log({"phase": "p1", "unit": uid, "wall_s": wall,
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        done.append(uid)
        print(f"[P1 {uid}] done ({wall:.1f}s)", flush=True)
    return {"done": done, "skipped_existing": skipped}


def phase_p2(manifest):
    """Same-config FP32 numerical gate from P1 predictions."""
    H_dec = int(manifest["semantics"]["H"])
    gate = {"phase": "P2", "formula": "abs(hhat_fwd_fp32 - hhat_adj) <= atol32 + rtol32*abs(hhat_fwd_fp32)",
            "atol32": ATOL32, "rtol32": RTOL32, "scope": f"H={H_dec}, all 4 active actions",
            "units": {}, "all_pass": None}
    all_pass = True
    any_finite_issue = False
    for f in sorted((OUT / "predictions").glob("predictions_*.json")):
        rec = json.loads(f.read_text())
        uid = rec["unit_id"]
        per = {}
        for js in sorted(rec["scorers_H4"]["hhat"]["fp32hvp_fwd"]):
            fwd = float(rec["scorers_H4"]["hhat"]["fp32hvp_fwd"][js])
            adj = float(rec["scorers_H4"]["hhat"]["fp32hvp_adjoint"][js])
            thr = ATOL32 + RTOL32 * abs(fwd)
            finite = all(math.isfinite(x) for x in (fwd, adj, thr)) and thr >= 0.0
            if not finite:
                any_finite_issue = True
            per[f"j{js}"] = {"fwd": fwd, "adj": adj, "abs_gap": abs(fwd - adj),
                             "threshold": thr, "finite": finite,
                             "pass": bool(finite and abs(fwd - adj) <= thr)}
        up = all(v["pass"] for v in per.values())
        gate["units"][uid] = {"per_action": per, "unit_pass": up}
        all_pass = all_pass and up
    n = len(gate["units"])
    gate["n_units"] = n
    gate["finite_check"] = "FAIL_NON_FINITE" if any_finite_issue else "OK"
    gate["all_pass"] = bool(all_pass and not any_finite_issue) if n else None
    gate["status"] = ("PASS" if gate["all_pass"] else
                      ("NOT_EVALUABLE_NO_PREDICTIONS" if n == 0 else "FAIL"))
    gate["claim_limit"] = ("pass means same-config implementations consistent only; "
                           "NOT native-derivative absolute correctness")
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    print(f"[P2] gate={gate['status']} over {n} units", flush=True)
    return gate


def phase_p3(rt, manifest, only_unit=None):
    """Per-unit 5-action dual-label replay. Score labels first; replay-score choice
    locked from score only; eval labels written to a separate module afterwards."""
    lam = float(manifest["semantics"]["lambda"])
    H_dec = int(manifest["semantics"]["H"])
    score_dir = OUT / "labels" / "score"
    eval_dir = OUT / "labels" / "eval"
    score_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for unit in manifest["units"]:
        uid = unit["unit_id"]
        if only_unit and uid != only_unit:
            continue
        sp = score_dir / f"score_labels_{uid}.json"
        ep = eval_dir / f"eval_labels_{uid}.json"
        if sp.exists() and ep.exists():
            skipped.append(uid)
            continue
        t0 = time.perf_counter()
        ctx, stream = _unit_ctx_stream(rt, unit)
        cand = ctx.candidate_batches(stream["stream_seed"])
        v_score, v_eval, hashes = {}, {}, {}
        for j in [0] + sorted(cand, key=int):
            br = _dual_label_replay_branch(rt, ctx, stream["batches"], stream["stream_seed"],
                                           H_dec, cand, action_j=(None if j == 0 else j), lam=lam)
            v_score[str(j)] = br["v_score"]
            v_eval[str(j)] = br["v_eval"]
            hashes[str(j)] = br["endpoint_model_hash"]
            for val in (br["v_score"], br["v_eval"]):
                if not math.isfinite(val):
                    raise ValueError(f"NON_FINITE replay label unit={uid} j={j}: {val!r}")
        wall = time.perf_counter() - t0
        base = v_score["0"]
        score_rec = {
            "unit_id": uid, "H": H_dec, "lambda": lam, "baseline": "j=0 no-op continues training",
            "v_score": v_score,
            "delta_L_score": {str(j): v_score[str(j)] - base for j in v_score},
            "gain_vs_noop_score": {str(j): base - v_score[str(j)] for j in v_score},
            "endpoint_model_hash": hashes,
        }
        gains = score_rec["gain_vs_noop_score"]
        best = max(gains, key=lambda j: (gains[j], -int(j)))
        score_rec["replay_score_oracle"] = {"chosen_id": int(best),
                                            "choose_noop": bool(gains[best] <= 0.0)}
        sp.write_text(json.dumps(score_rec, indent=1))
        # lock from score only (record hash), THEN write eval labels separately
        eval_rec = {
            "unit_id": uid, "H": H_dec,
            "score_labels_sha256_at_lock": _sha256_file(sp),
            "v_eval": v_eval,
            "delta_L_eval": {str(j): v_eval[str(j)] - v_eval["0"] for j in v_eval},
            "gain_vs_noop_eval": {str(j): v_eval["0"] - v_eval[str(j)] for j in v_eval},
            "endpoint_model_hash": hashes,
            "eval_batch_identity": "validation_split v_eval (frozen a0g split)",
        }
        ep.write_text(json.dumps(eval_rec, indent=1))
        _budget_log({"phase": "p3", "unit": uid, "wall_s": wall,
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        done.append(uid)
        print(f"[P3 {uid}] done ({wall:.1f}s) oracle={score_rec['replay_score_oracle']}", flush=True)
    return {"done": done, "skipped_existing": skipped}


def phase_dev(rt, manifest, seed, step, stream_id):
    """ONE end-to-end dev rehearsal on the dev unit (D state; never in acceptance):
    model rebuild already done (build_runtime) -> context build -> scorers ->
    gate self-check -> dual-label replay -> identity checks -> outputs -> clean exit.
    """
    dev = manifest["dev_unit"]
    t_dev0 = time.perf_counter()
    assert (int(seed), int(step), int(stream_id)) == (
        int(dev["train_seed"]), int(dev["step"]), int(dev["stream_id"]))
    unit = {"unit_id": dev["unit_id"], "train_seed": int(seed), "step": int(step),
            "stream_id": int(stream_id), "H": int(dev["H"]),
            "lambda": float(manifest["semantics"]["lambda"])}
    out = OUT / "dev_rehearsal"
    out.mkdir(parents=True, exist_ok=True)

    # chain step 1: context + stream identity (string/int key discipline)
    ctx, stream = _unit_ctx_stream(rt, unit)
    cand = ctx.candidate_batches(stream["stream_seed"])
    assert sorted(cand, key=int) == [1, 2, 3, 4], "candidate ids"
    assert set(stream.keys()) >= {"stream_id", "stream_seed", "batches"}

    # chain step 2: scorers (both configs)
    res_bf16 = aprime_driver.run_pass(rt, ctx, stream, hvp_fp32=False)
    res_fp32 = aprime_driver.run_pass(rt, ctx, stream, hvp_fp32=True)
    slim = {"bf16": _slim_scores(res_bf16), "fp32hvp": _slim_scores(res_fp32)}

    # chain step 3: same-config gate self-check
    H_dec = int(dev["H"])
    gchk = {}
    for js in sorted(slim["fp32hvp"]["hhat_fwd_C"][str(H_dec)]):
        fwd = float(slim["fp32hvp"]["hhat_fwd_C"][str(H_dec)][js])
        adj = float(slim["fp32hvp"]["hhat_adj"][str(H_dec)][js])
        gchk[f"j{js}"] = {"fwd": fwd, "adj": adj, "abs_gap": abs(fwd - adj),
                          "threshold": ATOL32 + RTOL32 * abs(fwd),
                          "pass": bool(abs(fwd - adj) <= ATOL32 + RTOL32 * abs(fwd))}

    # chain step 4: dual-label replay + identity vs frozen entry replay path
    lam = float(manifest["semantics"]["lambda"])
    rep_frozen = entry.run_replay_only(rt, ctx, stream, H_dec, len(cand), entry._new_counters())
    mine = {}
    for j in [0] + sorted(cand, key=int):
        mine[j] = _dual_label_replay_branch(rt, ctx, stream["batches"], stream["stream_seed"],
                                            H_dec, cand, action_j=(None if j == 0 else j), lam=lam)
    ident = {}
    for j in mine:
        ident[str(j)] = {
            "v_score_dual_label_branch": mine[j]["v_score"],
            "v_score_frozen_replay_only": float(rep_frozen["v_score"][str(j)]),
            "abs_gap": abs(mine[j]["v_score"] - float(rep_frozen["v_score"][str(j)])),
        }
    max_gap = max(v["abs_gap"] for v in ident.values())
    # zero-intervention == baseline cross-check
    assert abs(mine[0]["v_score"] - float(rep_frozen["v_score"]["0"])) <= 1e-9, \
        "zero-intervention baseline identity violated"

    report = {
        "dev_unit": dev["unit_id"], "chain": [
            "build_runtime (model rebuild)", "context+stream identity (str/int keys)",
            "candidate set j=1..4", "scorers bf16+fp32hvp", "gate self-check",
            "frozen entry replay (cross-check)", "dual-label replay x5",
            "identity vs frozen replay", "outputs + clean exit"],
        "model_dims": rt["model_config"],
        "gate_selfcheck": gchk,
        "identity_vs_frozen_replay": ident,
        "max_abs_identity_gap": max_gap,
        "pre_step_mv_step_trap_note": ("snapshots use optimizer-internal state['step']+1 "
                                       "(bias correction), not lr-schedule micro step"),
        "slim_scorers": slim,
    }
    (out / "dev_rehearsal_report.json").write_text(json.dumps(report, indent=1))
    _budget_log({"phase": "dev", "unit": dev["unit_id"],
                 "wall_s": time.perf_counter() - t_dev0,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"[DEV] rehearsal done; max identity gap vs frozen replay = {max_gap:.3e}",
          flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("dev", "p1", "p2", "p3", "batch"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--stream", type=int)
    ap.add_argument("--unit", help="restrict p1/p3 to one unit_id")
    args = ap.parse_args()

    manifest = _load_manifest()
    t0 = time.perf_counter()
    rt = entry.build_runtime(args.device)
    preload = time.perf_counter() - t0
    _budget_log({"phase": "preload", "wall_s": preload,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    if args.phase == "dev":
        assert args.seed and args.step and args.stream is not None
        phase_dev(rt, manifest, args.seed, args.step, args.stream)
    elif args.phase == "p1":
        phase_p1(rt, manifest, only_unit=args.unit)
    elif args.phase == "p2":
        phase_p2(manifest)
    elif args.phase == "p3":
        phase_p3(rt, manifest, only_unit=args.unit)
    else:  # batch: fixed order p1 -> p2 (gate) -> p3, stopping at gate failure
        phase_p1(rt, manifest)
        gate = phase_p2(manifest)
        if not gate["all_pass"]:
            print("STOPPED_AT_NUMERICAL_GATE; utility phase not entered", flush=True)
            (OUT / "COMPLETION_STATUS.json").write_text(json.dumps({
                "COMPLETION_STATUS": "STOPPED_AT_NUMERICAL_GATE",
                "NUMERICAL_GATE": "FAIL", "UTILITY_PILOT_RESULT": "NOT_COVERED"}, indent=1))
            return 0
        phase_p3(rt, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
