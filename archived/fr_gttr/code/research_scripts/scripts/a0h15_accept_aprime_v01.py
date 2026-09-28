"""A0H-1.5 round-2 A' independent acceptance driver (preregistration-gated).

Executes the FROZEN acceptance protocol on the three pre-registered unseen units
(see outputs/a0h15_vjp_shared_v01/accept_aprime/PREREGISTRATION.md, written and
hash-frozen BEFORE this script first touches the units):

  per unit:
    L1  B1 same-config gate on the hvp_fp32=True diagnostics pass
        |hhat_fwd_C - hhat_adj| <= atol32 + rtol32*|hhat_fwd_C|  (all H in {1,4,8}, all j)
    L1b BF16-forward vs FP32-adjoint decision consistency at H=4 under the FROZEN
        A0G selector (argmin predicted Delta L over j=0..4, no-op included),
        H=1/H=8 raw rankings reported for information.
    L2  chosen-action native +lambda replay utility: V_score gain vs no-op > 0 at H=4
        (frozen entry replay_only), V_eval secondary via frozen a0g._run_action_branch
        + explicit no-op V_eval branch.

2026-09-15 decision-semantics audit (outputs/a0h15_vjp_shared_v01/decision_semantics_audit/):
the original version of this driver selected argmax of the RAW hhat (max predicted
Delta L = worst action), violating the frozen A0G selector semantics
(fr/a0g_decision_utility_v01.py select_action = argmin predicted Delta L, j=0..4,
tie -> lowest j). Per the frozen A' retry rule 2, that code change after exposure
demotes the executed A' to dev-investigation (NO A'' this round); the original A'
verdicts are preserved UNCHANGED in this directory. This fixed driver restores the
frozen selector and MUST only be used for a future freshly pre-registered batch.

State contexts are built by replicating a0g._load_v07_state_contexts for arbitrary
(seed, step) with the SAME manifest identity / sha256 / protocol validations.
Frozen modules are only called, never modified.

Usage:
  PYTHONPATH=research/src CUDA_VISIBLE_DEVICES=1 python research/scripts/a0h15_accept_aprime_v01.py \
      --device cuda:0 --horizon 4 --budget-seconds 500
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT_DIR = "outputs/a0h15_vjp_shared_v01/accept_aprime"
ATOL32, RTOL32 = 1e-5, 1e-4
A_PRIME_UNITS = ((2030, 52), (2033, 64), (2037, 82))  # frozen at preregistration

_spec = importlib.util.spec_from_file_location(
    "a0h15_entry", PROJECT_ROOT / "research/scripts/a0h15_production_entry_v01.py")
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)

import a0g_decision_utility_v01 as a0g  # noqa: E402
import a0h15_vjp_diagnostics_v01 as diag  # noqa: E402

from fr.a0g_decision_utility_v01 import (  # noqa: E402  FROZEN selector, read-only
    select_action as frozen_select_action,
)

from fr.minimind_v03_runtime import raw_model  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.stream_rng import seed_future_stream  # noqa: E402


def build_context(root, cfg, partition, group_definition, seed: int, step: int):
    """Replication of a0g._load_v07_state_contexts for ONE arbitrary unit, keeping
    every frozen validation (identity, sha256, protocol, stream_seed, coverage)."""
    v07 = root / cfg["output_dir"]
    theta = a0g._read_json(v07 / f"seed_{seed}/theta_mapping_raw.json")
    theta_state = next(r for r in theta["states"] if int(r["step"]) == step)
    checkpoint_path = v07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
    streams = []
    for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
        manifest_path = a0g._resolve(root, rec["batch_manifest_path"])
        payload = a0g._read_json(manifest_path)
        if payload.get("manifest_version") != "0.2":
            raise RuntimeError("A' manifest version mismatch")
        metadata = payload.get("metadata")
        batches = payload.get("batches")
        actual = {"train_seed": int(metadata.get("train_seed", -1)),
                  "state_step": int(metadata.get("state_step", -1)),
                  "stream_id": int(metadata.get("stream_id", -1))}
        expected = {"train_seed": seed, "state_step": step,
                    "stream_id": int(rec["stream_id"])}
        if actual != expected:
            raise RuntimeError(f"A' manifest identity mismatch: {actual} != {expected}")
        if metadata.get("protocol") != "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM":
            raise RuntimeError("A' manifest protocol mismatch")
        content_sha = a0g.manifest_sha256(batches)
        if content_sha != str(payload.get("sha256")) or content_sha != str(rec["batch_manifest_sha256"]):
            raise RuntimeError("A' manifest sha mismatch")
        if int(payload["metadata"]["stream_seed"]) != int(rec["stream_seed"]):
            raise RuntimeError("A' manifest stream_seed mismatch")
        if len(batches) < max(a0g.HORIZONS) * int(cfg["accumulation_steps"]):
            raise RuntimeError("A' manifest does not cover H=8 window")
        streams.append({"stream_id": int(rec["stream_id"]),
                        "stream_seed": int(rec["stream_seed"]), "batches": batches,
                        "manifest_sha256": rec["batch_manifest_sha256"],
                        "manifest_path": str(manifest_path)})
    meta = a0g.torch_load_metadata(checkpoint_path)
    return a0g._StateContext(split="A_prime", seed=seed, step=step,
                             checkpoint_path=checkpoint_path, streams=streams,
                             group_definition=group_definition, partition=partition,
                             cfg=cfg, bank_meta=meta)


def run_pass(rt, ctx, stream, hvp_fp32: bool):
    res = diag._run_diagnostics_pass(
        torch=rt["torch"], model=rt["model"], optimizer=rt["optimizer"],
        scaler=rt["scaler"], autocast_ctx=rt["autocast_ctx"], dataset=rt["dataset"],
        named=rt["named"], checkpoint_path=ctx.checkpoint_path,
        batches=stream["batches"], stream_seed=stream["stream_seed"],
        cand_batches_by_j=ctx.candidate_batches(stream["stream_seed"]),
        val_score_tensors=rt["val_score_tensors"],
        accumulation_steps=int(rt["cfg"]["accumulation_steps"]), device=rt["device"],
        grad_clip=float(rt["cfg"]["grad_clip"]), base_lr=float(rt["cfg"]["learning_rate"]),
        global_micro_step=int(ctx.bank_meta["global_micro_step"]),
        total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
        get_lr=rt["get_lr"], timers=rt["timers"], hvp_fp32=hvp_fp32,
        full_diagnostics=False)
    return res


def ranking(scores: dict):
    """Informational raw-score ranking (descending hhat). NOTE: raw hhat is
    dL/dlambda, so descending hhat is the ANTI-selector order; decisions must
    use frozen_decision() below (argmin predicted Delta L, no-op included)."""
    return sorted(scores, key=lambda j: (-scores[j], int(j)))


def frozen_decision(hhat_by_j: dict, lam: float) -> int:
    """FROZEN A0G decision semantics: hhat = dL/dlambda (per unit lambda);
    predicted Delta L_j = lam * hhat_j; j=0 no-op has predicted Delta L = 0 and
    IS in the candidate set; select argmin with tie-break to the lowest j
    (exactly fr.a0g_decision_utility_v01.select_action)."""
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        predicted[int(j)] = float(lam) * float(v)
    return frozen_select_action(predicted)


def noop_val_eval(rt, ctx, stream, H):
    """Explicit no-op branch V_eval at H (frozen replay semantics, no injection)."""
    torch = rt["torch"]
    entry._restore_and_seed(rt, ctx, stream)
    for horizon in range(1, int(H) + 1):
        native_gradient, _, _ = entry._forward_backward(
            rt, ctx, stream["batches"], horizon, want_snapshot=False)
        entry._clip_and_step(rt, native_gradient, None)
    return float(a0g._val_loss_only(torch, rt["model"], rt["val_eval_tensors"],
                                    rt["autocast_ctx"]))


def slim(res, hvp_fp32):
    """Persist only JSON-safe scoring fields (raw res carries full tensor maps)."""
    return {"hvp_fp32": bool(hvp_fp32),
            "hhat_fwd_C": {hk: {str(j): float(v) for j, v in per.items()}
                           for hk, per in res["hhat_fwd_C"].items()},
            "vjp_shared_hhat_adj": {hk: {str(j): float(v) for j, v in
                                         res["vjp_shared"][hk]["hhat_adj"].items()}
                                    for hk in res["vjp_shared"]},
            "clip_trace": res["clip_trace"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--budget-seconds", type=float, default=500.0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    root = PROJECT_ROOT
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rt = entry.build_runtime(args.device)
    cfg = rt["cfg"]
    partition = __import__("fr.pilot_partition", fromlist=["load_partition"]).load_partition(
        root / cfg["output_dir"] / "partition_v07.json")
    group_definition = json.loads(
        (root / a0g.OUTPUT_DIR / "data_group_definition.json").read_text())

    verdicts = {}
    H_dec = int(args.horizon)
    for (seed, step) in A_PRIME_UNITS:
        label = f"{seed}_{step:03d}"
        if time.perf_counter() - t0 > args.budget_seconds:
            print("A_PRIME_BUDGET_EXHAUSTED before", label, flush=True)
            break
        ctx = build_context(root, cfg, partition, group_definition, seed, step)
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        unit = {"unit": f"{seed}/{step}", "stream_id": 0,
                "global_micro_step": int(ctx.bank_meta["global_micro_step"]),
                "optimizer_step": step, "H_decision": H_dec}

        res_bf16 = run_pass(rt, ctx, stream, hvp_fp32=False)
        (out_dir / f"diag_bf16_{label}_s0.json").write_text(json.dumps(slim(res_bf16, False), indent=1))
        res_fp32 = run_pass(rt, ctx, stream, hvp_fp32=True)
        (out_dir / f"diag_fp32hvp_{label}_s0.json").write_text(json.dumps(slim(res_fp32, True), indent=1))

        cand = sorted(res_fp32["hhat_fwd_C"][str(H_dec)], key=int)
        b1 = {}
        for hk in ("1", "4", "8"):
            for js in sorted(res_fp32["hhat_fwd_C"][hk], key=int):
                f = float(res_fp32["hhat_fwd_C"][hk][js])
                a = float(res_fp32["vjp_shared"][hk]["hhat_adj"][str(js)])
                thr = ATOL32 + RTOL32 * abs(f)
                b1[f"H{hk}_j{js}"] = {"fwd": f, "adj": a, "abs_gap": abs(f - a),
                                      "threshold": thr, "pass": bool(abs(f - a) <= thr)}
        ranks = {}
        decisions = {}
        for hk in ("1", "4", "8"):
            bf = {int(j): float(res_bf16["hhat_fwd_C"][hk][j]) for j in res_bf16["hhat_fwd_C"][hk]}
            fa = {int(j): float(res_fp32["vjp_shared"][hk]["hhat_adj"][j]) for j in res_fp32["vjp_shared"][hk]["hhat_adj"]}
            ranks[f"H{hk}"] = {
                "bf16_forward_scores": bf, "fp32_adjoint_scores": fa,
                "bf16_ranking": [int(j) for j in ranking(bf)],
                "fp32_ranking": [int(j) for j in ranking(fa)],
                # legacy raw argmax (ANTI-selector order), kept for audit traceability
                "raw_argmax_bf16": max(bf, key=lambda j: (bf[j], -j)),
                "raw_argmax_fp32": max(fa, key=lambda j: (fa[j], -j))}
            # FROZEN A0G five-action decisions (no-op j=0 in candidate set)
            decisions[f"H{hk}"] = {
                "frozen_chosen_bf16": frozen_decision(bf, entry.FROZEN_LAMBDA),
                "frozen_chosen_fp32_adj": frozen_decision(fa, entry.FROZEN_LAMBDA)}
        agree = (decisions[f"H{H_dec}"]["frozen_chosen_bf16"]
                 == decisions[f"H{H_dec}"]["frozen_chosen_fp32_adj"])
        b1_pass = all(v["pass"] for v in b1.values())
        unit.update({"B1_gate": b1, "B1_all_pass": b1_pass, "rankings": ranks,
                     "frozen_decisions": decisions,
                     "H_decision_action_agreement": bool(agree),
                     "chosen_action_fp32_adj": decisions[f"H{H_dec}"]["frozen_chosen_fp32_adj"]})

        # L2 utility: frozen entry replay (V_score gains) + V_eval secondary
        rep = entry.run_replay_only(rt, ctx, stream, H_dec, len(cand), entry._new_counters())
        (out_dir / f"replay_{label}_s0.json").write_text(json.dumps(rep, indent=1, default=float))
        chosen = decisions[f"H{H_dec}"]["frozen_chosen_fp32_adj"]
        v_gain = float(rep["gain_vs_noop"][str(chosen)])
        # chosen-action V_eval via frozen a0g branch + explicit no-op V_eval
        branch = a0g._run_action_branch(
            torch=rt["torch"], model=rt["model"], optimizer=rt["optimizer"],
            scaler=rt["scaler"], autocast_ctx=rt["autocast_ctx"], dataset=rt["dataset"],
            named=rt["named"], checkpoint_path=ctx.checkpoint_path,
            batches=stream["batches"], stream_seed=stream["stream_seed"],
            action_j=chosen, lam=entry.FROZEN_LAMBDA, sign=+1,
            cand_batches_by_j=ctx.candidate_batches(stream["stream_seed"]),
            val_score_tensors=rt["val_score_tensors"],
            val_eval_tensors=rt["val_eval_tensors"], g_reference=None,
            accumulation_steps=int(cfg["accumulation_steps"]), device=rt["device"],
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=rt["get_lr"], timers=rt["timers"])
        ve_noop = noop_val_eval(rt, ctx, stream, H_dec)
        ve_gain = ve_noop - float(branch["val_eval"][str(H_dec)])
        unit.update({
            "L2_replay": {"chosen_action": chosen,
                          "v_score_noop": float(rep["v_score"]["0"]),
                          "v_score_chosen": float(rep["v_score"][str(chosen)]),
                          "v_score_gain": v_gain,
                          "choose_noop": bool(rep["choose_noop"]),
                          "replay_chosen_id": int(rep["chosen_id"]),
                          "replay_chosen_gain": float(rep["gain_vs_noop"][str(rep["chosen_id"])])},
            "L2_veval_secondary": {"chosen_action_veval": float(branch["val_eval"][str(H_dec)]),
                                   "noop_veval": float(ve_noop), "veval_gain": float(ve_gain),
                                   "first_step": branch["first_step"]},
            "L2_gain_positive": bool(v_gain > 0.0)})
        unit["verdict"] = ("PASS" if (b1_pass and agree and v_gain > 0.0) else "FAIL")
        verdicts[f"{seed}/{step}"] = unit
        (out_dir / f"verdict_{label}_s0.json").write_text(json.dumps(unit, indent=1, default=float))
        print(f"[{seed}/{step}] B1={'PASS' if b1_pass else 'FAIL'} "
              f"agree={agree} gain={v_gain:.3e} -> {unit['verdict']}", flush=True)

    summary = {"tag": "A_PRIME_ACCEPTANCE", "units": verdicts,
               "preregistration": "outputs/a0h15_vjp_shared_v01/accept_aprime/PREREGISTRATION.md",
               "code_frozen_hash_file": "outputs/a0h15_vjp_shared_v01/accept_aprime/code_frozen_at_prereg.sha256"}
    (out_dir / "A_prime_verdicts.json").write_text(json.dumps(summary, indent=1, default=float))
    print("wrote", out_dir / "A_prime_verdicts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
