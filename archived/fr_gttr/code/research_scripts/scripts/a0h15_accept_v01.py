"""A0H-1.5 branch-A acceptance runner: frozen BF16 shared adjoint on acceptance set A.

Contract: docs/FR_A0H15_PROTOCOL.md §6 branch A + §1 read-once rule.
Runs ONE joint pass per acceptance unit (dev:2032/58, dev:2035/70, dev:2037/46,
stream 0) reusing the FROZEN a0h_adjoint_dev_check_v01 joint forward machinery
verbatim (read-only import) plus the algebraic shared-adjoint walk, then applies
the FROZEN A0H-1 decision-level comparison (ranking/selection/no-op) between
hhat_fwd["C"] and the shared adjoint. Writes accept_<seed>_<step>_s0.json.
After this run, acceptance data MUST NOT be used for debugging/threshold tuning;
if it is, demote to dev and record.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUTPUT_DIR = "outputs/a0h15_vjp_shared_v01"
ACCEPT_STATES = ("2032/58", "2035/70", "2037/46")  # frozen at protocol time

_spec = importlib.util.spec_from_file_location(
    "a0h_adjoint_dev_check_v01",
    PROJECT_ROOT / "research" / "scripts" / "a0h_adjoint_dev_check_v01.py",
)
_a0h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a0h1)


def run_accept_unit(*, a0g, torch, model, optimizer, scaler, autocast_ctx, dataset,
                    named, ctx, stream, val_score_tensors, cfg, get_lr, timers,
                    diag_mod):
    """One joint pass + shared algebraic adjoint walk for ONE acceptance unit.

    Reuses a0h15._run_diagnostics_pass with full_diagnostics=False (forward pass
    identical to the frozen joint pass; RNG discipline identical). The shared
    walk result vjp_shared[h]['hhat_adj'] is the candidate-independent w_0 score.
    """
    from fr.a0c_adamw_response_propagation_v01 import project_tangent

    cand = ctx.candidate_batches(stream["stream_seed"])
    res = diag_mod._run_diagnostics_pass(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, dataset=dataset, named=named,
        checkpoint_path=ctx.checkpoint_path, batches=stream["batches"],
        stream_seed=stream["stream_seed"], cand_batches_by_j=cand,
        val_score_tensors=val_score_tensors,
        accumulation_steps=int(cfg["accumulation_steps"]),
        device=str(model.device), grad_clip=float(cfg["grad_clip"]),
        base_lr=float(cfg["learning_rate"]),
        global_micro_step=int(ctx.bank_meta["global_micro_step"]),
        total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
        get_lr=get_lr, timers=timers, full_diagnostics=False)

    a_by_j = res["a_by_j"]
    shared = res["vjp_shared"]

    # hhat_fwd_C comes directly from the diagnostics pass (frozen formula
    # hhat_fwd_C[j][h] = <v_score_grad[h], theta tangent after h steps>), avoiding
    # any telescope-identity reconstruction ambiguity. Keys are str(h)/int(j).
    hhat_fwd_C = {}
    for horizon in a0g.HORIZONS:
        hk = str(horizon)
        hhat_fwd_C[hk] = {int(j): float(res["hhat_fwd_C"][hk][j]) for j in cand}

    methods = {}
    for horizon in a0g.HORIZONS:
        hk = str(horizon)
        fwd_by_j = hhat_fwd_C[hk]
        adj_by_j = {int(j): float(shared[hk]["hhat_adj"][str(j)]) for j in cand}
        rows, elementwise_ok = _a0h1._compare_hhat(fwd_by_j=fwd_by_j, adj_by_j=adj_by_j)
        fwd_rank, fwd_choice = _a0h1._ranking_and_selection(fwd_by_j)
        adj_rank, adj_choice = _a0h1._ranking_and_selection(adj_by_j)
        adj_sorted = sorted(adj_by_j.items(), key=lambda kv: (kv[1], kv[0]))
        best_active_score = adj_sorted[0][1]
        runner_up = adj_sorted[1][1] if len(adj_sorted) > 1 else float("nan")
        methods[f"C|{hk}"] = {
            "rows": rows,
            "elementwise_pass_frozen_gate": bool(elementwise_ok),
            "fwd_ranking": fwd_rank, "fwd_chosen_id": int(fwd_choice),
            "adj_ranking": adj_rank, "adj_chosen_id": int(adj_choice),
            "decision_match": bool(fwd_rank == adj_rank and fwd_choice == adj_choice),
            "noop_score": 0.0,
            "choose_noop": bool(adj_choice == 0),
            "best_active_score": float(best_active_score),
            "margin": float(best_active_score - runner_up),
        }
    return {"methods": methods,
            "shared_w0_norm": {h: shared[h]["w_0_norm"] for h in shared}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget-seconds", type=float, default=600.0)
    args = ap.parse_args()

    import a0g_decision_utility_v01 as a0g
    import a0h15_vjp_diagnostics_v01 as diag_mod
    from fr.full_state import load_full_training_state  # noqa: F401
    from fr.minimind_v03_runtime import make_amp, raw_model  # noqa: F401
    from fr.native_adamw_guard import assert_native_adamw
    from fr.native_schedule import set_native_micro_lr  # noqa: F401
    from fr.pilot_partition import load_partition
    from fr.rng import capture_rng_state, restore_rng_state  # noqa: F401

    root = PROJECT_ROOT
    out_dir = root / OUTPUT_DIR / "accept"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "research/configs/e3c_confirmatory_v07.json").read_text())
    v07_out = root / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"
    a0g_out = root / a0g.OUTPUT_DIR
    data_group_definition = json.loads(
        (a0g_out / "data_group_definition.json").read_text(encoding="utf-8"))
    validation_split = json.loads(
        (a0g_out / "validation_split.json").read_text(encoding="utf-8"))

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False)
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device)
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + a0g.EVAL_BATCH_SIZE] for i in range(0, len(idx), a0g.EVAL_BATCH_SIZE)]

    val_score_tensors = a0g._stack_all(dataset, _chunk(validation_split["v_score"]), args.device)
    timers = a0g._Timers(torch)
    # Acceptance units are a-priori frozen (protocol §1): A = {(2032,58),(2035,70),(2037,46)}.
    # The frozen a0g loader only exposes DEV_STATES, so we build minimal contexts here
    # using the same verified-loading discipline (manifest sha + metadata identity checks).
    from fr.batch_manifest import manifest_sha256  # same helper used by frozen loader
    dev_ctxs = []
    v07 = root / cfg["output_dir"]
    for state_label in ACCEPT_STATES:
        seed, step = state_label.split("/")
        seed_i, step_i = int(seed), int(step)
        theta = json.loads((v07 / f"seed_{seed_i}/theta_mapping_raw.json").read_text())
        theta_state = next(r for r in theta["states"] if int(r["step"]) == step_i)
        checkpoint_path = v07 / f"seed_{seed_i}/states/state_step_{step_i:03d}.pt"
        streams = []
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            if int(rec["stream_id"]) != 0:
                continue
            manifest_path = root / rec["batch_manifest_path"]
            payload = json.loads(manifest_path.read_text())
            metadata = payload.get("metadata"); batches = payload.get("batches")
            actual = {"train_seed": int(metadata.get("train_seed", -1)),
                      "state_step": int(metadata.get("state_step", -1)),
                      "stream_id": int(metadata.get("stream_id", -1))}
            expected = {"train_seed": seed_i, "state_step": step_i,
                        "stream_id": int(rec["stream_id"])}
            if actual != expected:
                raise RuntimeError(f"accept manifest identity mismatch: {actual} != {expected}")
            content_sha = manifest_sha256(batches)
            if content_sha != str(payload.get("sha256")) or content_sha != str(rec["batch_manifest_sha256"]):
                raise RuntimeError("accept manifest sha mismatch")
            streams.append({"stream_id": int(rec["stream_id"]),
                            "stream_seed": int(rec["stream_seed"]), "batches": batches})
        meta = a0g.torch_load_metadata(checkpoint_path)
        ctx = a0g._StateContext(
            split="accept", seed=seed_i, step=step_i, checkpoint_path=checkpoint_path,
            streams=streams, group_definition=data_group_definition,
            partition=partition, cfg=cfg, bank_meta=meta)
        dev_ctxs.append(ctx)

    wall_start = time.time()
    for ctx in dev_ctxs:
        if time.time() - wall_start > args.budget_seconds:
            print("BUDGET_EXHAUSTED; remaining units NOT_RUN", flush=True)
            break
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        label = f"dev:{ctx.seed}/{ctx.step}"
        print(f"[{label}] acceptance pass ...", flush=True)
        t0 = time.perf_counter()
        rep = run_accept_unit(
            a0g=a0g, torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named, ctx=ctx,
            stream=stream, val_score_tensors=val_score_tensors, cfg=cfg,
            get_lr=get_lr, timers=timers, diag_mod=diag_mod)
        wall = time.perf_counter() - t0
        out_path = out_dir / f"accept_{ctx.seed}_{ctx.step}_s0.json"
        out_path.write_text(json.dumps({
            "state": label, "stream_id": 0, "wall_seconds": wall,
            **rep}, indent=1, default=float))
        print(f"[{label}] wrote {out_path.name} ({wall:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
