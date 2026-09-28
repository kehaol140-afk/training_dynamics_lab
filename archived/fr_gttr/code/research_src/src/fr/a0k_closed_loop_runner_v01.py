"""A0K fresh-response closed-loop runner (core module).

Real training closed loop on the frozen v0.7 recipe. Three paired arms per
seed: A pure AdamW, B fixed-j1 at control steps, C fresh response controller
(production run_adjoint_only decision per event). Control events are
trajectory-invisible by construction: full state+RNG snapshot save, decision
on the snapshot, restore, then a single injected real step.

All controller semantics frozen by outputs/a0k_fresh_response_closed_loop_v01/
contract/contract.json; lookahead/candidate batches frozen in
contract/live_manifest.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from .a0j_future_sensitivity_cache_v01 import LAMBDA as _LAMBDA
from .a0j_future_sensitivity_cache_v01 import choose as _frozen_choose
from .a0j_future_sensitivity_cache_v01 import predicted as _frozen_predicted
from .hashing import stable_hash
from .minimind_v03_runtime import make_amp, raw_model, stack_batch
from .native_adamw_guard import assert_native_adamw
from .native_schedule import set_native_micro_lr
from .rng import capture_rng_state, restore_rng_state
from .stage_c_runtime import batches_from_train_pool
from .full_state import load_full_training_state, save_full_training_state

PROJECT_ROOT = Path(__file__).resolve().parents[3]

A0G_OUT = PROJECT_ROOT / "outputs/fr_decision_utility_v01"
V07_CFG = PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json"
EVAL_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# runtime construction (e3c seeding discipline: setup_seed BEFORE init_model)
# ---------------------------------------------------------------------------

def build_live_runtime(seed: int, device: str) -> dict:
    cfg = json.loads(V07_CFG.read_text())
    assert cfg.get("dtype") == "bfloat16" and int(cfg["accumulation_steps"]) == 2 \
        and float(cfg["grad_clip"]) == 1.0, "config drift"
    v07 = PROJECT_ROOT / cfg["output_dir"]
    partition = json.loads((v07 / "partition_v07.json").read_text())
    validation_split = json.loads((A0G_OUT / "validation_split.json").read_text())
    group_definition = json.loads((A0G_OUT / "data_group_definition.json").read_text())
    minimind = PROJECT_ROOT / "minimind"
    import sys
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    setup_seed(int(seed))
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False)
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=device)
    model.train()
    dataset = PretrainDataset(str(v07 / "e3c_subset.jsonl"), tokenizer,
                              max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + EVAL_BATCH_SIZE] for i in range(0, len(idx), EVAL_BATCH_SIZE)]

    import a0g_decision_utility_v01 as a0g
    rt = {
        "torch": torch, "cfg": cfg, "model": model, "optimizer": optimizer,
        "scaler": scaler, "autocast_ctx": autocast_ctx, "named": named,
        "dataset": dataset, "get_lr": get_lr, "device": device,
        "partition": partition, "group_definition": group_definition,
        "val_score_tensors": a0g._stack_all(dataset, _chunk(validation_split["v_score"]), device),
        "val_eval_tensors": a0g._stack_all(dataset, _chunk(validation_split["v_eval"]), device),
        "train_indices": list(map(int, partition["train_indices"])),
        "a0g": a0g,
    }
    return rt


def make_event_ctx(rt, manifest_event: dict, seed: int, step: int, snapshot_path):
    """Live ctx mirroring a0g._StateContext interface used by pe primitives."""
    cfg = rt["cfg"]
    return SimpleNamespace(
        checkpoint_path=Path(snapshot_path),
        bank_meta={
            "global_micro_step": int(step) * int(cfg["accumulation_steps"]),
            "total_micro_steps": int(cfg["planned_total_optimizer_steps"]) * int(cfg["accumulation_steps"]),
            "global_optimizer_step": int(step),
            "planned_total_optimizer_steps": int(cfg["planned_total_optimizer_steps"]),
        },
        candidate_batches=lambda stream_seed: {
            int(g) + 1: _cand_batches(rt, stream_seed, g)
            for g in range(4)
        },
    )


def _cand_batches(rt, stream_seed, g):
    from a0g_decision_utility_v01 import candidate_batches_for
    cfg = rt["cfg"]
    members = rt["group_definition"]["members"][str(g)]
    return candidate_batches_for(stream_seed, g, members,
                                 int(cfg["batch_size"]), int(cfg["accumulation_steps"]))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def eval_both(rt) -> tuple:
    a0g = rt["a0g"]
    torch = rt["torch"]
    rng_snap = capture_rng_state()
    v_score = float(a0g._val_loss_only(torch, rt["model"], rt["val_score_tensors"],
                                       rt["autocast_ctx"]))
    v_eval = float(a0g._val_loss_only(torch, rt["model"], rt["val_eval_tensors"],
                                      rt["autocast_ctx"]))
    restore_rng_state(rng_snap)
    return v_score, v_eval


def candidate_grad(rt, batches) -> dict:
    a0g = rt["a0g"]
    torch = rt["torch"]
    tensors = a0g._stack_all(rt["dataset"], batches, rt["device"])
    rng_snap = capture_rng_state()
    g = a0g._grad_on_batches(torch, rt["model"], rt["named"], tensors,
                             rt["autocast_ctx"], int(rt["cfg"]["accumulation_steps"]))
    restore_rng_state(rng_snap)
    del tensors
    return g


def rng_hash() -> str:
    import hashlib
    st = capture_rng_state()
    h = hashlib.sha256()
    for v in st.values():
        if hasattr(v, "numpy"):
            h.update(v.numpy().tobytes())
        elif isinstance(v, int):
            h.update(str(v).encode())
        elif isinstance(v, (list, tuple)):
            for x in v:
                h.update(x.cpu().numpy().tobytes() if hasattr(x, "cpu") else str(x).encode())
    return h.hexdigest()


def opt_hash(rt) -> str:
    return stable_hash(rt["optimizer"].state_dict())


def model_hash(rt) -> str:
    return stable_hash(raw_model(rt["model"]).state_dict())


# ---------------------------------------------------------------------------
# control events
# ---------------------------------------------------------------------------

def control_event_C(rt, manifest_event, seed, step, event_dir):
    """Full fresh-response decision. Returns (event_record, snap_path).

    Snapshot -> production run_adjoint_only on snapshot -> restore -> caller
    recomputes g_chosen fresh at the live state and executes the injection.
    full_state v0.2 does NOT save p.grad, so the adjoint walk's gradient
    pollution is cleared by zero_grad on restore; injection re-assigns
    p.grad from cloned tensors (assignment, not copy_).
    """
    import a0h15_production_entry_v01 as pe
    t_event0 = time.perf_counter()
    event_dir.mkdir(parents=True, exist_ok=True)
    snap_path = event_dir / f"event_{seed}_{step}.pt"

    rng_live = capture_rng_state()
    save_full_training_state(
        snap_path, rt["model"], rt["optimizer"], rt["scaler"],
        metadata={
            "protocol": "A0K_CLOSED_LOOP_EVENT_SNAPSHOT",
            "train_seed": int(seed), "event_step": int(step),
            "global_micro_step": int(step) * int(rt["cfg"]["accumulation_steps"]),
            "stream_seed": int(manifest_event["stream_seed"]),
            "lookahead_sha256": manifest_event["lookahead_sha256"],
        })
    rec = {
        "model_hash_before": model_hash(rt),
        "optimizer_hash_before": opt_hash(rt),
        "rng_state_hash": rng_hash(),
        "dataloader_position": int(step) * int(rt["cfg"]["accumulation_steps"]),
    }

    ctx = make_event_ctx(rt, manifest_event, seed, step, snap_path)
    stream = {"stream_id": 0, "stream_seed": int(manifest_event["stream_seed"]),
              "batches": manifest_event["lookahead_batches"]}

    t_dec0 = time.perf_counter()
    counters = pe._new_counters()
    adj = pe.run_adjoint_only(rt, ctx, stream, 4, 4, True, counters)
    t_dec1 = time.perf_counter()
    # decision mutated state/RNG: restore both. zero_grad clears adjoint-walk
    # grad pollution (grads are not part of full_state v0.2).
    load_full_training_state(snap_path, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    restore_rng_state(rng_live)

    hhat = {0: 0.0}
    for j, v in adj["scores_hhat_adj"].items():
        hhat[int(j)] = float(v)
    t_sel0 = time.perf_counter()
    chosen = int(_frozen_choose(hhat))  # applies lambda internally
    t_sel1 = time.perf_counter()
    pred = _frozen_predicted(hhat)  # lambda*hhat for logging

    rec.update({
        "chosen_action_id": chosen,
        "hhat": {str(j): hhat.get(j, 0.0) for j in range(5)},
        "predicted_delta_loss": {str(j): pred.get(j, 0.0) for j in range(5)},
        "w_0_norm": float(adj["w_0_norm"]),
        "final_model_hash_decision": adj["final_model_hash"],
        "counters": counters,
        "rollout_time": t_dec1 - t_dec0,
        "selection_time": t_sel1 - t_sel0,
    })
    rec["_pending"] = (time.perf_counter() - t_event0)
    return rec, snap_path


def recompute_gj_and_inject(rt, manifest_event, chosen: int, native_gradient: dict) -> float:
    """Fresh g_chosen at live state; injects by ASSIGNING p.grad (post-restore
    grads are None). Returns execution wall seconds."""
    t0 = time.perf_counter()
    if chosen == 0:
        for name, p in rt["named"]:
            p.grad = native_gradient[name].clone()
        return time.perf_counter() - t0
    batches = manifest_event["candidate_batches"][str(chosen)]["batches"]
    g_j = candidate_grad(rt, batches)
    for name, p in rt["named"]:
        p.grad = native_gradient[name] + _LAMBDA * (g_j[name] - native_gradient[name])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def train_arm(seed: int, arm: str, K: int, rt, manifest: dict, run_dir: Path,
              wc_budget_s=None, ghost_noop=False):
    """arm in {"A","B","C","A_wc","B_wc"}. Returns run summary dict.

    ghost_noop: engineering smoke only -- run the full C decision machinery at
    events but force j=0 execution (trajectory must stay bitwise = arm A).
    B_wc runs the fixed-j1 events under a wall budget (equal-wall-clock
    counterpart of B).
    """
    torch = rt["torch"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    total_micro = int(cfg["planned_total_optimizer_steps"]) * accum
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_events = "C" if arm == "C" else ("B" if arm in ("B", "B_wc") else None)

    batches = batches_from_train_pool(rt["train_indices"], int(cfg["batch_size"]),
                                      int(seed) + 1704, total_micro)
    events = manifest["cadence"][f"K{K}_control_steps"]
    event_set = set(map(int, events))
    eval_points = set(manifest["validation_identity"]["eval_points"])

    step_rows = []
    event_rows = []
    scaler = rt["scaler"]
    model = rt["model"]
    optimizer = rt["optimizer"]
    named = rt["named"]

    t_run0 = time.perf_counter()
    steps_done = 0
    preclip_norm = None
    for micro_idx in range(1, total_micro + 1):
        if wc_budget_s is not None and steps_done > 0 and \
                (time.perf_counter() - t_run0) >= wc_budget_s:
            break
        set_native_micro_lr(optimizer, rt["get_lr"], micro_idx, total_micro,
                            float(cfg["learning_rate"]))
        x, y = stack_batch(rt["dataset"], batches[micro_idx - 1], rt["device"])
        with rt["autocast_ctx"]:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()

        if micro_idx % accum == 0:
            opt_step = micro_idx // accum
            scaler.unscale_(optimizer)
            native_gradient = {name: p.grad.detach().clone() for name, p in named}

            inject_j = None
            if opt_step in event_set and arm_events is not None:
                key = f"{seed}/{opt_step}"
                mev = manifest["events"][key]
                eid = f"s{seed}_K{K}_{arm}_e{opt_step:03d}"
                edir = run_dir / "events" / eid
                t_evt0 = time.perf_counter()
                if arm_events == "C":
                    rec, snap = control_event_C(rt, mev, seed, opt_step, edir)
                    chosen = rec["chosen_action_id"]
                    if ghost_noop:
                        chosen = 0
                    rec["chosen_action_id"] = chosen
                    rec["ghost_noop"] = bool(ghost_noop)
                    t_exec = recompute_gj_and_inject(rt, mev, chosen, native_gradient)
                    rec["execution_time"] = t_exec
                    rec["controller_compute_time"] = rec["_pending"] + t_exec
                    rec["wall_clock_timestamp"] = time.time()
                    rec.update({"step": opt_step, "seed": seed, "control_id": eid,
                                "arm": arm, "K": K})
                    event_rows.append({k: v for k, v in rec.items() if k != "_pending"})
                    inject_j = chosen
                elif arm_events == "B":
                    rng_snap = capture_rng_state()
                    save_full_training_state(
                        edir / f"event_{seed}_{opt_step}.pt", model, optimizer, scaler,
                        metadata={"protocol": "A0K_CLOSED_LOOP_EVENT_SNAPSHOT",
                                  "train_seed": int(seed), "event_step": int(opt_step),
                                  "global_micro_step": micro_idx,
                                  "stream_seed": int(mev["stream_seed"])})
                    restore_rng_state(rng_snap)
                    t_exec = recompute_gj_and_inject(rt, mev, 1, native_gradient)
                    event_rows.append({
                        "step": opt_step, "seed": seed,
                        "control_id": edir.name, "arm": arm, "K": K,
                        "model_hash_before": model_hash(rt),
                        "optimizer_hash_before": opt_hash(rt),
                        "rng_state_hash": rng_hash(),
                        "dataloader_position": micro_idx,
                        "chosen_action_id": 1,
                        "hhat": None, "predicted_delta_loss": None,
                        "controller_compute_time": t_exec, "rollout_time": 0.0,
                        "selection_time": 0.0, "execution_time": t_exec,
                        "wall_clock_timestamp": time.time(),
                    })
                    inject_j = 1
            preclip_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])).detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps_done = opt_step

            if opt_step in eval_points:
                v_score, v_eval = eval_both(rt)
                step_rows.append({
                    "step": opt_step, "wall_s": time.perf_counter() - t_run0,
                    "v_score": v_score, "v_eval": v_eval,
                    "preclip_norm": preclip_norm,
                    "model_hash": model_hash(rt),
                })
            else:
                step_rows.append({
                    "step": opt_step, "wall_s": time.perf_counter() - t_run0,
                    "v_score": None, "v_eval": None,
                    "preclip_norm": preclip_norm,
                    "model_hash": model_hash(rt) if ((arm == "A" or ghost_noop) and opt_step in (46, 52, 58, 64, 70, 76, 82)) else None,
                })

    t_run1 = time.perf_counter()
    v_score, v_eval = eval_both(rt)
    step_rows.append({
        "step": steps_done, "wall_s": t_run1 - t_run0,
        "v_score": v_score, "v_eval": v_eval,
        "preclip_norm": preclip_norm,
        "model_hash": model_hash(rt),
    })
    summary = {
        "seed": seed, "arm": arm, "K": K,
        "steps_completed": steps_done,
        "wall_s": t_run1 - t_run0,
        "wc_budget_s": wc_budget_s,
        "final_v_score": v_score, "final_v_eval": v_eval,
        "n_control_events": len(event_rows),
        "trajectory_tail": model_hash(rt),
        "ghost_noop": bool(ghost_noop),
    }
    return summary, step_rows, event_rows
