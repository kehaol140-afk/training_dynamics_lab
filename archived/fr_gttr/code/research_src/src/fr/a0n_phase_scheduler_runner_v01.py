"""A0N response phase-aware scheduler runner (core module).

Pilot only (contract: outputs/a0n_phase_scheduler_pilot_v01/contract/contract.json,
frozen before any GPU). At every K=10 control event the production H=4 response
h_j is computed on a full-state+RNG snapshot (A0K machinery, unchanged); the
frozen phase metric rho = |mean(h_1..4)| / (std(h_1..4) + 1e-12) (population std,
same as the A0M diagnostic) selects the action:

  rho > 2            -> no-op (native AdamW step; injection of native clones)
  1 <= rho <= 2      -> inject g + 0.05*(g_j* - g), j* = argmin hhat_j, tie -> smallest id
  rho < 1            -> no-op

Nothing else changes vs the A0K frozen loop (optimizer/clipping/lambda/response
formula untouched). Arms A/B/A_wc/B_wc run via the untouched a0k train_arm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .a0k_closed_loop_runner_v01 import (
    control_event_C,
    eval_both,
    model_hash,
    recompute_gj_and_inject,
)
from .minimind_v03_runtime import stack_batch
from .native_schedule import set_native_micro_lr
from .stage_c_runtime import batches_from_train_pool

EPS = 1e-12
RHO_HIGH = 2.0   # rho > 2  -> high phase -> no-op
RHO_LOW = 1.0    # 1 <= rho <= 2 -> medium -> argmin inject; rho < 1 -> low -> no-op
J = (1, 2, 3, 4)


def rho_of(hhat: dict) -> float:
    hs = [float(hhat[j]) for j in J]
    mu = sum(hs) / len(hs)
    var = sum((x - mu) ** 2 for x in hs) / len(hs)          # population std (ddof=0)
    return abs(mu) / (var ** 0.5 + EPS)


def phase_policy(hhat: dict):
    """Frozen A0N policy. Returns (chosen, rho, phase).

    chosen = 0 no-op (high/low) or argmin_{j in 1..4} hhat_j (medium), tie ->
    smallest id (strict < scan).
    """
    rho = rho_of(hhat)
    if rho > RHO_HIGH:
        return 0, rho, "high"
    if rho >= RHO_LOW:
        best = min(J, key=lambda j: (float(hhat[j]), j))
        return int(best), rho, "medium"
    return 0, rho, "low"


def train_arm_C_phase(seed: int, K: int, rt, manifest: dict, run_dir: Path,
                      wc_budget_s=None, ghost_noop: bool = False):
    """Phase-aware scheduler arm. Returns (summary, step_rows, event_rows).

    ghost_noop: engineering smoke only -- full phase machinery (adjoint + rho)
    runs at every event but the action is forced to 0; trajectory must be
    bitwise identical to arm A.
    """
    torch = rt["torch"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    total_micro = int(cfg["planned_total_optimizer_steps"]) * accum
    run_dir.mkdir(parents=True, exist_ok=True)

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
            if opt_step in event_set:
                key = f"{seed}/{opt_step}"
                mev = manifest["events"][key]
                eid = f"s{seed}_K{K}_C_e{opt_step:03d}"
                edir = run_dir / "events" / eid
                rec, snap = control_event_C(rt, mev, seed, opt_step, edir)
                a0k_chosen = int(rec["chosen_action_id"])      # frozen A0K selector, audit only
                hhat = {int(j): float(v) for j, v in rec["hhat"].items()}
                chosen, rho, phase = phase_policy(hhat)
                if ghost_noop:
                    chosen = 0
                t_exec = recompute_gj_and_inject(rt, mev, chosen, native_gradient)
                rec.update({
                    "chosen_action_id": int(chosen),
                    "a0k_frozen_choice": a0k_chosen,
                    "rho": rho, "phase": phase,
                    "ghost_noop": bool(ghost_noop),
                    "execution_time": t_exec,
                    "controller_compute_time": rec["_pending"] + t_exec,
                    "wall_clock_timestamp": time.time(),
                    "step": opt_step, "seed": seed, "control_id": eid,
                    "arm": "C", "K": K,
                })
                event_rows.append({k: v for k, v in rec.items() if k != "_pending"})
                inject_j = chosen
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
                    "preclip_norm": preclip_norm, "model_hash": None,
                })

    t_run1 = time.perf_counter()
    v_score, v_eval = eval_both(rt)
    step_rows.append({
        "step": steps_done, "wall_s": t_run1 - t_run0,
        "v_score": v_score, "v_eval": v_eval,
        "preclip_norm": preclip_norm,
        "model_hash": model_hash(rt),
    })
    phases = [e["phase"] for e in event_rows]
    summary = {
        "seed": seed, "arm": "C", "K": K,
        "steps_completed": steps_done,
        "wall_s": t_run1 - t_run0,
        "wc_budget_s": wc_budget_s,
        "final_v_score": v_score, "final_v_eval": v_eval,
        "n_control_events": len(event_rows),
        "phase_occupancy": {p: phases.count(p) for p in ("high", "medium", "low")},
        "n_injections": int(sum(1 for e in event_rows if e["chosen_action_id"] != 0)),
        "trajectory_tail": model_hash(rt),
        "ghost_noop": bool(ghost_noop),
    }
    return summary, step_rows, event_rows
