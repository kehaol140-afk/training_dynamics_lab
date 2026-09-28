"""A0L response-guided continuous mixture runner (core module).

Arm A: native AdamW (original distribution = full train pool).
Arm B: fixed-j1 continuous mixture (all micro-batches from group 1).
Arm C: response mixture controller - every K=50 steps a production H=4
response on a full-state+RNG snapshot yields h_j; q = softmax(-gamma_t h)
with gamma_t = 1/(std(h)+eps); pi <- (1-alpha) pi + alpha q; snapshot
restored, pi held for the next window. Treatment = data composition only;
NO gradient injection, optimizer/clipping/H/lambda untouched.

Frozen by outputs/a0l_response_mixture_feasibility_v01/contract/contract.json;
events frozen in contract/a0l_events_manifest.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .a0k_closed_loop_runner_v01 import (
    build_live_runtime,
    eval_both,
    make_event_ctx,
    model_hash,
    opt_hash,
    rng_hash,
)
from .minimind_v03_runtime import stack_batch
from .native_schedule import set_native_micro_lr
from .full_state import load_full_training_state, save_full_training_state
from .rng import capture_rng_state, restore_rng_state
from .stage_c_runtime import batches_from_train_pool
from .stream_rng import derive_stream_seed

ALPHA = 0.1
K_STEPS = 50
CONTROL_STEPS = (50, 100)
EPS = 1e-12


def _pi0(manifest) -> list:
    return [float(x) for x in manifest["pi_0_empirical"]]


def _window_of_step(opt_step: int) -> int:
    if opt_step < CONTROL_STEPS[0]:
        return 0
    if opt_step < CONTROL_STEPS[1]:
        return 1
    return 2


def _arm_batches(rt, arm: str, seed: int, n_micro: int):
    """Arm A/A_wc: native full-pool stream. Arm B/B_wc: fixed group 1."""
    from a0g_decision_utility_v01 import candidate_batches_for
    cfg = rt["cfg"]
    bs = int(cfg["batch_size"])
    if arm in ("A", "A_wc"):
        return batches_from_train_pool(rt["train_indices"], bs,
                                       int(seed) + 1704, n_micro)
    # fixed-j1 = controller component j=1 = members["0"] (1038 samples,
    # the first/largest group; members are keyed 0-based, controller j is 1-based)
    return candidate_batches_for(int(seed) + 1704, 1,
                                 rt["group_definition"]["members"]["0"], bs, n_micro)


def _mixture_window_batches(rt, seed: int, pi, gen, offset: int, n_micro: int):
    """Arm C window batches: global micro indices offset..offset+n-1;
    component ~ pi from the window's dedicated generator; batch from the
    frozen candidate recipe keyed by the GLOBAL micro index."""
    from a0g_decision_utility_v01 import candidate_batches_for
    bs = int(rt["cfg"]["batch_size"])
    pi_t = torch.tensor(pi, dtype=torch.float64)
    out = []
    for k in range(n_micro):
        m = offset + k
        j = int(torch.multinomial(pi_t, 1, generator=gen).item())
        b = candidate_batches_for(derive_stream_seed(int(seed) + 1704, m), j,
                                  rt["group_definition"]["members"][str(j)], bs, 1)
        out.append(b[0])
    return out


def _window_generator(seed: int, window: int):
    g = torch.Generator()
    g.manual_seed(derive_stream_seed(int(seed) + 1704, 9090 + int(window)))
    return g


def response_event_C(rt, manifest_event, pi_before, seed, step, event_dir):
    """Snapshot -> production H=4 response -> restore. Returns (rec, snap).

    No action selection: the 'action' is the pi update done by the caller
    with the returned hhat. RNG-neutral by capture/restore.
    """
    import a0h15_production_entry_v01 as pe
    t0 = time.perf_counter()
    event_dir.mkdir(parents=True, exist_ok=True)
    snap_path = event_dir / f"event_{seed}_{step}.pt"
    rng_live = capture_rng_state()
    save_full_training_state(
        snap_path, rt["model"], rt["optimizer"], rt["scaler"],
        metadata={"protocol": "A0L_MIXTURE_EVENT_SNAPSHOT",
                  "train_seed": int(seed), "event_step": int(step),
                  "global_micro_step": int(step) * int(rt["cfg"]["accumulation_steps"]),
                  "stream_seed": int(manifest_event["stream_seed"]),
                  "pi_before": [float(x) for x in pi_before]})
    rec = {
        "model_hash_before": model_hash(rt),
        "optimizer_hash_before": opt_hash(rt),
        "rng_state_hash": rng_hash(),
        "dataloader_position": int(step) * int(rt["cfg"]["accumulation_steps"]),
        "stream_seed": int(manifest_event["stream_seed"]),
    }
    ctx = make_event_ctx(rt, manifest_event, seed, step, snap_path)
    stream = {"stream_id": 0, "stream_seed": int(manifest_event["stream_seed"]),
              "batches": manifest_event["lookahead_batches"]}
    t_dec0 = time.perf_counter()
    counters = pe._new_counters()
    adj = pe.run_adjoint_only(rt, ctx, stream, 4, 4, True, counters)
    t_dec1 = time.perf_counter()
    load_full_training_state(snap_path, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    restore_rng_state(rng_live)
    hhat = {int(j): float(v) for j, v in adj["scores_hhat_adj"].items()}
    rec.update({
        "hhat": {str(j): hhat.get(j, 0.0) for j in range(1, 5)},
        "rollout_time": t_dec1 - t_dec0,
        "controller_compute_time": time.perf_counter() - t0,
        "wall_clock_timestamp": time.time(),
    })
    return rec, snap_path


def _q_from_h(hhat: dict) -> tuple:
    hs = torch.tensor([hhat[j] for j in range(1, 5)], dtype=torch.float64)
    std = float(hs.std(unbiased=False))
    if std < EPS:
        q = torch.full((4,), 0.25, dtype=torch.float64)
    else:
        q = torch.softmax(-(hs / std), dim=0)
    return [float(x) for x in q], std


def train_arm(arm: str, seed: int, rt, manifest: dict, run_dir: Path,
              wc_budget_s=None):
    """arm in {A,B,C,A_wc,B_wc}. Returns (summary, step_rows, event_rows)."""
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    total_micro = int(cfg["planned_total_optimizer_steps"]) * accum
    run_dir.mkdir(parents=True, exist_ok=True)

    torch_ = rt["torch"]
    scaler = rt["scaler"]
    model = rt["model"]
    optimizer = rt["optimizer"]

    pi_windows = [_pi0(manifest), None, None]
    windows = [{"window": 0, "step_start": 1, "step_end": 49,
                "pi": list(pi_windows[0])}]
    accum = int(cfg["accumulation_steps"])
    if arm == "C":
        # window 0 = micros 0..97 (steps 1..49); later windows extend lazily
        batches = _mixture_window_batches(rt, seed, pi_windows[0],
                                          _window_generator(seed, 0), 0, 49 * accum)
    else:
        batches = _arm_batches(rt, arm, seed, total_micro)

    step_rows = []
    event_rows = []
    pi_traj = [{"step": 0, "pi": list(pi_windows[0]), "source": "pi_0_empirical"}]
    t_run0 = time.perf_counter()
    steps_done = 0
    micro_loss_acc = []
    window_losses = []

    for micro_idx in range(1, total_micro + 1):
        opt_step = (micro_idx - 1) // accum + 1
        if wc_budget_s is not None and steps_done > 0 and \
                (time.perf_counter() - t_run0) >= wc_budget_s:
            break
        # window switch at step boundaries 50 / 100
        if arm == "C" and opt_step in CONTROL_STEPS and \
                (micro_idx - 1) % accum == 0:
            w = _window_of_step(opt_step)
            mev = manifest["events"][f"{seed}/{opt_step}"]
            eid = f"s{seed}_K{K_STEPS}_{arm}_e{opt_step:03d}"
            pi_before = list(pi_windows[w - 1])
            rec, snap = response_event_C(rt, mev, pi_before, seed, opt_step,
                                         run_dir / "events" / eid)
            q, std_h = _q_from_h({int(j): rec["hhat"][str(j)] for j in range(1, 5)})
            pi_after = [(1 - ALPHA) * p + ALPHA * qq for p, qq in zip(pi_before, q)]
            pi_windows[w] = pi_after
            n_new = 50 * accum if w == 1 else (128 - 99) * accum
            offset = 98 + (w - 1) * 100
            batches.extend(_mixture_window_batches(
                rt, seed, pi_after, _window_generator(seed, w), offset, n_new))
            v_score, v_eval = eval_both(rt)
            rec.update({
                "step": opt_step, "seed": seed, "arm": arm,
                "window_index": w, "control_id": eid,
                "pi_before": pi_before, "q_j": q, "pi_after": pi_after,
                "gamma_t": 1.0 / std_h if std_h >= EPS else None, "std_h": std_h,
                "window_train_loss_mean":
                    sum(window_losses) / len(window_losses) if window_losses else None,
                "v_eval_at_event": v_eval, "v_score_at_event": v_score,
            })
            event_rows.append(rec)
            pi_traj.append({"step": opt_step, "pi": list(pi_after),
                            "hhat": dict(rec["hhat"]), "q": q, "source": "response"})
            windows.append({"window": w,
                            "step_start": opt_step,
                            "step_end": opt_step + 49 if opt_step + 49 <= 128 else 128,
                            "pi": list(pi_after)})
            window_losses = []

        if micro_idx > len(batches):
            raise RuntimeError(f"batch stream exhausted at micro {micro_idx}")
        set_native_micro_lr(optimizer, rt["get_lr"], micro_idx, total_micro,
                            float(cfg["learning_rate"]))
        x, y = stack_batch(rt["dataset"], batches[micro_idx - 1], rt["device"])
        with rt["autocast_ctx"]:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        micro_loss_acc.append(float(res.loss.detach().cpu()))
        window_losses.append(float(res.loss.detach().cpu()))
        scaler.scale(loss).backward()

        if micro_idx % accum == 0:
            scaler.unscale_(optimizer)
            preclip = float(torch_.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])).detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps_done = opt_step
            if opt_step % 10 == 0 or opt_step == 128:
                v_score, v_eval = eval_both(rt)
                step_rows.append({"step": opt_step,
                                  "wall_s": time.perf_counter() - t_run0,
                                  "train_loss": sum(micro_loss_acc) / len(micro_loss_acc),
                                  "v_score": v_score, "v_eval": v_eval,
                                  "preclip_norm": preclip,
                                  "window": _window_of_step(opt_step)})
            else:
                step_rows.append({"step": opt_step,
                                  "wall_s": time.perf_counter() - t_run0,
                                  "train_loss": sum(micro_loss_acc) / len(micro_loss_acc),
                                  "v_score": None, "v_eval": None,
                                  "preclip_norm": preclip,
                                  "window": _window_of_step(opt_step),
                                  "model_hash": model_hash(rt) if (arm == "A" and opt_step in (46, 52)) else None})
            micro_loss_acc = []

    t_run1 = time.perf_counter()
    v_score, v_eval = eval_both(rt)
    step_rows.append({"step": steps_done, "wall_s": t_run1 - t_run0,
                      "train_loss": sum(micro_loss_acc) / len(micro_loss_acc) if micro_loss_acc else None,
                      "v_score": v_score, "v_eval": v_eval, "preclip_norm": None,
                      "window": _window_of_step(steps_done),
                      "model_hash": model_hash(rt)})
    summary = {
        "seed": seed, "arm": arm, "K": K_STEPS,
        "steps_completed": steps_done,
        "wall_s": t_run1 - t_run0,
        "wc_budget_s": wc_budget_s,
        "final_v_score": v_score, "final_v_eval": v_eval,
        "n_control_events": len(event_rows),
        "trajectory_tail": model_hash(rt),
        "pi_traj": pi_traj, "windows": windows,
    }
    return summary, step_rows, event_rows
