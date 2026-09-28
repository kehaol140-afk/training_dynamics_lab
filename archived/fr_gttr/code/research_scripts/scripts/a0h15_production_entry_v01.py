"""A0H-1.5 independent production entry (round 2 amendment §4).

Thin adapter over the REAL frozen source primitives (fr.a0c tangent/adjoint steps,
a0g HVP/grad/val-loss helpers, a0h_adjoint walk). Does NOT call the official A0H-1
runner and does NOT write results through it. Mutually exclusive modes:

  forward_only  --hvp_precision {bf16,fp32} : baseline + candidate grads + future
      tangents (HVP per step when H>1) + endpoint V_score grad + scoring.
  adjoint_only  --hvp_precision fp32        : baseline + endpoint V_score grad +
      ONE shared reverse chain (w_0) + candidate grads + inner products.
      Never runs candidate future tangents; w_0 generated once per (state, H).
  replay_only   --backend native            : real +lambda action replay on V_score
      for M actions + no-op baseline scoring (A0G frozen semantics).

Also provides `time_matrix` (DEV_ENGINEERING_TIMING, amendment §5): interleaved
fixed-order repeats, full state restore per measurement, one-time preload reported
separately; and `selfcheck` (ENGINEERING_ELIGIBILITY rehearsal, amendment §4.5/4.6).

Call counters (per run): baseline_steps, candidate_grad_calls, hvp_calls,
reverse_adjoint_steps, clip_transpose_calls, endpoint_grad_calls, scoring_dot_calls,
replay_branches, replay_native_steps, replay_scoring_calls. Outer autograd call
counts are recorded but are NOT equal to mathematical derivative order or kernel work.
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics, sys, time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUTPUT_DIR = "outputs/a0h15_vjp_shared_v01"

_spec = importlib.util.spec_from_file_location(
    "a0h_adjoint_dev_check_v01",
    PROJECT_ROOT / "research" / "scripts" / "a0h_adjoint_dev_check_v01.py")
_a0h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a0h1)
FROZEN_LAMBDA = float(_a0h1.FROZEN_LAMBDA)

from fr.a0c_adamw_response_propagation_v01 import (  # noqa: E402
    adamw_tangent_step, clip_jvp, project_tangent)
from fr.full_state import load_full_training_state  # noqa: E402
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch  # noqa: E402
from fr.native_adamw_guard import assert_native_adamw  # noqa: E402
from fr.native_schedule import set_native_micro_lr  # noqa: E402
from fr.pilot_partition import load_partition  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.stream_rng import seed_future_stream  # noqa: E402

import a0g_decision_utility_v01 as a0g  # noqa: E402


# ---------------------------------------------------------------------------
# runtime construction (one-time; cost reported separately)
# ---------------------------------------------------------------------------
def build_runtime(device: str):
    cfg = json.loads((PROJECT_ROOT / "research/configs/e3c_confirmatory_v07.json").read_text())
    assert cfg.get("dtype") == "bfloat16" and int(cfg.get("accumulation_steps")) == 2 \
        and float(cfg.get("grad_clip")) == 1.0, "config drift"
    v07_out = PROJECT_ROOT / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    a0g_out = PROJECT_ROOT / a0g.OUTPUT_DIR
    data_group_definition = json.loads((a0g_out / "data_group_definition.json").read_text())
    validation_split = json.loads((a0g_out / "validation_split.json").read_text())
    minimind = PROJECT_ROOT / "minimind"
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
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=device)
    dataset = PretrainDataset(str(v07_out / "e3c_subset.jsonl"), tokenizer,
                              max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in named)
    assert int(cfg["num_hidden_layers"]) == 4 and int(cfg["hidden_size"]) == 256, \
        "model config mismatch vs frozen cfg"

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + a0g.EVAL_BATCH_SIZE] for i in range(0, len(idx), a0g.EVAL_BATCH_SIZE)]

    rt = {
        "torch": torch, "cfg": cfg, "model": model, "optimizer": optimizer, "scaler": scaler,
        "autocast_ctx": autocast_ctx, "named": named, "dataset": dataset,
        "val_score_tensors": None, "val_eval_tensors": None,
        "get_lr": get_lr, "timers": a0g._Timers(torch), "device": device,
        "n_trainable": n_trainable,
        "ctxs": a0g._load_v07_state_contexts(PROJECT_ROOT, cfg, partition, data_group_definition),
        "model_config": {"hidden_size": int(cfg["hidden_size"]),
                         "num_hidden_layers": int(cfg["num_hidden_layers"]),
                         "max_seq_len": int(cfg["max_seq_len"]),
                         "n_trainable_params": n_trainable},
    }
    rt["val_score_tensors"] = a0g._stack_all(dataset, _chunk(validation_split["v_score"]), device)
    rt["val_eval_tensors"] = a0g._stack_all(dataset, _chunk(validation_split["v_eval"]), device)
    return rt


def _new_counters() -> dict:
    return {"baseline_steps": 0, "candidate_grad_calls": 0, "hvp_calls": 0,
            "reverse_adjoint_steps": 0, "clip_transpose_calls": 0,
            "endpoint_grad_calls": 0, "scoring_dot_calls": 0,
            "replay_branches": 0, "replay_native_steps": 0, "replay_scoring_calls": 0,
            "outer_autograd_backward_calls": 0}


def _clone_map(m):
    return {k: v.detach().clone() for k, v in m.items()}


def _zeros_like(named):
    return a0g._zeros_like_named(named)


# ---------------------------------------------------------------------------
# native baseline step, mirroring the frozen pass (no probes, no tangent work)
# ---------------------------------------------------------------------------
def _forward_backward(rt, ctx, batches, horizon, *, want_snapshot):
    """Accumulate grads for one optimizer step up to post-unscale; returns grads,
    batch tensors, rng capture, lr — WITHOUT clip/step (caller decides)."""
    torch = rt["torch"]; model = rt["model"]; optimizer = rt["optimizer"]
    scaler = rt["scaler"]; autocast_ctx = rt["autocast_ctx"]
    cfg = rt["cfg"]; dataset = rt["dataset"]; named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    batch_tensors = []
    current_lr = None
    rng_before_native_forward = None
    for accumulation_index in range(accum):
        future_offset = (horizon - 1) * accum + accumulation_index + 1
        current_lr = set_native_micro_lr(
            optimizer, rt["get_lr"], int(ctx.bank_meta["global_micro_step"]) + future_offset,
            int(ctx.bank_meta["total_micro_steps"]), float(cfg["learning_rate"]))
        x, y = stack_batch(dataset, batches[(horizon - 1) * accum + accumulation_index],
                           rt["device"])
        batch_tensors.append((x, y))
        if horizon > 1 and accumulation_index == 0:
            rng_before_native_forward = capture_rng_state()
        with autocast_ctx:
            result = model(x, labels=y)
            loss = (result.loss + result.aux_loss) / accum
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    native_gradient = {name: p.grad.detach().clone() for name, p in named}
    snap = None
    if want_snapshot:
        pg0 = optimizer.param_groups[0]
        exp_avg = {p: optimizer.state[p]["exp_avg"].detach().clone() for _, p in named}
        exp_avg_sq = {p: optimizer.state[p]["exp_avg_sq"].detach().clone() for _, p in named}
        snap = {
            "exp_avg": {name: exp_avg[p] for name, p in named},
            "exp_avg_sq": {name: exp_avg_sq[p] for name, p in named},
            "lr": float(current_lr),
            # bias-correction step must be the OPTIMIZER internal step
            # (state["step"]+1), NOT the lr-schedule micro-step counter.
            "step_after": float(optimizer.state[named[0][1]]["step"]) + 1.0,
            "batch_tensors": batch_tensors,
            "rng_before_native_forward": rng_before_native_forward,
            "theta_start": {k: v.detach().clone() for k, v in raw_model(model).state_dict().items()},
            "weight_decay": float(pg0["weight_decay"]),
            "beta1": float(pg0["betas"][0]),
            "beta2": float(pg0["betas"][1]),
            "adam_eps": float(pg0["eps"]),
        }
    return native_gradient, batch_tensors, snap


def _clip_only(rt, native_gradient, snap=None):
    """clip_grad_norm_ on live grads ONLY (frozen order: tangents/HVP run after
    clip but BEFORE scaler.step so they see step-h weights and PRE-step m/v/step).
    Returns (preclip, clipped). Caller must later call _step_once."""
    torch = rt["torch"]; cfg = rt["cfg"]; named = rt["named"]
    preclip = float(torch.nn.utils.clip_grad_norm_(
        rt["model"].parameters(), float(cfg["grad_clip"])).detach().cpu())
    clipped = {name: p.grad.detach().clone() for name, p in named}
    if snap is not None:
        snap["preclip_norm"] = float(preclip)
        snap["clipped_gradient"] = clipped
        snap["native_gradient"] = native_gradient
    return preclip, clipped


def _step_once(rt):
    """Apply the native optimizer step (after any tangent/HVP work)."""
    rt["scaler"].step(rt["optimizer"])
    rt["scaler"].update()
    rt["optimizer"].zero_grad(set_to_none=True)


def _clip_and_step(rt, native_gradient, snap=None):
    """clip + immediate step (replay path only: no tangents between)."""
    preclip, clipped = _clip_only(rt, native_gradient, snap)
    _step_once(rt)
    return preclip, clipped


def _baseline_no_step(rt, ctx, batches, horizon, *, want_snapshot):
    """Frozen-order shared prefix: forward/backward + unscale (+PRE-step snapshot).
    Does NOT clip or step — caller interposes clip/tangents then _step_once."""
    native_gradient, batch_tensors, snap = _forward_backward(
        rt, ctx, batches, horizon, want_snapshot=want_snapshot)
    return native_gradient, batch_tensors, snap


def _candidate_grads(rt, cand):
    torch = rt["torch"]; cfg = rt["cfg"]; named = rt["named"]
    out = {}
    for j, cand_batches in cand.items():
        cand_tensors = a0g._stack_all(rt["dataset"], cand_batches, rt["device"])
        rng_snap = capture_rng_state()
        g_j = a0g._grad_on_batches(torch, rt["model"], named, cand_tensors,
                                   rt["autocast_ctx"], int(cfg["accumulation_steps"]))
        restore_rng_state(rng_snap)
        del cand_tensors
        out[j] = g_j
    return out


def _restore_and_seed(rt, ctx, stream):
    load_full_training_state(ctx.checkpoint_path, rt["model"], rt["optimizer"],
                             rt["scaler"], restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    seed_future_stream(int(stream["stream_seed"]))


def _candidates(ctx, stream, M):
    all_c = ctx.candidate_batches(stream["stream_seed"])
    keys = sorted(all_c, key=int)[: int(M)]
    return {j: all_c[j] for j in keys}


def _endpoint_grad(rt, counters):
    counters["endpoint_grad_calls"] += 1
    return a0g._grad_on_batches(rt["torch"], rt["model"], rt["named"],
                                rt["val_score_tensors"], rt["autocast_ctx"],
                                len(rt["val_score_tensors"]))


# ---------------------------------------------------------------------------
# mode: forward_only
# ---------------------------------------------------------------------------
def run_forward_only(rt, ctx, stream, H, M, hvp_fp32, counters):
    torch = rt["torch"]; cfg = rt["cfg"]; named = rt["named"]
    hvp_ctx = nullcontext() if hvp_fp32 else rt["autocast_ctx"]
    accum = int(cfg["accumulation_steps"])
    _restore_and_seed(rt, ctx, stream)
    cand = _candidates(ctx, stream, M)
    zero = _zeros_like(named)
    c_theta = {j: _clone_map(zero) for j in cand}
    c_m = {j: _clone_map(zero) for j in cand}
    c_v = {j: _clone_map(zero) for j in cand}
    a_by_j = {}
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = _baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=(horizon > 1))
        counters["baseline_steps"] += 1
        if horizon == 1:
            g_by_j = _candidate_grads(rt, cand)
            counters["candidate_grad_calls"] += len(cand)
            for j in cand:
                a_by_j[j] = {name: g_by_j[j][name] - native_gradient[name] for name, _ in named}
        native_clipped_gradient = _clip_only(rt, native_gradient, snap)[1]
        for j in cand:
            if horizon == 1:
                tC = a_by_j[j]
            else:
                # frozen discipline: HVP at step-h weights with the step-h
                # pre-forward rng state, leaving the live rng stream untouched.
                rng_after = capture_rng_state()
                restore_rng_state(snap["rng_before_native_forward"])
                tC, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"],
                                 c_theta[j], hvp_ctx, accum)
                restore_rng_state(rng_after)
                counters["hvp_calls"] += 1
            c_clip = clip_jvp(gradient_by_name=native_gradient,
                              tangent_gradient_by_name=tC,
                              max_norm=float(cfg["grad_clip"]))
            c_next = adamw_tangent_step(
                named_parameters=named, optimizer=rt["optimizer"],
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
                tangent_theta_by_name=c_theta[j], tangent_m_by_name=c_m[j],
                tangent_v_by_name=c_v[j])
            c_theta[j], c_m[j], c_v[j] = (c_next["tangent_theta"],
                                          c_next["tangent_m"], c_next["tangent_v"])
        _step_once(rt)  # native step only AFTER clip+tangents (frozen order)
    v_H = _endpoint_grad(rt, counters)
    scores = {}
    for j in cand:
        scores[str(j)] = float(project_tangent(v_H, c_theta[j]))
        counters["scoring_dot_calls"] += 1
    return {"mode": "forward_only", "hvp_fp32": bool(hvp_fp32), "H": int(H), "M": int(M),
            "scores_hhat_fwd": scores, "counters": counters,
            "final_model_hash": a0g.stable_hash(raw_model(rt["model"]).state_dict())}


# ---------------------------------------------------------------------------
# mode: adjoint_only (shared reverse chain; no candidate future tangents)
# ---------------------------------------------------------------------------
def run_adjoint_only(rt, ctx, stream, H, M, hvp_fp32, counters):
    torch = rt["torch"]; cfg = rt["cfg"]; named = rt["named"]
    hvp_ctx = nullcontext() if hvp_fp32 else rt["autocast_ctx"]
    accum = int(cfg["accumulation_steps"])
    assert hvp_fp32, "adjoint_only production candidate is fp32-hvp only this round"
    _restore_and_seed(rt, ctx, stream)
    cand = _candidates(ctx, stream, M)
    snapshots = {}
    a_by_j = {}
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = _baseline_no_step(
            rt, ctx, stream["batches"], horizon, want_snapshot=True)
        counters["baseline_steps"] += 1
        snapshots[horizon] = snap
        if horizon == 1:
            g_by_j = _candidate_grads(rt, cand)
            counters["candidate_grad_calls"] += len(cand)
            for j in cand:
                a_by_j[j] = {name: g_by_j[j][name] - native_gradient[name] for name, _ in named}
        _clip_only(rt, native_gradient, snap)
        _step_once(rt)  # snapshots hold PRE-step m/v (frozen discipline)
    v_H = _endpoint_grad(rt, counters)

    from fr.a0h_adjoint_v01 import adamw_tangent_step_adjoint

    def clip_transpose(b_map, snap):
        counters["clip_transpose_calls"] += 1
        return _a0h1._clip_transpose(
            native_gradient=snap["native_gradient"], b=b_map,
            grad_clip=float(cfg["grad_clip"]), preclip_norm=snap["preclip_norm"])

    H = int(H)
    p_th = _clone_map(v_H)
    p_m = _clone_map(_zeros_like(named))
    p_v = _clone_map(_zeros_like(named))
    p_store = {H: (_clone_map(p_th), _clone_map(p_m), _clone_map(p_v))}
    w_by_k = {}
    b_last = None
    for k in range(H, 0, -1):
        snap = snapshots[k]
        adj = adamw_tangent_step_adjoint(
            named_parameters=named, optimizer=rt["optimizer"], snapshot=snap,
            cotangent_theta_by_name=p_th, cotangent_m_by_name=p_m,
            cotangent_v_by_name=p_v, feedback_cotangent_by_name=None)
        counters["reverse_adjoint_steps"] += 1
        b_k = adj["cotangent_dc"]
        w_k = clip_transpose(b_k, snap)
        if k >= 2:
            w_by_k[k] = _clone_map(w_k)
        p_th = adj["cotangent_theta"]
        p_m = adj["cotangent_m"]
        p_v = adj["cotangent_v"]
        if k >= 2:
            rng_now = capture_rng_state()
            restore_rng_state(snap["rng_before_native_forward"])
            raw_model(rt["model"]).load_state_dict(snap["theta_start"])
            hvp_u, _ = a0g._hvp(torch, rt["model"], named, snap["batch_tensors"], w_k,
                                hvp_ctx, accum)
            counters["hvp_calls"] += 1
            restore_rng_state(rng_now)
            p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
            p_store[k - 1] = (_clone_map(p_th), _clone_map(p_m), _clone_map(p_v))
        b_last = b_k
    w_0 = clip_transpose(b_last, snapshots[1])
    scores = {}
    for j in cand:
        scores[str(j)] = float(project_tangent(w_0, a_by_j[j]))
        counters["scoring_dot_calls"] += 1
    return {"mode": "adjoint_only", "hvp_fp32": bool(hvp_fp32), "H": int(H), "M": int(M),
            "scores_hhat_adj": scores, "w_0_norm": float(a0g._norm(w_0)),
            "counters": counters,
            "final_model_hash": a0g.stable_hash(raw_model(rt["model"]).state_dict())}


# ---------------------------------------------------------------------------
# mode: replay_only (native +lambda action replay on V_score + no-op baseline)
# ---------------------------------------------------------------------------
def _replay_branch(rt, ctx, batches, stream_seed, H, cand_batches, action_j=None,
                   lam=FROZEN_LAMBDA):
    """One real replay branch: H native steps; at h=1 optionally inject
    g + lam*(g_j - g) pre-clip (A0G frozen action semantics). Returns V_score at H."""
    torch = rt["torch"]; cfg = rt["cfg"]; named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    _restore_and_seed_pre(rt, ctx, stream_seed)
    for horizon in range(1, int(H) + 1):
        native_gradient, _, snap = _forward_backward(rt, ctx, batches, horizon,
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
        _clip_and_step(rt, native_gradient, None)
    v = a0g._val_loss_only(torch, rt["model"], rt["val_score_tensors"], rt["autocast_ctx"])
    return float(v)


def _restore_and_seed_pre(rt, ctx, stream_seed):
    load_full_training_state(ctx.checkpoint_path, rt["model"], rt["optimizer"],
                             rt["scaler"], restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))


def run_replay_only(rt, ctx, stream, H, M, counters):
    torch = rt["torch"]
    all_cand = ctx.candidate_batches(stream["stream_seed"])
    cand = _candidates(ctx, stream, M)
    H = int(H)
    v_noop = _replay_branch(rt, ctx, stream["batches"], stream["stream_seed"], H, all_cand, None)
    counters["replay_branches"] += 1
    counters["replay_native_steps"] += H
    counters["replay_scoring_calls"] += 1
    results = {"0": {"v_score": v_noop, "gain_vs_noop": 0.0}}
    for j in cand:
        v = _replay_branch(rt, ctx, stream["batches"], stream["stream_seed"], H, all_cand, j)
        counters["replay_branches"] += 1
        counters["replay_native_steps"] += H
        counters["replay_scoring_calls"] += 1
        results[str(j)] = {"v_score": v, "gain_vs_noop": v_noop - v}
    gains = {j: r["gain_vs_noop"] for j, r in results.items()}
    best = max(gains, key=lambda j: (gains[j], -int(j)))
    choose_noop = gains[best] <= 0.0
    chosen_id = 0 if choose_noop else int(best)
    return {"mode": "replay_only", "backend": "native", "H": H, "M": int(M),
            "lambda": FROZEN_LAMBDA,
            "v_score": {j: r["v_score"] for j, r in results.items()},
            "gain_vs_noop": gains, "chosen_id": chosen_id,
            "choose_noop": bool(choose_noop), "counters": counters}


# ---------------------------------------------------------------------------
# timing harness (DEV_ENGINEERING_TIMING)
# ---------------------------------------------------------------------------
_METHODS = ("fp32_forward", "fp32_adjoint", "bf16_forward", "replay")


def _run_method(rt, ctx, stream, method, H, M):
    counters = _new_counters()
    if method == "fp32_forward":
        return run_forward_only(rt, ctx, stream, H, M, True, counters)
    if method == "bf16_forward":
        return run_forward_only(rt, ctx, stream, H, M, False, counters)
    if method == "fp32_adjoint":
        return run_adjoint_only(rt, ctx, stream, H, M, True, counters)
    if method == "replay":
        return run_replay_only(rt, ctx, stream, H, M, counters)
    raise ValueError(method)


def _measure(rt, fn):
    torch = rt["torch"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_ev.record()
    out = fn()
    end_ev.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {"wall_s": t1 - t0, "cuda_ms": start_ev.elapsed_time(end_ev),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "result": out}


def time_matrix(rt, ctx, stream, combos, repeats=3, warmup=1):
    torch = rt["torch"]
    records = []
    for _ in range(warmup):
        for (H, M) in combos:
            for m in _METHODS:
                _run_method(rt, ctx, stream, m, H, M)
    torch.cuda.synchronize()
    for rep in range(repeats):
        for (H, M) in combos:
            for m in _METHODS:  # pre-fixed interleaved order
                rec = _measure(rt, lambda: _run_method(rt, ctx, stream, m, H, M))
                records.append({"repeat": rep, "H": int(H), "M": int(M), "method": m,
                                "wall_s": rec["wall_s"], "cuda_ms": rec["cuda_ms"],
                                "peak_mem_mb": rec["peak_mem_mb"],
                                "scores": rec["result"].get("scores_hhat_fwd")
                                          or rec["result"].get("scores_hhat_adj")
                                          or rec["result"].get("gain_vs_noop"),
                                "counters": rec["result"]["counters"]})
    return records


def summarize(records):
    out = {}
    for r in records:
        key = (r["H"], r["M"], r["method"])
        out.setdefault(key, []).append(r)
    summary = []
    for (H, M, m), rs in sorted(out.items()):
        walls = [r["wall_s"] for r in rs]
        cuda = [r["cuda_ms"] / 1000.0 for r in rs]
        mem = [r["peak_mem_mb"] for r in rs]
        summary.append({"H": H, "M": M, "method": m, "n": len(rs),
                        "wall_median_s": statistics.median(walls),
                        "wall_min_s": min(walls), "wall_max_s": max(walls),
                        "cuda_median_s": statistics.median(cuda),
                        "peak_mem_max_mb": max(mem),
                        "counters_last": rs[-1]["counters"]})
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _ctx_by_label(rt, label):
    seed, step = label.replace("dev:", "").split("/")
    for c in rt["ctxs"]:
        if str(c.seed) == seed and str(c.step) == step:
            return c
    raise KeyError(label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("forward_only", "adjoint_only", "replay_only"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--states", required=True, help="dev:SEED/STEP")
        p.add_argument("--horizon", type=int, required=True)
        p.add_argument("--m", type=int, required=True)
        if name == "forward_only":
            p.add_argument("--hvp_precision", choices=("bf16", "fp32"), required=True)
        elif name == "adjoint_only":
            p.add_argument("--hvp_precision", choices=("fp32",), required=True)
        else:
            p.add_argument("--backend", choices=("native",), required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("time_matrix")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--states", required=True)
    p.add_argument("--combos", required=True, help="H:M comma list, e.g. 4:4,1:1")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", required=True)
    p = sub.add_parser("selfcheck")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--states", default="2031/52")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rt = build_runtime(args.device)
    preload_s = time.perf_counter() - t0

    if args.cmd == "time_matrix":
        ctx = _ctx_by_label(rt, args.states)
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        combos = [tuple(map(int, c.split(":"))) for c in args.combos.split(",")]
        recs = time_matrix(rt, ctx, stream, combos, repeats=args.repeats, warmup=args.warmup)
        summ = summarize(recs)
        payload = {"tag": "DEV_ENGINEERING_TIMING", "state": args.states,
                   "preload_one_time_s": preload_s,
                   "preload_note": "model init + dataset + val tensors + contexts; excluded from per-method timings",
                   "model_config": rt["model_config"], "combos": combos,
                   "repeats": args.repeats, "order": list(_METHODS),
                   "records": recs, "summary": summ}
    elif args.cmd == "selfcheck":
        ctx = _ctx_by_label(rt, f"dev:{args.states}")
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        fwd = run_forward_only(rt, ctx, stream, args.horizon, args.m, True, _new_counters())
        adj = run_adjoint_only(rt, ctx, stream, args.horizon, args.m, True, _new_counters())
        rep = run_replay_only(rt, ctx, stream, args.horizon, args.m, _new_counters())
        # model fingerprint check after a restore cycle
        load_full_training_state(ctx.checkpoint_path, rt["model"], rt["optimizer"],
                                 rt["scaler"], restore_rng=False)
        fp = a0g.stable_hash(raw_model(rt["model"]).state_dict())
        payload = {"tag": "ENGINEERING_SELFCHECK", "state": f"dev:{args.states}",
                   "preload_one_time_s": preload_s, "model_config": rt["model_config"],
                   "model_fingerprint_after_restore": fp,
                   "forward_fp32": fwd, "adjoint_fp32": adj, "replay_native": rep}
    else:
        ctx = _ctx_by_label(rt, args.states)
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        counters = _new_counters()
        if args.cmd == "forward_only":
            payload = run_forward_only(rt, ctx, stream, args.horizon, args.m,
                                       args.hvp_precision == "fp32", counters)
        elif args.cmd == "adjoint_only":
            payload = run_adjoint_only(rt, ctx, stream, args.horizon, args.m,
                                       args.hvp_precision == "fp32", counters)
        else:
            payload = run_replay_only(rt, ctx, stream, args.horizon, args.m, counters)
        payload["preload_one_time_s"] = preload_s
        payload["state"] = args.states
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
