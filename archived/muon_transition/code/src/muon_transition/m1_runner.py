"""Single-cell M1 training runner (frozen: protocol v1.1 + v1.2, configs/m1_baseline_prereg.json).

Loop per optimizer step: 4 micro-batches (zero_grad once, loss/accum backward), global-norm clip
1.0, step all optimizers, LR = reference * multiplier * schedule_factor(t). bf16 autocast on GPU.
Hooks every 100 steps + final step (diagnostic only). Final token NLL on frozen dev pools.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from . import config
from .checkpoint import save_training_state
from .data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream, sha256_file
from .eval_nll import evaluate_q
from .model import build_model, unique_named_parameters
from .observation_hooks import ObservationHook
from .optimizers import build_adamw_arm, build_muon_arm
from .partitions import build_pair_aligned_split
from .train_loop import batch_token_loss


def schedule_factor(t: int) -> float:
    if t <= config.WARMUP_STEPS:
        return t / config.WARMUP_STEPS
    span = config.B0_OPTIMIZER_STEPS - config.WARMUP_STEPS
    w = (t - config.WARMUP_STEPS) / span
    return config.COSINE_MIN_FRAC + 0.45 * (1.0 + math.cos(math.pi * w))


def prepare_data(verbose: bool = False):
    """Load deduped+split+packed domains and eval pools. Deterministic, matches M0 artifacts."""
    from .m1_data_cache import get_domain_windows

    a_train, b_train = get_domain_windows()
    pools = torch.load(config.M0_OUTPUTS / "data_audit" / "eval_pools.pt", weights_only=True)
    dev_pools = {"A": pools["A_dev"], "B": pools["B_dev"]}
    return a_train, b_train, dev_pools, pools


def probe_geometry(model, dev_pools, device):
    """Frozen probe: first 4 batches x 8 windows per domain from dev pools; 16 pairwise dots."""
    from .observation_hooks import _flat_grads

    grads = {"A": [], "B": []}
    for dom in ("A", "B"):
        for b in range(config.HOOK_PROBE_BATCHES_PER_DOMAIN):
            batch = dev_pools[dom][b * config.HOOK_PROBE_BATCH_SEQUENCES : (b + 1) * config.HOOK_PROBE_BATCH_SEQUENCES].to(device)
            loss = batch_token_loss(model, batch)
            loss.backward()
            grads[dom].append(_flat_grads([p for _, p in unique_named_parameters(model)]).cpu())
            model.zero_grad(set_to_none=True)
    dots, coses = [], []
    for ga in grads["A"]:
        for gb in grads["B"]:
            dots.append(float(ga @ gb))
            coses.append(float(torch.nn.functional.cosine_similarity(ga, gb, dim=0)))
    return sum(coses) / len(coses), sum(1 for d in dots if d < 0) / len(dots)


def flat_training_grads(model, device):
    gs = [p.grad.detach().reshape(-1).double() for _, p in unique_named_parameters(model)]
    return torch.cat(gs).cpu()


def run_cell(
    arm: str,
    lr_multiplier: float,
    weight_decay: float,
    p: float,
    seed: int,
    run_id: str,
    out_dir: Path,
    n_steps: int = config.B0_OPTIMIZER_STEPS,
    device: str = "cuda",
    verbose: bool = True,
    data=None,
    save_state: bool = True,
):
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "obs.jsonl"
    if obs_path.exists():
        obs_path.unlink()

    if data is None:
        data = prepare_data()
    a_train, b_train, dev_pools, _ = data

    from .mixture import MixtureStream, build_step_manifest

    manifest = build_step_manifest(a_train, b_train, p, n_steps, seed=config.BASE_SEED_M1_M2)
    stream = MixtureStream(manifest, a_train, b_train)

    model = build_model(seed=seed).to(device)
    ref_lr = config.ADAMW_REF_LR if arm == "adamw" else config.MUON_REF_LR
    lr = ref_lr * lr_multiplier
    if arm == "adamw":
        optimizers = [build_adamw_arm(model, lr=lr, weight_decay=weight_decay)]
    else:
        mu, aux = build_muon_arm(model, lr=lr, weight_decay=weight_decay)
        optimizers = [mu, aux]

    hook = ObservationHook(obs_path, every_n=config.HOOK_EVERY_N_STEPS, name=run_id)
    metrics_path = out_dir / "metrics.jsonl"
    hook.log_final = n_steps

    run_cfg = {
        "run_id": run_id, "stage": "M1", "arm": arm, "p": p, "seed": seed,
        "lr_multiplier": lr_multiplier, "lr": lr, "reference_lr": ref_lr,
        "weight_decay": weight_decay, "n_steps": n_steps,
        "tokens": n_steps * config.TOKENS_PER_OPTIMIZER_STEP,
        "manifest_p_hat": manifest["realized_p_hat"],
        "schedule": {"warmup_steps": config.WARMUP_STEPS, "cosine_min_frac": config.COSINE_MIN_FRAC},
        "grad_clip": config.GRAD_CLIP_NORM, "device": device,
    }
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast(device_type="cpu", enabled=False)

    t0 = time.time()
    losses = []
    nan_hit = False
    for step in range(1, n_steps + 1):
        sched = schedule_factor(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = lr * sched

        ctx = hook.before_step(model) if (hook.should_log(step) or step == n_steps) else None

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        micro_losses = []
        for batch in stream.micro_batches(step - 1):
            batch = batch.to(device)
            with autocast:
                loss = batch_token_loss(model, batch)
            (loss / config.GRAD_ACCUM_STEPS).backward()
            micro_losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], config.GRAD_CLIP_NORM
        )
        if not torch.isfinite(grad_norm):
            nan_hit = True
        g_flat = None
        if ctx is not None:
            g_flat = flat_training_grads(model, device)
        for opt in optimizers:
            opt.step()
        losses.append(sum(micro_losses) / len(micro_losses))

        if ctx is not None:
            was_training = model.training
            model.eval()
            dom_cos, conflict = probe_geometry(model, dev_pools, device)
            if was_training:
                model.train()
            rec = hook.after_step(
                model, ctx, step, grad_norm=float(grad_norm),
                domain_cosine=dom_cos, conflict_ratio=conflict, grad_flat=g_flat,
                extra={"train_loss": losses[-1], "lr": lr * sched, "final": step == n_steps},
            )
            assert rec is not None

        if step % 100 == 0 or step <= 5 or step == n_steps:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": losses[-1],
                                    "grad_norm": float(grad_norm), "lr": lr * sched}) + "\n")
            if verbose:
                dt = time.time() - t0
                print(f"[{run_id}] step {step}/{n_steps} loss={losses[-1]:.4f} "
                      f"({dt:.0f}s, {step/dt:.2f} it/s)", flush=True)

    wall = time.time() - t0
    model.eval()
    q = evaluate_q(model, dev_pools, device=device)

    opt_state_ok = True
    for opt in optimizers:
        n_state = len(opt.state)
        n_params = sum(len(g["params"]) for g in opt.param_groups)
        if n_state != n_params:
            opt_state_ok = False
    muon_bufs = 0
    if arm == "muon":
        muon_bufs = sum(1 for s in optimizers[0].state.values() if "momentum_buffer" in s)

    result = {
        **run_cfg,
        "wall_seconds": wall,
        "final_dev": q,
        "final_train_loss": losses[-1],
        "train_loss_mean_first20": sum(losses[:20]) / 20,
        "train_loss_mean_last20": sum(losses[-20:]) / 20,
        "loss_decreased": sum(losses[-20:]) / 20 < sum(losses[:20]) / 20,
        "any_nonfinite_gradnorm": nan_hit,
        "any_nan_loss": any(math.isnan(l) or math.isinf(l) for l in losses),
        "optimizer_state_entries": sum(len(o.state) for o in optimizers),
        "optimizer_state_ok": opt_state_ok,
        "muon_momentum_buffers": muon_bufs,
        "n_loss_points": len(losses),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    if save_state:
        save_training_state(
            out_dir / "final_state.pt", model, optimizers,
            cursor={"step": n_steps, "run_id": run_id},
            meta={"stage": "M1", "arm": arm, "lr_multiplier": lr_multiplier, "wd": weight_decay},
        )
    if verbose:
        print(f"[{run_id}] DONE in {wall:.0f}s final V_Q={q['v_q']:.4f} "
              f"NLL_A={q['nll_A']:.4f} NLL_B={q['nll_B']:.4f}", flush=True)
    return result
