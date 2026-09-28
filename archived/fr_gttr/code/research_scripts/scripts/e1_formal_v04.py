#!/usr/bin/env python3
"""v0.4 协议对齐 E1：R=16、H=[1,8,32]、FULL_MODEL_Q。

LEGACY_PILOT_DO_NOT_USE_FOR_FORMAL_RESULT

输出：
- s_r / sigma_q / 分布形状；
- first-step clipping；
- X_theta；
- fixed eval mean / p95 / p99 / CVaR99；
- Y_mean / Y_tail；
- B_theta / B_mean / B_tail。

本脚本严格不施加 ±epsilon。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

LEGACY_PILOT_STATUS = "LEGACY_PILOT_DO_NOT_USE_FOR_FORMAL_RESULT"

from fr.batch_manifest import save_manifest
from fr.direction import (
    normalize_direction, direction_norm, save_direction,
    project_current_gradients, parameter_displacement_projection,
)
from fr.eval_observables import evaluate_token_losses
from fr.full_state import load_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw
from fr.pilot_partition import load_partition
from fr.response_stats import distribution_summary, observational_slope
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed, seed_future_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/stage_c_v04.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    out = root / "outputs/stage_c_v04"
    state_path = out / "mid_state_v04.pt"
    part_path = out / "partition_v04.json"
    if not state_path.exists() or not part_path.exists():
        raise SystemExit("缺少 MID checkpoint/partition，请先运行 prepare_mid_checkpoint_v04.py")

    partition = load_partition(part_path)

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=args.device,
    )
    ds = PretrainDataset(str(out / "stage_c_subset.jsonl"), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)

    meta = load_full_training_state(state_path, model, optimizer, scaler)
    if meta.get("partition_sha256") != partition["sha256"]:
        raise RuntimeError("MID state 与 partition 不匹配")

    # reference theta 保存在完整 state 内，用于 X_theta。
    state_payload = torch.load(state_path, map_location="cpu", weights_only=False)
    reference_theta = state_payload["model"]

    # ------------------------------------------------------------
    # 1. 构造 FULL_MODEL_Q：只用 train-probe。
    # ------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    probe_idx = partition["probe_indices"]
    x_probe, y_probe = stack_batch(ds, probe_idx, args.device)
    with autocast_ctx:
        probe_res = model(x_probe, labels=y_probe)
        probe_loss = probe_res.loss + probe_res.aux_loss

    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
    q_raw = {
        name: (torch.zeros_like(p.detach()) if g is None else g.detach().clone())
        for (name, p), g in zip(named, grads)
    }
    q, probe_grad_norm = normalize_direction(q_raw)
    q_norm = direction_norm(q)
    q_path = out / "full_model_q_v04.pt"
    q_sha = save_direction(
        q_path, q,
        metadata={
            "protocol": "E1_V04",
            "partition_sha256": partition["sha256"],
            "probe_indices": probe_idx,
            "probe_loss": float(probe_loss.detach().float().cpu()),
            "probe_grad_norm_before_normalize": probe_grad_norm,
            "q_norm": q_norm,
        },
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()

    # ------------------------------------------------------------
    # 2. 固定 eval-probe 在 t 时刻的基准。
    # ------------------------------------------------------------
    ref_eval = evaluate_token_losses(
        model, ds, partition["eval_indices"],
        int(cfg["eval_batch_size"]), args.device, autocast_ctx,
    )

    R = int(cfg["e1_streams"])
    horizons = sorted(map(int, cfg["horizons"]))
    max_h = max(horizons)
    accum = int(cfg["accumulation_steps"])
    mid_micro = int(meta["global_micro_step"])
    total_micro = int(meta["total_micro_steps"])

    stream_rows = []
    s_values = []

    for r in range(R):
        load_full_training_state(state_path, model, optimizer, scaler)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        stream_seed = derive_stream_seed(int(cfg["stream_base_seed_e1"]), r)
        seed_future_stream(stream_seed)
        batch_seed = derive_stream_seed(stream_seed, 404)
        batches = batches_from_train_pool(
            partition["train_indices"],
            int(cfg["batch_size"]),
            batch_seed,
            max_h * accum,
        )
        manifest_sha = save_manifest(
            out / f"e1_stream_{r:03d}_batches.json",
            batches,
            metadata={"stream_id": r, "stream_seed": stream_seed, "batch_seed": batch_seed},
        )

        first_s = None
        first_preclip_norm = None
        first_clipped = None
        observations = {}

        for h in range(1, max_h + 1):
            # 一个 optimizer step = accum 个 micro-batch
            for a in range(accum):
                batch_idx = batches[(h - 1) * accum + a]
                x, y = stack_batch(ds, batch_idx, args.device)
                with autocast_ctx:
                    res = model(x, labels=y)
                    loss = (res.loss + res.aux_loss) / accum
                scaler.scale(loss).backward()

            # E1 在第一个 optimizer boundary 观察自然 raw gradient。
            if h == 1:
                scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                obs = project_current_gradients(named, q, grad_scale=scale)
                first_s = float(obs["projection"])
                s_values.append(first_s)

            # [FR-NATIVE-ADAMW] E1 完全 native，无 perturbation。
            scaler.unscale_(optimizer)
            preclip = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])
            ).detach().cpu())
            if h == 1:
                first_preclip_norm = preclip
                first_clipped = bool(preclip > float(cfg["grad_clip"]))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # MiniMind 原生是按 micro step 计算 LR；下一 optimizer step 前写入对应 LR。
            next_global_micro = mid_micro + h * accum
            if h < max_h:
                next_lr = float(get_lr(
                    next_global_micro + 1, total_micro, float(cfg["learning_rate"])
                ))
                for g in optimizer.param_groups:
                    g["lr"] = next_lr

            if h in horizons:
                x_theta = parameter_displacement_projection(model, reference_theta, q)
                ev = evaluate_token_losses(
                    model, ds, partition["eval_indices"],
                    int(cfg["eval_batch_size"]), args.device, autocast_ctx,
                )
                observations[str(h)] = {
                    "X_theta": x_theta,
                    "eval": ev,
                    "Y_mean": float(ref_eval["mean"] - ev["mean"]),
                    "Y_tail": float(ref_eval["cvar99"] - ev["cvar99"]),
                }

        stream_rows.append({
            "stream_id": r,
            "stream_seed": stream_seed,
            "batch_seed": batch_seed,
            "manifest_sha256": manifest_sha,
            "s": first_s,
            "first_preclip_grad_norm": first_preclip_norm,
            "first_clipped": first_clipped,
            "observations": observations,
        })

    dist = distribution_summary(s_values)
    sigma_q = float(dist["std"])
    clipping_rate = sum(bool(x["first_clipped"]) for x in stream_rows) / R

    B = {}
    for h in horizons:
        key = str(h)
        B[key] = {
            "theta": observational_slope(s_values, [x["observations"][key]["X_theta"] for x in stream_rows]),
            "mean": observational_slope(s_values, [x["observations"][key]["Y_mean"] for x in stream_rows]),
            "tail": observational_slope(s_values, [x["observations"][key]["Y_tail"] for x in stream_rows]),
        }

    eps = {f"rho_{rho:g}": float(rho * sigma_q) for rho in map(float, cfg["rho"])}
    technical_pass = bool(
        R == 16 and dist["std"] > 0 and dist["distinct_count"] >= 2
        and abs(q_norm - 1.0) < 1e-5
    )

    report = {
        "protocol": "E1_V04_PROTOCOL_ALIGNED_PILOT",
        "technical_pass": technical_pass,
        "scientific_intervention_enabled": False,
        "R": R,
        "horizons": horizons,
        "partition_sha256": partition["sha256"],
        "q": {
            "path": str(q_path),
            "sha256": q_sha,
            "norm": q_norm,
            "probe_grad_norm_before_normalize": probe_grad_norm,
        },
        "reference_eval": ref_eval,
        "s_distribution": dist,
        "sigma_q": sigma_q,
        "clipping_rate_first_step": clipping_rate,
        "epsilon_candidates": eps,
        "B": B,
        "streams": stream_rows,
        "boundaries": [
            "这是 1 seed × MID Stage-C pilot，不是跨 seed 证据。",
            "B 是 observational slope，不是 causal susceptibility。",
            "E1 不含人工扰动。",
        ],
    }
    (out / "e1_formal_v04_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# E1 v0.4 协议对齐 Pilot", "",
        f"- 技术判定：**{'PASS' if technical_pass else 'FAIL'}**",
        f"- R：`{R}`",
        f"- q norm：`{q_norm:.12g}`",
        f"- sigma_q：`{sigma_q:.12g}`",
        f"- mean_s：`{dist['mean']:.12g}`",
        f"- skewness：`{dist['skewness']:.12g}`",
        f"- excess kurtosis：`{dist['excess_kurtosis']:.12g}`",
        f"- MAD：`{dist['mad']:.12g}`",
        f"- first-step clipping rate：`{clipping_rate:.6f}`",
        "", "## epsilon 候选（E1 只计算，不执行）", ""
    ]
    for k, v in eps.items():
        lines.append(f"- {k}: `{v:.12g}`")
    lines += ["", "## B(theta/mean/tail)", ""]
    for h in horizons:
        x = B[str(h)]
        lines += [
            f"### H={h}",
            f"- B_theta: `{x['theta']['B']:.12g}`",
            f"- B_mean: `{x['mean']['B']:.12g}`",
            f"- B_tail: `{x['tail']['B']:.12g}`",
            ""
        ]
    (out / "e1_formal_v04_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("E1_V04_FORMAL_PASS" if technical_pass else "E1_V04_FORMAL_FAIL")
    print("R=", R)
    print("sigma_q=", sigma_q)
    print("clipping_rate_first_step=", clipping_rate)
    print("epsilon_candidates=", eps)
    print("report=", out / "e1_formal_v04_report.md")
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
