#!/usr/bin/env python3
"""E1 自然涨落 pilot：测量同一 Z_t 上不同随机流的 <q, g_t^(r)>。

核心科学对象
------------
q = g_probe / ||g_probe||
s_r = <q, g_t^(r)>
sigma_q = Std_r(s_r)

本脚本严格不做：
- 不施加 +epsilon / -epsilon；
- 不比较 B 与 chi；
- 不启动控制器。

v0.3 的目的只是回答：沿 probe 方向的自然 minibatch/随机流涨落是否可测，
并给出下一阶段线性响应实验的 epsilon 候选尺度。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from fr.batch_manifest import batchify_indices, make_permutation, save_manifest
from fr.direction import direction_norm, normalize_direction, project_current_gradients, save_direction
from fr.fluctuation_stats import summarize_projections
from fr.full_state import load_full_training_state, save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.rng import capture_rng_state, restore_rng_state
from fr.stream_rng import derive_stream_seed, seed_future_stream


def load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def filtered_stream_batches(dataset_len: int, excluded: set[int], batch_size: int, seed: int, needed: int):
    """独立 Generator 生成 stream batch，并显式排除 probe 样本。"""
    perm = make_permutation(dataset_len, seed)
    train_indices = [i for i in perm if i not in excluded]
    return batchify_indices(train_indices, batch_size)[:needed]


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 自然涨落 pilot")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e1_pilot_v03.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = load_cfg(root / args.config)
    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("请求 CUDA，但当前进程看不到 CUDA")

    out = root / "outputs/e1_fluctuation/v03_pilot"
    out.mkdir(parents=True, exist_ok=True)
    full_data = minimind / "dataset/pretrain_t2t_mini.jsonl"
    subset_lines = int(cfg.get("subset_lines", 4096))
    if subset_lines > 0:
        data_path = out / "e1_pilot_subset.jsonl"
        copy_jsonl_prefix(full_data, data_path, subset_lines)
    else:
        data_path = full_data

    setup_seed(int(cfg["state_seed"]))
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device)
    model.train()
    ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))

    autocast_ctx, scaler, device_type = make_amp(torch, args.device, cfg["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    accum = int(cfg["accumulation_steps"])
    warm_steps = int(cfg["warmup_optimizer_steps"])
    warm_micro = warm_steps * accum

    # ------------------------------------------------------------
    # A. 先产生一个非平凡 AdamW 状态 Z_t。
    # ------------------------------------------------------------
    warm_perm = make_permutation(len(ds), int(cfg["state_seed"]) + 17)
    warm_batches = batchify_indices(warm_perm, int(cfg["batch_size"]))[:warm_micro]
    if len(warm_batches) < warm_micro:
        raise SystemExit("数据子集太小，无法完成 warmup")

    for j, batch_idx in enumerate(warm_batches):
        step = j + 1
        lr = get_lr(step, max(warm_micro, 1), float(cfg["learning_rate"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = stack_batch(ds, batch_idx, args.device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()
        if step % accum == 0:
            # [FR-NATIVE-ADAMW] warmup 保持原生顺序。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if device_type == "cuda":
        torch.cuda.synchronize()

    state_path = out / "e1_reference_state.pt"
    save_full_training_state(
        state_path, model, optimizer, scaler,
        metadata={
            "protocol": "E1_V03_PILOT",
            "warmup_optimizer_steps": warm_steps,
            "optimizer_signature": optimizer_signature(optimizer),
            "config": cfg,
        },
    )

    # ------------------------------------------------------------
    # B. 用训练-held-out probe batch 构造 q。
    #    probe 采用数据子集末尾样本，并从所有 stream batch 中显式排除。
    # ------------------------------------------------------------
    probe_n = int(cfg["probe_batch_size"])
    if probe_n <= 0 or probe_n >= len(ds):
        raise SystemExit("probe_batch_size 无效")
    probe_indices = list(range(len(ds) - probe_n, len(ds)))
    probe_set = set(probe_indices)

    load_full_training_state(state_path, model, optimizer, scaler)
    saved_rng = capture_rng_state()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    x_probe, y_probe = stack_batch(ds, probe_indices, args.device)
    with autocast_ctx:
        probe_res = model(x_probe, labels=y_probe)
        probe_loss = probe_res.loss + probe_res.aux_loss

    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
    q_raw = {}
    for (name, p), g in zip(named, grads):
        q_raw[name] = torch.zeros_like(p.detach()) if g is None else g.detach().clone()
    q, probe_grad_norm = normalize_direction(q_raw)
    q_norm = direction_norm(q)
    q_sha = save_direction(
        out / "probe_direction_q.pt", q,
        metadata={
            "probe_indices": probe_indices,
            "probe_loss": float(probe_loss.detach().float().cpu()),
            "probe_grad_norm_before_normalize": probe_grad_norm,
            "q_norm": q_norm,
            "construction": "eval_mode held-out probe gradient; no optimizer step",
        },
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    restore_rng_state(saved_rng)

    # ------------------------------------------------------------
    # C. 从同一 Z_t 分叉 R 条自然随机流，只观察第一个 optimizer-boundary
    #    的“未裁剪累计梯度”，不执行 optimizer step。
    # ------------------------------------------------------------
    R = int(cfg["streams"])
    if R < 4:
        raise SystemExit("E1 pilot 建议 streams>=4；当前配置不足")
    stream_records = []
    projections, grad_norms = [], []

    for r in range(R):
        load_full_training_state(state_path, model, optimizer, scaler)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        stream_seed = derive_stream_seed(int(cfg["stream_base_seed"]), r)
        seed_future_stream(stream_seed)
        batch_seed = derive_stream_seed(stream_seed, 999)
        batches = filtered_stream_batches(
            len(ds), probe_set, int(cfg["batch_size"]), batch_seed, accum
        )
        if len(batches) < accum:
            raise SystemExit(f"stream {r} 无法生成足够 batch")
        if any(i in probe_set for b in batches for i in b):
            raise RuntimeError("probe/train stream 出现样本重叠")

        losses, batch_hashes = [], []
        for micro, batch_idx in enumerate(batches):
            x, y = stack_batch(ds, batch_idx, args.device)
            batch_hashes.append(stable_hash({"x": x, "y": y}))
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().float().cpu()) * accum)

        # float16 时 grad 仍带 scaler 比例；这里只读除以 scale，不调用 unscale_。
        grad_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        obs = project_current_gradients(named, q, grad_scale=grad_scale)
        projections.append(obs["projection"])
        grad_norms.append(obs["grad_norm"])

        manifest_path = out / f"stream_{r:03d}_batches.json"
        manifest_sha = save_manifest(
            manifest_path, batches,
            metadata={"stream_id": r, "stream_seed": stream_seed, "batch_seed": batch_seed},
        )
        stream_records.append({
            "stream_id": r,
            "stream_seed": stream_seed,
            "batch_seed": batch_seed,
            "manifest_sha256": manifest_sha,
            "losses": losses,
            "batch_hashes": batch_hashes,
            **obs,
        })

    stats = summarize_projections(projections, grad_norms)
    passed = bool(stats["finite"] and stats["nonzero_variance"])
    report = {
        "protocol": "E1_V03_PILOT",
        "technical_pass": passed,
        "scientific_intervention_enabled": False,
        "q": {
            "sha256": q_sha,
            "norm": q_norm,
            "probe_indices": probe_indices,
            "probe_grad_norm_before_normalize": probe_grad_norm,
        },
        "stats": stats,
        "streams": stream_records,
        "boundaries": [
            "E1 只证明自然涨落是否可测，不证明 B->chi。",
            "epsilon_candidates 仅供下一阶段 E2 pilot 选尺度，本脚本不会施加扰动。",
            "当前 subset/pilot 配置不能替代跨 seed、跨阶段正式实验。",
        ],
    }
    (out / "e1_fluctuation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# E1 v0.3 自然涨落 Pilot",
        "",
        f"- 技术判定：**{'PASS' if passed else 'FAIL'}**",
        "- 人工扰动：**关闭**",
        f"- R：`{stats['R']}`",
        f"- q norm：`{q_norm:.12g}`",
        f"- sigma_q：`{stats['sigma_q']:.12g}`",
        f"- mean_s：`{stats['mean_s']:.12g}`",
        f"- mean ||g||：`{stats['mean_grad_norm']:.12g}`",
        f"- sigma_q / mean||g||：`{stats['sigma_over_mean_grad_norm']:.12g}`",
        "",
        "## 下一阶段 epsilon 候选（这里只计算，不执行）",
        "",
    ]
    for k, v in stats["epsilon_candidates"].items():
        md.append(f"- {k}: `{v:.12g}`")
    md += [
        "",
        "## 科学边界",
        "",
        "E1 PASS 只表示沿当前 q 的自然随机涨落在该 pilot 上可测。",
        "它不是涨落—响应关系成立的证据，也不能据此启动控制器。",
        "",
    ]
    (out / "e1_fluctuation_report.md").write_text("\n".join(md), encoding="utf-8")

    print("E1_V03_FLUCTUATION_PASS" if passed else "E1_V03_FLUCTUATION_FAIL")
    print("sigma_q=", stats["sigma_q"])
    print("epsilon_candidates=", stats["epsilon_candidates"])
    print("report=", out / "e1_fluctuation_report.md")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
