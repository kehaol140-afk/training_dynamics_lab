#!/usr/bin/env python3
"""MiniMind + AdamW 的 E0 精确重放 smoke。

这个脚本做什么？
----------------
1. 从真实 `pretrain_t2t_mini.jsonl` 复制很小的前缀子集，只用于快速 smoke；
2. 使用 MiniMind 自己的模型、tokenizer、PretrainDataset、get_lr；
3. warmup 一个 optimizer step，让 AdamW m/v/step 变成非空；
4. 在 optimizer step + zero_grad 后保存 FR v0.2 完整状态；
5. 分支 A、B 都从这份状态重新加载；
6. 两条分支使用同一未来 batch manifest；
7. 比较 loss trace、batch hash、模型 hash、optimizer hash。

重要：E0 中不允许调用任何 `±epsilon*q` 扰动。
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

from fr.batch_manifest import build_native_single_gpu_batches, manifest_sha256, save_manifest
from fr.full_state import load_full_training_state, save_full_training_state
from fr.hashing import stable_hash
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature


def copy_jsonl_prefix(src: Path, dst: Path, n: int) -> None:
    """复制前 n 行到 smoke 子集；不修改原始数据。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniMind E0 smoke")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accumulation-steps", type=int, default=2)
    ap.add_argument("--warmup-optimizer-steps", type=int, default=1)
    ap.add_argument("--replay-optimizer-steps", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--subset-lines", type=int, default=64)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("请求 CUDA，但当前进程看不到 CUDA")

    out = root / "outputs/e0_replay/v02_smoke"
    out.mkdir(parents=True, exist_ok=True)
    subset = out / "smoke_subset.jsonl"
    full_data = minimind / "dataset/pretrain_t2t_mini.jsonl"
    if not full_data.exists():
        raise SystemExit(f"找不到数据集：{full_data}")
    copy_jsonl_prefix(full_data, subset, args.subset_lines)

    # ============================================================
    # [FR-RNG-CRITICAL] 初始化时固定 seed；后续真正 branch 起点由 full-state RNG 决定。
    # ============================================================
    setup_seed(2026)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        max_seq_len=args.max_seq_len,
        use_moe=False,
    )
    model, tokenizer = init_model(
        config,
        from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=args.device,
    )
    model.train()
    ds = PretrainDataset(str(subset), tokenizer, max_length=args.max_seq_len)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    assert_native_adamw(optimizer)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=amp_dtype)

    warmup_micro = args.warmup_optimizer_steps * args.accumulation_steps
    replay_micro = args.replay_optimizer_steps * args.accumulation_steps
    total_micro = warmup_micro + replay_micro

    # [FR-BATCH-STREAM] 显式锁定整个 smoke 所需 batch 顺序。
    batches = build_native_single_gpu_batches(len(ds), epoch=0, batch_size=args.batch_size)
    if len(batches) < total_micro:
        raise SystemExit("smoke 子集太小，不足以生成所需 batch")
    batches = batches[:total_micro]
    manifest_sha = save_manifest(
        out / "batch_manifest.json",
        batches,
        metadata={"epoch": 0, "batch_size": args.batch_size, "purpose": "E0 smoke"},
    )

    def get_batch(batch_indices: List[int]):
        xs, ys = zip(*(ds[i] for i in batch_indices))
        return torch.stack(xs).to(args.device), torch.stack(ys).to(args.device)

    def run_segment(start_micro_index: int, count: int) -> Dict:
        """严格按当前 MiniMind 顺序运行一段 micro-steps。"""
        losses = []
        batch_hashes = []
        for offset in range(count):
            j = start_micro_index + offset
            step = j + 1
            lr = get_lr(step, total_micro, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids, labels = get_batch(batches[j])
            batch_hashes.append(stable_hash({"x": input_ids, "y": labels}))

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()
            losses.append(float(loss.detach().float().cpu()) * args.accumulation_steps)

            if step % args.accumulation_steps == 0:
                # ====================================================
                # [FR-NATIVE-ADAMW] 必须保持 MiniMind 原生顺序：
                # unscale -> clip -> scaler.step(AdamW) -> update -> zero_grad
                #
                # [FR-PERTURBATION-CRITICAL]
                # E0 此处明确禁止任何 grad ± epsilon*q。
                # ====================================================
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        raw_model = getattr(getattr(model, "module", model), "_orig_mod", getattr(model, "module", model))
        return {
            "losses": losses,
            "batch_hashes": batch_hashes,
            "model_hash": stable_hash(raw_model.state_dict()),
            "optimizer_hash": stable_hash(optimizer.state_dict()),
        }

    # 先 warmup，使 AdamW 的 m/v/step 真正存在。
    optimizer.zero_grad(set_to_none=True)
    warm = run_segment(0, warmup_micro)

    state_path = out / "fr_full_state_after_warmup.pt"
    # ============================================================
    # [FR-REPLAY-CRITICAL] 保存点必须在完成 optimizer step + zero_grad 后。
    # ============================================================
    save_full_training_state(
        state_path,
        model,
        optimizer,
        scaler,
        metadata={
            "epoch": 0,
            "next_micro_index": warmup_micro,
            "accumulation_steps": args.accumulation_steps,
            "manifest_sha256": manifest_sha,
            "optimizer_signature": optimizer_signature(optimizer),
        },
    )

    # 分支 A：先从磁盘完整恢复，再运行。
    meta_a = load_full_training_state(state_path, model, optimizer, scaler)
    branch_a = run_segment(warmup_micro, replay_micro)

    # 分支 B：再次从同一个磁盘状态恢复，RNG 也被恢复到同一点。
    meta_b = load_full_training_state(state_path, model, optimizer, scaler)
    branch_b = run_segment(warmup_micro, replay_micro)

    checks = {
        "metadata_equal": meta_a == meta_b,
        "batch_hashes_equal": branch_a["batch_hashes"] == branch_b["batch_hashes"],
        "loss_trace_equal": branch_a["losses"] == branch_b["losses"],
        "model_hash_equal": branch_a["model_hash"] == branch_b["model_hash"],
        "optimizer_hash_equal": branch_a["optimizer_hash"] == branch_b["optimizer_hash"],
    }
    passed = all(checks.values())

    report = {
        "protocol": "E0_MINIMIND_SMOKE_V0.2",
        "passed": passed,
        "scientific_intervention_enabled": False,
        "device": args.device,
        "dtype": args.dtype,
        "manifest_sha256": manifest_sha256(batches),
        "warmup": warm,
        "branch_a": branch_a,
        "branch_b": branch_b,
        "checks": checks,
        "note": "smoke 只验证基础设施，不代表跨seed/跨阶段正式E0科学结论",
    }
    (out / "e0_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# E0 MiniMind Smoke 报告",
        "",
        f"- 总判定：**{'PASS' if passed else 'FAIL'}**",
        f"- device：`{args.device}`",
        f"- dtype：`{args.dtype}`",
        f"- batch manifest：`{manifest_sha}`",
        "- 人工扰动：**关闭**",
        "",
        "## 配对检查",
        "",
    ]
    md += [f"- {k}: **{v}**" for k, v in checks.items()]
    md += [
        "",
        "## 边界",
        "",
        "本结果只说明当前机器上 FR 完整状态 + 锁定 batch stream 的 smoke 能否重放。",
        "它不是 B→χ 验证，也不是涨落—响应关系成立的证据。",
    ]
    (out / "e0_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("E0_MINIMIND_SMOKE_PASS" if passed else "E0_MINIMIND_SMOKE_FAIL")
    print("checks=", checks)
    print("report=", out / "e0_smoke_report.md")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
