#!/usr/bin/env python3
"""E0 扩展验证：比 v0.2 smoke 更大的模型、更长的无扰动 A/B 重放。

这个脚本做什么
--------------
- 使用真实 MiniMind 模型、tokenizer、PretrainDataset、AdamW；
- 使用更大的 pilot 配置和更多 optimizer steps；
- warmup 后保存完整状态 Z_t；
- A/B 两个分支从同一个 Z_t 出发，使用相同 batch、相同 RNG；
- 不施加任何 ±epsilon*q；
- 要求 loss、模型、AdamW 状态继续一致。

它仍然是“扩展 E0 验证”，不是跨 seed / 跨阶段科学结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fr.batch_manifest import build_native_single_gpu_batches, save_manifest
from fr.full_state import load_full_training_state, save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature


def load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="E0 扩展精确重放验证")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e0_extended_v03.json")
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

    out = root / "outputs/e0_replay/v03_extended"
    out.mkdir(parents=True, exist_ok=True)
    full_data = minimind / "dataset/pretrain_t2t_mini.jsonl"
    subset_lines = int(cfg.get("subset_lines", 4096))
    if subset_lines > 0:
        data_path = out / "e0_extended_subset.jsonl"
        copy_jsonl_prefix(full_data, data_path, subset_lines)
    else:
        data_path = full_data

    setup_seed(int(cfg["seed"]))
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
    warm_micro = int(cfg["warmup_optimizer_steps"]) * accum
    replay_micro = int(cfg["replay_optimizer_steps"]) * accum
    total_micro = warm_micro + replay_micro
    batches = build_native_single_gpu_batches(len(ds), epoch=0, batch_size=int(cfg["batch_size"]))[:total_micro]
    if len(batches) < total_micro:
        raise SystemExit("数据子集太小，无法产生足够 batch")
    manifest_sha = save_manifest(out / "batch_manifest.json", batches, metadata={"purpose": "E0 v0.3 extended"})

    def run_segment(start: int, count: int):
        losses, b_hash = [], []
        for j in range(start, start + count):
            step = j + 1
            lr = get_lr(step, total_micro, float(cfg["learning_rate"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = stack_batch(ds, batches[j], args.device)
            b_hash.append(stable_hash({"x": x, "y": y}))
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().float().cpu()) * accum)
            if step % accum == 0:
                # ====================================================
                # [FR-NATIVE-ADAMW] 完全保留原生顺序。
                # [FR-PERTURBATION-CRITICAL] E0 明确禁止人工扰动。
                # ====================================================
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()
        return {
            "losses": losses,
            "batch_hashes": b_hash,
            "model_hash": stable_hash(raw_model(model).state_dict()),
            "optimizer_hash": stable_hash(optimizer.state_dict()),
        }

    warm = run_segment(0, warm_micro)
    state_path = out / "fr_full_state_after_warmup.pt"
    save_full_training_state(
        state_path, model, optimizer, scaler,
        metadata={
            "protocol": "E0_V03_EXTENDED",
            "next_micro_index": warm_micro,
            "manifest_sha256": manifest_sha,
            "optimizer_signature": optimizer_signature(optimizer),
            "config": cfg,
        },
    )

    meta_a = load_full_training_state(state_path, model, optimizer, scaler)
    a = run_segment(warm_micro, replay_micro)
    meta_b = load_full_training_state(state_path, model, optimizer, scaler)
    b = run_segment(warm_micro, replay_micro)

    checks = {
        "metadata_equal": meta_a == meta_b,
        "batch_hashes_equal": a["batch_hashes"] == b["batch_hashes"],
        "loss_trace_equal": a["losses"] == b["losses"],
        "model_hash_equal": a["model_hash"] == b["model_hash"],
        "optimizer_hash_equal": a["optimizer_hash"] == b["optimizer_hash"],
    }
    passed = all(checks.values())
    report = {
        "protocol": "E0_V03_EXTENDED",
        "passed": passed,
        "scientific_intervention_enabled": False,
        "config": cfg,
        "warmup": warm,
        "branch_a": a,
        "branch_b": b,
        "checks": checks,
    }
    (out / "e0_extended_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# E0 v0.3 扩展验证", "", f"- 判定：**{'PASS' if passed else 'FAIL'}**", "- 人工扰动：**关闭**", ""]
    md += [f"- {k}: **{v}**" for k, v in checks.items()]
    md += ["", "> 该结果只验证更大 pilot 配置下的精确重放，不代表 FR 关系成立。", ""]
    (out / "e0_extended_report.md").write_text("\n".join(md), encoding="utf-8")

    print("E0_V03_EXTENDED_PASS" if passed else "E0_V03_EXTENDED_FAIL")
    print("checks=", checks)
    print("report=", out / "e0_extended_report.md")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
