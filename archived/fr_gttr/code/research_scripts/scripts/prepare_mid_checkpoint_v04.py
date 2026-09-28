#!/usr/bin/env python3
"""建立 v0.4 的 1 seed × MID pilot 完整 checkpoint。

脚本做什么
----------
1. 固定 4096 行 pilot 子集；
2. 显式划分 train / train-probe / eval-probe；
3. probe/eval 从第 0 步开始就不参与训练；
4. 按计划总预算 128 optimizer steps 定义时间轴；
5. 训练到第 64 optimizer step，即 50% MID；
6. 保存完整 Z_t。

这里没有自然涨落分叉，也没有人工扰动。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from fr.batch_manifest import make_permutation, batchify_indices
from fr.full_state import save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.pilot_partition import build_partition, save_partition
from fr.stage_c_runtime import batches_from_train_pool


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
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    out = root / "outputs/stage_c_v04"
    out.mkdir(parents=True, exist_ok=True)

    data_path = out / "stage_c_subset.jsonl"
    copy_jsonl_prefix(
        minimind / "dataset/pretrain_t2t_mini.jsonl",
        data_path,
        int(cfg["subset_lines"]),
    )

    setup_seed(int(cfg["state_seed"]))
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
    model.train()
    ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))

    partition = build_partition(
        len(ds),
        int(cfg["partition_seed"]),
        int(cfg["probe_pool_size"]),
        int(cfg["eval_pool_size"]),
    )
    save_partition(out / "partition_v04.json", partition)

    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    accum = int(cfg["accumulation_steps"])
    mid_opt = int(cfg["mid_optimizer_step"])
    total_opt = int(cfg["planned_total_optimizer_steps"])
    if not (0 < mid_opt < total_opt):
        raise SystemExit("mid_optimizer_step 必须严格位于总预算内部")

    total_micro = total_opt * accum
    mid_micro = mid_opt * accum

    # ============================================================
    # [FR-BATCH-STREAM]
    # MID 训练只从 train_indices 取样。
    # probe_indices / eval_indices 从训练起点就永远留出。
    # ============================================================
    batches = batches_from_train_pool(
        partition["train_indices"],
        int(cfg["batch_size"]),
        int(cfg["state_seed"]) + 1704,
        mid_micro,
    )

    used = set()
    for micro_idx, batch_idx in enumerate(batches, start=1):
        used.update(batch_idx)
        global_micro = micro_idx
        lr = float(get_lr(global_micro, total_micro, float(cfg["learning_rate"])))
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = stack_batch(ds, batch_idx, args.device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()

        if micro_idx % accum == 0:
            # ====================================================
            # [FR-NATIVE-ADAMW]
            # 与已审计 MiniMind 顺序一致；这里没有 perturbation。
            # ====================================================
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if used & set(partition["probe_indices"]):
        raise RuntimeError("MID 训练误用了 probe")
    if used & set(partition["eval_indices"]):
        raise RuntimeError("MID 训练误用了 eval")

    state_path = out / "mid_state_v04.pt"
    save_full_training_state(
        state_path, model, optimizer, scaler,
        metadata={
            "protocol": "STAGE_C_V04_MID",
            "config": cfg,
            "partition_sha256": partition["sha256"],
            "global_optimizer_step": mid_opt,
            "global_micro_step": mid_micro,
            "planned_total_optimizer_steps": total_opt,
            "total_micro_steps": total_micro,
            "optimizer_signature": optimizer_signature(optimizer),
            "train_used_count": len(used),
        },
    )

    summary = {
        "state_path": str(state_path),
        "partition_path": str(out / "partition_v04.json"),
        "mid_fraction": mid_opt / total_opt,
        "partition_sha256": partition["sha256"],
        "train_used_count": len(used),
        "probe_overlap": len(used & set(partition["probe_indices"])),
        "eval_overlap": len(used & set(partition["eval_indices"])),
    }
    (out / "mid_prepare_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V04_MID_PREP_PASS")
    print("mid_fraction=", summary["mid_fraction"])
    print("probe_overlap=", summary["probe_overlap"])
    print("eval_overlap=", summary["eval_overlap"])
    print("state=", state_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
