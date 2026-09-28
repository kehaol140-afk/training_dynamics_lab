#!/usr/bin/env python3
"""v0.7 confirmatory：为 8 个 confirmation training seed 建立 7-state bank。

本脚本会产生正式 confirmatory 数据；运行前必须已有有效 PREREG_LOCK.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fr.full_state import save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import build_partition, load_partition, save_partition
from fr.stage_c_runtime import batches_from_train_pool
from scientific_preflight_v07 import assert_prereg_lock_valid, file_sha256


def parse_seeds(text, defaults):
    if not text:
        return list(map(int, defaults))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def collect_passed_state_reports(out: Path, all_seeds):
    passed = []
    reports = []
    for seed in map(int, all_seeds):
        p = out / f"seed_{seed}/state_bank_report.json"
        if not p.exists():
            continue
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if x.get("technical_pass") is True and int(x.get("train_seed", -1)) == seed:
            passed.append(seed)
            reports.append(x)
    return passed, reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e3c_confirmatory_v07.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    out = root / cfg["output_dir"]
    lock = assert_prereg_lock_valid(root, out)
    all_seeds = list(map(int, cfg["confirmation_train_seeds"]))
    seeds = parse_seeds(args.seeds, all_seeds)
    if any(seed not in set(all_seeds) for seed in seeds):
        raise SystemExit("v0.7 state bank 只允许 confirmation_train_seeds")

    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "e3c_subset.jsonl"
    copy_jsonl_prefix(
        minimind / "dataset/pretrain_t2t_mini.jsonl",
        data_path,
        int(cfg["subset_lines"]),
    )

    part_path = out / "partition_v07.json"
    if part_path.exists():
        partition = load_partition(part_path)
    else:
        setup_seed(int(all_seeds[0]))
        mmcfg0 = MiniMindConfig(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            max_seq_len=int(cfg["max_seq_len"]),
            use_moe=False,
        )
        m0, tok0 = init_model(
            mmcfg0,
            from_weight="none",
            tokenizer_path=str(minimind / "model"),
            device=args.device,
        )
        ds0 = PretrainDataset(str(data_path), tok0, max_length=int(cfg["max_seq_len"]))
        partition = build_partition(
            len(ds0),
            int(cfg["partition_seed"]),
            int(cfg["probe_pool_size"]),
            int(cfg["eval_pool_size"]),
        )
        save_partition(part_path, partition)
        del m0, ds0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for train_seed in seeds:
        seed_dir = out / f"seed_{train_seed}"
        states_dir = seed_dir / "states"
        report_path = seed_dir / "state_bank_report.json"

        if report_path.exists() and not args.force:
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if old.get("technical_pass") and int(old.get("train_seed", -1)) == train_seed:
                print(f"SEED_{train_seed}_STATE_BANK_SKIP_EXISTING_PASS")
                continue

        states_dir.mkdir(parents=True, exist_ok=True)
        setup_seed(train_seed)

        mmcfg = MiniMindConfig(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            max_seq_len=int(cfg["max_seq_len"]),
            use_moe=False,
        )
        model, tokenizer = init_model(
            mmcfg,
            from_weight="none",
            tokenizer_path=str(minimind / "model"),
            device=args.device,
        )
        model.train()
        ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
        autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
        assert_native_adamw(optimizer)
        optimizer.zero_grad(set_to_none=True)

        accum = int(cfg["accumulation_steps"])
        total_opt = int(cfg["planned_total_optimizer_steps"])
        total_micro = total_opt * accum
        state_steps = sorted(map(int, cfg["e3_state_steps"]))
        horizons = sorted(map(int, cfg["horizons"]))
        if max(state_steps) + max(horizons) > total_opt:
            raise RuntimeError("state_steps + max(H) 超出预注册总预算")
        max_micro = max(state_steps) * accum

        # [FR-BATCH-STREAM] 每个 confirmation training seed 使用独立训练 trajectory。
        train_batch_seed = train_seed + 1704
        batches = batches_from_train_pool(
            partition["train_indices"],
            int(cfg["batch_size"]),
            train_batch_seed,
            max_micro,
        )

        first_micro_batch = list(map(int, batches[0]))
        probe_set = set(partition["probe_indices"])
        eval_set = set(partition["eval_indices"])
        step0_probe_overlap = len(set(first_micro_batch) & probe_set)
        step0_eval_overlap = len(set(first_micro_batch) & eval_set)
        if step0_probe_overlap or step0_eval_overlap:
            raise RuntimeError("trajectory 第0步/首个 micro-batch 使用了 probe/eval")

        used = set()
        records = []
        for micro_idx, batch_idx in enumerate(batches, start=1):
            used.update(batch_idx)
            set_native_micro_lr(
                optimizer,
                get_lr,
                micro_idx,
                total_micro,
                float(cfg["learning_rate"]),
            )
            x, y = stack_batch(ds, batch_idx, args.device)
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()

            if micro_idx % accum == 0:
                # [FR-NATIVE-ADAMW] 保持 MiniMind 原生 unscale -> clip -> step -> zero_grad。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                opt_step = micro_idx // accum
                if opt_step in state_steps:
                    path = states_dir / f"state_step_{opt_step:03d}.pt"
                    st = save_full_training_state(
                        path,
                        model,
                        optimizer,
                        scaler,
                        metadata={
                            "protocol": "E3C_CONFIRMATORY_V07_STATE_BANK",
                            "train_seed": train_seed,
                            "state_optimizer_step": opt_step,
                            "state_fraction": opt_step / total_opt,
                            "global_optimizer_step": opt_step,
                            "global_micro_step": micro_idx,
                            "planned_total_optimizer_steps": total_opt,
                            "total_micro_steps": total_micro,
                            "partition_sha256": partition["sha256"],
                            "optimizer_signature": optimizer_signature(optimizer),
                            "prereg_lock_created_at_utc": lock.get("created_at_utc"),
                            "config": cfg,
                        },
                    )
                    records.append({
                        "step": opt_step,
                        "path": str(path),
                        "model_hash": stable_hash(st["model"]),
                        "optimizer_hash": stable_hash(st["optimizer"]),
                    })

        probe_overlap = len(used & probe_set)
        eval_overlap = len(used & eval_set)
        if probe_overlap or eval_overlap:
            raise RuntimeError(f"seed={train_seed} state trajectory 使用了 probe/eval")

        report = {
            "protocol": "E3C_CONFIRMATORY_V07_STATE_BANK",
            "technical_pass": len(records) == len(state_steps),
            "train_seed": train_seed,
            "train_batch_seed": train_batch_seed,
            "state_steps": state_steps,
            "horizons": horizons,
            "partition_sha256": partition["sha256"],
            "probe_overlap": probe_overlap,
            "eval_overlap": eval_overlap,
            "trajectory_step0_first_micro_batch": first_micro_batch,
            "trajectory_step0_probe_overlap": step0_probe_overlap,
            "trajectory_step0_eval_overlap": step0_eval_overlap,
            "prereg_lock_path": str(out / "PREREG_LOCK.json"),
            "prereg_lock_sha256": file_sha256(out / "PREREG_LOCK.json"),
            "states": records,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SEED_{train_seed}_V07_STATE_BANK_PASS")
        print(" probe_overlap=", probe_overlap, "eval_overlap=", eval_overlap)
        print(" step0_probe_overlap=", step0_probe_overlap, "step0_eval_overlap=", step0_eval_overlap)

        del model, optimizer, scaler, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed, _reports = collect_passed_state_reports(out, all_seeds)
    summary = {
        "protocol": "E3C_CONFIRMATORY_V07_MULTISEED_STATE_BANK_SUMMARY",
        "technical_pass": len(passed) == len(all_seeds),
        "seeds": all_seeds,
        "requested_seeds": seeds,
        "completed": passed,
        "missing_or_not_passed": [seed for seed in all_seeds if seed not in set(passed)],
        "partition_sha256": partition["sha256"],
        "seed_reports": [str(out / f"seed_{seed}/state_bank_report.json") for seed in all_seeds],
    }
    (out / "multiseed_state_bank_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("E3C_V07_MULTISEED_STATE_BANK_SUMMARY_WRITTEN")
    print("completed=", passed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
