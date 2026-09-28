#!/usr/bin/env python3
"""FR_A0G 前置：为正式评价铸造最小 confirmatory bank（seeds 2038–2041）。

遵循 v07 相同训练配方（同模型/同配置/同 state-step 定义/同 partition/同数据），
训练到 step 82，保存 {46,64,82} 三个 full training state，并为每个 state 铸造
8 条 E2 风格 future batch manifests（H=8 × accumulation 2 = 16 batches）。

这是新实验数据（不是 v07 bank 的改动），全部写入
outputs/fr_decision_utility_v01/confirmatory_bank/。

铸造后做两项验收：
  1) probe/eval 重叠检查（与 v07 相同）；
  2) 每个保存 state 跑两条 frozen 基线 replay（run_theta_branch sign=0），
     最终 model/optimizer hash 必须一致（确定性验收）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.batch_manifest import save_manifest
from fr.full_state import load_full_training_state, save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import load_partition
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed
from fr.theta_protocol import run_theta_branch

from fr.a0c_h1_adamw_jacobian_v01 import file_sha256


NEW_SEEDS = (2038, 2039, 2040, 2041)
STATE_STEPS = (46, 64, 82)
N_STREAMS = 8
E2_STREAM_BASE_SEED = 403202610  # FR_DECISION_UTILITY_V01 专用，区别于 v07 的 308202601
MANIFEST_PROTOCOL = "FR_DECISION_UTILITY_V01_E2_PAIRED_STREAM"
STATE_BANK_PROTOCOL = "FR_DECISION_UTILITY_V01_STATE_BANK"
BANK_DIR = "confirmatory_bank"


def two_level_seed(base, train_seed, state_step, stream_id):
    x = derive_stream_seed(int(base), int(train_seed))
    x = derive_stream_seed(x, int(state_step))
    return derive_stream_seed(x, int(stream_id))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _zeros_q(named, torch):
    return {name: torch.zeros_like(p.detach()) for name, p in named}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default=",".join(map(str, NEW_SEEDS)))
    ap.add_argument("--skip-training", action="store_true",
                    help="states/manifests 已存在时只重跑验收")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = _read_json(root / "research/configs/e3c_confirmatory_v07.json")
    out = root / "outputs/fr_decision_utility_v01" / BANK_DIR
    out.mkdir(parents=True, exist_ok=True)
    v07_out = root / cfg["output_dir"]

    # 只读输入：冻结 partition / 数据子集 / 配置（与 v07 完全相同）
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        use_moe=False,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dataset = None
    bank_index = {"protocol": "FR_DECISION_UTILITY_V01_CONFIRMATORY_BANK_INDEX", "seeds": {}}

    for train_seed in seeds:
        seed_dir = out / f"seed_{train_seed}"
        states_dir = seed_dir / "states"
        report_path = seed_dir / "bank_report.json"
        states_dir.mkdir(parents=True, exist_ok=True)

        if report_path.exists() and not args.skip_training:
            old = _read_json(report_path)
            if old.get("technical_pass") is True and int(old.get("train_seed", -1)) == train_seed:
                print(f"SEED_{train_seed}_SKIP_EXISTING_PASS")
                bank_index["seeds"][str(train_seed)] = str(report_path)
                continue

        # ---- 训练（v07 相同配方） ----
        setup_seed(train_seed)
        model, tokenizer = init_model(
            mmcfg, from_weight="none",
            tokenizer_path=str(minimind / "model"), device=args.device,
        )
        model.train()
        if dataset is None:
            dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
        assert_native_adamw(optimizer)
        optimizer.zero_grad(set_to_none=True)
        named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
        autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
        if scaler.is_enabled():
            raise RuntimeError("bf16 replay 不应启用 GradScaler")

        accum = int(cfg["accumulation_steps"])
        total_opt = int(cfg["planned_total_optimizer_steps"])
        total_micro = total_opt * accum
        if max(STATE_STEPS) + 8 > total_opt:
            raise RuntimeError("state_steps + H=8 超出预注册总预算")
        max_micro = max(STATE_STEPS) * accum

        train_batch_seed = train_seed + 1704  # 与 v07 相同的派生规则
        batches = batches_from_train_pool(
            partition["train_indices"], int(cfg["batch_size"]), train_batch_seed, max_micro,
        )
        probe_set = set(partition["probe_indices"])
        eval_set = set(partition["eval_indices"])
        first_micro = set(map(int, batches[0]))
        if (first_micro & probe_set) or (first_micro & eval_set):
            raise RuntimeError("trajectory 第0步使用了 probe/eval")

        records = []
        used = set()
        for micro_idx, batch_idx in enumerate(batches, start=1):
            used.update(int(i) for i in batch_idx)
            set_native_micro_lr(optimizer, get_lr, micro_idx, total_micro, float(cfg["learning_rate"]))
            x, y = stack_batch(dataset, batch_idx, args.device)
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            if micro_idx % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step = micro_idx // accum
                if opt_step in STATE_STEPS:
                    path = states_dir / f"state_step_{opt_step:03d}.pt"
                    st = save_full_training_state(
                        path, model, optimizer, scaler,
                        metadata={
                            "protocol": STATE_BANK_PROTOCOL,
                            "train_seed": train_seed,
                            "state_optimizer_step": opt_step,
                            "state_fraction": opt_step / total_opt,
                            "global_optimizer_step": opt_step,
                            "global_micro_step": micro_idx,
                            "planned_total_optimizer_steps": total_opt,
                            "total_micro_steps": total_micro,
                            "partition_sha256": partition["sha256"],
                            "optimizer_signature": optimizer_signature(optimizer),
                            "config": cfg,
                        },
                    )
                    records.append({
                        "step": opt_step,
                        "path": str(path),
                        "model_hash": stable_hash(st["model"]),
                        "optimizer_hash": stable_hash(st["optimizer"]),
                    })
                    print(f"SEED_{train_seed} STATE_{opt_step:03d} SAVED", flush=True)

        probe_overlap = len(used & probe_set)
        eval_overlap = len(used & eval_set)
        if probe_overlap or eval_overlap:
            raise RuntimeError(f"seed={train_seed} 训练轨迹使用了 probe/eval")

        # ---- E2 manifests（16 batches = H=8 x accum 2） ----
        manifest_records = []
        for step in STATE_STEPS:
            meta = load_full_training_state(states_dir / f"state_step_{step:03d}.pt",
                                            model, optimizer, scaler, restore_rng=False)
            checkpoint = torch.load(states_dir / f"state_step_{step:03d}.pt",
                                    map_location="cpu", weights_only=False)
            reference_theta = checkpoint["model"]
            state_path = states_dir / f"state_step_{step:03d}.pt"
            for r in range(N_STREAMS):
                stream_seed = two_level_seed(E2_STREAM_BASE_SEED, train_seed, step, r)
                batch_seed = derive_stream_seed(stream_seed, 404)  # 与 v07 相同 recipe
                future_batches = batches_from_train_pool(
                    partition["train_indices"], int(cfg["batch_size"]), batch_seed, 8 * accum,
                )
                manifest_path = (
                    seed_dir / "e2_stream_manifests" / f"state_step_{step:03d}"
                    / f"e2_stream_{r:03d}_batches.json"
                )
                manifest_sha = save_manifest(
                    manifest_path, future_batches,
                    metadata={
                        "protocol": MANIFEST_PROTOCOL,
                        "train_seed": train_seed,
                        "state_step": step,
                        "stream_id": r,
                        "stream_seed": stream_seed,
                        "batch_seed": batch_seed,
                        "horizon_max": 8,
                        "accumulation_steps": accum,
                    },
                )
                manifest_records.append({
                    "step": step, "stream_id": r, "stream_seed": stream_seed,
                    "batch_seed": batch_seed,
                    "batch_manifest_path": str(manifest_path),
                    "batch_manifest_sha256": manifest_sha,
                })

            # ---- 确定性验收：两条 frozen 基线 replay 哈希一致 ----
            rec = next(x for x in manifest_records if x["step"] == step and x["stream_id"] == 0)
            common = dict(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=dataset, state_path=state_path,
                reference_theta=reference_theta, q=_zeros_q(named, torch), named=named,
                batches=json.loads(Path(rec["batch_manifest_path"]).read_text())["batches"],
                stream_seed=int(rec["stream_seed"]), sign=0, epsilon=0.0,
                horizons=(8,), accumulation_steps=accum,
                batch_size=int(cfg["batch_size"]), device=args.device,
                grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
                global_micro_step=int(meta["global_micro_step"]),
                total_micro_steps=int(meta["total_micro_steps"]),
                get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
            )
            b_a = run_theta_branch(**common)
            b_b = run_theta_branch(**common)
            det_pass = (
                b_a["model_hash_final"] == b_b["model_hash_final"]
                and b_a["optimizer_hash_final"] == b_b["optimizer_hash_final"]
            )
            if not det_pass:
                raise RuntimeError(
                    f"seed={train_seed} step={step} 基线 replay 确定性验收失败: "
                    f"{b_a['model_hash_final']} vs {b_b['model_hash_final']}"
                )
            print(f"SEED_{train_seed} STATE_{step:03d} DETERMINISM_PASS "
                  f"model={b_a['model_hash_final'][:12]} opt={b_a['optimizer_hash_final'][:12]}",
                  flush=True)

        report = {
            "protocol": STATE_BANK_PROTOCOL,
            "technical_pass": len(records) == len(STATE_STEPS),
            "train_seed": train_seed,
            "train_batch_seed": train_batch_seed,
            "state_steps": list(STATE_STEPS),
            "partition_sha256": partition["sha256"],
            "data_path": str(data_path),
            "data_path_sha256": file_sha256(data_path),
            "probe_overlap": probe_overlap,
            "eval_overlap": eval_overlap,
            "trajectory_step0_probe_eval_overlap": sorted(first_micro & (probe_set | eval_set)),
            "states": records,
            "e2_manifests": manifest_records,
            "model_config": {
                "hidden_size": int(cfg["hidden_size"]),
                "num_hidden_layers": int(cfg["num_hidden_layers"]),
                "max_seq_len": int(cfg["max_seq_len"]),
                "batch_size": int(cfg["batch_size"]),
                "accumulation_steps": accum,
                "learning_rate": float(cfg["learning_rate"]),
                "grad_clip": float(cfg["grad_clip"]),
                "dtype": cfg["dtype"],
                "planned_total_optimizer_steps": total_opt,
            },
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        bank_index["seeds"][str(train_seed)] = str(report_path)
        print(f"SEED_{train_seed}_BANK_PASS", flush=True)
        del model, optimizer, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_path = out / "bank_index.json"
    index_path.write_text(json.dumps(bank_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BANK_INDEX = {index_path}")
    print("STATUS = COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
