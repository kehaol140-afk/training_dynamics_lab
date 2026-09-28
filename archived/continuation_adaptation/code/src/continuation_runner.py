#!/usr/bin/env python3
"""continuation_runner — 从 v07 bank C0 出发的单次确定性续训运行。

一条 run = 一个实验臂（A/B/C1/C2/D，或 A_repeat 确定性复跑）：
  1. 只读校验：C0 / 语料 / partition / config SHA256；
  2. fresh model + AdamW → load_full_training_state（model→optimizer→scaler→RNG，RNG 最后恢复）；
  3. apply_arm（load 之后才能改 betas；state 通道按臂定义清零/保留并回读验证）；
  4. C0 时刻 eval NLL（各臂必须一致）；
  5. 64 步原生时序续训（逐 micro-batch 写 LR；unscale→clip→step→update→zero_grad），
     逐步记录 train loss / pre-clip grad norm / moment norms / 有效学习率分布；
  6. 预算耗尽后 final eval NLL（primary）；落盘 metrics.json / loss_curve.csv /
     optimizer_state_statistics.csv。

训练循环逐行复刻 v07 bank 制备脚本（e3c_prepare_state_bank_v07.py:170-191），
不引入任何新优化器或控制逻辑。CRN：所有臂共享同一 batch manifest 与同一恢复后的 RNG。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

from vendored_fr.full_state import load_full_training_state
from vendored_fr.hashing import stable_hash
from vendored_fr.minimind_v03_runtime import make_amp, stack_batch
from vendored_fr.native_adamw_guard import assert_native_adamw
from vendored_fr.native_schedule import set_native_micro_lr
from vendored_fr.stage_c_runtime import batches_from_train_pool

from evaluation import load_partition_checked, run_eval_nll
from optimizer_utils import build_param_name_groups, optimizer_state_statistics
from state_adapter import apply_arm, check_arms_table_consistency


def bootstrap(lab_root: Path) -> None:
    for p in (str(SRC_DIR), str(Path(lab_root) / "minimind_base")):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LOSS_CURVE_COLUMNS = [
    "opt_step", "micro_start", "micro_end", "loss_mean",
    "lr_consumed", "preclip_grad_norm",
    "m_norm", "v_norm", "sqrt_vhat_mean",
    "efflr_mean", "efflr_median", "efflr_p10", "efflr_p90", "gain_mean",
]
OPT_STATS_COLUMNS = [
    "opt_step", "group", "n_coords", "step_min", "step_max",
    "m_norm", "v_norm", "sqrt_vhat_mean",
    "efflr_mean", "efflr_median", "efflr_p10", "efflr_p90", "gain_mean",
]


def _row(base: dict, stats: dict) -> list:
    return [base.get(c) if c in base else stats.get(c) for c in LOSS_CURVE_COLUMNS]


def run_one_arm(cfg: dict, lab_root: Path, run_id: str, device: str, run_dir: Path) -> dict:
    import torch

    lab_root = Path(lab_root).resolve()
    bootstrap(lab_root)
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    arm = "A" if run_id == "A_repeat" else run_id
    check_arms_table_consistency(cfg["arms"])

    bank_dir = lab_root / cfg["lab_root_relative"]["bank_dir"]
    state_path = bank_dir / cfg["c0"]["state_relpath"]
    subset_path = bank_dir / cfg["corpus"]["subset_relpath"]
    partition_path = bank_dir / cfg["corpus"]["partition_relpath"]

    # ---------- 1. 只读校验 ----------
    state_sha = sha256_file(state_path)
    corpus_sha = sha256_file(subset_path)
    if corpus_sha != cfg["corpus"]["subset_sha256_frozen"]:
        raise RuntimeError(f"语料 SHA256 与冻结值不一致: {corpus_sha}")

    seed = int(cfg["c0"]["seed"])
    t0 = int(cfg["c0"]["state_optimizer_step"])
    tr = cfg["train"]
    accum = int(tr["accumulation_steps"])
    budget = int(cfg["budget_optimizer_steps"])

    # ---------- 2. fresh world + C0 恢复 ----------
    setup_seed(seed)  # 设置 cudnn deterministic 标志；随后 C0 的 RNG 会覆盖种子状态
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_hidden_layers=int(cfg["model"]["num_hidden_layers"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        use_moe=bool(cfg["model"]["use_moe"]),
    )
    model, tokenizer = init_model(
        mmcfg,
        from_weight="none",
        tokenizer_path=str(lab_root / cfg["lab_root_relative"]["minimind_base"] / "model"),
        device=device,
    )
    model.train()
    ds = PretrainDataset(str(subset_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, device, tr["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["base_lr"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    md = load_full_training_state(state_path, model, optimizer, scaler=scaler)

    exp = cfg["c0"]["expected_metadata"]
    for key, expected in exp.items():
        got = md.get(key)
        if got != expected:
            raise RuntimeError(f"C0 metadata.{key}={got!r} != {expected!r}")
        if key not in ("protocol", "partition_sha256") and int(got) != int(expected):
            raise RuntimeError(f"C0 metadata.{key}={got} != {expected}")
    if int(md["state_optimizer_step"]) != t0 or int(md["train_seed"]) != seed:
        raise RuntimeError("C0 metadata 的 seed/step 与 config 不一致")

    partition = load_partition_checked(partition_path, md["partition_sha256"])
    model_hash_at_c0 = stable_hash(model.state_dict())

    # ---------- 3. 实验臂 ----------
    arm_record = apply_arm(optimizer, arm)
    group_used = optimizer.param_groups[0]
    beta1_used, beta2_used = (float(x) for x in group_used["betas"])
    eps_used = float(group_used["eps"])
    wd_used = float(group_used["weight_decay"])
    if abs(wd_used - float(tr["weight_decay"])) > 0:
        raise RuntimeError(f"weight_decay 与 config 不一致: {wd_used}")

    # ---------- 4. C0 时刻 eval ----------
    eval_indices = [int(i) for i in partition["eval_indices"]]
    eval_c0 = run_eval_nll(model, ds, eval_indices, int(cfg["eval"]["batch_size"]), device, autocast_ctx)

    # ---------- 5. CRN batch 流 ----------
    train_batch_seed = seed + 1704
    needed_micro = (t0 + budget) * accum
    all_batches = batches_from_train_pool(
        partition["train_indices"], int(tr["batch_size"]), train_batch_seed, needed_micro
    )
    cont_batches = all_batches[t0 * accum:]
    if len(cont_batches) != budget * accum:
        raise RuntimeError(f"续训 batch 数不足: {len(cont_batches)} != {budget * accum}")
    from vendored_fr.batch_manifest import manifest_sha256
    batch_manifest_sha = manifest_sha256(cont_batches)
    used_samples = {i for b in cont_batches for i in b}
    forbidden_pool = set(partition["probe_indices"]) | set(eval_indices)
    if used_samples & forbidden_pool:
        raise RuntimeError("续训 batch 与 probe/eval pool 重叠")

    # ---------- 6. 续训循环（原生时序复刻） ----------
    if tr["lr_schedule_mode"] != "original_axis":
        raise RuntimeError(f"未冻结的 schedule mode: {tr['lr_schedule_mode']}")
    total_micro = int(md["total_micro_steps"])
    gm_start = int(md["global_micro_step"])
    if gm_start + budget * accum > total_micro:
        raise RuntimeError("续训超出预注册 LR 轴（该 checkpoint 不适用 original_axis）")

    param_groups_map = build_param_name_groups(model)
    loss_rows, opt_rows = [], []
    window_losses = []
    t_wall = time.time()

    for k, batch_idx in enumerate(cont_batches, start=1):
        gm = gm_start + k
        lr_written = set_native_micro_lr(optimizer, get_lr, gm, total_micro, float(tr["base_lr"]))
        x, y = stack_batch(ds, batch_idx, device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()
        window_losses.append(float(res.loss.detach()))

        if k % accum == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            opt_step = k // accum
            stats = optimizer_state_statistics(
                optimizer, param_groups_map,
                lr=lr_written, beta1=beta1_used, beta2=beta2_used, eps=eps_used,
            )
            base = {
                "opt_step": opt_step,
                "micro_start": gm_start + k - accum + 1,
                "micro_end": gm_start + k,
                "loss_mean": sum(window_losses) / len(window_losses),
                "lr_consumed": lr_written,
                "preclip_grad_norm": float(total_norm),
            }
            loss_rows.append(_row(base, stats))
            for g, gs in sorted(stats["groups"].items()):
                opt_rows.append([opt_step, g] + [gs.get(c) for c in OPT_STATS_COLUMNS[2:]])
            window_losses = []

    wall_seconds = time.time() - t_wall

    # ---------- 7. final eval + 哈希 ----------
    eval_final = run_eval_nll(model, ds, eval_indices, int(cfg["eval"]["batch_size"]), device, autocast_ctx)
    model_hash_final = stable_hash(model.state_dict())
    optimizer_hash_final = stable_hash(optimizer.state_dict())

    # ---------- 8. 落盘 ----------
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOSS_CURVE_COLUMNS)
        for r in loss_rows:
            w.writerow([repr(v) if isinstance(v, float) else v for v in r])
    with (run_dir / "optimizer_state_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OPT_STATS_COLUMNS)
        for r in opt_rows:
            w.writerow([repr(v) if isinstance(v, float) else v for v in r])

    last_k = loss_rows[-8:]
    metrics = {
        "run_id": run_id,
        "arm": arm,
        "config_id": cfg["config_id"],
        "device": device,
        "torch_version": torch.__version__,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "wall_seconds": wall_seconds,
        "c0": {
            "state_path": str(state_path),
            "state_sha256": state_sha,
            "corpus_sha256": corpus_sha,
            "corpus_sha_verified": True,
            "partition_sha256": partition["sha256"],
            "metadata": {k: md[k] for k in (
                "protocol", "train_seed", "state_optimizer_step", "global_micro_step",
                "total_micro_steps", "planned_total_optimizer_steps", "partition_sha256")},
        },
        "hyperparams_used": {
            "betas": [beta1_used, beta2_used], "eps": eps_used,
            "weight_decay": wd_used, "grad_clip": float(tr["grad_clip"]),
            "lr_schedule_mode": tr["lr_schedule_mode"], "base_lr": float(tr["base_lr"]),
            "lr_first": loss_rows[0][LOSS_CURVE_COLUMNS.index("lr_consumed")] if loss_rows else None,
            "lr_last": loss_rows[-1][LOSS_CURVE_COLUMNS.index("lr_consumed")] if loss_rows else None,
        },
        "arm_record": arm_record,
        "batch_manifest_sha256": batch_manifest_sha,
        "n_optimizer_steps": len(loss_rows),
        "n_micro_batches": len(cont_batches),
        "model_hash_at_c0": model_hash_at_c0,
        "model_hash_final": model_hash_final,
        "optimizer_hash_final": optimizer_hash_final,
        "eval_at_c0": eval_c0,
        "eval_final": eval_final,
        "final_train_loss_mean_last8": (
            sum(r[LOSS_CURVE_COLUMNS.index("loss_mean")] for r in last_k) / len(last_k)) if last_k else None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_v01.json")
    ap.add_argument("--lab-root", default=str(SRC_DIR.parent.parent))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-id", required=True, help="A | A_repeat | B | C1 | C2 | D")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    lab_root = Path(args.lab_root).resolve()
    bootstrap(lab_root)
    cfg = json.loads((SRC_DIR.parent / args.config).read_text(encoding="utf-8"))
    metrics = run_one_arm(cfg, lab_root, args.run_id, args.device, Path(args.out_dir))
    print(json.dumps({
        "run_id": metrics["run_id"], "arm": metrics["arm"],
        "n_optimizer_steps": metrics["n_optimizer_steps"],
        "eval_final_mean": metrics["eval_final"]["mean"],
        "batch_manifest_sha256": metrics["batch_manifest_sha256"][:16],
        "model_hash_final": metrics["model_hash_final"][:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
