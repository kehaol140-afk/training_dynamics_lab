"""p4_runner — P4 冻结协议下的单 cell 单臂续训 runner（docs/P4_PROTOCOL.md §2–§6）。

与 smoke runner（continuation_runner.py）的差异均为 P4 冻结口径：
- 每 cell 预算 B = (128 − t0)/2（changed_budget_half_remaining_v1）；
- LR 变换 compressed_remaining_cosine_v1：第 k 个 micro step 的 LR 取
  gm_lr(k) = 2·t0 + (T_remaining/B)·k（比值恒为 2）处的原生 cosine 值，
  再乘 lr_multiplier；
- 监控评测只用 V_monitor（partition.probe_indices，32 样本，每 8 个
  optimizer step + 终步）；
- **本 runner 不计算 V_final**（结构性封存：封存评测在 Stage 2 由
  scripts/p4_sealed_eval.py 从 final_state.pt 一次性完成）；
- 落盘 metrics.json（无 V_final 字段）、loss_curve.csv、
  optimizer_state_statistics.csv、final_state.pt。

训练循环仍逐行复刻原生时序；CRN：同 cell 全臂共享同一 batch manifest
与同一恢复后的 RNG。加载时校验 configs/p4_full.json 的冻结 SHA256。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

from vendored_fr.full_state import load_full_training_state, save_full_training_state
from vendored_fr.hashing import stable_hash
from vendored_fr.minimind_v03_runtime import make_amp, stack_batch
from vendored_fr.native_adamw_guard import assert_native_adamw
from vendored_fr.native_schedule import set_native_micro_lr
from vendored_fr.stage_c_runtime import batches_from_train_pool

from continuation_runner import LOSS_CURVE_COLUMNS, OPT_STATS_COLUMNS, bootstrap, sha256_file
from evaluation import load_partition_checked, run_eval_nll
from optimizer_utils import build_param_name_groups, optimizer_state_statistics
from p4_selection import run_id_for
from state_adapter import apply_arm, check_arms_table_consistency

FROZEN_P4_CONFIG_SHA256 = "b4716c8efe6c7e490f4eed8f2b419505cc624fc9643730dfd6a5581d723a8fe2"
PLANNED_TOTAL_OPT_STEPS = 128
MONITOR_CADENCE = 8


def load_p4_config(path: Path) -> dict:
    cfg_path = Path(path)
    got = sha256_file(cfg_path)
    if got != FROZEN_P4_CONFIG_SHA256:
        raise RuntimeError(f"p4_full.json SHA256 与冻结值不一致: {got}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def resolve_cell(cfg: dict, cell_id: str) -> dict:
    for cell in cfg["checkpoints"]["cells"]:
        if cell["cell_id"] == cell_id:
            break
    else:
        raise ValueError(f"未知 cell: {cell_id}")
    t0 = int(cell["state_optimizer_step"])
    t_rem = int(cell["t_remaining"])
    budget = int(cell["budget_optimizer_steps"])
    if t_rem != PLANNED_TOTAL_OPT_STEPS - t0 or budget * 2 != t_rem:
        raise RuntimeError(f"cell {cell_id} 预算表与冻结规则不一致")
    if int(cell["budget_micro_steps"]) != 2 * budget:
        raise RuntimeError(f"cell {cell_id} micro 步数不一致")
    return cell


def make_lr_sequence(cell: dict, lr_multiplier: float, get_lr, base_lr: float, accum: int = 2):
    """compressed_remaining_cosine_v1：返回每个 micro step 的 (gm_lr, lr)。

    gm_lr(k) = t0·accum + ratio·k，ratio = T_remaining/B（冻结协议下恒为 2），
    k = 1..2B；lr(k) = multiplier · get_lr(gm_lr(k), 256, base_lr)。
    """
    t0 = int(cell["state_optimizer_step"])
    t_rem = int(cell["t_remaining"])
    budget = int(cell["budget_optimizer_steps"])
    n_micro = int(cell["budget_micro_steps"])
    ratio = t_rem / budget
    if ratio != 2 or n_micro != 2 * budget:
        raise RuntimeError("compressed_remaining_cosine_v1 仅冻结于 ratio=2 情形")
    seq = []
    for k in range(1, n_micro + 1):
        gm = accum * t0 + int(ratio * k)
        lr = float(lr_multiplier) * float(get_lr(gm, PLANNED_TOTAL_OPT_STEPS * accum, base_lr))
        seq.append((gm, lr))
    if seq[-1][0] != PLANNED_TOTAL_OPT_STEPS * accum:
        raise RuntimeError("LR 序列端点不等于 256（协议端点性质被破坏）")
    return seq


def run_one_cell_arm(
    cfg: dict,
    lab_root: Path,
    cell_id: str,
    arm: str,
    lr_multiplier: float,
    device: str,
    out_dir: Path,
) -> dict:
    import torch

    lab_root = Path(lab_root).resolve()
    bootstrap(lab_root)
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    check_arms_table_consistency(cfg["arms"])
    if arm not in cfg["arms"]:
        raise ValueError(f"未知臂: {arm}")

    cell = resolve_cell(cfg, cell_id)
    seed, t0 = int(cell["seed"]), int(cell["state_optimizer_step"])
    budget = int(cell["budget_optimizer_steps"])
    accum = int(cfg["train"]["accumulation_steps"])
    run_id = run_id_for(arm, lr_multiplier)

    bank_dir = lab_root / cfg["lab_root_relative"]["bank_dir"]
    state_path = bank_dir / cell["state_relpath"]
    subset_path = bank_dir / cfg["checkpoints"]["corpus"]["subset_relpath"]

    # ---------- 只读校验 ----------
    state_sha = sha256_file(state_path)
    if state_sha != cell["state_sha256"]:
        raise RuntimeError(f"C0 SHA256 与冻结清单不一致: {state_path}")
    if sha256_file(subset_path) != cfg["checkpoints"]["corpus"]["subset_sha256_frozen"]:
        raise RuntimeError("语料 SHA256 与冻结值不一致")

    # ---------- fresh world + C0 ----------
    setup_seed(seed)
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_hidden_layers=int(cfg["model"]["num_hidden_layers"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        use_moe=bool(cfg["model"]["use_moe"]),
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none",
        tokenizer_path=str(lab_root / cfg["lab_root_relative"]["minimind_base"] / "model"),
        device=device,
    )
    model.train()
    ds = PretrainDataset(str(subset_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, device, cfg["train"]["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["base_lr"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    md = load_full_training_state(state_path, model, optimizer, scaler=scaler)
    if int(md["train_seed"]) != seed or int(md["state_optimizer_step"]) != t0:
        raise RuntimeError("C0 metadata 与 cell 不一致")
    if int(md["global_micro_step"]) != accum * t0 or int(md["total_micro_steps"]) != 256:
        raise RuntimeError("C0 metadata 位置字段与 cell 不一致")

    partition = load_partition_checked(
        bank_dir / cfg["checkpoints"]["corpus"]["partition_relpath"], md["partition_sha256"])
    model_hash_at_c0 = stable_hash(model.state_dict())

    # ---------- 实验臂 ----------
    arm_record = apply_arm(optimizer, arm)
    group = optimizer.param_groups[0]
    beta1_used, beta2_used = (float(x) for x in group["betas"])
    eps_used, wd_used = float(group["eps"]), float(group["weight_decay"])
    if wd_used != float(cfg["train"]["weight_decay"]):
        raise RuntimeError("weight_decay 与冻结值不一致")

    # ---------- CRN batch 流 ----------
    needed_micro = (t0 + budget) * accum
    all_batches = batches_from_train_pool(
        partition["train_indices"], int(cfg["train"]["batch_size"]),
        int(cell["train_batch_seed"]), needed_micro,
    )
    cont_batches = all_batches[t0 * accum:]
    if len(cont_batches) != budget * accum:
        raise RuntimeError("续训 batch 数与预算不一致")
    from vendored_fr.batch_manifest import manifest_sha256
    batch_manifest_sha = manifest_sha256(cont_batches)
    used_samples = {i for b in cont_batches for i in b}
    forbidden_pool = set(partition["probe_indices"]) | set(partition["eval_indices"])
    if used_samples & forbidden_pool:
        raise RuntimeError("续训 batch 与 probe/eval pool 重叠")

    # ---------- LR 序列（compressed_remaining_cosine_v1） ----------
    lr_seq = make_lr_sequence(cell, lr_multiplier, get_lr, float(cfg["train"]["base_lr"]), accum)
    gm_first, lr_first = lr_seq[0]
    gm_last, lr_last = lr_seq[-1]

    def _wrapped_get_lr(current_step, total_steps, lr):
        return float(lr_multiplier) * float(get_lr(current_step, total_steps, lr))

    # ---------- 监控（仅 V_monitor = probe pool） ----------
    probe_indices = [int(i) for i in partition["probe_indices"]]
    cadence = sorted(set(range(MONITOR_CADENCE, budget + 1, MONITOR_CADENCE)) | {budget})
    monitor_points = []

    def _monitor(opt_step: int) -> None:
        nll = run_eval_nll(model, ds, probe_indices, int(cfg["evaluation"]["v_monitor"]["batch_size"]),
                           device, autocast_ctx)
        monitor_points.append({"opt_step": opt_step, **nll})

    # ---------- 续训循环（原生时序） ----------
    param_groups_map = build_param_name_groups(model)
    loss_rows, opt_rows = [], []
    window_losses = []
    t_wall = time.time()

    for k, batch_idx in enumerate(cont_batches, start=1):
        gm_lr_k, _ = lr_seq[k - 1]
        lr_written = set_native_micro_lr(
            optimizer, _wrapped_get_lr, gm_lr_k, PLANNED_TOTAL_OPT_STEPS * accum,
            float(cfg["train"]["base_lr"]),
        )
        x, y = stack_batch(ds, batch_idx, device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()
        window_losses.append(float(res.loss.detach()))

        if k % accum == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            opt_step = k // accum
            if opt_step in cadence:
                _monitor(opt_step)
            stats = optimizer_state_statistics(
                optimizer, param_groups_map,
                lr=lr_written, beta1=beta1_used, beta2=beta2_used, eps=eps_used,
            )
            base = {
                "opt_step": opt_step,
                "micro_start": accum * t0 + k - accum + 1,
                "micro_end": accum * t0 + k,
                "loss_mean": sum(window_losses) / len(window_losses),
                "lr_consumed": lr_written,
                "preclip_grad_norm": float(total_norm),
                "monitor_nll_mean": (monitor_points[-1]["mean"]
                                     if monitor_points and monitor_points[-1]["opt_step"] == opt_step else None),
            }
            loss_rows.append([base.get(c) if c in base else stats.get(c) for c in LOSS_CURVE_COLUMNS + ["monitor_nll_mean"]])
            for g, gs in sorted(stats["groups"].items()):
                opt_rows.append([opt_step, g] + [gs.get(c) for c in OPT_STATS_COLUMNS[2:]])
            window_losses = []

    wall_seconds = time.time() - t_wall
    if not monitor_points or monitor_points[-1]["opt_step"] != budget:
        _monitor(budget)
    final_monitor_nll = monitor_points[-1]["mean"]
    best_monitor_nll = min(p["mean"] for p in monitor_points)

    # ---------- final_state（供 Stage 2 封存评测） ----------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_full_training_state(
        out_dir / "final_state.pt", model, optimizer, scaler,
        metadata={
            "protocol": "CONTINUATION_ADAPTATION_P4_FULL_V01_RUN",
            "config_id": cfg["config_id"],
            "cell_id": cell_id, "seed": seed, "checkpoint_step": t0, "arm": arm,
            "lr_multiplier": float(lr_multiplier), "run_id": run_id,
            "budget_optimizer_steps": budget,
            "batch_manifest_sha256": batch_manifest_sha,
            "model_hash_final": stable_hash(model.state_dict()),
        },
    )
    model_hash_final = stable_hash(model.state_dict())

    # ---------- 落盘（无任何 V_final 字段） ----------
    with (out_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOSS_CURVE_COLUMNS + ["monitor_nll_mean"])
        for r in loss_rows:
            w.writerow([repr(v) if isinstance(v, float) else v for v in r])
    with (out_dir / "optimizer_state_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OPT_STATS_COLUMNS)
        for r in opt_rows:
            w.writerow([repr(v) if isinstance(v, float) else v for v in r])

    metrics = {
        "run_id": run_id,
        "arm": arm,
        "lr_multiplier": float(lr_multiplier),
        "cell_id": cell_id, "seed": seed, "checkpoint_step": t0,
        "budget_optimizer_steps": budget, "n_optimizer_steps": len(loss_rows),
        "n_micro_batches": len(cont_batches),
        "device": device, "torch_version": torch.__version__,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "wall_seconds": wall_seconds,
        "c0": {"state_path": str(state_path), "state_sha256": state_sha,
               "corpus_sha256": cfg["checkpoints"]["corpus"]["subset_sha256_frozen"],
               "partition_sha256": partition["sha256"]},
        "schedule": {"mode": "compressed_remaining_cosine_v1", "ratio": 2,
                     "gm_first": gm_first, "gm_last": gm_last,
                     "lr_first": lr_first, "lr_last": lr_last},
        "hyperparams_used": {"betas": [beta1_used, beta2_used], "eps": eps_used,
                             "weight_decay": wd_used,
                             "grad_clip": float(cfg["train"]["grad_clip"]),
                             "base_lr": float(cfg["train"]["base_lr"])},
        "arm_record": arm_record,
        "batch_manifest_sha256": batch_manifest_sha,
        "model_hash_at_c0": model_hash_at_c0,
        "model_hash_final": model_hash_final,
        "optimizer_hash_final": stable_hash(optimizer.state_dict()),
        "final_monitor_nll": final_monitor_nll,
        "best_monitor_nll": best_monitor_nll,
        "monitor_points": monitor_points,
    }
    blob = json.dumps(metrics, ensure_ascii=False, indent=2)
    for forbidden in ("v_final", "V_final", "eval_final", "final_eval"):
        if forbidden in blob:
            raise RuntimeError(f"metrics.json 含禁用字段 {forbidden}")
    (out_dir / "metrics.json").write_text(blob, encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SRC_DIR.parent / "configs" / "p4_full.json"))
    ap.add_argument("--lab-root", default=str(SRC_DIR.parent.parent))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arm", required=True, choices=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--lr-multiplier", type=float, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_p4_config(Path(args.config))
    metrics = run_one_cell_arm(cfg, Path(args.lab_root), args.cell, args.arm,
                               args.lr_multiplier, args.device, Path(args.out_dir))
    print(json.dumps({
        "run_id": metrics["run_id"], "cell": metrics["cell_id"],
        "n_steps": metrics["n_optimizer_steps"],
        "final_monitor_nll": metrics["final_monitor_nll"],
        "lr_last": metrics["schedule"]["lr_last"],
        "model_hash_final": metrics["model_hash_final"][:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
