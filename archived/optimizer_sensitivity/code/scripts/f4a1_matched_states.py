#!/usr/bin/env python3
"""F4A.1: reconstruct v07 5M matched states (frozen:
docs/F4A1_MATCHED_COVERAGE_PREREG.md; recipe audit:
docs/F4A1_5M_RECONSTRUCTION_AUDIT.md).

Line-faithful re-implementation of the legacy v07 training loop
(e3c_prepare_state_bank_v07.py), importing the legacy fr package and
minimind_base READ-ONLY. Writes new checkpoints to
outputs/f4a1_matched_coverage/new_states/ (never touches the legacy bank).

Usage (CUDA_VISIBLE_DEVICES set):
  python scripts/f4a1_matched_states.py train --seed 2030 --steps 26,64,102
  python scripts/f4a1_matched_states.py envelope   # original-bank envelope table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
MM = LAB / "minimind_base"
LEG = LAB / "legacy_fluctuation_response"
for p in (str(ROOT / "src"), str(MM), str(LEG / "research" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from model.model_minimind import MiniMindConfig  # noqa: E402
from trainer.trainer_utils import get_lr, init_model, setup_seed  # noqa: E402
from dataset.lm_dataset import PretrainDataset  # noqa: E402
from fr.full_state import save_full_training_state  # noqa: E402
from fr.minimind_v03_runtime import make_amp, stack_batch  # noqa: E402
from fr.native_schedule import set_native_micro_lr  # noqa: E402
from fr.native_adamw_guard import assert_native_adamw  # noqa: E402
from fr.stage_c_runtime import batches_from_train_pool  # noqa: E402
from fr.pilot_partition import load_partition  # noqa: E402

OUT = ROOT / "outputs" / "f4a1_matched_coverage"
OUT.mkdir(parents=True, exist_ok=True)
NEW = OUT / "new_states"
CFG = dict(hidden_size=256, num_hidden_layers=4, max_seq_len=128,
           batch_size=4, accumulation_steps=2, learning_rate=5e-4,
           grad_clip=1.0, dtype="bfloat16", planned_total_optimizer_steps=128)
DATA = LEG / "outputs" / "e3_confirmatory" / "v07" / "e3c_subset.jsonl"
PARTITION = LEG / "outputs" / "e3_confirmatory" / "v07" / "partition_v07.json"


def partition_path() -> Path:
    if PARTITION.exists():
        return PARTITION
    from opsens.f1_mechanism_v01 import fm  # noqa
    raise SystemExit("partition not found")


def cmd_train(seed: int, steps_str: str, device: str = "cuda:0") -> int:
    steps = sorted(int(x) for x in steps_str.split(","))
    save_dir = NEW / "v07_5m" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in steps
            if not (save_dir / f"state_step_{t:03d}.pt").exists()]
    if not todo:
        print(f"[f4a1] seed {seed}: all requested states exist, skip")
        return 0
    setup_seed(seed)
    mmcfg = MiniMindConfig(hidden_size=CFG["hidden_size"],
                           num_hidden_layers=CFG["num_hidden_layers"],
                           max_seq_len=CFG["max_seq_len"], use_moe=False)
    model, tokenizer = init_model(mmcfg, from_weight="none",
                                  tokenizer_path=str(MM / "model"), device=device)
    model.train()
    ds = PretrainDataset(str(DATA), tokenizer, max_length=CFG["max_seq_len"])
    autocast_ctx, scaler, _ = make_amp(torch, device, CFG["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(CFG["learning_rate"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)
    accum = CFG["accumulation_steps"]
    total_opt = CFG["planned_total_optimizer_steps"]
    total_micro = total_opt * accum
    part = load_partition(partition_path())
    train_batch_seed = seed + 1704
    max_micro = max(todo) * accum
    batches = batches_from_train_pool(part["train_indices"],
                                      CFG["batch_size"], train_batch_seed,
                                      max_micro)
    done = set()
    for micro_idx, batch_idx in enumerate(batches, start=1):
        set_native_micro_lr(optimizer, get_lr, micro_idx, total_micro,
                            CFG["learning_rate"])
        x, y = stack_batch(ds, batch_idx, device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()
        if micro_idx % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(CFG["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_step = micro_idx // accum
            if opt_step in todo:
                path = save_dir / f"state_step_{opt_step:03d}.pt"
                save_full_training_state(
                    path, model, optimizer, scaler,
                    metadata={
                        "protocol": "F4A1_MATCHED_COVERAGE_V07",
                        "train_seed": seed,
                        "state_optimizer_step": opt_step,
                        "global_optimizer_step": opt_step,
                        "global_micro_step": micro_idx,
                        "planned_total_optimizer_steps": total_opt,
                        "total_micro_steps": total_micro,
                        "partition_sha256": part["sha256"],
                        "config": {**CFG, "source_recipe":
                                   "e3c_prepare_state_bank_v07.py (read-only)"},
                    })
                done.add(opt_step)
                print(f"[f4a1] seed {seed}: saved state_step_{opt_step:03d} "
                      f"(loss={float(res.loss):.5f})", flush=True)
                if done == set(todo):
                    break
    print(f"[f4a1] seed {seed}: COMPLETE {sorted(done)}")
    return 0


def cmd_envelope() -> int:
    """Empirical envelope of the ORIGINAL 8-seed bank at step 64 (canary
    criteria inputs, straight from F4-A metrics)."""
    import csv
    vals = {"R_curv_32": [], "lambda_SO_32": [], "loss_h1": [],
            "fraction_sqrtv_lt_10eps": []}
    for m in (ROOT / "outputs" / "f4a_phase_map" / "raw" / "v07_5m").glob(
            "seed_*/step_64/metrics.json"):
        d = json.loads(m.read_text())
        vals["R_curv_32"].append(d["G"]["32"]["R_curv"])
        vals["lambda_SO_32"].append(d["G"]["32"]["lambda_SO"])
        vals["loss_h1"].append(d["G"]["1"]["S_eta"] and
                               json.loads((m.parent / "trajectory.csv").
                                          read_text().splitlines()[1]
                                          .split(",")[1]) if False else None)
        traj = list(csv.DictReader((m.parent / "trajectory.csv").open()))
        vals["loss_h1"][-1] = float(traj[0]["loss"])
        vals["fraction_sqrtv_lt_10eps"].append(
            d["indicators"]["fraction_sqrtv_lt_10eps"])
    import numpy as np
    out = {}
    for k, v in vals.items():
        out[k] = {"min": float(np.min(v)), "max": float(np.max(v)),
                  "median": float(np.median(v)), "n": len(v)}
    (OUT / "canary_envelope.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "envelope"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", default="26,102")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.cmd == "envelope":
        return cmd_envelope()
    assert args.seed is not None
    return cmd_train(args.seed, args.steps, args.device)


if __name__ == "__main__":
    sys.exit(main())
