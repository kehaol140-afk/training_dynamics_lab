"""Test 3 — CRN reproducibility（docs/EXPERIMENT_PROTOCOL.md §6.3）。

真实 C0（v07 bank seed_2030/state_step_064）：同臂两次独立运行（fresh
model/optimizer 实例）完成首个 optimizer step，要求 loss、pre-clip 梯度范数、
参数/optimizer 哈希逐位一致；batch manifest SHA 一致；C0 时刻 eval 一致。

运行要求：CUDA_VISIBLE_DEVICES 必须指向单卡（C0 的 CUDA RNG 状态按 1 设备保存）。
设备选择：FR_TEST_DEVICE（默认 cpu；cuda:0 用于 GPU 验证）。
"""
import _bootstrap  # noqa: F401
import json
import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LAB_ROOT = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(LAB_ROOT / "minimind_base"))

import torch

from vendored_fr.batch_manifest import manifest_sha256
from vendored_fr.full_state import load_full_training_state
from vendored_fr.hashing import stable_hash
from vendored_fr.minimind_v03_runtime import make_amp, stack_batch
from vendored_fr.native_adamw_guard import assert_native_adamw
from vendored_fr.native_schedule import set_native_micro_lr
from vendored_fr.pilot_partition import load_partition
from vendored_fr.stage_c_runtime import batches_from_train_pool

from evaluation import assert_eval_identical, run_eval_nll
from state_adapter import apply_arm

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "smoke_v01.json").read_text(encoding="utf-8"))
BANK = LAB_ROOT / CFG["lab_root_relative"]["bank_dir"]
STATE_PATH = BANK / CFG["c0"]["state_relpath"]


def _one_first_update(device, arm, world):
    """fresh model/optimizer → load C0 → apply arm → 1 个 optimizer step。"""
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    ds, partition, cont_batches = world

    setup_seed(int(CFG["c0"]["seed"]))
    mmcfg = MiniMindConfig(
        hidden_size=int(CFG["model"]["hidden_size"]),
        num_hidden_layers=int(CFG["model"]["num_hidden_layers"]),
        max_seq_len=int(CFG["model"]["max_seq_len"]),
        use_moe=False,
    )
    model, _ = init_model(
        mmcfg, from_weight="none",
        tokenizer_path=str(LAB_ROOT / "minimind_base" / "model"),
        device=device,
    )
    model.train()
    autocast_ctx, scaler, _ = make_amp(torch, device, CFG["train"]["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CFG["train"]["base_lr"]))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    md = load_full_training_state(STATE_PATH, model, optimizer, scaler=scaler)
    rec = apply_arm(optimizer, arm)

    eval_c0 = run_eval_nll(model, ds, [int(i) for i in partition["eval_indices"]],
                           int(CFG["eval"]["batch_size"]), device, autocast_ctx)

    accum = int(CFG["train"]["accumulation_steps"])
    base_lr = float(CFG["train"]["base_lr"])
    total_micro = int(md["total_micro_steps"])
    gm_start = int(md["global_micro_step"])
    losses, grad_norms = [], []
    for k, batch_idx in enumerate(cont_batches[:accum], start=1):
        lr_written = set_native_micro_lr(optimizer, get_lr, gm_start + k, total_micro, base_lr)
        x, y = stack_batch(ds, batch_idx, device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()
        losses.append(float(res.loss.detach()))
        if k % accum == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(CFG["train"]["grad_clip"]))
            grad_norms.append(float(total_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "lr": lr_written,
        "model_hash": stable_hash(model.state_dict()),
        "optimizer_hash": stable_hash(optimizer.state_dict()),
        "eval_c0": eval_c0,
        "betas": [float(x) for x in optimizer.param_groups[0]["betas"]],
        "arm_record_arm": rec["arm"],
    }


class TestCRNReproducibility(unittest.TestCase):
    def test_first_update_bitwise_reproducible(self):
        device = os.environ.get("FR_TEST_DEVICE", "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.skipTest("CUDA 不可用")

        from dataset.lm_dataset import PretrainDataset
        from trainer.trainer_utils import setup_seed

        subset_path = BANK / CFG["corpus"]["subset_relpath"]
        partition = load_partition(BANK / CFG["corpus"]["partition_relpath"])
        self.assertEqual(partition["sha256"], CFG["c0"]["expected_metadata"]["partition_sha256"])

        from model.model_minimind import MiniMindConfig
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(LAB_ROOT / "minimind_base" / "model"))
        ds = PretrainDataset(str(subset_path), tokenizer, max_length=int(CFG["model"]["max_seq_len"]))

        t0 = int(CFG["c0"]["state_optimizer_step"])
        accum = int(CFG["train"]["accumulation_steps"])
        budget = int(CFG["budget_optimizer_steps"])
        train_batch_seed = int(CFG["c0"]["seed"]) + 1704
        all_batches = batches_from_train_pool(
            partition["train_indices"], int(CFG["train"]["batch_size"]),
            train_batch_seed, (t0 + budget) * accum,
        )
        cont_batches = all_batches[t0 * accum:]
        world = (ds, partition, cont_batches)
        manifest_sha = manifest_sha256(cont_batches)
        self.assertEqual(len(cont_batches), budget * accum)

        for arm in ("A", "B", "C1", "C2", "D"):
            with self.subTest(arm=arm):
                setup_seed(int(CFG["c0"]["seed"]))
                r1 = _one_first_update(device, arm, world)
                r2 = _one_first_update(device, arm, world)
                self.assertEqual(manifest_sha256(cont_batches), manifest_sha)
                self.assertEqual(r1["losses"], r2["losses"], f"{arm}: micro loss 必须逐位一致")
                self.assertEqual(r1["grad_norms"], r2["grad_norms"], f"{arm}: 梯度范数必须逐位一致")
                self.assertEqual(r1["model_hash"], r2["model_hash"], f"{arm}: 参数更新必须逐位一致")
                self.assertEqual(r1["optimizer_hash"], r2["optimizer_hash"], f"{arm}: optimizer state 必须逐位一致")
                assert_eval_identical(r1["eval_c0"], r2["eval_c0"], f"{arm}: eval@c0")
                self.assertEqual(r1["betas"], [float(x) for x in
                    __import__("state_adapter").ARM_SPECS[arm]["betas"]])


if __name__ == "__main__":
    unittest.main()
