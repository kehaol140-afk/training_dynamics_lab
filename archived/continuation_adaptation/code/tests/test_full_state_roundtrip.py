"""Test 1 — Full state roundtrip（docs/EXPERIMENT_PROTOCOL.md §6.1）。

验证 vendored full_state.save/load：save → 破坏（继续训练 + 消耗四路 RNG）→
load 后 model / optimizer 哈希逐位一致，且未来随机数逐位重放。

注意：若 state 含 CUDA RNG（GPU 变体），恢复要求当前可见 CUDA 设备数与保存时
一致（=1），故 GPU 变体需以 CUDA_VISIBLE_DEVICES=<单卡> 运行，由
FR_RUN_CUDA_TESTS=1 门控。
"""
import _bootstrap  # noqa: F401
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vendored_fr.full_state import capture_rng_state, load_full_training_state, restore_rng_state, save_full_training_state
from vendored_fr.hashing import stable_hash


def _make_world(device):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def _train_steps(model, opt, device, n, seed):
    torch.manual_seed(seed)
    for _ in range(n):
        x = torch.randn(32, 8, device=device)
        y = torch.randn(32, 4, device=device)
        opt.zero_grad(set_to_none=True)
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    return model


def _consume_rng():
    random.random()
    np.random.rand()
    _ = torch.randn(3)
    if torch.cuda.is_available():
        _ = torch.randn(3, device="cuda")


def _future_samples():
    out = [random.random(), float(np.random.rand()), float(torch.randn(1))]
    if torch.cuda.is_available():
        out.append(float(torch.randn(1, device="cuda")))
    return out


class TestFullStateRoundtripCPU(unittest.TestCase):
    DEVICE = "cpu"

    def test_roundtrip_bitwise(self):
        device = self.DEVICE
        model, opt = _make_world(device)
        _train_steps(model, opt, device, 2, 11)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.pt"
            st = save_full_training_state(path, model, opt, metadata={"pos": "t"})
            h_model_saved = stable_hash(st["model"])
            h_opt_saved = stable_hash(st["optimizer"])

            # ---- 破坏：继续训练 + 消耗 RNG ----
            _train_steps(model, opt, device, 1, 12)
            _consume_rng()
            self.assertNotEqual(stable_hash(model.state_dict()), h_model_saved)

            # ---- 恢复：fresh 实例 ----
            model2, opt2 = _make_world(device)
            _train_steps(model2, opt2, device, 1, 13)  # 先弄脏
            md = load_full_training_state(path, model2, opt2)
            self.assertEqual(md, {"pos": "t"})

            self.assertEqual(stable_hash(model2.state_dict()), h_model_saved)
            self.assertEqual(stable_hash(opt2.state_dict()), h_opt_saved)

            # ---- 未来随机数逐位重放 ----
            snap = capture_rng_state()
            s1 = _future_samples()
            restore_rng_state(snap)
            s2 = _future_samples()
            self.assertEqual(s1, s2)


class TestFullStateRoundtripCUDA(TestFullStateRoundtripCPU):
    DEVICE = "cuda:0"

    def setUp(self):
        import os
        if os.environ.get("FR_RUN_CUDA_TESTS", "0") != "1" or not torch.cuda.is_available():
            self.skipTest("set FR_RUN_CUDA_TESTS=1 and a single CUDA_VISIBLE_DEVICES to run")


if __name__ == "__main__":
    print(json.dumps({"note": "run via: python -m unittest discover -s tests -v"}))
    unittest.main()
