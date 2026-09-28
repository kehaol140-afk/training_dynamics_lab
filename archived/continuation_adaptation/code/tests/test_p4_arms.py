"""Test B — P4 路径下的 beta/state 施加验证（docs/P4_PROTOCOL.md §4）。

用真实 C0（v07 bank seed_2030/state_step_064，CPU 加载）走 P4 runner 同款
加载序列（load_full_training_state → apply_arm），逐臂验证 optimizer 内部状态：
- B：betas=(0.9,0.99)，m/v/step 逐位保留；
- C1：v 精确为 0，m/step 保留；C2：m 精确为 0，v/step 保留；
- D：m=v=0 且 step=0；
- A：state 全保留且 betas 保持 (0.9,0.999)。
"""
import _bootstrap  # noqa: F401
import json
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LAB_ROOT = Path(__file__).resolve().parents[2]
for p in (str(SRC), str(LAB_ROOT / "minimind_base")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from vendored_fr.full_state import load_full_training_state  # noqa: E402
from vendored_fr.hashing import stable_hash  # noqa: E402
from vendored_fr.minimind_v03_runtime import make_amp  # noqa: E402
from vendored_fr.native_adamw_guard import assert_native_adamw  # noqa: E402

from p4_runner import load_p4_config  # noqa: E402
from state_adapter import _state_channel_hashes, apply_arm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_p4_config(ROOT / "configs" / "p4_full.json")
STATE_PATH = (LAB_ROOT / CFG["lab_root_relative"]["bank_dir"]
              / "seed_2030/states/state_step_064.pt")


def _loaded_optimizer():
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import init_model, setup_seed

    setup_seed(2030)
    mmcfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, max_seq_len=128, use_moe=False)
    model, _ = init_model(mmcfg, from_weight="none",
                          tokenizer_path=str(LAB_ROOT / "minimind_base" / "model"),
                          device="cpu")
    _, scaler, _ = make_amp(torch, "cpu", "bfloat16")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)
    load_full_training_state(STATE_PATH, model, optimizer, scaler=scaler)
    return model, optimizer


class TestP4ArmStates(unittest.TestCase):
    def test_arms_on_real_c0(self):
        for arm, expect_betas in (("A", (0.9, 0.999)), ("B", (0.9, 0.99)),
                                  ("C1", (0.9, 0.99)), ("C2", (0.9, 0.99)),
                                  ("D", (0.9, 0.99))):
            with self.subTest(arm=arm):
                model, optimizer = _loaded_optimizer()
                pre = _state_channel_hashes(optimizer)
                apply_arm(optimizer, arm)

                got_betas = tuple(float(x) for x in optimizer.param_groups[0]["betas"])
                self.assertEqual(got_betas, expect_betas)

                post = _state_channel_hashes(optimizer)
                m_zero = v_zero = step_zero = True
                m_kept = v_kept = step_kept = True
                for p, entry in optimizer.state.items():
                    m_zero &= bool(entry["exp_avg"].eq(0).all())
                    v_zero &= bool(entry["exp_avg_sq"].eq(0).all())
                    step_zero &= (float(entry["step"]) == 0.0)
                m_kept = (pre["m"] == post["m"])
                v_kept = (pre["v"] == post["v"])
                step_kept = (pre["step"] == post["step"])

                if arm == "A":
                    self.assertTrue(m_kept and v_kept and step_kept)
                    self.assertFalse(m_zero or v_zero or step_zero)
                elif arm == "B":
                    self.assertTrue(m_kept and v_kept and step_kept)
                elif arm == "C1":
                    self.assertTrue(v_zero and m_kept and step_kept)
                    self.assertFalse(m_zero or step_zero)
                elif arm == "C2":
                    self.assertTrue(m_zero and v_kept and step_kept)
                    self.assertFalse(v_zero or step_zero)
                else:  # D
                    self.assertTrue(m_zero and v_zero and step_zero)
                    self.assertFalse(m_kept or v_kept or step_kept)

    def test_c1_bias_correction_semantics_visible(self):
        """C1（v 清零、step 保留 t0=64）首步 v̂ 的定义性行为可观测。"""
        model, optimizer = _loaded_optimizer()
        apply_arm(optimizer, "C1")
        g = 0.5
        p = next(model.parameters())
        optimizer.zero_grad(set_to_none=True)
        p.grad = torch.full_like(p, g)
        optimizer.step()
        entry = optimizer.state[p]
        t = float(entry["step"])  # 65
        expected_v = (1 - 0.99) * g * g  # v = 0.99*0 + 0.01*g^2
        self.assertTrue(torch.allclose(entry["exp_avg_sq"], torch.full_like(p, expected_v)))
        vhat = expected_v / (1 - 0.99 ** t)
        self.assertGreater(vhat, expected_v,
                           "step 保留时 (1-0.99^65)≈0.476<1，v̂ = v/0.476 > v → 有效步长瞬态（协议定义）")


if __name__ == "__main__":
    unittest.main()
