"""FR full-state 的 CPU 单元测试。"""
import tempfile
import unittest
from pathlib import Path

import torch

from fr.full_state import load_full_training_state, save_full_training_state
from fr.hashing import stable_hash


class TestFullStateRoundtrip(unittest.TestCase):
    def test_model_optimizer_rng_roundtrip(self):
        torch.manual_seed(123)
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 先做一步，确保 AdamW m/v/step 已创建。
        x = torch.randn(2, 4)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.pt"
            save_full_training_state(p, model, opt, None, {"step": 1})
            model_hash = stable_hash(model.state_dict())
            opt_hash = stable_hash(opt.state_dict())

            # 保存状态之后的“未来随机数”。
            expected_random = torch.randn(5)

            # 故意破坏模型、optimizer 和 RNG。
            y = torch.randn(2, 4)
            bad_loss = model(y).pow(2).mean()
            bad_loss.backward()
            opt.step()
            _ = torch.randn(100)

            meta = load_full_training_state(p, model, opt, None)
            self.assertEqual(meta["step"], 1)
            self.assertEqual(stable_hash(model.state_dict()), model_hash)
            self.assertEqual(stable_hash(opt.state_dict()), opt_hash)
            self.assertTrue(torch.equal(torch.randn(5), expected_random))


if __name__ == "__main__":
    unittest.main()
