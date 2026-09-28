"""Test A — P4 schedule mapping 与冻结完整性（docs/P4_PROTOCOL.md §2–§3）。

覆盖全部 7 个 t0（即全部 56 cells 的预算表）：
- gm_first(k=1) = 2·t0 + 2，gm_last(k=2B) = 256；
- 序列长度 = 2B；lr_last = multiplier · 0.1·base_lr；lr 单调不增；
- ratio 恒为 2；预算恒等式 t_remaining = 128−t0、B = t_rem/2；
- 冻结完整性：configs/p4_full.json 的 SHA256 == runner 常量 == 协议文档记录值；
- 臂表三处一致（p4_full.json / smoke_v01.json / state_adapter.ARM_SPECS）。
"""
import _bootstrap  # noqa: F401
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "minimind_base"))

from trainer.trainer_utils import get_lr  # noqa: E402

from p4_runner import (  # noqa: E402
    FROZEN_P4_CONFIG_SHA256, load_p4_config, make_lr_sequence, resolve_cell,
)
from state_adapter import ARM_SPECS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TestScheduleMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_p4_config(ROOT / "configs" / "p4_full.json")

    def test_all_cells_schedule_and_budget(self):
        for cell in self.cfg["checkpoints"]["cells"]:
            t0 = cell["state_optimizer_step"]
            B = cell["budget_optimizer_steps"]
            with self.subTest(cell=cell["cell_id"]):
                self.assertEqual(cell["t_remaining"], 128 - t0)
                self.assertEqual(2 * B, cell["t_remaining"])
                self.assertEqual(cell["budget_micro_steps"], 2 * B)
                for mult in (0.5, 1.0, 2.0, 0.25, 4.0):
                    seq = make_lr_sequence(cell, mult, get_lr, 5e-4)
                    self.assertEqual(len(seq), 2 * B)
                    gm_first, lr_first = seq[0]
                    gm_last, lr_last = seq[-1]
                    self.assertEqual(gm_first, 2 * t0 + 2)
                    self.assertEqual(gm_last, 256)
                    self.assertAlmostEqual(lr_last, mult * 5e-5, places=12)
                    lrs = [lr for _, lr in seq]
                    self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])),
                                    "压缩 cosine 段必须单调不增")

    def test_worked_example_t046(self):
        cell = resolve_cell(self.cfg, "s2030_t046")
        seq = make_lr_sequence(cell, 1.0, get_lr, 5e-4)
        self.assertEqual(cell["budget_optimizer_steps"], 41)
        self.assertEqual(seq[0][0], 94)
        # 期望值直接由冻结公式计算（协议文档示例打印值 3.663e-4 系笔误，
        # 精确值为 3.66179e-4；以公式/config 为准，见评审报告 erratum）
        self.assertAlmostEqual(seq[0][1], get_lr(94, 256, 5e-4), places=15)
        self.assertAlmostEqual(seq[0][1], 3.661792956511227e-4, places=15)
        self.assertEqual(seq[-1][0], 256)
        self.assertAlmostEqual(seq[-1][1], 5e-5, places=12)

    def test_frozen_config_sha_integrity(self):
        got = hashlib.sha256((ROOT / "configs" / "p4_full.json").read_bytes()).hexdigest()
        self.assertEqual(got, FROZEN_P4_CONFIG_SHA256)
        doc = (ROOT / "docs" / "P4_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn(FROZEN_P4_CONFIG_SHA256, doc,
                      "协议文档必须记录冻结 config SHA256")

    def test_arms_tables_consistent(self):
        smoke = json.loads((ROOT / "configs" / "smoke_v01.json").read_text(encoding="utf-8"))
        self.assertEqual(smoke["arms"], self.cfg["arms"])
        check = {k: {"betas": list(v["betas"]),
                     "reset_m": v["reset_m"], "reset_v": v["reset_v"],
                     "reset_step": v["reset_step"]} for k, v in ARM_SPECS.items()}
        self.assertEqual(check, self.cfg["arms"])


if __name__ == "__main__":
    unittest.main()
