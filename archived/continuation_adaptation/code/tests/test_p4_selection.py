"""p4_selection 冻结选择规则的单测（docs/P4_PROTOCOL.md §5）。

- 内点最优：无扩张；
- 边界最优：needs_expansion 恰为对应扩张值（0.5→0.25，2→4），且仅一次；
- 扩张后选中 0.25/4 → boundary_unresolved=True；
- 并列（≤1e-6）：优先 1.0x，再取较小乘子；
- 表外乘子直接拒绝。
"""
import _bootstrap  # noqa: F401
import unittest

from p4_selection import select_best


class TestSelection(unittest.TestCase):
    def test_interior_best_no_expansion(self):
        r = {0.5: 7.10, 1.0: 7.00, 2.0: 7.05}
        sel = select_best(r)
        self.assertEqual(sel["selected_multiplier"], 1.0)
        self.assertIsNone(sel["needs_expansion"])
        self.assertFalse(sel["boundary_unresolved"])

    def test_low_boundary_triggers_expansion_once(self):
        r = {0.5: 6.90, 1.0: 7.00, 2.0: 7.10}
        sel = select_best(r)
        self.assertEqual(sel["selected_multiplier"], 0.5)
        self.assertEqual(sel["needs_expansion"], 0.25)
        self.assertFalse(sel["boundary_unresolved"])

        # 补跑 0.25 后重选：0.25 更差 → 0.5 胜，resolved
        r[0.25] = 6.95
        sel2 = select_best(r)
        self.assertEqual(sel2["selected_multiplier"], 0.5)
        self.assertIsNone(sel2["needs_expansion"])
        self.assertFalse(sel2["boundary_unresolved"])

        # 补跑 0.25 后 0.25 仍最优 → unresolved（接受，不再扩格）
        r[0.25] = 6.85
        sel3 = select_best(r)
        self.assertEqual(sel3["selected_multiplier"], 0.25)
        self.assertIsNone(sel3["needs_expansion"])
        self.assertTrue(sel3["boundary_unresolved"])

    def test_high_boundary_expansion(self):
        r = {0.5: 7.20, 1.0: 7.10, 2.0: 6.80}
        self.assertEqual(select_best(r)["needs_expansion"], 4.0)
        r[4.0] = 6.70
        sel = select_best(r)
        self.assertEqual(sel["selected_multiplier"], 4.0)
        self.assertTrue(sel["boundary_unresolved"])

    def test_tie_rule_prefers_1_then_smaller(self):
        # 差 4e-7 ≤ 1e-6 → 并列 → 优先 1.0x（内点，无扩张）
        r = {0.5: 6.9000000, 1.0: 6.9000004, 2.0: 7.0}
        sel = select_best(r)
        self.assertEqual(sel["selected_multiplier"], 1.0)
        self.assertIsNone(sel["needs_expansion"])

        # 无并列且 1.0 落败 → 选较小乘子 0.5（边界，但 0.25 已运行 → 不再扩格）
        r3 = {0.5: 6.89, 1.0: 6.90, 2.0: 7.2, 0.25: 7.1}
        sel3 = select_best(r3)
        self.assertEqual(sel3["selected_multiplier"], 0.5)
        self.assertIsNone(sel3["needs_expansion"])
        self.assertFalse(sel3["boundary_unresolved"])

    def test_tie_without_1x_prefers_smaller(self):
        r = {0.5: 6.80, 2.0: 6.80}  # 1.0 缺席并列：取较小（0.5，边界 → 扩张）
        sel = select_best(r)
        self.assertEqual(sel["selected_multiplier"], 0.5)
        self.assertEqual(sel["needs_expansion"], 0.25)

    def test_illegal_multiplier_rejected(self):
        with self.assertRaises(ValueError):
            select_best({0.5: 7.0, 1.0: 7.1, 3.0: 6.0})


if __name__ == "__main__":
    unittest.main()
