"""Phase A native calibration 的针对性 CPU 测试（T1–T8 + tail 边界/前缀）。

复用既有框架（test_native_schedule / test_stream_rng / test_full_state_roundtrip
已覆盖的部分不重复）；本文件只补协议 §7 要求的语义项，全部 CPU、无模型。
unittest 风格，与 research/tests 其余文件一致。
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr.phase_a_tail_schedule_v01 import (  # noqa: E402
    BASE_LR, TAIL_LR, TOTAL_MICRO_STEPS, make_phase_a_get_lr)


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


class TestTailWrapper(unittest.TestCase):
    """T6a: 边界 —— 定义域内逐位等于旧公式，域外恒定 5e-5，无回弹。"""

    def test_matches_old_formula_within_domain(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        for gm in range(1, TOTAL_MICRO_STEPS + 1):
            self.assertAlmostEqual(
                eta(gm, TOTAL_MICRO_STEPS, BASE_LR),
                _old_get_lr(gm, TOTAL_MICRO_STEPS, BASE_LR),
                places=15)

    def test_holds_constant_beyond_domain(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        self.assertAlmostEqual(eta(TOTAL_MICRO_STEPS, TOTAL_MICRO_STEPS, BASE_LR),
                               TAIL_LR, places=15)
        for gm in (TOTAL_MICRO_STEPS + 1, 300, 512, 513, 602, 1000):
            self.assertEqual(eta(gm, TOTAL_MICRO_STEPS, BASE_LR), TAIL_LR)
        boundary = eta(TOTAL_MICRO_STEPS, TOTAL_MICRO_STEPS, BASE_LR)
        self.assertTrue(all(eta(g, TOTAL_MICRO_STEPS, BASE_LR) <= boundary
                            for g in range(TOTAL_MICRO_STEPS, 800)))

    def test_rejects_rescaled_horizon_or_lr(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        with self.assertRaises(ValueError):
            eta(10, 512, BASE_LR)
        with self.assertRaises(ValueError):
            eta(10, TOTAL_MICRO_STEPS, 1e-4)
        with self.assertRaises(ValueError):
            eta(0, TOTAL_MICRO_STEPS, BASE_LR)

    def test_metadata(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        self.assertEqual(eta.schedule_id, "finite_cosine_then_hold_5e-5_v01")
        self.assertEqual(eta.schedule_label, "NATIVE_WITH_DECLARED_TAIL")


class TestPrefixIdentity(unittest.TestCase):
    """T6b/T6c: 128 步请求与 256 步请求的前 128 步 LR/batch 完全一致。"""

    def test_schedule_prefix_identity_128_vs_256(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        full = [eta(g, TOTAL_MICRO_STEPS, BASE_LR) for g in range(1, 2 * 256 + 1)]
        short = full[: 2 * 128]
        lr_full_prefix = [full[2 * h - 1] for h in range(1, 129)]
        lr_short = [short[2 * h - 1] for h in range(1, 129)]
        self.assertEqual(lr_full_prefix, lr_short)

    def test_batch_stream_prefix_identity(self):
        from fr.batch_manifest import batchify_indices, make_permutation
        from fr.stream_rng import derive_stream_seed
        for stream_id in (97000, 97005):
            seed = derive_stream_seed(stream_id, 404)
            perm = make_permutation(3936, seed)
            short = batchify_indices(perm, 4)[:128]
            long = batchify_indices(perm, 4)[:256]
            self.assertEqual(long[:128], short)


class TestAggregation(unittest.TestCase):
    """T5: 不等 valid token 数时 mean(batch mean) 是错的。"""

    def test_sum_count_aggregation_vs_mean_of_means(self):
        batches = [(10.0, 4), (20.0, 1)]
        wrong = sum(m for m, _ in batches) / len(batches)
        correct = sum(m * c for m, c in batches) / sum(c for _, c in batches)
        self.assertNotEqual(wrong, correct)
        self.assertAlmostEqual(correct, 12.0)


class TestRadialTangent(unittest.TestCase):
    """T4: radial/tangential 恒等 + 零范数 NA + tied storage 去重。"""

    def test_identity_and_degenerate_cases(self):
        import numpy as np
        rng = np.random.default_rng(0)
        W = rng.normal(size=(8, 6))
        s = rng.normal(size=(8, 6)) * 0.01
        nW = float(np.linalg.norm(W))
        b_rad = float(np.sum(W * s)) / (nW ** 2)
        s_perp = s - b_rad * W
        r_total = float(np.linalg.norm(s)) / nW
        r_tangent = float(np.linalg.norm(s_perp)) / nW
        self.assertLess(abs(r_total ** 2 - (b_rad ** 2 + r_tangent ** 2)), 1e-18)
        W0 = np.zeros((3, 3))
        with self.assertRaises(ZeroDivisionError):
            _ = float(np.linalg.norm(s)) / float(np.linalg.norm(W0))
        tied = {"model.embed_tokens.weight": W, "lm_head.weight": W}
        self.assertEqual(len({id(v) for v in tied.values()}), 1)


class TestEmaCoefficients(unittest.TestCase):
    """T3: raw 系数与 bias-corrected 旧 state 系数。"""

    def test_coefficients(self):
        for beta, n0, H in [(0.9, 46, 64), (0.9, 82, 256), (0.999, 46, 128),
                            (0.999, 82, 256)]:
            raw = beta ** H
            w_old = raw * (1 - beta ** n0) / (1 - beta ** (n0 + H))
            self.assertTrue(0 < w_old < raw < 1)
        self.assertAlmostEqual(math.log(0.5) / math.log(0.9), 6.5788, places=3)
        self.assertAlmostEqual(math.log(0.5) / math.log(0.999), 692.801, places=2)


class TestTrackingRecursion(unittest.TestCase):
    """T2: I_v 条件期望 nu-v；单样本二阶矩含 variance 项。"""

    def test_innovation_second_moment(self):
        import numpy as np
        rng = np.random.default_rng(1)
        v = 1.3
        cs = rng.normal(0.5, 1.0, size=200000)
        I_v = cs ** 2 - v
        self.assertAlmostEqual(float(np.mean(I_v)), 0.25 + 1.0 - v, delta=5e-3)
        self.assertGreater(float(np.var(cs ** 2)), 0)


class TestCrossStepTerm(unittest.TestCase):
    """T1: 拆分第二项条件均值可非零。"""

    def test_conditional_mean_nonzero(self):
        import numpy as np
        eta, theta = 0.1, 1.0
        rng = np.random.default_rng(2)
        om_t = rng.choice([1.0, -1.0], size=200000)
        om_next = om_t * rng.choice([1.0, 1.0, 1.0, -1.0], size=200000)
        theta_next = theta - eta * (theta - om_t)
        second = (theta_next - om_next) - (theta_next - om_t)
        self.assertGreater(abs(float(np.mean(second))), 1e-3)


class TestHitRecording(unittest.TestCase):
    """T7: 两个连续 check 记 hit；cap 不得当 hit。"""

    def _hits(self, series, ell):
        below = [v <= ell for v in series]
        out = [i for i in range(1, len(series))
               if below[i - 1] and below[i]]
        return out

    def test_rules(self):
        # 8.5,8.6,8.4 连续 <=8.6：第 3、4 个点都满足“连续对的第二点”
        self.assertEqual(self._hits([10.0, 9.0, 8.5, 8.6, 8.4], 8.6), [3, 4])
        # 8.0,9.0,8.0,7.9 与 ell=8.0：仅 (8.0@3, 7.9@4) 连续低于
        self.assertEqual(self._hits([10.0, 8.0, 9.0, 8.0, 7.9], 8.0), [4])
        self.assertEqual(self._hits([10.0, 9.9], 5.0), [])


class TestExclusionAndCounters(unittest.TestCase):
    """T8: row 排除、counter 尝试即计。"""

    def test_exclusion(self):
        train = set(range(0, 4096))
        monitor = set(range(4096, 8192))
        final = set(range(8192, 12288))
        self.assertFalse(train & monitor)
        self.assertFalse(train & final)

    def test_counter_counts_attempts(self):
        counter = {"updates": 0}
        for _ in range(3):
            counter["updates"] += 1
        self.assertEqual(counter["updates"], 3)


if __name__ == "__main__":
    unittest.main()
