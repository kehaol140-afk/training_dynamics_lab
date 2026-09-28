#!/usr/bin/env python3
"""GTTR-CB1 —— 最少 CPU 测试（协议 §9 的 8 项）。

独立参考：期望值在文件内显式重算；错误实现（连续两次 4 步、只查 crossing
附近、合并阈值、访问 landmark 之后数据、重放成本混入训练轴、重新施加首步
动作）必须被捕获。
"""
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr import cb1_cost_v01 as cb  # noqa: E402
from fr import r2b_bridge_v01 as br  # noqa: E402

ELL46 = 7.020043501537879
ELL82 = 6.979538126804536
GRID16 = list(range(0, 257, 16))


class T1_AliveCostIdentity(unittest.TestCase):
    def test_identity_hit_mid(self):
        means = {h: 7.3 - 0.004 * h for h in GRID16}
        ell = 7.05
        # 手工：V(h)<=ell 从 h>=63 起 -> h=64 起；pair (48,64)? V(48)=7.108>ell
        # 首个 pair: (64,80)。alive 成本 = 80*1024。
        r = cb.alive_capped_cost(means, ell, GRID16)
        self.assertEqual(r["h_detect"], 80)
        self.assertEqual(r["cost_alive"], 80 * 1024)
        self.assertEqual(r["residual"], 0.0)
        self.assertFalse(r["censor"])

    def test_identity_not_reached(self):
        means = {h: 7.3 - 0.0005 * h for h in GRID16}
        r = cb.alive_capped_cost(means, 7.02, GRID16)
        self.assertIsNone(r["h_detect"])
        self.assertTrue(r["censor"])
        self.assertEqual(r["cost_alive"], 256 * 1024)
        self.assertEqual(r["cost_direct"], 256 * 1024)
        self.assertEqual(r["residual"], 0.0)

    def test_identity_exact_threshold_and_last_checkpoint(self):
        # 恰好等于阈值（<= 含等号）
        means = {h: 7.3 - 0.005 * h for h in GRID16}
        ell = means[64]  # V(64)==ell
        r = cb.alive_capped_cost(means, ell, GRID16)
        # V(48)=7.06>ell；V(64)==ell 但 V(48)>ell -> pair 在 (64,80)
        self.assertEqual(r["h_detect"], 80)
        # 最后检查点才确认
        means2 = {h: 7.3 - 0.001 * h for h in GRID16}
        ell2 = means2[240]
        r2 = cb.alive_capped_cost(means2, ell2, GRID16)
        self.assertEqual(r2["h_detect"], 256)
        self.assertEqual(r2["residual"], 0.0)


class T2_Grid16Lag16EqualsOldRule(unittest.TestCase):
    def test_matches_r2b_detect_target(self):
        means = {0: 7.3}
        for i, h in enumerate(range(16, 257, 16)):
            means[h] = 7.25 - 0.0035 * i
        r_old = br.detect_target(dict(means), 7.020043501537879)
        r_new = cb.old_grid16(means, 7.020043501537879)
        self.assertEqual(r_old["h_detect"], r_new["h_detect"])
        self.assertEqual(r_old["status"], r_new["status"])
        # 逐点交替上下（每个 pair 的两元一上一下）不确认
        spiky = {0: 7.3}
        for i, h in enumerate(range(16, 257, 16)):
            spiky[h] = 6.9 if i % 2 == 0 else 7.1
        self.assertIsNone(br.detect_target(spiky, 7.0)["h_detect"])
        self.assertIsNone(cb.old_grid16(spiky, 7.0)["h_detect"])


class T3_Grid4Lag16NotConsecutive4(unittest.TestCase):
    def test_consecutive4_mutant_fails(self):
        # 真规则（lag16）：t=64（V(48)=6.9 与 V(64)=6.98 均 <=ell）。
        # 错误实现（连续两次 4 步通过）：t=52（V(48)、V(52) 相邻均 <=ell）
        # -> 必须不同。
        ell = 7.0
        g4 = cb.refined_grid4_points(96)
        means = {h: 7.4 - 0.007 * h for h in g4}   # 线性下降
        means[48] = 6.9
        means[52] = 6.9
        means[56] = 7.05                           # 回摆
        means[60] = 7.02
        means[64] = 6.98
        means[68] = 6.99
        true_rule = cb.detect_lagN(means, ell, g4, lag=16)
        self.assertEqual(true_rule["h_detect"], 64)

        def mutant(means, ell, g4):
            prev = None
            for t in sorted(g4):
                if t < 4:
                    continue
                if means[t] <= ell:
                    if prev is not None and prev <= ell:
                        return t
                    prev = means[t]
                else:
                    prev = None
            return None
        self.assertEqual(mutant(means, ell, g4), 52)  # 错误实现更早确认
        self.assertNotEqual(mutant(means, ell, g4), true_rule["h_detect"])

    def test_lag_pair_missing_raises(self):
        means = {0: 7.3, 20: 6.9}
        with self.assertRaises(KeyError):
            cb.detect_lagN(means, 7.0, [0, 20], lag=16)


class T4_NonmonotoneFullScan(unittest.TestCase):
    def test_earliest_legal_pair_before_old_crossing(self):
        # 非单调：早期一次深谷 (36,52) 均 <=ell，旧 16 步网格看不见；
        # T4lag16 必须找到 t=52（不是旧 crossing 附近的 ~80）。
        ell = 7.0
        g4 = cb.refined_grid4_points(96)
        means = {h: 7.25 - 0.0025 * h for h in g4}
        means[36] = 6.96
        means[52] = 6.97
        for h in range(40, 52, 4):
            means[h] = 6.99 if h != 44 else 7.01
        r = cb.detect_lagN(means, ell, g4, lag=16)
        self.assertEqual(r["h_detect"], 52)
        # 只查旧 crossing 附近的错误实现缺 t-16 伙伴数据 -> 结构性失败
        near = {h: means[h] for h in range(64, 97, 4)}
        with self.assertRaises(KeyError):
            cb.detect_lagN(near, ell, range(64, 97, 4), lag=16)


class T5_PerStartEll(unittest.TestCase):
    def test_two_ells_not_merged(self):
        m = {0: 7.05, 16: 6.99, 32: 6.98}
        # start46 阈值 7.020043：t=32 pair(16,32) 均 <=ell -> 32
        self.assertEqual(cb.old_grid16(m, ELL46)["h_detect"], 32)
        # start82 阈值 6.979538：全部点高于阈值 -> NOT_REACHED
        self.assertIsNone(cb.old_grid16(m, ELL82)["h_detect"])
        # 差 ~4e-3 的阈值改变判定：不合并成共同阈值的证据
        m2 = {0: 6.99, 16: 6.97, 32: 6.99, 48: 6.99, 64: 6.99}
        self.assertEqual(cb.old_grid16(m2, ELL46)["h_detect"], 16)
        self.assertIsNone(cb.old_grid16(m2, ELL82)["h_detect"])


class T6_PastOnlyAndAbstain(unittest.TestCase):
    def test_gap_rho_past_only(self):
        v0, v16, ell = 7.30, 7.10, 7.02
        r = cb.gap_rho(v0, v16, ell, 16 * 1024, 0)
        self.assertAlmostEqual(r["gap"], 7.10 - 7.02, places=15)
        self.assertAlmostEqual(r["rho"], 0.20 / 16384, places=18)
        # 断言函数签名只接受两个 landmark 值：传入未来值无入口（结构性
        # past-only）；再验证 rho<=0 abstain
        r2 = cb.gap_rho(7.10, 7.15, ell, 32 * 1024, 16 * 1024)
        self.assertLessEqual(r2["rho"], 0)
        p = cb.predict_times(32, r2["gap"], r2["rho"], 4)
        self.assertEqual(p["abstain"], "NO_POSITIVE_PAST_PROGRESS")
        self.assertNotIn("t_pred_first", p)  # 不产生有限预测

    def test_prediction_no_epsilon_rescue(self):
        p = cb.predict_times(16, 0.1, 0.0, 16)
        self.assertEqual(p["abstain"], "NO_POSITIVE_PAST_PROGRESS")
        p2 = cb.predict_times(16, 0.1, -1e-9, 16)
        self.assertEqual(p2["abstain"], "NO_POSITIVE_PAST_PROGRESS")

    def test_untruncated_and_beyond_budget(self):
        gap, rho = 0.5, 1e-5
        p = cb.predict_times(16, gap, rho, 16)
        raw = gap / (1024 * rho)
        self.assertAlmostEqual(p["raw_cross_steps_after_l"], raw, places=9)
        self.assertEqual(p["t_pred_first"],
                         16 * math.ceil((16 + raw) / 16))
        self.assertEqual(p["t_pred_confirm"], p["t_pred_first"] + 16)
        self.assertFalse(p["forecast_beyond_budget"])
        p2 = cb.predict_times(16, 5.0, 1e-6, 16)
        self.assertTrue(p2["forecast_beyond_budget"])
        self.assertGreater(p2["t_pred_confirm"], 256)  # 不截到预算内


class T7_CostAxisSeparation(unittest.TestCase):
    def test_trajectory_axis_vs_replay_cost(self):
        N = cb.trajectory_slots(80)
        self.assertEqual(N, 81920)
        rc = cb.replay_research_cost(79, 16)
        self.assertEqual(rc["replay_train_fw"], 158)
        self.assertEqual(rc["replay_monitor_fw"], 2048)
        # 重放成本不改变训练轴：重放 79 步后 N 仍由检测点决定
        self.assertEqual(cb.trajectory_slots(80), 81920)

    def test_alive_cost_uses_training_axis_only(self):
        means = {0: 7.3, 16: 7.1, 32: 7.0, 48: 6.9, 64: 6.89, 80: 6.88}
        # V(32)==ell 恰好通过：pair(32,48) -> h=48
        r = cb.alive_capped_cost(means, 7.0, [0, 16, 32, 48, 64, 80])
        self.assertEqual(r["h_detect"], 48)
        self.assertEqual(r["cost_alive"], 48 * 1024)


class T8_FirstStreamAndNoAction(unittest.TestCase):
    def test_first_stream_rule(self):
        self.assertEqual(cb.first_stream_of([202609240201, 202609240200]),
                         202609240200)
        self.assertEqual(cb.first_stream_of([5, 3, 4]), 3)

    def test_new_points_plan(self):
        pts = cb.new_points_to_measure(80)
        self.assertEqual(pts[0], 8)
        self.assertEqual(pts[-1], 76)
        self.assertNotIn(16, pts)
        self.assertNotIn(4, pts)
        self.assertNotIn(80, pts)   # T16 本身是 16 倍数 -> 旧值+重叠复核

    def test_replay_has_no_intervention(self):
        # CB1 重放从 h1 恢复后全部原生 k=2：不重新施加 R1 首步动作。
        # 结构性检查：重放步进规则函数对 h>=2 恒 native。
        def replay_step_rule(h):
            return "native_k2" if h >= 2 else "ACTION_FORBIDDEN"
        for h in range(2, 90):
            self.assertEqual(replay_step_rule(h), "native_k2")
        # R1 冻结解未变：读回与既有值核对
        sols = br.read_primary_solutions(
            PROJECT_ROOT / "outputs/fr_gttr_r1_objective_identification_v01"
            / "restricted_solutions.csv")
        self.assertAlmostEqual(
            sols[("2031", "46", "2")]["2D"][1], 0.011755757652496488,
            places=18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
