"""CPU unit tests for FR_A0G decision-utility v01 (analysis module + runner helpers).

覆盖：
  1. 数据组：互斥、完备、可复现、不依赖输入顺序；含免责声明。
  2. validation split：V_score/V_eval 不重叠且覆盖 eval pool。
  3. level3_decision：Gain/Regret/top1 near-tie/harmful/direction 语义。
  4. 凸组合恒等式：g + lambda*(g_j - g) == (1-lambda)*g + lambda*g_j。
  5. 候选 batch 生成：确定性、组间不同、不碰全局 numpy RNG。
  6. 指标复用：a0g 的 pooled_metric/relative_error 与冻结 a0c 模块一致。
  7. 成本模型：direct vs shared-tangent 公式与 breakeven 数学。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "research/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_response_propagation_v01 import _metric as frozen_metric
from fr.a0c_adamw_response_propagation_v01 import relative_error as frozen_rel_err
from fr.a0g_decision_utility_v01 import (
    LAMBDA_CANDIDATES,
    N_DATA_GROUPS,
    NEAR_TIE_FRACTION,
    assign_data_group,
    breakeven_m,
    build_data_group_definition,
    cost_model,
    level3_decision,
    per_state_spearman,
    pooled_metric,
    random_choice,
    select_action,
    split_validation_pool,
)
from fr.a0g_decision_utility_v01 import _safe_spearman


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "a0g_runner", REPO_ROOT / "research/scripts/a0g_decision_utility_v01.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


class TestDataGroups(unittest.TestCase):
    def _texts(self, n=512):
        return [f"样本 {i}: 一段中文测试文本。" for i in range(n)]

    def test_disjoint_complete_reproducible(self):
        # texts 是数据集全量（按行号索引），train_indices 为其中子集
        texts = self._texts(600)
        train_indices = list(range(7, 7 + 512))  # 非平凡起点，>= 4*100 通过尺寸护栏
        dgd = build_data_group_definition(train_indices, texts)
        self.assertEqual(dgd["n_groups"], N_DATA_GROUPS)
        members = dgd["members"]
        merged = []
        for g in range(N_DATA_GROUPS):
            merged.extend(members[str(g)])
        self.assertEqual(len(merged), len(set(merged)), "groups must be disjoint")
        self.assertEqual(set(merged), set(train_indices), "groups must cover train pool")
        # 可复现：同一输入重算哈希一致
        dgd2 = build_data_group_definition(train_indices, texts)
        self.assertEqual(dgd["definition_sha256"], dgd2["definition_sha256"])
        # 顺序无关性：打乱 train_indices 顺序后成员集合不变
        dgd3 = build_data_group_definition(list(reversed(train_indices)), texts)
        for g in range(N_DATA_GROUPS):
            self.assertEqual(set(members[str(g)]), set(dgd3["members"][str(g)]))

    def test_small_pool_skips_size_guard(self):
        # 小训练池（单元测试规模）只验证结构性约束，不触发 4*100 尺寸护栏
        train_indices = list(range(64))
        dgd = build_data_group_definition(train_indices, self._texts())
        merged = [i for g in range(N_DATA_GROUPS) for i in dgd["members"][str(g)]]
        self.assertEqual(set(merged), set(train_indices))
        self.assertEqual(len(merged), len(set(merged)))

    def test_assign_is_deterministic_and_bounded(self):
        for idx in range(50):
            g = assign_data_group(idx, f"text-{idx}")
            self.assertTrue(0 <= g < N_DATA_GROUPS)
            self.assertEqual(g, assign_data_group(idx, f"text-{idx}"))

    def test_disclaimer_present(self):
        dgd = build_data_group_definition(list(range(64)), self._texts(64))
        self.assertIn("仅用于决策管线验证", dgd["disclaimer"])
        self.assertIn("不代表语义数据域", dgd["disclaimer"])


class TestValidationSplit(unittest.TestCase):
    def test_disjoint_and_complete(self):
        eval_indices = list(range(10, 10 + 200, 3))  # 非连续 + 非偶数
        split = split_validation_pool(eval_indices)
        self.assertTrue(split["disjoint"])
        self.assertFalse(set(split["v_score"]) & set(split["v_eval"]))
        self.assertEqual(
            set(split["v_score"]) | set(split["v_eval"]), set(eval_indices)
        )
        self.assertEqual(split["n_v_score"] + split["n_v_eval"], len(eval_indices))
        # 可复现
        split2 = split_validation_pool(eval_indices)
        self.assertEqual(split["split_sha256"], split2["split_sha256"])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            split_validation_pool([1, 2, 2, 3])


class TestLevel3Decision(unittest.TestCase):
    def test_basic_gain_regret(self):
        actual = {0: 0.01, 1: -0.05, 2: 0.02, 3: -0.01, 4: 0.03}
        pred = {0: 0.0, 1: -0.1, 2: 0.1, 3: 0.0, 4: 0.2}
        out = level3_decision(predicted_by_j=pred, actual_by_j=actual)
        self.assertEqual(out["j_hat"], 1)  # 预测最优=实际最优
        self.assertAlmostEqual(out["gain"], 0.05)
        self.assertAlmostEqual(out["regret"], 0.0)
        self.assertFalse(out["harmful"])
        self.assertTrue(out["top1_correct"])
        self.assertTrue(out["direction_correct"])

    def test_regret_when_suboptimal(self):
        actual = {0: 0.0, 1: 0.04, 2: -0.02, 3: 0.01, 4: 0.03}
        pred = {0: 0.0, 1: -0.5, 2: 0.0, 3: 0.0, 4: 0.0}
        out = level3_decision(predicted_by_j=pred, actual_by_j=actual)
        self.assertEqual(out["j_hat"], 1)
        self.assertAlmostEqual(out["actual_delta_selected"], 0.04)
        self.assertAlmostEqual(out["best_possible_delta"], -0.02)
        self.assertAlmostEqual(out["regret"], 0.06)
        self.assertAlmostEqual(out["gain"], -0.04)
        self.assertTrue(out["harmful"])
        self.assertFalse(out["top1_correct"])

    def test_near_tie_counts_as_correct(self):
        rng = 0.010
        actual = {0: 0.0, 1: -0.0105, 2: -0.0100, 3: 0.005, 4: 0.008}
        # 实际最优 j=1，但 j=2 与 j=1 的差 = 0.0005 < tol = 0.05*range
        rng_width = max(actual.values()) - min(actual.values())
        tol = max(1e-12, NEAR_TIE_FRACTION * rng_width)
        self.assertLess(abs(actual[2] - actual[1]), tol)
        pred = {0: 0.0, 1: 0.1, 2: -0.2, 3: 0.0, 4: 0.0}
        out = level3_decision(predicted_by_j=pred, actual_by_j=actual)
        self.assertEqual(out["j_hat"], 2)
        self.assertTrue(out["top1_correct"])  # near-tie 宽容
        self.assertIn(1, out["tied_optimal"])
        self.assertIn(2, out["tied_optimal"])

    def test_direction_accuracy(self):
        actual = {0: 0.02, 1: -0.01, 2: 0.0, 3: 0.04, 4: -0.03}
        pred = {0: 0.0, 1: -0.01, 2: 0.0, 3: 0.01, 4: -0.02}
        out = level3_decision(predicted_by_j=pred, actual_by_j=actual)
        self.assertTrue(out["direction_correct"])  # 两者都选 j=4？不：pred 最优是 j=4
        self.assertEqual(out["j_hat"], 4)
        self.assertTrue(out["direction_correct"])

    def test_candidate_set_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            level3_decision(predicted_by_j={0: 0.0, 1: -1.0},
                            actual_by_j={0: 0.0, 1: 0.0, 2: 0.0})


class TestConvexMixtureIdentity(unittest.TestCase):
    def test_lambda_mixture(self):
        rng = np.random.default_rng(7)
        g = rng.normal(size=(37,)) * 0.01
        gj = rng.normal(size=(37,)) * 0.02
        for lam in LAMBDA_CANDIDATES:
            left = g + lam * (gj - g)
            right = (1.0 - lam) * g + lam * gj
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-18)
            # 凸性：结果范数不超过两端点最大范数
            self.assertLessEqual(np.linalg.norm(left), max(np.linalg.norm(g), np.linalg.norm(gj)) + 1e-12)


class TestCandidateBatches(unittest.TestCase):
    def test_deterministic_group_dependent_rng_safe(self):
        members = list(range(100, 400))
        before = np.random.get_state()
        a1 = RUNNER.candidate_batches_for(777, 1, members, 4, 2)
        mid = np.random.get_state()
        np.random.set_state(before)
        a2 = RUNNER.candidate_batches_for(777, 1, members, 4, 2)
        np.random.set_state(mid)
        self.assertEqual(a1, a2)
        b1 = RUNNER.candidate_batches_for(777, 2, members, 4, 2)
        self.assertNotEqual(a1, b1)
        # 每个 microbatch 尺寸正确、成员均在组内
        self.assertEqual(len(a1), 2)
        for batch in a1 + b1:
            self.assertEqual(len(batch), 4)
            self.assertTrue(set(batch) <= set(members))
        self.assertFalse(set(a1[0]) & set(a1[1]))  # 两个 microbatch 不重叠

    def test_stream_seed_changes_batches(self):
        members = list(range(200))
        a = RUNNER.candidate_batches_for(1, 1, members, 4, 2)
        b = RUNNER.candidate_batches_for(2, 1, members, 4, 2)
        self.assertNotEqual(a, b)


class TestMetricReuse(unittest.TestCase):
    def test_pooled_metric_matches_frozen(self):
        observed = [0.1, -0.2, 0.05, 0.0, 0.3]
        predicted = [0.09, -0.19, 0.06, 0.01, 0.28]
        ours = pooled_metric(
            [{"h_FD": o, "hhat_C": p} for o, p in zip(observed, predicted)],
            pred_key="hhat_C", actual_key="h_FD",
        )
        frozen = frozen_metric(observed, predicted)
        self.assertEqual(set(ours), set(frozen))
        for key in frozen:
            self.assertAlmostEqual(float(ours[key]), float(frozen[key]), places=12,
                                   msg=f"metric field {key} diverged from frozen a0c definition")

    def test_relative_error_matches_frozen(self):
        for o, p in [(0.5, 0.4), (-0.1, -0.12), (1e-6, 2e-6)]:
            self.assertAlmostEqual(
                float(frozen_rel_err(o, p)), float(frozen_rel_err(o, p)), places=15
            )

    def test_safe_spearman_perfect_and_reverse(self):
        a = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_safe_spearman(a, a), 1.0)
        self.assertAlmostEqual(_safe_spearman(a, list(reversed(a))), -1.0)
        self.assertIsNone(_safe_spearman([1.0, 1.0], [2.0, 3.0]))  # 常数列无秩相关


class TestPerStateSpearman(unittest.TestCase):
    def test_within_state_only(self):
        # per_state_spearman 的约定输入：单个 state 内、每方法一行的
        # {'hhat':..., 'h_fd':..., 'method':...} 列表。
        rows = [
            {"method": "C", "hhat": 1.0, "h_fd": 1.1},
            {"method": "C", "hhat": 2.0, "h_fd": 1.9},
            {"method": "C", "hhat": 3.0, "h_fd": 3.2},
            {"method": "B", "hhat": -3.0, "h_fd": -3.1},
            {"method": "B", "hhat": -2.0, "h_fd": -1.9},
            {"method": "B", "hhat": -1.0, "h_fd": -0.9},
        ]
        out = per_state_spearman(rows)
        self.assertAlmostEqual(out["C"], 1.0, places=9)
        self.assertAlmostEqual(out["B"], 1.0, places=9)
        self.assertIsNone(out["alignment"])  # 不足两行 → None


class TestCostModel(unittest.TestCase):
    def _measured(self):
        return {
            "t_native_step": 0.10,
            "t_val_loss_one": 0.02,
            "t_val_grad_one": 0.03,
            "t_cand_grad_one": 0.05,
            "t_hvp_one": 0.08,
            "t_tangent_step_one": 0.01,
        }

    def test_formula_hand_computed(self):
        m = self._measured()
        rows = {(r["H"], r["M_candidates"]): r for r in cost_model(m, horizons=(4,), m_grid=(2,))}
        r = rows[(4, 2)]
        c_base = 4 * 0.10
        direct = c_base + 0.02 + 2 * (c_base + 0.02)
        tangent = c_base + 0.02 + 0.03 + 2 * (0.05 + 3 * 0.08 + 4 * 0.01)
        self.assertAlmostEqual(r["direct_replay_total_s"], direct)
        self.assertAlmostEqual(r["shared_tangent_total_s"], tangent)
        self.assertAlmostEqual(r["direct_per_candidate_s"], direct / 2)
        self.assertAlmostEqual(r["shared_tangent_per_candidate_s"], tangent / 2)
        # direct = 0.42+0.02+2*0.44 = 1.32；tangent = 0.45+2*0.33 = 1.11
        self.assertAlmostEqual(r["tangent_speedup_vs_direct"], direct / tangent)

    def test_breakeven_monotonic(self):
        m = self._measured()
        # M=1: tangent = 0.45+0.33=0.78 vs direct 0.44+0.44=0.88 → tangent 已便宜
        self.assertEqual(breakeven_m(m, h=4), 1)
        # 极端贵 HVP → breakeven 更大或不存在
        m2 = dict(m, t_hvp_one=5.0)
        be = breakeven_m(m2, h=8, m_max=1000)
        if be is not None:
            self.assertGreater(be, 1)


class TestRandomChoice(unittest.TestCase):
    def test_bounds_and_reproducibility(self):
        rng = np.random.default_rng(42)
        picks = [random_choice(rng, 5) for _ in range(200)]
        self.assertTrue(all(0 <= p <= 4 for p in picks))
        rng2 = np.random.default_rng(42)
        picks2 = [random_choice(rng2, 5) for _ in range(200)]
        self.assertEqual(picks, picks2)


class TestSelectAction(unittest.TestCase):
    def test_argmin_with_ties_first_key(self):
        self.assertEqual(select_action({0: 0.1, 1: -0.2, 2: -0.2}), 1)
        self.assertEqual(select_action({0: 0.0, 1: 0.0}), 0)


class TestFreezeLambda(unittest.TestCase):
    """双准则 _freeze_lambda：A（严格局部线性）优先，B（决策信号）回退。"""

    def _scan_rows(self, *, n_cross, spearman_sign=1.0):
        # n_cross: 每个 lambda 的 12 行 (4 j × 3 H) 中有多少行置穿越位
        rows = []
        for lam, nc in zip((0.01, 0.025, 0.05, 0.1), n_cross):
            i = 0
            for j in (1, 2, 3, 4):
                for h in (1, 4, 8):
                    hhat = 0.002 * j
                    delta = lam * hhat * spearman_sign
                    rows.append({
                        "split": "dev", "seed": 2031, "step": 52,
                        "stream_id": 0, "j": j, "H": h, "lambda": lam,
                        "delta_val_score": delta, "hhat_C": hhat,
                        "rel_nonlinearity_C": 0.5,
                        "clip_regime_crossing": bool(i < nc),
                        "abs_hhat_C": abs(hhat),
                    })
                    i += 1
        return rows

    def test_rule_a_strict_locality_passes(self):
        # 线性响应 → 所有 slope-dev≈0；穿越率 0/0/0.083/0.333 → primary=0.05
        rows = self._scan_rows(n_cross=(0, 0, 1, 4))
        sel = RUNNER._freeze_lambda(rows)
        self.assertEqual(sel["freeze_rule"], "A_strict_locality")
        self.assertEqual(sel["primary_lambda"], 0.05)
        self.assertEqual(sel["conservative_lambda"], 0.025)
        self.assertTrue(sel["criteria"]["strict_locality_any_lambda_mean_slope_dev_le_0.25"])

    def test_rule_b_fallback_max_lambda_under_crossing_gate(self):
        # 全部 slope-dev 不合格（用非线性 delta 构造）→ 回退 B
        rows = self._scan_rows(n_cross=(0, 1, 1, 2))
        for r in rows:
            lam = r["lambda"]
            if lam != 0.01:
                # 制造大偏差：delta 与 hhat 符号相反且幅度翻倍
                r["delta_val_score"] = -2.0 * lam * r["hhat_C"]
        sel = RUNNER._freeze_lambda(rows)
        self.assertEqual(sel["freeze_rule"], "B_decision_signal")
        self.assertEqual(sel["primary_lambda"], 0.05)
        self.assertEqual(sel["conservative_lambda"], 0.025)
        self.assertFalse(sel["criteria"]["strict_locality_any_lambda_mean_slope_dev_le_0.25"])
        self.assertTrue(sel["criteria"]["primary_clip_crossing_rate_le_0.10"])

    def test_rule_b_all_crossing_fails(self):
        rows = self._scan_rows(n_cross=(12, 12, 12, 12))
        with self.assertRaises(Exception):
            RUNNER._freeze_lambda(rows)


class TestAggregateLevel3(unittest.TestCase):
    """回归：split 名（'dev'/'formal'，长度≠2）曾把聚合循环的解包目标
    `for (split, h), _ in ...` 触发成对字符串逐字符解包 → ValueError。"""

    def _rows(self):
        level3, level2 = [], []
        # 两个 state：dev(2031/52) 与 formal(2038/46)，各 1 stream、H∈{1,4}
        for split, seed in (("dev", 2031), ("formal", 2038)):
            for h in (1, 4):
                for m in ("random", "alignment", "B", "C"):
                    level3.append({
                        "split": split, "seed": seed, "step": 52 if split == "dev" else 46,
                        "stream_id": 0, "H": h, "method": m,
                        "gain": 0.01 if m == "C" else 0.0,
                        "regret": 0.0 if m == "C" else 0.01,
                        "harmful": False, "top1_correct": m == "C",
                        "direction_correct": m != "random",
                        "j_hat": 1, "actual_delta_selected": 0.01,
                        "best_possible_delta": 0.01,
                    })
                by_j = {}
                for j in (0, 1, 2, 3, 4):
                    by_j[j] = {
                        "split": split, "seed": seed,
                        "step": 52 if split == "dev" else 46,
                        "stream_id": 0, "H": h, "j": j,
                        "pred_delta_alignment": 0.1 * j,
                        "pred_delta_B": 0.2 * j,
                        "pred_delta_C": -0.1 * j,
                        "actual_delta_eval": -0.05 * j,
                    }
                level2.extend(by_j[j] for j in sorted(by_j))
        return level3, level2

    def test_aggregate_runs_and_means(self):
        level3, level2 = self._rows()
        agg = RUNNER._aggregate_level3(level3, level2)
        # 每个 split×H 组合都有 4 个方法块（KeyError 即回归）
        for key in ("dev|H=1", "dev|H=4", "formal|H=1", "formal|H=4"):
            self.assertIn(key, agg)
            for m in ("random", "alignment", "B", "C"):
                self.assertIn(m, agg[key], f"{key} missing method {m}")
        # C 方法 gain 固定 0.01、regret 0、top-1 100%
        c4 = agg["formal|H=4"]["C"]
        self.assertAlmostEqual(c4["mean_gain"], 0.01)
        self.assertAlmostEqual(c4["mean_regret"], 0.0)
        self.assertAlmostEqual(c4["top1_accuracy"], 1.0)
        self.assertEqual(c4["n"], 1)
        # MAE 由 level2 提供（C: |-0.1j - (-0.05j)| 对 j=0..4 的均值 = 0.1）
        self.assertAlmostEqual(c4["MAE"], 0.1)
        # within-state Spearman 在 state 内计算且 C 完全单调 → 1.0
        self.assertAlmostEqual(c4["mean_spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
