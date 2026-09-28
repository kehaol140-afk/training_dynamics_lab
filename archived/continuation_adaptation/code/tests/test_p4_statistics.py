"""p4_statistics 冻结统计口径的单测（docs/P4_PROTOCOL.md §7，合成数据）。

- Holm 校正已知解；
- seed 聚类置换检验：强改善 → p 极小；无效应 → p 大；同种子可复现；
- 聚类 bootstrap CI 方向正确；
- evaluate_gates：全门槛通过 → CONTINUE；seed 一致性 5/8 → STOP；
- paired_diffs / seed_cluster_means 基础正确性。
"""
import _bootstrap  # noqa: F401
import unittest

from p4_statistics import (
    cluster_bootstrap_ci, evaluate_gates, evaluate_variant, holm_correction,
    naive_bootstrap_ci, paired_diffs, seed_cluster_means,
    seed_cluster_permutation_test,
)

SEEDS = [2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037]
STEPS = [46, 52, 58, 64, 70, 76, 82]
META = {f"s{s}_t{t:03d}": (s, t) for s in SEEDS for t in STEPS}


def _synthetic_d(offset_by_seed=None, noise=0.0, seed0=77080001):
    import random
    rng = random.Random(seed0)
    d = {}
    for cell, (s, t) in META.items():
        base = (offset_by_seed or {}).get(s, -0.05)
        d[cell] = base + rng.uniform(-noise, noise)
    return d


class TestHolm(unittest.TestCase):
    def test_known_cases(self):
        h = holm_correction({"C1": 0.01, "C2": 0.04}, alpha=0.05)
        self.assertTrue(h["C1"]["significant"])
        self.assertTrue(h["C2"]["significant"])

        h = holm_correction({"C1": 0.04, "C2": 0.06}, alpha=0.05)
        self.assertFalse(h["C1"]["significant"])
        self.assertFalse(h["C2"]["significant"])

        h = holm_correction({"C1": 0.02, "C2": 0.06}, alpha=0.05)
        self.assertTrue(h["C1"]["significant"])
        self.assertFalse(h["C2"]["significant"])


class TestPermutation(unittest.TestCase):
    def test_strong_improvement_small_p(self):
        d = _synthetic_d({s: -0.10 for s in SEEDS})
        r = seed_cluster_permutation_test(d, META, reps=20000)
        # 全负差异下 p 收敛到 1/256（+计数涨落），上界放宽到 0.006
        self.assertLessEqual(r["p_value_one_sided"], 0.006)
        self.assertLess(r["observed_mean"], 0)

    def test_no_effect_large_p(self):
        d = _synthetic_d({s: 0.0 for s in SEEDS}, noise=0.5)
        r = seed_cluster_permutation_test(d, META, reps=5000)
        self.assertGreater(r["p_value_one_sided"], 0.10)

    def test_reproducible_with_same_seed(self):
        d = _synthetic_d(noise=0.3)
        r1 = seed_cluster_permutation_test(d, META, reps=2000)
        r2 = seed_cluster_permutation_test(d, META, reps=2000)
        self.assertEqual(r1["p_value_one_sided"], r2["p_value_one_sided"])

    def test_one_seed_reversal_visible_in_seed_means(self):
        offsets = {s: -0.1 for s in SEEDS}
        offsets[2035] = +0.2
        means = seed_cluster_means(_synthetic_d(offsets), META)
        self.assertGreater(means[2035], 0)
        self.assertEqual(sum(1 for v in means.values() if v < 0), 7)


class TestBootstrapAndGates(unittest.TestCase):
    def test_cluster_bootstrap_ci_direction(self):
        d = _synthetic_d({s: -0.10 for s in SEEDS}, noise=0.01)
        ci = cluster_bootstrap_ci(d, META, reps=5000)
        self.assertLess(ci["ci_high"], 0)
        naive = naive_bootstrap_ci(d, reps=5000)
        self.assertLess(naive["ci_high"], 0)

    def test_gates_continue_when_all_pass(self):
        d = _synthetic_d({s: -0.10 for s in SEEDS}, noise=0.02)
        st = {"C1": evaluate_variant(d, META, reps_perm=20000, reps_boot=2000)}
        holm = {"C1": {"significant": True, "p_value": 0.004}}
        gates = evaluate_gates(st, holm)
        self.assertTrue(gates["continue_direction"])
        self.assertTrue(gates["per_variant"]["C1"]["all_gates_passed"])

    def test_gates_stop_on_seed_inconsistency(self):
        offsets = {s: -0.10 for s in SEEDS}
        for s in (2035, 2036, 2037):  # 仅 5/8 seeds 改善
            offsets[s] = +0.10
        d = _synthetic_d(offsets, noise=0.01)
        st = {"C1": evaluate_variant(d, META, reps_perm=20000, reps_boot=2000)}
        holm = {"C1": {"significant": False, "p_value": 0.30}}
        gates = evaluate_gates(st, holm)
        self.assertFalse(gates["continue_direction"])
        self.assertEqual(st["C1"]["seeds_improved"], 5)
        self.assertFalse(gates["per_variant"]["C1"]["gate_checks"]["2_seed_consistency_ge_6of8"])

    def test_paired_diffs(self):
        table = {"c1": {"C1": 6.9, "A_best": 7.0}, "c2": {"C1": 7.1, "A_best": 7.0}}
        d = paired_diffs(table, "C1")
        self.assertAlmostEqual(d["c1"], -0.1, places=12)
        self.assertAlmostEqual(d["c2"], 0.1, places=12)


if __name__ == "__main__":
    unittest.main()
