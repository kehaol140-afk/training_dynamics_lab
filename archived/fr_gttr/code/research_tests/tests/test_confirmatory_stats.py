import unittest

try:
    from fr.confirmatory_stats import (
        analyze_confirmatory_horizon,
        holm_secondary_correction,
    )
except ImportError:
    analyze_confirmatory_horizon = None
    holm_secondary_correction = None


STEPS = [46, 52, 58, 64, 70, 76, 82]
STATE_FRACTIONS = [x / 128.0 for x in STEPS]
CONFIRMATION_SEEDS = list(range(2030, 2038))
WITHIN_SEED_PATTERN = [-3.0, 1.5, -1.0, 3.0, -0.5, 2.0, -2.0]


def make_positive_confirmatory_rows():
    rows = []
    for seed_pos, seed in enumerate(CONFIRMATION_SEEDS):
        seed_center = seed_pos - 3.5
        for step, sf, pattern in zip(STEPS, STATE_FRACTIONS, WITHIN_SEED_PATTERN):
            b_value = 0.4 * sf + pattern + 0.03 * seed_center
            chi_value = 10.0 + 0.2 * seed_center + 0.7 * sf + 2.0 * b_value
            rows.append({
                "train_seed": seed,
                "step": step,
                "state_fraction": sf,
                "B": b_value,
                "chi": chi_value,
                "clipping": {
                    "baseline_rate": 0.125,
                    "plus_rate": 0.25,
                    "minus_rate": 0.125,
                },
            })
    return rows


class TestConfirmatoryStats(unittest.TestCase):
    def test_primary_horizon_passes_only_when_registered_conditions_hold(self):
        # 会被这个测试抓住的 bug：把 state 当作 bootstrap cluster、漏掉 seed FE、
        # 或者把 LOSO 条件从 primary 判定中移除。
        self.assertIsNotNone(analyze_confirmatory_horizon, "missing fr.confirmatory_stats")

        analysis = analyze_confirmatory_horizon(
            make_positive_confirmatory_rows(),
            horizon=8,
            primary_horizon=8,
            bootstrap_reps=256,
            permutation_reps=512,
            random_seed=1707,
        )

        self.assertAlmostEqual(analysis["seed_fe_adjusted"]["K"], 2.0, places=10)
        self.assertGreater(analysis["cluster_bootstrap_ci_K"]["ci95"][0], 0.0)
        self.assertEqual(analysis["cluster_bootstrap_ci_K"]["cluster_unit"], "training_seed")
        self.assertGreater(analysis["within_seed_detrended"]["pearson"], 0.99)
        self.assertLess(analysis["circular_shift_permutation"]["p_one_sided_positive"], 0.05)
        self.assertGreater(analysis["loso"]["mean_relative_mae_improvement"], 0.0)
        self.assertEqual(analysis["loso"]["folds_improved"], 8)
        self.assertEqual(analysis["primary_confirmation"]["status"], "PASS")

    def test_primary_horizon_fails_when_K_direction_is_negative(self):
        # 会被这个测试抓住的 bug：只看 p-value/LOSO，不强制 K_8 > 0。
        self.assertIsNotNone(analyze_confirmatory_horizon, "missing fr.confirmatory_stats")

        rows = make_positive_confirmatory_rows()
        for row in rows:
            row["chi"] = -row["chi"]

        analysis = analyze_confirmatory_horizon(
            rows,
            horizon=8,
            primary_horizon=8,
            bootstrap_reps=128,
            permutation_reps=128,
            random_seed=1707,
        )

        self.assertLess(analysis["seed_fe_adjusted"]["K"], 0.0)
        self.assertEqual(analysis["primary_confirmation"]["status"], "FAIL")

    def test_holm_secondary_correction_excludes_primary_H8(self):
        # 会被这个测试抓住的 bug：把 H=8 primary endpoint 纳入 secondary 校正。
        self.assertIsNotNone(holm_secondary_correction, "missing fr.confirmatory_stats")

        corrected = holm_secondary_correction(
            {"1": 0.01, "2": 0.04, "4": 0.20, "8": 1e-12},
            primary_horizon=8,
            alpha=0.05,
        )

        self.assertNotIn("8", corrected)
        self.assertAlmostEqual(corrected["1"]["holm_p"], 0.03)
        self.assertAlmostEqual(corrected["2"]["holm_p"], 0.08)
        self.assertAlmostEqual(corrected["4"]["holm_p"], 0.20)
        self.assertTrue(corrected["1"]["significant_holm_0_05"])
        self.assertFalse(corrected["2"]["significant_holm_0_05"])


if __name__ == "__main__":
    unittest.main()
