"""E1 sigma_q 统计测试。"""
import unittest
from fr.fluctuation_stats import summarize_projections


class TestFluctuationStats(unittest.TestCase):
    def test_sigma_and_eps(self):
        s = summarize_projections([1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0])
        self.assertTrue(s["nonzero_variance"])
        self.assertGreater(s["sigma_q"], 0.0)
        self.assertAlmostEqual(s["epsilon_candidates"]["rho_0.5"], 0.5 * s["sigma_q"])


if __name__ == "__main__":
    unittest.main()
