import unittest
from fr.response_stats import observational_slope, paired_linearity_summary, distribution_summary

class TestResponseStats(unittest.TestCase):
    def test_observational_slope(self):
        r=observational_slope([1,2,3,4],[2,4,6,8])
        self.assertAlmostEqual(r["B"], 2.0, places=12)

    def test_perfect_odd(self):
        r=paired_linearity_summary([0,0,0],[1,1,1],[-1,-1,-1],0.5)
        self.assertAlmostEqual(r["odd_error"],0.0,places=12)
        self.assertAlmostEqual(r["chi"],2.0,places=12)

    def test_distribution(self):
        r=distribution_summary([1,2,3,4])
        self.assertEqual(r["n"],4)
        self.assertGreater(r["std"],0)
