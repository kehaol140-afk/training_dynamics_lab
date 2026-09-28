import unittest
from fr.mapping_stats import mapping_summary

class TestMappingStats(unittest.TestCase):
    def test_perfect_linear(self):
        B=[-1,-2,-3,-4,-5,-6,-7]
        chi=[-2,-4,-6,-8,-10,-12,-14]
        m=mapping_summary(B,chi)
        self.assertAlmostEqual(m['pearson_r'],1.0,places=12)
        self.assertAlmostEqual(m['spearman_rho'],1.0,places=12)
        self.assertAlmostEqual(m['slope_K'],2.0,places=12)
        self.assertAlmostEqual(m['r2'],1.0,places=12)
        self.assertAlmostEqual(m['sign_agreement'],1.0,places=12)
