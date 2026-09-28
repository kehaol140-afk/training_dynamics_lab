import math
import unittest
import numpy as np
from fr.a0_fr_diagnostics import (
    balanced_splits_unique, spearman_brown, observational_slope,
    direct_alpha_frozen_preconditioner, pearson, seed_fe_slope,
)

class TestReliability(unittest.TestCase):
    def test_balanced_unique_counts(self):
        self.assertEqual(len(balanced_splits_unique(8)),35)
        self.assertEqual(len(balanced_splits_unique(16)),6435)
    def test_spearman_brown(self):
        self.assertAlmostEqual(spearman_brown(0.5),2/3,places=12)
    def test_observational_slope(self):
        s=np.arange(8,dtype=float); y=2.5*s+3
        self.assertAlmostEqual(observational_slope(s,y),2.5,places=10)

class TestDirectPath(unittest.TestCase):
    def test_beta_decay_with_bias_correction(self):
        beta=0.9; t=40; mu=2.0; lrs=[1e-3]*4
        a=direct_alpha_frozen_preconditioner(mu_q=mu,beta1=beta,optimizer_step=t,lrs=lrs,weight_decay=0.0)
        self.assertEqual(len(a),4)
        self.assertTrue(all(np.isfinite(a)))
        self.assertLess(a[0],0.0)
        self.assertTrue(all(a[i+1] < a[i] for i in range(3)))
    def test_weight_decay_is_finite(self):
        a=direct_alpha_frozen_preconditioner(mu_q=1.0,beta1=0.9,optimizer_step=10,lrs=[1e-3]*8,weight_decay=0.01)
        self.assertTrue(np.isfinite(a).all())

class TestSeedFE(unittest.TestCase):
    def test_recovers_known_slope(self):
        rows=[]
        for seed in range(4):
            for j in range(7):
                sf=j/6
                B=((j-3)**2 - 4.0)*0.07 + seed*0.01
                chi=0.7*B+0.1*sf+seed*0.03
                rows.append({'seed':seed,'state_fraction':sf,'B':B,'chi':chi})
        got=seed_fe_slope(rows)
        self.assertAlmostEqual(got['K'],0.7,places=8)

if __name__=='__main__': unittest.main()
