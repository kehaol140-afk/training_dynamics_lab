import unittest
from fr.cross_seed_stats import analyze_horizon

class TestCrossSeedStats(unittest.TestCase):
    def test_B_adds_information_beyond_step(self):
        rows=[]
        # 4 seeds，chi 同时受 step 和 B 影响；B 在 seed/state 间有独立变化。
        for seed in [1,2,3,4]:
            for i,step in enumerate([0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
                B=(seed-2.5)*0.05 + ((i%3)-1)*0.03
                chi=-0.2 + 0.1*step + 2.0*B
                rows.append({
                    'train_seed':seed,'step':i,'state_fraction':step,
                    'B':B,'chi':chi
                })
        a=analyze_horizon(rows)
        self.assertGreater(a['step_adjusted']['incremental_partial_r2_B'],0.8)
        self.assertEqual(a['loso']['folds_improved'],4)
        self.assertGreater(a['detrended']['residual_pearson'],0.8)
