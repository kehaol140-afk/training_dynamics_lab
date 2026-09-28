import math
import unittest

import numpy as np

from fr.e4b_dynamic_timescale import (
    classify_screen,
    curve_collapse_diagnostic,
    fit_tau_plateau,
    loso_single_candidate,
)


FIT_CFG = {
    "tau_min": 0.1,
    "tau_max": 512.0,
    "coarse_log_grid_points": 2048,
    "refine_log_grid_points": 1024,
    "refine_rounds": 2,
    "c_min": -1.0,
    "c_max": 1.0,
    "boundary_tolerance_fraction": 0.001,
}


class TestTauFit(unittest.TestCase):
    def test_recovers_known_tau_and_plateau(self):
        h = np.asarray([1, 2, 4, 8, 12, 16, 24, 32], dtype=float)
        tau_true, c_true = 5.75, 0.58
        y = c_true + (1.0 - c_true) * np.exp(-h / tau_true)
        fit = fit_tau_plateau(h, y, FIT_CFG)
        self.assertTrue(fit["fit_success"])
        self.assertLess(abs(fit["tau_q"] - tau_true) / tau_true, 0.01)
        self.assertLess(abs(fit["c"] - c_true), 0.005)
        self.assertGreater(fit["r2"], 0.99999)


class TestLOSO(unittest.TestCase):
    def test_loso_runs_with_training_fold_standardization(self):
        rows = []
        for seed in range(8):
            for j, sf in enumerate(np.linspace(0.35, 0.65, 7)):
                B = -0.2 + 0.05 * j + 0.01 * seed
                z = 0.2 + 0.1 * j + 0.02 * seed
                chi = 0.05 * seed + 0.3 * sf + 0.4 * B + 0.8 * B * z
                rows.append({"seed": seed, "state_fraction": sf, "B": B, "chi": chi, "z": z})
        result = loso_single_candidate(rows, "z")
        self.assertEqual(result["n_folds"], 8)
        self.assertEqual(len(result["folds"]), 8)
        self.assertTrue(math.isfinite(result["mean_relative_mae_improvement"]))


class TestCurveCollapse(unittest.TestCase):
    def test_curve_collapse_diagnostic_is_finite(self):
        rows = []
        edges = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 1e30]
        horizons = [1, 2, 4, 8, 12, 16, 24, 32]
        for seed in range(8):
            for state in range(7):
                tau = 3.0 + 0.5 * state + 0.1 * seed
                sf = 0.35 + 0.05 * state
                for h in horizons:
                    B = 0.1 + 0.01 * state
                    x = h / tau
                    chi = 0.02 * seed + 0.1 * sf + B * math.exp(-0.2 * x)
                    rows.append({
                        "seed": seed, "state_fraction": sf, "horizon": h,
                        "normalized_horizon": x, "B": B, "chi": chi,
                    })
        cfg = {
            "x_bin_edges": edges,
            "minimum_group_n": 16,
            "supported_relative_rmse_gain_gte": 0.05,
            "mixed_relative_rmse_gain_gt": 0.0,
            "minimum_used_fraction_supported": 0.90,
            "minimum_used_fraction_mixed": 0.80,
        }
        r = curve_collapse_diagnostic(rows, cfg)
        self.assertIn(r["status"], {"SUPPORTED", "MIXED", "UNSUPPORTED"})
        self.assertTrue(math.isfinite(r["raw_horizon"]["pooled_rmse"]))
        self.assertTrue(math.isfinite(r["normalized_horizon"]["pooled_rmse"]))


if __name__ == "__main__":
    unittest.main()
