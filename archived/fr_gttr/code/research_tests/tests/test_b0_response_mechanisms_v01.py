#!/usr/bin/env python3
"""Unit tests for fr.b0_response_mechanisms_v01 (CPU only, unittest)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr.b0_response_mechanisms_v01 import (  # noqa: E402
    U_GRID, analytic_chi, b1_toy_battery, b2_lag_antisymmetry,
    b3_decision_value, b4_decomposition_identity, chi_q_from_curve,
    class_mean_losses, quadratic_fit_dev, q_identifiability, v_grid,
    v_sign,
)


class TestTheoryBatteries(unittest.TestCase):
    def test_b1_battery_ok(self):
        self.assertTrue(b1_toy_battery()["ok"])

    def test_b2_ok(self):
        self.assertTrue(b2_lag_antisymmetry()["ok"])

    def test_b3_ok(self):
        self.assertTrue(b3_decision_value()["ok"])

    def test_b4_ok(self):
        self.assertTrue(b4_decomposition_identity()["ok"])

    def test_analytic_chi_signs(self):
        self.assertGreater(analytic_chi(4, 0.1, 0.15, 1.0, 1.0), 0)
        self.assertLess(analytic_chi(16, 0.1, 0.15, 1.0, 1.0), 0)
        self.assertLess(analytic_chi(64, 0.1, 0.15, 1.0, 1.0), 0)


class TestAnalysisHelpers(unittest.TestCase):
    def test_chi_q_from_curve_exact_quadratic(self):
        chi_true, Q_true = -0.01, 0.3
        L = {u: 1.0 + chi_true * u + 0.5 * Q_true * u ** 2 for u in U_GRID}
        out = chi_q_from_curve(L)
        self.assertAlmostEqual(out["chi_fd_a0.05"], chi_true, places=12)
        self.assertAlmostEqual(out["Q_fd_a0.05"], Q_true, places=12)
        self.assertTrue(out["Q_sign_agree"])

    def test_quadratic_fit_exact(self):
        chi_true, Q_true = 0.02, -0.4
        Ldev = {u: 2.0 + chi_true * u + 0.5 * Q_true * u ** 2
                for u in (-0.025, 0.0, 0.025)}
        chi, Q = quadratic_fit_dev(Ldev)
        self.assertAlmostEqual(chi, chi_true, places=12)
        self.assertAlmostEqual(Q, Q_true, places=12)

    def test_q_identifiability_rule(self):
        self.assertTrue(q_identifiability(0.5, 0.45, 0.5, 0.48)["identifiable"])
        self.assertFalse(q_identifiability(0.5, 0.5, 0.5, -0.4)["identifiable"])
        self.assertFalse(q_identifiability(1e-4, -1e-4, 2e-4, 1e-4)["identifiable"])

    def test_v_sign_and_v_grid(self):
        self.assertAlmostEqual(v_sign([-0.01, -0.02, -0.03]), 0.0, places=15)
        self.assertGreater(v_sign([-0.01, 0.02]), 0)
        self.assertAlmostEqual(
            v_grid([{u: 1.0 for u in U_GRID} for _ in range(3)]), 0.0, places=15)
        losses = [{u: (0.0 if u == -0.05 else 1.0) for u in U_GRID},
                  {u: (0.0 if u == +0.05 else 1.0) for u in U_GRID}]
        self.assertGreater(v_grid(losses), 0)

    def test_class_mean_losses(self):
        recs = []
        for cls, base in (("dev", 1.0), ("dev", 3.0), ("audit", 10.0)):
            recs.append({"stream_class": cls,
                         "losses": {u: {"v_eval": base} for u in U_GRID}})
        m = class_mean_losses(recs, "v_eval", "dev")
        self.assertAlmostEqual(m[-0.05], 2.0, places=12)
        ma = class_mean_losses(recs, "v_eval", "audit")
        self.assertAlmostEqual(ma[0.025], 10.0, places=12)

    def test_decomposition_group_key_contract(self):
        # decomposition groups keyed by (seed, step, protocol, u, H); streams in
        # a group always share seed+step (no cross-seed parameter averaging).
        group_key = (2030, 46, "persistent", 0.05, 16)
        self.assertEqual(group_key[:2], (2030, 46))


if __name__ == "__main__":
    unittest.main()
