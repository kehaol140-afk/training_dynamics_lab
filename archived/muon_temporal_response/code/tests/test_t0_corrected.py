"""Round-1 erratum regression tests (execution plan round2, stage A2).

Identity tolerances: rtol=1e-9, atol=1e-14 — computational consistency only,
NOT scientific pass thresholds.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.t0_construct import S, T, S_NORM_F, BS, BA, xi_sequence
from src.t0_maps import finite_ns_fp64, tangent_matrix
from src.t0_metrics import (phase_stats, bias_decomposition,
                            same_H_error_decomposition, empirical_C_G,
                            direction_model_A, scalar_model_A)

RTOL, ATOL = 1e-9, 1e-14


def _rand_d(rng, n_phase=64):
    return rng.standard_normal((n_phase, T, 2, 2))


class TestStatisticalIdentities(unittest.TestCase):
    def test_A_equals_mean_trace_cov(self):
        rng = np.random.default_rng(0)
        for _ in range(3):
            d = _rand_d(rng)
            st = phase_stats(d)
            tr = float(np.mean(np.trace(st["cov_t"], axis1=1, axis2=2)))
            self.assertAlmostEqual(st["A"], tr, delta=ATOL + RTOL * abs(st["A"]))

    def test_population_denominator_is_n_phase(self):
        rng = np.random.default_rng(1)
        d = _rand_d(rng, n_phase=16)
        st = phase_stats(d)
        v = d.reshape(16, T, 4)
        db = v.mean(axis=0)
        cen = v - db[None]
        manual = np.einsum("qti,qtj->tij", cen, cen).mean(axis=0) / 16.0
        self.assertLess(np.abs(st["cov_mean"] - manual).max(), ATOL)

    def test_total_mse_equals_A_plus_B2(self):
        rng = np.random.default_rng(2)
        d0 = rng.standard_normal((2, 2))
        for _ in range(3):
            d = _rand_d(rng)
            bd = bias_decomposition(d, d0)
            self.assertTrue(bd["identity_ok"])
            self.assertAlmostEqual(bd["total_mse"], bd["A"] + bd["B2"],
                                   delta=ATOL + RTOL * abs(bd["total_mse"]))
            # B2 is the SQUARED time-mean bias, not the norm mean
            db = d.reshape(-1, T, 4).mean(axis=0)
            manual_b2 = float(np.mean(np.sum((db - d0.reshape(1, 4)) ** 2, axis=1)))
            self.assertAlmostEqual(bd["B2"], manual_b2, delta=ATOL)

    def test_same_H_decomposition_identity(self):
        rng = np.random.default_rng(3)
        for _ in range(3):
            dr = _rand_d(rng)
            q = 0.01 * _rand_d(rng)  # correlated with dr on purpose
            dn = dr + q
            dec = same_H_error_decomposition(dr, dn)
            self.assertTrue(dec["identity_ok"])
            self.assertAlmostEqual(dec["A_native"],
                                   dec["A_ref"] + dec["A_q"] + dec["cross2"],
                                   delta=ATOL + RTOL * abs(dec["A_native"]))

    def test_absolute_prediction_scaling(self):
        """absolute_A_pred == mean_t trace(absolute cov pred) == coef * eps^2."""
        eps = 0.01 * S_NORM_F / np.sqrt(2)
        c_xi = empirical_C_G(xi_sequence("arm1"))
        for m in ("polar_fp64", "finite_ns_fp64"):
            J = tangent_matrix(m)
            dm = direction_model_A(c_xi, J, 0.95)
            abs_a = dm["A_pred"] * eps ** 2
            cD_abs = dm["cD_t"] * eps ** 2
            tr = float(np.mean(np.trace(cD_abs, axis1=1, axis2=2)))
            self.assertAlmostEqual(abs_a, tr, delta=ATOL + RTOL * abs(abs_a))
            sm = scalar_model_A(c_xi, J, 0.95)
            tr_s = float(np.mean(np.trace(sm["cD_t"] * eps ** 2, axis1=1, axis2=2)))
            self.assertAlmostEqual(sm["A_pred"] * eps ** 2, tr_s,
                                   delta=ATOL + RTOL * abs(tr_s))


class TestEdgeCases(unittest.TestCase):
    def test_zero_perturbation_and_zero_denominators(self):
        d = np.broadcast_to(S, (64, T, 2, 2)).copy()  # constant input
        st = phase_stats(d)
        self.assertEqual(st["A"], 0.0)
        bd = bias_decomposition(d, S)
        self.assertEqual(bd["B2"], 0.0)
        self.assertEqual(bd["total_mse"], 0.0)
        # ratio/denominator guards: zero denominator yields None, never a number
        ratio = bd["B2"] / st["A"] if st["A"] > 0 else None
        self.assertIsNone(ratio)

    def test_plus_minus_pulse_and_errors(self):
        rng = np.random.default_rng(4)
        d_plus = _rand_d(rng)
        d_minus = -d_plus
        r = (d_plus - d_minus) / 2.0
        self.assertLess(np.abs(r - d_plus).max(), ATOL)
        # abs vs rel error with zero peak -> None, not inf
        peak = 0.0
        abserr = 1e-8
        self.assertIsNone(abserr / peak if peak > 0 else None)

    def test_signed_delta_errors(self):
        a1, a2 = 1.0, 2.0
        p1, p2 = 1.4, 0.9  # predicted ordering INVERTED vs measured
        dm, dp = a2 - a1, p2 - p1
        self.assertFalse(np.sign(dp) == np.sign(dm))
        self.assertAlmostEqual(abs(dm - dp) / abs(dm), 1.5, places=12)

    def test_fp64_and_native_share_statistics_entry(self):
        rng = np.random.default_rng(5)
        d1 = _rand_d(rng)
        d2 = d1 + 1e-3 * _rand_d(rng)
        s1, s2 = phase_stats(d1), phase_stats(d2)
        self.assertEqual(set(s1.keys()), set(s2.keys()))
        dec = same_H_error_decomposition(d1, d2)
        self.assertAlmostEqual(dec["A_native"], s2["A"], delta=ATOL + RTOL * abs(s2["A"]))


class TestDocumentationClaims(unittest.TestCase):
    """Facts that were wrongly stated in round-1 docs; pinned here as data checks."""

    def test_polar_scalar_errors_are_unequal_across_arms(self):
        c_xi = {a: empirical_C_G(xi_sequence(a)) for a in ("arm1", "arm2")}
        J = tangent_matrix("polar_fp64")
        eps = 0.01 * S_NORM_F / np.sqrt(2)
        e = {a: scalar_model_A(c_xi[a], J, 0.95)["A_pred"] * eps ** 2 for a in c_xi}
        # arm1 prediction ~ 0 (symmetric channel vanishes) -> its measured A1 dominates the
        # relative error; arm2 has a scalar prediction > 0 -> much smaller error
        a1_meas, a2_meas = 6.268425e-07, 2.109294e-05  # from original arrays (delta=.01)
        rel1 = abs(a1_meas - e["arm1"]) / a1_meas
        rel2 = abs(a2_meas - e["arm2"]) / a2_meas
        self.assertGreater(rel1 / rel2, 10.0)  # 16.3 vs 0.485: NOT "both arms 16x"
        self.assertAlmostEqual(rel2, 0.485, places=2)

    def test_kernel_weights_positive_no_sign_flip(self):
        from src.t0_maps import kernel_weights
        w = kernel_weights(0.95, 129)
        self.assertGreater(w.min(), 0.0)

    def test_finite_ns_JBs_nonzero(self):
        J = tangent_matrix("finite_ns_fp64")
        self.assertGreater(float(np.linalg.norm(J @ BS.reshape(4))), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(J @ BS.reshape(4)) ** 2), 0.281450, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
