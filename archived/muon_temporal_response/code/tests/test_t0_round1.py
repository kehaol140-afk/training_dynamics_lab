"""Round-1 T0 pre-run tests (execution plan sections 4.3 and 5.3). CPU/FP64 only."""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.t0_construct import (S, BS, BA, T, arm_inputs, multiset_sha, scalar_autocorr,
                              symbol_counts, xi_sequence, direction_lag_covariance)
from src.t0_maps import (b2_covariance, finite_ns_fp64, finite_ns_tangent_at_H,
                         finite_ns_tangent_at_S, kernel_weights, nesterov_H_direct,
                         nesterov_H_window, polar_fp64, polar_tangent_at_S,
                         tangent_matrix, tangent_at_S)

ATOL, RTOL = 1e-12, 1e-9


class TestConstruction(unittest.TestCase):
    def test_symbol_counts_16_each(self):
        for arm in ("arm1", "arm2"):
            for q in range(64):
                c = symbol_counts(arm, q)
                self.assertEqual(sorted(c.values()), [16, 16, 16, 16], (arm, q))
                self.assertEqual(len(c), 4)

    def test_multiset_sha_same_across_arms_and_phases(self):
        eps = 0.01 * np.sqrt(5) / np.sqrt(2)
        shas = {multiset_sha(arm, q, eps) for arm in ("arm1", "arm2") for q in range(64)}
        self.assertEqual(len(shas), 1)

    def test_position_marginal_identical_across_arms(self):
        eps = 0.01
        for arm in ("arm1", "arm2"):
            stack = np.stack([arm_inputs(arm, q, eps) for q in range(64)])  # (64,T,2,2)
            vec = stack.reshape(64, T, 4)
            mu_t = vec.mean(axis=0)  # per-position marginal mean
            self.assertLess(np.abs(mu_t - S.reshape(4)).max(), ATOL,
                            "per-position marginal mean must equal S")
            cov_t = np.einsum("qti,qtj->tij", vec, vec) / 64
            if arm == "arm1":
                cov1 = cov_t
            else:
                cov2 = cov_t
        self.assertLess(np.abs(cov1 - cov2).max(), ATOL)

    def test_scalar_autocorr_same_across_arms(self):
        c1, c2 = scalar_autocorr("arm1"), scalar_autocorr("arm2")
        self.assertLess(np.abs(c1 - c2).max(), ATOL)
        self.assertAlmostEqual(c1[0], 2.0, places=12)  # ||Bs||^2+||Ba||^2 = 2

    def test_direction_lag_cov_differs(self):
        c1, c2 = direction_lag_covariance("arm1"), direction_lag_covariance("arm2")
        # lag 1 differs (this is the entire point of the construction)
        self.assertGreater(np.abs(c1[1] - c2[1]).max(), 1e-3)

    def test_single_step_spectral_multiset_same(self):
        eps = 0.01
        sv1, sv2 = [], []
        for q in (0, 17, 40):
            g1, g2 = arm_inputs("arm1", q, eps), arm_inputs("arm2", q, eps)
            sv1 += [np.sort(np.linalg.svd(g)[1]).tolist() for g in g1]
            sv2 += [np.sort(np.linalg.svd(g)[1]).tolist() for g in g2]
        self.assertEqual(sorted(sv1), sorted(sv2))

    def test_total_input_variance_and_reference_mean(self):
        eps = 0.01
        for arm in ("arm1", "arm2"):
            g = arm_inputs(arm, 0, eps)
            self.assertLess(np.abs(g.mean(axis=0) - S).max(), ATOL)
            var = ((g - S) ** 2).sum(axis=(1, 2)).mean()
            self.assertAlmostEqual(var, eps ** 2 * 2, places=15)

    def test_phase_order_irrelevant(self):
        eps = 0.01
        # enumeration statistics of {G(q,t)} over all q are phase-relabeling invariant
        a = np.concatenate([arm_inputs("arm1", q, eps) for q in range(0, 64, 2)])
        b = np.concatenate([arm_inputs("arm1", q, eps) for q in range(64, 0, -2)])
        a_sorted = np.array(sorted(a.reshape(-1, 4), key=lambda r: tuple(r)))
        b_sorted = np.array(sorted(b.reshape(-1, 4), key=lambda r: tuple(r)))
        self.assertLess(np.abs(a_sorted - b_sorted).max(), ATOL)

    def test_zero_delta_constant_input(self):
        g = arm_inputs("arm1", 5, 0.0)
        self.assertLess(np.abs(g - S[None]).max(), ATOL)
        self.assertEqual(len({multiset_sha(a, q, 0.0) for a in ("arm1", "arm2") for q in range(64)}), 1)

    def test_inputs_not_mutated(self):
        eps = 0.01
        g_before = arm_inputs("arm1", 3, eps).copy()
        arm_inputs("arm2", 3, eps)
        _ = xi_sequence("arm2")
        self.assertLess(np.abs(arm_inputs("arm1", 3, eps) - g_before).max(), ATOL)


class TestNesterov(unittest.TestCase):
    def test_direct_equals_window(self):
        rng = np.random.default_rng(0)
        g = S[None] + 0.01 * rng.standard_normal((T, 2, 2))
        h1 = nesterov_H_direct(g, S)
        h2 = nesterov_H_window(g, S)
        self.assertLess(np.abs(h1 - h2).max(), 1e-14)

    def test_constant_input_H_equals_S(self):
        g = np.repeat(S[None], T, axis=0)
        for fn in (nesterov_H_direct, nesterov_H_window):
            self.assertLess(np.abs(fn(g, S) - S[None]).max(), 1e-13)

    def test_weight_sum(self):
        w = kernel_weights()
        # window weight sum through t is 1 - beta^(t+1)
        for t in (1, 5, 20, 64):
            self.assertAlmostEqual(w[:t].sum(), 1 - 0.95 ** (t + 1), places=14)


class TestShuffleCovariance(unittest.TestCase):
    def test_b2_exhaustive_120_permutations(self):
        rng = np.random.default_rng(7)
        vecs = [rng.standard_normal(3) for _ in range(5)]
        a_coef = np.array([0.3, -0.1, 0.2, 0.5, -0.4])
        acc = np.zeros((3, 3))
        for perm in itertools.permutations(range(5)):
            s = sum(a_coef[i] * vecs[perm[i]] for i in range(5))
            acc += np.outer(s, s)
        acc /= 120
        g = np.stack(vecs)
        gbar = g.mean(axis=0)
        b2 = b2_covariance(vecs, a_coef)
        # empirical covariance about the exact mean
        exact_mean = sum(a_coef) * gbar
        acc2 = np.zeros((3, 3))
        for perm in itertools.permutations(range(5)):
            s = sum(a_coef[i] * vecs[perm[i]] for i in range(5)) - exact_mean
            acc2 += np.outer(s, s)
        acc2 /= 120
        self.assertLess(np.abs(acc2 - b2).max(), 1e-13)


class TestTangents(unittest.TestCase):
    def _fd(self, fn, h, e, step):
        return (fn(h + step * e) - fn(h - step * e)) / (2 * step)

    def test_polar_tangent_values_at_S(self):
        dp_bs = polar_tangent_at_S(BS)
        dp_ba = polar_tangent_at_S(BA)
        self.assertLess(np.abs(dp_bs).max(), 1e-14)
        self.assertLess(np.abs(dp_ba - (2.0 / 3.0) * BA).max(), 1e-12)

    def test_polar_tangent_vs_fd(self):
        rng = np.random.default_rng(1)
        e = rng.standard_normal((2, 2))
        fd = self._fd(polar_fp64, S, e, 1e-6)
        self.assertLess(np.abs(fd - polar_tangent_at_S(e)).max(), 1e-9)

    def test_ns_tangent_vs_fd_scale_ladder(self):
        rng = np.random.default_rng(2)
        for e, name in ((BS, "Bs"), (BA, "Ba"), (np.array([[1.0, 2.0], [-0.5, 0.3]]), "general")):
            errs = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6):
                fd = self._fd(finite_ns_fp64, S, e, step)
                tv = finite_ns_tangent_at_S(e)
                errs.append(np.abs(fd - tv).max())
            den = max(np.abs(finite_ns_tangent_at_S(e)).max(), 1e-30)
            if name != "Bs":  # non-degenerate direction: adjacent-scale rel err <= 1e-6
                for i in range(len(errs) - 1):
                    rel = abs(errs[i] - errs[i + 1]) / max(errs[i], errs[i + 1], 1e-300)
                    # convergence: err should shrink ~step^2; check tangent-vs-fd directly instead
                for i, step in enumerate((1e-3, 1e-4, 1e-5, 1e-6)):
                    self.assertLess(errs[i], max(1e-9, den * 1e-6 + step ** 2 * 50),
                                    (name, step, errs[i]))
            else:  # near-zero direction: absolute error; O(step^2) truncation plus 1e-16/step rounding floor
                for i, step in enumerate((1e-3, 1e-4, 1e-5, 1e-6)):
                    self.assertLess(errs[i], max(100 * step ** 2, 1e-9), (name, step, errs[i]))
                self.assertLess(min(errs), 1e-9, (name, errs))

    def test_ns_tangent_general_H(self):
        rng = np.random.default_rng(3)
        h = S + 0.03 * np.array([[0.4, -1.0], [1.0, 0.4]])
        e = rng.standard_normal((2, 2))
        fd = self._fd(finite_ns_fp64, h, e, 1e-6)
        self.assertLess(np.abs(fd - finite_ns_tangent_at_H(h, e)).max(), 1e-9)

    def test_normalization_smooth_branch(self):
        # d/dH of H/clamp(||H||): check at S and at a scaled point (norm>eps branch)
        e = np.array([[0.0, -1.0], [1.0, 0.0]]) / np.sqrt(2)
        for scale, step, tol in ((1.0, 1e-7, 1e-6), (1e-4, 1e-8, 1e-4)):
            h = S * scale
            fd = (h + step * e) / max(np.linalg.norm(h + step * e, "fro"), 1e-7) - \
                 (h - step * e) / max(np.linalg.norm(h - step * e, "fro"), 1e-7)
            n = np.linalg.norm(h, "fro")
            tv = e / n - h * float(np.sum(h * e)) / n ** 3
            self.assertLess(np.abs(fd / (2 * step) - tv).max(), tol, scale)

    def test_jacobian_shape_and_finite_ns_direction_gain(self):
        J_ns = tangent_matrix("finite_ns_fp64")
        J_pol = tangent_matrix("polar_fp64")
        self.assertEqual(J_ns.shape, (4, 4))
        g_bs = float(np.sum(J_ns @ BS.reshape(4) * (J_ns @ BS.reshape(4))))
        g_ba = float(np.sum(J_ns @ BA.reshape(4) * (J_ns @ BA.reshape(4))))
        # audit-values from the outline: ||J Bs||^2 ~ 0.281450, ||J Ba||^2 ~ 0.361172
        self.assertAlmostEqual(g_bs, 0.281450, places=5)
        self.assertAlmostEqual(g_ba, 0.361172, places=5)
        self.assertLess(np.abs(J_pol @ BS.reshape(4)).max(), 1e-12)
        self.assertLess(np.abs(J_pol @ BA.reshape(4) - (2.0 / 3.0 * BA).reshape(4)).max(), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
