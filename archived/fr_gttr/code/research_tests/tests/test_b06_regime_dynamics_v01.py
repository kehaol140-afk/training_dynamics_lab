#!/usr/bin/env python3
"""Unit tests for fr.b06_regime_dynamics_v01 (CPU only, unittest)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr.b06_regime_dynamics_v01 import (  # noqa: E402
    GaussianHMM, PURITY_THRESHOLD, channels_to_matrix, drift_null_series,
    fit_gmm_grid, fit_switching_linear_maps, multivariate_phase_surrogate,
    occupancy_from_paths, persistence_stats, psi_channels_from_rows,
    q1_verdict, q2_chi_by_regime, q3a_verdict, q3b_local_vs_global_chi,
    retained_gain_decay, run_all_theory_tests, run_lengths, smooth_drift,
    split_train_test, Standardizer, t1_slds_recovery, t2_null_calibration,
    t3_drift_null_calibration, t4_switching_map_recovery,
    t5_local_global_chi_calibration, t6_metric_gate, transition_points,
    _sample_slds,
)


def _synth_rows(T=16, with_optional=True):
    rows = []
    for h in range(1, T + 1):
        r = {"h": h, "grad_norm": 0.5 + 0.01 * h,
             "update_norm": 0.2 + 0.005 * h,
             "train_loss": 7.0 - 0.01 * h,
             "preclip_norm": 0.6 + 0.01 * h,
             "lr_native": 5e-4, "clip_active": False}
        if with_optional:
            r.update({"m_norm": 0.3 + 0.001 * h, "v_norm": 0.04 + 0.0001 * h,
                      "g_cos": 0.1 if h > 1 else 0.0,
                      "upd_cos": 0.2 if h > 1 else 0.0,
                      "disp": 0.01 * h})
        rows.append(r)
    return rows


class TestFeatureConstruction(unittest.TestCase):
    def test_channels_present_and_shapes(self):
        ch = psi_channels_from_rows(_synth_rows(32))
        self.assertEqual(ch["log_g"].shape, (32,))
        self.assertEqual(ch["loss_slope"].shape, (32,))
        self.assertAlmostEqual(ch["loss_slope"][1], -0.01, places=12)
        self.assertAlmostEqual(ch["log_g"][0], np.log(0.51), places=12)
        m = channels_to_matrix(ch)
        self.assertEqual(m.shape, (32, 9))

    def test_optional_missing_nan(self):
        ch = psi_channels_from_rows(_synth_rows(8, with_optional=False))
        self.assertTrue(np.isnan(ch["log_m"]).all())
        self.assertTrue(np.isnan(ch["g_cos"]).all())


class TestStandardizationAndSplit(unittest.TestCase):
    def test_standardizer_train_only(self):
        rng = np.random.default_rng(0)
        mats = [rng.normal(size=(40, 3)) + k for k in range(4)]
        tr, te = split_train_test(mats, 0.75)
        self.assertEqual(len(tr[0]), 30)
        self.assertEqual(len(te[0]), 10)
        sc = Standardizer().fit(tr)
        z = sc.transform(np.concatenate(tr, axis=0))
        self.assertAlmostEqual(float(z.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(z.std()), 1.0, places=10)


class TestNulls(unittest.TestCase):
    def test_phase_surrogate_preserves_moments(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(64, 5)).cumsum(axis=0)
        Y = multivariate_phase_surrogate(X, rng)
        self.assertEqual(Y.shape, X.shape)
        np.testing.assert_allclose(Y.mean(axis=0), X.mean(axis=0), atol=1e-8)
        np.testing.assert_allclose(Y.std(axis=0), X.std(axis=0), atol=1e-8)

    def test_phase_surrogate_differs_from_original(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(64, 4))
        Y = multivariate_phase_surrogate(X, rng)
        self.assertGreater(float(np.abs(Y - X).mean()), 1e-6)

    def test_drift_null_shape_and_smooth_drift(self):
        rng = np.random.default_rng(3)
        mats = [rng.normal(size=(64, 6)) for _ in range(5)]
        d = smooth_drift(mats[0], w=9)
        self.assertEqual(d.shape, mats[0].shape)
        nulls = drift_null_series(mats, rng)
        self.assertEqual(len(nulls), 5)
        self.assertEqual(nulls[0].shape, mats[0].shape)


class TestHMM(unittest.TestCase):
    def test_em_loglik_non_decreasing(self):
        rng = np.random.default_rng(4)
        seqs, _, _ = _sample_slds(rng, 6, 40, 4, K=2)
        hmm = GaussianHMM(2, seed=11, max_iter=60).fit(seqs)
        hist = hmm.loglik_history_
        self.assertGreaterEqual(len(hist), 5)
        diffs = np.diff(hist)
        self.assertGreaterEqual(float(diffs.min()), -1e-6)

    def test_viterbi_recovers_separated_states(self):
        rng = np.random.default_rng(5)
        # two extremely separated blocks
        X = np.concatenate([np.full((20, 3), -3.0), np.full((20, 3), 3.0)])
        X += 0.01 * rng.normal(size=X.shape)
        hmm = GaussianHMM(2, seed=7, max_iter=100).fit([X])
        path, _ = hmm.viterbi(X)
        self.assertTrue((path[:20] == path[0]).all())
        self.assertTrue((path[20:] == path[-1]).all())
        self.assertNotEqual(path[0], path[-1])

    def test_hmm_beats_gmm_on_switching_data(self):
        rng = np.random.default_rng(6)
        seqs, _, _ = _sample_slds(rng, 12, 48, 5, K=2)
        tr, te = split_train_test(seqs, 0.75)
        from fr.b06_regime_dynamics_v01 import fit_hmm_grid
        hmm_table, _ = fit_hmm_grid(tr, te, ks=(1, 2), seed=8)
        self.assertGreater(hmm_table[1]["test_ll"], hmm_table[0]["test_ll"])
        gmm_table = fit_gmm_grid(tr, te, ks=(1, 2), seed=8)
        self.assertTrue(any(r["k"] == 2 for r in gmm_table))


class TestPersistenceStats(unittest.TestCase):
    def test_run_lengths_and_transitions(self):
        states = [0, 0, 1, 1, 1, 0]
        self.assertEqual(run_lengths(states), [(0, 2), (1, 3), (0, 1)])
        self.assertEqual(transition_points(states), [3, 6])
        pers = persistence_stats([states, [1, 1, 1, 1]])
        self.assertEqual(pers["n_runs"], 4)
        self.assertAlmostEqual(pers["mean_run_length"], 2.5)
        self.assertAlmostEqual(pers["occupancy"][0], 3 / 10)
        self.assertAlmostEqual(pers["occupancy"][1], 7 / 10)

    def test_occupancy_from_paths(self):
        occ = occupancy_from_paths([[0, 0, 1], [1, 1, 1]], K=2)
        self.assertAlmostEqual(occ[0], 2 / 6)
        self.assertAlmostEqual(occ[1], 4 / 6)


class TestSwitchingMaps(unittest.TestCase):
    def test_planted_maps_local_beats_global(self):
        res = t4_switching_map_recovery()
        self.assertTrue(res["ok"], msg=str(res))

    def test_q3a_verdict_paths(self):
        good = {"test_mse": {"local_within": 0.8, "local_across": 1.5,
                             "global_within": 1.2, "global_across": 1.3,
                             "n_local_within": 100, "n_local_across": 10,
                             "n_global_within": 100, "n_global_across": 10}}
        v = q3a_verdict(good, {"across_weight": 1.0})
        self.assertEqual(v["verdict"], "SUPPORTED")
        bad = {"test_mse": {"local_within": 1.2, "local_across": 1.5,
                            "global_within": 0.8, "global_across": 1.3,
                            "n_local_within": 100, "n_local_across": 10,
                            "n_global_within": 100, "n_global_across": 10}}
        v = q3a_verdict(bad, {"across_weight": 1.0})
        self.assertEqual(v["verdict"], "NOT_SUPPORTED")


class TestQ2Q3b(unittest.TestCase):
    def _occ(self):
        return {(2030, 46): {0: 0.9, 1: 0.1}, (2030, 82): {0: 0.1, 1: 0.9},
                (2031, 46): {0: 0.85, 1: 0.15}, (2031, 82): {0: 0.15, 1: 0.85},
                (2032, 46): {0: 0.8, 1: 0.2}, (2032, 82): {0: 0.2, 1: 0.8}}

    def _chi(self, base_by_cell):
        rows = []
        for (s, t), v in base_by_cell.items():
            for H in (4, 16, 64):
                rows.append({"seed": s, "step": t, "protocol": "persistent",
                             "H": H, "chi_dev": v, "chi_audit": v})
        return rows

    def test_q2_runs_and_verdict_structure(self):
        chi = self._chi({(2030, 46): -0.01, (2030, 82): -0.004,
                         (2031, 46): -0.011, (2031, 82): -0.005,
                         (2032, 46): -0.012, (2032, 82): -0.003})
        rules = {"alpha": 0.05, "min_seed_consistent": 2, "n_perm": 500,
                 "seed": 42}
        out = q2_chi_by_regime(chi, self._occ(), rules)
        self.assertIn(out["verdict"], {"SUPPORTED", "NOT_SUPPORTED", "UNRESOLVED"})
        self.assertEqual(len(out["rows"]), 3)  # 1 protocol x 3 H
        row = out["rows"][0]
        self.assertIn("perm_p", row)
        self.assertLess(row["perm_p"], 1.0)

    def test_q3b_pure_cells_split(self):
        chi = self._chi({(2030, 46): -0.01, (2030, 82): -0.004,
                         (2031, 46): -0.011, (2031, 82): -0.005,
                         (2032, 46): -0.012, (2032, 82): -0.003})
        rules = {"min_seed_folds": 2}
        seed_of = lambda c: c[0]  # noqa: E731
        out = q3b_local_vs_global_chi(chi, self._occ(), seed_of, rules)
        self.assertEqual(len(out["fold_rows"]), 3 * 2 * 6)
        self.assertIn("RMSE", out["metric"]["local"])
        for r in out["fold_rows"]:
            self.assertEqual(r["split"], "within_cluster")  # all occ max >= 0.8


class TestRetainedGain(unittest.TestCase):
    def test_decay_rows_and_corr(self):
        native, pulse, tp = {}, {}, {}
        # 8 trajectories; exponential after-effect g(t) = -a*exp(-(t-5)/tau)
        # with tau growing across trajectories; later first transition pairs
        # with larger tau -> positive Spearman by design
        for i, (seed, stream) in enumerate(
                [(2030, s) for s in (7500, 7501, 7502, 7503)]
                + [(2031, s) for s in (7500, 7501, 7502, 7503)]):
            key = (seed, 46, stream, 0.05)
            tau = 2.0 * (i + 1)
            native[key] = [{"h": h, "train_loss": 7.0 - 0.01 * h} for h in range(5, 65)]
            pulse[key] = [{"h": h,
                           "train_loss": 7.0 - 0.01 * h - 0.05 * float(np.exp(-(h - 5) / tau))}
                          for h in range(5, 65)]
            tp[(seed, 46, stream)] = [8 + 4 * i]
        out = retained_gain_decay(pulse, native, tp)
        self.assertEqual(len(out["rows"]), 8)
        self.assertIsNotNone(out["corr"]["spearman_rho"])
        self.assertGreater(out["corr"]["spearman_rho"], 0.5)


class TestVerdicts(unittest.TestCase):
    def test_q1_verdict_paths(self):
        base = {"best_k": 2,
                "hmm_table": [{"k": 1, "test_ll": -10.0}, {"k": 2, "test_ll": -8.0}],
                "gmm_table": [{"k": 1, "test_ll": -10.0}, {"k": 2, "test_ll": -8.5}],
                "persistence": {"mean_run_length": 20.0, "switch_rate": 0.02},
                "phase_null": {"test_ll": {"p95": -9.0},
                               "switch_rate": {"p95": 0.05}},
                "drift_null": {"mean_run_length": {"p95": 10.0}}}
        v = q1_verdict(base, {"min_mean_run_length": 8})
        self.assertEqual(v["verdict"], "SUPPORTED")
        bad = dict(base)
        bad["hmm_table"] = [{"k": 1, "test_ll": -8.0}, {"k": 2, "test_ll": -10.0}]
        v = q1_verdict(bad, {"min_mean_run_length": 8})
        self.assertEqual(v["verdict"], "NOT_SUPPORTED")
        k1 = dict(base)
        k1["best_k"] = 1
        v = q1_verdict(k1, {"min_mean_run_length": 8})
        self.assertEqual(v["verdict"], "NOT_SUPPORTED")


class TestTheoryBatteries(unittest.TestCase):
    def test_t1_slds_recovery(self):
        res = t1_slds_recovery()
        self.assertTrue(res["ok"], msg=str(res))

    def test_t2_null_calibration_reduced(self):
        res = t2_null_calibration(n_rep=4, n_surrogates=9, max_iter=60)
        self.assertTrue(res["ok"], msg=str({k: v for k, v in res.items() if k != "details"}))

    def test_t3_drift_null_calibration_reduced(self):
        res = t3_drift_null_calibration(n_rep=3, n_surrogates=9, max_iter=60)
        self.assertTrue(res["ok"], msg=str({k: v for k, v in res.items() if k != "details"}))

    def test_t4_switching_map_recovery(self):
        res = t4_switching_map_recovery()
        self.assertTrue(res["ok"], msg=str(res))

    def test_t5_local_global_chi_calibration(self):
        res = t5_local_global_chi_calibration()
        self.assertTrue(res["ok"], msg=str(res))

    def test_t6_metric_gate(self):
        res = t6_metric_gate()
        self.assertTrue(res["ok"], msg=str(res))


if __name__ == "__main__":
    unittest.main()
