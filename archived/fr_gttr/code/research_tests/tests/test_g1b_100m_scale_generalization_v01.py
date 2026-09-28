"""Unit tests for G1b ~100M scale-generalization module (pure, no GPU).

Run from repo root:
  /home/b205/miniforge3/envs/fr_minimind/bin/python -m unittest \
      research.tests.test_g1b_100m_scale_generalization_v01
"""
from __future__ import annotations

import math
import unittest

from fr import g1b_100m_scale_generalization_v01 as g1b
from fr.a0c_adamw_response_propagation_v01 import _metric as frozen_metric
from fr.a0c_adamw_response_propagation_v01 import relative_error as frozen_rel_err
from fr.batch_manifest import manifest_sha256


class TestSeedDerivation(unittest.TestCase):
    def test_two_level_seed_deterministic(self):
        a = g1b.two_level_seed(g1b.E1_BASE_SEED, 2042, 26, 0)
        b = g1b.two_level_seed(g1b.E1_BASE_SEED, 2042, 26, 0)
        self.assertEqual(a, b)
        self.assertIsInstance(a, int)

    def test_two_level_seed_distinct_inputs(self):
        seeds = {
            g1b.two_level_seed(g1b.E2_BASE_SEED, train_seed, step, stream_id)
            for train_seed in g1b.TRAIN_SEEDS
            for step in g1b.STAGE_STEPS.values()
            for stream_id in range(g1b.E2_STREAMS)
        }
        self.assertEqual(
            len(seeds),
            len(g1b.TRAIN_SEEDS) * len(g1b.STAGE_STEPS) * g1b.E2_STREAMS,
        )

    def test_g1b_stream_seeds_do_not_collide_with_g1a(self):
        # paired-scale design: same seeds/steps, fresh base seeds
        self.assertNotEqual(g1b.E1_BASE_SEED, 408202601)  # G1a E1 base
        self.assertNotEqual(g1b.E2_BASE_SEED, 409202601)  # G1a E2 base
        g1b_seeds = {
            g1b.two_level_seed(g1b.E2_BASE_SEED, s, st, r)
            for s in g1b.TRAIN_SEEDS for st in g1b.STAGE_STEPS.values()
            for r in range(g1b.E2_STREAMS)
        }
        g1a_seeds = {
            g1b.two_level_seed(409202601, s, st, r)
            for s in g1b.TRAIN_SEEDS for st in g1b.STAGE_STEPS.values()
            for r in range(g1b.E2_STREAMS)
        }
        self.assertTrue(g1b_seeds.isdisjoint(g1a_seeds))

    def test_batch_seed_derivation(self):
        s = g1b.two_level_seed(g1b.E1_BASE_SEED, 2042, 64, 3)
        self.assertEqual(
            g1b.batch_seed_from_stream_seed(s),
            g1b.batch_seed_from_stream_seed(s),
        )
        self.assertNotEqual(s, g1b.batch_seed_from_stream_seed(s))

    def test_stage_of_step(self):
        self.assertEqual(g1b.stage_of_step(26), "EARLY")
        self.assertEqual(g1b.stage_of_step(64), "MIDDLE")
        self.assertEqual(g1b.stage_of_step(102), "LATE")
        with self.assertRaises(ValueError):
            g1b.stage_of_step(46)


class TestFrozenConfig(unittest.TestCase):
    def test_expected_unique_params_in_mandate_window(self):
        self.assertGreaterEqual(g1b.EXPECTED_UNIQUE_PARAMS, 95_000_000)
        self.assertLessEqual(g1b.EXPECTED_UNIQUE_PARAMS, 110_000_000)
        self.assertEqual(g1b.EXPECTED_UNIQUE_PARAMS, 100_784_000)

    def test_scale_only_structure_matches_protocol(self):
        # hidden=768, layers=13, heads=12, kv=6, head_dim=64
        self.assertEqual(g1b.EXPECTED_UNIQUE_PARAMS % 64, 0)
        # heads*head_dim == hidden (12*64 == 768)
        self.assertEqual(12 * 64, 768)


class TestMetricIdentity(unittest.TestCase):
    def test_metric_is_frozen_object(self):
        self.assertIs(g1b._metric, frozen_metric)

    def test_relative_error_is_frozen_object(self):
        self.assertIs(g1b.relative_error, frozen_rel_err)

    def test_relative_error_semantics(self):
        self.assertEqual(g1b.relative_error(2.0, 2.5), 0.25)
        self.assertEqual(g1b.relative_error(0.0, 0.0), 0.0)
        self.assertIsNone(g1b.relative_error(0.0, 1.0))


class TestManifestValidation(unittest.TestCase):
    def _payload(self, **meta_over):
        batches = [[1, 2, 3, 4], [5, 6, 7, 8]] * 8
        metadata = {
            "protocol": g1b.E2_MANIFEST_PROTOCOL,
            "train_seed": 2042,
            "state_step": 26,
            "stream_id": 0,
            "stream_seed": 12345,
            "batch_seed": 999,
            "epsilon": 1e-4,
            "rho": 0.25,
        }
        metadata.update(meta_over)
        return {
            "manifest_version": "0.2",
            "sha256": manifest_sha256(batches),
            "metadata": metadata,
            "batches": batches,
        }

    def test_valid_manifest_passes(self):
        payload = self._payload()
        out = g1b.validate_g1b_e2_manifest(
            payload, expected_seed=2042, expected_step=26, expected_stream=0,
            expected_content_sha=payload["sha256"], accumulation_steps=2,
        )
        self.assertEqual(out["batches"], payload["batches"])
        self.assertEqual(out["content_sha256"], payload["sha256"])

    def test_wrong_protocol_rejected(self):
        payload = self._payload(protocol="FR_G1B_100M_E1_STREAM")
        with self.assertRaises(ValueError):
            g1b.validate_g1b_e2_manifest(
                payload, expected_seed=2042, expected_step=26, expected_stream=0,
                expected_content_sha=payload["sha256"], accumulation_steps=2,
            )

    def test_identity_mismatch_rejected(self):
        payload = self._payload()
        with self.assertRaises(ValueError):
            g1b.validate_g1b_e2_manifest(
                payload, expected_seed=2043, expected_step=26, expected_stream=0,
                expected_content_sha=payload["sha256"], accumulation_steps=2,
            )

    def test_content_tamper_rejected(self):
        payload = self._payload()
        payload["batches"][0][0] = 777
        with self.assertRaises(ValueError):
            g1b.validate_g1b_e2_manifest(
                payload, expected_seed=2042, expected_step=26, expected_stream=0,
                expected_content_sha=manifest_sha256(
                    [[1, 2, 3, 4], [5, 6, 7, 8]] * 8
                ),
                accumulation_steps=2,
            )

    def test_missing_epsilon_rejected(self):
        payload = self._payload()
        del payload["metadata"]["epsilon"]
        with self.assertRaises(ValueError):
            g1b.validate_g1b_e2_manifest(
                payload, expected_seed=2042, expected_step=26, expected_stream=0,
                expected_content_sha=payload["sha256"], accumulation_steps=2,
            )


def _synthetic_rows(c_noise: float = 0.0, b_noise: float = 0.5):
    """27 official rows where C tracks FD and B deviates (H>1)."""
    import numpy as np

    rng = np.random.default_rng(7)
    rows = []
    for seed in g1b.TRAIN_SEEDS:
        for stage, step in g1b.STAGE_STEPS.items():
            base = float(rng.normal(0.0, 1.0))
            for h in g1b.HORIZONS_OFFICIAL:
                chi_fd = base * h + float(rng.normal(0.0, 0.05))
                chi_c = chi_fd + float(rng.normal(0.0, c_noise))
                if h == 1:
                    chi_b = chi_c  # structural tie at H=1
                else:
                    chi_b = chi_fd + b_noise + float(rng.normal(0.0, b_noise))
                rows.append({
                    "seed": seed, "stage": stage, "step": step, "H": h,
                    "chi_fd": chi_fd, "chi_B": chi_b, "chi_C": chi_c,
                })
    return rows


class TestAggregations(unittest.TestCase):
    def test_grid_size_enforced(self):
        with self.assertRaises(ValueError):
            g1b.aggregate_metrics(_synthetic_rows()[:26])

    def test_paired_counts(self):
        stats = g1b.aggregate_metrics(_synthetic_rows())
        paired = stats["paired_h_gt_1"]
        self.assertEqual(paired["n"], 18)
        self.assertEqual(paired["losses"], 0)
        self.assertEqual(paired["wins"] + paired["ties"], 18)
        self.assertGreater(paired["mean_paired_improvement"], 0.0)

    def test_relative_error_summary(self):
        stats = g1b.aggregate_metrics(_synthetic_rows())
        rel = stats["relative_error_summary"]
        self.assertLess(rel["C_mean_h_gt_1"], rel["B_mean_h_gt_1"])

    def test_seed_means_present(self):
        stats = g1b.aggregate_metrics(_synthetic_rows())
        for seed in g1b.TRAIN_SEEDS:
            self.assertIn(str(seed), stats["seed_mean_paired_improvement"])
            self.assertGreater(
                stats["seed_mean_paired_improvement"][str(seed)], 0.0
            )


class TestVerdict(unittest.TestCase):
    def _stats(self, rows):
        return g1b.aggregate_metrics(rows)

    def test_strong_on_clean_synthetic(self):
        # shrink noise so R2 thresholds pass
        rows = _synthetic_rows(c_noise=1e-6, b_noise=0.4)
        verdict = g1b.decide_verdict(self._stats(rows))
        self.assertEqual(verdict["verdict"], "STRONG_SCALE_REPLICATION")
        self.assertTrue(verdict["conditions"]["cond1_c_pooled_r2_ge_0.90"]["pass"])
        self.assertTrue(verdict["conditions"]["cond2_h_gt_1_win_fraction_ge_0.80"]["pass"])
        self.assertTrue(verdict["conditions"]["cond3_h8_paired_and_predictive"]["pass"])
        self.assertTrue(verdict["conditions"]["cond4_seed_positive_ge_2"]["pass"])

    def test_no_replication_when_c_worse(self):
        rows = _synthetic_rows()  # default: C good, B bad at H>1
        # swap so C is worse than B at H>1
        for row in rows:
            if row["H"] > 1:
                row["chi_B"], row["chi_C"] = row["chi_C"], row["chi_B"]
        verdict = g1b.decide_verdict(self._stats(rows))
        self.assertEqual(verdict["verdict"], "NO_SCALE_REPLICATION")

    def test_tie_tolerance(self):
        rows = _synthetic_rows()
        stats = g1b.aggregate_metrics(rows)
        paired = stats["paired_h_gt_1"]
        self.assertEqual(paired["ties"], 0)
        # H=1 rows must not appear in paired stats
        self.assertTrue(all(p["H"] > 1 for p in paired["rows"]))

    def test_thresholds_are_frozen_values(self):
        self.assertEqual(g1b.VERDICT_THRESHOLDS["c_pooled_r2_min"], 0.90)
        self.assertEqual(g1b.VERDICT_THRESHOLDS["h_gt_1_win_fraction_min"], 0.80)
        self.assertEqual(g1b.VERDICT_THRESHOLDS["h8_r2_min"], 0.50)
        self.assertEqual(g1b.VERDICT_THRESHOLDS["seed_positive_count_min"], 2)


class TestCellErrors(unittest.TestCase):
    def test_zero_observed_handled(self):
        rows = [{
            "seed": 2042, "step": 26, "H": 1,
            "chi_fd": 0.0, "chi_B": 0.0, "chi_C": 0.0,
        }]
        errors = g1b.cell_relative_errors(rows)
        key = (2042, 26, 1)
        self.assertEqual(errors[key]["rel_err_B"], 0.0)
        rows[0]["chi_B"] = 1.0
        errors = g1b.cell_relative_errors(rows)
        self.assertTrue(math.isnan(errors[key]["rel_err_B"]))


if __name__ == "__main__":
    unittest.main()
