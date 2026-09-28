"""Unit tests for FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01 (pure CPU)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fr.a0c_adamw_response_propagation_v01 import relative_error as frozen_relative_error
from fr.batch_manifest import manifest_sha256, save_manifest
from fr.g1a_26m_scale_generalization_v01 import (
    E1_BASE_SEED,
    E2_BASE_SEED,
    HORIZONS_OFFICIAL,
    STAGES,
    TRAIN_SEEDS,
    aggregate_metrics as g1a_aggregate_metrics,
    cell_relative_errors as g1a_cell_relative_errors,
)
from fr.g2_data_distribution_generalization_v01 import (
    DATASET_MANIFEST_PROTOCOL,
    D1_POOL_SOURCE_RECORDS,
    E2_MANIFEST_PROTOCOL,
    G2_D1_POOL_SEED,
    G2_EVAL_SIZE,
    G2_PARTITION_SEED,
    G2_PROBE_SIZE,
    PROTOCOL,
    VERDICT_THRESHOLDS,
    cell_relative_errors,
    aggregate_metrics,
    cluster_bootstrap_paired_improvement,
    decide_data_verdict,
    jaccard,
    relative_error,
    select_pool_indices,
    shingle_digest,
    validate_dataset_manifest,
    validate_g2_e2_manifest,
)
from fr.pilot_partition import build_partition

REPO_ROOT = Path(__file__).resolve().parents[2]
G1A_CONFIG_PATH = REPO_ROOT / "research/configs/g1a_26m_v01.json"
G2_CONFIG_PATH = REPO_ROOT / "research/configs/g2_data_distribution_generalization_v01.json"
D0_PATH = REPO_ROOT / "outputs/e3_confirmatory/v07/e3c_subset.jsonl"
D0_SHA256 = "1e657af9baae0784e582d82ef936160473aa069878e422ef48683a718056b0f4"


class TestPoolAndPartition(unittest.TestCase):
    def test_pool_selection_deterministic_sorted_unique(self):
        n_source = 1_100_000
        a = select_pool_indices(n_source, G2_D1_POOL_SEED, D1_POOL_SOURCE_RECORDS)
        b = select_pool_indices(n_source, G2_D1_POOL_SEED, D1_POOL_SOURCE_RECORDS)
        self.assertEqual(a, b)
        self.assertEqual(len(a), D1_POOL_SOURCE_RECORDS)
        self.assertEqual(len(set(a)), len(a))
        self.assertEqual(a, sorted(a))
        self.assertTrue(all(0 <= i < n_source for i in a))

    def test_pool_selection_seed_sensitive(self):
        a = select_pool_indices(50_000, G2_D1_POOL_SEED, 4096)
        b = select_pool_indices(50_000, G2_D1_POOL_SEED + 1, 4096)
        self.assertNotEqual(a, b)

    def test_pool_selection_raises_when_source_smaller(self):
        with self.assertRaises(ValueError):
            select_pool_indices(100, G2_D1_POOL_SEED, 4096)

    def test_partition_sizes_and_disjointness(self):
        part = build_partition(D1_POOL_SOURCE_RECORDS, G2_PARTITION_SEED, G2_PROBE_SIZE, G2_EVAL_SIZE)
        self.assertEqual(len(part["probe_indices"]), G2_PROBE_SIZE)
        self.assertEqual(len(part["eval_indices"]), G2_EVAL_SIZE)
        self.assertEqual(len(part["train_indices"]),
                         D1_POOL_SOURCE_RECORDS - G2_PROBE_SIZE - G2_EVAL_SIZE)
        a, b, c = (set(part["train_indices"]), set(part["probe_indices"]),
                   set(part["eval_indices"]))
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(part["dataset_len"], D1_POOL_SOURCE_RECORDS)
        again = build_partition(D1_POOL_SOURCE_RECORDS, G2_PARTITION_SEED, G2_PROBE_SIZE, G2_EVAL_SIZE)
        self.assertEqual(part["sha256"], again["sha256"])


class TestVerdictTiers(unittest.TestCase):
    def _stats(self, *, r2=0.99, wins=18, n=18, h8_r2=0.9, h8_corr=0.9,
               seed_means=(0.1, 0.1, 0.1), c_mean_hgt1=0.01, b_mean_hgt1=0.1):
        rows = []
        for seed in TRAIN_SEEDS:
            for stage in STAGES:
                for h in (4, 8):
                    rows.append({"seed": seed, "stage": stage, "H": h,
                                 "err_B": b_mean_hgt1, "err_C": c_mean_hgt1})
        return {
            "pooled": {"C": {"R2": r2}},
            "paired_h_gt_1": {"n": n, "wins": wins, "rows": rows},
            "by_horizon": {"8": {"C": {"R2": h8_r2, "corr": h8_corr}}},
            "seed_mean_paired_improvement": {str(s): v for s, v in zip(TRAIN_SEEDS, seed_means)},
            "relative_error_summary": {"C_mean_h_gt_1": c_mean_hgt1,
                                       "B_mean_h_gt_1": b_mean_hgt1},
        }

    def test_strong(self):
        out = decide_data_verdict(stats=self._stats())
        self.assertEqual(out["verdict"], "STRONG_DATA_REPLICATION")
        self.assertTrue(out["pass"])
        self.assertTrue(all(c["pass"] for c in out["conditions"].values()))

    def test_partial_when_cond4_fails(self):
        out = decide_data_verdict(stats=self._stats(seed_means=(0.1, -0.1, 0.0)))
        self.assertEqual(out["verdict"], "PARTIAL_DATA_REPLICATION")
        self.assertFalse(out["pass"])
        self.assertFalse(out["conditions"]["cond4_seed_positive_ge_2"]["pass"])

    def test_no_when_c_not_better(self):
        out = decide_data_verdict(stats=self._stats(c_mean_hgt1=0.2, b_mean_hgt1=0.1,
                                                    wins=0, seed_means=(-0.1, -0.1, -0.1)))
        self.assertEqual(out["verdict"], "NO_DATA_REPLICATION")
        self.assertFalse(out["pass"])

    def test_cond_thresholds_are_frozen_values(self):
        self.assertEqual(VERDICT_THRESHOLDS["c_pooled_r2_min"], 0.90)
        self.assertEqual(VERDICT_THRESHOLDS["h_gt_1_win_fraction_min"], 0.80)
        self.assertEqual(VERDICT_THRESHOLDS["h8_r2_min"], 0.50)
        self.assertEqual(VERDICT_THRESHOLDS["seed_positive_count_min"], 2)
        self.assertEqual(VERDICT_THRESHOLDS["tie_tol"], 1e-12)

    def test_cond2_boundary_15_of_18(self):
        out = decide_data_verdict(stats=self._stats(wins=15, n=18))
        self.assertTrue(out["conditions"]["cond2_h_gt_1_win_fraction_ge_0.80"]["pass"])
        out14 = decide_data_verdict(stats=self._stats(wins=14, n=18))
        self.assertFalse(out14["conditions"]["cond2_h_gt_1_win_fraction_ge_0.80"]["pass"])

    def test_flag_tiers(self):
        self.assertEqual(decide_data_verdict(stats={}, contract_incompatible=True)["verdict"],
                         "CONTRACT_INCOMPATIBLE")
        self.assertEqual(decide_data_verdict(stats={}, dataset_unavailable=True)["verdict"],
                         "DATASET_UNAVAILABLE")
        self.assertEqual(decide_data_verdict(stats={}, resource_limit=True)["verdict"],
                         "RESOURCE_LIMIT")


class TestBootstrap(unittest.TestCase):
    def _cells(self):
        cells = []
        # heterogeneous improvements so the resample distribution is non-degenerate
        b_errs = {"EARLY": (0.20, 0.18), "MIDDLE": (0.12, 0.10), "LATE": (0.06, 0.05)}
        c_errs = {"EARLY": (0.05, 0.04), "MIDDLE": (0.03, 0.02), "LATE": (0.02, 0.015)}
        for seed in TRAIN_SEEDS:
            for stage in STAGES:
                for k, h in enumerate((4, 8)):
                    cells.append({
                        "seed": seed, "stage": stage, "H": h,
                        "err_B": b_errs[stage][k], "err_C": c_errs[stage][k],
                    })
        return cells

    def test_deterministic_and_labeled(self):
        a = cluster_bootstrap_paired_improvement(self._cells(), n_resamples=2000, seed=20260916)
        b = cluster_bootstrap_paired_improvement(self._cells(), n_resamples=2000, seed=20260916)
        self.assertEqual(a, b)
        self.assertEqual(a["n_cells"], 18)
        self.assertEqual(a["n_resamples"], 2000)
        self.assertEqual(a["rng_seed"], 20260916)
        self.assertGreater(a["observed_mean_improvement"], 0.0)
        bb = a["bootstrap"]
        self.assertLess(bb["pi95_low"], bb["pi95_high"])
        self.assertLess(bb["pi95_low"], a["observed_mean_improvement"])
        self.assertGreater(bb["pi95_high"], a["observed_mean_improvement"])
        self.assertGreaterEqual(bb["fraction_resamples_improvement_positive"], 0.0)
        self.assertLessEqual(bb["fraction_resamples_improvement_positive"], 1.0)
        self.assertEqual(bb["fraction_resamples_improvement_positive"], 1.0)
        self.assertIn("NOT a", a["interpretation_note"])
        self.assertIn("p-value", a["interpretation_note"])

    def test_empty_input(self):
        out = cluster_bootstrap_paired_improvement([], n_resamples=10, seed=1)
        self.assertEqual(out["n_cells"], 0)
        self.assertTrue(math.isnan(out["mean"]))

    def test_all_negative_improvement(self):
        cells = [dict(c, err_B=0.01, err_C=0.10) for c in self._cells()]
        out = cluster_bootstrap_paired_improvement(cells, n_resamples=500, seed=7)
        self.assertEqual(out["bootstrap"]["fraction_resamples_improvement_positive"], 0.0)
        self.assertLess(out["bootstrap"]["pi95_high"], 0.0)


class TestAggregationReuse(unittest.TestCase):
    def test_import_identity_not_redefinition(self):
        self.assertIs(cell_relative_errors, g1a_cell_relative_errors)
        self.assertIs(aggregate_metrics, g1a_aggregate_metrics)
        self.assertIs(relative_error, frozen_relative_error)

    def test_relative_error_semantics(self):
        self.assertEqual(frozen_relative_error(0.0, 0.0), 0.0)
        self.assertIsNone(frozen_relative_error(0.0, 1.0))
        self.assertAlmostEqual(frozen_relative_error(2.0, 1.0), 0.5)

    def test_cell_errors_keys(self):
        rows = [{"seed": 2042, "step": 26, "H": 4, "chi_fd": 1.0,
                 "chi_B": 0.9, "chi_C": 0.99}]
        out = cell_relative_errors(rows)
        key = (2042, 26, 4)
        for k in ("abs_err_B", "abs_err_C", "rel_err_B", "rel_err_C",
                  "signed_err_B", "signed_err_C"):
            self.assertIn(k, out[key])
        self.assertAlmostEqual(out[key]["rel_err_B"], 0.1)
        self.assertAlmostEqual(out[key]["rel_err_C"], 0.01)


class TestDatasetManifest(unittest.TestCase):
    def _manifest(self):
        return {
            "protocol": DATASET_MANIFEST_PROTOCOL,
            "created_at_utc": "2026-09-16T00:00:00Z",
            "d0": {"path": "a", "sha256": "x", "records": 4096, "bytes": 1},
            "d1": {"path": "b", "sha256": "y", "records": 100, "bytes": 2},
            "d1_pool": {"path": "c", "sha256": "z", "records": 4096, "bytes": 3},
            "partition": {"path": "d"},
            "overlap_diagnostic": {},
            "token_statistics_sample": {},
        }

    def test_valid_passes(self):
        validate_dataset_manifest(self._manifest())

    def test_missing_key_fails(self):
        payload = self._manifest()
        del payload["overlap_diagnostic"]
        with self.assertRaises(ValueError):
            validate_dataset_manifest(payload)

    def test_wrong_protocol_fails(self):
        payload = self._manifest()
        payload["protocol"] = "FR_G1A_26M_SCALE_GENERALIZATION_V01"
        with self.assertRaises(ValueError):
            validate_dataset_manifest(payload)

    def test_pool_record_count_fails(self):
        payload = self._manifest()
        payload["d1_pool"]["records"] = 4000
        with self.assertRaises(ValueError):
            validate_dataset_manifest(payload)


class TestG2E2ManifestValidator(unittest.TestCase):
    def _payload(self, protocol=E2_MANIFEST_PROTOCOL):
        batches = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        return {
            "manifest_version": "0.2",
            "metadata": {
                "protocol": protocol,
                "train_seed": 2042,
                "state_step": 26,
                "stream_id": 0,
                "stream_seed": 123,
                "batch_seed": 456,
                "rho": 0.25,
                "epsilon": 0.01,
            },
            "batches": batches,
            "sha256": manifest_sha256(batches),
        }

    def test_accepts_wellformed(self):
        payload = self._payload()
        out = validate_g2_e2_manifest(
            payload, expected_seed=2042, expected_step=26, expected_stream=0,
            expected_content_sha=payload["sha256"], accumulation_steps=2,
        )
        self.assertEqual(out["content_sha256"], payload["sha256"])
        self.assertEqual(len(out["batches"]), 4)

    def test_rejects_g1a_protocol_tag(self):
        payload = self._payload(protocol="FR_G1A_26M_E2_PAIRED_STREAM")
        with self.assertRaises(ValueError):
            validate_g2_e2_manifest(
                payload, expected_seed=2042, expected_step=26, expected_stream=0,
                expected_content_sha=payload["sha256"], accumulation_steps=2,
            )

    def test_rejects_bad_sha_and_identity(self):
        payload = self._payload()
        with self.assertRaises(ValueError):
            validate_g2_e2_manifest(
                payload, expected_seed=2042, expected_step=26, expected_stream=0,
                expected_content_sha="0" * 64, accumulation_steps=2,
            )
        with self.assertRaises(ValueError):
            validate_g2_e2_manifest(
                payload, expected_seed=2043, expected_step=26, expected_stream=0,
                expected_content_sha=payload["sha256"], accumulation_steps=2,
            )


class TestShingleHelpers(unittest.TestCase):
    def test_jaccard_bounds(self):
        a = shingle_digest("hello world " * 50)
        b = shingle_digest("hello world " * 50)
        self.assertEqual(jaccard(a, b), 1.0)
        c = shingle_digest("完全不相干的另一段中文文本内容" * 20)
        self.assertGreaterEqual(jaccard(a, c), 0.0)
        self.assertLessEqual(jaccard(a, c), 1.0)
        self.assertLess(jaccard(a, c), 0.01)

    def test_short_text(self):
        self.assertEqual(len(shingle_digest("短")), 1)


class TestConfigGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g1a = json.loads(G1A_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.g2 = json.loads(G2_CONFIG_PATH.read_text(encoding="utf-8"))

    # G1a scale-experiment provenance annotations; not scientific fields.
    G1A_MODEL_ANNOTATIONS = {
        "anchor_unique_params", "scale_ratio", "selection_reason", "param_count_note",
    }

    def test_model_block_identical_to_g1a(self):
        g1a_scientific = {k: v for k, v in self.g1a["model"].items()
                          if k not in self.G1A_MODEL_ANNOTATIONS}
        self.assertEqual(self.g2["model"], g1a_scientific)

    def test_training_core_identical_to_g1a(self):
        keys = ["batch_size", "accumulation_steps", "global_batch_sequences",
                "tokens_per_optimizer_step", "learning_rate", "optimizer",
                "grad_clip", "dtype", "grad_scaler",
                "planned_total_optimizer_steps", "total_micro_steps",
                "train_batch_seed_offset", "states", "train_seeds"]
        for key in keys:
            self.assertEqual(self.g2["training"][key], self.g1a["training"][key],
                             msg=f"training.{key} drifted from frozen G1a")

    def test_stream_seed_bases_match_g1a_module(self):
        self.assertIn(str(E1_BASE_SEED), json.dumps(self.g2))
        self.assertIn(str(E2_BASE_SEED), json.dumps(self.g2))

    def test_d0_reference_hash_matches_guard(self):
        self.assertEqual(self.g2["d0_reference"]["sha256"], D0_SHA256)

    def test_horizons_and_seeds(self):
        self.assertEqual(tuple(self.g2["response_protocol_inheritance"]["horizons_official"]),
                         tuple(HORIZONS_OFFICIAL))
        self.assertEqual(tuple(self.g2["training"]["train_seeds"]), tuple(TRAIN_SEEDS))


class TestD0HashGuard(unittest.TestCase):
    def test_d0_file_hash_matches_frozen_manifest(self):
        import hashlib
        digest = hashlib.sha256(D0_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, D0_SHA256)


if __name__ == "__main__":
    unittest.main()
