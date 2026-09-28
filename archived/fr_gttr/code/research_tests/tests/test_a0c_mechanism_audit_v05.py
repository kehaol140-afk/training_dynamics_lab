from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from a0c_mechanism_audit_v05 import (
    cluster_bootstrap_delta_r2,
    correlation_audit,
    resample_seed_clusters,
    run_audit,
)


class A0CMechanismAuditV05Tests(unittest.TestCase):
    def test_resample_seed_clusters_copies_whole_blocks_in_draw_order(self):
        source = pd.DataFrame(
            {
                "seed": [10, 10, 20, 20],
                "marker": [101, 102, 201, 202],
            }
        )

        sampled = resample_seed_clusters(source, [20, 10, 20])

        self.assertEqual(sampled["marker"].tolist(), [201, 202, 101, 102, 201, 202])
        self.assertEqual(sampled["seed"].tolist(), [20, 20, 10, 10, 20, 20])

    def test_cluster_bootstrap_returns_seed_level_percentile_summary(self):
        rows = []
        for seed in (10, 20, 30):
            for B, alpha in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rows.append(
                    {
                        "seed": seed,
                        "state_step": len(rows),
                        "horizon": 1,
                        "B": B,
                        "chi": alpha,
                        "alpha_direct": alpha,
                    }
                )
        data = pd.DataFrame(rows)

        deltas, summary = cluster_bootstrap_delta_r2(
            data, reps=8, random_seed=7
        )

        np.testing.assert_allclose(deltas, np.ones(8), atol=1e-12)
        self.assertAlmostEqual(summary["bootstrap_mean_delta_R2"], 1.0)
        np.testing.assert_allclose(summary["ci95"], [1.0, 1.0], atol=1e-12)
        self.assertEqual(summary["sampling_unit"], "seed")
        self.assertEqual(summary["clusters_per_replicate"], 3)
        self.assertEqual(summary["valid_reps"], 8)

    def test_correlation_audit_includes_overall_seed_and_horizon_levels(self):
        rows = []
        alpha = 1
        for seed in (10, 20):
            for horizon in (1, 2):
                for state_step in (1, 2):
                    rows.append(
                        {
                            "seed": seed,
                            "state_step": state_step,
                            "horizon": horizon,
                            "B": -alpha,
                            "chi": 2 * alpha,
                            "alpha_direct": alpha,
                        }
                    )
                    alpha += 1
        data = pd.DataFrame(rows)

        result = correlation_audit(data, horizons=(1, 2))

        self.assertEqual(
            result[["level", "group", "n"]].to_dict(orient="records"),
            [
                {"level": "overall", "group": "all", "n": 8},
                {"level": "seed", "group": "10", "n": 4},
                {"level": "seed", "group": "20", "n": 4},
                {"level": "horizon", "group": "1", "n": 4},
                {"level": "horizon", "group": "2", "n": 4},
            ],
        )
        np.testing.assert_allclose(result["corr_alpha_chi"], 1.0, atol=1e-12)
        np.testing.assert_allclose(result["corr_alpha_B"], -1.0, atol=1e-12)

    def test_run_audit_writes_models_cluster_bootstrap_correlations_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = (
                root
                / "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv"
            )
            input_path.parent.mkdir(parents=True)

            rows = []
            for seed_index, seed in enumerate((10, 20, 30)):
                for state_step in (1, 2, 3):
                    for horizon in (1, 2, 4, 8):
                        B = state_step - 2 + 0.03 * horizon + 0.05 * seed_index
                        alpha = (
                            (state_step - 2) * (0.2 + 0.01 * horizon)
                            + 0.04 * horizon
                            - 0.02 * seed_index
                        )
                        chi = (
                            0.35 * B
                            + 0.75 * alpha
                            + 0.001 * state_step * horizon
                            + 0.002 * seed_index * state_step
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "state_step": state_step,
                                "horizon": horizon,
                                "B": B,
                                "chi": chi,
                                "alpha_direct": alpha,
                            }
                        )
            pd.DataFrame(rows).to_csv(input_path, index=False)

            outputs = run_audit(root, bootstrap_reps=8, bootstrap_seed=123)

            output_dir = root / "outputs/a0_fr_audit/v05/mechanism"
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "A0C_MECHANISM_REPORT.md",
                    "cluster_bootstrap_delta_r2.json",
                    "correlation_audit.csv",
                    "model_summary.json",
                ],
            )
            self.assertEqual(outputs["output_dir"], output_dir)

            model_summary = json.loads(
                (output_dir / "model_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model_summary["models"]["base"]["formula"], "chi ~ 1 + B")
            self.assertEqual(
                model_summary["models"]["control"]["formula"],
                "chi ~ 1 + B + alpha_direct",
            )

            bootstrap = json.loads(
                (output_dir / "cluster_bootstrap_delta_r2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(bootstrap["sampling_unit"], "seed")
            self.assertEqual(bootstrap["bootstrap_reps"], 8)

            correlation = pd.read_csv(
                output_dir / "correlation_audit.csv", dtype={"group": str}
            )
            self.assertEqual(set(correlation["level"]), {"overall", "seed", "horizon"})

            report = (output_dir / "A0C_MECHANISM_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("chi ~ 1 + B", report)
            self.assertIn("chi ~ 1 + B + alpha_direct", report)
            self.assertIn("seed-cluster bootstrap", report)
            self.assertIn("8 次 seed-cluster bootstrap", report)
            self.assertIn("RNG seed=123", report)
            self.assertIn("只有 3 个独立 cluster", report)
            self.assertIn("Correlation audit", report)
            self.assertIn("covariance geometry", report)


if __name__ == "__main__":
    unittest.main()
