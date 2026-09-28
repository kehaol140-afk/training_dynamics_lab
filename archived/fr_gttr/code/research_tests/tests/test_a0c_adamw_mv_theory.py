import unittest
from collections import OrderedDict
import hashlib
import json

import torch

from fr.a0c_adamw_mv_theory import (
    aggregate_stream_results,
    assemble_summary,
    build_metric_breakdowns,
    compute_mu_nu,
    predict_chi,
    regression_metrics,
    render_report,
    validate_e2_manifest,
)


def _checkpoint():
    return {
        "model": OrderedDict(
            [
                ("w", torch.zeros(2)),
                ("b", torch.zeros(1)),
                ("buffer", torch.zeros(1)),
            ]
        ),
        "optimizer": {
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 0.1,
                    "betas": (0.9, 0.999),
                    "eps": 0.0,
                    "weight_decay": 0.01,
                }
            ],
            "state": {
                0: {
                    "exp_avg": torch.tensor([1.0, 123.0]),
                    "exp_avg_sq": torch.tensor([4.0, 0.0]),
                },
                1: {
                    "exp_avg": torch.tensor([2.0]),
                    "exp_avg_sq": torch.tensor([9.0]),
                },
            },
        },
    }


class TestAdamWMVTheoryArithmetic(unittest.TestCase):
    def test_mu_nu_align_by_name_and_skip_exact_zero_q_coordinates(self):
        q = OrderedDict(
            [("b", torch.tensor([3.0])), ("w", torch.tensor([2.0, 0.0]))]
        )
        gradient = OrderedDict(
            [("w", torch.tensor([5.0, 999.0])), ("b", torch.tensor([7.0]))]
        )

        result = compute_mu_nu(q, gradient, _checkpoint())

        self.assertAlmostEqual(result["mu_q"], 5.0, places=12)
        self.assertAlmostEqual(result["nu_q"], 2.5 + 126.0 / 27.0, places=12)
        self.assertEqual(result["parameter_names"], ["w", "b"])
        self.assertEqual(result["beta1"], 0.9)
        self.assertEqual(result["beta2"], 0.999)
        self.assertEqual(result["zero_q_coordinate_count"], 1)

    def test_prediction_includes_requested_perturbation_epsilon_once(self):
        result = predict_chi(
            eta=0.1,
            beta1=0.9,
            beta2=0.999,
            horizon=2,
            mu_q=5.0,
            nu_q=2.5 + 126.0 / 27.0,
            perturbation_epsilon=0.25,
        )

        expected_m = -0.1 * 0.25 * 5.0 * (1.0 - 0.9**2)
        expected_mv = expected_m + 0.1 * 0.25 * (2.5 + 126.0 / 27.0) * (
            1.0 - 0.999**2
        )
        self.assertAlmostEqual(result["chi_m"], expected_m, places=14)
        self.assertAlmostEqual(result["chi_mv"], expected_mv, places=14)

    def test_metrics_fit_observed_as_function_of_prediction(self):
        result = regression_metrics([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

        self.assertAlmostEqual(result["corr"], 1.0, places=12)
        self.assertAlmostEqual(result["R2"], -6.0, places=12)
        self.assertAlmostEqual(result["MAE"], 2.0, places=12)
        self.assertAlmostEqual(result["slope"], 0.5, places=12)
        self.assertAlmostEqual(result["intercept"], 0.0, places=12)
        self.assertEqual(result["regression_direction"], "observed=intercept+slope*predicted")


class TestAdamWMVTheoryAggregation(unittest.TestCase):
    def test_manifest_requires_exact_identity_and_content_hash(self):
        batches = [[1, 2], [3, 4]]
        encoded = json.dumps(batches, separators=(",", ":"), ensure_ascii=False).encode()
        payload = {
            "manifest_version": "0.2",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "metadata": {
                "protocol": "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM",
                "train_seed": 2030,
                "state_step": 46,
                "stream_id": 3,
                "stream_seed": 99,
                "rho": 0.25,
                "epsilon": 0.0125,
            },
            "batches": batches,
        }

        result = validate_e2_manifest(
            payload, expected_seed=2030, expected_state_step=46, expected_stream_id=3
        )
        self.assertEqual(result["batches"], batches)
        self.assertEqual(result["stream_seed"], 99)
        with self.assertRaisesRegex(ValueError, "state_step"):
            validate_e2_manifest(
                payload, expected_seed=2030, expected_state_step=52, expected_stream_id=3
            )

    def test_stream_average_is_arithmetic_and_gradient_is_float32(self):
        streams = [
            {
                "mu_q": 5.0,
                "nu_q": 1.0,
                "eta": 0.01,
                "gradient": {"w": torch.tensor([1.0, 3.0])},
            },
            {
                "mu_q": 5.0,
                "nu_q": 3.0,
                "eta": 0.01,
                "gradient": {"w": torch.tensor([3.0, 5.0])},
            },
        ]

        result = aggregate_stream_results(streams)

        self.assertEqual(result["mu_q"], 5.0)
        self.assertEqual(result["nu_q"], 2.0)
        self.assertEqual(result["eta"], 0.01)
        self.assertTrue(torch.equal(result["mean_gradient"]["w"], torch.tensor([2.0, 4.0])))
        self.assertEqual(result["mean_gradient"]["w"].dtype, torch.float32)

    def test_metric_breakdowns_have_both_requested_grains(self):
        rows = [
            {"seed": 1, "horizon": 1, "chi": 1.0, "chi_m": 2.0, "chi_mv": 1.0},
            {"seed": 1, "horizon": 2, "chi": 2.0, "chi_m": 4.0, "chi_mv": 2.0},
            {"seed": 2, "horizon": 1, "chi": 3.0, "chi_m": 6.0, "chi_mv": 3.0},
            {"seed": 2, "horizon": 2, "chi": 4.0, "chi_m": 8.0, "chi_mv": 4.0},
        ]

        result = build_metric_breakdowns(rows)

        self.assertEqual(result["chi_m"]["overall"]["R2"], -5.0)
        self.assertEqual(result["chi_mv"]["overall"]["R2"], 1.0)
        self.assertEqual(sorted(result["chi_m"]["per_seed"]), ["1", "2"])
        self.assertEqual(sorted(result["chi_mv"]["per_horizon"]), ["1", "2"])

    def test_report_discloses_epsilon_dimension_mismatch(self):
        summary = {
            "metrics": {
                "chi_m": {
                    "overall": {
                        "n": 4, "R2": -1.0, "corr": 0.8, "MAE": 0.2,
                        "slope": 2.0, "intercept": -0.1,
                    },
                    "per_seed": {},
                    "per_horizon": {},
                },
                "chi_mv": {
                    "overall": {
                        "n": 4, "R2": -0.5, "corr": 0.9, "MAE": 0.15,
                        "slope": 1.5, "intercept": -0.05,
                    },
                    "per_seed": {},
                    "per_horizon": {},
                },
            },
            "comparison": {
                "R2_improvement": 0.5,
                "remaining_unexplained_fraction": 1.5,
                "remaining_residual_MAE": 0.15,
            },
            "matched_state_count": 56,
            "matched_stream_count": 448,
        }

        report = render_report(summary)

        self.assertIn("已经除以", report)
        self.assertIn("rho*sigma_q", report)
        self.assertIn("Does adding v-path improve", report)
        self.assertIn("0.5", report)

    def test_summary_reports_direct_r2_gain_and_remaining_fraction(self):
        predictions = [
            {"seed": 1, "state_step": 46, "horizon": 1, "chi": 1.0, "chi_m": 2.0, "chi_mv": 1.0},
            {"seed": 1, "state_step": 46, "horizon": 2, "chi": 2.0, "chi_m": 4.0, "chi_mv": 2.0},
            {"seed": 2, "state_step": 46, "horizon": 1, "chi": 3.0, "chi_m": 6.0, "chi_mv": 3.0},
            {"seed": 2, "state_step": 46, "horizon": 2, "chi": 4.0, "chi_m": 8.0, "chi_mv": 4.0},
        ]

        summary = assemble_summary(
            predictions,
            state_results=[
                {"seed": 1, "state_step": 46, "nu_q": 1.0},
                {"seed": 2, "state_step": 46, "nu_q": -10.0},
            ],
            matched_stream_count=16,
            source_paths={"a0c": "/tmp/a0c.csv"},
            replay_metadata={"device": "cpu"},
        )

        self.assertEqual(summary["matched_state_count"], 2)
        self.assertEqual(summary["matched_stream_count"], 16)
        self.assertEqual(summary["comparison"]["R2_improvement"], 6.0)
        self.assertEqual(summary["comparison"]["remaining_unexplained_fraction"], 0.0)
        self.assertEqual(summary["fit_parameters"], [])
        self.assertIn("rho*sigma_q", summary["formula"]["chi_mv"])
        self.assertEqual(summary["sensitivity"]["max_abs_nu_state"]["seed"], 2)
        self.assertEqual(
            summary["sensitivity"]["excluding_max_abs_nu_state"]["R2_improvement"],
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
