from __future__ import annotations

import math
import json
import unittest
from pathlib import Path

from research.scripts.a0d_epsilon_convergence_v01 import _paired_last_lr

from fr.a0d_epsilon_convergence_v01 import (
    EPSILON_LEVELS,
    HORIZONS,
    SELECTED_STATES,
    assemble_summary,
    central_response,
    epsilon_convergence,
    epsilon8_candidates,
    h8_conclusion,
    prediction_metrics,
    render_report,
)


class TestEpsilonConvergenceArithmetic(unittest.TestCase):
    def test_epsilon8_candidates_include_contracting_but_nonmonotonic_cells(self):
        """Catches equating FD self-contraction with convergence to the tangent."""
        rows = [
            {"state": "a", "H": 8, "fd_changes_contract": True, "relative_error_monotonic": False},
            {"state": "b", "H": 4, "fd_changes_contract": False, "relative_error_monotonic": False},
            {"state": "c", "H": 2, "fd_changes_contract": True, "relative_error_monotonic": True},
        ]
        self.assertEqual(
            epsilon8_candidates(rows),
            [
                {"state": "a", "H": 8, "reason": "TANGENT_ERROR_NONMONOTONIC"},
                {"state": "b", "H": 4, "reason": "FD_NONCONTRACTING_AND_TANGENT_ERROR_NONMONOTONIC"},
            ],
        )

    def test_h8_advantage_distinguishes_endpoint_gain_from_nonmonotonic_path(self):
        """Catches treating a nonmonotonic midpoint as no epsilon/4 improvement."""
        convergence = [
            {
                "relative_error_descending": [0.02, 0.03, 0.01],
                "relative_error_monotonic": False,
            }
            for _ in range(3)
        ]
        result = h8_conclusion(
            convergence,
            full_metrics={"mean_relative_error": 0.02},
            quarter_metrics={"mean_relative_error": 0.01},
        )
        self.assertEqual(result["endpoint_improvement_count"], 3)
        self.assertEqual(result["monotonic_count"], 0)
        self.assertEqual(
            result["answer"],
            "YES_AT_EPSILON_QUARTER_ALL_3_WITH_NONMONOTONIC_INTERMEDIATE",
        )

    def test_normal_theta_branches_compare_their_available_last_lr(self):
        """Catches reading the capture-only first_step_lr field from normal branches."""
        plus = {"first_step_lr": None, "last_lr": 0.001}
        minus = {"first_step_lr": None, "last_lr": 0.001}
        self.assertEqual(_paired_last_lr(plus, minus), 0.001)

    def test_central_response_divides_by_the_actual_level_once(self):
        """Catches reusing registered epsilon for the half/quarter responses."""
        self.assertAlmostEqual(
            central_response(x_plus=1.25, x_minus=0.75, epsilon=0.125),
            2.0,
        )

    def test_convergence_uses_successive_fd_changes_and_tangent_errors(self):
        """Catches reversed epsilon ordering or comparing predictions across levels."""
        rows = [
            {"epsilon_level": "epsilon", "epsilon": 1.0, "chi_fd": 1.4, "chi_tangent": 1.0},
            {"epsilon_level": "epsilon/2", "epsilon": 0.5, "chi_fd": 1.1, "chi_tangent": 1.0},
            {"epsilon_level": "epsilon/4", "epsilon": 0.25, "chi_fd": 1.025, "chi_tangent": 1.0},
        ]
        result = epsilon_convergence(rows)
        self.assertAlmostEqual(result["fd_change_full_to_half"], 0.3)
        self.assertAlmostEqual(result["fd_change_half_to_quarter"], 0.075)
        self.assertAlmostEqual(result["fd_change_contraction_ratio"], 0.25)
        self.assertAlmostEqual(result["observed_central_difference_order"], 2.0)
        for actual, expected in zip(
            result["absolute_error_descending"], [0.4, 0.1, 0.025]
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(result["fd_changes_contract"])
        self.assertTrue(result["absolute_error_monotonic"])
        self.assertTrue(result["relative_error_monotonic"])

    def test_metrics_are_broken_down_by_epsilon_and_horizon(self):
        """Catches reporting only pooled metrics or mixing epsilon levels."""
        rows = []
        observed = {
            "epsilon": [-1.0, -2.0, -3.0],
            "epsilon/2": [-1.1, -2.1, -3.1],
            "epsilon/4": [-1.2, -2.2, -3.2],
        }
        for level, values in observed.items():
            for index, value in enumerate(values):
                rows.append(
                    {
                        "state": f"state-{index}",
                        "H": (2, 4, 8)[index],
                        "epsilon_level": level,
                        "chi_fd": value,
                        "chi_tangent": value + 0.1,
                    }
                )
        metrics = prediction_metrics(rows)
        self.assertEqual(metrics["overall"]["n"], 9)
        self.assertEqual(set(metrics["per_epsilon"]), set(EPSILON_LEVELS))
        self.assertEqual(set(metrics["per_horizon"]), {"2", "4", "8"})
        self.assertEqual(metrics["per_epsilon"]["epsilon/4"]["n"], 3)
        self.assertEqual(metrics["per_horizon_epsilon"]["8"]["epsilon/2"]["n"], 1)
        self.assertTrue(math.isfinite(metrics["per_epsilon"]["epsilon"]["R2"]))


class TestEpsilonConvergenceArtifact(unittest.TestCase):
    def test_generated_artifact_records_transitive_replay_and_branch_state_provenance(self):
        """Catches unverifiable replay claims after transitive code changes."""
        path = Path(
            "outputs/a0_fr_audit/response_propagation_v02_epsilon/summary.json"
        )
        if not path.is_file():
            self.skipTest("generated A0D artifact is not present")
        summary = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "replay_full_state",
            "replay_stream_rng",
            "replay_native_schedule",
            "replay_perturbation",
            "replay_direction",
            "replay_hashing",
            "runtime_minimind_v03",
            "manifest_validator",
            "model_minimind",
            "pretrain_dataset",
            "trainer_utils",
            "tokenizer_json",
            "tokenizer_config",
        }
        self.assertTrue(required.issubset(summary["input_hashes"]))
        for stream in summary["stream_results"]:
            for cell in stream["epsilon_results"]:
                self.assertIn("plus_optimizer_hash_final", cell)
                self.assertIn("minus_optimizer_hash_final", cell)
                self.assertIn("first_preclip_grad_norm", cell)
                self.assertIn("first_clipped", cell)

    @staticmethod
    def _fixture_rows():
        rows = []
        streams = []
        for seed, step in SELECTED_STATES:
            state = f"seed_{seed}/state_step_{step:03d}"
            for horizon in HORIZONS:
                tangent = -0.01 * horizon
                for level_index, level in enumerate(EPSILON_LEVELS):
                    observed = tangent + 0.004 / (4**level_index)
                    rows.append(
                        {
                            "state": state,
                            "seed": seed,
                            "step": step,
                            "H": horizon,
                            "epsilon_level": level,
                            "epsilon": 0.25 / (2**level_index),
                            "chi_fd": observed,
                            "chi_tangent": tangent,
                            "absolute_error": abs(observed - tangent),
                            "relative_error": abs(observed - tangent) / abs(observed),
                        }
                    )
            for stream_id in range(8):
                streams.append(
                    {
                        "state": state,
                        "seed": seed,
                        "step": step,
                        "stream_id": stream_id,
                        "epsilon_results": [
                            {"epsilon_level": level} for level in EPSILON_LEVELS
                        ],
                    }
                )
        return rows, streams

    def test_summary_requires_complete_locked_grain_and_full_config(self):
        """Catches missing states, epsilon cells, or reproducibility metadata."""
        rows, streams = self._fixture_rows()
        convergence = []
        for seed, step in SELECTED_STATES:
            state = f"seed_{seed}/state_step_{step:03d}"
            for horizon in HORIZONS:
                convergence.append(
                    {
                        "state": state,
                        "seed": seed,
                        "step": step,
                        "H": horizon,
                        **epsilon_convergence(
                            [
                                row for row in rows
                                if row["state"] == state and row["H"] == horizon
                            ]
                        ),
                    }
                )
        summary = assemble_summary(
            result_rows=rows,
            stream_results=streams,
            convergence_results=convergence,
            metrics=prediction_metrics(rows),
            source_paths={"v01_summary": "/read-only/summary.json"},
            input_hashes={"v01_summary": "abc"},
            source_config={"dtype": "bfloat16", "accumulation_steps": 2},
            audit_config={
                "epsilon_multipliers": [1.0, 0.5, 0.25],
                "horizons": list(HORIZONS),
            },
            audit_config_sha256="config-hash",
            runtime={"device": "cuda:0", "input_integrity": "PASS"},
            conclusions={"linear_region": "YES", "tangent_error": "DECREASES", "h8": "RETAINS_ADVANTAGE"},
        )
        self.assertEqual(summary["protocol"], "FR_A0D_EPSILON_CONVERGENCE_V01")
        self.assertEqual(summary["result_row_count"], 27)
        self.assertEqual(summary["stream_count"], 24)
        self.assertEqual(summary["epsilon_stream_count"], 72)
        self.assertEqual(summary["source_config"]["dtype"], "bfloat16")
        self.assertEqual(summary["audit_config_sha256"], "config-hash")
        self.assertEqual(summary["fit_parameters"], [])
        report = render_report(summary)
        self.assertIn("epsilon/4", report)
        self.assertIn("H=8", report)


if __name__ == "__main__":
    unittest.main()
