import unittest
from collections import OrderedDict

import torch

from fr.a0c_adamw_mv_consistent import (
    assemble_consistent_summary,
    compute_consistent_state_features,
    predict_unit_response,
    render_consistent_report,
)


def checkpoint_payload():
    return {
        "model": OrderedDict([("w", torch.zeros(2)), ("b", torch.zeros(1))]),
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
                    "exp_avg": torch.tensor([1.0, 2.0]),
                    "exp_avg_sq": torch.tensor([4.0, 16.0]),
                },
                1: {
                    "exp_avg": torch.tensor([3.0]),
                    "exp_avg_sq": torch.tensor([9.0]),
                },
            },
        },
        "metadata": {"train_seed": 2030, "state_optimizer_step": 46},
    }


def gradient_record():
    return {
        "seed": 2030,
        "state_step": 46,
        "aggregation": "ARITHMETIC_MEAN_OF_8_REAL_E2_PRECLIP_UNSCALED_GRADIENTS",
        "gradient": {
            "w": torch.tensor([0.5, -1.0]),
            "b": torch.tensor([2.0]),
        },
        "stream_audit": [{"first_step_lr": 0.08} for _ in range(8)],
    }


class TestStrictStateAlignment(unittest.TestCase):
    def test_state_specific_q_controls_both_projections(self):
        q1 = OrderedDict(
            [("w", torch.tensor([2.0, 4.0])), ("b", torch.tensor([3.0]))]
        )
        q2 = OrderedDict(
            [("w", torch.tensor([1.0, 1.0])), ("b", torch.tensor([1.0]))]
        )
        metadata = {"train_seed": 2030, "state_step": 46}

        first = compute_consistent_state_features(
            q_by_name=q1,
            q_metadata=metadata,
            q_sha256="q-one",
            expected_q_sha256="q-one",
            gradient_record=gradient_record(),
            checkpoint=checkpoint_payload(),
            seed=2030,
            state_step=46,
        )
        second = compute_consistent_state_features(
            q_by_name=q2,
            q_metadata=metadata,
            q_sha256="q-two",
            expected_q_sha256="q-two",
            gradient_record=gradient_record(),
            checkpoint=checkpoint_payload(),
            seed=2030,
            state_step=46,
        )

        self.assertAlmostEqual(first["mu_q"], 9.0, places=12)
        self.assertAlmostEqual(first["nu_q"], 1.75, places=12)
        self.assertEqual(first["eta"], 0.08)
        self.assertNotEqual(first["mu_q"], second["mu_q"])
        self.assertNotEqual(first["nu_q"], second["nu_q"])

    def test_wrong_q_state_or_hash_is_rejected(self):
        q = {"w": torch.ones(2), "b": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "q metadata state_step"):
            compute_consistent_state_features(
                q_by_name=q,
                q_metadata={"train_seed": 2030, "state_step": 52},
                q_sha256="actual",
                expected_q_sha256="actual",
                gradient_record=gradient_record(),
                checkpoint=checkpoint_payload(),
                seed=2030,
                state_step=46,
            )
        with self.assertRaisesRegex(ValueError, "q direction sha256"):
            compute_consistent_state_features(
                q_by_name=q,
                q_metadata={"train_seed": 2030, "state_step": 46},
                q_sha256="actual",
                expected_q_sha256="different",
                gradient_record=gradient_record(),
                checkpoint=checkpoint_payload(),
                seed=2030,
                state_step=46,
            )


class TestStrictUnitResponse(unittest.TestCase):
    def test_prediction_has_no_perturbation_epsilon_factor(self):
        result = predict_unit_response(
            eta=0.1,
            beta1=0.9,
            beta2=0.999,
            horizon=2,
            mu_q=2.0,
            nu_q=3.0,
        )

        self.assertAlmostEqual(result["chi_m"], -0.038, places=15)
        self.assertAlmostEqual(result["chi_v"], 0.0005997, places=15)
        self.assertAlmostEqual(result["chi_mv"], -0.0374003, places=15)

    def test_summary_and_report_answer_all_three_questions(self):
        rows = [
            {"seed": 1, "state_step": 46, "horizon": 1, "chi": 1.0, "chi_m": 2.0, "chi_mv": 1.0},
            {"seed": 1, "state_step": 46, "horizon": 2, "chi": 2.0, "chi_m": 4.0, "chi_mv": 2.0},
            {"seed": 2, "state_step": 46, "horizon": 1, "chi": 3.0, "chi_m": 6.0, "chi_mv": 3.0},
            {"seed": 2, "state_step": 46, "horizon": 2, "chi": 4.0, "chi_m": 8.0, "chi_mv": 4.0},
        ]
        summary = assemble_consistent_summary(
            rows,
            state_results=[{"seed": 1}, {"seed": 2}],
            source_paths={"gradient": "/tmp/gradient_g.pt"},
            global_q_unit_rows=[
                {"seed": 1, "state_step": 46, "horizon": 1, "chi": 1.0, "chi_m": 1.0, "chi_mv": 2.0},
                {"seed": 1, "state_step": 46, "horizon": 2, "chi": 2.0, "chi_m": 2.0, "chi_mv": 4.0},
                {"seed": 2, "state_step": 46, "horizon": 1, "chi": 3.0, "chi_m": 3.0, "chi_mv": 6.0},
                {"seed": 2, "state_step": 46, "horizon": 2, "chi": 4.0, "chi_m": 4.0, "chi_mv": 8.0},
            ],
        )
        report = render_consistent_report(summary)

        self.assertEqual(summary["fit_parameters"], [])
        self.assertNotIn("rho*sigma_q", summary["formula"]["chi_mv"])
        self.assertEqual(summary["comparison"]["R2_improvement"], 6.0)
        self.assertTrue(summary["state_specific_q_changes_v_path_conclusion"])
        self.assertIn("Does v-path improve after fixing scale", report)
        self.assertIn("Does state-specific q change the conclusion", report)
        self.assertIn("Is AdamW m+v sufficient", report)
        self.assertIn("State-specific q control", report)
        self.assertIn("SSE/SST", report)


if __name__ == "__main__":
    unittest.main()
