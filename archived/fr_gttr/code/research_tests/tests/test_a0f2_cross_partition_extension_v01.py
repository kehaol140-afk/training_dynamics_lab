from __future__ import annotations

import unittest

# Importing the runner first registers research/src on sys.path (same
# convention as the other research test modules).
import research.scripts.a0f2_cross_partition_extension_v01  # noqa: F401

from fr.a0c_adamw_response_propagation_v01 import relative_error
from fr.a0f_cross_state_generalization_v01 import (
    HORIZONS,
    SEEDS,
    STEPS,
    error_distributions,
    paired_b_c_comparison,
    prediction_metrics,
    sample_states,
)
from fr.a0f2_cross_partition_extension_v01 import (
    assemble_summary,
    full_states,
    render_report,
    validate_against_a0f,
)


def _row(seed, step, horizon, chi_fd, pred_a, pred_b, pred_c):
    return {
        "state": f"seed_{seed}/state_step_{step:03d}",
        "seed": seed,
        "step": step,
        "H": horizon,
        "chi_fd": chi_fd,
        "chi_A_momentum": pred_a,
        "chi_optimizer_only": pred_b,
        "chi_full_tangent": pred_c,
        "relative_error_A": relative_error(chi_fd, pred_a),
        "relative_error_optimizer_only": relative_error(chi_fd, pred_b),
        "relative_error_full_tangent": relative_error(chi_fd, pred_c),
        "absolute_error_A": abs(pred_a - chi_fd),
        "absolute_error_optimizer_only": abs(pred_b - chi_fd),
        "absolute_error_full_tangent": abs(pred_c - chi_fd),
    }


def _census_rows():
    states = full_states()
    rows = []
    for index, (seed, step) in enumerate(states):
        for horizon in HORIZONS:
            chi_fd = 0.1 * horizon * (1.0 + 0.01 * index)
            # C better than B better than A; B == C at H=1 (zero-control).
            pred_b = chi_fd * (1.0 + 0.2 * 0.1)
            pred_c = pred_b if horizon == 1 else chi_fd * (1.0 + 0.05 * 0.1)
            rows.append(
                _row(
                    seed, step, horizon, chi_fd,
                    pred_a=chi_fd * (1.0 + 0.5 * 0.1),
                    pred_b=pred_b,
                    pred_c=pred_c,
                )
            )
    return states, rows


def _stream_rows(states):
    return [
        {
            "seed": seed,
            "step": step,
            "stream_id": sid,
            "chi_optimizer_only": {str(h): 0.01 * h * (1 + 0.001 * sid) for h in HORIZONS},
            "chi_full_tangent": {str(h): 0.01 * h * (1 + 0.001 * sid) for h in HORIZONS},
        }
        for seed, step in states for sid in range(8)
    ]


class TestFullStates(unittest.TestCase):
    def test_census_covers_every_seed_and_step(self):
        states = full_states()
        self.assertEqual(len(states), len(SEEDS) * len(STEPS))
        self.assertEqual(len(set(states)), len(states))
        self.assertEqual({seed for seed, _ in states}, set(SEEDS))
        self.assertEqual({step for _, step in states}, set(STEPS))

    def test_census_is_a_superset_of_the_a0f_sample(self):
        states = full_states()
        sample = sample_states()
        self.assertTrue(set(sample).issubset(set(states)))
        self.assertLess(len(sample), len(states))


class TestValidateAgainstA0F(unittest.TestCase):
    def _a0f_like_summary(self, states, rows, streams):
        import copy

        row_keys = {(seed, step) for seed, step in states}
        return {
            "results": copy.deepcopy(
                [row for row in rows if (row["seed"], row["step"]) in row_keys]
            ),
            "stream_results": copy.deepcopy(
                [
                    stream for stream in streams
                    if (stream["seed"], stream["step"]) in row_keys
                ]
            ),
        }

    def test_shared_subset_reproduced_bit_exactly(self):
        states, rows = _census_rows()
        streams = _stream_rows(states)
        a0f_summary = self._a0f_like_summary(sample_states(), rows, streams)
        check = validate_against_a0f(
            result_rows=rows, stream_results=streams, a0f_summary=a0f_summary,
        )
        self.assertTrue(check["pass"])
        self.assertEqual(check["rows_checked"], len(sample_states()) * len(HORIZONS))
        self.assertEqual(check["streams_checked"], len(sample_states()) * 8)
        for gap in check["max_abs_gap_by_field"].values():
            self.assertEqual(gap, 0.0)

    def test_mismatch_raises(self):
        states, rows = _census_rows()
        streams = _stream_rows(states)
        a0f_summary = self._a0f_like_summary(sample_states(), rows, streams)
        a0f_summary["results"][0]["chi_full_tangent"] += 1.0
        with self.assertRaises(ValueError):
            validate_against_a0f(
                result_rows=rows, stream_results=streams, a0f_summary=a0f_summary,
            )

    def test_stream_mismatch_raises(self):
        states, rows = _census_rows()
        streams = _stream_rows(states)
        a0f_summary = self._a0f_like_summary(sample_states(), rows, streams)
        a0f_summary["stream_results"][0]["chi_optimizer_only"]["8"] += 1.0
        with self.assertRaises(ValueError):
            validate_against_a0f(
                result_rows=rows, stream_results=streams, a0f_summary=a0f_summary,
            )

    def test_no_shared_states_raises(self):
        states, rows = _census_rows()
        streams = _stream_rows(states)
        with self.assertRaises(ValueError):
            validate_against_a0f(
                result_rows=rows, stream_results=streams,
                a0f_summary={"results": [], "stream_results": []},
            )


class TestAssembleAndRender(unittest.TestCase):
    def _summary(self):
        states, rows = _census_rows()
        streams = _stream_rows(states)
        a0f_summary = {
            "results": [row for row in rows if (row["seed"], row["step"]) in set(sample_states())],
            "stream_results": [
                stream for stream in streams
                if (stream["seed"], stream["step"]) in set(sample_states())
            ],
        }
        crosscheck = validate_against_a0f(
            result_rows=rows, stream_results=streams, a0f_summary=a0f_summary,
        )
        return assemble_summary(
            result_rows=rows,
            stream_results=streams,
            metrics=prediction_metrics(rows),
            distributions=error_distributions(rows),
            paired=paired_b_c_comparison(rows),
            a0f_crosscheck=crosscheck,
            source_paths={},
            input_hashes={},
            runtime={
                "baseline_replay": "PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS",
                "stream_count": len(streams),
                "max_hvp_math_gradient_relative_gap_vs_native": 0.0,
                "clip_active_steps_h_gt_1": 0,
                "total_steps_h_gt_1": 1,
                "device": "cuda:0",
                "amp_dtype": "bfloat16",
                "torch_version": "2.11.0",
                "elapsed_seconds": 1.0,
            },
            conclusions={
                "c_vs_b_stability": "YES",
                "c_vs_b_explanation": "",
                "a0f_consistency": "BIT_EXACT_REPRODUCTION_OF_ALL_SHARED_STATES",
                "a0f_consistency_explanation": "",
                "stage_consistency": "STEPS_WITH_POSITIVE_MEAN_IMPROVEMENT_7_OF_7",
                "stage_explanation": "",
                "model_ordering": "C_BEATS_B_BEATS_A",
                "ordering_explanation": "",
                "recommended_next_step": "next",
                "further_question": "question",
            },
        )

    def test_assemble_grain(self):
        summary = self._summary()
        self.assertEqual(summary["protocol"], "FR_A0F2_CROSS_PARTITION_EXTENSION_V01")
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(
            summary["result_row_count"], len(SEEDS) * len(STEPS) * len(HORIZONS)
        )
        self.assertEqual(summary["stream_count"], len(SEEDS) * len(STEPS) * 8)
        self.assertEqual(
            summary["scope"]["sampling"]["scheme"], "full_census_of_v07_bank"
        )
        broken = summary["results"][:-1]
        with self.assertRaises(ValueError):
            assemble_summary(
                result_rows=broken,
                stream_results=summary["stream_results"],
                metrics=summary["metrics"],
                distributions=summary["error_distributions"],
                paired=summary["paired_b_c"],
                a0f_crosscheck=summary["a0f_crosscheck"],
                source_paths={},
                input_hashes={},
                runtime={},
                conclusions={},
            )

    def test_render_report_tables(self):
        report = render_report(self._summary())
        self.assertIn("# FR A0F2 Cross-Partition Extension v01 (Full v07 Bank)", report)
        self.assertIn("| horizon | model | R2 | corr | MAE | RMSE | mean relative error |", report)
        self.assertIn("Paired B-vs-C comparison per state", report)
        self.assertIn("| seed | steps |", report)
        self.assertIn("v06 replication bank", report)
        self.assertIn("Consistency with the frozen FR_A0F artifact", report)
        self.assertIn("H=1 has B == C by construction", report)


if __name__ == "__main__":
    unittest.main()
