from __future__ import annotations

import unittest

# Importing the runner first registers research/src on sys.path (same
# convention as the other research test modules).
import research.scripts.a0e_curvature_decomposition_v01  # noqa: F401

from fr.a0c_adamw_response_propagation_v01 import relative_error
from fr.a0e_curvature_decomposition_v01 import (
    HORIZONS_PROP,
    HORIZONS_TEST,
    PREDICTION_FIELDS,
    SELECTED_STATES,
    aggregate_gains,
    assemble_summary,
    prediction_metrics,
    render_report,
    row_gains,
    validate_against_a0c,
)


def _row(seed, step, horizon, chi_fd, pred_a, pred_b, pred_bh, pred_c, state=None):
    state = state or f"seed_{seed}/state_step_{step:03d}"
    return {
        "state": state,
        "seed": seed,
        "step": step,
        "H": horizon,
        "chi_fd": chi_fd,
        "chi_A_momentum": pred_a,
        "chi_optimizer_only": pred_b,
        "chi_BH_curvature": pred_bh,
        "chi_full_tangent": pred_c,
        "relative_error_A": relative_error(chi_fd, pred_a),
        "relative_error_optimizer_only": relative_error(chi_fd, pred_b),
        "relative_error_BH_curvature": relative_error(chi_fd, pred_bh),
        "relative_error_full_tangent": relative_error(chi_fd, pred_c),
        "absolute_error_A": abs(pred_a - chi_fd),
        "absolute_error_optimizer_only": abs(pred_b - chi_fd),
        "absolute_error_BH_curvature": abs(pred_bh - chi_fd),
        "absolute_error_full_tangent": abs(pred_c - chi_fd),
    }


def _full_rows():
    rows = []
    for seed, step in SELECTED_STATES:
        for horizon in HORIZONS_PROP:
            chi_fd = 0.1 * horizon
            rows.append(
                _row(
                    seed, step, horizon, chi_fd,
                    pred_a=chi_fd * (1.0 + 0.5 * 0.1),
                    pred_b=chi_fd * (1.0 + 0.2 * 0.1),
                    pred_bh=chi_fd * (1.0 + 0.1 * 0.1),
                    pred_c=chi_fd * (1.0 + 0.05 * 0.1),
                )
            )
    return rows


class TestRowGains(unittest.TestCase):
    def test_gains_match_definition_and_sum_to_total_reduction(self):
        row = _row(2037, 46, 2, 0.2, pred_a=0.30, pred_b=0.24, pred_bh=0.22, pred_c=0.21)
        gains = row_gains(row)
        # relative errors: A=0.5, B=0.2, BH=0.1, C=0.05
        self.assertAlmostEqual(gains["gain_optimizer_rel"], 0.3)
        self.assertAlmostEqual(gains["gain_H_rel"], 0.1)
        self.assertAlmostEqual(gains["clip_correction_rel"], 0.05)
        # gains telescope to the total A-to-C reduction
        total = (
            gains["gain_optimizer_rel"] + gains["gain_H_rel"]
            + gains["clip_correction_rel"]
        )
        self.assertAlmostEqual(total, 0.45)
        self.assertAlmostEqual(
            gains["gain_optimizer_abs"] + gains["gain_H_abs"]
            + gains["clip_correction_abs"],
            abs(0.30 - 0.2) - abs(0.21 - 0.2),
        )

    def test_negative_gain_is_reported_not_clipped(self):
        row = _row(2037, 46, 2, 0.2, pred_a=0.20, pred_b=0.26, pred_bh=0.30, pred_c=0.29)
        gains = row_gains(row)
        self.assertLess(gains["gain_optimizer_rel"], 0.0)
        self.assertLess(gains["gain_H_rel"], 0.0)


class TestAggregateGains(unittest.TestCase):
    def test_full_rows_produce_locked_grain(self):
        rows = _full_rows()
        gains = aggregate_gains(rows)
        self.assertEqual(len(gains["per_row"]), 9)
        self.assertEqual(set(gains["by_horizon"]), {"2", "4", "8"})
        for entry in gains["per_row"]:
            self.assertIn(entry["H"], HORIZONS_TEST)

    def test_wrong_grain_rejected(self):
        rows = _full_rows()[:-1]
        with self.assertRaises(ValueError):
            aggregate_gains(rows)


class TestPredictionMetricsReuse(unittest.TestCase):
    def test_metrics_match_frozen_a0c_semantics(self):
        rows = _full_rows()
        metrics = prediction_metrics(rows)
        self.assertEqual(set(metrics), set(PREDICTION_FIELDS))
        # hand-compute expected R2/MAE for model B over all 12 rows
        y = [row["chi_fd"] for row in rows]
        p = [row["chi_optimizer_only"] for row in rows]
        sst = sum((v - sum(y) / len(y)) ** 2 for v in y)
        expected_r2 = 1.0 - sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / sst
        expected_mae = sum(abs(pi - yi) for pi, yi in zip(p, y)) / len(y)
        self.assertAlmostEqual(metrics["B_optimizer_only"]["overall"]["R2"], expected_r2)
        self.assertAlmostEqual(metrics["B_optimizer_only"]["overall"]["MAE"], expected_mae)
        # monotone improvement ladder in this synthetic data
        self.assertGreater(
            metrics["BH_curvature_feedback"]["overall"]["R2"],
            metrics["B_optimizer_only"]["overall"]["R2"],
        )
        self.assertGreater(
            metrics["C_full_tangent"]["overall"]["R2"],
            metrics["BH_curvature_feedback"]["overall"]["R2"],
        )


class TestA0CCrosscheck(unittest.TestCase):
    def _a0c_like_summary(self, rows, streams):
        return {
            "results": [
                {
                    "seed": r["seed"], "step": r["step"], "H": r["H"],
                    "chi_A_momentum": r["chi_A_momentum"],
                    "chi_optimizer_only": r["chi_optimizer_only"],
                    "chi_full_tangent": r["chi_full_tangent"],
                }
                for r in rows
            ],
            "stream_results": streams,
        }

    def test_matching_values_pass(self):
        rows = _full_rows()
        streams = [
            {
                "seed": seed, "step": step, "stream_id": sid,
                "chi_optimizer_only": {str(h): 0.01 * h for h in HORIZONS_PROP},
                "chi_full_tangent": {str(h): 0.01 * h for h in HORIZONS_PROP},
            }
            for seed, step in SELECTED_STATES for sid in range(8)
        ]
        a0c = self._a0c_like_summary(rows, streams)
        result = validate_against_a0c(
            result_rows=rows, stream_results=streams, a0c_summary=a0c
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["rows_checked"], len(rows))

    def test_mismatch_raises(self):
        rows = _full_rows()
        streams = []
        a0c = self._a0c_like_summary(rows, streams)
        a0c["results"][0]["chi_optimizer_only"] += 1.0
        with self.assertRaises(ValueError):
            validate_against_a0c(
                result_rows=rows, stream_results=streams, a0c_summary=a0c
            )


class TestAssembleAndRender(unittest.TestCase):
    def _summary(self):
        rows = _full_rows()
        streams = [
            {
                "seed": seed, "step": step, "stream_id": sid,
                "chi_optimizer_only": {str(h): 0.01 * h for h in HORIZONS_PROP},
                "chi_full_tangent": {str(h): 0.01 * h for h in HORIZONS_PROP},
            }
            for seed, step in SELECTED_STATES for sid in range(8)
        ]
        metrics = prediction_metrics(rows)
        gains = aggregate_gains(rows)
        crosscheck = {
            "pass": True, "tolerance": 1e-9,
            "max_abs_gap_by_field": {"chi_optimizer_only": 0.0},
            "rows_checked": len(rows), "streams_checked": len(streams),
        }
        return assemble_summary(
            result_rows=rows,
            stream_results=streams,
            metrics=metrics,
            gains=gains,
            a0c_crosscheck=crosscheck,
            source_paths={},
            input_hashes={},
            runtime={
                "baseline_replay": "PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS",
                "max_saved_chi_abs_gap": 0.0,
                "input_integrity": "PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION",
                "hvp_backend": "MATH_SDPA",
                "max_hvp_math_gradient_relative_gap_vs_native": 0.0,
                "clip_active_steps_h_gt_1": 0,
                "total_steps_h_gt_1": 1,
                "max_abs_BH_minus_C_H8": 0.0,
                "device": "cuda:0",
                "amp_dtype": "bfloat16",
                "torch_version": "2.11.0",
                "elapsed_seconds": 1.0,
            },
            conclusions={
                "curvature_contribution": "POSITIVE_MEAN_GAIN_H_+0.1",
                "curvature_explanation": "",
                "optimizer_contribution": "POSITIVE_MEAN_GAIN_OPTIMIZER_+0.3",
                "optimizer_explanation": "",
                "bh_vs_c_gap": "MEAN_CLIP_CORRECTION_+0.05",
                "bh_vs_c_explanation": "",
                "recommended_next_step": "next",
                "further_question": "question",
            },
        )

    def test_assemble_grain_enforced(self):
        summary = self._summary()
        self.assertEqual(summary["protocol"], "FR_A0E_CURVATURE_DECOMPOSITION_V01")
        self.assertEqual(summary["status"], "COMPLETE")
        with self.assertRaises(ValueError):
            self._summary_broken_grain()

    def _summary_broken_grain(self):
        rows = _full_rows()[:-1]
        streams = [
            {
                "seed": seed, "step": step, "stream_id": sid,
                "chi_optimizer_only": {str(h): 0.01 * h for h in HORIZONS_PROP},
                "chi_full_tangent": {str(h): 0.01 * h for h in HORIZONS_PROP},
            }
            for seed, step in SELECTED_STATES for sid in range(8)
        ]
        return assemble_summary(
            result_rows=rows,
            stream_results=streams,
            metrics=prediction_metrics(_full_rows()),
            gains=aggregate_gains(_full_rows()),
            a0c_crosscheck={"pass": True},
            source_paths={},
            input_hashes={},
            runtime={},
            conclusions={},
        )

    def test_render_report_tables(self):
        report = render_report(self._summary())
        self.assertIn("# FR A0E Curvature Contribution Decomposition v01", report)
        self.assertIn("| state | H | chi_fd | A | B | BH | C |", report)
        self.assertIn("gain_optimizer = relA - relB", report)
        self.assertIn("gain_H = relB - relBH", report)
        self.assertIn("| model | overall R2 | corr | MAE | RMSE | mean relative error |", report)
        self.assertIn("BH_curvature_feedback", report)


if __name__ == "__main__":
    unittest.main()
