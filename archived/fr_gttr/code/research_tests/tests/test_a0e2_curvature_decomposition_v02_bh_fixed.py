from __future__ import annotations

import unittest

# Importing the runner first registers research/src on sys.path (same
# convention as the other research test modules).
import research.scripts.a0e2_curvature_decomposition_v02_bh_fixed  # noqa: F401

from fr.a0c_adamw_response_propagation_v01 import _metric, relative_error
from fr.a0e2_curvature_decomposition_v02_bh_fixed import (
    HORIZONS_PROP,
    PREDICTION_FIELDS,
    SCOPES,
    SELECTED_STATES,
    aggregate_gains,
    assemble_summary,
    prediction_metrics,
    render_report,
    row_gains,
    rows_in_scope,
    validate_against_a0c,
    validate_against_a0e,
)


def _row(seed, step, horizon, chi_fd, pred_a, pred_b, pred_bh_old, pred_bhf, pred_c,
         state=None):
    state = state or f"seed_{seed}/state_step_{step:03d}"
    return {
        "state": state,
        "seed": seed,
        "step": step,
        "H": horizon,
        "chi_fd": chi_fd,
        "chi_A_momentum": pred_a,
        "chi_optimizer_only": pred_b,
        "chi_BH_old_curvature": pred_bh_old,
        "chi_BH_fixed_curvature": pred_bhf,
        "chi_full_tangent": pred_c,
        "relative_error_A": relative_error(chi_fd, pred_a),
        "relative_error_optimizer_only": relative_error(chi_fd, pred_b),
        "relative_error_BH_old_curvature": relative_error(chi_fd, pred_bh_old),
        "relative_error_BH_fixed_curvature": relative_error(chi_fd, pred_bhf),
        "relative_error_full_tangent": relative_error(chi_fd, pred_c),
        "absolute_error_A": abs(pred_a - chi_fd),
        "absolute_error_optimizer_only": abs(pred_b - chi_fd),
        "absolute_error_BH_old_curvature": abs(pred_bh_old - chi_fd),
        "absolute_error_BH_fixed_curvature": abs(pred_bhf - chi_fd),
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
                    pred_bh_old=chi_fd * (1.0 + 0.3 * 0.1),
                    pred_bhf=chi_fd * (1.0 + 0.1 * 0.1),
                    pred_c=chi_fd * (1.0 + 0.05 * 0.1),
                )
            )
    return rows


def _minimal_summary(overrides=None):
    rows = _full_rows()
    metrics = {scope: prediction_metrics(rows, scope) for scope in SCOPES}
    gains = {scope: aggregate_gains(rows, scope) for scope in SCOPES}
    streams = [
        {
            "state": f"seed_{seed}/state_step_{step:03d}",
            "seed": seed,
            "step": step,
            "stream_id": stream_id,
        }
        for seed, step in SELECTED_STATES
        for stream_id in range(8)
    ]
    diagnostics = {
        "per_stream": [],
        "streams_initial_clip_active": 0,
        "future_clip_active_steps": 0,
        "future_total_steps": 0,
        "streams_bh_old_differs_from_B_at_H1": 0,
        "max_abs_BH_old_minus_B_H1": 0.0,
        "streams_bh_fixed_differs_from_B_at_H1": 0,
        "max_abs_BH_fixed_minus_B_H1": 0.0,
        "streams_bh_fixed_c_bit_identical_all_steps": 24,
        "max_abs_BH_fixed_minus_C": 0.0,
        "seed2037_step046_explanation": "synthetic",
    }
    a0f_a0f2_audit = {
        "files_checked": {"a": "p"},
        "sha256_at_a0e2_runtime": {"a": "d" * 64},
        "hashes_unchanged": "PASS_PRE_POST_SHA256_IDENTICAL",
        "a0f_frozen_wins_ties_losses": [96, 32, 0],
        "a0f_mean_improvement": 0.1,
        "a0f2_frozen_wins_ties_losses": [168, 56, 0],
        "a0f2_mean_improvement": 0.1,
    }
    params = dict(
        result_rows=rows,
        stream_results=streams,
        metrics_by_scope=metrics,
        gains_by_scope=gains,
        a0c_crosscheck={"pass": True, "max_abs_gap_by_field": {}, "rows_checked": 12,
                        "streams_checked": 24, "tolerance": 1e-9},
        a0e_crosscheck={"pass": True, "max_abs_gap_by_field": {}, "rows_checked": 12,
                        "streams_checked": 24, "tolerance": 1e-9},
        h1_identity={
            "max_abs_B_minus_BH_fixed_H1": 0.0,
            "max_abs_B_minus_C_H1": 0.0,
            "max_abs_BH_fixed_minus_C_H1": 0.0,
        },
        source_paths={},
        input_hashes={},
        runtime={
            "baseline_replay": "PASS",
            "max_saved_chi_abs_gap": 0.0,
            "input_integrity": "PASS",
            "hvp_backend": "x",
            "max_hvp_math_gradient_relative_gap_vs_native": 0.0,
            "clip_active_steps_h_gt_1": 0,
            "total_steps_h_gt_1": 0,
            "device": "cuda:0",
            "amp_dtype": "bfloat16",
            "torch_version": "2.11.0",
        },
        diagnostics=diagnostics,
        a0f_a0f2_audit=a0f_a0f2_audit,
        conclusions={
            "verdict1": "v1",
            "verdict2": "v2",
            "verdict3": "v3",
            "verdict4": "v4",
        },
    )
    if overrides:
        params.update(overrides)
    return assemble_summary(**params)


class TestGains(unittest.TestCase):
    def test_gains_match_definition_and_sum_to_total(self):
        row = _row(2037, 46, 2, 0.2, 0.30, 0.24, 0.26, 0.21, 0.22)
        gains = row_gains(row)
        # rel errors: A .5, B .2, BH_old .3, BH_fixed .05, C .1
        self.assertAlmostEqual(gains["gain_optimizer_rel"], 0.5 - 0.2)
        self.assertAlmostEqual(gains["gain_raw_curvature_fixed_rel"], 0.2 - 0.05)
        self.assertAlmostEqual(gains["gain_clip_JVP_rel"], 0.05 - 0.1)
        self.assertAlmostEqual(gains["total_rel_A_to_C"], 0.5 - 0.1)
        summed = (
            gains["gain_optimizer_rel"]
            + gains["gain_raw_curvature_fixed_rel"]
            + gains["gain_clip_JVP_rel"]
        )
        self.assertAlmostEqual(summed, gains["total_rel_A_to_C"])

    def test_aggregate_identity_holds_both_scopes(self):
        rows = _full_rows()
        for scope in SCOPES:
            gains = aggregate_gains(rows, scope)
            self.assertTrue(gains["identity_gain_sum_equals_total"])
            self.assertLessEqual(gains["max_identity_deviation"], 1e-12)
            summed = (
                gains["overall"]["gain_optimizer_rel"]
                + gains["overall"]["gain_raw_curvature_fixed_rel"]
                + gains["overall"]["gain_clip_JVP_rel"]
            )
            self.assertAlmostEqual(summed, gains["overall"]["total_rel_A_to_C"], places=15)

    def test_negative_gain_reported_not_clipped(self):
        # BH_fixed worse than B: raw curvature negative contribution must
        # surface as a negative gain, never be clamped to zero.
        row = _row(2037, 46, 2, 0.2, 0.3, 0.21, 0.25, 0.30, 0.22)
        gains = row_gains(row)
        self.assertLess(gains["gain_raw_curvature_fixed_rel"], 0.0)


class TestScopes(unittest.TestCase):
    def test_scope_row_counts(self):
        rows = _full_rows()
        self.assertEqual(len(rows_in_scope(rows, "all_12")), 12)
        self.assertEqual(len(rows_in_scope(rows, "h_gt_1_9")), 9)

    def test_scope_rejects_wrong_grain(self):
        rows = _full_rows()[:-1]
        with self.assertRaises(ValueError):
            rows_in_scope(rows, "all_12")
        with self.assertRaises(ValueError):
            prediction_metrics(rows, "all_12")

    def test_scope_metrics_use_frozen_metric(self):
        rows = _full_rows()
        for scope in SCOPES:
            scoped = rows_in_scope(rows, scope)
            block = prediction_metrics(rows, scope)
            for label, field in PREDICTION_FIELDS.items():
                expected = _metric(
                    [float(r["chi_fd"]) for r in scoped],
                    [float(r[field]) for r in scoped],
                )
                self.assertEqual(block[label]["overall"], expected)

    def test_all_12_and_h_gt_1_9_are_distinct_labeled_blocks(self):
        rows = _full_rows()
        metrics = {scope: prediction_metrics(rows, scope) for scope in SCOPES}
        self.assertIn("all_12", metrics)
        self.assertIn("h_gt_1_9", metrics)
        # The H=1 rows differ numerically between scopes.
        self.assertNotEqual(
            metrics["all_12"]["B_optimizer_only"]["overall"]["n"],
            metrics["h_gt_1_9"]["B_optimizer_only"]["overall"]["n"],
        )


class TestValidation(unittest.TestCase):
    def test_matching_values_pass(self):
        rows = _full_rows()
        a0c_summary = {
            "results": [
                {
                    "seed": r["seed"],
                    "step": r["step"],
                    "H": r["H"],
                    "chi_A_momentum": r["chi_A_momentum"],
                    "chi_optimizer_only": r["chi_optimizer_only"],
                    "chi_full_tangent": r["chi_full_tangent"],
                }
                for r in rows
            ],
            "stream_results": [],
        }
        check = validate_against_a0c(result_rows=rows, stream_results=[], a0c_summary=a0c_summary)
        self.assertTrue(check["pass"])
        self.assertEqual(check["rows_checked"], 12)

    def test_mismatch_raises(self):
        rows = _full_rows()
        a0c_summary = {
            "results": [
                {
                    "seed": rows[0]["seed"],
                    "step": rows[0]["step"],
                    "H": rows[0]["H"],
                    "chi_A_momentum": 999.0,
                    "chi_optimizer_only": rows[0]["chi_optimizer_only"],
                    "chi_full_tangent": rows[0]["chi_full_tangent"],
                }
            ],
            "stream_results": [],
        }
        with self.assertRaises(ValueError):
            validate_against_a0c(result_rows=rows[:1], stream_results=[], a0c_summary=a0c_summary)

    def test_a0e_bh_old_validation(self):
        rows = _full_rows()
        a0e_summary = {
            "results": [
                {
                    "seed": r["seed"],
                    "step": r["step"],
                    "H": r["H"],
                    "chi_BH_curvature": r["chi_BH_old_curvature"],
                }
                for r in rows
            ],
            "stream_results": [],
        }
        check = validate_against_a0e(result_rows=rows, stream_results=[], a0e_summary=a0e_summary)
        self.assertTrue(check["pass"])
        bad = {
            "results": [
                {
                    "seed": rows[0]["seed"],
                    "step": rows[0]["step"],
                    "H": rows[0]["H"],
                    "chi_BH_curvature": 999.0,
                }
            ],
            "stream_results": [],
        }
        with self.assertRaises(ValueError):
            validate_against_a0e(result_rows=rows[:1], stream_results=[], a0e_summary=bad)


class TestAssemble(unittest.TestCase):
    def test_full_rows_produce_locked_grain(self):
        summary = _minimal_summary()
        self.assertEqual(summary["result_row_count"], 12)
        self.assertEqual(summary["protocol"], "FR_A0E2_CURVATURE_DECOMPOSITION_V02_BH_FIXED")
        self.assertEqual(summary["scope"]["aggregation_scopes"]["all_12"]["rows"], 12)
        self.assertEqual(summary["scope"]["aggregation_scopes"]["h_gt_1_9"]["rows"], 9)

    def test_wrong_grain_rejected(self):
        rows = _full_rows()[:-1]
        with self.assertRaises(ValueError):
            _minimal_summary(overrides={"result_rows": rows})

    def test_metrics_must_cover_five_models(self):
        rows = _full_rows()
        metrics = prediction_metrics(rows, "all_12")
        del metrics["C_full_tangent"]
        with self.assertRaises(ValueError):
            _minimal_summary(overrides={"metrics_by_scope": {"all_12": metrics}})

    def test_report_renders_verdicts_and_scopes(self):
        report = render_report(_minimal_summary())
        self.assertIn("Verdict 1", report)
        self.assertIn("Verdict 2", report)
        self.assertIn("Verdict 3", report)
        self.assertIn("Verdict 4", report)
        self.assertIn("ALL 12 ROWS", report)
        self.assertIn("H>1 ONLY, 9 ROWS", report)
        self.assertIn("BH_fixed", report)
        self.assertIn("identity check", report)


if __name__ == "__main__":
    unittest.main()
