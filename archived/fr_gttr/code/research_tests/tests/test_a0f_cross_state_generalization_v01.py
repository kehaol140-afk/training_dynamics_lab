from __future__ import annotations

import unittest

# Importing the runner first registers research/src on sys.path (same
# convention as the other research test modules).
import research.scripts.a0f_cross_state_generalization_v01  # noqa: F401

from fr.a0c_adamw_response_propagation_v01 import relative_error
from fr.a0f_cross_state_generalization_v01 import (
    HORIZONS,
    SAMPLE_SEED,
    SEEDS,
    STATES_PER_SEED,
    STEPS,
    assemble_summary,
    distribution_stats,
    error_distributions,
    paired_b_c_comparison,
    prediction_metrics,
    render_report,
    sample_states,
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


def _sample_rows():
    states = sample_states()
    rows = []
    for index, (seed, step) in enumerate(states):
        for horizon in HORIZONS:
            chi_fd = 0.1 * horizon * (1.0 + 0.01 * index)
            # C better than B better than A on average, with one C regression;
            # at H=1 the real B and C tangents coincide (pulse only), so the
            # synthetic data models pred_c == pred_b there.
            c_slip = 0.002 if (seed, step) == states[3] else 0.0
            pred_b = chi_fd * (1.0 + 0.2 * 0.1)
            pred_c = pred_b if horizon == 1 else chi_fd * (1.0 + (0.05 + c_slip) * 0.1)
            rows.append(
                _row(
                    seed, step, horizon, chi_fd,
                    pred_a=chi_fd * (1.0 + 0.5 * 0.1),
                    pred_b=pred_b,
                    pred_c=pred_c,
                )
            )
    return states, rows


class TestSampling(unittest.TestCase):
    def test_sample_is_deterministic_and_covers_all_seeds(self):
        first = sample_states()
        second = sample_states()
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(SEEDS) * STATES_PER_SEED)
        self.assertEqual({seed for seed, _ in first}, set(SEEDS))
        for _, step in first:
            self.assertIn(step, STEPS)
        for seed in SEEDS:
            steps = sorted(step for s, step in first if s == seed)
            self.assertEqual(len(steps), STATES_PER_SEED)
            self.assertEqual(steps, sorted(set(steps)))

    def test_sample_depends_on_sample_seed(self):
        other = sample_states(sample_seed=SAMPLE_SEED + 1)
        self.assertNotEqual(set(other), set(sample_states()))


class TestDistributionStats(unittest.TestCase):
    def test_mean_std_median(self):
        stats = distribution_stats([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["std"], 1.2909944487358056)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertEqual(stats["n"], 4)

    def test_requires_two_finite_values(self):
        with self.assertRaises(ValueError):
            distribution_stats([1.0])
        with self.assertRaises(ValueError):
            distribution_stats([1.0, float("nan")])


class TestMetricsAndPairedComparison(unittest.TestCase):
    def test_metrics_reuse_frozen_semantics(self):
        _, rows = _sample_rows()
        metrics = prediction_metrics(rows)
        y = [row["chi_fd"] for row in rows]
        p = [row["chi_full_tangent"] for row in rows]
        sst = sum((v - sum(y) / len(y)) ** 2 for v in y)
        expected_r2 = 1.0 - sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / sst
        self.assertAlmostEqual(metrics["C_full_tangent"]["overall"]["R2"], expected_r2)
        self.assertEqual(
            set(metrics["C_full_tangent"]["per_horizon"]), {"1", "2", "4", "8"}
        )

    def test_paired_comparison_counts_wins_and_losses(self):
        _, rows = _sample_rows()
        paired = paired_b_c_comparison(rows)
        for horizon in HORIZONS:
            entry = paired["per_horizon"][str(horizon)]["relative_error"]
            self.assertEqual(entry["wins"] + entry["losses"] + entry["ties"], len(SEEDS) * STATES_PER_SEED)
            if horizon == 1:
                continue
            self.assertGreater(entry["wins"], 0)
            self.assertGreater(entry["distribution"]["mean"], 0.0)
        # H=1 has B == C exactly: all improvements are exactly zero
        h1 = paired["per_horizon"]["1"]["relative_error"]
        self.assertEqual(h1["wins"], 0)
        self.assertEqual(h1["losses"], 0)
        self.assertEqual(h1["ties"], len(SEEDS) * STATES_PER_SEED)
        self.assertAlmostEqual(h1["distribution"]["mean"], 0.0)

    def test_error_distributions_shape(self):
        _, rows = _sample_rows()
        dist = error_distributions(rows)
        for label in ("A_momentum", "B_optimizer_only", "C_full_tangent"):
            self.assertEqual(set(dist[label]), {"1", "2", "4", "8"})
            stats = dist[label]["2"]["relative_error"]
            self.assertEqual(stats["n"], len(SEEDS) * STATES_PER_SEED)


class TestAssembleAndRender(unittest.TestCase):
    def _summary(self):
        states, rows = _sample_rows()
        streams = [
            {
                "seed": seed, "step": step, "stream_id": sid,
                "chi_optimizer_only": {str(h): 0.01 * h for h in HORIZONS},
                "chi_full_tangent": {str(h): 0.01 * h for h in HORIZONS},
            }
            for seed, step in states for sid in range(8)
        ]
        metrics = prediction_metrics(rows)
        distributions = error_distributions(rows)
        paired = paired_b_c_comparison(rows)
        return assemble_summary(
            result_rows=rows,
            stream_results=streams,
            states=states,
            metrics=metrics,
            distributions=distributions,
            paired=paired,
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
                "consistency_across_states": "YES",
                "consistency_explanation": "",
                "model_ordering": "C_BEATS_B_BEATS_A",
                "ordering_explanation": "",
                "recommended_next_step": "next",
                "further_question": "question",
            },
        )

    def test_assemble_grain(self):
        summary = self._summary()
        self.assertEqual(summary["protocol"], "FR_A0F_CROSS_STATE_GENERALIZATION_V01")
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["result_row_count"], len(SEEDS) * STATES_PER_SEED * len(HORIZONS))
        broken = summary["results"][:-1]
        with self.assertRaises(ValueError):
            assemble_summary(
                result_rows=broken,
                stream_results=summary["stream_results"],
                states=sample_states(),
                metrics=summary["metrics"],
                distributions=summary["error_distributions"],
                paired=summary["paired_b_c"],
                source_paths={},
                input_hashes={},
                runtime={},
                conclusions={},
            )

    def test_render_report_tables(self):
        report = render_report(self._summary())
        self.assertIn("# FR A0F Cross-State Generalization v01", report)
        self.assertIn("| horizon | model | R2 | corr | MAE | RMSE | mean relative error |", report)
        self.assertIn("Paired B-vs-C comparison per state", report)
        self.assertIn("| seed | sampled steps |", report)
        self.assertIn("H=1 has B == C by construction", report)


if __name__ == "__main__":
    unittest.main()
