import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


try:
    from fr.a0c_momentum_theory_v02 import (
        build_summary,
        compute_mu_q_raw_v,
        safe_corr,
        safe_r2,
    )
except ImportError:
    build_summary = None
    compute_mu_q_raw_v = None
    safe_corr = None
    safe_r2 = None


CSV_FIELDS = [
    "seed",
    "state_step",
    "state_fraction",
    "horizon",
    "B",
    "chi",
    "alpha_direct",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def q_payload(*, sigma_q=0.4):
    # Reverse insertion order deliberately. Correct alignment must use names,
    # not blindly zip q insertion order to optimizer IDs.
    return {
        "metadata": {"sigma_q": sigma_q},
        "q": OrderedDict(
            [
                ("b", torch.tensor([3.0], dtype=torch.float32)),
                ("w", torch.tensor([2.0, 4.0], dtype=torch.float32)),
            ]
        ),
    }


def checkpoint_payload(*, seed, step, w_v, b_v):
    return {
        "model": OrderedDict(
            [
                ("w", torch.zeros(2, dtype=torch.float32)),
                ("b", torch.zeros(1, dtype=torch.float32)),
                ("unused_buffer", torch.zeros(1, dtype=torch.float32)),
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
                0: {"exp_avg_sq": torch.tensor(w_v, dtype=torch.float32)},
                1: {"exp_avg_sq": torch.tensor(b_v, dtype=torch.float32)},
            },
        },
        "metadata": {"train_seed": seed, "state_optimizer_step": step},
    }


def write_common_sources(root, rows, *, sigma_in_q=True):
    write_csv(
        root / "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv",
        rows,
    )
    q_path = root / "outputs/e1_fluctuation/v03_pilot/probe_direction_q.pt"
    q_path.parent.mkdir(parents=True, exist_ok=True)
    payload = q_payload()
    if not sigma_in_q:
        payload["metadata"].pop("sigma_q")
    torch.save(payload, q_path)
    report_path = root / "outputs/e1_fluctuation/v03_pilot/e1_fluctuation_report.json"
    report_path.write_text(
        json.dumps({"stats": {"sigma_q": 0.4}}), encoding="utf-8"
    )
    audit_path = root / "outputs/a0_fr_audit/v01/A0_DATA_AUDIT.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({"rho_values": [0.25]}), encoding="utf-8")


class TestMomentumTheoryCore(unittest.TestCase):
    def test_raw_v_mu_aligns_q_by_parameter_name(self):
        # A blind zip would pair b->[4,16] and fail shape validation. Correct
        # name alignment gives 4/2 + 16/4 + 9/3 = 9.
        self.assertIsNotNone(compute_mu_q_raw_v, "missing fr.a0c_momentum_theory_v02")
        out = compute_mu_q_raw_v(
            q_payload()["q"],
            checkpoint_payload(seed=1, step=46, w_v=[4.0, 16.0], b_v=[9.0]),
        )

        self.assertAlmostEqual(out["mu_q"], 9.0, places=12)
        self.assertEqual(out["parameter_names"], ["w", "b"])
        self.assertEqual(out["n_parameter_tensors"], 2)
        self.assertEqual(out["beta1"], 0.9)
        self.assertEqual(out["eta"], 0.1)

    def test_metrics_return_none_for_undefined_inputs(self):
        self.assertIsNotNone(safe_r2, "missing fr.a0c_momentum_theory_v02")
        self.assertIsNone(safe_r2([1.0], [1.0]))
        self.assertIsNone(safe_r2([2.0, 2.0], [2.0, 2.0]))
        self.assertIsNone(safe_corr([1.0], [1.0]))
        self.assertIsNone(safe_corr([1.0, 2.0], [3.0, 3.0]))

    def test_build_summary_uses_state_specific_v_and_existing_scale(self):
        self.assertIsNotNone(build_summary, "missing fr.a0c_momentum_theory_v02")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"seed": 7, "state_step": 46, "state_fraction": 0.4, "horizon": 1, "B": 0, "chi": -0.009, "alpha_direct": 0},
                {"seed": 7, "state_step": 46, "state_fraction": 0.4, "horizon": 2, "B": 0, "chi": -0.0171, "alpha_direct": 0},
                {"seed": 7, "state_step": 52, "state_fraction": 0.5, "horizon": 1, "B": 0, "chi": -0.0165, "alpha_direct": 0},
                {"seed": 7, "state_step": 52, "state_fraction": 0.5, "horizon": 2, "B": 0, "chi": -0.03135, "alpha_direct": 0},
            ]
            write_common_sources(root, rows, sigma_in_q=False)
            state_dir = root / "outputs/e3_confirmatory/v07/seed_7/states"
            state_dir.mkdir(parents=True)
            torch.save(
                checkpoint_payload(seed=7, step=46, w_v=[4.0, 16.0], b_v=[9.0]),
                state_dir / "state_step_046.pt",
            )
            torch.save(
                checkpoint_payload(seed=7, step=52, w_v=[1.0, 4.0], b_v=[4.0]),
                state_dir / "state_step_052.pt",
            )

            summary = build_summary(root)

        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["target_column"], "chi")
        self.assertEqual(summary["rho"], 0.25)
        self.assertEqual(summary["sigma_q"], 0.4)
        self.assertEqual(summary["sigma_q_source"]["kind"], "e1_report_stats")
        self.assertTrue(
            summary["checkpoint_source"].endswith("outputs/e3_confirmatory/v07")
        )
        self.assertAlmostEqual(summary["overall_R2"], 1.0, places=12)
        self.assertAlmostEqual(summary["overall_corr"], 1.0, places=12)
        self.assertAlmostEqual(summary["seed_R2"]["7"], 1.0, places=12)
        self.assertAlmostEqual(summary["horizon_R2"]["1"], 1.0, places=12)
        self.assertAlmostEqual(summary["horizon_R2"]["2"], 1.0, places=12)
        self.assertEqual(summary["missing_state_report"]["missing_state_count"], 0)
        self.assertEqual(summary["matched_state_count"], 2)
        self.assertEqual(summary["prediction_row_count"], 4)
        predicted = [row["chi_hat"] for row in summary["prediction_rows"]]
        for got, want in zip(predicted, [-0.009, -0.0171, -0.0165, -0.03135]):
            self.assertAlmostEqual(got, want, places=14)
        self.assertEqual(len(summary["matched_states"]), 2)
        self.assertEqual(summary["missing_states"], [])

    def test_v07_matching_never_falls_back_to_same_seed_state_in_v06(self):
        self.assertIsNotNone(build_summary, "missing fr.a0c_momentum_theory_v02")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"seed": 2030, "state_step": 46, "state_fraction": 0.4, "horizon": 1, "B": 0, "chi": -0.1, "alpha_direct": 0},
                {"seed": 2030, "state_step": 46, "state_fraction": 0.4, "horizon": 2, "B": 0, "chi": -0.2, "alpha_direct": 0},
            ]
            write_common_sources(root, rows)
            # A same-seed/same-step checkpoint exists, but only in the obsolete
            # v06 source. The corrected matcher must not consume it.
            state_dir = root / "outputs/e3_replication/v06/seed_2030/states"
            state_dir.mkdir(parents=True)
            torch.save(
                checkpoint_payload(seed=2030, step=46, w_v=[4.0, 16.0], b_v=[9.0]),
                state_dir / "state_step_046.pt",
            )

            summary = build_summary(root)

        self.assertEqual(summary["status"], "NO_MATCHED_STATES")
        self.assertIsNone(summary["overall_R2"])
        self.assertIsNone(summary["overall_corr"])
        self.assertEqual(summary["seed_R2"], {"2030": None})
        self.assertEqual(summary["horizon_R2"], {"1": None, "2": None})
        report = summary["missing_state_report"]
        self.assertEqual(report["requested_seeds"], [2030])
        self.assertEqual(report["available_seeds"], [])
        self.assertEqual(report["missing_state_count"], 1)
        self.assertEqual(report["entries"][0]["reason"], "EXACT_SEED_STATE_CHECKPOINT_NOT_FOUND")
        self.assertTrue(
            report["entries"][0]["expected_path"].endswith(
                "outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt"
            )
        )
        self.assertEqual(summary["matched_states"], [])
        self.assertEqual(summary["missing_states"], report["entries"])
        self.assertFalse(math.isnan(summary["input_row_count"]))


class TestMomentumTheoryCli(unittest.TestCase):
    def test_cli_writes_only_requested_v02_summary_for_missing_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"seed": 2030, "state_step": 46, "state_fraction": 0.4, "horizon": 1, "B": 0, "chi": -0.1, "alpha_direct": 0},
                {"seed": 2030, "state_step": 46, "state_fraction": 0.4, "horizon": 2, "B": 0, "chi": -0.2, "alpha_direct": 0},
            ]
            write_common_sources(root, rows)
            state_dir = root / "outputs/e3_replication/v06/seed_2030/states"
            state_dir.mkdir(parents=True)
            torch.save(
                checkpoint_payload(seed=2030, step=46, w_v=[4.0, 16.0], b_v=[9.0]),
                state_dir / "state_step_046.pt",
            )
            script = Path(__file__).resolve().parents[1] / "scripts/a0c_momentum_theory_v02.py"
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)

            result = subprocess.run(
                [sys.executable, str(script), "--project-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            output_dir = root / "outputs/a0_fr_audit/momentum_theory_v02_v07"
            output_path = output_dir / "summary.json"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_path.is_file())
            self.assertEqual([path.name for path in output_dir.iterdir()], ["summary.json"])
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "NO_MATCHED_STATES")
            for field in (
                "overall_R2",
                "overall_corr",
                "seed_R2",
                "horizon_R2",
                "matched_states",
                "missing_states",
                "checkpoint_source",
            ):
                self.assertIn(field, summary)


if __name__ == "__main__":
    unittest.main()
