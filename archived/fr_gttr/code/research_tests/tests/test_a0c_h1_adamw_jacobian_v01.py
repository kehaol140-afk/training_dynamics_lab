from __future__ import annotations

import copy
import unittest
from collections import OrderedDict

import torch

from fr.a0c_h1_adamw_jacobian_v01 import (
    adamw_h1_jacobian,
    assemble_summary,
    epsilon_levels,
    render_report,
    validate_e2_manifest,
)
from fr.batch_manifest import manifest_sha256


def checkpoint_fixture(*, gradient, step=9):
    return {
        "model": OrderedDict([("p", torch.tensor([0.7, -0.4], dtype=torch.float64))]),
        "optimizer": {
            "param_groups": [
                {
                    "params": [0],
                    "lr": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.03,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": False,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                }
            ],
            "state": {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.tensor([0.2, -0.1], dtype=torch.float64),
                    "exp_avg_sq": torch.tensor([0.04, 0.09], dtype=torch.float64),
                }
            },
        },
        "metadata": {"train_seed": 2037, "state_optimizer_step": step},
    }


def native_projected_step(checkpoint, gradient, q, *, pulse, max_norm):
    theta = checkpoint["model"]["p"].detach().clone()
    parameter = torch.nn.Parameter(theta.clone())
    group = checkpoint["optimizer"]["param_groups"][0]
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=group["lr"],
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        foreach=False,
    )
    saved = checkpoint["optimizer"]["state"][0]
    optimizer.state[parameter] = {
        "step": saved["step"].detach().clone(),
        "exp_avg": saved["exp_avg"].detach().clone(),
        "exp_avg_sq": saved["exp_avg_sq"].detach().clone(),
    }
    parameter.grad = gradient.detach().clone() + float(pulse) * q
    torch.nn.utils.clip_grad_norm_([parameter], max_norm, foreach=False)
    optimizer.step()
    return float(torch.sum(q * (parameter.detach() - theta)))


class TestExactAdamWJacobian(unittest.TestCase):
    def _assert_matches_native_central_difference(self, gradient, max_norm):
        q = torch.tensor([0.6, 0.8], dtype=torch.float64)
        checkpoint = checkpoint_fixture(gradient=gradient)
        result = adamw_h1_jacobian(
            checkpoint=checkpoint,
            q_by_name={"p": q},
            gradient_by_name={"p": gradient},
            effective_lr=0.01,
            grad_clip=max_norm,
        )
        delta = 1e-6
        plus = native_projected_step(
            copy.deepcopy(checkpoint), gradient, q, pulse=delta, max_norm=max_norm
        )
        minus = native_projected_step(
            copy.deepcopy(checkpoint), gradient, q, pulse=-delta, max_norm=max_norm
        )
        finite_difference = (plus - minus) / (2.0 * delta)

        self.assertAlmostEqual(result["chi_jac"], finite_difference, places=8)
        self.assertAlmostEqual(
            result["chi_jac"],
            result["chi_jac_m"] + result["chi_jac_v"],
            places=14,
        )
        self.assertEqual(result["optimizer_step_before"], 9.0)
        self.assertEqual(result["optimizer_step_after"], 10.0)
        self.assertGreater(result["mu_q_old"], 0.0)

    def test_unclipped_complete_jacobian_matches_native_adamw(self):
        """Catches frozen-v, missing bias correction, or wrong step indexing."""
        self._assert_matches_native_central_difference(
            torch.tensor([0.3, -0.4], dtype=torch.float64), 10.0
        )

    def test_clipped_complete_jacobian_matches_native_adamw(self):
        """Catches treating global-norm clipping as a scalar constant."""
        self._assert_matches_native_central_difference(
            torch.tensor([3.0, 4.0], dtype=torch.float64), 1.0
        )


class TestExperimentSummary(unittest.TestCase):
    def test_manifest_identity_is_checked_before_replay(self):
        """Catches substituting a stream or state manifest."""
        batches = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        payload = {
            "manifest_version": "0.2",
            "sha256": manifest_sha256(batches),
            "metadata": {
                "protocol": "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM",
                "train_seed": 2037,
                "state_step": 46,
                "stream_id": 0,
                "stream_seed": 123,
                "batch_seed": 456,
                "epsilon": 0.4,
                "rho": 0.25,
            },
            "batches": batches,
        }
        result = validate_e2_manifest(
            payload,
            expected_seed=2037,
            expected_step=46,
            expected_stream=0,
            expected_content_sha=payload["sha256"],
            accumulation_steps=2,
        )
        self.assertEqual(result["batches"], batches)
        with self.assertRaisesRegex(ValueError, "state_step=46"):
            validate_e2_manifest(
                payload,
                expected_seed=2037,
                expected_step=82,
                expected_stream=0,
                expected_content_sha=payload["sha256"],
                accumulation_steps=2,
            )

    def test_epsilon_levels_are_registered_halvings(self):
        """Catches an empirical epsilon grid or accidental zero pulse."""
        self.assertEqual(epsilon_levels(0.4), [0.4, 0.2, 0.1])

    def test_summary_exposes_required_state_and_epsilon_fields(self):
        """Catches a report that drops convergence or clipping evidence."""
        epsilon_rows = []
        state_results = []
        for seed, step in ((2037, 46), (2030, 46), (2037, 82)):
            rows = []
            for epsilon, chi_fd in ((0.4, -0.9), (0.2, -0.95), (0.1, -0.99)):
                row = {
                    "state": f"seed_{seed}/state_step_{step:03d}",
                    "seed": seed,
                    "step": step,
                    "epsilon": epsilon,
                    "chi_fd": chi_fd,
                    "chi_jac": -1.0,
                    "relative_error": abs(chi_fd + 1.0),
                    "clip_before": {"mean": 1.2},
                    "clip_after": {"mean": 1.0},
                    "mu_q_old": 10.0,
                    "mu_q_full": 2.0,
                }
                rows.append(row)
                epsilon_rows.append(row)
            state_results.append(
                {
                    "state": rows[0]["state"],
                    "seed": seed,
                    "step": step,
                    "epsilon": 0.1,
                    "chi_fd": -0.99,
                    "chi_jac": -1.0,
                    "relative_error": 0.01,
                    "clip_before": {"mean": 1.2},
                    "clip_after": {"mean": 1.0},
                    "mu_q_old": 10.0,
                    "mu_q_full": 2.0,
                    "epsilon_results": rows,
                    "convergence_monotonic": True,
                }
            )
        summary = assemble_summary(
            state_results=state_results,
            epsilon_rows=epsilon_rows,
            stream_results=[{}] * 24,
            source_paths={"checkpoint_root": "/read-only"},
            input_hashes={"inputs": "abc"},
            runtime={"device": "cuda:0"},
            conclusions={
                "h1_explains_real_response": "YES",
                "nu_anomaly_from_frozen_v": "YES",
                "hessian_feedback_next": "YES_FOR_H_GT_1",
            },
        )
        report = render_report(summary)

        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["state_count"], 3)
        self.assertEqual(summary["stream_count"], 24)
        self.assertEqual(summary["epsilon_row_count"], 9)
        for field in (
            "state", "seed", "step", "epsilon", "chi_fd", "chi_jac",
            "relative_error", "clip_before", "clip_after", "mu_q_old",
            "mu_q_full",
        ):
            self.assertIn(field, summary["epsilon_results"][0])
        self.assertIn("H1 是否能解释真实响应", report)
        self.assertIn("nu 异常是否来自冻结 v 近似", report)
        self.assertIn("是否需要 Hessian feedback", report)


if __name__ == "__main__":
    unittest.main()
