from __future__ import annotations

import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from research.scripts.a0c_adamw_response_propagation_v01 import (
    _hvp_on_accumulated_batches,
)

from fr.a0c_adamw_response_propagation_v01 import (
    HORIZONS,
    SELECTED_STATES,
    adamw_tangent_step,
    assess_baseline_replay,
    assemble_summary,
    clip_jvp,
    extract_v07_fd,
    old_momentum_kernel,
    prediction_metrics,
    project_tangent,
    render_report,
)


def _optimizer(parameter):
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.03,
        foreach=False,
    )
    optimizer.state[parameter] = {
        "step": torch.tensor(9.0, dtype=torch.float64),
        "exp_avg": torch.tensor([0.2, -0.1], dtype=torch.float64),
        "exp_avg_sq": torch.tensor([0.04, 0.09], dtype=torch.float64),
    }
    return optimizer


def _native_two_step(*, pulse, hessian, max_norm):
    parameter = torch.nn.Parameter(torch.tensor([0.7, -0.4], dtype=torch.float64))
    optimizer = _optimizer(parameter)
    q = torch.tensor([0.6, 0.8], dtype=torch.float64)
    linear = torch.tensor([-0.12, 0.08], dtype=torch.float64)
    for step in range(2):
        gradient = hessian @ parameter.detach() + linear
        if step == 0:
            gradient = gradient + float(pulse) * q
        parameter.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([parameter], max_norm, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(torch.sum(q * parameter.detach()))


def _analytic_two_step(*, hessian, max_norm, include_hessian):
    parameter = torch.nn.Parameter(torch.tensor([0.7, -0.4], dtype=torch.float64))
    optimizer = _optimizer(parameter)
    q = torch.tensor([0.6, 0.8], dtype=torch.float64)
    linear = torch.tensor([-0.12, 0.08], dtype=torch.float64)
    zero = torch.zeros_like(parameter)
    tangent_theta = {"p": zero.clone()}
    tangent_m = {"p": zero.clone()}
    tangent_v = {"p": zero.clone()}
    for step in range(2):
        gradient = hessian @ parameter.detach() + linear
        if step == 0:
            tangent_gradient = q
        elif include_hessian:
            tangent_gradient = hessian @ tangent_theta["p"]
        else:
            tangent_gradient = zero
        clipped = clip_jvp(
            gradient_by_name={"p": gradient},
            tangent_gradient_by_name={"p": tangent_gradient},
            max_norm=max_norm,
        )
        tangent = adamw_tangent_step(
            named_parameters=[("p", parameter)],
            optimizer=optimizer,
            clipped_gradient_by_name=clipped["clipped_gradient"],
            tangent_clipped_gradient_by_name=clipped["clip_jvp"],
            tangent_theta_by_name=tangent_theta,
            tangent_m_by_name=tangent_m,
            tangent_v_by_name=tangent_v,
        )
        tangent_theta = tangent["tangent_theta"]
        tangent_m = tangent["tangent_m"]
        tangent_v = tangent["tangent_v"]
        parameter.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([parameter], max_norm, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return project_tangent({"p": q}, tangent_theta)


class TestTangentPrimitives(unittest.TestCase):
    def test_global_clip_jvp_is_identity_when_clip_is_inactive(self):
        """Catches applying the active radial derivative below the clip bound."""
        gradient = torch.tensor([0.03, 0.04], dtype=torch.float64)
        direction = torch.tensor([0.4, -0.7], dtype=torch.float64)
        result = clip_jvp(
            gradient_by_name={"p": gradient},
            tangent_gradient_by_name={"p": direction},
            max_norm=1.0,
        )
        torch.testing.assert_close(result["clipped_gradient"]["p"], gradient)
        torch.testing.assert_close(result["clip_jvp"]["p"], direction)
        self.assertFalse(result["clip_active"])

    def test_global_clip_jvp_matches_independent_central_difference(self):
        """Catches freezing the clip coefficient or dropping the radial term."""
        gradient = torch.tensor([3.0, 4.0], dtype=torch.float64)
        direction = torch.tensor([0.4, -0.7], dtype=torch.float64)
        result = clip_jvp(
            gradient_by_name={"p": gradient},
            tangent_gradient_by_name={"p": direction},
            max_norm=1.0,
        )
        delta = 1e-6

        def clipped(value):
            coefficient = min(1.0 / (float(torch.linalg.vector_norm(value)) + 1e-6), 1.0)
            return value * coefficient

        expected = (clipped(gradient + delta * direction) - clipped(gradient - delta * direction)) / (2 * delta)
        torch.testing.assert_close(result["clip_jvp"]["p"], expected, rtol=2e-8, atol=2e-9)
        self.assertTrue(result["clip_active"])

    def test_two_step_full_tangent_matches_native_finite_difference(self):
        """Catches missing m/v memory, bias correction, decay, or curvature HVP."""
        hessian = torch.tensor([[0.7, 0.2], [0.2, 1.1]], dtype=torch.float64)
        delta = 1e-6
        finite_difference = (
            _native_two_step(pulse=delta, hessian=hessian, max_norm=0.5)
            - _native_two_step(pulse=-delta, hessian=hessian, max_norm=0.5)
        ) / (2 * delta)
        full = _analytic_two_step(hessian=hessian, max_norm=0.5, include_hessian=True)
        optimizer_only = _analytic_two_step(
            hessian=hessian, max_norm=0.5, include_hessian=False
        )
        self.assertAlmostEqual(full, finite_difference, places=8)
        self.assertGreater(abs(optimizer_only - finite_difference), 1e-6)

    def test_reverse_over_reverse_hvp_matches_quadratic_literal(self):
        """Catches differentiating the wrong loss or dropping accumulation terms."""
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(
                    torch.tensor([0.7, -0.4], dtype=torch.float64)
                )
                self.register_buffer(
                    "hessian",
                    torch.tensor([[0.7, 0.2], [0.2, 1.1]], dtype=torch.float64),
                )

            def forward(self, x, labels=None):
                loss = 0.5 * self.p @ self.hessian @ self.p
                return SimpleNamespace(loss=loss, aux_loss=loss.new_zeros(()))

        model = Quadratic()
        direction = torch.tensor([0.6, 0.8], dtype=torch.float64)
        batches = [(torch.zeros(1), torch.zeros(1)) for _ in range(2)]
        hvp, gradient = _hvp_on_accumulated_batches(
            torch=torch,
            model=model,
            named=[("p", model.p)],
            batch_tensors=batches,
            tangent_theta_by_name={"p": direction},
            autocast_ctx=nullcontext(),
            accumulation_steps=2,
        )
        torch.testing.assert_close(
            hvp["p"], torch.tensor([0.58, 1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            gradient["p"], torch.tensor([0.41, -0.30], dtype=torch.float64)
        )

    def test_old_kernel_uses_historical_first_future_lr_and_raw_v_definition(self):
        """Catches substituting the checkpoint's preceding-step LR into model A."""
        checkpoint = {
            "model": {"p": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            "optimizer": {
                "param_groups": [{"params": [0], "lr": 0.02, "betas": (0.9, 0.999), "eps": 0.0}],
                "state": {0: {"exp_avg_sq": torch.tensor([1.0, 4.0], dtype=torch.float64), "step": torch.tensor(5.0)}},
            },
        }
        result = old_momentum_kernel(
            checkpoint,
            {"p": torch.tensor([0.6, 0.8], dtype=torch.float64)},
            [1, 2],
            eta=0.019,
        )
        # mu = .36/1 + .64/2 = .68
        self.assertAlmostEqual(result["mu_q"], 0.68, places=7)
        self.assertAlmostEqual(result["predictions"]["1"], -0.001292, places=12)
        self.assertAlmostEqual(result["predictions"]["2"], -0.0024548, places=12)


class TestEvidenceAndReport(unittest.TestCase):
    def test_h8_replay_uses_equal_horizon_shadow_hash_not_saved_h32_hash(self):
        """Catches comparing an H=8 state hash with v07's H=32 final hash."""
        propagated = {
            "model_hash_final": "h8-model",
            "optimizer_hash_final": "h8-optimizer",
            "baseline_X_theta": {"1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0},
        }
        shadow = {
            "model_hash_final": "h8-model",
            "optimizer_hash_final": "h8-optimizer",
            "observations": {
                key: {"X_theta": value}
                for key, value in propagated["baseline_X_theta"].items()
            },
        }
        saved_h32 = {
            "model_hash_final": "h32-model",
            "optimizer_hash_final": "h32-optimizer",
            "X_theta": {"1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0},
        }
        result = assess_baseline_replay(
            propagated=propagated,
            shadow_h8=shadow,
            saved_v07_h32=saved_h32,
            horizons=HORIZONS,
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["shadow_model_hash_equal"])
        self.assertTrue(result["shadow_optimizer_hash_equal"])
        self.assertFalse(result["saved_final_hash_comparable"])
        self.assertEqual(result["max_X_abs_gap_vs_saved_v07"], 0.0)
        self.assertEqual(result["propagated_H8_model_hash"], "h8-model")
        self.assertEqual(result["shadow_H8_optimizer_hash"], "h8-optimizer")

    def test_saved_v07_branches_yield_exact_unit_central_differences(self):
        """Catches an epsilon double-scaling or baseline subtraction in chi_fd."""
        theta_state = {
            "epsilon": 0.25,
            "chi_theta": {"1": 3.0, "2": 5.0},
            "e2_streams": [
                {"stream_id": 0, "plus": {"X_theta": {"1": 1.0, "2": 2.0}}, "minus": {"X_theta": {"1": -0.5, "2": -0.5}}},
                {"stream_id": 1, "plus": {"X_theta": {"1": 2.0, "2": 4.0}}, "minus": {"X_theta": {"1": 0.5, "2": 1.5}}},
            ],
        }
        result = extract_v07_fd(theta_state, [1, 2])
        self.assertEqual(result["per_stream"][0]["chi_fd"]["1"], 3.0)
        self.assertEqual(result["per_stream"][1]["chi_fd"]["2"], 5.0)
        self.assertEqual(result["mean"]["1"], 3.0)
        self.assertEqual(result["mean"]["2"], 5.0)
        self.assertEqual(result["max_abs_vs_saved_chi"], 0.0)

    def test_summary_and_report_cover_complete_locked_comparison(self):
        """Catches incomplete state/horizon grain or omitted A/B/C conclusions."""
        rows = []
        streams = []
        for seed, step in SELECTED_STATES:
            state = f"seed_{seed}/state_step_{step:03d}"
            for horizon in HORIZONS:
                observed = -float(horizon)
                rows.append(
                    {
                        "state": state,
                        "seed": seed,
                        "step": step,
                        "H": horizon,
                        "chi_fd": observed,
                        "chi_A_momentum": observed * 2.0,
                        "chi_optimizer_only": observed * 1.2,
                        "chi_full_tangent": observed * 1.01,
                        "relative_error_A": 1.0,
                        "relative_error_optimizer_only": 0.2,
                        "relative_error_full_tangent": 0.01,
                    }
                )
            for stream_id in range(8):
                streams.append({"state": state, "seed": seed, "step": step, "stream_id": stream_id})
        metrics = prediction_metrics(rows)
        summary = assemble_summary(
            result_rows=rows,
            stream_results=streams,
            metrics=metrics,
            source_paths={"checkpoint_root": "/read-only"},
            input_hashes={"config": "abc"},
            runtime={
                "device": "cuda:0",
                "baseline_replay": "PASS_EXACT",
                "max_hvp_math_gradient_relative_gap_vs_native": 0.005,
                "hvp_backend": "PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA",
                "native_flash_double_backward": "UNSUPPORTED_BY_PYTORCH",
            },
            conclusions={
                "h1_extends_to_multistep": "YES",
                "failure_source": "CURVATURE_FEEDBACK",
                "mori_zwanzig_needed": "NO",
                "explanation": "fixture",
            },
        )
        report = render_report(summary)
        self.assertEqual(summary["protocol"], "FR_A0C_ADAMW_RESPONSE_PROPAGATION_V01")
        self.assertEqual(summary["result_row_count"], 12)
        self.assertEqual(summary["stream_count"], 24)
        self.assertEqual(summary["fit_parameters"], [])
        self.assertIn("H=1 Jacobian 能否扩展到多步", report)
        self.assertIn("失败来自 optimizer memory 还是 curvature feedback", report)
        self.assertIn("0.005", report)
        self.assertIn("A→B does not isolate optimizer memory", report)
        self.assertIn("是否需要 Mori-Zwanzig", report)


if __name__ == "__main__":
    unittest.main()
