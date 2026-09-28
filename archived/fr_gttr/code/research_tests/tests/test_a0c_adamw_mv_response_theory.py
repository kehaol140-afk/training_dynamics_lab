import math
import tempfile
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


try:
    from fr.a0c_adamw_mv_response_theory import (
        compute_mv_features,
        predict_unit_response,
        replay_first_step_gradient,
    )
except ImportError:
    compute_mv_features = None
    predict_unit_response = None
    replay_first_step_gradient = None

from fr.batch_manifest import save_manifest
from fr.full_state import save_full_training_state


def checkpoint_payload():
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
    }


class TestMvTheoryArithmetic(unittest.TestCase):
    def test_compute_mv_features_aligns_names_and_uses_raw_m_v_and_gradient(self):
        """Catches blind q insertion-order alignment or omission of m/g in nu."""
        self.assertIsNotNone(compute_mv_features, "missing theory implementation")
        q = OrderedDict(
            [
                ("b", torch.tensor([3.0])),
                ("w", torch.tensor([2.0, 4.0])),
            ]
        )
        gradient = {
            "w": torch.tensor([0.5, -1.0]),
            "b": torch.tensor([2.0]),
        }

        result = compute_mv_features(q, gradient, checkpoint_payload())

        self.assertAlmostEqual(result["mu_q"], 9.0, places=12)
        self.assertAlmostEqual(result["nu_q"], 1.75, places=12)
        self.assertEqual(result["parameter_names"], ["w", "b"])
        self.assertEqual(result["n_parameter_tensors"], 2)
        self.assertEqual(result["v_semantics"], "RAW_EXP_AVG_SQ_NO_BIAS_CORRECTION")
        self.assertEqual(result["m_semantics"], "RAW_EXP_AVG_NO_BIAS_CORRECTION")

    def test_unit_response_prediction_does_not_multiply_epsilon(self):
        """Catches repeating rho*sigma_q against already-normalized A0C chi."""
        self.assertIsNotNone(predict_unit_response, "missing prediction implementation")

        result = predict_unit_response(
            mu_q=2.0,
            nu_q=3.0,
            eta=0.1,
            beta1=0.9,
            beta2=0.999,
            horizon=2,
        )

        self.assertAlmostEqual(result["chi_m"], -0.038, places=15)
        self.assertAlmostEqual(result["chi_v"], 0.0005997, places=15)
        self.assertAlmostEqual(result["chi_mv"], -0.0374003, places=15)
        self.assertTrue(math.isfinite(result["chi_mv"]))


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
        self.forward_calls = 0

    def forward(self, x, labels):
        self.forward_calls += 1
        loss = torch.mean((self.linear(x) - labels) ** 2)
        return SimpleNamespace(loss=loss, aux_loss=torch.zeros_like(loss))


def tiny_stack_batch(ds, indices, device):
    rows = [ds[int(index)] for index in indices]
    x = torch.stack([row[0] for row in rows]).to(device)
    y = torch.stack([row[1] for row in rows]).to(device)
    return x, y


class TestFirstStepReplay(unittest.TestCase):
    def _fixture(self, root: Path, *, manifest_state=46):
        model = TinyLossModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        state_path = root / "state_step_046.pt"
        save_full_training_state(
            state_path,
            model,
            optimizer,
            scaler,
            metadata={
                "train_seed": 2030,
                "state_optimizer_step": 46,
                "global_micro_step": 92,
                "total_micro_steps": 256,
            },
        )
        manifest_path = root / "e2_stream_000_batches.json"
        save_manifest(
            manifest_path,
            [[0], [1], [0], [1]],
            metadata={
                "protocol": "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM",
                "train_seed": 2030,
                "state_step": manifest_state,
                "stream_id": 0,
                "stream_seed": 12345,
                "batch_seed": 67890,
                "rho": 0.25,
                "epsilon": 0.01,
            },
        )
        ds = [
            (torch.tensor([1.0]), torch.tensor([0.0])),
            (torch.tensor([2.0]), torch.tensor([0.0])),
        ]
        return model, optimizer, scaler, state_path, manifest_path, ds

    def test_replay_accumulates_then_clips_and_never_steps_optimizer(self):
        """Catches using one micro-batch, pre-unscale gradients, or optimizer.step."""
        self.assertIsNotNone(replay_first_step_gradient, "missing replay implementation")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            model, optimizer, scaler, state_path, manifest_path, ds = fixture

            def forbidden_step(*args, **kwargs):
                raise AssertionError("optimizer.step must not be called")

            optimizer.step = forbidden_step
            result = replay_first_step_gradient(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=nullcontext(),
                ds=ds,
                state_path=state_path,
                manifest_path=manifest_path,
                expected_seed=2030,
                expected_state_step=46,
                accumulation_steps=2,
                device="cpu",
                grad_clip=1.0,
                base_lr=0.1,
                global_micro_step=92,
                total_micro_steps=256,
                get_lr=lambda current, total, base: base,
                stack_batch=tiny_stack_batch,
            )

        self.assertEqual(model.forward_calls, 2)
        self.assertAlmostEqual(result["preclip_grad_norm"], 10.0, places=5)
        self.assertTrue(result["clipped"])
        self.assertAlmostEqual(
            float(result["gradient_by_name"]["linear.weight"].item()),
            0.9999999,
            places=5,
        )
        self.assertAlmostEqual(result["effective_lr"], 0.1, places=15)
        self.assertEqual(result["manifest_metadata"]["stream_id"], 0)

    def test_manifest_validation_rejects_wrong_state_before_forward(self):
        """Catches cross-state manifest substitution before any model use."""
        self.assertIsNotNone(replay_first_step_gradient, "missing replay implementation")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp), manifest_state=52)
            model, optimizer, scaler, state_path, manifest_path, ds = fixture

            with self.assertRaisesRegex(ValueError, "manifest state_step=52"):
                replay_first_step_gradient(
                    torch=torch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    autocast_ctx=nullcontext(),
                    ds=ds,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    expected_seed=2030,
                    expected_state_step=46,
                    accumulation_steps=2,
                    device="cpu",
                    grad_clip=1.0,
                    base_lr=0.1,
                    global_micro_step=92,
                    total_micro_steps=256,
                    get_lr=lambda current, total, base: base,
                    stack_batch=tiny_stack_batch,
                )

            self.assertEqual(model.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
