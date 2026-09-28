import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from fr.full_state import save_full_training_state
from fr.theta_protocol import run_theta_branch


class _TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, labels):
        loss = ((self.weight * x - labels) ** 2).mean()
        return SimpleNamespace(loss=loss, aux_loss=torch.zeros_like(loss))


class _StepCountingAdamW(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class TestThetaProtocolGradientObservation(unittest.TestCase):
    def test_returns_unscaled_accumulated_gradient_without_clip_or_step(self):
        model = _TinyLossModel()
        optimizer = _StepCountingAdamW(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ds = [
            (torch.tensor([1.0]), torch.tensor([1.0])),
            (torch.tensor([2.0]), torch.tensor([1.0])),
        ]

        def stack_batch(dataset, indices, device):
            xs, ys = zip(*(dataset[i] for i in indices))
            return torch.stack(xs), torch.stack(ys)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.pt"
            save_full_training_state(state_path, model, optimizer, scaler)
            result = run_theta_branch(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=nullcontext(),
                ds=ds,
                state_path=state_path,
                reference_theta={"weight": torch.tensor(0.0)},
                q={"weight": torch.tensor(1.0)},
                named=[("weight", model.weight)],
                batches=[[0], [1]],
                stream_seed=7,
                sign=0,
                epsilon=0.0,
                horizons=[1],
                accumulation_steps=2,
                batch_size=1,
                device="cpu",
                grad_clip=0.01,
                base_lr=0.1,
                global_micro_step=0,
                total_micro_steps=10,
                get_lr=lambda step, total, base: base,
                stack_batch=stack_batch,
                measure_s_first=False,
                restore_checkpoint_rng=False,
                capture_first_unscaled_gradient=True,
                stop_after_first_unscaled_gradient=True,
            )

        # At w=0 the two loss/2 gradients are -1 and -2, hence g_t=-3.
        self.assertAlmostEqual(
            float(result["first_unscaled_gradient"]["weight"]), -3.0, places=7
        )
        self.assertAlmostEqual(result["first_unscaled_grad_norm"], 3.0, places=7)
        self.assertAlmostEqual(result["first_step_lr"], 0.1, places=12)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(float(model.weight.detach()), 0.0)
        self.assertTrue(result["stopped_before_clip_and_step"])


if __name__ == "__main__":
    unittest.main()
