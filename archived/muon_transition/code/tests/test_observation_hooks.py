import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from muon_transition.observation_hooks import (
    ObservationHook,
    sampled_gradient_singular_values,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 96, bias=False)
        self.fc2 = nn.Linear(96, 128, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_observation_hook():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "obs.jsonl"
        hook = ObservationHook(out, every_n=2, name="t")
        model = Tiny()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        for step in range(1, 5):
            if not hook.should_log(step):
                x = torch.randn(8, 128)
                loss = model(x).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                continue
            ctx = hook.before_step(model)
            x = torch.randn(8, 128)
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            sv = sampled_gradient_singular_values(model)
            opt.step()
            rec = hook.after_step(model, ctx, step, grad_norm=1.23,
                                  sampled_sv=sv, domain_cosine=0.4, conflict_ratio=0.1)
            assert rec is not None
        lines = [json.loads(l) for l in open(out)]
        assert len(lines) == 2
        for rec in lines:
            for key in ("grad_global_norm", "sampled_gradient_singular_values",
                        "domain_gradient_cosine", "domain_conflict_ratio", "update_norm",
                        "weight_norm", "update_weight_ratio", "update_work_proxy"):
                assert key in rec, key
            assert rec["update_work_proxy"] is not None
            assert any(rec["sampled_gradient_singular_values"].keys())
    print("test_observation_hooks OK")


if __name__ == "__main__":
    test_observation_hook()
