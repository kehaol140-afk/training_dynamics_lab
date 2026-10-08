"""F1 mechanism library tests: closed-form state-channel blocks vs the vendored
frozen tangent at the c=0 anchor; anchor_update vs native torch.optim.AdamW
under a null gradient."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opsens.f0_bank_audit_v01 as f0
import opsens.f1_mechanism_v01 as fm


def _tiny_state(seed=3):
    torch.manual_seed(seed)
    model_keys = {"model.embed_tokens.weight": (17, 8),
                  "model.layers.0.self_attn.q_proj.weight": (8, 8),
                  "model.layers.0.mlp.gate_proj.weight": (8, 8)}
    state = {"model": {}, "optimizer": {"state": {}, "param_groups": [{
        "lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}]}}
    for i, (k, shape) in enumerate(model_keys.items()):
        state["model"][k] = torch.randn(shape) * 0.1
        state["optimizer"]["state"][i] = {
            "step": torch.tensor(46.0),
            "exp_avg": torch.randn(shape) * 1e-3,
            "exp_avg_sq": torch.rand(shape) * 1e-6 + 1e-12,
        }
    return state


def test_theta_channel_block():
    state = _tiny_state()
    named, opt = fm.fake_optimizer(state, torch.float64)
    hp = f0.group_hyperparams(state)
    gen = torch.Generator().manual_seed(1)
    q = {n: torch.randn(p.shape, generator=gen, dtype=torch.float64) for n, p in named}
    nrm = math.sqrt(sum(float((v ** 2).sum()) for v in q.values()))
    q = {k: v / nrm for k, v in q.items()}
    out = fm.run_tangent_state_channels(named, opt, q_theta=q)
    for n, p in named:
        expected = (1.0 - hp["lr"] * hp["weight_decay"]) * q[n]
        rel = float((out["tangent_theta"][n] - expected).norm() / expected.norm())
        assert rel < 1e-14, (n, rel)


def test_m_channel_block():
    state = _tiny_state()
    named, opt = fm.fake_optimizer(state, torch.float64)
    hp = f0.group_hyperparams(state)
    gen = torch.Generator().manual_seed(2)
    q = {n: torch.randn(p.shape, generator=gen, dtype=torch.float64) for n, p in named}
    nrm = math.sqrt(sum(float((v ** 2).sum()) for v in q.values()))
    q = {k: v / nrm for k, v in q.items()}
    out = fm.run_tangent_state_channels(named, opt, q_m=q)
    for n, idx, slot in f0.named_optimizer_entries(state):
        g = fm.gradient_channel_diagonal(slot["exp_avg_sq"], float(slot["step"]),
                                       hp["lr"], hp["beta1"], hp["beta2"], hp["eps"])
        expected = -fm.m_channel_diagonal(slot["exp_avg_sq"], float(slot["step"]),
                                       hp["lr"], hp["beta1"], hp["beta2"],
                                       hp["eps"]) * q[n]
        rel = float((out["tangent_theta"][n] - expected).norm() / expected.norm())
        assert rel < 1e-13, (n, rel)
    # dm' = beta1 * dm exactly
    for n, p in named:
        assert torch.allclose(out["tangent_m"][n], hp["beta1"] * q[n])


def test_v_channel_block_matches_analytic():
    state = _tiny_state()
    named, opt = fm.fake_optimizer(state, torch.float64)
    hp = f0.group_hyperparams(state)
    gen = torch.Generator().manual_seed(3)
    q = {n: torch.randn(p.shape, generator=gen, dtype=torch.float64) for n, p in named}
    nrm = math.sqrt(sum(float((v ** 2).sum()) for v in q.values()))
    q = {k: v / nrm for k, v in q.items()}
    out = fm.run_tangent_state_channels(named, opt, q_v=q)
    for n, idx, slot in f0.named_optimizer_entries(state):
        dv = fm.v_channel_diagonal(slot["exp_avg"], slot["exp_avg_sq"], float(slot["step"]),
                                   hp["lr"], hp["beta1"], hp["beta2"], hp["eps"])
        expected = dv * q[n]
        rel = float((out["tangent_theta"][n] - expected).norm() / expected.norm())
        assert rel < 1e-13, (n, rel)
    # dv' = beta2 * dv exactly
    for n, p in named:
        assert torch.allclose(out["tangent_v"][n], hp["beta2"] * q[n])


def test_anchor_update_matches_native_adamw_null_grad():
    state = _tiny_state()
    hp = f0.group_hyperparams(state)
    for name, idx, slot in f0.named_optimizer_entries(state):
        theta = state["model"][name].detach().to(torch.float64)
        p = torch.nn.Parameter(theta.clone())
        opt = torch.optim.AdamW([p], lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]),
                                eps=hp["eps"], weight_decay=hp["weight_decay"])
        opt.state[p] = {"step": slot["step"].double(),
                        "exp_avg": slot["exp_avg"].double(),
                        "exp_avg_sq": slot["exp_avg_sq"].double()}
        p.grad = torch.zeros_like(p)
        opt.step()
        native_delta = p.detach().double() - theta
        manual = fm.anchor_update(slot["exp_avg_sq"], slot["exp_avg"], float(slot["step"]),
                                  hp["lr"], hp["beta1"], hp["beta2"], hp["eps"],
                                  hp["weight_decay"], theta)
        rel = float((native_delta - manual).norm() / manual.norm())
        assert rel < 1e-12, (name, rel)


def test_vfloor_counterfactual_directional_behaviour():
    """Raising the v-floor must never increase the anchor denominator."""
    state = _tiny_state()
    hp = f0.group_hyperparams(state)
    name, idx, slot = next(iter(f0.named_optimizer_entries(state)))
    v = slot["exp_avg_sq"]
    g0 = fm.gradient_channel_diagonal(v, 46.0, hp["lr"], hp["beta1"],
                                       hp["beta2"], hp["eps"])
    for tau in (1e-8, 1e-7, 1e-6, 1e-5):
        v_t = torch.clamp(v, min=tau * tau)
        g_t = fm.gradient_channel_diagonal(v_t, 46.0, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"])
        assert bool((g_t <= g0 + 1e-15).all()), tau


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
