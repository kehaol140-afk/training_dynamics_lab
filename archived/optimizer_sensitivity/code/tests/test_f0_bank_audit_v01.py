import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opsens.f0_bank_audit_v01 as f0
from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step


def _tiny_state(seed=0, n=37):
    g = torch.Generator().manual_seed(seed)
    hp = {"lr": 3.7e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}
    theta = torch.randn(n, generator=g, dtype=torch.float64)
    m = torch.randn(n, generator=g, dtype=torch.float64) * 1e-3
    v = torch.rand(n, generator=g, dtype=torch.float64) * 1e-6
    v[0] = 0.0
    step_t = 46.0
    return theta, m, v, step_t, hp


class _Opt:
    pass


def _build(theta, m, v, step_t, hp, dtype=torch.float64):
    p = torch.nn.Parameter(theta.to(dtype).clone())
    opt = _Opt()
    opt.param_groups = [{"params": [p], "lr": hp["lr"], "betas": (hp["beta1"], hp["beta2"]),
                         "eps": hp["eps"], "weight_decay": hp["weight_decay"],
                         "amsgrad": False, "maximize": False, "differentiable": False}]
    opt.state = {p: {"step": torch.tensor(float(step_t)), "exp_avg": m.to(dtype).clone(),
                     "exp_avg_sq": v.to(dtype).clone()}}
    return [("x", p)], opt


class TestGrouping(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(f0.classify_group("model.embed_tokens.weight"), "embed_tied")
        self.assertEqual(f0.classify_group("model.layers.5.self_attn.q_proj.weight"), "attn_q")
        self.assertEqual(f0.classify_group("model.layers.5.self_attn.k_norm.weight"), "attn_qk_norm")
        self.assertEqual(f0.classify_group("model.layers.5.post_attention_layernorm.weight"), "rmsnorm")
        self.assertEqual(f0.classify_group("model.norm.weight"), "rmsnorm")
        self.assertEqual(f0.classify_group("model.layers.5.mlp.down_proj.weight"), "mlp_down")
        self.assertEqual(f0.classify_group("lm_head.weight"), "others")


class TestQuantiles(unittest.TestCase):
    def test_exact_quantiles(self):
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        q = f0.exact_quantiles(t, [0.0, 0.5, 1.0])
        self.assertEqual(q[0], 0.0)
        self.assertEqual(q[1], 2.0)
        self.assertEqual(q[2], 4.0)
        q2 = f0.exact_quantiles(t, [0.25])
        self.assertEqual(q2[0], 1.0)


class TestDiagonalGainIsFrozenTangent(unittest.TestCase):
    def test_c0_anchor_diagonal(self):
        theta, m, v, step_t, hp = _tiny_state()
        named, opt = _build(theta, m, v, step_t, hp)
        zero = {"x": torch.zeros_like(named[0][1])}
        out = adamw_tangent_step(
            named_parameters=named, optimizer=opt,
            clipped_gradient_by_name=zero,
            tangent_clipped_gradient_by_name={"x": torch.ones_like(named[0][1])},
            tangent_theta_by_name=zero, tangent_m_by_name=zero, tangent_v_by_name=zero)
        gain = f0.diagonal_gain(v, step_t, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"])
        self.assertTrue(torch.allclose(out["tangent_theta"]["x"].double(), -gain.double(), rtol=1e-12, atol=0.0))

    def test_zero_v_coordinate_hits_eps_floor(self):
        theta, m, v, step_t, hp = _tiny_state()
        gain = f0.diagonal_gain(v, step_t, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"])
        t_after = step_t + 1.0
        expected = hp["lr"] * (1 - hp["beta1"]) / ((1 - hp["beta1"] ** t_after) * hp["eps"])
        self.assertAlmostEqual(float(gain[0]), expected, places=20)


class TestManualVsNativeAdamW(unittest.TestCase):
    def test_match_fp64(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from f0_fd_validation import manual_adamw_update, native_adamw_step

        theta, m, v, step_t, hp = _tiny_state(seed=3)
        c = torch.randn(theta.numel(), generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 1e-4
        manual = manual_adamw_update(theta, m, v, step_t, c, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"])
        native = native_adamw_step(theta, m, v, step_t, c, hp, torch.float64)
        self.assertTrue(torch.allclose(manual.double(), native.double(), rtol=1e-12, atol=1e-15))


class TestFloorStats(unittest.TestCase):
    def test_zeros_and_thresholds(self):
        v = torch.zeros(10)
        v[0] = 1e-4
        s = f0.tensor_floor_stats(v, 1e-8)
        self.assertEqual(s["exact_zero_count"], 9)
        self.assertAlmostEqual(s["exact_zero_fraction"], 0.9)
        self.assertAlmostEqual(s["sqrt_v_lt_eps_fraction"], 0.9)


if __name__ == "__main__":
    unittest.main()
