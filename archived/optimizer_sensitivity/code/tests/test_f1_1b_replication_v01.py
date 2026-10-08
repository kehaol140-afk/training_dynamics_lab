"""F1 unit tests: module taxonomy, c=0 diagonal identity (incl. zero-v central
derivative), fp16/bf16 round-trip attribution, prereg constants."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opsens.f1_1b_replication_v01 as f1


def test_classify_groups_olmo_layout():
    assert f1.classify_groups("transformer.wte.weight") == [("embed", slice(None))]
    assert f1.classify_groups("transformer.ff_out.weight") == [("head", slice(None))]
    qk = f1.classify_groups("transformer.blocks.3.att_proj.weight")
    assert [g for g, _ in qk] == ["attn_q", "attn_k", "attn_v"]
    assert qk[0][1].stop - qk[0][1].start == f1.ATT_D
    ff = f1.classify_groups("transformer.blocks.3.ff_proj.weight")
    assert [g for g, _ in ff] == ["mlp_gate", "mlp_up"]
    assert ff[0][1].stop - ff[0][1].start == f1.MLP_H
    assert f1.classify_groups("transformer.blocks.3.attn_out.weight") == [("attn_o", slice(None))]
    assert f1.classify_groups("transformer.blocks.3.q_norm.weight") == [("qknorm", slice(None))]
    assert f1.classify_groups("transformer.ln_f.weight") == [("norm", slice(None))]
    assert f1.classify_groups("transformer.blocks.3.ff_out.weight") == [("mlp_down", slice(None))]
    # all groups covered
    seen = set()
    for name in ("transformer.wte.weight", "transformer.ff_out.weight",
                 "transformer.ln_f.weight", "transformer.blocks.0.att_proj.weight",
                 "transformer.blocks.0.attn_out.weight", "transformer.blocks.0.q_norm.weight",
                 "transformer.blocks.0.k_norm.weight", "transformer.blocks.0.attn_norm.weight",
                 "transformer.blocks.0.ff_norm.weight", "transformer.blocks.0.ff_proj.weight",
                 "transformer.blocks.0.ff_out.weight"):
        for g, _ in f1.classify_groups(name):
            seen.add(g)
    assert seen <= set(f1.MODULE_GROUPS)


def test_diagonal_gain_matches_vendored_tangent_c0():
    """Gradient channel at c=0: vendored frozen tangent == -G*q exactly (fp64)."""
    from opsens.g1c_olmo2_tangent import g1c_adamw_tangent_step
    torch.manual_seed(7)
    v = torch.rand(64, 32, dtype=torch.float64) * 1e-6
    m = torch.randn(64, 32, dtype=torch.float64) * 1e-3
    step_t, lr, b1, b2, eps, wd = 950001.0, 2.77e-4, 0.9, 0.95, 1e-8, 0.1
    q = torch.randn(64, 32, dtype=torch.float64)
    q = q / q.norm()
    g = f1.diagonal_gain(v, step_t, lr, b1, b2, eps)
    expected = -g * q

    p = torch.nn.Parameter(torch.zeros_like(v))

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p], "lr": lr, "betas": (b1, b2), "eps": eps,
                         "weight_decay": wd, "amsgrad": False, "maximize": False,
                         "differentiable": False}]
    opt.state = {p: {"step": torch.tensor(step_t), "exp_avg": m, "exp_avg_sq": v}}
    out = g1c_adamw_tangent_step(
        named_parameters=[("x", p)], optimizer=opt,
        clipped_gradient_by_name={"x": torch.zeros_like(p)},
        tangent_clipped_gradient_by_name={"x": q},
        tangent_theta_by_name={"x": torch.zeros_like(p)},
        tangent_m_by_name={"x": torch.zeros_like(p)},
        tangent_v_by_name={"x": torch.zeros_like(p)},
    )
    got = out["tangent_theta"]["x"]
    rel = float((got - expected).norm() / expected.norm())
    assert rel < 1e-14, rel


def test_zero_v_central_derivative_formula():
    """At exact-zero v coords the c=0 diagonal equals the central-FD limit:
    derivative of -lr*(1-b1)*c/(bc1*(a|c|+eps)) at c=0 is -lr*(1-b1)/(bc1*eps).
    One-sided differs by factor eps/(a*eps+eps) < 1 (prereg section 0)."""
    lr, b1, b2, eps = 2.77e-4, 0.9, 0.95, 1e-8
    step_t = 950001.0
    bc1 = 1.0 - b1 ** (step_t + 1)
    bc2 = 1.0 - b2 ** (step_t + 1)
    a = math.sqrt((1.0 - b2) / bc2)
    v = torch.zeros(8, dtype=torch.float64)
    g = f1.diagonal_gain(v, step_t, lr, b1, b2, eps)
    expected = lr * (1 - b1) / (bc1 * eps)
    assert torch.allclose(g, torch.full_like(g, expected))
    # central FD at eps_fd -> 0 approaches the analytic value
    for eps_fd in (1e-10, 1e-11, 1e-12):
        c_pos, c_neg = eps_fd, -eps_fd
        denom_p = a * abs(c_pos) + eps
        upd_p = -lr * (1 - b1) * c_pos / (bc1 * denom_p)
        denom_n = a * abs(c_neg) + eps
        upd_n = -lr * (1 - b1) * c_neg / (bc1 * denom_n)
        fd = (upd_p - upd_n) / (2 * eps_fd)
        rel = abs(fd - (-expected)) / expected
        assert rel <= a * eps_fd / eps + 1e-15, (eps_fd, rel)


def test_precision_roundtrip_counts():
    v = torch.tensor([0.0, 1e-20, 3e-8, 1e-3], dtype=torch.float32)
    m = torch.tensor([0.0, 1e-9, 2e-8, 1.0], dtype=torch.float32)
    rt = f1.precision_roundtrip(v, m)
    assert rt["v_math_zero"] == 1
    assert rt["v_bf16_rt0"] == 0          # bf16 exponent range covers 1e-20
    assert rt["v_fp16_rt0"] == 1          # 1e-20 < fp16 min subnormal ~6e-8
    assert rt["m_fp16_rt0"] == 2          # 1e-9, 2e-8 both < 6e-8


def test_prereg_constants_frozen():
    assert f1.Q_STREAM1_TENSOR_SHA256.startswith("08128dd4")
    assert f1.CHECKPOINT_SIZES[(380000, "optim.pt")] == 11879442406
    assert f1.PARAM_COUNT == 1484916736
    assert len(f1.MODULE_GROUPS) == 12


def test_modulewise_norms_slices():
    named = [("transformer.blocks.0.att_proj.weight",
              torch.nn.Parameter(torch.zeros(3 * f1.ATT_D, 4)))]
    t = {"transformer.blocks.0.att_proj.weight": torch.ones(3 * f1.ATT_D, 4)}
    mods = f1.modulewise_norms(named, t)
    total = math.sqrt(3 * f1.ATT_D * 4)
    for g in ("attn_q", "attn_k", "attn_v"):
        assert abs(mods[g] - total / math.sqrt(3)) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
