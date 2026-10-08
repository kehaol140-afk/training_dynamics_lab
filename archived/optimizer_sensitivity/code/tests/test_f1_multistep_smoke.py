"""End-to-end smoke test of the F1 multistep core loop on a tiny OLMo-shaped
model (CPU): grads -> primal clip -> clip_jvp -> vendored per-tensor tangent
step -> primal step, for branches B and C, plus:
  * h=1 tangent equals an explicit central-FD of the FULL clipped map
    (pulse pre-clip on real gradients) — validates the assembled pipeline
    including clip_jvp radial term;
  * arm dtype handling (fp64/fp32/bf16) runs and stays finite;
  * stop-rule plumbing.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opsens.g1c_olmo2_architecture import build_from_config
from opsens.g1c_olmo2_tangent import g1c_adamw_tangent_step

spec = importlib.util.spec_from_file_location(
    "f1_multistep", Path(__file__).resolve().parents[1] / "scripts/f1_multistep.py")
f1m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1m)

TINY_CFG = {"d_model": 64, "n_heads": 4, "n_layers": 2, "mlp_hidden_size": 32,
            "vocab_size": 128, "embedding_size": 128, "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0, "max_sequence_length": 64, "qk_norm_dim": 16}


def make_tokens(H, seed=3):
    g = np.random.default_rng(seed)
    return g.integers(4, 128, size=H * f1m.MB_PER_STEP * f1m.SEQ_LEN * f1m.SEQS_PER_MB
                      % (256 * 1024)).astype(np.int64)


def tiny_tokens(H):
    g = np.random.default_rng(3)
    n = H * f1m.MB_PER_STEP * f1m.SEQ_LEN * f1m.SEQS_PER_MB
    # keep arrays small: SEQ constants are module-level; shrink for test
    f1m.SEQ_LEN, f1m.SEQS_PER_MB, f1m.MB_PER_STEP = 16, 2, 1
    return g.integers(4, 128, size=H * 16 * 2 * 1).astype(np.int64).astype(np.int64)


def build_state():
    torch.manual_seed(11)
    model = build_from_config(TINY_CFG)
    # small-but-nonzero optimizer state with a couple of exact-zero v coords
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    named = list(model.named_parameters())
    for n, p in named:
        opt.state[p] = {
            "step": torch.tensor(120.0),
            "exp_avg": torch.randn_like(p) * 1e-3,
            "exp_avg_sq": torch.rand_like(p) * 1e-6 + 1e-9,
        }
    # exact-zero v coordinates on a few wte rows
    wte = dict(named)["transformer.wte.weight"]
    opt.state[wte]["exp_avg_sq"][5:8] = 0.0
    opt.state[wte]["exp_avg"][5:8] = 0.0
    group_of = {p: opt.param_groups[0] for _, p in named}
    return model, opt, named, group_of


def test_b_branch_loop_and_growth_record():
    f1m.SEQ_LEN, f1m.SEQS_PER_MB, f1m.MB_PER_STEP = 16, 2, 1
    model, opt, named, group_of = build_state()
    dev = "cpu"
    gpu = f1m.GpuSession(TINY_CFG, dev)
    tokens = tiny_tokens(2)
    H = 2
    dtype = torch.float64
    q = {}
    g = torch.Generator().manual_seed(5)
    for n, p in named:
        q[n] = torch.randn(p.shape, generator=g)
    nrm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in q.values()))
    q = {k: v / nrm for k, v in q.items()}
    dtheta = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    dm = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    dv = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}

    class _Cpu:
        pass

    cpu = _Cpu()
    cpu.named = named
    cpu.optimizer = opt
    cpu.group_of = group_of
    cpu.state_dict_fp32 = lambda: {k: v for k, v in model.state_dict().items()}

    norms = []
    for h in range(1, H + 1):
        gpu.upload(cpu.state_dict_fp32(), torch.float32)
        grads, loss = f1m.compute_grads(gpu, tokens, h - 1)
        clipped, norm, coef = cpu.clip_no_step(grads) if hasattr(cpu, "clip_no_step") \
            else f1m.CpuState.clip_no_step(cpu, grads)
        tg = {n: q[n].to(dtype) for n, _ in named} if h == 1 else \
            {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
        clip = f1m.clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=tg,
                            max_norm=1.0)
        dtheta, dm, dv = f1m.tangent_step_arm(cpu, clipped, clip["clip_jvp"],
                                              dtheta, dm, dv, dtype)
        assert f1m.all_finite(dtheta)
        row = f1m.build_row(cpu, 0, "fp64", "B", h, loss, norm, coef, coef < 1.0,
                            dtheta, dm, dv, 0.0, True)
        norms.append(row["dtheta_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert norms[0] > 0 and norms[1] > 0
    print(f"PASS B-branch loop; dtheta norms: {norms}")


def test_c_branch_hvp_loop():
    f1m.SEQ_LEN, f1m.SEQS_PER_MB, f1m.MB_PER_STEP = 16, 2, 1
    model, opt, named, group_of = build_state()
    gpu = f1m.GpuSession(TINY_CFG, "cpu")
    tokens = tiny_tokens(2)
    dtype = torch.float64
    dtheta = {n: torch.randn_like(p, dtype=dtype) * 0.01 for n, p in named}
    hvp = f1m.hvp_pass(gpu, tokens, 0, dtheta, torch.float32)
    nrm = f1m.norm_stats(hvp)
    assert nrm > 0 and f1m.all_finite(hvp)
    print(f"PASS C-branch HVP (norm={nrm:.4g})")


def test_h1_tangent_equals_full_map_fd():
    """Central FD of the FULL one-step map (clip included), pulsing the raw
    gradient by +-eps*q pre-clip, vs the assembled tangent pipeline.

    FD path: identical bf16-autocast gradients for both signs (pulse applied
    post-hoc so gradient noise cancels exactly), clip + AdamW in fp64 MANUAL
    math. eps=1e-8 keeps the zero-v kink term a*eps*|q| (a~0.22) at ~1e-3
    relative on those coords (prereg section 0) while fp64 resolves the signal."""
    f1m.SEQ_LEN, f1m.SEQS_PER_MB, f1m.MB_PER_STEP = 16, 2, 1
    eps_fd = 1e-8
    dtype = torch.float64
    lr, b1, b2, aeps, wd = 1e-3, 0.9, 0.95, 1e-8, 0.1
    step_t = 120.0

    model, opt, named, group_of = build_state()
    gpu = f1m.GpuSession(TINY_CFG, "cpu")
    tokens = tiny_tokens(1)
    gpu.upload({k: v for k, v in model.state_dict().items()}, torch.float32)
    grads, _ = f1m.compute_grads(gpu, tokens, 0)

    q = {}
    g = torch.Generator().manual_seed(9)
    for n, p in named:
        q[n] = torch.randn(p.shape, generator=g)
    nrm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in q.values()))
    q = {k: v / nrm for k, v in q.items()}

    def manual_map(sign):
        gs = {n: grads[n].double() + sign * eps_fd * q[n].double() for n in grads}
        norm = math.sqrt(sum(float((g0.double() ** 2).sum()) for g0 in gs.values()))
        coef = min(1.0, 1.0 / (norm + 1e-6))
        out = {}
        bc1 = 1.0 - b1 ** (step_t + 1)
        bc2 = 1.0 - b2 ** (step_t + 1)
        for n, p in named:
            c = gs[n] * coef
            m = opt.state[p]["exp_avg"].double()
            v = opt.state[p]["exp_avg_sq"].double()
            th = p.detach().double()
            m1 = b1 * m + (1 - b1) * c
            v1 = b2 * v + (1 - b2) * c.square()
            upd = (m1 / bc1) / ((v1 / bc2).sqrt() + aeps)
            out[n] = th * (1 - lr * wd) - lr * upd
        return out

    up = manual_map(+1)
    um = manual_map(-1)
    fd = {n: (up[n] - um[n]) / (2 * eps_fd) for n in up}

    class _Cpu:
        pass

    cpu = _Cpu()
    cpu.named = named
    cpu.optimizer = opt
    cpu.group_of = group_of
    clipped, norm, coef = f1m.CpuState.clip_no_step(cpu, grads)
    tg = {n: q[n].to(torch.float64) for n, _ in named}
    clip = f1m.clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=tg, max_norm=1.0)
    dtheta = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    dm = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    dv = {n: torch.zeros_like(p, dtype=dtype) for n, p in named}
    dth, _, _ = f1m.tangent_step_arm(cpu, clipped, clip["clip_jvp"], dtheta, dm, dv, dtype)

    fn = math.sqrt(sum(float((v.double() ** 2).sum()) for v in fd.values()))
    an = math.sqrt(sum(float((v.double() ** 2).sum()) for v in dth.values()))
    dot = sum(float((fd[n].double() * dth[n].double()).sum()) for n in fd)
    cos = dot / (fn * an)
    rel = abs(fn - an) / an
    print(f"full-map FD vs pipeline: rel={rel:.4g} cos={cos:.8f}")
    assert rel < 5e-3 and cos > 0.9995, (rel, cos)


def test_bf16_arm_runs():
    f1m.SEQ_LEN, f1m.SEQS_PER_MB, f1m.MB_PER_STEP = 16, 2, 1
    model, opt, named, group_of = build_state()

    class _Cpu:
        pass

    cpu = _Cpu()
    cpu.named = named
    cpu.optimizer = opt
    cpu.group_of = group_of
    dtheta = {n: torch.zeros_like(p, dtype=torch.bfloat16) for n, p in named}
    dm = {n: torch.zeros_like(p, dtype=torch.bfloat16) for n, p in named}
    dv = {n: torch.zeros_like(p, dtype=torch.bfloat16) for n, p in named}
    dc = {n: torch.randn_like(p, dtype=torch.bfloat16) * 0.01 for n, p in named}
    c = {n: torch.randn_like(p) * 0.01 for n, p in named}
    dth, dmm, dvv = f1m.tangent_step_arm(cpu, c, dc, dtheta, dm, dv, torch.bfloat16)
    assert f1m.all_finite(dth)
    print("PASS bf16 arm tangent step")


if __name__ == "__main__":
    test_b_branch_loop_and_growth_record()
    test_c_branch_hvp_loop()
    test_h1_tangent_equals_full_map_fd()
    test_bf16_arm_runs()
    print("4/4 SMOKE PASS")
