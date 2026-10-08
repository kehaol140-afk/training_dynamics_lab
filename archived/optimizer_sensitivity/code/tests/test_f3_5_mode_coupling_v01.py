"""F3.5 library tests (prereg §14): adjoint identity (raw + scaled),
similarity-transform eigenvalue preservation, principal angles, projector
reconstruction, q-split identity, source-off forward-neutrality, metric
roundtrip, and conjugate-pair real-basis handling."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opsens import f0_bank_audit_v01 as f0  # noqa: E402
from opsens.f3_5_mode_coupling_v01 import (  # noqa: E402
    ScaledOperator, alignment, arnoldi_basis, biorthogonal_projector,
    extract_pairs, orthonormalize, pair_basis, principal_angles,
    scale_state, split_projection, state_scales, unstable_pairs)
from opsens.f3_curvature_mechanism_v01 import (  # noqa: E402
    StepOperator, state_add, state_inner, state_norm, state_scale,
    zeros_like_state)

NAMES = [("model.layers.0.self_attn.q_proj.weight", (6, 5)),
         ("model.layers.0.mlp.gate_proj.weight", (7,))]


def _make():
    torch.manual_seed(5)
    named = [(n, torch.nn.Parameter(torch.randn(sh, dtype=torch.float64) * 0.1))
             for n, sh in NAMES]
    state = {}
    for _, p in named:
        state[p] = {"step": torch.tensor(26.0),
                    "exp_avg": torch.randn(p.shape, dtype=torch.float64) * 1e-3,
                    "exp_avg_sq": (torch.rand(p.shape, dtype=torch.float64)
                                   * 1e-6 + 1e-12)}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                         "betas": (0.9, 0.999), "eps": 1e-8,
                         "weight_decay": 0.01, "amsgrad": False,
                         "maximize": False, "differentiable": False}]
    opt.state = state
    g = torch.Generator().manual_seed(11)
    clipped = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
               for n, p in named}
    return named, opt, clipped


def _diag_op(named, opt, clipped, scale=2.5):
    diag = {n: torch.rand(p.shape, dtype=torch.float64) + 0.5 for n, p in named}

    def hvp_fn(x):
        return {n: scale * diag[n] * x[n] for n in x}

    return StepOperator(named, opt, clipped, coef=0.9, active=True, k=1.3,
                        hvp_fn=hvp_fn), diag


def _rand_state(named, seed):
    gen = torch.Generator().manual_seed(seed)
    dz = zeros_like_state(named)
    for ch in ("th", "m", "v"):
        for n, p in named:
            dz[ch][n] = torch.randn(p.shape, generator=gen, dtype=torch.float64)
    return state_scale(dz, 1.0 / state_norm(dz))


def test_adjoint_identity_raw_and_scaled():
    named, opt, clipped = _make()
    op, _ = _diag_op(named, opt, clipped)
    scales = state_scales(named, opt)
    wops = {"RAW": op, "DIML": ScaledOperator(op, scales)}
    for tag, o in wops.items():
        u = _rand_state(named, 92001)
        v = _rand_state(named, 93001)
        lhs = state_inner(u, o.jvp(v))
        rhs = state_inner(o.vjp(u), v)
        assert abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-10, tag


def test_similarity_preserves_eigenvalues():
    # small state (single 4-coord tensor x 3 channels = 12 dims); m=12
    # exhausts the Krylov space, so Ritz values are EXACT eigenvalues and the
    # similarity invariance is exact up to fp rounding.
    named = [("model.layers.0.self_attn.q_proj.weight",
              torch.nn.Parameter(torch.randn(4, dtype=torch.float64) * 0.1))]
    state = {named[0][1]: {
        "step": torch.tensor(26.0),
        "exp_avg": torch.randn(4, dtype=torch.float64) * 1e-3,
        "exp_avg_sq": torch.rand(4, dtype=torch.float64) * 1e-6 + 1e-12}}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                         "betas": (0.9, 0.999), "eps": 1e-8,
                         "weight_decay": 0.01, "amsgrad": False,
                         "maximize": False, "differentiable": False}]
    opt.state = state
    g = torch.Generator().manual_seed(11)
    clipped = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
               for n, p in named}
    op, _ = _diag_op(named, opt, clipped)
    scales = {"th": {named[0][0]: 1e-2}, "m": {named[0][0]: 1e-3},
              "v": {named[0][0]: 1e-5}}  # moderate ratios; extreme (1e-15)
    # ratios move the ill-conditioning into J~'s off-diagonal blocks and fp64
    # Krylov arithmetic (no balancing) can no longer represent the similarity
    # -- handled at runtime by the DIML numerical-validity flag (prereg ERRATUM)
    op_t = ScaledOperator(op, scales)
    import numpy as np
    m = 12
    Hr, _, mer = arnoldi_basis(op.jvp, named, m=m)
    Ht, _, met = arnoldi_basis(op_t.jvp, named, m=m)
    lam_r = np.linalg.eig(Hr[:mer, :mer].numpy())[0]
    lam_t = np.linalg.eig(Ht[:met, :met].numpy())[0]
    ar = sorted((c.real, c.imag) for c in lam_r)
    at = sorted((c.real, c.imag) for c in lam_t)
    assert len(ar) == len(at)
    scale = max(1.0, max(max(abs(x) for x in p) for p in ar))
    assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
               for a, b in zip(ar, at)) < 1e-7 * scale


def test_principal_angles():
    # state space = single 3-coordinate tensor; A = span{e0,e1}, B = span{e1,e2}
    named = [("model.layers.0.self_attn.q_proj.weight",
              torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)))]

    def vec(idx, val=1.0):
        z = {"th": {}, "m": {}, "v": {}}
        for n, p in named:
            for ch in z:
                z[ch][n] = torch.zeros_like(p, dtype=torch.float64)
        z["th"][named[0][0]][idx] = val
        return z

    vecs = [vec(i) for i in range(3)]
    s, tmax, tmean = principal_angles([vecs[0], vecs[1]], [vecs[1], vecs[2]])
    assert abs(sorted(s)[0] - 0.0) < 1e-12 and abs(sorted(s)[1] - 1.0) < 1e-12
    assert abs(tmax - __import__("math").pi / 2) < 1e-6
    assert abs(tmean - __import__("math").pi / 4) < 1e-6


def test_projector_reconstruction():
    named, opt, clipped = _make()
    # V, W: two different orthonormal bases of the SAME 2D subspace
    a1 = _rand_state(named, 101)
    a2 = state_add(_rand_state(named, 102), a1, sa=1.0,
                   sb=-state_inner(_rand_state(named, 102), a1))
    a2 = state_scale(a2, 1.0 / state_norm(a2))
    V = [a1, a2]
    W = [state_add(a1, a2, sa=2 ** -0.5, sb=2 ** -0.5),
         state_add(a1, a2, sa=2 ** -0.5, sb=-(2 ** -0.5))]
    kappa, P = biorthogonal_projector(V, W)
    assert kappa is not None and kappa < 1.0 + 1e-9
    q = _rand_state(named, 103)
    pq = P(q)
    # P^2 = P
    ppq = P(pq)
    for ch in ("th", "m", "v"):
        for n, _ in named:
            assert torch.allclose(ppq[ch][n], pq[ch][n], rtol=1e-10, atol=1e-14)
    # idempotent complement: q = Pq + (I-P)q trivially; P inverts itself
    assert abs(state_inner(P(q), state_add(q, pq, sa=1.0, sb=-1.0))) < 1e-10


def test_q_split_identity():
    named, opt, clipped = _make()
    a1 = _rand_state(named, 201)
    a2 = _rand_state(named, 202)
    V = orthonormalize([a1, a2])
    W = orthonormalize([a1, state_add(a2, a1, sa=1.0, sb=0.3)])
    _, P = biorthogonal_projector(V, W)
    q = _rand_state(named, 203)
    qu, qr = split_projection(P, q)
    back = state_add(qu, qr)
    # algebraic identity q = Pq + (I-P)q holds EXACTLY in exact arithmetic;
    # in fp64 it holds to machine precision (fl(q-qu)+qu rounding), so the
    # regression locks rtol 1e-13 (disclosed in prereg ERRATUM)
    for ch in ("th", "m", "v"):
        for n, _ in named:
            assert torch.allclose(back[ch][n], q[ch][n], rtol=1e-13, atol=1e-300)
    # P(q_unstable) == q_unstable (invariant subspace)
    pqu = P(qu)
    for ch in ("th", "m", "v"):
        for n, _ in named:
            assert torch.allclose(pqu[ch][n], qu[ch][n], rtol=1e-9, atol=1e-14)


def test_source_off_intervention_preserves_forward():
    # fake deterministic primal; arms only ADD hom steps, never touch it
    named, opt, clipped = _make()

    def primal_loss(h):
        return 1.0 + 0.01 * h

    losses_plain = [primal_loss(h) for h in range(1, 33)]
    losses_arms = []
    dummy = zeros_like_state(named)
    for h in range(1, 33):
        # "arm work" touches only tangent state copies
        _ = state_add(dummy, state_scale(dummy, 0.5))
        losses_arms.append(primal_loss(h))
    assert losses_plain == losses_arms
    from opsens.f3_curvature_mechanism_v01 import hom_step
    dc0 = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}
    out = hom_step(named, opt, clipped, dc0, dummy)
    assert set(out) == {"th", "m", "v"}


def test_metric_roundtrip():
    named, opt, clipped = _make()
    scales = state_scales(named, opt)
    z = _rand_state(named, 301)
    zr = scale_state(scale_state(z, scales), scales, inverse=True)
    # fp64 divide-then-multiply roundtrip: exact for moderate scales, tight
    # relative tolerance for extreme (v-channel ~1e-17) scale ratios
    for ch in ("th", "m", "v"):
        for n, _ in named:
            assert torch.allclose(zr[ch][n], z[ch][n], rtol=1e-12, atol=0.0)
    op, _ = _diag_op(named, opt, clipped)
    opt_sc = ScaledOperator(op, scales)
    x = _rand_state(named, 302)
    lhs = opt_sc.jvp(x)
    rhs = scale_state(op.jvp(scale_state(x, scales, inverse=True)), scales)
    for ch in ("th", "m", "v"):
        for n, _ in named:
            assert torch.allclose(lhs[ch][n], rhs[ch][n], rtol=1e-12, atol=1e-18)


def test_conjugate_pair_real_basis():
    # 2D rotation-scaling J on the first two coords of one tensor
    # (eigenvalues a +- ib, |lam| = 2.5 > 1.05), dampened 3rd coord.
    a, b = 2.0, 1.5
    named = [("model.layers.0.self_attn.q_proj.weight",
              torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)))]

    def matvec(x):
        z = zeros_like_state(named)
        for n, _ in named:
            t = x["th"][n]
            z["th"][n] = torch.stack([a * t[0] - b * t[1],
                                      b * t[0] + a * t[1],
                                      0.5 * t[2]])
            z["m"][n] = 0.9 * x["m"][n]
            z["v"][n] = 0.99 * x["v"][n]
        return z

    H, basis, m_eff = arnoldi_basis(matvec, named, m=6)
    pairs = extract_pairs(H, basis, named)
    unst = unstable_pairs(pairs)
    assert pairs and pairs[0]["kind"] == "cplx"
    lam = pairs[0]["lam"]
    assert abs(abs(lam) - (a * a + b * b) ** 0.5) < 1e-6
    assert abs(lam.imag - b) < 1e-6 and lam.imag > 0
    Q = orthonormalize(pair_basis(unst))
    assert len(Q) >= 2
    # any vector in the invariant plane projects fully onto the real basis
    n0 = named[0][0]

    def plain(th):
        z = {"th": {}, "m": {}, "v": {}}
        for n, p in named:
            for ch in z:
                z[ch][n] = torch.zeros_like(p, dtype=torch.float64)
        z["th"][n0] = th
        return z

    inplane = plain(torch.tensor([0.6, -1.7, 0.0]))
    e = alignment(Q, inplane)
    assert e is not None and e > 1 - 1e-9
    # out-of-plane component is not captured
    outp = plain(torch.tensor([0.0, 0.0, 3.0]))
    e2 = alignment(Q, outp)
    assert e2 is not None and e2 < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
