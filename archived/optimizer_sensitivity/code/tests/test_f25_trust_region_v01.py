"""F2.5 library tests: no-intervention bit-exactness vs the vendored frozen
tangent; intervention semantics; trust-radius scan."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step
from opsens.f25_trust_region_v01 import tangent_step_oracle, trust_radius


def _make_state(seed=5):
    torch.manual_seed(seed)
    shapes = [(9, 7), (7, 9), (63,)]
    names = ["a.weight", "b.weight", "c.bias"]
    named, state, groups = [], {}, []
    for n, sh in zip(names, shapes):
        p = torch.nn.Parameter(torch.randn(sh, dtype=torch.float64) * 0.1)
        named.append((n, p))
        state[p] = {"step": torch.tensor(26.0),
                    "exp_avg": torch.randn(sh, dtype=torch.float64) * 1e-3,
                    "exp_avg_sq": torch.rand(sh, dtype=torch.float64) * 1e-6 + 1e-12}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                         "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
                         "amsgrad": False, "maximize": False, "differentiable": False}]
    opt.state = state
    return named, opt


def _inputs(named, seed=11):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for kind in ("c", "dc", "dth", "dm", "dv"):
        out[kind] = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
                     for n, p in named}
    return out


def test_no_intervention_bit_exact():
    named, opt = _make_state()
    inp = _inputs(named)
    ref = adamw_tangent_step(
        named_parameters=named, optimizer=opt,
        clipped_gradient_by_name=inp["c"],
        tangent_clipped_gradient_by_name=inp["dc"],
        tangent_theta_by_name=inp["dth"],
        tangent_m_by_name=inp["dm"],
        tangent_v_by_name=inp["dv"])
    got = tangent_step_oracle(named, opt, inp["c"], inp["dc"], inp["dth"], inp["dm"], inp["dv"])
    for chan in ("tangent_theta", "tangent_m", "tangent_v"):
        for n in ref[chan]:
            assert torch.equal(ref[chan][n], got[chan][n]), (chan, n)


def test_mask_kills_only_v_channel():
    named, opt = _make_state()
    inp = _inputs(named)
    # dc=0, dm=0, dth=0, dv != 0 -> only the v->theta' term is active
    dc0 = {n: torch.zeros_like(p) for n, p in named}
    dm0 = {n: torch.zeros_like(p) for n, p in named}
    dth0 = {n: torch.zeros_like(p) for n, p in named}
    full = tangent_step_oracle(named, opt, inp["c"], dc0, dth0, dm0, inp["dv"])
    masks = {n: torch.zeros_like(p) for n, p in named}
    masked = tangent_step_oracle(named, opt, inp["c"], dc0, dth0, dm0, inp["dv"],
                                 dv_mask_by_name=masks)
    for n, p in named:
        assert not torch.equal(full["tangent_theta"][n], masked["tangent_theta"][n])
        assert torch.allclose(masked["tangent_theta"][n], torch.zeros_like(p), atol=0.0)
        # m/v recursions untouched by the mask
        assert torch.equal(full["tangent_m"][n], masked["tangent_m"][n])
        assert torch.equal(full["tangent_v"][n], masked["tangent_v"][n])
    # all-ones mask == no mask
    ones = {n: torch.ones_like(p) for n, p in named}
    keep = tangent_step_oracle(named, opt, inp["c"], dc0, dth0, dm0, inp["dv"],
                               dv_mask_by_name=ones)
    for n, p in named:
        assert torch.equal(full["tangent_theta"][n], keep["tangent_theta"][n])


def test_vfloor_changes_denominators_only():
    # v must actually contain coordinates below tau^2 for the clamp to bite;
    # handcraft a state with an extreme floor coordinate instead of _make_state.
    torch.manual_seed(7)
    shapes = [(4,)]
    names = ["a.weight"]
    named, state = [], {}
    v = torch.tensor([1e-20, 1e-12, 1e-6, 1e-2], dtype=torch.float64)
    for n, sh in zip(names, shapes):
        p = torch.nn.Parameter(torch.randn(sh, dtype=torch.float64) * 0.1)
        named.append((n, p))
        state[p] = {"step": torch.tensor(26.0),
                    "exp_avg": torch.randn(sh, dtype=torch.float64) * 1e-3,
                    "exp_avg_sq": v.clone()}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                         "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
                         "amsgrad": False, "maximize": False, "differentiable": False}]
    opt.state = state
    inp = _inputs(named)
    base = tangent_step_oracle(named, opt, inp["c"], inp["dc"], inp["dth"], inp["dm"], inp["dv"])
    flo = tangent_step_oracle(named, opt, inp["c"], inp["dc"], inp["dth"], inp["dm"], inp["dv"],
                              v_back_tau=1e-8)
    n0 = "a.weight"
    assert not torch.equal(base["tangent_theta"][n0], flo["tangent_theta"][n0])
    # tau -> 0 recovers the base exactly
    tiny = tangent_step_oracle(named, opt, inp["c"], inp["dc"], inp["dth"], inp["dm"], inp["dv"],
                               v_back_tau=0.0)
    for n, p in named:
        assert torch.equal(base["tangent_theta"][n], tiny["tangent_theta"][n])


def test_trust_radius_scan():
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    # all usable+resolvable -> grid max
    assert trust_radius(grid, [True] * 8, [True] * 8) == 1e-1
    # collapse: usable only at the two smallest
    u = [True, True, False, False, False, False, False, False]
    assert trust_radius(grid, u, [True] * 8) == 3e-4
    # unresolvable deltas are skipped (do not cap)
    r = [True, False, True, True, True, True, True, True]
    u = [True, True, True, True, True, True, True, True]
    assert trust_radius(grid, u, r) == 1e-1
    # nothing usable
    assert trust_radius(grid, [False] * 8, [True] * 8) == 0.0
    # later usable cannot rescue an earlier unusable resolvable delta
    u = [True, True, False, True, True, True, True, True]
    assert trust_radius(grid, u, [True] * 8) == 3e-4


def test_rden_masks_are_parameter_shaped_and_usable_in_oracle():
    """Regression: rden_topk_masks must return masks reshaped to each parameter's
    own shape (the 2026-09-26 crash was a flat 1-D mask multiplied against a
    2-D v_channel). Also verifies the top-k rule selects the smallest-v
    coordinates and that a masked oracle run leaves the forward state bitwise
    unchanged."""
    import opsens.f0_bank_audit_v01 as f0
    import opsens.f25_trust_region_v01 as f25

    state = object()  # opaque token; fakes below never read it
    v_a = torch.full((4, 5), 1e-4)
    v_b = torch.full((3, 2), 1e-2)
    v_b[2, 1] = 1e-12  # single smallest-v coordinate -> top R_den
    slots = {"a.weight": {"exp_avg_sq": v_a, "step": torch.tensor(26.0)},
             "b.weight": {"exp_avg_sq": v_b, "step": torch.tensor(26.0)}}

    orig_gp, orig_ne, orig_da = f0.group_hyperparams, f0.named_optimizer_entries, f0.denominator_anchor
    try:
        f0.group_hyperparams = lambda s: {"beta2": 0.999, "eps": 1e-8}
        f0.named_optimizer_entries = lambda s: iter(
            [(n, i, slots[n]) for i, n in enumerate(slots)])
        f0.denominator_anchor = lambda v, step, beta2, eps: torch.sqrt(v) + eps
        masks, meta = f25.rden_topk_masks(state, 1.0 / 26.0)  # k = 1 of 26
    finally:
        f0.group_hyperparams, f0.named_optimizer_entries, f0.denominator_anchor = (
            orig_gp, orig_ne, orig_da)

    assert meta["k"] == 1 and meta["total"] == 26
    assert masks["a.weight"].shape == v_a.shape
    assert masks["b.weight"].shape == v_b.shape
    # exactly the smallest-v coordinate knocked out (mask 0), all else kept
    assert float(masks["b.weight"][2, 1]) == 0.0
    assert float(masks["b.weight"].sum()) == 5.0
    assert float(masks["a.weight"].sum()) == 20.0

    # the exact oracle call path that crashed on 2026-09-26
    torch.manual_seed(11)
    named = [(n, torch.nn.Parameter(torch.randn(*slots[n]["exp_avg_sq"].shape) * 0.1))
             for n in slots]
    opt_state = {p: {"step": slots[n]["step"].clone(),
                     "exp_avg": torch.randn_like(p) * 1e-3,
                     "exp_avg_sq": slots[n]["exp_avg_sq"].clone()}
                 for n, p in named}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                         "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
                         "amsgrad": False, "maximize": False}]
    opt.state = opt_state
    g = torch.Generator().manual_seed(4)
    zm = {n: torch.randn(p.shape, generator=g) * 1e-3 for n, p in named}
    before = {p: {k: t.clone() for k, t in s.items()} for p, s in opt_state.items()}
    out = f25.tangent_step_oracle(named, opt, zm, zm, zm, zm, zm,
                                  dv_mask_by_name=masks)
    for p, s in opt_state.items():
        for k, t in s.items():
            assert torch.equal(t, before[p][k]), f"forward state mutated at {k}"
    assert all(bool(torch.isfinite(t).all()) for t in out["tangent_theta"].values())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
