"""F2.5 analysis-side tests: frozen metric rules from
docs/F2_5_TRUST_REGION_PREREG.md §2 (resolvable, usable both-arm sign rule,
central FD pairing, FD convergence, forward-untouched intervention)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from f25_build_outputs import (DELTA_GRID, EPS_ABS, R_THR, fd_convergence,
                               resolvable_flags, usable_flags)
from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step
from opsens.f25_trust_region_v01 import tangent_step_oracle


def _tab(rows_by_delta):
    """rows_by_delta: {delta: (Delta_plus, Delta_minus, D_H)} -> tab_h shape."""
    return {d: {"Delta_plus": dp, "Delta_minus": dm, "D_H": D}
            for d, (dp, dm, D) in rows_by_delta.items()}


def test_resolvable_eps_abs_both_arms():
    tab = _tab({d: (2e-6, 3e-6, 0.0) for d in DELTA_GRID})
    assert all(resolvable_flags(tab).values())
    tab = _tab({d: (2e-6, 5e-7, 0.0) for d in DELTA_GRID})  # minus arm under floor
    assert not any(resolvable_flags(tab).values())
    tab = _tab({d: (5e-7, 3e-6, 0.0) for d in DELTA_GRID})  # plus arm under floor
    assert not any(resolvable_flags(tab).values())


def test_resolvable_sign_stability_and_largest_autopass():
    # sign flips between adjacent deltas -> the last delta of the old run is
    # not resolvable (its successor disagrees); the first delta of the new run
    # is resolvable (stable toward larger deltas). Frozen rule looks FORWARD.
    rows = {1e-4: (1e-5, -1e-5, 0.0), 3e-4: (-1e-5, 1e-5, 0.0)}
    for d in DELTA_GRID[2:]:
        rows[d] = (-1e-5, 1e-5, 0.0)
    flags = resolvable_flags(_tab(rows))
    assert not flags[1e-4] and flags[3e-4]
    # largest delta self-compares -> auto-pass on sign even in a flipping grid
    assert flags[1e-1]


def test_usable_both_arms_sign_rule():
    # h*delta predicts positive plus-response and negative minus-response
    h = 2.0
    rows = {d: (h * d * 1.01, -h * d * 1.01, 0.0) for d in DELTA_GRID}  # R ~ 0.01
    u, rp, rm = usable_flags(_tab(rows), h)
    assert all(u.values())
    # flip the minus arm realized sign -> not usable
    rows[3e-3] = (h * 3e-3 * 1.01, +h * 3e-3 * 1.01, 0.0)
    u, _, _ = usable_flags(_tab(rows), h)
    assert not u[3e-3] and all(u[d] for d in DELTA_GRID if d != 3e-3)
    # R slightly above frozen 0.2 threshold -> not usable
    rows = {d: (h * d * 1.3, -h * d * 1.3, 0.0) for d in DELTA_GRID}  # R = 0.3
    u, _, _ = usable_flags(_tab(rows), h)
    assert not any(u.values())


def test_fd_pairing_and_convergence():
    # exact linear response L(delta) = L0 + h*delta (CRN-paired arms)
    h, L0 = -1.5, 6.0
    tab = _tab({d: (h * d, -h * d, (L0 + h * d - (L0 - h * d)) / (2 * d))
                for d in DELTA_GRID})
    resol = {d: True for d in DELTA_GRID}
    rows, hfd, nonconv, rich = fd_convergence(tab, resol)
    assert abs(hfd - h) < 1e-9
    assert not nonconv
    assert all(abs(r["adj_rel_diff"]) < 1e-9 for r in rows[1:])
    # drift > 20% within resolvable -> FD_NONCONVERGENT
    tab2 = _tab({d: (h * d, -h * d, h * (1 if d < 1e-2 else 3)) for d in DELTA_GRID})
    rows2, _, nonconv2, _ = fd_convergence(tab2, resol)
    assert nonconv2
    # unresolvable deltas do not break adjacency (only resolvable pairs count)
    resol3 = {d: d >= 1e-3 for d in DELTA_GRID}
    rows3, hfd3, _, _ = fd_convergence(tab, resol3)
    assert abs(hfd3 - h) < 1e-12
    assert rows3[0]["adj_rel_diff"] is None  # first resolvable has no predecessor


def test_fd_central_difference_formula():
    # D_H(delta) = [L(+delta) - L(-delta)] / (2 delta) with paired arms
    L0, slope = 5.0, 0.7
    for d in (1e-4, 1e-2, 1e-1):
        lp, lm = L0 + slope * d, L0 - slope * d
        D = (lp - lm) / (2 * d)
        assert abs(D - slope) < 1e-12


def test_intervention_leaves_forward_state_bitwise():
    torch.manual_seed(3)
    sh = (5, 4)
    p = torch.nn.Parameter(torch.randn(sh, dtype=torch.float64) * 0.1)
    state = {p: {"step": torch.tensor(26.0),
                 "exp_avg": torch.randn(sh, dtype=torch.float64) * 1e-3,
                 "exp_avg_sq": (torch.rand(sh, dtype=torch.float64) * 1e-6 + 1e-18)}}

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p], "lr": 3e-4, "betas": (0.9, 0.999),
                         "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False,
                         "maximize": False, "differentiable": False}]
    opt.state = state
    g = torch.Generator().manual_seed(9)
    c = torch.randn(sh, generator=g, dtype=torch.float64) * 1e-3
    dc = torch.randn(sh, generator=g, dtype=torch.float64) * 1e-3
    dth = torch.randn(sh, generator=g, dtype=torch.float64) * 1e-3
    dm = torch.randn(sh, generator=g, dtype=torch.float64) * 1e-3
    dv = torch.randn(sh, generator=g, dtype=torch.float64) * 1e-3
    before = {k: v.clone() for k, v in state[p].items()}
    mask = torch.zeros(sh, dtype=torch.float32)
    tangent_step_oracle([("a.weight", p)], opt, {"a.weight": c},
                        {"a.weight": dc}, {"a.weight": dth}, {"a.weight": dm},
                        {"a.weight": dv}, dv_mask_by_name={"a.weight": mask},
                        v_back_tau=1e-8)
    for k, v in state[p].items():
        assert torch.equal(v, before[k]), f"forward state mutated at {k}"
    # none-intervention remains bit-exact vs the vendored frozen tangent
    named = [("a.weight", p)]
    ref = adamw_tangent_step(named_parameters=named, optimizer=opt,
                             clipped_gradient_by_name={"a.weight": c},
                             tangent_clipped_gradient_by_name={"a.weight": dc},
                             tangent_theta_by_name={"a.weight": dth},
                             tangent_m_by_name={"a.weight": dm},
                             tangent_v_by_name={"a.weight": dv})
    got = tangent_step_oracle(named, opt, {"a.weight": c}, {"a.weight": dc},
                              {"a.weight": dth}, {"a.weight": dm}, {"a.weight": dv})
    for ch in ("tangent_theta", "tangent_m", "tangent_v"):
        assert torch.equal(ref[ch]["a.weight"], got[ch]["a.weight"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
