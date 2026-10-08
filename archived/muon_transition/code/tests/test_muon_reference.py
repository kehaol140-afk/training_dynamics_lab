import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.reference_muon import muon_reference_step, newton_schulz_reference


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def test_muon_matches_reference():
    from torch.optim import Muon

    m = config.MUON_FREEZE
    torch.manual_seed(7)
    shapes = [(512, 512), (1664, 512), (512, 1664), (512, 256), (256, 512)]
    lerp_cosines, documented_cosines, rel_errs = [], [], []
    for shape in shapes:
        param = torch.randn(*shape) * 0.02
        grad = torch.randn(*shape)
        param0 = param.clone()

        opt = Muon(
            [param],
            lr=0.02,
            weight_decay=m["weight_decay_default"],
            momentum=m["momentum"],
            nesterov=m["nesterov"],
            ns_steps=m["ns_steps"],
            ns_coefficients=tuple(m["ns_coefficients"]),
            eps=m["eps"],
            adjust_lr_fn=m["adjust_lr_fn"],
        )
        param.grad = grad.clone()
        opt.step()

        state = {}
        ref_lerp = muon_reference_step(
            param0, grad, state, lr=0.02, momentum=m["momentum"], nesterov=m["nesterov"],
            weight_decay=m["weight_decay_default"], ns_steps=m["ns_steps"],
            adjust_lr_fn=m["adjust_lr_fn"], variant="lerp_code",
        )
        state = {}
        ref_doc = muon_reference_step(
            param0, grad, state, lr=0.02, momentum=m["momentum"], nesterov=m["nesterov"],
            weight_decay=m["weight_decay_default"], ns_steps=m["ns_steps"],
            adjust_lr_fn=m["adjust_lr_fn"], variant="documented",
        )

        delta_impl = param - param0
        delta_lerp = ref_lerp - param0.double()
        delta_doc = ref_doc - param0.double()
        lerp_cosines.append(_cos(delta_impl, delta_lerp))
        documented_cosines.append(_cos(delta_impl, delta_doc))
        rel_errs.append(float((delta_impl.double() - delta_lerp).norm() / delta_lerp.norm()))

    assert min(lerp_cosines) > 0.999, lerp_cosines
    assert max(rel_errs) < 0.05, rel_errs
    assert min(documented_cosines) > 0.99, documented_cosines
    print("test_muon_matches_reference OK",
          {"lerp_cos_min": min(lerp_cosines), "rel_err_max": max(rel_errs),
           "documented_cos_min": min(documented_cosines)})


def test_muon_state_and_groups():
    from torch.optim import Muon

    m = config.MUON_FREEZE
    p = torch.randn(64, 32, requires_grad=True)
    opt = Muon([p], lr=0.02, momentum=m["momentum"], nesterov=m["nesterov"],
               ns_steps=m["ns_steps"], ns_coefficients=tuple(m["ns_coefficients"]),
               eps=m["eps"], adjust_lr_fn=m["adjust_lr_fn"],
               weight_decay=m["weight_decay_default"])
    p.grad = torch.randn(64, 32)
    opt.step()
    sd = opt.state_dict()
    key = next(iter(sd["state"]))
    assert key in (0, "0")
    assert set(sd["state"][key].keys()) == {"momentum_buffer"}
    buf = sd["state"][key]["momentum_buffer"]
    expected = m["momentum"] * 0 + (1 - m["momentum"]) * p.grad
    assert torch.allclose(buf, expected.to(buf.dtype), atol=1e-6)
    g = sd["param_groups"][0]
    assert g["momentum"] == 0.95 and g["nesterov"] is True and g["ns_steps"] == 5
    assert tuple(g["ns_coefficients"]) == (3.4445, -4.7750, 2.0315)
    assert g["eps"] == 1e-7
    print("test_muon_state_and_groups OK")


def test_muon_rejects_non_2d():
    from torch.optim import Muon

    v = torch.randn(8, requires_grad=True)
    try:
        Muon([v], lr=0.02)
    except ValueError:
        print("test_muon_rejects_non_2d OK")
        return
    raise AssertionError("Muon accepted a 1D parameter")


def test_newton_schulz_scale_invariance():
    G = torch.randn(128, 96)
    o1 = newton_schulz_reference(G)
    o2 = newton_schulz_reference(G * 37.5)
    assert _cos(o1, o2) > 0.9999
    print("test_newton_schulz_scale_invariance OK", _cos(o1, o2))


if __name__ == "__main__":
    test_muon_matches_reference()
    test_muon_state_and_groups()
    test_muon_rejects_non_2d()
    test_newton_schulz_scale_invariance()
