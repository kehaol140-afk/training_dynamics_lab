import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.model import build_model
from muon_transition.optimizers import build_adamw_arm, build_muon_arm
from muon_transition.routing import route_parameters


def test_adamw_aux_matches_formula():
    torch.manual_seed(0)
    p = torch.randn(16, 8)
    g = torch.randn(16, 8)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 0.1
    opt = torch.optim.AdamW([p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
    p0 = p.clone()
    p.grad = g.clone()
    opt.step()

    m = (1 - beta1) * g
    v = (1 - beta2) * g * g
    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)
    manual = (p0 - lr * wd * p0) - lr * m_hat / (v_hat.sqrt() + eps)
    assert torch.allclose(p, manual, atol=1e-9), (p - manual).abs().max()
    print("test_adamw_aux_matches_formula OK")


def test_arm_construction_and_frozen_hparams():
    model = build_model(seed=1)
    muon_opt, aux_opt = build_muon_arm(model, lr=0.02, weight_decay=0.1)

    _, aux_route, _ = route_parameters(model)
    routed_ids = {id(p) for p in muon_opt.param_groups[0]["params"]}
    aux_ids_opt = {id(p) for p in aux_opt.param_groups[0]["params"]}
    aux_ids_route = {id(p) for p in aux_route}
    assert routed_ids.isdisjoint(aux_ids_opt)
    assert aux_ids_opt == aux_ids_route

    g = aux_opt.param_groups[0]
    assert g["betas"] == config.ADAMW_BETAS
    assert g["eps"] == config.ADAMW_EPS
    assert g["weight_decay"] == 0.1

    mg = muon_opt.param_groups[0]
    assert mg["momentum"] == config.MUON_FREEZE["momentum"]
    assert mg["nesterov"] is True
    assert mg["ns_steps"] == 5
    assert tuple(mg["ns_coefficients"]) == (3.4445, -4.7750, 2.0315)

    model2 = build_model(seed=1)
    adamw = build_adamw_arm(model2, lr=3e-4, weight_decay=0.1)
    n_adamw_params = sum(p.numel() for p in adamw.param_groups[0]["params"])
    _, aux2, _ = route_parameters(model2)
    from muon_transition.model import unique_named_parameters
    assert n_adamw_params == sum(p.numel() for _, p in unique_named_parameters(model2))
    print("test_arm_construction_and_frozen_hparams OK")


if __name__ == "__main__":
    test_adamw_aux_matches_formula()
    test_arm_construction_and_frozen_hparams()
