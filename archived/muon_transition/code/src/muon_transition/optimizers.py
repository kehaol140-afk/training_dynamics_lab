from __future__ import annotations

import torch

from . import config
from .model import unique_named_parameters
from .routing import route_parameters


def build_adamw_arm(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    params = [p for _, p in unique_named_parameters(model)]
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=config.ADAMW_BETAS,
        eps=config.ADAMW_EPS,
        weight_decay=weight_decay,
    )


def build_muon_arm(model: torch.nn.Module, lr: float, weight_decay: float):
    """Return (muon_opt, adamw_aux_opt) for the Muon arm."""
    from torch.optim import Muon

    muon_params, aux_params, _ = route_parameters(model)
    m = config.MUON_FREEZE
    muon_opt = Muon(
        muon_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=m["momentum"],
        nesterov=m["nesterov"],
        ns_steps=m["ns_steps"],
        ns_coefficients=tuple(m["ns_coefficients"]),
        eps=m["eps"],
        adjust_lr_fn=m["adjust_lr_fn"],
    )
    aux_opt = torch.optim.AdamW(
        aux_params,
        lr=lr,
        betas=config.ADAMW_BETAS,
        eps=config.ADAMW_EPS,
        weight_decay=weight_decay,
    )
    return muon_opt, aux_opt


def zero_all(optimizers) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def step_all(optimizers) -> None:
    for opt in optimizers:
        opt.step()
