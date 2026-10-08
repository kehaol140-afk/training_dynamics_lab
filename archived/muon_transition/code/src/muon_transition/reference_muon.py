"""Independent reference implementation of the frozen Muon semantics (test oracle only).

Written from the torch.optim._muon documentation math (docstring) plus a second variant
mirroring the exact lerp code path, both in float64. Used by tests/test_muon_reference.py.
Never used in training.
"""
from __future__ import annotations

import math

import torch


def newton_schulz_reference(
    G: torch.Tensor,
    steps: int = 5,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = 1e-7,
) -> torch.Tensor:
    X = G.detach().double().clone()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    n = X.norm().clamp(min=eps)
    X = X / n
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if transposed:
        X = X.T
    return X


def adjust_lr_reference(lr: float, shape, mode: str | None) -> float:
    A, B = int(shape[0]), int(shape[1])
    if mode is None or mode == "original":
        ratio = math.sqrt(max(1.0, A / B))
    elif mode == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(A, B))
    else:
        ratio = 1.0
    return lr * ratio


def muon_reference_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    lr: float,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    adjust_lr_fn: str | None = None,
    variant: str = "documented",
) -> torch.Tensor:
    """Return the updated parameter. state is mutated with the new momentum buffer.

    variant="documented": B <- mu*B + g ; u = g + mu*B (nesterov)
    variant="lerp_code" : B <- mu*B + (1-mu)*g ; u = (1-mu)*g + mu*B (nesterov)
    Both differ only by a positive scalar multiple before Newton-Schulz, which is
    scale-invariant up to the eps clamp.
    """
    buf = state.get("momentum_buffer")
    if buf is None:
        buf = torch.zeros_like(grad, dtype=torch.float64)
    buf = momentum * buf + (1.0 - momentum if variant == "lerp_code" else 1.0) * grad.double()
    state["momentum_buffer"] = buf
    if nesterov:
        if variant == "lerp_code":
            u = (1.0 - momentum) * grad.double() + momentum * buf
        else:
            u = grad.double() + momentum * buf
    else:
        u = buf
    O = newton_schulz_reference(u, steps=ns_steps)
    adjusted = adjust_lr_reference(lr, param.shape, adjust_lr_fn)
    new_param = param.double() * (1.0 - lr * weight_decay)
    new_param = new_param - adjusted * O
    return new_param
