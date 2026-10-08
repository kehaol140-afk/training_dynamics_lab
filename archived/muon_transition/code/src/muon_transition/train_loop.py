from __future__ import annotations

import torch
import torch.nn.functional as F

from .optimizers import step_all, zero_all


def batch_token_loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Mean CE over valid next-token targets of a fully packed window batch."""
    res = model(batch)
    logits = res.logits[..., :-1, :].contiguous()
    targets = batch[..., 1:].contiguous()
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def train_step(
    model: torch.nn.Module,
    optimizers: list,
    batch: torch.Tensor,
    grad_clip: float = 1.0,
) -> dict:
    """One micro-step: zero_grad -> forward -> backward -> global-norm clip -> step all.
    Shared verbatim by both arms; the optimizer list is the only arm-dependent part.
    """
    zero_all(optimizers)
    loss = batch_token_loss(model, batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None], grad_clip
    )
    step_all(optimizers)
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}
