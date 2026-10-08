from __future__ import annotations

import torch

from . import config


def domain_nll_sum_count(
    model: torch.nn.Module,
    windows: torch.Tensor,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """Token NLL as sum/count over a fixed window pool. Windows are fully packed
    (no padding): every window contributes seq_len - 1 valid next-token targets.
    """
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    nll_sum = torch.zeros((), dtype=torch.float64)
    token_count = 0
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch_size):
            batch = windows[i : i + batch_size].to(device)
            res = model(batch)
            logits = res.logits[..., :-1, :].contiguous()
            targets = batch[..., 1:].contiguous()
            losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            nll_sum += losses.double().cpu()
            token_count += int(targets.numel())
    if was_training:
        model.train()
    if token_count == 0:
        raise ValueError("empty evaluation pool")
    return {"nll_sum": float(nll_sum), "token_count": token_count}


def evaluate_q(
    model: torch.nn.Module,
    pools: dict,
    q: float = 0.5,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """pools: {"A": windows_tensor, "B": windows_tensor}. Returns per-domain NLL
    (sum/count) and weighted V_Q = q * NLL_A + (1 - q) * NLL_B. q is frozen at 0.5.
    """
    out = {}
    for dom in ("A", "B"):
        sc = domain_nll_sum_count(model, pools[dom], batch_size=batch_size, device=device)
        out[f"nll_{dom}"] = sc["nll_sum"] / sc["token_count"]
        out[f"tokens_{dom}"] = sc["token_count"]
    out["v_q"] = q * out["nll_A"] + (1.0 - q) * out["nll_B"]
    out["q"] = q
    return out
