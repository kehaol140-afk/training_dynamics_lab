"""Optional observation hooks for M1/M2 (exploratory only, never primary evidence).

Usage (training loop, every N steps):
    hook = ObservationHook(out_path, every_n=100)
    ctx = hook.before_step(model)                    # snapshot theta, capture grads after backward
    ... backward ...
    hook.after_step(model, optimizers, ctx, grad_norm=..., probe_fn=...)

Writes one JSON line per logged step to outputs/observations/<name>.jsonl.
update_work_proxy = -g^T (theta_after - theta_before)  (proxy quantity; no other name permitted).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _flat_grads(params) -> torch.Tensor | None:
    gs = [p.grad.detach().reshape(-1).double() for p in params if p.grad is not None]
    if not gs:
        return None
    return torch.cat(gs)


def _flat_params(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).double() for p in params])


class ObservationHook:
    def __init__(self, out_path: str | Path, every_n: int = 100, name: str = "obs"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n = int(every_n)
        self.name = name
        self._file = None

    def should_log(self, step: int) -> bool:
        return step > 0 and step % self.every_n == 0

    def before_step(self, model: torch.nn.Module):
        params = [p for p in model.parameters()]
        return {"params": params, "theta_before": _flat_params(params).clone()}

    def after_step(
        self,
        model: torch.nn.Module,
        ctx: dict,
        step: int,
        grad_norm: float | None = None,
        sampled_sv: dict | None = None,
        domain_cosine: float | None = None,
        conflict_ratio: float | None = None,
        grad_flat: torch.Tensor | None = None,
        extra: dict | None = None,
    ) -> dict | None:
        params = ctx["params"]
        g_flat = grad_flat if grad_flat is not None else _flat_grads(params)
        theta_after = _flat_params(params)
        delta = theta_after - ctx["theta_before"]
        if g_flat is not None and g_flat.device != delta.device:
            g_flat = g_flat.to(delta.device)
        record = {
            "name": self.name,
            "step": int(step),
            "grad_global_norm": None if grad_norm is None else float(grad_norm),
            "sampled_gradient_singular_values": sampled_sv or {},
            "domain_gradient_cosine": None if domain_cosine is None else float(domain_cosine),
            "domain_conflict_ratio": None if conflict_ratio is None else float(conflict_ratio),
            "update_norm": float(delta.norm()),
            "weight_norm": float(theta_after.norm()),
            "update_weight_ratio": float(delta.norm() / theta_after.norm().clamp(min=1e-12)),
        }
        if g_flat is not None and delta.numel() == g_flat.numel():
            record["update_work_proxy"] = float(-(g_flat @ delta))
        else:
            record["update_work_proxy"] = None
        if extra:
            record.update(extra)
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def sampled_gradient_singular_values(
    model: torch.nn.Module, name_filter=None, top_k: int = 5
) -> dict:
    """Top-k singular values of gradient matrices for Muon-route-shaped (2D) weights.

    name_filter: optional predicate on parameter name to subsample matrices (cheap logging).
    """
    out = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.grad is not None and (name_filter is None or name_filter(name)):
            if p.shape[0] > 64 and p.shape[1] > 64:
                sv = torch.linalg.svdvals(p.grad.detach().float())
                out[name] = [float(x) for x in sv[:top_k]]
    return out
