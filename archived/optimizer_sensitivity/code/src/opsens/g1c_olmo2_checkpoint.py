"""Checkpoint restoration for FR_G1C_1B_FEASIBILITY_GATE_V01 (Phase 0 gate).

Restores the official OLMo-2-0425-1B unsharded checkpoint (model.pt /
optim.pt / train.pt) into the G1C architecture module and a real
torch.optim.AdamW whose state is the checkpoint's true m/v/step.

Mapping rule (fail-fast, CONTRACT_INCOMPATIBLE on any mismatch):
  - model.pt keys must exactly equal the architecture state_dict keys.
  - optim.pt param_groups carry ``param_names`` (FSDP-prefixed); the prefix
    ``_fsdp_wrapped_module.`` is stripped and each name must equal a model.pt
    key. State (exp_avg / exp_avg_sq / step) is attached to the parameter of
    the same name, on CPU, fp32, with shape equality asserted.
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch

FSDP_PREFIX = "_fsdp_wrapped_module."


def load_model_state(model_pt_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "model" in payload and isinstance(payload["model"], Mapping):
        state = dict(payload["model"])
    elif isinstance(payload, Mapping):
        state = dict(payload)
    else:
        raise ValueError(f"model.pt has unexpected payload type {type(payload)}")
    return state


def attach_model_state(model, state: Mapping[str, torch.Tensor]) -> Dict:
    expected = set(model.state_dict().keys())
    actual = set(state.keys())
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"model.pt key mismatch: missing={missing[:5]} extra={extra[:5]} "
            f"(n_missing={len(missing)}, n_extra={len(extra)})")
    for name, tensor in state.items():
        param = model.state_dict()[name]
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"shape mismatch for {name}: {tuple(tensor.shape)} vs {tuple(param.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"unexpected dtype for {name}: {tensor.dtype}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite values in {name}")
    model.load_state_dict({k: v.clone() for k, v in state.items()})
    norms = {k: float(v.norm()) for k, v in state.items()}
    return {
        "n_tensors": len(state),
        "n_elements": int(sum(v.numel() for v in state.values())),
        "tensor_norms_sample": {k: norms[k] for k in sorted(norms)[:4]},
        "weight_norm_total": float(sum(n ** 2 for n in norms.values()) ** 0.5),
    }


def build_optimizer(optim_pt_path: str, model) -> tuple[torch.optim.AdamW, Dict]:
    payload = torch.load(optim_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "param_groups" not in payload or "state" not in payload:
        raise ValueError("optim.pt is not an optimizer state_dict")
    param_by_name = dict(model.named_parameters())
    state_by_id = payload["state"]
    groups_meta = []
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters()], lr=0.0, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0, amsgrad=False, foreach=None, fused=None,
        maximize=False, differentiable=False,
    )
    # Rebuild param_groups exactly as saved (names -> params), then attach state.
    saved_groups = payload["param_groups"]
    new_param_groups = []
    id_to_param = {}
    for group in saved_groups:
        names = [str(n) for n in group["param_names"]]
        params = []
        for name in names:
            key = name[len(FSDP_PREFIX):] if name.startswith(FSDP_PREFIX) else name
            if key not in param_by_name:
                raise ValueError(f"optimizer param name not in model: {name}")
            params.append(param_by_name[key])
        for pid, name in zip(group["params"], names):
            id_to_param[int(pid)] = (name, param_by_name[
                name[len(FSDP_PREFIX):] if name.startswith(FSDP_PREFIX) else name])
        new_param_groups.append({
            "params": params,
            "lr": float(group["lr"]),
            "betas": tuple(float(b) for b in group["betas"]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            # torch AdamW.step() requires these group keys to exist
            "amsgrad": bool(group.get("amsgrad", False)),
            "foreach": group.get("foreach"),
            "fused": group.get("fused", False),
            "maximize": bool(group.get("maximize", False)),
            "differentiable": bool(group.get("differentiable", False)),
            "capturable": bool(group.get("capturable", False)),
            "fused_step": group.get("fused_step", False),
            "decoupled_weight_decay": bool(group.get("decoupled_weight_decay", False)),
        })
        groups_meta.append({
            "n_params": len(params),
            "lr": float(group["lr"]),
            "betas": [float(b) for b in group["betas"]],
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "initial_lr": float(group.get("initial_lr", group["lr"])),
            "max_grad_norm": float(group.get("max_grad_norm", -1.0)),
            "amsgrad": bool(group.get("amsgrad", False)),
            "foreach": group.get("foreach"),
            "fused": group.get("fused"),
            "first_name": names[0], "last_name": names[-1],
        })
    optimizer.param_groups = new_param_groups
    # Attach state.
    step_values = []
    mv_report = {}
    for pid, entry in state_by_id.items():
        if int(pid) not in id_to_param:
            raise ValueError(f"optimizer state id {pid} not covered by param_groups")
        name, param = id_to_param[int(pid)]
        keys = set(entry.keys())
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(keys):
            raise ValueError(f"optimizer state for {name} missing keys: {sorted(keys)}")
        m = entry["exp_avg"].detach().clone()
        v = entry["exp_avg_sq"].detach().clone()
        step = entry["step"].detach().clone()
        if tuple(m.shape) != tuple(param.shape) or tuple(v.shape) != tuple(param.shape):
            raise ValueError(f"m/v shape mismatch for {name}")
        if m.dtype != torch.float32 or v.dtype != torch.float32:
            raise ValueError(f"m/v dtype mismatch for {name}: {m.dtype} {v.dtype}")
        if bool((v < 0).any()):
            raise ValueError(f"negative exp_avg_sq for {name}")
        if not bool(torch.isfinite(m).all()) or not bool(torch.isfinite(v).all()):
            raise ValueError(f"non-finite m/v for {name}")
        step_value = float(step.reshape(-1)[0])
        step_values.append(step_value)
        optimizer.state[param]["step"] = step.to(torch.float32)
        optimizer.state[param]["exp_avg"] = m
        optimizer.state[param]["exp_avg_sq"] = v
        if len(mv_report) < 4:
            mv_report[name] = {"m_norm": float(m.norm()), "v_mean": float(v.mean()),
                               "step": step_value}
    unique_steps = sorted(set(step_values))
    report = {
        "n_state_entries": len(state_by_id),
        "groups": groups_meta,
        "step_values_unique": unique_steps[:8],
        "step_values_min": min(step_values),
        "step_values_max": max(step_values),
        "mv_report": mv_report,
        "m_total_norm": float(sum(optimizer.state[p]["exp_avg"].norm() ** 2
                                  for p in model.parameters()) ** 0.5),
        "v_total_mean": float(sum(optimizer.state[p]["exp_avg_sq"].sum()
                                  for p in model.parameters()) /
                              sum(p.numel() for p in model.parameters())),
    }
    return optimizer, report


def inspect_train_state(train_pt_path: str) -> Dict:
    payload = torch.load(train_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("train.pt is not a mapping")
    rng = payload.get("rng", {})
    report = {
        "keys": sorted(payload.keys()),
        "global_step": int(payload.get("global_step", -1)),
        "global_train_tokens_seen": int(payload.get("global_train_tokens_seen", -1)),
        "epoch": int(payload.get("epoch", -1)),
        "world_size": int(payload.get("world_size", -1)),
        "has_rng_cuda": "cuda" in rng,
        "has_rng_numpy": "numpy" in rng,
        "has_rng_python": "python" in rng,
        "has_rng_torch": "torch" in rng,
        "has_scheduler_state": any("scheduler" in str(k).lower() for k in payload.keys()),
    }
    return report
