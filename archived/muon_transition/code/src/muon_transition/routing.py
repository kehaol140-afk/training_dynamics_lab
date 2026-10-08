from __future__ import annotations

import torch

from . import config
from .model import unique_named_parameters


def route_parameters(model: torch.nn.Module):
    """Split unique trainable parameters into (muon_route, adamw_aux_route).

    Muon route: hidden-layer 2D linear matrix weights only (attn q/k/v/o, mlp gate/up/down).
    AdamW aux route: embedding (= tied lm_head), all norm weights, any bias/scalar/vector param.
    Tied embedding/head resolves to one tensor and is registered exactly once, in the aux route.
    """
    muon_route = []
    aux_route = []
    table = []
    seen = set()
    for name, p in unique_named_parameters(model):
        if id(p) in seen:
            raise RuntimeError(f"parameter appears twice in unique iteration: {name}")
        seen.add(id(p))
        tail = name
        for pattern in config.MUON_ROUTE_PATTERNS:
            if tail.endswith(pattern):
                tail = pattern
                break
        if tail in config.MUON_ROUTE_PATTERNS:
            if p.ndim != 2:
                raise RuntimeError(f"Muon-route param is not 2D: {name} {tuple(p.shape)}")
            muon_route.append(p)
            route = "muon"
        else:
            aux_route.append(p)
            route = "adamw_aux"
        table.append({"name": name, "shape": list(p.shape), "numel": int(p.numel()), "route": route})

    is_tied = model.config.tie_word_embeddings
    table.append(
        {
            "name": "lm_head.weight",
            "shape": list(model.lm_head.weight.shape),
            "numel": int(model.lm_head.weight.numel()),
            "route": "tied_to_model.embed_tokens.weight" if is_tied else "MISSING",
        }
    )
    return muon_route, aux_route, table


def audit_routing(model: torch.nn.Module) -> dict:
    muon_params, aux_params, table = route_parameters(model)

    muon_ids = [id(p) for p in muon_params]
    aux_ids = [id(p) for p in aux_params]
    overlap = sorted({i for i in muon_ids if aux_ids.count(i) > 0})

    all_unique = {id(p) for _, p in unique_named_parameters(model)}
    routed = set(muon_ids) | set(aux_ids)
    missing = all_unique - routed
    extra = routed - all_unique

    n_muon = sum(p.numel() for p in muon_params)
    n_aux = sum(p.numel() for p in aux_params)
    total = sum(p.numel() for _, p in unique_named_parameters(model))

    return {
        "total_parameters": int(total),
        "muon_parameters": int(n_muon),
        "adamw_aux_parameters": int(n_aux),
        "muon_tensor_count": len(muon_params),
        "adamw_aux_tensor_count": len(aux_params),
        "overlap_count": len(overlap),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "counts_add_up": bool(n_muon + n_aux == total),
        "tied_embedding_is_lm_head": bool(
            model.model.embed_tokens.weight is model.lm_head.weight
        ),
        "tie_registered_once": len([i for i in aux_ids if i == id(model.model.embed_tokens.weight)]) == 1,
        "non_2d_in_muon_route": [],
        "routing_table": table,
    }
