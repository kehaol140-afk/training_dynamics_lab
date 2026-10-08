from __future__ import annotations

import random

import torch

from . import config


def _ensure_importable() -> None:
    import sys

    if str(config.MINIMIND_BASE) not in sys.path:
        sys.path.insert(0, str(config.MINIMIND_BASE))


def build_model(seed: int | None = None, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    _ensure_importable()
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    cfg = MiniMindConfig(**config.MODEL_CONFIG)
    model = MiniMindForCausalLM(cfg)
    model = model.to(dtype)
    if model.config.tie_word_embeddings:
        assert model.model.embed_tokens.weight is model.lm_head.weight
    return model


def unique_named_parameters(model: torch.nn.Module):
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield name, p
