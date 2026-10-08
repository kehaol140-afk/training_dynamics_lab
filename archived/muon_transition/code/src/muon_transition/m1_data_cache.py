"""Cached domain train-window tensors for M1 runs (deterministic, matches M0 audit)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from . import config
from .data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
from .partitions import build_pair_aligned_split

_CACHE: dict = {}
_CACHE_PATH = Path("/tmp/opencode/mte_domain_train_windows.pt")


def get_domain_windows():
    if "data" in _CACHE:
        return _CACHE["data"]
    if _CACHE_PATH.exists():
        blob = torch.load(_CACHE_PATH, weights_only=True)
        _CACHE["data"] = (blob["A_train"], blob["B_train"])
        return _CACHE["data"]

    sys.path.insert(0, str(config.PROJECT_ROOT))
    from scripts.m0_gate3_data_audit import dedup_keep_first

    tok = TokenizerHandle()
    deduped = {}
    for name, path in (("A", config.DOMAIN_A_PATH), ("B", config.DOMAIN_B_PATH)):
        records = load_domain(path)
        texts = [conversation_to_text(r) for r in records]
        keep = dedup_keep_first(texts)
        deduped[name] = ([records[i] for i in keep], [texts[i] for i in keep])

    manifest = build_pair_aligned_split(deduped["A"][0], deduped["B"][0])
    out = {}
    for name in ("A", "B"):
        token_lists = tok.encode_texts(deduped[name][1])
        tl = [token_lists[i] for i in sorted(manifest[name]["train"])]
        out[f"{name}_train"] = pack_domain_stream(tl, config.SEQ_LEN, config.BOS_ID, config.DOC_SEP_ID)

    expected = {"A": 385071, "B": 295825}
    for name in ("A", "B"):
        if out[f"{name}_train"].shape[0] != expected[name]:
            raise RuntimeError(
                f"{name} train windows {out[f'{name}_train'].shape[0]} != M0 audit {expected[name]}"
            )
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, _CACHE_PATH)
    _CACHE["data"] = (out["A_train"], out["B_train"])
    return _CACHE["data"]
