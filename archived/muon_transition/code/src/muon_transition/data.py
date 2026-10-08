from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from . import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def conversation_to_text(record: dict) -> str:
    turns = record.get("conversation", [])
    parts = []
    for turn in turns:
        human = str(turn.get("human", "") or "").strip()
        assistant = str(turn.get("assistant", "") or "").strip()
        parts.append("H: " + human)
        parts.append("A: " + assistant)
    return "\n".join(parts)


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.split())


def load_domain(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TokenizerHandle:
    def __init__(self, path: Path | None = None):
        from tokenizers import Tokenizer

        self.path = path or config.TOKENIZER_PATH
        self.tok = Tokenizer.from_file(str(self.path))
        self.vocab_size = self.tok.get_vocab_size()
        self.id_bos = self.tok.token_to_id("<s>")
        self.id_eos = self.tok.token_to_id("</s>")

    def encode_texts(self, texts: list[str]) -> list[list[int]]:
        encs = self.tok.encode_batch(texts, add_special_tokens=False)
        reserved = {0, 1, 2}
        return [[t for t in e.ids if t not in reserved] for e in encs]


def pack_domain_stream(
    doc_token_lists: list[list[int]],
    seq_len: int,
    bos_id: int,
    doc_sep_id: int,
) -> torch.Tensor:
    """Freeze packing rule: per-doc stream = enc(doc) + DOC_SEP; windows are consecutive
    seq_len slices of the concatenated per-domain stream, each prefixed with BOS.
    Returns int64 tensor (n_windows, seq_len). Effective loss tokens per window = seq_len - 1.
    """
    stream: list[int] = []
    for toks in doc_token_lists:
        stream.extend(toks)
        stream.append(doc_sep_id)
    n_windows = len(stream) // (seq_len - 1)
    used = n_windows * (seq_len - 1)
    windows = []
    for w in range(n_windows):
        chunk = stream[w * (seq_len - 1) : (w + 1) * (seq_len - 1)]
        windows.append([bos_id] + chunk)
    return torch.tensor(windows, dtype=torch.long)


@dataclass
class DomainData:
    name: str
    path: Path
    records: list[dict]
    texts: list[str]
    split_indices: dict
    token_lists: list[list[int]]
    windows: torch.Tensor


def duplicate_stats(texts: list[str]) -> dict:
    seen = {}
    n_dup_docs = 0
    for t in texts:
        key = normalize_for_dedup(t)
        if key in seen:
            n_dup_docs += 1
        else:
            seen[key] = True
    return {"documents": len(texts), "exact_duplicate_documents": n_dup_docs,
            "exact_duplicate_rate": n_dup_docs / max(len(texts), 1),
            "unique_normalized_documents": len(seen)}
