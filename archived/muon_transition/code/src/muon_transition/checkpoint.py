from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from . import config

FORMAT_VERSION = "mte_state_v1"


def _payload_sha(obj: dict) -> str:
    """Deterministic sha over model_state + optimizer_states + cursor + meta.
    Excludes rng (contains non-byte-stable container objects); integrity of the
    training-relevant payload is what matters for resume verification."""
    h = hashlib.sha256()
    h.update(json.dumps(obj.get("cursor", {}), sort_keys=True, default=str).encode())
    h.update(json.dumps(obj.get("meta", {}), sort_keys=True, default=str).encode())
    for opt_state in obj["optimizer_states"]:
        h.update(json.dumps(opt_state["param_groups"], sort_keys=True, default=str).encode())
        for state in opt_state["state"].values():
            h.update(json.dumps({k: v for k, v in state.items() if not torch.is_tensor(v)},
                                sort_keys=True, default=str).encode())
            for k in sorted(state.keys()):
                if torch.is_tensor(state[k]):
                    h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    for k in sorted(obj["model_state"]):
        t = obj["model_state"][k]
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizers: list,
    cursor: dict,
    meta: dict | None = None,
) -> dict:
    model_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    payload = {
        "format": FORMAT_VERSION,
        "model_state": model_state,
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "cursor": cursor,
        "meta": meta or {},
        "torch_version": torch.__version__,
    }
    payload["payload_sha256"] = _payload_sha(payload)

    from vendored_fr.rng import capture_rng_state

    payload["rng"] = capture_rng_state()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
        tmp = Path(f.name)
    torch.save(payload, tmp)
    tmp.rename(path)
    return {"path": str(path), "payload_sha256": payload["payload_sha256"]}


def load_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizers: list,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT_VERSION:
        raise RuntimeError(f"unexpected checkpoint format: {payload.get('format')}")
    stored_sha = payload.pop("payload_sha256")
    recomputed = _payload_sha(payload)
    if stored_sha != recomputed:
        raise RuntimeError("checkpoint payload sha mismatch")

    model.load_state_dict(payload["model_state"])
    if len(optimizers) != len(payload["optimizer_states"]):
        raise RuntimeError("optimizer count mismatch on load")
    for opt, state in zip(optimizers, payload["optimizer_states"]):
        opt.load_state_dict(state)

    from vendored_fr.rng import restore_rng_state

    cursor = payload["cursor"]
    restore_rng_state(payload["rng"])
    return cursor


def parameter_fingerprint(model: torch.nn.Module) -> str:
    from .model import unique_named_parameters

    h = hashlib.sha256()
    for name, p in unique_named_parameters(model):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
