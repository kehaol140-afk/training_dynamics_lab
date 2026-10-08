"""Token-level training mixture manifests (frozen semantics, protocol v1.1 §2.3 / v1.2).

Per optimizer step the run consumes WINDOWS_PER_STEP windows of 256 tokens. For nominal p
(fraction of A-windows), each step draws exactly round(p * WINDOWS_PER_STEP) A-windows and the
remainder from B, so realized p_hat is exact at every step (window tokens are uniform 256).
Window order within each domain is sequential over the domain's packed train stream; the slot
interleave is a deterministic seeded shuffle identical across all runs (CRN across cells).
"""
from __future__ import annotations

import torch

from . import config

WINDOWS_PER_STEP = config.MICRO_BATCH_SEQUENCES * config.GRAD_ACCUM_STEPS


def build_step_manifest(
    a_train_windows: torch.Tensor,
    b_train_windows: torch.Tensor,
    p: float,
    n_steps: int,
    seed: int,
) -> dict:
    n_a_per_step = round(p * WINDOWS_PER_STEP)
    n_b_per_step = WINDOWS_PER_STEP - n_a_per_step
    needed_a = n_a_per_step * n_steps
    needed_b = n_b_per_step * n_steps
    if needed_a > a_train_windows.shape[0] or needed_b > b_train_windows.shape[0]:
        raise RuntimeError(
            f"insufficient windows for p={p}: need A={needed_a}/{a_train_windows.shape[0]}, "
            f"B={needed_b}/{b_train_windows.shape[0]}"
        )

    g = torch.Generator().manual_seed(seed)
    manifest = {"p_nominal": p, "n_a_per_step": n_a_per_step, "n_b_per_step": n_b_per_step,
                "windows_per_step": WINDOWS_PER_STEP, "n_steps": n_steps, "seed": seed, "steps": []}
    cursor_a = 0
    cursor_b = 0
    for _ in range(n_steps):
        a_idx = list(range(cursor_a, cursor_a + n_a_per_step))
        b_idx = list(range(cursor_b, cursor_b + n_b_per_step))
        cursor_a += n_a_per_step
        cursor_b += n_b_per_step
        slots = torch.randperm(WINDOWS_PER_STEP, generator=g).tolist()
        sources = ["A"] * n_a_per_step + ["B"] * n_b_per_step
        manifest["steps"].append(
            [{"slot": int(s), "domain": sources[i], "index": a_idx[i] if sources[i] == "A" else b_idx[i - n_a_per_step]}
             for i, s in enumerate(slots)]
        )
    manifest["realized_p_hat"] = (n_a_per_step * n_steps) / (WINDOWS_PER_STEP * n_steps)
    return manifest


class MixtureStream:
    """Materializes batches from a step manifest: n_micro batches of MICRO_BATCH_SEQUENCES windows."""

    def __init__(self, manifest: dict, a_windows: torch.Tensor, b_windows: torch.Tensor):
        self.manifest = manifest
        self.a = a_windows
        self.b = b_windows

    def micro_batches(self, step_index: int):
        entries = self.manifest["steps"][step_index]
        mbs = config.MICRO_BATCH_SEQUENCES
        for i in range(config.GRAD_ACCUM_STEPS):
            chunk = entries[i * mbs : (i + 1) * mbs]
            ids_a = [e["index"] for e in chunk if e["domain"] == "A"]
            ids_b = [e["index"] for e in chunk if e["domain"] == "B"]
            parts = [self.a[ids_a], self.b[ids_b]] if ids_a and ids_b else (
                [self.a[ids_a]] if ids_a else [self.b[ids_b]]
            )
            yield torch.cat(parts, dim=0)
