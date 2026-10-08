"""OLMo-2-0425-1B (stage1 run peteish1) architecture for FR_G1C_1B_FEASIBILITY_GATE_V01.

[FR-G1C-ADAPTER] Versioned, tested external adapter (frozen G1C_PROTOCOL.md
section 5 / feasibility protocol section 6.8 sanction): the MiniMind stack is
not used; this module re-implements only the forward architecture needed to
consume the official unsharded checkpoint. Module names are chosen so that
state_dict keys match the official checkpoint keys exactly
(e.g. ``transformer.blocks.0.att_proj.weight``); any mismatch fails fast at
load time (CONTRACT_INCOMPATIBLE). No frozen artifact is modified.

Registered architecture semantics (from the official step950000 config.yaml,
all fields read at build time):
  - block_type=sequential, norm_after=True  -> post-norm block
    x = attn_norm(x + attn_out(attn(x))); x = ff_norm(x + ff_out(mlp(x)))
  - attention_layer_norm=True               -> per-head q_norm/k_norm
    (RMSNorm over head_dim) applied BEFORE RoPE
  - rope=True, rope_full_precision=True     -> fp32 NeoX rotate-half RoPE
  - activation_type=swiglu, fused ff_proj   -> silu(h0) * h1
  - weight_tying=False                      -> separate wte and lm_head
    (the official lm_head key is ``transformer.ff_out.weight``)
  - include_bias=False                      -> no biases anywhere
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class G1CRMSNorm(nn.Module):
    """RMS norm (config layer_norm_type='rms'), affine, no bias."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class G1CRoPE(nn.Module):
    """GPT-NeoX rotate-half RoPE, computed in fp32 (rope_full_precision=True)."""

    def __init__(self, head_dim: int, theta: float, max_seq: int):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        # NeoX rotate-half pairs dim i with dim i + head_dim/2.
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, n_heads, T, head_dim]
        dtype = q.dtype
        T = q.shape[-2]
        cos = self.cos[:T].to(q.device)[None, None, :, :]
        sin = self.sin[:T].to(q.device)[None, None, :, :]
        q = q.float()
        k = k.float()
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q.to(dtype), k.to(dtype)


class G1CQKNorm(nn.Module):
    """OLMo-2 QK RMSNorm: normalized per head over head_dim, affine per head.

    The official checkpoint stores the affine weight with shape (d_model,) —
    one scale per (head, head_dim) pair — while the normalization itself is
    per-head RMS over head_dim (config ``attention_layer_norm: true``).
    """

    def __init__(self, d_model: int, n_heads: int, eps: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_heads, T, head_dim]
        dtype = x.dtype
        w = self.weight.float().view(self.n_heads, self.head_dim)
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * w[None, :, None, :]).to(dtype)


class G1CBlock(nn.Module):
    """OLMo2 sequential block, post-norm (config norm_after=True).

    Submodules are direct attributes so state_dict keys match the official
    checkpoint: att_proj / attn_out / q_norm / k_norm / attn_norm / ff_norm /
    ff_proj / ff_out under ``transformer.blocks.{i}.``.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, eps: float,
                 qk_norm_dim: int, rope_theta: float, max_seq: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if int(qk_norm_dim) != self.head_dim:
            raise ValueError(
                f"qk_norm_dim {qk_norm_dim} must equal head_dim {self.head_dim}")
        self.att_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = G1CQKNorm(d_model, n_heads, eps)
        self.k_norm = G1CQKNorm(d_model, n_heads, eps)
        self.attn_norm = G1CRMSNorm(d_model, eps)
        self.ff_norm = G1CRMSNorm(d_model, eps)
        self.ff_proj = nn.Linear(d_model, 2 * mlp_hidden, bias=False)
        self.ff_out = nn.Linear(mlp_hidden, d_model, bias=False)
        self.rope = G1CRoPE(self.head_dim, rope_theta, max_seq)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.att_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)
        # Math-backend SDPA (supports double backward); native flash has none.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        a = a.transpose(1, 2).reshape(B, T, -1)
        x = self.attn_norm(x + self.attn_out(a))
        h = self.ff_proj(x)
        h0, h1 = h.chunk(2, dim=-1)
        x = self.ff_norm(x + self.ff_out(F.silu(h0) * h1))
        return x


class G1CTransformer(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, n_layers: int,
                 mlp_hidden: int, embedding_size: int, eps: float,
                 rope_theta: float, max_seq: int, qk_norm_dim: int):
        super().__init__()
        self.wte = nn.Embedding(embedding_size, d_model)
        self.blocks = nn.ModuleList([
            G1CBlock(d_model, n_heads, mlp_hidden, eps, qk_norm_dim, rope_theta, max_seq)
            for _ in range(n_layers)
        ])
        self.ln_f = G1CRMSNorm(d_model, eps)
        self.ff_out = nn.Linear(d_model, embedding_size, bias=False)

    def forward_hidden(self, input_ids: torch.Tensor, *, checkpoint_blocks: bool = True) -> torch.Tensor:
        x = self.wte(input_ids)
        for block in self.blocks:
            if checkpoint_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.ln_f(x)


class G1COLMo2(nn.Module):
    """Outer wrapper so state_dict keys carry the official ``transformer.`` prefix."""

    def __init__(self, *, d_model: int, n_heads: int, n_layers: int,
                 mlp_hidden: int, vocab_size: int, embedding_size: int,
                 eps: float, rope_theta: float, max_seq: int,
                 qk_norm_dim: int):
        super().__init__()
        self.transformer = G1CTransformer(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            mlp_hidden=mlp_hidden, embedding_size=embedding_size, eps=eps,
            rope_theta=rope_theta, max_seq=max_seq, qk_norm_dim=qk_norm_dim)
        self.vocab_size = int(vocab_size)

    def forward_hidden(self, input_ids: torch.Tensor, *, checkpoint_blocks: bool = False) -> torch.Tensor:
        return self.transformer.forward_hidden(input_ids, checkpoint_blocks=checkpoint_blocks)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.ff_out(self.forward_hidden(input_ids))


def build_from_config(model_cfg: Mapping) -> G1COLMo2:
    d_model = int(model_cfg["d_model"])
    n_heads = int(model_cfg["n_heads"])
    mlp_hidden = model_cfg.get("mlp_hidden_size")
    if mlp_hidden is None:
        # official config: mlp_ratio=8 counts the FUSED swiglu output dim
        mlp_hidden = int(model_cfg["mlp_ratio"]) * d_model // 2
    # qk RMSNorm operates per head over head_dim (config has no explicit key)
    qk_norm_dim = model_cfg.get("qk_norm_dim") or d_model // n_heads
    return G1COLMo2(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=int(model_cfg["n_layers"]),
        mlp_hidden=int(mlp_hidden),
        vocab_size=int(model_cfg["vocab_size"]),
        embedding_size=int(model_cfg["embedding_size"]),
        eps=float(model_cfg["layer_norm_eps"]),
        rope_theta=float(model_cfg["rope_theta"]),
        max_seq=int(model_cfg["max_sequence_length"]),
        qk_norm_dim=int(qk_norm_dim),
    )


def chunked_mean_ce(hidden: torch.Tensor, lm_head_weight: torch.Tensor,
                    targets: torch.Tensor, chunk: int = 1024,
                    recompute_logits: bool = True) -> torch.Tensor:
    """Mean cross-entropy over all positions; logits computed chunk-wise.

    recompute_logits=True wraps the lm_head in a checkpoint so no full
    [T, vocab] logits tensor is retained (memory strategy, protocol 6.7).
    Deterministic: fixed chunk boundaries, fixed summation order.
    """
    T = hidden.shape[1]
    total = None
    count = 0
    for start in range(0, T, chunk):
        h = hidden[:, start:start + chunk]
        t = targets[:, start:start + chunk]
        if recompute_logits:
            logits = torch.utils.checkpoint.checkpoint(
                lambda x: F.linear(x, lm_head_weight), h, use_reentrant=False)
        else:
            logits = F.linear(h, lm_head_weight)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="sum")
        total = loss if total is None else total + loss
        count += t.numel()
    return total / float(count)
