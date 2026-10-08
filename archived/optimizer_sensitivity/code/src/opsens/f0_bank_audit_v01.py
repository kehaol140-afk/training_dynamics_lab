"""F0 existing-bank sensitivity audit library (v01).

Frozen scope in configs/f0_bank_audit_v01.json. All bank access is READ-ONLY.
The one-step map analysed is the frozen AdamW tangent (vendored
a0c_adamw_response_propagation_v01.adamw_tangent_step) evaluated at the
c=0 anchor (null next-step gradient), where the gradient->theta' channel is
EXACTLY the diagonal map G_i = lr*(1-beta1)/((1-beta1^(t+1))*(sqrt(vhat1_i)+eps))
with vhat1 = beta2*v/(1-beta2^(t+1)); on any coordinate with c_i = 0 the
dsqrt cross term vanishes identically, so this diagonal is exact per
coordinate, independent of the unknown true next-batch gradient.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


LAB_ROOT = Path(__file__).resolve().parents[3]
LEGACY = LAB_ROOT / "legacy_fluctuation_response"
MINIMIND_BASE = LAB_ROOT / "minimind_base"
D0_CORPUS = LEGACY / "outputs/e3_confirmatory/v07/e3c_subset.jsonl"

GROUP_RULES: Sequence[Tuple[str, str]] = (
    ("embed_tied", r"^model\.embed_tokens\.weight$"),
    ("attn_q", r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$"),
    ("attn_k", r"^model\.layers\.\d+\.self_attn\.k_proj\.weight$"),
    ("attn_v", r"^model\.layers\.\d+\.self_attn\.v_proj\.weight$"),
    ("attn_o", r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$"),
    ("attn_qk_norm", r"^model\.layers\.\d+\.self_attn\.(q_norm|k_norm)\.weight$"),
    ("rmsnorm", r"^model(\.layers\.\d+\.(input_layernorm|post_attention_layernorm)|\.norm)\.weight$"),
    ("mlp_gate", r"^model\.layers\.\d+\.mlp\.gate_proj\.weight$"),
    ("mlp_up", r"^model\.layers\.\d+\.mlp\.up_proj\.weight$"),
    ("mlp_down", r"^model\.layers\.\d+\.mlp\.down_proj\.weight$"),
)


def classify_group(name: str) -> str:
    for group, pattern in GROUP_RULES:
        if re.match(pattern, name):
            return group
    return "others"


def load_bank_state(path: Path) -> Dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def named_optimizer_entries(state: Dict) -> List[Tuple[str, int, Dict]]:
    """Map optimizer state slots to parameter names.

    The model state dict carries both `model.embed_tokens.weight` and the tied
    `lm_head.weight` copy while the optimizer holds one entry per unique
    parameter; the tied duplicate is dropped by name before matching order.
    """
    model_keys = [k for k in state["model"].keys() if k != "lm_head.weight"]
    opt_state = state["optimizer"]["state"]
    opt_keys = list(opt_state.keys())
    entries = []
    for idx, name in zip(opt_keys, model_keys):
        slot = opt_state[idx]
        shape = tuple(state["model"][name].shape)
        if tuple(slot["exp_avg"].shape) != shape or tuple(slot["exp_avg_sq"].shape) != shape:
            raise ValueError(f"shape mismatch at slot {idx} vs {name}: {slot['exp_avg'].shape} vs {shape}")
        entries.append((name, idx, slot))
    if len(entries) != len(model_keys):
        raise ValueError(f"optimizer slots {len(entries)} != model params {len(model_keys)}")
    return entries


def group_hyperparams(state: Dict) -> Dict:
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1:
        raise ValueError(f"expected single param group, got {len(groups)}")
    g = groups[0]
    return {
        "lr": float(g["lr"]),
        "beta1": float(g["betas"][0]),
        "beta2": float(g["betas"][1]),
        "eps": float(g["eps"]),
        "weight_decay": float(g.get("weight_decay", 0.0)),
    }


def exact_quantiles(t: torch.Tensor, qs: Sequence[float]) -> List[float]:
    x = t.detach().to(torch.float64).flatten()
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    s, _ = torch.sort(x)
    n = s.numel()
    out = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        val = s[lo] if hi == lo else float(s[lo]) * (1 - frac) + float(s[hi]) * frac
        out.append(float(val))
    return out


QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.01, 0.05, 0.5)


def tensor_floor_stats(t: torch.Tensor, eps: float) -> Dict:
    """Stats for a nonnegative tensor (v=exp_avg_sq): raw quantiles of v,
    sqrt(v) threshold fractions, and exact-zero counts."""
    x = t.detach().to(torch.float64).flatten()
    n = x.numel()
    zeros = int(torch.count_nonzero(x == 0.0).item())
    sqrt_x = torch.sqrt(torch.clamp(x, min=0.0))
    thresholds = {
        "sqrt_v_lt_eps": eps,
        "sqrt_v_lt_10eps": 10.0 * eps,
        "sqrt_v_lt_100eps": 100.0 * eps,
        "sqrt_v_lt_1000eps": 1000.0 * eps,
    }
    qvals = exact_quantiles(x, [0.0, 0.001, 0.01, 0.1, 0.05, 0.5, 0.99, 0.999, 1.0])
    out = {
        "n_coords": n,
        "exact_zero_count": zeros,
        "exact_zero_fraction": zeros / n if n else float("nan"),
        "min": qvals[0],
        "p0_001": qvals[1],
        "p0_01": qvals[2],
        "p0_1": qvals[3],
        "p5": qvals[4],
        "median": qvals[5],
        "p99": qvals[6],
        "p99_9": qvals[7],
        "max": qvals[8],
    }
    for label, thr in thresholds.items():
        out[label + "_fraction"] = float((sqrt_x < thr).sum().item()) / n if n else float("nan")
    return out


def signed_stats(t: torch.Tensor, eps: float) -> Dict:
    """Stats for a signed tensor (m=exp_avg): quantiles of |m|."""
    x = t.detach().to(torch.float64).flatten().abs()
    qvals = exact_quantiles(x, [0.0, 0.001, 0.01, 0.1, 0.05, 0.5, 0.99, 0.999, 1.0])
    return {
        "n_coords": int(x.numel()),
        "abs_min": qvals[0],
        "abs_p0_001": qvals[1],
        "abs_p0_01": qvals[2],
        "abs_p0_1": qvals[3],
        "abs_p5": qvals[4],
        "abs_median": qvals[5],
        "abs_p99": qvals[6],
        "abs_p99_9": qvals[7],
        "abs_max": qvals[8],
    }


def vhat1_anchor(v: torch.Tensor, step: float, beta2: float) -> torch.Tensor:
    t_after = float(step) + 1.0
    bc2 = 1.0 - beta2 ** t_after
    return beta2 * v / bc2


def diagonal_gain(v: torch.Tensor, step: float, lr: float, beta1: float, beta2: float, eps: float) -> torch.Tensor:
    """Exact per-coordinate gradient->theta' Jacobian at the c=0 anchor."""
    t_after = float(step) + 1.0
    bc1 = 1.0 - beta1 ** t_after
    denom = torch.sqrt(vhat1_anchor(v, step, beta2)) + eps
    return lr * (1.0 - beta1) / (bc1 * denom)


def denominator_anchor(v: torch.Tensor, step: float, beta2: float, eps: float) -> torch.Tensor:
    return torch.sqrt(vhat1_anchor(v, step, beta2)) + eps


def token_frequency_bins(corpus_path: Path = D0_CORPUS, tokenizer_dir: Path = MINIMIND_BASE / "model",
                         n_bins_head_frac: float = 0.2) -> Dict:
    """Pre-registered rule (frozen in configs/f0_bank_audit_v01.json):
    count token ids over all text bodies; rank by count desc;
    head = top 20% ids, mid = 20-80%, tail = bottom 20% among count>0,
    never = count==0. bos/eos appear per-sample and are listed separately."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)
    vocab = int(tok.vocab_size)
    counts = torch.zeros(vocab, dtype=torch.int64)
    n_rows = 0
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            for i in ids:
                counts[i] += 1
            n_rows += 1
    order = torch.argsort(counts, descending=True)
    observed = int((counts > 0).sum().item())
    n_head = max(1, int(round(observed * n_bins_head_frac)))
    head_ids = order[:n_head]
    tail_ids = order[max(0, observed - n_head):observed]
    mid_ids = order[n_head:observed - n_head] if observed - n_head > n_head else order[n_head:observed]
    never_ids = order[observed:]
    return {
        "vocab": vocab,
        "n_rows": n_rows,
        "counts": counts,
        "observed": observed,
        "head": head_ids.tolist(),
        "mid": mid_ids.tolist(),
        "tail": tail_ids.tolist(),
        "never": never_ids.tolist(),
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "tokenizer_name": str(tokenizer_dir),
    }


def embedding_row_bins(bins: Dict) -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(bins[k], dtype=torch.long) for k in ("head", "mid", "tail", "never")}


def fake_optimizer_from_bank(state: Dict, dtype: torch.dtype) -> Tuple[List[Tuple[str, torch.Tensor]], object]:
    """Build named leaf parameters + a minimal optimizer-like object carrying
    bank m/v/step and group hyperparams, for the vendored frozen
    adamw_tangent_step (which reads optimizer.param_groups and optimizer.state)."""
    hp = group_hyperparams(state)
    named = []
    opt_state = {}
    for name, idx, slot in named_optimizer_entries(state):
        p = torch.nn.Parameter(state["model"][name].detach().to(dtype).clone())
        named.append((name, p))
        opt_state[p] = {
            "step": torch.tensor(float(slot["step"])),
            "exp_avg": slot["exp_avg"].detach().to(dtype).clone(),
            "exp_avg_sq": slot["exp_avg_sq"].detach().to(dtype).clone(),
        }

    class _Opt:
        pass

    opt = _Opt()
    opt.param_groups = [{"params": [p for _, p in named], "lr": hp["lr"],
                         "betas": (hp["beta1"], hp["beta2"]), "eps": hp["eps"],
                         "weight_decay": hp["weight_decay"], "amsgrad": False,
                         "maximize": False, "differentiable": False}]
    opt.state = opt_state
    return named, opt


def run_tangent_at_null_gradient(named, optimizer, dc_by_name: Dict, *,
                                 dm_by_name: Optional[Dict] = None,
                                 dv_by_name: Optional[Dict] = None) -> Dict:
    from .a0c_adamw_response_propagation_v01 import adamw_tangent_step

    zero = {name: torch.zeros_like(p.detach()) for name, p in named}
    return adamw_tangent_step(
        named_parameters=named,
        optimizer=optimizer,
        clipped_gradient_by_name={name: torch.zeros_like(p.detach()) for name, p in named},
        tangent_clipped_gradient_by_name=dc_by_name,
        tangent_theta_by_name=zero,
        tangent_m_by_name=dm_by_name if dm_by_name is not None else zero,
        tangent_v_by_name=dv_by_name if dv_by_name is not None else zero,
    )



def norm_by_group(named, tensor_by_name: Dict) -> Dict[str, float]:
    total = {}
    for name, _ in named:
        g = classify_group(name)
        t = tensor_by_name[name].detach().to(torch.float64)
        total[g] = total.get(g, 0.0) + float(torch.sum(t * t).item())
    return {k: math.sqrt(v) for k, v in total.items()}


def total_norm(tensor_by_name: Dict) -> float:
    acc = 0.0
    for t in tensor_by_name.values():
        x = t.detach().to(torch.float64)
        acc += float(torch.sum(x * x).item())
    return math.sqrt(acc)
