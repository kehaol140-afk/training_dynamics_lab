"""F1 1B replication library (v01) — frozen scope in docs/F1_1B_REPLICATION_PREREG.md.

All checkpoint access is READ-ONLY (external/peteish1/, SHA256-gated downloads).
Vendored frozen infrastructure: g1c_olmo2_{architecture,checkpoint,tangent}
(verbatim from legacy fr/, see docs/LEGACY_PROVENANCE.md) and the F0
a0c_adamw_response_propagation_v01 helpers.

The one-step map analysed mirrors F0: gradient->theta' channel of one AdamW
step at the c=0 anchor, exactly the per-coordinate diagonal
G_i = lr*(1-beta1)/((1-beta1^(t+1))*(sqrt(vhat1_i)+eps)),
vhat1 = beta2*v/(1-beta2^(t+1)). At exact-zero v coordinates this diagonal is
the CENTRAL (two-sided) directional derivative; the one-sided derivative
saturates at lr*(1-beta1)/(bc1*(a|c|+eps)) with a=sqrt((1-beta2)/bc2)
(prereg section 0).
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .g1c_olmo2_checkpoint import inspect_train_state, load_model_state

LAB_ROOT = Path(__file__).resolve().parents[3]
LEGACY = LAB_ROOT / "legacy_fluctuation_response"
G1C_ROOT = LEGACY / "outputs/generalization_g1c_feasibility_v01"
EXT = LAB_ROOT / "optimizer_sensitivity" / "external" / "peteish1"
OUT = LAB_ROOT / "optimizer_sensitivity" / "outputs" / "f1_1b_replication"

STEPS = (380000, 950000, 1530000)
PARAM_COUNT = 1484916736
N_TENSORS = 131
FSDP_PREFIX = "_fsdp_wrapped_module."

CHECKPOINT_SHA256 = {
    (380000, "model.pt"): "ca2fa02c8192651c7280e848e96c69f954451f784f26b97e8b11385c54470fa7",
    (380000, "optim.pt"): "3e48789a515e32d6d44618fecc77efd896b1055ce467568e06a79eab9df3bae6",
    (950000, "model.pt"): "6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b",
    (950000, "optim.pt"): "ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2",
    (1530000, "model.pt"): "d55f32683d56d1e61d0b919d265a611bebec864875db404bcb0642b0dd419120",
    (1530000, "optim.pt"): "58f7f01d73a1e7772568a70234759ef82fa3d65367b75a813dc5b60e17158884",
}
CHECKPOINT_SIZES = {
    (380000, "model.pt"): 5939703038,
    (380000, "optim.pt"): 11879442406,
    (950000, "model.pt"): 5939703038,
    (950000, "optim.pt"): 11879442406,
    (1530000, "model.pt"): 5939703038,
    (1530000, "optim.pt"): 11879442406,
}
Q_STREAM1_TENSOR_SHA256 = "08128dd4f86429399d9b732778d18e81b2a35f76ef978d71a25c3b883ca2cbb0"

ATT_D = 2048
MLP_H = 8192


def classify_groups(name: str) -> List[Tuple[str, slice]]:
    """Return [(group, row-slice)] for one parameter name (prereg-frozen rules)."""
    if name == "transformer.wte.weight":
        return [("embed", slice(None))]
    if name == "transformer.ff_out.weight":
        return [("head", slice(None))]
    if name == "transformer.ln_f.weight":
        return [("norm", slice(None))]
    m = re.match(r"^transformer\.blocks\.(\d+)\.(att_proj|attn_out|q_norm|k_norm|attn_norm|ff_norm|ff_proj|ff_out)\.weight$", name)
    if not m:
        return [("others", slice(None))]
    kind = m.group(2)
    if kind == "att_proj":
        return [
            ("attn_q", slice(0, ATT_D)),
            ("attn_k", slice(ATT_D, 2 * ATT_D)),
            ("attn_v", slice(2 * ATT_D, 3 * ATT_D)),
        ]
    if kind == "attn_out":
        return [("attn_o", slice(None))]
    if kind in ("q_norm", "k_norm"):
        return [("qknorm", slice(None))]
    if kind in ("attn_norm", "ff_norm"):
        return [("norm", slice(None))]
    if kind == "ff_proj":
        return [
            ("mlp_gate", slice(0, MLP_H)),
            ("mlp_up", slice(MLP_H, 2 * MLP_H)),
        ]
    if kind == "ff_out":
        return [("mlp_down", slice(None))]
    return [("others", slice(None))]


MODULE_GROUPS = ("embed", "head", "attn_q", "attn_k", "attn_v", "attn_o",
                 "mlp_gate", "mlp_up", "mlp_down", "qknorm", "norm", "others")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_path(step: int, name: str) -> Path:
    return EXT / f"step{step}_{name}"


def verify_checkpoint(step: int, name: str, skip_sha: bool = False) -> Dict:
    p = ckpt_path(step, name)
    exp_size = CHECKPOINT_SIZES[(step, name)]
    exp_sha = CHECKPOINT_SHA256[(step, name)]
    size = p.stat().st_size if p.exists() else -1
    out = {"path": str(p), "exists": p.exists(), "size": size,
           "size_ok": size == exp_size}
    if out["size_ok"] and not skip_sha:
        out["sha256"] = sha256_file(p)
        out["sha_ok"] = out["sha256"] == exp_sha
    else:
        out["sha256"] = None
        out["sha_ok"] = False if not out["size_ok"] else (skip_sha or False)
    return out


def load_model_cfg(step: int) -> dict:
    import yaml
    cfg = yaml.safe_load((G1C_ROOT / f"checkpoints/step{step}_config.yaml").read_text(encoding="utf-8"))
    return cfg["model"]


def load_optim_payload(step: int) -> Dict:
    return torch.load(str(ckpt_path(step, "optim.pt")), map_location="cpu", weights_only=False)


def optim_named_state(step: int, payload: Optional[Dict] = None):
    """Ordered [(name, {step, exp_avg, exp_avg_sq})] + meta with per-group name lists.

    Mirrors the frozen G1C mapping (FSDP prefix strip; order = param_groups order).
    """
    if payload is None:
        payload = load_optim_payload(step)
    groups = payload["param_groups"]
    state = payload["state"]
    group_names: List[List[str]] = []
    name_to_pid: Dict[str, int] = {}
    order: List[str] = []
    hp = []
    for g in groups:
        names_g = []
        for pid, raw in zip(g["params"], g["param_names"]):
            name = raw[len(FSDP_PREFIX):] if raw.startswith(FSDP_PREFIX) else raw
            name_to_pid[name] = int(pid)
            names_g.append(name)
            order.append(name)
        group_names.append(names_g)
        hp.append({
            "n_params": len(names_g),
            "lr": float(g["lr"]),
            "betas": (float(g["betas"][0]), float(g["betas"][1])),
            "eps": float(g["eps"]),
            "weight_decay": float(g["weight_decay"]),
            "initial_lr": float(g.get("initial_lr", g["lr"])),
            "max_grad_norm": float(g.get("max_grad_norm", -1.0)),
        })
    out = []
    for name in order:
        entry = state[name_to_pid[name]]
        out.append((name, {
            "step": float(entry["step"].reshape(-1)[0]),
            "exp_avg": entry["exp_avg"],
            "exp_avg_sq": entry["exp_avg_sq"],
        }))
    meta = {"groups": hp, "group_names": group_names, "order": order}
    return out, meta


def vhat1(v: torch.Tensor, step_t: float, beta2: float) -> torch.Tensor:
    t_after = step_t + 1.0
    bc2 = 1.0 - beta2 ** t_after
    return beta2 * v / bc2


def diagonal_gain(v: torch.Tensor, step_t: float, lr: float, beta1: float,
                   beta2: float, eps: float, compute_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Exact c=0-anchor gradient->theta' diagonal (central derivative at zero-v).

    compute_dtype: cast v FIRST so the whole arithmetic runs in this dtype
    (python scalars are weak-typed and would otherwise keep v's storage dtype)."""
    if compute_dtype is not None:
        v = v.to(compute_dtype)
    t_after = step_t + 1.0
    bc1 = 1.0 - beta1 ** t_after
    denom = torch.sqrt(vhat1(v, step_t, beta2)) + eps
    return lr * (1.0 - beta1) / (bc1 * denom)


def exact_quantiles(x: torch.Tensor, qs: Sequence[float]) -> List[float]:
    x = x.detach().reshape(-1)
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


V_QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.05, 0.5, 0.99, 0.999, 1.0)
SQRTV_THRESHOLDS = (1.0, 10.0, 100.0, 1000.0)


def floor_stats(v_f64: torch.Tensor, eps: float) -> Dict:
    x = v_f64.flatten()
    n = x.numel()
    zeros = int(torch.count_nonzero(x == 0.0).item())
    sqrt_x = torch.sqrt(torch.clamp(x, min=0.0))
    q = exact_quantiles(x, V_QUANTILES)
    out = {
        "n_coords": n,
        "exact_zero_count": zeros,
        "exact_zero_fraction": zeros / n if n else float("nan"),
        "min": q[0], "p0_001": q[1], "p0_01": q[2], "p0_1": q[3], "p5": q[4],
        "median": q[5], "p99": q[6], "p99_9": q[7], "max": q[8],
    }
    for m in SQRTV_THRESHOLDS:
        out[f"sqrt_v_lt_{int(m)}eps_fraction"] = float((sqrt_x < m * eps).sum().item()) / n if n else float("nan")
    return out


def precision_roundtrip(v_f32: torch.Tensor, m_f32: torch.Tensor) -> Dict:
    """Frozen round-to-zero attribution (prereg section 4)."""
    out = {}
    for label, t in (("v", v_f32), ("m", m_f32)):
        pos = t > 0
        out[f"{label}_n"] = t.numel()
        out[f"{label}_math_zero"] = int((~pos).sum().item())
        out[f"{label}_bf16_rt0"] = int((pos & (t.to(torch.bfloat16) == 0)).sum().item())
        out[f"{label}_fp16_rt0"] = int((pos & (t.to(torch.float16) == 0)).sum().item())
    return out


def gain_stats(g_f64: torch.Tensor) -> Dict:
    q = exact_quantiles(g_f64, (0.0, 0.001, 0.5, 0.999, 1.0))
    return {"gain_min": q[0], "gain_p0_001": q[1], "gain_median": q[2],
            "gain_p99_9": q[3], "gain_max": q[4]}


def signed_abs_stats(m_f64: torch.Tensor) -> Dict:
    q = exact_quantiles(m_f64.abs(), (0.0, 0.001, 0.01, 0.1, 0.5, 0.99, 0.999, 1.0))
    return {"m_abs_min": q[0], "m_abs_p0_001": q[1], "m_abs_p0_01": q[2],
            "m_abs_p0_1": q[3], "m_abs_median": q[4], "m_abs_p99": q[5],
            "m_abs_p99_9": q[6], "m_abs_max": q[7]}


def fake_group_optimizer(named_state, meta, dtype: torch.dtype, theta_by_name=None):
    """(named leaf params, optimizer-like object) carrying checkpoint m/v/step and
    heterogeneous group hyperparams, for the vendored g1c_adamw_tangent_step.

    theta values are irrelevant to the tangent math at the c=0 anchor; zeros are
    used unless real theta is supplied (stage 5 FD uses real theta)."""
    named = []
    opt_state = {}
    for name, slot in named_state:
        if theta_by_name is not None:
            base = theta_by_name[name]
        else:
            base = torch.zeros_like(slot["exp_avg"])
        p = torch.nn.Parameter(base.detach().to(dtype).clone())
        named.append((name, p))
        opt_state[p] = {
            "step": torch.tensor(float(slot["step"])),
            "exp_avg": slot["exp_avg"].detach().to(dtype).clone(),
            "exp_avg_sq": slot["exp_avg_sq"].detach().to(dtype).clone(),
        }
    param_of = dict(named)
    payload_groups = []
    for g, names_g in zip(meta["groups"], meta["group_names"]):
        payload_groups.append({
            "params": [param_of[n] for n in names_g],
            "lr": g["lr"], "betas": g["betas"], "eps": g["eps"],
            "weight_decay": g["weight_decay"], "amsgrad": False, "maximize": False,
            "differentiable": False,
        })

    class _Opt:
        pass

    opt = _Opt()
    opt.param_groups = payload_groups
    opt.state = opt_state
    return named, opt


def modulewise_norms(named, tensor_by_name) -> Dict[str, float]:
    """Per-module L2 norms using prereg row-slices on fused tensors."""
    acc: Dict[str, float] = {}
    for name, _ in named:
        t = tensor_by_name[name].detach().to(torch.float64)
        for g, sl in classify_groups(name):
            x = t if sl == slice(None) else t[sl]
            s = float(torch.sum(x * x).item()) if x.numel() else 0.0
            acc[g] = acc.get(g, 0.0) + s
    return {k: math.sqrt(v) for k, v in acc.items()}


def total_norm(tensor_by_name) -> float:
    acc = 0.0
    for t in tensor_by_name.values():
        x = t.detach().to(torch.float64)
        acc += float(torch.sum(x * x).item())
    return math.sqrt(acc)


def argmax_coordinate(named, tensor_by_name):
    """Global max-|value| coordinate -> (param, group, flat_idx, row, value)."""
    best = None
    for name, _ in named:
        t = tensor_by_name[name].detach().to(torch.float64)
        flat = int(t.abs().argmax().item())
        val = float(t.flatten()[flat].item())
        if best is None or val > best[4]:
            row = flat // t.shape[1] if t.dim() == 2 else flat
            grp = next(g for g, sl in classify_groups(name)
                       if sl == slice(None) or (row >= sl.start and row < sl.stop))
            best = (name, grp, flat, int(row), val)
    return best


def regenerate_q_stream1(model) -> Dict[str, torch.Tensor]:
    """Frozen legacy recipe (CLEANUP_G1C_LARGE_FILES_20260925.md): iid N(0,1) over
    named_parameters order, unit total L2; q_seed frozen there."""
    q_seed = 4655128343037472768
    gen = torch.Generator(device="cpu").manual_seed(q_seed)
    q = {}
    for name, p in model.named_parameters():
        q[name] = torch.randn(p.shape, generator=gen, dtype=torch.float32)
    total = math.sqrt(sum(float(v.double().pow(2).sum()) for v in q.values()))
    return {k: (v / total).contiguous() for k, v in q.items()}


def direction_tensor_sha256(q: Dict[str, torch.Tensor]) -> str:
    """fr.direction.direction_sha256 semantics (frozen): sorted names, name bytes +
    dtype str + shape str + tensor bytes."""
    h = hashlib.sha256()
    for name in sorted(q):
        t = q[name].detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()
