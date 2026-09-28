"""梯度方向 q 的构造、保存与投影工具。

当前 v0.3 只实现 E1 所需能力：
1. 用训练数据中明确保留的 probe batch 构造 g_probe；
2. q = g_probe / ||g_probe||；
3. 计算自然累计梯度在 q 上的投影 <q, g>。

重要：本模块不施加任何人工扰动。±epsilon*q 仍然禁止进入 E1。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple


def normalize_direction(q_by_name: Dict[str, "object"], eps: float = 1e-30):
    """对按参数名保存的方向做全局 L2 归一化，并返回 (q, 原始范数)。"""
    import torch
    sq = torch.zeros((), dtype=torch.float64)
    for q in q_by_name.values():
        sq += q.detach().double().cpu().pow(2).sum()
    norm = torch.sqrt(sq)
    if not torch.isfinite(norm) or float(norm) <= eps:
        raise RuntimeError(f"probe 梯度范数无效：{float(norm)}")
    out = {k: v.detach().clone() / float(norm) for k, v in q_by_name.items()}
    return out, float(norm)


def direction_norm(q_by_name: Dict[str, "object"]) -> float:
    """计算方向全局 L2 范数。"""
    import torch
    sq = torch.zeros((), dtype=torch.float64)
    for q in q_by_name.values():
        sq += q.detach().double().cpu().pow(2).sum()
    return float(torch.sqrt(sq))


def direction_sha256(q_by_name: Dict[str, "object"]) -> str:
    """计算方向张量内容的稳定 SHA256，方便审计是否被替换。"""
    h = hashlib.sha256()
    for name in sorted(q_by_name):
        t = q_by_name[name].detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_direction(path: str | Path, q_by_name: Dict[str, "object"], metadata: Dict) -> str:
    """保存 q 与 metadata，并返回 q 的 SHA256。"""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = direction_sha256(q_by_name)
    payload = {
        "direction_version": "0.3",
        "sha256": sha,
        "metadata": dict(metadata),
        "q": {k: v.detach().clone().cpu() for k, v in q_by_name.items()},
    }
    torch.save(payload, path)
    return sha


def project_current_gradients(
    named_parameters: Iterable[Tuple[str, "object"]],
    q_by_name: Dict[str, "object"],
    *,
    grad_scale: float = 1.0,
):
    """只读计算未裁剪梯度的 <q,g> 与 ||g||。

    grad_scale 用于 float16 GradScaler：这里在读取时除以 scale，避免调用
    scaler.unscale_(optimizer) 改变 GradScaler 内部状态。BF16 时 grad_scale=1。

    为避免把整套大模型梯度反复拷回 CPU，乘加在梯度所在设备完成，最终只把
    两个标量（投影、梯度范数）返回 CPU。
    """
    import torch
    if grad_scale <= 0:
        raise ValueError("grad_scale 必须为正数")

    dot = None
    sq = None
    used = 0

    # ============================================================
    # [FR-OBSERVATION-ONLY] 这里只读取 p.grad，不修改模型/优化器/RNG。
    # [FR-PERTURBATION-CRITICAL] E1 明确禁止在这里写入 ±epsilon*q。
    # ============================================================
    for name, p in named_parameters:
        if p.grad is None:
            continue
        if name not in q_by_name:
            raise KeyError(f"q 中缺少有梯度参数：{name}")
        g = p.grad.detach().float() / float(grad_scale)
        q = q_by_name[name].detach().to(device=g.device, dtype=torch.float32)
        if g.shape != q.shape:
            raise ValueError(f"{name} 的 grad/q 形状不一致：{g.shape} vs {q.shape}")
        d = (g * q).sum()
        s = g.pow(2).sum()
        dot = d if dot is None else dot + d
        sq = s if sq is None else sq + s
        used += 1

    if dot is None or sq is None:
        raise RuntimeError("当前没有可观测梯度")

    return {
        "projection": float(dot.detach().double().cpu()),
        "grad_norm": float(torch.sqrt(sq).detach().double().cpu()),
        "used_parameter_tensors": used,
    }

def collect_current_gradients(named):
    grads=[]

    for name,p in named:
        if p.grad is not None:
            grads.append(
                p.grad.detach().reshape(-1).float().cpu()
            )

    import torch
    return torch.cat(grads)

def load_direction(path: str | Path):
    """加载并校验已冻结方向 q。"""
    import torch
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    q = payload["q"]
    expected = direction_sha256(q)
    if expected != payload.get("sha256"):
        raise RuntimeError("q 文件 SHA256 校验失败")
    return q, dict(payload.get("metadata", {})), expected


def parameter_displacement_projection(model, reference_state: Dict[str, "object"], q_by_name: Dict[str, "object"]) -> float:
    """计算 X_theta=<q, theta-current - theta-reference>。

    [FR-OBSERVATION-ONLY] 只读取当前参数，不修改训练状态。
    """
    import torch
    raw = getattr(model, "module", model)
    raw = getattr(raw, "_orig_mod", raw)
    total = None
    for name, p in raw.named_parameters():
        if not p.requires_grad or name not in q_by_name:
            continue
        if name not in reference_state:
            raise KeyError(f"reference_state 缺少参数：{name}")
        q = q_by_name[name].to(device=p.device, dtype=torch.float32)
        ref = reference_state[name].to(device=p.device, dtype=torch.float32)
        d = ((p.detach().float() - ref) * q).sum()
        total = d if total is None else total + d
    if total is None:
        raise RuntimeError("没有可用于参数位移投影的参数")
    return float(total.detach().double().cpu())
