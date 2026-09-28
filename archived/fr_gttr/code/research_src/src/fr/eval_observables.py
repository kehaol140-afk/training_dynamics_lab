"""固定 eval-probe 的 mean / p95 / p99 / CVaR99 观测。

本模块是 `[FR-OBSERVATION-ONLY]`：
- 不 backward；
- 不 optimizer step；
- 不修改参数；
- 评估前捕获 RNG，评估后恢复 RNG；
- 评估结束恢复调用前的 train/eval 模式。
"""
from __future__ import annotations
import math
from typing import Sequence, Dict

from .rng import capture_rng_state, restore_rng_state
from .minimind_v03_runtime import stack_batch


def evaluate_token_losses(model, ds, indices: Sequence[int], batch_size: int, device: str, autocast_ctx) -> Dict[str, float | int]:
    import torch
    import torch.nn.functional as F

    if not indices:
        raise ValueError("eval indices 不能为空")

    rng = capture_rng_state()
    was_training = bool(model.training)
    model.eval()
    token_losses = []

    try:
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                x, y = stack_batch(ds, batch_idx, device)
                with autocast_ctx:
                    res = model(x, labels=y)
                    logits = res.logits[..., :-1, :].contiguous()
                    target = y[..., 1:].contiguous()

                # 统计时转 float32，避免 BF16 分位数精度影响。
                flat_logits = logits.float().reshape(-1, logits.shape[-1])
                flat_target = target.reshape(-1)
                losses = F.cross_entropy(
                    flat_logits, flat_target,
                    ignore_index=-100,
                    reduction="none",
                )
                valid = flat_target.ne(-100)
                if valid.any():
                    token_losses.append(losses[valid].detach().cpu())

        if not token_losses:
            raise RuntimeError("eval-probe 没有有效 token")

        z = torch.cat(token_losses).double()
        n = int(z.numel())
        mean = float(z.mean())
        p95 = float(torch.quantile(z, 0.95))
        p99 = float(torch.quantile(z, 0.99))
        # CVaR99：最坏 1% token loss 的平均值。
        k = max(1, int(math.ceil(0.01 * n)))
        cvar99 = float(torch.topk(z, k=k, largest=True).values.mean())
        return {
            "token_count": n,
            "mean": mean,
            "p95": p95,
            "p99": p99,
            "cvar99": cvar99,
        }
    finally:
        model.train(was_training)
        restore_rng_state(rng)
