# TASK-00 / v0.3 已确认实现事实

根据服务器当前 MiniMind 输出，预训练路径已确认：

```text
forward/loss
 -> scaler.scale(loss).backward()
 -> accumulation boundary
 -> scaler.unscale_(optimizer)
 -> clip_grad_norm_
 -> scaler.step(optimizer)
 -> scaler.update()
 -> optimizer.zero_grad(set_to_none=True)
```

并确认：
- optimizer 为 `torch.optim.AdamW`；
- 默认存在 gradient accumulation；
- 当前研究人工扰动未来候选位置是 unscale 之后、clip 之前；
- v0.3 的 E0/E1 均**不施加人工扰动**；
- E1 观察的是 optimizer-boundary 前的“未裁剪、已按真实尺度解释的累计梯度”。
