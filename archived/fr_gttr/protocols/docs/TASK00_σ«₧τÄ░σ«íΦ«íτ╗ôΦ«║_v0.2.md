# TASK-00 当前实现审计结论（v0.2）

> 本文件记录根据当前 MiniMind `train_pretrain.py` 结构已经确认的事实。
> 服务器运行 `task00_source_audit.py` 后，以服务器实际源码报告为最终依据。

## 1. 训练主路径

当前预训练循环的核心顺序为：

1. forward；
2. `loss = (res.loss + res.aux_loss) / accumulation_steps`；
3. `scaler.scale(loss).backward()`；
4. 到达 accumulation 边界；
5. `scaler.unscale_(optimizer)`；
6. `clip_grad_norm_`；
7. `scaler.step(optimizer)`；
8. `scaler.update()`；
9. `optimizer.zero_grad(set_to_none=True)`。

因此未来一次性梯度脉冲的**首选候选插入位置**是：

`unscale_ 之后、clip_grad_norm_ 之前`。

理由：此处梯度已经处于真实未缩放尺度，同时 AdamW 尚未消费梯度；随后仍经过原生 clipping。

## 2. AdamW

当前代码只向 `torch.optim.AdamW` 显式传入 `lr`，因此其余超参数采用当前 PyTorch AdamW 默认值。
正式实验必须在运行时记录实际 optimizer param_group，不能只依赖文档默认值。

## 3. 学习率

预训练没有独立 scheduler 对象，而是在每个 micro-step 调用 `get_lr(...)` 后直接写入 param_group。
因此“scheduler state”本质上由 epoch / step / iters / 初始 learning_rate 决定。

## 4. MiniMind 原生 resume 的精确重放问题

MiniMind 的 `lm_checkpoint` 会把模型 `state_dict` 转成 `half().cpu()` 后保存。
这适合常规续训，但不满足 E0 的“无损完整状态”要求。

原生 resume 保存：
- model（但已 half）；
- optimizer；
- epoch；
- step；
- world_size；
- 通过 kwargs 可保存 scaler。

没有看到原生 checkpoint 保存：
- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- Torch CUDA RNG；
- 显式 batch manifest。

所以 E0 必须使用独立 FR full-state checkpoint，且不覆盖 MiniMind 原生 checkpoint。

## 5. 数据顺序

单 GPU非 DDP 路径每个 epoch：

`setup_seed(42 + epoch) -> torch.randperm(len(train_ds)) -> SkipBatchSampler`

理论上可以按 epoch / batch_size / skip 重建顺序。
但正式 paired branch 仍建议保存显式 batch manifest，减少隐式依赖。

`PretrainDataset.__getitem__` 本身没有随机数据增强，主要是确定性 tokenizer + padding + labels。

## 6. checkpoint 边界风险

E0 只能在“完成 optimizer step 且 zero_grad 后”的清洁边界保存状态。
特别注意：若 epoch 最后存在不满 accumulation_steps 的尾部累积，原训练脚本的保存时机可能早于尾部 optimizer step。
因此 FR checkpoint 不应机械复用“step == iters”的原生保存点。
