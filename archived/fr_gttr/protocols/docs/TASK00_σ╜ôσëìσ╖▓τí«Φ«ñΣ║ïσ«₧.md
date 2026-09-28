# TASK-00 当前已确认事实（根据 2026-09-02 服务器 grep 输出）

> 这不是完整 `IMPLEMENTATION_AUDIT.md`，只记录当前已经从源码输出中直接确认的事实。未知项不猜。

## 1. 当前预训练主训练顺序

`minimind/trainer/train_pretrain.py` 已确认：

1. 取 batch，移动到 device；
2. 调用 `get_lr(...)` 计算当前学习率，并写入每个 optimizer param group；
3. 在 autocast 中 forward；
4. `loss = res.loss + res.aux_loss`；
5. `loss /= accumulation_steps`；
6. `scaler.scale(loss).backward()` 累积梯度；
7. 到 optimizer 更新边界时：`scaler.unscale_(optimizer)`；
8. `clip_grad_norm_(..., grad_clip)`；
9. `scaler.step(optimizer)`；
10. `scaler.update()`；
11. `optimizer.zero_grad(set_to_none=True)`。

## 2. 对新 FR 实验最重要的直接结论

### 梯度累积存在

默认 `accumulation_steps=8`。因此正式 `±epsilon*q` 不能简单地在每个 micro-batch backward 后随便插入；必须先明确我们要扰动的是“单个 micro-batch 梯度”还是“一个 optimizer update 前的累计梯度”。第一版更适合对**optimizer 真正消费前的累计梯度**做脉冲。

### AMP / unscale 存在

代码调用 `scaler.scale(loss).backward()`，并在 optimizer update 前执行 `scaler.unscale_(optimizer)`。

因此未来如果要对“真实未缩放梯度”施加 `epsilon*q`，合理候选位置是：

`scaler.unscale_(optimizer)` **之后**、`clip_grad_norm_` **之前**。

这样扰动与原始梯度处于同一数值尺度，并且正负分支都会经过完全相同的 clipping。

> 这是根据当前主训练流得到的候选位置；正式接线仍需 E0/adapter 审计批准。

### 原生 AdamW

代码通过 `optim.AdamW(model.parameters(), lr=args.learning_rate)` 创建优化器。当前调用只显式传入 `lr`；其它 AdamW 参数应在运行时记录 optimizer signature，而不要在审计文档里凭经验写死。

### 没有独立 scheduler 对象的直接证据

当前主训练循环每个 step 直接调用 `get_lr(...)` 并写入 param group。现有 grep 没显示预训练脚本创建 `scheduler` 对象。正式完整状态中应把“学习率路径/当前位置”作为状态，而不是机械要求一个不存在的 scheduler object。

## 3. Checkpoint 当前已确认

训练脚本保存两类东西：

- 单独的半精度模型权重 `.pth`；
- 调用 `lm_checkpoint(...)`，传入 `model / optimizer / scaler / epoch / step`。

resume 路径至少明确恢复：

- model state；
- optimizer state；
- scaler state。

但是，仅凭当前 grep 输出还不能确认 `lm_checkpoint` 是否保存 Python/NumPy/Torch/CUDA RNG、DataLoader/Sampler 状态，因此这些项目前仍应视为**待审计**。

## 4. 当前 E0 仍缺什么？

至少还需要确认：

1. `trainer_utils.py::lm_checkpoint` 的完整保存/加载字段；
2. `SkipBatchSampler` 如何决定续训后的 batch；
3. `setup_seed` 和 DataLoader worker 的 seed 行为；
4. Dataset 是否内部使用随机操作；
5. `torch.compile` 在 E0 是否关闭；
6. 是否需要 deterministic algorithm 设置才能达到 bitwise replay。

因此当前仍不能宣布 `E0_READY=YES`。
