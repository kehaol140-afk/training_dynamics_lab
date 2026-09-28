# TASK-00 当前实现审计结论

已由服务器当前 MiniMind 源码确认：

- 预训练优化器：torch.optim.AdamW
- 存在梯度累积
- 默认 accumulation_steps = 8
- 使用 AMP / GradScaler 接口
- optimizer 更新前执行 scaler.unscale_
- 随后执行 gradient clipping
- 随后 scaler.step(optimizer)
- 然后 scaler.update()
- 最后 optimizer.zero_grad(set_to_none=True)

E0 不直接使用 MiniMind 普通权重文件作为完整因果重放状态。
FR v0.2 单独保存模型、AdamW optimizer state、GradScaler、RNG 和实验位置 metadata。

最终是否允许 E0，以 research/scripts/task00_source_audit.py
和 research/scripts/e0_preflight.py 的服务器实测结果为准。
