# TASK-00 服务器真实源码审计报告

> `[FR-OBSERVATION-ONLY]`：本报告只读源码，没有修改 `minimind/`。

## 核心判定

- 原生 MiniMind resume 是否满足 E0 精确重放：**NO**
- 使用 FR v0.2 full-state 后是否允许进入 E0 smoke：**YES**

## 状态表

| 项目 | 状态 | 说明 |
|---|---|---|
| 原生 AdamW | PASS | train_pretrain.py |
| 梯度累积 | PASS | 按 micro-step 判断 accumulation 边界 |
| backward→unscale→clip→step→zero_grad 顺序 | PASS | 未来扰动首选 unscale 后、clip 前 |
| 原生 checkpoint 保存 optimizer | PASS | 包含 AdamW state |
| 原生 checkpoint 保存 epoch/step | PASS | 训练位置 |
| 原生 checkpoint 保持模型原始 dtype | FAIL | 检测到 half().cpu() |
| 原生 checkpoint 保存 RNG | MISSING | E0 由 FR full_state 补齐 |
| 单 GPU randperm 数据顺序 | PASS | 每 epoch 可重建，但正式 paired branch 仍保存 manifest |
| PretrainDataset __getitem__ 随机性 | PASS | 未检测到 random.* |

## 关键结论

1. 当前训练主路径必须保持 `backward → unscale → clip → scaler.step(AdamW) → scaler.update → zero_grad`。
2. 未来 `±epsilon*q` 的首选候选插入位置是 `unscale` 后、`clip` 前；E0 当前禁止任何扰动。
3. MiniMind 原生 resume 面向常规续训，不是 E0 无损 checkpoint；FR v0.2 单独保存原始 dtype + optimizer + scaler + RNG。
4. 单 GPU非 DDP 的 epoch 顺序可由 `42+epoch` 重建，但 paired branch 使用显式 batch manifest 更安全。
5. FR checkpoint 必须放在 optimizer step + zero_grad 之后的干净边界。

## 源码 SHA256

- `train_pretrain.py`: `0c630aa6b5dce11a35f9176ff8337930a72a79bffee0a9282a4378b8005955f8`
- `trainer_utils.py`: `c613d485c8861caccc9d920e4639a8053cdfb4d69e56fa33280a631a5530234b`
- `lm_dataset.py`: `e6dd56b5745015ddc1579932c292ec7a5e1cc33796a5eadf4911065d92f1c6b8`
