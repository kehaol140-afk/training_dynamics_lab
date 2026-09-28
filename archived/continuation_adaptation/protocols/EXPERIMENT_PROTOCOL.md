# EXPERIMENT_PROTOCOL — Optimizer State Adaptation for Continuation under Changed Budget
# （第一轮：smoke / 淘汰实验实现协议）

冻结日期：2026-09-26（`configs/smoke_v01.json` 与本文档同时冻结；此后不得依据结果修改口径）。

---

## 0. 研究问题（继承 PROJECT_CONTEXT.md，不扩展）

Transformer checkpoint（含 AdamW m/v/step）在改变训练预算/优化器超参后续训时，
继承的 optimizer state 是否仍然适配？简单的 state 适配能否优于认真调参的普通续训 baseline？

**本协议只覆盖实现正确性验证（smoke）。** 大规模淘汰矩阵（P4）的完整统计口径
（多 seed × 多 checkpoint × 配对检验 × 停止条件）在 smoke 通过评审后另行冻结，此处只预留接口。

---

## 1. 实验单元

- **C0（checkpoint）**：v07 state bank 中的完整训练态（`full_state` v0.2 格式）：
  model（fp32）+ AdamW m/v/step + scaler（bf16 下为空）+ 四路 RNG + metadata。
  - smoke 用：seed 2030 / state_step 064（metadata: `global_micro_step=128`, `total_micro_steps=256`,
    `planned_total_optimizer_steps=128`, `partition_sha256=5ecaf0a3…`）。
  - 禁止重新训练生成 C0；只读加载，每次记录文件 SHA256。
- **B（continuation budget）**：64 optimizer steps = 128 micro-batches（accum=2）。
- **数据**：D0 = `e3c_subset.jsonl`（4096 行，SHA256 冻结值 `1e657af9…b0f4`）。
  partition（train/probe/eval 互斥，SHA 校验加载）。**eval pool（128 样本）从未参与 C0 训练**，
  作为 validation；probe pool 不使用。

## 2. 续训训练循环（逐行复刻 v07 bank 制备循环）

与 `e3c_prepare_state_bank_v07.py:170-191` 完全一致的原生时序：

1. 每 micro-batch 前：`set_native_micro_lr(optimizer, get_lr, gm, total_micro, base_lr)`
   （gm = C0 的 `global_micro_step` + k，k=1..128）；
2. `stack_batch` → bf16 autocast forward（`(res.loss + res.aux_loss)/accum`）→ `scaler.scale(loss).backward()`；
3. accum 边界：`unscale_` → `clip_grad_norm_(1.0)` → `scaler.step` → `scaler.update` → `zero_grad(set_to_none=True)`。

**LR schedule（smoke 冻结口径）**：`mode=original_axis` — 沿预注册轴继续：
`lr(gm) = get_lr(gm, total_micro=256, base_lr=5e-4)`（cosine 衰减至峰值 10%）。
所有臂共享同一 schedule、同一 batch 流。*"changed budget" 的 schedule 重推口径
（如 replanned cosine）属 P4 冻结项，smoke 不实现、不解读。*

**Batch 流（CRN）**：`batches_from_train_pool(partition.train_indices, batch_size=4,
seed=train_seed+1704, needed=(t0+B)×accum)` 后取切片 `[t0×accum:]` ——
即原轨迹在该位置本会消费的、确定性的后续 batch。所有臂共享同一 manifest（SHA256 记录并断言一致）。

## 3. 实验臂（唯一定义；config 与 state_adapter 双向一致性断言）

| 臂 | betas | m | v | step 计数 | 目的 |
|---|---|---|---|---|---|
| **A** Strong continuation | (0.9, **0.999**)（原值，不动） | 继承 | 继承 | 继承 | 强 baseline（P4 才做 LR/WD 网格） |
| **B** HP migration | (0.9, **0.99**) | 继承 | 继承 | 继承 | 改 HP + 陈旧 state 的失配观测 |
| **C1** State adaptation | (0.9, 0.99) | 继承 | **清零** | 继承 | 单通道 state 适配（重置 v） |
| **C2** State adaptation | (0.9, 0.99) | **清零** | 继承 | 继承 | 单通道 state 适配（重置 m） |
| **D** Reset baseline | (0.9, 0.99) | 清零 | 清零 | **归零** | changed-HP 下全新 optimizer 下界 |

**已冻结的定义性说明（不解读、不调整）**：

- `optimizer.load_state_dict` 会用 checkpoint 的 param_groups 覆盖新 optimizer 的
  betas → 改 β 必须发生在 load **之后**（`state_adapter.apply_arm` 强制此顺序，
  Test 2 专项验证）。
- C1/C2 的 step 计数**保持 t0**（任务定义未提及者一律不动）。其 bias-correction
  后果（如 C1 首步 v̂ = (1−β2)g²/(1−β2^(t0+1))，分母远小于 1，产生大有效步长瞬态）
  是定义的一部分，属于被测对象，不是 bug。
- weight_decay 全臂 0.01（原值不动）；A 臂的 LR/WD 调优是 P4 内容，smoke 一律用
  schedule 原值。
- 不实现任何"聪明"的 moment 重构；C 臂只有清零式变体。

## 4. 评价

- **Primary**：final validation NLL —— 64 步耗尽后在 eval pool 上用 vendored
  `evaluate_token_losses`（逐 token CE，mean/p95/p99/CVaR99；no-grad、eval 模式、
  RNG 捕获/恢复）测得的 `mean`。
- **Secondary（诊断，不作判据）**：逐步 train loss、pre-clip grad norm、
  ‖m‖/‖v‖（全局 + 分参数组）、有效学习率分布（`eff_lr_i = lr/(√v̂_i + eps)` 与
  F0 口径增益 `G_i = lr(1−β1)/((1−β1^t)(√v̂_i+eps))` 的 mean/median/p10/p90）。
- 另在 C0 时刻（step 0）测一次 eval NLL（各臂必须完全一致——作为正确性断言）。
- **不以短期 loss 评价成败**（smoke 只报告完成度/正确性/可复现性，不解读性能）。

## 5. 确定性与 CRN 设计

- 每次运行：fresh model/optimizer → `load_full_training_state`（model→optimizer→scaler→RNG 顺序，RNG 最后恢复）→ `apply_arm`（state 适配本身不消耗 RNG）。
- 所有臂从同一 RNG 状态、同一 batch manifest 出发；模型 dropout=0.0；
  `PretrainDataset.__getitem__` 无随机性 → 臂间唯一差异 = optimizer state/betas。
- cudnn deterministic 标志经 `setup_seed` 设置；CUDA RNG 恢复要求运行时可见设备数 = 保存时 = 1。
- smoke 附加 `A_repeat` 运行：同 checkpoint 同臂重复完整 64 步，要求模型/optimizer
  哈希与 loss 曲线逐位一致（隐藏非确定性检测）。

## 6. 必需测试（GPU 实验前全部通过）

1. **Full state roundtrip**：save → 破坏（继续训练 + 消耗 RNG）→ load →
   model/optimizer 哈希逐位一致 + 未来随机数逐位重放。
2. **Beta modification correctness**：(a) 证明 `load_state_dict` 确实会用旧 betas
   覆盖（陷阱存在性）；(b) load 后改 betas 生效且 m/v/step 不被触碰；
   (c) 手推 AdamW 单步数学验证 β2=0.99 实际参与更新；(d) 全臂 state 语义
   （清零通道、保留通道、step）逐项验证。
3. **CRN reproducibility**：真实 C0（seed 2030/step 64）同臂两次独立运行首个
   optimizer step：loss、梯度范数、参数位移哈希逐位一致；batch manifest SHA 一致。

## 7. smoke 范围与产物

- 范围：1 seed（2030）× 1 checkpoint（step 064）× 臂 {A, A_repeat, B, C1, C2, D} × 64 步。
- 产物：`outputs/smoke_v01/runs/<arm>/{metrics.json, loss_curve.csv, optimizer_state_statistics.csv}`
  + `outputs/smoke_v01/smoke_report.json`（完成度/eval 可用性/臂间轨迹可分性/确定性/异常，不解读性能）。

## 8. 红线

- 不修改 `legacy_fluctuation_response/`、`minimind_base/`、`optimizer_sensitivity/`；
- 不引入新 optimizer / GTTR / hypergradient / HVP / 二阶求解 / controller / 理论机制；
- 本阶段只做 checkpoint continuation + optimizer state manipulation；
- 不依据 smoke 结果调整任何口径；P4 矩阵未经评审不得启动。
