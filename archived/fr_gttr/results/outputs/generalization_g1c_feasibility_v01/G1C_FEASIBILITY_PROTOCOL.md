# G1C Feasibility Protocol — FR_G1C_1B_FEASIBILITY_GATE_V01

> 预注册冻结：2026-09-17T02:15:50Z（任何 GPU 科学计算与任何 Phase 1/2
> 响应结果之前写成）。本任务是 **feasibility gate**，**不是正式 G1c 科学
> 实验**。禁止生成任何泛化结论；禁止 scale claim；禁止 generalization
> claim；禁止"OLMo-1B response validated"类表述（模型家族已切换为
> OLMo-2-0425-1B，见 §2）。
> 指标一律严格 import 冻结的 `fr.a0c_adamw_response_propagation_v01` 的
> `_metric` / `relative_error`，禁止重定义；冻结制品一律只读。

## 1. 任务目标与判定

唯一目标：验证 OLMo-2-0425-1B 官方 standard checkpoint 是否具备进入
G1c 科学响应实验（FD/B/C 有限时间响应）的条件。

- Phase 0 失败 → 停止，标记 BLOCKED。
- Phase 1 失败 → 停止，标记 BLOCKED。
- Phase 2 只输出 feasibility 描述矩阵（R2/corr/MAE/relative error），
  **不**做论文式结论、**不**判定 EXTERNAL_STRESS_SUPPORTED（那是正式
  G1c 实验 B14 的冻结标签，本 gate 不使用）。
- 全部通过 → 标记 **READY_FOR_G1C_SCIENCE**（可执行性声明，非科学结论）。

## 2. 模型与制品（B1/B2 的合法继承）

- 外部模型：**OLMo-2-0425-1B**（stage1 run `peteish1`，1.24B params，
  Apache-2.0，官方 standard unsharded checkpoint）。原 primary OLMo-1B
  已在 2026-09-16 直链复核中被取代：官方 CSV 证明 peteish1 standard
  checkpoint 的 model+optimizer+trainer state 完整公开（复核制品
  `outputs/generalization_g1c_checkpoint_recheck_v01/`，判定
  FULL_STANDARD_CHECKPOINT_AVAILABLE）。此切换在查看任何科学结果之前
  完成，符合冻结 G1C_PROTOCOL.md §3 的切换规则；理由为"同一 AllenAI
  官方系列中约 1B 且公开完整 optimizer/trainer state 的 checkpoint"。
- checkpoint 目录（自复核 manifest 逐字继承，先冻结）：
  `https://olmo-checkpoints.org/ai2-llm/peteish1/step{380000,950000,1530000}-unsharded/`
- Phase 0 只下载 **step950000-unsharded** 四件（config.yaml / model.pt /
  optim.pt / train.pt）；step380000 / step1530000 仅在 Phase 1 PASS 后
  下载（B3 下载策略）。

## 3. Phase 0 — checkpoint restoration gate

仅 step950000，仅验证可恢复性：

1. `model.pt` 能否恢复全部模型参数（key/shape 与自建架构逐一对齐，
   fail-fast：任何不匹配 → CONTRACT_INCOMPATIBLE → BLOCKED）。
2. `optim.pt` 能否恢复真实 AdamW optimizer state：逐参数确认
   `exp_avg`、`exp_avg_sq`、`step` 存在且形状与参数一致。
3. `train.pt` 是否包含 training step、scheduler state、rng state。
4. 记录：checkpoint 三文件 SHA256、config.yaml SHA256、optimizer
   metadata（lr/betas/eps/weight_decay 逐 group、param group 数、
   各 group 参数名覆盖、max_grad_norm、amsgrad 检查）。
5. 指标性核对（非科学）：m/v 范数、step 值、lr 与 config 一致性。

输出：`G1C_PHASE0_REPORT.md`。任一失败 → 停止。

## 4. Phase 1 — single-state response smoke（step950000 × 1 stream × H=1）

依次验证：

1. **deterministic replay**：同一 batch 连续两次前向+反向，loss 与梯度
   位级一致（记录 max abs gap，要求 0.0）。
2. **gradient propagation**：基线 microbatch 梯度聚合、全局 clip
   （max_grad_norm=1.0，torch 语义）可执行，clip 诊断量记录。
3. **H=1 analytic identity**：B 与 C 切线在 H=1 与解析一步预测一致
   （B≡C 于 H=1 结构性恒等，记录 |B−C| gap）；B/C 对 chi_fd 的相对
   误差记录（仅 smoke 数值，不外推）。
4. **HVP/double backward 可执行性**：math-backend attention 的
   reverse-over-reverse HVP 在 seq=4096、bf16 autocast 下完成且不 OOM；
   记录峰值显存、HVP vs native 梯度的一致性检查量。

禁止扩大实验（不做 H>1、不加 stream、不加 checkpoint）。

输出：`G1C_PHASE1_REPORT.md`。任一失败 → 停止。

## 5. Phase 2 — feasibility matrix（仅当 Phase 1 PASS）

- checkpoint：step380000 / step950000 / step1530000（3）
- future streams：2（冻结 seed，见 §7）
- H：{1, 4, 8}
- 总 **18 cells**。每 cell 记录 chi_FD / chi_B / chi_C、abs/rel/signed
  error（冻结 `relative_error`），聚合 C vs B 的 R2 / corr / MAE /
  relative error（冻结 `_metric`，描述性）。
- **禁止**：论文式结论、scale claim、generalization claim、
  EXTERNAL_STRESS_* 标签、跨 checkpoint 趋势断言。

## 6. 冻结的方法学细节（先于任何结果）

**6.1 q 方向与 epsilon**：每 stream 独立 q = 全参数 iid N(0,1)（full
support；support 规模 = 全参数，禁止随机子采样参数——红线）。σ_q = q
元素标准差（目标 1.0，实测记录）；ε = 0.25 × σ_q（RHO=0.25 继承 v07/
G1a/G1b 规则）。δ_g = q 单位脉冲加于 pulse 步梯度。

**6.2 future stream（B5 条件命名）**：官方 Dolma v2 预训练语料的
dolma2-tokenized `.npy`（olmo-data.org，stage1 YAML `paths:` 中登记的
proof-pile-2 分片）。本轮无法从 trainer state 恢复原始未来数据顺序 →
明确称为 "**conditional response under frozen in-distribution future
streams**"，不得称 original training trajectory replay。每 stream =
从单一分片冻结偏移处取连续 token 窗（≥ H_max × 4 microbatch × 4096 =
131,072 tokens）；stream 间偏移不同、seed 冻结。记录分片 URL、字节
range、SHA256、npy header（shape/dtype）。

**6.3 optimizer step 口径（登记的偏差）**：每未来步 = 4 个 microbatch
× 4096 tokens 梯度累积后一次 clip+AdamW 步。**不是**原生
global_train_batch_size=512 的完整批次（显式登记为 feasibility 口径
偏差；tokenizer/objective/loss/corpus 不变）。

**6.4 loss 可观测量 X**：窗内全部 token 的 mean cross-entropy
（token-averaged，bf16 autocast 前向、fp32 loss 累加）。

**6.5 LR schedule**：优先使用 train.pt 中 scheduler state 逐步恢复 lr；
若 scheduler state 不可解析，回退为常数 lr = checkpoint param_groups
恢复值（记录在案）。

**6.6 精度**：amp_bf16 autocast（与 stage1 `precision: amp_bf16` 一致），
master fp32。

**6.7 内存策略（B7）**：optimizer m/v 与 master 参数常驻 CPU；GPU 仅
承载计算副本（fp32 参数 + fp32 梯度 + 检查点化激活）；attention 用
math backend（支持 double backward；native flash 无双反向，同冻结
caveat 逻辑）；CE logits 分块重计算。禁止 quantize / low-rank HVP /
truncate layers / LoRA / freeze layers。

**6.8 AdamW 切线 adapter（B4/B5 授权）**：冻结
`adamw_tangent_step` 的 `single()` 齐性检查是 MiniMind 单 group 假设。
若 optim.pt 存在多 param group（异质 weight_decay），启用版本化、
经测试的 adapter `g1c_adamw_tangent_step`（数学与冻结实现逐行一致，
仅将齐性聚合改为逐 group 返回；单元测试：与冻结实现在齐性 group 上
逐位一致 + 朴素参考实现在异质 group 上逐位一致）。若实际为单 group，
仍用冻结原函数。

**6.9 红线**：不得修改/覆盖 A0A–A0G、A0H/A″、5M anchor、G1a/G1b/G2/G3
任何冻结 CSV/JSON/checkpoint/q/verdict；不得重训练或微调 1B；不得因
结果不好切换 checkpoint；所有产物进本版本化目录；完成后不自动启动
正式 G1c 科学实验（须用户新授权）。

## 7. 冻结 seed / 常量

- stream seeds（两个 future stream）：derive_stream_seed(20260917, 1)
  与 (20260917, 2)（`fr.stream_rng.derive_stream_seed`，与 v07 同函数）。
- q seed per stream：derive_stream_seed(stream_seed, 77)。
- npy 分片与偏移：Phase 2 前从 stage1 YAML `paths:` 冻结选择并记录
  （先选分片再取流，不因下载失败换流）。

## 8. 产物

`outputs/generalization_g1c_feasibility_v01/`：本协议、experiment_config
.json、checkpoints/（三 step 四件 + SHA256）、code/（architecture /
adapter / runner，含单测）、results/（per_cell.csv、summary.json、
phase 报告 ×2、DECISION.md）、G1C_FEASIBILITY_REPORT.md、COMMANDS.log、
ENVIRONMENT.json、SOURCE_HASHES.txt、SHA256SUMS.txt。
