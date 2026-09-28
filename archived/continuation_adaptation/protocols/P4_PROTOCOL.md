# P4_PROTOCOL — Optimizer State Adaptation under Changed Compute Budget
# （大规模淘汰实验协议，冻结版）

冻结日期：2026-09-26。冻结件：本文档 + `configs/p4_full.json`（SHA256
`b4716c8efe6c7e490f4eed8f2b419505cc624fc9643730dfd6a5581d723a8fe2`）。
**Stage 1a 启动后，两者任何字段不得修改。** 本协议在 schedule 定义上取代
`docs/EXPERIMENT_PROTOCOL.md` §2 的占位说明（smoke 协议本身不变）。

研究问题（冻结，不可为结果修改）：

> Transformer checkpoint 在改变计算预算（B = T_remaining/2）与改变 AdamW
> 超参时序（压缩 LR schedule、β2 迁移）下续训时，继承的 optimizer state
> (m, v, step) 是否仍然适配？简单清零式 state 适配（C1/C2）能否优于认真
> LR 调参的强续训 baseline（A）？

这是**证伪实验**：阴性结果是可接受的最终结果。停止条件见 §8。

---

## 1. Checkpoint 集（冻结）

- 来源：v07 state bank（只读，禁止生成新 checkpoint）：
  `legacy_fluctuation_response/outputs/e3_confirmatory/v07/`
- **56 个 cell = 8 seeds（2030–2037）× 7 步位（46,52,58,64,70,76,82）**，
  每个 C0 含 model（fp32）+ AdamW m/v/step + scaler + 四路 RNG + metadata。
- 逐 cell 的 `state_relpath` 与 **SHA256 已全部登记**于 `configs/p4_full.json`
  的 `checkpoints.cells`（56 条，冻结时逐文件实测）。
- 通用事实（抽验 seed_2037/t082 与 seed_2033/t046 一致）：
  `global_micro_step = 2·t0`、`total_micro_steps = 256`、
  `planned_total_optimizer_steps = 128`、`partition_sha256 = 5ecaf0a3…`。
- 每 seed 的 train batch 流种子：`train_batch_seed = seed + 1704`
  （实测 2030→3734 … 2037→3741，与 bank 报告一致）。
- 语料 D0（4096 行，SHA `1e657af9…b0f4`）与 partition（SHA 校验加载）不变。

## 2. Continuation Budget（冻结）

规则名：`changed_budget_half_remaining_v1`。对每个 checkpoint：

```
T_remaining = 128 − t0          （原计划剩余 optimizer steps）
B           = T_remaining / 2   （新预算，所有臂同 cell 同 B）
micro steps = 2B                （accum = 2）
```

| t0 | T_remaining | B (opt steps) | micro steps |
|---|---|---|---|
| 46 | 82 | 41 | 82 |
| 52 | 76 | 38 | 76 |
| 58 | 70 | 35 | 70 |
| 64 | 64 | 32 | 64 |
| 70 | 58 | 29 | 58 |
| 76 | 52 | 26 | 52 |
| 82 | 46 | 23 | 46 |

（7 个 T_remaining 均为偶数，除法精确。）注意：P4 各 cell 的 B 与 smoke
不同（smoke 只做实现验证，B=64 且不压缩）；P4 不复用 smoke 的数值结论。

## 3. LR Schedule Transformation（冻结）

规则名：`compressed_remaining_cosine_v1`（schedule remapping）。

- 原生 cosine：`lr(gm) = base_lr · (0.1 + 0.45·(1 + cos(π·gm/256)))`，base_lr=5e-4。
- 该 checkpoint 的"原剩余 LR 曲线"= 原计划在 gm = 2t0+1 … 256 上将要施加的 LR 序列。
- **压缩映射**：续训第 k 个 micro step（k = 1..2B）的 LR 取原曲线在比例位置
  k/(2B) 处的值：

```
gm_lr(k) = 2·t0 + (T_remaining/B)·k = 2·t0 + 2k      （每个 t0 的比值都恰为 2）
lr_cont(k) = lr_multiplier · lr(gm_lr(k))
```

- 端点性质：k=2B 时 `gm_lr = 2t0 + 4B = 2t0 + 2·T_remaining = 256`，即
  **每个 cell 的 schedule 都在 B 内完整走完**，终点 LR = 0.1·base_lr = 5e-5。
- 工作示例（cell s2030_t046，t0=46，B=41）：k=1 → gm_lr=94 → lr=3.663e-4；
  k=82 → gm_lr=256 → lr=5e-5。
- LR 仍按原生时序逐 micro-batch 写入（`native_schedule.set_native_micro_lr`），
  optimizer 实际消费累积边界最后一个 micro-batch 写入的值。
- **所有臂共享同一 base schedule 与同一 batch 流（CRN）**；唯一差异来源：
  ① 臂定义的 state/betas；② arm A 网格期的 lr_multiplier（选中值随后同 cell
  全臂沿用）。**不做任何跨臂 schedule 对比**（本研究 state 适配，不研究
  scheduler 设计）。

## 4. 实验臂（冻结，与 smoke 一致）

| 臂 | betas | m | v | step | lr_multiplier |
|---|---|---|---|---|---|
| A Strong continuation | (0.9, **0.999**) | 继承 | 继承 | 继承 | 网格 {0.5, 1.0, 2.0} 调优 |
| B HP migration | (0.9, **0.99**) | 继承 | 继承 | 继承 | = 该 cell 选中的 A 乘子 |
| C1 State adaptation | (0.9, 0.99) | 继承 | **清零** | 继承 | 同 B |
| C2 State adaptation | (0.9, 0.99) | **清零** | 继承 | 继承 | 同 B |
| D Reset baseline | (0.9, 0.99) | 清零 | 清零 | **归零** | 同 B |

- 改 β 必须在 `load_state_dict` 之后（`state_adapter.apply_arm` 强制并回读验证）。
- C1/C2 的 step 保持 t0（任务定义未提及者不动）；其 bias-correction 瞬态
  （如 C1 首步 v̂ 极小 → 有效步长激增）是定义的一部分，属被测对象。
- weight_decay 全臂 0.01 不动；grad_clip 1.0 不动；训练循环逐行复刻原生时序。
- **禁止**实现 moment 重构 / 历史回放 / kernel 估计 / 自适应方法。

## 5. 调参规则（Arm A，冻结）

- 网格：lr_multiplier ∈ {0.5, 1.0, 2.0}（作用在压缩后 schedule 上）。
- **选择判据：该 cell 预算耗尽时的 final V_monitor NLL（最低者胜）**。
- V_monitor = partition `probe_indices`（32 样本，从未参与 C0 训练）；
  **任何阶段不得读取 V_final**（结构性保证：runs 不计算 V_final，见 §6）。
- 并列规则：差 ≤ 1e-6 视为并列 → 优先 1.0×，再取较小乘子。
- 边界扩张：若最优在边界（0.5× 或 2.0×）→ **自动且仅一次**扩展
  （0.25× 或 4.0×）。扩张后仍落边界 → 接受该值并标记
  `boundary_unresolved=true`（计入报告，不得二次扩张）。
- 下游传递：选中乘子（含扩张后）同 cell 应用于 B/C1/C2/D；B/C/D 不另行调参。
- 时序：每 cell 的 B/C/D runs 必须在该 cell 选择完成之后启动。
- 监控评测节奏：每 8 个 optimizer step + 最终步（B 非 8 倍数时含 B 本身）；
  `best_monitor_nll` = 节奏点上的最小值（诊断字段，不作判据）。

## 6. Evaluation Protocol（冻结）

- **Primary = final validation NLL**：预算耗尽时 V_final pool 上的逐 token CE
  均值。不是 early loss、不是 recovery speed、不是 training loss。
- **V_final = partition `eval_indices`（128 样本，从未参与 C0 训练）**。
- **封存机制（两段式）**：
  - Stage 1（全部 runs）：每 run 落盘 metrics.json（**不含任何 V_final 字段**）、
    loss_curve.csv、optimizer_state_statistics.csv、final_state.pt（full_state
    格式，供封存评测）。runs 期间只计算 V_monitor。
  - Stage 2（全部 runs + 全部选择完成后）：专用 sealed-evaluation 脚本从
    final_state.pt 一次性计算 V_final NLL，输出带 SHA256 台账的封存结果；
    执行前先写 `outputs/p4_full/SEAL_LOCK.json`（run 清单 + 各 metrics SHA）。
- **Secondary（仅解释性，禁止用于成功宣称）**：loss 曲线、pre-clip grad norm、
  ‖m‖/‖v‖、有效学习率分布（efflr = lr/(√v̂+eps)；gain 口径见协议 P1）、
  update norm ‖u‖（u = vec(θ_final − θ_C0)）、cos(u_A, u_X)（X∈{B,C1,C2,D}，
  封存阶段计算）、monitor NLL 曲线、recovery speed。
- 每 run metrics.json 至少含：final_monitor_nll、best_monitor_nll、seed、
  checkpoint_step、arm、lr_multiplier、budget_steps、batch_manifest_sha256、
  model_hash_final。

## 7. Statistical Analysis（冻结）

配对单元 = cell（seed × checkpoint）。对每个 primary 比较
（**C1−A_best、C2−A_best 两个，Holm 校正族**）：

```
d(cell) = NLL_final(Cx, V_final) − NLL_final(A_best, V_final)      （改善 < 0）
```

- **关键结构事实**：同一 seed 的 7 个 checkpoint 来自同一条轨迹，且各 cell
  的续训 batch 流切片互相重叠（如 t0=46 取 [92:174)，t0=52 取 [104:180)），
  cell 间差异在 seed 内强相关 → **primary 推断按 seed 聚类**。
- Primary 检验：8 个 seed 级均值 d̄_s 的配对 sign-flip 置换检验
  （单侧，改善方向 d<0；20000 次；重采样 RNG 种子 77080001）。
  最小可达 p = 2⁻⁸ ≈ 0.0039 < Holm 阈值 0.025，功效可达。
- Primary CI：按 seed 聚类 bootstrap（10000 次，percentile）的均值配对差 CI。
- 描述性：56 个 d 的 mean/median/std；逐 seed 均值（8 值）；逐 checkpoint
  均值（7 值）；naive cell-level bootstrap CI（标注"反保守，仅参考"）。
- 次要比较（B−A_best、D−A_best、B−D、C1−C2）：仅描述，不设门槛。
- A_best 定义：该 cell 经冻结规则（含自动单次扩张）选中的 A run 的 V_final 值。

## 8. Stop Conditions（冻结）

**继续研究条件（须全部满足，对 C1 或 C2 至少一个成立）**：

1. 聚合：mean(d) < 0（final V_final NLL，C < A_best）；
2. Seed 一致性：≥ 6/8 seeds 的 d̄_s < 0；
3. Checkpoint 一致性：≥ 5/7 步位的 d̄_t < 0；
4. 统计显著性：Holm 校正后 cluster 置换 p < 0.05；
5. 网格稳健：优势相对**扩张后**调参的 A_best 仍然存在（扩张在 Stage 1a
   自动完成，不存在事后扩格空间）。

**任一触发即停止（写终局报告，不追加轮次、不加复杂方法救场）**：

- C ≤ A_best（聚合 final NLL）；
- 改善仅出现在 early loss / recovery speed / monitor 曲线而非 final NLL；
- 改善局限于单一 seed（<6/8）或单一 checkpoint（<5/7）；
- LR 调参（扩张后 A_best）使优势消失；
- Holm 校正后两个 C 变体均不显著；
- 技术失败率 > 4/56 cells（暂停复核；除非不改协议可修复，否则按停止处理）。

## 9. 执行顺序与治理

```
Stage 0   封存前置 + 确定性抽查（cell s2030_t064 × arm A ×1.0 重跑位级一致）
Stage 1a  A 网格：3 乘子 × 56 cells（+ 自动单次边界扩张）
Stage 1b  每 cell 按 V_monitor 选择，产出 selected-multiplier 台账
Stage 1c  B/C1/C2/D：4 臂 × 56 cells（用选中乘子）
Stage 2   SEAL_LOCK → 封存评测（V_final 一次，SHA 台账）
Stage 3   单遍分析，对照 §7/§8 出裁决报告
```

- 技术失败定义：异常退出、loss/monitor NLL 出现 NaN/Inf、确定性抽查失败。
  失败 run 允许原样重跑（确定性保证一致），不允许改参数重跑。
- 预估规模：392–448 runs（单卡 4–6 h，h256 模型 ~30 s/run）；
  final_state.pt 约 25MB/run ≈ 10GB（磁盘余 246G，充裕）。
- 红线（同前）：冻结区只读；无新 optimizer/GTTR/hypergradient/HVP/二阶/
  controller/重构类方法；口径不为结果修改；secondary 不得作为成功依据。

## 10. Stage 1 前的实现缺口（预登记，均不改本协议）

以下为已批准、尚未编写的工程项（P4 实现阶段完成并单测后方可开跑）：

1. runner 扩展：per-cell 预算表、`compressed_remaining_cosine_v1` schedule、
   lr_multiplier、monitor 评测钩子（每 8 步 + 终步）、final_state.pt 保存、
   去除 run 内 V_final 计算；
2. sealed-evaluation 脚本（Stage 2，SEAL_LOCK + SHA 台账 + update 向量/
   cosine 计算）；
3. 分析脚本（Stage 3，§7 口径硬编码，单遍执行）。
