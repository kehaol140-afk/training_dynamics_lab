# P4_PROTOCOL_REVIEW — 协议冻结评审报告

评审日期：2026-09-26。对象：`docs/P4_PROTOCOL.md` + `configs/p4_full.json`
（冻结 SHA256 `b4716c8efe6c7e490f4eed8f2b419505cc624fc9643730dfd6a5581d723a8fe2`）。
结论：**协议可评审通过；P4 未启动，等待批准。**

自检已通过：56 cells 唯一、预算表逐 cell 核算（t_remaining=128−t0、B=T_rem/2、
micro=2B 全部成立）、臂表与 smoke 冻结值逐字段相等、SHA 抽查 3/56 与磁盘一致、
LR 端点恒等式 2t0+4B=256 全 cell 成立、置换检验最小可达 p=0.0039 < Holm 最小阈
值 0.025（功效可达）。

---

## 1. 全变量冻结清单

| 变量 | 冻结值 | 位置 |
|---|---|---|
| Checkpoint 集 | 56 = 8 seeds × 7 步位；逐 cell SHA256 | config `checkpoints.cells` |
| 预算规则 | B = (128−t0)/2，逐 t0 表（23–41 steps） | §2 + config `budget_rule` |
| LR 变换 | `compressed_remaining_cosine_v1`：gm_lr(k)=2t0+2k，lr=m·lr(gm_lr)，终点恒为 5e-5 | §3 + config `lr_schedule_transform` |
| 臂 | A/B/C1/C2/D（betas、m/v/step 通道、乘子来源），与 smoke 逐字段一致 | §4 + config `arms` |
| A 调参 | 网格 {0.5,1,2}；判据=final V_monitor NLL；并列→1.0×；边界自动单次扩张 {0.25,4}；选中值传 B/C/D | §5 + config `arm_A_tuning` |
| 评测 | Primary=final V_final NLL（128 样本封存池）；V_monitor=probe(32)；两段式封存 | §6 + config `evaluation` |
| 统计 | 配对 d=Cx−A_best；族={C1,C2} Holm α=0.05；primary=按 seed 聚类 sign-flip 置换（20000，RNG 77080001）+ 聚类 bootstrap（10000） | §7 + config `statistics` |
| 停止条件 | 继续 5 项全满足；停止 6 项任一触发 | §8 + config |
| 执行顺序 | Stage 0→1a→1b→1c→2→3，SEAL_LOCK 机制 | §9 |
| 确定性抽查 | s2030_t064 × A×1.0 重跑位级一致 | config `determinism_spot_check` |
| 技术失败阈值 | >4/56 cells → 暂停复核 | §8 |

## 2. 歧义审计（发现并已消除的决策点）

1. **B 是常数还是逐 checkpoint 变化？** 冻结为逐 cell：B=(128−t0)/2（23–41
   步）。任务原文"Changed-budget continuation… B = T_remaining/2"在 56 个
   checkpoint 上必然产生不同 B，已显式列表消除隐含分歧。
2. **"压缩原剩余曲线"的离散映射是否精确？** T_rem/B 恒等于 2（7 个 T_rem
   全偶），映射 gm_lr(k)=2t0+2k 无插值、无舍入；端点恒等于 256（schedule 在
   B 内完整收尾），已给出工作示例。
3. **LR 消费语义**：沿用原生"累积边界最后 micro-batch 写入值"，与 smoke/
   bank 制备一致（`native_schedule` 语义），无新口径。
4. **V_monitor vs V_final 的池分配**：冻结 V_monitor=probe(32)、V_final=
   eval(128)，两者均从未参与 C0 训练（partition 冻结文件保证）；不切分、
   不复用、不新造池。
5. **A_best 的定义**：含自动单次扩张后、按 final V_monitor NLL 选中的 A run
   的 V_final 值；"boundary_unresolved"只作标记不再扩格——消除"边跑边扩格"
   的自由度。
6. **C 的两个变体如何进入推断**：{C1, C2} 构成一个 Holm 族（消除"挑好的
   变体"的多重比较漏洞）；一致性门槛对通过 Holm 的变体逐项检查。
7. **cell 间相关性**：同 seed 7 个 checkpoint 同轨迹、续训流切片重叠
   （t0=46→[92:174)，t0=52→[104:180)，…），cell 独立假设不成立——primary
   检验按 seed 聚类（8 簇），naive cell-level CI 仅作参考并标注反保守。
8. **C1/C2 的 step 计数**：保持 t0（未提及者不动）；其 bias-correction 瞬态
   属被测定义。与 smoke 一致，无新决策。
9. **tie 与并列判定**：≤1e-6 判并列，优先 1.0× 再取小值——消除选择随机性。
10. **失败 run 处置**：只允许原样重跑（确定性保证一致），禁止改参重跑。

## 3. 泄漏审计（tuning leakage）

- **结构性防泄漏**：runs 阶段不计算 V_final（metrics.json 无该字段）；
  A 选择只能用 V_monitor；V_final 仅在全部 runs+选择完成后由独立脚本从
  final_state.pt 一次性计算，SEAL_LOCK + SHA 台账留痕。选择阶段接触 V_final
  在结构上不可能。
- **无事后自由度**：网格与扩张规则、并列规则、判据在运行前冻结；扩张自动
  触发且仅一次；B/C/D 不调参；停止条件不可逆（任一触发即停，不允许加方法）。
- **多重比较受控**：primary 族={C1,C2} Holm；其余比较（B/D 系列、C1−C2）
  明示为描述性。
- **单一方向假设**：置换检验单侧（改善方向 d<0）预注册，避免事后选侧。

## 4. 残余风险（知情接受，不构成协议修改理由）

1. **V_monitor 仅 32 样本（≈3.8k tokens，SE≈0.04 nats）**：A 的乘子选择存在
   选择噪声；配对（同 cell 同流）与并列规则缓解，但不能消除。若未来复审核
   实选择噪声主导，正确动作是接受阴性结果，不是换池。
2. **cluster 数仅 8**：primary 检验最小 p=0.0039，对中等效应量功效有限；
   这是 8-seed bank 的先天上限（不再增加 checkpoint/seed，避免生成新数据）。
3. **同 seed cell 相关性**：checkpoint 一致性门槛（5/7）在相关样本上偏宽松，
   故已置于显著性门槛（按 seed 聚类）之后联合生效。
4. **A 网格 unresolved-boundary**：若大量 cell 扩张后仍落边界，说明乘子网格
   覆盖不足，报告须显式给出占比；该情形下 C>A 的结论强度相应降级（在报告中
   标注，不改变停止规则的机械执行）。
5. **h256/4L 极小模型 + ≤41 步预算**：结论只裁决本方向是否值得继续，不支持
   任何规模外推主张。
6. **工程缺口**（§10 已预登记）：runner 扩展、sealed-eval 脚本、分析脚本
   尚未编写；须先实现并单测（含 LR 映射端点断言、乘子传递、SEAL_LOCK 流程）
   再开跑，期间不得触碰本协议。

## 5. 裁决

- 8 项必需内容（checkpoint/预算/schedule/臂/调参/评测/统计/停止）全部冻结且
  机器可核（config + 自检通过）；
- 已知歧义 10 项全部消除；未发现隐藏调参通道；
- **P4 GPU 矩阵未启动**。等待批准；批准后先完成 §10 工程项与单测，再进
  Stage 0。
