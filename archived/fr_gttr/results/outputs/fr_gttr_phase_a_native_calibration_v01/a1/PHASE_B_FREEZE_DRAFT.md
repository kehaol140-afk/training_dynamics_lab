# PHASE_B_FREEZE_DRAFT.md — v2（修订草案，未获准前不运行）

日期：2026-09-23（v2 取代 v1）。状态：**DRAFT_FOR_APPROVAL**。
不运行 GPU；不改变 GTTR v0.2 理论；不增加其它动作通道（仅 LR 乘性通道）。
v1→v2 修订：M0 废除（原为"u=0 真实臂"，概念错误）；真实臂固定 3 条；
评分模型与真实臂分离；评分算法唯一化、可执行化；预算按 3×6×1×64 重建。

---

## 1. 固定状态 manifest（不混用）

| state_id | seed | n0 | bank 状态 | V(h=0) 相对 ell | 流（每状态 1 条） |
|---|---|---|---|---|
| S0 | 2030 | 46 | v07 full_state v0.2 | above（+0.269） | 97006 |
| S1 | 2030 | 82 | 同上 | above（+0.165） | 97007 |
| S2 | 2031 | 46 | 同上 | above（+0.261） | 97008 |
| S3 | 2031 | 82 | 同上 | above（+0.158） | 97009 |
| S4 | 2032 | 46 | 同上 | above（+0.290） | 97010 |
| S5 | 2032 | 82 | 同上 | above（+0.212） | 97011 |

- **仅 bank 状态**；h128/h256 A1 snapshot 不进入 manifest、不与 bank 混用。
- 6 个 bank 状态在 h=0 全部 above ell（A1 余量表）；若在 64 步内出现 ≤ell 读数，
  仍按 A1 规则登记为 CALIBRATION_ONLY，不汇总任何"达标节约"。
- **路径条件限定**：一状态一流 ⇒ 设计为 path-conditioned pilot；同一批序内
  M0/动作臂严格配对；**不宣称状态层跨流稳定性**（跨流需另批预算与多流/状态）。
- 流冲突核查已过：已注册 stream_id 仅 97000–97005；97006–97011 命中均为
  长数字/哈希子串。

## 2. 命名与角色（v2 冻结）

| 名称 | 角色 | 说明 |
|---|---|---|
| **NATIVE** | 真实臂 | 不做任何干预；LR = 冻结调度值（含声明尾部） |
| **LR_DOWN** | 真实臂 | 前 4 个 optimizer updates 乘 e^{−0.05}，之后释放 |
| **LR_UP** | 真实臂 | 前 4 个 optimizer updates 乘 e^{+0.05}，之后释放 |
| **MODEL_GTTR** | 评分模型（预测近似） | 含 GTTR v0.2 所声明的几何反馈项 |
| **MODEL_NO_GEOMETRY** | 评分模型（预测近似） | 同一状态、同一读出、同一设置；**只去掉所声明的几何反馈项**（其余逐位相同） |

- v1 的 "M0 = u=0 真实臂" 废除：无局部几何反馈的预测近似不是真实臂，
  现名 MODEL_NO_GEOMETRY；它不与任何真实臂共享身份。
- 真实臂仅 NATIVE / LR_DOWN / LR_UP 三条；**不设第四条真实臂**
  （u=0 的 native 行为已由 NATIVE 覆盖，重复无科学作用）。

## 3. 动作定义（已明确，冻结）

- LR′_t = LR_sched_t × exp(u_t)，u_t ∈ {−0.05, 0, +0.05}（u_t=0 ⇔ NATIVE）。
- 干预窗：horizon 第 1–4 个 optimizer updates；第 5 步起释放回调度值。
- 读出 horizon：H ∈ {1, 4, 16, 64}（不变）；monitor 点集 {0, 1, 4, 16, 64}，
  h=0 每状态 1 次（3 臂共用），臂内 H 点各 1 次。
- clip / batch / accum / 精度 / optimizer 内部状态不动；诊断不注入。

## 4. GTTR_SCORE —— 唯一可执行评分算法（两模型共骨架，差异仅几何项）

**GTTR_SCORE(state, model ∈ {GTTR, NO_GEOMETRY}, H_int = 64)**：

1. **输入（决策时点 t0 冻结集合）**：bank 状态 (θ₀, m₀, v₀, n0)；冻结调度
   {LR_t}；该状态 SIM 流种子；冻结 ell 与 bracket 常数；A1 已登记的 t0 前
   monitor 历史。除此之外无任何输入。
2. **当前可用时点**：只允许使用 t0 及之前的信息；任何真实候选分支的观测
   不得进入评分（结构上：评分在所有真实臂启动前完成并封存，§5）。
3. **近似梯度场（probe 阶段，SIM 流）**：从 SIM 流（独立种子、与真实流隔离、
   单独计费）取 P=2 个 probe microbatch，做 forward+backward，**不执行
   optimizer step**，θ/m/v/游标/RNG 全恢复（bitwise 校验）。聚合得 ĝ₀。
   真实流游标不动。
4. **几何信息来源（所声明的几何反馈 = 此项）**：MODEL_GTTR 在同 2 个 probe
   microbatch 上做 Q=4 个有限差分 HVP：Ĥδ_q ≈ [ĝ(θ₀+εδ_q) − ĝ₀]/ε，
   δ_q 为 4 个固定方向（冻结种子），ε 冻结；得曲率作用算子 Ĥ 的 4 方向
   近似。MODEL_NO_GEOMETRY：Q=0，Ĥ ≡ 0。**除此项外两模型逐位相同。**
5. **monitor 梯度（评分目标的一次性梯度）**：在 θ₀ 上对 V_monitor 4096 行
   做 1 次 forward+backward（no step，状态恢复），得 ∇ℓ_mon(θ₀)。
   这是评分目标对参数变化的线性敏感度；只此一次，6 状态共用其臂间比较
   （θ₀ 相同）。
6. **optimizer 虚拟传播（内部 horizon）**：从 (θ₀, m₀, v₀, n0) 出发模拟
   AdamW（betas/eps/wd 冻结；bias correction 用 n0+t；**clip 不在虚拟传播
   中重放**——声明近似，clip 系数数据依赖，留作近似风险登记）共
   H_int=64 步。虚拟梯度场：g_t = ĝ₀ + Ĥ(θ_t − θ₀)（GTTR）或
   g_t = ĝ₀（NO_GEOMETRY）。动作臂的 LR′ 前 4 步按 §3，之后调度值。
7. **评分目标**：pred_NLL_u(H) = NLL₀ + Σ_{t≤H} ⟨∇ℓ_mon(θ₀), θ_t − θ_{t−1}⟩，
   NLL₀ 用该状态 h=0 全库读数（真实 monitor，t0 时点）。评分 =
   pred_Δ_u(H) = pred_NLL_NATIVE(H) − pred_NLL_u(H)（正 = 预测动作优于
   native）。虚拟传播零 forward（纯参数空间仿真）。
8. **成本**：每状态 probe = P + 2Q = 10 次 microbatch 梯度（10 fwd + 10 bwd）；
   monitor 梯度 = 128 fwd + 128 bwd；虚拟传播 = 0 fwd。评分只跑一次
   （3 臂、4 H 点全部由同一虚拟轨迹切片）。

## 5. 预测保存与防泄漏协议（硬约束）

- 所有 pred_Δ_u(H)（2 模型 × 3 臂 × 4 H × 6 状态 = 96 个数值 + 完整中间量）
  连同模型版本、输入哈希，在**任何真实评估分支启动前**写入
  `pre_branch_predictions.json` 并记录 sha256 与时间戳；runner 断言
  该文件先于首个真实 update 存在，否则拒绝启动。
- 真实分支的观测（observed outcome）写入独立 `observed_outcomes.json`；
  **真实未来分支标签不回填预测文件**；两文件只许用 state_id/H/arm 键关联。
- SIM 流（probe 批）允许使用，但逐批登记、单独计费（预算表"模型构建"行），
  与真实流物理隔离（不同种子、游标/RNG 恢复校验）。
- 若运行时发现评分不可执行（如 probe 恢复 bitwise 校验失败），**如实降级**：
  只跑 3 条真实臂并登记观测，全程标 **INTERVENTION_ONLY_DESIGN**，
  不得声称 GTTR 模型验证。

## 6. 验证目标与判定（结果前冻结）

- 主问题：pre-action 预测能否正确排序 3 臂的 observed outcome；
  几何反馈项是否降低预测误差。
- 指标（每 state × H）：排序一致（3 臂 observed Δ 的符号/次序 vs pred）；
  预测绝对误差 |pred − obs|；MODEL_GTTR vs MODEL_NO_GEOMETRY 的配对差
  （6 状态 × 4 H，block SE 仅作参考）。
- observed outcome 定义：真实臂 V_monitor 全库 NLL 在 h=H 相对 h=0 的变化；
  臂间配对差用同批序（同流同 seed）逐行配对。native 外部参照（A1，流
  97000–97005）非配对，仅一致性交叉参照。
- 全部 hit/达标语义保持 CALIBRATION_ONLY；不汇总节约。

## 7. 预算表（v2，完整分列）

| 行 | 内容 | planned | cap | forward 口径 |
|---|---|---|---|---|
| 主训练 | 3 臂 × 6 状态 × 64 步 | **1152 updates** | 1280 | 2304 fwd（2 micro/步） |
| smoke | off/on 各 8 步（16 updates）+ 评分管线的 CPU 测试 | 16 updates | 含上行 | 32 fwd |
| 模型构建 | probe 梯度 6×(2+8)=60 microbatch 梯度；monitor 梯度 6×128 eval | 60 + 768 fwd | 900 | 828 fwd（+等量 backward，backward 另计 wall 不计 fwd cap） |
| 评分 | 虚拟传播 2 模型 × 3 臂 × 6 状态 × 64 步 | **0 fwd** | 0 | 纯参数空间仿真；wall 列入预算 |
| 候选重放及恢复 | snapshot 0（bank 状态只读恢复）；失败恢复储备 | 0 | 96 updates 储备 | 恢复重跑按实际计费 |
| 主 monitor 读出 | h=0 共享 6 + 臂×H 点 72 = **78 全库 eval** | 78 evals | 84 | 9984 fwd（128/eval） |
| probes/HVP/V_final | — | 0 / Q=4 FD-HVP 计在模型构建 / 0 | — | 无新 HVP 外部调用、无 final 评价 |
| **合计** | | **1168 updates** | 1280 | **13116 fwd**（cap 14500） |
| GPU wall | steps ~277s + evals ~257s + probe/梯度 ~90s + setup ~100s | ~720s（估） | 5400 | smoke 后按 A1 规程重报实测 ETA |

- 失败与恢复全部计费；不超 cap 则继续，超 cap 保存已产出并报告。
- 预算内无余量支持第四条真实臂或跨流扩展；如未来需要，另行报批。

## 8. 待批准项

1. §2 命名与 3+2 臂/模型结构；2. §3 动作（exp 乘性、前 4 步、H 集）；
3. §4 评分算法全部常数（P=2, Q=4, ε, δ_q 种子, H_int=64, 一阶 monitor 评分、
   虚拟传播不重演 clip 的近似）；4. §5 防泄漏协议与 INTERVENTION_ONLY_DESIGN
   降级规则；5. §7 预算 caps。

**未获明确批准前不运行 Phase B；不 commit/push；不建监视器。**
