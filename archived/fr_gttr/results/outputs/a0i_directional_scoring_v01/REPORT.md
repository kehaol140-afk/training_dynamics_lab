# A0I — Post-A0H20 Algorithm Efficiency Study

- 轮次：A0I（A0H20 之后的算法工程轮，不是新的独立任务效用验收）
- 日期：2026-09-16
- 冻结基线：`outputs/a0h_corrected_accept_v01/`（本轮未修改其中任何文件）
- 产出目录：`outputs/a0i_directional_scoring_v01/`
- 环境：conda `fr_minimind`，torch 2.11.0，CUDA 13.0，物理 GPU1（`CUDA_VISIBLE_DEVICES=1`），双卡全程有外部任务（共享 GPU 条件，未杀任何外部进程）

## 0. 摘要

A0I-A（纯离线、无 GPU、无标签变更）：对 A0H20 冻结标签做适应性与可解释性诊断。
控制器在 12 个冻结单元上动作分布 {j0:1, j1:3, j2:5, j3:1, j4:2}，频繁切换、无单动作支配，
适应性真实存在。但相对 dev-fixed(j=1) 的增益高度异质（3 种子大幅为正、2030 约零、
2031/2032 为负）。事后回看最优固定动作是 j=2（标签 POSTHOC_DIAGNOSTIC_ONLY）。

A0I-B（dev-only 方向导数评分，3 个 dev 状态 × stream 0，H=4，M=4+no-op）：
恒等式 `h_j = D_θL_j[w_0] − w_0ᵀg` 在 math-SDPA backend 下数值等价成立
（gate 54/72 通过，未通过项的偏差量级 ~2.8e-5 且全部不改变决策），
18/18 pass 五动作排序与选择完全一致；逐候选评分墙钟 B/A ≈ 0.745（更快），
scoring 阶段峰值显存 766→634 MB（改善）。但**含选中候选全梯度重算的总决策成本
A 1.491s vs B 1.433s（SIMILAR）**：JVP 省下的评分时间被必须保留的
selected-gradient recompute（~0.046s）与共享前缀（~1.0s）摊薄。
环境 SDPA（flash）无 forward-mode AD → ambient JVP = UNSUPPORTED_IN_CURRENT_BACKEND。
METHOD C（共享池方向导数）= NOT_APPLICABLE（候选池两两不相交，结构性）。

## 1. A0I-A：冻结标签适应性与可解释性诊断

### 1.1 方法

纯离线重分析 `analysis/analysis.json` + 每单元 `predictions/predictions_UID.json` +
`labels/score|eval/eval_labels_UID.json`。7 个选择器并列：controller（生产冻结语义）、
BF16 C、FP32 fwd、replay-score oracle、eval oracle、dev-fixed j=1、no-op。
选择器语义严格复现冻结规则：`argmin_j(λ·ĥ_j, action_id)`，h_0=0，精确并列取最小 id。
未改动任何标签。

### 1.2 动作频率（controller，12 单元）

| action | j0(no-op) | j1 | j2 | j3 | j4 |
|---|---|---|---|---|---|
| 次数 | 1 | 3 | 5 | 1 | 2 |

逐种子：2030{j1,j4}，2031{j2,j3}，2032{j2,j2}，2033{j0,j2}，2034{j1,j4}，2035{j1,j2}。
无单动作支配；j2 略多但远未固化。→ **适应性存在且明显**。

### 1.3 DeltaGain_adaptive（controller − dev-fixed(j=1)，配对）

- 总均值：**+5.76e-5**
- 逐种子均值：2033 **+3.15e-4**，2035 +2.00e-4，2034 +9.73e-5，2030 −9.8e-7，
  2032 −8.62e-5，2031 −1.80e-4
- 异质性高：增益不是均匀正，而是集中在 2033/2035/2034。
- 事后回看最优固定动作 = j=2（均值 1.039e-4，**POSTHOC_DIAGNOSTIC_ONLY**）；
  controller 相对它的均值差 +4.28e-5（仍有净增值，但幅度小于相对 j=1 的表观值）。

### 1.4 Regret 分解（从 CSV 重算，闭合同格）

均值 Regret_eval = **8.7808e-5** = 2.3002e-5（adjoint − replay 项）
+ 6.4805e-5（replay − eval-oracle 项）；闭包残差 max |residual| = **0.0**（精确）。
第一项 2.30e-5 是数值/精度差异项，**不是**数值误差垃圾——它量化了评分器精度本身造成的损失上界。

### 1.5 案例研究（三类典型单元，互不相同）

- **有害单元** `2032_046_s1`：controller 选 j2，score 增益 +1.98e-4 但 eval 仅 +7.33e-5
  （且为 12 单元中唯一有害）。score→eval 迁移失败。完整五动作表见 `case_study.json`。
- **no-op 单元** `2033_046_s0`：全部 ĥ_j > 0，controller 正确弃权（j0）；
  该单元同时是 eval-oracle 单元（eval 下 no-op 也最优）——弃权是真实信号而非失败。
- **最大 regret 单元** `2031_046_s1`：regret 3.42e-4。预测 top3 的 predicted_delta
  彼此相差 ~8e-6（近并列）；controller 选 j3，eval-oracle 为 j2。近并列决策对评分误差最敏感。

### 1.6 校准（x = λ·ĥ vs y = ΔL_score；pooled 约定见 calibration.json）

- 选中动作级（12 点）：Pearson 0.635，MAE 9.80e-5，RMSE 1.285e-4，R² 0.403，
  OLS slope 0.605 / intercept −8.21e-5（系统性低估斜率 + 负偏置）。
- 全候选级（60 点含 no-op）：Pearson 0.834，R² 0.696，slope 0.971（全候选层面几乎无偏）。
- ΔL_score→ΔL_eval（选中 12 点）：Pearson 0.429，R² 0.184 —— **仅观测性**，不可作效用外推。
- 含义：controller 的排序在候选集层面是好的；选中点级别的幅度校准有偏但一致。

## 2. A0I-B：标量方向导数评分可行性

### 2.1 合约与 gate（GPU 前冻结）

`contract.json`：恒等式 `h_j = D_θL_j[w_0] − w_0ᵀg`（w_0 固定切向量，不微分）；
gate atol32=1e-5、rtol32=1e-4（判据 abs(fwd−ref) ≤ atol + rtol·abs(ref)）；
决策保持 = 五动作排序 + chosen-id；benchmark 协议（5 repeats + 1 warmup、A/B 交替顺序、
stage 级 CUDA 计时、M=1/2/4 仅计时、等效 native step 为工程指标）。
7 个状态字段分离，无复合 PASS。

### 2.2 探针结论（probe_2031_52.json）

- ambient（flash SDPA）JVP：`NotImplementedError`（torch 2.11.0 的
  `_scaled_dot_product_flash_attention` 无 forward-mode AD）
  → **UNSUPPORTED_IN_CURRENT_BACKEND**（合法裁决）。
- math SDPA backend 下 METHOD B 正常 → 全部 B 数据在 math backend 下采集，
  并以 METHOD A_math（同 kernel 反向模式）作隔离对照。

### 2.3 数值等价（3 状态 × 6 pass = 18 评分 pass）

| 指标 | 值 |
|---|---|
| gate 通过 | 54/72（18 pass × 4 候选） |
| abs_err max / median | 2.83e-5 / 4.22e-6 |
| abs_err isolated（对 A_math）max / median | 2.29e-5 / 7.11e-6 |
| rel_err max / median | 7.36e-3 / 1.82e-3 |
| 排序一致 A_vs_B / A_vs_A_math | 18/18 / 18/18 |
| chosen-id 一致 | 18/18 / 18/18 |

端到端误差（对 ambient flash 的 A）含 kernel 差异；隔离误差（对 math 的 A）才是
forward-mode vs reverse-mode 累积差异本身，量级 2.3e-5，在共享前缀 w_0 的
下游不改变任何决策。未通过的 gate 项全部为 kernel 差异项，且无一改变排序/选择。

### 2.4 工程性能（每状态 5 repeats 中位数；共享 GPU 条件）

| stage | 2031/52 | 2034/64 | 2036/76 |
|---|---|---|---|
| shared prefix（rollout+endpoint grad+反向伴随→w_0） | 1.085 | 1.081 | 0.961 |
| scoring A（全梯度点积，生产参考） | 0.407 | 0.544 | 0.388 |
| scoring B（逐候选 JVP） | 0.303 | 0.346 | 0.295 |
| selected-gradient recompute（B 必须） | 0.046 | 0.048 | 0.040 |
| native step 参考 | 0.056 | 0.047 | 0.047 |
| **总决策成本 A** | **1.491** | **1.624** | **1.348** |
| **总决策成本 B（含 recompute）** | **1.433** | **1.475** | **1.296** |
| 等效 native steps A | 26.6 | 34.5 | 28.7 |
| 等效 native steps B | 25.6 | 31.3 | 27.6 |

- 逐候选评分：**B 更快**（B/A ≈ 0.745，scoring 阶段省 ~25% 墙钟、~17% 显存）。
- 总决策成本：**SIMILAR**（B 只省 ~4%）。共享前缀 ~1s 占绝对主导；recompute 把
  scoring 的节省再吃掉一部分。A 方法的总成本不含 recompute（它的全梯度副产物本来就留着）。
- 峰值显存（scoring stage alloc）：A 766 MB vs B 634 MB（**IMPROVED**）。

### 2.5 边际成本 T(M)=a+bM（M=1/2/4 嵌套前缀，仅计时，LOCAL_AFFINE_DIAGNOSTIC）

| state | method | a (s) | b (s/候选) |
|---|---|---|---|
| 2031/52 | A | 0.020 | 0.099 |
| 2031/52 | B | 0.037 | 0.067 |
| 2034/64 | A | −0.037 | 0.144 |
| 2034/64 | B | −0.011 | 0.089 |
| 2036/76 | A | 0.005 | 0.095 |
| 2036/76 | B | 0.005 | 0.072 |

b（每候选边际成本）：B 比 A 低 ~30%（0.067–0.089 vs 0.095–0.144 s/候选）。
a 的负值（2034/64）是 3 点仿合在共享 GPU 噪声下的正常产物，不解释物理含义。
**缩放判断：不足以外推 M=8/16/32**（3 点仿合、单模型、共享 GPU；路线图 D 节的
S(M) 非平凡性提示仍在）。

### 2.6 METHOD C（共享池方向导数）= NOT_APPLICABLE（结构性）

候选池为 4 个两两不相交 hash 组（逐对重叠恒 0，3 个 dev 状态数值核验 + 组定义核验）。
α 分解 `L_j = Σ α_ji ℓ_i` 代数上精确（α 与 θ 无关），但池不交 ⇒ 每个逐样本方向导数
只服务一个候选 ⇒ 32 次逐样本 JVP ≥ 8 次 micro JVP，无可摊销项。
flash JVP 本就不可用，逐样本 JVP 更无额外收益。详见 `method_c_derivation.md`。
含义：真实 M 扩展若继续按不相交子组构造候选，METHOD C 永远不适用；
只有 mixture 型动作空间（同池不同加权）才值得复核。

## 3. 裁决（§12 七字段，分离状态，无复合 PASS）

| 字段 | 裁决 |
|---|---|
| A0I_A_DIAGNOSTIC_COMPLETE | YES |
| DIRECTIONAL_SCORE_EQUIVALENCE | **SUPPORTED** |
| DIRECTIONAL_DECISION_PRESERVATION | **18/18** pass 同选择；排序一致 18/18 |
| PER_CANDIDATE_JVP_SPEED | **FASTER**（B/A scoring 墙钟比 0.745；ambient flash JVP = UNSUPPORTED_IN_CURRENT_BACKEND，数据在 math SDPA 下采集） |
| SHARED_POOL_DIRECTIONAL | **NOT_APPLICABLE**（不相交 hash 组，结构性） |
| MEMORY_RESULT | **IMPROVED**（766 MB vs 634 MB） |
| TOTAL_DECISION_COST_RESULT | **SIMILAR**（A 1.491s vs B 1.433s，含 selected-gradient recompute） |

决策树定位（§13）：**CASE B**——数学改写有效、决策保持，但 JVP 在本 backend 下
不是总成本层面的加速（只在 scoring 子阶段加速）。backend/精度不是误差来源
（隔离误差 2.3e-5 且零决策翻转）。

## 4. GPU 预算与过程

- 上限 1800 s；记账总额 ≈ **749 s**（METERED 254.46 + METERED_THEN_LOST 254.5 +
  两次崩溃 run 估计 ~240）。详见 `a0i_b/gpu_budget_log.json`。
- 重试 2 次（1 次 warmup untimed dict 缺 cuda_ms；1 次 CSV fieldnames 并集 bug
  在全部 GPU 完成后崩溃）；无用户中断；无环境变更；无外部任务被杀。

## 5. 边界与禁令遵守

未新增 acceptance batch / 训练种子 / checkpoint / 长训练；未碰 M=8/16/32、hybrid
selector、replay-gain 拟合、margin 弃权规则、λ、lr action、AdamW、V_eval 规则；
未挑 dev 状态；未把暴露的 A0H20 数据当独立验证（A0I-A 明确标注为同一冻结标签的
离线重分析）。A0G/A0H/A0H20 结论未改动。H5 = NOT_TESTED。

## 6. 下一步（建议，未自动启动）

1. 若未来要把方向导数评分搬进生产路径：先解决 backend 缺口（flash SDPA 无
   forward-mode AD），否则生产（ambient）与验证（math）kernel 不一致会持续贡献
   ~2e-5 级端到端误差。
2. 总成本被共享前缀（~1s，随 H 增长）主导。比起 scoring 的常数级优化，
   前缀成本（H 的线性项）才是 M/H 扩展时的真正杠杆——但这属于新轮次新合约，
   不在本轮范围。
3. M=8 扩展前需预冻结新数据子组候选；届时 METHOD C 对不相交构造仍不适用。
4. A0I-A 的校准偏置（选中级 slope 0.605）与 score→eval 弱迁移（R² 0.184）是
   控制器本身的研究线索，留待效用轮次新实验处理，本轮不做规则改动。
