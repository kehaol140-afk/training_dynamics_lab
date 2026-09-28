# FR 算法路线图（Post-A0I，当前有效）

- 版本：POST_A0I_V01，2026-09-16（纯文档轮，无 GPU、无新实验）
- 状态：**roadmap / 非预注册协议，不授权任何实验**
- 历史：`docs/FR_ALGORITHM_ROADMAP_POST_A0H20.md` 保持冻结不覆盖
  （其 I 节含 Post-A0I 修正登记）；本文是其后续
- 数字事实来源：`outputs/a0i_directional_scoring_v01/`（A0I 冻结制品，
  审计包 SHA256 `2dca15c4…9775`）；A0G/A0H/A0H20 冻结结论原样

---

## A. 当前算法成熟度（四层定位）

1. **Mechanistic response**：current tested regime 内 **established**。
   完整光滑 tangent（Model C，AdamW 状态 Jacobian + clip-JVP + 未来
   HVP）在 H≤8 内解释几乎全部响应（A0C/A0D/A0E2/A0F2，v0.7 bank
   56 态普查 168/224 改善、0 负）；adjoint 实现精确（fp32-HVP 候选
   max 4.1e-10）。
2. **One-shot task utility**：**supported**（A0G Level-3 决策效用 +
   A0H20 独立批次 POSITIVE_POINT_ESTIMATE：Gain_eval +1.467e-4、
   5/6 seed 正、harmful 1/12）。
3. **Efficient implementation**：**partial**。共享伴随 + directional
   scoring 有局部收益（scoring -25% 墙钟、-30% 边际、-17% 显存），
   但 total decision cost 仍 ~26–31 等效 native steps，太贵。
4. **Closed-loop / equal-wall-clock training**：**NOT TESTED**。

**当前不是 "adjoint correctness" 主问题**。主要风险已转移为：

- **compute economics**（shared prefix 成本）
- **score→eval transfer**（选中级 R² 0.184，观测性）
- **adaptive value**（ADVANTAGE_OVER_FIXED_UNCERTAIN）
- **repeated-control utility**（one-shot → 闭环未验证）

---

## B. 当前最重要的新计算假设（动作注入 vs 未来敏感度分离）

### B.1 参考控制器（已验证身份）

联合状态 `z_t = (θ_t, m_t, v_t)`（AdamW 一阶矩/二阶矩 + 参数；
t 为 optimizer step index）。未来 joint-state tangent 步进：

```
A_k = D_k + G_k K_k E_θ
```

其中 `D_k` = 固定梯度下 optimizer-local 状态 Jacobian，`G_k` =
optimizer 状态对 raw gradient 输入的导数，`K_k` = 训练梯度对 θ 的
导数（future HVP），`E_θ` = 从联合状态取参数通道的投影。

终点伴随：

```
p_(t+H) = ( ∇_θ L_score(θ_(t+H)), 0, 0 )
```

（只 score 依赖 θ；optimizer 状态通道的终点 cotangent 为零。）
反向传播 H 步得 `p_(t+1)`（三通道 cotangent）。首步 response
direction：

```
w_t = G_t^T p_(t+1)        （候选无关，只算一次）
```

候选动作 `a_j = g_j − g_t`（数据加权梯度差），响应分数：

```
h_j = w_t^T a_j ,   h_0 = 0
predicted_delta_j = λ · h_j
j* = argmin_{j=0..M} ( predicted_delta_j , action_id )   # 精确并列取最小 id
```

**当前已验证的 scientific identity = one-shot response-guided
data-reweighting controller，不是 learning-rate controller。** 动作
只注入数据加权梯度通道（post-unscale/pre-clip），LR/optimizer 超参
不在动作空间内。

### B.2 决策时刻的分解

在决策时刻 t，评分只需要：

```
h_j = p_t^T G_t a_j ,   w_t = G_t^T p_t
```

本文用简化记号 `p_t` 表示"完整 H-step lookahead 回传到首步之后的
三通道 cotangent"（即 B.1 的 `p_(t+1)`，t 为决策步索引）。
**索引语义注记**：`p_(t+H)` 是终点伴随（只在终点求值），经 H 步
反向传播后才得到首步可用的 `p_t`；两者不可混用。冻结 A0H/A0H20
实现里的 `w_0` 即此 `w_t`。

### B.3 新假设：缓存未来敏感度，刷新当前 G

若最近一次完整 future adjoint 在时刻 τ（`age = t − τ`）：

- **Direct stale-w**：`w_τ = G_τ^T p_τ`，误差

  ```
  w_t − w_τ = G_t^T (p_t − p_τ) + (G_t − G_τ)^T p_τ
  ```

- **新候选 cached-p + fresh-G**：`w̃_t = G_t^T p_τ`，误差

  ```
  w_t − w̃_t = G_t^T (p_t − p_τ)
  ```

  即**去掉 stale-w 的 `(G_t−G_τ)^T p_τ` 项**——在当前 θ/m/v/raw
  grad/clip/LR/optimizer step 都按精确 `G_t` 刷新，只近似未来敏感
  度 `p`。

**算法含义**：缓存昂贵的 future sensitivity `p_τ`（~1s 的 shared
prefix 产物）；每次决策只花 `G_t^T p_τ` 的解析/局部代价（见 C 节）。
保留当前 optimizer action-injection semantics 不变，近似只发生在
未来敏感度通道。

**标记：`UNVERIFIED APPROXIMATE CONTROLLER`**。不能写成已证实比
stale-w 更准：stale-w 的两项误差可能抵消（`G_t−G_τ` 与 `p_t−p_τ`
非独立），去掉一项在特定轨迹上也可能更差；只能由实验回答。

---

## C. G_t^T p 的解析结构

`G_t^T p` 完全由**当前局部量**决定：

- current raw gradient `g_t`
- current `m_t / v_t`（AdamW 一阶/二阶矩）
- bias correction（`1 − β^t` 项）
- learning rate / weight decay / `β1, β2, eps`（parameter-group
  hyperparameters）
- clipping Jacobian（`J_clip(g_t)`，clip 激活与否决定其形式）

它**不需要未来 Transformer HVP**，因此理论上远便宜于重算完整
`p_t`（`p_t` 的昂贵部分 = H 步 baseline rollout + endpoint grad +
每步 HVP/curvature-feedback 的反向链）。

外部推导中已有 `G_t^T p` 的解析公式（AdamW 更新映射对 raw
gradient 输入的伴随 + clip 转置）。**任何引用必须标记**：

```
ANALYTIC DERIVATION / NOT YET SERVER-PRODUCTION-VALIDATED
```

本地的 double finite-difference 抽查属推导自验，**不得写成服务器
算法实验结果**。进入 A0J-P1 前须在服务器路径上以冻结 gate
（atol32=1e-5 / rtol32=1e-4）对照数值版 `G_t^T p` 验证。

---

## D. Cache validity 的正确指标

**禁止用全向量 cosine 作为主要 gate**。cosine 不决定排序。

直接决定动作排序的是候选 j,l 的 **score-gap error**：

```
delta_gap_(j,l) = ( p_t − p_τ )^T G_t ( a_j − a_l )
```

no-op 用 `a_0 = 0` 代入（`delta_gap_(j,0) = (p_t−p_τ)^T G_t a_j`）。

未来缓存实验至少报告（每个 age、每个 state）：

- exact score vector（fresh `w_t` 算出的 5 个 h_j）
- cached-p score vector（`w̃_t = G_t^T p_τ`）
- direct stale-w score vector（`w_τ`）
- pairwise gap error（全部 `M(M+1)/2` 对，含 no-op 对）
- top1-top2 margin（预测空间）
- best-nonzero-vs-noop margin
- chosen-id agreement
- action-rank agreement（五动作全序）

cache ages 首批候选：`age = 1, 2, 4`。**仅记录为 roadmap，不运行。**

**允许否定**：如果 cache age=1 就明显破坏候选排序或效用（chosen-id
agreement 显著下降或 DeltaGain 变负），应允许停止缓存路线，不得
为了保路线而放松 gate。

---

## E. Refresh 与 scoring frequency 必须拆开

定义：

```
K_r = full future-sensitivity refresh interval（p_τ 重算间隔）
K_s = candidate rescore / intervention interval（评分/干预间隔）
```

近似 controller 的额外成本：

```
overhead = [ T_prefix / K_r
           + ( T_score + T_selected_grad + T_GTp ) / K_s ]
         / T_native_step
```

其中 `T_prefix` = 完整 shared prefix（rollout + endpoint grad +
反向伴随，`p_τ` 的产物）；`T_GTp` = 解析 `G_t^T p_τ` 的局部代价
（C 节，理论上 << T_prefix）。

**明确**：

- 缓存 prefix ≠ candidate scoring 免费；
- **不得**把整个 decision cost 除以 K_r、同时仍每步重评分而漏记
  scoring cost（分子两项必须同时存在）；
- 低频 intervention（大 K_s）与低频 future-sensitivity refresh
  （大 K_r）是两个独立的算法选择，可分别取值、分别消融。

---

## F. Score→eval 的独立突破方向（FUTURE SCIENTIFIC BRANCH）

固定相同 baseline future trajectory，终点 score gradients：

```
v_b   （第 b 个预定 score sub-batch 在终点的 ∇L_score）
```

预冻结权重：

```
v̄ = Σ_b ω_b v_b
```

由于 reverse response 对 terminal cotangent **线性**：

```
R_t^T v̄ = Σ_b ω_b ( R_t^T v_b )
```

因此多个预定 score sub-batch 可以先平均 terminal gradients，再
**只传播一次** full adjoint。动机：降低 score sampling noise
（单 sub-batch 的 ∇L_score 是噪声读数），而不是改动力学。

**约束**：

- 不能用已看 V_eval 调 `ω_b`（权重须预冻结或按与 V_eval 无关的
  规则定义）；
- 不证明解决 distribution shift（score 分布 ≠ eval 分布）；
- **不与 cache-p 实验在同一轮共同修改**（单次只动一个变量）；
- `Σ_val`（验证集）与早期 training-fluctuation `Σ` 不是同一个
  对象，公式移植须重新论证。

**标记：`FUTURE SCIENTIFIC BRANCH`**（记录，不执行）。

---

## G. HVP approximation 只作为 profiler-gated backup

中心差分候选（替代 math-SDPA HVP 的解析 HVP）：

```
K̃_ρ q = ‖q‖ · [ g_native(θ + ρ q̄) − g_native(θ − ρ q̄) ] / (2ρ)
```

其中 `q̄ = q/‖q‖`。**仅当 profiler（A0J-P0）证明 math-HVP 是 prefix
主导项时才测试。**

**标记：`APPROXIMATE / NOT EXACT ADJOINT`**。风险清单：

- ρ 的 truncation vs finite-precision tradeoff（ρ 大→截断误差，
  ρ 小→差分淹没）；
- bf16 autocast 下扰动损失（bf16 perturbation loss）；
- Hessian symmetry 只是理想性质（FD 算子不精确自伴）；
- FD 算子不精确线性；
- 两次 native backward 未必比一次 math-HVP 便宜。

本轮不实现。

---

## H. Candidate batching correction（工程选项，非 sample-sharing）

更正表述：当前 hash candidates 样本不重叠 ⇒ **没有 shared-pool
sample reuse**；但这不等于候选评分无任何批处理空间。

仍可构造向量输出函数：

```
F(θ) = ( L_1(θ), ..., L_M(θ) )
```

并尝试一次/vectorized JVP：

```
JVP(F; w) = ( w^T g_1, ..., w^T g_M )
```

**只是 future engineering option**。任何实现必须保持：候选独立
attention/mask/token normalization/RNG/dropout 语义与 peak memory
约束；不得与 sample-sharing 的 α 分解公式（A0I method C）混淆——
METHOD C 的 NOT_APPLICABLE 判定只针对样本复用摊销，不排除本节的
调度/packing 收益。

---

## I. 当前一级算法健康指标（dashboard 固定六项）

二级诊断（Pearson / R² / score MAE / forward-adjoint numerical
error）继续保留，但**不能取代**以下一级指标。

1. **TASK_UTILITY**：mean Gain_eval、harmful rate、per-seed/state
   sign。当前冻结参考：mean ≈ **+1.467e-4**、**5/6** seed 正、
   harmful **1/12**。
2. **ADAPTIVITY_VALUE**：`DeltaGain_adaptive-fixed`。当前：正点
   估计但 heterogeneous / uncertain（ADAPTIVE_CHOICES_OBSERVED /
   ADVANTAGE_OVER_FIXED_UNCERTAIN）。状态：**OPEN**。
3. **SELECTION_QUALITY**：Regret_eval、empirical finite-candidate
   oracle capture。当前：Regret mean ≈ **8.78e-5**、capture ≈
   **62.6%**（diagnostic only）。
4. **DECISION_COST**：total actionable decision seconds、
   equivalent native steps/decision。当前 directional
   implementation：**roughly 26–31 native-step equivalents**。
   状态：**TOO_EXPENSIVE_FOR_FREQUENT_CONTROL**。
5. **APPROXIMATION_UTILITY_RETENTION**（为缓存/近似方法设立）：
   `DeltaGain_approx-exact = Gain_approx − Gain_exact` 以及
   chosen-id preservation(age)。当前：**NOT_TESTED**。
6. **EQUAL_WALLCLOCK_TRAINING_GAIN**：
   `DeltaL_budget(T) = L_baseline(T) − L_controller(T)`。当前：
   **NOT_TESTED**。

---

## J. 下一实验顺序（只记录，不执行）

### A0J-P0：bounded shared-prefix profiler / cost accounting

目的：拆分 ~1s shared prefix 的构成。至少区分：

- baseline H-step lookahead（rollout）
- endpoint V_score gradient
- each future HVP / curvature-feedback operation
- analytic AdamW/clipping VJP（反向链中的局部段）
- state snapshot/restore
- precision/backend conversion（bf16 autocast、math-vs-flash SDPA）
- synchronization（CUDA sync、host-device 边界）
- scientific computation vs audit/logging overhead

profiler 只用于**归因**；正式 wall-clock 一律用无-profiler
CUDA-synchronized timing（沿用 A0H15/A0I 口径）。

**之后依据 profiler 只开一个主分支**：

- 若 **future lookahead / adjoint propagation** 主导 → **A0J-P1**：
  cached future adjoint `p_τ` + exact current `G_t`（B.3）；cache
  ages 1/2/4，dev-only 首轮；比较 exact fresh-p vs cached-p+fresh-G
  vs stale-w；先看 score-gap error / chosen-id preservation / cost
  reduction，再决定是否需要 replay utility；age=1 明显失效则允许
  停止缓存路线。
- 若 **HVP 单项主导** → 才考虑 G 节 finite-difference HVP branch。
- 若 **restore/build/sync 主导** → 优先工程复用/静态缓冲（非科学
  近似）。

**隔离规则**：score-readout averaging（F 节）单独作为以后科学分支，
不要和 cache-p 第一轮同时修改。M=8/16/32 候选扩展（A0J 原义）在
成本经济学闭合前不是优先项。

---

## 附：与本文档冲突时的优先级

新科学结论 > 本文档；本文档 > 旧 roadmap；一切 > 口头记忆。
任何实验（含 profiler）启动前须另行预注册合约；本文档不授权实验。

## K. A0J 结果回填（2026-09-16，只记录，不授权新实验）

K.1 执行顺序更新：A0J-P0 profiler → 预注册门 S_upper median 4.4147
≥1.25 → full P1（cached-p + fresh-G 分支，FD-HVP / 工程复用分支未开）
→ dev replay → 成本 → dashboard → audit。COMPLETE。

K.2 与本文档 B/D/E 节的关系：动作注入 vs 未来敏感度分离（B）在
fresh 路径上保持有效；cached 路径（复用 p_tau）在 age≥1 即破坏
决策保持（D 节 preservation 优先级的直接验证）；refresh 与 scoring
成本拆开（E）已实测：refresh 1.279s vs cached non-refresh 0.912s
（含保守候选重算），节省真实但救不了决策。

K.3 裁决：§18 CASE B（含 C 元素）——future sensitivity 在当前
setup 下变化太快，简单时间复用不成立；CACHE_P 一致但仅略优于
STALE_W（fresh G_t 有价值、窗口太短）；缓存路线停止。不得用
cosine 高或调 threshold 挽救。H6 仍 NOT_TESTED。

K.4 若未来重访：refresh 触发应基于实测 p-drift 而非固定 age；或只
复用 endpoint 梯度方向配新鲜标量。两条都须新预注册轮。
