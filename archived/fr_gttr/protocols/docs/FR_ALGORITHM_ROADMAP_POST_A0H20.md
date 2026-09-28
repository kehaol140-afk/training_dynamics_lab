# FR 算法路线图（post-A0H20，A0H-2.0 之后）

> 2026-09-16。**本文档不是预注册协议，不授权任何实验、不启动任何 GPU 任务**；
> 唯一用途是 /compact 与新上下文交接后恢复算法路线与推理链。所有数字取自
> 冻结制品（`outputs/a0h_corrected_accept_v01/`、`outputs/a0h15_vjp_shared_v01/`），
> 不得凭记忆改写。任何 A0I 及后续实验须另行预注册并由用户授权；旧 A′ FAIL、
> selector 语义审计、corrected exposed re-score 的历史身份保持不变。
> 背景状态见 `docs/FR_PROJECT_STATUS.md` §25–§26；批次细节见
> `outputs/a0h_corrected_accept_v01/UTILITY_PILOT_REPORT.md`。

## A. 核心公式（冻结语义）

候选共享伴随打分器的完整链条（全部冻结，禁止改动）：

- `p_H = ∇ℓ_H`：终端 V_score 对 θ_{t+H} 的梯度（基轨迹终点，候选无关）。
- `p_k = A_kᵀ p_{k+1}`：伴随递归，A_k = 第 k 步冻结 tangent 块（AdamW 状态
  更新 + clip-JVP + 曲率 HVP），从 H 到 1 单次反向行走。
- `w_0 = G_0ᵀ p_1`：注入点的伴随载荷（候选无关）。
- `h_j = w_0ᵀ(g_j − g) = project_tangent(w_0, a_j)`，a_j = g_j − g
  （j=1..M 非零动作）。
- `pred_delta_j = λ·h_j`；predicted gain = −λ·h_j。
- `j* = argmin_{j=0..M}(pred_delta_j, action_id)`：五动作（含 no-op）上的
  冻结 A0G 选择器，精确 tie 取最小 action_id；公共正 λ 不改变排序。
- `h_0 = 0`：j=0 no-op = 继续训练（不是跳过 optimizer step），预测 ΔL=0，
  合法参与 argmin。

## B. 含义（已闭合的命题与未闭合的命题）

共享伴随**不是新 scorer**，而是 forward full-tangent response scorer 的
**共享反向实现**（同一数学对象的两条计算路径；A0H-1 fp32 探针 4.1e-10、
A0H20 同配置 FP32 gate PASS 12/12）。A0H20 证明的闭环是：

> numerical consistency（gate PASS）→ decision preservation（同配置与
> 跨配置 12/12 五动作一致）→ independent short-term V_eval utility
> （主终点 +1.467e-4，POSITIVE_POINT_ESTIMATE）

该闭环只在**当前有限设置**（one-step 注入、conditional lookahead、H=4、
λ=0.05、step-46 状态、5M 模型、12 单元）内成立。**不**代表长期训练最优、
不代表闭环/repeated 收益、不代表部署价值。方法定位：
**one-step response-guided training-action scorer/controller prototype**。

## C. 诊断数字（本批经验分解，不得外推为总体量）

- oracle_eval_mean_gain（每单元五动作 V_eval gain 最优值的均值）
  ≈ **2.345e-4**（= 主终点 1.467e-4 + mean Regret 8.78e-5，簿记闭合）。
- adjoint/oracle mean-gain ratio ≈ **62.6%**——**仅本批经验 oracle 诊断
  比例**，不能称总体 optimality capture。
- Regret 分解：**8.78e-5 ≈ 2.30e-5 + 6.48e-5**。
  - 第一项 2.30e-5 = adjoint choice 与 V_score finite-replay choice 在
    V_eval 上的差（fp32adj − replay-score-oracle 配对差异）——混合了
    有限动作预测误差与 score/eval 分布差，**不能称纯数值误差**；
  - 第二项 6.48e-5 = V_score replay choice 与 empirical V_eval oracle
    的差（score→eval 泛化间隙，观测定位 pearson 0.429）。
- 观测性定位（仅观测，无归因）：corr(λ·ĥ_sel, ΔL_score_sel)=0.635；
  无幅度/原生导数探针，禁止 λ/非线性/代理失效归因。

## D. 成本模型（为什么不能从 M=4 外推 M=32）

设伴随路径与逐候选回放路径的墙钟均为候选数 M 的仿射函数：

- `T_adj(M) = a + b·M`（a = 公共基轨迹 + 单次反向行走 + w_0 的 fixed
  cost；b = 每候选边际：完整 g_j 反向 + 点积）
- `T_replay(M) = c + d·M`（c = replay fixed；d = 每候选 replay 边际）
- speedup `S(M) = (c + d·M)/(a + b·M)`
- `S′(M) = (d·a − b·c)/(a + b·M)²`：符号由 (da−bc) 决定。

DEV_ENGINEERING_TIMING（H=4/M=4，GPU1 独占）：fp32_adjoint 1.321s vs
native_replay 1.546s（R_replay=1.170，墙钟降 14.55%）只给出 S(4)。**不能**
从单点推断 speedup 必随 M 增长：若 b（每候选完整梯度反向）较大而 a 较小，
S(M) 可能随 M 下降。下一阶段任务是**识别 fixed cost 与 marginal candidate
cost**（分解 a、b、c、d），而非直接假定 M=32 更优。

## E. 真实候选扩展约束（凸组合不能冒充 M 扩展）

候选梯度 g_j 的任意凸组合 g_α = Σ_k α_k g_k（Σα_k=1, α_k≥0）的伴随
得分为：

> h_α = w_0ᵀ(g_α − g) = Σ_k α_k·w_0ᵀ(g_k − g) = Σ_k α_k·h_k ≥ min_k h_k

即**凸包内部点的 predicted ΔL 不差于包内最优顶点但也不会更优**（h_0=0
的 no-op 方向同样被凸包包含）。因此：

- 重复或凸组合候选**不能**算真实 M 扩展，只能做吞吐/压力测试；
- 真实 M 扩展必须定义**新动作自由度**：预冻结的新数据子组（新 partition）、
  更细的子分组、或不同 minibatch 构成——每个新候选须先有独立身份再入池；
- A0J 的 M=4→8 必须是 4 个**新预冻结候选**，不是现有 4 候选的重采样。

## F. 突破口：scalar directional scoring（UNVERIFIED COMPUTATIONAL OPPORTUNITY）

由 h_j = wᵀ(g_j − g) 出发的两条代数恒等式（纯数学变换，**未经实验验证**）：

1. 若 g_j = ∇_θ L_j（候选 j 的完整数据梯度），则
   `h_j = D_θ L_j[w] − wᵀg`：每个候选只需**一次标量方向导数**（JVP），
   不需要完整梯度向量。
2. 共享数据池时 g_j = Σ_i α_ji·∇ℓ_i；令 d_i = D_θ ℓ_i[w]（每样本标量
   方向导数），则 `h_j = (α_j − α_0)ᵀ d`：一次逐样本方向导数 pass
   （与候选数无关）+ 每候选一个稀疏点积。

**标记：UNVERIFIED COMPUTATIONAL OPPORTUNITY**。潜在收益是把每候选
边际成本 b 从"完整反向"降为"标量 JVP/稀疏点积"；须未来验证：
(a) score equivalence（directional h_j 与完整伴随 h_j 的数值一致）；
(b) decision preservation（argmin 排序/选择保持）；
(c) wall-clock（真实加速）；(d) peak memory（JVP 与反向的内存差）。
在验证前不得进入任何验收/效用口径。

## G. margin 定理的用途边界（decision-preservation 工具）

近似打分器 s̃（低精度/directional/截断）保持 exact action 的充分条件：

> 若 max_j |s̃_j − s_j| ≤ eps 且 exact top1-top2 margin γ > 2·eps，
> 则 argmin(s̃) = argmin(s)（exact action preserved）。

**仅**用于 future approximation / low-precision / directional-scoring 的
decision-preservation 分析；**不**证明 V_eval utility（margin 是 score 空间
量）；**不在现有 12 单元上拟 abstention threshold**（阈值拟合属被禁止的
分类器式设计）。near-tie 命中（A0H20 中 top-1 near-tie 6/12）只作诊断读数。

## H. 下阶段顺序（只记录路线，非授权）

1. **A0I-A**：existing-label offline adaptivity diagnostics——用已冻结的
   A0H20 12 单元五动作双标签，离线（纯 CPU）分析自适应选择 vs fixed
   action 的逐单元差结构、near-tie 分布、score→eval 泛化间隙的形态；
   不产生新实验数据。
2. **A0I-B**：scalar directional scoring 可行性——在**现有 dev**（已降级
   数据/演练单元）上验证 F 节四条（score equivalence、decision
   preservation、wall-clock、peak memory），M=4/H=4；dev-only 无验收身份。
3. **A0J**：预定义真实候选扩展——先 M=4→8（4 个新预冻结数据子组候选，
   暴露前冻结台账），条件满足再 16/32；每步先验冻结。
4. **A0K**：repeated/closed-loop controller——one-shot → repeated 注入，
   equal-step 与 equal-wall-clock 两口径对比直接训练。

**当前优先级 A0I-A → A0I-B**。明确**不是**：直接 M=32；改 AdamW/lr/λ；
hybrid selector；在现有 12 单元上拟合任何阈值或规则。

---

## I. A0I 结果回填（2026-09-16，I=A0I 之后）

**F 节机会验证状态（四条全有答案，dev-only）**：

| F 节验证项 | 结果 |
|---|---|
| (a) score equivalence | **成立（math backend）**：隔离误差 max 2.29e-5 / median 7.11e-6，端到端 max 2.83e-5（含 kernel 差）；gate 54/72，未通过项零决策影响 |
| (b) decision preservation | **18/18**（五动作排序 + chosen-id，对 A 与 A_math 双对照） |
| (c) wall-clock | scoring 子阶段 B/A≈0.745（更快）；**总决策成本 SIMILAR**（前缀 ~1s 主导 + recompute 0.046s） |
| (d) peak memory | **改善**：766→634 MB（-17%） |

**F 节标注从 UNVERIFIED 更新为：VERIFIED_ON_DEV_MATH_BACKEND_WITH_
CAVEATS**——caveats：ambient flash SDPA 无 forward-mode AD（probe
NotImplementedError）；总成本无加速；只覆盖 M=4/H=4 与 3 个 dev 态。

**C 节诊断回顾**：regret 分解离线重算闭合同格（8.7808e-5 =
2.3002e-5 + 6.4805e-5，max |residual|=0.0），数字维持冻结值。

**E 节约束回顾**：M=1/2/4 嵌套前缀计时给出 T(M)=a+bM（b：A 0.095–
0.144、B 0.067–0.089 s/候选，LOCAL_AFFINE_DIAGNOSTIC，3 点仿合、
共享 GPU，不足以外推）；candidate marginal cost 确实被方向导数
压低 ~30%，但 shared prefix 在总成本中占比使 S(M) 改善受限——
**M 扩展的成本杠杆在共享前缀（H 的线性项）而非 scoring 子阶段**。

**A0I-A 对 A 节问题"增益从哪来"的回答（Post-A0I 修正）**：观察到状态条件化动作选择（动作频率分散、无单动作支配，ADAPTIVE_CHOICES_OBSERVED），但动作频繁切换本身不能单独证明净收益（ADVANTAGE_OVER_FIXED_UNCERTAIN）；增益种子异质；controller 相对事后最优固定动作的点估计差 +4.28e-5，须带不确定性注记。

**G 节 margin 定理边界回顾**：max-regret 单元（2031_046_s1）预测
top3 近并列 ~8e-6，与近似评分器误差同量级——margin 工具对这类
单元的保护边界正是其用途所在；本轮未做任何 abstention 阈值拟合。

**下阶段**：A0I-A/B 已闭合。A0J（预冻结真实候选扩展 M=4→8）与
backend 缺口（flash forward-AD）均须另行预注册；A0I 不自动启动
任何后续。METHOD C 判定 NOT_APPLICABLE **仅针对样本复用摊销**
（Post-A0I 修正：不再写"永久不适用"）；不相交构造不排除
vector-output JVP / candidate batch packing 等调度开销工程方案；
mixture 型动作空间的 α 分解复核仍然开放。

**Post-A0I 修正登记（2026-09-16）**：numerical gate 54/72 与
decision preservation 18/18 分开计数；regret 第一项可为负、
非"评分器精度损失上界"；校准 slope 0.971 不单独称无偏。
当前有效路线图为 `docs/FR_ALGORITHM_ROADMAP_POST_A0I.md`
（本节及本文保持历史，不覆盖）。
