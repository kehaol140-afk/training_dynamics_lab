# FR A0H-1.5 协议 —— VJP 候选共享伴随（诊断合同）

> 本文件在观察任何 A0H-1.5 结果**之前**冻结诊断合同、样本角色、数值容差与判据。
> 冻结的原始 BF16 gate（A0H-1 合同，atol=1e-6/rtol=1e-4，逐元）**读入不变、继续冻结**；
> 本文只定义 A0H-1.5 的**新**检查。
> 对 A0H-1 已交付结论的修正（口径收窄）见 docs/FR_A0H15_CORRECTIONS.md，与本合同同日冻结。

## 0. 目标与核心问题

冻结的 BF16 前向响应打分器能否支撑一个**候选共享**的反向打分器，使其在
**原始数值标准**下与冻结前向一致，且**不预设数值 Hessian 自伴**？
三个子问题：(a) 共享性（w_0 与候选无关）；(b) 独立决策保持（排序/选择/no-op）；
(c) 是否比真实回放便宜（计时，A0H-2 小规模）。

关键概念修正（先于任何实现）：
- `<Kx,y> ≠ <x,Ky>`（浮点）**不能**推出"K 的独立转置不存在"。浮点算子一般既非自伴
  也无精确转置伴随；VJP_K(anchor,y) 是**锚点处微分规则** DK(anchor)ᵀ y，不是
  全局映射的转置。共享伴随的正确构造：对**切线输入**做可微封装（θ/m/v/训练输入
  数学上固定），在**锚点（实际轨迹）**处对该封装取 VJP。
- `|forward_BF16 − forward_FP32| > tol` **不能**推出"任何 BF16 伴随都不可能通过"。
  它只说明该比较在原始容差下超出；新的共享伴随构造（本协议 §5）此前未测过。

## 1. 样本角色（先于结果冻结）

- **诊断集 D**（只读复用 A0H-1 已完成的指纹匹配步骤，不重新跑训练）：
  3 个原始 dev 单元，与 A0H-1 相同：
  D1 = dev:2031/52 × stream 0；D2 = dev:2034/64 × stream 0；D3 = dev:2036/76 × stream 0。
  其中 C|8 失败单元 = {D1, D2}；C|8 通过单元 = {D3}；C|4 失败单元 = {D2}（j=2）。
  望远镜闭包（§3）与线性/自伴探针（§4）在 D1、D2 + D3（一个通过单元作对照）上进行。
- **独立验收集 A**（≤3 个未用 dev 单元，先验选择，读一次）：
  A1 = dev:2032/58 × stream 0；A2 = dev:2035/70 × stream 0；A3 = dev:2037/46 × stream 0。
  （v07 bank dev 分区其余可用 (seed,step) 组合；本批选 3 个，均不同于诊断集。
  与 A0G/A0H-1 共用同一 v07 bank 只读快照。）
  用途：仅在 §6 Branch A gate 全部通过后读一次做验证；**一旦读取，不再调整
  实现或阈值后把同一批当未见数据**；若据其调试，整批降为开发数据并如实记录。
- 不根据任何 A0H-1.5 观测结果增删/调换验收单元。

## 2. 完整数值配置（冻结记录，实现必须逐项满足）

与 A0G/A0H-1 冻结管线一致，不得漂移：
- 梯度注入位置：动作 `a_j = g_j − g`，post-unscale/pre-clip 单步（hvp_direction_mode="pre"）。
- 参数集：θ 全参数；切线联合态 z=(δθ,δm,δv)；优化器状态 m/v（exp_avg/exp_avg_sq）。
- β1=0.9, β2=0.999, eps=1e-8, weight_decay=0.01, lr=5e-4（调度器按冻结 runner）。
- bias correction：bc1=1−β1^(t+k), bc2=1−β2^(t+k)，step_after=t+k。
- clip：max_norm=1.0；clip 转置公式冻结（A0H 协议 §1.3）。
- dtype/累加：bf16 autocast（fwd/bwd），fp32 主权重与切线，投影/诊断点积 float64 累加。
- autocast：与冻结 runner 相同上下文；TF32：冻结 runner 未显式开启 ⇒ 不开启。
- attention backend：math-SDPA HVP 代理（`_hvp` + `_math_sdpa_context` double backward）；
  native flash 无 double backward，不换。
- RNG/dropout：seed_future_stream + capture_rng_state/restore_rng_state 纪律；
  前向与伴随各自的 HVP 调用都要恢复 `rng_before_native_forward`。
- graph rebuild：每单元重建；快照存步前 (m,v)、clip 后梯度 c、g、lr、clip 系数、
  rng_before_native_forward、步起点 θ 的 state_dict。
- loss 归一化：与冻结 runner 相同（grad 已按 accumulation_steps=2 unscale）。
- H post-step 读出：θ_{t+H}，H 个 native optimizer step 之后。
- **"FP32-HVP" 的精确定义**（仅诊断/对照用，Branch B 另作不同配置候选）：
  对 7 个 `_hvp` 调用点把 `autocast_ctx` 换成 `nullcontext()`（HVP 的 fwd+bwd 不 autocast，
  参数张量本身已是 fp32）；候选梯度与读出梯度保持冻结 bf16 不变。
- 封装器（VJP 源）：对冻结切线步的**切线输入** δc（或等价地 a_j）做可微封装；
  基 θ/m/v 快照、训练 batch、RNG、lr/clip 系数全部作为**常量**注入（数学固定，
  不进入微分图）。**禁止** `.detach()` / `no_grad` 截断方向图；
  方向探针必须非零；不得把 None 梯度静默补零；
  若嵌套 AD 不支持 ⇒ 保存最小复现并判 NOT_SUPPORTED（不静默换后端/精度）。

## 3. 望远镜恒等式闭包（诊断检查 CHK-TELE，BF16 算子，诊断点积仅 FP64）

对实际轨迹（B 与 C，k=1..H）逐步计算并记录（每候选）：
`T_j = (p_1ᵀx_1 − w_0ᵀa_j) + Σ_{k=1}^{H−1} (p_{k+1}ᵀx_{k+1} − p_kᵀx_k)`
其中 x_k = 冻结前向联合切线状态（δθ,δm,δv），p_k = 伴随联合余切，w_0 = J_1ᵀb_1。
- 仅诊断点积用 float64 累加；算子本身保持冻结 BF16。
- 逐项记录带符号残差与绝对值（每 k、每候选）。
- 判据（预注册）：`max_k |项_k| ≤ atol_tele + rtol_tele · max_k |端点项|`，
  **atol_tele=1e-3，rtol_tele=1e-2**（依据：BF16 HVP 代理自身噪声地板实测
  ~6e-5~6e-6 量级（H=8，A0H-1 fp32 探针），余切范数放大后逐项量级 ~1e-3 内，
  取 100×地板为上限；该容差不回用于原始 BF16 逐元 gate）。
- 望远镜闭包是**一致性诊断，不证明根因**；大步长项必须沿实际方向分解
  （不强制 SDPA 归因）。

## 4. 可重复性/线性性/自伴 vs 独立 VJP（诊断检查 CHK-LIN）

对诊断集每单元 × C 模型 × 每个未来步 k（k=2..H，反馈 HVP 点）：
- **REP**（≥3 次同方向重复）：`|K(x)_r − K(x)_1|∞` 与均值/方差；分离确定性偏差与随机变化。
- **LIN-K0**：K(0) 的范数（应 ~0 容差外；记录实际值）。
- **LIN-NEG**：K(−x)+K(x) 范数；**LIN-α**：α∈{0.5, 2, 1.5}（预固定）
  `|K(αx)−αK(x)|∞`；**LIN-ADD**：`|K(x+y)−K(x)−K(y)|∞`（x,y 为实际方向与其正交扰动）。
  判据（预注册）：线性残差相对量 `≤ 1e-2`（BF16 代理非线性噪声，100× 实测地板）；
  K(0) 相对 `‖K(x)‖` 的相对量 `≤ 1e-2`。仅诊断，不回用于 gate。
- **SA**：自伴缺陷 `<Kx,y> − <x,Ky>`（y=伴随 w_k）。
- **IVJP**：独立 VJP 缺陷 `<Kx,y> − <x, VJP_K(anchor,y)>`，anchor = 微分锚点
  （该步实际轨迹点），VJP 对 §2 封装器取。
- 输出原始内积与向量范数，不只给比值。
- 多个锚点：零方向锚（起点切线为 0 的基轨迹锚）+ 实际候选切线方向锚
  （每候选 j 一个锚），检验候选依赖性。
- 判据（预注册）：IVJP 缺陷相对量应显著小于 SA 缺陷相对量
  （配对比较，报告二者与比值）；无数值硬 gate（诊断性质）。

## 5. VJP 候选共享伴随（A0H-1.5 主实现）

- CPU 合成：≥1 个**非对称**线性算子（显式矩阵 K≠Kᵀ）上验证共享伴随能复现
  K 的**精确**转置作用（中心差分/解析对照 ≤1e-6），以捕捉"K 代 Kᵀ"实现错误。
- 封装器 forward 与冻结切线步一致（create_graph 前后输出对齐，容差 1e-12 CPU / GPU 同进程位级）。
- 生产打分器：**w_0 只由公共基轨迹 + 同一终点梯度 v_H 导出**；
  **不得读任何候选未来切线 x_k（k≥2），不得跑 M 次响应**；
  分数 `ĥ_j = w_0ᵀ a_j`。
- 检查 G_0 与完整 θ/m/v 步转置（封装级 FD 对照）。
- 若任何伴随步需要候选特定 x_k ⇒ 标 CANDIDATE_DEPENDENT_VJP（Branch B 证据）。

## 6. 分支（客观判据，自动执行，不问）

对每个判定给出 per-item 状态 ∈
{PASS, FAIL, NOT_SUPPORTED, MISSING_DATA, NOT_COVERED, NOT_RUN, PARTIAL}。
所有分支输出：chosen_id, choose_noop, noop_score, best_active_score, margin，
及有标签时的实际 Gain/harmful/Regret。no-op 零分是定义，不等于选择 no-op；
无 oracle-noop 样本 ⇒ 拒绝边界记 NOT_COVERED。
原始 BF16 逐元 gate 结果沿用 A0H-1 的 FAIL（NUMERICAL_LIMIT），**不改判**。

- **Branch A**（全部满足才进入）：
  (A1) BF16 封装器共享伴随在**原始冻结容差** atol=1e-6/rtol=1e-4 下逐元匹配冻结 BF16 前向；
  (A2) 独立转置验证通过（封装 VJP vs 逐步解析转置，GPU 同进程，判据 ≤1e-3 相对，
  依据 BF16 地板；CPU 合成 ≤1e-6）；
  (A3) 共享性：w_0 在候选锚点间最大相对差 ≤1e-3；
  (A4) 决策 gate：3 诊断单元 × H∈{1,4,8} 排序+选择+no-op 与冻结前向一致（同 A0H-1 判法）。
  全过 ⇒ 在**独立验收集 A**（读一次）验证决策 gate；
  通过 ⇒ 自动进入 §7 计时（A0H-2 小规模）。
- **Branch B**（任一失败 ⇒ 记录并继续，不问）：
  把 **FP32-HVP 伴随**（仅 7 个 `_hvp` 调用点换 nullcontext，§2 定义）注册为
  **不同配置候选**：
  (B1) 先过同配置 FP32-HVP **前向** vs FP32-HVP 伴随的逐元检查（新容差对
  atol32=1e-5, rtol32=1e-4——依据 fp32 机器精度与前向自身重复性，预注册）；
  (B2) 再与**冻结 BF16 前向 C** 比选择：不得把 `|adj−ref32| ≤ |frozen−ref32|+tol`
  作为唯一标准；必须 (i) 逐元报告两式，(ii) 决策级（排序/选择/no-op）匹配，
  或仅预指定近并列（阈值 2·(atol+rtol·max|ĥ|)，同 A0H-1）且回放证据显示
  mean Gain 不下降且无新增 harmful 选择；
  (B3) **不得**追认"BF16 同配置等价"；原始 gate 的 FAIL 保留。
  在独立验收集 A 上评估所选动作；若冻结 dev 回放收益列（λ 扫描 dev-only CSV）
  可用则复用，否则做小次新回放：≤4 候选、H≤8、新目录 outputs/a0h15_vjp_shared_v01/accept_replay/。
  FP32 计时条件：B1 PASS 且（决策匹配冻结 C 于独立验收集，或仅预指定近并列+
  回放证据）。只宣称"小规模 dev 筛选"，不宣称统计非劣效。
- **Branch C**：资源/预算（GPU 30 分钟累计）耗尽 ⇒ 保存阶段产物，继续离线部分，
  交付 WAITING_FOR_GPU 与精确可恢复命令，不建无限 waiter。

## 7. 计时（A0H-2 小规模，条件触发）

- 生产入口 = 公共基轨迹 + 终点 V_score 梯度 + 单次 backward + 候选梯度 + 点积；
  诊断探针不进生产计时。
- 主比较：真实 +λ 回放 vs 冻结前向 C vs 生产伴随；B 作补充。BF16/FP32 标注。
- 冻结矩阵：dev 单元 1 × H∈{1,4,8} × M∈{1,2,4}；预算允许再加 ≤2 单元 × H=4/M=4。
  M=非零候选数；M=1/2 取固定顺序前缀。M>4 未授权。
- 1 次预热 + ≥3 次完整重复，方法顺序预固定交错；同步墙钟 + CUDA 时间 + 峰值显存。
- 独占 gate：util 中位 <10% 且无其他计算进程才算正式计时；否则标 shared，
  不报告加速结论。不杀外部任务。本轮新增 GPU 累计墙钟 ≤30 分钟（含诊断/计时）。

## 8. 交付

更新 status/index/本协议执行记录/handoff；A0G/A0H-1 原 FAIL 保留；
审计包（不含权重/梯度；manifest 不含自身；tar SHA256 单独记录并验证 listable/unpackable）；
阶段产物按完成度打包。

---

## 附录 A（事后执行记录，非合同部分；2026-09-15）

本节为执行后补记，不改变上文任何冻结条款。

### A.1 GPU 预算执行
- 全部新增 GPU 作业累计 822s / 1800s（逐项见
  `outputs/a0h15_vjp_shared_v01/contract/gpu_budget_log.txt`），未触顶。
- 外部作业（PID 5027 等）未杀；本轮全部运行为 shared（GPU1 独占 gate 未满足）。

### A.2 诊断结论（D = 2031/52, 2034/64, 2036/76 × stream 0）
- CHK-LIN（D1）：K 确定性（重复间隙 0）、K(0)=0、负线性=0、α∈{0.5,2} 精确、
  α=1.5 与可加性 ~4–6e-3（BF16 噪声级）⇒ 算子线性成立（BF16 地板内）。
- CHK-SA/CHK-IVJP（D1）：SA 缺陷 ≈ IVJP 缺陷（~2e-2 rel @ H=8）⇒ 主导项是
  BF16 HVP 代理非自伴，不是伴随实现错误。IVJP 三阶链 35/35 可行
  （预检 gap 2.8e-4）。
- CHK-TELE：D1 最差 8.9e-3 > atol_tele=1e-3 ⇒ 望远镜闭包**未达**预注册容差，
  如实记 FAIL；按 §3 定义，闭包是一致性诊断，不证明根因。
- 共享伴随 ĥ_adj vs 冻结前向 ĥ_fwd（原始 gate atol=1e-6/rtol=1e-4，D1）：
  H=1 max|Δ|=5.0e-11 PASS；H=4 max|Δ|=1.36e-6 PASS（容差 1.84e-6）；
  H=8 失败但与冻结 C|8 FAIL 一致（继承 BF16 地板，非回归）。
  共享伴随与冻结伴随实现偏差：H=1/4 ≤1.3e-11，H=8 ~1e-6（代理噪声级）。
  同进程位级检查：共享 ĥ == 冻结 joint pass ĥ_fwd（确定性探针 PASS）。
- 决策保持（dev）：D1 全 H 排序/选择一致；B1 配置下 3 单元 × 3 H 全一致。

### A.3 分支执行
- Branch A 进入后，验收集 A 在调试中暴露 accept runner 重建层 bug
  （望远镜恒等式在 PRE-step x 约定下重建出 H−1 读出，fwd 全零）。
  按 §1 读一次规则，**A 批（2032/58, 2035/70, 2037/46）整批降为开发数据**，
  记录于 `outputs/a0h15_vjp_shared_v01/contract/acceptance_disposition_readonce.md`。
  Branch A 独立验收 = NOT_COVERED（NOT 自动进入 §7）。修复后以 dev-validation
  身份重跑（`accept/*.json` 已标 DEV_VALIDATION_ONLY；v9 无效输出归档
  `accept_v9_invalid_archived/`）。
- Branch B：B1 PASS（FP32-HVP 同配置前向 vs 共享伴随，3 单元 × 3 H，
  max gap 4.1e-10 << atol32=1e-5）；B2 PASS（与冻结 BF16 C 的 5 键全排序
  含 no-op 9/9 完全一致，选择全一致；两式逐元并报）；B3 遵守（不追认 BF16
  同配置等价，A0H-1 原 FAIL 保留）。FP32-HVP 注册为不同配置候选（dev 级证据）。
- §7 计时：NOT_COVERED。触发条件要求独立验收集上的决策匹配；A 批已降级，
  本轮只有 dev 级筛选证据，按 §6 不宣称、不触发。
- Branch C：未进入（预算未耗尽；交付不含 WAITING_FOR_GPU）。

### A.4 修正与不变量
- A0G 原四份 CSV、原 runner、λ/Rule B、split、候选与读出协议、A0H-1 原结果
  与原 gate 全部未改（A0H-1 原 FAIL 保留）。
- 冻结源哈希：18/18 未变（contract/source_hashes_start.txt 逐项复核，
  结束时再核一次见 handoff）。
- A0H-1 审计包 SHA256 复核 = 3b237946…db6c93，一致。

### A.5 第二轮执行记录（2026-09-15，DEV_TIMING 修正案授权）

- **顺序**（用户本轮 mandate）：dev 核账与生产入口验证 → 开发集工程计时 →
  条件满足后一次 A′ 独立验收。全部自主完成。
- **§1 核验**：第一轮审计包字节/SHA256 复核一致。
- **§2 台账**：`ledger/comparison_ledger.{csv,json}`（180 行，方法对两端显式）
  + README 更正"位级一致"声明（探针崩溃未完成；H1/H4 ~5e-11、H8 ~1e-6；
  与 H8 gate FAIL 无矛盾——不同比较对）。
- **§3 簿记**：6 运行 FP64 闭合全 CLOSED（max rel 1.9e-16）；不证明伴随正确。
- **§4 生产入口**：`research/scripts/a0h15_production_entry_v01.py`（三模式互斥）。
  修复两个正确性 bug：clip/step 冻结次序（切线与 HVP 必须在 scaler.step 前，
  读 PRE-step m/v/step、在 step-h 权重上跑 HVP）；快照 step_after 用优化器内部
  step+1（非 lr 调度 micro-step 计数）。自校验：FP32 fwd/adj 与冻结 pass 同进程
  位级相等（gap=0）；同配置 B1 gate max 1.2e-10；配置错配（dtype、hidden_size
  512 漂移）预读 AssertionError；ENGINEERING_ELIGIBLE=PASS（仅工程资格）。
- **§5 计时**（DEV_ENGINEERING_TIMING，GPU1 独占实测 avg util 6.2%）：H4/M4
  fp32_fwd 3.403s / fp32_adj 1.321s / bf16_fwd 3.631s / replay 1.546s；
  R_structure=2.577、R_replay=1.170、R_frozen=2.750；分数跨重复逐位一致；
  扩展矩阵 H{1,4,8}×M{1,2,4}（7 组合）同场同法。bf16 HVP 版本在 4.98M 模型上
  略慢于 fp32 版本（kernel-launch 开销主导的假设，仅作备注）。
- **§6 A′**：预注册 3 单元（2030/52、2033/64、2037/82；∉ D ∪ 降级批 ∪ 本轮
  dev），重试规则先冻结；驱动两次基础设施崩溃（序列化、键类型）均非评分逻辑，
  冻结模块哈希全程未变。结果：L1 B1 gate 3/3 PASS；BF16 前向 vs FP32 伴随排序
  9/9 完全一致；**L2 3/3 单元 verdict FAIL**（ĥ-argmax 行动 realized V_score
  gain 为负：−2.5e-4/−7.3e-5/−2.8e-4）。诚实报告，不重试。失败定位于决策规则层
  （响应代理 argmax ≠ 效用 argmax），伴随数值保真度证据完整。
- **不变量**：A0G 4 CSV、原 runner、λ/Rule B、split、A0H 原 gate 只读未变；
  冻结源哈希未变；GPU 累计 ~1270s/1800s。

## 附录 B（事后执行记录，非合同部分；2026-09-15 决策语义审计）

- **审计范围**：仅离线复核 round-2 已存制品（无 GPU、无新实验、无新 A′/A″）；
  回答 "A′ L2 3/3 negative gain 属 A/B/C/D 哪类"。制品：
  `outputs/a0h15_vjp_shared_v01/decision_semantics_audit/`（17 件 + SHA256SUMS）。
- **结论：Branch A**。hhat 语义 = d L_score(θ_{t+H})/dλ（Case 1，无符号变换）；
  冻结 A0G selector = argmin predicted ΔL（j=0..4 含 no-op，tie 最小 j，
  `fr/a0g_decision_utility_v01.py:225-229`），A0G formal 360/360 交叉验证；
  预修复 A′ 驱动（哈希 937798f6…417fe10，已存档）用 raw ĥ argmax（j=1..4），
  且预注册文本 "argmax-ĥ" 条款本身写错 → 非 Branch B；ID/no-op/标签/序列化
  36/36 恒等式全过 → 非 Branch C。
- **历史解释更正**：附录 A 中"失败定位于决策规则层（响应代理 argmax ≠ 效用
  argmax）"的归因错误，真实根因是验收驱动 selector 方向错误。原 A′ 3/3 FAIL
  记录保留；已执行 A′ 按冻结重试规则 2 降级 dev-investigation（非有效 FAIL
  亦非 PASS），本轮无 A″。
- **离线纠正重评分（已有五动作真实标签）**：冻结规则 3/3 选中 replay
  gain-argmax 最优动作，gain 全正（+8.91e-4 / +3.58e-4 / +2.35e-4）；
  仅作 dev-investigation 证据，不追认为 unseen confirmation。
- **修复与测试**：`a0h15_accept_aprime_v01.py` 恢复冻结 select_action
  （frozen_decision）；新增 `research/tests/test_a0h15_selector_semantics_v01.py`
  7/7 通过（含顺序不变性、tie-break、A′ 标签离线重评分）。冻结 a0g 模块、
  λ、动作定义、AdamW、gate 一律未动。
- **计时口径不变**：DEV_ENGINEERING_TIMING；R_structure=2.577、
  R_replay=1.170（speedup）与 time reduction=14.55% 分别陈述；未重跑 GPU。
