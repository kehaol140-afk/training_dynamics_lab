# G1C Feasibility Protocol — Documented Amendments (pre-result)

> 记录时间：2026-09-17（任何 Phase 1/2 GPU 结果之前）。
> 机制：与 §6.2 npy-header 修正同一机制——不改冻结协议正文，
> 以本文件登记。三条修正均使本 gate **更靠近**冻结 v07/G1a/G1b/a0c
> 已验证语义，不引入任何新科学设计。修正理由全部登记在案。

## A1 — q 归一化与 ε 口径（修正 §6.1）

- §6.1 原文："q = 全参数 iid N(0,1)；σ_q = q 元素标准差（目标 1.0，
  实测记录）；ε = 0.25 × σ_q"。
- 修正后（与冻结 v07/G1b 一致）：
  1. q 生成：全参数 iid N(0,1)（full support，禁止参数子采样——红线
     不变），随后 **归一化到单位总 L2 范数**（‖q‖₂=1，v07 direction
     惯例；q SHA256/norm/σ 全部记录）。
  2. σ_q = 冻结 stream 基线梯度投影 s = ⟨q, g⟩ 样本的标准差
     （v07 E1 / G1b 规则；样本 = 基线 run 的 microbatch 梯度投影），
     **不是** q 元素标准差。
  3. ε = 0.25 × σ_q（RHO=0.25 不变）。
- 理由：若按原文（σ_q≈1 → ε≈0.25，且 q 不归一），±εq 脉冲总范数
  ~0.25×√1.48e9 ≈ 9600，对 ‖g‖~O(1) 的梯度是 4 个数量级的脉冲，
  全局 clip 深度饱和，FD 落在 clip 非线性区之外，切线 JVP 无法对应，
  H=1 analytic identity 按构造必失败。修正后脉冲范数 ~ε≪‖g‖，
  处于线性响应区，与冻结 MiniMind 实验同一 regime。

## A2 — 可观测量 X（修正 §6.4）

- §6.4 原文："X = 窗内全部 token 的 mean cross-entropy"。
- 修正后：X_h = ⟨q, θ_h − θ₀⟩（参数位移投影，fp32 master 权重精确
  计算，不经前向）。future 梯度 g_h 仍来自同一 mean-CE 训练目标
  （前向 + 反向照常执行，bf16 autocast），只是 FD 差分的标量读出
  改用位移投影。
- 理由（数值可行性，feasibility 核心问题）：
  - 1B 模型在注册 ε 尺度下，mean-CE 前向读出的 FD 信号
    |X⁺−X⁻| ~ 2ε·|dX/dε| ~ 1e-9 量级；
  - bf16 autocast 前向在 X≈7 处 ulp ≈ 3e-2，信号被前向量化噪声
    淹没 ~7 个数量级；即使 FD 对改用 fp32 前向（ulp ~5e-7），
    线性区 ε 下信号仍低于可读门限；
  - 参数位移投影是冻结 v07/a0c theta-response 框架的注册可观测量，
    fp32 下精确可算（Δθ~2.8e-4，fp32 ulp ~1.5e-11，SNR ~10²），
    且驱动完全相同的 FD/B/C optimizer-tangent 机制阶梯。
- 本修正不影响 Phase 1 四项验证的任何一项；CE 前向/反向/clip/HVP
  全部仍被执行并记录。

## A3 — FD 分支脉冲语义（澄清 §6.1/§6.3）

- 澄清后（与冻结 v07 `apply_signed_gradient_impulse_` / a0c response
  框架一致）：
  1. FD 分支 = 从恢复 checkpoint 确定性重放的 native 基线，仅在
     **第一个未来步**的原始梯度上预 clip 加 ±ε·q 脉冲；
  2. 切线初始条件全零（θ 切线、m 切线、v 切线）；第 1 步梯度切线
     = q（B、C 相同，脉冲步即扰动机制本身）；
  3. h≥2：B 置未来梯度切线 = 0（optimizer-only）；C 置为该步
     future loss 对当前 θ 切线的 HVP（reverse-over-reverse，
     math-backend attention）；
  4. H=1 结构性恒等 B≡C 因此**按构造精确成立**（h≥2 项不存在），
     记录 |B−C|（期望 0.0）；ε 与 ε/2 两级 FD 对 chi_B 的相对误差
     （冻结 `relative_error`）探测有限 ε 非线性（clip 曲率）。
- 理由：消除 §6.1 "δ_g = q 单位脉冲" 的歧义（单位脉冲指方向 q，
  幅度 ε）；与冻结实现逐语义对齐。

## A4 — 序列长度级联扩展（修正 §6.7/B8，测量驱动）

- 实测（step950000，math-backend attention，create_graph 需保留两阶图）：
  - seq 2048：前向+一阶图保留即 23.9GB；block-checkpointed create_graph
    梯度 61.7GB / HVP 72.7GB 且病态慢（host 侧 101% 单核 CPU）；
    双反向总需求 ~45-60GB → 共享 GPU0（~16GB 空闲）不可行，
    独占 32GB 单卡亦不可行；seq 1024 同样不可行（~24GB）。
- 修正后：本 gate 全管线 SEQ_LEN=512（B8 级联 4096→2048→**512**）。
  每 microbatch 仍 4096 tokens（= 8 × 512 序列），token 预算与
  optimizer step 口径 §6.3 完全不变。HVP 分支 dθ 在 GPU 侧以
  fp16 存放（点积在 fp32 内完成）。
- 记录：2048/1024 的内存标度见 Phase 1 报告 notes，供正式 G1c
  科学实验在独占 ≥40GB 显存 GPU 上选择更长序列。
- **A4 事实更正（2026-09-17，Phase 1 复跑后）**："seq 512 峰值
  ~15GB" 的前提有误——**激活内存由 token 数决定，与 seq/batch
  切分无关**（8×512 与 2×2048 同占 ~24GB 一阶图）。seq 512 单独
  不能把双反向压进共享 GPU0：实测 HVP 段 GPU0 打满 31.8GB
  （分配器顶到上限后 cudaMalloc 重试，hvp_pass 单趟 >1.5h 未完）。
  真正的显存杠杆是**分批 chunk 化**（见 A5），不是 seq 长度。
  SEQ_LEN=512 仍保留（与冻结数据窗口 reshape 兼容、单序列粒度
  chunk 的数学精确性，见 A5），但其作用修正为：chunk 化的形状
  基底。

## A5 — HVP 分批 chunk 化（工程修正，数学恒等；修正 §6.7）

- 内容：hvp_pass 内每个 microbatch（8×512 tokens）沿 **batch 维**
  切成 1×512 的子块，逐子块做 create_graph 前向 + 一阶梯度 +
  ⟨g, dθ⟩ 点积 + 二阶反向，HVP 按等权（所有子块 token 数相同）
  在 fp32 CPU 侧累加后平均。
- 数学依据：目标函数为全体 token 的 mean CE，对 batch 行的切分
  是**精确线性**分解——chunked HVP ≡ 整批 HVP（至 fp 重结合
  误差，fp32 累加，记录 repeat_max_gap 于报告）。与冻结 v07/a0c
  JVP 语义零偏离；不动 clip/optimizer/FD 的任何语义。
- 依据的测量：双反向需在单卡同时驻留 params+g+hvp+dθ（bf16/
  fp16 各 ~2.9GB ≈ 11.7GB 常驻）+ 两阶激活图（4096 tokens
  ~45GB → 512 tokens ~5GB）。chunk=1 序列时峰值 ~15GB，适配
  共享 GPU0；每子块后 empty_cache + 低显存预检。
- 对 Phase 2 的影响：C 分支 h≥2 的每个 HVP 调用同此成本
  （记录于 Phase 2 报告的 per-cell timing）。


## A6 — C 分支切向量递归发散（测量发现；runner 逐分支隔离，语义冻结）

- 发现（2026-09-18，Phase 2 attempt 1）：(step=380000, stream_id=1)
  在 C 分支 h=6 发散。原始数据：
  - native h=6 出现梯度尖峰：|g|=322.029480（邻步 ~81–130），
    clip 系数 0.003105；
  - C 分支 chi 序列：h=1..5 = −1.850, −3.540, −5.067, −6.297, −8.943
    （与 B 分支一致到 ~1e-2），**h=6 跳到 +269.617**；
  - h=7 HVP 后 dθ 中出现 inf → tg 出现 NaN → 冻结
    `clip_jvp`（a0c_adamw_response_propagation_v01.py:85）按冻结
    行为抛出 `ValueError: gradient/tangent contains NaN or Inf for
    transformer.wte.weight`。
- 机制（分析，非新实验）：该 batch 的梯度尖峰经冻结 AdamW 切向步
  与 bf16 HVP 传播后，在 v̂≈0 的参数处触发冻结 JVP 的奇异性
  （项 m̂·dv̂/(2·v̂·den²) 随 v̂→0 爆炸，如稀有 token 的 wte 行），
  得到有限但巨大的 dθ_h → 下一步递归被 inf/NaN 污染。B 分支对同一
  数据未触发（B 不使用 HVP，dθ 量级保持正常）。
- 处置（工程性，**不动任何冻结模块**）：
  1. `cmd_phase2` 对 B / C / FD 三分支分别 try/except。某分支失败
     时该 (step, stream) 的对应 chi 记录为 NaN，行 status 记
     `partial`/`failed`，**其余分支与全部其余 cell 继续执行**；
  2. 失败原因原文记入 per_cell.csv 的 branch_error_* 列；
  3. 汇总指标（冻结 `_metric`）仅在有限值对上计算。
- 语义不变性：切向步、clip、optimizer、FD 脉冲、q、ε、可观测量
  全部照旧；隔离只改变**失败后的控制流**（abort → 记录并继续），
  对成功 cell 的结果零影响。
- 对正式 G1c 的含义：**记录，不在本 gate 决策**。该不稳定性是
  C 分支（HVP 递归）在梯度尖峰 batch 上的真实性质，正式实验设计
  （更多 stream/更大样本）需考虑 cell 级失败率与触发条件。
  本 gate 的 Phase 2 矩阵将如实给出每 cell 的 status。
