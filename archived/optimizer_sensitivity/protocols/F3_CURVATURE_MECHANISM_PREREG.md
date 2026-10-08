# F3 — Curvature Feedback Mechanism 预注册（100M，LR channel）

- 冻结日期：2026-09-27，**在任何 F3 数值计算之前**。上游：F0 `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE`、F1 `V_FLOOR_CAUSAL`、F2 `PRACTICAL_HYPERGRADIENT_FAILURE`、F2.5 `TRUST_REGION_COLLAPSE_CORRELATED_ONLY`。
- 动机：F2.5 否定了简单因果链 "v-floor → long-horizon trust-region collapse"（backward-only v-floor repair 不能恢复 trust radius）。F1 的局部 optimizer-state ill-conditioning 结论保留，但 F2 中真正需要解释的现象是：**full curvature-feedback branch C 在 EARLY phase 出现巨大的 multi-step tangent amplification，而 optimizer-only branch B 有界、同一模型 MIDDLE phase 的 C branch 也有界**。
- 核心问题：**为什么 EARLY Transformer training 中的 curvature feedback 会把 sensitivity 在十几到几十步内放大 10³–10⁵ 倍，而同一模型 MIDDLE phase 不会？**
- 本轮仍是机制研究，**不是方法论文实验**。禁止开发 optimizer / stabilizer / controller。只回答上述问题。
- 不做事后假设检验：所有网格、种子、阈值、判定程序在本文件冻结后不得调整；探索性发现只允许以 "exploratory" 标注报告，不得反向设为 PASS gate。

## 1. 冻结的实验对象（与 F2/F2.5 完全相同）

| 项 | 冻结值 |
|---|---|
| 模型 | MiniMind 100.78M（G1b 冻结配置：hidden 768 / layers 13 / heads 12 / kv 6 / inter 2432 / vocab 6400 / seq 128 / tied；`../minimind_base/model/model_minimind.py`，只读） |
| cells | g1b bank：EARLY `(2042, 26)`、EARLY `(2043, 26)`、MIDDLE `(2042, 64)`（只读） |
| 恢复 | bank state 的 model + AdamW m/v/step |
| future streams | 每 cell 一条，`stream_seed = derive_stream_seed(20260926, {1,2,3})`；`make_permutation(3936, stream_seed)` → batchify(4) → 每 optimizer 步 2 microbatch（accum 2）；rollout 只用 `train_indices`；`eval_indices` 永不进入训练 batch |
| LR | 原生 finite-cosine：`lr_h = get_lr(micro0 + (h-1)*2, 256, 5e-4)`；**F3 不做任何 δ 扰动**（无 FD 臂）；tangent 臂 LR 不缩放 |
| L_val | partition_v07 `eval_indices`（128 条）mean CE；fp32 无 autocast 前向 |
| H | {1,…,32} 逐步传播；读出点 H_READOUTS = (1, 2, 4, 8, 16, 32) |
| 训练语义 | bf16-autocast 前向/反向（GPU）、fp32 CPU 原生 AdamW、global clip 1.0（CLIP_DENOM_EPS=1e-6）——legacy 注册路径 |
| 设备 | `CUDA_VISIBLE_DEVICES` ∈ {0,1}；两 GPU 可并行不同 cells |
| 精度臂 | 唯一主臂：**fp64 tangent（CPU）+ fp32-HVP（GPU 双反向，math-SDPA）**，即 F2 的 `fp64/C` 精度语义；不跑 fp32/frozen/bf16 臂 |
| 停止规则 | 与 F2 相同：切线非有限或 ‖dz‖ > 1e30 → 该臂在该 H 停止并记录，其余臂继续；primal 非有限 → 该 cell 终止 |

## 2. 冻结的切线分解 J_t(α) = A_t + α·C_t

逐步定义（与 F2 管线逐步对齐，唯一变化是 dc 的构造）：

```
grads_h      = ∇L_batch(θ_h)                          # primal，bf16-autocast
clip:        c_h, coef_h                              # primal
dc_arm_h(α,mask) = α · mask ⊙ clip_jvp(g_h, HVP_h(dθ_arm_h))   # h≥2；h=1 时 dc=0
dz_arm_{h+1} = adamw_tangent_step(c_h, dc_arm_h, dz_arm_h) + s_h
s_h          = θ_{h+1} − θ_h（LR 非齐次源；source-off 臂 s_h ≡ 0）
```

- **逐臂 HVP 各自计算**：每臂的 HVP 输入是该臂自己的 dθ_arm_h（双反向每次一臂）。这使 J_t(α) 成为良定义的线性算子 A_t + α·C_t（C_t = ∂step/∂dc ∘ K_t 限制在 dθ 分量上，K_t = clip_jvp∘HVP_t）；α=1、mask 全开时与 F2 `fp64/C` 的逐步计算序列逐项相同（单测位级 gate 见 §7）。h=1 时所有臂 dz 相同（纯非齐次步），故 h=2 的 HVP 输入自然一致。
- A_t 对应 α=0（dc≡0，纯 optimizer-state 传播 = F2 B branch）；C_t 对应 α·(dc_full) 进入 tangent 的部分（= F2 C branch 增量）。
- 冻结 gate：**α=0 必须复现 F2 `fp64/B`；α=1 必须复现 F2 `fp64/C`**（容差见 §7）。否则停止主实验。
- 每个 (cell) 的全部臂共享同一条 primal rollout；primal 与 α、mask、probe、source 全部无关。

## 3. F3-RQ1 — curvature feedback strength sweep（α sweep）

- ALPHA_GRID = (**0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00**)（7 点，冻结）。
- 每 cell × α：32 步 forced 切线（s_h 正常加入）。逐步记录（h=1..32 全记）：
  `‖dθ‖, ‖dm‖, ‖dv‖, ‖dz‖, h_η(H∈H_READOUTS), dominant_module, module-wise tangent shares（F0 九组）, ‖HVP_h(dθ_α)‖, ‖g_h‖, clip_coef_h, loss_h`。
- 主指标（冻结公式）：
  **G_forced(H, α; cell) = ‖dθ_H(α)‖₂ / max(‖dθ_1(α)‖₂, 1e-30)**，H∈H_READOUTS。
- 主要问题：EARLY 是否对 α 呈强烈非线性/阈值式增长，而 MIDDLE 平滑有界？
- **禁止事后定义 α_c**。允许把观察到的阈值作为探索性结果报告（标注 `exploratory`），不得作为 PASS gate。

## 4. F3-RQ2 — impulse / homogeneous propagator（source-off）

- 目的：区分"每步持续 source 累积"与"未来训练 Jacobian product 本身放大"。
- 构造：第一步注入一次单位切线 q₀，此后 s_h ≡ 0（h≥1），仅传播 dz_{h+1} = J_h(α) dz_h。
- **q₀（冻结定义）**：theta-only 单位切线，方向 = LR source 在 h=1 的方向：
  `q₀.θ = (θ₁ − θ₀)/‖θ₁ − θ₀‖`，`q₀.m = q₀.v = 0`。θ₁−θ₀ 取自 primal rollout（与 α 无关）。
- **4 个冻结 random probes**（`torch.Generator().manual_seed(seed)`，CPU fp64 randn，单位 L2 归一化）：
  | probe | seed | 定义 |
  |---|---|---|
  | theta_only | 91001 | 仅 θ 通道非零 |
  | m_only | 91002 | 仅 m 通道非零 |
  | v_only | 91003 | 仅 v 通道非零 |
  | full_random | 91004 | 三通道 iid，整体归一 |
  | attn_q_localized | 91005 | 仅 attn_q 参数 θ 通道 iid，整体归一 |
- arms：α ∈ {0, 1} × 5 probes；3 cells。逐步记录 ‖dz‖、‖dθ‖、‖dm‖、‖dv‖（h=1..32）。
- 冻结指标：
  **G_impulse(H, α, probe; cell) = ‖dz_H‖₂ / ‖dz_0‖₂**；**λ_H = (1/H)·ln G_impulse(H, α, probe; cell)**。
- 判读（冻结）：若 EARLY α=1 在 source-off 下仍巨大 amplification，而 α=0 与 MIDDLE 不出现 → full propagator instability 得强支持。

## 5. F3-RQ3 — curvature module causal ablation（receiver/output-channel ablation）

- **forward trajectory 完全不改**。只在 dc_arm（该臂 curvature feedback 向量，clip_jvp 之后、进入 tangent 之前）按参数 group 施加 0/1 mask：`dc_masked[n] = dc_arm[n] · mask[n]`。
- **这是 curvature receiver / output-channel ablation，不是 Hessian source block 分解**。除非另行显式计算 H·(P_module δθ)，报告不得声称识别 Hessian input source。
- 参数 group 用 F0 taxonomy（`f0.classify_group`）：embed_tied / attn_q / attn_k / attn_v / attn_o / attn_qk_norm / rmsnorm / mlp_gate / mlp_up / mlp_down。
- 冻结 arms（全部 α=1）：

  | arm | mask 语义 |
  |---|---|
  | FULL | 全开（= α=1 参考） |
  | NO_ATTN_Q / NO_ATTN_K / NO_ATTN_V / NO_ATTN_O | 该 group 坐标置零 |
  | NO_MLP | mlp_gate+mlp_up+mlp_down 置零（aggregate，主 arm） |
  | NO_EMBED | embed_tied 置零 |
  | NO_NORM | attn_qk_norm+rmsnorm 置零（aggregate） |
  | ONLY_ATTN_Q / ONLY_ATTN_K / ONLY_MLP | 仅该 group 坐标保留，其余置零 |
  | NO_MLP_UP / NO_MLP_GATE / NO_MLP_DOWN | 副 arm（aggregate 的细分，报告为 secondary） |

- 每 cell × arm：32 步 forced 切线。指标：G_forced(H=16,32)、h_η、module shares、逐步范数；并记录 source-off 版本的 G_impulse(H=16,32)（α=1 固定、mask 变化，同表流程）。
- 冻结描述性阈值：**LARGE_EFFECT := 最终 tangent norm（H=32）降低 ≥ 10×**（相对 FULL）。仅描述性，不作唯一 PASS 条件。
- 关键问题：EARLY 中 remove attn_q curvature feedback 是否大幅降低 multi-step amplification？MIDDLE 是否效应较弱？

## 6. F3-RQ4 — local spectrum vs transient amplification

目的：区分 (A) true local spectral instability；(B) non-normal transient amplification；(C) time-varying/parametric amplification；(D) mixed。不预设 non-normal。

- **offsets**（相对 cell 起点）：(**0, 1, 3, 7, 15**)（冻结；offset o 即在 primal 前进 o 步后的 θ_{t0+o} 处取算子；o+1>32 跳过——本 grid 不会触发）。
- 每 (cell, offset, α∈{0,1}) 估计两个量：
  1. **σ_max(J_t)**：matrix-free power iteration on J^T J，σ_max = √λ_max(J^T J)。冻结：PI_ITERS=20，PI_TOL=1e-6（相对 σ 变化早停），PI_SEED=91011。
  2. **ρ(J_t)**：matrix-free Arnoldi（Krylov dim **ARNO_M=16**），Hessenberg 特征值最大模。冻结可靠性判据：残差 `h_{m+1,m}·|s_m| / |θ| ≤ 0.05`（s_m = Ritz 向量末分量）；不满足 → 该点记 **NOT_ESTIMATED**，**禁止用 σ_max 代替 ρ**。
- 冻结 non-normality proxy（仅当 ρ 可靠）：`σ_max / max(ρ, 1e-30)`；辅证：固定 probe 下 `‖J_t^k q‖` vs `ρ(J_t)^k` 的偏离（标注为 proxy）。
- JVP/VJP 实现：JVP = §2 递推的齐次部分（无 s_h）；VJP = 其转置（dc 映射的转置利用 H 对称性：dcᵀ(w) = coef·H·w − k·⟨g,w⟩·H·g，k 为冻结标量；实现须过单测：⟨Jq, w⟩ = ⟨q, Jᵀw⟩）。
- **禁止为得到 "non-normal" 结论调整 Arnoldi 配置**（迭代数/Krylov 维/残差容差全部冻结）。

## 7. 实现前 reproduction gates（先于任何 F3 主实验；任一失败 → 停止并修代码）

1. **G1（B 复现）**：α=0 跑出的逐步 `dtheta_u_norm` 与 `outputs/f2_hypergradient/raw_{cell}.json` 中 `fp64/B` 逐点相对差 ≤ 1e-12。
2. **G2（C 复现）**：α=1（mask 全开）与同一 raw 中 `fp64/C` 逐点相对差 ≤ 1e-12。
3. **G3（mask 中性）**：mask 全开 + α=1 与 α=1 无 mask 调用逐位相等（`torch.equal`）。
4. **G4（mask 不改 primal）**：任意 mask 干预下，primal 逐步 loss 与 δ=0 baseline rollout 逐步 loss 完全一致（容差 1e-9，覆盖 JSON round-trip）。
5. **G5（source-off 线性）**：source-off 传播对任意 probe 满足 J(a·q) = a·J(q)（数值容差 1e-12），且 s_h 全零；future batches / primal trajectory 与 forced 臂完全一致（同一 primal）。
6. **G6（VJP 正确性）**：随机 q,w 上 ⟨Jq, w⟩ = ⟨q, Jᵀw⟩（相对容差 1e-10）。
7. 单测覆盖：冻结常量、指标公式、module 分类完备性（F0 九组不重不漏）、mask 形状回归（F2.5 事故复测）。

## 8. 冻结的判定程序

### 8.1 RQ4 机制分类（每 cell 独立，α=1；术语按 §6 A–D）

冻结阈值：ρ 显著 > 1 的 margin = **1.05**；transient growth 阈值 = **G_impulse(H=16, α=1, q₀) ≥ 10**；α=0 对照阈值 = G_impulse(H=16, α=0, q₀) < 3；non-normal proxy 阈值 = σ_max > 2·ρ。

- **SPECTRAL_UNSTABLE**：≥2 个 EARLY offsets 上 ρ 可靠且 > 1.05，且该 cell 满足 transient growth 阈值。
- **NONNORMAL_TRANSIENT**：上述 EARLY offsets 上 ρ 可靠且 ≤ 1.05，且 σ_max > 2·ρ，且 transient growth 阈值满足、α=0 对照阈值满足。
- **TIME_VARYING_AMPLIFICATION**：无 spectral 证据（ρ NOT_ESTIMATED 或 ≤1.05 于全部 EARLY offsets），但 transient growth 阈值满足。
- **MIXED**：≥2 类同时有明确证据（明确证据 = 该类全部冻结条件成立）。
- **UNRESOLVED**：关键诊断（ρ 可靠性 / impulse 一致性）无法确立。

聚合：机制标签适用于 **≥2/3 cells 成立且至少覆盖 2 个 EARLY cells**；EARLY 两 cell 结论不一致 → 降级为 MIXED 或 UNRESOLVED（报告如实记录）。

### 8.2 F3 总 verdict（允许主标签 + 次标签）

按序检查（首个命中为主标签）：

1. **CURVATURE_FEEDBACK_CAUSAL**：α=0→1 sweep 在 ≥2/3 cells（含 2 EARLY）单调且 G_forced(H=32) 增幅 ≥ 10×，且 impulse（α=1, q₀）G_impulse(16) ≥ 10（即 growth 不是纯 source 累积）。
2. **ATTN_Q_CURVATURE_BOTTLENECK**：在 1 成立前提下，NO_ATTN_Q 使 G_forced(H=32) 降 ≥10×（LARGE_EFFECT）于 2 个 EARLY cells，且 ONLY_ATTN_Q 保留 ≥50% FULL 增长。
3. **SPECTRAL_INSTABILITY / NONNORMAL_TRANSIENT_GROWTH / TIME_VARYING_PARAMETRIC_GROWTH**：RQ4 分类结果直接映射（§8.1）。
4. **MIXED_MECHANISM**：≥2 个机制类证据同时成立。
5. **CURVATURE_CORRELATED_ONLY**：α 有效应但 module ablation 无主导模块 / 效应弥散，或 impulse 不放大（growth 主要由 source 累积解释）。
6. **NO_CURVATURE_MECHANISM**：α=0 与 α=1 growth 无差异。
7. **UNRESOLVED**：其余。

### 8.3 必须回答的五个问题

1. α=0 vs α=1 是否复现 B/C 巨大差异？
2. source-off impulse 是否仍出现 EARLY growth？
3. 哪个 curvature output module causally controls growth？
4. local spectrum 能否解释？
5. EARLY/MIDDLE 差异是什么？

## 9. 冻结的输出

- `docs/F3_CURVATURE_MECHANISM_REPORT.md`（明确区分 FACT / SUPPORTED_INTERPRETATION / HYPOTHESIS / NOT_ESTIMATED；无可靠 ρ 不得声称 non-normal instability 已证明）。
- `outputs/f3_curvature_mechanism/`：
  `summary.json, alpha_sweep.csv, forced_growth.csv, impulse_growth.csv, module_ablation.csv, module_only.csv, spectrum.csv, finite_time_growth.csv, curvature_ratio_angle.csv, raw/`。
- raw/ 按 RQ 分文件：`rq1_{cell}.json, rq2_{cell}.json, rq3_{cell}.json, rq4_{cell}.json, primal_{cell}.json`，**禁止覆盖已有 raw artifact**（resume 采用读断点续跑）。

## 10. 执行顺序与 resume

顺序：RQ1（α sweep）→ 完整性检查 → RQ2（impulse）→ 检查 → RQ3（module ablation）→ 检查 → RQ4（spectrum，最贵，最后）→ RQ5（分解，随 RQ1 数据免费计算 + 后处理）。每步允许 resume，禁止覆盖 raw。

## 11. 红线

不开发 F3 方法、不加 optimizer/stabilizer/controller、不做 gradient clipping 或 hypergradient clipping 方法、不下 1B、不训练新模型、不改 forward AdamW、不改 legacy、不改 F2/F2.5 已提交代码、不改本预注册的任何 gate/阈值/网格/判定程序、不为得到特定机制结论调整 Arnoldi 配置；负结果原样保存；不 push。
