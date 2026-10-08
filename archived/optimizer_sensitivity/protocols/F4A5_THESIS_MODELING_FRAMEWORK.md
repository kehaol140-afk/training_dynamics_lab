# Thesis Modeling Framework — State-Space Sensitivity → Stability-Aware Optimization

（毕设用框架整理；所有断言的实证等级见 RESEARCH_SYNTHESIS。）

## 1. 问题定义

在 long-horizon 学习率超参优化（hypergradient / sensitivity 估计）中，为什么以及何时反向展开（unrolling）会失效？我们将 AdamW 训练建模为增广状态动力系统，研究其切线动力学的有限时间稳定性，并利用结论设计在线安全展开策略。

## 2. 增广状态空间

z_t = (θ_t, m_t, v_t)（参数 + AdamW 一二阶矩）。一步映射 z_{t+1} = F(z_t; η, batch_t)。bank checkpoints = 该系统的冻结轨迹采样（5M/26M/100M × 多 seed × 多 phase）。

## 3. Jacobian 线性化（切线传播）

对 log η 通道的敏感度由齐次切线递推 dz_{h+1} = J_h(α) dz_h + s_h 刻画：
- **A 部分**（α=0）：optimizer-only 状态传播（vendored AdamW tangent）；
- **C 部分**（α=1 增量）：curvature feedback（clip_jvp∘HVP）；
- s_h：LR 非齐次源。

## 4. Optimizer-state conditioning（F0/F1）

一步 Jacobian 在 c=0 锚点的病态：v→θ′ 通道按 v̂^{-3/2} 爆炸（100M step26 单步 A_v ~ 1e9-2.4e9），由 second-moment floor 因果主导（τ=eps 修复 70% σ_max）。**病态是基础设施，不是引爆本身。**

## 5. Curvature feedback 与有限时间放大（F2/F3）

- α=0 有界（G_B_32 ≈ 20-25 全 scale/phase）；α=1 在 EARLY 放大 10²–10⁵（R_curv_32 至 1.6e7）。
- source-off impulse 仍增长（λ_SO_32 > 0 于全部 90 观测）→ propagator 自身不稳定，非源累积。
- 局部谱：ρ(J_t, α=1) > 1.05 可靠成立于全部 cells/offsets（MIDDLE 亦然）——谱不稳定存在但不区分 phase（σ_max ~1e26：v-floor 病态方向）。
- 模块机制 DISTRIBUTED；单一 unstable eigenspace 解释 METRIC_DEPENDENT（F3.5）。

## 6. Finite-time propagator 的宏观定律（F4A/F4A.1/F4A.5）

- **log10 R_curv_32 ≈ progress 的稳定凹函数**（LOSO MAE 0.43 dex、Spearman 0.75；M0 最优模型）：curvature 过剩放大普适自愈（90/90 观测 LATE 为 curvature-neutral）。
- scale 效应：R_curv 非单调（5M≈26M≪100M），但 **EARLY λ_SO_32 随 N 单调（+1.0）**——指数增长率有 scale 结构，相对放大比没有。
- seed 随机效应占 41.8% 方差且不可跨轨迹迁移——**宏观可预测、个体不可预测**。
- cheap indicators（A_v/A_m/clip/K_lr/vfloor）在控制 progress 后无独立预测力（共线假象）。

## 7. Hierarchical stochastic model（实证骨架）

y_R(s, k, p) = f(p) + u_seed + ε，u_seed ~ (0, σ²_between=0.42·σ²_total)；f 为凹（Huber 稳健）；scale 项 ~ 0（非单调）；分类风险可排序（AUROC 0.84）但不可阈值决策（balacc 0.54）。

## 8. Stability-aware optimization（F4B，SAU）

不给先验模型（F4A.5 gate 禁用 prescreen），而在展开中在线监测当前轨迹的 g(H)=‖dθ_H‖/‖dθ_1‖ 与增量比 q(H)，按冻结预算网格选择 H_safe，再以 h(H_safe) 做线性超参决策；在同一 CRN future 上以真实 ±δ rollout 的 decision regret 评价（详见 F4B 报告）。

## 9. Limitations

- 覆盖：5M×8seeds（matched 三点 + 原 7 点）、26M/100M×3seeds×3 点；p>0.8 无观测；1B 未触及。
- 跨环境（fr_minimind→minimind-exp）分布等价（canary 判据），非 bitwise。
- R_curv 度量依赖 α=0/α=1 双臂的比值语义；F3.5 显示 eigenspace 几何 metric-dependent——本框架只用参数通道比值，避免该问题。

## 10. Future scale validation

按 F4A MISSING_COVERAGE_REQUESTS 优先级补点；若 300M/1B 可用，验证：(i) EARLY λ_SO 单调性是否延续；(ii) R_curv 非单调（26M 凹陷）是否消失；(iii) p_heal 后移率。
