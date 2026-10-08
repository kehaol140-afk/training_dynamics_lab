# Research Synthesis F0 → F4B（2026-09-30）

证据等级：ESTABLISHED（冻结管线 + 逐位/分布 gate 验证）/ SUPPORTED（跨 seed/phase 复现）/ REJECTED（预注册否定）/ OPEN。

## ESTABLISHED

1. AdamW 一步切线 map 病态随规模恶化、随训练自愈（F0：26M/100M A>1；FD 验证 1e-11）。
2. v-floor 因果主导单步病态（F1：τ=eps 触 0.05% 前向 → σ_max −70%；A_v ~ 1e9-2.4e9@100M step26；fp32/fp64 一致排除精度伪影）。
3. EARLY 真实 hypergradient/长程敏感性脆弱（F2：EARLY 2/3 cells rel_err >1@H16；MIDDLE 可用）。
4. curvature feedback 是 EARLY 多步过剩放大的必要驱动（F3 α-sweep：α=0 有界 ~25、α=1 至 1.6e7；source-off λ32 > 0 于全部观测）。
5. 局部谱不稳定真实且普遍（F3 ρ>1.05 于 3 cells×5 offsets 全可靠成立，含 MIDDLE）——但**不区分 phase**（F3.5 确认）。
6. Phase 自愈定律（F4A/F4A.1/F4A.5：90 states）：log10 R_curv_32 ≈ 凹函数 of progress；LATE 全部 curvature-neutral；M0 LOSO MAE 0.43 dex。
7. Seed 异质大且不可迁移（41.8% 方差；同 (scale,p) 极差至 5.3 dex；M4/M5 泛化更差）。
8. Matched 覆盖下 R_curv 无单调 scale 律（5M≈26M≪100M），EARLY λ_SO 随 N 单调（+1.0）。
9.（执行层）长任务存活协议：linger + setsid/tmux + 原子 resume + 逐位 gate——四次进程死亡均被 gate/审计捕获且零数据污染。

## SUPPORTED

- 机制链的当前最佳表述：**病态基础设施（v-floor）常在 + curvature feedback 在早期窗口被方向选择性激发/维持（耦合结构 metric-dependent）+ phase 主导的自愈**。"stable training, unstable sensitivities"。
- 模块/子空间机制 DISTRIBUTED（F3 RQ3 + F3.5 projector 非必要）。
- A_theta/A_v/clip 等池化相关是 progress/LR 共线的假象（控制后无独立解释力；F4A.5 Huber 符号翻转）。
- 长程 unrolling 的决策风险（F4B：FIXED16/32 harmful 率最高、dominant 子集 regret 中位 0.317 vs H=1 的 0）。

## REJECTED

1. v-floor → long-horizon trust-region collapse 单因果链（F2.5：4 种 backward-only 干预 δ* 不变）。
2. 单一 attn-Q（或任何单模块/单 eigenspace）机制（F3 DISTRIBUTED；F3.5 METRIC_DEPENDENT + projector 移除不消除增长）。
3. 简单确定性 scale 律（F4A/F4A.1 非单调；F4A.5 scale 项不增益）。
4. 跨轨迹 cheap prescreen（F4A.5 gate：AUROC 0.84 但 balacc 0.54 → 禁用）。
5. 在线 adaptive unrolling 的增益（F4B：SAU 退化为 H=1，NO_METHOD_GAIN @ δ=0.05/0.02）。

## OPEN

- EARLY λ_SO 单调 scale 结构的机制（为何指数率随 N 增而相对放大比不增）。
- p>0.8 之后 λ_SO 是否趋零；p_heal 的 scale 后移是否结构性（覆盖受限）。
- 5M EARLY 峰值与 26M 凹陷（覆盖/种子混合？）。
- 更大 N（300M/1B）验证；放宽 SAU 预算网格（B_λ≥0.5）后的自适应价值（须新预注册）。
- δ 之外的动作空间 / 非线性决策规则是否有增益。

## 统一证据链

F0 病态 → F1 v-floor 因果 → F2 下游脆弱（EARLY-specific）→ F2.5 否定单因果 → F3 curvature+谱不稳定（机制层）→ F3.5 否定单一子空间解释（几何层）→ F4A/A.1/A.5 phase 定律 + 异质 + 无 scale 律（宏观层）→ F4B 方法层诚实负结果（H=1 近优；自适应无增量）。每一层的负结果都约束了下一层的假设空间。
