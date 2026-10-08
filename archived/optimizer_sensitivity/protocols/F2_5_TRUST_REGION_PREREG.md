# F2.5 — Trust-Region Collapse 预注册（100M，LR channel）

- 冻结日期：2026-09-26，在任何 F2.5 计算之前。上游：F1 `V_FLOOR_CAUSAL`、F2 `PRACTICAL_HYPERGRADIENT_FAILURE`（EARLY 2/3 cells）。
- 目标：验证 "hypergradient trust-region collapse" 并闭合 F1 v-floor → F2 practical failure 的因果链。不开发任何方法；forward AdamW / frozen future stream / CRN 全部不变。
- cells（与 F2 完全相同）：EARLY `2042:26`、EARLY `2043:26`、MIDDLE `2042:64`（G1b 100.78M bank，只读）；future stream = `derive_stream_seed(20260926, {1,2,3})`，batch manifest SHA 与 F2 相同；L_val = partition_v07 `eval_indices`（128 条）fp32 无 autocast 前向。

## 1. 冻结的扰动网格与 horizon

- 对称 log-LR 网格：δ ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1}（± 两侧完整 rollout，共 16 臂 + 1 个 δ=0 baseline，每 cell）。
- H 读出 {1, 2, 4, 8, 16, 32}（嵌套于 32 步 rollout）。

## 2. 冻结的量与判据

每 cell × H × δ 记录：
- `ΔL_H(δ) = L_H(η·e^δ) − L_H(η)`（CRN：同 manifest、同 batch、同 dtype 路径）；
- exact tangent hypergradient `h_H`（fp64/C：vendored 切线 + fp32-HVP 曲率反馈 + 非齐次 u 项，即 F2 管线；fp64/B 记录为对照）；
- 一阶预测 `h_H·δ`；
- `R_H(δ) = |ΔL_H(δ) − h_H·δ| / (|h_H·δ| + ε_metric)`，**ε_metric = 1e-7 nats（冻结）**；
- central FD `D_H(δ) = [L_H(+δ) − L_H(−δ)]/(2δ)`。

**linearization_usable(δ,H) := R_H(δ) ≤ 0.2 AND sign(h_H·δ) == sign(ΔL_H(δ))**（冻结阈值）。

**可解性（resolvable）**：`RESOLVABLE(δ_i,H) := |ΔL_H(±δ_i)| > ε_abs AND sign(ΔL_H(δ_i)) == sign(ΔL_H(δ_next))`，其中 ε_abs = **1e-6 nats（冻结）**，δ_next 为网格中下一个更大值（最大 δ 用自身符号）。不可解的 δ 标记 `BELOW_NUMERICAL_RESOLUTION`，**不得**通过提高读数精度伪造信号；usable 判定只在 resolvable 的 δ 上进行。

**trust radius**：`δ*(H) = max{δ : 所有更小的 resolvable δ 也 usable}`（无任何 usable δ 时 δ*(H)=0 并记录；全部 δ usable 时 δ*(H)=1e-1 即网格上限）。

**FD 收敛检查（取代单一 δ ground truth）**：对每 H 报告 D_H(δ) 全网格 + 相邻比稳定性 `|D(δ_i)−D(δ_next)|/|D(δ_next)|`；`h_H^FD := D_H(最小 resolvable δ)`，若 D 在 resolvable 范围内漂移 >20% 标 `FD_NONCONVERGENT`。Richardson 视图：对最小三个 resolvable δ 拟合 `D(δ)=h+c·δ` 外推 δ→0（报告，不作 gate）。

## 3. 冻结的 backward-only oracle interventions（机制验证专用，不得称为方法）

forward trajectory 逐位保持原 AdamW rollout（干预只存在于 tangent/JVP 数学内部）。切线递推的 per-parameter 数学与 vendored 冻结函数逐项相同（单测：无干预位级相等），仅以下修改：

| intervention | 定义（冻结） |
|---|---|
| `none` | 原样（= F2 管线） |
| `mask_den_0.01%` | 在 v→θ′ 通道项 `−m̂·dsqrt/denom²` 上，把 F1 R_den top 0.01% 坐标（按 **bank 锚点态** R_den=1/(√v̂₁+eps) 全局降序，坐标集冻结后逐步不变）的贡献置零；dm̂/denom（m 通道）不动 |
| `mask_den_0.1%` | 同上，top 0.1% |
| `vfloor_eps` | 切线线性化仅在数学上以 `v_back = max(v, ε²)`（ε=1e-8）替代 v（M、Dv 两块的公共分母；F1-B 语义），forward 的 m/v 演化不变 |
| `vfloor_10eps` | 同上，τ=10ε |

对每种 intervention 重算：tangent norm growth（‖dθ/du‖_h 逐步）、`h_H`、`R_H(δ)`、`δ*(H)`（用与 baseline 完全相同的 ΔL_H(δ)——forward 未变，ΔL 复用）。

## 4. 冻结的关联分析（n=3，如实标注）

per cell 记录（取自 F0/F1 制品，只读）：
- F1 `A_v`（v 通道锚点放大，state_channel_amplification.csv，fp64 双 seed 均值）；
- F0/F1 `σ_max`（power_iteration.csv / G 通道对角 max）；
- `attn_q risk share`（module_localization.csv G2_share_attn_q）；
- `floor coordinate fraction`（F0 per_parameter_group.csv：∑(√v<10eps 分数 × 组坐标数)/总坐标）。

对每个 H：把以上 4 个 conditioning 量与 `δ*(H)`、`tangent norm growth`（‖dθ/du‖_H32/‖dθ/du‖_H1）做秩方向对照（n=3 只报告方向一致性，不声称显著性）。

## 5. 冻结的判定程序（按序，首个命中）

定义 collapse 模式：**`δ*(H=16) ≤ δ*(H=1)/10` 或 `δ*(H=16)=0`（在 EARLY cells）**。
定义 re-expansion：**某 v-floor 干预（mask 或 vfloor）使 `δ*_intervened(H) ≥ 10 × δ*_baseline(H)` 于同一 EARLY cell 的 collapse H**。

1. `FD_RESOLUTION_LIMITED`：在 ≥2 个 (EARLY cell × H≥16) 组合上，整个 δ 网格无 resolvable δ；
2. `TRUST_REGION_COLLAPSE_CAUSALLY_LINKED`：collapse 模式成立 且 存在 re-expansion，且 MIDDLE cell 无 collapse（或 collapse 弱得多：δ*_MIDDLE(16) > δ*_EARLY(16)）；
3. `TRUST_REGION_COLLAPSE_CORRELATED_ONLY`：collapse 模式成立 但 无任何 re-expansion；
4. `TANGENT_CORRECT_BUT_FINITE_RESPONSE_NONLINEAR`：collapse 模式不成立，但存在 (cell,H) 使 R_H(δ)>0.2 于 δ≥1e-2，同时 `|h_H − h_H^FD|/|h_H| ≤ 0.2`（切线是真导数，失效源于真实曲率）；
5. `NO_TRUST_REGION_COLLAPSE`：≥2/3 cells 全部 H 有 δ*(H) ≥ 2e-2；
6. `UNRESOLVED`：其余。

## 6. 冻结的输出与红线

- `docs/F2_5_TRUST_REGION_REPORT.md`
- `outputs/f2_5_trust_region/{summary.json, trust_radius.csv, fd_convergence.csv, backward_intervention.csv, conditioning_vs_radius.csv}`
- 红线：不开发 F3 方法、不加 optimizer、不下 1B、不改 forward AdamW、不为判定调阈值；负结果原样保存。切线干预实现必须通过"无干预 == vendored 冻结函数位级相等"单测后方可用于实验。
