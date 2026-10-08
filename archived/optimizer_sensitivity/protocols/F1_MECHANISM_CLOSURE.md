# F1 — Mechanism Closure：AdamW full-state map 切线病态的因果机制

- 日期：2026-09-26。仓库：`optimizer_sensitivity/`。规则先于计算冻结于 `configs/f1_mechanism_v01.json`。
- 输入（全部 READ-ONLY）：v07 bank（4.98M）、G1a bank（26.68M）、G1b bank（100.78M）；与 F0 完全同源。
- 前置：F0 verdict `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE`。本轮不开发任何 optimizer/stabilization/controller；forward AdamW 规则未动。
- 1B 计划已在任何计算前取消（见 `docs/F1_1B_REPLICATION_PREREG.md` 顶部取消声明；0 字节下载，legacy 未动）。

## 0. 总判定（F1_VERDICT）

**主标签：`V_FLOOR_CAUSAL`**；次标签（同一机制的解析面）：**v→θ′ 状态通道主导**（`OTHER_STATE_CHANNEL` 限定词：病态几乎全部住在 second-moment 通道的 Jacobian 块里，而该块爆炸正是 v-floor 的解析后果 `Dv ∝ m̂·v̂^{-3/2}`）。明确排除：`PRECISION_DOMINATED`（fp32 与 fp64 的放大率一致到 6 位以上，bf16 到 4 位且全部有限）；`NO_CLEAR_MECHANISM` 不成立。

一句话机制：**训练早期（26M/100M 的 step26 附近）注意力 Q 投影中大量坐标的 second moment 落到极低 floor（v~1e-17 量级），使完整优化器状态 map `z=(θ,m,v)→z′` 的 Jacobian 中 `∂θ′/∂v` 块按 `v̂^{-3/2}` 爆炸；对 v 通道单位随机扰动的单步放大在 100M step26 达 2.4e9（m 通道 143，θ 通道恒等于 1.0）。抬升 v-floor 因果性地、单调地修复 conditioning——τ=eps 仅触碰 396 个坐标、forward 单步更新只变 0.05%，就把 σ_max 砍掉 70%；τ=10eps 前向变化 3.6% 时砍掉 96.5%。primal 沿梯度方向安全（m̂-对齐方向放大 0.4–3.5），tangent 张在全空间生病——这就是 "stable training, unstable sensitivities" 的机制内核。**

## 1. 冻结口径与实现保真

- 锚点与 F0 相同：c=0（下一步零梯度）。该锚点上 full-state 一步 map 的 Jacobian 逐坐标解析（bank 无精确零 v，F0 已核验 74 态）：
  `dθ′ = (1−lr·wd)·dθ − M·dm + Dv·dv`，`dm′=β1·dm`，`dv′=β2·dv`；
  `M = lr·β1/(bc1·(√v̂₁+eps)) = (β1/(1−β1))·G`，`Dv = lr·m̂₁·(β2/(2·bc2·√v̂₁))/(√v̂₁+eps)²`。
- 全部放大率经 vendored 冻结 `adamw_tangent_step` 计算，与上述闭式逐块相等（单测 `tests/test_f1_mechanism_v01.py` 5/5 PASS，rel<1e-13）；c=0 锚点 forward 反事实数学与真实 `torch.optim.AdamW` 零梯度步位级一致（单测）。
- **勘误（对 F0 文档）**：F0 报告随记 "m 通道增益 = β1·G" 把动量递推与状态通道导数混写；正确的 m-状态通道为 `M=(β1/(1−β1))·G`（×9）。F0 的全部测量只走梯度通道，**不受影响**。

## 2. F1-A：blockwise state-channel amplification（`state_channel_amplification.csv`，189 行）

fp64、2 个随机种子平均（种子间差 <1%）：

| 规模/步 | θ 通道 A | m 通道 A | **v 通道 A** |
|---|---:|---:|---:|
| 5M step46（early） | 1.000 | 9.41 | **3.10e6** |
| 5M step82 | 1.000 | 3.92 | 5.43e5 |
| 26M step26（early） | 1.000 | 41.1 | **1.37e8** |
| 26M step102 | 1.000 | 5.41 | 1.10e6 |
| 100M step26（early） | 1.000 | 142.8 | **2.45e9** |
| 100M step64 | 1.000 | 68.4 | 3.23e8 |
| 100M step102 | 1.000 | 20.8 | 4.26e7 |

- **θ 通道严格良性（A≡1.0，即 (1−lr·wd)）**；m 通道按规模恶化（9.4→41→143）但量级温和；**v 通道比 m 通道大 5–7 个数量级，是 full-state 病态的几乎全部**。
- v 通道规模趋势（early）：3.1e6 → 1.4e8 → 2.4e9（约 ×44、×18）；训练自愈（100M：2.4e9→3.2e8→4.3e7，约 ×57）——与 F0 梯度通道 σ_max 的趋势/自愈同构，但幅度大得多。
- 结构化方向安全：m̂-对齐的 m 通道扰动 A_θ = 0.40–3.50（与 F0 "梯度能量方向良性" 一致）；病态由**远离梯度能量的随机 v 方向**触发。
- 解析核对：`Dv ∝ m̂·v̂^{-3/2}`——低 v 且 m̂≠0 的坐标是放大源；A_v 的大数不是奇异值意义上的 σ_max（那由梯度通道 G 的最大值 2198 代表 F0 口径），而是**分布式的大质量**（见 F1-C：f90≈13–35%）。

## 3. F1-B：v-floor 因果干预（`vfloor_intervention.csv`，135 行）

反事实 `v_τ = max(v, τ²)`（frozen-state probe，不进任何真实训练），方向在原始 v 上冻结：

100M seed2042（step26 σ_max=2258 基线；step64=295；step102=66）：

| τ | σ_max(τ) | σ_max 降幅 | **forward 单步更新相对变化** | 触碰坐标数 |
|---|---:|---:|---:|---:|
| eps(1e-8) | 679 | −70% | **0.05%** | 396 |
| 10eps | 77.8 | −96.5% | **3.6%** | 5.5e5 |
| 100eps | 7.89 | −99.7% | 31.6% | 2.4e7 |
| 1000eps | 0.79 | −99.97% | 99.5% | 9.2e7 |

- v 通道放大 A_v(τ)：1.78e9 → 1.63e9 → 3.19e8 → 1.59e7 → 1.31e5（单调）；m/随机方向放大同向改善（见 CSV）。
- **单调改善在每个规模、每个 step、每个种子上成立（27/27 组）**。
- 因果与混淆的切分：τ≤10eps 时 forward 改变 ≤3.6%（eps 时仅 0.05%）而 conditioning 已修复 70–96.5%——**sensitivity 效应不能被 forward 大改解释**；τ≥100eps 后 forward 改变主导，不再计入证据（预注册要求的双记录正是为此）。
- 机制细节：τ=eps 只改 396 个坐标却砍 70% σ_max——极端增益住在极少数极端 floor 坐标；τ 增大后进入分布式 floor 质量（与 F1-C 一致）。

## 4. F1-C：risk-coordinate knockout（`risk_knockout.csv` 324 行 + `risk_knockout_top64_meta.csv` 1152 行）

两个解析 risk（非经验定义）：`R_den = 1/(√v̂₁+eps)`；`R_der = |∂θ′/∂v|`（AdamW 对 v 的解析导数）。仅在 tangent diagnostic 中置 G=0：

100M seed2042 step26（σ_max=2258，A_random=14.77）：

| risk | mask | σ_max↓ | 掩埋坐标对 A² 贡献 |
|---|---|---:|---:|
| R_den | top 0.01% | **−85.9%**（→319） | 8.8% |
| R_den | top 0.1% | −93.1% | 26.4% |
| R_den | top 1% | −97.4% | 58.4% |
| R_den | top 5% | −99.0% | 80.7% |
| R_der | top 0.01–5% | −37.8%（平顶） | 6.8→74.8% |

- **σ_max 由极小子集控制**：R_den top 0.01%（约 1e4 坐标）即减 86%；top-64 元数据 63/64 落在 attn_q，v∈[3.5e-18, 4.4e-17]、|m| 极小（≤2e-8）——"双衰竭"坐标。
- **但随机方向放大是分布式的**：90% 的 A² 质量需要 f90 = 12.8%（100M step26）~35%（step102；5M ~35–49%）的坐标。即：存在小而尖的 σ_max 核 + 宽厚的分布式 floor 质量，二者同源（v-floor）。
- R_der 掩埋对 σ_max 平顶于 −37.8%：σ_max 坐标 |m̂| 小（R_der 低）——两个 risk 捕捉机制的不同剖面。

## 5. F1-D：module localization（`module_localization.csv` 27 行 + `_stability.csv` 36 行）

ΣG²（梯度通道能量，与病态同源）份额：

- **early（step26）bottleneck = attn_q**：100M 三种子 53–72%（attn_k 次之 12–15%）；26M 2/3 种子 attn_q 27–63%。
- 训练漂移：step≥64 后 mlp_gate/up/down 接管（26M/100M），5M 后期 embed_tied 最高（38–40%）。
- 稳定性：26M↔100M 匹配（seed,step）Kendall τ 均值 0.83（n=9）；同规模跨 phase τ 均值 0.87（n=27）——**模块排序稳定，相位依赖明确**。
- 结论：存在稳定但有相位结构的 bottleneck（early=attention-Q，mid/late=MLP，5M late=embedding），与 F0 §3 一致并细化。

## 6. F1-E：precision decomposition（`precision_decomposition.csv`，405 行）

100M 三个代表态，primal map 不变、只换 sensitivity 算术精度：

| 精度 | A_θ | A_m | A_v |
|---|---:|---:|---:|
| fp64 | 1.000000 | 132.89 | 1.7834e9 |
| fp32 | 1.000000 | 132.89 | 1.7834e9 |
| bf16 | 1.000000 | 132.89 | 1.7839e9 |

- fp32 与 fp64 一致到 ≥6 位，bf16 到 4 位、全部有限——**F0/F1 放大是数学 conditioning，不是 precision 产物**。
- FD 验证（θ/m 加性、v 乘性-log；冻结网格）：fp64 全通道通过——θ/m 通道 rel≈1e-16（本质线性），v-log 通道 rel≈3–5e-12、cos=1.0；fp32 下 θ/m≈3e-8 通过，**v 通道 fp32 FD 失败（0.15–0.5，减法消解地板）**——即 v 通道的病态大到 fp32 连其有限差分验证都无法承载（解析 JVP 本身 fp32 可算）。这是 F0 "fp32 分辨地板" 发现在 full-state 视角下的强化。
- H=1（预注册声明：本轮不重建 future batch 流，H>1 属 F2 的 rollout 范围）。

## 7. 五个问题式的回答

1. **放大来自哪个通道？** v（second moment）通道，超 m 通道 5–7 个数量级；θ 通道严格良性。
2. **v-floor 抬升是否单调改善 conditioning？** 是（27/27 组单调），且 τ≤10eps 时修复几乎不触碰 forward（0.05–3.6%）——因果方向成立。
3. **小 subset 是否控制病态？** σ_max：是（0.01% 坐标、98% attn_q → −86%）；A_random：否（f90≈13–35% 分布式）。
4. **稳定 bottleneck 模块？** early=attn_q（100M 全种子 53–72%），phase 漂移到 MLP/embed；排序跨规模/种子稳定（τ≈0.83–0.87）。
5. **precision 主导吗？** 否；但 fp32 已无法 FD-验证 v 通道（病态的实用推论）。

## 8. 与 F0/G1c 的衔接

- F0 的梯度通道 σ_max（39→290→2198）是同一 floor 机制在 `∂θ′/∂c` 块上的投影；本轮把机制定位到 `∂θ′/∂v` 块（大 6 个数量级）并给出因果干预证据。
- 旧 G1c 1B（seed evidence）：wte 精确零 v + fp16 HVP 溢出与本机制完全同构（v-floor → 极端对角增益 → 多步复合放大）；其 SCALE_TREND+PRECISION_ARTIFACT 归因与本轮 V_FLOOR_CAUSAL 相容（1B 侧不作为正式证据，未复测）。

## 9. 红线遵守与限定

- 零新训练、零 1B 下载（1B 计划取消前 0 字节）、零 legacy 修改；forward AdamW 未动；规则先于结果冻结；未为结论调任何阈值。
- 限定：c=0 锚点未含 clip 径向项与真实下一步梯度交叉项（与 F0 相同的 NOT_ESTIMATED，属 F2 rollout 范围）；frozen-state 干预不是训练算法；per-coordinate gradient 未在 bank 中存储（top-64 元数据以 m̂ 记录并如实标注 NOT_STORED）。

## 10. 制品

| 文件 | 内容 |
|---|---|
| `outputs/f1_mechanism/state_channel_amplification.csv` | 189 行 = 21 态 × {θ,m,v}×2 seed + m̂-对齐 |
| `outputs/f1_mechanism/vfloor_intervention.csv` | 135 行 = 9 态 × 5 τ × 3 通道方向（含 forward 差） |
| `outputs/f1_mechanism/risk_knockout.csv` | 324 行（knockout + kill-curve + f90） |
| `outputs/f1_mechanism/risk_knockout_top64_meta.csv` | 1152 行 top-64 坐标元数据 |
| `outputs/f1_mechanism/module_localization.csv` / `_stability.csv` | 27 + 36 行 |
| `outputs/f1_mechanism/precision_decomposition.csv` | 405 行（精度臂 + 三通道 FD） |
| `outputs/f1_mechanism/summary.json` | 聚合 + verdict |

## 11. F2 gate

F1 机制明确（V_FLOOR_CAUSAL，因果干预 + 分布结构 + 模块定位 + precision 排除四线合一）→ **允许进入 F2 hypergradient pilot**（按 `docs/F2_HYPERGRADIENT_PILOT_PREREG.md` 预注册后执行）。
