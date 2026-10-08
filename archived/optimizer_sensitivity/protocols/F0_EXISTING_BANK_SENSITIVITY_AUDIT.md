# F0 — Existing Bank Sensitivity Audit（既有冻结 bank 的 AdamW 切线条件数审计）

- 日期：2026-09-26。仓库：`optimizer_sensitivity/`（新项目第一轮，F0 只读离线）。
- 口径冻结：`configs/f0_bank_audit_v01.json`（频率分箱规则、参数组划分、方向类、精度变体、FD ε 网格均先于结果冻结）。
- 输入（全部 READ-ONLY）：v07 bank（4.98M，8 seeds × 7 steps = 56 态）、G1a bank（26.68M，3×3=9 态）、G1b bank（100.78M，3×3=9 态）；tokenizer = `minimind_base/model`（vocab 6400）；语料 D0 = e3c_subset.jsonl（4096 行，SHA256 `1e657af9…b0f4`）。
- 分析对象：**一步 AdamW map 在 c=0 锚点（下一步零梯度）的梯度→θ′ 通道 Jacobian J₀**。该锚点下 per-coordinate Jacobian 恰为对角 `G_i = lr(1−β1)/((1−β1^(t+1))·(√(v̂₁_i)+eps))`（v̂₁ = β2·v/(1−β2^(t+1))），且在任何 c_i=0 坐标上精确成立、与未知真实 batch 无关；clip 不激活（g=0 ⇒ 系数 1）；θ 通道因子 (1−lr·wd)（良性）；m 通道增益 = β1·G_i。数学实现 = vendored 冻结 `adamw_tangent_step`（provenance 见 docs/LEGACY_PROVENANCE.md）。
- 本轮 NOT_ESTIMATED（明确不做、不近似）：clip 径向耦合（需真实下一步梯度方向）、c≠0 的 m̂·dsqrt 交叉项逐坐标值、HVP/θ→梯度反馈通道。G1a/G1b 的逐步 preclip 范数（training_dynamics.json）作为记录上下文使用（clip 系数 ≈0.7–1.0）。

## 0. 总判定（F0_VERDICT）

**TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE（受下列限定）**：在 primal 训练完全正常的冻结轨迹上，一步 AdamW 切线 map 的条件数随规模显著恶化、随训练步数显著自愈；26M/100M 上随机方向的单位梯度扰动经一步即被放大（A>1）。机制（分母下限）已由 FD 验证到 1e-11。G1c 1B 异常归因：**SCALE_TREND（主）+ PRECISION_ARTIFACT（近因，多步复合后溢出）**，IMPLEMENTATION_ARTIFACT 被排除；MiniMind 配置中不存在 G1c 式精确零 v 坐标（tied head 机制差异，见 §3）。

## 1. 五个机器可读制品

| 文件 | 内容 |
|---|---|
| `outputs/f0_existing_bank_audit/summary.json` | 聚合 + verdict 块 + 74 态清单（含 G1a/G1b 逐步 preclip） |
| `outputs/f0_existing_bank_audit/per_parameter_group.csv` | 740 行 = 74 态 × 10 参数组：v/m 分位数、exact-zero、√v<eps 阈值分数、对角增益分位 |
| `outputs/f0_existing_bank_audit/embedding_frequency_bins.csv` | 296 行 = 74 态 × 4 频率箱：freq/分母/有效 Adam scale 统计 |
| `outputs/f0_existing_bank_audit/sensitivity_conditioning.csv` | 189 行 = 9 态 × 3 精度 × 7 方向类：A(q) 总量与分组分量 |
| `outputs/f0_existing_bank_audit/fd_jvp_comparison.csv` | 336 行 FD-vs-解析（2 锚点 × 4 切片 × ε 网格 × fp64/fp32）+ `nan_inf_probes.json`（72 条 bf16 对抗探针） |
| `outputs/f0_existing_bank_audit/power_iteration.csv` | 9 行 σ_max(J₀) + 主导坐标定位 |

## 2. 问题 1：second-moment 下限是否随规模变化（YES，剧烈，且按模块分层）

v（exp_avg_sq）下限（各 seed/step 最坏值）与 eps-floor 坐标占比：

| 组 | 指标 | 5M | 26M | 100M |
|---|---|---:|---:|---:|
| attn_q | v_min | 2.1e-14 | 6.5e-17 | **1.9e-18** |
| attn_q | v_p0.001 | 2.3e-13 | 2.9e-15 | **1.5e-16** |
| attn_q | √v<10eps 坐标占比 | 0 | 7.1e-3 | **0.125** |
| attn_k | √v<10eps 占比 | 0 | 1.3e-3 | **0.058** |
| mlp_gate | v_min | 7.0e-14 | 5.9e-16 | 4.1e-17 |
| embed_tied | v_min | 3.3e-13 | 7.8e-14 | 1.6e-14 |
| attn_q | 对角增益 max | 54.7 | 815 | **2623** |
| mlp_gate | 对角增益 max | 30.1 | 305 | 988 |
| embed_tied | 对角增益 max | 13.8 | 28.1 | 62.0 |

- 注意力/MLP 的 v 下限随规模下降 2–3 个数量级；**100M 的 attn_q 有 12.5% 坐标 √v̂ < 10·eps**（即有效分母进入 Adam eps 主导区）。
- **全库 74 态 × 740 组：exact-zero v 坐标数 = 0**（与 G1c/OLMo 的"精确零 exp_avg_sq"结构性不同，见 §3）。
- 训练自愈：attn_q 增益 max 随步数 26→64→102 衰减（100M：2258→295→63，约 36×；26M：328→35→7.7）。病态集中在训练早期。

## 3. 问题 2/3：哪些模块危险；token frequency 与 v-floor 的关系

- **最危险模块：attn_q > attn_k > mlp_gate/up > …> embed（最轻）**。σ_max(J₀) 的主导坐标全部落在 attn_q / mlp_gate（power_iteration.csv），从不在 embedding。
- **embedding 频率箱（74 态最坏值）**：分母下限 ~1.1e-6–4.5e-6（√v̂ 地板 ≫ eps=1e-8）；有效 Adam scale max：5M ≈7–9，26M ≈9–16，100M ≈15–26。
- **关键机制发现（修正 G1c 预期）**：MiniMind 用 **tied lm_head**，输出层 softmax 梯度流到全部 6400 个词行——**never-token 行的 v 非零（~1e-9 行级，坐标至 1e-13）且不比 tail 更差**；频率效应体现在幅度而非地板存在性：tail 方向增益 ≈ head 的 6–10×（Phase 4 A_embed：如 100M step26 tail 3.50 vs head 0.52），但 embedding 远不是全局最危险模块。
- 对 G1c 的含义：OLMo 的"精确零 v 坐标"（untied/大 vocab/罕见 token）是**配置特有放大器**，在 MiniMind 系不存在；MiniMind 系的等价物是"注意力/MLP 早期下限坐标 + Adam eps 地板"。

## 4. 问题 4：局部放大与 v-floor 的关系（精确成立）

A(q)=‖J₀q‖/‖q‖（fp64 参考精度，9 态；随机方向对 seed 稳定，±0.5%）：

| 规模 | step | A_random | A_tail | A_head | A_grad-aligned(m̂) | A_mlp | **σ_max(J₀)** |
|---|---|---:|---:|---:|---:|---:|---:|
| 5M v07 | 46 | 1.05 | 1.21 | 0.117 | 0.092 | 1.04 | 39.4 |
| 26M | 26 | 4.43 | 2.12 | 0.339 | 0.237 | 3.17 | 290 |
| 26M | 64 | 1.65 | 1.66 | 0.104 | 0.121 | 1.72 | 50.5 |
| 26M | 102 | 0.564 | 0.659 | 0.022 | 0.048 | 0.603 | 10.6 |
| 100M | 26 | **14.8** | 3.50 | 0.524 | 0.383 | 7.63 | **2198** |
| 100M | 64 | 6.17 | 2.80 | 0.109 | 0.206 | 4.87 | 279 |
| 100M | 102 | 2.10 | 1.10 | 0.029 | 0.089 | 1.80 | 51.3 |

- 规模趋势（匹配归一化步 26/EARLY）：σ_max 39→290→2198（5M→100M ~56×）；A_random 1.05→4.43→**14.8**。
- **A_random>1（26M/100M 早中期）意味着单位梯度扰动经一步 AdamW 即被范数放大**——即使不含曲率反馈，H 步切线范数也会几何式增长；这是 G1c "超线性放大"在受控阶梯上的定量对应物。
- 方向结构：放大集中在**远离梯度能量的方向**（m̂ 对齐方向良性 0.09–0.38；tail≈10×head）。
- 精度：fp32 与 fp64 的 A 一致（有限差未损失结构）；bf16 单步切线数学全部 finite（189 行 0 非有限）——单步不溢出，G1c 的 NaN 需**多步复合**（见 §5）。

## 5. 问题 5：解析 JVP vs 有限差分（一致；A/B/C/D 四分判定成立）

- **（C 实现）**：fp64 手写 AdamW 前向数学 vs 真实 `torch.optim.AdamW`：位级一致（单测）；FD vs 冻结解析 JVP：全部 48 组合（3 规模 × 4 切片 × 2 锚点）best rel_err ≈ **1e-11–1e-12，cos=1.0**；ε 收敛教科书式（v07 embed_never c0_zero：rel 0.27@ε=1e-2 → 2e-11@ε=1e-8 → 1e-10 舍入地板@1e-10）。vendored 冻结切线 = 真实 AdamW 一步 map 的真 Jacobian，实现排除缺陷。
- **（B 精度）**：fp32 native AdamW best rel ~1e-5–1e-7（cos≥0.99999），ε=1e-6 即触减法消除地板（rel 1e-3）——fp32 可在 ε≥1e-5 分辨该 Jacobian，再往下不可。bf16 对抗探针 72 条（含 never/tail 行、ε_fd∈{1e-6,1e-4,1e-2}）：**0 条 NaN/Inf**，max|update|=0.107（100M mlp_gate）。
- **（A 数学）**：条件数是真实的数学性质（fp64 下成立），不是数值伪影。
- **（D G1c 特定性）**：单步有限 + 多步复合放大（§4 A>1；σ_max 千级）⇒ G1c 的 h≈7 处 fp16 溢出与"tangent 范数→3e11"可由复合机制解释；且 MiniMind 银行无精确零 v（§3），G1c 的精确零坐标属外部配置结构。

## 6. 问题 6：是否已出现"primal 稳定但 tangent 病态"的可信证据

**是（限定在既有 bank 覆盖的范围）**：
1. 三批 bank 的 primal 训练全部正常（legacy：loss 正常、replay 位级一致、G1a/G1b 判定 STRONG_REPLICATION）；
2. 同一批 checkpoint 上，一步切线 map 的 σ_max 与随机方向放大随规模单调恶化（39/290/2198；1.05/4.43/14.8），26M/100M 上 A_random>1；
3. 机制（v-floor → Adam 分母 → 增益）经 FD 1e-11 级验证；
4. 病态集中在早期步 + 注意力/MLP + 低梯度能量方向，且随训练自愈——这解释了为何 **primal（沿梯度方向，良性 A≈0.1–0.4）从不生病，而 tangent（张在全空间）生病**。

## 7. 问题 7：G1c 1B 异常归因

| 选项 | 判定 | 依据 |
|---|---|---|
| SCALE_TREND | **主因（支持）** | 受控阶梯 5M→26M→100M 上条件数单调恶化（本审计 §2/§4） |
| PRECISION_ARTIFACT | **近因（支持）** | 单步 bf16 全有限 ⇒ 溢出需多步复合；G1c 诊断 fp16 溢出为直接触发 |
| EXTERNAL_CHECKPOINT_ARTIFACT | 部分支持（作为放大器） | 精确零 v 坐标在 MiniMind 系不存在（tied head）；系 OLMo 配置（untied/大 vocab/罕见 token）特有结构，非"损坏" |
| IMPLEMENTATION_ARTIFACT | **排除** | FD 验证 1e-11；legacy G1c 四分类已排除 A；本审计复核 |
| UNRESOLVED | 残留 | 1B 本身的 v-floor 结构未直接测量（本轮不下载） |

## 8. 问题 8：是否值得重下 1B checkpoint

**建议：值得（RECOMMEND），但作为 F1 预注册轮并带强制防护**：
1. 先做与本 F0 相同的**离线 v-floor 普查**（零成本判别精确零坐标规模、分母地板、逐组增益）——若 OLMo 的 wte 精确零坐标成片存在，则其在数学上等价于把增益推到 lr(1−β1)/(bc1·eps)·clip_coef 的极端对角；
2. 切线递推全部 **fp64 累积**（或 fp32 + 逐坐标 eps-floor 守卫），逐 h 记录范数轨迹，预设溢出即停的停止条件（不是事后改阈值）；
3. 检验点：step-0 checkpoint（v≡0 全零极端情形）与三个晚 checkpoint 的 σ_max 是否延续 39→290→2198→? 的趋势；
4. 判定预注册：区分"MiniMind 型地板（无精确零，1e-16–1e-18）"与"OLMo 型精确零"两种机制在 1B 切线递推中的不同爆发形态。
风险提示：单 checkpoint ~17.8G 下载（恢复锚点已存 legacy `docs/CLEANUP_G1C_LARGE_FILES_20260925.md`）；磁盘余量 247G 足够。

## 9. 限定与红线遵守

- 本轮零新训练、零 1B 下载、零 legacy 修改（legacy 全程只读，Git HEAD `6ac4942` 未动）；
- forward AdamW 规则未改（FD 用真实 `torch.optim.AdamW` 对照）；
- 未为结论调整任何阈值；c=0 锚点、方向类、ε 网格、频率箱规则均先于结果冻结于 config；
- 覆盖限定：Phase 4 仅 seed 2042（26M/100M）；c=0 锚点未含 clip 径向项与 c≠0 交叉项（NOT_ESTIMATED，需真实下一步梯度/batch replay——属 F1 候选工作）；单步结论外推到 H 步递推时必须计入复合效应与 clip 系数（26M/100M clip 近恒激活，系数 ≈0.7–1.0 会等比缩小 A 但不改变结构）；
- 频率计数用文本 token（不含 bos/eos；二者每样本出现、属 head 箱，不影响分箱结论）。
