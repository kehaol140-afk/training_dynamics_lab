# F2.5 — Trust-Region Collapse 报告（100M，LR channel）

- 日期：2026-09-27。预注册：`docs/F2_5_TRUST_REGION_PREREG.md`（冻结于 2026-09-26，commit `e333ffa`，先于任何 F2.5 计算）。本报告全部数值由该预注册的冻结公式产出，未做任何阈值/网格/判据调整。
- 判决（预注册 §5 判定程序，按序首个命中）：**`TRUST_REGION_COLLAPSE_CORRELATED_ONLY`**
- cells：EARLY `2042:26`、EARLY `2043:26`、MIDDLE `2042:64`（G1b 100.78M bank 只读；CRN stream `derive_stream_seed(20260926,{1,2,3})`；manifest SHA 与 F2 相同）。

## 0. 执行摘要

1. **collapse 模式（预注册操作性定义）成立**：三格、全部 10 条臂，δ*(H=1)=1e-1（全网格 usable），而 δ*(H≥2)=0。按 §5 定义（EARLY cells δ*(16)=0）collapse 模式为真。
2. **无任何 re-expansion**：4 种 backward-only oracle 干预（mask R_den top 0.01%/0.1%、vfloor τ=eps/10eps）在所有 (cell, H) 上均未改变 δ*（保持 0），h_tangent 相对变化 ≤0.92%（多数 ≤0.1%）。因果支 prong 失败，故判决为 CORRELATED_ONLY 而非 CAUSALLY_LINKED。
3. **FD 侧无稳定 plateau**：除 H=1 外，全部 cell × H≥2 的 central FD 在 resolvable 范围内漂移 >20%（`FD_NONCONVERGENT`），EARLY H=16 的 D(δ) 跨 δ 摆动 76.4 → −23.6 → −15.4 → …（与 F2 的 EARLY 长程 FD 退化一致）。按冻结措辞：finite-intervention local regime 在 H≥2 处于 **unresolved / collapsed at accessible numerical resolution**；不得表述为"tangent 错误"。H=1 处 FD 与 tangent 紧密一致（R≈0.02–0.09）。
4. **干预无效的结构性原因（B/C 对照）**：tangent 放大几乎完全由网络曲率反馈（C 臂 HVP 路径）驱动，而非 AdamW 切线内部的 v→θ′ 通道。collapse  horizon 处 hC 比 hB 大 2–5 个数量级（例：2043:26 H=32，hC=+3205.4 vs hB=−0.021；2042:26 H=16，hC=−5.915 vs hB=−0.051）。干预只作用于优化器内部 v 通道，物理上无法触及曲率通道——这是对"F1 v-floor → F2 失败"因果链的直接检验，结果为**否定**。
5. **conditioning → tangent 放大：强支持**；**conditioning → δ*：无可用相关**（δ* 在 H≥2 全部被 FD 分辨率地板钉在 0，三格无差异，n=3 秩相关退化）。tangent 增长倍数（|dθ/du(H)|/|dθ/du(1)|）与 A_v 三格排序完全一致：2043:26（A_v=4.68e9，growth H32=2.05e5）> 2042:26（1.78e9，1.16e3）> 2042:64（1.27e8，29.0）。

## 1. 数据质量与交叉验证

| 项 | 结果 |
|---|---|
| manifest SHA | 2042:26=`e40dd1f4…`、2042:64=`602b403f…`、2043:26=`c3862f60…`，与 F2 冻结值一致 |
| forward 一致性 | 三格 `forward_bitwise_match=True`，`max_baseline_loss_abs_diff=0.0`（10 条切线臂共享单一 primal rollout，前向轨迹与独立 δ=0 baseline rollout 逐位一致） |
| tangent 复现 F2 | `none\|C` h_tangent 与 F2 fp64/C 参考值全部一致：2042:26 H1 −0.018488（ref −0.0185）、H16 −5.9148（ref −5.91）；2043:26 H1 −0.005548（ref −0.00555）、H16 +5.3328（ref +5.33）；2042:64 H1 −0.089262（ref −0.0893） |
| 冻结位级单测 | `test_no_intervention_bit_exact` 通过（红线：无干预 == vendored 冻结函数位级相等）；`vfloor_eps` 臂 h_tangent 与 none 臂在多数读出**逐位相同**，独立佐证 clamp 未触碰任何坐标 |
| 测试 | F2.5 库 5/5、F2.5 分析 6/6、legacy 套件全绿（f0 audit OK、f1 mechanism 5/5、f1 synthetic ALL PASS、f1 multistep smoke 4/4、f1 1b replication 6/6） |
| 臂健康 | 30 臂（3 cells × 10）全部 6 个读出点 finite、无 stop、无 NONFINITE/GUARD |

实现事故记录（不影响数据有效性）：首轮运行在 cell 1 切线 h=1 处因 `rden_topk_masks` 输出未 reshape 为参数形状而崩溃（FD 相位数值与重跑逐位一致）；修复后重跑全部三格，坐标选择逻辑未动（掩码阈值/k 值修复前后完全相同）。

## 2. δ*(H) 主表（连续前缀规则，冻结）

| cell | phase | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---|---|---|---|---|---|---|
| 2042:26 | EARLY | **0.1** | 0 | 0 | 0 | 0 | 0 |
| 2043:26 | EARLY | **0.1** | 0 | 0 | 0 | 0 | 0 |
| 2042:64 | MIDDLE | **0.1** | 0 | 0 | 0 | 0 | 0 |

（10 条臂的 δ* 完全相同；表为 none|C 臂。n_resolvable/n_usable 见 `trust_radius.csv`。）

**前缀规则失效机制**：δ*(H≥2)=0 并非"所有 δ 都不可用"——更大 δ 仍大量 usable（例：2042:26 H=2 在 δ≥3e-3 全部 8 点 R≤0.095；2042:64 H=16 有 6/8 usable）。钉死 δ* 的是**最小 resolvable δ**：

- 2042:26 H=2，δ=1e-4：R+=0.398、R−=1.571（|h·δ|≈2.5e-6，FD 噪声主导）→ 前缀断裂 → δ*=0。
- 2042:64 H=2，δ=1e-4：R+=0.019、R−=0.204（差 0.004 越线）→ 同样归零。
- 2042:26 H=16：全部 R ∈ [0.18, 40.2]（FD 非收敛，见 §3）。
- 2042:64 H=16，δ=1e-4：R−=1.011 → 前缀断裂。

**EARLY vs MIDDLE 差异**：在 δ* 层面**无差异**（三格同型）。差异体现在 (a) tangent 增长倍数（H32：EARLY 1.16e3/2.05e5 vs MIDDLE 29.0）；(b) EARLY H≥16 的 R 全网格 >0.2（FD 混沌），MIDDLE 仅最小 1–2 个 δ 越线；(c) n_usable 的 δ 分布。即：MIDDLE 的 FD 噪声地板更薄，但同样无法通过冻结前缀规则。

## 3. FD 收敛（冻结检查，report-only）

- **H=1：收敛**。三格 D(1e-4..1e-1) 稳定（2042:26 D≈−0.0172，跨 δ 变化 <1.3%；2043:26 −0.00477→−0.00499；2042:64 −0.0837）。h_FD 与 h_tangent 相对差 6–12%，R 全网格 ≤0.09 → δ*=0.1。
- **H≥2：全部 FD_NONCONVERGENT**（唯一例外 2042:64 H=2 收敛，但 δ* 仍因单点越线为 0）。EARLY H=16 的 D(δ) 跨 δ 摆动一个量级以上（76.35 → −23.64 → −15.39 → −9.68 → −1.76 → +0.59 → −1.05 → −0.09，2042:26；40.05 → 15.19 → −1.92 → 1.73 → 0.11 → 0.67 → 0.18 → 0.30，2043:26）。
- h_FD（最小 resolvable δ 的 D）与 h_tangent 在 H≥2 普遍符号相反或量级背离（例：2042:26 H=16，h_FD=+76.35 vs h_tan=−5.91；2043:26 H=32，h_FD=−11.91 vs h_tan=+3205；2042:64 H=32，h_FD=−2.07 vs h_tan=−0.83）。
- **无稳定 plateau**。按预注册冻结措辞：H≥2 的 finite-intervention local regime **unresolved / collapsed at accessible numerical resolution**。切线精度在 H≥2 无法由本 FD 数据裁决；H=1 处两者一致。

## 4. backward-only oracle 干预结果（机制探针，非方法）

h_tangent 变化（C 臂；B 臂对所有干预逐位不变，符合预期——B 臂无曲率反馈，优化器内部通道贡献可忽略）：

| cell | H | none | mask 0.01% | mask 0.1% | vfloor eps | vfloor 10eps | 相对变化上限 |
|---|---|---|---|---|---|---|---|
| 2042:26 | 16 | −5.9148 | −5.9204 | −5.9695 | −5.9148（位同） | −5.9102 | 0.92% |
| 2042:26 | 32 | −8.6055 | −8.6123 | −8.6761 | −8.6055（位同） | −8.6020 | 0.82% |
| 2043:26 | 16 | +5.3328 | +5.3340 | +5.3457 | +5.3328（位同） | +5.3121 | 0.39% |
| 2043:26 | 32 | +3205.41 | +3206.47 | +3215.65 | +3205.41（位同） | +3201.65 | 0.32% |
| 2042:64 | 32 | −0.83319 | −0.83321 | −0.83340 | −0.83319（位同） | −0.83319 | 0.025% |

（`vfloor_eps`（v←max(v,1e-16)）在这些 bank 上是 no-op：无坐标的原始 v<1e-16，与 none 逐位相等；`vfloor_10eps`（v←max(v,1e-14)）触碰 √v<1e-7 的坐标——占比 0.55%/1.21%/0.025%（floor_frac 列），与 F1 一致。）

**re-expansion 检验：失败**。所有干预在所有 (cell,H) 上 δ* 保持 0；re-expansion 要求 δ*_intervened ≥ 10×δ*_baseline，实际 0→0，无任何再扩张。干预方向甚至非单调（mask 使 |h| 略增，vfloor_10eps 使 |h| 略减）。

**物理解释（B/C 对照）**：collapse horizon 处 tangent 响应由曲率反馈通道主导（hC/hB：2042:26 H16 为 116×，2043:26 H32 为 1.5e5×，2042:64 H32 为 3.6×）。本预注册的干预只修改 AdamW 切线内部的 v→θ′ 敏感性（dv_mask）与 M/Dv 分母（v_back），结构上无法触及网络 HVP 通道。因此"优化器状态干预 → trust radius 再扩张"的因果检验在这些 horizon 上是**结构性不可行**的：瓶颈不在 v-floor 通道。F1 的 v-floor 因果性（forward σ_max −70%/−96.5%）在 forward 干预语义下成立，但其效应经 C 臂曲率放大后才进入 h_tangent；直接对切线内部 v 通道做 backward-only 干预只捕捉到该因果链的 <1% 残差。

## 5. conditioning vs trust radius

| cell | phase | A_v (F1E fp64\|v) | σ_max | attn_q share | floor_frac | growth(H32) | δ* 模式 |
|---|---|---|---|---|---|---|---|
| 2042:26 | EARLY | 1.78e9 | 2197.7 | 0.668 | 0.55% | 1.16e3 | 0.1,0,0,0,0,0 |
| 2043:26 | EARLY | 4.68e9 | −2623.3* | 0.716 | 1.21% | 2.05e5 | 0.1,0,0,0,0,0 |
| 2042:64 | MIDDLE | 1.27e8 | 279.1 | 0.457 | 0.025% | 29.0 | 0.1,0,0,0,0,0 |

\* 负号标记 provenance：seed 2043 在 F0 `power_iteration.csv` 无行，此处为 −diag_gain_max 代理值（|σ| 量级与 2042:26 相当）。

- **conditioning → tangent 放大**：三格排序与 A_v 完全一致（4.68e9 > 1.78e9 > 1.27e8 ↔ growth 2.05e5 > 1.16e3 > 29.0），与 F0/F1 的 conditioning 图景一致。
- **conditioning → δ***：无可检验相关。δ* 在 H≥2 被 FD 分辨率地板统一钉在 0，三格无变异（n=3 秩相关退化）。预注册的相关性 prong 在数据上没有可分辨的支持，也无可分辨的反对。

## 6. 判定程序执行（预注册 §5，按序）

1. `FD_RESOLUTION_LIMITED`：要求 ≥2 个 (EARLY × H≥16) 组合**整个网格无 resolvable δ**。实际 n_resolvable = 8/8/6/4（2042:26 H16/H32、2043:26 H16/H32）——**不命中**（注意：存在 resolvable 但 unusable 的 δ，与"无 resolvable"是不同情形）。
2. `TRUST_REGION_COLLAPSE_CAUSALLY_LINKED`：collapse 模式成立（EARLY δ*(16)=0 ✓），但 re-expansion 不存在（§4），且 MIDDLE 同样 collapse（δ*(16)=0，不满足"无 collapse 或弱得多"）——**不命中**。
3. `TRUST_REGION_COLLAPSE_CORRELATED_ONLY`：collapse 模式成立 且 无任何 re-expansion——**命中**。✅

## 7. 与 F0/F1/F2 结论链的关系

- F0（tangent 病态）与 F1（v-floor 因果）在本实验中作为**冻结输入**复现良好：A_v、σ_max、floor_frac、attn_q 定位全部一致。
- F2（PRACTICAL_HYPERGRADIENT_FAILURE）在更细 δ 网格上复核：H=1 全网格 usable（FD 与切线一致）；H≥2 的 FD 臂全部非收敛，EARLY 呈混沌摆动——支持 F2 的"FD 估计本身退化"表述，并明确其分辨率边界在 δ≥3e-4（H=2 处 3e-4 起 R≤0.36→0.07 恢复 usable）。
- **因果链 v-floor → conditioning → trust-region collapse：不支持（本实验范围内）**。conditioning → 切线放大（经由网络曲率通道）：支持。v-floor → δ*：被 backward-only 干预检验否定（效应 <1%，无再扩张）。δ* 的操作性 collapse 主要反映 **FD 分辨率极限**，在三格（含低 conditioning 的 MIDDLE）同型出现。

## 8. 局限

- n=3 cells，conditioning↔δ* 相关检验功效不足；且 δ* 地板化使相关检验退化。
- H≥2 的切线精度不可由本实验裁决（FD 臂非收敛）；"tangent 错"与"tangent 对但 FD 测不到"在 H≥2 观测等价，本报告不做超越冻结措辞的断言。
- vfloor_eps 在这些 bank 为 no-op（无 v<1e-16 坐标）；vfloor_10eps 仅覆盖 floor_frac 列出的 0.025%–1.21% 坐标。
- 干预的结构性不可行结论（§4）依赖 C/B 分解的解读；B 臂本身无曲率反馈，其绝对量级不代表 forward 干预效应（forward 干预会改变 primal 轨迹，超出本预注册范围）。

## 9. 产物清单

- 原始：`outputs/f2_5_trust_region/raw_{2042_26,2042_64,2043_26}.json`（fd/tangent 全行 + manifest + baseline_trace + forward 一致性）
- 分析：`outputs/f2_5_trust_region/{summary.json, trust_radius.csv(180), fd_convergence.csv(144), backward_intervention.csv(18), conditioning_vs_radius.csv(18)}`
- 运行日志：`outputs/f2_5_trust_region/f25_run.log`（含 FD 检查点与完成时间戳）
- 代码：`src/opsens/f25_trust_region_v01.py`、`scripts/f25_trust_region.py`、`scripts/f25_build_outputs.py`、`tests/test_f25_{trust_region,analysis}_v01.py`
- 审计：`docs/F2_5_RESUME_AUDIT.md`

## 10. 红线声明

未开发 F3 方法、未加 optimizer、未下 1B、未改 forward AdamW、未调任何冻结阈值/网格/判据；负结果原样保存；干预实现通过位级相等单测后才用于实验。本报告完成后停止，等待人工/ChatGPT 审阅。
