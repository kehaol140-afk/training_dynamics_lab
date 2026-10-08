# ADAMW 当前研究报告（第二轮审计 / v3 任务包）

- 报告日期：2026-10-08（UTC 05:12 起；服务器本地 CST UTC+0800）
- 审计目录：`/home/b205/projects/training_dynamics_lab/audits/ADAMW_RESEARCH_REVIEW_UPDATED_20261008_051201UTC`
- 研究根：`/home/b205/projects/training_dynamics_lab`
- 范围：`optimizer_sensitivity/`、`continuation_adaptation/`；`legacy_fluctuation_response/` 仅作历史参照与溯源
- 历史对照：`reference/ADAMW_RESEARCH_REVIEW_BUNDLE_previous.zip`（旧审计窗口 2026-09-30T06:11–06:54Z），仅作对照，不继承其判定
- 性质：纯只读审计。未运行训练/GPU 评估/HVP/测试/新实验；未修改代码、配置、原始结果与旧报告；未解封受限评估数据；轻量 CPU 核算仅限已获准读取的既有结果
- 执行模式：按用户执行模式指令全程非交互批处理，连续完成清点→阅读→证据复核→报告→差异分析→自检→打包

---

## 1. 执行摘要

1. **continuation_adaptation 完全冻结**：自旧审计以来零写入（哈希与时间戳双重确认）。P4 终局判定 **STOP** 原样成立：C1 +1.587064 nats（8/8 seeds 劣于 A_best，Holm p=1.0），C2 +0.019356；B/D 差异仅描述性。
2. **optimizer_sensitivity 的 F0–F3.5 冻结**：11 项关键制品与文档的 sha256 与旧包 manifest 逐一相等（字节级不变），六个研究单元结论原样继承。
3. **F4 链（尺度×阶段×种子图谱 → 形式建模 → 稳定性感知展开 → 事后剖析 → 干预尺度挖掘）在旧窗口之后全部完结**，构成本轮最大增量。核心新结论：
   - 切向失稳呈**相结构**（相位自愈：严格持续愈合 13/14；LATE 相 24/24 观测曲率中性 R_curv≈1），但**无尺度单调趋势**、种子异质性高；
   - 匹配 5M 覆盖后，EARLY 相 λ_SO 随 N 单调（+1.0 单调性得分），R_curv 非单调——尺度结构集中在早期恢复速率而非晚期放大；
   - 宏观演化可用 **M0 progress-only 模型**预测（LOSO MAE 0.4249，pooled-OOF 0.4249；种子方差占 41.79%）；微观逐格创伤不可预测（prescreen balacc 0.539，按 prereg 自动禁入）——"宏观可预测、微观不可预测"；
   - **SAU 无方法收益（NO_METHOD_GAIN）**：SAU 退化为 FIXED H=1；实用收益已在 H≤2 内兑现（中位 H_suff=1，92.9%/90.5% 单元 H_suff≤2）；**可观测性缺口成立**（离线 AUROC 0.8367 vs 在线最优 0.685）；
   - F2.5 信任域表面 H_best≡1；F4C 尺度弱依赖且符号随 δ 翻转；STAGE D 未触发；
   - 事后决策备忘定调：**机制研究（方向 A）优先**，观测受限（B）为次要叙事，有限干预时标（C）不支持方法化，D 方法不支持，E 不必要停止；外部验证首选"第二配方/数据集 50M–150M"。
4. **从建模到优化收益的缺口仍然开着**：F4 链证明了"何时失稳、何时自愈、宏观可预测"，但没有任何一个 F4 产物产生可直接部署的优化器改进；SAU（本轮唯一的方法化尝试）被自身的可观测性剖析否决。这是当前从建模走向收益的核心缺口。
5. 登记 **4 项文档级瑕疵**（median 惯例声明缺失、行数笔误、erratum 正文与 canonical 数字不一致、消融阈值表述过紧），均不改变任何判定（§9）。

---

## 2. 为什么建立这套模型（研究动机）

AdamW 是当代 LM 训练默认优化器，但其大规模训练轨迹中反复出现**不可归因的 loss 毛刺/失稳**。既有诊断（loss 曲线事后归因）无法区分三类机制：(i) 优化器自身的数值病态；(ii) 数据/曲率冲击与优化器的耦合；(iii) 信任域/二阶信息失效。三条研究线的共同目标是**用可控的切向动力学（tangent dynamics）把"失稳"变成可测量、可分解、可外推的对象**，并回答：失稳是否在更大尺度下恶化？能否提前预测？能否通过自适应展开（SAU）等干预在触发时消解？

`continuation_adaptation` 是并行线：测试"冻结主干 + 继续预训练"类适配协议是否优于 from-scratch 重训及最优 checkpoint 续训，为算力决策提供依据。

`legacy_fluctuation_response` 为前身工程（FR/GTTR 链），提供银行化 checkpoint 与早期机制证据，本轮仅溯源。

## 3. 核心数学框架（tangent dynamics）

状态增广 z=(θ, m, v)。对 AdamW 更新写一步映射 Φ_h，沿主轨迹线性化得切向递归

  dz_{h+1} = J_h(α)·dz_h + s_h

其中 J_h(α) 为一步 Jacobian 对切向扰动的作用，α 为曲率反馈开关：

- **B 臂（α=0）**：仅优化器动力（m,v 通道），无曲率反馈；
- **C 臂（α=1）**：经 clip_jvp∘HVP 注入曲率反馈（二阶项进入切向演化）；
- **SO（source-off）脉冲**：在 h=1 注入一次性切向脉冲后关闭源项，测自由演化。

核心度量：
- G_B、G_C：B/C 臂切向增益（||dz_H||/||dz_1||，H=32 为主视界）；
- R_curv = G_C/G_B：曲率反馈贡献比（>1 曲率放大，≈1 曲率中性）；
- G_SO、λ_SO = ln(G_SO)/H：源关闭增长率，刻画恢复/放大时标；
- S_η、h_η：η 扫描的表面/临界步长指标。

**假设**（各单元 prereg 显式声明）：主轨迹局部线性化有效（切向幅度小）；HVP 经 clip_jvp 截断可控；银行态（bank states）取自真实训练检查点、覆盖早/中/晚相与多种子；网格（η、δ、H）先验固定，避免事后择优。

## 4. F0–F3.5 冻结结论（字节级不变，原样继承）

| 单元 | 结论关键词 | 状态 |
|---|---|---|
| F0 银行敏感度审计 | TANGENT_ILL_CONDITIONING_AT_SCALE：切向增益随尺度恶化，病态是真实且可测的 | 冻结成立 |
| F1 机制闭合 | V_FLOOR_CAUSAL：v-floor→θ′ 通道闭合，影响 ∝ v̂^(−3/2)，机制层面归因成立 | 冻结成立 |
| F2 实用超梯度 | PRACTICAL_HYPERGRADIENT_FAILURE：工程可行的超梯度估计噪声吞没信号 | 冻结成立 |
| F2.5 信任域 | TRUST_REGION_COLLAPSE_CORRELATED_ONLY：信任域失效仅见于相关谱设置，非普适 | 冻结成立 |
| F3 曲率机制 | SPECTRAL_INSTABILITY（ρ>1.05 判据）+ METRIC_DEPENDENT（F3.5）：不稳定模耦合存在但度量依赖 | 冻结成立 |
| F3.5 模耦合 | 不稳定模耦合报告定稿 | 冻结成立 |

以上 11 项制品/文档 sha256 与旧 manifest 全部一致（见 `analysis/RECOMPUTE_NOTES.md` §二），本轮零新增证据、零改动。

## 5. F4 链：本轮主增量

### 5.1 F4-A 尺度×阶段×种子图谱（74 银行态：5M/26M/100M × EARLY/MID/LATE × 多种子 × 6 步）

- **检验内容**：R_curv、λ_SO 随 (N, 相位, 种子, 步位) 的结构；愈合判定（严格持续愈合）。
- **数值**：phase_map_long.csv 444 行；严格持续愈合 **13/14**（erratum 修正，非 14/14；唯一未愈合为 seed 2032，覆盖窗口内未愈）；LATE 相 24/24 观测 R_curv≈1（曲率中性）；5M 覆盖仅限 MID（bank 天然限制，erratum 已声明）。
- **判定**：**PHASE_STRUCTURE_ONLY_NO_SCALE_TREND**（相结构真实、无尺度单调趋势）+ **HIGH_SEED_HETEROGENEITY**。

### 5.2 F4A.1 匹配 5M 覆盖（16 新态，重建 legacy 配方；金丝雀 seed 2030）

- **动机**：消除"5M 只测 MID"对尺度趋势判定的混淆。
- **验证门**：金丝雀五项 prereg 判据全过（stream_id 4090=4074+16；分布等价 PASS；轨迹逐点一致，loss h=1 = 7.443730）。
- **数值**：EARLY 相 λ_SO 对 N 单调（单调性得分 +1.0），R_curv 非单调；LATE 全部 24 观测曲率中性。
- **判定**：**NONMONOTONIC_SCALE_STRUCTURE**——尺度信息集中在早期恢复速率，不存在"越大越不稳定"的简单律。

### 5.3 F4A.5 形式建模（90 态数据集，LOSO 交叉验证，prereg 门禁）

- **候选模型**：M0 progress-only / M1 scale-only / M2 progress+scale / M3 progress+phase / M4 全量 / M5 交互。
- **数值**：M0 最优 LOSO MAE **0.4249**（pooled-OOF 同值），fold 中位 MAE **0.4041**（canonical recompute.json；erratum 正文 0.4219 为笔误）；种子方差占比 **41.79%**；prescreen 门 balacc 0.539 < 阈值 → **MODEL_PRESCREEN DISABLED_BY_PREREG**（按 prereg 不进入选择）。
- **判定**：宏观 progress 项承载主要可预测结构；尺度/相位交互增益不抵方差；微观逐格创伤不可预测。

### 5.4 F4B 稳定性感知展开 SAU（方法化尝试）

- **设计**：在线按 λ̂(H) 选择展开视界 H∈{1,2,8,16,32,64}，动作网格含提前停止/缩 η；δ=.05 主网格 + δ=.02 敏感性。
- **数值**：SAU 选中 H 恒 1，与 FIXED H=1 **逐位相等**（λ̂(2)≈0.35–0.38 > 网格上限 B_λ=0.30 所致）；oracle 逐格最优 H 中位亦为 1；dominant-subset BEST_FIXED 中位 regret 0.3165 vs SAU 0；δ=.02 时 FIXED32 均值 0.1317 最优、SAU 0.1810；H2−H1 之差随 δ 变号（+.0512/−.0746）。
- **判定**：**NO_METHOD_GAIN**；F2.5 表面 H_best≡1 再次确认短视界。

### 5.5 F4B 事后剖析（为何无收益）

- **可观测性**：离线诊断 AUROC **0.8367**（δ=.05）/ 0.7904（δ=.02, H_sep=1）显著高于在线可实现 AUROC **0.685**/0.6728 → **OBSERVABILITY_GAP_SUPPORTED**。
- **效用结构**：中位 H_suff=1；92.9%（δ=.05）/90.5%（δ=.02）单元 H_suff≤2 → **UTILITY_ALREADY_SHORT_HORIZON**（收益已在最小视界内兑现，自适应没有操作空间）。

### 5.6 F4C 干预尺度挖掘 + 一致性审计 + 决策备忘

- F4C：WEAK_SCALE_DEPENDENCE；H2−H1 符号随 δ 翻转；STAGE D 按 prereg 未触发（DISABLED）。
- 一致性审计：headline 14 项（12 VERIFIED/3 STALE_REPORT/1 OVERWRITTEN_ARTIFACT）全部登账闭合。
- 决策备忘（F4_POSTMORTEM_DECISION_MEMO）：**方向 A（机制）> B（观测受限叙事）> C（有限干预时标）**；D（方法化）不支持；E（停止）不必要。外部验证首选选项 2：第二配方/数据集 50M–150M（优于 300M 复刻）。

## 6. continuation_adaptation（P4，冻结终局）

- 设计：56 cells × 8 arms = 448 runs；stage2 封存 280 行评估表；stage3 Holm + 四重门。
- 头部数值（本轮从封存表独立重算一致）：
  - C1 − A_best = **+1.587064** nats，CI [1.2283, 1.9644]，Holm p=1.0，0/8 seeds 胜出；
  - C2 − A_best = **+0.019356**，CI [0.0140, 0.0247]，0/8 seeds 胜出；
  - 描述性：B − A_best −0.013881；D − A_best +0.060853；B − D −0.074735。
- 判定：**STOP**（verdict 字段；gates.continue_direction=false，四重门全 fail）。本轮无任何新证据，判定冻结。

## 7. 当前有效结论总表（截至 2026-10-08）

| # | 结论 | 状态 | 关键证据 |
|---|---|---|---|
| 1 | 切向病态随尺度恶化（F0） | 成立（冻结） | f0 power_iteration.csv |
| 2 | v-floor 因果闭合（F1） | 成立（冻结） | f1 summary.json |
| 3 | 实用超梯度失败（F2） | 成立（冻结） | f2 summary.json |
| 4 | 信任域坍塌仅限相关谱（F2.5） | 成立（冻结） | f2_5 summary.json |
| 5 | 谱不稳定存在但度量依赖（F3/F3.5） | 成立（冻结） | f3/f3_5 summary |
| 6 | 失稳呈相结构、13/14 严格自愈、LATE 曲率中性 | 成立（新增） | f4a_phase_map/* |
| 7 | 无"越大越不稳定"单调律；EARLY λ_SO 单调于 N | 成立（新增，收紧旧猜测） | f4a1 matched_summary |
| 8 | 宏观可预测（M0, LOSO MAE 0.4249）/ 微观不可预测 | 成立（新增） | f4a5 model_comparison |
| 9 | SAU 无方法收益 + 可观测性缺口 | 成立（新增，否定方法化） | f4b_sau, f4b_postmortem |
| 10 | 收益已在 H≤2 短视界兑现 | 成立（新增） | horizon_sufficiency.csv |
| 11 | 干预尺度弱依赖、符号随 δ 翻转 | 成立（新增） | f4c mining_summary |
| 12 | P4 STOP（C1/C2 劣于基线） | 成立（冻结） | sealed table + stage3 |

**需收紧/撤回**：旧审计中"F4-A 将给出尺度律"的预期被撤回为 NONMONOTONIC_SCALE_STRUCTURE；"SAU 有望获得方法收益"被 F4B 结果否定；F3 系列的所有"普适性"表述此前已由 F2.5/F3.5 收紧，本轮维持。erratum 链（F4A 13/14、F4A5 数字）已按 canonical 口径修正。

## 8. 从建模到优化收益的缺口（现状评估）

1. **干预通道缺失**：F4 链刻画了失稳/自愈的规律，但本轮唯一方法化尝试（SAU）因效用已短视界兑现 + 在线可观测性不足而无收益。当前没有任何经审计验证可部署的优化器改进。
2. **观测差距是硬约束**：离线 0.8367 vs 在线 0.685 的 AUROC 差意味着"知道会发生什么"与"触发时能认出什么"之间有结构性落差；任何在线干预方法都必须先跨越该差距。
3. **外推未验证**：全部 F4 证据来自单配方银行态；决策备忘已选定外部验证（第二配方/数据集 50M–150M），尚未执行——这是把"规律"升级为"机制定律"前的必要一步（注：本轮不安排新实验，仅登记）。
4. **宏观/微观断裂**：M0 能预测宏观 progress 结构，但微观逐格创伤不可预测；若收益来自微观规避，则当前模型不足以支撑。
5. **数学缺口**：λ_SO 单调于 N 的机制解释（为何恢复速率随尺度改善）尚无闭合推导；R_curv 非单调亦无机制级说明。

## 9. 文档级瑕疵登记（不改判定）

1. F4-A 报告逐步 5M "median" 实为 median_high 惯例（n=8 偶数取上中位），未声明；
2. F4A.1 报告称 matched_phase_map.csv 为 90 行，实为 42 行（90 行并集在 f4a5 dataset.csv）；
3. F4A5_METRIC_ERRATUM.md 正文 MEDIAN_FOLD_MAE 0.4219 与同文档 canonical recompute.json 的 0.4041 矛盾（本轮独立复算确认 0.4041）；
4. F4A.5 报告"predictor ablation ΔMAE ≤ 0.01"与 CSV 矛盾（max |Δ|=0.0368），数值仍小不改变 M0 选择。

## 10. 证据与复核索引

- 复核脚本与输出：`analysis/verify_all.py`、`analysis/verify_out.txt`
- 复核记录：`analysis/RECOMPUTE_NOTES.md`
- 证据副本：`evidence/`（63 文件，804K；关键 CSV/JSON/文档按原相对路径）
- 旧包对照：`old_bundle_extract/`（基线报告、manifest、PROGRESS）
- 变更分析：`UPDATE_DIFF.md`
- 索引：`MODEL_INDEX.csv`、`EXPERIMENT_INDEX.csv`、`KEY_RESULTS.csv`、`CLAIM_LEDGER.csv`、`COVERAGE.csv`
