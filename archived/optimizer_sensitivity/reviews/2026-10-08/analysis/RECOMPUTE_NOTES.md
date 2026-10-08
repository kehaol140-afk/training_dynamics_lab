# RECOMPUTE_NOTES — 轻量 CPU 复核记录

- 审计目录：`/home/b205/projects/training_dynamics_lab/audits/ADAMW_RESEARCH_REVIEW_UPDATED_20261008_051201UTC`
- 复核窗口：2026-10-08 05:12 UTC 起（本地 CST 13:12）
- 复核脚本：`analysis/verify_all.py`（输出存于 `analysis/verify_out.txt`）
- 环境：`/home/b205/miniforge3/envs/minimind-exp/bin/python`（numpy 2.2.6 / scipy 1.15.3 / sklearn 1.5.1 / pandas 2.3.3）
- 性质：只读 + 轻量 CPU 核算。未运行任何训练、GPU 评估、HVP、测试或新实验；未解封受限评估数据。

## 一、续作项目（continuation_adaptation / P4）复核 — 全部 PASS

1. **封存表重算**：`outputs/p4_full/stage2/sealed_evaluation_table.csv`（280 数据行 + 表头 = 281 行）。
   - C1 − A_best：均值 **+1.587064** nats，seed 聚类 bootstrap CI **[1.2283, 1.9644]**（stage3 官方 [1.2352, 1.9550]，微小差异来自实现细节，结论一致：正向、远离 0）。
   - C2 − A_best：均值 **+0.019356**，CI **[0.0140, 0.0247]**（stage3 [0.0141, 0.0246]）。
   - 中位数：C1 1.2773，C2 0.020282（与旧审计一致）。
   - 8/8 seeds 两个变体均劣于 A_best（C1: 0/8 胜出，C2: 0/8 胜出），Holm 校正 p = 1.0。
   - 描述性次结果：B − A_best = **−0.013881**；D − A_best = **+0.060853**；B − D = **−0.074735**（CI [−0.0810, −0.0687]），全部描述性、不作显著性主张。
2. **哈希核验**：P4_PROTOCOL.md（1909df17…）、stage3/analysis_report.json（b9ab1fd5…）、stage2/sealed_evaluation_table.csv（f8e8ef05…）、SEAL_LOCK.json（8b269dd4…）与旧包 BUNDLE_MANIFEST.key_source_sha256_at_audit **完全一致**。
3. **时间戳核验**：continuation_adaptation 全部文件 mtime ≤ 2026-09-26/27，旧审计窗口（2026-09-30）之后零写入。
4. **结论**：P4 STOP 判定为**冻结终局**，本轮旧结论原样成立，无新证据。

## 二、optimizer_sensitivity F0–F3.5 复核 — 全部 PASS（字节级不变）

以下输出制品与文档的 sha256 与旧包 manifest 逐一相等（verify_out.txt 第 61–71 行）：

| 制品 | 哈希前缀 |
|---|---|
| f0_existing_bank_audit/power_iteration.csv | 90afb326… |
| f1_mechanism/summary.json | b3adb732… |
| f2_5_trust_region/summary.json | 083b312a… |
| f2_hypergradient/summary.json | 7069c2ae… |
| f3_5_mode_coupling/summary.json | a5589299… |
| f3_curvature_mechanism/summary_final.json | 9fb2112b… |
| docs/F0_EXISTING_BANK_SENSITIVITY_AUDIT.md | db4ffc68… |
| docs/F1_MECHANISM_CLOSURE.md | f9560373… |
| docs/F3_CURVATURE_MECHANISM_FINAL_REPORT.md | 21133711… |
| docs/F3_5_UNSTABLE_MODE_COUPLING_REPORT.md | a9c23440… |

结论：F0–F3.5 六个研究单元自旧审计以来**未被触碰**，其结论（F0 切向病态、F1 v-floor 因果闭合、F2 实用超梯度失败、F2.5 信任域坍塌、F3 谱不稳定、F3.5 不稳定模耦合、F3.5-metric 度量依赖）本轮原样继承。

## 三、F4 链复核 — 全部 PASS

### F4-A 尺度×阶段×种子图谱（74 银行态 × 6 步）
- `phase_map_long.csv` 444 行 = 74 态 × 6 步；`healing_intervals.csv` 14 行（13 INTERVAL_CENSORED + 1 NOT_HEALED_IN_COVERAGE = seed 2032）。
- 从 `raw/` 逐态 metrics.json 重算的 G_B_32/G_C_32/R_curv 与长表逐格一致（例：seed 2030 step46，G_C_32=116.013358、G_B_32=24.318463、R_curv=4.7706）。
- 报告逐步"中位数"表实为 **median_high 惯例**（偶数 n=8 时取上中位）：statistics.median_high 复现报告 5M 各步值（如 step46 median_high=28.6）；标准 median 略低（25.99）。数据与表格无矛盾，仅为惯例声明缺失。**登记为文档级瑕疵 #1**。
- 严格持续愈合：13/14（erratum 已改口，非 14/14）；5M 覆盖仅限 MID 相（erratum 已声明）。

### F4A.1 匹配 5M 覆盖（16 新态 + 金丝雀）
- 金丝雀 seed 2030 五项 prereg 判据全部通过：stream_id 4090 = 4074 + 16（新态并入不污染旧流）；分布等价（canary_envelope.json 与 raw 直读一致）；轨迹 loss h=1 7.443730 与 bank 原始记录一致。
- 关键数：canary R_curv_32 = 6.0305，λ_SO = 0.08872，与 matched_summary.csv 一致。

### F4A.5 形式建模（90 态数据集）
- `dataset.csv` 90 行（= F4-A 74 + F4A.1 16 的匹配子集并集）；六个候选模型 pooled-OOF 指标全部重算 == `model_comparison.csv`：
  - M0 progress-only：LOSO MAE **0.4249**（最优）；M2 +scale：0.4441；M3 progress+phase：0.4795；M1 scale-only 0.5088；M4 全量 0.4967；M5 交互 0.5011。
  - M0 LOSO fold MAE 中位 **0.4041** == 官方 recompute.json（erratum 文档正文写 0.4219，**登记为文档级瑕疵 #3**，以 canonical 0.4041 为准，本轮独立复算确认 0.4041）。
- 种子方差占比 0.4179 == `variance_decomposition.csv`。
- prescreen 门 balacc 0.539 → MODEL_PRESCREEN 按 prereg 自动 DISABLED_BY_PREREG（不进入模型选择）。
- **predictor_ablation.csv 最大 |ΔMAE| = 0.0368**，与报告"≤0.01"表述不符；相对 M3−M2 差距 0.0546 仍属小量，不改变 M0 选择。**登记为文档级瑕疵 #4**。

### F4B SAU（稳定性感知展开）
- `summary.json`（= delta_0p05/summary.json）：SAU 逐格选中 H 恒为 1， regrets 与 FIXED H=1 **逐位相等**（退化原因：λ̂(2) ≈ 0.35–0.38 恒超出动作网格上限 B_λ = 0.30）。
- oracle 逐格最优 H 的中位数亦为 1；dominant-subset 上 BEST_FIXED 中位 regret 0.3165 vs SAU 0。
- δ=0.02 摘要：FIXED H=32 均值 0.1317 最优，SAU 0.1810；H2−H1 之差随 δ 变号：δ=.05 时 +0.0512 [+0.000, +0.113]，δ=.02 时 −0.0746 [−0.158, +0.000]。
- F2.5 微表面：H_best ≡ 1（全格），再次支持短视界。

### F4B 事后剖析（可观测性）
- `utility_by_horizon.csv` 504 行 = 2δ × 42 格 × 6H；列名实为 `nregret`（脚本曾按 `normalized_regret` 取值失败，已更正）。
- 中位 H_suff = 1；92.9%（δ=.05）/ 90.5%（δ=.02）单元格 H_suff ≤ 2 → **UTILITY_ALREADY_SHORT_HORIZON**。
- 离线 AUROC 0.8367 vs 在线最优 0.685（δ=.05）；δ=.02 时 H_sep=1 AUROC 0.7904 / balacc 0.6728 → **OBSERVABILITY_GAP_SUPPORTED**。

### F4C 干预尺度挖掘
- bootstrap CIs 精确复现（seed 聚类，重写为 groupby 实现规避 MultiIndex int64 键问题）：δ=.05 H2−H1 +0.0512 [+0.000, +0.113]；δ=.02 −0.0746 [−0.158, +0.000] → **WEAK_SCALE_DEPENDENCE**，且 F4C 的 STAGE D 确证实验按 prereg 条件未触发 → DISABLED。

### F4 一致性审计
- `headline_numbers.csv` 14 项：12 VERIFIED / 3 STALE_REPORT / 1 OVERWRITTEN_ARTIFACT —— 所有口径分歧均已由该审计登账并出 erratum，本轮未发现新不一致。

## 四、legacy_fluctuation_response 冻结核验
- git HEAD = 6ac4942（与旧审计时一致）；2026-09-30 之后仅出现 `__pycache__/*.pyc` 运行时缓存（非研究制品，未跟踪）。
- 关键文档 FR_PROJECT_STATUS.md 哈希 36229054… == 旧 manifest。
- 结论：legacy 研究内容**冻结**，仅作历史参照，本轮不扩展。

## 五、登记在案的文档级瑕疵（不改任何判定）

| # | 位置 | 内容 | 处置 |
|---|---|---|---|
| 1 | F4-A 报告逐步 5M "median" 表 | 实为 median_high 惯例（n=8 偶数取上中位），未声明 | 以 canonical CSV 为准；数值本身无错 |
| 2 | F4A.1 报告 | 称 matched_phase_map.csv 为 90 行，实为 42 行（90 行并集在 f4a5 dataset.csv） | 以文件为准 |
| 3 | F4A5_METRIC_ERRATUM.md 正文 | MEDIAN_FOLD_MAE 写 0.4219，同文档 canonical recompute.json 为 0.4041 | 以 0.4041 为准，本轮独立复算确认 |
| 4 | F4A.5 报告消融表述 | "predictor ablation ΔMAE ≤ 0.01" 与 CSV 矛盾（max |Δ|=0.0368） | 数值仍小，不改变 M0 选择 |

## 六、环境说明
- 复核使用 miniforge env `minimind-exp`（numpy 2.2.6/pandas 2.3.3/scipy 1.15.3/sklearn 1.5.1）；系统 python3 仅 numpy 2.3.5 不足以支撑全部核算。
- 全部核算为确定性重算（同种子/同 CRN 语义），未引入新随机性结论。
