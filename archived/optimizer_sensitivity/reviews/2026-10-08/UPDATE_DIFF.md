# UPDATE_DIFF — 旧结论 / 新增证据 / 当前判断 差异表

- 旧基线：ADAMW_RESEARCH_REVIEW（旧审计窗口 2026-09-30T06:11–06:54Z，`old_bundle_extract/ADAMW_RESEARCH_REVIEW.md`）
- 新证据窗口：旧窗口结束 → 2026-10-08 本轮复核
- 变更类型标签：**UNCHANGED**（无变化）/ **CONFIRMED**（新增证据支持）/ **REVISED**（收紧或改写）/ **WITHDRAWN**（撤回）/ **NEW**（旧窗口无对应内容）
- 规则：旧报告的建议不等于已执行；以服务器实际制品为准。所有数值本轮均已独立复核（`analysis/RECOMPUTE_NOTES.md`）。

## A. 项目级状态差异

| 项目 | 旧窗口末状态 | 当前状态 | 类型 |
|---|---|---|---|
| optimizer_sensitivity F0–F3.5 | 已完结 | 11 项关键制品/文档 sha256 == 旧 manifest，字节级不变 | UNCHANGED |
| optimizer_sensitivity F4 链 | F4-A 仅 17/74 单元完成、无 summary | F4A→F4A.1→F4A.5→F4B→F4B事后→F4C→一致性审计→决策备忘全部完结 | NEW |
| continuation_adaptation P4 | 已终局 STOP（2026-09-26） | 四件套哈希一致、mtime 全部 ≤09-27，零写入 | UNCHANGED |
| legacy_fluctuation_response | 参照/溯源 | git HEAD 6ac4942 不变；窗口后仅 .pyc 缓存（非研究制品） | UNCHANGED |

## B. 结论级差异表

| # | 旧结论（旧报告） | 新增证据（本轮核验来源） | 当前判断 | 类型 |
|---|---|---|---|---|
| 1 | F0 切向病态随尺度恶化 | 制品字节不变（f0 power_iteration.csv 90afb326…） | 成立 | UNCHANGED |
| 2 | F1 v-floor 因果闭合（∝v̂^−3/2） | 制品字节不变（f1 summary b3adb732…） | 成立 | UNCHANGED |
| 3 | F2 实用超梯度失败 | 制品字节不变（f2 summary 7069c2ae…） | 成立 | UNCHANGED |
| 4 | F2.5 信任域坍塌仅限相关谱 | 制品字节不变（f2_5 summary 083b312a…）；F4B/F4C 再次显示 H_best≡1 短视界 | 成立且被新链旁证 | CONFIRMED |
| 5 | F3/F3.5 谱不稳定但度量依赖 | 制品字节不变（f3/f3_5 summary） | 成立 | UNCHANGED |
| 6 | "F4-A 将回答尺度律问题"（预期） | F4-A 74/74 完成：13/14 严格自愈（erratum 改口）、LATE 24/24 曲率中性、无尺度单调趋势、种子高异质 | **PHASE_STRUCTURE_ONLY_NO_SCALE_TREND**：相结构真实、尺度律不存在 | REVISED |
| 7 | 5M 覆盖不足存疑 | F4A.1 16 新态重建 + 金丝雀五项门全过（stream_id 4090=4074+16；轨迹一致 loss h=1 7.443730） | 覆盖补齐：EARLY λ_SO 单调于 N（+1.0），R_curv 非单调 | NEW |
| 8 | 宏观演化可否预测未答 | F4A.5：M0 progress-only LOSO MAE 0.4249（fold 中位 0.4041 canonical）；prescreen balacc 0.539 → 按 prereg 禁入微观预测 | **宏观可预测、微观不可预测**；M0 为 canonical 模型 | NEW |
| 9 | SAU 被提出为候选方法（旧报告建议方向） | F4B：SAU 逐格 == FIXED H=1（λ̂(2)>B_λ=0.30）；oracle 中位 H=1；dominant-subset BEST_FIXED 中位 regret 0.3165 vs SAU 0；δ=.02 FIXED32 0.1317 < SAU 0.1810 | **NO_METHOD_GAIN**：方法化尝试被否定（建议≠已验证，本轮以实际制品否决） | WITHDRAWN（作为方法） |
| 10 | 收益 horizon 结构未刻画 | F4B 事后：中位 H_suff=1；92.9%/90.5% 单元 H_suff≤2 | **UTILITY_ALREADY_SHORT_HORIZON**：收益已在 H≤2 兑现，自适应无操作空间 | NEW |
| 11 | 在线可观测性存疑 | 离线 AUROC 0.8367（δ=.05）/0.7904（δ=.02）vs 在线最优 0.685/0.6728 | **OBSERVABILITY_GAP_SUPPORTED**：结构性观测差距成立 | NEW |
| 12 | 干预尺度依赖未知 | F4C：H2−H1 随 δ 变号（+0.0512 [0,.113] / −0.0746 [−.158,0]）；F2.5 表面 H_best≡1；STAGE D 未触发 | **WEAK_SCALE_DEPENDENCE**，确证实验 DISABLED_BY_PREREG | NEW |
| 13 | P4：C1 +1.587064 / C2 +0.019356，STOP | 封存表独立重算一致（C1 CI [1.2283,1.9644]；C2 [0.0140,0.0247]；0/8 seeds；Holm p=1.0）；哈希一致 | STOP 冻结，原样成立 | UNCHANGED |
| 14 | 外部验证必要性（旧报告展望） | F4_POSTMORTEM_DECISION_MEMO：首选选项 2（第二配方/数据集 50M–150M）；EXTERNAL_VALIDATION_PLAN 成文 | 方向维持、方案具体化（尚未执行；本轮不安排） | REVISED |
| 15 | 论文建模框架 | THESIS_MODELING_FRAMEWORK_FINAL + F4A5_THESIS_MODELING_FRAMEWORK：M0 为承载宏观结构的 canonical 基线 | 框架以 M0/pooled-OOF/LOSO 为准 | NEW |

## C. 数值级关键复核（旧 vs 本轮）

| 量 | 旧报告值 | 本轮独立重算 | 一致性 |
|---|---|---|---|
| C1 − A_best 均值 | +1.587064 | +1.587064 | 完全一致 |
| C1 CI | [1.2352, 1.9550] | [1.2283, 1.9644] | 实现细节级差异，结论一致 |
| C2 − A_best 均值 | +0.019356 | +0.019356 | 完全一致 |
| C2 CI | [0.0141, 0.0246] | [0.0140, 0.0247] | 一致 |
| B − A_best / D − A_best / B − D | −0.013881 / +0.060853 / −0.074735 | 同 | 完全一致 |
| F4A.5 M0 LOSO MAE | 0.4249 | 0.4249（六模型全部复算 == model_comparison.csv） | 完全一致 |
| F4A.5 fold MAE 中位 | canonical 0.4041（erratum 正文误写 0.4219） | 0.4041 | 以 canonical 为准 |
| F4B SAU vs FIXED1 | 报告：逐位相等 | 逐位相等（summary 直读 + 比较） | 完全一致 |
| F4C H2−H1 CI | 报告 [+0.000,+0.113] / [−0.158,+0.000] | 精确复现 | 完全一致 |
| 金丝雀 stream_id | 4090 | 4090（=4074+16） | 完全一致 |

## D. 撤回/收紧汇总

- **撤回**：尺度单调恶化律（改为非单调相结构）；"SAU 作为可行方法"（NO_METHOD_GAIN）。
- **收紧**：F4-A 愈合率 14/14 → 13/14（erratum）；F4A.5 中位 fold MAE 以 canonical 0.4041 为准；F4 结论外推范围限定为单配方银行态（外部验证未执行）。
- **维持**：F0–F3.5 全部冻结结论；P4 STOP。

## E. 缺陷登记（文档级，不改判定）

详见 `analysis/RECOMPUTE_NOTES.md` §五：median_high 惯例未声明；F4A.1 行数笔误（90 vs 42）；F4A5 erratum 正文数字矛盾；F4A.5 消融阈值表述过紧（max |Δ|=0.0368）。
