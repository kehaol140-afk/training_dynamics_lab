# CLOSEOUT_CORRECTIONS — 有限收尾勘误

**日期**: 2026-09-22。**性质**: 仅修正 6 个已有不一致；0 新增模型 forward/backward/HVP/optimizer update；不重做审计。
**依据**: claims_status.csv 自动重算、A0T/A0L/G3 现场文件、cpu_reference_tests.py 源码复核。原文件保留，本文件为准。

---

## C1. primary_verdict 双重计数 → 重算 = 25

从 `claims_status.csv` 程序化重算（不人工凑数）：

| primary_verdict | 数 | claims |
|---|---|---|
| SUPPORTED_WITHIN_SCOPE | 12 | C01,C02,C03,C05,C06,C07,C09,C10,C11,C19,C20,C21 |
| NOT_SUPPORTED | **6** | C04,C08,C12,C13,C16,C17 |
| NOT_TESTED | 3 | C22,C23,C25 |
| MIXED | 2 | C14,C15 |
| PARTIAL | 1 | C24 |
| UNRESOLVED | 1 | C18 |
| **合计** | **25** | |

- 不一致来源： audit_summary_v2.json 写 "NOT_SUPPORTED: 7（含 C14 成本面）" —— C14 的 primary_verdict 已是 MIXED，成本面是**子结论**，重复计入。
- 规则： primary_verdict 每 claim 计一次；子结论放 `secondary_labels` 列（C14: `SUPPORTED(效用)/NOT_SUPPORTED(成本)`；C15: `假设一/二SUPPORTED; 假设三UNRESOLVED`）。
- 26 = 25 + C14 一次重复；更正后 **25**，与 unique claim 数一致。

## C2. A0T 的 "128" = microsteps，非 optimizer steps

- 旧 retrospective 明确： 36 cell 共享 **128-micro 流**，Hmax=64（memory: fr-experiment-conventions）。
- v07 换算（B0 REPORT.md:21，现场记录）: **1 optimizer step = 2 microbatch (accum=2) = 1024 tokens**。
- ⇒ 128 microsteps = **64 optimizer steps = 65,536 tokens/rollout**（不是 128×1024=131,072）。
- H∈{4,16,64} 以 optimizer step 计；Hmax=64 = 流全长，吻合。
- 更正 `horizon_alignment_v2.csv` A07 行： `updates_per_rollout 128→64(microsteps=128)`，tokens `128×1024→65,536`。
- 列分四栏，不混： **microsteps=128 / optimizer_steps=64 / 有效主轨迹 tokens=65,536 / 辅助开销另计**（chi 评分、候选 rollout、FD 参考、eval —— 按 CPU T6 入账，不并入主轨迹）。

## C3. A0 系列 update 账： "未记账" ≠ "实际 0"

- 现场： a0t/a0u/a0s/a0k/a0l/a0n/a0o/a0p/a0q 的 `gpu_budget_log.json` **只有 wall_s/cell phase 与 occupancy，没有 optimizer_updates 字段**（ Earlier scan 的 "max_cum=0" 是字段缺失的默认值，不是 0 updates 的记录）。
- A0T 明确执行了真实干预 rollout（36 cell × 64 optimizer steps，GPU 计时 87.9s/cell phase 起）——实际发生了 optimizer updates，只是**该账本未按 update 单位记账**。
- 三个口径分开：
  1. **历史实际 updates（update 单位）**: **UNKNOWN**（未记账，非 0）；
  2. **本轮审计新增**: 0（只读复核）；
  3. **历史成本单位**: wall_s + evals（有账，a0t 各 phase 记录完整）。
- 更正 `cost_target_audit_v2.csv` RA-A0-SERIES 行： `actual_updates_run_level` 由 `0` 改 `UNKNOWN(not tracked in update unit)`；备注 "optimizer_updates=0 为字段缺失默认，非零消耗记录"。

## C4. data-mixture: A0L 已有实验，不能统称"从未测试"

- 现场： `a0l_response_mixture_feasibility_v01` = **Response-guided Continuous Mixture Feasibility**（contract 冻结）：臂 A native / B fixed group-1 全混 / C response mixture controller（K=50，events@step 50/100：snapshot→H=4 response h_j→q=softmax(−γh)→pi 更新）；seeds 2030–2032；pi_0=经验组比例 [0.2637,0.2472,0.2508,0.2383]；只改数据构成，无梯度注入。
- 结果（现场 REPORT）: **FORMAL CASE B**，且 B≥C on median（C 仅在 seed 2030 更好 +1.27e-3；2031/2032 B 更好 +2.19e-2/+7.46e-3）；controller 的 pi 漂移太小太稀（max |Δpi|~0.061）不足以分离 C 与 B；**per mandate 不进入 continuous response controller 路线**。
- 更正：
  - `claims_status.csv` C24: data-mixture 从"未测"改为 "**已有经验覆盖（pilot: 3 seeds, 2 events, 无扩展）**"；仍 PARTIAL（beta/eps/batch 未测）。
  - `validity_domain.csv` D07: data-mixture 动作移出 UNTESTED_EXTENSION 主行；**新 GTTR 近似对 mixture 通道的覆盖另列**: h_j 排序 seed-dependent、std 归一化使 q 塌缩、2 events/K=50 不足以测长时 —— 即"动作测过，近似外推未测"。

## C5. G3 continuation 与 E3  guard

**G3 起点核对**（training_manifest + REPORT）:
- G3a 算子控制臂: 在**冻结 G1a 26M clip=1.0 bank**（seed 2042 × states 26/64/102）上的 FD/B/BH_fixed/C 配对复制 —— bank 实验，无新训练。
- G3b 干预臂: **no-clip 训练，seeds 2042/2043/2044，states 26/64/102，clip_grad_norm 从未调用** —— 从 G1a bank stage checkpoint fork。
- ⇒ **G3 = continuation（起点 = stage checkpoint，非初始化）**。更正 `horizon_alignment_v2.csv` A18: `continuation_only 否→是`。
- 边界核对： G1a/G1b/G2 = **从初始化的基础训练**（planned 128 steps, terminated 102，单语料小规模）——continuation=否 正确；但"基础训练"≠"完整预训练"。
- ⇒ 原 Q5 表述"全为 continuation"更正为: **全部干预/验证实验（A0/B/C 系列）为 continuation；G1a/G1b/G2 为例外（基础训练，仍非完整预训练）**。

**E3 guard**: B0.6 的 rho=0.996（近单位单图 vs switching 族）是**离散 regime 切换假设的拟合问题**。**它不是短长收益反转的独立证据**——与反转无因果链。短长反转证据 = E1（A0T-extra H 反转）+ E2（A0P 长尾）+ CPU T2（单谷解析反例）。`AUDIT_REVIEW_SUPPLEMENT.md` §5.1 映射表已含此意，此处显式钉死。

## C6. T8 聚合门数值 bug: 0.535 → **0.593892**

- 冻结定义（EXACT-5, c1_gttr_probe_validation_v01.py:132-139，independent_recomputation 68/68 行验证）: **E = sqrt(Σ_a SSE_a / Σ_a TSS_a)**。
- 各点 "比值" 0.43/1.166 是 norm-scale 的 per-point E（即 r_a = sqrt(SSE_a/TSS_a)）⇒ SSE_a = r_a²·TSS_a。
- 等 TSS（每点 1.0，权重各 1/7）:

| 点 | ratio r_a | SSE_a=r_a²·TSS | TSS | 权重 |
|---|---|---|---|---|
| 1–6 | 0.43 | 0.1849 ×6 | 1.0 | 1/7 ×6 |
| 7 | 1.166 | 1.359556 | 1.0 | 1/7 |
| Σ | | **2.468956** | **7.0** | |

- **E = sqrt(2.468956/7) = 0.593892** ≈ 协议 §7 T8 的 "约 0.594" ✓。
- v1 `cpu_counterexamples.json` 的 0.535143 = **朴素均值** (6×0.43+1.166)/7 —— 源码注释自承"等 TSS 简化： 聚合 = 均值"且 verdict 自比自通过（`abs(agg_E − 均值公式)<1e-12`）。**参考实现 bug**（ratios 未平方即聚合）。
- 门结论不变： 聚合 **PASS**（0.5939<1）；最大单点 **FAIL**（1.166>1）。逐点 max 仍不得替代聚合门。
- 影响域： 仅 T8 合成示例；真实 C1 复算（68/68）用的是正确冻结定义，**不受影响**。
- 处置： `cpu_counterexamples.json` 的 T8 条目由本更正取代（不重写原文件）；若重跑 CPU 参考，T8 应得 0.593892。

---

## 更正后关键数值（一览）

| 项 | 旧 | 更正 |
|---|---|---|
| primary_verdict 合计 | 26（NOT_SUPPORTED 7） | **25**（NOT_SUPPORTED 6） |
| A0T 单位 | 128×1024 updates×tok | **128 micro = 64 optimizer steps = 65,536 tok** |
| A0 历史 updates | 0 | **UNKNOWN（未记账；非 0）**；本轮新增 0 |
| data-mixture | 从未测试 | **A0L pilot 已测**（边界内）；近似外推未测 |
| G3 continuation | 否 | **是**（G1a bank stage fork）；G1a/b/G2 仍否 |
| T8 聚合 E | 0.535143 | **0.593892**（门结论不变） |
| E3 (rho=0.996) | （隐含作反例） | **非反转证据**；反转证据 = E1+E2+CPU T2 |
