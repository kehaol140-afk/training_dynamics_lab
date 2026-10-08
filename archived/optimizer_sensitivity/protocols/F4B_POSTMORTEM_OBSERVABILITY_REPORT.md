# F4B Postmortem — Observability Frontier 报告

- 日期：2026-09-30。预注册 `docs/F4B_POSTMORTEM_OBSERVABILITY_PREREG.md`（`09e3930`）。数据：canonical delta_0p05/0p02 action outcomes + F4A h_H/g/indicators；零新 rollout；h_min=0（敏感性 0.005 结论不变）。

## 0. Verdict

**主：`OBSERVABILITY_GAP_SUPPORTED`；次：`UTILITY_ALREADY_SHORT_HORIZON`（δ=.02 侧：风险在 H=1 即在线可见但无可获益）**

## 1. 核心 FACT

**Utility frontier**：两 δ 的 median H_suff(ε=0.05) 均为 **1**；H_suff ≤ 2 的 cell 占 **92.9%（δ=.05）/ 90.5%（δ=.02）**——绝大多数 cell 在 H≤2 已达 near-oracle 决策。mean normalized regret 随 H 的曲线（fig1）：δ=.05 单调恶化（H1 最优），δ=.02 在 H=2 谷底后恶化。

**Online observability（LOSO，目标 CONTINUE_WORTHWHILE(h)≥0.05）**：

| δ | h=1 | h=2 | h=4 | h=8 | h=16 | H_sep |
|---|---|---|---|---|---|---|
| .05 | AUROC .605 / balacc .487 | .685/.653 | .173/.486 | .400/.473 | .665/.600 | **NOT_OBSERVED** |
| .02 | **.790/.673 (n+=8)** | .500/.500 | .319/.486 | .130/.486 | .659/.559 | **1** |

（h=4/8 的 AUROC<0.5 为跨 fold 符号翻转，n+≈5 极不平衡所致，如实报告。）

**Offline diagnostic（B8）**：以 R_curv_32 + λ_SO_32 预测 harmful(H=32)：δ=.05 **AUROC 0.837**（n+=7）——**完整稳定性诊断对"长展开有害"有强排序力，而 δ=.05 的在线 prefix（任何 h≤16）都不可见**（在线最强 AUROC 0.685）。δ=.02 harmful 仅 4 例（AUROC 0.48，无信息量）。

## 2. 结论（对照 prereg §6 条件）

1. **SAU 失败的信息时序解释成立（δ=.05）**：长展开的真实风险存在且可被完整诊断预测，但在线可见信号在任何已观测 h 上不可分辨——不是阈值没调好（网格上限 0.30 vs λ̂(2) 基线 ~0.35-0.38 只是 SAU 退化的表层原因；即便放宽，h=2 的最优可达 AUROC 也仅 ~0.69）。
2. **但 gap 无 exploitation 价值**：H_suff 中位 = 1——可见性缺口存在的同时，短 horizon 已近-oracle（UTILITY_ALREADY_SHORT_HORIZON 同样成立）。两者共同构成对 SAU 类方法的**双重否定**：晚到（gap）且无用（已够短）。
3. δ 依赖结构：小 δ（0.02）下风险更早可见（H_sep=1）但 harmful 率低；大 δ（0.05）下风险更实质但更晚不可见——提示 intervention size 与 observability/useful-horizon 存在耦合（正是 STAGE C 的对象）。

## 3. 制品

`outputs/f4b_postmortem/`：utility_by_horizon.csv（420 行）、horizon_sufficiency.csv、continue_benefit.csv、online_observability.csv、offline_diagnostic.csv、observability_frontier.json、summary.json、figures/fig1。
