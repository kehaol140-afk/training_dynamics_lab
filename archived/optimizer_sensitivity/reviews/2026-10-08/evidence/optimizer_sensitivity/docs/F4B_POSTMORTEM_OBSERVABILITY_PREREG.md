# F4B Postmortem — Observability Frontier 预注册（exploratory，post-hoc 但先冻结分析定义）

- 日期：2026-09-30，先于任何 postmortem 计算。数据：canonical `outputs/f4b_sau/delta_0p05|0p02/`（action_outcomes）+ F4A/F4A.1 metrics（h_H、g、indicators）。**不跑新 rollout。** 本研究为 F4B 负结果的 post-mortem，不构成 confirmatory claim。
- 核心问题：SAU 失败是（A）预算网格太紧，还是（B）**OBSERVABILITY GAP**——当在线可见信号能分辨"继续展开有害/有益"时，短 horizon 的效用已经达到近-oracle。

## 1. 冻结的 utility-by-horizon

对每 (delta, cell) 与 H ∈ {1,2,4,8,16,32}：action = argmin_a h_H·a（a ∈ {−δ,0,+δ}），**h_min = 0**（敏感性：0.005）；在 H_eval=32 真实 action losses 上计算 regret(H)、normalized_regret(H)（÷(max−min)）、harmful(H)（L(chosen)>L_zero）、oracle_action_match(H)。**不用未来信息选 H。**

## 2. 冻结的 frontier 量

- r*(cell) = min_H regret(H)；H_suff(cell, eps_u) = 最小 H 使 normalized_regret(H) ≤ normalized_regret_oracle + eps_u，**eps_u ∈ {0, 0.05, 0.10}**（主 0.05）。
- Benefit_continue(h) = normalized_regret(h) − min_{H>h} normalized_regret(H)；Cost_continue(h) = max(0, −Benefit)；**CONTINUE_WORTHWHILE(h) := Benefit_continue(h) ≥ 0.05**（h ∈ {1,2,4,8,16}；h=32 无未来）。

## 3. 冻结的 online 特征（prefix-only）

h 时点可见：log g(h)、λ̂(h)、q(h)、|h_eta(h)|、sign(h_eta(h))、sign 变化（vs 上一 readout）、Δlog g、Δh_eta、clip_coef_1、progress、scale dummies、logA_m、logA_v、vfloor、K_lr。禁止 g32/R_curv_32/未来量。

## 4. LOSO observability model（每 h 独立）

logistic（sklearn，C=1，max_iter 5000；secondary：depth-3 单树作 robustness）。目标 CONTINUE_WORTHWHILE(h)。报告 AUROC / PR-AUC / balanced accuracy（阈值 0.5）/ n_positive。**H_sep** := 最早满足 AUROC ≥ 0.75 **且** balacc ≥ 0.65 **且** n_positive ≥ 5 的 h；否则 NOT_OBSERVED。

## 5. Offline diagnostic（B8，独立于 online）

完整 F4A 诊断（R_curv_32、λ_SO_32、g32）LOSO logistic 预测 harmful(H=32)（L(chosen@32) > L_zero）与 CONTINUE_WORTHWHILE(1)。强 offline + 弘 online = observability gap 证据。

## 6. GAP 判定（B7，跨两 delta）

`OBSERVABILITY_GAP_SUPPORTED` 需同时：(i) 多数 cell 的 H_suff(0.05) ≤ 2（短 horizon 已近-oracle）；(ii) online H_sep 缺失或 H_sep > 多数 H_suff；(iii) offline diagnostic AUROC ≥ 0.75。其它标签：`ONLINE_RISK_EARLY_OBSERVABLE`（H_sep ≤ 2 且两指标达标）；`UTILITY_ALREADY_SHORT_HORIZON`（(i) 成立而 (ii) 不成立——online 可见但无需用）；`THRESHOLD_GRID_FAILURE_ONLY`（若放宽等价预算至 λ̂(2) 基线水平即可分离 continue-worthwhile——以 λ̂(2) 分布 vs 网格上限 0.30 的事后核对判定）；`NO_GENERAL_STRUCTURE` / `UNRESOLVED`。

## 7. 红线

不改旧 prereg/结果；负结果保留；不 push。
