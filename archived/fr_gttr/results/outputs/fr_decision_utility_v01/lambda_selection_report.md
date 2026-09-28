# LAMBDA SELECTION REPORT — FR_DECISION_UTILITY_V01

lambda 只在开发状态（seeds 2030–2037 中未参与 A0C–A0F2 锁定的少量状态）上扫描确定；
正式评价状态（新 seeds 2038–2041）不参与 lambda 选择。

## 开发扫描（候选 lambda，+lambda 分支，V_score 读数）

准则 A（严格局部线性）：mean slope-dev vs λ0 ≤ 0.25 且 clip 穿越 ≤ 0.10。
准则 B（决策信号，回退）：点态线性不成立时，在满足 clip 穿越 ≤ 0.10 的候选中取
最大 lambda（最大化 SNR/符号一致率）；排序信号（组内 Spearman）不依赖点态线性。

| lambda | mean |hhat_C| | mean slope-dev vs λ0 | p90 slope-dev | clip crossing | sign agree | mean signal | mean dev | SNR | Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.00313257446444 | undefined | undefined | 0 | 0.665492957746 | 3.13257446444e-05 | 6.31425546571e-05 | 0.4961114547 | 0.272222222222 |
| 0.025 | 0.00313257446444 | 4.09389952911 | 8.23896789216 | 0.0416666666667 | 0.781690140845 | 7.83143616111e-05 | 7.19936023603e-05 | 1.08779612415 | 0.416666666667 |
| 0.05 | 0.00313257446444 | 1.88487238239 | 4.92777794025 | 0.0729166666667 | 0.848591549296 | 0.000156628723222 | 8.9629900014e-05 | 1.74750527667 | 0.561111111111 |
| 0.1 | 0.00313257446444 | 1.20096880596 | 2.39066199207 | 0.166666666667 | 0.904929577465 | 0.000313257446444 | 0.000128766365997 | 2.43275830625 | 0.719444444444 |

## 冻结决定

- 主 lambda（primary）: **0.05**
- 保守 lambda（conservative）: **0.025**
- 冻结规则: B_decision_signal
- 开发组内 Spearman（hhat_C vs 实际 ΔL，跨动作）: 0.561111111111

理由: 准则 A（严格局部线性，slope-dev vs λ0=0.01 ≤ 0.25）在所有候选上均不通过：开发扫描显示响应带有 ~9.0e-05 量级的绝对 deviation floor（replay 位级确定，非随机噪声），点态线性不成立。改用准则 B（决策信号）：在满足 clip 穿越 ≤ 0.10 的候选中取最大 lambda 以最大化 SNR、符号一致率与组内排序信号（三者随 lambda 单调上升；primary 处 mean within-state Spearman = 0.561，是本实验 Level 2/3 决策评价的关键，不依赖点态线性）。clip 稳定门把 lambda 封顶（0.10 的穿越率 > 0.10 越限）。

## 通过准则核对

- PASS — action_gradients_finite
- PASS — convex_mixture_by_construction
- PASS — not_all_actions_near_zero
- FAIL — strict_locality_any_lambda_mean_slope_dev_le_0.25
- PASS — primary_clip_crossing_rate_le_0.10
- PASS — freeze_rule
- PASS — primary_mean_within_state_spearman_ge_0.5

说明: action gradient 为 g_action = g + lambda*(g_j - g)，是与基准 batch 梯度同归一化
（8 样本均值）的凸混合；lambda 冻结后对正式状态只读使用。
