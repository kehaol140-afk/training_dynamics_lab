# F4A.5 — Formal Stability Modeling 预注册

- 冻结日期：2026-09-30，先于任何拟合。数据：F4A 原 74 states + F4A.1 matched 16 states = **90 states**（canary 控制 state 不入数据集）。canary/F4A.1 已证 5M matched 家族等价。
- 目标：区分 scale / training-progress / seed 异质 / cheap indicators 的可解释统计模型。不强行寻找 power law。

## 1. Response（冻结）

- 主：`y_R = log10(R_curv_32)`（F4-A pipeline 全程 log10 口径）。
- 次：`y_lambda = lambda_SO_32`（原尺度）。
- healing interval：仅 secondary descriptive。

## 2. 数据与预测子（禁止 target-derived leakage）

每行：scale、logN、seed、step、progress p、y_R、y_lambda、log A_m、log A_v、log A_theta（**仅 sensitivity analysis**：与 LR/progress 强共线，不作 primary 因果预测子）、log(K_lr+1e-30)、vfloor=fraction_sqrtv_lt_eps、clip_ratio=首步 clip 系数、grad_norm_1（=1/clip_coef_1，当 coef<1，否则取 1.0；推导式记录）、lr_1、normalized_lr。

## 3. 模型候选（冻结；progress smooth ≤ 2 df：p + p²）

- **M0**：y ~ p（OLS）。
- **M1**：y ~ logN + p + logN:p。
- **M2**：y ~ scale-dummies(2) + p + p² + scale:p。
- **M3**：M2 + logA_m + logA_v + clip_ratio + logK_lr + vfloor。
- **M4**：M2 + seed 随机截距（within-seed demeaning OLS 估共享斜率 + per-seed 截距；LOSO 预测用训练 seeds 截距的总体均值）。
- **M5**：M4 + seed 随机 p 斜率（per-seed 斜率向 pooled 斜率收缩，k=2 经验 Bayes 近似；仅当 LOSO 稳定才保留）。
- y_lambda 用同 M0–M4 结构（M5 略）。

## 4. Robustness

(1) 上述 Gaussian/OLS；(2) **Huber**（`sklearn.HuberRegressor`，epsilon=1.35）作用于 M3 与 M4 的 within-seed demeaned 形式，检查关键系数（logA_v、logA_m、clip_ratio）符号与显著性稳定性。不安装新依赖（sklearn/scipy 已在环境）。

## 5. 交叉验证（冻结）

**LOSO（leave-one-seed-out）**：同 seed id 的所有 scale/state 同 fold。报告 MAE、RMSE、Spearman、R²（辅助）。分类目标 `R_curv_32 ≥ 10`（curvature-dominant）：额外报 AUROC、balanced accuracy、PR-AUC（cheap-risk classifier = M3 特征集 + logistic）。

## 6. 选择规则（冻结）

"within 5% of best LOSO MAE 时选最简单模型"（简单度排序 M0<M1<M2<M3<M4<M5）。同时检查：系数稳定性（跨 fold 符号一致）、scale 符号稳定性、seed 随机效应幅度、残差结构、held-out 极端 seed（100M 2044）表现。

## 7. 必答问题

(1) progress 是否主导？(2) matched coverage 后 scale effect 是否成立？(3) seed 随机效应方差占比？(4) 控制 progress+scale 后 A_v/clip_ratio/A_m/K_lr 是否仍有独立解释力（M3 vs M2 的 LOSO 增量 + 系数）？(5) 是否存在可靠 cheap risk predictor（LOSO AUROC/balanced-acc）？(6) healing-time 模型是否值得（样本/删失结构评估）。

## 8. Healing model（secondary）

interval-censored descriptive：报告各 (scale,seed) 的 p_heal 区间（沿用 F4A healing_intervals 语义，扩展 F4A.1 点后重算）。不做精确 AFT 回归（样本 14 条、区间宽）。

## 9. Verdict（冻结标签集）

`ROBUST_PHASE_MODEL_WITH_SCALE_EFFECT / ROBUST_PHASE_MODEL_NO_SCALE_EFFECT / STATE_AWARE_MODEL_IMPROVES_GENERALIZATION / SEED_EFFECT_DOMINANT / MODEL_PREDICTABILITY_LIMITED / NO_GENERALIZABLE_MODEL / UNRESOLVED`。操作化：ROBUST_PHASE := 最优 LOSO MAE ≤ 0.45 log10 且 Spearman ≥ 0.7；WITH_SCALE_EFFECT := M2/M3 中 scale 项（或 logN 项）在 ≥ 4/5 folds 符号一致且 M2 LOSO-MAE 优于 M0 至少 5%；STATE_AWARE := M3 LOSO-MAE 优于 M2 至少 5%；SEED_DOMINANT := M4 截距方差占比 > 50% 且 M4/M5 增量 > 10%。

## 10. F4B enable rules（冻结）

- Primary SAU：无条件允许（不依赖 scaling law）。
- **SAU+PreScreen 启用 gate**：cheap-risk classifier 的 LOSO **AUROC ≥ 0.75 且 balanced accuracy ≥ 0.65**。FAIL → `MODEL_PRESCREEN = DISABLED_BY_PREREG`，不得降低 gate。

## 11. 红线

不 push；不因结果调整本 prereg 任何阈值/模型集/判据；负结果保存；全部 seed 全报。
