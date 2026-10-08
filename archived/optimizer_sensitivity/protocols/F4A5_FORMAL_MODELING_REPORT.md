# F4A.5 — Formal Stability Modeling 报告

- 日期：2026-09-30。预注册 `docs/F4A5_FORMAL_MODELING_PREREG.md`（commit `a3ba97b`，先于拟合）。数据：**90 states**（F4A 74 + F4A.1 matched 16；canary 控制不入集），`outputs/f4a5_modeling/dataset.csv`。
- 工具：numpy/sklearn/scipy（环境已有，零新依赖）。Robustness：Huber(ε=1.35) on within-seed-demeaned M3。

## 0. 总判定

**`ROBUST_PHASE_MODEL_NO_SCALE_EFFECT`**；**F4B PreScreen gate：FAIL（AUROC 0.842 ≥ 0.75 但 balanced accuracy 0.539 < 0.65）→ `MODEL_PRESCREEN = DISABLED_BY_PREREG`**（primary SAU 不受影响）。

## 1. 模型比较（LOSO by seed，y = log10 R_curv_32，n=90）

| 模型 | MAE | RMSE | Spearman | R² |
|---|---|---|---|---|
| **M0（progress-only）** | **0.425** | **0.545** | **0.747** | 0.587 |
| M1（logN+p+交互） | 0.452 | 0.578 | 0.714 | 0.552 |
| M2（scale 固定效应+p+p²） | 0.431 | 0.565 | 0.724 | 0.568 |
| M3（M2+indicators） | 0.485 | 0.632 | 0.684 | 0.460 |
| M4（+seed 随机截距） | 0.620 | 0.793 | 0.588 | 0.292 |
| M5（+随机 p 斜率） | 0.620 | — | — | —（未见 seed 无斜率可迁移，等价 M4 预测） |

**选择规则（within-5% 最简）：M0。** progress 单变量即达到最优 LOSO——加 scale、indicators、seed 结构全部**损害**跨 seed 泛化。

## 2. 七问回答

1. **progress 是否主导？是**——M0 MAE 0.425（≤0.45 冻结线）且 Spearman 0.747（≥0.7）。经验形式：log10 R_curv_32 ≈ a − 9.5p + 11p²（Huber 稳健符号：p 负、p² 正）。
2. **scale effect 是否成立？否**——M2 不优于 M0（0.431 vs 0.425），scale 项跨 fold 符号不稳定，Huber 下符号翻转；与 F4A.1 的 NONMONOTONIC 一致。
3. **seed 随机效应方差占比？41.8%**（between-seed 截距方差/(+残差)）——大，但 M4 的 LOSO MAE 反而最差（0.620）：seed 偏移对**未见 seed 不可迁移**，只能作为不确定性来源报告，不能作为预测器。
4. **indicators 控制 progress+scale 后是否仍有独立解释力？否**——M3 (0.485) 劣于 M2 (0.431)；logA_m/logA_v/vfloor 跨 fold 符号不稳定（Huber 复核符号翻转）；clip_ratio/logK_lr 符号稳健（+0.80/+0.91）但移除任一项的 ablation ΔMAE ≤ 0.01（predictor_ablation.csv）。**F4A 的池化相关（A_theta −0.81 等）确认为 progress/LR 共线所致，无独立因果预测力。**
5. **可靠 cheap risk predictor？分类（R_curv≥10）LOSO：AUROC 0.842 / balanced acc 0.539 / PR-AUC 0.556**——排序能力可以（AUROC 过线），但 0.5 阈值决策的 balanced accuracy 明显低于 0.65 gate → **不启用 F4B model-prescreen**。
6. **healing-time 模型？暂不值得精确回归**——14 条区间删失、区间宽 0.05–0.30；healing_model.csv 更新（含 F4A.1 点）后仅作 descriptive 输入。

## 3. y_lambda（secondary）

M0 亦最优（LOSO MAE 0.0358 vs M2 0.0367、M4 0.0623）——λ_SO_32 同样由 progress 主导；scale 项不增益（F4A.1 的 EARLY 单调性被 progress 主效应吸收）。

## 4. 解释（SUPPORTED_INTERPRETATION）

- **宏观可预测、个体不可预测**：curvature 放大的对数幅度是 progress 的稳定函数（跨 seed LOSO MAE 0.43 dex ≈ ×2.7），但 seed 间 ±0.4–2 dex 的偏移构成不可约不确定性（41.8% 方差）。
- 这直接定义了 F4B 的设计空间：**在线方法不应依赖跨轨迹的先验模型**（无可靠 prescreen），而应依赖**当前轨迹自身已观测的量**（正是 SAU 的设计：用已展开的 g(H) 在线决策）——F4A.5 的负结果与 F4B primary 路线互为印证。

## 5. 制品

`outputs/f4a5_modeling/`：dataset.csv、model_comparison.csv、loso_predictions.csv、coefficient_summary.csv、variance_decomposition.csv、predictor_ablation.csv、healing_model.csv、huber_robustness.json、summary.json（verdict + gate）。
