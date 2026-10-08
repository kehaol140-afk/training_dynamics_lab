# F4 Consistency & Provenance Audit（2026-09-30）

STAGE A 总审计。逐项状态见 `outputs/f4_consistency/headline_numbers.csv`（14 项：12 VERIFIED / 3 STALE_REPORT（F4A.5 M0 的 RMSE/Spearman/R² 誊写错误，见 `docs/F4A5_METRIC_ERRATUM.md`）/ 1 OVERWRITTEN_ARTIFACT（F4B 顶层共享 CSV，见 `docs/F4B_ARTIFACT_ERRATUM.md`））。

## A1 F4A.5 指标一致性

- canonical = POOLED_OOF（`model_comparison.csv`/`summary.json`，与 `loso_predictions.csv` 独立重算逐位一致：M0 MAE 0.4249 / RMSE 0.7546 / Spearman 0.8195 / R² 0.3461）；MACRO_FOLD 备报（MAE 0.4509 / Spearman 0.8987）。prereg 未指定聚合 → 双报并命名锁定（测试）。
- 报告差异定性：**STALE_REPORT**（仅 RMSE/Spearman/R² 三格，无 74-state 旧缓存或列错位证据；dataset/loso 表均为 90 行）。
- verdict 稳健性：判据在两种聚合下均满足 → `ROBUST_PHASE_MODEL_NO_SCALE_EFFECT` 不变；F4B gate 不变。

## A2 F4B primary/secondary 覆盖

- raw 42+42 完整（未重跑）；顶层 CSV 被 δ=0.02 覆盖属实；已从 raw 重建 `delta_0p05/`、`delta_0p02/` 双命名空间（9 表+summary+figures），primary 重建值与历史 summary 一致（测试锁定）；旧顶层文件保留为历史。

## A3 headline 交叉核对

`outputs/f4_consistency/headline_numbers.csv`：F4A.1（matched R_curv/λ_SO）与 F4B（regret/harmful/horizon）全部 headline 与 canonical raw 逐项 VERIFIED；仅上列 erratum 项异常。

## A4 反覆盖

分析管线 delta 命名空间化；`tests/test_f4b_namespacing.py` 5/5 PASS（secondary 不覆盖 primary、delta↔目录一致、headline 可重建、POOLED/MACRO 名称不混用、canary 值锁定）。

## Consistency verdict

**`CONSISTENCY_RESTORED`**（raw 零丢失、零重算、全部 canonical 可复现且测试锁定；两处报告/表层层错误已以 erratum 披露且不影响任何 verdict）。→ 允许进入 STAGE B。
