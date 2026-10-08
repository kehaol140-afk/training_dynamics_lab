# F4A.5 Metric Erratum（2026-09-30）

原报告 `docs/F4A5_FORMAL_MODELING_REPORT.md` §1 表格中部分 M0 指标**与 canonical artifact 不一致**。原报告不作修改。

## 事实

| M0 指标 | 原报告（错） | canonical `summary.json`/`model_comparison.csv` | 从 `loso_predictions.csv` 独立重算（POOLED_OOF） |
|---|---|---|---|
| MAE | 0.425 | 0.4249 | 0.4249 ✓ |
| RMSE | 0.545 | **0.7546** | 0.7546 ✓ |
| Spearman | 0.747 | **0.8195** | 0.8195 ✓ |
| R² | 0.587 | **0.3461** | 0.3461 ✓ |

- 差异原因：报告表格中 RMSE/Spearman/R² 为撰写时的**誊写错误**（STALE_REPORT——不与任何 artifact 或聚合方式对应；仅 MAE 正确，且 M1–M5 的 MAE 列与 canonical 一致）。
- 聚合方式澄清（prereg 未指定 Spearman 聚合，按 A1 要求双报）：canonical = **POOLED_OOF**（全部 90 行 out-of-fold 预测合并计算，即分析脚本/`model_comparison.csv` 实际口径）；**MACRO_FOLD**（11 个 held-out seed fold 平均）：M0 MAE 0.4509 / RMSE 0.6249 / **Spearman 0.8987**；MEDIAN_FOLD_MAE 0.4219。全部重算值存 `outputs/f4_consistency/f4a5_metric_recompute.json`。
- **对结论的影响：无**。verdict 判据（MAE ≤ 0.45 且 Spearman ≥ 0.7）在两种聚合下都满足（POOLED 0.4249/0.8195；MACRO 0.4509/0.8987）；M0 仍为 within-5% 最简选择（各模型 MAE 排序不变）；`ROBUST_PHASE_MODEL_NO_SCALE_EFFECT` 与 F4B gate（AUROC 0.842 / balacc 0.539，来自独立分类管线）均不受影响。
- 旧 74-state 版本混入排查：`loso_predictions.csv` 90 行、`dataset.csv` 90 行一致，无 74-state 旧缓存混入。
