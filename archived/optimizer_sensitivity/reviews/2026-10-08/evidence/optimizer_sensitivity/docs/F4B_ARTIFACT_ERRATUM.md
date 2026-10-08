# F4B Artifact Erratum（2026-09-30）

原报告 `docs/F4B_STABILITY_AWARE_UNROLLING_REPORT.md` 引用的顶层共享 CSV 被 secondary 运行覆盖；本 erratum 记录审计与修复。原报告不修改。

## 事实与审计

1. **raw 完整**：`outputs/f4b_sau/raw/` 含 42×`*_d0.05` + 42×`*_d0.02` rollout 目录（各含 DONE + rollout.json，含 canary 值 4.755785 等）。**无任何 raw 丢失，未重跑任何 GPU rollout。**
2. **覆盖确认**：顶层共享表（per_method_regret / selected_horizons / threshold_selection / action_outcomes / per_seed_results，mtime 09-30 17:28；及其 17:36 的派生表）实际内容为 **δ=0.02** 的分析输出（分析脚本当时无条件写顶层路径）；δ=0.05 的表层数据被覆盖。两个 summary（summary.json、summary_delta0.02.json）未受影响。
3. **修复**：分析脚本已加 delta 命名空间（`outputs/f4b_sau/delta_0pXX/`，含 9 表 + summary + figures），两个 delta 均已**从 raw 重建**；重建后 primary 数字与历史 primary summary 逐项一致（verdict NO_METHOD_GAIN、SAU mean 0.1035 等，测试锁定）。
4. 顶层旧共享文件按指令**保留为历史 artifact**（其内容 = δ=0.02），不再被引用。
5. 影响面：原报告 §1 表格数字全部来自未被覆盖的 summary.json（δ=.05）与 summary_delta0.02.json——**报告数字无误**；被覆盖的只有中间 CSV 表。
6. 防再发：`tests/test_f4b_namespacing.py`（5/5 PASS）锁定——secondary 不覆盖 primary、delta/目录一致、headline 可由 canonical 重建、聚合口径不混用。
