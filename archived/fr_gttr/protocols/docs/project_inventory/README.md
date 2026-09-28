# 项目盘点总摘要（README）

生成：2026-09-12 · 非侵入式整理（只扫描/索引/候选恢复，零移动、零删除、零修改）。详细内容见同目录各专项文档。

**1. 核心代码在哪？**
`research/src/fr/`（分析模块 + replay 原语，含冻结的 a0c/a0e2/a0f2 等）+ `research/scripts/`（runner）+ `research/tests/`。指标唯一定义：`research/src/fr/a0c_adamw_response_propagation_v01.py`。上游训练代码 `minimind/`（只读）。入口文档：`docs/FR_PROJECT_STATUS.md`、`docs/FR_EXPERIMENT_INDEX.md`。

**2. 冻结实验结果在哪？**
`outputs/a0_fr_audit/` 下 8 个冻结实验目录：response_propagation_v01（A0C）、H1_adamw_jacobian_v01、response_propagation_v02_epsilon（A0D）、curvature_decomposition_v01（A0E，作废存档）、curvature_decomposition_v02_bh_fixed（A0E2）、mv_theory_consistent、cross_state_generalization_v01（A0F）、cross_partition_extension_v01（A0F2）。输入 bank：`outputs/e3_confirmatory/v07/`（56 态，3.5 GB，锁定）。

**3. 哪些目录最混乱？**
根目录混放 19 个旧轮次工作区（`FR_A0C_*` 等）、7 个 `_fr_*_incoming` 传入包、6 个升级前快照、3 个 tar.gz；`docs/` 有大小写双份 E0 记录与乱码名 TASK00 文件；`research/configs/` 有乱码名 README；存在空壳目录（`checkpoints/full_state`、`data/eval`、`research/analysis/`）和一个 0 B 的误建文件 `rm`。

**4. 哪些文件明显重复？**
逐字节级：A0F2 summary 在审计包中的 5.7 MB 归档副本（有意）；`docs/TASK00_σ╜ôσëì…md` 是正式版的乱码名精确副本；theta_protocol 4359 B 旧版存在于 tar、/tmp、v05/v06 包（三处相同）。归档-解压对：paper_figures tar vs 目录。同尺寸 .pt 状态文件（56+21 个 63.3 MB）**不是**重复。

**5. 哪些现在绝对不要碰？**
`research/**`、`outputs/**`（含 v07/v06 bank 与全部冻结制品）、`minimind/`、`theta_protocol.pyc`、`CONFIRMATORY_PREREGISTRATION_V07.md`、`baseline_manifest/`、`FR_REPORT_AUDIT_BUNDLE_20260912/`、锁定配置、`logs/` 现有日志。一切移动/删除必须等 decision_utility_v01 的 Codex 任务完全结束（见 SAFE_REORGANIZATION_PLAN.md）。

**6. theta_protocol 源码恢复进度？**
**EXACT_FROZEN_SOURCE_FOUND = NO**。冻结源（6,341 B，sha `4a127209…`）在整机、`/tmp`、VSCode 历史、tar 成员、`/mnt/c`、`/mnt/d` 均未找到。当前正式路径为字节码恢复文件 `research/src/fr/theta_protocol.pyc`（sha `61e1bae3…`，编译自冻结源），已经 A0E2 baseline-replay 与 A0C 逐位复现验证，运行无风险。最接近的源码候选（VSCode 4887 B 中间版、v06 旧版 4359 B）已存 `recovery_candidates/theta_protocol/` 并附专项报告；若他日找到精确源，须 sha256 验证 + Phase 5 回归后才可替换。

**7. 未来如何整理？**
按 SAFE_REORGANIZATION_PLAN.md 六阶段执行：备份 → 冻结哈希基线 → 归档 incoming/旧工作区 → 代码规范化 → import 核验 + 冻结 replay 回归 → 仅删已验证重复。现实可回收约 **1.12 GB**（主体是 mv_theory_final 的 1.07 GB 张量，需专项确认）。本轮已落地：9 份索引/计划文档 + ARTIFACT_REGISTRY.csv（357 行，含 SHA256）+ 3 个恢复候选。

## 文档清单

| 文件 | 内容 |
|---|---|
| `PROJECT_TREE.md` | 按功能分组的目录索引（大小/文件数/引用/可疑重复） |
| `EXPERIMENT_CODE_MAP.md` | 10 个实验的 runner/src/tests/输入/输出/依赖/冻结映射 |
| `DUPLICATE_AND_ORPHAN_AUDIT.md` | 重复、孤儿、残留逐项审计（KEEP/ARCHIVE_LATER/REVIEW 建议） |
| `ARTIFACT_REGISTRY.csv` | 357 行制品登记（path/type/size/sha256/mtime/status/referenced_by） |
| `STORAGE_REPORT.md` | 磁盘占用 Top 榜与可回收估算 |
| `PROPOSED_LAYOUT.md` | 未来目录结构与旧→新映射、硬约束清单 |
| `SAFE_REORGANIZATION_PLAN.md` | Phase 0–6 安全重组计划与回滚预案 |
| `../../recovery_candidates/theta_protocol/THETA_PROTOCOL_RECOVERY_REPORT.md` | theta 恢复专项结论 |
