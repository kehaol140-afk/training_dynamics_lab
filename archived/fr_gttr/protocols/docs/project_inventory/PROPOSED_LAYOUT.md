# 建议未来目录结构（PROPOSED_LAYOUT）

生成：2026-09-12 · **仅建议，本轮不执行任何移动**。基于当前真实项目（9.3 GB）设计，原则是：冻结路径不动、引用不变、只把「历史层积」从活跃面剥离。

## 目标结构

```
project/                        # /home/b205/projects/fluctuation_response
  docs/                         # 保留原状（含 project_inventory/、specs/）
  research/                     # 保留原状（src/fr、scripts、tests、configs）
  minimind/                     # 保留原状（上游只读依赖）
  outputs/                      # 保留原状（全部实验产出；被 input_hashes 锁定，绝不可动）
  archive/                      # 【新建，Codex 停止后】所有历史传入包/快照/旧工作区
    framework_incoming/v01..v06/
    framework_backups/v02..v06/   # 各快照内 .git 一并归档
    legacy_workspaces/            # 现根目录 FR_A0C_* 等 19 个工作区
    legacy_mv_chain/              # mv_theory_v02..v06、mv_response_v01
    doc_variants/                 # E0/TASK00 待人工比对的版本变体
  artifacts/                    # 【可选】大制品软登记目录（不搬文件，只放 README 指针）
  recovery_candidates/          # 已建（theta_protocol 候选，见专项报告）
  logs/                         # 保留原状
  baseline_manifest/            # 保留原状
  checkpoints/ data/            # REVIEW 后删除空壳（full_state/native/eval）
```

## 旧路径 → 建议新路径映射

| 旧路径 | 建议新路径 | 性质 |
|---|---|---|
| `_fr_framework_incoming/`、`_fr_v02_incoming/`…`_fr_v06_incoming/` | `archive/framework_incoming/v01…v06/` | 移动（低危，无引用） |
| `_fr_framework_backups/research_before_v0*` | `archive/framework_backups/` | 移动（低危） |
| `FR_A0C_*` ×17、`FR_A0_Audit_Package_v01/`、`FR_AUDIT_V02_A0C_RESIDUAL/` | `archive/legacy_workspaces/` | 移动（低危；工作区内 fix 脚本的路径以根为基准，移动后在归档内只读保存） |
| `outputs/a0_fr_audit/mv_theory_v02…v06/`、`mv_response_v01/` | `archive/legacy_mv_chain/`（保留 summary，迁出 outputs 活跃面）或就地保留仅 summary | **谨慎**：input_hashes 记录的是文件内容哈希而非路径，移动 summary 不破坏哈希；但为绝对安全建议**就地保留**、只删除 1.07 GB 张量（单独决策） |
| `outputs/a0_fr_audit/mv_theory_final/gradient_g.pt` | 不移动；确认后删除（全项目最大可回收项 1.07 GB） | 删除需专项确认 |
| `docs/E0_smoke_PASS_20260902.md`、`docs/TASK00_σ«₧τÄ░…md` | `archive/doc_variants/`（人工比对后） | 移动（低危） |
| `research/README_ΦäÜµ£¼…md`、`research/configs/README_Θàìτ╜«…md` | 重命名为 `README_scripts.md`、`README_configs.md`（或恢复中文名） | 重命名（低危，无 import 引用） |
| 根目录 `rm`（0 B） | 删除 | 低危 |
| `checkpoints/full_state`、`checkpoints/native`、`data/eval`、`research/analysis/` | 删除空壳 | 低危（确认零引用后） |
| `FR_paper_figures_v01.tar.gz` | `archive/` 或删除（与 outputs 解压目录二选一） | REVIEW |
| 全项目 `__pycache__/` | 统一清理（可再生） | 低危 |
| 根目录 `*.tar.gz` ×2（report_audit_bundle、v06_snapshot） | `archive/` | 低危 |

## 绝对不能移动 / 不能改（硬约束）

| 路径 | 原因 |
|---|---|
| `research/src/fr/**`（含 `theta_protocol.pyc`） | 冻结模块 + input_hashes 锁定；任何路径/内容变化即破坏冻结链 |
| `research/scripts/a0*.py`、`research/tests/test_a0*.py` | 冻结 runner/test |
| `outputs/e3_confirmatory/v07/**` | v0.7 locked bank，所有实验输入源；含 56×63MB checkpoint + q + manifest |
| `outputs/e3_replication/v06/**` | v06 bank（未来跨 partition 复现输入） |
| `outputs/a0_fr_audit/{response_propagation_v01,curvature_decomposition_v01,curvature_decomposition_v02_bh_fixed,response_propagation_v02_epsilon,H1_adamw_jacobian_v01,mv_theory_consistent,cross_state_generalization_v01,cross_partition_extension_v01}/**` | 冻结实验制品（报告/数字来源） |
| `minimind/` | v07 prereg 明确禁止修改上游训练代码 |
| `docs/FR_PROJECT_STATUS.md`、`docs/FR_EXPERIMENT_INDEX.md` | 刚建立的上下文恢复入口 |
| `CONFIRMATORY_PREREGISTRATION_V07.md`、`baseline_manifest/`、`FR_REPORT_AUDIT_BUNDLE_20260912/` | 协议锁定文件与审计证据链 |
| `research/configs/e3c_confirmatory_v07.json` 等锁定配置 | input_hashes 锁定 |
| `logs/` 现有文件 | 运行记录（新日志仍在写入） |

## 执行时机

- 上表所有移动/删除/重命名 **必须等 decision_utility_v01（Codex 任务）完全结束**后，按 SAFE_REORGANIZATION_PLAN.md 的 Phase 0–6 执行。
- 本轮（2026-09-12）已执行且安全的仅为：新建 `docs/project_inventory/` 文档、`recovery_candidates/theta_protocol/` 候选——均为新增路径，不遮盖任何原路径。
