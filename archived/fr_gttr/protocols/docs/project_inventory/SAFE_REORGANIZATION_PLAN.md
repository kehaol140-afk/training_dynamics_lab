# 安全重组计划（SAFE_REORGANIZATION_PLAN）

生成：2026-09-12 · 本计划仅在**decision_utility_v01（Codex 任务）完全结束**后执行；本轮（2026-09-12）不执行任何 Phase。

## 前置条件（进入 Phase 0 前必须全部满足）

1. 另一个 Codex 会话的 validation-loss / data-reweighting 实验已结束：无相关进程、`outputs/` 无新文件写入、对方确认完成。
2. 取得项目所有者对 PROPOSED_LAYOUT.md 与 DUPLICATE_AND_ORPHAN_AUDIT.md 中各项 REVIEW 决策的签字。
3. 当前无其他正在运行的 FR 实验（`ps aux | grep -E 'research/scripts|fr_minimind'` 为空，logs/ 无新写入）。

## Phase 0 — 全量备份

- 用 `rsync -aH --numeric-ids`（或 tar）把项目根 9.3 GB 备份到**项目外**介质（另一磁盘/挂载）；至少覆盖 `research/`、`outputs/a0_fr_audit/`（冻结制品）、`docs/`。
- 单独复制三份恢复候选与 theta_protocol.pyc 到备份介质（双保险）。
- 备份完成后 `sha256sum -c` 抽样校验，记录备份位置与校验结果到 `docs/project_inventory/BACKUP_RECORD.md`。

## Phase 1 — 冻结哈希基线

- 对全部硬约束清单（PROPOSED_LAYOUT.md「绝对不能移动」表）生成 SHA256 基线：`find research outputs/a0_fr_audit outputs/e3_confirmatory outputs/e3_replication minimind -type f -exec sha256sum {} + > docs/project_inventory/FROZEN_BASELINE_SHA256.txt`（大文件多，预计 10–20 分钟）。
- 保留 v07 bank 的现有 `input_hashes`（summary.json 内）作为第二重基线，不得改动。

## Phase 2 — 归档重复与历史层积（只移低风险项）

按 PROPOSED_LAYOUT.md 映射执行，顺序：
1. 新建 `archive/` 及各子目录。
2. 移动 `_fr_*_incoming/`、`_fr_framework_backups/` → `archive/framework_*`（用 `git -C` 校验各快照 .git 完整）。
3. 移动 19 个旧工作区（`FR_A0C_*`、`FR_A0_*`、`FR_AUDIT_*`）→ `archive/legacy_workspaces/`。
4. 移动待比对文档变体 → `archive/doc_variants/`。
5. 移动根目录 tar.gz（保留 FR_v06_code_audit_snapshot 至 archive，theta 旧源已另存）。
6. 每步后 `du -sh` 与文件计数核对，写入执行日志。

**禁止在 Phase 2 触碰**：`outputs/` 内任何实验目录（含 mv 旧链——删除决策单独走 Phase 6）、`research/`、`minimind/`。

## Phase 3 — 代码布局规范化

- `research/README_ΦäÜµ£¼….md` → `README_scripts.md`；`research/configs/README_Θàìτ╜«….md` → `README_configs.md`（或恢复中文原名，先 grep 确认零引用）。
- 确认 `research/src/fr/batch_stream.py` 零引用（含动态加载检查：`grep -rn "importlib\|__import__" research/`）后移入 `archive/legacy_modules/`。
- 一次性 runner（check_*/task00/prepare_mid_checkpoint_v04）保持原位（移动收益低、破坏 RUN_ORDER 文档引用风险高）；如需移动，同步更新 RUN_ORDER_V*.md。

## Phase 4 — import 路径核验（不改 PYTHONPATH）

- 本计划**不修改** `PYTHONPATH=research/src` 约定，不建任何 symlink 遮盖原路径。
- 对移动过的 Python 文件逐一核验相对/绝对 import 仍可达；因 Phase 2/3 只移出无引用文件，预期零 import 变化——用 `python -c "import fr.<mod>"` 冒烟全部 a0 模块。

## Phase 5 — 冻结 replay 回归测试

- 运行 `python -m unittest research.tests.test_a0e2_curvature_decomposition_v02_bh_fixed`（14 测试，含 A0C/A0E 复现校验逻辑）。
- 补跑一次快速回归：A0C runner 以「复现校验」模式运行（若 runner 支持 summary 对比开关）；至少执行 `test_a0c_adamw_response_propagation_v01`、`test_v07_prereg_lock`、`test_v07_recovery_extractor`、`test_theta_protocol`。
- 对 v07 bank 抽样 3 个 checkpoint + 3 个 q + 3 个 manifest 做 SHA256 与 `FROZEN_BASELINE_SHA256.txt` 比对，必须 100% 一致。
- 任何不一致：立即停止，从 Phase 0 备份恢复。

## Phase 6 — 仅删除已验证重复

删除清单（逐项需所有者确认，且经前面 Phase 验证）：
1. `outputs/a0_fr_audit/mv_theory_final/gradient_g.pt`（1.07 GB）——确认 mv_theory_consistent summary 已留存、无人需要该张量。
2. `docs/` 乱码名精确副本（`TASK00_σ╜ôσëì…md`）。
3. 0 B 的 `rm` 文件；`checkpoints/full_state`、`checkpoints/native`、`data/eval`、`research/analysis/` 空壳。
4. 全项目 `__pycache__/`（`find . -name __pycache__ -type d -exec rm -rf {} +`）。
5. `FR_paper_figures_v01.tar.gz`（与解压目录比对完整性后二选一）。

**永不删除清单**：v07/v06 bank、minimind dataset、全部冻结 a0_fr_audit 制品与代码、审计包、prereg/协议文档、`theta_protocol.pyc`（除非精确源 4a127209… 找回并通过回归）。

## 回滚预案

- 每个 Phase 独立提交执行日志（含 before/after 的 du、find 计数、SHA256 抽查）。
- 任一 Phase 失败：仅回滚该 Phase 的移动（rsync 逆操作），不进入下一 Phase。
- 保留 `archive/` 至少一个季度再评估是否进一步压缩。
