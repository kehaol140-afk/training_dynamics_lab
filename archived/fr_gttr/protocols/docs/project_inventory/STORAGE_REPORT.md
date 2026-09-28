# 磁盘占用报告（STORAGE_REPORT）

生成：2026-09-12 · 只读统计，未删除任何文件。

## 总览

| 项 | 值 |
|---|---|
| 项目总大小 | **9.3 GB**（du -sh .） |
| 文件总数 | ~3,300（outputs 1,758 + research 369 + 其余 ~1,200 含小文件与 manifest） |
| >1 GB 文件 | 1 个（数据集） |
| >100 MB 文件 | 2 个 |
| >100 MB 目录 | 5 个 |

## Top 30 最大文件

| # | 大小 | 路径 |
|---|---:|---|
| 1 | 1,183.6 MB | `minimind/dataset/pretrain_t2t_mini.jsonl`（训练数据集，KEEP） |
| 2 | 1,065.4 MB | `outputs/a0_fr_audit/mv_theory_final/gradient_g.pt`（旧链张量，**REVIEW——最大可回收项**） |
| 3 | 63.3 MB ×22 | `outputs/e3_confirmatory/v07/seed_20{30..37}/states/state_step_*.pt`（56 个中的前 22；全组 56×63.3MB ≈ 3.54 GB，v07 bank，KEEP-LOCKED） |
| 23 | 63.3 MB | `outputs/e0_replay/v03_extended/fr_full_state_after_warmup.pt` |
| 24–30 | 63.3 MB | v06 replication states（21×63MB ≈ 1.39 GB 中的其余，KEEP） |

（v07 56 态 + v06 21 态同尺寸 63.3 MB，属不同 seed/step 的合法快照，非重复。）

## Top 20 最大目录

| # | 大小 | 目录 |
|---|---:|---|
| 1 | 4.6G | `outputs/e3_confirmatory/v07/`（v07 locked bank） |
| 2 | 1.8G | `outputs/e3_replication/v06/`（v06 bank） |
| 3 | 1.2G | `outputs/a0_fr_audit/`（其中 mv_theory_final 独占 1.07G） |
| 4 | 1.2G | `minimind/`（dataset 1.18G） |
| 5 | 447M | `outputs/e3_mapping/` |
| 6 | 87M | `outputs/stage_c_v04/` |
| 7 | 86M | `outputs/e1_fluctuation/` |
| 8 | 75M | `outputs/e0_replay/` |
| 9 | 58M | `outputs/a0_fr_audit/mv_theory_v06/`（旧链，REVIEW） |
| 10 | 11M | `FR_REPORT_AUDIT_BUNDLE_20260912/`（审计证据，KEEP） |
| 11 | 4.1M | 全项目 `__pycache__/` ×37（可再生） |
| 12 | 3.9M | `research/`（全部活跃代码） |
| 13 | 3.7M | `_fr_framework_backups/`（6 快照，各含 .git 136K） |
| 14 | 1.8M | `outputs/a0_fr_audit/paper_figures_v01/` |
| 15 | 1.1M | `_fr_v06_incoming/` |
| 16 | 948K | `_fr_v05_incoming/` |
| 17 | 852K | `_fr_v04_incoming/` |
| 18 | 796K | `_fr_v03_incoming/` |
| 19 | 648K | `_fr_v02_incoming/` |
| 20 | 576K | `docs/` |

## 可回收空间估算（不执行）

| 候选 | 空间 | 条件 |
|---|---:|---|
| `mv_theory_final/gradient_g.pt` + 旧 mv 链（v02–v06、mv_response） | **~1.10 GB** | 确认 mv_theory_consistent 已永久取代且无人需要该张量 |
| v06 replication bank（若未来确认不做跨 partition 复现） | ~1.39 GB | **不建议**——A0F2 结论的下一步正是 v06 复现 |
| `__pycache__` 全清 | 4.1 MB | 随时可再生 |
| 根目录 `*.tar.gz` ×3 + paper_figures 二选一 | ~3.4 MB | 归档后 |
| `_fr_*_incoming/` + `_fr_framework_backups/` | ~8.5 MB | theta 旧源已存恢复候选后 |
| 空目录 + 0 B `rm` + 乱码副本 | <1 MB | 确认后 |
| **合计（现实可期待）** | **~1.12 GB**（≈12%） | 主体就是 mv_theory_final 张量 |

**不可触碰**：v07 bank 3.54 GB、v06 bank 1.39 GB、minimind dataset 1.18 GB、全部冻结 a0_fr_audit 制品（除上述旧 mv 张量外）。
