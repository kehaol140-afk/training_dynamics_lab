# 项目总目录索引（PROJECT_TREE）

生成：2026-09-12 · 非侵入式只读盘点 · 项目根：`/home/b205/projects/fluctuation_response`（总大小 **9.3 GB**，1758+ 个输出文件 + 369 个 research 文件）

按功能分组（非机械树）。「是否仍被引用」基于 runner/test 的 import 与 input_hashes 静态检查。

## A. 核心代码（活跃，绝对不能动）

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `research/src/fr/` | FR 核心库：冻结分析模块（a0c/a0d/a0e/a0e2/a0f/a0f2…）+ replay 原语（full_state/perturbation/stream_rng/batch_manifest/direction/hashing/native_schedule/rng/minimind_v03_runtime 等） | ~40 .py + 1 .pyc | 3.9M（含 __pycache__ 1.1M） | 是（全部 runner/test） | 否 |
| `research/src/fr/theta_protocol.pyc` | **恢复性文件**：theta replay 分支字节码（sha 61e1bae3…），替代被删的冻结源 4a127209… | 1 | 7 KB | 是（所有冻结 runner 经 `run_theta_branch` 依赖） | 与 `__pycache__/theta_protocol.cpython-311.pyc` 内容相同（loader 机制所需，不算重复） |
| `research/scripts/` | 实验 runner：e0/e1/e3/e4 系列 + a0 系列 + check_*/scientific_preflight | ~46 .py | 668K pycache 另计 | 是 | 否 |
| `research/tests/` | unittest 套件（含 test_theta_protocol.py、test_v07_prereg_lock.py） | ~30 .py | 500K pycache 另计 | 是 | 否 |
| `research/configs/` | 锁定配置（e3c_confirmatory_v07.json 等 8 个 + 4 代 README） | ~13 | 小 | 是（input_hashes 锁定） | 2 个乱码名 README（见 DUPLICATE 审计） |
| `research/adapters/` | minimind 适配器模板 v02 | 2 | 小 | 部分 | 否 |
| `research/reports/` | TASK00 源码审计产物 | 2 | 小 | 存档 | 否 |
| `research/analysis/` | （空目录） | 0 | 0 | 否 | REVIEW |

## B. 上游训练代码（只读依赖）

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `minimind/` | 原始 MiniMind 训练代码 + **数据集 pretrain_t2t_mini.jsonl（1.18 GB，全项目最大文件）** | 55 | 1.2G | 是（trainer/model/dataset 被 replay 使用；v07 prereg 禁止修改） | 否 |

## C. 实验产出（outputs/，8.1 GB）

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `outputs/e3_confirmatory/v07/` | **v0.7 锁定 56 态 bank**：checkpoint 56×63MB + q + E2 manifest + theta mapping + prereg lock | 1481 | 4.6G | 是（一切 A0 实验输入源；SHA256 锁定） | 否（同尺寸 .pt 是不同 seed，非重复） |
| `outputs/e3_replication/v06/` | v06 replication bank（21 态，含 discovery seed） | 43 | 1.8G | 是（跨 partition 复现的未来输入） | 否 |
| `outputs/a0_fr_audit/` | **全部 A0 冻结实验**（A0C/H1/A0D/A0E/A0E2/A0F/A0F2/mv_* 等 32 个子目录） | 86 | 1.2G | 是（结论来源；冻结只读） | `mv_theory_final/gradient_g.pt` **1.07 GB**（被 mv_theory_consistent 取代，REVIEW）；A0F2 summary 与根目录 audit bundle 逐字节重复（有意归档副本） |
| `outputs/e3_mapping/` | e3 theta mapping | 12 | 447M | 是 | 否 |
| `outputs/stage_c_v04/`、`e0_replay/`、`e1_fluctuation/`、`e4a/`、`e4b_dynamic_state/` | 早期阶段产出 | 202 | ~250M | 部分（历史存档） | 否 |

## D. 文档与规格

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `docs/` | 项目文档：FR_PROJECT_STATUS.md、FR_EXPERIMENT_INDEX.md（今日新建）、TASK00 审计、研究手册 PDF、specs/、superpowers/ | 21+ | 576K | 是 | E0_SMOKE 大小写两份（内容不同，REVIEW）；2 个乱码名 TASK00 文件（其一为精确副本） |
| 根目录 `*.md` | RUN_ORDER / MANUAL_REVIEW / CODEX_AUDIT / CONFIRMATORY_PREREGISTRATION_V07 等 ~30 份协议与审计文档 | ~40 | 小 | 是 | 否 |

## E. 传入包与快照（历史incoming，可日后归档）

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `_fr_framework_incoming/` + `_fr_v02…v06_incoming/` | FR 框架 v0.1–v0.6 原始传入包（未解压进正式树的历史版本） | 7 目录 412 文件 | 4.8M | 否（被正式 research/ 取代） | 互相之间大量同源文件（版本演进，非逐字节重复） |
| `_fr_framework_backups/` | v02–v06 升级前 research/ 快照 ×6（每个含 .git 136K） | 651 | 3.7M | 否 | 各快照内 src/fr/rng.py 等跨版本逐字节相同（预期） |
| `FR_REPORT_AUDIT_BUNDLE_20260912/` | 2026-09-12 报告审计包：9 个实验的冻结副本 + MANIFEST | 43 | 11M | 是（审计证据链） | 09_a0f2/summary.json 与 outputs 正式版逐字节相同（**有意**归档副本） |

## F. 其他/残留

| 目录 | 用途 | 文件数 | 大小 | 仍被引用 | 可疑重复 |
|---|---|---:|---:|---|---|
| `FR_A0C_*`（17 个）+ `FR_A0_Audit_Package_v01/` + `FR_AUDIT_V02_A0C_RESIDUAL/` | 旧轮次 audit 工作区（每个 1–3 文件：README + fix 脚本/报告） | ~40 | ~250K | 否（结果已并入 a0_fr_audit） | 否（但属一次性工作区） |
| `baseline_manifest/` | 环境基线（conda/pip/sha256/nvidia-smi） | 8 | 44K | 是 | 否 |
| `logs/` | 运行日志（a0e2 日志 19:36 最新；e4b1 pid） | 3 | 36K | 是 | 否 |
| `checkpoints/full_state`、`checkpoints/native`、`data/eval` | **空目录** | 0 | 0 | 否 | REVIEW |
| 根目录 `rm` | **0 字节误建文件**（Sep 9） | 1 | 0 | 否 | REVIEW（疑似 `> rm` 手误） |
| `*.tar.gz` ×3（根目录） | paper_figures（0.1M）/ v06_code_audit（1.5M）/ report_audit_bundle（1.8M） | 3 | 3.4M | 部分 | paper_figures tar 与 `outputs/a0_fr_audit/paper_figures_v01/`（1.8M）为归档-解压对，REVIEW |
| `__pycache__/` ×37（全项目） | Python 字节码缓存 | — | 4.1M | 自动生成 | 可再生，SAFE |
