# 散落与重复代码审计（DUPLICATE_AND_ORPHAN_AUDIT）

生成：2026-09-12 · 只报告，不删除。判定基于 SHA256（内容级）与文件名/大小（疑似级）。

## A. 逐字节完全重复（内容级 SHA256 相同）

| # | 路径 | 大小 | SHA256 | 完全重复 | mtime | 可能用途 | 建议 |
|---|---|---:|---|---|---|---|---|
| A1 | `FR_REPORT_AUDIT_BUNDLE_20260912/09_a0f2_cross_partition/summary.json` ↔ `outputs/a0_fr_audit/cross_partition_extension_v01/summary.json` | 5,715,591 B | `d5840dedc3e70755…` | 是 | 2026-09-12 | 审计包要求内嵌冻结副本作证据链 | **KEEP**（bundle 为有意归档；正式版在 outputs） |
| A2 | `docs/TASK00_σ╜ôσëìσ╖▓τí«Φ«ìΣ║ïσ«₧.md` ↔ `docs/TASK00_实现审计结论_v0.2.md` | ~KB 级 | `085c60b88764ba36…` | 是 | 乱码名文件略旧 | 乱码文件名副本（编码事故产物） | **SAFE_DUPLICATE**（Codex 停止后可删乱码名者） |
| A3 | `_fr_framework_backups/*/src/fr/rng.py`（6 快照互相同）+ `_fr_v02…v06_incoming/*/research/src/fr/rng.py` + 正式 `research/src/fr/rng.py` | 2,388–2,958 B | 多版本 | 部分 | 2026-09-02 | 框架版本演进中的同源文件 | **ARCHIVE_LATER**（incoming/backups 整体归档时再判） |
| A4 | `recovery_candidates/theta_protocol/theta_protocol_v06_tar_4359B.py` ↔ `/tmp/theta_recover/…/theta_protocol.py` ↔ `_fr_v05/v06_incoming/*/theta_protocol.py`（4359 B，sha `17840228…`） | 4,359 B | `17840228…` | 是 | 2026-09-02 | v0.5/v0.6 时代旧版源码 | **KEEP**（恢复候选，见专项报告） |

## B. 归档-解压残留对

| # | 路径 | 大小 | mtime | 可能用途 | 建议 |
|---|---|---:|---|---|---|
| B1 | `FR_paper_figures_v01.tar.gz`（0.1M, Sep-2）↔ `outputs/a0_fr_audit/paper_figures_v01/`（1.8M） | tar 104K / dir 1.8M | 2026-09-02 | 论文图归档与解压目录并存（tar 疑为早期/部分打包） | **REVIEW**（核对 tar 完整性后二选一） |
| B2 | `FR_REPORT_AUDIT_BUNDLE_20260912.tar.gz`（1.8M）↔ `FR_REPORT_AUDIT_BUNDLE_20260912/`（11M） | 1.8M / 11M | 2026-09-12 | 审计包 tar 与已解压目录 | **KEEP** 目录版（含完整 9 实验副本），tar 可 ARCHIVE_LATER |
| B3 | `FR_v06_code_audit_snapshot.tar.gz`（1.5M, Sep-12） | 1.5M | 2026-09-12 | v06 代码审计快照（内含旧 theta_protocol.py，已提取为恢复候选） | **KEEP**（审计证据） |

## C. 同名不同内容 / 版本变体

| # | 路径 | 大小 | 差异 | 建议 |
|---|---|---:|---|---|
| C1 | `docs/E0_SMOKE_PASS_20260902.md`（485 B, 14:28）vs `docs/E0_smoke_PASS_20260902.md`（558 B, 12:14） | <1K | 大小写两套 E0 通过记录，**内容不同**（非副本，是两轮 smoke） | **REVIEW**（保留较新者，旧者归档） |
| C2 | `docs/TASK00_σ«₧τÄ░σ«íΦ«íτ╗ôΦ«║_v0.2.md`（sha `292aa045…`）vs `docs/TASK00_实现审计结论_v0.2.md`（sha `085c60b8…`） | ~KB | 乱码名版本与正名版本**内容不同**（两版实现审计结论） | **REVIEW**（人工比对后决定保留哪版） |
| C3 | `research/README_ΦäÜµ£¼Φ»┤µÿÄ.md` 与 `research/configs/README_Θàìτ╜«Φ»┤µÿÄ.md` | 小 | UTF-8 文件名被错误重编码为 Latin-1 显示（原名应为「脚本说明」「配置说明」） | **REVIEW**（Codex 停止后重命名回原中文名或统一英文名） |
| C4 | `outputs/e3_confirmatory/v07/seed_*/states/state_step_*.pt`（56 文件同尺寸 63.3M）及 v06 21 文件同尺寸 | 63.3M×56 | 同尺寸**非重复**（不同 seed/step 的合法快照） | **KEEP**（勿误判为重复） |

## D. 孤儿模块 / 一次性 runner / 空目录

| # | 路径 | 大小 | 最后修改 | 可能用途 | 建议 |
|---|---|---:|---|---|---|
| D1 | `research/src/fr/batch_stream.py` | 1,750 B | 2026-09-02 | **零 import 引用**（scripts/tests 均无；`check_native_batch_stream.py` 实际 import 的是 `batch_manifest`） | **REVIEW**（确认无动态加载后可 ARCHIVE_LATER） |
| D2 | `research/scripts/check_*.py`、`task00_source_audit.py`、`prepare_mid_checkpoint_v04.py` | 小 | v03–v06 时代 | 一次性检查/迁移脚本，完成态 | **ARCHIVE_LATER** |
| D3 | `research/analysis/`（空）、`checkpoints/full_state/`、`checkpoints/native/`、`data/eval/`（空） | 0 | Sep-1 | 早期规划的占位目录，从未使用 | **REVIEW**（确认后可在整理期移除空壳） |
| D4 | 根目录 `rm`（0 字节） | 0 | Sep-9 | 疑似 `> rm` 手误产生的空文件 | **REVIEW**（无内容，整理期可删） |
| D5 | `FR_A0C_*` ×17 + `FR_A0_Audit_Package_v01/` + `FR_AUDIT_V02_A0C_RESIDUAL/` | 共 ~250K | 9 月初 | 旧轮次一次性 audit 工作区（README + fix 脚本），结论已并入 a0_fr_audit | **ARCHIVE_LATER** |
| D6 | `mv_theory_v02…v06`、`mv_response_v01`、`mv_theory_final/`（**含 1.07 GB gradient_g.pt**） | 共 1.16G | 9 月初 | mv 理论迭代旧链，已被 mv_theory_consistent 取代 | **REVIEW**（summary 存档；1.07 GB 张量可在你确认后清理——全项目最大可回收项） |
| D7 | `_fr_v02…v06_incoming/` + `_fr_framework_incoming/` + `_fr_framework_backups/` | ~8.5M | 2026-09-02 | 框架传入包与升级前快照（含 v05/v06 时代 theta_protocol.py 旧版） | **ARCHIVE_LATER**（theta 旧源已另存恢复候选） |
| D8 | 全项目 37 个 `__pycache__/` | 4.1M | — | 可再生字节码缓存 | **SAFE**（整理期统一清理可再生） |
| E1 | `/tmp/tmp*/research/src/fr/theta_protocol.py`（12 B "# synthetic"） | 12 B | 今日 | 外部解压测试残留，**不在项目内** | 忽略（不属本项目资产） |

## 汇总

- 逐字节重复大文件：仅 A1（5.7 MB，有意归档副本）→ 实际可回收 ≈ 0。
- 主要可回收候选：**mv_theory_final/gradient_g.pt 1.07 GB**（D6，需你确认）+ 缓存/空壳/旧工作区 ~20 MB。
- 真正「孤儿代码」极少：仅 `batch_stream.py` 一个模块零引用，runner 侧一次性脚本属正常历史层积。
- 未发现任何会影响正在运行的实验的重复/孤儿问题。
