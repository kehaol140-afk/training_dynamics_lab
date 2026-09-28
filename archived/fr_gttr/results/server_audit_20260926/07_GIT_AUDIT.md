# 07 — Git 全量审计（只读，未做任何修改）

审计范围：/home/b205/projects（深度 4）全部 .git；/home/b205 浅层（深度 3）其他仓库。

## 1. 仓库清单

| 仓库 | 性质 | remote | 状态 |
|---|---|---|---|
| **projects/fluctuation_response** | 主研究仓库 | git@github.com:kehaol140-afk/fluctuation-response.git | master，3 commits，**全部未 push** |
| projects/fluctuation_response/_fr_framework_backups/research_before_v0{2,3,4,5,5,6}_*/（6 个） | 备份壳 | 无 | **全部零提交**（仅 staged .gitignore） |
| projects/fluctuation_response/minimind | **非仓库**（无 .git） | — | 被主仓库 ignore |
| ~20 个 /home/b205 下其他仓库 | 与本研究无关（meeting/Jitsi/NRET/baselines/framework-audit 等） | 见下表 | 不影响本项目 |

## 2. 主仓库 fluctuation_response（重点）

- **branch**：仅本地 `master`；无任何 remote-tracking ref（从未成功 fetch/push）。
- **log（全部 3 条）**：
  1. `6d89d09` F0 前基线：研究代码/测试/配置/实验协议与关键索引首次入库（2026-09-25 12:07）
  2. `b0bec9f` 忽略规则修正：FR_A0C_* 需同时匹配目录与游离文件
  3. `591b4ab` 整理：README 与 G1c 大文件清理记录
- **未 push**：全部 3 commits（`log --branches --not --remotes` = 全部历史）。**唯一副本在本机。**
- **status**：无 staged/modified；**11 项 untracked**：`FR_GTTR_F0_CALFIX_Revalidation_v1.md`、`GTTR_F0_Implementation_Consistency_Package_v1.zip`、`research/scripts/f0_{calfix_run,f0_diag_small_T,ic_adapter,terminal_rule_run}_v01.py`、`research/scripts/f1_closure_run_v01.py`、`research/src/fr/f0_terminal_rule_v01.py`、`research/tests/test_f0_{clipjvp_audit,terminal_rule}_v01.py`、`tools/`。
  → 即 **F0/CALFIX/F1/F2 的代码与协议从未进入版本控制**（F2a/F2b 无独立脚本目录，分析在既有脚本内）。
- **stash**：空。**tag**：空。**LFS**：未安装、未使用。
- **tracked 规模**：414 文件 / .git 5.6M；无 >100MB tracked 文件。
- **.gitignore 要点**：outputs/、checkpoints/、data/、logs/、minimind/、`_fr_*/`、`FR_A0C_*`、`FR_REPORT_AUDIT_BUNDLE_20260912/`、`*.tar.gz`、`*.log`、`r`、`rm`、`ENVIRONMENT_V07.txt`、`baseline_manifest/`、`.claude/.codex/.agents/`、`*.pt/*.pth/*.bin`、`.env*` 等；**例外**：`!research/src/fr/theta_protocol.pyc`（唯一强制入库的 pyc）。
- **工作区 >100MB 文件**：79 个，全部在被 ignore 的 outputs/ 下（G1b/G2/G3 的 state/q 张量与 pretrain_hq.jsonl 等）。

## 3. 6 个备份仓库（_fr_framework_backups/）

`research_before_v02(20260902_005245)/v03/v04/v05×2/v06` 六个目录各含独立 .git：
全部 = `git init` 后仅 add 了 .gitignore、**零 commit**，内容全 untracked——**备份在 git 层面零保护**，仅是文件快照（合计 2.4M）。
其 .gitignore：`__pycache__/ *.pyc *.log *.pt *.pth *.bin outputs/ cache/`。

## 4. /home/b205 其他仓库（浅层，与本研究无直接关系，登记备查）

| 仓库 | 状态要点 |
|---|---|
| meeting（cheatofrom/meeting） | master ahead 31 未 push；3 本地分支无 upstream |
| meeting/jitsi-meet-web-src | feature/ai-assistant 本地分支 |
| NRET、MeasureConsistentSFT、ThermoPathCV | 本地私有仓库（13/11/1 commits 未 push，无 remote） |
| framework-audit/llamafactory、axolotl(+worktree) | shallow clone，detached HEAD |
| baselines/{skewadam, Adam-mini, nanowhale} | 干净 clone |
| .hermes/hermes-agent | 1 修改文件 |
| FDA_INIT、minimind-master | **空壳 .git（非仓库残留）** |

## 5. 风险汇总与建议（只建议，不执行）

1. **最高风险**：主仓库 3 commits 从未 push（无任何异地副本）；服务器单点故障将丢失全部研究代码/协议/索引历史（outputs 另有 SHA256 锚点但本体同样单点）。
2. F0/F1/F2 前沿工作（11 项 untracked）未入库——建议 `git add` + commit（.zip 16K 可入库）。
3. 6 个备份壳仓库无意义（零提交）——未来重构时可直接当普通目录归档。
4. minimind 无版本控制，来源只有 zip 级 SHA256——若上游持续更新，建议补文件级 manifest 固定当前副本。
5. 未 push 的本地私有仓库（NRET 等）与本研究无关，但同样无异地备份。
