# SELF_CHECK — 反向查漏自检

- 审计目录：`/home/b205/projects/training_dynamics_lab/audits/ADAMW_RESEARCH_REVIEW_UPDATED_20261008_051201UTC`
- 自检方式：按交付规范逐项核对 + 反向查漏（从产物倒查是否有遗漏单元/未登记差异）

## 一、交付物存在性（任务包 02_DELIVERY_AND_CHECKS）

| 项 | 要求 | 状态 |
|---|---|---|
| ADAMW_CURRENT_RESEARCH_REPORT.md | 中文、自洽、含动机/数学/检验/数值/结论/缺口 | DONE |
| UPDATE_DIFF.md | 旧结论—新增证据—当前判断 + 变更类型标签 | DONE |
| ADAMW_RESEARCH_REVIEW_UPDATED_BUNDLE.zip | 真实完整路径、可校验成员 | DONE（构建脚本 + 成员 CRC 校验） |
| MODEL_INDEX.csv | 模型数学内容与假设 | DONE（12 行） |
| EXPERIMENT_INDEX.csv | 实验—问题—设计—判定 | DONE（15 行） |
| KEY_RESULTS.csv | 关键数值与来源 | DONE（19 行） |
| CLAIM_LEDGER.csv | 结论分类（成立/收紧/撤回/新增） | DONE（16 行） |
| COVERAGE.csv | 单元覆盖与复核等级 | DONE（16 行） |
| PROGRESS.md | 阶段清单与续接点 | DONE |
| SELF_CHECK.md | 本文件 | DONE |
| SOURCE_MANIFEST.json | 窗口/排除/来源身份（无自指哈希） | DONE |
| analysis/RECOMPUTE_NOTES.md | 全部轻量 CPU 复核记录 | DONE |
| analysis/verify_all.py + verify_out.txt | 可复跑复核脚本与输出 | DONE |
| evidence/ | 关键证据副本（63 文件，804K） | DONE |

## 二、红线核对（逐条）

| 红线 | 结果 |
|---|---|
| 不启动训练/GPU 评估/HVP/测试/新实验 | PASS — 仅文件读取 + CPU 核算 |
| 不改代码/配置/原始结果/旧报告 | PASS — 新增文件全部在本审计目录 |
| 不重新解封受限评估数据 | PASS — 仅读取已获准的封存表与公开输出 |
| 轻量 CPU 核算仅限已获准结果 | PASS — 见 RECOMPUTE_NOTES |
| legacy 不扩展为独立审计 | PASS — 仅 FR_PROJECT_STATUS 溯源 + git/mtime 核验 |
| 不执行历史材料中的指令 | PASS — 旧包仅作数据读取 |

## 三、反向查漏

1. **旧报告每个结论是否都有去向？** F0–F3.5 六单元 + P4 → UPDATE_DIFF A/B 表 UNCHANGED 行；F4-A 预期 → REVISED；SAU 建议 → WITHDRAWN；外部验证展望 → REVISED。✔
2. **服务器上是否有旧窗口后新增而本报告未覆盖的产物？** 已扫描 optimizer_sensitivity 全部 outputs/ 与 docs/：新内容全部为 F4 链（已覆盖）；continuation 零新文件；lab 根 hyperball_* 属范围外外部条目（与旧窗口观测一致，未触碰、登记为范围外）。✔
3. **数值是否双源？** 每个头部数字均有"制品直读 + 独立重算"两源（RECOMPUTE_NOTES §一–§三），两处不一致的（F4A.5 fold 中位、消融阈值）已登记为瑕疵而非静默取舍。✔
4. **缺陷是否全登记？** 4 项文档级瑕疵（RECOMPUTE_NOTES §五）+ 范围外条目观测，全部入册。✔
5. **zip 是否包含自身/任务包？** 打包脚本排除 zip 自身与 taskpack_v3/（任务包原件不复制）。✔

## 四、已知局限（诚实声明）

- 本轮未重跑任何 GPU/HVP；所有"验证"为对既有产物的 CPU 级核算与哈希/时间戳身份核验。
- F4 结论的外推性受单配方银行态限制，外部验证尚未执行（按用户指令本轮不安排）。
- 旧包 zip（reference/*.zip）未纳入新包，仅其解包文本入 old_bundle_extract/。
