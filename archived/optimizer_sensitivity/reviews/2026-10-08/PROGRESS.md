# PROGRESS — AdamW 研究复核（v3 任务包 / 第二轮审计）

- 任务根：`/home/b205/projects/training_dynamics_lab`（只读）
- 任务包：`audits/CODEX_ADAMW_REVIEW_TASKPACK_v3.zip`（已就地解包至本审计目录 `taskpack_v3/`）
- 产出目录：`audits/ADAMW_RESEARCH_REVIEW_UPDATED_20261008_051201UTC/`
- 历史对照包：`reference/ADAMW_RESEARCH_REVIEW_BUNDLE_previous.zip`（已解包至本目录 `old_bundle_extract/`；旧审计窗口 2026-09-30T06:11–06:54Z）
- 本轮窗口：2026-10-08 05:12 UTC 起（本地 CST 13:12）；时区服务器 CST (UTC+0800)
- 执行模式：按用户执行模式指令，全链路非交互批处理，一次完成至打包

## 阶段清单

- [x] 任务包 00–03 全部读取（v3 交付规范、任务定义、检查清单、历史上下文）
- [x] 旧审计包读取与基线建立（ADAMW_RESEARCH_REVIEW.md 全文、PROGRESS、BUNDLE_MANIFEST）
- [x] 清点：optimizer_sensitivity / continuation_adaptation / legacy_fluctuation_response（legacy 仅溯源）
- [x] 阅读：F4 链全部 prereg/report/erratum/synthesis/memo（F4A、F4A.1、F4A.5、F4B、F4B 事后、F4C、一致性审计、决策备忘、外部验证计划、论文建模框架）
- [x] 冻结核验：F0–F3.5 全部关键制品+文档 sha256 == 旧 manifest（11/11 一致）；P4 四件套哈希一致；legacy HEAD 6ac4942 且窗口后零研究写入
- [x] 证据复核：analysis/verify_all.py（输出 verify_out.txt），覆盖 P4 封存表、F4-A/F4A.1/F4A.5/F4B/F4B事后/F4C 全部头部数字，全部 PASS
- [x] 文档级瑕疵登记：4 项（见 RECOMPUTE_NOTES.md 第五节）
- [x] evidence/ 复制（63 文件，804K，保持原相对路径）
- [x] analysis/RECOMPUTE_NOTES.md
- [x] ADAMW_CURRENT_RESEARCH_REPORT.md / UPDATE_DIFF.md
- [x] MODEL_INDEX / EXPERIMENT_INDEX / KEY_RESULTS / CLAIM_LEDGER / COVERAGE
- [x] SELF_CHECK.md / SOURCE_MANIFEST.json
- [x] ADAMW_RESEARCH_REVIEW_UPDATED_BUNDLE.zip 生成与成员校验

## 新旧差异主线（详见 UPDATE_DIFF.md）

- 旧窗口结束时 F4-A 仅 17/74 单元完成 → 本轮 F4 链已全部完结（F4A → F4A.1 → F4A.5 → F4B → F4B 事后 → F4C → 一致性审计 → 决策备忘）
- continuation_adaptation：零变化，P4 STOP 冻结
- F0–F3.5 与 legacy：字节级不变

## 红线遵守

三研究目录零写入；未运行训练/GPU/HVP/测试/notebook；未解封受限评估数据；轻量 CPU 核算仅限已获准读取的既有结果；新增文件仅写入本审计目录；未执行历史材料中的任何指令。
