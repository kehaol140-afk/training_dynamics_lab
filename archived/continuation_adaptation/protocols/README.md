# continuation_adaptation

**Training Continuation under Changed Compute Budget: Optimizer State Adaptation for AdamW**

科学证伪实验（第一轮）。研究问题：Transformer checkpoint 在改变训练预算/优化器
超参后续训时，继承的 AdamW state (m, v, step) 是否仍然适配？简单 state 适配
能否优于认真调参的普通续训 baseline？

**不假设假设成立。淘汰实验：失败即停止该方向，不追加复杂方法救场。**

## 当前阶段

P0–P3 完成；P4 协议已冻结（`docs/P4_PROTOCOL.md` + `configs/p4_full.json`，
评审 `docs/P4_PROTOCOL_REVIEW.md`）。**P4 执行基础设施已实现并通过 33 项测试
（runner / selection / statistics / sealed-eval / analyze 脚本）。P4 GPU 矩阵
未启动，等待批准后按 Stage 0→1a→1b→1c→2→3 执行。**

P4 工程件（协议 §10 预登记项，均已实现、未运行）：
```
src/p4_runner.py           单 cell 单臂 runner（预算表 / 压缩 schedule / 乘子 /
                            V_monitor / final_state.pt；不接触 V_final）
src/p4_selection.py         Stage 1b 选择规则（冻结：并列/边界单次扩张/台账）
src/p4_statistics.py        Stage 3 统计纯函数（seed 聚类置换 / bootstrap / Holm）
scripts/p4_stage1.py        Stage 1 编排器（grid / bcd / status，可断点续跑）
scripts/p4_sealed_eval.py   Stage 2 封存评测（SEAL_LOCK / V_final 一次 / 不可覆盖）
scripts/p4_analyze.py       Stage 3 单遍分析（裁决 CONTINUE/STOP）
```

## 目录

```
docs/PROJECT_CONTEXT.md      项目上下文（审计 + 方向定义，自 lab root 复制）
docs/EXPERIMENT_PROTOCOL.md  实验协议（smoke 冻结）
docs/P4_PROTOCOL.md          P4 大规模淘汰实验协议（冻结）
docs/P4_PROTOCOL_REVIEW.md   P4 协议冻结评审报告
docs/LEGACY_PROVENANCE.md    vendored 模块 SHA256 登记表
configs/smoke_v01.json       smoke 冻结配置
configs/p4_full.json         P4 冻结配置（56 cells 含逐文件 SHA256）
src/vendored_fr/             legacy 冻结基础设施 verbatim 副本（12 模块，勿改）
src/optimizer_utils.py       betas 修改 / moment 清零 / 只读统计
src/state_adapter.py         实验臂唯一定义（A/B/C1/C2/D）与施加验证
src/evaluation.py            validation NLL 薄封装（vendored eval_observables）
src/continuation_runner.py   单 run 续训 runner（原生时序复刻）
scripts/run_smoke_v01.py     P3 smoke 驱动
tests/                       三个必需测试（roundtrip / beta / CRN）
outputs/                     运行产物
```

## 运行

环境：conda `fr_minimind`（torch 2.11.0 + CUDA，Python 3.11）。
**必须单卡可见**（C0 的 CUDA RNG 状态按 1 设备保存）：`CUDA_VISIBLE_DEVICES=<gpu>`。

```bash
cd continuation_adaptation
export CUDA_VISIBLE_DEVICES=1   # 选一张空闲卡

# 测试（CPU 快速 + GPU 验证）
python -m unittest discover -s tests -v
FR_RUN_CUDA_TESTS=1 python -m unittest discover -s tests -v
FR_TEST_DEVICE=cuda:0 python -m unittest tests.test_crn_reproducibility -v

# smoke（约几分钟）
python scripts/run_smoke_v01.py --device cuda:0
```

## 红线

- 冻结区 `legacy_fluctuation_response/`、`minimind_base/`、`optimizer_sensitivity/` 只读；
- 不引入新 optimizer / GTTR / hypergradient / HVP / 二阶求解 / controller / 理论机制；
- 本阶段只有 checkpoint continuation + optimizer state manipulation；
- 口径先于结果冻结（`configs/smoke_v01.json` + 协议文档），不为过关改阈值；
- smoke 只报告实现状态与可复现性，不解读性能。
