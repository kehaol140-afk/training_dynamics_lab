# 05 — 可复用资产清单

> 每项：exact path / 用途 / 依赖 / 是否经测试 / 能否不改直接复用 / 磁盘大小。
> 环境基线：conda env `fr_minimind`（/home/b205/miniforge3/envs/fr_minimind），`PYTHONPATH=research/src`，
> `CUDA_VISIBLE_DEVICES=0`，torch 2.11.0 + CUDA 13.0，2× RTX 5090 32GB。环境快照：`ENVIRONMENT_V07.txt`、`baseline_manifest/`（8 文件：conda_list/environment.yml/pip_freeze/nvidia_smi/torch_gpu/dataset_sha256/minimind_code_sha256）。

## A. 代码（research/，总 9.0M；Git 已 track 414 文件）

### A1. 核心冻结分析模块（research/src/fr/，68 .py + 1 .pyc）

| 资产 | 路径 | 用途 | 测试 | 直接复用 |
|---|---|---|---|---|
| **指标唯一定义源** | `research/src/fr/a0c_adamw_response_propagation_v01.py` | `_metric`/`relative_error`（L29/L36）；SELECTED_STATES/HORIZONS 常量 | 是（单测+多实验位级复验） | 是（import-only，禁重定义） |
| **AdamW 完整切线步进** | 同上 + a0e2 模块 | `adamw_tangent_step`（m/v 一阶矩、bias correction、eps、decoupled wd、clip Jacobian） | 是 | 是 |
| **clip JVP** | 同上 `clip_jvp`（L50–136） | 全局 L2 clip 的平滑 JVP：`c·q − radial·g` | 是（CPU FD 1e-6 级） | 是 |
| **theta_protocol（pyc-only！）** | `research/src/fr/theta_protocol.pyc`（6,980B） | `run_theta_branch` 等 replay 原语；源 .py 已丢失；Git 有例外规则强制 track | 行为等价已验（A0C/A0E2 位级 gap=0） | 是（每次跑前确认存在） |
| 伴随/生产入口 | `research/scripts/a0h15_production_entry_v01.py` | forward_only/adjoint_only/replay_only 三模式；fp32-HVP 伴随 | 是（同进程位级 gap=0 自校验） | 是 |
| LR 响应源项 | `research/src/fr/a0o_lr_response_v01.py` | e_lr=−adaptive(c)−wd·θ 纯 θ 通道推 + 冻结 δ=0.05 | 是 | 是（import-only） |
| G 线通用框架 | `research/src/fr/g1a_26m_scale_generalization_v01.py`（g1b/g2/g3 同构） | scale-only/数据/clip 干预全套：bank 构建、manifest、smoke、replay 验证、mechanism 分解、aggregate_metrics | 21–31 单测/实验 | 是（新泛化轴改 config 即可） |
| 冻结指标复用层 | g1a 模块的 `cell_relative_errors`/`aggregate_metrics` | 统一口径（G2 强制 import，无第三口径） | 是 | 是 |
| 统计工具 | `confirmatory_stats.py`(18K)、`cross_seed_stats.py`、`response_stats.py`、`fluctuation_stats.py`、`mapping_stats.py` | E 系全部统计 | 是 | 是 |
| RNG/流控制 | `rng.py`、`stream_rng.py`、`batch_stream.py`、`batch_manifest.py` | CRN/流种子/batch manifest（derive_stream_seed 配方） | 是 | 是 |
| replay 运行时 | `minimind_v03_runtime.py`、`stage_c_runtime.py`、`full_state.py`、`perturbation.py`、`native_adamw_guard.py`、`native_schedule.py` | E0/E1/E2 全套 replay 原语 | 位级验证 | 是 |
| 其余实验模块 | a0d/a0e2/a0f/a0f2/a0g/a0h/a0j/a0k/a0l/a0n/a0p/a0q/a0s/a0t*/a0u/b0*/c0/c1/cb1/d1/f0/phase_* /r1/r2b | 各实验库逻辑（全部冻结只读） | 各自单测 | 参考实现，勿改 |

### A2. Runner 脚本（research/scripts/，~150 个）

- 模式惯例：`<exp>_{run,analyze,audit_pack}_v01.py` 三件套 + contract 冻结 + live_manifest + SHA256SUMS。可直接作为新实验模板。
- 入口文档：`research/scripts/README_SCRIPTS_V03..V06.md`（含中文版，注意一份文件名编码异常）。
- 最新前沿：`f0_terminal_rule_run_v01.py`、`f1_closure_run_v01.py`、`phase_b_fd_diagnostic_v01.py`（79K，最大脚本）。

### A3. 测试（research/tests/，57 个 test_*.py）

- 覆盖全部主要模块；含 `test_theta_protocol.py`、`test_v07_prereg_lock.py`、`test_v07_recovery_extractor.py`。
- 运行方式：`PYTHONPATH=research/src python -m unittest discover -s research/tests -p "test_<mod>.py" -v`。

### A4. 配置（research/configs/，20 个 JSON + README V03–V06）

- 关键：`e3c_confirmatory_v07.json`（bank 生成）、`g1a/g1b/g2/g3_v01.json`（泛化线模板）、`e0/e1/e3/e3r/e4b/stage_c` 系列。
- configs 通过 input_hashes 被冻结引用，**勿改名移动**。

### A5. Adapter

- `research/adapters/minimind_adapter_v02.py` + template：MiniMind 模型接入 FR replay 的适配层（新项目对接 MiniMind 的起点）。

## B. 数据

| 资产 | 路径 | 大小 | 用途 | 复用 |
|---|---|---|---|---|
| **v07 训练池 D0** | `outputs/e3_confirmatory/v07/e3c_subset.jsonl` | ~8M | 4096 条训练池（SHA 1e657af9…b0f4），一切 anchor 实验数据源 | 只读直取 |
| D1 池（G2） | `outputs/generalization_g2_data_distribution_v01/data/d1_pool.jsonl` | ~4M | 4096 条 pretrain_hq 采样 + `partition_g2_d1.json`（32/128/3936） | 只读 |
| D1 原始语料 | `outputs/generalization_g2_data_distribution_v01/data/pretrain_hq.jsonl` | **1.6G** | MiniMind 官方 pretrain_hq（HF 删除前 revision 3a9bbe61，SHA 82b19375…=LFS oid） | 可归档（有 SHA 可重取） |
| MiniMind 语料 | `minimind/dataset/pretrain_t2t_mini.jsonl` | **1.2G** | 1,270,238 行上游语料 | 是 |
| tokenizer | `minimind/model/tokenizer.json`（vocab 6400） | ~1M | 全项目唯一 tokenizer | 是 |
| partition/schema | v07 `partition_v07.json`、`PREREG_LOCK.json` | 小 | probe/eval/train 划分 + 预注册锁 | 只读 |
| dataset SHA | `DATASET_SHA256.txt`、`baseline_manifest/dataset_sha256.txt` | 小 | 数据完整性锚点 | 只读 |

## C. 模型 / checkpoint

| 资产 | 路径 | 大小 | 内容 | 复用 |
|---|---|---|---|---|
| **v07 56 态 bank（唯一认证输入源）** | `outputs/e3_confirmatory/v07/seed_203{0..7}/{states,q,e1_stream_manifests,e2_stream_manifests}/` | **4.6G** | 每 seed 7×(state .pt 含 optimizer 态 + q .pt + E1/E2 manifest)；`state_bank_report.json`、`multiseed_theta_summary.json` | 只读；**绝不删除**；A0 系一切实验唯一输入 |
| G1a 26M bank | `outputs/generalization_g1a_26m_v01/bank/`（9 state .pt）+ theta q | ~3.7G 目录 | 26M 3 seed × 3 stage 完整态 | 可归档（SHA256SUMS 有哈希；重训可再生成但改变 seed 语义） |
| G1b 100M bank | `outputs/generalization_g1b_100m_v01/bank/`（9×1.2G） | ~11G | 100M 完整态 | 归档候选（同上） |
| G2/G3 bank | g2/g3 outputs 内 | 含在 5.3G/3.7G 内 | 26M D1 态 / no-clip 态 | 同上 |
| A0K 事件快照 | `outputs/a0k_fresh_response_closed_loop_v01/runs/`（72 事件 × 完整 .pt 快照） | **14G** | 闭环每事件 model+optimizer+scaler+RNG 快照 | 归档候选（counterfactual replay 可再生成） |
| A0N 运行态 | `outputs/a0n_phase_scheduler_pilot_v01/` | **7.5G** | phase pilot 全臂运行 | 归档候选 |
| A0O rollout cells | `outputs/a0o_lr_response_validation_v01/` | **4.5G** | LR 响应 cells（A0S/A0T 冻结输入！） | 保留（A0S/A0T 依赖） |
| R2B 轨迹 | `outputs/fr_gttr_r2b_solved_update_bridge_v01/` | 4.5G | 36 条 256 步轨迹（R1/CB1 输入） | 保留至 GTTR 线关闭 |
| E3 replication | `outputs/e3_replication/` | 1.8G | v06 4-seed 数据 | 归档候选 |
| a0_fr_audit | `outputs/a0_fr_audit/`（含 mv_theory_final/gradient_g.pt 1.1G） | 1.2G | A0 系冻结审计（**结论源**） | **绝不删除**（gradient_g.pt 除外可归档） |
| 根 checkpoints/ 与 data/ | `checkpoints/{native,full_state}/`、`data/{eval,probe,raw}/` | ~0 | 空脚手架 | 无内容 |

## D. 实验基础设施（跨资产）

1. **CRN 三分支 + pulse-before-clip 协议**（`perturbation.py` + theta_protocol.pyc）：全部因果实验的核心语义。
2. **身份门 S1/S2 惯例**：与 v07 bank 逐位比对 + ghost no-op 不可见性检验——任何新训练臂的入场门。
3. **冻结纪律链**：contract.json + live_manifest（seq 先于结果）+ SHA256SUMS + audit_pack tar（manifest 不含自身）。
4. **位级复验链**：A0E2↔A0C gap=0；A0F2↔A0F gap=0；G1a/G1b/G2/G3 replay 36/36 位级；H1 传播 gap 2.3e-9。
5. **GPU 预算记账**：budget_log/gpu_budget_log（B0/C0/C1 系 18,000 update 上限账本 17,994/18,000）。
6. **实验↔代码↔config 映射表**：`docs/project_inventory/EXPERIMENT_CODE_MAP.md`（09-12 版，覆盖 E/A0 早期）。
7. **审计包模板**：`a0{g,i,j,k,l,n,o,p,q,s,t,u}*_audit.tar.gz`、G 线审计包（SHA 记录于 INDEX/STATUS）。
8. **独立 CPU 校验包**：`tools/gttr_f0_ic_v1/`（START_HERE + verify_reference.py + independent_cpu_checks.json）。

## E. 文档资产

- `docs/FR_PROJECT_STATUS.md`（1443 行，§1–§49 全量状态）+ `docs/FR_EXPERIMENT_INDEX.md`（实验总索引）
- `docs/FR_GENERALIZATION_STATUS.md` / `FR_GENERALIZATION_HANDOFF.md`（G 线）
- `docs/FR_RESEARCH_RETROSPECTIVE_A0G_A0U.md`（401 行英文学术回溯，Part1-3）
- `FR_REPORT_AUDIT_BUNDLE_20260912/REPORT_FACTS+RED_FLAGS`（A0 系论文数字措辞规则）
- 根目录 ~25 份协议 MD（RUN_ORDER V03–V07、GTTR 系列、PhaseB、F0/CALFIX 等）
- `docs/superpowers/`（6 plans + 2 specs，A0C 时代设计文档）
- 研究手册：`docs/Transformer_AdamW_Fluctuation_Response_ResearchManual_v2.pdf` + `ResearchPlan_AI_v2.md`

## F. 大小与归档速览（详见 08_DISK_AUDIT.md）

- 总计 outputs 71G；**绝不删除**：v07 bank 4.6G、a0_fr_audit 结论件、全部 SHA256SUMS/contract/manifest/审计包（合计 <2G 的"结论层"）。
- 可归档（有 SHA 或可再生成）：G1b bank 11G、A0K 快照 14G、A0N 7.5G、pretrain_hq 1.6G、E3 replication 1.8G、mv gradient_g.pt 1.1G——约 37G。
