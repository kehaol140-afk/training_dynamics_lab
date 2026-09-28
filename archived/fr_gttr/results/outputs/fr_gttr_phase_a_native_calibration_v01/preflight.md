# Phase A Preflight — SCHEDULE_BLOCKED at §4 case C

日期：2026-09-22。协议：FR_GTTR_PhaseA_Native_Target_Calibration_Claude_Code.md v1.0。
本轮 GPU main / smoke / monitor evaluation 均未执行；仅完成 §1 现场核查与 §4 schedule 判定。

## 1. 现场核查结论（已完成）

| 项 | 结论 |
|---|---|
| 入口 runner | [research/src/fr/a0k_closed_loop_runner_v01.py](../../../research/src/fr/a0k_closed_loop_runner_v01.py)（per-micro `set_native_micro_lr → forward → backward`；`micro_idx%accum==0` 处 unscale+step） |
| 单步语义 | AdamW（betas=(0.9,0.999), eps=1e-8, wd=0.01），全局 L2 clip τ=1.0 post-unscale；1 optimizer step = 2 microbatches = 8×128 = 1024 tokens |
| scheduler | [minimind/trainer/trainer_utils.py:40-41](../../../minimind/trainer/trainer_utils.py#L40-L41) `get_lr(current_step,total_steps,lr)=lr*(0.1+0.45*(1+cos(pi*current_step/total_steps)))` |
| 计划总量 | `planned_total_optimizer_steps=128`（[e3c_confirmatory_v07.json](../../../research/configs/e3c_confirmatory_v07.json)）→ micro 总量 256 |
| 数据 | D0 `pretrain_t2t_mini.jsonl` 1,270,238 行（sha256=6dd6716c…）；train=rows[0:4096] 内 3936+32+128；V_monitor=rows[4096:8192]、V_final=rows[8192:12288] 均未触碰（行级零重叠由构造保证；无 doc ID → DOCUMENT_INDEPENDENCE_UNVERIFIED） |
| 六个 checkpoint | bank anchors t46/t82 × seeds 2030/2031/2032，n0=46/82（真实累计 optimizer steps） |
| 预算 | 计划 1552 updates，硬 cap 1664；本报告消耗 0 |

## 2. §4 schedule 判定：case C → SCHEDULE_BLOCKED

判定顺序（协议 §4）：

- **A 不成立**：scheduler 无 horizon-independent 延长；代码、全部 config、已批准文档（CLOSEOUT / TARGET_ACTION_BRIDGE_SPEC / NEXT_VALIDATION_SPEC）中均无 terminal tail 明文。
- **B 不成立**：`0.1*lr=5e-5` 是有限 cosine 的**最小值**而非声明的正值 terminal floor + 固定尾部约定；不存在"到 floor 后保持"的文档或代码。
- **C 成立**：有限 cosine 定义；越界后公式回升/周期循环（见下表）；存在多个**冲突的隐含约定**（详见 §3）；无任何明文延长语义。

### 2.1 越界回升证据（anchor 46，base_lr=5e-4，total_micro=256）

| h | global_micro | 消费 LR | 备注 |
|---|---|---|---|
| 0 | 92 | 3.712e-4 | |
| 64 | 220 | 7.160e-5 | 接近 floor |
| 82 | 256 | **5.000e-5** | **floor 最小值点** |
| 128 | 348 | 1.788e-4 | **已回升 3.6×** |
| 192 | 476 | 4.784e-4 | 接近峰值 |
| 255 | 602 | 3.762e-4 | 仍在高位 |

完整 6 cell × h={0,1,32,64,128,192,255,256} 的 LR 序列见 [schedule_candidates.csv](schedule_candidates.csv)。
若按 prior runner 的隐含约定（`get_lr(gm,256,5e-4)` 逐字延续），256 步 continuation 会**先降到 floor 再回升约 10 倍**——这不是任何已声明的 schedule 语义。

### 2.2 冲突的隐含约定

| 约定 | 来源 | 语义 |
|---|---|---|
| total=256 逐字延续 | [c0_run_v01.py:202-204](../../../research/scripts/c0_run_v01.py)、[c1_late_run_v01.py:199](../../../research/scripts/c1_late_run_v01.py)、[a0k_closed_loop_runner_v01.py:101](../../../research/src/fr/a0k_closed_loop_runner_v01.py)（planning 值）、[native_future_lrs](../../../research/src/fr/a0_fr_diagnostics.py#L503) | 越界后随 cosine 回升/周期，无声明 |
| total=128 终点语义 | e3c_confirmatory_v07.json `planned_total_optimizer_steps=128` | 终点 LR=5e-5，之后未定义 |
| TARGET_ACTION_BRIDGE_SPEC 256 步方案 | 已批准设计文档 | 只批准预算/观测终点，**未声明 schedule 尾部语义** |

无任何文档把其中一种上升为"明确授权的 continuation schedule"。按协议 §4 C，本文件不授权发明新 tail 救实验。

## 3. 阻塞处置（按协议）

已交付：config 候选（本目录 config.json）、schedule 表（schedule_candidates.csv）、本 preflight。
**未执行**：GPU smoke、GPU main、monitor 评价、CPU T1–T8（协议要求它们"正式训练前"做；schedule 阻塞在更前一层）。

## 4. 一个需要用户决定的问题

本轮 continuation 的 schedule 尾部语义请选择其一（或给出明文授权文件路径）：

1. **沿用隐含约定 verbatim**：`eta(gm)=get_lr(gm, 256, 5e-5? no—5e-4)`，即接受先降至 5e-5 再回升至 ~4.8e-4 的 cosine 回弹（标 IMPLICIT_UNDECLARED，protocol §4 未授权此选项，仅作为问题呈现）；
2. **声明 tail = 冻结 LR 于 5e-5（=0.1·base）**：NATIVE_WITH_DECLARED_TAIL，需用户明文确认该 floor 即约定尾部；
3. **声明其他 tail**（如维持 anchor 处当前 LR 不变）：需用户明文给出公式与来源；
4. **维持 BLOCKED**：Phase A 不运行 GPU 部分。

在用户以明文批准其中一项之前，按协议 §4 C，不运行 GPU main、不 smoke、不改写 128→256、不 hold peak、不重 warmup。

## 5. 资源消耗

model forward 0 / backward 0 / optimizer updates 0 / HVP 0 / V_final eval 0 / GPU 秒 0。
