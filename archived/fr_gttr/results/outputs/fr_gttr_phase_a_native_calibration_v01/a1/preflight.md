# Phase A Preflight（a1 执行轮）— UNBLOCKED，smoke PASS

日期：2026-09-22。协议：FR_GTTR_PhaseA_Native_Target_Calibration_Claude_Code.md v1.0
+ FR_GTTR_Experiment_Roadmap_and_Next_Run_v1.md §3–§8。
上一轮 SCHEDULE_BLOCKED 记录保留于 ../preflight.md（未改动）。

## 1. 现场核查结论（全部通过）

| 项 | 结论 |
|---|---|
| 入口 runner | `research/src/fr/a0k_closed_loop_runner_v01.py`（per-micro `set_native_micro_lr → forward → backward`；`micro_idx%accum==0` unscale+clip+step） |
| 单步语义 | AdamW（betas=(0.9,0.999), eps=1e-8, wd=0.01），全局 L2 clip τ=1.0 post-unscale；1 optimizer step = 2 microbatches × 4 行 × 128 token slots |
| 参数组 | 唯一组（bank optimizer_signature + `assert_native_adamw` 双核实）；无 decay/no-decay 分组；master weights fp32、moments fp32、compute bf16 autocast |
| scheduler | `minimind/trainer/trainer_utils.py:40-41`；本轮明文尾部见 schedule_amendment.md（NATIVE_WITH_DECLARED_TAIL） |
| 正常/skip 路径 | 本配方无 grad-nan skip；每步 clip+step 必执行；finite 失败即停该轨迹并保留 |
| 数据 | D0 `pretrain_t2t_mini.jsonl` 1,270,238 行（sha256=6dd6716c…84ab3689，逐行重算）；e3c_subset == D0[0:4096]（cmp 恒等）；train=partition 3936 行（全 <4096） |
| 六个 checkpoint | seeds 2030/2031/2032 × step 46/82；真实 n0=46/82，global_micro=92/164，optimizer 内部 step 计数 = n0（逐文件核实） |

## 2. 数据门（dataset_contract.json 全量）

- 行 ID：train/probe/eval 3936+32+128 互斥且覆盖 rows[0:4096)；
  train 与 V_monitor rows[4096:8192)、V_final rows[8192:12288) 行级零相交。
- doc/source ID：D0 仅 `text` 字段，无 doc ID → DOCUMENT_INDEPENDENCE_UNVERIFIED。
- 精确重复：train/V_monitor/V_final 三集合内部与两两之间，**文本级与
  token 序列级（tokenizer 逐行重编码）重复均为 0**。
- V_monitor 已逐行复制到 a1/data/v_monitor_rows4096_8192.jsonl（sha256 登记），
  原 D0 未改动。训练采样只从 partition train_indices 取（stream 构造
  `batches_from_train_pool`），结构上排除 monitor/final。
- 本轮未运行 final 模型评价；V_final 只做身份/重叠登记。

## 3. Stream 注册

| cell (seed,start) | stream_id | 状态 |
|---|---|---|
| (2030,46) | 97000 | 无冲突 |
| (2030,82) | 97001 | 无冲突 |
| (2031,46) | 97002 | 无冲突 |
| (2031,82) | 97003 | 无冲突 |
| (2032,46) | 97004 | 无冲突 |
| (2032,82) | 97005 | 无冲突 |

既有全部注册 stream_seed 已扫描（outputs/**/*.json），无 97000..97005
的真实训练使用（命中均为哈希串偶然子串）。batch 种子由冻结 A0K 配方
`derive_stream_seed(stream_id, 404)` 派生。

## 4. CPU 测试

`research/tests/test_phase_a_native_calibration_v01.py`（13 项新）+
复用 test_native_schedule / test_stream_rng / test_full_state_roundtrip，
合计 17 项全部 OK。覆盖：tail 边界/无回弹/拒 horizon 重拉伸、128-vs-256
schedule 与 batch 前缀恒等、sum/count 聚合、radial/tangent 恒等+零范数
NA+tied 去重、EMA 系数、tracking 递推、跨步第二项条件均值非零、
hit 记录规则、row 排除与 counter 语义。

## 5. Smoke（16 真实 updates + 3 次全库 monitor）

- off 臂 8 步（无 monitor/无诊断）vs on 臂 8 步（h0/4/8 monitor，h=1 分层诊断）：
  - theta/m/v/step/lr **bitwise 相等**；
  - RNG（python/numpy/torch cpu/cuda）状态相等；
  - 数据游标相等（16 micros）。
- verdict=PASS（smoke_checks.json）。
- ETA：实测 monitor 3.28s/次、step 亚秒级、setup 45.4s 一次性；
  总 ETA ≈1506s（含 smoke），< 5400s cap（eta_estimate.json）。

## 6. 资源口径

计划 1552 updates（main 1536 + smoke 16），硬 cap 1664；monitor forward
计划 7296、cap 7800；总 forward 计划 ≈10400、cap 12000；墙钟 cap 5400s。
新增 gradient calls=0、HVP=0、V_final eval=0；训练 backward 照常发生并计数。
历史 18,730 段与 C1/C1L 账不改；本轮 actual 单独记录于 resource_log.jsonl。
