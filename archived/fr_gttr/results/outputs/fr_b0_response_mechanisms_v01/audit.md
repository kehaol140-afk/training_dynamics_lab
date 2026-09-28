# FR B0 机制验证 — 阶段 A 仓库审查产物

日期：2026-09-20。执行：Claude Code（Kimi K2.7 后端）。本文件在查看任何 B0 新结果之前写入。

## 1. 实际调用的接口（全部现场确认，未编造）

| 用途 | 接口 | 位置 |
|---|---|---|
| runtime 构造 | `fr.a0k_closed_loop_runner_v01.build_live_runtime(seed, device)` | research/src/fr/a0k_closed_loop_runner_v01.py:42 |
| 事件上下文 | `fr.a0k_closed_loop_runner_v01.make_event_ctx(rt, event, seed, step, snapshot_path)` | 同文件 :94 |
| 完整状态保存/恢复 | `fr.full_state.save_full_training_state` / `load_full_training_state`（model 原始 dtype + AdamW state + scaler + 四类 RNG + metadata，`fr_state_version=0.2`） | research/src/fr/full_state.py |
| 冻结单步原语 | `pe._baseline_no_step`（forward/backward+unscale，含 per-micro 原生 LR 写入 `set_native_micro_lr` 与 pre-step 快照）、`pe._clip_and_step`、`pe._step_once`、`pe._restore_and_seed` | research/scripts/a0h15_production_entry_v01.py:211,204,197,233 |
| persistent LR rollout（冻结语义参照） | `fr.a0q_response_control_map_v01.run_channel_rollout(rt, ctx, stream, H, "lr", u, counters)`：`g["lr"]=snap["lr"]*e^u` 每个 optimizer step 覆盖（u=0 不触碰→位级 native） | research/src/fr/a0q_response_control_map_v01.py:131 |
| A0O 解析响应（只读参照） | `fr.a0o_lr_response_v01.run_lr_response` / `DELTA=0.05` / `lr_pulse` | research/src/fr/a0o_lr_response_v01.py |
| 嵌套中途评估模板 | `fr.a0u_response_schedule_discovery_v01.run_schedule_rollout`（per-step multiplier + checkpoint 处 RNG 包裹的 `_val_loss_only`） | research/src/fr/a0u_response_schedule_discovery_v01.py:166 |
| 评估读数 | `a0g_decision_utility_v01._val_loss_only(torch, model, batch_tensors, autocast_ctx)`（eval-mode forward，mean loss，结束恢复 train） | research/scripts/a0g_decision_utility_v01.py:252 |
| eval 张量 | `rt["val_score_tensors"]` / `rt["val_eval_tensors"]`（A0G 冻结 validation_split 上 `_stack_all`） | research/src/fr/a0k_closed_loop_runner_v01.py:86-87 |
| future stream | `fr.stream_rng.derive_stream_seed(stream_seed, 404)`（A0K 冻结配方）+ `fr.stage_c_runtime.batches_from_train_pool(train_indices, bs, batch_seed, n_micro)` | research/src/fr/stream_rng.py:18, research/src/fr/stage_c_runtime.py:8 |
| 训练 RNG 显式流 | `seed_future_stream(stream_seed)`（python/np/torch-cpu/torch-cuda 四类） | research/src/fr/stream_rng.py:27 |
| RNG 捕获/恢复 | `fr.rng.capture_rng_state` / `restore_rng_state` | research/src/fr/rng.py |
| AdamW  native 守卫 | `fr.native_adamw_guard.assert_native_adamw`（不修改 AdamW） | research/src/fr/native_adamw_guard.py |

B0 新增代码只含：一个 rollout adapter（`b0_run_v01.py`，逐字调用上述冻结原语；multiplier 语义与 a0u.run_schedule_rollout 的 `mult_fn(h)` 同型）、CPU 理论测试、纯统计/绘图脚本。**未重写、未简化 AdamW**；optimizer 步骤一律经 `pe._clip_and_step`（内部 `scaler.step(optimizer)`）。

## 2. 可用 seed / checkpoint / future stream / 数据顺序 / 验证划分

- v0.7 bank：`outputs/e3_confirmatory/v07/`，8 训练 seed（2030–2037）× 7 step（46–82）= 56 态；`seed_*/states/state_step_*.pt`（逐一确认 2030 的 7 个文件存在）。
- B0 主样本（固定排序取前 3 个可用训练 seed）：**seeds {2030, 2031, 2032} × 绝对 optimizer step {46, 82}**，4 条新 future streams **{7500, 7501, 7502, 7503}**（前 2 dev、后 2 audit）。stream seed 7500–7503 在 research/ 与冻结 outputs 中 grep 确认**未被任何旧实验使用**（7110–7113/7200–7207/7300–7301/7400–7401 已占用）。
- 数据顺序：stream 内容 = 在冻结 train pool（`partition_v07.json` 的 `train_indices`，3936 条）内按 `derive_stream_seed(x,404)` 置换取 batch；与旧评估流不重合（不同 stream_seed → 不同 batch 序列；旧评估从未用 7500–7503）。
- 验证划分：A0G 冻结 `outputs/fr_decision_utility_v01/validation_split.json`（v_score / v_eval 两子集，eval_batch_size=8）。B0 主读数 = **V_eval**（协议 Stage E "固定 V_eval"），V_score 作为与 A0O/A0T 连续的次要读数同时记录。
- 既有 2 个 A0T 反转 cell（冻结 `outputs/a0t_response_lr_control_pilot_v01/runs/cells.jsonl` 中 `aliased=false`）：**(2030, 46, x7110) chi=+2.7390e-2、(2030, 64, x7110) chi=+1.9963e-3** → 仅作独立 stress-case 复核（旧流 7110），不混入主样本统计。

## 3. 精度 / backend / 累积 / 有效 batch / token / clip / wd / LR schedule

- 精度：**bf16 autocast**（冻结 v07 配置 `research/configs/e3c_confirmatory_v07.json`，`dtype=bfloat16`，GradScaler 存在但 bf16 下禁用——a0k 运行时断言）。与 A0O/A0T/A0U 完全同一配置；不做 FP32/BF16 混比。
- attention backend： MiniMind 默认（A0O/A0T 同路径；本轮无 HVP/adjoint，math-SDPA caveat 不适用）。
- 梯度累积：accumulation_steps=2；batch_size=4 seqs → **有效 batch = 8 seqs = 1024 tokens/optimizer step**（seq_len=128）。
- clip：grad_clip=1.0（冻结）；wd：AdamW weight_decay=0.01（bank 原生值，A0Q 已断言）。
- LR schedule：MiniMind 原生 `get_lr(micro_step, total=256, base=5e-4) = base*(0.1+0.45*(1+cos(pi*micro/total)))`，**每 micro-step 写入** `optimizer.param_groups[*].lr`；optimizer step 消费的是 accumulation 窗口最后一个 micro-step 写入值（`snap["lr"]`，research/src/fr/native_schedule.py）。训练位置：bank step s → micro 基线 2s；64-step rollout 到 micro 2s+128（step 82 时达 292 > 256，get_lr 仍 > 0 恒正（≥0.1·base），A0T 同情形已验证可运行，语义不动）。
- 模型：4,983,296 unique params；state_dict 46 keys；buffer 仅 freqs_cos/freqs_sin（RoPE，训练不变）；embed 与 lm_head 绑定（state_dict 只出现一次）→ 参数平均安全。

## 4. A0T horizon 单位

A0T runner 内 `assert accum * max(H_GRID) == N_MICRO`（2×64=128 micro）且每条臂从 checkpoint 逐步 rollout → **H 单位 = optimizer steps**（H=4 ⇒ 8 micro-batches）。B0 全轮统一 optimizer step 口径；microbatch 数 = 2×H。

## 5. GPU 真实情况（运行期实测，非记忆）

- 实测 2× RTX 5090 32GB。会话开始时：GPU0 19.5GB used / 50% util；GPU1 21.9GB used / 67% util，均有外部任务（revision_experiments 训练进程，PID 679532/694649）。**不关闭、不干扰外部进程**；B0 5M 模型峰值 ~2–3GB，选空余量 ≥8GB 的卡，缺省 cuda:1（A0O/A0T/A0U 同），运行时探测并记录。共享争用抬高墙钟不影响结果（CRN 配对同场执行）。

## 6. 原始结果复核状态

- **已复核原始记录**（本审查直接读取）：A0T `runs/cells.jsonl`（36 cells，2 个非别名 cell 身份与数值如上）；v07 bank 目录结构；冻结原语源码（a0o/a0q/a0u/a0k/full_state/pe）；v07 配置。
- **仅 retrospective 摘要、未在本轮重开原始表的**：A0O stage1 gate 数字、A0Q J 矩阵、A0S/A0T-extra/A0U 聚合指标（STATUS/INDEX 转述；B0 不依赖其精确数值做判定，仅作背景）。
- **缺失**：A0Q 未持久化 rollout 损失（A0S 已补测，B0 不需要）；theta_protocol.py 源文件丢失（以冻结字节码 .pyc 恢复，STATUS §13 已确认行为等价）——**B0 运行前已确认 research/src/fr/theta_protocol.pyc 存在**。

## 目录映射（协议建议 → 仓库实际约定）

协议建议 `experiments/fr_b0_response_mechanisms/` + `outputs/fr_b0_response_mechanisms_v01/`。仓库约定为 `research/src/fr/`（核心模块）+ `research/scripts/`（runner）+ `outputs/`（制品）。映射：

- 核心纯函数模块：`research/src/fr/b0_response_mechanisms_v01.py`
- runner/测试/汇总：`research/scripts/b0_{theory_tests,run,analyze}_v01.py`
- 单元测试：`research/tests/test_b0_response_mechanisms_v01.py`
- 全部制品：`outputs/fr_b0_response_mechanisms_v01/`

## 7. Smoke 阶段检查定义修正（2026-09-20，任何主实验结果之前）

- 第一次 smoke（`runs/smoke_checks_attempt1.json`）中 `release_m_v_retained_H16=false`：初版检查把"release 后 H=16 处 m/v 与 persistent 臂逐位相等"当作不变量，这在物理上不可能成立——release 后 theta 路径即分叉 ⇒ 梯度不同 ⇒ 自第 5 步起 m/v 必然分叉。
- 修正（写入 config.json `release_semantics_correction_note`，原始失败证据保留于 `runs/smoke_checks_attempt1.json`）：
  1. 有效的逐位 m/v 恒等式位于 **H=4**（新增 `pulse4_eq_persistent_H4_optstate`，含 m/v/step）；
  2. H=16 处改为两个可检验性质：`(a) release_m_v_diverged_from_persistent_H16`（脉冲臂 m/v 与持续臂分叉，"释放 ≠ 持续干预的延续"）与 `(b) release_m_v_retained_H16`（脉冲臂 m/v 从自身 H=4 状态连续演化：step 计数 +12、exp_avg 已移动、无 None/重置）。
- 修正发生在 smoke 阶段（该阶段协议用途即拦截此类错误），先于 main/stress 任何结果；非结果驱动的标准放松。
