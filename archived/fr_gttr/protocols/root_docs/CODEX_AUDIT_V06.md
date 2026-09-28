# CODEX_AUDIT_V06

审计日期：2026-09-02

范围：当前项目根目录 `/home/b205/projects/fluctuation_response` 下的 `minimind/`、当前安装的 `research/`、`RUN_ORDER_V06.md`、`MANUAL_REVIEW_V06.md`、`research/configs/e3r_cross_seed_v06.json`，以及已有 `outputs/e3_replication/v06/` 产物。历史 `_fr_*_incoming/` 和 `_fr_framework_backups/` 仅作为非主路径背景，不作为当前 v0.6 主实现判定对象。

执行约束：只读审计源码和既有输出；未修复代码；未运行正式 GPU 实验；未运行会产生 `__pycache__` 或实验产物的测试/脚本。本文档是本次审计按要求写出的唯一新文件。

## A. MiniMind 原生训练语义

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:27`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:31`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:35`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:38`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:40`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:42`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:43`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:44`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:46`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:47`
- `/home/b205/projects/fluctuation_response/minimind/trainer/train_pretrain.py:49`
- `/home/b205/projects/fluctuation_response/research/src/fr/native_schedule.py:18`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:55`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:61`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:126`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:128`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:137`

证据：
- MiniMind 原生 `train_pretrain.py` 在每个 loader step 内先计算并写入 `get_lr`，随后进入 `autocast_ctx` forward，loss 除以 `accumulation_steps` 后 backward；在 accumulation boundary 执行 `scaler.unscale_ -> clip_grad_norm_ -> scaler.step -> scaler.update -> optimizer.zero_grad(set_to_none=True)`。
- v0.6 state-bank 主路径在每个 micro-batch 开始前调用 `set_native_micro_lr()`，随后 forward/backward，并在 optimizer boundary 执行 unscale/clip/native AdamW step/update/zero_grad。
- v0.6 paired branch 统一走 `theta_protocol.run_theta_branch()`，其中每个 future micro-batch 都按 `global_micro_step + future_offset` 写入 LR，符合 MiniMind 原生 micro-batch LR 时序。

风险：
- 旧 standalone v0.4 `e1_formal_v04.py:191-199` 和 `e2_epsilon_linearity_v04.py:118-126` 在 optimizer step 后为下一步写一次 LR，而不是每个 micro-batch forward 前写 LR；它们不在 `RUN_ORDER_V06.md:4-8` 的 v0.6 GPU 主路径中，但若被当作当前正式 E1/E2 使用，会偏离协议。
- `scientific_code_preflight_v06.py:31-56` 主要用字符串顺序门禁，能拦常见结构错误，但不能替代本次人工语义审计。

## B. full-state replay

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:33`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:67`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:70`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:71`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:72`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:73`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:79`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:121`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:125`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:127`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:129`
- `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py:132`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:32`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:36`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:42`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:44`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:62`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:66`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:70`
- `/home/b205/projects/fluctuation_response/research/src/fr/rng.py:80`
- `/home/b205/projects/fluctuation_response/research/tests/test_full_state_roundtrip.py:13`
- `/home/b205/projects/fluctuation_response/research/tests/test_rng_roundtrip.py:10`

证据：
- `_clone_to_cpu()` 对 tensor 做 `detach().clone().cpu()`，没有 half/bfloat16 强制转换；`capture_full_training_state()` 保存 raw model state、optimizer state、scaler state、metadata，并在同一逻辑点捕获 RNG。
- `rng.py` 捕获 Python random、NumPy、Torch CPU、Torch CUDA all；恢复时按 Python/NumPy/Torch CPU/CUDA 顺序恢复，并检查 CUDA 可用性与设备数量。
- `load_full_training_state()` 明确先 `torch.load`，再恢复 model、optimizer、scaler，最后恢复 RNG，避免 load 过程消耗 RNG 后污染 branch 起点。
- CPU 单元测试覆盖了 model/optimizer/RNG roundtrip；RNG 测试文件明确说明当前单元测试只验证 Python/NumPy/Torch CPU，CUDA RNG 留到服务器接线阶段。

风险：
- 本次遵守“不运行正式 GPU 实验”，没有动态验证 CUDA RNG roundtrip；实现层面有 CUDA capture/restore，但运行证据只来自既有代码与非 GPU 单测说明。
- DataLoader/Sampler/worker RNG 未由 `full_state.py` 独立保存。v0.6 主路径通过显式 batch manifest 和 `stack_batch()` 规避 DataLoader worker 状态，但未来若重新引入多 worker DataLoader，当前 full-state 工具本身不足以覆盖该状态。
- data position 由调用方 metadata、`global_micro_step`、partition/config 和外部 state-bank report 共同表达，不是 `full_state.py` 强 schema 字段。

## C. probe/eval 数据隔离

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:20`
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:27`
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:28`
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:29`
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:30`
- `/home/b205/projects/fluctuation_response/research/src/fr/pilot_partition.py:41`
- `/home/b205/projects/fluctuation_response/research/src/fr/stage_c_runtime.py:8`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:55`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:117`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:119`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:169`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:171`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2026/state_bank_report.json:15`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2027/state_bank_report.json:15`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2028/state_bank_report.json:15`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2029/state_bank_report.json:15`

证据：
- `build_partition()` 用固定 permutation 切出 `probe_indices`、`eval_indices`、`train_indices`，并显式检查三者互斥后计算 partition hash。
- `batches_from_train_pool()` 只接收 `train_indices` 并在其中生成 future batches。
- v0.6 multiseed state bank 先构造/加载固定 partition，然后每个 training seed 只从 `partition['train_indices']` 取训练 batch，并在结束时检查 `probe_overlap` 和 `eval_overlap`，任一非零直接报错。
- 既有四个 seed 的 state-bank report 均显示 `probe_overlap: 0`、`eval_overlap: 0`。

风险：
- 隔离单位是 4096 行 subset 的行索引；代码未做跨行文本重复/近重复检测。协议要求的是 pool 隔离，当前实现满足；语义去重不在本版协议内。

## D. q_t

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:104`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:107`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:111`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:114`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:115`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:119`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:124`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:125`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:17`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:23`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:24`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:26`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2026/theta_mapping_raw.json:17`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2027/theta_mapping_raw.json:17`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2028/theta_mapping_raw.json:17`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/seed_2029/theta_mapping_raw.json:17`

证据：
- `e3r_multiseed_theta_v06.py` 对每个 seed 的每个 state 先 `load_full_training_state()`，再用同一 held-out `partition['probe_indices']` 做 eval-mode probe forward，并通过 `torch.autograd.grad()` 构造该 state 自己的 `q_raw`。
- `normalize_direction()` 用全局 L2 norm 归一化所有参数方向，并拒绝非 finite 或过小 norm；`direction_norm()` 后写入 state result。
- 既有四个 seed 的 raw mapping 文件记录首个 state 的 `q_norm` 均约等于 1；同文件内各 state 也都有各自的 `q_norm`、`sigma_q`、`epsilon` 记录。

风险：
- v0.6 raw 输出记录 `q_norm` 与 `probe_grad_norm`，但没有为每个 state 持久化 q tensor 或 q sha256；后验只能证明实现会重算并记录 norm，不能直接从 raw 输出复核 q 的完整张量内容。

## E. E1

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:55`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:71`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:72`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:74`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:75`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:76`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:80`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:67`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:73`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:75`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:89`
- `/home/b205/projects/fluctuation_response/research/src/fr/direction.py:98`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:129`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:141`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:145`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:152`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:67`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:71`

证据：
- `run_theta_branch()` 在 accumulation 内完成所有 micro-batch backward 后，若 `measure_s_first=True`，在 `scaler.unscale_(optimizer)` 之前读取当前 accumulated gradient 投影。
- `project_current_gradients()` 只读 `p.grad`，若 GradScaler 启用则读取时除以 `grad_scale`，避免调用 `unscale_` 改变 optimizer/scaler 状态；该函数明确标注 `[FR-OBSERVATION-ONLY]` 与 E1 禁止写入扰动。
- v0.6 E1 调用 `run_theta_branch(sign=0, epsilon=0.0, measure_s_first=True)`；`apply_signed_gradient_impulse_()` 对 `sign == 0` 或 `epsilon == 0` 立即 return，因此 E1 没有实际人工 perturbation。

风险：
- E1 在共享 `theta_protocol` 内仍会经过 `apply_signed_gradient_impulse_()` 的 no-op 分支；从数值行为上是无扰动，但严格审计时应继续保留 no-op 单测与 rho=0 自检。
- v0.6 raw state-level 输出没有保存每个 E1 stream 的完整 `s_values` 和 batch/hash 明细，只保存聚合后的 `B_theta`、`sigma_q` 等；复核 stream 级细节需重新运行或增加审计产物。

## F. E2/E3 paired intervention

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:32`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:40`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:61`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:63`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:80`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:82`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:88`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:90`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:97`
- `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.py:99`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:36`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:67`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:74`
- `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py:82`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:157`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:160`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:179`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:191`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:192`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:193`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:194`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:203`

证据：
- 每个 branch 一开始都 `load_full_training_state()`，随后用同一 `stream_seed` 调用 `seed_future_stream()`；branch 内每个 micro-batch 用相同 LR 函数、相同 `global_micro_step` 和 `total_micro_steps` 写 LR。
- 扰动位置是 accumulation backward 完成后，先 `scaler.unscale_(optimizer)`，仅当 `h == 1` 时调用 `apply_signed_gradient_impulse_()`，再 `clip_grad_norm_ -> scaler.step -> scaler.update -> optimizer.zero_grad`；`h > 1` 不再施加扰动。
- v0.6 E2 对每个 stream 构造 `common` 参数，baseline/plus/minus 共用同一 `state_path`、`reference_theta`、`q`、`batches`、`stream_seed`、LR path；唯一变化是 `sign` 和 `epsilon`。
- `epsilon = rho * sigma_q`，且 `rho` 来自 `e3_primary_rho=0.25`；每个 state 先由 E1 stream 估计自己的 `sigma_q`。
- 对 baseline/plus/minus 的 batch manifest hash 做一致性检查；每个 training seed 至少做一次 rho=0 no-op，自检 model/optimizer hash 和 observations 完全一致。

风险：
- v0.6 `batch_manifest_hash` 是对 index manifest 的 hash，不是对实际 batch tensor bytes 的逐分支持久化记录；直接 `stack_batch()` 是确定性的，因此主路径风险低，但审计证据粒度低于旧 v0.4 E2 中的 batch tensor hash。
- 旧 standalone `e2_epsilon_linearity_v04.py` 存在 A 项所述 LR 时序风险；v0.6 正式 E3R 主路径已改用 `theta_protocol.py`。

## G. random stream 与 training seed

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/configs/e3r_cross_seed_v06.json:47`
- `/home/b205/projects/fluctuation_response/research/configs/e3r_cross_seed_v06.json:53`
- `/home/b205/projects/fluctuation_response/research/configs/e3r_cross_seed_v06.json:54`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:80`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:93`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:117`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:118`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_prepare_multiseed_state_bank_v06.py:119`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:28`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:30`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:31`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:32`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:133`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_multiseed_theta_v06.py:171`
- `/home/b205/projects/fluctuation_response/research/src/fr/stream_rng.py:18`
- `/home/b205/projects/fluctuation_response/research/src/fr/stream_rng.py:27`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/multiseed_state_bank_summary.json:2`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/multiseed_state_bank_summary.json:3`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/multiseed_theta_summary.json:2`

证据：
- 配置明确列出 independent training seeds `2026/2027/2028/2029`，并为 v0.6 E1/E2 使用不同 stream base seed。
- state-bank 脚本对每个 `train_seed` 执行 `setup_seed(train_seed)`，重新初始化模型，并使用 `train_batch_seed = train_seed + 1704` 形成独立训练 batch trajectory。
- `two_level_seed(base, train_seed, state_step, stream_id)` 把 training seed、state step、stream id 全部混入 future stream seed；E1 和 E2 分别用 `v06_e1_stream_base_seed` 与 `v06_e2_stream_base_seed`，避免跨 seed future stream seed 复用。
- 既有 multiseed state-bank summary 显示四个 seed 的 state bank technical pass。

风险：
- `outputs/e3_replication/v06/multiseed_theta_summary.json` 当前只列出 seed 2029，说明最近一次 mapping 脚本可能用 `--seeds 2029` 覆盖了全局 summary；个别 `seed_2026` 到 `seed_2029/theta_mapping_raw.json` 与 `e3r_cross_seed_analysis.json` 仍包含四个 seed，但仅看 summary 会误以为只有一个 seed 完成。
- partial rerun 的 overwrite summary 是运行追踪风险，不改变 `two_level_seed()` 的跨 seed 隔离语义。

## H. 统计

判定：PASS

文件路径与行号：
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:1`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:57`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:75`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:88`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:103`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:127`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:172`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:176`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:182`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:186`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:187`
- `/home/b205/projects/fluctuation_response/research/src/fr/cross_seed_stats.py:188`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:51`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:53`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:54`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:59`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:60`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:61`
- `/home/b205/projects/fluctuation_response/research/scripts/e3r_analyze_v06.py:62`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/e3r_cross_seed_analysis.json:4`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/e3r_cross_seed_analysis.json:11`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/e3r_cross_seed_analysis.json:12`
- `/home/b205/projects/fluctuation_response/outputs/e3_replication/v06/e3r_cross_seed_analysis.json:651`

证据：
- `cross_seed_stats.py` 顶部文档明确区分 training seed、state、random stream；`analyze_horizon()` 先对每个 seed 独立做 mapping，再给 pooled raw、within-seed detrended、step-adjusted、LOSO。
- `residualize_within_seed()` 在每个 seed 内线性去除 `state_fraction` 趋势；`first_differences()` 在同 seed 相邻 state 上做一阶差分。
- `step_adjusted_pooled()` 明确比较 step-only 与 step+B；`leave_one_seed_out()` hold out 一个 training seed，在未见 seed 上比较 MAE。
- `e3r_analyze_v06.py` 输出 `scientific_replication_pass: None`，并在 boundaries 中写明“4 个 training seed 才是独立科学重复单位”“pooled 28 state 仅作描述，不能替代 seed-level 复现”“scientific_replication_pass 故意不自动判定”。
- 既有 cross-seed analysis JSON 记录 `n_training_seeds: 4`、`states_per_seed: 7`，且保留上述 boundaries。

风险：
- 单元测试 `test_cross_seed_stats.py:17-19` 有 synthetic threshold `>0.8`，但该阈值只验证玩具数据中 B 确实增加解释力，不进入正式分析脚本，也未用于给科学正结论硬编码过关。
- 当前统计输出没有自动 p-value/CI 或科学 pass/fail；这符合“不要自动宣布正结论”的协议，但意味着最终解释仍需人工科学判断。

## 附加边界检查

文件路径与行号：
- `/home/b205/projects/fluctuation_response/AGENTS.md:1`
- `/home/b205/projects/fluctuation_response/AGENTS.md:4`
- `/home/b205/projects/fluctuation_response/AGENTS.md:8`
- `/home/b205/projects/fluctuation_response/AGENTS.md:15`
- `/home/b205/projects/fluctuation_response/RUN_ORDER_V06.md:1`
- `/home/b205/projects/fluctuation_response/RUN_ORDER_V06.md:3`
- `/home/b205/projects/fluctuation_response/RUN_ORDER_V06.md:6`
- `/home/b205/projects/fluctuation_response/RUN_ORDER_V06.md:11`

证据：
- 项目约束声明当前允许 E0/E1/E2、E3 theta mapping、E3-R cross-seed replication + detrending；禁止 module sweep、controller、V2A/V2B、按结果改阈值。
- 当前 v0.6 run order 是 check、unit tests、scientific preflight，然后 GPU state bank、GPU multiseed theta、CPU analyze；如果 preflight fail，不进入 GPU。
- 对 `research/` 与 `minimind/` 的禁用项字符串搜索未发现当前研究代码实现 module sweep/controller/V2A/V2B；命中 `controller` 的主要来源是 MiniMind 训练语料文本，不是研究代码模块。

## 总体结论

v0.6 主路径，即 `e3r_prepare_multiseed_state_bank_v06.py -> e3r_multiseed_theta_v06.py -> e3r_analyze_v06.py`，在代码结构上符合本提示词的 A-H 协议要求：MiniMind native AdamW 语义、full-state replay、probe/eval 隔离、state-local q_t、E1 自然涨落观测、E2/E3 paired intervention、跨 seed random stream 隔离、以及不把 28 state 冒充 28 个独立训练实验的统计边界。

需要保留的审计风险是：旧 standalone v0.4 E1/E2 脚本的 LR 写入时序不应再作为正式 v0.6 paired 主路径；CUDA RNG 没有在本次只读审计中动态验证；v0.6 raw mapping 未持久化每个 state 的 q tensor/stream 明细；`multiseed_theta_summary.json` 当前只反映最近 partial seed 2029 rerun，而不是四 seed 完整状态。

READY_FOR_CROSS_SEED_SCIENTIFIC_INTERPRETATION = YES
