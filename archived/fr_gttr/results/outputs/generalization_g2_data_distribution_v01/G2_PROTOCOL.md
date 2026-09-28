# G2 Protocol — FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01

> 预注册冻结：2026-09-16T14:20:00Z（在任何 G2 训练/响应结果打开前写成并冻结）。
> 本文件与 `research/configs/g2_data_distribution_generalization_v01.json` 相互印证；
> 数值阈值、判定分层、D1 选择记录、pool/partition 规则一经冻结不得根据结果修改。
> 口径红线：主统计口径严格 import 冻结的
> `fr.g1a_26m_scale_generalization_v01.cell_relative_errors` /
> `aggregate_metrics`（其内部再 import 冻结 a0c 的 `_metric` / `relative_error`），
> **禁止重新定义第三种 error caliber**。

## 1. 唯一科学问题（B0）

模型规模、架构、tokenizer、训练目标、AdamW、clipping、token 预算、因果响应协议
**全部固定**，仅预训练语料从 D0 换成 D1 时：完整切线 C 是否仍然准确预测
有限差分（FD）响应，并在 H>1 系统性优于 optimizer-only 切线 B？

- 这不是算法实验。除语料外任何字段与冻结 G1a 26M 逐位一致。
- D0 vs D1 的对比是 **matched descriptive comparison**（按 seed/stage/H 身份配对），
  不是样本配对（paired）的因果数据集实验：两条轨迹喂的是不同语料。

## 2. D0 核验（B1）

- D0 = 冻结 G1a 实际训练池：`outputs/e3_confirmatory/v07/e3c_subset.jsonl`
  （4096 条记录，SHA256 `1e657af9baae0784e582d82ef936160473aa069878e422ef48683a718056b0f4`，
  已核验与冻结 G1a config/manifest 一致）。以冻结 manifest 为准——
  它不是仓库里的 `minimind/dataset/pretrain_t2t_mini.jsonl` 全量文件。
- D0 **只读直取**（冻结 G1a `results/per_state_horizon.csv`、`summary.json`、
  `mechanism_decomposition.csv` 等），**永不重跑 D0 训练**。

## 3. D1 选择与来源（B2，先于任何结果冻结）

- 优先级规则：(1) MiniMind 官方 `pretrain_hq.jsonl` 兼容且充足且与 D0 为不同语料
  → 用之；(2) 其它本地 MiniMind 官方资产；(3) 下载到版本化数据目录并记录 provenance；
  (4) 都不行 → `DATASET_UNAVAILABLE`（不做随机替代、不引入 FineWeb/英文语料、
  不做任何"科学"）。
- **实际执行**：`pretrain_hq.jsonl` 不在本地，且已从 HF 主分支删除（2026-03-26
  commit `3d8cf7da91c2`），但在删除前的官方 revision
  `3a9bbe611244811e45b96661a84047f112c8ed01`（2026-01-17 上传）仍存在。
  按规则 (3) 下载到
  `outputs/generalization_g2_data_distribution_v01/data/pretrain_hq.jsonl`
  （本地 SHA256 待 dataset 阶段与 LFS oid
  `82b19375460d44feb15937ea3712121b5292271cf18eb87407b6d3e27427c18b` 核验）。
- **D1 在本文件冻结之时、在任何 G2 响应结果打开之前即已锁定**；
  不存在"按结果挑数据集"的通道：下载发生在任何训练之前，替代语料从未进入比较。
- schema 核验：`{"text": <string>}`，与 D0 完全一致 → **不需要 adapter**（B4）；
  若字段名不同也只允许纯 schema adapter。

## 4. 预处理公平性（B4）

- D1 与 D0 共用完全相同的 tokenizer（vocab 6400）、seq_len 128、packing、
  special tokens、causal LM loss、normalization、PretrainDataset 管线
  （BOS/EOS、截断 126+2、padding mask）、dataloader 语义。
- 唯一允许的差异：字段名不同时的纯 schema adapter（本实验未触发）。
- 禁止任何内容过滤、质量平衡、语料混合、按内容挑样本。

## 5. Pool 与 partition 冻结规则

- D1 pool = 源语料全量 N 条记录上 `make_permutation(N, seed=60202601)`
  排列的前 4096 个索引，升序排序 → `data/d1_pool.jsonl`（4096 条，与 D0 pool
  同规模）。无任何内容选择。
- partition = `build_partition(4096, seed=60202603, probe_size=32, eval_size=128)`
  → `partition_g2_d1.json`（probe/eval/train = 32/128/3936，与 v07/G1a 相同规模）。
- probe/eval 永久排除在训练轨迹之外（q 构造只用 probe batch）。

## 6. 模型与训练（B5/B6/B7，逐字段克隆 G1a）

- 模型：26,681,728 unique params；hidden 512 / layers 7 / heads 8 / kv 4 /
  head_dim 64 / intermediate 1664（冻结默认公式）/ vocab 6400 / seq 128 /
  tie embeddings / 无 MoE。`_reinit_model` 同款。
- 训练：AdamW betas=(0.9,0.999) eps=1e-8 wd=0.01；clip=1.0；bf16 无 GradScaler；
  batch 4 × accum 2 = 1024 tokens/optimizer-step；cosine LR（set_native_micro_lr，
  256 micro-steps 轴）；128 个 planned optimizer steps；train batch seed =
  train_seed + 1704（同冻结规则）。
- 状态：EARLY/MIDDLE/LATE = step 26/64/102（≈20.3/50/79.7%），训练在 102 步停止。
- seeds = 2042/2043/2044（与 G1a 相同；因语料不同，D0 vs D1 非样本配对）。
- 年龄一律按 tokens/steps/schedule fraction 计，不按"文件 epoch"计。

## 7. 响应协议继承（B8）

- 模型阶梯：FD（paired 有限差分）/ B（optimizer-only）/ C（full tangent,
  clip-JVP）；**无 Model A**；BH_fixed 仅进机制子集。
- 9 states × 4 preregistered streams × H∈{1,4,8} = 27 个 official cells /
  36 条 baseline 流。
- `future_stream_manifest.json` 在任何 E2 响应分支**之前**冻结（36 条流，
  每条含完整 micro-batch 清单）。
- stream seed 派生规则与 G1a 完全相同（E1 base 408202601 / E2 base 409202601，
  two_level_seed）——同一批索引流在 D1 语料上重放，用于 matched descriptive 对比。
- ε = 0.25·σ_q（σ_q = 16 条 E1 流 s 的 ddof=1 标准差）；CRN 三分支；rho0-noop；
  H=1 局部恒等；HVP = math-SDPA reverse-over-reverse surrogate（继承 mixed-backend
  caveat）。

## 8. Smoke gate（B9）

- 状态：**seed 2042 / EARLY / step 26**（按 B9 指定；G1a 用的是 MIDDLE，G2 用 EARLY）。
- 检查项：checkpoint 完整恢复（model+optimizer m/v+scaler+RNG）；optimizer state
  维度/名称匹配；q 名称/形状严格一致；FD ± 分支 CRN 一致；H=1 解析 B vs FD 一致；
  B/C shadow baseline 哈希相等；HVP 可执行且有限；无意外 optimizer step；输出全有限；
  **train-loader deterministic replay（双重派生相等）+ dataset/pool/partition 哈希核对**。
- 策略：实现 bug 可修复并记录 CHANGE_LOG 后重 smoke；**协议不兼容 →
  CONTRACT_INCOMPATIBLE，停止并报告**。

## 9. 主矩阵与误差定义（B10）

- 9 states 全部跑 FD/B/C（无 Model A）。
- 每 cell 记录 abs/rel/signed 误差。**只使用冻结 G1a/G1b 主聚合定义**
  （`cell_relative_errors` + `aggregate_metrics`），不允许第三种口径。
- 官方 cell 口径：每 (seed,stage,H) cell 先对 4 条流的 chi 取均值 →
  一个 cell 级相对误差（冻结 a0c `relative_error`：|p−o|/|o|，obs==0 时
  pred==0→0 否则 None）。stream-first 口径（G1a §7）仅用于机制子集与稳健性叙述。

## 10. 机制子集（B11）

- seed 2042 × 3 stages × 4 streams × H={1,4,8}，含 BH_fixed（迁移诊断，
  **不是**判定的必要条件）。
- gain_curvature = err_B − err_BH_fixed；gain_clip = err_BH_fixed − err_C。
- **stream-first 与 official-cell 两种口径都报告、都标注、永不混用**。

## 11. 判定分层（B12，阈值冻结于任何结果之前）

- 分析网格：9 states × H∈{1,4,8} = 27 行；H=1 行为结构性平局（B≡C），
  不计入 paired 胜负，但计入 pooled R²/MAE。
- **STRONG_DATA_REPLICATION**：同时满足——
  cond1 C pooled R²（27 行）≥ 0.90；
  cond2 18 个 H>1 行中 ≥80%（≥15/18）err_C < err_B（平局 |Δ|≤1e-12）；
  cond3 H=8（9 行）：paired mean err_C < mean err_B 且 C R²(H=8) ≥ 0.50 且 corr(H=8) > 0；
  cond4 ≥2/3 seeds 的 mean(err_B − err_C) > 0。
- **PARTIAL_DATA_REPLICATION**：pooled H>1 mean rel err C < B，但非全部 STRONG 条件成立。
- **NO_DATA_REPLICATION**：C 不整体优于 B，或 C 对 FD 失去预测效度。
- **RESOURCE_LIMIT / DATASET_UNAVAILABLE / CONTRACT_INCOMPATIBLE**：定义见 config。
- 阈值永不因结果改变。

## 12. Dynamics 诊断（B13）

- 训练期逐步记录：loss、lr、preclip grad norm、clip active、param norm。
- future 逐步记录：clip active、clip coefficient、HVP norm、native grad norm。
- 分布偏移诊断：训练 loss、梯度/preclip norm、clip 激活率、param norm、
  future grad norm、HVP norm/曲率代理、token/长度样本统计。
- 仅作解释，**不得用于事后过滤状态**。

## 13. D0 vs D1 对比与 bootstrap（B14/B15）

- D0 从冻结 G1a 直读；对比 = 按 (seed,stage,H) 身份配对的 matched descriptive
  比较；禁止样本配对式因果措辞。
- D1 H>1 paired B/C 的 cluster bootstrap：seed → stage → cell 层级，
  20000 resamples，rng 20260916；报告 mean paired improvement、95% percentile
  interval、以及"**bootstrap 重采样中差值为正的比例**"。该比例**不是** p 值、
  不是后验概率、不是显著性概率。3 seeds ⇒ 探索性不确定性；
  streams ≠ independent runs。

## 14. 资源规则（B16）

- GPU 共享；永不 kill 外部任务；GPU1 不动。
- OOM 时只允许已有的保科学内存修复（如 math-SDPA HVP）；禁止缩小模型/seq/
  batch/H/optimizer/clip/precision；失败的 state 不得悄悄丢弃——报
  RESOURCE_LIMIT。

## 15. 输出目录（B17）

`outputs/generalization_g2_data_distribution_v01/` 至少包含：
`G2_PROTOCOL.md`（本文件）、`DATASET_PROVENANCE.md`、`dataset_manifest.json`、
`experiment_config.json`、`training_manifest.json`、`checkpoint_manifest.json`、
`future_stream_manifest.json`、`smoke_report.json`、
`results/{summary.json, per_stream.csv, per_state_horizon.csv,
mechanism_decomposition.csv, d0_vs_d1_comparison.csv, bootstrap.json}`、
G2 报告、`CHANGE_LOG.md`、`COMMANDS.log`、`ENVIRONMENT.json`、
`SOURCE_AND_INPUT_HASHES.txt`、`SHA256SUMS.txt`。

## 16. 文档与交接（B18）

完成后更新 `docs/FR_PROJECT_STATUS.md`、`FR_EXPERIMENT_INDEX.md`、
`FR_GENERALIZATION_STATUS.md`、`FR_GENERALIZATION_HANDOFF.md`（D1 provenance /
矩阵 / verdict / 指标 / dynamics / 机制子集 / 不确定性 / caveats）。
若 G2 完成（STRONG 或明确完成态）→ HANDOFF next = G1c（~1B 压力测试），
**但不启动**；若 PARTIAL/NO → next = G2 诊断/边界分析。

## 17. 禁改红线（继承）

不得修改或覆盖 A0A–A0G 冻结科学结果、A0H/A″ 算法线、5M anchor、G1a 26M、
G1b 100M、G3 冻结 CSV/JSON/raw results、旧 checkpoint/q/verdict；不得自动启动
1B、G4/A0G-lite、新 optimizer、beta sweep、新 clipping sweep、新算法实验；
所有新产物进入本版本化目录。
