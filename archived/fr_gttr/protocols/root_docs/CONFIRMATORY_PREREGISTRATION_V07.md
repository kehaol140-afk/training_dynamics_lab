# v0.7 Confirmatory FR Preregistration

生成日期：2026-09-02

## Protocol Boundary

本文件锁定 v0.7 Confirmatory FR Experiment。该实验是确认性实验，不允许根据结果调整协议、阈值、endpoint、seed、state grid、horizon、统计方法或 perturbation 位置。

禁止修改：

- `minimind/`
- native AdamW 语义
- perturbation 位置
- `rho=0.25`
- state grid
- horizon grid
- bootstrap/permutation cluster 单位
- Primary endpoint `H=8`
- PASS/FAIL 条件

## Discovery / Confirmation Separation

Discovery seeds：

- `2026`
- `2027`
- `2028`
- `2029`

这些 seed 标记为 `DISCOVERY_ONLY_DO_NOT_USE_FOR_PRIMARY_CONFIRMATORY`。它们不得进入 v0.7 Primary Confirmatory Test 的拟合、p-value、bootstrap CI 或 LOSO 判定。

Confirmation seeds：

- `2030`
- `2031`
- `2032`
- `2033`
- `2034`
- `2035`
- `2036`
- `2037`

v0.7 Primary Confirmatory Test 只使用这 8 个 training seeds。

## Fixed Scientific Protocol

State grid 固定：

- `46`
- `52`
- `58`
- `64`
- `70`
- `76`
- `82`

Horizon 固定：

- `1`
- `2`
- `4`
- `8`
- `12`
- `16`
- `24`
- `32`

Primary endpoint 固定：

- `H=8`

Stream 数固定：

- E1: `R=16`
- E2: `R=8`

Perturbation 固定：

```text
accumulation backward
-> scaler.unscale_
-> ±epsilon*q
-> clip_grad_norm_
-> native AdamW
-> zero_grad
```

扰动只允许在第一个 optimizer step 施加一次。

q 构造：

- 每个 training seed x state 独立构造 `q_t`。
- `q_t` 只从 held-out train-probe 构造。
- probe/eval 从 trajectory 第0步永久排除。

Paired branch 约束：

- baseline/+/- 必须从相同完整 `Z_t` 恢复。
- baseline/+/- 必须使用相同 batch manifest。
- baseline/+/- 必须使用相同 RNG stream。
- baseline/+/- 必须使用相同 LR path。
- 唯一允许差异是 `±epsilon*q`。

## Audit Artifacts

每个 seed x state 必须保存：

- q tensor
- q SHA256
- q norm
- probe grad norm
- `sigma_q`
- `epsilon`

每个 E1 stream 必须保存：

- `s_r`
- `stream_seed`
- `batch_seed`
- batch manifest
- manifest SHA256/hash
- first preclip gradient norm

每个 paired E2 stream 至少保存：

- `stream_seed`
- `batch_seed`
- batch manifest SHA256/hash
- baseline/+/- first preclip norm
- baseline/+/- clipping status
- baseline/+/- `X_theta(H)`
- rho=0 no-op result

## Primary Confirmatory Statistics

Primary model for `H=8`：

```text
chi_ij = alpha_seed_i + c * state_fraction_ij + K_8 * B_ij + error_ij
```

`alpha_seed_i` 是 training-seed fixed effect。

`H8_CONFIRMATION = PASS` 当且仅当以下三个条件同时成立。

条件 A：

- `K_8 > 0`
- 以完整 training seed 为 cluster 的 bootstrap 95% CI 下界 `> 0`
- bootstrap 固定随机种子
- 默认 `10000` replicates
- cluster 单位必须是完整 training seed

条件 B：

- 对每 seed 内分别去除 `state_fraction` 线性趋势
- 计算 residual `B` vs residual `chi` association
- 零假设检验使用 within-seed circular-shift permutation
- 保持每个 seed 内时间顺序结构
- 不允许普通随机打散全部 56 个点
- 固定随机种子
- 默认 `20000` replicates
- one-sided `p < 0.05`
- 预注册方向为 positive association

条件 C：

- 8-fold Leave-One-Training-Seed-Out
- 每 fold 训练使用其余 7 个 seed
- 测试完全 held-out seed
- M0: `chi = a + c*state_fraction`
- M1: `chi = a + c*state_fraction + K*B`
- 要求 mean relative MAE improvement `> 0`
- 要求至少 `6/8` folds improvement `> 0`

否则输出 `FAIL` 或 `INCONCLUSIVE`，不得自动降低阈值。

## Secondary Horizon Curve

Secondary horizons：

- `1`
- `2`
- `4`
- `12`
- `16`
- `24`
- `32`

同时报告 `H=8` 的完整曲线指标，但 `H=8` 仍是 primary endpoint，不纳入 secondary multiplicity correction。

每个 horizon 输出：

- per-seed Pearson
- per-seed Spearman
- seed-FE adjusted `K_H`
- cluster-bootstrap 95% CI(`K_H`)
- within-seed detrended Pearson
- circular-shift permutation p
- first-difference Pearson
- LOSO mean MAE improvement
- LOSO folds improved
- clipping baseline/+/- rates

若报告 secondary p-value 显著性，使用 Holm correction；`H=8` primary 不纳入 secondary correction。

## Discovery Comparison

确认分析全部结束后，可以读取 discovery results，只生成 `DISCOVERY_VS_CONFIRMATION_COMPARE` 部分。

允许比较：

- `K_H` 方向
- horizon curve 形状
- LOSO gain

Discovery 数据不得进入 confirmation primary statistics。

## Environment

正式实验环境固定：

- conda env: `fr_minimind`

正式 GPU confirmatory run 前必须生成 `ENVIRONMENT_V07.txt`，包含：

- python version
- torch version
- torch CUDA
- GPU
- conda list relevant packages

正式 GPU confirmatory run 前必须通过：

- 全部 CPU tests
- CUDA RNG roundtrip
- `scientific_preflight_v07.py`

任一失败则禁止 GPU confirmatory run。

## Preregistration Lock

正式 GPU confirmatory 数据开始产生前，必须对以下文件计算 SHA256 并写入 `outputs/e3_confirmatory/v07/PREREG_LOCK.json`：

- `research/configs/e3c_confirmatory_v07.json`
- `research/src/fr/confirmatory_stats.py`
- `research/scripts/e3c_analyze_confirmatory_v07.py`
- `CONFIRMATORY_PREREGISTRATION_V07.md`

一旦生成 lock，正式 confirmatory 数据开始产生后，上述文件禁止修改。

如果必须修改，必须产生新的 protocol version，并明确旧 confirmatory run 作废。
