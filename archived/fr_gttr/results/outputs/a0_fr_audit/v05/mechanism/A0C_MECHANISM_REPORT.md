# A0C Mechanism Report

## 技术摘要

- 两个既定模型保持不变：`chi ~ 1 + B` 与 `chi ~ 1 + B + alpha_direct`。整体 `delta_R2=0.77059962`。
- 1000 次 seed-cluster bootstrap 的 `delta_R2` mean 为 `0.76438714`，95% percentile CI 为 `[0.67188288, 0.87585352]`。
- Overall correlation：`corr(alpha_direct, chi)=0.86489195`，`corr(alpha_direct, B)=-0.14186708`。
- 本审计保持观察性边界，不修改实验定义或阈值，不进入 covariance geometry。

## 两个预设模型仍给出相同的整体解释率增量

| model | formula | n | R2 |
| --- | --- | --- | --- |
| base | `chi ~ 1 + B` | 448 | 0.00082199 |
| control | `chi ~ 1 + B + alpha_direct` | 448 | 0.77142161 |

控制模型相对基础模型的样本内解释率增量为 `delta_R2=0.77059962`。模型含截距，定义与 v04 一致。

## Seed-cluster bootstrap 在完整 seed block 上重采样

每个 replicate 从 8 个 seed 中有放回抽取 8 个 seed，并拼接每个被抽 seed 的全部行；未在 cluster 内抽 row。

| quantity | value |
| --- | --- |
| point estimate delta_R2 | 0.77059962 |
| bootstrap mean delta_R2 | 0.76438714 |
| bootstrap standard deviation | 0.05182472 |
| percentile 95% CI | [0.67188288, 0.87585352] |
| valid replicates | 1000/1000 |

该区间把 seed 作为重采样单位，因此保留一个 seed 内 state 与 horizon 行的共同变动；只有 8 个独立 cluster，区间精度仍受 seed 数量限制。

## Correlation audit

Pearson 相关系数按 overall、逐 seed 和 v04 指定的 H=1/2/4/8 输出。表格保留全部精确分层值，因此不另加图形。

| level | group | n | corr(alpha_direct, chi) | corr(alpha_direct, B) |
| --- | --- | --- | --- | --- |
| overall | all | 448 | 0.86489195 | -0.14186708 |
| seed | 2030 | 56 | 0.80312228 | -0.22355503 |
| seed | 2031 | 56 | 0.92624662 | -0.01045171 |
| seed | 2032 | 56 | 0.96544987 | 0.17285397 |
| seed | 2033 | 56 | 0.95119170 | 0.44612721 |
| seed | 2034 | 56 | 0.91851071 | -0.51663002 |
| seed | 2035 | 56 | 0.90807962 | -0.50790855 |
| seed | 2036 | 56 | 0.94898855 | -0.36067267 |
| seed | 2037 | 56 | 0.77671836 | -0.30586274 |
| horizon | 1 | 56 | 0.91761112 | 0.29311786 |
| horizon | 2 | 56 | 0.83934268 | 0.22474991 |
| horizon | 4 | 56 | 0.68801160 | 0.02694215 |
| horizon | 8 | 56 | 0.69954797 | -0.07356445 |

逐 seed 的 `corr(alpha_direct, chi)` 范围为 `[0.77671836, 0.96544987]`；逐 horizon 范围为 `[0.68801160, 0.91761112]`。相关结果用于检查 direct channel 与 chi/B 的共同变化，不提供因果分解。

## 数据、定义与复现设置

- 输入：`outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv`
- 输入 SHA256：`59e32a6dff4ce499232f220440e7aefa3f7e4cef1760058ea70eb318ed268b13`
- 样本：448 行，8 个 seed；correlation horizon 范围固定为 [1, 2, 4, 8]。
- 相关系数：未残差化的 Pearson sample correlation。
- Bootstrap：seed 为 cluster，1000 次，固定 RNG seed=20260909，percentile 95% CI。
- 计算仅使用 CPU。

## 限制与审计边界

- Cluster bootstrap 对 8 个 seed 重采样，不把 448 行当作独立 cluster；但 seed 数较少，percentile CI 可能较离散。
- Pearson correlation 未控制 B、state_step 或其他路径，只描述成对线性共变。
- `alpha_direct` 仍是诊断性直接通道；模型和相关均不识别精确 AdamW 因果机制。
- Covariance geometry 明确排除在 v05 范围之外，未读取或计算其产物。
- [FR-RNG-CRITICAL] bootstrap 随机种子固定。
- [FR-OBSERVATION-ONLY] 本报告仅作观察性机制审计。

## 建议的后续解读

- 联合阅读 cluster CI、逐 seed correlation 与逐 horizon correlation，判断 direct channel 关联是否由单一 seed 或单一短 horizon 主导。
- 不依据本审计调整既有阈值，也不将其用于进入 covariance geometry。

## 待回答问题

- 在更多预先指定的独立 seed 上，cluster bootstrap 区间宽度是否稳定？
