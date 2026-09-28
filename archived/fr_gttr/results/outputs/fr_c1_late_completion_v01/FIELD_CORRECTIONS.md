# FIELD CORRECTIONS — FR_C1_LATE_COMPLETION_V01 字段更正（0 GPU 离线核对，2026-09-21）

核对脚本：`research/scripts/c1_late_fieldcheck_v01.py`（全部数字由其从
`runs/late_records.jsonl` 已存充分统计重算；不重新训练、不加 probe、不改阈值）。
原 REPORT.md / KEY_DATA_FOR_REVIEW.md 保留，本文件为追加勘误。

## 0. 冻结定义（所有更正的基准，源码行号）

| 量 | 定义 | 来源 |
|---|---|---|
| mean_norm² | ‖x̄‖² = ‖Σx_j‖²/n² | `fr/c1_gttr_probe_validation_v01.py:105-108` (mean_sq_norm) |
| mean_sample_norm² | (1/n)·Σ‖x_j‖² | group_stats.sum_sq_norm / n |
| centered_SS | Σ‖x_j−x̄‖² | `c1_late_run_v01.py:_centered_ss` |
| SE_vec | √(centered_SS/[n(n−1)]) | `fr/c1_gttr_probe_validation_v01.py:111-118` (se_vec_of) |
| inner (triple) | ⟨x̄_a,x̄_b⟩（means 级） | `fr/c1_gttr_probe_validation_v01.py:121-123` |
| triple sse/tss | ‖P̄−Ȳ‖² / ‖Ȳ‖²（means 级） | `_triple_stats` → sse_error |
| 残差配对 | 同库同索引 x_j = p_A,j−y_A,j（audit） | `c1_late_run_v01.py` rd = P_list[j] − Y |
| 残差 mean/SE | mean(norm) / SE_of_norm（per-probe 标量列） | R._mean_se(resid_norm) |
| stop-v full/stop/ratio/cos | per-probe ‖Z_j‖, ‖Zstop_j‖, stop/full, cos → mean/SE | `_sv_norms` + `_mean_se` |
| Zdiff_audit q+ | per-probe ⟨Z_j−Zstop_j, q+⟩ → mean/SE | `c1_late_run_v01.py:344` |

## 1. 字段更正表

| # | 位置 | 原标签/表述 | 真实定义（已核验） | 处置 |
|---|---|---|---|---|
| E1 | Table A `target_norm`/`pred_norm` | "target_norm" | ‖x̄‖ = norm(mean)（mean_sq_norm 开方） | 标签保留，加注 norm(mean)；数值不变 |
| E2 | Table A `SE`（target/pred） | "SE" | SE_vec = √(centered_SS/[K(K−1)]) | 更名 SE_vec；数值不变 |
| E3 | Table A `resid_SE` | "residual_SE"（与 SE 并列暗示同类） | **SE_of_norm**：per-probe 库内残差范数列 ‖p_A,j−y_A,j‖ 的 SE，**不是** SE_vec | 更名 resid_SE_of_norm；并补 `resid_mean(norm)` 列 |
| E4 | Table A（缺列） | — | target/prediction 的 mean(norm)/SE_of_norm 为 per-probe 标量统计，**未持久化** | 补两列 = NOT_SAVED（不猜测补齐） |
| E5 | Table C full/stop 行 | 同一组 Z 统计重复出现在 field 与 shadow 行 | full/stop/ratio/cos 是 **shadow 对象**（Z_A、Zstop_A）的统计 | field 行 full/stop 列改 "—"，仅保留 field 分解量 |
| E6 | Table C "norm ratio" | 单一 "ratio" | 原报 1.2222/1.2363 = **mean_of_probe_ratios**；另有 ratio_of_mean_norms = ‖Z̄stop‖/‖Z̄‖ = **1.0486/1.0432**（新列） | 拆为四统计量并列（见 §3） |
| E7 | Table D "full_minus_stop" | 误标 | 该列实取 `fixed_dir_proj.Zstop_A.q+.mean` = **stop_mean**（stop 信号在 q+ 上的投影均值） | 重建四列：full_mean / stop_mean / paired_diff_mean / paired_diff_SE |
| E8 | Table A `SNR` | 单一比值 | 实为 ratio_to_3SE = ‖x̄‖/(3·SE_vec)；另列 signal_over_SE = ‖x̄‖/SE_vec | 拆为两列（见 §4） |
| E9 | Table C/正文 "shadow_norm_ratio" | 未注明统计级 | mean_of_probe_ratios（per-probe 比值取均值，≠ 均值之比） | 正文引用时必须带限定词 |

## 2. Task 1 独立复算（target/prediction，48 行；此处 full32 主行）

公式：mean_norm² = mean_sq_norm；mean_sample_norm² = sum_sq_norm/n；SE_vec² = centered_SS/[n(n−1)]。
恒等校验：mean_norm² + centered_SS/n − mean_sample_norm² = 0（48/48 行，max |diff| = 0.0）。

| anchor ch group | n | mean_norm² | mean_sample_norm² | centered_SS | SE_vec | ident_diff |
|---|---|---|---|---|---|---|
| R field target | 32 | 2.386481e-03 | 4.338878e-03 | 6.247671e-02 | 7.936029e-03 | 0 |
| R field prediction | 32 | 2.152660e-03 | 4.045898e-03 | 6.058359e-02 | 7.814869e-03 | 0 |
| R shadow target | 32 | 3.593481e-06 | 6.527912e-06 | 9.390180e-05 | 3.076671e-04 | 0 |
| R shadow prediction | 32 | 3.129304e-06 | 5.995104e-06 | 9.170561e-05 | 3.040480e-04 | 0 |
| C field target | 32 | 3.835481e-03 | 7.344392e-03 | 1.122852e-01 | 1.063911e-02 | 0 |
| C field prediction | 32 | 3.441841e-03 | 6.901812e-03 | 1.107191e-01 | 1.056466e-02 | 0 |
| C shadow target | 32 | 4.633345e-06 | 8.343486e-06 | 1.187245e-04 | 3.459508e-04 | 0 |
| C shadow prediction | 32 | 4.049093e-06 | 7.655158e-06 | 1.153941e-04 | 3.410641e-04 | 0 |

mean(norm)/SE_of_norm（target/prediction）：**NOT_SAVED**（per-probe 标量列未持久化；probe_banks 只存样本 ID）。残差的对应量已保存，见 §3。

## 3. Task 2 field 同 probe 残差——定义裁决（无数值矛盾，标签混用）

配对定义（runner 源码）：x_j = p_A,j − y_A,j（**audit 库内、同索引**；非跨库配对）。

| cell | mean(norm) 已存 | SE_of_norm 已存 | norm(mean)=√local_SS | local_SS | Jensen 间隙 = mean(norm)−norm(mean) |
|---|---|---|---|---|---|
| R | 8.512193e-3 | 1.929651e-4 | 3.890366e-3 | 1.513495e-5 | +4.621827e-3 ≥ 0 ✓ |
| C | 1.017302e-2 | 2.792151e-4 | 3.405361e-3 | 1.159648e-5 | +6.767658e-3 ≥ 0 ✓ |

- local_SS ≡ 残差的 mean_norm²（‖x̄‖²，x̄ = P̄_A−Ȳ_A）：两来源逐位一致（match=True）。
- 用户提出的"若 SE 是 SE_vec，mean 不能同时为 mean norm"——裁决：**该 SE 不是 SE_vec**，是与 mean 同属 per-probe 范数标量列的 SE_of_norm，二者自洽；原表错在把 SE_of_norm 与 target/pred 的 SE_vec 同列于 "SE" 名下且未声明。
- 残差的 mean_sample_norm² / centered_SS / SE_vec：**NOT_SAVED**（mean_j⟨p_A,j,y_A,j⟩ 未持久化；triple 的 inner_pA_yA = ⟨P̄_A,Ȳ_A⟩ 为 means 级；禁止反推补齐）。

## 4. Task 3 表 D 重建（shadow；full/stop 为 Z_A、Zstop_A 在固定 q+ 上的投影）

| cell | full_mean | stop_mean | paired_diff_mean | paired_diff_SE | 恒等差 (full−stop)−paired |
|---|---|---|---|---|---|
| R | 2.396305e-4 | 2.476426e-4 | −8.012156e-6 | 9.266598e-7 | 3.2e-20 |
| C | −2.807036e-6 | −3.447412e-6 | +6.403761e-7 | 5.806321e-7 | −2.0e-21 |

原 "full_minus_stop" 列 = 本表 **stop_mean**（用户用 target−该列≈paired 差值验证，完全正确）。field 行不设 full/stop 列。

## 5. Task 4 表 C 四统计量拆分（shadow；field 行只保留分解量）

| cell | norm(mean) full | norm(mean) stop | ratio_of_mean_norms | mean(norm) full | mean(norm) stop | mean_of_probe_ratios | mean cos |
|---|---|---|---|---|---|---|---|
| R | 1.895648e-3 | 1.987726e-3 | **1.048574** | 2.531910e-3 | 3.112285e-3 | **1.222215**±0.024260 | 0.874561 |
| C | 2.152521e-3 | 2.245479e-3 | **1.043186** | 2.868029e-3 | 3.513664e-3 | **1.236338**±0.026927 | 0.866872 |

要点：mean_of_probe_ratios（1.222/1.236）≠ ratio_of_mean_norms（1.049/1.043），Jensen 效应使逐 probe 比值均值高于均值之比；两个量各自合法，引用必须限定。原报告所有 "norm ratio 1.2222/1.2363" 的表述在更正后特指 **mean_of_probe_ratios**（数值不变）。

## 6. Task 5 SNR 拆分、resolved 复核、COMBINED6 复算

| anchor ch | signal_over_SE | ratio_to_3SE | resolved（复核） |
|---|---|---|---|
| R field | 6.155677 | 2.051892 | True |
| R shadow | 6.161359 | 2.053786 | True |
| C field | 5.821093 | 1.940364 | True |
| C shadow | 6.222042 | 2.074014 | True |

COMBINED6 固定集合（A/t0;Bp/t16;A/t48，不删 anchor）12 行 E **与原 aggregate_gate_inputs.csv 逐位一致**（max diff < 1e-12）；thr 与布尔不变。

## 7. 主判定影响

**保持不变（无需勘误数值）**：全部 E / SSE / TSS（means 级定义复核一致）；resolved 标志与 distinct-t；OLD4/LATE2/COMBINED6 全部门布尔；预算与覆盖；泄漏顺序与恒等性结论。

**需要勘误（标签/定义级，数值已在上方给出）**：
1. Table A：resid_SE → SE_of_norm，补 mean(norm)；target/pred 补 NOT_SAVED 的 mean(norm)/SE_of_norm 列（E3/E4）。
2. Table C：field 行撤下 full/stop；ratio 拆为 mean_of_probe_ratios 与 ratio_of_mean_norms 两列（E5/E6）。
3. Table D："full_minus_stop" 实为 stop_mean；重建 full/stop/paired 三列（E7）。
4. Table A SNR → signal_over_SE 与 ratio_to_3SE 两列（E8）。
5. 正文引用 stop-v ratio 时须写 mean_of_probe_ratios；ratio_of_mean_norms≈1.04–1.05 为新增信息：冻结 P 使 **均值范数** 增约 4–5%，而逐 probe 范数均值增 22–24%——"v 通道携带增量信息"的定性结论方向不变，量级表述须注明统计级。

** NOT_TESTED 维持**：H16/H64 预测、adaptive 训练价值、最终训练改进、能耗收益。停止条件维持：不进入 C2。
