# FD 局部算子诊断 v01 — 勘误与动作差补充 (v02, 0-GPU)

日期：2026-09-23。输入：仅本目录已保存 CSV（fd_scale_diagnostic / ad_reference_comparison /
block_perturbation_summary）。无 torch、无 GPU、无新数值路径、未恢复 Phase B、未调阈值、未重选尺度。
性质：勘误（修正 v01 REPORT/KEY_DATA 中的表述）+ 动作差补充诊断。旧记录原样保留，本文件不覆盖任何旧文件。

## 任务 1 — 表 E 勘误：10.04 / 5.85e-3 是 norm²，不是 norm

对象：errvec d = y(δ0) − y(δ0/2)（S3/S1 首方向）。两条独立复算路径：

| state | A: Σblock errvec_sq（直接保存的充分统计） | B: fd_rel_adj×max(‖y(δ0)‖,‖y(δ0/2)‖) | ‖d‖=√A |
|---|---|---|---|
| S3 | 10.038436 | 3.168349 | 3.168349 |
| S1 | 0.005848 | 0.076471 | 0.076471 |

两来源一致。派生交叉内积（极化恒等式，交叉内积未直接保存，标 DERIVED）：

| state | ‖y(δ0)‖² | ‖y(δ0/2)‖² | ⟨y(δ0),y(δ0/2)⟩ derived | cos derived | cos saved (CSV) |
|---|---|---|---|---|---|
| S3 | 6.269174 | 20.131940 | 8.181339 | 0.72824311 | 0.72824315 |
| S1 | 1.301647 | 1.341981 | 1.318890 | 0.99790456 | 0.99790413 |

**勘误**：旧表述「差 3 个数量级」的对象是 **范数平方比** = 1716.6（≈10^3.23）；
**范数本身比** = 41.4（≈10^1.62），不到两个数量级。
正确表述：‖d‖²(S3)/‖d‖²(S1)≈1.7e3；‖d‖ 比≈41。

## 任务 2 — 尺度结论勘误

固定集合内各相邻尺度对的 fd_rel 与 0.20 门对照（原门只查 (δ0,δ0/2) 这一对）：

| 相邻对 | S3 fd_rel | S1 fd_rel |
|---|---|---|
| δ0↔2δ0 | 0.7228 | 0.1085 |
| δ0/2↔δ0 | 0.7061 | 0.0660 |
| δ0/4↔δ0/2 | 0.3652 | 0.0224 |
| δ0/8↔δ0/4 | 0.1396 | 0.0061 |

- **保留原失败**：v3 门在首方向查 (δ0,δ0/2)，S3=0.7061>0.20 → FAIL 原样保留，不追认 PASS。
- **勘误旧表述**：「固定集合内没有任何尺度能通过 0.20 门」错误。(δ0/4↔δ0/8) 对 fd_rel=0.1396<0.20
  （S1 全部四对均<0.20）。
- **两件事分开**：① FD(h)→AD(h→0) 的收敛（S3 sym_rel_err 0.98→0.10，δ0/8 处仍未收敛；S1 至 0.002）
  是**导数估计**问题；② 有限步线性模型在真实参考位移上是否有效（任务 3/5）是**预测**问题。
  小尺度对过门只说明 ① 在改善，不构成 ② 的证据，也不回溯改变 v3 的门判定。

## 任务 3 — D5 误差定义（明确化）

对象：**固定 SIM、固定 θ0、参考一步位移** s_ref0（clip+AdamW 虚拟步，‖s_ref0‖ 见 h_eff 列），
**不是已运行 MAIN 的真实轨迹**。定义：
actual = G(θ0+c·s_ref0) − G(θ0)；pred = h_eff·y（线性模型，h_eff=c·‖s_ref0‖，y∈{FD@δ0, FD@2δ0, AD}）；
relative_error = ‖actual−pred‖ / ‖actual‖（分母=实际变化范数，下表 denom 列）；
cos(actual, pred)：需端点向量内积，**DATA_MISSING**（未保存）。

| state | 检查点 | h_eff | ‖actual‖ | 预测 | ‖pred‖ | ‖residual‖ | denom | rel_err | cos |
|---|---|---|---|---|---|---|---|---|---|
| S3 | s_ref0 | 0.1080 | 5.3819e-01 | FD@δ0 | 2.7038e-01 | 5.6130e-01 | 5.3819e-01 | 1.0430 | DATA_MISSING |
| S3 | s_ref0 | 0.1080 | 5.3819e-01 | FD@2δ0 | 1.5019e-01 | 4.8820e-01 | 5.3819e-01 | 0.9071 | DATA_MISSING |
| S3 | s_ref0 | 0.1080 | 5.3819e-01 | AD | 6.5520e-01 | 1.0046e+00 | 5.3819e-01 | 1.8666 | DATA_MISSING |
| S3 | exp05_s_ref0 | 0.1135 | 5.4726e-01 | FD@δ0 | 2.8424e-01 | 5.7711e-01 | 5.4726e-01 | 1.0546 | DATA_MISSING |
| S3 | exp05_s_ref0 | 0.1135 | 5.4726e-01 | FD@2δ0 | 1.5789e-01 | 4.9488e-01 | 5.4726e-01 | 0.9043 | DATA_MISSING |
| S3 | exp05_s_ref0 | 0.1135 | 5.4726e-01 | AD | 6.8880e-01 | 1.0455e+00 | 5.4726e-01 | 1.9104 | DATA_MISSING |
| S1 | s_ref0 | 0.1059 | 1.3074e-01 | FD@δ0 | 1.2085e-01 | 3.1184e-02 | 1.3074e-01 | 0.2385 | DATA_MISSING |
| S1 | s_ref0 | 0.1059 | 1.3074e-01 | FD@2δ0 | 1.1735e-01 | 3.3655e-02 | 1.3074e-01 | 0.2574 | DATA_MISSING |
| S1 | s_ref0 | 0.1059 | 1.3074e-01 | AD | 1.2364e-01 | 3.1217e-02 | 1.3074e-01 | 0.2388 | DATA_MISSING |
| S1 | exp05_s_ref0 | 0.1114 | 1.3782e-01 | FD@δ0 | 1.2705e-01 | 3.4294e-02 | 1.3782e-01 | 0.2488 | DATA_MISSING |
| S1 | exp05_s_ref0 | 0.1114 | 1.3782e-01 | FD@2δ0 | 1.2337e-01 | 3.6293e-02 | 1.3782e-01 | 0.2633 | DATA_MISSING |
| S1 | exp05_s_ref0 | 0.1114 | 1.3782e-01 | AD | 1.2998e-01 | 3.4670e-02 | 1.3782e-01 | 0.2516 | DATA_MISSING |

（‖pred‖=h_eff×‖y‖、‖residual‖=rel_err×‖actual‖ 均由保存标量重构；机器可读版：d5_error_decomposition.csv）

## 任务 4 — FP32/FP64 「逐位一致」勘误（落实为真实字段）

| 项 | FP32 路径 | FP64 路径 | 一致性 |
|---|---|---|---|
| path | ORIG_FP32（FD）/ ORIG_FP32_MATHREF（AD） | REF_FP64 | — |
| dtype | float32 | float64 | — |
| backend | PyTorchMemEffAttention f32 fmha_cutlassF + cutlass SIMT sgemm | cutlass tensorop d884 GEMM | 不同后端（REFERENCE_BACKEND_CHANGE） |
| fd_rel 序列 | 0.7228/0.7061/0.3652/0.1396 | 0.7228/0.7061/0.3652/0.1396 | **4 位显示精度一致** |
| fd_cos 序列 | 0.707582/0.728243/0.951932/0.990387 | 0.707581/0.728244/0.951931/0.990387 | 5 位一致，第 6 位有差 → **非逐位** |
| ‖y_AD‖ | 6.067428e+00（保存 7 位） | 6.0674（console 仅 4 位） | 4 位一致 |
| FP64 基线 | —（未测 FP32 对应项） | loss_rel=2.342e-08, g_cos=1.00000000, g_rel=7.217e-07 | — |
| 向量 abs/rel 差 | — | — | **DATA_MISSING**（向量未保存） |

**勘误**：旧表述「逐位一致/逐位相同」过强。落实为：fd_rel 在 4 位显示精度一致；fd_cos 第 6 位有差
（0.707582 vs 0.707581 等），故**不得称逐位一致**；向量级绝对/相对差因向量未保存 = DATA_MISSING。

## 任务 5 — 动作差补充诊断（dG_action）

定义：dG_action = G(θ0+e^0.05·s_ref0) − G(θ0+s_ref0)；Δα=(e^0.05−1)‖s_ref0‖；
pred_action = Δα·y（y 沿单位方向 v1，故 pred 范数 = Δα·‖y‖）。

| state | 预测器 | Δα=‖Δ位移‖ | ‖pred_action‖ | ‖dG_action‖ | abs_err | rel_err | cos |
|---|---|---|---|---|---|---|---|
| S3 | AD | 5.5366e-03 | 3.3593e-02 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |
| S3 | FD@δ0 | 5.5366e-03 | 1.3863e-02 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |
| S3 | FD@2δ0 | 5.5366e-03 | 7.7002e-03 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |
| S1 | AD | 5.4309e-03 | 6.3394e-03 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |
| S1 | FD@δ0 | 5.4309e-03 | 6.1961e-03 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |
| S1 | FD@2δ0 | 5.4309e-03 | 6.0166e-03 | DATA_MISSING | DATA_MISSING | DATA_MISSING | DATA_MISSING |

**DATA_MISSING 说明**：两个检查点的 ΔG 向量及其交叉内积未保存（仅存各自的范数
0.5382/0.5473 (S3)、0.1307/0.1378 (S1)），按指令**不能从各自范数反推差向量范数**，
故 dG_action 的 norm/abs_err/rel_err/cos 全部 DATA_MISSING。
仅列原 FD 预测作并列参照（AD 与 FD@δ0/@2δ0 的 pred_action 范数上表），不择优、不改模型。
机器可读版：action_difference_diag.csv。

## 任务 6 — 结论限定（替换 v01 结论段）

**可保留**：固定起点线性场（fixed-startpoint linear field）在 **S3 × 原SIM × 首方向** 的
参考一步位移范围内**受限**（S3 AD 预测 rel_err=1.87/1.91；S1 同定义下 0.24/0.25）。

**不得据此宣称**（v01 中过强或易被误读之处一并撤回）：
- ~~非光滑/不可微~~ → 未做光滑性检验，只能称「强非线性/高曲率（fd_rel 尺度依赖显著）」；
- 相变 /  regimes 切换 → 未检验，不宣称；
- 「纯 checkpoint 根因」 → 未定位到任何单一机制，不宣称；
- NLL 或 tokens-to-target 层面的失败 → 本诊断对象是一步位移响应 G，不涉及任何评测集表现，不宣称。

**AD 参照的限定**：AD-HVP 只改进**导数核验**（区分 FD 收敛/未收敛），
**不自动解决有限步预测**——即使导数精确（rel_err→0），线性外推在有限步上的有效性须由任务 3/5
类位移检查单独评估。

## 成本

本轮 v02：0 前向 / 0 反向 / 0 AD-HVP / 0 updates / 0 V_* 调用；纯 CSV 解析。
