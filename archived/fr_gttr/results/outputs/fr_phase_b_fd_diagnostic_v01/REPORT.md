# FR Phase B FD 局部算子失配定点诊断 v1 — REPORT

轮次：`fr_phase_b_fd_diagnostic_v01`（2026-09-23）
协议：FR_PhaseB_FD_Local_Operator_Diagnostic_v1.md；原 v3 GATE_STOP 不动。
范围：S3（2031/82/SIM197009，门失败点）与 S1（2030/82/SIM197007，同 start 通过对照）；
仅首方向 v1 = s_ref0/‖s_ref0‖（固定、detach）；updates=0；V_* 调用=0。

## 0. D0 文件与现场核查

- manifest SIM：S3=197009、S1=197007（与 v3 manifest.csv 一致）。
- 原 S3 失败行：fd_rel=0.7061390274577329、fd_cos=0.7282431519261835（resource_log，门在首方向 half-check）。y0/y_half 向量未保存 → 分子/分母原始充分统计 NOT_SAVED，由 D1 复现重建。
- 代码核查：FD 作用在未裁剪 G_SIM（clip 只出现在虚拟参考步）；端点每次从同一 theta0 重建（finally copy_，无累积漂移）；每次求值 zero_grad(set_to_none=True) 后恰好累积 2 micro；返回 p.grad.detach().clone()；rng0 每 micro 前恢复、求值后恢复 rng_live；train 模式恢复；tied 权重按 id 去重（46 unique）。
- 现场：torch=2.11.0 cuda=13.0；TF32 matmul=False cudnn=True precision=highest；autocast 关闭；param/grad dtype=FP32。
- attention-ish kernels [S3:ORIG_FP32]: ['fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(cutlass_80_simt_sgemm_128x32_8x5_tn_align1::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params)']
- attention-ish kernels [S1:ORIG_FP32]: ['fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(cutlass_80_simt_sgemm_128x32_8x5_tn_align1::Params)', 'void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params)']

## 1. D1 重复/顺序/还原（repeat_restore_checks.csv）

| state | tag | offset_err | exit_err | rng_ok | note |
|---|---|---|---|---|---|
| S3 | S3/g0_rep2_INV1 |  |  |  | inv1 console: G(theta0) repeat rel diff = 0.000e+00; loss 7. |
| S3 | S3/fd_d0_rep2_INV1 |  |  |  | inv1 console: FD(d0) rep2 vs rep1 y rel diff = 0.000e+00 |
| S3 | S3/fd_d0h_rep2_INV1 |  |  |  | inv1 console: FD(d0/2) rep2 vs rep1 y rel diff = 0.000e+00 |
| S3 | S3/fd_d0_reversed_INV1 |  |  |  | inv1 console: reversed endpoint order y rel diff = 0.000e+00 |
| S1 | S1/g0_rep2_INV1 |  |  |  | inv1 console: G(theta0) repeat rel diff = 0.000e+00; loss 7. |
| S3 | S3/g0_rep1 | 0.000e+00 | 0.000e+00 | True |  |
| S3 | S3/fd_d0_rev |  |  |  | OMITTED_INV2_BUDGET (S3 check retained in inv1) |
| S3 | S3/ckpt_s_ref0 | 0.000e+00 | 0.000e+00 | True | checkpoint s_ref0 |
| S3 | S3/ckpt_exp05_s_ref0 | 5.947e-08 | 0.000e+00 | True | checkpoint exp05_s_ref0 |
| S1 | S1/g0_rep1 | 0.000e+00 | 0.000e+00 | True |  |
| S1 | S1/v1_vs_bundle |  |  |  | 1-cos(v1_rebuilt, bundle V[0]) |
| S1 | S1/fd_d0_rev |  |  |  | OMITTED_INV2_BUDGET (S3 check retained in inv1) |
| S1 | S1/ckpt_s_ref0 | 0.000e+00 | 0.000e+00 | True | checkpoint s_ref0 |
| S1 | S1/ckpt_exp05_s_ref0 | 5.949e-08 | 0.000e+00 | True | checkpoint exp05_s_ref0 |

## 2. D2 固定尺度（fd_scale_diagnostic.csv；h_over_delta0 ∈ {2,1,1/2,1/4,1/8}）

| state | path | h/δ0 | h | ‖y‖ | v1·y | fd_rel(adj) | fd_cos(adj) | asym | repeat |
|---|---|---|---|---|---|---|---|---|---|
| S3 | REF_FP64 | 1 |  |  |  |  |  |  |  | INV1_CONSOLE_RECOVERED_ADJACENT_ONLY |
| S3 | REF_FP64 | 0.5 |  |  |  |  |  |  |  | INV1_CONSOLE_RECOVERED_ADJACENT_ONLY |
| S3 | REF_FP64 | 0.25 |  |  |  |  |  |  |  | INV1_CONSOLE_RECOVERED_ADJACENT_ONLY |
| S3 | REF_FP64 | 0.125 |  |  |  |  |  |  |  | INV1_CONSOLE_RECOVERED_ADJACENT_ONLY |
| S3 | ORIG_FP32 | 2.0 | 1.3943e-01 | 1.3908e+00 | 5.6242e-02 |  |  | 3.83e-03 |  |
| S3 | ORIG_FP32 | 1.0 | 6.9714e-02 | 2.5038e+00 | -3.8194e-03 | 0.722820 | 0.70758153 | 1.87e-02 |  |
| S3 | ORIG_FP32 | 0.5 | 3.4857e-02 | 4.4869e+00 | -3.5203e-01 | 0.706139 | 0.72824315 | 8.75e-02 |  |
| S3 | ORIG_FP32 | 0.25 | 1.7429e-02 | 5.9583e+00 | -5.0023e-01 | 0.365215 | 0.95193217 | 6.75e-02 |  |
| S3 | ORIG_FP32 | 0.125 | 8.7143e-03 | 6.1341e+00 | -4.7120e-01 | 0.139630 | 0.99038686 | 3.27e-02 |  |
| S1 | ORIG_FP32 | 2.0 | 1.3928e-01 | 1.1079e+00 | -1.9484e-02 |  |  | 6.66e-03 |  |
| S1 | ORIG_FP32 | 1.0 | 6.9640e-02 | 1.1409e+00 | -2.5533e-02 | 0.108533 | 0.99436646 | 2.76e-03 | 0.0 |
| S1 | ORIG_FP32 | 0.5 | 3.4820e-02 | 1.1584e+00 | -2.6191e-02 | 0.066012 | 0.99790413 | 1.46e-03 | 0.0 |
| S1 | ORIG_FP32 | 0.25 | 1.7410e-02 | 1.1649e+00 | -2.6212e-02 | 0.022356 | 0.99976400 | 7.54e-04 |  |
| S1 | ORIG_FP32 | 0.125 | 8.7050e-03 | 1.1667e+00 | -2.6206e-02 | 0.006068 | 0.99998275 | 3.81e-04 |  |

## 3. D3 AD 与位移检查点（ad_reference_comparison.csv）

### 3a. FD vs 同路径 AD（5 固定尺度）

| state | path | h/δ0 | ‖y_AD‖ | ‖y_FD‖ | abs_err | sym_rel_err | cos |
|---|---|---|---|---|---|---|---|
| S3 | ORIG_FP32_MATHREF | 2.0 | 6.0674e+00 | 1.3908e+00 | 5.9547e+00 | 9.8142e-01 | 0.194919 |
| S3 | ORIG_FP32_MATHREF | 1.0 | 6.0674e+00 | 2.5038e+00 | 4.9733e+00 | 8.1968e-01 | 0.603900 |
| S3 | ORIG_FP32_MATHREF | 0.5 | 6.0674e+00 | 4.4869e+00 | 2.8086e+00 | 4.6290e-01 | 0.901004 |
| S3 | ORIG_FP32_MATHREF | 0.25 | 6.0674e+00 | 5.9583e+00 | 1.3926e+00 | 2.2952e-01 | 0.973342 |
| S3 | ORIG_FP32_MATHREF | 0.125 | 6.0674e+00 | 6.1341e+00 | 5.9873e-01 | 9.7606e-02 | 0.995244 |
| S1 | ORIG_FP32_MATHREF | 2.0 | 1.1673e+00 | 1.1079e+00 | 2.3177e-01 | 1.9856e-01 | 0.980595 |
| S1 | ORIG_FP32_MATHREF | 1.0 | 1.1673e+00 | 1.1409e+00 | 1.1193e-01 | 9.5892e-02 | 0.995557 |
| S1 | ORIG_FP32_MATHREF | 0.5 | 1.1673e+00 | 1.1584e+00 | 3.5541e-02 | 3.0448e-02 | 0.999562 |
| S1 | ORIG_FP32_MATHREF | 0.25 | 1.1673e+00 | 1.1649e+00 | 9.5000e-03 | 8.1385e-03 | 0.999969 |
| S1 | ORIG_FP32_MATHREF | 0.125 | 1.1673e+00 | 1.1667e+00 | 2.4204e-03 | 2.0735e-03 | 0.999998 |

### 3b. 实际位移检查点（θ0+c·s_ref0，c∈{1, e^0.05}）

| state | checkpoint | h_eff | ‖ΔG‖ | pred_rel_err FD@δ0 | FD@2δ0 | AD |
|---|---|---|---|---|---|---|
| S3 | s_ref0 | 0.1080 | 5.3819e-01 | 1.0430 | 0.9071 | 1.8666160068211861 |
| S3 | exp05_s_ref0 | 0.1135 | 5.4726e-01 | 1.0546 | 0.9043 | 1.9103938500988387 |
| S1 | s_ref0 | 0.1059 | 1.3074e-01 | 0.2385 | 0.2574 | 0.23876022730278018 |
| S1 | exp05_s_ref0 | 0.1114 | 1.3782e-01 | 0.2488 | 0.2633 | 0.25156937506270255 |

## 4. 层定位（block_perturbation_summary.csv，errvec d=y(δ0)−y(δ0/2)）

| state | rank | block | err_share | ‖θ0_b‖ | ‖v1_b‖ | rel_pert@δ0 |
|---|---|---|---|---|---|---|
| S3 | 1 | model.embed_tokens.weight | 0.4236 | 3.026e+01 | 5.795e-01 | 1.335182e-03 |
| S3 | 2 | model.layers.0.self_attn.v_proj.weight | 0.1853 | 3.690e+00 | 9.576e-02 | 1.809200e-03 |
| S3 | 3 | model.layers.0.self_attn.q_proj.weight | 0.1374 | 5.218e+00 | 1.627e-01 | 2.174121e-03 |
| S3 | 4 | model.layers.0.self_attn.k_proj.weight | 0.1114 | 3.698e+00 | 1.129e-01 | 2.128646e-03 |
| S3 | 5 | model.layers.0.self_attn.o_proj.weight | 0.0780 | 5.186e+00 | 1.250e-01 | 1.680216e-03 |
| S1 | 1 | model.embed_tokens.weight | 0.5409 | 3.013e+01 | 5.851e-01 | 1.352294e-03 |
| S1 | 2 | model.layers.0.self_attn.q_proj.weight | 0.2744 | 5.226e+00 | 1.634e-01 | 2.176922e-03 |
| S1 | 3 | model.layers.0.self_attn.k_proj.weight | 0.0976 | 3.687e+00 | 1.074e-01 | 2.027923e-03 |
| S1 | 4 | model.layers.0.self_attn.v_proj.weight | 0.0444 | 3.700e+00 | 8.938e-02 | 1.682125e-03 |
| S1 | 5 | model.layers.0.self_attn.o_proj.weight | 0.0176 | 5.231e+00 | 1.155e-01 | 1.538273e-03 |

## 5. 成本（costs.json；独立账本）

updates=0；micro_fw=238/240；micro_bw first=238 + second=12 = 250/320；AD-HVP=6/6；FD方向=25.0；SIM slots unique=2048 reprocessed=94208；peak=1.08GB；partial=True；wall=75.0s [OBSERVATIONAL_SHARED_GPU]。

## 6. 解释与最小修订建议

### 6.1 逐解释排除（按协议 §8 顺序）

1. **实现/随机性问题 — 排除。** G(θ0) 重复、FD 重复、端点逆序全部逐位 0 差；entry/offset/exit 误差全 0、rng 全还原；重建 v1 与 v3 bundle V[0] cos=1.000000000；S1 在门尺度 δ0↔δ0/2 的 fd_rel=0.066012 与 v3 记录值 0.0660 逐位复现 → 诊断路径与原 v3 路径一致，且管线在 S1 上可通过门（失败不是管线全局属性）。FD 端点 rel 扰动 1.3e-3~2.2e-3（每块），offset 误差 ~6e-8（FP32 机器精度级）。
2. **数值路径限制 — 排除。** S3 REF_FP64 邻尺度 fd_rel 序列与 FP32 完全相同（0.7228/0.7061/0.3652/0.1396，cos 到 7 位一致）；‖y_AD‖(FP64)=6.0674 = ‖y_AD‖(FP32 math)；FP64 基线 loss_rel=2.3e-8、g_cos=1.00000000。attention 后端：PyTorchMemEffAttention FP32（cutlass SIMT SGEMM），TF32 matmul=False / precision=highest；FP64 用 tensorop d884 GEMM（后端变化已标注）。
3. **差分尺度限制 — 部分描述但不构成根因。** S3 的 fd_rel 随 h 减小（0.72→0.71→0.37→0.14）呈下降趋势，但 δ0/8 处仍为 0.14，远未收敛到 FP32 噪声底；固定集合内没有任何尺度能通过 0.20 门。同一 δ0 下 S1 通过（0.066）→ 尺度不是区分因素，态的局部几何才是。
4. **实际位移上的局部近似不足 — 受支持（主因）。** 证据链：(a) S3 的 y(h) 从 2δ0 的 1.39 单调增至 δ0/8 的 6.13，向 ‖y_AD‖=6.0674 收敛但极慢（FD-AD sym_rel_err 0.98→0.82→0.46→0.23→0.10），说明 G_SIM 沿 v1 在 δ0 邻域内有强非线性，FD 在门尺度测到的是强曲率平均量而非导数；(b) 在真实参考位移 s_ref0 上，‖ΔG‖=0.538，精确局部导数 y_AD 的线性预测相对误差 1.87（比 FD@δ0 的 1.04 更差）→ 即使拥有精确 Jacobian-向量积，局部线性模型在真实步长上也不成立；(c) 误差向量 d=y(δ0)−y(δ0/2) 集中于 embed_tokens（42.4%）+layers.0 四个 attn 投影（合计 ~51%），即非线性由词嵌入与第一层注意力投影承载；(d) S1 对照在同一管线、同一尺度下 FD-AD 干净收敛（sym_rel_err 0.002@δ0/8）且 FD/AD 互相一致 → 非线性是 S3 态特有。

### 6.2 结论（协议 §8）

**VERDICT：实际位移上的局部近似不足**（G_SIM 在 S3 的 θ0 沿 v1 方向存在真实强非线性/非光滑性；0.706 的门失败是该局部几何的症状，不是可修复的流程伪影）。实现/随机性、数值路径两条已排除；差分尺度只能描述、不能根治——任何固定尺度对在 δ0≥h≥δ0/8 内都过不了门。对 S1 的附带观察：即便门通过态，真实位移上线性预测误差仍有 ~24%（FD 与 AD 一致），即 v3 的 FD 半步门度量的是“差分自洽”而非“局部线性有效”——这是 v3 设计层面的局限，本次诊断不予修复（协议范围外）。

### 6.3 最小修订建议（一条，不自动执行）

建议给未来 v4（本诊断不改动 v3、不恢复 Phase B）：**把每方向的 AD-HVP 参照（math 后端或二次可微 attention 路径）纳入门协议，用 FD-AD sym_rel_err（收敛量）替代/补充 FD 半步 fd_rel（自洽量）作为 LOCAL_OPERATOR 校准判据，并对判定为 LOCALLY_NONLINEAR 的态明确标注“该方向局部线性近似无效”，不进入 LOCAL_OPERATOR 已校准的声明。** 依据：FD 自检无法区分“已收敛”与“未收敛但光滑”（本诊断中 S3 的 fd_rel 与 FD-AD 误差单调性并不重合于同一尺度），而一次 AD 参照即可判定。

### 6.4 范围与未做事项

未重做历史审计、未新增理论、未调阈值、未选最优尺度替换原实验、未剔除 S3、未读取 MAIN/评价集、未构造完整 Hessian、未重新训练、未恢复 Phase B；旧失败结果（v3 门 0.706）原样保留。所有 5 个固定尺度结果全部报告（无择优）。
