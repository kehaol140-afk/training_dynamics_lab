# A0P Learning-rate Response Temporal Kernel Validation

日期：2026-09-18。合同 `contract/contract.json`（sha256 `f7b04169…0aef`）先于任何 GPU 冻结。

## 0. Verdict

- Stage-1 门禁：FAIL（pooled n=84，Pearson +0.7314，sign accuracy 0.9405，rel err median 0.2508）。CRN 全 cell 逐位复现：True。
- Q1 稳定衰减时间尺度：NO （median sign stability 1.000（规则 ≥5/7），median best R² 0.351（≥0.8），CV(τ_c) 3.317（<0.5），n=12）
- Q2 H=4 窗口含长期方向信息：YES （S4/S64 sign agreement 1.000（规则 ≥10/12），Pearson +0.8607（>0））
- Q3 跨 seed/stage 共享时间尺度：NO （CV(τ_c) 3.317，τ_c median 174.20，组中位最大偏离 2.311762125636857，normalized-shape pairwise Pearson median +0.490）
- 离线重算与 driver 一致：None（mismatches: []）
- **STAGE-1 门禁未过**：按冻结合同 STOP，driver Stage 2 未执行。下文第 4–5 节的拟合/记忆度量/Q 规则数值为对**已记录核**的离线描述性诊断（公式与合同冻结一致），用于解释门禁失败机制，**不构成对门禁的绕过或任何阈值调整**。

## 1. 理论定义

脉冲核（mandate 定义）：step t 施加 u=log(η/η₀)（仅此一步，step t+1 起原生调度自然恢复），R(τ) = dL(t+τ)/du|_{u=0}，τ∈{1,2,4,8,16,32,64} 冻结网格。解析式（A0O 物理、源项重定位）：R(τ)=⟨p_th 进入反向迭代 1, lr₁·e_lr(1)⟩，e_lr=−adaptive(c₁)−λ_wd·θ₁（A0O amendment-A2 纯 θ 通道推），p_th 为端点 cotangent dL/dθ_{t+τ} 经 A0H15 生产伴随（adamw_tangent_step_adjoint + clip_transpose + k≥2 fp32 HVP）回溯步 k=τ..2。与 A0O 持久 schedule 输入的关系（线性叠加）：h_persist(H)=Σ_{j<H}K(j)，即持久响应是脉冲核的累积和，H=4 生产响应 ≈ 前 3 个 dyadic 格点核之和 S4——这是短期响应到长期收益的理论桥梁。

## 2. 冻结与实现

- δ=0.05（自 A0O 冻结模块导入，未重定义）；u∈{−δ,0,+δ}；分支 u=0 运行两次做 CRN 逐位校验。
- bank 状态（只读）：seeds {2030,2031,2037} × steps {46,82}（早/晚两阶段），每状态 2 条 stream（7300/7301），共 12 cell。
- stream 内容：`batch_seed=derive_stream_seed(stream_seed,404)` + `batches_from_train_pool`（128 micro-batches = 64 步 × accum 2；单 permutation 前缀，分支共享 → CRN）。provenance 门复用 A0O-A1 键映射（内容不复用）。
- val loss 在每个 grid τ 检查点评估，RNG capture/restore 包裹，不扰动训练轨迹；eval 协议与 A0O/A0K 冻结 v_eval 一致。
- 拟合：scipy curve_fit（maxfev=20000，无 bounds，p0 冻结 A=R(1)/τ_c=8/α=1）；指数 A·e^{−τ/τ_c} 与幂律 A·τ^{−α} 均报 SSE 与 R²，不以拟合选择进入判定。
- 记忆度量（冻结定义）：积分响应 I=Σ|R(τ)|（dyadic 求和）；半衰期=|R|≤|R(1)|/2 的首个 τ（格点间线性插值，未达记 null）；sign stability=sign(R(τ))==sign(R(1)) 的格点占比。
- 判定规则（预先注册于合同）：Q1=median sign stability≥5/7 ∧ median max(R²)≥0.8 ∧ CV(τ_c)<0.5；Q2=sign(S4)==sign(S64)≥10/12 ∧ Pearson(S4,S64)>0；Q3=CV(τ_c)<0.5 ∧ 各 seed/step 组中位偏离≤2×。
- 禁止项执行：无 controller、无 λ 调整、无 optimizer/scheduler 改动、不根据结果选择 τ/δ/状态；12 cell 全部报告。

## 3. Stage-1 数值验证（analytic vs FD）

| 池 | n | Pearson | sign acc | rel err median |
|---|---|---|---|---|
| pooled | 84 | +0.7314 | 0.9405 | 0.2508 |
| τ=1 | 12 | +0.7241 | 1.0000 | 0.1555 |
| τ=2 | 12 | +0.4263 | 1.0000 | 0.3020 |
| τ=4 | 12 | +0.8881 | 1.0000 | 0.1478 |
| τ=8 | 12 | +0.8892 | 1.0000 | 0.3083 |
| τ=16 | 12 | +0.3733 | 0.8333 | 0.3482 |
| τ=32 | 12 | +0.9141 | 0.9167 | 0.2330 |
| τ=64 | 12 | +0.7009 | 0.8333 | 0.3698 |

## 4. 脉冲核与拟合（per cell）

| seed | step | stream | R(1) | R(4) | R(16) | R(64) | τ_c(exp) | R²exp | α(pow) | R²pow | I=Σ|R| | half-life | sign stab |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2030 | 46 | 7300 | -3.079e-03 | -3.101e-03 | +1.145e-04 | +6.548e-04 | 5.05 | 0.814 | 0.729 | 0.657 | 1.071e-02 | 6.1 | 0.43 |
| 2030 | 46 | 7301 | -2.484e-03 | -1.858e-03 | +2.339e-03 | +2.269e-03 | 5.75 | 0.401 | 0.948 | 0.305 | 1.625e-02 | not_reached | 0.57 |
| 2030 | 82 | 7300 | -2.820e-03 | -1.447e-03 | -2.029e-03 | -1.978e-03 | 2513.57 | 0.001 | 0.073 | 0.147 | 1.351e-02 | 4.7 | 1.00 |
| 2030 | 82 | 7301 | -3.206e-03 | -1.398e-03 | -1.477e-03 | -1.837e-03 | 253.28 | 0.064 | 0.155 | 0.423 | 1.386e-02 | 3.6 | 1.00 |
| 2031 | 46 | 7300 | -4.273e-03 | -3.831e-03 | -1.073e-03 | +4.127e-03 | 22.77 | 0.389 | 0.439 | 0.307 | 2.281e-02 | 7.9 | 0.86 |
| 2031 | 46 | 7301 | -4.550e-03 | -3.582e-03 | -2.709e-03 | -3.524e-03 | 909.45 | 0.008 | 0.099 | 0.263 | 2.327e-02 | 6.9 | 1.00 |
| 2031 | 82 | 7300 | -3.461e-03 | -1.813e-03 | -2.899e-03 | -2.104e-03 | 244.37 | 0.067 | 0.124 | 0.271 | 1.605e-02 | 4.6 | 1.00 |
| 2031 | 82 | 7301 | -2.947e-03 | -1.936e-03 | -2.107e-03 | -2.778e-03 | 2069028399.39 | -0.000 | 0.025 | 0.021 | 1.630e-02 | 7.5 | 1.00 |
| 2037 | 46 | 7300 | -6.398e-03 | -1.838e-03 | -3.593e-03 | -3.462e-03 | 177.45 | 0.075 | 0.120 | 0.164 | 2.727e-02 | 2.2 | 1.00 |
| 2037 | 46 | 7301 | -8.727e-03 | -6.080e-03 | -4.989e-03 | -4.740e-03 | 116.99 | 0.303 | 0.152 | 0.556 | 3.883e-02 | 23.2 | 1.00 |
| 2037 | 82 | 7300 | -4.285e-03 | -2.072e-03 | -3.252e-03 | -2.396e-03 | 170.96 | 0.058 | 0.199 | 0.313 | 1.756e-02 | 3.9 | 1.00 |
| 2037 | 82 | 7301 | -4.292e-03 | -1.987e-03 | -1.605e-03 | -2.012e-03 | 64.52 | 0.221 | 0.323 | 0.755 | 1.570e-02 | 3.7 | 1.00 |

## 5. Q1/Q2/Q3 与 response temporal law（gate FAIL 后的描述性诊断）

- **Q1** 未通过：median sign stability 1.000，median best-fit R² 0.351，CV(τ_c)=3.317（τ_c 值见上表）。
  两 law 拟合优度中位 R²：exp 0.071 vs pow 0.306（描述性比较，不进入判定）。
- **Q2** 通过：S4（dyadic 窗 {1,2,4} 累积核，对应生产 H=4 响应的理论对应量）与 S64（全网格累积）sign agreement 1.000（12/12），Pearson +0.8607。即短期响应方向对长期方向的预测力。
- **Q3** 未通过：组中位 τ_c：{2030: 129.51535358578116, 2031: 576.9134489372144, 2037: 143.97320671560595, 'step46': 69.87914840289872, 'step82': 248.8288920204692}；归一化核形 pairwise Pearson 中位 +0.490。

**Response temporal law（本轮结论）**：LR 脉冲响应核随 τ 的衰减律 （未达稳定时间尺度判据；12 cell 中位形状更接近 power law），积分记忆 I 与半衰期见表；H=4 窗口 包含 长期方向信息；时间尺度 不共享。

## 6. 预算与制品

- GPU：864.9s / 1800s。budget log 中 n_crashes=1 为门禁 FAIL 标记（driver 惯例 crashed=not PASS），非进程崩溃——12 cell 全部正常完成。逐 cell 计时见 gpu_budget_log.json（append-only）。
- 制品：`outputs/a0p_lr_temporal_kernel_v01/`（contract/ runs/ analysis/ reports/ hashes/ audit/）；代码 `research/src/fr/a0p_lr_temporal_kernel_v01.py` + `research/scripts/a0p_{run,analyze,audit_pack}_v01.py`。
- A0O 及全部更早冻结制品未动；docs append-only。

## 7. 方法指纹

- `research/src/fr/a0p_lr_temporal_kernel_v01.py` sha256 `87f959db60da510d…`
- `research/scripts/a0p_run_v01.py` sha256 `efa6f18fcd1b6b9e…`
- `research/scripts/a0p_analyze_v01.py` sha256 `afb10813d8a5bd57…`
- `research/scripts/a0p_audit_pack_v01.py` sha256 `9d4dab0ee07d89c8…`
- `outputs/a0p_lr_temporal_kernel_v01/contract/contract.json` sha256 `f7b04169d371a56f…`
