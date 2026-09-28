# A0O Learning-Rate Input Response Validation — REPORT

## 0. Verdict

STAGE 1 PASS -> Stage 2 executed. See tables below.

## 1. 理论定义

输入通道：optimizer schedule 的乘性 log-LR 扰动 u = log(eta/eta0)，
eta_k(u) = eta_k * exp(u)，在 H 步未来 rollouts 的**每一步**生效
（persistent schedule input，不是 one-shot pulse，区别于 A0K data injection）。
u in {-delta, 0, +delta}，delta = 0.05（结果前冻结）。

真实 AdamW 更新（decoupled weight decay included，mandate 要求）：

    theta+ = (1 - lr*lambda_wd)*theta - lr*adaptive(c)
    adaptive(c) = (m1/bc1) / (sqrt(v1/bc2) + eps)

u 通道切线（链式法则 d lr/du = lr at u=0）：

    d theta_{k+1} / du = lr_k * e_lr(k) ,   e_lr = -adaptive(c_k) - lambda_wd * theta_k

关键结构事实：m1/v1 不依赖 lr（被消费的梯度 c_k 与 u 无关），因此 LR pulse 是
**纯 theta 通道推动**（pulse 步 dm1=dv1=0），不是 A0K data injection 的 gradient-frame shift。

Analytic response: h_eta = sum_k <p_th^{(k)}, lr_k*e_lr(k)>，其中 p_th^{(k)} 为
reverse 迭代 k 入口的 theta-cotangent（= dL/d(theta_{k+1})，post-step-k 状态），
由 frozen reverse adjoint（adamw_tangent_step_adjoint + clipping transpose + fp32 future HVP，
与 A0H15 production 完全一致）产生；后续步的 gradient/m/v 通道效应由该 adjoint 链自动携带。
一阶 exact（tangent）；O(u^2) 项由 FD 参考覆盖/排除。

FD 参考: (L(+delta) - L(-delta)) / (2*delta)，L = endpoint theta_{t+H}(u) 的 V_score。

## 2. 实现细节

- 核心模块 research/src/fr/a0o_lr_response_v01.py：`lr_pulse` 由 snapshot 的
  exp_avg/exp_avg_sq/clipped_gradient/theta_start/lr/wd/betas 计算单位 shift
  e_lr = -adaptive(c) - wd*theta；`run_lr_response` 镜像 A0H15 run_adjoint_only
  （forward snapshot -> v_H -> reverse adjoint），在每次 reverse 迭代入口用
  p_th（= dL/d(theta_{k+1})）与 lr_k*e_lr(k)  contraction 累加 h_eta；
  k>=2 时恢复 snap theta_start + rng 做 fp32 HVP，链式回传至 p_th。
- `run_lr_rollout`：每个 horizon 的 native forward/backward 后把 param_groups 的 lr
  改为 snap['lr']*exp(u) 再 _clip_and_step；u=0 分支与 baseline bitwise 一致（CRN）。
- Driver research/scripts/a0o_run_v01.py：Stage 1 = 12 states x 4 streams x H{1,4} = 96 cells
  （states = v0.7 bank seeds 2030/2031/2037 x steps 46/52/64/82，即 A0C locked states；
  A0K event key 仅作 provenance gate（amendment A1：2030/2031 同 seed、2037->2032、step 降至 10 的倍数）；
  每 stream 的 lookahead 内容由 frozen recipe 生成（amendment A2：batch_seed = derive_stream_seed(stream_seed, 404)，
  batches_from_train_pool(train_indices, bs, batch_seed, 8*accum)，与 A0K manifest builder 同推导）；
  每 cell 跑 4 次 rollout (u=+d, 0, -d, 0) 以提供 FD + 曲率 sanity + CRN bitwise check。
- Stage-1 gate: pooled |Pearson| >= 0.8 AND sign accuracy >= 0.9；FAIL -> STOP。
- Stage 2: states (2030, 46/64/82) x 8 streams (7200-7207), H=4；
  actions = increase(u=+d)/baseline(u=0)/decrease(u=-d)；prediction = sign(h_eta) -> argmin_u h_eta*u；
  metrics = direction accuracy / median regret vs realized optimum / calibration Pearson / null=zero-response。
- Contract frozen before any GPU: contract.json (amendments A1/A2) sha256 c2399d1ad9e5d69b...; live_manifest module hash notes seq 0-11。

## 3. 数值验证 (Stage 1)

Gate: **PASS** (frozen rule: |Pearson|>=0.8 AND sign_acc>=0.9, pooled)。

| pool | n | Pearson | sign acc | rel err median | rel err p90 |
---|---|---|---|---|---|
| all H | 96 | +0.9552 | 0.9688 | 0.2250 | 0.8219 |
| H=1 | 48 | +0.9550 | 0.9792 | 0.2661 | 0.7541 |
| H=4 | 48 | +0.9517 | 0.9583 | 0.1671 | 0.8955 |

- CRN bitwise (second u=0 rollout == first, all cells): **OK**
- |h_eta| median = 5.3178e-03; |FD| median = 5.2938e-03 (LR 通道在 delta=0.05 下的 score-scale 响应幅度，供闭环设计参考)。

### per-cell (H=4, all 96 stage-1 cells 见 runs/stage1_cells.jsonl)

| seed | step | stream | H | h_eta | FD | rel err | CRN |
---|---|---|---|---|---|---|---|
| 2030 | 46 | 7110 | 4 | +2.73899e-02 | +2.63584e-02 | 0.039 | OK |
| 2030 | 46 | 7111 | 4 | -1.31345e-02 | -1.41305e-02 | 0.070 | OK |
| 2030 | 46 | 7112 | 4 | -1.27050e-02 | -1.14012e-02 | 0.114 | OK |
| 2030 | 46 | 7113 | 4 | -1.13214e-02 | -1.48034e-02 | 0.235 | OK |
| 2030 | 52 | 7110 | 4 | -1.32695e-02 | -1.28597e-02 | 0.032 | OK |
| 2030 | 52 | 7111 | 4 | -1.41649e-02 | -9.78887e-03 | 0.447 | OK |
| 2030 | 52 | 7112 | 4 | -1.29160e-02 | -1.55300e-02 | 0.168 | OK |
| 2030 | 52 | 7113 | 4 | -1.30548e-02 | -1.26630e-02 | 0.031 | OK |
| 2030 | 64 | 7110 | 4 | +1.99627e-03 | +4.79221e-03 | 0.583 | OK |
| 2030 | 64 | 7111 | 4 | -1.88061e-04 | +2.30908e-03 | 1.081 | OK |
| 2030 | 64 | 7112 | 4 | -9.92924e-03 | -8.93950e-03 | 0.111 | OK |
| 2030 | 64 | 7113 | 4 | -9.01621e-03 | -6.41167e-03 | 0.406 | OK |
| 2030 | 82 | 7110 | 4 | -7.77196e-03 | -8.39889e-03 | 0.075 | OK |
| 2030 | 82 | 7111 | 4 | -9.42750e-03 | -7.97093e-03 | 0.183 | OK |
| 2030 | 82 | 7112 | 4 | -4.35516e-03 | -6.70671e-03 | 0.351 | OK |
| 2030 | 82 | 7113 | 4 | -7.48941e-03 | -7.15256e-03 | 0.047 | OK |
| 2031 | 46 | 7110 | 4 | -1.13053e-02 | -8.40127e-03 | 0.346 | OK |
| 2031 | 46 | 7111 | 4 | -8.02064e-03 | -1.24890e-02 | 0.358 | OK |
| 2031 | 46 | 7112 | 4 | -4.02612e-04 | +1.04368e-03 | 1.386 | OK |
| 2031 | 46 | 7113 | 4 | -1.55677e-02 | -1.50734e-02 | 0.033 | OK |
| 2031 | 52 | 7110 | 4 | -9.43633e-03 | -6.16252e-03 | 0.531 | OK |
| 2031 | 52 | 7111 | 4 | -7.66184e-03 | -3.62098e-03 | 1.116 | OK |
| 2031 | 52 | 7112 | 4 | -2.82512e-03 | -3.84748e-03 | 0.266 | OK |
| 2031 | 52 | 7113 | 4 | -7.81320e-03 | -7.29680e-03 | 0.071 | OK |
| 2031 | 64 | 7110 | 4 | -4.01232e-03 | -4.60923e-03 | 0.130 | OK |
| 2031 | 64 | 7111 | 4 | -7.71013e-03 | -9.64463e-03 | 0.201 | OK |
| 2031 | 64 | 7112 | 4 | -2.43578e-03 | -3.76105e-03 | 0.352 | OK |
| 2031 | 64 | 7113 | 4 | -8.55128e-03 | -5.18143e-03 | 0.650 | OK |
| 2031 | 82 | 7110 | 4 | -9.61429e-03 | -1.12152e-02 | 0.143 | OK |
| 2031 | 82 | 7111 | 4 | -9.73496e-03 | -1.08516e-02 | 0.103 | OK |
| 2031 | 82 | 7112 | 4 | -9.25249e-03 | -9.88722e-03 | 0.064 | OK |
| 2031 | 82 | 7113 | 4 | -9.39400e-03 | -9.17673e-03 | 0.024 | OK |
| 2037 | 46 | 7110 | 4 | -7.72751e-03 | -3.22163e-03 | 1.399 | OK |
| 2037 | 46 | 7111 | 4 | -1.24284e-03 | -6.74665e-03 | 0.816 | OK |
| 2037 | 46 | 7112 | 4 | -1.37649e-02 | -1.53309e-02 | 0.102 | OK |
| 2037 | 46 | 7113 | 4 | -8.66616e-03 | -6.46591e-03 | 0.340 | OK |
| 2037 | 52 | 7110 | 4 | -1.70459e-02 | -1.78689e-02 | 0.046 | OK |
| 2037 | 52 | 7111 | 4 | -2.05993e-02 | -2.04778e-02 | 0.006 | OK |
| 2037 | 52 | 7112 | 4 | -1.60659e-02 | -1.83147e-02 | 0.123 | OK |
| 2037 | 52 | 7113 | 4 | -1.94943e-02 | -1.87820e-02 | 0.038 | OK |
| 2037 | 64 | 7110 | 4 | -7.83335e-03 | -8.88944e-03 | 0.119 | OK |
| 2037 | 64 | 7111 | 4 | -5.50042e-03 | -6.87659e-03 | 0.200 | OK |
| 2037 | 64 | 7112 | 4 | -2.65065e-03 | -2.60174e-03 | 0.019 | OK |
| 2037 | 64 | 7113 | 4 | -1.05887e-02 | -9.49919e-03 | 0.115 | OK |
| 2037 | 82 | 7110 | 4 | -1.96094e-03 | -3.99947e-03 | 0.510 | OK |
| 2037 | 82 | 7111 | 4 | -2.49806e-03 | -2.99454e-03 | 0.166 | OK |
| 2037 | 82 | 7112 | 4 | -4.74690e-03 | -2.21372e-03 | 1.144 | OK |
| 2037 | 82 | 7113 | 4 | -4.53818e-03 | -8.14140e-03 | 0.443 | OK |

## 4. 有限动作预测 (Stage 2)

- cells: 24 (states 2030 x steps 46/64/82, 8 streams each, H=4)
- direction accuracy: **0.875** (null always-baseline: 0.125)
- realized best-action counts: increase 17 / baseline 3 / decrease 4 of 24
- median regret (chosen - optimal realized): **0.000e+00** vs null 4.803e-04 -> better_than_null = **True**
- calibration Pearson (corrected, chosen-action branch): **+0.942** (driver<=seq9 误报 -0.250：单边分支取反，见 analysis/stage2_metrics_corrected.json)
- stage-2 subset central-FD Pearson: +0.984; curvature median (L(+d)+L(-d)-2L(0)) = -1.335e-04 (小,一阶主导)

| seed | step | stream | h_eta | L(-d) | L(0) | L(+d) | CRN |
---|---|---|---|---|---|---|---|
| 2030 | 46 | 7200 | -1.01124e-02 | 7.265865 | 7.265357 | 7.264731 | OK |
| 2030 | 46 | 7201 | -1.35474e-02 | 7.266037 | 7.265552 | 7.265056 | OK |
| 2030 | 46 | 7202 | -1.09331e-02 | 7.268623 | 7.268039 | 7.267545 | OK |
| 2030 | 46 | 7203 | -1.57445e-02 | 7.261357 | 7.260719 | 7.259944 | OK |
| 2030 | 46 | 7204 | -1.42181e-02 | 7.264860 | 7.263988 | 7.263303 | OK |
| 2030 | 46 | 7205 | -1.60509e-02 | 7.264681 | 7.263890 | 7.262949 | OK |
| 2030 | 46 | 7206 | -9.13146e-03 | 7.269968 | 7.269655 | 7.269277 | OK |
| 2030 | 46 | 7207 | -1.70202e-03 | 7.264306 | 7.264497 | 7.264147 | OK |
| 2030 | 64 | 7200 | +1.90028e-02 | 7.183640 | 7.184553 | 7.185479 | OK |
| 2030 | 64 | 7201 | -8.11576e-04 | 7.181062 | 7.180894 | 7.180920 | OK |
| 2030 | 64 | 7202 | +5.08898e-04 | 7.184310 | 7.184282 | 7.184348 | OK |
| 2030 | 64 | 7203 | +1.36840e-03 | 7.178136 | 7.178412 | 7.178492 | OK |
| 2030 | 64 | 7204 | -2.23753e-03 | 7.181726 | 7.181246 | 7.181328 | OK |
| 2030 | 64 | 7205 | +2.17002e-03 | 7.185787 | 7.186064 | 7.186098 | OK |
| 2030 | 64 | 7206 | +6.08089e-03 | 7.189752 | 7.190408 | 7.190474 | OK |
| 2030 | 64 | 7207 | -8.91465e-04 | 7.181870 | 7.181874 | 7.181746 | OK |
| 2030 | 82 | 7200 | -7.97910e-03 | 7.122787 | 7.122647 | 7.122058 | OK |
| 2030 | 82 | 7201 | -4.11731e-03 | 7.125969 | 7.125707 | 7.125360 | OK |
| 2030 | 82 | 7202 | -7.01760e-03 | 7.126784 | 7.126528 | 7.125963 | OK |
| 2030 | 82 | 7203 | -8.26250e-03 | 7.123571 | 7.123375 | 7.122752 | OK |
| 2030 | 82 | 7204 | -5.74220e-03 | 7.127610 | 7.127352 | 7.127113 | OK |
| 2030 | 82 | 7205 | -7.81164e-03 | 7.126199 | 7.125866 | 7.125400 | OK |
| 2030 | 82 | 7206 | -9.33740e-03 | 7.125216 | 7.124929 | 7.124361 | OK |
| 2030 | 82 | 7207 | -6.14166e-03 | 7.126097 | 7.125872 | 7.125493 | OK |

## 5. Q1 / Q2 / Q3

- **Q1** (LR-input response 存在吗？): YES — Stage 1 gate PASS: pooled Pearson r = +0.9552, sign accuracy = 0.9688, analytic h_eta 与 central FD 在全部 96 cells 上一致 (CRN bitwise OK)。
- **Q2** (比 random/fixed 更有信息量吗？): Stage 2: direction accuracy = 0.875 (null = 0.125); median regret = 0.000e+00 vs null 4.803e-04 (better than null: True); calibration Pearson (corrected, chosen-branch) = +0.942; stage-2 subset central-FD Pearson = +0.984。
- **Q3** (相对 data response 是新控制维度吗？): 存在一个新的控制维度（optimizer input），但 Stage-2 证据显示其预测力 强于 always-baseline null。

## 6. 是否值得进入闭环控制

值得探索闭环控制，但仅作为 pilot 且必须 equal wall-clock；禁止根据本结果修改 delta / 添加 threshold / phase。

纪律：delta=0.05 冻结；未修改 optimizer/scheduler/lambda；无 threshold/phase 添加；
未训练任何 controller；Stage 3 (closed-loop A/B/C arms) 本报告不启动，需显式授权。

## 7. Budget

gpu_budget_log TOTALS: {"cumulative_wall_s": 4020.41, "budget_s": 1800, "remaining_s": -2220.41, "n_crashes": 0}

说明：cumulative_wall_s 4020s 超过 1800s 预算，超出部分来自结果前缺陷修复的
5 次 launch（import/bank-state/snapshot-lr/val-loss 崩溃 + 1 次主动停止的 dc-channel
错误理论运行），全部在 live_manifest seq 2-9 记录、零有效结果被采用；最终有效运行
= 最后一次 launch（Stage 1 gate PASS + Stage 2 完成）。budget log append-only，如实记录。

## 8. Fingerprints

```json
{
 "contract/contract.json": "c2399d1ad9e5d69b1504413a21971a59b0c6fd74980003b15c2ed7444adfcb8f",
 "contract/live_manifest.json": "fa87497309803dfffb5251f5a2fe7d222015a5ccb50321569292f88826a20f01",
 "runs/stage1_cells.jsonl": "8776e28067c6facd0885e75a9d3dc5c72875b76dbf4689f511fd4f1c5a7142d6",
 "runs/stage1_gate.json": "7215e381d517befc57ffc3867c6157127c51378693c219f50c5959619de2ec80",
 "gpu_budget_log.json": "40f85a0f037c94e6ebd382df2dacae1619ccaf29a3d0013520de9815ecca2c4d",
 "runs/stage2_metrics.json": "8efb64a902df3960362e7c6844e161b5677ff0082c76bc727b0654948e7d22fc",
 "research/src/fr/a0o_lr_response_v01.py": "65dccaa2b6de7c94ea931c0f867c4592f275441996a1d75cff047e122095ddd7",
 "research/scripts/a0o_run_v01.py": "bf43aa0142a5e3d0e310ebf1e55c834636d83cdb9d99255ca309af6cf25847e2",
 "research/scripts/a0o_analyze_v01.py": "3ec5e3ef097ceb6b872fae710c23d7f258ac432fdfbcae7105c6b91683f3a46a"
}
```
