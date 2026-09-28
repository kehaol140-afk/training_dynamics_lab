# A0Q Response-Control Map Exploration

日期：2026-09-18。前置：STATUS §40（A0G–A0P 正负结果台账）已先于本轮写入。合同 sha256 `543851ca…71d2` 先于任何 GPU 冻结。

## 0. Verdict

- **lr**：gate PASS（Pearson +0.8423，sign 1.000，rel err 0.188）；median |h|=8.293e-03（1.00×LR），sign stability 1.00，CV(|h|) 0.49，1/3 unique signs
- **clip**：gate FAIL（Pearson +0.9357，sign 0.833，rel err 0.090）；median |h|=0.000e+00（0.00×LR），sign stability 0.25，CV(|h|) 1.49，2/3 unique signs
- **wd**：gate FAIL（Pearson +0.1476，sign 0.500，rel err 1.001）；median |h|=2.058e-06（0.00×LR），sign stability 0.92，CV(|h|) 0.37，1/3 unique signs
- 稳定通道（预注册规则：gate PASS ∧ sign stability≥2/3 ∧ 3/3 seed sign agreement，按 median |h| 排序）：['lr']
- CRN 12/12：True；clip 在全部 48 step 中激活 10 次。离线重算一致：True

## 1. 理论与冻结

三通道均为持久乘性对数扰动（u=log(x/x₀)，δ=0.05 自冻结 a0o 导入，H=4 每步施加，与 A0O 验证的 FD 参考同语义）。解析源项：
- LR：A0O amendment-A2 纯 θ 通道推，h_lr=Σ⟨p_th,lr·e_lr⟩（冻结 lr_pulse import-only）。
- WD：纯 θ 通道（AdamW 中 wd 不触碰 m/v），h_wd=Σ⟨p_th,−lr·λ₀·θ_k⟩，λ₀=0.01（v0.2 bank 存储的原生值）。
- CLIP：dc 通道，clipped=g·min(1,τ/‖g‖)，∂clipped/∂τ=ĝ·1[preclip>τ₀]，h_clip=Στ₀⟨b_k,ĝ_k⟩·1[active]，b_k 为冻结 clip transpose 入口的 dc-cotangent，τ₀=1.0（cfg grad_clip 冻结断言）。
12 cell：seeds {2030,2031,2037} × steps {46,82} × streams {7400,7401}；stream 配方 derive_stream_seed(·,404)+batches_from_train_pool（8 micro-batches）；provenance 门复用 A0O-A1。禁止 controller/optimizer/λ 改动；不根据结果选择通道或状态。

## 2. Response matrix J（12 cell × 3 通道）

| seed | step | stream | h_lr | fd_lr | h_clip | fd_clip | h_wd | fd_wd |
|---|---|---|---|---|---|---|---|---|
| 2030 | 46 | 7400 | -1.5130e-02 | -1.1302e-02 | +0.0000e+00 | +0.0000e+00 | +1.3368e-06 | +6.0618e-04 |
| 2030 | 46 | 7401 | -1.0670e-02 | -7.6056e-03 | +0.0000e+00 | +0.0000e+00 | +1.0522e-06 | +2.3842e-06 |
| 2030 | 82 | 7400 | -7.4646e-03 | -8.1718e-03 | +0.0000e+00 | +0.0000e+00 | +2.1869e-06 | -7.3791e-04 |
| 2030 | 82 | 7401 | -4.7005e-03 | -4.8679e-03 | +0.0000e+00 | +4.5300e-05 | +2.0296e-06 | -3.2723e-04 |
| 2031 | 46 | 7400 | -1.0424e-02 | -1.2822e-02 | +0.0000e+00 | +0.0000e+00 | +2.0863e-06 | -1.6868e-04 |
| 2031 | 46 | 7401 | -1.5602e-02 | -1.3126e-02 | +0.0000e+00 | +0.0000e+00 | +2.1875e-06 | +4.6134e-04 |
| 2031 | 82 | 7400 | -9.1209e-03 | -1.0426e-02 | -3.7343e-03 | -3.1668e-03 | +2.7410e-06 | -1.1027e-04 |
| 2031 | 82 | 7401 | -6.0907e-03 | -7.1210e-03 | +7.3411e-05 | +1.5670e-03 | +2.7587e-06 | +1.3012e-03 |
| 2037 | 46 | 7400 | -1.0598e-02 | -1.1171e-02 | +0.0000e+00 | +0.0000e+00 | +6.4315e-07 | -2.0862e-05 |
| 2037 | 46 | 7401 | -7.1996e-03 | -1.8990e-03 | -2.2421e-03 | +2.5034e-04 | -3.5251e-06 | -6.9737e-05 |
| 2037 | 82 | 7400 | -3.0827e-03 | -3.9291e-03 | +5.0875e-03 | +6.4975e-03 | +1.9830e-06 | -2.6822e-04 |
| 2037 | 82 | 7401 | -1.5505e-03 | -1.0169e-03 | +3.2825e-03 | +5.3811e-03 | +1.8366e-06 | +1.2052e-03 |

## 3. 逐通道评价（预注册四指标）

| 通道 | median |h| | ×LR magnitude | sign stability | seed sign agreement | CV(|h|) | FD Pearson | FD sign |
|---|---|---|---|---|---|---|---|
| lr | 8.293e-03 | 1.00 | 1.00 | 1/3 unique signs | 0.49 | +0.842 | 1.000 |
| clip | 0.000e+00 | 0.00 | 0.25 | 2/3 unique signs | 1.49 | +0.936 | 0.833 |
| wd | 2.058e-06 | 0.00 | 0.92 | 1/3 unique signs | 0.37 | +0.148 | 0.500 |

per-seed median h：
- lr: 2030: -9.067e-03(-1)，2031: -9.772e-03(-1)，2037: -5.141e-03(-1)
- clip: 2030: +0.000e+00(+0)，2031: +0.000e+00(+0)，2037: +1.641e-03(+1)
- wd: 2030: +1.683e-06(+1)，2031: +2.464e-06(+1)，2037: +1.240e-06(+1)

## 4. 结论：哪个通道稳定、哪个适合作为未来控制变量

- 预注册规则下进入候选（稳定且跨 seed 符号一致）的通道（按 |h| 排序）：**lr**。
- clip 未过 gate：符号不稳（0.83）
- wd 未过 gate：FD 一致性不足（Pearson 0.14756108359122122）；符号不稳（0.50）
- **推荐未来控制变量（预注册规则内）：lr**。判定依据 = gate PASS + 跨 seed 符号一致 + 相对幅度。
- 与 A0K/A0L/A0N 负结果联合限定：任何通道进入闭环前必须证明相对 fixed 策略的 equal-wall-clock 净收益；A0P 限定长 τ 一阶外推不可靠（本轮 H=4 窗口内评估不受影响）。

## 5. 预算与制品

- GPU：315.0s / 1800s（budget log n_crashes 为 gate 标记非进程崩溃）。逐 cell 计时见 gpu_budget_log.json（append-only）。
- 制品：`outputs/a0q_response_control_map_v01/`（contract/ runs/ analysis/ reports/ hashes/ audit/）；代码 `research/src/fr/a0q_response_control_map_v01.py` + `research/scripts/a0q_{run,analyze,audit_pack}_v01.py`。
- A0O/A0P 及全部更早冻结制品未动；docs append-only。

## 6. 方法指纹

- `research/src/fr/a0q_response_control_map_v01.py` sha256 `c37277072086a00b…`
- `research/scripts/a0q_run_v01.py` sha256 `c7b0915884b9928e…`
- `research/scripts/a0q_analyze_v01.py` sha256 `517835a66ac962e2…`
- `research/scripts/a0q_audit_pack_v01.py` sha256 `342f3a64f8ecffaa…`
- `outputs/a0q_response_control_map_v01/contract/contract.json` sha256 `543851caf5793a65…`
