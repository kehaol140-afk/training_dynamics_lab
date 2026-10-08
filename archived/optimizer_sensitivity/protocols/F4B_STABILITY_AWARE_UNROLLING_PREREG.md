# F4B — Stability-Aware Unrolling (SAU) 预注册

- 冻结日期：2026-09-30，先于任何 method evaluation。上游 gate 状态：F4A.5 cheap-risk classifier **FAIL**（balanced acc 0.539 < 0.65）→ **`MODEL_PRESCREEN = DISABLED_BY_PREREG`**，SAU+PreScreen variant 不运行。Primary SAU 不依赖模型。
- 目标：利用轨迹自身的在线敏感性增长信号，动态选择安全的 hypergradient unroll horizon，用于真实 LR 决策。不修改 forward AdamW；不做 spectral clipping；不压 unstable eigenspace。

## 1. 冻结的评估单元（D9）

42 个 cells（全部来自既有 bank/F4A.1，无新训练，不按结果挑选）：
- g1b_100m：seeds 2042/2043/2044 × steps 26/64/102（9）
- g1a_26m：同 3×3（9）
- v07_5m：seeds 2030–2037 × steps 26/64/102（24；26/102 为 F4A.1 matched，64 为原 bank）
future stream：沿用各 cell 在 F4A/F4A.1 已冻结的 stream id（config.json 记录）。

## 2. 冻结的敏感性信号（D2）

全部来自**当前 unroll 已计算的量**（F4A C 臂 forced 切线，参数通道）：
- g(H) = ‖dθ_H‖/‖dθ_1‖（H ∈ {1,2,4,8,16,32}）
- λ̂(H) = log g(H) / H
- q(H) = g(H)/(g(H/2)+eps)
- 决策用 h(H) = h_eta（C 臂 θ 切线与 ∇L_val 的内积，F2 语义，F4A 已存于各 readout）
禁止：future validation loss、future FD、test action outcome。

## 3. 冻结的 horizon 规则（D3）

从 H=1 起沿 {1,2,4,8,16,32} 递增；H_safe = 最大满足"此前所有 readout 满足预算"的 H：
- g(H) ≤ B_g，q(H) ≤ B_q，λ̂(H) ≤ B_λ。
预算网格（**8 组**，非全组合）：
  (B_g,B_q,B_λ) ∈ {(8,4,.10),(8,4,.20),(16,4,.10),(16,8,.20),(32,8,.10),(32,8,.30),(64,16,.20),(128,16,.30)}。
h_min 网格：{0, 0.005, 0.01, 0.02}。两都在 TRAIN folds 内选择：最小化 train 平均 normalized regret；tie-break：更大 median H_safe，再更简单（网格序更早）。

## 4. 冻结的决策任务（D5–D8）

- 通道 u = log η；actions A = {−0.05, 0, +0.05}（η ← η·exp(a)）；secondary robustness δ=0.02。
- 线性预测 ΔL̂(a) = h(H_safe)·a；取 argmin；|h| < h_min → a=0。
- **Ground truth**：对每 cell 在同一 CRN stream 上真实 rollout 三臂（−δ/0/+δ）至 H_eval=32（LR 全程缩放 exp(a)），L = **partition_v07 eval 集 128 行 mean CE（fp32 无 autocast，F2 语义）在 θ_{t+32} 处**。
- Regret = L(chosen) − min_a L(a)；Regret_norm = Regret/(max−min+1e-30)。
- compute cost = 该方法选用的 unroll 步数 H（切线步计数；primal replay 三方法共享不计差）。

## 5. 冻结的 baselines（D10）

Fixed-H ∈ {1,2,4,8,16,32}；BEST_FIXED_H_TRAIN（train folds 内最优固定 H）；Oracle-H（每 cell 事后最优，仅上界）；SAU。（SAU+PreScreen：disabled。）

## 6. LOSO（D11）

fold = 一个 seed 的全部 cells（跨 scale 同 id 同 fold）。每 fold：TRAIN 选 (B_g,B_q,B_λ)、h_min、BEST_FIXED_H；TEST 全冻结。

## 7. Primary metrics（D12）

normalized/raw regret、oracle-action accuracy、harmful rate（L(chosen)>L_zero）、median selected H、mean compute（切线步）、catastrophic sensitivity rate（g(32) > 1e4）；按 scale/progress/curvature regime（dominant: R_curv_32≥10 vs neutral <3；中间不入分层）分层。

## 8. Success gate（D13，冻结）

PRIMARY SUCCESS 需同时：
- A. SAU 的 LOSO **median** normalized regret < BEST_FIXED_H_TRAIN 的 median；
- B. SAU harmful rate ≤ BEST_FIXED_H；
- C. curvature-dominant subset 上 SAU normalized regret 中位 < BEST_FIXED_H 中位；
- D. curvature-neutral subset 上 SAU 中位 ≤ BEST_FIXED_H 中位 × 1.5（不过度截断）；
- E. SAU mean compute ≤ Fixed-H32 的 compute（=32）。
**PRACTICAL_GAIN**：median normalized regret 相对 BEST_FIXED_H 改善 ≥ 20% 且 > 半数 held-out seeds 同方向。

## 9. Canary（D14）

g1b 2042:26（EARLY）+ 2042:64（MIDDLE）。若 |L_plus − L_minus| < 数值分辨率（同环境重复 rollout 的可重复性水平，实测以 L_zero 双跑差为零为分辨率下界，取 |L_plus−L_minus| > 1e-6 为可分辨）→ `UTILITY_NOT_RESOLVABLE`，停止扩大。

## 10. Verdict（D15）

`SAU_OUTPERFORMS_FIXED_H / SAU_REDUCES_HARMFUL_DECISIONS / SAU_COMPUTE_QUALITY_TRADEOFF / MODEL_PRESCREEN_ADDS_VALUE / NO_METHOD_GAIN / UTILITY_NOT_RESOLVABLE / HIGH_SEED_HETEROGENEITY / UNRESOLVED`。

## 11. 红线

不 push；不因结果调整任何冻结网格/阈值/gate；负结果保存；不发明第二个算法救结果。
