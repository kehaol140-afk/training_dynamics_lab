# F4B — Stability-Aware Unrolling (SAU) 报告

- 日期：2026-09-30。预注册 `docs/F4B_STABILITY_AWARE_UNROLLING_PREREG.md`（commit `653cdd7`）。SAU+PreScreen：**DISABLED_BY_PREREG**（F4A.5 gate FAIL：balacc 0.539 < 0.65），未运行。
- 评估：42 cells（100M/26M 全 3×3 + 5M 8 seed × 3 matched 点），42×3 CRN action rollouts（δ=0.05）+ 42×3（δ=0.02 secondary），L = eval 集 fp32 mean CE @ H_eval=32。Canary PASS（|L₊−L₋| = 0.27/0.07 » 数值分辨率）。

## 0. 总判定

**`NO_METHOD_GAIN`**（δ=0.05 与 δ=0.02 一致；prereg 冻结程序输出，保留负结果）。

## 1. 事实（LOSO by seed，11 folds）

| 方法 | 中位 norm-regret | 均值 | harmful | 中位 H | oracle-acc |
|---|---|---|---|---|---|
| FIXED1 | 0.000 | **0.104** | **0.119** | 1 | 0.833 |
| FIXED4 | 0.000 | 0.117 | 0.143 | 4 | 0.833 |
| FIXED32 | 0.000 | 0.135 | 0.167 | 32 | 0.810 |
| FIXED16 | 0.000 | 0.222 | 0.190 | 16 | 0.738 |
| BEST_FIXED_TRAIN | 0.000 | 0.144 | 0.167 | 1 | 0.786 |
| **SAU** | 0.000 | **0.104** | **0.119** | **1** | 0.833 |
| Oracle-H | 0.000 | 0.043 | 0.095 | 1 | 0.905 |

- **SAU 在全部 11 个 train fold 上选择同一最紧预算 (8,4,0.10) 并于所有 cell 停在 H=1**——等价于 FIXED1。原因（事后诊断，如实报告）：全部 42 cell 的 λ̂(H=2) = ln G_C(2)/2 ≈ 0.35–0.38 **超过冻结网格的最大 B_λ=0.30**——即使健康轨迹 H=2 的瞬时指数率也高于网格上限，预算网格相对真实信号基线校准过紧。按红线**不改网格重跑**；该设计失误作为负结果的一部分记录。
- Gate：A（中位严格优于 BEST_FIXED）**FAIL**（0=0 平局）；B（harmful ≤）PASS（0.119=0.119）；C（dominant 子集 SAU 0.000 < BEST_FIXED 0.317）PASS；D PASS；E PASS（mean_H 1.0 ≤ 32）。
- δ=0.02 secondary：同样 NO_METHOD_GAIN（SAU=FIXED1 均值 0.181；此 δ 下 FIXED32 均值略优 0.132，中位同 0）。

## 2. 科学解读（SUPPORTED_INTERPRETATION）

1. **本任务设定下 H=1 已接近最优**：oracle-H 中位亦为 1（oracle 均值 0.043 vs H=1 的 0.104）——δ=0.05 的 LR 决策价值集中在最早的梯度信息；F2 时代"EARLY 长程 hypergradient 失效"的镜像结论是**长程 unrolling 在本 utility 下无增益且有风险**（FIXED16/32 的 harmful 率 0.19/0.17 > H=1 的 0.12）。
2. **"不稳定轨迹需要更短 unroll"的方向性证据存在**（dominant 子集：长 H 的 BEST_FIXED 中位 0.317 vs 0.000；FIXED16 全局最差），但**在线监测没有提供超越"H=1 恒定"的增量**：监测信号（g/q/λ）在 H=2 处对健康/病态轨迹不可分（两者 λ̂(2) 同为 ~0.35-0.38——分化出现在更晚的 H），而预算网格须在 H=2 就做截断决定。
3. BEST_FIXED_TRAIN 本身过拟合（0.144 > FIXED1 0.104）：train-fold 上选 H 不稳定——与 F4A.5 "个体不可预测"一致。
4. 方法论教训（负结果的价值）：**在 sensitivity 信号的早期可分辨性弱于效用曲线变化率的场合，自适应截断退化为保守常数**；若未来放宽网格（B_λ ≥ 0.5）重跑，须作为新预注册实验，本轮不追溯。

## 3. 制品

`outputs/f4b_sau/`：summary.json、summary_delta0.02.json、evaluation_cells.csv、threshold_selection.csv、loso_folds.csv、per_method_regret.csv、per_seed_results.csv、per_scale_results.csv、per_phase_results.csv、selected_horizons.csv、compute_cost.csv、action_outcomes.csv、figures/（2 图）、raw/（84 rollouts）。
