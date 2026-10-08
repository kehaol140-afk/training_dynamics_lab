# F4A.1 — Matched 5M Coverage 报告

- 日期：2026-09-30。预注册 `docs/F4A1_MATCHED_COVERAGE_PREREG.md`（commit `ce694cc`，先于一切新 5M 计算）；recipe 审计 `docs/F4A1_5M_RECONSTRUCTION_AUDIT.md`（RECIPE_RECONSTRUCTABLE）。
- 数据：新 5M matched states 16 个（8 seeds × step26/step102，legacy recipe 逐行复刻、legacy 只读 import、新 checkpoint 只写项目 outputs）+ canary 控制 1 个（seed 2030 step64）。

## 0. 总判定（§6 冻结标签）

**主标签：`NONMONOTONIC_SCALE_STRUCTURE`；次标签：`HIGH_SEED_HETEROGENEITY`。**

## 1. Canary（B3，冻结判据）——PASS

seed 2030 重训至 102，其 step64 对照原 8-seed 家族包络：R_curv_32=6.03 ∈ [1.87, 9.27] ✓、λ_SO_32=0.0887 ∈ [0.0405, 0.1088] ✓、v-floor ✓、loss_h1=7.4437 越上界 0.080——按 prereg 允许项（单项越界 < 1 家族 IQR=0.111）**PASS**。跨环境（fr_minimind→minimind-exp）不假定 bitwise；分布等价成立。

## 2. Matched scale × progress（B6；中位数 [min..max]，n_seed）

| p | 量 | 5M | 26M | 100M | Spearman(logN) |
|---|---|---|---|---|---|
| .203 | R_curv_32 | 11.9 [5.0..155] | 9.1 [6.7..118] | **744** [86..1.6e7] | +0.5（非单调） |
| .203 | λ_SO_32 | 0.084 | 0.134 | **0.335** | **+1.0（单调）** |
| .500 | R_curv_32 | 2.61 | 1.85 | **6.31** [1.3..11.3] | +0.5 |
| .500 | λ_SO_32 | 0.064 | 0.055 | 0.107 | +0.5 |
| .797 | R_curv_32 | 1.07 [0.92..1.38] | 0.94 | 0.92 | −1.0（均 <3） |
| .797 | λ_SO_32 | 0.016 | 0.0087 | 0.0139 | −0.5 |

## 3. 正式回答（B6 问题）

1. **same p 下 R_curv 是否随 N 增大？** 否——非单调（5M≈26M ≪ 100M 在 EARLY；MID 同型；LATE 全部 ~1）。100M 独大，5M/26M 同级。
2. **same p 下 λ_SO 是否随 N 增大？** **EARLY 是**（+1.0 单调：0.084→0.134→0.335）；MID 弱同向；LATE 无趋势（全 ~0.01 量级）。
3. **early scale effect 是否成立？** 对 source-off 指数率成立（单调），对 curvature 放大比 R_curv 不成立（非单调：受 seed 异质主导，100M step26 单 seed 达 1.6e7）。
4. **late 是否全部 curvature-neutral？** **是**——p=.797 三 scale 全部 24 条 seed×scale 观测 R_curv_32 < 3（中位 ~0.9-1.1，最大 1.38）；λ_SO 仍为正（~0.009-0.016）→ 按 erratum 术语：LATE = CURVATURE-NEUTRAL REGIME，非 fully stable。

## 4. 对 F4-A 结论的修正力

- Erratum 修正 2 的覆盖偏差已消除：5M 现有 matched 三点。原 F4-A "peak-vs-N 非单调" 结论**保持并强化**（5M EARLY 峰值 11.9 > 26M 9.1，仍非单调）。
- 新增事实：**EARLY λ_SO 的单调 scale 增长**（传播子指数率）与 R_curv 的非单调并存——两者捕捉不同对象（指数率 vs curvature 相对放大），F4A.5 建模须分开处理。
- 5M EARLY 也存在 curvature-dominant（max 155）——phase 结构三 scale 完全 matched 成立。

## 5. 制品

`outputs/f4a1_matched_coverage/`：`matched_phase_map.csv`（90 行 = 74 原 + 16 新）、`matched_summary.csv`、`canary_envelope.json`、`new_states/`（17 checkpoints，不入 Git）、`canary_raw/`、`batch.log`。新 state 度量并入 `outputs/f4a_phase_map/raw/`。
