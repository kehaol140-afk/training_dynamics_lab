# F4-A — Scale × Phase × Seed Stability Map 报告

- 日期：2026-09-30。预注册：`docs/F4A_SCALE_PHASE_SEED_MAP_PREREG.md`（commit `3ac9762`，先于一切 phase-map 计算）。输入：74 个既有 bank state（`docs/F4A_EXISTING_BANK_INVENTORY.md`），零新训练。
- **Reproduction gate：3/3 PASS（逐位）**——g1b (2042,26)/(2042,64)/(2043,26) 的 B/C/SO 三臂 H=32 终范数与 F3 已提交 raw（rq1 α=0/α=1 臂、rq2 q0 臂）逐位相等，逐步 loss 与 current-env frozen reference 逐位一致（gate 一次抓出并修正了本 runner 的 stream-id 映射笔误后全 PASS——gate 体系有效性的直接证据）。
- 制品：`phase_map_long.csv`（74×6 H 读出）、`phase_map_summary.csv`、`healing_intervals.csv`、`indicator_correlations.csv`、`exploratory_models.csv`、`figures/`（3 图）、`raw/`（74 单元原子目录）。

## 0. 总判定（§13 冻结标签）

**主标签：`PHASE_STRUCTURE_ONLY_NO_SCALE_TREND`；次标签：`HIGH_SEED_HETEROGENEITY`。**

## 1. FACT

### RQ1 — self-healing（是，普适）

中位 R_curv_32（跨 seed）随 progress 单调衰减，三个 scale 全部：

| scale | p≈0.20 | 0.36 | 0.41 | 0.45 | 0.5 | 0.55–0.64 | 0.80 |
|---|---|---|---|---|---|---|---|
| 5M | — | 28.6 | 17.8 | 4.65 | 2.68 | 2.64/2.16/1.63 | — |
| 26M | 9.13 | — | — | — | 1.85 | — | 0.94 |
| 100M | 743.95 | — | — | — | 6.31 | — | 0.92 |

- **14/14 条 seed 轨迹全部自愈**（healing_intervals.csv：6/14 在覆盖内观测到 crossing 区间，其余 censored 于相邻 checkpoint 间；1 条 5M 轨迹 NOT_HEALED_IN_COVERAGE 是因 mid-window 复发点而非末值）。
- λ_SO_32 在**所有 74 个 state 均 > 0**（SOURCE_OFF_GROWING 普适），但幅度随 progress 系统收窄（100M：0.335→0.107→0.014；5M：0.143→0.038）。propagator 的有限时间增长从不完全消失，只是指数率衰减一个数量级级。

### RQ2 — 跨 seed（方向可复现，幅度高异质）

- 方向：自愈方向 14/14 一致；EARLY 为每 scale 峰值一致。
- 幅度：同 (scale,step) 内跨 seed 极差最高 **5.3 个数量级**（100M step26：86.3 / 744 / 1.58e7——seed 2044 爆炸）；26M step26 达 1.25 dex（6.7/9.1/118）；5M step46 达 1.86 dex。冻结异质判据（极差>1 dex）在多组命中 → 次标签。

### RQ3 — scale（峰值非单调；窗口有后移迹象）

- 峰值中位：5M=28.6 > 26M=9.1 < 100M=744 → 对 logN 的 Spearman=0.50 < 0.7 → 冻结判据下 **scale 趋势不成立**（注意覆盖不对称：5M 首点 p=0.36、26M/100M 首点 p=0.20，5M 峰值若在更早则被低估——如实标注）。
- 窗口：100M 3 seed 中 2 个的 p_heal 落在 (0.5, 0.80]（step64 仍 MODERATE/DOMINANT：6.31 中位）；26M 全部在 p=0.5 前治愈；5M 集中于 0.45–0.59 → **"更大模型 unstable window 更长"获得温和支持**（PARTIAL，受 checkpoint 密度限制）。

### RQ4 — cheap indicators（vs log R_curv_32，n=74）

| indicator | Spearman(raw) | Pearson(log-log) |
|---|---|---|
| **A_theta** | **−0.813** | −0.639 |
| **clip_ratio**（首步 clip 系数） | **+0.692** | +0.440 |
| **A_m** | +0.513 | +0.521 |
| A_v | +0.482 | +0.501 |
| v_floor_fraction | +0.195 | +0.252 |
| K_lr | −0.129 | −0.320 |

- **A_theta 是最强（负）预测子**：θ 通道单步增益越接近 1（病态越藏在状态通道），多步 R_curv 越大——与 F1/F3 的"stable training, unstable sensitivities"一致。clip 系数（大梯度/强 clip ↔ 更强 curvature 放大）是第二个。v-floor 比例与 K_lr 在跨尺度汇集下预测力弱。

### Exploratory models（EXPLORATORY_MODEL_ONLY，LOSO 跨 seed）

| 模型 | R² | MAE | Spearman |
|---|---|---|---|
| C（scale 固定效应 + p） | **0.306** | 0.451 | 0.736 |
| B（logN + logstep） | 0.285 | 0.426 | 0.736 |
| A（logN + p） | 0.277 | 0.446 | 0.713 |
| D（logN×p 交互） | 0.216 | 0.482 | 0.703 |

- 全部 R²≤0.31：**seed 异质性主导残差**，任何无 seed 结构的确定性模型都不足。排序能力尚可（ρ≈0.71–0.74）。不宣布任何 scaling law。

## 2. 与上游的一致性

- F3 的 EARLY/MIDDLE 对比（86/744 vs 1.3@100M）在更密的图上保持并扩展为全 phase 曲线；MIDDLE step64 的 6.31 中位（含 1.31–11.30 跨 seed）解释了 F3 时代"MIDDLE 有界"的判断（有界但仍为 MODERATE 区）。
- F3.5 的"无单一 unstable subspace"与本次"seed 间 5 dex 幅度差"同向：不稳定强度由轨迹具体状态而非尺度唯一决定。

## 3. 覆盖缺口（详见 MISSING_COVERAGE_REQUESTS）

- 5M 无 p<0.36 的 state；26M/100M 无 p∈(0.203,0.5) 内部与 p>0.797 之后；26M 每 seed 仅 3 点（目标 4）。本轮零新训练，缺口如实标注。

## 4. 执行 provenance

- 双流（GPU0/GPU1）setsid 驱动，单元级 DONE 原子 resume；100M 单元 ~4.3 min、26M ~1.5 min、5M ~40 s；全程无 OOM/reboot（RAM 峰值 <10GB）。一次 driver 启动早期因 stream 映射笔误 FAIL（被 loss gate 逐位拦截）→ 修正 commit 后重启，无重复计算。
