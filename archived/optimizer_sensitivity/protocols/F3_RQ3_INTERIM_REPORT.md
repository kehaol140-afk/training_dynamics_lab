# F3 RQ3 Interim Report — Module Causal Ablation（两 EARLY cells 全 28 臂 + MIDDLE rq1/rq2 上下文）

- 日期：2026-09-29（RQ3 EARLY 全臂完成、RQ4 2043 完成时点）。执行：opencode，tmux 持久流（g0=GPU0/g1=GPU1），resumable chunked runner（commit 4fc24fe + 72a111e）。
- 数据：`outputs/f3_curvature_mechanism/raw/rq3_2042_26.json`、`rq3_2043_26.json`（各 32 步 × 28 臂，`f3_check_raw.py` PASS：losses 与 current-env frozen reference 逐位一致、h=32 全臂在场、无 stop）。
- **执行正确性锚定（FACT）**：两 cell 的 FULL 臂与已提交 `rq1_*.json` 的 α=1.0 臂在全部 32 步的 `dtheta_norm/dm_norm/dv_norm/dz_norm/hvp_norm/shares/top_module/h_eta` 上 **0 mismatch（全精度逐位相等）**；SO|FULL 与 `rq2_*.json` 的 `a1.0|q0_lrsource` 臂逐位相等。探针（01:37–01:53）与正式运行同样逐位一致（探针制品 `rq3_probe_2042_26.archived.json` 保留）。

## 1. 主表（forced G_forced(H=32) 与 source-off G_impulse(H=32)）

### EARLY 2042:26（FULL G32 = 2188.2，final |dθ|=1770.2，SO|FULL G32 = 129.4）

| arm | final ‖dθ‖@H32 | G32 | SO G32 | FULL/arm |
|---|---|---|---|---|
| FULL | 1770.2 | 2188.2 | 129.4 | 1.00× |
| NO_ATTN_Q | 426.5 | 527.2 | 16.0 | **4.15×** |
| NO_ATTN_K | 763.4 | 943.6 | 19.3 | 2.32× |
| NO_ATTN_V | 553.2 | 683.9 | 33.3 | 3.20× |
| NO_ATTN_O | 1068.2 | 1320.4 | 37.7 | 1.66× |
| NO_MLP | 152.1 | 188.0 | 20.7 | **11.64×**（LARGE_EFFECT，唯一 ≥10×） |
| NO_EMBED | 1595.4 | 1972.1 | 123.2 | 1.11× |
| NO_NORM | 1761.7 | 2177.6 | 127.4 | 1.00× |
| ONLY_ATTN_Q | 37.9 | 46.9 | 3.7 | 保留 2.1% |
| ONLY_ATTN_K | 29.4 | 36.4 | 2.2 | 保留 1.7% |
| ONLY_MLP | 33.2 | 41.0 | 3.2 | 保留 1.9% |
| NO_MLP_UP/GATE/DOWN | 767/724/615 | 949/895/760 | 31/29/29 | 2.3×/2.4×/2.9× |

### EARLY 2043:26（FULL G32 = 18877.3，final |dθ|=13570.4，SO|FULL G32 = 45836.5）

| arm | final ‖dθ‖@H32 | G32 | SO G32 | FULL/arm |
|---|---|---|---|---|
| FULL | 13570.4 | 18877.3 | 45836.5 | 1.00× |
| NO_ATTN_Q | 8123.8 | 11300.7 | 3665.2 | **1.67×** |
| NO_ATTN_K | 40922.3 | 56925.7 | 12750.2 | **0.33×（去除反升 3.0×）** |
| NO_ATTN_V | 26185.8 | 36426.3 | 21069.4 | 0.52×（反升） |
| NO_ATTN_O | 13289.7 | 18486.8 | 16524.4 | 1.02× |
| NO_MLP | 25155.5 | 34993.1 | 9781.0 | 0.54×（反升） |
| NO_EMBED | 11970.2 | 16651.3 | 18775.6 | 1.13× |
| NO_NORM | 13724.4 | 19091.6 | 45350.2 | 0.99× |
| ONLY_ATTN_Q | 564.3 | 785.0 | 158.4 | 保留 4.2% |
| ONLY_ATTN_K | 85.1 | 118.4 | 23.3 | 保留 0.6% |
| ONLY_MLP | 475.0 | 660.7 | 12.8 | 保留 3.5% |
| NO_MLP_UP/GATE/DOWN | 18990/21332/22503 | 26416/29674/31303 | 25789/25133/26656 | 均 0.6–0.7×（反升） |

## 2. 六个必答问题（FACT）

1. **FULL 复现 EARLY 强增长？** 是。G32 = 2188（2042）/ 18877（2043），与 rq1 α=1.0 逐位一致（锚定 0 mismatch）；source-off impulse 也强增长（SO G16 = 31.3 / 108.0，均 ≥ 冻结阈值 10）。
2. **NO_ATTN_Q 降低多少？** forced：2042 降 4.15×、2043 仅 1.67×（均 < 10× LARGE_EFFECT）。impulse（SO）：2042 降 8.07×、2043 降 12.51×——attn_q 在 impulse 设定下贡献更大，但 forced 设定下远非主导。
3. **ONLY_ATTN_Q 单独重现主要增长？** 否。仅保留 FULL G32 的 2.1%（2042）/ 4.2%（2043）。
4. **NO_MLP / ONLY_MLP？** 2042：NO_MLP 11.64×（两 cell 中唯一 LARGE_EFFECT）；2043：NO_MLP 反而使增长升至 1.9×。ONLY_MLP 保留 1.9%/3.5%。跨 seed 方向都不一致。
5. **EARLY 两 seed 是否一致？** **不一致**（模块归因层面）：2042 的最大单模块效应是 NO_MLP（11.6×），2043 的最大是 NO_ATTN_Q（1.67×）且多个 remove 臂反向（去除某模块的 curvature feedback 反而放大增长）。共同点：所有 ONLY_* 臂都 ≤4.2%、所有 remove 臂都 <12×。
6. **MIDDLE 是否明显不同？** 是（基于已完成的 MIDDLE rq1/rq2；MIDDLE rq3 正在运行）：α sweep G32 ∈ [20.1, 29.0]、α1/α0 比值仅 1.31；q0 impulse α=1 G16 = 2.03（<10，有界；EARLY 为 31/108）。另注（exploratory）：MIDDLE 的 m_only/v_only random probes G16 = 7.3/11.2——状态通道对随机方向仍放大（与 F1 的 v-floor 病态一致），但 θ-channel LR-source 方向不放大。

## 3. 中间分类（task 标签集）

**DISTRIBUTED_CURVATURE**

依据：无任何单一模块 necessary（最大 remove 效应 11.6×@2042-NO_MLP，但同臂在 2043 为 0.54×；NO_ATTN_Q ≥10× 于 0/2 EARLY cells → 预注册 §8.2 verdict-2 的 LARGE_EFFECT 条件失败）、无任何单一模块 sufficient（ONLY_* ≤4.2%）、跨 seed 归因不一致、多臂反号。α 通道整体有效应（sweep 比值 86×/744×）且 impulse 显著放大 → 增长真实存在于 curvature feedback 通道，但其 receiver 模块分布是弥散的。

（注：预注册 §8.2 按序判定中 v2 = ATTN_Q_CURVATURE_BOTTLENECK 已不成立；v1 的严格单调性在 2042:26 不满足，最终 verdict 待 RQ4 全部完成后由 `f3_postprocess.py` 冻结程序汇总。）

## 4. RQ4 2043:26 先行结果（FACT，complete）

α=1 的 ρ(J_t)（Arnoldi m=16，全部 residual ≤ 3.9e-5 « 0.05，reliable=5/5）：

| offset | 0 | 1 | 3 | 7 | 15 |
|---|---|---|---|---|---|
| ρ(α=1) | 3.534 | 4.578 | 1.911 | 1.726 | **154.879** |
| ρ(α=0) | 1.186 | 1.200 | 1.166 | 1.129 | 1.135 |

5/5 offsets ρ > 1.05（冻结 margin）且 transient G16(q0,α=1)=108 ≥ 10 → 按冻结 §8.1，2043:26 的机制分类为 **SPECTRAL_UNSTABLE**（α=0 臂的 ρ 也略 >1，如实记录；非 gate 条件）。σ_max ~1e25–1e26（v 通道病态方向主导，与 F1 一致）。

## 5. 状态与剩余

- DONE（checker PASS）：rq1 ×3 cells、rq2 ×3 cells、rq3 EARLY ×2（全 28 臂）、rq4 2043:26。
- RUNNING：rq4:2042:26（g0）、rq3:2042:64（g1，rq3p 已完成 priority 五臂）。
- 之后：rq4:2042:64 → RQ5 后处理（f3_postprocess.py，输出 summary_final.json + 8 CSV）→ final report。
