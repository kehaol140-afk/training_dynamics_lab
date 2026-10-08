# F3 Compatibility Report — CURRENT_ENV_REFERENCE_ESTABLISHED

- 日期：2026-09-27。
- 依据协议：`docs/F3_REPRODUCTION_GATE_AMENDMENT.md`（commit `d63a8e1`，先于任何 compat 计算冻结并提交）。
- 环境快照：`outputs/f3_compatibility/environment_manifest.json`（HEAD `d63a8e1`，torch 2.11.0+cu128 / CUDA 12.8 / driver 591.44 / 2× RTX 5090，`tf32_matmul=False`，`tf32_cudnn=True`，matmul precision `highest`，cudnn_benchmark=False，deterministic_algorithms=False）。
- **Verdict：`CURRENT_ENV_REFERENCE_ESTABLISHED`**。R0 4/4 gates PASS，R1 3/3 gates PASS。F3 主实验的 reproduction 比较改对 current-environment frozen reference（容差仍为 1e-12）。

## 1. R0 — current-environment repeatability（全部 PASS）

每个 cell 3 个独立进程 repeats。原始完整 traces：`outputs/f3_compatibility/raw/compat_{seed}_{step}_rep{1,2,3}.json`。

**核心事实：9 次运行按 cell 分组逐位一致**（跨独立进程，GPU1）——每 cell 的 3 个 repeat 在全部 32 步的 primal loss、clip 系数、B/C 切线范数上完全相等：

| Gate | 冻结容差 | 实测 | 结果 |
|---|---|---|---|
| A: primal_loss_relative_spread_max（3 cells × h=1..32） | ≤ 1e-6 | **0.0**（全部逐位相等） | PASS |
| B: C/B growth_ratio_maxmin（3 cells × H∈{16,32}） | ≤ 1.10 | **1.0**（全部） | PASS |
| C: EARLY 2042:26 branch ordering 3/3（H=16 与 H=32） | 3/3 | 3/3 | PASS |
| D: EARLY/MIDDLE 非重叠（min G_C^early(16) > max G_C^middle(16)） | 严格 > | 2042:26：×53.79；2043:26：×8.59 | PASS |

明细：`outputs/f3_compatibility/repeatability.csv`。

注：MIDDLE（2042:64）的 informational branch_order 行（C>B？）为 False——G_C(16)/G_B(16) ≈ 0.92 < 1，与 archive 的 MIDDLE 弱放大 phenomenology 一致；frozen gate C 仅约束 EARLY 2042:26，不受影响。

## 2. R1 — F2 phenomenology compatibility（全部 PASS）

| Gate | 冻结阈值 | 实测 | 结果 |
|---|---|---|---|
| P1: EARLY C/B growth ratio，每 cell × 每 repeat × H∈{16,32} | ≥ 5 | 2042:26：H16 = 31.59，H32 = 86.29；2043:26：H16 = 5.447，H32 = 743.95（6/6 组合全部 ≥5） | PASS |
| P2: median G_C^early(16) / median G_C^middle(16) | ≥ 5 | 31.19 | PASS |
| P3: H=1 B/C ‖dθ₁/du‖ 相对差 ≤ 1e-12；H=2 的 G_C(2)/G_B(2) ∈ [0.9, 1.1] | — | H1 相对差 = **0.0**（9/9）；H2：2042:26 = 0.99212，2043:26 = 1.0168，MIDDLE = 0.9747（9/9 在区间内） | PASS |

明细：`outputs/f3_compatibility/phenomenology_gate.csv`。

### 与 archived F2 的 phenomenology 对照（informational，非 gate）

| 量 | archive（2026-09-26） | current（rep1） |
|---|---|---|
| 2042:26 G_C(16)/G_B(16) | 24.12 | 31.59 |
| 2042:26 G_C(32)/G_B(32) | 45.64 | 86.29 |
| 2042:26 G_C(2)/G_B(2) | 0.992125 | 0.992124 |
| 2043:26 G_C(16)/G_B(16) | 11.56 | 5.447 |
| 2043:26 G_C(32)/G_B(32) | 8090 | 744 |
| 2043:26 G_C(2)/G_B(2) | 1.0169 | 1.0168 |
| MIDDLE G_C(16)/G_B(16) | 0.9219 | 0.9218 |
| MIDDLE G_C(32)/G_B(32) | 1.311 | 1.311 |
| MIDDLE G_C(2)/G_B(2) | 0.97473 | 0.97472 |

H=2 比值与 archive 在 ~1e-5 相对差内一致；C≫B（EARLY）/ C≈B（MIDDLE）的定序与 archive 完全相同。绝对 growth 有差异（环境漂移），R1 gates 本来就只约束 ratio/ordering，故 PASS 有效。

## 3. Frozen reference（已建立）

- 目录：`outputs/f3_compatibility/frozen_reference/`（`baseline_summary.json`、`primal_traces.csv`、`B_tangent_traces.csv`、`C_tangent_traces.csv`、`module_shares.csv`、`hvp_norms.csv`、`repeatability.csv`）。
- 选取：每 cell 的 repeat 1（数据前冻结的选择规则，amendment §6；同日 repeats 逐位等价，选取不影响内容）。
- `baseline_summary.json` 含 3-cell 合并 `tangent` 行（`{cell, arm: fp64/B|fp64/C, h, dtheta_u_norm}`）与 per-cell full-precision `losses`，可被 F3 gate `--ref-raw` 直接消费（gate 已按 cell 过滤 tangent 行、以 ref 内 losses 作为 G4 参考——对 frozen-reference 模式，archive 的 `f2_run.log` 是 stale 参考，不再适用）。
- **Bundle SHA256：`6f2650e04d5933006648a3a47fa18cda90a2b1da357f6c88586927cb4388bdf6`**（`outputs/f3_compatibility/CURRENT_ENV_REFERENCE_SHA256.txt`）。

## 4. 结论与后续

- archived F2 raw 保留为历史独立 evidence，不再作为 F3 execution gate。
- F3 主实验（RQ1–RQ5）按原预注册恢复执行：alpha grid、H、cells、module arms、spectrum 定义、verdict 判定程序全部不变（amendment §7 不变项声明）；唯一变化是 reproduction reference 换为 current-environment frozen reference，G1/G2 容差仍为 1e-12。
- 若执行期间 environment manifest 发生实质变化，须停止并重新走 R0/R1（amendment §3）。
