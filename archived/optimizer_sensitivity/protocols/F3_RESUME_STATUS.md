# F3 Resume Status（phase 0 制品级状态盘点）

- 生成：2026-09-29，opencode 接管后只读验证（验证方法：`scripts/f3_check_raw.py` + 直接 JSON 解析；不依赖文件名）。
- 参照：prereg `docs/F3_CURVATURE_MECHANISM_PREREG.md`（f51ce82+658301c）、amendment `docs/F3_REPRODUCTION_GATE_AMENDMENT.md`（d63a8e1）、`docs/F3_COMPATIBILITY_REPORT.md`（7510aec）。
- Current-environment frozen reference SHA256（已确认在位）：`6f2650e04d5933006648a3a47fa18cda90a2b1da357f6c88586927cb4388bdf6`。
- compatibility（R0/R1/frozen reference/复航 gate）状态：**全部 COMPLETE，不再重跑**。

## RQ1 — alpha sweep（ALPHA_GRID 7 点 × 32 步 forced）

| cell | Status | 证据 |
|---|---|---|
| EARLY 2042:26 | **COMPLETE** | `raw/rq1_2042_26.json`：32 steps × 7 α 臂、无 stop、全部有限、h_eta 6 读出点齐、losses 与 frozen ref 逐位一致（`f3_check_raw.py` PASS）；sidecar `rq1_2042_26.state.pt` 11.3G 在位（未入库） |
| EARLY 2043:26 | **COMPLETE** | 同上（`raw/rq1_2043_26.json`，checker PASS） |
| MIDDLE 2042:64 | **NOT_STARTED** | 无文件 |

## RQ2 — source-off impulse（α∈{0,1} × 5 probes + q0_lrsource ×2）

| cell | Status | 证据 |
|---|---|---|
| EARLY 2042:26 | **COMPLETE** | `raw/rq2_2042_26.json`：32 steps × 12 臂、无 stop、有限、checker PASS |
| EARLY 2043:26 | **COMPLETE** | 同上（checker PASS） |
| MIDDLE 2042:64 | **NOT_STARTED** | 无文件 |

## RQ3 — module causal ablation（14 forced 臂 + 14 SO| 臂 × 3 cells）

| cell | Status | 证据 |
|---|---|---|
| EARLY 2042:26 | **INVALID**（4 次启动、零数据行） | `rq3_2042_26.out` 仅含 4 条 banner（18:09/19:55/21:51/23:25）；无 `raw/rq3_2042_26.json`。根因（代码审读确认）：(a) 原 runner 单进程持有 28 个 fp64 臂 ≈136GB 工作集 > 125GB RAM → swap thrash；(b) SO 臂在 h≥2 无传播路径，且汇总日志行 `row['arms']['SO|FULL']` 在 h=2 必 KeyError。属执行层缺陷，frozen 数学（prereg §5）未实现完。 |
| EARLY 2043:26 | **INVALID** | 同上（`rq3_2043_26.out` 4 条 banner） |
| MIDDLE 2042:64 | **NOT_STARTED** | 无文件 |

## RQ4 — local spectrum（offsets 0/1/3/7/15 × α∈{0,1} × 3 cells）

| cell | Status |
|---|---|
| 全部 3 cells | **NOT_STARTED**（无 `rq4_*.json`） |

## RQ5 — curvature ratio/angle decomposition

| 项 | Status |
|---|---|
| 主定义（prereg §10：随 RQ1 数据免费计算 + 后处理） | **PARTIAL**：EARLY 两 cell 的 `decomposition` 已在 `rq1_*.json` 内（α=0/1 每步 norm_A/norm_C/cos）；MIDDLE 待 RQ1 MIDDLE 完成后由 `f3_postprocess.py` 汇总输出 `finite_time_growth.csv` / `curvature_ratio_angle.csv` |

## 汇总与恢复策略

- 可复用已完成制品：rq1/rq2 EARLY 四组 raw（已 commit 130b8f7）+ 全部 compatibility 制品。禁止重跑覆盖。
- 待做（按序）：RQ3 runner 执行层重构（分块流式、SO 臂 h≥2 传播修复，回归锚点 = 已提交 rq1 α=1.0 臂 与 rq2 `a1.0|q0_lrsource` 臂）→ RQ1/RQ2 MIDDLE → RQ3 全矩阵（优先级：2042:26 五主臂 → 2043:26 → 2042:64 → 补全 14 臂）→ RQ4 → RQ5 后处理 → final report。
- 长任务一律 tmux 内执行，禁止 agent foreground shell。
