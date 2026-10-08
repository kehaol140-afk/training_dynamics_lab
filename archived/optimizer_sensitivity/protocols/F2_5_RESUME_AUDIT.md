# F2.5 — Resume Audit（接管断点审计）

- 日期：2026-09-26（本轮接管时建立）。审计方式：100% 只读（进程/GPU/tmux/screen/Git/文件系统/制品 schema），随后只对 OpenCode WIP 做验证性运行与最小修复，未启动任何 F2.5 计算前已完成本表。
- 接管者：Claude（无 OpenCode/GLM 聊天记录，全部状态自磁盘恢复）。

## 0. 运行环境检查（PHASE 0.1）

| 项 | 结果 |
|---|---|
| `ps aux` | 无任何 python 训练/实验进程（仅 vscode-server / pylance / pet） |
| `nvidia-smi` | 2× RTX 5090；GPU0 407MiB（Xwayland），GPU1 0MiB；无实验进程 |
| `tmux ls` / `screen -ls` | 无会话 |
| PID/lock/running marker | 无 |
| 项目内日志 | 仅历史日志：`outputs/f0_*_run.log`、`outputs/f1_mechanism/{f1_mechanism_run.log,run.out}`、`outputs/f2_hypergradient/{f2_run.log,run.out,run.attempt*.out}`、`outputs/f1_1b_replication/download.log`（全部为 F0/F1/F2 已完成轮次的历史记录，mtime ≤ 2026-09-26 20:39） |

**结论：无旧任务在运行。不存在重复启动风险。**

## 1. Git 状态（PHASE 0.2）

- 分支 `master`，HEAD = `e333ffa`（F2.5 预注册冻结）。
- `git status`：仅 2 个 untracked 文件 —— `src/opsens/f25_trust_region_v01.py`、`tests/test_f25_trust_region_v01.py`（OpenCode WIP，未提交）。
- `git diff` / `git diff --cached`：空（无 tracked 修改）。
- 未执行任何 reset/clean/checkout。

## 2. OpenCode 进度判定（PHASE 2/3）

| component | status | completed | expected | artifact |
|---|---|---:|---:|---|
| 1. prereg | **COMPLETE** | 1 | 1 | `docs/F2_5_TRUST_REGION_PREREG.md`（git `e333ffa`，文件 mtime 2026-09-26 21:40，**先于任何 F2.5 计算**；δ 网格 8 点、H{1..32}、3 cells、ε_metric=1e-7、ε_abs=1e-6、R≤0.2+符号判据、连续前缀 δ*、4 种 backward-only oracle 干预、6 标签判定程序、红线全部冻结） |
| 2. baseline rollout (δ=0, L0) | NOT_STARTED | 0 | 3 cells | — |
| 3. positive-delta rollout | NOT_STARTED | 0 | 3 cells × 8 δ | — |
| 4. negative-delta rollout | NOT_STARTED | 0 | 3 cells × 8 δ | — |
| 5. central FD | NOT_STARTED | 0 | 3 cells × 8 δ × 6 H | — |
| 6. tangent h_H | NOT_STARTED | 0 | 3 cells × 5 interventions × {fp64/C, fp64/B} × 6 H | 库已备（见下） |
| 7. trust-radius analysis | NOT_STARTED | 0 | 3 cells × 5 interventions × 6 H | 库已备（见下） |
| 8. backward intervention | NOT_STARTED | 0 | 4 interventions × 3 cells | 库已备（见下） |
| 9. conditioning-vs-radius | NOT_STARTED | 0 | 3 cells | 上游 F0/F1 制品齐备 |
| 10. final report | NOT_STARTED | 0 | 1 | — |

**OpenCode 实际做到哪里**：F2.5 预注册已冻结并提交（`e333ffa`）；分析库 `f25_trust_region_v01.py` 与单测 `test_f25_trust_region_v01.py` 已写出但**未提交、未运行、无 runner、无 outputs、无报告**。即：冻结设计完成，计算零启动。

## 3. OpenCode WIP 代码审计（逐项验证）

### 3.1 `src/opsens/f25_trust_region_v01.py`

| 项 | 审计结果 |
|---|---|
| 冻结常量 | `F25_DELTA_GRID=(1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,5e-2,1e-1)`、`F25_H_READOUTS=(1,2,4,8,16,32)`、`F25_EPS_METRIC=1e-7`、`F25_EPS_ABS=1e-6`、`F25_R_THRESHOLD=0.2` —— 与 prereg §1/§2 逐字一致 ✓ |
| `tangent_step_oracle` | per-parameter 数学为 vendored 冻结 `adamw_tangent_step` 逐行复刻；两个 backward-only 钩子：`dv_mask_by_name`（只置零 v→θ′ 通道项 `−m̂·dsqrt/denom²`，m 通道不动）、`v_back_tau`（`v←max(v,τ²)`，M/Dv 公共分母，F1-B 语义）；forward 状态不动 ✓ 与 prereg §3 一致 |
| 无干预位级相等 | 单测 `test_no_intervention_bit_exact` PASS（与 vendored 函数 `torch.equal`）✓ 满足 prereg §6 红线 |
| `trust_radius` | 连续前缀规则：升序扫描，跳过 unresolvable；遇到第一个 unusable 且 resolvable 的 δ 即返回当前半径；全部 usable 返回网格上限 1e-1；无可解返回 0.0 ✓ 与 prereg §2 一致 |
| `compute_R` / `usable` | `R=|ΔL−h·δ|/(|h·δ|+ε_metric)`；usable = R≤0.2 且 sign(h·δ)==sign(ΔL) ✓ |

**发现的问题与最小修复（接管者执行，均不改变冻结规则）**：

1. `test_vfloor_changes_denominators_only` 原实现用 `_make_state()`（v∈[1e-12,1e-6]），τ²=1e-16 的 clamp 是 no-op，导致 `not torch.equal` 断言必失败（**测试构造缺陷，非库缺陷**）。修复：改用含 v=1e-20 坐标的手工 state。修复后 4/4 PASS。
2. `rden_topk_masks(state, fraction, classify_fn, diag_fn)`：`classify_fn` 传入但未使用（死参数）；`diag_fn` 回调存在误用风险——若传 `diagonal_gain`（=c·R_den），`1.0/r` 会选中**最安全**坐标而非最危险（G 与 R_den 是正比例，取 1/G 的 top = 取 R_den 的 bottom）。prereg 冻结规则是"按 bank 锚点态 R_den=1/(√v̂₁+eps) 全局降序"。**修复**：移除两个回调，函数内部直接用 `f0.denominator_anchor` 计算 R_den（=1/(√v̂₁)+eps 的倒数），签名变为 `rden_topk_masks(state, fraction)`；坐标集选择规则不变。修复后全部单测仍 4/4 PASS。
3. （记录不改）`usable()` 不内置 resolvable 判定 —— 与 prereg 一致（resolvable 由 runner 按冻结规则计算后传入 `trust_radius`）。

### 3.2 测试基线

- 接管时 OpenCode 单测状态：3/4 PASS（上述缺陷 1 失败）。修复后 **4/4 PASS**。
- 既有全套测试复核：f0 6/6、f1_mechanism 5/5、f1_1b 6/6、f1_multistep smoke 4/4、f1_scripts_synthetic ALL PASS —— **无回归**。

## 4. 上游制品可用性确认（conditioning 输入，prereg §4）

| 量 | 来源 | 2042:26 | 2043:26 | 2042:64 |
|---|---|---:|---:|---:|
| A_v（v 通道放大，fp64） | f1 summary `F1E_precision_amplification` | 1.783e9 | 4.684e9 | 1.266e8 |
| σ_max（梯度通道） | f0 `power_iteration.csv`（仅 seed 2042）/ `per_parameter_group.csv` diag_gain_max 按组取 max | 2258.3 | （无 power_iteration 行 → 用 G 通道对角 max 代理，prereg §4 允许） | 295.1 |
| attn_q G2 share | f1 `module_localization.csv` | 0.537 | （g1b 2043:26 行存在） | 0.175 |
| √v̂<10eps 坐标占比 | f0 `per_parameter_group.csv`（组加权聚合） | 0.0599(attn_q) | 0.1251(attn_q) | 0.0032(attn_q) |

tangent norm growth 将由本轮 F2.5 rollout 直接产生。

## 5. 断点续做计划（与 prereg 对齐，不新增计算范围）

1. 写 `scripts/f25_trust_region.py`（runner：FD 17 臂 + δ=0 baseline + 5 interventions × {fp64/C, fp64/B} 切线臂，共享同一 primal rollout；mask 集在 bank 锚点态冻结）。
2. 3 cells 顺序执行（GPU1），raw JSON 落盘 `outputs/f2_5_trust_region/`。
3. 写 `scripts/f25_build_outputs.py`：trust_radius.csv / fd_convergence.csv / backward_intervention.csv / conditioning_vs_radius.csv / summary.json（全部按 prereg 冻结公式）。
4. 补 runner 侧最小单测（正/负 δ 配对、central FD 公式、resolvable 规则、intervention 不修改 forward trajectory——以 raw 中 primal loss/clip 轨迹与 F2 位级一致性验证）。
5. `docs/F2_5_TRUST_REGION_REPORT.md` + 按 prereg §5 判定程序给出 verdict。

**禁止事项确认**：不改 forward AdamW、不动 F2 已提交代码、不改 prereg、不覆盖任何已有制品、不做 F3。
