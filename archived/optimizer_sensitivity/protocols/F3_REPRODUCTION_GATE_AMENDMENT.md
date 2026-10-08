# F3 Reproduction Gate Amendment — Current-Environment Compatibility Gate

- 日期：2026-09-27。
- 本文档是对 `docs/F3_CURVATURE_MECHANISM_PREREG.md` 的**增订**，不覆盖、不修改原预注册的任何内容。原预注册与 blocked 运行产物（commit `a2e49f9`）完整保留为 provenance。
- 本文档在任何 compatibility 计算之前冻结并提交。所有阈值、容差、repeat 次数、判定程序均在数据之前写死；任何 R0/R1 失败即按 §6 停止。

## 1. 事实（全部精确引自 blocked report / raw artifacts）

1. F3 实现曾存在一个真实的 ordering bug：`dc`（HVP）与 vendored `adamw_tangent_step` 在 `opt.step()` **之后**执行（HVP 读取 post-step θ，tangent 读取 post-step optimizer state）。已修复为与 F2 冻结语义一致的 pre-step 顺序（grads → clip → HVP → clip_jvp → tangent → `opt.step()` → 加 s_h）。修复后单元测试 12/12 PASS（全部测试 45/45 PASS）。
2. 输入侧漂移已审计并排除：代码（git 干净，自 F2 commit `87d8225` 未变）、bank state（mtime 2026-09-16）、数据（mtime 2026-09-02）、`minimind_base`（mtime 2026-09-01）、torch 环境（2026-06-30 安装）、batch manifest（`e40dd1f4709ca8b5`，与 F2 记录一致）。
3. 尽管如此，**当前执行环境重新执行 archived F2 trajectory 出现 primal numerical drift**：F2 冻结脚本 2026-09-27 重跑的 h=1 loss = 7.43338680267334，而 archive（2026-09-26 三次 attempt 一致）= 7.43325（%.5f 打印精度；h=1 相对差 ~1.9e-5）；FD 相位 h_FD(16)/h_FD(32) 相差 5%–108%。
4. 修复后 F3 冻结 gate 对 archive 的失败量化（`outputs/f3_curvature_mechanism/raw/gate_{2042_26,2043_26,2042_64}.json`）：
   - EARLY `2042:26`：G1 max = 3.49e-3，G2 max = 0.888；
   - EARLY `2043:26`：G1 max = 1.28e-3，G2 max = 0.908；
   - MIDDLE `2042:64`：G1 max = 5.93e-5，G2 max = 1.73e-3，G4 max = 5.08e-4。
   即：EARLY full-C trajectory 把这种微小执行差异显著放大；MIDDLE 的 drift 小得多。
5. 同日（2026-09-27）环境内部可复现性：两个独立进程、GPU0/GPU1、28GB 显存压力下 h=1 前向逐位一致；修复后 F3 对同日 F2 重跑参考复现 **G1 = G2 = 0.0**（32 步 × B/C 两臂逐位，`raw/gate_diag_sameday_2042_26.json`）。

## 2. 结论（只陈述事实，不做未经证明的解释）

archive-level exact replay（对 `outputs/f2_hypergradient/raw_{cell}.json` 逐点 1e-12）在当前执行环境不足以作为 F3 主实验的可操作的 reproduction contract。因此增设 **CURRENT-ENVIRONMENT COMPATIBILITY GATE**（R0 + R1 + frozen reference），通过之后 F3 主实验的 reproduction 比较全部改对 current-environment frozen reference，容差仍为 1e-12（同环境下已证明可达）。archived F2 保留为历史独立 evidence，不再作为 execution gate。

本文档不声称"chaos 的证明"；§1.3/§1.4 仅报告测量的 drift 与放大现象。

## 3. STEP A — 环境冻结（先于任何 compat 计算）

生成 `outputs/f3_compatibility/environment_manifest.json`，字段至少包括：git HEAD、python 版本、torch 版本、CUDA runtime、NVIDIA driver、GPU model、transformers（若可导入）、numpy、`CUDA_VISIBLE_DEVICES`、TF32 flags（`torch.backends.cuda.matmul.allow_tf32`、`torch.backends.cudnn.allow_tf32`）、cudnn deterministic/benchmark、torch deterministic settings（`torch.are_deterministic_algorithms_enabled()`）、SDPA backend 支持集、matmul precision、autocast dtype（primal bf16）、`PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`、`OMP_NUM_THREADS` 等相关环境变量、三 cell bank checkpoint sha256、future-stream manifest sha256（每 cell）、关键代码文件 sha256。

禁止为匹配 archive 切换任何 backend/flag。F3 后续全部 compat + 主实验使用同一 frozen environment（以该 manifest 为准；若环境 manifest 发生实质变化，须停止并重新走 R0/R1）。

## 4. R0 — current-environment repeatability（冻结协议）

**不比较 archive。** 对每个 cell，使用与原 F3 完全相同的 checkpoint、optimizer state、future stream、CRN、B/C branch 定义（即修复后的 F2-faithful 管线），在**独立进程**中重复运行：

| cell | repeats（冻结） |
|---|---|
| EARLY `2042:26` | 3 |
| EARLY `2043:26` | 3 |
| MIDDLE `2042:64` | 3 |

预算评估：单次运行 ~3 min，9 次 ~30 min，允许 3/3/3。

**每 repeat 保存**（`outputs/f3_compatibility/raw/compat_{seed}_{step}_rep{r}.json`，h=1..32 逐步 + 读出点 H∈{1,2,4,8,16,32}）：
- complete primal loss trace、clip coefficient trace、grad norm trace；
- B 臂与 C 臂的 `‖dθ/du‖, ‖dm/du‖, ‖dv/du‖, ‖dz/du‖` traces；
- h_eta（H 读出点，fp32 eval forward）、module shares（F0 九组）、HVP norms（C 臂）。

**指标（冻结定义）**，对每 cell 的 3 个 repeats r∈{1,2,3}：
1. `primal_loss_relative_spread(h) = (max_r loss_r(h) − min_r loss_r(h)) / |mean_r loss_r(h)|`，h=1..32；
2. `G_X(r,H) = ‖dθ_H/du‖_X(r) / max(‖dθ_1/du‖_X(r), 1e-30)`（X∈{B,C}；H∈{16,32}）；
   `C_growth_ratio_maxmin(H) = max_r G_C(r,H) / min_r G_C(r,H)`，B 同理；
3. branch ordering：每 repeat 是否有 `G_C(r,16) > G_B(r,16)` 且 `G_C(r,32) > G_B(r,32)`；
4. EARLY/MIDDLE ordering：每 EARLY cell 的 `min_r G_C(r,16)` 是否 > MIDDLE 的 `max_r G_C(r,16)`。

**R0 gates（冻结容差，数据前写死）**：
- **A**：所有 cell、所有 h∈{1..32}：`primal_loss_relative_spread(h) ≤ 1e-6`；
- **B**：所有 cell、H∈{16,32}：`C_growth_ratio_maxmin(H) ≤ 1.10` 且 `B_growth_ratio_maxmin(H) ≤ 1.10`；
- **C**：EARLY `2042:26` 的 3/3 repeats 满足 branch ordering（H=16 与 H=32）；
- **D**：每个 EARLY cell：`min_r G_C^early(r,16) > max_r G_C^middle(r,16)`（3×3 非重叠）。

**R0 任一 gate 失败 → 立即停止 F3，verdict = `CURRENT_ENVIRONMENT_NOT_REPEATABLE`，不进入主实验。**

## 5. R1 — F2 phenomenology compatibility（冻结协议，仅在 R0 PASS 后执行）

使用同一批 9 个 repeat runs（§4 的 runs 同时服务 R0 与 R1）。不要求数值等于 archived F2；要求复现其核心 phenomenology。**R1 gates（冻结阈值）**：

- **P1**：两个 EARLY cells 各自、每个 repeat：`G_C(r,16)/G_B(r,16) ≥ 5` 且 `G_C(r,32)/G_B(r,32) ≥ 5`；
- **P2**：`median{G_C(r,16) over 6 个 EARLY runs(2 cells × 3 repeats)} / median{G_C(r,16) over 3 个 MIDDLE runs} ≥ 5`；
- **P3**：H=1 基本行为：每 repeat 内 B 与 C 的 `‖dθ_1/du‖` 相对差 ≤ 1e-12（dc≡0，两臂一致）；H=2：`0.9 ≤ G_C(2)/G_B(2) ≤ 1.1`（与 archive 在 H=2 的 qualitative semantics 不冲突；archive 值 1.582504/1.595065 ≈ 0.992）。

**R1 任一 gate 失败 → 立即停止 F3，verdict = `F2_PHENOMENOLOGY_NOT_REPRODUCED`，不跑 alpha sweep / module ablation。**

## 6. Frozen reference（仅在 R0+R1 PASS 后建立）

- 每 cell 取 **repeat 1** 的完整 traces 作为 frozen reference（同日 repeats 已证明逐位等价；reference 的选择在数据前冻结，避免挑选）。
- 保存 `outputs/f3_compatibility/frozen_reference/`：`baseline_summary.json`（含 `tangent` 行 `{arm: fp64/B|fp64/C, h, dtheta_u_norm}`，直接可被 F3 gate 模式 `--ref-raw` 消费）、`primal_traces.csv`、`B_tangent_traces.csv`、`C_tangent_traces.csv`、`module_shares.csv`、`hvp_norms.csv`、`repeatability.csv`；并生成 `CURRENT_ENV_REFERENCE_SHA256.txt`（对整个 frozen_reference 目录的规范哈希）。
- 后续 F3 全部 arms 必须：相同 checkpoint/state、相同 future stream、相同 environment manifest、相同 forward trajectory 定义。
- **F3 的 α=0 必须对 current reference B 逐步 `dtheta_u_norm` 达相对 ≤ 1e-12；α=1 必须对 current reference C 达相对 ≤ 1e-12。** 除此之外不再把 archived F2 的逐 float equality 作为 F3 execution gate。

## 7. 不变项声明

除 reproduction reference 的替换之外，本 amendment 不改原 F3 的任何科学内容：alpha grid、H、cells、module arms、spectrum 定义（PI/Arnoldi 配置）、verdict 判定程序全部照旧。fail 的 R0/R1 结果如实保存并提交。

## 8. 执行顺序与红线

1. 本文档 commit（"Amend F3 reproduction gate for current-environment compatibility"）；
2. 捕获 environment_manifest.json；
3. 运行 9 个 compat repeats（独立进程）；
4. R0 判定 →（PASS）→ R1 判定 →（PASS）→ 建立 frozen reference；
5. 生成 `docs/F3_COMPATIBILITY_REPORT.md`，verdict ∈ {`CURRENT_ENV_REFERENCE_ESTABLISHED`, `CURRENT_ENV_NOT_REPEATABLE`, `F2_PHENOMENOLOGY_NOT_REPRODUCED`, `UNRESOLVED`}；
6. 仅当 `CURRENT_ENV_REFERENCE_ESTABLISHED`：恢复 F3 RQ1–RQ5，reproduction 比较改对 frozen reference。

红线：禁止在任何 R0/R1 结果出来后修改本文档的任何阈值/定义；禁止为匹配 archive 切换 backend/flag；禁止 push、reset --hard、clean、rebase/squash 旧历史；每阶段本地 commit。
