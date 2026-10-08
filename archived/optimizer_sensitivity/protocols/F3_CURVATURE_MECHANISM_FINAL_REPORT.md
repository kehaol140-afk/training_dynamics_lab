# F3 — Curvature Feedback Mechanism 最终报告（SPECTRAL_INSTABILITY）

- 日期：2026-09-29（数据生产 00:37–07:22 CST，全部 tmux 持久流内完成，零中断）。
- 预注册：`docs/F3_CURVATURE_MECHANISM_PREREG.md`（f51ce82 + 658301c，未做任何修改）；reproduction 契约按 `docs/F3_REPRODUCTION_GATE_AMENDMENT.md`（d63a8e1，先于一切 compat 结果提交）替换为 current-environment frozen reference（SHA256 `6f2650e0…8bdf6`），复航 gate PASS（80241f4）。blocked 历史（a2e49f9 + `docs/F3_CURVATURE_MECHANISM_REPORT.md` + 旧 `summary.json`）完整保留。
- 数据完整性（FACT）：RQ1×3、RQ2×3、RQ3×3（各 28 臂）、RQ4×3 全部 `f3_check_raw.py` PASS（每 cell 全部 32 步 primal loss 与 frozen reference **逐位相等**、h=32 全臂在场、无 stop）。RQ3 的 FULL 臂对已提交 rq1 α=1.0 臂、SO|FULL 对 rq2 `a1.0|q0_lrsource` 臂在三个 cell 上**全精度逐位锚定 0 mismatch**（执行重构正确性证明；探针 72a111e）。
- 汇总制品：`outputs/f3_curvature_mechanism/summary_final.json` + 8 个冻结 CSV（alpha_sweep / forced_growth / impulse_growth / module_ablation(_secondary) / module_only / spectrum / finite_time_growth / curvature_ratio_angle）。

## 0. 总判定（§8.2 冻结程序，`f3_postprocess.py` 输出）

**主标签：`SPECTRAL_INSTABILITY`（= §8.1 聚合 `SPECTRAL_UNSTABLE`，两 EARLY cells 一致；MIDDLE cell = UNRESOLVED）。**

按序判定：
- v1 `CURVATURE_FEEDBACK_CAUSAL`：**不成立**（严格单调性仅 2043:26 满足；2042:26 与 MIDDLE 在小 α 处 G_forced 轻微下降，非单调——尽管 α 端点比值 86×/744×/1.31×）。
- v2 `ATTN_Q_CURVATURE_BOTTLENECK`：**不成立**（前置 v1 失败；且 NO_ATTN_Q 的 LARGE_EFFECT（≥10×）在 0/2 EARLY cells 满足：4.15×/1.67×）。
- §8.1 机制分类：2042:26 与 2043:26 均为 `SPECTRAL_UNSTABLE`（ρ 可靠且 >1.05 于 5/5 offsets + transient G16 ≥ 10）；MIDDLE `UNRESOLVED`（有谱证据但无 transient）。

## 1. FACT（五问五答，全部冻结公式产出）

### Q1 — α=0 vs α=1 是否复现 B/C 巨大差异？**是。**
G_forced(H=32)：α=0 三 cell 均 ≈ 22–25（有界、phase 无关）；α=1 = 2188.2（2042:26）/ 18877.3（2043:26）/ 29.0（MIDDLE）。端点比值 86.3× / 744.0× / 1.31×。α sweep 非线性强烈（2043: α=0.25→1.0 从 219.7 → 18877；2042: α=0.5 处才开始爆发）。

### Q2 — source-off impulse 是否仍 EARLY growth？**是。**
q0（LR-source 方向）source-off，α=1：G16 = 31.31（2042:26）/ 107.96（2043:26）/ 2.03（MIDDLE）；λ₃₂ = ln G32/32 = 0.152 / 0.335 / 0.044 nats/步。EARLY 的放大**不是** source 累积：同一臂 forced G32（2188/18877）与 impulse G32（129.4/45836）同量级（2043 impulse 甚至更大）。
方向选择性（FACT）：EARLY 中 q0 方向 G16 = 31–108，而随机方向 probe 仅 4.2–6.0（m/v 通道）、theta_only 1.2–1.4——**LR source 方向选择性激发增长**；MIDDLE 中 m/v 通道随机 probe 仍放大 7.3–11.2（F1 的状态通道病态仍在），但 q0 仅 2.03。

### Q3 — 哪个 curvature output module causally controls growth？**没有单一模块（分布式）。**

forced 设定 FULL/arm（H=32 final norm 比）：

| arm | 2042:26 | 2043:26 | MIDDLE |
|---|---|---|---|
| NO_ATTN_Q | 4.15× | 1.67× | 1.20× |
| NO_ATTN_K | 2.32× | **0.33×（反升）** | 1.10× |
| NO_ATTN_V | 3.20× | 0.52×（反升） | 1.17× |
| NO_ATTN_O | 1.66× | 1.02× | 1.16× |
| NO_MLP | **11.64×（唯一 ≥10×）** | 0.54×（反升） | 0.97× |
| NO_EMBED | 1.11× | 1.13× | 1.11× |
| NO_NORM | 1.00× | 0.99× | 1.00× |
| ONLY_ATTN_Q | 保留 2.1% | 保留 4.2% | 保留 94%（0.94×） |
| ONLY_ATTN_K | 1.7% | 0.6% | 106% |
| ONLY_MLP | 1.9% | 3.5% | 142%（1.42× 反降） |

- 无必要模块（最大 remove 效应 11.6× 出现在 2042-NO_MLP，但同臂在 2043 反向 0.54×；跨 seed 不一致）；无充分模块（ONLY_* ≤ 4.2%）；2043 中 4 个 remove 臂反向（去掉某模块的 curvature feedback 反而放大增长 1.9–3.0×）。
- impulse 设定下 attn_q 贡献更大（SO|NO_ATTN_Q / SO|FULL = 8.1× / 12.5×）但仍非唯一（SO|NO_MLP 6.3×/4.7×）。
- MIDDLE：模块消融完全惰性（全部 0.94–1.42×）。

### Q4 — local spectrum 能否解释？**能（冻结判据下）。**

ρ(J_t)（matrix-free Arnoldi m=16，α=1，**全部 30 个 (cell,offset) reliable**，residual ≤ 1.3e-3 « 0.05，多数 ≤ 1e-5）：

| cell | off=0 | 1 | 3 | 7 | 15 |
|---|---|---|---|---|---|
| 2042:26 | 2.329 | 2.412 | 2.763 | 4.907 | 4.528 |
| 2043:26 | 3.534 | 4.578 | 1.911 | 1.726 | **154.879**（resid 5.4e-15） |
| 2042:64 (MIDDLE) | 2.126 | 2.055 | 1.850 | 2.202 | 4.049 |

- 两 EARLY cells：5/5 offsets ρ > 1.05（冻结 margin）且 transient G16(q0,α=1) = 31.3/108.0 ≥ 10 → `SPECTRAL_UNSTABLE`（≥2 offsets 条件超额满足）。
- σ_max(J) ≈ 1e22–1e26（v 通道病态方向主导，量级与 F1 的 v⁻³ᐟ² 机制一致）；non-normality proxy σ/ρ 天文数字但 ρ > margin，故 NONNORMAL 类不适用（且不声称 non-normal 已证明——本判定走谱路径，无需它）。
- α=0 对照的 ρ = 1.047–1.200（30 点中仅 off7-MIDDLE=1.0475 ≤ 1.05）：optimizer-only 切线算子本身**边缘性**谱不稳定（≈+0.05–0.20/步），如实记录（非 §8.1 gate 条件）。

### Q5 — EARLY/MIDDLE 差异是什么？（冻结标签之外的结构性事实，标注 SUPPORTED_INTERPRETATION）

**FACT 层**：(a) 局部 α=1 算子在两个 phase 都谱不稳定（MIDDLE ρ = 1.85–4.05，5/5 > 1.05）；(b) 但 MIDDLE 的 q0 impulse G16 = 2.03（不激发），EARLY = 31–108（激发）；(c) 2043 的 ρ 沿轨迹急剧增长（off3 的 1.9 → off15 的 154.9），2042 与 MIDDLE 增长温和（≤4.9/4.0）；(d) RQ5 分解：逐步 r_curv = ‖C dz‖/‖A dz‖ ≈ 0.02–0.6（读出步），cos(A dz, C dz) ≈ 0（|cos| ≤ 0.25，全 cell/arm）——**无逐步 constructive alignment**。

**SUPPORTED_INTERPRETATION**：EARLY/MIDDLE 的差别不在于"局部算子稳定 vs 不稳定"，而在于（i）训练早期 θ 通道/LR-source 方向与不稳定子空间存在强耦合（方向选择性：q0 ≫ 随机 probe），且该耦合把切线送入 v-floor 病态方向（F1 机制）后逐步行走于变化中的算子族；（ii) EARLY 算子族本身沿轨迹 parametrically 恶化（2043 off15 ρ=154.9）。逐步曲率增量小（r_curv~0.1）且与 A-步近正交（cos≈0）→ 放大是**逐步行走算子乘积的谱累积**，而非单步增益或固定 constructive 几何。MIDDLE 中不稳定谱局部仍在（ρ≈2–4），但 θ 方向解耦（q0=2.03）、模块消融惰性、α=1 端点仅 1.31×。

### RQ5（curvature ratio/angle，随 RQ1 数据 + 后处理）

`curvature_ratio_angle.csv` / `finite_time_growth.csv`（主定义，α=1 轨迹）+ exploratory per-arm 分解（rq3 五机制核心臂，`raw/rq3_*.json` 内 `decomposition` 字段）：读出步 r_curv：2042 FULL 0.096–0.166；2043 FULL 0.042–0.603（h=16 瞬时升高后回落）；MIDDLE FULL 0.063–0.122。cos 全部 |·| ≤ 0.25。结论同 Q5(d)。

## 2. 与上游 F1/F2/F2.5 的衔接（FACT + INTERPRETATION）

- F1 的 v-floor 病态（A_v~2.4e9）在 σ_max ~1e22–1e26 中直接可见（v 通道方向）；MIDDLE 随机 m/v probe 仍放大 7–11×——**病态本身不是 EARLY 特有**。
- F2/F2.5 的"C 臂 EARLY 爆炸、B 臂有界、MIDDLE 有界"现象在 α sweep 端点比值（86×/744×/1.31×）与 impulse（31/108/2.0）中完整复现（且 RQ3 FULL/rq1 α=1 逐位锚定 = F2-faithful 管线）。
- 本轮把 F2.5 遗留的"为什么 EARLY 独有"细化为：**不稳定谱常数存在，EARLY 特有的是方向耦合 + 算子族沿轨迹恶化**。v-floor 单因果链（F2.5 已否定）进一步让位于"病态基础设施 + 谱激发"双因素图像。

## 3. NOT_ESTIMATED / 边界声明

- HVP source block 分解（H·(P_module δθ)）未计算——RQ3 是 receiver/output-channel ablation，不声称识别 Hessian 输入源（prereg §5 红线遵守）。
- ρ 的全部 30 个估计均 reliable；无 NOT_ESTIMATED 项。未为任何结论调整 Arnoldi 配置（m=16、res tol 0.05、PI 20/1e-6 全冻结）。
- α_c 阈值未做事后定义；α sweep 的非单调小 α 下降（2042/MIDDLE，~0.1–0.6%）按探索性观察报告。

## 4. 执行 provenance（FACT）

- 长任务全部在 tmux session `f3`（窗口 g0=GPU0 / g1=GPU1）内经 `scripts/run_f3_driver.sh` 执行：done-marker 幂等、每 stage checker 通过才落标、5 分钟资源采样（`outputs/f3_curvature_mechanism/monitor/`，已 gitignore）。零 WSL reboot、零 OOM、零 Xid；GPU 峰值 < 5 GB/进程，CPU RAM 峰值 ~40 GB/进程。
- RQ3 为 chunked streaming 重构（commit 4fc24fe/72a111e；修掉 2026-09-28 版本的 136GB-RSS thrash 与 SO 臂 h≥2 缺失两个执行缺陷）；单测 5/5 + 全仓 8 套件 PASS；对已提交 rq1/rq2 臂逐位锚定三 cell 0 mismatch。
- Commit 链（本报告前）：… → 80241f4(gate PASS) → … → 4fc24fe(runner) → 72a111e(probe) → 6422b2d(EARLY+MIDDLE rq1-rq4 部分) → 10c99dc(rq3 MIDDLE) → 本次最终提交。不 push。

## 5. 结论

F3 的答案：EARLY Transformer 训练中 curvature feedback 的巨大 finite-time sensitivity 放大，在冻结判据下是**局部一步算子谱不稳定（ρ(J_t)>1，遍布 EARLY 窗口并在 2043 内沿轨迹恶化至 ~155）被 LR-source 方向选择性激发**的结果；该激发的 receiver 结构是分布式的（无必要/充分模块），逐步几何近似正交（无 constructive alignment）。MIDDLE 的局部算子同样 ρ>1 但 θ 方向解耦、模块惰性、端点效应仅 1.31×——phase 差异在耦合与算子演化，不在病态的存在性。
