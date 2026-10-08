# F3.5 — Unstable-Mode Coupling 报告（METRIC_DEPENDENT）

- 日期：2026-09-30（数据生产 2026-09-29 11:21 – 09-30 04:56，tmux/setsid 持久流，resume-safe，最终 STREAM COMPLETE rc=0）。
- 预注册：`docs/F3_5_UNSTABLE_MODE_COUPLING_PREREG.md`（commit `2ac9d45`，先于一切 F3.5 计算冻结；实现层澄清见其 §16 ERRATUM）。输入契约与 F3 完全相同（checkpoints/optimizer state/future streams/CRN/frozen runtime/forward AdamW，零新训练）。
- 完整性（FACT）：三 cell 的 traj 重构均与 F3 已提交 raw **逐位锚定**（forced = rq1 α=1.0 臂、impulse = rq2 q0 臂终范数逐位相等；逐步 loss 与 current-env frozen reference 逐位相等）；RAW 度量下全部 15 个 (cell,offset) 的 ρ 与 F3 `spectrum.csv` **逐位相等**（锚点 gate 全 PASS）；adjoint 一致性 gate 3×3 全 PASS（disc ∈ [7.0e-9, 1.0e-6] ≤ 1e-5 冻结容差）；DIML 数值有效性：全部 15 点 ρ 相对偏差 ≤ 0.93% « 5%（无一被标 DIML_UNSTABLE——真实 per-tensor 尺度比远好于最坏情形）。
- 制品：`outputs/f3_5_mode_coupling/summary.json` + 9 个冻结 CSV + `raw/` 12 个原子 JSON。

## 0. 总判定（§11 冻结程序，postprocess 输出）

**主标签：`METRIC_DEPENDENT`。**

- excitation 支持：RAW = null、DIML = null（无一满足"两类 EARLY cells 的 E 汇总 > 2× MIDDLE"）。
- persistence 支持：**RAW = false（方向反转）vs DIML = true（EARLY 为 MIDDLE 的 9–16×）→ 主结论反向** → 按冻结规则判 METRIC_DEPENDENT。
- gate 全 PASS（无 LEFT_OPERATOR_GATE_FAIL）；两 metric 数据均数值有效（非 INCOMPLETE）。

## 1. FACT（按 RQ）

### RQ3.5-1 — 实际 tangent 对 unstable LEFT 子空间的激发（E_left）

| cell | RAW forced/impulse 中位 | DIML forced/impulse 中位 |
|---|---|---|
| 2042:26 | 8.7e-9 / 4.8e-9 | 9.0e-5 / 3.1e-4 |
| 2043:26 | 7.2e-9 / 4.1e-9 | 1.0e-4 / 1.0e-4 |
| 2042:64 (MIDDLE) | **1.4e-7** / 3.5e-9 | **1.1e-4 / 5.7e-5** |

- 两 metric 下 E_left 都极小（RAW ~1e-9；DIML ~1e-4–1e-3）。**不存在 EARLY > MIDDLE 的分离**：RAW 下 MIDDLE 的 forced E_left 反而最大（1.4e-7）；DIML 下三 cell 同量级。
- metric 稳健性判定：E_left(forced) median AGREE（都无分离）、auc DISAGREE；impulse median AGREE、auc DISAGREE。

### RQ3.5-2 — LR source 选择性激发（S_exc，t=1..32）

RAW 中位：EARLY 1.84e-8 / 2.30e-6 vs MIDDLE 2.52e-7（无分离，MIDDLE 最高）；DIML 中位：EARLY 7.5e-3 / 1.56e-2 vs MIDDLE 6.6e-3（2043 为 MIDDLE 的 2.4×，但 2042:26 仅 1.14× → 不满足"两 EARLY 均 ≥2×"）。auc 双 DISAGREE。**F3 报告中的"LR-source 方向选择性"在 q0≫随机 probe 的意义上成立（F3 已测），但在"source 投影进 unstable LEFT 子空间"的意义上不成立或 metric 依赖。**

### RQ3.5-3 — unstable 子空间时间持续性（U = right）

| 量 | 2042:26 | 2043:26 | 2042:64 (MIDDLE) |
|---|---|---|---|
| RAW O_prop（区间均值） | 0.233 | 0.239 | **0.349** |
| DIML O_prop | **0.0029** | **0.0052** | 0.0003 |
| RAW O_sub（1→3→7→15） | 0.30/0.21/0.08/0.019 | 类似衰减 | 类似衰减 |

- 静态 O_sub 在两 metric 下都快速衰减（子空间本身在旋转）。
- **propagated overlap 的 phase 排序在两 metric 间反转**：RAW 下 MIDDLE 最高（0.349 > EARLY 0.233/0.239 → persistence 不支持）；DIML 下 EARLY 为 MIDDLE 的 9–16×（0.0029/0.0052 vs 0.0003 → 支持 persistence）。这是主结论 METRIC_DEPENDENT 的直接来源。
- 附加量 O_prop_gain（exploratory，未归一化子空间中介增益）：EARLY 350–1876、传播范数 1e3–1e4 —— unstable 子空间向量在 1–8 步内自身爆炸式增长，与 ρ(J)>1 的逐步复合一致。

### RQ3.5-4 — mode-removal / mode-only causal probe（offset 0）

| cell×metric | κ(WᵀV) | 状态 | 结论 |
|---|---|---|---|
| 2042:26 RAW | 9.35e7 | NOT_ESTIMATED（κ>1e3 护栏） | — |
| 2042:26 **DIML** | **2.27** | **OK（d_u=2）** | G32: full=129.4、unstable-only=**2705.8**、removed=**328.1** → **SUFFICIENT=True，NECESSARY=False** |
| 2043:26 RAW / DIML | 7.8e6 / 维度不匹配 | NOT_ESTIMATED | — |
| 2042:64 RAW / DIML | 1.3e6 / 维度不匹配 | NOT_ESTIMATED | — |

- RAW 坐标下 left/right unstable 子空间近乎平行退化（κ~1e6–1e8，σ_max/ρ~1e26 的极强非正规性在 RAW 几何中的体现）；prereg 冻结的 κ≤1e3 护栏正确触发。
- 唯一可执行的 causal probe（2042:26 DIML）：不稳定分量单独传播增长 2706×（远超 full 的 129×）→ 充分性成立；但**移除**不稳定分量后增长反而升至 328×（充分非必要——移除后剩余分量中其他方向接管增长，与 F3 的 DISTRIBUTED 模块结论同构）。

### RQ3.5-5 — 随机方向对照（初始激发 vs H32 增长，n=6/cell/metric，exploratory）

- Spearman(E_left_init, log10 G32)：**RAW = +1.00 / +0.94 / +0.89**（三 cell 一致强正相关——v_only probe 初始 E=3e-4 最高、G=1.1e10 最大）；**DIML = −0.60 / −0.31 / −0.60（反号）**。初始不稳定左投影在 RAW 几何下是增长的强预测子，在 DIML 几何下反向——第三处 metric 反转。

## 2. 解释（SUPPORTED_INTERPRETATION，标签之外）

1. **RAW Euclidean 几何被 v-floor 病态主导**（σ_max~1e22–26 全部住在 v 通道方向）：unstable 左子空间在 RAW 中几乎与一切实际方向正交（E~1e-9），其"激发/持续"结构对本科学问题缺乏操作性。
2. **DIMENSIONLESS 几何**（按初始 frozen state 每 tensor 每 channel 归一）给出的物理不同：激发仍小（≤1.6%）但 persistence 出现清晰 EARLY-only 结构（9–16×），且唯一良态 causal probe 支持"不稳定分量充分但非必要"。
3. 因此 F3.5 的核心方法论结论：**"unstable mode coupling 是否解释 phase 差异"这一问题的答案依赖状态空间度量；在预注册的双 metric 协议下不能宣称任何单边机制**。这与 F3 的 SPECTRAL_INSTABILITY 判定相容：谱不稳定本身（ρ>1.05 遍布三 cell、与 metric 无关——特征值 similarity 不变量）是稳健的；耦合结构不是。
4. 谨慎注记（HYPOTHESIS 级）：DIML 下 EARLY persistence 信号 + causal 充分性提示"早期训练的不稳定子空间在缩放几何中更连贯地自我延续"，但受制于单 probe 可用性（κ 护栏）与激发强度弱，不升级为结论。

## 3. 执行 provenance 与事故记录（FACT）

- 计算全部在 GPU1 单流串行完成；每 (cell,offset,metric) 原子落盘、断点安全。总 wall ≈ 13.5h（含重跑）。
- 事故与修复（全部在 git 历史可查）：① tmux server 连续死亡 4 次——前两次为 systemd-logind 会话清理（`Linger=no`；已 `enable-linger` 根治），后两次为内核 OOM-killer（RSS 123GB：left 提取期 basis+3 组子空间共存）——以 streaming 提取 + U 子空间磁盘卸载 + `malloc_trim`/mmap 阈值三补丁根治（峰值降至 ~90GB 内）；② 一次外部干预将 runner 重命名为 `.disabled-20260929`（经 git 验证为纯重命名后恢复）；③ O_prop 定义实现偏差（未归一化）在首个 cell 完成后由人工复核发现，按冻结定义修正并对该 cell 重跑（json 删除重算，锚点重新 PASS）；④ 其余勘误（伴随算子方向、共轭实基、起始向量约定）均在 prereg ERRATUM 披露且先于正式计算。
- 无 reboot（全程同一 boot）；无 NVIDIA Xid；无 sudo/systemctl/GPU reset。

## 4. 输出与后续

- `outputs/f3_5_mode_coupling/summary.json`（verdict=METRIC_DEPENDENT）、9 CSV（adjoint_consistency / eigenpairs_summary / tangent_unstable_alignment / source_unstable_alignment / subspace_rotation / propagated_overlap / mode_projection_probe / random_probe_relation / metric_robustness）、`raw/` 12 JSON。runtime 大张量缓存已按 prereg 释放，不入 Git。
- 对 F4 的含义（供人工决策，本轮不进入）：机制链应写成——**谱不稳定（metric 无关、稳健）+ 病态基础设施（v-floor，RAW 几何主导）+ 耦合/持续性（metric 依赖，DIML 下 EARLY-only）**；任何 F4 方法设计若依赖"压制特定不稳定子空间"，必须先在 DIML 类缩放几何中复验其因果必要性（本 probe 显示非必要）。
