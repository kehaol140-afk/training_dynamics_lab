# FR-C1 GTTR probe 验证报告（C1 probe validation v01）

日期：2026-09-21。GPU：单卡 RTX 5090（与他人 vLLM 进程共享，全程仅用小 footprint + host 卸载）。
输入冻结件：`outputs/fr_c0_gttr_validation_v01/`（只读）、`outputs/fr_c0x_offline_explanation_v01/`（C0X 结论）。
本轮为 GTTR 理论探索；未开发 controller / 新 optimizer；完成本协议后停止，不进入 C2。

## 0. 执行摘要

| 判定（协议 §10） | 结果 |
|---|---|
| C0_RULE_REVIEW | **ERRATUM_REQUIRED**（继承 C0X：半库门逐点误用；原 C0 判定保留不覆盖） |
| C1_EXECUTION | **PARTIAL**（执行、防泄漏顺序、预算全部干净；覆盖对称减至 4/6 anchor，原因见 §6） |
| SAME_PROBE_APPROXIMATION | **8/8 anchor-channel 优于 M0**（零预测基线），分辨性见 §3 |
| INDEPENDENT_BANK_PREDICTION（field） | **LOCAL_SUPPORT_ON_SELECTED_ANCHORS**（在已测的 A/t0 + Bp/t16 集合上） |
| SHADOW_PREDICTION（shadow） | **UNRESOLVED**（E 门值全达标，但 t0 处 shadow 目标在所有已测 K 低于 3SE 分辨下限） |
| ERROR_DECOMPOSITION | **MIXED**（cross 项保留，占 e² 的 1.8%–19.4%，符号不定） |
| FUTURE_H16_H64_PREDICTION / ADAPTIVE_TRAINING_VALUE / ENERGY_BENEFIT | **NOT_TESTED** |

一句话：在严格防泄漏的新 32+32 probe 库上，**field 通道的一阶几何运输预测（g+Hs 再 clip）首次在冻结门意义下获得跨库独立支持**（两 cell K32 聚合 E≤0.51、两半库 E≤0.70、每 cell 两个不同 t 可分辨）；**shadow 通道的预测误差同样达标但 t0 信号低于分辨下限**，按协议结论是"当前测量未解决"，不是"理论无效"。同 probe 近似在全部 8 个 anchor-channel 上优于零预测基线。

## 1. 设计回顾（与协议 §5–§9 的对照）

- 复用 C0 冻结 cell（checkpoint step 46，流 7110/7111），每 cell 固定 anchor 子集；新 dev/audit 各 32 probe（种子 2026092103/2026092104），与 C0 库、rollout 批次按样本 ID 全部不相交（smoke 实测 PASS）。
- 每 anchor：dev+audit g0 → 真实 clip+step 得 s_t、后更新公共 state S+、q+=grad V_diag(θ+s)（主读出方向，在任何 audit 终点前固定）→ dev HVP 预测 mean(P_D)、mean(Q_D) 于 K=8/16/32 **先保存（sha256 + order_index）** → 之后才读 audit HVP、audit g1、dev g1（CONDITIONED_DIAGNOSTIC，不回填不重选）。
- U 逐 probe 用真实 clip、m/v、bias correction、P 与 wd 语义；从未把均值梯度喂给 Adam 式非线性（CPU 测试 T4 验证 mean(U(g))≠U(mean(g))）。
- stop-v（C0 对基线约定：P 冻结自 audit 库均值 clip 后 g0，a2v/b2v 于 t_after）只作 ablation；另存局部 c→U 导数 D_i（T7 有限差分验证）。
- floor=0 的语义沿用 C0X 结论：same-probe 位级可重复，**不代表系统误差为 0**（HVP 代理截断 ~4.5e-3 已由 smoke 门实测并达标 <1e-2）。

## 2. 回答协议 §12 的七个问题

### Q1. C0 原判定有无误用规则？修订影响哪些结论？

有。C0 分析器把冻结的**聚合**半库门（同一 resolved 集合上的 SSE 比）实现成**逐点**门，C/Bp/t16 单点 1.166 单独导致 NOT_SUPPORTED。C0X 已按原规则复算：g1/g2 通过（聚合 E_Mgeom R=0.5291、C=0.4676≤0.8），g3 半库聚合门因 C0 未保存半库 SSE/TSS 而 DATA_MISSING；原 verdict 保留、erratum 追加。**C1 不作为 C0 的重新验收**（协议 §10 明文），但 C1 的新库按固定 ID 保存了充分统计，使同类门在新数据上可完整求值（见 Q3）。

### Q2. C0 同一目标上是否真的反转，W/K 在何处抵消？

C0X X3 已逐区间验证恒等式（残差 ≤1.4e-17）：gain 是 ΔW 与 ΔK 两个大得多的流的差，例如 C/Bp−A 在 [4,16) 区间 ΔW=−0.0547、ΔK=−0.0551，几乎完全抵消后 gain=+0.0004。63 步累计 gain 仅 0.006–0.020（V_diag≈7.2 的 0.1–0.3%），且 R cell 的 Bp 收益呈"先小负后转正"的延迟兑现形态，与 A0U Case B（固定 +5% 延迟约 6 步）一致。A0T−v_eval 轴差维持 AXIS_MISMATCH，未通过减常数强行复现反转。

### Q3. 局部几何近似在同 probe 上是否准确，跨库又损失了多少？

**同 probe（CONDITIONED）**：8/8 anchor-channel 优于 M0。以 e_local=‖P̄_A−Ȳ_A‖（field）/‖Q̄_A−Z̄_A‖（shadow）对目标范数 ‖Ȳ_A‖/‖Z̄_A‖：

| anchor | field e_local/‖Ȳ‖ | shadow e_local/‖Z̄‖ |
|---|---|---|
| C/A/t0 | 6.6% | 15.8% |
| R/A/t0 | 10.3% | 21.6% |
| C/Bp/t16 | 25.3% | 41.9% |
| R/Bp/t16 | 33.9% | 51.9% |

**跨库**：e_bank=‖P̄_D−P̄_A‖ 是同 probe 误差的 2–20 倍（t0 处相差最大），说明独立库均值预测的误差主体是跨库波动，不是局部模型缺陷。精确分解 e=e_bank+e_local 逐 anchor 恒等成立（平方恒等残差 ≤8.7e-19），cross 项占 e² 的 1.8%–19.4%、符号不定——按协议不做 norm 百分比归因，只报绝对平方量与 cross。

**独立库预测的冻结门**（每 cell 在其已测 anchor 集上聚合，公式与 C0 冻结定义相同）：

| 通道 | cell | K=32 E | 两半库 E | 可分辨 t 数 | 判定 |
|---|---|---|---|---|---|
| field | R | 0.505 | 0.700 / 0.497 | 2 | LOCAL_SUPPORT_ON_SELECTED_ANCHORS |
| field | C | 0.398 | 0.386 / 0.694 | 2 | （同上） |
| shadow | R | 0.667 | 0.872 / 0.632 | **1** | UNRESOLVED |
| shadow | C | 0.585 | 0.603 / 0.980 | **1** | （同上） |

shadow 的 E 门值全部达标（≤0.8、半库 ≤0.98），但 **t0 处 shadow 目标在所有已测 K≤32 都低于 3SE 分辨下限**（‖Z̄_A‖≈0.0039–0.0045，比值 ≈2.8–2.9），"每 cell 至少两个不同 t 可分辨"的前置条件不成立。按协议"达到上限仍不分辨时，结论是当前测量未解决"。

两点结构性备注：
- **K 增大不单调改善**（R field E：0.68→0.70→0.50），符合协议"不假设随 K 单调"的事先声明；K=32 相对 K=8 的改善是真实的（采样平均），K=8→16 的抖动是 16-probe 抽样波动。
- 固定 ID 半库中 first16 与 K=16 前缀是**同一组 probe**（都是前 16 个 ID），两者统计量恒等；半库设计实际新增信息是 last16（及其与 first16 的对称性核查）。

### Q4. 增加 probe 后哪些不确定性减小，哪些缺陷仍在？

**减小**：
- C0 的 PREDICTION_SAMPLING_UNRESOLVED（预测侧 SE 从未保存）→ 本轮每 K 都保存预测/目标双侧 SE，跨库 E 的采样不确定性可估。
- C0 半库只存比值、聚合门不可求值 → 本轮按固定 ID 保存半库 SSE/TSS/inner，聚合门完整可算。
- 同 probe 近似是否优于基线 → 8/8 明确优于 M0。

**仍在**：
- shadow 通道 t0 分辨不足（见 Q3）；probe 库从 8（C0）增至 32 只把比值从 <1 推到 ≈2.9，要到 3SE 需 ~K≥40 或更大位移。
- e_local 在 t16 明显大于 t0（field 从 ~8% 到 ~30%）：有限位移/曲率截断或 HVP 代理截断（~4.5e-3 量级）在 LR 干预臂上变大，两者本轮不可分离。
- 单点（per-probe）误差结构：per-probe shadow 残差 SE 未保存（保存的是均值级 e_local）；field per-probe 残差 SE 已保存。
- A/t48 两个 anchor 本轮未测（见 §6），late-time 泛化未检验。

### Q5. 实际 AdamW 输出耦合怎样影响这个局部增量，证据是否足够？

stop-v ablation（P 冻结自 audit 均值 clip g0，m 照常更新）显示：**冻结 P 不改变增量的方向结构（per-probe inner 全为正），但改变幅度**——stop/full 范数比 1.11–1.33（t16 处最大），‖Z_full−Z_stop‖ 的均值范数达 full 范数的 0.5–1.0 倍，两侧 SE 同量级。即 v 通道（二阶矩/预条件）承载着实信号，把"局部增量"当成与 P 无关的量会系统性偏差 15–30%。D_i（固定 common-state 下局部 c→U 导数）在主读出方向上的投影：C/A/t0 +0.158、R/A/t0 +0.153、C/Bp/t16 +0.050、**R/Bp/t16 −0.064**——LR 上调臂在 R cell 的局部导数读出分量反号，提示 AdamW 非线性耦合具有 arm/cell 依赖性。证据足够支持"耦合存在且方向依赖"这一描述性结论；不足以上升到机制归因（D_i 未含 clip-Jacobian 因子；stop-v 是人工对基线，不是 optimizer 候选）。

### Q6. GTTR 哪一段有证据，哪一段仍需要理论闭合？

- **Geometry/Transport（field 局部一阶预测）**：在已测 anchor 集上获得冻结门意义的跨库独立支持（本轮最强新证据）。注意 C0 自身 verdict 维持 UNRESOLVED（规则执行错误 + 门值缺失），C1 不覆盖 C0。
- **Tracking**：C1 不是 tracking 检验；stop-v 表明 v 通道有实信号（与 tracking 重要相容），但 mu/nu 条件化本身未测。
- **Readout**：一阶 shadow 读出增量 −q+ᵀQ̄_D、−q+ᵀZ̄_A 数量级 1e-4–1e-3（q 范数 ~0.73–0.83），多数 |mean|>3SE（标为 SHADOW_FIRST_ORDER_READOUT，非真实 loss）；不推出未来曲率累积。
- **需要理论闭合**：shadow 通道在 t0 的可分辨性（信号小是物理小还是 probe 数不足？）；e_local 在干预臂增大的来源（有限位移 vs 代理截断）；局部导数投影反号的机制。

### Q7. 真实更新与全部诊断开销、停止原因、数据不足分别是什么？

见 §6 预算表。停止原因：协议固定 6 anchor 全覆盖会因首跑失败计入而 432>416 HVP；在 cap 内取对称 4-anchor 方案，余量 112 HVP 主动保留（不为补测而自适应改设计）。数据不足：A/t48（late-time）、shadow t0 分辨、per-probe shadow 残差。

## 3. 分辨性细节（probe_summary.csv）

- field 目标：K=32 时 8/8 可分辨；R/Bp/t16 在 K=8 时不可分辨（0.188 vs 3·SE=0.204），K≥16 可分辨——小 K 抽样波动可以翻转单点分辨标志，故门只用聚合。
- shadow 目标：t0 两个 cell 在 K=8/16/32 全部不可分辨（比值 2.8–2.9）；t16 在 K=32 可分辨。
- same-probe field 残差 SE：t0 为 1.3–2.2e-4，t16 为 6.6–11.7e-3（与 t16 目标范数比仍 ≤7%）。

## 4. 数字完整性与防泄漏

- 每 anchor：预测 order_index < audit order_index（0<1, 2<3, 4<5, 6<7），预测文件 mtime < records mtime；预测 sha256（对 K=8/16/32 的 mean(P_D)、mean(Q_D) group sum 的 fp64 字节）在读取任何 audit 终点前落盘；事后篡改会改变哈希。
- 分解恒等式残差 ≤8.7e-19（fp64 舍入）。
- smoke：diag 开/关全 state 哈希一致（8 updates）；HVP 代理一阶梯度与原梯度相对差 max 4.5e-3 < 1e-2 门；floor 四量全 0.0（same-probe 位级可重复）。

## 5. 不确定性清单

| 条目 | 状态 |
|---|---|
| shadow t0 目标 3SE 分辨 | 不可分辨（所有 K≤32），shadow 判定因此 UNRESOLVED |
| A/t48 late-time 覆盖 | 本轮不测（预算算术，见 §6） |
| e_local(t16) 来源：有限位移 vs HVP 代理截断 | 不分离（代理截断 ~4.5e-3 与 t16 e_local 同量级） |
| per-probe shadow 同 probe 残差 SE | NOT_SAVED（保存的是均值级 e_local；field 通道已保存） |
| stop-v 的 "相互方向" 逐 probe 余弦 | 只保存均值级 inner/范数；跨 probe 聚合 cos 为描述量，不解释为逐 probe 相关 |
| 旧 18,000 cap 与 C1 新 cap | 分别记账，不相除 |
| A0T−v_eval 轴差非常数部分 | 不可计算（C0X 已述，维持 AXIS_MISMATCH） |

## 6. 预算与真实命令（round-total，失败计入，两个计数器集同时记）

| 计数器 | smoke | 失败首跑 | main（4 anchors） | **round total** | cap |
|---|---|---|---|---|---|
| optimizer updates | 8 | 1 | 36 | **45** | 192 |
| effective-batch grad evals | 19 | 66 | 520 | **605** | 1400 |
| HVP evals | 16 | 32 | 256 | **304** | 416 |
| micro-forward calls | 63 | 194 | 1544 | **1801** | 3200 |
| GPU wall | ~2 min | ~3 min | 307 s | <25 min | 90 min |

- shadow 纯代数时间 15.6 s（单独计数，不占 GPU 诊断 eval）；旧账本 17,994/18,000 原样保留，C1 独立列账，不做除法。
- 失败首跑：R/A/t0 anchor 的 sha 步骤 KeyError（对已修 bug），消耗 1 update / 66 grads / 32 HVPs / 194 micros，按协议"失败/recheck/salvage 全部计入"记入 `runs/failed_attempts.json`。修复：预测 sha 与保存只覆盖冻结的 mean(P_D)、mean(Q_D)（协议 §7.1 原文），全 MAIN_GROUPS 的 K 前缀统计仍在 records 中保存。
- 覆盖决策：全量 6 anchor 需 16+32+384=432>416 HVP。对称减至每 cell 的 A/t0+Bp/t16（256 HVP，t∈{0,16}），round 304/416，余量 112 保留。这是范围缩减而非执行失败，故 C1_EXECUTION=PARTIAL 并在 verdict 中注明。

真实命令：

```bash
# CPU 理论测试（9/9）
CUDA_VISIBLE_DEVICES="" PYTHONPATH=research/src python research/tests/test_c1_gttr_probe_validation_v01.py
# smoke（含库不相交、diag 状态恒等、floor、HVP 门）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=research/src python research/scripts/c1_run_v01.py --phase smoke --gpu 0
# main（4 anchors，先存预测后读 audit）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=research/src python research/scripts/c1_run_v01.py --phase main --gpu 0
# 离线分析（0 GPU）
CUDA_VISIBLE_DEVICES="" PYTHONPATH=research/src python research/scripts/c1_analyze_v01.py
```

产物：`probe_summary.csv`、`error_decomposition.csv`、`aggregate_E.csv`、`fixed_projection.csv`、`update_feedback.csv`、`verdicts.json`、`fig_c1_1_error_vs_K.png`、`fig_c1_2_decomposition.png`、`fig_c1_3_fixed_direction.png`、`smoke_checks.json`、`budget_log.json(l)`、`state_provenance.csv`、`runs/{anchor_predictions,anchor_records,step_records,probe_banks,failed_attempts}.json(l)`。

## 7. 图的读法

- `fig_c1_1_error_vs_K.png`：聚合 E 随 K；灰虚线 0.8、灰点线 1.0。field 两 cell 全 K≤0.70；shadow K=8 偏高（小样本），K=32 回落；两半库空心点落在 K=16 竖线上（first16≡K16 前缀，见 §3 备注）。
- `fig_c1_2_decomposition.png`：|e|、|e_bank|、|e_local| 逐 anchor（log 轴）。t0 处 e_local 比 e_bank 低约一个量级；t16 差距收窄。
- `fig_c1_3_fixed_direction.png`：预定方向 q+ 上的投影均值±SE。field 组（P/Y）为正且远大于各自 resid；shadow 组（Q/Z 及其 stop 变体）为负、量级 1e-4–1e-3；d_c_to_u 在 R/Bp/t16 反号。
