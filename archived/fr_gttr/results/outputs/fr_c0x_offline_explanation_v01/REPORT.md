# FR-C0X 离线规则复算报告（C0X offline explanation v01）

日期：2026-09-21。零 GPU。输入为 C0 冻结工件（sha256 见 config.json），未修改任何 C0 文件。
 governing protocol：仓库根目录 `FR_C0X_C1_GTTR_Offline_and_Probe_Validation_Claude_Code.md` 的 C0X 部分；
被复算规则出处：`FR_C0_Geometry_Tracking_Validation_Claude_Code.md` §10.1–§10.3。

## X1. 半库门：1.166 是单点误差，不是整 cell 聚合误差 —— ERRATUM_REQUIRED

**冻结规则原文（§10.2，行 366）**：
> 按固定ID将dev和audit各分成前4/后4，**使用全库已确定的同一anchor集合**；两个半库中Mgeom均不劣于各自M0（E_Mgeom<=1），不重新选anchor。

E_M 的定义（同节"主预测指标"）为**同一 resolved anchor 集合 A 上的聚合**
`E_M = sqrt(Σ_{a∈A}||p_a−ȳ_a||² / Σ_{a∈A}||ȳ_a||²)`（每 cell 每 channel 每半库一个数），
不是逐点指标。

**事实核查**：1.166372379695529 是 **C/Bp/t16 单点**的 last-4 半库比值
（`runs/anchor_records.jsonl` 中该 anchor `halfbank.last4.E_Mgeom_mu`），不是聚合指标。
C0 分析器（`research/scripts/c0_analyze_v01.py` `geom_verdict()`，约 L294–306）把该门实现为
"每个 resolved anchor 的 f4/l4 E 均 ≤1"（逐点门），并以此单点作为整 cell 半库门失败依据——
这是对冻结规则的**误用**。

**按真正冻结规则的复算**（A = 修正 floor 下的 resolved 集合，每 cell 7 个：
{A,Bp,Bm}@t0 + {A,Bp,Bm}@t48 + Bp@t16）：

| 门（field channel） | cell R | cell C |
|---|---|---|
| g1 signal：≥6 resolved、≥2 arm、≥2 distinct t | PASS（7, 3 arms, t∈{0,16,48}） | PASS（同左） |
| g2 聚合 E_Mgeom ≤ 0.8 | **PASS，E=0.5291** | **PASS，E=0.4676** |
| g3 半库**聚合** E ≤ 1（f4、l4 各一） | **DATA_MISSING** | **DATA_MISSING** |

- g2 的聚合 E 可由保存的逐点 (E_a, y_norm_a) 精确复算：SSE_a = E_a²·y_a²。
  注意它比 C0 报告中的"算术平均 E"（R 0.44 / C 0.42）更严格地按冻结公式计算。
- g3 不可复算：C0 运行器只保存了**逐点**半库 E 比值，从未保存半库 SSE / targetSS；
  仅有比值而没有权重，聚合值在数学上不可恢复（改半库范数即可任意改变聚合 E）。
  按协议：标 DATA_MISSING，不从几个 norms 反推。

**Erratum（C0X-X1-01，详见 verdict_comparison.json）**：
- 原 verdict（C0 summary.json）**原样保留、不覆盖**：`LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT`。
- 错误位置：`c0_analyze_v01.py` geom_verdict 半库检查（逐点代替聚合）。
- 按冻结规则复算的 verdict：`LOCAL_GEOMETRY_UNRESOLVED`
  （g1、g2 通过；g3 门值统计量缺失，既不能确认 SUPPORTED 也不能维持 NOT_SUPPORTED；
  "UNRESOLVED" 此处语义 = 门值统计量缺失，非 signal 不足）。
- 敏感性披露（不重选）：C/Bp/t16 的 resolved 判定余量仅 ~7.3e-3（y_norm=0.09191 vs 3·SE=0.08458），
  且该 anchor 占 C cell target SS 的 26.3%；即使把它从 A 剔除（仅为示敏，未执行），
  C cell 聚合 E 仍 ≤0.8。半库聚合门是否通过仍不可知。

**Shadow channel 复算**：A_U 仅 t48 三点（1 个 distinct t），聚合 E_U：R=0.4333、C=0.4056；
g1 计数门失败 → `LOCAL_GEOMETRY_UNRESOLVED` **不变**。

**规则优先级核查**：C0 `live_manifest.json` 的 `budget_events` 仅含
CONFIG_FROZEN / THEORY_TESTS_ALL_OK / IMPLEMENTATION_FROZEN / BUDGET_ADJUSTMENT_USER_APPROVED 四项；
CONFIG_FROZEN 之后无任何规则 amendment → 不构成分歧来源（非 PROTOCOL_AMBIGUOUS）。

## X2. floor 修正后的一致性复算

- 数值底：floor2（same-probe 配对、每 probe 4 次重复）在 grad/hvp/v/q 四量上**精确为 0.0**
  （位级可重复），取代交叉 probe 池化的 1.65。supersession 已在 C0 披露。
- **语义声明（重要）**：floor=0 只代表"同一评测的位级可重复性"，**不代表系统数值误差为 0**。
  未覆盖的系统误差源：HVP 代理截断（保存量 `hvp_first_grad_rel_gap_max` 可见，~5e-3 量级）、
  bf16 主权重、Adam ε 分母、8-probe 采样偏差、轴定义差（见 X3）。C0 的判据均按"分辨率门"
  （3 倍规则）使用 floor，工程意义不变。
- 一致性：field/shadow/半库/all-anchor 的 resolved 标记全部用同一 floor=0 在同一脚本中重算
  （rule_recalculation.csv），不存在混用旧 floor 的标记。原运行器落盘的 runner flag 是旧 floor
  下的产物，已全部废弃并以重算值为准（与 C0 analyzer 的做法一致）。
- 采样维度：C0 未保存 dev 预测侧 SE（PREDICTION_SAMPLING_UNRESOLVED 从未触发）、
  半库均值差范数、固定方向标量投影误差 → 均标 DATA_MISSING（signal_resolution.csv）。

## X3. 同一 V_diag 上的 gain 与累计 W/K（V_eval 另行报告）

约定：`step_records` 行 h = 第 h 步**更新前**状态（已走 h−1 步），h=1..64；
`W_total_prev(h) = −q_{h−1}ᵀs_{h−1}`，`K_V(h) = v(h)−v(h−1)+W_total_prev(h)`（逐行恒等，已验证到 1e-17）。
步 t 的 realized readout = 行 t+1 的 W_total_prev。**h=65 行不存在**（64 次更新后的状态未记录），
最后一个区间因此是 [16,63]（47 步），不做插值、不外推。

恒等式（逐 interval 验证，残差 ≤1.4e-17）：
`gain_interval(native−a) = Σ_t[(W_t^a−W_t^native) − (K_t^a−K_t^native)] = [v_nat(t1)−v_nat(t0)] − [v_a(t1)−v_a(t0)]`

cell R（V_diag，native=A；gain>0 = 该 arm 比 A 好）：

| comparison | [0,4) | [4,16) | [16,63] | 累计 1..63 |
|---|---|---|---|---|
| Bp−A：ΔW | +0.00008 | +0.01812 | +0.00959 | +0.02778 |
| Bp−A：ΔK | +0.00352 | +0.01229 | +0.00402 | +0.01983 |
| Bp−A：**gain** | **−0.00345** | **+0.00583** | **+0.00557** | **+0.00795** |
| Bm−A：gain | +0.00162 | −0.00472 | −0.00939 | −0.01248 |
| Bp−Bm：gain | −0.00507 | +0.01054 | +0.01495 | +0.02043 |

cell C（同结构）：

| comparison | [0,4) | [4,16) | [16,63] | 累计 1..63 |
|---|---|---|---|---|
| Bp−A：gain | +0.00060 | +0.00043 | +0.00576 | +0.00680 |
| Bm−A：gain | −0.00122 | −0.00086 | −0.00353 | −0.00561 |
| Bp−Bm：gain | +0.00183 | +0.00129 | +0.00929 | +0.01240 |

**读法**：
1. R cell：Bp 的 gain 在 [0,4) 为**小负**（−0.0034），[4,16) 转正（+0.0058），[16,63] 维持正
   （+0.0056）——与 A0U 的 Case B 发现（LR 上调收益 ≈ 固定 +5% **延迟约 6 步**兑现）在符号与
   延迟尺度上一致：收益不是瞬时产生，而是经动量/二阶矩再平衡后逐步兑现。C cell 的 Bp−A 在
   [0,16) 近零（+0.0006/+0.0004）、[16,63] 转正（+0.0058），同样非瞬时。
2. Bm（LR 下调）在 two cells 的 gain 均为负且后期更负（R：−0.0047→−0.0094；
   C：−0.0009→−0.0035）；Bp−Bm 三段在两个 cell 全部为正且大体单调——与 A0Q
   "LR 是单调稳定通道" 相容。
3. **量级结构**：gain 是 W、K 两个大得多的流的差（例：C/Bp−A 在 [4,16) ΔW=−0.0547、
   ΔK=−0.0551，几乎相消后 gain=+0.0004）。增益是小残差，这解释了为何需要分辨率门才可分辨，
   也提示 readout 记账下的"收益"对 W、K 各自的大变化敏感。
4. 量纲：63 步累计 gain ~0.006–0.020（V_diag 损失 ~7.2 的 ~0.1–0.3%），属小效应。

**V_eval（另轴）**：`cell_metrics.csv` 的 h=4/16/64 readout（post-update），
差值 native−a 见 gain_balance_by_interval.csv 中 `axis=V_eval` 行；不与 V_diag 混排。

**轴映射（target_axis_map.csv）**：A0T cells 轴 = 全量 v_score（64 样本）均值 CE；
C0 v_diag = v_score 前 8 个 ID（1 chunk）；C0 v_eval = 不相交的 64 样本 split；
B0 stress 轴仅用于位级恒等门。15 组 A0T−C0-v_eval 对照全部差 ~+0.106 常数——
两轴样本集合不相交，非常数部分**离线不可计算**；维持 C0 结论：**AXIS_MISMATCH 是轴定义差，
不是轨迹分歧**（轨迹恒等由 B0 位级门独立确立）。不通过减常数强行复现 reversal。

## X4. 账本复算

- 本轮批准：417 updates（用户批准 +9 超额，BUDGET_ADJUSTMENT_USER_APPROVED）。
- 调和后 C0 计数（跳过 RECONCILIATION 行）：project 累计 **17,994 / 18,000**；
  运行器静态基 under-report 与 attempt-1 崩溃步数已在 C0 REPORT 披露。
- round-total logged forward_calls = 2,428 > 2,400 cap（+28，两次零更新 salvage floor）——
  保持披露，不用 per-process 计数隐藏。
- 旧 18,000 上限保留历史含义，不重置；C1 预算独立记账（ledger_recheck.json）。

## 不确定性与 DATA_MISSING 清单

| 条目 | 状态 |
|---|---|
| 半库**聚合** E（g3 门值） | DATA_MISSING（C0 未保存半库 SSE/targetSS；C1 将按固定 ID 保存充分统计） |
| dev 预测侧采样 SE / PREDICTION_SAMPLING_UNRESOLVED | DATA_MISSING（C0 未保存） |
| 半库均值差范数、固定方向标量投影误差（§10.1 要求） | DATA_MISSING（C0 未保存） |
| U 与 stop-v 相互方向（dot） | DATA_MISSING（只存了 norm） |
| D_i 局部 c→U 导数 | DATA_MISSING（C0 未做；C1 含 stop-v arm 可部分补上） |
| A0T−v_eval 常数偏移的非常数部分 | 不可计算（样本集不相交） |
| h=64 更新后 v_diag 状态 | 未记录（区间止于 63 步） |
| 系统数值误差（HVP 截断等） | 非零；floor2 只证位级可重复性（见 X2） |

## 对 GTTR 各段的净影响（C0X 部分）

- **Geometry（field 局部预测）**：C0 的 NOT_SUPPORTED 降级为规则执行错误 → 修正后 UNRESOLVED。
  可说的正面证据：g1/g2 过门（聚合 E 0.53/0.47），但冻结门无法完整求值，不能宣布支持。
- **Tracking / Readout**：C0 PASS 判定不受 X1 影响（不涉及半库门），维持。
- **Transport**：X3 的 W/K 记账恒等精确成立；延迟兑现形态与 A0U Case B 一致（经验支持，非机制）。

C1 将补齐 g3 门的充分统计（32-probe 新库 + 按固定 ID 的半库/投影统计），
并对 field/shadow 做 cross-bank 与 same-probe 的误差分解。
