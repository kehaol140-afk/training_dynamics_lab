# FR-C0X audit trail（v01，2026-09-21）

## 性质
纯离线复算：0 optimizer updates、0 gradient evals、0 HVP、0 micro-forwards、0 GPU 秒。
被审对象：`outputs/fr_c0_gttr_validation_v01/` 全部冻结工件（sha256 记录于本目录 config.json）。
未修改任何 C0 文件；原 C0 verdict 原样保留，erratum 以追加形式存在于 verdict_comparison.json。

## 规则来源与优先级
1. 被复算规则：`FR_C0_Geometry_Tracking_Validation_Claude_Code.md` §10.1–§10.3（行 345–389 已逐字核对）。
2. 规则优先级核查：C0 `live_manifest.json.budget_events` 共 4 项事件
   （CONFIG_FROZEN / THEORY_TESTS_ALL_OK / IMPLEMENTATION_FROZEN / BUDGET_ADJUSTMENT_USER_APPROVED），
   CONFIG_FROZEN 之后无任何规则 amendment → 不存在后验改规则；结论非 PROTOCOL_AMBIGUOUS。

## 关键裁决记录
- **X1**：冻结半库门（§10.2 行 366）= 同一 resolved 集合 A 上的**聚合** E ≤ 1；
  C0 分析器实现为**逐点** E ≤ 1（c0_analyze_v01.py geom_verdict，约 L294–306）。
  C/Bp/t16 last4 = 1.166 为**单点**比值（anchor_records.jsonl 核实），非聚合指标。
  → ERRATUM_REQUIRED；复算 field verdict = LOCAL_GEOMETRY_UNRESOLVED；
  半库聚合 E 因 C0 未保存半库 SSE/targetSS 而 DATA_MISSING。
- **X2**：floor2（same-probe，=0.0）为唯一数值底；所有 resolved 标记由本脚本重算；
  floor=0 仅证位级可重复性，系统误差（HVP 截断等）另行披露。
- **X3**：行级记账恒等 K_V = Δv + W_prev 全表验证（≤1.4e-17）；区间 gain 恒等复算一致；
  h=65 行不存在，末区间止于 63 步，未插值；V_diag 与 V_eval 分轴报告；A0T 轴差维持 AXIS_MISMATCH。
- **X4**：417 批准有据（budget_log 用户批准事件）；调和累计 17,994/18,000；
  forward_calls round-total 2,428 > 2,400（+28）保持披露。

## 复算可审计性
- 全部复算由 `research/scripts/c0x_analyze_v01.py` 一个脚本完成（确定性、无 GPU、无随机性）。
- 复现命令：`CUDA_VISIBLE_DEVICES="" PYTHONPATH=research/src python research/scripts/c0x_analyze_v01.py`
- 输出：rule_recalculation.csv、verdict_comparison.json、signal_resolution.csv、
  gain_balance_by_interval.csv、target_axis_map.csv、ledger_recheck.json、
  fig_c0x1_halfbank_gate.png、fig_c0x2_gain_intervals.png、config.json、REPORT.md。

## 已知限制
- 半库聚合门不可从 C0 工件恢复（保存格式缺陷：只存比值不存 SSE/TSS）——由 C1 的新探针库
  按固定 ID 重新采集充分统计，但按协议 C1 不作为 C0 的重新验收。
- A0T−v_eval 偏移的非常数部分不可计算（样本集不相交）。
