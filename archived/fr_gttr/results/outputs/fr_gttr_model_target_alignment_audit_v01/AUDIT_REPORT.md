# FR/GTTR Model-Target Alignment Audit — AUDIT REPORT

**审计轮次**: fr_gttr_model_target_alignment_audit_v01
**日期**: 2026-09-22
**范围**: A0C→C1L 全部实验 + G系列泛化 + B0/B0.5/B0.6 机制链 + 预算台账
**执行方式**: 5个并行深读代理(代码实现核验 / 实验覆盖核验 / 数学对象核验 / 目标对齐核验 / 预算台账核验) + 主代理汇总

---

## 1. 总体判定

**MISALIGNED_FOR_DEPLOYMENT_CLAIMS**（对部署/能力声明不对齐；对内部LM-NLL轴研究结论对齐）。

全部实验读出均为 LM NLL 轴（V_score / V_eval / V_diag），无独立 final 测试集，无下游能力评测。控制器与 schedule 结论仅限冻结规则、in-sample 流。短期（H≤64）干预 gain 已被三处独立证据（T1/T2/T3）证伪为最终训练收益的代理。**LM NLL 降低不得宣传为通用能力提高。**

## 2. 覆盖矩阵摘要（25 claims）

| 层级 | 计数 | 代表 |
|---|---|---|
| EXACT_FOR_IMPLEMENTATION | 2 | C01 AdamW语义 / C02 状态可恢复性 |
| EMPIRICALLY_COVERED | 16 | C03-C17, C21, C24 |
| EMPIRICALLY_COVERED（负结果） | — | C04, C08, C16, C17 |
| NOT_SUPPORTED | 4 | C04(A模型), C08(脉冲核), C16(三假设), C17(regime切换) |
| NOT_TESTED | 4 | C22(全局几何), C23(下游能力), C25(>128步), 1B |
| PROTOCOL_MISMATCH（已勘误） | 1 | C18(C0聚合门) |

完整表见 [coverage_matrix.csv](coverage_matrix.csv)。

## 3. 精确恒等式核验（全部 PASS）

| ID | 恒等式 | 证据 | 状态 |
|---|---|---|---|
| EXACT-1 | AdamW单步更新语义 | e3c:138; a0c:211-247 与 GTTR 骨架一致 | SUPPORTED_WITHIN_SCOPE |
| EXACT-2 | Z_t 可恢复性（56态bank+RNG+hash全等） | full_state.py; rng.py; a0c 24/24 shadow hash | SUPPORTED_WITHIN_SCOPE |
| EXACT-3 | clip 算子（tau=1.0 全局L2, unscale后） | c0:92-107 | SUPPORTED_WITHIN_SCOPE |
| EXACT-4 | 记账恒等式 DeltaV=-W+K; K=DeltaV+W | fp64残差≤2.8e-20 | SUPPORTED_WITHIN_SCOPE |
| EXACT-5 | 聚合门 E=sqrt(ΣSSE/ΣTSS) | 68/68行独立复算吻合 | SUPPORTED_WITHIN_SCOPE |
| EXACT-6 | 统计充分统计恒等 | 48/48行 diff=0.0 | SUPPORTED_WITHIN_SCOPE |

**残余限制**: foreach/fused/AMSGrad 路径未走；scaler disabled 未建模；theta_protocol.py 源码缺失（仅 pyc，已反汇编重建，语义已核验）——审计可见性缺口，非语义缺口。

## 4. Gate 结果

| Gate | 结果 |
|---|---|
| G1 bank fidelity | PASS |
| G2 replay identity | PASS |
| G3 impulse response | **FAIL**（A0P 无稳定时间尺度） |
| G4 recoverability | PASS |
| G5 glossary | PASS |
| G6 checkpoint fidelity | PASS |

## 5. 关键反例（已登记）

1. **T1**（A0T-extra）: 2 偏差细胞 H=4 赢、H=16/64 反转 → 短期≠长期。
2. **T2**（A0P）: 脉冲核长尾，半衰期4-8步+长记忆噪声尾，gate FAIL → 短期核外推失败。
3. **T3**（B0.6）: rho=0.996 近单位单图拟合不劣于任何 switching 族 → 局部地形不唯一决定全局。
4. **T4/T5**: 范数与曲率不决定目标收益（HYP-2 反例）。

## 6. 假设检验状态（3项 downgrade）

| 假设 | 状态 |
|---|---|
| HYP-1 物理量（温度/熵产生）为目标代理 | NOT_SUPPORTED |
| HYP-2 Hessian trace/λ_max/‖g‖/W/K 为能力指标 | NOT_SUPPORTED |
| HYP-3 短期(H≤64)干预 gain 外推最终训练改进 | CONTRADICTED_OR_MISIMPLEMENTED |

**B2 桥梁（证书）无任何已验证 Lyapunov 函数；B1 的 epsilon 界不存在。**

## 7. 预算台账审计 — 新发现（HIGH）

**B0 smoke 712 updates 未计入 B0 TOTALS。** B0 TOTALS 只计 main(11,280)+stress(2,928)=14,208；smoke 712 在独立目录存在但未进汇总。若计入，B0 累计 = 15,097 > 15,097 之前申报值，且 **B0 已超 18,000 cap by 289**（14,208+712+B0.5+B0.6 = 18,289 在 B0.6 后超 cap 49，至 C0 起全面超 cap）。

**行动**: 发布 erratum 更新台账基数至 18,289；后续所有实验以 18,289 为累计起点。完整表见 [cost_target_audit.csv](cost_target_audit.csv)。

## 8. Errata 登记（本审计新发布）

1. **ERR-B0-LEDGER**: B0 smoke 712 漏计 → 累计基数应为 18,289（HIGH）。
2. C0X 已发勘误维持有效（聚合门误用）。
3. C1-Late t48 = 全局 step94/128 (73%) 重建态，bank 无该 step —— 非原训练真实状态点（MEDIUM，已在 C1L 报告披露）。

## 9. 覆盖缺口（NOT_TESTED 完整清单）

- C22 全局几何（跨谷 / 慢流形 / 非局部障碍）
- C23 下游任务能力（无任何评测）
- C25 >128 步时域（全部证据 ≤128 步）
- 1B 外部架构（G1c 可行性 READY 未启动）
- clip 激活态连续控制 / 联合动作（A0Q）
- beta/eps/batch size 动作（C24）
- 长期（>64 步）收益预测（A0S FAIL-case-2）

## 10. 结论与发布建议

1. **可发布**: bank 保真度、重放恒等式、AdamW/clip 实现一致性、C 模型 H≤8 预测力、G 系列强复制、clip 因果（G3, +0.094）。
2. **带边界发布**: LR 控制图（H=4 纯 θ 通道）、可控性 C_t→G_t（H=4, 同流）、局部场 field（4-6 anchors）。
3. **不得发布**: 任何"提高模型能力"声明；短期 gain→最终收益外推；物理量/Hessian 为目标代理；部署 break-even 成本结论。
4. **必需 erratum**: B0 台账（§7）。

---

*证据文件: coverage_matrix.csv / theorem_assumption_checks.csv / target_dictionary.csv / horizon_alignment.csv / cost_target_audit.csv / audit_summary.json / KEY_DATA_FOR_REVIEW.md / NEXT_VALIDATION_SPEC.md*

---

## 勘误追加（2026-09-22, AUDIT_REVIEW_SUPPLEMENT.md 为准）

本报告 §5/§7 及 audit_summary.json 的预算与计数口径以下列更正为准（详见 AUDIT_REVIEW_SUPPLEMENT.md §8 勘误清单 ERR-S1..S5）：

1. **ERR-S1（预算）**: B0 更正值 = **14,920**（14,208 TOTALS + 712 smoke = 2×356，见 gpu_budget_log.json phases）；旧文 "15,097/新增 889/剩余 177" 为审计侧算术，无 run 级依据，作废，177 标 UNRESOLVED。第一轮（18,000 cap）更正累计 = **18,730**（超 cap 730 = B0 smoke 712 + C0 成功 smoke 24）。旧文 "18,289 超 cap" 口径错位：18,289 是 **C0 前**中间值（17,577+712）。
2. **ERR-S2（计数）**: 25 claims 为 unique 总数；27 是多标签桶计数（C18、C04 各双标签），非统计错误，不删条目。
3. **ERR-S4（编号）**: 本文 §5 的 T1/T2/T3（经验发现）与协议 CPU 测试 T1–T9 编号冲突，重命名为 **E1/E2/E3**。
4. **ERR-S5（补交付）**: validity_domain.csv（协议 §8）已补齐；claims_status.csv / cost_target_audit_v2.csv / horizon_alignment_v2.csv / audit_summary_v2.json 一并交付。
5. **N1 状态**: T=512 标 **HORIZON_UNJUSTIFIED**；N1 为 objective feasibility study（无在线 GTTR 策略）；N2/N3/N4 未授权，"~4,900 updates" 非批准。1B = 参数量（~1B dense 外部 Transformer），非 token 量。
