# AUDIT REVIEW SUPPLEMENT — 关键证据补充交付

**性质**: 对 fr_gttr_model_target_alignment_audit_v01 的证据补充与勘误。**0 新增模型 forward/backward/HVP/optimizer update**；仅使用已有文件与 CPU 参考重算。
**日期**: 2026-09-22。原 AUDIT_REPORT.md 保留有效，勘误以本节为准（§8 勘误清单）。

---

## 1. Claims 计数解释：25 是 unique 总数，27 是多标签桶计数

- `coverage_matrix.csv` 共 **25 行（C01–C25），即 25 个 unique claims**。
- 上轮摘要写 "2+16+4+4+1=27 覆盖 25" 是**多标签桶计数未声明**：C18 同时计入 PROTOCOL_MISMATCH（已勘误）与勘误后 EMPIRICALLY_COVERED；C04 同时计入 NOT_SUPPORTED（作为预测模型）与"阶梯零级保留"（EMPIRICAL  ladder）。25 + 2 次重复 = 27。
- **不是统计错误，不删改任何条目**；规则明确为"标签允许重叠，primary_verdict 每 claim 唯一"（见 `claims_status.csv`）。
- 按 primary_verdict 计数：SUPPORTED_WITHIN_SCOPE 12 / NOT_SUPPORTED 7 / NOT_TESTED 3 / MIXED 2（C14 效用 SUPPORTED+成本 NOT_SUPPORTED；C15 假设一二 SUPPORTED+假设三 UNRESOLVED——严格说 C15 属 MIXED，修正 v1 将其单列为 SUPPORTED 的口径）/ PARTIAL 1（C24）/ UNRESOLVED 1（C18）。

### 1.1 全部 claim 速览（完整主张/证据路径/状态见 claims_status.csv）

| 层 | claims | primary verdict |
|---|---|---|
| 精确实现 | C01 AdamW语义 / C02 Z_t可恢复 / C03 clip算子 | SUPPORTED_WITHIN_SCOPE |
| 局部近似 | C05 B模型 / C06 C模型 / C07 LR响应 / C10 可控性 / C18 C0场 / C19 C1 / C20 C1L | SUPPORTED（C18 UNRESOLVED） |
| 控制/闭环 | C09 控制图 / C11 A0T冻结规则 | SUPPORTED（in-sample 边界内） |
| 负结果 | C04 A模型 / C08 脉冲核 / C12 schedule / C13 自适应价值 / C16 B0.5三假设 / C17 regime切换 | NOT_SUPPORTED |
| 混合 | C14 效用/成本 / C15 B0三假设 | MIXED |
| 泛化 | C21 G系列 | SUPPORTED（边界内） |
| 未测 | C22 全局几何 / C23 下游能力 / C25 >128步 | NOT_TESTED |
| 部分 | C24 动作全集 | PARTIAL |

---

## 2. 预算对账（run 级，不预设任何总数正确）

### 2.1 直接回答四个问题

**Q: 15,097−14,208=889，为何只说新增 712？剩余 177 是什么？**
**15,097 没有 run 级来源。** 核验了 B0 的 gpu_budget_log.json（9 个 phase）、smoke_checks\*.json、summary.json、B0.5 的 PRIOR_CONSUMED 字段——均无 889 或 177 的记录。15,097 是上轮审计侧的算术数字，**作废**；更正值 **B0 = 14,920 = 14,208 + 712**。177 标 **UNRESOLVED**，不补造。

**Q: 旧上限账 17,994，为何更正后是 18,289？**
18,289 不是 17,994 的更正值，而是**更正后的 C0 前中间累计**：17,577（B0.6 止旧累计）+ 712（B0 smoke）= 18,289。上一轮摘要把它直接当"超 cap"总数，口径错位。17,994 的更正值是 **18,730**（见 2.2）。

**Q: 18,289−712=17,577，是否用了 C0 前旧基数？**
是。17,577 是 B0.6 结束时（C0 之前）的旧现场累计。18,289 = 更正后的 C0 前累计。

**Q: C0 的 417 计入哪个账本？**
计入**第一轮（18,000 cap，B0 起）**。C0 REPORT.md:25 现场记录："approved this round: 417 = 33 disclosed smoke-attempt waste + 384 main"。即 C0 的账**已含 33 失败 smoke**，但**成功 smoke 的 24 updates 未计入任何总数**（budget_log.jsonl 中 smoke 计数器 1..24 独立于 project_cumulative 推进）——这是本次对账的第二个新发现。

### 2.2 第一轮（18,000 cap，B0 起）逐实验对账

| 实验 | phase 分解（run 级） | 旧计数 | 新发现 | 更正 |
|---|---|---|---|---|
| B0 | smoke att1 356（FAIL）+ smoke att2 356 + main 6×2304=13,824 + stress 384 | 14,208（TOTALS 只含 main+stress） | **+712** | **14,920** |
| B0.5 | smoke 9（FAIL）+64 + main 1,536 | 1,609 | 0（已含 smoke） | 1,609 |
| B0.6 | smoke 16×2 + failed 64 + main 1,536 + stress 128 | 1,760 | 0（已含 smoke+failed） | 1,760 |
| C0 | main 384 + failed smoke 33 + **成功 smoke 24（未计）** | 417 | **+24** | **441** |
| **第一轮合计** | | **17,994** | **+736** | **18,730 → 超 cap 730** |

证据位置：B0 `gpu_budget_log.json` phases + `budget_after_smoke.smoke=356`；B0.5 `TOTALS` + `NOTE`（含 attempt-1 覆盖事故说明）；B0.6 phases；C0 `budget_log.jsonl`（main 384 行 + RECONCILIATION notes + smoke counter 1..24）与 `REPORT.md:25`。

### 2.3 三个口径的输出

1. **原 18,000 授权范围（第一轮）**：旧 17,994 → 更正 **18,730**，**超限 730**（712 B0 smoke + 24 C0 成功 smoke）。历史 cap 数值本身不改。
2. **独立授权轮次**：C1 = 45 updates（cap 192, within）；C1-Late = 106 updates（cap 128, within）。均与第一轮分开记账（协议 §0）。
3. **全项目实际累计**：**刻意不合并为单一数字**——A0 系列为更早独立轮（预算单位 wall-seconds/evals，a0k–a0u 的 optimizer_updates=0）；G 系列（g1a/g1b/g2/g3）各自独立授权，manifest 只记 128 planned×1024 tokens、terminated at 102，per-arm 汇总不在 manifest → 精确总数 UNRESOLVED；G1c 为可行性/诊断轮，无 update ledger。**未获 run 级依据的差额一律 UNRESOLVED。**

完整表：`cost_target_audit_v2.csv`（每 phase 一行，含 source_path 与记录位置）。

---

## 3. 九个目标轴数据字典

口径：**无证据的泄漏不写成已发生**。逐轴核验后，"A0T/A0U in-sample" 的精确含义是：

1. **数据训练重叠：否（无证据）**。V_score/V_eval 来自 `a0g validation_split` 冻结划分，eval_indices 与 train_indices 按样本 ID 不相交；probe_q 为独立 32 条。未发现 train/eval ID 重叠证据——**不声称发生泄漏**。
2. **反复用于选择：是**。V_score 参与 A0T/A0U 的 controller/schedule 选择与全部 chi/FD 参考（开发轴）；V_eval 参与 A0G/A0K/A0L 动作效用评价；C0/C1 probe 取自开发 bank 态。**反复用于选择 ⇒ 开发信息，不构成独立 final**——这是"in-sample"的确切含义。
3. **开发 state/stream：是**。A0T 的控制信号 chi 与评估 rollout 共用 bank 衍生 128-micro 流（同流，in-sample）；C1L t48 为重建态。

| target_id | 数据/样本数 | tokenizer/context | reduction/mask | 模式/精度 | 用途 | 参与选动作/调参/停止 | 重叠 | 独立final | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| L_train | D0 前4096行（train_indices=3936），~8条/step | Minimind vocab6400 ctx128 | CE mean/valid tokens ÷accum(2)；ignore_index=-100，每行127预测位 | train dropout=0 bf16+fp32 master | 训练介质 | 否（干预经train batch施加） | 与probe/eval按ID不相交 | 否 | e3c config；target_dictionary.csv |
| probe_q | D0 内冻结32条（seed注册） | 同上 | eval前向+autograd.grad逐分量，无reduction | eval bf16(A0C)/fp32(C1 HVP) | 干预方向构造=q | 是（方向来源） | 与train/eval不相交 | 否 | a0c/c1 configs |
| V_diag | v_score 前8 ID，1 chunk | 同上 | CE mean/valid | eval fp32 | C0/C1记账读出（W/K，q+固定方向） | 否 | 是 V_score 子集 | 否 | c0/c1 configs |
| V_score | sorted(eval_indices)偶数位，64 | 同上 | 逐batch CE mean→np.mean(8批) | eval bf16 | A0C/A0O–A0U 全部 chi 与 FD 参考 | **是（A0T/A0U选择与chi）** | 与V_eval不相交；与评估同开发流 | **否** | a0g validation_split |
| V_eval | sorted(eval_indices)奇数位，64 | 同上 | 同V_score | eval bf16(A0G/A0K)/fp32(A0H20) | A0G/A0H/A0K/A0L 决策效用 | **是（动作效用评价与选择）** | 与V_score不相交；被反复用于动作评价 | **否（最接近final但已参与选择）** | 同上 |
| v_monitor | — | — | — | — | 仅见于审计建议文档 | — | — | 不存在 | 无生产代码 |
| V_final | — | — | — | — | — | — | — | **不存在（无独立最终测试集）** | 无 |
| G1a/G1b_train | D0，seeds 2042-2044，stage 26/64/102 | 同上 | 同L_train | bf16 | 26M/100M scale-only 复制训练 | 否 | — | 否 | g1a/g1b training_manifest |
| G2_train | D1 pretrain_hq(rev 3a9bbe61)，seed 60202601 前4096 | 同上 | 同L_train | bf16 | 数据分布泛化训练 | 否 | 与D0 shingle Jaccard 1.6e-7 | 否 | g2 training_manifest |

---

## 4. 19 条 arm 同轴时域表

完整列（干预长度/观测长度/累计 tokens/gain/gain_vs_fixed/不确定性/continuation）见 `horizon_alignment_v2.csv`。关键行（含 gain 的）直接摘录：

| arm | 目标轴 | 干预长度 | 观测长度 | updates×tokens | gain_vs_native | gain_vs_fixed | 不确定性 | continuation |
|---|---|---|---|---|---|---|---|---|
| A07 A0T (36cell) | v_score | H{4,16,64} | 同左 | 128×1024/cell | H4 median **+4.41e-4**（sign 35/36）；H64 median +4.7e-3（2 细胞 H16/64 反转） | ≈0（C≡B 34/36 别名） | n=36，median/IQR | 是 |
| A08 A0U (12cell) | v_score | W=64 | 64 | 64×1024 | 正（Case B dominant）；B−C final n.s. | AUC B−C **p=1.2e-2**（C 更差） | n=12 | 是 |
| A09 A0G–J | v_eval | H=4 选择 | H=20 | — | +1.467e-4，PI [+8.7e-5, +2.1e-4] | NA | n=12，决策保持 12/12 | 是 |
| A10 A0K/L/N | v_eval@128 | T=128 | 128 | 128×1024 | A0K C−A +7.44e-4；A0L B−C +7.46e-3；A0N B−C −6.78e-4 | 负/混合（C≤B 6/9） | n=3–6 pilot | 是 |
| A11 B0 (216 main+6 stress) | v_eval | H{4,16,64} | 64 | 64×1024 | NA 单值：H4 delta 恒 0（0/24）；H64 disp 0.109；内点极小 4/36 全 hindsight | NA | 216 arms | 是 |
| A18 G3 | v_eval | T 窗 | 同 | 102×1024 | **+0.094**（clip 因果） | NA | 单规模单配置 | 否 |

其余 arm（A01–A06, A12–A17, A19）为诊断/复制/未测，gain 列按协议填 **NA**（如 C1/C1L 的 E=0.49/0.36 是误差比，不是 gain）。

两条纪律已执行：
- **不把已知位移的局部预测当 H48/H64 预测**：A14–A16（C0/C1/C1L）信息时点 = 已知真实 s/Splus/qplus 的动作后局部条件近似（CPU T1 PASS 证实信息门有效）；A16 的 t48 是**分叉后索引**（全局 step94/128=73%），且为**重建态**（bank 无该 step）。
- **不同目标不做常数平移合并**：v_score / v_eval / v_diag / 诊断统计量分列，未做任何跨轴平移。

---

## 5. 数学桥梁与 CPU 反例

### 5.1 编号映射（重要）

协议 T1–T9 = **CPU 反例测试**（cpu_counterexamples.json）。AUDIT_REPORT.md v1 的"T1/T2/T3"是**经验发现**，编号冲突，重命名：

| v1 编号 | 新编号 | 内容 | 与协议 CPU 测试的关系 |
|---|---|---|---|
| T1 | **E1** | A0T-extra：2 细胞 H=4 赢、H=16/64 反转 | 与协议 T2（单谷时域反转）同**主题**；E1 是本训练设置中的观测 |
| T2 | **E2** | A0P：脉冲核长尾，无稳定时间尺度 | 与协议 T2/T7 主题相关 |
| T3 | **E3** | B0.6：rho=0.996 近单位单图，切换族无增益 | 与协议 T3（局部地形≠全局）主题相关 |

**纪律**：CPU 测试（T1–T9）证明"可能发生"，E1/E3 是"在此训练设置中观察到"。**不得把 1 个数学反例 + 1 个经验观测称作多次独立 LLM 实验证据。**

### 5.2 HYP-1/2/3 原文与修订状态（theorem_assumption_checks.csv）

| ID | 主张原文 | 证据 | 修订状态 |
|---|---|---|---|
| HYP-1 | 温度/熵产生/广义 Langevin 参数可解释训练收益 | 无证明单调关系；无量纲桥梁；未与独立任务目标校准 | **NOT_SUPPORTED（作为目标代理）**；物理量单位不得人为换成 loss/joules |
| HYP-2 | Hessian trace/λ_max/‖g‖/W/K 分量作为能力指标 | 无单调界或任务校准；候选诊断量 | **NOT_SUPPORTED（作为能力指标）**；T4/T5 反例 |
| HYP-3 | 短期(H≤64)干预 gain 外推最终训练改进 | 违反证据：A0P 长尾（E2）、A0T-extra 反转（E1）、A0S FAIL-case-2 | **CONTRADICTED_OR_MISIMPLEMENTED（作为外推）**；B2 无已验证 Lyapunov 函数；B1 的 ε 界不存在 |

### 5.3 T1–T9 数值（全部 PASS；至少 T2/T4/T6/T8/T9 已按要求直接展示）

| # | 测试问题 | 解析预期 | 实际数值 | 误差/反向检查 | 结论边界 |
|---|---|---|---|---|---|
| T1 | 已知位移≠开环预测：动作前访问真实 s/Splus/qplus 必须被信息门拒绝 | RuntimeError | leak_detected=true，pre_action_reads_ok=true | 无信息门时 stub 在动作前读到真实 s（RuntimeError 抓到） | field/shadow 是已知真实位移的局部条件近似，非动作前可用开环预测，更非 H48/H64 |
| **T2** | 单谷短长时域反转：V=0.5(x²+6y²)，GD η0=1/4，前4步改 η | dV_H/dη_pulse=−4Σλ²x0²(1−η0λ)^(2H−1) | H4：解析 **0.591064** vs 中心差分 **0.591076**；H16：解析 **−0.00053568** vs 差分 **−0.00053568** | 符号反转确认（+→−）；差分与解析一致 | 反转无需切换/非凸/模型错误——纯数学事实 |
| T3 | 相同局部地形≠全局结果：f2=f1+A·bump | 局部邻域（|x|<0.5）f 与导数全等 | local gap=**0.0**，local dgrad gap=**0.0**；屏障中心 gap=**3.678794**；GD 从 x=0：f1 终点 3.0，f2 小步长 0.803 / 大步长 3.0 | 大步长可翻越屏障（轨迹随步长改变） | 局部测量不能无条件恢复全地形；不声称 FR 真实轨迹跨过此类障碍 |
| **T4** | 相同更新范数≠同目标收益：V=0.5‖θ−(1,0)‖²，θ=(0,0) | ‖s+‖=‖s−‖=0.1，ΔV 分别 −0.095/+0.105 | norm+ = norm− = **0.1**；ΔV+ = **−0.095**，ΔV− = **+0.105** | 标量范数/温度无法区分二者 | 目标读取方向不可被范数替代 |
| T5 | 曲率重参数化≠能力变化：θ=aφ | 同物理点 f 不变，φ 坐标 Hessian=a² | a=1: f=0, H=1；a=3: f=0, **H=9** | — | 不声称相同 η 下轨迹不变；原始曲率指标不能无条件作能力标尺 |
| **T6** | 步数/tokens/成本排序可相反 | B 步数更少但 tokens 更多 | A：100 步/102,400 upd tok；B：80 步/**163,840** upd tok；probe/candidate 开销 122,880 两者；total A=225,280 B=286,720 | fewer_steps_more_tokens=**true** | probe/candidate 开销必须入账，不算免费 |
| T7 | 未达标路径不能消失 | hit_rate=3/5=0.6；2 条右删失保留 | hit=3/5，censored=2，末端损失 [0.55, 0.61]；只对成功者平均得 τ̄=90，朴素"节约"38 步被**拒绝**；稀疏网格 {50,100} 的插值 τ=60 已披露不确定区间 | 反向检查：删失败 run 会丢 hit_rate 字段（JSON 强制保留） | 删失上限≠真实达标时间 |
| **T8** | 聚合门 vs 单点门 | 6×0.43+1×1.166，等 TSS | 聚合 E=**0.535**（<1 PASS）；最大单点 **1.166**（>1 FAIL） | 两字段并列暴露差异；这正是 C0X erratum 的数值形态 | 逐点 max 不得替代固定聚合 SSE/TSS |
| **T9** | 统计恢复条件：K=16, D=20 合成向量 | mean‖X‖²=‖X̄‖²+均方偏差；SE_vec²=SE_norm²+(a²−b²)/(K−1) | a=1.8070, SE_norm=0.05397, b=1.2881, SE_vec²=0.10999；分解残差 **0.0**；SE_norm vs SE_vec 差 **0.2777** | 把 SE_norm 当 SE_vec 的混淆被 gap 暴露 | 恢复仅 same-K/同对象/同权重/独立分组成立；旧 raw 缺失≠可恢复 |

`validity_domain.csv`（协议 §8 要求）v1 缺失，本补充已按现有证据补齐（11 行，D01–D11），无新增计算。

---

## 6. 审计真正回答了什么（七问分别回答，非单一 headline）

| # | 问题 | 回答 | 边界 |
|---|---|---|---|
| Q1 | AdamW 更新语义覆盖 | **EXACT**：实现=原生 torch.optim.AdamW 默认，GTTR 骨架公式一致（C01–C03，EXACT-1/2/3） | foreach/fused/AMSGrad 未走；scaler disabled；wd 无 mask |
| Q2 | 局部几何近似覆盖 | **H≤8 有界覆盖**：C 模型 rel err 0.0045–0.0219；H=1 恒等；clip-JVP+HVP 光滑传播；已知真实位移的局部条件近似（field/shadow）在 4–6 anchors 聚合过门 | H>8 未测；非光滑残差未建模；HVP 代理截断 ~5e-3；**条件近似 ≠ 自主多步预测**（CPU T1） |
| Q3 | 目标间关系 | 训练场↔验证 NLL：同口径；几何代理量→目标：**无已证单调关系**（HYP-1/2 降级）；短期 NLL gain→长期/能力：**无桥梁**（HYP-3 违反） | 相关性≠可优化关系 |
| Q4 | 局部预测是否被误写成自主预测 | v1 审计已拦截：C0/C1 全部标注"已知真实 s/Splus/qplus 的动作后局部条件近似"；C1 预测先落盘再 audit；CPU T1 信息门 PASS | C1L t48=重建态已披露 |
| Q5 | tokens-to-target 单位/起点/停止/总开销 | **NOT_TESTED**：无 ell_star 阈值（无结果前约定）、无 monitor 协议、无独立 final；token 计量已规范（1024 tokens/update step=2×8×128÷accum 口径已核）；成本字段完整率：updates/evals/wall_s 有，joules 无 | 全部"达标"声明禁止；continuation≠全预训练节约 |
| Q6 | 不预测全轨迹时的合法桥梁 | B1：无 ε 界（field/shadow 向量 E 不是全动作集合上的上界）；B2：无已验证 Lyapunov/progress 证书（B0.6 排除离散切换后剩余连续非平稳/非线性/投影外变量）；B3：原则可行（直接测累计成本）但需独立同目标读出+固定基线+匹配调参——A0L 的 +7.46e-3 是 in-sample，不满足 | 优化器可能利用模型误差：chi 信息时点滞后正是被利用的误差面 |
| Q7 | 现在能支持什么结论 | 可保留：bank 保真/重放恒等/AdamW-clip 一致/C 模型 H≤8/G 系列强复制/clip 因果/冻结 sign 规则（H=4, in-sample）；带边界：LR 控制图、可控性、局部场；**必须降级/勘误**：B0 台账、HYP-1/2/3、短期 gain→最终收益外推 | 最小缺口=N1 类长时 continuation + 预注册阈值 |

**两点澄清（按用户要求）**：
- 没有下游任务**不能**宣称能力提升；但也**不能**因此否定限定于独立 LM-NLL 轴的效率研究——该轴内结论自洽（双分裂 V_score/V_eval、防泄漏落盘、冻结规则）。
- 没有长期/成本证据 ⇒ tokens-to-target 为 **NOT_TESTED**；不因"所有读出都叫 NLL"就判长期目标已对齐，也不因缺长期数据而把已有 NLL 结论判 FAIL。

---

## 7. 下一步设计依据（不执行）

### N1（唯一提出项）依据审计

- **T=512 的来源：HORIZON_UNJUSTIFIED。** 现有 baseline 曲线全部 ≤128 步（v07 continuation、B0/B0.6 为 64 步窗、A0K/L/N 为 128 pilot）；无结果前约定的 ell_star，目标距离未知；v07 计划训练量为 128 步/轮（training_manifest 同口径），512 = 4× 任何现有运行长度，**无 baseline 曲线、计划量或目标距离支持**。
- 按协议 §9，候选长度必须由 baseline 曲线+计划训练量+目标距离+预算决定；**当前四项均不具备 ⇒ 标 HORIZON_UNJUSTIFIED**，不换一个任意时长充数。
- **N1 在比较什么**：当前**不存在在线 GTTR 策略**（chi 需模拟未来流，信息时点=模拟后）。实际可执行比较 = native vs 固定 ±5% LR vs 冻结 sign(chi) 规则（offline 信号）。⇒ **objective feasibility study**（协议 §9.4 口径），不发明控制器充数。
- N1 状态：**未授权，0 updates**。

### N2/N3/N4 = 未授权备选，不与 N1 捆绑

- 不存在任何 4,900 updates 的批准。v1 NEXT_VALIDATION_SPEC 的合计数字撤销为"估算草案，未批准"。
- **N4 的 "1B" = 参数量**（~1B 外部 dense 自回归 Transformer  with 真实 optimizer state），**非 token 量**；OLMoE-1B-7B 因 MoE 架构被 G1c 排除（EXTERNAL_MODEL_PROVENANCE.md）。候选 ladder 受控部分止于 100M（G1b）。

---

## 8. 勘误清单（追加，原报告保留）

| # | 位置 | 旧 | 更正 | 依据 |
|---|---|---|---|---|
| ERR-S1 | cost_target_audit.csv / audit_summary.json | B0=15,097；新增 889；超 cap 289 | **B0=14,920；新增 712；177=UNRESOLVED**；第一轮更正 18,730/超 cap 730 | B0 gpu_budget_log phases；smoke jsons |
| ERR-S2 | audit_summary.json | 桶计数 27 未声明多标签 | 25 unique + 多标签规则；C14/C15 改 MIXED | claims_status.csv |
| ERR-S3 | AUDIT_REPORT.md §7 | "18,289 超 cap" | 18,289=**C0 前**更正中间值；第一轮（含 C0 441）=18,730；另发现 C0 成功 smoke 24 未计 | C0 REPORT.md:25 + budget_log.jsonl |
| ERR-S4 | AUDIT_REPORT.md §5 T1–T3 | 与协议 CPU T 编号冲突 | 经验发现改号 **E1–E3**，映射见 §5.1 | 协议 §7 vs cpu_counterexamples.json |
| ERR-S5 | v1 文件 | validity_domain.csv 缺失 | 已补齐（D01–D11，纯合成现有证据） | validity_domain.csv |

## 9. "原摘要 — 数据 — 更正结论"表

| 原摘要（v1） | run 级数据 | 更正结论 |
|---|---|---|
| "B0 实际 15,097（含 smoke 712），超 cap 289" | B0 phases: 356+356+13,824+384；TOTALS=14,208 | **B0=14,920**；超 cap 数字见下行；15,097/177 作废/UNRESOLVED |
| "更正后累计 18,289 超 cap" | 17,994+712+24=18,730 | **第一轮更正=18,730，超 cap 730**；18,289 仅为 C0 前中间值 |
| "27 项覆盖分布" | claims_status.csv 25 行 | **25 unique claims，多标签**；primary verdict 分布见 §1 |
| "T1/T2/T3 三反例" | cpu_counterexamples.json 9 项 + 经验记录 | **协议 T1–T9（CPU）与 E1–E3（经验）分列**，禁止混计 |
| "C0=417 已计入" | C0 REPORT:25 + smoke counter 24 | C0 更正 **441**（成功 smoke 24 新发现） |
| "N1 T=512，预算 ~4,900" | 无 baseline 依据；无在线策略 | **HORIZON_UNJUSTIFIED**；objective feasibility study；N2–N4 未授权；无 4,900 批准 |
