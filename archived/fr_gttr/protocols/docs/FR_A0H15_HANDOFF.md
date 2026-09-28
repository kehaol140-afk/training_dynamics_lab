# FR A0H-1.5 交接 —— VJP 候选共享伴随

> 2026-09-15。上游：A0G（冻结）、A0H-1（伴随等价，原 FAIL 保留）、
> A0H-1.5 协议 `docs/FR_A0H15_PROTOCOL.md`（含事后附录 A 执行记录）。
> 本文回答核心问题并给出 A0H-2 小规模验收/计时的精确入口。

## 0. 核心问题与答案

**问：冻结的 BF16 前向响应打分器能否支撑候选共享反向打分器，在原始数值标准下
一致，且不预设数值 Hessian 自伴？**

1. **共享伴随已获得（是）**。w_0 只由公共基轨迹 + 终点 V_score 梯度导出，
   `ĥ_j = ⟨w_0, a_j⟩`；CPU 非对称算子合成 5/5 精确转置。**更正（第二轮 §2）**：
   第一轮"GPU 同进程位级一致"声明过强——确定性探针 `/tmp/test_determinism.py`
   在打印 a0h15 一侧前即因键类型崩溃，**从未完成比较**；可用的跨运行证据为
   H=1/4 近位级（~5e-11）、H=8 ~1e-6（BF16 地板）。第二轮生产入口自校验中，
   入口 FP32 前向/伴随与冻结 pass 同进程**位级相等**（gap=0）已确立（新证据）。
   K 的线性性在 BF16 地板内
   成立（CHK-LIN）；SA≈IVJP 缺陷证明主导误差是 BF16 HVP 代理非自伴而非实现。
2. **原始数值标准：部分满足（如实分级）**。
   - H=1：逐元 gate PASS（max|Δ| ~5e-11，D1）。
   - H=4：逐元 gate PASS（max|Δ| 1.36e-6 ≤ 容差 1.84e-6，D1）。
   - H=8：逐元 gate FAIL——与冻结 C|8 原 FAIL 同一签名（BF16 HVP 噪声地板），
     非伴随回归；A0H-1 的 fp32 探针显示前向自身地板即达此量级（**更正**：这是
     强证据但非"对任何伴随"的全称证明；且该说法仅限 H=8——H=4 存在 PASS，
     不得外推为 H≥4）。
3. **不预设自伴：满足**。共享伴随只用 K 的独立 VJP（三阶链 35/35 可行），
   未用 K=Kᵀ；CPU 非对称 K≠Kᵀ 测试 5/5 通过。
4. **独立决策保持：dev 级成立**。
   - BF16 共享伴随 vs 冻结前向：D1 全 H 排序/选择一致；降级批 dev-validation
     9 组合全 decision_match=True。
   - FP32-HVP 配置 vs 冻结 BF16 C：5 键全排序（4 候选 + no-op）9/9 完全一致。
   - **独立验收集验收：NOT_COVERED**——A 批在调试中暴露 accept runner 重建层
     bug，按协议 §1 读一次规则整批降为开发数据（
     `contract/acceptance_disposition_readonce.md`）。不追认为未见数据。
5. **计时（A0H-2 小规模）：NOT_COVERED**。§6/§7 触发条件要求独立验收集决策匹配；
   本轮只有 dev 级证据。预算未耗尽（822s/1800s），非资源原因。

## 1. 分支与逐项状态

| 项 | 状态 | 证据 |
|---|---|---|
| CPU 非对称算子共享伴随（≤1e-6） | PASS | `outputs/a0h15_vjp_shared_v01/cpu_test/test_a0h15_vjp.txt`（5/5 OK） |
| 封装器 forward 一致性（GPU） | 更正后：近位级（非位级） | 原"位级"探针崩溃未完成（第二轮 §2 更正）；实际证据 H=1/4 ~5e-11、H=8 ~1e-6；第二轮入口自校验同进程 gap=0（新） |
| CHK-LIN 线性性（BF16 地板内） | PASS | `diagnostics/diag_2031_52_s0.json` diag.lin |
| CHK-SA / CHK-IVJP | PASS（探针可行）+ 发现：缺陷同源于 BF16 HVP 非自伴 | 同上 diag.sa / diag.ivjp |
| CHK-TELE 望远镜闭包 | FAIL（最差 8.9e-3 > atol_tele=1e-3；按 §3 仅为一致性诊断，不证明根因） | 同上 tele |
| 原始 gate H=1（BF16 共享伴随 vs 冻结前向） | PASS（~5e-11） | 同上 vjp_shared vs `outputs/a0h_adjoint_cost_v01/dev_check/unit_2031_52_s0.json` per_candidate |
| 原始 gate H=4 | PASS（1.36e-6 / 容差 1.84e-6） | 同上 |
| 原始 gate H=8 | FAIL（NUMERICAL_LIMIT，与冻结 C|8 同签名，不改判） | 同上 + A0H-1 fp32 探针 |
| 共享性 w_0 候选无关 | PASS | w_0 构造不读候选 x_k（代码审查 + 构造路径） |
| 决策保持（dev，BF16 配置） | PASS | D1 全 H + 降级批 dev-validation 9/9 decision_match |
| Branch A 独立验收 | NOT_COVERED（A 批降级） | `contract/acceptance_disposition_readonce.md` |
| Branch B1（FP32-HVP 同配置 fwd vs 伴随，atol32=1e-5/rtol32=1e-4） | PASS（3 单元 × 3 H，max 4.1e-10） | `diagnostics/branch_B1B2_summary.json`、`diag_*_fp32hvp_s0.json` |
| Branch B2（vs 冻结 BF16 C 决策级） | PASS（5 键排序 9/9 一致；逐元两式并报） | 同上 |
| Branch B3（不追认 BF16 等价） | PASS（遵守） | 原始 FAIL 未改 |
| §7 计时（A0H-2 小规模） | NOT_COVERED（触发条件未满足；非预算原因） | `contract/gpu_budget_log.txt`（822s/1800s） |
| 冻结源哈希 18 文件 | PASS 未变 | `contract/source_hashes_start.txt` 逐项复核 |
| A0H-1 审计包 SHA256 | PASS（3b237946…db6c93 一致） | `outputs/a0h_adjoint_cost_v01/a0h1_audit_package.tar.gz.sha256` |

## 1b. 第二轮追加（2026-09-15，DEV_TIMING 修正案）

| 项 | 状态 | 证据 |
|---|---|---|
| 生产入口（三模式互斥 thin adapter） | 就绪且自校验通过 | `research/scripts/a0h15_production_entry_v01.py`；`production/selfcheck_d1_h4_m4.json` |
| 入口数值等价（FP32 fwd/adj vs 冻结 pass） | PASS（同进程 gap=0，逐候选位级相等） | 同上 selfcheck |
| 同配置 B1 gate（入口内 fwd vs adj） | PASS（max 1.2e-10 ≤ 1e-5） | 同上 |
| 配置错配预读失败（排练） | PASS（dtype/hidden 漂移均 AssertionError，冻结 cfg 哈希不变） | eligibility JSON |
| ENGINEERING_ELIGIBLE | **PASS**（仅工程资格，无验收/效用身份） | `production/engineering_eligibility_d1_h4_m4.json` |
| H4/M4 同场计时（四模式，1 预热+3 交错） | 完成，DEV_ENGINEERING_TIMING | `production/timing_d1_h4_m4.json` |
| H{1,4,8}×M{1,2,4} 矩阵扩展 | 完成（7 组合） | `production/timing_d1_matrix_ext.json` |
| FP64 簿记闭合（6 运行） | CLOSED（max 残差 1.9e-16；不证明伴随正确） | `bookkeeping/closure_*.json` |
| A′ 独立验收（2030/52、2033/64、2037/82） | **FAIL（原始记录保留）**：L1 B1 gate 3/3 PASS、排序 9/9 完全一致；L2 ĥ-argmax 行动 realized V_score gain 3/3 为负 → 按冻结判据 FAIL。**历史解释更正（2026-09-15 决策语义审计，Branch A 确认）**：失败根因是 A′ 驱动 selector 方向错误——raw ĥ argmax（j=1..4、no-op 不入集）代替冻结 A0G 的 argmin predicted ΔL（j=0..4、no-op 在内）；预注册文本 "argmax-ĥ" 条款本身写错，实现忠实执行了错误文本。冻结规则离线重评分 3/3 选中 replay 最优动作、gain 全正（+8.91e-4/+3.58e-4/+2.35e-4）。按冻结重试规则 2，已执行 A′ 降级 dev-investigation，**不能作为预定算法的有效验收（无论 PASS/FAIL）**，本轮无 A″ | `accept_aprime/A_prime_verdicts.json`、`decision_semantics_audit/`（审计包，含 five_action 总表 36/36 恒等式、selector 单测 7/7） |
| 第二轮 GPU 预算 | 1270s/1800s | `contract_round2/gpu_budget_log_round2.txt` |
| 决策语义审计（第三轮，离线无 GPU） | **Branch A 确认**；驱动已修复（frozen_decision=冻结 select_action，预修复哈希存档）；唯下一步=用户预注册新验收批后用修复驱动重跑同一两层验收 | `decision_semantics_audit/A0H_ROUND2_DECISION_SEMANTICS_AUDIT.md` |

## 2. 运行方式（复现/接续）

- 环境：conda env `fr_minimind`，`PYTHONPATH=research/src`，物理 GPU1 经
  `CUDA_VISIBLE_DEVICES=1`；torch 2.11.0；外部作业不可杀。
- 诊断 pass（BF16 全探针）：
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=research/src python research/scripts/a0h15_vjp_diagnostics_v01.py --device cuda:0 --states 2031/52`
- lean（acceptance 模式，跳过 CHK 探针）：加 `--lean`。
- Branch B（FP32-HVP 配置）：加 `--hvp-fp32`（7 个 `_hvp` 调用点换 nullcontext）。
- accept runner：`research/scripts/a0h15_accept_v01.py --device cuda:0`
  （注意：其 ACCEPT_STATES 现为**已降级**的 dev 单元，仅产 DEV_VALIDATION_ONLY）。
- CPU 测试：`python -m unittest research.tests.test_a0h15_vjp -v`。

## 3. A0H-2 建议入口（下一轮，需用户确认新验收批 A′）

1. 先验冻结一个新验收批 A′（≠ D、≠ 已降级批；协议修正案），再读一次。
2. 在 A′ 上跑 accept runner（--lean）验证 Branch A/B 决策 gate；
   通过 ⇒ 触发 §7 计时（矩阵 1 单元 × H{1,4,8} × M{1,2,4}，预热 1 + ≥3 交错重复；
   独占 gate util<10%，否则标 shared 不报加速结论）。
3. 生产伴随入口 = 公共基轨迹 + 终点 V_score 梯度 + 单次 backward + 候选梯度 +
   点积（`hhat_fwd_C` 与 `vjp_shared` 已由同一 pass 产出，计时脚本可直接复用
   `a0h15_vjp_diagnostics_v01._run_diagnostics_pass(..., full_diagnostics=False)`）。
4. 计时上限仍按本轮剩余预算或新预算另批。

## 4. 目录索引

- 诊断输出：`outputs/a0h15_vjp_shared_v01/diagnostics/`（diag_*_s0.json、
  diag_*_fp32hvp_s0.json、branch_B1B2_summary.json、summary.json、run_*.log）
- accept（dev-validation）：`outputs/a0h15_vjp_shared_v01/accept/*.json`；
  v9 无效输出归档 `accept_v9_invalid_archived/`
- 合同与合规：`outputs/a0h15_vjp_shared_v01/contract/`
  （gpu_budget_log.txt、acceptance_disposition_readonce.md、
  source_hashes_start.txt、protocol_hashes.txt）
- 新代码：`research/scripts/a0h15_vjp_diagnostics_v01.py`（主 pass）、
  `research/scripts/a0h15_accept_v01.py`、`research/scripts/a0h15_analyze_v01.py`、
  `research/tests/test_a0h15_vjp.py`
- 备注：`research/a0h15/` 为空目录（原计划的代码位置；实际按仓库惯例落在
  research/scripts 与 research/tests，特此如实记录）。
- A0H-1.5 审计包：`outputs/a0h15_vjp_shared_v01/a0h15_audit_package.tar.gz`
  （manifest 不含自身；tar SHA256 单独记录于同名 `.sha256` 文件；
  已验证 listable/unpackable；无权重/梯度/检查点）。

## 5. A0H-2.0 修正验收 pilot（A″，2026-09-16）—— 完成

按 §3 入口第 1 条执行：新批次在 GPU 暴露前完成暴露台账
（`outputs/a0h_corrected_accept_v01/exposure_ledger.json`，56 态排除 9 态 → 47
合格，轮转选 6 态×2 流=12 单元、6 seed）+ 冻结合同
（`contract/contract.json`，修正语义：no-op=继续训练、argmin predicted ΔL、
撤销强制非 no-op 条款，旧原文保留）+ acceptance_manifest（含暴露前冻结的
dev-fixed 动作 j=1）。dev 全链演练（2031/52 s0，双标签 replay vs 冻结 entry
replay 位级一致 gap=0）→ P1 12 单元三 scorer → P2 同配置 FP32 gate PASS
12/12 → P3 五动作双标签 replay（score 先锁定，eval 分文件哈希后写）→ P4 分析。

**结果（五字段分离，无合成 PASS）**：COMPLETION_STATUS=COMPLETE；
PROTOCOL_VALID=PASS；NUMERICAL_GATE=PASS；DECISION_PRESERVATION=同配置与
跨配置 12/12 五动作一致；UTILITY_PILOT_RESULT=**POSITIVE_POINT_ESTIMATE**。
主终点 FP32 伴随所选动作 V_eval Gain **+1.467e-4**（bootstrap 95% PI
[+8.66e-5, +2.08e-4]，探索性非 p 值），per-seed 5/6 正；harmful 1/12、
no-op 1/12（合法 Gain=0）；与 no-op/随机期望配对差异区间不跨 0。解释红线：
符号标签 ≠ 显著优于所有方法；不作 λ/非线性/代理失效归因；不改判旧 A′。

- 制品：`outputs/a0h_corrected_accept_v01/`（审计包
  `a0h20_corrected_accept_audit.tar.gz` 46844B SHA256 `41fe8632…1ff7`）
- 新代码：`research/scripts/a0h20_{contract,manifest,corrected_accept,analyze,audit_pack}_v01.py`、
  `research/tests/test_a0h20_acceptance_v01.py`（12/12）
- GPU：991.1s/1800s；本轮停止，任何扩展须另行预注册。

## 6. 跨会话最短恢复摘要（2026-09-16 文档轮，交接冻结）

> 新上下文**只读四个文件**即可恢复 A0H 线全貌：
> `docs/FR_PROJECT_STATUS.md`（§17–§26）+ `docs/FR_EXPERIMENT_INDEX.md`
> （A0H 各行）+ 本文档 + `docs/FR_ALGORITHM_ROADMAP_POST_A0H20.md`
> （A–H 节，公式/成本模型/路线）。以下 8 点是最小恢复集。

1. **科学链**：A0G（完整切线 C 的决策效用成立 1.54e-4，shared-tangent
   墙钟成本否定）→ A0H-1/1.5（前向切线的共享反向伴随：数学等价、
   w_0 候选无关、H=1/4 原始 gate PASS、H=8 被 BF16 HVP 地板阻塞不改判、
   H4/M4 伴随 vs replay 1.17×）→ A0H20（修复 selector 后的一次预注册
   独立效用 pilot）。
2. **selector 修复历史**：旧 A′ 3/3 FAIL 的驱动用了 raw ĥ argmax
   （预注册文本条款即错）；决策语义审计 Branch A 确认冻结语义 =
   ĥ=dL_score/dλ、argmin predicted ΔL（j=0..4 含 no-op，tie 最小 ID）。
   冻结规则离线重评分 3/3 选中最优且 gain 全正；原 A′ FAIL 保留、整批
   降级 dev-investigation，不得改判为 PASS。
3. **A0H20 最终数字**：12 单元（6 seed×step46×2 流）gate PASS 12/12、
   决策保持 12/12 双向、五动作双标签 12×5 完整；主终点 V_eval mean Gain
   **+1.467e-4**（bootstrap PI 探索性）、per-seed 5/6 正、harmful 1/12、
   no-op 1/12、mean Regret 8.78e-5、oracle 命中 4/12；
   UTILITY_PILOT_RESULT=POSITIVE_POINT_ESTIMATE（符号标签，非 p 值）。
4. **为何 adjoint 不再是 correctness 主问题**：同配置 FP32 前向 vs
   共享伴随 gate PASS 12/12（atol32=1e-5/rtol32=1e-4），跨精度
   （BF16 C vs FP32 伴随）五动作决策保持 12/12；正确性链已闭合
   （路线图 B 节）。剩余 gap 在效用与成本，不在伴随数值。
5. **为何下一步先研究 candidate scoring 边际成本**：共享伴随只省
   传播部分，每候选仍需完整 g_j 反向（成本模型 T_adj(M)=a+bM 的 b
   未分解）；M=4 的 1.17× 单点不能外推（S′(M) 符号取决于 da−bc）；
   凸组合/重复候选不增加真实自由度（h_α=Σα_k h_k≥min_k h_k），
   扩大 M 前必须先测 b 与定义新候选。
6. **directional-derivative 公式**（UNVERIFIED COMPUTATIONAL
   OPPORTUNITY）：h_j = wᵀ(g_j−g) = D_θ L_j[w] − wᵀg；共享池时
   h_j = (α_j−α_0)ᵀ d，d_i = D_θ ℓ_i[w]。须经 score equivalence /
   decision preservation / wall-clock / peak memory 四验证（路线图 F 节）。
7. **为何 convex-combination 不能冒充真实 M 扩展**：凸包内点得分
   不差于顶点最优但也不更优；真实扩展须预冻结新数据子组/细分组/
   minibatch 候选；重复/凸组合只能做吞吐测试（路线图 E 节）。
8. **A0I-A/B/J/K 顺序**：A0I-A（existing-label offline adaptivity
   diagnostics，纯离线）→ A0I-B（directional scoring 在现有 dev 的
   可行性，M=4/H=4）→ A0J（预定义真实候选扩展 M=4→8，条件性 16/32）
   → A0K（repeated/closed-loop，equal-step 与 equal-wall-clock 对比）。
   当前优先级 A0I-A → A0I-B；不是直接 M=32、不是改 AdamW/λ、不是
   hybrid selector。任何一步都须另行预注册。

算法瓶颈转移全文见 `docs/FR_PROJECT_STATUS.md` §26；诊断分解
（oracle 2.345e-4 / 62.6% / regret 8.78e-5 ≈ 2.30e-5+6.48e-5）与
margin 定理边界见路线图 C/G 节。

---

## 附录：A0I（2026-09-16，post-A0H20 算法工程轮）

A0I 完成，制品 `outputs/a0i_directional_scoring_v01/`（审计包
`a0i_audit.tar.gz` 47582B SHA256 `2dca15c4…9775`）。要点：

1. **方向导数评分（h_j = D_θL_j[w_0] − w_0ᵀg，w_0 固定）数值等价
   成立**：math-SDPA backend 下 gate 54/72、排序/chosen-id 18/18 一致；
   但 **ambient flash SDPA 无 forward-mode AD（UNSUPPORTED_IN_CURRENT
   _BACKEND）**，生产路径想用 JVP 须先解决 backend 缺口。
2. **子阶段更快、总成本 SIMILAR**：scoring B/A≈0.745、显存 766→634MB；
   含 selected-gradient recompute 后总决策成本 A 1.348–1.624s vs
   B 1.296–1.475s。共享前缀（~1s）主导。每候选边际 b：B 低 A ~30%。
3. **METHOD C = NOT_APPLICABLE**（候选池不相交，结构性）。
4. A0I-A 离线诊断：controller 有真实适应性但种子异质（+5.76e-5 均值；
   2033 +3.15e-4 vs 2031 −1.80e-4）；regret 分解 8.7808e-5 =
   2.3002e-5 + 6.4805e-5 闭合同格。
5. 运行环境同前（conda fr_minimind、CUDA_VISIBLE_DEVICES=1、torch
   2.11.0、共享 GPU）。GPU 记账 ≈749s/1800s，2 次脚本重试，无环境变更。

恢复上下文先读本附录 + `outputs/a0i_directional_scoring_v01/REPORT.md`。
A0H20 冻结结论与本 handoff 前文全部原样保留。

---

## 附录：Post-A0I compact 恢复摘要（2026-09-16，纯文档轮）

**compact 后最先读取的 4 个文件**：
1. `docs/FR_PROJECT_STATUS.md`（§23 A0H-2.0 / §26 瓶颈转移 / §27 A0I /
   §28 Post-A0I 定位与修正登记）
2. `docs/FR_EXPERIMENT_INDEX.md`（A0H-2.0 与 A0I 两行）
3. 本文件（`docs/FR_A0H15_HANDOFF.md`）
4. `docs/FR_ALGORITHM_ROADMAP_POST_A0I.md`（**新**：当前有效路线图；
   旧 `FR_ALGORITHM_ROADMAP_POST_A0H20.md` 保持历史不覆盖）

**十行恢复**：

1. **A0G task utility（冻结）**：完整切线响应 C 在独立评价数据上的
   训练决策效用得到支持（Level-3 H=4 mean Gain C 1.54e-4 > B
   1.31e-4）；墙钟成本优势不成立（0.39×，无 break-even）。
2. **A0H adjoint（冻结）**：候选共享 VJP 伴随 w_0 候选无关成立；
   fp32-HVP 候选精确（max 4.1e-10）；bf16 HVP 噪声地板使逐元 gate
   在 H≥4 不可满足（NUMERICAL_LIMIT，非实现错误）。
3. **selector 语义修复（冻结）**：A′ 驱动曾用 raw ĥ argmax（方向错），
   冻结语义 = hhat=dL/dλ（Case 1）+ argmin predicted ΔL（j=0..M
   含 no-op）+ 精确并列取最小 id。驱动已修复，A′ FAIL 保留不改判。
4. **A0H20 独立效用（冻结）**：修复语义后的预注册新批次 12 单元，
   gate 12/12、决策保持 12/12、主终点 Gain_eval **+1.467e-4**
   （5/6 seed 正、harmful 1/12、no-op 1/12 合法）——
   POSITIVE_POINT_ESTIMATE。
5. **A0I directional 结果（冻结）**：`h_j = D_θL_j[w_t] − w_tᵀg_t`
   恒等式在 math backend 下 SUPPORTED；numerical gate **54/72** 与
   decision preservation **18/18** 是两个独立指标；same-kernel max
   err 2.29e-5 / cross-kernel 2.83e-5。
6. **为什么不再死磕 per-candidate JVP**：scoring 子阶段墙钟 -25%、
   显存 -17%、边际成本 -30%，但 selected-gradient recompute 必须
   计入，总决策成本只改善数个百分点（SIMILAR）；candidate scoring
   不再是唯一主导瓶颈。
7. **shared prefix 为什么成为主问题**：每决策 ~1s 的 rollout +
   endpoint grad + 反向伴随（A0I 三态中位 0.96–1.08s）占总成本
   ~70%；总决策成本仍 26–31 个等效 native steps；equal-wall-clock
   完全未验证。
8. **cached-p + fresh-G 公式（UNVERIFIED）**：`w_t = G_tᵀp_t`，
   近似 `w̃_t = G_tᵀp_tau`，误差 = `G_tᵀ(p_t − p_tau)`（stale-w 另
   加 `(G_t−G_tau)ᵀp_tau` 项）。缓存未来敏感度 p_tau，当前 G_t 按
   当前 optimizer 状态精确刷新。可能因误差抵消而更差，未验证。
9. **K_r / K_s 区分**：K_r = 完整 future-sensitivity 刷新间隔；
   K_s = 候选重评分/干预间隔。overhead = [T_prefix/K_r +
   (T_score+T_selected_grad+T_GTp)/K_s]/T_native_step。两者是不同
   算法选择，不得把整个 decision cost 只除以 K_r 却每步重评分。
10. **下一步**：A0J-P0 bounded shared-prefix profiler（归因优先），
    然后依据 profiler 只开一个主分支（cached-p / FD-HVP / 工程复用）。
    六项一级算法健康指标见新 roadmap I 节；全部下一步 NOT_STARTED。

**A0I 解释修正（数字不变）**：① 动作频繁切换 ≠ 净收益已证
（ADAPTIVE_CHOICES_OBSERVED / ADVANTAGE_OVER_FIXED_UNCERTAIN）；
② regret 第一项可为负，不是"评分器精度损失上界"；③ numerical
gate 与 decision preservation 分开（54/72 vs 18/18），不声称所有
gate failure 完全由 backend 差异导致；④ 校准 slope 0.971 不单独
称无偏，选中级 0.605 受选择条件影响；⑤ METHOD C 不写"永久不适用"，
vector-JVP/packing 仍是工程选项。

## 附录：A0J（2026-09-16，post-A0I controller 结构修改轮）

最短恢复摘要：A0J = Future-Sensitivity Reuse Study，已完成，判定
**TIME_SCALE_NEGATIVE_RESULT（§18 CASE B，含 C 元素）**。问题：缓存
p_tau（age 1/2/4）+ fresh G_t（w̃=G_t^T p_tau）vs 完整 fresh-p。
Implementation gate PASS 30/30（曾 FAIL：模块 `_clip` tick 漏
`pe._step_once`，探针链定位、修复后逐位 0.0——自检一致 ≠ 生产复现，
§9 生产复现行必需）。Preservation：CACHE_P 5/6→4/6→1/6，
STALE_W 5/6→4/6→0/6；ΔGain_cache−exact 全负；根因=未来敏感度在
1–4 native-step 窗口内 decorrelate 远快于成本结构所需（margin
median 3.8e-5 vs 误差 ~2e-3）。cosine 高（0.64–0.96）与翻转共存。
成本节省真实（0.91s vs 5.62s，CASE D 排除）。H6=NOT_TESTED。
制品 `outputs/a0j_cached_future_sensitivity_v01/`，审计包
`a0j_audit.tar.gz` 9067189B SHA256 `21c1a2f9…3e5559a`；
dashboard `ALGORITHM_HEALTH_DASHBOARD.json`；全文 `REPORT.md`。
下一步须新预注册轮；不自动启动。
