# A0H-2.0 修正验收 pilot（A″）结论报告

> run_id：`A0H20_CORRECTED_ACCEPTANCE_V01`　日期：2026-09-16
> 合同：`contract/contract.json`（暴露前冻结，SHA256 `b692ede3…46b46`）
> 预注册：`acceptance_manifest.json` + `exposure_ledger.json`（GPU 暴露前冻结）
> 原则：五个状态字段分离输出，**不存在合成 PASS**；PASS 不是本轮目标。

## 0. 批次身份

- 6 状态 × 每态 2 条预登记未来流 = **12 单元**；6 个训练 seed（2030–2035），状态
  全部为 v0.7 bank step 46（暴露台账轮转选取，仅按身份/可用性/暴露记录，
  从未按 hhat/gain/损失挑选）。
- 决策 horizon H=4；λ=0.05（Rule B 全局冻结）；5 动作（j=0 no-op 继续训练 +
  j=1..4 单步注入）；split 为冻结 A0G validation_split。
- dev 演练单元：2031/52 s0（D 态，dev-only，不入验收分析）。

## 1. 状态字段（分离）

| 字段 | 值 |
|---|---|
| COMPLETION_STATUS | **COMPLETE**（12/12 预测 + 12/12 分数标签 + 12/12 eval 标签） |
| PROTOCOL_VALID | **PASS**（单元齐备；选择可由预测重算；五动作标签完整；eval 锁定哈希与分数文件一致；全部报告值 finite；gate 阈值可解析） |
| NUMERICAL_GATE | **PASS**（12 单元 × 4 动作，同配置 FP32 前向 vs 共享伴随，冻结 atol32=1e-5/rtol32=1e-4） |
| DECISION_PRESERVATION | 同配置 12/12 排序/选择/no-op 位一致；跨配置（BF16 C vs FP32 伴随）12/12 一致 |
| UTILITY_PILOT_RESULT | **POSITIVE_POINT_ESTIMATE** |

## 2. 主终点（预注册）

FP32 伴随所选动作的 **V_eval Gain**（= v_eval(no-op) − v_eval(chosen)）：

- **均值 +1.467e-4**（态内两流平均 → 6 态等权；全部 12 单元含 no-op/有害如实计入）
- 分层配对 bootstrap（seed→state→stream，10000×，rng 260916，95% 百分位）：
  **[+8.66e-5, +2.08e-4]** —— 探索性不确定性区间，**不是 p 值**；流数 ≠ 独立 seed 数。
- per-seed：2030 **+1.99e-4**、2031 **+9.12e-5**、2032 **−1.84e-5**（唯一负 seed）、
  2033 **+9.75e-5**、2034 **+2.54e-4**、2035 **+2.56e-4**（6 seed 中 5 正 1 负）。

## 3. 配对差异（fp32adj − 基线，同聚合）

| 基线 | mean | 95% PI（探索性） |
|---|---|---|
| no-op | **+1.47e-4** | [+8.66e-5, +2.08e-4] |
| BF16 C argmin | 0（12/12 选择相同） | [0, 0] |
| FP32 前向 argmin | 0（12/12 选择相同） | [0, 0] |
| replay-score oracle | −2.30e-5 | [−6.30e-5, +1.77e-5] |
| dev-fixed（旧 A′ 冻结动作 j=1） | +5.76e-5 | [−5.35e-5, +1.73e-4] |
| 均匀随机期望 | +8.29e-5 | [+1.91e-5, +1.47e-4] |

- 与 no-op、随机期望的配对差异区间不跨 0；与 dev-fixed、oracle 的区间跨 0。
- oracle 差异为负符合预期（oracle 按 V_score 终点最大化选取，不读 eval）。
- **未验证的主张**：比所有方法显著更优；任何方法的统计显著性；任何部署性能结论。

## 4. 次指标

- harmful_rate（V_eval Gain<0）= **1/12 = 8.3%**（2032_046_s1，−7.33e-5，如实计入）
- no-op 比例（fp32adj）= **1/12 = 8.3%**（2033_046_s0 合法选择 no-op，Gain=0）
- Regret_eval：mean 8.78e-5，max 3.42e-4
- oracle 精确命中率 4/12；near-tie top-1 6/12
- 有限动作预测误差（λ·ĥ_sel − ΔL_score_sel）：mean −2.71e-5，|mean| 1.01e-4

## 5. 决策保持（五动作，冻结 tie-break）

- 同配置 FP32 前向 vs FP32 伴随：**12/12** 排序一致、chosen_id 一致、no-op 位一致
- 跨配置 BF16 C 前向 vs FP32 伴随：**12/12** 一致
- near-tie 余量仅诊断（见 analysis.json）；精确 tie 按冻结最小 ID 规则。

## 6. 观测性定位（仅观测，不作归因）

- corr(λ·ĥ_sel, ΔL_score_sel) = 0.635；corr(ΔL_score_sel, ΔL_eval_sel) = 0.429
- **无** 幅度/原生导数探针证据，故**不作**「λ 过大/非线性/代理失效」等任何归因；
  不据此调 λ、不重选候选、不开 A‴。

## 7. 数值 gate 明细

每单元 4 动作 |fwd−adj| 与阈值见 `gate.json`；12/12 单元通过；全部值 finite、
阈值非负。通过含义仅限「同配置实现一致」，**不**代表 native 导数绝对正确。

## 8. 执行与预算

- GPU 累计墙钟 **991.1 s / 1800 s**（dev 361.6 含演练 + batch 629.5：preload 35.2、
  P1 488.2、P3 106.1）；GPU1 共享空闲使用，未杀外部任务；无重试、无中断。
- 代码变化（暴露前/暴露后纪律）：(i) dev 预算日志补记（序列化级， retroactive，
  有 mtime 重建证明）；(ii) P1 改用 production-entry 模式在 H=4 计分（成本理由，
  round-2 selfcheck 已证同配置位级一致；暴露前决定，未读任何分数）。
  无数字/选择/标签逻辑变更；无实施变更受已读分数影响。
- 冻结源哈希：见 `acceptance_manifest.json.source_hashes`（驱动新指纹
  `5310aee7…219f` 于批次启动前归档）。

## 9. 缺失表

空 —— 计划 12 单元预测/分数标签/eval 标签全部完成，无缺失项。

## 10. 解释边界（红线）

- POSITIVE_POINT_ESTIMATE 仅为预注册主终点均值的**符号标签**。
- bootstrap 区间为探索性不确定性，不是频率学确认；流数 ≠ 独立 seed 数（6 seed）。
- 单 seed（2032）为负、单单元有害、oracle 命中率 4/12 —— 全部如实报告。
- 不声称任务效用已在所有状态证明；不声称部署加速；不重开 A′ 判定。
- 旧 A′ FAIL 记录与全部历史制品保持冻结原样。

## 11. 受证据限制的下一步建议（非行动）

本轮证据支持「在未见单元上、冻结语义下，FP32 伴随选择器获得正的点估计 V_eval
增益且决策保持完整」。任何扩展（更多 seed/状态、H=8 效用、λ 扫描、hybrid 规则）
均属**新实验**，须另行预注册；本轮不作、不建议立即执行。
