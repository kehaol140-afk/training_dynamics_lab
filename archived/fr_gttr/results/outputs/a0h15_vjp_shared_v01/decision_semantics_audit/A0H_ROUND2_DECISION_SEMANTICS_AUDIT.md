# A0H Round-2 决策语义与五动作效用审计（2026-09-15）

**分支判定：Branch A —— A0H A′ acceptance 实现的 selector 偏离 A0G 冻结决策语义。**
不是 Branch B（实现忠实执行了预注册文本，错误在文本的 "argmax-ĥ" 条款本身，
实际所选动作确实错了）；不是 Branch C（ID/no-op/标签/序列化映射全部核验通过，
36/36 恒等式）；不足以进入 Branch D（语义纠正后 3/3 选中效用最优动作且 gain 为正）。

## 1. hhat 的真实数学语义

**ĥ_j = d L_score(θ_{t+H}) / dλ（Case 1）**，λ 为单位幅度的 post-unscale/pre-clip
梯度动作 `g_action = g + λ(g_j − g)` 的系数。源码定位：
- 动作与 tangent 约定：`research/scripts/a0g_decision_utility_v01.py` L7-15、L576-625；
- 前向 ĥ：`project_tangent(v_H, r_H)`，L540-548（注释原文 "hhat = dL(θ_{t+H})/dλ"）；
- A0H-1.5 前向 ĥ（`hhat_fwd_C`）：`research/scripts/a0h15_vjp_diagnostics_v01.py` L608-616，
  同一约定（B1 同配置 gate 已核验 fwd≈adj 至 1e-10）；
- 无符号翻转、无 utility 变换；λ 缩放只在预测端：`predicted_ΔL_j = λ·ĥ_j`（L1015-1037）。

## 2. 冻结 selector 与 A′ 实现的差异

| | 冻结 A0G | 预修复 A′ 驱动 |
|---|---|---|
| 选择规则 | `argmin_j predicted_ΔL = argmin_j λ·ĥ_j` | raw `argmax_j ĥ_j` |
| 候选集 | j=0..4（**no-op 在内**，预测 0.0） | j=1..4（**no-op 被排除在排序外**） |
| tie-break | 最小 j | 最小 j（同） |
| 源码 | `fr/a0g_decision_utility_v01.py:225-229` | `a0h15_accept_aprime_v01.py:110-111,184-203` |

因 λ>0：`argmin predicted_ΔL = argmin ĥ = argmax predicted_gain`；
而 A′ 用的 `argmax ĥ = argmax predicted_ΔL = **最差动作**`。
方向错误的后果被数据证实：A0G formal 冻结数据 360/360 个单元的 j_hat 等于
argmin pred_delta_C（含 no-op 情形 6/360）。

## 3. 五动作审计（H=4，λ=0.05，stream 0；标签 = 已保存 replay 真实值）

| 单元 | ĥ(H4) j1..j4 | 原程序选择 | 原 gain | 冻结规则选择 | 纠正 gain | replay 最优 | 最优 gain |
|---|---|---|---|---|---|---|---|
| 2030/52 | −0.01571, +0.006411, −0.001809, +0.002406 | **j2** | −2.548e-4 | **j1** | **+8.914e-4** | j1 | +8.914e-4 |
| 2033/64 | −0.002701, −0.001725, +0.005056, −0.004206 | **j3** | −7.349e-5 | **j4** | **+3.580e-4** | j4 | +3.580e-4 |
| 2037/82 | −0.002517, −0.002230, −0.000143, +0.004694 | **j4** | −2.795e-4 | **j1** | **+2.355e-4** | j1 | +2.355e-4 |

- 冻结规则 3/3 选中 replay gain-argmax（效用最优），**纠正后 gain 3/3 为正**；
- 原 3/3 negative gain 记录**保留**（verdict JSON 未动）；
- proxy 符号正确率 H=4：3/4、4/4、3/4（12 个 active 候选 9/12）；
- 恒等式审计 36/36：Gain=L_base−L_action、no-op Gain=0、oracle≥0、Regret≥0 全过；
- 顺序不变性（Test 4）与 tie-break（Test 5）单测通过。

## 4. "9/9 排名一致"的重新解释

- "四个候选" = **仅 j=1..4**，不含 j=0 no-op；
- 两 scorer 相对 zero（no-op）的位置一致：3 单元 no-op 在 raw 降序中均位于
  j2/j4 之间，且 BF16/FP32 位置逐一相同（five_action_rank_match_raw_score_order=9/9，
  见 rank_match_reanalysis.csv）；
- **五动作 chosen_id 一致**：冻结规则下 BF16 前向 vs FP32 伴随 3/3 相同；
- 冻结 C 配置（BF16 前向）与 FP32 伴随在相同 selector 下 **3/3 选择相同动作**；
- 但注意：9/9 "排序一致" 对任何单调变换都成立，**不能**用来论证规则方向正确——
  这正是本轮之前误用的点。

## 5. L2 合同审计（详见 L2_CONTRACT_AUDIT.md）

L2 原文 = 「所选动作（预注册写的是 argmax-ĥ）经 +λ replay 的 V_score gain>0」
+ 两配置动作一致 + L1 gate。实现与原文逐项一致；**错误在预注册文本的 selector
条款**，不在 gain 计算或判据松紧。L2 数值判据保持不变；按冻结重试规则 2，
已执行 A′ 降级为 dev-investigation，本轮无 A″。

## 6. 计时复核（未重跑 GPU；DEV_ENGINEERING_TIMING 标记保持）

H=4/M=4 原始 3 次交错固定序（fp32_forward→fp32_adjoint→bf16_forward→replay）：

| 方法 | runs (s) | median | mean | std |
|---|---|---|---|---|
| fp32_forward | 3.3818 / 3.4819 / 3.4029 | 3.402883 | 3.422201 | 0.0528 |
| fp32_adjoint | 1.3437 / 1.3115 / 1.3207 | 1.320674 | 1.325282 | 0.0166 |
| bf16_forward | 3.6087 / 3.6313 / 3.7714 | 3.631305 | 3.670494 | 0.0881 |
| replay | 1.5456 / 1.5042 / 1.6570 | 1.545594 | 1.568943 | 0.0790 |

- **R_structure = 2.577**，**R_replay = 1.170**，R_frozen = 2.750；
- **speedup ≈ 1.17×** 与 **time reduction ≈ 14.55%** 分别陈述，不混用；
- GPU1 计时窗采样 avg util 6.2%、max 24%（外部队列仅空上下文）→
  维持 DEV_ENGINEERING_TIMING（exclusive-in-practice），不升级为部署性能结论。

## 7. 口径（按 mandate §11）

- ✅「当前开发工程条件下，共享 FP32-HVP 伴随相对同配置前向响应观察到约 2.58×
  加速，相对 native replay 观察到约 1.17× speedup（约 14.6% 墙钟下降）。」
- ✅「FP32 forward-adjoint 同配置数值一致性已获得强支持。」
- ✅「原 A′ L2 负结果系 selector 方向错误所致，已离线纠正；A′ 降级
  dev-investigation，不能作为预定算法的有效验收（无论 PASS/FAIL）。」
- ❌ 不写「正式部署加速已证明」「scorer 已在所有状态证明任务效用」「A′ 已证明
  方法失败」。

## 8. 处置与唯一下一步

1. 驱动已修复（恢复冻结 `select_action`；预修复哈希存档
   `accept_aprime/driver_sha256_before_semantics_fix.txt`）；新增单测
   `research/tests/test_a0h15_selector_semantics_v01.py` **7/7 通过**。
2. A′ 三单元按冻结规则离线纠正重评分：3/3 会 L2 PASS（增益见上表）——
   仅作 dev-investigation 证据，**不追认为 unseen confirmation**。
3. **唯一下一步**：用户先验冻结一个新验收批 A″′（≠D、≠降级批 A、≠A′ 三单元、
   ≠本轮 dev 单元），用修复后驱动跑同一两层验收；冻结语义、λ、容差、
   runner 一律不动。
