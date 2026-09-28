# L2 Contract Audit — A0H-1.5 Round-2 A′ Acceptance

**结论：本轮 Branch A（selector 实现偏离 A0G 冻结语义）。L2 数值判据本身未被改动、未被放宽或收紧。**

## 1. 预注册原文（accept_aprime/PREREGISTRATION.md，冻结于暴露前）

> - L2 actual utility:
>   - Chosen action (FP32 adjoint argmax-ĥ at H=4) realized utility via native
>     +λ replay (λ=0.05 frozen): V_score gain vs no-op branch > 0 at H=4
>     (choose_noop=False). V_eval gain for the chosen action reported as the
>     held-out secondary check (frozen a0g _run_action_branch, read-only call).
> - Unit verdict: PASS iff L1 gate PASS AND H=4 chosen-action agreement AND
>   L2 V_score gain > 0. Otherwise FAIL with numbers.

## 2. L2 准则逐项释义（按原文，不改写）

| 条款 | 含义 |
|---|---|
| "Chosen action (FP32 adjoint argmax-ĥ at H=4)" | 被测动作 = FP32 伴随 ĥ 的 **argmax**（预注册文本如此写明） |
| "realized utility via native +λ replay (λ=0.05 frozen)" | 用冻结 replay（+λ 分支）实测 V_score |
| "V_score gain vs no-op branch > 0 at H=4 (choose_noop=False)" | 所选动作 V_score gain（相对 no-op）必须为正，且 replay 不得选 no-op |
| "V_eval gain … held-out secondary check" | V_eval gain 只作次级报告，不进 verdict |
| "PASS iff L1 gate PASS AND H=4 chosen-action agreement AND L2 V_score gain > 0" | 单元判据：L1 数值 gate + 两配置动作一致 + L2 gain>0 |

**L2 原文不要求**：命中 replay 最优、平均 Gain 不下降、harmful 不新增、V_eval 为正。
**L2 原文不是**：「scorer 选择与 frozen C 相同」——它只要求 BF16 前向与 FP32 伴随
两个配置之间的动作一致（且当时写的是 argmax 一致）。

## 3. 代码实现 vs 原文对齐

预修复驱动（SHA256 `937798f6…417fe10`）逐项核对：

| 条款 | 实现（文件:行） | 一致？ |
|---|---|---|
| argmax-ĥ 选动作 | a0h15_accept_aprime_v01.py L110-111 `ranking()` 降序 + L192-193 `max(key=(score,-j))` + L203 chosen | ✅ 与**预注册文本**一致 |
| λ=0.05 replay | L201 `entry.run_replay_only`（entry L42 `FROZEN_LAMBDA=_a0h1.FROZEN_LAMBDA`） | ✅ |
| gain vs no-op > 0 且 choose_noop=False | L204 `v_gain=gain_vs_noop[chosen]`；L233 `L2_gain_positive = v_gain>0`；L234 verdict | ✅ |
| V_eval 次级 | L206-221 `_run_action_branch(sign=+1)` + `noop_val_eval`；`veval_gain=ve_noop−ve_chosen` | ✅ |
| verdict 复合判据 | L234 `PASS iff b1_pass and agree and v_gain>0` | ✅ |

**实现忠实执行了预注册文本。问题在预注册文本本身**："argmax-ĥ" 与 A0G 冻结决策
语义冲突——冻结 selector 是 `argmin_j λ·ĥ_j`（j=0..4 含 no-op，
`research/src/fr/a0g_decision_utility_v01.py:225-229`），因为 ĥ = dL/dλ，
argmax ĥ = 最大预测 ΔL = **最差动作**。已用冻结科学数据交叉验证：A0G formal
level3 中 method C 的 j_hat 在 360/360 个 (state,stream,H) 上等于
`argmin_j pred_delta_C`（含 no-op，6/360 选 no-op）。

## 4. 处置（不改判据、不改历史）

1. **L2 数值判据（chosen-action V_score gain > 0）保持不变**；不因看到 3/3 为负而
   放宽，也不为翻案而收紧。
2. 预修复 A′ 的 FAIL 记录**全部保留**（原始 verdict/replay/diag JSON 未动）。
3. 按预注册冻结重试规则第 2 条（暴露后改 scoring/gate/replay 逻辑 → 降级），
   **已执行的 A′ 整批降为 dev-investigation，本轮不再有 A″**；该 A′ 既不能再作为
   「预定算法的有效 FAIL」，也不能翻案为 PASS——它是错误 selector 下的执行。
4. 修复后的驱动（`frozen_decision()` = 冻结 `select_action`）只用未来**新预注册**
   批次。
5. 用**已保存的五动作真实标签**做离线纠正重评分（aprime_reanalysis.csv）：
   冻结规则 3/3 选中 replay 最优动作，gain 全为正（+8.91e-4 / +3.58e-4 / +2.35e-4）。
