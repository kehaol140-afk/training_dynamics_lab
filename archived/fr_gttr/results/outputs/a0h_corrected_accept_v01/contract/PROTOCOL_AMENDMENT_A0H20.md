# A0H-2.0 修正验收 pilot 协议修订（人可读版；与 contract.json 同源互校验）

> 生成于 2026-09-16，**在任何 A″ 评分/损失/标签暴露之前冻结**。
> 机器可读版：`contract.json`（SHA256 `b692ede3405cf49871bbbf01c829bcaed1e92e8e21e6749d9ac2530f89546b46`）。本文件与 contract.json
> 由同一配置生成（`research/scripts/a0h20_contract_v01.py`），互为校验。

## 1. 语义冻结（mandate §二）

- action_ids = [0,1,2,3,4]；**j=0 = no-op**：不施加额外干预、继续原始训练更新
  （**不是跳过 optimizer step**）；j>0：post-unscale/pre-clip 仅 h=1 一步施加
  `g' = g + λ(g_j − g)`，λ=0.05 对所有非零动作同一正值；
- H=4，读出 θ_{t+H} post-step；
- **hhat = dL_score/dλ**（Case 1，无符号变换、无 utility 变换）；
- predicted_ΔL_j = λ·ĥ_j；predicted_gain_j = −λ·ĥ_j；ĥ_0=0、predicted_ΔL_0=0；
- **selected_id = min(action_ids, key=(predicted_ΔL_j, action_id))**，精确比较、
  tie 取最小 ID；**不新增**浮点舍入、容差并列或获益阈值；
- 公共正 λ 不改变排序——旧程序的错误是 **argmax + 遗漏 no-op**，不是排序时没乘 λ；
- 分数输出同时保存 ĥ、λ·ĥ、−λ·ĥ 并标明单位。

## 2. 撤销条款（旧原文保留）

撤销「必须 choose_noop=False 且每个单元 Gain>0」。原因：选择 no-op 是合法输出
（Gain=0），A0G 冻结语义本身在 formal 中 6/360 合法选 no-op；全部 no-op 不算
观察到正收益；有害动作如实计入。旧条款原文保留于
`accept_aprime/PREREGISTRATION.md` 与 `decision_semantics_audit/L2_CONTRACT_AUDIT.md`。

## 3. 状态字段分离输出

**COMPLETION_STATUS / PROTOCOL_VALID / NUMERICAL_GATE / DECISION_PRESERVATION /
UTILITY_PILOT_RESULT 分开输出，绝不合成单一 PASS。** 按预注册条件合理停止
不等于协议违规，也不等于所有阶段完成。

## 4. 数值 gate（沿用冻结阈值，来源可解析）

`|hhat_fwd_C − hhat_adj| ≤ atol32 + rtol32·|hhat_fwd_C|`，atol32=1e-5、rtol32=1e-4
（来源：a0h15_accept_aprime_v01.py L35 与 round-2 预注册文本实际应用于 FP32-HVP
候选的公式）。适用范围：H=4 全部 4 个非零动作；H=1/H=8 差值仅作诊断。
全部数值须 isfinite；NaN/Inf → NON_FINITE 并阻止 gate/聚合/argmin。
通过只称「同配置实现一致」，不称 native 导数绝对正确。

## 5. 方法与信息隔离

scorers：BF16 前向 C（历史配置比较）、FP32-HVP 前向（同配置参照）、
FP32-HVP 共享伴随（本轮候选）。选择策略：no-op、BF16C argmin、FP32fwd argmin、
FP32adj argmin、replay-score oracle（V_score 最小终点损失，tie 最小 ID）、
均匀随机期望（由五动作标签直接求均值）、dev-fixed（A″ 结果打开前由旧 A′
三状态五动作 V_score 标签冻结的固定动作）。
adjoint_only 不得依赖候选未来 tangent 构造 w0、不得隐藏 joint forward；
forward reference 单独运行（审计开销），保存运行模式指纹与调用计数。
**V_eval 五动作标签同次 replay 分开落盘；任何被测选择器不得读 eval；
全部分析预定选择锁定后才读 eval**（测试验证此隔离；不能保证则降级）。

## 6. 批次计划

6 状态 × 每态 2 条预登记未来流 = 12 单元；≥3 训练 seed 目标；只用已有
checkpoint。状态按 (train_seed, 真实 optimizer step, checkpoint ID) 排序后
跨 seed 轮转选取；每态按已登记 stream ID 稳定顺序取两条合法流。仅按
身份/可用性/暴露记录选取，**禁止按 hhat/gain/损失/历史效果挑选**。
3–5 态 → 开任何结果前登记缩减批次；<3 态或无 2 条合法流 → NO_ELIGIBLE_BATCH。

## 7. 分析计划

主终点：FP32 伴随所选动作 V_eval Gain（态内两流平均 → 态等权）。
配对基线：no-op、BF16 C、FP32 前向、replay-score、随机期望、dev-fixed。
次指标：harmful_rate、no-op 比例、Regret_eval、组内排名、oracle 命中率、
有限动作预测误差。bootstrap：seed→state→stream 分层配对 10000 次、
rng 260916、95% 百分位；仅 1 seed 时不报 seed 层区间；区间仅为探索性
不确定性，有利比例不是频率学 p 值。UTILITY_PILOT_RESULT 取
NOT_COVERED / INCOMPLETE / POSITIVE|ZERO|NEGATIVE_POINT_ESTIMATE。

## 8. 预算与禁止项

GPU 累计墙钟 ≤1800 s（含 dev 演练），短时等待合计 ≤10 分钟。
禁止：新决策规则、hybrid/replay-gain、改 λ、改 AdamW/lr/β/ε/clip、
新训练 seed/checkpoint/长训练、v06、low-rank、sparse-HVP、H=8 效用、
全矩阵计时重做。

## 9. 互校验

- contract.json SHA256：`b692ede3405cf49871bbbf01c829bcaed1e92e8e21e6749d9ac2530f89546b46`
- 本 MD 文件 SHA256（生成后由脚本回写 contract.sha256 记录）：见
  `contract.sha256` 与 `contract_crosscheck.json`。
