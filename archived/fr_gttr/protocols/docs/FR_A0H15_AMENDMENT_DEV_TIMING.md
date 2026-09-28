# A0H-1.5 协议修正案 —— DEV_ENGINEERING_TIMING 与 A′（2026-09-15，第二轮）

> 本轮授权（用户指示，2026-09-15）的工作顺序修订：
> **现有 dev 核账与生产入口验证 → 开发集工程计时 → 条件满足后一次 A′ 独立验收**。
> 本修正案在打开任何新结果之前登记；不改变原协议冻结条款，只新增授权与判据。
> 原 §7（A0H-2 小规模计时）的 NOT_COVERED 状态保持不变；本修正案不授予任何
> 独立效用 PASS。

## 0. 新增授权与身份

- **DEV_ENGINEERING_TIMING**：在开发集（D 与已降级 A 批）上进行的工程计时，
  **不依赖 A′ 通过**、不改变旧 §7 未执行状态、不授予独立决策效用结论。
  全部此类结果标注 `DEV_ENGINEERING_TIMING`。
- **主候选**：FP32-HVP 共享伴随，身份 = "不同响应数值配置候选"，**不是**冻结
  BF16 C 的严格等价实现。A0H-1.5 Branch B 的 B1/B2 PASS 是其 dev 级证据。
- 本轮不改 AdamW、lr、β、ε、clip、候选动作或损失定义；不做 low-rank/sparse-HVP、
  A0I、v06、新训练 seeds、M>4 或长训练。
- 只读集合（与上轮相同）：A0G 四份原科学 CSV、原 runner、λ/Rule B、split、
  原 A0H gate 与既有结果。新代码/输出用独立 run_id 与新子目录。

## 1. 指纹基线（执行前保存）

- 冻结源哈希基线：`contract_round2/source_hashes_round2_start.txt`（18 文件）。
- 轮次 GPU 预算：**新 1800s 上限**（本轮所有 GPU 作业：定向采集、生产检查、
  计时、可能的 A′），日志 `contract_round2/gpu_budget_log_round2.txt`。
- 历史 A0H-1.5 状态保持原状：独立验收 NOT_COVERED、计时未执行、A 批已降级。

## 2. 新增交付物与判据（摘要，细节见各阶段交付）

1. **comparison_ledger.csv/json**：所有数值比较对的显式台账（方法对、单元、H、
   容差、逐项判定）。"位级一致"必须写明比较两端与覆盖单元。
2. **端点/逐步记账闭合**：FP64 点积累计，恒等式
   `SF−SA = EF−EA+R0+ΣRk`；只证记账，不证伴随正确。未定位残差标 UNRESOLVED。
3. **独立生产入口**（thin adapter，4 互斥模式）：forward_only(bf16|fp32)、
   adint_only(fp32)、replay_only(native)。调用计数与共享性随 M 检查。
4. **ENGINEERING_ELIGIBLE**：同配置 FP32 数值 + 模型重建核对 + 完整 dev 演练
   通过后记 PASS，自动进入计时，不问。
5. **H4/M4 同场四模式计时**：FP32 fwd、FP32 adjoint、BF16 fwd、native replay；
   比值 R_structure、R_replay、R_frozen 只用同场配对时间。
6. **A′**（一次，≤3 单元，预登记，与 D/降级 A/本轮开发单元全不同）；两轮验收
   （FP32 同配置数值 + 相对冻结 BF16 C 的动作差异与实际效用）。

## 3. 故障与重试规则（打开 A′ 前冻结）

- 结果未产生或暴露前的、预先列明的基础设施故障（设备 OOM、manifest 读错、
  序列化失败、入口崩溃）：允许**一次**等价重试并记录。
- 若据已暴露的评分/误差/决策结果修改代码或规则：按预定范围将涉及单元降为
  开发数据，本轮不再自动寻找 A″。
- 已降级旧 A 批不得恢复独立身份。

## 执行记录（2026-09-15 收尾）

 Amendment 全部执行完毕：§二 台账与更正、§三 簿记闭合（6/6）、§四 生产入口
（ENGINEERING_ELIGIBLE=PASS，含两个正确性 bug 的修复与位级等价验证）、§五
H4/M4 同场计时（R_structure=2.577 / R_replay=1.170 / R_frozen=2.750）+ 7 组合
矩阵、§六 A′ 一次验收（3 单元，预注册先冻结；verdict 3/3 FAIL——L1 全 PASS、
排序 9/9 一致，L2 ĥ-argmax realized gain 为负；诚实报告）。GPU 累计 ~1270s/1800s。
A′ 的 DEV_ENGINEERING_TIMING 授权不改变原 §7、不授予效用 PASS（本修正案范围
声明保持不变）。
