# FR 实验框架 v0.6 中文版

## 目标

v0.6 不再增加新机制，而是验证 v0.5 的核心现象能否跨训练 seed 复现。

主问题：

1. H=1 的 B_theta -> chi_theta 映射是否在多个独立训练 seed 中都出现？
2. H=8 的“幅值映射失败”是否可复现？
3. H=32 的不稳定性是否可复现？
4. 控制训练进度 step 后，B_theta 是否仍然提供额外的 chi_theta 预测信息？

## 独立训练 seed

默认：

- 2026
- 2027
- 2028
- 2029

每个 seed 都重新：
- 初始化模型；
- 生成训练 batch trajectory；
- 训练并保存 7 个 state；
- 构造该 state 自己的 q_t；
- E1 估计 B_theta；
- E2 估计 chi_theta。

不同训练 seed 的 future stream seed 也不同。

## 每 seed 的 state

`46, 52, 58, 64, 70, 76, 82`

每 seed 7 个 state，共 28 个 state points。

注意：
- 28 个 state 不能简单当成 28 个完全独立实验；
- 真正独立的科学重复单位是 4 个 training seed。

## 统计认证

每个 H 分别输出：

### A. 每 seed 独立映射
- Pearson
- Spearman
- R2
- K
- sign agreement

### B. pooled raw mapping
仅作描述。

### C. within-seed 去训练趋势
在每个 seed 内分别回归：
- B ~ state_fraction
- chi ~ state_fraction

然后计算 residual_B 与 residual_chi 的相关。

这回答：
“B 与 chi 的相关是否只是二者共同随训练推进变化？”

### D. 一阶差分
同 seed 相邻 state：

- Delta B
- Delta chi

再比较 Delta B 与 Delta chi。

这进一步消除慢变化的训练趋势。

### E. Leave-One-Seed-Out
用 3 个 seed 训练，预测第 4 个 seed。

比较：

step-only:
    chi = a + c*step

和：

step+B:
    chi = a + c*step + K*B

如果加入 B 后，未见 seed 的 MAE 稳定下降，
才是更强的“B 提供了训练进度之外的信息”的证据。

## 当前不做

- module directions
- AdamW memory 扩展变量
- controller
- V2A/V2B
- mean/tail 主检验

这些都等跨 seed 认证以后再决定。
