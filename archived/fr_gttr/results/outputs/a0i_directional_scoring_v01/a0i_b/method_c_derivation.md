# A0I-B METHOD C (SHARED_POOL_DIRECTIONAL) 结构推导与适用性判定

> 2026-09-16，A0I-B。本文件只含离线推导与数值核验，不含新实验。
> 结论：**METHOD C = NOT_APPLICABLE**（结构性，非数值性）。

## 1. 候选构造回顾（冻结，来自 `a0g_decision_utility_v01.candidate_batches_for`）

- 数据按 `sha256(sample_id|content) mod 4` 分入 4 个 hash 组（`N_DATA_GROUPS=4`），
  组大小 1038 / 973 / 987 / 938，**两两不相交**（并集 3936 = 总样本数）。
- 候选 j（j=1..4）对应组 (j-1)：用 Philox 局部 Generator
  `key = stream_seed XOR ((9000+group_id)<<32)` 从组内无放回抽取
  `n_micro(2) x batch_size(4) = 8` 个样本，切成 2 个 micro-batch。
- 损失：`L_j = sum_micro (CE_micro + aux)/2`，`CE_micro` 为该 micro-batch
  有效 token（label != -100）上的均值交叉熵。模型 `use_moe=False` ⇒
  `aux_loss = 0`。参数 fp32，前向 bf16 autocast。

## 2. alpha 分解（精确）

micro-batch 的有效 token 数记 `V(micro)`，样本 i 的有效 token 数记
`v_i`。每个 micro-batch 的 CE 可写为

    CE_micro = sum_{i in micro} (v_i / V(micro)) * ell_i,

其中 `ell_i` 为样本 i 的 token-mean 交叉熵（对 θ 可微，与 θ 无关的
归一化常数提出）。于是

    L_j = sum_{i in B_j} alpha_ji * ell_i,
    alpha_ji = v_i / (2 * V(micro(i))).

`alpha_ji` 只依赖数据（token 计数），与 θ 无关 ⇒

    g_j = nabla_theta L_j = sum_{i in B_j} alpha_ji * nabla_theta ell_i   (精确)
    h_j = w^T (g_j - g) = sum_{i in B_j} alpha_ji * d_i - w^T g,
    d_i = D_theta ell_i[w]  （逐样本标量方向导数）

**代数恒等式成立**（纯链式法则，无近似）。

## 3. 数值核验（离线，2026-09-16）

对全部 3 个 dev 状态的 stream 0 实际重构候选 batch：

| state | 每候选样本数 | 候选间样本重叠 | 候选内重复 | 样本是否全部来自本组 |
|---|---|---|---|---|
| 2031/52 | 8 | 0（全部 6 对） | 无 | 是 |
| 2034/64 | 8 | 0（全部 6 对） | 无 | 是 |
| 2036/76 | 8 | 0（全部 6 对） | 无 | 是 |

组级核验：`outputs/fr_decision_utility_v01/data_group_definition.json`
4 组两两交集为空（union=3936=size 之和）。

## 4. 为何 NOT_APPLICABLE

METHOD C 的效率前提是**多个候选共享同一基础池**：池上的一次方向导数
pass 服务所有候选，每候选只剩一个稀疏点积，从而把"每候选边际成本"
压成常数。

本构造中 `B_1..B_4` 来自**互不相交**的 hash 组（上表，重叠恒为 0）：

- 并集池 `U = B_1 ∪ B_2 ∪ B_3 ∪ B_4`（32 样本）是**不交并**；
- 每个 `d_i` 只被恰好一个候选使用 ⇒ 全部逐样本方向导数工作量
  = 32 次逐样本 JVP，**不低于** METHOD B 的 8 次 micro-batch JVP
  （总样本数相同，逐样本粒度反而更差）；
- 稀疏点积 `(alpha_j - alpha_0)^T d` 的节省不存在（没有可复用项）。

因此即使代数恒等式精确成立，METHOD C 在当前候选结构下**不能降低
candidate marginal cost**，判定 **NOT_APPLICABLE**（结构性）。

进一步：逐样本 `d_i` 需 vmap/逐样本 forward-mode；而本轮 probe 已确认
flash SDPA 在当前 torch 2.11 下无 forward-mode AD（NotImplementedError，
见 `a0i_b/probe_2031_52.json`），逐样本 JVP 亦需 math backend，无额外收益。

## 5. 对 A0J 的含义

真实 M 扩展（4→8）必须引入**新预冻结数据子组**候选（路线图 E 节）；
若新候选同样按不相交子组构造，METHOD C 依然不适用；若未来候选定义为
同一池上的不同加权组合（mixture 型 action space），METHOD C 才有
复核价值。本轮不为此改候选科学定义。
