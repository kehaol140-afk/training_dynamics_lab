# G3_PROTOCOL — FR_G3_CLIPPING_INTERVENTION_V01

冻结时间：2026-09-16T18:00Z（任何 G3 结果打开之前）。本协议与
`research/configs/g3_clipping_intervention_v01.json` 同时冻结，阈值不得
按结果更改。

## 0. 红线

- 5M anchor / G1a 26M / G1b 100M / A0A–A0G / A0H/A″ 冻结制品一律只读。
- G1a 冻结结果直接作为 clip=1.0 对照读取，**不重跑**。
- 新工作全部进入 `outputs/generalization_g3_clipping_intervention_v01/`。
- 本轮完成 G3 后停止；不自动启动 G2。

## 1. 科学问题（唯一）

A0E2 识别的 clipping Jacobian 机制是否有可干预的因果证据。

模型阶梯（冻结定义）：B = optimizer-only tangent；BH_fixed = B + future
raw curvature（初始脉冲同 B/C 经 J_clip(g_t)）；C = BH_fixed + 未来步
clip-JVP 精确传播。clip=1.0 下 C 与 BH_fixed 存在非零增益（gain_clip =
err_BH_fixed − err_C > 0）。理论上 clipping 完全关闭时 J_clip = I，
BH_fixed 与 C 在数学定义上退化为同一响应传播，除数值实现误差外满足
BH_fixed ≈ C。

核心预测（注册）：

- clip=1.0：gain_clip > 0；
- no-clip：gain_clip ≈ 0；
- interaction：Delta_clip = mean(gain_clip_clip1) − mean(gain_clip_noclip) > 0。

本轮不要求 curvature gain 消失。

## 2. 为什么用 26M（固定，不换 scale）

hidden=512/layers=7/heads=8/kv=4/head_dim=64/intermediate=1664，vocab 6400，
seq 128（与 G1a 逐字段一致，训练时实测断言 unique params = 26,681,728）。
原因：① 26M full tangent 已验证（G1a STRONG 冻结）；② 26M clip=1.0 下
clip activation 近 100%（training 36/36、future 第一步 36/36），是 clipping
干预的高信息条件；③ 计算成本明显低于 100M。

## 3. G3a — operator-level control（B3）

- 输入：冻结 G1a bank（seed 2042 × steps 26/64/102），**不重训**。
- 每 state 4 streams（沿用 G1a 冻结 E2 manifest），H={1,4,8}。
- 同一 baseline trajectory（clip=1.0 native replay）上构造两个 tangent
  版本：
  - (A) 冻结 C（真实 J_clip，a0e2 runner 原样）；
  - (B) C_identity：仅在 tangent propagation 中把**未来步**（h>1）的
    clip-JVP 替换为 identity；h=1 脉冲仍经 J_clip(g_t)；HVP、AdamW 传播、
    baseline 全部不变。
- 验证：C_identity ≈ BH_fixed（容差 max abs gap ≤ 1e-12）。
- **G3a 不是最终科学 no-clip 结论**（baseline 本身来自 clip=1 训练）；
  它只回答"BH_fixed 与 C 的差异是否确实来自 clipping Jacobian operator"。
- 不一致处理：先查实现定义，不得直接进 G3b；确认实现问题则修复重跑
  G3a 并记 CHANGE_LOG；不可解释科学差异 ⇒ CONTRACT_MISMATCH 停止。

## 4. G3b — 真正 no-clip training intervention

- 唯一科学改变：clip threshold 1.0 → **禁用**。
- 训练实现：训练循环**真正不调用** `clip_grad_norm_`（逐步记录
  clip_grad_norm_called=false）；preclip norm 仍测量但不改变梯度；
  记录 counterfactual would-clip（preclip>1.0）指示。
- 其余全部与 G1a 相同：模型结构、D0、tokenizer、seq 128、objective、
  AdamW(0.9/0.999/1e-8/0.01)、LR schedule、global batch、tokens/step、
  总步数预算 128、BF16 语义、data order 规则（train_batch_seed=seed+1704）、
  seeds 2042/2043/2044、response 定义。
- registered states：step 26/64/102（EARLY/MIDDLE/LATE），共 9 个
  no-clip states。保存完整 model/optimizer m/v/RNG/scheduler/step/data
  position/metadata。
- 冻结 response 原语（run_theta_branch / a0c / a0e2）只通过
  g·min(max_norm/(‖g‖+1e-6),1) 实现 clip；no-clip 条件传入
  CLIP_DISABLED_MAX_NORM=1e30，clamp 精确等于 1.0（数学恒等、梯度位级
  不变）。该 sentinel 仅因冻结框架无其它代码路径而使用，**训练不使用**。

## 5. Smoke gate（B6，正式 no-clip 实验前）

seed 2042 / EARLY(step 26)。检查：checkpoint 恢复、optimizer state、
q 对齐、FD +/- CRN、rho0-noop、H1 B=BH_fixed=C、H1 analytic vs FD、
HVP finite、baseline replay、无意外 optimizer step、全部 finite；
**no-clip 专项**：任何分支不得报告 clipping active；no-clip 下
BH_fixed 与 C 在全部记录 H 上 max abs gap == 0（期望位级一致）。明显
不一致先定位，不得继续正式统计（理论退化失败 ⇒ CONTRACT_MISMATCH）。

## 6. 正式 G3b 矩阵（B7）

3 seeds × 3 stages = 9 states；每 state 4 条预注册 future streams；
H={1,4,8}；27 个 state-horizon 单元。全部计算 FD/B/BH_fixed/C
（用于证明机制 collapse，不是只算 C）。

## 7. clip=1.0 对照（B8）

直接读取冻结 `outputs/generalization_g1a_26m_v01/`。paired identity：
G3 沿用 G1a 的 E1/E2 stream seed 生成规则（同一 base、同一
two_level_seed 派生），future stream 逐位一一对应；若不能对应则明确
记录并降级为 condition-level comparison（本轮按构造可 paired）。

## 8. 指标（B9）

每个 condition/seed/stage/H 记录 chi_FD/chi_B/chi_BH/chi_C 及
absolute/relative/signed error（定义继承 G1a，`_metric`/`relative_error`
从冻结 a0c 模块 import）。gain_curvature = err_B − err_BH_fixed；
gain_clip = err_BH_fixed − err_C。主机制指标
Delta_clip = mean(gain_clip_clip1) − mean(gain_clip_noclip)；
分别给 pooled / H1 / H4 / H8 / per-seed / per-stage。

## 9. 判定（B10，冻结，结果前写入）

- **MECHANISM_CONFIRMED**：同时满足
  1. G3a：C_identity 与 BH_fixed 在容差内一致（max abs gap ≤ 1e-12）；
  2. no-clip 正式条件 pooled |gain_clip| ≤ 0.2 × clip=1.0 pooled
     |gain_clip|；
  3. no-clip 下 BH_fixed/C 预测本身近似一致（max rel gap/|chi_FD| ≤ 1e-6）
     且二者对 FD 保持合理预测能力（各自 pooled R² ≥ 0.5 且 mre ≤ 0.5）；
  4. Delta_clip > 0。
- **MECHANISM_PARTIAL**：operator control 成立且 gain_clip 明显缩小
  （shrink ratio ≤ 0.5）但未达 20% 阈值，或 seed/stage 存在异质性。
- **MECHANISM_NOT_CONFIRMED**：关闭 clipping 后 BH_fixed 与 C 仍保持与
  clip 条件相近的差距（shrink ratio > 0.5），或 operator control 本身
  无法支持差异来自 J_clip。
- **RESOURCE_LIMIT** / **CONTRACT_MISMATCH**：按 B3/B6 定义。

## 10. 统计（B11）

paired comparison（seed/stage/stream/H 严格一一对应）；mean/median/
per-seed/per-stage interaction；seed→stage→stream 层次 cluster
bootstrap（20000 次，rng seed 20260916）给 mean interaction 的 95% PI。
主问题是机制 interaction，不只报 pooled R²。

## 11. 训练动力学环境（B12）

clip=1.0 与 no-clip 同时记录：training loss、gradient norm、preclip
norm、parameter norm、LR、future gradient norm、HVP proxy、clip
activation。no-clip 条件 clip activation 定义 0，另记 counterfactual
would-clip 指示（不影响训练）。仅用于解释，不得事后筛状态。

## 12. 科学边界（B13）

- G3a 不得当作真正因果训练 regime 证据；主证据来自 G3b。
- 不得写"clipping 是唯一反馈机制"（curvature 仍存在）。
- 不得写"无 clip 时 C 一定完全等于 FD"（理论预测只是 BH_fixed 与 C
  退化为同一切线结构；FD 仍含有限 ε 非线性误差）。
- 不得因 no-clip 训练 loss 好坏改变状态选择。

## 13. 资源规则（B14）

允许共享 GPU；不得杀外部任务；OOM 按项目既有合法方式处理，不得改
模型/seq/global batch/H。

## 14. 代码与测试（B15）

新增 `research/src/fr/g3_clipping_intervention_v01.py`、
`research/scripts/g3_clipping_intervention_v01.py`、
`research/tests/test_g3_clipping_intervention_v01.py`；不得修改旧
A0E2/G1a 测试语义。测试至少覆盖：no-clip J_clip=I；no-clip synthetic
下 BH_fixed==C；operator-control identity；H1 B=BH=C；interaction
metric；manifest freeze；paired identity；no-clip 训练无意外 clipping。

## 15. 输出（B16/B18）

`outputs/generalization_g3_clipping_intervention_v01/`：G3_PROTOCOL.md、
experiment_config.json、training_manifest.json、checkpoint_manifest.json、
future_stream_manifest.json、operator_control/{summary.json,per_stream.csv}、
no_clip_regime/{summary.json,per_stream.csv,per_state_horizon.csv,
mechanism_decomposition.csv}、results/{clipping_interaction.csv,summary.json}、
G3_CLIPPING_INTERVENTION_REPORT.md、CHANGE_LOG.md、COMMANDS.log、
ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES.txt、SHA256SUMS.txt、
g3_clipping_intervention_audit_package.tar.gz（不含 checkpoint/optimizer
大文件/大 q/HVP 张量；含 source/config/protocol/summary/CSV/report/
manifests/hash；自校验 + tar SHA256）。

## 16. 执行后 provenance 勘误说明（2026-09-16 增补，仅文字，不改任何数值）

本节后置于全部执行完成之后，目的是澄清 §0/§7 "clip=1.0 对照直接读取
冻结 G1a、不重跑" 的执行实况，**不构成对协议阈值的任何修改**。

1. **直接读取的部分（真直读，冻结制品原字节未动）**：clip 条件的
   `chi_fd` / `chi_B` / `chi_C` 三 seed 全部 9 态 × 4 流 × H={1,4,8}
   来自冻结 G1a `results/per_stream.json`（SHA256
   `361243f33e10cf156b185981cc254e455dbde0df429fe72930edf472e2967be3`）；
   seed 2042 的 `chi_BH_fixed` 来自冻结 G1a `mechanism/*.json`
   （机制子实验 G1a 只跑了 seed 2042）。
2. **G3 clip-reference extension（非直读、非重训）**：seed 2043/2044 的
   `chi_BH_fixed` **不存在于** G1a 冻结制品（G1a 机制子实验仅 seed
   2042；冻结 `results/per_stream.*` 无 BH_fixed 字段）。该 72 行
   （2 seeds × 3 stages × 4 streams × H={1,4,8}）由本实验 `clipref`
   子阶段（COMMANDS.log 2026-09-16T12:43:33Z）用**冻结 a0e2 runner**
   （grad_clip=1.0）对**只读** G1a bank checkpoint/theta/q 新算得到，
   未重训任何 checkpoint、未改动 G1a 任何字节（G1a SHA256SUMS 不变）。
   产物 `clip_reference/per_stream_bh_fixed.csv`，SHA256
   `82808f7541724eb07e09ad2176f5551d3891dc8867a4e0bd241b7b07e457e1a0`。
   即用冻结轨迹补算了 clip 条件 BH_fixed 科学评估，属于**新增科学
   评估**而非"全部直读冻结 G1a"。
3. **extension 的位级验证**：同一 clipref 路径对 seed 2042 重算 vs
   冻结 mechanism JSON，max abs gap = 0.0（位级一致），证明新算路径
   在存在冻结值的区域精确复现冻结值。
4. **口径声明**：G3 正式 gain_clip / Delta_clip 统计使用 official-cell
   口径（4 流 chi 先平均、再对 cell-mean 求相对误差），与 no-clip
   条件完全同口径；这与 G1a 报告 §7 机制子集数字（stream-level
   误差先平均）数值相近但**聚合层级不同**，两数字各自冻结、互不替代。
