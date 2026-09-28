# CHANGE_LOG — FR_G3_CLIPPING_INTERVENTION_V01

记录 G3 执行期间的实现修正与重跑（如有）。阈值与协议不变。

## 2026-09-16 预注册后实现修正（测试阶段发现，任何 G3 结果打开之前）

1. `cmd_g3a_operator`：清理 checkpoint micro-step 恢复逻辑（先
   `load_full_training_state` 取 metadata 再构造 runner kwargs；移除草稿
   条件表达式）。纯实现整理，无语义改变。
2. `_run_baseline_tangents_c_identity` 辅助函数来源从 a0c runner 改为
   a0e2 runner（`_zeros_like_named/_snapshot_gradients/_hvp_on_accumulated_
   batches/_norm/_difference_norm`），与该 driver 逐行复制的源函数一致。
3. `cmd_train`：每 seed 重初始化模型后重算 `named`（修正 param_norm 引用
   旧模型参数的 bug——测试阶段发现，未进入执行）。
4. `cmd_tangents`：每 stream 额外落盘 `no_clip_regime/streams/*.json`
   （step_trace 供 analyze 动力学统计）。
5. `cmd_analyze`：future-trace 统计改从落盘 JSON 读取（修正原草稿中
   glob 路径不存在的问题）。
6. `cmd_audit`：大包排除规则收窄为 `bank/` 与 `theta/seed_*/q/`
   （manifest 与 theta_mapping JSON 保留入包）。
7. operator_control/summary.json 键名对齐 render_report（`n_cells`、
   `max_abs_gap`、`mean_abs_gap`、`max_rel_gap`）。
8. 单元测试（18/18 通过，GPU 无关）：no-clip J=I、合成 BH_fixed==C、
   operator-control 恒等、H1 B=BH=C、interaction bootstrap、manifest 冻结
   校验、paired identity 对冻结 G1a 记录、训练循环静态审计
   （`clip_grad_norm_(` 不得在 cmd_train 源码中出现）、冻结阈值 verdict
   分层、指标必须来自冻结 a0c 模块。

（执行期条目在下续记；如无异常将标注“执行期无修正”。）

## 2026-09-16 执行期 — G3a 首次运行后的 cross-check 定义修正（重跑 G3a）

- **观察**：首次 G3a 主检验通过（同 run 内 C_identity ≡ BH_fixed，
  36/36 位级一致），但 frozen cross-check max gap = 0.039。
- **定位**：逐行比对确认 ablation driver 与冻结 a0e2 代码一致；差异源於
  冻结实现的 HVP 方向取 C 切线轨迹（`tangent_theta_by_name=c_theta`），
  identity override 从 horizon 3 起改变 c_theta → HVP → ablation run 的
  BH_fixed 输入。这是**设计上的预期行为**，不是实现不一致。
- **修正**：cross-check 收窄为「未修改的 a0e2 重跑 vs G1a 冻结 mechanism
  JSON」（该比较位级一致）；ablation-vs-real 的 BH_fixed 差距单独记录为
  信息项 `ablation_BH_fixed_vs_unmodified_real_max_abs_gap` 并附说明。
- **判定影响**：无。operator control 的冻结通过准则是同 run 内
  C_identity ≈ BH_fixed（max abs gap ≤ 1e-12），首次与重跑均满足
  （max abs gap = 0.0）。按 B3 流程修复实现定义并重跑 G3a，记此条目。

## 2026-09-16 执行期 — train 配置补键（任何训练结果产生之前）

- `train` 首次启动即因 `KeyError: train_batch_seed_offset` 退出
  （任何 state 未写出）。
- 修正：配置 training 块补数值键 `"train_batch_seed_offset": 1704`
  （与冻结规则 train_seed+1704 及 G1a 配置一致；纯文档/实现缺口补齐，
  无语义改变）。重跑 train。



## 5. 执行完成登记（2026-09-16，无进一步修正）

流水线全阶段（preregister → g3a_operator → train → theta → smoke → clipref →
tangents → analyze → audit）一次性顺序完成；§1–§3 所列全部修正后无新缺陷。
analyze/audit 干净通过；单测 18/18 全绿（完成时复跑确认）。文档 B17 四项
（FR_GENERALIZATION_STATUS / FR_GENERALIZATION_HANDOFF / FR_PROJECT_STATUS §29 /
FR_EXPERIMENT_INDEX 头部+G3 行）均已更新。冻结结论 MECHANISM_CONFIRMED。

## 6. 只读 provenance 专项核验（2026-09-16，仅文字修正）

对 clip-reference provenance 与 gain_clip 口径做只读核验：无新训练、无新
replay、无数值制品改动。结论：

1. G1a +0.01472（summary.json mechanism_decomposition.pooled_h_gt_1，
   = 0.014718237630233754）= seed 2042 机制子集、stream-level 相对误差
   先对 4 流平均、再 per-cell 求 gain、6 cell 等权平均；已从
   mechanism_decomposition.csv 独立重算复现。
2. G3 clip=1 +0.09398（results/summary.json clip_condition_gains，
   = 0.09397950627466435）= 3 seeds 全矩阵、official-cell 口径（4 流 chi
   先平均、cell-level 相对误差）、18 cell 等权平均；per-seed 2042/2043/
   2044 = +0.015864223389/+0.017845419115/+0.248228876320，已从最底层
   stream 数据独立重算全部复现（含 no-clip +0.00000/+0.158240861437 与
   bootstrap mean +0.0856330527、95% PI、P(>0)=0.99985 逐位复现）。
3. 2043/2044 的 clip 条件 BH_fixed **不存在于** G1a 冻结制品（G1a 机制
   子集仅 seed 2042；per_stream 无 BH_fixed 字段），系本实验 clipref
   新增科学评估（**G3 clip-reference extension**：冻结 a0e2 runner +
   只读 G1a bank，未重训、未改冻结字节，2042 同路径 cross-check=0.0）。
4. 2042 两数字差异（0.01472 vs 0.01586）来源 = 误差聚合层级（stream-level
   先平均 vs cell-mean 后取误差），非 stream 数/绝对相对/权重/H 加权差异。
5. interaction 两条件严格同口径 paired（同 metric、同 support、同
   official-cell 聚合、同权重）；bootstrap 输入为 stream-level 配对
   （n=72），其 observed mean +0.08564 与 official-cell pooled +0.09398
   为同一配对的不同聚合层级，非矛盾；P(>0) 为 bootstrap 后验概率，
   非频率学 p-value。
6. 判定 MECHANISM_CONFIRMED 维持不变。

文字修正（仅 Markdown provenance 表述）：G3_PROTOCOL.md 增补 §16
（执行后 provenance 勘误说明）；G3_CLIPPING_INTERVENTION_REPORT.md §6/§7
补 provenance 与口径说明；FR_GENERALIZATION_STATUS §3c、
FR_GENERALIZATION_HANDOFF §D.1、FR_PROJECT_STATUS §29 相应措辞、
FR_EXPERIMENT_INDEX G3 行同步。SHA256SUMS.txt 仅更新上述两个 Markdown
的哈希行；审计 tar 为核验前历史快照，其 SHA256
（3c2c6fbf…ff3b）不变、仍描述该快照。数值制品（CSV/JSON/summary）
零改动。

## 7. Phase A official-cell bootstrap（2026-09-16，纯统计对齐，只读）

新增 `officialboot` 子命令：与主点估计完全相同的 official-cell
estimator（4 流 chi 先平均 → cell-level 相对误差 → per-cell gain_clip
差 → 18 cell 等权平均），重采样层级 seed → stage → H-cell（H 在
stage 内整体成对），20000 次、rng 20260916。observed = +0.0939795063
（与冻结主点估计逐位一致）；bootstrap mean +0.09521030、median
+0.08317223、95% PI [+0.00688686, +0.32209561]、重采样中差值为正的比例
0.9978（明确不称为 p-value/posterior/significance probability）。
与 stream-level bootstrap 方向一致 → 无 STATISTICAL_MISMATCH；
official-cell bootstrap 为主统计口径，原 stream-level bootstrap 降为
robustness diagnostic。新增 results/official_cell_bootstrap.json 与
results/official_cell_bootstrap_cells.csv；旧 bootstrap 数值未动。
