# F2 — Hypergradient Pilot 预注册（100M，learning-rate channel）

- 冻结日期：2026-09-26，**在查看任何 F2 结果之前**。上游 gate：F1 verdict `V_FLOOR_CAUSAL`（docs/F1_MECHANISM_CLOSURE.md §11 允许本轮）。
- 目标：验证 F0/F1 的 ill-conditioning 是否实际损坏一个真实 hypergradient。不开发任何方法。
- 定义：`h_eta = d L_val(theta_{t+H}) / d log(eta)`。

## 1. 冻结对象

| 项 | 冻结值 |
|---|---|
| 模型 | MiniMind 100.78M（G1b 冻结配置：hidden 768 / layers 13 / heads 12 / kv 6 / inter 2432 / vocab 6400 / seq 128 / tied；代码 `../minimind_base/model/model_minimind.py`，只读） |
| states（cells） | g1b bank：`(2042, 26)`、`(2042, 64)`、`(2043, 26)`（EARLY×2 + MIDDLE×1；EARLY 为 F1 病态最强相位） |
| 恢复 | bank state 的 model + AdamW m/v/step（RNG 只影响 batch 序，已由显式 manifest 取代） |
| future streams | 每 cell 一条，冻结 `stream_seed = derive_stream_seed(20260926, {1,2,3})`；`make_permutation(len(train_indices)=3936, stream_seed)` → batchify(batch_size=4) → 每 optimizer 步 2 个 microbatch（accum 2，全局 8 seq，1024 token/步，与 G1b 训练协议一致）；rollout 只用 `train_indices`（partition_v07.json），`eval_indices` 永不进入训练 batch |
| LR | 原生 finite-cosine 延续：`lr_h = get_lr(micro_step_h, 256, 5e-4)`（frozen `trainer_utils.get_lr`）；±delta 臂逐 步乘 `exp(±delta)`；tangent 臂不缩放 |
| L_val | partition_v07.json `eval_indices`（128 条冻结）mean CE；**主读数 fp32 无 autocast 前向**（臂间 CRN 同 batch 同 dtype），副读数 bf16-autocast（评测噪声对照） |
| H | 读出点 {1, 2, 4, 8, 16, 32}（单次 32 步 rollout 嵌套读出） |
| delta 网格 | log-η CRN central FD：{0.02, 0.05, 0.1, 0.25, 0.5}；**主 delta*=0.05**（A0O heritage），跨小 delta 收敛性作为 FD 有效性 gate |
| 训练语义 | bf16-autocast 前向/反向（GPU）、fp32 CPU 原生 AdamW、global clip 1.0（torch 语义，CLIP_DENOM_EPS=1e-6）——legacy 注册路径，逐臂完全一致（CRN） |
| 设备 | `CUDA_VISIBLE_DEVICES=1`（GPU0 外部占用） |

## 2. 冻结的 hypergradient 计算

- **Ground truth（FD）**：`h_FD(delta) = [L_val(eta·e^delta) − L_val(eta·e^−delta)] / (2·delta)`（CRN：同 manifest、同批、同 dtype 路径，仅 LR 不同）。
- **Tangent（forward 敏感度）**：在 δ=0 的 baseline rollout 上传播 u=log η 方向的切线 `dz/du = (dθ/du, dm/du, dv/du)`：
  - 齐次部分：vendored 冻结 `adamw_tangent_step`（dc=0；CRN 下梯度对 u 不显变，clip 系数不依赖 η）；
  - 非齐次项：`∂θ′/∂u = θ_{h+1} − θ_h`（AdamW 更新整体 ∝ η 的解析事实；单测锁定）；
  - `h_tangent(H) = <∇_θ L_val(θ_H), dθ_H/du>`（∇L_val：128 条 eval 集 fp32 无 autocast 反向，8×16 batch 平均）。
- **分支**：B（optimizer-only）与 C（含 HVP 反馈：`dc_{h+1} = HVP(θ_h; dθ_h/du)`，逐 步 GPU double-backward，经冻结 `clip_jvp` 施加 clip 的切线）。
- **精度臂**：
  - `fp64`：切线 CPU fp64 + C 分支 fp32-HVP（无 autocast fp32 double-backward；100M 显存余量充分）
  - `fp32`：切线 CPU fp32 + C 分支 fp32-HVP
  - `frozen`：切线 CPU fp32 + C 分支 bf16-autocast-HVP（legacy 注册路径）
- 三条臂共享同一条 δ=0 primal rollout（位级同一）。

## 3. 冻结的指标与停止规则

每 cell × H ∈ {1,2,4,8,16,32} × arm：
- `h_FD(delta)`（5 delta）；`rel_err(arm,H) = |h_tangent − h_FD(0.05)| / |h_FD(0.05)|`；符号一致；`||dθ_H/du||`、`||dm_H/du||`、`||dv_H/du||` 增长；NaN/Inf flag。
- 伴随记录（机制关联）：per-step clip 系数、preclip 范数、`||dθ_h/du||` 的模块份额（F0 分组）、该 state 的 σ_max / v-floor（取自 F1/F0 制品）。
- 跨 cell 相关：Pearson/Spearman（h_tangent vs h_FD 于 3 cell × H≤16 的 12 个点）。
- 停止规则：任一臂切线非有限 → 该臂停在该 H（记录），其余臂继续；primal 非有限 → 该 cell 终止。事后不调阈值。

## 4. 冻结的判定程序（F2 verdict）

前置 gate（FD 有效性）：在 H=1，`max_{δ∈{0.02,0.1}} |h_FD(δ)−h_FD(0.05)| / |h_FD(0.05)| ≤ 0.3` 于 ≥2/3 cell 成立；且 |h_FD(0.05)| 超过其跨 δ 离散度 2 倍（信号显著性）。不满足 → `UNRESOLVED`（FD 自身不可信），负结果原样保留。

按序判定（首个命中）：
1. `PRACTICAL_HYPERGRADIENT_FAILURE`：fp64/C 臂在 H≤16 出现 rel_err > 1 或非有限 或符号翻转（vs FD），且 gate 通过；
2. `PRECISION_LIMIT_ONLY`：fp64/C 全 H≤16 rel_err ≤ 0.25 且符号一致，但 fp32 或 frozen 臂 rel_err > 1 / 非有限；
3. `CONDITIONED_BUT_USABLE`：全部精度臂（fp64 与 frozen）H≤16 rel_err ≤ 0.25 且符号一致；
4. `NO_DOWNSTREAM_EFFECT`：|h_FD| 全 cell 显著低于 δ 离散度（hypergradient 目标本身可忽略）；
5. 其余 → `UNRESOLVED`。

## 5. 冻结的输出

`docs/F2_HYPERGRADIENT_PILOT.md`；`outputs/f2_hypergradient/{summary.json, hypergradient_vs_fd.csv, horizon_growth.csv, precision_comparison.csv}`（+ 运行日志、batch manifest SHA256）。

## 6. 红线

不开发 stabilization/optimizer/controller；forward AdamW 不动；legacy 与 bank 只读；不为通过 gate 改阈值；负结果原样保存。
