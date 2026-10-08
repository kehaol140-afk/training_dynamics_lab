# F4-A — Scale × Phase × Seed Stability Map 预注册

- 冻结日期：2026-09-30，**在任何 F4-A phase-map 计算之前**提交本文件。
- 上游：F0 `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE` → F1 `V_FLOOR_CAUSAL` → F2 `PRACTICAL_HYPERGRADIENT_FAILURE` → F2.5 `TRUST_REGION_COLLAPSE_CORRELATED_ONLY` → F3 `SPECTRAL_INSTABILITY` → F3.5 `METRIC_DEPENDENT`（`docs/F4A_EXISTING_BANK_INVENTORY.md` 为输入清单）。
- 本轮为机制制图（mechanism mapping），**不是方法实验**：不做 F4-B、不开发 optimizer、不下载 1B、**零新训练**、只用既有 74 个 bank state。

## 1. 核心问题（冻结）

- **RQ1**：curvature-driven sensitivity amplification 是否随 training progress 系统衰减/自愈？
- **RQ2**：phase profile 是否跨 seed 可复现？
- **RQ3**：规模增大时 unstable 强度是否更强、unstable window 是否更长、self-healing 是否推迟？
- **RQ4**：哪些 cheap local indicators 预测 multi-step instability？
- 本轮**不拟合最终 scaling law**（只做 §12 的 exploratory candidate comparison，标 `EXPLORATORY_MODEL_ONLY`）。

## 2. 冻结的实验对象

74 个既有 state（inventory CSV 为准）：5M×8seed×7step（46–82）、26M×3seed×3step（26/64/102）、100M×3seed×3step（26/64/102）。共享协议（inventory 实证）：D0 corpus、partition_v07、seq128、batch 4×accum2、bf16 autocast、AdamW(.9/.999/1e-8/wd.01)、clip 1.0、LR 5e-4 cosine on 256-micro 轴、planned_total=128 steps。progress **p = step/128**（全部 scale 同一可恢复定义）；tokens_seen 记录（=step×1024，g1a/g1b metadata 自带 cumulative_tokens 交叉核验）。

## 3. future streams（冻结）

- g1b (2042,26)/(2043,26)/(2042,64) 复用 F3 frozen streams（stream_id 1/2/3）——服务于 §10 reproduction gate 的逐位锚定。
- 其余 71 state：新建 **F3-protocol** 确定性 stream：`stream_seed = derive_stream_seed(20260926, id)`，`id = 4000 + bank_index`，bank_index 为 inventory CSV 行序（0-based，scale 顺序 v07_5m → g1a_26m → g1b_100m，seed/step 升序）。manifest SHA（`manifest_sha256(batches)`）随每单元 config.json 落盘。同一 state 的 B/C/source-off/indicators 共用该 stream；不看结果重采样。

## 4. 冻结的 metrics（H ∈ {1,2,4,8,16,32}，逐步传播，读出点即 H 集合）

管线的算子/臂语义与 F2/F3 current frozen semantics 完全一致（pre-step grads → clip → 每臂自身 HVP（α=1 才有）→ hom_step → advance → +source）：

- **Metric A** `G_B_theta(t,H) = ‖dθ_H‖_B / max(‖dθ_1‖_B, eps_metric)`（α=0 optimizer-only forced 臂；θ 通道）。
- **Metric B** `G_C_theta(t,H)`（α=1 full-curvature forced 臂）。
- **Metric C（核心）** `R_curv(t,H) = G_C_theta / (G_B_theta + eps_metric)`；重点 `R_curv_16`、`R_curv_32`。
- **Metric D** `G_SO_theta(t,H)`：source-off impulse，q₀ = F3 冻结定义（θ-only 单位切线，方向 = 该 state 首步 LR source），α=1；`lambda_SO(t,H) = ln G_SO_theta / H`；重点 H=16/32。
- **Metric E** `S_eta(t,H) = ‖dθ_{t+H}/d log η‖` := α=1 forced 臂的 ‖dθ_H‖（未归一化终范数）。h_eta（validation inner product，F2 语义）在 H 读出点保存为 auxiliary，**不作 gate**。
- `eps_metric = 1e-30`。停止规则与 F3 相同（切线非有限或 ‖dz‖>1e30 → 该臂停，如实记录）。

## 5. cheap local indicators（每 state，逐步或首步，全部复用 F0–F3 已验证量）

1. v-floor：`fraction_sqrtv_lt_{eps,10eps,100eps}`（F0 定义，Adam eps=1e-8）。
2. 状态通道放大 `A_theta/A_m/A_v`：F1 已验证闭式 estimator（c=0 锚点 per-coordinate Jacobian 增益，对单位随机方向；实现直接调用 F1 模块，不重写）。
3. module shares（θ 通道，F0 九组归并为 7 组：attn_q/k/v/o、mlp、embed、norm）于 H=32 终态。
4. clipping：`clip_active`、`clip_coefficient`、`grad_norm`、`clip_threshold=1.0`、`grad_norm/clip_threshold`（首步 + 全 32 步中位）。
5. LR：`lr`、`normalized_lr = lr/5e-4`（首步）。
6. `K_lr = ‖H_t q_lr‖ / (‖q_lr‖+eps)`：首步 HVP 沿 q₀（LR-source 方向）的范数（每 state 一次，随 source-off 臂的 h=1 计算）。

## 6. 计算预算（冻结）

每 state 仅做：B/C/SO 三臂 H≤32 传播 + §5 indicators。**禁止**：Arnoldi、eigendecomposition、left/right eigenspace、module causal ablation、α sweep、projector、任何 Fisher/二阶大规模估计。单 state 预算 ≈ 2 HVP/步 × 32 步 + 3 hom/步（5M/26M 显著更便宜）。

## 7. 执行顺序与 reproduction gate（冻结）

顺序：**100M → 26M → 5M**（100M 先验证 runner）。gate：F4-A runner 在 g1b (2042,26)/(2043,26)/(2042,64)（F3 streams）上，B/C/SO 三臂 H=32 终范数与 F3 已提交 raw（rq1 α=1.0 臂、rq2 q0 臂、compat B 臂）**逐位相等**（同一管线同一环境的强化判据，严于"定性复现"，frozen reference 语义），且逐步 loss 与 current-env frozen reference 逐位一致。**gate FAIL → 停止主 map 并报告**。gate PASS → 其余 71 state 全跑。

## 8. 原子化 / resume

单元 = (scale, seed, step)。目录 `outputs/f4a_phase_map/raw/<scale>/seed_<seed>/step_<step>/`：`config.json`（stream seed/SHA、config SHA、protocol）、`metrics.json`（§4 全量 + §5）、`trajectory.csv`（逐步三臂范数 + loss/clip）、`run.log`、`DONE`。resume：DONE 存在即跳过。禁止多进程写同一汇总 CSV——汇总由 postprocess 单点生成。

## 9. 跨 seed summary（PHASE 10）

按 scale × progress-bin 汇总 `log R_curv_32`、`lambda_SO_32`、`log S_eta_32`：median/mean/std/IQR/min/max/n_seed。progress-bin 冻结为 {EARLY: p∈[0,0.34)、MID: [0.34,0.67)、LATE: [0.67,1]}，同时保留连续 p 的原始行（不做 bin 替代）。**所有 seed 全部报告，不做挑选**。

## 10. descriptive labels（冻结阈值，仅描述性）

`CURVATURE_DOMINANT: R_curv_32 ≥ 10`；`CURVATURE_MODERATE: 3 ≤ R_curv_32 < 10`；`CURVATURE_WEAK: R_curv_32 < 3`；`SOURCE_OFF_GROWING: lambda_SO_32 > 0`。不称 phase transition，不据结果调整。

## 11. p_heal（探索性，censored 处理）

per (scale, seed)：`p_heal` = 首个满足"R_curv_32 < 3 且其后全部已观测 state 保持 <3"的 p。相邻 checkpoint 间隔内不知精确 crossing → 记 `[p_lower, p_upper]` 与 status ∈ {OBSERVED, INTERVAL_CENSORED, NOT_HEALED_IN_COVERAGE, INSUFFICIENT_POINTS}。

## 12. exploratory candidate models（EXPLORATORY_MODEL_ONLY，LOSO）

对 `log R_curv_32`（亦对 `lambda_SO_32`）比较：A `~ a + b log N + c p`；B `~ a + b log N + c log(step+1)`；C `scale 固定效应 + 线性 p`；D `~ a + b log N + c p + d logN·p`。评估：leave-one-seed-out 的 R²/MAE/Spearman（按 seed 聚合预测）。不宣布 power law/exponent/universality。

## 13. verdict（冻结标签集）

`SCALE_PHASE_STRUCTURE_CONFIRMED / PHASE_STRUCTURE_ONLY_NO_SCALE_TREND / SCALE_TREND_NO_PHASE_STRUCTURE / HIGH_SEED_HETEROGENEITY / NO_GENERALIZABLE_STRUCTURE / INSUFFICIENT_COVERAGE / UNRESOLVED`（允许主+次）。判定前必须回答 §1 四问；操作化判据：phase 结构 = 每 scale 内 EARLY 与 LATE 的 median log R_curv_32 差异方向在 ≥2/3 seeds 一致；scale 趋势 = peak R_curv_32（per seed max）与 p_heal 上界随 N 单调（Spearman ≥ 0.7 跨 scale）；seed 异质性 = 同 (scale,step) 的 log R_curv_32 跨 seed 极差 > 1（即 >10×）且方向不一致。

## 14. 输出

`docs/F4A_SCALE_PHASE_SEED_MAP_REPORT.md`、`docs/F4A_EXISTING_BANK_INVENTORY.md`、`docs/F4A_MISSING_COVERAGE_REQUESTS.md`、`docs/F4A_MODELING_FACTS.md`；`outputs/f4a_phase_map/{summary.json, bank_inventory.csv, phase_map_long.csv, phase_map_summary.csv, healing_intervals.csv, indicator_correlations.csv, exploratory_models.csv, figures/, raw/}`。

## 15. 红线

不覆盖 F0–F3.5 结果；不 push；不 reset/rebase/clean；legacy 只读；零新训练；不为任何结论调整 §4/§10/§13 冻结量；负结果与 NOT_ESTIMATED/MISSING 原样保存；长任务 tmux/setsid + 资源快照 + OOM 只 fail 当前单元。
