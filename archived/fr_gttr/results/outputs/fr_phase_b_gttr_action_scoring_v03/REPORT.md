# Phase B v3 — GTTR 动作前评分 + 真实分支验证：正式报告

轮次：`fr_phase_b_gttr_action_scoring_v03`
日期：2026-09-23
协议：`FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md` +
`FR_PhaseB_SharedGPU_TokenPrimary_Amendment_v1.md`（共享 GPU；墙钟仅
OBSERVATIONAL_SHARED_GPU 记录，5400s 为一次性运维提示，不作任何门）
环境：conda env `fr_minimind`，CUDA_VISIBLE_DEVICES=0（共享 RTX 5090），
PYTHONPATH=research/src:research/scripts

## 0. 执行摘要（结论先行）

**本轮在预测封存之前被预注册数值门终止：**
`LOCAL_OPERATOR_NOT_CALIBRATED` @ **S3**（seed 2031, n0=82, gm0=164）。

- 第一 Krylov 方向 FD half-check：`fd_rel = 0.7061 > 0.20`（门限），
  `fd_cos = 0.7282` —— δ 与 δ/2 两点 FD 斜率严重不一致，局部场算子
  在该状态下不可校准。
- 按 v3 §3.2：**停止 GPU 主流程**；不调 ε、不剔除状态、不部分封存、
  不裁剪 horizon。预测文件从未生成，`seal.json` 不存在，`main` 未启动。
- 门内状态对照：S0 fd_rel=0.0011、S1 fd_rel=0.0660、S2 fd_rel=0.0206
  （均 ≪ 0.20，rank=4 FULL_Q4）——门失败是 **S3 局部性质**，非全局
  实现故障（同一 SIM/FD 代码路径在 S0–S2 通过）。
- 预算与账本：全部计数含 smoke/失败，未超任何 cap（见 §6）。
- 科学读数 B–F 为 NOT_TESTED（预测未封存 ⇒ 真实分支被协议禁止）。
  这是门的预期行为：评分模型在 6/6 状态可构建是 main 的前置条件。

## 1. 预注册状态与门结果

| 阶段 | 结果 |
|---|---|
| preflight（D0 sha、V_monitor≡A1、split sha、bank 6 态、stream 冲突、schedule 边界） | PASS |
| cpu_tests（T1–T8 + amendment 计时器隔离，31 项） | PASS |
| model_smoke（S0 完整 P2+Q4+half-check+q0，30fw+30bwd） | PASS（rank=4, fd_rel=0.0011, orth 1.1e-8） |
| native_smoke（S0 off/on 8 步，θ/m/v/step/lr 逐位一致、RNG 一致、数据 token 一致、lr_u0≡native schedule） | PASS（16 updates） |
| build：S0–S2 建模 + FD 门 | PASS（fd_rel 0.0011 / 0.0660 / 0.0206） |
| build：S3 FD 门 | **FAIL → STOP**（fd_rel=0.7061, fd_cos=0.7282） |
| build：S4、S5 | 未执行（停止门之后） |
| predictions.csv / seal.json | 从未生成 |
| main（3 臂 × 6 态真实分支） | 未启动（协议禁止：无封存预测） |
| analyze | 合成数据干跑验证通过；真实数据无（main 未运行） |

## 2. A–G 读数（按 v3 §9 结构；不适用项标注）

**A. 6 状态配置与数据角色**
S0–S5 = seeds {2030,2030,2031,2031,2032,2032} × n0 {46,82,46,82,46,82}；
bank = `outputs/e3_confirmatory/v07/seed_*/states/state_step_0{46,82}.pt`
（full_state v0.2，单参数组 AdamW betas=(0.9,0.999), eps=1e-8, wd=0.01，
τ=1.0，accum=2，1 update=1024 input slots）。冻结 schedule
`finite_cosine_then_hold_5e-5_v01`；动作
η_k^u = η_base(n0+k)·exp(u·1[k<4])，u∈{0,−0.05,+0.05}。
数据角色：V_monitor=D0[4096:8192)（4096 行，与 A1 逐字节一致）；
V_control_ref64=validation_split 的 64 个 v_score ID（FP32、no-autocast、
batch 8、token 分母 sum/count）；SIM 流 197006–197011；真实流 97006–
970011 严格配对（未启用）；smoke 流 97020。V_final 调用=0。

**B. H4/H64 预测动作与三候选值**
NOT_TESTED —— 预测从未生成/封存（门终止于 S3；v3 禁止部分封存）。

**C. 真实三分支 H4/H64 control 差与 monitor gain±SE**
NOT_TESTED —— main 未启动。

**D. H1/4/16/64 固定集 SSE/TSS/E 与覆盖**
NOT_TESTED（评分误差）。可用的构建侧诊断：S0–S2 rank=4 FULL_Q4；
‖y_j‖ 见 `model_construction_reconstructed.csv`；
`MODEL_COMPUTE_FOOTPRINT` 见 G。

**E. H4/H64 选择效用 vs 固定 up/down（含 3-seed 警示）**
NOT_TESTED。

**F. e_total/e_propagation/e_readout 分解**
NOT_TESTED（分解恒等式已在 T7 与合成数据干跑中验证：144/144 行
identity_holds=True）。

**G. 成本（全部含 smoke/失败；OBSERVATIONAL_SHARED_GPU 不计门）**

| 计数 | 计划 | 实际 | cap |
|---|---|---|---|
| optimizer updates | 1168 | 16 | 1280 |
| 全部 forward | 13562 | 570 | 14500 |
| 全部 backward micro-calls | 2546 | 162 | 2920 |
| model-build fwd / bwd | 210 | 130 | 360 |
| FD 方向估计 | 35 | 22.0 | 50 |
| monitor 整库 eval | 81 | 3 | 84 |
| control 整库 eval | 81 | 3 | 90 |
| 虚拟算术步 | 2688 | 1536 | （算术，仅记录） |
| V_final 调用 | 0 | 0 | 0 |

token 分账：SIM probe 1024 slots/状态 ×5（含 S3 部分）累计 5120 slots /
4855 nonpad / 4815 valid（账本 counters.json）；真实训练 token 仅 smoke
16 updates（S0 流 97020，逐 step 三列见 step_metrics——build 时已清空
probe 行，最终账本以 counters.json 为准）；monitor 纯评价 3 次（smoke）
×4096 行；control 纯评价 3 次 ×64 行。峰值显存 0.82 GB。
`TIME_TO_TARGET/WALL_CLOCK_SPEEDUP/ENERGY_BENEFIT = NOT_EVALUATED_SHARED_GPU`。

## 3. S3 门失败诊断（有限、不下结论外推）

事实（`model_construction_reconstructed.csv` + resource_log）：
- 同一 SIM/FD 代码路径，S0–S2 的 fd_rel 为 1e-3–7e-2 量级；S3 第一方向
  δ 与 δ/2 的 FD 斜率相对偏差 0.706（余弦 0.728）。
- S3 的 g0/参考位移构建未报错（门在 half-check 触发，属 v3 设计的
  "局部算子不可校准"分支）。
- 实现层排除：T3 线性场解析解吻合（rel<1e-6）、随机性恢复负测试带牙齿、
  计时器隔离测试通过 —— 门失败不是实现 bug 的已知形态。

协议内解释边界：v3 冻结，不允许 ε 调参/状态剔除/几何改动，故本轮
**不对**"GTTR 场假设在 S3 为何不成立"展开机制归因（候选方向留给后续
轮次设计：更强非线性、投影外变量、秩选择等，均超出本轮授权）。

## 4. 封存与不可变性

- `seal.json` 不存在；`predictions.csv/selections.csv/virtual_h_metrics.csv`
  从未写出。`_main_arm_leak` 检查从 build 到 main 全程未放行任何真实臂行
  （观测文件仅有 arm=smoke 的 12480 行，均为封存前合法探针）。
- 账本 counters.json 自 smoke 起累计、未重置；失败计数=1（S3 门停止）。
- 未 commit/push；未动旧结果旧代码；未进入 Phase C；未训练 controller。

## 5. 交付物清单（outputs/fr_phase_b_gttr_action_scoring_v03/）

- `config.json`、`manifest.csv`（preflight 冻结记录）
- `model_smoke_checks.json`、`smoke_checks.json`（PASS 记录）
- `eta_estimate.json`（ops 提示；exceeds_ops_hint=false）
- `model_bundles/S{0,1,2}_bundle.pt`（q0/g0/V/Y/范数；S3+ 不存在）
- `model_construction_reconstructed.csv`（S0–S2 由 bundle 后验重构，
  S3 取自日志；标注 RECONSTRUCTED/GATE_STOP 来源）
- `round_status.json`、`resource_log.jsonl`、`counters.json`
- `plot_gate_fd_rel.png`（1/1 图）
- 单元测试：`research/tests/test_phase_b_gttr_v03.py`（31 项）
- 实现：`research/src/fr/phase_b_gttr_score_v03.py`、
  `research/scripts/phase_b_gttr_v03.py`

## 6. 复现与下一步

复现（新一轮、新预测、新封存）：preflight → cpu_tests → model_smoke →
native_smoke → build →（门通过时）main → analyze。恢复语义：本轮无
安全边界可恢复（预测未封存）；任何重试都是新轮次。

下一步候选（不在本轮执行）：针对 S3 类状态的局部算子不可校准性做
机制诊断轮（新协议新预算），或按 v3 原样重试以检验门失败的重现性。

**STOP —— 不自动进入 Phase C。**
