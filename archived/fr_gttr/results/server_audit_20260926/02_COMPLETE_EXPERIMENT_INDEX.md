# 02 — 完整实验索引（人类可读版）

> 每行一个实验/实验版本。verdict 一律取冻结制品。完整机器可读版见 `01_COMPLETE_EXPERIMENT_INDEX.csv`（同名列）。
> 通用信息（除非另行注明）：
> - 模型：MiniMind dense Transformer（anchor 4,983,168 unique params；hidden 256×4 层 GQA 8/4，vocab 6400，tied embeddings）
> - 数据：`outputs/e3_confirmatory/v07/e3c_subset.jsonl`（4096 条）+ minimind_tokenizer
> - 优化器：原生 AdamW β=(0.9,0.999) eps=1e-8 wd=0.01，global grad-clip 1.0，bf16 autocast（GradScaler 禁用），batch 4 × accum 2，seq 128，1024 input slots/optimizer step
> - 协议：CRN 三分支（baseline/+εq/−εq），pulse 位置 `backward → unscale_ → ±ε·q → clip_grad_norm_ → AdamW`（pulse-before-clip），ρ=0.25
> - 指标唯一定义源：`research/src/fr/a0c_adamw_response_propagation_v01.py` 的 `_metric`/`relative_error`（禁止重定义）
> - Git：仓库 2026-09-25 才首次 commit（3 commits，全部未 push）；此前所有实验的 commit 记录为 UNKNOWN(pre-git)

## A. E 链（基础设施 + 原始假设检验，09-01 → 09-09）

| # | experiment_id | 日期 | 位置 | 问题 | verdict |
|---|---|---|---|---|---|
| 1 | E0-replay-v02-smoke | 09-02 | outputs/e0_replay/v02_smoke | 保存态能否位级重放训练 | PASS（5/5 hash） |
| 2 | E0-replay-v03-extended | 09-02 | outputs/e0_replay/v03_extended | 更大配置精确重放 | PASS |
| 3 | E1-fluctuation-v03-pilot | 09-02 | outputs/e1_fluctuation | 构造 q、σ_q 可测 | PASS（技术；FR 关系不由此声明） |
| 4 | StageC-E1-formal-v04 | 09-02 | outputs/stage_c_v04 | R=16 formal 涨落测量 | PASS（σ_q=0.0893，首步 clip 率 0.75） |
| 5 | StageC-E2-linearity-v04 | 09-02 | outputs/stage_c_v04 | ε 线性域扫描 | TECH PASS / 科学判定需人工（ρ=0 no-op PASS） |
| 6 | E3-theta-mapping-v05 | 09-02 | outputs/e3_mapping | χ 可由旧 θ 预测？ | TECH PASS / 科学未自动判定（H8 r=−0.083 崩） |
| 7 | E3R-cross-seed-v06 | 09-03 | outputs/e3_replication | 跨 seed 复现 | TECH PASS / 科学未判（LOSO −0.0175） |
| 8 | **E3C-confirmatory-v07-bank** | 09-03 | outputs/e3_confirmatory/v07 | 预注册 H8 主检验 | **H8_CONFIRMATION: FAIL**（LOSO 5/8<6）；副产品=56 态冻结 bank |
| 9 | E4A-lowdim-state-v08 | 09-03 | outputs/e4a_lowdim_state | 低维状态预测 | EXPLORATORY（完整性 PASS 56/56） |
| 10 | E4B0-dynamic-v08 | 09-03 | outputs/e4b_dynamic_state | 数据审计 | DATA_INTEGRITY PASS |
| 11 | E4B1-q-rotation-v08 | 09-03 | outputs/e4b_dynamic_state | q 方向随 H 旋转？ | OBSERVATION（Rq 0.958→0.880） |
| 12 | E4B1b-branching-screen-v08 | 09-03 | outputs/e4b_dynamic_state | CPU 分支筛选 | EXPLORATORY |
| 13 | E4B2A-tau-q-v08 | 09-09 | outputs/e4b_dynamic_state | 动态塌缩时间尺度 | CURVE_COLLAPSE_CONFIRMATORY=NO（exploratory） |

## B. A0 审计预备链（09-09 → 09-10）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 14 | A0-data-audit-v01 | outputs/a0_fr_audit/v01 | bank 完整性/混杂/可靠性 | A0_DATA_INTEGRITY=PASS |
| 15 | A0-partial-fr-v02 | outputs/a0_fr_audit/v02_partial_fr | 控制 alpha_direct 后 B 显著？ | delta_R²=0.7706（p=4.7e-11） |
| 16 | A0-stratified-v03 | outputs/a0_fr_audit/v03_stratified | 分层 bootstrap | CI [0.7246, 0.8163] |
| 17 | A0-mechanism-v05 | outputs/a0_fr_audit/v05_mechanism | 机制回归 | mean 0.7644 CI [0.6719,0.8759] |
| 18 | A0-momentum-kernel-v06 | outputs/a0_fr_audit/v06_momentum_kernel | α(H)=A(1−β1^H)+C 拟合 | R²=0.9959 |
| 19 | A0-adaptive-audit-v08 | outputs/a0_fr_audit/v08_adaptive_audit | μ_q/β2/wd 特征 | R²=0.316（scaffold） |
| 20 | A0C-prep-state-audit | 根目录 5 个 FR_A0C_* 包 | 需要导出什么 optimizer 态 | need_export=exp_avg 等（PREP） |
| 21 | A0C-prep-theory-response | 根目录 3 包 | 闭式理论构建 | NEEDS_V_STATE_EXPORT→READY（PREP） |
| 22 | A0C-prep-mv-response | 根目录 6 包 + a0c_adamw_mv_theory.py | m/v 闭式理论迭代 | 多次失败迭代（ν 反常）（PREP，负） |
| 23 | A0C-momentum-theory-fit | outputs/a0_fr_audit/momentum_theory_fit_v01..v02_v07 | 动量公式拟合 | v02_v07 COMPLETE（A 模型基础） |
| 24 | mv-theory-final | outputs/a0_fr_audit/mv_theory_final | v 路径改善？ | **RETRACTED（禁引）** |
| 25 | mv-theory-consistent | outputs/a0_fr_audit/mv_theory_consistent | 严格语义重检 | FROZEN：v 路径 0/8 seed、0/8 horizon 改善 |
| 26 | A0C-root-prep-packages(17) | 根目录 17 个 FR_A0C_* 目录 | 上述预备链工作区 | legacy workspace（权威结果在 outputs/） |

## C. A0 机制链（09-10 → 09-12，项目核心正结果）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 27 | H1-adamw-jacobian-v01 | outputs/a0_fr_audit/H1_adamw_jacobian_v01 | H=1 局部 Jacobian 成立？ | YES（ε/4 rel err 0.000146–0.019；H>1 需 HVP） |
| 28 | **A0C-response-propagation-v01** | outputs/a0_fr_audit/response_propagation_v01 | A/B/C 阶梯 vs χ_fd | **C R²=0.9977 mre 0.0274；H>1 9/9 C 胜 B** |
| 29 | A0D-epsilon-convergence | outputs/a0_fr_audit/response_propagation_v02_epsilon | 残差=有限 ε 伪影？ | 是（ε/4 mre 0.00452；H≤8 无需记忆核） |
| 30 | A0E-curvature-decomposition-v01 | outputs/a0_fr_audit/curvature_decomposition_v01 | 初版曲率分解 | **VOIDED（BH 初始化 bug，禁引）** |
| 31 | **A0E2-curvature-decomposition-v02-bh-fixed** | outputs/a0_fr_audit/curvature_decomposition_v02_bh_fixed | 修复后分解 | H1 恒等 gap=0；curvature +0.0686 / clip-JVP +0.1148 |
| 32 | A0F-cross-state-v01 | outputs/a0_fr_audit/cross_state_generalization_v01 | 32 态泛化 | C 胜 B 96/128（0 负） |
| 33 | **A0F2-cross-partition-v01** | outputs/a0_fr_audit/cross_partition_extension_v01 | 56 态普查 | C 胜 B 168/224（0 负，7/7 stage 正） |

## D. A0 效用与成本链（09-14 → 09-16）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 34 | **A0G-decision-utility-v01** | outputs/fr_decision_utility_v01 | C 选动作有真效用？划算？ | 效用成立（Gain 1.54e-4，Pearson 0.905）/ **成本不成立**（speedup<1） |
| 35 | A0H-0-offline-audit | outputs/a0h_adjoint_cost_v01/offline_audit | 伴随可行性离线审计 | 18/18 PASS |
| 36 | A0H-1-adjoint-dev | outputs/a0h_adjoint_cost_v01/dev_check | 伴随=转置？ | PARTIAL（H8 逐元被 bf16 地板阻塞；决策级全过） |
| 37 | A0H15-round1 | outputs/a0h15_vjp_shared_v01 | 共享 VJP 伴随 | 建立成立；验收批降级 DEV |
| 38 | A0H15-round2 | 同上（ledger/production/accept_aprime） | 台账+生产入口+计时+A′ | 计时 R_replay=1.17；**A′ FAIL（后证实为驱动 bug）** |
| 39 | A0H15-DSA | …/decision_semantics_audit | A′ 失败根因 | Branch A：selector 方向写反；Branch D 不成立 |
| 40 | **A0H20-corrected-accept** | outputs/a0h_corrected_accept_v01 | 修复后独立效用 pilot | **POSITIVE_POINT_ESTIMATE**（+1.467e-4，5/6 seed 正） |
| 41 | A0I-A-offline-diagnostics | outputs/a0i_directional_scoring_v01/a0i_a | 增益来源 | ADAPTIVE_CHOICES_OBSERVED / ADVANTAGE_UNCERTAIN |
| 42 | A0I-B-directional-scoring | …/a0i_b | 标量评分降本 | CASE B（子阶段−30%，总 SIMILAR） |
| 43 | A0J-cached-future-sensitivity | outputs/a0j_cached_future_sensitivity_v01 | 缓存 p_τ 复用 | **TIME_SCALE_NEGATIVE**（margin 差 2–3 个量级） |

## E. A0 闭环控制链（09-17）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 44 | **A0K-closed-loop** | outputs/a0k_fresh_response_closed_loop_v01 | 闭环净收益 | CASE A（+7.44e-4）但 M5 选择器不优于固定（−1.29e-4） |
| 45 | A0L-mixture | outputs/a0l_response_mixture_feasibility_v01 | 连续混合控制 | CASE B（固定 j1 ≥ controller） |
| 46 | A0M-temporal-structure | outputs/a0m_response_temporal_structure_v01 | 响应相位结构（离线） | CASE A（共模 8–10× 衰减；argmin 相位依赖 8/9） |
| 47 | A0N-phase-scheduler | outputs/a0n_phase_scheduler_pilot_v01 | ρ 门控策略 | CASE B（B≥C；结构存在≠可利用） |

## F. A0 optimizer 输入链（09-18）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 48 | **A0O-lr-response** | outputs/a0o_lr_response_validation_v01 | LR 响应存在可预测 | **Q1/Q2/Q3 全 YES**（Pearson 0.9552） |
| 49 | **A0P-lr-temporal-kernel** | outputs/a0p_lr_temporal_kernel_v01 | 稳定时间核？ | **GATE FAIL**（无稳定衰减律；长记忆噪声尾） |
| 50 | A0Q-response-control-map | outputs/a0q_response_control_map_v01 | J=[LR,clip,wd] | LR 唯一稳定；clip 事件驱动；wd 可忽略 |
| 51 | A0S-controllability | outputs/a0s_controllability_identification_v01 | |χ| 预测干预增益 | PASS（Spearman 0.762，LOSO 泛化） |
| 52 | A0T-lr-control-pilot | outputs/a0t_response_lr_control_pilot_v01 | sign 控制器 | PASS（+4.41e-4，35/36；C≡B 别名 34/36） |
| 53 | A0T-extra-identity | outputs/a0t_extra_response_controller_identity_v01 | 控制器身份 | Case B=全局 LR 放大；状态依赖反转不持久 |
| 54 | A0U-schedule-discovery | outputs/a0u_response_schedule_discovery_v01 | 离线 schedule 发现 | Case B≈固定 ×1.05 延迟 6 步；信息量≈1 bit |

## G. 泛化 G 线（09-16 → 09-19）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 55 | **G1a-26M-scale** | outputs/generalization_g1a_26m_v01 | 5.354× 规模复制 | **STRONG_SCALE_REPLICATION**（18/18） |
| 56 | **G1b-100M-scale** | outputs/generalization_g1b_100m_v01 | 20.2× 规模复制 | **STRONG_SCALE_REPLICATION**（18/18） |
| 57 | G1c-external-1b | outputs/generalization_g1c_external_1b_v01 | 外部 1B | BLOCKED（初判，后被复核推翻） |
| 58 | G1c-checkpoint-recheck | outputs/generalization_g1c_checkpoint_recheck_v01 | 直链复核 | AVAILABILITY_RECHECK_PASSED |
| 59 | G1c-feasibility-1b | outputs/generalization_g1c_feasibility_v01 | 1B feasibility gate | Phase0/1 PASS；Phase2 中止：**C 分支失败（scale-dependent）** |
| 60 | **G2-data-distribution** | outputs/generalization_g2_data_distribution_v01 | 换语料 | **STRONG_DATA_REPLICATION**（18/18） |
| 61 | **G3-clipping-intervention** | outputs/generalization_g3_clipping_intervention_v01 | 去 clip 干预 | **MECHANISM_CONFIRMED**（gain_clip 0.094→0.000） |

## H. B0/C0 机制线（09-20 → 09-21）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 62 | B0-response-mechanisms | outputs/fr_b0_response_mechanisms_v01 | 脉冲释放/中心轨迹/二阶 | H1/H2 SUPPORTED；H3 UNRESOLVED |
| 63 | B05-mechanism-id | outputs/fr_b05_mechanism_identification_v01 | 载体=θ位移 vs m/v 记忆 | θ 位移 SUPPORTED；记忆核 ~12 步衰减 |
| 64 | B06-regime-dynamics | outputs/fr_b06_regime_dynamics_v01 | regime 切换 | Q1 NOT_SUPPORTED（headline NOT_TESTED） |
| 65 | C0-geometry-tracking-abort | outputs/fr_c0_geometry_tracking_validation_v01 | C0 首试 | SOURCE_NOT_FOUND（中止） |
| 66 | C0-gttr-validation | outputs/fr_c0_gttr_validation_v01 | 一步几何预测 | field NOT_SUPPORTED_IN_PILOT；shadow UNRESOLVED |
| 67 | C0X-offline-explanation | outputs/fr_c0x_offline_explanation_v01 | 规则复算 | **ERRATUM_REQUIRED**（聚合门被实现成逐点门） |
| 68 | **C1-gttr-probe-validation** | outputs/fr_c1_gttr_probe_validation_v01 | 跨库独立验证 | field LOCAL_SUPPORT；shadow UNRESOLVED |
| 69 | C1L-late-completion | outputs/fr_c1_late_completion_v01 | 补 t48 anchors | late 4/4 RESOLVED；COMBINED6 field+shadow PASS |
| 70 | C1L-fieldcheck-errata | …/FIELD_CORRECTIONS.md | 表格标签勘误 | E1–E9（主判定不变） |

## I. GTTR 线（09-22 → 09-26）

| # | experiment_id | 位置 | 问题 | verdict |
|---|---|---|---|---|
| 71 | GTTR-model-target-alignment | outputs/fr_gttr_model_target_alignment_audit_v01 | 声明对齐审计 | MISALIGNED_FOR_DEPLOYMENT_CLAIMS |
| 72 | GTTR-PhaseA-A0A1 | outputs/fr_gttr_phase_a_native_calibration_v01 | 目标校准 | COMPLETE（仅校准；未进 B/C/D/E） |
| 73 | GTTR-PhaseB-action-scoring-v3 | outputs/fr_phase_b_gttr_action_scoring_v03 | 未执行动作判别 | **S3 门终止 LOCAL_OPERATOR_NOT_CALIBRATED** |
| 74 | GTTR-PhaseB-FD-diagnostic | outputs/fr_phase_b_fd_diagnostic_v01 | S3 失败定点 | 非实现 bug（位级可复现） |
| 75 | GTTR-D1-direct-readout | outputs/fr_gttr_d1_direct_readout_v01 | 直接读出干预 | **NO_OBSERVED_GAIN** |
| 76 | GTTR-R1-objective-id | outputs/fr_gttr_r1_objective_identification_v01 | 目标辨识（离线） | 二次型结构辨识完成 |
| 77 | GTTR-R2B-solved-update-bridge | outputs/fr_gttr_r2b_solved_update_bridge_v01 | 已解首步改动 | 一步有利（9/12）但 **终态 NLL 未获益**（8/12） |
| 78 | GTTR-CB1-cost-measurement | outputs/fr_gttr_cb1_cost_measurement_v01 | 成本/进展指标 | 代数 PASS；rho 非可靠预测子 |
| 79 | GTTR-F0-terminal-rule-solver | outputs/fr_gttr_f0_terminal_rule_solver_v01 | 256 步终局 LR 规则 | **BLOCKED_AT_CALIBRATION**（SENS 与 FD 差 6 量级） |
| 80 | GTTR-F0-CALFIX | outputs/fr_gttr_f0_calfix_revalidation_v01 | D3/D4 修复复测 | FULL_HORIZON_GRADIENT_UNRESOLVED_AFTER_FIX |
| 81 | GTTR-F0-IC | outputs/fr_gttr_f0_implementation_consistency_v01 | 实现一致性 | 2 缺陷已修；Z16/Z256 对象错配确认 |
| 82 | GTTR-F1-implementation-closure | outputs/fr_gttr_f1_implementation_closure_v01 | 实现闭合 | b 坐标梯度修正确认（−0.311 vs −0.345） |
| 83 | GTTR-F2a-gradient-direction | outputs/fr_gttr_f2a_gradient_direction_v01 | 梯度方向可用？ | **DIRECTION_NOT_CONFIRMED**（最优在 b=+0.02） |
| 84 | GTTR-F2b-second-order | outputs/fr_gttr_f2b_second_order_v01 | 二阶结构 | **二阶预测失败**（曲率全尺度为负；J 最优在 +0.02） |

## J. 事件、治理与辅助（穿插全程）

| # | experiment_id | 位置 | 内容 | verdict |
|---|---|---|---|---|
| 85 | recovery-theta-protocol | recovery_candidates/theta_protocol/ | 源码丢失取证 | RECOVERED-AS-PYC（源永久丢失） |
| 86 | FR_REPORT_AUDIT_BUNDLE-20260912 | FR_REPORT_AUDIT_BUNDLE_20260912/ | A0 系报告事实审计 | PASS + 措辞规则 |
| 87 | REPORT_AUDIT-A0G-docx-20260915 | docs/report_audit/ | A0G 审计报告 | 存在（docx 未机器解析） |
| 88 | docs-project-inventory-20260912 | docs/project_inventory/ | 全项目盘点（含 artifact registry 69KB CSV） | INFRA（尺寸数字已过时） |

**合计：91 个实验/版本条目**（CSV 为 39 列机器可读版；根目录 17 个预备包合并为 1 行、版本链合并计）。
