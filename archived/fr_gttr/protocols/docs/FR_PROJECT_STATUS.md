# FR 项目状态总览（长期维护）

最后更新：2026-09-17（**A0N Response Phase-aware Pilot 完成：CASE B（C≈B，B≥C median）**——验证 A0M phase structure（ρ=|mean(h)|/(std(h)+1e-12)）能否指导训练策略；合同先于 GPU 冻结（ρ>2 no-op / 1≤ρ≤2 注入 argmin / ρ<1 no-op；K=10 事件复用 A0K frozen manifest；seeds 2030–2032）；门禁 S1/S2/POST-MAIN 全 PASS（bank 逐位、ghost 不可见、与 frozen A0K K10 逐位一致）；结果：C vs A median +6.24e-4（2/3 正）、B−C median −6.78e-4（1/3 正）→ B≥C；等墙钟尺度条件成立；phase occupancy 非退化（H/M/L=4/5/2 中位）、ρ(t) 复现 A0M phase structure；判定 **CASE B**：phase scheduling 无超额收益，response 主要发现固定策略（与 A0L B≥C 互证）；GPU 416.2s/1800s 0 crash；制品 outputs/a0n_phase_scheduler_pilot_v01/ 审计包 34037B SHA256 `47ab2b02…e3ab`；停止条件已执行）。此前：2026-09-17（**A0M Response Temporal Structure Analysis 完成：CASE A（phase structure 在幅度与 argmin 身份）**——纯机制分析轮，只用 frozen artifacts（A0K events/step logs/counterfactual replay、A0L events），零训练零 GPU，不改 optimizer/λ/controller、不设计算法；E1 trajectory（rank change median 0.833）/E2 phase clustering（ARI 0.076 低，簇分 early vs mid+late）/E3 stability（cos@100steps 0.629 随 lag 不衰减，秩 Spearman≈0）/E4 utility（single h Pearson −0.866 强对齐，window-averaged w≥2 退化 → w=1 最优）/E5 fixed-j1 原因（j1 argmin 仅 0.306 非全局偏好；stage-dependent optimum 8/9 series）；common-mode 诊断 |mean|/std median 1.85、共模 early→late ~8× 衰减——机制图景：大幅值共模（阶段衰减负 phase structure 主载体）+小幅值差分（argmin phase 依赖 = schedule 机会信号，单次事件秩为近并列噪声）；判定 **CASE A**；制品 outputs/a0m_response_temporal_structure_v01/ （SHA256SUMS 12 项）；停止条件已执行——不设计算法/不改写结论）。此前：2026-09-17（**A0L Response-guided Continuous Mixture Feasibility 完成：CASE B（B≥C 限定）**——验证训练响应从一次性动作选择升级为持续数据混合控制；合同+events manifest 先于 GPU 冻结（pi_0=经验组占比 [1038,973,987,938]/3936，γ_t=1/(std(h)+eps) 冻结、α=0.1、K=50 steps {50,100}；treatment=数据构成：每微步按 π_window 专用 Generator 采样，无梯度注入，optimizer/clip/H=4/λ=0.05 全不动）；pilot 3 seeds(2030–2032)×5 臂（A 原生 AdamW / B fixed-j1 连续混合 / C response 混合 controller + A_wc/B_wc 同 T_C）；身份 gates S1/S2 逐位 PASS；结果（v_eval@128，增益=损失下降）：C vs A median +6.17e-3（2/3 正，2031 −3.02e-2 负如实保留）、B vs A median +4.90e-3、B−C median +7.46e-3（B 不劣于 C，2/3 seed B 优）；排序 2030 C<B<A / 2031 A<B<C / 2032 B<C<A 无跨 seed 一致；等墙钟尺度条件成立（3/3 A_wc/B_wc 完成 128 步）；pi 轨迹仅 2 次更新且移动小；判定 **CASE B**（B≥C：response controller 未优于平凡 fixed-j1，continuous controller 路线不进入；response-discovered schedule 解读受 B≥C 削弱）；工程透明：S2 smoke 捕获 arm B members off-by-one（973<1024），零科学结果时修复、hash note 先记 live_manifest；GPU 351.8s/1800s 0 crash；制品 outputs/a0l_response_mixture_feasibility_v01/ 审计包 55273B SHA256 `ec4d877c…601b`；A0G–A0K 历史制品未动；停止条件已执行——不自动扩展/不改 λ/optimizer）。此前：2026-09-17（**A0K Fresh Response 闭环验证完成：FORMAL_CASE_A_WITH_QUALIFICATIONS**——第一轮真实训练闭环（frozen v0.7 配方，T=128 计划轨迹）；合同+live_manifest（72 events，events sha `8c099f51…`）先于 GPU 冻结；身份 gates 全过（S1 arm A 与 bank step46/52 逐位一致、S2 ghost no-op 轨迹不可见、S3 事件 ~2.85s）；K=20 主×6 seeds + K=10×3 seeds，三臂配对 + A_wc/B_wc 同 T_C 墙钟臂；M1 等步 median +7.44e-4（4/6 正）、M2 等墙钟同值（9/9 基线在 T_C 内完成 128 步——尺度条件）、M5 adaptive vs fixed-j1 median −1.29e-4（C≤B 6/9，收益来自干预非选择）、M4 no-op 率 0%（门从未生效）有害率 0–50%、M3 ROI ~5e-5 nats/s；72 事件离线 argmin 重建全 PASS；形式判定 CASE A 附三项限定（Q2 负/尺度条件/符号弱一致）；工程透明：B_wc 缺陷分析前自查修复重跑 9 臂（hash notes 全记录）；GPU 1693.7s/1800s 0 crash；制品 `outputs/a0k_fresh_response_closed_loop_v01/` 审计包 289629B SHA256 `64b16f97…fd51`；A0G–A0J 历史制品未动；停止条件已执行——不自动开 A0L/λ/block planning/optimizer 改动，详见 §34）。此前：2026-09-16（**G1c checkpoint 可用性复核完成：AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY**——FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01：按官方 docs/Checkpoints.md 修正核验逻辑（directory URL 404 ≠ checkpoint 不存在；旧 release 用 .pt 扩展名），从官方 configs/official-0425/OLMo-2-0425-1B.csv 按 step 冻结 4 个代表 checkpoint（step0 / 380000 / 950000 / 1530000 = 0%/19.92%/49.81%/80.22%），逐文件 curl 探测：4/4 checkpoint 的 config.yaml + train.pt 全读 HTTP 200，model.pt（5.94GB）+ optim.pt（11.88GB，≈2×model 符合 AdamW m+v）HTTP 200 且 1MB range read 验证（PK zip magic、ETag 逐步不同、redirect=0 无替代 host）；判定 FULL_STANDARD_CHECKPOINT_AVAILABLE，上一轮 EXTERNAL_CHECKPOINT_UNAVAILABLE 修正为 AVAILABILITY_RECHECK_PASSED（核验路径修正，非科学改判；HF screening 保留为历史记录）；零完整大文件下载、零 GPU。制品 outputs/generalization_g1c_checkpoint_recheck_v01/（SHA256SUMS 71 项自校验 PASS）。下一步 = G1c 1B feasibility gate，须用户授权、不自动启动，详见 §33）。此前同日：2026-09-16（**G1c External ~1B Stress Test 阻塞：EXTERNAL_CHECKPOINT_UNAVAILABLE**——按冻结协议 B2 核验官方 AllenAI OLMo ~1B 全系列（OLMo-1B 336 branch 抽样 9 + main、OLMo-1B-hf、OLMo-1B-0724-hf、OLMo-1B-0724-unsharded、OLMo-2-0425-1B 268 branch 抽样 4、OLMoE-1B-7B（MoE 排除）、OLMo-7B 抽样 1；olmo-checkpoints.org 404）：全系列公开 checkpoint 均不含 optimizer state / train state，不得补造 → 在任何科学结果查看前停止科学实验；零训练、零重放、零 GPU 运行（实测硬件 2× RTX 5090 32GB，与 A100-40GB 假设不符已登记）。制品 `outputs/generalization_g1c_external_1b_v01/`，审计包 SHA256 `c78dbbfb…854c9`，详见 §32。外部 ~1B 轴未覆盖；下一步是否更换外部家族须用户新授权，不自动执行）。此前同日：2026-09-16（**A0J Future-Sensitivity Reuse 完成**：cached p_tau + fresh G_t 在 age1 即失 full-rank（3/6）、age4 chosen-id CACHE_P 1/6 / STALE_W 0/6，ΔGain_cache−exact 全负（−6.8e-5/−1.5e-4/−4.9e-5）；implementation gate PASS 30/30（曾 FAIL：模块 `_clip` 漏调 `pe._step_once`，探针链定位、修复逐位 0.0）；成本节省真实（cached non-refresh 0.912s vs exact 5.618s，CASE D 排除）；根因=未来敏感度 1–4 步内 decorrelate 过快（margin median 3.8e-5 vs score 误差 ~2e-3）；cosine 0.64–0.96 与翻转共存；判定 §18 CASE B（含 C 元素），缓存路线停止；H6=NOT_TESTED；GPU ≈1641s/1800s，4 次 infra 崩溃重试；制品 `outputs/a0j_cached_future_sensitivity_v01/`，审计包 9067189B SHA256 `21c1a2f9…3e5559a`，详见 §31。下一步须新预注册，不自动启动）。此前同日：2026-09-16（**G2 数据分布泛化完成**：26M 配置逐字段不动、仅预训练语料 D0→D1（D1=MiniMind 官方 pretrain_hq.jsonl @ revision 3a9bbe61，1,413,103 条，与 D0 几乎不相交：13-gram shingle Jaccard 均值 1.6e-7），判定 **STRONG_DATA_REPLICATION**——C pooled R²=0.9999452367、mre 0.0025988312（D0 G1a 为 0.99958/0.00792）；H>1 paired 18/18/0、mean improvement +0.133621（bootstrap 95% PI [+0.067160, +0.251205]、重采样差值为正比例 1.0，非 p 值）；三 seed 均值全正（+0.20627/+0.07919/+0.11541）；机制双口径均正（stream-first gain_curvature +0.17178 / gain_clip +0.03396）；阈值冻结未动。制品 `outputs/generalization_g2_data_distribution_v01/`，审计包 SHA256 `9553ffa5…157a7`，详见 §30。下一步=G1c（~1B 压力测试），须逐项授权、不自动启动）。此前同日：2026-09-16（**G1b ~100M 规模泛化完成**：4.98M→26.68M→100.78M 第二次 scale-only 复制，判定 **STRONG_SCALE_REPLICATION**——C pooled R²=0.99949、H>1 paired 18/18/0、机制 gain_curvature +0.0732 / gain_clip +0.0362 均正、H=1 恒等 gap=0；制品 `outputs/generalization_g1b_100m_v01/`，审计包 SHA256 `6fe9b0f9…a5f85`，详见 §28。下一步=G3 clipping intervention，须用户授权、不自动启动）。此前同日：2026-09-16（纯文档轮：§25 A0H-2.0 最终状态与算法定位、§26 算法瓶颈转移归档；算法路线图（**非预注册协议，不授权实验**）见 `docs/FR_ALGORITHM_ROADMAP_POST_A0H20.md`；本轮未跑 GPU、未重算任何科学结果、未动任何冻结制品。**A0H-2.0 修正验收 pilot（A″）完成**：暴露台账选 6 态×2 流=12 单元、6 seed；合同先于暴露冻结（no-op 合法、argmin predicted ΔL 冻结语义、撤销强制非 no-op 条款）；dev 演练位级一致 → 数值 gate PASS 12/12 → 决策保持同配置/跨配置 12/12 → 主终点 FP32 伴随所选动作 V_eval Gain **+1.467e-4**（bootstrap 95% PI [+8.66e-5, +2.08e-4]，探索性）、per-seed 5/6 正；UTILITY_PILOT_RESULT=**POSITIVE_POINT_ESTIMATE**（符号标签，非"显著优于所有方法"）；GPU 991.1s/1800s。旧 A′ FAIL 保留不改判。详见 §23 与 `outputs/a0h_corrected_accept_v01/`（审计包 SHA256 `41fe8632…1ff7`）。此前：2026-09-15（A0H-1.5 完成并交接：候选共享 VJP 伴随成立，原始 gate H=1/4 PASS、H=8 维持 NUMERICAL_LIMIT 不改判；Branch B FP32-HVP 候选 B1/B2 PASS；第二轮 dev 核账/生产入口/工程计时/A′ 验收完成——A′ 3/3 FAIL 原始结果保留；同日决策语义审计完成：**Branch A 确认**，A′ FAIL 归因于 A′ 驱动 selector 方向错误（raw ĥ argmax 而非冻结 argmin predicted ΔL），A′ 整批降级 dev-investigation，见 §22 与
`outputs/a0h15_vjp_shared_v01/decision_semantics_audit/`）。冻结制品一律只读；新实验开新目录，不得覆盖。运行环境：conda env `fr_minimind`、`CUDA_VISIBLE_DEVICES=0`、`PYTHONPATH=research/src`、torch 2.11.0 / cuda:0 / bf16。本文档给未来 Claude / Codex / ChatGPT 快速恢复上下文；数字一律取自冻结制品，不要凭记忆改写。

## 1. 当前科学问题

在真实训练轨迹（MiniMind 类 transformer，AdamW，bf16 autocast，grad_clip=1.0，accumulation=2）上，量化**参数扰动 q 对未来 H 步 validation-readout 标量 X 的因果响应** chi，并回答：响应中有多大比例可由**局部光滑 tangent（optimizer 状态 + 曲率反馈 + clipping Jacobian）**解释，剩余残差是否需要记忆核（Mori–Zwanzig）机制。科学动机是把响应读数用作**数据加权等训练动作**的效用度量：只有理解响应本身由什么构成，才能判断一个 reweighting 动作的收益来自优化器动力学、曲率，还是未建模的记忆项。当前结论：**H≤8 内完整光滑 tangent（Model C）已把 H>1 平均相对误差压到 0.0219（注册 ε 下）/ 0.0045（ε/4 下），暂不需要记忆核**；v07 residual-FR 审计进一步显示残差 FR 对 B 的回归 R² 仅 0.103，不构成核拟合依据。

## 2. 严格定义

- **state**：v0.7 bank 中一个 (seed, step) 训练快照，含 optimizer state。锁定三态：seed_2037/step_046、seed_2030/step_046、seed_2037/step_082；bank 全量为 8 seed(2030–2037) × 7 step(46–82) = 56 态。
- **q**：每态独立构造的单位方向张量，只从 held-out train-probe 构造，probe/eval 自轨迹第 0 步永久排除。已注册、SHA256 锁定，严禁改动。
- **horizon H**：扰动施加后 forward 的优化器步数，锁定 grid {1,2,4,8}（v0.7 主 endpoint 为 H=8，bank 注册 grid 另含 12/16/24/32）。
- **chi（响应）**：`chi = mean_stream[(X_plus − X_minus)/(2*epsilon_registered)]`，8 条成对 E2 stream 平均；tangent 单位为"每单位 post-unscale 梯度脉冲的导数"。
- **B 类预测**：chi_A_momentum / chi_optimizer_only / chi_BH_fixed_curvature / chi_full_tangent 等模型对同一 chi_fd 的预测值；指标 `_metric`/`relative_error` 唯一定义在 `fr/a0c_adamw_response_propagation_v01.py`，任何新实验必须 import，禁止重定义。

## 3. 实验协议（pulse-before-clip / CRN）

扰动位置（v0.7 prereg 锁定，禁止改动）：`accumulation backward → scaler.unscale_ → ±epsilon*q → clip_grad_norm_ → native AdamW → zero_grad`，每分支只在第一个 optimizer step 施加一次，即 **pulse 在 clip 之前**进入梯度——这是 B/C 初始步必须经 `J_clip(g_t)` 的原因。CRN（common random numbers）：baseline/+/- 三分支从相同完整 Z_t 恢复，共用 batch manifest、RNG stream、LR path，唯一允许差异是 `±epsilon*q`。rho=0.25。replay 约束：bf16 autocast、GradScaler 禁用、真实 microbatch LR、E2 manifest 只读。

## 4. 当前正确理论

- **AdamW augmented state**：局部状态为 (θ, m, v, t)。完整 tangent 步进（`adamw_tangent_step`，冻结于 a0c 模块 L160-276）含 m/v 一阶矩更新、bias correction、eps、decoupled weight decay，以及全局 clip 的 Jacobian。记经 clip-JVP 后的梯度通道为 dc，则 `dm1 = β1·dm + (1−β1)·dc`、`dv1 = β2·dv + 2(1−β2)·c·dc`（c 为 clip 激活时的缩放系数，未激活时为 1）、`dθ ← (1−lr·wd)·dθ − lr·[dm̂/√v̂ − m̂·d√v̂/v̂]`，其中带帽量为 bias-corrected 商。
- **clip JVP**：`J_clip(g)q = c·q − radial·g`，c=min(1/(||g||+1e-6),1)，radial=max_norm·⟨g,q⟩/(||g||·(||g||+1e-6)²)；clip 未激活时 jvp=tangent（恒等）。实现：`clip_jvp`（a0c 模块 L50-136）。由于 pulse-before-clip 协议，初始脉冲必定经过此映射。
- **H1 Jacobian**：H=1 无 future 梯度反馈，响应由局部 AdamW 映射的 Jacobian 完全解释（H1_adamw_jacobian_v01 验证：ε/4 相对误差 0.000146–0.018989，均不劣于全 ε；旧 frozen v 项存在 ν 反常，完整 v 路径修正可忽略）。
- **模型阶梯**（A0C 冻结定义，禁止改动）：
  - **A_momentum** = `-eta*mu_raw*(1-beta1**H)`（历史动量核）；
  - **B** = 完整 AdamW tangent，初始脉冲 δg=q 经 `J_clip(g_t)`，未来步 δg=0；
  - **BH_old**（A0E，**已废弃**）= B+未来 raw HVP，但初始脉冲用裸 q（bug）；
  - **BH_fixed**（A0E2）= 初始脉冲与 B/C 完全相同（`J_clip(g_t)q`），未来步 raw HVP 不做 clip-JVP；
  - **C** = B + 未来 `clip_jvp(H(loss)·δθ)`（完整 tangent）。

## 5. v0.7 最终 locked verdict

v0.7 Confirmatory 已锁定（CONFIRMATORY_PREREGISTRATION_V07.md）：8 个 confirmation seed（2030–2037；2026–2029 仅 discovery），state grid {46,52,58,64,70,76,82}，horizon {1,2,4,8,12,16,24,32}，主 endpoint H=8，E1 R=16 / E2 R=8。bank 位于 `outputs/e3_confirmatory/v07/`（56 态 checkpoint + q + E2 manifest + theta mapping），一切 A0 实验的输入只从此 bank 读取。

## 6. A0C 核心数字（response_propagation_v01，冻结）

3 态 × H{1,2,4,8} × 8 stream = 12 行 + 24 stream。all-12 口径：A mre 6.3636；B R²=0.8388、mre 0.1650；C R²=0.9977、mre 0.0274。H>1（9 行）口径：A 6.7582 / B 0.2053 / C 0.0218656，**C 在 9/9 行 beat B**；逐 H 的 C mre：H2 0.0315、H4 0.0229、H8 0.0112，B 则随 H 发散（0.0903→0.1825→0.3430）。分解：optimizer 机制 gain +6.5530，曲率反馈（B→C）再 +0.1834。A0C 的 A→B 对比同时改变 preconitioner/bias-correction/clip/LR/wd/当前梯度依赖，**不能**解读为"optimizer 记忆"单独贡献。8 条 replay 约束（EXACT_V07_STATE_Q…、NO_FITTED_OR_EMPIRICAL_VARIABLE 等）全部 PASS；A0E2 在同 run 内对 A0C 逐位复现，max gap=0.0。

## 7. H1 核心结果（H1_adamw_jacobian_v01）

H=1 局部 Jacobian 理论成立：ε/4 下三态相对误差 0.000146–0.018989 且逐态不劣于全 ε；确认 frozen v 项 ν 反常（seed2037/step46 旧 v 项 −0.4775 vs 完整局部 v 路径 0.00594，比值 0.0124）。推论：H>1 必须经未来梯度传播（HVP + clip/activation 边界），这正是 A0C→A0E2 链。

## 8. A0D（response_propagation_v02_epsilon，冻结）

对 C 模型做 ε{1,1/2,1/4} 收敛：27 行。池化 mre 0.0218656→0.0089109→0.0045242（ε/4 时 R²=0.99982）。FD 收缩 7/9 cell；H=8 在 ε/4 下 3/3 态改善（0.0111703→0.00646787）。结论：C 的残差主要是**有限 ε 伪影**，H≤8 暂不需要 Mori–Zwanzig 核；5 个候选 cell 建议 ε/8 诊断，但**不得**用当前有限振幅残差拟合记忆项。

## 9. A0E2（curvature_decomposition_v02_bh_fixed，冻结）——旧 A0E BH bug 已废弃

旧 A0E 的 BH 在初始步用裸 q 跳过 `J_clip(g_t)`（runner L324 vs L311-320），H=1 即与 B/C 不一致（seed2037/step46：BH −0.081556 vs B=C −0.056120）。stream 级实证：该态 4/8 条 stream 初始 clip 激活（preclip norm 0.97–18.27），BH_old−B 的 H=1 gap 分别为 −0.068219/−0.008939/−0.063576/−0.062755；clip 未激活的 stream 三者逐位一致。**旧 A0E 的 gain_H<0"raw curvature 有害"结论整体作废**。A0E2 修正后（24 stream 上 max|B−BH_fixed|=max|B−C|=max|BH_fixed−C|=0，精确 0，满足 ≤1e-10 要求）：

- H>1 9 行：B 0.205255 → **BH_fixed 0.136628**（raw curvature 正贡献 gain_raw_curvature_fixed=+0.068627；逐 H：H2 −0.0133 / H4 +0.0231 / H8 +0.1961）→ C 0.0218656（clip-JVP gain +0.114762）。
- 三项 gain 加总=total 恒等成立（max 偏差 8.9e-16）。clip 未激活未来步 BH_fixed 与 C 逐位一致（3/24 全程一致；首次发散步=首次 clip 激活步，24/24）；clip 激活步 gap 一致出现（48/168 步激活）。
- 旧 gain_H=−0.135725 与修正值之差 −0.204352 **完全来自初始化污染**。

## 10. A0F（cross_state_generalization_v01，冻结）

v0.7 bank 随机 32 态 × H{1,2,4,8} = 128 行：C beat B **96/128**（32 平=H=1 零对照，0 负），平均改善 +0.124154，8/8 seed 全正；mre A=4.5230 / B=0.1286 / C=0.004461。

## 11. A0F2（cross_partition_extension_v01，冻结）

全 56 态普查 × H{1,2,4,8} = 224 行：C beat B **168/224**（56 平，0 负），平均改善 +0.121789，7/7 训练 stage 全正（最弱 +0.080713）；mre A=4.3579 / B=0.1277 / C=0.005872。与 A0F 共享 128 行/256 stream 逐位复现（gap=0）。v06 跨 partition 复现**不是 replay 扩展**，需新建 E2/E3 级观测协议，排在下一步之后。

## 12. math-SDPA HVP caveat（永远伴随 C/BH_fixed）

native flash-attention 不支持 double backward，C/BH_fixed 的 HVP 用 **math-SDPA 代理**（`PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA`），是混合后端解析代理而非精确 flash 核 tangent；运行期记录 max_hvp_math_gradient_relative_gap_vs_native=0.0050579、native clip 张量 L2 gap 1.52e-07。解释最小残差时必须带此限定；A0D 显示残差随 ε 收敛，主结论不依赖该代理的精确性。

## 13. theta_protocol.py 源码丢失事件及当前状态

`research/src/fr/theta_protocol.py`（6341 B，SHA256 `4a127209…`，提供 `run_theta_branch` 等 replay 原语）于 **2026-09-12 00:16–19:00 之间被删**（A0F2 完成时仍在；全盘 48 个副本无一匹配冻结哈希，最接近的现存版本为 2026-09-02 的 4359 B 旧版与 VSCode 历史的 4887 B 版，均非冻结版）。处置：以编译自冻结源的字节码恢复为 `research/src/fr/theta_protocol.pyc`（SHA256 `61e1bae3…`，sourceless loader 透明解析），行为等价经 A0E2 baseline-replay 哈希一致性 + A0C 逐位复现（gap=0.0）验证；A0F/A0F2 及全部冻结制品的 input_hashes 仍记录原源文件哈希 `4a127209…`。**所有冻结 runner 依赖该模块；每次跑实验前先确认文件存在。建议排查删除原因。**

## 13b. 冻结与复现规则（所有未来会话必须遵守）

- 冻结制品 = `outputs/a0_fr_audit/<dir>/` 下各 summary/report 及 `research/src/fr/a0*_…`、`research/scripts/a0*_…` 模块：只读，永不编辑、永不重跑。
- 新实验一律新建 `a0X_…` 模块 + runner + 测试，写新输出目录；指标 import 自 a0c 模块。
- 复现校验链：A0E2 同 run 内 A0C gap=0.0、A0E BH_old gap=0.0、A0F2 对 A0F 共享行 gap=0.0（tol 1e-9）；H1 传播对冻结 H1-Jacobian 制品 gap 2.32e-09。
- 运行环境：conda env `fr_minimind`（/home/b205/miniforge3/envs/fr_minimind/bin/python）、`CUDA_VISIBLE_DEVICES=0`、`PYTHONPATH=research/src`、torch 2.11.0 / RTX 5090 / cuda:0 / bf16。

## 14. 绝对禁止再引用的旧结论

1. 旧 A0E「gain_H<0，raw curvature 有害」——BH 初始化 bug 产物，已被 A0E2 推翻。
2. 旧 BH 的任何数值（尤其 H=1 的 −0.081556 vs B/C −0.056120）。
3. 旧 A0E 汇总表"9 行"标题实算 12 行的口径混淆（A0E2 已用显式 scope 修复）。
4. mv_theory 旧 v 路径改善结论（mv_theory_final）——mv_theory_consistent 已证 v 路径不改善（0/8 seed、0/8 horizon）。
5. 任何"theta_protocol.py 源文件还在"的假设（现为 .pyc）。

## 15. 当前最新 next step

A0G（决策效用，见 §17）已接手本节三项：validation-loss readout、real data-reweighting actions、cost benchmark 均已纳入 A0G 单一实验。**A0G 已全部完成并冻结**（科学计算 + 分层统计审计 + cost-only 计时补测 + finalize，`summary.json` status=COMPLETE）：决策效用成立，相对逐候选回放的墙钟成本优势不成立。最新 next step = **v06 跨 partition 复现**（排在 A0G 之后，需新建 E2/E3 级观测协议，非 replay 扩展）。

执行注意：任何新实验须遵守 §13b 规则（新目录、import 冻结指标、输入只取自 v07 bank 与冻结 summary）；cost benchmark 复用 A0C/A0D 既有 replay 与计时框架，不得改动冻结 runner。跨会话恢复一律先读 `docs/FR_A0G_HANDOFF.md`。

## 16. DO NOT REDO

以下均已完成并冻结，**禁止重复运行**：

- v0.7 confirmatory bank / 56 态 checkpoint / q / E2 manifest 生成（含 ε、rho=0.25、CRN 协议）；
- A0C response_propagation_v01（3 态 × H{1,2,4,8}，已被 A0E2 在同 run 内逐位复现 gap=0.0）；
- H1_adamw_jacobian_v01（H=1 局部理论已确立）；
- A0D epsilon 收敛 ε{1,1/2,1/4}（ε/8 诊断属新实验，不在此列）；
- mv_theory_consistent（448 行，v 路径结论已封）；
- A0E curvature_decomposition_v01（**整体被 A0E2 取代，不得重跑或引用**）；
- A0E2 curvature_decomposition_v02_bh_fixed（H1 恒等已达精确 0）；
- A0F / A0F2 跨状态普查（224 行 census 完备，v06 复现需新协议而非重跑）。

## 17. A0G（决策效用，fr_decision_utility_v01）—— 科学已完成，成本计时进行中

**目的**：检验完整切线响应（C）能否选中"真降低未来 validation loss"的数据加权动作，及是否抵得上相对逐候选回放的计算成本。跨会话恢复先读 `docs/FR_A0G_HANDOFF.md`。

**协议（冻结）**：动作 `g_action = g + λ(g_j − g)` 于 post-unscale/pre-clip、单次 optimizer step；V_score/V_eval 分离；formal = seeds 2038–2041 × steps {46,64,82} × 8 streams = 12 态/96 单元；λ primary 0.05 / conservative 0.025（Rule B_decision_signal，clip_crossing≤0.10）；**Rule A（slope_dev≤0.25）在 dev 全候选 FAILED，保留为负结果，不得改写为"严格线性成立"**。readout off-by-one 已修（统一 θ_{t+H} post-step）。

**状态**：主 GPU 计算 100% 完成（96/96，全 shadow_pass=True），四份原始科学 CSV 已固化未动；聚合曾遇 `for (split,h),_ in ...` 字符串解包 bug（已修 + 加单测），由离线脚本 `a0g_postprocess_v01.py` 补齐；cost-only 计时补测（3 态 × 8 streams，独占 GPU）已完成并经 VERIFY（计时副本 vs 冻结 level1 CSV max|Δhhat|=0）；`a0g_finalize_report_v01.py` 已装配，**`summary.json` status=`COMPLETE`**。

**成本结论（§12 计时补测，独占 RTX 5090 / torch 2.11.0 / CUDA 13.0）**：实测组件 median——t_native_step 0.0942s、t_cand_grad 0.0499s、t_hvp 0.168s、t_tangent_B/C 0.241s、回放 wall ±λ ~1.23s。**H=4 M=4：shared-tangent-C total 6.77s vs direct 2.65s（speedup 0.39×）、shared-B 4.74s（0.56×）、direct-fd 4.77s（0.56×）**；H∈{1,4,8} × M∈{1,2,4} 的 shared-tangent_B/C 的 vs_direct_speedup **全部 <1**，break-even 扫描 M≤64 **全为 None**。**准则 5（相对逐候选回放的墙钟效率）不成立**——本模型规模与计时口径下，完整切线的共享传播不比真实回放更快；决策效用成立不依赖成本结论。

**关键科学数字（formal，V_eval）**：Level-1 H=4 C 切线 vs 中心差分 Pearson **0.905**、符号 **0.880**（B 0.660/0.779，alignment 0.102/0.701）。Level-3 H=4（stream n=96）mean Gain：C **1.54e-4** > B 1.31e-4 > alignment 8.05e-5 > random 9.22e-5；mean Regret：C **1.04e-4** < B 1.26e-4 < random 1.65e-4；harmful：C **0.177** < B 0.188 < random 0.229；top-1：C **0.448** > B 0.396；组内 spearman：C **0.366** > B 0.281。分层 bootstrap（10000×）：H=4 C vs random/alignment 的 Gain、Regret P(有利)≥0.96；**H=4 C vs B 方向一致 P≈0.84 但 CI 跨 0（不得称显著）**；**H=8 C vs B 稳健（Gain CI=[1.62e-5,1.38e-4] 全正、P=0.995，Regret P(<0)=0.996）**。

**措辞红线**：✅「完整切线响应的任务层训练决策效用已在独立评价数据上得到支持。」✅可陈述成本否定结果：「在本模型规模与计时口径下，完整切线的共享传播相对逐候选真实回放无墙钟成本优势（speedup<1，无 break-even）。」❌禁止「完整切线已经带来训练效率收益。」

**制品**：`outputs/fr_decision_utility_v01/`（summary.json、四份原始 CSV、statistical_audit/ 5 件、cost_only/ 计时中）；脚本 `a0g_decision_utility_v01.py`（runner+分析）、`a0g_postprocess_v01.py`、`a0g_cost_timing_v01.py`、`a0g_statistical_audit_v01.py`、`a0g_finalize_report_v01.py`；测试 26/26 通过。

> **2026-09-14 注记**：§17 中"成本计时进行中"已成为历史——cost-only 计时已完成并经 VERIFY（见上「状态」）。成本否定结论维持。

## 18. A0H（伴随降成本审计与验证，a0h_adjoint_cost_v01）—— A0H-1 PARTIAL，交接完成

**目的**：在不损失完整切线决策质量的前提下降低响应计算成本——对冻结前向切线的实际线性算子取反向伴随，把"每候选×每步"HVP 换成单次反向行走。合同/容差/清单执行前冻结于 `docs/FR_A0H_PROTOCOL.md`；交付与下一步见 `docs/FR_A0H_HANDOFF.md`。

**A0H-0（离线审计）**：PASS（18 项；Rule B 全局 λ=0.05 确认，冻结制品无漂移）。

**A0H-1 结果（PARTIAL）**：
- CPU 合成验证 PASS：伴随=冻结前向转置（逐通道 FD 1e-6；恒等式 1e-10；4/4 测试）。
- GPU dev 等价（3 态 × stream0 × H∈{1,4,8}，冻结容差 atol=1e-6/rtol=1e-4）：**逐元 FAIL（NUMERICAL_LIMIT）**——H=1 与 B 全精确（≤1.4e-10）、C|4 2/3 态通过、C|8 1/3 通过；**决策级全 PASS**（排序/选择/no-op 全部一致，前向复演近位级：H1/H4 ~5e-11、H8 ~1e-6——**更正**：原"位级 Δ=0"表述过强）。
- 定位（fp32 探针，诊断不放宽 gate）：转置实现精确（fp32-HVP 伴随 vs fp32 参照 max|Δĥ|=4.1e-10@H=8）；失败根因=冻结 bf16 math-SDPA HVP 代理转置不对称（真实方向残差 2.4e-4..7.3e-2），其噪声地板（前向对自身 fp32 参照 H=4 达 1.43e-5、H=8 达 5.78e-5）**超过冻结容差** ⇒ 冻结合同下 **H=8** 逐元 gate 在本实现下不可满足（**更正**：原"H≥4 对任何伴随不可满足"过强且与 C|4 的 PASS 冲突；fp32 探针是强证据但非"任何伴随"的全称证明）。
- 计时：**NOT_RUN**（协议 gate：等价 PASS 前不开始；未报未验证加速）。
- 制品：`outputs/a0h_adjoint_cost_v01/{offline_audit,dev_check}/`；模块 `research/src/fr/a0h_adjoint_v01.py`；runner `research/scripts/a0h_adjoint_dev_check_v01.py`；测试 `research/tests/test_a0h_adjoint_v01.py`（4/4）；审计包 `outputs/a0h_adjoint_cost_v01/audit_package/`（manifest + 单独 SHA256）。

**下一步（未启动）**：A0H-2 唯一候选=预注册"参照法"等价准则（fp32-HVP 前向参照 r，要求 |ĥ_adj−r| ≤ |ĥ_fwd−r| + 冻结容差）+ 通过后按冻结矩阵计时。不得自动开始。


## 19. A0H-1.5（VJP 候选共享伴随，a0h15_vjp_shared_v01）—— 共享伴随已建立，独立验收 NOT_COVERED

**目的**：检验冻结 BF16 前向打分器能否支撑**候选共享**反向打分器（w_0 与候选无关，
不预设数值 Hessian 自伴），在原始数值标准下保持一致并降低成本。合同冻结于
`docs/FR_A0H15_PROTOCOL.md`（+事后附录 A 执行记录）；交付见
`docs/FR_A0H15_HANDOFF.md`。

**结果摘要**：
- **共享伴随获得（PASS）**：w_0 仅由公共基轨迹 + 终点 V_score 梯度导出；
  CPU 非对称 K 合成 5/5。**更正（第二轮 §2）**：原"GPU 同进程位级一致"声明过强
  （探针崩溃未完成比较）；实际跨运行证据 H=1/4 ~5e-11、H=8 ~1e-6；第二轮生产
  入口自校验确立同进程位级相等（gap=0，新证据）。
  CHK-LIN 线性性在 BF16 地板内成立；CHK-SA≈CHK-IVJP 缺陷同源 ⇒ 主导误差为
  BF16 HVP 代理非自伴（"排除所有实现错误"的表述过强，更正为"无伴随实现错误的
  独立证据"）。
- **原始逐元 gate（atol=1e-6/rtol=1e-4，D1）**：H=1 PASS（~5e-11）；H=4 PASS
  （1.36e-6 ≤ 1.84e-6）；H=8 FAIL（与冻结 C|8 同签名，BF16 地板，非回归，不改判）。
  CHK-TELE 未达容差（最差 8.9e-3，如实记 FAIL；按协议仅为一致性诊断）。
- **决策保持（dev 级 PASS）**：BF16 配置 D1 全 H 排序/选择一致；FP32-HVP 配置
  vs 冻结 BF16 C 的 5 键全排序（含 no-op）3 单元 × 3 H = 9/9 完全一致。
- **Branch B（FP32-HVP 不同配置候选）**：B1 PASS（同配置 fwd vs 伴随 max 4.1e-10
  << atol32=1e-5，含 H=8）；B2 PASS（逐元两式并报 + 决策级全一致）；B3 遵守。
- **Branch A 独立验收：NOT_COVERED**——验收集 A（2032/58, 2035/70, 2037/46）在
  调试 accept runner 重建层 bug 后按协议 §1 读一次规则**整批降为开发数据**
  （`outputs/a0h15_vjp_shared_v01/contract/acceptance_disposition_readonce.md`）。
- **§7 计时（A0H-2 小规模）：NOT_COVERED**（触发条件需独立验收集决策匹配；
  GPU 累计 822s/1800s 未耗尽，非资源原因）。

**制品**：`outputs/a0h15_vjp_shared_v01/`（diagnostics/、accept/（DEV_VALIDATION_ONLY）、
contract/）；代码 `research/scripts/a0h15_{vjp_diagnostics,accept,analyze}_v01.py`、
`research/tests/test_a0h15_vjp.py`（5/5）。冻结源哈希 18/18 未变；A0H-1 审计包
SHA256 复核一致。审计包 `a0h15_audit_package.tar.gz`（manifest 不含自身，
SHA256 单独记录）。

**下一步**：A0H-2 需用户先验冻结新验收批 A′（≠D、≠降级批），accept runner 验证后
触发 §7 计时；生产伴随入口已可用（同 pass 产 hhat_fwd_C 与 vjp_shared）。


## 20. A0H-1.5 第二轮（dev 核账 → 生产入口验证 → 工程计时 → A′ 验收）—— 完成；A′ FAIL（诚实记录）

授权：`docs/FR_A0H15_AMENDMENT_DEV_TIMING.md`（DEV_ENGINEERING_TIMING，独立于 A′，
不改变原 §7，不授予效用 PASS）。顺序按用户本轮 mandate：现有 dev 核账与生产入口
验证 → 开发集工程计时 → 条件满足后一次 A′ 独立验收。

- **§1 制品核验**：第一轮审计包 49671 字节 SHA256 cf7ec456… 复核一致（逐文件）。
- **§2 比较台账**：`ledger/comparison_ledger.{csv,json}`（180 行，7 个显式方法对，
  每行两端方法+精度/后端+数值+容差+判定）+ README 更正：原"位级一致"声明撤回
  （探针 `/tmp/test_determinism.py` 崩溃未完成）；H8 gate FAIL 与跨运行 ~1e-6
  一致性无矛盾（不同比较对）；H≥4 不可满足、100% 地板等过强表述已更正。
- **§3 FP64 簿记闭合**：6 个运行（3 单元 × bf16/fp32hvp）SF−SA = EF−EA+R0+ΣRk
  全部闭合（max rel 1.9e-16，EA=0）；明确簿记不证明伴随正确。
- **§4 生产入口**：`research/scripts/a0h15_production_entry_v01.py` 三模式互斥
  （forward_only --hvp_precision bf16|fp32 / adjoint_only --hvp_precision fp32 /
  replay_only --backend native）；发现并修复两个正确性 bug（clip/step 顺序须按
  冻结次序、快照 step_after 误用 micro-step 计数——应为优化器内部 step+1）；
  修复后自校验：FP32 前向/伴随与冻结 pass 同进程**位级相等**（gap=0），同配置
  B1 gate max 1.2e-10；配置错配排练（dtype/hidden 漂移）预读 AssertionError；
  **ENGINEERING_ELIGIBLE=PASS**（`production/engineering_eligibility_d1_h4_m4.json`）。
- **§5 工程计时（DEV_ENGINEERING_TIMING，GPU1 独占实测）**：H=4/M=4 中位墙钟
  fp32_forward 3.403s / fp32_adjoint 1.321s / bf16_forward 3.631s /
  native_replay 1.546s ⇒ R_structure=2.577、R_replay=1.170、R_frozen=2.750；
  分数跨重复逐位一致；扩展矩阵 H{1,4,8}×M{1,2,4} 共 7 组合
  （`production/timing_d1_{h4_m4,matrix_ext}.json`）。
- **§6 A′ 独立验收**：按预注册（`accept_aprime/PREREGISTRATION.md`，先注册后暴露，
  重试规则先冻结；驱动两次基础设施崩溃——序列化/键类型——均非评分逻辑，冻结
  模块哈希全程未变）：3 单元（2030/52、2033/64、2037/82，均 ∉ D ∪ 降级批 A ∪
  本轮 dev）。结果：**L1 B1 同配置 gate 3/3 PASS；BF16 前向 vs FP32 伴随排序
  9/9 完全一致（含完整 4 键序）；L2 ĥ-argmax 行动 realized V_score gain 3/3 为负
  （−2.5e-4/−7.3e-5/−2.8e-4）⇒ 3/3 单元 verdict FAIL（冻结判据，诚实报告，不重试）**。
  注：伴随保真度证据完整（排序零偏差），失败源于响应代理 argmax ≠ 效用 argmax
  （replay 自身 gain-argmax 行动与 ĥ-argmax 不一致），属行动选择启发式性质。
  （**2026-09-15 解释修订**：此归因未经语义核对，状态改为 UNRESOLVED /
  AUDIT REQUIRED，见 §21；本节原始结果与数字全部保留不改判。）
- **§7 收尾**：审计包 `outputs/a0h15_vjp_shared_v01/a0h15_audit_package_round2.tar.gz`
  （manifest 不含自身 + 单独 SHA256 + 可列可解验证）；GPU 累计 ~1270s/1800s；
  冻结源文件哈希未变（A0G 4 CSV、原 runners、λ/Rule B、split、A0H 原 gate 只读）。

**下一步（唯一建议，已被 §21 冻结取代）**：A′ FAIL 定位为决策规则层（响应代理 argmax 与 realized
效用脱钩），而非伴随数值保真度。若继续：下一轮预注册"决策规则改进候选
（如 replay-gain 代理或混合准则）"，在新一批未见单元上重跑同一两层验收；
不改动冻结伴随实现与容差。（**2026-09-15 注**：在决策语义审计（§21）完成前，
该下一步不启动——不得设计 replay-gain/hybrid rule、不得改 λ、不得开新 A′/A″。）


## 21. A0H-1.5 第二轮解释状态修订（2026-09-15）：UNRESOLVED / AUDIT REQUIRED，决策语义审计为唯一优先任务

本节只修订 §20 中 A′ FAIL 的**解释状态**；§20 的工程结果与 A′ 原始执行
结果**全部保留、不改判、不删除**。

- **保持成立的工程结果（DEV_ENGINEERING_TIMING 标记不变）**：H=4/M=4
  FP32 forward ≈3.403 s、FP32 shared adjoint ≈1.321 s、native replay
  ≈1.546 s；同配置 forward/adjoint ≈2.58×，伴随相对 replay ≈1.17×
  （墙钟下降 ≈14.6%）。
- **保持成立的 A′ 原始执行结果**：3/3 单元 L2 verdict FAIL（冻结判据，
  诚实报告）；FP32 前向-伴随同配置数值 gate 3/3 PASS；四个 active
  candidate 排序 9/9 一致；已选动作 realized V_score gain 3/3 为负。
- **解释状态修订为 UNRESOLVED / AUDIT REQUIRED**：当前**不能**据此认定
  response proxy 或 finite-action decision rule 已经科学失效。此前摘要中的
  "ĥ-argmax"表述建立在一个未确认的前提上——hhat 在实际 runner 中究竟代表
  dL/dλ、−dL/dλ，还是经过 utility 变换的量，**尚未核对**。
- **必须核对的完整数据链**：raw hhat → 符号转换 → λ 缩放 → selector →
  action ID → no-op → replay action → V_score/V_eval Gain。
- **"四个 active candidate 排序一致" ≠ 五动作 chosen_id 一致**：前者不含
  j=0 no-op；后续任何报告必须单独给出 **five_action_rank_match**、
  **chosen_id_match**、**choose_noop_match** 三项。
- **当前唯一优先任务**：A0H round-2 **decision semantics audit**（只读
  语义核对，不改算法）。审计完成前冻结：不设计 replay-gain/hybrid rule、
  不改 λ、不开新 A′/A″、不改 AdamW（含 lr/β/ε/clip/动作/loss）。
- **审计结果的两条处置路径**：若确认 selector/sign/ID/no-op 任一环节偏离
  A0G 冻结语义 → 视为**实现语义恢复问题**（修实现对齐冻结语义），而不是
  新算法设计；若语义完全正确 → 才进入 finite-action utility 失效分析
  （届时再议决策规则改进候选）。

  **（2026-09-15 审计完成后更新：走第一条路径——Branch A 确认，见 §22。）**


## 22. A0H round-2 决策语义与五动作效用审计结果（2026-09-15，Branch A 确认）

审计目录（全部离线，无 GPU、无新实验）：
`outputs/a0h15_vjp_shared_v01/decision_semantics_audit/`（含
A0H_ROUND2_DECISION_SEMANTICS_AUDIT.md、five_action_decision_audit.csv、
L2_CONTRACT_AUDIT.md、selector_contract.json、selector_tests.txt 7/7、
SHA256SUMS.txt 等 17 件 + SHA256SUMS）。

**历史解释更正（不删除 §20 原始记录）**：§20 中"A′ FAIL 定位于决策规则层
（响应代理 argmax ≠ 效用 argmax）"的归因**错误**。真实根因是 A′ 验收驱动
自身的 selector 方向错误：

1. **hhat 语义（Case 1 确认）**：ĥ = d L_score(θ_{t+H})/dλ，无符号变换、无
   utility 变换；预测 ΔL = λ·ĥ，预测 gain = −λ·ĥ（源码定位见
   selector_contract.json）。
2. **冻结 A0G selector = argmin_j predicted ΔL（j=0..4，no-op 在内预测 0.0，
   tie 取最小 j）**（`fr/a0g_decision_utility_v01.py:225-229`）；A0G formal
   冻结数据交叉验证 360/360 单元 j_hat == argmin pred_delta_C（6/360 选 no-op）。
3. **预修复 A′ 驱动用了 raw ĥ 的 argmax（j=1..4，no-op 不进排序）**——即选
   预测 ΔL 最大的最差动作。预注册文本（PREREGISTRATION.md）的 "argmax-ĥ"
   条款本身就写错了方向，实现忠实执行了错误文本（故非 Branch B 的纯报告笔误）。
4. **ID/no-op/标签/序列化映射全部无误**（36/36 恒等式：Gain=L_base−L_action、
   no-op Gain=0、oracle≥0、Regret≥0；顺序不变性与 tie-break 单测通过）——
   非 Branch C。
5. **离线纠正重评分（已有五动作真实标签）**：冻结规则 3/3 选中 replay
   gain-argmax 最优动作，gain 全正（2030/52 j1 +8.91e-4；2033/64 j4 +3.58e-4；
   2037/82 j1 +2.35e-4）——**Branch D（finite-action utility 失效）不成立**。
6. **处置**：原 A′ 3/3 FAIL 记录保留；按预注册冻结重试规则 2，已执行 A′
   降级 dev-investigation（既非有效 FAIL 亦非 PASS），**本轮无 A″**；驱动已
   修复（`frozen_decision()` = 冻结 `select_action`；预修复哈希存档），新增
   单测 7/7。"9/9 排序一致"重新解释：仅覆盖 j=1..4，且对任意单调方向都成立，
   不能论证规则方向（此前误用点）；五动作 chosen_id 在冻结规则下 BF16 前向
   vs FP32 伴随 3/3 一致。
7. **计时结论不变**（DEV_ENGINEERING_TIMING）：R_structure=2.577、
   R_replay=1.170（speedup）/time reduction=14.55% 分别陈述；未重跑 GPU。
8. **L2 判据未改**：数值标准（所选动作 V_score gain>0）保持原样，未放宽
   未收紧（详见 L2_CONTRACT_AUDIT.md）。
9. **唯一下一步**：用户先验冻结新验收批（∉D、∉降级批 A、∉A′ 三单元、
   ∉本轮 dev 单元），修复后驱动跑同一两层验收；冻结语义/λ/容差/runner
   一律不动。

## 23. A0H-2.0 修正验收 pilot（A″，2026-09-16）—— COMPLETE，效用点估计为正

按 §22 唯一下一步执行：暴露台账（56 态 bank 排除 9 态：D 3、降级 A 3、A′ 3
→ 47 合格；A0C–F mechanism-only 暴露记为历史不自动排除，宁缺勿补）→
按 (seed, step, ckpt) 轮转选 **6 态（2030–2035 step 46）× 2 流 = 12 单元，
6 个训练 seed**。合同 `outputs/a0h_corrected_accept_v01/contract/contract.json`
在任何 A″ 暴露**之前**冻结（修正条款：j=0 no-op=继续训练、j>0 单步
g+λ(g_j−g)、λ=0.05、H=4、ĥ=dL/dλ、selected_id=min(predicted ΔL, id) 精确
tie；撤销「必须选非 no-op 且每单元 Gain>0」——旧原文保留于
`accept_aprime/PREREGISTRATION.md` 与 `L2_CONTRACT_AUDIT.md`）。
dev-fixed 动作在暴露前由旧 A′ 三状态 replay V_score 增益冻结为 **j=1**。

**执行**：dev 全链演练（2031/52 s0，双标签分支 vs 冻结 replay 位级一致
gap=0，gate 自校 PASS）→ P1 12 单元三 scorer 预测（production-entry 模式
H=4，暴露前成本决定）→ P2 同配置 FP32 gate **PASS 12/12** → P3 12 单元
五动作双标签 replay（分数标签先锁定，eval 哈希后写分文件）→ P4 分析。
GPU 累计 **991.1s/1800s**，无中断无重试，未杀外部任务。

**结果（五字段分离，无合成 PASS）**：
- COMPLETION_STATUS=**COMPLETE**；PROTOCOL_VALID=**PASS**；
  NUMERICAL_GATE=**PASS**；DECISION_PRESERVATION=同配置与跨配置（BF16 C vs
  FP32 伴随）**12/12** 五动作排序/chosen_id/no-op 位一致；
  UTILITY_PILOT_RESULT=**POSITIVE_POINT_ESTIMATE**。
- 主终点（FP32 伴随所选动作 V_eval Gain，态内两流→6 态等权）：
  **+1.467e-4**，bootstrap 95% PI [+8.66e-5, +2.08e-4]（探索性，非 p 值）；
  per-seed 5/6 正（2032 −1.84e-5 为唯一负 seed）。
- harmful_rate=1/12、no-op 比例=1/12（合法 Gain=0）、Regret mean 8.78e-5、
  oracle 精确命中 4/12；与 no-op/随机期望配对差异区间不跨 0，与
  dev-fixed/oracle 跨 0；与 BF16 C/FP32 前向 12/12 同选（差异恒 0）。
- 解释红线：POSITIVE_POINT_ESTIMATE 仅为预注册主终点均值的符号标签；
  不作「λ 过大/非线性/代理失效」归因（无幅度/原生导数探针证据）；
  不声称比所有方法显著优、不声称部署性能、不改判旧 A′。
- 制品：`outputs/a0h_corrected_accept_v01/`（合同/台账/manifest、预测、
  gate、双标签、analysis、UTILITY_PILOT_REPORT.md、审计包
  `a0h20_corrected_accept_audit.tar.gz` 46844B SHA256
  `41fe8632…1ff7`）；新代码 `research/scripts/a0h20_{contract,manifest,corrected_accept,analyze,audit_pack}_v01.py`
  + 单测 12/12（含 eval 隔离接口测试）。本轮到此停止；任何扩展（更多
  seed/状态、H=8 效用、λ 扫描、hybrid）均为需另行预注册的新实验。

## 24. G1a — 26M 模型规模泛化验证（2026-09-16）—— COMPLETE，STRONG_SCALE_REPLICATION

**唯一科学问题**：模型参数 4.98M→26.68M（scale-only：只改 hidden 256→512 /
layers 4→7，数据 D0/tokenizer/seq/batch/optimizer 族与全部超参/clip 1.0/bf16/
训练代码/因果响应协议全部不变）时，(1) 完整切线 C 是否仍预测真实 FD 响应；
(2) C 是否仍系统性优于 optimizer-only B；(3) optimizer state→curvature→
clipping 机制阶梯是否仍存在。

**配置与训练**：实测 unique params **26,681,728**（5.354×；预注册估值
26,682,048 在开任何结果前按实测更正，模型尺寸选择不变）；heads 8/kv 4
（head_dim=64），intermediate=1664（冻结默认公式）。seeds 2042/2043/2044；
EARLY=step26（20.31%）/MIDDLE=64（50%）/LATE=102（79.69%），训练在 102
终止；9 态 × 4 future streams = 36 条 baseline 流，36 份 manifest 全部先于
任何 E2 分支写入 future_stream_manifest.json；H 官方网格 {1,4,8}（冻结
runner 的 H=2 副产物不进统计）。

**主结果（27 正式单元）**：pooled C R²=**0.99958**、corr=0.99985、
MAE=0.001055、mean rel err 0.00792（median 0.00232）；pooled B R²=0.83204、
MAE=0.021645、mean rel err 0.12033。H>1 paired（18 行）：**C 18 胜 / 0 平 /
0 负**，mean paired improvement +0.16861；per-seed +0.0732/+0.1112/+0.3214
（3/3 正）；per-stage EARLY +0.1493 / MIDDLE +0.2966 / LATE +0.0599（3/3 正）。
H=8：C mean rel err 0.00799 vs B 0.23664，C R²=0.99931、corr>0。

**机制子实验（seed 2042，B→BH_fixed→C，FD 参考）**：H=1 恒等 max gap=0；
pooled（3 stage × H>1）gain_curvature=**+0.05738**、gain_clip=**+0.01472**
（均为正；对照 5M anchor A0E2 的 +0.06863/+0.11476——curvature 段存活，
clip 段变小但方向一致；未预设逐单元为正，实际 6/6 单元两项 gain 均正）。

**动力学上下文（仅解释，未筛状态）**：26M 训练期 clip 激活率 1.0（每步
preclip norm mean 1.391），future 第一步 clip 激活 36/36；5M anchor
future 第一步 clip 激活 0.25、preclip norm mean 2.154。26M 下 clip 几乎
恒激活，机制仍在 C 的 clip-JVP 段体现（gain_clip>0 但小于 anchor）。

**工程完整性**：smoke gate（2042/64）14/14 PASS（H=1 analytic vs FD rel
err 3.48e-4）；baseline replay 36/36 逐位精确（shadow hash + X gap=0）；
rho0-noop 36/36 通过；max native-clip tensor L2 gap 2.09e-7；HVP math-vs-
native gap 6.46e-3（mixed-backend caveat 原样保留）；无 OOM、无杀外部任务
（GPU0 共享使用，GPU1 外部任务未动）。实现级修复全部记录在
`outputs/generalization_g1a_26m_v01/CHANGE_LOG.md`（含参数计数更正；
未改任何科学定义/阈值/seed/grid）。

**判定（冻结三级阈值，G1A_PROTOCOL.md §10）**：**STRONG_SCALE_REPLICATION**
（cond1–cond4 全过：C pooled R²≥0.90；H>1 胜率 18/18≥80%；H=8 paired+
R²≥0.50+corr>0；3/3 seed 为正）。

**范围声明（红线）**：G1a 只属于模型规模泛化；**不能替代 A0F2**（5M 56 态
普查）；**不能自动证明 100M/1B 成立**。**未启动 G1b（~100M）**——是否进入
G1b 由用户决定。

**制品**：`outputs/generalization_g1a_26m_v01/`（G1A_PROTOCOL.md、
experiment_config.json、training/checkpoint/future_stream_manifest、
theta_mapping_raw ×3、per_stream/per_state_horizon/mechanism_decomposition、
summary.json、G1A_26M_SCALE_GENERALIZATION_REPORT.md、smoke_report.json、
ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES.txt、SHA256SUMS.txt、
CHANGE_LOG.md、COMMANDS.log）；审计包 `g1a_26m_audit_package.tar.gz`
（254KB，不含 bank/ 与 9 个 q 张量——大文件仅在 SHA256SUMS 记录哈希）
SHA256 `7b36669c…f92b0`。代码：`research/src/fr/g1a_26m_scale_generalization_v01.py`
+ `research/scripts/g1a_26m_scale_generalization_v01.py` + 单测 21/21。

### 24.1 G1a final frozen result —— 归档核实与解释边界（2026-09-16 补充，仅文档轮）

本节为冻结后的纯文档归档补充，不改动 §24 任何数字。归档时按 mandate 重新从
冻结制品逐一核读（`results/summary.json`、`results/per_state_horizon.csv` 27 行、
`results/mechanism_decomposition.csv`、`G1A_26M_SCALE_GENERALIZATION_REPORT.md`、
`G1A_PROTOCOL.md`、`experiment_config.json`、training/checkpoint/future_stream 三
manifest、`SHA256SUMS.txt`），全部数字与 §24 一致；10 个关键文件磁盘 SHA256 与
SHA256SUMS Section 1 逐条 MATCH；审计包重算 SHA256 =
`7b36669c0e1b323bc44a9f52ad0355c1e85cdd22003d3b811be53378323f92b0`（254KB、
251 members），与冻结记录一致。数字来源是制品本身，不是聊天摘要。

**解释边界（冻结，写作与引用时必须保留）**：

1. G1a 证明的是 **5.354×（4,983,168→26,681,728 unique params）规模复制到
   26M** 这一件事；**不证明 100M/1B**，也不替代 A0F2 的 5M 56 态普查。
2. **mixed-backend HVP caveat 原样保留**：C/BH_fixed 用 math-SDPA
   reverse-over-reverse 代理（native flash-SDPA 无 double backward），
   math-vs-native gap = 6.46e-3；与 A0C/A0E2 同一冻结 caveat。
3. **chi_FD 是注册有限 ε 中央差分**（每态 ε = 0.25·σ_q，σ_q 来自 16 条 E1
   流；各态 ε 介于 0.0091–0.2064）；**C 是零脉冲局部光滑切线**。二者之间的
   有限 ε 解释 caveat 与 A0C 相同（A0D 已表明残差主要为有限 ε 伪影），不得
   把 C 读成对任意脉冲幅度成立的非线性响应。
4. **clip gain 比 5M 明显缩小（+0.01472 vs anchor +0.11476）是观察结果，
   不是已确认的机制定律**；其成因未定（26M clip 近恒激活只是登记的相关
   上下文），禁止单独据此推出确定性机制结论。
5. **18/18（H>1 paired 全胜）是 26M、clip=1.0、本协议下的冻结经验结果，
   不是未来规模必须 100% 胜出的理论要求**；不得把 win-fraction=1 写成
   任何后继实验（含 G1b）的通过判据或先验保证。

**路由**：泛化阶段总状态见 `docs/FR_GENERALIZATION_STATUS.md`（新建）；
下一阶段交接（G1b 规划与禁改清单）见 `docs/FR_GENERALIZATION_HANDOFF.md`
（新建）。两者均只含计划与解释，**本轮未启动任何新实验**。

## 25. A0H-2.0 最终状态与算法定位（2026-09-16 文档轮，交接冻结）

本节按交接 mandate 把 A0H20 的最终状态**显式**归档（数字逐一取自冻结制品
`outputs/a0h_corrected_accept_v01/analysis/analysis.json`、`summary.json`、
`gate.json`、`UTILITY_PILOT_REPORT.md`，非凭记忆）。§23 的执行记录保留不动，
本节是其交接用固化视图。

**批次身份**：A0H20 = 选择语义修复（§22 Branch A）后的**预注册独立短期任务效用
pilot**（A″）；**6 个训练 seed（2030–2035）× 每 seed 1 个状态（optimizer
step 46）× 2 条预登记流 = 12 单元**；H=4、λ=0.05（Rule B 全局冻结）；
**5 动作选择器含 j=0 no-op（= 继续训练，合法动作，预测 ΔL=0）**，
j=1..4 = 单步 `g+λ(g_j−g)` post-unscale/pre-clip 注入 h=1。dev-fixed
对照动作暴露前冻结为 j=1（旧 A′ 三态 replay V_score 增益等权均值 argmax）。

**冻结 scorer 与选择语义**：ĥ = dL_score(θ_{t+H})/dλ（Case 1，无符号变换、
无 utility 变换）；predicted ΔL_j = λ·ĥ_j；**selection =
argmin_j(λ·ĥ_j, action_id)，精确 tie 取最小 ID**——公共正 λ 是单调缩放，
不改变排序（单测覆盖：argmin predicted ΔL ≡ argmax predicted gain、
common-positive-λ order preservation）。冻结实现：
`fr/a0g_decision_utility_v01.py:225-229` `select_action`，A0H20 驱动经
`frozen_decision()` 调用。

**执行链（暴露前冻结 → 暴露后分层）**：合同+暴露台账+acceptance manifest
先于任何 A″ GPU 暴露冻结（contract SHA256 `b692ede3…46b46`，人机双向
hash 互验）→ dev 全链演练（双标签 replay vs 冻结 entry replay 位级一致
gap=0）→ P1 预测 → P2 gate → P3 五动作双标签 replay（score 先锁定、
eval 分文件哈希后写）→ P4 分析。GPU 991.1s/1800s。

**结果（五状态字段分离，无合成 PASS）**：
- **NUMERICAL_GATE=PASS 12/12**：同配置 FP32 前向 vs FP32 共享伴随
  （冻结 atol32=1e-5 / rtol32=1e-4，H=4 全部 4 非零动作）。
- **DECISION_PRESERVATION=12/12 双向**：同配置五动作排序/chosen-id/no-op
  位 12/12 一致；BF16 C vs FP32 伴随 chosen-id 12/12 一致。
- **标签完整性**：五动作 V_score 标签 12 单元×5 完整、五动作 V_eval 标签
  12 单元×5 完整（PROTOCOL_VALID=PASS 的组成部分）。
- **UTILITY_PILOT_RESULT=POSITIVE_POINT_ESTIMATE**：FP32 伴随所选动作
  V_eval mean Gain = **+1.467e-4**（态内两流→6 态等权；分层配对
  bootstrap 10000×/rng 260916 的 95% PI [+8.66e-5, +2.08e-4]，
  **探索性不确定性区间**）；**per-seed 5/6 正**（2032 −1.84e-5 唯一负
  seed）；**harmful 1/12**（2032_046_s1，如实计入）；**no-op 1/12**
  （2033_046_s0 合法选择，Gain=0）；**mean Regret 8.78e-5**；**exact
  V_eval oracle hit 4/12**。
- **配对差异（fp32adj − 基线，同聚合）**：vs no-op **+1.467e-4**（PI 不跨
  0）；vs uniform-random 期望 **≈+8.3e-5**（PI 不跨 0）；vs dev-fixed
  (j=1) **≈+5.8e-5**（PI 跨 0）；vs V_score finite-replay selector
  （oracle）**≈−2.3e-5**（PI 跨 0，oracle 按 V_score 最大化选取不读
  eval，负差符合预期）；vs BF16 C / FP32 前向 = 0（12/12 同选）。

**解释红线（写作与引用必须保留）**：不把 bootstrap favorable result 写成
frequentist p-value（流数≠独立 seed 数，仅 6 seed；区间为探索性）；
不把本 pilot 写成跨阶段、闭环、部署或大模型泛化证据（one-step、
conditional lookahead、5M 模型、step-46 状态）；不作 λ/非线性/代理失效
归因（无幅度/原生导数探针证据）；不声称显著优于所有方法；不改判旧 A′
（3/3 FAIL 原始记录保留，其降级 dev-investigation 身份不变）。

**算法定位（冻结表述）**：A0H20 已使"共享伴随是否能够保留 forward
response controller 的选择并获得独立短期任务收益"得到**正向原型证据**。
当前方法应定位为 **one-step response-guided training-action
scorer/controller prototype**，而不是已经验证的完整训练算法。

**制品**：`outputs/a0h_corrected_accept_v01/`（合同/台账/manifest、
predictions/、gate.json、labels/{score,eval}/、analysis/、
UTILITY_PILOT_REPORT.md、COMPLETION_STATUS.json）；审计包
`a0h20_corrected_accept_audit.tar.gz` **46844 字节**、52 条目、SHA256
`41fe86326e568b223d16d9b146b9523dd1e5a19c05f5e88762f17a9be6251ff7`
（可列可解已验证）；manifest 51 文件逐文件 SHA256 自验通过。

## 26. 算法瓶颈转移（A0H-2.0 之后，2026-09-16）

A0H 线的主问题已经转移：**"adjoint 是否算对"已闭合**（A0H-1 fp32 探针
精确到 4.1e-10；A0H-1.5 共享伴随成立；A0H20 同配置 FP32 gate PASS
12/12 + 跨精度决策保持 12/12）——**不再把压低 forward-adjoint
numerical error 作为主线**。当前主线瓶颈（按优先级）：

1. **状态自适应控制是否稳定优于 fixed action**：A0H20 中 vs dev-fixed
   (j=1) 仅 ≈+5.8e-5 且区间跨 0；自适应选择的增量价值尚未确立。
2. **共享伴随之外每个候选完整 gradient 的边际评分成本是否成为瓶颈**：
   共享伴随只省 HVP/传播部分，每个候选仍需完整 `g_j` 反向；候选评分
   的边际成本结构（fixed vs marginal）尚未分解。
3. **真实候选池扩大后的 cost–utility scaling**：M=4 的 1.17× speedup
   （DEV_ENGINEERING_TIMING）不能外推；线性成本模型下 speedup 随 M
   的走向取决于 (da−bc) 符号，未识别（见路线图 D 节）。
4. **单次局部 Gain 能否在 repeated/closed-loop training 中转化为净训练
   收益**：A0H20 是 one-shot 条件前瞻；重复注入的非线性累积、
   state 分布漂移、与 optimizer 动力学的耦合全部未验证。
5. **equal-step 与 equal-wall-clock 条件下是否优于直接训练**：任何
   controller 声明训练效率前必须同时报两种口径（A0G 成本否定结论
   仍适用于 shared-tangent 前向路径）。

**路线记录（仅顺序，非授权）**：A0I-A（existing-label offline
adaptivity diagnostics）→ A0I-B（scalar directional scoring 在现有 dev
上的可行性，M=4/H=4）→ A0J（预定义真实候选扩展 M=4→8，条件性 16/32）
→ A0K（repeated/closed-loop controller，equal-step 与
equal-wall-clock 对比）。当前优先级 **A0I-A → A0I-B**；不是直接 M=32、
不是改 AdamW/λ、不是 hybrid selector。详见
`docs/FR_ALGORITHM_ROADMAP_POST_A0H20.md`（A–H 节）。

## 27. A0I 算法工程轮完成（2026-09-16，A0H20 之后）

A0I = Post-A0H20 Algorithm Efficiency Study（A0I-A 冻结标签离线适应
诊断 + A0I-B 标量方向导数评分可行性）。本轮是算法工程轮，不是新的
独立任务效用验收；A0H20 结论与制品原样冻结未动。

### 27.1 A0I-A 发现（纯离线，无 GPU，无标签变更）

- controller 动作频率 {j0:1, j1:3, j2:5, j3:1, j4:2}，逐种子无单动作
  支配。**Post-A0I 修正**：动作频繁切换只证明 controller 没有退化为
  固定动作，**不能单独证明状态自适应产生净收益**。正确状态标记：
  `ADAPTIVE_CHOICES_OBSERVED` / `ADVANTAGE_OVER_FIXED_UNCERTAIN`。
- DeltaGain(controller − dev-fixed j=1) 配对均值 **+5.76e-5**（正点
  估计可保留，但必须同时注明 seed heterogeneity 与现有不确定性），
  逐种子高度异质：2033 +3.15e-4 / 2035 +2.00e-4 / 2034 +9.73e-5 /
  2030 −9.8e-7 / 2032 −8.62e-5 / 2031 −1.80e-4。事后回看最优固定动作
  = j=2（1.039e-4，POSTHOC_DIAGNOSTIC_ONLY）；controller 相对它
  +4.28e-5。
- Regret 分解重算闭合同格：8.7808e-5 = 2.3002e-5（adjoint−replay）
  + 6.4805e-5（replay−eval-oracle），max |residual| = 0.0。
  **Post-A0I 修正**：第一项 `L_eval(a_adj) − L_eval(a_scoreReplay)`
  **可以为负**；它不是"评分器精度损失上界"，不能解释为"只要提高
  response accuracy 就可消除"。若需要原 score 目标下的选择 regret，
  应另写 `L_score(a_adj) − L_score(a_scoreReplay) ≥ 0`——仅定义，
  本轮不重新开展实验。
- 案例（三类互异）：有害=2032_046_s1（score→eval 迁移失败）；
  no-op=2033_046_s0（全 ĥ>0 正确弃权，且恰为 eval-oracle 单元）；
  max-regret=2031_046_s1（regret 3.42e-4，预测 top3 近并列 ~8e-6）。
- 校准（x=λ·ĥ vs y=ΔL_score）：选中级 12 点 Pearson 0.635/R² 0.403/
  slope 0.605/intercept −8.21e-5；全候选 60 点 Pearson 0.834/slope
  0.971/intercept −5.16e-6/MAE 9.70e-5/R² 0.696；score→eval 迁移
  Pearson 0.429/R² 0.184（仅观测性）。**Post-A0I 修正**：全候选级
  slope≈0.971 **不能单独称"无偏"**——必须结合 intercept、mean error
  与 R² 一起解释；选中级 slope≈0.605 受选择条件（selection
  conditioning）与数值范围影响，不能直接当作整体 predictor
  calibration。

### 27.2 A0I-B 结果（dev-only，3 dev 态 × stream 0，H=4，M=4+no-op）

- 恒等式 `h_j = D_θL_j[w_0] − w_0ᵀg`（w_0 固定切向量）在 math-SDPA
  backend 下数值等价成立。**两个指标必须分开**：numerical gate =
  **54/72**；decision preservation = **18/18**（五动作排序 + chosen-id，
  对 A 与 A_math 双对照）——18/18 chosen-id preservation 不是
  72/72 数值等价。误差参照：same-kernel（math-vs-math）max
  2.29e-5 / median 7.11e-6；cross-kernel（math-JVP vs flash-反向）
  max 2.83e-5 / median 4.22e-6。cross-kernel 略高提示 backend 差异
  有贡献，但本轮**未做逐行分解**，不能声称所有 numerical gate
  failure 完全由 flash-vs-math backend 导致。
- **ambient（flash）SDPA 无 forward-mode AD → UNSUPPORTED_IN_CURRENT
  _BACKEND**（probe 记录 NotImplementedError）；全部 B 数据在 forced
  math backend 下采集。
- 工程性能（每态 5 repeats 中位数，共享 GPU）：scoring A 0.388–0.544s
  vs B 0.295–0.346s（B/A≈0.745，**子阶段更快**）；scoring 峰值显存
  766→634 MB（**改善**）；但总决策成本（B 含 selected-gradient
  recompute ~0.046s）A 1.348–1.624s vs B 1.296–1.475s → **SIMILAR**
  （共享前缀 ~1s 主导）。等效 native steps A 26.6–34.5 / B 25.6–31.3。
- 边际成本 T(M)=a+bM（M=1/2/4 仅计时，LOCAL_AFFINE_DIAGNOSTIC）：
  b：A 0.095–0.144、B 0.067–0.089 s/候选（B 低 ~30%）；a 负值属
  共享 GPU 噪声下 3 点仿合产物，无物理含义；不足以外推 M=8/16/32。
- METHOD C（共享池方向导数）= **NOT_APPLICABLE**（结构性）：候选池
  两两不相交（逐对重叠 0，数值核验），α 分解精确但无可摊销项。
  **Post-A0I 修正**：不相交只排除"重复样本去重"收益；不再写
  "永久不适用"——未来仍可能存在 vector-output JVP、candidate
  batch packing/chunking 等减少调度开销的工程方案（见新 roadmap
  H 节）。
- 七字段分离裁决：EQUIVALENCE=SUPPORTED；DECISION_PRESERVATION=
  18/18；JVP_SPEED=FASTER（子阶段）；SHARED_POOL=NOT_APPLICABLE；
  MEMORY=IMPROVED；TOTAL_DECISION_COST=SIMILAR。决策树定位 **CASE B**：
  数学改写有效、决策保持，但 JVP 在本 backend 下不是总成本层面的加速。
- GPU 预算：记账 ≈749s/1800s（METERED 254.46 + 一次完成全部 GPU
  工作后崩溃于 CSV 写盘的 run ≈254.5 + 两次早期崩溃估计 ~240）；
  2 次脚本级重试，无用户中断，无环境变更，外部任务未动。

### 27.3 制品与红线

- 制品：`outputs/a0i_directional_scoring_v01/`（REPORT.md、
  ALGORITHM_HEALTH_DASHBOARD.json、a0i_a/ 8 文件、a0i_b/ 含
  run_*/ 与 probe/合约/gpu_budget_log、审计包
  `a0i_audit.tar.gz` 47582B SHA256 `2dca15c4…9775`）。
- 红线遵守：未新增 acceptance batch/训练种子/长训练；未碰
  M=8/16/32、hybrid selector、replay-gain 拟合、margin 弃权规则、
  λ、lr action、AdamW、V_eval 规则；未挑 dev 状态；未把暴露的
  A0H20 数据当独立验证；A0G/A0H/A0H20 结论未改；H5=NOT_TESTED。
- 含义：A0I-A 回答了"增益从哪来"（状态自适应，但种子异质）；
  A0I-B 回答了"标量结构能否压边际成本"——**能压 scoring 子阶段
  （b 低 ~30%、显存 -17%），压不动总成本（前缀主导）**。下一步
  （A0J 预冻结真实候选扩展 / backend 缺口 / 闭环）须另行预注册，
  本轮不自动启动。

## 28. G1b — ~100M 模型规模泛化验证（2026-09-16）—— COMPLETE，STRONG_SCALE_REPLICATION

第二次规模复制：4.98M（v07 anchor，A0C/A0E2/A0F2/A0G 冻结）→ 26.68M（G1a，
STRONG_SCALE_REPLICATION 冻结）→ **100,784,000**（本轮）。协议
`FR_G1B_100M_SCALE_GENERALIZATION_V01` 冻结于 2026-09-16T15:35Z（任何结果
打开之前），阈值冻结同 G1a；不重做 A0A–A0G/A0F2/G1a/A0H/A″；未改任何
5M/26M 冻结制品。

**Scale-only 配置**（静态计数先行，结果前冻结）：hidden=768、layers=13、
heads=12、kv=6、head_dim=64、intermediate=2432（冻结默认公式
ceil(768·π/64)·64，非手挑）；实测 unique params = 100,784,000
（20.225× anchor、3.777× G1a），state_dict elements = 105,699,200；
vocab 6400 / seq 128 / tied embeddings；与 G1a 完全一致的项（dataset D0、
tokenizer、objective、AdamW 0.9/0.999/1e-8/0.01、clip 1.0、BF16、CRN、q 语义、
ε=0.25σ_q 语义、post-unscale/pre-clip pulse、FD/B/BH_fixed/C 定义、
mixed-backend HVP caveat）逐项保留。train seeds 2042/2043/2044（paired-scale
design，非新 seed replication）；EARLY=26/MIDDLE=64/LATE=102；128 planned
optimizer steps，1024 tokens/step 与 G1a 相同 token 预算；训练在 102 终止。

**主结果**（27 正式单元，H={1,4,8}，9 态 × 4 流，36 manifest 全先于 E2 冻结）：

- **C pooled（27 行）**：R²=**0.99949**、corr 0.99994、MAE 0.001468、
  mean rel err 0.008760 / median 0.005079。
- **B pooled（27 行）**：R²=0.92725、mean rel err 0.078779。
- **H>1 paired（18 行）**：C beat B **18/18/0**（tie 0，loss 0），
  mean paired improvement **+0.10503**（median +0.08037）；per-seed 3/3 正
  （2042: +0.10932、2043: +0.09568、2044: +0.11008）；per-stage 3/3 正
  （EARLY +0.14787 / MIDDLE +0.07972 / LATE +0.08749）。
- **H=8**：C mean rel err 0.008729 vs B 0.165960；C R²=0.99917、corr>0。
- **机制子实验**（seed 2042 × 3 stage）：H=1 恒等 max|BH_fixed−B| = 0（精确）；
  gain_curvature=**+0.07325**、gain_clip=**+0.03618**（pooled H>1 均值，
  带符号、不预设）；B→BH_fixed→C 阶梯保留。
- **判定**：**STRONG_SCALE_REPLICATION**（冻结四级条件全过：C pooled
  R²≥0.90；H>1 胜率 100%≥80%；H=8 paired + R² 0.99917 + corr>0；
  3/3 seed 正）。

**Clipping regime 三规模并列**（G1a 登记的观察在第三点延续；**不拟合
经验定律、原因未定**，仅作 G3 前置观察）：

| 规模 | future 第一步 clip rate | future 全步 clip rate | training clip rate | preclip norm mean | gain_curvature | gain_clip |
|---|---:|---:|---:|---:|---:|---:|
| 5M anchor | 0.25 | n/a（未记录） | n/a | 2.154 | +0.06863 | +0.11476 |
| 26M G1a | 1.0 | 0.98958 | 1.0 | 1.368 | +0.05738 | +0.01472 |
| 100M G1b | 1.0 | 1.0 | 1.0 | 1.510 | +0.07325 | +0.03618 |

**工程**：train 每 seed 峰值 ≈2.56 GB；smoke（2042/EARLY，mandate 指定态）
13/13 通过、H=1 vs FD 0.0183、GPU peak 11,032 MB、math/native HVP gap
3.06e-3；tangents 36/36 baseline replay 逐位一致；无 OOM、无 retry、无
RESOURCE_LIMIT；GPU0 共享、未杀外部任务。

**解释边界（冻结）**：① 仅证明 ~100M，不能替代 A0F2、不能自动外推
1B；② mixed-backend HVP caveat 原样（math-SDPA reverse-over-reverse 代理）；
③ chi_FD 为注册有限 ε 中央差分、C 为零脉冲切线（A0C 同 caveat）；
④ 18/18 非未来规模理论要求；⑤ clip-regime 表为三规模观察非定律、原因未定。

制品 `outputs/generalization_g1b_100m_v01/`（G1B_PROTOCOL.md、config、三
manifest、smoke_report、results/{summary.json, per_stream.csv,
per_state_horizon.csv, mechanism_decomposition.csv,
scale_comparison_5m_26m_100m.csv}、REPORT、CHANGE_LOG、COMMANDS.log、
ENVIRONMENT.json、HASHES、SHA256SUMS）；审计包 250 members，bank/ 与 q
大文件仅留哈希，审计包 SHA256 `6fe9b0f9…a5f85`。代码
`research/src/fr/g1b_100m_scale_generalization_v01.py` + runner + 单测 24/24。
下一步=G3（clipping intervention）但**不自动启动**，须用户授权。

## 29. G3 — 26M clipping 干预（2026-09-16）—— COMPLETE，MECHANISM_CONFIRMED

科学问题：A0E2 识别的 clipping-Jacobian 机制是否有可干预的因果证据。
26M 固定配置（与 G1a 逐字段一致）；clip=1.0 对照直接读取冻结
`outputs/generalization_g1a_26m_v01/`（chi_fd/B/C 三 seed + 2042 BH_fixed
直读；2043/2044 的 BH_fixed 不在冻结制品中，由 clipref 从只读 G1a bank
新算补全——G3 clip-reference extension，未重训、未改冻结字节，2042 重算
cross-check = 0.0；G3_PROTOCOL.md §16）；
no-clip 条件 3 seeds（2042/2043/2044）× 3 stages 真正重训——训练循环
**从不调用 `clip_grad_norm_`**（三 seed 调用计数全 0，逐步 manifest
记录），preclip norm 仅测量不修改梯度，counterfactual would-clip 率
1.000/1.000/0.980（干预落在高信息区）。冻结响应原语在 no-clip 分支经
CLIP_DISABLED_MAX_NORM=1e30（clamp ≡ 1.0 数学恒等，梯度位级不变）。
paired stream identity：G3 沿用 G1a E1/E2 stream seed 生成规则，
future batch/RNG 跨条件按构造逐位一致。

**G3a operator control**（冻结 G1a bank，seed 2042 × EARLY/MIDDLE/LATE ×
4 streams，H={1,4,8}）：C_identity（未来步 clip-JVP 换 identity，其余
不变）≡ BH_fixed，36/36 单元位级一致（max abs gap = 0.0 ≤ 1e-12）；
未修改 a0e2 重跑 vs G1a 冻结 mechanism JSON cross-check max gap = 0.0。
按 B3 定义 G3a 只归因 operator、非因果 regime 证据。执行期一次
cross-check 定义修正（ablation 的 BH_fixed 因 HVP 方向沿 C 切线、被
identity override 从 horizon 3 起改变而不同——预期行为，非实现
不一致），修复后重跑 G3a 通过，记 CHANGE_LOG。

**G3b no-clip 正式矩阵**：9 态 × 4 streams × H={1,4,8} = 27 单元，
FD/B/BH_fixed/C 全算。BH_fixed ≡ C **逐位崩溃**（27 单元 max abs gap =
0.0，理论退化成立）。pooled：B R²=0.93821/mre 0.11162；BH_fixed=C
R²=**0.99964**/mre 0.006124（median 0.002261）——no-clip 下 C 对 FD
保持高精度。

**机制增益对照（pooled H>1，18 官方单元/条件）**：clip=1.0
gain_curvature +0.07463 / gain_clip +0.09398；no-clip +0.15824 /
**+0.00000**。Interaction：Delta_clip=+0.09398；层次 cluster bootstrap
（seed>stage>stream，20000 次）mean +0.08563、95% PI [+0.00863,
+0.29384]、P(Δ>0)=0.99985；per-seed +0.01586/+0.01785/+0.24823，
per-stage EARLY +0.00047/MIDDLE +0.25669/LATE +0.02478（异质性为真实
数据属性——clip 条件下 seed 2044 MIDDLE 的 BH_fixed 误差 0.54–0.78 而
C 仅 0.02–0.04——登记为观察）。

**判定（阈值冻结于 G3_PROTOCOL.md B10，结果前写入，未改）**：
**MECHANISM_CONFIRMED**——cond1 operator control ✓；cond2 no-clip
pooled |gain_clip| = 0 ≤ 0.2×clip ✓；cond3 no-clip BH/C 各自 R²≥0.5、
mre≤0.5、max rel gap vs FD = 0 ≤ 1e-6 ✓；cond4 Δ>0 ✓。

**解释边界（冻结）**：① 因果声明 rests on G3b 真正 no-clip 轨迹，
G3a 仅为 operator control；② **curvature 增益 no-clip 下不消失**
（+0.158 vs clip +0.075，反而更大）——clipping 不是唯一反馈；
③ no-clip 不意味 C==FD（有限 ε 非线性仍在）；④ per-seed/stage
异质性不拟合定律；⑤ 干预仅证明 26M clip→response 因果链，不外推
其它规模/optimizer/数据分布。

**工程**：smoke（2042/EARLY）16/16（含 no-clip BH≡C 理论退化位级
检验、H1 B=BH=C、H1 vs FD 8.8e-4）；theta 36/36 E2 流、rho0-noop
36/36；tangents replay 36/36 逐位精确；单测 18/18；无 OOM、无 retry、
无 RESOURCE_LIMIT；GPU0 共享、未杀外部任务。

制品 `outputs/generalization_g3_clipping_intervention_v01/`（G3_PROTOCOL、
config、四 manifest、smoke_report、operator_control/、bank/、theta/、
clip_reference/、no_clip_regime/{per_stream.csv, per_state_horizon.csv,
mechanism_decomposition.csv, streams/}、results/{summary.json,
clipping_interaction.csv}、G3_CLIPPING_INTERVENTION_REPORT.md、
CHANGE_LOG、COMMANDS.log、ENVIRONMENT.json、HASHES、SHA256SUMS）；
审计包 252 members（bank/ 与 theta q 大文件仅留哈希），SHA256
`3c2c6fbf…ff3b`。代码 `research/src/fr/g3_clipping_intervention_v01.py`
+ runner + 单测 18/18。下一步=G2（26M 数据分布泛化，**已完成**，见 §30）。

## 30. G2 — 26M 数据分布泛化（2026-09-16）—— COMPLETE，STRONG_DATA_REPLICATION

科学问题（B0）：模型规模/架构/tokenizer/训练目标/AdamW/clipping/token 预算/
因果响应协议全部固定，仅预训练语料 D0→D1 时，完整切线 C 是否仍准确预测
FD 响应、并在 H>1 系统性优于 optimizer-only 切线 B。

**数据（B1/B2/B3）**：D0 = 冻结 G1a 实际训练池
`outputs/e3_confirmatory/v07/e3c_subset.jsonl`（4096 条，SHA256
`1e657af9…b0f4`）。D1 = MiniMind 官方 `pretrain_hq.jsonl`，HF
`jingyaogong/minimind_dataset` 删除前 revision
`3a9bbe611244811e45b96661a84047f112c8ed01`（2026-01-17 上传，2026-03-26
从 HEAD 删除），1,413,103 条，SHA256 `82b19375…18b` = LFS oid。D1 pool =
全量排列（seed 60202601）前 4096 索引升序（SHA256 `5d4c5e6b…a882`），
partition（seed 60202603）probe/eval/train = 32/128/3936（SHA256
`63fed388…f050`）。分布偏移核验：D1 全量 100% 截断在 126 tokens（D0 均值
114.9、73% 截断）、平均字符长 464 vs 266；13-gram shingle Jaccard 均值
1.576e-7（max 0.003546，语料几乎不相交）；D0 精确重复行 0、D1 820 行
（0.058%）。schema 双侧均为 `{"text": str}` → 无 adapter（B4）；同一
tokenizer/管线/PretrainDataset。**D0 只读直取冻结 G1a 结果，永不重跑**。

**设计**：26M 配置与 G1a 逐字段一致（参数量断言 26,681,728；单测比对
G1a config 科学字段子集）；seeds 2042/2043/2044 × stages 26/64/102 ×
H={1,4,8} = 27 official cells（9 states × 4 preregistered streams）；
主统计口径严格 import 冻结 `fr.g1a_26m_scale_generalization_v01` 的
`cell_relative_errors`/`aggregate_metrics`（再入 a0c `_metric`），无第三
种口径；E2 manifest 校验器按 G1a 逐检查镜像（仅协议 tag 不同）。
smoke（B9：seed 2042/EARLY/step 26）18/18 通过，H=1 解析 vs FD 相对误差
1.83e-3，含 train-loader deterministic replay 与 dataset/pool/partition/
state-metadata 哈希核对。

**主结果（27 行 pooled）**：B R²=0.9114962992/corr 0.9918458572/MAE
0.0178247738/mre 0.0916795650；C R²=**0.9999452367**/corr 0.9999876414/MAE
0.0004293364/mre **0.0025988312**。per-H（C）：H1 R² 0.9998531159
（B≡C 结构性平局）；H4 R² 0.9998802541、mre 0.0030133082 vs B mre
0.0758277891；H8 R² 0.9999297443、mre 0.0023441441 vs B mre 0.1967718646。
H>1 paired 18/0/0（win fraction 1.0），mean improvement **+0.133621**
（D0 G1a 为 +0.16861，仍在同一量级）；per-seed 均值 +0.2062701656/
+0.0791880674/+0.1154050690（3/3 正）。

**机制子集（B11，seed 2042）**：H=1 恒等 gap = 0.0；stream-first
gain_curvature **+0.1717813477** / gain_clip **+0.0339615257**；
official-cell 口径 +0.172470 / +0.033800（两种口径都报告、永不混用）。
对照 D0 G1a：curvature +0.0574 / clip +0.0147——D1 上曲率增益反而更大。

**判定（B12，阈值冻结于结果前，未动）**：**STRONG_DATA_REPLICATION**——
cond1 C pooled R² 0.9999452367 ≥ 0.90 ✓；cond2 H>1 18/18 wins ≥ 15/18 ✓；
cond3 H=8 mean err_C 0.002344 < mean err_B 0.196772、R²_C_h8 0.9999297443
≥ 0.50、corr > 0 ✓；cond4 3/3 seed 均值 > 0 ✓。

**Dynamics（B13，仅诊断、从不用于过滤）**：训练 loss 3.6333(D0) vs
3.5575(D1)；preclip norm 1.3906 vs 1.2115；clip 率 1.0 vs 0.9314；future
首步 clip 率 1.0 vs 0.8611；future grad norm 1.3684 vs 1.0934；**HVP norm
0.2837 vs 1.9911（~7×）**；token 长度统计见上。全部登记为观察。

**不确定性（B15）**：D1 H>1 paired 的层次 cluster bootstrap（seed→stage→
cell，20000 resamples，rng 20260916）：observed +0.133621、mean +0.133125、
median +0.121668、95% PI [**+0.067160, +0.251205**]、重采样中差值为正的
比例 = **1.0**（该比例**不是** p 值/后验/显著性概率；3 seeds ⇒ 探索性）。

**解释边界（冻结）**：① D0 vs D1 是 matched descriptive comparison（按
seed/stage/H 身份配对），**不是**样本配对的因果数据集实验；② 非零
shingle 重叠（~1.6e-7）与 D1 内部 820 行重复 ≠ 数据集被更换；③ HVP
norm ~7× 与 clip 率下降是 D1 分布属性，不改变 C 的预测效度；④ 干预仅
证明 26M 当前 optimizer/clip 配置的响应结构，不外推其它 optimizer。

**工程**：theta 36/36 E2 流全过、rho0-noop 36/36；tangents 36/36 baseline
replay 位级精确；单测 31/31；无 OOM、无 retry、无 RESOURCE_LIMIT；
GPU0 共享、未杀外部任务、未触碰 GPU1。一次审计重入缺陷（第二次 audit
把旧 bundle 自包含）修复并复跑，记 CHANGE_LOG §3。

制品 `outputs/generalization_g2_data_distribution_v01/`（G2_PROTOCOL、
DATASET_PROVENANCE、dataset_manifest、config、run/training/checkpoint/
future_stream 四 manifest、smoke_report、data/d1_pool.jsonl、
partition_g2_d1.json、results/{summary.json, per_stream.csv,
per_state_horizon.csv, mechanism_decomposition.csv,
d0_vs_d1_comparison.csv, bootstrap.json}、G2_DATA_DISTRIBUTION_GENERALIZATION_REPORT、
CHANGE_LOG、COMMANDS.log、ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES、
SHA256SUMS）；审计包 256 members（bank/、theta q、raw D1 语料等大文件仅留
哈希），SHA256 `9553ffa5…157a7`。代码
`research/src/fr/g2_data_distribution_generalization_v01.py` + runner + 单测
31/31。下一步=G1c（~1B 压力测试）但**不自动启动**，须逐项授权。

## 28. Post-A0I 定位（2026-09-16，纯文档轮，无 GPU）

A0I 已完成并冻结。本轮只做持久文档更新（修正解释 + 新建
`docs/FR_ALGORITHM_ROADMAP_POST_A0I.md`），未运行任何 GPU 实验，
未重新生成任何 A0I 数字，未触碰任何历史 scientific artifact。

### 28.1 固化后的当前结论

- directional scoring 的**数学恒等式与 decision preservation 成立**
  （等价 SUPPORTED；decision preservation 18/18，独立于 numerical
  gate 54/72）。
- 工程上：candidate-scoring 子阶段墙钟约降 25%，candidate 边际成本
  约降 30%，scorer 峰值显存约降 17%；**selected-gradient recompute
  必须计入**；总 actionable decision cost 只改善约数个百分点，
  总结果 = **SIMILAR**（不是 major speedup，也不是算法失败）。
- **candidate scoring 不再是唯一主导瓶颈**；主要剩余成本集中在
  shared prefix / future-response computation（每决策 ~1s，A0I 三态
  中位数 0.96–1.08s，占总成本 ~70%）。
- 总决策成本仍约 **26–31 个等效 native steps/决策**（A0I-B 实测
  25.6–31.3）；**equal-wall-clock training gain 完全未验证**
  （H6 = NOT_TESTED）。
- 当前核心瓶颈转移为：**shared prefix 成本** + **score→eval
  transfer** + **adaptive value 不确定性**。
- 下一首选假设（未验证）：**cached future sensitivity（p_tau）+
  fresh current G_t**——缓存昂贵未来敏感度，当前 G_t 按当前
  theta/m/v/raw-grad/clip/LR/optimizer step 精确刷新，只近似未来
  敏感度。公式与误差分解见 `FR_ALGORITHM_ROADMAP_POST_A0I.md` B 节。
  该假设可能因误差项抵消而更差，不允许预先写成"比 stale-w 更准"。

### 28.2 修正登记（对 §27 的解释性更正，数字不变）

1. **适应性措辞**：动作频繁切换只证明没有退化为固定动作
   （ADAPTIVE_CHOICES_OBSERVED），不能单独证明净收益
   （ADVANTAGE_OVER_FIXED_UNCERTAIN）。
2. **Regret 第一项**：可以为负；不是"评分器精度损失上界"；
   score 目标下的选择 regret 需另写 `L_score(a_adj) −
   L_score(a_scoreReplay) ≥ 0`（仅定义，无新实验）。
3. **数值等价 vs 决策保持**：54/72 ≠ 18/18，分开计数；
   未做逐行分解，不声称所有 gate failure 完全由 backend 差异导致。
4. **校准**：slope≈0.971 不单独称无偏，须结合 intercept/MAE/R²；
   选中级 slope≈0.605 受选择条件影响，非整体 predictor calibration。
5. **shared-pool**：不相交只排除样本去重收益；不写"永久不适用"，
   vector-output JVP / batch packing 仍是未来工程选项。

### 28.3 下一步（仅记录，NOT_STARTED）

下一实验 = **A0J-P0：bounded shared-prefix profiler / cost
accounting**（先归因 ~1s shared prefix，再依据 profiler 单开一个主
分支：cached-p、FD-HVP 或工程复用三选一）。详见新 roadmap J 节。
本轮未启动任何实验。

## 31. A0J Future-Sensitivity Reuse Study（2026-09-16）—— COMPLETE，TIME_SCALE_NEGATIVE_RESULT（CASE B）

Post-A0I 第一次真正修改 controller 计算结构的算法实验。问题：复用
缓存的 future sensitivity p_tau（age 1/2/4）+ 重新计算当前真实 AdamW
输入作用 G_t（w̃=G_t^T p_tau），能否以明显更低成本做出与完整 fresh-p
controller 相同/足够好的动作。合同先于 GPU 冻结
（`outputs/a0j_cached_future_sensitivity_v01/contract.json`，
contract_sha256 9a306711…；dev manifest 冻结 3 anchors × 2 streams，
module_sha256 共 5 次透明更新，全部在查看任何结果之前记录）。

流程纪律：P0 profiler → 预注册成本门（per-repeat interleaved
S_upper median **4.4147** ≫ 1.25 → 进入完整 P1）→ implementation
gate（apply_Gt_T 复现生产 w/scores/chosen-id，冻结容差
atol32=1e-5/rtol32=1e-4）→ 三路比较（EXACT / CACHE_P / STALE_W，同
a_j(t)=g_j(t)−g_t，候选永远重算，禁止复用 G_tau）→ cache age 1/2/4
→ dev replay utility（先冻结 decisions + sha256 再 replay，90 branches）
→ 成本核算 → dashboard → 文档 → audit package。

Implementation gate 曾经 FAIL：24 行自检全 0.0 但 6 行生产复现行全部
超差（score diff 4.5e-4–1.6e-3，确定性）。探针链（prod-vs-prod 0.0 /
verbatim-vs-verbatim 0.0 / 模块≠verbatim / bisect 定位到 rollout 段 /
per-call trace 显示 endpoint post_model hash 分叉）定位到模块
`_clip` tick 漏调用 `pe._step_once(rt)`——pending horizon 的优化器步
没执行，horizon age+2..age+H 在未步进状态上累积 p.grad。修复
（3ee5ded9，复看后逐位 0.0）→ 全 P1 gate **PASS 30/30**。该 bug
说明仅自检一致的 gate 不够，§9 生产复现行是必要一行。

主要结果（n=6 units/age，PRIMARY=chosen-id preservation）：

| age | CACHE_P preserve | STALE_W preserve | dGain cache-exact | dGain stale-exact |
|---:|---:|---:|---:|---:|
| 1 | 5/6 | 5/6 | −6.76e-5 | −6.76e-5 |
| 2 | 4/6 | 4/6 | −1.47e-4 | −1.47e-4 |
| 4 | 1/6 | 0/6 | −4.90e-5 | −9.27e-5 |

full-rank agreement 更差（age1 仅 3/6）。根因是时间尺度：exact
top1-top2 margin median 3.8e-5 / min 3.9e-6，而 age1 score 误差已
~2e-3（age4 ~8e-3），误差超出 margin 2–3 个数量级，argmin 翻转不可
避免。cosine(w_exact,w_cache)=0.955/0.823/0.643（age1/2/4）与翻转
共存——再次确认 cosine 不得作 gate。CACHE_P 一致但仅略优于
STALE_W（fresh G_t 有价值，p 复用窗口太短）→ **§18 CASE B**
（含 CASE C 元素），缓存路线停止；不靠 cosine 或调 threshold 挽救。

成本（shared-GPU 争用抬高绝对值，native-step 等效为可比的量）：
exact 决策 5.618s ≈ 104 等效 native steps（安静包络 0.0542s）；
refresh 1.279s ≈ 24；cached non-refresh 边际 0.912s ≈ 17.3
（GtTp 0.132 + candidate score 0.594 + selected-action execution
0.185，如实含保守候选重算）。节省真实存在（CASE D 排除），失败不
是工程杠杆问题。cache memory：p 三通道 57.0MiB、w 19.0MiB、
p/P=3.0。Kr/Ks 投影 8 格全部 COST_ONLY / UTILITY_NOT_VALIDATED。

H6 EQUAL_WALLCLOCK_TRAINING_GAIN=**NOT_TESTED**；未把摊销投影或
单步 replay gain 冒充 equal-wall-clock training gain。GPU 记账
≈1641s/1800s（含 preload/探针/崩溃会话估计），4 次 infra 类崩溃
重试（E4 KeyError、NameError、AttributeError、ImportError），同
manifest/同算法，修复与查看结果之间严格分开；环境零变更。

制品 `outputs/a0j_cached_future_sensitivity_v01/`（dashboard、
REPORT.md、decisions_frozen.json+sha256、P0/P1/replay/cost 全 CSV、
gpu_budget_log、manifest_sha256.txt）；审计包
`a0j_audit.tar.gz` 9067189B SHA256 `21c1a2f9…3e5559a`。代码
`research/src/fr/a0j_future_sensitivity_cache_v01.py`（hash 链
246c3858→…→fa62027b）+ `research/scripts/a0j_*.py` + 单测 12/12。

下一步（仅记录，NOT_STARTED）：以本轮为时间尺度负结果基线；若重访
future-sensitivity 摊销，方向只能是 refresh 触发基于实测 p-drift
而非固定 age、或复用 endpoint 梯度方向+新鲜标量，均须新预注册轮。
## 32. G1c — External ~1B Stress Test（2026-09-16）—— BLOCKED，EXTERNAL_CHECKPOINT_UNAVAILABLE

科学定位（B0）：非 pure scale test。外部 1B 同时改变 architecture /
tokenizer / corpus / optimizer 超参 / training schedule，唯一允许的主
结论是"核心有限时间状态空间响应结构（FD/B/C）能否在一个公开的、约 1B、
具有真实 optimizer state 的外部 LLM 训练系统上继续成立"；禁止解释为
scaling law 或 MiniMind→1B 单因素外推。

**B2 制品核验（在任何科学结果查看之前）**：HF API 逐 revision 文件列表
检查官方 AllenAI OLMo ~1B 全系列——allenai/OLMo-1B（main + 9 个跨全程
抽样 revision，全系列 336 branch）、OLMo-1B-hf、OLMo-1B-0724-hf、
OLMo-1B-0724-954000steps-unsharded（仅 config.yaml + model.safetensors）、
OLMo-2-0425-1B（main + 3 抽样 branch，全系列 268）、OLMoE-1B-7B-0924
（MoE，架构排除）、OLMo-7B（抽样 1，且 7B 超资源范围）；历史官方域名
olmo-checkpoints.org = HTTP 404。**结果：0 个 state 文件（optim*/train*/
scheduler*/rng*）**——全系列公开 checkpoint 均只有模型权重。

OLMo-1B 原则上满足候选理由（1.24B autoregressive Transformer；AdamW
betas(0.9,0.95) wd0.1 lr4e-4；max_grad_norm=1.0；336 个公开 step
checkpoint；Dolma/olmo-mix v1.6 语料与 tokenizer 公开），但不满足 B1
决定性要求（unsharded checkpoint 含 model+optimizer+train state）。
按 B2：不得补造 optimizer state；同官方系列内无可切换对象 →
**EXTERNAL_CHECKPOINT_UNAVAILABLE，停止科学实验**。此标签是制品可用性
声明，不是机制的科学负结果。

执行事实：大文件**零下载**；GPU **零运行**；B3/B6/B10/B11/B14 未触发。
硬件探针实测 2× RTX 5090 32GB（GPU0 共享余量 ~15.8GB），与任务书
"A100-PCIE-40GB"假设不符，已登记（RESOURCE_FEASIBILITY.md §1 /
ENVIRONMENT.json）；若未来重试，B7 offload + B8 native-2048 优先/
short-context 备选的资源策略直接复用。

制品 `outputs/generalization_g1c_external_1b_v01/`（G1C_PROTOCOL、
EXTERNAL_MODEL_PROVENANCE、RESOURCE_FEASIBILITY、
external_checkpoint_manifest、future_stream_manifest（BLOCKED）、
experiment_config、smoke_report（NOT_RUN_BLOCKED）、results/ 五件套
（零数据行，标注阻塞）、G1C 报告、CHANGE_LOG、COMMANDS.log、
ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES、SHA256SUMS、
screening_evidence/ 4 个证据小文件）；审计包 21 文件 +
SHA256SUMS 自校验 + 读回验证通过，SHA256 `c78dbbfb…854c9`。

**对第一篇的影响**：受控 MiniMind scale ladder（5M→26M→100M）、clipping
干预（G3）、数据分布泛化（G2）结论不受影响；外部 ~1B 轴未覆盖，原因
=官方系列不发布 optimizer state。是否更换外部家族（须含公开 optimizer
state 的 checkpoint）由用户决定，不自动执行。

## 33. G1c checkpoint 直链可用性复核（2026-09-16）—— AVAILABILITY_RECHECK_PASSED

**任务**：FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01——对 §32 的
EXTERNAL_CHECKPOINT_UNAVAILABLE 判定做官方 standard checkpoint
**直链文件级复核**。只读 HTTP 核验；无 GPU、无科学 response、无完整大
文件下载（每大文件 ≤1 MB range read）。

**核验逻辑修正（§1）**：官方 `docs/Checkpoints.md` 明确 unsharded
checkpoint = `config.yaml + model.safetensors + optim.safetensors +
train.pt`，旧 release 用 `.pt` 扩展名；且 **directory URL 不能直接访问、
目录内文件公开可访问**。故"目录 URL 404 ≠ checkpoint 不存在"；§32 的
HF screening 保留为有效历史（HF repos 是 weights-oriented publishing
path），但其负面解读限定于 HF 路径。

**执行**：官方 CSV `configs/official-0425/OLMo-2-0425-1B.csv`（195
checkpoint，step 0…1,907,359；stage2 CSV 一并登记未选取）按 step 冻结
4 个代表 checkpoint（EARLIEST step0、EARLY step380000=19.92%、
MIDDLE step950000=49.81%、LATE step1530000=80.22%，先于任何探测），
逐文件 `curl -sIL` + range GET。

**结果（4/4 checkpoint 四件齐全）**：config.yaml（163–164KB）与
train.pt（14–27KB）完整读取 HTTP 200；safetensors 变体 404（旧
release 情形），按官方文档回退 `model.pt`（5,939,703,038 B，四 step
同字节数、ETag 逐步不同）与 `optim.pt`（~11.88 GB ≈ 2× model，
符合 AdamW m+v fp32）均 HTTP 200，前 1 MB 实际读取验证（PK\x03\x04
torch zip magic；train.pt 可见 train/data.pkl）。全部 redirect=0，
无官方替代 host（olmo-data.org 仅承载 Dolma 语料）。目录 URL 与文件
状态显著不同（目录 404 / config.yaml 200），与官方文档逐字一致。

**判定**：**FULL_STANDARD_CHECKPOINT_AVAILABLE** → 上一轮
EXTERNAL_CHECKPOINT_UNAVAILABLE 修正为 **AVAILABILITY_RECHECK_PASSED
/ READY_FOR_1B_FEASIBILITY**（核验路径修正，不是科学结论改判）。
**G1c 可以恢复继续**；下一步 = **FR_G1C_1B_FEASIBILITY_GATE**（G1c 1B feasibility gate，B3/B6 起），
**须用户授权、本轮不自动启动**（不自动下载完整 checkpoint、不自动
运行 GPU）。资源登记：单 checkpoint 磁盘 ~17.8 GB（fp32），实测硬件
2× RTX 5090 32GB 须重新探测余量后按 B6/B7 评估。

**已验证存在（4/4 checkpoint）**：OLMo standard unsharded checkpoint 的 config.yaml + model.pt + optim.pt + train.pt 四件齐全（model.safetensors/optim.safetensors 404 属官方文档旧 release .pt 情形）。**定位说明**：G1c 属 **external architecture validation**（外部架构/tokenizer/语料/optimizer 超参联合改变，confounded by design），**不是 5M/26M/100M 的 scale-only continuation**（scale-only 阶梯 = G1a/G1b）。科学实验未执行，禁止表述"OLMo-1B response validated"。

**制品**：`outputs/generalization_g1c_checkpoint_recheck_v01/`（协议 /
manifest / per_file_http_probe.csv 16 行 + pt_fallback_probe.csv 8 行 /
HOST_MIGRATION_NOTES / 报告 / COMMANDS.log / SOURCE_HASHES / SHA256SUMS
71 项自校验 PASS / probe_evidence 原始 curl 头与前 1MB）。


## 34. A0K Fresh Response Closed-loop Validation（2026-09-17）—— COMPLETE，FORMAL_CASE_A_WITH_QUALIFICATIONS

**任务**：第一轮真实训练闭环实验：fresh response controller 在真实训练预算约束下是否有净训练收益。非新算法；controller 计算结构不动；唯一变化是把冻结的生产决策接入真实训练循环。

**冻结（先于任何 GPU）**：`outputs/a0k_fresh_response_closed_loop_v01/contract/contract.json` + `live_manifest.json`（72 events = 6 seeds × 12 unique steps；events sha256 `8c099f519ed746e3…`）；H=4、λ=0.05、5 候选（j=0 no-op 原生 AdamW；j>0 注入 g+λ(g_j−g)）、selector argmin_j(λ·ĥ_j) tie→最小 id、生产 `run_adjoint_only` 打分；禁 cache/stale/JVP/block planning。K=20 主 cadence、K=10 敏感性；seeds 2030–2032 主 + 2033–2035 预算扩展；T_total=128（= 冻结计划轨迹）。

**设计**：每 seed 三臂完全配对（A 纯 AdamW / B 固定 j=1 同 cadence 注入 / C fresh controller：事件前快照 model+optimizer+scaler+RNG → 快照上生产决策 → 恢复 → 执行所选动作一次）+ A_wc/B_wc 同 T_C 墙钟臂（合同 §12：controller 全部开销计入 T_C）。等步 + 等墙钟双比较。每控制事件存完整快照 .pt + 17 字段日志（含 hhat j0–j4、predicted ΔL、四段时间戳）；离线 A0H20 replay（age=0、H=4、5 分支、双 V_score/V_eval 标签）72 事件全部 offline-argmin 重建 PASS。

**身份 gates（科学运行前工程门槛）**：S1 arm A(2031) 与 v0.7 bank step46/52 逐位一致（`2f40340e…` / `ada46c18…`）；S2 ghost no-op（全套 C 决策机制 + 强制 j=0）轨迹不可见（尾部哈希逐位一致）→ RNG 卫生成立；S3 事件成本 mean 2.85s。

**结果**（K=20 主 cadence，6 seeds；增益 = 损失下降，正 = controller 更好）：
- M1 等步 A−C @128：**median +7.44e-4 nats**，4/6 正（2032 −4.0e-4、2034 −7.5e-4 负，如实保留）；K=10（3 seeds）median +1.64e-3，3/3 正。
- M2 等墙钟（PRIMARY）：median +7.44e-4——**9/9 对基线在 T_C 内完成全部 128 步**（T_C≈29–71s vs T_A128≈9–16s），基线从未失步 → Gain_time 数值上 == 等步增益（尺度条件，见限定）。T_extra 17–56s/run；M3 ROI median ~5e-5 nats/s。
- M5 adaptive vs fixed-j1（Q2）：等步 median **−1.29e-4**，C≤B 6/9 对——**selector 不优于冻结固定动作**；收益来自干预本身（数据组梯度注入 λ=0.05），不来自自适应选择。
- M4：no-op 率 **0%**（全部 72 事件均存在 ĥ_j<0 候选，argmin 门从未对 no-op 生效）；有害率 0–50%（K=10 高于 K=20）；chosen 分布跨 j1–j4 无支配；平均短视界增益 O(1e-4–2e-3)。
- 每 seed 全数报告（禁只报收益大 seed）：M1 范围 −7.5e-4…+2.0e-3。

**判定**：合同 §17 形式符号规则 → **CASE A**（等步>0 且墙钟>0），附三项限定（报告 §10）：① Q2 为负——选择器本轮无增益；② 墙钟赢在尺度——基线未失步，更大规模下 T_extra 会使基线失步、符号可能翻转；③ 符号一致性弱（K20 4/6、量级 O(1e-3)，n=6 不足以确立显著性）。

**工程透明（全部分析前记录于 live_manifest module_hash_notes）**：driver attempt1 PROJECT_ROOT 路径 bug（零 GPU 消耗）修复；B_wc 臂事件排除缺陷自查发现、**分析前**修复并用 a0k_rerun_bwc_v01.py 重跑 9 臂替换 phase_rows；events 哈希链全程未动。GPU **1693.7s/1800s，0 crash**。共享双卡（外部任务在场）。

**制品**：`outputs/a0k_fresh_response_closed_loop_v01/`（contract/ runs/ dashboard/ reports/A0K_REPORT.md audit/ hashes/）；审计包 289629B SHA256 `64b16f97d33cb300b4459b37cf0effea5823fca8c6d613eaba822f046086fd51`。A0G/A0H/A0I/A0J 历史制品未动。

**停止条件**：已执行——不自动开 A0L / λ 调节 / block planning / optimizer 改动；等待分析。


## 35. A0L Response-guided Continuous Mixture Feasibility（2026-09-17）—— COMPLETE，CASE B（B≥C 限定）

**任务**：验证训练响应能否从"一次性动作选择"（A0K）升级为"持续数据混合控制"。非修复 selector；验证 continuous control hypothesis。不改 optimizer / H / λ / 模型结构。小规模 pilot，不自动扩展。

**冻结（先于任何 GPU）**：`outputs/a0l_response_mixture_feasibility_v01/contract/contract.json`（sha256 `a541dd3c…`）+ `a0l_events_manifest.json`（6 events = 3 seeds × steps {50,100}，复用 A0K 冻结 stream seeds 内容重哈希并核验一致，events sha `04bd9f2b…`）。pi_0 = 经验组占比 [1038,973,987,938]/3936 = [0.2637,0.2472,0.2508,0.2383]（4 组划分 3936 训练池 = original distribution）。controller：每 K=50 steps 快照 → 生产 H=4 response h_j → q=softmax(−γ_t·h)，γ_t=1/(std(h)+1e-12)（冻结 scale-free 形式）→ π←0.9π+0.1q，α=0.1 冻结 → 恢复。**treatment = 数据构成**：每微步按 π_window 从专用 torch.Generator 采样组分（与全局 RNG 隔离，三臂 dropout 序列逐位相同），无梯度注入；optimizer/clipping/H=4/λ=0.05 全不动。窗：π_0 步 1–49、π_50 步 50–99、π_100 步 100–128。seeds 预冻结 [2030,2031,2032]，不扩展。

**设计**：每 seed 5 臂——A 原生 AdamW（全池流）；B fixed-j1 连续混合（全部微批来自组 1）；C response 混合 controller；A_wc/B_wc 同 T_C 墙钟臂（T_C = C 实测墙时，A_wc/B_wc 同场同预算）。身份 gates：S1 arm A(2031) 与 v0.7 bank step46/52 逐位一致 PASS；S2 arm B(2030) 两次运行轨迹尾部逐位一致 PASS。每控制事件 21 字段日志（pi_before/h_j/q_j/pi_after/window train loss/v_eval/γ_t/std_h…）。

**结果**（final v_eval@128；增益 = 损失下降，正 = 更好；3 seeds 全报）：

| seed | L_A | L_B | L_C | C−A | B−A | B−C | 排序 |
|---|---|---|---|---|---|---|---|
| 2030 | 6.94879 | 6.94389 | 6.94262 | +6.17e-3 | +4.90e-3 | +1.27e-3（C 优） | C<B<A |
| 2031 | 6.93000 | 6.93838 | 6.96024 | −3.02e-2 | −8.38e-3 | −2.19e-2（B 优） | A<B<C |
| 2032 | 6.99005 | 6.95657 | 6.96403 | +2.60e-2 | +3.35e-2 | −7.46e-3（B 优） | B<C<A |

- 评价 1 Final Eval Loss：见表。
- 评价 2 Gain vs AdamW：C median **+6.17e-3**（2/3 正）；B median +4.90e-3。
- 评价 3 Gain vs fixed-j1（B−C）：median **−7.46e-3**——B 不劣于 C，2/3 seed B 优。
- 评价 4 等墙钟：尺度条件成立（3/3 对 A_wc/B_wc 在 T_C 内完成全部 128 步，T_extra≈6–7s），Gain_time 数值==等步增益；median vs AdamW +6.17e-3。
- 评价 5 pi 轨迹：仅 2 次更新，|π−π_0| 移动小（最大 component 1：0.2637→0.3248，seed 2032 step100）；step-50 的 h_j 排序 seed 依赖（2030 最不利组 1、2031 组 4、2032 组 3），q 经 std 归一化后温和再权重。

**判定**：合同确定性规则 → **CASE B**（非全 seed C<B<A；C 对 A median 为正），附限定：**B ≥ C**——|median(C−B)| = 1.52×|median(B−A)|，response controller 未优于平凡 fixed-j1 混合，跨 seed 排序无一致。按 mandate 含义：continuous response controller 路线**不进入**；"response-discovered schedule" 备选解读被 B≥C 削弱（固定组 1 混合无需 response 计算即可拿到 median 收益）。hypothesis 检验受 pilot 设计限制：仅 2 个 K=50 更新、α=0.1，混合信号 translated 的幅度/频率不足以分离 C 与 B。

**工程透明**：driver attempt1 S1 PASS 后 S2 smoke 崩溃——arm B `members["1"]`（组 973 < 1024 所需）off-by-one，应为 `members["0"]`（1038，controller 组分 j=1）；**零科学结果存在时**修复（arm A 路径未动），hash note 先于重跑记入 contract/live_manifest.json。GPU **351.8s/1800s，0 crash**。共享双卡外部任务在场。

**制品**：`outputs/a0l_response_mixture_feasibility_v01/`（contract/ runs/ dashboard/ reports/A0L_REPORT.md audit/ hashes/）；审计包 55273B SHA256 `ec4d877c340965df8b50ec5b6bc5222a07b0123c2937877b9be70aa1aac1601b`。A0G/A0H/A0I/A0J/A0K 历史制品未动。

**停止条件**：已执行——pilot 完成即停；不自动扩展、不改 λ/optimizer、不开新轮。等待分析。


## 36. A0M Response Temporal Structure Analysis（2026-09-17）—— COMPLETE，CASE A（phase structure 在幅度与 argmin 身份，不在秩序列）

**任务**：纯机制分析轮——验证 response 是否包含训练阶段信息。只用 frozen artifacts（A0K events/step logs/counterfactual replay、A0L events），零训练、零 GPU、零 optimizer/λ/controller 改动；不设计算法。

**方法**：E1 trajectory（rank/top-action/Δij）；E2 phase clustering（[h1..h4, v_eval, preclip_norm] PCA+KMeans）；E3 stability（cos+Spearman at lags 10–110 steps，冻结事件网格 K=10→10 步间距，mandate 的 k=1/5 步低于网格已注明）；E4 utility（pooled Pearson/Spearman h_j vs replay Gain_j，single vs window-averaged w=1..4）；E5 fixed-j1 原因（per-phase mean h_j + argmin identity）。分类规则分析前固定于脚本 docstring。

**结果**（dashboard）：
- rank change frequency（per-seed median）= **0.833**；但 Spearman(h_t, h_{t+k})≈0（秩为近并列噪声）。
- response cosine @100 steps = **0.629**，且随 lag 不衰减（0.52–0.79 平台）→ 方向稳定。
- phase clustering ARI = **0.076**（低）/ silhouette 0.574；crosstab 显示簇把 early 与 mid+late 分开但 mid/late 不可分。
- single h utility Pearson = **−0.866**（p≈1e-44，负号=对齐：h 越低 gain 越高）；window-averaged w=2..4 反而退化到 −0.45…−0.60 → **w=1 最优，时间窗口不改善**。
- j1 dominance（argmin 占比）= **0.306** → j1 非全局偏好。
- stage dependent optimum = **8/9 seed-series** 的 per-phase argmin mode 随阶段变化（K10 每 phase 4 事件：2030 1,1,3；2031 4,2,1；2032 2,4,4）。
- top-action predicted vs actual match = 0.444（chance 0.25）。
- common-mode 诊断：|mean(h_j)|/std(h_j) median **1.85**（共模主导）；共模 early −9.5e-3 / mid −3.0e-3 / late −1.2e-3（**~8× 衰减**）；within-event Spearman(h,gain) 均值 −0.233。

**机制图景**（单一解释贯通全部观测）：response 向量 = 大幅值共模（全部候选共享的"干预有益"信号）+ 小幅值差分（哪个候选最好）。共模随训练阶段 ~8–10× 衰减（= phase structure 的主载体，方向因此稳定、cos 平台）；差分相对噪声小 → 单次事件秩翻转频繁（rank change 0.83、lag Spearman≈0）；但 phase 平均后 argmin 身份 8/9 series 随阶段变化（= schedule 机会信号）；E4 强相关主要来自共模的幅度变化；固定 j1 的长期收益（A0L B≥C）与本观测一致——j1 仅 30.6% 事件 argmin，fixed-j1 并非 response 全局偏好的产物。

**判定**：**CASE A — response 存在明显 phase structure**（规则：rank change 0.83≥0.5 且 stage-dependent optimum 8/9≥0.5）。限定：phase structure 的载体是**幅度衰减 + argmin 身份的 phase 依赖性**，而非稳定的秩序列；聚类 ARI 低说明 [h, v_eval, grad norm] 线性簇不能完整标定阶段；per-phase n 小（K20 每 phase 2 事件）。

**制品**：`outputs/a0m_response_temporal_structure_v01/`（analysis/ 8 CSV、dashboard/、reports/{A0M_REPORT.md, phase_cluster_report.md}、hashes/、SHA256SUMS.txt 12 项）；脚本 `research/scripts/a0m_temporal_structure_v01.py`。GPU 0s（纯离线）。

**停止条件**：已执行——不设计新算法、不改 optimizer/λ、不训练 controller、不运行 A0N、不改写已有结论。


## 37. A0N Response Phase-aware Pilot（2026-09-17）—— COMPLETE，CASE B

**任务**：验证 A0M 发现的 response phase structure（共模 ρ=|mean(h)|/(std(h)+1e-12) 随阶段 ~8–10× 衰减）能否指导训练策略。非最终算法。不改 optimizer/λ/response 公式，无复杂 controller、无 adaptive mixture、无 cache。

**冻结（先于任何 GPU）**：`outputs/a0n_phase_scheduler_pilot_v01/contract/contract.json`（sha256 `1730c7a7…f4939`）。策略：每 K=10 事件算生产 H=4 response h_j 与 ρ；ρ>2 → no-op；1≤ρ≤2 → 注入 argmin hhat_j（g+0.05·(g_j−g)，tie→最小 id）；ρ<1 → no-op。事件复用 A0K frozen K10 manifest（read-only）；seeds 2030–2032；T=128；A/B 臂走未改动的 A0K train_arm（B=盲注 j1）。

**门禁（全 PASS）**：S1 arm A(2031) 与 v0.7 bank step46/52 逐位一致；S2 ghost-C(2030)（全套 phase 机制强制 no-op）尾部哈希 == arm A；POST-MAIN 本批 A/B final_v_eval 与 frozen A0K K10 summaries 逐位一致（3/3 seeds）。

**结果**（final v_eval@128；增益=损失下降；3 seeds 全报）：

| seed | L_A | L_B | L_C | C−A | B−C | 注入 | occ H/M/L | ρ median |
|---|---|---|---|---|---|---|---|---|
| 2030 | 6.948787 | 6.947484 | 6.948163 | +6.24e-4 | −6.78e-4 | 5/12 | 4/5/3 | 1.78 |
| 2031 | 6.930000 | 6.928036 | 6.930264 | −2.66e-4 | −2.23e-3 | 4/12 | 8/2/2 | 2.54 |
| 2032 | 6.990052 | 6.988852 | 6.988792 | +1.26e-3 | +5.95e-5 | 6/12 | 2/9/1 | 2.02 |

- Gain vs AdamW：median **+6.24e-4**（2/3 正）。
- Gain vs fixed-j1：median **−6.78e-4**（1/3 正）→ **B ≥ C**。
- 等墙钟：尺度条件成立（3/3 A_wc/B_wc 完成 128 步），Gain_time 数值==等步。
- Controller overhead：T_extra median ~13–30s（12 事件 × ~2.6s adjoint），占 T_C ~35–60%。
- Phase occupancy 非退化（H/M/L = 4/5/2 中位），ρ(t) 轨迹清晰显示早期 high（ρ 3–7）→ 中期 medium → 局部 low（ρ 0.01–0.59）——A0M 的 phase structure 在闭环中复现。
- medium 事件的 argmin 与 A0K 冻结 selector 选择多数一致；策略如冻结执行。

**判定**：冻结规则 → **CASE B（C≈B；response 主要发现固定策略）**。限定：① B≥C median（与 A0L 一致）——ρ 门控去掉的 high/low 事件注入并未造成净损失但也无净收益，phase scheduling 未带来超额收益；② C vs A median 为正（+6.24e-4，2/3）说明 medium 期注入仍有微弱净收益；③ n=3 pilot，量级 O(1e-4–1e-3)。

**制品**：`outputs/a0n_phase_scheduler_pilot_v01/`（contract/ runs/ analysis/rho_trajectory.csv dashboard/ reports/A0N_REPORT.md audit/ hashes/）；审计包 34037B SHA256 `47ab2b026ece0ecb5012afd898202f90f4e80789ef83b1d3386b1ba9b977e3ab`。GPU **416.2s/1800s，0 crash**。A0G–A0M 历史制品未动。

**停止条件**：已执行——pilot 完成即停；不扩展、不改 optimizer/λ、不开新轮。

## 38. A0O Learning-rate Input Response Validation（2026-09-18）—— COMPLETE，Q1/Q2 YES

**任务**：验证训练动力学响应框架可从 data perturbation（A0G–A0N）扩展到 **optimizer input perturbation**：对 AdamW 原生学习率扰动 u = log(η/η₀)，是否存在稳定、可预测的有限时间响应 h_eta = dL_score(θ_{t+H}(u))/du|_{u=0}？三阶段：Stage 1 数值验证（analytic+FD）；Stage 2 有限动作预测（increase/baseline/decrease，argmin h_eta）；Stage 3 独立 eval（仅前两阶段 PASS 才执行）。

**冻结（先于任何 GPU）**：`outputs/a0o_lr_response_validation_v01/contract/contract.json`（最终 sha256 `c2399d1a…`，amendment 后重冻结）。δ=0.05 提前固定（禁止按 eval 调整）；η(u)=η₀·e^u，u∈{−δ,0,+δ}；真实 AdamW 更新 θ_{k+1}=(1−ηλ_wd)θ_k−ηd_k（weight decay 必须包含）；状态 z=(θ,m,v)；optimizer/scheduler/λ 全不动。门禁：pooled |Pearson|≥0.8 AND sign accuracy≥0.9，S1 失败即 STOP。**Amendment A1**：A0O bank 状态（46/52/58/64/70/76/82）与 A0K live manifest 事件键（10..120 步、seeds 2030–2035）不匹配 → 确定性映射 2030/2031→自身、2037→2032，step→不超过它的最大 10 的倍数（仅作 stream provenance 门）。**Amendment A2**（理论修正，先于任何有效结果）：LR pulse 是**纯 θ 通道推**——m₁/v₁ 对 lr 无关（被消费梯度 c_k 不含 u），dc-channel/gradient-frame 旧公式错误（污染 m/v）；正确式 dθ_{k+1}/du = lr_k·e_lr(k)，e_lr = −adaptive(c_k) − λ_wd·θ_k；h_eta = Σ_k ⟨p_th^{(k)}, lr_k·e_lr(k)⟩，p_th 为进入反向迭代 k 的 θ-cotangent（复用 A0H15 生产伴随：`adamw_tangent_step_adjoint` + clip_transpose + k≥2 fp32 HVP）。stream 内容随 stream_seed：`batch_seed=derive_stream_seed(stream_seed,404)` + `batches_from_train_pool`（A0K 配方）。全部 module hash 变更在查看结果前记入 live_manifest（seq 0–12）。

**Stage 1 门禁（PASS）**：v0.7 bank seeds 2030/2031/2037 × steps 46/52/64/82 × streams 7110–7113 × H∈{1,4}，n=96。pooled **Pearson +0.9552、sign accuracy 0.9688**、rel err median 0.225（H1：0.9550/0.9792；H4：0.9517/0.9583）；u=0 分支逐位复现 baseline（CRN 全 cell OK）。→ **Q1 = YES**：LR-input 响应存在，analytic h_eta 与中心 FD（u 施加于每个 optimizer step 的持久 schedule 输入）一致。

**Stage 2 结果**：seed 2030 × steps 46/64/82 × streams 7200–7207，H=4，n=24。direction accuracy **0.875**（null=always-baseline 0.125）；median regret **0.0**（null 4.80e-4）；校准 Pearson 修正后 **+0.942**（driver ≤seq9 因单侧分支取反误报 −0.250；已从保存 cells 离线重算并修正，见 `analysis/stage2_metrics_corrected.json`）；Stage-2 子集 central-FD 相关 0.984；真实最优动作计数 increase 17 / baseline 3 / decrease 4；曲率 median −1.3e-4（一阶主导）。→ **Q2 = YES**：显著优于 always-baseline null；**Q3 = YES**：optimizer input 是新的、有真实预测力的控制维度。

**判定**：值得探索闭环控制，但**仅作为 pilot 且必须 equal wall-clock**（A0K/A0L/A0N「固定策略有竞争力」的限定仍成立）。**Stage 3 未启动**（需显式授权）。

**制品**：`outputs/a0o_lr_response_validation_v01/`（contract/ runs/ analysis/ reports/A0O_REPORT.md hashes/ audit/）；审计包 tar SHA256 `96d889f20f5be657f0864f79e92122de2d99ee848a8e62a0dcbe678351ad8d90`（15 文件，manifest 全部自校验通过）。代码 `research/src/fr/a0o_lr_response_v01.py` + `research/scripts/a0o_{run,analyze,audit_pack}_v01.py`。GPU **4020s/1800s 超限**：全部来自结果查看前的 5 次缺陷修复 launch（设备序、A1 键映射、clip 模块导入、bank 状态加载、snapshot lr、val_loss 签名、A2 理论修正等，均先于任何有效结果并在 live_manifest seq 0–12 留痕），最终一次 launch 产出全部有效结果。A0G–A0N 历史制品未动。

**停止条件**：已执行——Stage 1 PASS → Stage 2 完成即停；Stage 3 不自动启动；不改 δ、不加 threshold/phase、不改 optimizer/scheduler/λ、不扩大候选。

## 39. A0P Learning-rate Response Temporal Kernel Validation（2026-09-18）—— COMPLETE，STAGE-1 GATE FAIL（负结果，机制诊断完成）

**任务**：验证 A0O 发现的 LR response 是否具有稳定时间尺度，建立短期响应→长期训练收益的理论桥梁。脉冲核定义（mandate）：step t 施加 u=log(η/η₀)（仅此一步），R(τ)=dL(t+τ)/du|_{u=0}，τ∈{1,2,4,8,16,32,64} 冻结网格；analytic 式 = A0O 验证伴随源项重定位至 k=1（冻结 a0o 模块 import-only，δ=0.05 未重定义）。

**冻结（先于任何 GPU）**：合同 sha256 `f7b04169…0aef`；bank seeds {2030,2031,2037} × steps {46,82} × streams {7300,7301} = 12 cell；Stage-1 门禁沿用 A0O 阈值（pooled |Pearson|≥0.8 ∧ sign≥0.9），FAIL→STOP；拟合（exp/power，scipy curve_fit 冻结 p0）+记忆度量（I、半衰期、sign stability）+Q1/Q2/Q3 判定规则全部预先注册；禁止 controller/λ/optimizer/τ 选择。live_manifest seq 0–5。

**执行**：单次 launch，12/12 cell 完成，**864.9s/1800s**，0 进程崩溃，CRN 12/12 逐位复现。

**Stage-1 门禁：FAIL**（冻结标准）——pooled n=84：Pearson **+0.7314**（<0.8），sign accuracy **0.9405**（≥0.9 ✓），rel err median 0.2508。per-τ：τ≤8 sign accuracy 全部 1.000、Pearson 0.43–0.89；τ16/τ64 sign 降至 0.833。按合同 STOP，driver Stage 2 未执行。

**失败机制诊断（非缺陷）**：①代码路径与 τ 无关（rewind 步数是唯一变量），CRN 全净，短 τ sign 全对——无缺陷指标；②|R| 中位 1.5–3.9e-3 全程无显著衰减，长 τ FD 散点由一阶切线预测失效（轨迹混沌发散 + O(δ²) 曲率）主导；③与 A0M「response 方向 ~100 步不衰减（cos@100=0.629）」一致。

**离线描述性诊断**（gate FAIL 后对已记录核的计算，公式冻结一致，不构成绕过）：Q1=NO（median sign stability 1.000 但 median best R² 0.351<0.8、CV(τ_c)=3.32≫0.5；exp R² 中位 0.071 vs pow 0.306）；Q2=YES 作为描述性统计（S4/S64 sign agreement 12/12、Pearson +0.861）；Q3=NO（τ_c 组中位 70–577 跨 seed/step 不集中；归一化核形 pairwise Pearson 中位 +0.490）。半衰期 3.6–7.9（早段快速衰减）后长尾不收敛。

**Response temporal law（本轮输出）**：LR 脉冲响应核呈**快速初始衰减（τ1→τ4 约减半，半衰期 ~4–8 步）+ 长记忆噪声尾巴（|R|~2e-3 不收敛、符号在长 lag 不稳定）**；单指数与幂律均不能作为稳定时间尺度律（best R² 中位 0.351）；短期（H=4 窗）方向信息与长期方向在描述性层面一致（12/12），但**一阶解析核未通过长 τ 的 FD 验证 → 短期→长期理论桥梁未确立**。诚实结论：Q1 NO（不存在可验证的稳定衰减时间尺度）；Q2 描述性 YES、验证性 NOT ESTABLISHED；Q3 NO。

**制品**：`outputs/a0p_lr_temporal_kernel_v01/`（contract/ runs/ analysis/ reports/A0P_REPORT.md hashes/ audit/）；审计包 13 文件（stage2 driver 制品正确缺席）SHA256 `59d9d3cc4500bcc20dd442258413eac8701f114e007dc13ab557ae83f873dbcf`，自校验通过。代码 `research/src/fr/a0p_lr_temporal_kernel_v01.py` + `research/scripts/a0p_{run,analyze,audit_pack}_v01.py`。A0O 及全部更早冻结制品未动。

**停止条件**：已执行——Stage-1 gate FAIL → STOP（无 relaunch、无阈值调整、无 Stage-2 GPU、不自动启动下一轮）。

## 40. A0G–A0P 结果台账（A0Q 前置更新，2026-09-18）

A0Q（Response-Control Map Exploration）启动前，按 mandate 将 A0G–A0P 已确立的正/负结果固化如下（只读台账，不改写任何历史结论）：

**正结果（已验证）**：
1. **数据扰动响应存在**——A0C/A0E2/A0F/A0F2：完整 AdamW tangent（B/C 模型）对数据 pulse 的有限时间响应稳定可预测；C（含 clip-JVP 反馈）mean rel err 0.0059–0.0219，跨 56 态普查 C beats B 168/224（A0F2）。
2. **LR 扰动响应存在**——A0O：u=log(η/η₀) 纯 θ 通道推（amendment-A2 理论修正后），n=96 Pearson +0.9552、sign 0.9688，analytic=FD。
3. **Optimizer input response 存在**——A0O Stage 2：direction accuracy 0.875 vs null 0.125，regret 0.0 vs 4.8e-4，校准 +0.942；optimizer input 是有真实预测力的新控制维度。
4. **Response 具有 phase 结构**——A0M：共模 ρ=|mean(h)|/(std(h)+1e-12) 阶段衰减（early→late ~8–10×），ARI 低（秩为近并列噪声），但 argmin 身份 stage-dependent（stage-dependent optimum 8/9 series）；A0N 闭环复现 ρ(t) phase 轨迹。

**负结果（已验证）**：
1. **Action selector 未超过 fixed 策略**——A0K/A0L/A0N：response-guided 动作选择不劣于但不优于 fixed-j1（B≥C：A0L B−C median +7.46e-3，A0N B−C median −6.78e-4，1/3 正）。
2. **Continuous mixture 未超过 fixed mixture**——A0L：response 混合 controller（C）vs fixed-j1 连续混合（B）：B 不劣于 C（2/3 seed B 优）。
3. **LR temporal kernel 不满足简单指数 memory**——A0P：脉冲核无稳定衰减时间尺度（gate FAIL：Pearson +0.7314<0.8；exp R² 中位 0.071、pow 0.306 均非可用律；CV(τ_c)=3.32）；快初始衰减（半衰期 ~4–8）+长记忆噪声尾；短期→长期理论桥梁未确立。

**对本轮的含义**：已验证的物理输入通道 = 数据构成（A0C 线）与学习率（A0O）；待测通道 = **clip 阈值**与 **weight decay**（均为 optimizer 标量输入，扰动不改 AdamW 主体）。A0K–A0P 负结果共同限定：任何未来 controller 必须证明相对 fixed 策略的 equal-wall-clock 净收益（A0K/A0L/A0N），且响应核的长 τ 一阶外推不可靠（A0P）。

## 41. A0Q Response-Control Map Exploration（2026-09-18）—— COMPLETE，LR 通道 = 唯一稳定控制变量

**任务**：基于 A0G–A0P（台账 §40）建立训练动力学响应矩阵 J=[LR, clip, WD]。非算法设计轮：寻找有物理和优化意义的控制输入（physical input channel，而非 scheduler 标定）。前置：STATUS §40 正负结果台账已按 mandate 先于一切写入。

**冻结（先于任何 GPU）**：合同 sha256 `543851ca…71d2`。三通道均为持久乘性对数扰动 u=log(x/x₀)，δ=0.05（自冻结 a0o 导入），H=4 每步施加（A0O 验证语义；A0P 已否定长 τ 一阶外推）。解析源项：LR=A0O 纯 θ 通道推（冻结 lr_pulse import-only）；WD=纯 θ 通道 −lr·λ₀·θ_k（λ₀=0.01 为 v0.2 bank 原生值）；CLIP=dc 通道 τ₀⟨b_k,ĝ_k⟩·1[preclip>τ₀]（b_k 为冻结 clip transpose 入口 dc-cotangent，τ₀=1.0 有运行时断言）。12 cell：seeds {2030,2031,2037} × steps {46,82} × streams {7400,7401}。逐通道 gate 沿用 A0O 阈值（|Pearson|≥0.8 ∧ sign≥0.9）；稳定通道规则预注册（gate PASS ∧ sign stability≥2/3 ∧ 3/3 seed 符号一致，按 median |h| 排序）。live_manifest seq 0–4（seq 4 = 结果前缺陷修复：rollout 每步读 snap["lr"] 需 want_snapshot=True）。

**执行**：launch 1 在首 cell rollout 崩溃（零有效结果，已重置）；launch 2 完成 12/12 cell，**315.0s/1800s**，CRN 12/12，0 进程崩溃（budget n_crashes=1 为 gate 标记）。

**Response matrix J 结果**（analytic/FD，n=12/cell 通道）：

| 通道 | gate | Pearson | sign | rel err | median |h| | ×LR | sign stab | seed 符号 |
|---|---|---|---|---|---|---|---|---|
| LR | **PASS** | +0.842 | **1.000** | 0.188 | 8.29e-3 | 1.00 | 1.00 | 3/3 全负 |
| clip | FAIL | +0.936 | 0.833 | 0.090 | 0（7/12 cell 恒 0）| 0.00 | 0.25 | 2/3 种符号 |
| wd | FAIL | +0.148 | 0.500 | 1.001 | 2.06e-6 | 2.5e-4 | 0.92 | 3/3 全正 |

**结论**：
1. **LR 是唯一通过预注册稳定性判据的通道**（gate PASS + sign stability 1.00 + 3/3 seed 符号一致，median |h| 8.3e-3，CV 0.49）→ **未来控制变量的推荐通道**（预注册规则内唯一候选）。响应恒为负（增大 lr 短期升高 val score 损失）与 A0O 一致。
2. **clip 响应是真实的但条件性**：仅在 clip 绑定事件中出现（5/12 cell 激活，激活时 |h|~2–6e-3 与 LR 同量级、rel err 最优 0.090、Pearson 0.936），7/12 cell 中 4/4 步 preclip<τ₀ → 通道恒 0。作为控制变量 = 事件驱动型（knockout 型）通道，非连续稳定通道；其门控由梯度范数与 τ₀=1.0 的相对位置决定。
3. **wd 响应符号高度稳定（11/12 正、3/3 seed 正、CV 0.37）但幅度 ~1e-6，比 LR 低 4 个数量级**；在冻结 δ=0.05 下 FD 完全由噪声主导（Pearson 0.148）→ 物理存在但优化上可忽略，且冻结 δ 下不可验证。

**输出（mandate 三问）**：① 稳定通道 = **LR**（唯一）；② 更适合作为未来控制变量 = **LR**（预注册规则内唯一候选；限定：A0K/A0L/A0N 已证 fixed 策略竞争力，任何闭环须 equal-wall-clock 证明净收益）；③ physical input channel 判定：clip 为事件驱动物理通道（潜力在于绑定事件期），wd 非有效控制输入。本轮未设计 controller、未改 optimizer/λ、未按结果选择通道或状态。

**制品**：`outputs/a0q_response_control_map_v01/`（contract/ runs/ analysis/ reports/A0Q_REPORT.md hashes/ audit/）；审计包 15 文件（manifest 双根 15/15 校验通过）SHA256 `bee0ea6c4bdaed185e8f58f137b69c53f97fe96022b989692df78d0304f20a2f`。代码 `research/src/fr/a0q_response_control_map_v01.py` + `research/scripts/a0q_{run,analyze,audit_pack}_v01.py`。A0O/A0P 及全部更早冻结制品未动。

**停止条件**：已执行——单阶段 map 完成即停；不自动启动下一轮。

---

## 42. A0S Response-Based Controllability Identification（2026-09-18）—— COMPLETE，PASS（有限时域可控性确认）

**问题**：response magnitude C_t=‖J_u‖ 是否含有可用可控性信息——状态响应越强，是否越容易通过干预获得收益？机制验证轮，非算法设计轮。

**设计（合约先冻结，seq 0）**：
- D1（A0S-1 主数据集，纯离线）：冻结 A0O stage1 cells H=4，n=48（seeds 2030/2031/2037 × steps 46/52/64/82 × streams 7110–7113）；解析 h_eta 与 rollout L0/L± 同记录配对。
- D2（A0S-2/3 + 规定 CSV）：A0Q 12 cells（streams 7400/7401），χ 取自冻结 A0Q cells.jsonl，rollout 损失由 a0s_run_v01 新跑（u=0×2 CRN + LR ±δ 持久 H=4，全部经由冻结 import-only a0q.run_channel_rollout，δ=0.05/H=4 自冻结 a0o/a0q）。A0Q 未持久化 rollout 损失，此为补测。
- 定义：G=L0−min(L+,L−)；C=|χ_LR|；C_J=χ_lr²+χ_clip²+χ_wd²；预测动作=−sign(χ_LR)。
- Gate 冻结：A0S-1 = Spearman ρ>0 ∧ p<0.05 ∧ LOSO pooled Pearson>0；A0S-2 = 准确率>0.5 ∧ 二项 p<0.05；A0S-3 = Model B LOSO-MSE < Model A 判 clip 有增益。

**结果（全部 seed/stream 报告）**：
- **A0S-1 magnitude→gain：PASS**。D1（n=48）：Pearson +0.832（p=2.3e-13）、Spearman +0.762（p=3.2e-10）、OLS β=0.0449（95%CI [0.0360,0.0538]）、R²=0.693；LOSO 3 fold 全部正相关（0.896/0.674/0.853），pooled Pearson +0.825、pooled R²=0.681——跨 seed 泛化成立。D2（n=12）：Spearman +0.818（p=1.1e-3）、LOSO pooled Pearson +0.480，gate 亦 PASS。
- **A0S-2 direction：PASS**。D1 46/48（二项 p=8.4e-12，混淆阵 44TP/2FP/2TN）；D2 12/12（p=4.9e-4）。响应符号预测有效动作方向。
- **A0S-3 matrix：LR 绝对主导**。C_J median share：lr=1.0000、clip=0.0000、wd≈1e-7；Model B（3 通道）LOSO MSE 4.84e-8 差于 Model A（仅|χ_lr|）3.48e-8（R² 0.196 vs −0.117）——加 clip/wd 不改善预测，与 A0Q 一致。

**判定：PASS（有限 H=4 时域）**——J_u 含有可用可控性信息，且可跨 seed 泛化。FAIL-case-2 声明随报告：短期 G 绝不可读作长期训练收益（A0P 已证无稳定时间衰减核）；且绝对增益量级微小（G/L0~7e-5）。

**纪律**：GPU 66.2s/1800s（单次 launch，0 crash，CRN 12/12）；live_manifest seq 0–2（合约→三文件→审计脚本）全部先于结果；A0O/A0Q 及更早制品只读未动；无 controller/scheduler/神经预测器/optimizer/lambda 改动；δ、H 未改。

**制品**：`outputs/a0s_controllability_identification_v01/`（contract/ runs/ analysis→runs/analysis.json reports/ hashes/ audit/）；代码 `research/src/fr/a0s_controllability_identification_v01.py` + `research/scripts/a0s_{run,analyze,audit_pack}_v01.py`；审计 tar sha256 `dcd92b8672b30bdfe35326de114e367604744a47407acf7314444f773183fc11`（17 文件，双根校验通过）。规定交付物 a0s_controllability_table.csv（10 列 × 12 行）+ A0S_REPORT.md（假设/方程/统计/图/PASS-FAIL）齐备。

**停止条件已执行**：STOP after A0S，不自动启动任何后续轮次。

---

## 43. A0T Response-guided Learning Rate Control Pilot（2026-09-18）—— COMPLETE，PASS（冻结规则）

**问题**：A0S 发现的可控性信息能否转化为有用的固定训练控制规则？机制发现后的首次算法验证。固定 pilot：不优化 controller、不调超参。

**冻结设计（合约 seq 0）**：controller：读冻结 chi_lr（D1=A0O stage1 H=4 h_eta，D2=A0Q h_lr，只读不重算），chi<0→u=+0.05，chi>0→u=−0.05，每步持久施加（η_native·e^u）；无 threshold/confidence/phase/adaptation。36 cells（D1 24=streams 7110/7111×steps 46/52/64/82；D2 12=streams 7400/7401×steps 46/82，结果无关选择规则）。三臂：A native／B 固定 +0.05／C response-guided。H∈{4,16,64} 共享一条 128-micro 冻结配方流（前缀嵌套匹配窗口）。C==B 逐位相同规则（action=+0.05 时只跑一次、别名复用）。CRN：A@H4×2 全 36 细胞通过。

**结果（GPU 1351.4s/1800s，单次 launch，0 crash）**：
- **Gate PASS**（冻结规则）：(i) arm C gain_H4 median +4.41e-4，Wilcoxon p=2.0e-10；(ii) sign consistency 35/36，binom p=1.1e-9；(iii) C 不差于 B（median 差 0；仅 2 个非别名细胞可判别，两者 C 均胜 B：+1.25e-3 vs −1.39e-3、+2.29e-4 vs −2.50e-4）。
- **C_t 三分位单调**：D1 success 0.875/1.0/1.0、median gain 2.2e-4/4.6e-4/5.9e-4；D2 全 1.0、3.2e-4/5.0e-4/5.6e-4——**可控性幅度预测 controller 成功**。
- **侵蚀**：median gain 随 H 增长（4.4e-4→9.5e-4→4.7e-3，持久扰动复利累积），但 H=16 success 降至 0.806（C）——方向优势随 horizon 波动，A0P 长尾警示成立。

**关键物理发现**：响应符号随 stream 变化——2/18 状态（D1 2030/46、2030/64）两条冻结流 chi_lr 符号相反，controller 按流取不同动作（报告修正 seq 2 已留痕，无计算改动）。C-vs-B 在这些 bank 上按构造近退化（冻结 chi<0 占 34/36），pilot 验证的是 response→action 管线端到端可用性 + 罕见正响应态上的方向保险价值。

**限制（报告强制声明）**：短期（有限 H）gain ≠ 最终训练收益（A0P 长尾非指数）；绝对增益微小；未做任何 controller 优化/超参调节。

**纪律**：live_manifest seq 0–3 全部先于相应执行；A0O/A0Q/A0S 及更早制品只读；δ=0.05、H 网格冻结；无结果驱动改动。

**制品**：`outputs/a0t_response_lr_control_pilot_v01/`（contract/ runs/ analysis.json reports/{A0T_REPORT.md, 规定 CSV 9 列×18 状态, supplementary, plots/} hashes/ audit/）；代码 `research/src/fr/a0t_response_lr_control_pilot_v01.py` + `research/scripts/a0t_{run,analyze,audit_pack}_v01.py`；审计 tar sha256 `43d075ab6dd34bf0966f70749f5573b4cc01e432c77bc09de6522588e7f4646f`（15 文件，双根校验通过）。

**停止条件已执行**：STOP after experiment，不自动启动任何后续轮次。

---

## 44. A0T-extra Response Controller Identity Analysis（2026-09-18）—— COMPLETE，判定：主身份 = Case B（全局 LR 放大方向）+ 微小状态依赖反转（不持久）

**问题**：A0T 响应引导控制是真自适应控制，还是只发现了全局 LR 缩放方向？纯离线分析，仅用冻结 A0T 制品，无新训练（GPU 0s）。

**冻结分析（合约 seq 0，5 项）**：动作分布／动作熵／轨迹切换统计／固定策略对比（仅用已有 rollout gains；固定 −0.05 只在 2 个非别名细胞存在——不可补测，识别界限预注册）／收益分解 response=fixed+adaptive。

**结果（36 cells 冻结 A0T bank）**：
- **动作分布**：pooled P(+0.05)=0.944、P(−0.05)=0.056；per seed：2030=0.833/0.167，2031/2037=1.0/0。
- **动作熵**：pooled H(a)=0.310 bits（最大 1.0）；seed 2030=0.65 bits，2031/2037=0——两 seed 为纯全局动作。
- **切换**：12 条轨迹仅 1 条有切换（D1 2030 x7110：− + − +，跨 steps 46→52→64→82，3 次=最大可能）；总切换 3 次。
- **分解**：H=4 response median=+4.41e-4 = best-fixed median + adaptive median 0（34/36 别名细胞按构造为 0）；可判别的 2 细胞引导收益 vs 固定+0.05：+2.64e-3、+4.79e-4（均为正）。
- **关键发现（seq 2 报告修正留痕）**：2 个偏差细胞在响应时域 H=4 取胜的 −0.05 动作，**在 H=16/64 反转**（固定+0.05 为正、引导−0.05 为负，两细胞皆然）——状态依赖成分不超出响应时域，与 A0P 不稳定长尾核一致。

**判定（冻结三案例规则）**：Case A（自适应显著）**不可确立**（n=2，功效不足，且缺全体固定−0.05 rollout——离线 mandate 禁止生成）；Case B（全局方向发现）**确认为主身份**（熵 0.31 bits、adaptive≈0、固定+0.05 gain 显著为正 p=7.1e-7）；Case C（response 更差）**排除**（A0T 已证 response gain 显著为正）。

**纪律**：live_manifest seq 0–3 先于执行；全部输入只读；无 GPU；无 controller 改动；无结果驱动规则变更。

**制品**：`outputs/a0t_extra_response_controller_identity_v01/`（contract/ runs/analysis.json reports/{A0T_EXTRA_REPORT.md, 3 CSV, plots/} hashes/ audit/）；代码 `research/src/fr/a0t_extra_response_controller_identity_v01.py` + `research/scripts/a0t_extra_{analyze,audit_pack}_v01.py`；审计 tar sha256 `91ff98e70001f6373de65935b3366b91ae87aed697789471060aa3bc50aa4e9c`（11 文件，双根校验通过）。

**停止条件已执行**：STOP after analysis，不自动启动任何后续轮次。

## 45. A0U Response-Guided Schedule Discovery Pilot（2026-09-18）—— COMPLETE，判定：**Case B（response 调度 ≈ 固定缩放发现；调度层面确认 A0T-extra 控制器身份结论）**

**问题**：LR 响应动力学能否揭示优于固定手工 LR 规则的**训练 schedule**？response 作**离线 schedule 发现信号**（schedule 先于训练从 response 信息生成、开环执行）——**不是**另一个在线步级控制器（A0T-extra 教训）。

**冻结设计**（contract seq 0 先于 GPU；live_manifest seq 0–2）：response 只读（A0O stage1 H=4 h_eta；12 条轨迹 = seeds{2030,2031,2037}×streams{7110-7113}，knots steps{46,52,64,82}）；冻结 schedule 规则 = 每 knot 对**之前 knots**（≤K=4，partial window，空窗→native）取均值 R_avg，R_avg<0→η·e^{+0.05}，>0→η·e^{−0.05}；W=64 rollout 自 bank step 46（窗口 47–110），段 multiplier：offsets 1–6 native、7–18 m(52)、19–36 m(64)、37–64 m(82)；checkpoint val loss @bank{58,70,82,94,110}；臂 A native / B 恒定×1.05 / C response schedule；12 细胞全自 step 46；CRN A×2/细胞（12/12 逐位）。

**Stage 1（发现能力，描述性）**：11/12 轨迹全 knot χ<0（例外 s2030×7110 = A0T 已知跨流分歧态）；轨迹内符号一致率中位 1.0；相位 |χ| 中位 early 0.0113→middle 0.0088→late 0.0076（9/12 late<early）；**跨 seed 符号一致性**：streams 7111/7112/7113 每 knot 全 1.0（7110 两 knot 0.67）——符号场是稳定 seed 通用相位属性；但幅度轨迹形 Pearson 跨 seed 对多数弱（−0.93…+0.85）——无可复现幅度 schedule 可发现，可利用内容≈符号（一比特）。

**Stage 2（发现的 schedule）**：12/12 仅切换 1 次（native→单方向），无回切；K=4 平均从未完整兑现（最大窗 3）；11/12 = [1, e^{+0.05}×3]，1/12（s2030×7110）= [1, e^{−0.05}×3]；identical_to_B_frac=0（e^{0.05}≈1.0513≠1.05 且有 6 步 native 前缀）。

**Stage 3（优化性能，12 细胞配对）**：final loss@step110——A−B median **+5.68e-3**（12/12，p=4.9e-4）、A−C median **+4.87e-3**（11/12，p=3.4e-3）、B−C median −3.3e-4（3/12，p=0.052 n.s.）；AUC——A−B +1.62e-1（p=4.9e-4）、A−C +1.35e-1（p=1.6e-2）、B−C −3.6e-2（2/12，**p=1.2e-2，C 全曲线更差**）；早期改进@step58：B +7.0e-4（9/12，p=0.034）vs C +1.6e-4（7/12，p=0.52）——C 的首段按构造为 native，让渡大部分早期收益；后期稳定性 late_std A 2.60e-2/B 2.71e-2/C 2.68e-2 无实质差异。唯一 LR-down 细胞（s2030×7110）同时输给 A（+4.4e-3）和 B（+1.2e-2）—— confidently wrong，符合 A0P/A0T-extra 对正 χ 流的预测。

**机制分解**：C ≈ B 延迟 6 优化步；B−C final 缺口 −3.3e-4 ≈ 冻结规则空窗前缀的成本，AUC 缺口即让渡的早期段。响应符号一旦为"LR-up"，K=4 池化（大多为同一批负 knot）不再产生新决策——**schedule 的全部可利用信息 = 符号比特**。

**判定（冻结 case 规则）**：**Case B**（response ≈ 固定缩放发现）——Case A 失败（B−C 非正，AUC 显著负）；Case C 失败（C−A 与 B−A 同向显著为正）。调度层面镜像 A0T-extra 控制器身份结论。

**成本**：schedule 生成 0.05s CPU（离线符号平均） vs 训练 475.5s GPU（≈119s/臂）——发现近乎免费，验证是成本。窗口增益 ~5e-3（val loss ~7.07）比 A0S H=4 时域尺度大 50–100×（持续扰动复利，同 A0T）；**A0S FAIL-case-2/A0P 长记忆告诫强制**：短窗 schedule 收益不蕴含最终训练改进。

**工程**：launch-1 cuda:1 被环境变量隐藏 → 崩溃于 model.to()（**结果前**修复：GPU 1 被外来进程占 30.6GB → cuda:0；manifest seq 2 先记后跑）；GPU 475.5s/1800s 单 launch 0 crash；单元测试 7/7 PASS（hash 前）。

**制品**：`outputs/a0u_response_schedule_discovery_v01/`（contract/ runs/{cells.jsonl,summary.json,driver.log,gpu_budget_log.json} analysis/{response_phase_analysis.csv,schedule_comparison.csv,analysis.json,plots/} reports/A0U_REPORT.md audit/）；代码 `research/src/fr/a0u_response_schedule_discovery_v01.py` + `research/scripts/a0u_{run,analyze}_v01.py` + `research/tests/test_a0u_response_schedule_discovery_v01.py`；审计 tar sha256 `540a2b69c0f09711bc891655dcb03c53a900287595e05e887fc4b681d6e233b5`（15 文件，双根校验通过）。

**停止条件已执行**：STOP after experiment——不自动启动任何后续 controller/schedule 轮次。


---

## 46. B0.6 Regime Dynamics / Switching Linear Response（2026-09-21）—— COMPLETE，判定：Headline NOT_TESTED（Q1 NOT_SUPPORTED：ψ_t 无亚稳态 regime 结构）

**问题**：B0/B0.5 之后的剩余解释类——局部动力学 regime 是否在训练中切换，导致单一 tangent/linear response 模型局部有效、全局失效？B0.6 检验 transformer 训练是否呈现**亚稳态 regime / 切换动力学**；若有，**switching linear response**（per-regime 局部映射）模型能否解释单一全局映射不能解释的内容。不训练新模型；仅自冻结 v07 checkpoints 原生回放，对冻结 B0 diag 做逐位门控（B0.6 是 B0 系列，非 A0 算法线）。

**冻结设计**（config seq 0 先于任何 GPU 结果；live_manifest seq 0–10）：9 通道 ψ_t=[log_g, log_upd, log_m, log_v, g_cos, upd_cos, log_disp, loss_slope, log_preclip]，全部 named-trainable 子集距离（B0.5 教训：raw state_dict 含 1.64M 冻结 rotary 缓存，永不进入任何距离）。Q1（支持性）：Gaussian HMM k∈{1..4}，held-out test LL 选型；须同时击败两个冻结零假设——phase null（多元相位随机化替代，n=39）与 drift null（w=9 平滑漂移移植+相位随机化残差，n=39，修正条款）；persistence 规则（run≥8、switch rate 低于 phase-null p95、run 高于 drift-null p95）。Q3a（PRIMARY/headline）：per-regime 一步线性映射 vs 全局单映射，held-out MSE 分 within/across（z_t≠z_{t+1}）；冻结规则：local 须整体更优且 across>within。Q2（支持性）：χ_H 按 dominant regime 分组置换检验。Q3b（支持性）：occupancy-mixture vs 常数 χ LOCO-CV（冻结 a0c `_metric`，1e-12 tie 地板）。描述性伴随：retained-gain 半衰期 vs 首个 Viterbi 转移（hindsight labels only）。headline_rule：Q1 NOT_SUPPORTED ⇒ Q3a/Q3b 重标 NOT_TESTED。

**结果**（6 cells=seeds{2030,2031,2032}×steps{46,82}×4 streams{7500-7503}=24 轨迹×64 步 + stress 2 轨迹；26×64 全部与冻结 B0 diag 逐位相等，gate ALL_PASS）：
- **理论门 T1–T6 全 PASS（full settings）**——HMM 恢复种植切换系统（T1）；两个零假设带在定标重复中包含 stationary/drift 数据（T2/T3）；switching-map 拟合恢复种植 per-regime 映射且 out-of-sample 胜过全局映射、deficit 集中于转移点（T4）；LOCO-CV χ 机制恰好在 occupancy 依赖时偏好 local、对 exact-constant χ 从不偏好（T5 双向重复）；冻结 a0c `_metric` 门控成立（T6）。管线已校准；以下否定非机械伪影。
- **Q1 NOT_SUPPORTED**：k=2 held-out LL 优于 k=1（−2261.0 vs −2498.5；k=3/4 过拟合 −3034.8/−3388.1；GMM 一致）`cond_multimodal: true`；但 **k=2 不击败 stationary phase null**（−2261.0 vs p95 −2114.6，`cond_hmm_beats_phase_null_ll: false`）；**persistence 方向与亚稳态相反**：mean run 25.6 步、switch rate 0.0238/step——比零假设带**更短更频繁**（phase-null run 中位 43.9/p95 64.0、switch-rate p95 0.01396；drift-null run p95 64.0）；occupancy 0.83/0.17，转移矩阵 [[0.886,0.114],[0.005,0.995]]；stress 解码一致（0.82/0.18, run 32）。连续漂移零假设在错误方向上被击败：真漂移会产生比数据**更长**的 run。
- **Q3a（PRIMARY）按 headline_rule 重标 NOT_TESTED**（独立读数 NOT_SUPPORTED）：held-out one-step MSE local 0.29609 vs global 0.29558（local 非整体更优，`cond_local_better_overall: false`）；**across-transition 测试步 n=0/360**（sticky Viterbi×16 步测试窗 ⇒ 结构退化）；谱半径：regime 0（occupancy 0.17）**1.013**（扩张，n_train=263）、regime 1 0.997、global 0.996；within-cell 变体 UNRESOLVED。switching 模型退化为单一近单位全局映射。
- **Q2 NOT_TESTED**：6/6 cells 同 dominant regime 1（occupancy 0.63–0.91），无第二组可比较。
- **Q3b NOT_TESTED**（原始信号 3/3 seed fold 偏好 local——脆弱、n=6、minority occupancy 0.09–0.38、无 regime 前提下无意义）。
- **描述性**：retained-gain 半衰期 vs 首个 Viterbi 转移 Spearman ρ=−0.369，p=0.076，n=24；0 行低于 1e-7 地板。hindsight only，不可部署。

**判定（冻结规则）**：**Headline NOT_TESTED**（Q1 NOT_SUPPORTED ⇒ switching-linear-response 无对象）。实质结论：*64 步时域、4.98M 规模下，所记录的约减态 ψ_t 无亚稳态动力学 regime 证据；单一近单位（谱半径 0.996）线性约减映射与任何 regime-switching 族拟合一样好；B0/A0P 的短期响应准确/长期失效现象不能由 ψ_t 中的离散 regime 切换解释。*

**限制**：across-split 结构退化（sticky Viterbi×16 步窗 ⇒ 0 across 测试步；更公平 within/across 检验需更长记录时域）；pilot 规模（3 seeds×2 steps×4 streams，streams 非独立 seed）；9 维投影外观测态（θ 分布几何/per-layer 谱/clip 事件组织内的 regime 结构未被排除，A0Q：clip 为事件驱动）；64 步时域界限（驻留时间≫64 步的亚稳态 NOT_TESTED，A0P 长记忆告诫逐字适用）；NOT_SUPPORTED 是对此状态表示/规模/时域的陈述，非对 transformer 训练一般地无 regime 的证明。术语纪律保持：报告无任何物理相变主张。

**MZ 含义**：（1）此规模下离散切换修正被排除——Markov-state 记忆项在 64 步无根据；（2）存留约减动力学=单一近单位映射（谱半径 0.996），与 B0.5 theta 位移+~12 步 m 核画面一致；响应预测长期发散由近单位特征值+未解析连续波动累积，非离散切换；（3）排除法边界：长期失效不由 optimizer 记忆（B0.5）或 ψ_t 离散切换（B0.6）解释——剩余候选（a）64 步窗外的连续非平稳、(b) 一步线性映射未捕获的非线性、(c) ψ_t 投影外的未解析变量；kernel/Markov 混合 MZ 模型无冻结证据链支持；（4）实践含义：未来 control/schedule 设计在此证据上应假设单一连续约减 regime（受时域告诫约束）。

**纪律**：不训练新模型（仅原生回放，multiplier 1.0，0 eval calls，0 theta captures）；26×64 逐位门控 ALL_PASS；live_manifest seq 0–10 全部先于相应 launch（8 个 pre-completion 修复：诊断容差 seq 4、REC/PSI 字段分离 seq 5、LAUNCH_ID 预算合并 seq 6、gate 打印 seq 7、analyzer 三 bug seq 8–10，全部先记后跑）；指标全程 import 冻结 `fr.a0c_adamw_response_propagation_v01._metric`，禁止重定义；冻结 a0*/b0* 模块与制品只读；预算：smoke 16+16 + 失败主尝试 64（seq 5 已记录无行）+ main 1,536 + stress 128 = **1,760 updates**，project cumulative **17,577/18,000**（per-launch merged TOTALS）；T1–T6 full settings（theory_tests_ok）；术语禁令保持（无 temperature/free energy/entropy production/equilibrium/detailed balance/field theory/phase transition）。

**制品**：`outputs/fr_b06_regime_dynamics_v01/`（config.json、live_manifest.json、B06_regime_dynamics_report.md、runs/{psi_main.csv 24×64, psi_stress.csv 2×64, smoke_checks.json, gate.json, gpu_budget_log.json, driver_summary.json, analyze.log, analysis.json, theory_tests.json, summary.json}、figures/fig1..fig7、audit/）；代码 `research/src/fr/b06_regime_dynamics_v01.py`（23/23 unittest PASS，298s，先于任何 GPU launch）+ `research/scripts/b06_{record_psi,gate,analyze}_v01.py` + `research/tests/test_b06_regime_dynamics_v01.py`。

**停止条件已执行**：STOP after experiment——不自动启动任何后续轮次。

## 47. C0 Geometry→Tracking→Transport→Readout Validation（2026-09-21）—— COMPLETE，判定：local_training_field = LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT；local_shadow_update = LOCAL_GEOMETRY_UNRESOLVED

**问题**：C0 是 B0 系之后的首个 **geometry** 轮：训练场局部线性预测是否成立——一步真实更新位移 s_t 后，probe 梯度变化 Δc 是否被一阶几何模型 Mgeom：g(θ+s)≈g+H s（再经真实 clip）预测，优于冻结梯度基线 M0？并同步检验 tracking（e_m/e_v/W 分解）、transport（realized s_t 与解析位移一致性）、readout（V_diag 台账 + K_V）。**不设计算法、不进入 C1/controller**。

**冻结设计**（config.json sha256 `e00ce24…` seq 0；live_manifest 4 events）：2 cells（R=reversal case stream 7110 / C=non-reversal control stream 7111，同 checkpoint step 46）× 3 arms（A u=0 / Bp +0.05 / Bm −0.05，持久每步）× 64 真实更新 + 24 anchors（t_pre∈{0,4,16,48}×3 arms×2 cells）。每 anchor：dev/audit 双 bank 各 8 probes（CPU 验证 16 micros 唯一零碰撞）→ V_diag+q 单次 eval-mode 前向+反向 → clip+step 得 realized s_t → snapshot post/rewind pre → dev HVPs(tangent s_t) → **predictions 先写盘（order_index）** → restore post → audit endpoint 梯度 → y 统计 + half-bank 前 4/后 4 固定 ID 分裂。预测模型：M0（Δc=0，E≡1）／Mgeom=g+H s 再真实 C 重 clip（PRIMARY）／MnoJ 消融；shadow U-map 经真实 clip/moments/P；stop-v 单零梯度成本变体。判定规则（冻结）：每 cell ≥6 resolved anchors 覆盖 ≥2 arms 与 ≥2 个 t、Mgeom E 均 ≤0.8（signal-resolved anchors 上）、dev/audit half-bank E_Mgeom ≤1；resolved=‖ȳ‖>3√(SE²+floor²) 工程分辨率门（非显著性）；信号不足标 UNRESOLVED，不改阈值/不换 case/不筛选救结论。

**结果**（main 384 updates 全完成；诊断 960 grad-evals/192 HVP/2064 forwards，caps 内）：
- **formula_implementation PASS**：T1–T9 CPU 理论测试 20/20（0 updates；4 项测试开发期修复披露，含核心发现——tangent 必须 consume CLIPPED 梯度 C(g)，raw-g 传递多步漂移 13%）。
- **smoke 门（含抢救披露）**：两次 smoke 尝试因 runner bug 损失 33 updates（9+24，全部入账，用户批准 +9 超额，本轮 417 总额）。diag-identity 以零成本重建并 **PASS**：(a) attempt-2 的 smoke_steps.jsonl diag-off vs diag-on 行逐字段相等（lr/preclip/clip/s_norm 全 0 差）；(b) 外部逐位门——main cell-R arm-A 对冻结 B0 stress_diag native 行（2030/46/7110）h=1..64：lr_native/lr_applied/preclip_norm/clip_active **逐位 0 差**，s_norm vs update_norm 最大 1.25e-8（归约精度容差 1e-6 内）。HVP proxy-vs-native gap gate PASS（5.4e-3）。
- **floor 超标披露**：smoke/floor 阶段把跨 probe 对混入 floor，违反冻结定义（"repeated IDENTICAL evaluations"）→ pooled floor grad=1.65；零 update `--phase floor2` 按同 probe 重复配对重测 = **全量 0.0**（逐位确定）。修正 floor 为 operative，阈值未动，双值均披露。
- **local_training_field_prediction = LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT**：每 cell 7 resolved anchors（t=0×3 arms + t=48×3 + Bp@t=16），覆盖 3 arms/3 t；mean E_Mgeom 0.42–0.44（≤0.8 ✓）；**但 C/Bp/t=16 的 last-4 half-bank E=1.166>1 → 规则失败**。干净 resolved anchors（t=0/48）E_Mgeom≈0.35–0.45 = 对冻结梯度基线 55–60% 误差削减；t=4（post-pulse）全部 below-gate 且 E>1；E_MnoJ≥E_Mgeom（clip-JVP 项贡献真实）。**敏感性披露**：裁决系于单个 marginal anchor（y_norm 超 3·SE 仅 ~7e-3），无 re-selection。
- **local_shadow_update_prediction = LOCAL_GEOMETRY_UNRESOLVED**：仅 t=48 anchors resolved（3/cell），<6 且仅 1 个 t；t=0 U-map 信号低于 3·SE 门。E_stopv≈E_Mgeom_U（stop-v 消融不改变信息）。
- **tracking_signal_resolution PASS**：24/24 anchors 记录；W_field+W_track+W_wd≡W_total 恒等残差 max 2.1e-7；transport_err_rel ~1e-5（realized s_t 与解析位移 −η·r̂−η·wd·θ 一致）；e_v 可忽略（A0Q wd 判定一致），e_m~0.37–0.52。
- **readout_accounting PASS**：telescope 残差 ~1e-16（378 K_V 行）；K_V 负 90/378（精确代数余项，非构造保证正）；V_eval 独立报告不与 V_diag 混同。
- **A0T 交叉验证 = AXIS_MISMATCH（非轨迹分歧）**：全部 15 对比超 5e-4 容差但呈系统 ~0.106 常数偏移——溯源证实 A0T cells.jsonl losses 读在完整 v_score 轴（a0q.run_channel_rollout→val_score_tensors），C0 v_eval 是独立 64 样本 split、v_diag 是 v_score 前 8 ID，二者无共享轴，5e-4 绝对比对 post hoc 不可满足；cell-R arm-A 轨迹同一性由 B0 逐位门独立确立。order discipline（predictions 先于 audit 梯度）24/24 ✓；E_M0≡1 24/24 ✓。
- **3 项固定 NOT_TESTED**：future_H16_H64_prediction / adaptive_training_value / energy_benefit（另最终训练改进未声明）。

**判定**：**local_training_field = LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT**（信号充分但 half-bank 规则失败）；**local_shadow_update = LOCAL_GEOMETRY_UNRESOLVED**。实质读数：一步几何预测在干净 resolved anchors 上将训练场梯度变化误差减半以上，但 dev→audit 半库稳定性在 marginal t=16 anchor 上破缺，按冻结规则不给予 pilot 级支持；shadow-update 通道信号不足以裁决。

**限制**：64 步时域、4.98M、单 checkpoint（step 46）×2 streams；HVP 为 math-SDPA 代理（native flash 无 double backward）；t=4 post-pulse 区 anchors below resolution gate（信号存在但 <3·SE）；floor 超标修正已披露；round-total logged forwards 2428 超 2400 cap 28（两个零 update 抢救 floor 过程，披露）；A0T 绝对轴交叉验证不可行（axis mismatch）；NOT_SUPPORTED/UNRESOLVED 仅限该 pilot 规模/时域非一般性证明。

**纪律**：用户批准 417 updates（408+9），project cumulative **17,994/18,000**（runner 静态基数行少记 33，已 reconcile 披露）；失败尝试 33 全部入账；order discipline、predictions-before-audit 全过；术语禁令保持；停止条件执行——不自动进入 C1/controller。

**制品**：`outputs/fr_c0_gttr_validation_v01/`（config.json、live_manifest.json、audit.md、theory_tests.json、smoke_checks.json、smoke_checks_floor2.json、budget_log.jsonl、cell_metrics.csv、readout_balance.csv、local_geometry_predictions.csv、shadow_update_predictions.csv、tracking_diagnostics.csv、summary.json、REPORT.md、runs/{step,smoke_steps,anchor_records,anchor_predictions,smoke_anchors,smoke_predictions}.jsonl、probe_banks.json、figures/fig1..fig5）；代码 `research/src/fr/c0_gttr_validation_v01.py`（20/20 unittest PASS）+ `research/scripts/c0_{run,analyze}_v01.py` + `research/tests/test_c0_gttr_validation_v01.py`。

**停止条件已执行**：STOP after C0——不自动启动 C1 或 controller。

---

## 48. C0X 离线规则复算 + C1 GTTR Probe Validation（2026-09-21）—— C0X COMPLETE（ERRATUM_REQUIRED）；C1 COMPLETE（field = LOCAL_SUPPORT_ON_SELECTED_ANCHORS；shadow = UNRESOLVED）

**问题**：C0X（协议第一轮，纯离线）：复算 C0 冻结判定规则是否被正确执行——(X1) C/Bp/t16 last-4 E=1.166 是单点比值还是聚合门？(X2) floor 修正（跨 probe 配对违规 → same-probe floor=0.0）后 field/shadow/half-bank 判定是否一致重算？(X3) 同一 V_diag 上三臂 realized gain 与累计 W/K 在区间 [0,4)/[4,16)/[16,64) 是否记账平衡？(X4) 账本与审计不确定性是否变动？C1（协议第二轮，GPU）：在严格防泄漏的新 probe 库上验证 GTTR 局部几何预测的**跨库独立可预测性**——dev 库 HVP 一阶预测 mean(P_D) 能否预测 audit 库实测 endpoint mean，优于零基线 M0；同 probe 近似误差 vs 跨库波动误差分解；stop-v ablation 检验 v 通道是否承载实信号；固定读出方向投影。仍是 GTTR 理论探索，非 controller/optimizer 开发；完成本协议后停止，不进入 C2。

**冻结设计**：
- C0X：零 GPU、零新测量，只读 C0 冻结制品；按协议原文（§10.2 聚合 SSE 比门、same-probe floor 定义）复算；原 verdict 保留不覆盖，规则执行错误以 erratum 追加。
- C1（config seq 0 先于任何 GPU 结果；anchors/预测组/判定规则全冻结）：复用 C0 冻结 2 cells（R stream7110 / C stream7111，step 46）；每 cell 新 dev+audit probe 库各 32（seed 2026092103/2026092104，与 C0 库及 rollout 批次按样本 ID 全不相交）；计划 6 anchors（A/t0、Bp/t16、A/t48 ×2 cells），**实际对称执行 4 anchors**（A/t0+Bp/t16；全 6 anchor 因首跑失败计入后 432>416 HVP cap，算术披露）。K=8/16/32 固定前缀 + 固定 ID 两半（first16≡K16 前缀，同一组 probe，披露）；每 anchor：dev/audit g0 → 真实 clip+step 得 realized s_t → q+=grad V_diag(θ+s) 主读出方向（audit 终点前固定）→ dev HVP 预测 **mean(P_D)、mean(Q_D) 先 sha256 落盘 + order_index** → 之后才读 audit HVP/g1（CONDITIONED，不回填不重选）；U 逐 probe 真实 clip/moments/bias-correction/P/wd，从不把均值梯度喂进 Adam 非线性（T4）；stop-v ablation（P 冻结自 audit 均值 clip g0）；分解 e=e_bank+e_local 保留 cross 项。判定（冻结）：LOCAL_SUPPORT_ON_SELECTED_ANCHORS ⟺ 两 cell K32 聚合 E≤0.8 且两半库 E≤1 且每 cell ≥2 distinct t 可分辨；resolved=‖ȳ‖>3·SE（floor=0）；E-gate 过但分辨前置不成立 = UNRESOLVED；E-gate 值不过 = RULE_NOT_MET。预算：≤192 updates / ≤1,400 grads / ≤416 HVPs / ≤3,200 micro-forwards（round-total，失败/recheck/salvage 全计，不除旧 18,000）；GPU wall ≤90 min。

**结果**：
- **C0X**：X1 = **ERRATUM_REQUIRED**——C0 分析器把冻结的聚合半库门（同一 resolved 集合上的 SSE 比）实现成逐点门，C/Bp/t16 单点 1.166 单独触发 NOT_SUPPORTED；按原规则复算聚合 E：field g1/g2 门通过（R=0.5291、C=0.4676 ≤0.8），g3 半库聚合门因 C0 未保存半库 SSE/TSS = **DATA_MISSING**；原判定保留于 C0 summary.json，erratum C0X-X1-01 追加。X2 = 一致重算（floor2 same-probe 全 0.0；全部 resolved 标志离线重算）。X3 = 逐区间记账平衡（区间恒等残差 ≤1.4e-17；63 步累计 gain 仅 0.006–0.020 = V_diag 的 0.1–0.3%；R cell Bp 收益"先小负后转正"，与 A0U 延迟兑现一致）。X4 = 账本不变（417 approved，17,994/18,000）；A0T−v_eval 轴差维持 AXIS_MISMATCH。
- **C1 smoke 全 PASS**：库不相交（dev∩audit、vs C0 库、vs rollout 批次）✓；diag on/off 全 state sha256 一致（8 updates）✓；same-probe floor 四量全 0.0 ✓；HVP 代理 gap max 4.5e-3 < 1e-2 门 ✓。
- **C1 主结果**：(a) **field 跨库预测 LOCAL_SUPPORT_ON_SELECTED_ANCHORS**——K32 聚合 E：R 0.505 / C 0.398；两半库 E≤0.700；每 cell 2 个 distinct t 可分辨。GTTR Geometry/Transport 段最强正向证据。(b) **shadow = UNRESOLVED**——E 门值全达标（K32 R 0.667/C 0.585，半库 ≤0.980），但 t0 处 shadow 目标所有 K≤32 低于 3SE 分辨下限（比值 ≈2.8–2.9），"≥2 distinct t"前置不成立 → 当前测量未解决（非理论无效）。(c) **SAME_PROBE 8/8 优于 M0**；field e_local/‖ȳ‖：t0 6.6–10.3% → t16 25.3–33.9%（shadow：15.8–21.6% → 41.9–51.9%）；跨库 e_bank 是同 probe 误差的 2–20 倍（t0 最大）——独立库均值预测误差主体是 probe 间波动，非局部模型缺陷。(d) **ERROR_DECOMPOSITION MIXED**：恒等残差 ≤8.7e-19，cross 项占 e² 的 1.8–19.4%、符号不定（不做 norm 百分比归因）。(e) **stop-v**：冻结 P 使 shadow 增量范数系统性 +11–33%（t16 最大），‖Z_full−Z_stop‖ 达 full 范数 0.5–1.0 倍，方向结构保持（inner 全正）——v 通道承载实信号。(f) **AdamW 耦合方向依赖**：D_i 在 q+ 上投影 C/t0 +0.158、R/t0 +0.153、C/t16 +0.050、**R/t16 −0.064 反号**。(g) K 增大不单调（R field E 0.68→0.70→0.50），符合事先声明；K32 相对 K8 改善稳定。
- **C1_EXECUTION = PARTIAL**（执行/防泄漏/预算全干净；覆盖 4/6）：order discipline 全过（预测 order_index 0<1,2<3,4<5,6<7，预测 mtime < records mtime，sha256 先于任何 audit 读）；分解恒等 ≤8.7e-19；失败首跑（R/A/t0 的 sha KeyError——sha 曾覆盖全部 MAIN_GROUPS，但 Y_A/P_A/Z_A/Q_A 只在 stage 3 累积；协议 §7.1 冻结预测组 = mean(P_D)、mean(Q_D)，修复后限定）按协议计入 `runs/failed_attempts.json`（1 update/66 grads/32 HVPs/194 micros）。

**判定**：C0X：X1 ERRATUM_REQUIRED（其余三项 UNCHANGED/一致重算）；C0 原 verdict 保留不覆盖，"C0 field 局部几何不支持"作为科学结论被 C0X+C1 削弱（依据是规则误用；C1 在新独立库给出 field 正面支持）。C1：field = **LOCAL_SUPPORT_ON_SELECTED_ANCHORS**（4-anchor 已测集）；shadow = **UNRESOLVED**（分辨前置不成立）；SAME_PROBE = 8/8 优于 M0；ERROR_DECOMPOSITION = MIXED；future_H16_H64_prediction / adaptive_training_value / energy_benefit = **NOT_TESTED**。

**限制**：A/t48 两 anchor 未测（预算算术，late-time 泛化未检验）；shadow per-probe 残差 SE 未保存（保存均值级 e_local，field 通道已保存）；first16 半库 ≡ K16 前缀（半库新增信息 = last16）；stop-v 方向 cos 为均值级描述量（Jensen 伪影披露）；4.98M、单 checkpoint、64 步时域；HVP 为 math-SDPA 代理；shared GPU（vLLM 在场，host 卸载纪律）；R/Bp/t16 field K=8 单点不可分辨（0.188 vs 0.204）——门只用聚合，不影响判定。

**纪律**：防泄漏顺序（g0 双库→真实 step→dev HVP+预测先存→audit 终点）全过；预测组限定冻结 §7.1（bug 修复透明记录）；stop-v 只作 ablation 非 optimizer 候选；旧 18,000 账本不动（17,994 原样），C1 独立列账不相除；失败/recheck/salvage 全计入（round-total）；术语禁令保持（无 temperature/free energy/entropy production/equilibrium/detailed balance/field theory/phase transition）。**预算（C1 round-total）**：updates 45/192、grads 605/1,400、HVPs 304/416、micro-forwards 1,801/3,200、GPU wall <25 min/90 min；shadow 纯代数 15.6 s 单独计数。repo 根目录留有 smoke/main/main2 三个 redirect 日志（`outputs_fr_c1_*.log`，已在审计轨迹披露）。

**制品**：C0X `outputs/fr_c0x_offline_explanation_v01/`（REPORT.md、audit.md、config.json、rule_recalculation.csv、signal_resolution.csv、gain_balance_by_interval.csv、ledger_recheck.json、target_axis_map.csv、verdict_comparison.json、fig_c0x1/2）；C1 `outputs/fr_c1_gttr_probe_validation_v01/`（REPORT.md、GTTR_EVIDENCE_UPDATE.md、config.json、smoke_checks.json、budget_log.json(l)、state_provenance.csv、probe_summary.csv、error_decomposition.csv、aggregate_E.csv、fixed_projection.csv、update_feedback.csv、verdicts.json、runs/{anchor_predictions,anchor_records,step_records,probe_banks,failed_attempts}.json(l)、fig_c1_1/2/3）；代码 `research/src/fr/c1_gttr_probe_validation_v01.py`（GroupAcc host-fp64 化）+ `research/scripts/c1_run_v01.py` + `research/scripts/c1_analyze_v01.py` + `research/tests/test_c1_gttr_probe_validation_v01.py`（9/9 PASS）。

**停止条件已执行**：STOP after C0X+C1——不自动进入 C2、架构搜索、HMM、长记忆核、完整 Hessian 扫描或长期训练。下一步唯一最有价值的问题（GTTR_EVIDENCE_UPDATE）：在 A/t48（A0U 延迟收益区）重复同一冻结测量（2 anchors × 64 HVP + 37 updates，独立成轮）。

## 49. C1-Late 遗留 Anchor 补齐 + 旧数据离线复核（2026-09-21）—— COMPLETE，判定：late 四 anchor×channel 全部 LATE_ANCHOR_RESOLVED_AND_IMPROVED；COMBINED6 field PASS（3 distinct t）、shadow PASS（2 distinct t {16,48}，带纪律保留）

**问题**：按协议 `FR_C1L_GTTR_LateAnchor_DataFirst_Claude_Code.md` 完成两件事。(O1–O3，零 GPU)：旧 C0X/C1 关键数据离线复核——不导入生产聚合 helper，用小参考计算直接从原始 SSE/TSS 重算旧 4 anchor 的逐 anchor/channel/K8/K16/K32/first16/last16 统计、聚合门、resolved 标志与 distinct-t，交叉核对已发布 aggregate_E.csv，并列出缺失项（DATA_MISSING/NOT_SAVED）。(Task 2，GPU)：执行原 C1 因预算算术遗漏的 R/A/t48、C/A/t48 两个 late anchors（native A 臂、更新前 local t=48、全重放、同一 probe 库/顺序/随机 recipe、泄漏顺序与冻结判定规则全部沿用 C1）。仍属 GTTR 理论验证，不开发 controller/optimizer；完成本轮后停止。

**冻结设计**：状态恢复规则先行——现场无 t48 snapshot（C1 仅存 t0/t16），采用全原生重放 49 updates/cell（合计 98 ≤ 上限），标量级恒等可声明、完整 theta/m/v/RNG 逐位一致不可声明。每 anchor：dev+audit g0 → 真实 clip+step → q+=grad V_diag(θ+s) 固定 → dev HVP 预测先 sha256 落盘 + order_index → 后读 audit（CONDITIONED，不回填）；U 逐 probe（非 U(mean(g))）；固定 K=8/16/32 与固定 ID 两半（first16≡K16 前缀，披露）；stop-v 仅诊断不写回；分解 e=p_D−y_A、e_bank、e_local 保 signed_cross=2⟨e_bank,e_local⟩ 与恒等残差，不做 norm 百分比归因。新数据保存充分统计（sum_sq_norm、centered_SS、inner、sse/tss 全 K/half、shadow 配对残差、stop-v 每 probe norm/cosine）。判定冻结：同 C1（E≤0.8、半库≤1、≥2 distinct t resolved；floor=0；E 过而分辨不成立=NUMERICAL_UNRESOLVED）；三 scope：OLD4（重算核对）、LATE2（单 anchor 判定）、COMBINED6（固定 3-anchor 集合 A/t0+Bp/t16+A/t48，不丢点）。预算独立建账：≤128 updates / ≤600 grads / ≤192 HVPs / ≤1,800 micro-forwards / 45 min；旧 17,994/18,000 与旧 C1（45/605/304/1801）不动、不相除。

**结果**：
- **O1 复核**：4 旧 anchor 逐 anchor/channel/K/half SSE/TSS/E 独立重算与已发布 aggregate_E.csv **完全一致**（old_aggregate_recheck 全 match）；resolved 标志与 probe_summary.csv 0 不一致；63 步端点/索引核验（h∈2..64，区间 [0,4)=[1..4]、[4,16)=[5..16]、[16,63]=[17..63]，无 h=65）；C0X erratum 维持（不追认完整 PASS）。
- **状态恢复与恒等**：全重放 48+48 updates 达 pre-update t=48（anchor h=49，cumulative step 94，t_after=95）；与 C0 原生 A 臂 h=49 逐步行对比，v_pre/q_pre_norm/s_norm/eta_t 四项 × 2 cell 全部 diff=0.000e+00（**标量级**恒等，非完整态逐位）；泄漏顺序 late 0<1、2<3；冻结预测 per_K mean 与实测 group_stats 逐位一致（12/12 组，diff=0.000e+00）。
- **late 测量（Table A，K32）**：R field target 0.048852±0.007936 resolved（ratio 2.05），E=0.2364（K8 0.4016/first16 0.3195/last16 0.3171）；R shadow target 0.001896±0.000308 resolved，E=0.2478；C field target 0.061931±0.010639 resolved，E=0.2382；C shadow target 0.002153±0.000346 resolved，E=0.2224。四 verdict = **LATE_ANCHOR_RESOLVED_AND_IMPROVED**。
- **COMBINED6（固定 3-anchor 集合）**：field R E=0.4914、C E=0.3618，各 3 distinct t → PASS；shadow R E=0.6646、C E=0.5706（半库 ≤0.954），distinct t={16,48} → PASS。纪律保留：(a) late 通过**不回溯**改变 t0 shadow NUMERICAL_UNRESOLVED 事实；(b) {16,48} 可分辨**不是**独立总体确认（同一 R/C 对、相邻 t、同批 probe 库）。LATE2 scope 因单 t 构造上 distinct-t=1，scope 级标 NUMERICAL_UNRESOLVED，仅作单 anchor 判定载体。
- **分解（Table C）**：field bank_SS/total_SS 0.837（R）/0.973（C），local 0.113/0.053，|signed_cross|/total_SS 4.9%/2.6%；恒等残差 ≤5.5e-20。field 同 probe 配对残差 audit mean±SE：R 8.512e-3±1.93e-4，C 1.017e-2±2.79e-4。
- **stop-v（诊断）**：Z audit norm ratio 1.222（R）/1.236（C）、mean cosine 0.875/0.867；Q dev ratio 1.303/1.290——v 通道增量信息维持。d_c_to_u q+ 投影 R −0.0638、C −0.0887。clip activity：dev 18/32 全 active（0 crossings）；audit R 3/2/1 crossings、C 无。
- **q+ 固定读出（Table D）**：|q+| 0.8927（R）/0.8885（C）；field pred/target 投影 −8.63e-3/−9.11e-3（R）、−6.17e-4/−6.36e-4（C）；C 的 Q_D/Z_A 等 |mean|≤2·SE 不计显著读数（预注册规则）。

**判定**：O1 = RECHECK_PASS（与已发布值一致，erratum 维持）；O2 = 轴/索引核验一致（AXIS_MISMATCH 维持）；O3 = 缺失清单落盘。late anchors = **LATE_ANCHOR_RESOLVED_AND_IMPROVED（4/4）**；COMBINED6 field = **PASS（3 distinct t）**、shadow = **PASS（2 distinct t，带 (a)(b) 纪律保留）**；future_H16_H64_prediction / adaptive_training_value / final_training_improvement / energy_benefit = **NOT_TESTED**。

**限制**：COMBINED6 的 shadow distinct-t 仅 {16,48}（t0 仍不可分辨，属旧事实非新证据）；标量级恒等 ≠ 完整态逐位；first16≡K16 前缀；stop-v cos 为均值级；4.98M、单 checkpoint、单 seed 对；HVP=math-SDPA 代理；late E 比较（t0 R 0.2799/C 0.2191 → t16 R 0.5203/C 0.4457 → t48 R 0.2364/C 0.2382）非单调，不得外推为趋势定律。

**纪律**：三笔账分开（旧 17,994/18,000 | 旧 C1 45/605/304/1801 | 本轮 106/279/144/835）；失败/smoke/复核全入账；不混轴、不减常数对齐；不丢点聚合；禁语保持。**预算（本轮 round-total）**：updates 106/128、grads 279/600、HVPs 144/192、micro-forwards 835/1,800、主过程 GPU wall 231 s/45 min；shadow 算术 768 steps/8.9 s 另计；训练伴生 micro-forwards 196 另披露。根目录 redirect 日志 `outputs_fr_c1late_{smoke,main}.log` 两个已披露。

**制品**：`outputs/fr_c1_late_completion_v01/`（config.json、smoke_checks.json、resource_log.json(l)、state_and_probe_provenance.csv、late_probe_statistics.csv、old_probe_statistics.csv、late_error_decomposition.csv、late_fixed_projection.csv、late_update_feedback.csv、aggregate_gate_inputs.csv、criterion_trace.json、old_anchor_recheck.csv、old_aggregate_recheck.csv、old_data_missing.csv、KEY_DATA_FOR_REVIEW.md、REPORT.md、GTTR_EVIDENCE_UPDATE.md、runs/{probe_banks,smoke_steps,late_predictions,late_records,late_steps}.json(l) + 两 cell post-state pt）；代码 `research/scripts/c1_late_run_v01.py`（run+analyze）、`research/scripts/c1_late_analyze_v01.py`、`research/scripts/c1_late_report_v01.py`、`research/tests/test_c1_late_reference_v01.py`（2/2 PASS）。

**停止条件已执行**：STOP after C1-Late——不自动进入 C2、controller、新 optimizer 或任何新科学问题。

### 49a. C1-Late 字段核对勘误（2026-09-21 追加，0 GPU）

按指示对 C1-Late 报告做离线字段核对（`research/scripts/c1_late_fieldcheck_v01.py`，
全部数字由已存充分统计重算）：Task1 独立复算 target/prediction 的 mean_norm² /
mean_sample_norm² / centered_SS / SE_vec，恒等 ‖x̄‖²+centered_SS/n=mean_sample_norm²
在 48/48 行精确成立；Task2 裁决 field 残差 SE 为 SE_of_norm（per-probe 库内范数列），
与 mean(norm) 自洽（Jensen 间隙 ≥0），无数值矛盾，系表头标签混用；Task3 确认
Table D "full_minus_stop" 实为 stop_mean，重建 full/stop/paired 三列（恒等差 ≤3.3e-20）；
Task4 将 Table C 拆为 norm(mean)/mean(norm)/ratio_of_mean_norms(1.0486/1.0432)/
mean_of_probe_ratios(1.2222/1.2363) 四统计量，full/stop 归为 shadow 专用；
Task5 SNR 拆为 signal_over_SE(5.82–6.22) 与 ratio_to_3SE(1.94–2.07)，resolved 全复核 True，
COMBINED6 十二行 E 逐位不变。**主判定全部保持不变；本次为标签/定义级勘误（E1–E9，
见 outputs/fr_c1_late_completion_v01/FIELD_CORRECTIONS.md），原报告保留并追加 ERRATA。**
停止条件维持：不进入 C2。
