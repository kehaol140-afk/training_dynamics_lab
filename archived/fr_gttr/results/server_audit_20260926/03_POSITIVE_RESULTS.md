# 03 — 可靠正结果（全部有冻结制品证据）

标注：RAW_VERIFIED = 审计时直接读取制品 summary/verdicts 核验；REPORT_VERIFIED = 报告正文核验。

## P1. AdamW 完整切线 C 是参数扰动因果响应的高精度预测器（项目第一正结果）

- 证据：`outputs/a0_fr_audit/response_propagation_v01/`（A0C）：3 锁定态 × H{1,2,4,8} × 8 流；
  all-12 口径 **B R²=0.8388 / C R²=0.9977（mre 0.0274）**；H>1（9 行）C 9/9 胜 B（C mre 0.0219 vs B 0.2053）。RAW_VERIFIED
- 全 bank 普查：`outputs/a0_fr_audit/cross_partition_extension_v01/`（A0F2）：56 态 × 224 行，
  **C 胜 B 168/224（56 平=H1 结构零对照，0 负）**，7/7 训练阶段全正；C pooled R²=0.99781、mre 0.005872。RAW_VERIFIED
- 跨实验位级复现链：A0E2 同 run 复现 A0C gap=0.0；A0F2 对 A0F 共享行 gap=0.0。RAW_VERIFIED
- 语义边界：C=零脉冲局部光滑切线；χ_fd=注册有限 ε 中央差分；math-SDPA HVP 代理（native flash 无 double backward，math-vs-native gap 0.005–0.016）。

## P2. 机制阶梯可分解（A0E2）且其中 clip 段有因果干预证据（G3）

- A0E2（RAW_VERIFIED）：H>1 pooled **B 0.2053 → BH_fixed 0.1366（raw curvature +0.0686）→ C 0.0219（clip-JVP +0.1148）**；
  三段 gain 加总恒等（偏差 8.9e-16）；H=1 恒等 gap=0（24/24 流精确）。
- G3 clipping intervention（RAW_VERIFIED）：26M 真正 no-clip 重训（clip 调用计数 0）使
  **gain_clip 从 +0.094 精确崩溃到 +0.000（BH_fixed≡C 位级，27/27 单元）**；
  Δ_clip=+0.09398，层次 bootstrap 95% PI [+0.0086, +0.2938]，P(Δ>0)=0.99985。
  判定 MECHANISM_CONFIRMED——clip→response 因果链的干预证据。
- 注意：curvature 段只有相关性证据（G3 中 no-clip 下 curvature gain +0.158 反而更大，即它是独立机制，但其因果干预未做）。

## P3. 跨规模复制（controlled ladder，scale-only）

- G1a 26.68M（5.354×）：C R²=0.99958、H>1 paired **18/18/0**、mean +0.16861。`outputs/generalization_g1a_26m_v01/`。RAW_VERIFIED
- G1b 100.78M（20.225×）：C R²=0.99949、**18/18/0**、mean +0.10503。`outputs/generalization_g1b_100m_v01/`。RAW_VERIFIED
- 三规模机制表（冻结登记）：C pooled R² 均 ≥0.9977；H>1 全胜；curvature gain +0.057~+0.073；clip gain +0.015~+0.115（观察，非定律）。

## P4. 跨数据分布复制

- G2 D0→D1（100% vs 73% 截断、shingle Jaccard 1.6e-7、HVP norm ~7×）：**C R²=0.99995、mre 0.00260、18/18/0**。
  `outputs/generalization_g2_data_distribution_v01/`。RAW_VERIFIED

## P5. H=1 局部 Jacobian 精确成立；残差=有限 ε 伪影

- H1（RAW_VERIFIED）：ε/4 相对误差 0.000146–0.018989（三态单调收敛）；ν 反常归因于冻结 v 近似（ratio 0.0124）。
- A0D（RAW_VERIFIED）：ε{1,1/2,1/4} 池化 mre 0.0219→0.0089→0.0045（ε/4 R²=0.99982）；FD 收缩 7/9 cell；
  **H≤8 不需要 Mori–Zwanzig 记忆核**（残差主要是有限 ε 伪影）。

## P6. LR 是唯一稳定的 optimizer 控制通道，且响应是可计算导数

- A0O（RAW_VERIFIED）：u=log(η/η₀) 持久扰动，Stage1 n=96 **Pearson +0.9552 / sign 0.9688 / CRN 全过**；
  Stage2 direction accuracy **0.875**（null 0.125）、median regret 0.0（null 4.8e-4）、校准修正后 +0.942。
  理论：h_eta = Σ⟨p_θ, lr·e_lr⟩，e_lr=−adaptive(c)−wd·θ（纯 θ 通道，A2 修正）。
- A0Q（RAW_VERIFIED）：三通道 map——LR PASS（Pearson 0.842/sign 1.000）；clip FAIL（事件驱动，7/12 cell 恒 0）；
  wd FAIL（幅度低 4 个量级）。**LR 为预注册规则内唯一控制变量候选。**

## P7. 响应幅度含短时域可控性信息

- A0S（RAW_VERIFIED）：|χ_LR| vs 实现干预增益 D1 n=48：Pearson +0.832（p=2.3e-13）、Spearman +0.762（p=3.2e-10）；
  LOSO 跨 seed 泛化（3 折全正，pooled +0.825）；方向 46/48；LR 通道占 C_J 份额 1.0。
  A0T：C_t 三分位与 controller 成功率单调（0.875/1.0/1.0）。
  强制 caveat：短期 G（G/L0~7e-5）不得外推最终训练收益（A0P）。

## P8. 闭环 response 干预在短窗口内有小幅净正收益

- A0K（RAW_VERIFIED）：等步 A−C median **+7.44e-4 nats**（K=20，6 seeds；K=10 +1.64e-3，3/3 正）；等墙钟尺度条件成立（基线未失步）。
- A0T（RAW_VERIFIED）：sign 控制器 H=4 median gain **+4.41e-4**（Wilcoxon p=2.0e-10；sign consistency 35/36，p=1.1e-9）。
- A0H-2.0 A″（RAW_VERIFIED）：独立验收 V_eval Gain **+1.467e-4**（PI [+8.7e-5,+2.1e-4] 探索性；5/6 seed 正）。
- 共同限定：收益来自**干预本身**（固定动作也能拿到中位收益）；选择器无超额价值（见 N 系列）；等墙钟赢是尺度条件。

## P9. 响应存在相位结构（幅度衰减 + argmin 身份）

- A0M（RAW_VERIFIED）：共模 |mean|/std median **1.85**；共模 early→late **~8–10× 衰减**；argmin 身份 phase 依赖 8/9 series；
  单 h 效用 Pearson −0.866（w=1 最优，窗口平均反而退化）。A0N 闭环复现 ρ(t) 相位轨迹。

## P10. GTTR field 通道：一步几何输运预测跨库成立

- C1（RAW_VERIFIED）：新独立 probe 库 K32 聚合 E：R 0.505 / C 0.398；半库 ≤0.700；2 distinct t → **LOCAL_SUPPORT_ON_SELECTED_ANCHORS**；
  SAME_PROBE 8/8 优于 M0；误差分解恒等 ≤8.7e-19。
- C1-Late（RAW_VERIFIED）：late t48 anchors 全部 RESOLVED_AND_IMPROVED（E 0.222–0.248）；COMBINED6 field PASS（3 distinct t）、
  shadow PASS（t={16,48}，带纪律保留）。GTTR Geometry/Transport 段的最强正向证据。

## P11. 工程与方法学正资产

- 位级 replay 基础设施（E0/E1）+ CRN 协议：全部实验的身份门（S1/S2）与逐位复验链的来源。RAW_VERIFIED
- 共享伴随（A0H-1.5/2.0）：FP32 同配置 gate PASS 12/12；跨精度决策保持 12/12；工程计时 R_replay=1.170（伴随比回放快 ~14.6%，DEV 口径）。RAW_VERIFIED
- 方向导数评分恒等式（A0I-B）：数值等价 SUPPORTED + 决策保持 18/18（独立于 54/72 数值 gate）。RAW_VERIFIED
- 审计纪律本身：合同先于 GPU、SHA256 审计包、负结果保留、勘误不改判——本项目全部 88 条记录可用此纪律复现。
