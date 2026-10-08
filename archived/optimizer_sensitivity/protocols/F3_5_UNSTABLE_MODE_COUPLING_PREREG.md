# F3.5 — Unstable-Mode Coupling 预注册（100M，LR channel）

- 冻结日期：2026-09-29，**在任何 F3.5 eigenspace 数值计算之前**。本文件 commit 先于任何 F3.5 结果。
- 上游：F0 `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE` → F1 `V_FLOOR_CAUSAL` → F2 `PRACTICAL_HYPERGRADIENT_FAILURE` → F2.5 `TRUST_REGION_COLLAPSE_CORRELATED_ONLY` → F3 `SPECTRAL_INSTABILITY`（commit 79a0e81，`docs/F3_CURVATURE_MECHANISM_FINAL_REPORT.md`）。
- F3 已确立的事实（本研究的出发点）：α=0 时三 cell G32≈25–29 有界；α=1 EARLY 2188/18877 vs MIDDLE 29；source-off impulse EARLY 仍增长（λ32 = +0.152/+0.335，MIDDLE −0.044）；module ablation = DISTRIBUTED；**ρ(J_t, α=1) > 1.05 在 EARLY 与 MIDDLE 的全部 5 offsets 都可靠成立**（MIDDLE 1.85–4.05）但 MIDDLE 无 finite-time explosion。
- **F3.5 唯一科学问题：为什么局部不稳定谱在 EARLY 被实际 tangent/source 持续激发，而 MIDDLE 没有？** 本轮不开发任何方法（F4 禁止）、不训练、不改 forward AdamW、不改 F3 已冻结结果。

## 1. 冻结的实验对象（与 F2/F3 逐项相同）

| 项 | 冻结值 |
|---|---|
| 模型 / cells | MiniMind 100.78M；EARLY (2042,26)、EARLY (2043,26)、MIDDLE (2042,64)（bank 只读） |
| future streams / CRN / LR / L_val / 训练语义 | 与 F3 prereg §1 完全相同（stream_seed/manifest 不变） |
| reproduction 契约 | current-environment frozen reference（SHA256 `6f2650e0…8bdf6`）；每条 trajectory 的逐步 primal loss 必须与该 reference **逐位相等**（checker 同 F3） |
| offsets | 0, 1, 3, 7, 15（与 F3 RQ4 完全一致）；H ≤ 32 |
| operator | F3 `StepOperator`（α=1，clip 线性化系数按 F3 RQ4 冻结构造；HVP = fp32 GPU 双反向） |
| Arnoldi | 与 F3 冻结配置完全一致：Krylov 维 m=16、2-pass reorthogonalized MGS、起始向量 seed 91011（RAW 随机单位 full-state；DIMENSIONLESS 度量下起始向量 = 同一 RAW 随机向量的 S⁻¹ 缩放）、每-pair residual 容差 0.05（`|h_{m+1,m}·(e_m^T y_j)|/|λ_j| ≤ 0.05`） |

## 2. 冻结的两个状态空间 metric（PHASE D）

theta/m/v 数值尺度不同，所有 alignment/subspace 结论必须双 metric 报告：

- **RAW**：full-state Euclidean（th‖m‖v 直接拼接）。
- **DIMENSIONLESS**：以每 cell 的 **bank 初始 frozen state** 为基准，对每个 parameter tensor × 每 channel：
  `s_x[n] = sqrt(mean(x_0[n]²)) + eps_scale`，x ∈ {theta, m, v}；`eps_scale = 1e-30`（dtype-independent，仅防零除，**禁止看结果后调整**）。整个 H≤32 内 S 固定不重估。z̃ = S⁻¹z；J̃ = S⁻¹JS；J̃ 的伴随 matvec = S⁻1∘vjp∘S。eigenvalues 在 similarity 变换下不变（单测锁定），alignment/subspace 几何按缩放坐标解释。

**METRIC_ROBUST 规则**：主要 phase-difference 结论只有当 RAW 与 DIMENSIONLESS 定性一致（方向相同）时才可标注 METRIC_ROBUST；若矛盾 → 标 `METRIC_DEPENDENT`，禁止择优。

## 3. 冻结的 adjoint 一致性 gate（PHASE E，先于一切 left-eigenspace 使用）

对每 cell，在 offset 0，用 3 对固定 random probes（u_i seed=9200i，v_i seed=9300i，i=1..3，CPU fp64 randn 单位化 full-state）检查：

`disc_i = |⟨u_i, J v_i⟩ − ⟨J^T u_i, v_i⟩| / max(|⟨u_i, J v_i⟩|, 1e-30) ≤ 1e-5`（RAW Euclidean；S 对角故 gate 与 metric 选择无关）。

**任一 disc > 1e-5 → `LEFT_OPERATOR_GATE_FAIL`：停止全部 left-eigenspace excitation 声称（RQ3.5-1 的 E_left、RQ3.5-2 的 S_exc、RQ3.5-4 的 projector 全部 NOT_ESTIMATED），仅允许 right eigenspace / temporal rotation / source-off growth 继续，最终不得声称 SOURCE_EXCITATION 已证明。**

## 4. Dominant unstable eigenspaces（PHASE F）

- 对每 (cell, offset, metric)：α=1 算子的 Arnoldi top-k eigenpairs，**k=8（按 |λ| 排序，仅保留 residual ≤ 0.05 的 pair）**；对伴随算子（Jᵀ / 缩放伴随）同样取 top-8 → left。
- **strict unstable**：|λ| > 1.05（冻结阈值，禁改）；同时保留连续量 |λ| 与全部 top-8 谱数据。
- **复共轭 pair 处理**：每对共轭 pair (λ, λ̄) 以其实不变二维子空间 {Re v, Im v}（归一化、实 QR 正交化）表示；实特征值 → 一维。全部后续量（W_u/U_u 基、projector、module shares）基于实基。
- left/right 匹配：按 nearest-eigenvalue、共轭 pair 作为整体匹配；报告每对 |Δλ|（描述性）。
- **不把 full eigenvectors 写入 Git**。正式 artifact 仅：eigenvalue、residual、module energy shares（θ 通道，Re+Im 合并范数）、alignment 摘要、hash/seed/config。runtime 基向量缓存仅在计算进程内存/临时 runtime/ 目录（gitignored，cell 处理完即删）。
- **回归锚点（冻结）**：RAW 度量下 right Arnoldi 的 ρ = max|λ| 必须与 F3 `spectrum.csv` 的对应 (cell, offset, alpha=1.0) 值**完全相等**（同算法同起点），否则停止修代码。

## 5. RQ3.5-1 — 实际 tangent 对 unstable LEFT 子空间的激发

- 两个真实方向（与 F3 语义逐位一致并回归锁定，见 §8）：(a) forced α=1 LR-channel tangent δz_t（= F3 rq3 FULL 臂）；(b) source-off impulse δz_t（= F3 rq2 q0 臂）。
- 在 sampled offset o 的 reliable unstable left 实基 W_u（|λ|>1.05 的 top-8 子集，正交化 Q(W_u)）上：
  `E_left(t) = ‖Q(W_u)ᵀ δz̃_t‖ / ‖δz̃_t‖`（δz̃ = 度量相应坐标）。t 取 sampled offsets：impulse 含 t=0（q0 本身），forced t≥1（t=0 为零切线，跳过）。
- 辅助（非主指标）：E_right(t) 用 U_u 同式。
- 报告 per-cell trajectory **median / max / AUC**（AUC = sampled t 上的梯形积分 /(t_max−t_min)；forced 与 impulse 分开）。RAW 与 DIMENSIONLESS 双报告。

## 6. RQ3.5-2 — LR source 的选择性激发

- s_t := 第 t 步的 LR 非齐次源 inh_t = θ_{t+1} − θ_t（θ-only），由 frozen primal replay 精确重构（逐步 loss 与 frozen reference 逐位一致为硬 gate）。
- `S_exc(t) = ‖Q(W_u(o(t)))ᵀ s̃_t‖ / ‖s̃_t‖`；**冻结映射** o(t) = max{offsets} ≤ t−1（t=1→0；t=2,3→1；t=4..7→3；t=8..15→7；t=16..32→15）。
- 比较 EARLY vs MIDDLE 的 S_exc 分布（median/max/AUC over t=1..32）。

## 7. RQ3.5-3 — unstable 子空间时间持续性

记 U_t = offset t 的 reliable unstable RIGHT 实基（正交化）。相邻 sampled offsets（Δstep ∈ {1,2,4,8}）：

- 静态：`O_sub(t,t') = mean singular values(Q(U_t)ᵀ Q(U_t'))`，同时报 θ_max/θ_mean（arccos of singular values，clamp 到 [0, π/2]）。
- **propagated（主指标）**：把 U_t 的每个 basis vector 沿**真实** J 序列（α=1 hom steps，逐步 HVP）从 t 推到 t'：逐向量传播、逐向量即时投影到 Q(U_{t'})（增量式，不驻留全 ΦU_t）：
  `O_prop(t,t') = mean singular values( Q(U_{t'})ᵀ Φ_{t→t'} Q(U_t) )`。
- intervals：0→1、1→3、3→7、7→15。**15→31：NOT_ESTIMATED**（不为此新增 offset-31 eigenspace，控制计算预算）。
- 双 metric。

## 8. Trajectory 重构与回归（冻结）

每 cell 一次 replay：32 步 primal（逐步断言 loss 与 frozen reference 逐位相等）+ 同 replay 内传播 (a) forced α=1 切线与 (b) SO q0 切线（F3 冻结语义：pre-step grads → clip → 每 臂自身 HVP → hom_step → advance → +inh）；快照 δz 于 t∈{0,1,3,7,15,32}（impulse 含 t=0）与 s_t（t=1..32）到 runtime cache（fp64，仅本 cell 处理期间存在）。**回归断言（硬 gate）**：两条切线在 h=32 的 dtheta/dz 范数必须与 F3 已提交 raw（rq1 α=1.0 臂 / rq2 a1.0|q0_lrsource 臂）**逐位相等**（F3 已证该管线 0-mismatch 锚定）。

## 9. RQ3.5-4 — mode-removal / mode-only causal probe（offset 0，仅 source-off）

- 在 offset 0：取 reliable unstable right/left 实基 V_u, W_u（各自正交化后），κ = cond(W_uᵀV_u)。**冻结阈值 κ ≤ 1e3**；超限 → 本 probe 全部标 NOT_ESTIMATED（不得用 biorthogonal projector），仅保留 left-alignment 诊断。
- `P_u = V_u (W_uᵀ V_u)⁻¹ W_uᵀ`（正交化基下 W_uᵀV_u 奇异值 = 子空间主角余弦）；`q_unstable = P_u q0`；`q_removed = q0 − P_u q0`（恒等式 q0 = q_unstable + q_removed 精确成立，单测锁定）。
- 三个初始方向在**完全相同**的 future J 序列下 source-off 传播到 H=32（forward trajectory 逐位不变，单测锁定）。
- 报告 A. natural-norm（‖Φq‖ 终值）与 B. normalized（G = ‖Φq‖₂/‖q‖₂）。RAW 与 DIMENSIONLESS 各自的 projector 分别执行与传播。
- 描述性标签（非唯一 verdict gate）：**MODE_COMPONENT_NECESSARY** := G_removed(H32) ≤ G_full(H32)/10（B 版）；**MODE_COMPONENT_SUFFICIENT** := G_unstable_norm(H32) ≥ 0.5·G_full_norm(H32)。

## 10. RQ3.5-5 — 随机方向对照（不新增随机 probe）

复用 F3 冻结 probes（theta_only/m_only/v_only/full_random/attn_q_localized，seeds 91001–91005）+ q0。计算其 **initial E_left（offset 0 W_u）** vs **H32 growth**（growth 直接取 F3 `raw/rq2_*.json` 的 a1.0 臂；q0 用 §8 重构值）。样本量 6/cell/metric：只报 Spearman（exploratory）、rank ordering、cell 级定性关系，**禁止显著性声称**。

## 11. 冻结的判定程序（PHASE L）

记 `E^X_c` = cell c 的 X-方向 E_left/S_exc trajectory 汇总（median 与 AUC 分别评估），`O^prop_c` = 该 cell 的 O_prop 区间均值。

1. **SOURCE_EXCITATION 支持**：两个 EARLY cells 的（forced-tangent、impulse-tangent、source 三者中**至少一类**）E 汇总均 > MIDDLE 对应值的 **2×**，且该分离在 RAW 与 DIMENSIONLESS 下同向。
2. **MODE_PERSISTENCE 支持**：两个 EARLY cells 的 O^prop 均值均 > MIDDLE 的 **1.25×**，且 O_sub 不给出反向主导证据，双 metric 同向。
3. 两者皆成立 → `EXCITATION_AND_PERSISTENCE`；仅 1 → `PHASE_SPECIFIC_MODE_EXCITATION`；仅 2 → `UNSTABLE_SUBSPACE_PERSISTENCE`。
4. ρ>1 已知但 1、2 均不成立（alignment/persistence 都不能分离 phase）→ `LOCAL_SPECTRUM_ONLY`。
5. 若 RQ3.5-4 causal probe（NECESSARY/SUFFICIENT）与上述分析同向 → 在报告中升级为 **causal mechanism support**（描述性，不改变主标签规则）。
6. RAW 与 DIMENSIONLESS 任一主结论反向 → 主标签 `METRIC_DEPENDENT`（附两 metric 各自结果）。
7. §3 gate FAIL → `LEFT_OPERATOR_GATE_FAIL`（right-only 结果如实报告）。
8. 其余/关键量 NOT_ESTIMATED 占比过高 → `UNRESOLVED`。

允许主标签 + 次标签。允许的报告标签全集：`PHASE_SPECIFIC_MODE_EXCITATION` / `UNSTABLE_SUBSPACE_PERSISTENCE` / `EXCITATION_AND_PERSISTENCE` / `LOCAL_SPECTRUM_ONLY` / `MODE_COUPLING_NOT_SUPPORTED` / `METRIC_DEPENDENT` / `LEFT_OPERATOR_GATE_FAIL` / `UNRESOLVED`。`MODE_COUPLING_NOT_SUPPORTED`：E_left/S_exc 在 EARLY 与 MIDDLE 间无可分离差异（如两者都≈0 或都饱和）且 O_prop 无差异——即 mode-coupling 框架本身无抓手。

## 12. 冻结的输出（PHASE M）

`docs/F3_5_UNSTABLE_MODE_COUPLING_REPORT.md`；`outputs/f3_5_mode_coupling/`：`summary.json`、`adjoint_consistency.csv`、`eigenpairs_summary.csv`、`tangent_unstable_alignment.csv`、`source_unstable_alignment.csv`、`subspace_rotation.csv`、`propagated_overlap.csv`、`mode_projection_probe.csv`、`random_probe_relation.csv`、`metric_robustness.csv`、`raw/`（每计算单元一个原子 JSON：cell×offset×metric 的 eigen/alignment、cell 的 traj/adjoint/probe）。`runtime/`（gitignored）：trajectory 快照、sources、临时基向量，cell 处理完删除。

## 13. 执行与安全（PHASE N）

tmux 内执行（`scripts/run_f3_5_driver.sh`）；每 cell 顺序处理（RAM 峰值预算 ≤ 70GB/进程，串行保证 ≤ 1 个重进程）；每计算单元原子写（tmp+rename）；支持 resume（单元级 done 判据 = 对应 raw JSON 存在且完整）；禁止多进程写同一 CSV（CSV 由 postprocess 从 raw 单点生成）。每 job 前 `date/uptime/free -h/df -h/nvidia-smi` 快照 + 5 分钟采样。禁止 sudo/reboot/systemctl/GPU reset/驱动 reload；OOM/CUDA error → 该 job fail 并保存日志，不做系统级恢复。

## 14. 测试（PHASE O，先于任何正式计算通过）

新增 `tests/test_f3_5_mode_coupling_v01.py` 至少覆盖：(1) J/Jᵀ adjoint identity（synthetic + scaled wrapper）；(2) similarity 变换保 eigenvalues（synthetic）；(3) principal angle 实现正确性；(4) projector 重建（P²=P、秩、谱分量恢复）；(5) q_full = q_unstable + q_removed 恒等；(6) source-off intervention 不改 forward（逐位 loss 相等）；(7) RAW/DIMENSIONLESS 缩放往返 + J̃ matvec 等价；(8) 复共轭 pair 实基处理（复特征向量组合在实基上投影范数守恒）。既有 F3/F2/F1/F0 测试全数通过。

## 15. 红线

不改 F3 已冻结结果与 prereg；不开发 optimizer/stabilizer/controller（F4 禁止）；零新训练；不下 1B；不改 legacy；不改 F2/F2.5 已提交代码；不为本轮任何结论调整已冻结阈值/网格/判定程序；负结果与 NOT_ESTIMATED 原样保存；不 push；每阶段本地 commit。

## 16. ERRATUM（2026-09-29，commit 2ac9d45 之后、任何 F3.5 eigenspace 计算之前）

§2 中"J̃ 的伴随 matvec = S⁻¹∘vvp∘S"为笔误：J̃ = S⁻¹JS 的 Euclidean 伴随是 **S∘vjp∘S⁻¹**（S 实对角）。该错误由预注册的单测（§14 第 1 项 adjoint identity，DIMENSIONLESS 分支）在实现阶段捕获并修正（`ScaledOperator.vjp`）。不涉及任何阈值、gate、判定程序或科学内容的变更；此处按红线要求透明披露。

另注（同日，同属实现层澄清）：§9 的恒等式 q0 = q_unstable + q_removed 在精确算术下逐项成立；在 fp64 下因 fl(q−Pq)+Pq 的舍入以机器精度（rtol 1e-13，单测锁定）成立。另发现 F3 冻结库 `zeros_like_state` 的 th/m/v dict 条目共享同一 tensor 对象——F3 全部历史计算仅做整条目赋值（不写共享 tensor），不受影响；F3.5 的 Ritz 向量合成改为按 (channel, name) 新建 tensor，避免跨通道泄漏。

第三项（DIML 数值有效性，同日、先于任何 F3.5 计算）：§1 中 DIML 起始向量"S⁻¹ 缩放同一 RAW 随机向量"的约定在极端尺度比（v-floor 张量 ~1e-15）下使起始向量被最小尺度通道主导 ~1e15，fp64 MGS 数值退化（由 similarity 单测捕获）；修正为：同一 seed-91011 随机抽取（数值相同的向量）在各自度量坐标中直接使用。即便如此，per-tensor 尺度比极端时 J̃ 的跨通道块本身可达 ~1e15，fp64 Krylov（无 balancing）无法忠实复现 similarity 不变的谱（显式矩阵 + LAPACK balancing 可以，单测证实实现正确）。因此增设**数值有效性旗标**（实现层，非科学阈值）：每 (cell, offset) 若 DIML 的 rho_all 与 F3 RAW anchor 相对偏差 > 5%，该 offset 的 DIML eigendata（E_left/E_right/S_exc/O_sub/O_prop/probe）标 `DIML_UNSTABLE`，只记录不参与 metric-robustness 判定（计为 INCOMPLETE 而非矛盾）。
