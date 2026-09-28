# 00 — CHATGPT HANDOFF：fluctuation_response 全项目交接（2026-09-26 全量只读审计）

> 本文件自包含。读者不需要预先知道任何实验名称。
> 所有数字均来自服务器上实际存在的冻结制品（summary.json / REPORT.md / verdicts / SHA256SUMS），
> 审计时已从磁盘重新核读。标注规则：RAW_VERIFIED=直接读制品核验；REPORT_VERIFIED=读报告核验；
> SUMMARY_ONLY=只在汇总文档中出现；CONFLICT=文档间冲突（见 10 号文档）。

---

## 1. 项目背景（一段话版）

**研究主题**：Transformer + AdamW 训练动力学的**涨落—响应（Fluctuation–Response, FR）关系**。
核心科学问题：对真实训练轨迹中的模型参数施加一个微小扰动 q，未来 H 步 validation 读数 X 会如何变化（响应 χ）？
这个响应能否由**局部光滑切线（optimizer 状态 + 曲率反馈 HVP + clipping Jacobian）**解释？
如果能，它能否作为**训练动作（数据加权 / 学习率调整）的效用度量**，即用于控制训练？

- 模型：MiniMind 系 dense Transformer（4.98M anchor → 26.7M → 100.8M；外部 OLMo-2-1B 曾尝试）
- 优化器：原生 PyTorch AdamW（β 0.9/0.999，eps 1e-8，wd 0.01），grad clip 1.0，bf16 autocast，accum 2，seq 128
- 数据：MiniMind pretrain 子集（D0=e3c_subset 4096 条；D1=官方 pretrain_hq 4096 条）
- 硬件：2× RTX 5090 32GB；conda env `fr_minimind`；torch 2.11.0
- 位置：`/home/b205/projects/fluctuation_response/`（Git 仓库，master 分支 3 commits，**全部未 push**）
- 时间跨度：2026-09-01（框架部署）→ 2026-09-26 00:22（最后一个实验 F2b）
- 纪律文化：全程"合同先于 GPU 冻结、预注册、SHA256 审计包、负结果如实保留、禁止事后改阈值"。

## 2. 三条研究线 + 一条前置链

| 线 | 名称 | 时间 | 问题 |
|---|---|---|---|
| E 链（前置） | replay/涨落/theta 映射 | 09-01–09-09 | 能否位级重放训练？涨落-响应原始假设（E3 主检验）是否成立？ |
| FR 主线（A0 系列） | 机制 + 决策效用 + 控制 | 09-09–09-18 | 响应由什么构成（A0C–A0F2）？有用吗（A0G）？便宜吗（A0H–A0J）？能控制训练吗（A0K–A0U）？ |
| 泛化 G 线 | 机制边界 | 09-16–09-19 | 机制阶梯跨规模（G1a/G1b/G1c）、跨数据（G2）、clip 干预（G3）是否成立？ |
| GTTR 线（含 B0/C0） | 几何-追踪-输运-读出 + 终局规则 | 09-20–09-26 | 局部几何预测（B0/C0/C1）→ 目标校准 → 动作评分 → 规则识别 → 全时域 LR 规则求解（F0/F1/F2） |

## 3. 研究演化叙事（最重要的一节）

### 3.1 第一阶段：基础设施与原始假设的失败（E 链）

1. **E0 replay**（09-02）：证明可以从保存的 (θ, m, v, RNG, batch manifest) **位级精确重放** MiniMind 训练（5 项 hash 全等）。这是之后一切因果实验的地基。PASS。
2. **E1 fluctuation**（09-02）：构造只从 held-out train-probe 出发的单位扰动方向 q（R=8/16 流，σ_q≈0.09），并建立 CRN（共同随机数）三分支 ±ε 协议：`backward → unscale → ±ε·q → clip_grad_norm_ → AdamW → zero_grad`（**pulse 在 clip 之前**）。PASS（技术性）。
3. **E3 theta mapping v05**（09-02）：检验"响应 χ 可由旧 θ 预测"——H=1 Pearson 0.637 尚可，H=8 r=−0.083 崩掉。**科学判定：未自动判定（人工审核=不支持）**。
4. **E3-R cross-seed v06**（09-03）：4 seeds 复现，within-seed 去趋势后 residual Pearson 仅 0.311，LOSO mean improvement −0.0175。
5. **E3-C confirmatory v07**（09-03，**项目最重要负结果之一**）：8 seeds × 7 steps × 8 horizons 预注册主检验，**H8_CONFIRMATION = FAIL**（LOSO 5/8 < 预注册 6/8）。原始"普适涨落-响应比例关系"假设失败。但副产品是 **v07 56 态认证 bank**（8 seed × 7 step 完整 checkpoint + q + E2 manifest），成为之后所有 A0 实验的唯一冻结输入源。
6. **E4A/E4B**（09-03–09-09）：低维状态/动态时间尺度探索，均 exploratory（q 方向随 horizon 旋转 Rq 0.958→0.880；曲线塌缩 NO）。

### 3.2 转折点：从"比例关系失败"转向"机制分解"（A0C 审计链，09-09–09-12）

7. **A0 data audit v01–v08**（09-09–09-10）：对 v07 bank 的残差 FR 做归因。发现控制 alpha_direct（动量核直接项）后 B（optimizer 切线）的增量 R²=0.77（p=4.7e-11）——失败的比例关系里藏着可分解机制。同时期的 17 个根目录 `FR_A0C_*` 工作包 + `outputs/a0_fr_audit/` 下的 mv_theory v01–v06 链试图用闭式 m/v 理论解释，**全部失败**（旧 v 路径产生 ν 反常 −1.3e6 量级）。
8. **H1 Jacobian**（09-10）：H=1 响应由局部 AdamW 映射 Jacobian **完全解释**（ε/4 相对误差 0.000146–0.019）。ν 反常确认来自冻结 v 近似假象。
9. **A0C response_propagation_v01**（09-10，**项目最重要正结果**）：模型阶梯 A（标量动量核，mre 6.36）/ B（optimizer-only 切线，mre 0.165）/ **C（完整切线=optimizer+曲率 HVP+clip-JVP，mre 0.0274，R²=0.9977）**。H>1 9/9 行 C 胜 B。指标 `_metric`/`relative_error` 冻结于此，后续所有实验禁止重定义。
10. **A0D**（09-11）：C 的残差随 ε 收敛（ε/4 mre 0.0045，R²=0.99982）→ 残差主要是有限 ε 伪影，H≤8 **不需要 Mori–Zwanzig 记忆核**。
11. **A0E → A0E2**（09-12）：A0E 发现"raw curvature 有害"(gain −0.136)；A0E2 定位为 **BH 初始化 bug**（初始脉冲未过 clip-JVP），修复后 raw curvature 正贡献 +0.0686，clip-JVP +0.1148。旧 A0E 整体作废（列入禁引清单）。
12. **A0F/A0F2**（09-12）：跨 56 态全普查，C 胜 B 168/224（56 平=H1 结构性平局，0 负），7/7 训练阶段全正。机制结论封闭。
13. 期间事故：`theta_protocol.py` 源码于 09-12 00:16–19:00 之间被删，全盘 48 个副本无匹配冻结哈希；以冻结源编译的字节码 `theta_protocol.pyc` 恢复（行为等价经位级复验），**至今仍是 pyc-only 依赖**。

### 3.3 第二阶段：决策效用与成本（A0G–A0J，09-14–09-16）

14. **A0G decision utility**（09-14）：C 能否选出真正降低未来 val loss 的数据加权动作？**效用成立**（H=4 Pearson 0.905，mean Gain 1.54e-4>random，H=8 C vs B bootstrap P=0.995）；**成本不成立**（shared-tangent 恒慢于逐候选回放，speedup<1，无 break-even）。
15. **A0H-0/1**（09-14）：伴随（adjoint）降成本。H=1/B 精确，C|H=8 逐元 gate 被 **bf16 math-SDPA HVP 代理的转置不对称噪声地板**阻塞（实现本身精确到 4.1e-10）。
16. **A0H-1.5 两轮 + A′ 验收**（09-15）：候选共享 VJP 伴随建立；A′ 验收 3/3 FAIL——后经**决策语义审计**定位为验收驱动 selector 方向写反（argmax vs argmin），非算法失效。
17. **A0H-2.0 (A″)**（09-16）：修复语义后的独立效用 pilot，**POSITIVE_POINT_ESTIMATE**（V_eval Gain +1.467e-4，5/6 seed 正）。
18. **A0I**（09-16）：A0I-A 增益来源分解（选择器贡献不确定/异质）；A0I-B 标量方向导数评分（子阶段 −30% 成本但总成本 SIMILAR，shared prefix 主导）。
19. **A0J**（09-16）：缓存 future sensitivity 复用——**时间尺度负结果**（决策 margin 3.8e-5 比 age=1 缓存误差 ~2e-3 低 2–3 个量级，chosen-id 保持 5/6→1/6 崩溃）。

### 3.4 第三阶段：真实闭环控制（A0K–A0N，09-17）

20. **A0K closed loop**：fresh controller 在真实训练闭环等步中位 **+7.44e-4 nats**（CASE A），但 **M5：自适应选择器不优于固定 j=1 注入**（−1.29e-4）；收益来自干预本身。
21. **A0L mixture**：连续数据混合控制 CASE B（固定 j1 混合 ≥ response controller，2/3 seed）。
22. **A0M**（纯离线）：响应存在**相位结构**（共模 |mean|/std≈1.85，early→late 衰减 8–10×；argmin 身份 phase 依赖 8/9）——CASE A。
23. **A0N phase scheduler**：ρ 门控利用相位结构——CASE B（B≥C），结构存在≠结构可利用。

### 3.5 第四阶段：optimizer 输入通道（A0O–A0U，09-18）

24. **A0O LR response**：u=log(η/η₀) 响应存在且解析=FD（Pearson 0.9552，方向准确率 0.875 vs null 0.125）——Q1/Q2/Q3 全 YES。LR 是纯 θ 通道推（理论修正 A2）。
25. **A0P impulse kernel**：**Stage-1 gate FAIL**（Pearson 0.731<0.8）。响应核=快初始衰减（半衰期 4–8 步）+长记忆噪声尾，**无稳定时间尺度**——短期→长期桥梁未确立。项目最清晰的护栏负结果。
26. **A0Q control map**：J=[LR, clip, wd]。LR 唯一稳定通道；clip 事件驱动（7/12 cell 恒 0）；wd 幅度低 LR 4 个量级。
27. **A0S controllability**：|χ| 预测干预增益（Spearman 0.762，LOSO 泛化）PASS——短时域可控性坐标成立。
28. **A0T LR control pilot**：sign(χ)→u=±0.05 控制器 PASS（H=4 中位 +4.4e-4，35/36 一致）；但 C≡B 按构造别名 34/36。
29. **A0T-extra 身份分析**：控制器身份=**全局 LR 放大方向发现**（熵 0.31 bit），仅有的 2 个可判别 cell 的状态依赖反转在 H=16/64 反转为有害。
30. **A0U schedule discovery**：response 引导的 schedule ≈ 延迟 6 步的固定 ×1.05 LR 缩放（Case B）；可利用信息 ≈ 1 bit（符号）。

### 3.6 泛化线（G，09-16–09-19，与 A0 线交错进行）

31. **G1a 26M**：STRONG_SCALE_REPLICATION（C R²=0.99958，18/18 全胜）。
32. **G1b 100M**：STRONG_SCALE_REPLICATION（C R²=0.99949，18/18）。
33. **G1c external 1B**：OLMo 官方 checkpoint 初判无 optimizer state → BLOCKED；直链复核推翻（standard checkpoint 四件套齐全）→ feasibility gate：Phase 0/1 PASS，**Phase 2 C 分支失败**（主因 C=scale-dependent limitation：1B 深剪裁 regime 下 C 递推超线性放大 → fp16 溢出；A 实现 bug/D optimizer 异常排除）。61G 大文件已于 09-25 清理（保留协议/结果/SHA256）。
34. **G2 data distribution**：D0→D1 大幅语料偏移下 STRONG_DATA_REPLICATION（C R²=0.99995，18/18）。
35. **G3 clipping intervention**：**MECHANISM_CONFIRMED**——no-clip 真重训使 gain_clip 从 +0.094 精确崩溃到 0.000（BH_fixed≡C 位级），clip→response 因果链获干预证据。

### 3.7 机制深挖与 GTTR 线（B0/C0/C1 + GTTR，09-20–09-26）

36. **B0**（09-20）：脉冲释放/持续扰动、中心轨迹承载、二阶曲率最优幅度（UNRESOLVED）。
37. **B0.5**（09-20）：保留收益经 θ 位移（非 m/v 记忆）承载；(m,v) 记忆核 ~12 步内衰减。
38. **B0.6**（09-21）：regime dynamics——ψ_t **无亚稳态 regime 证据**（Q1 NOT_SUPPORTED），单一近单位谱半径线性映射同样好。Headline NOT_TESTED。
39. **C0**（09-21）：一步几何预测 Mgeom=g+H·s——干净 anchors 上误差减半（E≈0.4），但半库规则失败 → LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT；**C0X 离线复算发现判定器把聚合门实现成逐点门（ERRATUM）**。
40. **C1**（09-21）：新独立 probe 库上 field 通道获得 **LOCAL_SUPPORT_ON_SELECTED_ANCHORS**（GTTR Geometry/Transport 最强正向证据）；shadow UNRESOLVED。
41. **C1-Late**（09-21）：补齐 t48 anchors，COMBINED6 field+shadow 双 PASS（带纪律保留）。
42. **GTTR Model-Target alignment audit**（09-22）：25 条论文式声明审计——**MISALIGNED_FOR_DEPLOYMENT_CLAIMS**（LM NLL 降低≠通用能力提高）。
43. **PhaseA A0+A1 native calibration**（09-23）：6 条 native 轨迹 256 步，finite_cosine_then_hold schedule 冻结；未进入 B/C/D/E。
44. **PhaseB action scoring v3**（09-23）：预注册数值门在 S3 终止——**LOCAL_OPERATOR_NOT_CALIBRATED**（fd_rel=0.706>0.20）；FD diagnostic 证明位级可复现、非实现 bug。
45. **D1 direct readout**（09-23）：**NO_OBSERVED_GAIN**（预算终点 NLL 与目标 token 效应均无增益）；范数匹配 RAW 优于 NORM。
46. **R1 objective identification**（09-24）：离线辨识二次型目标（K 矩阵条件数 28.5–49.9）。
47. **R2B solved update bridge**（09-24）：已求解首步改动的一步效应 −2.9e-4（9/12 有利）**但终态 NLL 未获益**（8/12 为正）；不外推持续规则。
48. **CB1 cost measurement**（09-24）：成本恒等式代数层 PASS；rho 不能作可靠剩余成本预测子（中位相对误差 0.20）。
49. **F0 terminal rule solver**（09-25）：256 步全时域 LR 规则求解——**BLOCKED_AT_CALIBRATION**（terminal 导数 SENS 与 FD 差 6 个量级）。
50. **F0-CALFIX + F0-IC + F1 closure**（09-25）：修复链——定位两个实现缺陷（D3 q 计数缩放/D4 读出）、FP32 FD 分辨限制、对象错配（Z16 vs Z256）；修复后 b 坐标 SENS=−0.311 与 FD=−0.345 一致，但 **h256 全时域梯度仍未校准**（FULL_HORIZON_GRADIENT_UNRESOLVED_AFTER_FIX）。
51. **F2a/F2b**（09-26 00:02/00:22，最新）：规则梯度方向 **DIRECTION_NOT_CONFIRMED**（best_J 在 b=+0.02 而二阶预测 b*=−0.0146 失败，曲率全尺度为负）。**GTTR 终局规则求解线当前停在方向未确认状态。**

## 4. 全部实验类别总览（91 个实验/版本条目）

- E 链 replay/涨落/映射/确证/低维：10 条
- A0 审计预备链（含根目录 17 包）：~10 条
- A0C–A0F2 机制链：8 条
- A0G–A0U 决策/成本/控制链：19 条
- G 线：7 条
- B0/C0/C1 线：8 条
- GTTR 线：14 条
- 辅助审计包/验证包：~6 条
- 合计 CSV 91 行（见 01_COMPLETE_EXPERIMENT_INDEX.csv，39 列）

## 5. 最重要正结果（按可靠性排序，全部 RAW_VERIFIED）

1. **完整 AdamW 切线 C 是参数扰动因果响应的高精度预测器**（A0C/A0F2：R²=0.9977–0.9978，mre 0.0059–0.027；56 态普查 0 反例）。
2. **机制阶梯可分解且可干预**：optimizer state→curvature→clipping 三段增益（A0E2 +0.0686/+0.1148 anchor；G3 no-clip 干预使 clip 段精确归零=因果证据）。
3. **跨规模/跨数据复制**：5M→26.7M→100.8M scale-only（G1a/G1b STRONG×2）+ D0→D1 语料大偏移（G2 STRONG）。
4. **H=1 局部 Jacobian 理论精确成立**（H1）；残差=有限 ε 伪影（A0D）。
5. **LR 是唯一稳定 optimizer 控制通道**（A0O Pearson 0.955；A0Q），响应是冻结优化器的可计算导数而非经验拟合。
6. **响应幅度含短时域可控性信息**（A0S Spearman 0.762 跨 seed 泛化）。
7. **响应引导的闭环干预有小幅净正收益**（A0K +7.44e-4；A0H-2.0 +1.47e-4 点估计）。
8. **响应存在相位结构**（A0M 共模 8–10× 衰减；A0N 闭环复现）。
9. **GTTR field 通道跨库支持**（C1/C1-Late：E≈0.36–0.50 优于 M0 基线）。

## 6. 最重要负结果（每条都排除了一个方向）

1. **E3-C H8_CONFIRMATION FAIL**：普适涨落-响应比例关系不成立（LOSO 5/8）。→ 排除"直接类比物理 FR 定律"路线，催生机制分解。
2. **选择器不优于固定动作**（A0K M5/A0L/A0N/A0I-A 四重独立收敛）：自适应选择的增量价值 ∈{0,负}。→ 排除"response 智能选择器"路线（pilot 规模）。
3. **无成本优势**（A0G/A0I/A0K ROI）：5M 规模 response 计算不比逐候选回放便宜。→ 研究价值只能来自"回放不可行"的场景。
4. **A0P 无稳定时间核**：一阶响应是有限时域对象，短期增益不得外推最终训练。→ 全项目强制 caveat。
5. **可利用信息 ≈ 1 bit**（A0T-extra/A0U）：单符号 bank 上任何 response 规则退化为固定 LR 缩放+迟滞前缀。→ 未来的自适应方法必须先建立符号多样性 bank。
6. **缓存/ amortization 失败**（A0J）：决策 margin 低于任何 age≥1 代理漂移 2–3 个量级。
7. **G1c 1B C 分支失败**（scale-dependent limitation，fp16 溢出近因）：完整切线递推在 1B 深剪裁 regime 超线性放大。→ 1B 外部轴未覆盖。
8. **B0.6 无 regime 切换**、**B0.5 记忆核 12 步内衰减**：离散 regime/Markov 记忆修正被排除（64 步/4.98M 范围）。
9. **PhaseB 局部算子未校准**（fd_rel=0.706）+ **D1 无读出增益** + **F0 终局梯度未校准** + **F2a 方向未确认**：GTTR 从"局部几何"到"终局规则"的桥尚未建成——当前活跃失败链。
10. **旧 A0E "raw curvature 有害" 作废**（BH 初始化 bug）；mv_theory v 路径作废——两个已推翻的早期结论，禁引。

## 7. 当前真实结论（一句话分层）

- **observation exists**：是——数据扰动与 LR 扰动的有限时域响应都存在且可测量（RAW_VERIFIED）。
- **prediction works**：是——C 模型在 5M/26M/100M/D1 数据上 R²>0.997（1B 除外）。
- **causal mechanism established**：部分——clip 段有干预证据（G3）；curvature 段只有相关性（G3 备注：no-clip 下 curvature 增益不消失）。
- **control works**：短时域是（A0K/A0T 冻结判据 PASS）；**control beats fixed baseline：否**（A0K M5/A0L/A0N/A0U 一致）。
- **computationally useful**：否（5M 规模无墙钟优势）。
- **scalable**：机制预测到 100M 复制；控制/效用未在 >5M 测试；1B 机制失败（C 分支）。

## 8. 剩余开放问题

1. response 选出的干预收益能否存活到 end-of-training（A0P 说核不外推——缺一个"收益持续"证明）。
2. 是否存在 (θ,m,v) 的低维函数能在**决策精度**（非 cosine 精度）预测响应（A0J 只排除了 sensitivity 缓存）。
3. 什么训练 regime 产生**多符号、horizon-persistent** 的 χ（打破 1-bit 简并）。
4. GTTR 终局规则：F0/F2 卡在全时域梯度方向；二阶预测失败——需要新校准路径或放弃该规则类。
5. 1B C 分支失败的修复（eps 地板/罕见 token 零 v 坐标）是否可行。
6. 效用/控制结论在 26M+ 规模完全未测（G4 A0G-lite 未启动）。

## 9. 可复用基础设施（详见 05_REUSABLE_ASSETS.md）

- **v07 56 态认证 bank**（4.6G，`outputs/e3_confirmatory/v07/`）：一切 A0 实验唯一输入源，禁止改动。
- **冻结分析模块库**（`research/src/fr/`，68 个 .py）：`a0c_adamw_response_propagation_v01.py`（指标唯一定义源）、`adamw_tangent_step`/`clip_jvp`（AdamW 切线步进）、`theta_protocol.pyc`（**pyc-only**）、A0H15 生产伴随入口、A0O LR 响应源项等。
- **Runner 三件套模式**（contract→run→analyze→audit_pack，150 个脚本）+ 单测 57 个。
- **审计包惯例**：SHA256SUMS + manifest + 可列可解验证。
- **MiniMind 副本**（1.2G）：未修改的上游 minimind-3，含 1.2G pretrain 语料。
- replay/CRN/seed 控制基础设施（E0/E1 遗产）。

## 10. 服务器与 Git 状态（速览，详见 07/08/09）

- 磁盘 1007G 已用 75%（剩 247G）；`/home/b205/projects` 占 72G，其中 `outputs/` 占 71G。
- 最大目录：g1b_100m 14G、a0k 14G、a0n 7.5G、g2 5.3G、e3_confirmatory 4.6G、r2b 4.5G、a0o 4.5G。
- Git：master 3 commits（09-25 首次入库），remote 已配置（github kehaol140-afk/fluctuation-response）但**从未 push**；11 项 untracked（F0/F1 代码+tools/）；无 stash/tag/LFS；6 个备份仓库全为零提交空壳；minimind 不是 git 仓库。
- `theta_protocol.py` 源码丢失事件（09-12）至今未排查原因。
- 运行环境：conda `fr_minimind`、PYTHONPATH=research/src、CUDA_VISIBLE_DEVICES=0。

## 11. 给下一个研究者/新项目的建议入口

1. 想复用机制结论 → 读 `docs/FR_PROJECT_STATUS.md` §6–§12 + A0C/A0E2/A0F2 制品。
2. 想做控制/效用 → 必须先过 A0P 护栏（不外推短期增益）+ A0K M5 负结果（须打赢 fixed baseline + equal wall-clock）。
3. 想做 20M–300M controlled experiment → MiniMind 副本可直接用（见 06_MINIMIND_AUDIT.md），建议 wrapper/adapter 而非魔改；G1a/G1b 的 config（`research/configs/g1*_v01.json`）与训练协议是现成模板。
4. 想清理磁盘 → 见 08_DISK_AUDIT.md 的归档候选（g1b bank 11G、a0k 快照 14G、a0n 7.5G、g2 语料 1.6G 等）与绝不删除清单（v07 bank、全部 SHA256SUMS、全部 contract/manifest）。

## 12. 审计方法与可信度声明

本审计 100% 只读。证据来源：文件系统枚举（55 个 outputs 子目录逐一盘点）、Git 只读命令、全部关键 .md/.json 文档阅读、子代理交叉核验（Git/磁盘/outputs/MiniMind/research 五路独立）。大型 .pt/checkpoint 未加载内容，只登记路径/大小/mtime/引用关系。文档数字与制品冲突之处全部列入 10_INCONSISTENCIES_AND_MISSING_ARTIFACTS.md，未替读者裁决。
