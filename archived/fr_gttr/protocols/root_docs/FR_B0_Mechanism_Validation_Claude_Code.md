# FR B0：有限时域响应的机制辨别与下一阶段物理建模

日期：2026-09-20
用途：可直接交给运行于 FR 远程仓库的 Claude Code。执行端可以是用户现有的 Kimi K2.7 配置。
状态：这是研究设计与执行指令，不是已运行实验报告。设计者未访问远程仓库；既有文件位置和接口必须现场确认。

## 0. 任务与授权范围

请在当前 FR 仓库中完成 B0：先审查现有接口，再实现最小诊断脚本、运行 CPU 理论测试、运行 2-cell GPU smoke；数值检查通过后，在本协议的规模内完成主诊断并出表出图。不要只交计划，也不要把任务自动扩展到新优化器、新架构、长训练或大规模扫描。

本轮问题：
1. 短期 LR 响应的变化，来自干预结束后的状态后效应，还是来自继续施加干预？
2. 响应能否分为流间平均参数轨迹贡献与流间分散贡献？
3. 当前幅度范围内的二阶控制曲率是否可可靠辨认，是否存在不同于固定最大 LR-up 的最优幅度？
4. 现有数据是否足以支持后续条件准稳态模型？不足就明确报告，不要宣称平衡态或场论已经成立。

优先级：正确的训练语义和数值结果 > 能运行并出图 > 简洁可复现 > 工程封装。保持科研脚本风格，不重构训练框架、不引入数据库/服务/复杂调度器、不递归生成文件 SHA、不安装无关依赖。只记录必要配置、现有版本信息、真实命令和运行结果。

## 1. 既有证据与解释边界

先读取实际存在的：
- docs/FR_PROJECT_STATUS.md
- docs/FR_EXPERIMENT_INDEX.md
- A0O/A0P/A0Q/A0S/A0T/A0T-extra/A0U 的 contract、report、逐 cell 原始记录
- 冻结的 AdamW 单步、数据流、恢复状态、验证损失、adjoint/FD runner

已知 retrospective 报告 A0O 的短时 LR 响应验证较好，而 A0T 大部分 guided 动作与 fixed-up 相同；少数不重合单元的 H=4 优势在 H=16/64 反转。必须将这些数字与原始记录对应后再复用。

以下不是已经证明的事实：全训练不可控、所有平衡解释均错误、不存在长时导数、不存在廉价充分统计量、标量输入不能自适应、响应幅度没有信息。不要把某个代理或策略失败改写为全局不可能性。

对标量损失，一行 Jacobian 的秩天然不超过 1，禁止用它论证控制空间只能是一维。

## 2. 目录与接口原则

新目录建议为 experiments/fr_b0_response_mechanisms/ 和 outputs/fr_b0_response_mechanisms_v01/；若仓库有统一约定，使用其约定并记录映射。这些是拟新增位置，不是已确认存在的路径。

原则上只需：
- 一个最小 runner/adapter：调用冻结实现，不能抄写并悄悄改写 AdamW；
- 一个 CPU 理论测试脚本；
- 一个汇总与绘图脚本；
- 一个明确的配置文件。

实际函数名现场读取确认。禁止编造接口。默认不修改旧实验输出，不提交或推送代码，不改系统配置，不关闭其他训练进程。不要打印密钥或整个环境变量表。

记录可获得的 Claude Code 版本与实际后端 model ID，但无法读取 model ID 不阻塞科学任务。不要擅自更换用户的模型、供应商或已能工作的接入配置。

## 3. 科学定义

完整训练状态至少包括 theta、m、v、optimizer step、scheduler 状态和随机数状态；若还有 gradient scaler、训练中可变 buffers 或其他状态，也必须恢复。

令 baseline 在绝对 optimizer step k 的 LR 为 eta_native(k)。本轮定义：

u = log(eta / eta_native), eta(k;u) = eta_native(k) * exp(u).

因此 u=0.05 的 multiplier 是 exp(0.05)，不是 1.05。旧实验若使用 1.05，保留其真实语义，不得悄悄重标记。

损失差：Delta L_H(u) = L_H(u) - L_H(0)。收益：Gain_H(u) = L_H(0) - L_H(u)。

对指定干预协议 p：
chi_H^p = d L_H^p(u)/du at u=0,
Q_H^p = d^2 L_H^p(u)/du^2 at u=0.

Q 是控制标量 u 的二阶曲率，包含完整训练轨迹的非线性，不等于参数 Hessian，也不得混用。

central FD：
chi_fd(a) = [L(+a)-L(-a)]/(2a)
Q_fd(a) = [L(+a)+L(-a)-2L(0)]/a^2.

比较 a=0.05 与 a=0.025。二阶差分的噪声被 1/a^2 放大；Q 不可辨认时输出 UNRESOLVED，不任意截断成正数或补 epsilon 后宣称有效。

训练的当前 LR 直接输入项为 b_eta_theta=-eta*(d+wd*theta)，当前 m/v 的直接偏导为零；后续 m/v 会通过 theta 和未来梯度变化而改变。禁止冻结后续 m/v 来伪装完整 LR 响应。

## 4. 阶段 A：仓库审查与冻结本轮设置

输出 audit.md，列明：
1. 实际调用的 baseline、restore、train_step、evaluation、adjoint/FD 接口。
2. 可用 seed、checkpoint、原始 future-stream ID、数据顺序和验证集划分。
3. 当前 precision、attention backend、梯度累积、有效 batch、token 数、clip、weight decay、LR schedule。
4. A0T 的 horizon 单位究竟是 optimizer step 还是 microbatch；本轮统一为 optimizer step，并保留对应 microbatch 数。
5. 真实 GPU 可用情况；仅使用空闲设备，不假设本地记忆中的 GPU 配置就是远程配置。
6. 哪些原始结果已复核，哪些只有 retrospective 摘要，哪些缺失。

冻结配置再看新实验结果。不得依据响应符号挑主样本、修改指标或调松阈值。

默认主样本：按固定排序取 3 个可用训练 seeds，每 seed 取绝对 step 46 和 82 两个 checkpoint。若缺失，按预先写明的最近可用 step 规则替换，避免重复，并记录原因。不要为补齐 checkpoint 新开长训练。少于 3 seeds 时使用全部可用 seeds，并标明证据更弱。

每个 checkpoint 使用 4 条新的、与旧评估不重合的 future streams；在配置中登记真实 ID。前 2 条标为 development，后 2 条标为 audit。两类流的语料分布和构造规则一致，只是独立随机实现。这里的 audit 只是新诊断的保留流，不足以把 3 个 seed 的 pilot 称为正式大样本证明。

默认 3*2*4=24 cells。主样本不能混入专门挑出的正响应或反转案例。既有 2 个 A0T 反转 cell 可单独做 stress-case 复核，独立标注且不混入主样本统计。

## 5. 阶段 B：先通过 CPU 理论测试

实现和运行下列小测试。先构造能因符号/索引错误而失败的检查，再实现，不以“程序能跑”替代数学检查。

### B1. 慢进展 + 横向涨落的玩具模型

L(x,y)=L_star-a*x+(kappa/2)*y^2
x_{k+1}=x_k+eta*a
y_{k+1}=(1-eta*kappa)*y_k+eta*sigma*xi_k
E[xi]=0, Var[xi]=1

q_{k+1}=(1-eta*kappa)^2*q_k+eta^2*sigma^2
q0=eta0*sigma^2/[kappa*(2-eta0*kappa)]，初始化后固定 q0，不随干预 u 求导。
eta=eta0*exp(u)

解析响应：
chi_H=-H*eta0*a^2 + eta0*sigma^2/(2-eta0*kappa)^2 * [1-(1-eta0*kappa)^(2H)].

取 eta0=0.1, a=0.15, kappa=1, sigma=1。H=4 的 chi 应为正，H=16 和 64 应为负。FP64 递推 + central FD 应与解析式一致；另测 sigma=0 只保留负的进展项。主单元测试使用确定性的 q 递推，Monte Carlo 仅作可选可视化，不让抽样误差掩盖实现错误。

测试应能抓住：exp(u) 误写成 1+u；q0 错误地随 u 更新；pulse4 与 persistent 混淆；H 的 off-by-one；Gain 符号错误。

### B2. 稳态不等于详细平衡

二维 Gaussian AR(1)：r_next=M*r+eps，Cov(eps)=Xi=0.36 I。
分别取：
- M=0.8 I；
- M=diag(0.8,-0.8)；
- M=0.8 Rot(0.35)。

三者均有 Sigma=I，满足 Sigma=M Sigma M^T+Xi。
检验 lag 反对称量 A_lag=M Sigma-Sigma M^T：前两者为零，旋转情形非零。

意义：同一个稳定协方差不决定是否可逆；一步交替振荡也不自动意味着非平衡。该结论限定 Gaussian Markov 模型且所有坐标采用同一种时间反演规则，不能直接外推到完整 AdamW 的物理熵产生。

### B3. 决策价值与边界饱和

在线性局部损失 Delta L=chi*u，u∈[-delta,delta] 下，验证：
V_sign=delta*(mean(abs(chi))-abs(mean(chi))).
全部 chi 同号时 V_sign=0。

对 Delta L=chi*u+0.5*Q*u^2，Q>0，验证：
u_star=clip(-chi/Q,-delta,delta)。
用全负 chi 但内部最优幅度不同的例子验证“单通道/同符号不排除幅度自适应”。对 Q<=0 不能应用内部极小值公式，应直接比较边界与零动作。

### B4. 有限集合分解恒等式

对同一个 checkpoint 的 S 条流，定义 theta_bar=(1/S)sum_s theta_s；
L_ens=(1/S)sum_s L(theta_s)，L_center=L(theta_bar)，J_disp=L_ens-L_center。

验证 L_ens=L_center+J_disp，以及三者 central FD 的同一恒等式。J_disp 不强制非负。禁止把不同 seed 的参数取平均。

## 6. 阶段 C：2-cell 数值 smoke

从主配置取第一个 checkpoint 和 2 条 development streams，先跑 smoke，不额外按结果挑样本。

必须检查：
- u=0 的新 runner 与冻结原 runner 的 theta/m/v/step/loss 一致；
- 相同状态、相同流、相同 RNG 重放能达到既有数值精度；
- 同一个 u 的 pulse4 与 persistent 在 H=4 完全相同；
- release 后 multiplier 回到 1，但 m/v 和已形成的参数状态保留，scheduler 使用正确绝对 step；
- 所有 arms 使用相同 batch、dropout 等随机流；
- evaluation 不消耗或改变训练 RNG，eval 后恢复训练模式；
- 实际 microbatch/token 数一致，不把多个 microbatch 当多个 optimizer steps。

精度：优先复用已有经过验证的 FP32 响应诊断配置。若原实验为 BF16，而本轮需 FP32，则全套配对 arms 和读数统一到同一 FP32 配置，另做与原配置的连接说明；不要将 FP32 和 BF16 的不同函数当成同一导数校验。保留原 attention backend 的语义；需要更换 backend 时单独记录。

在 H=4 用已有验证过的解析/伴随响应核对 central FD。不能自行重写不完整的优化器微分。若接口不可用，标为 ANALYTIC_CHECK_UNAVAILABLE；只继续有充分 FD 数值一致性支持的诊断，不把纯 FD 结果称为伴随验证。

测试阈值优先沿用已存在的可靠精度门槛。新增误差标准由 CPU 精确测试和 u=0 重放噪声确定，在主结果前写入 config。不要用接近零的 chi 做相对误差除法；同时报告绝对误差和有分辨率单元的相对误差/符号一致性。

二阶 Q 的稳定性是科学诊断，不是必须通过的指标。Q 不可靠可以继续收集一阶和释放实验，但不得输出曲率控制结论。

若 restore、RNG、schedule、no-op 或 H=4 分支同一性失败，停止扩展 GPU 主实验，修复并保留失败证据。不要通过放宽标准掩盖程序问题。

## 7. 阶段 D：主实验，仅 5M 现有模型与局部续训

u_grid=[-0.05,-0.025,0,+0.025,+0.05]
readout_H=[4,16,64]

每个 cell 只需 9 条 64-step 轨迹：
1. native：全程 u=0。
2. persistent(u)：4 个非零 u，从 step 0 到 63 始终干预。
3. pulse4_release(u)：4 个非零 u，仅前 4 个 optimizer steps 干预；其后恢复 native LR。

prefix 一致时可以复用共同前缀，但必须确认包括 optimizer/RNG/scheduler 的完整分叉状态一致。不要把 pulse4 称为 A0P 的单步 impulse。

记录 H=4/16/64 的 readout；它们来自同一条 64-step nested trajectory，而不是每个 H 重新抽流。Gain 的 baseline 必须来自相同 cell、相同 H。

真实 optimizer 更新上限：本轮最多 18,000 次，含 smoke、复核和重放；主计划 24*9*64=13,824 次，复用前缀可以减少。禁止自动扩大 checkpoint、seed、H、模型或扫描密度。评估、HVP/adjoint 和 CPU 开销另行计数计时，不能隐藏。读取已有缓存只在完整语义一致时复用。

如果本地已有更严格的资源限制，采用更严格者。执行中需要超出上限时保存现有结果并报告原因，不自动追加实验。

对每条轨迹记录逐步 scalar diagnostics：真实 LR、clip 是否激活、gradient/update norm、相邻更新余弦、训练 loss、有效 tokens 和 wall time。分层 norm 仅在已有接口易于获取时采集；不要因此计算全 Hessian 或全参数协方差。

## 8. 阶段 E：流间平均轨迹/分散分解

对每个固定 (seed, checkpoint, protocol, u, H)，在其 4 条流之间做：
theta_bar=(1/4)sum_s theta_s
L_ens=(1/4)sum_s L_val(theta_s)
L_center=L_val(theta_bar)
J_disp=L_ens-L_center。

注意：
- 使用同一个固定 V_eval、同一 evaluation precision/reduction。
- theta_bar 是参数平均，不是预测概率平均或多个 loss 的平均。
- 四条流都起自相同 checkpoint；不得跨 seeds、checkpoints、H、protocol 或 u 混合平均。
- 不需要保存完整协方差，也不需要假设 Hessian 正定。
- 对 tied parameters 正确维护绑定；若存在训练改变的 buffers，先确认 averaging 语义，不擅自平均不适用的 buffer。
- 为节省磁盘，按组在 CPU 累加 theta 并即时评估 theta_bar，不默认持久化所有 theta/m/v 轨迹；只保存复核需要的少量状态和 scalar 原始数据。

用相同 FD 公式得到 chi_ens、chi_center、chi_disp，检查 chi_ens=chi_center+chi_disp。

本节的“中心”只是有限流集合平均参数；不能直接命名为 central flow、真实慢流形或平衡态。J_disp 只覆盖流间分散，不能排除所有流同步发生的确定性振荡。结合相邻 update cosine 可提示交替动态，但它也不能单独证明振荡机制。

4 条流的统计分辨率有限。任何均值/方差机制判断都应标为 pilot 证据，而非完整概率分布的识别。

## 9. 阶段 F：分析与防止自我验证

### F1. 干预时长与后续价值

逐 cell 比较 persistent 与 pulse4_release 在 H=4/16/64 的 Gain 和 chi。
- 若只有 persistent 反转，说明继续施加干预是不可忽略的因素；不能宣称纯状态后效应已经识别。
- 若 release 后也反转，说明前缀产生的状态效果相对于读出时间具有变化；它并不单独区分 theta 几何、m/v 记忆或其他机制。
- 正常/反转都报告，不只画支持理论的单元。

### F2. 二阶曲率与幅度

报告两种 a 下的 chi/Q 差异，Q 的分辨率、符号和不确定性。
在 development streams 使用 -0.025/0/+0.025 拟合二阶局部曲线，检查对 ±0.05 的幅度外推；另在 audit streams 检查平均响应的流外泛化。这两种检验分别标记，不把同一条流的 FD 拟合与评分当独立验证。

若 Q>0 且可辨认，记录 u_star=-chi/Q 是否落在区间内部，以及边界饱和率。若 Q<=0、接近零或不可辨认，明确分类，不输出貌似稳定的 Newton 控制器。

本轮不部署在线 controller。任何使用真实未来 rollout 或 V_eval 选出的动作，标为 hindsight/oracle，不称为可在线获得。

### F3. 相对于固定策略的信息价值

必须保留 native 与固定 exp(+0.05) 两个基线。也报告本轮网格的 best-fixed，但若从同一评估样本选取，仅可称 hindsight best-fixed。

可输出有限网格 hindsight upper bound：
V_grid = min_j mean_i L_i(u_j) - mean_i min_j L_i(u_j) >= 0。

它衡量本样本中“事后知道每个 cell 的完整损失曲线”可能提供的最大增益，不是部署收益。对照同符号线性极限 V_sign。不能把 oracle 利用未来数据的价值归因给只见当前状态的控制器。

### F4. 精度与统计

优先报告逐 seed 的平均、逐 checkpoint/stream 原始值、效应量和范围。时间点/多个 streams 不是新的独立训练 seeds。3 seeds 的 pilot 不输出貌似强有力的小 p 值；如使用 seed-cluster bootstrap，明确小 cluster 数下的不可靠性和探索性质。

记录每个指标的真实单位，NLL 使用正确 token denominator。观察到的 Gain 低于数值分辨率时标为 UNRESOLVED。

## 10. 输出和验收

输出目录必须包含：
- audit.md：真实接口、数据、旧结果复核、缺失信息；
- config.json：主实验前冻结的定义、样本、流、幅度、精度、预算和阈值；
- theory_tests.json 与 numerical_checks.json：真实测试结果，不能手填 PASS；
- cell_metrics.csv：cell/protocol/u/H/native loss/arm loss/Delta/Gain/流类别/数值状态；
- response_curvature.csv：chi、Q、幅度一致性、内部最优点和饱和状态；
- decomposition.csv：L_ens、L_center、J_disp 和相应响应；
- summary.json 与 REPORT.md：结论、未解决项、真实运行命令、资源计数；
- figures/：至少脉冲释放对持续干预、短长时域响应、loss-vs-u、中心/分散响应分解四类图。

图从已保存的原始表生成，不能硬编码结果。轴标签明确 H、loss difference/gain 的符号、u 是 log multiplier。保留负结果和不确定性。

REPORT.md 必须分别给出：
1. implementation/numerical status；
2. intervention-duration hypothesis；
3. center/dispersion decomposition evidence；
4. second-order amplitude identifiability；
5. local quasi-stationarity：本轮未直接检验时为 NOT_TESTED，不能因为曲线平滑标为 PASS；
6. architecture conclusion：NOT_TESTED；
7. training energy saving：NOT_ESTABLISHED。

每个科学假设采用 SUPPORTED / NOT_SUPPORTED / UNRESOLVED / NOT_TESTED，并写限定的样本与时域。数值错误单独标 FAIL。科学负结果不算工程失败，也不授权你改研究目标直到找到正结果。

最后用简短文字回答：本轮最强证据是什么、哪个机制仍不可区分、Q 是否可用、是否值得进入 B1。附可复核的输出路径、实际命令和测试摘要，不只说“完成”。

## 11. 后续 B1/B2 路线，仅作理解，不在本轮自动执行

### B1：条件准稳态与闭合

在相同 checkpoint 的多条独立 native continuation 上，先定义固定的低维坐标/稳定子空间和中心估计，再比较：
- 一阶局部 Markov 模型；
- 带有限记忆的模型；
- 时变系数模型。

所有投影、窗口、正则化只用 development 数据决定，预测 audit 流和未见干预。不要用为每条测试轨迹量身拟合的中心，制造残差稳定。

离散局部模型：r_next=M_t r+eps，Cov(eps)=Xi_t。
协方差预测：Sigma_next=M_t Sigma_t M_t^T+Xi_t。
只有当局部状态可近似稳定、残差创新假设足够可信且 relaxation 比 coefficient drift 快，才考虑冻结条件稳态：Sigma*=M Sigma* M^T+Xi。

检验 covariance recursion 是否独立预测 held-out 数据，而不是直接把 Xi 定义成残差使等式恒等成立。学习到的 M 若非正规，还应检查 ||M^h|| 的瞬态放大，而不只检查谱半径。快振荡不等于快弛豫。

在 Gaussian Markov 且所有变量采用普通偶时间反演的条件下，M Sigma*=Sigma* M^T 是详细平衡判据；若包括 m 或其他有特殊时间反演候选的变量，先明确反演规则，不能从反对称相关量直接给出物理熵产生。

无稳定性/闭合证据时转向非平稳记忆模型，而不是强行配平衡态。

### B2：响应与涨落的同一生成模型

对低维条件 Gaussian 模型 r_next=f(r,u)+eps，Xi(u)>0，路径 action（固定初始分布）为：
S_u = 0.5 sum_k [eps_k^T Xi_k(u)^(-1) eps_k + log det(2*pi*Xi_k(u))]。

有限时间 LR 响应应包括 drift 和 covariance 两部分；只拟合 additive-force response 不足以代表 LR response。

在共同可微密度和积分求导条件下：d E[L]/du = -Cov(L,d S_u/du)。

要求同一组参数同时预测均值、协方差和未见干预响应。只把梯度展开重写成 action，不算额外科学贡献；必须说明新模型相对于冻结 tangent/简单 AR 的预测增益或可压缩性。

不把这里的概率 action、统计 irreversibility 或 NLL 改名为 GPU 焦耳。硬件节能只能在后续同任务目标、同预算、包含诊断开销的比较中测量。

## 12. 理论参考（阅读线索，不替代本仓库证据）

- Wen et al., Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective, arXiv:2410.05192。慢进展与高曲率振荡分离的候选机制。
- Cohen et al., Understanding Optimization in Deep Learning with Central Flows, arXiv:2410.24206。时间平均优化轨迹；不能直接当完整 AdamW 的已证定理。
- Godreche & Luck, Characterising the nonequilibrium stationary states of Ornstein-Uhlenbeck processes, arXiv:1807.00694。稳态协方差、概率流、详细平衡和响应。
- Martin, Siggia & Rose, Statistical Dynamics of Classical Systems, Phys. Rev. A 8, 423 (1973), DOI:10.1103/PhysRevA.8.423。随机动力学生成泛函与响应场。
- Bordelon, Chaudhry & Pehlevan, Infinite Limits of Multi-head Transformer Dynamics, arXiv:2405.15712。Transformer 动态平均场；精确理论主要针对 SGD，Adam 部分有限制。
- Hatano & Sasa, Steady State Thermodynamics of Langevin Systems, arXiv:cond-mat/0010405。连续稳态族与驱动；引用其热力学解释前必须检查模型条件。

## 执行结束条件

完成 B0 的真实实现、理论测试、允许规模的实验与报告后停止。若被数据缺失、数值错误、设备不可用或预算阻塞，保留已完成部分并提交清楚的阻塞报告。不要虚构补齐结果，不自动进入 B1/B2、26M/100M 或新架构实验。
