# FR C0-GTTR：方向几何—moment 跟踪—位移传递—目标读出的验证

用途：交给当前 FR 仓库中的 Claude Code 执行。
状态：实验协议与执行提示词；不是远程运行结果。设计者没有访问远程仓库。

## 0. 本轮做什么、不做什么

在当前环境中直接完成：现场审查 → 配置冻结 → CPU 理论测试 → GPU smoke → 预算内机制 pilot → 结果表、图和报告。不要只返回一个计划；也不要超出本协议扩大实验。

本轮只检验同一个 AdamW 模型的三个层次：

1. 精确更新、位移分解和完整切向导数是否与 frozen runner 一致。
2. 当前真实更新经过的训练梯度场，能否由局部几何近似预测得比“冻结训练梯度”更好；这种改进能否通过实际 clipping、m/v 和 P 的耦合，传递到下一次更新。
3. 现有短长收益反转能否被精确的目标读出记账定位；哪些量有可分辨信号，值得下一阶段尝试提前预测。

重要：第 3 项在本轮是事后机制诊断，不是 H=16/64 开环预测。成功通过代数恒等式、单步梯度场预测或事后累计分解，都不能被写成“已经解释并提前预测长期反转”。

本轮不训练新模型，不开发 controller，不搜索学习率，不调整 beta，不做新 batch 策略，不再扩展 HMM/GMM，不全面扫描 Hessian，不拟合记忆核，不声称低秩、平衡态或节能。

科研优先级：训练语义正确 > 数值结果可核对 > 完成有限问题并出图 > 工程封装。复用现有依赖与 frozen primitives；不重构训练框架，不引入数据库/服务/复杂调度，不生成大型审计归档，不递归 hash 仓库，不自动提交或推送，不修改模型接入配置，不干扰其他任务。

---

## 1. 必须先现场读取的材料

读取实际存在的 FR_PROJECT_STATUS、FR_EXPERIMENT_INDEX，以及以下实验的 config / report / 相关原始表：

- A0C、A0E2；查找 A0F2 的原始报告与 optimizer-only B vs complete C 比较。
- A0O、A0P、A0Q、A0S、A0T、A0T-extra、A0U。
- B0、B0.5、B0.6 的实际 budget log、配置、逐 cell 数据和已使用的接口。
- frozen AdamW 单步、完整恢复、stream 构造、validation readout、HVP/clip-JVP primitives。

A0F2 缺失不阻塞本轮，记录 SOURCE_NOT_FOUND；不把 retrospective 或用户转述改写为独立原始验证。不要搜索无关目录或重做旧项目。

输出简短 audit.md：真实函数/路径映射、参数组与 decay mask、精度与 attention backend、clip 实现、beta/epsilon/step 语义、梯度累积、每 microbatch 的样本和 token 数、实际使用的验证集合及随机源。

已知用户最后报告总账 17,577/18,000，但该数字必须现场核对，不能直接作为已核验值。旧日志的修订只能有明确依据；保留原值及修订说明，不覆盖历史证据。

若本仓库已有 C0 标签，使用同含义的新输出版本并记录映射；不要覆盖旧实验。建议输出名 outputs/fr_c0_gttr_validation_v01/，这是新建建议路径，不是声称已存在。

---

## 2. 预算：不借用户的“继续”自动扩大旧上限

### 2.1 真正训练更新的硬上限

- 旧项目总上限继续为 18,000 optimizer updates。
- 本轮新增上限 408 次真实 optimizer updates，且必须同时满足旧项目剩余额度。
- smoke：1 个 cell × 3 个 arm × 4 steps × 2 次（诊断开/关）=24 updates。
- main：2 个 cells × 3 个 arms × 64 steps=384 updates。
- 若实际旧账为 17,577，全部完成后为 17,985，余 15；不是重新获得 18,000 次额度。
- 失败尝试、重放、stress、重试都计入实际消耗。CPU 合成测试不计入 GPU 训练更新。

如果余额不足完成该配置，不私自换 seeds/减少 arms 后沿用原 headline，不把诊断训练标成“只分析”来免计数。完成不需 GPU 的审查与测试，输出 BUDGET_BLOCKED 或 PARTIAL，并明确剩余需求。不得自动开启新预算轮次。

### 2.2 不以 updates 隐藏 probe 开销

本轮另设保守工程上限：

- GPU active diagnostics 总墙钟预算 60 分钟；单个无法中断的 kernel 结束后检查。初次 smoke 根据实测给 ETA，不承诺固定耗时。
- 非训练 gradient evaluations ≤1,400；HVP 调用 ≤256。
- 非训练 forward calls ≤2,400；每次梯度/HVP内部实际 forward 也进入 forward 计数，不重复或漏记。
- 代数 shadow-step 次数独立记录，不算真实训练更新，但报告数量、耗时。
- 显存不足或预计超过上限，保存阶段结果并停止。不要靠默默缩小 probe bank 或替换目标完成“同一个”实验。

这些是工作量限制，不是物理能耗估计。不要从耗时或理论 FLOPs 推断节能比例。

---

## 3. 当前证据边界：作为约束，而不是待迎合的结论

- A0C/A0E2 支持完整 stepwise tangent 与未来 gradient / clip feedback 的必要性，具体范围以原报告为准。
- A0O 是有限时域 LR-input response 的既有验证，不重新发明一个不含 future gradient feedback 的替代导数。
- A0S 主要支持固定幅度下 Gain ≈ delta * |chi| 的校准，不命名成新的可控性定律。
- A0P 限制所测试的 impulse 预测和简单核拟合，不证明长期导数不存在。
- A0T-extra 的历史反转案例是发现案例，不是独立确认集；A0U 的 fixed-up baseline 必须保留。
- B0 的中心/分散分解不等于慢流/快流；pulse4/persistent 比例不是记忆保留率。
- B0.5 的一次 m/v 移植不允许删除 moments 或断言无 optimizer memory。
- B0.6 headline NOT_TESTED；Q1/Q3a 不支持当前分区，不排除所有真实区域转换。

不得预设本轮必须得到 SUPPORT。允许数值不分辨、模型失败和局部几何范围不足。

---

## 4. 冻结科学定义与记号

### 4.1 时间、数据、目标分开

窗口 h=0 表示恢复 checkpoint 后、任何新更新之前；h=4 表示完成 4 次 optimizer updates。所有偏差校正、scheduler 和 moments 使用真实累计 step，不在窗口重新编号。

固定结构/目标/数据分布时总体损失 L_train(theta) 固定；H(theta_t) 可随所处位置变化。表示是参数的观测函数，不无依据增加独立 rho 状态。

原始当前训练梯度 g_t = grad_theta ell(theta_t; omega_t)，omega 包含实际 batch 与必要随机性；c_t = C_tau(g_t) 是实际进入 moments 的梯度。

定义训练场与二阶原点矩：
mu_t = E[c_t | F_t]，nu_t = E[c_t^2 | F_t]。
F_t 的含义是当前随机训练 probe 尚未揭示前的状态/历史信息。实际固定 future-stream 的逐路径实验与新 batch 重采样诊断分开。

固定独立读出 V_diag(theta)，q_t = grad V_diag(theta_t)。最终既有收益读数使用冻结的 V_eval；两者可不同，不能把小 V_diag 的收益直接命名为原 V_eval 收益。

### 4.2 真实更新与偏差校正

bar_m_t / bar_v_t 指处理当前 c_t 后、用于本次位移的 bias-corrected moments：

bar_m_t = a_t bar_m_{t-1} + (1-a_t)c_t
bar_v_t = b_t bar_v_{t-1} + (1-b_t)c_t^2
P_t = diag(1/(sqrt(bar_v_t)+eps))
r_t = P_t bar_m_t
d_t = -eta_t r_t
s_t = d_t - eta_t lambda_wd theta_t

标准零初始化、固定 beta 且累计步数从 t=0 计数时：
a_t = beta1*(1-beta1^t)/(1-beta1^(t+1))
b_t = beta2*(1-beta2^t)/(1-beta2^(t+1))。

参数组、decay mask、初始化与非标准分母以 frozen code 为准。beta 本轮不改变。

### 4.3 Transport 不等于 tangent

对 eta_{t-1}>0：

d_t = T_t d_{t-1} - eta_t(1-a_t)P_t c_t
T_t = a_t*(eta_t/eta_{t-1})*P_t*P_{t-1}^{-1}。

T_t 是 realized displacement carry-over coefficient；它依赖当前 c_t，经由 bar_v_t 与 forcing 耦合。不能拿 T_t 单独替代 A_t=dF_t/dz_t。

checkpoint 窗口首步若缺上一实际位移，不伪造 d_{-1}；该恒等式从具备真实相邻记录的下一步开始检查。

### 4.4 当前步 log-LR 输入

u = log(eta/eta_native)，eta=eta_native*exp(u)。
固定当前输入与更新前状态：b_eta = [s_t,0,0]。
禁止把 d_t 已含 eta 的位移又乘一次 eta。旧实验若用 1.05，保留原语义，不能静默重命名为 exp(0.05)。

### 4.5 训练传播与读出几何

H_batch = Hessian ell(theta; omega)
G_t = d c_t/d theta = J_clip(g_t) H_batch
H_V = Hessian V_diag(theta)

理想全局 L2 clip 的非边界 Jacobian：
J=I（inactive）；J=tau/||g||*(I-gg^T/||g||^2)（active）。
实际实现可能含数值稳定项，应按实际导数，不暗改实现。

G_t 一般不对称；E[J_clip H_batch] 不能替成 J_clip(E[g]) H_population。
P^(1/2) H_batch P^(1/2) 只是特定预条件曲率，不是完整 AdamW Jacobian。

完整扰动链（仅示固定 beta/tau/decay 与 log-LR 输入）：
delta_c = G_t delta_theta
delta_m = a_t delta_m_prev + (1-a_t)delta_c
delta_v = b_t delta_v_prev + 2(1-b_t)c_t*delta_c
Omega_t = diag(bar_m_t/[2sqrt(bar_v_t)(sqrt(bar_v_t)+eps)^2])
delta_theta_next = (1-eta_t lambda)delta_theta - eta_t P_t delta_m
                   + eta_t Omega_t delta_v + s_t delta_u。

Omega 式只在相关 v>0 处使用；零矩坐标与 clip 边界单独标注，不添加隐式 epsilon 改变被研究函数。

### 4.6 Tracking 与 Readout

e_m = bar_m - mu；e_v = bar_v - nu。
两者递推分别是：
e_m,t = a_t e_m,t-1 - a_t(mu_t-mu_t-1) + (1-a_t)(c_t-mu_t)
e_v,t = b_t e_v,t-1 - b_t(nu_t-nu_t-1) + (1-b_t)(c_t^2-nu_t)。

W_field = eta*q^T P mu
W_track = eta*q^T P e_m
W_wd = eta*lambda*q^T theta
W_total = -q^T s = W_field+W_track+W_wd
K_V = V(theta+s)-V(theta)+W_total
Delta V = -W_total+K_V。

K_V 是精确读出余项，不保证正；若无独立曲率测量，不能把“按差值定义 K_V 后恒等式成立”称为 Hessian 机制验证。
以上是 realized 路径记账，份额不等于互相独立的因果效应。

---

## 5. 样本：2-cell mechanism pilot，不伪装成统计确认实验

只复用已存在 checkpoint，不新训练 checkpoint。

- Cell R：按 (seed, checkpoint_step, stream_id) 固定排序，取 A0T-extra 两个已知 discriminative reversal cases 的第一个。必须从原始表复核 ID、动作时长与收益符号，不能只靠转述硬编码。
- Cell C：从已有制品选一个已知 non-reversal control。优先同 checkpoint、另一条 stream；若不存在，再优先同 seed、最近 checkpoint。多个候选按恢复时 native loss 的距离排序，最后按 ID 打破并列。
- 两个均为历史已知案例，只用于机制 pilot；不能因指定了 C 就声称是随机确认样本。
- 不把 probe batches 或 64 个时间点当独立训练 seeds；本轮不作显著性或总体推广结论。
- 若历史 R 无法恢复或配对协议不一致，报告 SOURCE/STATE_UNAVAILABLE，不临时搜索更“漂亮”的案例。

固定三臂：
A: native，u=0；B+: persistent u=+0.05；B-: persistent u=-0.05。
全程最多64步，在 h=4/16/64 使用同一 nested trajectory 的读数。三臂同初始完整状态、同训练 batch/dropout 随机流。

本轮不新增 pulse4 臂。已有 B0 pulse4 数据可以只读并单独解释，但不得用 persistent 的反转推断“同一次脉冲的后效应反转”。

先把选择规则、真实 IDs 和旧结果已知性写入 config，再运行 GPU。

---

## 6. CPU 理论与实现测试：少量明确单元测试，不构建新框架

先测试，再实现；使用已有 unittest/pytest。CPU 小模型用 FP64。建议 rtol=1e-7、atol=1e-9 校验精确解析对 finite difference，若有限差分尺度不合适，先在 CPU 校准并记录，不能看真实主结果再改门槛。

T1：完整 AdamW 与 bias-corrected m/v 改写一致。包含非恒定 LR、非零 decay、mask 参数组、clip active/inactive、恢复累计步号。首步 a=0/b=0 的初始化不除零。

T2：realized d_t carry-over 恒等式；加入非恒定 P 和 eta，错误地忽略 P_t/P_{t-1} 或重复乘 eta 的实现必须不能通过。

T3：完整状态 + log-LR 的 analytic tangent 对 central FD。用小维正定二次训练目标；active/inactive clip 都测；读出使用与训练 Hessian 不同的二次 V。v 坐标取严格正，边界另测并标注。

T4：训练场 mu 与 validation q 不同。检查 e_m 对 mu 的递推以及 readout 三项。把 q 误替 mu 的版本应无法通过语义测试；跟踪递推数值恒等式本身不验证 mu 估计正确。

T5：clip 顺序及非对称性。检查 C(E[g])!=E[C(g)] 的具体反例；构造不交换的 J_clip/H，验证 JH-(JH)^T 非零。真实 runner 使用哪种全局范数/稳定项，要写清楚。

T6：update nonlinear expectation。用两点梯度分布，例如标量 c∈{1,3}、概率各1/2，m_prev=0、v_prev=1、a=b=0.5、eps=0.01、eta=1、decay=0；验证 mean(P(c)*m(c)) 与用 mean(c)、mean(c^2)代入形成的更新不相等。禁止把均值梯度放进 Adam 后称为平均 Adam 更新。

T7：固定二次谷的 pulse4 响应反转。L=(x^2+6y^2)/2，(x0,y0)=(1,1)，native eta=0.25；只改变前4步 eta。解析 dL_H/deta_pulse = -4 sum_i lambda_i^2 x_i0^2 (1-eta0 lambda_i)^(2H-1)。H4=0.591064453125>0，H16≈-0.0005356792152846719<0。这个例子只说明反转不要求跨谷，不能直接认作 Transformer 机制。

T8：本轮 prediction adapters 的正负对照。对于无 clip 切换的二次训练目标，g(theta+s)=g(theta)+Hs；冻结梯度模型在 Hs 非零时有误差，几何模型应精确（FP64）。再构造有 clip active-set 改变的二次例子：raw affine gradient 后重新按真实 C clip 的预测应准确，而固定 clip-JVP 的有限位移近似允许失准。最后加一个平滑非二次例子，验证模型不被写成“无论如何 PASS”。

T9：readout 与 RNG。任意两步验证 telescope；K_V允许负。smoke adapter的读出不能改变训练 RNG、grad buffers、mode 或 optimizer state。

所有测试只确认公式/实现和特定合成实例，不把 ALL_PASS 写成真实模型假设成立。

---

## 7. GPU smoke：24 updates，检查 recorder 不改变轨迹

在 Cell R：A/B+/B- 各跑4步，两次：一次不插入新诊断，一次插入与 main 相同类型的诊断。总24次更新。

检查：
1. 参数、m、v、step、scheduler、训练 RNG 等最终状态按旧 frozen 精度要求一致；若旧门支持 bitwise，则沿用，不人为放宽。
2. 与可用旧制品的共享字段一致；旧记录没有保存完整参数时，不把标量匹配写成完整状态逐位复核。
3. V_diag梯度不污染真实训练的 p.grad，不混入optimizer.step；评估后恢复训练/attention/dropout状态与RNG。
4. 实际 tokens = grad_accum × microbatch_size × effective_seq_len（变长则逐batch数有效token），不得预设 1024 无条件正确。
5. eta_native*exp(u) 与绝对 step 正确；decay随真实eta变化。
6. 针对一个小 probe 使用既有可靠的 HVP/clip-JVP primitive；沿用其数值精度门。不要把 BF16 forward proxy 的解析导数强行用另一FP32函数的FD验证。若可用的proxy不能分辨信号，标注 NUMERICAL_UNRESOLVED，停止相关结论。

smoke的诊断计数纳入上限。失败先留证并修复，但新增真实更新一律计数；预算不足重试就停止，不删失败账。

---

## 8. Main A：逐步真实读出记账（解释作用，不宣称提前预测）

每个 cell/arm 的 h=0..64：

- 记录冻结的训练diag、eta、clip active、grad/update norms；已有数据可复用但必须语义相同。
- 固定小规模 V_diag：优先使用现有 V_score 固定子集，最多一个真实有效训练 batch 的token量；它不与 V_eval 重合，不能根据结果挑样本。若无现成集合，按固定样本ID规则建集合并事先冻结。
- 每个实际更新前计算 V_diag(theta_t) 和 q_t；更新后通过下一节点值或终点值获得 Delta V。
- 记录 W_total=-q_t^T s_t、K_V=Delta V+W_total。流式累计，不保存384份完整gradient。
- 在 h=4/16/64 读取旧 V_eval，核对三臂收益与历史记录（历史设置相同才作数值复核）。
- 记录 T_t 的代数误差。其元素/范数仅为描述，不能使用它单独重演counterfactual。

输出：每个case单独画累计 W_total 差、K_V差及真实 Gain；V_diag与V_eval收益分开。定位优势在哪些更新段被抵消，不解释成独立因果贡献。

默认不把 K_V改名“曲率惩罚”，也不默认 q^T s<0。无差异或不反转也按真实结果记录。

---

## 9. Main B：同一真实状态上的训练场与局部几何预测

### 9.1 Probe配置与允许的信息

每条64步轨迹仅在更新前的局部索引 t∈{0,4,16,48} 做probe，共 2×3×4=24 个 anchors。

每个cell固定独立训练probe库：8个 development batches、8个 audit batches。采用与真实有效训练batch相同的梯度累积和全局clip语义。不得将8个probe合并成一个更大的batch后只clip一次。

- probe来自训练分布，不是validation。
- development与audit独立，均与本轮真实future stream的具体batch不重叠；不能用尚未揭示的真实future梯度。
- 每个probe的dropout/其他随机种子固定；在theta与theta+s两个位置复用同一probe随机性。
- 所有arms使用相同probe索引与配对随机性；不同anchors复用固定库，因此统计并非独立。
- 这是有限probe分布下的条件重采样诊断，不等于已知真实future stream的oracle。

当前实际更新得到 s_t 后，预测器可以使用 s_t（它是当前batch和当前更新的已知结果）。该检验预测的是这一步造成的训练场变化，不声称在当前batch揭示之前就已预测 s_t。

诊断计算只能在冻结副本或可恢复的functional state进行；不能推进真实训练状态。

### 9.2 先写预测，后读audit终点

在theta_t：
- dev库计算原始 g_j、c_j=C(g_j) 和 H_j s_t，调用已有HVP；不形成全Hessian。
- audit库只计算原始 g_j、c_j 及c_j^2。

固定三个候选，均无可训练参数：

M0（无几何反馈）：pred g_j^+ = g_j。
Mgeom（PRIMARY）：pred g_j^+ = g_j + H_j s_t；随后用真实 C 重新clip。
MnoJ（诊断消融）：预测Delta c为平均H_j s_t，忽略clip变换。只用于检查clip作用，不当作候选训练算法。

可同时记录 infinitesimal J_clip(g_j) H_j s_t 与 Mgeom finite-displacement delta的差异，成本复用现有HVP；不另起第四拟合模型。

先将预测向量的必要统计/摘要写入prediction记录，再计算audit在theta_t+s_t的梯度。不要通过audit终点选择模型、幅度、方向或重算更有利的HVP。

Audit目标：
y_j = C(g_j(theta_t+s_t))-C(g_j(theta_t))
y_mu = mean_audit(y_j)。

Mgeom预测：
p_mu = mean_dev[C(g_j+H_j s_t)-C(g_j)]。
M0预测是0。

必须同时报告：
- paired Delta mu 的预测误差（主要检验几何漂移）；
- 两bank在同一theta_t的mean c差异（初值/采样不确定性）；
- 如报告完整mu_next预测，初值误差与漂移误差分开，不偷偷用audit初值替换后称为部署预测。

记录各probe clip active-set是否改变。不要把crossing全部删掉；整体误差与non-crossing分层均报告。

### 9.3 将同一几何近似传递到真正的moment/预条件耦合

本项是一个固定后更新状态上的 shadow next-update 诊断，不是运行未来真实batch。

固定共同后更新状态：theta'=theta_t+s_t，真实m/v（刚完成当前步），正确的下一累计step和各arm下一eta。
定义 U(g;common_state)：对输入raw gradient g执行真实 clipping、moments、bias correction和P计算，返回下一次参数位移；绝不修改main状态。

Development预测：
p_U = mean_dev[ U(g_j+H_j s_t;common_state) - U(g_j;common_state) ]。

Audit目标：
y_U = mean_audit[ U(g_j(theta_t+s_t);common_state) - U(g_j(theta_t);common_state) ]。

M0预测y_U为0。

两个U使用同一个common_state，因此分离了“参数位移引起的梯度场变化对下一次更新的影响”。这不是全部next-step变化：直接参数项、旧moments和eta在此对比里被固定，不能遗漏这个限定。

必须先对每个probe做非线性U，再平均；禁止用mean g或mean c直接跑一次Adam来代替。

只增加一个零新gradient成本的消融：stop-v variant，在一对U中让新梯度更新m，但P固定为同probe基准g形成的P。它只检查预条件反馈的增量，清楚标注人工诊断，不替换真实模型，也不依结果选择它为新optimizer。

若case的y_U不可分辨，不能由模型误差看似小宣称moments机制已验证。

### 9.4 Tracking/readout稀疏诊断

在anchor的同一theta上，用audit库计算hat mu、hat nu及误差估计，形成：
hat e_m=bar m_t-hat mu；hat e_v=bar v_t-hat nu；
W_field/W_track/W_wd，用实际当前P_t与同一q_t。

可在相邻theta_t、theta_t+s_t及下一实际步moments可用后检查tracking恒等式，但这只是代数检查，不证明hat mu正确。

不要将m/v原始范数大小直接比较为因果重要性。优先报告它们对应的实际更新差异或读出投影。

训练场几何读数可用mean raw-gradient turning与HVP投影，但不命名为“识别山谷编号”。若估计的mu场经clip后不保守，不把它的Jacobian强行对称化。

---

## 10. 误差、分辨率与判定：小pilot不制造显著性

### 10.1 采样分辨率

对每个anchor的audit向量目标y_j（分别用于y_mu、y_U），K=8：

ybar = (1/K)sum y_j
SE_vec = sqrt[ sum ||y_j-ybar||^2 / (K*(K-1)) ]。

它是均值向量采样误差尺度的估计，不是严格多维置信球，也不是独立seed统计。另保存两个4-probe半库均值的差值与scalar投影误差。

numerical_floor来自smoke重复诊断；定义resolved = ||ybar|| > 3*sqrt(SE_vec^2 + numerical_floor^2)。这个3倍规则是本轮固定的工程分辨率门，不称为显著性。

development预测同样计算采样误差。预测bank不稳定时另外标 PREDICTION_SAMPLING_UNRESOLVED，不通过加probe/挑seed救结论。

### 10.2 主预测指标

对每个cell，单独计算signal-resolved anchors上的：
E_M = sqrt[ sum ||prediction_M - ybar||^2 / sum ||ybar||^2 ]。
M0在这个定义下为1。同时报告绝对误差与全部anchors结果，不能只呈现筛选后子集。

可在q或当前位移等固定读出方向报告scalar误差，所有方向必须在看到audit目标之前确定。不要为提高拟合事后挑方向。

判定：
- 两个cells各有至少6个resolved anchors、各覆盖至少2种arm与2个不同t；Mgeom相对M0在两个cells的E均≤0.8；按固定ID将dev和audit各分成前4/后4，使用全库已确定的同一anchor集合；两个半库中Mgeom均不劣于各自M0（E_Mgeom<=1），不重新选anchor；记 LOCAL_GEOMETRY_SUPPORTED_IN_PILOT。
- signal足够但以上改进规则失败：LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT。
- signal或预测采样不足：LOCAL_GEOMETRY_UNRESOLVED。
- 0.8是预先指定的实用改进门，不是物理常数，不调松。

y_mu与y_U分别判定。可能出现训练场可预测但传递到更新不可分辨，不能合并成ALL_PASS。

MnoJ与stop-v只做消融描述；clip inactive时等价是合理结果，不能强求“新机制贡献”。

### 10.3 明确的证据上限

输出独立verdict：
formula_implementation;
local_training_field_prediction;
local_shadow_update_prediction;
tracking_signal_resolution;
readout_accounting;
future_H16_H64_prediction = NOT_TESTED;
adaptive_training_value = NOT_TESTED;
energy_benefit = NOT_TESTED。

如果所有精确恒等式都通过而局部预测失败，headline应是“公式一致但局部闭合未获支持”，不是“统一理论已证实”。

---

## 11. 代码规模与数据产物

建议最小结构（由现场adapter映射，不编造旧函数）：
- 一个experiment模块：理论小函数、只读diagnostic、pure shadow map。
- 一个runner：audit/smoke/main入口，复用frozen train step。
- 一个analysis脚本：CSV/JSON到图和报告。
- 一个CPU测试文件。

不复制整套optimizer作为main训练器。为了CPU独立测试允许最小reference，但必须对照真实primitive；禁止reference和被测实现调用同一错误helper后互相证明。

输出至少：
- config.json：结果前冻结，case/probe IDs、精度、各上限、定义和门槛。
- audit.md、theory_tests.json、smoke_checks.json。
- budget_log.json：真实updates、所有诊断计数、耗时、失败尝试；old total、this total、project total分别列。
- cell_metrics.csv：每cell/arm/h真实V_diag与V_eval及gain。
- readout_balance.csv：每一步W_total/K_V及稀疏training-field/track拆分。
- local_geometry_predictions.csv：anchor、原始目标norm、误差、clip-crossing、采样误差、resolved标志；dev/audit严格区分。
- shadow_update_predictions.csv：相同口径的非线性optimizer-map预测。
- tracking_diagnostics.csv：mu/nu估计信号与不确定性、实际readout贡献。
- summary.json：上述独立verdict，不能只放一个headline。
- REPORT.md：事实、推导、假设、未测试分开；失败和partial结果照常出具。

图控制在最多5张：
1. 两个case三臂的指定目标Gain随h；V_diag与V_eval不要混轴或混名。
2. 累计readout方向项和曲率余项与实际终点差。
3. local gradient-field预测误差 vs frozen-field基线，含clip状态。
4. shadow-update预测误差及stop-v消融。
5. tracking/readout信号与probe不确定性。

图来自已保存表，不另手工改数字。保持现有绘图风格，不制作PPT、审计tar或巨型checkpoint合集。

---

## 12. 最终执行顺序与停止规则

1. 只读审查、核预算与case来源，确认A0F2证据级别。
2. 冻结本轮config与definitions，不冻结结论。
3. CPU T1–T9；修复时保留真实失败摘要。
4. GPU smoke 24 updates；通过后才main。
5. Main 384 updates与固定稀疏probe；按case/arm流式落盘，超上限就停。
6. 分析、图、报告。纯CPU复算已有表检查sum/telescope与verdict。
7. 只追加一条项目索引/状态说明，不改写历史结论。不自动进入C1或controller。

最终向用户反馈必须包含：
- 真正验证了哪些等价关系；
- 局部几何模型是否在有分辨率的目标上优于无反馈基线；
- 这个改善是否经过完整m/v/P耦合仍保留；
- 反转的事后读出记账说了什么、没有说什么；
- 哪些信号不足，哪些假设受到反例约束；
- 明确写“本轮未检验提前预测H16/H64收益，也未检验训练算法净收益”；
- 实际命令、输出路径、所有更新/诊断费用与剩余预算。

不必为每一步向用户确认；在本授权范围内执行完。遇到真正的数据、预算或数值阻塞则保存产物并给具体原因，不猜测补齐、不转去做另一个研究问题。
