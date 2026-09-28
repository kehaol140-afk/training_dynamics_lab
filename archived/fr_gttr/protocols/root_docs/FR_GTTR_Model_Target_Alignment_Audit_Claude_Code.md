# FR GTTR 模型覆盖—目标一致性长审计

版本：v1.0，2026-09-22。
用途：交给现有 FR 仓库的 Claude Code＋Kimi K2.7 执行。
性质：审计与验证协议，不是审计结果，不是已成立的新定理。
编写依据：本对话的 A0G–A0U retrospective、B/C 系列完成摘要、C1-Late 字段勘误，以及挂载的原执行协议。编写者未读取远程仓库的最新原始结果。

## 0. 用户要求与本轮授权

用户关心的不是只复算局部误差，而是：GTTR 是否覆盖真实 AdamW 训练语义及相关损失地形；能否支撑实际性能目标上的成本优化；会不会出现短期有效、长期无效，或者代理量变好而最终任务不改善。

本轮允许：
- 读取现有 FR 文档、配置、代码、日志、原始充分统计及已保存的学习曲线；
- 编写小型 CPU 参考计算，独立复算关键结果；
- 运行低维数学反例、反向测试、单位/计数/信息边界测试；
- 生成模型—目标覆盖审计、明确证据缺口，以及一个最小必要的后续验证方案。

本轮不允许：
- 新的实际模型 forward、backward、HVP、optimizer update、GPU 评估或长训练；
- 为审计重新访问从未使用的最终测试集、调用外部付费模型或下载大数据集；
- 启动 C2/controller/新 optimizer/架构实验；
- 自动扩展任何旧预算。旧账 17,994/18,000、C1 和 C1-Late 的账分别保留，以现场记录为准。

这是长审计，不是长训练。没有长期轨迹，就明确写 NOT_TESTED，并提供能补上该缺口的最小验证设计；不能靠推导或措辞把缺失数据变成已验证。

执行到审计报告完成，不只返回计划。局部资料缺失不得阻塞其他独立审计事项；不反复寻找有利解释。默认至多 90 分钟工作窗口，CPU 小实验每组设置实际超时，不占训练 GPU。超过窗口先保存已完成结果并列明缺口，不为完整性启动新计算。

保持现有环境和 Kimi 接入。不要升级依赖、重构框架、建立服务/数据库、打大型审计 tar、递归 hash 全仓库、自动 commit/push。不得影响其他训练任务。必要的 config/结果身份记录足够。

## 1. 必须最终回答的七个问题

Q1. GTTR 精确骨架到底覆盖了哪些实际训练语义？哪些只是数学重写？
Q2. 当前有效近似覆盖哪些状态、方向、动作、扰动幅度、精度和局部时域？
Q3. 训练场、验证 NLL、下游能力、几何代理量之间哪些关系被测过？哪些只是相关性或假设？
Q4. 已知真实位移的局部预测，是否被误写成自主多步预测或最终收益预测？
Q5. tokens-to-target / time-to-target / joules-to-target 的单位、起点、目标、停止规则和总开销是否明确？
Q6. 不预测完整轨迹时，可以依赖什么目标相关的不等式或校准进行优化？它的未验证假设是什么？
Q7. 现在能支持什么范围的论文/算法结论？要升级结论，最少还缺哪一项证据？

不得用“整个 GTTR PASS/FAIL”替代七个问题。每个结论附数据范围和来源。

## 2. 阅读清单、证据等级与事实来源

从现场定位并读取真实路径：
- FR_PROJECT_STATUS、FR_EXPERIMENT_INDEX、GTTR_EVIDENCE_UPDATE；
- A0C/A0E2/A0F2（存在则读原件）、A0O/A0P/A0Q/A0S/A0T/A0T-extra/A0U；
- B0、B0.5、B0.6 的冻结配置、报告、原始表和判据代码；
- C0、C0X、C1、C1-Late 的协议、结果前 amendments、errata；
- FIELD_CORRECTIONS.md、fieldcheck_task1..5、mean/centered/sum sufficient statistics；
- 实际 frozen runner、AdamW/clip/HVP/replay/评估实现；
- 模型、数据采样、tokenizer、有效 batch、padding/label mask、precision、scheduler 和目标轴定义；
- 已有长期训练记录、learning curves、cost logs、train/dev/monitor/test 分工，存在多少就审多少。

SOURCE_RAW：原始数据或代码；SOURCE_DERIVED：能从原件复算的派生数据；SOURCE_REPORT：只有执行报告；SOURCE_CHAT：只有历史转述；SOURCE_MISSING：没有找到。
数学分类另列 EXACT_IDENTITY / CONDITIONAL_THEOREM / EMPIRICAL_LOCAL / HYPOTHESIS / CONTRADICTED_OR_MISIMPLEMENTED。

文件写明含义与历史助手解释不同时，并列原表述、数据支持的边界、建议修订。不要默默重写历史文件。不存在的 A0F2 或最终测试结果不能编造。

## 3. 统一数学对象：先做符号/实现对应，不新增模型

### 3.1 完整状态和真实目标

固定架构、任务和数据分布时：
L_train(theta)=E_omega ell(f_theta;omega)。
这是固定参数函数；H(theta_t) 可沿轨迹变化。minibatch 换样本不等于总体目标自身漂移。

Z_t 至少包括 theta、moments、实际 optimizer step 及必要调度/执行状态。
数据游标、随机状态、梯度累积缓冲、AMP scaler、训练 buffers 等：按真实代码确认是否影响未来状态；不要因为文档只写(theta,m,v)就认定它已包含所有恢复条件。
也可将已知时序/随机输入写为 F_t 的输入，但须明确可用信息。表示 phi(theta) 不无依据变成独立物理自由度。

Z_{t+1}=F_t(Z_t,a_t,omega_t)。
若每步 F 都调用真实网络求梯度，它可以准确表示训练，但本身不构成节省计算的闭合预测理论。

### 3.2 精确 AdamW 骨架（单参数组写法）

g_t=grad ell(theta_t;omega_t)
c_t=Clip_tau(g_t)
mbar_t=a_t*mbar_prev+(1-a_t)*c_t
vbar_t=b_t*vbar_prev+(1-b_t)*c_t**2
P_t=diag(1/(sqrt(vbar_t)+eps))
r_t=P_t*mbar_t
d_t=-eta_t*r_t
s_t=d_t-eta_t*lambda_wd*theta_t
theta_next=theta_t+s_t

本节 a_t/b_t 是固定 betas 下偏差校正的系数，不是政策动作名。
审计实际累计步号、初始化、参数组、decay mask、skip update、grad None、AMSGrad、fused/foreach、epsilon 放置、unscale/accumulate/clip顺序、loss reduction。

T_t=a_t*(eta_t/eta_prev)*P_t*P_prev^{-1} 只是 realized carry-over coefficient，不等于 A_t=dF_t/dZ_t。eta_prev=0 等情形不能直接相除。
当前 log-LR 的直接输入是[s_t,0,0]（固定当前状态/梯度/其他设置），不是再额外乘一次 eta。
反事实里 P 的改变必须通过当前 c²→v→P 求导。未来moments会经theta→g反馈改变。

### 3.3 Geometry / Tracking / Readout 的边界

固定当前batch随机性：G_t=D_theta c_t=J_clip(g_t)*H_batch,t。
真实clip的epsilon、active set和不可微边界必须对应runner。
G_t通常不是对称矩阵，不能直接当新标量loss的Hessian。
mu_t=E[c_t|F_t]、nu_t=E[c_t²|F_t]是同一冻结状态的条件量。
E[Clip(g)]通常不等于Clip(E[g])；E[AdamW(g)]通常不等于AdamW(E[g])。

q_t=grad V_diag(theta_t)只做明确目标的readout，不替代训练场mu_t。
DeltaV=-W+K，W=-q^T*s，K=DeltaV+W是精确已实现记账。
K恒等式成立不是Hessian近似/机制成立；K可负。

### 3.4 当前已经检验的近似

在已知真实s和起点theta上：g1_hat=g0+H0*s。
field p=Clip(g1_hat)-Clip(g0)，y=Clip(g1)-Clip(g0)。
shadow p=U(g1_hat;Splus)-U(g0;Splus)，y=U(g1;Splus)-U(g0;Splus)。

这是一种给定当前真实位移的局部条件近似。使用真实Splus、s、qplus的顺序应列入信息清单。
不能把它当成在动作执行前已知的全状态预测，也不能把native方向准确外推到任意beta/batch/clip/LR动作。

## 4. 审计 A：更新覆盖、地形覆盖与能力覆盖必须分别报告

生成 coverage_matrix.csv，至少包含：
claim_id、数学对象、实际代码位置、被测试模型/阶段/动作/目标、所需假设、现有反例、证据文件、结论范围、未覆盖项。

独立审计下列层次：
1. 更新语义：是否等于所用AdamW，而不只是近似SGD。
2. 局部场近似：覆盖方向/步长/clip regime/精度；已知位移和自主预测分开。
3. 全局几何：是否真的观测到跨谷、非局部障碍、慢流形；没有就NOT_TESTED。
4. 模型能力：是否有实际语言模型NLL/下游测试，而不只是梯度和Hessian。
5. 反事实动作：是否验证整个允许动作集合，还是仅native/已实现方向。
6. 时域：H_local、干预长度、恢复长度、观测终点、累计token分别记录。
7. 尺度/任务：5M局部结论不得搬到未测试模型或其他任务。

不自动进行全Hessian扫描，不把更多几何指标当作提高可信度的默认手段。

## 5. 审计 B：把目标、能力和成本写在同一个问题中

### 5.1 目标数据字典

逐一列 L_train、probe目标、V_diag、V_score、V_eval、V_monitor、V_final及每个下游A_j：
- 数据/样本ID、数量、tokenizer与context；
- loss的sum/count/reduction、label mask/padding、nats/bits；
- eval/train模式、dropout、精度；
- 用途：训练/构造模型/调参选动作/停止监测/最终确认；
- 何时首次访问、是否反复用于选择、是否与其他集合重叠；
- 名称相同但数据不同、名称不同但实际相同的情况。

反复用于动作选择的数据就是开发信息，不能继续称完全独立最终测试。
只审既有访问信息，不在本轮读取新的最终测试内容。

### 5.2 主要候选目标（本轮提出定义，不编造数值阈值）

把用户的“target to token”规范为 tokens-to-target：

G_target={theta: V_ref(theta)<=ell_star，并满足已声明的能力/泛化约束}。
 tau_star=inf{t: theta_t进入G_target}。
 N_tau=sum_{t<tau_star} n_update,t。
目标：min_policy E[N_tau]。

V_ref可在理论上表示固定参考分布NLL；实际操作必须指定monitor协议和独立final确认。
若以多个能力指标为约束，需要指定其方向、容差和评测协议；无任务证据不得把NLL降低宣传为通用能力提高。

明确不同目标：
- fixed-model tokens-to-heldout-NLL-target；
- tokens-to-multiple-quality-targets；
- continuation tokens from an existing checkpoint；
- pretraining-from-initialization tokens；
- time-to-target；
- joules-to-target。
它们不能相互替代。

无事先阈值的旧数据不能事后挑一个最有利ell_star做确认性结论。可描述threshold-sensitivity，但标POSTHOC_EXPLORATORY；正式阈值需来自结果前约定或独立开发基线。

### 5.3 token与资源计量

每step记录真实loss-bearing/update tokens、输入processed tokens、global batch、microbatch、accumulation、world size、context length及mask；不要默认所有token定义相同。

单列：
N_update、N_probe、N_candidate_screened、N_eval、N_replay、unique/data-exposure（可得时）；
forward、backward、HVP、shadow arithmetic、wall-clock、joules。
被看过但未用于最终更新的candidate batches不能从数据暴露和成本账中消失。

不把HVP token简单当成一次训练token；不把tokens直接等价焦耳。
报告每次部署开销与一次性模型辨识/调参开销。amortization声明必须给重复部署次数和明细。

### 5.4 到达目标与未到达

只能使用已记录的检查点或评估点。
报告首次观测达标的step/token、最近前一个未达标评估、确认评估时间、最大观察预算、是否退出目标区。
两个评估点之间存在实际到达时刻不确定性，禁止不披露地线性插值成精确命中时间。
采用连续两次达标只能表示预定监测上的持续确认，不是永久达标。

未达标、NaN、OOM、崩溃单列，不能删掉后只平均成功者。
右删失报告hit_count/total、censor budget、固定预算末端质量；不能把删失上限当真实达标时间。
从checkpoint开始只叫continuation效率，不声称全预训练token节约。

## 6. 审计 C：局部指标到长期目标是否存在合法桥梁

### 6.1 历史证据复核

只用已保存的同轴数据，逐cell/arm列：
actual_intervention_duration、readout_horizon、tokens、gain_vs_native、gain_vs_fixed、uncertainty、action_known_time。

严格分开：
- 一次固定前缀干预后的不同读出时域；
- 持续干预的长度同时随H变化；
- qplus投影；
- 实际一步DeltaV；
- H4/H16/H64真实gain；
- 完整训练质量。

A0T历史反转、A0U fixed ablation的负结果必须保留；C0目标轴不匹配不能靠常数偏移修复。
已有64步数据不足以检验1024步或全程时，写NOT_TESTED，不从t48 anchor推断H48。

### 6.2 不预测全轨迹时的三种目标桥梁

不要强制全部实现，只审计哪一种现在具有条件。

B1：目标相关局部模型/可信域。
写明允许动作集合、目标估计、误差上界或校准区间与动作选择的信息。
若对同一真正目标J(a)有sup_a|Jhat-J|<=eps，并以delta求解近似最优，则
J(ahat)<=min_a J(a)+2eps+delta。
但当前field/shadow向量E不是该eps，更不是整个动作集合上的上界。

B2：目标一致的progress/Lyapunov证书。
必须说明Phi是否非负、有界下界、何时接近目标集、是否允许目标外停滞。
若主张E[DeltaPhi|I]<=-kappa E[cost|I]直到目标，需逐条审计可积性、停止/失败路径、成立状态动作区域及观测证据。
不得在已知结果后任意选系数把Phi拼成下降；不得只在成功轨迹上验证漂移。
Phi下降、gradient norm变小或稳定性本身不等于模型能力提高。

B3：以真实累计成本为对象的策略/续训验证。
可以通过直接测量固定策略与候选动作的累计效果验证，不要求高维轨迹预测。
但要有独立同目标读出、合适时域、调参匹配和固定动作基线。
仅“比native好”不代表存在自适应选择价值。

所有桥梁至少回答：假设在哪里可检验？可能违反时如何发现？优化器是否可能利用模型误差？

### 6.3 代理量不准替代真实目标

温度、熵产生、广义Langevin参数、Hessian trace、lambda_max、gradient norm、W/K分量、shadow范数都是候选解释量/诊断量。
没有证明的单调关系、界或独立任务校准，不得代替V_ref或能力指标。
非平衡模型允许描述动力学，但不自动提供目标函数。场论或热力学变量的单位不得人为换成loss或joules。

## 7. CPU-only反例与反向测试（检验论证，不证明真实LLM机制）

不要求制造正结果；每个测试记录预期数学事实、运行结果、适用边界。若失败先检查测试实现，不放宽数学判定。

T1 — 已知位移不等于开环预测。
构建依赖图，逐字段标AVAILABLE_BEFORE_ACTION、AFTER_CURRENT_GRADIENT、AFTER_CURRENT_STEP、FUTURE_OR_AUDIT。
只读检查实际调用顺序。用stub在禁止访问时抛错。不能把需要真实未来theta的程序标为online。

T2 — 单谷短长时域反转。
V(x,y)=0.5*(x²+6y²)，初值(1,1)，GD eta0=1/4。
只改变前4步eta，之后恢复。
验证dV_H/deta_pulse=-4*sum_i lambda_i²*x_i0²*(1-eta0*lambda_i)^(2H-1)。
H4导数应正，H16应负；中心差分在两个幅度上与解析趋势一致。
结论仅为反例：反转无需切换、非凸性或模型错误。

T3 — 相同局部地形不唯一决定全局结果。
构造f1(x)=0.5*(x-3)²；f2=f1+A*bump((x-1)/r)，r<0.5、bump在|u|>=1为0，内部exp(-1/(1-u²))。
验证x=0邻域的函数与导数相同，远处不同。记录平滑性与使用区域。
这说明局部测量不能无条件恢复整个地形；不声称FR真实轨迹跨过该障碍。

T4 — 相同更新范数不代表同目标收益。
V(theta)=0.5||theta-(1,0)||²，theta=(0,0)，s+=(0.1,0)、s-=(-0.1,0)。
相同位移范数，DeltaV分别-0.095、+0.105。
目标读取方向不可被范数或“温度”替代。

T5 — 曲率重参数化不等于能力变化。
f(theta)=0.5*(theta-1)²，theta=a*phi。
同一物理函数点的预测/函数值不变而phi坐标Hessian=a²。
明确：这不是声称相同eta下GD/Adam轨迹不变，只说明原始曲率指标不能无条件作为任务能力标尺。

T6 — 步数、tokens与成本排序可以相反。
构造100步×1024 tokens与80步×2048 tokens，复算步数更少但token更多。
再加入probe与candidate成本，验证它们不会被算成免费；不开真实模型。

T7 — 未达标路径不能消失。
构造一个达到目标和一个预算内未达到目标的记录；单独输出hit_rate、censoring、末端损失。
确保参考计算拒绝只对达标者平均后作总体token节约结论，也拒绝把稀疏评估的插值当精确到达。

T8 — 聚合门与单点门。
六点比值0.43、一点1.166、等TSS；聚合E约0.594而最大单点>1。
拒绝以逐点max替代固定聚合SSE/TSS；保留missing统计标记。

T9 — 统计恢复条件。
若same K残差mean(norm)=a、SE_norm按样本std(ddof=1)/sqrtK、norm(mean)=b，验证
mean_sample_norm²=a²+(K-1)SE_norm²；
SE_vec²=SE_norm²+(a²-b²)/(K-1)。
用合成向量独立验证，随后仅在真实定义和分组匹配时恢复已有残差SE。
不同对象、不同权重或bootstrap SE禁止套用。旧raw缺失与“等价充分统计可恢复”必须区分。

每项测试都应含一条故意错误实现的反向检查（如把SE_norm当SE_vec、删失败run、用逐点max），确保参考审计会抓到错误。

## 8. 参数/动作/尺度的适配边界

输出 validity_domain.csv：
architecture/model_size、tokenizer、dataset、training_stage、native_optimizer、precision、clip活动、perturbation_type、幅度、信息时点、局部/全程时域、readout、uncertainty。

区分 EXACT_FOR_IMPLEMENTATION、EMPIRICALLY_COVERED、UNTESTED_EXTENSION。

现有六个anchors是已选R/C pair的三位置，不是六个独立模型，也不等于已覆盖晚期预训练。
t=48只表示该局部分叉之后的索引，需和真实累计step/token及计划总训练量同时列出。

若下一步允许beta/batch变化：
- beta变化需审计偏差校正历史，不直接套固定beta幂式；
- batch size变化需审计step/token schedule、裁剪与梯度均值定义；
- data-mixture变化同时改变平均场和数据目标，不默认是仅改变噪声；
- 无覆盖的动作只能进入新的验证设计，不能直接被model optimizer自由搜索。

## 9. 后续最小长时验证：本轮只设计，不启动

基于前述审计，只选择一个最关键缺口。产出 NEXT_VALIDATION_SPEC.md，必须写出：

1. 要验证的是continuation tokens-to-target、固定token末端质量，还是从初始化的完整训练；禁止混称。
2. 一个明确主目标和独立最终评价，必要能力guardrail；没有现有适合小模型的能力任务就标明，仅作LM loss结论。
3. target阈值/确认方式只能从预先规则或独立baseline开发集确定；不得看候选臂结果挑阈值。
4. 至少native与当前最佳合理固定干预基线；若暂无可执行GTTR策略，就标objective feasibility study，不发明控制器充数。
5. 固定完整干预时长与真实读出时域，嵌套检查点；明确超过64步是较长continuation而非自动“最终训练”。
6. 初始checkpoint、seed/stream、批次token轴和最终训练终点。不能只沿用R/C结果挑最有利案例。
7. tokens主指标，实际时间/能耗及全部诊断费另列；预算不足先做范围有限pilot，不宣称完成全程验证。
8. 未达标/失败、稀疏评估间隔与目标集退出情况的处理。
9. outcome单位为独立run/seed，非每step或每probe；功效不足给区间和pilot标签，不制造p<0.05。
10. 新增updates、grad/HVP、eval、wall-time的最坏情况预算和现场计价条件；与旧账独立。

本轮不自行决定一个无依据的“合理512/1024步足以证明长期”。应由现有baseline曲线、训练计划、目标距离和预算决定候选长度，并在不具备依据时明确待确认。

## 10. 输出文件与关键数据（用户必须能独立判断）

建议输出目录：outputs/fr_gttr_model_target_alignment_audit_v01/。
只需下列核心文件，不做大型审计打包：
- AUDIT_REPORT.md：中文，回答七问，结论旁写实际数据和边界；
- coverage_matrix.csv；
- target_dictionary.csv；
- validity_domain.csv；
- horizon_alignment.csv；
- cost_target_audit.csv；
- theorem_assumption_checks.csv；
- cpu_counterexamples.json；
- KEY_DATA_FOR_REVIEW.md；
- NEXT_VALIDATION_SPEC.md；
- audit_summary.json（独立字段，不做单一headline）。
可将细表合并至一个子目录，命名以清晰为准。

最终聊天回复必须直接给出以下表，不只返回文件路径：
A. 精确更新 / 局部近似 / 反事实动作 / 能力 / 长期目标 / 成本 六层证据范围。
B. 每个目标轴的数据ID、reduction、用途、是否参与选动作、是否独立final。
C. 有同目标历史数据的每个arm的干预长度、观测长度、累计tokens、gain、与fixed差值及不确定性；无数据填NA。
D. 各模型诊断量到最终目标的桥梁：公式或假设、当前检验证据、失效反例。
E. 真实成本字段完整率、未达标路径是否保留、评估时间间隔是否明确。
F. CPU反例的数值输出（至少T2/T4/T6/T8/T9）及其不允许外推的范围。
G. 哪些旧结论可保留、哪些必须降级/勘误、下一项必要验证是什么。

审计summary分别使用：
SUPPORTED_WITHIN_SCOPE / NOT_SUPPORTED / NOT_TESTED / DATA_MISSING / PROTOCOL_MISMATCH / IMPLEMENTATION_MISMATCH。
数学恒等式通过与empirical support分开；没有数据时不得自动判NOT_SUPPORTED。

## 11. 外部理论参考：只做假设对照，不替代用户证据

优先原始论文与官方实现：
- Loshchilov & Hutter, Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
- Wu et al., Understanding Short-Horizon Bias in Stochastic Meta-Optimization: https://arxiv.org/abs/1803.02021
- Liu et al., Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models, ICML 2023: https://proceedings.mlr.press/v202/liu23ao.html
- Dinh et al., Sharp Minima Can Generalize For Deep Nets: https://arxiv.org/abs/1703.04933
- Chen, Menickelly & Scheinberg, Stochastic Optimization Using a Trust-Region Method and Random Models: https://arxiv.org/abs/1504.04231
- Rosenberg et al., Near-optimal Regret Bounds for Stochastic Shortest Path: https://proceedings.mlr.press/v119/rosenberg20a.html

文献说明某现象可能发生，不表示它就是FR已确认机制；有限状态、凸性、Gaussian、稳定性、IID等条件须逐条核对。论文没读到的定理不补写。
不能联网时保留本节为待查参考，不编造“已核验”的引用。

## 12. 完成条件

完成审计不要求GTTR得到正结论。
最终必须有一份用户能直接使用的定位：
“GTTR现在能在什么条件下做什么；不能做什么；若用于tokens-to-target等优化，最小还缺什么”。

审计结束停止。若没有长期/能力/成本数据，明确这些项仍NOT_TESTED。
不要把审计时长、测试数量或文档篇幅写成理论正确性的证据。
