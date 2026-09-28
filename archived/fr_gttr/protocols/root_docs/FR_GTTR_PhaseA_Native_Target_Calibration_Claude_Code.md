# FR GTTR Phase A — Native目标校准、token历史尺度与分层更新诊断

版本：v1.0；设计日期：2026-09-22。
执行环境：现有FR仓库、Claude Code + Kimi K2.7；不更换模型/网关/依赖。

## 0. 任务与授权边界

本文件是一项新的、受限的实验计划。用户将本文件和启动指令交给执行端后，只执行本文允许的Phase A；不自动扩展到Phase B/C。

目标：建立明确continuation schedule下、独立于既有研究读出的LM-NLL基线曲线，校准后续tokens-to-target的可行目标，同时记录不增加梯度调用的少量GTTR诊断。

本轮不是：新optimizer/controller、GTTR策略收益比较、全轨迹预测、真实mu/nu辨识、loss spike因果机制验证、完整预训练、下游能力、能耗收益。

本轮只用native AdamW一条算法臂；6条轨迹 = seeds {2030,2031,2032} × 原bank起点标签 {46,82}。恢复真实累计step，不用标签覆盖optimizer计数。每条最长256次实际optimizer更新。64/128/256是同一更新轨迹的观测位置，绝不按终点重算schedule。

禁止重新展开全项目历史大审计。仅核查运行所需的真实接口、数据划分、schedule、状态和预算。原历史账、C1账、C1L账均不改，不把旧超限修成新授权。

## 1. 先读现场最少必要材料

1. 最终CLOSEOUT_CORRECTIONS.md、TARGET_ACTION_BRIDGE_SPEC.md（如果名称不同，定位实际文件，不编造路径）。
2. FR_PROJECT_STATUS与EXPERIMENT_INDEX中最新结论、已校正的token口径和相关bank。
3. 原native runner、AdamW单步、clipping、make_stream、loss reduction与eval实现；原scheduler及训练计划。
4. C0/C1/C1L当前协议/errata，只取GTTR语义和统计定义，不再重算全部旧实验。

写一页以内的preflight.md：真实入口、六个checkpoint、实际n0、optimizer设置、数据/目标、schedule和预算。找不到A0F2不阻塞本轮，标SOURCE_NOT_FOUND即可。

实现复用冻结primitives，不重构框架，不安装新依赖，不自动commit/push，不建服务/数据库/大型审计tar，不启动常驻tail/grep监视器。日志和表必须边运行边保存；一次正式完成报告即可。

## 2. 理论语义（统一模型，不新增物理状态）

### 2.1 训练场与随机输入

原始训练梯度g(theta;omega)，真实裁剪后c(theta;omega)=C_tau(g)。固定omega与实际位移s时，已验证的候选近似为g(theta+s;omega)≈g(theta;omega)+H(theta;omega)s。

实际跨步变化可精确分解为：

c(theta_{t+1};omega_{t+1})-c(theta_t;omega_t)
= [c(theta_{t+1};omega_t)-c(theta_t;omega_t)]
+ [c(theta_{t+1};omega_{t+1})-c(theta_{t+1};omega_t)].

这是选定中间点后的代数拆分，不是两项独立的因果分解。theta_{t+1}已经依赖omega_t，第二项不自动是零条件均值噪声。本轮不增加旧batch重算，因此上述两项不分别测量，标NOT_MEASURED。

### 2.2 Tracking索引

统一使用更新前raw states m_t,v_t，当前c_t，更新后m_{t+1},v_{t+1}：
m_{t+1}=beta1*m_t+(1-beta1)*c_t；
v_{t+1}=beta2*v_t+(1-beta2)*c_t^2。

mu_t=E[c_t|F_t]、nu_t=E[c_t^2|F_t]，F_t不包括尚未揭示的当前batch。
e_pre_m=m_t-mu_t；e_pre_v=v_t-nu_t。
e_pre_m(t+1)=beta1*e_pre_m(t)-(mu_{t+1}-mu_t)+(1-beta1)*(c_t-mu_t)。
二阶同理。漂移与随机项可能相关，不删交叉项。真实bias correction、累计step及P仍保留原算法。

当前innovation I_m=c_t-m_t、I_v=c_t^2-v_t可记录。
若pre state可测，E[I_v|F_t]=nu_t-v_t成立；不能把单样本||I_v||、归一化值或其平方称为真实tracking error。
E||I_v||^2=||nu_t-v_t||^2+tr Cov(c_t^2|F_t)。
本轮0额外probe梯度，因此不估计mu/nu，不声称已分离真实漂移与噪声。

### 2.3 Transport与几何诊断

实际参数位移s=W_after-W_before；d=-eta*P*mhat是不含衰减的算法位移。分别保存actual s与按算法重建的d，避免把数值舍入差异隐藏。

对非零参数矩阵W：
r_total=||s||F/||W||F；r_adaptive=||d||F/||W||F；
b_rad=<W,s>/||W||F^2；s_parallel=b_rad*W；s_perp=s-s_parallel；
r_tangent=||s_perp||F/||W||F。
恒等检查r_total^2=b_rad^2+r_tangent^2，保存绝对/相对残差。
对普通逐参数标量decay，衰减项与W平行，完整位移的perp与d的perp相同（以实际浮点误差为限）。

归一化方向变化alpha_exact=||W_after/||W_after||F-W_before/||W_before||F||F。
这些是参数坐标诊断，不是模型功能/能力，更不能由它们代替NLL。
零范数/过小分母标NA和valid=false，保存原始分子分母，不静默加epsilon制造数值结论。

### 2.4 EMA历史尺度

beta固定且0<beta<1：h_half=log(0.5)/log(beta)。
每步训练token数固定Btok时，token_half=Btok*h_half；raw历史系数=beta^H。
从原累计n0次更新的bias-corrected旧moment看，其系数为：
w_old_corr=beta^H*(1-beta^n0)/(1-beta^(n0+H))。
这些只是滤波权重，不是response恢复时间、物理记忆寿命、稳定性或泛化证据。
变量token batch时使用累计实际token坐标；beta变化需用乘积，不能套常数公式。本轮不改变beta/batch。

## 3. 数据与目标：不更名、不混轴、不访问final表现

已报告候选D0共有1,270,238行；现场确认实际数据身份和长度。
- train：保持原冻结train ID列表及采样recipe（不能把前4096内旧eval行也加入训练）。
- 旧V_score/V_eval/V_diag：开发历史，保持名称，不用作本轮新monitor。
- V_monitor：D0 rows[4096:8192]，4096行。
- V_final：D0 rows[8192:12288]，4096行；本轮仅登记ID/哈希/重叠检查，模型evaluation调用数必须为0。

前置检查仅限已使用集合和新选集合：行ID相交、可用doc/source ID相交、完全重复token序列。共享文档或完全重复有则停止数据门，报告具体ID，不偷偷换行。无doc ID则标DOCUMENT_INDEPENDENCE_UNVERIFIED，不以行不交声称文档级独立；无已知冲突时可继续受限校准。无需扫描全部1.27M行寻找所有近重复。

未来训练采样器明确排除monitor/final ID；不扩大train support。本轮未访问final NLL；读取内容哈希用于数据身份不等于模型评价，但不得用内容质量挑选final。

一次固定以下定义：tokenizer、context、next-token shift、valid mask、ignore_index、padding、train/eval模式、forward dtype、CE和sum/count精度。
V_monitor = 总valid-target NLL_sum / 总valid_target_count。批间不能直接无权平均mean-loss，除非独立证明分母完全一致。

训练成本分开记录input/nonpad tokens_seen和valid supervised tokens；1024/step是待现场确认的历史口径，不因shift默认相等。论文tokens-to-target的主token定义须在config中写明，两个计数全程都保留。

本轮使用全部4096 monitor行，不暗换64/256行子集。eval_batch_rows=32（纯评价设置），128个forward batches/完整monitor evaluation。若无法在现设备承载，正式main前停止给出资源方案，不自动改变batch、目标或采样数。

## 4. continuation schedule：先把native定义完整

只允许一个预先确定的公共原生schedule，不新增终端cooldown分支。

按如下顺序现场确定：
A. 原scheduler本就有明确的、合法的horizon-independent延长或已有明文terminal tail：按原规则使用。
B. 原scheduler只在有限计划范围内定义，但存在明确的正值terminal LR floor及固定尾部约定：只按这个已存在约定延续；写清来源，标NATIVE_WITH_DECLARED_TAIL。
C. 若只有有限cosine定义、终点为0、公式越界会回升/周期循环、存在多个冲突约定或无明文延长语义：SCHEDULE_BLOCKED。仅交付config候选、schedule表和一个需要用户决定的问题，不运行GPU main；不能自行把计划128重写成256/512，不能任意hold peak/重warmup。

本文件不授权发明一个新tail来“救实验”。若用户/现场最新已批准文档已经确定tail，以该明确授权为准，附其路径。

冻结eta_global(n)和6条实际eta序列，至少显示local h=0,1,32,64,128,192,255及原schedule边界前后值。
CPU检查：128步请求与256步请求的前128个batch IDs、LR、beta、decay完全一致，除非仅输出长度不同。

如果原schedule尾部很小而NLL停滞，保留这一校准结果；不调高LR或选择其他schedule后继续同一注册。
无终端cooldown；结论只属这条继续训练轨迹的自然读出，不等于最佳可交付cooldown checkpoint。

## 5. 六条主轨迹与观测

模型：原约4.98M模型，实际架构/参数数以bank为准，不扩26M/1B。
起点：原seeds2030/2031/2032 × 标签46/82；恢复真实累计n0。
每个起点一个新continuation stream，拟定IDs97000..97005，按(seed,start)字典序分配。若任一ID在既有注册使用过，所有六个ID整体+1000，重复直到无冲突；结果前冻结，不按结果更换。
train数据分布、accum、batch、dropout、精度、clip、decay、beta不变。

每条主轨迹真实执行到H=256；无按结果停止的128/256二选一。
monitor读数h={0,32,64,96,128,160,192,224,256}，共9次/轨迹、54次整库评价。
h代表完成的optimizer更新次数；pre/post/global计数均另列。
达到开发候选阈值也不提前停止，本轮固定预算校准。

训练NaN/Inf时停止该轨迹，保留失败、已经消耗token和最后状态；不替换seed，不调参重跑同一科学case。其余已定cases在预算内继续，标PARTIAL/FAILED_CASE，不把失败删掉。
保存完整可恢复state在h=128/256（未到达则不伪造）；h=0引用源checkpoint。不必每一步存高维张量。文件中包含模型、optimizer、scheduler、真实step、训练RNG与数据位置。

## 6. 零额外梯度的轻量诊断

每optimizer步：lr、beta、wd、train NLL sum/count、preclip_norm、clip coefficient/flag、actual update norm、实际token计数、finite标志。复用已有日志，不再为同一标量重做forward/backward。

在h=1和h=16,32,...,256：
- 记录全部唯一存储的attention/MLP二维矩阵的第2.3节统计；按真实parameter name/shape分组，tied参数去重。
- embeddings单列，不与hidden matrices混合；一维norm/bias参数仅给基础norm，不用矩阵尺度假设。
- raw pre m/v、current c、post mhat/vhat/P的norm；I_m、I_v范数（明确INNOVATION_PROXY）。
- TRAIN_BATCH_ALIGNMENT=-<c,s>/(||c||||s||)，不称validation alignment。
- P相对变化的分子||P_post-P_pre||和分母||P_pre||，偏差校正与首次零state处理要明文。
- 总s、adaptive d、decay分量分别记录，不能把global LR造成的径向变化都归因于direction。

不能计算/报告未测Hessian、真实mu/nu、真实noise covariance、tracking error、q_V、W/K、future response、spike predictor。
这些诊断不会更改任何更新；smoke验证diagnostics on/off训练状态一致。

## 7. CPU测试与一个最小smoke

正式训练前做8项针对性CPU测试，已有可复用测试不重复造框架：
T1 梯度更换batch的拆分恒等；用一维c(theta,omega)=theta-omega、omega=±1展示theta_next依赖omega_t时第二项条件均值可非零。
T2 raw moments tracking递推索引恒等；测试I_v的条件期望为nu-v，norm平方还包含variance。
T3 EMA raw系数和bias-corrected旧state系数；beta=.9/.999，n0=46/82,H64/128/256。
T4 radial/tangential恒等与零范数NA；含decay和tied storage去重。
T5 loss sum/count加权聚合；构造不等valid token数证明mean(batch mean)会错。
T6 公共schedule前缀与data prefix恒等，不把horizon输入用于重新拉伸LR。
T7 达标记录保留删失/失败，两个连续check与单次check区别，不能以budget cap填达标。
T8 row/doc排除、final evaluation禁止、counter真实累加、简易恢复重放。

Smoke：最先一个cell，8步diagnostics off + 8步diagnostics on，从同完整状态和同stream恢复，共16实际updates。
On臂在h0/4/8调用monitor，Off臂不评价；只比较最终theta/m/v/step和训练RNG/游标，确认evaluation恢复mode/RNG且不干扰训练。
主实验不能复用smoke的已进展状态，必须从原checkpoint重新开始。
若原runner允许bitwise一致则使用bitwise；否则继承已有有来源的数值规则，不能现场按结果松门。只标量一致不得声称全状态一致。
Smoke失败先定位，所有更新和eval计费；超过本轮余量即停止，不反复重建smoke。

## 8. 预算（独立新轮次；计划不是耗时保证）

主updates：6*256=1536；smoke=16；计划1552。
硬上限actual optimizer updates=1664（剩112用于失败/恢复；不能换成第7个case）。
新增诊断gradient calls=0；新增HVP=0；final evaluation calls=0。
训练backward照常发生，必须记录，不用“0 grads”指代总训练成本。

按eval batch32：
- 主monitor full evals=54；forward batches=54*128=6912。
- smoke monitor full evals=3；forward batches=384。
- 计划monitor forward calls=7296；本轮monitor/数值复核合计forward cap=7800。
- 训练micro forwards在accum=2下主3072+smoke32；实际计数不按公式代替日志。
- round TOTAL model-forward calls cap=12000（训练、monitor、失败/复核全部包含）。
- 主monitor token slots=54*4096*128=28,311,552；主训练在1024/step成立时=1,572,864 input tokens。
评价开销不可隐藏：本轮旨在建立可靠轴，不声称其evaluation cadence是最终低成本策略。

GPU任务墙钟硬上限90分钟（含本轮训练/eval/诊断/恢复，不含离线阅读与CPU文档工作）。只用一块明确空闲且有授权GPU，不停其他用户任务。
smoke后依据真实单位耗时给出完整run ETA及计算式；若保守预计超过cap，保存preflight+smoke结果并停止提出方案，不能暗减monitor样本/改频率/换seed。

统一round counter：每次真实执行才计入；成功、失败、retry、salvage都计；不用每进程各自cap规避总cap。历史18,730授权段、C1、C1L及既有账不改。

## 9. NLL曲线、不确定性与目标校准

每次monitor保存每行或每doc的NLL_sum/valid_count及ID，足以复算同一目标的paired change。保存per-eval全库sum/count，不只保存mean。
优先按可用doc ID构造固定评估block；无doc ID则使用64行连续block，标BLOCK_PROXY，不声称已得到真正独立doc样本。
对同一cell跨时点的NLL差，用相同block索引配对bootstrap（CPU,1000次,固定seed）；它只衡量该参考库的条件采样差，不是独立训练seed不确定性。
训练seed只有3个，两个起点相关；分别画6条曲线、每个start组3条seed结果，不把54个时间点当54个独立样本，不输出总体显著性星号。

主曲线指标：
- Vmon(h), h∈既定9点；actual token横轴。
- ΔV(0→64), ΔV(0→128), ΔV(0→256), ΔV(128→256)。
- 分段NLL变化/更新token，仅描述，不推为动作价值。
- seed内两起点分别报告，不能直接pool不同初始loss。
- fixed256末端NLL、失败/未完成数、训练/评价wall。

开发阈值生成规则在任何GPU前冻结：
每个起点组g∈{46,82}，若3个seed都有h128读数，
ell_candidate[g]=median_seed Vmon(seed,g,h128)。
这只是开发候选目标，不是结果前已经知道的数字；不能称本轮阈值外推独立验证。

给出内部可达性说明：
- 对每个seed报告V0-ell、ell-V256及对应V0/V256 block SE。
- 只有所有3个seed的V0>ell且至少2/3在V256<ell，才标INTERNAL_TARGET_BRACKETED。
- 更严格的“噪声余量”另列V0-ell>2SE0及ell-V256>2SE256，不把这当三seed总体置信结论。
- 条件不足则TARGET_UNRESOLVED/FIXED_BUDGET_ONLY；不换成另一分位数或另一时点来通过。
- 每个起点独立目标是一项独立continuation任务；不把不同ell的成本直接pool成同一target。

可离线展示候选ell下的监测规则：两个连续既定monitor检查点≤ell，在第二点记detected_hit；未发生记CENSORED/NOT_CONFIRMED。last-above/first-below/confirmation分别保存，不把检查间隔线性插值成真正首次穿越。
本轮所有hit/tokens均CALIBRATION_ONLY；不实际早停、不查询final确认、不宣称已测独立tokens-to-target。
以后PhaseC应固定相同monitor节奏，并在预定候选checkpoint一次性final确认；失败即确认失败，不能反复查final决定继续训练。PhaseA不执行PhaseC。

## 10. 理论诊断分析边界

1. 两个beta各给h_half、token_half、beta^64/128/256、bias-corrected旧state系数。不得因beta2^256较大宣称optimizer全部未适应，也不得把它当实际响应长记忆。
2. 给实际矩阵层统计表：同一矩阵不同h、同h不同矩阵；显示r_total/r_adaptive/b_rad/r_tangent/alpha_exact和innovation原始分子分母。
3. 仅描述这些量与同时/后续NLL区间的关联；可给逐seed可视化，不拟合复杂regression，不把大量时间点伪作样本。
4. 不能由innovation下降宣称真实tracking error下降；不能由角度或相对步长升高宣称功能学习更快。
5. 无干预对照，本轮无法识别“某量改变引起收益”。不提训练spike阈值、β2修改、batch缩放或新controller。
6. PhaseB的候选动作只可在后续根据这一校准提一个设计；它需要同时比较目标差值误差、决策间隔、实际成本和合理固定基线，不能用当前相关性直接授权。

## 11. 文件与交付（少而足够）

建议新建（既有模块可小改复用）：
- research/scripts/phase_a_native_calibration_v01.py：preflight/smoke/main入口；
- research/scripts/phase_a_native_analyze_v01.py：纯离线分析；
- research/tests/test_phase_a_native_calibration_v01.py：针对性参考测试；
- outputs/fr_gttr_phase_a_native_calibration_v01/。
不要求不存在的函数名；现场映射只写入preflight，复用原runner。

输出核心文件：
config.json；preflight.md；dataset_contract.json；schedule.csv；
theory_tests.json；smoke_checks.json；resource_log.jsonl；
trajectory_metrics.csv；monitor_row_stats.csv（或等价parquet，无新依赖）；
layer_diagnostics.csv；ema_timescales.csv；target_calibration.csv；
status.json；REPORT.md；KEY_DATA_FOR_REVIEW.md；
完整h128/h256 snapshots与明确manifest。
最多5张图：NLL-token；分段进展；EMA保留权重；分层相对/切向更新；训练/评价成本。不要画GTTR比native提升图，因为本轮只有native。

不要生成一个笼统GTTR_PASS。独立状态：
IMPLEMENTATION；DATA_SPLIT；SCHEDULE；COVERAGE；
NATIVE_PROGRESS_MEASURABLE；TARGET_CALIBRATION；DIAGNOSTIC_AVAILABILITY。
TRUE_TRACKING_ERROR / DRIFT_NOISE_SEPARATION / ACTION_VALUE_PREDICTION /
ADAPTIVE_VALUE / CONFIRMED_TOKENS_TO_TARGET / GENERALIZATION_OR_ENERGY_GAIN
均为NOT_TESTED。技术失败和目标不可分辨不是GTTR理论被证伪。

最终聊天直接展示六张关键表（不能只发“已保存”）：
A. 6个cell的seed、checkpoint标签、真实n0、stream、beta、batch/accum、每步实际tokens、LR首尾、schedule语义、是否完成256。
B. 每cell h0/64/128/256的NLL sum/count/mean、paired变化及SE、累计token、wall；其余时点附件保留。
C. 各beta的step/token半衰期、raw/bias-corrected旧权重；不能只给一项“记忆短”。
D. 矩阵诊断全表保存在文件；聊天列真实name及top3相对步长，并注明这些是描述性排序，不用于选动作；创新量标签与分子分母。
E. 每组ell_candidate、规则、逐seed初末目标余量、噪声余量、回看hit/censor状态；final_call_count必须0。
F. 全部资源planned/actual/cap、失败尝试、train/eval分账、实际coverage与NOT_TESTED列表。

报告用中文，明确区分用户实验事实、数学恒等式、诊断观察和未来假设。不用论文结论替代本轮数据。任务完成后停止；不自动进入Phase B/C。

## 12. 后续Phase B/C接口（本轮不执行）

PhaseB：只研究一个动作与一个机制假设。信息时点、收益符号、对照、模型误差与决策margin在读结果前冻结。可用same-stage/relative-step控制、留出seeds或streams，但不能把PhaseA相关性当因果。不得重新包装A0U sign schedule为新GTTR算法。

PhaseC：在独立确认轨迹与固定target下比较native、充分调优固定baseline、GTTR规则，所有调参/probe/eval/失败成本分账。共同cooldown或终端分支如被引入，必须将其token/时间计费，且不再宣称所有终点属于同一嵌套实际轨迹。当前不必建立复杂价值网络或要求所有未来参数都可预测。

## 13. 文献依据（用于明确边界，不要求现场重复大检索）

- Wen et al., Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective, arXiv:2410.05192。借鉴继续训练与终端读出的区别，不预设本项目河谷/快慢分离成立。
- Pagliardini et al., The AdEMAMix Optimizer: Better, Faster, Older, ICLR 2025。提示旧梯度可能有用，不把最小化tracking norm当目标。
- Bai et al., Adaptive Preconditioners Trigger Loss Spikes in Adam, arXiv:2506.04805v2（2026-05-25）。预条件反馈应保留；有简化假设和事后信息的指标不能直接作在线controller。
- Marek et al., Small Batch Size Training for Language Models, arXiv:2507.07101v4。token半衰期有比较价值，不将所有beta强制同尺度缩放。
- Wen et al., Fantastic Pretraining Optimizers and Where to Find Them, arXiv:2509.02046。调参/衰减阶段影响排序，不用中间loss推出最终排名。
- Wen et al., Fantastic Pretraining Optimizers... II: Hyperball Optimization, arXiv:2606.16899v1。借用参数相对步长与方向诊断，不假设每个矩阵严格尺度不变，不改optimizer。
- Ma and Chen, WSqD, arXiv:2607.10959v1。基础schedule与terminal cooldown需分开定义，不直接更换成WSqD。
- Wu et al., Understanding Short-Horizon Bias in Stochastic Meta-Optimization, arXiv:1803.02021。局部最优不等于最终成本最优。
