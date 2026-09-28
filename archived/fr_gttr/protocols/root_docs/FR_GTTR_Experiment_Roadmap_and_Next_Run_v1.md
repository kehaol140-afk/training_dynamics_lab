# GTTR 实验推进路线 v1：从有效模型验证到目标成本优化

日期：2026-09-22。
依据：用户已审阅的 GTTR_Unified_Model_Derivation_v0_2.md、当前对话中已同意的语义修订、C0/C1/C1-Late 用户报告及审计收尾。
性质：下一阶段实验设计。文中的样本量、阈值生成方式和 go/no-go 规则是新设计，不是既有实验结果。
执行环境：当前 FR 仓库，Claude Code + Kimi K2.7，不更换模型/网关。

## 0. 总原则与执行范围

理论分析暂停扩张。继续使用真实网络、真实 AdamW、Geometry–Tracking–Transport–Readout 主干。
不增加温度、亚稳态、记忆核、表征自由度等新解释模型。有效性不足时，先用反例和误差定位最小缺口，再决定是否修订理论。

用户将本文件和末尾的启动指令交给执行端后，本次只执行 A0（一次性准备）与 A1（native 目标校准）。
B/C/D/E 是后续路线，不构成本轮 GPU 授权。它们的模型配置和计算额度在对应阶段运行前单独冻结。
不将一次同意变成自动执行全部路线；也不把尚未开始的阶段写成已通过。

本轮新声明的尾部：原 global_micro <=256 时保持 get_lr(global_micro,256,5e-4)；超过256后恒定5e-5。
此条作为本次实验选择生效，不声称旧实验已有这一约定。不要改历史 get_lr 或旧实验输出。

### 为什么这样推进

已有证据支持部分位置的局部场预测与 shadow 更新增量预测；没有证明任意动作可判别，也没有证明新的规则降低独立目标的达标成本。
因此分开测三件事：
1. 模型能否在动作执行前正确比较候选动作；
2. 相比充分调节的固定策略，状态依赖选择是否有价值；
3. 全部诊断/选择开销计入后，最终质量与成本是否真正改善。

不再重复整个 A0/B0/C0 审计，不把恒等式 PASS 当作这三项的替代。

## 1. 全路线一览

|阶段|科学问题|主要实验|进入下一阶段的依据|
|---|---|---|---|
|A0+A1（本次执行）|真实目标轴和 continuation 基线是否可用？|6条native轨迹，最多256步|数据/schedule明确，曲线可读；候选目标可用或明确固定预算口径|
|B（待单独冻结）|GTTR对未执行候选动作是否有目标相关判断力？|小规模配对动作试验，预测先落盘、真实分支随后执行|相对无几何/固定动作有可分辨信息；同一动作后续效应不被短期评分掩盖|
|C（待单独冻结）|一个简单GTTR规则是否实际节约成本？|native、固定最优、GTTR、盲干预对照|同目标质量不下降，候选规则相对固定基线有实际增益且成本可接受|
|D（待单独冻结）|开发期效果能否独立复现？|冻结全部选择，用新训练seed/新起点流与封存final确认|独立NLL与成本结论，而非调参数据回看|
|E（最后）|结论能否从小模型续训迁移？|26M，再100M；必要时从初始化训练|只在前面有效时扩展，不默认直接1B|

所有阶段的 negative/unresolved 都是允许结果。不要为了模型故事修改阈值、增加样本到显著或筛选成功seed。

## 2. 阶段 B/C/D 的预先研究路线（不是本轮实现要求）

### B. 动作前模型检查，而不是再测已知位移

先只调 log-LR 一个通道，起始候选 u={-0.05,0,+0.05}，实际 multiplier=exp(u)。
固定 beta、clip、batch、weight decay 语义和基准schedule，不同时引入更多动作。
这不是重跑旧sign controller：模型必须先给出候选值/差值和选择，而不是读真实未来收益后打标签。

建议 pilot：从阶段A保存的轨迹中按预定规则取6个状态，例如3 seeds的起点46轨迹h0/h128，每状态2条新future streams。
每个state-stream配对3个动作：干预前4个optimizer updates，之后恢复相同native LR；保留形成的theta/m/v。
各分支读取H={1,4,16,64}，始终是同一次4步动作的后续读出，不在比较H时延长干预。
主真实更新数为6*2*3*64=2304；准备、模型HVP、eval、smoke另估，未在本文件授权。
这是开发pilot，3个seed，不把12 streams或多个H当独立训练seed。

模型与参照：
- M0：同一优化器与读出、没有局部训练几何反馈的冻结梯度近似；
- M_GTTR：v0.2局部梯度场近似，加真实clip/m/v/P与独立的开发目标读出；
- 真实候选重放：仅用于事后标签和成本参照；不是在线可用的免费oracle；
- native和固定u=+0.05：必须保留。

M_GTTR使用的方向基、probe数、刷新方式、内部短窗在B运行前一次性锁定；只允许一个主模型和一个明确消融，不搜索大批低秩/记忆模型。
可以使用短虚拟rollout评分，但不要求先通过2/4/64步全部参数预测。核心验收是目标动作差，而非参数轨迹重建。
模型输入不得包含真实未来梯度、真实theta_H或audit/final标签。backward后决策可使用已计算的当前batch梯度；未来只能按已声明训练采样律在模型内平均/模拟，并计成本。
旧V_score可作为明确的开发/控制目标；V_monitor提供开发期独立于当前评分的审查，V_final此阶段封存。V_monitor一旦用于方案选择，就不能在论文中冒充final。

关键输出：
- 每state/stream/action/H的预测值、实际NLL、相对native与固定动作gain、误差/SE、输入可用时点；
- 每对动作的predicted_difference、observed_difference、paired SE、decision margin、correct/incorrect/unresolved；
- 可分辨比例必须与条件准确率同时报；不能删去难点只报准确率；
- h4选择在h16/h64的实际regret和反转；
- 一个事后有限网格最佳动作的收益上界诊断，与最佳固定动作比较。它是hindsight，不是可部署收益；
- 模型构建/刷新/评分耗时 vs 直接候选重放；
- 按训练seed、起点分别展示，不用总体cosine代替。

继续规则：
- 模型能解释field但不能分辨动作：不进入复杂在线控制，先报告目标信号/误差或可用固定改进；
- 所有真实最优动作仍是同一边界：这是当前动作集与状态上的固定调节结果，不扩大动作网格来制造自适应；
- 模型比M0好但短期评分长期逆转：冻结此策略失败结论，只针对读出/动作时长做一次明确修订；不立即加大理论；
- 只有在真实动作价值有空间、模型抓得到且诊断成本可能覆盖时，进入C。
这些是进展判据，不是普遍不可能性定理。

### C. 首个实际规则与最终目标 pilot

Phase B之后，只冻结一个可执行规则。推荐初版低频决策：每16步决策一次，选择未来4步的u，余下12步回native；与B相同动作语义。
如果改变这一定义，需要另行明确而不能沿用B为同一动作验证。
选择标准使用B冻结的模型目标与不确定性规则；不让执行模型临时发明阈值。
当前没有已验证GTTR policy，所以本文件不伪称其已经可用。

至少比较：
1. native AdamW；
2. 在独立开发预算中调好的固定LR multiplier/schedule（必须允许它优于窄±.05网格；调参全部计成本）；
3. GTTR规则；
4. 无状态信息的盲动作对照：动作分布从开发数据先冻结，不能使用当前测试GTTR轨迹的未来动作来重排。
盲对照用于区分“多做了干预”与“选择时机有用”。可以同时列相同占空比的固定动作对照。

所有臂同一checkpoint、同一训练流和固定评价规则；实际轨迹分叉是正常结果，不强制theta相等。
先3 seeds作工程/机制pilot，样本量和预算在B数据回来后列清。若仍用256步，只称该预算的续训pilot。

主评价：
- 同一预定义目标的confirmed reach fraction、检测到达token/删失/未确认；
- 同token预算末端NLL；
- 含模型probes/HVP/评分的wall-clock；
- 与固定最优对照的配对增益及逐seed结果。
10% continuation-token改善可设为值得扩展的工程目标，不是预言或统计显著性阈值。只有更小效果时应由额外成本和估计精度决定是否值得继续。
节省训练tokens但端到端更慢，必须报告为token-efficiency现象，不称加速/节能。

若完整模型有预测价值但online成本过高，可以研究开发期学习的固定/阶段规则；必须算开发成本及明确摊销范围，不把训练所得的固定设置宣称自适应。

### D. 独立确认

冻结policy、参数、目标阈值、schedule、cadence、final使用方式和预算之后，使用未用于调参的训练seed与continuation streams。
建议至少5个真正不同训练seed，需新建checkpoint时单列前缀训练成本；旧seed增加stream不算新训练seed。
final每个已规定输出checkpoint只作一次确认；失败不依据final结果继续训练重查。
可预先规定“monitor确认点checkpoint”和“共同固定预算checkpoint”两个不同目标的final报告，但必须一次性固定，不能最后选较好者当唯一结果。
未达标报告删失和固定预算NLL，不能只平均成功者，也不能把预算上限当真实达标时间。
报告数据级与训练seed级不确定性，不能把两者混同。
独立tokens-to-target成功仍只属于continuation；要称完整预训练收益，需另从初始化完成足够训练预算。

### E. 规模/配方迁移

先26M，再视结果100M，不直接1B。每个baseline获得合理独立调参预算。
换数据、context、batch或precision时一次改变一个主要因素并记录区别。
固定理论结构不代表近似误差跨规模不变。报告r/probe/刷新/通信的成本。
没有真实功率积分，不报joules；没有下游评价，不报能力全面提升。

## 3. 本轮实际运行：A0 一次性准备

### 3.1 理论与范围

GTTR v0.2及已经明确的语义修订作为工作版本，不再写新理论章节。
本輪只有native calibration，没有新controller，没有GTTR算法性能结论。
现场只读：最近closeout、原PhaseA预检与runner/scheduler、实际bank和数据口径。不要为缺A0F2重启审计。

需要落实的语义：
- 唯一参数与decay/no-decay组；
- 真实累计optimizer计数/global-micro，正常/skip路径；
- master weights/compute dtype/moments dtype的真实清单；
- 正常clip与累积/reduction；
- 评价恢复train mode、RNG与数据位置。
这些只核查实际已启用功能，不引入FP8/MoE/分布式新框架。

### 3.2 解除scheduler阻塞：本轮明确的设计选择

标记：NATIVE_WITH_DECLARED_TAIL。
schedule_id：finite_cosine_then_hold_5e-5_v01。

eta(gm) = old_get_lr(gm,256,5e-4), gm<=256；
eta(gm) = 5e-5, gm>256。

gm必须与旧runner真实传入索引一致。若确有不同LR参数组，则保持原组比例并给出对应尾值，不把组全部改为同一个LR。
不重置m/v、bias correction、step或RNG，不把total_micro从256改512。
不使用max(floor, old_get_lr)（它不能去掉越界回弹）。
不增加warmup、峰值保持、终端cooldown。只是为未定义尾部选择恒定延续，不称最优schedule。
此选择只通过新PhaseA wrapper实现；旧scheduler函数和历史文件保持不变。
验证gm边界前后与0..原定义范围一致，并确认请求128/256时前128更新完全共用相同schedule。
若已有PhaseA代码仅做预检未main，允许在新配置revision中执行；保留原BLOCKED记录。
若现场发现该组实际终点不是报告的5e-5、索引有冲突，则输出具体值后停，不临时选择另一个尾值。

### 3.3 数据与目标

保持原train IDs（历史约3936行，但以实际冻结列表为准）。不扩大train support。
V_monitor候选D0 rows[4096:8192]，4096行；V_final rows[8192:12288]，4096行。
所有区间均半开区间。沿用已有ID注册若未变更。
检查当前train/历史开发/monitor/final行ID、可用doc ID、所选集合的精确token序列重复。
发现已知冲突则输出具体ID并停；没有doc ID时可继续受限pilot但写DOCUMENT_INDEPENDENCE_UNVERIFIED。
不扫描全语料近重复，不重开数据大工程；将来正式泛化实验再完善相应数据分割。
训练采样支持明确排除monitor/final。仅查看final身份哈希/重叠信息，不运行final模型评价，也不按文本内容挑final。

固定tokenizer、context、shift、mask、ignore_index、eval dtype。
V_monitor=全库NLL_sum/valid_target_count，保存每行或每doc的sum/count。
同时计input token slots、nonpad tokens、valid loss tokens；本輪1024/optimizer step只能作为预期，现场按mask和accum核对。
不把每microbatch loss均值再无权平均当作全库token mean。

## 4. A1 六条native轨迹与观测

seeds {2030,2031,2032} × checkpoint标签 {46,82}，共6条；真实n0从checkpoint读取。
每条一条新continuation stream；默认97000..97005按(seed,start)排序分配。
若这些ID已被非空真实训练使用，则整体+1000直至无冲突；只注册而从未运行的不重复更改。
不根据数据结果更换seed、起点、stream。

每条运行256 optimizer steps（若实际出现skip，记录尝试/成功/成本，不静默延长补满另一个口径）。
固定监测h={0,32,64,96,128,160,192,224,256}，全4096行，eval batch32，128 forward batches/次。
不在校准期间命中候选阈值就早停；所有成功轨迹运行同一固定窗口。
仅数值失败停止该轨迹，保留失败和全部成本；不自动调LR、beta或precision救该case。

保存h128/h256可恢复checkpoint，含模型、optimizer、scheduler、step、RNG、data cursor；h0引用原checkpoint。
无需每步保存高维tensor。若h64已有便宜snapshot可选保存但不因此扩大预算。

### 4.1 最少诊断

每optimizer step复用已有：LR、clip coefficient/flag、train NLL sum/count、actual update norm、参数组、token计数、finite/skip状态。
在h=1和16的倍数记录唯一attention/MLP二维矩阵：
- W_norm、s_norm、自适应d_norm；
- relative_update=||s||/||W||；
- radial=<W,s>/||W||²，tangent位移norm及其分解残差；
- raw pre m/v、c、I_m=c-m_pre、I_v=c²-v_pre 的norm；
- I_m/I_v列名明确INNOVATION_PROXY，不称真实tracking error。
零/过小分母用NA+原值，不静默epsilon救数字；tied参数去重。
不新增Hessian、mu/nu/噪声协方差、表征rank、attention entropy测量；这些不是本轮问题。

### 4.2 EMA尺度

报告beta1/beta2、step_half=log(.5)/log(beta)、token_half（固定实际Btok条件下）、beta^64/128/256，以及
old_corrected_weight=beta^H*(1-beta^n0)/(1-beta^(n0+H))。
只是滤波权重，不是实际响应寿命或物理弛豫时间。

## 5. 实现检查：复用，不重复造框架

复用原PhaseA T1–T8已有测试，只补新tail wrapper的边界/前缀检查。
重点：sum/count、zero vs None、参数去重、计数、随机恢复、禁final eval。
不再为了审计回溯每个旧实验的细枝末节。

smoke固定第一个cell：8步diagnostics off与8步on，同完整初态/stream，共16真实updates。
on臂h0/4/8做monitor；off不评价。核对最终theta/m/v/step/RNG/data cursor不被诊断改变。
若支持bitwise就用bitwise；若原有实现只保证有界数值一致则使用原有已声明容差，不按结果放松。
参数范数相同不等于完整状态相同，报告准确的核对级别。
所有失败尝试计费，发生真实实现bug时先停止修复，只在剩余额度内按同smoke复测。

## 6. A1预算：只执行本轮，不覆盖后续路线

main=6*256=1536 updates；smoke=16；计划1552；硬cap1664（112恢复余量不换新科学cases）。
额外诊断gradient=0、HVP=0、V_final model-eval=0。
训练backward照常存在，不能把总gradient calls报0。

main monitor evaluations=54，forward batches=6912；smoke=3*128=384；计划monitor forwards=7296；monitor/复核cap7800。
所有训练+monitor+失败 model forward cap12000。
GPU任务墙钟cap90分钟；包含本轮train/eval/诊断/恢复，不是理论阅读时间。
smoke后给出实测单位时间及剩余ETA；明显无法在cap内完成则保存smoke并报告，不擅减样本、改变monitor频率或目标口径。

历史18000授权段已经查得18730实际更新，不改历史；独立C1/C1L等账不改。本轮单独记录actual counts及planned/cap。
不声称拥有真实能耗结论，除非本轮明确单列实际功率测量（本计划不要求新增）。

## 7. Phase A指标与决定规则

每条曲线报告monitor sum/count/mean、same-row/doc paired NLL变化与不确定性，数据block不确定性与训练seed变化分开。
只有3个训练seeds，两个起点相关；6条轨迹或54个读数不是6/54个独立训练seed。
可沿用固定64行block proxy+1000次CPU paired bootstrap；有doc ID优先固定doc blocks。标明它不是总体训练seed置信区间。

开发候选阈值规则（结果前冻结）：
对每个起点g，若3 seeds均有h128，ell_candidate[g]=median_seed V_monitor(seed,g,h128)。
输出V0-ell、ell-V256及噪声余量；各起点任务分列，不跨不同阈值汇总tokens。
若候选没有位于初末之间，TARGET_UNRESOLVED/FIXED_BUDGET_ONLY，不改分位数/时点。

本轮可以回看两个连续monitor检查点<=ell的detected_hit；所有此类记录均CALIBRATION_ONLY，不能称独立tokens-to-target确认。
保存last-above、first-below、second-confirmation；不插值造精确hit；未达标保留censored=true，不把cap当hit。
V_final全轮不读表现。本轮“达到目标”只是开发阈值可行性，不是算法效果。

阶段A成功不是GTTR_PASS：
- IMPLEMENTATION、DATA_SPLIT、SCHEDULE、COVERAGE逐项报告；
- NATIVE_PROGRESS_MEASURABLE：原数据级变化与不确定性数值，而不是一个“loss下降”；
- TARGET_CALIBRATION：可用/不可用；
- ACTION_VALUE、ADAPTIVE_GAIN、FINAL_CONFIRMED_TARGET、ENERGY_BENEFIT全部NOT_TESTED。

若low-tail导致停滞：保留结果，下一项最多研究一个明确定义的baseline schedule改变；不直接扩模型理论。

## 8. 交付与直接展示的数据

核心文件：
config.json、preflight.md、schedule_amendment.md、dataset_contract.json、schedule.csv；
smoke_checks.json、resource_log.jsonl、trajectory_metrics.csv、monitor_row_stats.csv；
layer_diagnostics.csv、ema_timescales.csv、target_calibration.csv；
status.json、REPORT.md、KEY_DATA_FOR_REVIEW.md、h128/h256快照manifest。
保持小而可复算，无大型审计tar，无每步tensor存储。

最终聊天直接给表，不只回复文件路径：
1. 六cell seed/起点/n0/stream/LR首尾/tail/beta/batch/accum/实际token/完成状态；
2. h0/64/128/256 monitor sum/count/mean、paired差/SE、累积token与wall；
3. 每起点ell候选、逐seed初末余量、回看hit/censor及final_call_count=0；
4. 少量固定层的原始norm、radial/tangent及innovation proxy统计，完整其余值落CSV；
5. planned/actual/cap，训练与评价分别计数，失败尝试不漏；
6. 只回答下一轮缺哪项数据/机制，不自动启动B。

完成后停止。不commit/push、不安装无关依赖、不更改模型接入、不干扰他人GPU、不建常驻monitor。

## 9. 立即可转发的Claude启动指令

请读取本文件，按本次选定的 finite_cosine_then_hold_5e-5_v01 明确尾部，完成A0+A1。
理论暂停扩张，不重开大审计，不启动B/C/D/E。
复用旧PhaseA已有代码，仅在新入口包裹声明的schedule tail，旧scheduler/历史输出保持原样。
对照第3–8节的数据、状态、smoke、资源和报告要求；关键数字直接在最终回复展示。
这次的目标是得到真实native LM-NLL/token参考曲线和可用的开发目标，不是证明GTTR已经优化训练。
