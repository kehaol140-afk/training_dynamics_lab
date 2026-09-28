# FR C0X + C1：GTTR 离线解释与误差来源验证

用途：交给现有 FR 仓库内的 Claude Code 执行。
状态：新一轮研究执行协议，不是远程运行结果。设计者只读取了 C0 协议和用户完成摘要，未读取远程 C0 原始 CSV。

## 0. 本轮目标、授权与范围

用户已授权：继续完成 C0 离线解释，并在没有实现/资源阻塞时继续小规模理论验证。

先完成 C0X（只读取已有制品，CPU 离线复算，0 新 forward/gradient/HVP/optimizer update），然后进入 C1（固定状态上的新 probe 诊断）。这不是让 C0 改成 PASS 的重跑，而是区分：

1. C0 规则执行/读出定义是否存在错误；
2. 局部几何预测误差中，跨 probe 库的波动与同 probe 近似缺陷各有多大；
3. 这种误差经过真实 clipping、m/v、预条件器后，怎样出现在更新增量与固定读出方向上。

不要求模型得到正结果。负结果、低分辨率、缺少资料均是合法输出。
不开发 controller、新 optimizer、LR/beta/batch 策略，不训练新 checkpoint，不扩展 HMM/GMM，不搜索新理论，不拟合长记忆核，不做完整 Hessian 扫描，不自动进入 C2。

科研优先：定义正确、能运行、能复核、能出原始表和图；不重构仓库，不新增数据库/服务/复杂调度，不制作审计 tar，不递归 hash，不自动 commit/push，不改模型接入设置，不影响其他训练进程。只 hash 本轮 config 与必要结果文件即可。

## 1. 必读材料和证据等级

现场读取：
- FR_C0_Geometry_Tracking_Validation_Claude_Code.md；
- C0 执行前冻结的 config、任何结果前批准的 amendment；
- outputs/fr_c0_gttr_validation_v01/ 内 REPORT、summary、prediction/readout/cell 表、per-probe 统计、floor 原测与 floor2、smoke 记录、budget log；
- 实际 C0 runner、analysis 和 frozen AdamW/clip/HVP/restore primitives；
- 与本轮有关的 A0T-extra 原始 cell/目标定义，以及 A0C/A0E2/A0O 的记录；A0F2 原报告若缺失仅标 SOURCE_NOT_FOUND，不阻塞其他部分。

不要把用户摘要当成原始数据，不把上一轮助手的推论当成已经证实的事实。

当前仅作为待核对读数：
- formula PASS；field 正式 NOT_SUPPORTED，平均 E 约 0.42–0.44；
- C/Bp/t16 的 last-4 半库相关读数 1.166，但其是单点还是聚合尚需核对；
- shadow UNRESOLVED；tracking/accounting PASS；
- 历史总 updates=17,994/18,000；C0 forwards=2,428/2,400，有 28 次超额；
- A0T full v_score 与 C0 64-sample v_eval 不同，不能直接减常数后宣称复现。

实际文件、累计步号、参数组、decay mask、clip 数值稳定项、attention backend、精度均以现场实现为准。不要编造缺失函数或路径。

## 2. GTTR 冻结主干：同一套更新的四种分析，而非四个独立模型

固定模型/目标/数据分布时，L_train(theta) 为固定函数，H(theta_t) 可沿轨迹变化。不得仅因 representation learning 添加独立 rho 动力学状态。

Geometry：
g_t=grad ell(theta_t;omega_t)，c_t=C_tau(g_t)。
固定当前 probe 随机性时，G_t=D_theta c_t=J_clip(g_t) H_batch,t。
G_t 一般非对称；H_batch 与独立目标 V 的 H_V 不混用。

Tracking：
mu_t=E[c_t|F_t]，nu_t=E[c_t^2|F_t]，F_t 在该 probe 随机输入揭示前定义。
bar_m_t=a_t bar_m_prev+(1-a_t)c_t；bar_v_t=b_t bar_v_prev+(1-b_t)c_t^2。
e_m=bar_m-mu；e_v=bar_v-nu。
同一个冻结状态上的 probe 均值，不能用不同 theta 上的训练流均值替代。
E[P*m] 一般不等于 E[P]E[m]，E[Adam(g)] 一般不等于 Adam(E[g])。

Transport：
P=diag(1/(sqrt(bar_v)+eps))，r=P*bar_m，d=-eta*r，s=d-eta*lambda_wd*theta。
d_t=T_t d_prev-eta_t(1-a_t)P_t c_t，
T_t=a_t(eta_t/eta_prev)P_t P_prev^{-1}。
T_t 只是已实现轨迹的位移分解系数，不替代 counterfactual tangent A_t。
当前 log-LR 的直接输入为 [s_t,0,0]；后续 moments 会经 theta→g→m/v 间接改变。

Readout：
q=grad V_diag(theta)，W_total=-q^T s，K_V=V_diag(theta+s)-V_diag(theta)+W_total。
Delta V=-W_total+K_V；K_V 可为负，不等于正的曲率惩罚。
W_field=eta*q^T P mu；W_track=eta*q^T P(bar_m-mu)；W_wd=eta*lambda*q^T theta。
这些是 realized 记账，不是独立可干预的因果份额。

完整模型仍是 z_next=F_t(z,u,omega)，z 包含实际 theta/m/v/step 及必须的执行状态。
未来路径条件响应和当前可用信息下的期望收益必须区分。

## 3. C0X：离线完成四项复核

建议新建 outputs/fr_c0x_offline_explanation_v01/；保留 C0 原结果。

### X1. 原始判据是否被正确执行

逐条对照 C0 协议第 10.1、10.2 节与执行前 config/amendment。
不能按结果选择较宽的版本；有相互矛盾且无法确定优先级的结果前规则时，报告 PROTOCOL_AMBIGUOUS。

主要公式：
E=sqrt(sum_{a in A} ||p_a-y_a||^2 / sum_{a in A} ||y_a||^2)。
A 是每个 cell 用最终正确 floor 和全库规则确定的固定 resolved anchor 集合。

输出每个 cell、field/shadow、full/first4/last4 的：
- anchor IDs、count、distinct arms、distinct local t；
- SSE=sum||p-y||^2、target_SS=sum||y||^2、聚合 E；
- 每个 anchor 的原始 norm、SE、floor、resolved 与单点误差；
- 分母接近零时明确 UNRESOLVED，不除以任意大 epsilon 伪造稳定值。

核实 1.166 属于 C cell 的整个半库聚合还是单个 C/Bp/t16。
原协议的半库门用同一 A 集合的聚合 E<=1，不默认“任何单点 E>1 就失败”。
若代码确实错用了规则：保存原 verdict、错误位置、按真正冻结规则复算的 verdict 和 erratum；不覆盖原始文件。
若 1.166 本来就是聚合指标，则保留原判定，不反复寻找替代指标。

floor 修正：cross-probe 1.65 与 same-probe floor2=0 的定义不同；检查所有 resolved flags、field/shadow 半库指标和 verdict 是否已统一重算。
0 只代表测得的重复性，不代表系统精度/HVP 偏差为零。

### X2. probe 误差与 shadow 分辨率

只利用已保存数据。对每个 anchor 分别列出 development prediction 和 audit target 的均值 norm、SE_vec、两个半库均值差、固定投影统计。
定义 SE_vec=sqrt(sum_j||x_j-xbar||^2/[K(K-1)])。
不要把 3SE 当成严格多维置信球。

读取既有 MnoJ 和 stop-v 消融：full U 与 stop-v 的目标 signal、SE、方向和误差分别报告。
小信号不自动表示 preconditioner 抹掉信息；统一标量缩放同时缩放均值与SE，不改变SNR。
仅有更新幅值较小，不能解释信噪比为什么变差。

若没有 per-probe vectors/充分统计，相关条目标 DATA_MISSING；不能从几个 norms 反推向量差或交叉项，也不在 C0X 开新 GPU 计算。

### X3. 同一个目标上的累计收益解释

对 C0 的 R/C、A/Bp/Bm，各自读取：
- V_diag 上 h=0/4/16/64 的损失与 gain；
- V_eval 上同样时点；没有逐步数据时只报已有点，不插值当实测；
- V_diag 上累计 delta W 与 delta K，以及 native−arm gain。

正确符号：
G_H^a=V_H^native−V_H^a
     =sum_t[(W_t^a−W_t^native)−(K_t^a−K_t^native)]。
区间只用 [0,4)、[4,16)、[16,64)；不能按曲线挑新的转折段做主结论。
既报告 arm−native，也报告 Bp 与 Bm 的直接比较，避免混淆“仍优于native”与“最优方向改变”。

只在 C0 目标自身确实出现反转时称“本目标上的反转”。
若无反转，照常报告，不调换 case，不把历史 A0T 标签当本轮结果。

A0T axis：列样本 IDs/规模、reduction、精度和目标名称；无共享轴则保持 AXIS_MISMATCH。
不减去约0.106强行对齐，不把近常数offset当作完全相同目标。
若可从已存数据计算 offset 的非恒定部分，则与 arm gain 尺度比较；否则标不可计算。

### X4. 账本、判定与证据摘要

只追加修订，不删失败记录：核对 417 updates 的批准记录及总账；2,428 forwards 超 2,400 上限28次须明确保留。
“每进程不超限”不能替代轮次总限。
外部 lr/clip/step 等标量匹配不能宣称完整 theta/m/v bitwise 相同。

C0X 给出：
- 什么是实现等价已核对；
- 什么是局部模型的经验改进；
- 什么只是记账；
- 哪些环节因误差/样本而没有结论；
- 是否存在影响 C1 的真实执行错误。

## 4. 从 C0X 进入 C1 的条件

C0X 完成后无需再次询问用户，可以按本协议继续 C1，但只在以下条件成立时：
1. 实际 gradient/clip/HVP/shadow-update 定义已确定，未发现尚未修复的状态污染、未来信息泄露或导数实现错误。
2. C0 的两个具体 cells/arms 及关键训练状态可恢复。
3. 所需资源在本轮新诊断上限内可完成。

C0 NOT_SUPPORTED 不阻止 C1，因为 C1 正是辨析误差来源，不要求先看到正结果。
C0X 的旧 per-probe 数据缺失不必阻止可恢复状态上的 C1，但旧 C0 对应结论仍标缺失。
存在真实实现错误时：先CPU验证修复，再用新版本进行数值门；保留 C0 原结果的有效范围说明。不能悄悄换实现后称严格复现。
无法恢复状态、导数代理无法验证或预计超预算时停在 BLOCKED/PARTIAL，不另搜更漂亮的case。

## 5. C1 新额度：与旧 18,000 次上限分账

用户这次允许继续理论验证。本提示词执行时，为下面这一个固定诊断开启独立小额度；不改旧 C0 总账，也不把旧18,000上限写成重新获得。

- C0X：0 新 forward/gradient/HVP/update。
- C1：最多192次真实 optimizer updates，全部只用于恢复/重放既有诊断状态；预计132 main replay +8 smoke=140。
- 有有效完整 snapshot 时直接复用，不为用完预算重放。
- 非训练 effective-batch gradient evaluations ≤1,400，HVP effective-batch evaluations ≤416。
- 非训练实际 micro-forward 调用 ≤3,200；内部HVP/重复floor/诊断读出中的forward全部计入，不能按进程重置。
- 上述 effective-batch 和 micro-call 两套计数同时保存；不同依赖primitive内部调用不能漏报。
- 本任务占用GPU的运行墙钟上限90分钟；CPU分析单列。根据首个小块实测给ETA，超过预计就保存停止，不承诺固定耗时。
- shadow算术次数/耗时另记；不算真实训练update，不等于没有成本。
- 所有失败尝试、重试、数值复核、salvage都计入对应总计；不要结束后再补一个超额“零update”修复。

旧账预计17,994单列；C1实际费用单列；可展示全项目累加，但不要继续把新合计除以旧18,000做合规判断。
本轮不测或宣称能量收益。

## 6. C1 固定设计：6个 anchors、新probe，不改变模型

复用 C0 的真实 Cell R/Cell C，具体seed/checkpoint/stream ID从其config读取，不硬编码猜测。
每个cell固定3个anchors：
- native A，更新前 t=0；
- persistent Bp（u=+0.05），更新前 t=16；
- native A，更新前 t=48。
共6个。既包含旧边缘位置，也包含预先规定的早/中/晚对照，不根据C0X结果替换。

A路径需要运行至完成第49次update，取得t48的真实s；Bp路径至完成第17次update。
两cells main replay =2*(49+17)=132 updates。
smoke：Cell R/native前4步诊断开/关各一次，8 updates。可核验已有完整snapshot降低真实开销，但不得仅凭标量匹配宣称全状态等同。

相同初始状态/不同streams、不同arm之间的相关性必须记录；6 anchors不是6独立seeds。
这些位置是历史条件下选择的机制样本，不是新的总体确认集。

每个cell：32个新development probes D、32个新audit probes A，固定ID顺序。
- 与C0 probes及真实rollout未来批次分开，按可核对样本/token IDs检查；不能只检查seed不同。
- 来自同一训练采样分布，每个probe使用与真实训练相同的effective batch/gradient accumulation，然后clip一次。
- 不把32个probes合为更大的训练batch。
- 同probe在theta、theta+s、预测梯度计算中复用相同batch和dropout随机性。
- 两banks互不重叠；跨anchor/arms保持固定配对设计并承认相关。
- 样本数一次性固定32。K=8/16/32仅是相同32个probe的固定前缀读数，不看结果决定继续采样，不扩至64。

新模型仍只有 M0 和 Mgeom。MnoJ及stop-v仅为既有诊断消融，不引入可调系数/拟合器。

## 7. C1 核心测量：用精确的误差分解区分跨库波动与局部模型缺陷

固定anchor，保存theta、实际s、后更新共同optimizer state S+及正确下一累计step/eta。
对于每个probe j定义：

g0_j=g(theta;omega_j)
h_j=H(theta;omega_j)s
g1_j=g(theta+s;omega_j)

P_j=C(g0_j+h_j)-C(g0_j)       # 局部几何预测的训练场变化
Y_j=C(g1_j)-C(g0_j)           # 同probe真实训练场变化

Q_j=U(g0_j+h_j;S+)-U(g0_j;S+) # 预测的shadow更新增量
Z_j=U(g1_j;S+)-U(g0_j;S+)     # 同probe真实shadow增量

U逐probe使用真实clip、m/v、bias correction、P和衰减mask，固定相同S+；禁止用均值梯度替代逐probe非线性更新的均值。

### 7.1 信息防泄漏顺序

第一阶段只读取 D 的起点g/Hs，构成各K的 mean(P_D)、mean(Q_D)，保存固定预测。
在预测保存后，才能读取任何probe终点的g1，以及audit Hs。

为了辨析误差，预测保存后允许在audit上计算P_A/Q_A，并在development上计算Y_D/Z_D。
这些same-probe量明确标为 CONDITIONED_DIAGNOSTIC，不能回填、校准或选择前面已保存的独立跨bank预测。
这区分“路径/同probe条件近似有效”与“对新probe分布的均值预测有效”。

### 7.2 必须重算的精确分解（field与shadow各一套）

field：
e=mean(P_D)-mean(Y_A)
e_bank=mean(P_D)-mean(P_A)
e_local=mean(P_A)-mean(Y_A)

e=e_bank+e_local。
||e||^2=||e_bank||^2+||e_local||^2+2<e_bank,e_local>。

shadow把P/Y分别换成Q/Z，公式相同。

不可把两项norm百分比相加成“误差贡献比例”；交叉项可能大且为负。
e_local是当前条件下的有限样本近似缺陷，包含数值/HVP代理误差和有限位移误差；没有精度验证就不能全部叫Taylor截断误差。
e_bank是一组实测跨库差，不是严格的总体采样方差，也不等于不可约噪声下界。

在独立同分布probe、有限二阶矩条件下，理论参照：
E||mean(P_D)-mean(Y_A)||^2
 =||E[P]-E[Y]||^2+tr Cov(P)/K_D+tr Cov(Y)/K_A。
只用作误差尺度解释，不假设高斯、不假设每一组样本都必须随K单调改善。

## 8. 数值与读出方向

继承C0固定精度和可靠HVP代理。若代理与真实forward目标不同，要明确两者，不把代理准确性当真实函数精度。
独立数值验证与same-probe重复性分别报告；floor=0只表示重复计算结果一致。
小型CPU测试验证误差分解、符号和聚合。GPU只做少量与main相同primitive的检查，全部计预算；不为追逐machine precision开启长debug循环。

在读任何audit终点前，确定两个投影方向：
- 主读出方向：q+=grad V_diag(theta+s)，目标及样本定义与C0完全一致；最多6次effective-batch gradient，不重选validation数据；
- 次方向：实际s/||s||，若norm为0标缺失。

分别报告P/Y的预定方向投影，以及shadow一阶读出增量 -q+^T Q、-q+^T Z。
这不是执行下一步的真实loss，也不包含未来的曲率累积；必须标 SHADOW_FIRST_ORDER_READOUT。
不能观察audit以后挑一个投影方向让信号更显著。
vector指标与scalar指标分开；scalar可辨识不替代旧C0 vector gate。

## 9. 预条件器诊断：复用已有数值，不增加模型

对每个probe，使用C0中明确定义的stop-v：在一对U中，P固定为基准g0形成的P，m依然按输入更新。
报告Z_full与Z_stop的均值norm、SE、相互方向，以及Z_full−Z_stop。
如果做贡献平方分解，保留交叉项，不用norm比例假定独立可加。

可以另外计算固定common-state下的局部c→U导数用于诊断，相关v_i>0时：
D_i=−eta[(1−a)P_i−(1−b)m_i c_i/(sqrt(v_i)(sqrt(v_i)+eps)^2)]。
该a/b是正确下一步偏差校正系数，c/m/v为同一probe的基准shadow量。
原始g的输入变化先经过实际clip；遇到边界/零矩坐标按frozen primitive处理或标不适用，不隐藏加平滑。

核心问题是full和stop-v的SNR与样本方向结构是否不同，而不是仅问full位移norm是否小。
本诊断不能推出“冻结v更好”，也不试验新的optimizer。

## 10. 统计输出与判定，禁止新的择优筛选

C1为误差来源机制诊断，不作为C0的重新验收。C0正式判定只由C0X按原规则计算。

1. 主输出覆盖全部6 anchors；field/shadow分别计算每cell三个固定anchor的聚合E与绝对SSE，不按新resolved结果筛掉anchors。
2. 每anchor报告K=8/16/32的目标norm、prediction SE、target SE、same-probe residual SE、数值floor、3SE分辨标志。
3. K32两个预先固定16-probe半库，各自用全部同一anchors计算E；不重新选anchor。
4. 每个cell、每个时间位置单独报告，不只呈现由大信号anchor主导的总均值。n=2 cells不作seed总体显著性。
5. scalar投影可以给均值±3SE等描述性范围，但不是严格总体置信结论。不得把参数坐标数当样本数。
6. K之间嵌套、不同anchors共享probe库，曲线不是独立重复实验。

结果使用分层描述，不制造一个新的“统一理论PASS”：
- C0_RULE_REVIEW：UNCHANGED / ERRATUM_REQUIRED / PROTOCOL_AMBIGUOUS / DATA_MISSING。
- C1_EXECUTION：VALIDATED / NUMERICAL_UNRESOLVED / PARTIAL / BLOCKED。
- SAME_PROBE_APPROXIMATION：按anchor列实测误差与分辨性，是否优于M0。
- INDEPENDENT_BANK_PREDICTION：按cell、时间、K列E和半库；只有双方cell K32 E≤0.8、两个16半库E≤1，且每cell至少两个不同t的目标均可分辨，才可写 LOCAL_SUPPORT_ON_SELECTED_ANCHORS。否则区分 RULE_NOT_MET 与 UNRESOLVED，不能用单点替代聚合。
- SHADOW_PREDICTION：同口径独立判断，不能由field结论推断。
- ERROR_DECOMPOSITION：报告bank/local/cross-term，允许MIXED或UNRESOLVED，不强行把误差唯一归因。
- FUTURE_H16_H64_PREDICTION / ADAPTIVE_TRAINING_VALUE / ENERGY_BENEFIT：固定NOT_TESTED。

3SE和0.8是事先指定的工程/实用门，不是物理定律。
达到上限仍不分辨时，结论是“当前测量未解决”，不是“理论无效”或“不可预测已证”。

## 11. 数据保存与最小实现

复用C0 runner/diagnostic primitives；新增一个离线脚本、一个小C1 runner/analysis模块和必要测试即可，不复制整套训练器。

a. 结果先保存原始标量表，再出图。
b. 保存足以重算所有预定K/半库/聚合E/SE/误差交叉项的统计：各组count、mean norm²、sum norm²、SSE、必要inner products以及逐probe固定方向投影。
c. 仅保存单点norm而没有均值差/交叉项不能声称支持误差重构。
d. 临时dense向量只在单个anchor内处理；可使用现有流式/Welford累积，不创建所有状态的全梯度巨型archive。不需要随机向量bootstrap；避免为了统计框架额外花GPU。
e. 若已有per-probe向量便于CPU复核可复用，不要求强制删除。不要删除旧原始数据。

C0X至少输出：
- audit.md、rule_recalculation.csv、verdict_comparison.json；
- target_axis_map.csv、gain_balance_by_interval.csv；
- signal_resolution.csv、REPORT.md。

C1至少输出：
- config.json、state_provenance.csv、smoke_checks.json、budget_log.json；
- probe_summary.csv、error_decomposition.csv、fixed_projection.csv；
- update_feedback.csv、verdicts.json、REPORT.md。

另输出一份短 GTTR_EVIDENCE_UPDATE.md：分“数学恒等式 / 已测经验支持 / 尚不分辨 / 被特定实验削弱的假设 / 未检验”五栏，写明下一步唯一最有价值的问题，不自动实施下一轮。

最多5张图：旧C0聚合/单点差异；同目标累计W/K与gain；C1跨库/同probe误差随K；shadow和stop-v的signal/SE；固定读出方向误差。
每张图独立，由已保存表生成，不人工改数，不制作PPT。

## 12. 启动前小测试与最终报告

CPU至少验证：
- 单点E>1但全库聚合E<1的实例，防止误写门槛；
- e=e_bank+e_local及平方范数交叉项恒等式；
- 同probe差与cross-probe差数值floor的区别；
- mean(U(g))不等于U(mean(g))；
- W/K累计gain符号；
- common-state shadow不污染真实theta/m/v/RNG，预测保存后audit不可回填。

C1 smoke确认诊断开/关真实状态一致；外部旧表缺完整state时，只说明哪些共同字段已核对，不把它们夸大成全state identity。

最终用中文解释科研结论，附实际运行命令和路径，不只发PASS/FAIL列表。
明确回答：
1. C0原判定有无误用规则？若有，修订影响哪些结论？
2. C0同一目标上是否真的反转，W/K在何处抵消？
3. 局部几何近似在同probe上是否准确，跨库又损失了多少？
4. 增加probe后哪些不确定性减小，哪些缺陷仍在？
5. 实际AdamW输出耦合怎样影响这个局部增量，证据是否足够？
6. GTTR哪一段有证据，哪一段仍需要理论闭合？
7. 真实更新与全部诊断开销、停止原因、数据不足分别是什么？

结尾不得宣称已提前预测长期反转、获得新optimizer或节能。完成本协议后停止，不自动进入C2。
