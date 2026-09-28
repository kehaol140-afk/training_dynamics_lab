# FR Phase B v3：GTTR 动作前评分与真实分支验证

日期：2026-09-23。
执行端：现有 Claude Code + Kimi K2.7，沿用当前仓库和环境。
本文件是下一轮实验的执行规范，不是已经获得的实验结果。

## 0. 本轮目的、授权和与 v2 的关系

用户将本文件与启动提示词交给执行端后，执行本文件内的 CPU 测试、数值 smoke、模型评分、真实配对分支、分析和报告；不要只返回计划。仅遇到真实状态/数据/实现/预算阻塞时停止并说明。不自动进入 Phase C，不设计新 optimizer/controller，不重做历史大审计。

科学问题：
1. 局部梯度场几何信息，是否改善动作执行前的 NLL 差预测？
2. 由 H=4 或 H=64 评分选出的同一个四步干预，在实际 H=1/4/16/64 上表现如何？
3. 误差主要来自近似训练传播，还是冻结起点目标梯度的一阶读出？
4. 相对固定 LR-up，是否存在可辨识的选择收益，以及取得信息需要多少计算？

保留 v2 的 6 个原 bank 状态、3 个真实动作、4 步干预后释放、最大观测 H=64、P=2 microbatches、最多 Q=4 个方向、旧数据与真实 AdamW 配方。

本文件明确修订 v2：
- 模型内每个虚拟步重新执行真实 clipping，不采用“预测原始梯度但忽略 clipping”的版本。
- 不使用 V_monitor 梯度构造评分；开发侧建立 V_control_ref64，V_monitor 用来审查选择的迁移效用，V_final 封存。
- 把 P/Q 对应的梯度场近似、方向构造、有限差分精度及误差字段写成可执行定义，不允许临场自创替代模型。
- 内部轨迹仍算到64，并同时保存 H=1/4/16/64 的评分；H4是首要模型检验，H16/H64为预先规定的外推检验，不事后选择最佳时域。
- 评分的几何、统计和读出简化全部如实标注。低秩参考模型尚未获得新验证，不将旧 C1 的局部支持当作本模型已通过。
- 修正预算：同时包含 smoke、开发目标评价、模型构建的 backward 和虚拟 optimizer 算术。

v2完整文本如果在仓库中存在则读取并保存差异说明；本文件中的新增近似选择属于此次修订，不声称 v2 本来已写明。

## 1. 固定训练对象与三个真实动作

### 1.1 状态与真实流

按 (seed,start) 排序：

|state_id|seed|原 bank start 标签|真实 evaluation stream|
|---|---:|---:|---:|
|S0|2030|46|97006|
|S1|2030|82|97007|
|S2|2031|46|97008|
|S3|2031|82|97009|
|S4|2032|46|97010|
|S5|2032|82|97011|

真实累计步号/global-micro 从 checkpoint 读取，不由名称猜测。h128/h256 快照不混入该表。
上述 stream 若仅注册、尚未执行，保持 ID；若确有非空真实结果，不能覆盖，说明冲突并停止主实验。不要依据结果更换状态、seed或数据流。
每状态一条新未来流，3个动作严格配对。这是3个训练seeds、6个状态的开发 pilot，不是6个独立训练seeds，也没有足够重复识别状态条件下完整future-stream分布。

模型 SIM 流：每状态使用 197006..197011，对应读取两个microbatches；与真实evaluation流分离。若ID已执行过，则整个SIM ID块+1000直到无冲突，依据纯元数据在结果前冻结。SIM可以从相同train support抽样；“随机流ID不同”不宣称语料分布独立。

### 1.2 实际配方

复用 Phase A 已核验的约4.98M网络、真实train support、tokenizer/context/mask、AdamW参数组、累积与数值语义。预期配方是 weights/moments FP32、训练BF16 autocast、scaler disabled、beta=.9/.999、eps=1e-8、wd=.01、global L2 clip=1、microbatch4×accum2。若实际冻结配方不同，先列具体差异，不偷偷改成预期数。

唯一schedule：
- old global_micro <=256：old_get_lr(gm,256,5e-4)；
- gm>256：5e-5。
保持真实索引、bias correction、moments和组比例。只用新wrapper，旧代码和历史结果不改。128/64等读出不改变同一条schedule前缀。

### 1.3 动作

三条真实臂：NATIVE、LR_DOWN、LR_UP。
取 u∈{0,-.05,+.05}，multiplier=exp(u)，不是1+u。
对第k次optimizer更新（k=0..63）：
    eta_k^u = eta_base(n0+k) * exp(u * 1[k<4])
前4次更新干预，此后恢复native LR，保留形成的theta/m/v，不重置状态。
同一个动作始终观察H={1,4,16,64}，不随H延长干预。
不改变beta、clip、数据权重、batch或weight decay。LR乘子同时作用于原AdamW自适应位移和解耦衰减的本次eta。

## 2. 目标、数据角色与评分公式

### 2.1 V_control_ref64：开发评分目标

读取旧V_score的64行具体ID；不得根据文本或loss重新挑行。定义本轮开发参考轴：
    V_control_ref64(theta) = sum(row_nll_sum) / sum(row_valid_target_count)
使用固定eval mode、FP32 reference计算、实际shift/mask、eval batch8。

如果此sum/count或dtype与旧V_score数值语义不同，给新target_id=V_control_ref64_v1，不与旧A0T绝对loss直接拼接。读取其ID只是复用开发资料，不声称它是独立测试集。

计算 q0=grad V_control_ref64(theta0) 时，按全库token分母累积梯度，不无权平均8个microbatch loss。保留每行sum/count以独立复算。

### 2.2 V_monitor 与 V_final

V_monitor继续使用 Phase A 4096行全库sum/count，保持原eval dtype/模式。它不参与本轮q0、方向基、参数选择和预测生成；真实分支后测量。
旧phase A已用monitor进行开发校准，因此称“独立于本次评分计算的监测读出”，不称最终独立测试。
V_final禁止任何模型评价、梯度和动作选择，仅允许已登记身份核对。
Phase A的ell只作为报告状态附注，不参与本次三动作评分、早停或更换状态；不重新估阈值。

### 2.3 本轮唯一模型目标

两个预测模型分别产生虚拟轨迹 theta_hat[k;u]。对H∈{1,4,16,64}：
    J_hat_M(H,u) = q0^T (theta_hat_M[H;u] - theta_hat_M[H;0])
数值越小越好。原生项严格J_hat(H,0)=0。
    selected_u_M(H) = argmin_u J_hat_M(H,u)
平局按 |u|更小优先，再按固定动作顺序{0,-.05,+.05}；浮点相等容差1e-12 nats只作数值平局，不冒充统计置信区间。

本公式是“冻结起点开发目标梯度的一阶终点差评分”。它不含目标Hessian，不等于精确终点NLL；不把q0∑s的代数恒等当作读出近似已验证。

主要选择使用H4；同时预注册H64选择，报告H4所选动作在H16/H64的效果，以及H64预测的真实表现。不能看结果后改主时域。

主预测行数=6 states×2 models×3 actions×4 H=144，含48个native零行；去掉native后的非零候选行是96。选择行=6×2×4=48。

## 3. 唯一低成本训练场近似：P2 + rank≤4 frozen reference field

### 3.1 参考梯度

P=2明确指两个microbatches，合成一个与真实accum=2一致的effective-batch梯度，不是两个独立effective batches。
在theta0固定SIM两批及对应train-mode随机掩码，按真实loss reduction/accum recipe计算原始未裁剪参考梯度：
    g0 = G_SIM(theta0)

参考梯度、FD-HVP用FP32参数与FP32参考forward/backward（关闭autocast）；不要改变真实训练的BF16 autocast。明确这是参考数值模型与实际执行的近似差别，而不是逐比特同一函数。
同一SIM随机性在所有FD正/负端点和所有方向中恢复；不使用真实evaluation stream的future gradients，不移动真实游标。

重要限定：本模型把同一个经验梯度场冻结使用至H64；不是模拟了64个独立新batch，不是声称已估计真实条件期望，也不是完整随机训练系统。两种模型采用相同这个限制。不能将虚拟的64步写成真实训练tokens或64批新数据。

### 3.2 方向与FD-HVP

为避免Q=4只是名字、没有实际映射，本轮固定如下方向规则，结果前写入config。

先用g0经过真实clip与当前optimizer状态，在独立克隆上只执行一次optimizer算术，得到native参考位移s_ref0；不计算新网络梯度、不写回真实状态。这计入virtual_optimizer_steps，不计作真实训练更新。

首方向 v1=s_ref0/||s_ref0||。若s_ref0为零，使用g0/||g0||；两者都零则rank=0并标NO_REFERENCE_DIRECTION，两个模型退化一致，不换case。

采用最多4阶的Krylov/Arnoldi式方向：
1. 已得正交单位方向vj，计算yj≈H0 vj。
2. 以两次modified Gram-Schmidt将yj对当前V正交化，非零残量归一化作为下一个方向。
3. 若残量norm <=1e-6*||yj||或yj为0，则停止扩秩；不随机补方向、不增加Q。

FD步长：
    delta_fd = 1e-3 * max(||theta0||_2,1.0)
单位方向下：
    yj = [G_SIM(theta0+delta_fd*vj)-G_SIM(theta0-delta_fd*vj)]/(2*delta_fd)
参数空间norm针对唯一参与本配方更新的参数，FP32点运算，norm/dot可用FP64归约。记录参数总norm、delta、每块相对扰动，不声称该步长已被旧实验验证。

每状态第一方向另算delta/2作为数值/非线性一致性诊断；主算子仍使用delta结果。
    fd_rel = ||y_delta-y_half|| / max(||y_delta||,||y_half||)
    fd_cos = cosine(y_delta,y_half)
若一边零另一边非零、出现非有限值，或fd_rel>0.20，则标LOCAL_OPERATOR_NOT_CALIBRATED。不能把这种结果改成训练效果失败。
此时停止GPU主流程并保存已有数据；不调epsilon到“能过”为止。若二者都是精确零，标ZERO_FD_SIGNAL，无cos，不制造非零信号。
数值容差是本轮新冻结的工程检查，不是物理规律。

rank<Q的真实构建费用按实际记录。完整Q4每状态基础2 + FD(4方向×2端点×2micro)=18 gradient forwards；第一方向half-check另4。

### 3.3 两个模型的唯一区别

令V=[v1..vr]，Y=[y1..yr]，theta0为所有动作共同参考点：

MODEL_NO_GEOMETRY：
    g_hat(theta)=g0

MODEL_GTTR：
    g_hat(theta)=g0 + Y V^T(theta-theta0)

两者都在每一虚拟步：
- 对原始g_hat重新执行真实global clipping；
- 按真实raw m/v公式更新，再执行实际偏差校正；
- 用完整原坐标P和解耦weight decay形成位移；
- 使用同一真实累计步号、tail schedule和动作长度；
- 不把投影梯度平方当作原v的投影；
- 不把v、P冻结，也不把grad=None改成zero。

如果当前dense配方存在无法通过同一参数活跃集合复现的None/skip语义，报告具体问题后停止，不悄悄简化。

虚拟传播无网络forward，但有全维点运算、状态复制、显存和墙钟成本。两模型×3动作×64步×6状态=2304个主virtual optimizer steps，另记录建模初始算术和smoke，不报成本为0。

每H记录：||theta_hat-theta0||、相对于V的投影残差比例、clip比例、m/v/update norm及finite标志。不要用观察到的真实未来state修正虚拟state，不拟合事后校准系数。
虚拟轨迹非有限时停止该虚拟分支，保留此前H的预测；后续H标INVALID，不写虚构数值或删case。主实验仍报告无预测覆盖率，默认无法选动作时选择NATIVE并标FALLBACK，不将其计入“有效模型准确率”。

## 4. 信息隔离、预测封存和真实分支

### 4.1 顺序

1. 固定配置、state manifest、SIM/real/data IDs、目标与模型公式。
2. CPU测试、模型数值smoke、真实native off/on smoke。
3. 对全部6个主状态构建两个评分模型，并保存全部144预测行、48选择行及中间数值/输入来源。
4. 封存预测文件及一个简短摘要hash，order_index写明；全部主预测完成后才能开始任何MAIN真实分支。
5. 运行每状态三条真实分支，结果独立写observed文件。预测文件只读。

smoke使用与主真实evaluation streams不同的固定测试stream；只检查实现/RNG和数值，不用其loss调科学超参数。若修复代码改变模型，必须在MAIN前重新生成一致版本的主预测，不混版本。
SIM/model构建不能读真实MAIN分支未来batch、monitor loss或final结果。

### 4.2 真值评价

每状态共同h0评价一次，3个真实动作分别在H={1,4,16,64}评价。
V_control_ref64用于同目标模型误差，V_monitor用于选择效用迁移。都保存逐行NLL_sum/count。
主整库评价次数：6+6×3×4=78次/target。
不设第四条真实native，不按H另开训练分支。

定义真实同轴差：
    J_obs_ctrl(H,u)=V_control(theta_H^u)-V_control(theta_H^0)
    gain_monitor(H,u)=V_monitor(theta_H^0)-V_monitor(theta_H^u)
两个量的符号定义在CSV字典里固定。不能用monitor与control相减称预测误差。

在每个真实H还计算：
    linear_obs(H,u)=q0^T(theta_H^u-theta_H^0)
用每臂q0^T(theta_H^u-theta0)的标量再相减即可，不必保存全部真实高维分支端点。
于是对每模型：
    e_total = J_hat - J_obs_ctrl
    e_propagation = J_hat - linear_obs
    e_readout = linear_obs - J_obs_ctrl
核对e_total=e_propagation+e_readout。
这可区分虚拟传播问题与冻结q0的一阶读出问题；只是事后诊断，不回填预测。

## 5. 最少实现测试与smoke

复用已有frozen测试和runner，只补与本轮有关的内容。不重写框架。
建议CPU测试：
T1 exp(u)语义，0..3步干预、k>=4释放；状态不重置。
T2 真实AdamW reference算术与克隆一步一致，clip binding/nonbinding均覆盖。
T3 小二次函数上FD-HVP方向与解析结果相符；随机性恢复错误的负测试能失败。
T4 V正交、HU映射形状、rank-break正确，零方向不造信号。
T5 在theta0两模型使用完全相同g0，所以H1的虚拟状态和同目标评分相同；否则不得运行main。
T6 两模型的u0相对目标=0；真值native相对目标=0；本地顺序调换不改变结果。
T7 scalar error decomposition、sum/count和完整动作网格聚合能独立复算，不用被测helper自比自。
T8 故意试图提前读observed MAIN或final会被显式代码结构/路径断言拒绝；无需大型安全框架。

真实smoke：第一个bank状态，使用独立固定smoke stream，native8步off与8步on，共16真实updates。
on包含模型/评价等诊断，按h0/4/8评价两目标；最终theta/m/v/step/RNG/cursor应与off一致。复用已有bitwise语义；标量一致不能替代完整恢复检查。
模型数值smoke是一个state的完整P2/Q4/half-check和q0；成本另计。完整主构建可安全复用相同版本的缓存，但要记录是否复用，预算下列按不复用保守规划。
如有真实代码bug，停止修复，有限重试计费；不得因科学预测不准调参“修复”。

## 6. 预算：新一轮分账，不修改历史

预算按rank=4、每个effective batch=2 microbatches、control64行/batch8、monitor4096行/batch32规划。

|项目|计划|
|---|---:|
|MAIN真实updates：6×3×64|1152|
|smoke真实updates：8+8|16|
|真实updates合计|1168|
|真实训练forward/backward micro-calls|2336 / 2336|
|MAIN模型梯度构建：6×(2+16+4+8)|180 fwd +180 bwd|
|一个smoke模型构建：2+16+4+8|30 fwd +30 bwd|
|模型构建合计|210 fwd +210 bwd|
|MAIN monitor：78×128|9984 fwd|
|MAIN control：78×8|624 fwd|
|smoke monitor：3×128|384 fwd|
|smoke control：3×8|24 fwd|
|评价forward合计|11016|
|全部计划forward|13562|
|全部计划backward micro-calls|2546|
|FD方向估计（含每state第一方向half-check）|35 =7×(4+1)|
|主虚拟optimizer步|2304（smoke另384，初始化算术另计）|

硬上限同时生效：
- 真实optimizer updates<=1280；
- 全部model forward<=14500；
- 全部backward micro-calls<=2920；
- 模型构建gradient forward/backward各<=360；
- FD方向估计<=50（只用于有限失败恢复，不改变科学P/Q/epsilon）；
- monitor整库eval<=84，control整库eval<=90；
- 所有GPU任务墙钟<=5400s（90分钟，包括构建、虚拟、真值、eval和恢复）。
V_final=0。

各类cap同时检查，不能因为某个子cap有余额就超过round-total。未来由rank提前结束/缓存复用节约的费用不能自动拿去扩样本。
一次普通main更新有2个梯度micro-call，不能报“总gradient=0”。
每次FD-HVP包含2端点×2micro梯度，不计成0梯度/0forward。
模型内部64步算术不是0成本，单列时间、峰值显存、参数存储量；SIM的两批数据暴露与真实训练tokens分别记录。
主真实input slots=1152×1024=1,179,648；含smoke为1,196,032。valid loss tokens按mask实算，不能以slots替代。
旧18000超限账、独立C1/C1L、PhaseA账均只引用，不重置、不覆盖、不把本轮并入旧cap。

smoke后给实测model/real/eval单位耗时和ETA；预计无法满足任一cap时保存结果停止，不能减评价数据、更换目标或缩短horizon却继续称同一完整实验。

## 7. 统计读数与判定：不把有益干预等同自适应有益

### 7.1 数据级不确定性

control用固定8行block（64行共8块），monitor沿PhaseA固定64行block（4096行共64块）；1000次seeded paired block bootstrap，各动作同次重采样同一block集合，按sum/count聚合。
标签BLOCK_PROXY，无doc ID时不声称文档级独立。数据级SE不代替future-stream或训练seed不确定性。
各动作对差D_ab的可分辨标记：abs(D_ab)>3*paired_SE_ab；所有未分辨结果仍展示，不删点。
模型本轮只有一组SIM effective batch，不假装有可校准的模型预测SE。

### 7.2 模型误差

对每state/H/model与u=±.05：保存pred、truth、error、error²、truth²。
固定全state全action集合计算SSE/TSS；TSS=0时标TARGET_ZERO，不通过添加epsilon制造E。
计算E=sqrt(SSE/TSS)，无几何模型与GTTR按相同集合比较。
另报告每state与每seed；多个H及两个起点不能伪装成独立seed。
H1两模型应相同，是一致性校验，不是几何带来增益的证据。

### 7.3 决策

对每state/H、三个动作实际值，给出真实最优、第二优、间隔与其SE；未分辨时标NO_RESOLVED_WINNER，而不是硬分对错。
每模型预存的selected_u4与selected_u64，在所有H报告：
- control的真实损失差/相对网格最优regret；
- monitor的gain_vs_native、gain_vs_fixed_up、gain_vs_fixed_down；
- 与另一模型的paired selection差。
实际网格最优和“在全部6个状态上事后选择的最佳固定动作”都标HINDSIGHT，不能称真实在线策略。
不调更广固定LR网格，本轮只作有限动作价值检验；以后算法比较仍需强固定基线。

### 7.4 科学判定分开，允许不同层给不同结果

至少输出：
IMPLEMENTATION_GATE；MODEL_BUILD_NUMERICS；PREDICTION_SEAL；
MODEL_COVERAGE_H4/H64；SAME_TARGET_MODEL_ERROR；
GEOMETRY_INCREMENTAL_INFORMATION；MONITOR_SELECTION_TRANSFER；
HORIZON_PERSISTENCE；MODEL_COST；FUTURE_POLICY_VALUE/ENERGY/FINAL_GENERALIZATION。

本轮不强制制造一个总体科学PASS：
- 数据不足以分辨动作→ACTION_VALUE_UNRESOLVED；
- 真实赢家全为同一固定动作→NO_OBSERVED_ADAPTIVE_HEADROOM，不扩大网格制造异质性；
- H4预测有益但H64反转→SHORT_HORIZON_ONLY，不外推最终收益；
- GTTR不优于M0→NO_OBSERVED_GEOMETRY_INCREMENT，保留原始数据；
- 虚拟数值失效→报告覆盖不足，不剔除后只报成功状态；
- 相同目标预测差、monitor迁移或真正策略收益分别判断。
不从本pilot宣称普遍不可能性，也不因局部平均改善就宣称最终算法有效。

累计达标tokens、实际adaptive训练价值、最终能力和能耗仍NOT_TESTED。ell只用于说明真实状态与原目标的关系。

## 8. 必须保存并在最终聊天展示的关键数据

不得只返回PASS/FAIL和输出路径。至少交付下列CSV及数据字典。

1. manifest.csv：state/seed/checkpoint/n0/gm0、真实/SIM流、目标ID、参数组与dtype、schedule、tokens口径。
2. model_construction.csv：每state的P/Q/rank、delta、v/y norm、FD half-check、orthogonality、投影残差、模型开销、grad source。
3. predictions.csv：完整144行；state/model/action/H/J_hat、selected_u、finite、seal_order。
4. observed_row_stats.csv：state/action/H/target/row_id/nll_sum/valid_count；future真值单独文件。
5. observed_actions.csv：逐state/action/H同轴NLL、native/fixed差、paired SE、实际tokens和wall。
6. prediction_error.csv：control pred/truth、error/SSE/TSS，以及linear_obs、e_propagation、e_readout。
7. decision_pairs.csv：3个动作对的pred_D/truth_D/SE/resolved/rank；不省未分辨。
8. choices_and_regret.csv：H4/H64预选动作在H1/4/16/64上的实际control regret、monitor收益与固定臂比较。
9. aggregate_inputs.csv：逐state/seed/H的SSE/TSS及固定集合，参考脚本不复用生产聚合helper。
10. resource_log.csv + summary.json：planned/actual/cap、真实训练/模型构建/虚拟/评价/失败各项。
11. theory_tests.json、smoke_checks.json、REPORT.md、KEY_DATA_FOR_REVIEW.md、config.json。

最终回复直接展示：
A. 6状态真实配置与全部数据角色。
B. 6状态×2模型在H4/H64的predicted actions，以及3候选预测值。
C. 6状态的真实三动作H4/H64 control差和monitor gain±SE，H16反转案例同列。
D. H1/4/16/64的GTTR/M0固定集合SSE、TSS、E和覆盖率。
E. H4/H64选择相对fixed-up/down的真实效用，标明data SE/只有3seed。
F. e_total/e_propagation/e_readout，至少逐state H4/H64；不让一个score误差掩盖两个失效来源。
G. 所有实际成本，模型时间与真实候选重放时间、峰值显存；不能只说“虚拟0 forward”。

最多5张图，从CSV重生成；不新增代理指标或事后平滑改变判定。

## 9. 实现风格与退出规则

优先复用现有frozen AdamW/clip/CRN/eval接口。可新建research/src/fr/phase_b_gttr_score_v03.py、research/scripts/phase_b_gttr_v03.py和一份reference test；路径按实际项目最小适配，不发明不存在的旧函数接口。
无需新依赖、服务、数据库、巨型hash manifest、全仓审计tar或常驻tail监视器。
不修改旧结果；新目录建议outputs/fr_phase_b_gttr_action_scoring_v03/。不commit/push，不改Kimi配置，不干扰其他训练进程。
读取最近PhaseA修订及v2草案足够，不因为A0F2缺失而重开历史审计。

执行结束一次性给正式报告。若具体实现/数值/预算门失败，则给出已存原始值和消耗，停止；不自动跳到Phase C或偷偷改成只做真实干预图。
