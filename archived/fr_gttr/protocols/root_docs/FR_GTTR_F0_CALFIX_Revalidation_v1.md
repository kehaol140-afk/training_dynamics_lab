# GTTR-F0：修复后完整终点导数复测（CAL-FIX v1）

**状态：建议的下一步协议，尚未运行。** 用户将本文件与启动指令交给 OpenCode＋GLM5.3 后，仅执行本轮修复后复测。不是自动重启 F0 全部 grid/solver/audit 的授权。

## 1. 已知事实及结论范围

依据用户提供的最新核验截图，而非本地独立执行远程代码：

- CAL-2 的 inline clipping JVP 使用逐参数块局部 dot，原 value 路径使用全局 dot；该敏感性实现有确证缺陷。
- CAL-1 使用原 a0c 全局 clipping JVP；完整源码版本不可恢复，旧梯度仍未验证。不能说 CAL-2 的缺陷已经解释 CAL-1 的全部失配。
- init_sensitivity 的浅拷贝别名影响了非零灵敏度的诊断调用。报告说明其未影响正式全零起步的主循环；不扩大归因。
- 修复后的双块 clipping 与完整一步 CPU JVP 已通过所报告的独立参考检查；这不等于256步已校准。
- 旧9个端点的绝对 NLL 未保存，只有4个对称差可恢复。不能反推出 J(0,0)、单边收益或最优规则。
- 当前状态为 UNVERIFIED_PENDING_FIXED_RERUN。保留旧记录，撤回以旧SENS为依据的“实现无缺陷”“混沌已证明”等陈述。

## 2. 固定不变的科学问题

目标保持 J_256(phi)=V_design(theta_256(phi))。

规则仍为原F0：

    psi[t] = 2*t/255 - 1
    Z(b) = sum(w[t]*exp(b*psi[t])) / sum(w[t])
    eta[t] = w[t]*exp(a+b*psi[t]) / Z(b)

原state、start46、seed2030、原校准流及全部训练/评价语义不变。旧截图流ID为202609250400，以原manifest核对为准，不换一个更易通过的流。

所有运行的 schedule 总时钟都是256；读取h16、h32不等于将schedule重设为16或32。

不改变网络、batch、clipping阈值、Adam epsilon、beta、weight decay、train support、mask/reduction或参考后端。不缩短终点，不增加FSL/ELR/DMFT/影子梯度，不改用截断灵敏度。

## 3. 修复版本与CPU检查

冻结当前修复版的相关源文件、config、运行入口：复制到本轮source_snapshot目录并生成SHA256清单。无需commit，但每个运行结果必须有run_id和source_hash，不能覆盖不同版本CSV。

复用并重跑已完成的有限CPU测试，不再重审旧项目：

1. 全局clipping JVP的dot_global = sum_p <g_p, Hs_p> 在唯一参数集合上一次求和，各参数块共用同一标量；inactive与active分支均与value对应。
2. 双块g=(3,4)、h=(4,-3)反例，独立公式/AD/FD一致；错误局部dot版本被测试拒绝。
3. 非零S_theta/S_m/S_v初始化的独立内层容器与存储；参数/方向/状态之间不得有意外可变别名。
4. 完整一步theta/m/v JVP，包含非零历史、两方向、直接d_eta和共享参数语义。
5. vbar零分支按已验证逻辑处理；不增加v_floor、epsilon或nan_to_num救结果。

精度名称必须真实：FP32值路径/FP32图内HVP＋FP64点积、结果存储和灵敏度递推，不称完整FP64训练或完整FP64 HVP。

若CPU失败，修复记录另存并停止GPU启动。GPU结果开始后不在同一run下继续改源码。

## 4. 本轮唯一运行矩阵

### 4.1 短smoke：8步＋8步

同一固定状态/流分别执行value-only和value＋SENS，共16次trajectory updates。比较theta/m/v/step/LR/RNG/data cursor；不读取V_confirm或V_final，不新增全库目标评价。

### 4.2 十条完整256步参考轨迹

|run角色|phi|条数|
|---|---|---:|
|普通value基线|(0,0)|1|
|修复SENS基线，维护a/b两个方向|(0,0)|1|
|a方向有限差分|(±.01,0)、(±.005,0)|4|
|b方向有限差分|(0,±.01)、(0,±.005)|4|

每条从同一个原bank完整恢复，使用同一规定随机流。不能把上一条试验终点作为下一条起点。

每条在h={16,32,256}记录同一V_design的全库sum/count/mean，共30次目标评价。SENS基线的三次目标评价同时取得q_H并计算dot(q_H,S_H[j])；不要重复前向只为分开“loss”和“gradient”。

在一个h256基线终点上再评价一次作为重复读数，共31次目标评价。

value-only与SENS基线在已规定重叠位置核对参数/moments/计数/RNG/游标。未保持同一value轨迹时，先记REFERENCE_PATH_MISMATCH，不能将后面的差异解释为纯导数问题。

旧FD差只作历史比较，本轮8条扰动轨迹全部重新执行并保存绝对目标值。因为旧端点绝对读数缺失，不据旧差分假装补齐新版本端点。

## 5. 保存哪些数据，避免再次只留下一个FD

- calibration_run_ledger.csv：run_id、源码/config/state/data hash、实际dtype/backend、phi、真实步号与scheduleID。
- terminal_values.csv：每run每h的sum/count/mean，BASE重复值独立行，禁止只留导数。
- sensitivity_check.csv：h×coord，SENS、两尺度FD、各绝对差、方向、原协议tol。
- fd_pair_consistency.csv：两尺度FD单独的一致性，不让大SENS放大FD内部容差。
- value_identity.csv：基线普通/SENS的重叠身份结果与精确检查对象。
- sensitivity_norms.csv：利用已计算量记录每步两方向的S_theta/S_m/S_v范数、clip active状态、有限值；它是诊断，不据此宣称Lyapunov指数或混沌。
- costs.json与append-only resource_log：smoke、main、失败/恢复、HVP/目标梯度和评价全部计费。
- 关键h256终点状态或不可变可追溯的保存引用；存储不可用时在运行前说明，不再事后编造未保存字段。

## 6. 判据：保持原主门，单列更清楚的误差

原F0门原样计算并报告，旧结果不回写：

    epsV = max(abs(repeated_V_difference), 1e-7)
    f = max(5e-4, 4*epsV/.005)
    tol_original = f + .20*max(abs(SENS),abs(FD_.01),abs(FD_.005))

另独立报告：

    fd_diff = abs(FD_.01-FD_.005)
    tol_FD = f + .20*max(abs(FD_.01),abs(FD_.005))
    sens_diff = abs(SENS-FD_.005)

独立FD诊断使用原20%比例和原绝对floor，目的只是避免巨大SENS导致FD内部一致性形同虚设；不追改原门，不放松阈值。

完整坐标确认要求：value身份成立；原门通过；独立FD对也通过；读数不是全部处于LOW_SIGNAL。a/b逐项报告，不用“其中b通过”冒充二维求解可以恢复。

h16/h32为同一T256轨迹的预注册伴随检查，必须有SENS和FD全列才能比较；符号随h改变不自动等于混沌或导数错误。

## 7. 结果后的下一步

- 两坐标h256均校准：报告REFERENCE_FULL_HORIZON_GRADIENT_CALIBRATED；它只说明此状态/流/规则邻域的导数可用，不是算法收益。提交恢复原F0 grid/solver的剩余预算清单，本轮先停止。
- 一坐标LOW_SIGNAL/不一致：报告PARTIAL_OR_UNRESOLVED，不临时删除某个坐标或扩大epsilon。
- 同路径value正确但全程SENS仍不一致：报告FULL_HORIZON_GRADIENT_UNRESOLVED_AFTER_FIX。说明已修缺陷未充分解释旧现象，但仍不宣称已证明混沌、非光滑或所有规则不可优化。
- value身份不一致：实现/参考路径问题优先，不用灵敏度放大解释。

本轮不运行grid/solver/audit/confirm；不将端点中某个较好值当作求得新规则。下一步是否恢复原求解器或单独开展无导数的规则空间对照，由本轮读数决定。

## 8. 计算预算

在每step两个microbatch、全库V_design4096行/batch32=128 forward、HVP共享当前一阶图的计数语义下：

- 10×256+16 = 2576 trajectory updates。
- training forward/backward各5152。
- full SENS最多2×255=510个effective-batch HVP方向；8步SENS smoke最多14，共524。
- 目标评价31×128=3968 forward，合计forward=9120。
- HVP/VJP microcall最多2×524=1048。
- 3个目标梯度×128=384 backward；合计backward/VJP=5152+1048+384=6584。

|计数|计划上界|本补丁上限|
|---|---:|---:|
|trajectory updates|2576|2816|
|network forward|9120|11000|
|backward/VJP microcall|6584|8000|
|effective-batch HVP方向|524|576|
|V_design整库评价（含3grad）|31|36|
|V_confirm / V_final|0|0|

如果现场实现HVP需要额外前向，先按真实方式重新计费并确保全部子cap均满足，不能把额外工作藏起来。不得为省费用改数据/精度/目标。

该补丁费用在F0账本中新增独立run段；历史4864等旧费用以当前账本复核，不清零、不覆盖。F0原总cap与本补丁cap同时约束，恢复后续阶段之前必须重新核对剩余额度。

墙钟仅OBSERVATIONAL_SHARED_GPU，不因资源竞争慢而改变科学条件。不停止其他任务，不建常驻监控。

## 9. 给OpenCode＋GLM5.3的执行要求

仅执行本文件的CPU复核、16步smoke和十条固定完整轨迹。少量必要脚本修改可以做，但不得顺手重构或恢复其他实验分支。

所有结果写入新的修复复测目录，旧CAL-1/CAL-2保留并链接。提交一次正式报告，直接展示h16/32/256的SENS与FD、原始目标端点、身份核验和预算，不只给PASS或文件路径。

无需commit/push；源码快照和hash足够，但两次运行不能共用一个会被覆盖的CSV。完成后停止。
