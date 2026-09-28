# FR GTTR-R2B：既有目标解到真实续训后果的桥接实验 v1

日期：2026-09-24。执行端：OpenCode＋GLM5.3。

## 0. 研究主线与本轮授权

本文件是下一阶段实验设计，不是已完成的实验结果。用户将本文件与启动指令转交执行端后，仅授权本轮预检、CPU测试、smoke、真实配对续训和报告；不授权R3、重新辨识、扩大控制域或训练新控制器。

唯一主线：GTTR真实更新关系 → 真实NLL目标在控制坐标中的局部表达 → 已封存的受限求解结果 → 真实训练后果与检测达标tokens。

R1中的 m(x,y) 只是目标变化的局部二次表达，不是新增物理模型、状态模型或预测网络。在论文说明中可写作 Jhat_GTTR(x,y|z)，现有代码字段m无需重命名或重构。本轮不增加任何新目标代理。

### 已有证据的准确位置

按用户提供的R1与0-GPU补充：主半径受限解11/12的y在上界，1/12的y内部但x仍在边界；同轴收益10/12区间排除0；check256为12/12均值正、8/12区间排除0；二阶相对一阶只在一个条件改变动作且实际增值未确认。均为受限局部、开发期证据，不是最终训练收益。

保留原PhaseB/S3的GATE_STOP和D1的NORM负结果。本轮不修冻结场预测器，不给旧实验回写PASS，不新增温度、阶段标签、记忆核或经验alpha函数。

## 1. 本轮只回答一个问题

将R1已经在FIT上求出并封存的动作，真实执行一次；此后所有臂回到完全相同定义的基线训练规则。

问：相对于R1的LR-only解，完整GTTR受限解的局部收益能否在真实后续训练中保留，是否影响规定目标下的检测达标tokens？

这是离线因果/有效性桥接，不是上线新优化器：
- 不逐步重新求解；
- 不将起点解固定使用256步；
- 不因为一次改动有益就宣称重复使用一定有益；
- 不从结果较好的状态挑下一轮“代表案例”；
- 没有检测token差异时可能是量化分辨率不足，不自动等于局部NLL结论无效。

## 2. 动作来源：只使用R1真实已封存的主解

从原R1的 restricted_solutions.csv、model_coefficients.csv、prediction_seal.json 和后续补充读入主半径 r=0.05 的原始全精度解。

对状态z和参考倍率k：

    a_LR(z,k) = (x_LR, 0)
    a_GTTR(z,k) = (x_2D, y_2D)

它们来源于同一个任务目标：

    J_z,k(a) = E[V_ctrl(theta_plus + D*a)-V_ctrl(theta_plus)|z,k]
    Jhat_GTTR(a) = b^T*a + 0.5*a^T*K*a

    a_LR = argmin Jhat_GTTR(x,0)
    a_GTTR = argmin Jhat_GTTR(x,y), (x,y) in [-.05,.05]×[0,.05]

本轮不根据AUDIT的CI决定是否保留某个状态，不把y取0.05重新当成全局最优系数。包括2031/46未分辨状态与其内部y解，原样进入。

不得从截图手抄或四舍五入alpha。不得改半径、PSD截断、换成Schur无约束解或用一阶伴随结果替代主解。读取已封存系数重算仅用于CPU一致性检查，不构成新拟合。任何差异先报告，不自行择优。

## 3. 一次真实更新的精确实现

在每臂的第一个当前batch上，使用真实累计梯度及clip后c，正常更新raw moments和偏差校正：

    m_plus = beta1*m_old + (1-beta1)*c
    v_plus = beta2*v_old + (1-beta2)*c^2
    mbar = correct(m_plus, true_parameter_step)
    vbar = correct(v_plus, true_parameter_step)
    P = 1/(sqrt(vbar)+eps)

对该步的参考eta（见下一节），定义：

    d0 = -eta*(P*mbar + wd*theta)
    d1 =  eta*P*(mbar-c)

当前动作对应：

    eta_new   = eta*(1+x)
    alpha_new = y/(1+x)
    theta_new = (1-eta_new*wd)*theta
                -eta_new*P*((1-alpha_new)*mbar + alpha_new*c)

应与 theta+d0+x*d0+y*d1 在规定参考容差内吻合。

本轮真实臂只有三个：

|arm|第一个optimizer更新|之后第2至第256次更新|
|---|---|---|
|BASE_K2|a=(0,0)|原AdamW＋参考k=2的共同schedule|
|GTTR_LR_ONLY|原a_LR(z,2)|同上|
|GTTR_FULL|原a_GTTR(z,2)|同上|

干预长度严格是一更新，不是四步pulse，也不是整段固定读出。

真实实现优先采用原代码的原生更新顺序，只替换当前一步的eta/读出。没有范数匹配；不回写r_Q到moments；不冻结v/P。共享参数只更新一次，参数组/decay mask/grad=None语义不变。

三个臂给定同一初态和首batch时，首步raw m/v与计数应相同，theta一般不同；第二步以后梯度和moments可以分叉，不能再要求跨臂m/v相同。恢复基线是恢复更新规则，不是重置参数、moments或采样位置。

## 4. 范围：强参考配方、全部原状态、两条新流

### 4.1 原状态

- seeds 2030、2031、2032 × 原bank start46/82，共6个状态，S0–S5按(seed,start)排列。
- 真实n0、参数组计数及global_micro调用点从checkpoint和原R1执行记录核对。
- 不混入h128/h256快照，不以R1收益大小筛选。

### 4.2 本轮只采用参考k=2

理由是D1中所测固定LR2.0较强，避免只在较弱参考配方附近证明收益；它仍不是充分调优的最优AdamW。

这是事前明确的资源范围选择，不是选取R1较好的条件。全部6个状态的k2都进入，包括未分辨状态。本轮不做k1，不将本轮结果合并成覆盖全部12个R1条件。

schedule：

    eta_base(gm) = old_get_lr(gm,256,5e-4), gm<=256
                   5e-5,                  gm>256
    eta_ref(gm) = 2*eta_base(gm)

第一个更新临时再乘(1+x_arm)。之后立刻恢复eta_ref，不重warmup、不重设gm、不重新拉伸total、不增加cooldown。末端参考eta=1e-4。

R1报告的首步eta来自其真实gm调用位置（例如gm94/166），不要从start标签自行重新假定gm92/164。必须输出实际n_before/n_after/gm_used/eta_ref，与相应R1原件核对。如果真实语义冲突，只报告这一具体阻塞；不改旧数据，不重开历史大审计。

### 4.3 配对新流

每个state两条新future streams，每条stream三臂完全配对。

提议种子块202609240200至202609240211，S_i使用base+2i与base+2i+1。元数据检查与所有已执行记录不冲突；冲突则整个块+1000，结果前冻结，不按loss选择。

不复用R1 FIT/AUDIT作为此次真实首batch；仍从原冻结train support抽样。记录实际sample IDs/masks，不能将新的seed称为文档独立。

总共：6 state × 2 streams × 3 arms = 36条主轨迹。

三个训练seeds仍是三个训练seeds；不是36个独立模型，也不是12个独立训练seeds。两个新stream提供有限的路径重复，不宣称已精确识别未来随机分布。

## 5. 目标、时域与监测规则

### 5.1 真实训练长度

每条主轨迹完整运行256次optimizer更新，不因提前达到阈值而停止，不根据后续好坏延长。

该上限复用Phase A/D1的开发窗口，具有已观察到native进展和目标可达性的依据；但本轮k2、新流下是否达标仍需要实测，不保证。256步不称全预训练或最终渐近行为。

### 5.2 目标角色

- V_ctrl：R1原64行FP32 reference，eval mode、sum/count、相同token/mask与后端语义。只在h0与h1读取，不求梯度。用于检查R1求解结果进入真实训练后的即时同轴效用。
- V_monitor：Phase A/D1相同4096行全库，原dtype/mode、sum/count。用于真实轨迹和检测目标。
- V_check256：本轮不另测，避免重复扩大评价。
- V_final：仅保留登记身份，0模型调用。

旧开发目标没有变成新的独立final；本轮新的是训练随机流，不是未被研究使用的任务目标。无doc ID限制继续声明。

### 5.3 评价位置

V_monitor：h=0,1,4，以及h=16,32,...,256。

其中h0每个state可以跨两个stream和三个臂复用一次，前提是真正相同theta和评价语义。其余每条真实轨迹独立读取。

V_ctrl：h0每state共享一次，h1每条轨迹各测一次。

h1对应本次受限解实际执行后。h4仅作早期后效应读数，不是干预持续到第4步。

### 5.4 主要比较与方向

primary对照：GTTR_FULL vs GTTR_LR_ONLY。

定义同状态、同stream、同h的结构收益：

    G_struct(h) = V_monitor(theta_LR(h)) - V_monitor(theta_FULL(h))

>0表示完整解优于LR-only。

伴随：BASE_K2 vs LR-only、BASE_K2 vs FULL。不只展示相对BASE的收益。

本轮主质量终点：G_struct(256)。h1、h4、h16、h64、h128作为预定的机制/持续性伴随，不事后挑最有利时域作为主结果。保存所有监测点。

本轮主要成本读数：下述检测达标input tokens与未达标分布。质量和检测成本可得不同结论，不能合成任意分数。

### 5.5 固定阈值与离散检测

按start读取PhaseA原全精度阈值，不手抄显示值：

    start46: about 7.020043501537879
    start82: about 6.979538126804536

两个起点是不同阈值的续训任务，分别报告，不平均成共同绝对目标。初始monitor不高于其目标时登记START_TARGET_MISMATCH，核对身份；不能替换更困难阈值、起点或seed。

正式检测网格只用h=0,16,32,...,256；额外h1/h4不参与检测，以免改变各阶段比较口径。

连续两次NLL<=ell的第二时点记h_detect。保存该confirmed pair、此前above点以及完整曲线。h_detect×每步slots只在实际计数支持时换算；valid loss tokens按mask真实累计。

所有轨迹都运行到256。未达标记NOT_REACHED，N_detect=NA，预算上限单列；不能删除。两臂都达标的paired ratio可报告，但不得作为全样本无偏成本估计。数值失败和外部中断另分。

单次局部干预可能只对应不到16步的目标变化。若两臂检测时点相同，报告NO_DETECTED_TOKEN_DIFFERENCE_AT_16_STEP_GRID；不宣称数学首达成本严格相等、不改变阈值或监测间隔“找出收益”。

V_final未确认，因此只称开发monitor的检测token结果，不称独立最终质量保证。

## 6. 统计与允许的结论

1. 逐state/stream/arm输出原始NLL sum/count和所有差值。
2. monitor数据级不确定性复用冻结64连续行BLOCK_PROXY paired bootstrap；它不是训练seed置信度，不假设无doc ID数据满足文档独立。
3. 分开给每个seed、start、stream结果；可给seed内两stream平均，再给跨seed描述，不能将36轨迹当36独立模型。
4. h1的R1预测是对原probe分布的条件均值，不要求新两个stream分别命中预测。可以报告新流效果对原FIT预期的关系，但不得用两个样本造精确分布结论。
5. 主终点必须展示效应量和不确定性，避免仅给“10/12通过”的成功率。小样本未分辨不包装成无效。
6. 原R1中的10/12、8/12作为既有结果，不用作新stream筛选条件。

建议分别输出：
- IMPLEMENTATION_AND_COVERAGE
- ONE_STEP_EFFECT_ON_NEW_STREAMS
- TERMINAL_NLL_VS_LR_ONLY
- EFFECT_PERSISTENCE_OR_REVERSAL
- DETECTED_TARGET_TOKENS（含grid与censor限定）
- DEPLOYABLE_OPTIMIZER_BENEFIT = NOT_TESTED
- SECOND_ORDER_EXTRA_VALUE = NOT_TESTED_THIS_ROUND
- V_FINAL_CONFIRMED = NOT_TESTED

如果h1正、后来消失/反转：说明局部求解结果的持续效果不足，本轮不能直接推成反复训练规则。不要据此否定所有GTTR。

如果同token末端正但检测token无差：只报告固定预算质量信号或检测分辨率限制，不宣布成本节约。

如果均无可靠收益：按本干预/范围保留负结果或未分辨，不临时将一次改为每4/16步施加。

如果有收益：只支持一次model-solved intervention＋共同基线的后果；周期求解、廉价闭合和完整训练仍须独立验证。

## 7. GTTR目标与本轮成本桥的数学含义

最终研究目标仍是达到固定任务质量的累计成本，不新增一个目标代理。

在可达、可积、共同后续基线pi0等条件下，当前更新后的两状态z_PLUS、z_LR的差决定：

    E[C_FULL - C_LR]
      = E[J_pi0(z_PLUS) - J_pi0(z_LR)]

本式指首步相同成本抵消后、后续均执行pi0的无限时域理想成本。当前有限预算和离散monitor只能测对应受限读数，不能将未达标末端剩余值删成0来宣称估计了完整J_pi0。

本轮真实续训就是检验这段联系，不在线训练价值网络、不模拟候选未来后选动作。模型解已经从原FIT封存；所有后续实际结果仅用于评价。

R1建模的总历史成本单独引用costs.json或现有报告，不重复混进本轮run-level账，也不能称建模信息免费。若未来讨论部署净效率，需说明模型辨识成本如何支付/摊销。此次是离线桥接实验，没有已证明的净计算节约。

## 8. 实现范围与最少测试

优先复用现有真实runner、R1控制坐标和目标评价，新增一个小wrapper/脚本及CPU分析；不重构平台、不创建新服务或新数据系统。

预检只需读：R1原系数/解/封存及补充、实际bank恢复接口、修正后的PhaseA schedule/data/target、当前训练runner。原D1只用于确认已有计数/评价接口，不重审全部A0/B0账本。

CPU/已有tensor测试至少包含：
- (x,y)到(eta',alpha')的精确映射，含wd与参数组；错误漏wd应被检测。
- a=(0,0)走native；LR-only确实y=0；不得加入norm matching。
- 更新mask仅第一个optimizer update有效，第2步及以后固定原规则。
- 相同外生首梯度下首步raw moments和计数一致；theta分叉后不得强求跨臂moment一致。
- 读取primary受限解，不能读取无约束Schur或half解；用独立小参考计算核对两个特殊边界/内部例子，不重拟合。
- schedule累计索引正确、不回升，倍率2尾部固定1e-4。
- 监测h1/h4不参与正式target检测；等号<=、未达标/失败、input与valid计数正确。
- 数据/RNG与评价隔离，唯一参数与grad=None语义沿真实实现。

无需HVP门、FD门或新数学诊断。近似系数已来自原R1，不在本轮更改。

### Smoke：40真实updates

固定seed2030/start46、独立smoke流202609240300（冲突按整块元数据顺延）：
- BASE_K2诊断off/on各8步；
- GTTR_FULL诊断off/on各8步；
- GTTR_LR_ONLY 8步。

monitor整库6次：公共h0一次，加五条h8。smoke不新增V_ctrl评价。off/on比的是同一规则自身的参数/m/v/step/RNG/游标，不要求候选与base相同。

只检查语义、有限性和隔离，不按smoke loss调整动作或终止科学比较。若代码错误须记录失败消耗，修代码后重新smoke，总预算不重置。

## 9. 新轮预算：共享GPU只用计算量硬上限

按microbatch4、accum2、输入context128、monitor4096/evalbatch32、ctrl64/evalbatch8。

|项目|计划|硬上限/说明|
|---|---:|---:|
|主真实更新|6×2×3×256 = 9216|计入总量|
|smoke真实更新|40|计入总量|
|总真实updates|9256|9600|
|训练forward|18512|随实际更新和失败计|
|训练backward microcalls|18512|19200|
|主monitor整库eval|6+36×18 = 654|计入总量|
|smoke monitor eval|6|计入总量|
|monitor整库eval总数|660|700|
|monitor forward|660×128 = 84480|89600|
|V_ctrl整库eval|6+36 = 42|60|
|V_ctrl forward|42×8 = 336|480|
|全部网络forward|18512+84480+336 = 103328|110000|
|新增q/HVP/probe gradients|0/0/0|0/0/0|
|V_final模型调用|0|0|
|主训练input slots|9216×1024 = 9437184|actual另算valid/nonpad|
|含smoke训练input slots|9256×1024 = 9478144|按实际计|

“新增probe gradients=0”不等于真实训练backward=0。没有用于当前选择的候选重放；三个arm是实验对照，其真实训练成本完整计入。

以上是新独立账本，不借旧余量。所有cap同时约束，实际监测分批不同则先按真实实现核算，不暗改batch或目标。失败/恢复/重复评价全部计费。wall只记录OBSERVATIONAL_SHARED_GPU，无90分钟科学硬门；不因外部资源竞争更慢删样本或判模型失效。

数值NaN/Inf终止该轨迹并保留失败；外部OOM/中断在完整可恢复状态下续跑，否则PARTIAL；不降低batch、更改精度或重挑随机流救结果。不干扰其他进程。

保存每条主轨迹h1与h256完整状态或等价可恢复制品，共72个（tied参数不重复），用于以后追查而非本轮重跑。不要每步存高维张量。若存储限制无法承担，GPU main前报告具体大小，可改为保留h1完整状态和h256必要结果，但不改变数值实验设计；不可伪造已保存快照。

## 10. 输出必须直接体现主线，而不是再创造模型名字

建议目录 outputs/fr_gttr_r2b_solved_update_bridge_v01/。

最低制品：
- config.json / state_stream_action_manifest.json（含R1原解来源和版本）。
- implementation_checks.json / smoke_checks.json。
- actual_schedule.csv（gm、计数、base/reference/applied eta、alpha、intervention_flag）。
- monitor_metrics.csv / control_step1.csv / per_row_stats（可压缩，保存配对统计所需数据）。
- gains_vs_horizon.csv / target_detection.csv / token_counts.csv。
- resource_log.jsonl / snapshot_manifest.json。
- REPORT.md / KEY_DATA_FOR_REVIEW.md，最多4张图。

最终聊天至少直接展示：
A. 六state的原R1主解、两新stream、真实n0/gm、首步动作与第2步恢复规则。
B. 逐state/stream的h1 control差与monitor差（FULL−LRonly、各自−BASE）。
C. h4/16/64/128/256的结构gain、原始sum/count、数据级SE与逐seed/stream差异。
D. 两目标组分别的first-confirmed-pair、second-confirmation、input/valid tokens、未达标/失败，以及配对eligible数。
E. 实验总成本与历史R1辨识成本分开；真实/辅助tokens和共享环境wall分别列。
F. 一句准确结论：一次受限GTTR目标解是否具有即时/持续/检测成本收益；哪些部分不确定。

不得：
- 将m(a)宣称为又一套独立动力学理论；
- 将一次改动后的收益乘以256外推持续规则；
- 把r=.05的边界解称为全局内生最优幅度；
- 从本轮结果宣称二阶信息比一阶更有价值；
- 调用final、重新选目标或增加新alpha；
- 自动进入R3、线上重复求解或扩规模。

完成后一次性报告并停止。不要commit/push，不改OpenCode/GLM接入配置，不安装无关依赖，不建立常驻monitor。
