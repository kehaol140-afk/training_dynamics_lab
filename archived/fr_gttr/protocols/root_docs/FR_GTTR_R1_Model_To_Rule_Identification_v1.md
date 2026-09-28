# FR GTTR-R1：从目标系数辨识到规则排除的实验计划 v1

日期：2026-09-23。执行端：现有 Claude Code + Kimi K2.7。

## 0. 本轮定位与授权范围

本文件提出一项新的、受限的模型辨识实验，不是已经验证的新优化器，也不是恢复旧 Phase B。用户将本文件与启动指令交给执行端后，仅执行 R0（准备/CPU）和 R1（定点目标系数辨识与留出检验）；R2/R3 是后续路线，本轮不启动。

研究问题：在允许小范围学习率调整之后，当前梯度相对于历史 moment 的读出变化，是否仍有可辨识的目标收益？GTTR 能否从真实目标的方向导数与曲率，推导“应修改/不应修改/无法决定”的结论？

与 D1 的区别：不预先指定一个非零 alpha 当作训练规则；不把 native 范数设为应保留的约束；不从最终好坏倒推机制。先从真实更新导出控制坐标，识别目标系数，求解受限子问题，再用独立采样检验解。测量半径、样本数是实验配置，不宣称为理论最优超参数。

不恢复冻结梯度场、不预测 64 步参数、不改 beta/clip/batch/decay，不重做大审计，不自动扩展 RAW/NORM 或 LR 网格。D1 保留为经验消融；旧 GATE_STOP 和账本原样保留。

R1 不证明真实偏差/噪声各自的因果贡献，不宣称已经优化 tokens-to-target。它识别的是联合随机更新对任务的净作用。允许无收益、二阶模型不够局部、样本不足、模型成本不值得等结果。

## 1. 精确控制切片：先定义改什么，再从目标决定改多少

在同一恢复状态 z=(theta,m_old,v_old,n,...)、某一当前训练样本 omega 下，按实际配方计算累计梯度、global clipping、raw moments、bias correction：

    c = clip(g(omega))
    m_plus = beta1*m_old + (1-beta1)*c
    v_plus = beta2*v_old + (1-beta2)*c^2
    mbar = correct(m_plus, true_step)
    vbar = correct(v_plus, true_step)
    P = 1/(sqrt(vbar)+eps)

P 允许依赖当前 c。不同 omega 的 P 不能冻结为共同平均矩阵。相同样本的两个参考 LR 共用相同 m_plus/v_plus，因为本配方中它们不依赖本次 LR。

针对参考学习率 eta（含已声明 schedule 和参考倍率 k），定义：

    d0 = -eta*(P*mbar + wd*theta)        # native完整位移，含衰减
    d1 =  eta*P*(mbar-c)                # 当前梯度读出系数的位移方向
    theta_plus = theta + d0
    D = [d0, d1]

引入控制坐标 a=(x,y)：

    theta(a) = theta_plus + x*d0 + y*d1
    eta_new  = eta*(1+x)
    alpha_new = y/(1+x),  x>-1

这是下式的一次更新精确重参数化（在声明实数参考算术中）：

    theta_new=(1-eta_new*wd)*theta
              -eta_new*P*((1-alpha_new)*mbar+alpha_new*c)

本轮只识别这个受限的两维动作切片，不声称它覆盖所有更新规则。第一维是学习率混杂对照；第二维检验增加当前梯度直接读出的额外价值。raw moments的更新公式不变。

所有参数组、decay mask、唯一参数和实际step语义必须按冻结实现展开；上述标量写法不能覆盖掉参数组差异。grad=None与零梯度不混同。若当前 dense 配方支持不了精确映射，报告具体冲突，不自行改变算法。

R1研究的主可行域：x∈[-r,r], y∈[0,r]。y=0是“只改LR”，不是失败兜底之外的隐藏新方法。若模型推出y=0，就接受“不增加当前梯度”的结果。该研究域不证明减少当前梯度或其他机制无效。

## 2. GTTR 目标系数：不冻结随机P，不优化范数代理

固定开发参考目标 V_ctrl=V_control_ref64（同 PhaseB 已冻结64行、FP32 reference、eval mode、全库 sum/count）。真实 NLL 差定义为：

    J_z,k(a) = E_omega[ V_ctrl(theta_plus(omega)+D(omega)*a)
                       -V_ctrl(theta_plus(omega)) | z,k ]

这是真实任务函数对一次更新切片的条件期望；不是未来轨迹模拟，也不是验证集梯度驱动完整训练。本轮只是离线辨识。当前样本分布遵守冻结 train sampler。

在每个 theta_plus 计算：

    q_plus = grad V_ctrl(theta_plus)
    H_plus = Hessian V_ctrl(theta_plus)
    b_i = D_i^T q_plus_i                  # 2维
    K_i = D_i^T H_plus_i D_i              # 2×2对称矩阵

对 FIT 样本平均得到 b_hat、K_hat：

    m(a) = b_hat^T*a + 0.5*a^T*K_hat*a

不估计完整Hessian；仅两次目标方向HVP。每个方向先detach，不对方向的构造求导。q/H必须在同一theta_plus、同一scalar目标、同一math后端与精度下计算。不能使用训练Hessian冒充目标Hessian。

控制坐标a的Hessian正是上述K，因为theta(a)对a仿射。P对当前样本的依赖已包含于每个D_i，不能因对a求导而额外/错误地对当前样本求导。

逐probe系数包含随机相关性：E[D^T H D]不能换成E[D]^T E[H]E[D]，q_plus也依赖当前训练样本导致的theta_plus。

局部二次式的余项在适当光滑域中可由O(||D*a||^3)描述；本轮不把Lipschitz常数视为已知，也不据六个state宣布统一误差界。R1的有限偏移检验就是要检验这个近似能否使用。

### 2.1 为什么这不是“再挑一个alpha”

模型先输出b、K，再比较：

    a_LR = argmin_{x∈[-r,r]} m(x,0)
    a_2D = argmin_{x∈[-r,r],y∈[0,r]} m(x,y)

    gain_structure_pred = m(a_LR)-m(a_2D)

嵌套域保证模型内这个差非负，但这只是求解器性质，不是实测收益；必须在AUDIT上验证。

若K正定且解在内部，可以消去x：

    b_eff = b2-K21*b1/K11
    K_eff = K22-K21*K12/K11
    y_star = -b_eff/K_eff
    x_star = -(b1+K12*y_star)/K11

随后受原可行域约束。b_eff不是任意新proxy，而是这个任务二次模型在允许LR调整后，对读出改动的剩余一阶驱动力。

若K半正定且LR-only解处 b2+K21*x_LR≥0，则y=0符合相应KKT条件，可作为模型内排除“增加当前梯度”的依据。边界、奇异或不定K不能乱用Schur公式。

一般情况用二维二次函数在矩形域上的完整候选枚举：角点、各边上的内部极小点、可行内部驻点、原点。无需黑箱调参器。保留负特征值，不将K事后截成PSD，不加未经规定的jitter来制造最优内点。稳定性阈值只作线性代数的数值处理，报告原特征值/条件数；无法数值求解则标UNRESOLVED。

### 2.2 Tracking对应关系和不能声称的事

对采样前mu、历史偏差b_hist=mbar_prev-mu、创新xi=c-mu：

    mbar-c = rho*(b_hist-xi)
    d1 = eta*rho*P*(b_hist-xi)
    b2 = E[q_plus^T*d1]

所以b2确实读取了当前梯度与历史差异经过真实P后对任务的总作用。但本轮不分别识别mu/b_hist/xi，不能把b2<0称为“消除了有害动量滞后”。若未来要作偏差/噪声的机制归因，需另行设计条件均值估计及其误差。

## 3. 状态、参考LR与样本角色

### 3.1 固定状态

原v07 bank，seeds2030–2032 × start46/82，共6个state，按(seed,start)排序。真实n0从checkpoint读。不用D1好坏筛state，不把S3剔除；不混入h128/h256 snapshots。

参考LR倍率 k∈{1,2}，两者都对应同一恢复state上的一次counterfactual参考更新，共12个state×k条件。k2来源于D1显示基线仍有LR空间的事实，不称已调到最优；这一控制用于避免只在弱基线附近发现“结构收益”。

使用已核验finite_cosine_then_hold_5e-5_v01，读取真实gm。只从原参数状态产生一次参考端点，不连续推进训练。记录configured_peak与applied_eta，不把start46的peak写成当前LR。

### 3.2 当前训练样本

每个state固定16个新的effective-batch probes：FIT 8，AUDIT 8；每个effective batch含2个microbatch，保持真实batch、accum、mask、clipping、BF16训练梯度路径及FP32状态。

提议固定整块SIM种子202609230100..202609230105（每state一个可重复序列，前8 FIT、后8 AUDIT），通过纯元数据排除已执行冲突；若冲突整块+1000，结果前冻结。SIM仅从原train support取样。保存实际ID/mask/采样方式，不能把不同seed自动称为文档独立。

同state的k1/k2共享这16个probes。FIT用于估计b/K和求解，AUDIT用于检验；先封存全部FIT系数和解，再读取AUDIT结果。不用AUDIT重新选r、修系数或增加样本。

8个样本只是定点pilot，不声称已经精确估计全梯度噪声分布。不要把12个条件或16个probes当成12/16个独立训练seeds。

### 3.3 目标角色

- V_ctrl：旧开发参考64行，FP32可微reference、eval batch8、sum/count。FIT计算q/H，AUDIT做真实端点NLL。它是建模目标，不是final。
- V_check256：从PhaseA monitor ID排序后取前256行，纯ID规则事前冻结；eval batch32，继承PhaseA评价精度/模式。只在AUDIT上读取primary半径的native、a_LR、a_2D三端点，不计算梯度、不用于求解。名称必须不同于4096行全monitor；仅是目标迁移检查。
- V_final：0模型调用。不能声称R1通过final或能力评估。
- 当前D0不存在doc ID的限制保持，不开全语料大审计，也不声称行不交就完全独立。

## 4. 计算顺序与一次性固定测量域

### R0：CPU与接口准备

只读最新纠正后的PhaseA/D1配方、v0.2、score64 ID与state恢复工具。不要重审旧预算。

冻结全部配置，包括r_primary=.05，r_half=.025。它们是测量/验证尺度，不能称为已推导的最优算法系数。primary=.05报告为主，half只作预先规定的模型尺度检查；primary失败而half较好不能事后重命名主结果。

CPU测试见§8。真实schedule/step/参数共享冲突必须在GPU前说明。

### R1a：FIT系数辨识

每state计算FIT8的真实g/c/raw moments一次；每probe分别构造k1/k2的d0,d1与theta_plus。

在隔离state副本上，计算V_ctrl的q_plus与H_plus作用于d0,d1。每个target microbatch梯度以全库valid denominator加权，8个microbatch相加，不能把batch means无权平均。

目标参考使用FP32、autocast off、可二阶微分的math attention。若原memory-efficient后端不支持二阶，允许这一条已知数学后端；同步记录base loss、q、dtype与后端。不要升级依赖、重写网络、用有限差分调到过门。math路径仍不支持时记录METHOD_UNSUPPORTED，不临时丢掉K冒充同一个模型。

可按target microbatch做一次create_graph梯度，再两次方向二阶VJP；方向detach，第二次前按需要retain_graph。随后释放图，避免保存全维Hessian或跨probe图。

保存每probe b1,b2,k11,k12_direct,k21_direct,k22、||d0||、||d1||、dot(d0,d1)、base sum/count。保留K非对称残差，不能先平均抹掉明显错误；对数值合理的小非对称可用(K+K^T)/2求解，但报告原始量。

按FIT8平均得到每条件b/K，求r=.05/.025各自的a_LR与a_2D。全部12条件的系数、解、曲率状态、boundary标志和预测点表一起封存。R1a不读取AUDIT标签。

### R1b：AUDIT真实有限偏移

对r∈{.05,.025}，固定网格：

    x∈{-r,0,+r}, y∈{0,r/2,r}

两个网格合并后16个唯一坐标。再加入两半径各自的a_LR、a_2D，最多20个唯一坐标（含native）。坐标去重仅为复用，不能删除不利点。

对每个AUDIT probe及参考k，真实计算g/c、moments，构造theta_plus、d0、d1；在所有坐标读取V_ctrl真实NLL。不能从二次式生成“真实值”。每个端点都从相同base恢复，eval mode和随机状态隔离。

保存逐probe逐action的DeltaV=sum/count difference。FIT模型预测条件均值；不要求它预测每个probe实现。主要比较预测均值与AUDIT均值，且保留逐probe差异。

V_check256仅测primary的native/a_LR/a_2D；输出对应NLL差与paired误差。它的数值不能直接与V_ctrl相减作模型误差，因为目标不同。

本轮没有主训练、没有长期梯度场模拟。所有端点都是counterfactual临时状态。optimizer算术、参数拷贝、HVP和评价都计费，不能因为committed_updates=0就称免费。

## 5. 不以PASS代替数据：关键统计与判定

### 5.1 系数和几何

每条件输出 b1/b2 的FIT均值、SE和bootstrap区间；K三个独立元素、特征值、条件数、Gram矩阵D^TD的统计。

b_eff和K_eff仅在条件适用时输出，不适用则NA并说明。若d1与d0近共线，明确结构与LR在当前切片上难区分，不把它当稳定独立机制。

bootstrap以effective-batch probe为单位；k1/k2和动作保持配对。FIT与AUDIT独立重抽；1000次CPU参考。此CI仅对当前有限probe分布/固定目标的估计，不是训练seed总体置信度。报告采样器相关性限制。不要造“8个样本已证明机制”的p值。

### 5.2 模型误差

固定网格分别输出各坐标 J_hat、J_audit、paired SE、absolute_error；计算完整固定网格SSE/TSS（不含必为0的native可作为固定规则），小TSS标LOW_SIGNAL而不是塞任意floor制造PASS。

目标相关半径误差要直接报告 maxabs_error、RMSE，以及相对于两种解收益差的大小。小半径较好只说明局部性，不自动部署小半径规则。

不要把观察点上的误差上界称为连续域统一数学上界。

### 5.3 模型推出的结构增益，必须用AUDIT确认

    gain_pred = m(a_LR)-m(a_2D)
    gain_audit = mean[V_ctrl(theta(a_LR))-V_ctrl(theta(a_2D))]

同一AUDIT probe配对计算gain，保存均值/SE/CI。

如果a_2D与a_LR相同或y_star=0，输出MODEL_RECOMMENDS_NO_READOUT_CHANGE。不要为得到非零解改正则/半径/目标。

如果预测结构有益但AUDIT不支持，输出PREDICTED_BENEFIT_NOT_CONFIRMED；可能是有限样本、模型余项或目标读取问题，按数据分开。

如果同轴AUDIT有益、V_check方向不同，输出TARGET_TRANSFER_NOT_ESTABLISHED；不能仅凭V_ctrl通过进入独立成本声明。

允许每条件不同判定，汇总按seed、start、k分层，不用12个条件当12个独立seeds。不设“必须多少条件通过”的强制绿灯；本轮完成后由研究者根据关键表决定下一步，不能自动训练新optimizer。

### 5.4 不能从结果推出的结论

- 不能从b2或二次最优解宣称mu、噪声、历史偏差分别已辨识。
- 不能从K高特征值/Gram夹角宣称哪一层负责泛化。
- 不能从局部结构增益称“tokens-to-target改进”。
- 不能因QH方向有益而宣称QH或范数grafting首创。
- 不能以R1重新给旧NORM/v3赋予成功结论。

## 6. 下一阶段路线：不在本轮自动执行

### R2：从已验证系数关系到一条可执行规则

先根据R1决定是否有超出LR的读出空间；没有就明确排除此分支在所测域的进一步开发。

若有，必须写清b/K或其所需组合如何在线获得、何时计算、误差如何影响规则。允许成本较高的离线辨识；不能把它伪装成在线免费量。若系数跨状态稳定，可研究从联合目标最小化得到的固定结构规则；若明显时变，则研究廉价估计或有限频率更新。不能仅凭相关图手工画一条alpha函数。

规则必须具有：原目标/已验证代理、求解式、可行域、无证据时的native/LR-only回退、开销和适用域。若暂时没有经济可行实现，结论是MODEL_NOT_YET_DEPLOYABLE，不随意补一个固定系数。

### R2b：局部目标与剩余成本的离线桥接

使用同一明确监测目标和基线pi0。设J_pi0(z)为未来一直用基线到目标的期望剩余成本。在目标可达、可积、状态及停止规则充分的条件下：

    A_pi0(z,a)=E[c(z,a,omega)+J_pi0(z_next)]-J_pi0(z)

对于有适当停止条件的完整策略pi：

    J_pi(z0)-J_pi0(z0)=E_pi sum_{t<tau} A_pi0(z_t,pi(z_t))

这给出真正的成本桥，不把DeltaNLL直接替代A_pi0。

可用少量配对的真实续训来验证局部规则是否具有剩余成本上的有利作用；这是离线因果验证，不要求在线预测整条轨迹。未来预算/监测分辨率不足以分辨时保持UNRESOLVED。有限预算未达标必须保留；上式不允许用成功者均值冒充完整J。

不要求对所有访问状态逐一证明该式的负优势才可以做算法实验；但若用有限观测声称理论保证，就必须明确覆盖。最终仍需独立完整续训比较。

### R3：最终训练与成本确认

冻结由模型得到的规则后，比较native、匹配LR预算基线、模型推导规则、关键机制消融。使用新seeds/streams、同起点及scheduler。现有PhaseA256窗口可以作开发范围，不自动代表全预训练。已有ell来自开发校准，不重新挑有利阈值。

主结果：同训练input slots的NLL、预算内达标比例、检测达标tokens与删失/失败、终点独立确认。final只按冻结计划调用；no-doc限制保留。

共享GPU不比较wall-clock或能耗优劣；仍记录所有训练、诊断、HVP、目标数据暴露和调参成本。新规则若使用开发目标梯度，该数据就是训练/控制资源，基线公平性与额外数据暴露必须另行设计。

模型局部正确但长期不利时：改目标桥/规则，不谎称全部GTTR错误；算法有效但机制归因不足时：效果与机制分开。不能重新回到随意融合多个超参数的试法。

## 7. R0/R1预算：0真实训练更新，不等于0计算

按6state、2参考k、FIT8/AUDIT8、train micro=2、control64/evalbatch8、check256/evalbatch32估算：

|项目|计划上界|
|---|---:|
|当前训练probe gradient forward|6×16×2=192|
|FIT目标 q/H 共用图 forward|6×2×8×8=768|
|目标HVP方向求值|6×2×8×2=192|
|AUDIT control endpoint forward|6×2×8×20×8=15360|
|AUDIT check endpoint forward|6×2×8×3×8=2304|
|合计 network forward|18624|
|一阶与二阶backward/VJP microcall合计|192+768×3=2496|
|committed真实训练updates|0|
|V_final calls|0|

新独立硬上限：全部model forward 22000；全部backward/VJP microcall 3000；effective目标HVP方向224；committed optimizer updates0；V_final calls0。失败/二阶参考重试/数值smoke均计入，不借旧余额。virtual/reference optimizer算术及counterfactual参数状态评价数量另记，不报0成本。

q/H若未复用图而需要重算，要据实际路径更新预估并受同cap约束，不隐瞒计算量。第一固定FIT样本可以作数值smoke并在版本/语义完全一致时复用；不能根据这个样本loss调科学参数。需要修代码则旧消耗照计。

无共享GPU墙钟硬门，wall仅OBSERVATIONAL_SHARED_GPU。外部中断按完整状态恢复，所有计数不重置；没有可恢复信息则PARTIAL，不减小batch/换precision后继续冒充同配置。

保留小张量统计，不持久保存全Hessian、全probe参数栈或每步高维状态。不建服务/数据库/大型审计tar。非模型被测对象不增添额外依赖。

## 8. 最少CPU测试（参考计算独立于被测helper）

T1 精确坐标映射：eta_new/alpha_new直接AdamW读出与theta_plus+x*d0+y*d1一致；含wd、参数组和native点。错误漏乘eta/漏wd能失败。

T2 m/v与当前控制坐标无关；跨probe不得冻结P；grad=None/zero与共享参数语义不变。

T3 已知二次目标中 b=D^Tq、K=D^THD与解析值一致，矩形全局求解与穷举细网格参照吻合。

T4 正定/不定/奇异/边界解，特别是y_star=0应正确输出无读出修改；不得自动PSD裁剪。

T5 Schur消元仅在合法条件下使用，与2D解析驻点一致；设置两个方向相同/共线，识别不能独立区分。

T6 q/H作用方向必须detach；人为允许方向依赖theta的错误版本应使检验失败。

T7 例子：相同mu和Sigma但不同c^2波动导致不同Adam更新统计，防止mean(Adam)变Adam(mean)；跨probe相关项不能被替换。

T8 数据角色、FIT先封存、AUDIT不能回写系数；native差严格0；sum/count/SE/缺失/预算计算独立复核；6×2不是12独立seed。

CPU通过只验证定义和实现，不当作真实LLM模型支持。

## 9. 文件与最终必须直接展示的表

建议新增独立目录outputs/fr_gttr_r1_objective_identification_v01/；实现只需一个数学/算子模块、一个runner、一个CPU分析脚本和必要测试。命名可沿仓库习惯，但不重构旧系统、不commit/push、不改Kimi配置。

制品：
- DESIGN_AND_DERIVATION.md：本协议公式与必要本地接口映射；不扩展宏观理论。
- config.json / state_probe_manifest.json / prediction_seal.json。
- per_probe_coefficients.csv、model_coefficients.csv、restricted_solutions.csv。
- audit_target_values.csv、audit_model_errors.csv、incremental_gain.csv、check_transfer.csv。
- model_scope_and_rejections.csv、costs.json、REPORT.md、KEY_DATA_FOR_REVIEW.md。
- 最多4张图：系数/曲率及不确定性；模型与真实目标面；加入读出相对LR-only的实际gain；反例/未分辨状态。

最终聊天必须直接给：
A. 12个条件的真实state/k/eta、FIT/AUDIT数量、dtype/backend与覆盖。
B. b1/b2、K11/K12/K22、SE/CI、特征值/条件数、可用时b_eff/K_eff。
C. 两半径的x_LR、x_2D、y_2D及换回eta multiplier/alpha，pred_gain与边界标志。
D. AUDIT上相同两解的真实目标差、paired SE/CI、模型误差；native/LR-only不删。
E. V_check256的目标迁移读数，不与control轴混算。
F. 哪些条件应排除读出改动、哪些支持进一步验证、哪些无法决定；不只给总体PASS。
G. 全部实际计算成本、失败、未执行项。

本轮成功标准不是一定推出非零系数，而是测量和求解链可复算，并能用数据明确限制规则。得到y=0、模型不够局部、噪声过大或成本不合理，都可能是有效研究结果。

## 10. 参考与边界

本计划的控制坐标和Schur判据是从当前GTTR与真实AdamW读出推导的提案；未声称新颖性或实际收益。
- Ma & Yarats, Quasi-hyperbolic momentum and Adam for deep learning, arXiv:1810.06801：读出族先例，不是本项目理论效果证据。
- Agarwal et al., Disentangling Adaptive Gradient Methods from Learning Rates, arXiv:2002.11803：学习率/方向混杂与grafting先例。
- Chen, Menickelly & Scheinberg, Stochastic Optimization Using a Trust-Region Method and Random Models, arXiv:1504.04231：局部随机模型准确性条件的重要性，非完整AdamW的现成收敛定理。
- Wu et al., Understanding Short-Horizon Bias in Stochastic Meta-Optimization, arXiv:1803.02021：局部目标与较长训练目标可能错位，不能用R1替代R2/R3。

停止：完成R0/R1后不自动进入R2/R3，不训练新optimizer，不恢复PhaseB预测器。
