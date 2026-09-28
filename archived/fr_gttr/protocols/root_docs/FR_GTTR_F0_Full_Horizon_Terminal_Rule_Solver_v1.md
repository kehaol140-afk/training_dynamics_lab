# FR GTTR-F0：固定 token 预算的全时域规则求解实验 v1

日期：2026-09-25。执行端：OpenCode＋GLM5.3。
状态：新实验方案；本文件不是已经完成的结果。用户转发本文件及启动提示词后，仅执行这里限定的一轮。

## 0. 目标、既有证据与本轮边界

**Goal:** 在固定 262,144 个训练 input slots 的续训预算下，利用真实 GTTR 更新的全时域灵敏度，求解一条贯穿全部 256 步的学习率规则；随后在未参与搜索的新流和封存目标上检验终点 NLL。

**Architecture:** 真实网络在每一步计算梯度，真实 AdamW 状态照常演化，同时递推两个规则参数的全状态灵敏度。用完整预算终点的目标梯度求规则梯度，并做有限次数受限下降。搜索使用 FP32/MATH 参考计算；冻结规则后用原生 BF16 训练执行，评价两者的迁移差异。

**Tech Stack:** 现有 FR 仓库、现有 PyTorch、现有 v07 4.98M dense Transformer 和冻结数据/训练 primitives。不安装无关依赖、不升级框架、不换执行端。

本轮承接用户已经确认的主线：固定预算终点 NLL 为主要目标；tokens-to-target 不再作为本轮门；不修原冻结场 Phase B，不复用 R1 的单步系数，不延长 CB1 的检测网格研究。

历史报告提供问题背景，不构成本轮方法已有效的证据：R1有局部方向支持；R2B/CB1没有可靠的单次改动成本收益；D1的范数匹配无收益。F0不把这些结果回写，不重建历史总账。

本轮是**完整预算 continuation**，不是从初始化的完整预训练，不是廉价全程有效模型的最终完成，也不宣称新的原创优化器。前向/反向超参数梯度已有文献；本轮检验其在当前真实 GTTR 语义下，是否能从最终目标形成有用的持续规则。

本轮明确不做：FSL/ELR/DMFT联合拟合、batch/moment/clip/架构改动、初始化优化、价值网络、在线候选重放、阈值求导、动态学习率反馈策略。持续规则在部署时是已冻结 schedule，不依赖后续目标观测。

## 1. 固定实验范围

### 1.1 训练配置

- 原 v07 bank；所有起点都是 start46。
- 搜索 FIT：seed2030、2031，各一条固定新训练流。
- 确认 AUDIT：seed2030、2031、2032，各两条不同于 FIT 的新流。
- seed2032未参与本轮搜索，但曾用于历史研究，不能称为全项目从未见过的seed。
- 保留原模型、参数组、唯一参数身份、共享embedding/head、mask、训练reduction、batch=4、accum=2、context=128、1024 input slots/更新。
- weights/moments FP32；beta1=.9、beta2=.999、eps=1e-8、wd=.01、global clip=1.0，若现场冻结配方与这些摘要不符，报告真实冲突，不擅自改成摘要。
- 每次 rollout 都从对应 bank 的完整状态恢复，raw moments、真实累计计数不重置。
- T=256 是主目标窗口，N=256×1024=262144 input slots；valid_loss_tokens、nonpad_tokens按实际mask累计，三者不混同。
- 不因为loss曲线好坏提前停止正常轨迹。数值非有限可终止该轨迹并记失败，不能删掉后继续声称该规则稳定。

### 1.2 新流与随机性

建议保留整块 stream_id：202609250400–202609250407：
- FIT seed2030 → 400；seed2031 → 401；
- AUDIT seed2030 → 402/403；seed2031 → 404/405；seed2032 → 406/407。
使用完整ID，不能只记录后三位。以元数据检查是否已用；若冲突，整块按+100顺延，直到未用，结果前落盘。不能基于loss选择stream。

同一seed、同一stream的所有候选和有限差分重放必须使用共同数据/随机掩码配方（CRN）；不同phi导致不同theta，但不改变抽样律。参考与原生backend的随机数消耗可不同，不能无证据声称跨backend逐位一致。确认比较首先在相同backend内配对。

## 2. 规则类：两个参数，全程应用

### 2.1 参考schedule及尾部语义

从已确认的有限cosine后hold规则生成256步的基准eta_base[t]，参考倍率2：

    w[t] = 2 * eta_base[t],  t=0,...,255

eta_base使用原训练调用点对应的真实global-micro计数。按R1/R2B当前调用记录，start46的首应用gm预期94、参考k2首eta约7.323586e-4；metadata和实际函数是最终依据。不能以configured_peak=5e-4替代applied_eta，也不能按起点名重新解释n0。

eta_base越过原cosine终点后严格保持5e-5，故w的参考尾部为1e-4。不得继续执行越界回弹cosine。

新规则整体定义为：

    phi = (a,b)
    psi[t] = 2*t/(T-1)-1
    Z(b) = sum_t w[t]*exp(b*psi[t]) / sum_t w[t]
    eta[t;phi] = w[t] * exp(a+b*psi[t]) / Z(b)

    Phi = [-log(2), log(2)] × [-log(2), log(2)]

因此：

    sum_t eta[t;phi] = exp(a)*sum_t w[t]

- a调节整条LR的总尺度；a=-log2/0/+log2在b=0时对应原配方倍率1/2/4。
- b重新分配早晚阶段；b<0相对前置，b>0相对后置。
- 在同a下，改变b不改变sum eta，减少单纯整体放大的混杂；但不保证相同训练动力学，也不保证累积weight decay乘积严格相同。
- w已包含原cosine与参考tail；新规则允许调整tail的时间形态，这一改变在此明确声明，不继续称为“原hold tail完全不变”。
- schedule_id = gttr_terminal_normalized_tilt_T256_v1。
- 本轮不另加cooldown，不重warmup、不延长T；预算末端读原始theta，不选中间最佳checkpoint、不新增权重EMA。
- SHORT64也使用这个T=256的psi和Z，只把其求解目标改为h64，不重新将schedule压缩为64步。

Z、psi及eta用float64标量稳定计算，并在每次更新按原接口传入；记录每条schedule的hash、sum_eta、首/末/min/max eta、前后阶段累计eta，以及sum log(1-eta*wd)。检查所有声明衰减因子合法。

重要：这是一项预先限定的规则族，不声称它覆盖所有持续策略。a/b可行域是实验范围，不是理论最优性结论。解在边界时标RULE_DOMAIN_BOUNDARY，不自动扩域。

### 2.2 对规则参数的直接导数

    d_eta/da = eta
    mu_psi(b) = sum_t w[t]*exp(b*psi[t])*psi[t] / sum_t w[t]*exp(b*psi[t])
    d_eta/db = eta * (psi[t]-mu_psi(b))

必须包含Z对b的导数，不能把归一化常数detach后漏掉这一项。

## 3. 目标、评价数据及封存

### 3.1 搜索目标

V_design使用原V_monitor的4096行（D0 rows[4096:8192)，优先读取原manifest中的实际ID），全库token-weighted sum/count。参考计算采用FP32、eval模式、MATH SDPA，target_id单独命名：

    V_design_monitor_FP32_MATH_sumcount_v1

数据行和mask与旧monitor对应，但mode/precision若不同，不把绝对NLL与旧BF16表合并成同一数值轴。本轮所有待比较checkpoint用同一个新评价语义。

FULL目标：

    J_256(phi) = mean_{FIT seed2030/2031}[V_design(theta_256(phi))]

SCALE目标相同，只约束b=0。SHORT目标为同两条FIT流的V_design(theta_64(phi))，不读取该候选h256作为选择依据。

现有monitor由本轮明确作为开发求解目标。它不再承担独立确认角色。实际选择根据FIT上的指定目标；中途曲线不选参。

### 3.2 独立于本轮搜索的确认数据

原V_final（D0[8192:12288)）继续封存，本轮调用=0。

新增：V_confirm_F0，首候选为D0 rows[12288:16384)，4096行。结果前用可用metadata核查训练支持集、旧评价/开发ID及本轮V_design不重叠。

若已有使用记录，按完整4096行块顺序后移，最多检查32块；按ID及使用记录选择，绝不读取loss选择。若使用身份不能确认，停在final阶段之前，报告CONFIRM_SET_IDENTITY_UNRESOLVED，不静默复用旧开发集。没有doc_id时保持DOC_INDEPENDENCE_UNVERIFIED：行ID分离不证明文档级独立。

V_confirm_F0只在所有规则和AUDIT终点checkpoint封存后读取。每个预定checkpoint一次全库评价，规则/范围/solver不因其结果改变。之后这份确认数据已经被使用，不能在后续新方法中仍称完全未见过。

目标评价batch32：4096行共128 forward。compute gradient时按每批valid-token count加权累积，不能无条件平均128个batch均值。mask/denominator逐点输出。

## 4. 求解器：完整前向灵敏度，不冻结未来梯度场

### 4.1 两条真实计算路径

REF32：FP32网络/weights/moments，autocast关闭，MATH attention，TF32关闭或等价highest设置，记录实际版本和backend；保留原train dropout概率与输入/累积语义。每一步都重新计算真实g(theta_t;omega_t)。

NATIVE：原BF16 autocast训练，FP32 weights/moments，原backend与scaler设置。所有确认臂均使用此路径；不能只让候选使用不同精度。

搜索在REF32求解。NATIVE确认检验参考求解能否转移到实际训练执行。两路径不假定参数逐位相同，参考HVP不是逐比特BF16程序的导数。

### 4.2 全状态灵敏度

因为只求两个参数，优先使用**完整前向灵敏度**，而不是保存256步全部反向图。

对j∈{a,b}，维护：

    S_theta[j] = dtheta/dphi_j
    S_m[j] = dm/dphi_j
    S_v[j] = dv/dphi_j

起点固定且不依赖phi，所以三者初值全0。每个真实参考更新都用当前theta的真实batch Hessian作用，不使用冻结H0、旧R1算子或虚拟的未来梯度。

正常分支递推：

    g = grad L_batch(theta)
    Hg_j = Hessian(L_batch)(theta) @ S_theta[j]
    c = real_clip(g)
    dc_j = J_real_clip(g) @ Hg_j

    m_plus = beta1*m + (1-beta1)*c
    v_plus = beta2*v + (1-beta2)*c*c
    dm_plus_j = beta1*S_m[j] + (1-beta1)*dc_j
    dv_plus_j = beta2*S_v[j] + (1-beta2)*2*c*dc_j

    mbar/vbar及其导数使用真实参数累计step做相同bias correction
    P = 1/(sqrt(vbar)+eps)
    r = P*mbar + wd*theta

    dtheta_plus_j = S_theta[j]
                   - d_eta_j*r
                   - eta*(P*dmbar_j + dP_j*mbar + wd*S_theta[j])
    theta_plus = theta - eta*r

vbar>0时：

    dP_j = -dvbar_j/[2*sqrt(vbar)*(sqrt(vbar)+eps)^2]

实际代码中的grad=None、decay mask、共享参数及clip分母epsilon要按真实实现展开，不能用理想公式覆盖执行差异。

vbar=0时不直接除0，也不偷偷加v_floor或改变eps。先识别结构上始终不参与的零分支；只有能证明该方向导数为零的分支才用零导数。其他非光滑/不可微分支记录并中止该导数实验；不能用nan_to_num伪造有限hypergradient。

实现建议：每一步用create_graph的一阶梯度计算该步HVP，再用显式灵敏度方程递推；释放本步计算图、detach数值状态与灵敏度数组，避免意外保留全历史图。**detach计算图不等于清零灵敏度**：S必须完整传到下一步。不得漏掉dm/dv反馈。

当某个S_theta精确为0时可解析跳过该方向HVP，计数应减少而不是记成一次真实计算。

终点梯度：

    q_H = grad V_design(theta_H)
    grad_phi_j J_H = dot(q_H, S_theta_H[j])

这是预算终点目标的导数，不是每步验证梯度之和。T256臂必须传递到256，不截断4/16/64步；不把途中目标当终点。SHORT64是明确命名的对照。

前向灵敏度与完整伴随是同一链式求导的两种计算方向；选择前向是因为参数维数只有2，并非省略长期影响。没有任何新未来预测网络。

### 4.3 HVP与共享参数

HVP作用于该步真实scalar训练loss；两microbatch的权重、mask和累积方式与实际batch一致。可逐micro计算后按原权重合成，不得平方/裁剪每micro后再假装等于全batch结果。

方向S_theta当作给定的方向向量，不在HVP内部对其构造再次求导。共享参数只有一个自由度；functional_call的tie语义必须明确。不能复制共享weight后分开更新。

每方向每effective batch的HVP记一次；若由两个micro backward实现，还要记两次HVP/VJP microcall。终点q是额外目标backward，不能混入训练backward后消失。

若当前网络二次导数不能由MATH后端支持，报告具体算子。允许一次保持数学函数的参考实现修正并用既定测试核验；不安装新训练框架、不改变结构，不改用stop-gradient近似后仍称全程精确灵敏度。

## 5. 固定求解流程与基线

### 5.1 同域固定3×3网格（既是对照，也提供共同起点）

    a_grid = {-log2, 0, +log2}
    b_grid = {-log2, 0, +log2}

九条完整规则都在两条FIT流上用REF32运行256步，使用同一终点目标。保存每条完整结果，不因差而删点。

- GRID2D256：在九点中选FIT平均终点NLL最小的规则。
- 共同起点phi_init：只从b=0的三个固定尺度中按同一目标选择。FULL/SCALE/SHORT都从这个相同起点出发。
- 全FIT共同有限才允许该grid点参与选择；任一轨迹数值失败则该点记invalid/+inf，失败和成本保留。
- 精确相等时按(|a|+|b|, a, b)字典序选择，避免目标无差却随机选形状。
- 最好grid位于边界只标范围限制，不扩大a/b。

这个网格对照不使用GTTR导数。它与梯度求解的计算量不完全相等，因此不能声称本轮是等搜索成本竞赛；必须逐方法报告搜索费用。本轮检验的是终点作用和规则价值，而不是先宣称更省搜索计算。

### 5.2 三个固定求解器

1. FULL256：最小化J_256(a,b)，两参数。
2. SCALE256：最小化J_256(a,0)，只调a。
3. SHORT64：最小化J_64(a,b)，两参数，但256步schedule时钟不变。

各最多3次外层梯度迭代，每次梯度由两个FIT trajectories平均。每次都从对应原bank恢复，不能把不同外层试验接着训练。

设g为允许坐标上的完整hypergradient。用固定候选步长：

    h = (log2)/2 * {1, 1/2, 1/4}
    phi_trial = project_box(phi - h*g/max(||g||_infinity, 1e-12))

按顺序最多3次试步；只使用该求解器自己的FIT目标做Armijo：

    J(phi_trial) <= J(phi) + 1e-4 * dot(g, phi_trial-phi)

J_trial由新的完整真实REF32 rollout得到，不用线性预测值代替。每次真实试步全部计费。先接受第一个满足条件的点，不试完后任意选最小。投影后delta=0可直接跳过；三个都不满足则本求解器停止，保留最好已接受点。最多3轮，不增加迭代救结果。

g精确为0则停止；低于校准分辨水平时标LOW_SIGNAL，并保留原点/当前点，不伪造显著梯度。所有评估轨迹保持相同FIT CRN。

SHORT64的设计是“较短目标优化的整条规则”对照，不是用它参加主优化。它的共同起点来自固定T256尺度选择，此共享起点必须披露，不称完全独立的短时方法。它的训练后半段不得拿来调参。

冻结后的五个确认臂：

    BASE_K2       : phi=(0,0)
    SCALE256      : full-budget scale-only解
    FULL256       : full-budget两参数解（主候选）
    SHORT64       : short目标两参数解
    GRID2D256     : 固定九点的best-dev解

如果两个标签对应完全相同的schedule，允许复用一次轨迹，但保留所有标签、标IDENTICAL_POLICY，不把它们当不同独立样本。

## 6. 仅做必要的验证，不把旧预测器门带回来

### 6.1 CPU理论/实现测试（至少8类）

1. schedule在phi=0恢复w；sum eta等式；d_eta/db包含Z导数；SHORT的256时钟不变。
2. 小型平滑网络/二次loss上，显式全状态前向灵敏度对照未截断完整autograd；T>=16。
3. 明确构造terminal与短时梯度不同的测试，保证FULL不误用H64目标。
4. clip active/inactive的方向反馈与参考JVP相符，边界另标。
5. 偏差校正、共享参数、grad=None/0、weight decay直接LR导数保持。
6. 明确反例：故意清零S_m/S_v或detach梯度反馈应使测试失败，防止只传播eta直接项。
7. 目标sum/count梯度与逐样本加权一致，确认数据访问顺序与seal检查。
8. 预算计数包括HVP microcall、目标backward、失败和所有候选rollout；wall不作为科学门。

### 6.2 GPU smoke（最多64 trajectory updates）

固定FIT seed2030 stream，phi=(0,0)，每条16步：
- REF32普通运行；REF32＋SENS运行；
- 旧NATIVE原k2运行；新NATIVE schedule wrapper运行。

同backend内核查参数/moments/step/RNG/eta一致。REF32对NATIVE不要求逐位一致；明确二者数值语义。最多8次V_design全库读数（其中最多1次目标gradient）。不读取V_confirm/V_final。

### 6.3 完整256步终点导数校准

固定FIT seed2030、phi=(0,0)，一条REF32完整256步＋两参数SENS。
用共同随机输入分别计算phi±epsilon*e_j的真实终点值：

    epsilon={0.01,0.005}, j=a,b

共8条value-only完整rollouts。比较的是**整个256步训练终点对规则参数的导数**，不是原S3的局部梯度场半步门。

重复评价同一终点一次，eps_V=max(重复目标绝对差,1e-7)。令：

    abs_floor=max(5e-4, 4*eps_V/0.005)
    tol_j=abs_floor + 0.20*max(|g_sens_j|,|FD_.01_j|,|FD_.005_j|)

报告两尺度FD、SENS、符号、误差与clip active记录。
- 两个FD的一致性和SENS对小尺度FD误差均<=tol：该坐标校准通过。
- 三者都小于abs_floor：LOW_SIGNAL，不把符号正确当强证据。
- 有分辨信号且不一致/非有限：FULL_HORIZON_GRADIENT_UNRESOLVED，停止导数求解阶段；不调epsilon、精度、T或drop状态救结果，不烧确认集。

该验证只是实现与参考导数可用性检查，不是GTTR所有可能方法的准入门。发生失败则完整报告具体数字与已花成本，不无限增加诊断轮。

## 7. 冻结与原生训练确认

### 7.1 封存顺序

网格及三个solver完成后，保存五条policy、256步eta数组、全部FIT历史、hash和算法版本。禁止读取确认loss后再选policy。

用这些固定policy，在AUDIT三个原state×两条新流上执行NATIVE训练：五臂各256步，最多30条。没有在线搜索、目标梯度/HVP，只有查表eta和原AdamW。

V_design可在h0/64/128/256读出过程，但所有规则已锁定，不据此提前停止或改为best checkpoint。共同h0按seed共享一次。

另作参考路径对照：在AUDIT每seed的第一条流上，BASE_K2与FULL256各运行一次REF32到256，共6条。其目的只是检查从可微参考到native的迁移，不是增加新的方法选择轮。

所有NATIVE与REF32确认checkpoint及trajectory status保存后，才一次性评价V_confirm_F0：最多36个终点、每checkpoint一次；固定eval模式FP32/MATH和sum/count，对所有训练精度一视同仁。原V_final调用仍为0。

### 7.2 主要比较及统计

主比较：NATIVE h256 的FULL256 vs SCALE256，在V_confirm_F0上的配对NLL差。

伴随比较：
- FULL256 vs BASE_K2：是否优于旧参考配方；
- FULL256 vs SHORT64：预算终点求解是否比短时目标更有价值；
- FULL256 vs GRID2D256：是否比同域简单搜索更有竞争力（搜索费用不等，必须分列）；
- REF32与NATIVE的FULL-BASE差：参考到执行的迁移；
- h64/h128 V_design只解释时域，不更换主终点。

方向统一：gain(A vs B)=V_B-V_A，正值表示A更好。paired_difference可另列但不能混符号。

逐seed、逐stream保留；3seeds不是30个独立模型。可做1000次seed→stream两级配对bootstrap，固定rng=202609250599，报告样本范围和CI；3个顶层seed的区间只作pilot不确定性描述，不据其宣称广泛总体显著性。

评价数据64行连续block bootstrap可作为数据级伴随，但不能替代训练seed不确定性。没有doc_id就明确BLOCK_PROXY。

主要效果字段至少为：mean gain、每seed的两流均值、正/负/tie、95%描述区间、失败数。不强制必须出现正结果。若均值正但CI跨0，标GAIN_UNRESOLVED；若候选数值失败多于比较臂，失败不能删掉只平均成功者。失败率与成功样本质量分开展示，不凭空给NaN一个有限NLL惩罚。

规则同一、solver未移动、边界最优、早期曲线好但terminal无收益都应如实报告。

## 8. 预算：所有真实轨迹与导数研究成本都算

以下为不考虑复用、不早停、每solver都试完3×3试步的计划上界：

|阶段|trajectory update evaluations|
|---|---:|
|smoke四条×16|64|
|终点校准：1条SENS＋8条FD，各256|2304|
|九点grid×2FIT×256|4608|
|FULL256：(3梯度＋9试步)×2×256|6144|
|SCALE256：同上|6144|
|SHORT64：(3梯度＋9试步)×2×64|1536|
|REF32 audit：3×2×256|1536|
|NATIVE audit：3×2×5×256|7680|
|**合计**|**30016**|

正向状态更新确实被计算就计1个trajectory_update_eval，无论是在研究、FD、SENS还是最终训练中。灵敏度数组递推不是额外训练轨迹，但其HVP和算术单列。阶段间同点复用可减少实际费用，不能产生新的独立证据。

最大计划训练micro-forward/backward = 2×30016=60032。
目标评价：design至多207次全库（smoke8、校准10、grid18、三个solver各24、native audit93、reference audit6）；confirm至多36次；合计243×128=31104。
其中最多20次design目标需要gradient，计20×128=2560次target backward micro-call。
HVP方向上界：smoke32＋calibration512＋FULL3072＋SCALE1536＋SHORT768=5920；每次对应2个micro VJP，最多11840。
所以最大计划：

    network_forward = 60032 + 31104 = 91136
    all_backward_VJP_microcalls = 60032 + 2560 + 11840 = 74432

|计数|计划上界|本轮硬上限|
|---|---:|---:|
|trajectory_update_evals|30016|34000|
|network_forward|91136|110000|
|all_backward/VJP_microcalls|74432|90000|
|train-HVP effective-batch directions|5920|7000|
|design full-set evals（含带gradient）|207|250|
|confirm full-set evals|36|40|
|design target gradient evals|20|24|
|原V_final调用|0|0|

所有cap同时满足。两micro合并梯度后HVP复用若不额外forward，只记真实调用；若某实现重新forward，必须增计并提前预测费用，不能把表中计划当实际。

失败/重放/参考backend修正均计入；旧账不改，不自动扩域/增seed/增外层轮数。shared GPU wall只OBSERVATIONAL_SHARED_GPU，不设90分钟科学门，不按GPU利用率归因算法好坏。

search、gradient construction、FD calibration、reference audit、native audit、evaluation分别列账。单次部署一条已求解schedule只有查表开销，并不意味着得到该schedule的搜索成本免费。

若达到预算，仅在完整阶段边界保存并报告PARTIAL，不能删除不利条件保证表面全覆盖。

## 9. 文件、实现与验收数据

建议最小新增：
- research/src/fr/f0_terminal_rule_v01.py：schedule、目标、灵敏度、求解器；
- research/scripts/f0_terminal_rule_run_v01.py：分阶段运行与计数；
- research/scripts/f0_terminal_rule_analyze_v01.py：统计与报告；
- research/tests/test_f0_terminal_rule_v01.py：CPU测试。

可以按现有仓库惯例调整模块名，但不要另建框架、重构历史runner或重写整个dataset。

输出目录：outputs/fr_gttr_f0_terminal_rule_solver_v01/。

至少输出：
1. config/manifest/target_dictionary/state_and_streams/schedule_arrays；
2. cpu_tests/smoke_checks/terminal_gradient_check.csv；
3. grid_terminal.csv（九点×两FIT全部数据）；
4. solver_history.csv（每个g、试步、投影、接受原因、J64或J256）；
5. frozen_policies.json + policy_seal.json；
6. audit_terminal.csv、audit_learning_curves.csv、paired_comparisons.csv；
7. reference_native_transfer.csv；
8. cost_by_stage.csv/resource_log；
9. REPORT.md、KEY_DATA_FOR_REVIEW.md，规定图表。

建议四张图：固定schedule曲线；FIT终点目标迭代；NATIVE完整预算学习曲线；主终点逐seed配对gain。图只展示，不参与选参。

最终聊天必须能直接核验：
- 首/末eta、sum_eta、a/b，五条规则是否实际不同；
- 两坐标全T导数、两尺度FD和误差；
- 每轮完整目标下降/不下降，哪些规则触边或没有更新；
- 每seed/stream五臂终点确认NLL的sum/count/mean及配对差；
- FULL是否超过scale-only、SHORT与grid；
- REF32是否与NATIVE得到不同排序；
- 全部搜索与训练费用，主要训练tokens和额外费用分开；
- 哪些结果是支持、未分辨、失败、未测试。

不要只返回“PASS＋文件路径”。旧实验不覆盖；不把CPU通过等同于算法收益；不把2维终点优化包装成全局最优或FSL/DMFT已经闭合。

## 10. 执行顺序与停止

1. 只读必要现场文件；冻结新数据角色、流和T256 schedule定义。
2. 最小代码及CPU测试。
3. GPU smoke与完整终点导数校准。
4. 固定grid与三个求解器。
5. 封存规则；真实NATIVE及REF32 audit。
6. 封存checkpoint后一次性读confirm。
7. 分析并给一次正式报告，停止。

代码错误可在未读取确认数据前修复，保留失败记录并记费；不因为loss不好改变规则、solver、T或终点。
不commit/push、不安装新依赖、不修改OpenCode/GLM连接、不占用或停止其他GPU任务、不建常驻monitor。任务有明确阻塞才报告，不仅交付另一个草案；完成后不自动进行从初始化预训练或放大模型。

## 11. 来源与推导责任

历史配置/结果来自用户FR报告及本地冻结文件；本文件新增的两参数normalized-tilt规则类、solver配置及F0实验矩阵是本轮建议，不是文献已证明的最佳配方。

- Franceschi et al., ICML 2017, Forward and Reverse Gradient-Based Hyperparameter Optimization：前向/反向完整训练超参数导数的方法依据；不作为本项目算法首创或有效性证据。`https://proceedings.mlr.press/v70/franceschi17a.html`
- PyTorch官方HVP：`https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html`
- functional_call与共享参数：`https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html`
- 数值语义：`https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html`

本轮不使用论文中的SGD核回归风险泛函替代真实AdamW；完整参考求解可作为后续廉价有效闭合的比较基准，而不是提前宣布闭合已完成。
