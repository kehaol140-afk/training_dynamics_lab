# FR GTTR-D1：直接更新规则与实际续训效果 pilot v1

日期：2026-09-23。执行端：现有 Claude Code＋Kimi K2.7。

## 0. 本轮决策、性质与授权边界

研究路线改为：真实训练结构 → 一个明确的直接更新改动 → 实际任务/训练 token 结果 → 必要的机制解释。

- 原 Phase B v3 及 S3 的 GATE_STOP 保留，不恢复、不回写 PASS。
- 不继续构造冻结梯度场、不计算 FD/AD-HVP、不预测 LR 候选、不训练控制器。
- GTTR 保留真实 AdamW 语义、当前梯度/历史状态关系和任务 Readout；不要求其全轨迹预测先通过。
- 本文件是一个新提议的、直接规则的开发 pilot，不是从 S3 推导出的必胜方法，也不作新颖性声明。
- 用户将本文件及启动指令转发给执行端，即授权在本文范围和预算内实现、CPU 测试、smoke、main 和报告。不是授权后续扩规模或自动调参。
- 只在已知低成本范围测试一个候选系数 alpha=0.10，不允许看到结果后增加 alpha、改公式、选层、换目标或替换 seed。
- 主指标是训练 input-token 效率及同预算 monitor NLL；墙钟仅 OBSERVATIONAL_SHARED_GPU。不得以共享GPU耗时判方法好坏或施加科学硬停止时间。
- 文件中实际配方和旧结果来自用户历史报告。现场先读取最近一次已修订的 Phase A 原件，仅核对本轮必需接口，不重开全项目审计。

## 1. 一个机制问题与候选规则

### 1.1 问题

保持 raw m/v 的更新、预条件器、batch、clip 和 schedule 不变，仅改变“从已更新 moments 读出本次数据更新方向”的方式：增加少量当前梯度的参与，并控制瞬时数据更新幅度，是否改善实际 continuation NLL/达标训练 tokens？

这是一项当前梯度—历史读出结构的实验，不是“已证明动量滞后有害”，也不是对真实条件 tracking error 的直接估计。

### 1.2 相关已有工作，必须如实标注

- Ma & Yarats, Quasi-hyperbolic momentum and Adam for deep learning, ICLR 2019, arXiv:1810.06801，Section 5。QHAdam 将当前梯度与 moment 读出混合；nu2=1 保留通常的二阶矩分母。
- 官方实现：facebookresearch/qhoptim，qhoptim/pyt/qhadam.py。它的内部 exp_avg 表示法可能与本仓库 raw moment 不同；不能直接把已有 raw m/v 加载进去导致双重/错误偏差校正。本轮优先复用冻结 raw m/v primitives，再实现明确读出公式。
- Agarwal et al., Disentangling Adaptive Gradient Methods from Learning Rates, arXiv:2002.11803：通过 grafting 区分方向与步幅的贡献。
- AdamW 解耦衰减保留原配方语义。

因此，本轮候选是“QH 式当前梯度读出＋逐唯一参数张量的数据更新范数匹配”的受控实验组合。不把它包装成已经确立的新优化器或GTTR独创理论。若有效，仍需与已有工作作更细比较、匹配调参和独立验证。

### 1.3 唯一公式

以下 p 是唯一参数张量，g(p) 是 optimizer 参数组，c_p 是本次真实梯度累积之后、全局 clipping 之后的梯度。

按冻结 runner 正常执行：

    m_plus = beta1 * m_old + (1-beta1) * c
    v_plus = beta2 * v_old + (1-beta2) * c^2
    mhat = bias_correct(m_plus, true_parameter_step)
    vhat = bias_correct(v_plus, true_parameter_step)
    P = 1/(sqrt(vhat)+eps)

这部分所有臂的公式相同，但实际梯度随轨迹分叉而不同，不能宣称跨臂 m/v 数值相同。

本步 AdamW 数据更新方向（不含 eta 与 weight decay）：

    r_A = P * mhat

固定 alpha=0.10：

    r_Q = P * [(1-alpha)*mhat + alpha*c]
        = (1-alpha)*r_A + alpha*(P*c)

RAW_QH 读出使用 r_Q，不做范数匹配。它在所用 clipping/decay 配方中对应 QHAdamW 的 nu1=0.9、nu2=1 读出；不声称复现 QHAdam 原论文的整套训练配方/最优超参数。nu1=beta1 时也与 NAdam 家族有关，但不冒充任意实现的逐位 NAdam。

NORM_MATCHED_QH（候选）对每个唯一张量 p 独立计算：

    A = ||r_A||_F
    Q = ||r_Q||_F
    B = ||P*c||_F
    if alpha == 0: use exact native code path
    elif A == 0: r_N = r_A  # zero data step; decay handled separately
    elif Q <= 1e-12 * max(A, B): r_N = r_A  # cancellation fallback
    else: r_N = (A/Q) * r_Q

如果任何输入/中间量为 NaN/Inf，标 NUMERICAL_FAILURE，保存证据并终止该轨迹；不能用 fallback 偷偷吞掉非有限值。

真实参数更新：

    theta_plus = (1-eta*wd)*theta_old - eta*r_method

保留现有等价实现顺序、参数组、decay mask、grad=None/zero 语义。共享 embedding/head 只更新一次。标量/向量/矩阵等所有参与更新的唯一参数张量都使用同一公式，不事后挑层。eps=AdamW epsilon 不改，也不添加任意 norm epsilon 到分母破坏范数关系。

### 1.4 有什么控制住了，什么没有

在同一参数张量、同一当前状态和梯度下、非退化分支中：

    ||r_N||_F = ||r_A||_F

它限制的是当前同状态下的数据步幅，而不是总参数位移（后者还含 decay），也不是候选与真实 baseline 两条长期轨迹之间始终相同的步长。

归一化只是把 r_Q 投影到半径 ||r_A|| 的球面；没有保证它更接近真实总体梯度、具有下降性或减少最终成本。当前梯度可能引入额外波动，norm matching 也可能无益。

不把局部范数恒等式当算法收益；不把 current-batch dot alignment 当独立任务收益。

不更改已存raw moments，不回写 r_Q 或 r_N 到moment；没有新增全维持久状态。临时向量、reduce与算术开销仍需记录，不称零成本。

## 2. 五个真实训练臂：一个候选＋必要对照

|arm_id|方法|LR multiplier k|alpha|用途|
|---|---|---:|---:|---|
|ADAMW_LR050|冻结原AdamW|0.5|0|较小固定LR对照|
|ADAMW_LR100|冻结原AdamW|1.0|0|native基线|
|ADAMW_LR200|冻结原AdamW|2.0|0|较大固定LR对照|
|QH_RAW_A010|r_Q|1.0|0.10|已有QH读出/不匹配范数消融|
|QH_NORM_A010|r_N|1.0|0.10|唯一主候选|

- LR倍率整段256步固定应用，作用于原AdamW本步eta，因此连同解耦衰减一起按其标准定义缩放。明确这三臂是常规固定LR配方对照，不是隔离纯数据步幅的因果对照。
- 候选与QH_RAW的schedule/eta/decay完全相同，只比较范数匹配。
- 不是四步pulse，所有规则从fork起持续执行到h256；不再虚拟评分或选动作。
- alpha=0.10只是预先选择的温和pilot配置，不是推导出的最优值，不允许追加alpha网格。
- LR{0.5,1,2}是粗网格，不等于充分调优的AdamW。若最优落边界，必须报告 LR_BASELINE_NOT_BRACKETED，不能宣布战胜“最优AdamW”。
- 本轮不加Cautious/Muon/AdEMAMix等更多臂。若需要扩展对照，在本轮结果后另提出，不自动补跑。

## 3. 状态、数据和schedule

### 3.1 首轮只取一个起点，控制规模

固定三个原bank状态：seeds {2030,2031,2032}、原 start=46，真实n0从checkpoint核验。start=82不在本轮；它可以留作后续起点迁移检验，但不是未曾观察的final数据。

真实future streams提议：99000、99001、99002。若仅注册而未执行可以复用；若存在非空已执行记录，整个ID块按+1000依次选择未用块，纯元数据决策，在结果前冻结，不按loss选择。只从原train support抽样。

每seed五臂共享数据/RNG序列，分别从同一full state恢复。optimizer更新差异不许改变采样顺序。三个seed是开发pilot，不是15个独立训练seeds。

建议固定执行顺序（纯运维，非择优）：
- seed2030: ADAMW_LR100, QH_NORM_A010, ADAMW_LR050, QH_RAW_A010, ADAMW_LR200
- seed2031: ADAMW_LR050, QH_RAW_A010, ADAMW_LR200, ADAMW_LR100, QH_NORM_A010
- seed2032: ADAMW_LR200, ADAMW_LR100, QH_NORM_A010, ADAMW_LR050, QH_RAW_A010

### 3.2 配方沿用已校准Phase A

预期约4.98M dense模型、参数/moments fp32、bf16 autocast、GradScaler disabled；microbatch4×accum2；beta1=.9、beta2=.999、eps1e-8、wd.01、global L2 clip=1。
读取最近PhaseA配方，实际不同则说明具体冲突，不自行改成预期值，也不引入新配方。

scheduler：

    eta_base(gm) = old_get_lr(gm,256,5e-4), gm<=256
                   5e-5,                 gm>256
    eta_arm(gm) = k_arm * eta_base(gm)

同一累计索引，不拉伸total，不重warmup，不重置m/v/n，不新增cooldown。三个倍率尾部分别2.5e-5/5e-5/1e-4；候选和QH_RAW尾部5e-5。

### 3.3 数据角色

- train仍用冻结train IDs，不能额外扩成D0全库。
- V_monitor沿用PhaseA D0[4096:8192)及实际shift/mask/dtype/mode，4096行全库sum/count。
- V_final沿用已封存身份，不做模型调用。
- 旧V_score/V_eval不用于当前规则选择；本轮不计算任何验证梯度。
- 无doc ID的独立性限制保持不变，不用相邻行标签保证文档独立，也不再次重开全语料审计。
- 本轮monitor不参与每步更新，但它是开发评价，不叫完全独立final。

## 4. 长度、目标与评价规则

### 4.1 所有成功轨迹运行完整h256

每臂256 optimizer更新，15条主轨迹；不得因为已达到ell而提前停止。这样仍可检查固定预算末端质量、后续反弹与短期收益消失。

monitor检查点：h=0,16,32,...,256，共17个。h0在同seed五臂之间共享一次；每臂其余16点都测。

本轮监测间隔由PhaseA的32改为16，是新实验预先规定；不能拿新hit直接对比旧PhaseA的32步确认时点声称提速。所有本轮方法使用相同16步网格。

### 4.2 阈值不重新估计

只取PhaseA start46的全精度冻结阈值：

    ell_star = 7.020043501537879

正式从已修订target_calibration文件读取，显示值仅交叉核对。若读取结果超出显示精度解释的差异，说明冲突，不挑新值。没有新阈值搜索、没有根据候选曲线重设目标。

检查初始state在该目标上；若意外不高于ell，标START_TARGET_MISMATCH并查身份，不能用h128/h256替换。

### 4.3 主指标与达标记录

主质量指标：每seed、每臂在h256的monitor NLL，及相同token预算下相对ADAMW_LR100、各LR对照和QH_RAW的差。

主要成本读数：满足预定连续两个monitor检查点NLL<=ell的第二检查时点，记h_detect，其前缀成功训练input slots记N_detect。没有达到则NOT_REACHED，N_detect=NA、预算上限单列。

同时保存：last_observed_above、first_in_confirmed_pair、second_confirmation、所有检查点原始值。称“离散监测下的检测达标成本”，不称精确连续首次穿越。不假设两次检查之间NLL单调，也不证明在更早未观测时没有过阈值。

数学首次达标、monitor连续确认和V_final确认分开；本轮final没有调用，所以不能称独立质量已确认。N_detect是开发目标下的新流筛查证据。

固定完整训练之后可以回读其前缀token，实验总成本仍计完整256步，不谎称实际实验在hit处已结束。

### 4.4 四种计数同时保存

train_input_token_slots / nonpad_input_tokens / valid_loss_tokens / successful_and_attempted_optimizer_updates。
预期每成功步1024 input slots，h256为262144 slots。valid loss tokens必须从真实mask逐步累加，不等同262144。
相同seed的五臂同一步应有相同输入与mask计数；发生数值失败则截至失败处保存完整前缀。

本轮无额外训练数据探针：aux_probe_gradient_calls=0，HVP=0，candidate_rollout_for_selection=0。monitor输入处理量和forward单独计，不混入更新tokens，也不从总实验成本中删除。

## 5. 分析规则：效果先行，机制不反推

### 5.1 必须比较所有固定基线

报告每seed、每臂h64/128/256的NLL和目标检测结果，不能只与native比较。

- RAW_QH vs ADAMW_LR100：当前梯度读出混合是否有作用。
- NORM_QH vs RAW_QH：范数匹配是否在本配方下带来额外变化。
- NORM_QH vs三种固定LR：改善是否超出简单LR配方调整。

本轮选择的“最佳固定LR”是三候选网格中的开发/事后最优，标BEST_OF_GRID_DEVELOPMENT，不叫已调到最优AdamW。必须展示全部三条。

可选的CPU companion：LOSO，每次用另外两seed的h256均值挑一个全局LR，再在留出seed比较。只作小样本描述，不当独立大样本确认，也不能覆盖完整逐seed结果。

### 5.2 不确定性

- 逐seed结果必须直接展示。n=3不做夸张总体显著性声明；15臂不是15个独立seeds。
- 评价数据不确定性复用PhaseA的64行连续block paired bootstrap（无doc ID，标BLOCK_PROXY），同一bootstrap样本内各臂配对，用sum/count重算NLL。
- bootstrap默认1000次，CPU；不把数据级SE当训练seedSE。
- h256和差值保留原始sum/count；只报告足以支持比较的有效数字。

### 5.3 有限预算未达标和失败

不得只平均成功者的检测成本。
逐臂报告n_reached/3、n_not_reached、n_numerical_failed、全部个体N_detect。
双方都达标时可以报告paired token ratio；必须同时给eligible pairs数量，不把它作为全样本无偏平均。
如报告min(N_detect,N_budget)的截尾统计，未达标按budget填入仅用于该截尾量，同时标它不是达标时间。
发生NaN/Inf终止该轨迹，保存为失败，不换seed、不自动降低LR重跑。外部OOM/中断与数值失败分开，前者可按完整快照和冻结RNG精确续跑，开销不重置。

### 5.4 效果判定不是“理论全部PASS”

必须分别给出：
IMPLEMENTATION / COVERAGE / FINAL_BUDGET_NLL_EFFECT / DETECTED_TARGET_TOKEN_EFFECT / NORM_MATCHING_ABLATION / FIXED_LR_COMPARISON。
结果状态可以是SCREENING_SIGNAL、NO_OBSERVED_GAIN、MIXED、BASELINE_GRID_BOUNDARY、UNRESOLVED或PARTIAL。

建议约10%检测训练token降低是值得后续复验的工程量级，不是本轮能达到的承诺，也不是物理定律。
若候选只有终点NLL改善而未缩短检测token，报告固定预算质量信号，不硬称达标成本降低。
若只短期过阈而末端明显更差，报告质量/检测成本权衡，不宣称统一改善。
若候选不超过本轮固定LR网格，则不宣称更新结构具有超越LR的优势。
若RAW_QH同样或更好，不声称范数匹配是收益机制。
即使全部正向，仍是已知checkpoint上的开发pilot，不是完整预训练、最终能力、墙钟或能耗结果。

## 6. 最少诊断：仅解释规则是否实际执行

每真实step记录loss_train、lr、clip_pre_norm/coef、global_update_norm、token计数和有限标记；复用已有便宜字段。

只在h={1,16,64,128,256}记录每唯一参数张量的：
- ||r_A||、||r_Q||、||r_N||及同状态范数残差；
- dot(r_A,r_N)、cos/angle（零向量NA）；
- c^T r_A、c^T r_N和差（current_training_proxy，不是monitor贡献）；
- ||theta||、adaptive step norm、完整step norm；
- fallback计数、sum_square/norm定义、临时/持久state bytes。

NORM_QH轨迹上的r_A是它本状态的native-formula reference，不是另一条真实native轨迹的r_A。禁止跨轨迹误说范数被严格控制相同。

不算q/Hessian、attention entropy、噪声张量、预测器精度或真实conditional tracking error。不因为某诊断与NLL相关就声称因果机制证明。

## 7. 实现路径与CPU测试

优先新建独立小模块与runner wrapper，使用现有冻结训练、数据和评价primitives。
建议文件（现场按已有模块布局微调路径，不为命名重构）：
- research/src/fr/d1_direct_readout_v01.py：纯tensor读出与边界处理。
- research/scripts/d1_run_v01.py：preflight/smoke/main，原runner循环最小注入。
- research/scripts/d1_analyze_v01.py：CPU统计与表/图。
- research/tests/test_d1_direct_readout_v01.py：独立参考公式测试。

八项最少CPU测试：
T1 alpha=0走原native路径（包括decay和None/zero）；alpha=.1公式对独立参考一致。
T2 m/v与bias correction只按当前c更新，r_Q/r_N不回写moments；相同外生c序列时三种读出raw moments一致（参数轨迹可不同）。
T3 RAW_QH与nu1=.9、nu2=1、decoupled wd的独立公式一致；不依赖新增安装第三方库。
T4 同状态逐tensor norm匹配、方向与r_Q一致；zero/cancellation/共享参数处理正确。
T5 通过简单反例证明current-batch一阶对齐不保证独立目标下降，不把其写成成功门。
T6 schedule base前缀和tail正确，k=.5/1/2只缩放eta，不改累计步号和floor时点。
T7 sum/count、token口径、两次连续检测、未达标保留；h0共享不得少计/多计模型调用。
T8 故意错误版本（如r_N写回m、每step重置n、valid=slots、norm分子分母反转）至少有对应测试能失败；不把生产helper自比自当独立验证。

native浮点身份可用原代码路径保证；新规则按独立公式设明确FP32容差，不能为了bitwise在新规则上改定义。

## 8. Smoke与main执行

### 8.1 只读准备

读取最近PhaseA config/data contract/scheduler/target、三个原bank states和当前runner，不回溯重审所有A0/B0账。
写config、state/data/arm manifest，冻结alpha、LR网格、monitor频率、阈值、执行顺序和代码版本摘要。
如参数组或恢复语义与上述冲突，报具体阻塞；不能自行更换模型或数据。

### 8.2 Smoke计划40真实updates

使用独立smoke流（99100；若冲突按元数据递增完整块），固定seed2030/start46：
- ADAMW_LR100：diagnostics off/on各8步，共16；
- QH_NORM_A010：diagnostics off/on各8步，共16；
- QH_RAW_A010：8步接口/有限性检查。

六次全库monitor评价：同一公共h0一次，native off/on各h8两次，candidate off/on各h8两次，RAW h8一次。
它们只检验评价/RNG隔离和接口，不以smoke的loss高低调参数或停止科学比较。

off/on必须不改变该规则自己的theta/m/v/step/LR/data/RNG；候选无需与native轨迹一致。
遇到真实代码错误可修复、记录并重新smoke，全部消耗受总cap约束；不得更改方法公式以消除训练不良结果。

### 8.3 Main

15条主轨迹，按§3顺序执行，每条256步。共享h0每seed只评价一次并复用。
不按中途好坏停止臂，不用monitor选每步更新。保存每条h128/h256完整状态（共30个main snapshots），若数值失败保存失败前状态和证据。
无常驻monitor，只用同步日志/必要会话运行方式；不停止或干扰其他GPU工作。

## 9. 新独立预算：所有失败都计入

计算基于真实训练microbatch4×accum2、monitor4096行/eval_batch32；现场只核实际口径，不擅自改变batch或monitor规模。

|项目|计划|硬上限|
|---|---:|---:|
|main真实updates|3×5×256=3840|计入总量|
|smoke真实updates|40|计入总量|
|总真实updates（含失败/重试）|3880|4096|
|训练forward microcalls|7760|随总量，含所有失败|
|训练backward microcalls|7760|8192|
|主monitor整库eval|3+3×5×16=243|计入总量|
|smoke monitor整库eval|6|计入总量|
|总monitor整库eval|249|270|
|monitor forward microcalls|249×128=31872|34560|
|全部model forward|7760+31872=39632|45000|
|额外梯度/HVP/FD|0/0/0|0/0/0|
|真实候选重放用于选择|0|0|
|V_final模型调用|0|0|
|主训练input slots|3840×1024=3932160|按真实batch计|
|含smoke训练input slots|3880×1024=3973120|按实际计|
|monitor input slots|249×4096×128=130547712|按实际计|

所有cap同时约束，不以一个宽松总额替代子cap。预算余额不用于增加alpha、LR点、seed或更换实验。
更新规则没有额外网络梯度，不代表评价免费；本轮频密monitor成本很大，属于研究测量，不声称部署会以相同频率全库评价。
wall只记录OBSERVATIONAL_SHARED_GPU，不设科学硬时间上限；数据loader/eval/setup等能分列就分列，没有则NOT_SAVED，不估算填表。
旧18000范围、PhaseA、PhaseB及诊断账本全部保持原样，本轮独立记录实际run-level成本，不向旧账借余量。

## 10. 报告必须给关键数据，不只给结论

输出目录：outputs/fr_gttr_d1_direct_readout_v01/
最低产物：
- config.json、manifest.json、implementation_checks.json、smoke_checks.json；
- step_metrics.csv、monitor_metrics.csv；
- 固定h完整逐行NLL_sum/count（允许压缩NPZ，带行ID/形状/schema；不强迫大CSV）；
- target_hits.csv、seed_method_results.csv、paired_comparisons.csv；
- readout_diagnostics.csv、resource_log.jsonl；
- SNAPSHOT_MANIFEST.json、SUMMARY.json、REPORT.md、KEY_DATA_FOR_REVIEW.md；
- 最多4张结果图：各seed NLL vs训练input tokens、逐seed固定预算gain、达标/未达标表图、候选与RAW的方向/范数诊断。图不需要一次只挑最好seed。

最终聊天直接展示以下表：
A. 3seeds×5arms配置、真实起点、stream、LR首尾、alpha、样本覆盖。
B. 每seed/arm的h0/64/128/256 NLL sum/count/mean；paired h256差和数据级SE。
C. 每seed/arm的last_above/first/second检查点、input/valid tokens、reached/censored/failed、terminal NLL；V_final_calls=0。
D. 每arm reached/3、全部个体成本、固定预算表现；三种固定LR全列，不只列较差native。
E. 同状态范数匹配最大误差、fallback次数、方向cos、RAW vsNORM；所有量名字明确norm(mean)或mean(norm)和分母。
F. actual/planned/cap，训练/评价/失败、额外state bytes与observational墙钟。

摘要结论最多回答：
1. 该直接规则有没有在本pilot改善真实目标或检测训练token？
2. 它是否超过所测固定LR网格及对应QH_RAW？
3. 当前改动、尺度控制、噪声代价分别有哪些可见迹象？（机制仅有限解释）
4. 哪些结果失败/未达标/不确定？
5. 后续只允许提出一个必要验证，不能自动执行。

## 11. 不做的事情与下一阶段条件

不再以预测器或HVP门作为本轮前置，不修S3，不生成LR候选评分，不通过调整eta去“避免非线性”。
不声称本方法能修复S3的局部模型，不把同状态方向改善当长期收益保证。
不扩到26M/100M/1B，不增加batch/epsilon/beta/clip搜索，不重新校准ell。
不覆盖旧输出，不commit/push、不打大型审计tar、不安装新依赖、不改Kimi接入、不重建实验平台。

若有开发信号，下一轮可在另一起点、新的独立训练seeds/streams及重新匹配的有限LR调参预算下确认，再使用预先规定的final评价。
若没有收益，记录该具体规则和配置的负结果，停止这条候选；不由一次pilot否定全部直接更新研究，也不堆叠多个临时补偿项。

## 12. 参考资料（仅为已知设计关系，不是本项目效果证据）

- Ma, J.; Yarats, D. Quasi-hyperbolic momentum and Adam for deep learning. ICLR 2019. https://arxiv.org/abs/1810.06801
- QHAdam 官方实现和QHAdamW wrapper: https://github.com/facebookresearch/qhoptim/blob/master/qhoptim/pyt/qhadam.py
- Agarwal, N. et al. Disentangling Adaptive Gradient Methods from Learning Rates. 2020. https://arxiv.org/abs/2002.11803
- Loshchilov, I.; Hutter, F. Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101

这些原始来源表明混合当前梯度/EMA和范数grafting均有先例；本轮不提出首创声明。将网络实验收益与机制解释、通用收敛和最终节能分开。
