# OpenCode 第三轮执行任务：T2b / T3-mini 目标价值前提实验

日期：2026-09-29。配套研究依据：`MUON_EFFECT_FIRST_RESEARCH_OUTLINE_v3_CN.md`。

## 0. 本轮要直接做完什么

**完成必要勘误后，在同一任务内完成小型全状态非线性回放、候选锁定和真实分支终点评价。不要再用一整轮只生成平台审计文档；不要因为单点JVP失准先去补长理论。**

这是用户对上一轮建议的继续执行：先建立目标相关效果证据，再决定理论/方法怎么深入。物理只是解释和设计对照的工具，不设“必须发现热力学现象”的门。

新增授权口径与旧T2a不同：本轮允许全参数梯度采集、开发目标读出与少量真实参数更新；上限见§9。旧实验输出、原始数据、固定配比M2裁决不修改。

开始前读取：
- 配套v3研究大纲；
- `ROUND2_AUDIT_AND_RESEARCH_ASSESSMENT_CN.md`（交接包附带）；
- 当前仓库已存在的 T2A_PROTOCOL、T2A_REPORT、configs/t2a_scale_v1.json；
- 实际生产模型、训练、路由、checkpoint与开发评价入口。

复用 `muon_temporal_response/`，新增本轮模块即可。不另建通用训练框架；不改安装环境或旧兄弟项目；不移动/删除他人未提交文件。研究文件缺失或资产不匹配时报告真实阻塞，不编造结果或换锚点。

---

## 1. 固定研究对象与资产

### 1.1 唯一锚点

沿用T2a：
`m1_muon_lr2.0_wd0.02_p0.5_s2030/final_state.pt`

T2a记录SHA：
`4f68ce2423d663f87fbb44f051abe92d6ccd09118ca42a80261497ee13f095e4`

实际路径从现有配置读取。应为MiniMind 30,025,216参数，step2048，Muon56矩阵及辅助AdamW状态完整。不是充分调优最优点，只是固定诊断起点。

### 1.2 真实续训配方

- 恢复全部模型、buffer、Muon和辅助AdamW状态及计数器。
- 主Muon beta=.95，原Nesterov/NS5/系数/归一化/shape scaling/dtype不变。
- **常量LR平台**：每参数组固定使用checkpoint保存的末步LR。主Muon应为0.001，不能误用配置峰值0.01。辅助LR读checkpoint原值并写入本轮配置，不手填猜测、不用库默认值。
- WD、辅助betas/eps、原decay mask沿用checkpoint与生产配置；不新增warmup、不延长原cosine公式。
- 全程同一组LR/WD，不重新调参；每步仍正常推进应推进的optimizer state/step。
- 4个micro-batches ×32窗口，256 slots/window；每实际update预期32,768 slots /32,640有效targets。
- 不改变batch成员、tokenizer、mask、loss归一化、global clip或数据域。

既有数据来源/去重和终态选择局限保留，不将本轮内部筛查描述成干净的最终外部确认。

## 2. 必要勘误（仅使用已有数据，不延长为新审计项目）

1. 在另存勘误中说明5+5均值差不能独立证明非平稳；写出IID抽样参考点效应，不宣称已证明IID或平稳。
2. 区分误差方差比、RMS比、向量残差；T2a“中尺度30%误差”不得继续写成单纯幅度误差。
3. 源码需要非clamp分支时检查H_ref范数；不把toy死区或某个余项数值作为通用门。
4. 能直接读取服务器原矩阵时导出六个10×10 Gram及范数即可；缺失则登记，不补跑T2a作为本轮前提。
5. 重用已更正的总体协方差/epsilon²/均值偏移平方口径。

产出 `docs/ROUND2_ERRATUM_TARGET_SCOPE.md`，无需重跑整套T0/M0或48条基线。

## 3. 数据：两个10-batch窗口＋一段共同128-batch收尾

源训练stream的完整optimizer batch定义保持不变。

- W0：原T2a十个batch，即生产manifest零基切片 `[2048:2058)`，优先复用已有确切成员清单。
- W1：同生产生成规则的后十个batch，切片 `[2058:2068)`。
- C：同规则随后的128个batch，切片 `[2068:2196)`。

验证三个块的窗口ID不重叠且在现有训练域可用范围内；W0必须与旧T2a manifest逐成员匹配。每batch仍64A＋64B，不为制造顺序效应形成单域块。

每条真实分支都从同一step2048状态恢复。源数据切片索引只是样本身份，**不能误当optimizer step或LR时钟**。全部分支真实执行138步，optimizer逻辑最终计数相同。

### 3.1 预定八个排列

每个窗口独立生成八个10元素均匀随机排列：
`numpy.random.Generator(numpy.random.PCG64(883000 + 100*b + k)).permutation(10)`，
b∈{0,1}，k∈{0,...,7}。

运行前生成、检查唯一性并落盘。极少数重复发生时，用预先规定的“该种子+10000后继续生成直到唯一”规则处理，发生在任何结果读取前并登记。

pi0是k=0的预定shuffle参考，不由结果选取。八条均属于正常随机shuffle；没有人为最差排列。八个候选不是八个训练种子。

## 4. 目标划分：只使用旧开发池，原confirm继续封存

需要两个用途不同的目标：
- Q_select：计算目标梯度和虚拟轨迹终点NLL，选择候选；
- Q_check：只检查事先锁定的选择与实际分支结果。

使用现有dev文档/线程清单，按原语料中线程/翻译关联组的全局group ID进行划分。所有相连文档必须同组。规则：
`sha256(('T2b_Q_v1|' + group_id).encode()).digest()[0] % 2`
为0进入select、为1进入check。

在组级划分后按原tokenization/packing规则形成两个目标集，各域最多使用排序后前512个可用完整窗口；若少于512使用全部，记录文档/组ID、真实数量、token counts，不用train或confirm补齐。任一目标域完全为空或无法保证组隔离时BLOCKED，不能切成跨线程窗口假装解决。

不因指标好坏重切目标。Q_check与Q_select互斥，但旧dev已参与旧M1/M2，所以只能叫**本轮选择持出的开发检查**，不能称从未使用的final。

两个目标都固定：V=0.5*(sum_A/count_A)+0.5*(sum_B/count_B)。报告两域分项；不能用拼接后的总tokens加权替代域等权。

目标梯度不clip、不参与训练、不写回optimizer；按域sum/count正确求导。计算/评价恢复模型模式并隔离RNG，不污染生产数据流。

## 5. 全状态冻结梯度非线性回放

### 5.1 只在theta*求训练梯度

对W0、W1和C的共148个唯一batch，在同一个theta*计算完整生产梯度：
清grad →4micro前向反向/原loss权重 →所有unique参数global clip一次 →捕获post-clip G。

**全参数都需要**：Muon56矩阵＋所有辅助AdamW参数。原T2a仅六矩阵不能构成全模型目标预测。可以校验W0六矩阵旧数据，但不能用旧局部数据冒充其他参数。

模型参数在采集时不改变。W0/W1共20个全梯度可缓存；C的128份梯度尽量逐份产生后送给全部虚拟状态，处理完释放，避免存一个17GB全量尾bank。

### 5.2 每条候选正常递推状态

初始化16份逻辑独立的虚拟模型/完整optimizer状态，均从同一锚点开始。

每个窗口先按八种候选顺序送入其10份G，然后全部送入同一C的128份冻结G。**排列内不重置M，不重置辅助m/v/step。**

原生Muon和辅助AdamW更新照常作用于虚拟参数，包括WD。不得将G引用原地修改污染另一条候选；必要clone后赋grad。已post-clip的G不再二次clip。

“冻结”只指G是在theta*计算；虚拟模型不是用于下一个batch重新求G。原生产安装函数、dtype和route保持一致。

内存采用流式/CPU卸载是允许的；不得悄悄删除辅助组、变低精度或反复重算尾梯度不计费。维护16个独立状态并非要求16份Transformer激活。

### 5.3 数学对照与选择

一次在theta*计算全参数q_select。
记录每候选step10和step138的虚拟参数位移、Muon/辅助组分项、M尾差。

在step138定义：
S_fin(pi)=V_Qselect(virtual_theta_pi)。
S_lin(pi)=q_select dot (virtual_theta_pi-virtual_theta_pi0)。

主候选 pi_F=argmin S_fin；副候选pi_L=argmin S_lin，只作较简单读出的对照。
同一分数差≤1e-6视为并列，选最小k。所有分数锁定后生成 `selection_lock.json`。

有限V_select本身只用于候选选择，不作为效果成功指标。qdot也不是终点保证，不叫热量。可用合成小向量核对Nesterov求和和WD加权位移恒等式，无需重新扫描toy矩阵。

本轮不使用固定H_ref Jacobian筛选候选，不计算完整训练HVP或超梯度。

## 6. 真实分支：必须重算梯度，不用冻结G替代训练

完成所有虚拟分数和候选锁定之后，执行16条真实分支：
每条重载共同完整锚点 →候选10batch顺序训练 →公共C的128batch正常训练。

每一步都在当前theta计算新梯度，全参数global clip，Muon＋辅助AdamW实际step。

在step10和step138保存Q_check NLL。step10只诊断，不能据此改变公共收尾、挑新排列、扩LR或提前宣布成功。每条的输入多重集、batch成员、共同尾顺序、LR/WD、更新数完全一致。

额外允许重复W0/pi0完整138步一次，核对模型/评价重复性；该重复不能计成独立样本。原生BF16自然流程不换成FP64。

若某条出现真实配方NaN/Inf，登记recipe divergence，不能改参数后当作同一run；与文件缺失/代码异常等technical failure分开。虚拟轨迹失效也单独登记，不能据此删去真实候选让分数变好。主选择全部无效时标记selector unavailable，而非强选NaN。

无需等待S_lin/S_fin预测“看起来通过”才执行真分支。这一步就是要区分有用效果是否存在和预测是否有用。

## 7. 统计与裁决

对窗口b、时点h∈{10,138}，令v[b,k,h]为实际Q_check NLL。

主收益：
G_F[b,h]=mean_k v[b,k,h] - v[b,k_F,h]，改善>0。
副收益G_L同理。另报告v[b,0,h]-v[b,k_F,h]。

事后空间描述：
G_pool[b,h]=mean_k v[b,k,h]-min_k v[b,k,h]。
这个min来自Q_check，是**事后候选库最低值**，不能当作已选择方法、总体性能上界或独立成功。

报告S_fin、S_lin与实际v的排序、差值误差，以及前缀/收尾的收益。秩相关仅描述，候选数8、窗口2、同一训练种子，不给种子层面显著性。

可做按文档/线程cluster的配对评价bootstrap，明确只反映固定训练结果的评价抽样误差，不替代训练种子误差；不是必需的新工程模块。

### 7.1 预先固定的研究筛查阈值

`delta_screen=0.001` nats/有效target，仅用于本轮投入决策，不是普适收益标准。
重复run得到的NLL差记为e_repeat，分辨下限delta_num=max(1e-8,10*e_repeat)。运行后报告原值；不能降低delta_screen以求通过。

本轮判定：
- `TARGET_SELECTION_SIGNAL`：两个窗口G_F[b,138]均≥max(delta_screen,delta_num)，且主候选均不劣于预定pi0（容差delta_num）。这是局部开发信号，不是论文方法已成立。
- `EFFECT_WITHOUT_RELIABLE_SELECTION`：未满足上一项，但两窗G_pool[b,138]均≥该阈值。说明当前候选池中有事后差异，当前诊断选择未建立；不把oracle最低值报成方法收益。
- `NO_USEFUL_SIGNAL_THIS_SETTING`：完整有效数据不足以满足上述目标空间判断，或只有step10优势而收尾消失。保留单窗口/幅度不足详情，停止该版本，不据此否定所有Muon顺序机制。
- `BLOCKED/INCOMPLETE/RECIPE_DIVERGENCE`：未完整执行、数值或资产失败；不当作数学假设被否定。

若pi_L有信号而pi_F无信号，单独列为“辅助读出探索性结果”，不事后将pi_L改成注册主方法。

## 8. 禁止重新扩展研究范围

- 不为候选分数好看缩小自然扰动；不拟合死区掩码。
- 不修改beta、LR、WD、NS次数、token预算或数据域。
- 不新增完整T2的四状态1024batch矩阵；不补建父轨迹。
- 不使用confirm；不在看到Q_check后调分数、选窗口、增候选。
- 不将“两窗口”称两独立训练seed。
- 不将额外forward/backward、回放、目标查询视为免费；本轮昂贵诊断不是在线算法。
- 不输出已证明FDT破坏、真实温度、熵产生或能耗收益。

## 9. 新增预算（不沿用旧T2a的零更新约束）

本轮总上限：**3 GPU-hours**，失败尝试、编译、梯度采集、虚拟回放、目标梯度和评价都计入。两卡各运行1h是2 GPU-hours。CPU数值检查最多30min；新输出最多8GiB。

预定工作量：
- 148个唯一全参数冻结梯度batch = 4,830,720有效targets的梯度计算；
- 16×138=2208次虚拟全optimizer更新；
- 16×138=2208次实际更新 =72,069,120有效训练targets；
- 允许一次138步重复，总实际更新≤2346 =76,573,440有效targets；
- q_select与evaluation token消耗另列，但在同一GPU预算内。

原始source slots也单列，不能继续把32,768 slots标成有效targets。

达到上限立即保存进度并停止，不能私自加预算、减模型、删辅助组或切换另一个短协议。

## 10. 最少回归测试与输出

只补与本轮有关的测试，不重复整个平台初始化：

1. 两窗口/公共尾manifest、完整batch成员、目标group划分相互隔离。
2. 一步从theta*的原生真实更新与使用同一G的虚拟更新一致（模型、Muon和辅助state）。
3. 候选之间G/状态无别名污染；每条初始state相同。
4. 全参数唯一、全局clip一次、sum/count及两域权重正确。
5. constant group LR来自checkpoint，主Muon=.001；全步数/WD权重/计数器正确。
6. selection_lock写完前Q_check不可访问；confirm全程不可访问。

建议文件（按已有结构落地，不要求重构）：
- `docs/T2B_TARGET_SCREEN_PROTOCOL_v1.md`
- `configs/t2b_target_screen_v1.json`
- 一个接线生产循环/全状态回放的运行脚本
- 一个分析脚本和最少测试
- `outputs/t2b_target_screen_v1/`

关键结果：
- source/checkpoint/config hash、batch/目标manifest；
- shadow_scores.csv、selection_lock.json；
- actual_objectives.csv（逐域sum/count/NLL、V_Q、step10/138）；
- prefix_tail_diagnostics.csv（记忆尾差、位移及读出分项）；
- target_gain_summary.json、成本台账、运行日志；
- REPORT_CN.md：有效结果、失败/未执行项、主裁决及后续建议。

无需保存所有候选所有步的梯度/激活。优先缓存20个前缀G、流式公共尾，实际终态保存模型或必要state；不突破8GiB。审阅包目标≤10MB，仅表格、配置、代码、报告、小样例，不带完整checkpoint/语料/全梯度。

**完成该轮后停止，等待用户审阅。即使结果好，也不自动进入新种子、大模型、记忆核改动或控制器训练。**
