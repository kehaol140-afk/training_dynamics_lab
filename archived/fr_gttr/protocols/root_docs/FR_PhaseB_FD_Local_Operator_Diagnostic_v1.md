# FR Phase B：局部FD算子失配的定点诊断 v1

日期：2026-09-23。执行端：现有 Claude Code + Kimi K2.7。
本文件是新诊断建议。用户转发本文件和启动指令后，仅执行下述范围；不代表原Phase B已获恢复许可。

## 0. 问题与不变项

用户报告原Phase B v3在S3（seed2031，原bank start82）首方向失败：fd_rel=0.7061、fd_cos=0.7282，冻结门fd_rel<=0.20。S0/S1/S2分别报告0.0011/0.0660/0.0206。主预测没有封存，MAIN真实分支未开始。

本轮目标：区分状态/随机性/还原实现、参考数值误差、差分尺度误差与方向上的真实局部非线性。不是让S3重新得到PASS。

- 原GATE_STOP和全部旧文件保持不变。
- 不调原阈值；不删S3，不换SIM批，不随机换方向。
- 不运行Phase B主实验，不构造动作评分，不评价V_control/V_monitor/V_final。
- 不加新的物理变量、HMM、memory kernel或控制器。
- 墙钟只记录OBSERVATIONAL_SHARED_GPU，不因共享GPU等待设置科学失败门。
- 不重做历史审计，不commit/push，不改模型接入配置，不干扰其他任务。

## 1. 固定对象

只测两个状态：
1. S3：失败点，seed2031/start82；使用原manifest对应SIM197009（若旧配置实际不同，以原manifest为准，明确记录）。
2. S1：同start82的通过对照，seed2030；保持其原SIM197007。

S1/S3数据流不同，所以它们只是复现/数值流程对照，不足以分离“纯checkpoint效应”和“SIM batch效应”。不在本轮增加交叉换batch实验。

只研究首方向v1。v1来自原参考native位移s_ref0归一化，必须固定、detach，并跨差分尺度和数值对照保持同一坐标方向。不继续构建rank4。

原差分定义：
G_SIM是固定2个microbatches、按真实accum=2合成的未裁剪梯度。
delta0=1e-3*max(norm(theta0),1)。
y(h)=[G_SIM(theta0+h*v1)-G_SIM(theta0-h*v1)]/(2*h)。
fd_rel(h)=norm(y(h)-y(h/2))/max(norm(y(h)),norm(y(h/2)))。

参考FD应按v3使用FP32参数、关闭autocast的FP32 forward/backward。不能先假定故障来自BF16；必须核实实际dtype/后端。

## 2. D0：先0 GPU核查已有文件

读取原v3、token-primary amendment、现场config、构建代码、S3失败记录以及已有bundles。不要只读报告。

输出：
- 原失败行及fd_rel/余弦的分子分母定义；从现有向量或充分统计独立复算；没有则NOT_SAVED。
- theta0/v1范数、delta0、s_ref0范数、SIM IDs、mask/reduction、参数唯一映射和tied weights处理。
- 原raw/clipped梯度路径，确认FD作用在未裁剪G_SIM上。
- +h/-h两端是否从同一theta0重建，不使用逐次加减造成累积漂移。
- 每次G_SIM是否清空梯度、恰好累积同两个microbatches；返回向量是否clone而非后来被覆盖的.grad引用。
- 参数、buffers、训练模式、随机数/掩码、游标和None/zero语义恢复。
- 真实FP32/TF32设置、attention后端、fused kernel、autocast和实际梯度dtype。

若已有文件直接显示实现违反协议，先列证据；只允许修复这一具体问题，保存diff，再按下面固定诊断验证。不要一边修复多个无关模块一边改变算术路径。

## 3. D1：原路径定点复现

每状态完全独立恢复原bank与对应SIM输入，重建或读取并核验固定v1。

- theta0上的G_SIM重复两次。
- 在原delta0和delta0/2上各执行两次完整FD，每次恢复同一参数、buffer、随机性和梯度累积状态。
- 尽量包含一次端点顺序反转，检查+/-顺序依赖。
- 每次写入实际实现的参数偏移，检查其与期望±h*v1的差别。
- 不只报标量loss相同；比较完整梯度或可复核的差向量统计。

重复不一致时标REPEAT_OR_STATE_INCONSISTENCY，先定位状态和数值后端，不直接进入“物理非线性”结论。

## 4. D2：固定尺度诊断，禁止逐结果调参

在原FP32路径，对每状态同一v1、同一SIM，用固定集合：
    h/delta0 in {2, 1, 1/2, 1/4, 1/8}。
已经复现的delta0、delta0/2可在状态/版本/配置完全一致时复用。

保存所有点，包括失败、零信号和非有限点。不得选择一个最好点替代原v3结果。
报告相邻尺度差与余弦，并在有3个有效连续尺度时给出差值缩放比。该比例只作描述，不以它强行判断理论阶数。

增加两个与实际位移有关、事前定义的检查点：
    theta0+s_ref0
    theta0+exp(0.05)*s_ref0。
只计算冻结SIM的梯度变化，不写回真实训练状态。
若已有对应端点可复用则复用；这些是局部适用范围诊断，不是控制器/真实收益实验。

每点记录按层的相对扰动，特别是零范数块、微小参数块和未改变坐标比例。全局1e-3相对范数不能代替每层扰动尺度。

## 5. D3：独立导数/数值参考

优先使用已有可靠的autograd HVP原语；只求一个方向，不构造全Hessian。
对同一标量SIM loss、同一参数坐标、同一固定v1，计算
    y_AD = grad_theta( dot(grad_theta L_SIM, stop_gradient(v1)) )。
保持未裁剪梯度、相同loss权重、相同参数去重与随机输入。
不能对v1的构造反向求导；不要误算另一个对象。

两个预定义算术路径：
A. 原协议FP32路径原样复现；记录实际矩阵乘精度与attention kernel。
B. 隔离副本上的高精度参考（FP64、可二阶微分的等价math算子；尽可能复用同一代码）。

B路径保留原token/mask/损失/模型结构/训练模式和固定随机掩码。FP64仅是诊断参考，不改变真正训练配方。
若不得不改变kernel，分别核对base loss、base gradient与随机mask，清楚标REFERENCE_BACKEND_CHANGE。改变计算函数或不能固定同一掩码时不能把差异全归因dtype。
若确有必要分离dtype与kernel影响，允许一条匹配B后端的严格FP32参考；不增加其他架构/版本搜索。

在可用参考路径上对同一5尺度做FD，并与该路径的y_AD比较。
正/负端点都从同一base构造。AD未获支持时记录原始错误；允许一次最小等价math-backend参考尝试，不能重写网络、升级依赖、用全Jacobian或全Hessian绕过。

必须明确：AD也是指定可微参考图的导数，不是逐比特浮点程序的数学导数，不自动是真实训练收益的答案。

## 6. 直接打印的关键数据

### A. 身份与扰动
state_id/seed/n0/SIM_ids/path_id/param_dtype/grad_dtype/autocast/TF32/attention_backend/
||theta0||/||v1||/delta0/||s_ref0||/
delta0/||s_ref0||/delta0/(abs(exp(.05)-1)*||s_ref0||)。
分母为0标NA，不偷偷加常数制造有效比值。

### B. FD数值（每state/path/scale）
h、Lplus/Lminus、||gplus||、||gminus||、||gplus-gminus||、||y(h)||、
fd_rel邻尺度、fd_cos、v1^T y(h)、repeat绝对/相对差、有限性、
实际参数偏移误差、正负端点不对称性、还原状态误差。

保存可复算的norm²/inner_product，不只保存比值；归约可FP64，但注明梯度求值本身的精度。

### C. AD与模型残差
||y_AD||、FD对同path AD的absolute_error、symmetric_relative_error、cos。
目标导数或分子过小时列RAW_SMALL_SIGNAL，不用人工噪声floor把门抬过去。
在两个实际位移检查点列：真实SIM梯度变化范数、
用原delta FD/对应AD预测的残差；仅检验该局部方向，不外推H64。

### D. 层定位
原失败delta/half差向量平方范数中top5参数块占比，以及原始分子分母。
只作定位，不直接宣称该层导致非线性或学习问题。

### E. 成本
真实optimizer update、虚拟初始化算术、
普通梯度micro-forward/一阶backward、AD二阶backward、FD方向数、
SIM唯一数据量与重复处理量、峰值显存和observational_shared_GPU wall。

## 7. 成本边界：新诊断单列，不借用旧Phase B余额

- 真实optimizer updates=0；不允许推进训练。
- V_control/V_monitor/V_final网络调用=0。
- 全部模型micro-forward<=240。
- 全部autograd backward/VJP micro-call<=320（包括AD-HVP的一阶和二阶遍历，分列）。
- AD-HVP方向求值<=6（每个作用于一个effective-batch loss的方向算一次；microbatch细项另记）。
- 原/参考梯度、FD重复、失败尝试和后端检查全计费；参考改变不得隐藏成本。
- 不设共享GPU墙钟科学硬门；外部OOM/中断如实记PARTIAL，保留计数。

这些上限只用于上述两状态、首方向和固定尺度诊断；有余额也不能增加样本或恢复主实验。

## 8. 结论决策

允许INCONCLUSIVE，不强制四选一。
- 若重复/顺序/恢复不一致：优先实现或随机状态问题。
- 若参考FP64/匹配后端的FD/AD一致，而原FP32不一致：数值路径限制得到支持；仍区分kernel与dtype。
- 若参考精度提高后原delta仍有偏差，而更小固定尺度趋近AD：原尺度超出局部可用范围得到支持。
- 若FD/AD自身可重复，但自然步长处梯度变化近似仍差：该实际位移方向上的局部线性适用范围受到限制。
- 若只有两种参考路径函数不同，或所有信号都很弱：保持未分辨，不叫地形非光滑/混沌/跨谷。

不要用“S0–S2过门、CPU测试通过”推导“S3一定是物理局部性质”。
不要把FD/AD一致称为64步预测、动作选择或tokens-to-target通过。

## 9. 产物与停止

新增独立目录 outputs/fr_phase_b_fd_diagnostic_v01/，仅：
DIAGNOSTIC_PLAN.json、REPORT.md、KEY_DATA_FOR_REVIEW.md、
fd_scale_diagnostic.csv、ad_reference_comparison.csv、repeat_restore_checks.csv、
block_perturbation_summary.csv、costs.json、必要方向/参考统计与最多2张图。

主程序只做一个小诊断脚本加必要参考测试，不建平台、不打大型tar、不重复审计历史。
最终聊天直接呈现A–E核心表、哪些解释受支持/未分辨、下一步最小修改建议。
若建议换差分规则/AD-HVP/精度，给出新版本方案，不回写原v3、
不自动生成主预测、不恢复MAIN、不进入Phase C。

文献仅用于数值方法参考：PyTorch官方 numerical_accuracy、gradcheck、autograd.functional.hvp；按现场安装版本使用API，不安装或升级依赖。
