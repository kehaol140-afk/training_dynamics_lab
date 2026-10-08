# Muon 第二轮执行任务：T0 勘误 + T2a 真实尺度校准

版本：v1；日期：2026-09-28。

**授权范围：按本文件完成零训练后处理修正和一项有界真实尺度校准。通过前置测试后可以执行本轮，不必每完成一小项再次等待。不得启动完整 T2、T3/T4，不得补建训练轨迹。**

本文件是根据用户已认可的独立审计制定的新执行计划。其中单状态、10 个 batch、5+5 分割、扰动幅度与成本限额是本轮设置；不回写为旧 T0 的预注册内容。本任务不是新优化器开发。

## 1. 依据、项目与工作方式

先读随附 `audit/ROUND1_AUDIT_REPORT_CN.md`，并对照服务器已有：
- `docs/ROUND1_PROTOCOL.md`、`docs/T0_REPORT.md`；
- `docs/IMPLEMENTATION_ALIGNMENT.md`、`docs/ASSET_AND_COST_REPORT.md`；
- `docs/T2_PROTOCOL_DRAFT.md`；
- `outputs/t0_round1/` 的原始 NPZ/JSON/CSV；
- 原 Muon 研究大纲和第二版设计稿（若已在项目中）。

在现有 `muon_temporal_response/` 继续，不另建通用平台。路径以服务器实际位置为准，不猜测不存在的目录或 checkpoint。

原 T0 协议、原始数组、原报告、旧 Muon/M2/GTTR/P4 实验和安装环境只读。允许修正当前新项目的后处理代码，并新增本轮脚本与测试；修改前保存代码 hash/版本。所有更正另写 erratum 和 corrected 输出。不要覆盖旧结果、批量重跑旧实验、清理他人的工作区或更改安装版 Muon。

独立审计是待复核证据，不是必须逐字复制的答案。若原文件与审计不符，保存最小复现与差异；不要为对齐数字改输入或统计定义。若该差异影响后续数学正确性，停在修正阶段报告。

优先级：正确得到可复算结果 > 必要测试与台账 > 文档外观。不要新增审批系统、通用调度框架或重复整仓库审计。所有说明和最终报告用中文。

## 2. 阶段 A：零训练修正（先完成，后进入校准）

### A1. 统计修正

依据原始数组独立复算后，修正下列问题：

1. **总体协方差分母**：64 个循环相位是完整有限总体。`phase_stats` 的 covariance 与 A 均用除以 n_phase 的口径，不使用 n_phase-1。
2. **幅度单位**：单位扰动的预测 covariance 乘 `epsilon_perturb**2` 后，才能与绝对 covariance 比较。单位系数字段统一带 `_over_eps2`，不能再次给已缩放数据乘幅度。
3. **均值—方差分解**：
   - `A = mean_t mean_phase ||D - mean_phase(D)||_F^2`；
   - `B2 = mean_t ||mean_phase(D)-D0||_F^2`；
   - `total_mse = mean_t mean_phase ||D-D0||_F^2 = A+B2`。
   `mean_t ||bias_t||` 及其平方可以另存，但不能替代 B2。D0 必须来自同映射、同数值路径的零扰动基线。
4. **对比量**：A1、A2 各自误差与 Delta_A=A2-A1 的误差、符号分开。保留绝对误差、相对误差和近零标记，不能只报告单臂误差。
5. **量化归因**：检查真实进入 NS 的 H，而不是把 G 的 cast 统计写成 H。对相同已捕获 H 比较 FP64 与原生输出。
6. **原生数值误差**：令 Dr=Phi_FP64(H)、Dn=Phi_native(H)、q=Dn-Dr，以一致的样本与中心化定义复算 `A_n=A_r+A_q+2*cross`，同时记录均值差；不假定 q 独立或零均值。
7. **报告口径**：纠正脉冲绝对/相对误差、反转尺度、FP64 脉冲“变号”、原生与 FP64 同名范数字段的口径不一致，以及 arm 与 Bs/Ba 单通道混用等问题。全部文字从对应数据生成。
8. **归一化边界**：norm+eps 与 clamp(min=eps) 的导数是不同公式。修正文档中“光滑分支导数相同”的错误；当前主分析只绑定安装版 clamp 映射。n=eps 不作为无条件可微点。

不能把审计数值硬编码进分析器。主表若与旧值一致，明确写“哪些字段未受影响”；不把少数字段错误扩大为全部实验无效。

### A2. 回归测试与输出

至少新增以下测试：
- A == mean_t(trace(cov_t))；
- absolute_A_pred == mean_t(trace(absolute_cov_pred_t))；
- total_mse == A+B2；
- 同 H 数值误差交叉项分解恒等式；
- 零扰动、零分母、正负差值、绝对/相对脉冲误差；
- FP64 与原生数据经同一个统计入口采用相同定义。

数值恒等式的 FP64 单测初始使用 rtol=1e-9、atol=1e-14；这是计算一致性容差，不是科学效应的通过阈值。失败先定位，不能为通过而静默放宽。

保存 `docs/ROUND1_ERRATUM.md` 与 `outputs/t0_round1_corrected/`。复用原始数组，不再运行原来的全 T0 GPU 矩阵。错误修正前后、源数组 SHA、测试和成本必须可追踪。

## 3. 阶段 B：T2a 单状态真实尺度校准

### B1. 本轮问题及边界

只回答：**当前真实 checkpoint 的自然梯度变化处于什么尺度？在相同真实 H 上，FP64/FP32 有限 NS 与原生 BF16 NS 的误差、有限幅度响应和可分辨性如何？**

这是冻结参数、共同初始 optimizer state 下的一步输入—输出校准，不是10步续训，也不是完整时序 bank，不检验终点 NLL。

本轮禁止：
- 任何 model/aux optimizer 的 `step()`，训练续跑、父轨迹生成、其他 checkpoint 搜索；
- 访问 dev、eval、confirm 评价集，或开展验证 NLL 评估；
- 改 LR、momentum、NS 次数、数据处理或全局 clipping 规则；
- 把原生路径替换成 FP32/FP64 后仍称原生；
- 根据看到的响应挑层、挑 batch、增加扰动幅度；
- 移植 T0 的 4e-3 死区、0.2%误差作为普遍阈值；
- 新增完整 Jacobian、Hessian、d×d 大协方差或低秩矩阵替代原矩阵执行 NS。

本轮不加入新的坐标旋转扫描。该对照保留为后续候选，不挤占当前真实尺度校准预算。

### B2. 唯一 checkpoint 和矩阵

按旧 run 配置与实际文件定位唯一锚点：
`m1_muon_lr2.0_wd0.02_p0.5_s2030`
即 MiniMind 30M、Muon reference LR=0.01、WD=0.02、p=0.5、seed2030、step2048 的终态。

此处 LR 是用于识别来源轨迹的 reference LR，不是声称 checkpoint 末步 LR 仍为0.01；本轮无参数更新，也不设置新 LR。

核对 model/config/hash、完整 optimizer state、真实 EMA momentum buffer、cursor、RNG。锚点不存在、缺状态、SHA不符或源语义不一致时，标记 BLOCKED；不要自动选另一个更好状态或重新训练。

六个矩阵固定为以下含义与维度，模块名前缀可以按实际模型解析并记录，不能替换层：
- `model.layers.1.self_attn.q_proj.weight`：512×512；
- `model.layers.1.self_attn.k_proj.weight`：256×512；
- `model.layers.1.mlp.down_proj.weight`：512×1664；
- `model.layers.6.self_attn.q_proj.weight`：512×512；
- `model.layers.6.self_attn.k_proj.weight`：256×512；
- `model.layers.6.mlp.down_proj.weight`：512×1664。

采用人类第2、第7层，即零基索引1、6。形状或 Muon 路由不符即报告。六矩阵合计 2,490,368 个元素，代码实算并检查。

### B3. 先冻结 config/manifest，再采集10个batch

从锚点保存的 train cursor 后，按原域内顺序和原组批逻辑取接续的10个完整 optimizer batches。新建诊断 manifest，不修改旧总步数/schedule/config。

每batch=128个256-slot窗口；micro_batch=32、accum=4；p=0.5 时每batch A/B各64窗口。保留生产EOS/BOS、mask、模型模式、前向精度和随机源约定。使用完整train池的生产处理，不改为只对六层反向。

仅用训练数据；10个batch的实际window ID、成员、顺序、域标签和RNG映射事前封存。组内与组间重复检查按已有文档/thread规则记录；前5和后5的诊断输入分区不得重复使用相同窗口。若无法实现既定取样，报告阻塞，不另搜“更明显”的batch。

写 `configs/t2a_scale_v1.json` 和 `docs/T2A_PROTOCOL_v1.md`，填入真实路径、SHA、batch manifest、下面的固定数值与预算。字段齐全且阶段A通过后自动执行，无需再等待一次形式审批。

前5个batch为校准输入；后5个是未用于构造参考点的检查输入。它们不是未见训练种子或独立最终测试集，不作显著性推断。

### B4. 只计算梯度，不改变模型或生产状态

每个batch先清空梯度，完成4个micro-batch的生产归一化反向；随后在所有有梯度的unique参数（Muon和辅助AdamW组）上，执行生产相同的global clip，阈值1.0。

保存剪裁前全局范数、实际剪裁系数、targets计数、六矩阵的clip后 G_i。不能只对六矩阵局部裁剪。

按实际labels计数，预期为：
- 每batch 32,768 input slots；32,640有效next-token targets；
- 10个batch共327,680 slots、326,400有效targets。

若实测不一致，先定位mask/归一化差异，不直接修改预期使其过关。train loss仅作梯度采集诊断，不能当作验证性能。

模型参数与buffer、原优化器状态在采集前后做内容hash检查；不调用任何生产optimizer.step，不推进训练scheduler，不覆盖旧checkpoint/cursor。新采集过程中允许诊断RNG和独立cursor推进，但不得写回旧实验。

### B5. 本轮H定义：每个batch从同一个M_*出发

对每个预选矩阵，将checkpoint内的真实动量复制为 M_*。每个batch都独立计算：
```
M_i = beta*M_* + (1-beta)*G_i
H_i = (1-beta)*G_i + beta*M_i
```
实际执行使用与生产一致的fp32 lerp次序。beta从绑定配置读取并核验应为0.95。数学检查为 H_i=beta^2*M_*+(1-beta^2)*G_i。

**本轮禁止把M_i递推给下一个batch。**这样的共同状态设计用于校准谱映射；10个样本的离散性不能称为沿真实训练时间的协方差。完整时序回放留给完整T2。

使用前5个H的FP64均值，再一次转换为FP32，得到每矩阵共同参考 H_ref；保存参考值/hash和转换方法。后5个不参与修改参考点、选择层或调整规则。

第一阶段只输出前5个的校准结果并写 `calibration_lock.json`（参考hash、数值规则和代码hash），再处理后5个的响应标签；不为持出误差改变模型。这里不额外拟合“死区掩码”。

### B6. 相同H的三路径比较

输入首先固定为捕获的FP32 H；三条路径分别为：
1. 同一H提升到FP64，执行安装语义一致的有限NS；
2. 同一H用FP32有限NS、无内部BF16转换；它是数值对照，非生产Muon；
3. 同一H调用安装版原生NS函数，内部BF16、不修改安装代码。

形状、transpose、norm.clamp、eps、NS系数和次数全部按本轮已绑定源码。FP32诊断禁用TF32，退出诊断后恢复开关；生产梯度采集开关保持原样。不要在外层autocast中让FP64/FP32参考偷偷降精度。

谱输出字段 D_raw 不含LR、WD或形状因子；s(r,c)单独登记。可另存s*D_raw统计，但不能与D_raw同名、跨层混算；本轮没有真实parameter update。

所有矩阵用完整形状计算NS，不用2×2切片或投影后的小矩阵冒充真实输出。JVP只针对有限NS参考，按方向计算；原生BF16不声称存在同一个无限小导数。

先进行小随机矩阵/矩形/归一化分支单测及既有测试，不额外使用真实batch做smoke。

### B7. 自然差异和有限幅度测量

所有10个实际H_i的完整三路径输出是主数值观察。保存自然偏移比例 `||H_i-H_ref||/||H_ref||`，不将自然差异强制缩到toy的小扰动区域。

额外诊断对每个H_i使用自然方向 E_i=H_i-H_ref，不挑方向，幅度固定：
`alpha={1e-3, 1e-2, 5e-2, 0.25, 1.0}`，正负两侧均测，另含零扰动参考。

这些alpha是自然差异的倍率，不是T0中的delta。每行必须记录实际相对扰动。

为避免输入表示差异混杂：先生成一次实际FP32输入H_in，然后clone给三条映射；alpha=+1的自然点强制使用保存的原始H_i，不以“相减再相加”的舍入版本代替。所有JVP沿实际 `H_in.double()-H_ref.double()` 计算，避免将输入舍入误计为NS误差。重复点复用已有输出，不重复计算梯度。

每个点记录以下矩阵残差的范数/方向统计：
```
L = J_FP64(H_ref)[H_in-H_ref]
delta_native = Phi_native(H_in)-Phi_native(H_ref)
r_real = Phi_FP64(H_in)-Phi_FP64(H_ref)-L
r_numeric = [Phi_native(H_in)-Phi_FP64(H_in)]
            -[Phi_native(H_ref)-Phi_FP64(H_ref)]
```
必须核对向量恒等式 `delta_native-L = r_real+r_numeric`；不能用“范数相加等于残差范数”代替。

记录：绝对误差、真实响应范数、预测范数、余弦（零向量时为null）、H转换BF16前后改变元素比例、原生输出零响应标志。相对残差统一除以 max(||delta_native||,||L||)，分母为0则null并给出ZERO_BOTH，不能人为加一个大常数隐藏误差。小分母的原始值必须保留。

原生H_ref及batch1、batch6的自然H重复映射一次以核对确定性；这不额外计算batch梯度。重复性和代理准确性分开报告。

本轮不固定新的“误差多少即可证明方法有效”的百分比。不把T0的0.2%、4e-3或特定比值当成通关目标。前5确定的参考/规则在后5原样检验，完整报告不支持、零响应与排序不可靠的条件。

### B8. 分解与统计边界

对前5/后5各自用除以n的经验描述量，报告各映射输出均值、方差迹和相对参考输出的均值偏移平方。real-data中后5的输入均值可能不同于H_ref，须报告输入均值漂移；不能将全部平均输出变化归为非线性整流。

同H的数值误差再分成 Dr、q和交叉项，采用阶段A修正后的统计入口。只算trace/内积或小规模投影，不显式分配(矩阵元素数)^2的协方差。

本轮不做p值、不把层数或batch数当成独立训练种子、不根据update variance推断NLL收益。

## 4. 计算、存储与异常处理

- 阶段A：CPU后处理，数值计算累计上限30分钟，不跑GPU重训练。
- 阶段B：一张空闲GPU，累计占用墙钟上限30分钟，包括梯度采集、NS、JVP、重复映射；到限在安全边界保存结果停止。该值为预算，不是预计耗时。
- 新生成原始数据、统计与包累计不超过1GB；到限保存现状。不要为赶时限静默缩矩阵、改变精度或减少要报告的失败条件。
- 最多10个唯一完整batch，正常为40次micro-batch backward；失败尝试也记账，不通过替换batch或无记录重试突破上限。
- 真正的模型更新数必须为0。NS矩阵回放次数不能写成训练updates。
- 锚点、源码、数据或完整梯度语义无法确认时先停；普通可修复的代码问题允许在不改实验定义的前提下修正并留下记录。

存储优先保留G_i、M_*、H_ref和自然H_i/D_i，方便独立复算；幅度诊断仅存标量表与固定样例完整数组，不保存所有方向点的大矩阵副本。FP32纯G约99.6MB，依据实际shape预估并写入台账。

## 5. T2草案修订（仅文档，不执行）

修正六矩阵2,490,368元素、两条Muon父轨迹提供四态、G按bank只保存一次和流式回放成本。不再使用“123GB”或未经真实验证的BF16死区掩码。

说明旧M1/M2状态清单“40终态”与“48成功run”的差异，只列是否缺失/未计入，不重跑恢复。

完整T2仍是待批任务：不能启动2种子父轨迹、4状态×2bank×128batch采集或16排列全量实验。仅给出实测速率外推，并标明NS重放/JVP与梯度成本应分开。

## 6. 交付和停止

建议交付：
- `docs/ROUND1_ERRATUM.md`；
- `docs/T2A_PROTOCOL_v1.md`、`configs/t2a_scale_v1.json`；
- `docs/T2A_REPORT.md`；
- `docs/T2_PROTOCOL_DRAFT_v2.md`（不覆盖原草案）；
- `outputs/t0_round1_corrected/`：修正统计和比对表；
- `outputs/t2a_scale_v1/`：manifest、raw索引、map_errors.csv、response_errors.csv、summary.json、cost_ledger.json、测试/运行日志；
- 可复跑代码、必要回归测试和命令。

最终报告分三栏：
1. **技术正确性**：哪些不变量通过、哪些问题阻塞；
2. **数学/数值适用范围**：FP64线性化余项、FP32偏差、BF16偏差与自然尺度分别是什么；
3. **研究判断范围**：是否支持准备完整T2、证据缺什么；不作训练收益结论。

允许的状态为 `COMPLETED_DIAGNOSTIC`、`PARTIAL_BUDGET`、`BLOCKED_TECHNICAL`。另行说明科学模型的SUPPORTED/NOT_SUPPORTED/UNRESOLVED，不用一个ALL_PASS混合技术正确性和科学结论。

给出轻量审阅包：报告、配置、CSV/JSON、代码、测试、固定第一层q_proj在batch1和batch6的关键原始数组；完整raw留服务器，按需另传。不要打包训练语料、模型checkpoint、环境、密钥或旧项目大文件。

完成本轮后停止。不要自动进入完整T2，更不要进入T3/T4或提出优化器修改来救场。
