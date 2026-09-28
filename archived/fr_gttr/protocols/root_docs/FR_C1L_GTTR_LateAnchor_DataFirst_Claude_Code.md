# FR C1-Late：补齐既定测量，并以关键数据独立复核 GTTR

版本：v1.0（2026-09-21）
用途：在现有 FR 仓库交给 Claude Code + Kimi K2.7 执行。
性质：执行协议；不是新实验结果。设计者读取了用户 C0、C0X+C1 完成摘要及原 C0/C1 协议，未读取远程原始结果与代码。

## 0. 授权、目的与范围

用户要求继续理论探索，提供下一步实验指令，并保证即使执行模型解释错误，也能从关键原始数字复核结论。

本轮只做两件事：
1. CPU 离线提取/复核已完成 C0X+C1 的关键数据、缺失项与判定。
2. 补齐原 C1 已预定但因预算未完成的两个 native A/t48 anchors：C0/C1 的 Cell R 和 Cell C，各一个；沿用原模型、probe 库、预测顺序、K、半库、目标和指标。

不设计新 optimizer/controller，不改 LR/beta/batch 策略，不训练新 checkpoint，不扩展模型/seed，不拟合隐藏 regime、慢快模型或记忆核，不自动进入后续轮次。A/t48 是原方案覆盖补齐，不称为独立总体确认，也不称为“已证明的 A0U 延迟收益区”。

先读原件再运行；不要仅复述用户摘要。无需只返回计划。条件满足后直接执行到本轮停止点。

## 1. 必读文件与真实语义

读取现场实际路径：
- FR_C0_Geometry_Tracking_Validation_Claude_Code.md；
- FR_C0X_C1_GTTR_Offline_and_Probe_Validation_Claude_Code.md；
- 两轮结果前 config/amendments；
- outputs/fr_c0x_offline_explanation_v01/；
- outputs/fr_c1_gttr_probe_validation_v01/；
- 实际 C1 runner、analysis、CPU tests、frozen AdamW/clip/HVP/restore primitives；
- C0/C1 state_provenance、snapshot、probe ID 列表、target-axis 定义、预算日志；
- 已有 GTTR_EVIDENCE_UPDATE.md、FR_PROJECT_STATUS、FR_EXPERIMENT_INDEX。

从 config 恢复 R/C 的真实 seed、checkpoint、stream、local t 与累计 optimizer step；不得凭历史转述硬编码。

沿用当前 Kimi 接入和现有环境，不升级依赖、不切换模型、不重构框架、不自动 commit/push、不干扰其他训练进程。小量 config/关键结果 hash 即可，不创建审计 tar、数据库或常驻监视服务。

### 1.1 当前摘要只是待核对读数

- C0X 确认 C0 聚合半库门曾误实现为逐点门；追加 erratum，不覆盖原结果。
- C0 重算 field E 为 R=0.5291、C=0.4676；原半库充分统计缺失，g3=DATA_MISSING，不能追认完整 PASS。
- C1 实际完成 A/t0、Bp/t16，共 4/6 anchors；K32 field E=R 0.505/C 0.398，半库<=0.700，所选位置得到局部支持。
- shadow E=R 0.667/C 0.585，但 t0 目标/SE 约2.8–2.9，未通过3SE；shadow=UNRESOLVED。
- 8/8 是4 anchors ×2 channels，不是8个独立seeds。
- old ledger=17,994/18,000；C1另计45 updates/605 effective gradients/304 HVP/1801 micro-forwards。
- 部分旧 shadow 同probe残差SE未保存；缺失不得由均值误差或其他SE推算。
- “63步累计”须解释端点索引；不能自动把63写成64。

现场原数据与摘要不同，则两者并列并给出来源，不偷偷选择有利版本。

## 2. GTTR 统一模型：精确骨架与近似分开

### 2.1 单步骨架

一个参数组写为（真实实现按组及decay mask处理）：

g_t = grad ell(theta_t; omega_t)
c_t = Clip_tau(g_t)
mbar_t = a_t*mbar_prev + (1-a_t)*c_t
vbar_t = b_t*vbar_prev + (1-b_t)*c_t**2
P_t = diag(1/(sqrt(vbar_t)+eps))
r_t = P_t*mbar_t
d_t = -eta_t*r_t
s_t = d_t - eta_t*lambda_wd*theta_t
theta_next = theta_t+s_t

标准固定beta、零初始化且累计步号从0计数时：
a_t=beta1*(1-beta1**t)/(1-beta1**(t+1))；b_t类似。
诊断local t与真实累计步号分开，禁止窗口内重置bias correction。

Geometry：固定probe随机性时 G_t=D_theta c_t=J_clip(g_t)*H_batch,t。
实际clip的数值稳定项、顺序、active boundary按照frozen实现，不悄悄换理想公式。
G_t一般非对称；H_batch不等于H_V，E[Clip(g)]不等于Clip(E[g])。

Tracking：mu_t=E[c_t|F_t]，nu_t=E[c_t**2|F_t]，e_m=mbar-mu，e_v=vbar-nu。
e_m(t)=a_t*e_m(t-1)-a_t*(mu_t-mu_prev)+(1-a_t)*(c_t-mu_t)。
e_v(t)=b_t*e_v(t-1)-b_t*(nu_t-nu_prev)+(1-b_t)*(c_t**2-nu_t)。
F_t是该probe随机输入揭示前的信息；固定状态条件均值不能用不同theta上的流均值替代。

Transport：
d_t=T_t*d_prev-eta_t*(1-a_t)*P_t*c_t，
T_t=a_t*(eta_t/eta_prev)*P_t*P_prev^{-1}。
T_t只为已实现轨迹传递系数，不是counterfactual Jacobian A_t；相邻eta为0时回到原更新，不强行相除。
当前log-LR直接输入为[s_t,0,0]，不存在额外一层eta。

Readout：q_t=grad V_diag(theta_t)，W_t=-q_t^T*s_t，K_V=DeltaV+W_t。
DeltaV=-W+K_V；K_V不保证正，W/K为已实现记账而非独立因果份额。

完整响应仍来自z_next=F_t(z,u,omega)的链式求导；沿基准轨迹的逐步A_t已经允许几何变化。
固定模型/数据分布/目标时L(theta)固定，不能因feature learning就另建独立的L_t或rho物理自由度。

### 2.2 本轮只检验局部近似

已知当前真实位移s后，用起点信息预测同probe在theta+s的梯度：
g1_hat = g0+H0*s。
再经真实clip与真实shadow U。
这是局部条件预测，不是提前预测未来H16/H64，更不是完整训练优化。

## 3. 先做离线数据复核（0 新 GPU 调用）

建立 outputs/fr_c1_late_completion_v01/；不要覆盖C0或C1原制品。

### O1. 复核原规则与旧充分统计

独立从CSV/JSON重算，不调用原分析器的聚合helper来“独立证明自己”。
对C1已完成的固定两个anchors/cell，各channel、K8/16/32、first16/last16，列SSE/TSS/E。
E=sqrt(sum_anchor ||p-y||^2 / sum_anchor ||y||^2)。
不取平均单点比值，不筛掉不分辨anchor，不把一个单点E>1当作聚合失败。
缺少SSE/TSS或SE所需统计直接标DATA_MISSING；不得从四舍五入的E反推足以正式判定的原数。

核对：field和shadow是否各自使用正确y、floor、分辨规则与distinct t；first16与K16是同一组，不算两次独立证据。
原C0 g3缺失继续保留；新的t48不会修复原C0历史数据缺失。

### O2. 读出与索引

列出每个目标名称、数据ID/数量、reduction、eval mode/精度、local h与参数更新次数的对应。
V_diag与V_eval、A0T-v_score分开。无共同目标则维持AXIS_MISMATCH，禁止减常数对齐。
对已有数据输出[0,4)、[4,16)、[16,64)的实测端点和W/K累计差；若只有63项，指出缺哪一步或具体索引定义，不插值、不补零、不新增GPU去补历史readout。
先列Bp-native和Bm-native，再列Bp-vs-Bm。代数telescoping正确不是反转机制已证实。

### O3. 数据缺失不伪造

旧shadow per-probe残差SE有向量或充分统计才可离线恢复；否则保持NA。
旧D_i/q+读数列真实定义、符号、单位、所属cell/arm/t/K；摘要定义不明时不解释成导数显著反转。

## 4. A/t48状态恢复、probe及信息边界

只补 R/A/t48 与 C/A/t48（t为更新前local索引）。
取得theta48、真实s48和更新后的共同状态S+：theta49、moments、next step/schedule/RNG等。

优先使用已有可核验完整snapshot。仅凭lr、clip、loss等标量匹配不能声称完整state bitwise一致。
若从原checkpoint完整重放，每cell最多49次真实update，main最坏98；若从C1较近的native snapshot接续，按真实step数计算并逐项记录。
不要承诺“37 updates”；这个数字只有在snapshot的真实累计索引支持时才能采用。

两个late anchors沿用C1同一cell的32个D、32个A probe IDs及顺序、batch recipe和随机性。它们是原覆盖补齐，并非新增独立总体样本。
不能因t0信号弱重新抽probe，也不能改变batch size、K、clip或阈值。
旧IDs不可用则STOP/DATA_MISSING，不改用新库后仍称严格补齐。
主probe库之间仍需不重叠；每probe的theta/theta+s/HVP使用相同batch与dropout随机状态；均按原effective batch accumulation后clip一次。

## 5. 原封不动保留C1预测对象

每个probe j：
g0_j = grad ell(theta;omega_j)
h_j = H(theta;omega_j)*s
g1_j = grad ell(theta+s;omega_j)

field：
pF_j = Clip(g0_j+h_j)-Clip(g0_j)
yF_j = Clip(g1_j)-Clip(g0_j)

shadow：
pU_j = U(g0_j+h_j;S+)-U(g0_j;S+)
yU_j = U(g1_j;S+)-U(g0_j;S+)

对应旧符号为P/Y/Q/Z。U接受raw gradient并内部执行一次真实clip及m/v/P更新；两项均使用同一S+及下一步eta/step/decay mask。
不要对field差再clip，也不要把U的两个输入放在不同moments或不同theta/decay状态上。
U是只读反事实算术，不把shadow step写回真实训练器。
E[U(g)]!=U(E[g])，必须逐probe计算后再平均。

### 5.1 预测顺序

1. 先读D起点g/Hs，生成各固定K的mean(pF_D)、mean(pU_D)，保存磁盘。
2. 在读取audit终点前，固定q+=grad V_diag(theta+s)及实际s/||s||方向。q+是已知当前一步终点的固定读出，不是未来收益梯度。
3. 之后才能读取任意main probe终点g1及audit Hs。
4. 后算p_A/y_D等为CONDITIONED_DIAGNOSTIC，不修改已保存D预测。

不改变原C1关于start audit访问的更严格顺序。避免为了预计算输入全量读入audit终点。

## 6. 固定K、误差与统计

一次性完成D32+A32。K8、K16、K32为嵌套前缀；K32 halves固定first16和last16。
禁止看结果后增加K、改变顺序或替换anchors。K变大不要求每次单调改善。

field与shadow分别设：
p_D=mean(p_j in D)，p_A=mean(p_j in A)，y_A=mean(y_j in A)。
e=p_D-y_A；e_bank=p_D-p_A；e_local=p_A-y_A。

保存并核对：
e=e_bank+e_local。
||e||²=||e_bank||²+||e_local||²+2<e_bank,e_local>。
交叉项可负，不把norm比值当可加贡献，不把bank差称不可约噪声。

### 6.1 SE与分辨性

对任意向量样本x_j：
SE_vec(xbar)=sqrt(sum_j||x_j-xbar||²/[K(K-1)])。
必须分别保存prediction SE、audit target SE、same-probe residual mean的SE：x_j=p_j-y_j。
后者必须用逐probe配对差计算，不能用SE(p)+SE(y)代替。

numerical_floor的定义和单位沿用C1。same-probe重复性与systematic accuracy分别记录。
resolved：||y_A|| > 3*sqrt(SE_target²+floor²)，按原严格比较。
分母为0：同时给target_norm和SE/floor；若两者皆0则UNRESOLVED，而不是构造无穷SNR。
3SE是工程分辨门，不称严格高维置信球或物理临界点。

### 6.2 stop-v与固定方向

复用C1 stop-v：同一probe的一对U固定P为基准g0形成的P，m仍按各输入变化。
保存yU_full、yU_stop及逐probe paired difference；报告均值norm、SE、norm ratio、mean cosine。
固定q+读出：x_j=-q+^T yU_j，hat x_j=-q+^T pU_j；另报s_unit投影。
必须输出scalar预测/目标/配对残差均值与SE、full-stop配对差SE。
不得仅用范数大11%等读数断言signal显著、冻结v更好或发生降噪。
如报告D_i，附数学定义和具体投影，不能只给符号串；不得把HVP或q+方向按audit结果重选。

## 7. 不再丢失关键统计：具体保存要求

所有统计先保存再出结论。单anchor内可暂存向量或做FP64/Welford累积，不创建全轨迹巨型梯度归档。
原forward/HVP精度不改；FP64仅用于已有输出的稳定标量累积，并注明。

### 7.1 每组充分统计（bank D/A，K8/K16/K32/last16；first16与K16同组别名）

对field/shadow及stop-v：
- n、sum_j||x_j||²、||mean(x)||²、Welford centered_SS，x分别为p、y、p-y及full-stop；
- same-probe必要inner products或直接配对残差centered_SS；
- per-probe固定投影的pred、target、residual、full、stop、paired_diff；
- 每anchor每group的mean(p_D)、mean(p_A)、mean(y_A)三者norm²和两两inner products；
- SSE=||mean(p_D)-mean(y_A)||²、TSS=||mean(y_A)||²；
- bank_SS、local_SS、signed_cross=2<e_bank,e_local>及恒等残差；
- clip active counts at g0/gpred/g1、active-set crossing counts；
- 数值floor、预测保存时点、probe ID与随机recipe标识。

若用mean向量计算完成后释放，可只存上述可复算统计；不能只存单点norm而丢掉mean差或cross。
用足够数字精度保存（双精度标量建议17有效位）。近零统计给绝对值，不用舍入为0后判门。

### 7.2 最低文件集

config.json
state_and_probe_provenance.csv
resource_log.json
late_probe_statistics.csv
late_error_decomposition.csv
late_fixed_projection.csv
late_update_feedback.csv
aggregate_gate_inputs.csv
criterion_trace.json
old_data_missing.csv
KEY_DATA_FOR_REVIEW.md
REPORT.md
GTTR_EVIDENCE_UPDATE.md

现有test/smoke记录可引用或附精简结果。不需要额外hash体系。

## 8. 聚合与判定：三个范围分别报告

### 8.1 原C1四anchor结果

只离线复核，版本与原结果保留。发现错误追加erratum，不能覆写历史。

### 8.2 本轮两个late anchors

每个cell只有一个late时间，因此不套用“每cell至少两个distinct t”的全覆盖结论。
逐anchor/channel报告E、半库E、SE与分辨性；使用LATE_ANCHOR_RESOLVED_AND_IMPROVED、RULE_NOT_MET或UNRESOLVED等范围明确的描述。
单anchorE<=0.8与半库<=1是同口径描述，不创造一个“统一GTTR已通过”headline。

### 8.3 原计划六anchor完整覆盖的补齐读数

只有状态/目标/probe/实现/指标一致，且旧充分统计足够，才汇总：每cell A/t0、Bp/t16、A/t48。
主聚合使用这三个固定anchors全部，不按新resolved标志删点。
E=sqrt(sum SSE / sum TSS)，不能平均单点E。
K32两16半库也使用同一固定集合。
分别判断field与shadow：两cells各自K32 E<=0.8，两个half E<=1，每cell至少2个distinct t的target可分辨。
输出每一个布尔条件及原始SSE/TSS/resolve count。
旧统计不足则COMBINED_GATE_DATA_MISSING；新late通过不能填补旧缺失。
若全覆盖条件满足，仅称“原预定六个位置上的局部支持，经补齐获得”，不称独立重复或全预训练规律。原C1的PARTIAL历史仍保留。
未满足需区分NUMERICAL_UNRESOLVED、SIGNAL_UNRESOLVED、RULE_NOT_MET，不用单一FAIL掩盖。

A/t0与A/t48可作同arm两位置比较；Bp/t16仍有arm/time共同变化，不能由其差值宣称纯时间趋势。
shadow在t48可分辨不能追溯性改变t0未分辨的事实。scalar投影不替代vector gate。

固定NOT_TESTED：未来H16/H64提前预测、adaptive训练价值、最终训练性能、能耗收益、全局山谷识别。

## 9. 强制把关键数据直接写进最终回复

KEY_DATA_FOR_REVIEW.md必须由保存的表自动生成，不手抄数值；最终聊天回复原样带关键表，不只说“全部制品已保存”。

### 表A：逐anchor数据（field/shadow分别一表）

旧数据足够时列完整6个anchor，否则旧项NA+原因；新增两anchor不得省略。
列：cell / arm / local_t / cumulative_step / K / target_norm / SE_target / floor / signal_ratio / resolved / pred_SE / residual_SE / SSE / TSS / E。
K32为聊天主表，K8/16放KEY_DATA文件与CSV。

### 表B：每cell聚合判据

列：scope(OLD4/LATE2/COMBINED6) / cell / channel / group(full32,first16,last16) / anchor_IDs / n_anchor / n_distinct_t_resolved / SSE_sum / TSS_sum / E / threshold / criterion_true_false_or_missing。
此表就是用来防止再次把逐点门写成聚合门；不能只填E。

### 表C：新增late位置误差分解与v反馈

每late anchor/channel列：total_SS / bank_SS / local_SS / signed_cross / identity_residual。
shadow另列：full_signal_norm / full_SE / stop_signal_norm / stop_SE / norm_ratio / mean_cosine。

### 表D：新增late固定q+读出

列：qplus_norm / pred_mean / target_mean / target_SE / same_probe_residual_mean / residual_SE / full_minus_stop_mean / paired_diff_SE。
均值±3SE可描述，但不称总体显著性或新优化收益。

### 表E：成本与覆盖

planned/actual anchors；每个恢复来源与准确replay update数；smoke/main/failure/recheck/analysis分项计数及round总数与上限。
附旧17,994上限账、旧C1账、本轮新账，分开，不重新除以旧18,000。

最终另用中文回答：
1. 数学/实现身份与经验支持分别是什么？
2. field的近似缺陷和跨库差在late点怎样变化？
3. shadow分辨性是否与早期不同，原六点覆盖门能否复算并通过？
4. 动态v反馈有何已测效应，哪些读数缺统计支持？
5. 哪些量仍是缺失，哪些结论仍NOT_TESTED？
不得写“唯一原因已找到”“已解释H64反转”或“可以据此改optimizer”。

## 10. 新一轮额度：留足实际恢复与失败余量

旧总账预计17,994/18,000，旧C1独立45 updates，均保持原义。
本提示词被用户交给Claude执行，即授权以下独立C1-Late诊断额度；不占用或重置旧18,000上限。

- 真实optimizer updates <=128，全部仅恢复旧状态。
  无快照时main最坏2*49=98，diagnostic on/off 4步各一次smoke最多8，余22。有效snapshot可减少。
- nontraining effective-batch gradients <=600；
- effective-batch HVP <=192：主测128，smoke/数值检查预留16，余48用于一次性有依据的恢复；不因有余量增加科学样本；
- nontraining micro-forward calls <=1800，包含HVP内部forward、重复floor、读出；
- GPU任务占用墙钟<=45分钟，CPU离线单列；
- shadow更新算术另计次数/耗时，不当真实training update，也不称无成本。

各费用的单位和计数规则沿用C1，额外列primitive调用明细，避免effective-batch与microbatch混算。训练forward另记，不能隐藏。
所有失败、重试、salvage、验证计入轮次总数；不能按进程重置计数。
提前估计所需工作能否完成。预计超限就停止并保存PARTIAL，不临时少测K、单边anchor或换库。
只允许一次有具体原因的修复重试，不反复在同一失败处消耗预算；不可修复就BLOCKED。
不承诺37updates或固定分钟数；显示实际snapshot推导和首批实测ETA。

## 11. 小型独立验证与停止

复用C1测试，补两个很小但独立的CPU参考检查：
1. 七个相同TSS的anchors，六个E=.43，一个E=1.166；聚合E约.594<1，逐点失败不得否定聚合。
2. 对小维度向量直接算SE/paired residual/交叉项，与保存充分统计的独立重算一致；缺centered_SS必须返回DATA_MISSING。

独立检查器直接使用CSV的SSE/TSS/count/centered_SS，不导入生产判据helper。代码可以很短，不建第二套训练框架。

GPU：复用已验证frozen primitives；确认新记录不改变真实theta/m/v/step/RNG，probe shadow不污染状态。
若本轮只修改统计保存，优先最小验证；不得把旧标量门升级描述成未测过的完整state bitwise。

完成本轮后停止；不自动转到控制器或更长时域。只发一次最终正式报告。无需tail/grep常驻monitor；如临时启动了本轮自有monitor，结束时关闭并记录，不处理其他用户任务。
