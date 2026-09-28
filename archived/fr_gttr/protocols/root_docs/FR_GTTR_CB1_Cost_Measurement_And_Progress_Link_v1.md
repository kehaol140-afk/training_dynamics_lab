# FR GTTR-CB1：达标成本读出与状态进展指标的有限验证 v1

日期：2026-09-24。执行端：OpenCode＋GLM5.3。

## 0. 本轮目的、来源和范围

用户当前要求：不是增加一个无依据的目标代理，而是检查 GTTR 的状态/目标进展量如何与真实训练效率发生联系，并明确最终求解对象。

本轮选择的桥是：**R2B 的一次已求解首步改动，此后持续回到共同 AdamW k=2 的剩余成本**。不在本轮改成持续在线策略，不重新估计 alpha，不改变初始化/网络/采样/optimizer，也不恢复冻结场预测器。

已有事实按用户提供的 R2B 报告理解：36 条旧轨迹全部在 256 步内完成原检测规则，FULL 与 LR_ONLY 11/12 配对在原 16 步网格没有检测差异。原结论不变。本轮只完成：

1. CB0：全部旧 36 条轨迹的成本公式与状态进展读出，纯离线。
2. CB1：固定第一条 stream 的 18 条旧轨迹做确定性重放，提高观测网格分辨率，但保持原 16 步确认间隔；检验是否新增了可分辨信息。
3. 用一个事前规定、无拟合参数的“当前目标差距/近期单位-token净进展”关系作最小闭合诊断。它不是新在线模型，也不是所有未来算法必须通过的门。

没有新训练随机流、没有新候选算法、没有新目标梯度/HVP；重放仍是真实 GPU 计算，全部计费。原实验与本轮结果分别登记，不将重测当作独立重复。

本轮不是证明 token 效率已经提升，也不是求出新训练规则。它要决定：成本读出是否足够分辨，以及最简单的 GTTR 进展指标能否解释部分剩余成本。失败/无变化同样完整报告。

## 1. 数学关系：目标没有更换

### 1.1 GTTR 的净目标进展

对同一个固定监测目标 V，在实际位移 s 上定义：

    W = -grad V(theta)^T s
    K_V = V(theta+s)-V(theta)+W
    progress = V(theta)-V(theta+s) = W-K_V

以上是同一目标的精确记账关系，不表示本轮测到了 W、K_V 各自。CB1 仅通过真实 V 差计算净 progress；没有新 q/H 测量时，单独的 W/K、真实条件漂移和噪声协方差均标 NOT_MEASURED。

对于区间 [l0,l1]：

    rho_obs = (V(l0)-V(l1)) / (N_slots(l1)-N_slots(l0))

它的单位是 nats/有效目标token / 训练input slot。它是已实现路径的区间净进展率，不是已经辨识的 E[progress|state]，不能冒充条件无偏漂移或通用物理常数。

### 1.2 达标成本等于存活指示量的 token 加权和

对给定检测规则，将“尚未确认达标”作为状态标记。网格为 t_0,...,t_M，累计训练 slots 为 N_j，A_j=1 表示到 t_j 仍未确认达标。

    C_capped = sum_{j=1..M} (N_j-N_{j-1}) * A_{j-1}

这逐路径等于：已在预算内确认则取确认时的训练 slots；否则取预算上限，同时 censor=True。

不能用 NLL 曲线面积替代，不能将未达标上限当成真实首达成本。本轮不使用软阈值、不做普通 autograd 求硬 hit 时间，不用插值当真值。

原 token 主口径仍是 train_input_token_slots；valid_loss_tokens/nonpad 单列，从 mask/prefix 真实计数。重放花费是额外研究成本，不增加原路径的 elapsed-training-token 横坐标。

## 2. 两种停止读数必须分开

### 2.1 原主要读数完全保留

    OLD_GRID16_LAG16
    T16 = min{t in {16,32,...,256}: V(t)<=ell and V(t-16)<=ell}

这是原“相邻两次16步监测均通过”的同一规则。使用源文件全精度阈值，start46/start82分别读取；<=不改。原结果和判定不覆盖。

### 2.2 新的观测分辨率诊断

    REFINED_GRID4_LAG16
    T4lag16 = min{t in {16,20,24,...,256}: V(t)<=ell and V(t-16)<=ell}

本轮细化的是观测网格，**确认间隔仍是16个optimizer updates**。不是两次连续4步检查，不是8步即可确认，不要求新加的所有中间点均低于阈值。它也是一种新、明确标记的读出协议，结果不能与旧协议混为同一个首达时间。

在路径与数值语义完全一致时，OLD_GRID16 是新网格的子集，所以 T4lag16<=T16（旧已达标时）。因此新测量只需覆盖至各轨迹原 T16，即可确定完整的 T4lag16；不必续跑至256。必须测所有此前缺少的4步网格点，不能只挑原 crossing 附近，以免漏掉非单调路径更早满足的配对。

保持原 h=0 判断；初始已低于目标则按原身份规则单独标记，不换更难阈值。硬件/数值差异若使旧检查点复核不一致，不能把差异当成新训练收益。

## 3. 固定状态和重复轨迹

### 3.1 CB0 离线覆盖

读取原 R2B 全部36轨迹的 monitor sum/count、target hits、原 R1解、snapshot manifest和资源记录。只补本轮所需字段，不重审A0/B0历史。

### 3.2 CB1 GPU覆盖

固定6个原state：seeds2030–2032 × start46/82。
每state只选原R2B两条stream中 **数值ID较小的一条**，包含 BASE_K2 / GTTR_LR_ONLY / GTTR_FULL 三臂，共18轨迹。

按原方案通常对应后缀200/202/204/206/208/210，但以完整原manifest为准，不从截图猜ID。

该选择是预算范围，不按gain、是否显著或是否出现旧token差异筛选。它不是新增独立数据；另一路旧stream的结果仍在CB0全表报告，不作为被排除的失败。

从原 h=1 的 full-state snapshot 恢复，包含theta/m/v/step/CPU与CUDA RNG/sampler cursor/执行模式。不要重新施加R1动作。恢复后全部是共同原生AdamW k=2，准确沿旧轨迹推进。

若必要full-state或确定性数据恢复信息缺失，标 REPLAY_IDENTITY_UNAVAILABLE；不要用新流替换。不要假装标量相同就等于完整状态逐位相同。

## 4. CB0：纯离线任务

1. 直接从全精度monitor序列重建OLD_GRID16_LAG16，核对全部36条原hit和阈值。
2. 逐路径重建alive A_j，独立计算survival-area成本与direct-hit成本；应相等。
3. 分seed/start/stream报告FULL-LR_ONLY、FULL-BASE和LR_ONLY-BASE：不只给中位数。
4. 保存R1局部预测收益与R2B即时/末端真实gain供描述；V_ctrl与V_monitor不同轴，不计算混轴“预测误差”。
5. 计算landmark h=16和32的past-only读出，公式见§6；仅使用截至landmark的数据。
6. 成本对账仅解决R2B记录中相关forward字段，不重新建立全项目账；本轮新成本账单独计。

## 5. CB1：有限确定性重放与更细目标读取

### 5.1 smoke

固定第一个选定BASE轨迹的h1全态；diagnostic-off/on各推进8步，共16 updates。on路径可以在h4、h8测monitor，并在起点h1核对旧值，最多3次整库eval。

检查新增evaluation与记录不改变theta/m/v/计数、RNG和cursor。使用原评价后端/精度/顺序，不为了通过换batch、dtype或reduction。

### 5.2 main

18条轨迹分别从h1推进到原各自T16（不是从初始化新训练）。

- 新计算4步网格中原先不存在的monitor全4096行sum/count，eval batch32。
- h0、h4、以及原16倍数的值可使用旧同轴数据；另在每条的h16和旧T16做两处重叠复核（相同点仅算一次）。
- 必须明确各行source为OLD_REUSED / REPLAY_NEW / REPLAY_OVERLAP，不混装成全新评价。
- 每步有原training标量/索引记录时核对对应值；已有h256全态只有实际抵达h256时可比较，不因此强制所有轨迹跑满。
- 主monitor评价全库，不改成256行子集；input/mask/precision/分母必须原样。
- 相同原始数值重复预期可严格一致；如文件序列化造成小差异，展示绝对差及原保存精度。只有能证明不改变所有使用阈值判断且满足原已冻结复算容差时才使用；不得事后新造容差救结果。否则停止该身份核验并报告具体证据，不输出细化首达结论。
- 不在看到新hit以后改变测量网格或目标。即使某臂提前达标，也继续到旧T16完成预定网格。

main后只得到旧训练轨迹更细的目标读数，不等于多训练了一个新方法。

## 6. 指标到效率的最小闭合检查（无拟合、非在线规则）

目的：检验GTTR的净目标进展读出是否足够支撑最简单的剩余成本关系，而不是增加一个黑箱J网络。

### 6.1 固定landmarks

主landmark l=16：past window [0,16]。
伴随landmark l=32：past window [16,32]。

定义：

    gap_l = V(l)-ell
    rho_l = [V(l-16)-V(l)] / [N(l)-N(l-16)]

只对l之前尚未确认、gap_l>0的路径建立此诊断。若rho_l<=0，则输出 NO_POSITIVE_PAST_PROGRESS，不能加小epsilon把它变成有限正成本。

只有在“未来净进展近似保持近期速率、阈值附近不反复回摆”等额外假设下，才有近似：

    predicted_raw_cross_tokens_after_l = gap_l/rho_l

本轮实际inputslots/step固定为1024时，按新观察网格给一个完全预定义的检测预测：

    t_pred_first = 4 * ceil((l + gap_l/(1024*rho_l))/4)
    t_pred_confirm = t_pred_first + 16

保留原始未截断预测，超过256标 FORECAST_BEYOND_BUDGET；不能把超过预算的值截到刚好使误差变小。若实际slots计数与1024不一致，使用已核对prefix逆映射，不擅自使用valid token均值替代。

这是无拟合参数的最低阶闭合诊断，不是已验证定律。它的失败不会否定全部GTTR，也不是未来所有训练方法的准入门。

### 6.2 评价方法

- 在CB0全36轨迹上，用旧网格版本（ceil到16再+16）与旧hit比较。
- 在CB1固定18轨迹上，用新4步网格版本与T4lag16比较。
- 保存每行gap、rho、实际/预测剩余tokens、绝对误差、相对误差、abstain原因、首达前最低/最高区间进展等描述量。
- 不用未来段loss估rho，不用已实现终点overshoot给预测补答案。
- 用k=2下真实schedule/n0作为标签分层，不据结果拟合衰减修正。
- FULL与LR_ONLY的预测成本差和实际成本差分别列出，ties单列；不将零差硬记为正确方向。
- 不拟合新网络、不追加特征、不扫描landmark/窗口，不从很多相关系数里挑最好值。
- 如rho不稳定，报告“该最低阶闭合未建立”；不把它直接归因于某一种moments/曲率机制。
- 本輪没有测到真实条件漂移及其噪声分解，不把路径增量样本方差当作Sigma或Gamma。

## 7. 要分别报告的结论

- COST_READOUT_IDENTITY：实现的alive加权和与对应协议检测成本是否一致（代数/实现层）。
- REPLAY_IDENTITY：已有路径是否忠实恢复（执行层）。
- RESOLUTION_INCREMENT：更细网格新分辨了哪些差异，哪些仍为ties；原16步结论保留。
- PROGRESS_TO_COST_DIAGNOSTIC：预定的gap/rho闭合在各条件是否有用、误差多大；不声称普遍成立。
- ONE_SHOT_COST_EFFECT：对选定旧路径的配对作用，仅范围内报告；不是独立重复。
- SUSTAINED_POLICY_BENEFIT / INITIALIZATION_OPTIMIZATION / NET_COMPUTE_SAVING / FINAL_QUALITY_CONFIRMED：本轮均NOT_TESTED。

不能把细化网格上的正结果回写成原R2B成功，不能凭某条提前达标宣布普遍提升，也不能说网格ties证明精确数学首达时间相同。

不默认能够找到GTTR单一标量充分统计。rho/gap只是由GTTR目标记账引出的最低阶候选联系；是否需要更多状态，应依据后续问题而不是本轮无限扩维。

## 8. 预算与停止规则

主replay上界：18*(256-1)=4590 updates；smoke16，所以计划最坏4606。

每轨迹新4步点最多47个（复用原0、4、16倍数）；18条最多846次新整库monitor，加每轨迹最多2次重叠验证36次，加smoke3次，计划最坏885次整库eval。

|计数|计划最坏|本轮硬上限|
|---|---:|---:|
|真实replay optimizer updates|4606|4800|
|train forward/backward microcalls|9212/9212|各9600|
|monitor全4096行eval|885|900|
|network forwards总计|122492|125000|
|新增probe/target gradient或HVP|0|0|
|V_ctrl/V_final新调用|0/0|0/0|

实际按原T16往往显著少于最坏上界；在GPU前从旧hit表输出准确计划计数。不能看到时间估计后暗减样本或监测集。全部失败、重试、状态恢复都计费；如果身份或成本约束不满足则报告PARTIAL/BLOCKED，不自动借用旧余额。

墙钟只OBSERVATIONAL_SHARED_GPU，不设算法优劣门，不干扰其他GPU任务。

旧训练研究成本、本轮replay成本、假设单次部署成本分开。本轮重复处理同样数据不代表新的训练效率收益。

## 9. 最少测试与实现

复用原runner、数据与RNG工具，一个CPU分析模块＋一个replay wrapper即可，不重构。

CPU至少检查：
1. alive-area与direct capped hit逐路径相等，含未达标、恰好等于阈值、最后检查点达标。
2. GRID16_LAG16严格等价旧连续两次规则。
3. GRID4_LAG16不能被实现成连续两次4步；构造反例使错误实现失败。
4. 非单调曲线不能只查原 crossing 附近；全网格求最早合法pair。
5. 正确读取两start各自ell，不合并成共同阈值。
6. past-only特征不会访问landmark之后的目标值；非正速率返回abstain。
7. 真实elapsed-training-token轴与replay额外计算成本分开。
8. 固定首stream规则、共享参数和已冻结R1解不改变；CB1不再应用首步动作。

实现和测试的通过不代表状态指标到成本映射成立。

## 10. 交付与聊天必须呈现

目录建议：outputs/fr_gttr_cb1_cost_measurement_v01/。

- PROTOCOL_AND_SCOPE.md / config.json / selected_replays.csv。
- old_cost_identity.csv（全部36）。
- dense_monitor.csv（source标签、sum/count/mean、实际tokens）。
- detection_resolution.csv（旧/新grid、lag、首次合法pair、检测cost、差异、censor）。
- progress_cost_diagnostic.csv（全部past-only输入与未截断预测）。
- paired_cost_effect.csv（三臂全部比较，分seed/start/stream）。
- replay_identity.csv / resource_log / REPORT.md / KEY_DATA_FOR_REVIEW.md。

聊天直接展示：
A. 全36原成本恒等式最大残差；确认旧11/12检测tie事实是否复现。
B. 固定18路径原T16与新T4lag16，标明共同16步确认间隔。
C. 各配对FULL-LR_ONLY成本差、正负/tie、分起点结果；不能只给总中位数。
D. landmark16/32的gap/rho、预测与真实剩余成本、abstain、原始误差。
E. 哪一项仅是代数通过，哪一项有经验支持，哪一项未分辨。
F. 全部新计算及来源数据角色，无V_final。

不commit/push，不新增依赖，不修改OpenCode/GLM接入，不创建常驻monitor。完成CB0+CB1后停止，不自动训练新规则或进入R3。

## 11. 后续研究方向（不授权）

当前阶段不优化初始化，也不改变网络架构；最终希望求的是AdamW上的非预知更新规则参数phi，例如定义明确的eta/读出系数如何随状态选取。

CB1只检验一次求解改动的成本读出与最低阶状态联系。若有可分辨成本信号，后续另行定义和求解持续规则；若没有，不将旧首步解反复应用来“放大收益”。若gap/rho映射不足，不随机添加新指标凑拟合，更不能换目标。

相关外部方法参考：Kötzing & Krejca, First-Hitting Times Under Additive Drift, arXiv:1805.09415，说明条件漂移与期望首次到达时间可以在明确条件下连接；不将其定理无条件用于AdamW，也不把单路径rho_obs视作条件漂移界。
