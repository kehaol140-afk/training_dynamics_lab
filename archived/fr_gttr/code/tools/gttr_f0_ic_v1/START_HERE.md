# GTTR-F0-IC v1：理论—生产实现一致性检查

执行端：OpenCode＋GLM5.3。性质：当前 F0 的有限源码核验与独立 CPU 实验；不是新 optimizer、不是完整 Transformer 收益验证。

**用户转交本文件即请求执行本文件限定的工作；本轮不授权任何 GPU 运行、真实大模型续训、grid/solver/audit/confirm。**

## 0. 本轮要交付什么

不是再写一份计划，也不是只重复附件的独立 demo。必须把当前修复后的**真实生产更新/SENS函数**接到一个可独立检查的小型问题上，检验其是否实现了声明的数学关系；并解释旧、新 h16 数值为何不一致。

最终分别给出：

1. FORMULA_REFERENCE：独立数学参考的运行结果。
2. PRODUCTION_KERNEL：实际生产函数在所测 CPU 夹具上的一致性。
3. LOSS_AND_READOUT：训练梯度/目标梯度/时点/参数映射的一致性。
4. HORIZON_PROVENANCE：旧 scan 与 CAL-FIX 是否计算同一对象。
5. FULL_TRANSFORMER_256_SENS：保持 NOT_REVALIDATED/UNRESOLVED，本轮无权改为 PASS。

核心问题不是“如何让导数数值变小”，而是“计算的是否是同一个函数、同一个状态转移的导数”。

## 1. 固定项与范围

- 最终目标仍为原 T=256、固定训练 slots 的终点 V_design。
- 规则仍为 eta[t]=w[t]*exp(a+b*psi[t])/Z(b)，psi[t]=2t/255−1，Z在全部256位置求和。
- 不改变网络、batch、train support、mask/reduction、clip阈值、Adam eps、beta、wd、参数组、共享参数或原参考精度语义。
- CPU 夹具可以在 h=16/32/256 读出，但 schedule_total 始终256；它们不是新的真实训练目标或新增用户模型样本。
- 原 CAL-1/CAL-2/CAL-FIX 全部保留。不得覆盖、将不同版本拼成一个运行，或利用旧丢失字段补造历史。
- 本轮不重新解释整个 A0—CB1 历史，不搜索新理论，不添加 FSL/ELR/DMFT/新状态网络。
- 只允许对**已被独立反例确认的实现偏离**做最小修复。每项必须保留修改前证据、差异和修复后检查。至多两轮有明确证据的修复—CPU回归；仍不一致就停止，不能无限改到通过。

## 2. 输入和现场阅读：只读必要文件

从仓库实际路径定位，不假装本文件已经看到远程源码：

- 原 F0、CALFIX 协议和两轮 config/manifest/source_snapshot。
- `research/src/fr/f0_terminal_rule_v01.py`、`research/scripts/f0_terminal_rule_run_v01.py`（实际路径不同则登记真实路径）。
- 实际调用的 clip、schedule、训练 loss、目标评价与 HVP helpers。
- 原 CPU 测试、CAL-1/CAL-2/CAL-FIX 的 ledger、horizon_scan.csv、terminal_values.csv、sensitivity_check.csv、sensitivity_norms.csv。
- 附件 `gttr_consistency_reference/verify_reference.py`、README、历史 JSON。

附带的历史 JSON 是前一轮本地参考结果，不是本轮已经 PASS。需在服务器 CPU 环境重新执行。

建立一个小表 `source_map.csv`：功能、真实模块路径、函数名、行号范围、源码SHA、生产/参考/测试角色。用实际 imported_module.__file__ 核对，防止运行了另一份同名模块。不得导入有 GPU/训练启动副作用的入口；若存在副作用，应把问题登记并作最小入口隔离，不能偷偷执行 main。

输出目录：`outputs/fr_gttr_f0_implementation_consistency_v01/`。

## 3. IC-1：先解释旧、新 h16 是否可比（0 GPU）

用户已给出的待核对现象：

- 旧 scan h16 的 a：SENS约−22.11378、FD约−22.10589。
- CAL-FIX h16 的 a：SENS约−6.2818e4、FD(.01)约+.4857、FD(.005)约+.0922。

逐项导出并对比：

`run_id / source_sha / state_sha / seed / n0 / true_parameter_step / schedule_total / executed_updates / readout_h / first_batch_id / stream_id / rng_recipe / gm_at_lr_call / target_id / target_count / reduction / mode / dtype / backend`。

必要时从已存源码和配置在CPU重建两条 eta 序列及 d_eta/da、d_eta/db；这是重建配置，不是恢复缺失的历史运行证据。特别核对：

1. 旧 scan 是否调用了 T=h，从而改变 w/psi/Z，而不是从 T256 轨迹截取前h步？
2. n0、执行更新数和读出位置是否有+1/−1差异？
3. 目标是否同一目标，训练 minibatch loss 是否被误当 V_design？
4. 旧诊断是否经过已确认的初始化字典别名路径？
5. 中间 q、S 是否 clone 后保存，是否被后续步骤覆盖；标量是否当场计算并写入？
6. 数据/RNG/mode/backend是否真的匹配？

每个原因只按证据写：CONFIRMED_DIFFERENCE / MATCHED / NOT_RECOVERABLE。旧源码无法恢复就写 NOT_RECOVERABLE；不能按当前代码重演后称“证明旧代码当时这样跑”。

若不能恢复旧 h16 身份，报告它不能继续用作“16步通过”的证据；这不妨碍检查当前生产代码。

## 4. IC-2：先运行独立参考，再验证真实生产内核（CPU）

### 4.1 附件原样基准

将参考目录复制到本轮 output/reference_run/，在那里运行，保留交付附件不变：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python outputs/fr_gttr_f0_implementation_consistency_v01/reference_run/verify_reference.py
```

记录命令、退出码、PyTorch版本、CPU dtype 和新生成 JSON。不能安装新的训练框架。

附件有4个参数、2个块、固定合成数据、非零旧moments、解析g/H、三种clip阈值和h16/32/256。比较显式前向灵敏度、独立完整反向求导、两尺度FD。这只是独立数学参考，不是生产函数检查。

### 4.2 必须做的连接：调用真实生产函数

新增一个薄适配器，将附件的4参数拆成两个实际参数张量，向当前修复版生产 optimizer/SENS 内核喂入同一批合成输入。

- 适配器只做输入输出打包、明确的dtype转换、函数签名匹配；不能重写生产裁剪、moments、P或S公式后称它是生产路径。
- 输出记录实际 import 路径、函数源码SHA及调用证据。
- 合成多步测试使用解析训练g/H，因此首先隔离并检查optimizer/SENS递推，不假装它也已经验证真实Transformer HVP。
- 每步记录两种实现的theta/m/v及S_theta/S_m/S_v；不能只比较最终一个标量。出现不一致时，报告最早的t、状态分量和方向。
- 合成参考中的ridge、数据和阈值仅用于测试夹具，不加入真实F0训练。

**精度纪律：**

若生产内核原本支持FP64，做同输入同精度比较；若内核硬编码FP32，保留原路径并做独立同FP32 primal/reverse参考。FP64结果另作参考，不能为了过门把生产函数改成全FP64，也不能对不同dtype要求逐位相等。

预设工程核对容差（只针对本轮合成夹具，不追改原F0门）：

- FP64同对象：err_norm <= 1e-10 + 1e-8*max(norm_prod,norm_ref)。
- FP32或混合精度同primal语义：err_norm <= 1e-6 + 1e-4*max(norm_prod,norm_ref)。
- 同步输出err_norm/max(norm_ref,1e-30)、最大逐元素差、对象范数和分支距离。工程floor不是科学噪声估计。
- 若参考与生产的primal先不一致，先定位primal，不解释S为正确。
- 不用FP32的极小FD作为唯一真值；附件FP64 FD和独立reverse分别标注。达不到容差就定位或UNRESOLVED，不能放宽直到通过。

### 4.3 有限的必要反例

复用已存在测试，只有缺失时补充，不重复建设一套测试平台：

1. **跨块clip**：g=(3,4),h=(4,−3),tau=1,eps_clip=1e−6。正确约(.8,−.6)，错误逐块dot约(.512,−.216)。生产函数必须与独立值函数AD/FD一致。
2. **非零初始敏感性和无别名**：修改一方向S_m不应影响S_theta/S_v/另一方向。字典不共享不代表tensor storage一定不共享，两者都查。
3. **全状态一步JVP**：active/inactive clip、非零m/v与S、d_eta直接项、wd、bias correction都进入。
4. **零v与None/零梯度**：按生产声明处理，不能引入epsilon或v_floor掩盖。结构零与未定义分支分开。
5. **规则导数**：Z(b)项、sum eta=e^a sum w、sum d_eta/db=0；规则时钟始终256。
6. **同时读取旧状态**：S_theta/S_m/S_v更新不能因为循环顺序使用已被改写的本步值；两方向不能串扰。

若生产接口实在无法由薄适配器调用，只标 PRODUCTION_ADAPTER_BLOCKED 并给确切依赖；不要复制其核心公式建另一份“等价实现”后宣称生产通过。

## 5. IC-3：HVP与目标读出检查，不让“正确内核＋错误输入”漏过

### 5.1 HVP输入/输出与参数映射

静态检查实际HVP路径：

- scalar loss是实际两个microbatch及真实权重/ignore mask的组合；方向detach；params与grads的排列一致。
- 唯一参数顺序，tied weight只计一次；None与结构零的约定一致。
- dot的fp64累积不等于FP64图内HVP；每段dtype如实列出。
- 使用给定S_theta方向，不对S的构造再次求导；每步图释放但S数值保留。

若现有HVP helper可在CPU小型模型上调用，使用一个包含共享weight和两个不同有效token数microbatches的固定小夹具做对照。比较实际helper的HVP与独立构造的同scalar loss参考。不能把真实4.98M模型搬到CPU跑256步，不能为此重构生产网络。

如果helper深嵌runner无法独立调用，明确标HVP_INTEGRATION_NOT_EXECUTED并保存完整源码，不能用仅检查optimizer内核的结果替代。

### 5.2 目标值与目标梯度必须来自同一个标量

原V_design声明为全库sum/count。检查目标值与q是否都对应：

    V = sum_j nll_sum_j / sum_j valid_count_j
    q = grad_theta V

不要在各批valid_count不同的情况下，让value用sum/count、q却用mean(batch_mean)。不要把这一目标约定反过来“修正”训练loss；训练仍按原配方。

CPU两microbatch夹具必须有不同valid_count。独立参考可一次构造完整目标再求grad，生产路径按其分批累积；检查value与q而非只看value。

在明确已保存字段支持时，离线检查实际h16/32/256的q_norm、S_norm、dot(q,S)及分块和。缺向量就写NOT_SAVED，不从各自范数反推dot。

重点检查：

- q_h对应theta_h，而非theta_(h−1)或后来theta_T；
- S_h在读出时立即clone/标量化，不保存会被后续修改的引用；
- eval mode、grad buffer、RNG和data cursor按原合同隔离/恢复；
- value-only与SENS参数轨迹一致不能证明这里的q或dot正确。

可微参考和浮点程序不要求逐比特导数一致；但必须能说明当前q、HVP、S分别属于哪个参考对象。

## 6. 有限修复规则

本轮至多两轮有具体反例的最小实现修复。必须保留：

`defect_id / file:function:line / expected_formula / failing_input / before_output / independent_expected / minimal_diff / after_output / test_command / exit_code / source_hash_before_after`。

不允许通过修改F0目标、T、eta范围、eps、精度配方、mask或CPU比较容差来修复失败；不能删除失败case。

修复后重跑相关独立检查和已有回归，但不触发GPU。若没有找到缺陷，就写“已检查组件未见缺陷”，不得写“全部代码正确，因此只能是混沌”。

## 7. 算力、数据与执行边界

- 在本轮CPU进程启动前设置 CUDA_VISIBLE_DEVICES=""；不得调用.cuda()或借助后台worker绕过。
- GPU kernel、真实训练optimizer updates、新的真实网络q/HVP、任何实际V_design/V_confirm/V_final forward，均为0。
- 允许：源码读取、CSV/JSON复算、必要small-tensor检查点元数据读取、独立CPU合成实验。
- CPU synthetic updates/AD/FD耗时单列，不混入原真实轨迹更新数，也不虚报为“全轮无计算”。
- 不反复扫描合成参数，固定附件九组为主，有限反例为辅，最多两轮证据驱动修复。
- 只查看必要依赖，不重新审查整个A0—CB1，不运行全部项目tests导致其他实验启动。
- 如果旧源码/数据/夹具缺失，保存缺口，不安装新框架，不重新训练补历史。
- 不commit/push、不改OpenCode/GLM接入、不关闭其他任务、不创建常驻监控。
- CPU墙钟仅记录，不作为科学结论。

## 8. 最終制品与聊天必须呈现的内容

最小输出：

1. `implementation_contract.csv`：公式→生产函数→参考→测试→误差→覆盖边界。
2. `h16_provenance_comparison.csv`：旧/新诊断的真实对象对照，不可恢复字段明确标记。
3. `production_reference_compare.csv`：九组合成轨迹，实际生产内核与独立参考；首次分歧和theta/m/v/S分项。
4. `hvp_readout_checks.csv`：哪些生产路径实际执行、哪些仅静态核对。
5. `defects_and_minimal_fixes.md`：若无缺陷，明确范围，不编造一个修复。
6. `source_snapshot/` 与SHA清单；`reference_run/`原样参考的新结果。
7. `KEY_DATA_FOR_REVIEW.md` 与 `REPORT.md`、执行命令/退出码/环境、`resource_log.json`。
8. `review_bundle.zip`：打包关键源码、薄适配器、测试、上述小CSV/JSON和简要报告；不打包模型权重、语料、token/password或无关历史目录。

聊天中直接展示：

- h16冲突是否同对象、已确认差异是什么；
- 生产step是否真的调用，模块SHA及两三个最关键数值；
- 独立reference / production-kernel / HVP / target-q四层分开；
- 每个已确认bug的修复前后反例；
- 剩余哪些必须用真实模型检查，本轮为什么还不能证明T256 SENS正确。

状态字段建议：

    FORMULA_REFERENCE = PASS / FAIL / NOT_RUN
    PRODUCTION_KERNEL = PASS_WITHIN_TEST_SCOPE / FAIL / BLOCKED
    LOSS_AND_READOUT = CHECKED_WITHIN_SCOPE / DEFECT_FOUND / PARTIAL
    HORIZON_PROVENANCE = SAME_OBJECT / DIFFERENT_OBJECT / NOT_RECOVERABLE
    KNOWN_DEFECTS = NONE_FOUND_IN_CHECKED_SCOPE / FIXED_CPU_ONLY / UNFIXED
    FULL_TRANSFORMER_256_SENS = NOT_REVALIDATED
    GRID_SOLVER_AUDIT_CONFIRM = NOT_RUN

最终不得只给一个“理论实现没有问题”或“全部PASS”。

## 9. 停止后的决定

- 若找到新缺陷：本轮只最小修复＋CPU回归，提出一次有明确入口的GPU复测建议，不执行。
- 若旧h16并非同对象：追加口径解释，不能把差异解释成新动力学证据。
- 若小系统生产一致，但HVP/读出未覆盖：明确下一检查范围；不能直接重跑完整F0。
- 若所有当前可查组件无缺陷：整理review_bundle交由外部逐行检查；结论仍是当前有限覆盖，不是长期SENS、真实FP32数值或终点优化成功。

**本轮到此结束。目标是获得能定位问题的证据，不是为了恢复F0而制造一个PASS。**
