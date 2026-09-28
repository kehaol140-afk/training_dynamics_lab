# PROJECT_CONTEXT — Training Continuation under Changed Compute Budget
# (Optimizer State Adaptation for AdamW)

生成日期：2026-09-26。性质：只读审计 + 实验设计文档。本文档未伴随任何代码修改、任何训练运行、任何新算法提出。

---

## 1. 当前项目状态

### 1.1 仓库布局（2026-09-26 重组后，见 PATH_MIGRATION_20260926.md）

| 目录 | 性质 | 状态 |
|---|---|---|
| `legacy_fluctuation_response/` | 旧方向 FR（涨落-响应）+ GTTR 全部实验（git HEAD `6ac4942`，status 干净） | **只读冻结**。55 个 outputs 子目录，71G，6073 文件，迁移前后 SHA256 校验一致 |
| `minimind_base/` | minimind 原生训练框架独立副本（50 文件，与 legacy 内 minimind 逐文件一致） | 只读基线代码 |
| `optimizer_sensitivity/` | 新项目（独立 git 仓库，4 commits，无 remote）。已完成 F0 只读 bank 审计：`TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE`（G1c 1B 异常归因 SCALE_TREND + PRECISION_ARTIFACT） | 活跃，F0 已完结 |
| `audits/FULL_SERVER_AUDIT_20260926/` | 全服务器审计交接包（12 文档：91 实验索引、正/负结果、可复用资产、磁盘/git 审计、重组方案） | 只读 |

### 1.2 运行环境

- GPU：2 × RTX 5090（32G；当前 1 张部分占用）
- conda env：`fr_minimind`（torch 2.11.0 + CUDA 可用），路径 `/home/b205/miniforge3/envs/fr_minimind/bin/python`
- 磁盘：1007G 用 75%，余 246G（足够本方向第一阶段，无需下载 1B checkpoint）
- 运行约定：项目根 cwd + `PYTHONPATH` 注入；`CUDA_VISIBLE_DEVICES` 显式指定

### 1.3 关键冻结数据资产（本方向直接可用）

| 资产 | 位置 | 内容 |
|---|---|---|
| **v07 state bank** | `legacy_fluctuation_response/outputs/e3_confirmatory/v07/`（4.6G） | **56 个完整训练态**（8 seeds × 7 checkpoint 步位，含 step 46–82），每态 = model(原 dtype) + AdamW m/v/step + scaler + 四路 RNG + partition sha（`full_state` v0.2 格式，`test_full_state_roundtrip` 位级验证）。模型 h256/4L/seq128，语料 D0 4096 行子集 |
| G1a 26M bank | `outputs/generalization_g1a_26m_v01/bank` | 9 态（26.68M 参数），同格式 |
| G1b 100M bank | `outputs/generalization_g1b_100m_v01/bank` | 9 态（100.78M 参数） |
| Stage-C MID state | `outputs/stage_c_v04/mid_state_v04.pt` | 单个中间 checkpoint（step 64/128） |
| 语料 D0 | v07 冻结记录 SHA256 `1e657af9…b0f4` | 预训练 jsonl（minimind 语料子集） |

### 1.4 治理纪律（沿用，不重新发明）

- legacy 一切只读；新工作进独立目录，按 `optimizer_sensitivity/docs/LEGACY_PROVENANCE.md` 的 vendor+SHA256 登记模式取用
- 口径先于结果冻结（config json 预注册，红线条款：不为过关改阈值、不烧封存集）
- 封存评估纪律：`write_seal/verify_seal`（r1 中工具）+ V_monitor/V_final 分离（V_final 至今 0 调用）
- 未决风险：全部 git commit 仅在本机（push 失败，异地备份未建立）

---

## 2. 项目结构审计结果

### 2.1 minimind_base（原生训练框架）

| 项 | 位置 | 要点 |
|---|---|---|
| 训练入口 | `trainer/train_pretrain.py` | `train_epoch` + epoch 循环；AMP autocast + GradScaler（仅 fp16）；grad accumulation（默认 8）；unscale→clip(1.0)→step→update 时序 |
| model | `model/model_minimind.py` | `MiniMindForCausalLM`（HF PreTrainedModel），默认 768/8L/GQA/RoPE/QK-Norm/SwiGLU，**tied embedding**，自带 generate。默认配置 ~64M 参数 |
| optimizer | `train_pretrain.py:138` | `torch.optim.AdamW(model.parameters(), lr=5e-4)` — **只用 PyTorch 默认** β=(0.9,0.999), eps=1e-8, wd=0.01；无 param groups |
| scheduler | `trainer/trainer_utils.py:40-41` | 手写 cosine：`lr·(0.1+0.45·(1+cos(πt/T)))`，衰减到峰值 10%，**无 warmup、无 scheduler 对象**（纯 step 函数，不进 checkpoint） |
| checkpoint | `trainer_utils.py:63-106` | 原生 resume：model(**fp16 截断**) + optimizer state + scaler + epoch/step + world_size；原子写；`--from_resume 1` 恢复 + `SkipBatchSampler` 跳 batch |
| evaluation | `eval_llm.py` | **纯生成交互测试，无 loss/NLL 评测；全库无 validation set、无 train/val 划分** |
| dataset | `dataset/lm_dataset.py` | HF load_dataset(jsonl) 全量作 train；BPE tokenizer vocab=6400；seq_len 默认 340；labels pad→-100 |
| seed | `trainer_utils.py:54-61` | `setup_seed(42+rank)` + 每 epoch `setup_seed(42+epoch)`；cudnn deterministic |
| 其他脚本 | train_full_sft/lora/dpo/grpo/ppo/agent | 同骨架，pretrain→sft→rl 流水线，与本方向无关 |

**能力清单（六个重点项）**

| 能力 | minimind_base 原生 | legacy FR 框架 | optimizer_sensitivity（vendored） |
|---|---|---|---|
| AdamW state 保存 | ✅（resume pth，但 model 被 fp16 截断） | ✅ `full_state.py`（原 dtype + m/v/step） | ✅ verbatim |
| optimizer state 恢复 | ✅（`--from_resume 1`） | ✅ `load_full_training_state` + 位级 roundtrip 测试 | ✅ verbatim（**当前无调用路径**） |
| continuation training | 部分（同计划续训） | ✅ `stage_c_runtime` + `theta_protocol.pyc`（`run_theta_branch`）；LR 可在线改，schedule 可注入；**改 beta 无现成路径** | ✅ 原语在，无 runner 接线 |
| hyperparameter sweep | ❌ 无通用 harness（GTTR 各轮手搓网格） | ❌ 同左 | ❌ |
| seed 控制 | ✅ 基础（setup_seed） | ✅ 强：四路 RNG capture/restore（含 CUDA philox 状态）+ stream seed 派生 + 独立 Generator batch manifest + 位级重放测试 | ✅ verbatim |
| validation NLL | ❌ 无 | ✅ `eval_observables.evaluate_token_losses`：固定 eval-pool 逐 token CE（mean/p95/p99/CVaR99），RNG/模式保护 | 部分（eval 模块未 vendor） |

### 2.2 legacy FR 框架核心模块（`legacy_fluctuation_response/research/src/fr/`，v0.6.0-cn）

- `full_state.py` — 完整训练态 Z_t = model + optimizer(m/v/step) + scaler + metadata(位置/签名/partition sha) + RNG。save/load 原子、版本校验、RNG 最后恢复。**本方向 checkpoint 格式的事实标准**
- `rng.py` / `stream_rng.py` / `batch_manifest.py` / `batch_stream.py` — 确定性骨干：四路 RNG roundtrip、流种子派生、显式 batch 清单落盘哈希（绕开 DataLoader）
- `native_adamw_guard.py` / `native_schedule.py` — 断言原生 AdamW、记录 optimizer_signature；逐 micro-batch 写 LR 复刻原生时序
- `stage_c_runtime.py` — train pool 内确定性 future batch + `get_continuation_lr`（沿预注册 LR 轴续训）
- `eval_observables.py` — eval-pool per-token NLL 统计（`[FR-OBSERVATION-ONLY]`）
- `pilot_partition.py` — probe/eval/train 三块互斥划分（probe/eval 从第 0 步起不入训练池，带 sha）
- `theta_protocol.pyc` — **仅字节码**（源码丢失，历史版本在 recovery_candidates/ 与 _fr_v05/v06_incoming）。接口可从 test_theta_protocol.py 恢复：load Z_t → 锁流 → 续跑 horizon → 可选首梯度捕获/clip 前停止。**新方向不应依赖 pyc**（见风险 §7）
- `perturbation.py` / `direction.py` / `fluctuation_stats.py` / `response_stats.py` / `mapping_stats.py` — 扰动-响应测量（旧方向专属接口，但 direction 的梯度/位移投影是只读可复用的观测工具）

### 2.3 optimizer_sensitivity（新项目，vendor 模式范本）

15 个文件从 legacy verbatim 复制（SHA256 双向核对，仅 minimind_adapter_v02 改 2 行 import）。F0 已完结（只读 bank 审计，判定见 `docs/F0_EXISTING_BANK_SENSITIVITY_AUDIT.md`）。**其中 `adamw_tangent_step`/对角增益 G_i 的统计代码（`f0_bank_audit_v01.py`）对本研究 secondary 指标（effective LR、v 分布）直接可复用。**

---

## 3. GTTR 历史实验阅读总结

### 3.1 时间线与裁决（2026-09-21 → 09-26）

| 轮次 | 问题 | 裁决 |
|---|---|---|
| C0/C0X/C1/C1L | 局部一阶几何场（g+Hs 再 clip）可否预测真实更新后梯度场 | field 在选定 anchor 跨库 PASS（C1L: E≈0.22-0.25）；shadow UNRESOLVED→带保留 PASS；C0 原判被 C0X 勘误（门规则误用） |
| Phase A | native continuation 基线 + tail schedule（`finite_cosine_then_hold_5e-5_v01`） | PASS，为 GTTR 各轮提供 V_monitor/ell/start 语义 |
| Phase B v3 | GTTR 动作前评分（冻结参考梯度场 + FD-HVP 低秩场模型选动作） | **GATE_STOP**：S3 局部算子 FD 门失败（真实强非线性，非实现问题）；v4 建议未执行，冻结 |
| D1 | R1 读出方向固化为训练规则（QHAdamW α=0.1 RAW/NORM vs 3 档固定 LR） | **NO_OBSERVED_GAIN**（不如固定 LR 网格）；NORM 匹配劣于 RAW |
| R1 / R2B | 二次模型辨识 / 一次解干预+共同续训 | R1: 10/12 局部方向支持；R2B: h256 G_struct +3.36e-4（SE 5.63e-4，**不显著**）；3/12 即时有利后 h256 **反转** |
| CB1 | 成本恒等式与 rho 剩余成本预测 | 恒等式 36/36 精确；rho 闭合误差中位 0.20 → **不可作预测子** |
| F0 → CALFIX → IC → F1 | 全时域前向灵敏度 + 终点 hypergradient 求解 LR 规则 | 实现闭环成功（合成 5.6e-17；b 坐标 SENS=FD 校准通过），但 a 坐标 FD 噪声主导 |
| F2a / F2b | 已校准导数能否预测有限步长外真实 J(b) | **DIRECTION_NOT_CONFIRMED**；二阶外推失效，实际最优 b 与一阶梯度**反向** ——"导数算得对但用不上"，GTTR 实质终局 |
| 09-26 | 仓库迁移重组，GTTR 线终结 | 保护 commit `6ac4942`（仅 F0/CALFIX/F1 前沿代码入库） |

### 3.2 资产分类（明确标记）

**✅ 可复用（通用基础设施，与 GTTR 无关）**

- Transformer 模型 + 原生训练循环复刻：`minimind_base/` 全套 + 各 runner 内联的逐行复刻循环（范式见 `e3c_prepare_state_bank_v07.py:170-191`）
- checkpoint：`full_state.py`（save/load/capture，含 roundtrip 测试）
- continuation runtime：`stage_c_runtime.py`、`prepare_mid_checkpoint_v04.py`（C0 制备范式：warmup 到 t0 → 存 Z_t → 断言 partition 无泄漏）
- RNG/数据确定性：`rng.py`、`stream_rng.py`、`batch_manifest.py`、`batch_stream.py`、`pilot_partition.py`
- optimizer 语义守卫：`native_adamw_guard.py`、`native_schedule.py`
- evaluation：`eval_observables.py`（eval-pool NLL）；`f0_bank_audit_v01.py`（v/m 分布、对角增益统计——secondary 指标直接可用）
- logging/统计：`response_stats.py`、`confirmatory_stats.py`、`cross_seed_stats.py`、`hashing.py`、`write_seal/verify_seal`（r1 内）
- 治理范式：config 预注册 + 红线 + SHA256 provenance + 封存集纪律 + runner 三件套（run/analyze/test）模式
- 冻结 bank：v07（56 态）/G1a/G1b——**多 seed 多 checkpoint 要求的原料已在手**

**❌ 旧方向专属（不迁移、不维护）**

- HVP/切线-伴随递推：`a0c_adamw_response_propagation_v01.py`（adamw_tangent_step 除外，已被 optimizer_sensitivity vendor）、`a0h_adjoint_v01`、`a0j_future_sensitivity_cache`、`a0h15_*`、`a0e/a0e2`
- hypergradient solver / 终点规则：`f0_terminal_rule_v01` 及其 runner、`f0_calfix_run_v01`、`f0_ic_adapter_v01`、`f1_closure_run_v01`
- GTTR controller / 评分 / 规则：`phase_b_gttr_score_v03`、`c0/c1/cb1/d1/r1/r2b` 全部 src+runner+tests、`phase_a_tail_schedule_v01`
- 扰动-响应测量协议：`perturbation.py`、`direction.py` 的 probe 构造、E1/E2/E3 线（作为科学结论保留，作为代码不再使用）

### 3.3 对新方向的直接教训（必须写进设计）

1. R2B/D1 反复显示：**短期（h1–h16）收益与 final NLL 脱钩甚至反转** → 本方向判据必须锚定 final validation NLL
2. D1 显示干预收益不超过**固定 LR 网格边界** → arm A（strong baseline）必须真调参，"C 打败未调 baseline"不算数
3. Phase B 显示局部导数信息在真实步长上不可靠 → 本方向第一阶段**不做任何基于导数的方法**（与用户"不假设任何方法有效"一致）
4. 位级重放/CRN 是本仓库最成熟的能力，是配对实验降方差的正确工具

---

## 4. 新研究问题定义

**RQ**：一个 Transformer checkpoint C0（含 AdamW m/v/step）在计算预算变化后需要以新训练计划（预算 B、新 LR schedule、可能新 β）继续训练时，继承的 optimizer state 是否仍适配新计划？对 state 做适配/重构能否优于"继承原 state + 认真调 LR/WD"的强 baseline？

**不预设任何方法有效。第一阶段只做方向存续判断。**

### 4.1 变量定义

- **C0（checkpoint）**：完整训练态 Z_t0（full_state v0.2 格式）。 elimination round 直接使用 v07 bank（8 seeds × 7 步位），满足多 seed 多 checkpoint 要求且零新 C0 训练成本
- **B（continuation budget）**：C0 之后的新预算（optimizer steps），且新计划与原计划不同——操作化为：原计划剩余预算为 T_rem，新预算 B ≠ T_rem（首轮取 B = T_rem/2，即"预算削减"情景），所有 arm 的 LR schedule 在 B 内重新推导（如压缩 cosine）
- **场景说明**：v07 bank 的 C0 位于原 cosine 中段（step 46–82/128），对应"训练中途预算重规划"情景；"训满原 schedule 后追加预算"情景留作第二阶段（需新制 C0，h256 成本可忽略）

### 4.2 实验臂（4 臂，均为普通 AdamW 变体，无新算法）

| 臂 | optimizer state | 超参数 | 回答什么 |
|---|---|---|---|
| **A：Strong continuation** | 继承 m/v/step | **原 β**；LR/WD 小网格调优（schedule 按 B 重推） | 强 baseline：认真调参的继承续训。C 必须打败它 |
| **B：Hyperparameter migration** | 继承 m/v/step | **改 β**（如 β2 0.999→0.99）+ 新 LR schedule（LR 取 A 网格最优，保持可比） | HP 迁移时陈旧 state 是帮还是拖（B vs D 隔离） |
| **C：State reconstruction** | **不继承**：moment 以非继承方式初始化（最小选项：m 清零保 v、保 m 清零 v；不做任何"聪明"重构算法） | 与 B 完全相同 | 核心问题：非继承 state 初始化在新 HP 下是否有价值 |
| **D：Reset baseline** | m、v 全清零（全新 AdamW，state step 归零） | 与 B 完全相同 | 对照：changed-HP 下重置的下界 |

**关键对比**：C > A（核心，见 §6）；B vs D（继承 state 在新 HP 下的净效应——若 B < D，说明 state 失配，才有 C 的存在意义）；A vs D（原 HP 下继承的价值基线）。

### 4.3 评价

- **Primary**：final validation NLL（预算 B 耗尽时，封存 eval-pool 逐 token CE mean；附 p99/CVaR99）。eval-pool 沿用 pilot_partition 纪律：从 C0 训练池排除、sha 封存、判定只读一次
- **Secondary**（诊断，不作为存续判据）：
  - loss recovery speed：恢复到参照 loss 水平所需 steps（沿 Phase A ell 阈值语义）
  - pre-clip gradient norm（逐 step）
  - moment norms：||m||、||v||（分参数组）
  - effective learning rate：逐坐标增益 G_i = lr(1−β1)/((1−β1^{t+1})(√v̂+eps)) 的分布统计（直接复用 optimizer_sensitivity 的 `f0_bank_audit_v01.py` 统计口径）
- **监控**：V_monitor 周期性 NLL 曲线（不烧 V_final）

---

## 5. 第一阶段实验设计（最小淘汰实验）

### 5.1 设计矩阵

- C0：v07 bank 全部 56 态（8 seeds × 7 步位）——多 seed 多 checkpoint 由冻结资产直接满足
- B：首轮固定单一预算（如 64 optimizer steps，B = T_rem/2），排除预算维度交互
- 臂：A（LR×WD 网格，约 3×2=6 格）+ B/C/D（各 1–2 个 state 变体）≈ 10 runs / (seed×step) 组合
- 总量：56 × 10 ≈ 560 条 continuation run，每条 64 steps（h256/4L，RTX 5090 上每条分钟级）→ **总计算量约几十 GPU 时以内，属"最小淘汰"量级**
- 配对降方差：同 C0 的所有臂共享 continuation 数据流（同一 batch manifest + stream seed，CRN）；不同 seed/步位间做配对统计

### 5.2 调参纪律（防泄漏）

- A 臂的 LR/WD 网格选择只看 V_monitor 曲线 + train loss，**不看 V_final**
- final 判定一次性读取封存 eval（write_seal → 全部 run 完成 → verify_seal → 统计）

### 5.3 统计口径（预注册，沿用 v07 惯例）

- 配对比较（同 seed × 同步位配对差值），permutation test（预注册次数，如 20000）+ bootstrap CI（10000），Holm 校正 α=0.05
- 报告：逐步位方向一致性（7 个 t0 中 C>A 的比例）、逐 seed 一致性（8/8？）、效应量分布
- 判定主体：8 seeds × 7 步位聚合的 C − A 配对分布

### 5.4 与 GTTR 教训的对齐

- 判据 = final NLL，不是短期 loss（§3.3-1）
- A 是调过参的强 baseline，且若 C 的收益落在 A 网格边界内 → 视同"普通 LR/WD tuning 可获得"（§3.3-2）
- 无导数、无 controller、无 solver（§3.3-3）

---

## 6. 停止条件（预注册）

**继续研究条件（全部满足）**：

1. C > A：final validation NLL 上 C 臂显著优于 A 臂最优格（配对检验过预注册阈值）
2. 多 seed：≥6/8 seeds 方向一致
3. 多 checkpoint：≥5/7 步位方向一致（不依赖单一 t0）
4. 不依赖短期 loss：若 C 仅在 recovery speed/早期 monitor 曲线占优而 final NLL 不占优 → **不算通过**
5. 收益不可被普通调参替代：C − A 效应量超出 A 网格外推（必要时扩 A 网格复核一次）

**停止条件（任一触发即停**，写结题报告，不追加轮次）：

- C ≤ A（含统计不显著且效应量微小）
- 收益可通过 LR/WD tuning 获得（扩 A 网格后差距消失）
- 仅在单一 seed/单一 checkpoint/短期 loss 上出现优势
- 实现层面发现根本性混淆（如 β 修改与 bias correction 交互无法定义干净，见 §7）

---

## 7. 风险点

1. **β 修改无现成代码路径**（全仓无恢复后写 betas 的代码；`optimizer.load_state_dict` 会原样覆盖 param_groups）——需先 load 再改 group 的 helper + 单测验证 bias correction step 计数语义正确。这是唯一必须新写的"机制性"代码，属于 plumbing 而非算法
2. **β2 变化与 bias correction 交互**：继承的 state step t 使 (1−β2^t) 语义改变，moment 范数不可直接跨 β 比较——C/B/D 的 moment norm 曲线解释需在 config 中预注册口径
3. **模型尺度外推**：h256/4L 是极小模型，淘汰结论只裁决"方向是否值得继续"，不构成对真实规模的主张；若通过，第二阶段用 G1a 26M bank 复核，再谈更大规模
4. **场景覆盖**：v07 bank C0 位于 cosine 中段，"训满后追加预算"情景未覆盖——若淘汰轮通过，需补该情景
5. **theta_protocol.pyc 依赖风险**：sourceless 字节码，勿作为新 runner 基础；continuation runner 应参照有源码的 `e3c_prepare_state_bank_v01.py` 模式新写（普通训练循环，无 GTTR 依赖）
6. **A 网格覆盖不足**导致假阳性（C 赢了弱 A）——网格边界处必须外扩复核（§6 停止条件 2）
7. **eval pool 规模**：4096 行子集的 eval pool 较小，NLL 均值的标准误需在预注册中给出最小可检测效应；若功效不足，宁可扩 eval pool 重制 C0，不可事后换判据
8. **单机 git 无异地备份**——新项目开始前应先建立 remote（HTTPS+token 或代理）
9. **GPU 占用**：5090 #0 当前有 12G 占用，排程时显式 `CUDA_VISIBLE_DEVICES` 并先探测

---

## 8. 下一步实施计划（未写任何代码，全部待执行）

| 阶段 | 内容 | 产出 | 门 |
|---|---|---|---|
| **P0 项目骨架** | 新建独立目录（建议 `continuation_adaptation/`），按 vendor 模式从 legacy 取用：full_state / rng / stream_rng / batch_manifest / batch_stream / stage_c_runtime / native_adamw_guard / native_schedule / eval_observables / pilot_partition / hashing + optimizer_sensitivity 的 f0_bank_audit 统计件；逐文件 SHA256 登记进 LEGACY_PROVENANCE.md；建 git remote | 新仓库 + provenance | 全部 roundtrip 单测通过 |
| **P1 预注册 config** | 冻结：C0 集合（v07 56 态清单+sha）、B、4 臂定义（含 β、LR/WD 网格、schedule 重推公式、state 初始化选项）、eval 判据、统计口径、停止条件（本文 §4–§6 直接转成 json+md） | configs/*.json（date_frozen） | 口径评审通过前零 GPU |
| **P2 最小机制代码** | continuation runner（复刻 e3c 循环模式）+ load 后改 betas 的 helper + NLL/grad norm/moment norm/G_i 日志 + 单测（beta 修改正确性、state roundtrip、CRN 位级重放） | src + tests | 单测全绿，含一次 1-seed×1-step 干跑位级核对 |
| **P3 smoke** | 1 seed × 1 步位 × 4 臂 × 短 horizon（GPU 分钟级），检查日志/指标/封存流程 | smoke 报告 | 无异常才进 P4 |
| **P4 淘汰矩阵** | 56 C0 × ~10 runs，两卡并行 | 原始 run 制品 | — |
| **P5 分析与裁决** | 封存读取 → 配对统计 → 对照 §6 停止条件 → 写结论报告（通过/停止，均为有效结果） | REPORT.md + 决策 | 一次性判定，不追加轮次 |

**红线（继承自 optimizer_sensitivity）**：不改 forward AdamW 规则；不为过关改阈值；legacy 只读；不预设方法有效；不提出复杂 optimizer；不添加新理论名称（一切命名用描述性 ID）。

---

## 附：本文件的信息来源

- 四路只读审计（minimind_base / legacy research/src/fr / GTTR 历史线 / optimizer_sensitivity + audits），均带 file:line 证据
- `PATH_MIGRATION_20260926.md`、`audits/FULL_SERVER_AUDIT_20260926/`（00/05/08/11 号文档）
- `optimizer_sensitivity/docs/{LEGACY_PROVENANCE,F0_EXISTING_BANK_SENSITIVITY_AUDIT}.md`
- legacy `docs/FR_PROJECT_STATUS.md`（终止于 §49a）及根目录各 GTTR 协议文档
- 环境实测（nvidia-smi / df / torch 2.11.0+CUDA / v07 bank 文件实查）
