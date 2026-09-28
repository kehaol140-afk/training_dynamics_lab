# FR Phase B：共享 GPU 环境下的 token 优先与计时修订

日期：2026-09-23。
执行端：现有 Claude Code + Kimi K2.7。
基础协议：`FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md`。

## 0. 适用范围

用户已说明远程 GPU 有其他任务占用，目前优先研究 token 效率，不用共享环境的墙钟比较判断模型优劣。本文件只修订基础协议中的计时门、成本读数与结论范围；不改变模型、状态、数据、动作、时域、预测封存、数值检查或真实分支设计。

执行时读取基础协议和本补充；冲突之处仅在本文件声明的范围内以本补充为准。不得据此增加样本、延长训练或自动进入 Phase C。

## 1. 科学指标与阶段边界

### 当前 Phase B

仍完成六个状态、三个 LR 动作、四步干预后释放，以及 H={1,4,16,64} 的预测/真实分支验证。

主要科学读数保持：
- 同一 control 目标上的动作差预测误差、SSE/TSS、regret；
- 在相同累计真实训练 tokens 处的 monitor NLL 差；
- 预选动作相对于 native / fixed-up / fixed-down 的效用；
- H4 选择在 H16/H64 的效果与可能反转；
- 建模所需的梯度、forward/backward、FD-HVP、虚拟算术和数据暴露量。

当前每条分支最多64次更新，不新增达标训练、不重估 ell、不因已达开发目标而提前停止既定分支。
`TOKENS_TO_TARGET_IMPROVEMENT=NOT_TESTED` 继续保留。

在主轨迹无跳步且配方保持每次1024 input slots时，H={1,4,16,64} 分别对应1024、4096、16384、65536 input slots。有效监督tokens从真实mask累积，不用input slots代替。

### 后续正式成本实验的方向（本文件不授权执行）

将“达到同一预先冻结的质量标准所需累计训练input tokens”作为主要效率目标，同时报告有效监督tokens、成功率、未达标以及固定预算末端NLL。维持一致的数据角色、监测/确认规则；不只平均成功者，不把检测时点当精确首次穿越。

这不要求完整参数轨迹预测，但需要一个真正用模型选动作的策略和相应固定基线。

## 2. 取消共享墙钟的科学门与本轮硬时限

原协议中的“所有GPU任务墙钟<=5400s”和“ETA超过墙钟cap则停止”在本轮不再作为硬限制或科学判据。

替换为：
- `hardware_mode = SHARED_GPU`。
- 墙钟可按现有低成本日志继续记录，标签为 `OBSERVATIONAL_SHARED_GPU`。
- 5400s只保留为一次性运维提示阈值，不计为预算超限，不触发模型FAIL，不自动删减数据或重跑。
- ETA只帮助用户了解进度；不根据吞吐或等待决定动作、LR、batch、评价频率、随机流、时域或停止位置。
- 不计算或宣称本轮time-to-target、墙钟speedup、Joules-to-target或能耗收益。
- 不增加GPU利用率采集、跨进程监控或“排除他人任务时间”的复杂修正。
- 不暂停、结束或修改其他用户/任务的进程。

运行继续受第3节的固定计算量上限控制。若外部宿主/作业平台超时、网络断开、人工终止或共享显存导致OOM，保存已完成结果和计数，报告`RESOURCE_INTERRUPTED`或`RESOURCE_BLOCKED`；这是执行覆盖状态，不是科学上的失败。

需要恢复时，仅在能够恢复完整theta/m/v、累计step、schedule、RNG、数据游标及预测封存状态的前提下，从安全边界继续。不得把恢复开销清零，也不得重新生成较有利的预测。不能安全恢复就交付PARTIAL，不偷偷改batch或跳过某些臂。

## 3. 计算量预算全部保持不变

以下上限仍同时生效，且全部包括失败尝试、smoke、恢复与复核：
- 实际optimizer updates <=1280；
- 全部网络forward <=14500；
- 全部backward micro-calls <=2920；
- 模型构建gradient forward/backward各 <=360；
- FD方向估计 <=50；
- monitor整库eval <=84；
- control整库eval <=90；
- V_final模型调用 =0。

基础计划仍为1168次真实更新、13562次forward、2546次backward micro-calls，按实际rank与缓存复用如实报告。

等待本身不消耗训练token，但重复执行、候选模拟、probes与失败计算不能因未计入训练主路径就隐去。

## 4. token与辅助成本账本

在现有日志中追加必要字段即可，不建立新数据库或全项目审计体系。

逐state/arm/H报告：
- `optimizer_updates_successful`；
- `optimizer_update_attempts`（若与成功数相同则直接注明）；
- `train_input_token_slots`；
- `train_nonpad_input_tokens`（可取得时）；
- `train_valid_loss_tokens`；
- `nll_sum`、`valid_eval_count`、`nll_mean`；
- `gain_vs_native`、`gain_vs_fixed_up/down`及paired SE。

按用途分别报告辅助数据与计算：
- SIM/probe/FD-HVP参考数据的处理次数和token slots；
- 开发目标梯度构建数据；
- control/monitor纯评价数据；
- smoke、失败与恢复；
- 虚拟optimizer算术步数、峰值显存（沿用已有记录）。

重复使用同一批数据：unique数据量与实际处理次数不能混同。虚拟64步没有读取新batch时，不计作64批真实训练tokens，但仍记录其算术计算。HVP不能强行换算为一个统一的“等效训练token”。

如果有跳步或失败，消耗的数据与计算保留在资源账；不要只统计成功更新就暗示完整方法更便宜。主轨迹逻辑token轴与实际重放/恢复开销分列。

## 5. 报告与测试的小修订

保持原科学误差门、模型覆盖、预测顺序和有限状态检查不变。

- 原`MODEL_COST`改为或拆为`MODEL_COMPUTE_FOOTPRINT`：只报告调用次数、数据处理量和显存，不能因共享墙钟较慢判科学失败。
- `TIME_TO_TARGET`、`WALL_CLOCK_SPEEDUP`、`ENERGY_BENEFIT`本轮均标`NOT_EVALUATED_SHARED_GPU`。
- `TOKENS_TO_TARGET_IMPROVEMENT`维持`NOT_TESTED_PHASE_B`；不能用本轮定长NLL差替代它。
- 可以画NLL-vs-training-tokens、相同H/token的动作差；不新增wall-clock算法排名图。
- 确认计时器没有进入LR/scheduler、采样、模型选择、门判定和训练token计数。
- CPU小测试：人为将elapsed time改为5401或更大时，数值门、动作选择与既定计算次数不变，只有运维提示改变。
- 若已有host级硬超时，不能声称已被移除；按第2节交付可恢复的PARTIAL状态。

最终报告开头写清本轮使用共享GPU、墙钟只作运行记录；直接展示固定token处的预测/真实结果和全部计数，不以“GPU受占用”掩盖数值错误或降低原模型精度门。

## 6. 执行方式

本补充不是新一轮实验，也不是重新冻结所有科学参数。最小修改现有config/计时守卫/报告字段后，按基础Phase B协议继续执行。

如果主实验尚未开始，记录本补充后执行。若已经开始且科学流程未改变，保留既有预测与结果，仅追加计时/报告变更记录，不重跑已完成部分；不能借补充修改结果前应固定的模型参数。

完成后停止，不自动进入Phase C，不新增控制器、模型或大规模审计。
