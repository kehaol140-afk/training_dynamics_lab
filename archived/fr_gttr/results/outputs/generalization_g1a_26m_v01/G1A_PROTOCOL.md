# G1a — 26M 模型规模泛化验证协议（FR_G1A_26M_SCALE_GENERALIZATION_V01）

预注册冻结时间：2026-09-16T13:30Z（任何正式结果打开之前）。本文件与
`research/configs/g1a_26m_v01.json` 内容一致；冲突时以本文件 + config 联合为准。
判定阈值在本文件 §10 冻结，不得根据结果修改。

## 1. 科学问题

唯一问题：当模型参数从 4.98M 扩大到 ~26.7M，而训练数据、优化器、clipping、
tokenizer、训练代码与因果响应协议尽量保持不变时：

1. 完整切线 C 是否仍预测真实有限差分响应；
2. C 是否仍系统性优于 optimizer-only B；
3. optimizer state → curvature → clipping 机制阶梯是否仍存在。

本轮是 mechanism scale generalization，不是 A0H 算法效率实验，不重做 A0A–A0G。

## 2. Scale-only 配置（以 v07 anchor 实际冻结代码为基础）

| 项 | anchor (v07) | G1a (26M) |
|---|---|---|
| 代码世代 | minimind model_minimind.py + trainer_utils.py | **同一份冻结代码，不改一行** |
| hidden_size | 256 | **512** |
| num_hidden_layers | 4 | **7** |
| num_attention_heads / kv | 8 / 4 (GQA) | 8 / 4（不变，head_dim=64 合法） |
| intermediate_size | 832 | 1664（同一默认公式 ceil(h·π/64)·64，非手挑） |
| vocab / tokenizer | 6400 / minimind model | 不变 |
| seq len | 128 | 不变 |
| position encoding / norm / activation | RoPE / RMSNorm / SiLU | 不变 |
| 参数量（unique） | 4,983,168 | **26,681,728**（5.352×；实测写入 manifest；预注册估值 26,682,048 在开结果前更正，hidden/layers/heads/intermediate 不变） |
| 数据集 D0 | e3c_subset.jsonl (4096 行) | 同一份文件（sha256 1e657af9…，只读） |
| partition | partition_v07.json | 同一份文件只读复用（probe/eval 排除规则不变） |
| microbatch / accum | 4 / 2 | 不变（global batch 8 seq = 1024 tokens/step） |
| optimizer | AdamW(betas .9/.999, eps 1e-8, wd 0.01) | 不变 |
| LR | 5e-4 cosine (get_lr) | 不变 |
| clip | 1.0 | 不变 |
| precision | bf16 autocast, GradScaler 禁用 | 不变 |

选 512×7 的理由：25–30M 内最接近 26M 的合法配置；heads=8 保持 head_dim=64；
intermediate 走冻结默认公式。模型大小在预注册阶段一次性选定，不根据结果回改。

## 3. 训练

- seeds（预注册，未冲突：v07 用 2030–2037，A0G 用 2038–2041）：**2042、2043、2044**。
- planned_total_optimizer_steps = 128（256 micro），与 anchor 相同 schedule 定义。
- 训练状态用归一化进度而非机械 step：**EARLY=26（20.3125%，26624 tokens）、
  MIDDLE=64（50%，65536 tokens）、LATE=102（79.6875%，104448 tokens）**。
- 训练在 LATE(step 102) 后终止（更晚无注册状态）；LR schedule 仍定义在
  total_micro=256 上，所有被回放的 future step 看到与 anchor 协议相同的 scheduled LR。
- train_batch_seed = train_seed + 1704（与 v07 同一派生规则）。
- 每个 checkpoint 保存：model 原始 dtype state_dict、AdamW m/v/step、scaler state、
  完整 RNG（python/numpy/torch cpu/cuda）、global optimizer step、metadata（config、
  partition sha256、optimizer signature、hashes）。

## 4. 继承的冻结因果协议（逐项）

- pulse：accumulation backward → unscale → ±ε·q → clip(1.0) → AdamW → zero_grad，
  **只在首个 optimizer step 施加一次**（pulse-before-clip）。
- CRN：baseline/plus/minus 从同一完整 Z_t 恢复，共用 batch manifest、RNG stream、
  LR path，唯一差异 ±ε·q；rho0-noop（sign=+1, ε=0）必须与 baseline 逐位一致。
- ε = 0.25·σ_q，σ_q = 16 条 E1 stream 上 s 的 std(ddof=1)（v07 冻结规则原样）。
- q：每态独立，全部 32 个 probe 索引单 batch 的 probe 梯度，全局 L2 归一化；
  probe/eval 自轨迹第 0 步永久排除（partition 只读复用）。
- future LR：`set_native_micro_lr` 沿同一 256-micro cosine 轴。
- HVP：math-SDPA reverse-over-reverse 代理（native flash 无 double backward）——
  与 A0C/A0E2 相同的 mixed-backend caveat 永远伴随。
- 官方 horizon 网格 **H = {1, 4, 8}**（不新增 2/12/16/32）。冻结 tangent runner
  内部会逐步走完 1..max(H)，其在 H=2 的预测是副产物，一律不进正式统计；
  FD 分支只以 {1,4,8} 运行。
- Model A 不是 G1a 主比较。

## 5. 正式状态矩阵与 stream

- 3 seeds × 3 stages = **9 训练状态**；每状态 **4 条预注册 future stream**，
  共 **36 条 baseline future stream**；state-horizon 正式单元 **9 × 3 = 27**。
- stream seed 派生：`two_level_seed(base, train_seed, state_step, stream_id)`，
  `batch_seed = derive_stream_seed(stream_seed, 404)`，E1 base=408202601，
  E2 base=409202601（G1a 专属 base，派生规则与 v07 相同）。
- **顺序规则**：全部 36 条 future-stream manifest 生成并写入
  `future_stream_manifest.json` 之后，才允许执行任何 E2 响应分支。

## 6. 主比较（9 states 全部）

chi_FD（E2 配对中心差分，同一冻结定义）、chi_B（optimizer-only tangent）、
chi_C（full tangent）。BH_fixed 不在全部 9 态计算。

## 7. 机制子实验

固定 seed 2042 的 EARLY/MIDDLE/LATE 三态：B、BH_fixed、C、FD，H={1,4,8}。
H=1 时 B=BH_fixed=C 须在数值容差内一致（构造性恒等检查）。
重点量（带符号报告，不预设逐单元为正）：
gain_curvature = err_B − err_BH_fixed；gain_clip = err_BH_fixed − err_C。

## 8. 数值/实现 gate（正式实验前）

对 seed 2042 / step 64 单状态 smoke：checkpoint 完整恢复；optimizer state
维度/名字正确；q 与参数严格同名同 shape；FD ± 分支 CRN 一致；H=1 analytic/FD
基本一致；B/C shadow baseline hash 正常；HVP 可执行；无意外 optimizer step；
全部输出 finite。实现 bug 可修复并重新 smoke（留修改记录）；科学定义问题
⇒ CONTRACT_INCOMPATIBLE，停止并报告。

## 9. 统计

每 seed × stage × H 输出 chi_FD / chi_B / chi_C 及 absolute / relative / signed
error。汇总：A 全部 27 行 pooled（R²、Pearson、MAE、mean+median relative error）；
B 按 H；C 按 seed；D 按 stage；E H>1 paired err_B − err_C；并报告 C beats B /
ties / C worse than B 计数（H=1 为构造性平局，不计入 paired 胜负）。
指标一律 import `fr.a0c_adamw_response_propagation_v01._metric` / `relative_error`。

## 10. 预注册判定（冻结，不得根据结果更改）

分析网格 = 27 行（H=1 行进入 pooled R²/MAE，但因 B≡C 构造恒等，不计入 paired 胜负）。

**STRONG_SCALE_REPLICATION** 当且仅当同时满足：
1. C pooled R²（27 行）≥ 0.90；
2. 18 个 H>1 行中 ≥80%（≥15/18）err_C < err_B（tie：|err_C−err_B| ≤ 1e-12）；
3. H=8（9 行）：paired mean err_C < mean err_B，且 C 的 H=8 R² ≥ 0.50 且 corr > 0；
4. ≥2/3 个 seed 在其 H>1 行（每 seed 6 行）上 mean(err_B − err_C) > 0。

**PARTIAL_SCALE_REPLICATION**：H>1 pooled mean relative error 满足 C < B，
但未满足 STRONG 全部条件（含 seed/stage 明显异质）。
**NO_SCALE_REPLICATION**：C 未总体优于 B，或 C 对 FD 失去有效预测能力。

不要求复制 anchor 的 168/168 全赢。

## 11. 训练动力学环境（仅解释，不得事后筛状态）

每步记录 training loss、LR、preclip grad norm、clip active、param norm；
future 每步记录 clip active/coefficient、hvp norm、native grad norm。
与 5M anchor（v07 theta mapping 的同口径量）并列报告。

## 12. 资源规则

允许使用当前可用 GPU，记录共享情况，不杀外部任务。允许（须记录）：
不改变数学定义的 gradient checkpointing；保持 global batch/tokens 的
microbatch/accumulation 调整；已有内存安全 HVP。禁止：OOM 后偷偷缩小模型、
改 seq len / global batch / clip / optimizer / 精度语义 / horizon、删除失败状态。
若 26M full tangent 在当前硬件不可实现 ⇒ 输出 RESOURCE_LIMIT + 完整诊断，
不得降规模冒充完成。

## 13. 不做的事

A0G task utility、A0H/A″、adjoint、新 decision controller、v06、第二数据集、
no-clip、64M/100M/1B、optimizer family change、beta/epsilon sweep、scalar FDR
重验证、修改任何旧论文结果。本轮只回答：26M 下原机制是否复现。

## 14. 输出

`outputs/generalization_g1a_26m_v01/`：G1A_PROTOCOL.md、experiment_config.json、
training_manifest.json、checkpoint_manifest.json、future_stream_manifest.json、
per_stream.csv、per_state_horizon.csv、mechanism_decomposition.csv、summary.json、
G1A_26M_SCALE_GENERALIZATION_REPORT.md、COMMANDS.log、ENVIRONMENT.json、
SOURCE_AND_INPUT_HASHES.txt、SHA256SUMS.txt（审计包不含大 checkpoint）。
