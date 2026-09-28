# G1C Protocol — FR_G1C_EXTERNAL_1B_STRESS_TEST_V01

> 预注册冻结：2026-09-16T16:05:00Z。本文件在任何科学响应结果之前写成；
> 实验在执行 B2 公开制品核验时即被阻塞（EXTERNAL_CHECKPOINT_UNAVAILABLE），
> 本文件包含两部份：§1-§17 为冻结的预注册协议；§18 为阻塞结果记录
> （后写，明确标注，不改动 §1-§17 的任何规则）。
> 指标一律 import 冻结的 `fr.a0c_adamw_response_propagation_v01` 的
> `_metric`/`relative_error`，禁止重定义；冻结制品一律只读。

## 1. 科学定位（B0）

本实验**不是** pure scale test，也**不是** MiniMind scale-only replication。
外部 1B 模型同时改变 architecture / tokenizer / corpus / optimizer
hyperparameters / training schedule，是**多变量外部压力测试**。

唯一允许的主结论：

> **核心有限时间状态空间响应结构（FD/B/C 机制阶梯）能否在一个公开的、
> 约 1B 规模、具有真实 optimizer state 的外部 LLM 训练系统上继续成立？**

禁止表述为：纯 scale 因果效应、scaling law、MiniMind→1B 单因素外推。

## 2. Primary external model（B1）

- Primary 候选固定：**AllenAI OLMo-1B**，标准 OLMo training checkpoint 格式；
  优先官方 OLMo repository / official checkpoint index。
- 选择理由（登记）：① 约 1B 参数（1.24B）；② autoregressive Transformer；
  ③ AdamW（官方 native config：lr=4e-4、betas=(0.9,0.95)、wd=0.1）；
  ④ global gradient clipping max_grad_norm=1.0（2.1B-token warmup ×5）；
  ⑤ 公开 intermediate checkpoints（336 个 step revision，每 1000 步）；
  ⑥ 要求 unsharded checkpoint 含 model + optimizer state + train state。
- **禁止**仅使用 HF inference-only checkpoint 并重初始化 optimizer state；
  必须读取真实 `optim.safetensors` / equivalent 与 `train.pt` / equivalent。

## 3. 公开制品与许可证核验（B2）

建立本目录后先生成 `EXTERNAL_MODEL_PROVENANCE.md`、
`external_checkpoint_manifest.json`、`G1C_PROTOCOL.md`（本文件）、
`RESOURCE_FEASIBILITY.md`，记录：repository、revision、license、架构、
参数量、训练数据、tokenizer、optimizer/betas/eps/wd、LR schedule、
grad clipping、precision、checkpoint 格式、URL、下载后 hashes。

- 若官方 OLMo-1B 不再提供 optimizer state：**不得自己补造**；检查同一
  AllenAI OLMo 官方系列中约 1B 且公开完整 optimizer/trainer state 的
  checkpoint；**只能在查看任何科学响应结果之前切换**并记录原因；
  **不得根据响应效果切换模型**。
- 若没有合格公开 checkpoint → **EXTERNAL_CHECKPOINT_UNAVAILABLE**，
  停止科学实验。（本实验实际走此分支，见 §18。）

## 4. 下载策略（B3）

不一次下载全部 checkpoint。先选一个候选 EARLY checkpoint 做 feasibility，
优先官方 unsharded 格式；下载 config / model / optimizer / training state，
验证 hashes、optimizer m/v completeness、parameter-name mapping、
global step、LR、scheduler metadata、data/training metadata。
只有 feasibility 通过后才下载其它 stage。

## 5. External optimizer semantics（B4）

- 不把 MiniMind AdamW 超参强行套到 OLMo；使用 checkpoint 自己的真实
  betas/eps/wd/lr/clip/optimizer state。
- 冻结 a0c 的通用 AdamW Jacobian / B / C 数学对 (β1,β2,ε,wd) 是参数化的
  → 数学上支持外部超参，无需修改冻结实现。
- 若实现层有 MiniMind-specific assumption：新增**版本化、经测试**的
  external adapter，**不得修改**冻结 MiniMind 实现。

## 6. 数据与 future stream（B5）

优先使用与所选 OLMo checkpoint 原始训练分布一致的官方公开 Dolma /
OLMo training data；目标是**固定、可复现、与 training stack 一致的
future minibatch stream**，不是重建整个训练集。必须记录 data source /
revision/hash / tokenizer / sequence length / packing / loss masking /
sample identities。若原始未来数据顺序无法从 trainer state 恢复，允许
使用预注册固定 Dolma future-stream pool，但必须明确称为
"**conditional response under frozen in-distribution future streams**"，
不得称为 "original training trajectory replay"。若无法合法构造分布内
future stream → **DATA_STREAM_UNAVAILABLE**，停止科学测试。

## 7. 资源 feasibility gate（B6）

假定主硬件单卡 A100-40GB。先用 1 checkpoint × 1 future stream × H=1 依次
验证：model+optimizer restore → baseline fwd/bwd → gradient injection →
H1 B/Jacobian → central FD → HVP/second-order path → full C → peak GPU
memory → all finite；然后才尝试 H=4 单 stream。**不要先跑正式矩阵。**

## 8. 内存策略（B7）

允许（不改数学定义）：activation checkpointing、sequential HVP、
layer-wise recomputation、optimizer/inactive tensor CPU offload、安全时
立即释放 graph、deterministic microbatch replay、本项目已有 memory-safe
HVP。允许"一次只加载一个 checkpoint/stream"。
禁止：quantize model/optimizer states、low-rank/sparse HVP、truncate
layers、randomly subsample parameters、LoRA、freeze layers、用近似替换
真 C（这些属于后续算法论文）。

## 9. Sequence length 规则（B8）

不得为省显存把外部模型原生 context（OLMo=2048）偷偷改到 128 仍称原训练
条件。先测原生 sequence length；若原生 HVP 在 40GB 不可执行，允许在打开
正式科学结果**之前**登记 **SHORT-CONTEXT STRESS SUBPROTOCOL**（如
seq=128/256），但 tokenizer/loss/objective/future corpus 不变，明确标注
局部短上下文 stress test，不得称原生 OLMo 训练动力学完整复现。
优先级：native context > registered short-context stress > RESOURCE_LIMIT。

## 10. Checkpoint stages（B9）

feasibility 通过后选最多 3 个公开 checkpoint（EARLY/MIDDLE/LATE），依据
tokens seen / 归一化官方训练进度（约 20/50/80%），**预先冻结**，不得根据
response 结果选 checkpoint。若完整 optimizer checkpoint 步距稀疏，选最接近
20/50/80% 的合法 checkpoint。若下载/存储成本过大，允许在任何 response
结果前预注册 EARLY+LATE 两 stage 为 minimum external stress set。

## 11. 正式实验矩阵（B10）

稀疏 external stress test，不复制 G1a/G1b 完整网格。首选：3 checkpoints
× 2 frozen future streams × H={1,4,8}；模型只跑 FD / B / C（不跑 Model A）；
共 9 个 checkpoint-horizon cells，每 cell 两条 stream；BH_fixed 只做
mechanism subset。若 H=8 超出可执行资源，在打开正式 H=4 结果**之前**
预注册 H={1,4} 为正式 minimum matrix，H8=RESOURCE_NOT_COVERED。

## 12. Mechanism subset（B11）

一个预注册 checkpoint（优先 MIDDLE）× 2 streams × {B, BH_fixed, C, FD} ×
H={1,4}（H=8 若正式可行则加）。目标只检查 optimizer-only → curvature →
clipping-aware C 机制阶梯是否仍有信息；不得要求每单元严格 B > BH > C。

## 13. 数值 gate（B12）

优先通过 H1：analytic/Jacobian vs central FD 的 relative/absolute error，
至少两个 epsilon level，确认不是单一有限差分尺度的偶然。外部
architecture/backend 改变，不假定 MiniMind mixed-backend caveat 仍相同：
必须重新记录 attention implementation、double-backward support、HVP
backend、native-vs-surrogate 差异；若用不同 backend HVP，重新建立对应
caveat。

## 14. 主要指标（B13）

每 checkpoint/H：chi_FD、chi_B、chi_C + absolute/relative/signed error；
汇总 C R² / Pearson / MAE / mean+median relative error 及 B 对应指标；
H>1：err_B − err_C，wins/ties/losses。样本小，不得把 pooled row 数冒充
独立训练 run。

## 15. 判定标签（B14，冻结）

- **EXTERNAL_STRESS_SUPPORTED**：① H1 数值 gate 通过；② C 在全部已覆盖
  H>1 cell 总体保持有效预测，pooled R² ≥ 0.80；③ mean(err_B−err_C) > 0；
  ④ 至少 2/3 checkpoint（若只有 2 个则 2/2）平均 C error < B error；
  ⑤ 不存在 C 随 H 明显完全崩塌。
- **EXTERNAL_STRESS_MIXED**：C 仍有明显预测能力，但优势或跨 stage/horizon
  不稳定。
- **EXTERNAL_STRESS_NOT_SUPPORTED**：C 失去有效预测能力或总体不优于 B。
- **RESOURCE_LIMIT**：合法 full tangent 在当前资源下无法执行。
- **EXTERNAL_CHECKPOINT_UNAVAILABLE / DATA_STREAM_UNAVAILABLE /
  CONTRACT_INCOMPATIBLE**：按实际情况。

这只是 stress-test 标签，不是统计显著性结论。

## 16. 呈现与统计边界（B15/B16）

- 不得把 5M/26M/100M/OLMo-1B 画成一条纯 scaling curve 或拟合
  error vs parameter count 定律；controlled ladder 与 external stress
  必须分隔呈现。
- OLMo 可能只有一条公开训练 trajectory：不得 bootstrap stream 数伪装
  独立 seed；只报告 checkpoint-level / stream-level / H-level 描述性结果，
  最多给同 cell B-vs-C paired stream 差的 descriptive interval；不写
  cross-seed generalization / statistical confirmation。

## 17. 红线（继承）

不得修改或覆盖 A0A–A0G、A0H/A″、5M anchor、G1a、G1b、G2、G3 的任何冻结
CSV/JSON/checkpoint/q/verdict；不得重训练或微调 1B 制造 checkpoint；不得因
结果不好切换模型或 checkpoint；所有 G1c 产物进入本版本化目录；本轮完成后
不得自动启动第二个外部 1B 模型 / 7B / D2 / A0G-lite / 新 optimizer /
算法实验。

## 18. 执行结果记录（后写，2026-09-16；阻塞，无科学结果）

B2 制品核验结论：**EXTERNAL_CHECKPOINT_UNAVAILABLE**。核验范围与证据见
`EXTERNAL_MODEL_PROVENANCE.md` 与 `external_checkpoint_manifest.json`：
官方 AllenAI OLMo ~1B 系列全部公开 checkpoint（OLMo-1B 336 个 step
revision 抽样 9 个 + main、OLMo-1B-hf、OLMo-1B-0724-hf、
OLMo-1B-0724-954000steps-unsharded、OLMo-2-0425-1B 268 个 branch 抽样 4
个、OLMoE-1B-7B-0924（MoE，架构排除）、OLMo-7B 抽样 1 个）**均不含任何
optimizer state / train state 文件**；历史官方 checkpoint 域名
olmo-checkpoints.org 已 404。未查看任何科学响应结果即得出结论；未执行
任何下载（大文件）/GPU 运行。此标签**不是**机制的科学负结果，只是公开
制品不可用声明。详细时间线与命令见 CHANGE_LOG.md / COMMANDS.log。
