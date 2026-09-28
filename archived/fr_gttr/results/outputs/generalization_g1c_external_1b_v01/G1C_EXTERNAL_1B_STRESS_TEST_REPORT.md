# G1C External ~1B Stress Test Report — FR_G1C_EXTERNAL_1B_STRESS_TEST_V01

Created: 2026-09-16T16:05:00Z. **Status: EXTERNAL_CHECKPOINT_UNAVAILABLE ——
实验在 B2 公开制品核验阶段被阻塞，未执行任何科学计算。** 本报告登记
核验过程、结论与边界；不存在任何响应数值结果。

## 1. 任务与定位

按冻结协议（G1C_PROTOCOL.md §1）：对约 1B 外部模型（architecture /
tokenizer / corpus / optimizer 超参 / schedule 同时改变）做有限时间状态
空间响应结构（FD/B/C）的外部压力测试；唯一允许的主结论是机制结构是否
在一个公开的、具有真实 optimizer state 的 ~1B 外部训练系统上继续成立。

## 2. 执行摘要（时间线）

1. **B2 核验**：HF API 逐 revision 检查官方 AllenAI OLMo ~1B 全系列
   （7 个 repo、14 个 revision/branch 抽样、共 900+ branch 的全量枚举），
   并 probe 历史官方 checkpoint 域名 olmo-checkpoints.org（404）。
2. **结论**：全系列公开 checkpoint **均不含 optimizer state / train
   state**（无任何 optim*/train*/scheduler*/rng* 文件）。OLMo-1B 在
   原则上满足 B1 的 5 条候选理由（~1B、autoregressive Transformer、
   AdamW betas(0.9,0.95) wd0.1、clip=1.0、336 个公开 step checkpoint），
   但不满足决定性要求（unsharded checkpoint 含 model+optimizer+train
   state）。OLMoE-1B-7B 为 MoE 被架构排除。
3. **按 B2 规则**：不得补造 optimizer state；同官方系列内无可切换对象；
   结论在任何科学响应结果查看之前得出 →
   **EXTERNAL_CHECKPOINT_UNAVAILABLE，停止科学实验**。
4. 后续阶段（B3 下载 / B6 GPU gate / B10 矩阵 / B11 机制子集 / B14
   判定）**全部未触发**；smoke_report 状态 NOT_RUN_BLOCKED。

## 3. 核验证据（详见 EXTERNAL_MODEL_PROVENANCE.md §3 与
external_checkpoint_manifest.json）

- allenai/OLMo-1B：main + 9 个跨全程抽样 revision（step20000 →
  step738020），每 revision 仅 model.safetensors + pytorch_model.bin +
  tokenizer/config；
- allenai/OLMo-1B-0724-954000steps-unsharded：仅 config.yaml +
  model.safetensors（"unsharded" 指单文件权重，无 optim.pt/train.pt）；
- allenai/OLMo-2-0425-1B：268 branch 抽样 4 个，model-only；
- allenai/OLMoE-1B-7B-0924：MoE，排除；
- allenai/OLMo-7B：抽样 1 个，model-only（且 7B 超出资源范围）；
- olmo-checkpoints.org：HTTP 404。

## 4. 判定

**EXTERNAL_CHECKPOINT_UNAVAILABLE**（B14 冻结标签之一）。声明：此标签
描述公开科研制品的可用性，**不是** FD/B/C 机制的任何科学负结果；
本实验零训练、零重放、零响应计算。

## 5. 对第一篇的意义与边界

- 受控 MiniMind scale ladder（5M→26M→100M，G1a/G1b）、clipping 干预
  （G3）、数据分布泛化（G2）三项结论**不受影响**，照常冻结。
- 外部 ~1B 轴**未覆盖**，原因：官方 AllenAI OLMo 系列不发布 optimizer
  state。不得用 model-only checkpoint 重初始化 optimizer state 顶替
  （B1 红线）；不得因此回改任何内部实验结论。
- 任何未来外部压力测试（更换外部家族，须用户新授权与新预注册）应
  把"公开 checkpoint 是否含真实 optimizer state"作为第一门禁。

## 6. 资源与硬件事实

实测 2× RTX 5090 32GB（GPU0 共享余量 ~15.8GB；GPU1 外部不可动），
与任务书"A100-40GB"假设不符，已登记（RESOURCE_FEASIBILITY.md §1；
ENVIRONMENT.json）。阻塞发生在 GPU 使用之前，本轮零 GPU 运行。

## 7. 制品清单

本目录全部文件（G1C_PROTOCOL / EXTERNAL_MODEL_PROVENANCE /
RESOURCE_FEASIBILITY / external_checkpoint_manifest /
future_stream_manifest / experiment_config / smoke_report /
results/{summary.json, per_stream.csv, per_checkpoint_horizon.csv,
mechanism_subset.csv, external_vs_internal_summary.csv} / 本报告 /
CHANGE_LOG / COMMANDS.log / ENVIRONMENT.json / SOURCE_AND_INPUT_HASHES /
SHA256SUMS / screening_evidence/）；审计包
`g1c_external_1b_audit_package.tar.gz`（SHA256 见 AUDIT 文件）。
