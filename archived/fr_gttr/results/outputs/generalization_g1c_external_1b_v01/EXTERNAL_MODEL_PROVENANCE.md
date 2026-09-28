# EXTERNAL MODEL PROVENANCE — FR_G1C_EXTERNAL_1B_STRESS_TEST_V01

Created: 2026-09-16T16:05:00Z (before any scientific result; the experiment was
blocked at this stage). Purpose: B1/B2 public-artifact and license verification
for the primary external ~1B model.

## 1. Primary candidate and rationale (B1)

- **Model**: AllenAI **OLMo-1B**（官方系列；HF `allenai/OLMo-1B`）。
- **License**: **Apache-2.0**（HF repo metadata `license: apache-2.0`，
  见 screening_evidence/OLMo-1B_HF_README.md）。
- **选择理由登记**：
  1. approximately 1B parameters（1.24B；d_model=2048、n_layers=16、
     n_heads=16、vocab=50280、RoPE、SwiGLU、weight tying）；
  2. autoregressive Transformer；
  3. AdamW（官方 native config：name=adamw、lr=4e-4、betas=(0.9,0.95)、
     weight_decay=0.1）；
  4. global gradient clipping **max_grad_norm=1.0**（前 2,097,152,000
     tokens warmup，factor 5.0）；
  5. 公开 intermediate checkpoints：336 个 step revision
     （step20000-tokens84B … step738020-tokens3095B，每 1000 步）；
  6. 训练数据（Dolma / olmo-mix v1.6）与 tokenizer（gpt-neox-olmo-dolma
     v1.5，vocab 50280）公开。
- **决定性要求（B1 末条）**：unsharded checkpoint 必须含 model state +
  optimizer state + train state（optim.safetensors / equivalent +
  train.pt / equivalent）。**核验结果：不满足**（见 §3）。

## 2. 训练栈事实（来自官方 native config，登记供 B4/B5）

来源：`allenai/OLMo-1B-0724-954000steps-unsharded` 的 `config.yaml`
（SHA256 `2b3d838854d18f6e59c494a8ddef91e0efa2eb2b8de5f3af6b65526559b0131b`，
存 screening_evidence/OLMo-1B-0724_unsharded_config.yaml）：

| 字段 | 值 |
|---|---|
| optimizer | adamw, lr=4e-4, weight_decay=0.1, betas=[0.9, 0.95] |
| scheduler | cosine_with_warmup, t_warmup=10,485,760,000 tokens, t_max=3e12, alpha_f=0.1 |
| grad clip | max_grad_norm=1.0（2.1B-token warmup ×5.0） |
| precision | amp_bf16 |
| attention | flash_attention=true（0724 run；2024-02 原始 run 为 false） |
| batch | global_train_batch_size=512, device_train_microbatch_size=4 |
| save_interval | 1000（steps） |
| checkpointer | sharded_checkpointer=olmo_core, reset_optimizer_state=false |
| data | s3://ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/...（Dolma v1.6 mix） |
| tokenizer | GPT-NeoX BPE, vocab 50280（repo allenai/gpt-neox-olmo-dolma-v1_5） |

注：`reset_optimizer_state=false` 证明原训练确保存了 optimizer state，
但 AllenAI **未随公开 checkpoint 发布**这些 state 文件。

## 3. Optimizer/trainer state 核验结果（B2 决定性）

HF API 逐 revision 文件列表核验（2026-09-16，命令见 COMMANDS.log）：

| repo | 规模 | 核验范围 | state 文件（optim*/train*/scheduler*/rng*） |
|---|---|---|---|
| allenai/OLMo-1B | 1.24B | main + 9 个跨全程抽样 revision（step20000/50000/114000/380000/442000/502000/560000/621000/738020），全系列共 336 个 branch | **0**（每 revision 仅 model.safetensors、pytorch_model.bin、tokenizer/config） |
| allenai/OLMo-1B-hf | 1.24B | repo 级 | **0**（HF 格式定义上不含 optimizer state） |
| allenai/OLMo-1B-0724-hf | 1.24B | repo 级 | **0**（同上） |
| allenai/OLMo-1B-0724-954000steps-unsharded | 1.24B | main | **0**——仅 config.yaml + model.safetensors；"unsharded" 指单文件模型权重，不含 optim.pt/train.pt |
| allenai/OLMo-2-0425-1B | ~1.24B | main + 3 个抽样 branch（stage1-step0 / stage1-step1300000 / stage2-ingredient3-step23852），全系列 268 branch | **0** |
| allenai/OLMoE-1B-7B-0924 | MoE 1B-active/7B-total | repo 级 | **0**；且架构为 MoE，非 dense ~1B autoregressive Transformer，**架构排除** |
| allenai/OLMo-7B | 7B | 1 个抽样 revision（step279000，全系列 559） | **0**；且规模超出本实验资源范围 |

历史官方 checkpoint 域名 `https://olmo-checkpoints.org/`：HTTP 404（已下线）。
官方 README（v0.2.4 tag 与当前 main）均只描述模型权重 checkpoint 的发布
（"We are releasing many checkpoints for these models, for every 1000
training steps"），从未提供 optimizer state 下载渠道。

**结论：官方 AllenAI OLMo ~1B 系列不存在任何公开含真实 optimizer state /
trainer state 的 checkpoint。** 按 B2：不得自己补造；同系列内无可切换
对象；在任何科学响应结果查看之前得出结论 →
**EXTERNAL_CHECKPOINT_UNAVAILABLE**，停止科学实验。

## 4. 下载与哈希记录（B2/B3）

- 大文件（模型/optimizer 权重）：**零下载**（核验阶段即阻塞，B3 的
  feasibility 下载未触发）。
- 小文件（制品核验证据，存 screening_evidence/）SHA256：
  - OLMo-1B_main_config.json：`89767baeb3784222457a8e18e7821640bd45021cb7d4b3b1c0d7de5a7ab59c9d`
  - OLMo-1B-0724_unsharded_config.yaml：`2b3d838854d18f6e59c494a8ddef91e0efa2eb2b8de5f3af6b65526559b0131b`
  - OLMo-2-0425-1B_main_config.json：`3dc581384d61b3366fc34a95d12c20e595014ea62352469c17c3e3a77f32882a`
  - OLMo-1B_HF_README.md：`b1e5c82e6b36de5dd7756017308eacba7483c1647aa23aeb041007fa7a53390f`

## 5. 阻塞声明

`EXTERNAL_CHECKPOINT_UNAVAILABLE` 是**制品可用性标签**，不是对 FD/B/C
响应机制的任何科学负结果；本实验未训练、未重放、未计算任何响应数值。
