# 06 — MiniMind 审计

对象：`/home/b205/projects/fluctuation_response/minimind/`（1.2G）。

## 1. 版本与来源

- 上游：`jingyaogong/minimind`（GitHub/HF/ModelScope），README 为 **minimind-3 时代**（最新更新日志 2026-04-01；minimind-3 64M / minimind-3-moe 198M-A64M），架构对齐 Qwen3/Qwen3-MoE。
- **非 Git 仓库**（无 .git）；主仓库 `.gitignore` 含 `minimind/` 条目，完全 untracked/ignored。来源完整性仅靠 zip 级 SHA256：
  `MINIMIND_SOURCE_SHA256.txt` = `c06cf0b2a69b560478e851547897e342f9bcd06b705e2ec9d30b6cc0618282ac  /mnt/d/b205_transfer/minimind.zip`（WSL 挂载盘传输）。
- README 内无 commit hash → 精确上游 revision UNKNOWN；如需逐文件追溯建议补文件级 manifest。

## 2. 本地修改：**无**

- 对全部 .py 检索 `fluctuation|fr_|perturb`：**零命中**（仅语料/README 中的自然语言词汇）。
- `response` 命中均为上游工具调用/RL 语义（`<tool_response>`）。
- 结论：这是**干净的上游副本**；FR 研究从未魔改 MiniMind 本体（FR 侧通过 `research/adapters/minimind_adapter_v02.py` 与冻结 runtime 模块接入）。`lm_dataset.py` 仅有一段上游自带的注释掉的调试打印。

## 3. 结构

```
minimind/
├── model/model_minimind.py (288 行)  # Dense + MoE；model_lora.py (rank16)
│   ├── tokenizer.json / tokenizer_config.json (vocab 6400)
├── trainer/  train_pretrain.py, train_full_sft.py, train_lora.py, train_distillation.py,
│             train_dpo.py, train_ppo.py, train_grpo.py, train_agent.py(+rollout_engine),
│             train_tokenizer.py, trainer_utils.py
├── dataset/ lm_dataset.py + pretrain_t2t_mini.jsonl (1.2G, 1,270,238 行)
├── scripts/ chat_api.py, serve_openai_api.py, web_demo.py, convert_model.py, eval_toolcall.py
├── eval_llm.py；requirements.txt；images/ 12M
```

## 4. 模型（MiniMindConfig 默认）

hidden 768 / layers 8 / heads 8 / kv-heads 4（GQA，head_dim 96）/ vocab 6400 / intermediate ⌈768π/64⌉·64=2432 /
max_position 32768（RoPE 预计算 buffer，θ=1e6）/ dropout 0 / rms_eps 1e-6 / SiLU / SwiGLU / **tied embeddings** /
QK-RMSNorm / SDPA flash / MoE（use_moe=False 时忽略：4 expert top-1 aux 5e-4）/ YaRN 外推开关。
注意：FR 实验用的 anchor 是 hidden256×4L（4.98M）与 G1a/G1b 尺寸——通过 config 覆盖，非默认。

## 5. 训练管线（train_pretrain.py 实测审计）

| 项 | 事实 |
|---|---|
| 优化器 | `torch.optim.AdamW(model.parameters(), lr=args.learning_rate)` —— **未显式设 β/wd ⇒ PyTorch 默认 β=(0.9,0.999) eps=1e-8 wd=1e-2**（FR 项目标注 [FR-NATIVE-ADAMW] 的"原生"含义） |
| LR schedule | 手写余弦：`lr·(0.1+0.45·(1+cos(π·step/total)))`，起 ≈1.0×、末 ≈0.1×；逐步写入 param_groups。（GTTR PhaseA 后来另冻结 `finite_cosine_then_hold_5e-5_v01` 尾部语义——这是 FR 侧声明，不是 minimind 改动） |
| grad clip | `clip_grad_norm_(params, 1.0)`，位于 `scaler.unscale_` 之后（与 FR pulse 协议插入点一致） |
| AMP | `--dtype bfloat16` 默认 → `autocast(bf16)`；`GradScaler(enabled=(dtype=='float16'))` ⇒ **bf16 下 scaler 禁用直通** |
| 梯度累积 | `accumulation_steps=8` 默认（FR 实验用 2）；epoch 尾余数补一次更新 |
| batch/seq | batch 32 默认；`--max_seq_len` **默认 340**（SFT 默认 768）→ PretrainDataset 定长截断（max_length−2 + bos/eos，pad→−100） |
| seed | `setup_seed(42+rank)`；每 epoch `setup_seed(42+epoch)` 后 randperm 重排 |
| checkpoint | 推理权重 `out/pretrain_<hidden>.pth`（**仅 state_dict，half().cpu()**）；续训 `checkpoints/*_resume.pth` = **model + optimizer.state_dict() + scaler + epoch + step + world_size + wandb_id**，原子写入（.tmp+replace） |
| resume | `SkipBatchSampler` 支持跳批续训；DDP 时 DistributedSampler |
| DDP | 支持（DDP wrapper、rank/world_size 记录） |

## 6. 对"20M/50M/100M/300M 级 controlled experiment"的适配性评估

- **适合**。理由：(1) 无本地魔改，行为=上游；(2) config 全可覆盖（G1a/G1b 已示范 26.68M/100.78M scale-only：只改 hidden/layers/heads，其余逐字段一致）；(3) resume 完整保存 optimizer 态——满足 FR 系"真实 optimizer state"硬要求；(4) clip+unscale 顺序与 bf16 语义与 FR 协议兼容；(5) 4 个冻结 config 模板现成（research/configs/g1*_v01.json）。
- **注意事项**：(a) 唯一 seq-length 入口是 `--max_seq_len`（Pretrain 默认 340 vs SFT 768，勿混）；(b) 推理权重 half() 落盘——需要 fp 态时必须用 resume 路径；(c) 每 epoch 重排 seed=42+epoch——流复现须用 FR 侧 batch manifest，勿依赖上游 loader 顺序；(d) vocab 6400 小词表，300M 级是否换 tokenizer 属新变量（须单变量原则）；(e) MoE 分支未测试。

## 7. 应保持原样 vs wrapper 接入点

- **保持原样**：`model/model_minimind.py`、`dataset/lm_dataset.py`、`trainer/*.py` 全部（上游语义=实验对照的一部分；任何改动都会破坏与 G1a/G1b/v07 的可比性）。
- **wrapper/adapter 接入**（推荐，且是本项目既有做法）：
  - `research/adapters/minimind_adapter_v02.py`（模型构建/加载适配）；
  - `research/src/fr/minimind_v03_runtime.py` + `stage_c_runtime.py`（replay 运行时、AMP/clip/脉冲插入）；
  - 新实验=新 runner + 新 config + 新输出目录（§13b 冻结规则），**绝不编辑 minimind/ 内文件**。

## 8. 磁盘

- minimind/ 总 1.2G，其中 `dataset/pretrain_t2t_mini.jsonl` 1.2G（1,270,238 行）。若新项目不用该语料可归档，但 G2/D1 类跨语料对照会用到 pretrain_hq（在 outputs/g2 下，1.6G）。

## 9. requirements.txt（关键）

datasets 3.6.0 / transformers 4.57.6 / trl 0.13.0 / numpy 1.26.4 / sentencepiece 0.2.0 / einops 0.8.1 / wandb 0.22.3 / swanlab 0.9.8 / streamlit 1.50.0 / modelscope 1.37.0；注释掉 torch 2.6.0、peft、matplotlib。实际运行用 conda env `fr_minimind`（torch 2.11.0），非此文件锁定版本——复现以 `baseline_manifest/` 为准。
