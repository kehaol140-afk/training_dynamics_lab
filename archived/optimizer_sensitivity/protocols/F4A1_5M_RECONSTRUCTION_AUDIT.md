# F4A.1 — 5M v07 训练 Recipe 重构审计（只读）

- 日期：2026-09-30。来源：`legacy_fluctuation_response/research/scripts/e3c_prepare_state_bank_v07.py`（275 行，只读）+ `research/src/fr/` 运行库 + bank metadata（`outputs/e3_confirmatory/v07/seed_*/states/state_step_*.pt` 的 metadata.config）+ `multiseed_state_bank_summary.json`。

## 恢复的完整 recipe（逐项，全部有代码/metadata 出处）

| 项 | 值 | 出处 |
|---|---|---|
| architecture | MiniMindConfig(hidden_size=256, num_hidden_layers=4, max_seq_len=128, use_moe=False)；heads/kv 走 MiniMindConfig 默认（8/4） | 脚本 L123-128 + metadata.config；当前环境重建 state_dict 形状与 bank 47 张量**逐项一致**（F4-A inventory 已验证） |
| dataset | `PretrainDataset(e3c_subset.jsonl, tokenizer, max_length=128)`（4096 行，SHA 见 legacy DATASET_SHA256） | 脚本 L136 + config.subset_lines=4096 |
| tokenizer | `minimind_base/model` | 脚本 L129-133 |
| 初始化/RNG | `setup_seed(train_seed)` 后 `init_model(from_weight="none")` | 脚本 L121, L91/L129；`setup_seed` = minimind_base/trainer/trainer_utils.py:54 |
| data order | `batches_from_train_pool(partition.train_indices, batch_size=4, seed=train_seed+1704, needed_micro)`：`make_permutation(len(pool), seed+1704)` → `batchify_indices(ordered, 4)`，不触 probe/eval | 脚本 L152-158 + fr/stage_c_runtime.py:8 |
| batch/accum | microbatch 4 × accum 2（1024 tokens/opt-step） | config |
| optimizer | `torch.optim.AdamW(params, lr=5e-4)`（其余默认：betas .9/.999、eps 1e-8、wd 0.01）+ `assert_native_adamw` | 脚本 L138-139 |
| LR schedule | 每 micro 前写入 `get_lr(micro_idx, total_micro=256, 5e-4)`（MiniMind 原生 cosine） | fr/native_schedule.py:18 |
| precision | bf16 autocast（`make_amp`，GradScaler enabled=False for bf16） | fr/minimind_v03_runtime.py:22 |
| clip | 每 opt step：`unscale_ → clip_grad_norm_(params, 1.0) → scaler.step → update → zero_grad` | 脚本 L186-191 |
| 损失 | `(res.loss + res.aux_loss) / accum` | 脚本 L182 |
| total planned | 128 opt steps（256 micro）；states 保存于 opt_step ∈ {46,52,58,64,70,76,82} | config + metadata |
| seeds | confirmation 2030–2037（train_batch_seed = seed+1704） | config + 脚本 L152 |
| state 保存 | `save_full_training_state(path, model, optimizer, scaler, metadata)` | fr/full_state.py |
| partition | pilot partition（train/probe/eval 划分，SHA `5ecaf0a3…93f4f`——与 F2/F3/F4-A pipeline 所用同一划分） | v07 summary + F3 PARTITION |

## 可重构性判定

**RECIPE_RECONSTRUCTABLE**：全部要素（架构/初始化 RNG/数据顺序/优化器/调度/精度/clip/保存格式）都有唯一确定的代码出处，且所需模块（fr 包、trainer_utils、model_minimind）在当前环境可直接只读 import 并运行（MiniMind 代码与 F2/F3/F4-A 共用同一 `minimind_base`）。

## 已知限制（诚实声明）

- 原 v07 训练环境为 conda env `fr_minimind`（config.environment_conda_env）；当前环境 `minimind-exp`（torch 2.11.0+cu128）。**跨环境不假定 bitwise 轨迹复现**——canary（B3）采用分布等价判据（当前环境对同一 recipe 的重训样本须落在原 8 seed 家族在最近 matched step 的经验包络内），判据在 prereg 冻结。
- `setup_seed` 的实现细节（torch/cuda/numpy/python random 全置 seed）在 minimind_base 只读可查；当前环境执行同一函数。
- 新 states 写入 `outputs/f4a1_matched_coverage/new_states/`（项目 outputs），**不写入、不污染 legacy bank 目录**。
