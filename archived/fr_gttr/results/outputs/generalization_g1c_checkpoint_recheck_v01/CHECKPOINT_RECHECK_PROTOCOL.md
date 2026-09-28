# CHECKPOINT_RECHECK_PROTOCOL — FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01

> 预注册冻结：2026-09-16T16:49:26Z（任何 HTTP 探测之前）。
> 目的：对上一轮 G1c 的 `EXTERNAL_CHECKPOINT_UNAVAILABLE` 判定做
> **官方 standard/unsharded checkpoint 直链文件级复核**。
> 本轮**无 GPU、无科学 response、无 HVP/FD/B/C、无完整大文件下载**
> （每个大文件读取上限 1 MB；config.yaml / train.pt 小文件允许完整读取）。

## 1. 核验逻辑修正（本轮第一条规则）

读取 OLMo 官方 `docs/Checkpoints.md`（快照 `OLMo_official_Checkpoints.md`，
SHA256 `f84cf11d…`）并记录其正式定义：

- OLMo **standard（unsharded）checkpoint** 目录包含四件：
  `config.yaml`、`model.safetensors`、`optim.safetensors`、`train.pt`；
  **旧 release 使用 `.pt` 扩展名**（`model.pt` / `optim.pt`）。
- 官方明确声明：CSV 中列出的 **directory URL 本身不能直接访问**，
  但 **directory 内的具体文件是公开访问对象**。

因此本轮规则：

1. "**目录 URL 404**" 不得等价为 "checkpoint 不存在"或
   "training state 未保存"。
2. 必须逐文件探测
   `.../stepXXXX-unsharded/{config.yaml,train.pt,model.safetensors,optim.safetensors}`；
   safetensors 缺失时按官方文档回退 `model.pt` / `optim.pt`。
3. 上一轮 HF repo/revision screening **保留为有效历史记录**——它只证明
   HF model repos 是 weights-oriented publishing path，**不能**单独证明
   OLMo standard checkpoint training state 不存在。

## 2. 官方 CSV 读取与 checkpoint 冻结（先于任何探测）

- 读取 `configs/official-0425/OLMo-2-0425-1B.csv`（stage1 主训练 run，
  195 个 checkpoint，step 0 … 1,907,359；SHA256 `3db00360…`）。
  一并登记 `OLMo-2-0425-1B-stage2.csv`（stage2 续训，71 行，
  SHA256 `4df0d7a5…`；本轮不从中选取）。
- 选取规则（**仅按 step**，不得按"哪个恰好能下载"挑选）：
  EARLIEST=第一个合法条目；EARLY/MIDDLE/LATE=最接近最终 step 的
  20%/50%/80% 的已列出 step。
- **冻结结果**（写入 `selected_checkpoint_manifest.json`）：

  | role | step | 进度 | URL |
  |---|---:|---:|---|
  | EARLIEST_LEGAL | 0 | 0.00% | …/peteish1/step0-unsharded/ |
  | EARLY ~20% | 380000 | 19.92% | …/peteish1/step380000-unsharded/ |
  | MIDDLE ~50% | 950000 | 49.81% | …/peteish1/step950000-unsharded/ |
  | LATE ~80% | 1530000 | 80.22% | …/peteish1/step1530000-unsharded/ |

## 3. 逐文件探测方法

每个冻结目录 × {config.yaml, train.pt, model.safetensors, optim.safetensors}：

1. `curl -sIL`（HEAD + 跟随重定向）记录：final HTTP status、redirect 数与
   链、content-length、content-type、ETag、last-modified。
2. 对 200 的文件做实际可读性验证：小文件（≤10 MB）完整 GET；
   大文件 `curl -r 0-1048575`（**1 MB 上限**）range GET，记录
   content-range 与前 1 MB 的 SHA256。
3. safetensors 返回非 200 时，追加探测官方文档指定的回退名
   `model.pt` / `optim.pt`（同方法）。
4. 全部命令逐条记录于 `COMMANDS.log`；逐文件结果记录于
   `per_file_http_probe.csv`（safetensors）与 `pt_fallback_probe.csv`（.pt）。

## 4. 禁止事项（继承红线）

- 禁止以 `https://olmo-checkpoints.org/` 根或目录 URL 的 HTTP 状态
  作为任何可用性结论（仅可作对照记录）。
- 禁止下载完整 model/optimizer/Dolma/future streams。
- 禁止非官方镜像作为主要来源；官方替代 host 仅当官方文档/配置/
  issue 明确迁移声明时才可做一次对应文件级核验。
- 禁止修改任何冻结科学结果；禁止 GPU；禁止启动 HVP/FD/B/C。

## 5. 分类规则（冻结，唯一允许的三个判定）

- **A. FULL_STANDARD_CHECKPOINT_AVAILABLE**：同一 checkpoint 的
  model + optimizer + trainer state + config 均存在且可读
  → 上一轮判定必须修正为 **AVAILABILITY_RECHECK_PASSED**。
- **B. PARTIAL_STANDARD_CHECKPOINT_AVAILABLE**：config/train 在而
  model 或 optimizer 缺失；若全部冻结 checkpoint 都缺 model/optimizer →
  **STANDARD_CHECKPOINT_STORAGE_PARTIALLY_UNAVAILABLE**（不得写作
  "HF 没有 state"）。
- **C. FULL_STANDARD_CHECKPOINT_UNAVAILABLE**：全部冻结 checkpoint 完成
  文件级探测 + 无官方替代 host + model/optimizer/train 无法同时恢复
  → 方可保持 `EXTERNAL_CHECKPOINT_UNAVAILABLE`，且原因必须写作
  "official standard-checkpoint storage specific required files
  unavailable"。
