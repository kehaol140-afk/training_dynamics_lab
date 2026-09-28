# CHECKPOINT_AVAILABILITY_REPORT — FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01

> 报告时间：2026-09-16T17:05Z。范围：只读 HTTP 文件级核验；无 GPU、无科学
> response、无完整大文件下载（每大文件 ≤1 MB range read）。

## 1. 最终判定

# **A. FULL_STANDARD_CHECKPOINT_AVAILABLE**

4 个预注册代表 checkpoint（EARLIEST step0 / EARLY step380000 / MIDDLE
step950000 / LATE step1530000，选自官方 `configs/official-0425/
OLMo-2-0425-1B.csv`，stage1 run peteish1）的 **model + optimizer +
trainer state + config 四件全部 HTTP 200 且实际可读**。

按冻结分类规则 §5-A：**上一轮 `EXTERNAL_CHECKPOINT_UNAVAILABLE` 修正为
`AVAILABILITY_RECHECK_PASSED`**，**G1c 可以恢复继续**（下一步 =
G1c 1B feasibility gate，须用户授权后启动，本轮不自动执行）。

## 2. 官方 docs 对 unsharded checkpoint 的定义（本轮修正核验逻辑的依据）

OLMo 官方 `docs/Checkpoints.md`（快照 SHA256 `f84cf11d…`）：

- standard（unsharded）checkpoint 目录 = `config.yaml` +
  `model.safetensors` + `optim.safetensors` + `train.pt`；
  **旧 release 用 `.pt` 扩展名**（`model.pt` / `optim.pt`）。
- **"These 'directory' URLs cannot currently be directly accessed, but
  files within the directory are publicly accessible."**
- HF Hub 上的 OLMo-1B/OLMo-7B 等是 Transformers/HF-OLMo **转换产物**
  （weights-oriented），按定义不含 training state——上一轮的 HF
  screening 记录保留为有效历史，但其解读修正为"HF 路径不发布
  training state"，**不能**推出 standard checkpoint 无 training state。

## 3. 实际读取的官方 CSV

| CSV | 行数 | step 范围 | SHA256 | 用途 |
|---|---:|---|---|---|
| `configs/official-0425/OLMo-2-0425-1B.csv`（stage1） | 195（含表头） | 0 … 1,907,359 | `3db00360…c9e3b` | **选取来源**（`official_csv_snapshot.csv`） |
| `configs/official-0425/OLMo-2-0425-1B-stage2.csv`（stage2 续训） | 72（含表头） | 1,000 … 23,852（3 ingredient） | `4df0d7a5…c839d` | 登记在案，本轮不选取 |

## 4. 冻结检查的 4 个 checkpoint 与逐文件真实 HTTP 状态

统一前缀 `https://olmo-checkpoints.org/ai2-llm/peteish1/`；redirect 数
**全部为 0**（host 直接服务，无替代 host 跳转）。`✓=200 且实际读取验证`。

| step（角色） | config.yaml | train.pt | model.safetensors | optim.safetensors | model.pt（官方回退名） | optim.pt（官方回退名） |
|---|---|---|---|---|---|---|
| 0（EARLIEST） | ✓ 163,645 B 全读 | ✓ 14,476 B 全读 | 404 | 404 | ✓ 5,939,703,038 B，1MB range 验证，ETag `04b289a1…` | ✓ 11,879,442,342 B，1MB range 验证，ETag `624c48bf…` |
| 380000（EARLY 19.92%） | ✓ 163,723 B | ✓ 26,508 B | 404 | 404 | ✓ 5,939,703,038 B，ETag `bc5a6ea2…` | ✓ 11,879,442,406 B，ETag `fc8bffd3…` |
| 950000（MIDDLE 49.81%） | ✓ 163,724 B | ✓ 17,036 B | 404 | 404 | ✓ 5,939,703,038 B，ETag `4c76210f…` | ✓ 11,879,442,406 B，ETag `177210d1…` |
| 1530000（LATE 80.22%） | ✓ 163,725 B | ✓ 26,572 B | 404 | 404 | ✓ 5,939,703,038 B，ETag `b33c9aa8…` | ✓ 11,879,442,406 B，ETag `3f73439c…` |

格式判读（与官方 docs 一致）：该 run 属"旧 release `.pt` 扩展名"情形——
safetensors 变体不存在（404），但 `model.pt`/`optim.pt` 存在且可读；
四 checkpoint 的 model.pt 字节数完全相同（同构模型），ETag 逐步不同
（内容真实随 step 变化）；optim.pt ≈ 2× model.pt 字节数，与 AdamW
m+v 两阶矩 fp32 存储一致。**前 1 MB 均为 `PK\x03\x04` zip magic**
（torch save 格式），train.pt 内可见 `train/data.pkl` 成员名；
config.yaml 为真实 YAML（`run_name: peteish1-20241108_215905`，
d_model=2048、n_layers=16、swiglu、rope，与 ~1.24B 参数一致）。

逐文件明细（content-length / content-range / content-type / ETag /
last-modified / 读取字节 SHA256）见 `per_file_http_probe.csv`（16 行）
与 `pt_fallback_probe.csv`（8 行）；原始 curl 头存于 `probe_evidence/`。

## 5. 是否存在官方替代 host

**否。** 全部 200 响应 redirect_count=0，由 olmo-checkpoints.org 直接
服务；官方 docs/CSV/YAML 无 checkpoint 文件迁移声明。`olmo-data.org`
仅承载 Dolma 语料/eval 数据（stage1 YAML `paths:`），非 checkpoint host。

## 6. 目录 404 与具体文件状态是否不同

**是，显著不同且与官方文档逐字一致**：同一时刻
`…/step950000-unsharded/` 目录 URL → HTTP 404，而
`…/step950000-unsharded/config.yaml` → HTTP 200。目录 404 是对象存储
网关无前缀列表权限的预期行为，**不作可用性结论**。上一轮以根 404
登记的负面证据按此修正解读（历史记录保留，不删除）。

## 7. 训练栈事实登记（供 feasibility gate 使用，源自 config.yaml 快照与 stage1 YAML）

AdamW `lr=4.0e-4`、`wd=0.1`、`eps=1e-8`、`betas=(0.9,0.95)`、
`decay_norm_and_bias=true`、`decay_embeddings=false`；
`max_grad_norm=1.0`；`precision=amp_bf16`；`max_sequence_length=4096`；
`global_train_batch_size=512` / `microbatch=4`（native context 4096，
B8 需按此评估）；数据为 Dolma v2 预处理 .npy（`olmo-data.org`，
dolma2 tokenizer）。资源提示（仅登记，未验证）：单个 checkpoint 需
model.pt ~5.94GB + optim.pt ~11.88GB ≈ **17.8 GB 磁盘**（fp32），
restore 后训练态内存 ~3× 参数 ×4B ≈ 15GB 级 + HVP 开销——feasibility
gate 必须在实测硬件（2× RTX 5090 32GB，单卡可用余量需重新探测）上
按 B6/B7 重新评估。

## 8. 结论与下一步

- 判定：**A. FULL_STANDARD_CHECKPOINT_AVAILABLE**（4/4 冻结 checkpoint
  四件齐全可读）。
- 上一轮 `EXTERNAL_CHECKPOINT_UNAVAILABLE` **需要修正**：
  → `AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY`。
  修正性质：核验路径修正（HF weights-path screening 不覆盖官方
  standard-checkpoint 存储路径），**不是**科学结论改判；G1c 预注册
  协议（B0–B17）与全部红线继续有效。
- **可以重新进入 G1c 1B feasibility gate**——但本轮到此停止：不自动
  下载完整 checkpoint、不自动运行 GPU。feasibility 的启动须用户授权。
