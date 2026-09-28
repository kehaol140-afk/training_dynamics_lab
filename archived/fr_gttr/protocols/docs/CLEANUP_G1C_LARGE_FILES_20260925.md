# G1c 大型文件清理说明（2026-09-25）

## 背景

`outputs/generalization_g1c_feasibility_v01/`（G1c 1B feasibility gate，
OLMo-2-0425-1B / peteish1）曾占用约 61G。该 gate 已中止：Phase 0/1 PASS、
Phase 2 中途中止（C 分支失败，见 `G1C_C_FAILURE_DIAGNOSTIC_REPORT.md`），
正式 G1c 科学实验从未授权（协议 §6.9 红线）。本轮确认 8 个大文件均非
F0 或其他任何在跑/待跑实验的输入（F0 协议与 f0_* 代码零引用）后删除；
协议、配置、数值结果、报告、manifest、SHA256 与恢复所需信息全部保留。

## 删除清单（8 个文件，共 65,336,855,410 字节 ≈ 60.8 GiB）

| 文件（相对 `outputs/generalization_g1c_feasibility_v01/`） | 字节数 | 恢复锚点 |
|---|---:|---|
| `checkpoints/step380000_model.pt` | 5,939,703,038 | SHA256 `ca2fa02c8192651c7280e848e96c69f954451f784f26b97e8b11385c54470fa7`（`SHA256SUMS_phase2.txt`） |
| `checkpoints/step380000_optim.pt` | 11,879,424,306 | SHA256 `3e48789a515e32d6d44618fecc77efd896b1055ce467568e06a79eab9df3bae6`（同上） |
| `checkpoints/step950000_model.pt` | 5,939,703,038 | SHA256 `6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b`（`SHA256SUMS_step950000.txt`） |
| `checkpoints/step950000_optim.pt` | 11,879,424,306 | SHA256 `ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2`（同上） |
| `checkpoints/step1530000_model.pt` | 5,939,703,038 | SHA256 `d55f32683d56d1e61d0b919d265a611bebec864875db404bcb0642b0dd419120`（`SHA256SUMS_phase2.txt`） |
| `checkpoints/step1530000_optim.pt` | 11,879,424,306 | SHA256 `58f7f01d73a1e7772568a70234759ef82fa3d65367b75a813dc5b60e17158884`（同上） |
| `streams/q_stream1.pt` | 5,939,709,539 | 张量级 SHA256 `08128dd4f86429399d9b732778d18e81b2a35f76ef978d71a25c3b883ca2cbb0`（与 phase1.json / cdiag_summary.json / PHASE1 报告三处存档一致） |
| `streams/q_stream2.pt` | 5,939,709,539 | 张量级 SHA256 `81c4273671ad07f601078fc1692ceb0f213c3b05e897f568d2d2b4bfd7ffa119`（删除前自文件内嵌 payload 提取并存档于本文件） |

磁盘变化（`df -h /`）：770G used / 187G avail → **709G used / 248G avail**
（实际释放 61G）。删除后该目录余 15M。

## 恢复方法

**checkpoints（6 个 .pt）— 重新下载**：
官方源 `https://olmo-checkpoints.org/ai2-llm/peteish1/step{380000,950000,1530000}-unsharded/`
（model.pt / optim.pt；来源与下载脚本见协议 `G1C_FEASIBILITY_PROTOCOL.md` §2 与
`code/g1c_download.sh`）。下载后按上表 SHA256 与目录内保留的
`SHA256SUMS_step950000.txt` / `SHA256SUMS_phase2.txt` 校验。`train.pt` 与
`config.yaml` 未删除，无需恢复。

**q_stream（2 个 .pt）— 冻结种子确定性再生成**：
q 为全参数（131 个张量、共 1,484,916,736 元素）iid N(0,1)、单位总 L2 范数
（协议 §6.1 / Amendment A1），由 `research/scripts/g1c_1b_feasibility_v01.py`
按冻结种子经 `fr.direction.save_direction`（direction_version 0.3）生成：

- q_stream1：base_seed 20260917，stream_seed `8779495656483434628`
  =derive_stream_seed(20260917,1)，q_seed `4655128343037472768`
  =derive_stream_seed(stream_seed,77)；生成时 state_step 380000。
- q_stream2：stream_seed `9988186605176222288`=derive_stream_seed(20260917,2)，
  q_seed `6083905373015550499`；生成时 state_step 950000。

再生成后用 `fr.direction.direction_sha256` 对照上表张量级 SHA256 验证。

## 保留物（未删除）

协议 3 份 + 修订、PHASE0/1 报告、C 失败诊断报告、`results/`（phase0/1
json、cddiag、运行日志）、两个 SHA256SUMS、SOURCE_HASHES、
future_stream_manifest.json、三组 `train.pt`/`config.yaml`、tokenizer、
两个 token 流 `.bin`、`code/` 代码快照、COMMANDS.log、ENVIRONMENT.json。
