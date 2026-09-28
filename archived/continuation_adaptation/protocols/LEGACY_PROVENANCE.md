# LEGACY_PROVENANCE — continuation_adaptation vendor 登记表

vendor 日期：2026-09-26。来源仓库：`/home/b205/projects/training_dynamics_lab/legacy_fluctuation_response/`
（Git HEAD `6ac4942eeedaf3ed0e675c7f67c0e5f9975a64e8`，只读冻结，本项目不改动其中任何文件）。

## 1. Verbatim 复制的代码模块（12 个）

复制方式：`cp` 逐文件复制；下表 SHA256 于 2026-09-26 对源与副本双向核对一致（`sha256sum`）。
源路径前缀：`legacy_fluctuation_response/research/src/fr/`；目标前缀：`src/vendored_fr/`。

| 文件 | SHA256 | 用途 |
|---|---|---|
| `full_state.py` | `709334f8fb229fc38869787dc0d5a7f57a3d687276b6f2392564a90f0f5853f` | 完整训练态 Z_t（model/optimizer/scaler/RNG/metadata）save/load |
| `rng.py` | `0203e0ef0e81e71b4c58a2ec502579307655a85b3f9eab4566c3c1c779ad40d` | Python/NumPy/Torch CPU+CUDA RNG capture/restore |
| `stream_rng.py` | `242956de271a8b839ddb3b39cc7e63d48b63ed253841255580d34e39ecd867a4` | 流种子派生（本轮 CRN 未用，保留备用） |
| `batch_manifest.py` | `6bf1703f0e09f4b713113bfa05ad3eccb83119c8ad083d4a1df5688ac932ffb` | 独立 Generator randperm + 显式 batch 清单 + SHA256 |
| `batch_stream.py` | `2614f7bd1a4d95cf3d3cf10b37da3c8c7c683eba2f45631d946ce80c30c6d9bb` | stream_id → 多路 seed（本轮未用，保留备用） |
| `eval_observables.py` | `f131c7f0f8a965f0c22f2939201c9520ffc45507498e6798bae1c208a386319` | eval-pool 逐 token CE（mean/p95/p99/CVaR99），RNG 保护 |
| `minimind_v03_runtime.py` | `fb17f8ea4db28da5c7b0bb3f0d5397131c42ca237a55b31a608330ea8c11938f015` | make_amp / stack_batch / copy_jsonl_prefix |
| `stage_c_runtime.py` | `692a7f8e143359601ab79d94887d5df176d1ce0bc4dd3bc303a5b5950d21d1a0` | train pool 内确定性 future batch |
| `pilot_partition.py` | `9fece66acd9a81e523597e24d7a08e8bd0c0754d4174eaaaeaae905892a72c9f` | train/probe/eval 互斥划分（SHA 校验加载） |
| `hashing.py` | `ccc94714b4c2c00c5d5c3bd05fffad3b2c4da811ca8ff0e802651b0daddf7536` | 嵌套状态稳定 SHA256 |
| `native_adamw_guard.py` | `19cdae886e8fa57c8672c9aa323070e975675a35317f55a481c8d026ad47ad40d` | 只读 optimizer 签名 + 原生 AdamW 断言 |
| `native_schedule.py` | `225bcd942090ea6be5c7564dc18f29ab6ee47a80547ff6238b92f6f9b9b6b263` | 逐 micro-batch 写入原生 LR 时序 |

`src/vendored_fr/__init__.py` 为本项目新增的包声明文件（原包 `__init__.py` 含旧项目版本号声明，未复制）。

## 2. 明确不 vendor 的内容

- 全部 GTTR 专有代码：`f0_terminal_rule_*`、`c0/c1/cb1/d1/r1/r2b_*`、`phase_a/phase_b_*`、
  `a0*` 系列（切线/伴随/HVP/controller）
- `perturbation.py` / `direction.py`（扰动-响应测量协议，旧方向专属）
- `theta_protocol.pyc`（sourceless 字节码，依赖风险，新 runner 不依赖它）
- `confirmatory_stats.py` / `cross_seed_stats.py` 等统计模块（P4 分析阶段按需再议）

## 3. 只读数据依赖（不复制、不修改，加载时校验）

| 资产 | 路径（相对 lab root） | 校验 |
|---|---|---|
| v07 state bank（C0 checkpoints） | `legacy_fluctuation_response/outputs/e3_confirmatory/v07/seed_{2030..2037}/states/state_step_{046..082}.pt` | 每次加载记录文件 SHA256 + metadata `partition_sha256` 一致性断言 |
| train/probe/eval partition | `legacy_fluctuation_response/outputs/e3_confirmatory/v07/partition_v07.json` | `load_partition` 内建 SHA256 校验 |
| 语料子集 D0（4096 行） | `legacy_fluctuation_response/outputs/e3_confirmatory/v07/e3c_subset.jsonl` | SHA256 `1e657af9baae0784e582d82ef936160473aa069878e422ef48683a718056b0f4`（冻结记录值，加载时核对） |
| 模型代码 + tokenizer | `minimind_base/`（`model/model_minimind.py`、`model/tokenizer.json`、`dataset/lm_dataset.py`、`trainer/trainer_utils.py`） | 经 `sys.path` 只读导入，不复制不修改 |

## 4. 环境

- conda env `fr_minimind`（Python 3.11.16，torch 2.11.0 + CUDA）
- 运行约定：cwd = `continuation_adaptation/`，`PYTHONPATH` 由脚本自行注入（src + minimind_base）
- GPU：单卡可见（`CUDA_VISIBLE_DEVICES` 显式指定；C0 的 CUDA RNG 恢复要求保存/加载设备数一致 = 1）
