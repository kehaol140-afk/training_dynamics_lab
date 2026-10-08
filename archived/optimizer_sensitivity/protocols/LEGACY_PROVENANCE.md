# LEGACY_PROVENANCE — optimizer_sensitivity vendor 登记表

来源仓库（只读）：`/home/b205/projects/training_dynamics_lab/legacy_fluctuation_response/`（迁移前绝对路径 `/home/b205/projects/fluctuation_response/`；Git HEAD `6ac4942`）。

vendor 日期：2026-09-26。目的：F0 既有 bank sensitivity 审计所需的冻结基础设施。
**未 vendor**（按 mandate 排除）：A0 controller、A0L/A0N/A0U scheduling、GTTR terminal-rule 代码、任何已失败的 long-kernel/control policy。

| 文件（src/opsens/ 下） | SHA256（vendor 副本） | 与 legacy 原件比对 | 原绝对路径（迁移前） | 来源实验/角色 | 复制方式 |
|---|---|---|---|---|---|
| a0c_adamw_response_propagation_v01.py | `7b1dde12db88986eba889b5bf9a79e5d73327b67cc0bbebbfc8194b39caec433` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/a0c_adamw_response_propagation_v01.py` | A0C response_propagation_v01（冻结指标 _metric/relative_error、adamw_tangent_step、clip_jvp 唯一定义源；A0C–A0F2/G1a/G1b/G2/G3 全线引用） | 原样复制（verbatim） |
| rng.py | `0203e0ef0e81e71b4c58ae2ec502579307655a85b3f9eab4566c3c1c779ad40d` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/rng.py` | v0.6 冻结框架基线（E0–E3 基础设施，2026-09-02；全部实验 CRN/RNG 控制） | 原样复制（verbatim） |
| stream_rng.py | `242956de271a8b839ddb3b39cc7e63d48b63ed253841255580d34e39ecd867a4` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/stream_rng.py` | v0.6 冻结框架基线（future stream seed 配方） | 原样复制（verbatim） |
| batch_stream.py | `2614f7bd1ad95cfb3d3cf10b37da8c8c7c683eba2f45131d946ce80430c6d9bb` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/batch_stream.py` | v0.6 冻结框架基线（E2 batch 流） | 原样复制（verbatim） |
| batch_manifest.py | `6bf1703f0e09f4b713113bfa05ad3eccb83119c8ad0838d4a1df5688ac932ffb` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/batch_manifest.py` | v0.6 冻结框架基线（batch manifest/排列） | 原样复制（verbatim） |
| full_state.py | `709334f8fb229fc38869787dc22f2f7a57a3d687276b6f2392564a90f0e5853f` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/full_state.py` | v0.6 冻结框架基线（full-state checkpoint 保存/加载，fr_state_version 0.2） | 原样复制（verbatim） |
| hashing.py | `ccc94714b4c2c00c5d5c3bd05fffad3b2c4da811ca8ff0e802651b0daddf7536` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/hashing.py` | v0.6 冻结框架基线（stable_hash） | 原样复制（verbatim） |
| native_adamw_guard.py | `19cdae886e8fa57c8672c9aa323070770e975675ad35617f55a48ddc02677d7c` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/native_adamw_guard.py` | v0.6 冻结框架基线（[FR-NATIVE-ADAMW] 守卫） | 原样复制（verbatim） |
| native_schedule.py | `225bcd942090ea6be5c7564dc18f29ab6ee47a80547ff6238b92d6f9b9b6b263` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/native_schedule.py` | v0.6 冻结框架基线（原生 LR 路径） | 原样复制（verbatim） |
| minimind_v03_runtime.py | `fb17f8ea4db285dda061a4376daf5a2b0fcf1a012e38fd72052ab033e114e235` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/minimind_v03_runtime.py` | v0.3 MiniMind 运行时（FR 侧模型构建/AMP/replay 语义） | 原样复制（verbatim） |
| stage_c_runtime.py | `692a7f8e143359601ab79d94887d5df176d1ce0bc4dd3bc303a5b5950d21d1a0` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/stage_c_runtime.py` | Stage C v0.4 运行时（E1/E2 执行） | 原样复制（verbatim） |
| perturbation.py | `80be419a024f68a2ecccbd3ca2bc8d290be7ccd9f05fe75122e883c8c0756820` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/perturbation.py` | v0.6 冻结框架基线（pulse-before-clip 扰动语义） | 原样复制（verbatim） |
| direction.py | `7040c5b6be82b8eebaedb7daa2ed0533c011249776c229cf533a5dff7037bebd` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/direction.py` | A0 系列（held-out probe 方向构造，2026-09-10） | 原样复制（verbatim） |
| theta_protocol.pyc | `61e1bae38a2b20d0407e134f215094060624178da215d2232c94739d1d7751e7` | 一致 | `/home/b205/projects/fluctuation_response/research/src/fr/theta_protocol.pyc` | theta_protocol replay 原语（源码 2026-09-12 丢失后经冻结源编译字节码恢复；run_theta_branch/seed_future_stream 等；全部 A0 runner 依赖） | 原样复制（verbatim） |
| minimind_adapter_v02.py | `8a4e500047037b9c1ab0dc7c5a8c854dcb2e69421753dd2d83eaf86f245d16be` | 不同（重构） | `/home/b205/projects/fluctuation_response/research/adapters/minimind_adapter_v02.py` | MiniMind 适配层 v0.2（模型构建/加载接入 FR replay 语义） | 重构（仅 2 行 import 路径） |

## 唯一重构说明

- `minimind_adapter_v02.py`：仅第 14–15 行 `from fr.*` → `from .*`（包名迁移），其余逐字节一致；legacy 原件未改动。
- `theta_protocol.pyc`：sourceless 字节码原样复制（cpython-311，与 legacy SHA256 一致）；源 .py 已永久丢失（2026-09-12 事件），行为等价性见 legacy docs/FR_PROJECT_STATUS.md §13。
- 其余 14 个文件与 legacy 逐字节一致（SHA256 双向核对）。

## F1 追加 vendor（2026-09-26，1B replication 所需冻结 G1C 适配器）

| 文件（src/opsens/ 下） | SHA256（vendor = legacy 逐字节一致） | 原绝对路径（迁移后） | 角色 |
|---|---|---|---|
| g1c_olmo2_architecture.py | `cd2b23f5a33165390d5d7c19a262e7d3c115c680c0ac305b066abf331a265822` | `legacy_fluctuation_response/research/src/fr/g1c_olmo2_architecture.py` | OLMo-2-0425-1B 冻结架构适配器（G1C_PROTOCOL §5/6.8 版本化产物；state_dict 键逐字对齐官方 checkpoint） |
| g1c_olmo2_checkpoint.py | `e2889921c935857b43f7d845949872ef51f9b0e5549cba8f4187028d3eb12098` | 同上目录 | 官方 unsharded model.pt/optim.pt/train.pt 恢复器（FSDP 前缀剥离、真实 torch AdamW state 重建） |
| g1c_olmo2_tangent.py | `880b48ccf90ea76bc584f7aa9c62c80a9f59017cfc58fcf03e35ef8ca96d4a32` | 同上目录 | 异构 param-group AdamW 切线步（冻结逐行拷贝，仅 group 聚合不同；单测证明与冻结函数在单 group 下全等） |

- 三个文件均 verbatim（SHA256 双向核对一致）；legacy 原件未动。
- F1 只读数据依赖（不复制）：legacy `outputs/generalization_g1c_feasibility_v01/{checkpoints/step*_config.yaml, step*_train.pt, streams/stream1_tokens.bin, streams/dolma2_tokenizer.json, results/cdiag/*}`。

## 只读数据依赖（不复制、不移动、不修改）

- v07 bank（4.98M，56 态）：`legacy_fluctuation_response/outputs/e3_confirmatory/v07/`
- G1a bank（26.68M，9 态）：`legacy_fluctuation_response/outputs/generalization_g1a_26m_v01/bank/`
- G1b bank（100.78M，9 态）：`legacy_fluctuation_response/outputs/generalization_g1b_100m_v01/bank/`
- 训练语料 D0：`legacy_fluctuation_response/outputs/e3_confirmatory/v07/e3c_subset.jsonl`（SHA256 `1e657af9…b0f4`）
- tokenizer：`../minimind_base/model/tokenizer.json`（vocab 6400，逐文件 manifest 见 `../audits/minimind_base_file_manifest.sha256`）
