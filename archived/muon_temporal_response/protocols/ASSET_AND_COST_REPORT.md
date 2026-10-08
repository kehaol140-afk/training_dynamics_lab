# ASSET_AND_COST_REPORT — 资产、缺口与成本（T1）

日期：2026-09-28。性质：本轮实测盘点。所有路径为服务器实际路径；SHA 已在
`IMPLEMENTATION_ALIGNMENT.md` 与 `references/` 登记。旧项目一律只读。

---

## 1. 现有资产（可复用，无需新建）

### 1.1 平台与代码

| 资产 | 路径 | 状态 |
|---|---|---|
| MiniMind 30M 模型定义 | `../minimind_base/model/model_minimind.py`（SHA 47f124f5…，PROVENANCE §2） | 只读可用 |
| 训练 runner（M1/M2 生产循环） | `../muon_transition_experiment/src/muon_transition/m1_runner.py` | 只读参考；T2 新写采集器，不回改 |
| Muon/AdamW 双臂构造 + routing | 同上 `optimizers.py`、`routing.py`（56 矩阵 routing 表在 `outputs/m0/routing_report.json`） | 只读可用 |
| 确定性数据流（thread split、打包、exact-p manifest、CRN） | `data.py`、`mixture.py`、`partitions.py`、`m1_data_cache.py`（含 /tmp 缓存 `mte_domain_train_windows.pt`，A=385,071 / B=295,825 train windows） | 只读可用 |
| checkpoint 格式 | `checkpoint.py`（mte_state_v1，payload SHA + 原子写 + 四路 RNG） | T2 直接复用模式 |
| eval（NLL sum/count） | `eval_nll.py` + `outputs/m0/data_audit/eval_pools.pt`（dev/confirm 各域 512 windows，封存纪律） | 只读可用 |
| Muon 冻结审计 | `outputs/m0/muon_freeze.json` + `docs/MUON_IMPLEMENTATION_AUDIT.md` | 本轮 SHA 复核一致 |

### 1.2 完整训练态（Muon 均为 step 2048 终点）

| 资产 | 数量 | 内容 |
|---|---|---|
| M1 runs | 12（adamw 6 / muon 6，p=0.5，seed 2030） | `outputs/m1/m1_*/final_state.pt`：model fp32 + Muon momentum×56（fp32）+ AdamW aux m/v/step + RNG + cursor |
| M2 runs | 28（24 base + 4 justified ext） | `outputs/m2/m2_*/final_state.pt` 同格式 |
| dev/confirm eval pools | 每域 512 windows | 封存纪律保留；本轮零访问 |

**限制（实测）**：单一初始化种子（2030）；单一 LR/WD 网格配置各格；无 step512/1536 中间态；
无逐步 G/M/H/D 日志（见 IMPLEMENTATION_ALIGNMENT §3.1）。

## 2. 缺失与需后续授权补建的资产（T2 视角）

| 缺口 | v2 建议口径 | 现状 | 补建成本估计（不在本轮执行） |
|---|---|---|---|
| Muon 中间训练态（step 512 / 1536） | 2 开发种子 × 2 阶段 = 4 状态 | **不存在**（不能从 final_state/日志推定） | 每条父轨迹 ≤2048 updates（M1 实测 ~10-13 min/run on 2×5090）；4 条 ≤ 1 GPU 时；需明确授权（本轮禁止） |
| 第二开发种子 | v2：2 个初始化种子 | 只有 seed 2030 | 同上（父轨迹时顺带保存） |
| 逐步 G/M/H/D 采集器 | T2 核心测量 | 不存在（obs hooks 无矩阵记录） | 新代码；单 batch 梯度成本须先测吞吐 |
| 梯度 bank | 每状态 2×128 batches | 不存在 | 上限 1024 次完整 batch 梯度计算，零 committed updates；吞吐待测（拒绝凭经验猜 GPU 时长） |
| AdamW 对照状态 | v2：放真实路径层 | M1/M2 有 AdamW final states（12+28），中间态同样缺失 | 同 Muon 口径 |

## 3. 本轮（T1+T0）成本台账（实测）

| 项 | 实测 | 预算 | 状态 |
|---|---|---|---|
| CPU 数值总时长 | 测试 ~2.6s + run_t0_cpu 4.8s + 汇总 ~13s + 探索性核对 ≪ | 30 min | 远低于预算 |
| GPU（单卡 cuda:0，RTX 5090） | `run_t0_native.py` 110.7 s（含全部主矩阵×2 重复 + 脉冲×2 + 对齐 + 量化检查） | 15 min | ~12% 预算 |
| 合成递推次数 | FP64 动量递推 67,072 + 原生 optimizer steps 67,072（各含脉冲） | 单独计数 | 已记录 `cost_ledger.json` |
| 输出总量 | 2,322,258 B（~2.2 MB，NPZ/CSV/JSON/PNG） | 1 GB | ~0.2% |
| 新增 Transformer optimizer updates | **0** | 0 | 符合 |
| 新增完整 batch 梯度计算 | **0** | 0 | 符合 |
| 确认集（confirm pools）访问 | **0** | 0 | 符合 |
| 父轨迹训练 | **0** | 0 | 符合 |
| 旧目录修改 | **0**（全部只读） | 0 | 符合 |

## 4. T2 存储估算（按实际配置推算，供草案，非承诺）

六矩阵（layers[1]/layers[6] × {q_proj 512×512, k_proj 256×512, down_proj 512×1664}）：
每矩阵元素数 262,144 / 131,072 / 851,968 → 六矩阵合计 ≈ 7.49M 元素。
fp32 全矩阵保存 G/M/H/D 四量 = 7.49M × 4 × 4 B ≈ **119.8 MB / batch / 状态**。
每 bank 128 batches → **15.3 GB / bank / 状态**；4 状态 × 2 bank = **~123 GB**（超默认 1 GB
口径，属 T2 预算另批）。低维投影选项（如 512 维随机正交基 ×双投影）可降至
7.49M→(0.26M+…) 量级，覆盖率/残差必须记录（v2 §8）。磁盘余量 ~246 GB（PROJECT_CONTEXT），
全矩阵方案可行但紧张，草案建议混合：K 投影 + 全矩阵抽帧。
