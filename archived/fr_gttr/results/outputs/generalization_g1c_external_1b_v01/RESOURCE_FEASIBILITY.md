# RESOURCE FEASIBILITY — FR_G1C_EXTERNAL_1B_STRESS_TEST_V01

Created: 2026-09-16T16:05:00Z. Status: **BLOCKED_BEFORE_GPU** —
实验在 B2 公开制品核验阶段即得出 EXTERNAL_CHECKPOINT_UNAVAILABLE，
B6 的 GPU feasibility gate 从未触发，本节登记实测硬件、资源推算与
若未来重试时的可行路径（仅设计，不构成执行授权）。

## 1. 实测硬件（2026-09-16 nvidia-smi 探针）

| GPU | 型号 | 总显存 | 探针时已用 | 备注 |
|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 5090 | 32,607 MiB | 16,787 MiB | 与外部任务共享；本项目使用 GPU0 |
| 1 | NVIDIA GeForce RTX 5090 | 32,607 MiB | 23,958 MiB | 外部任务；**永不可杀、永不触碰** |

**与用户假设的偏差**：任务书假设"单卡 A100-PCIE-40GB"；实测为
2× RTX 5090 32GB（GPU0 可用余量约 15.8GB）。已如实登记
（ENVIRONMENT.json discrepancy_note）。对本轮无实际影响（阻塞在
GPU 之前），但对任何未来重试是硬约束：比 40GB 假设更紧。

## 2. 若 checkpoint 可用时的显存推算（OLMo-1B, 1.24B params）

- 模型权重 bf16：≈2.5GB；fp32 master 视实现 ≈5GB。
- AdamW m/v fp32：≈9.9GB（单卡驻留）。
- 权重 + m/v + 梯度 ≈15–17GB，已超过 GPU0 当前余量（~15.8GB）→
  必须启用 B7 允许路径：optimizer state 常驻 CPU、逐层加载/回写
  （offload）、activation checkpointing、sequential HVP、microbatch
  化 deterministic replay。
- 原生 context=2048、global batch=512（microbatch 4）下，activation
  显存即便 checkpointing 后于 HVP double-backward 期间仍峰值紧张；
  B8 顺序为 native 2048 先试 → 不可行则登记 short-context stress
  （seq 128/256，tokenizer/objective/corpus 不变）→ 再不可行
  RESOURCE_LIMIT。

## 3. 磁盘与带宽

- 磁盘剩余 316GB（/home 所在盘）。单个 unsharded OLMo-1B checkpoint
  （model 2.5GB + m/v ~10GB + train state <1GB）≈13GB；3 个 stage
  ≈39GB，磁盘可承受。
- 带宽未测速（未进入下载阶段）。

## 4. Feasibility gate 设计（B6，未执行）

1 checkpoint × 1 stream × H=1，依次：model+optimizer restore →
baseline fwd/bwd → gradient injection → H1 B/Jacobian → central FD（≥2
epsilon levels，B12）→ HVP/second-order path → full C → peak memory 记录
→ all-finite 检查；通过后 H=4 单 stream；再后才允许进入正式矩阵（B10）。

## 5. 结论

本轮 **无需、也未进行任何 GPU 实验**；阻塞原因是公开制品不可用而非
资源。若未来用户授权更换外部模型家族（须新预注册），本节硬件事实与
B7/B8 资源策略直接复用。
