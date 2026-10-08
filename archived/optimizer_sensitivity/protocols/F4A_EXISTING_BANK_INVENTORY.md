# F4-A — 既有 Bank Inventory（Scale × Phase × Seed 覆盖盘点）

- 日期：2026-09-30。方法：直接扫描 legacy 三 bank 的全部 `state_step_*.pt` 并 `torch.load` 读取 metadata（非文档转抄）；模型配置按 legacy 协议重建并经 **state_dict 形状逐项匹配验证**（三尺度全部 47/79/145 张量形状一致，可完整 restore）。
- 训练协议（从 bank metadata / legacy 协议核验）：三 scale 完全同构——corpus D0（e3c_subset 4096 行）、partition_v07（同一 train/eval 划分）、seq 128、microbatch 4 × accum 2（1024 tokens/optimizer-step）、bf16 autocast 前向、AdamW(0.9, 0.999, eps 1e-8, wd 0.01)、global clip 1.0、LR 5e-4 cosine（`get_lr`，**total_micro=256 即 128 optimizer steps**）、planned_total_optimizer_steps=128（全部 bank metadata 实证）。
- 架构：5M = 256×4L×8h/4kv（实测 4,983,296 unique params；legacy manifest 记 4,983,168，差 128 为 legacy 计数口径差，state_dict 完全一致）；26M = 512×7L×8h/4kv（26,681,728 ✓）；100M = 768×13L×12h/6kv（100,784,000 ✓）。
- future stream：g1b 的 (2042,26)/(2043,26)/(2042,64) 复用 F3 frozen streams（stream_id 1/2/3，`derive_stream_seed(20260926)` 协议）；其余 71 个 state 无 F3-protocol stream，按 F4-A prereg 冻结规则新建确定性 stream（同一协议、manifest SHA 落盘）。legacy v07 自带 e1/e2 stream manifests 属 legacy probe 协议（不同构造），本盘如实记录为 available-legacy 但不复用。

## Coverage table（PHASE 2）

| scale | seeds | 每 seed steps | num_states | 覆盖评估 |
|---|---|---|---|---|
| 5M (v07) | 8（2030–2037） | 7（46/52/58/64/70/76/82） | 56 | ✅ 超额（目标 ≥3 seeds × ≥4 点） |
| 26M (G1a) | 3（2042–2044） | 3（26/64/102） | 9 | ⚠️ 每 seed 3 点 < 目标 4 → **MISSING_COVERAGE**（见 requests 文档） |
| 100M (G1b) | 3（2042–2044） | 3（26/64/102） | 9 | ✅ 达标（目标 ≥3 点/seed） |

- progress p = step/128（全部 scale 可恢复同一 planned_total=128）：26→0.203（EARLY）、46/52/58→0.359/0.406/0.453（MID）、64→0.5（MID）、70/76/82→0.547/0.594/0.641（MID/LATE 边界内）、102→0.797（LATE）。
- 注意（FACT）：5M bank 不含 p<0.36 的 state（v07 协议 state 采集从 step46 起）；26M/100M 不含 p∈(0.203,0.5) 与 p>0.797 的中间密度。这些是**覆盖空洞**，不伪造。
- machine-readable：`outputs/f4a_phase_map/bank_inventory.csv`（74 行，含 per-state path/metadata/micro0/tokens_seen）。
