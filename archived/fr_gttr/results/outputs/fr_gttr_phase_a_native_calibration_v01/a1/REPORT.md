# Phase A 原生目标定标实验报告（a1）

日期：2026-09-23
协议：FR_GTTR_PhaseA_Native_Target_Calibration_Claude_Code.md v1.0
+ FR_GTTR_Experiment_Roadmap_and_Next_Run_v1.md §3–§8
schedule_id：`finite_cosine_then_hold_5e-5_v01`（label = NATIVE_WITH_DECLARED_TAIL，见 schedule_amendment.md）

本报告只含 Phase A0 + A1。**未进入 B/C/D/E**，未改 optimizer，未训练 controller，未 commit/push。

文中分四类陈述，不可混读：
- **[事实]**：本轮实验直接测量/验证的内容。
- **[恒等式]**：数学上可证明的式子（与数据无关，CPU 测试已验证）。
- **[诊断]**：训练过程中顺带记录的描述性观测量；分子分母为真实数值，
  但只作未来阶段输入，不用于本轮任何结论或动作选择。
- **[假设]**：未来阶段的候选解释，本轮未检验。

---

## 1. 六 cell 真实配置 [事实]

| seed | start(n0) | stream_id | LR 首→尾 | 尾部到达 h | β1/β2 | batch×accum | tokens/step | 完成 |
|---|---|---|---|---|---|---|---|---|
| 2030 | 46 (gm0=92) | 97000 | 3.712e-4 → 5e-5 | h=82 达 floor | 0.9 / 0.999 | 4 行×2 | 1024 slots（931.5 valid 平均） | 256/256 |
| 2030 | 82 (gm0=164) | 97001 | 1.788e-4 → 5e-5 | h=0 已在 floor | 同上 | 同上 | 同上 | 256/256 |
| 2031 | 46 | 97002 | 3.712e-4 → 5e-5 | h=82 | 同上 | 同上 | 同上 | 256/256 |
| 2031 | 82 | 97003 | 1.788e-4 → 5e-5 | h=0 已在 floor | 同上 | 同上 | 同上 | 256/256 |
| 2032 | 46 | 97004 | 3.712e-4 → 5e-5 | h=82 | 同上 | 同上 | 同上 | 256/256 |
| 2032 | 82 | 97005 | 1.788e-4 → 5e-5 | h=0 已在 floor | 同上 | 同上 | 同上 | 256/256 |

- 全配方冻结：原生 AdamW（单参数组，eps=1e-8，wd=0.01，moments/weights fp32，
  bf16 autocast，GradScaler disabled），全局 L2 clip τ=1.0 post-unscale。
- 尾部规则：gm ≤ 256 逐位等于旧 `get_lr(gm,256,5e-4)`；gm > 256 恒 5e-5。
- 每 cell 训练累计 valid loss tokens = 262,144（256 step × 1024 slots，去 padding 后
  实测 mean 931.5 valid/step，全 6 cell 合计 input slots 1,572,864，nonpad 1,443,144，
  valid loss tokens 1,430,856）。
- clip 事件 1177/1536 步（76.6%）；全程 0 非有限值；update norm 首 0.197 → 尾 0.031。

## 2. Monitor NLL（V_monitor = D0 rows[4096:8192)，4096 行全库）[事实]

聚合 = 总 sum / 总 count（禁 mean-of-mean）。6 cell 每点 count 恒 = 475,330
valid tokens。paired 差为同一 cell 同 4096 行的逐行差；SE = 64 行连续 block
paired bootstrap（1000 reps，seed 77070001），**BLOCK_PROXY**（无 doc ID，
不声称 doc 级独立），只反映参考库条件采样不确定性。

### h0 / h64 / h128 / h256 的 sum / count / mean

| seed/start | h=0 sum (mean) | h=64 sum (mean) | h=128 sum (mean) | h=256 sum (mean) |
|---|---|---|---|---|
| 2030/46 | 3464479.7 (7.2886) | 3372549.1 (7.0952) | 3336837.3 (7.0200) | 3283662.8 (6.9082) |
| 2030/82 | 3395778.8 (7.1440) | 3346459.2 (7.0403) | 3317583.9 (6.9795) | 3265279.7 (6.8695) |
| 2031/46 | 3460973.1 (7.2812) | 3366450.4 (7.0823) | 3333419.3 (7.0129) | 3275830.3 (6.8917) |
| 2031/82 | 3392699.0 (7.1376) | 3345657.8 (7.0386) | 3315809.8 (6.9758) | 3258440.7 (6.8551) |
| 2032/46 | 3474745.5 (7.3102) | 3392669.2 (7.1375) | 3359505.9 (7.0677) | 3303598.9 (6.9501) |
| 2032/82 | 3418340.5 (7.1915) | 3371580.4 (7.0931) | 3341696.1 (7.0303) | 3288461.4 (6.9183) |

全部 count = 475330。

### paired 差（前 − 后，正 = NLL 下降）± SE

| seed/start | Δ0→64 | Δ0→128 | Δ0→256 | Δ128→256 |
|---|---|---|---|---|
| 2030/46 | 0.1934 ± 0.0015 | 0.2686 ± 0.0020 | 0.3804 ± 0.0025 | 0.1119 ± 0.0008 |
| 2030/82 | 0.1038 ± 0.0010 | 0.1645 ± 0.0019 | 0.2745 ± 0.0019 | 0.1100 ± 0.0007 |
| 2031/46 | 0.1989 ± 0.0012 | 0.2683 ± 0.0020 | 0.3895 ± 0.0020 | 0.1212 ± 0.0008 |
| 2031/82 | 0.0990 ± 0.0007 | 0.1618 ± 0.0016 | 0.2825 ± 0.0016 | 0.1207 ± 0.0008 |
| 2032/46 | 0.1727 ± 0.0014 | 0.2425 ± 0.0020 | 0.3601 ± 0.0020 | 0.1176 ± 0.0009 |
| 2032/82 | 0.0984 ± 0.0008 | 0.1612 ± 0.0019 | 0.2732 ± 0.0019 | 0.1120 ± 0.0008 |

逐 cell 训练墙钟 101–137s；6 cell 合计 main 649s。

## 3. 目标候选 ell [事实]

冻结规则（结果前定）：每起点组 ell = 3 seeds 的 h128 monitor mean 的中位数。
余量判据：V0−ell > 2·SE0 且 ell−V256 > 2·SE256（逐 seed）。
hit = 两个连续 monitor 点 ≤ ell（在第二点记录）。
全部 hit 仅 CALIBRATION_ONLY；final 模型评价 final_call_count = 0。

| start | ell 候选 | 组成 seeds h128 | 逐 seed V0−ell | 逐 seed ell−V256 | SE0 / SE256 | 噪声余量 | hit (h) | censored | group bracketed |
|---|---|---|---|---|---|---|---|---|---|
| 46 | **7.0200** | 7.0200 / 7.0129 / 7.0677 | +0.2686 / +0.2612 / +0.2902 | +0.1118 / +0.1212 / +0.1176 | ≈0.0050 / ≈0.0050 | 6/6 PASS | 160, 160, 224 | 0/6 | True |
| 82 | **6.9795** | 6.9795 / 6.9758 / 7.0303 | +0.1645 / +0.1581 / +0.2120 | +0.1100 / +0.1207 / +0.1120 | ≈0.0050 / ≈0.0050 | 6/6 PASS | 160, 160, 224 | 0/6 | True |

（hit 按 (2030, 2031, 2032) 顺序；start=46 的 hit 全为 h=160，start=82 的
2030/2031 hit h=160，2032 hit h=224。）

结论 [事实]：两个起点组的内部目标均被 bracket（全部 3 seed V0>ell，
全部 3 seed V256<ell），噪声余量 6/6 通过，无 censored，无需
TARGET_UNRESOLVED。所有 hit 只登记定标事件，**不宣称任何成本改善**。

## 4. 分层诊断（描述性，分子分母为真实数值）[诊断]

诊断点在 h=1 及每 16 步（h=16..256），全程不改变训练状态/RNG/数据位置
（smoke 已证 bitwise 轨迹恒等）。仅列 h=256 各 cell 相对更新总量 top-3，
全表见 layer_diagnostics.csv。r_total = ‖s‖/‖W‖（相对位移），
b_rad = radial 分量，r_tangent = tangent 分量 [恒等式：r_total² = b_rad² + r_tangent²]。

| cell | param | r_total | b_rad |
|---|---|---|---|
| 2030/46 | model.layers.3.self_attn.q_proj.weight | 8.9e-4 | （见 CSV） |
| 2030/46 | model.layers.3.self_attn.k_proj.weight | 8.8e-4 | |
| 2030/46 | model.layers.2.self_attn.k_proj.weight | 8.3e-4 | |
| 2032/82 | model.layers.2.self_attn.k_proj.weight | 8.1e-4 | |
| 2032/82 | model.layers.2.self_attn.q_proj.weight | 7.6e-4 | |
| 2032/82 | model.layers.1.self_attn.q_proj.weight | 7.4e-4 | |

- 共同观察 [诊断]：top 相对更新集中于最后层 self_attn q/k 投影，量级 ~1e-3/步；
  embedding/lm_head（tied）与 mlp 层相对位移小 2–5 倍。
- I_m / I_v 的范数已按 INNOVATION_PROXY 登记（真实分子分母，见 CSV），
  **不作为真实 tracking error**；新增诊断 gradient=0、HVP=0 未触发任何异常。
- 未计算：attention entropy、全 Hessian、任何新噪声张量。

## 5. 资源对账 [事实]

| 项 | planned | actual | cap |
|---|---|---|---|
| optimizer updates | 1552 (main 1536 + smoke 16) | 1552 | 1664 |
| monitor/复核 forward | 7296 | 7296 | 7800 |
| 训练+评价总 forward | 10400 | 10400 | 12000 |
| 其中训练 forward | 3104 | 3104 | — |
| monitor eval 次数 | 57 | 57 | — |
| snapshots (h128/h256 × 6 cell) | 12 | 12 | — |
| 失败恢复 | 0 | 0 | — |
| GPU 墙钟 | — | 708s (smoke 59 + main 649) | 5400s |
| 新增 gradient calls | 0 | 0 | — |
| 新增 HVP calls | 0 | 0 | — |
| V_final eval calls | 0 | 0 | — |

actual = planned 逐位相等；0 失败、0 恢复、0 重试。smoke 先于 main，
smoke 中 h=0 的 monitor 读数与 main 复算逐行差 < 1e-6（同初态独立重算一致）。
历史 18,730 段及 C1/C1L 账未动；本轮 actual 单独记于 resource_log.jsonl。

## 6. 下一阶段缺什么 [事实清单，未启动]

按 Roadmap，Phase B（动作价值预读模型 + paired action branches）开工前
尚缺以下输入，本轮一律未做：

1. **action-value 预读模型**：需基于 GTTR v0.2 冻结理论形式化 φ(s) 与
   预读 horizon；本轮无任何 forward-model 训练。
2. **paired action branches**：需要成对分支控制器结构 + 固定 baseline
   的调参预算上限（本轮预算全耗于定标，未留 controller 预算）。
3. **独立确认 seed/stream**：定标 ell 来自 2030–2032 × 46/82 本身；
   独立确认集（新 seed、新 stream_id 段）未生成。
4. **doc 级独立性**：D0 无 doc ID，DOCUMENT_INDEPENDENCE_UNVERIFIED 保持。
5. **t0 / 长 horizon 监测**：本轮 horizon 固定 256 步；h128/h256 snapshot
   （12 个 full_state v0.2 可恢复状态）已存，供后续续训/回放。

**本轮到此停止。** 不进入 Phase B，不训练控制器，不 commit/push。

---

### 附：产物清单

- `config.json` `preflight.md` `schedule_amendment.md` `dataset_contract.json`
- `schedule.csv` `ema_timescales.csv`（h_half(0.9)=6.5788 步，h_half(0.999)=692.8 步）[恒等式]
- `theory_tests.json`（17 项 CPU 测试全 OK）`smoke_checks.json`（PASS）
- `eta_estimate.json` `resource_log.jsonl`
- `monitor_row_stats.csv`（逐行 sum/count，6 cell × 9 monitor 点 × 4096 行）
- `step_metrics.csv` `layer_diagnostics.csv`
- `trajectory_metrics.csv` `target_calibration.csv` `analysis_summary.json`
- `snapshots/`（12 个 full_state v0.2）+ `SNAPSHOT_MANIFEST.json`
- `data/v_monitor_rows4096_8192.jsonl`
