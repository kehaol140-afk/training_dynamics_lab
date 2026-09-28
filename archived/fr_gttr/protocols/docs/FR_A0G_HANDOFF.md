# FR_A0G 决策效用实验 — 跨会话交接冻结

> 本文档给「下一会话」的 Claude / 操作者一键恢复完整上下文。所有数字取自冻结制品
> （`outputs/fr_decision_utility_v01/summary.json` 与 `statistical_audit/`），不要凭记忆改写。
> 生成时间：2026-09-14 11:50（最后由冻结会话写入）。

---

## A. 实验目的（一句话）

回答：**完整切线响应（Model C）能否帮我们在真实训练里选中一个"真能降低未来 validation loss"的数据加权动作？** 以及这份收益是否抵得上它相对"逐候选真实回放"的额外计算成本。
这是把 A0C–A0F2 已冻结的「响应理论」落到「训练决策效用 + 成本」上的收官实验。

## B. 当前进度（精确到哪一步）

- **科学主实验（GPU 计算）已 100% 完成**：formal 设计 12 states × 8 streams = 96 个 stream 单元，
  全部以 `shadow_pass=True` 跑完，**四份原始科学 CSV 已固化、未被触碰**。
- **Level-1 / Level-1b / Level-3 聚合 + 分层统计审计已完成**（纯 CPU 离线后处理，
  因为主进程在聚合阶段因一个 `for (split, h), _ in ...` 解包 bug 崩溃——见 D 节，已修复并用离线脚本补齐）。
- **成本计时补测（cost-only benchmark）已完成**（2026-09-14）。它只测时间、不产生新科学数据，
  不影响任何已冻结的科学结论。计时副本经 VERIFY（vs 冻结 level1 CSV max|Δhhat|=0）。
- `a0g_finalize_report_v01.py` 已装配：`summary.json` status=**COMPLETE**，报告与顶层 SHA256SUMS 已重建。
- **最终判定：决策效用（A）成立；墙钟成本优势（B）不成立**（详见 G/H 节成本结论）。

## C. 已冻结协议（绝不可改动）

1. **动作定义**：候选数据组 j 的动作是 `g_action = g + λ·(g_j − g)`，施加在
   post-unscale / pre-clip 位置，做**单次 optimizer step**，随后 baseline 分支照常继续
   （tangent 预测 `r_H = dθ_{t+H}/dλ` 每单位 λ；有限预测 `ΔL̂ = λ·ĥ`，`ĥ = hhat`）。
2. **V_score / V_eval 分离**：调 λ、选 H 用 V_score（dev）；一切最终结论只在 V_eval（formal，
   新 seeds 2038–2041）上报告。
3. **formal 设计**：seeds **2038–2041** × steps **{46, 64, 82}** × **8 streams** = 12 states / 96 单元。
4. **λ 已冻结**：primary **0.05**、conservative **0.025**。
   选取规则 = **Rule B_decision_signal**（clip_crossing ≤ 0.10）。
   **重要**：Rule A（严格局部 slope_dev ≤ 0.25）在全部 dev 候选上 **FAILED**——
   这是一个被保留的**负结果**，**永远不得改写成"严格有限振幅线性成立"**。
5. **readout off-by-one 修复**（此前会话完成）：baseline val / v_H / r_H / branch readout
   统一在 `θ_{t+H}` post-step 处读，H 步预测对齐。

## D. 已修复的 bug（绝不可复现 / 绝不可再犯）

- **`_aggregate_level3` 字符串解包 bug**：原代码
  `for (split, h), _ in sorted({(k[0], k[1]) for k in blocks})` 会把 split 字符串
  `"formal"` 逐字符解包，触发 `ValueError`。已修为
  `for split, h in sorted({(k[0], k[1]) for k in blocks}):`，并新增单测
  `TestAggregateLevel3`（覆盖 splits="dev"/"formal" 触发路径）。26/26 测试通过。
- 该 bug 只影响**汇总展示**，不影响四份原始 CSV；离线后处理脚本
  `research/scripts/a0g_postprocess_v01.py` 已用修复后的聚合函数补齐全部聚合。
- **计时副本 `SampleTimers` 缺 `snapshot_peak`**（本会话发现并修复）：计时脚本里的
  简化 `SampleTimers` 只实现了 `start/tick`，而 runner 的 `_run_action_branch`/`_run_tangent_pass`
  会调 `timers.snapshot_peak("peak_pass")`（记录显存峰值，与计时无关），导致
  `AttributeError: 'SampleTimers' object has no attribute 'snapshot_peak'`。已在
  `a0g_cost_timing_v01.py` 的 `SampleTimers` 补上同签名 `snapshot_peak`（顺带保留 peak 供核查）。
  这是**计时-only** 修复，未触碰 runner/科学代码。
- **`summary` 作用域 bug**（本会话第二次发现并修复）：计时全部完成后、写出产物前，
  原代码在 L872 提前执行 `summary["gpu_meta"] = gpu_meta`，但 `summary` 要到 L887 才赋值，
  触发 `UnboundLocalError: cannot access local variable 'summary'`——**导致 cost_timing_raw.csv /
  cost_breakdown.csv / cost_summary.json / COST_ONLY_BENCHMARK.md / SHA256SUMS.txt 全部未写出**
  （计时数据当时只在内存）。已删除该提前赋值行（gpu_meta 本就由下方 summary 字典的
  `"gpu_meta": gpu_meta` 键统一入册）。属计时-only 修复，未触碰 runner/科学代码；
  修复后 waiter 正常进入门控。**教训：产物写出前的任何汇总赋值都要确认变量已定义。**

## E. 绝不可重跑 / 绝不可触碰的数据

- **四份原始科学 CSV**（`outputs/fr_decision_utility_v01/` 下，
  level1 / level1b / level2 / level3 原始行）：**只读，永不编辑、永不重跑、永不重生成**。
- A0C–A0F2 全部冻结制品。
- 已冻结的 λ、formal split、候选数据组、正式 science result。

## F. 关键科学数字（formal，V_eval）

**Level-1（formal H=4，n=384）——C 的切线预测 vs 中心差分：**
- Pearson **0.905**，符号准确率 **0.880**，MAE 1.14e-3（优于 B 的 0.660/0.779、alignment 的 0.102/0.701）。

**Level-3 任务层（formal H=4，stream n=96，越高越好的列=Gain/top1/spearman/direction，越低越好=regret/harmful/MAE）：**

| 方法 | mean Gain | mean Regret | harmful | top-1 | spearman | direction |
|---|---|---|---|---|---|---|
| random | 9.22e-5 | 1.65e-4 | 0.229 | 0.208 | — | 0.594 |
| alignment | 8.05e-5 | 1.77e-4 | 0.323 | 0.250 | 0.028 | 0.677 |
| B（仅优化器切线） | 1.31e-4 | 1.26e-4 | 0.188 | 0.396 | 0.281 | 0.813 |
| **C（完整切线）** | **1.54e-4** | **1.04e-4** | **0.177** | **0.448** | **0.366** | **0.823** |

**分层 bootstrap（10000 次，seed→state→stream；delta = C − baseline）：**
- **H=4**：C vs random / alignment 的 Gain、Regret 的 **P(有利) ≥ 0.96**（方向一致）；
  C vs B 的 Gain P(>0)=0.84、Regret P(<0)=0.85，**CI 跨 0**。
- **H=8**：C vs B 的 Gain **CI=[1.62e-5, 1.38e-4] 全正、P(>0)=0.995**（稳健）；Regret P(<0)=0.996。

**措辞红线（必须遵守）：**
- ✅ 允许：「完整切线响应的任务层训练决策效用已在独立评价数据上得到支持。」
- ✅ H=8 可称 C 稳健优于 B；H=4 只能说 C 相对 B **方向一致但不显著**。
- ❌ 禁止：「完整切线已经带来训练效率收益。」（成本结论未定，见 G/H）。

## G. 成本计时补测设计（已完成，结论见 H 节）

- **目的**：量化「shared-tangent 一次传播预测 M 个候选」相对「逐个真实回放 M 个候选」
  的 wall-clock 成本，算 speedup 与 break-even M*。
- **范围**：3 个 formal 状态 **(2038,46) (2039,64) (2041,82)** × 8 streams；纯计时，无新科学数据。
- **方法**：独占 GPU（门：util 中位 <10% 且无其他计算 PID，绝不杀外部任务）；
  每状态 1 次 warm-up（弃用）+ ≥3 次独立样本；主指标 median + IQR（保留 mean±std）；
  `cuda.synchronize` 夹 `perf_counter`；8 个组件独立计时键；分支引擎用一次性计时器隔离。
- **成本公式**：
  - `direct(h,M) = h·t_native + t_vloss + M·(h·t_native + t_vloss)`（每候选一次完整 h 步回放）
  - `shared_B(h,M) = base + M·(t_cand + h·t_tstep_B)`（无 HVP）
  - `shared_C(h,M) = base + M·(t_cand + (h−1)·t_hvp + h·t_tstep_C)`
  - 比较 M ∈ {1,2,4}，外推 M ∈ {8,16,32}（须标注"基于实测组件的成本模型外推"）；break-even M*∈1..64。

## H. 成本补测结论（已完成，2026-09-14）

- **历史**：前一 waiter 曾等独占 GPU ~2.3 小时；**WSL2 实例重启**（dmesg 见
  `init-systemd(Ubuntu-Recovered)` / `init failed`）把它杀死；后续又修掉两个计时副本 bug
  （`SampleTimers.snapshot_peak` 缺失、`summary` 作用域，见 D 节），最终 waiter **PID 22799**
  在独占 GPU（`GATE_PASS util 8%`）上跑完全部 3 态 × 8 streams 计时。
- **计时产物**（`cost_only/`，SHA256 已核）：`cost_timing_raw.csv`、`cost_breakdown.csv`、
  `cost_summary.json`、`COST_ONLY_BENCHMARK.md`、`SHA256SUMS.txt`；VERIFY（计时副本 vs 冻结
  level1 CSV）**PASS**（max|Δhhat|=0，tol 1e-09）。

**实测组件（median s，IQR）**：t_native_step 0.0942[0.028]、t_cand_grad 0.0499[0.012]、
t_hvp 0.168[0.027]、t_tangent_B 0.241[0.166]、t_tangent_C 0.242[0.160]、t_val_loss 0.153、
t_val_grad 0.153、回放 wall ±λ ~1.23[0.31]、state_load 0.083（公式不含）。

**成本表核心（H=4 M=4，total s / vs direct speedup）**：direct_replay 2.649（=1）、
direct_replay_fd 4.768（0.56）、shared_tangent_B 4.736（0.56）、**shared_tangent_C 6.767（0.39）**。
H∈{1,4,8} × M∈{1,2,4} 的 shared_tangent_B/C 的 speedup **全部 <1**；break-even 扫描 M≤64
对 shared_tangent_B/C **全为 None**（无 break-even）。外推 M∈{8,16,32} 亦无一 >1。

**成本结论（准则 5）**：在本模型规模（MiniMind 类）与计时口径下，**完整切线的共享传播相对逐候选
真实回放无墙钟成本优势**——尽管它只需一次 baseline 回放 + 每候选一次切线步进，但切线步进
（含 HVP）的常数开销高于直接回放。决策效用成立**不依赖**此成本结论。

## I. GPU 空闲后的自动链（已于 2026-09-14 全部执行完毕）

1. **计时**：waiter 22799 在独占 GPU 上跑完 3 态 × 8 streams → 产出 `cost_only/`（VERIFY PASS）。✅
2. **装配**：`python research/scripts/a0g_finalize_report_v01.py` 合并 cost_summary + bootstrap
   → `summary.json` status=**COMPLETE**，重渲染 `FR_DECISION_UTILITY_REPORT.md`，重建顶层
   `SHA256SUMS.txt`。✅
3. **回填文档**：成本结论已写入本文件 H 节、FR_PROJECT_STATUS.md §17、FR_EXPERIMENT_INDEX.md A0G 行。✅
4. **最终判定（已落地到 `summary.json.final_verdict`）**：准则 1–4 成立（决策效用）；
   准则 5（相对逐候选回放的墙钟效率）**不成立**——speedup 恒 <1、无 break-even。
   措辞红线更新为：可陈述决策效用成立 + 成本否定结果；仍**禁止**「完整切线已经带来训练效率收益」。

## J. 运行环境

conda env `fr_minimind`（`/home/b205/miniforge3/envs/fr_minimind/bin/python`）、
`CUDA_VISIBLE_DEVICES=1`（物理 GPU1，脚本内 `--device cuda:0`）、`PYTHONPATH=research/src`、
torch 2.11.0 / RTX 5090 ×2 / bf16。计时基准必须在**独占** GPU 上做，不得与外部任务共享。

## K. 绝对禁止（红线，任何时候不得违反）

- 重新运行 A0G 科学主实验（四份原始 CSV 已完备）。
- 重新调 λ / 改变 primary λ=0.05、conservative λ=0.025。
- 改变 formal split / 候选数据组 / 正式 science result。
- 编辑/重生成四份原始科学 CSV。
- 覆盖 A0C–A0F2 任何冻结结果。
- kill 任何外部 GPU 任务；删除任何 checkpoint；`git reset` / `git clean`；清理 outputs。

---

## 下一会话第一条指令（原文照抄，逐字执行）

```
A0G 已全部完成（summary.json status=COMPLETE），无需再跑任何实验或装配。先读三份文档恢复上下文：
docs/FR_A0G_HANDOFF.md、docs/FR_PROJECT_STATUS.md（§17）、docs/FR_EXPERIMENT_INDEX.md（A0G 行）。
最终结论：完整切线响应的任务层训练决策效用已在独立评价数据（V_eval，seeds 2038–2041）上得到支持
（Level-1 H=4 Pearson 0.905；Level-3 H=4 mean Gain 1.54e-4、Regret 1.04e-4、top-1 0.448；
分层 bootstrap H=8 C vs B 稳健 P=0.995、H=4 C vs B 方向一致 CI 跨 0）；但其相对逐候选真实回放的
墙钟成本优势不成立（cost-only 计时：H=4 M=4 shared-C 6.77s vs direct 2.65s，speedup 0.39，
break-even 全 None）。下一步是 v06 跨 partition 复现（需新建 E2/E3 级观测协议，非 replay 扩展）。
全程禁止：重跑 A0G 科学主实验、重调 lambda、改 formal split、改四份原始科学 CSV、kill 外部 GPU 任务、git reset/clean。
```
