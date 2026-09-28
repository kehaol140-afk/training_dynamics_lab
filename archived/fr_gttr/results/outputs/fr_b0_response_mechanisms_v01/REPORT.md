# FR B0 响应机制验证报告 (v01)

生成: `research/scripts/b0_analyze_v01.py` | 输入: `b0_run_v01.py` 原始表 | 判定规则: `config.json` (任何结果之前冻结)

**统计标签**: pilot/exploratory - seed (n=3) 是唯一独立簇; streams/timepoints 不是独立 seeds。u*/V_grid 类量为 hindsight/oracle 标签。

## 0. 运行命令、输出路径、资源计数

```
CUDA_VISIBLE_DEVICES=<picked> PYTHONPATH=research/src python research/scripts/b0_run_v01.py --mode smoke|main|stress && python research/scripts/b0_analyze_v01.py
```
- 原始表: `outputs/fr_b0_response_mechanisms_v01/runs/` (main_cells.jsonl 216 行, stress_cells.jsonl 6 行, main_diag.csv, stress_diag.csv, main_decomposition.csv, smoke_checks.json, gpu_budget_log.json, smoke_checks_attempt1.json)
- 分析产物: `outputs/fr_b0_response_mechanisms_v01/` (cell_metrics.csv, response_curvature.csv, decomposition.csv, release_vs_persistent.csv, summary.json, figures/f1-f6)

- **资源计数**: optimizer updates = 14208 / cap 18000 (78.9%); eval_calls = 1656; theta_captures = 666; theta_bar_evals = 162 (开销另计, 无隐藏)

## 1. 实现正确性: PASS

- Smoke 15 项数值门 ALL_PASS = True (snapshot/CRN/冻结 runner 逐位一致/nested readout/pulse==persistent@H4 losses+theta+m_v/release 语义/负 u/1024 tokens per step/128 micro/eval RNG 隔离)
- 主样本内独立复核: pulse==persistent @H=4 违例 = 0 (216 条 rollout 全量)
- 训练语义: 1 optimizer step = 2 microbatch (accum=2) = 8 seqs x 128 tokens = **1024 tokens/step**; 每 micro-step 由 get_lr 重写 native LR, 干预 multiplier = exp(u) 作用于该 step 的 consumed lr; pulse4_release 在第 4 步后恢复 native LR, 保留 theta/m/v (smoke 验证 m/v 从自身 H=4 状态连续演化且与 persistent 臂分叉)

## 2. 假设一 (干预时长): 持续 vs 前 4 步脉冲后释放 - **SUPPORTED (方向保持, 幅度约 13%)**

delta = L(pulse4_release) - L(persistent), 同 cell 同 stream 配对 (v_eval; 判据 |mean|>SEM 且 >1e-7, pilot):

| H | supported/总组合 | max abs mean delta |
|---|---|---|
| 4 | 0/24 | 0.000e+00 |
| 16 | 23/24 | 4.995e-03 |
| 64 | 24/24 | 5.130e-03 |

- H=4: delta 恒为 0 (bitwise, 与 smoke 一致)
- H=64: 脉冲臂保留的持续响应份额中位数 = 0.098 ((L_pulse-L_nat)/(L_pers-L_nat); 越小释放抹除越多)
- 方向: release delta 符号随 cell 的 chi 符号走 (释放抹掉 persistent 增益, 不产生新方向); u=+0.05 处脉冲增益约 0.15x, 负 u 同向对称

## 3. 假设二 (中心 vs 分散): **SUPPORTED - 长程由中心轨迹承载**

| H | |resp_ens| | |resp_center| | |resp_disp| | disp/(ctr+disp) | resp_disp<0 比例 |
|---|---|---|---|---|---|
| 4 | 2.019e-04 | 3.615e-04 | 2.651e-04 | 0.423 | 0.46 |
| 16 | 1.094e-03 | 8.621e-04 | 7.280e-04 | 0.458 | 0.52 |
| 64 | 1.963e-03 | 2.133e-03 | 2.614e-04 | 0.109 | 0.40 |

- J_disp >= 0 全程成立 (违例 0) - 凸性 sanity 通过; resp_disp<0 表示干预**降低**流间分散
- 解读: H=64 时 ensemble 响应约 89% 由中心 (流间平均参数轨迹) 承载, 分散贡献小且常为负; H=4 时分散约占一半 - 短程响应部分来自分散结构变化

## 4. 假设三 (二阶曲率 & 最优幅度): **Q = UNRESOLVED; 无内点极小证据**

| H | chi mean | +/− cells | Q identifiable |
|---|---|---|---|
| 4 | +5.550e-03 | 2/4 | 2/6 |
| 16 | -1.299e-02 | 1/5 | 0/6 |
| 64 | -9.291e-02 | 0/6 | 0/6 |

- 分类计数 (36 行 = 6 cells x 2 protocols x 3 H): {'INTERIOR_MINIMUM': 4, 'BOUNDARY_SATURATED': 16, 'FLAT_OR_NEGATIVE_RESPONSE': 16}
- 内点极小: s2030 t46 persistent H=4 u*=+0.0016 (chi=-5.82e-03, Q=+3.65e+00) [hindsight]
- 内点极小: s2030 t46 pulse4_release H=4 u*=+0.0016 (chi=-5.82e-03, Q=+3.65e+00) [hindsight]
- 内点极小: s2032 t46 persistent H=4 u*=+0.0013 (chi=-1.10e-02, Q=+8.68e+00) [hindsight]
- 内点极小: s2032 t46 pulse4_release H=4 u*=+0.0013 (chi=-1.10e-02, Q=+8.68e+00) [hindsight]
- H>=16: Q 全部 UNRESOLVED (dev/audit 分歧尺度 > |Q|; 内点外推误差 ~1e-2 与响应同量级)
- 响应方向随 H 翻转: H=4 混合 (2/6 偏 LR-down), H=64 全部 6 cells chi<0 (LR-up 单调有利), best-on-grid = +0.05 边界 - 与 A0O/A0T Case B 兼容 (方向依赖 horizon); 但二阶最优幅度本身不可辨识

## 5. 局部准平稳性: NOT_TESTED

- 本轮未做条件平稳性闭合; 平滑曲线不构成证据

## 6. 架构结论: NOT_TESTED

- 单一 4.98M minimind v0.7 配置; 无 26M/100M, 无架构搜索

## 7. 能量/节能: NOT_ESTABLISHED

- 未测功耗, 未做长训练

## 8. Stress cases (A0T 反转 cell 独立复核, stream 7110, v_score)

- s2030_t46:
  - native_+0.000: H4=7.270499 H16=7.206216 H64=7.068539
  - persistent_+0.050: H4=7.271886 H16=7.205255 H64=7.061111
  - persistent_-0.050: H4=7.26925 H16=7.208082 H64=7.078991
- s2030_t64:
  - native_+0.000: H4=7.183027 H16=7.137565 H64=7.047541
  - persistent_+0.050: H4=7.183277 H16=7.136212 H64=7.043077
  - persistent_-0.050: H4=7.182797 H16=7.13873 H64=7.051981
- 两 cell 在 H=64 均为 +0.05 降低损失 (7.061<7.069; 7.043<7.048), 与主样本 H=64 方向一致; H=4 差异微小

## 9. 未检验的物理假设

- 流间独立性 (4 streams/seed 仅 pilot)
- 更长 horizon 的脉冲核 (A0P 已 NEGATIVE; 本轮仅 64 步)
- 非二次结构 (仅 5 点网格, Q 已 UNRESOLVED)
- 条件准平稳性 (B1 前置): NOT_TESTED

## 10. 是否进入 B1

- 假设一 SUPPORTED (释放 != 持续, 幅度约 13%, 方向保持)
- 假设二 SUPPORTED (H=64 中心承载约 89%)
- 假设三 UNRESOLVED (Q 不可辨识; 仅单调/边界饱和结构可辨)
- 判定: 可进入 B1, 但动机应建立在'一阶梯度响应 + 边界饱和' (Case B) 上, 而非二阶曲率; 若 B1 需要二阶信息, 需先解决 Q 的 stream-out 稳定性。B0 本身不授权 B1 执行, 仅给出兼容性结论
