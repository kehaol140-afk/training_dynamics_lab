# COST ONLY BENCHMARK — FR_DECISION_UTILITY_V01（§12 成本核算）

本基准只测 GPU 时间，不产出或修改任何科学结果。计时副本的数值输出经与冻结的
`level1_local_derivative.csv` 抽查一致（见文末 VERIFY）。

## 环境

- GPU: NVIDIA GeForce RTX 5090（物理序号 1，device cuda:0）
- PyTorch 2.11.0 / CUDA 13.0
- 独占性: gate_passed（正式计时前无其他计算进程，util 中位数 <10%）
- 计时后 util 59% / 显存 29126 MiB / 其他计算进程 [25242, 25242]
- 目标状态: ['2038/46', '2039/64', '2041/82']，每状态 8 streams（n=28 stream 样本，另有每状态 1 次 warm-up 弃用）
- 总墙钟: 686s；primary lambda = 0.05

## 计时方法

- 每计算组成独立计时键；每区间前后 `torch.cuda.synchronize()` 后读 `time.perf_counter()`；
- 每键记录 total/count/mean/std/median/IQR，**分母一律为实际 tick 数**；
- 分支引擎（真实回放）与切线 pass 使用分离 timers，无共用键污染；
- 主报告口径：**median [IQR]**，同时保留 mean±std（cost_summary.json）；
- 每 horizon 完整 native step = fwd_bwd + unscale + clip + step 四键之和（逐 horizon 成对）。

## 实测组件（median [IQR] 秒）

| 组件 | median | IQR | mean±std | n |
|---|---:|---:|---:|---:|
| t_native_step（完整 native micro-step） | 0.0941757 | 0.0284742 | 0.0928887±0.0167016 | 192 |
| t_cand_grad_one（单候选 g_j） | 0.0498876 | 0.0122682 | 0.0526353±0.0091221 | 96 |
| t_hvp_one（单候选单 horizon HVP） | 0.168164 | 0.0268432 | 0.167486±0.0190387 | 672 |
| t_tangent_B_one（单候选单 horizon B 切线） | 0.24087 | 0.166139 | 0.208287±0.0764186 | 768 |
| t_tangent_C_one（单候选单 horizon C 切线） | 0.241651 | 0.159727 | 0.20559±0.0743873 | 768 |
| t_val_loss_one（endpoint V 双 split loss） | 0.153037 | 0.0663052 | 0.152659±0.0332322 | 72 |
| t_val_grad_one（endpoint V_score grad） | 0.153009 | 0.0395203 | 0.155926±0.0220056 | 72 |
| t_v_t（对齐基线 v_t，每 state 一次） | 0.289947 | — | 1.93483±2.90783 | 3 |
| hhat 投影（每 stream 一次） | 3.11866 | 1.15381 | 2.80168±0.616371 | 24 |
| 回放 wall +λ（单候选） | 1.22686 | 0.30645 | 1.15694±0.153272 | 24 |
| 回放 wall −λ（单候选） | 1.23477 | 0.309006 | 1.21628±0.262832 | 24 |
| state_load（checkpoint 装载，成本公式不含） | 0.0827245 | — | 0.0832998±0.00271702 | 3 |

回放净成本（公式用）: t_replay_net = wall − state_load
- +λ: 1.14414s；−λ: 1.15205s

## 成本公式

```
direct(h,M)    = h·t_native + t_vloss + M·(h·t_native + t_vloss)
direct_fd(h,M) = h·t_native + t_vloss + M·2·(h·t_native + t_vloss)   # ±λ 两次回放
shared_B(h,M)  = h·t_native + t_vloss + t_vgrad + M·(t_cand + h·t_tstep_B)
shared_C(h,M)  = h·t_native + t_vloss + t_vgrad + M·(t_cand + (h−1)·t_hvp + h·t_tstep_C)
```

注：shared_B 无 HVP（B 切线在 horizon>1 用零输入方向）；shared_C 的 HVP 按候选×horizon 计费。
若同一管线同时评估 B 与 C，HVP 计算实际被两者共享（此处按 C 全价、B 零价归属，
对 B 略有高估、对 C 全计，是保守口径）。t_tangent_*_one 均含 clip_jvp+adamw_tangent_step。

## 成本表（M∈{1,2,4} 实测组件内插）

### H=1

| M | 方法 | total s | s/候选 | vs direct speedup |
|---:|---|---:|---:|---:|
| 1 | direct_replay | 0.494426 | 0.494426 | 1 |
| 1 | direct_replay_fd | 0.741639 | 0.741639 | 0.666667 |
| 1 | shared_tangent_B | 0.69098 | 0.69098 | 0.715543 |
| 1 | shared_tangent_C | 0.691761 | 0.691761 | 0.714735 |

| 2 | direct_replay | 0.741639 | 0.370819 | 1 |
| 2 | direct_replay_fd | 1.23606 | 0.618032 | 0.6 |
| 2 | shared_tangent_B | 0.981738 | 0.490869 | 0.755435 |
| 2 | shared_tangent_C | 0.983299 | 0.49165 | 0.754235 |

| 4 | direct_replay | 1.23606 | 0.309016 | 1 |
| 4 | direct_replay_fd | 2.22492 | 0.556229 | 0.555556 |
| 4 | shared_tangent_B | 1.56325 | 0.390813 | 0.7907 |
| 4 | shared_tangent_C | 1.56638 | 0.391594 | 0.789124 |

break-even（shared < direct 的最小 M）: shared_tangent_B: M*=None, shared_tangent_C: M*=None

### H=4

| M | 方法 | total s | s/候选 | vs direct speedup |
|---:|---|---:|---:|---:|
| 1 | direct_replay | 1.05948 | 1.05948 | 1 |
| 1 | direct_replay_fd | 1.58922 | 1.58922 | 0.666667 |
| 1 | shared_tangent_B | 1.69612 | 1.69612 | 0.62465 |
| 1 | shared_tangent_C | 2.20373 | 2.20373 | 0.480766 |

| 2 | direct_replay | 1.58922 | 0.79461 | 1 |
| 2 | direct_replay_fd | 2.6487 | 1.32435 | 0.6 |
| 2 | shared_tangent_B | 2.70949 | 1.35474 | 0.586539 |
| 2 | shared_tangent_C | 3.72472 | 1.86236 | 0.426669 |

| 4 | direct_replay | 2.6487 | 0.662175 | 1 |
| 4 | direct_replay_fd | 4.76766 | 1.19192 | 0.555556 |
| 4 | shared_tangent_B | 4.73622 | 1.18406 | 0.559243 |
| 4 | shared_tangent_C | 6.76669 | 1.69167 | 0.391433 |

break-even（shared < direct 的最小 M）: shared_tangent_B: M*=None, shared_tangent_C: M*=None

### H=8

| M | 方法 | total s | s/候选 | vs direct speedup |
|---:|---|---:|---:|---:|
| 1 | direct_replay | 1.81289 | 1.81289 | 1 |
| 1 | direct_replay_fd | 2.71933 | 2.71933 | 0.666667 |
| 1 | shared_tangent_B | 3.0363 | 3.0363 | 0.597071 |
| 1 | shared_tangent_C | 4.2197 | 4.2197 | 0.429625 |

| 2 | direct_replay | 2.71933 | 1.35966 | 1 |
| 2 | direct_replay_fd | 4.53222 | 2.26611 | 0.6 |
| 2 | shared_tangent_B | 5.01315 | 2.50658 | 0.542439 |
| 2 | shared_tangent_C | 7.37994 | 3.68997 | 0.368476 |

| 4 | direct_replay | 4.53222 | 1.13305 | 1 |
| 4 | direct_replay_fd | 8.15799 | 2.0395 | 0.555556 |
| 4 | shared_tangent_B | 8.96685 | 2.24171 | 0.505441 |
| 4 | shared_tangent_C | 13.7004 | 3.42511 | 0.330808 |

break-even（shared < direct 的最小 M）: shared_tangent_B: M*=None, shared_tangent_C: M*=None

### M∈{8,16,32}（基于实测组件的成本模型外推，非真实运行）

| H | M | 方法 | total s | s/候选 | vs direct speedup |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | shared_tangent_B | 2.72628 | 0.340786 | 0.816098 |
| 1 | 8 | shared_tangent_C | 2.73253 | 0.341566 | 0.814233 |
| 1 | 16 | shared_tangent_B | 5.05235 | 0.315772 | 0.831815 |
| 1 | 16 | shared_tangent_C | 5.06484 | 0.316552 | 0.829764 |
| 1 | 32 | shared_tangent_B | 9.70447 | 0.303265 | 0.840646 |
| 1 | 32 | shared_tangent_C | 9.72946 | 0.304046 | 0.838487 |
| 4 | 8 | shared_tangent_B | 8.7897 | 1.09871 | 0.542415 |
| 4 | 8 | shared_tangent_C | 12.8506 | 1.60633 | 0.371006 |
| 4 | 16 | shared_tangent_B | 16.8966 | 1.05604 | 0.53298 |
| 4 | 16 | shared_tangent_C | 25.0185 | 1.56366 | 0.359957 |
| 4 | 32 | shared_tangent_B | 33.1105 | 1.0347 | 0.527972 |
| 4 | 32 | shared_tangent_C | 49.3542 | 1.54232 | 0.354203 |
| 8 | 8 | shared_tangent_B | 16.8742 | 2.10928 | 0.483458 |
| 8 | 8 | shared_tangent_C | 26.3414 | 3.29268 | 0.309702 |
| 8 | 16 | shared_tangent_B | 32.689 | 2.04307 | 0.471397 |
| 8 | 16 | shared_tangent_C | 51.6234 | 3.22646 | 0.298499 |
| 8 | 32 | shared_tangent_B | 64.3186 | 2.00996 | 0.465069 |
| 8 | 32 | shared_tangent_C | 102.187 | 3.19335 | 0.292724 |

## VERIFY（计时副本 vs 冻结科学 CSV）

- state formal:2038/46 stream 0: max |Δhhat| = 0 （tol 1e-09）→ PASS

## 引用

组件明细: cost_timing_raw.csv；结构化结果: cost_summary.json；
成本表: cost_breakdown.csv。
