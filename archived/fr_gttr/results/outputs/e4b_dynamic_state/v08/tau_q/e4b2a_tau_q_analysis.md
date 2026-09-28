# E4B-2A v0.8 q-Dynamics Low-dimensional Timescale

本报告为 post-confirmatory exploratory mechanism study；不改写 v0.7 confirmatory FAIL。

## Data Integrity

- source formal q-rotation state files: `56`
- fitted seed×state curves: `56` (expected 56)
- formal B/chi: `NOT_RECOMPUTED`
- GPU replay: `NO`

## Frozen Fit

`Rbar_q(H) = c + (1-c) * exp(-H/tau_q)`

- tau range: `[0.1, 512.0]` optimizer steps
- coarse log grid: `4096`
- refine log grid: `2048` × `3` rounds
- scipy: `NOT_REQUIRED`
- deterministic numpy bounded fit: `YES`

## Fit Quality

- E4B2A_TIMESCALE_FIT_STATUS: `MIXED`
- median R2: `0.885334`
- fit success fraction: `100.0000%`
- any-boundary-hit fraction: `0.0000%`
- tau_q median: `4.77244`
- tau_q IQR: `[3.26795, 6.64703]`
- tau_q range: `[1.33842, 13.6171]`
- c median: `0.58978`
- c IQR: `[0.53544, 0.667031]`
- c range: `[0.253247, 0.738555]`

### tau_q variation

- total state-level std: `2.7218`
- std of seed means: `1.34518`
- std of state-step means: `1.57404`

### Worst 5 fits

| seed | state | tau_q | c | R2 | RMSE | boundary |
|---:|---:|---:|---:|---:|---:|---|
| 2037 | 76 | 2.00759 | 0.695403 | 0.516246 | 0.084528 | tau=False, c=False |
| 2035 | 70 | 1.33842 | 0.709398 | 0.630513 | 0.0337497 | tau=False, c=False |
| 2036 | 70 | 2.36854 | 0.707923 | 0.663325 | 0.0496757 | tau=False, c=False |
| 2033 | 76 | 2.00175 | 0.709963 | 0.705486 | 0.0487236 | tau=False, c=False |
| 2030 | 64 | 2.31545 | 0.573069 | 0.718442 | 0.0788093 | tau=False, c=False |

## H8 Single-state Screens

| candidate | status | mean gain | median gain | folds improved | interaction sign stability |
|---|---|---:|---:|---:|---:|
| tau_q | UNSUPPORTED | -15.1851% | -16.1357% | 0/8 | 8/8 (negative) |
| c | UNSUPPORTED | -2.1222% | -2.2243% | 2/8 | 7/8 (negative) |

## Curve-collapse Diagnostic

主要统计不使用 chi/B；而是在 raw-H 组或固定 H/tau_q bin 内拟合 `chi ~ seed_FE + state_fraction + B`，比较 pooled residual RMSE。

- raw-H pooled RMSE: `0.0311427`
- H/tau_q pooled RMSE: `0.0425272`
- relative RMSE improvement: `-36.5556%`
- normalized used fraction: `99.3304%`
- E4B2A_CURVE_COLLAPSE: `UNSUPPORTED`

### Fixed normalized-horizon bins

| bin | n | used | RMSE |
|---:|---:|---|---:|
| 0 | 43 | True | 0.0169732 |
| 1 | 54 | True | 0.0298559 |
| 2 | 57 | True | 0.0462756 |
| 3 | 74 | True | 0.0562916 |
| 4 | 98 | True | 0.0452171 |
| 5 | 86 | True | 0.0370888 |
| 6 | 33 | True | 0.0457492 |
| 7 | 3 | False | NA |

## Boundary

- [FR-OBSERVATION-ONLY] tau_q/c 只由 future baseline q dynamics 拟合。
- [FR-METRIC] H8 tau_q 与 c 分开筛选，不做 tau_q+c 或其它多变量组合。
- [FR-METRIC] curve-collapse 是描述性诊断，不是新的 confirmation。
- [FR-IO] 只写 tau_q/ 新目录。
- 不进入 controller，不重写 v0.7。

## Final

E4B2A_DATA_INTEGRITY = PASS
E4B2A_TIMESCALE_FIT_STATUS = MIXED
E4B2A_MEDIAN_TAU_Q = 4.772444883
E4B2A_TAU_Q_IQR = [3.267949359, 6.647026164]
E4B2A_MEDIAN_R2 = 0.8853339414
E4B2A_BOUNDARY_HIT_FRACTION = 0
E4B2A_TAU_Q_H8_STATUS = UNSUPPORTED
E4B2A_C_H8_STATUS = UNSUPPORTED
E4B2A_CURVE_COLLAPSE = UNSUPPORTED
E4B2A_PLOT = SKIPPED_MATPLOTLIB_UNAVAILABLE
