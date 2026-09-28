# E4B-1b v0.8 Stochastic Branching Screen

本报告为 post-confirmatory exploratory mechanism study；不改写 v0.7 confirmatory FAIL。

## Data Integrity

- E4B-1 exact-replay state outputs: `PASS`
- source state files: `56` (expected 56)
- state×H rows: `448` (expected 448)
- candidate: `D_q = sample std_r(R_q), ddof=1`
- formal B/chi source: E4B-1 state JSON 中由 v0.7 raw 转存；`NOT_RECOMPUTED`

## H8 Primary Exploratory Screen

- mean relative MAE improvement: `0.0372401`
- median relative MAE improvement: `0.0798048`
- folds improved: `6/8`
- B×Dq sign stability: `5/8` (negative)
- E4B1B_H8_STATUS: `WEAK`

### Per-held-out-seed

| seed | baseline MAE | +Dq MAE | relative gain | delta interaction |
|---:|---:|---:|---:|---:|
| 2030 | 0.0396407 | 0.0400365 | -0.9985% | 0.00840236 |
| 2031 | 0.0375561 | 0.0338982 | 9.7397% | -0.00179603 |
| 2032 | 0.0338882 | 0.0293931 | 13.2646% | -0.00313713 |
| 2033 | 0.0307621 | 0.0282917 | 8.0305% | 0.00120418 |
| 2034 | 0.0228153 | 0.0216382 | 5.1595% | -0.0038212 |
| 2035 | 0.0249323 | 0.0229551 | 7.9304% | 0.000195235 |
| 2036 | 0.0248509 | 0.0227328 | 8.5231% | -0.00130436 |
| 2037 | 0.0423236 | 0.0515743 | -21.8572% | -0.00100444 |

## Supporting Horizons

| H | status | mean gain | folds improved | interaction sign stability |
|---:|---|---:|---:|---:|
| 4 | UNSUPPORTED | -2.2054% | 4/8 | 8/8 (negative) |
| 12 | WEAK | 3.9619% | 5/8 | 8/8 (positive) |

## Boundary

- [FR-OBSERVATION-ONLY] D_q 只来自已完成的 baseline q-rotation replay。
- [FR-METRIC] candidate selection 只使用 D_q；mean/median/IQR/MAD 仅描述。
- [FR-IO] 本脚本只写 branching/ 新目录。
- v0.7 H8_CONFIRMATION=FAIL 保持不变；本结果不能称为 confirmed。
- 不进入 controller。

## Final

E4B1B_DATA_INTEGRITY = PASS
E4B1B_H8_STATUS = WEAK
E4B1B_H8_MEAN_GAIN = 0.03724010517
E4B1B_H8_FOLDS_IMPROVED = 6/8
E4B1B_H8_DELTA_SIGN_STABILITY = 5/8 (negative)
