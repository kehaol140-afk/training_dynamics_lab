# E4B-1 v0.8 q-Direction Rotation

本报告为 exploratory mechanism study；不改写 v0.7 confirmatory FAIL。

## Replay Integrity

- seeds: `8`
- states/seed: `7`
- streams/state: `8`
- total baseline trajectories: `448`
- all formal state outputs exact replay: `PASS`

## Horizon Curve

| H | mean Rq | state-to-state std | mean stream dispersion | conditional LOSO gain | folds improved | delta sign stability |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.957512 | 0.046763 | 0.022654 | -1.5546% | 4/8 | 7/8 (negative) |
| 2 | 0.880391 | 0.086169 | 0.052682 | -11.0777% | 4/8 | 6/8 (negative) |
| 4 | 0.733309 | 0.126984 | 0.095569 | -11.1829% | 2/8 | 6/8 (negative) |
| 8 | 0.649639 | 0.115010 | 0.104035 | -0.1660% | 3/8 | 8/8 (negative) |
| 12 | 0.651520 | 0.093362 | 0.089309 | -0.3080% | 4/8 | 5/8 (positive) |
| 16 | 0.634841 | 0.090554 | 0.085809 | -0.6565% | 3/8 | 8/8 (negative) |
| 24 | 0.611016 | 0.092874 | 0.086363 | -5.7879% | 4/8 | 7/8 (positive) |
| 32 | 0.591016 | 0.101493 | 0.072402 | -3.1401% | 3/8 | 5/8 (negative) |

## H8 Primary Exploratory Mechanism Screen

- mean relative MAE improvement: `-0.00165976`
- median relative MAE improvement: `-0.00451315`
- folds improved: `3/8`
- B×Rq interaction sign stability: `8/8` (negative)
- E4B1_H8_CONDITIONAL_STATUS: `UNSUPPORTED`

### H8 per-held-out-seed

| seed | baseline MAE | +Rq MAE | relative gain | delta interaction |
|---:|---:|---:|---:|---:|
| 2030 | 0.0396407 | 0.0360445 | 9.0721% | -0.0128475 |
| 2031 | 0.0375561 | 0.0377987 | -0.6461% | -0.0113211 |
| 2032 | 0.0338882 | 0.0339751 | -0.2565% | -0.0124357 |
| 2033 | 0.0307621 | 0.0216409 | 29.6506% | -0.0106478 |
| 2034 | 0.0228153 | 0.0268034 | -17.4799% | -0.0120723 |
| 2035 | 0.0249323 | 0.0253637 | -1.7302% | -0.0111895 |
| 2036 | 0.0248509 | 0.0233268 | 6.1328% | -0.015126 |
| 2037 | 0.0423236 | 0.0533576 | -26.0706% | -0.0124543 |

## Boundary

- [FR-OBSERVATION-ONLY] Rq 只由 baseline replay + fixed probe 测量。
- [FR-REPLAY-CRITICAL] 每条 trajectory 的最终 model/optimizer hash 与 v0.7 baseline raw 一致。
- Rq 是未来动态机制诊断量，不应直接当作可在线获取的 controller state。
- H8 只作 exploratory mechanism screening；不能改写 v0.7 的 H8_CONFIRMATION=FAIL。

## Final

E4B1_REPLAY_INTEGRITY = PASS
E4B1_Q_ROTATION_MEASURED = YES
E4B1_H8_CONDITIONAL_STATUS = UNSUPPORTED
E4B1_RQ_MONOTONE_DECAY = NO
