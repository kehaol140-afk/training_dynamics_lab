# v0.7 Locked Confirmation Recovery

- generated_at_utc: `2026-09-03T04:43:05Z`
- PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS: `YES`
- PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY: `YES`
- DISCOVERY_COMPARISON_STATUS: `UNAVAILABLE_DUE_TO_DEGENERATE_4-SEED_CLUSTER_BOOTSTRAP`
- H8_CONFIRMATION: `FAIL`

## Operational Equivalence

- recovery 只执行 locked analysis 中 discovery appendix 之前的 confirmation-only path。
- recovery 不读取 discovery seeds `2026..2029`。
- H=8 数据、函数、参数、随机种子、bootstrap reps、permutation reps 与 locked config/locked stats 一致。
- 未添加、删除或修改 prereg A/B/C 条件。
- 未引入 post-data scientific degree of freedom。

## Root Cause Diagnosis

- n=4 bootstrap 单个 replicate 只含 1 个 unique seed 的概率：`4/4^4 = 1/64` = `0.015625`
- 10000 reps 期望退化 replicate 数：`156.25`
- 固定 RNG 复现：H=1 discovery 第 `29` 个 replicate 抽到 `[2027, 2027, 2027, 2027]`

## Primary H=8

- status: `FAIL`
- K_8: `0.376861`
- cluster CI95 lower: `0.190872`
- detrended Pearson: `0.540557`
- circular-shift p(one-sided positive): `0.000149993`
- LOSO mean relative MAE improvement: `0.0522218`
- LOSO folds improved: `5/8`
- A_K8_positive_cluster_ci_lower_gt_0: `True`
- B_detrended_circular_shift_one_sided_p_lt_0_05: `True`
- C_LOSO_mean_gain_gt_0_and_at_least_6_of_8_folds: `False`

## Horizons

### H=1
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.200296`
- cluster CI95(K_H): `[0.118755, 0.286045]`
- detrended Pearson: `0.389553`
- circular-shift p: `0.00779961`
- first-difference Pearson: `0.287303`
- LOSO mean MAE improvement: `0.0369474`
- LOSO folds improved: `4/8`

### H=2
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.141723`
- cluster CI95(K_H): `[-0.0177513, 0.315759]`
- detrended Pearson: `0.216579`
- circular-shift p: `0.0519474`
- first-difference Pearson: `0.101992`
- LOSO mean MAE improvement: `-0.0201649`
- LOSO folds improved: `5/8`

### H=4
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.294994`
- cluster CI95(K_H): `[0.126737, 0.564369]`
- detrended Pearson: `0.428142`
- circular-shift p: `9.9995e-05`
- first-difference Pearson: `0.414201`
- LOSO mean MAE improvement: `0.0133353`
- LOSO folds improved: `5/8`

### H=8
- endpoint_role: `PRIMARY`
- seed-FE K_H: `0.376861`
- cluster CI95(K_H): `[0.190872, 0.670764]`
- detrended Pearson: `0.540557`
- circular-shift p: `0.000149993`
- first-difference Pearson: `0.603388`
- LOSO mean MAE improvement: `0.0522218`
- LOSO folds improved: `5/8`

### H=12
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.204333`
- cluster CI95(K_H): `[0.0291026, 0.448511]`
- detrended Pearson: `0.304011`
- circular-shift p: `0.0435478`
- first-difference Pearson: `0.368043`
- LOSO mean MAE improvement: `0.0804963`
- LOSO folds improved: `7/8`

### H=16
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.0741857`
- cluster CI95(K_H): `[-0.107963, 0.271719]`
- detrended Pearson: `0.171808`
- circular-shift p: `0.136243`
- first-difference Pearson: `0.239498`
- LOSO mean MAE improvement: `0.00175129`
- LOSO folds improved: `7/8`

### H=24
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.104651`
- cluster CI95(K_H): `[-0.0428002, 0.371444]`
- detrended Pearson: `0.19251`
- circular-shift p: `0.124994`
- first-difference Pearson: `0.225503`
- LOSO mean MAE improvement: `-0.0331731`
- LOSO folds improved: `5/8`

### H=32
- endpoint_role: `SECONDARY`
- seed-FE K_H: `0.196779`
- cluster CI95(K_H): `[-0.0407843, 0.448246]`
- detrended Pearson: `0.259664`
- circular-shift p: `0.0380981`
- first-difference Pearson: `0.263068`
- LOSO mean MAE improvement: `-0.0223671`
- LOSO folds improved: `4/8`
