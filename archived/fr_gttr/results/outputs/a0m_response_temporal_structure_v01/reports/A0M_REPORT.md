# A0M Response Temporal Structure — REPORT

OFFLINE analysis of frozen A0K/A0L artifacts only. No training, no GPU, no optimizer/lambda/controller changes. Response convention: raw hhat = dV_score/dλ, NEGATIVE = beneficial.

## Dashboard

| Metric | Result |
|---|---|
| rank change frequency (median per seed) | 0.833 |
| response cosine @100 steps (mean) | 0.629 |
| phase clustering ARI / silhouette | 0.076 / 0.574 |
| single h utility Pearson (mean K) | -0.866 |
| window averaged utility Pearson (best w) | w=1: -0.866 |
| j1 dominance (argmin fraction) | 0.306 |
| stage dependent optimum (frac seeds) | 0.889 |
| top-action predicted vs actual match | 0.444 |

## Verdict: CASE A — response 存在明显 phase structure

## E1 trajectory
- median rank-change frequency per seed = 0.833
- per-seed top-action counts and Δij in analysis/response_trajectory.csv

## E3 stability (cos / spearman by lag)
| source | K | lag_steps | cos_mean | cos_sd | spe_mean | n |
|---|---|---|---|---|---|---|
| a0k | 10 | 10 | 0.666 | 0.296 | 0.097 | 33 |
| a0k | 10 | 20 | 0.646 | 0.269 | 0.027 | 30 |
| a0k | 10 | 50 | 0.57 | 0.392 | 0.029 | 21 |
| a0k | 10 | 80 | 0.662 | 0.4 | -0.05 | 12 |
| a0k | 10 | 100 | 0.646 | 0.392 | 0.233 | 6 |
| a0k | 20 | 20 | 0.708 | 0.205 | 0.127 | 30 |
| a0k | 20 | 80 | 0.74 | 0.185 | 0.067 | 12 |
| a0k | 20 | 100 | 0.612 | 0.35 | -0.033 | 6 |
| a0l | 50 | 50 | 0.816 | 0.198 | 0.067 | 3 |

## E4 utility prediction
| window | K | n | pearson | pearson_p | spearman |
|---|---|---|---|---|---|
| 1 | 10 | 144 | -0.8545 | 0 | -0.5536 |
| 1 | 20 | 144 | -0.8766 | 0 | -0.3846 |
| 2 | 10 | 144 | -0.5611 | 0 | -0.5171 |
| 2 | 20 | 144 | -0.6007 | 0 | -0.359 |
| 3 | 10 | 144 | -0.4513 | 0 | -0.4811 |
| 3 | 20 | 144 | -0.6323 | 0 | -0.3737 |
| 4 | 10 | 144 | -0.4735 | 0 | -0.4814 |
| 4 | 20 | 144 | -0.6349 | 0 | -0.3827 |
- top-action match rate = 0.444

## E5 fixed-j1 analysis
| source | K | seed | phase | n_events | mean_h1 | mean_h2 | mean_h3 | mean_h4 | argmin_frac_j1 | argmin_mode |
|---|---|---|---|---|---|---|---|---|---|---|
| a0k | 10 | 2030 | early | 4 | -0.007214 | -0.008102 | -0.008463 | -0.008378 | 0.25 | 1 |
| a0k | 10 | 2030 | late | 4 | -0.000888 | 0.000373 | -0.00277 | -0.00103 | 0.25 | 3 |
| a0k | 10 | 2030 | mid | 4 | -0.004054 | -0.001061 | -0.000817 | -0.003056 | 0.5 | 1 |
| a0k | 10 | 2031 | early | 4 | -0.01578 | -0.01514 | -0.01547 | -0.01574 | 0.25 | 4 |
| a0k | 10 | 2031 | late | 4 | -0.001281 | -0.000604 | -0.000756 | -0.001274 | 0.75 | 1 |
| a0k | 10 | 2031 | mid | 4 | -0.003605 | -0.004043 | -0.004911 | 0.000468 | 0 | 2 |
| a0k | 10 | 2032 | early | 4 | -0.007499 | -0.01134 | -0.008973 | -0.007686 | 0 | 2 |
| a0k | 10 | 2032 | late | 4 | -0.002019 | -0.000917 | -0.001575 | -0.001646 | 0.25 | 4 |
| a0k | 10 | 2032 | mid | 4 | -0.003178 | -0.002088 | -0.00282 | -0.00314 | 0.25 | 4 |
| a0k | 20 | 2030 | early | 2 | -0.006056 | -0.00635 | -0.008712 | -0.007327 | 0 | 2 |
| a0k | 20 | 2030 | late | 2 | -0.000757 | 0.000491 | -0.001036 | -0.000534 | 0.5 | 1 |
| a0k | 20 | 2030 | mid | 2 | -0.002967 | -0.001362 | -0.000667 | -0.002346 | 0.5 | 1 |
| a0k | 20 | 2031 | early | 2 | -0.02704 | -0.0201 | -0.02295 | -0.02102 | 0.5 | 1 |
| a0k | 20 | 2031 | late | 2 | -0.000861 | -0.00057 | -0.00067 | -0.001001 | 0 | 2 |
| a0k | 20 | 2031 | mid | 2 | -0.003861 | -0.001135 | -0.006148 | 0.00185 | 0 | 3 |
| a0k | 20 | 2032 | early | 2 | -0.004543 | -0.0067 | -0.009625 | -0.007413 | 0 | 3 |
| a0k | 20 | 2032 | late | 2 | -0.003475 | -0.000822 | -0.000138 | -0.002533 | 0.5 | 1 |
| a0k | 20 | 2032 | mid | 2 | -0.000912 | -0.002252 | -0.006093 | -0.002811 | 0 | 3 |
| a0k | 20 | 2033 | early | 2 | -0.004073 | -0.003965 | -0.00335 | -0.003887 | 0 | 3 |
| a0k | 20 | 2033 | late | 2 | -0.002122 | -0.000537 | -0.000341 | -0.001045 | 1 | 1 |
| a0k | 20 | 2033 | mid | 2 | -0.003962 | 0.000452 | -0.008942 | -0.006095 | 0 | 3 |
| a0k | 20 | 2034 | early | 2 | -0.00756 | -0.001672 | -0.004786 | -0.001923 | 0.5 | 1 |
| a0k | 20 | 2034 | late | 2 | -0.001166 | -0.0005 | -0.001561 | -0.000833 | 0.5 | 1 |
| a0k | 20 | 2034 | mid | 2 | -0.01003 | -0.00107 | -0.003961 | -0.000539 | 0.5 | 1 |
| a0k | 20 | 2035 | early | 2 | -0.007222 | -0.005109 | -0.004153 | -0.00304 | 1 | 1 |
| a0k | 20 | 2035 | late | 2 | -0.001627 | -0.001079 | -0.001881 | -0.002807 | 0 | 4 |
| a0k | 20 | 2035 | mid | 2 | -0.00557 | -0.003666 | -0.003321 | -0.003613 | 0.5 | 1 |
| a0l | 50 | 2030 | late | 1 | -0.00111 | 0.000485 | -0.001448 | -0.000743 | 0 | 3 |
| a0l | 50 | 2030 | mid | 1 | -0.002029 | -0.005595 | -0.005674 | -0.005235 | 0 | 3 |
| a0l | 50 | 2031 | late | 1 | -0.001104 | -0.001043 | -0.000594 | -0.001319 | 0 | 4 |
| a0l | 50 | 2031 | mid | 1 | -0.005405 | -0.005349 | -0.007197 | -0.004316 | 0 | 3 |
| a0l | 50 | 2032 | late | 1 | -0.002726 | -0.001306 | -0.000995 | -0.001598 | 1 | 1 |
| a0l | 50 | 2032 | mid | 1 | -0.005129 | -0.005116 | -0.003608 | -0.004405 | 1 | 1 |

## Common-mode diagnostic (explains E1/E3/E4 jointly)
- |mean(h_j)| / std(h_j) per event: median = 1.85 (>>1: common-mode dominated)
- common mode by phase: early -9.54e-03 / mid -2.99e-03 / late -1.17e-03 (~10x decay early->late)
- within-event Spearman(h_j, gain_j) mean = -0.233 (ranking signal is real but small vs common mode)

## Notes
- A0K step logs do not contain training loss; phase features use v_eval@step and preclip_norm (gradient norm) instead. Optimizer state statistics not saved in any frozen artifact — E2 runs without them (noted in phase_cluster_report.md).
- A0L contributes only 2 events/seed (steps 50/100, mixture-regime response, same production scorer units); used as secondary series in E1/E3/E5.
- Mandate lags k=1,5 steps are below the frozen event grid (K=10 -> 10-step spacing); reported at available lags 10/20/50/80/100 steps.
- No algorithm designed; mechanism report only. A0G/A0H/A0I/A0K/A0L artifacts untouched.
