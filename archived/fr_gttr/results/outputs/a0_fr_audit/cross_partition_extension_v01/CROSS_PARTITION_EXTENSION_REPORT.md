# FR A0F2 Cross-Partition Extension v01 (Full v07 Bank)

## Technical summary

1. **Full tangent 在全库 56 状态上是否稳定优于 optimizer-only：YES_C_BEATS_B_IN_168_OF_224_STATE_HORIZON_ROWS_MEAN_IMPROVEMENT_+0.121789。** Paired per-state comparison over the full 56-state v07 census and horizons H=1,2,4,8: C beats B in 168/224 rows (ties 56, losses 0); mean relative-error improvement +0.121789, median +0.0420282, worst regression +0. H=1 rows are a zero-control (B == C by construction, no future step).
2. **与 FR_A0F 32 状态随机抽样结果的一致性：BIT_EXACT_REPRODUCTION_OF_ALL_SHARED_STATES。** All 128 state-horizon rows and 256 streams shared with the frozen FR_A0F artifact are reproduced with max absolute gap 0 (tolerance 1e-09); the A0F random sample is a strict subset of this census.
3. **跨训练阶段的一致性：STEPS_WITH_POSITIVE_MEAN_IMPROVEMENT_7_OF_7。** Mean B-to-C relative-error improvement (H>1) is positive at every one of the 7 locked training stages (steps 46-82); the weakest stage mean is +0.0807127. Mean relative error over all rows: A=4.35789, B=0.127661, C=0.00587211.

## Coverage

Full census of the v07 confirmatory bank: 8 seeds x 7 steps (46-82) = 56 states, 8 paired E2 streams per state (448 streams total). The FR_A0F 32-state random sample is a strict subset and is reproduced and cross-checked inside this run.

| seed | steps |
|---:|---|
| 2030 | 46, 52, 58, 64, 70, 76, 82 |
| 2031 | 46, 52, 58, 64, 70, 76, 82 |
| 2032 | 46, 52, 58, 64, 70, 76, 82 |
| 2033 | 46, 52, 58, 64, 70, 76, 82 |
| 2034 | 46, 52, 58, 64, 70, 76, 82 |
| 2035 | 46, 52, 58, 64, 70, 76, 82 |
| 2036 | 46, 52, 58, 64, 70, 76, 82 |
| 2037 | 46, 52, 58, 64, 70, 76, 82 |

The v06 replication bank is not included: its theta mappings carry no per-state q files, no E2 streams, and no batch manifests, and its locked horizon set is {1,8,32}; hosting the A/B/C comparison there requires generating new E2/E3 observations rather than replaying registered ones.

## Per-horizon metrics over all 56 states

Metrics are computed by the frozen FR_A0C code (imported verbatim). Each horizon pools the per-state stream-mean responses of all 56 states.

| horizon | model | R2 | corr | MAE | RMSE | mean relative error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A_momentum | -176.898137781 | 0.925328322675 | 0.180977337075 | 0.199160642073 | 3.54583799233 |
| 1 | B_optimizer_only | 0.986118922947 | 0.99514900542 | 0.00040887831638 | 0.0017592574516 | 0.00609520252943 |
| 1 | C_full_tangent | 0.986118922947 | 0.99514900542 | 0.00040887831638 | 0.0017592574516 | 0.00609520252943 |
| 2 | A_momentum | -228.369845179 | 0.870607801868 | 0.349187534965 | 0.385020709916 | 3.83092312661 |
| 2 | B_optimizer_only | 0.864387358868 | 0.961869384827 | 0.00434428128883 | 0.0093619435218 | 0.0459312013228 |
| 2 | C_full_tangent | 0.987221390306 | 0.994851854077 | 0.000661522453826 | 0.00287380788745 | 0.00594183748479 |
| 4 | A_momentum | -266.420754178 | 0.743364265079 | 0.649424656804 | 0.716059924552 | 4.5231447461 |
| 4 | B_optimizer_only | 0.42194826177 | 0.853836270858 | 0.0202961363726 | 0.0332916384504 | 0.155224863505 |
| 4 | C_full_tangent | 0.984413329476 | 0.99296433382 | 0.0011895231062 | 0.00546674090906 | 0.00746372149835 |
| 8 | A_momentum | -287.237982236 | 0.743698889583 | 1.10756209546 | 1.21525861438 | 5.5316593284 |
| 8 | B_optimizer_only | 0.105201976336 | 0.854405641632 | 0.0532054993014 | 0.0677104362486 | 0.303393264924 |
| 8 | C_full_tangent | 0.998843443106 | 0.999517664925 | 0.000882867336998 | 0.00243431404391 | 0.00398769235537 |

## Per-state relative error distribution across 56 states (mean / std / median)

| horizon | model | mean | std | median | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A_momentum | 3.54583799233 | 0.844035423948 | 3.41927815439 | 2.65716357756 | 8.68290048301 |
| 1 | B_optimizer_only | 0.00609520252943 | 0.0222842063826 | 0.00122253948186 | 0.000368347814048 | 0.125268404332 |
| 1 | C_full_tangent | 0.00609520252943 | 0.0222842063826 | 0.00122253948186 | 0.000368347814048 | 0.125268404332 |
| 2 | A_momentum | 3.83092312661 | 1.08498646949 | 3.6279347432 | 2.73684517916 | 10.0303507642 |
| 2 | B_optimizer_only | 0.0459312013228 | 0.0684379792421 | 0.0192638956921 | 0.00207858835599 | 0.390732068373 |
| 2 | C_full_tangent | 0.00594183748479 | 0.0215465383448 | 0.00154636740923 | 7.95887021157e-06 | 0.139106630062 |
| 4 | A_momentum | 4.5231447461 | 1.61304320002 | 4.16876137377 | 2.95040417596 | 11.6078233408 |
| 4 | B_optimizer_only | 0.155224863505 | 0.186778569056 | 0.0739702674776 | 0.0121269373572 | 0.914176680296 |
| 4 | C_full_tangent | 0.00746372149835 | 0.0302154330464 | 0.00199091110812 | 0.000115414697667 | 0.221182932161 |
| 8 | A_momentum | 5.5316593284 | 2.03152329764 | 4.81330238165 | 3.35596161149 | 13.6514068515 |
| 8 | B_optimizer_only | 0.303393264924 | 0.270475625436 | 0.185653657032 | 0.0439715878626 | 1.18297774575 |
| 8 | C_full_tangent | 0.00398769235537 | 0.00766819311607 | 0.00234688042814 | 4.53952890267e-06 | 0.0526469571583 |

## Paired B-vs-C comparison per state (improvement = relErr_B - relErr_C)

A positive improvement means the full tangent C predicts that state's response better than optimizer-only B. H=1 has B == C by construction (zero-control).

| horizon | mean improvement | std | median | wins | ties | losses | win rate | worst regression | best improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 |
| 2 | 0.039989363838 | 0.0531130270576 | 0.0174835391861 | 56 | 0 | 0 | 1 | 0.00199255708527 | 0.251625438311 |
| 4 | 0.147761142006 | 0.170117658718 | 0.0731579862728 | 56 | 0 | 0 | 1 | 0.0113463318098 | 0.692993748135 |
| 8 | 0.299405572569 | 0.267889879548 | 0.184035993977 | 56 | 0 | 0 | 1 | 0.0430662315549 | 1.17506408233 |
| **all** | 0.121789019603 | 0.197560260277 | 0.0420282305321 | 168 | 56 | 0 | 0.75 | 0 | 1.17506408233 |

## Per-seed summary (mean relative error, paired B-to-C)

| seed | mean rel err B | mean rel err C | mean improvement | C-beats-B rows | rows |
|---:|---:|---:|---:|---:|---:|
| 2030 | 0.20618926668 | 0.021670453354 | 0.184518813326 | 21 | 28 |
| 2031 | 0.161168238369 | 0.00249437079871 | 0.15867386757 | 21 | 28 |
| 2032 | 0.0533681726462 | 0.00157678518466 | 0.0517913874615 | 21 | 28 |
| 2033 | 0.0936829568968 | 0.00257796345246 | 0.0911049934443 | 21 | 28 |
| 2034 | 0.115923399316 | 0.00118116519864 | 0.114742234117 | 21 | 28 |
| 2035 | 0.123929719305 | 0.00333138693056 | 0.120598332375 | 21 | 28 |
| 2036 | 0.0922488801915 | 0.00205477404726 | 0.0901941061443 | 21 | 28 |
| 2037 | 0.174778431158 | 0.0120900087696 | 0.162688422388 | 21 | 28 |

## Consistency with the frozen FR_A0F artifact

- Shared states reproduced: `True`; rows checked: 128, streams checked: 256.
- Max absolute reproduction gap by field: row:chi_A_momentum=0, row:chi_full_tangent=0, row:chi_optimizer_only=0, stream:chi_full_tangent=0, stream:chi_optimizer_only=0.

## Scope and method

- Same replay protocol and frozen FR_A0C A/B/C definitions as FR_A0F; only the state set changes (census instead of random sample).
- Model A uses each state's registered first-future-step LR (`eta`) from the frozen `mv_theory_consistent` artifact (covers all 56 states).

## Robustness, replay integrity, and limitations

- Baseline replay: `PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS` over 448 streams.
- Max math-SDPA HVP-forward gradient relative gap vs native flash: `0.0159493466134`.
- Clipping was active on 1075/3136 future steps (H>1).
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- Tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; per-state residuals may include finite-epsilon clip/activation boundary crossings.

## Recommended next step

The full-bank census confirms the FR_A0F sample. Cross-partition replication on the v06 bank requires generating a v06-compatible observation protocol (per-state q, E2 streams, batch manifests, and a {1,2,4,8}-horizon registration); schedule it as a new E2/E3-grade experiment rather than a replay extension. Do not fit any memory kernel before that replication exists.

## Further question

Does the C advantage persist on a different model scale or optimizer configuration once the v06 observation protocol exists?