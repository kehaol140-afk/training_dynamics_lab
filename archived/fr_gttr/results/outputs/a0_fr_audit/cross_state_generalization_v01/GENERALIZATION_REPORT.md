# FR A0F Cross-State Generalization v01

## Technical summary

1. **Full tangent 是否稳定优于 optimizer-only：YES_C_BEATS_B_IN_96_OF_128_STATE_HORIZON_ROWS_MEAN_IMPROVEMENT_+0.124154。** Paired per-state comparison over 32 sampled states and horizons H=1,2,4,8: C beats B in 96/128 rows (ties 32, losses 0); mean relative-error improvement +0.124154, median +0.0441067, worst regression +0. H=1 rows are a zero-control (B == C by construction, no future step).
2. **优势是否跨 seed、跨训练阶段一致：SEEDS_WITH_POSITIVE_MEAN_IMPROVEMENT_8_OF_8。** Mean B-to-C relative-error improvement is positive on 3/4 horizons and on 8/8 seeds. Mean relative error over all rows: A=4.52298, B=0.128616, C=0.0044613.
3. **A/B/C 三层模型的总体排序：C_BEATS_B_BEATS_A。** Ordering is by mean relative error over all sampled state-horizon rows; no fitted or empirical variable enters any model.

## Sampled states

Deterministic sampling: per-seed independent RNG seeded with `20260911 + seed`, 4 of 7 steps per seed; 32 states total, 8 seeds.

| seed | sampled steps |
|---:|---|
| 2030 | 52, 70, 76, 82 |
| 2031 | 46, 52, 64, 70 |
| 2032 | 46, 58, 70, 82 |
| 2033 | 46, 58, 70, 76 |
| 2034 | 46, 52, 58, 70 |
| 2035 | 46, 52, 64, 82 |
| 2036 | 46, 52, 64, 76 |
| 2037 | 46, 58, 64, 76 |

## Per-horizon metrics over all sampled states

Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim). Each horizon pools the per-state stream-mean responses of all sampled states.

| horizon | model | R2 | corr | MAE | RMSE | mean relative error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A_momentum | -232.345506966 | 0.857606015319 | 0.192195109388 | 0.207693161305 | 3.68294264039 |
| 1 | B_optimizer_only | 0.993341489678 | 0.99690212574 | 0.000279958842686 | 0.00110945837769 | 0.00554343232553 |
| 1 | C_full_tangent | 0.993341489678 | 0.99690212574 | 0.000279958842686 | 0.00110945837769 | 0.00554343232553 |
| 2 | A_momentum | -257.531059993 | 0.816521000353 | 0.370377223047 | 0.400164706864 | 3.97461244264 |
| 2 | B_optimizer_only | 0.935151849215 | 0.981421022901 | 0.00401814335595 | 0.00633769328357 | 0.0452380818438 |
| 2 | C_full_tangent | 0.997206046486 | 0.998728219139 | 0.000399116807844 | 0.00131550349822 | 0.00458440046196 |
| 4 | A_momentum | -272.790502091 | 0.737131852142 | 0.688081453974 | 0.742250240051 | 4.67403527532 |
| 4 | B_optimizer_only | 0.619433290472 | 0.909730516643 | 0.0199997584179 | 0.0276730198906 | 0.153501164842 |
| 4 | C_full_tangent | 0.998872613435 | 0.999522565371 | 0.000598093139373 | 0.00150618245748 | 0.00445812737781 |
| 8 | A_momentum | -305.622503447 | 0.694600754266 | 1.17392979301 | 1.26296408692 | 5.7603190081 |
| 8 | B_optimizer_only | 0.142985052875 | 0.862338797803 | 0.0549701219847 | 0.0667702783928 | 0.310180463555 |
| 8 | C_full_tangent | 0.999789781659 | 0.999920040112 | 0.000613131835026 | 0.00104574137286 | 0.0032592198625 |

## Per-state relative error distribution (mean / std / median across states)

| horizon | model | mean | std | median | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A_momentum | 3.68294264039 | 1.02097377482 | 3.58527938889 | 2.65716357756 | 8.68290048301 |
| 1 | B_optimizer_only | 0.00554343232553 | 0.0218748649194 | 0.00122870680992 | 0.000369636094869 | 0.125268404332 |
| 1 | C_full_tangent | 0.00554343232553 | 0.0218748649194 | 0.00122870680992 | 0.000369636094869 | 0.125268404332 |
| 2 | A_momentum | 3.97461244264 | 1.25030071935 | 3.7568653785 | 2.73684517916 | 10.0303507642 |
| 2 | B_optimizer_only | 0.0452380818438 | 0.0555773655887 | 0.0192638956921 | 0.00207858835599 | 0.218977884042 |
| 2 | C_full_tangent | 0.00458440046196 | 0.0153346653635 | 0.00190397356755 | 1.05312180938e-05 | 0.0882703892419 |
| 4 | A_momentum | 4.67403527532 | 1.6789439414 | 4.19488437813 | 2.95040417596 | 11.6078233408 |
| 4 | B_optimizer_only | 0.153501164842 | 0.163496156775 | 0.0720911932739 | 0.0121269373572 | 0.613005842496 |
| 4 | C_full_tangent | 0.00445812737781 | 0.0107233101138 | 0.00209560685525 | 0.000115414697667 | 0.0618820962465 |
| 8 | A_momentum | 5.7603190081 | 2.26090395447 | 5.03126609391 | 3.35596161149 | 13.6514068515 |
| 8 | B_optimizer_only | 0.310180463555 | 0.276577876665 | 0.174097700053 | 0.0439715878626 | 1.15345422971 |
| 8 | C_full_tangent | 0.0032592198625 | 0.00478779329535 | 0.00236815759649 | 4.53952890267e-06 | 0.0256084891824 |

## Paired B-vs-C comparison per state (improvement = relErr_B - relErr_C)

A positive improvement means the full tangent C predicts that state's response better than optimizer-only B. H=1 has B == C by construction (no future step, hence no curvature feedback), so it serves as a zero-control.

| horizon | mean improvement | std | median | wins | ties | losses | win rate | worst regression | best improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 32 | 0 | 0 | 0 | 0 |
| 2 | 0.0406536813819 | 0.0482092535254 | 0.0170799737672 | 32 | 0 | 0 | 1 | 0.00199255708527 | 0.153386164773 |
| 4 | 0.149043037465 | 0.160023840461 | 0.0692729651527 | 32 | 0 | 0 | 1 | 0.0113463318098 | 0.608831676235 |
| 8 | 0.306921243692 | 0.27487409137 | 0.172626057526 | 32 | 0 | 0 | 1 | 0.0430662315549 | 1.14838702325 |
| **all** | 0.124154490635 | 0.198681252884 | 0.0441066906212 | 96 | 32 | 0 | 0.75 | 0 | 1.14838702325 |

## Per-seed summary (mean relative error, paired B-to-C)

| seed | mean rel err B | mean rel err C | mean improvement | C-beats-B rows | rows |
|---:|---:|---:|---:|---:|---:|
| 2030 | 0.092625229566 | 0.00149318117138 | 0.0911320483946 | 12 | 16 |
| 2031 | 0.170155759765 | 0.0032275662862 | 0.166928193479 | 12 | 16 |
| 2032 | 0.0624849813183 | 0.00199329301424 | 0.0604916883041 | 12 | 16 |
| 2033 | 0.0593240331892 | 0.0017624550959 | 0.0575615780933 | 12 | 16 |
| 2034 | 0.131981576674 | 0.00128205161809 | 0.130699525056 | 12 | 16 |
| 2035 | 0.18390072777 | 0.00344337472135 | 0.180457353049 | 12 | 16 |
| 2036 | 0.106273984265 | 0.00241986525928 | 0.103854119006 | 12 | 16 |
| 2037 | 0.222179992586 | 0.0200685728891 | 0.202111419697 | 12 | 16 |

## Scope and method

- Same replay protocol, horizons H=1,2,4,8, eight paired E2 streams per state, and frozen FR_A0C A/B/C definitions.
- Model A uses each state's registered first-future-step LR (`eta`) from the frozen `mv_theory_consistent` artifact.
- Every stream restores the exact checkpoint, registered RNG, and registered future batches; whole-state hashes of the propagated baseline are compared against a fresh existing-protocol H=8 shadow branch.

## Robustness, replay integrity, and limitations

- Baseline replay: `PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS` over 256 streams.
- Max math-SDPA HVP-forward gradient relative gap vs native flash: `0.010527212733` (does not bound the HVP error; qualifies C's smallest residuals, as in FR_A0C).
- Clipping was active on 565/1792 future steps (H>1).
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- All tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; per-state residuals may include finite-epsilon clip/activation boundary crossings, so the stability claim is about the tangent model class, not about exactness at any single state.

## Recommended next step

If uniformly favorable, extend the same frozen A/B/C comparison to the full 56-state v07 bank and to the v06 replication bank for cross-partition generalization; do not fit any memory kernel before that.

## Further question

Are the B-beating-C state-horizon rows explained by active clipping at the registered epsilon (see A0E decomposition) rather than by missing smooth curvature?