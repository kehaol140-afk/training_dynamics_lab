# FR A0D Finite-Horizon Epsilon Convergence Audit

## Technical summary

1. **Finite differences enter the linear region: PARTIAL_7_OF_9_STATE_HORIZONS.** Successive finite-difference changes contract in 7/9 state-horizon cells under exact epsilon halving.
2. **Full-tangent error as epsilon decreases: PARTIAL_4_OF_9_STATE_HORIZONS.** Relative error decreases monotonically in 4/9 cells (absolute error: 4/9). Pooled mean relative error changes from 0.0218656 at epsilon to 0.00452423 at epsilon/4.
3. **H=8 retains the full-tangent advantage: YES_AT_EPSILON_QUARTER_ALL_3_WITH_NONMONOTONIC_INTERMEDIATE.** H=8 epsilon/4 error is lower than full-epsilon error in 3/3 states, while strict three-level monotonicity holds in 0/3. Its pooled mean relative error changes from 0.0111703 to 0.00646787, with epsilon/4 R2=0.999644 and corr=0.999858.

## Smaller pulses test the same zero-pulse tangent

The table is used instead of a trend chart because each curve has only three registered epsilon anchors. Errors compare the stream-mean finite difference with the unchanged A0C full-tangent prediction.

| state | H | epsilon level | epsilon | chi_fd | chi_tangent | absolute error | relative error |
|---|---:|---|---:|---:|---:|---:|---:|
| seed_2037/state_step_046 | 2 | epsilon | 0.343006229187 | -0.0831821646531 | -0.090524686705 | 0.00734252205191 | 0.0882703892419 |
| seed_2037/state_step_046 | 2 | epsilon/2 | 0.171503114594 | -0.0882120768789 | -0.090524686705 | 0.00231260982604 | 0.0262164763359 |
| seed_2037/state_step_046 | 2 | epsilon/4 | 0.0857515572968 | -0.0895284930434 | -0.090524686705 | 0.000996193661563 | 0.0111271130307 |
| seed_2037/state_step_046 | 4 | epsilon | 0.343006229187 | -0.131721904397 | -0.139873131962 | 0.00815122756564 | 0.0618820962465 |
| seed_2037/state_step_046 | 4 | epsilon/2 | 0.171503114594 | -0.139020484245 | -0.139873131962 | 0.00085264771703 | 0.00613325238837 |
| seed_2037/state_step_046 | 4 | epsilon/4 | 0.0857515572968 | -0.13939907493 | -0.139873131962 | 0.00047405703225 | 0.00340071863812 |
| seed_2037/state_step_046 | 8 | epsilon | 0.343006229187 | -0.187717752006 | -0.192524920027 | 0.00480716802159 | 0.0256084891824 |
| seed_2037/state_step_046 | 8 | epsilon/2 | 0.171503114594 | -0.200684580706 | -0.192524920027 | 0.00815966067846 | 0.0406591311089 |
| seed_2037/state_step_046 | 8 | epsilon/4 | 0.0857515572968 | -0.195556833456 | -0.192524920027 | 0.00303191342892 | 0.0155040014472 |
| seed_2030/state_step_046 | 2 | epsilon | 0.0138748625235 | -0.121867887571 | -0.122505760615 | 0.000637873043929 | 0.00523413555977 |
| seed_2030/state_step_046 | 2 | epsilon/2 | 0.00693743126176 | -0.122306746219 | -0.122505760615 | 0.000199014395936 | 0.00162717431449 |
| seed_2030/state_step_046 | 2 | epsilon/4 | 0.00346871563088 | -0.12245204237 | -0.122505760615 | 5.37182443769e-05 | 0.000438688022977 |
| seed_2030/state_step_046 | 4 | epsilon | 0.0138748625235 | -0.213234390119 | -0.21434959922 | 0.00111520910024 | 0.00522996829738 |
| seed_2030/state_step_046 | 4 | epsilon/2 | 0.00693743126176 | -0.213918189031 | -0.21434959922 | 0.000431410188198 | 0.00201670643413 |
| seed_2030/state_step_046 | 4 | epsilon/4 | 0.00346871563088 | -0.214258737637 | -0.21434959922 | 9.08615828112e-05 | 0.000424074106911 |
| seed_2030/state_step_046 | 8 | epsilon | 0.0138748625235 | -0.325153506104 | -0.326869252535 | 0.00171574643145 | 0.005276727451 |
| seed_2030/state_step_046 | 8 | epsilon/2 | 0.00693743126176 | -0.326384777089 | -0.326869252535 | 0.000484475446345 | 0.00148436900356 |
| seed_2030/state_step_046 | 8 | epsilon/4 | 0.00346871563088 | -0.326265767987 | -0.326869252535 | 0.00060348454802 | 0.00184967167026 |
| seed_2037/state_step_082 | 2 | epsilon | 0.0407303748051 | -0.049851961905 | -0.0499075718341 | 5.56099290726e-05 | 0.00111550131525 |
| seed_2037/state_step_082 | 2 | epsilon/2 | 0.0203651874025 | -0.0498981260289 | -0.0499075718341 | 9.44580513592e-06 | 0.000189301801243 |
| seed_2037/state_step_082 | 2 | epsilon/4 | 0.0101825937013 | -0.0498507714671 | -0.0499075718341 | 5.6800366965e-05 | 0.00113940798293 |
| seed_2037/state_step_082 | 4 | epsilon | 0.0407303748051 | -0.0761822229814 | -0.0763000911819 | 0.000117868200556 | 0.00154718772889 |
| seed_2037/state_step_082 | 4 | epsilon/2 | 0.0203651874025 | -0.0763259730015 | -0.0763000911819 | 2.58818195205e-05 | 0.000339095834651 |
| seed_2037/state_step_082 | 4 | epsilon/4 | 0.0101825937013 | -0.075936778411 | -0.0763000911819 | 0.000363312770964 | 0.00478441117159 |
| seed_2037/state_step_082 | 8 | epsilon | 0.0407303748051 | -0.094801363302 | -0.0950502782739 | 0.000248914971837 | 0.00262564759796 |
| seed_2037/state_step_082 | 8 | epsilon/2 | 0.0203651874025 | -0.0951962056843 | -0.0950502782739 | 0.000145927410425 | 0.00153291204598 |
| seed_2037/state_step_082 | 8 | epsilon/4 | 0.0101825937013 | -0.0948558290511 | -0.0950502782739 | 0.00019444922279 | 0.0020499448978 |

## Predictive agreement by epsilon level

Each row pools the three states and H=2,4,8 (nine observations). The pooled endpoint improvement is consistent with reduced finite-amplitude bias, but only 4/9 state-horizon cells improve monotonically; it does not distinguish finite-amplitude truncation from tangent-backend or omitted-dynamics error.

| epsilon level | R2 | correlation | MAE | RMSE | mean relative error |
|---|---:|---:|---:|---:|---:|
| epsilon | 0.997519778961 | 0.99930632154 | 0.00268801547958 | 0.00405705569912 | 0.0218655714023 |
| epsilon/2 | 0.998796326095 | 0.999434386028 | 0.00140234147634 | 0.00285068659495 | 0.00891093547414 |
| epsilon/4 | 0.999817862659 | 0.999911164914 | 0.00065164342874 | 0.001103428946 | 0.00452422566317 |

## Successive-halving diagnostics

`fd contraction` compares |chi(epsilon/2)-chi(epsilon/4)| with |chi(epsilon)-chi(epsilon/2)|. `p` is the descriptive log2 ratio of those changes; no convergence threshold or fitted variable is introduced.

| state | H | fd change full→half | fd change half→quarter | contraction ratio | p | error monotonic |
|---|---:|---:|---:|---:|---:|---|
| seed_2037/state_step_046 | 2 | 0.00502991222587 | 0.00131641616447 | 0.261717522167 | 1.93391757785 | True |
| seed_2037/state_step_046 | 4 | 0.00729857984861 | 0.000378590684779 | 0.0518718288533 | 4.26890495385 | True |
| seed_2037/state_step_046 | 8 | 0.0129668287001 | 0.00512774724954 | 0.395451144467 | 1.33842862519 | False |
| seed_2030/state_step_046 | 2 | 0.000438858647993 | 0.000145296151559 | 0.331077334863 | 1.59475984586 | True |
| seed_2030/state_step_046 | 4 | 0.000683798912038 | 0.000340548605386 | 0.498024491399 | 1.0057114033 | True |
| seed_2030/state_step_046 | 8 | 0.0012312709851 | 0.000119009101675 | 0.0966554910459 | 3.3710044948 | False |
| seed_2037/state_step_082 | 2 | 4.61641239367e-05 | 4.73545618291e-05 | 1.0257870786 | -0.0367313035309 | False |
| seed_2037/state_step_082 | 4 | 0.000143750020077 | 0.000389194590484 | 2.70744025132 | -1.43692950032 | False |
| seed_2037/state_step_082 | 8 | 0.000394842382262 | 0.000340376633215 | 0.862056984017 | 0.214144856848 | False |

## Scope, definitions, and replay integrity

- Scope is the three locked A0C states, eight E2 streams per state, H=2,4,8, and epsilon multipliers 1, 1/2, 1/4.
- `chi_fd=(X_plus-X_minus)/(2*epsilon_level)` is a derivative per unit post-unscale gradient pulse. The denominator changes with each epsilon level; epsilon is never multiplied into `chi_tangent`.
- Full-epsilon branches are the immutable confirmatory v07 measurements. Half- and quarter-epsilon branches are new strict replays using the same checkpoint, state-specific q, manifest batches, stream RNG, BF16 autocast, two-microbatch accumulation, clipping, LR schedule, and native AdamW update.
- `chi_tangent` is read unchanged from the A0C v01 mixed-backend full-tangent artifact and is identical across epsilon levels. Native flash-SDPA does not expose double backward, so its auxiliary HVP used math-SDPA; the recorded first-gradient backend gap remains a limitation, not a fitted correction.
- Input integrity: `PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION`; replay provenance: `PASS_TRANSITIVE_CODE_AND_BRANCH_STATE_HASHES`; full-epsilon source check: `PASS_EXACT_V07_VS_A0C_V01`; new replay branch count: `96`.
- The inherited math/native first-gradient gap is `0.00505791896242`, comparable to the pooled epsilon/4 mean relative error. It is not an HVP-error bound, so the endpoint improvement cannot by itself rule out tangent-backend error.
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.

## Limitations and robustness

The audit varies only pulse amplitude while holding the registered states, directions, batches, RNG, and horizons fixed. With eight streams and three states, per-H correlation/R2 has n=3 and should be read as a consistency diagnostic, not a population estimate. Clipping-regime crossings are part of the finite-epsilon response and may make convergence non-monotonic for individual streams even when the state mean contracts.

## Recommended next step

Treat the epsilon/4 discrepancy as the observed residual, not a bound on tangent-model error. Run a fixed epsilon/8 diagnostic for all 5 cells with non-contracting finite differences or nonmonotonic tangent error, including all H=8 cells; do not fit a new memory term from the present finite-amplitude residuals.

## Further question

For any non-contracting state-horizon cell, does epsilon/8 restore central-difference contraction without changing clipping regime?
