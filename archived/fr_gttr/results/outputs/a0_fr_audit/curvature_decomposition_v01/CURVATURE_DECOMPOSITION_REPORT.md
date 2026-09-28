# FR A0E Curvature Contribution Decomposition v01

## Technical summary

1. **Curvature feedback contribution (gain_H = error_B - error_BH)：NONPOSITIVE_MEAN_GAIN_H_-0.135725。** Adding only delta_g=H delta_theta to model B changes the mean H in {2,4,8} relative error from B=0.205255 to BH=0.340979 (gain_H=-0.135725); BH beats B in 6/9 rows. The FR_A0C full tangent reaches C=0.0218656.
2. **Optimizer memory contribution (gain_optimizer = error_A - error_B)：POSITIVE_MEAN_GAIN_OPTIMIZER_+6.55297。** Replacing the historical A kernel by the complete optimizer-only tangent B changes mean relative error from A=6.75822 to B=0.205255 (gain_optimizer=+6.55297). As noted in FR_A0C, A-to-B also changes the local preconditioner/bias correction/clip/LR-schedule/weight-decay treatment, so this gain is the full optimizer-side contribution, not memory alone.
3. **BH 与 FR_A0C full tangent 的差距（clip-JVP 修正）：MEAN_CLIP_CORRECTION_+0.319114。** BH differs from the FR_A0C full tangent only by the clip-JVP radial correction applied to the HVP feedback; its mean relative-error effect is +0.319114. Max |BH-C| response gap at H=8 across streams is 0.609221. Clipping was active on 48/168 future steps (H>1).

## Model ladder

- **A_momentum** — historical kernel `-eta*mu_raw*(1-beta1**H)` (FR_A0C, unchanged).
- **B_optimizer_only** — FR_A0C complete stepwise AdamW tangent; pulse `delta_g=q`, future `delta_g=0`.
- **BH_curvature_feedback** — Model B plus the single future term `delta_g = H(loss) delta_theta` (raw math-SDPA HVP; no clip-JVP radial correction, no other higher-order term).
- **C_full_tangent** — FR_A0C full tangent: the HVP feedback passes through the exact global-clip Jacobian.

Wherever clipping is inactive BH and C coincide by construction; the BH-to-C gap therefore isolates the clipping Jacobian contribution.

## Per state-horizon error table (three decomposition models + A reference)

`rel X` is absolute prediction error divided by `|chi_fd|`. `gain_optimizer = relA - relB`, `gain_H = relB - relBH`.

| state | H | chi_fd | A | B | BH | C | rel A | rel B | rel BH | rel C | gain_optimizer | gain_H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_2037/state_step_046 | 2 | -0.0831821646531 | -0.91752845345 | -0.101397219059 | -0.154627767686 | -0.090524686705 | 10.0303507642 | 0.218977884042 | 0.858905311387 | 0.0882703892419 | 9.81137288018 | -0.639927427345 |
| seed_2037/state_step_046 | 4 | -0.131721904397 | -1.66072650074 | -0.176069341039 | -0.269871397029 | -0.139873131962 | 11.6078233408 | 0.336674730338 | 1.04879665433 | 0.0618820962465 | 11.2711486105 | -0.712121923988 |
| seed_2037/state_step_046 | 8 | -0.187717752006 | -2.75032915788 | -0.275889740975 | -0.379236380705 | -0.192524920027 | 13.6514068515 | 0.469705118598 | 1.02024782767 | 0.0256084891824 | 13.1817017329 | -0.550542709072 |
| seed_2030/state_step_046 | 2 | -0.121867887571 | -0.615779145088 | -0.122930828219 | -0.122505760615 | -0.122505760615 | 4.05284170722 | 0.0087220733022 | 0.00523413555977 | 0.00523413555977 | 4.04411963391 | 0.00348793774242 |
| seed_2030/state_step_046 | 4 | -0.213234390119 | -1.11456025261 | -0.217869509132 | -0.214351066191 | -0.21434959922 | 4.22692541286 | 0.0217372020075 | 0.00523684791878 | 0.00522996829738 | 4.20518821085 | 0.0165003540888 |
| seed_2030/state_step_046 | 8 | -0.325153506104 | -1.84582323435 | -0.34743554015 | -0.325353203041 | -0.326869252535 | 4.67677481466 | 0.0685277372943 | 0.000614162030515 | 0.005276727451 | 4.60824707736 | 0.0679135752638 |
| seed_2037/state_step_082 | 2 | -0.049851961905 | -0.206500491722 | -0.0519987564118 | -0.0510615349797 | -0.0499075718341 | 3.14227412183 | 0.0430633905826 | 0.0242632993467 | 0.00111550131525 | 3.09921073125 | 0.0188000912359 |
| seed_2037/state_step_082 | 4 | -0.0761822229814 | -0.373765890016 | -0.0905794462491 | -0.074757730486 | -0.0763000911819 | 3.90620876352 | 0.188984026775 | 0.018698489485 | 0.00154718772889 | 3.71722473675 | 0.17028553729 |
| seed_2037/state_step_082 | 8 | -0.094801363302 | -0.618993690456 | -0.141339406829 | -0.0865708332323 | -0.0950502782739 | 5.52937541081 | 0.49090057259 | 0.0868186889203 | 0.00262564759796 | 5.03847483822 | 0.40408188367 |

## Aggregate metrics over the nine H in {2,4,8} rows

Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim); layout matches FR_A0C.

| model | overall R2 | corr | MAE | RMSE | mean relative error |
|---|---:|---:|---:|---:|---:|
| A_momentum | -167.728835442 | 0.745277565937 | 0.799411692634 | 1.06531210944 | 6.3635543984 |
| B_optimizer_only | 0.838828326821 | 0.966736929314 | 0.0207034235996 | 0.0329250593449 | 0.164982733525 |
| BH_curvature_feedback | 0.232497511557 | 0.833083498466 | 0.0372716560767 | 0.0718492278627 | 0.312990916105 |
| C_full_tangent | 0.99767960167 | 0.999336137038 | 0.00256971562366 | 0.00395060383803 | 0.0274408507825 |

## Contribution decomposition

| quantity | mean over H in {2,4,8} | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|
| gain_optimizer = relA - relB | 6.55296538354 | 5.65156774845 | 6.39785385268 | 7.60947454949 |
| gain_H = relB - relBH | -0.135724742346 | -0.205879799455 | -0.175112010869 | -0.0261824167128 |
| clip_correction = relBH - relC (diagnostic) | 0.319113919336 | 0.264594240059 | 0.334690913152 | 0.358056604797 |
| total A-to-C mean relative error reduction | 6.73635456053 | | | |

A positive gain means the later model has smaller error. The three relative-error gains sum exactly to the A-to-C mean relative-error reduction, so the decomposition is complete by construction.

## Absolute-error decomposition

| quantity | mean over H in {2,4,8} |
|---|---:|
| gain_optimizer (abs error) | 0.953166336472 |
| gain_H (abs error) | -0.019126214559 |
| clip_correction (abs error) | 0.0433044918602 |

## Consistency with the frozen FR_A0C artifact

- Reproduced A/B/C row values match `response_propagation_v01/summary.json`: `True`; rows checked: 12, streams checked: 24.
- Max absolute reproduction gap by field: chi_A_momentum=0, chi_full_tangent=0, chi_optimizer_only=0, stream:chi_full_tangent=0, stream:chi_optimizer_only=0.

## Scope and method

- Same three locked v07 states, eight paired E2 streams per state, and propagation horizons H=1,2,4,8 as FR_A0C; the requested test set is H in {2,4,8} (nine state-horizon rows).
- BH shares the complete AdamW m/v/bias-correction/weight-decay tangent with B; its only addition is the future gradient feedback `delta_g = H(loss) delta_theta` evaluated on the same accumulated loss as C (reverse-over-reverse autograd, math-SDPA HVP forward, RNG snapshot/restore around the auxiliary replay).
- At H=1 all tangent models reduce to the same pulse Jacobian, so H=1 is excluded from the gain aggregates (it is retained in `results` for continuity with FR_A0C).

## Robustness, replay integrity, and limitations

- Baseline replay: `PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS`; max saved-chi aggregation gap: `0`.
- Input integrity: `PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION`; HVP backend: `PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA`.
- Max math-SDPA HVP-forward gradient relative gap vs native flash: `0.00505791896242`; this does not bound the HVP error and qualifies the smallest residuals, as in FR_A0C.
- Clipping was active on 48/168 future steps (H>1); on all inactive steps BH and C coincide exactly.
- Max |BH - C| response gap at H=8 across streams: `0.609220728188`.
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- As in FR_A0C, all tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; remaining residuals may include finite-epsilon clip/activation boundary crossings.

## Recommended next step

Use BH as the curvature-feedback reference model: it attributes gain_H without mixing in the clipping Jacobian. If the BH-to-C clip correction grows with horizon, evaluate clipping-regime crossing explicitly before fitting any memory kernel.

## Further question

Does a raw-HVP feedback model (BH) suffice at longer horizons, or does the clip-JVP correction become the dominant part of the B-to-C gain?