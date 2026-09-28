# FR A0C Complete AdamW Response Propagation v01

## Technical summary

1. **H=1 Jacobian 能否扩展到多步：YES_WITH_HVP_SURROGATE_AND_QUANTIFIED_RESIDUAL。** At H>1, mean relative error is A=6.75822, B=0.205255, C=0.0218656; C beats B in 9/9 state-horizon rows. The largest H=1 C error at registered epsilon is 0.125268. C uses a math-SDPA HVP surrogate because native flash-SDPA double backward is unavailable.
2. **失败来自 optimizer memory 还是 curvature feedback：CURVATURE_FEEDBACK_EXPLAINS_B_RESIDUAL;A_TO_B_NOT_MEMORY_IDENTIFIED。** B-to-C isolates future curvature feedback and reduces H>1 mean relative error by +0.183389. A-to-B reduces it by +6.55297, but that contrast also changes the local preconditioner, bias correction, clipping, LR schedule, weight decay, and current-gradient dependence, so it does not identify optimizer memory alone.
3. **是否需要 Mori-Zwanzig memory kernel：DEFER_PENDING_MULTISTEP_EPSILON_CONVERGENCE。** The full smooth tangent leaves H>1 mean relative error 0.0218656. Because the target uses finite registered epsilon and can cross clipping boundaries, first repeat only an epsilon-halving response check before attributing this residual to an unresolved memory kernel.

## A/B/C comparison across the three states

Each row is a unit post-unscale pulse response. `rel A/B/C` is absolute prediction error divided by `|chi_fd|`.

| state | H | chi_fd | A momentum | B optimizer-only | C full tangent | rel A | rel B | rel C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_2037/state_step_046 | 1 | -0.0498724233704 | -0.482909712342 | -0.0561198622662 | -0.0561198622662 | 8.68290048301 | 0.125268404332 | 0.125268404332 |
| seed_2037/state_step_046 | 2 | -0.0831821646531 | -0.91752845345 | -0.101397219059 | -0.090524686705 | 10.0303507642 | 0.218977884042 | 0.0882703892419 |
| seed_2037/state_step_046 | 4 | -0.131721904397 | -1.66072650074 | -0.176069341039 | -0.139873131962 | 11.6078233408 | 0.336674730338 | 0.0618820962465 |
| seed_2037/state_step_046 | 8 | -0.187717752006 | -2.75032915788 | -0.275889740975 | -0.192524920027 | 13.6514068515 | 0.469705118598 | 0.0256084891824 |
| seed_2030/state_step_046 | 1 | -0.0651866565067 | -0.324094286889 | -0.0655277072244 | -0.0655277072244 | 3.97178877176 | 0.00523190996332 | 0.00523190996332 |
| seed_2030/state_step_046 | 2 | -0.121867887571 | -0.615779145088 | -0.122930828219 | -0.122505760615 | 4.05284170722 | 0.0087220733022 | 0.00523413555977 |
| seed_2030/state_step_046 | 4 | -0.213234390119 | -1.11456025261 | -0.217869509132 | -0.21434959922 | 4.22692541286 | 0.0217372020075 | 0.00522996829738 |
| seed_2030/state_step_046 | 8 | -0.325153506104 | -1.84582323435 | -0.34743554015 | -0.326869252535 | 4.67677481466 | 0.0685277372943 | 0.005276727451 |
| seed_2037/state_step_082 | 1 | -0.0279827403554 | -0.108684469327 | -0.0280386989096 | -0.0280386989096 | 2.88398233864 | 0.00199975247338 | 0.00199975247338 |
| seed_2037/state_step_082 | 2 | -0.049851961905 | -0.206500491722 | -0.0519987564118 | -0.0499075718341 | 3.14227412183 | 0.0430633905826 | 0.00111550131525 |
| seed_2037/state_step_082 | 4 | -0.0761822229814 | -0.373765890016 | -0.0905794462491 | -0.0763000911819 | 3.90620876352 | 0.188984026775 | 0.00154718772889 |
| seed_2037/state_step_082 | 8 | -0.094801363302 | -0.618993690456 | -0.141339406829 | -0.0950502782739 | 5.52937541081 | 0.49090057259 | 0.00262564759796 |

## Full tangent isolates curvature feedback from optimizer-only propagation

The compact table is preferable to a trend chart for twelve exact audit points. Model B carries all optimizer memory while setting future gradient feedback to zero; the B-to-C change therefore isolates the smooth Hessian-feedback contribution along the same baseline streams. A→B does not isolate optimizer memory: A also omits the exact local preconditioner, bias correction, clipping Jacobian, future LR schedule, weight decay tangent, and current-gradient dependence.

| model | overall R2 | corr | MAE | RMSE | mean relative error |
|---|---:|---:|---:|---:|---:|
| A_momentum | -167.728835442 | 0.745277565937 | 0.799411692634 | 1.06531210944 | 6.3635543984 |
| B_optimizer_only | 0.838828326821 | 0.966736929314 | 0.0207034235996 | 0.0329250593449 | 0.164982733525 |
| C_full_tangent | 0.99767960167 | 0.999336137038 | 0.00256971562366 | 0.00395060383803 | 0.0274408507825 |

## Scope and response definition

- Three locked confirmatory v07 states, eight paired E2 streams per state, and horizons H=1,2,4,8.
- `chi_fd` is the original registered finite-epsilon central response `(X_plus-X_minus)/(2*epsilon)` averaged over streams; epsilon is not multiplied into any prediction.
- All predictions are derivatives per unit pulse in the exact state-specific q direction used by the original measurement.
- Model A preserves the historical first-future-step LR and raw-v `mu`; B and C use each future optimizer step's scheduled LR and complete state.

## Complete optimizer propagation with a qualified math-SDPA HVP

For every stream the complete checkpoint is restored, the registered RNG is seeded, and the registered future batches are replayed with BF16 autocast and two-microbatch accumulation. The baseline order is `forward -> loss -> backward -> unscale -> clip -> optimizer.step`. At the pulse step both tangents use `delta_g=q`. Later B uses `delta_g=0`, while C obtains `H(loss) delta_theta` from the same accumulated loss using reverse-over-reverse autograd. The optimizer-side JVP is exact: both directions pass through the native global clipping rule and complete bias-corrected AdamW m/v/parameter update, including Adam epsilon and decoupled weight decay. PyTorch does not implement double backward for the native flash-SDPA kernel, so only the auxiliary HVP forward uses math-SDPA; C is therefore a mixed-backend analytic surrogate, not an exact derivative of the flash kernel implementation.

## Robustness, replay integrity, and limitations

- Baseline replay: `PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS`; max saved-chi aggregation gap: `0`.
- Input integrity: `PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION`; HVP backend: `PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA`.
- Native flash double backward: `UNSUPPORTED_DERIVATIVE_FOR_FLASH_ATTENTION_BACKWARD`. The maximum relative gap between the math-SDPA HVP forward's first gradient and the native flash gradient is `0.00505791896242`; this does not bound the HVP error and qualifies C's smallest residuals.
- Whole-state replay uses a fresh existing-protocol H=8 shadow branch: propagated/shadow model and optimizer hashes match on 24/24 streams. The original v07 final hashes are H=32 and are retained as provenance but are not compared to an H=8 state.
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- C is a zero-pulse smooth tangent, while `chi_fd` uses the registered finite epsilon. A discrepancy may include finite-epsilon clip/activation boundary crossings; this is especially relevant to seed 2037/state 46, whose H=1 epsilon-convergence audit showed a smaller error at epsilon/4.
- A Hessian-vector product captures smooth curvature along the baseline trajectory. It does not represent discrete changes of clipping regime or other non-smooth branch changes between finite ±epsilon trajectories.

## Recommended next step

Run the same H=2,4,8 finite differences at epsilon/2 and epsilon/4 for these three states without changing the tangent model. If the residual converges to C, no Mori-Zwanzig term is needed at H<=8; if a stable residual remains, fit no kernel yet—first project that residual against candidate omitted state variables.

## Further question

Does the C residual converge to zero with pulse amplitude, or remain finite after clipping-regime changes disappear?
