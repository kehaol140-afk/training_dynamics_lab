# FR A0C H=1 Complete AdamW Jacobian v01

## Technical summary

1. **H1 是否能解释真实响应：YES。** Across the three states, epsilon/4 relative errors range from 0.000146092 to 0.0189892; all quarter-epsilon errors are no larger than their full-epsilon values: True.
2. **nu 异常是否来自冻结 v 近似：YES。** For seed 2037/state 46, the old H1 v term is -0.477526116, while the complete local v-path is 0.0059423339 (absolute ratio 0.012444).
3. **是否需要 Hessian feedback：YES_FOR_H_GT_1。** H=1 has no future-gradient feedback after the post-unscale pulse. Extending the validated local map to H>1 requires propagation through later gradients, including Hessian-vector and clipping/activation boundary effects.

## Three-state result

The headline row for each state uses epsilon/4; all three epsilon levels follow below.

| state | epsilon | chi_fd | chi_jac | relative error | mu_q_old | mu_q_full |
|---|---:|---:|---:|---:|---:|---:|
| seed_2037/state_step_046 | 0.0857515572968 | -0.0550541882998 | -0.0561198604278 | 0.0189892155795 | 13187.7940145 | 1521.74394595 |
| seed_2030/state_step_046 | 0.00346871563088 | -0.0655044755819 | -0.0655277049033 | 0.000354496184288 | 8850.69939064 | 1776.84668972 |
| seed_2037/state_step_082 | 0.0101825937013 | -0.0280346022895 | -0.0280386985239 | 0.000146092174901 | 6252.07449889 | 1612.66934861 |

## Epsilon convergence

Exact lookup values are shown as a table because the experiment has only nine points; a chart would add no resolution.

| state | epsilon level | epsilon | chi_fd | chi_jac | relative error | clip active (+/base/-) |
|---|---|---:|---:|---:|---:|---|
| seed_2037/state_step_046 | epsilon | 0.343006229187 | -0.0498724233704 | -0.0561198604278 | 0.111323103973 | 1.0/0.5/0.5 |
| seed_2037/state_step_046 | epsilon/2 | 0.171503114594 | -0.0532953511216 | -0.0561198604278 | 0.0503299417482 | 0.875/0.5/0.5 |
| seed_2037/state_step_046 | epsilon/4 | 0.0857515572968 | -0.0550541882998 | -0.0561198604278 | 0.0189892155795 | 0.75/0.5/0.5 |
| seed_2030/state_step_046 | epsilon | 0.0138748625235 | -0.0651866565067 | -0.0655277049033 | 0.00520464431095 | 0.0/0.0/0.0 |
| seed_2030/state_step_046 | epsilon/2 | 0.00693743126176 | -0.0654362869913 | -0.0655277049033 | 0.00139510321788 | 0.0/0.0/0.0 |
| seed_2030/state_step_046 | epsilon/4 | 0.00346871563088 | -0.0655044755819 | -0.0655277049033 | 0.000354496184288 | 0.0/0.0/0.0 |
| seed_2037/state_step_082 | epsilon | 0.0407303748051 | -0.0279827403554 | -0.0280386985239 | 0.00199574771683 | 0.25/0.25/0.25 |
| seed_2037/state_step_082 | epsilon/2 | 0.0203651874025 | -0.0280229608644 | -0.0280386985239 | 0.000561283523607 | 0.25/0.25/0.25 |
| seed_2037/state_step_082 | epsilon/4 | 0.0101825937013 | -0.0280346022895 | -0.0280386985239 | 0.000146092174901 | 0.25/0.25/0.25 |

## Scope and definitions

- H=1 only; three preregistered v07 states; eight registered paired E2 streams per state.
- `chi_fd` keeps the original unit-response definition `(X_plus-X_minus)/(2*epsilon)` and is averaged over streams.
- `chi_jac` is the stream-mean exact local native-AdamW derivative, including clipping, updated m/v, step+1 bias correction, Adam epsilon, and actual LR.
- `mu_q_old` is the preceding raw-v projection. `mu_q_full` is the exact Jacobian projected into the same first-moment scale; it is diagnostic, not fitted.

## Complete single-step specification

The replay order is `forward -> loss -> backward -> unscale -> pulse -> clip -> optimizer.step`. The clipping JVP uses PyTorch's `max_norm/(norm+1e-6)` rule. The formula is `G0=[aM+2bV Diag(c)]JC`; here `J=I` because the pulse is injected directly in gradient coordinates after unscale. Native decoupled weight decay is executed in finite differences, while its derivative with respect to the pulse is zero at fixed input theta.

## Validation and limitations

- Input state/q/manifest integrity: `PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION`.
- Full-epsilon reproduction of saved v07 H1 branches: `PASS_EXACT`.
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- A local H1 match tests the optimizer-input Jacobian only. It does not establish H>1 dynamics, where parameter displacement changes later gradients and may cross activation or clipping boundaries.

## Recommended next step

Propagate the H1 tangent through future gradients with Hessian-vector products and retain stepwise clipping Jacobians; keep activation-boundary changes as a separate non-smooth residual diagnostic.

## Further question

For H>1, how much of the remaining response comes from smooth Hessian-vector propagation versus discrete clipping/activation boundary changes?
