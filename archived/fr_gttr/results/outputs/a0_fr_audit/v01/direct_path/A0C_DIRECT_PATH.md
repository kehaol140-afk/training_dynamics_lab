# A0c AdamW Direct-Memory Diagnostic

A0C_DATA_INTEGRITY = PASS

## Frozen-preconditioner direct baseline

- Includes real future optimizer-step LR, beta1 memory decay, first-moment bias correction, state-specific frozen vhat preconditioner, and decoupled weight-decay propagation.
- Excludes clipping Jacobian, perturbation-induced v change, Hessian/future-gradient feedback, activation switching.
- Therefore this is a **diagnostic baseline**, not exact causal response.

BETA1 = 0.9
BETA2 = 0.999
H8_ORIGINAL_K = 0.3768608979
H8_RESIDUAL_K = 0.6842526705
H8_ORIGINAL_R = 0.5405571946
H8_RESIDUAL_R = 0.8771563195
H8_ORIGINAL_LOSO = mean_gain 0.118858%, 4/8
H8_RESIDUAL_LOSO = mean_gain 55.282484%, 8/8
DIRECT_KERNEL_MEDIAN_SHAPE_CORR = 0.9854904658
DIRECT_PATH_EXPLAINS_MOST = PARTIAL
K_ABS_REDUCTION = -0.815664
R_ABS_REDUCTION = -0.622689

## Boundary

- [FR-METRIC] 不用本诊断改写 v0.7 confirmation。
- 若 clipping first-step 高，unclipped direct baseline 的解释必须降级。
