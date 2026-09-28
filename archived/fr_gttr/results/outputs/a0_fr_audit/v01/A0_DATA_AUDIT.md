# A0 Data Audit

A0_DATA_INTEGRITY = PASS

## Core definitions

- Y_H: `X_theta(H)=<q_t, theta_(t+H)-theta_t>`
- B: `Cov(delta s,Y)/Var(delta s), s=<q_t,g_first>`
- chi: `mean_stream[(X_plus-X_minus)/(2 epsilon)]`
- E1 streams/state: `[16]`
- E2 streams/state: `[8]`
- causal CRN: `E2_BASELINE_PLUS_MINUS_SHARE_STREAM_SEED_BATCHES_AND_LR_PATH_WITHIN_STREAM`
- natural future semantics: `E1_EACH_STREAM_HAS_OWN_STREAM_SEED_AND_BATCH_MANIFEST_THROUGH_HORIZON`
- sampler: `WITHOUT_REPLACEMENT_SINGLE_PERMUTATION_OF_TRAIN_POOL`
- rho: `[0.25]`
- grad_clip: `1.0`
- beta1/beta2: `(0.9, 0.999)`
- AdamW eps: `1e-08`
- weight_decay: `0.01`
- scaler source inference: `BF16_DISABLED_BY_SOURCE`
- E1 per-stream future X saved: `False`

## Boundary

- [FR-OBSERVATION-ONLY] CPU/read-only audit。
- v0.7 H8_CONFIRMATION=FAIL 保持不变。
- 2030..2037 不再视为新的 held-out confirmation。
