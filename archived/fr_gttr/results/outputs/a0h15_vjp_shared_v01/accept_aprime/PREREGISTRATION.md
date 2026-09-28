# A0H-1.5 Round-2 A′ Acceptance Pre-Registration (amendment DEV_TIMING §A′)

Frozen BEFORE any A′ exposure (no GPU work has touched the A′ units at the time
of writing this file). This registration is append-only evidence; the retry rules
below were fixed prior to opening A′.

## Pre-registered units (exactly 3, ≤3 cap)

| unit | seed/step | status check |
|---|---|---|
| A′-1 | 2030/52 | not in D {2031/52, 2034/64, 2036/76}; not in demoted batch A {2032/58, 2035/70, 2037/46}; not a round-2 dev/timing unit (all round-2 GPU work used 2031/52 only); stream-0 manifest present in v07 bank |
| A′-2 | 2033/64 | same checks |
| A′-3 | 2037/82 | same checks |

Selection rule (fixed): spread across three previously untouched seeds, steps
{52, 64, 82}; the first 3 units in that fixed enumeration order.

## Frozen failure / retry rules

1. One retry is allowed for a unit ONLY if the failure is a pre-exposure
   infrastructure failure (process crash, CUDA OOM, or manifest/identity
   validation error raised BEFORE any numeric output for that unit is written).
2. Any code change to scoring/gate/replay logic after the first A′ exposure
   → A′ is demoted to dev-investigation for this round; NO A″ this round.
   (The two production-entry bugs found during ENGINEERING_ELIGIBILITY were
   fixed and verified BEFORE this registration; entry code is frozen from here.)
3. Numeric failures (L1 gate, ranking disagreement, non-positive utility) are
   reported honestly with full numbers; they do not trigger retries.

## Acceptance criteria (two layers, per unit)

- L1 numerics (FP32-HVP shared adjoint as candidate config):
  - B1 same-config gate: for every (H, j) in {1,4,8} × {1,2,3,4}:
    |hhat_fwd_C − hhat_adj| ≤ atol32 + rtol32·|hhat_fwd_C|, atol32=1e-5, rtol32=1e-4.
    Both quantities from the SAME hvp_fp32=True a0h15 pass on the unit.
- L1 decision consistency vs BF16 C:
  - BF16 forward ĥ (hhat_fwd_C from a separate hvp_fp32=False pass, the frozen
    precision config) ranking at H=4 equals FP32 adjoint ranking at H=4;
    argmax-ĥ action identical. H=1, H=8 rankings also reported.
- L2 actual utility:
  - Chosen action (FP32 adjoint argmax-ĥ at H=4) realized utility via native
    +λ replay (λ=0.05 frozen): V_score gain vs no-op branch > 0 at H=4
    (choose_noop=False). V_eval gain for the chosen action reported as the
    held-out secondary check (frozen a0g _run_action_branch, read-only call).
- Unit verdict: PASS iff L1 gate PASS AND H=4 chosen-action agreement AND
  L2 V_score gain > 0. Otherwise FAIL with numbers.

## Frozen pipeline parameters (unchanged from contract)

HORIZONS=(1,4,8), M=4 (all four data-group candidates), EVAL_BATCH_SIZE=8,
FROZEN_LAMBDA=0.05, atol32=1e-5, rtol32=1e-4, original runners/configs read-only.

## Work products (new directory, no overwrites)

outputs/a0h15_vjp_shared_v01/accept_aprime/ — per-unit diag JSONs (bf16 + fp32),
replay utility JSON, verdict JSON, this preregistration copy.
