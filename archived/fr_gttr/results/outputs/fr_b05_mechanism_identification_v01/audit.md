# FR B0.5 Mechanism Identification — audit.md

Date: 2026-09-20. Status: DRAFTED alongside the run; final resource numbers are
filled from `runs/gpu_budget_log.json` / driver output after completion.

## 1. Mandate and scope

B0 follow-up mandated by the user: identify the physical mechanism behind the
persistent LR-up benefit. Explicitly forbidden: new optimizer design, large-scale
experiments. Three competing hypotheses pre-registered in `config.json`:
H1 time reparameterization / path acceleration; H2 slow-fast quasi-steady
dynamics (only the separation tau_fast << tau_slow is tested); H3 optimizer
memory after-effect (m/v memory channel vs direct parameter displacement).

## 2. Interfaces actually called (verified by reading the source, not assumed)

- `a0k.build_live_runtime(seed, device)`, `a0k.make_event_ctx(...)`,
  `a0k.model_hash`, `a0k.opt_hash`
  — `research/src/fr/a0k_closed_loop_runner_v01.py`
- `pe._restore_and_seed`, `pe._baseline_no_step(..., want_snapshot=True)`,
  `pe._clip_and_step`, `pe._new_counters`
  — `research/scripts/a0h15_production_entry_v01.py:204/211/233`
- `a0g._val_loss_only` (v_eval / v_score, RNG isolated via
  `fr.rng.capture_rng_state` / `restore_rng_state`)
  — `research/scripts/a0g_decision_utility_v01.py`
- `a0q.run_channel_rollout` (frozen-runner smoke cross-check, u=0 endpoint v_score)
- `fr.stage_c_runtime.batches_from_train_pool`, `fr.stream_rng.derive_stream_seed`
  (frozen A0K stream recipe, batch_seed = derive_stream_seed(x, 404))
- `fr.full_state.save/load_full_training_state` (v0.2 full state)
- `fr.minimind_v03_runtime.raw_model` (state_dict access)
- AdamW memory accessed read-only via `optimizer.state_dict()["state"]`
  (two-level dict; in-place `copy_` used ONLY for the arm-D intervention).

The B0 driver `research/scripts/b0_run_v01.py` was used as the semantic reference
for step semantics; B0.5 re-implements the per-step loop because B0.5 needs
per-step (not per-readout-H) state captures and distances. The step sequence
(`_restore_and_seed` → `_baseline_no_step` → optional `g["lr"]=snap["lr"]*m` →
`_clip_and_step`) is copied verbatim from B0's `run_arm` and cross-checked
against the frozen runner in smoke (`frozen_runner_agreement_endpoint`, bitwise
v_score at endpoint).

## 3. Data and bank

- Frozen bank: `outputs/e3_confirmatory/v07/`, seeds 2030/2031/2032 x steps 46/82
  (= B0 main sample, continuity). One stream: 7500 (B0 frozen dev pool).
- 6 cells x 4 arms (A native / B persistent +0.05 / C pulse4_release /
  D pulse4 + m-v transplant at h=4) x 64 optimizer steps.
- Config: batch 4 x accum 2 = 1024 tokens/optimizer step; lr 5e-4 consumed per
  step (schedule-rewritten per micro-step); clip 1.0; bf16; seq 128.
  Verified in-run by `tokens_per_step_1024` smoke check.

## 4. Step semantics (identical to B0)

u = log(eta/eta_native); multiplier exp(u). Persistent: exp(u) every step.
pulse4_release: exp(u) steps 1-4 then multiplier 1.0; theta/m/v/step retained
(B0-validated). Arm D: after the h=4 step, exp_avg/exp_avg_sq of every parameter
overwritten in place with arm A's post-step-4 values (bitwise copy); theta and
the optimizer step counter untouched; AdamW step counter is identical across
arms at every h by construction (same number of steps from the same checkpoint).
Arm D is labeled an INTERVENTION arm, not a training method.

## 5. Verification status of prior results

Read and re-used as priors (not re-run): A0O report (pure theta-channel push;
e_lr = -adaptive(c) - wd*theta; m/v lr-independent at input), A0T-extra
(controller identity: Case B dominant; H=4 reversal inverts at H>=16), A0U
(schedule ~ fixed +5% delayed 6 steps; one-bit content), A0P (impulse-kernel
gate FAIL — no stable decay law; mandatory long-horizon caveat), B0 (pulse4
retains ~9.8% of persistent response at H=64; release != persistent).
The retrospective (`docs/FR_RESEARCH_RETROSPECTIVE_A0G_A0U.md`) and
`FR_B0_Mechanism_Validation_Claude_Code.md` were read in full; no doc mentions
B0.5 (fresh construction).

## 6. Known limitations / boundaries

- 3 seeds (pilot); one stream per cell; stream/timepoints are not independent
  seeds. Seed-cluster statistics only, exploratory.
- H2 estimation window is 64 steps with max lag 16; tau_slow is a lower bound
  when no crossing occurs. Only "tau_fast << tau_slow" (or absence) is claimed.
- H1 alignment is within-sample descriptive (uses the same trajectory it
  evaluates); labeled descriptive, not deployable.
- Val losses recorded every 4 steps (16 points per arm) — the alignment grid.
- Distance bookkeeping covers trainable parameters (state_dict additionally
  contains frozen rotary buffers that the optimizer never touches; excluding
  them was forced by a smoke-caught shape mismatch, see below).
- H2 smooth component implemented as a centered moving average of velocities
  (w=9, edge-shrinking window), equivalent to differentiating the moving
  average of theta up to boundary handling; torch-native autocorrelation
  mirrors the CPU-tested python reference in fr.b05_mechanism_v01.

## 7. Failure evidence and fixes (chronological)

1. CPU stage-B tests initially failed 2/14: `test_pure_reparameterization_aligns`
   (accelerated arm legitimately clamps at the right edge of the native s-curve;
   metric must exclude clamped points) and `test_implied_lag_constant_for_pulse`
   (exact lag constancy holds only under uniform native LR). Fixed by
   masked_rel_err + exact inversion-based H1 statement; config amendment B05-A1
   (pre-GPU). Evidence: test run logs in session; amendment recorded in
   config.json.
2. First GPU smoke attempt: RuntimeError shape mismatch (6,621,696 vs 4,983,296)
   — state_dict includes 1.64M frozen buffer elements absent from the trainable
   set used for update vectors. Distances/cosines restricted to trainable
   parameters. Re-run: 14/14 PASS. Failure output preserved in
   `runs/smoke_stdout.log` (first run's traceback was overwritten by the second
   run's log; the traceback text is quoted in this audit).
3. Smoke check `distances_native_zero` first read FAIL 13/14 despite all-zero
   values: the check re-opened the CSV while the writer buffer was unflushed.
   Fixed with fstep.flush() before re-read; 14/14 PASS. Not a science failure;
   kept here as execution evidence.

## 8. Resource accounting (final)

- B0 (prior, frozen): 14,208 / 18,000 optimizer updates.
- B0.5 smoke attempt 1: 9 updates (aborted by the shape-mismatch bug below).
- B0.5 smoke (passing): 64 updates + 26 eval calls + 56 theta/opt captures.
- B0.5 main: 1,536 updates + 768 eval calls + 1,536 captures.
- B0.5 total: **1,609 updates, 802 evals, 3,202 captures** (all counted;
  evals/captures are overhead, never hidden).
- Project cumulative: **15,817 / 18,000 (87.9%)**.
- `gpu_budget_log.json` TOTALS was patched post-hoc across the three driver
  processes (each overwrote TOTALS with its own BUDGET); the patch note and
  per-phase entries are preserved in the file. Wall time: ~14 min for the main
  6-cell phase on one RTX 5090.

## 9. Numerical determinism evidence

- `snapshot_identity`, `crn_native_repeat`, `frozen_runner_agreement_endpoint`
  (bitwise v_score), `val_eval_rng_isolation` (bitwise theta endpoint with vs
  without mid-run evals), `distances_native_zero` (exact zeros) — all PASS in
  smoke before any main result.
- Main-sample implementation identities asserted in smoke only (T=8); the same
  code paths run at T=64. `pulse4_eq_persistent_H4` (losses) and
  `D_mv_eq_A_mv_bitwise` validate the arm semantics at the transplant step.
