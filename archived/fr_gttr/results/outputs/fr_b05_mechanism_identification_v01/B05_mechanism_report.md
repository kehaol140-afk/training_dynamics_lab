# FR B0.5 — Mechanism Identification Report

**Date:** 2026-09-20
**Output dir:** `outputs/fr_b05_mechanism_identification_v01/`
**Config (frozen before any B0.5 GPU result):** `config.json` (amendments B05-A1, B05-A2, both pre-GPU)
**Runner / analyzer:** `research/scripts/b05_run_v01.py` / `b05_analyze_v01.py`
**Analysis math:** `research/src/fr/b05_mechanism_v01.py` (14 CPU tests, PASS, run before any GPU)
**Audit:** `audit.md` (interfaces, failure evidence, fixes)

---

## 0. What was asked

Identify the physical mechanism behind the persistent LR-up benefit (A0T/B0
phenomenon: multiplying the learning rate by exp(+0.05) at every step lowers
validation loss over a 64-step window). Three competing hypotheses were
pre-registered with frozen decision rules. No new optimizer was designed; no
large-scale experiment was run.

The system under study is a **controlled non-equilibrium dynamical system**
(training step z_{t+1} = F(z_t, u_t) with z = (theta, m, v), u = log-LR input),
observed through a **response function** (validation loss readout). All claims
below are bounded by the recorded 64-step horizon (A0P long-horizon caveat:
short-window rollout gains must not be extrapolated to final training).

## 1. Experiment

- **Sample:** 6 cells = seeds {2030, 2031, 2032} x bank steps {46, 82}
  (identical to B0's main sample for continuity; no cherry-picking by response
  sign). One frozen development stream (7500) per cell. Pilot scale: 3 seeds,
  one stream — seeds are the only independent clusters.
- **Arms (same checkpoint, same CRN stream within a cell):**

| arm | protocol |
|---|---|
| A native | u = 0 for all 64 steps; also the reference trajectory |
| B persistent | u = +0.05 (multiplier exp(0.05) = 1.0513) at every step |
| C pulse4_release | u = +0.05 at steps 1-4, native afterwards; state retained |
| D transplant (**intervention arm, not a training method**) | identical to C through step 4; after the step-4 optimizer step, optimizer memory (exp_avg, exp_avg_sq) of every parameter overwritten in place with arm A's post-step-4 values; theta and the optimizer step counter untouched; steps 5-64 at native LR |

- **u semantics:** u = log(eta/eta_native), eta(k;u) = eta_native(k)*exp(u).
  1 optimizer step = 2 microbatches = 1024 tokens (verified in-run).
- **Recordings (every step, every arm):** ||theta_t - theta_t^A||_F,
  ||m_t - m_t^A||_F, ||v_t - v_t^A||_F, train loss, grad norm, update norm,
  cos(theta_arm - theta^A, native update); validation loss (v_eval, v_score)
  every 4 steps; native-arm velocity structure for H2 (lags 1-16).
- **Verification:** 14/14 smoke checks PASS before any main run (snapshot
  identity, CRN repeat, frozen-runner bitwise agreement, self-distance exact
  zeros, eval RNG isolation, pulse==persistent at H=4, D-transplant bitwise
  state at h=4, s(t) monotonicity, frozen excess LR time post-release, token
  count, budget). Two pre-main failures were caught and fixed (see audit.md
  sec.7): a state-dict/buffer shape mismatch and a too-strict H1 test
  formulation (config amendment B05-A1).
- **Budget:** 1,609 optimizer updates (smoke attempt 1: 9; smoke: 64; main:
  1,536) + 802 eval calls + 3,202 state captures (overhead, counted).
  Project cumulative including B0: **15,817 / 18,000 (87.9%)**.

## 2. Hypothesis table (frozen decision rules in config.json)

| # | Hypothesis | Equation / construction | Decision rule (frozen) |
|---|---|---|---|
| H1 | Time reparameterization / path acceleration: persistent LR-up = native trajectory traversed faster; aligning by cumulative LR time s(t) = sum_k eta_k collapses the curves | s_A from native lr; L̃_X(t) = L_A(s_A^{-1}(s_X(t))) by monotone interpolation (clamped tail excluded); alignment_rel_err = ||L_X - L̃_X|| / ||L_X - L_A|| on the val grid; pulse: Delta_C(t) = L_C(t) - L_A(t), implied lag l(t) = s_A^{-1}(s_C(t)) - t | SUPPORTED iff alignment_rel_err(B) < 0.5 in >=5/6 cells AND pulse lag approx constant AND Delta_C not decayed in >=5/6 cells; NOT_SUPPORTED iff B alignment <=1/6 cells or Delta decays in >=5/6 |
| H2 | Slow-fast separation: velocity = smooth component + residual; test only whether tau_fast << tau_slow (no stationary-state claim) | smooth = centered w=9 moving average of velocities; tau_fast = 1/e crossing of residual velocity autocorrelation C_v(D), D<=16; tau_slow = 1/e crossing of smooth-direction persistence mean_t cos(s_t, s_{t+D}) | SUPPORTED iff tau_fast observed in >=5/6 cells and tau_fast < tau_slow/4; NOT_SUPPORTED iff tau_fast unobserved in >=4/6 or tau_fast >= tau_slow/2 in >=4/6 |
| H3 | Optimizer memory after-effect: the retained post-release benefit of pulse4 is carried mainly by (m, v) | at h >= 5, L_C - L_A = (L_D - L_A) + (L_C - L_D); G_X = L_A - L_X; m/v-channel share = (G_C - G_D)/G_C at H = 16, 64 (v_score) | SUPPORTED iff share > 0.5 in >=5/6 cells; NOT_SUPPORTED iff share < 0.2 in >=5/6 cells |

## 3. Results

### 3.1 H1 — time reparameterization: **NOT_SUPPORTED**

Persistent-arm alignment relative error by cell:
{2.54, 0.64, 3.55, 0.76, 1.14, 0.90} — **0/6 cells below 0.5**; in 4/6 cells the
reparameterized prediction is *worse* than the same-time baseline (rel err > 1).
Figure f2 (left) shows the signature: the accelerated-time prediction
L_A(s_A^{-1}(s_B(t))) systematically deviates from the persistent arm's actual
loss — the persistent arm is not the native trajectory sped up.

Pulse arm: the implied post-release lag is small (mean 0.47-0.49 steps) but not
cleanly constant (range 0.68-1.0 steps across cells, reflecting genuine
non-uniformity of the native LR schedule); the excess-LR-time bookkeeping is
exact (smoke check `pulse_excess_frozen_post_release`). Delta_C(t) does not show
a consistent frozen shift: delta_64/delta_8 = {2.25, 0.78, -0.32, 2.31, 2.78,
0.82} — |Delta| grows or holds in 5/6 cells (it does not decay to zero), but the
inversion-based alignment still fails (rel err {2.57, 1.18, 2.59, 0.97, 0.80,
2.67}). **Label: NOT_SUPPORTED** by the frozen rule; the after-effect is not a
constant-lag time shift either.

### 3.2 H2 — slow-fast separation: **NOT_SUPPORTED** (as measured)

Per cell (identical across all 6 cells within the recorded resolution):

| quantity | value | interpretation |
|---|---|---|
| residual velocity autocorrelation at lag 1 | 0.89-0.90 | fast component correlation time ~9-10 steps (1/e crossing at lag 9-10) |
| tau_fast (C_v(resid) < 1/e) | 9-10 steps | observed in 6/6 cells |
| tau_slow (direction persistence of w=9-smoothed velocity < 1/e) | 5 steps | observed in 6/6 cells — *shorter* than tau_fast |
| smooth-direction persistence at lag 8-9 | **-0.06 ... -0.19** | the smoothed direction reverses at lag ~9 |
| smooth-component velocity share | ~11% of velocity norm (median ||s||/||v|| = 1.9e-2 / 1.7e-1) | no dominant slow drift component in-window |

There is **no separation tau_fast << tau_slow**; if anything the ordering is
inverted within this window. Claim boundary: this falsifies the *specific*
w=9-scale slow-fast hypothesis as operationalized; it does not by itself rule
out structure at scales longer than the 64-step window (that is NOT_TESTED
here). No stationary-state claim of any kind is made.

### 3.3 H3 — optimizer memory after-effect: **NOT_SUPPORTED** (memory channel rejected; direct parameter displacement carries the retained effect)

Gains vs native at h=64 (v_score; median persistent gain G_B = 4.6e-3,
consistent with A0T's frozen-bank scale):

| cell | G_B | G_C (pulse) | G_D (transplant) | m/v-channel share (G_C-G_D)/G_C |
|---|---|---|---|---|
| s2030_t46 | 5.26e-3 | 5.90e-4 | 4.55e-4 | 0.23 |
| s2030_t82 | 4.11e-3 | 3.28e-4 | 3.75e-4 | -0.14 |
| s2031_t46 | 4.07e-3 | 6.79e-5 | 3.01e-4 | -3.44 (G_C at resolution floor) |
| s2031_t82 | 4.03e-3 | 4.98e-4 | 5.35e-4 | -0.07 |
| s2032_t46 | 5.26e-3 | 4.01e-3 | 3.35e-3 | 0.17 |
| s2032_t82 | 4.33e-3 | 3.87e-4 | 6.15e-4 | -0.59 |

- **G_D tracks G_C in every cell** (max |G_D - G_C| = 2.2e-4, mostly far less).
  Resetting the optimizer memory (m, v) to the native trajectory's values at
  h=4 does **not** remove the pulse arm's retained gain. m/v-channel share
  < 0.2 in 5/6 cells (the -3.44 outlier has G_C at the resolution floor;
  |share| there is meaningless as a ratio but the absolute gains still agree
  within 2.3e-4). **The retained separation acts through the theta-displacement
  channel (L_D - L_A), not through the m/v memory channel (L_C - L_D).**
- Distance evidence (medians): after the transplant, arm D's optimizer memory
  re-diverges from native almost as fast as arm C's — d_m at h=8: 3.8e-3 (D)
  vs 5.0e-3 (C); at h=64: 6.5e-3 vs 7.2e-3. d_v stays ~1000x smaller than d_m
  throughout (2e-6 scale): the memory channel, where it acts at all, is
  carried by m. Within ~12 steps of the transplant, D's m is nearly as far
  from native as C's — the transplanted memory is overwritten by the running
  gradient statistics on a short horizon.
- Direction of separation: cos(theta_arm - theta_A, native update) median for
  C/D decays 0.63 (h=8) -> 0.30 (h=16) -> 0.09-0.13 (h=32-64): post-release
  the separation progressively decorrelates from the native update direction
  while ||delta theta|| keeps growing (f3, f6) — the pulse pushes the
  parameters onto a diverging path rather than along the native one.

### 3.4 Auxiliary facts

- Persistent arm separation: d_theta grows to ~0.1-0.9 by h=64 (cell-dependent),
  cos_sep 0.81 -> 0.44: the persistent benefit accumulates along a direction
  that stays partially aligned with the native update but increasingly deviates.
- clip was active on 48% of all steps (735/1536) — the clip channel is
  event-driven in this regime (consistent with A0Q), so LR-response theory's
  clip-JVP structure matters.
- Train loss, grad norm (median 0.99, near the clip threshold 1.0), and update
  norm (median 7.4e-2) trajectories are recorded per arm in
  `runs/step_records.csv`.

## 4. Falsification evidence (summary)

| hypothesis | falsifying observation | strength |
|---|---|---|
| H1 path acceleration | alignment_rel_err(B) >= 0.64 in all 6 cells, > 1 in 4/6; reparametrized curve visibly deviates (f2) | rejected by frozen rule (0/6 cells) |
| H1 constant-lag shift for the after-effect | pulse inversion alignment fails (rel err 0.80-2.67); implied lag not constant (range up to 1.0 step) | rejected as a pure-shift description |
| H2 tau_fast << tau_slow | tau_slow (5 steps) < tau_fast (9-10 steps) in 6/6 cells; smoothed direction anti-persistent at lag 9 | rejected as operationalized (w=9, 64-step window) |
| H3 m/v-memory-carried after-effect | transplant of native m/v at h=4 leaves G_C essentially unchanged (share < 0.2 in 5/6 cells); D's m re-diverges within ~12 steps | rejected by frozen rule |

## 5. Which physical model survives

None of the three hypotheses survives as stated. The measured facts that any
subsequent model must reproduce:

1. **The persistent benefit is a continued-input effect, not a state
   after-effect and not a reparameterization.** Release semantics matter (B0):
   after the input stops, the retained benefit is small (here 1.6%-76% of
   persistent, cell-dependent) and acts through the parameter displacement
   accumulated during the pulse, not through optimizer memory.
2. **The optimizer memory channel (m, v) has a short kernel at this scale.**
   Transplanted native m/v is overwritten by running gradient statistics within
   ~12 steps; v barely diverges at all (d_v ~ 2e-6). In memory-kernel language:
   the projection of the post-release dynamics onto the (m, v) state has a
   fast-decaying kernel relative to the 64-step window; the slow part of the
   retained separation lives in theta. (Memory-kernel vocabulary is used here
   only as a descriptive decomposition; no closure or stationarity is claimed.)
3. **No slow-fast separation is measurable at the w=9 scale within 64 steps.**
   Velocities are dominated by a single fast component (correlation time
   ~9-10 steps) and an oscillatory, non-persistent smoothed direction
   (reversal at lag ~9). A quasi-steady state is **not** claimed; whether any
   slower structure exists beyond the recorded horizon is NOT_TESTED.
4. The post-release separation decorrelates from the native update direction
   (cos -> 0.1 by h=64): the pulse steers the trajectory transversally. This is
   compatible with B0's finding that the long-horizon response is largely
   carried by the inter-stream dispersion structure and with A0T-extra's
   non-persistent horizon-dependent reversals, and with A0P's negative kernel
   result (no stable temporal law extends the response beyond its horizon).

Most defensible one-line statement: **on this bank, at 5M scale and 64-step
horizon, the persistent LR-up response function of this controlled
non-equilibrium dynamical system is carried by continued direct parameter
displacement along a partially transversal direction; the optimizer memory
state contributes a fast-decaying channel and does not store the benefit.**

## 6. Verdict labels (per config.json rules)

| conclusion | label | basis |
|---|---|---|
| Arm semantics / numerical identities (pulse==persistent at H=4; D m/v == A m/v bitwise at h=4; D theta == C theta; s-bookkeeping; RNG isolation) | **PROVED** | smoke 14/14, bitwise |
| H1 time reparameterization | **NOT_SUPPORTED** | frozen rule met for rejection (0/6 alignment cells) |
| H1 constant-lag after-effect | **NOT_SUPPORTED** (as pure shift) | inversion alignment fails; lag not constant |
| H2 tau_fast << tau_slow (w=9, 64-step window) | **NOT_SUPPORTED** | tau_slow < tau_fast in 6/6 cells |
| H3 m/v-memory-carried retained gain | **NOT_SUPPORTED** | share < 0.2 in 5/6 cells; G_D tracks G_C |
| "Retained pulse benefit acts through theta displacement (D-A channel)" | **SUPPORTED** (descriptive, within-sample) | consistent with H3 channel decomposition in 6/6 cells by absolute gains; no pre-registered rule attached — descriptive label |
| "Memory kernel of (m,v) decays within ~12 steps post-transplant" | **SUPPORTED** (descriptive, within-sample) | d_m^D(t) re-divergence curve |
| Longer-horizon (>64 steps) structure; other banks/scales; final-training effect | **NOT_TESTED** | out of scope by mandate |
| Quasi-steady state / any stationary description | **NOT_CLAIMED** | no supporting measurement exists |

## 7. Limits

- Pilot: 3 seeds x 2 steps x 1 stream; seed-cluster statistics only; no
  small-p claims. Shares near cells with G_C at the resolution floor (s2031_t46)
  are reported as absolute gains, not ratios.
- H2's window (64 steps, max lag 16, w=9) cannot see structure slower than the
  window; the H2 verdict is explicitly "as operationalized".
- H1 alignment uses the same trajectory it evaluates (within-sample,
  descriptive; labeled as such in summary.json).
- All gains are short-window rollout gains on a mid-training segment
  (mandatory A0P/A0S caveat).

## 8. Terminology compliance

This report uses only: controlled non-equilibrium dynamical system, response
function, memory kernel (descriptive decomposition only), quasi-steady state
(only as NOT_CLAIMED). It does not use: temperature, free energy, entropy
production, equilibrium, detailed balance, field theory. Compliance checked by
inspection; the analysis code contains no such strings.

## 9. Artifacts

- Raw tables: `runs/step_records.csv` (6 cells x 4 arms x 64 steps),
  `runs/val_curves.csv`, `runs/h2_curves.csv`, `runs/cells.jsonl`,
  `runs/smoke_checks.json` (14/14), `runs/gpu_budget_log.json` (patched TOTALS
  with note).
- Analysis tables: `h1_alignment.csv`, `h2_timescales.csv`, `h3_channels.csv`,
  `summary.json`.
- Figures: `figures/f1_val_curves.png` (all arms), `f2_h1_alignment.png`,
  `f3_distances.png` (d_theta/d_m/d_v), `f4_h2_curves.png`,
  `f5_h3_gains.png` (channel gains), `f6_cos_sep.png`.
- Reproduce: `PYTHONPATH=research/src python research/scripts/b05_run_v01.py
  --mode smoke|main` then `python research/scripts/b05_analyze_v01.py`;
  CPU tests: `python -m unittest research.tests.test_b05_mechanism_v01`.
