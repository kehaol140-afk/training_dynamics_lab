# FR B0.6 — Regime Dynamics / Switching Linear Response Report

**Date:** 2026-09-21
**Output dir:** `outputs/fr_b06_regime_dynamics_v01/`
**Config (frozen before any B0.6 GPU result):** `config.json`
**Live manifest:** `live_manifest.json` (seq 0–10, all before the corresponding launch)
**Recorder / gate / analyzer:** `research/scripts/b06_record_psi_v01.py` /
`b06_gate_v01.py` / `b06_analyze_v01.py`
**Analysis math:** `research/src/fr/b06_regime_dynamics_v01.py` (23 CPU tests,
PASS in 298 s, run before any GPU launch)
**CRN gate:** 26 trajectories × 64 steps bitwise equal to frozen B0 diag
native rows (`runs/gate.json`, ALL_PASS)

---

## 0. What was asked

B0-series follow-up. B0 established that short-horizon response is accurate but
long-horizon benefit fails; B0.5 established that the retained after-effect is
carried by direct parameter displacement with a fast (~12-step) optimizer-state
memory kernel — i.e. the failure is not in (m, v) memory. The remaining
explanation class: the local dynamical regime itself switches during training,
so a single tangent/linear response model is locally valid but globally
invalid. B0.6 tests whether transformer training exhibits **metastable
dynamical regimes / switching dynamics** and, if so, whether a **switching
linear response** model (per-regime local maps) explains what a single global
map cannot. No new model training; native replay from frozen v07 checkpoints
only, bitwise-gated against the frozen B0 diagnostic rollout.

The system under study is a **controlled non-equilibrium dynamical system**
(training step z_{t+1} = F(z_t, u_t), z = (θ, m, v), u = log-LR input),
observed through training-loss and gradient/update statistics. All claims are
bounded by the recorded 64-step horizon (A0P long-horizon caveat).

Terminology discipline (frozen): this report uses **metastable regime**,
**switching dynamics**, **non-stationary reduced model**, **switching linear
response** only. No claim of a physical phase transition is made.

## 1. State representation

Per trajectory (one nested native 64-step rollout from a frozen bank
checkpoint), the reduced state ψ_t ∈ R^9 is built from recorder channels
(`psi_channels_from_rows`):

| channel | definition |
|---|---|
| log_g | log ‖g_t‖ (post-unscale native gradient norm) |
| log_upd | log ‖θ_{t+1} − θ_t‖ over the named trainable subset |
| log_m | log ‖m_t‖ (Adam exp_avg, trainable subset) |
| log_v | log ‖v_t‖ (Adam exp_avg_sq, trainable subset) |
| g_cos | cos(g_t, g_{t−1}); 0.0 at t = 1 by convention |
| upd_cos | cos(Δθ_t, Δθ_{t−1}); 0.0 at t = 1 by convention |
| log_disp | log ‖θ_t − θ_0‖ from trajectory start (trainable subset) |
| loss_slope | train_loss(t) − train_loss(t−1); 0.0 at t = 1 |
| log_preclip | log pre-clip gradient norm |

`lr_native` (a controlled schedule input) and the `train_loss` level
(cell/step-specific) are recorded but excluded from clustering channels.
B0.5 lesson applied: all norms/cosines use the named trainable subset; the raw
`state_dict()` carries a 1.64M frozen rotary buffer that never enters any
distance. The response magnitude χ_H is cross-sectional only (per-cell frozen
B0 `response_curvature.csv`), never a per-step channel.

## 2. Mathematical model

**Q1 (supporting).** Do ψ_t trajectories show metastable regime structure?
A Gaussian HMM (full covariance, sticky-initialized transition matrix) is fit
on pooled standardized ψ_t over a k ∈ {1, 2, 3, 4} grid; model selection by
held-out test log-likelihood (train window = first 75% of each trajectory).
Apparent regimes must beat BOTH frozen nulls:

- **phase null** (stationary linear Gaussian): multivariate phase-randomized
  surrogates (shared random phase per frequency across channels, moments
  preserved), n = 39, seeded;
- **drift null** (continuous drift, no discrete switching): per-trajectory
  smooth drift (w = 9 moving average) transplanted from a random donor
  trajectory (possibly time-reversed) plus a phase-randomized stationary
  residual, n = 39, seeded.

Persistence requirements (frozen rules): best-k HMM test log-likelihood above
the phase-null p95 band AND mean run length ≥ 8 steps with switch rate below
the phase-null p95 AND mean run length above the drift-null p95.

**Q3a (PRIMARY — the headline).** Switching linear response vs single global
response, reduced to the observed state. Local model: ψ_{t+1} = A_{z_t} ψ_t +
b_{z_t} + ε (per-regime least squares on train windows; the reduced form of
R = C Φ_i B). Global model: ψ_{t+1} = A ψ_t + b + ε (single Φ). Comparison on
held-out one-step prediction error, split into **within-cluster** steps
(z_t = z_{t+1}) and **across-transition** steps (z_t ≠ z_{t+1}). Frozen
support rule: the local model is supported iff its overall held-out error
(within + across, across weight 1.0) is below the global model's AND its
across-transition error exceeds its within-cluster error. Regime parameters
are reported (A_i, b_i, residual covariance, spectral radius) as the
deliverable switching linear response model.

**Q2 (supporting).** Does the frozen response χ_H differ between dominant
regimes? Cells grouped by dominant Viterbi regime; statistic = |mean
difference| across the two dominant groups; seeded permutation test
(n = 20000); seed-sign consistency (≥ 2/3 seeds) required in at least half of
the (protocol, H) rows.

**Q3b (supporting).** Occupancy-mixture (local) vs pooled-constant (global)
prediction of χ_H under leave-one-cell-out CV; errors via the frozen a0c
`_metric` (imported, never redefined); numerical-tie resolution floor 1e-12.

**Descriptive companion (hindsight labels only).** Pulse4-release retained
gain g(t) = loss_pulse(t) − loss_native(t), t ≥ 5, from frozen B0 diag rows;
half-life = first t with |g(t)| ≤ 0.5|g(5)| (censored at 64);
Spearman(half-life, first Viterbi transition) — within-sample/descriptive.

**Robustness.** (i) All clustering fitted on the 24 main trajectories; the 2
stress trajectories are decoded with the main scaler/HMM as an out-of-sample
check. (ii) Q3a repeated under within-cell standardization.

## 3. Sample and replay identity

- Main: 6 cells = seeds {2030, 2031, 2032} × bank steps {46, 82} × 4 frozen
  streams {7500, 7501, 7502, 7503} = 24 native trajectories × 64 steps.
- Stress: (2030, 46, 7110), (2030, 64, 7110) = 2 trajectories × 64 steps.
- Replay semantics: native LR only (multiplier 1.0 at every step; the recorder
  never touches `param_groups lr`), T = 64, **no eval calls and no theta
  captures** (zero RNG perturbation relative to the frozen B0 training
  sequence), frozen v07 recipe: same bank checkpoints, same stream recipe
  (`derive_stream_seed(x, 404)` + `batches_from_train_pool`), same frozen
  primitives `pe._baseline_no_step` + `pe._clip_and_step`, same
  `_TrainLossProbe` as `b0_run_v01.run_arm`.
- Identity gate: every shared field (grad_norm, update_norm, update_cos_prev,
  train_loss, preclip_norm, lr_native, lr_applied, multiplier, clip_active)
  bitwise equal to the frozen B0 `main_diag.csv` / `stress_diag.csv` native
  rows for all 26 trajectories × 64 steps (`runs/gate.json`, ALL_PASS, 0
  mismatches). Inline smoke gate (h = 1..8) passed before the main launch.

Budget: smoke 16 + 16 (one failed-attempt diagnostic tolerance, seq 4) +
failed main row-write attempt 64 (seq 5, recorded, no rows) + main 1,536 +
stress 128 = **1,760 optimizer updates**; project cumulative **17,577 /
18,000** (`gpu_budget_log.json`, per-launch merged TOTALS); 0 eval calls.

## 4. Results

Theory gate first: **T1–T6 all PASS at full settings**
(`runs/theory_tests.json`, `theory_tests_ok: true`) — HMM recovers planted
switching systems (T1); both null bands contain stationary/continuous-drift
data in calibrated repetitions (T2, T3); the switching-map fit recovers
planted per-regime maps and beats the global map out of sample with the
deficit concentrated at transitions (T4); the LOCO-CV chi machinery favors
local exactly when chi is occupancy-dependent and never on exact-constant chi
(T5, both directions, repeated); the frozen a0c `_metric` gate holds (T6).
The pipeline is calibrated; the negatives below are not artifacts of the
machinery.

### 4.1 Q1 (supporting): no metastable regime structure — NOT_SUPPORTED

- Held-out likelihood prefers k = 2 over k = 1 (test LL −2261.0 vs −2498.5;
  k = 3/4 overfit: −3034.8 / −3388.1; GMM agrees: best test LL at k = 2).
  `cond_multimodal: true`.
- **But the k = 2 fit does not beat the stationary phase null**: data test LL
  −2261.0 vs phase-null p95 −2114.6 (`cond_hmm_beats_phase_null_ll: false`) —
  phase-randomized surrogates (identical per-channel spectra and
  cross-channel linear correlations, no switching) reach *better* held-out
  likelihood than the data's HMM.
- **Persistence fails in the direction opposite to metastability**: mean
  Viterbi run length 25.6 steps, switch rate 0.0238/step — data runs are
  *shorter* and switch *more often* than the null bands (phase-null run
  median 43.9, p95 64.0, switch-rate p95 0.01396; drift-null run p95 64.0).
  `cond_persistence: false`, `cond_beats_drift_null: false`.
- Occupancy 0.83 / 0.17; pooled transition matrix [[0.886, 0.114],
  [0.005, 0.995]] — sticky, but stickiness alone is what a stationary
  surrogate also produces; nothing here requires discrete metastable regimes.
- Stress robustness: the two stress trajectories decoded with the main
  scaler/HMM give the same picture (occupancy 0.82/0.18, mean run 32).

**Q1 = NOT_SUPPORTED**: over 64-step windows at this scale, ψ_t shows no
metastable switching structure beyond what stationary linear-Gaussian
surrogates with the same spectra reproduce. (The amendment-required
continuous-drift null is beaten in the wrong direction: real drift would
produce *longer*, not shorter, runs than the data.)

### 4.2 Q3a (PRIMARY): switching linear response — NOT_TESTED (standalone read: NOT_SUPPORTED)

Per the frozen `headline_rule`, Q1 NOT_SUPPORTED ⇒ Q3a is relabeled
NOT_TESTED (no regimes to switch between). The standalone Q3a numbers,
reported for transparency:

- Held-out one-step MSE, test windows (last 25% of each trajectory):
  **local (per-regime maps) 0.29609 vs global (single map) 0.29558** — the
  local model is not better overall (`cond_local_better_overall: false`).
- **Across-transition test steps: n = 0 of 360** for both models — the sticky
  Viterbi paths place no transition inside any held-out window, so the
  within/across split degenerates (`cond_local_across_gt_within` not
  evaluable; the verdict falls back to the overall comparison).
- Fitted maps (deliverable parameters): regime 0 (occupancy 0.17): spectral
  radius **1.013** (expanding), n_train = 263; regime 1 (occupancy 0.83):
  spectral radius **0.997**; global single map: spectral radius **0.996**
  (marginally stable; full W/A/b/residual covariance in
  `runs/analysis.json → q3a_primary.switch_res`).
- Within-cell-standardization robustness variant: local better overall but
  still zero across-transition steps → UNRESOLVED (no transition signal
  either way).

**Interpretation**: with no transitions in held-out windows and no
null-beating regime structure, the switching linear response model reduces to
the single global map — which is itself near-unit (spectral radius 0.996),
i.e. the reduced state is close to a marginally stable single linear regime
over the recorded horizon.

### 4.3 Q2 (supporting): χ by regime — NOT_TESTED

All six cells have the **same dominant regime** (state 1 occupancy 0.63–0.91
per cell; state 0 never dominant) → insufficient group diversity in every
(protocol, H) row. There is no second dominant group to compare χ against.

### 4.4 Q3b (supporting): occupancy-mixture vs constant χ — NOT_TESTED (raw signal: 3/3)

Per the frozen headline rule, relabeled NOT_TESTED. Transparent raw signal:
the LOCO-CV machinery's median per-seed error difference favored the local
(occupancy-mixture) model in **3/3 seed folds** — but with one dominant
regime per cell, cell occupancies span only 0.09–0.38 in the minority state,
n = 6 cells, and the a0c-metric comparison operates on |chi| ~ 1e-2 values;
the signal is fragile, descriptive, and moot without supported regimes.

### 4.5 Descriptive companion: retained-gain decay vs Viterbi transitions

24 pulse4-release (+0.05) trajectories (frozen B0 diag): retained gain g(t)
half-life vs first Viterbi transition — Spearman **ρ = −0.369, p = 0.076,
n = 24**; 0 rows below the frozen 1e-7 resolution floor. No positive coupling
between transition location and retained-gain decay (if anything, negative).
Hindsight labels only; not deployable.

### 4.6 Headline

**Headline verdict: NOT_TESTED** — per the frozen headline rule, since Q1
(regime existence) is NOT_SUPPORTED, the switching-linear-response question
has no object. Substantively: *at the 64-step horizon and 4.98M scale, the
recorded reduced state ψ_t carries no evidence of metastable dynamical
regimes; a single near-unit linear reduced map fits as well as any
regime-switching family, and the short-horizon-accurate / long-horizon-fail
phenomenon is not explained by discrete regime switching in ψ_t.*

## 5. Failure cases and limits

- **Degenerate across-split (structural)**: sticky Viterbi paths × 16-step
  held-out windows ⇒ zero across-transition test steps; Q3a's
  transition-concentration arm could not be evaluated even though the
  pipeline was calibrated to recover exactly such structure on planted data
  (T4). A longer recorded horizon (or regime assignments with more
  transitions) is needed for a fair within/across test. This is a limitation
  of the sample, not of the fit quality.
- **Pilot scale**: 3 seeds × 2 checkpoints × 4 streams (+2 stress); streams
  are not independent seeds; no strong small-p claims were made.
- **Observed-state reduction**: ψ_t is a 9-dim norm/cosine/slope projection.
  Regime structure living outside this projection (e.g. in the full
  θ-distribution geometry, per-layer spectra, or clip-event organization —
  A0Q found clip is event-driven) is not excluded.
- **Horizon bound**: 64 steps. Metastability with residence times ≫ 64 steps
  is NOT_TESTED here; nothing in this report speaks to longer horizons (A0P
  long-memory caveat applies verbatim).
- **χ legs (Q2/Q3b) structural degeneracy**: single dominant regime per cell
  made Q2 untestable and Q3b's raw 3/3 favor fragile.
- **Verdict asymmetry**: NOT_SUPPORTED (Q1) is a statement about this state
  representation, scale, and horizon — not a proof that metastable regimes do
  not exist in transformer training generally. Terminology discipline
  maintained: no claim of any physical phase transition.

## 6. Implications for Mori–Zwanzig coarse graining

1. **Discrete-switching correction is rejected at this scale.** The MZ
   program here asked whether the resolved (tangent/linear-response) model
   fails because the unresolved dynamics hop between metastable regimes. The
   answer from B0.6 is no for the recorded ψ_t: the apparent two-regime HMM
   is outperformed by stationary surrogates, and its runs are less persistent
   than the null bands. A Markov-state memory term is therefore not the
   warranted coarse-graining correction at 64 steps.
2. **The surviving reduced dynamics is a single marginally stable map.** The
   global one-step map on pooled standardized ψ_t has spectral radius 0.996 —
   near unit, consistent with B0.5's finding that retained gains are carried
   by direct, partially transverse parameter displacement with a fast
   (~12-step) optimizer-state kernel. The reduced state barely contracts per
   step; long-horizon divergence of response predictions accumulates from
   near-unit eigenvalues plus unresolved continuous fluctuations, not from
   discrete switches.
3. **What remains unresolved for MZ.** Combined with A0P (no stable decay
   timescale; noise-dominated long-lag tail) and B0.5 (fast m-kernel), the
   long-horizon failure of response transfer is now bounded by elimination:
   not optimizer-state memory (B0.5), not discrete regime switching in ψ_t
   (B0.6). The remaining candidates are (a) continuous non-stationarity
   beyond the 64-step window, (b) nonlinearity not captured by one-step
   linear maps (the present test was linear in ψ), and (c) unresolved
   variables outside the 9-dim projection. A kernel/Markov hybrid MZ model
   would need evidence this experiment set out to provide and did not find;
   building one anyway would be unsupported by the frozen evidence chain.
4. **Practical consequence for the FR program.** The single-map reduced
   model with near-unit spectral radius predicts that short-horizon response
   stays accurate but error compounds ~linearly over long horizons — matching
   the B0/A0P phenomenology without any regime machinery. Any future control
   or schedule design should assume a single continuous reduced regime on
   this evidence, subject to the horizon caveat.

## 7. Artifacts and reproduction

- Experiment root: `config.json`, `live_manifest.json`,
  `gpu_budget_log.json` (per-launch merged TOTALS).
- Raw: `runs/psi_main.csv` (24 × 64), `runs/psi_stress.csv` (2 × 64),
  `runs/smoke_checks.json`, `runs/gate.json`,
  `runs/driver_summary.json`, `runs/analyze.log`.
- Analysis: `runs/analysis.json`, `runs/theory_tests.json`, `runs/summary.json`.
- Figures: `figures/fig1_psi_channels.png` … `fig7_q2_q3b_chi.png`.
- Integrity: `audit/SHA256SUMS` (whole output tree + code, staged).
- Reproduce: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=research/src python
  research/scripts/b06_record_psi_v01.py --mode smoke|main|stress`, then
  `python research/scripts/b06_gate_v01.py`, then `python
  research/scripts/b06_analyze_v01.py`; CPU tests: `python -m unittest
  research.tests.test_b06_regime_dynamics_v01`.
