# G3 — Clipping Intervention Report (FR_G3_CLIPPING_INTERVENTION_V01)

## 1. Question and design

- G3a operator control: on the frozen G1a 26M clip=1.0 bank (seed 2042 x
  EARLY/MIDDLE/LATE, 4 streams/state, H={1,4,8}), C_identity (future
  clip-JVP replaced by identity, everything else unchanged) must equal
  BH_fixed.  Not a causal regime claim -- the baseline trajectory is
  still clip=1.0-trained.
- G3b true intervention: 26M retrained with clipping genuinely disabled
  (clip_grad_norm_ never called), seeds 2042/2043/2044, states
  26/64/102; formal FD/B/BH_fixed/C matrix paired against the frozen
  G1a clip=1.0 condition (identical stream seeds).
- Core prediction: gain_clip > 0 under clip=1.0; gain_clip ~ 0 under
  no-clip; Delta_clip = mean(gain_clip_clip1) - mean(gain_clip_noclip) > 0.

## 2. Model configuration (26M, fixed by protocol)

- hidden=512, layers=7, heads=8, kv=4, head_dim=64, intermediate=1664
- vocab=6400, seq=128; unique params = 26,681,728
- Everything identical to G1a except clipping disabled in training.

## 3. G3a operator control

- pass = **True**; n stream-H cells = 36
- max |C_identity - BH_fixed| = 0
- mean |C_identity - BH_fixed| = 0
- max relative gap = 0
- frozen-G1a cross-check max gap = 0

## 4. G3b no-clip training

- seeds [2042, 2043, 2044]; states [26, 64, 102]; clip_grad_norm_ called: **False** (manifest-recorded)
- counterfactual would-clip rate (preclip norm > 1.0): 1.0000
- training preclip norm mean (no-clip) = 1.3544 vs G1a clip=1.0 = 1.3906

## 5. No-clip formal results (27 official cells)

| model | pooled R2 | corr | MAE | mean rel | median rel |
|---|---:|---:|---:|---:|---:|
| B | 0.9382138310 | 0.9937054695 | 0.0155950978 | 0.1116178843 | 0.0565060232 |
| BH_fixed | 0.9996392022 | 0.9999088302 | 0.0009745211 | 0.0061239767 | 0.0022605902 |
| C | 0.9996392022 | 0.9999088302 | 0.0009745211 | 0.0061239767 | 0.0022605902 |

- BH_fixed vs C collapse: max abs gap = 0, max rel gap vs FD = 0 (identity expected).

## 6. Clip vs no-clip mechanism gains (pooled H>1)

| condition | gain_curvature | gain_clip |
|---|---:|---:|
| clip=1.0 (G1a frozen) | +0.07462852 | +0.09397951 |
| no-clip (G3b) | +0.15824086 | +0.00000000 |

- Metric/aggregation (identical for both conditions): official cell = mean
  of the 4 stream chi per (seed, stage, H); relative error = |pred-obs|/|obs|
  (frozen a0c `relative_error`) against the cell-mean chi_FD; per-cell
  gain_clip = rel_err_BH_fixed - rel_err_C over H in {4,8}; pooled = equal
  weight over 18 cells (3 seeds x 3 stages x 2 H).
- Clip-condition provenance: chi_fd/chi_B/chi_C for all 3 seeds are direct
  reads of frozen G1a `results/per_stream.json`; seed-2042 chi_BH_fixed is a
  direct read of frozen G1a `mechanism/*.json`; seed-2043/2044 chi_BH_fixed
  is NOT present in any frozen G1a artifact (G1a mechanism subset ran seed
  2042 only) and was computed by this experiment's `clipref` phase (frozen
  a0e2 runner, grad_clip=1.0) from the read-only frozen G1a bank -- **G3
  clip-reference extension**: no checkpoint retrained, no frozen byte
  modified; the same path reproduces the seed-2042 frozen values with max
  abs gap = 0.0. See G3_PROTOCOL.md §16.
- These clip-condition numbers therefore differ in aggregation level from
  the G1a report §7 mechanism-subset figure (+0.01472, seed 2042, stream-
  level errors averaged within each (stage,H) cell); both are frozen and
  neither supersedes the other.

## 7. Delta_clip interaction

- pooled mean Delta_clip = +0.09397951 (official-cell level, same metric,
  same (seed,stage,H) support and same equal-cell weighting in both
  conditions; paired by exact cell key)
- **official-cell hierarchical bootstrap (PRIMARY statistical caliber)**:
  same official-cell estimator, resampling seed > stage > H-cell
  (20000 resamples, rng 20260916): observed = +0.09397951, mean
  +0.09521030, median +0.08317223, 95% percentile interval
  [+0.00688686, +0.32209561], fraction of bootstrap resamples with
  Delta>0 = 0.9978 (this fraction is NOT a p-value, posterior
  probability, or significance probability).  Source:
  `results/official_cell_bootstrap.json`.
- cluster bootstrap (seed>stage>stream): mean +0.08563305, 95% PI [+0.00862619, +0.29383619], P(Delta>0) = 0.9999
- the stream-level bootstrap is retained as a **robustness diagnostic**;
  its input is the stream-level counterpart (n=72 paired stream-H
  cells, per-stream relative errors), so its observed mean (+0.08564)
  and the official-cell pooled mean (+0.09398) are different aggregation
  levels of the same pairing, not a discrepancy; the bootstrap mean equals
  the stream-level observed mean up to Monte-Carlo error (20000 resamples,
  rng 20260916). Direction is consistent between the two calibers
  (both positive), so no STATISTICAL_MISMATCH. P(Delta>0) is a bootstrap
  posterior probability, not a frequentist p-value.
- bootstrap input is the stream-level counterpart (n=72 paired stream-H
  cells, per-stream relative errors), so its observed mean (+0.08564) and
  the official-cell pooled mean (+0.09398) are different aggregation
  levels of the same pairing, not a discrepancy; the bootstrap mean equals
  the stream-level observed mean up to Monte-Carlo error (20000 resamples,
  rng 20260916). P(Delta>0) is a bootstrap posterior probability, not a
  frequentist p-value.

### per-H / per-seed / per-stage interaction

| group | mean Delta_clip |
|---|---:|
| H4 | +0.06741059 |
| H8 | +0.12054842 |
| seed_2042 | +0.01586422 |
| seed_2043 | +0.01784542 |
| seed_2044 | +0.24822888 |
| EARLY | +0.00046609 |
| MIDDLE | +0.25669328 |
| LATE | +0.02477915 |

## 8. Verdict (frozen thresholds)

**MECHANISM_CONFIRMED**

| condition | pass | value |
|---|---:|---|
| cond1_operator_control | True | {"pass": true, "g3a_max_abs_gap": 0.0, "tol": 1e-12} |
| cond2_gain_clip_collapse_le_20pct | True | {"pass": true, "gain_clip_clip1": 0.09397950627466435, "gain_clip_noclip": 0.0, "shrink_ratio": 0.0, "threshold": 0.2} |
| cond3_noclip_bh_c_identity_and_predictive | True | {"pass": true, "pooled_R2_BH_fixed": 0.9996392022020426, "pooled_R2_C": 0.9996392022020426, "pooled_mre_BH_fixed": 0.006123976689801907, "pooled_mre_C": 0.006123976689801907, "bh_c_max_rel_gap_vs_fd": 0.0} |
| cond4_delta_clip_positive | True | {"pass": true, "delta_clip": 0.09397950627466435} |

## 9. Dynamics context (explanation only, no post-hoc filtering)

| quantity | clip=1.0 (G1a) | no-clip (G3b) |
|---|---:|---:|
| training clip rate | 1.0000 | 0.0000 (by definition) |
| training preclip norm mean | 1.3906 | 1.3544 |
| counterfactual would-clip rate (preclip>1) | n/a | 1.0000 |
| training loss mean | 3.6333 | 3.6508 |
| future first-step clip rate | 1.0000 | 0.0000 |
| future all-step clip rate | 0.9896 | 0.0000 |
| future grad norm mean | 1.3295 | 1.2758 |
| future HVP norm mean | 0.2837 | 0.1867 |

## 10. Anomalies / incidents

- smoke gate: overall_pass=True, h1_analytic_vs_fd_rel_err=0.000878903
- GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log
- no OOM; no retry; no RESOURCE_LIMIT

## 11. Scope statement

- G3a is an operator control, NOT causal training-regime evidence.
- The causal claim rests on G3b no-clip training trajectories.
- Curvature feedback is not claimed to vanish; only the clip-JVP leg is
  intervened on.  No-clip does not imply C == FD (finite-epsilon FD
  nonlinearity remains).
- Clipping-regime changes across scales remain observations; no
  empirical law is fitted and no causal attribution beyond the G3b
  intervention is made.

## 12. Next step

G2 (26M data-distribution generalization) is a user decision; it is NOT
started by this experiment.
