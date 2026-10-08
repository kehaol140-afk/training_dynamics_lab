# muon_temporal_response — RESULTS (status report)

## Purpose

Characterize the temporal response of the Muon update map (2×2 toy and real anchors):
FP64 analytic maps (polar / finite-NS) vs the native BF16 path — tangent prediction
ability, bias decomposition, real-gradient magnitude calibration, and whether candidate
orderings carry exploitable target value.

## Stage verdicts (each frozen, with errata)

### T0 round1 + corrected (2026-09-28/29)
- Main table A1/A2/Delta_A/ratio_A holds; FP64 **direction model accurate, scalar model
  principled-incapable** (A1=A2); cross-channel (Bs↔Ba) measured 0 = tangent-predicted 0;
  native two-pass bit-identical; map-only attribution A rel-diff ≤ 0.024%.
- Erratum (dimension fix): B2/A at δ=5e-2 = NS arm1 6.32% / native arm1 **62.32%**
  (same-order, non-negligible) — the earlier "exceeds A" phrasing was dimensionally wrong.
- Evidence: `protocols/T0_REPORT.md`, `protocols/ROUND1_ERRATUM.md`,
  `results/outputs/t0_round1*/`.

### T2a scale_v1 (2026-09-29)
- Real-batch calibration: at natural perturbation median |δ|/|L|≈0.68 linearization
  fails (cos≈0.66) — **the toy dead-zone does not transfer**; same-H BF16 relative
  variance only 0.04–0.08%. No training-benefit conclusions (by design).
- Interpretive errata (target-scope, data-only): the "front-5 vs back-5 mean shift"
  does NOT independently prove non-stationarity (consistent with IID reference-point
  effect, ratio 1.58 ≈ 1.5 expected); the "~30% magnitude error" row mixes residual-ratio
  vs amplitude-ratio conventions — must be quoted with dimension. Numerals unchanged.
- Evidence: `protocols/T2A_REPORT.md`, `protocols/ROUND2_ERRATUM_TARGET_SCOPE.md`,
  `results/outputs/t2a_scale_v1/`.

### T2b target screen (2026-09-29) — `NO_USEFUL_SIGNAL_THIS_SETTING`
- The candidate library's own post-hoc endpoint spread (3.2–4.3e-5 nats, W0/W1) is
  ~30× below the preregistered screen δ=1e-3: **no exploitable space exists**, selection
  moot. Frozen-replay open-loop 138-step integration degrades V by ~1.85 nats vs the
  anchor (virtual 4.24 vs actual 2.288) — small-displacement validity fails at this
  window/LR; S_fin retains only weak ranking (Spearman +0.40/+0.24), S_lin anti-correlates
  (−0.76). Common tail shrinks prefix differences ~10× (Nesterov buffer β¹²⁸≈1.4e-3) —
  "terminal memory effect". Per v3 decision table row 4: **stop this version of the
  target-screening line**; no window/LR/anchor rescue.
- Evidence: `protocols/T2B_REPORT_CN.md`, `results/outputs/t2b_target_screen_v1/`.

## Not started (do not cite as done)

Full T2 (2 parent trajectories × 4 states × 2 banks × 128-batch × 16-permutation
replay), T3, T4 — 0% executed; T2 error-threshold freeze (pre-unblinding) not done;
Muon intermediate training states (step 512/1536) and second dev seed never built.

## Negative learnings that transfer (per 2026-10-08 history review)

Toy dead-zone thresholds are not real-scale obstacles; one-step screens cannot replace
finite-budget endpoint checks; frozen full-state replay is not a qualified trajectory
predictor at 138-step scale.

## Large artifacts not stored on GitHub

`outputs/t2a_scale_v1/raw_natural.pt` (627 MB), `raw_sample_l1q.pt` (176 MB) —
server-only; all CSV/JSON summaries and figures archived. ROUND1/2/3 review zips stay
on the server (ROUND2 = 140 MB).

## Original evidence paths

Root: `/home/b205/projects/training_dynamics_lab/muon_temporal_response/`;
`docs/{T0_REPORT,ROUND1_ERRATUM,T2A_REPORT,ROUND2_ERRATUM_TARGET_SCOPE,T2B_REPORT_CN}.md`;
`outputs/{t0_round1,t0_round1_corrected,t2a_scale_v1,t2b_target_screen_v1}/`.
