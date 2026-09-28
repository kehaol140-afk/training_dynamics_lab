# G1C C-Failure Diagnostic Report (V01)

**Mandate**: G1C_C_FAILURE_DIAGNOSTIC_V01 — classify the OLMo-1B C-branch (HVP curvature
response) failure into: A. implementation bug / B. numerical precision limit /
C. 1B-scale real stability boundary / D. optimizer state anomaly.

**Target cell**: checkpoint step=380000, stream 1, C branch, registered fp16-HVP path.
**Date**: 2026-09-19. **Status**: diagnostic complete. All four checks executed.

**Constraints honored**: no change to scientific definitions, epsilon, protocol
thresholds, or frozen modules (`clip_jvp`, `g1c_adamw_tangent_step`,
`project_tangent` untouched). fp32 HVP input appears only in the explicitly
labelled `fp32_hvp_input_DIAGNOSTIC.json` (DIAGNOSTIC_ONLY: true) and enters no
official result. This report contains **no scientific conclusions and no
generalization claims**; OLMo-1B response is **not** validated by anything here.

---

## 0. Classification (required output)

| Candidate class | Verdict | One-line basis |
|---|---|---|
| A. implementation issue | **RULED OUT** | Failure is bit-exact deterministic across two in-run repeats and four independent process launches; B branch on identical code path is stable to H=8; the raise comes from the frozen NaN/Inf guard firing as designed. |
| B. numerical limitation | **CONFIRMED (contributing)** | The *specific NaN signature* at h=7 is an fp16-range overflow: with fp32 HVP input the same horizon completes finitely (chi=-34158). Under the registered fp16 path this is the proximate trigger. |
| C. scale-dependent limitation | **CONFIRMED (primary)** | The *instability itself* is dtype-independent: fp32 shows the identical qualitative trajectory (sign flip at h=6, chi → +3.7e6 by h=8, tangent norm → 3e11), while 26M/100M under the same frozen protocol stay O(1) (p50 0.47/0.65). |
| D. optimizer state anomaly | **RULED OUT** | m/v are finite and well-formed at every horizon; vhat=0 coordinates are the natural geometry of rare-token embedding rows in a normal 380k-step AdamW checkpoint, not corruption. |

**Primary classification: C (scale-dependent limitation), with B as the
proximate trigger of the observed NaN under the registered fp16 HVP path.**

Mechanism chain (all statements are arithmetic facts recorded by the
instrumentation, not new science): exact-zero `exp_avg_sq` coordinates exist in
`transformer.wte.weight` at every horizon → the AdamW denominator
`sqrt(vhat)+eps` bottoms out at the eps floor (1e-4 measured) → per-coordinate
adaptive gain 1/(sqrt(vhat)+eps) ≈ 1e4 → the C recursion (tangent feedback
through the HVP) amplifies superlinearly from h≈3 under 1B's deep-clip regime
(clip coefficient 0.003–0.012 vs 0.7–1.0 at 26M/100M) → tangent magnitude
reaches the fp16 representable ceiling inside the h=7 HVP → Inf/NaN → frozen
`clip_jvp` guard raises `ValueError: gradient/tangent contains NaN or Inf for
transformer.wte.weight`.

---

## 1. Check 1 — failing-cell instrumentation (registered fp16 path)

Error: `ValueError: gradient/tangent contains NaN or Inf for
transformer.wte.weight`. **First failure horizon: exactly h=7** (6 horizons
completed). The NaN is born *inside the h=7 HVP computation* (hvp_norm=NaN,
grad_norm=92.5 healthy, entry tangent norm finite 3.35e7); the frozen guard
catches it at the clip-JVP stage. Failing parameter group: `transformer.wte.weight`
(embedding; the same parameter named by the error in all 4 phase-2 failures).

Per-horizon record (bf16 registered path; vhat stats are global min/max over all
parameters, bias-corrected with step+1):

| h | vhat_min | vhat_max | sqrt(vhat)_min | sqrt(vhat+1e-8)_min | m_absmax | HVP norm | grad norm | clip coef | θ tangent in→out | chi |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 1.000 | 119.15 | 0.008393 | 0 → 5.502 | -1.85009 |
| 2 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 0.2316 | 109.56 | 0.009127 | 5.502 → 11.38 | -3.53963 |
| 3 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 93.41 | 130.77 | 0.007647 | 11.38 → 18.23 | -5.06703 |
| 4 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 6721.08 | 128.42 | 0.007787 | 18.23 → 3435 | -6.29677 |
| 5 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 1.059e6 | 81.99 | 0.012196 | 3435 → 1.406e5 | -8.94299 |
| 6 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | 6.786e8 | 322.38 | 0.003102 | 1.406e5 → 3.348e7 | +269.617 |
| 7 | 0 | 1.886e-4 | 0 | 1.000e-4 | 1.798e-3 | **NaN** | 92.55 | — | 3.348e7 → — | — |

(The θ-tangent in→out chain is exact: out(h) = in(h+1). The AdamW step
attenuates the HVP's superlinear growth by ~40–240× per step at these
magnitudes but cannot arrest it. Exact values in
`results/cdiag/instrumented_bf16.json`.)

Key readings:
- `vhat_min = 0` **exactly, at every horizon** — there are dead/rare wte rows.
  `sqrt(vhat+1e-8)_min = 1e-4` confirms the eps floor is what bounds the
  denominator at those coordinates (adaptive gain ≈ 1e4 there).
- The frozen zero-vhat guard in `g1c_adamw_tangent_step` did **not** fire
  (`dvhat = 0` at untouched rows), so the singularity is not an AdamW-JVP
  division by zero; it is the HVP output becoming NaN in fp16.
- Tangent-norm trajectory (θ in → out): 0→5.5 → 11.4 → 18.2 → 3.4e3 →
  1.4e5 → 3.3e7. The HVP map itself is where the superlinear growth lives
  (HVP norm 1 → 0.23 → 93 → 6.7e3 → 1.06e6 → 6.79e8); the AdamW step
  attenuates it ~40–240× per step but cannot arrest it.
- Clip active at every horizon with coefficient 0.0031–0.0122 (deep-clip
  regime; native grad norms 82–322 vs global norm 1.0).

## 2. Check 2 — comparison with 5M / 26M / 100M records

Aggregated from the frozen G1a (26M) / G1b (100M) step-trace banks
(`results/cdiag/scale_comparison_summary.json`, 252 HVP rows each) plus this
diagnostic's 1B measurements:

| quantity | 26M (G1a) | 100M (G1b) | 1B (this diagnostic) |
|---|---|---|---|
| HVP norm p50 [min, max] | 0.155 [0.033, 10.14] | 0.146 [0.037, 2.31] | 1.0 → 0.23 → 93 → … → 6.8e8, then NaN |
| tanC θ norm p50 [max] | 0.471 [1.54] | 0.649 [1.80] | 3.35e7 by h=6 (unbounded growth) |
| tanB θ norm p50 | 0.471 | 0.663 | linear +4.9/step → 40.8 at H=8 |
| native grad norm p50 | 1.257 | 1.453 | 82–322 |
| clip regime | active, coef ≈ 0.7–1.0 | active, coef ≈ 0.7–1.0 | active, coef 0.003–0.012 |
| chi_C range | [-0.46, -0.026] | [-0.41, -0.022] | -8.94 → +269.6 → NaN at h=7 |
| vhat statistics | not stored in frozen records | not stored in frozen records | vhat_min=0 exact (all h); sqrt floor 1e-4 |

**Limitation (acknowledged)**: the frozen G1a/G1b records store tangent-v
norms, not raw per-coordinate vhat, so their zero-vhat geometry cannot be
verified retroactively; their tanC_v p50 (~6.6e-6) indicates small-v
coordinates exist but with O(1) clip coefficients the C recursion remains
bounded at those scales. The 1B-vs-small-scale contrast is therefore stated on
HVP-norm, tangent-norm, clip-regime, and chi trajectories — all directly
recorded — with vhat geometry measured directly only at 1B.

## 3. Check 3 — three minimal verifications

**(a) B-branch control on the same failing checkpoint/stream, H=8: PASS.**
chi: -1.85009, -3.53964, -5.06685, -6.43993, -7.67074, -8.79239, -9.79114,
-10.69707 — all finite, smoothly drifting; tangent θ norm grows *linearly*
(5.50 → 40.81, ≈ +4.9/step). The optimizer-only path is stable to H=8; the
instability requires the HVP feedback present only in C.

**(b) fp32-HVP-input diagnostic (NOT official): COMPLETED, no NaN.**
8/8 horizons finite: chi = -1.85009, -3.53963, -5.06703, -6.29540, -8.84690,
+255.139, -34158.3, +3.6633e6; tangent θ norm → 2.97e11; HVP norm → 4.88e11.
Identical to fp16 through h=3 (bit-level at h=1–3), first numeric drift at h=4,
same qualitative divergence (sign flip at h=6). Conclusion: the fp16 NaN at h=7
is a range-overflow of an already-exploding signal (numerical trigger), and the
explosion is dtype-independent (scale-dependent instability).

**(c) Determinism repeat of the failing state: PASS.** Two consecutive
registered-path runs: identical error string, identical first-failure horizon
(7), max |Δchi| = **0.0** over the 6 completed horizons, and field-level
comparison of all 98 numeric per-h diagnostics shows max abs diff 0.0 (the h=7
hvp_norm NaN matches NaN). The chi trajectory is additionally bit-identical to
phase-2 attempt 2 and to the earlier instrumented launch — four independent
process launches, one deterministic failure. Not a race, not hardware
transients, not data-loader nondeterminism.

## 4. Why each rejected class is rejected

- **A (implementation)**: would produce run-to-run variance, failure at
  inconsistent horizons, or misbehavior in B (same code path, same guards) —
  none observed. The raise is the frozen guard working as specified.
- **D (optimizer state anomaly)**: `exp_avg`/`exp_avg_sq` are finite and
  bias-correction well-formed at every horizon (m_absmax 1.8e-3, vhat_max
  1.9e-4); exact-zero vhat rows are the expected footprint of rare tokens in a
  50304-token embedding after 380k steps, present already before any tangent
  step (h=1 entry state). Nothing in the state suggests corruption.

## 5. Consequence for the G1c feasibility gate (executability language only)

Under the frozen registered protocol, the C branch at OLMo-1B
step380000/stream1 terminates at h≈7 with a deterministic guard raise; this is
the same signature as the 4/4 C-branch failures of phase 2 (the surviving
h=7 of pair (1530000, stream1) is consistent with cell-to-cell data
dependence of the horizon, not with nondeterminism). B and FD branches remain
executable. Whether the formal G1c science experiment can proceed without a C
branch at 1B is a **protocol question for the user** — this diagnostic does
not decide it and draws no scientific inference.

## 6. Reproducibility / artifacts (all under `outputs/generalization_g1c_feasibility_v01/`)

- `results/cdiag/instrumented_bf16.json` — per-h diagnostics, registered path
- `results/cdiag/determinism.json`, `determinism_repeat.json` — check (c)
- `results/cdiag/b_control.json`, `b_control_diagnostics_raw.json` — check (a)
- `results/cdiag/fp32_hvp_input_DIAGNOSTIC.json` — check (b), DIAGNOSTIC_ONLY
- `results/cdiag/scale_comparison_summary.json`, `scale_comparison_raw.jsonl` — check 2
- `results/cdiag/analysis_notes.json` — extraction notes
- `results/cdiag_run.log` — run log
- Harness: `research/scripts/g1c_1b_feasibility_v01.py cdiag --step 380000
  --stream 1` (instrumentation is caller-side only; frozen modules unchanged)

## 7. Residual limitations

1. Deep instrumentation covers step380000/stream1; the other phase-2 failing
   cells (950000/1530000 × streams 1/2) share the identical error signature and
   are treated as the same failure class without per-cell re-instrumentation.
2. G1a/G1b banks lack raw vhat statistics (see §2).
3. The fp32 diagnostic modifies only the HVP *input cast*; the surrounding
   compute remains the registered pipeline — it isolates input-precision
   effects, not all conceivable numerical variants (e.g., full-fp32 model
   compute was not run and is not part of this mandate).
