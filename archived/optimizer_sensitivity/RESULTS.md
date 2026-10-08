# optimizer_sensitivity — RESULTS (F0 → F4C complete chain)

> Evidence-linked synthesis. Every verdict transcribed from frozen reports/summaries on
> the server (`optimizer_sensitivity/docs/`, `outputs/`); independent re-audit:
> `reviews/2026-10-08/`. Numeric corrections in `ERRATA_20261008.md` take precedence
> where noted. **SAU is NOT a successful optimization method** (see §7).

## 1 Research question and motivation

AdamW is the default LM optimizer, yet large runs exhibit unattributable loss
spikes/instabilities. Post-hoc loss forensics cannot distinguish (i) optimizer-internal
numerical ill-conditioning, (ii) data/curvature shocks coupled to the optimizer,
(iii) trust-region/second-order information failure. This line uses controlled **tangent
dynamics** to make "instability" measurable, decomposable, and extrapolable: does it
worsen with scale? can it be predicted? can adaptive unrolling (SAU) defuse it?
Evidence: `protocols/RESEARCH_SYNTHESIS_F0_F4B.md`; `reviews/2026-10-08/ADAMW_CURRENT_RESEARCH_REPORT.md` §2

## 2 AdamW full-state mathematical modeling

Augmented state z=(θ, m, v); one-step map Φ_h linearized along the primal trajectory:
`dz_{h+1} = J_h(α)·dz_h + s_h`, α = curvature-feedback switch. Arms: B (α=0, optimizer
tangent only), C (α=1, curvature via clip_jvp∘HVP), SO (source-off impulse at h=1).
Metrics: G_B/G_C (tangent gains, H=32 main horizon), R_curv = G_C/G_B, λ_SO = ln(G_SO)/H,
plus the F1 closed forms `dθ′ = (1−lr·wd)dθ − M·dm + Dv·dv` with
`Dv ∝ m̂·v̂^{-3/2}`. Hypotheses preregistered per unit; banks are real training
checkpoints spanning phases/scales/seeds.
Evidence: `reviews/2026-10-08/ADAMW_CURRENT_RESEARCH_REPORT.md` §3; `protocols/F1_MECHANISM_CLOSURE.md` §1; `reviews/2026-10-08/MODEL_INDEX.csv`

## 3 Per-unit protocols

All units preregistered before computation (docs in `protocols/`): F0 read-only bank
audit; F1 c=0-anchor closed-form Jacobians + v-floor interventions; F2 hypergradient
vs FD on 3 cells; F2.5 trust-region δ-grid; F3 α-sweep/source-off impulse/module
ablation/local spectrum with a bit-exact reproduction gate (current-env frozen
reference, SHA256 `6f2650e0…`); F3.5 RAW-vs-DIML metric coupling probes; F4A 74-state
phase map (zero new training); F4A.1 16 new 5M matched-coverage states (canary
stream_id 4090 = 4074+16, five gates PASS); F4A.5 90-state LOSO model comparison; F4B
42-cell × 3 CRN action rollouts at δ=0.05 primary / 0.02 secondary (canary PASS);
F4B-postmortem observability; F4C mining over F4B + F2.5 raws.
Evidence: `protocols/F*_PREREG.md` (each frozen at a named commit); `protocols/F3_CURVATURE_MECHANISM_FINAL_REPORT.md`

## 4 Results and key numbers

| Unit | Verdict | Key numbers |
|---|---|---|
| F0 | `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE` | 26M/100M tangent gains A>1; FD-verified to 1e-11 |
| F1 | `V_FLOOR_CAUSAL` (v→θ′ channel dominant) | single-step v-channel amplification 2.4e9 @100M step26 (m 143, θ 1.0); τ=eps touches 396 coords, forward +0.05% → σ_max −70%; τ=10eps → −96.5% |
| F2 | `PRACTICAL_HYPERGRADIENT_FAILURE` | EARLY 2/3 cells rel_err>1 @H≤16; MIDDLE usable; FD ground truth itself degrades |
| F2.5 | `TRUST_REGION_COLLAPSE_CORRELATED_ONLY` | H=1 usable on full δ-grid; H≥2 FD arms non-convergent; boundary δ≥3e-4 |
| F3 | `SPECTRAL_INSTABILITY` | ρ(J)>1.05 at 5/5 offsets both EARLY cells (residual ≤1.3e-3); α=0 bounded ~25 vs α=1 up to 1.9e4 (endpoints 86×/744×/1.31×); module ablation DISTRIBUTED (no necessary/sufficient module); α=0 ρ=1.047–1.200 (optimizer-only marginal) |
| F3.5 | `METRIC_DEPENDENT` | persistence: RAW false (MIDDLE highest 0.349) vs DIML true (EARLY 9–16× MIDDLE) → main conclusion flips with metric; no EARLY>MIDDLE excitation separation in either |
| F4A | `PHASE_STRUCTURE_ONLY_NO_SCALE_TREND` + `HIGH_SEED_HETEROGENEITY` | median R_curv_32 decays with progress at all scales; **strict sustained healing 13/14** (seed 2032 NOT_HEALED_IN_COVERAGE); LATE 24/24 curvature-neutral; λ_SO>0 on 74/74; seed spread up to 5.3 dex; peak-vs-N Spearman 0.50<0.7 |
| F4A.1 | `NONMONOTONIC_SCALE_STRUCTURE` | matched coverage: EARLY λ_SO monotone in N (+1.0 score); R_curv non-monotone (5M≈26M≪100M peak) |
| F4A.5 | `ROBUST_PHASE_MODEL_NO_SCALE_EFFECT`; prescreen gate FAIL | M0 progress-only: LOSO MAE **0.4249** (POOLED_OOF), fold-median MAE **0.4041** (canonical; see ERRATA); seed variance 41.8%; classifier AUROC 0.842 but balanced acc 0.539 → MODEL_PRESCREEN DISABLED_BY_PREREG |
| F4B | `NO_METHOD_GAIN` (δ=.05 & .02) | SAU ≡ FIXED H=1 bitwise (λ̂(2)≈0.35–0.38 > grid cap B_λ=0.30); oracle median H=1; harmful rates FIXED16/32 (0.19/0.17) > H=1 (0.12); δ=.02 FIXED32 mean 0.1317 < SAU 0.1810 |
| F4B-post | `OBSERVABILITY_GAP_SUPPORTED` + `UTILITY_ALREADY_SHORT_HORIZON` | offline AUROC 0.8367 (δ=.05) vs online-best 0.685; δ=.02 H_sep=1 (0.7904/balacc 0.6728) with zero headroom; median H_suff=1, 92.9%/90.5% cells ≤2 |
| F4C | `WEAK_SCALE_DEPENDENCE` → STAGE D DISABLED | H2−H1 regret: δ=.05 +0.0512 [+0.000,+0.113] vs δ=.02 −0.0746 [−0.158,+0.000] (sign flip); F2.5 micro-surface H_best≡1 |
| Consistency | `CONSISTENCY_RESTORED` | headline_numbers.csv: 14 rows = **10 VERIFIED / 3 STALE_REPORT / 1 OVERWRITTEN_ARTIFACT** (see ERRATA for the 12/3/1 miscount) |

Evidence: `results/outputs/*/summary*.json` + the frozen CSVs; `reviews/2026-10-08/KEY_RESULTS.csv` (19 rows, double-sourced)

## 5 Positive findings (ESTABLISHED)

1. Tangent ill-conditioning is real, measurable, worsening with scale at one step (F0).
2. v-floor causally dominates single-step conditioning (F1; τ=eps −70% σ_max).
3. Curvature feedback is a necessary driver of EARLY multi-step excess amplification
   (F3: α=0 bounded vs α=1 explosive; source-off λ32>0 everywhere).
4. Local spectral instability is real and universal across cells (F3, incl. MIDDLE) —
   but does not distinguish phase (F3.5).
5. Phase self-healing law: log10 R_curv_32 ≈ concave function of progress; 13/14 strict;
   LATE fully curvature-neutral (F4A/A.1/A.5, 90 states).
6. Macro-predictable / micro-unpredictable (F4A.5 M0 LOSO 0.4249 dex vs prescreen balacc 0.539).
7. Long-horizon unrolling is decision-risky (F4B dominant-subset regret 0.317 vs 0 @H=1).
8. Execution-layer protocol: linger+setsid/tmux, atomic resume, bit-exact gates — four
   process deaths caught with zero data contamination.
Evidence: `protocols/RESEARCH_SYNTHESIS_F0_F4B.md` (ESTABLISHED list)

## 6 Negative findings (REJECTED, preregistered)

1. v-floor → trust-region collapse single-causal chain (F2.5).
2. Any single-module / single-eigenspace mechanism (F3 DISTRIBUTED; F3.5 METRIC_DEPENDENT).
3. Simple deterministic scale law (F4A/F4A.1 non-monotone; F4A.5 scale term no gain).
4. Cross-trajectory cheap prescreen (F4A.5 gate FAIL → disabled).
5. **Online adaptive unrolling gain (F4B SAU: NO_METHOD_GAIN)** — SAU degenerates to
   FIXED H=1; grid cap only a surface cause; h=2 discriminability <0.75 regardless.
Evidence: `protocols/RESEARCH_SYNTHESIS_F0_F4B.md` (REJECTED list); `protocols/F4B_STABILITY_AWARE_UNROLLING_REPORT.md`

## 7 Retracted / tightened claims (do not mis-cite)

- **SAU is not a successful method.** The 2026-09-30 earlier review suggested SAU as a
  candidate direction; F4B artifacts refuted it (UPDATE_DIFF row 9, WITHDRAWN-as-method).
- "All 14/14 trajectories heal" tightened to **13/14 strict** (F4A_REPORT_ERRATUM).
- "5M verifies full EARLY→LATE phase structure" tightened to MID-window-only coverage
  (F4A_REPORT_ERRATUM), later cured by F4A.1 matched coverage.
- Deterministic scale-worsening law withdrawn → non-monotone phase structure.
- F4A pooled indicator correlations (A_theta −0.81 etc.) = collinearity artifacts, no
  independent predictive power (F4A.5 Huber sign-flips).
- LATE = curvature-neutral regime, NOT "fully stable" (λ_SO>0 at 74/74 points).
Evidence: `reviews/2026-10-08/UPDATE_DIFF.md` §D; `protocols/F4A_REPORT_ERRATUM.md`; `ERRATA_20261008.md`

## 8 Established mechanism (best current statement)

Ill-conditioning infrastructure (v-floor) is always present + curvature feedback is
excited/maintained direction-selectively (metric-dependent coupling) within the early
window + phase-dominated self-healing thereafter. "Stable training, unstable
sensitivities." Evidence: `protocols/RESEARCH_SYNTHESIS_F0_F4B.md` (SUPPORTED)

## 9 External generalization NOT established

All conclusions come from single-recipe bank states (one corpus/recipe across
8+3+3 seeds). External validation (second recipe / second dataset, 50M–150M) is
**planned but not executed** (`protocols/EXTERNAL_VALIDATION_PLAN.md`); paper direction
ranking (mechanism A > observability B > finite-timescale C; method D unsupported) is a
plan, not a result. Evidence: `protocols/F4_POSTMORTEM_DECISION_MEMO.md` §5/§7

## 10 Theoretical modeling starting point

M0 progress-only model `log10 R_curv_32 ≈ a − 9.5p + 11p²` (Huber signs) as canonical
macro baseline; POOLED_OOF / MACRO_FOLD aggregation named and locked; seed random
effects = uncertainty source (41.8%), not predictor. Evidence:
`protocols/THESIS_MODELING_FRAMEWORK_FINAL.md`; `results/outputs/f4a5_modeling/`

## 11 Reference value for future work

Reusable: full-state tangent infra (vendored `adamw_tangent_step`), frozen banks
(v07/G1a/G1b + F4A.1 5M matched set), CRN/bit-exact gate discipline, delta-namespaced
analysis pipeline (`tests/test_f4b_namespacing.py`), observability-gap framing,
honest-negative decision protocol. Evidence: `protocols/RESEARCH_SYNTHESIS_F0_F4B.md`; `reviews/2026-10-08/COVERAGE.csv`

## 12 Evidence paths

- Source root: `/home/b205/projects/training_dynamics_lab/optimizer_sensitivity/`
- Per-unit: `docs/F*.md` (mirrored in `protocols/`), `outputs/<unit>/summary*.json` +
  frozen CSVs (mirrored in `results/outputs/`)
- Independent re-audit: `reviews/2026-10-08/` (report + UPDATE_DIFF + KEY_RESULTS.csv +
  RECOMPUTE_NOTES + verify_all.py/verify_out.txt + evidence/)
- Canonical corrections: `ERRATA_20261008.md` (this archive)
