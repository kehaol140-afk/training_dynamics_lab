# optimizer_sensitivity — Experiment Index

All units with frozen verdicts. Dates = data-production window. Evidence paths relative
to the source project root (mirrored under `results/outputs/` when small).

| # | Unit | Question | Verdict | Key evidence |
|---|---|---|---|---|
| 1 | F0 bank audit (09-26) | Is tangent ill-conditioning real at scale? | `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE` | `outputs/f0_existing_bank_audit/` |
| 2 | F1 mechanism closure (09-26) | Which channel causes it? | `V_FLOOR_CAUSAL` (v→θ′ dominant; PRECISION_DOMINATED excluded) | `outputs/f1_mechanism/` |
| 3 | F1 1B replication (09-26) | Replicate at 1B | **CANCELLED before compute** (0 bytes) | `outputs/f1_1b_replication/` (download.log only) |
| 4 | F2 hypergradient pilot (09-26) | Are tangent hypergradients usable? | `PRACTICAL_HYPERGRADIENT_FAILURE` (EARLY fails / MIDDLE usable) | `outputs/f2_hypergradient/` |
| 5 | F2.5 trust region (09-26/27) | FD-usable neighborhood? | `TRUST_REGION_COLLAPSE_CORRELATED_ONLY` | `outputs/f2_5_trust_region/` |
| 6 | F3 compatibility (09-27) | Env-drift recovery | `CURRENT_ENV_REFERENCE_ESTABLISHED`; gate PASS | `outputs/f3_compatibility/` |
| 7 | F3 curvature mechanism RQ1–RQ5 (09-29) | Curvature/spectrum mechanism | `SPECTRAL_INSTABILITY` (v1 CURVATURE_FEEDBACK_CAUSAL and v2 ATTN_Q bottleneck rejected; MIDDLE UNRESOLVED) | `outputs/f3_curvature_mechanism/` (summary_final.json + 8 CSVs) |
| 8 | F3 RQ3 interim (09-29) | Module ablation interim | superseded by #7 final | `docs/F3_RQ3_INTERIM_REPORT.md` |
| 9 | F3.5 unstable-mode coupling (09-29/30) | LEFT-subspace coupling? | `METRIC_DEPENDENT` (RAW vs DIML flip on persistence) | `outputs/f3_5_mode_coupling/` |
| 10 | F4A phase map (09-30) | Scale×phase×seed structure | `PHASE_STRUCTURE_ONLY_NO_SCALE_TREND` + `HIGH_SEED_HETEROGENEITY` (strict healing **13/14** per erratum) | `outputs/f4a_phase_map/` |
| 11 | F4A.1 matched coverage (09-30) | Cure 5M MID-only gap | `NONMONOTONIC_SCALE_STRUCTURE` (canary 5/5 PASS; 16 new states) | `outputs/f4a1_matched_coverage/` |
| 12 | F4A.5 formal modeling (09-30) | Predict R_curv? | `ROBUST_PHASE_MODEL_NO_SCALE_EFFECT`; prescreen **DISABLED_BY_PREREG** | `outputs/f4a5_modeling/` |
| 13 | F4B SAU (09-30) | Adaptive unrolling gain? | `NO_METHOD_GAIN` (SAU≡FIXED1; δ=.05/.02 agree) | `outputs/f4b_sau/` (delta_0p05/ canonical) |
| 14 | F4B postmortem (09-30) | Why did SAU fail? | `OBSERVABILITY_GAP_SUPPORTED` + `UTILITY_ALREADY_SHORT_HORIZON` | `outputs/f4b_postmortem/` |
| 15 | F4C intervention-scale mining (09-30) | δ-dependence of useful horizon | `WEAK_SCALE_DEPENDENCE`; STAGE D DISABLED | `outputs/f4c_mining/` |
| 16 | F4 consistency audit (09-30) | Artifact/report consistency | `CONSISTENCY_RESTORED` (14 rows: 10 VERIFIED / 3 STALE / 1 OVERWRITTEN — see ERRATA_20261008) | `outputs/f4_consistency/headline_numbers.csv` |
| 17 | Decision memo + synthesis (09-30) | Program direction | Mechanism paper A > observability B > C; method D unsupported; stop E unnecessary | `docs/F4_POSTMORTEM_DECISION_MEMO.md`, `docs/RESEARCH_SYNTHESIS_F0_F4B.md` |
| 18 | External validation plan (09-30) | Next step | **WRITTEN, NOT EXECUTED** | `docs/EXTERNAL_VALIDATION_PLAN.md` |
| 19 | Independent re-audit (10-08) | Whole-program verification | 11/11 frozen artifacts byte-identical; F4 chain complete; 4 documentation defects registered (none verdict-changing) | `reviews/2026-10-08/` |
