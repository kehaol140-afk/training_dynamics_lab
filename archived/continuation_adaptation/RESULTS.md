# continuation_adaptation — Results Summary

> All numbers below are read directly from frozen artifacts on the research server
> (`continuation_adaptation/outputs/`), mirrored under `results/outputs/` in this archive.
> Nothing was re-run; no interpretation beyond the frozen `analysis_report.json` is offered.

## Research question

When a Transformer checkpoint (with full AdamW state) continues training under a **changed
compute budget** and **changed hyperparameter schedule** (compressed LR schedule + β2
migration), is the inherited optimizer state (m, v, step) still well-matched? Can simple
zeroing-style state adaptation beat a genuinely tuned plain-continuation baseline?

Evidence: `protocols/P4_PROTOCOL.md` header; `protocols/PROJECT_CONTEXT.md` §4

## Experimental design (frozen in `configs/p4_full.json` + `protocols/P4_PROTOCOL.md`)

- **C0 set**: 56 cells = **8 seeds (2030–2037) × 7 checkpoint positions (t0 = 46, 52, 58,
  64, 70, 76, 82 optimizer steps)** from the frozen legacy v07 bank (full_state v0.2:
  model fp32 + m/v/step + scaler + 4-way RNG + partition sha).
- **Budget**: rule `changed_budget_half_remaining_v1` — B = T_remaining/2 = (128 − t0)/2
  optimizer steps per cell (e.g. t0=46 → B=41; t0=64 → B=32). LR schedule compressed to
  traverse the remaining native cosine exactly within B, ending at 0.1·base_lr.
- **Arms (8 runs per cell = 448 runs)**:

| Arm | betas | m | v | step | lr_multiplier |
|---|---|---|---|---|---|
| A strong continuation (grid: A_lr0.5/1/2 + auto boundary 4.0) | (0.9, 0.999) | inherit | inherit | inherit | tuned per cell {0.5, 1.0, 2.0} (+one-time 4.0 expansion) |
| B HP migration | (0.9, 0.99) | inherit | inherit | inherit | = selected A multiplier |
| C1 state adaptation | (0.9, 0.99) | inherit | **zeroed** | inherit | = B |
| C2 state adaptation | (0.9, 0.99) | **zeroed** | inherit | inherit | = B |
| D reset baseline | (0.9, 0.99) | zeroed | zeroed | **reset to 0** | = B |

- **Primary comparison**: C1 − A_best and C2 − A_best on final sealed validation NLL
  (V_final, eval-pool token CE), seed-clustered permutation test (20000 reps) + cluster
  bootstrap CI (10000 reps), Holm α=0.05; gates: mean<0, ≥6/8 seeds, ≥5/7 checkpoints,
  Holm-significant.
- **Tuning discipline**: arm A selected on final V_monitor (32 held-out probe samples)
  only; V_final computed exactly once at Stage 2 under SEAL (never during tuning).
- Evidence: `protocols/P4_PROTOCOL.md` §2–§7; `configs/p4_full.json`

## Implementation validation (P3 smoke, `results/outputs/smoke_v01/smoke_report.json`)

All 8 checks true: all_runs_complete, eval_works, c0_model_state_identical_across_runs,
eval_at_c0_identical_across_runs, batch_manifest_identical_across_runs,
determinism_A_repeat (bit-level), arms_produce_distinct_trajectories,
arm_application_verified. Smoke per-arm eval means (diagnostic only): A 7.00008, B 6.96525,
C1 7.21302, C2 6.97887, D 6.99070. 33 infrastructure tests passed (runner / selection /
statistics / sealed-eval / analyze).
Evidence: `results/outputs/smoke_v01/smoke_report.json`; `results/outputs/test_p4_infra/`

## P4 execution

- **Stage 0**: C0 sanity run complete (`results/outputs/p4_stage0/run/metrics.json`).
- **Stage 1**: all 56 cells × 8 arms completed; per-run metrics/loss-curve/state-stats
  preserved; per-cell selection ledgers with frozen tie/boundary rules
  (`results/outputs/p4_full/selection/*_ledger.json`).
- **Stage 2 (sealed evaluation)**: `SEAL_LOCK.json` written 2026-09-26T06:24:45Z over
  **n_runs = 448** with per-run metrics/final_state SHA256; V_final computed once;
  full table in `sealed_evaluation_table.csv` (448 rows × nll_mean/p95/p99/cvar99/
  update_norm/cos_vs_abest).
- **Stage 3**: single-pass analysis → `analysis_report.json` (verdict).

## Final metrics (from `results/outputs/p4_full/stage3/analysis_report.json`)

Primary (paired diff in final NLL, negative = better; 56 cells):

| Comparison | Aggregate mean diff | Cluster-bootstrap 95% CI (by seed) | Holm p | seeds improved | checkpoints improved |
|---|---|---|---|---|---|
| **C1 − A_best** | **+1.5871** | [1.2352, 1.9550] | 1.0 | 0/8 | 0/7 |
| **C2 − A_best** | **+0.0194** | [0.0141, 0.0246] | 1.0 | 0/8 | 0/7 |

Secondary descriptive (naive cell bootstrap, anti-conservative, diagnostic only):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| B − A_best | **−0.01388** | [−0.01737, −0.01027] |
| D − A_best | **+0.06085** | [+0.05565, +0.06619] |
| B − D | **−0.07473** | [−0.08103, −0.06871] |
| C1 − C2 | +1.56771 | [1.25545, 1.91353] |

Per-checkpoint C1−A_best means rise monotonically with t0 (0.724 @46 → 2.741 @82);
per-seed means 0.94–2.35, direction uniform (all 8 seeds positive).
`continue_direction` gate: **false** (all four sub-gates fail for both C1 and C2).

## Final verdict

**STOP** — preregistered stop condition "C ≤ A (including statistically non-significant
with tiny effect)" is triggered. Per the frozen protocol this is a terminal negative
result for the direction; no rescue rounds.

## Important limitation (must not be hidden)

- **a_grid_boundary_accounting: cells_with_expansion = 56, cells_boundary_unresolved = 56.**
  In every cell the tuned A optimum sat on the (post-expansion) grid boundary. For a
  *positive* C > A claim this would have forced a strength downgrade (the direction's own
  review notes treat unresolved boundaries as weakening evidence). Here the verdict is
  STOP (C ≤ A): the boundary issue means A may be *under*-tuned, which if anything makes
  the already-negative C-vs-A comparison conservative in the direction of the stopping
  decision — but the mirrored implication is that the small descriptive B < A_best
  advantage (−0.0139) is **not** interpretable as a real HP-migration gain, since the
  baseline it compares against is boundary-limited. The frozen report records this flag
  explicitly; no post-hoc resolution was attempted.
- Scale: h256/4L (4.98M params) only; the protocol states elimination-round conclusions
  adjudicate "direction worth continuing", not behavior at realistic scale.
- Single budget rule (B = T_rem/2); "train-to-completion then extend" scenario not covered.

Evidence: `results/outputs/p4_full/stage3/analysis_report.json` (`a_grid_boundary_accounting`,
`stop_rule_note`); `protocols/P4_PROTOCOL_REVIEW.md`

## Scientific interpretation (strictly from the frozen report)

1. Core question answered negatively: **simple state adaptation does not beat tuned
   continuation** — C2 (zero-m) ≈ A + 0.019 (slightly worse, uniform direction), C1
   (zero-v) catastrophically worse (+1.59 mean; zeroing v explodes effective LR via
   bias-corrected v̂ → this transient is part of the arm's frozen definition).
2. Descriptively, **inherited state remains valuable under HP migration**: B − D =
   −0.0747 (inherited beats full reset) and D − A_best = +0.0609 (reset hurts even vs a
   boundary-limited tuned baseline).
3. These are the report's `secondary_descriptive` fields, flagged `descriptive_only:
   true`; the frozen verdict rests only on the C-vs-A primary gates.

Evidence: `results/outputs/p4_full/stage3/analysis_report.json` (`primary`,
`secondary_descriptive`, `verdict`)

## Reproducibility

- Frozen configs with SHA256 provenance (`configs/p4_full.json` — 56-cell C0 list with
  per-file hashes; `configs/smoke_v01.json`); SEAL_LOCK with per-run metrics + final_state
  hashes; selection ledgers per cell; CRN design (shared batch manifest + stream seed
  within cell); deterministic replay infrastructure vendored from legacy
  (`code/src/vendored_fr/`, SHA registry in `protocols/LEGACY_PROVENANCE.md`).
- Environment: conda `fr_minimind` (Python 3.11, torch 2.11.0 + CUDA), single-GPU
  visibility requirement (CUDA RNG states saved per 1 device).
- Re-run entry points: `code/scripts/run_smoke_v01.py`, `p4_stage1.py`, `p4_sealed_eval.py`,
  `p4_analyze.py` (Stage 2/3 scripts are one-shot by design: SEAL_LOCK forbids overwrite).

## Removed-from-GitHub large artifacts

452 `final_state.pt` files (28.0 GB) under `outputs/p4_full/stage1/`, `outputs/smoke_v01/runs/`,
`outputs/p4_stage0/run/`. Every one of the 448 formal runs' final states is hash-locked in
`results/outputs/p4_full/stage2/SEAL_LOCK.json` (`final_state_sha256`, `model_hash_final`).
Cleanup recommendation: `SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION` (see
`../../docs/CLEANUP_PREVIEW.md`).

## Original evidence paths

- Project root: `/home/b205/projects/training_dynamics_lab/continuation_adaptation/`
- Smoke: `.../outputs/smoke_v01/smoke_report.json`
- Seal: `.../outputs/p4_full/stage2/SEAL_LOCK.json`
- Sealed evaluation: `.../outputs/p4_full/stage2/sealed_evaluation.{json,csv}`
- Verdict: `.../outputs/p4_full/stage3/analysis_report.json`
- Protocols: `.../docs/P4_PROTOCOL.md`, `.../docs/EXPERIMENT_PROTOCOL.md`,
  `.../docs/P4_PROTOCOL_REVIEW.md`, `.../docs/PROJECT_CONTEXT.md`,
  `.../docs/LEGACY_PROVENANCE.md`
