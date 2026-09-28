# continuation_adaptation — Training Continuation under Changed Compute Budget (archived)

**Status: COMPLETED 2026-09-26, terminal verdict STOP (preregistered falsification result).**

## What this was

A preregistered elimination experiment asking whether an inherited AdamW optimizer state
(m, v, step) still "fits" a Transformer checkpoint when training continues under a changed
compute budget and changed hyperparameter schedule — and whether simple zeroing-style state
adaptation (arms C1/C2) can beat a **tuned** strong-continuation baseline (arm A).

Design philosophy (frozen red lines): no new optimizer, no reconstruction algorithms,
failure is a valid terminal outcome. It failed: **C ≤ A**, verdict `STOP`.

## One-line result

Across 56 cells (8 seeds × 7 checkpoint positions × 8 arms = 448 sealed runs), neither
state-adaptation variant beat the tuned baseline (Holm p = 1.0 both), C1 (zero-v) was
strongly harmful, and inherited-state + β2-migration (B) was descriptively the best arm —
see `RESULTS.md` for exact numbers and the boundary-unresolved limitation.

## Directory layout

```
archived/continuation_adaptation/
├── README.md           this file
├── RESULTS.md          evidence-linked scientific summary
├── code/
│   ├── src/            framework: vendored_fr/ (12 frozen legacy modules, SHA-registered),
│   │                   optimizer_utils.py, state_adapter.py, evaluation.py,
│   │                   continuation_runner.py, p4_runner.py, p4_selection.py,
│   │                   p4_statistics.py
│   ├── scripts/        run_smoke_v01.py, p4_stage1.py, p4_sealed_eval.py, p4_analyze.py
│   └── tests/          roundtrip / beta-modification / CRN reproducibility + p4 infra (33 tests)
├── configs/
│   ├── smoke_v01.json  frozen smoke config
│   └── p4_full.json    frozen P4 config (56 cells, per-file SHA256)
├── protocols/
│   ├── README.md       original project README
│   ├── PROJECT_CONTEXT.md  audit + direction definition (P0 deliverable)
│   ├── EXPERIMENT_PROTOCOL.md  smoke protocol (frozen)
│   ├── P4_PROTOCOL.md  full P4 preregistration (frozen; §8 = stop rules)
│   ├── P4_PROTOCOL_REVIEW.md  freeze review
│   └── LEGACY_PROVENANCE.md   vendored module SHA256 registry
└── results/outputs/    ALL small evidence files (1500 files, ~16 MB; no checkpoints):
    ├── smoke_v01/      smoke_report.json + per-arm metrics/stats
    ├── p4_stage0/run/  C0 sanity run metrics
    ├── p4_full/stage1/ per-run metrics.json / loss_curve.csv /
    │                   optimizer_state_statistics.csv for all 448 runs + selection ledgers
    ├── p4_full/selection/  56 per-cell selection ledgers (frozen rule + boundary flags)
    ├── p4_full/stage2/ SEAL_LOCK.json + sealed_evaluation.json + sealed_evaluation_table.csv
    └── p4_full/stage3/ analysis_report.json  (final verdict)
```

## Not stored on GitHub

452 checkpoint files (`final_state.pt`, 28 GB total) remain on the research server at
`continuation_adaptation/outputs/`. Their SHA256 hashes for every sealed run are preserved
in `results/outputs/p4_full/stage2/SEAL_LOCK.json`, so integrity remains verifiable.

## Source

Read-only copy from `/home/b205/projects/training_dynamics_lab/continuation_adaptation/`
(git-tracked upstream: none; the source project had no VCS). Nothing on the server was
modified during archiving.
