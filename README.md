# training_dynamics_lab — Research Archive

Frozen research-archive repository for completed research lines from the
`training_dynamics_lab` research server (`/home/b205/projects/training_dynamics_lab/`).

Each archived line contains: code, frozen configs/protocols, small evidence files
(summaries / sealed evaluations / verdict JSONs / key figures), and evidence-linked
Markdown experiment summaries. Large artifacts (checkpoints, frozen state banks,
datasets, raw streams) are intentionally **not** stored here; they are catalogued
per-project in `ARTIFACT_MANIFEST.md` / `RESULTS.md` with sizes, reasons and
delete-safety flags.

## Archived projects

| Project | Status | Main conclusion | Code | Results |
|---|---|---|---|---|
| **FR / GTTR** (fluctuation-response → geometry-to-training-rule) | COMPLETED, frozen 2026-09-26 | Tangent propagation of full AdamW state explains short-horizon response (C R²=0.9977; replicates 26M/100M/external data; clipping causally confirmed), but the preregistered FR universality law FAILED, and every GTTR attempt to convert derivatives into a training rule failed its gate → line terminated ("derivatives computed correctly but not usable") | `archived/fr_gttr/code/` | `archived/fr_gttr/RESULTS.md`, `EXPERIMENT_INDEX.md` (91 entries) |
| **continuation_adaptation** (optimizer-state adaptation under changed budget) | COMPLETED 2026-09-26, terminal | Preregistered elimination experiment: across 56 cells × 8 arms (448 sealed runs), simple state adaptation does **not** beat tuned continuation (C1 −A +1.587, C2 −A +0.019, Holm p=1.0) → verdict **STOP**; inherited state still descriptively beats reset (B−D −0.075) | `archived/continuation_adaptation/code/` | `archived/continuation_adaptation/RESULTS.md` |

## Repository layout

```
├── README.md                          this file
├── ARCHIVE_STAGING_REPORT.md          what was copied / excluded / verified (audit trail)
├── archived/
│   ├── fr_gttr/                       README, RESULTS, EXPERIMENT_INDEX, code/, protocols/, results/
│   └── continuation_adaptation/       README, RESULTS, code/, configs/, protocols/, results/
└── docs/
    └── CLEANUP_PREVIEW.md             proposed server cleanup classification (nothing deleted yet)
```

## Other active projects

Other active projects remain on the research server and will be archived separately
after completion (as of 2026-09-29: `optimizer_sensitivity` [F3 running], 
`muon_temporal_response` [T2 pending authorization], plus the `minimind_base`
vendored framework and `audits/` governance bundle). They are deliberately **not**
part of this archive.

## Provenance & integrity

- Source projects were treated strictly read-only during archiving.
- The FR/GTTR experiment index is transcribed from the frozen 39-column
  `01_COMPLETE_EXPERIMENT_INDEX.csv` (mirrored at
  `archived/fr_gttr/results/server_audit_20260926/`).
- continuation_adaptation sealed runs (448) are hash-locked in
  `archived/continuation_adaptation/results/outputs/p4_full/stage2/SEAL_LOCK.json`
  (metrics + final_state SHA256 per run).
- No experiment was re-run and no result was re-derived for this archive.
