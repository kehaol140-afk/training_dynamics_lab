# training_dynamics_lab — Research Archive

Frozen research-archive repository for completed research lines from the
`training_dynamics_lab` research server (`/home/b205/projects/training_dynamics_lab/`).

Each archived line contains: code, frozen configs/protocols, small evidence files
(summaries / sealed evaluations / verdict JSONs / key figures), and evidence-linked
Markdown experiment summaries. Large artifacts (checkpoints, frozen state banks,
datasets, raw streams) are intentionally **not** stored here; they are catalogued
per-project with sizes, reasons and delete-safety flags.

## Archived projects

| Project | Status | Main conclusion | Code | Results |
|---|---|---|---|---|
| **FR / GTTR** | Terminated by protocol, frozen 2026-09-26 | Tangent propagation of full AdamW state explains short-horizon response (C R²=0.9977; replicates at 26M/100M/external data; clipping causally confirmed), but the preregistered FR universality law FAILED and every GTTR attempt to convert derivatives into a training rule failed its gate → line terminated ("derivatives computed correctly but not usable") | `archived/fr_gttr/code/` | `archived/fr_gttr/RESULTS.md`, `EXPERIMENT_INDEX.md` (91 entries) |
| **continuation_adaptation** | Terminal (P4 STOP) 2026-09-26 | Across 56 cells × 8 arms (448 sealed runs; sealed table 280 rows = 56×5 arms), simple state adaptation does **not** beat tuned continuation (C1 −A +1.587, C2 −A +0.019, Holm p=1.0) → STOP; inherited state still descriptively beats reset (B−D −0.075) | `archived/continuation_adaptation/code/` | `RESULTS.md` + `ERRATA_20261008.md` |
| **optimizer_sensitivity** | Completed 2026-09-30 (F0→F4C + decision memo; re-audited 2026-10-08) | "Stable training, unstable sensitivities": v-floor ill-conditioning + direction-selective early curvature excitation (spectral instability, metric-dependent coupling) + phase self-healing (13/14 strict; LATE 24/24 curvature-neutral); macro-predictable (M0 LOSO MAE 0.4249) / micro-unpredictable (prescreen balacc 0.539); **SAU = NO_METHOD_GAIN (not a successful method)**; external validation written but NOT executed | `archived/optimizer_sensitivity/code/` | `RESULTS.md`, `EXPERIMENT_INDEX.md`, `ERRATA_20261008.md`, `reviews/2026-10-08/` |
| **muon_transition** | Terminal negative 2026-09-27 | AdamW→Muon does not change the optimal data mixture (p\*=0.5 both; R=0.0000; NO_SIGNAL_STOP_LINE; M3 blocked by stop rule) | `archived/muon_transition/code/` | `archived/muon_transition/RESULTS.md` |
| **muon_temporal_response** | Three bounded stages complete; T2-full/T3/T4 never started | T0: direction model accurate, scalar incapable (native arm1 B2/A 62.32%); T2a: toy dead-zone does not transfer to real scale; T2b: `NO_USEFUL_SIGNAL_THIS_SETTING` (candidate spread 30× below screen threshold; frozen replay fails at 138 steps) | `archived/muon_temporal_response/code/` | `archived/muon_temporal_response/RESULTS.md` (status form) |

## Repository layout

```
├── README.md                          this file
├── ARCHIVE_STAGING_REPORT.md          2026-09-29 first-round audit trail
├── archived/
│   ├── fr_gttr/                       frozen 2026-09-26, archived 2026-09-29
│   ├── continuation_adaptation/       terminal 2026-09-26, archived 2026-09-29, errata 2026-10-08
│   ├── optimizer_sensitivity/         complete 2026-09-30, archived 2026-10-08 (incl. reviews/)
│   ├── muon_transition/               terminal 2026-09-27, archived 2026-10-08
│   └── muon_temporal_response/        archived 2026-10-08 (status-report form)
└── docs/
    ├── CLEANUP_PREVIEW.md             2026-09-29 cleanup classification (executed same day)
    ├── ARCHIVE_COVERAGE_20261008.md   per-directory status + coverage matrix
    ├── RESEARCH_INDEX.md              one-row-per-line index
    ├── RESEARCH_TIMELINE.md           artifact-dated timeline
    └── INCREMENTAL_ARCHIVE_AUDIT.md   2026-10-08 integrity/security/diff audit
```

## Other active / pending projects

Other projects remain on the research server and will be archived separately:
`muon_spectral_allocation` (closed loop, pending archive round),
`hyperball_reproduction` (A1 ran but undocumented — needs report + provenance review
before archiving), `modded-nanogpt` (third-party base), `minimind_base` (vendored
framework), `audits/` (governance bundles, partially mirrored).

## Provenance & integrity

- Source projects were treated strictly read-only during archiving; no experiment was
  re-run and no result was re-derived.
- FR/GTTR index transcribed from the frozen 39-column CSV (mirrored at
  `archived/fr_gttr/results/server_audit_20260926/`).
- continuation_adaptation: 448 sealed runs hash-locked in SEAL_LOCK.json (byte-identical
  since 2026-09-26, verified again 2026-10-08).
- optimizer_sensitivity: independent 2026-10-08 re-audit bundle included under
  `archived/optimizer_sensitivity/reviews/2026-10-08/` (byte-verified F0–F3.5, full F4
  chain review, recompute script + output).
- Numeric corrections are consolidated in per-project `ERRATA_20261008.md` files;
  original server reports are never modified.
