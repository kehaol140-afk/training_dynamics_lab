# ARCHIVE COVERAGE — 2026-10-08

Inventory of all research units under `/home/b205/projects/training_dynamics_lab/`,
their completion status (from READMEs / final reports / artifacts / processes — not
plans), and archive coverage in this repository.

| Directory | Research unit | Status (evidence) | Archived here? |
|---|---|---|---|
| `legacy_fluctuation_response/` | FR / GTTR line (91 entries) | **TERMINATED_BY_PROTOCOL** 2026-09-26 (GTTR stop rules; frozen git HEAD `6ac4942`) | YES — `archived/fr_gttr/` (2026-09-29) |
| `continuation_adaptation/` | Optimizer-state adaptation under changed budget | **TERMINATED_BY_PROTOCOL** 2026-09-26 (P4 verdict STOP) | YES — `archived/continuation_adaptation/` + ERRATA_20261008 |
| `optimizer_sensitivity/` | Stable training / unstable sensitivities (F0→F4C) | **COMPLETED** 2026-09-30 (decision memo; consistency audit); 10-08 re-audit byte-verified. External validation plan **written, NOT executed** | YES — `archived/optimizer_sensitivity/` (this round) |
| `muon_transition_experiment/` | Optimizer × data mixture | **TERMINATED_BY_PROTOCOL** 2026-09-27 (NO_SIGNAL_STOP_LINE; M3 blocked by stop rule) | YES — `archived/muon_transition/` (this round) |
| `muon_temporal_response/` | Muon map temporal response (T0/T2a/T2b) | **ACTIVE→PAUSED with three complete bounded stages**; T2-full/T3/T4 NOT_STARTED; T2b version stopped by its decision table | YES (status-report form) — `archived/muon_temporal_response/` (this round) |
| `muon_spectral_allocation/` | Spectral allocation line (P0/P1→R0/R1→R2→R2F) | Closed loop, AH branch formally closed, HOLD_NO_NEW_TOPIC (per 10-08 history review; 12 GB) | **NO — pending** (candidate for next archive round) |
| `hyperball_reproduction/` | Hyperball / MuonH reproduction (A0+A1) | A0 complete; **A1 actually ran to completion but has no report and a stale README** (tuned Muon 3.27906 vs MuonH 3.27921 @3250 steps, single seed) — documentation debt | **NO — deliberately not auto-archived** (unreviewed external-origin project; needs human decision + report authoring) |
| `modded-nanogpt/` | Third-party code base (public repo from bundle, head `4ea6b937`) | Infrastructure for hyperball A1, not a research unit | NO (third-party; re-derivable from upstream) |
| `minimind_base/` | Vendored minimind framework (read-only) | Code base, no experiments | NO (mirrored inside fr_gttr provenance; server copy retained) |
| `audits/` | Governance bundles (2026-09-26 / 09-30 / 10-08 ×2 / taskpacks) | Read-only audit artifacts | PARTIAL — AdamW 10-08 bundle in `archived/optimizer_sensitivity/reviews/2026-10-08/`; muon history review cited but not fully copied (next round) |

Rules applied: plans are not results (external-validation plans, unstarted T2/T3/T4,
M3, F4C STAGE D are all marked not-executed); nothing of unknown provenance
(`hyperball_*`) was archived automatically.
