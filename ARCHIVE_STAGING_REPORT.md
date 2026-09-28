# ARCHIVE STAGING REPORT

- Date: 2026-09-29
- Staging repo: `/home/b205/projects/training_dynamics_lab_archive_staging`
- Branch: `main`; Remote: `https://github.com/kehaol140-afk/training_dynamics_lab.git`
- Commit: **`6947af8eb9ff93b95be5370df20796cb7f46a6c5`** — "Archive FR/GTTR and continuation adaptation experiments"
- Working tree after commit: clean (0 uncommitted paths)
- Method: read-only copy from the research server; no source file was modified, moved, renamed or deleted; no experiment was re-run; all conclusions transcribed from frozen artifacts.

## FR / GTTR (source: `legacy_fluctuation_response/`)

- Code files copied: **333** (`research/src` + `scripts` + `tests` + `configs` + `a0h15` + `adapters` + `analysis` + `reports` + `tools/gttr_f0_ic_v1` + root gradient script; `__pycache__` excluded; `theta_protocol.pyc` deliberately retained and documented as sourceless bytecode)
- Protocol/document files copied: **61** (18 root `FR_*.md` + preregistration `CONFIRMATORY_PREREGISTRATION_V07.md` + `AGENTS.md` + environment/dataset sha records + CODEX audit docs + `docs/` selection incl. `FR_PROJECT_STATUS.md`, `FR_EXPERIMENT_INDEX.md`, research manual v2 (pdf) & plan, handoffs, retrospectives, `specs/`, `project_inventory/`, `report_audit/`)
- Experiments indexed: **91 / 91** entries in `EXPERIMENT_INDEX.md` (transcribed from the frozen 39-column CSV; includes positive / negative / failed-gate / blocked / cancelled / voided / retracted / exploratory)
- Result files preserved: **2926** (2913 small evidence files mirroring `outputs/` layout — summary.json / REPORT.md / verdict CSVs / seals / small figures, each <5 MB; + 12-doc server audit bundle incl. the source-of-truth CSV; + ARTIFACT_MANIFEST.md)
- Key documents authored: `README.md`, `RESULTS.md` (11 sections, evidence-linked), `EXPERIMENT_INDEX.md`, `results/ARTIFACT_MANIFEST.md`

## continuation_adaptation (source: `continuation_adaptation/`)

- Code files copied: **33** (`src/` incl. 12 vendored_fr frozen modules, `scripts/`, `tests/`; no datasets/checkpoints)
- Configs copied: **2** (`smoke_v01.json`, `p4_full.json` with 56-cell SHA256 manifest)
- Protocol files copied: **6** (README + PROJECT_CONTEXT + EXPERIMENT_PROTOCOL + P4_PROTOCOL + P4_PROTOCOL_REVIEW + LEGACY_PROVENANCE)
- Result files preserved: **1500** (ALL non-checkpoint outputs: smoke_report.json; per-run metrics.json / loss_curve.csv / optimizer_state_statistics.csv for all 448 runs; 56 selection ledgers; stage0 run metrics; SEAL_LOCK.json; sealed_evaluation.json + table.csv; analysis_report.json; test_p4_infra artifacts; 8 small logs)
- Key documents authored: `README.md`, `RESULTS.md` (evidence-linked, includes the 56/56 boundary-unresolved limitation)

## Repo statistics

| Metric | Value |
|---|---|
| Tracked files | **4869** |
| Working-tree size (excl. `.git`) | **90 MB** (`.git` object store: 47 MB) |
| Largest tracked file | 3.13 MB (`archived/fr_gttr/results/outputs/fr_decision_utility_v01/summary.json`) |
| Files > 5 MB | 0 |
| Files > 100 MB | 0 |

## Excluded artifacts (remain on server, catalogued)

| Line | Excluded files | Excluded size | Catalogue |
|---|---|---|---|
| FR / GTTR | 3160 | 70.3 GB | `archived/fr_gttr/results/ARTIFACT_MANIFEST.md` (full per-file CSV embedded) |
| continuation_adaptation | 452 (`final_state.pt` ×452) | 28.0 GB | `archived/continuation_adaptation/RESULTS.md` §Removed-from-GitHub + SEAL_LOCK hashes |
| **Total** | **3612** | **98.3 GB** | |

## Potential server cleanup (preview only — nothing deleted)

| Category | Size |
|---|---|
| SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION | **~71.1 GB** (legacy ~43.1 + continuation ~28.0) |
| REVIEW_BEFORE_REMOVE | ~7.4 GB |
| KEEP_SERVER_FOR_ACTIVE_DEPENDENCY | ~21.0 GB |
Details: `docs/CLEANUP_PREVIEW.md`.

## FR banks marked DO_NOT_DELETE_YET

- `legacy_fluctuation_response/outputs/e3_confirmatory/v07/` — 4.52 GB (56 states; P4 C0 set + optimizer_sensitivity anchors)
- `legacy_fluctuation_response/outputs/generalization_g1a_26m_v01/bank/` — 2.79 GB (optimizer_sensitivity F0/F1/F3 anchors)
- `legacy_fluctuation_response/outputs/generalization_g1b_100m_v01/{bank,theta}/` — 13.68 GB (optimizer_sensitivity F0/F1 100M anchors)

## Potential secrets

None found. Scan of staged content for `.env*` / credential / id_rsa / *.pem / *.key / *.kdbx filenames: 0 hits. Regex scan for api-key / token / password / hf_/ghp_/sk- assignments in all tracked text files: 0 hits. (Filename-level "token" hits are tokenizer data only.)

## Integrity statements

- Source directories modified: **NO** (both sources treated read-only; verified by their unchanged mtimes / git status where applicable)
- Push performed: **NO** (per instruction — awaiting human verification)
- Experiments re-run: **NO**
- Conclusions authored beyond frozen artifacts: **NONE** (unknown items stay UNKNOWN; retracted/voided work marked DO-NOT-CITE)
