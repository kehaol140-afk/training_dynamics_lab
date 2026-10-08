# INCREMENTAL ARCHIVE AUDIT — 2026-10-08

Pre-commit audit of the incremental archive round (optimizer_sensitivity, muon_transition,
muon_temporal_response, continuation errata, index docs).

## 1 Old-archive integrity

| Check | Result |
|---|---|
| `archived/fr_gttr/` | **UNTOUCHED** — `git diff HEAD` shows zero changes under fr_gttr; commit chain `6947af8`→`1c592ea` preserved (no rebase/reset/init) |
| `archived/continuation_adaptation/` SEAL | `SEAL_LOCK.json` sha256 **identical** to the server original (byte-for-byte); no results file modified; only `ERRATA_20261008.md` added |
| Working tree before this round | clean at `1c592ea` (= GitHub main HEAD) |

## 2 New content inventory

| Path | Files | Content |
|---|---|---|
| `archived/optimizer_sensitivity/code/` | 68 | src/opsens + scripts + tests (no checkpoints, no nested .git, no __pycache__) |
| `archived/optimizer_sensitivity/configs/` | 2 | frozen config JSONs |
| `archived/optimizer_sensitivity/protocols/` | 45 | all docs/*.md incl. every prereg/report/erratum + EXTERNAL_VALIDATION_PLAN + synthesis + decision memo |
| `archived/optimizer_sensitivity/results/outputs/` | 591 | summaries / verdict CSVs / figures / small logs (all <5 MB, no *.pt) |
| `archived/optimizer_sensitivity/reviews/2026-10-08/` | 77 | independent re-audit: report, UPDATE_DIFF, 5 CSVs, SELF_CHECK, PROGRESS, SOURCE_MANIFEST, analysis/{RECOMPUTE_NOTES,verify_all.py,verify_out.txt}, evidence/ (63), bundle zip |
| `archived/muon_transition/` | 277 | code 44 + protocols 9 + results 224 (all non-.pt outputs) |
| `archived/muon_temporal_response/` | 90 | code 17 + protocols 12 + configs + vendor_snapshot/audit/references 11 + results 50 (no *.pt; raw_natural/raw_sample excluded) |
| `archived/continuation_adaptation/ERRATA_20261008.md` | 1 | sealed-table row-count correction (448 runs vs 280 rows) + dual-CI documentation |
| `docs/` | 4 | ARCHIVE_COVERAGE_20261008, RESEARCH_INDEX, RESEARCH_TIMELINE, this file |
| `README.md` | 1 (modified) | five archived lines + pending list |

## 3 Scientific consistency resolution (STEP 4 of the task)

| Issue | Resolution |
|---|---|
| F4A strict healing count | **13/14** used everywhere (F4A_REPORT_ERRATUM; seed 2032 NOT_HEALED_IN_COVERAGE) |
| F4A.5 M0 fold-median MAE | canonical **0.4041** (`f4_consistency/f4a5_metric_recompute.json`); the 0.4219 in F4A5_METRIC_ERRATUM text is itself a typo — recorded in `archived/optimizer_sensitivity/ERRATA_20261008.md` E2 |
| F4B namespaces | canonical = `delta_0p05/` + `delta_0p02/` (rebuilt from raw, test-locked); top-level shared CSVs marked **HISTORICAL_INVALID** (content = δ0.02) — ERRATA E4 |
| F4 consistency counts | actual headline_numbers.csv = **14 rows: 10 VERIFIED / 3 STALE / 1 OVERWRITTEN**; the "12/3/1" in the audit doc and in review KEY_RESULTS KR-17 is a miscount — ERRATA E1 |
| continuation sealed table | 448 formal runs (SEAL) vs **280 rows** (56×5 arms) clarified; original archive text said 448 rows — corrected via continuation ERRATA E1 |
| continuation dual CIs | stage3 CI [1.2352,1.9550] vs 10-08 recompute [1.2283,1.9644] both retained with method note — ERRATA E2 |

Unresolved: none blocking. Standing rule: artifact values beat prose (ERRATA).

## 4 Size / security / duplication

- Files >100 MB: **0**. Files >5 MB: **0** (largest tracked file remains 3.13 MB fr_decision_utility summary from round 1).
- Secret scan (key/token/password patterns over py/sh/json/md/csv/yaml): **0 hits**.
- Duplicate large results: review-bundle `evidence/` overlaps `results/outputs/` by design
  (63 small evidence copies kept inside the review snapshot for self-containment); the
  `ADAMW_RESEARCH_REVIEW_UPDATED_BUNDLE.zip` (425 KB) duplicates the expanded review files —
  retained intentionally as the byte-stable snapshot; `old_bundle_extract/` was NOT copied.
- Totals: 6038 tracked-candidate files, 110 MB working tree (was 90 MB / 4869 files).

## 5 Excluded large artifacts (server-only; nothing deleted)

- optimizer_sensitivity: 212 files / **73.1 GB** — 6 × F3 `*.state.pt` sidecars (5.4–13.8 GB each),
  16 × F4A.1 5M matched-coverage bank states (63.3 MB each), oversized raw JSONs.
  Catalogue: `archived/optimizer_sensitivity/results/ARTIFACT_MANIFEST.md`.
- muon_transition: ~49 × `final_state.pt` ≈ 15 GB (incl. the T-line anchor
  `m1_muon_lr2.0_wd0.02_p0.5_s2030/final_state.pt` still referenced by muon_temporal configs).
- muon_temporal_response: `raw_natural.pt` 627 MB + `raw_sample_l1q.pt` 176 MB + ROUND2/3 zips (140 MB+).
- Server sources remain untouched (read-only copies throughout).
