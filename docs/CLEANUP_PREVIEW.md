# CLEANUP PREVIEW — legacy_fluctuation_response & continuation_adaptation

> **NOTHING HAS BEEN DELETED.** This document only classifies server artifacts in the two
> archived research lines and estimates reclaimable space. Execute only after a human has
> verified the GitHub archive (push + spot-check). Generated 2026-09-29, read-only audit.
> Paths are relative to `/home/b205/projects/training_dynamics_lab/`.

Legend:
- `KEEP_SERVER_FOR_ACTIVE_DEPENDENCY` — required by ACTIVE projects (optimizer_sensitivity, muon_temporal_response). **DO NOT DELETE.**
- `KEEP_SOURCE_CODE` — tiny code/protocol trees; now also on GitHub, keep on server as working reference.
- `SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION` — evidence preserved on GitHub (summaries, seals, verdicts, hashes); raw bulk can go once the archive is verified.
- `REVIEW_BEFORE_REMOVE` — needs a human decision (frozen-but-unreferenced banks, datasets, retracted artifacts, duplicates, forensic evidence).

---

## 1. legacy_fluctuation_response (72 GB total)

### 1.1 KEEP_SERVER_FOR_ACTIVE_DEPENDENCY

| Path | Size | Reason |
|---|---|---|
| `outputs/e3_confirmatory/v07/` | 4.52 GB | v07 bank (56 full states). C0 set of continuation_adaptation P4 (per-file SHA256 in its frozen config) and general anchor bank for optimizer_sensitivity. Includes D0 corpus `e3c_subset.jsonl` (3 MB, SHA-frozen). |
| `outputs/generalization_g1a_26m_v01/bank/` | 2.79 GB | G1a 26M bank (9 states). optimizer_sensitivity F0/F1/F3 anchors (cells 2042/2043 @ step 26/64 come from this bank). |
| `outputs/generalization_g1b_100m_v01/bank/` | 10.30 GB | G1b 100M bank (9 states). optimizer_sensitivity F0/F1 100M anchors. |
| `outputs/generalization_g1b_100m_v01/theta/` | 3.38 GB | G1b q-state exports, part of the same frozen bank family. |

**Subtotal kept for active dependency: ~21.0 GB.**

### 1.2 KEEP_SOURCE_CODE (also archived to GitHub)

| Path | Size | Reason |
|---|---|---|
| `research/` (src, scripts, tests, configs) | ~9 MB | All experiment code; archived at `archived/fr_gttr/code/`. |
| `docs/`, root `FR_*.md` & preregistration docs, `AGENTS.md`, `DATASET_SHA256.txt`, `ENVIRONMENT_V07.txt` | ~5 MB | Protocols/status; archived at `archived/fr_gttr/protocols/`. |
| `tools/gttr_f0_ic_v1/` | <1 MB | GTTR-F0-IC verification package; archived. |
| `.git/` (repo at HEAD `6ac4942`, clean) | ~50 MB | Local git history (remote push never succeeded). Keep until GitHub archive verified; contains the only VCS history of this line. |
| `recovery_candidates/theta_protocol/` | 36 KB | Forensic evidence of the source-loss incident (#85); keep. |

### 1.3 SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION

Rationale: for every directory below, all verdict/summary/report/seal files ≤5 MB were
copied to `archived/fr_gttr/results/outputs/` (2913 files, ~48 MB, mirroring the exact
relative layout); only checkpoints / raw states / streams / logs remain server-side.
Full per-file list of what stayed behind: `archived/fr_gttr/results/ARTIFACT_MANIFEST.md`.

| Path | Size | Experiment(s) | Evidence on GitHub? | Active dependency? | Recommended action |
|---|---|---|---|---|---|
| `outputs/a0k_fresh_response_closed_loop_v01/` | 13.73 GB | A0K closed loop | summary.json + small files yes | none known | remove after verification |
| `outputs/a0n_phase_scheduler_pilot_v01/` | 7.42 GB | A0N | yes | none known | remove after verification |
| `outputs/fr_gttr_r2b_solved_update_bridge_v01/` | 4.48 GB | GTTR R2B | yes | none known | remove after verification |
| `outputs/a0o_lr_response_validation_v01/` | 4.45 GB | A0O | yes | none known | remove after verification |
| `outputs/generalization_g3_clipping_intervention_v01/` | 3.69 GB | G3 | yes | none known | remove after verification |
| `outputs/generalization_g1a_26m_v01/` (excl. `bank/`) | 0.90 GB | G1a aux outputs | yes | bank kept separately | remove non-bank part after verification |
| `outputs/fr_gttr_d1_direct_readout_v01/` | 1.86 GB | GTTR D1 | yes | none known | remove after verification |
| `outputs/e3_replication/` | 1.74 GB | E3R | yes | none known | remove after verification |
| `outputs/fr_gttr_phase_a_native_calibration_v01/` | 0.76 GB | Phase A | yes | none known | remove after verification |
| `outputs/fr_decision_utility_v01/` | 0.75 GB | A0G | yes | none known | remove after verification |
| `outputs/fr_gttr_f0_calfix_revalidation_v01/` | 0.56 GB | CALFIX | yes | none known | remove after verification |
| `outputs/fr_phase_b_gttr_action_scoring_v03/` | 0.57 GB | Phase B | yes | none known | remove after verification |
| `outputs/fr_c1_gttr_probe_validation_v01/` | 0.50 GB | C1 | yes | none known | remove after verification |
| `outputs/e3_mapping/` | 0.44 GB | E3 | yes | none known | remove after verification |
| `outputs/a0l_response_mixture_feasibility_v01/` | 0.37 GB | A0L | yes | none known | remove after verification |
| `outputs/fr_c1_late_completion_v01/` | 0.25 GB | C1L | yes | none known | remove after verification |
| `outputs/a0j_cached_future_sensitivity_v01/` | 0.14 GB | A0J | yes | none known | remove after verification |
| `outputs/fr_c0_gttr_validation_v01/` | 0.12 GB | C0 | yes | none known | remove after verification |
| `outputs/stage_c_v04/`, `e1_fluctuation/`, `e0_replay/`, `e4a*/`, `e4b*/`, `a0_fr_audit/` (excl. mv_theory_final), `a0h*`, `a0h15*`, `a0i*`, `a0s/t/u/q/p/m`, `b0*`, `cb1`, `f0/f1/f2a/f2b/ic` dirs | ~1.3 GB combined | E/A0/B0 chains | yes | none known | remove after verification |
| root `FR_A0C_*` prep packages (17 dirs) | <10 MB | A0C prep workspaces (authoritative results live in `outputs/a0_fr_audit/`) | n/a (workspaces) | none | remove after verification (negligible) |

**Subtotal SAFE (legacy): ~43.1 GB**

### 1.4 REVIEW_BEFORE_REMOVE

| Path | Size | Reason |
|---|---|---|
| `outputs/generalization_g2_data_distribution_v01/` (bank 2.79 GB + corpus `data/pretrain_hq.jsonl` 1.55 GB + aux) | 5.21 GB | G2 bank is a frozen 9-state bank not currently referenced by active projects, but it is the only copy; corpus is a dataset (license/size). Human call: keep bank? corpus re-downloadable? |
| `outputs/a0_fr_audit/mv_theory_final/` | 1.04 GB | Artifact of RETRACTED (do-not-cite) experiment. Candidate for earliest deletion, but retraction record argues for explicit human sign-off. |
| `outputs/generalization_g1c_*` (3 dirs) | 0.02 GB | External OLMo/1B line (abandoned); contains external tokenizer streams + proxy recovery report (internal-info). Tiny; review for licensing before any publication-side use. |
| `minimind/` | 1.17 GB | Duplicate of lab-root `minimind_base/` (SHA-identical file set). Deleting the legacy copy is safe **only if** `minimind_base/` (outside this cleanup scope) is confirmed kept. |
| `_fr_framework_backups/` (6 pre-v06 snapshots w/ git dirs) | <10 MB | Historical framework backups; contain old git objects. Tiny; keep or remove at will after review. |

**Subtotal REVIEW (legacy): ~7.4 GB**

---

## 2. continuation_adaptation (28 GB total)

### 2.1 KEEP_SOURCE_CODE

| Path | Size | Reason |
|---|---|---|
| `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `README.md` | ~0.6 MB | Archived at `archived/continuation_adaptation/`; keep server copy as reference. |

### 2.2 SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION

| Path | Size | Reason | Evidence on GitHub? | Recommended action |
|---|---|---|---|---|
| `outputs/p4_full/stage1/**/final_state.pt` (448 files) | 27.71 GB | Terminal experiment; every run's final_state SHA256 + model hash is sealed in `stage2/SEAL_LOCK.json` (on GitHub). All per-run metrics/loss curves/state statistics CSVs copied. | yes (hashes + metrics) | remove after verification |
| `outputs/p4_stage0/run/final_state.pt` | 0.06 GB | sanity-run state | yes | remove after verification |
| `outputs/test_p4_infra/**/final_state.pt` (3 files) | 0.19 GB | infrastructure test states | yes | remove after verification |
| `__pycache__/` (3 dirs) | <1 MB | cache | n/a | remove anytime |

**Subtotal SAFE (continuation): ~28.0 GB**

### 2.3 REVIEW_BEFORE_REMOVE

None — no active project depends on continuation_adaptation outputs (its C0 inputs live
in the legacy v07 bank, which is separately protected).

---

## 3. Totals

| Category | Size |
|---|---|
| KEEP_SERVER_FOR_ACTIVE_DEPENDENCY (banks, legacy) | **21.0 GB** |
| KEEP_SOURCE_CODE (both projects) | <0.1 GB |
| **SAFE_TO_REMOVE_AFTER_GITHUB_VERIFICATION** | **~71.1 GB** (legacy ~43.1 + continuation ~28.0) |
| REVIEW_BEFORE_REMOVE | ~7.4 GB |
| Disk headroom today | 119 GB free of 1007 GB |

Execution checklist before deleting anything:
1. Push this archive to GitHub and human-verify (spot-check EXPERIMENT_INDEX rows vs
   `results/outputs/...` evidence files).
2. Re-confirm no active project resolves paths into a candidate directory
   (`optimizer_sensitivity` configs reference v07/G1a/G1b by SHA256 — those stay).
3. Delete in descending-size order, one directory at a time, re-running the active
   projects' frozen-reference integrity checks after each batch.
