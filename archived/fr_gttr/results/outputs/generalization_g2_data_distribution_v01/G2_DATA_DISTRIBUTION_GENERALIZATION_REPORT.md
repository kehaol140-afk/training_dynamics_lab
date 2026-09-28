# G2 — Data Distribution Generalization Report (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)

## 1. Question and design

- Sole question: with model scale / architecture / tokenizer / objective /
  AdamW / clipping / token budget / response protocol held fixed, does the
  full tangent C still predict the finite-difference response and beat
  optimizer-only B at H>1 when only the pretraining corpus changes (D0 -> D1)?
- D0 = frozen G1a pool `outputs/e3_confirmatory/v07/e3c_subset.jsonl` (read directly; never rerun).
- D1 = `outputs/generalization_g2_data_distribution_v01/data/pretrain_hq.jsonl` selected and frozen before any G2 response result;
  pool/partition rules fixed in G2_PROTOCOL.md (B2/B3/B4).
- This is NOT an algorithm-generalization experiment; D0 vs D1 cells are a
  matched descriptive comparison by (seed, stage, H) identity, NOT a
  sample-paired causal dataset experiment (training data differed).

## 2. Model / training configuration (fixed, identical to G1a)

- hidden=512 layers=7 heads=8 kv=4 head_dim=64 intermediate=1664 vocab=6400 seq=128
- unique params = 26681728 (must equal 26,681,728)
- AdamW betas=(0.9,0.999) eps=1e-8 wd=0.01; clip=1.0; bf16 GradScaler disabled;
  global batch 8 seqs x 128 tokens = 1024 tokens/optimizer-step; cosine LR;
  128 planned optimizer steps; states at steps 26/64/102 (~20.3/50/79.7%).

## 3. Dataset provenance summary

- D0: 4096 records, sha256 `1e657af9baae0784...`
- D1: 1413103 records, sha256 `82b19375460d44fe...`
- D1 pool: 4096 records (same size as D0 pool), sha256 `5d4c5e6b181e7791...`
- overlap diagnostic: exact duplicate lines = 820; sampled 13-gram shingle Jaccard = 0.0000

## 4. D1 official results (27 cells: 3 seeds x 3 stages x H={1,4,8})

| model | pooled R2 | corr | MAE | mean rel |
|---|---:|---:|---:|---:|
| B | 0.9114962992 | 0.9918458572 | 0.0178247738 | 0.0916795650 |
| C | 0.9999452367 | 0.9999876414 | 0.0004293364 | 0.0025988312 |

### per-horizon C

| H | C R2 | C mean rel | B mean rel |
|---|---:|---:|---:|
| 1 | 0.9998531159 | 0.0024390414 | 0.0024390414 |
| 4 | 0.9998802541 | 0.0030133082 | 0.0758277891 |
| 8 | 0.9999297443 | 0.0023441441 | 0.1967718646 |

## 5. Verdict (frozen B12 thresholds)

**STRONG_DATA_REPLICATION**

| condition | pass | value |
|---|---:|---|
| cond1_c_pooled_r2_ge_0.90 | True | 0.9999452366798873 |
| cond2_h_gt_1_win_fraction_ge_0.80 | True | {"pass": true, "wins": 18, "n": 18} |
| cond3_h8_paired_and_predictive | True | {"pass": true, "mean_err_C_h8": 0.002344144128852778, "mean_err_B_h8": 0.19677186457577328, "R2_C_h8": 0.999929744345081 |
| cond4_seed_positive_ge_2 | True | {"pass": true, "seed_means": {"2042": 0.20627016561867764, "2043": 0.07918806740165422, "2044": 0.11540506901374357}} |

## 6. D1 mechanism subset (seed 2042; B -> BH_fixed -> C, FD reference)

### 6a. stream-first aggregation (identical caliber to G1a report §7)

- gain_curvature = 0.1717813477
- gain_clip      = 0.0339615257

### 6b. official-cell diagnostic (same aggregation as §4)

- gain_curvature = 0.1724697629
- gain_clip      = 0.0338004027
- The two calibers differ only in the order of stream-mean and relative-
  error; both are reported and never mixed.

## 7. D0 vs D1 matched descriptive comparison (seed/stage/H identity)

| quantity | D0 (G1a frozen) | D1 (G2) |
|---|---:|---:|
| C pooled R2 | 0.999578 | 0.999945 |
| C pooled mean rel err | 0.007924 | 0.002599 |
| B pooled mean rel err | 0.120329 | 0.091680 |
| C<B win fraction (H>1) | 1.000000 | 1.000000 |
| mean(err_B - err_C) H>1 | 0.168608 | 0.133621 |
| training loss mean | 3.633267 | 3.557545 |
| training preclip norm mean | 1.390629 | 1.211487 |
| training clip rate | 1.000000 | 0.931373 |
| future first-step clip rate | 1.000000 | 0.861111 |
| future grad norm mean | 1.368350 | 1.093442 |
| future HVP norm mean | 0.283666 | 1.991102 |
| mech gain_curvature (stream-first) | 0.057376 | 0.171781 |
| mech gain_clip (stream-first) | 0.014718 | 0.033962 |

Full per-cell table: `results/d0_vs_d1_comparison.csv`

## 8. Bootstrap uncertainty (B15; D1 H>1 paired B/C improvement)

- observed mean paired improvement = 0.1336211007
- cluster bootstrap (seed>stage>cell, 20000 resamples, rng 20260916): mean 0.1331245902, median 0.1216679727, 95% PI [0.0671603068, 0.2512048005]
- fraction of bootstrap resamples with improvement > 0 = 1.00000 (NOT a p-value / posterior / significance probability)

## 9. Dynamics / distribution-shift diagnostics (B13; explanation only)

- training loss mean: D0 3.633267 vs D1 3.557545 (seed 2042, steps 1..102)
- training preclip norm mean: D0 1.390629 vs D1 1.211487
- training clip rate: D0 1.0000 vs D1 0.9314
- future first-step clip rate: D0 1.0000 vs D1 0.8611
- future grad norm mean (first step): D0 1.368350 vs D1 1.093442
- future HVP norm mean: D0 0.283666 vs D1 1.991102
- token statistics sample: see dataset_manifest.json token_statistics_sample (D1 mean token length 126.00 vs D0 114.93).
- These variables are context only (B13); no state was filtered on them.

## 10. Anomalies / incidents

- smoke gate: overall_pass=True, h1_analytic_vs_fd_rel_err=0.00182764
- baseline replay exactness: replay_pass_all=True
- max native-vs-analytic clip tensor L2 gap = 2.08475e-07; max HVP math-vs-native gradient relative gap = 0.00471247 (mixed-backend surrogate caveat, inherited)
- GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log
- D1 obtained from pre-deletion official HF revision (deleted from HEAD 2026-03-26); hash-verified; see DATASET_PROVENANCE.md
- no OOM; no retry; no RESOURCE_LIMIT

## 11. Scope statement

- G2 varies ONLY the pretraining corpus; it is not an algorithm experiment.
- D0/D1 comparison is matched descriptive by construction, not a
  sample-level paired causal dataset experiment.
- Mixed-backend HVP caveat inherited unchanged (MATH SDPA surrogate).
- 26M only; no extrapolation to 1B.

## 12. Next step

G2 complete. HANDOFF next = G1c (~1B stress test) — a user decision; NOT started by this experiment.
