# G1b — ~100M Scale Generalization Report (FR_G1B_100M_SCALE_GENERALIZATION_V01)

## 1. Model configuration (scale-only)

- Codebase: `minimind/model/model_minimind.py + minimind/trainer/trainer_utils.py (frozen v07 generation, unchanged)` (frozen v07 generation, unmodified).
- hidden=768, layers=13, heads=12 (kv=6, head_dim=64), intermediate=2432 (frozen default formula), vocab=6400, seq_len=128.
- Unique parameters: **100,784,000** (anchor 4,983,168 -> 20.225x; G1a 26,681,728 -> 3.777x). Checkpoint tensor elements: 105,699,200.
- Static count on CPU BEFORE any training/response result; config frozen immediately (no other size tried).
- Tokenizer / position encoding / normalization / activation / attention implementation / loss: unchanged.

## 2. Training

- Seeds: 2042, 2043, 2044 (same seeds as G1a — paired-scale design, not a new independent seed replication).
- Schedule: 128 planned optimizer steps (256 micro), microbatch 4 x accum 2 = 1024 tokens/step, lr 5e-4 cosine, AdamW(0.9/0.999, eps 1e-8, wd 0.01), clip 1.0, bf16 autocast, GradScaler disabled.
- Stages (normalized progress, identical to G1a): EARLY=step 26 (20.31%, 26624 tokens); MIDDLE=step 64 (50%, 65536 tokens); LATE=step 102 (79.69%, 104448 tokens).
- Training terminated at LATE step 102 (no later registered state).

## 3. Official matrix

- 3 seeds x 3 stages = 9 states; 4 future streams per state = 36 baseline streams; H = [1, 4, 8] -> 27 official state-horizon cells.
- All 36 stream manifests were generated and frozen before any E2 response branch executed.
- Epsilon rule: rho=0.25 x sigma_q(16 E1 streams), per state: seed2042/step26=0.0268939, seed2042/step64=0.0095878, seed2042/step102=0.0147365, seed2043/step26=0.0339602, seed2043/step64=0.0167762, seed2043/step102=0.0192027, seed2044/step26=0.0292924, seed2044/step64=0.0171076, seed2044/step102=0.0318434.

## 4. FD / B / C pooled results (27 official rows)

| model | R2 | corr | MAE | RMSE | mean rel err | median rel err |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.9272524942 | 0.9946733685 | 0.01585864685 | 0.02755401882 | 0.07877855189 | 0.07054310757 |
| C | 0.9994888337 | 0.9999427741 | 0.001468065995 | 0.002309707704 | 0.008760484109 | 0.005078645046 |

## 5. By horizon

| H | R2 B | R2 C | corr B | corr C | MAE B | MAE C | mean rel B | mean rel C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9988899323 | 0.9988899323 | 0.9998855671 | 0.9998855671 | 0.000542766852 | 0.000542766852 | 0.008571854948 | 0.008571854948 |
| 4 | 0.9728332127 | 0.9989698985 | 0.998559342 | 0.9998888691 | 0.0099341201 | 0.001657693079 | 0.06180425758 | 0.008980911717 |
| 8 | 0.8107614913 | 0.9991714195 | 0.9926861803 | 0.999919916 | 0.0370990536 | 0.002203738053 | 0.1659595431 | 0.008728685661 |

## 6. B vs C (H>1 paired, 18 rows)

- C beats B: **18**, ties: **0**, C worse than B: **0** (win fraction 1).
- Mean paired improvement err_B - err_C: **+0.105027** (median +0.0803674).
- Per-seed mean paired improvement (H>1 rows): seed 2042: +0.109322, seed 2043: +0.0956783, seed 2044: +0.110081.
- Per-stage mean paired improvement (H>1 rows): EARLY: +0.147872, MIDDLE: +0.0797157, LATE: +0.0874933.

### Per-cell table (27 official rows)

| seed | stage | step | H | chi_FD | chi_B | chi_C | rel err B | rel err C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2042 | EARLY | 26 | 1 | -0.09316537528 | -0.09482675997 | -0.09482675997 | 0.01783264102 | 0.01783264102 |
| 2042 | EARLY | 26 | 4 | -0.2844962475 | -0.3075234061 | -0.2891586197 | 0.08094011339 | 0.01638816767 |
| 2042 | EARLY | 26 | 8 | -0.3920097767 | -0.4781744338 | -0.3976517558 | 0.2198023169 | 0.01439244486 |
| 2042 | MIDDLE | 64 | 1 | -0.05896519808 | -0.05903951796 | -0.05903951796 | 0.001260402472 | 0.001260402472 |
| 2042 | MIDDLE | 64 | 4 | -0.1797812068 | -0.1946224743 | -0.1802946576 | 0.08255182968 | 0.002855975925 |
| 2042 | MIDDLE | 64 | 8 | -0.2602459089 | -0.3067825533 | -0.2613021972 | 0.178817967 | 0.004058808313 |
| 2042 | LATE | 102 | 1 | -0.0217911634 | -0.02183799592 | -0.02183799592 | 0.002149151987 | 0.002149151987 |
| 2042 | LATE | 102 | 4 | -0.06868618197 | -0.07123977772 | -0.06899878689 | 0.03717772171 | 0.004551205358 |
| 2042 | LATE | 102 | 8 | -0.1010992268 | -0.1114934658 | -0.1014957425 | 0.1028122498 | 0.003922044625 |
| 2043 | EARLY | 26 | 1 | -0.05548250532 | -0.05668906113 | -0.05668906113 | 0.02174659923 | 0.02174659923 |
| 2043 | EARLY | 26 | 4 | -0.1651408505 | -0.1847706905 | -0.1686155577 | 0.1188672578 | 0.0210408705 |
| 2043 | EARLY | 26 | 8 | -0.2207175948 | -0.2901466249 | -0.2254492813 | 0.3145604689 | 0.02143774071 |
| 2043 | MIDDLE | 64 | 1 | -0.05430834439 | -0.05466536708 | -0.05466536708 | 0.006573993198 | 0.006573993198 |
| 2043 | MIDDLE | 64 | 4 | -0.1758727791 | -0.1803719356 | -0.1770035243 | 0.02558188097 | 0.006429336275 |
| 2043 | MIDDLE | 64 | 8 | -0.2645353132 | -0.285421071 | -0.2661897028 | 0.07895262653 | 0.006253946189 |
| 2043 | LATE | 102 | 1 | -0.01630922789 | -0.01636660892 | -0.01636660892 | 0.00351831625 | 0.00351831625 |
| 2043 | LATE | 102 | 4 | -0.05193329607 | -0.05322853052 | -0.05221219871 | 0.02494034747 | 0.005370401317 |
| 2043 | LATE | 102 | 8 | -0.07719218915 | -0.08311887036 | -0.07758422088 | 0.07677825024 | 0.005078645046 |
| 2044 | EARLY | 26 | 1 | -0.06924700164 | -0.07061156309 | -0.07061156309 | 0.01970571165 | 0.01970571165 |
| 2044 | EARLY | 26 | 4 | -0.2169749776 | -0.2322810668 | -0.2213117938 | 0.07054310757 | 0.01998763276 |
| 2044 | EARLY | 26 | 8 | -0.2991953244 | -0.3567152652 | -0.3041264372 | 0.1922487959 | 0.0164812495 |
| 2044 | MIDDLE | 64 | 1 | -0.04932160809 | -0.04940234327 | -0.04940234327 | 0.001636912822 | 0.001636912822 |
| 2044 | MIDDLE | 64 | 4 | -0.1583493558 | -0.1632180354 | -0.1584053984 | 0.03074644433 | 0.0003539178457 |
| 2044 | MIDDLE | 64 | 8 | -0.2342068097 | -0.2588474619 | -0.2350531431 | 0.1052089489 | 0.003613615684 |
| 2044 | LATE | 102 | 1 | -0.0132607011 | -0.01329680954 | -0.01329680954 | 0.002722965903 | 0.002722965903 |
| 2044 | LATE | 102 | 4 | -0.03988779075 | -0.04327384996 | -0.04004138658 | 0.08488961522 | 0.003850697801 |
| 2044 | LATE | 102 | 8 | -0.05521784017 | -0.06761171985 | -0.05540114551 | 0.224454264 | 0.003319676023 |

## 7. Mechanism subset (seed 2042, B -> BH_fixed -> C, FD reference)

| stage | H | mean rel err B | mean rel err BH_fixed | mean rel err C | gain_curvature (B-BHf) | gain_clip (BHf-C) |
|---|---:|---:|---:|---:|---:|---:|
| EARLY | 1 | 0.01781309039 | 0.01781309039 | 0.01781309039 | 0 | 0 |
| EARLY | 4 | 0.08096659536 | 0.01027042002 | 0.01636479689 | 0.07069617534 | -0.006094376866 |
| EARLY | 8 | 0.2197455912 | 0.06637536967 | 0.01437312124 | 0.1533702215 | 0.05200224843 |
| MIDDLE | 1 | 0.001250088415 | 0.001250088415 | 0.001250088415 | 0 | 0 |
| MIDDLE | 4 | 0.08271439615 | 0.04070922351 | 0.003113923051 | 0.04200517265 | 0.03759530046 |
| MIDDLE | 8 | 0.1792843449 | 0.08753866275 | 0.004062813909 | 0.09174568211 | 0.08347584884 |
| LATE | 1 | 0.002133185694 | 0.002133185694 | 0.002133185694 | 0 | 0 |
| LATE | 4 | 0.03739949692 | 0.01232437479 | 0.004640419809 | 0.02507512213 | 0.007683954976 |
| LATE | 8 | 0.1029806148 | 0.04637784885 | 0.003983133602 | 0.05660276595 | 0.04239471525 |

- H=1 structural identity max |BH_fixed - B| = 0 (requirement: B = BH_fixed = C at H=1).
- Pooled over 3 stages x H>1: gain_curvature = +0.0732492 (mean), gain_clip = +0.0361763 (mean).
- BH_old deprecated byproduct values were computed by the frozen runner but are excluded from all official G1b statistics.

## 8. Clipping regime across scales (diagnostic, front-line observation)

- 5M anchor (A0C 3 states x 8 streams): future first-step clip rate 0.25; future all-step clip rate n/a; first-step preclip norm mean 2.15372 (median 0.945805); training clip rate n/a (not recorded at anchor).
- G1a 26M (frozen): future first-step clip rate 1; future all-step clip rate 0.9895833333; first-step preclip norm mean 1.36835 (median 1.28288); training clip rate 1.
- G1b 100M (this experiment): future first-step clip rate 1; future all-step clip rate 1; first-step preclip norm mean 1.51011 (median 1.48443); training clip rate 1.
- gain_curvature / gain_clip by scale: 5M=0.06862689086/0.1147622861; 26M=0.05737619866/0.01471823763; 100M=0.07324918995/0.03617628185.
- These are three scale points of a registered observation for the planned G3 clipping intervention; NO empirical law is fitted across them and the cause of any gain change is NOT determined.

## 9. Scale comparison 5M / 26M / 100M (scale trend, NOT a strict paired test)

| scale | C pooled R2 | C MAE | C mean rel | C median rel | B mean rel | H8 mean rel C | H8 mean rel B | mean paired improvement (H>1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5M anchor (A0F2 census, 224 rows, H={1,2,4,8}; 56 states) | 0.9978128053 | 0.0007856978034 | 0.005872113467 | undefined | 0.1276611331 | 0.003987692355 | 0.3033932649 | 0.1217890196 |
| 26M G1a (27 rows, H={1,4,8}; 3 seeds x 3 stages) | 0.9995784934 | 0.001054991479 | 0.007923837804 | 0.002319798394 | 0.1203291912 | 0.00799383281 | 0.2366420377 | 0.16860803 |
| 100M G1b (27 rows, H={1,4,8}; 3 seeds x 3 stages) | 0.9994888337 | 0.001468065995 | 0.008760484109 | 0.005078645046 | 0.07877855189 | 0.008728685661 | 0.1659595431 | 0.1050271017 |
- 5M/26M rows are read from frozen artifacts (v07 A0C/A0E2 records + G1a summary); the three scales are NOT the same model at different sizes of training state, so this table is a scale trend, not a paired statistical test.

## 10. Verdict (frozen thresholds, G1B_PROTOCOL.md sec.8)

**STRONG_SCALE_REPLICATION**

| condition | pass | value |
|---|---:|---|
| cond1_c_pooled_r2_ge_0.90 | True | 0.9994888337  |
| cond2_h_gt_1_win_fraction_ge_0.80 | True | 1 {'wins': 18, 'n': 18} |
| cond3_h8_paired_and_predictive | True |  {'mean_err_C_h8': 0.008728685661468802, 'mean_err_B_h8': 0.1659595431384173, 'R2_C_h8': 0.9991714194871452, 'corr_C_h8': 0.9999199159556812} |
| cond4_seed_positive_ge_2 | True |  {'seed_means': {'2042': 0.10932225863091294, '2043': 0.09567831532525999, '2044': 0.11008073104669543}} |

## 11. Anomalies, OOM, retries, caveats

- smoke gate: overall_pass=True, h1_analytic_vs_fd_rel_err=0.0182915, gpu_peak_memory_mb=11032.28564453125, math_native_hvp_gap_max=0.00306406
- GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log; no external workload was killed
- no device microbatch adjustment was needed (batch 4 x accum 2 as registered)
- HVP uses the math-SDPA reverse-over-reverse surrogate (native flash-SDPA has no double backward): C/BH_fixed are mixed-backend analytic surrogates, same frozen caveat as A0C/A0E2/G1a.
- chi_FD uses the registered finite epsilon per state; C is a zero-pulse smooth tangent — the same finite-epsilon interpretation caveat as A0C/G1a applies.

## 12. Scope statement

G1b is model-scale generalization only (the second scale step). It does NOT replace A0F2, does NOT modify G1a, and does NOT by itself establish generalization at 1B or beyond.

## 13. Next step

G3 (clipping intervention, clip=1.0 vs no-clip) is a user decision; it is NOT started by this experiment.
