# G1a — 26M Scale Generalization Report (FR_G1A_26M_SCALE_GENERALIZATION_V01)

## 1. Model configuration (scale-only)

- Codebase: `minimind/model/model_minimind.py + minimind/trainer/trainer_utils.py (frozen v07 generation, unchanged)` (frozen v07 generation, unmodified).
- hidden=512, layers=7, heads=8 (kv=4, head_dim=64), intermediate=1664 (frozen default formula), vocab=6400, seq_len=128.
- Unique parameters: **26,681,728** (anchor 4,983,168; ratio 5.354x). Checkpoint tensor elements: 29,958,528.
- Tokenizer / position encoding / normalization / activation / attention implementation / loss: unchanged.

## 2. Training

- Seeds: 2042, 2043, 2044 (preregistered; no identity conflict).
- Schedule: 128 planned optimizer steps (256 micro), microbatch 4 x accum 2 = 1024 tokens/step, lr 5e-4 cosine, AdamW(0.9/0.999, eps 1e-8, wd 0.01), clip 1.0, bf16 autocast, GradScaler disabled.
- Stages (normalized progress, not 46/64/82): EARLY=step 26 (20.31%, 26624 tokens); MIDDLE=step 64 (50%, 65536 tokens); LATE=step 102 (79.69%, 104448 tokens).
- Training terminated at LATE step 102 (no later registered state).

## 3. Official matrix

- 3 seeds x 3 stages = 9 states; 4 future streams per state = 36 baseline streams; H = [1, 4, 8] -> 27 official state-horizon cells.
- All 36 stream manifests were generated and frozen before any E2 response branch executed.
- Epsilon rule: rho=0.25 x sigma_q(16 E1 streams), per state: seed2042/step26=0.0103524, seed2042/step64=0.00912637, seed2042/step102=0.0123983, seed2043/step26=0.0383323, seed2043/step64=0.00972002, seed2043/step102=0.017137, seed2044/step26=0.0316633, seed2044/step64=0.206397, seed2044/step102=0.0112139.

## 4. FD / B / C pooled results (27 official rows)

| model | R2 | corr | MAE | RMSE | mean rel err | median rel err |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.832039728 | 0.9730019246 | 0.02164495592 | 0.0401865582 | 0.1203291912 | 0.06216756589 |
| C | 0.9995784934 | 0.9998508973 | 0.001054991479 | 0.00201316651 | 0.007923837804 | 0.002319798394 |

## 5. By horizon

| H | R2 B | R2 C | corr B | corr C | MAE B | MAE C | mean rel B | mean rel C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9988861975 | 0.9988861975 | 0.999695307 | 0.999695307 | 0.0005042901573 | 0.0005042901573 | 0.009035582607 | 0.009035582607 |
| 4 | 0.8684408257 | 0.999328579 | 0.9697585559 | 0.9997550931 | 0.01689481351 | 0.001042175126 | 0.1153099532 | 0.006742097995 |
| 8 | 0.6358165449 | 0.9993109104 | 0.9517051848 | 0.999756961 | 0.0475357641 | 0.001618509154 | 0.2366420377 | 0.00799383281 |

## 6. B vs C (H>1 paired, 18 rows)

- C beats B: **18**, ties: **0**, C worse than B: **0** (win fraction 1).
- Mean paired improvement err_B - err_C: **+0.168608** (median +0.0954886).
- Per-seed mean paired improvement (H>1 rows): seed 2042: +0.0732236, seed 2043: +0.111241, seed 2044: +0.32136.
- Per-stage mean paired improvement (H>1 rows): EARLY: +0.149335, MIDDLE: +0.296569, LATE: +0.05992.

### Per-cell table (27 official rows)

| seed | stage | step | H | chi_FD | chi_B | chi_C | rel err B | rel err C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2042 | EARLY | 26 | 1 | -0.0954121011 | -0.09553394918 | -0.09553394918 | 0.001277071642 | 0.001277071642 |
| 2042 | EARLY | 26 | 4 | -0.2916069137 | -0.3106446137 | -0.2920012973 | 0.06528548896 | 0.001352449428 |
| 2042 | EARLY | 26 | 8 | -0.4205205459 | -0.4887044827 | -0.4205989001 | 0.162141749 | 0.0001863266514 |
| 2042 | MIDDLE | 64 | 1 | -0.05169990167 | -0.05172033764 | -0.05172033764 | 0.0003952805486 | 0.0003952805486 |
| 2042 | MIDDLE | 64 | 4 | -0.1653176672 | -0.170983748 | -0.165314447 | 0.03427389757 | 1.947884826e-05 |
| 2042 | MIDDLE | 64 | 8 | -0.2457649025 | -0.2714660891 | -0.2454263045 | 0.1045763098 | 0.001377731475 |
| 2042 | LATE | 102 | 1 | -0.01616950451 | -0.01620590599 | -0.01620590599 | 0.002251242589 | 0.002251242589 |
| 2042 | LATE | 102 | 4 | -0.05196130563 | -0.05305570214 | -0.05209383077 | 0.02106175925 | 0.002550458268 |
| 2042 | LATE | 102 | 8 | -0.07809532875 | -0.08295032525 | -0.07846071222 | 0.06216756589 | 0.004678685303 |
| 2043 | EARLY | 26 | 1 | -0.05076588757 | -0.05206954107 | -0.05206954107 | 0.02567971446 | 0.02567971446 |
| 2043 | EARLY | 26 | 4 | -0.1540881901 | -0.1718489824 | -0.1581395122 | 0.1152638128 | 0.0262922298 |
| 2043 | EARLY | 26 | 8 | -0.2146999467 | -0.2727870838 | -0.2205407119 | 0.2705503097 | 0.02720431604 |
| 2043 | MIDDLE | 64 | 1 | -0.0524533709 | -0.05251041954 | -0.05251041954 | 0.001087606599 | 0.001087606599 |
| 2043 | MIDDLE | 64 | 4 | -0.161566004 | -0.1729070824 | -0.161389269 | 0.07019470778 | 0.001093887521 |
| 2043 | MIDDLE | 64 | 8 | -0.2364874133 | -0.2739529283 | -0.2368796619 | 0.1584249853 | 0.001658644599 |
| 2043 | LATE | 102 | 1 | -0.01555967967 | -0.0155704532 | -0.0155704532 | 0.0006924006466 | 0.0006924006466 |
| 2043 | LATE | 102 | 4 | -0.04963854039 | -0.05077830754 | -0.04955612495 | 0.02296133497 | 0.001660311452 |
| 2043 | LATE | 102 | 8 | -0.07275306399 | -0.0793261779 | -0.07257913079 | 0.09034827611 | 0.00239073369 |
| 2044 | EARLY | 26 | 1 | -0.06002601754 | -0.06138376789 | -0.06138376789 | 0.02261936412 | 0.02261936412 |
| 2044 | EARLY | 26 | 4 | -0.1794384084 | -0.2017609772 | -0.1834572437 | 0.124402401 | 0.02239673997 |
| 2044 | EARLY | 26 | 8 | -0.254540513 | -0.3204119938 | -0.260391467 | 0.2587858413 | 0.02298633669 |
| 2044 | MIDDLE | 64 | 1 | -0.06025440087 | -0.06187887989 | -0.06187887989 | 0.02696033806 | 0.02696033806 |
| 2044 | MIDDLE | 64 | 4 | -0.1313849512 | -0.2028782363 | -0.1309916466 | 0.5441512481 | 0.002993528269 |
| 2044 | MIDDLE | 64 | 8 | -0.1702141126 | -0.3203080276 | -0.1713816988 | 0.8817947741 | 0.006859514673 |
| 2044 | LATE | 102 | 1 | -0.01741440936 | -0.01742063022 | -0.01742063022 | 0.000357224798 | 0.000357224798 |
| 2044 | LATE | 102 | 4 | -0.0546748695 | -0.05687252195 | -0.05454803483 | 0.040194928 | 0.002319798394 |
| 2044 | LATE | 102 | 8 | -0.07795382604 | -0.08894442125 | -0.07759506646 | 0.1409885284 | 0.004602206168 |

## 7. Mechanism subset (seed 2042, B -> BH_fixed -> C, FD reference)

| stage | H | mean rel err B | mean rel err BH_fixed | mean rel err C | gain_curvature (B-BHf) | gain_clip (BHf-C) |
|---|---:|---:|---:|---:|---:|---:|
| EARLY | 1 | 0.001266172739 | 0.001266172739 | 0.001266172739 | 0 | 0 |
| EARLY | 4 | 0.06561048464 | 0.007689321307 | 0.001438695319 | 0.05792116333 | 0.006250625988 |
| EARLY | 8 | 0.161968008 | 0.0314660251 | 0.00278145611 | 0.1305019829 | 0.02868456899 |
| MIDDLE | 1 | 0.0003870212835 | 0.0003870212835 | 0.0003870212835 | 0 | 0 |
| MIDDLE | 4 | 0.03422097393 | 0.009396994079 | 0.002086386326 | 0.02482397985 | 0.007310607752 |
| MIDDLE | 8 | 0.1041650053 | 0.03504261612 | 0.002625378851 | 0.06912238919 | 0.03241723727 |
| LATE | 1 | 0.002220797652 | 0.002220797652 | 0.002220797652 | 0 | 0 |
| LATE | 4 | 0.0210627293 | 0.005358770682 | 0.002571490269 | 0.01570395862 | 0.002787280413 |
| LATE | 8 | 0.06237307923 | 0.01618936114 | 0.005330255778 | 0.04618371809 | 0.01085910536 |

- H=1 structural identity max |BH_fixed - B| = 0 (requirement: B = BH_fixed = C at H=1).
- Pooled over 3 stages x H>1: gain_curvature = +0.0573762 (mean), gain_clip = +0.0147182 (mean).
- BH_old deprecated byproduct values were computed by the frozen runner but are excluded from all official G1a statistics.

## 8. Training dynamics context vs 5M anchor

- G1a (26M) future-step clip activation rate (step 1 of 36 baseline streams): 1; 5M anchor (A0C 3 states, 8 streams each): 0.25.
- G1a first-step preclip grad norm mean over streams: 1.36835 (median 1.28288); anchor 5M: 2.15372 (median 0.945805).
- G1a training-phase clip activation (bank steps): 1; mean preclip norm: 1.39063.
- These variables are context only; no state was filtered on them.

## 9. Verdict (frozen thresholds, G1A_PROTOCOL.md §10)

**STRONG_SCALE_REPLICATION**

| condition | pass | value |
|---|---:|---|
| cond1_c_pooled_r2_ge_0.90 | True | 0.9995784934  |
| cond2_h_gt_1_win_fraction_ge_0.80 | True | 1 {'wins': 18, 'n': 18} |
| cond3_h8_paired_and_predictive | True |  {'mean_err_C_h8': 0.007993832810363855, 'mean_err_B_h8': 0.23664203773652956, 'R2_C_h8': 0.9993109104328931, 'corr_C_h8': 0.9997569610322016} |
| cond4_seed_positive_ge_2 | True |  {'seed_means': {'2042': 0.07322360675403, '2043': 0.11124055060270589, '2044': 0.3213599327808822}} |

## 10. Anomalies, OOM, retries, caveats

- smoke gate: overall_pass=True, h1_analytic_vs_fd_rel_err=0.000347625
- GPU sharing recorded at run time; see ENVIRONMENT.json / COMMANDS.log
- HVP uses the math-SDPA reverse-over-reverse surrogate (native flash-SDPA has no double backward): C/BH_fixed are mixed-backend analytic surrogates, same frozen caveat as A0C/A0E2.
- chi_FD uses the registered finite epsilon per state; C is a zero-pulse smooth tangent — the same finite-epsilon interpretation caveat as A0C applies.

## 11. Scope statement

G1a is model-scale generalization only. It does NOT replace A0F2 (56-state census at 5M) and does NOT by itself establish generalization at 100M/1B or beyond.

## 12. Next step

G1b (~100M) is a user decision; this experiment does not start it.
