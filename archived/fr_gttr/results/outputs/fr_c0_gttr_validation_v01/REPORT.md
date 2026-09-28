# C0-GTTR Validation Report

- experiment: `FR_C0_GTTR_VALIDATION_V01` (frozen config sha256 `e00ce243cd05527b…`)
- date: 2026-09-21
- round scope: Geometry -> Tracking -> Transport -> Readout validation, 2 cells x 3 arms x 64 real optimizer steps + 24 anchors
- this report is produced post hoc by `research/scripts/c0_analyze_v01.py`; no new optimizer updates are consumed by analysis.

## 0. Verdict dictionary

| verdict | status |
|---|---|
| formula_implementation | **PASS** |
| local_training_field_prediction | **LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT** |
| local_shadow_update_prediction | **LOCAL_GEOMETRY_UNRESOLVED** |
| tracking_signal_resolution | **PASS** |
| readout_accounting | **PASS** |
| future_H16_H64_prediction | **NOT_TESTED** |
| adaptive_training_value | **NOT_TESTED** |
| energy_benefit | **NOT_TESTED** |

Fixed NOT_TESTED statements (independent): H=16/H=64 horizon gain prediction, real online policy value, final training improvement, energy benefit are NOT tested by this round and claim nothing.

## 1. Budget ledger

- project cumulative after C0 (reconciled): **17994** / cap 18000 (approved this round: 417 = 33 disclosed smoke-attempt waste + 384 main; runner's raw last row says 17961 because it uses the static config base and omits the reconciled 33)
- diagnostics totals vs caps: grad-evals 1084/1400, HVP 248/256, forwards 2428/2400, GPU-active wall 795s/3600
- caps: per-process all within caps (runner-enforced). Round-total logged sums: grads 1084/1400, HVP 248/256, forwards 2428/2400 (overage = zero-update salvage floors, disclosed), wall 795s/3600.
- smoke attempts 1 and 2 failed on runner bugs after 9 and 24 real updates; both are counted in `budget_log.jsonl` (RECONCILIATION rows) and disclosed here. User approved +9 over the original 408 plan.

## 2. Smoke gates (with salvage disclosure)

- hvp proxy-vs-native gap gate: **PASS** (max rel gap 0.005356)
- **numerical floor supersession (disclosed):** the smoke and floor phases computed `max_pairwise_diff` over a list that MIXED different probes, so the recorded floor measured cross-probe gradient diversity, not numerical repeatability — contrary to the frozen definition ('max ||diff|| of repeated IDENTICAL evaluations').  A zero-update `--phase floor2` process re-measured the floor pairing repeats of the SAME probe only.  Both values are reported; the corrected floor is operative for the frozen gate; no threshold was changed.
  - original pooled floor: {'grad': 1.6500472516120002, 'hvp': 0.17023550631815346, 'v': 0.0, 'q': 0.0}
  - corrected same-probe floor (operative): {'grad': 0.0, 'hvp': 0.0, 'v': 0.0, 'q': 0.0}
- diag-identity (reconstructed, 0 new updates): **PASS**
  - internal (attempt-2 rows, diag off vs on, h=1..8): True, max diffs {'lr_native': 0.0, 'lr_applied': 0.0, 'preclip_norm': 0.0, 'clip_active': 0.0, 's_norm': 0.0}
  - external (main cell-R arm-A vs B0 stress_diag native 2030/46/7110, h=1..64): True, max diffs {'lr_applied': 0.0, 'lr_native': 0.0, 'preclip_norm': 0.0, 's_norm_vs_update_norm': 1.248508321727293e-08, 'clip_active_mismatch': 0}

## 3. A0T cross-check (v_eval, tolerance 5e-4)

- status: **AXIS_MISMATCH** (all-within-tolerance flag: False; mean |diff| 0.1061 vs tolerance 0.0005)
- axis finding: Systematic ~0.1 offset at EVERY (cell, arm, h): provenance check shows A0T cells.jsonl losses were read on the FULL v_score split (a0q.run_channel_rollout endpoint -> a0g._val_loss_only over rt['val_score_tensors']), while C0 v_eval is the dedicated 64-sample split and C0 v_diag is the 8-ID prefix of v_score.  No C0 readout shares the A0T axis, so the 5e-4 absolute check is not satisfiable post hoc.  Trajectory identity for cell-R arm-A is established independently by the B0 stress_diag bitwise gate.  Arm mapping: C0 Bp<->A0T B (fixed +0.05); C0 Bm<->A0T C only for cell R (u=-0.05 controller); cell C A0T C==B (aliased).

## 4. Per-cell gains (v_diag trajectory; v_eval at h=4/16/64)

| cell | arm | v_diag h1 | v_diag h64 | ΔV_diag | gain vs A (ΔV) | v_eval h4 | h16 | h64 |
|---|---|---|---|---|---|---|---|---|
| R | A | 7.32765 | 7.12426 | -0.203387 | 0 | 7.16687 | 7.09946 | 6.97036 |
| R | Bp | 7.32765 | 7.11631 | -0.211335 | -0.00794792 | 7.16901 | 7.09868 | 6.96426 |
| R | Bm | 7.32765 | 7.13674 | -0.190906 | 0.0124807 | 7.16505 | 7.10061 | 6.97807 |
| C | A | 7.32765 | 7.14876 | -0.178894 | 0 | 7.15234 | 7.09432 | 6.97678 |
| C | Bp | 7.32765 | 7.14196 | -0.18569 | -0.00679636 | 7.15149 | 7.09559 | 6.97233 |
| C | Bm | 7.32765 | 7.15436 | -0.173288 | 0.00560522 | 7.15287 | 7.09528 | 6.98156 |

## 5. Local geometry (training field, channel A: y_mu)

Resolution gate: ||ybar|| > 3*sqrt(SE^2 + floor^2), floor 0 (engineering gate, not significance).

| cell | arm | t | resolved | E_M0 | E_Mgeom | E_MnoJ | halfbank f4/l4 |
|---|---|---|---|---|---|---|---|
| R | A | 0 | True | 1 | 0.398472 | 0.398472 | 0.527631/0.547564 |
| R | A | 4 | False | 1 | 1.69567 | 2.80269 | 1.40222/2.32945 |
| R | A | 16 | False | 1 | 0.867149 | 0.986737 | 0.696129/1.42185 |
| R | A | 48 | True | 1 | 0.422086 | 0.423156 | 0.545561/0.591684 |
| R | Bp | 0 | True | 1 | 0.40218 | 0.40218 | 0.531549/0.551186 |
| R | Bp | 4 | False | 1 | 1.7332 | 3.79509 | 1.43399/2.35333 |
| R | Bp | 16 | True | 1 | 0.577696 | 0.601795 | 0.577922/0.945885 |
| R | Bp | 48 | True | 1 | 0.447496 | 0.45283 | 0.595175/0.64241 |
| R | Bm | 0 | True | 1 | 0.395559 | 0.395559 | 0.524938/0.544718 |
| R | Bm | 4 | False | 1 | 1.53222 | 1.95026 | 1.32421/2.04133 |
| R | Bm | 16 | False | 1 | 1.44646 | 1.86738 | 0.82307/2.43057 |
| R | Bm | 48 | True | 1 | 0.432638 | 0.433909 | 0.562046/0.614718 |
| C | A | 0 | True | 1 | 0.352439 | 0.352439 | 0.46328/0.50081 |
| C | A | 4 | False | 1 | 0.677402 | 0.685382 | 0.780323/0.910932 |
| C | A | 16 | False | 1 | 0.734942 | 0.753144 | 0.560265/1.54587 |
| C | A | 48 | True | 1 | 0.427602 | 0.437114 | 0.547147/0.627214 |
| C | Bp | 0 | True | 1 | 0.356584 | 0.356584 | 0.4692/0.504357 |
| C | Bp | 4 | False | 1 | 0.68978 | 0.697929 | 0.789719/0.922467 |
| C | Bp | 16 | True | 1 | 0.602567 | 0.60613 | 0.538729/1.16637 |
| C | Bp | 48 | True | 1 | 0.450178 | 0.462076 | 0.552011/0.674058 |
| C | Bm | 0 | True | 1 | 0.350393 | 0.350393 | 0.460099/0.498648 |
| C | Bm | 4 | False | 1 | 0.645954 | 0.65319 | 0.755092/0.873399 |
| C | Bm | 16 | False | 1 | 0.721131 | 0.736802 | 0.647718/1.46227 |
| C | Bm | 48 | True | 1 | 0.434413 | 0.446813 | 0.548859/0.617501 |

- per-cell resolved-anchor summary: {"R": {"n_resolved": 7, "arms": ["A", "Bm", "Bp"], "t": [0, 16, 48], "mean_E": 0.4394467455855756}, "C": {"n_resolved": 7, "arms": ["A", "Bm", "Bp"], "t": [0, 16, 48], "mean_E": 0.4248824023276066}}
- half-bank E_Mgeom<=1 over resolved anchors: False (all-anchor max 2.431; scope note: half-bank <=1 applied over signal-resolved anchors (the population the frozen rule defines E_M over).  Under the alternative all-anchor reading the max half-bank E is 2.431, which sits at unresolved anchors where ybar is below the resolution gate (noise ratios).)
- sensitivity disclosure: the verdict failure comes from ONE resolved anchor — cell C, arm Bp, t=16 (marginally resolved, y_norm exceeds 3*SE by ~7e-3) — whose last-4 half-bank E_Mgeom = 1.166 violates the <=1 rule. All cleanly-resolved anchors (t=0, t=48, both cells; plus R/Bp/t=16) have E_Mgeom ~0.40-0.45 with half-bank E<=0.95. Had the marginal anchor fallen below the gate, the headline would read SUPPORTED; no re-selection was applied, and this knife-edge dependence is itself reported as a finding.
- verdict: **LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT**

## 6. Shadow update (channel B: y_U; stop-v ablation)

| cell | arm | t | resolved_U | E_Mgeom_U | E_stopv | halfbank f4/l4 |
|---|---|---|---|---|---|---|
| R | A | 0 | False | 0.775624 | 1.00155 | 0.801206/1.05006 |
| R | A | 4 | False | 2.25454 | 2.09744 | 1.79713/3.00816 |
| R | A | 16 | False | 1.04389 | 1.40716 | 0.736288/1.6513 |
| R | A | 48 | True | 0.41804 | 0.545674 | 0.512201/0.617918 |
| R | Bp | 0 | False | 0.778764 | 1.0039 | 0.802179/1.05025 |
| R | Bp | 4 | False | 2.28528 | 2.12416 | 1.8225/3.01643 |
| R | Bp | 16 | False | 0.670649 | 0.907432 | 0.601994/1.07107 |
| R | Bp | 48 | True | 0.453934 | 0.631115 | 0.56548/0.720199 |
| R | Bm | 0 | False | 0.774292 | 1.00035 | 0.800793/1.04927 |
| R | Bm | 4 | False | 2.11294 | 1.94717 | 1.73859/2.76015 |
| R | Bm | 16 | False | 1.91875 | 2.84069 | 0.862871/3.24932 |
| R | Bm | 48 | True | 0.409605 | 0.544304 | 0.506178/0.626478 |
| C | A | 0 | False | 0.807356 | 1.06874 | 0.830839/1.10518 |
| C | A | 4 | False | 1.08137 | 1.59475 | 1.00844/1.38867 |
| C | A | 16 | False | 1.03368 | 2.47928 | 0.770088/2.22119 |
| C | A | 48 | True | 0.389758 | 0.575785 | 0.504472/0.604037 |
| C | Bp | 0 | False | 0.811675 | 1.0733 | 0.832703/1.10775 |
| C | Bp | 4 | False | 1.07129 | 1.59391 | 1.00224/1.37181 |
| C | Bp | 16 | False | 0.858753 | 1.87682 | 0.730343/1.74017 |
| C | Bp | 48 | True | 0.434769 | 0.618873 | 0.534149/0.675927 |
| C | Bm | 0 | False | 0.806789 | 1.06814 | 0.83021/1.10719 |
| C | Bm | 4 | False | 1.07472 | 1.57523 | 1.00532/1.3901 |
| C | Bm | 16 | False | 1.04398 | 2.29243 | 0.874314/2.15123 |
| C | Bm | 48 | True | 0.395138 | 0.576197 | 0.50482/0.596056 |
- half-bank over resolved U-anchors and per-cell summary: {"per_cell": {"R": {"n_resolved": 3, "arms": ["A", "Bm", "Bp"], "t": [48], "mean_E": 0.42719335013561305}, "C": {"n_resolved": 3, "arms": ["A", "Bm", "Bp"], "t": [48], "mean_E": 0.40655493023917183}}, "halfbank_resolved_le_1": true, "halfbank_all_anchor_max": 3.2493192867196448, "halfbank_scope_note": "half-bank <=1 applied over signal-resolved anchors (the population the frozen rule defines E_M over).  Under the alternative all-anchor reading the max half-bank E is 3.249, which sits at unresolved anchors where ybar is below the resolution gate (noise ratios)."}
- verdict: **LOCAL_GEOMETRY_UNRESOLVED**

## 7. Tracking diagnostics (moments + transport)

| cell | arm | t | e_m | e_v | W_field | W_track | W_wd | W_total | transport err rel |
|---|---|---|---|---|---|---|---|---|---|
| R | A | 0 | 0.373555 | 0.00722049 | 0.0149266 | -0.0116507 | 3.39151e-07 | 0.00327623 | 1.39395e-05 |
| R | A | 4 | 0.497338 | 0.00651193 | 0.37997 | -0.343486 | -7.85024e-07 | 0.0364839 | 1.15885e-05 |
| R | A | 16 | 0.410243 | 0.00578512 | 0.00872627 | -0.00459855 | -4.57495e-07 | 0.00412727 | 1.5436e-05 |
| R | A | 48 | 0.404372 | 0.00454138 | 0.00760901 | -0.00377282 | -2.88271e-07 | 0.00383591 | 3.17705e-05 |
| R | Bp | 0 | 0.373555 | 0.00722049 | 0.0156919 | -0.0122481 | 3.55893e-07 | 0.00344421 | 1.7534e-05 |
| R | Bp | 4 | 0.516074 | 0.0065396 | 0.601945 | -0.543088 | -9.36153e-07 | 0.058856 | 1.32133e-05 |
| R | Bp | 16 | 0.426622 | 0.00568219 | 0.0140086 | -0.00797248 | -5.84136e-07 | 0.00603555 | 1.55822e-05 |
| R | Bp | 48 | 0.399347 | 0.00454967 | 0.00873753 | -0.00451293 | -2.95852e-07 | 0.00422431 | 3.59382e-05 |
| R | Bm | 0 | 0.373555 | 0.00722049 | 0.0141986 | -0.0110825 | 3.23167e-07 | 0.00311644 | 1.60713e-05 |
| R | Bm | 4 | 0.47811 | 0.00650113 | 0.2171 | -0.196565 | -6.4677e-07 | 0.0205346 | 1.37618e-05 |
| R | Bm | 16 | 0.394212 | 0.00586115 | 0.00718308 | -0.00543405 | -3.04002e-07 | 0.00174872 | 1.88333e-05 |
| R | Bm | 48 | 0.400152 | 0.00451275 | 0.00539079 | -0.00206833 | -2.5069e-07 | 0.00332221 | 4.0896e-05 |
| C | A | 0 | 0.380392 | 0.00727678 | 0.0146106 | -0.012842 | 3.4467e-07 | 0.00176898 | 1.37786e-05 |
| C | A | 4 | 0.388057 | 0.00714285 | 0.0151001 | -0.0116399 | 5.85043e-07 | 0.0034608 | 1.27566e-05 |
| C | A | 16 | 0.394676 | 0.00580772 | 0.0233376 | -0.0256742 | 1.69829e-07 | -0.00233637 | 1.33125e-05 |
| C | A | 48 | 0.381981 | 0.0043873 | 0.00739449 | -0.00464317 | -3.23427e-07 | 0.002751 | 3.13361e-05 |
| C | Bp | 0 | 0.380392 | 0.00727678 | 0.0153597 | -0.0135004 | 3.61993e-07 | 0.00185968 | 1.72309e-05 |
| C | Bp | 4 | 0.389327 | 0.00713799 | 0.0159377 | -0.0123569 | 6.14472e-07 | 0.00358141 | 1.4635e-05 |
| C | Bp | 16 | 0.394773 | 0.00588248 | 0.0257322 | -0.0268767 | 1.37679e-07 | -0.00114433 | 1.34828e-05 |
| C | Bp | 48 | 0.379834 | 0.00438524 | 0.00955999 | -0.00669976 | -3.01083e-07 | 0.00285992 | 3.64461e-05 |
| C | Bm | 0 | 0.380392 | 0.00727678 | 0.013898 | -0.0122157 | 3.28161e-07 | 0.0016827 | 1.583e-05 |
| C | Bm | 4 | 0.386545 | 0.00715074 | 0.0142617 | -0.0109158 | 5.53765e-07 | 0.00334648 | 1.48218e-05 |
| C | Bm | 16 | 0.404345 | 0.00572457 | 0.0251369 | -0.0282449 | 2.28587e-07 | -0.00310769 | 1.68191e-05 |
| C | Bm | 48 | 0.385557 | 0.00438423 | 0.00621423 | -0.003679 | -3.24121e-07 | 0.0025349 | 3.95916e-05 |
- W_field+W_track+W_wd == W_total identity residual max: 2.118e-07; verdict **PASS**

## 8. Readout balance (V_diag ledger)

| cell | arm | ΣK_V | ΣW_total | ΔV_diag | telescope residual |
|---|---|---|---|---|---|
| C | A | 0.12835 | 0.307244 | -0.178894 | 5.55112e-17 |
| C | Bm | 0.169424 | 0.342713 | -0.173288 | -1.11022e-16 |
| C | Bp | 0.074352 | 0.260042 | -0.18569 | -1.38778e-17 |
| R | A | 0.0835041 | 0.286891 | -0.203387 | 5.55112e-17 |
| R | Bm | 0.0826461 | 0.273552 | -0.190906 | -6.93889e-17 |
| R | Bp | 0.103333 | 0.314668 | -0.211335 | -2.77556e-17 |
- K_V is an exact algebraic remainder, NOT guaranteed positive; stats n=378, negative=90, mean 0.00169738
- V_eval was read under the frozen readout-h rule only and is reported separately (Section 3/4); it is NOT mixed into the V_diag ledger.
- verdict: **PASS**

## 9. Order discipline and bookkeeping checks

- predictions written before audit endpoint grads at every anchor: **True**
- E_M0 == 1 at every anchor (both channels): **True**
- theory T1-T9 CPU tests: ALL_OK (0 updates; fixes during test development disclosed in theory_tests.json and Section 10)

## 10. Disclosures

- 33 smoke-attempt optimizer updates lost to two runner bugs (9 + 24), counted and reconciled in `budget_log.jsonl`; user approved +9 over plan.
- Numerical floors: after the smoke crash a zero-update `--phase floor` process re-measured floors; it then emerged that the floor computation had pooled cross-probe pairs contrary to the frozen 'repeated IDENTICAL evaluations' definition, and a second zero-update `--phase floor2` process re-measured the floor correctly (all quantities bitwise-repeatable, floor = 0).  Supersession disclosed in Section 2 and smoke_checks_floor2.json.
- Terminology bans honored throughout: no banned thermodynamic metaphor appears in any artifact (temperature, free energy, entropy production, equilibrium, detailed balance, field theory, phase transition).

## 11. File manifest (sha256)

- `config.json`: `e00ce243cd05527b38c63e1a7c81beead1ee89396b3f025399ffbb41892bd1a5`
- `smoke_checks.json`: `184d01938118bf2a954288724805b73fcd939dac200e0f0d32ad94d7291e3ebe`
- `smoke_checks_floor2.json`: `cb64ea2f7d704aad939fd4615e177d9361c7e1e7412c056a9ba8c277c4c72762`
- `theory_tests.json`: `e727057d8030cdca89d521f33008926233b397265b98afbfbfe88e8bb136953e`
- `budget_log.jsonl`: `50d8b44d3619b1f0bba088f0250047d55bc6b537af90b6f951646654b09f22e3`
- `cell_metrics.csv`: `83edfaffeac170d3589728a955c842e7e3b8e7486e3816930dfb5a0f6c676c4b`
- `readout_balance.csv`: `bb775a163f30325c74cc91ab8479551f239c3a6f9d1bf505872ff42d256f13e1`
- `local_geometry_predictions.csv`: `cc1d4b93c9e1592d0fe4ab1c7bd77752bd9c08c35b55ef1ac294e4a9b1f10fe6`
- `shadow_update_predictions.csv`: `44a4530531d7de49df10536b02bd6013ba759f640769a85cd830332a534445de`
- `tracking_diagnostics.csv`: `11e12de8e9b092b28271ddde11e17e7b39cff56a5a99cac428b008a98ed8b18d`
- `summary.json`: `976820dce53e3103b3d946e142f0b6e70e82f8f3b8bba0855db7c9b073b5a04b`
- `runs/step_records.jsonl`: `6d06235ff9f98e47229556bdd4f87a1adc7bcbce43b9d4bf73be87d752524897`
- `runs/anchor_records.jsonl`: `a99f3513697255dae8c4c437bb86c61c5e3094f5e5f0250ce4448766bafbdb30`
- `runs/anchor_predictions.jsonl`: `2852d0a7a28db055725fdc7bc23be7dfa99e3e215014f38c588679dd1503b053`
- `runs/smoke_steps.jsonl`: `60db54853a80dba4fd106f9a48bc6caa26b1067b0b7be2a702754e96bb21207e`
- `runs/probe_banks.json`: `d038ae8a42bca45031ebefe45e6cb3f2fc5ca6ad9b541495afe98c546ec51444`
- `figures/fig1_local_geometry_E.png`: `71a09d335ae26e019f1bc7d10648782be63f4ef98e53b147bbb1c4ccdf6a117f`
- `figures/fig2_v_diag_trajectories.png`: `8f7bd76d2503819b8cd52df53b3afb5fbb67e3a2fadd7728ef6719bbca1ad7ea`
- `figures/fig3_readout_KV.png`: `618a3f56b2e22aa6b4c5099302867699214f7f3d30c20e11ace8365a0661e4ce`
- `figures/fig4_tracking_decomposition.png`: `e928665c5f25471d766ea8dff42bdafec9c80ba09ad7b3822b303d7b4cdb4770`
- `figures/fig5_shadow_update_E.png`: `427536fa110fae429ab50442c8f50c771fd16d5b42665c69bcae55d8b31c3ba3`

## 12. STOP statement

C0 ends here per protocol: one index/status line will be appended to the project status file; no C1 or controller work is started.
