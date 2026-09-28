# KEY DATA FOR REVIEW — FR_C1_LATE_COMPLETION_V01（自动生成，禁止手抄数字）

**勘误（2026-09-21 字段核对）**：本文件及以下各表为原始口径；字段级更正见
`FIELD_CORRECTIONS.md`（E1–E9，标签/定义级，主判定数值不变）。

来源：本目录 CSV/JSON（late_probe_statistics / old_probe_statistics /
aggregate_gate_inputs / late_error_decomposition / late_fixed_projection /
late_update_feedback / old_anchor_recheck / old_aggregate_recheck /
old_data_missing / criterion_trace / resource_log / smoke_checks /
state_and_probe_provenance）。生成器：research/scripts/c1_late_report_v01.py。

## 0. 身份与状态恢复

- 状态恢复：full native replay from frozen checkpoint outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt; 49 real updates per cell (h=1..49; anchor at t_pre=48). No on-disk snapshot at A/t48 exists (C1 snapshots: t0/t16 only); snapshot-continuation shortcuts NOT used.
- replay 更新数：R 48 + C 48 = 96 到达 pre-update t=48；anchor 更新 h=49；cumulative step 94；t_after=95。
- 恒等性（标量级，非完整 theta/m/v/RNG 逐位）：与 C0 原生 A 臂 h=49 逐步记录对比，v_pre / q_pre_norm / s_norm / eta_t 四项 × 2 cell 全部 diff = 0.000e+00。
- 泄漏顺序：prediction order_index 全部先于 audit（late: 0<1, 2<3；old: 0..7 单调）；冻结预测 per_K mean 与实测 group_stats 完全一致（diff = 0.000e+00，12/12 组）。

provenance:
| cell | checkpoint | stream_seed | bank_global_step | replay_updates_to_reach_t48 | anchor_update_h | cumulative_step_at_anchor | t_after_post_update | arm | u | anchors | state_recovery | probe_bank_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | /home/b205/projects/fluctuation_response/outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt | 7110 | 46 | 48 | 49 | 94 | 95 | A | 0.0 | A@t48 | full native replay from frozen checkpoint outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt; 49 real updates per cell (h=1..49; anchor at t_pre=48). No on-disk snapshot at A/t48 exists (C1 snapshots: t0/t16 only); snapshot-continuation shortcuts NOT used. | C1 runs/probe_banks.json (same seeds/order) |
| C | /home/b205/projects/fluctuation_response/outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt | 7111 | 46 | 48 | 49 | 94 | 95 | A | 0.0 | A@t48 | full native replay from frozen checkpoint outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt; 49 real updates per cell (h=1..49; anchor at t_pre=48). No on-disk snapshot at A/t48 exists (C1 snapshots: t0/t16 only); snapshot-continuation shortcuts NOT used. | C1 runs/probe_banks.json (same seeds/order) |

smoke:
| check | value |
|---|---|
| definition | same-probe pairing only (identical evaluations); cross-probe diversity is NOT a floor |
| diag_state_identity | True |
| diag_state_identity_detail | {'sig_off': '9e2c4109fecb5e9dfcc9568bf9ec9415c470bf713fa790b5889a6c3288dc7952', 'sig_on': '9e2c4109fecb5e9dfcc9568bf9ec9415c470bf713fa790b5889a6c3288dc7952', 'shared_fields_checked': ['lr_native', 'lr_applied', 'full_state_sha256'], 'note': 'no external C0/B0-style full-state table exists for C1; only these common fields + full-state hash checked; not extrapolated beyond', 'updates': 8} |
| numerical_floor_same_probe | {'grad': 0.0, 'hvp': 0.0, 'v': 0.0, 'q': 0.0} |
| hvp_proxy_native_rel_gap_max | 0.004472299818394586 |
| hvp_precision_gate | PASS |
| bank_disjointness | {'dev_audit_overlap': [], 'c0_overlap': [], 'rollout_overlap': [], 'dev_probes': 32, 'audit_probes': 32, 'status': 'PASS'} |
| per_step_wall_s_smoke_on | [0.19889420096296817, 0.16407820000313222, 0.18651439994573593, 0.18320890003815293] |
| counters | {'optimizer_updates': 8, 'grad_evals': 19, 'hvp_calls': 16, 'micro_forward_calls': 63, 'shadow_steps': 0, 'shadow_seconds': 0.0} |
| shadow_seconds | 0.0 |

## 1. Table A — late anchors (K32 主读数)

| anchor | ch | target_norm | SE | floor | SNR | resolved | pred_norm | pred_SE | resid_SE | SSE | TSS | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R/A/t48 | field | 0.0488516 | 0.00793603 | 0.0 | 2.05 | True | 0.0463968 | 0.00781487 | 0.000192965 | 0.000133393 | 0.00238648 | 0.2364 |
| R/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000455527 | 0.00282492 | 0.4016 |
| R/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000254475 | 0.00249318 | 0.3195 |
| R/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000133393 | 0.00238648 | 0.2364 |
| R/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000254475 | 0.00249318 | 0.3195 |
| R/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000239486 | 0.00238137 | 0.3171 |
| R/A/t48 | shadow | 0.00189565 | 0.000307667 | 0.0 | 2.05 | True | 0.00176898 | 0.000304048 | 6.63507e-06 | 2.20742e-07 | 3.59348e-06 | 0.2478 |
| R/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 6.57527e-07 | 4.38486e-06 | 0.3872 |
| R/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 3.72198e-07 | 3.77834e-06 | 0.3139 |
| R/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 2.20742e-07 | 3.59348e-06 | 0.2478 |
| R/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 3.72198e-07 | 3.77834e-06 | 0.3139 |
| R/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 4.02053e-07 | 3.55213e-06 | 0.3364 |
| C/A/t48 | field | 0.0619313 | 0.0106391 | 0.0 | 1.94 | True | 0.0586672 | 0.0105647 | 0.000279215 | 0.000217536 | 0.00383548 | 0.2382 |
| C/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.00087261 | 0.0043208 | 0.4494 |
| C/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000405406 | 0.00377538 | 0.3277 |
| C/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000217536 | 0.00383548 | 0.2382 |
| C/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000405406 | 0.00377538 | 0.3277 |
| C/A/t48 | field | NA | NA | NA | NA |  | NA | NA | NA | 0.000423987 | 0.004087 | 0.3221 |
| C/A/t48 | shadow | 0.00215252 | 0.000345951 | 0.0 | 2.07 | True | 0.00201224 | 0.000341064 | 6.34551e-06 | 2.29177e-07 | 4.63335e-06 | 0.2224 |
| C/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 8.52579e-07 | 5.2506e-06 | 0.4030 |
| C/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 4.31453e-07 | 4.4979e-06 | 0.3097 |
| C/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 2.29177e-07 | 4.63335e-06 | 0.2224 |
| C/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 4.31453e-07 | 4.4979e-06 | 0.3097 |
| C/A/t48 | shadow | NA | NA | NA | NA |  | NA | NA | NA | 4.44536e-07 | 4.96867e-06 | 0.2991 |

## 1b. Table A — late anchors 全部 K/half 明细

| anchor | ch | K/half | SSE | TSS | E |
|---|---|---|---|---|---|
| R/A/t48 | field | 32 | 0.000133393 | 0.00238648 | 0.2364 |
| R/A/t48 | field | K8 | 0.000455527 | 0.00282492 | 0.4016 |
| R/A/t48 | field | K16 | 0.000254475 | 0.00249318 | 0.3195 |
| R/A/t48 | field | K32 | 0.000133393 | 0.00238648 | 0.2364 |
| R/A/t48 | field | first16 | 0.000254475 | 0.00249318 | 0.3195 |
| R/A/t48 | field | last16 | 0.000239486 | 0.00238137 | 0.3171 |
| R/A/t48 | shadow | 32 | 2.20742e-07 | 3.59348e-06 | 0.2478 |
| R/A/t48 | shadow | K8 | 6.57527e-07 | 4.38486e-06 | 0.3872 |
| R/A/t48 | shadow | K16 | 3.72198e-07 | 3.77834e-06 | 0.3139 |
| R/A/t48 | shadow | K32 | 2.20742e-07 | 3.59348e-06 | 0.2478 |
| R/A/t48 | shadow | first16 | 3.72198e-07 | 3.77834e-06 | 0.3139 |
| R/A/t48 | shadow | last16 | 4.02053e-07 | 3.55213e-06 | 0.3364 |
| C/A/t48 | field | 32 | 0.000217536 | 0.00383548 | 0.2382 |
| C/A/t48 | field | K8 | 0.00087261 | 0.0043208 | 0.4494 |
| C/A/t48 | field | K16 | 0.000405406 | 0.00377538 | 0.3277 |
| C/A/t48 | field | K32 | 0.000217536 | 0.00383548 | 0.2382 |
| C/A/t48 | field | first16 | 0.000405406 | 0.00377538 | 0.3277 |
| C/A/t48 | field | last16 | 0.000423987 | 0.004087 | 0.3221 |
| C/A/t48 | shadow | 32 | 2.29177e-07 | 4.63335e-06 | 0.2224 |
| C/A/t48 | shadow | K8 | 8.52579e-07 | 5.2506e-06 | 0.4030 |
| C/A/t48 | shadow | K16 | 4.31453e-07 | 4.4979e-06 | 0.3097 |
| C/A/t48 | shadow | K32 | 2.29177e-07 | 4.63335e-06 | 0.2224 |
| C/A/t48 | shadow | first16 | 4.31453e-07 | 4.4979e-06 | 0.3097 |
| C/A/t48 | shadow | last16 | 4.44536e-07 | 4.96867e-06 | 0.2991 |

## 1c. Table A — old anchors 复核 (K32, O1 独立重算)

| anchor | ch | target_norm | SE | floor | SNR | resolved | pred_norm | pred_SE | resid_SE | SSE | TSS | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C/A/t0 | field | 0.0523756 | 0.00760779 | 0.0 | 2.29 | True | 0.0523984 | 0.00757673 | 0.000131281 | 0.000131663 | 0.00274321 | 0.2191 |
| C/A/t0 | shadow | 0.00385756 | 0.00134705 | 0.0 | 0.95 | False | 0.00395294 | 0.00129594 | NA | 4.16308e-06 | 1.48808e-05 | 0.5289 |
| C/Bp/t16 | field | 0.0869308 | 0.0158159 | 0.0 | 1.83 | True | 0.105391 | 0.0353044 | 0.00657527 | 0.00150117 | 0.00755697 | 0.4457 |
| C/Bp/t16 | shadow | 0.00794606 | 0.00220211 | 0.0 | 1.20 | True | 0.0102427 | 0.00435266 | NA | 2.25204e-05 | 6.31399e-05 | 0.5972 |
| R/A/t0 | field | 0.0519429 | 0.00881886 | 0.0 | 1.96 | True | 0.0524072 | 0.00869957 | 0.000220577 | 0.000211307 | 0.00269807 | 0.2799 |
| R/A/t0 | shadow | 0.00451824 | 0.0016488 | 0.0 | 0.91 | False | 0.00471372 | 0.00153505 | NA | 7.31393e-06 | 2.04145e-05 | 0.5986 |
| R/Bp/t16 | field | 0.173737 | 0.0333704 | 0.0 | 1.74 | True | 0.23679 | 0.0506147 | 0.0116855 | 0.00817128 | 0.0301847 | 0.5203 |
| R/Bp/t16 | shadow | 0.0188612 | 0.00405904 | 0.0 | 1.55 | True | 0.0289046 | 0.00626717 | NA | 0.00016021 | 0.000355745 | 0.6711 |

## 1d. O1 聚合交叉核对（独立重算 vs 已发布 aggregate_E.csv）

| cell | ch | group | SSE_sum | TSS_sum | E_recomp | E_published | match |
|---|---|---|---|---|---|---|---|
| R | field | full32 | 0.00838258 | 0.0328827 | 0.5049 | NA | True |
| R | field | half:first16 | 0.0173015 | 0.0353567 | 0.6995 | NA | True |
| R | field | half:last16 | 0.00793434 | 0.0321469 | 0.4968 | NA | True |
| R | shadow | full32 | 0.000167524 | 0.000376159 | 0.6673 | NA | True |
| R | shadow | half:first16 | 0.000308803 | 0.000405653 | 0.8725 | NA | True |
| R | shadow | half:last16 | 0.000148317 | 0.000371211 | 0.6321 | NA | True |
| C | field | full32 | 0.00163283 | 0.0103002 | 0.3982 | NA | True |
| C | field | half:first16 | 0.00154878 | 0.0104154 | 0.3856 | NA | True |
| C | field | half:last16 | 0.00513976 | 0.0106824 | 0.6936 | NA | True |
| C | shadow | full32 | 2.66834e-05 | 7.80207e-05 | 0.5848 | NA | True |
| C | shadow | half:first16 | 3.13184e-05 | 8.60028e-05 | 0.6035 | NA | True |
| C | shadow | half:last16 | 7.68406e-05 | 8.00329e-05 | 0.9799 | NA | True |

全部 match = True

## 2. Table B — 聚合门输入与逐 cell 布尔

| scope | cell | ch | group | anchors | SSE_sum | TSS_sum | E | thr | criterion |
|---|---|---|---|---|---|---|---|---|---|
| OLD4 | R | field | full32 | A/t0;Bp/t16 | 0.00838258 | 0.0328827 | 0.5049 | 0.8 | TRUE |
| OLD4 | R | field | first16 | A/t0;Bp/t16 | 0.0173015 | 0.0353567 | 0.6995 | 1.0 | TRUE |
| OLD4 | R | field | last16 | A/t0;Bp/t16 | 0.00793434 | 0.0321469 | 0.4968 | 1.0 | TRUE |
| LATE2 | R | field | full32 | A/t48 | 0.000133393 | 0.00238648 | 0.2364 | 0.8 | TRUE |
| LATE2 | R | field | first16 | A/t48 | 0.000254475 | 0.00249318 | 0.3195 | 1.0 | TRUE |
| LATE2 | R | field | last16 | A/t48 | 0.000239486 | 0.00238137 | 0.3171 | 1.0 | TRUE |
| COMBINED6 | R | field | full32 | A/t0;Bp/t16;A/t48 | 0.00851598 | 0.0352692 | 0.4914 | 0.8 | TRUE |
| COMBINED6 | R | field | first16 | A/t0;Bp/t16;A/t48 | 0.017556 | 0.0378499 | 0.6811 | 1.0 | TRUE |
| COMBINED6 | R | field | last16 | A/t0;Bp/t16;A/t48 | 0.00817383 | 0.0345282 | 0.4865 | 1.0 | TRUE |
| OLD4 | R | shadow | full32 | A/t0;Bp/t16 | 0.000167524 | 0.000376159 | 0.6673 | 0.8 | TRUE |
| OLD4 | R | shadow | first16 | A/t0;Bp/t16 | 0.000308803 | 0.000405653 | 0.8725 | 1.0 | TRUE |
| OLD4 | R | shadow | last16 | A/t0;Bp/t16 | 0.000148317 | 0.000371211 | 0.6321 | 1.0 | TRUE |
| LATE2 | R | shadow | full32 | A/t48 | 2.20742e-07 | 3.59348e-06 | 0.2478 | 0.8 | TRUE |
| LATE2 | R | shadow | first16 | A/t48 | 3.72198e-07 | 3.77834e-06 | 0.3139 | 1.0 | TRUE |
| LATE2 | R | shadow | last16 | A/t48 | 4.02053e-07 | 3.55213e-06 | 0.3364 | 1.0 | TRUE |
| COMBINED6 | R | shadow | full32 | A/t0;Bp/t16;A/t48 | 0.000167744 | 0.000379753 | 0.6646 | 0.8 | TRUE |
| COMBINED6 | R | shadow | first16 | A/t0;Bp/t16;A/t48 | 0.000309175 | 0.000409431 | 0.8690 | 1.0 | TRUE |
| COMBINED6 | R | shadow | last16 | A/t0;Bp/t16;A/t48 | 0.000148719 | 0.000374763 | 0.6299 | 1.0 | TRUE |
| OLD4 | C | field | full32 | A/t0;Bp/t16 | 0.00163283 | 0.0103002 | 0.3982 | 0.8 | TRUE |
| OLD4 | C | field | first16 | A/t0;Bp/t16 | 0.00154878 | 0.0104154 | 0.3856 | 1.0 | TRUE |
| OLD4 | C | field | last16 | A/t0;Bp/t16 | 0.00513976 | 0.0106824 | 0.6936 | 1.0 | TRUE |
| LATE2 | C | field | full32 | A/t48 | 0.000217536 | 0.00383548 | 0.2382 | 0.8 | TRUE |
| LATE2 | C | field | first16 | A/t48 | 0.000405406 | 0.00377538 | 0.3277 | 1.0 | TRUE |
| LATE2 | C | field | last16 | A/t48 | 0.000423987 | 0.004087 | 0.3221 | 1.0 | TRUE |
| COMBINED6 | C | field | full32 | A/t0;Bp/t16;A/t48 | 0.00185037 | 0.0141357 | 0.3618 | 0.8 | TRUE |
| COMBINED6 | C | field | first16 | A/t0;Bp/t16;A/t48 | 0.00195418 | 0.0141907 | 0.3711 | 1.0 | TRUE |
| COMBINED6 | C | field | last16 | A/t0;Bp/t16;A/t48 | 0.00556375 | 0.0147694 | 0.6138 | 1.0 | TRUE |
| OLD4 | C | shadow | full32 | A/t0;Bp/t16 | 2.66834e-05 | 7.80207e-05 | 0.5848 | 0.8 | TRUE |
| OLD4 | C | shadow | first16 | A/t0;Bp/t16 | 3.13184e-05 | 8.60028e-05 | 0.6035 | 1.0 | TRUE |
| OLD4 | C | shadow | last16 | A/t0;Bp/t16 | 7.68406e-05 | 8.00329e-05 | 0.9799 | 1.0 | TRUE |
| LATE2 | C | shadow | full32 | A/t48 | 2.29177e-07 | 4.63335e-06 | 0.2224 | 0.8 | TRUE |
| LATE2 | C | shadow | first16 | A/t48 | 4.31453e-07 | 4.4979e-06 | 0.3097 | 1.0 | TRUE |
| LATE2 | C | shadow | last16 | A/t48 | 4.44536e-07 | 4.96867e-06 | 0.2991 | 1.0 | TRUE |
| COMBINED6 | C | shadow | full32 | A/t0;Bp/t16;A/t48 | 2.69126e-05 | 8.2654e-05 | 0.5706 | 0.8 | TRUE |
| COMBINED6 | C | shadow | first16 | A/t0;Bp/t16;A/t48 | 3.17499e-05 | 9.05007e-05 | 0.5923 | 1.0 | TRUE |
| COMBINED6 | C | shadow | last16 | A/t0;Bp/t16;A/t48 | 7.72851e-05 | 8.50016e-05 | 0.9535 | 1.0 | TRUE |

## 2b. 门追踪（含 distinct-t 判据）

| scope | cell | ch | status | E_K32 | E_first16 | E_last16 | n_t | distinct_t |
|---|---|---|---|---|---|---|---|---|
| COMBINED6 | C | field | PASS | 0.3618 | 0.3711 | 0.6138 | 3 | [0, 16, 48] |
| COMBINED6 | C | shadow | PASS | 0.5706 | 0.5923 | 0.9535 | 2 | [16, 48] |
| COMBINED6 | R | field | PASS | 0.4914 | 0.6811 | 0.4865 | 3 | [0, 16, 48] |
| COMBINED6 | R | shadow | PASS | 0.6646 | 0.8690 | 0.6299 | 2 | [16, 48] |
| LATE2 | C | field | NUMERICAL_UNRESOLVED | 0.2382 | 0.3277 | 0.3221 | 1 | [48] |
| LATE2 | C | shadow | NUMERICAL_UNRESOLVED | 0.2224 | 0.3097 | 0.2991 | 1 | [48] |
| LATE2 | R | field | NUMERICAL_UNRESOLVED | 0.2364 | 0.3195 | 0.3171 | 1 | [48] |
| LATE2 | R | shadow | NUMERICAL_UNRESOLVED | 0.2478 | 0.3139 | 0.3364 | 1 | [48] |
| OLD4 | C | field | PASS | 0.3982 | 0.3856 | 0.6936 | 2 | [0, 16] |
| OLD4 | C | shadow | NUMERICAL_UNRESOLVED | 0.5848 | 0.6035 | 0.9799 | 1 | [16] |
| OLD4 | R | field | PASS | 0.5049 | 0.6995 | 0.4968 | 2 | [0, 16] |
| OLD4 | R | shadow | NUMERICAL_UNRESOLVED | 0.6673 | 0.8725 | 0.6321 | 1 | [16] |

late 单 anchor 判定：
| cell | arm | anchor | channel | E_K32 | E_first16 | E_last16 | resolved | signal_ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|
| R | A | 48 | field | 0.23642144364878584 | 0.3194810454409857 | 0.31712196335184517 | True | 2.051892170207198 | LATE_ANCHOR_RESOLVED_AND_IMPROVED |
| R | A | 48 | shadow | 0.24784765566811973 | 0.31386038695466134 | 0.3364319791354514 | True | 2.0537864048091823 | LATE_ANCHOR_RESOLVED_AND_IMPROVED |
| C | A | 48 | field | 0.23815260189081944 | 0.3276909269231278 | 0.32208752393439993 | True | 1.9403643210162216 | LATE_ANCHOR_RESOLVED_AND_IMPROVED |
| C | A | 48 | shadow | 0.22240190313405 | 0.3097148033942767 | 0.2991117819168617 | True | 2.074014145558594 | LATE_ANCHOR_RESOLVED_AND_IMPROVED |

## 3. Table C — late error decomposition + shadow stop-v

| anchor | ch | total_SS | bank_SS | local_SS | signed_cross | ident_resid | Z_full_norm | Z_stop_norm | ratio | cos | paired_diff | paired_diff_SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R/t48 | field | 0.000133393 | 0.000111691 | 1.51349e-05 | 6.56666e-06 | 2.71051e-20 | 0.00253191 | 0.00311229 | 1.2222 | 0.8746 | 0.00150479 | 0.000128743 |
| R/t48 | shadow | 2.20742e-07 | 1.68365e-07 | 3.57625e-08 | 1.6615e-08 | 5.29396e-23 | 0.00253191 | 0.00311229 | 1.2222 | 0.8746 | 0.00150479 | 0.000128743 |
| C/t48 | field | 0.000217536 | 0.000211634 | 1.15965e-05 | -5.69443e-06 | 5.42101e-20 | 0.00286803 | 0.00351366 | 1.2363 | 0.8669 | 0.00166819 | 0.000115177 |
| C/t48 | shadow | 2.29177e-07 | 1.94801e-07 | 2.01462e-08 | 1.42298e-08 | -2.64698e-23 | 0.00286803 | 0.00351366 | 1.2363 | 0.8669 | 0.00166819 | 0.000115177 |

## 4. Table D — fixed q+ readout（原始 <x,q+> 投影；SHADOW_FIRST_ORDER_READOUT = -q+ᵀx̄，非 loss）

| cell | ch | ||q+|| | pred_mean | pred_SE | tgt_mean | tgt_SE | resid_mean | resid_SE | full-stop | paired f-s | paired f-s SE | d_c_to_u(q+) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | field | 0.8927 | -0.00862847 | 0.000314675 | -0.00911348 | 0.000343273 | 0.000628381 | 3.31647e-05 | NA | NA | NA | -0.0637838 |
| R | shadow | 0.8927 | 0.000223725 | 8.90782e-06 | 0.00023963 | 8.79221e-06 | -1.86245e-05 | 8.12046e-07 | 0.000247643 | -8.01216e-06 | 9.2666e-07 | -0.0637838 |
| C | field | 0.8875 | -0.000617234 | 0.000255923 | -0.000636341 | 0.000259279 | 6.79418e-05 | 4.2229e-05 | NA | NA | NA | -0.0887293 |
| C | shadow | 0.8875 | -5.97477e-06 | 5.09422e-06 | -2.80704e-06 | 4.7011e-06 | -2.12493e-06 | 7.13642e-07 | -3.44741e-06 | 6.40376e-07 | 5.80632e-07 | -0.0887293 |

## 5. clip activity（dev 全体 active；audit 有 crossings）

```
{
 "R/t48": {
  "dev": {
   "active_g0": 18,
   "active_gpred": 18,
   "active_g1": 18,
   "cross_g0_gpred": 0,
   "cross_g0_g1": 0,
   "cross_gpred_g1": 0,
   "n": 32
  },
  "audit": {
   "active_g0": 13,
   "active_gpred": 16,
   "active_g1": 15,
   "cross_g0_gpred": 3,
   "cross_g0_g1": 2,
   "cross_gpred_g1": 1,
   "n": 32
  }
 },
 "C/t48": {
  "dev": {
   "active_g0": 18,
   "active_gpred": 18,
   "active_g1": 18,
   "cross_g0_gpred": 0,
   "cross_g0_g1": 0,
   "cross_gpred_g1": 0,
   "n": 32
  },
  "audit": {
   "active_g0": 15,
   "active_gpred": 15,
   "active_g1": 15,
   "cross_g0_gpred": 0,
   "cross_g0_g1": 0,
   "cross_gpred_g1": 0,
   "n": 32
  }
 }
}
```

## 6. O2 轴/目标/索引清单 与 O3 缺失清单

轴定义（不混轴、不减常数对齐）：V_diag = v_score 前 8 个 ID（1 chunk，eval 模式，fp32 CE）；
V_eval = 冻结 64-sample；A0T = v_score 全轴。A0T−v_eval AXIS_MISMATCH 维持。
C0 readout_balance 索引：63 行/cell/arm，h∈2..64；区间 [0,4)=[upd 1..4]，[4,16)=[5..16]，[16,63]=[17..63]；无 h=65 行，不插补。

| item | status | note |
|---|---|---|
| C0 half-bank aggregate gate (g3) | DATA_MISSING | C0 never saved half-bank SSE/TSS; aggregate per-point recheck impossible. C0X verdict unchanged (erratum, original preserved). New late data does NOT repair this historical gap. |
| OLD C1 shadow per-probe same-probe residual SE | NOT_SAVED | Old runner saved field residual norms only. probe_summary.csv K32 shadow rows carry the FIELD residual value (analyzer fill artifact) - excluded from any shadow inference. |
| OLD C1 stop-v per-probe norms/ratio/cosine | PARTIAL | Only mean-level inner and diff-norm saved for old anchors; per-probe norm ratio and cosine not recoverable offline. |
| A0T - C0 v_eval cross-axis absolute comparison | AXIS_MISMATCH | v_score full axis vs independent 64-sample v_eval; no shared axis; constant-offset alignment forbidden. Unchanged. |
| C0 readout ledger h=65 row | NOT_PRESENT (index definition, not imputed) | 63 ledger rows per cell/arm, h in 2..64 (updates 1..63); intervals [0,4)=[1..4], [4,16)=[5..16], [16,63]=[17..63]; no interpolation/zero-fill; no new GPU spent on history. |
| OLD C1 per-group sumsq/centered_SS | NOT_SAVED | Old records saved mean_sq_norm and SE only; per-probe dispersion (centered_SS) not recomputable beyond the saved SE. NEW late records save full sufficient statistics. |

## 7. Table E — 成本与覆盖

| 项目 | 说明 | 数值 |
|---|---|---|
| 本轮计划 late anchors | 2 (R/A/t48, C/A/t48) | 2/2 执行 |
| 状态恢复 | 无 t48 snapshot（C1 仅存 t0/t16）→ 全重放 | 49 updates/cell × 2 = 98（与 provenance 一致） |
| smoke 过程 | updates/grads/HVP/micros | 8/19/16/63 |
| 主过程 | updates/grads/HVP/micros | 98/260/128/772 |
| 本轮合计（含 smoke） | updates/grads/HVP/micros | 106/279/144/835 |
| 本轮 caps | updates/grads/HVP/micros/wall_s | 128/600/192/1800/2700 |
| within_caps |  | True |
| 主过程 GPU wall | s | 231.0 |
| shadow 算术（另计） | steps / s | 768 / 8.91 |
| 训练用 micro-forwards（另披露） | replay 梯度伴生 | 196 (=98×2) |
| 旧账 1：C0/C0X 轮 | updates（不重置、不并入） | 17,994 / 18,000 |
| 旧账 2：C1 轮 | updates/grads/HVP/micros | 45/605/304/1801 |
| 本轮新账 | updates/grads/HVP/micros | 106/279/144/835 |
