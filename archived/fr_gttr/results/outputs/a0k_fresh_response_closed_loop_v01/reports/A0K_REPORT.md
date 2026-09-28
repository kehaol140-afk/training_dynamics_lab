# A0K Fresh Response Closed-loop Validation — REPORT

dashboard table (all seeds, both cadences; gains = loss reduction, positive = controller better):

| seed | K | L_A@128 | L_B@128 | L_C@128 | M1 A−C | M5 C−B | T_C(s) | T_extra(s) | steps A_wc | L_A_wc@T_C | M2 Gain_time | ROI/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2030 | 10 | 6.94879 | 6.94748 | 6.94802 | 0.000769913 | -0.000532508 | 49.33 | 39.81 | 128 | 6.94879 | 0.000769913 | 1.934e-05 |
| 2030 | 20 | 6.94879 | 6.94816 | 6.9478 | 0.000984848 | 0.000356138 | 29.13 | 17.35 | 128 | 6.94879 | 0.000984848 | 5.676e-05 |
| 2031 | 10 | 6.93 | 6.92804 | 6.92836 | 0.00164223 | -0.000320077 | 70.92 | 55.52 | 128 | 6.93 | 0.00164223 | 2.958e-05 |
| 2031 | 20 | 6.93 | 6.92915 | 6.92914 | 0.000857234 | 8.88109e-06 | 28.97 | 16.98 | 128 | 6.93 | 0.000857234 | 5.049e-05 |
| 2032 | 10 | 6.99005 | 6.98885 | 6.98824 | 0.00181478 | 0.000614941 | 51.33 | 39.42 | 128 | 6.99005 | 0.00181478 | 4.604e-05 |
| 2032 | 20 | 6.99005 | 6.98951 | 6.99045 | -0.000398874 | -0.000940382 | 32.35 | 19.62 | 128 | 6.99005 | -0.000398874 | -2.033e-05 |
| 2033 | 20 | 6.90923 | 6.90859 | 6.9086 | 0.000630915 | -1.19209e-05 | 34.74 | 22.8 | 128 | 6.90923 | 0.000630915 | 2.767e-05 |
| 2034 | 20 | 6.94307 | 6.94285 | 6.94382 | -0.00075376 | -0.000972271 | 30.34 | 18.55 | 128 | 6.94307 | -0.00075376 | -4.062e-05 |
| 2035 | 20 | 6.93033 | 6.92887 | 6.92911 | 0.00121802 | -0.000246823 | 41.44 | 25.68 | 128 | 6.93033 | 0.00121802 | 4.743e-05 |

## 1. Protocol

Frozen by contract.json (sha in hashes/method_fingerprints.json) before any GPU: fresh response controller vs pure AdamW (A) vs fixed-j1 (B) on the frozen v0.7 recipe; H=4, λ=0.05, 5 candidates (j=0 no-op native AdamW; j>0 inject g+λ(g_j−g)), selector argmin_j(λ·ĥ_j) tie→smallest id, production run_adjoint_only scoring, no cache/no stale gradients/no JVP. Control events at steps <128 multiples of K before step t; T_total=128. K=20 primary, K=10 secondary, seeds 2030–2032 (expansion 2033–2035 by budget).

## 2. Setup / identity gates

- S1 arm-A bitwise == v0.7 bank at steps 46/52: **True** (46: 2f40340e9810…, 52: ada46c189449…)
- S2 ghost no-op (full C machinery, forced j=0) trajectory-invisible: **True**, tail_match=True, 6 events, mean event cost 2.85s
- All arms per seed share identical init (setup_seed before init_model), data order, dropout RNG; control events RNG-neutral (capture/restore around all controller computation).

## 3. Q1 — is the fresh response effective? (equal-step, M1)

Median equal-step gain A−C @128 (K=20): **0.000744075** nats; per seed: {"2030/K10": 0.00077, "2030/K20": 0.000985, "2031/K10": 0.001642, "2031/K20": 0.000857, "2032/K10": 0.001815, "2032/K20": -0.000399, "2033/K20": 0.000631, "2034/K20": -0.000754, "2035/K20": 0.001218}. K=10 median: 0.00164223.

## 4. Q2 — adaptive vs fixed-j1? (M5)

Equal-step C−B median: **-0.000129372**; wall-clock C−B median: **-0.000129372**. Per seed (equal-step): {"2030/K10": -0.000533, "2030/K20": 0.000356, "2031/K10": -0.00032, "2031/K20": 9e-06, "2032/K10": 0.000615, "2032/K20": -0.00094, "2033/K20": -1.2e-05, "2034/K20": -0.000972, "2035/K20": -0.000247}. C≤B in 6/9 pairs — the selector does not beat the frozen fixed action this round; the benefit comes from the intervention itself, not the selection.

## 5. Q3 — economics? (equal-wall-clock, M2 — PRIMARY)

T_extra = T_C − T_A128 ≈ 19.09s per run (rollout+response+snapshot+restore+candidate+execution all inside T_C). **Scale note:** in 6/6 primary pairs the baseline completed the full 128-step planned trajectory inside T_C, so Gain_time == equal-step gain and the baseline was never step-starved; at this scale the controller spends 17–56s extra wall time per run for ≤2e-3 nats. Median Gain_time A_wc−C: **0.000744075**; per seed: {"2030/K10": 0.00077, "2030/K20": 0.000985, "2031/K10": 0.001642, "2031/K20": 0.000857, "2032/K10": 0.001815, "2032/K20": -0.000399, "2033/K20": 0.000631, "2034/K20": -0.000754, "2035/K20": 0.001218}.

## 6. Q4 — where does the gain come from? (M4 + distribution)

- K=10 (36 events): no-op rate 0.00, positive rate 0.78, harmful rate 0.22, mean short-horizon gain 0.000510931, mean fixed-j1 gain 0.000429943
  - seed 2030: actions {'0': 0, '1': 4, '2': 1, '3': 5, '4': 2}, mean gain v_eval 0.000174706, v_score 0.000287061
  - seed 2031: actions {'0': 0, '1': 4, '2': 4, '3': 1, '4': 3}, mean gain v_eval 0.000923003, v_score 0.000512213
  - seed 2032: actions {'0': 0, '1': 2, '2': 2, '3': 3, '4': 5}, mean gain v_eval 0.000435084, v_score 0.000479435
- K=20 (36 events): no-op rate 0.00, positive rate 0.81, harmful rate 0.19, mean short-horizon gain 0.000411857, mean fixed-j1 gain 0.000349277
  - seed 2030: actions {'0': 0, '1': 2, '2': 1, '3': 2, '4': 1}, mean gain v_eval 0.000225226, v_score 0.000234991
  - seed 2031: actions {'0': 0, '1': 1, '2': 1, '3': 2, '4': 2}, mean gain v_eval 0.00176182, v_score 0.000717511
  - seed 2032: actions {'0': 0, '1': 1, '2': 0, '3': 2, '4': 3}, mean gain v_eval 0.000130594, v_score 0.000435879
  - seed 2033: actions {'0': 0, '1': 2, '2': 0, '3': 2, '4': 2}, mean gain v_eval 3.15905e-05, v_score 0.000159025
  - seed 2034: actions {'0': 0, '1': 3, '2': 1, '3': 2, '4': 0}, mean gain v_eval 7.60357e-05, v_score 0.00029091
  - seed 2035: actions {'0': 0, '1': 3, '2': 1, '3': 0, '4': 2}, mean gain v_eval 0.000245869, v_score 0.000267893

## 7. ROI (M3)

Median ROI = Gain_time/T_extra: **3.755e-05** nats/s; per seed: {"2030/K10": 1.9e-05, "2030/K20": 5.7e-05, "2031/K10": 3e-05, "2031/K20": 5e-05, "2032/K10": 4.6e-05, "2032/K20": -2e-05, "2033/K20": 2.8e-05, "2034/K20": -4.1e-05, "2035/K20": 4.7e-05}.

## 8. Per-seed variance

- seed 2030 K=10: M1 0.000769913, M2 0.000769913, chosen events 12
- seed 2030 K=20: M1 0.000984848, M2 0.000984848, chosen events 6
- seed 2031 K=10: M1 0.00164223, M2 0.00164223, chosen events 12
- seed 2031 K=20: M1 0.000857234, M2 0.000857234, chosen events 6
- seed 2032 K=10: M1 0.00181478, M2 0.00181478, chosen events 12
- seed 2032 K=20: M1 -0.000398874, M2 -0.000398874, chosen events 6
- seed 2033 K=20: M1 0.000630915, M2 0.000630915, chosen events 6
- seed 2034 K=20: M1 -0.00075376, M2 -0.00075376, chosen events 6
- seed 2035 K=20: M1 0.00121802, M2 0.00121802, chosen events 6

## 9. Budget accounting

gpu_budget_log TOTALS: cumulative 1693.713s / 1800s, crashes 0. Phases: smoke_S1_armA_bitwise_bank 21.19s, smoke_S2_ghost_noop_invisible 32.897s, smoke_S1_armA_bitwise_bank 19.262s, smoke_S2_ghost_noop_invisible 30.445s, smoke_S3_event_cost 0.001s, main_K20_seed_2030 84.109s, main_K20_seed_2031 85.75s, main_K20_seed_2032 90.554s, main_K10_seed_2030 112.72s, main_K10_seed_2031 142.456s, main_K10_seed_2032 108.374s, main_K20_seed_2033 93.247s, main_K20_seed_2034 106.645s, main_K20_seed_2035 100.865s, counterfactual_replay 490.105s, rerun_Bwc_seed_2030_K10 46.602s, rerun_Bwc_seed_2030_K20 15.58s, rerun_Bwc_seed_2031_K10 18.091s, rerun_Bwc_seed_2031_K20 15.438s, rerun_Bwc_seed_2032_K10 17.679s, rerun_Bwc_seed_2032_K20 15.724s, rerun_Bwc_seed_2033_K20 14.786s, rerun_Bwc_seed_2034_K20 15.584s, rerun_Bwc_seed_2035_K20 15.609s

## 10. Interpretation (contract §17)

**CASE A** (formal, contract sign rules) — equal-step>0 AND wall-clock>0 → net training benefit. Next (NOT auto-started): block planning / efficiency.
Basis: median equal-step gain 0.000744075, median Gain_time 0.000744075 (primary cadence K=20).

Qualifications attached to the formal case:
- Q2 is NEGATIVE: C≤B in 6/9 pairs (median -0.000129372); the fresh-response *selector* shows no advantage over the frozen fixed-j1 action. The gain comes from the intervention (data-group gradient injection at λ=0.05), not from adaptivity.
- The wall-clock win is scale-conditional: baseline never lost steps at this size (6/6 pairs); ROI median 3.755e-05 nats/s. At scales where T_extra would starve baseline steps, the same overhead flips the sign unless per-event cost drops sharply.
- Sign consistency is weak: K=20 equal-step positive in 4/6 seeds, negative in 2 (2032: −4.0e-4, 2034: −7.5e-4); K=10 positive in 3/3. Gains are O(1e-3) nats, close to run-to-run intervention noise; n=6 seeds cannot establish significance.
- M4: the controller never chose j=0 (no-op rate 0%): at every event some candidate had negative ĥ, so the argmin gate was never exercised against the no-op; harmful-rate dispersion (0–50%) shows per-event quality varies even when the gate passes.

## 11. Conclusion

Q1: see §3. Q2: see §4. Q3: see §5. Q4: see §6. Single-step replay gains (M4) are short-horizon diagnostic labels, **not** equal-wall-clock training gains; M2 is the economics verdict. Stop condition honored: no A0L / λ tuning / block planning / optimizer changes auto-started.

## 12. Fingerprints

See hashes/method_fingerprints.json. Historical A0G/A0H/A0I/A0J artifacts untouched.
