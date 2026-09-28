# A0N Response Phase-aware Pilot — REPORT

dashboard (final v_eval @128, equal steps; gains = loss reduction, positive = better):

| seed | L_A | L_B | L_C | gain vs AdamW | gain vs fixed-j1 | gain_time vs AdamW | gain_time vs fixed-j1 | inj | occ H/M/L | rho median |
---|---|---|---|---|---|---|---|---|---|---|
| 2030 | 6.94879 | 6.94748 | 6.94816 | 0.000624061 | -0.00067836 | 0.000624061 | -0.00067836 | 5/12 | 4/5/3 | 1.294 |
| 2031 | 6.93 | 6.92804 | 6.93026 | -0.000265539 | -0.00222784 | -0.000265539 | -0.00222784 | 2/12 | 8/2/2 | 2.445 |
| 2032 | 6.99005 | 6.98885 | 6.98879 | 0.00125933 | 5.94854e-05 | 0.00125933 | 5.94854e-05 | 7/12 | 4/7/1 | 1.799 |

## 1. Protocol

Frozen by contract.json before any GPU. Phase metric rho = |mean(h_1..4)|/(std(h_1..4)+1e-12) (population std). Policy: rho>2 -> no-op; 1<=rho<=2 -> inject argmin hhat_j (g+0.05*(g_j-g), tie->smallest id); rho<1 -> no-op. Response formula, optimizer, lambda, clipping untouched. K=10 events (steps 10..120) reused from A0K frozen manifest; seeds 2030-2032; T=128. Arms: A native, B fixed-j1 blind injection every event, C phase-aware, A_wc/B_wc at T_C.

## 2. Gates

- S1 arm A(2031) bitwise == v0.7 bank @46/52: **True**
- S2 ghost-C(2030) forced no-op == arm A tail: **True**
- POST-MAIN A/B final_v_eval bitwise == frozen A0K K10: **True**

## 3. Mandated metrics

1. Gain vs AdamW (equal steps): median **0.000624061** {"2030": 0.000624, "2031": -0.000266, "2032": 0.001259}
2. Gain vs fixed-j1 (equal steps): median **-0.00067836** {"2030": -0.000678, "2031": -0.002228, "2032": 5.9e-05} (1/3 seeds positive)
3. Equal wall-clock: vs AdamW median 0.000624061, vs fixed-j1 median -0.00067836 (scale condition HOLDS: A_wc/B_wc steps [128, 128, 128]/[128, 128, 128])
4. Controller overhead: T_extra median 33.2s, frac of T_C median 0.7354
5. Phase occupancy (median events/12): high 4 / medium 5 / low 2; injections median 5/12; rho median 1.888

## 4. rho(t) trajectory
per analysis/rho_trajectory.csv (seed, step, rho, phase, chosen action).

## 5. Verdict

Frozen rules -> **CASE B**: C≈B——response 主要发现固定策略

## 6. Budget

gpu_budget_log TOTALS: {"cumulative_wall_s": 416.185, "budget_s": 1800, "remaining_s": 1383.815, "n_crashes": 0}

## 7. Conclusion

Pilot only, per mandate: no optimizer/lambda changes, no expansion, no new rounds. A0K/A0L/A0M conclusions unchanged.
