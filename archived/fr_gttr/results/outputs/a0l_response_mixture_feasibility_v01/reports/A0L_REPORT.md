# A0L Response-guided Continuous Mixture Feasibility — REPORT

dashboard (final v_eval @128; gains = loss reduction, positive = better):

| seed | L_A | L_B | L_C | gain C vs A | gain B vs A | C vs B (B−C) | Gain_time vs A | Gain_time vs B | ordering |
|---|---|---|---|---|---|---|---|---|---|
| 2030 | 6.94879 | 6.94389 | 6.94262 | 0.00616932 | 0.00489604 | 0.00127327 | 0.00616932 | 0.00127327 | C < B < A |
| 2031 | 6.93 | 6.93838 | 6.96024 | -0.0302451 | -0.00838143 | -0.0218637 | -0.0302451 | -0.0218637 | A < B < C |
| 2032 | 6.99005 | 6.95657 | 6.96403 | 0.0260255 | 0.0334828 | -0.00745738 | 0.0260255 | -0.00745738 | B < C < A |

## 1. Protocol

Frozen by contract.json before any GPU. Arms: A native AdamW; B fixed-j1 continuous mixture (all micro-batches from group 1); C response mixture controller (K=50, events before steps 50/100: snapshot -> production H=4 response h_j -> q=softmax(-gamma_t h), gamma_t=1/(std(h)+eps) -> pi=(1-0.1)pi+0.1 q -> restore; pi held per window). pi_0 = empirical group proportions [0.2637,0.2472,0.2508,0.2383] (groups partition the 3936-sample train pool = original distribution). Treatment = data composition only; no gradient injection; optimizer/clipping/H/lambda untouched. Pilot: seeds 2030–2032, no expansion.

## 2. Identity gates

- S1 arm A(2031) bitwise == v0.7 bank at 46/52: **True**
- S2 arm B(2030) determinism (two runs identical tail): **True**

## 3. Evaluation items

1. Final Eval Loss: see dashboard table.
2. Gain vs AdamW: C median **0.00616932** per seed {"2030": 0.006169, "2031": -0.030245, "2032": 0.026025}; B median 0.00489604.
3. Gain vs fixed-j1 (B−C): median **-0.00745738** {"2030": 0.001273, "2031": -0.021864, "2032": -0.007457}.
4. Equal wall-clock gain vs AdamW: median **0.00616932** (A_wc completed all 128 steps inside T_C in every pair — scale condition, Gain_time == equal-step gain numerically; T_extra 6.373s).
5. pi trajectory (C arm):
  - seed 2030:
    - step 0: pi_0 = [0.2637, 0.2472, 0.2508, 0.2383]
    - step 50: pi_before [0.2637, 0.2472, 0.2508, 0.2383] -> q [0.0322, 0.3405, 0.3589, 0.2684] -> pi_after [0.2406, 0.2565, 0.2616, 0.2413] (hhat [-0.00203, -0.0056, -0.00567, -0.00523], std_h 0.001513)
    - step 100: pi_before [0.2406, 0.2565, 0.2616, 0.2413] -> q [0.3024, 0.0341, 0.4805, 0.183] -> pi_after [0.2468, 0.2343, 0.2835, 0.2355] (hhat [-0.00111, 0.00048, -0.00145, -0.00074], std_h 0.0007304)
  - seed 2031:
    - step 0: pi_0 = [0.2637, 0.2472, 0.2508, 0.2383]
    - step 50: pi_before [0.2637, 0.2472, 0.2508, 0.2383] -> q [0.126, 0.1194, 0.7105, 0.0441] -> pi_after [0.25, 0.2344, 0.2967, 0.2189] (hhat [-0.00541, -0.00535, -0.0072, -0.00432], std_h 0.001036)
    - step 100: pi_before [0.25, 0.2344, 0.2967, 0.2189] -> q [0.2382, 0.1893, 0.0345, 0.538] -> pi_after [0.2488, 0.2299, 0.2705, 0.2508] (hhat [-0.0011, -0.00104, -0.00059, -0.00132], std_h 0.0002636)
  - seed 2032:
    - step 0: pi_0 = [0.2637, 0.2472, 0.2508, 0.2383]
    - step 50: pi_before [0.2637, 0.2472, 0.2508, 0.2383] -> q [0.4198, 0.4114, 0.0369, 0.1319] -> pi_after [0.2793, 0.2636, 0.2294, 0.2277] (hhat [-0.00513, -0.00512, -0.00361, -0.00441], std_h 0.0006251)
    - step 100: pi_before [0.2793, 0.2636, 0.2294, 0.2277] -> q [0.734, 0.0835, 0.0519, 0.1306] -> pi_after [0.3248, 0.2456, 0.2116, 0.218] (hhat [-0.00273, -0.00131, -0.00099, -0.0016], std_h 0.0006536)

## 4. Interpretation (mandate cases)

Per-seed orderings: {"2030": "C < B < A", "2031": "A < B < C", "2032": "B < C < A"}
|median(C−B)|/|median(B−A)| = 1.52
**CASE B** — C approx B > A -> consider response-discovered schedule. OBSERVED QUALIFICATION (this pilot): B >= C on median — B beats C in 2/3 seeds (2030 C better by +1.27e-3; 2031/2032 B better by +2.19e-2/+7.46e-3), |median(C-B)| = 1.52 x |median(B-A)|, so 'C approx B' holds only weakly: the response controller did NOT outperform the trivial fixed-j1 mixture.

## 5. Budget accounting

gpu_budget_log TOTALS: cumulative 351.81s / 1800s, crashes 0.

## 6. Conclusion

Verdict: FORMAL CASE B (deterministic rule: strict C<B<A in all seeds not met; median C-gain vs A positive) — with the qualification that B >= C on median: the response mixture controller did not outperform the trivial fixed-j1 mixture in this pilot, and no cross-seed ordering is consistent. The pi trajectories show the controller moved only modestly from pi_0 (alpha=0.1, 2 events; max |pi-pi_0| ~ +0.061 on component 1, seed 2032 step 100): the response signal was translated into mixture updates too small and too infrequent to separate C from B. Per the mandate, the continuous response controller route is NOT entered on this evidence; the fallback reading (response-discovered schedule) is weakened by B >= C — a fixed group-1 mixture captures the available gain without any response computation. Q: why the response cannot directly control long-term mixture — its h_j ordering at step 50 was seed-dependent and its scale (std-normalized) collapses q toward moderate reweighting; two K=50 updates are insufficient to test the hypothesis at longer horizons. Stop condition honored: no expansion, no lambda/optimizer changes, no new rounds auto-started.
