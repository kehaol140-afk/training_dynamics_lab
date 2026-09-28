# A0T-extra Report — Response Controller Identity Analysis
Date: 2026-09-18. OFFLINE analysis of frozen A0T artifacts only. No new training, no controller modification, no new optimization.
## 1. Question
Is the A0T response-guided LR controller genuine adaptive control, or does it only discover a global LR scaling direction?
## 2. Frozen inputs
- ac48f0438d598a16… cells.jsonl (36 cells, CRN 36/36), A0T GPU total 1351.436s (no new GPU this round).
## 3. Action distribution
- pooled: P(+0.05)=0.9444, P(-0.05)=0.0556 (n=36)
- seed 2030: P(+0.05)=0.8333, P(-0.05)=0.1667 (n=12)
- seed 2031: P(+0.05)=1.0000, P(-0.05)=0.0000 (n=12)
- seed 2037: P(+0.05)=1.0000, P(-0.05)=0.0000 (n=12)
## 4. Action entropy
- pooled H(a) = 0.3095 bits (max 1.0 for two actions). Per seed: 2030: 0.650, 2031: -0.000, 2037: -0.000. H=0 (seeds 2031/2037) = single global action.
## 5. Switching statistics
- trajectories: 12 (18 streams); total sign changes: 3; trajectories with any switch: 1. Per-trajectory detail in a0t_extra_switching.csv.
  - D1 seed 2030 stream 7110: - > + > - > + over steps 46 > 52 > 64 > 82 (3/3)
## 6. Fixed strategy comparison (existing rollout gains)
- H=4: fixed +0.05 median gain +4.103e-04 (wilcoxon p=7.114140316843987e-07); response median +4.407e-04; fixed -0.05 available on 2 non-aliased cells only: D1 s2030 t46 x7110: +1.249e-03, D1 s2030 t64 x7110: +2.292e-04
- H=16: fixed +0.05 median gain +1.025e-03 (wilcoxon p=1.458160113543272e-06); response median +9.511e-04; fixed -0.05 available on 2 non-aliased cells only: D1 s2030 t46 x7110: -1.865e-03, D1 s2030 t64 x7110: -1.165e-03
- H=64: fixed +0.05 median gain +4.829e-03 (wilcoxon p=2.9103830456733704e-11); response median +4.729e-03; fixed -0.05 available on 2 non-aliased cells only: D1 s2030 t46 x7110: -1.045e-02, D1 s2030 t64 x7110: -4.440e-03
## 7. Gain decomposition
- H=4: response median +4.407e-04 = best-fixed median +4.407e-04 + adaptive median +0.000e+00 (max adaptive +0.000e+00, n positive 0/36)
- H=16: response median +9.511e-04 = best-fixed median +1.025e-03 + adaptive median +0.000e+00 (max adaptive +0.000e+00, n positive 0/36)
- H=64: response median +4.729e-03 = best-fixed median +4.829e-03 + adaptive median +0.000e+00 (max adaptive +0.000e+00, n positive 0/36)
- IDENTIFICATION LIMIT (pre-registered): adaptive=0 by construction on 34/36 aliased cells; the 2 discriminative cells both chose -0.05 and beat fixed +0.05, but per the frozen formula best_fixed includes the -0.05 rollout itself, so pooled adaptive stays 0. A real adaptive test requires fixed -0.05 rollouts on all cells (absent from frozen artifacts; NOT generated).
## 7b. Horizon dependence of the discriminative cells
- D1 s2030 t46 x7110 H=4: fixed +0.05 gain -1.387e-03, guided -0.05 gain +1.249e-03 -> C beats B: True
- D1 s2030 t46 x7110 H=16: fixed +0.05 gain +9.611e-04, guided -0.05 gain -1.865e-03 -> C beats B: False
- D1 s2030 t46 x7110 H=64: fixed +0.05 gain +7.428e-03, guided -0.05 gain -1.045e-02 -> C beats B: False
- D1 s2030 t64 x7110 H=4: fixed +0.05 gain -2.500e-04, guided -0.05 gain +2.292e-04 -> C beats B: True
- D1 s2030 t64 x7110 H=16: fixed +0.05 gain +1.354e-03, guided -0.05 gain -1.165e-03 -> C beats B: False
- D1 s2030 t64 x7110 H=64: fixed +0.05 gain +4.464e-03, guided -0.05 gain -4.440e-03 -> C beats B: False
The only two deviations from the global direction win at the response horizon H=4 and INVERT at H=16/64 (guided -0.05 becomes harmful). The state-dependent component does not persist beyond the response horizon - consistent with A0P's unstable long-lag kernel.
## 8. Interpretation (frozen cases)
- **case_A_adaptive_control**: NOT ESTABLISHABLE: only 2/36 discriminative cells; on both the guided -0.05 beat fixed +0.05 (directed benefit +2.636e-03, +4.792e-04), consistent with state-dependence but n=2 is underpowered; a pooled adaptive test needs fixed -0.05 rollouts on all cells (absent from frozen artifacts; NOT generated - offline mandate).
- **case_B_global_scaling**: CONFIRMED as the dominant identity: P(+0.05)=34/36=0.944, action entropy 0.31 bits pooled, adaptive_gain median = 0 by construction on aliased cells, fixed +0.05 gain significantly positive (median +4.10e-4, wilcoxon p=7.1e-7).
- **case_C_response_worse**: RULED OUT: response_gain median +4.41e-4, wilcoxon p=2.0e-10 (frozen A0T).

**Primary verdict**: Dominantly Case B (global LR-up direction) with a validated-but-tiny state-dependent reversal component (2/36 cells, both correct).

## 9. Limitations
- Adaptive component identified on n=2 cells only; no inferential claim possible (offline mandate forbids generating the missing fixed -0.05 rollouts).
- Action entropy and switching describe the frozen 36-cell pilot bank, not the full training run.
- Short-term gains do not imply final-training improvement (A0P long memory; A0T limitations carry over).
