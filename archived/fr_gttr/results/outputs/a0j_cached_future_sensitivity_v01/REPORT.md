# A0J Future-Sensitivity Reuse Study — REPORT

Round: A0J (post-A0I first controller-computation-structure modification). All semantics frozen per contract.json; only the future-sensitivity refresh policy varied, on pre-exposed dev trajectories only.

## ALGORITHM HEALTH DASHBOARD

| Metric | Result | Status |
|---|---:|---|
| Frozen one-shot V_eval Gain | +1.467e-4 (5/6 seeds positive, harmful 1/12) | CITE_ONLY (A0H20 frozen) |
| Adaptive vs fixed | positive point estimate, heterogeneous | OPEN |
| Regret / oracle capture | Regret 8.78e-5 / capture 62.6%; NEW: cache chosen-id preservation age1 5/6, age2 4/6, age4 1/6 | UPDATED |
| Exact decision cost | 5.618s = 104 native steps (quiet envelope) | MEASURED |
| Cache approximation utility retention | age1 -6.76e-05 / age2 -1.47e-04 / age4 -4.90e-05 (DeltaGain vs exact) | MEASURED (dev approximation evidence) |
| Equal-wall-clock final training gain | NOT TESTED | OPEN |

## 1. P0: prefix composition, top-3, S_upper

- Gate statistic (per-repeat interleaved, median): **S_upper = 4.4147** (min 3.7349, max 5.4991) >> 1.25 -> full P1 authorized.

- Whole-pass median wall: 5.618s (contended); stage-sum 1.917s; recomputable prefix (B+C+D+E) 1.361s.

- Top-3 prefix cost components (stage medians):
  - E3_curvature_HVP: 0.4553s
  - E1_optimizer_local_analytic_VJP: 0.3374s
  - G_candidate_gradient_generation: 0.2490s

## 2. G_t^T p implementation gate

- PASS (30/30 rows): 24 apply_Gt_T-vs-inline self-check rows at max_abs_w_diff=0.0; 6 production-reproduction rows (run_adjoint_only) all within frozen FP32 tolerance atol32=1e-5/rtol32=1e-4 with chosen-id match 6/6. Tolerances were frozen before any result was viewed.

## 3. Cache age 1/2/4: chosen-id preservation (PRIMARY, n=6 units/age)

| age | CACHE_P preserve | STALE_W preserve | CACHE_P rank-agree | STALE_W rank-agree |
|---:|---:|---:|---:|---:|
| 1 | 5/6 | 5/6 | 3/6 | 3/6 |
| 2 | 4/6 | 4/6 | 1/6 | 1/6 |
| 4 | 1/6 | 0/6 | 0/6 | 0/6 |

## 4. Pairwise action-gap error

- All pairs j<l over {0..4}, aged rows n=180: CACHE_P mean|err| 2.345e-03 (max 1.547e-02); STALE_W mean|err| 2.387e-03 (max 1.023e-02).

- Mismatches are near-ties: exact top1-top2 predicted-margin median 3.837e-05, min 3.899e-06; score errors (~2.0e-03 at age1, growing to 8.1e-03 at age4) exceed margins by ~2-3 orders of magnitude, so argmin flips are expected once errors exceed ~1e-5.

- On the exact top1-top2 pair (n=18): CACHE_P mean|err| 1.780e-03, STALE_W 1.544e-03. Fresh G_t does NOT systematically reduce score-space error at age4 (cache mean max|dh| 8.12e-03 vs stale 6.55e-03) yet preserves chosen-id better (1/6 vs 0/6) and loses less utility: decision quality and score-error magnitude dissociate, which is why score error is secondary and preservation is primary.

- SECONDARY diagnostic only: cosine(w_exact,w_cache) = 0.955 / 0.823 / 0.643 at age1/2/4 — high cosine coexists with flipped decisions; cosine is not a gate and was never used as one.

## 5. Dev utility (DeltaGain vs exact, n=6 units/age)

### §13 main result table

| age | cache chosen-id preserve | stale-w preserve | cache ΔGain vs exact | stale ΔGain vs exact | cache non-refresh cost | exact cost |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5/6 | 5/6 | -6.757e-05 | -6.757e-05 | 0.912s | 5.618s |
| 2 | 4/6 | 4/6 | -1.468e-04 | -1.468e-04 | 0.912s | 5.618s |
| 4 | 1/6 | 0/6 | -4.899e-05 | -9.271e-05 | 0.912s | 5.618s |

| age | exact Gain | cache Gain | stale Gain | dGain cache-exact | dGain stale-exact | cache chosen-match | stale chosen-match |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.404e-04 | 1.729e-04 | 1.729e-04 | -6.757e-05 | -6.757e-05 | 5/6 | 5/6 |
| 2 | 2.850e-04 | 1.383e-04 | 1.383e-04 | -1.468e-04 | -1.468e-04 | 4/6 | 4/6 |
| 4 | 1.708e-04 | 1.219e-04 | 1.219e-04 | -4.899e-05 | -9.271e-05 | 1/6 | 0/6 |

Dev approximation evidence on pre-exposed trajectories; NOT independent utility validation. Replay used frozen decisions (sha256 a0bd8911345d19e2...) verified before the run.

## 6. Costs (shared-GPU contention inflates absolutes; ratios comparable)

- exact decision: 5.618s (= 104 native steps at quiet 54.2ms; 40 at contended 142ms)

- cache refresh (full fresh-p recompute): 1.279s (= 24 native steps)

- cached non-refresh marginal: 0.912s (GtTp 0.132 + candidate score 0.594 + selected-action execution 0.185; = 17 native steps)

- cache memory: p (3-channel theta/m/v) 59799552 B; w 19933184 B; single-channel P 19933184 B; p/P = 3.0.

## 7. Kr/Ks cost projection (COST_ONLY / UTILITY_NOT_VALIDATED)

| K_r | K_s | extra native-step equiv | beyond measured max age | claim |
|---:|---:|---:|---|---|
| 1 | 1 | 710 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 2 | 1 | 503 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 4 | 1 | 399 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 2 | 2 | 355 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 4 | 2 | 251 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 4 | 4 | 178 | False | COST_ONLY / UTILITY_NOT_VALIDATED |
| 8 | 4 | 126 | True | COST_ONLY / UTILITY_NOT_VALIDATED |
| 8 | 8 | 89 | True | COST_ONLY / UTILITY_NOT_VALIDATED |

Only (K_r<=4, K_s<=2) sit inside the measured age-4 window; (8,4) and (8,8) extrapolate beyond it and are flagged.

## 8. Verdict

**B/C hybrid -> primary CASE B** — future sensitivity changes too rapidly for simple temporal reuse under the current setup at H=4/lambda=0.05 on this dev bank; cache route STOP at ages>=1 as tested; do not tune thresholds to rescue

- CASE B trigger: CACHE_P and STALE_W both already alter actions at age1 (full-rank agreement only 3/6 for both arms; score errors ~2e-3 vs exact top1-top2 margins median 3.8e-5)
- utility degrades: mean dGain_cache-exact is negative at every age (-6.8e-5 / -1.5e-4 / -4.9e-5)
- CASE C element: CACHE_P is consistently but only mildly better than STALE_W (age4 chosen 1/6 vs 0/6; dGain -4.9e-5 vs -9.3e-5), i.e. fresh G_t has value but the p reuse window is short
- CASE D ruled out: non-refresh cost saving is real (0.91s vs 5.62s exact, ~6x) so the failure is not engineering leverage
- cosine remains high (0.95/0.82/0.64 cache w at age1/2/4) while decisions flip: cosine must not be used to rescue the route

The cache route as specified (cached p_tau + fresh G_t, ages 1/2/4) does not preserve decisions and does not retain utility on this dev bank. The negative result is a genuine time-scale finding: future sensitivity decorrelates on the 1-4 native-step horizon far faster than the cost structure requires for reuse.

## 9. GPU totals / crashes / retries / environment

- cumulative GPU seconds (estimated, incl. preloads/probes/crashed sessions): 1640.760753221839 of 1800s budget

- crashes: 4 — p0_attempt1_E4_KeyError; p1_attempt1_NameError_rank_agreement; p1_attempt2_AttributeError_norm_map_dict; p1_replay_attempt1_ImportError_stable_hash

- retries: all infra/serialization-class under same manifest/algorithm; no results viewed between an attempt and its fix; 4 module-hash updates logged in manifest BEFORE result viewing

- environment changes: none (no upgrades; no external tasks killed)

## 10. Audit tar

- path: /home/b205/projects/fluctuation_response/outputs/a0j_cached_future_sensitivity_v01/a0j_audit.tar.gz
- bytes: 9066973
- SHA256: b46c96179aa275bdc1575430becd9905a9b2b7aa432487c982d14aa248240bef
- entries: 51 files (self-verify + listable/unpackable OK)

## 11. Next-step suggestions (not auto-started)

- Treat this as a time-scale negative result; do not tune cache thresholds or switch to cosine gating to rescue temporal reuse.
- If future-sensitivity amortization is revisited, candidates are: much shorter reuse windows (sub-step, K<1 not meaningful here), p refresh tied to measured p-drift rather than fixed ages, or reusing only the endpoint-gradient *direction* with a fresh scalar — each needs a new pre-registered round.
- Independent of the cache verdict, the P0 cost map and the production-reproduction gate (apply_Gt_T reproduces production w/scores/chosen-id bitwise) are reusable engineering assets for any future controller study.

## §17 separate output fields

```json
{
 "P0_PROFILE_COMPLETE": "YES",
 "CACHE_IMPLEMENTATION_GATE": "PASS",
 "CACHE_DECISION_PRESERVATION": {
  "age1": "5/6",
  "age2": "4/6",
  "age4": "1/6"
 },
 "STALE_W_DECISION_PRESERVATION": {
  "age1": "5/6",
  "age2": "4/6",
  "age4": "0/6"
 },
 "CACHE_UTILITY_RETENTION": {
  "age1": {
   "DeltaGain": -6.757179896036784e-05,
   "DeltaGain_stale": -6.757179896036784e-05
  },
  "age2": {
   "DeltaGain": -0.00014675656954447427,
   "DeltaGain_stale": -0.00014675656954447427
  },
  "age4": {
   "DeltaGain": -4.898508389790853e-05,
   "DeltaGain_stale": -9.270509084065755e-05
  }
 },
 "CACHE_COST_REDUCTION": {
  "refresh_cost_s": 1.2788531294936547,
  "cached_nonrefresh_marginal_cost_s": 0.9119390700070653,
  "exact_decision_cost_s": 5.618431106995558,
  "native_step_equivalents": {
   "exact": 103.71332988856499,
   "refresh": 23.606966779935274,
   "cached_nonrefresh": 16.833923172636435
  }
 },
 "CACHE_MEMORY_COST": {
  "p_3channel_bytes": 59799552,
  "w_bytes": 19933184,
  "P_single_channel_bytes": 19933184,
  "p_over_P": 3.0
 },
 "EQUAL_WALLCLOCK_TRAINING_GAIN": "NOT_TESTED"
}
```
