# FR / GTTR — Results Summary

> All conclusions below are transcribed from frozen reports/summaries/seals on the research
> server. Nothing was re-run or re-derived during archiving. "Evidence:" gives the original
> server path relative to `legacy_fluctuation_response/`; the same evidence (small files)
> is mirrored under `results/outputs/` in this archive when size permits.

---

## 1 Research question

Can the *response* of AdamW training to small, controlled parameter perturbations be
(1) predicted exactly by smooth tangent propagation of the full optimizer state
(θ, m, v), (2) generalized across scale / data distribution / clipping regime, and
(3) converted into a cheaper or better training decision rule (GTTR) than tuned baselines?

Evidence: `docs/FR_PROJECT_STATUS.md` §1; `protocols/root_docs/FR_FRAMEWORK_README.md`

## 2 Research evolution (as documented, not inferred)

1. **Infrastructure (E-chain, 09-02)**: bit-exact replay of MiniMind training from saved
   full state + RNG + batch manifest (E0 smoke/extended PASS), construction of unit probe
   directions q and measurement of fluctuation scale σ_q (E1).
   Evidence: `outputs/e0_replay/v02_smoke/summary.json`; `outputs/e1_fluctuation/`
2. **Original FR hypothesis (E3/E3C, 09-02–09-03)**: response predictable from prior θ
   (theta-mapping). Preregistered confirmatory test at H=8 across 8 seeds **failed**
   universality; its by-product, the 56-state frozen bank, became the substrate for
   everything after.
   Evidence: `outputs/e3_confirmatory/v07/`; `protocols/root_docs/CONFIRMATORY_PREREGISTRATION_V07.md`
3. **A0 audit & mechanism (09-09–09-12)**: regressed what actually explains response;
   discovered the full smooth tangent (C ladder) explains nearly everything at H≤8 —
   no memory kernel needed — and closed the mechanism across 56 states.
   Evidence: `outputs/a0_fr_audit/response_propagation_v01/summary.json`
4. **Utility / cost / closed-loop / LR-control (A0G–A0U, 09-14–09-18)**: task-level utility
   of tangent-scored actions supported; wall-clock cost negative; LR (not clip/wd) is the
   only stable control channel; a sign LR controller passes its pilot.
   Evidence: `outputs/fr_decision_utility_v01/`; `outputs/a0o_lr_response_validation_v01/`;
   `outputs/a0t_response_lr_control_pilot_v01/`
5. **Generalization (G-line, 09-16–09-19)**: mechanism replicates at 26M, 100M, and on a
   large external corpus; clipping causally confirmed by intervention; external 1B
   replication blocked at phase 2 (scale-dependent C-branch failure).
   Evidence: `outputs/generalization_g1a_26m_v01/` … `generalization_g3_clipping_intervention_v01/`
6. **B0/C0 → GTTR (09-20–09-26)**: mechanism carriers identified; one-step geometric field
   validated with reservations (shadow unresolved); then the GTTR program (Phase A/B/D/R/CB/F)
   attempted to turn derivatives into rules and **failed at every decisive gate**, ending the line.
   Evidence: `protocols/root_docs/FR_GTTR_Experiment_Roadmap_and_Next_Run_v1.md`;
   `FR_GTTR_F2b_Second_Order_v1.md`; `docs/FR_PROJECT_STATUS.md` (termination section)

## 3 Main positive findings

### 3.1 A0C response propagation — the core mechanism result
- All-12 conditions: A mre 6.3636; B R²=0.8388 mre 0.1650; **C R²=0.9977 mre 0.0274**.
- H>1 (9 rows): C mre 0.0219 vs B 0.2053; **C beats B 9/9**.
- ε→ε/4 convergence: mre 0.00452 (residual confirmed as finite-ε artifact).
- Interpretation (frozen): full smooth tangent of (θ,m,v) explains nearly all response at
  H≤8; no memory kernel needed.
- Evidence: `outputs/a0_fr_audit/response_propagation_v01/summary.json`;
  `outputs/a0_fr_audit/response_propagation_v02_epsilon/summary.json`

### 3.2 A0E2 curvature decomposition (bug-fixed version)
- H=1 identity gap = 0 (24/24); H>1: B 0.2053 → BH_fixed 0.1366 (raw curvature +0.0686)
  → C 0.0219 (clip-JVP +0.1148); gain identity dev 8.9e-16.
- Evidence: `outputs/a0_fr_audit/curvature_decomposition_v02_bh_fixed/summary.json`
  (Note: v01 is VOIDED, see §5.)

### 3.3 A0F2 cross-partition closure
- 56-state census: **C beats B 168/224, 56 ties, 0 losses**; mean +0.1218; 7/7 checkpoint
  stages positive (weakest +0.0807); mre A 4.36 / B 0.128 / C 0.0059.
- Evidence: `outputs/a0_fr_audit/cross_partition_extension_v01/summary.json`

### 3.4 A0G decision utility (utility yes, cost no)
- L1 H4: C-Pearson 0.905, sign 0.880; L3 gain C 1.54e-4; cost: shared-C 6.77 s vs direct
  2.65 s (0.39×), break-even none → **utility supported / wall-clock cost negative**.
- Later corrected-accept pilot (A0H20): +1.467e-4, 5/6 seeds positive.
- Evidence: `outputs/fr_decision_utility_v01/summary.json`;
  `outputs/a0h_corrected_accept_v01/summary.json`

### 3.5 A0O LR response validation
- S1 n=96: **Pearson +0.9552**, sign 0.9688; analytic = FD; closed form
  e_lr = −adaptive(c) − wd·θ. Q1/Q2/Q3 all YES.
- Evidence: `outputs/a0o_lr_response_validation_v01/summary.json`

### 3.6 A0T LR control pilot
- gain_C_H4 median **+4.41e-4** (p=2.0e-10); consistency 35/36 (p=1.1e-9); terciles
  monotone; H16 success 0.806.
- Evidence: `outputs/a0t_response_lr_control_pilot_v01/summary.json`

### 3.7 G1a 26M / G1b 100M scale replication
- G1a: C R²=0.99958 mre 0.00792 (B 0.12033); H>1 18/18/0, mean +0.16861; replay 36/36
  bit-exact → **STRONG_SCALE_REPLICATION**.
- G1b: C R²=0.99949 mre 0.00876 (B 0.0788); H>1 18/18/0, mean +0.10503 →
  **STRONG_SCALE_REPLICATION**.
- Evidence: `outputs/generalization_g1a_26m_v01/summary.json`;
  `outputs/generalization_g1b_100m_v01/summary.json`

### 3.8 G2 data-distribution replication
- External large corpus: C R²=0.99995 mre 0.00260; H>1 18/18/0 mean +0.1336; bootstrap PI
  [+0.067, +0.251] → **STRONG_DATA_REPLICATION**.
- Evidence: `outputs/generalization_g2_data_distribution_v01/summary.json`

### 3.9 G3 clipping intervention (causal)
- C_identity == BH_fixed 36/36 bit-level; no-clip collapse 27/27; Δ_clip +0.09398,
  PI [+0.0086, +0.2938], P=0.99985 → **MECHANISM_CONFIRMED** (clip → response causal chain;
  curvature gain does NOT vanish without clip).
- Evidence: `outputs/generalization_g3_clipping_intervention_v01/summary.json`

## 4 Main negative findings

### 4.1 E3C preregistered confirmation — the founding negative
- K_8=0.3769; cluster CI lower 0.1909>0; circular-shift p=0.00015; **LOSO 5/8 < required 6**
  → `H8_CONFIRMATION: FAIL` (bank technical pass 8/8). Universal FR proportionality not
  supported; association exists (detrended r=0.5406) but universality fails.
- Evidence: `outputs/e3_confirmatory/v07/` (verdict in `summary.json` / frozen index)

### 4.2 A0P LR temporal kernel — gate failure
- Pooled Pearson +0.7314 (<0.80 threshold) FAIL; best R² 0.351; no stable decay law
  (fast initial decay + long noisy tail) → `STAGE-1 GATE FAIL → STOP`.
- Evidence: `outputs/a0p_lr_temporal_kernel_v01/summary.json`

### 4.3 GTTR D1 direct readout — no gain
- Implementation PASS, coverage 15/15, but **FINAL_BUDGET_NLL=NO_OBSERVED_GAIN** and
  TARGET_TOKEN=NO_OBSERVED_GAIN; NORM matching worse than RAW; fixed-LR baseline at grid
  boundary → `NO_OBSERVED_GAIN (both endpoints)`.
- Evidence: `outputs/fr_gttr_d1_direct_readout_v01/summary.json`

### 4.4 GTTR R2B solved-update bridge — one-step gain does not transfer
- One-step H1 FULL-LR mean −2.877e-4 (9/12 favorable) but **terminal G_struct(256) mean
  +3.362e-4 (8/12 positive = not improved)**; 3/12 reversals; DEPLOYABLE/SECOND_ORDER/V_FINAL
  not tested.
- Evidence: `outputs/fr_gttr_r2b_solved_update_bridge_v01/summary.json`

### 4.5 GTTR F2a gradient direction — direction not confirmed
- grad_b = −0.31078; sign match 3/4; best J at **b=+0.02** (J=6.69827), i.e. opposite the
  first-order gradient → `DIRECTION_NOT_CONFIRMED`.
- Evidence: `outputs/fr_gttr_f2a_gradient_direction_v01/summary.json`

### 4.6 GTTR F2b second order — prediction fails
- h2 negative at all scales (−21…−47); b*=−0.0146 but J(b*)=6.7127 > J(0)=6.7099 >
  J(+0.02)=6.6983 → `SECOND-ORDER PREDICTION FAILS`.
- Evidence: `outputs/fr_gttr_f2b_second_order_v01/summary.json`

### 4.7 Other notable negatives
- A0P-style time-structure: A0N phase scheduler CASE B (structure exists ≠ exploitable);
  A0L mixture CASE B (fixed ≥ controller); A0K selector not better than fixed (−1.29e-4).
- A0J cached future sensitivity: `TIME_SCALE_NEGATIVE` (margin 2–3 orders off).
- G1c external 1B: phase-2 C-branch failure (scale-dependent), line abandoned (#57–59).
- Evidence: `outputs/a0n_phase_scheduler_pilot_v01/`; `outputs/a0j_cached_future_sensitivity_v01/`;
  `outputs/generalization_g1c_feasibility_v01/`

## 5 Invalid / retracted work — DO NOT CITE

| Entry | Verdict | Reason |
|---|---|---|
| **A0E curvature decomposition v01** (`outputs/a0_fr_audit/curvature_decomposition_v01/`) | **VOIDED — DO NOT CITE** | BH initial pulse used raw q skipping J_clip; H=1 already inconsistent; superseded by v02_bh_fixed |
| **mv_theory_final** (`outputs/a0_fr_audit/mv_theory_final/`) | **RETRACTED — DO NOT CITE** | Claim numbers void (delta R2=−1.85; chi_mv R2=−6.33); superseded by `mv_theory_consistent` (v-path improvement 0/8 seeds, 0/8 horizons) |
| GTTR Phase B v3 | GATE_STOP (not invalid — honest prereg stop) | S3 fd_rel=0.7061 > 0.20 locality gate failure; real strong non-linearity, not an implementation bug |
| C0 first attempt | SOURCE_NOT_FOUND (aborted) | later re-run as C0-gttr-validation |

Evidence: frozen index entries #30, #24, #73, #65 in
`results/server_audit_20260926/02_COMPLETE_EXPERIMENT_INDEX.md`

## 6 GTTR final conclusion (frozen)

GTTR was terminated 2026-09-26 by its own stop rules. The frozen synthesis (from
`PROJECT_CONTEXT.md` §3.1 and the F2a/F2b protocols):

1. Derivative quantities could be **computed correctly** (F1 implementation closure:
   synthetic residual 5.6e-17; b-coordinate SENS=FD calibration passed; CB1 cost identity
   36/36 exact) — implementation was not the failure mode.
2. But calibrated derivatives did **not** predict finite-step reality: local FD operators
   uncalibrated at real step sizes (Phase B S3), direction of the terminal hypergradient
   not confirmed (F2a: optimum opposite the gradient), and second-order structure fails
   (F2b). One-step favorable interventions did not survive to terminal NLL (R2B), and
   direct readout showed no gain vs a tuned fixed-LR grid (D1).
3. Frozen one-line verdict (audits/02 index, GTTR-line header): "导数算得对但用不上"
   — derivatives computed correctly but not usable — the de-facto end of the line.

**Scope discipline:** the positive A0/G mechanism results (§3) are statements about
response prediction, scale/data generalization and causal clipping structure at 5M–100M
under the frozen protocol. They are **not** claims of long-term training-loss improvement,
and must not be cited as such.

Evidence: `outputs/fr_gttr_f2a_gradient_direction_v01/`; `outputs/fr_gttr_f2b_second_order_v01/`;
`outputs/fr_gttr_f1_implementation_closure_v01/`; `outputs/fr_gttr_cb1_cost_measurement_v01/`

## 7 What remains scientifically useful (per frozen asset classification)

- **Response propagation result & metric definitions**: `research_src/src/fr/a0c_adamw_response_propagation_v01.py`
  is the sole metric-definition source (`_metric`/`relative_error`); C-ladder mechanism.
- **Full-state checkpoint format**: `full_state.py` (save/load/capture, bit-level roundtrip
  tests) — adopted verbatim by successor projects (optimizer_sensitivity,
  continuation_adaptation vendored copies).
- **Determinism / CRN infrastructure**: `rng.py`, `stream_rng.py`, `batch_manifest.py`,
  `batch_stream.py`, `pilot_partition.py` — four-channel RNG capture, explicit batch
  manifests, probe/eval/train partitions.
- **Frozen banks**: v07 (56 states), G1a, G1b — still consumed by ACTIVE server projects
  (see `results/ARTIFACT_MANIFEST.md`, DO_NOT_DELETE_YET).
- **Evaluation & statistics**: `eval_observables.py` (eval-pool NLL),
  `confirmatory_stats.py`, `cross_seed_stats.py`, seal discipline (write_seal/verify_seal).
- **Governance pattern**: config pre-registration + red lines + SHA256 provenance +
  V_monitor/V_final separation — reused by all successor projects.
- **Mechanism findings** (§3.1–3.9) as scientific record.

Evidence: `docs/FR_PROJECT_STATUS.md`; `results/server_audit_20260926/05_REUSABLE_ASSETS.md`;
successor vendored copies in `optimizer_sensitivity/docs/LEGACY_PROVENANCE.md` (on server)

## 8 Limitations (frozen-documented)

- Model scale: anchor results at 4.98M params (h256/4L); controlled ladder to 26M/100M;
  external 1B blocked at phase 2 (C-branch, scale-dependent). No claim beyond tested scales.
- Corpus: D0 = 4096-row minimind subset; G2 uses a large external corpus but conclusions
  are protocol-bound.
- E3C universality negative means the original FR *law* framing is unsupported even though
  the tangent-mechanism results stand.
- Short-horizon emphasis: H≤32 typical; R2B explicitly showed short-horizon gains can
  reverse at terminal NLL — horizon-dependence is part of the record.
- theta_protocol.pyc is sourceless bytecode (see README).

Evidence: `results/server_audit_20260926/10_INCONSISTENCIES_AND_MISSING_ARTIFACTS.md`;
frozen CSV `limitations` column per experiment.

## 9 Reproducibility

- All runners are cwd-relative (`PYTHONPATH=research/src`), no hardcoded `/home/b205` paths
  (verified during the 09-26 migration audit).
- Bit-exactness culture: replay hash suites (5/5, 36/36), roundtrip unit tests, CRN
  three-branch protocol, per-config SHA256 provenance, sealed evaluation discipline.
- Environment: conda `fr_minimind`, torch 2.11.0 + CUDA, 2×RTX 5090; recorded per-era in
  `ENVIRONMENT_V07.txt` and config manifests.
- Git history of the source repo ends at commit `6ac4942` (2026-09-26, clean status).
  Pre-09-25 experiments are `UNKNOWN(pre-git)` by construction.
- Evidence: `protocols/root_docs/ENVIRONMENT_V07.txt`; `results/server_audit_20260926/07_GIT_AUDIT.md`

## 10 Large artifacts not stored on GitHub

See `results/ARTIFACT_MANIFEST.md` for the full list (3160 files, ~70.3 GB). Headline:

| Artifact | Size | Note |
|---|---|---|
| v07 bank (56 states) | 4.6 GB | **DO_NOT_DELETE_YET** — active dependency of successor projects |
| G1a bank | 3.7 GB | **DO_NOT_DELETE_YET** |
| G1b bank (bank + theta) | 14 GB | **DO_NOT_DELETE_YET** |
| G2 corpus pretrain_hq.jsonl | 1.55 GB | dataset, excluded |
| mv_theory_final/gradient_g.pt | 1.07 GB | RETRACTED experiment artifact |
| a0k / a0n / a0o / r2b / d1 raw outputs | 1.9–14 GB each dir | raw states & streams, excluded |

## 11 Original server sources

- Project root: `/home/b205/projects/training_dynamics_lab/legacy_fluctuation_response/`
  (read-only frozen; git HEAD `6ac4942`, clean).
- Experiment outputs: `.../legacy_fluctuation_response/outputs/` (55 subdirectories, 71 GB).
- Code: `.../legacy_fluctuation_response/research/` (src 2.2 MB / scripts 5.3 MB / tests 1.3 MB / configs 116 KB).
- Index source of truth: `/home/b205/projects/training_dynamics_lab/audits/FULL_SERVER_AUDIT_20260926/`
  (mirrored at `results/server_audit_20260926/`).
- Migration & integrity: `/home/b205/projects/training_dynamics_lab/PATH_MIGRATION_20260926.md`.
