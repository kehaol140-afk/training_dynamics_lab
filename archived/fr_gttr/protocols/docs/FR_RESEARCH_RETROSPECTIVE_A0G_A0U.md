# FR Research Retrospective: Finite-Time Response Dynamics and Control in Transformer Training
## From the Fluctuation-Response Hypothesis to A0U (2026-09-10 → 2026-09-18)

**Provenance note.** The theory document named in the retrospective mandate
(`FR_State_Action_Objective_Theory_and_A0O_Design_20260918.md`) is **not present in
the repository** under that name. Part 3 is therefore grounded in the frozen theory
blocks that do exist: the A0O contract's `frozen_science_semantics.theory`
(`outputs/a0o_lr_response_validation_v01/contract/contract.json`), STATUS §2–§4
(strict definitions, model ladder), and the frozen A0C/A0E2 mechanism papers. All
numerical claims below are taken from `docs/FR_PROJECT_STATUS.md` §17–§45 and
`docs/FR_EXPERIMENT_INDEX.md` (frozen artifacts) and from the per-experiment reports
under `outputs/`. Where a number survives only in a high-level summary and not in a
report I re-verified, it is marked **[summary-level]**.

---

## Part 1 — Complete Experiment Timeline

The chain is: hypothesis → experiment → observation → interpretation → failure
boundary. "PASS/FAIL" below refers to the experiment's **own frozen gate**, not to
whether the science was valuable — several FAILs are the project's most important
results.

| Experiment | Scientific question (assumption tested) | Method | Result | Gate | Meaning |
|---|---|---|---|---|---|
| **A0G** (decision utility) | Can the full tangent response **C** select data-reweighting actions that *truly* reduce future validation loss (not just fit FD)? Assumption: response is a task-level decision signal. | Formal: 4 seeds × steps {46,64,82} × 8 streams, 96 cells; action g+λ(g_j−g), λ=0.05, H=4/8; V_score/V_eval separation; + cost timing vs per-candidate replay | Level-1: C vs FD Pearson **0.905**, sign 0.880. Level-3 H=4: C mean Gain **1.54e-4** > B > alignment > random; H=8 C vs B bootstrap CI excludes 0. **Cost: shared tangent slower than direct replay (speedup <1, no break-even up to M=64)** | Utility PASS; **cost FAIL** | Response is a valid *decision* signal independent of FD fit. But at 5M scale, computing it is **not cheaper than just replaying candidates** — the economics of response-based control are unfavorable before any control question is asked. |
| **A0H** (adjoint cost) | Can a reverse adjoint replace per-candidate-per-step HVPs and cut cost without losing decision quality? Assumption: the frozen forward tangent's linear structure permits cheap transposition. | A0H-1: CPU synthetic (PASS), GPU dev equivalence vs frozen tolerances; A0H-1.5: candidate-shared VJP adjoint; A0H-2.0 (A″): corrected acceptance pilot, 12 units, FP32 gate | Elementwise gate **FAIL at H=8** (bf16 math-SDPA HVP proxy is transpose-asymmetric; noise floor exceeds frozen tolerance); decision-level equivalence PASS throughout. A″: FP32 gate 12/12, decision preservation 12/12, V_eval gain **+1.467e-4** (bootstrap PI [+8.7e-5, +2.1e-4], labeled POSITIVE_POINT_ESTIMATE, not a significance claim) | Elementwise FAIL (H=8); A″ pilot PASS | The adjoint is the *correct transpose of what was actually computed* — the failure is the forward proxy's asymmetry, not the adjoint. Lesson: decision-level equivalence survives elementwise numerical failure; FP32 HVPs remove the floor. |
| **A0I** (algorithm efficiency) | A0I-A: where does the controller gain come from — adaptation or intervention? A0I-B: can scalar directional scoring compress marginal decision cost? | Offline label re-diagnosis (A0I-A); dev-only scalar-identity scoring with 7-field verdict (A0I-B) | A0I-A: ΔGain(controller − fixed j1) = +5.76e-5 mean but **seed-heterogeneous** (−1.8e-4…+3.2e-4); action switching observed ≠ net adaptive gain. A0I-B: scoring sub-stage −30%, memory −17%, **total cost SIMILAR** (shared prefix dominates) | CASE B | First honest separation of "intervention helps" from "selection helps": **selection's contribution is uncertain-to-negative**. Marginal-cost engineering cannot beat the shared-prefix floor. |
| **A0J** (cached sensitivity) | Can cached future-sensitivity p_tau (age 1/2/4) + fresh G_t reproduce fresh-controller actions at lower cost? Assumption: p_tau is stable enough to cache. | Pre-registered cost gate → implementation gate (30/30 after a real `_clip`/`_step_once` bug found by probe chain) → 3-way EXACT/CACHE_P/STALE_W comparison, n=6/age | Chosen-id preservation collapses with age (5/6 → 4/6 → 1/6); ΔGain cache−exact all negative; margin median 3.8e-5 vs age-1 score error ~2e-3 (**2–3 orders too coarse**); cosine 0.955 coexists with flips | **TIME_SCALE_NEGATIVE (CASE B)** | The response's *direction* is not the bottleneck — the **decision margin between top candidates is 2–3 orders below the drift of any cheap surrogate**. Defined the compression boundary: no caching scheme at this cadence can work. |
| **A0K** (fresh closed loop) | Does a fresh-response controller produce **net training gain under a real training budget**? Assumption: closed-loop use of the A0H-2.0 decision mechanism beats both native and its own fixed-action ablation. | First true closed loop: 6 seeds × 12 events, K=20 cadence, 5 candidate data-mix injections; 3 arms (A native / B fixed-j1 / C controller) + equal-wall-clock arms; identity gates S1–S3 PASS | Equal-step A−C median **+7.44e-4** (4/6 positive); equal-wall-clock scale condition holds (baselines never lost steps). **M5: adaptive vs fixed-j1 = −1.29e-4 (C≤B 6/9)**; no-op gate never fired; ROI ~5e-5 nats/s | **FORMAL CASE A with 3 qualifications** | Closed-loop response control *works end-to-end and is net-positive vs native at this scale* — but the gain comes from the **intervention**, not the **selector**. Q2 (adaptivity) is negative. Wall-clock win is a scale artifact. |
| **A0L** (continuous mixture) | Can response graduate from one-shot action selection to **continuous data-mixture control**? Assumption: a π-updating mixture controller beats any fixed mixture. | 3 seeds, K=50 cadence, frozen scale-free q=softmax(−γh), α=0.1; 5 arms incl. fixed-j1 continuous mixture; treatment = data composition only, optimizer untouched | C vs A median +6.17e-3 (2/3 seeds); **B−C median −7.46e-3 — fixed-j1 mixture ≥ controller in 2/3 seeds**; π moved little (max component 0.264→0.325); cross-seed rankings inconsistent | **CASE B (B≥C)** | A trivial fixed mixture captures the median benefit without any response computation. The continuous-control hypothesis **fails at pilot scale**; "response-discovered schedule" readings are weakened. |
| **A0M** (temporal structure) | Does the response contain **training-phase information**? (Pure mechanism analysis, zero GPU.) | Offline over frozen A0K/A0L events: E1 trajectory, E2 phase clustering, E3 stability, E4 utility, E5 fixed-j1 cause; classification rule frozen in script docstring | Common mode \|mean(h)\|/std median **1.85** — responses are dominated by a shared "intervention helps" component that decays **~8–10×** early→late; argmin identity is phase-dependent in **8/9** seed-series; but rank order is near-tie noise (rank-change 0.83, lag Spearman ≈ 0); **window-averaging h (w=2..4) degrades utility — w=1 optimal**; j1 is argmin in only 30.6% of events | **CASE A** (phase structure exists — in *magnitude and argmin identity*, not in rank order) | Gave the field its working decomposition: response = large common mode + small differential. Phase structure is real and carries the schedule-relevant signal — but per-event, the differential is below noise. |
| **A0N** (phase-aware pilot) | Can A0M's ρ(t) phase structure **guide training policy** (ρ-gated injection)? | Frozen ρ-gating (ρ>2 no-op, 1≤ρ≤2 inject argmin, ρ<1 no-op), A0K K=10 events, 3 seeds; ghost gate PASS; ρ(t) reproduced the A0M phase decay in-loop | C−A median +6.24e-4 (2/3); **B−C median −6.78e-4 (B≥C)**; ρ occupancy non-degenerate; gated injections caused no net loss but no net gain | **CASE B** | Phase-aware *gating* adds nothing over blind fixed injection. The phase structure is observable in-loop but **not exploitable by this policy class**. |
| **A0O** (LR input response) | Does the response framework extend from **data perturbation to an optimizer-input channel** — is the LR response a predictable causal object? Assumption: LR pulse is a pure θ-channel push through frozen AdamW. | Persistent multiplicative log-LR u=log(η/η₀), δ=0.05 frozen, every step for H steps; analytic h_eta via frozen adjoint vs FD; Stage 2 action selection on 3 states × 8 streams | Stage 1 (n=96): Pearson **+0.9552**, sign accuracy **0.9688** → gate PASS. Theory confirmed: h_eta = Σ_k ⟨p_th^(k), lr_k·e_lr(k)⟩, e_lr = −adaptive(c) − wd·θ (m/v are lr-independent). Stage 2: direction accuracy **0.875** vs null 0.125, regret 0.0, calibration +0.942 | **Q1/Q2/Q3 YES** | LR is a *clean, theoretically understood* input channel — the response is not a black-box empirical fit but a computable derivative of the frozen optimizer. This opened the entire O→U control line. |
| **A0P** (impulse kernel) | Does the LR response define a **temporal law** (impulse response R(τ)) with a stable decay timescale? Assumption: short-horizon controllability extends to a kernel usable for long-lag prediction. | Impulse u at step 1 only; analytic kernel via adjoint with source at k=1; pooled n=84; frozen A0O-imported gate (Pearson ≥ 0.8 AND sign ≥ 0.9) | Gate **FAIL**: pooled Pearson **+0.7314** (<0.8), sign 0.9405. Structure: fast initial decay (half-life ~4–8 steps, halving τ1→τ4) + **long noisy non-converging tail** (｜R｜~1.5–3.9e-3 throughout); neither exponential (R² 0.071) nor power law (R² 0.306) fits; CV(τ_c)=3.32 | **STAGE-1 GATE FAIL → STOP** | **The project's cleanest negative result.** First-order tangent response is a *short-horizon* object; there is no stable temporal law bridging to long horizons. Long-horizon prediction needs something else (curvature integration, memory-kernel, or re-rolled response). |
| **A0Q** (control map) | What is the full **input→response map** across optimizer channels? Assumption: multiple channels are simultaneously controllable. | One shared forward+rewind gives J = [h_lr, h_clip, h_wd]; persistent u=log(x/x₀), δ=0.05; 12 cells; per-channel A0O gates | LR: Pearson +0.842, sign 1.000 → PASS. clip: Pearson +0.936 but sign 0.833 → **FAIL** (clip never binds in 7/12 cells; event-driven knockout). wd: Pearson +0.148 → FAIL (｜h｜~2e-6, 4 orders below LR, FD noise-dominated) | LR PASS; clip/wd FAIL | **LR is the sole stable control channel**; clip is real but event-driven; wd is negligible. The control-map methodology works; the channel inventory is thin. |
| **A0S** (controllability identification) | Does response **magnitude contain controllability information** (can ｜χ｜ predict intervention gain and direction)? Assumption: response is not only a sign oracle. | D1: frozen A0O bank n=48, fully offline (analytic h paired with recorded rollout losses). D2: A0Q cells + new rollout losses via import-only frozen runner. OLS + LOSO; C_J channel shares; Model A (LR-only) vs Model B (3-channel) | A0S-1: D1 Spearman **+0.762** (p=3.2e-10), OLS β=0.0449 CI [0.036, 0.054], LOSO pooled Pearson +0.825 (all folds positive). A0S-2 direction: D1 46/48, D2 12/12. A0S-3: LR median C_J share **1.0**; Model B LOSO-MSE *worse* than Model A | **PASS (finite H=4 only)** | Response magnitude is a **usable controllability coordinate** at short horizon, and LR alone carries all of it. Mandatory caveat registered: short-term G (G/L0~7e-5) must not be read as long-term gain (A0P). |
| **A0T** (LR control pilot) | Does the A0S controllability convert into a useful **fixed control rule**? Assumption: sign(χ) → u=±0.05 every step beats native. | 36 cells; frozen sign controller; arms A native / B fixed +0.05 / C guided; H∈{4,16,64} on one nested 128-micro stream; C≡B aliasing rule (34/36); CRN 36/36 | Gate PASS: gain_C,H4 median **+4.41e-4** (p=2.0e-10); sign consistency 35/36; C not worse than B. C_t terciles **monotone** (success 0.875/1.0/1.0). Erosion: H=16 success dips to 0.81. **χ sign is stream-dependent** (2/18 states disagree across streams) **[summary-level detail verified in session]** | **PASS (frozen rule)** | End-to-end response→action pipeline works; controllability magnitude *predicts* controller success. But C≡B by construction on 34/36 cells — the design cannot yet separate guidance from fixed scaling. |
| **A0T-extra** (controller identity) | **What IS the controller** — true adaptive control, or discovery of a global scaling direction? (Offline, frozen A0T artifacts.) | Five frozen analyses: action distribution/entropy, switching, fixed-policy comparison, gain decomposition | P(+0.05)=0.944; H(a)=0.310 bits; 1/12 trajectories switches; adaptive gain median **0 by construction**; the 2 discriminative cells: guided −0.05 wins at H=4 (+2.64e-3, +4.79e-4) but **inverts at H=16/64** | **Case B dominant** | The controller's identity is "global LR-up discovery" + a validated-but-tiny state-dependent reversal that does **not** persist beyond the response horizon. Claims of adaptive value require banks with sign diversity. |
| **A0U** (schedule discovery) | Can response dynamics reveal a better **training schedule** than fixed LR rules? (Offline schedule signal, open-loop — explicitly not another online controller.) | Frozen partial-window K=4 sign rule on 4-knot χ trajectories → per-segment multipliers over W=64 rollouts from step 46; arms A native / B ×1.05 / C schedule; 12 cells; checkpoints @bank{58,70,82,94,110} | Stage 1: 11/12 trajectories all-χ<0; cross-seed **sign** agreement 1.0 at every knot (streams 7111–7113) but magnitude-trajectory Pearson mostly weak — exploitable content ≈ **one bit**. Stage 2: 12/12 schedules switch exactly once; K=4 never materializes. Stage 3: A−C +4.87e-3 (p=3.4e-3) but B−C −3.3e-4 (n.s.) final / **−3.6e-2 (p=1.2e-2) on AUC**; C ≈ B delayed 6 steps | **Case B** | Schedule-level confirmation of A0T-extra: **on a single-sign bank, any response-driven rule degenerates to fixed scaling + an inaction prefix**. The frozen rule's empty-window prefix cost is measurable (−3.3e-4). |

**Timeline reading.** Three regime shifts: (G–J) "can the response be computed
cheaply enough to use" → answered mostly *no* at 5M scale; (K–N) "can it control
training" → control works but selection adds nothing; (O–U) "can it control the
*optimizer*" → clean channel physics, but the exploitable information content
collapses to one sign bit, and long-horizon extension fails.

---

## Part 2 — Three Levels of Conclusions

### Level 1 — Proven facts (directly experiment-supported)

1. **Finite-time input→response structure exists and is measurable.** For data
   pulses (A0C: C model R²=0.9977 at H≤8, reproduced bit-exactly by A0E2) and for
   the LR input channel (A0O: Pearson +0.9552 vs FD, sign 0.969, n=96).
2. **The response is a computable derivative of the frozen optimizer, not an
   empirical fit.** A0O proved the LR pulse is a pure θ-channel push with
   e_lr = −adaptive(c) − wd·θ; A0Q extended the same adjoint machinery to clip/wd
   channels with one shared pass.
3. **AdamW augmented-state tangent predicts local perturbation response.** Model C
   (complete stepwise tangent + clip-JVP curvature feedback) explains H≤8
   (H>1 mean rel err 0.0219 at registered ε); H=1 is the exact local Jacobian.
   No memory-kernel term was needed at these horizons (A0C conclusion, still
   standing after A0P).
4. **LR is the strongest — in fact sole *stable* — optimizer control channel.**
   A0Q: clip is real but event-driven (never binds in 7/12 cells); wd is 4 orders
   of magnitude below LR and FD-noise-dominated.
5. **Response magnitude contains short-horizon controllability information.**
   A0S PASS: ｜χ｜ predicts realized intervention gain (Spearman +0.762, LOSO
   +0.825); sign predicts direction (46/48 + 12/12); A0T: controllability
   magnitude terciles predict controller success monotonically.
6. **Closed-loop injection at frozen cadence yields small positive short-window
   gains.** A0K +7.44e-4 nats equal-step; A0G task-level utility; A0H-2.0 A″
   +1.467e-4 (point estimate). Equal-wall-clock scale condition holds at 5M.
7. **The response decomposes into a large common mode + small differential, and
   the common mode carries the phase structure** (A0M: ｜mean｜/std ≈ 1.85;
   ~8–10× decay early→late; argmin identity phase-dependent in 8/9 series).
8. **Response sign is a stream-dependent, state-dependent object** (A0T: 2/18
   states have opposite-sign χ across the two frozen streams) — it is a property
   of (state, future data), not of the state alone.

### Level 2 — Negative results and boundaries (what did NOT work, and why it matters)

| # | Failure | Boundary it draws | Why it is scientifically valuable |
|---|---|---|---|
| N1 | **Scalar/aggregate response surrogates mislead.** Window-averaged response degrades utility (A0M, w=1 optimal); cosine similarity coexists with decision flips (A0J, cos=0.955 with 5/6→1/6 preservation collapse); pooled scalar statistics hide seed heterogeneity (A0I-A). | Any control law consuming smoothed/cosine-level response statistics is unsound; decisions live at the margin scale (3.8e-5), not the norm scale. | Sets the precision contract for the whole field: **response methods must be evaluated at the decision level, not the vector-similarity level.** |
| N2 | **Scalar FDR-style closure does not hold.** The early hypothesis that a scalar fluctuation-response readout would suffice collapsed: the response needs the full augmented state (θ,m,v) and curvature feedback (A0C ladder); scalar sufficient statistics of the optimizer state do not predict it (A0J cache failure). | No cheap sufficient statistic of (θ,m,v) is known that predicts the response at decision precision. | Local linear-response theory *works* — but only with the full tangent propagator. This is the precise sense in which "fluctuation-response for transformers" is true and false. |
| N3 | **Response computation has no wall-clock advantage at 5M.** A0G (shared tangent slower than replay), A0I-B (marginal cost −30% but total SIMILAR), A0K ROI ~5e-5 nats/s. | At small scale, response-based selection is economically dominated by direct candidate replay. | Reframes the research question from "cheaper" to "only-possible-when-replay-is-impossible" (large models, where replay cost explodes). |
| N4 | **The selector adds nothing over a fixed action.** A0K M5 (−1.29e-4), A0I-A (uncertain), A0L (B≥C), A0N (B≥C). | On tested banks, adaptive selection ∈ {0, negative} net of fixed-intervention baseline. | Four independent experiments converge — this is a robust boundary, not a one-off. |
| N5 | **Data-mixture controller fails against trivial fixed mixture** (A0L). | Continuous control of data composition via response adds no value over injecting the empirically best fixed group. | Kills the "continuous controller" branch at pilot scale. |
| N6 | **Phase-aware ρ-gating fails against blind fixed injection** (A0N). | Observable in-loop phase structure (A0M, reproduced) is not exploitable by threshold-gating policies. | Separates "structure exists" (A0M CASE A) from "structure is usable" (A0N CASE B) — a distinction the field often conflates. |
| N7 | **Response-guided controller AND schedule both collapse to fixed LR-up scaling.** A0T-extra (entropy 0.31 bits, adaptive≈0), A0U (C ≈ B delayed 6 steps; AUC significantly worse p=1.2e-2; the frozen K=4 rule never even materialized its averaging window). | On single-sign banks, the *entire* exploitable information content of the response is one bit (the sign). | Identifies the exact missing ingredient (sign diversity / horizon-persistent reversal) any future adaptive method must first establish. |
| N8 | **No temporal law: impulse-kernel gate FAIL** (A0P). Pooled Pearson +0.7314 < 0.8; fast initial decay + long noisy non-decaying tail; no exponential or power-law timescale. | First-order response is a **finite-horizon** object; there is no stable kernel for long-lag prediction, and short-horizon gains must never be extrapolated to final training. | The single most important guardrail in the project (A0S/A0T mandatory caveat originates here). |
| N9 | **State-dependent reversal does not persist beyond the response horizon** (A0T-extra: H=4 win inverts at H=16/64). | Adaptive claims survive only within the horizon the response was computed at. | Bounds the validity radius of every response-based policy tested so far. |
| N10 | **Cheap sensitivity caching is invalid** (A0J): margin (3.8e-5) is 2–3 orders below drift of any age≥1 surrogate. | Response freshness is not an engineering luxury; it is structurally required at this cadence. | Closes the "amortize the adjoint" branch; redirects compression research toward state representations rather than sensitivity caching. |

### Level 3 — Open questions

1. **Does response predict optimization improvement, or only local
   controllability?** Every positive gain so far is a short-window rollout gain on a
   mid-training segment. A0P says the kernel does not extend; nothing yet connects
   response sign/magnitude to final-training loss. *Missing proof: a response-selected
   intervention whose benefit survives to end-of-training.*
2. **Can the expensive response be compressed into a cheap state representation?**
   A0J ruled out caching the *sensitivity*; the complementary question — is there a
   low-dimensional function of (θ, m, v, trajectory statistics) that *predicts* the
   response at decision precision — is untouched. *Missing proof: any surrogate with
   decision-level (not cosine-level) equivalence.*
3. **Is there a theoretically justified control law?** All gains decompose as
   "fixed LR-up discovery" + tiny non-persistent reversals (A0T-extra, A0U). A control
   law derived from the A0O θ-channel theory (e.g., magnitude-proportional steps from
   analytic h_eta) has never been tested; only sign rules were. *Missing proof: a
   theory-derived law that beats its own fixed-action ablation at equal wall-clock.*
4. **What breaks the single-sign degeneracy?** 11/12 A0U trajectories are all-negative;
   the one positive-χ cell was "confidently wrong" at long horizon. Are there training
   regimes (late phase, different data, larger models with different clip regimes —
   G1a/G1b clip always binds) where χ sign is diverse and persistent? *Missing proof:
   a bank/stage with multi-sign, horizon-persistent response.*
5. **Does any control result transfer to scale?** Mechanism (A/B/C ladder) replicated
   at 26M and 100M (G1a/G1b) and across data distributions (G2), but **no control
   experiment has ever been run above 5M**. *Missing proof: one A0T-style pilot on a
   G-line bank.*

---

## Part 3 — The Physics Connection: What Corresponds and What Does Not

### 3.1 Valid mathematical correspondence

The correspondence that **survived contact with experiment** is the **controlled
dynamical systems / tangent-linear** one:

| Training object | Dynamical-systems object | Where proven |
|---|---|---|
| Training step z_{t+1} = F(z_t, u_t) with z=(θ,m,v), u=(η, τ, λ, data) | Controlled nonlinear map; AdamW update + clip is the frozen F | A0C model ladder; A0Q channel map |
| First-order response h = ⟨v_H, S⟩, S = Σ_k T_k e_u(k) | Tangent-linear propagator Φ = T_H…T_1 applied to input matrix column e_u; **R = CΦB** with B = pulse channel, C = val-loss gradient readout | A0O (LR: exact e_lr; m/v lr-independent), A0Q (clip: dc-channel with J_clip transpose; wd: θ-channel), A0H (adjoint = Φᵀ, exactly) |
| Non-equilibrium | Driven, dissipative trajectory with no stationary measure; response measured *along* the driven path, not near a fixed point | A0M (response drifts with phase; common mode decays 8–10×; no time-translation invariance — cos@lag ≈ 0.629 plateau, lag-Spearman ≈ 0) |
| Controllability | Which input channels move the readout, and does ｜response｜ predict realized gain | A0S (PASS, H=4), A0T (terciles monotone), A0Q (channel inventory) |
| Memory kernel (Mori–Zwanzig) | Residual of projection onto local tangent; would appear as a long-lag kernel | **Tested and not needed at H≤8** (A0C: C explains response); **short→long bridge FAILS** (A0P: no stable decay law; noisy tail ｜R｜~1.5–3.9e-3) — an *open* boundary, not an established kernel |
| Coarse graining | Reducing (θ,m,v,batch) to a response readout or a policy input | Works at norm level (A0M common mode), **fails at decision level** (A0J margin vs noise) |

The honest summary: the project is a **rigorous instance of tangent-linear
(response) theory for a specific non-equilibrium system**, with the adjoint,
controllability coordinates, and temporal-kernel hypotheses all *experimentally
resolved* (positive or negative).

### 3.2 Invalid or unsupported interpretations

The following analogies must **not** be claimed, because nothing in the data
supports them and they have produced zero algorithmic content:

| Tempting analogy | Why it fails here |
|---|---|
| m = "momentum particle" | m is an EMA of clipped gradients — a filter state, not a physical momentum carrying inertia between collisions. No conservation law, no reversibility. |
| v = "temperature" / "noise strength" | v is the second-moment EMA of the gradient, entered through a *deterministic* bias-corrected quotient m̂/√v̂. There is no fluctuation statistics, no Einstein relation, no equipartition. v responds to intervention deterministically (A0Q: wd channel negligible; LR response is a pure θ-channel push — m/v are exactly lr-independent). |
| loss = "free energy" | Loss is a readout functional of one trajectory, not a thermodynamic potential. There is no variational principle, no landscape measure. The project never needed one: all predictions come from the tangent propagator. |
| learning rate = "temperature" | LR is a **control input u**, and the whole O→U line shows it behaves like one: persistent u compounds (A0T gains grow with H), response to u is a directional derivative (A0O), and a 5% schedule change is a *policy*, not a thermostat. |
| "response exists ⇒ better optimizer" | Refuted in four rounds: A0K M5, A0L, A0N, A0T-extra, A0U. Response existence is a *measurement* fact; optimization improvement requires information the response does not (currently) contain beyond one sign bit. |

Discipline note: the project's every algorithmic gain came from **tangent linear
systems theory** (adjoints, controllability, persistent inputs). Every appeal to
thermodynamic language would have been decoration; the two negative-kernel results
(A0P, A0J) are precisely where a *thermodynamic* intuition would have predicted
smooth decay and reuse — and was wrong.

---

## Part 4 — Where the Project Stands Now

> Draft statement: *"Transformer training is an observable non-equilibrium
> dynamical system with measurable finite-time input-response structure. However,
> converting response information into superior optimization policy remains
> unresolved."*

**Evaluation: accurate, and it can be made sharper.** After A0U the strongest
scientifically defensible statement is:

1. **Observable non-equilibrium system — YES, with a channel inventory.** The
   system is observable through at least one clean input channel (LR: A0O PASS at
   n=96, theory-exact), one event-driven channel (clip: A0Q), and one dead channel
   (wd). The observation operator (frozen adjoint + FD cross-check) is validated
   to Pearson ≥0.84 on every channel where a signal exists.
2. **Finite-time input-response structure — YES, bounded in horizon.** Structure is
   real and computable at H≤8 (A0C) and for persistent inputs over 64 steps (A0T,
   A0U), but **no temporal law extends it** (A0P gate FAIL). "Finite-time" is not a
   temporary limitation of the method; it is the current measured boundary of the
   phenomenon.
3. **"Converting response information into superior optimization policy remains
   unresolved" — correct but should be strengthened to: on every bank tested, the
   conversion is *resolved negatively against adaptivity***. Response-guided
   selection, continuous mixture control, phase-gated scheduling, and open-loop
   schedule discovery all fail to beat their fixed-action ablations (N4–N7 above).
   The one component that works is the *intervention itself* (A0K M1).
4. **Add a fourth clause the draft omits:** *the exploitable information content of
   the response on single-sign banks is one bit* (A0U Stage 1: cross-seed sign
   agreement 1.0, magnitude trajectories irreproducible). Any future policy claim
   must first demonstrate a setting where the response carries more than its sign.

---

## Part 5 — Research Directions

Each branch: hypothesis / supporting evidence / main risk / minimal experiment.
Branches are **not ranked by intuition**; ordering below follows dependency logic
(each branch's missing proof is stated).

### Branch A — Theory-derived LR control law (closes N7)

- **Core hypothesis:** a control law derived from the A0O θ-channel theory —
  u_t proportional to the *analytic* h_eta (magnitude, not just sign), with step
  size justified by the tangent model — beats both native and its own best-fixed
  ablation at equal wall-clock.
- **Supporting evidence:** A0O (exact e_lr, h_eta computable in one adjoint pass);
  A0S (magnitude predicts gain, β=0.0449); A0T (sign rule already beats native);
  A0T-extra/A0U (sign-only is provably insufficient → magnitude is the untested
  degree of freedom).
- **Main risk:** single-sign degeneracy makes any monotone law ≈ fixed scaling
  again (A0U); magnitude-proportional steps may destabilize at long H (A0T
  erosion, A0P tail).
- **Minimal experiment:** on the 2 A0T non-aliased cells + 2 fresh positive-χ
  cells (requires new rollouts, ~1 GPU-hour budget): frozen u = −clip(κ·h_eta) law,
  κ fixed pre-registration, vs arms A/B(sign)/B(magnitude-matched fixed). Missing
  proof supplied if: magnitude arm beats sign arm and its fixed ablation.

### Branch B — Controllability analysis paper (synthesis, lowest new-evidence cost)

- **Core hypothesis:** transformer training admits a causal input→response map
  with a measurable controllability coordinate; this map and its boundaries
  (channel inventory, horizon boundary, compression boundary) constitute a
  publishable characterization.
- **Supporting evidence:** A0O/A0Q (map), A0S (controllability coordinate), A0M
  (phase structure), A0P/A0J (boundaries), A0T/A0U (policy-level collapse).
- **Main risk:** descriptive paper; needs a "why care" — supplied by the negative
  results' precision (they falsify plausible published intuitions: kernel reuse,
  phase-gated scheduling, adaptive-vs-fixed claims).
- **Minimal experiment:** none required for the synthesis itself; one strengthening
  axis recommended: A0Q-style channel map on a G-line bank (26M, where clip always
  binds — G1a) to show the channel inventory is model-dependent. Missing proof
  supplied by: clip-channel PASS at 26M where the v07 bank showed event-driven
  failure.

### Branch C — Better control variable search (closes the thin-inventory half of N7)

- **Core hypothesis:** channels beyond (LR, clip, wd) — momentum β, Adam ε, batch
  composition structure, grad-norm scale — contain *independent, sign-diverse*
  controllability not spanned by the LR direction.
- **Supporting evidence:** A0Q methodology generalizes (one shared pass, per-channel
  gates); G-line banks show different clip regimes (26M: clip always active) — the
  channel inventory is demonstrably not universal; A0M shows argmin identity is
  phase-dependent (multiple candidate directions carry signal).
- **Main risk:** most channels will be event-driven like clip; sign diversity may
  still not appear (the bank, not the channel, may be the bottleneck).
- **Minimal experiment:** A0Q-style map with 2 added channels (β1 multiplicative,
  batch-size-scaled gradient) on 12 v07 cells + 6 G1a cells. Missing proof: a
  channel with sign-diverse, gate-PASS response.

### Branch D — Non-equilibrium coarse-grained state (closes N2/N10)

- **Core hypothesis:** there exists a computationally cheap statistic ψ(state,
  trajectory) — not a cached sensitivity — that predicts the response at decision
  precision, enabling control when the adjoint is unaffordable (large models).
- **Supporting evidence:** A0J defines the negative boundary precisely (margin
  3.8e-5 vs surrogate drift 2e-3); A0M shows the common-mode component is smooth
  and phase-structured (the predictable part); A0S shows a linear map from ｜χ｜ to
  gain exists.
- **Main risk:** may be *provably impossible* at decision precision (the
  differential component may be sensitive to the full optimizer state — A0J's
  result is consistent with this); a partial surrogate predicts the common mode
  but not the differential, which is exactly the part selection needs.
- **Minimal experiment:** offline, zero GPU: regress recorded responses (A0K/A0O
  banks, n≈200+) on hand-crafted ψ (norms, moments, phase labels); **evaluate at
  decision level** (top-1/top-2 margin prediction), not cosine. Missing proof: any
  ψ with margin-level predictive power; if none, a certified impossibility argument
  at fixed horizon would itself close the branch.

### Branch E — Scale transfer of control (closes the external-validity gap)

- **Core hypothesis:** response-guided LR control gains replicate — and the
  sign-degeneracy breaks — at 26M/100M, where training dynamics (clip regime,
  curvature scale, HVP norm ~7× larger per G2 diagnostics) differ qualitatively.
- **Supporting evidence:** mechanism (A/B/C) replicated at 26M/100M (G1a/G1b) and
  across data (G2); A0T/A0U pipelines are model-agnostic by construction; G1a
  diagnostics show clip *always* active at 26M (channel inventory changes with
  scale).
- **Main risk:** cost (26M adjoint rollouts are ~5× slower; budget discipline
  needed); gains may shrink with model size; banks at 26M were built for mechanism
  tests, not control (no pre-registered event streams for control).
- **Minimal experiment:** A0T-style 12-cell pilot on the G1a 26M bank at one stage,
  frozen sign rule, arms A/B/C, H∈{4,16}. Missing proof: any control result at
  >5M, and observation of whether χ-sign diversity increases with scale.

**Dependency note.** Branch A's missing proof (magnitude law) is a prerequisite for
any honest "adaptive control" claim at any scale; Branch E without Branch A would
just re-measure the sign-degeneracy at higher cost. Branches B and D are
independent and can proceed in parallel offline.

---

## Part 6 — Potential Paper Positioning

### 6.1 Analysis paper — "Finite-time input-response structure of transformer training"

- **Novelty:** first *causal, adjoint-validated* input→response map of optimizer
  channels in transformer training: exact LR channel theory (pure θ-channel push),
  event-driven clip channel, dead wd channel; a validated controllability
  coordinate; measured phase structure; and three precisely located boundaries
  (horizon — A0P; compression — A0J; adaptivity — A0K/L/N/T-extra/U).
- **Required additional evidence:** (i) one multi-scale point (Branch E-lite:
  A0Q channel map at 26M) to show the inventory is not 5M-specific; (ii) a
  consolidated decision-level evaluation appendix (margins, not cosines). Both are
  cheap relative to what exists.
- **Risk:** negative-result papers face framing resistance. Mitigation: the paper's
  positive core (A0O theory + A0S controllability) is strong; the negatives are
  *boundary theorems* with falsification value (they kill published-plausible
  approaches). Risk of being scooped on the tangent part is low — the adjoint-grade
  rigor (CRN, bitwise replay gates, frozen contracts) is the moat.
- **Verdict: viable now**, pending the two strengthening items.

### 6.2 Algorithm paper — "Response-guided control of training"

- **Novelty claim a paper would want:** adaptive control beats fixed policy.
- **Current evidence against the claim:** A0K M5, A0I-A, A0L, A0N, A0T-extra,
  A0U — six rounds, all ≤ fixed ablation. As of 2026-09-18 there is **no setting**
  in which a response-guided adaptive policy beats its fixed-action control at
  equal wall-clock.
- **What additional evidence would be required:** Branch A's minimal experiment
  (magnitude law beating sign rule *and* fixed) at n≥12 cells with CRN — or a
  sign-diverse bank (Branch C/E) where even the sign rule separates from fixed.
- **Risk of writing it now:** high — it would be a paper whose own ablations defeat
  its thesis. The honest algorithmic statement today is "persistent LR-up
  intervention helps short-term at 5M; we do not know why beyond the sign, and it
  does not survive our own fixed-action control."
- **Verdict: not viable now.** The shortest path runs through Branch A → then E.

---

## Part 7 — Current Understanding in One Page

**What we know.**
Transformer training under AdamW is an observable controlled dynamical system.
Pulses to the data gradient and persistent inputs to the LR are **causal, linear-
response-consistent observables**: the full augmented-state tangent (with clip-JVP
curvature feedback) predicts data-pulse response to R²≈0.998 at H≤8, and the LR
channel has an exact θ-channel theory validated at Pearson +0.955 (n=96). LR is
the sole stable optimizer control channel (clip event-driven, wd 4 orders down).
Response magnitude is a short-horizon controllability coordinate (Spearman +0.762;
LOSO +0.825) that predicts realized gain and controller success. The response
decomposes into a large, phase-decaying common mode and a below-noise differential;
its sign is a stable seed-general but stream-dependent property. Closed-loop
persistent injection at frozen cadence is net-positive short-term (+7.4e-4 nats)
and survives equal-wall-clock comparison at 5M.

**What we disproved.**
Scalar/aggregate surrogates at decision level (window-averaging, cosine — they
flip decisions while looking faithful). Cheap sensitivity caching (margins are 2–3
orders below surrogate drift). Wall-clock advantage of tangent/adjoint methods over
direct replay at 5M. Adaptive selection over fixed action — six rounds, zero wins.
Continuous data-mixture control over fixed mixture. Phase-gated scheduling over
blind injection. Any temporal law extending response beyond its horizon (impulse
kernel gate FAIL; no exp/power-law timescale). Horizon-persistent state-dependent
reversal (H=4 wins invert by H=16). Thermodynamic readings of optimizer state
(no fluctuation statistics, no Einstein relation; every gain came from tangent
linear systems theory).

**What is promising.**
(1) Theory-derived magnitude control (Branch A) — the one untested degree of
freedom, with exact theory and a validated magnitude→gain map already in hand.
(2) Sign-diverse regimes (Branch C/E) — clip regimes and curvature scales change
with model size; the single-sign degeneracy that sank A0U may be a small-model,
mid-phase artifact. (3) A controllability analysis paper (Branch B) — most of the
evidence exists and the negative boundaries have real falsification value.
(4) Coarse-grained state prediction (Branch D) — A0J precisely bounded sensitivity
caching; state-side compression is unexplored.

**What should not be attempted again.**
Sign-only LR rules on single-sign banks (they re-discover fixed scaling; A0U
proved the information content is one bit). Kernel/timescale fits to impulse
responses (A0P gate failed; do not relaunch with relaxed thresholds). Cache/reuse
schemes for response vectors at K≥10 cadence (A0J). Adaptive-vs-fixed claims
without a fixed-action control arm (six rounds of evidence). Any "final training
improvement" claim from short-window rollout gains (A0P/A0S caveat is mandatory).

**The next 3 research questions.**
1. **Does a magnitude-derived LR control law (u ∝ analytic h_eta) beat both the
   sign rule and its fixed-action ablation?** (Branch A; closes the adaptivity
   question either way; ~1 GPU-hour + pre-registration.)
2. **Is the χ-sign degeneracy a property of the bank or of the system?** — an A0Q+
   A0T replication at 26M (G1a bank, clip always active) measuring sign diversity,
   channel inventory, and one control pilot. (Branch B+E; decides whether the whole
   control line scales or the boundary is fundamental.)
3. **Can any cheap state statistic predict the response at decision precision
   (margin-level), where caching the sensitivity failed?** (Branch D; offline;
   would convert the response from an expensive oracle into a usable signal for
   large models — or certify that it cannot be done.)

---
*Compiled 2026-09-18 from frozen artifacts only: FR_PROJECT_STATUS.md §17–§45,
FR_EXPERIMENT_INDEX.md, and per-experiment reports under outputs/. No new
computations were run for this retrospective. Items marked [summary-level] were not
independently re-verified against raw cell records.*
