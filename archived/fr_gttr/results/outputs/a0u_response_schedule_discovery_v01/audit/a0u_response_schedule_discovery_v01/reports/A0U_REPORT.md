# A0U Response-Guided Schedule Discovery Pilot — Final Report

**Date:** 2026-09-18
**Output:** `outputs/a0u_response_schedule_discovery_v01/`
**Verdict: CASE B — the response-guided schedule is functionally equivalent to (in fact, a slightly delayed version of) fixed +5% LR scaling. The response contributes one bit of schedule information ("LR up"); it does not reveal exploitable phase-dependent structure beyond that.**

---

## 1. What was tested

Can LR response dynamics reveal a better TRAINING SCHEDULE than fixed hand-designed
LR rules? Response was used as an OFFLINE schedule-discovery signal (this is **not**
another online controller — the A0T-extra lesson). Frozen design:

- **Response bank (read-only):** frozen A0O stage1 `h_eta` (H=4), 12 trajectories =
  3 seeds {2030, 2031, 2037} × 4 streams {7110..7113}, knots at bank steps (46, 52, 64, 82).
- **Frozen schedule rule:** per knot, `R_avg` = mean of χ over previous knots (up to K=4,
  partial windows; empty → keep native); `R_avg < 0 → η·e^{+0.05}`, `> 0 → η·e^{−0.05}`.
  Knots define segment multipliers over a W=64-step rollout from bank state step 46
  (window steps 47–110): offsets 1–6 native, 7–18 `m(52)`, 19–36 `m(64)`, 37–64 `m(82)`.
- **Comparison (12 cells, all from step 46):** A native / B constant ×1.05 / C response
  schedule; val-loss checkpoints at bank steps (58, 70, 82, 94, 110); CRN A×2 per cell
  (12/12 bitwise). One 128-micro-batch frozen-recipe stream per cell.

GPU: 475.5s / 1800s budget, single launch after one **pre-results** device fix
(cuda:1 hidden by env → cuda:0; foreign processes held GPU 1; manifest seq 2).

## 2. Stage 1 — does the response define training phases? (discovery ability)

Descriptive (no inferential gate, per A0P-negative precedent):

| finding | value |
|---|---|
| trajectories with χ < 0 at all 4 knots | 11/12 (the exception is s2030×7110, the known A0T cross-stream disagreement state) |
| median sign consistency within trajectory | 1.0 |
| phase \|χ\| medians (early/middle/late = {46}/{52,64}/{82}) | 0.0113 → 0.0088 → 0.0076 (late weaker than early in 9/12) |
| per-knot cross-seed SIGN agreement (streams 7111/7112/7113) | 1.0 at every knot; 7110: 0.67 at knots 46/64 |
| cross-seed Pearson of 4-knot magnitude trajectories | mostly weak: median ≈ 0.1, range −0.93 … +0.85 |

**Reading:** the response sign field is a *stable, seed-general phase property* (sign at
each knot agrees across seeds on 3/4 streams), and its magnitude decays descriptively
over the window. But the magnitude *trajectory shape* is seed-dependent — there is no
reproducible magnitude schedule to discover. The exploitable content is essentially the
sign — one bit.

## 3. Stage 2 — what schedules were discovered?

- **All 12 schedules switched exactly once** (native → one direction); none ever switched
  back. K=4 averaging never fully materialized (max window = 3 previous knots).
- 11/12 schedules = `[1.0, e^{+0.05}, e^{+0.05}, e^{+0.05}]` — keep native for the first
  6 optimizer steps, then LR-up for the remaining 58.
- 1/12 (s2030×7110) = `[1.0, e^{−0.05}×3]` — LR-down after step 52 (driven by χ(46) = +0.0274).
- `identical_to_B_frac = 0`: schedules are formally distinct from arm B (e^{0.05} ≈ 1.0513
  ≠ 1.05, and a 6-step native prefix).

## 4. Stage 3 — optimization performance (12 cells, paired)

| comparison (final loss @ step 110) | median | frac positive | Wilcoxon p |
|---|---|---|---|
| A − B (native − fixed+5%) | **+5.68e-3** | 12/12 | 4.9e-4 |
| A − C (native − response sched.) | **+4.87e-3** | 11/12 | 3.4e-3 |
| B − C (fixed − response sched.) | −3.3e-4 | 3/12 | 0.052 (n.s.) |

| comparison (loss-curve AUC) | median | frac positive | Wilcoxon p |
|---|---|---|---|
| A − B | +1.62e-1 | 12/12 | 4.9e-4 |
| A − C | +1.35e-1 | 11/12 | 1.6e-2 |
| B − C | −3.6e-2 | 2/12 | **1.2e-2 (C worse)** |

- **Early improvement** (offset 12, bank step 58): B = +7.0e-4 (9/12, p=0.034);
  C = +1.6e-4 (7/12, p=0.52). C forfeits most of the early gain — its first segment
  is native by construction (no previous knots at knot 46).
- **Late-stage stability** (std of losses @ {82, 94, 110}): A 2.60e-2 / B 2.71e-2 /
  C 2.68e-2 — no meaningful difference; slopes equal.
- The single LR-down cell (s2030×7110) is worse than BOTH A (+4.4e-3) and B (+1.2e-2):
  its schedule was confidently wrong, as A0P/A0T-extra would predict for a positive-χ
  stream.

**Mechanistic decomposition:** C ≈ B delayed by 6 optimizer steps. The measured B−C
final deficit (−3.3e-4) ≈ the cost of the frozen rule's empty-window prefix, and the
AUC deficit is exactly the forfeited early segment. Once the response sign is "LR up",
the K=4 rule adds nothing further — every subsequent R_avg pools mostly the same
negative knots and repeats the same decision.

## 5. Interpretation (frozen case rule)

**Case B — "response discovers a simple global scaling direction."**

- Case A fails: B − C is not positive (median −3.3e-4; significantly NEGATIVE on AUC).
- Case C fails: C − A is positive and significant in the same direction as B − A.
- This is the schedule-level mirror of the A0T-extra controller-identity verdict
  (dominantly global LR-up discovery): at the schedule level the response's entire
  exploitable content is the sign bit, and the frozen averaging rule's main effect was
  an inaction prefix, not phase-adaptive switching.

## 6. Practical significance and cost

- At this 64-step persistent window, gains are ~50–100× larger than A0S's H=4
  short-horizon scale (val loss ~7.07, gains ~5e-3) — persistence compounds, as in A0T.
- **Cost:** schedule generation 0.05s CPU (offline sign averaging) vs training
  475.5s GPU total (≈119s per arm). Discovery is essentially free; validation is the cost.
- Mandatory caveat (A0S FAIL-case-2 / A0P long memory): these are short-window rollout
  gains on a mid-training segment; they do NOT imply final-training improvement.

## 7. Limitations (pre-registered + observed)

1. **In-sample schedule:** χ at knot t is computed on the same stream the schedule is
   evaluated on (inherent to the frozen bank). No out-of-sample generalization claim.
2. **Sparse knots:** 4 knots per trajectory; K=4 never materialized (max window 3), so
   the "averaging" part of the rule was never exercised — only its sign and prefix logic.
3. **Single-bit effective information** on this bank (χ < 0 at 46 in 11/12 cells);
   schedules cannot be distinguished from fixed scaling when the bank has no sign diversity.
4. 12 cells, single start state (step 46); window gains are short-horizon (A0P).
5. Pre-results device fix (manifest seq 2) — no results existed before the fix.

## 8. Deliverables

- `contract/contract.json`, `contract/live_manifest.json` (seq 0–2)
- `research/src/fr/a0u_response_schedule_discovery_v01.py` (module)
- `research/scripts/a0u_run_v01.py`, `research/scripts/a0u_analyze_v01.py`
- `research/tests/test_a0u_response_schedule_discovery_v01.py` (7 tests, PASS)
- `runs/cells.jsonl` (12 records), `runs/summary.json`, `runs/driver.log`, `gpu_budget_log.json`
- `analysis/response_phase_analysis.csv`, `analysis/schedule_comparison.csv`,
  `analysis/analysis.json`, `analysis/plots/a0u_summary.png`
- audit pack under `audit/`

**STOP:** per contract stop-rule, no further controller/schedule rounds are started.
