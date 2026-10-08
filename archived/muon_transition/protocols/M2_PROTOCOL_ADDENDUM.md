# M2_PROTOCOL_ADDENDUM — M1.5 patch (LR boundary rule + M2 execution spec)

Date frozen: 2026-09-26, after M1 completion, before any M2 run.
Nature: **addendum to MUON_PROTOCOL_v1.1/v1.2** — does not modify them; supplements M2 execution.
Trigger context: M1 (p=0.5) best cells landed at multiplier 2.0 for **both** arms
(`outputs/m1/M1_SUMMARY.md`), i.e. on the search-grid boundary.

---

## 1. LR boundary rule (frozen)

1. Base LR multiplier grid: **{0.5, 1, 2}** (unchanged; reference LRs and WD grid per v1.2).
2. For any (optimizer × mixture p) cell: if the best cell (lowest dev V_Q; tie-break lower LR,
   then lower WD) has multiplier **2**, **one** extension to multiplier **4** is allowed — at both
   WD values (2 extra runs for that optimizer × p).
3. If, after the single extension, the best cell is at multiplier **4**, the entry is recorded as
   **LR_RANGE_UNRESOLVED** for that (optimizer × p). **No further extension is permitted** —
   neither now nor after seeing the extended results, for any p, for either optimizer.
4. Extensions are justified only by base-grid results (never by extended-grid results) and are
   logged with their justification in the run config. The total extension budget is exactly one
   round.
5. Already-justified extensions at freeze time (from M1, p=0.5): AdamW mult 4 (LR 1.2e-3) ×
   WD {0.01, 0.02}; Muon mult 4 (LR 0.02) × WD {0.01, 0.02} — 4 cells.

## 2. M2 execution spec (frozen)

- Factors: optimizer ∈ {AdamW, Muon} × p ∈ {0.25, 0.50, 0.75}; per combination search
  LR mult {0.5, 1, 2} (+4 under §1) × WD {0.01, 0.02}.
- p=0.5 cells come from M1 (12 runs, seed 2030); p=0.25/0.75 are new (24 base runs + conditional
  extensions), same seed 2030, same data pipeline (thread-split, dedup, packing, exact-p
  manifests), same model/tokenizer/budget/eval as v1.2. Realized p_hat exact per step.
- All runs: B0 = 2^26 tokens, 2048 steps, identical schedule, clip 1.0, observation hooks every
  100 steps (diagnostic only).
- Selection surface: **dev V_Q only** (`V_Q = 0.5·NLL_A + 0.5·NLL_B`, q=0.5 frozen). Confirm pools
  remain sealed (M3 only).

## 3. M2 readout (frozen)

1. Per (optimizer × p) search table: dev V_Q (with NLL_A / NLL_B) for every cell.
2. Mixture–NLL curve: best dev V_Q vs p, per optimizer.
3. Selected mixtures: **p*_AdamW**, **p*_Muon** = argmin over p of best dev V_Q
   (tie-break: p=0.5, then lower p; logged).
4. Migration cost: **R = Muon(p*_AdamW) − Muon(p*_Muon)**, where Muon(·) is Muon's best cell at
   that mixture (its own tuned HP at that p). Symmetric diagnostic: R^AdamW = AdamW(p*_Muon) −
   AdamW(p*_AdamW).
5. Observation logs retained per run (gradient cosine, conflict ratio, update statistics);
   diagnostic only.

## 4. Stop rule (frozen)

M2 answers **only**: does an optimizer-dependent mixture signal exist?

- **No signal** if p*_AdamW = p*_Muon (grid-identical), or if the p* difference is one grid step
  but the favored-mixture advantage of each optimizer is within its own across-WD/parity spread
  (operationally: |ΔV_Q at own p* vs other's p*| ≤ max same-p cell spread for that optimizer —
  a deliberately conservative noise proxy, since M2 is single-seed). → **STOP the line**; negative
  result is valid and terminal.
- **Signal** if p*_AdamW ≠ p*_Muon and each optimizer's advantage exceeds the proxy above. →
  **Await M3 confirmation** (3 new seeds, no retuning; noise threshold frozen in the M3 config
  before any M3 run).

No mechanism interpretation, no new methods, no additional mixtures, no post-hoc re-search.
