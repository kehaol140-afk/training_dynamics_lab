# muon_transition — RESULTS

## Research question

Frozen question (protocol v1.1): under identical model, token budget, data domains and a
fair equal-size hyperparameter search, does changing the optimizer from AdamW to Muon
change the optimal training token mixture p between two domains — i.e. is
p\*_AdamW ≠ p\*_Muon? Negative results valid and terminal. No new algorithms; Muon not
assumed better.
Evidence: `protocols/MUON_PROTOCOL_v1.1.md`

## Stages executed

| Stage | Content | Verdict |
|---|---|---|
| M0 gates 1–4 | Muon source freeze (installed `_muon.py` SHA-audited), routing, data audit, pipeline audit | 4/4 PASS |
| M1-0 smoke | pipeline + eval on 1 cell | OK |
| M1 (12 runs) | p=0.5, seed 2030, B0=2^26, 2×5090, ~10–13 min/run; gate = 6/6 stable cells | PASS; best dev V_Q AdamW 2.6412 (lr 6e-4/wd0.01) vs Muon 2.3512 (lr 0.01/wd0.02), both at lr_mult 2.0 boundary |
| M2 (36 new + 6 m1 + 6 extension cells) | p∈{0.25,0.5,0.75} full grid + one-time boundary extension (mult=4.0) per frozen v1.1 §5 | 48 training cells + reused m1 rows, all ok=True |
| M2 readout + stop rule | p\* selection, migration cost R, stop rule | **`NO_SIGNAL_STOP_LINE`**; M3 (12 confirmation runs) NOT entered |

## Key numbers (from `results/outputs/m2/M2_SUMMARY.md`)

Best V_Q per arm × p:

| arm | p=0.25 | p=0.5 | p=0.75 |
|---|---|---|---|
| AdamW | 2.6650 | **2.6262** | 2.7041 |
| Muon | 2.3715 | **2.3437** | 2.3887 |

- p\*_AdamW = 0.5, p\*_Muon = 0.5 (identical); R = Muon(p\*_AdamW) − Muon(p\*_Muon) = **0.0000**;
  symmetric diagnostic R^AdamW = 0.0000.
- Stop-rule detail: adv_muon = 0.0000 vs proxy_muon 0.1539; adv_adamw = 0.0000 vs 0.8652.

## Negative results and limitations

- LR search resolution: 4/6 (arm,p) optima at the mult=4.0 extension boundary
  (LR_RANGE_UNRESOLVED) — a tuning-resolution limitation that **does not threaten the
  equal-p\* conclusion** (both arms share the same grid; the p-comparison is within-arm).
- Single seed (2030), single model scale (~30M MiniMind), two-domain synthetic split;
  conclusions are protocol-bound (2026-10-08 review §0.5 re-verified from raw result.json).

## Not executed

M3 confirmation runs (blocked by the stop rule — negative is terminal); no other stages
were preregistered.

## Large artifacts not stored

`outputs/m1/*/final_state.pt` + `outputs/m2/*/final_state.pt` (~49 files, ~15 GB,
356 MB each) remain on the server; per-run small metrics and both summary tables are
archived here. Anchor state for the sibling muon_temporal line
(`m1_muon_lr2.0_wd0.02_p0.5_s2030/final_state.pt`) is among them and was still referenced
by that line's T2* configs at archive time.

## Original evidence paths

- Root: `/home/b205/projects/training_dynamics_lab/muon_transition_experiment/`
- `outputs/m0/M0_REPORT.md`; `outputs/m1/m1_summary.json`; `outputs/m2/M2_SUMMARY.md`;
  `scripts/m2_readout.py`; protocols in `docs/`.
