# muon_transition — Optimizer-Dependent Data Mixture Interaction (archived)

**Status: COMPLETED 2026-09-27, terminal negative (`NO_SIGNAL_STOP_LINE`); M3 not entered per frozen stop rule.**
Independently re-verified 2026-10-08 (`audits/muon_history_review_20261008_121303UTC/`):
36 raw `result.json` cells re-parsed, matching M2_SUMMARY cell-for-cell.

## One-line result

Under identical model/token budget/domains and a fair equal-size LR×WD search, switching
AdamW → Muon does **not** change the optimal two-domain token mixture:
**p\*_AdamW = p\*_Muon = 0.5**, migration cost R = 0.0000. (Muon's absolute advantage in
V_Q — 2.3437 vs 2.6262 — is a known separate fact and irrelevant to this question.)

## Layout

```
archived/muon_transition/
├── README.md  RESULTS.md
├── code/            src/muon_transition + scripts + tests (6 test modules)
├── configs/         frozen generated configs
├── protocols/       MUON_PROTOCOL v1.1/v1.2, M2 addendum, dataset card, provenance,
│                    checklists, handoff audit, project README
└── results/outputs/ all 224 non-checkpoint files (m0 gates; m1/m2 summary.json,
                      M1/M2_SUMMARY.md, per-run metrics; smoke; logs)
```

Not stored: ~49 × `final_state.pt` (~15 GB) — see `RESULTS.md`.

Source: `/home/b205/projects/training_dynamics_lab/muon_transition_experiment/` (read-only copy).
