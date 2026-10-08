# muon_temporal_response — Muon Map Temporal Response (archived, bounded-completion status)

**Status: three preregistered stages COMPLETE with frozen verdicts (T0, T2a, T2b);
the full T2 replay program and T3/T4 were NEVER STARTED. The line is archived as a
status report — no terminal claim is made beyond the per-stage verdicts.**
Re-verified 2026-10-08 by `audits/muon_history_review_20261008_121303UTC/` (§0.4).

## Layout

```
archived/muon_temporal_response/
├── README.md  RESULTS.md
├── code/            src (t0_construct/t0_maps/t0_metrics/t2b_common) + scripts + tests
├── configs/         frozen stage configs (t0_round1, t2a_scale_v1, t2b_target_screen_v1)
├── protocols/       all 12 docs (protocols, reports, errata, review indexes)
├── vendor_snapshot/ installed-Muon source snapshot (SHA-verified, read-only)
├── audit/ references/  handoff materials
└── results/outputs/ 50 small evidence files (t0_round1, t0_round1_corrected,
                      t2a_scale_v1, t2b_target_screen_v1)
```

Not stored: `raw_natural.pt` (627 MB) and `raw_sample_l1q.pt` (176 MB) — see RESULTS.

Source: `/home/b205/projects/training_dynamics_lab/muon_temporal_response/` (read-only copy).
Review bundles ROUND1/2/3 (.zip) preserved in `protocols/` copies of the review indexes;
the zips stay on the server.
