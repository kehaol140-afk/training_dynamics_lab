# A0C AdamW m+v Strict-Consistency Design

This audit changes only experimental alignment relative to
`outputs/a0_fr_audit/mv_theory_final`.

- Target rows remain the 448 rows in
  `outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv`.
- For each seed/state, q is loaded from the exact `q_path` recorded in the v07
  `theta_mapping_raw.json`; its internal direction hash and seed/state metadata
  must validate.
- The first-step gradient is the already-saved eight-E2-stream arithmetic mean
  in `outputs/a0_fr_audit/mv_theory_final/gradient_g.pt`. No forward, backward,
  clipping, optimizer step, or training is rerun.
- Raw checkpoint `exp_avg` and `exp_avg_sq` provide m and v. Parameter names and
  shapes must match q and g exactly.
- Predictions omit perturbation epsilon because A0C
  `chi=mean[(X_plus-X_minus)/(2*epsilon)]` is a unit-perturbation response.
- No theory term, fit parameter, bias correction, calibration, checkpoint, or
  prior output is changed.

Outputs are confined to
`outputs/a0_fr_audit/mv_theory_consistent/{summary.json,A0C_MV_CONSISTENT_REPORT.md}`.

