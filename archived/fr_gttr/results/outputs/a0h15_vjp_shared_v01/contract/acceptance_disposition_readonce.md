# A0H-1.5 acceptance-batch disposition (read-once rule, protocol §1)

Protocol rule (verbatim): 一旦读取，不再调整实现或阈值后把同一批当未见数据；
若据其调试，整批降为开发数据并如实记录。

## What happened
- Acceptance set A = {dev:2032/58, dev:2035/70, dev:2037/46} × stream 0 was read in
  accept_A_run .. accept_A_run9 (see gpu_budget_log.txt).
- Those runs exposed a reconstruction bug in a0h15_accept_v01.py: hhat_fwd_C was
  reassembled from telescope terms under the corrected PRE-step x convention,
  which yields v_H^T θ_post@(H−1) − ĥ_adj instead of hhat_fwd_C; outputs showed
  fwd ≡ 0.0000000 (INVALID).
- The bug was debugged using observations from these very units (zero-fwd signature),
  and the fix (direct hhat_fwd_C emission from the pass) was validated on dev 2031/52.

## Disposition
- Per the read-once rule, ALL of batch A is DEMOTED TO DEV, effective
  2026-09-15T04:05Z. It must not be presented as unseen acceptance data in A0H-1.5
  or A0H-2.
- Consequence: Branch A independent acceptance on A = NOT_COVERED (invalidated by
  read-once consumption). Any rerun of these units is dev/diagnostic only and is
  labeled as such in outputs.
- The dev-set conclusions (D = 2031/52, 2034/64, 2036/76) are unaffected: D was
  frozen before acceptance reads and the diagnostics pass itself was not changed
  in any way that alters forward/adjoint values (only an added hhat_fwd_C output
  computed from existing quantities, mirroring frozen A0H-1 lines 330-335).

## Dev-validation rerun of demoted units
- accept_devval_v10.json (units 2032/58, 2035/70, 2037/46) is engineering
  validation of the fixed accept runner only. It carries NO acceptance status.

## Path forward for Branch A acceptance
- Requires a fresh unseen batch (new protocol amendment defining A′) — this is
  outside the current frozen contract, so Branch A auto-proceed to §7 timing is
  NOT triggered. §7 timing status: NOT_COVERED this round.
