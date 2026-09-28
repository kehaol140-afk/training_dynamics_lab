# G1a Change Log (implementation fixes; no scientific definition was altered)

All entries are pre-result or implementation-level fixes. No threshold, seed,
state grid, horizon grid, or protocol definition was changed after results were
opened. Verdict thresholds remain exactly as frozen in G1A_PROTOCOL.md §10.

1. **Param-count bookkeeping correction (before any result opened).**
   Preregistration estimate `expected_unique_params=26682048` was corrected to
   the measured `26681728` (instantiation of the frozen MiniMindConfig;
   hidden=512 / layers=7 / heads=8 / kv=4 / intermediate=1664 unchanged).
   Recorded in `research/configs/g1a_26m_v01.json` (`param_count_note`) and
   G1A_PROTOCOL.md §2. The runner asserts the measured count at train time.
2. **Config machine-readable key.** Added `training.train_batch_seed_offset=1704`
   (numeric form of the frozen `train_seed + 1704` v07 rule).
3. **Analysis-module typo.** `HORIZONS_OFFICIAL` misspelled inside
   `validate_g1a_e2_manifest` (would have raised NameError at first use).
4. **Runner pre-execution fixes** (found by static review/tests before the
   affected phase first ran):
   - smoke gate: FD branch kwargs leaked `q_by_name` to `run_theta_branch`
     (replaced with an explicit local `branch()` helper);
   - smoke gate: optimizer-state check iterated `state_dict()` id keys
     (StopIteration); now iterates `optimizer.state` keyed by parameter object;
   - mechanism phase: deprecated byproduct key corrected to the frozen runner's
     actual `chi_BH_old_curvature`;
   - analyze phase: mechanism mean-relative-error aggregation now guards
     degenerate (None) cells instead of emitting NaN into JSON.
5. **Smoke-gate rerun policy.** The smoke gate failed once on item 4
   (StopIteration), was fixed, and re-run; the passing run is the one recorded
   in `smoke_report.json`. This is an implementation bug fix, allowed by
   G1A_PROTOCOL.md §8 (implementation bugs fixable with change log).
6. **Audit-bundle scope.** First `audit` attempt used a zipfile-only
   integrity API (AttributeError, no bundle finalized) and would have bundled
   the 9 large q tensors (~960MB). Final policy: `bank/` and `theta/q/`
   (large tensors, reproducible from frozen protocol inputs) are hashed in
   SHA256SUMS.txt section 2 but excluded from the tar bundle; integrity
   verified by full read-back of every bundled member.
