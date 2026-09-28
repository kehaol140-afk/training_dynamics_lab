# C0-GTTR Pre-flight Source Check — 2026-09-21

## Blocker: protocol file SOURCE_NOT_FOUND

`FR_C0_Geometry_Tracking_Validation_Claude_Code.md` was requested from the repo
root. Exhaustive search results:

- repo root: ABSENT (30 .md files present, no C0 file)
- entire repo (case-insensitive `*c0*` / `*GTTR*` / geometry+tracking grep): ABSENT
- `~/`, `/tmp`, system-wide maxdepth-6: ABSENT
- no partial C0 setup exists: no `outputs/*c0*` dir, no `research/{src/fr,scripts,tests}/*c0*` files

Consequence: specified anchors, frozen config values, T1-T9 definitions,
pass/fail thresholds, 3-arm details, and the probe/HVP/eval call-time caps
("完整协议中的调用与时间上限") are NOT recoverable from the repository.
Per protocol discipline (config frozen before any GPU; 禁止虚构补齐), NO
implementation, NO GPU launch, NO budget consumption was started.
Optimizer updates consumed this round: **0**. Eval/HVP/probe calls: **0**.

## Verified from actual logs (not transcriptions)

- B0 `outputs/fr_b0_response_mechanisms_v01/gpu_budget_log.json`: TOTALS 14,208 updates
- B0.5 `outputs/fr_b05_mechanism_identification_v01/gpu_budget_log.json`: TOTALS 1,609
- B0.6 `outputs/fr_b06_regime_dynamics_v01/gpu_budget_log.json`: TOTALS 1,760; project_cumulative **17,577**
- Sum check: 14,208 + 1,609 + 1,760 = **17,577 / 18,000** — matches the user's last report.
- Headroom: 423. Mandated C0 cap: smoke 24 + main 384 = 408 (fits, 15 spare).

## Mandated reading inventory

- FR_PROJECT_STATUS.md / FR_EXPERIMENT_INDEX.md: present (B0.6 = §46 / latest row).
- A0C: `outputs/a0_fr_audit/response_propagation_v01/` (REPORT + summary.json; pre-contract era, no config.json).
- A0E2: config OK. A0O/A0P/A0Q/A0S/A0T/A0T-extra/A0U: contract.json all OK.
- A0F2 standalone report: **FOUND** — `outputs/a0_fr_audit/cross_partition_extension_v01/CROSS_PARTITION_EXTENSION_REPORT.md` (NOT SOURCE_NOT_FOUND).
- B0/B0.5/B0.6 configs + budget logs: all OK (B0.6 audit pack verified 25/25).

## To proceed

Provide the protocol file content (paste into chat or restore the file at repo
root). On receipt: freeze config (seq 0) → T1-T9 CPU tests → 24-update smoke →
2-cell × 3-arm × 64-step pilot (384) → analysis/report, within 408 updates.
