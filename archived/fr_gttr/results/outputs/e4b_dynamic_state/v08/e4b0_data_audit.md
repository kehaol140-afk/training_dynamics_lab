# E4B-0 v0.8 Dynamic State Data Audit

本报告为只读数据可用性审计，不改变 v0.7 / E4A 科学结论。

## Decision

- E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION: `NO`
- E4B_BASELINE_REPLAY_REQUIRED: `YES`
- E4B_EXACT_REPLAY_POSSIBLE: `YES`
- E4B_EXPECTED_GPU_WORK: 需要 baseline-only replay：8 seeds × 7 states × 8 streams；每个 stream 仅做一次 max(H)=32 step rollout，并在 [1, 2, 4, 8, 12, 16, 24, 32] 中途 observation-only probe。
- E4B0_READY: `YES`

## Aggregate Evidence

- all_theta_mapping_raw_present: `True`
- all_state_bank_reports_present: `True`
- all_expected_state_pt_present: `True`
- current_q_evidence_all_seeds: `True`
- future_q_or_probe_gradient_all_seeds: `False`
- manifest_evidence_all_seeds: `True`
- rng_evidence_all_seeds: `True`

## Per-seed Summary

| seed | states | current q | future q/probe | manifest | RNG | paired |
|---:|---:|---:|---:|---:|---:|---:|
| 2030 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2031 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2032 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2033 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2034 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2035 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2036 | 7 | 14 | 0 | 784 | 337 | 1982 |
| 2037 | 7 | 14 | 0 | 784 | 337 | 1982 |

## Boundaries

- [FR-OBSERVATION-ONLY] 本审计不运行模型、不修改 optimizer、不生成新的 B/chi。
- [FR-REPLAY-CRITICAL] exact replay 判定只基于现有文件/字段证据；若为 NO，不应盲目开始 GPU replay。
- [FR-RNG-CRITICAL] E4B-1 必须复用原 future stream 的 batch order 与 RNG 语义。
- [FR-HASH] E4B-1 应记录 q_t/q_future hash，但 summary 不保存高维 tensor。
