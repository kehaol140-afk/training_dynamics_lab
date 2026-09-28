# A0a Reliability / Errors-in-Variables

A0A_DATA_INTEGRITY = PASS
H = 8

## chi reliability

- exhaustive unique 4+4 splits: `35`
- rho_chichi full median: `0.979999`
- P05/P95: `[0.944662, 0.994098]`

## B reliability

- **UNAVAILABLE from existing v0.7 raw.**
- v0.7 `e1_streams` 保存了 `s_r` 和 first preclip norm，但没有保存每条 stream 的 `X_theta(H)`。
- 因此不能合法地对 16 条 E1 stream 做 8+8 split-half；不能从聚合 B 反推。
- 下一步若要完成 A0a，需要 baseline-only E1 exact replay，复用原 16 个 manifest/stream seed，只额外保存 per-stream X_theta(H)。

## Observed H8 diagnostic

- seed-FE K (recomputed from frozen raw): `0.376861`
- within-seed detrended r: `0.540557`

## Boundary

- Spearman–Brown/disattenuation 只在 classical independent additive measurement-error 假设下解释。
- 不能用本报告改写 v0.7 confirmation。

## Final

RHO_BB_H8_MEDIAN = UNAVAILABLE
RHO_CHICHI_H8_MEDIAN = 0.9799985092127281
K_H8_OBSERVED = 0.3768608979182702
R_H8_OBSERVED = 0.5405571946383922
B_MEASUREMENT_NOISE_CAN_EXPLAIN_K377 = B_RELIABILITY_UNAVAILABLE_FROM_EXISTING_RAW
