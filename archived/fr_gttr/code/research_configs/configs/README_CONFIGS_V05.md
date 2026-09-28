# v0.5 E3 配置说明

- state steps: `[46,52,58,64,70,76,82]`
- primary rho: `0.25`
- E1 streams/state: `16`
- E2 streams/state: `8`
- horizons: `[1,8,32]`
- seed: `2026`

7 个 state 都来自同一训练 seed，因此这是 state-level pilot，
不能把 7 个点当成 7 个独立训练 seed。
