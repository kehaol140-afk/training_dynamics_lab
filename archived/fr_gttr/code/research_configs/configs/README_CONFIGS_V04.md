# v0.4 配置说明

`stage_c_v04.json` 是 Stage-C pilot 的冻结配置。

核心：
- `planned_total_optimizer_steps=128`
- `mid_optimizer_step=64`
- 因此 MID 被明确定义为该 pilot 预算的 50%
- `probe_pool_size=32`
- `eval_pool_size=128`
- `e1_streams=16`
- `e2_streams=8`
- `horizons=[1,8,32]`
- `rho=[0.10,0.25,0.50,1.00]`

当前参数仍是 protocol pilot，不是最终跨 seed discovery 配置。
