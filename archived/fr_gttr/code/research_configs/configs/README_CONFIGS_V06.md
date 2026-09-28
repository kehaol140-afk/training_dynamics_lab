# v0.6 配置

- training seeds: 2026,2027,2028,2029
- states/seed: 46,52,58,64,70,76,82
- rho: 0.25
- H: 1,8,32
- E1 streams/state: 16
- E2 streams/state: 8

不同 training seed 使用不同模型初始化、不同训练 batch trajectory，
并使用不同 future stream seeds。
