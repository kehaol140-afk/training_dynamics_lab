# G1C Phase 1 Report — single-state response smoke

- cell: step950000 x stream1 x H=1
- verdict: **PASS** (feasibility smoke, not science)
- q sha256: 08128dd4f86429399d9b732778d18e81b2a35f76ef978d71a25c3b883ca2cbb0
- sigma_q: 0.001011787867  epsilon: 0.0002529469667  (RHO=0.25)

## Checks

| check | ok | detail |
|---|---|---|
| replay_gpu_loss_bitexact | True | loss_gap=0 |
| replay_gpu_grads_nonwte_bitexact | True | max_gap_nonwte=0 wte_gap=0 (wte: embedding atomicAdd, expected non-deterministic on CUDA) |
| replay_cpu_fp32_bitexact | True | loss_gap=0 max_grad_gap=0 |
| gradient_propagation | True | loss=10.661172 |g|=106.192673 clip_coef=0.009417 finite=True |
| h1_identity_B_eq_C | True | |B1-C1|=0 |
| h1_fd_vs_tangent | True | [{"level": "eps", "epsilon": 0.00025294696671568535, "chi_fd": -1.5392963161366064, "rel_err_B": 0.0001699318610891975, "rel_err_C": 0.0001699318610891975, "X_plus": -0.0004025605184147188, "X_minus": 0.0003761601496720478}, {"level": "eps/2", "epsilon": 0.00012647348335784268, "chi_fd": -1.53944087782638, "rel_err_B": 7.601058259577578e-05, "rel_err_C": 7.601058259577578e-05, "X_plus": -0.00020789855612257202, "X_minus": 0.00018149834436174275}] |
| hvp_executability | True | peak_mem_gb=20.2 time=646.2s/584.7s |hvp|=2.49014 finite=True repeat_max_gap=0 <q,hvp>=0.000231095 |

## FD vs tangent (H=1)

| level | epsilon | chi_fd | rel_err_B | rel_err_C | X_plus | X_minus |
|---|---|---|---|---|---|---|
| eps | 0.000252947 | -1.5392963 | 0.000169932 | 0.000169932 | -0.000402561 | 0.00037616 |
| eps/2 | 0.000126473 | -1.5394409 | 7.60106e-05 | 7.60106e-05 | -0.000207899 | 0.000181498 |

## Notes

- Observable: parameter-displacement projection X_h=<q, theta_h-theta_0> (Amendment A2; fp64 bookkeeping, exact).
- q: iid N(0,1) full support, unit total L2 norm (Amendment A1).
- FD impulse: ±epsilon*q on first future step gradients pre-clip (Amendment A3).
- wte gradient non-determinism on CUDA is a known embedding-backward atomicAdd effect; CPU fp32 replay is bit-exact.
- HVP: bf16-materialized double backward (protocol 6.7 memory strategy; same amp_bf16 numerics class), seq=512 per sequence.