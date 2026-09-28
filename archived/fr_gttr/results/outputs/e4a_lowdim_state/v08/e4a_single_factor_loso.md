# E4A v0.8 Low-dimensional State Screening

本报告是 exploratory mechanism study；不改写 v0.7 confirmatory 结果，不输出 confirmed。

## Baseline H8 LOSO

- model: `M0 = seed_FE + state_fraction + B`
- mean MAE: `0.0320961`
- median MAE: `0.0323251`

| held-out seed | M0 MAE |
|---:|---:|
| 2030 | 0.0396407 |
| 2031 | 0.0375561 |
| 2032 | 0.0338882 |
| 2033 | 0.0307621 |
| 2034 | 0.0228153 |
| 2035 | 0.0249323 |
| 2036 | 0.0248509 |
| 2037 | 0.0423236 |

## H8 Candidate Ranking

Ranking 只依据 H=8；H=4/H=12 仅作邻近 horizon 对照。

| rank | candidate | status | mean gain | median gain | folds improved | delta sign stability | majority sign |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | mu_q | UNSUPPORTED | -0.0319037 | 0.0230357 | 4/8 | 8/8 | positive |
| 2 | momentum_alignment | UNSUPPORTED | -0.0369002 | -0.00817738 | 4/8 | 5/8 | negative |
| 3 | update_alignment | UNSUPPORTED | -0.0391986 | -0.049151 | 2/8 | 8/8 | negative |
| 4 | NSR_q | UNSUPPORTED | -0.071603 | -0.0802491 | 3/8 | 7/8 | positive |
| 5 | sigma_q | UNSUPPORTED | -0.0755996 | -0.100016 | 3/8 | 8/8 | positive |

## Per-Fold H8 Gains

### mu_q

| held-out seed | relative MAE improvement | delta interaction | delta sign |
|---:|---:|---:|---|
| 2030 | -0.502606 | 0.386256 | positive |
| 2031 | -0.226378 | 0.76066 | positive |
| 2032 | 0.370742 | 0.440845 | positive |
| 2033 | -0.119776 | 0.652851 | positive |
| 2034 | 0.100827 | 0.550165 | positive |
| 2035 | 0.0758896 | 0.611651 | positive |
| 2036 | 0.0726548 | 0.676821 | positive |
| 2037 | -0.0265833 | 0.599589 | positive |

### momentum_alignment

| held-out seed | relative MAE improvement | delta interaction | delta sign |
|---:|---:|---:|---|
| 2030 | -0.0606784 | -0.054951 | negative |
| 2031 | -0.0482266 | 0.0260781 | positive |
| 2032 | 0.0700747 | -0.0396962 | negative |
| 2033 | -0.32195 | 0.0570476 | positive |
| 2034 | 0.0629726 | -0.0170191 | negative |
| 2035 | 0.0183901 | -0.0131628 | negative |
| 2036 | -0.0347448 | 0.0511151 | positive |
| 2037 | 0.0189613 | -0.0093323 | negative |

### update_alignment

| held-out seed | relative MAE improvement | delta interaction | delta sign |
|---:|---:|---:|---|
| 2030 | -0.0462328 | -0.0519043 | negative |
| 2031 | -0.139277 | -0.195796 | negative |
| 2032 | -0.0104467 | -0.136218 | negative |
| 2033 | 0.219627 | -0.128534 | negative |
| 2034 | -0.302847 | -0.236116 | negative |
| 2035 | -0.108589 | -0.178514 | negative |
| 2036 | 0.126246 | -0.106586 | negative |
| 2037 | -0.0520693 | -0.148887 | negative |

### NSR_q

| held-out seed | relative MAE improvement | delta interaction | delta sign |
|---:|---:|---:|---|
| 2030 | -0.663739 | 0.106419 | positive |
| 2031 | 0.372734 | 0.229521 | positive |
| 2032 | 0.0539305 | 0.172546 | positive |
| 2033 | -0.13268 | -0.0662069 | negative |
| 2034 | -0.179838 | 0.278302 | positive |
| 2035 | -0.0278184 | 0.22561 | positive |
| 2036 | -0.20184 | 0.170525 | positive |
| 2037 | 0.206427 | 0.058986 | positive |

### sigma_q

| held-out seed | relative MAE improvement | delta interaction | delta sign |
|---:|---:|---:|---|
| 2030 | -0.147165 | 0.787338 | positive |
| 2031 | -0.259813 | 1.26811 | positive |
| 2032 | -0.121846 | 1.4973 | positive |
| 2033 | 0.190944 | 1.00742 | positive |
| 2034 | 0.0445621 | 1.04473 | positive |
| 2035 | -0.0781848 | 1.23583 | positive |
| 2036 | 0.14674 | 1.23782 | positive |
| 2037 | -0.380035 | 0.317785 | positive |

## H4/H8/H12 Contrast

| candidate | H4 mean gain | H4 delta sign | H8 mean gain | H8 delta sign | H12 mean gain | H12 delta sign |
|---|---:|---|---:|---|---:|---|
| mu_q | 0.0329459 | positive | -0.0319037 | positive | -0.223861 | positive |
| momentum_alignment | -0.046238 | positive | -0.0369002 | negative | -0.123122 | positive |
| update_alignment | -0.0723532 | negative | -0.0391986 | negative | -0.00643166 | negative |
| NSR_q | -0.155231 | positive | -0.071603 | positive | -0.0589955 | negative |
| sigma_q | -0.0540319 | negative | -0.0755996 | positive | 0.0455771 | negative |

## Exploratory Boundary

- E4A_BEST_SINGLE_STATE: `NONE`
- E4A_BEST_STATUS: `UNSUPPORTED`
- 单一低维状态量明显解释 seed heterogeneity: `False`
- controller 建议: `NOT_PERFORMED`
- 多变量组合搜索: `NOT_PERFORMED`
