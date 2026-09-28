# D1 关键数据速览

verdicts: {
 "IMPLEMENTATION": "PASS",
 "COVERAGE": "PASS",
 "FINAL_BUDGET_NLL_EFFECT": "NO_OBSERVED_GAIN",
 "DETECTED_TARGET_TOKEN_EFFECT": "NO_OBSERVED_GAIN",
 "NORM_MATCHING_ABLATION": "RAW_BETTER_THAN_NORM",
 "FIXED_LR_COMPARISON": "BASELINE_GRID_BOUNDARY"
}

h256 means (rows=seed, cols=arm):
| seed | ADAMW_LR050 | ADAMW_LR100 | ADAMW_LR200 | QH_RAW_A010 | QH_NORM_A010 |
|---|---|---|---|---|---|
| 2030 | 7.04666 | 6.90410 | 6.70077 | 6.89515 | 6.91746 |
| 2031 | 7.04395 | 6.89475 | 6.68841 | 6.87067 | 6.89559 |
| 2032 | 7.08833 | 6.95348 | 6.76116 | 6.93820 | 6.96760 |

Δh256 QH_NORM_A010-ADAMW_LR100 per seed: 2030: +0.01336, 2031: +0.00084, 2032: +0.01412
detection reached/3: ADAMW_LR050: 0, ADAMW_LR100: 3, ADAMW_LR200: 3, QH_RAW_A010: 3, QH_NORM_A010: 3
norm resid max abs/rel: 3.052e-05 / 1.659e-07
fallbacks: {'NONE': 690}
budget actual: {"updates": 3904, "train_fw": 7808, "train_bw": 7808, "monitor_evals": 253, "monitor_fw": 32000, "total_fw": 39808, "failures": 0, "numerical_failures": 0, "snapshots": 30}
