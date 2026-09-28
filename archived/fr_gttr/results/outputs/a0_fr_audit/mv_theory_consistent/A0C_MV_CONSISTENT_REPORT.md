# A0C AdamW m+v Strict-Consistency Report

## Answers

1. Does v-path improve after fixing scale? **No**. R2 change: `-201.96300895`.
2. Does state-specific q change the conclusion? **No—the v-path improvement/non-improvement conclusion matches the global-q control evaluated at the same unit-response scale.**
3. Is AdamW m+v sufficient? **No**. Direct R2: `-409.420690759`; residual `SSE/SST = 1-R2`: `410.420690759`.

The q conclusion above uses a saved global-q feature control recomputed without epsilon, so both sides use the same unit-response scale. The comparison is diagnostic and introduces no fitted parameter.

## Overall metrics

| prediction | n | corr | R2 | MAE |
|---|---:|---:|---:|---:|
| chi_m | 448 | 0.867193267032 | -207.457681809 | 1.14676271287 |
| chi_mv | 448 | 0.571131018158 | -409.420690759 | 1.29066064037 |

## State-specific q control

Both rows below use the corrected unit-response scale. This isolates the q alignment change from the removed perturbation-scale factor.

| q source | chi_m R2 | chi_mv R2 | v-path delta R2 |
|---|---:|---:|---:|
| exact state-specific q | -207.457681809 | -409.420690759 | -201.96300895 |
| prior global q features | -546.493332405 | -559.826511264 | -13.3331788586 |

## Numerical behavior

- Largest `|nu_q|`: seed `2037`, state `46`, `nu_q=-1304077.32385`.
- Largest `|chi_v|`: seed `2037`, state `46`, H=`32`, `chi_v=-15.0463342095`.
- v-path improves `0/8` seeds and `0/8` horizons.

## Alignment audit

- Matched states: `56`
- Prediction rows: `448`
- q: exact v07 seed/state direction loaded with internal tensor-hash validation.
- g: existing state-level mean of eight matched E2 pre-clip unscaled gradients.
- Scale: no `rho*sigma_q` factor; target and prediction are unit responses.
- No forward, backward, clipping, optimizer step, fit, or calibration was run.

## Per-seed metrics

```json
{
  "chi_m": {
    "2030": {
      "n": 56,
      "corr": 0.8282527363250886,
      "R2": -268.79083132034276,
      "MAE": 1.352426142034815,
      "slope": 0.07103026265137959,
      "intercept": -0.08001575231596429,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2031": {
      "n": 56,
      "corr": 0.9297696470112647,
      "R2": -216.7782013052696,
      "MAE": 1.0953718705584115,
      "slope": 0.09799869345603729,
      "intercept": -0.03946753828922507,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2032": {
      "n": 56,
      "corr": 0.9519739084456865,
      "R2": -186.24053680619517,
      "MAE": 0.9930999268748092,
      "slope": 0.11569241739651964,
      "intercept": -0.05243937442224206,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2033": {
      "n": 56,
      "corr": 0.9528513496893578,
      "R2": -152.63665965097616,
      "MAE": 1.1285688393043996,
      "slope": 0.12076539413851364,
      "intercept": -0.039284733730487104,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2034": {
      "n": 56,
      "corr": 0.9223055864501075,
      "R2": -197.55610535025994,
      "MAE": 1.144465774499139,
      "slope": 0.09968548854791032,
      "intercept": -0.05697285213446743,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2035": {
      "n": 56,
      "corr": 0.9058304336614501,
      "R2": -178.77870474327497,
      "MAE": 1.155574457156526,
      "slope": 0.10455162604524158,
      "intercept": -0.0513288727330882,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2036": {
      "n": 56,
      "corr": 0.9451784726211686,
      "R2": -173.09439696376458,
      "MAE": 1.1080618337284938,
      "slope": 0.10891824271072036,
      "intercept": -0.052784658183739924,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2037": {
      "n": 56,
      "corr": 0.7500918107424821,
      "R2": -396.16016946150404,
      "MAE": 1.1965328588202373,
      "slope": 0.055380707321152085,
      "intercept": -0.0811775148493008,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  },
  "chi_mv": {
    "2030": {
      "n": 56,
      "corr": 0.8271789867111646,
      "R2": -319.29478430538893,
      "MAE": 1.4409553564386695,
      "slope": 0.06381213338985836,
      "intercept": -0.08549664084392267,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2031": {
      "n": 56,
      "corr": 0.9250194482445014,
      "R2": -233.4593861458932,
      "MAE": 1.1251017374227104,
      "slope": 0.09270244219035031,
      "intercept": -0.043374906081936004,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2032": {
      "n": 56,
      "corr": 0.947417979814193,
      "R2": -195.6425102887698,
      "MAE": 1.0128541911494358,
      "slope": 0.11154127280685065,
      "intercept": -0.05514396225858234,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2033": {
      "n": 56,
      "corr": 0.9499083156891293,
      "R2": -170.19004298482926,
      "MAE": 1.1795026827749475,
      "slope": 0.11268963719114851,
      "intercept": -0.04427173531286541,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2034": {
      "n": 56,
      "corr": 0.9387332974810774,
      "R2": -215.50809344756726,
      "MAE": 1.1810290293351406,
      "slope": 0.09553023473152805,
      "intercept": -0.059024998549687485,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2035": {
      "n": 56,
      "corr": 0.9114811076248789,
      "R2": -192.94140941021183,
      "MAE": 1.1922604116428503,
      "slope": 0.10042770681798671,
      "intercept": -0.053202887736559484,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2036": {
      "n": 56,
      "corr": 0.9409278550588931,
      "R2": -184.71779065817248,
      "MAE": 1.1375304804881397,
      "slope": 0.10432728484116671,
      "intercept": -0.055691091745091625,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2037": {
      "n": 56,
      "corr": 0.4170794143815114,
      "R2": -2774.649937069836,
      "MAE": 2.056051233679992,
      "slope": 0.009047654884232287,
      "intercept": -0.1360718744600661,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  }
}
```

## Per-horizon metrics

```json
{
  "chi_m": {
    "1": {
      "n": 56,
      "corr": 0.9253283226754575,
      "R2": -176.89813778080358,
      "MAE": 0.18097733707542737,
      "slope": 0.1427915285523937,
      "intercept": -0.016526686176355193,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2": {
      "n": 56,
      "corr": 0.8706078018676695,
      "R2": -228.36984517946186,
      "MAE": 0.34918753496511473,
      "slope": 0.12038535396496301,
      "intercept": -0.03587879469736416,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "4": {
      "n": 56,
      "corr": 0.743364265078612,
      "R2": -266.4207541784915,
      "MAE": 0.6494246568039264,
      "slope": 0.09781604261597869,
      "intercept": -0.06542838339531563,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "8": {
      "n": 56,
      "corr": 0.7436988895830361,
      "R2": -287.2379822357329,
      "MAE": 1.1075620954625676,
      "slope": 0.0965962516826729,
      "intercept": -0.07790666426653145,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "12": {
      "n": 56,
      "corr": 0.7447418309714879,
      "R2": -365.15922921954274,
      "MAE": 1.4158759380025914,
      "slope": 0.08700794628926606,
      "intercept": -0.09358209926162936,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "16": {
      "n": 56,
      "corr": 0.7314715108388212,
      "R2": -449.40961222210575,
      "MAE": 1.6261797144682382,
      "slope": 0.07791747269180349,
      "intercept": -0.10465630774245294,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "24": {
      "n": 56,
      "corr": 0.7356848686437637,
      "R2": -622.7618959915915,
      "MAE": 1.866585006747155,
      "slope": 0.0676053021839849,
      "intercept": -0.11032706658225983,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "32": {
      "n": 56,
      "corr": 0.7500172286494337,
      "R2": -770.4089475923741,
      "MAE": 1.978309419451811,
      "slope": 0.0625440125042504,
      "intercept": -0.10745879067566819,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  },
  "chi_mv": {
    "1": {
      "n": 56,
      "corr": 0.6900893620478563,
      "R2": -236.0854261085324,
      "MAE": 0.1927223358953792,
      "slope": 0.07620695540237077,
      "intercept": -0.030972969126562333,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2": {
      "n": 56,
      "corr": 0.6238098564189378,
      "R2": -307.98298956135477,
      "MAE": 0.3726657876061981,
      "slope": 0.060607276118388106,
      "intercept": -0.06062471597278726,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "4": {
      "n": 56,
      "corr": 0.5098789322438925,
      "R2": -367.6847785308053,
      "MAE": 0.6963342290590637,
      "slope": 0.04537810908244947,
      "intercept": -0.10484935300964641,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "8": {
      "n": 56,
      "corr": 0.47563341084365673,
      "R2": -422.2672603083337,
      "MAE": 1.2011938829536655,
      "slope": 0.038527525030218765,
      "intercept": -0.1504984836696562,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "12": {
      "n": 56,
      "corr": 0.4432779466447039,
      "R2": -578.3518425424913,
      "MAE": 1.5560433320151539,
      "slope": 0.02964002812393719,
      "intercept": -0.18427443281543845,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "16": {
      "n": 56,
      "corr": 0.39793701618234373,
      "R2": -772.7533785001991,
      "MAE": 1.8126968516037234,
      "slope": 0.022209237674809555,
      "intercept": -0.2050835247601294,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "24": {
      "n": 56,
      "corr": 0.3429736646033605,
      "R2": -1284.241983962829,
      "MAE": 2.14524700632458,
      "slope": 0.013842200457640206,
      "intercept": -0.22046111549570502,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "32": {
      "n": 56,
      "corr": 0.30519321741069266,
      "R2": -1930.347064097563,
      "MAE": 2.3483816974741214,
      "slope": 0.009455747496814205,
      "intercept": -0.2220768290576591,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  }
}
```
