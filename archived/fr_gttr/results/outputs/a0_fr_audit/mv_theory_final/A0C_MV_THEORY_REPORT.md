# A0C AdamW m+v Theory Response

## 结论

- Does adding v-path improve? **No**. R2 变化为 `-1.84610338655`。
- AdamW adaptive preconditioning sufficient? **No**. m+v 的直接预测 R2 为 `-6.33463814741`。
- 剩余未解释比例（`1-R2`）为 `7.33463814741`，残差 MAE 为 `0.169457010296`。
- 分层一致性：逐 seed 有 `1` 个 R2 改善；逐 horizon 有 `0` 个 R2 改善。

## Overall metrics

| prediction | n | corr | R2 | MAE | slope | intercept |
|---|---:|---:|---:|---:|---:|---:|
| chi_m | 448 | 0.145064658975 | -4.48853476086 | 0.164150720706 | 0.0784139129745 | -0.179740502716 |
| chi_mv | 448 | 0.123205625417 | -6.33463814741 | 0.169457010296 | 0.0528516884037 | -0.180947594199 |

## 量纲与 epsilon 约定

A0C 的观测 `chi=mean[(X_plus-X_minus)/(2*epsilon)]` 已经除以 `epsilon=rho*sigma_q`，因而是单位扰动响应。按本次明确指令，主预测仍乘一次 `rho*sigma_q`；这使预测具有原始位移尺度，而目标具有单位响应尺度。该约定不是 拟合参数，但构成必须保留的量纲不一致，R2/MAE 的幅值解释须带此 caveat。

另一个预先指定的不一致是方向：本审计按要求使用全局 `v03_pilot/probe_direction_q.pt`，而 v07 A0C `chi` 是沿各 seed/state 自身的 state-specific q 测得。因此结果是所指定全局 q 理论量与 state-specific q 响应的比较。

## Replay audit

- Matched states: `56`
- Matched E2 streams: `448`
- Gradient capture: after `scaler.unscale_(optimizer)`, before clipping and before `optimizer.step()`.
- No optimizer step, training, checkpoint write, empirical scale, intercept correction, or calibration was used to form predictions.

## Pre-clip sensitivity audit

最大 `|nu_q|` 位于 seed `2037`、state `46`，`nu_q=-190335.399987`。排除此单一 state 仅作为事后稳健性检查时，m+v 相对 momentum-only 的 R2 变化仍为 `-0.00343182350948`；该排除不改变主结果，也未用于选择或校准模型。

## Numerical replay caveat

原 v07 分支在 CUDA bfloat16 下运行；本次因约束与运行环境使用 CPU，沿既有 `make_amp` CPU 路径以 FP32 forward/backward。保存的原始 pre-clip 范数与 CPU 重放范数可在 `state_results[*].stream_audit` 中逐流核对。

## Per-seed metrics

```json
{
  "chi_m": {
    "2030": {
      "n": 56,
      "corr": 0.5387497537128134,
      "R2": -1.1524971880824908,
      "MAE": 0.12969294327559716,
      "slope": 0.6361266467713653,
      "intercept": -0.1499437224088568,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2031": {
      "n": 56,
      "corr": 0.8165391820387495,
      "R2": -1.4470369902877418,
      "MAE": 0.12422546248797033,
      "slope": 2.6329658934006175,
      "intercept": -0.06129477572287326,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2032": {
      "n": 56,
      "corr": 0.7172754605853996,
      "R2": -2.826436763017767,
      "MAE": 0.15361669063107583,
      "slope": 2.010118870332466,
      "intercept": -0.11764760464984468,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2033": {
      "n": 56,
      "corr": 0.7513675696096035,
      "R2": -2.298794512167487,
      "MAE": 0.1735985929273141,
      "slope": 5.2211721849540425,
      "intercept": -0.06345057609183957,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2034": {
      "n": 56,
      "corr": 0.7921826411145219,
      "R2": -2.0813061106607353,
      "MAE": 0.1570460038345217,
      "slope": 3.0054446601054887,
      "intercept": -0.09095920116614695,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2035": {
      "n": 56,
      "corr": 0.5387503800586143,
      "R2": -2.1166060762212906,
      "MAE": 0.1609588682978355,
      "slope": 2.258393391763527,
      "intercept": -0.12158777974552436,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2036": {
      "n": 56,
      "corr": 0.6241009393651702,
      "R2": -2.223236813095045,
      "MAE": 0.16332998556683273,
      "slope": 2.3886315933410884,
      "intercept": -0.11980102549296082,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2037": {
      "n": 56,
      "corr": 0.2699773338871972,
      "R2": -35.24874727061991,
      "MAE": 0.25073721862969667,
      "slope": 0.04368291028328438,
      "intercept": -0.1471335754670642,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  },
  "chi_mv": {
    "2030": {
      "n": 56,
      "corr": 0.5375575932086825,
      "R2": -1.1565659916957731,
      "MAE": 0.1297832475911301,
      "slope": 0.6345443818966485,
      "intercept": -0.15009928708798415,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2031": {
      "n": 56,
      "corr": 0.8136182260787194,
      "R2": -1.4504097733361618,
      "MAE": 0.12427479260555074,
      "slope": 2.6242672748612432,
      "intercept": -0.06175945555248874,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2032": {
      "n": 56,
      "corr": 0.7179628134818201,
      "R2": -2.8322306300229947,
      "MAE": 0.1537373154108643,
      "slope": 2.0217467364948503,
      "intercept": -0.11747742344527558,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2033": {
      "n": 56,
      "corr": 0.7479207914704595,
      "R2": -2.304143282783968,
      "MAE": 0.17370840848431873,
      "slope": 5.233314769372553,
      "intercept": -0.06370842471341548,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2034": {
      "n": 56,
      "corr": 0.7944739258184073,
      "R2": -2.0754589266141377,
      "MAE": 0.15695466424375978,
      "slope": 2.9946941040870803,
      "intercept": -0.09103993752808681,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2035": {
      "n": 56,
      "corr": 0.5386587693075556,
      "R2": -2.1242946560385465,
      "MAE": 0.16116986576576917,
      "slope": 2.2828850810355923,
      "intercept": -0.12130319640220705,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2036": {
      "n": 56,
      "corr": 0.620328986269797,
      "R2": -2.2289850782938387,
      "MAE": 0.16343661178932295,
      "slope": 2.380998664500525,
      "intercept": -0.12029416919857462,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2037": {
      "n": 56,
      "corr": 0.2683950450935903,
      "R2": -59.166325579557004,
      "MAE": 0.2925911764775734,
      "slope": 0.0340708266999005,
      "intercept": -0.14767837759022298,
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
      "corr": 0.13490704924171082,
      "R2": -9.913798223854357,
      "MAE": 0.045178189265822614,
      "slope": 0.07263239651226809,
      "intercept": -0.048698693804450896,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2": {
      "n": 56,
      "corr": 0.07051376534354445,
      "R2": -11.487370837831858,
      "MAE": 0.08092137364578636,
      "slope": 0.034018420786367375,
      "intercept": -0.08793197128035579,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "4": {
      "n": 56,
      "corr": 0.018800536056826064,
      "R2": -10.800196792519028,
      "MAE": 0.12974524683985708,
      "slope": 0.008631132512839578,
      "intercept": -0.14263636297855334,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "8": {
      "n": 56,
      "corr": 0.03166297234278908,
      "R2": -8.981935719577864,
      "MAE": 0.18390785674280677,
      "slope": 0.014348416223860898,
      "intercept": -0.2038439279652213,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "12": {
      "n": 56,
      "corr": 0.03388113222195796,
      "R2": -10.017034898538407,
      "MAE": 0.2117728691042612,
      "slope": 0.013810208264653104,
      "intercept": -0.23644027895196473,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "16": {
      "n": 56,
      "corr": 0.025786222142028473,
      "R2": -11.165612341433613,
      "MAE": 0.22248301880190022,
      "slope": 0.009583282806829043,
      "intercept": -0.2501325467941556,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "24": {
      "n": 56,
      "corr": 0.030888648949381252,
      "R2": -13.429721955012226,
      "MAE": 0.22350556013633227,
      "slope": 0.009903232861331488,
      "intercept": -0.2527543237357021,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "32": {
      "n": 56,
      "corr": 0.04208363387174798,
      "R2": -15.223764594795707,
      "MAE": 0.2156916511140776,
      "slope": 0.012243818335131537,
      "intercept": -0.24542988979964028,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  },
  "chi_mv": {
    "1": {
      "n": 56,
      "corr": 0.12167676673233872,
      "R2": -10.580926257262877,
      "MAE": 0.045611286729861276,
      "slope": 0.058907868108445886,
      "intercept": -0.0488114147967919,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "2": {
      "n": 56,
      "corr": 0.05991832366371185,
      "R2": -12.395113384273088,
      "MAE": 0.08178713547639964,
      "slope": 0.025857756426488607,
      "intercept": -0.08806556989314333,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "4": {
      "n": 56,
      "corr": 0.012459964228090236,
      "R2": -11.947926643581194,
      "MAE": 0.13147503984318426,
      "slope": 0.005061308684425277,
      "intercept": -0.14275085597850715,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "8": {
      "n": 56,
      "corr": 0.02282256275197246,
      "R2": -10.462959199242313,
      "MAE": 0.18736053394928845,
      "slope": 0.00893904630749963,
      "intercept": -0.20412260915221328,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "12": {
      "n": 56,
      "corr": 0.02452039179879993,
      "R2": -12.247592650232688,
      "MAE": 0.21694154930750015,
      "slope": 0.008422953568714534,
      "intercept": -0.236785290809512,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "16": {
      "n": 56,
      "corr": 0.015819858544686705,
      "R2": -14.401182736452668,
      "MAE": 0.22936084827906505,
      "slope": 0.004823854008057086,
      "intercept": -0.2504887986137105,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "24": {
      "n": 56,
      "corr": 0.01818867298004004,
      "R2": -19.462394923428228,
      "MAE": 0.2337813936389794,
      "slope": 0.004521620036419143,
      "intercept": -0.25320535747474454,
      "regression_direction": "observed=intercept+slope*predicted"
    },
    "32": {
      "n": 56,
      "corr": 0.025514953175674173,
      "R2": -24.909006795915985,
      "MAE": 0.22933829514401102,
      "slope": 0.005430551562748659,
      "intercept": -0.2460170740856416,
      "regression_direction": "observed=intercept+slope*predicted"
    }
  }
}
```
