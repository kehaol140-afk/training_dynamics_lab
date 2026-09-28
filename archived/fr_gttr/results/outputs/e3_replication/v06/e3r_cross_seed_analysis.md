# v0.6 跨 seed FR 认证

- training seeds: `[2026, 2027, 2028, 2029]`
- states/seed: `7`
- 科学结论：**未自动判定**

## H=1

### 每 seed 独立映射
- seed 2026: Pearson=`0.70436`, Spearman=`0.857143`, R2=`0.496123`, K=`1.07158`, sign=`1.000`
- seed 2027: Pearson=`0.798062`, Spearman=`0.535714`, R2=`0.636903`, K=`1.62866`, sign=`1.000`
- seed 2028: Pearson=`0.546608`, Spearman=`0.5`, R2=`0.298781`, K=`0.449372`, sign=`1.000`
- seed 2029: Pearson=`-0.17137`, Spearman=`-0.285714`, R2=`0.0293678`, K=`-0.370317`, sign=`1.000`

### Pooled raw（仅描述）
- Pearson: `0.362881`
- Spearman: `0.410509`
- R2: `0.131682`
- K: `0.523301`

### Within-seed 去训练趋势
- residual Pearson: `0.31133`
- residual Spearman: `0.459223`
- first-difference Pearson: `0.260102`
- first-difference Spearman: `0.382609`

### step-adjusted pooled
- step-only R2: `0.671026`
- step+B R2: `0.690773`
- incremental partial R2(B): `0.0600266`
- adjusted K_B: `0.210977`

### Leave-One-Seed-Out
- folds improved: `3/4`
- mean relative MAE improvement: `-0.0174973`
- median relative MAE improvement: `0.115611`
- hold seed 2026: step-only MAE=`0.00227461`, step+B MAE=`0.00332722`, improvement=`-0.462765`, train K_B=`0.248063`
- hold seed 2027: step-only MAE=`0.00823559`, step+B MAE=`0.00736611`, improvement=`0.105576`, train K_B=`0.142455`
- hold seed 2028: step-only MAE=`0.00503567`, step+B MAE=`0.00440296`, improvement=`0.125646`, train K_B=`0.0612378`
- hold seed 2029: step-only MAE=`0.00923838`, step+B MAE=`0.00774589`, improvement=`0.161554`, train K_B=`0.372404`

## H=8

### 每 seed 独立映射
- seed 2026: Pearson=`0.157567`, Spearman=`0.107143`, R2=`0.0248274`, K=`0.227751`, sign=`0.857`
- seed 2027: Pearson=`0.777856`, Spearman=`0.321429`, R2=`0.60506`, K=`0.743971`, sign=`0.429`
- seed 2028: Pearson=`0.424612`, Spearman=`0.214286`, R2=`0.180295`, K=`0.399957`, sign=`0.429`
- seed 2029: Pearson=`0.0309574`, Spearman=`-0.0357143`, R2=`0.000958359`, K=`0.0749641`, sign=`0.714`

### Pooled raw（仅描述）
- Pearson: `0.490769`
- Spearman: `0.260536`
- R2: `0.240855`
- K: `0.545538`

### Within-seed 去训练趋势
- residual Pearson: `0.508023`
- residual Spearman: `0.444992`
- first-difference Pearson: `0.461105`
- first-difference Spearman: `0.345217`

### step-adjusted pooled
- step-only R2: `0.625369`
- step+B R2: `0.729805`
- incremental partial R2(B): `0.27877`
- adjusted K_B: `0.368437`

### Leave-One-Seed-Out
- folds improved: `3/4`
- mean relative MAE improvement: `0.0986323`
- median relative MAE improvement: `0.12359`
- hold seed 2026: step-only MAE=`0.0257915`, step+B MAE=`0.0193726`, improvement=`0.248878`, train K_B=`0.375002`
- hold seed 2027: step-only MAE=`0.0427973`, step+B MAE=`0.0357352`, improvement=`0.165012`, train K_B=`0.255719`
- hold seed 2028: step-only MAE=`0.0254945`, step+B MAE=`0.0233997`, improvement=`0.0821667`, train K_B=`0.372571`
- hold seed 2029: step-only MAE=`0.03456`, step+B MAE=`0.0380688`, improvement=`-0.101528`, train K_B=`0.419038`

## H=32

### 每 seed 独立映射
- seed 2026: Pearson=`0.735976`, Spearman=`0.392857`, R2=`0.54166`, K=`0.832066`, sign=`0.857`
- seed 2027: Pearson=`0.822546`, Spearman=`0.0357143`, R2=`0.676582`, K=`1.83236`, sign=`0.571`
- seed 2028: Pearson=`0.0142959`, Spearman=`0.142857`, R2=`0.000204372`, K=`0.0158143`, sign=`0.714`
- seed 2029: Pearson=`-0.52005`, Spearman=`-0.321429`, R2=`0.270452`, K=`-0.577698`, sign=`0.429`

### Pooled raw（仅描述）
- Pearson: `0.188905`
- Spearman: `0.161467`
- R2: `0.035685`
- K: `0.226541`

### Within-seed 去训练趋势
- residual Pearson: `0.0965846`
- residual Spearman: `0.321292`
- first-difference Pearson: `0.129807`
- first-difference Spearman: `0.133913`

### step-adjusted pooled
- step-only R2: `0.55026`
- step+B R2: `0.557171`
- incremental partial R2(B): `0.0153683`
- adjusted K_B: `0.100747`

### Leave-One-Seed-Out
- folds improved: `2/4`
- mean relative MAE improvement: `-0.118883`
- median relative MAE improvement: `-0.00336322`
- hold seed 2026: step-only MAE=`0.0295776`, step+B MAE=`0.0294067`, improvement=`0.00577691`, train K_B=`0.00910223`
- hold seed 2027: step-only MAE=`0.0489842`, step+B MAE=`0.0495967`, improvement=`-0.0125034`, train K_B=`-0.033426`
- hold seed 2028: step-only MAE=`0.0277404`, step+B MAE=`0.027095`, improvement=`0.0232653`, train K_B=`0.0728366`
- hold seed 2029: step-only MAE=`0.0446433`, step+B MAE=`0.066611`, improvement=`-0.492073`, train K_B=`0.477654`
