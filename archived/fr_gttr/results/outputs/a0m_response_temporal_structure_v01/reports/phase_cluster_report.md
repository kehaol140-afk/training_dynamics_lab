# A0M-E2 Phase Clustering Report

features: ['h1', 'h2', 'h3', 'h4', 'v_eval', 'preclip_norm'] (h1..h4 response, v_eval@step, preclip_norm@step); optimizer state statistics NOT available in frozen logs (not saved) — noted.
n events = 72 (A0K only; A0L has 2 events/seed, too few).
phase labels: step<=40 early, 41..84 mid, >=85 late (tertiles of 128-step plan).
PCA explained variance ratio: [0.663, 0.174]
KMeans(k=3) vs phase ARI = 0.0759
silhouette = 0.5744

## cluster x phase crosstab
| phase | 0 | 1 | 2 |
|---|---|---|---|
| early | 13 | 9 | 2 |
| late | 24 | 0 | 0 |
| mid | 23 | 1 | 0 |
