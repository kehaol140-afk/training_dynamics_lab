# F4C Intervention-Scale × Horizon Mining 报告（exploratory）

- 预注册 `docs/F4C_INTERVENTION_SCALE_MINING_PREREG.md`（先于计算）；零新 rollout；数据 = F4B canonical（δ=.02/.05）+ F2.5 raw（8δ×6H×3 cells，L±/L0 完整恢复）。

## 0. Verdict：**`WEAK_SCALE_DEPENDENCE`** → STAGE D（confirmatory F4C）**DISABLED**

## 1. C2 paired（42 cells，H_eval=32）

- mean normalized regret(H2−H1)：δ=.05 **+0.0512**（seed-blocked bootstrap CI95 [+0.000, +0.113]，H2 更差）；δ=.02 **−0.0746**（[−0.158, +0.000]，H2 更好）——**H1/H2 相对优劣随 intervention size 反向**，与 F4B 的 FIXED1/FIXED2 交叉一致（.05: 0.104/0.155；.02: 0.181/0.106）。
- 但 per-cell H_best 分布两 δ 都被 H=1 主导（38/42 与 33/42；H_best≤2 占 93%/88%）——interaction 只在 H2 边际上可见，未改变最优 horizon 的众数。

## 2. C3 F2.5 micro-surface（3×100M cells，语义：各 H 自身端点 L）

- H_best(δ) = 1 于全部 8 个 δ × 3 cells（kendall τ 未定义/常数；多数小 δ 行 unresolvable，range ≤1e-6 者已剔除）。**无 δ 依赖信号**，与 C2 不同向。

## 3. 判定

Paired 方向稳定（反号交互）但（i）CI 均擦 0 边界、（ii）F2.5 surface 平坦 → 按冻结判据 WEAK_SCALE_DEPENDENCE；不满足 STAGE D 的 "SUPPORTED 或非常接近且证据明确" → **confirmatory F4C 不执行**（负结果保留：finite-intervention size 与 useful horizon 的耦合最多是二阶效应）。
