# F4C Intervention-Scale × Horizon Mining 预注册（exploratory）

- 日期：2026-09-30，先于计算。零新 rollout；数据：F4B canonical（δ=.02/.05，42 cells，H_eval=32）+ F2.5 raw（3×100M cells，δ ∈ {1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,5e-2,1e-1} × H ∈ {1,2,4,8,16,32}，L0/L± 真实 CRN 损失；注意其 L 在各 H 自身端点——语义差异如实标注，不与 F4B 混合估计）。
- C2（paired，F4B 42 cells）：per (δ,cell)：H_best = argmin_H normalized_regret(H)（ties 取更小 H）、H_suff(ε=.05)；per δ 的 fixed-H 平均 regret 曲线；**H2−H1 的 mean-regret 差**（seed-blocked bootstrap 10⁴ 重采样 CI，逐 δ）；LOSO-selected BEST_FIXED_H(delta)（取自 F4B canonical fold_hp）。
- C3（F2.5 micro-surface，机制支持）：per (cell, δ)：用 F4A h_H（F2 同源 tangent 内积）按 F4B 冻结规则选 action，在各 (δ,H) 的真实 L±/L0 上得 regret(H,δ)；H_best(δ) = argmin_H（ties 更小 H）；输出 surface。3 cells → 仅 mechanistic micro-surface，不外推。
- C4 verdict（冻结）：`INTERVENTION_SCALE_HORIZON_COUPLING_SUPPORTED` := 两 δ paired 方向稳定（δ=.05 的 best-H 分布随机占优于（stochastically ≤）δ=.02，且 H2−H1 regret 差符号随 δ 从有利转不利或差值显著）且 F2.5 surface 定性同向（H_best(δ) 随 δ 非增，Kendall τ ≤ 0 于 ≥2/3 cells）；`WEAK_SCALE_DEPENDENCE` := 方向一致但上述显著性/同向性部分不满足；`CELL_SPECIFIC_ONLY` / `NO_SCALE_HORIZON_COUPLING` / `INSUFFICIENT_EXISTING_DATA` / `UNRESOLVED`。仅 SUPPORTED（或非常接近且证据明确）才允许 STAGE D confirmatory。
