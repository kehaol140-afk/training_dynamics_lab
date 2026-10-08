# optimizer_sensitivity

**Stable Training, Unstable Sensitivities: Conditioning of AdamW Training Trajectories at Scale**

核心问题：训练 primal dynamics 稳定时，对完整 AdamW 训练轨迹求导得到的
tangent / adjoint / hypergradient dynamics 是否随规模、second moment 接近零、
mixed precision、embedding token frequency、gradient clipping 而病态？

Seed evidence（不预设结论，须独立验证）：legacy G1c 1B 中观察到的
primal 稳定 + full tangent C 超线性放大 + rare-token 零/极小 exp_avg_sq +
eps-dominated denominator + sensitivity 递推溢出。

## 当前阶段（F0，只读离线）
- 使用 legacy 冻结 bank（v07 4.98M / G1a 26.68M / G1b 100.78M），
  不新训练、不下载 1B、不修改 legacy。
- 产出：`docs/F0_EXISTING_BANK_SENSITIVITY_AUDIT.md` +
  `outputs/f0_existing_bank_audit/{summary.json, per_parameter_group.csv,
  embedding_frequency_bins.csv, fd_jvp_comparison.csv}`

## 结构
- `src/opsens/` — 冻结基础设施 vendor（见 docs/LEGACY_PROVENANCE.md）+ 新分析代码
- `configs/` — F0 审计冻结口径
- `scripts/` — runner
- `tests/` — 单测
- 运行环境：conda env `fr_minimind`，`PYTHONPATH=src`，只读访问 `../legacy_fluctuation_response/`

## 红线
- 不修改 forward AdamW 规则；
- 不为让假设通过而改阈值；
- legacy 一切只读；
- 不从 seed evidence 预设 G1c 归因（SCALE_TREND / ARTIFACT 各选项由数据裁决）。
