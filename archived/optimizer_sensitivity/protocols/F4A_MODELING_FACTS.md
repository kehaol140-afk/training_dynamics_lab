# F4-A — Modeling Facts（交给 ChatGPT 用于下一步正式建模）

仅含 FACT / SUPPORTED_PATTERN / NEGATIVE_RESULT / OPEN_QUESTION。不提出正式模型。

## FACT

- F1（FACT）：74/74 state 的三臂管线与 F3 冻结语义逐位兼容（3 个参考 cell 三臂终范数逐位相等；losses 与 current-env frozen reference 逐位一致）。
- F2（FACT）：R_curv_32 中位随 progress 单调衰减于全部三个 scale：5M 28.6(p=.36)→1.63(p=.64)；26M 9.13(p=.20)→0.94(p=.80)；100M 744(p=.20)→6.31(p=.50)→0.92(p=.80)。
- F3（FACT）：14/14 seed 轨迹自愈（R_curv_32 最终 <3）；p_heal 区间见 healing_intervals.csv（多为 INTERVAL_CENSORED，宽 0.05–0.30）。
- F4（FACT）：λ_SO_32 > 0 于 74/74 state（source-off 有限时间增长从不消失）；幅度随 progress 从 ~0.14–0.34 衰减到 ~0.01。
- F5（FACT）：同 (scale,step) 跨 seed 幅度极差最高 5.3 dex（100M step26：86.3/744/1.58e7）；26M 1.25 dex；5M 1.86 dex。
- F6（FACT）：指标排序（Spearman vs log R_curv_32，n=74）：A_theta −0.813、clip_ratio +0.692、A_m +0.513、A_v +0.482、v_floor_fraction +0.195、K_lr −0.129。
- F7（FACT）：LOSO 探索模型 R² ≤ 0.306（C: scale 固定效应+p 最好；D: logN×p 交互最差 0.216）；排序 Spearman 0.70–0.74。
- F8（FACT）：峰中位 R_curv_32：5M=28.6、26M=9.1、100M=744（对 logN Spearman=0.50，冻结判据 <0.7 不成立）。

## SUPPORTED_PATTERN

- SP1：curvature-driven multi-step 放大是一个**早期训练窗口现象**，跨 5M–100M 与全部 14 seed 普适自愈（与 F0 的 conditioning-随规模恶化/随训练自愈一致，扩展到 multi-step R_curv）。
- SP2：**A_theta ≈ 1（θ 通道单步近中性）伴随更大的多步放大**——病态住在状态通道（F1），多步爆炸来自其与 curvature feedback 的复合，而非 θ 通道本身。
- SP3：100M 的 unstable 窗口最晚关闭（2/3 seed 的 p_heal ∈ (0.5,0.80]；step64 仍 MODERATE）。
- SP4：clip 系数（proxy for 大梯度期）与 R_curv 正相关——不稳定窗口与高梯度/强 clip 期重合。

## NEGATIVE_RESULT

- N1：无干净 scale 趋势（F8：非单调；受 5M 覆盖起点晚混淆——见 OPEN_QUESTION）。
- N2：v_floor_fraction 与 K_lr 作为跨尺度汇集预测子弱（|ρ|<0.2）——F1 的单步 v 通道病态与多步 R_curv 的跨尺度线性联系不成立。
- N3：确定性平滑模型（A–D）解释 ≤31% 方差——**seed 级随机效应必须进模型**。
- N4：α=0 臂有界（G_B_32 ~20–25）跨所有 scale/phase 稳定——放大专属 curvature 通道（F3 结论的全量复证）。

## OPEN_QUESTION

- Q1：5M 真实峰值（p<0.36 无观测）是否使 peak-vs-N 单调？（补数据请求 #3）
- Q2：p_heal 的 scale 后移是阈值敏感还是结构性的（100M 2042 在 p<0.5 已愈，2043/2044 在 (0.5,0.8]）？
- Q3：λ_SO_32 普适为正——传播子是否存在不消失的基底指数率（即使 LATE）？现有数据只到 p=0.80。
- Q4：A_theta 的强负相关是因果 proxy 还是共变量（与 clip/progress 相关）？需偏相关/分层分析。
- Q5：seed 2044 的 1.58e7 爆炸是否代表可预测的亚群体（更早 v-floor？更强 clip？）？

## 建模输入提示（非模型提案）

- 因变量建议 log10 R_curv_32 与 λ_SO_32 双目标；结构需含 seed 随机截距/斜率；scale 用 logN；progress 用连续 p（勿 bin 化替代）；healing 用区间删失生存形式（healing_intervals.csv 直接可用）。
