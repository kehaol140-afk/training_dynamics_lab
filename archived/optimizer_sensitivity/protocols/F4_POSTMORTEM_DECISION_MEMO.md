# F4 Postmortem Decision Memo（2026-09-30）

## 1. SAU 失败主因？**信息时序（observability），非阈值校准**。δ=.05：offline 诊断（R_curv_32+λ_SO）AUROC 0.837 预测长展开有害，但 online prefix 在任何 h≤16 不可分辨（最强 AUROC 0.685）；网格上限 0.30 vs λ̂(2)≈0.35 只是表层退化原因——即便放宽阈值，h=2 可达分辨力仍 <0.75。
## 2. long-horizon harm 可被完整诊断解释？**部分**：offline AUROC 0.837（排序强），但 harmful 仅 7/42 例且 balacc 0.557——解释力是排序性的，不是决策级。
## 3. early prefix 为何不能提前识别？因为在可见的 g/q/λ/|h| 及 state indicators 里，"将继续恶化"的轨迹与"已自愈"轨迹在 h≤2 时统计不可分（λ̂(2) 同为 ~0.35-0.38；分化出现在更晚，而那时效用已无增益——H_suff 中位=1）。风险信号与效用信号的时间窗**错位**。
## 4. useful horizon 是否依赖 intervention size？**弱**：H1/H2 相对 regret 随 δ 反号（.05: H2 差 +0.051；.02: H2 好 −0.075，CI 均擦 0），F2.5 8-δ surface 平坦（H_best≡1）——至多二阶交互。
## 5. 论文方向排序（基于 evidence）：
**A. Mechanism paper（stable primal / unstable differentiation）——第一优先**：F0-F4A.5 的完整证据链（病态→curvature 放大→distributed 机制→phase 自愈定律→seed 异质）全部 ESTABLISHED/SUPPORTED，且每一层的负结果都在约束机制解释，是现成的强 story。
**B. Observability-limit paper——第二**：本 postmortem 的 gap 结构（offline 可见 vs online 不可见 vs 短 horizon 已够）新颖且诚实，但 harmful 正例仅 7-8 例/δ，需要扩 cells/δ 网格才能立稳。
**C. Finite-intervention-timescale paper——弱**：WEAK_SCALE_DEPENDENCE，二阶交互 + F2.5 平坦，单独不足以成文，可作 A/B 的一个 section。
**D. Method paper——不支持**：SAU 双重否定（gap 且 H_suff=1），prescreen gate FAIL。
**E. stop——不必要**（A 线已经成熟）。
## 6. 是否再做方法？**否**（STAGE F 两个 gate 均不满足：δ=.02 H_sep=1 达标但 continuation headroom 为零；F4C 非 SUPPORTED）。方法留待外部验证或新问题设定。
## 7. External validation 推荐：**选项 2（second recipe / second dataset，50M–150M）优先于 300M replication**——机制链的 scale 侧（λ_SO 单调）已有三点支持，更大的风险是 recipe/dataset 特异性（8+3+3 seeds 全部同一 recipe/corpus）；详见 docs/EXTERNAL_VALIDATION_PLAN.md。
