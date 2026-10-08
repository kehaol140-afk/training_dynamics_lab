# F4-A Report Erratum（2026-09-30）

本文件是对 `docs/F4A_SCALE_PHASE_SEED_MAP_REPORT.md`（commit `9015d81`）的两项修正与一项术语统一。原报告不作改动。

## 修正 1 — self-healing 计数不符合 prereg 严格判据

原报告 §1/RQ1 表述"14/14 条 seed 轨迹全部自愈"**不准确**。按 prereg §11 的 sustained-healing 严格判据（R_curv_32 < 3 且其后全部已观测 state 保持 <3），`healing_intervals.csv` 的实际结果为：

- **严格 sustained healing：13/14**（13 条 INTERVAL_CENSORED，即覆盖内已进入并保持 <3，精确 crossing 落于相邻 checkpoint 区间内）；
- **1 条不满足：5M seed 2032 = `NOT_HEALED_IN_COVERAGE`**——其 R_curv_32 序列（31.9 / 44.4 / 15.6 / 4.6 / **31.6** / 2.9 / 6.3）在 p≈0.55 处复发（31.6），末两点 2.9/6.3 中 6.3 > 3，不满足"其后保持 <3"。

对结论的影响：phase 自愈的**方向性证据**仍然成立（该轨迹亦从 31.9 降至 ~3-6 量级），但"全部轨迹严格自愈"的表述应降级为"13/14 严格满足、1/14 呈 mid-window 复发且未在覆盖内完成自愈"。F4-A 主标签不受影响（PHASE_STRUCTURE 仅要求方向在 ≥2/3 seeds 一致），但 HIGH_SEED_HETEROGENEITY 次标签获得又一实例（复发式轨迹存在）。

## 修正 2 — 5M 不具备完整 EARLY→LATE phase 覆盖

原报告在多处把三 scale 并列为同等 phase 证据。事实上按 prereg 的 phase 定义（EARLY p<0.34、MID 0.34≤p<0.67、LATE p≥0.67）：

- **5M 的 7 个 step（46–82）对应 p=0.359–0.641，全部落在 MID bin**——5M 没有任何 EARLY 或 LATE 观测点；
- 26M/100M 有 matched EARLY(p=.203)/MID(p=.5)/LATE(p=.797)。

因此 5M 当前只能支持 **within-window decay trend**（窗口内衰减），不能声称 5M 已验证完整 EARLY→LATE phase structure。原报告中"三 scale 全部自愈"的表意应读作"三个 scale 的已覆盖窗口内均呈衰减；其中 26M/100M 覆盖完整 phase 三段，5M 仅覆盖 MID 段"。该覆盖不对称同时是 peak-vs-N 非单调结论（F8）的最大混淆源（5M 峰值若在 p<0.36 则被系统性低估）——已在 F4A.1（matched coverage completion）中处理。

## 术语统一

原报告部分行文将 LATE 段称为"fully stable / 治愈"。由于 **λ_SO_32 在 74/74 个观测点均为正**（source-off 有限时间增长不消失，仅幅度衰减一个数量级），后续统一使用：

- **CURVATURE-DOMINANT REGIME**（R_curv_32 ≥ 3，curvature 诱导的过剩放大显著）
- **CURVATURE-NEUTRAL / WEAK REGIME**（R_curv_32 < 3）

描述对象是 *curvature-induced excess amplification 的自愈*，不是所有 tangent growth 的消失。LATE = curvature-neutral regime，不等于 fully stable phase。
