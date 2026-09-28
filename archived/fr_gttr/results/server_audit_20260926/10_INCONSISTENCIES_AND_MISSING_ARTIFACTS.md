# 10 — 不一致与缺失制品清单（文档 vs raw output）

> 审计立场：不替读者裁决；每条给出双方证据与项目内已记录的处置。分类：
> [CONFLICT] 文档间冲突；[ERRATUM] 项目内勘误（原记录保留）；[OVERCLAIM] 过强表述撤回；
> [MISSING] 制品缺失/不可验证；[SUMMARY_ONLY] 数字只存在于汇总层；[HYGIENE] 文档卫生问题。

## A. 科学结论级冲突（全部有项目内处置记录）

1. **[CONFLICT→已裁决] 旧 A0E "raw curvature 有害"（gain_H=−0.1357） vs A0E2（+0.0686 正贡献）**
   - 根因：旧 BH 初始脉冲用裸 q 跳过 J_clip（H=1 即与 B/C 不一致）；差值 −0.204352 完全来自初始化污染。
   - 处置：STATUS §14 禁引清单第 1–3 条；A0E 整体废弃仅存档。RAW_VERIFIED。
2. **[CONFLICT→已裁决] mv_theory_final（声称 v 路径改善） vs mv_theory_consistent（0/8 seed、0/8 horizon）**
   - 处置：mv_theory_final 列入禁引；consistent 版冻结。RAW_VERIFIED。
3. **[CONFLICT→ERRATUM] C0 field 判定 LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT vs C0X 复算（聚合门通过 R=0.529/C=0.468）**
   - 根因：C0 分析器把冻结的**聚合**半库门实现成**逐点**门（单点 1.166 触发）。
   - 处置：ERRATUM_REQUIRED；原 verdict 保留不覆盖；C1 在独立库翻转出 field 正面支持。
   - 遗留：g3 半库聚合门因 C0 未保存半库 SSE/TSS = **[MISSING] DATA_MISSING**。
4. **[CONFLICT→已裁决] G1c EXTERNAL_CHECKPOINT_UNAVAILABLE（HF 筛查）vs 复核 FULL_STANDARD_CHECKPOINT_AVAILABLE（直链探测）**
   - 根因：directory URL 404 ≠ 文件不存在；旧 release 用 .pt 扩展名。处置：判定为核验路径修正（非科学改判）；HF 筛查保留为历史。
5. **[ERRATUM 链] F0 terminal 导数 SENS=−5060.5 vs FD≈−0.005（差 6 个量级）**
   - 解析链：CAL-2 inline clipping JVP 局部 dot 缺陷 → D3（q 计数缩放）/D4（读出）缺陷（F0-IC 证实并修复）→ HORIZON_PROVENANCE 对象错配（旧 h16 Z16 归一 vs CALFIX Z256）→ F1 Phase2 FP32 FD 分辨限制。
   - 终态：b 坐标 SENS −0.31078 vs FD −0.34537 一致；但 **h256 全时域梯度仍未校准**（FULL_HORIZON_GRADIENT_UNRESOLVED_AFTER_FIX）→ 属于未解决而非已裁决。
6. **[OVERCLAIM→撤回] A0H15 第一轮"GPU 同进程位级一致"**
   - 实情：探针崩溃未完成比较；跨运行证据仅 H1/4 ~5e-11、H8 ~1e-6；第二轮入口自校验才确立 gap=0。README 已更正。
7. **[OVERCLAIM→撤回] A0H-1 "H≥4 逐元 gate 对任何伴随不可满足"**
   - 与 C|4 的 2/3 PASS 冲突；更正为"冻结 bf16 管线噪声地板超过冻结容差"（fp32 探针是强证据非全称证明）。
8. **[CONFLICT-口径并存] v07 的 r=0.5406（within-seed 去趋势 H8）vs pooled R²=0.0008（448 行无固定效应）**
   - REPORT_RED_FLAGS #1 裁定：不矛盾、口径不同，写作必须双报并保留 H8_CONFIRMATION=FAIL 总判定。
9. **[CONFLICT] GTTR 论文式声明 vs 证据（model-target alignment 审计）**
   - 25 claims：EXACT 2 / EMPIRICALLY_COVERED 16 / **NOT_SUPPORTED 4** / NOT_TESTED 4 / PROTOCOL_MISMATCH 1 → MISALIGNED_FOR_DEPLOYMENT_CLAIMS。

## B. 报告内更正（数字未变，判定不变）

10. **[ERRATUM] A0O Stage-2 校准符号**：driver ≤seq9 单侧分支取反误报 −0.250；离线修正 +0.942（`stage2_metrics_corrected.json`；原 runs/stage2_metrics.json 保留并注明）。
11. **[ERRATUM] A0T seq2 报告修正**：`actions_disagree` 浮现（2/18 状态两流 χ 符号相反——真实物理，非计算错误）；无计算改动。
12. **[ERRATUM] C1L 字段核对 E1–E9**：表头标签/定义混用（SE_of_norm、Table D 列名、Table C 统计量拆分、SNR 拆分）；恒等式全部成立，12 行 E 逐位不变。
13. **[HYGIENE] PhaseB S3 fd_rel=0.706 vs S0–S2 通过**：经 FD diagnostic 证明位级可复现（非 bug/非随机）——是真实的局部性边界，非文档错误，登记以防误读为缺陷。

## C. 结果前缺陷修复（透明留痕，无科学污染）

14. A0J implementation gate 首次 FAIL（`_clip` 漏调 `pe._step_once`，endpoint hash 分叉）→ 修复后 30/30；教训（生产复现行必要）已制度化。
15. A0K B_wc 臂事件排除缺陷：分析前自查发现并用 `a0k_rerun_bwc_v01.py` 重跑替换；events 哈希链未动。
16. A0L arm B `members["1"]` off-by-one：S2 smoke（零科学结果）阶段修复。
17. A0U launch-1 cuda:1 被占 → cuda:0（结果前，manifest seq 2 先记后跑）。
18. A0O 5 次结果前缺陷修复 launch（设备序/键映射/导入/加载/signature/A2 理论修正）→ GPU 4020s/1800s 超限全部归因于此，live_manifest seq 0–12 留痕。

## D. 缺失/不可验证制品

19. **[MISSING] `theta_protocol.py` 源码（6341B，SHA 4a127209…）**：2026-09-12 00:16–19:00 之间被删，全盘 48 副本无匹配；pyc 行为等价已验，但源永久缺失、删除原因未查明。**最高优先级遗留事件。**
20. **[MISSING] F0 旧 9 端点绝对 NLL**：只有 4 个对称差可恢复；不能反推 J(0,0)/单边收益/最优规则（CALFIX §1 明示）。
21. **[SUMMARY_ONLY] E3 系科学判定**：e3_mapping/e3_replication 的科学结论层为"未自动判定"（人工审核口径），制品只有技术 PASS 标志；E3C 的 DISCOVERY_COMPARISON_STATUS=UNAVAILABLE（4-seed bootstrap 退化）。
22. **[SUMMARY_ONLY] A0C 根目录 17 个预备包**：结果仅存在于各包 README/scripts（outputs/a0_fr_audit 为权威）；mv 链中间版本（v02 NO_MATCHED_STATES 等）无独立冻结 summary。
23. **[MISSING] A0G 审计 docx（docs/report_audit/FR_Report_A0G_Audit_20260915.docx）**：本次审计未机器解析 docx 内容，仅登记存在。
24. **[MISSING] C1 shadow t0 分辨**：所有 K≤32 低于 3SE 比值 2.8–2.9（UNRESOLVED 维持；C1L 不回溯改变该事实）。
25. **[MISSING] A0H15 验收集 A**：按 read-once 规则整批降级 DEV——Branch A 独立验收 NOT_COVERED（历史性空缺，不可补）。
26. **[MISSING] 计划中从未执行的实验**（文档提及但无制品）：v06 跨 partition 复现（A0F2 遗留 next step）、A0O Stage 3 独立 eval、A0H-2 正式计时（A0H-1 gate 后）、G4（A0G-lite 26M）、PhaseA 的 B/C/D/E 阶段（仅 A0+A1 执行）、F0 solver 正式运行（BLOCKED）。

## E. 文档卫生

27. [HYGIENE] `docs/E0_SMOKE_PASS_20260902.md` 与 `E0_smoke_PASS_20260902.md` 大小写不同的两份副本并存。
28. [HYGIENE] `docs/TASK00_σ«₧τÄ…`（GBK mojibake 文件名）与正常中文版 `TASK00_实现审计结论_v0.2.md` 并存（内容同源）。
29. [HYGIENE] `AGENTS.md` 的"当前允许/禁止"清单停留在 E0–E3-R 时代（09-02），与项目实际进度（已完成至 F2b）严重脱节——新会话若只读 AGENTS.md 会误判项目阶段。
30. [HYGIENE] 根目录 `r`、`rm`、`project_tree.txt` 为 0 字节残留；`research/a0h15/`、`research/analysis/` 为空目录；12 个 `outputs_fr_*.log` 散落根目录（部分已在审计轨迹披露）。
31. [HYGIENE] `docs/project_inventory/` 尺寸数字过时（09-12 的 9.3GB vs 现 71GB），文件数 1758 vs 现远多于此——标注"仅 code map 仍有效"。

## 计数汇总

- CONFLICT 类（含已裁决/未决）：9 项（A 组 1–9）
- ERRATUM/报告更正（判定不变）：4 项（B 组）
- 结果前修复留痕：5 项（C 组）
- MISSING / SUMMARY_ONLY / 未执行：8 项（D 组 19–26，其中 19 为最严重）
- 文档卫生：5 项（E 组）
- **合计 31 项登记；其中存在冲突或无法验证的结论 = 17 项（A 组 9 + D 组 8）。**
