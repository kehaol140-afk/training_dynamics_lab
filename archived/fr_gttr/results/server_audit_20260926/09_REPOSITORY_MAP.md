# 09 — 仓库地图（目录结构与作用）

```
/home/b205/projects/                      ← 审计父工作区（72G）
├── 86153/llm-review-gain/                ← 空目录（8K，无内容，历史残留）
├── FULL_SERVER_AUDIT_20260926/           ← 本次审计输出（仅文本）
└── fluctuation_response/                 ← 主研究仓库（Git master，3 commits 未 push）
    ├── README.md                         ← 项目导览（三条研究线 + 冻结纪律摘要）
    ├── AGENTS.md                         ← 会话约束（允许 E0-E3-R；禁止 module sweep/controller/V2A/V2B/改阈值过关）
    ├── CONFIRMATORY_PREREGISTRATION_V07.md / ENVIRONMENT_V07.txt / DATASET_SHA256.txt / MINIMIND_SOURCE_SHA256.txt
    ├── RUN_ORDER_V03..V07.md, RUN_A0_V01.md, RUN_E4B{0,1}_V08.md, RUN_E4B_DYNAMIC_V08.md
    │     ← 各版本执行顺序与 E4B 运行说明（合同先于 GPU 冻结的历史痕迹）
    ├── MANUAL_REVIEW_V03..V06.md, V06_AUDIT_REMEDIATION.md, V07_{IMPLEMENTATION_AUDIT,PREREG_LOCK_REVISION2_AUDIT}.md
    │     ← 框架 v0.3→v0.7 的人工审查/整改链
    ├── CODEX_AUDIT_{PROMPT_V06,V06}.md, CODEX_AUDIT_ONLY_PROMPT.md, LOCAL_AUDIT_CHECKLIST.md
    │     ← 外部审计提示词与结果（25K）
    ├── FR_FRAMEWORK_README.md, 人工审查导读_v0.2.md
    │
    ├── 【根目录 GTTR 协议文档（09-22→09-26 活跃）】
    │   FR_B0_Mechanism_Validation_Claude_Code.md        （B0 机制验证协议）
    │   FR_C0_Geometry_Tracking_Validation_... .md       （C0）
    │   FR_C0X_C1_GTTR_Offline_and_Probe_Validation_...  （C0X+C1）
    │   FR_C1L_GTTR_LateAnchor_DataFirst_... .md         （C1-Late）
    │   FR_GTTR_Experiment_Roadmap_and_Next_Run_v1.md    （GTTR A-E 总路线）
    │   FR_GTTR_Model_Target_Alignment_Audit_... .md
    │   FR_GTTR_PhaseA_Native_Target_Calibration_... .md （A0+A1）
    │   FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md + FR_PhaseB_SharedGPU_TokenPrimary_Amendment_v1.md
    │   FR_PhaseB_FD_Local_Operator_Diagnostic_v1.md
    │   FR_GTTR_D1_Direct_Update_Pilot_... .md
    │   FR_GTTR_R1_Model_To_Rule_Identification_v1.md
    │   FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md
    │   FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md
    │   FR_GTTR_F0_Full_Horizon_Terminal_Rule_Solver_v1.md（426 行）
    │   FR_GTTR_F0_CALFIX_Revalidation_v1.md
    │   （另 GTTR_F0_Implementation_Consistency_Package_v1.zip 16K）
    │
    ├── 【根目录 A0C 时代工作包（09-09→09-10，17 个，legacy）】
    │   FR_A0_Audit_Package_v01/  FR_AUDIT_V02_A0C_RESIDUAL/
    │   FR_A0C_{AdamW_Adaptive_Audit, AdamW_Theory_Response_v01/v03/v03_fix,
    │           AdamW_m_v_Response_v01, AdamW_m_v_Theory_v02..v06(+v06_gradient.py),
    │           Momentum_Kernel_Audit, Momentum_Theory_Fit, Residual_FR_Audit,
    │           Optimizer_State_Code_Audit, Optimizer_State_Export, State_Audit}_v01/
    │     ← 每个含 README+scripts；权威结果在 outputs/a0_fr_audit/
    │
    ├── 【根目录杂项】
    │   FR_REPORT_AUDIT_BUNDLE_20260912/ + .tar.gz (1.8M)   ← A0 系报告事实审计
    │   FR_paper_figures_v01.tar.gz (1.6M) / FR_v06_code_audit_snapshot.tar.gz (142K)
    │   outputs_fr_*.log（12 个根级 redirect 日志，c1/c1late/calfix/cb1/f1/r1/r2b 各轮）
    │   r, rm, project_tree.txt（空占位）；opencode.json（本地工具配置）
    │
    ├── docs/（2.6M）
    │   ├── FR_PROJECT_STATUS.md（1443 行，§1–§49 —— 项目状态唯一权威源）
    │   ├── FR_EXPERIMENT_INDEX.md（实验总索引表）
    │   ├── FR_GENERALIZATION_STATUS.md / FR_GENERALIZATION_HANDOFF.md（G 线）
    │   ├── FR_A0G/A0H/A0H15_{HANDOFF,PROTOCOL,AMENDMENT}_*.md、FR_ALGORITHM_ROADMAP_POST_{A0H20,A0I}.md
    │   ├── FR_RESEARCH_RETROSPECTIVE_A0G_A0U.md（401 行英文回溯）
    │   ├── CLEANUP_G1C_LARGE_FILES_20260925.md / G1C_FEASIBILITY_SUMMARY_20260925.md
    │   ├── E0_SMOKE_PASS_20260902.md（两份大小写不同副本）
    │   ├── TASK00_*.md（v0.3 审计；注意一份文件名 GBK 乱码 + 一份正常中文）
    │   ├── project_inventory/（09-12 盘点：ARTIFACT_REGISTRY.csv 69K、EXPERIMENT_CODE_MAP、
    │   │     PROJECT_TREE、DUPLICATE_AND_ORPHAN_AUDIT、SAFE_REORGANIZATION_PLAN、STORAGE_REPORT）
    │   ├── report_audit/FR_Report_A0G_Audit_20260915.docx（1.4M）
    │   ├── specs/（2 份 A0C 设计 spec）/ superpowers/{plans×6, specs×2}
    │   └── Transformer_AdamW_..._ResearchManual_v2.pdf + ResearchPlan_AI_v2.md
    │
    ├── research/（9.0M，Git tracked 核心）
    │   ├── CHANGE_AUDIT.md（v0.6 变更审计）
    │   ├── src/fr/（68 .py + theta_protocol.pyc）   ← 冻结分析模块库（见 05_A）
    │   ├── scripts/（~150 runner + README_SCRIPTS_V03..V06）
    │   ├── configs/（20 JSON + README_V03..V06）
    │   ├── tests/（57 test_*.py）
    │   ├── adapters/（minimind_adapter_v02 + template）
    │   ├── reports/（TASK00_SOURCE_AUDIT）/ analysis/（空）/ a0h15/（空）
    │   └── a0h15/（空目录，历史残留）
    │
    ├── outputs/（71G，55 个实验目录 —— 见 01/02 索引与 08 磁盘）
    │   ├── e0_replay / e1_fluctuation / stage_c_v04 / e3_mapping / e3_replication
    │   ├── e3_confirmatory/v07/（56 态 bank —— 唯一认证输入源）
    │   ├── e4a_lowdim_state / e4b_dynamic_state
    │   ├── a0_fr_audit/（33 子目录：v01–v08、mv_theory*、H1、A0C/A0D/A0E/A0E2/A0F/A0F2、paper_figures）
    │   ├── fr_decision_utility / a0h_* / a0h15_* / a0h_corrected / a0i / a0j / a0k / a0l / a0m /
    │   │   a0n / a0o / a0p / a0q / a0s / a0t / a0t_extra / a0u
    │   ├── generalization_g1a / g1b / g1c_{external,checkpoint_recheck,feasibility} / g2 / g3
    │   ├── fr_b0 / fr_b05 / fr_b06 / fr_c0_{geometry_tracking(中止),gttr} / fr_c0x / fr_c1 / fr_c1_late
    │   └── fr_gttr_{model_target_alignment,phase_a,d1,r1,r2b,cb1,f0_terminal,f0_calfix,f0_ic,f1,f2a,f2b}
    │       + fr_phase_b_{gttr_action_scoring_v03,fd_diagnostic}
    │
    ├── minimind/（1.2G，非 git，未修改上游副本 —— 见 06）
    ├── checkpoints/ data/ logs/（空脚手架 + 3 小日志）
    ├── baseline_manifest/（8 文件环境/数据/代码哈希）
    ├── recovery_candidates/theta_protocol/（源码恢复取证 4 文件）
    ├── tools/gttr_f0_ic_v1/（F0 IC 独立校验包）
    ├── _fr_framework_backups/（6 个零提交备份仓库）
    ├── _fr_framework_incoming/ + _fr_v02..v06_incoming/（框架传入包，含零提交 .git×6）
    ├── .claude/ .codex/ .agents/（本地工具配置）
    └── .git/（5.6M，见 07）
```

## 关键文件索引（按用途）

| 要做什么 | 先读 |
|---|---|
| 恢复项目上下文 | `docs/FR_PROJECT_STATUS.md` 头部更新日志 + §13b 冻结规则 + §14 禁引清单 |
| 查任何实验 | `docs/FR_EXPERIMENT_INDEX.md` → 对应 outputs/<dir>/summary.json |
| 查实验↔代码↔config 接线 | `docs/project_inventory/EXPERIMENT_CODE_MAP.md`（09-12 版，覆盖 E/A0 早期） |
| 论文数字措辞 | `FR_REPORT_AUDIT_BUNDLE_20260912/REPORT_RED_FLAGS.md` |
| 跑任何 A0 系实验前 | 确认 `research/src/fr/theta_protocol.pyc` 存在（STATUS §13） |
| 复现环境 | `ENVIRONMENT_V07.txt` + `baseline_manifest/` |
