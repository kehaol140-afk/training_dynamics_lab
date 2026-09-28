# 实验—代码—结果映射（EXPERIMENT_CODE_MAP）

生成：2026-09-12 · 只读静态映射（import / input_hashes / 目录约定）。环境：`PYTHONPATH=research/src`，conda env `fr_minimind`。

| 实验 | runner（research/scripts/） | src module（research/src/fr/） | tests（research/tests/） | 输入 | 输出目录（outputs/a0_fr_audit/） | 依赖的旧实验 | 最终状态 | 冻结 |
|---|---|---|---|---|---|---|---|---|
| **v0.7** bank | `e3c_prepare_state_bank_v07.py`、`e3c_multiseed_theta_v07.py`、`e3c_extract_locked_confirmation_v07.py`、`e3c_analyze_confirmatory_v07.py`、`scientific_preflight_v07.py` | `full_state.py`、`batch_manifest.py`、`stream_rng.py`、`perturbation.py`、`direction.py`、`hashing.py`、`rng.py`、`native_schedule.py`、`native_adamw_guard.py`、`minimind_v03_runtime.py`、`stage_c_runtime.py`、`theta_protocol(.pyc)`、`confirmatory_stats.py`、`mapping_stats.py` | `test_v07_prereg_lock.py`、`test_v07_recovery_extractor.py`、`test_full_state_roundtrip.py`、`test_batch_manifest.py`、`test_perturbation.py`、`test_rng_roundtrip.py`、`test_stream_rng.py`、`test_theta_protocol.py`、`test_hashing.py`、`test_native_schedule.py`、`test_direction.py` | 8 confirmation seed × 7 step；rho=0.25；pulse-before-clip 协议 | `outputs/e3_confirmatory/v07/`（bank 本体，非 a0_fr_audit） | 无（树根） | COMPLETE（prereg 锁定） | **是**（协议+bank，禁改） |
| **A0C** | `a0c_adamw_response_propagation_v01.py` | `a0c_adamw_response_propagation_v01.py`（**指标唯一来源** `_metric`/`relative_error`） | `test_a0c_adamw_response_propagation_v01.py` | v0.7 锁定 3 态（seed2037/46、seed2030/46、seed2037/82）× H{1,2,4,8} × 8 E2 stream；H1/mv 旧制品 | `response_propagation_v01/` | v0.7、H1、mv_theory_consistent | COMPLETE（H>1：A 6.7582/B 0.2053/C 0.0218656，C 9/9 行胜） | **是** |
| **mv_theory_consistent** | `a0c_adamw_mv_consistent.py` | `a0c_adamw_mv_consistent.py` | `test_a0c_adamw_mv_consistent.py` | v0.7 bank 448 行（8 seed × 7 step × H grid） | `mv_theory_consistent/` | v0.7；取代 `mv_theory_final`（旧） | COMPLETE（v 路径 0/8 seed、0/8 horizon 改善；v 反常属 frozen v 假象） | **是** |
| **H1** | `a0c_h1_adamw_jacobian_v01.py` | `a0c_h1_adamw_jacobian_v01.py` | `test_a0c_h1_adamw_jacobian_v01.py` | 同 A0C 3 态 × ε 扫描 | `H1_adamw_jacobian_v01/` | v0.7 | COMPLETE（ε/4 rel err 0.000146–0.018989；确立 H=1 局部理论） | **是** |
| **A0D** | `a0d_epsilon_convergence_v01.py` | `a0d_epsilon_convergence_v01.py` | `test_a0d_epsilon_convergence_v01.py` | A0C 3 态 × H{2,4,8} × ε{1,1/2,1/4} | `response_propagation_v02_epsilon/` | A0C | COMPLETE（ε/4 池化 mre 0.0045242；无需记忆核） | **是** |
| **A0E old** | `a0e_curvature_decomposition_v01.py` | `a0e_curvature_decomposition_v01.py` | `test_a0e_curvature_decomposition_v01.py` | 同 A0C | `curvature_decomposition_v01/` | A0C | **SUPERSEDED**（BH 初始 pulse 跳过 J_clip 的 bug；gain_H<0 结论作废） | 是（仅存档，**禁止重跑/引用**） |
| **A0E2** | `a0e2_curvature_decomposition_v02_bh_fixed.py` | `a0e2_curvature_decomposition_v02_bh_fixed.py` | `test_a0e2_curvature_decomposition_v02_bh_fixed.py`（14 测试） | 同 A0C + 冻结 A0C 响应 + 冻结 A0E BH_old 复现校验；**依赖 `theta_protocol.pyc`** | `curvature_decomposition_v02_bh_fixed/` | A0C、A0E（复现校验） | COMPLETE（H1 恒等精确 0；raw curvature 正贡献 +0.068627） | **是** |
| **A0F** | `a0f_cross_state_generalization_v01.py` | `a0f_cross_state_generalization_v01.py` | `test_a0f_cross_state_generalization_v01.py` | v0.7 bank 随机 32 态 × H{1,2,4,8} | `cross_state_generalization_v01/` | A0C | COMPLETE（96/128 胜、0 负、mean +0.124154） | **是** |
| **A0F2** | `a0f2_cross_partition_extension_v01.py` | `a0f2_cross_partition_extension_v01.py` | `test_a0f2_cross_partition_extension_v01.py` | v0.7 全 56 态 × H{1,2,4,8} | `cross_partition_extension_v01/` | A0F（共享行逐位 crosscheck） | COMPLETE（168/224 胜、0 负、7/7 stage 正） | **是** |
| **decision_utility_v01** | —（未在本仓库发现） | — | — | — | —（未发现任何文件/进程写入） | （将依赖）A0C/A0F2 冻结结论 | **RUNNING（外部）**：由另一个 Codex 会话规划/运行 validation-loss / data-reweighting 实验；本仓库内零文件 | 否 |

## 关键说明

1. **theta_protocol.pyc 是全链枢纽**：v0.7 replay → A0C → A0E2/A0F/A0F2 的 runner 全部经 `run_theta_branch` 间接依赖。文件缺失事件与恢复细节见 `recovery_candidates/theta_protocol/THETA_PROTOCOL_RECOVERY_REPORT.md`。
2. **decision_utility_v01 零痕迹**：全仓库 grep（py/md/json）无匹配，outputs 无新目录，ps 无 FR 相关 python 进程（仅其他项目的 FDA_INIT 任务在跑）。因此本轮整理**不会与该实验发生路径冲突**；但按任务要求，凡涉及 research/、outputs/ 的任何实际变更仍须等其完全结束。
3. 指标铁律：所有 `_metric`/`relative_error` 必须 `from fr.a0c_adamw_response_propagation_v01 import …`（A0E2 亦遵守）；违反即破坏冻结可比性。
4. 旧 `mv_theory_v02…v06` 系列与根目录 `FR_A0C_*` 工作区不在上表：前者已被 mv_theory_consistent 取代，后者为旧轮次一次性工作区（见 DUPLICATE_AND_ORPHAN_AUDIT.md）。
