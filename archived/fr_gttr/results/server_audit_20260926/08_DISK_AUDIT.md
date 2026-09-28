# 08 — 磁盘审计（只读）

## 1. 文件系统

```
/dev/sdd  1007G  710G used  247G avail  75%  挂载 /
```

## 2. 总量

- `/home/b205/projects` = **72G**（其中 fluctuation_response 72G；86153/ 8K 空目录）

## 3. fluctuation_response 一级目录（按大小）

| 目录 | 大小 | 说明 |
|---|---|---|
| **outputs/** | **71G** | 全部实验制品（99% 的占用） |
| minimind/ | 1.2G | 上游副本（语料 1.2G） |
| FR_REPORT_AUDIT_BUNDLE_20260912/ | 11M(+tar 1.8M) | 报告审计 bundle |
| research/ | 9.0M | 代码/测试/配置 |
| .git/ | 5.6M | 主仓库 |
| docs/ | 2.6M | 文档（含 PDF 手册、report_audit docx 1.4M） |
| _fr_framework_backups/ + 6×_fr_*_incoming/ | ~4.7M | 历史框架快照/传入包 |
| 其余（FR_A0C_* 17 目录、recovery_candidates、baseline_manifest、tools、logs、checkpoints、data） | <200K each | 小型工作区/脚手架 |
| 空占位残留：`r`（0B）、`rm`（0B）、`project_tree.txt`（0B） | 0 | gitignore 已点名 |

## 4. outputs/ 各实验目录（全部 55 项，按大小）

```
14G  generalization_g1b_100m_v01          ← 9×1.2G state bank
14G  a0k_fresh_response_closed_loop_v01   ← 72 事件完整快照
7.5G a0n_phase_scheduler_pilot_v01
5.3G generalization_g2_data_distribution_v01 (含 pretrain_hq.jsonl 1.6G)
4.6G e3_confirmatory                      ← v07 56 态 bank【绝不删除】
4.5G fr_gttr_r2b_solved_update_bridge_v01
4.5G a0o_lr_response_validation_v01       ← A0S/A0T 冻结输入
3.7G generalization_g3_clipping_intervention_v01
3.7G generalization_g1a_26m_v01
1.9G fr_gttr_d1_direct_readout_v01
1.8G e3_replication
1.2G a0_fr_audit                          ← A0 系结论层【绝不删除】(含 mv gradient_g.pt 1.1G)
776M fr_gttr_phase_a_native_calibration_v01
766M fr_decision_utility_v01
574M fr_phase_b_gttr_action_scoring_v03
571M fr_gttr_f0_calfix_revalidation_v01
508M fr_c1_gttr_probe_validation_v01
447M e3_mapping
381M a0l_response_mixture_feasibility_v01
254M fr_c1_late_completion_v01
144M a0j_cached_future_sensitivity_v01
128M fr_c0_gttr_validation_v01
87M  stage_c_v04 / 86M e1_fluctuation / 75M e0_replay
15M  generalization_g1c_feasibility_v01（61G 大文件已于 09-25 删除）
9.6M fr_gttr_cb1 / 9.3M g1c_checkpoint_recheck / 5.3M fr_b0 / 4.1M e4b
3.1M fr_gttr_f0_terminal_rule_solver / 2.2M fr_b06 / 1.6M fr_b05
1.5M a0h15 / 1.2M e4a / 1.2M a0h_adjoint
<1M  其余 a0i/a0j(分析件)/a0p…a0u/fr_c0x/fr_f0_ic/fr_f1/fr_f2a/fr_f2b/fr_gttr_r1/g1c_external 等
8.0K fr_c0_geometry_tracking_validation_v01（中止占位）
```

## 5. 大文件（>1G，共 12 个）

| 文件 | 大小 |
|---|---|
| outputs/generalization_g2_.../data/pretrain_hq.jsonl | 1.6G |
| outputs/a0_fr_audit/mv_theory_final/gradient_g.pt | 1.1G |
| outputs/generalization_g1b_100m_v01/bank/seed_204{2,3,4}/states/state_step_{026,064,102}.pt | 9×1.2G |
| minimind/dataset/pretrain_t2t_mini.jsonl | 1.2G |

>100MB 文件共 **79 个**（全部 .pt/.jsonl，全在 ignored outputs/ 或 minimind 下；无 tracked 大文件）。

## 6. checkpoints/ data/ logs/

- `checkpoints/{native,full_state}/`、`data/{eval,probe,raw}/`：**空脚手架**（各 4K）——真实制品都在 outputs/。
- `logs/`：3 个小文件（a0e2_bh_fixed_v02.log 3.2K；e4b1_q_rotation_v08.log 21K + pid）。

## 7. 未来清理建议（只建议，不执行）

### 绝对不应该删除（结论层 + 唯一输入源，合计 <10G）
- `outputs/e3_confirmatory/v07/`（4.6G，56 态 bank——一切 A0 实验唯一输入，禁再生成）
- `outputs/a0_fr_audit/` 结论件（各 summary/report/CSV/MD）
- 全部 `SHA256SUMS*`、`contract/`、`manifest`、`*_audit.tar.gz`、`PREREG*`、`tools/`、`recovery_candidates/`
- `outputs/a0o_.../`（A0S/A0T 冻结输入 cells）、`outputs/fr_gttr_*` 与 `fr_c0/c1*` 的全部 verdict/summary 件
- `minimind/`（上游唯一副本）+ `research/` + `.git/` + `docs/`

### 可安全归档（移至冷盘/对象存储；先核对各目录 SHA256SUMS 完整）
| 候选 | 大小 | 依据 |
|---|---|---|
| G1b bank 9×1.2G state .pt（保留 q/manifest/SHA） | ~11G | SHA256SUMS 有哈希；G1b 已冻结归档核验 |
| A0K 72 事件 .pt 快照 | ~14G | counterfactual replay 理论可再生成（但昂贵——归档优于删除） |
| A0N runs 大文件 | ~7G | 结论件保留即可 |
| pretrain_hq.jsonl（G2） | 1.6G | 官方可再取（有 revision+SHA） |
| e3_replication 大件、mv_theory gradient_g.pt | ~2.5G | 结论在 summary；gradient_g 仅复算用 |
| 合计可释放 | **~36G** | |

### 可重新生成（如需要）
- G1a/G2/G3 bank（重训可再生成，但会改变时间戳/不可位级等同——不建议删）
- 一切 __pycache__、*.log、空占位（r/rm/project_tree.txt）

### 已发生的清理记录
- G1c feasibility 61G（8 个 .pt）已于 2026-09-25 删除（docs/CLEANUP_G1C_LARGE_FILES_20260925.md，SHA256 锚点保留，commit 591b4ab）。
