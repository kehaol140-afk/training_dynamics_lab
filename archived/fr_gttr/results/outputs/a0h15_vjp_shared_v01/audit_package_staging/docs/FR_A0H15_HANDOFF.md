# FR A0H-1.5 交接 —— VJP 候选共享伴随

> 2026-09-15。上游：A0G（冻结）、A0H-1（伴随等价，原 FAIL 保留）、
> A0H-1.5 协议 `docs/FR_A0H15_PROTOCOL.md`（含事后附录 A 执行记录）。
> 本文回答核心问题并给出 A0H-2 小规模验收/计时的精确入口。

## 0. 核心问题与答案

**问：冻结的 BF16 前向响应打分器能否支撑候选共享反向打分器，在原始数值标准下
一致，且不预设数值 Hessian 自伴？**

1. **共享伴随已获得（是）**。w_0 只由公共基轨迹 + 终点 V_score 梯度导出，
   `ĥ_j = ⟨w_0, a_j⟩`；CPU 非对称算子合成 5/5 精确转置；GPU 上共享 ĥ 与冻结
   joint pass 前向 ĥ 同进程位级一致（确定性探针）。K 的线性性在 BF16 地板内
   成立（CHK-LIN）；SA≈IVJP 缺陷证明主导误差是 BF16 HVP 代理非自伴而非实现。
2. **原始数值标准：部分满足（如实分级）**。
   - H=1：逐元 gate PASS（max|Δ| ~5e-11，D1）。
   - H=4：逐元 gate PASS（max|Δ| 1.36e-6 ≤ 容差 1.84e-6，D1）。
   - H=8：逐元 gate FAIL——与冻结 C|8 原 FAIL 同一签名（BF16 HVP 噪声地板），
     非伴随回归；A0H-1 的 fp32 探针已证该地板对**任何**伴随不可满足。
3. **不预设自伴：满足**。共享伴随只用 K 的独立 VJP（三阶链 35/35 可行），
   未用 K=Kᵀ；CPU 非对称 K≠Kᵀ 测试 5/5 通过。
4. **独立决策保持：dev 级成立**。
   - BF16 共享伴随 vs 冻结前向：D1 全 H 排序/选择一致；降级批 dev-validation
     9 组合全 decision_match=True。
   - FP32-HVP 配置 vs 冻结 BF16 C：5 键全排序（4 候选 + no-op）9/9 完全一致。
   - **独立验收集验收：NOT_COVERED**——A 批在调试中暴露 accept runner 重建层
     bug，按协议 §1 读一次规则整批降为开发数据（
     `contract/acceptance_disposition_readonce.md`）。不追认为未见数据。
5. **计时（A0H-2 小规模）：NOT_COVERED**。§6/§7 触发条件要求独立验收集决策匹配；
   本轮只有 dev 级证据。预算未耗尽（822s/1800s），非资源原因。

## 1. 分支与逐项状态

| 项 | 状态 | 证据 |
|---|---|---|
| CPU 非对称算子共享伴随（≤1e-6） | PASS | `outputs/a0h15_vjp_shared_v01/cpu_test/test_a0h15_vjp.txt`（5/5 OK） |
| 封装器 forward 一致性（GPU 位级） | PASS | 同进程确定性探针（shared ĥ == 冻结 joint ĥ_fwd） |
| CHK-LIN 线性性（BF16 地板内） | PASS | `diagnostics/diag_2031_52_s0.json` diag.lin |
| CHK-SA / CHK-IVJP | PASS（探针可行）+ 发现：缺陷同源于 BF16 HVP 非自伴 | 同上 diag.sa / diag.ivjp |
| CHK-TELE 望远镜闭包 | FAIL（最差 8.9e-3 > atol_tele=1e-3；按 §3 仅为一致性诊断，不证明根因） | 同上 tele |
| 原始 gate H=1（BF16 共享伴随 vs 冻结前向） | PASS（~5e-11） | 同上 vjp_shared vs `outputs/a0h_adjoint_cost_v01/dev_check/unit_2031_52_s0.json` per_candidate |
| 原始 gate H=4 | PASS（1.36e-6 / 容差 1.84e-6） | 同上 |
| 原始 gate H=8 | FAIL（NUMERICAL_LIMIT，与冻结 C|8 同签名，不改判） | 同上 + A0H-1 fp32 探针 |
| 共享性 w_0 候选无关 | PASS | w_0 构造不读候选 x_k（代码审查 + 构造路径） |
| 决策保持（dev，BF16 配置） | PASS | D1 全 H + 降级批 dev-validation 9/9 decision_match |
| Branch A 独立验收 | NOT_COVERED（A 批降级） | `contract/acceptance_disposition_readonce.md` |
| Branch B1（FP32-HVP 同配置 fwd vs 伴随，atol32=1e-5/rtol32=1e-4） | PASS（3 单元 × 3 H，max 4.1e-10） | `diagnostics/branch_B1B2_summary.json`、`diag_*_fp32hvp_s0.json` |
| Branch B2（vs 冻结 BF16 C 决策级） | PASS（5 键排序 9/9 一致；逐元两式并报） | 同上 |
| Branch B3（不追认 BF16 等价） | PASS（遵守） | 原始 FAIL 未改 |
| §7 计时（A0H-2 小规模） | NOT_COVERED（触发条件未满足；非预算原因） | `contract/gpu_budget_log.txt`（822s/1800s） |
| 冻结源哈希 18 文件 | PASS 未变 | `contract/source_hashes_start.txt` 逐项复核 |
| A0H-1 审计包 SHA256 | PASS（3b237946…db6c93 一致） | `outputs/a0h_adjoint_cost_v01/a0h1_audit_package.tar.gz.sha256` |

## 2. 运行方式（复现/接续）

- 环境：conda env `fr_minimind`，`PYTHONPATH=research/src`，物理 GPU1 经
  `CUDA_VISIBLE_DEVICES=1`；torch 2.11.0；外部作业不可杀。
- 诊断 pass（BF16 全探针）：
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=research/src python research/scripts/a0h15_vjp_diagnostics_v01.py --device cuda:0 --states 2031/52`
- lean（acceptance 模式，跳过 CHK 探针）：加 `--lean`。
- Branch B（FP32-HVP 配置）：加 `--hvp-fp32`（7 个 `_hvp` 调用点换 nullcontext）。
- accept runner：`research/scripts/a0h15_accept_v01.py --device cuda:0`
  （注意：其 ACCEPT_STATES 现为**已降级**的 dev 单元，仅产 DEV_VALIDATION_ONLY）。
- CPU 测试：`python -m unittest research.tests.test_a0h15_vjp -v`。

## 3. A0H-2 建议入口（下一轮，需用户确认新验收批 A′）

1. 先验冻结一个新验收批 A′（≠ D、≠ 已降级批；协议修正案），再读一次。
2. 在 A′ 上跑 accept runner（--lean）验证 Branch A/B 决策 gate；
   通过 ⇒ 触发 §7 计时（矩阵 1 单元 × H{1,4,8} × M{1,2,4}，预热 1 + ≥3 交错重复；
   独占 gate util<10%，否则标 shared 不报加速结论）。
3. 生产伴随入口 = 公共基轨迹 + 终点 V_score 梯度 + 单次 backward + 候选梯度 +
   点积（`hhat_fwd_C` 与 `vjp_shared` 已由同一 pass 产出，计时脚本可直接复用
   `a0h15_vjp_diagnostics_v01._run_diagnostics_pass(..., full_diagnostics=False)`）。
4. 计时上限仍按本轮剩余预算或新预算另批。

## 4. 目录索引

- 诊断输出：`outputs/a0h15_vjp_shared_v01/diagnostics/`（diag_*_s0.json、
  diag_*_fp32hvp_s0.json、branch_B1B2_summary.json、summary.json、run_*.log）
- accept（dev-validation）：`outputs/a0h15_vjp_shared_v01/accept/*.json`；
  v9 无效输出归档 `accept_v9_invalid_archived/`
- 合同与合规：`outputs/a0h15_vjp_shared_v01/contract/`
  （gpu_budget_log.txt、acceptance_disposition_readonce.md、
  source_hashes_start.txt、protocol_hashes.txt）
- 新代码：`research/scripts/a0h15_vjp_diagnostics_v01.py`（主 pass）、
  `research/scripts/a0h15_accept_v01.py`、`research/scripts/a0h15_analyze_v01.py`、
  `research/tests/test_a0h15_vjp.py`
- 备注：`research/a0h15/` 为空目录（原计划的代码位置；实际按仓库惯例落在
  research/scripts 与 research/tests，特此如实记录）。
- A0H-1.5 审计包：`outputs/a0h15_vjp_shared_v01/a0h15_audit_package.tar.gz`
  （manifest 不含自身；tar SHA256 单独记录于同名 `.sha256` 文件；
  已验证 listable/unpackable；无权重/梯度/检查点）。
