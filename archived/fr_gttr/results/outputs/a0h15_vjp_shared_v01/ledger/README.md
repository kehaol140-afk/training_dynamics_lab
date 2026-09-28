# comparison_ledger 说明（2026-09-15，第二轮）

生成器：`research/scripts/a0h15_comparison_ledger_v01.py`（纯离线，无 GPU）。
输出：`comparison_ledger.csv`（180 行）、`comparison_ledger.json`。

## 每对比较的两端（显式）

| run_id | reference | candidate | 语义 |
|---|---|---|---|
| a0h1_frozen_dev_check | frozen_BF16_forward | old_analytic_adjoint_BF16 | 冻结单元内原 gate（只读复核） |
| a0h15_rerun_bf16_d1 | frozen_BF16_forward | rerun_BF16_forward | **前向复现**（跨运行，a0h15 pass 的 hhat_fwd_C vs 冻结 per_candidate） |
| a0h15_shared_bf16_d1 | rerun_BF16_forward | BF16_IVJP_shared | **伴随等价**（同 pass 内共享伴随 vs 前向，原 gate） |
| a0h15_shared_vs_frozenfwd_d1 | frozen_BF16_forward | BF16_IVJP_shared | 跨运行交叉对（历史"位级"声明所指的 pair） |
| a0h15_fp32hvp_b1 | FP32_HVP_forward | FP32_HVP_adjoint_shared | Branch B1 同配置 gate（atol32/rtol32） |
| a0h15_fp32hvp_b2_elementwise | frozen_BF16_forward | FP32_HVP_adjoint_shared | B2 逐元并报表（不同配置；B2 判定以决策级为准） |
| a0h15_devval_demotedA | rerun_BF16_forward | BF16_IVJP_shared | 降级批 dev-validation（无验收身份） |

**前向复现 ≠ 伴随等价**，分别由 run_id 区分，不混写。

## "位级一致"声明的更正（§2.3 冲突定位）

- 第一轮交接中"GPU 同进程共享 ĥ 与冻结 joint pass 前向 ĥ 位级一致（确定性探针）"
  的措辞**过强，现更正**：探针脚本 `/tmp/test_determinism.py` 在打印 a0h15 一侧前
  即因 `vjp_shared` 键类型（str vs int）崩溃，**从未完成比较**；该声明无证据支持。
- 实际可用的跨运行证据（本台账 a0h15_rerun_bf16_d1 / a0h15_shared_vs_frozenfwd_d1，
  磁盘制品复核）：H=1、H=4 一致到 ~5e-11（近位级）；**H=8 仅一致到 ~1e-6**
  （BF16 跨运行噪声地板），不是位级。
- 因此"H8 单元伴随 vs 前向 gate FAIL"与"H8 跨运行一致到 1e-6"**无矛盾**：
  前者是同 pass 内 |fwd−adj| ~1.9e-6..2.7e-6（超出 atol=1e-6 量级的 gate），
  后者是前向自身的跨运行重现误差。两者描述不同比较对。

## 历史过强结论的更正（新文档口径）

- ~~"H≥4 逐元 gate 对任何伴随不可满足"~~ → **更正**：冻结单元中 H=4 存在
  PASS（D1、D3 的 C|4 通过；本台账 a0h1_frozen_dev_check 中 H=4 全 PASS）。
  不可满足的说法**仅限 H=8**（在本实现、本单元、本 gate 下；且"任何伴随"的
  全称依据只是 fp32 探针显示的前向自身地板，不是对所有可能实现的证明）。
- ~~"100% BF16 地板、排除所有实现错误"~~ → **更正**：地板证据强
  （FP32-HVP 同配置闭合 4.1e-10 vs BF16 ~2e-6），但"排除所有实现错误"
  是过强表述；正确说法是"残差量级与 BF16 HVP 代理非自伴缺陷一致，
  且无伴随实现错误的独立证据"。
- 原始 FAIL 与数值全部保留未改（见 a0h1_frozen_dev_check 行）。

## 汇总（verdict 计数）

| run_id | PASS | FAIL |
|---|---|---|
| a0h1_frozen_dev_check | 30 | 6（H8：D1 j1-3、D2 j1-3，均为原始 FAIL） |
| a0h15_rerun_bf16_d1 | 12 | 0 |
| a0h15_shared_bf16_d1 | 9 | 3（H8 j1-3） |
| a0h15_shared_vs_frozenfwd_d1 | 9 | 3（同上） |
| a0h15_fp32hvp_b1 | 36 | 0 |
| a0h15_fp32hvp_b2_elementwise | 16 | 20（不同配置下的逐元差，预期内） |
| a0h15_devval_demotedA | 27 | 9（降级批，无验收身份） |
