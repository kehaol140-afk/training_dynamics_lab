# FR_A0H-1 交接 —— 反向伴随等价性与响应计算成本

日期：2026-09-14。合同与容差见 [FR_A0H_PROTOCOL.md](FR_A0H_PROTOCOL.md)（§1–§5 于执行前冻结；§6 为执行记录）。
本文件是 A0H-1 的交付判定与下一步交接；**未启动 A0H-2、v06、新种子 formal 实验或任何新训练**。

## 0. 交付判定总览

| 项目 | 判定 | 说明 |
|---|---|---|
| A0H-0 离线审计（前序） | PASS | `outputs/a0h_adjoint_cost_v01/offline_audit/`（18 文件，Rule B 全局 λ=0.05 确认） |
| 离线比较是否支持决策价值 | **是** | C 前向切线在 3 dev 状态 × H∈{1,4,8} 上给出非平凡候选排序，离线审计未发现问题（详见 §2） |
| CPU 合成验证（伴随 vs 冻结前向的转置） | PASS | 逐通道中心差分 1e-6；恒等式 1e-10；4/4 测试 |
| GPU dev 等价（逐元，冻结容差 atol=1e-6/rtol=1e-4） | **FAIL（NUMERICAL_LIMIT，已定位）** | H=1 与 B 全精确；C|4 2/3 状态通过；C|8 1/3 通过；**失败根因是冻结 bf16 HVP 代理自身的噪声地板超过冻结容差，不是伴随实现错误**（证明见 §3） |
| GPU dev 等价（决策级：排序/选择/no-op） | **PASS** | 3 状态 × {B,C} × H∈{1,4,8} 全部一致，no-op 恒 0 |
| 计时（§4 矩阵） | **NOT_RUN** | 协议 gate：等价 PASS 后才开始；逐元 gate FAIL ⇒ 本轮不计时 |
| A0H-1 总体 | **PARTIAL** | 等价与成本两半各交付一半：等价性已证（实现精确 + 决策一致），成本未测 |

## 1. 证据路径

- 协议与执行记录：`docs/FR_A0H_PROTOCOL.md`（§6）
- CPU 测试：`research/tests/test_a0h_adjoint_v01.py`（运行：
  `python -m unittest research.tests.test_a0h_adjoint_v01 -v`，4/4 OK）
- 伴随模块：`research/src/fr/a0h_adjoint_v01.py`（只含冻结算子的转置，无新前向模型）
- GPU runner：`research/scripts/a0h_adjoint_dev_check_v01.py`
  （命令：`CUDA_VISIBLE_DEVICES=1 PYTHONPATH=research/src python research/scripts/a0h_adjoint_dev_check_v01.py --device cuda:0 --budget-seconds 1200 --probe-fp32-hvp`）
- GPU 等价单元报告：`outputs/a0h_adjoint_cost_v01/dev_check/unit_{2031_52,2034_64,2036_76}_s0.json`、`dev_check_summary.json`
- fp32 诊断探针：`outputs/a0h_adjoint_cost_v01/dev_check/fp32_hvp_probe_{2031_52,2034_64,2036_76}_s0.json`
- 汇总分析：`outputs/a0h_adjoint_cost_v01/dev_check/fp32_hvp_probe_analysis.json`
- A0H-0 离线审计：`outputs/a0h_adjoint_cost_v01/offline_audit/`

## 2. 离线比较是否支持决策价值

支持。A0G 已冻结的 dev λ 扫描与 Level 3 证据（本任务全程只读）表明完整切线 C 的
ĥ 在 4 候选间产生稳定区分（本轮 runner 复算的前向 ĥ 与冻结 runner 位级一致，
再次独立确认），且离线审计（18 项 PASS）确认 λ/Rule B/分组/拆分/读出索引冻结无漂移。
伴随路径的决策级等价（§3）进一步表明：用伴随替换前向后，排序/选择/no-op 不变。

## 3. 伴随是否与完整前向切线一致

分三层回答：

1. **实现是精确的（已证明）。** fp32-HVP 探针下，伴随 ĥ 对 fp32-HVP 前向参照的端到端
   max|Δĥ| = **4.1e-10**（H=8，含 7 次反馈 HVP；H=1 与 B 路径 ≤1e-10）。
   CPU 合成：对冻结前向每输入通道的中心差分 VJP 全部 ≤1e-6（多数 ~1e-12），
   clip 转置内积恒等式与两步复合恒等式 ≤1e-10。
2. **bf16 逐元 gate 在 H≥4 对任何伴随都不可满足（已定位，非实现错误）。**
   冻结 bf16 前向相对**它自己的 fp32-HVP 版本** max|Δĥ| = 1.43e-5（H=4）、5.78e-5（H=8），
   大于冻结容差（≈1e-6..2e-6）。根因：冻结管线 bf16 math-SDPA HVP 代理在真实方向上的
   转置不对称性相对残差 2.4e-4..7.3e-2（fp32 下 3.2e-8..6.0e-7；随机方向 2.3e-3..2.3e-2）。
   前向自身噪声地板已超过容差 ⇒ 无论伴随如何实现，H≥4 逐元比较必失败。
   实测 bf16 伴随 vs bf16 前向 max|Δĥ| ≤6.2e-6（小于上述地板，因两者共享同一例程部分相关）。
3. **决策级完全一致。** 3 状态 × {B,C} × H∈{1,4,8}：排序、选择（argmin，tie 小 j）、
   no-op（恒 0）全部一致；前向复演位级一致（Δ=0）。

协议要求的不满足处置已执行：记 NUMERICAL_LIMIT/FAIL 并定位原因；**未放宽冻结容差**，
**未参照伴随误差倒推**。

## 4. 伴随是否更快

**本轮无法作答（计时未执行）。** 协议 §4 gate 要求逐元等价 PASS 后才开始计时；
gate FAIL ⇒ 计时 NOT_RUN（非 WAITING_FOR_GPU；GPU 期间为共享使用，约 6 分钟，
远低于 30 分钟上限，未杀外部任务）。

附带观察（**非正式计时，不构成加速结论**）：runner 日志中冻结前向 pass ≈22–25 s/单元，
联合 pass（前向复演 + 快照 + 全部伴随行走 + 诊断）≈28–30 s/单元。
解析预期（未验证）：伴随把每步 M 个 HVP 换为单次反向行走 ((H−1) 次 HVP，M 无关)，
H=4/M=4 时 HVP 数 12→3，应显著低于 C 前向；与真实回放（A0G 测得 H=4/M=4 2.65 s）的
相对关系只能由 A0H-2 的正式计时矩阵回答。

## 5. 保留的 caveats（不弱化）

- H=4 为 primary horizon 的 C-vs-B 区分仍是 A0G 的不确定性结论（本任务只读沿用）。
- H=8 为辅助 horizon（协议冻结）。
- dev 证据 3 个独立训练种子；formal 4 种子（2038–2041）本轮未动。
- 条件前瞻（conditional lookahead）：未来流给定，非 current-state-only forecast。
- 混合后端：冻结管线 bf16 autocast + math-SDPA HVP 代理；本轮发现其 H≥4 逐元噪声地板
  > 冻结容差（§3.2），这本身是 A0G 冻结 pipeline 的一个性质记录。
- GPU 等价在共享 GPU 上进行（外部任务并行）；数值结论不受影响（同进程内确定性已验证：
  前向复演 Δ=0、快照重算 HVP 相对差 = 0），正式计时仍需 util<10% 独占 gate。

## 6. 下一步（不自动启动）

### A0H-2 候选设计（唯一候选；针对本轮定位的 gate 不可满足问题）

**预注册"参照法"等价准则 + 保持 bf16 伴随（与冻结管线同一算子）**：

1. 准则（先于任何伴随比较写入协议）：在同一 runner 内额外计算一次 **fp32-HVP 前向参照**
   r（HVP 不用 bf16 autocast；其余与冻结管线一致；M 个方向的前向切线 pass）。
   等价判定改为：对每个候选 j，
   `|ĥ_adj − r_j| ≤ |ĥ_fwd_frozen − r_j| + (atol + rtol·|r_j|)`（容差数值沿用冻结值）。
   含义：伴随必须至少像冻结前向逼近同一数学量一样准——这正是"保持完整切线决策质量"
   的语义化。可事先满足性：本轮 fp32 探针已实测伴随 max 偏差 4.1e-10 ≪ 前向自身地板。
2. 计时门：上述准则 PASS 后按 §4 冻结矩阵计时（util<10% 独占 gate 照旧）。
3. 成本注记：参照 pass 本身与 C 前向同价，只在 dev 等价检查用；运行期伴随路径成本与 M 无关。
4. 明确不做：不改冻结前向/CSV/λ/拆分；不把容差倒推放宽；不比较 M>4。

若 A0H-2 计时显示伴随更快 ⇒ 按冻结的新种子/partition 验证计划推进（formal 4 种子、
8 流、Gain/Regret confirmatory，计划另行冻结，不自动开始）。若更慢 ⇒ 回到本文件给出
下一个单一候选设计。
