# NEXT VALIDATION SPEC — 下一阶段验证提案

（基于本审计缺口；每项含 gates 与预算。预算基数 = 18,289，见 ERR-B0-LEDGER。）

---

## N1. 长期收益桥（最高优先级 — 直击 HYP-3/T1/T2）

**缺口**: 全部现有证据 ≤128 步；短期 gain 三处被证伪为最终收益代理。
**假设**: 存在一个可观测标量 c_t（候选：C_t=‖J_u‖、regime 特征、B0.6 的 9 维 ψ_t 投影），使 sign(c_t) 在 T=512/1024 步尺度上仍预测干预方向。
**方法**: 冻结 v07 bank 前段（step≤82 已训部分），从 3 个锚点 fork T=512 步 rollout；臂 = native / persistent-LR±5%（A0T 冻结规则）/ adaptive（chi 在线重算，付 replay 成本）；每 64 步记录 chi 与事后 gain 的滚动相关。
**Gates**:
- G-N1-1: 滚动 Spearman≥0.6 且 direction≥10/12 于 ≥2 个独立 fork（独立于选择流）。
- G-N1-2: adaptive 臂 final ΔL0 优于 fixed 臂（paired, p<0.05, n≥6 fork）。
- G-N1-3: 无反转细胞（H 扫描 64→512 单调或平台，不允许 T1 式符号翻转）。
**预算**: 估算 ~2,200 updates（3 fork × 3 臂 × 512 步 + replay 开销 ×2.0）。
**Fallback**: 若 G-N1-1 FAIL → 接受"响应不可用于长期"，转 N2；若 G-N1-2 FAIL → 冻结 controller 结论为 Case B（固定 +5%），关闭自适应线。
**预期**: PASS（H=4 可控性在 4 倍延长窗内保持）或明确关闭长期外推声明。

## N2. Out-of-sample 控制验证（直击 A0T/A0U in-sample 缺陷）

**缺口**: A0T/A0U 的 chi 与评估同流；无 out-of-sample。
**假设**: 冻结 sign(chi) 规则在**未见流**上保持方向 35/36 级别的 sign accuracy。
**方法**: 从 bank 选 12 个新种子 fork 未见 128-micro 流；控制臂同 A0T（±5% 持久）；chi 从**独立 bank 流**导出（非评估流）。
**Gates**:
- G-N2-1: sign ≥30/36。
- G-N2-2: median gain >0 且 C-B 别名比例 ≤ A0T 的 34/36。
**预算**: ~1,500 updates。
**Fallback**: FAIL → 全部 controller 结论降级为 "in-sample only"。
**预期**: 方向保持（A0S D2 新 rollout 已示 12/12），幅度缩小。

## N3. 下游能力探针（最小可行 — 直击 C23）

**缺口**: 全部读出为 LM NLL；无任何能力评测。
**假设**: 短期干预 gain（ΔLM-NLL）与下游探针（HellaSwag-8/算术-8 精确匹配，MiniMind 探针集）方向一致。
**方法**: 在 A0L 最强的 +7.46e-3 臂与 native 臂的 T=128 终点 checkpoint 上跑冻结探针集（每臂 2 个 fork）；探针集规模固定（≤16 题）。
**Gates**:
- G-N3-1: ΔNLL 方向与 Δprobe 方向一致（≥3/4 臂对）。
**预算**: ~400 updates（探针评测无梯度）。
**Fallback**: 探针无信号（ floor 效应）→ 登记 "C23 NOT_TESTABLE at this scale"，禁止能力声明写入口径。
**预期**: 弱一致或 floor；主要价值是堵住声明漏洞。

## N4. G1c 1B 可行性启动（外部架构 — 直击 C21 边界）

**缺口**: 1B 未测；全部复制在 4.98M/26M/100M。
**假设**: C 模型 rel err ≤0.05 在 1B 且 H≤4。
**方法**: G1c feasibility 已 READY；跑 3 anchors × H{1,4} × 2 streams；沿用冻结协议 + math-SDPA 代理。
**Gates**: G-N4-1: 与 G1a/G1b 同阈（rel err ≤0.05; R²≥0.999）。
**预算**: ~800 updates。
**Fallback**: OOM/代理失效 → 登记架构依赖，不外推。
**预期**: PASS（scale-only 趋势）或暴露 HVP 代理截断新尺度失效。

## N5. B0 台账 Erratum 执行（行政，立即）

重算 outputs/fr_b0_response_mechanisms_v01/b0 summary.json TOTALS：14,208 → 15,097；更新全部后续累计声明；预算基数改 18,289。**0 updates。**

---

**总预算估算**: N1-N4 ≈ 4,900 updates + N5 行政。全部完成后累计 ≈ 23,200。建议 N1 → N5（行政先行）→ N2 → N3 → N4 顺序执行。
