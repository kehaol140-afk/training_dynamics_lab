# F2 — Hypergradient Pilot：ill-conditioning 的真实下游后果（100M，LR channel）

- 日期：2026-09-26。预注册：`docs/F2_HYPERGRADIENT_PILOT_PREREG.md`（先于任何 F2 结果冻结）。上游 gate：F1 `V_FLOOR_CAUSAL`。
- 对象：`h_eta = d L_val(theta_{t+H}) / d log(eta)`，MiniMind 100.78M，G1b bank 3 cells（EARLY×2 + MIDDLE×1），冻结 future stream（manifest SHA256 见 summary.json），CRN central FD ground truth（δ∈{0.02,0.05,0.1,0.25,0.5}），H 读出 {1,2,4,8,16,32}。
- 不开发任何方法；forward AdamW 未动；bank/legacy 只读。

## 0. 总判定（F2_VERDICT）

**`PRACTICAL_HYPERGRADIENT_FAILURE`**（相位限定：EARLY cells 失败、MIDDLE cell 可用）。

FD 有效性 gate PASS（3/3 cells：H=1 处 h_FD 跨 δ 收敛 <0.1%、信号超离散度 2×）。判定依据（prereg §4 规则 1）：fp64/C 切线 hypergradient 在 2/3 cells（两个 EARLY）于 H≤16 出现 rel_err > 1：

| cell（相位） | H | h_FD(0.05) | h_tangent(fp64/C) | rel_err | ‖dθ/du‖ |
|---|---:|---:|---:|---:|---:|
| 2042:26（EARLY） | 16 | −1.05（δ 散布 −0.09..0.59） | **−5.91** | **4.63** | 291 |
| 2042:26 | 32 | 1.57（散布 −0.22..1.57） | **−8.61** | — | 938 |
| 2043:26（EARLY） | 16 | 0.18（散布 −0.11..0.67） | **5.33** | **29.6** | 129 |
| 2043:26 | 32 | −0.52（散布 −0.13..0.51） | **3.21e3** | — | **1.47e5** |
| 2042:64（MIDDLE） | 16 | −0.416（δ 全程稳定） | −0.452 | 0.087 | 7.1 |
| 2042:64 | 32 | −0.732（δ 全程稳定） | −0.833 | 0.139 | 14.8 |

短 horizon 一致良好：H=1/2/4 时三 cells 的 fp64/C rel_err 全部 ≈ 6–9%，符号全部一致（H≤8 符号命中 16/18 读数点，Pearson 0.82 / Spearman 同量级）。

## 1. 失败的解剖（机制关联）

1. **爆炸只发生在 C 分支**（含 HVP 曲率反馈）：‖dθ/du‖ 在 EARLY cells 按每步 ~×2 复合（0.8→291→938；另一 seed 到 1.47e5），B 分支（optimizer-only）全程有界（H=32 仅 18–21）且与 FD 在 H≥4 后符号失配——曲率通道既是正确性所必需（MIDDLE cell 中 C 把 rel_err 从 B 的 >100% 修到 8–14%），又是 EARLY 爆炸的载体。
2. **u-切线的主导模块 = attn_q（3/3 cells、全部 H 读数）**——与 F1 的 v-floor bottleneck 完全同位；B 分支主导模块是 mlp_gate/down（良性质量）。
3. **相位对齐**：失败的两个 cell 正是 F1 中 v-floor 病态最强的 EARLY 相位（100M step26：σ_max=2258、A_v=2.4e9）；MIDDLE（step64，自愈后）完全可用。F1 的机制预报了 F2 的失败位置。
4. **不是 precision 造成**：fp32 与 fp64 切线在 cells 1–2 位级一致；frozen（bf16-HVP 注册路径）与 fp64 在爆炸前一致、爆炸后也只差 ~12%（3.65e3 vs 3.21e3）。失败随精度不变，随相位与机制坐标变化。
5. **FD ground truth 本身也在退化**：EARLY cells 的 h_FD 在 H≥8/16 跨 δ 符号翻转——不是评测噪声（fp32 前向读数），而是 L_val(η) 在长 horizon 的曲率/混沌；即被测对象本身在病态相位不可有限差分。两部分合成"practical failure"：切线在读数上爆炸，FD 在验证上失效。

## 2. 逐项指标（prereg §3）

- rel_err vs FD：见 §0 表；EARLY H≤8 良好（≤0.25），H=16 起 >1。
- 符号准确率：fp64/C 全部 18 个读数点 16/18（两个错点均在 EARLY H=16/32）。
- Pearson 0.823（H≤16 的 15 点，fp64/C vs h_FD(0.05)）。
- candidate-LR ranking（sign(h) 排序 ±δ 候选）：同符号准确率。
- NaN/Inf：0（无 arm 触发停止规则；爆炸为有限值放大，非溢出——100M 尚未到 G1c 1B 的 fp16 溢出域）。
- ‖dθ/du‖ 增长：EARLY 指数（×2/步），MIDDLE 近线性（29×/32 步）。
- 伴随记录：每步 loss/clip 系数在 run.out；u-切线模块份额在 raw_*.json（C 臂 attn_q 主导）。

## 3. 结论与建议（下一步）

1. **conditioning 确实损坏真实 hypergradient**：F0/F1 的 ill-conditioning 不只是诊断量——它使 EARLY 训练相位（≈前 20% 训练）的 LR-hypergradient/bilevel/长程 unrolled differentiation 在 100M 上实际不可用（误差 >4–30×、方向由 attn_q 病态坐标主导），而同一模型同一代码在 MIDDLE 相位完全可用（≤14%）。对实践的粗译：**hypergradient 类方法的适用窗口随训练自愈打开，在早期关闭**；规模越大（F0 阶梯）窗口越晚开。
2. 建议的 F3 候选（本轮未做、未预注册，仅列出）：(a) 把 v-floor/相位指标做成 hypergradient 可用性预报器（σ_max、A_v、attn_q 份额 → 可用/不可用分类）；(b) 在 EARLY 相位做 bounded-H 自适应（用本文的 ‖dθ/du‖ 增长率作为在线 early-stop 信号）；(c) 300M/1B external validation（F1 机制明确后才值得，见 F0 §8 的防护）。
3. 红线遵守：无方法开发、无阈值调整、legacy/bank 只读、负结果（EARLY 失败、FD 长程失效）原样保存。

## 4. 制品

| 文件 | 内容 |
|---|---|
| `outputs/f2_hypergradient/hypergradient_vs_fd.csv` | 3 cells × 5 arms × 6 H：h_tan / h_FD / rel_err / 符号 / δ 散布 |
| `outputs/f2_hypergradient/horizon_growth.csv` | 全部 per-step 切线范数轨迹与停止标志 |
| `outputs/f2_hypergradient/precision_comparison.csv` | fp64/fp32/frozen/B-vs-C 臂间对照 |
| `outputs/f2_hypergradient/summary.json` | gates + verdict + 机制关联 + 相关性 |
| `outputs/f2_hypergradient/raw_*.json` | 每 cell 原始读数（含 manifest SHA、stream seed） |
| `outputs/f2_hypergradient/f2_run.log`, `run*.out` | 运行日志（含 primal loss/clip 轨迹） |
