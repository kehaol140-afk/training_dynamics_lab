# F3 — Curvature Feedback Mechanism 报告（BLOCKED_AT_REPRODUCTION_GATE）

- 状态：**BLOCKED**。主实验（RQ1–RQ5）**未执行**。原因：冻结 reproduction gate G1/G2（相对容差 1e-12，对 archived F2 raw）在当前环境下不可达——**F2 自己的冻结脚本今天重跑也无法逐位复现自己的 archived raw**。按预注册 §7 与用户指令（gate 失败 → 停止主实验、修代码、停止等待人工审核），本轮停止于 gate。
- 日期：2026-09-27。预注册：`docs/F3_CURVATURE_MECHANISM_PREREG.md`（f51ce82 + 658301c）。
- 关键正面结果：**修复一处 F3 实现顺序 bug 后，F3 的 α=0/α=1 切线在"同日环境参考"下对 F2 `fp64/B`、`fp64/C` 的逐步范数复现相对差 = 0.0（逐位一致，32 步 × 2 臂）**。F3 实现的语义正确性已用最大严格度验证。

---

## 1. FACT（直接测量）

### 1.1 冻结 gate 对 archived raw 失败（修复后代码，环境差异所致）

- gate G1/G2 定义：α=0 逐步 `dtheta_u_norm` vs `outputs/f2_hypergradient/raw_{cell}.json` 的 `fp64/B`；α=1（mask 全开）vs `fp64/C`；逐点相对差 ≤ 1e-12。
- 三 cell 冻结 gate（修复后代码，对 archive）**全部失败**，纯环境漂移所致：
  `2042:26` G1 max=3.49e-3 / G2 max=0.888；`2043:26` G1 max=1.28e-3 / G2 max=0.908；`2042:64`（MIDDLE）G1 max=5.93e-5 / G2 max=1.73e-3。G4 max=0.227 / 0.0357 / 5.08e-4。**EARLY 的 G2 巨大失败是环境噪声经 ill-conditioned C 臂的混沌放大；MIDDLE C 臂有界故漂移保持微小**——与 F2 的 EARLY ill-conditioning 现象一致。
- G3（mask 中性）/G5（source-off 线性）/G6（VJP 伴随）：单测 12/12 PASS（含 `torch.equal` 位级断言与 1e-10 伴随容差）。
- G4（mask 不改 primal）：F3 全部臂共享同一条 primal rollout（runner 结构保证）；实现层面无 per-arm 前向。

### 1.2 F2 冻结脚本无法复现自己的 archive（决定性证据）

- 2026-09-26 18:39–19:20（F2 原跑，archive 来源）h=1 loss = **7.43325**（三次 attempt 一致）；FD 相位 `h_FD(16)`、`h_FD(32)` 存于 `raw_2042_26.json`。
- 2026-09-27 02:36 重跑**同一份冻结脚本**（`git status` 干净、同 env/数据/bank）：
  - h=1 loss = **7.43339**（完整值 7.43338680267334，与 F2 原跑差 1.4e-4，相对 ~1.9e-5）。
  - FD 相位表 vs archive：h_FD(1) 相对差 3.3e-5–1.7e-4；h_FD(16)/h_FD(32) 相差 4.5%–108%（多步 rollout 对 ~1e-4 前向扰动的混沌放大）。
  - 重跑完整 raw 存于 `diag_f2rerun_raw_2042_26.json`（archive 未被覆盖）。

### 1.3 当日（2026-09-27）内部可复现性 — 鲁棒

- 两个独立进程 `scripts/f3_determinism_probe.py`：h=1 loss 与梯度校验和**逐位一致**。
- GPU0 与 GPU1 各自运行：逐位一致。
- GPU0 在 28GB 显存压力（并行 hog）下：逐位一致。
- F3 修复后 gate vs **同日** F2 重跑 raw（`--ref-raw diag_f2rerun_raw_2042_26.json`，诊断专用、非 gate 替代）：**G1 max = 0.0，G2 max = 0.0（32 步逐位）**，`raw/gate_diag_sameday_2042_26.json`。

### 1.4 发现并修复的 F3 实现 bug（顺序错误）

- 现象：同日同 primal（loss 逐步一致到打印精度）下，修复前 F3 gate 的 C 臂在 h=2 即偏离 F2 重跑（1.57912 vs 1.582495），混沌放大。
- 根因：`primal_step` 在计算 dc（HVP）与调用 vendored `adamw_tangent_step` 之前就执行了 `opt.step()`，导致 (a) `rig.hvp` 在 POST-step θ 处取 HVP（应为 PRE-step θ_h）；(b) tangent 步读取 POST-step optimizer state（vendored 函数文档要求 "before the native step executes"）。F2 冻结顺序：grads → clip → HVP → clip_jvp → tangent（PRE-step state）→ `opt.step()` → 加 s_h。
- 修复：拆分为 `primal_grads`（快照 θ_prev、grads、clip，不步进）与 `primal_advance`（`opt.step()`）；gate/rq1/rq2/rq3 四个循环统一改为：dc+hom_step（pre-step）→ advance → 加 s_h。rq2 的 q0 臂在 h=1 的传播于 advance 后计算——θ-only probe 的齐次切线为 (1−lr·wd) 标量、不读取 optimizer state，故逐位等价（代码注释已说明）。rq4 原本即正确（算子在固定点构造）。单元测试 12/12 PASS。
- 修复效果：同日参考下 G1/G2 由 ~1e-3–O(1) 变为 **0.0**。

### 1.5 环境漂移排查（输入侧全部排除）

- `scripts/f2_hypergradient.py` 及全部 src：git 干净，自 F2 commit（87d8225）未变。
- bank state `state_step_026.pt`：mtime 2026-09-16，sha256 全程未核对变更（原文件未动）。
- 数据 `e3c_subset.jsonl`、`partition_v07.json`：mtime 2026-09-02。
- `minimind_base` 模型/数据文件：mtime 2026-09-01。
- torch 环境：2026-06-30 安装，未变；驱动 591.44（昨晚版本无法回溯核实）。
- batch manifest：`e40dd1f4709ca8b5`，与 F2 记录一致。

## 2. SUPPORTED_INTERPRETATION（有证据支持）

1. **F3 切线实现（修复后）与 F2 fp64/B、fp64/C 逐步计算序列逐项一致**——在同日环境参考下 32 步逐位复现（相对差恰为 0.0）。这是"α=0↔B、α=1↔C"语义等价的最强验证。
2. **archived F2 raw 是"环境陈旧"的**：其 primal 前向在 ~1e-4 尺度上与今日计算不等价，且 F2 冻结脚本今日自跑亦产生不同的（自洽的）轨迹。因此"对 archive 逐点 1e-12"这一 gate 在当前环境下**不可由任何代码实现满足**。
3. **bf16 primal 多步 rollout 对 ~1e-4 级前向扰动混沌敏感**（FD 表 48–108% 漂移；C 臂 tangent 范数量级差异）。任何跨天/跨环境的 bitwise gate 若建立在 bf16 primal 之上，其 1e-12 前提都不可靠。

## 3. HYPOTHESIS（未确证）

- 隔夜 GPU 数值行为变化的具体机制：**未确定**。显存压力假说已被实验否定（28GB hog 下逐位不变）。候选（均无法回溯核实）：驱动模块状态变化、clock/电源状态、其他瞬态系统层变化。不影响 §2 的操作性结论。

## 4. NOT_ESTIMATED（本轮无法回答）

- RQ1–RQ5 的一切数值（α sweep、impulse、module ablation、spectrum、decomposition）：**未运行**（gate 未过，禁止绕过）。
- 机制分类（§8.1）与总 verdict（§8.2）：**不适用**——研究未执行。按 prereg 语义本轮无 verdict；状态记为 BLOCKED_AT_REPRODUCTION_GATE（非 UNRESOLVED——后者用于已执行但诊断不足的情形）。

## 5. 提交人工决定的选项（不在本轮权限内）

1. **人批 prereg 修订**：在固定环境 pin（同日、同机、无并发）下**重生成**三个 cell 的 F2 reference raw（重跑冻结 F2 脚本），将 G1/G2 参考指向新 raw（或等价地：把 gate 改为"F3 α=0/1 vs **同日重跑** F2 参考，1e-12"），随后按 §10 顺序执行 RQ1→RQ5。风险：环境再漂移会使新 reference 再次失效；可考虑把"同日重跑参考"写入执行协议。
2. **维持冻结 gate**：则 F3 主实验在 environment drift 未消除前无法启动；需先查明并固定隔夜数值漂移机制（当前无结论）。
3. 其他人工指示。

负结果与本报告如实提交；未 push。

## 附录 A：证据文件清单（outputs/f3_curvature_mechanism/）

| 文件 | 内容 |
|---|---|
| `raw/gate_2042_26_preorderfix.json` | 修复前冻结 gate（失败，含实现 bug） |
| `raw/gate_{2042_26,2043_26,2042_64}.json` | 修复后冻结 gate vs archive（失败，环境所致） |
| `raw/gate_diag_sameday_2042_26.json` | 修复后 gate vs 同日 F2 重跑参考（G1=G2=0.0） |
| `diag_f2rerun_raw_2042_26.json` | 2026-09-27 F2 冻结脚本重跑完整 raw（同日参考） |
| `f3_run.log` | F3 运行日志 |
| `../f3_curvature_mechanism_gate_*.out` | 各次运行 stdout |

## 附录 B：验证命令（可复跑）

```bash
# 同日参考诊断（G1=G2=0.0）：
CUDA_VISIBLE_DEVICES=1 python scripts/f3_curvature_mechanism.py --rq gate --cell 2042:26 \
  --ref-raw outputs/f3_curvature_mechanism/diag_f2rerun_raw_2042_26.json \
  --out-name gate_diag_sameday_2042_26.json
# 冻结 gate（对 archive，预期失败）：
CUDA_VISIBLE_DEVICES=1 python scripts/f3_curvature_mechanism.py --rq gate --cell 2042:26
# 单进程前向探针：
CUDA_VISIBLE_DEVICES=1 python scripts/f3_determinism_probe.py
# 单元测试：
python tests/test_f3_curvature_mechanism_v01.py
```
