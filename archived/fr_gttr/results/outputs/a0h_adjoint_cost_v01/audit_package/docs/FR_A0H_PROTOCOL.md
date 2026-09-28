# FR_A0H 协议 —— 伴随等价与响应计算成本（A0H-1）

> 本文件在写任何伴随 GPU 比较/计时**之前**冻结数学合同、数值容差与评测清单。
> 状态机：合同冻结 → CPU 合成验证 → GPU dev 等价检查 →（通过才）计时。
> 输入只读：A0G 冻结制品与 v07 bank dev 状态；绝不写回冻结 CSV。

## 0. 背景与目标

A0G 已冻结：完整切线（C）的决策效用成立，但 shared-tangent 传播相对逐候选
真实回放**无墙钟优势**（H=4 M=4：6.77s vs 2.65s）。A0H-1 检验一条保持
**完整切线决策质量**（与 C 完全相同的候选排序/选择）的降成本路径：
对冻结前向切线的实际线性算子取**反向伴随**，把"每候选 × 每步"的 HVP/切线
步进替换为"单次反向行走"，候选数 M 只进入最后的一次点积。

## 1. 数学合同（冻结）

### 1.1 前向（与冻结 runner 完全一致的算子）

联合切线状态 `z = (δθ, δm, δv)`，逐参数张量。动作步计入 H：
readout 在 `θ_{t+H}`（H 个 native optimizer step 之后），与冻结
`_run_tangent_pass`（`research/scripts/a0g_decision_utility_v01.py`，
`hvp_direction_mode="pre"`）对齐。

- 动作：`a_j = g_j − g`（候选组梯度减基线梯度，post-unscale/pre-clip 语义）。
- 初始步（k=1）：`δc_1 = J_1 a_j`（`J_1 = J_clip(g_1)`，冻结 `clip_jvp`），
  初始联合状态不扰动（δθ=δm=δv=0）：
  `δz_1 = G_0 a_j`，`G_0 = A_1 ∘ J_1`。
- 未来步（k=2..H）：模型 B `δg_k = 0`；模型 C `δg_k = H_k δθ_{k−1}`
  （`H_k` 为训练 loss 在微步起点 `θ_{t+k−1}` 处的 Hessian 作用，
  **math-SDPA 代理**，与冻结 A0C/A0E2/A0G 合同一致，不换 unroll）；
  `δc_k = J_k δg_k`；`δz_k = A_k δz_{k−1}`。
- `A_k`（冻结 `adamw_tangent_step` 的线性部分，native 量：m,v 步前状态、
  c=clip 后梯度、lr、wd、β1、β2、eps、step_after=t+k）：

  `dm1 = β1 δm + (1−β1) δc`
  `dv1 = β2 δv + 2(1−β2) c⊙δc`
  `δθ+ = d δθ + D_m⊙dm1 + D_v⊙dv1`，`d = 1−lr·wd`
  `D_m = −lr / (bc1·(√v̂+eps))`
  `D_v = +lr·m̂ / (bc2·2√v̂·(√v̂+eps)²)`（**正号**，已由冻结实现数值确认到机器精度：
  前向 `dθ += −lr·adaptive_jvp`，dv 路径 `adaptive_jvp` 自带负号；
  `√v̂=0` 坐标取 0，且冻结前向要求该坐标 `dv̂=0`）
  其中 `m̂=m1/bc1, v̂=v1/bc2`，`m1=β1·m+(1−β1)·c`、`v1=β2·v+(1−β2)·c²`
  （步后矩；快照存步前 optimizer 矩 m,v 与 clip 后梯度 c，伴随侧按冻结前向
  同样的公式重构步后矩），`bc1=1−β1^{t+k}, bc2=1−β2^{t+k}`。

- 读出：`hhat_j = ⟨v_H, δθ_H⟩`，`v_H = ∇θ L_score(θ_{t+H})`（冻结
  `_grad_on_batches` 于 V_score），`ℓ_H = (v_H, 0, 0)`。

### 1.2 反向伴随（冻结公式）

`p_H = ℓ_H`；`p_{k−1} = A_k^T p_k`（k=H..1）：
- `u_m = p_m^+ + D_m⊙p_θ^+`，`u_v = p_v^+ + D_v⊙p_θ^+`
- `b_k = (1−β1) u_m + 2(1−β2) c⊙u_v`
- `p_θ = d·p_θ^+ (+ K_k^T J_k^T b_k)`（C 未来步含反馈项；B 无）
- `p_m = β1 u_m`，`p_v = β2 u_v`
- 反馈 transpose 顺序 `K^T J^T`（HVP 的 VJP 先于 clip-JVP 的 VJP），不得交换。
- 起点读出：`w_0 = J_1^T b_1`，`ĥ_j = ⟨w_0, a_j⟩`；
  **不对未扰动的起点状态做反馈 HVP**（动作是外生梯度扰动）。
- 有限动作一阶损失分数：`λ·ĥ_j`，越小越优先；no-op（a_0=0）分数恒 0。
- H=1 无未来参数—梯度反馈；H=4/H=8 是不同读出（各自独立的 v_H 与行走）。

### 1.3 transpose 实现约定

- `J^T`：独立实现 `clip_jvp_transpose`（active 时
  `J^T w = coef·w − κ·g·⟨g,w⟩`，`κ=max_norm/(‖g‖(‖g‖+1e−6)²)`；inactive 恒等），
  并用内积恒等式数值核验（不凭理想对称性跳过）。
- `K^T`：标量 loss 的 Hessian 对称 ⇒ 代理 HVP 算子自伴；**用同一 math-SDPA
  `_hvp` 例程以 `u` 为方向计算 `K^T u`**，并在每个 dev 状态数值报告
  代理对称性残差 `max |⟨x,Hy⟩−⟨Hx,y⟩|`（诊断，不事后放宽）。
- 伴随在真实 rollout 上按步快照 native 量（m,v,c,g,lr,clip 系数、
  `rng_before_native_forward`、步起点 model state）；反向行走时对步 k≥2
  从快照重算该步 HVP（前向同样的 RNG 恢复纪律），代价 (H−1) 次 HVP，
  与候选数 M 无关。

## 2. 数值容差（冻结，先于任何伴随误差观测）

- dtype：与 A0G 相同（bf16 autocast、fp32 主权重/切线、投影用 float64 累加）。
- CPU 合成（float64 参考）：内积恒等式与端到端 ĥ 等价 ≤ **1e−10**（绝对）。
- GPU dev（bf16 原生 + fp32 切线，操作序不同）：
  每候选 `|ĥ_adj − ĥ_fwd| ≤ atol + rtol·|ĥ_fwd|`，**atol=1e−6，rtol=1e−4**；
  排序必须一致；仅允许近并列差异，近并列阈值 `2·(atol+rtol·max|ĥ|)`；
  no-op 分数差必须恒为 0。选择（argmin，tie 取小 j）必须一致。
- 容差依据 dtype(fp32 机器精度 ~1.2e−7)、冻结 pipeline 的投影归约精度与
  原实现重复性（A0G 计时副本 vs 冻结 CSV max|Δhhat|=0 的位级确定性先例）
  预定；**不参考待验伴随误差或 formal 选择结果倒推**；不借原有限差分
  误差容忍更大转置错误。不满足 ⇒ 记 NUMERICAL_LIMIT/FAIL 并定位原因。

## 3. GPU dev 等价检查清单（预先冻结）

- 状态：dev 三态 **(2031,52) (2034,64) (2036,76)**（v07 bank，与 formal 分离）；
  每状态 **1 条既有未来流（stream 0）**；候选为冻结 4 数据组（既有顺序 j=1..4）。
- 顺序：先在 **1 个 dev 单元**完成 H∈{1,4,8} 的局部内积 + 四候选分数比较
  （含排序、no-op 差、最优/次优间隔）；通过后扩到其余 ≤2 态（每态 stream 0）。
- 方法运行顺序事先固定：forward-C → adjoint-C（每单元同序）；比较在
  同一基准、候选、读出与数据流下进行。同一 runner 内做快照重算自校验
  （前向已算的 HVP 方向 vs 快照重算）定位为后端/快照问题。
- B 模型伴随（无反馈）作为免费副检查（同容差）。
- 若选择完全一致且有现成 dev 回放收益列，可验证 Gain/Regret 一致；
  dev V_score 回放收益只在 λ 扫描 CSV（dev-only）存在 —— 只作方向性核对并注明。

## 4. 有界成本诊断（等价通过后才开始）

- 只测四种实际可运行方法：真实正向有限动作回放（V_score 评分）、
  仅优化器前向切线（B）、完整前向切线（C）、完整伴随（adjoint-C）。
- 计时矩阵（预先冻结）：dev 单元 1 测 H∈{1,4,8} × M∈{1,2,4}；
  其余 ≤2 个 dev 单元仅 H=4、M=4。M=非零候选数，no-op 恒在；
  候选子集按既有顺序取前 M。每组合 ≥1 次预热 + 3 次完整重复，
  方法顺序事先固定交错；报中位数与离散度。
- 端到端口径：同步墙钟（`cuda.synchronize` 夹 `perf_counter`）+ CUDA 时间 +
  峰值显存；计入基准未来轨迹、候选梯度、读出梯度、状态保存/重算、打分排序。
- 规模 profiler 单独小跑（正式计时关 profiler）；先收集 trace 再解释瓶颈。
- GPU 纪律：只读查看；不杀外部任务；忙则先做 CPU/离线；无独占资源记
  WAITING_FOR_GPU；本轮 GPU 累计墙钟上限 **30 分钟**；缺项记 PARTIAL。
- 计时 gate（沿用 A0G）：util 中位 <10% 且无其他计算进程才正式计时；
  否则计时部分标 WAITING_FOR_GPU 交付，不报告未验证加速。

## 5. 交付判定

- 伴随等价：PASS/FAIL（容差见 §2）。
- 成本：在已测条件下是否更快（中位数 speedup + 离散度；不外包、不外推 M>4）。
- 负结果如实交付；不修改规则把失败变成功；不写"反向伴随是新链式法则理论"。

## 6. 执行记录（2026-09-14 后补；§1–§5 为冻结前合同，未改动）

- CPU 合成验证（先于 GPU）：`research/tests/test_a0h_adjoint_v01.py` 4/4 PASS
  （逐通道 FD 1e-6；clip 转置内积恒等式 1e-10；两步复合恒等式 1e-10）。
- GPU dev 等价：3 状态 × stream 0，H∈{1,4,8}，报告
  `outputs/a0h_adjoint_cost_v01/dev_check/unit_*.json` 与 `dev_check_summary.json`。
  - 前向复演位级一致（max|Δĥ|=0，final_model_hash 一致）。
  - B（无 HVP）全部精确（≤1.4e-10）；C|1 精确（≤5e-11）。
  - C|4：2/3 状态逐元通过；C|8：1/3 通过 ⇒ 逐元 gate FAIL（NUMERICAL_LIMIT）。
  - 排序/选择/no-op：3 状态 × {B,C} × H 全部一致。
- 归因（诊断探针，不改变 gate/容差）：`fp32_hvp_probe_*.json`、
  `fp32_hvp_probe_analysis.json`。fp32-HVP 伴随对 fp32-HVP 前向参照端到端
  max|Δĥ|=4.1e-10（H=8）⇒ 转置实现精确；冻结 bf16 前向对自身 fp32-HVP 参照
  max|Δĥ|：H=4 达 1.43e-5、H=8 达 5.78e-5 > 冻结容差 ⇒ 逐元 gate 在 H≥4 对
  任何伴随不可满足（噪声地板来自冻结 bf16 math-SDPA HVP 代理的转置不对称性，
  探针：bf16 2.4e-4..7.3e-2，fp32 3.2e-8..6.0e-7）。
- 计时：按 §4 gate（等价通过前不开始），本轮未执行。
