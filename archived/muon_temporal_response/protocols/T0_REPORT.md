# T0_REPORT — 固定多重集方向—时间受控实验（第一轮）

日期：2026-09-28。配置：`configs/t0_round1.json`（运行前冻结，SHA 见内）。
构造、口径、预算全部按 `references/Muon_Temporal_Round1_T1_T0_Execution_Plan_v1.md`。

**本报告不回答**：是否降低 Transformer NLL、是否省 GPU 能量、是否形成新优化器、是否可发表。
64 个循环相位是有限枚举总体，**不是 64 个独立训练种子**，不做相应显著性检验。

---

## 0. 一句话结论

固定梯度多重集、单时刻分布与标量 lag 相关完全相同的两种方向—时间安排，在理想 polar、
FP64 有限 NS、原生 Muon（BF16 NS）三条路径上均产生可复算的输出方差差（A2≠A1）；
FP64 下方向模型预测近乎精确、标量模型原理性失效；原生路径在三个扰动尺度上都与实数
代理不同——小尺度因 BF16 量化抹掉对称通道而**更接近 polar**，中尺度出现**比值反转**，
大尺度回到同号但幅度偏离。方向模型在原生上仅旋转通道可靠。

---

## 1. 技术正确性栏

| # | 检查 | 结果 | 证据 |
|---|---|---|---|
| T1 | 每臂每相位四种符号组合各 16 次；跨臂跨相位输入 multiset SHA 唯一 | VALID（20/20 测试通过） | `tests/test_t0_round1.py`；`python tests/test_t0_round1.py` |
| T2 | 64 相位位置边际均值=S、协方差两臂一致；标量自相关两臂逐 lag 相同（C(0)=2.0） | VALID | 同上 |
| T3 | 方向 lag 协方差两臂不同（lag-1 差 > 1e-3）；单步谱多重集一致 | VALID | 同上 |
| T4 | 相位枚举顺序不改变统计；delta=0 → 常量输入、A=0（FP64 精确 0；FP64 finite_ns 浮点残差 6.4e-31 记为近零） | VALID | 同上 + `t0_summary.json` |
| T5 | Nesterov 递推 = 有限窗展开（<1e-14）；常量输入 H≡S；权重和=1−β^(t+1) | VALID | 同上 |
| T6 | B2 公式 vs 5 向量 120 排列穷举（<1e-13） | VALID | 同上 |
| T7 | polar 切线：DP_S[Bs]=0（<1e-14）、DP_S[Ba]=(2/3)Ba（<1e-12）；NS 切线 vs 中心差分阶梯 {1e-3..1e-6}，非零方向含截断/舍入下限内通过，Bs 近零方向按绝对误差二阶收敛 | VALID | 同上 |
| T8 | NS 方向增益 ‖J Bs‖²=0.28145、‖J Ba‖²=0.36117（对上原大纲 §9 审计值） | VALID | 同上 |
| T9 | 原生完整 step 与捕获 D 对齐：max\|Δparam+D\|=5.96e-8（lr=1, wd=0，8/8） | VALID | `native_gpu_meta.json` |
| T10 | 原生两遍完整重复 **bit 级一致**（主矩阵+脉冲全部条件） | VALID | `native_gpu_meta.json` repetition_check |
| T11 | 每相位重置 M0、两臂独立构造互不污染 | VALID | 测试 `test_inputs_not_mutated` + runner 结构 |

## 2. 现象栏（逐映射、逐尺度；A_j 定义见 config，先跨相位中心化再时间平均）

### 2.1 主矩阵：A1 / A2 / 比值（实测）

| 映射 | delta=1e-3 | delta=1e-2 | delta=5e-2 |
|---|---|---|---|
| polar_fp64 | 6.27e-9 / 2.11e-7 / **33.650** | 6.27e-7 / 2.11e-5 / **33.650** | 1.57e-5 / 5.27e-4 / **33.625** |
| finite_ns_fp64 | 1.387e-7 / 1.754e-7 / **1.2647** | 1.392e-5 / 1.754e-5 / **1.2597** | 3.81e-4 / 4.38e-4 / **1.1511** |
| native_muon | 5.10e-9 / 1.72e-7 / **33.678** | 4.28e-5 / 1.71e-5 / **0.4011** | 2.78e-4 / 4.78e-4 / **1.7182** |

- `polar_fp64`：**PRESENT**（方向模型预测比值 33.650，实测 33.6502/33.6495/33.6253，
  非线性随 delta 缓慢显现）。对称通道（Bs 时间方向）在理想 polar 下如理论预期近乎消失。
- `finite_ns_fp64`：**PRESENT**。比值 1.2647→1.1511 随幅度下降（非线性，H1 同源）。
  与 v2 §3 报告的有限幅度回放比值 1.26472（其 epsilon_perturb=0.001 绝对值）吻合；
  v2 的线性化值 1.2566 来自 Markov(α=0.8) 构造口径，与本周期构造的线性化 1.2647
  不是同一数——**不作为通关答案，仅口径登记**。
- `native_muon`：**PRESENT（三个尺度均非零可复算）**，但与 FP64 代理定性不同：
  - delta=1e-3：比值 33.68 ≈ polar。机制（实测）：BF16 NS 输入将对角元（2.0/1.0，
    量化间距 7.8e-3/3.9e-3）上的对称扰动整个抹掉（bf16_distinct_fraction=0.5：
    只有非对角元保留），原生只剩旋转通道——行为退化为 polar 型。
  - delta=1e-2：比值 **0.4011，反转**（A1>A2）。对称通道以量化跳变形式回归
    （native A1=4.28e-5 vs FP64 代理 1.39e-5，×3.1），非对称通道与代理吻合
    （1.71e-5 vs 1.75e-5）。
  - delta=5e-2：比值 1.718（代理 1.151）。
  - delta=0：A=0（零响应，bit 级）。
- 输出均值偏移 B_j(t)（H1 类，诊断）：finite_ns delta=5e-2 时 arm1 B_mean=4.65e-3、
  arm2=3.46e-4——**均值偏移两臂差 ~13 倍且量级超过 A 本身**；原生同向
  （见 `t0_metrics.csv` mean_shift 列）。这直接支持"更新方差减小 ≠ 更新期望不变，
  更不等于终点改善"。
- 完整回放（full replay）即本表实测本身：作为诊断标签保留，不称为廉价在线方法。

### 2.2 独立脉冲响应（基线 G=S、M0=S、T=64、第 16 步脉冲）

| 映射 | 通道 | dp=1e-3 | dp=1e-2 | dp=5e-2 |
|---|---|---|---|---|
| polar | Ba→Ba | 峰 0.06500，切线误差 3.4e-10 | 同，3.4e-8 | 同，8.6e-7 |
| polar | Bs→* | 全零（理论精确零） | 全零 | 全零 |
| finite_ns | Bs→Bs | 峰 0.018908，误差 1.4e-8 | 峰 0.018909，1.4e-6 | 峰 0.018942，3.4e-5 |
| finite_ns | Ba→Ba | 峰 0.058595，误差 3.9e-10 | 同，3.9e-8 | 峰 0.058594，9.7e-7 |
| 交叉（Bs↔Ba） | 两条 FP64 | 实测 0 = 切线预测 0（块对角精确成立） | | |
| native | Bs→Bs | **完全零响应（bitwise）** | **完全零响应（bitwise）** | 峰 0.05241，与切线差 4.5e-2（×2.8） |
| native | Ba→Ba | 峰 0.058657，误差 1.3e-4 | 峰 0.058747，1.5e-4 | 峰 0.058868，2.7e-4 |
| native | 交叉 | 全零 | 全零 | 全零 |

判定：
- FP64 脉冲：**PRESENT**，一阶核预测在两档幅度内精确到 1e-8—1e-10（响应形状
  = w_{t−16}·切线，非单调、随动量核衰减并变号，无"单调衰减"表述）。
- native **Ba 通道**：PRESENT，割线响应与实数切线差 ~0.2%（量化保留小数值档位）。
- native **Bs 通道**：dp≤1e-2 **NOT_RESOLVED（量化死区：±脉冲的 D 序列 bit 级相同）**；
  dp=5e-2 PRESENT 但割线 0.0524 vs 切线 0.0189（有限幅度+离散跳变叠加，不解释为导数）。
- 不要求微扰越小越接近导数：实测即反例（Bs 通道越小越死；Ba 通道 0.2% 平台）。

### 2.3 原生与 FP64 代理差异的归因（map-only 对照，复用既有数组）

| delta | H_native vs H_fp64 max | D_native vs D_maponly max | A_maponly vs 纯FP64 A |
|---|---|---|---|
| 1e-3 | 7.8e-7 | 3.7e-3 | 一致（1.387e-7/1.754e-7） |
| 1e-2 | 6.2e-7 | 2.2e-2 | 一致 |
| 5e-2 | 7.4e-7 | 2.2e-2 | 一致 |

**结论（实测）**：动量/输入路径的 fp32 舍入可忽略（~1e-6）；原生偏差几乎全部来自
NS 算子的 BF16 执行（对相同捕获 H 做 FP64 映射即恢复代理值）。

## 3. 预测栏（模型对 A1/A2/ΔA 的误差；J 只取自解析切线，从未拟合输出）

两模型拥有同一映射信息（同一 J、同一滤波），唯一差别是输入相关结构是否方向分辨。
标量模型在构造上必然给 A1=A2（标量 lag 相关两臂相同）。

| 映射 | delta | 方向模型 rel err A1 / A2 / ΔA | 标量模型 rel err A1 / A2 |
|---|---|---|---|
| polar_fp64 | 1e-3 | 0.000 / 0.000 / ~0 | 16.33 / 16.33 |
| polar_fp64 | 5e-2 | 0.000 / 0.000 / ~0 | 16.32 / 16.32 |
| finite_ns_fp64 | 1e-3 | 0.000 / 0.000 / ~0 | 0.132 / 0.128 |
| finite_ns_fp64 | 1e-2 | 0.004 / 0.000 | 0.128 / 0.128 |
| finite_ns_fp64 | 5e-2 | 0.090 / 0.000 | 0.031 / 0.031 |
| native（J=finite_ns 切线代理） | 1e-3 | **26.21** / 0.022 / 反号 | 0.132（同左列结构） |
| native | 1e-2 | **0.676** / 0.023 / 反号 | — |
| native | 5e-2 | **0.246** / 0.083 / 符号同、幅度差 | — |

- FP64 两条映射：**方向模型有效**（相对误差 0—9%，随幅度如非线性预期增长；
  每时刻协方差矩阵 Frobenius 误差见 summary），**标量模型原理性失效**
  （对 polar A1 误差 16 倍 = 恰好漏掉整个通道；对 finite_ns 也系统性 3—13%）。
- native：方向模型的**旋转（Ba）通道可靠**（2—8%），**对称（Bs）通道在全部三个尺度
  失效**（26×/0.68/0.25，ΔA 两个小尺度反号）。失效原因可归因（§2.3）：实数切线
  不含 BF16 量化死区/跳变。用 polar 切线做 J 的对照更差（summary `direction_polarJ_*`）。
- 每时刻协方差误差与逐条件全部数字：`t0_summary.json`、`t0_metrics.csv`。

## 4. 分栏判定汇总

| 维度 | 判定 |
|---|---|
| 技术 | **VALID**（11/11 组检查通过；原生路径可调用并绑定 SHA；无 BLOCKED） |
| 现象 | polar：PRESENT；finite_ns_fp64：PRESENT（幅度依赖 1.26→1.15）；native：PRESENT（33.7 / 0.40 / 1.72，机制与代理不同）；native Bs 脉冲 dp≤1e-2：NOT_RESOLVED（量化死区）；其余 NOT_RESOLVED 无 |
| 预测 | FP64：方向模型准、标量模型失效（构造使然）。native：方向模型仅 Ba 通道可用；ΔA 排序在 delta=1e-2 反转，**不能**外推为训练收益 |

## 5. 对 T2 准备的含义（只列证据，不做启动决定）

1. **支持准备**：原生路径差异真实、可复算、bit 级可重复；非对角（旋转）通道可被
   实数代理预测到 ~1e-4 相对精度；输入/动量 fp32 路径与 FP64 演化一致（~1e-6）。
2. **必须先解决**：任何 T2 误差门限必须计入 BF16 量化死区——相对扰动 <~4e-3 的
   对角（大元素）方向扰动可被整个抹掉。真实梯度扰动幅度 vs 该死区的比例未知，
   属 T2 首个吞吐测试内容；门限须在揭盲前冻结。
3. 方向模型不能裸用到原生全通道；T2 对照组须含"实数代理 + 量化死区掩码"的失效边界。
4. 均值偏移量级可与输出方差相当——T2 的机制模型必须同时记录均值偏移，不能只报方差。

## 6. 复跑命令与产物

```bash
cd /home/b205/projects/training_dynamics_lab/muon_temporal_response
python tests/test_t0_round1.py                 # 20 项预运行测试
python scripts/run_t0_cpu.py                   # FP64 主矩阵+预测+脉冲 (~5s CPU)
CUDA_VISIBLE_DEVICES=0 python scripts/run_t0_native.py   # 原生路径 (~111s GPU, 单卡)
python scripts/collect_results.py              # 汇总+图 (~6s CPU)
```

产物（`outputs/t0_round1/`）：`t0_summary.json`、`t0_metrics.csv`、`pulse_response.csv`、
`cpu_fp64.npz`、`native_gpu.npz`、`fig1_A_vs_delta.png`、`fig2_model_errors.png`、
`fig3_pulse_response.png`、`cost_ledger.json`、两份 meta json。
每行 CSV 带完整 condition id（map/delta/arm/rep、pulse_dir/proj_dir/delta_pulse/t）；
配置与源 SHA 见 `configs/t0_round1.json`。

## 7. 本轮明示未做

Transformer 训练（0 updates）、真实 batch 梯度采集（0）、父轨迹补建（0）、确认集访问（0）、
T2/T3/T4 任何实际运行、新优化器/controller/超梯度、物理温度与熵产生估计、
update_work_proxy 的焦耳解释、16×16/16×32/32×16 与 beta/奇异值扫描（留后续）。
v2 的 legacy_supplement（epsilon_perturb=0.001 绝对值复算）因原配置 eps_norm 等字段
不可完整复原，本轮未重跑，仅在新口径下吻合其报告比值。
