# T2A_REPORT — 单状态真实尺度校准结果（第二轮阶段 B）

日期：2026-09-29。协议：`docs/T2A_PROTOCOL_v1.md`；配置：`configs/t2a_scale_v1.json`。
锚点：`m1_muon_lr2.0_wd0.02_p0.5_s2030`（SHA `4f68ce24…`，核对通过：cursor step2048、
56 个 fp32 EMA momentum、四路 RNG、末步 lr=0.001 与 cosine 一致）。
状态：**COMPLETED_DIAGNOSTIC**（预算内完成全部 10 batch、660 个诊断点）。
**本报告不含任何训练收益结论。**

---

## 第一栏：技术正确性

| 检查 | 结果 |
|---|---|
| 锚点 SHA / 结构 / momentum / cursor / RNG | 通过（运行前核对，见协议） |
| 10 batch × 4 micro backward，生产 bf16 autocast、生产 loss/累积语义 | 通过；有效 targets 32,640/batch（**= 预期**），10 batch 共 326,400 |
| 全参数（Muon 56 + aux 34）global clip 1.0 | 通过；**preclip 全局范数 0.337–0.364，10/10 未触发 clip**（clip_coef=1.0） |
| 六矩阵剪裁后 G 保存；无局部 clip | 通过 |
| 模型参数 hash + torch RNG 采集前后不变 | 通过（bit 级相同） |
| optimizer.step 调用 / scheduler 推进 / dev·eval·confirm 访问 | **0 / 0 / 0** |
| H 构造数学校验 H_i=β²M_*+(1−β²)G_i（fp64） | 全 60 点 < 1e-6 |
| H_ref = 前5 fp64 均值→单次 fp32 转换；`calibration_lock.json` 先于后5 标签写入 | 通过；cal 半区输入均值漂移 5.3e-8（恰为单次转换舍入量级，已入测试） |
| manifest 前5/后5 window 集合互斥、SHA `1e8b29c7…` | 通过 |
| 向量恒等式 `delta_native−L = r_real+r_numeric`（660 点） | 最大绝对误差 3.7e-9（浮点一致） |
| 三路径重复映射（H_ref/batch1/batch6 × fp64/fp32/native） | 全部 bit 级一致 |
| 回归测试 | 48 项全绿（16 T2a + 12 勘误 + 20 原轮） |
| 成本 | GPU 26.5 s + 一次统计代码失败的重复运行 ≈30 s（同 manifest，无新增唯一 batch）＝ 56.5 s / 1800 s 预算；输出 842.4 MB / 1 GB；CPU 后处理 < 1 s |

## 第二栏：数学/数值适用范围（核心实测）

### 1. 自然尺度：真实 batch 间 H 波动是 O(1) 相对量

六矩阵的 `||H_i−H_ref||/||H_ref||`：
- 前 5（校准）：0.50–0.57（均值 0.542）
- 后 5（检查）：0.63–0.73（均值 0.681）；**输入均值漂移 0.37–0.40**（数据流前进使后 5 的均值不同于前 5，非平稳）

真实训练中的 batch-to-batch 谱映射输入变化**远大于** T0 toy 的 δ≤5e-2；任何基于
"小扰动线性化"的模型在自然尺度上需要重新审视（见 §3）。

### 2. 同一真实 H 上的三路径数值误差（自然点，总体分解 A_native=A_ref+A_q+2·cross）

| 比较 | A_q/A_ref（cal） | A_q/A_ref（check） | cross2/A_ref | mean_q |
|---|---|---|---|---|
| FP32 vs FP64 | ~1.3e-12 | ~1.3e-12 | ~-1e-12 | ~1e-11 |
| **原生 BF16 vs FP64** | **3.6e-4 – 8.0e-4** | **3.6e-4 – 7.3e-4** | ~-3e-4 | ~1e-7 |

（六矩阵逐值见 `map_errors.csv`；分解恒等式全部通过。）
**真实满谱矩阵上，BF16 的同 H 相对数值方差 ≈ 0.04–0.08%**——比 T0 的 2×2 对角 toy
（A_q/A_r ≈ 1.0）低约三个数量级。toy 中观察到的对角量化格点死区/强抵消在真实矩阵上
不存在：**600 个非零 alpha 诊断点中零响应 0 个**。T0 的 4e-3 死区与 0.2% 误差
**不是**真实矩阵的通用阈值（本轮实测否定其迁移性）。

### 3. 有限幅度响应：量化噪声地板 vs 非线性余项（中位数，六矩阵×10 batch）

| 实际相对扰动 | \|δ_native\|/\|L\| | cos(δ_native, L) | rel_resid | \|r_numeric\|（地板） | \|r_real\|（非线性） |
|---|---|---|---|---|---|
| ~7e-4 (α=1e-3) | 11.5 | 0.086 | 1.00 | 0.19 | 1.2e-5 |
| ~7e-3 (α=1e-2) | 1.95 | 0.51 | 0.86 | 0.26 | 1.2e-3 |
| ~3.4e-2 (α=5e-2) | 1.07 | 0.935 | 0.35 | 0.29 | 3.0e-2 |
| ~0.17 (α=0.25) | 0.994 | 0.947 | 0.32 | 0.33 | 0.71 |
| ~0.68 (α=1, 自然尺度) | 0.78 | 0.66 | 0.76 | 0.33 | 7.9 |

三条边界（实测，非先验阈值）：
1. **rel ≲ 7e-3：响应不可用**——BF16 输出量化噪声地板（|r_numeric|≈0.2–0.3，与 α 无关）
   淹没线性信号；但响应**非零**（与 toy 死区定性不同）。
2. **rel ≈ 3e-2–2e-1：线性预测部分可用**——方向 cos 0.94–0.95，幅度误差 ~30% 中位；
   此时 r_real 与 r_numeric 同阶（比值 0.3–10），量化与非线性共同贡献。
3. **rel ≈ 0.7（自然尺度）：线性化失效**——非线性余项（|r_real|≈8–11）比噪声地板大
   ~30 倍；|δ_native|/\|L\|≈0.78、cos≈0.66。FP64 线性 JVP 不能预测自然 batch 间的
   完整输出差分。

### 4. 非线性均值偏移（输出，相对 H_ref 的输出）

- cal 半区（输入均值零漂移）：native 输出 mean_shift_rel **0.20–0.38**——纯非线性整流，
  真实尺度下巨大。
- check 半区（输入漂移 0.37–0.40）：输出偏移 0.44–0.56。
- 后 5 的输入均值漂移 ≠ 0 表明"沿训练时间"的 H 分布在 10 batch 内已可见非平稳；
  本轮共同 M_* 设计的 10 个 H_i 离散性**不能**当作真实训练时序协方差（协议已声明）。

## 第三栏：研究判断范围

**支持准备完整 T2 的证据**（技术层面）：
1. 原生路径在真实矩阵上数值误差小（同 H 方差占比 ≤8e-4）、bit 级可重复；
2. 全链路（生产语义梯度→全局 clip→共同 M_*→H→三路径）零训练成本可采集，
   实测吞吐 ~2.6 s/batch（含前向反向+clip），1024 batch 外推 ≈ 45 分钟单卡——
   完整 T2 的 4 态×2bank×128 batch 梯度采集在单卡 ~3 小时量级（外推，非承诺）；
3. 分解/恒等式框架（阶段 A 入口）在真实数据上全部成立。

**不支持或需重新设计的**（模型层面，SUPPORTED/NOT_SUPPORTED/UNRESOLVED）：
- NOT_SUPPORTED：把 T0 的 BF16 死区掩码或 0.2% 误差作为真实矩阵门限（实测否定）。
- NOT_SUPPORTED：以单一固定 H_ref 的线性 JVP 预测自然 batch 间完整输出差分
  （自然尺度 rel~0.7 时 cos 0.66——线性代理不覆盖自然波动）。
- UNRESOLVED：方向—时间统计在真实训练中是否提供超出标量统计的决策信息——
  本轮只校准了数值边界，未做任何时序模型比较。
- UNRESOLVED：小扰动窗口（rel 3e-2–2e-1，线性 cos≥0.94）在真实 T2 中是否可用
  ——取决于完整 T2 的排列干预实际产生的 H 偏移幅度（纯重排在多大程度上改变 H？），
  这正是完整 T2 要测的第一件事。
- 本轮**不回答**训练收益、终点 NLL、GPU 能量。

**剩余阻塞（进入完整 T2 前）**：
1. 中间训练态（step512/1536）与第二开发种子仍缺失（需父轨迹授权；T2 草案 v2 已更新成本）；
2. 真实梯度扰动 vs clip：本锚点 10/10 未触发 clip；其他状态/阶段是否触发需在 T2 采集时逐 batch 记录；
3. T2 误差门限须在揭盲前冻结，且必须按本轮实测边界（噪声地板 ~0.3 绝对输出范数、
   线性可用窗口 rel≈3e-2–2e-1）制定，不得沿用 T0 数字。

## 附：产物与复跑

`outputs/t2a_scale_v1/`：`batch_manifest.json`、`calibration_lock.json`、`raw_natural.pt`
（G_post_clip/M_star/H_ref/H_i/D_ref/D_i 全量）、`raw_sample_l1q.pt`（第一层 q_proj
batch1/batch6 全部 alpha 点完整数组）、`map_errors.csv`、`response_errors.csv`、
`summary.json`、`cost_ledger.json`、`run_log.txt`。

```bash
cd /home/b205/projects/training_dynamics_lab/muon_temporal_response
python scripts/run_t0_corrected.py        # 阶段A（CPU）
CUDA_VISIBLE_DEVICES=1 python scripts/run_t2a_scale.py   # 阶段B（~27s GPU）
python tests/test_t2a_scale.py && python tests/test_t0_corrected.py
```
