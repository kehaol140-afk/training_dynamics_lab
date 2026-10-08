# IMPLEMENTATION_ALIGNMENT — 安装 Muon 与研究公式的对齐报告（T1）

日期：2026-09-28。性质：本轮实测核对（只读平台审计）。范围：仅与新问题（方向—时间结构对实际
Muon 更新的影响）有关的事实；不重复整仓库审计。旧 M2 保持 `NO_SIGNAL_STOP_LINE`。

---

## 1. 安装 Muon 溯源（实测）

| 字段 | 值 | 证据 |
|---|---|---|
| 源文件 | `/home/b205/miniforge3/envs/fr_minimind/lib/python3.11/site-packages/torch/optim/_muon.py` | 本轮 `sha256sum` 实测 |
| SHA256 | `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52` | 与 `muon_transition_experiment/outputs/m0/muon_freeze.json` 冻结值一致 |
| 版本 | torch 2.11.0（wheel，CUDA 13.0），conda env `fr_minimind` | `python -c "import torch; ..."` 实测 |
| 使用方式 | `from torch.optim import Muon`，无本地拷贝、无 patch、无更新规则包装 | `src/muon_transition/optimizers.py:23-37`（旧项目，只读核对） |
| 备选实现 | LlamaFactory third_party（注册未用）；skewadam 变体（拒绝） | `docs/PROVENANCE.md` §3 |

**结论（实测）**：本轮 T0 的 `native_muon` 路径绑定此文件此 SHA；T0 运行期间未升级、未修改环境。

---

## 2. 实际 G→M→H→D→参数更新 全链条（对照表）

依据 `_muon.py`（SHA 38e95918…）`_single_tensor_muon`（L304-318）与
`_zeropower_via_newtonschulz`（L31-70），以及生产调用方 `optimizers.py`、`m1_runner.py`。

| 阶段 | 安装代码语义 | 代码位置 | 与原大纲 §4 的对应 |
|---|---|---|---|
| 输入 G | `p.grad`，fp32（bf16 autocast 下参数/梯度仍 fp32） | `_muon.py:305`；M1/M2 生产循环 `m1_runner.py:140-145` | \(G_t\) |
| 动量 M | `buf.lerp_(grad, 1-momentum)` → \(M_t=\beta M_{t-1}+(1-\beta)G_t\) | `_muon.py:310` | **EMA 约定，与原大纲 (4) 相同** |
| Nesterov H | `update = grad.lerp(buf, momentum)` → \(H_t=(1-\beta)G_t+\beta M_t\)，**用更新后的 \(M_t\)** | `_muon.py:311` | 与原大纲 \(H_t\) 相同 |
| 归一化 | `X = H.bfloat16()` 后 `X.div_(X.norm().clamp(min=eps))`，eps=1e-7 | `_muon.py:55-59` | **与原大纲 E1 的 \(\|H\|_F+\epsilon\) 不同**（见 §4 分叉） |
| 转置 | rows>cols 时先转置（使 rows≤cols），NS 后转回 | `_muon.py:56-57, 68-69` | 原大纲未显式；2×2 不触发 |
| NS 迭代 | \(X_{k+1}=aX_k+(bA_k+cA_k^2)X_k\)，\(A_k=X_kX_k^\top\)，系数 (3.4445, −4.775, 2.0315)，5 步，**BF16 内部执行**，`addmm` 实现 | `_muon.py:61-66` | 与原大纲 §8 一致（原大纲为实数） |
| 谱算子输出 D | NS 输出（BF16），**不乘形状因子** | `_muon.py:313` | 本轮 T0 的 D 定义 = 此量 |
| 形状因子 | `adjusted_lr = lr·sqrt(max(1,A/B))`（adjust_lr_fn=None→"original"）**并入 lr，不进 D** | `_muon.py:73-84, 315` | 原大纲 \(s(r,c)\) 作用位置不同：大纲乘在 D 上，安装乘在 lr 上。数学上等价于更新量 = lr·s·D，但 D 本身不含 s。2×2 时 s=1 |
| 参数更新 | `param.mul_(1-lr*wd)` 后 `param.add_(update, alpha=-adjusted_lr)`；**WD 用未调整 lr** | `_muon.py:317-318` | 本轮合成实验 wd=0、lr=1 仅用于对齐验证 |

### 2.1 已识别的文档—代码不一致（不静默改写，列明分叉）

1. **类 docstring 写 \(B_t\leftarrow\mu B_{t-1}+g_t\)（未归一化）**，实际代码是 `lerp`
   （EMA，\((1-\mu)\) 加权）。旧项目 M0 Gate1 已识别并数值验证（NS 尺度不变性下两者
   更新方向一致）。本研究全部采用**代码事实**（EMA）。`_muon.py:220` vs `_muon.py:310`。
2. **原大纲 E1 归一化 \(\|H\|_F+\epsilon\) 与安装版 `clamp(min=eps)`（除以
   \(\max(\|H\|_F,\epsilon)\)）不是同一映射**。本轮 T0 的 `finite_ns_fp64` 与
   `native_muon` 均按安装版 clamp 语义；原大纲 E1 公式仅作为另一映射的理论式保留，
   未在本轮数值路径中使用。二者在 \(\|H\|_F\gg\epsilon\) 的光滑分支上导数相同
   （\(E/n-H\langle H,E\rangle/n^3\)），在阈值附近不同；本轮测试已分别覆盖
   （`test_normalization_smooth_branch`，阈值/秩亏单列不进主矩阵）。
3. **理想 polar 与有限 NS 是两个映射**（map(S)：polar(S)=I；NS5(S)=diag(0.6888,1.1142)），
   全部结果分开报告，不互相替代。

### 2.2 momentum buffer 归一化约定与 M0 坐标换算

- 安装 buffer 即 EMA 坐标（\(\beta M+(1-\beta)G\)），fp32（首见梯度时 `zeros_like` lazy init）。
- 本轮 T0 的共同初态 \(M_0=S\) 直接写入 buffer（EMA 坐标，无需换算）。
- 原大纲有限窗展开 \(H_t=\beta^{t+1}M_0+\sum_j w_jG_{t-j}\)，\(w_0=1-\beta^2\)，
  \(w_j=(1-\beta)\beta^{j+1}\) 对 EMA 坐标精确成立——本轮已实测（递推 vs 展开
  最大误差 < 1e-14，`test_direct_equals_window`；权重和 = \(1-\beta^{t+1}\)）。

---

## 3. 生产平台（M1/M2）训练与数据管线事实（只读核对）

平台：`muon_transition_experiment`（MiniMind 30M，`model_minimind.py`，
hidden 512 / 8 层 / GQA 8/4 / QK-Norm / SwiGLU / tied embedding）。

| 核对项 | 事实 | 证据 |
|---|---|---|
| 参数量 | 30,025,216；Muon 路由 56 个 2D 矩阵（26,738,688 参数）+ AdamW aux 34 张量（含 tied embedding） | `outputs/m0/routing_report.json` |
| 完整 optimizer batch | 每 step 128 个 window（4 micro-batch × 32 序列），每 window 256 token | `config.py:43-45`、`mixture.py` |
| micro-batch / 累积 | 4 micro-batch，`(loss/4).backward()` 累积后统一 clip+step；累积内顺序在精确算术下不改变梯度和 | `m1_runner.py:137-155` |
| loss 归一化 | 每 micro-batch 对有效 targets 的 mean CE；跨 micro-batch 等权（各 1/4） | `train_loop.py:9-18` |
| **input slots vs 有效 targets** | 每 window 256 个 input slots = BOS(1) + 255 个内容位；**有效预测 targets = 255/window**（`logits[..., :-1]` vs `targets = batch[..., 1:]`），无 padding（全打包窗口）；每 step 有效 targets = 128×255 = 32,640 | `data.py:62-82`、`train_loop.py:12-13`、M0 data_report packing_rule |
| 全局 clipping | `clip_grad_norm_(所有有梯度参数, 1.0)`：**跨 Muon+aux 全部参数联合全局范数**，clip 后才 step；保存剪裁后范数（日志 grad_norm 为剪裁前返回值） | `m1_runner.py:146-148` |
| batch 排列方式 | 每 step 固定配比 `round(p·128)` 个 A-window（域内顺序游标）+ 槽位交错为 seed=2030 的确定性 `randperm`（跨 run CRN 相同）；**域内 window 顺序是顺序读取，非 shuffle** | `mixture.py:35-52` |
| 域比例口径 | 每 step 精确固定（realized p_hat 精确），非仅全程固定 | `mixture.py` manifest |
| LR schedule | 32 步 warmup + cosine 到 10%；逐 step 写入两个 optimizer 的所有 group | `m1_runner.py:27-32, 130-133` |
| Muon 臂 aux | embedding/norm/bias 由 AdamW(β=0.9,0.95, eps=1e-8) 同 lr/wd 更新 | `optimizers.py:38-44` |

### 3.1 checkpoint 与日志字段（实测）

- **存在的完整 checkpoint**：M1 12 个 + M2 28 个 run 的 `final_state.pt`（step 2048 终点，
  `mte_state_v1`：model_state(91 tensors, fp32) + optimizer_states（Muon 56 个
  `momentum_buffer` fp32 + AdamW aux exp_avg/exp_avg_sq/step）+ 四路 RNG + cursor + payload
  SHA256）。已抽验一个 M2 Muon checkpoint 的结构（本轮实测读出）。
- **中间 checkpoint（step 512/1536 等）：不存在**。旧 M1/M2 循环只在训练终点保存一次。
  本轮**不**从 final_state 或稀疏日志推定任何中间态存在。v2 建议的 step512/1536
  状态属于"缺失、需后续授权补建"（见 ASSET_AND_COST_REPORT §2）。
- **每 100 步日志实际字段**：
  - `metrics.jsonl`：`{step, loss(该步 micro-mean), grad_norm, lr}`（每 100 步 + 前 5 步 + 末步）
  - `obs.jsonl`：`{name, step, grad_global_norm, sampled_gradient_singular_values(空dict),
    domain_gradient_cosine, domain_conflict_ratio, update_norm, weight_norm,
    update_weight_ratio, update_work_proxy, train_loss, lr, final}`
  - **没有任何逐步 G/M/H/D/Φ(H) 矩阵记录，没有动量 buffer 快照**——T2 必须新建采集。
  - `update_work_proxy = -g^T(θ_after-θ_before)`，命名固定为 proxy（不称焦耳）。

---

## 4. 对 T0 实现的约束落实

| 执行计划要求 | 落实 |
|---|---|
| D 不含 LR/WD/形状因子 | T0 D = `_zeropower_via_newtonschulz(H_t)` 原始输出；lr=1、wd=0 仅在完整 step 对齐验证中使用 |
| 捕获原始 NS 输出而非参数差分反推 | 主矩阵 D 直接捕获生产 NS 函数输出；完整 `Muon.step()` 轨迹独立运行，对齐检查 `max|Δparam + D| = 5.96e-8`（fp32 加法舍入量级，8/8 试例） |
| 原生 buffer/参数 dtype 与生产一致 | fp32 参数/梯度/buffer，NS 内部 BF16（CUDA），与 M1/M2 训练路径一致 |
| 不用自写近似冒充原生 | `native_muon` 调用安装包 `torch.optim.Muon` 与 `_zeropower_via_newtonschulz`，未改安装包 |
| finite_ns_fp64 与安装语义逐条对齐 | 转置分支（2×2 不触发，已记录）、clamp(min=1e-7) 归一化、(3.4445,−4.775,2.0315)×5、addmm 等价算子顺序；唯一差异 = 无 BF16 cast（FP64） |

## 5. 遗留差异清单（供审查）

1. 安装 docstring 与代码的动量记账不一致（§2.1-1）——已按代码事实执行。
2. 原大纲 E1（norm+eps）与安装（clamp）为不同映射——本轮主矩阵全部用 clamp 语义；
   E1 只作为理论对照保留在原大纲中。
3. v2 §3 引用的 PyTorch v2.11.0 在线源码描述（lerp/clamp/addmm）与服务器安装文件一致（本轮 SHA 核对）。
4. KellerJordan 仓库源（norm+1e-7）未直接使用；服务器无网络，以其在安装文件内的 permalink 登记。
