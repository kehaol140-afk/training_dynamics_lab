---
title: "Transformer-AdamW 局部涨落-响应关系实验计划（AI执行版）"
version: "v2.0"
date: "2026-09-01"
language: "zh-CN"
status: "PROTOCOL_DRAFT_FOR_LAB_WORKSTATION"
primary_goal: "从 native AdamW 完整状态出发，严格检验自然 mini-batch 涨落能否预测局部人工脉冲响应，并确定关系在哪个训练位置、方向与时间尺度成立或失效。"
source_basis:
  - "2026-07-21《使用非平衡态动力学思想优化 Transformer》"
  - "2026-08-11《非平衡多时间尺度记忆优化——Transformer训练实验总结、证据边界与停止裁决》"
  - "2026-08-19《使用非平衡态动力学思想优化 Transformer》"
important_scope_change_from_v1:
  - "不再把约250M token预设为确定Phase transition，只作为crossover-candidate采样位置。"
  - "新增方向异质性审计与mean/tail双响应门禁。"
  - "明确V2A/V2B旧代码不是新实验科学主干；实验从native AdamW重新开始。"
  - "旧8.11结果只作为先验与负对照，不作为FDT/FR成立证据。"
---

# 0. 给 Codex / GLM / Claude Code / 其他代码智能体的执行总指令

你正在实现一个 **从头建立、严格配对、可证伪、分层停止** 的 Transformer-AdamW 涨落-响应实验。

不要把经典平衡态涨落-耗散定理（FDT）成立写成前提；不要把 8.11 的 V2A、V2B、χ/Q、Phase 标签直接当作本实验的自然涨落或 susceptibility；不要先设计在线控制器。

本实验只检验：

> 从同一完整 native AdamW 训练状态出发，不同 mini-batch 随机流造成的自然梯度涨落，是否包含关于一个独立、极小、正负对称的梯度脉冲在未来 1、8、32 个 optimizer step 上产生何种响应的信息；这种关系是否跨 random stream、跨训练 seed、跨训练位置、跨方向保持，并且平均响应是否与尾部风险一致。

## 0.1 科学上不允许擅自改变的约束

1. **实验主干从 native AdamW 开始。** V2A/V2B 不是 baseline，也不是 perturbation 本身。
2. **完整状态恢复**：至少恢复 `theta`、AdamW `m/v`、optimizer step、scheduler/LR、数据/采样状态（若原训练支持）、Python/NumPy/Torch CPU/Torch CUDA RNG、AMP/GradScaler 状态。
3. **组间放开随机性，组内锁定随机性**：
   - 不同 `stream_id` 使用不同 future batch / dropout 随机流，采样自然随机性；
   - 同一 `stream_id` 内 `baseline / plus / minus` 必须使用相同 batch 顺序、dropout RNG、AMP 路径和其它可控随机性。
4. **人工扰动是单步 impulse**：只在指定 optimizer step 注入一次，之后恢复原始 native AdamW；不得把 V2A 的持续记忆混合系数 α 当作 ε。
5. **默认注入点**：AMP unscale 之后、gradient clipping 之前、AdamW moment update 之前。必须按真实代码审计确认。
6. **控制方向 q 不使用正式 validation/test 梯度。** 第一版使用冻结 train-probe pool 构造。
7. **H 固定为 {1,8,32}**。一条 32-step rollout 同时采样三个 horizon。
8. **训练位置只称 early / crossover-candidate / mid / late。** 不得预先宣称存在确定 Phase A/B 相变。
9. **mean 与 tail 分开判定**：至少记录 eval mean loss 和 token-loss tail（推荐 p99/CVaR99）。平均改善不能覆盖尾部恶化。
10. **seed 是高层独立重复单位，stream 是 state 内局部重复单位。** 不得把 token、validation sample 或连续 step 当成跨轨迹独立重复来夸大显著性。
11. pilot 只允许冻结：`epsilon/rho`、q 构造、eval-probe、统计门槛。正式 discovery/holdout 后不得看结果再改。
12. 只有通过 `replay -> measurable fluctuation -> local linearity -> directional audit -> B->chi -> holdout -> mean/tail` 后，才允许 E7 控制实验。
13. 所有失败 seed/checkpoint/horizon/direction 必须保留。

---

# 1. 研究背景与 8.11 对本实验的影响

## 1.1 7.21 / 8.19 已建立的边界

已有工作反复说明：

- 温度、梯度范数、耗散代理可以描述训练状态，但不能作为“越小越好”的单调控制目标；
- 样本温度、梯度方向一致性、有效 batch 降温、梯度去相干等直接控制没有稳定泛化收益；
- 离线 checkpoint surgery 的相关信号不能直接推出在线 freeze/scale 控制有效；
- 受控随机递推视角要求区分“自然轨迹预测”和“动作响应预测”。

因此新实验的缺失桥梁是：

```text
自然随机涨落 -> 直接测量的因果响应 -> 跨条件泛化 -> 才讨论控制
```

## 1.2 8.11 已经给出的先验（只作为先验，不当作 FR 证据）

8.11 在 75% 训练位置、单 checkpoint、单 seed、128 optimizer step、冻结 proxy validation 下观察到：

- strict V2A (alpha=0.05) 相对 native AdamW 有小幅平均 proxy-loss 优势；
- 关闭快记忆混合后约 84.24% 的既有优势洗脱，提示响应主要依赖持续驱动，而非永久“相变”；
- 响应随位置和剂量变化；强脉冲不等价于持续低剂量；
- 模块响应各向异性，V/O 和 MLP Down 效应密度较高，但稀疏组合不能保留全部收益且 CVaR99 恶化；
- χ/Q、sleep/wake 等廉价标量状态对未来优势的 held-out 预测不足。

这些结果只支持：

```text
局部 response 值得测；response 很可能有方向、位置、时间尺度和尾部风险依赖。
```

它们 **没有** 建立：

```text
自然 fluctuation B -> 人工 susceptibility chi
```

原因包括：旧实验没有 ±epsilon 中心差分、没有多 stream 自然涨落估计、主要是持续 memory-kernel intervention、缺少多 seed / 多训练位置泛化。

## 1.3 8.11 发现的旧方法问题，新实验必须修复

1. 单 seed + 单 checkpoint 不能支持跨轨迹规律；
2. 旧 95% CI 只能解释为冻结设置内局部不确定性，不能替代 seed-level 泛化；
3. `layer_temperature` 与 gradient norm 重合，旧 Phase 边界不能作为物理相变证据；
4. χ/Q held-out 失败说明“人工设计 scalar -> future benefit”不能再凭直觉；
5. mean improvement 与 CVaR99 harm 可能同时存在；
6. V2A 是 sustained kernel intervention，不是单步 impulse susceptibility。

---

# 2. 旧 8.11 代码：继承什么，不继承什么

## 2.1 推荐决策：科学实验从新仓库开始

实验室工位电脑创建全新仓库，例如：

```text
fr_minimind_lab/
```

不要直接复制旧 V2A/V2B 仓库作为主实验仓库。原因：

- 新问题是 native AdamW 的自然涨落与 impulse response；
- 旧代码包含 fast-state、legacy/strict 分支、动态门控、稀疏模块等大量与新主问题无关的自由度；
- 从旧优化器分支继续改容易把历史状态或实现差异混进 baseline。

## 2.2 如果旧代码可取得，只允许“择优移植”的工具

先审计，再复制以下已验证思想/工具：

- checkpoint 完整恢复与 resume 检查；
- optimizer state (`m/v/step`) 一致性审计；
- RNG snapshot / hash；
- `alpha=0` / no-op 路径的 bitwise 或 near-bitwise 对齐测试思路；
- frozen-gradient correctness test；
- 逐阶段 hash / 参数增量误差定位；
- ABBA / paired timing 仅用于后期工程计时；
- source-of-truth / protocol hash / RUN_CONFIG 记录方式。

**复制前要求**：每个工具都必须能脱离 V2A 状态运行，并通过新仓库测试。

## 2.3 不建议迁移的旧代码

- fast EMA / dual-memory optimizer 主实现；
- legacy fast bootstrap；
- V2B χ/Q controller；
- sleep/wake sentinel；
- sparse module gate；
- strong heat pulse；
- 旧 Phase classifier；
- 用 `layer_temperature` 命名 gradient norm 的逻辑。

这些可以作为未来 negative-control 或机制扩展，但不得进入 E0-E5 主实验路径。

---

# 3. 核心建模

## 3.1 完整受控随机状态

记：

```math
Z_t=(\theta_t,m_t,v_t,s_t^{data},s_t^{rng},s_t^{amp},t)
```

原始训练：

```math
Z_{t+1}=F_t(Z_t,u_t,\omega_{t+1})
```

这里不假设平稳、凸、高斯噪声、详细平衡或热浴。

## 3.2 冻结方向 q_t

用与正式 eval 分离的 train-probe pool：

```math
g_t^{probe}=\nabla_\theta L_{probe}(\theta_t)
```

```math
q_t=\frac{g_t^{probe}}{\|g_t^{probe}\|_2+\epsilon_q}
```

同一个 `(train_seed, checkpoint)` 的所有 stream 与 branch 使用同一个 q。

### 第一版方向层级

- Primary: `FULL_MODEL_Q`；
- Secondary diagnostic（只在主实验之后/并行低成本记录）：`ATTN_VO_Q`, `MLP_DOWN_Q`, `ATTN_QK_Q`, `MLP_GATE_UP_Q`；
- Negative control: norm-matched random direction / orthogonal random direction。

不要只因 8.11 的 V/O / Down 效应密度较高就把它们设成唯一方向。

## 3.3 自然涨落

从完全相同的 `Z_t` 出发，生成 R 条不同 future random streams：

```math
g_t^{(r)}, \quad r=1,\ldots,R
```

方向投影：

```math
s_t^{(r)}=\langle q_t,g_t^{(r)}\rangle
```

自然涨落：

```math
\delta s_t^{(r)}=s_t^{(r)}-\bar s_t
```

自然尺度：

```math
\sigma_{q,t}=Std_r[s_t^{(r)}]
```

同时记录 `skewness/kurtosis/MAD/p95/p99/clipping-conditioned distribution`。

## 3.4 未来响应 observables

### Primary dynamical response

```math
X_{t,H}^{\theta,(r)}=\langle q_t,\theta_{t+H}^{(r)}-\theta_t\rangle
```

### Secondary optimizer responses

- q 方向一阶矩变化；
- preconditioned q 上的等效更新尺度；
- gradient norm / q projection / cosine 等摘要。

### Functional mean response

```math
Y_{t,H}^{mean,(r)}=L_{eval}(\theta_t)-L_{eval}(\theta_{t+H}^{(r)})
```

### Functional tail response（新增，来自 8.11 教训）

对固定 eval-probe 的 token loss：

```math
Y_{t,H}^{tail,(r)}=CVaR99_t-CVaR99_{t+H}^{(r)}
```

也同时输出 p95/p99。若 `Y_tail < 0` 表示尾部恶化。

## 3.5 自然关系 B

对于任意响应 Y：

```math
B_{t,H}^{Y}=\frac{Cov_r(\delta s_t^{(r)},Y_{t,H}^{(r)})}{Var_r(\delta s_t^{(r)})+\epsilon_B}
```

这是 observational slope，不是 causal susceptibility。

## 3.6 单步 ±epsilon impulse

对每个 `stream_id=r`，建立三条 common-random-number branch：

```math
g_t^{(r,0)}=g_t^{(r)}
```

```math
g_t^{(r,+)}=g_t^{(r)}+\varepsilon_t q_t
```

```math
g_t^{(r,-)}=g_t^{(r)}-\varepsilon_t q_t
```

仅此一步修改 raw gradient；之后恢复 native AdamW。

自然尺度标定：

```math
\varepsilon_t=\rho\sigma_{q,t}
```

pilot 初始：

```text
rho in {0.10, 0.25, 0.50, 1.00}
```

8.11 已提示强剂量可能与小剂量性质不同，因此新版比 v1 增加 0.10 点。

## 3.7 因果 susceptibility

```math
\chi_{t,H}^{Y}=\frac{E_r[Y_{t,H}^{(+)}]-E_r[Y_{t,H}^{(-)}]}{2\varepsilon_t}
```

对 `X_theta / Y_mean / Y_tail` 分别计算。

## 3.8 局部线性与奇对称

```math
\Delta Y_+=Y_+-Y_0,\quad \Delta Y_-=Y_--Y_0
```

理想局部线性：

```math
\Delta Y_+\approx-\Delta Y_-
```

奇对称误差：

```math
E_{odd}=\frac{|\Delta Y_++\Delta Y_-|}{|\Delta Y_+-\Delta Y_-|+\epsilon}
```

pilot 后冻结阈值，不在代码中预设一个为了过门禁的数值。

## 3.9 最终目标不是经典 FDT 等式

第一版检验：

```math
\chi_{t,H}\stackrel{?}{=}f_H(B_{t,H},Z_t,q)
```

最小模型：

```math
\chi_{s,p,H}=\alpha_H+\kappa_HB_{s,p,H}+\xi
```

若方向/位置依赖明显：

```math
\chi=\alpha_H+\kappa_HB+\gamma_{position,H}+\delta_{position,H}B+\eta_{direction,H}+u_{seed}+\xi
```

不要强求全阶段统一 `K`。

---

# 4. Random stream 设计

## 4.1 组间：不同 stream 采样自然随机性

```text
same Z_t
  |- stream_000 -> B12, B81, B04, B56, ...
  |- stream_001 -> B37, B02, B99, B15, ...
  |- stream_002 -> B08, B42, B27, B05, ...
```

## 4.2 组内：baseline / plus / minus 锁定相同 future randomness

```text
stream_000 baseline: B12, B81, B04, B56, ...
stream_000 plus:     B12, B81, B04, B56, ...
stream_000 minus:    B12, B81, B04, B56, ...
```

尽可能锁定：Python、NumPy、torch CPU/CUDA、dataloader generator/sampler、dropout、GradScaler。

核心口诀：

```text
组间放开随机性，组内锁定随机性。
```

---

# 5. 训练位置与 seed：取消“预设相变”

## 5.1 训练位置

优先使用现有 checkpoint；如果实验室电脑没有旧 checkpoint，则用一次 native MiniMind baseline 训练生成四个冻结状态。

建议相对进度：

| position_id | 建议位置 | 解释 |
|---|---:|---|
| EARLY | 10%-15% | 强非平稳早期 |
| CROSSOVER_CANDIDATE | 25%-35% | 旧诊断转折附近，只是候选位置 |
| MID | 60%-70% | 中后期 |
| LATE | 85%-95% | 后期低速演化区 |

若沿用旧 token 预算约 1B，可近似用 100M / 250M / 650M / 950M，但报告中用 actual_tokens，不宣称 250M 是物理相变点。

## 5.2 seed 分工

推荐：

```text
pilot: 1个seed x MID
exploration/discovery: 3个seed x 4 positions
holdout: 1个完全未参与冻结的seed x 4 positions
optional final confirm: 第5个seed（若holdout曾被提前查看）
```

如算力有限，先用 2 discovery + 1 holdout，但不能用 stream 数量冒充 seed 数量。

---

# 6. 实验阶段 / 决策门禁（v2）

# E0. 新工位环境 + native AdamW 基线审计

## 目的
证明新机器上 baseline 可复现，科学主干没有继承 V2A/V2B 隐状态。

## 必做

- 记录 GPU/driver/CUDA/PyTorch/precision；
- 冻结 MiniMind commit；
- 跑 native AdamW smoke；
- 完整 checkpoint save/resume；
- 两条 action=0 branch 在相同 stream 下 32 step 一致；
- `rho=0` perturbation 路径 no-op；
- 确认真实顺序：`backward -> unscale -> perturb -> clip -> optimizer.step -> scheduler.step`（按实际代码修改记录）。

## Gate
`E0_PASS` 后才能做 E1。

# E1. 自然涨落 pilot

```yaml
seed_count: 1
position: MID
R: 16
branches: [baseline]
H: [1,8,32]
q: FULL_MODEL_Q
```

输出：`s_r`, `sigma_q`, 分布形状、clipping、B_theta、B_mean、B_tail、batch/RNG hash。

若 eval mean/tail 噪声太大，优先扩大固定 eval-probe，不要先放大 ε。

# E2. epsilon 局部线性扫描

```yaml
rho: [0.10,0.25,0.50,1.00]
R: 8-16
branches: [baseline,plus,minus]
```

检查：

- plus/minus 奇对称；
- `chi(rho)` 在小 rho 间是否相容；
- clipping rate 是否因扰动改变；
- 参数 response 是否稳定；
- mean/tail 是否出现方向分裂。

若全都非线性，缩小到 `{0.025,0.05,0.10}` 做最后一次 pilot；仍失败，标记 `NO_LOCAL_LINEAR_REGIME`，停止简单 FR 主线。

# E3. 方向异质性 audit（8.11 新增要求）

目的：判断 susceptibility 是否强烈依赖 q。

至少比较：

```text
FULL_MODEL_Q
NORM_MATCHED_RANDOM_Q
```

预算允许再加：

```text
ATTN_VO_Q
MLP_DOWN_Q
ATTN_QK_Q
MLP_GATE_UP_Q
```

注意：模块方向只作为机制诊断；不得把单模块高效应密度直接解释为唯一机制。

输出：`chi_theta / chi_mean / chi_tail` 的 direction profile。

# E4. Discovery：跨训练位置、跨 seed 的 B -> chi

推荐：

```text
3 discovery seeds x 4 positions x R=16 x 3 branches x 32 steps
```

每 state 输出：

```text
B_theta(H), B_mean(H), B_tail(H)
chi_theta(H), chi_mean(H), chi_tail(H)
odd_symmetry_error(H)
paired stream CI
response decay / sign reversal across H
```

还要建立简单预测 baseline：

```text
position + lr + current loss + grad norm
```

可选负对照加入旧 8.11 的 scalar χ/Q 摘要（如果代码可安全重算），目的是检验新 B 是否比旧廉价状态信号有增量价值；不是把 χ/Q 当主指标。

# E5. Frozen holdout seed

正式查看前冻结：q、rho、H、eval-probe、tail 定义、拟合模型、pass/fail。

必须回答：

1. discovery 的 `B -> chi` 是否预测 held-out seed？
2. 是否至少在一个预先定义训练位置范围内符号稳定？
3. B 是否比 `progress/lr/loss/grad norm` baseline 提供增量预测信息？
4. `chi_theta` 成立而 `chi_mean` 不成立时，只允许动力学层结论；
5. `chi_mean > 0` 但 `chi_tail < 0` 时，禁止进入控制。

# E6. 机制定位（关系失败或只部分成立时）

## E6-A Gradient pulse vs Parameter pulse

Gradient pulse 会改变 `m/v/theta`；Parameter pulse 保持 baseline `m/v`，只对参数做等效局部位移。

解释：

- parameter pulse 稳、gradient pulse 不稳 -> AdamW memory/preconditioning 可能是破坏源；
- 仅 late 稳 -> state/nonstationarity 依赖；
- 所有都不稳 -> 一维投影不足或强高维耦合。

## E6-B 多维扩展条件

只有当一维 `B_q -> chi_q` 失败、但方向 audit 显示明显结构时，才考虑低维投影向量：

```math
\mathbf{B}_H,\quad \mathbf{\chi}_H
```

或 covariance-response matrix。不要一开始就估计全高维矩阵。

## E6-C response relaxation

8.11 washout 提示有限记忆。若 `|chi_1|>|chi_8|>|chi_32|` 或出现系统性衰减，可拟合辅助响应时间：

```math
|\chi_H|\approx A e^{-H/\tau_R}
```

仅在数据支持时使用，不预设指数模型。

# E7. 在线控制（默认禁止）

必须同时满足：

- E2 有线性区；
- E5 held-out 预测通过；
- mean 与 tail 无冲突；
- effect 大于数值/评估噪声；
- 控制幅度始终在验证过的 linear-response scale 内。

否则不启动。

---

# 7. 统计规范（修正 8.11 的外推风险）

## 7.1 state 内局部统计单位 = random stream

同一 stream 的 plus/minus/base 是配对数据。

```text
D_pm_r = Y_plus_r - Y_minus_r
D_pb_r = Y_plus_r - Y_base_r
D_mb_r = Y_minus_r - Y_base_r
```

用 paired bootstrap over streams 估计 state-level CI。

## 7.2 泛化统计单位 = training seed

- checkpoint/position 属于 seed 内重复；
- stream 属于 state 内重复；
- validation token/sample 不是独立训练重复。

因此“跨 seed 规律”不能通过对 token/sample bootstrap 得到。

## 7.3 旧 8.11 数值如何使用

- `0.000468...` 可作为 effect-size 先验和 eval precision 参考；
- 旧 CI 不能当新实验 power 的唯一依据，除非确认其 bootstrap 单位；
- 不拿旧 V2A 胜负作为新 FR 的 success criterion。

## 7.4 成功分层

- **Dynamical FR**：B 能预测 `chi_theta`；
- **Functional FR**：B 能预测 `chi_mean`；
- **Robust Functional FR**：同时 `chi_mean` 与 `chi_tail` 无冲突；
- **Generalizable FR**：held-out seed 有增量预测价值；
- **Control-eligible FR**：前四层都通过后才允许控制。

---

# 8. 推荐代码目录（新工位）

```text
fr_minimind_lab/
├─ README.md
├─ BASELINE_LOCK.md
├─ configs/
│  ├─ workstation.yaml
│  ├─ paths.example.yaml
│  ├─ pilot.yaml
│  ├─ discovery.yaml
│  ├─ holdout.yaml
│  └─ protocol_frozen.yaml
├─ src/
│  ├─ minimind_adapter.py
│  ├─ state_restore.py
│  ├─ rng_snapshot.py
│  ├─ stream_manifest.py
│  ├─ probe_direction.py
│  ├─ perturbation.py
│  ├─ branch_runner.py
│  ├─ observables.py
│  ├─ tail_metrics.py
│  ├─ eval_probe.py
│  ├─ logging_schema.py
│  └─ statistics.py
├─ scripts/
│  ├─ 00_environment_and_baseline_audit.py
│  ├─ 01_replay_audit.py
│  ├─ 02_natural_fluctuation_pilot.py
│  ├─ 03_epsilon_scan.py
│  ├─ 04_direction_audit.py
│  ├─ 05_run_discovery.py
│  ├─ 06_freeze_protocol.py
│  ├─ 07_run_holdout.py
│  ├─ 08_parameter_pulse_mechanism.py
│  └─ 09_control_ONLY_IF_PASS.py
├─ tests/
│  ├─ test_native_baseline_identity.py
│  ├─ test_rng_replay.py
│  ├─ test_rho_zero_noop.py
│  ├─ test_branch_batch_identity.py
│  ├─ test_checkpoint_optimizer_restore.py
│  ├─ test_horizon_capture.py
│  ├─ test_tail_metric_deterministic.py
│  └─ test_no_v2a_state_leakage.py
└─ outputs/
   ├─ audit/
   ├─ pilot/
   ├─ direction/
   ├─ discovery/
   ├─ holdout/
   └─ mechanism/
```

---

# 9. 每条 branch 的最小日志 schema

```json
{
  "experiment_id": "...",
  "git_commit": "...",
  "environment_hash": "...",
  "train_seed": 44,
  "position_id": "MID",
  "checkpoint_tokens": 650000000,
  "stream_id": 3,
  "direction_id": "FULL_MODEL_Q",
  "branch": "plus",
  "rho": 0.25,
  "epsilon_abs": 0.000123,
  "q_norm": 1.0,
  "grad_norm_t": 0.0,
  "grad_q_projection_t": 0.0,
  "sigma_q_state": 0.0,
  "clip_triggered_t": false,
  "lr_t": 0.0,
  "horizon": 8,
  "theta_q_response": 0.0,
  "eval_mean_start": 0.0,
  "eval_mean_h": 0.0,
  "eval_mean_improvement": 0.0,
  "eval_cvar99_start": 0.0,
  "eval_cvar99_h": 0.0,
  "eval_cvar99_improvement": 0.0,
  "runtime_sec": 0.0,
  "batch_manifest_hash": "...",
  "rng_snapshot_hash": "..."
}
```

必须保存：`RUN_CONFIG`, `ENVIRONMENT`, `STATE_AUDIT`, `STREAM_MANIFEST`, raw branch metrics, state summary, statistics, report。

---

# 10. 在实验室工位电脑的执行顺序

## Day 0 / 环境

1. 新建仓库；
2. 冻结 MiniMind 版本与数据版本；
3. 记录 GPU/CUDA/PyTorch；
4. native AdamW smoke；
5. 找到/生成可 resume checkpoint。

## Stage A / 代码审计

让 AI 只做 `TASK-00`：输出真实训练顺序和 checkpoint/RNG 能力，不改科学逻辑。

## Stage B / E0 replay

先证明同 state + 同 stream + action=0 可以重复 32 步。

## Stage C / E1-E2 pilot

只在 1 seed × MID 做自然涨落和 epsilon scan。此时不要跑多 seed。

## Stage D / E3 direction audit

先 full q + random q；模块 q 作为第二层。

## Stage E / E4 discovery

扩展 3 seed × 4 positions。

## Stage F / freeze + E5 holdout

冻结协议后才打开 holdout seed。

## Stage G / E6 mechanism 或 E7 control

根据结果二选一，不同时开启多个新方向。

---

# 11. 给代码智能体的任务拆分

## TASK-00：只审计，不改代码

输出 `IMPLEMENTATION_AUDIT.md`：

- forward/loss/backward；
- AMP unscale；
- grad clip；
- optimizer.step；
- scheduler.step；
- checkpoint save/load；
- optimizer state；
- dataloader/sampler RNG；
- dropout/Torch RNG；
- eval 函数；
- determinism 限制。

并给出真实顺序：

```text
forward -> loss -> backward -> unscale -> [candidate injection point] -> clip -> optimizer.step -> scheduler.step
```

## TASK-01：实现 native deterministic replay

不要引入 V2A/V2B。完成 E0。

## TASK-02：实现 q / stream / natural fluctuation

只 baseline，不 perturb。

## TASK-03：实现 ±epsilon impulse

要求 `rho=0` no-op，plus/minus 同 stream batch/RNG hash 完全一致。

## TASK-04：实现 mean + tail observables

固定 eval-probe，输出 mean / p95 / p99 / CVaR99。

## TASK-05：实现 pilot + protocol freeze

E1/E2/E3 完成后生成 `protocol_frozen.yaml`，后续禁止自动改参数。

## TASK-06：正式 discovery/holdout

AI 只执行冻结协议，不根据结果自动 sweep。

---

# 12. 最终判断树

```text
E0 replay pass?
  no -> 工程修复，不做科学结论
  yes
   |
E1 natural fluctuation measurable?
  no -> 改观测方向/尺度一次；仍无则停止
  yes
   |
E2 local ±epsilon linear response?
  no -> 缩小epsilon一次；仍无则非线性/停止简单FR
  yes
   |
E3 response direction-dependent?
  yes/no -> 记录结构，不作为失败
   |
E4 B predicts chi in discovery?
  no -> E6: AdamW memory / high-D / nonstationarity
  yes
   |
E5 held-out seed + incremental prediction pass?
  no -> 轨迹特异/阶段特异，限制结论
  yes
   |
mean response and tail response consistent?
  no -> 机制结论，不进入控制
  yes
   |
E7 control-eligible -> 才做小幅控制
```

---

# 13. 本计划允许的最终结论语言

### 若强正结果

“在预注册的小扰动线性区内，native AdamW 的自然 mini-batch 涨落对局部 causal susceptibility 具有跨 seed 的增量预测信息，并在明确训练位置/方向上复现。”

### 若只在 late / 特定方向成立

“未观察到全训练统一的 FR law，但在低波动后期/特定方向上出现可复现的 state-dependent local fluctuation-response relation。”

### 若只预测参数 response

“自然涨落包含局部参数动力学响应信息，但该信息未稳定传递到 eval mean/tail；不能外推泛化控制。”

### 若 B 不能预测 chi

“局部人工响应可测，但 spontaneous fluctuation 与 intervention response 无稳定对应；结果支持 observational fluctuation 与 causal response 的分离。”

### 若连线性 response 都不存在

“在预注册尺度及更小补充尺度下未找到稳定局部线性响应区，因此经典线性涨落-响应近似不适用于当前 observable / AdamW 设置。”

禁止无证据使用：严格 FDT、物理温度、熵产生、确定相变、热浴、Kramers 逃逸、训练能耗节省。
