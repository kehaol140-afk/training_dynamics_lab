# TARGET—ACTION BRIDGE SPEC — 目标-动作桥设计（第一阶段）

**日期**: 2026-09-22。**状态**: 设计 + **一个待批准**的最小 GPU 校准方案。**0 已执行 GPU 工作**。
**定位（钉死）**: 小模型（v07 MiniMind-S 4.98M bank）、明确起点（bank anchors）、**独立 LM-NLL 轴的 continuation 效率**。
**不声称**: 完整预训练收益、下游能力、部署收益。无 Lyapunov/progress 证书，无全动作集 ε 界。

---

## 1. 数据角色与口径

| 角色 | 轴 | 数据 | 状态 |
|---|---|---|---|
| 训练 | L_train | D0 rows[0:4096]（train_indices=3936） | 既有，冻结 |
| 建模/调参（开发） | V_score + probe_q | eval_indices 偶/奇位划分内 | **保留开发属性，不更名、不冒充 final** |
| 停止监测 | **V_monitor（新建）** | **D0 rows[4096:8192]** | 新轴：D0 共 1,270,238 行、sha256 已核（6dd6716c…），rows≥4096 **从未被任何实验触碰**；ID 清单+sha256 **先注册后使用**；不用于任何动作选择 |
| 最终确认 | **V_final（新建，本轮不读）** | D0 rows[8192:12288] | 只注册 ID/hash；只在预注册 endpoint 访问 |

- **tokenizer/mask/loss**: Minimind vocab6400 ctx128；CE mean over valid tokens ÷ accum(2)；ignore_index=−100；每行 127 个预测位；nats；eval dropout=0；bf16 autocast + fp32 master。
- **重叠检查（可证）**: train 3936 + probe 32 + eval 128 ⊂ rows[0:4096]；V_monitor/V_final ⊂ rows[4096:] ⇒ 零重叠由构造保证，注册时输出 ID 清单 hash。
- V_score/V_eval 的既有开发属性（参与过 A0T/A0U/A0G 选择）**原样保留并披露**，不洗白。

## 2. tokens-to-target 目标定义与阈值来源

- 定义： G_target = {θ: V_monitor(θ) ≤ ℓ*}；τ = 首次达标 step；N_τ = Σ_{t<τ} n_update,t；目标 min E[N_τ]（协议 §5.2 口径）。
- **ℓ* 只有两条合法来源**: (a) 结果前约定；(b) 独立 baseline 开发资料。
- **执行顺序（防 POSTHOC）**: Phase A 只跑 **native 臂** on V_monitor → 形成 baseline 曲线 → 由该曲线分位数定 ℓ*（写入 Phase B 协议并冻结）→ Phase B 才允许读候选臂结果。**禁止看候选臂挑 ℓ***；阈值敏感度分析一律标 POSTHOC_EXPLORATORY。
- **无阈值依据时**: 先做**固定 token 预算可行性校准**（matched-budget 末端质量比较），不宣称达标时间。

## 3. 未达标 / 稀疏评估 / 失败成本报告

- hit_count/total；右删失保留（censor budget、固定预算末端质量）；**禁止**把 step 间线性插值当精确 τ（CPU T7）。
- 连续两次达标 = 预定监测上的持续确认，≠永久达标。
- NaN/OOM/崩溃单列，不删后平均；probe/candidate/eval tokens 全部入账（CPU T6）；wall-time 与 joules 分列（joules 暂无表，标缺失）。

## 4. 信息时点： "当前 batch backward 后、参数 step 前决定 LR"

**runner 适配性（已核代码，0 GPU）**: `a0k_closed_loop_runner_v01.py:281-310` 的循环为 per-micro: `set_native_micro_lr → forward → backward`；当 `micro_idx % accum == 0`（第 2 个 microbatch 的 backward 之后、optimizer.step() 之前）执行 `unscale_ + 克隆 native_gradient + 干预注入`。**该决策点正是"本步梯度已知、参数未更新"的窗口，A0K 已在同一窗口注入**。accum=2 下 lr 由 optimizer.step() 消费 ⇒ 本窗口内仍可改写本步 lr。✅ 适配，无需改 runner 结构。
（细节约束： accum=2 时有效 LR 粒度 = optimizer step；per-micro 不同 multiplier 不作用于同一次 step()。）

**候选位移**（协议 §3.2 同式，**仅描述一步位移**）:
s_t(u) = −η_native·exp(u)·(P_t·mbar_t + wd·θ_t)

| 输入 | 时点 | 类别 |
|---|---|---|
| θ_t, m_t→mbar_t, v_t→P_t, η_native(t)（get_lr）, wd | 决策窗口内全部已知 | **当前已知** |
| 本步累计梯度 g_t（两 microbatch backward 后） | 决策窗口内已知 | **当前已知** |
| 响应信号 chi/h_j（H=4 伴随） | 需额外 probe（forward/backward/HVP） | **需额外 probe，计费** |
| V_score/V_monitor/V_eval 读数 | 8 batch eval forward | **需额外 probe，计费** |
| 实际 gain（事后） | step 后真实结果 | **未来真实结果** |
| 未来 batch 梯度 | 不可得 | 未来真实结果 |
| 未来流 chi 模拟（A0K 式 rollout） | 可预模拟 | **可预模拟，必须计费** |
| 候选动作评分 rollout | 可预模拟 | **可预模拟，必须计费** |

**明确**: s_t(u) 是一步位移模型。**不**据此宣称任何长期动作已最优；无 B2 证书、无 B1 ε 界。

## 5. 局部模型如何服务目标相关判断（不要求长期轨迹预测先通过）

- **用途限定为三**: (i) 小固定动作集上的**排序信号**（A0S: C_t→G_t，H=4 Spearman 0.762/0.818，仅排序非证书）；(ii) **动作门控**（|signal| < 预注册 floor ⇒ 回退 native）；(iii) 预算分配（响应幅度大处采样更密）。
- **局部优势外推长期的风险检验（写进协议）**: 固定预算臂 + 每 64 步嵌套检查点；预注册**反转检查**（E1 式跨窗口符号翻转 ⇒ 该动作窗自动关闭）；per-seed 报告，不 pooled；滚动 sign/Pearson 低于 floor 自动回退。
- 失效发现路径： 嵌套检查点读出 → 信号-收益滚动相关断裂 → 回退 native 并记录。

## 6. 待批准最小 GPU 校准方案（objective feasibility study）

**无可执行 GTTR 策略**（chi 需未来流模拟；现存仅旧冻结 sign 规则）⇒ 按协议 §9.4 标 **objective feasibility study**。旧 sign 规则只作**对照臂**，不冒充新 GTTR 策略。

| 项 | 内容 |
|---|---|
| 目的 | 校准"目标-动作桥"在最小时域是否承载信号；产出 Phase B 的 ℓ* baseline 资料（native V_monitor 曲线） |
| 起点 | 预注册 4 fork 点 = seeds {2030, 2031} × anchors {t46, t82}（bank 既有，全用，非 R/C 挑选）；每 fork 3 条 CRN streams |
| 臂 | A native；B fixed +5% LR（最有依据固定动作）；C 旧冻结 sign(chi) 规则（**对照**，非新策略）。3 臂 |
| 观测终点 | **256 optimizer steps**，依据 = 2× 现有同协议最长窗（64=B0/C0 窗；128=A0K/L/N 窗；256=下一倍点），嵌套检查点 @64/128/256。**不填 512/1024**（无 baseline/计划量/目标距离依据） |
| 备选（若 256 不批） | **128 步变体**：同现有窗，仅增 V_monitor 轴+预注册；成本减半；标注"不加新时域信息" |
| token 轴 | 1024 tok/optimizer step（accum=2）；256 步 = 262,144 update tokens/rollout；eval/probe 另计 |
| 评估 | V_score / V_eval / V_monitor @每检查点；每轴 64 样本=8 batches forward |
| seeds/streams | anchor seeds 2030/2031；streams 7110/7111/7400（A0T CRN 惯例） |
| 统计单位 | 独立 run/fork（4×3=12 forks/臂），per-seed 报告；功效不足给区间+pilot 标签，不制造 p<0.05 |

**最坏预算（现场计价，封顶）**:

| 资源 | 256 步方案 | 128 步变体 |
|---|---|---|
| optimizer updates | 3 臂 × 4 forks × 3 streams × 256 = **9,216** | **4,608** |
| smoke（2×, 历史惯例） | 256 | 128 |
| 失败/重试津贴（历史： B0.5 9, B0.6 64, C0 33） | 128 | 128 |
| **updates 合计** | **≤ 9,600** | **≤ 4,864** |
| eval batches（fwd only） | 3 臂×4×3×3 检查点×3 轴×8 = 2,592 | 1,296 |
| 新 HVP / 新 grads | **0**（冻结 sign 规则用既有 chi；无自适应臂） | 0 |
| wall-time（worst, B0 速率 0.08s/upd + eval ~0.15s/batch + 2× margin） | ≈ 2,700 s（45 min） | ≈ 1,500 s（25 min） |

- **预算性质**: 独立新授权轮次；与第一轮 18,000（已结账，更正 18,730）、C1（45）、C1L（106）**分开记账**，不占用、不追溯。
- **不捆绑**: 不含 1B/规模扩展；不含下游评测（V_final 本轮不读）；不含新 optimizer/架构/控制器路线。
- **批准后第一步**: 注册 V_monitor/V_final ID 清单 + sha256 + Phase B 的 ℓ* 规则草案（结果前冻结），再动 GPU。

---

**需要用户批准**: 上表 updates/evals/wall-time 最坏预算（二选一： 256 步主方案 ≤9,600 updates / 128 步变体 ≤4,864 updates），以及 V_monitor/V_final 的数据注册。批准前 0 GPU。
