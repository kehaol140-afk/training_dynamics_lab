# F1_1B_REPLICATION_PREREG — 已取消（CANCELLED_BEFORE_ANY_1B_COMPUTE）

**状态：CANCELLED（2026-09-26，在开始任何 1B 计算之前）。以下原预注册正文完整保留作历史存档，其 gate 不再执行。**

- 取消原因：外部决策——1B/G1c checkpoint 不再下载、不恢复、不重建（磁盘与网络约束；旧 G1c 历史结果降级为 seed evidence，不作为新项目正式证据）。
- 取消时的执行状态（诚实记录）：
  1. 预注册冻结并 commit（b8ef9ec）；
  2. 下载循环启动后全程网络不可达（代理外网节点中断 >3h），**0 字节下载**，`external/peteish1/` 为空；
  3. 无任何 legacy 文件被写入或修改；
  4. stage 3–6 代码从未在 1B 数据上运行（仅在合成 tiny 模型上 dry-run 通过）。
- 后继：F1 改道为 `docs/F1_MECHANISM_CLOSURE.md`（既有 5M/26M/100M bank 上的机制闭环，规则冻结于 `configs/f1_mechanism_v01.json`）。

---

# F1 — 1B Replication 预注册（OLMo-2-0425-1B / peteish1）

- 冻结日期：2026-09-26（在任何 1B optimizer-state 读取、任何 1B sensitivity 计算之前）。
- 上游：F0 verdict `TANGENT_ILL_CONDITIONING_CONFIRMED_AT_SCALE`（docs/F0_EXISTING_BANK_SENSITIVITY_AUDIT.md）；legacy G1c feasibility + C-failure diagnostic（legacy `outputs/generalization_g1c_feasibility_v01/`）。
- 本轮唯一目标：确认 5M→26M→100M 的 conditioning scale trend 是否在 1B 延续，并对旧 G1c C-branch 失败给出 A–F 分类。**不开发新算法、不改 forward AdamW、不做 FRD/controller/data-mixing。**
- 红线：legacy 全程只读；不为通过 gate 调整任何阈值；所有 gate 先于结果冻结于本文。

## 0. 冻结的研究对象与比较基线（F0 已冻结数字）

一步 AdamW map 的 gradient→θ′ 通道在 c=0 锚点（下一步零梯度）的精确逐坐标对角
`G_i = lr(1−β1)/((1−β1^(t+1))·(√(v̂₁_i)+eps))`，`v̂₁ = β2·v/(1−β2^(t+1))`；
m 通道增益 = β1·G_i；θ 通道因子 (1−lr·wd)。

F0 参考值（EARLY/matched step 26，fp64）：

| 量 | 5M | 26M | 100M | 1B 预期比较点 |
|---|---:|---:|---:|---|
| σ_max(J₀) | 39.4 | 290 | 2198 | step380000（三 checkpoint 中最早） |
| A_random | 1.05 | 4.43 | 14.8 | step380000 |

已知 1B 结构差异（先于结果声明，影响解释但不改 gate）：β2=0.95（MiniMind 0.999）；untied lm_head（`transformer.ff_out.weight`）+ wte 存在精确零 v 坐标（G1c cdiag：vhat_min=0 每步成立）；深 clip regime（系数实测 0.003–0.012，MiniMind 0.7–1.0）。c=0 锚点上精确零 v 坐标的 `G_i = lr(1−β1)/(bc1·eps)` 是**中心差分对称导数**（c→θ′ 映射在该坐标有 |c| 型折点，单侧导数饱和于 `1/(a|c|+eps)·lr(1−β1)/bc1`，a=√((1−β2)/bc2)≈0.2236）——该折点收敛要求 ε_fd ≪ eps/a ≈ 4.5e−8，故本预注册的 fp64 ε 网格下探到 1e−12（F0 只到 1e−10）。

## 1. 冻结的 checkpoint / 模型 / 状态标识

| 项 | 冻结值 |
|---|---|
| 模型 | OLMo-2-0425-1B，stage1 run `peteish1`（AllenAI 官方，Apache-2.0） |
| 架构 | d_model 2048，n_heads 16，n_layers 16，vocab 100278，embedding_size 100352，msl 4096，SwiGLU fused ff_proj（mlp_hidden 8192），att_proj fused QKV，QK-norm，post-norm，untied head；131 张量 / 1,484,916,736 参数（legacy G1C 架构适配器语义，逐键 fail-fast 对齐） |
| checkpoints | step380000 / step950000 / step1530000（unsharded 四件套中 model.pt + optim.pt 需重下；config.yaml / train.pt 已在 legacy 只读保留，直接引用） |
| optimizer-state 源 | 官方 `optim.pt`（真实 exp_avg / exp_avg_sq / step，逐参数 fp32 存储） |
| exact state step | 存储值：380001 / 950001 / 1530001（torch 语义：下一次 step 的 bias-correction t = 存储值+1；phase0 已核验唯一性） |
| model dtype | 存储 fp32；GPU 梯度路径 bf16 autocast（legacy G1c 注册语义，seq512×8/mb×4mb/step）；primal CPU AdamW fp32 |
| optimizer-state dtype | 存储 fp32；分析读取按 fp32→fp64 精确升位；bf16/fp16 round-trip 单独统计（§5） |
| Adam hyper | β1=0.9，β2=0.95，eps=1e-8，amsgrad=false，全部 group 一致（phase0 核验） |
| weight decay | group0（130 参数，blocks/head/ln_f）wd=0.1；group1（wte）wd=0.0 |
| lr | 逐 checkpoint 读自 optim.pt param_groups 常数（step950000 已知 2.7699376575216877e-4；380000/1530000 下载后先记录再分析；无 scheduler state → constant-lr fallback，legacy 协议 6.5） |
| grad clipping | max_grad_norm=1.0 全局，torch clip_grad_norm_ 语义（CLIP_DENOM_EPS=1e-6）；深 clip 预期系数 0.003–0.012 |

## 2. 冻结的下载与完整性

- 来源（legacy 冻结复核 manifest 逐字继承）：`https://olmo-checkpoints.org/ai2-llm/peteish1/step{N}-unsharded/{model.pt,optim.pt}`
- 目标路径（**新项目内，不写 legacy**）：`optimizer_sensitivity/external/peteish1/step{N}_{model,optim}.pt`
- 尺寸/SHA256 锚点（来自 legacy `docs/CLEANUP_G1C_LARGE_FILES_20260925.md`，下载后强制校验）：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| step380000_model.pt | 5,939,703,038 | ca2fa02c8192651c7280e848e96c69f954451f784f26b97e8b11385c54470fa7 |
| step380000_optim.pt | 11,879,424,306 | 3e48789a515e32d6d44618fecc77efd896b1055ce467568e06a79eab9df3bae6 |
| step950000_model.pt | 5,939,703,038 | 6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b |
| step950000_optim.pt | 11,879,424,306 | ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2 |
| step1530000_model.pt | 5,939,703,038 | d55f32683d56d1e61d0b919d265a611bebec864875db404bcb0642b0dd419120 |
| step1530000_optim.pt | 11,879,424,306 | 58f7f01d73a1e7772568a70234759ef82fa3d65367b75a813dc5b60e17158884 |

- 下载顺序：380000（G1c 失败 checkpoint，优先）→ 950000 → 1530000；断点续传 + 重试循环（网络当前中断，历史成功率：legacy 循环重试后 ~2.7MB/s 可用）；不下 tokenizer（legacy 已有）、不下任何训练语料。
- 磁盘预算：3×17.8G=53.5G ≤ 当前可用 246G；q_stream 用冻结种子确定性再生成（§6），不下载。
- SHA256 不匹配 → 删除重下；仍不匹配 → 停止并报告，不进入分析。

## 3. 冻结的模块分组（stage 3/4/6 全部共用）

对 OLMo-2 融合权重按行切片（row-slice 于输出维）：

| 组 | 规则（冻结） |
|---|---|
| embed | `transformer.wte.weight` 全部 |
| head | `transformer.ff_out.weight`（untied lm_head）全部 |
| attn_q | `blocks.*.att_proj.weight` 行 [0:2048) |
| attn_k | 行 [2048:4096) |
| attn_v | 行 [4096:6144) |
| attn_o | `blocks.*.attn_out.weight` 全部 |
| mlp_gate | `blocks.*.ff_proj.weight` 行 [0:8192) |
| mlp_up | 行 [8192:16384) |
| mlp_down | `blocks.*.ff_out.weight`（block 级，[2048,8192]）全部 |
| qknorm | `blocks.*.{q_norm,k_norm}.weight` |
| norm | `blocks.*.{attn_norm,ff_norm}.weight` + `transformer.ln_f.weight` |
| others | 残余（预期为空；非空则在报告中列出） |

## 4. 冻结的 stage 3 指标（optimizer-state audit）

对每个 checkpoint（3 个）× 每组（12 组）：

- v=exp_avg_sq（fp32 存储原值，统计在 fp64）：exact-zero 数/占比、min、p0.001、p0.01、p0.1、p5、median、p99、p99.9、max；
- `√v < {eps, 10eps, 100eps, 1000eps}` 坐标占比（eps=1e-8；与 F0 完全同阈值）；
- m=exp_avg 绝值分位（abs_min … abs_max 同 F0 集合）；
- 精度往返（round-to-zero 归因，冻结规则）：
  - `math_zero`：fp32 存储 v==0（数学零）；
  - `bf16_rt0`：v>0 但 `v.to(bfloat16)==0`（bf16 指数域宽，预期≈0）；
  - `fp16_rt0`：v>0 但 `v.to(float16)==0`（fp16 subnormal 下限 ~6e-8，预期大量 floor 坐标归零——量化"若以 fp16 存储会伪造精确零"）；
  - m 同样三分类；
- 对角增益 G（§0 公式，fp64）：每组 min/p0.001/median/p99.9/max；
- 逐 checkpoint 的 lr、存储 step、组覆盖率 sanity。

输出：`outputs/f1_1b_replication/state_floor_by_module.csv`（3×12 行 + 精度列）。

## 5. 冻结的 stage 4 方向类与精度臂（1-step conditioning，c=0 锚点）

方向（全部单位 L2，fp64 构造，种子冻结 20260926；空类记 NOT_APPLICABLE）：

1. `random_seed_11`、`random_seed_22`：全支持 iid N(0,1)；
2. `grad_aligned_mhat`：单位化 exp_avg；
3. `toprisk_q`：全部层 attn_q 切片内 denominator `√(v̂₁)+eps` 最小的 0.1% 坐标上 iid N(0,1)（规则冻结：按 denominator 升序取前 ⌊0.001·n_q⌋ 个坐标；这是 F0 "最危险 direction family = attn_q 低分母坐标" 的 1B 对应物）；
4. `mlp_only`：全部 mlp_gate/up/down 支持；
5. `embed_tail`：wte 行均值(mean over row of v) 最小的 1024 行；
6. `embed_head_rows`：行均值最大的 1024 行；
7. `embed_zero_v_rows`：含 ≥1 精确零 v 坐标的 wte 行全体（OLMo 特有类）。

每个方向 × 每 checkpoint × 精度臂 {fp64, fp32, bf16}（线性化算术精度；forward AdamW 规则不变）：
- `A(q)=‖J₀q‖/‖q‖` 总量 + §3 分组分量；
- σ_max(J₀)：fp64 幂迭代（对角映射，8 次迭代或 1e-12 收敛）+ 主导坐标定位（参数/组/wte 行号）；
- clip 调整副指标（冻结公式，非事后）：`A_clip(q) = coef_meas · A(q)`，coef_meas 取 stage 6 primal 该 checkpoint 第 1 步实测 clip 系数（1B 深_clip 的结构性缩放，与 F0 的 0.7–1.0 情形区分）。

输出：`outputs/f1_1b_replication/local_conditioning.csv`。

## 6. 冻结的 stage 5 FD 验证（单步，先于一切 multi-step）

锚点 A0（c=0，与 stage 4 对角精确对应）+ 锚点 A1（c=m̂，激活交叉项），每 checkpoint：

- 切片（冻结）：(a) wte 精确零 v 行（≤64 行）；(b) wte 非零低 v 行（denominator 最小的 64 个非零坐标所在行）；(c) attn_q 低 denominator 64 坐标；(d) mlp_gate 随机 4096 坐标；(e) 全模型 random_seed_11（fp64 manual math only）。
- ε 网格（冻结）：fp64 {1e-2 … 1e-12}（12 点）；fp32 native AdamW {1e-2 … 1e-6}（6 点，F0 同款下限）。
- 对照：vendored `g1c_adamw_tangent_step` 解析 JVP（fp64）。
- **G1 gate（冻结）**：
  - 非零 v 切片 (b)(c)(d)(e)：∃ε 使 rel_err ≤ 1e-6 且 cos ≥ 0.999999 → PASS；
  - 零 v 切片 (a)：rel_err(ε) 在 ε∈{1e-8→1e-12} 单调下降且 min rel_err ≤ 5e-2 → PASS（折点地板 a·ε/eps，§0 推导）；
  - fp32：∃ε≥1e-5 使 rel_err ≤ 1e-3 → PASS。
  - 全部 fp64 切片 PASS → G1 PASS；任一 fp64 切片 FAIL → **停止一切 multi-step**，转实现排查（分类 E 候选）。

输出：`outputs/f1_1b_replication/fd_validation.csv`。

## 7. 冻结的 stage 6 multi-step tangent growth

批流/RNG（冻结）：legacy `streams/stream1_tokens.bin`（262,144 个 int32 token，只读）；SEQ=512、8 seq/mb、4 mb/step（legacy 注册语义）。H≤16 用原始流；H=32 用**循环扩展**（周期 262,144 token，规则先于运行冻结，所有 arm 共用同一扩展 → 跨 arm 比较精确成立）。无 dropout/无随机增强；GPU bf16 autocast 前向（cuda:1；GPU0 被外部占用）。

冲击方向（冻结）：q_stream1 按 legacy 冻结配方再生成（base_seed 20260917 → stream_seed 8779495656483434628 → q_seed 4655128343037472768；iid N(0,1) 全支持、单位总 L2；张量级 SHA256 必须等于 legacy 存档 `08128dd4f86429399d9b732778d18e81b2a35f76ef978d71a25c3b883ca2cbb0`，否则停止）。冲击方式：h=1 的 pre-clip 切线梯度 dc=q，随后 B 分支 dc=0 / C 分支 dc=HVP(θ_h; dθ_h)。

臂（冻结）：

| arm | 分支 | 切线累积精度 | HVP | H | checkpoints |
|---|---|---|---|---|---|
| B_fp64 | B（optimizer-only） | fp64 | — | 32 | 380000, 950000, 1530000 |
| B_fp32 | B | fp32 | — | 32 | 同上 |
| B_bf16 | B | bf16 | — | 32 | 同上 |
| C_registered | C | fp32 | bf16 model + fp16 dθ（legacy 注册路径） | 16 | 380000 |
| C_fp64 | C | fp64 | fp32 model + fp32 dθ | 16 | 380000（950000 资源允许标记位） |

primal：所有 arm 共享同一次 native pass（bf16-autocast GPU 梯度 → fp32 CPU AdamW + clip 1.0），逐 arm 切线更新在每步梯度算出后立即执行（c_h 不落盘）。

每步记录（冻结列）：h、arm、branch、‖dθ‖、‖dm‖、‖dv‖、total、§3 分组 ‖dθ‖ 分量、max|dθ| 坐标（参数/组/局部 v/局部 denominator/局部 m）、clip_coef、clip_active、preclip_grad_norm、loss、HVP 范数（B 为 0）、JVP 出范数、dtype、NaN/Inf flag。

**停止规则（冻结）**：任一 arm 切线张量出现非有限值或 ‖dθ‖>1e30 → 停该 arm（记录首现 h，其余 arm 继续）；primal 非有限 → 停整个 checkpoint。事后不调阈值、不重启。

输出：`outputs/f1_1b_replication/multistep_growth.csv`、`outputs/f1_1b_replication/failure_trace.csv`（含 legacy cdiag 冻结参考列：C h=7 NaN、fp32-HVP ‖tangent‖→2.97e11@h8、B 线性 +4.9/step）。

## 8. 冻结的 gate（全部先于结果）

- **G1（FD）**：§6。FAIL → 停止，分类 E 候选。
- **G2（scale trend，主比较点 step380000）**：
  - `CONTINUES`：σ_max(1B,380000) > 2198 且 A_random > 1；
  - `STRONGER`：σ_max > 21,980（10×100M）或 A_random > 44.4（3×100M）；
  - `REVERSED`：σ_max < 290（跌破 26M）；
  - `FLAT`：其余。
  - 副指标 A_clip 同向报告（深 clip 结构性缩放，见 §5）。
- **G3（primal 正常）**：3 checkpoint 全部：每步 loss/grad_norm/clip_coef 有限，clip_coef∈(0,1]，且 32 步内 |loss_H − loss_1| < 2.0 nats。
- **G4（tangent 失稳）**：C 分支超线性（ρ_h=‖dθ_h‖/‖dθ_{h−1}‖ 连续 ≥3 步非下降且 ‖dθ_H‖/‖dθ_1‖ ≥ 100）或任一 arm 在 h≤16 触发 §7 停止规则，同时 primal 满足 G3。
- **G5（机制定位）**：报告 ‖dθ_H‖ 的分组份额（>50% → 模块定位成功）、fp64 vs frozen arm 的发散步差（precision 定位）、clip 系数对增益的缩放比（clipping 定位）、stage 3 floor 结构（second-moment floor 定位）。

**F1 进入 F2 的 PASS（= mandate 六条件）**：G1 PASS ∧ G3 PASS ∧ G2∈{CONTINUES,STRONGER} ∧ G4 成立 ∧ G5 至少一类定位 ∧ 非 E 分类。

## 9. 冻结的 G1c failure 分类程序（stage 7）

按序判定，首个命中即终止：

1. G1 FAIL → **E IMPLEMENTATION_FAILURE**；
2. C_fp64（380000）触发停止规则或 ‖dθ_16‖/‖dθ_1‖ ≥ 100 → **A STRUCTURAL_CONDITIONING**；若 C_registered/C_fp32 显著更早失败（首 NaN 步差 ≥2）则加注 B-qualification（precision 放大为次要）；
3. 否则若 C_fp32/C_registered 触发停止而 C_fp64 ‖dθ_16‖/‖dθ_1‖ < 100 → **B PRECISION_AMPLIFIED_CONDITIONING**；
4. 否则若仅 bf16/fp16-HVP 路径失败、fp32-HVP 各臂干净 → **C PRECISION_ONLY**；
5. 否则若 G2∈{REVERSED,FLAT} 且 stage 3/4 的 floor/模块结构与 MiniMind 阶梯机制不一致 → **D CHECKPOINT_SPECIFIC**；
6. 其余 → **F UNRESOLVED**（必须列明缺失证据）。

已冻结的先验（F0 §7）：主候选 SCALE_TREND、次选 PRECISION_ARTIFACT、E 基本排除——本程序允许推翻，判定只看本预注册下的新测量。

## 10. 冻结的产物与路径

- `docs/F1_1B_REPLICATION_PREREG.md`（本文）
- `outputs/f1_1b_replication/summary.json`、`state_floor_by_module.csv`、`local_conditioning.csv`、`fd_validation.csv`、`multistep_growth.csv`、`failure_trace.csv`
- 原始重量级中间物：`external/peteish1/`（checkpoint 二进制 + SHA256SUMS）、`outputs/f1_1b_replication/*.log`
- 代码：`src/opsens/f1_1b_replication_v01.py` + `scripts/f1_*.py` + 单测；vendor 三个 legacy G1C 模块（architecture/checkpoint/tangent，逐字节复制 + SHA256 登记，legacy 原件不动）。

## 11. 环境与资源（冻结）

- conda `fr_minimind`（torch 2.11.0+CUDA13）；`CUDA_VISIBLE_DEVICES=1`（GPU0 被外部作业占用 12.4G/93%）；RAM 125G（预算：model 5.9G + primal m/v 11.9G + fp64 切线 35.6G + fp32 切线 17.8G + bf16 切线 8.9G ≈ 80G 峰值）；
- 参考 wall-time（legacy 实测）：HVP ~646s/步、梯度 pass ~1–2 min/步、CPU fp64 切线 ~30–60s/步；预计 GPU 总时长 ~13h，下载 ~5.5h（网络中断时挂重试循环）。
