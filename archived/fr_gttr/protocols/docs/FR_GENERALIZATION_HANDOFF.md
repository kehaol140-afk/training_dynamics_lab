# FR 泛化阶段交接（FR_GENERALIZATION_HANDOFF v1）

> 建立：2026-09-16，G1a 冻结后的纯文档交接；同日 G1b 完成后更新；
> G2 完成后更新；G1c 已执行并在制品核验阶段阻塞
> （G1c 首轮 BLOCKED / EXTERNAL_CHECKPOINT_UNAVAILABLE，§D.3；
> 2026-09-16 直链复核修正为 AVAILABILITY_RECHECK_PASSED，§D.4）。
> 恢复上下文读本文件 +
> `docs/FR_GENERALIZATION_STATUS.md` + `docs/FR_PROJECT_STATUS.md` §24/§24.1/§28/§30。
> 指标一律 import 冻结的 `fr.a0c_adamw_response_propagation_v01` 的
> `_metric`/`relative_error`，禁止重定义；冻结制品一律只读。

## A. 核心科学主线（一句话层级）

1. **scalar FR（A/momentum 代理）失效** → 2. **AdamW 增广状态切线（B：
   optimizer-only）系统性更好** → 3. **H1 局部 Jacobian 恒等（H=1 响应由
   局部 AdamW 映射完全解释）** → 4. **H>1 需要未来梯度 HVP（BH_fixed）+
   clipping JVP（C），机制阶梯 B→BH_fixed→C 增益可分解且加总恒等** →
5. **A0F2：5M 全 56 态普查稳健（C beat B 168/224，0 负，7/7 stage 正）** →
6. **A0G：响应的任务层决策效用成立（H=4 Pearson 0.905），墙钟成本优势
   不成立** → 7. **进入泛化阶段：机制在更大规模/其它分布/其它 clip
   工况下是否仍成立（G1/G2/G3）**。

## B. 已完成（冻结，禁重跑禁改判）

- **Anchor 5M**：v07 bank（8 seed × 7 stage = 56 态）；A0C/A0D/A0E2/A0F/
  A0F2/A0G（+ A0H 线：伴随数学成立、逐元 gate 被 bf16 HVP 地板阻塞、
  A0H-2.0 修正 pilot 正点估计）。
- **G1a 26M**：COMPLETE / **STRONG_SCALE_REPLICATION**（见 §C）。
- **G1b ~100M**：COMPLETE / **STRONG_SCALE_REPLICATION**（见 §C.1；
  2026-09-16 冻结）。
- **G3 26M clipping 干预**：COMPLETE / **MECHANISM_CONFIRMED**（见 §D.1；
  2026-09-16 冻结）。G3a operator control + G3b 真正 no-clip 重训，
  paired with G1a。
- **G2 26M 数据分布泛化**：COMPLETE / **STRONG_DATA_REPLICATION**（见
  §D.2；2026-09-16 冻结）。同 26M 模型/训练/协议，仅预训练语料
  D0→D1。
- **G1c ~1B 外部压力测试**：首轮 EXECUTED → **BLOCKED /
  EXTERNAL_CHECKPOINT_UNAVAILABLE**（§D.3；2026-09-16 冻结）；同日直链
  复核 **AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY**
  （§D.4；官方 OLMo-2-0425-1B standard checkpoint 四件齐全可读，上一轮
  判定修正为核验路径修正、非科学改判）。科学实验仍未执行；feasibility
  gate 须用户授权、不自动启动。

## C. G1a 关键冻结数字（溯源：`outputs/generalization_g1a_26m_v01/`）

- 规模：4,983,168 → **26,681,728** unique params（**5.354×**，scale-only）。
- 矩阵：3 seeds（2042/2043/2044）× 3 stages（26/64/102，归一化
  20.3/50/79.7%）= 9 态 × 4 streams = 36 baseline 流；H={1,4,8}，27 正式
  单元；机制子实验 seed 2042。
- pooled：C R²=**0.99958** / mre 0.00792；B R²=0.83204 / mre 0.12033。
- H>1 paired：**18/18/0**，mean +0.16861；per-seed 3/3 正；per-stage 3/3 正。
- H=8：C mre 0.00799 vs B 0.23664；C R²=0.99931。
- 机制：H=1 恒等 gap=0；gain_curvature **+0.05738**、gain_clip
  **+0.01472**（anchor +0.06863/+0.11476；6/6 单元均正）。
- clip regime shift（观察）：26M future 第一步 clip 36/36（anchor 0.25）；
  training clip 恒激活；preclip norm mean 1.368（anchor 2.154）。
- 完整性：smoke 14/14（H=1 vs FD 3.5e-4）；replay 36/36 逐位精确；
  rho0-noop 36/36；HVP math-vs-native gap 6.46e-3（mixed-backend caveat）。
- 审计包：`g1a_26m_audit_package.tar.gz` 254KB，SHA256
  `7b36669c0e1b323bc44a9f52ad0355c1e85cdd22003d3b811be53378323f92b0`。

## C.1 G1b 关键冻结数字（溯源：`outputs/generalization_g1b_100m_v01/`）

- 规模：26,681,728 → **100,784,000** unique params（**20.225×** anchor、
  3.777× G1a；hidden=768/layers=13/heads=12/kv=6/head_dim=64/
  intermediate=2432 冻结默认公式）；state_dict elements=105,699,200。
- 矩阵：同 G1a（3 seeds × 3 stages = 9 态 × 4 streams = 36 baseline 流；
  H={1,4,8}，27 正式单元；机制子实验 seed 2042）。
- pooled：C R²=**0.99949** / mre 0.008760（median 0.005079）；B R²=0.92725 /
  mre 0.078779。
- H>1 paired：**18/18/0**，mean +0.10503；per-seed 3/3 正；per-stage 3/3 正。
- H=8：C mre 0.008729 vs B 0.165960；C R²=0.99917。
- 机制：H=1 恒等 gap=0（精确）；gain_curvature **+0.07325**、gain_clip
  **+0.03618**（pooled H>1 均值）。
- clip regime 三点（观察非定律、原因未定）：

  | 规模 | future 第一步 clip | future 全步 clip | training clip | preclip norm mean | gain_curvature | gain_clip |
  |---|---:|---:|---:|---:|---:|---:|
  | 5M | 0.25 | n/a | n/a | 2.154 | +0.06863 | +0.11476 |
  | 26M | 1.0 | 0.98958 | 1.0 | 1.368 | +0.05738 | +0.01472 |
  | 100M | 1.0 | 1.0 | 1.0 | 1.510 | +0.07325 | +0.03618 |

- 完整性：smoke（2042/EARLY）13/13（H=1 vs FD 0.0183、GPU peak 11,032 MB、
  math/native HVP gap 3.06e-3）；replay 36/36 逐位精确；rho0-noop 36/36；
  无 OOM、无 retry、无 RESOURCE_LIMIT。
- 审计包：`g1b_100m_audit_package.tar.gz` 250 members，SHA256
  `6fe9b0f99aac2f4c759cd2f07d606931b89a631fa65faa3720e32ade9f7a5f85`。

## D. G1c（~1B 外部压力测试）：首轮 BLOCKED → 直链复核 PASSED（READY_FOR_1B_FEASIBILITY）

G2 已完成并冻结（§D.2）。**G1c External ~1B Stress Test 本轮已按逐项授权
执行，结果 = BLOCKED / EXTERNAL_CHECKPOINT_UNAVAILABLE**（§D.3 历史记录）；
**同日官方直链文件级复核修正为 AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY**（§D.4：官方 OLMo-2-0425-1B standard checkpoint 的 model+optimizer+trainer+config 四件齐全可读）。科学实验至今未执行。

**科学定位（B0，冻结表述）**：这不是 MiniMind scale-only replication，
也不是纯 scale 因果实验——外部 1B 同时改变架构、tokenizer、语料、
optimizer 超参与训练 schedule；唯一允许的主结论是"核心有限时间状态
空间响应结构（FD/B/C）能否在一个公开的、约 1B、具有真实 optimizer
state 的外部 LLM 训练系统上继续成立"。不得解释为 scaling law 或
MiniMind→1B 单因素外推。本轮结论是**制品可用性声明，不是机制的科学
负结果**。

**第一篇泛化覆盖地图（G2 冻结后，G1c 执行后更新）**：within-
configuration（A0F2，8×7=56 态）✅；scale ladder（G1a/G1b，
5M→26.7M→100.8M）✅；mechanism intervention（G3，clipping on/off）✅；
data distribution（G2，D0→D1）✅；task relevance（A0G）✅。**仍未覆盖**：
external architecture / billion-scale checkpoint（G1c 制品已核验可用、
科学实验尚未执行，轴保持开放直至 feasibility gate 后实测）/ broader
multi-corpus / optimizer-family。**GENERALIZATION_CORE_COMPLETE 不声明**
（B18 的触发条件不满足）。

**下一步候选（均不自动启动，须用户新授权）**：**FR_G1C_1B_FEASIBILITY_GATE（G1c 1B
feasibility gate，B3/B6 起；现为当前唯一下一步，前提已满足——官方
OLMo-2-0425-1B standard checkpoint 四件可用）**；Paper freeze；D2 数据集
（新目录版本化）。

## D.1 G3 关键冻结数字（溯源：`outputs/generalization_g3_clipping_intervention_v01/`）

- 矩阵：26M 固定（与 G1a 逐字段一致）；clip=1.0 对照 = 冻结 G1a
  直读（chi_fd/B/C 三 seed + 2042 BH_fixed）+ **G3 clip-reference
  extension**：2043/2044 的 BH_fixed 不在冻结 G1a 制品中，由 clipref
  用冻结 a0e2 runner 对只读 G1a bank 新算（未重训、未改冻结字节，
  cross-check 2042 = 0.0；G3_PROTOCOL.md §16）；no-clip = 3 seeds × 3 stages 真正重训
  （`clip_grad_norm_` 调用计数 0/0/0；counterfactual would-clip 率
  1.000/1.000/0.980）。paired stream identity 按构造逐位一致。
- G3a：C_identity ≡ BH_fixed 36/36 位级（max gap 0.0 ≤ 1e-12）；
  未修改 a0e2 重跑 vs 冻结 JSON cross-check = 0.0。**operator control，
  非因果 regime 证据**。
- no-clip 正式矩阵（27 单元）：BH_fixed ≡ C 逐位崩溃（max gap 0.0）；
  pooled B R²=0.93821/mre 0.11162；BH_fixed=C R²=**0.99964**/mre 0.006124
  （median 0.002261）。
- 机制增益（pooled H>1）：clip gain_curvature +0.07463 / gain_clip
  +0.09398；no-clip +0.15824 / **+0.00000**。
- Interaction：Delta_clip=+0.09398；bootstrap（20000）mean +0.08563、
  95% PI [+0.00863, +0.29384]、P(>0)=0.99985；per-seed +0.01586/
  +0.01785/+0.24823；per-stage EARLY +0.00047/MIDDLE +0.25669/LATE
  +0.02478（异质性为真实数据属性，登记为观察）。
- 判定：**MECHANISM_CONFIRMED**（冻结四级条件全过；阈值未改）。
- 边界：curvature 增益 no-clip 下不消失（+0.158）——clipping 非唯一
  反馈；no-clip 不意味 C==FD；干预仅证 26M clip→response 因果链。
- 完整性：smoke 16/16（含理论退化位级检验）；replay 36/36 逐位精确；
  rho0-noop 36/36；无 OOM/retry/RESOURCE_LIMIT。单测 18/18。审计包
  SHA256 `3c2c6fbf…ff3b`。

## D.2 G2 关键冻结数字（溯源：`outputs/generalization_g2_data_distribution_v01/`）

- 问题（B0）：除预训练语料 D0→D1 外，一切与冻结 G1a 26M 逐字段一致
  （模型 26,681,728 unique、AdamW betas/eps/wd、clip=1.0、bf16、batch 4
  ×accum 2、128 planned steps、seeds 2042/2043/2044、states 26/64/102、
  H={1,4,8}、CRN、ρ=0.25、E1/E2 stream seed 规则、ε=0.25σ_q）；主统计
  口径严格 import 冻结 G1a `cell_relative_errors`/`aggregate_metrics`，
  无第三口径；单测比对 G1a config 科学字段逐位一致。
- 数据：D0 = 冻结 G1a 实际训练池 e3c_subset.jsonl（4096 条，SHA256
  `1e657af9…b0f4`，只读直取）；D1 = MiniMind 官方 pretrain_hq.jsonl
  （HF `jingyaogong/minimind_dataset` 删除前 revision
  `3a9bbe611244811e45b96661a84047f112c8ed01`，1,413,103 条，SHA256
  `82b19375…18b` = LFS oid）；D1 pool（seed 60202601 前 4096 索引）
  SHA256 `5d4c5e6b…a882`；partition（seed 60202603，32/128/3936）
  SHA256 `63fed388…f050`。schema 双侧 `{"text": str}` → 无 adapter。
- 矩阵：9 态 × 4 preregistered streams = 36 baseline 流；27 正式单元；
  机制子实验 seed 2042（BH_fixed，双口径）。
- pooled（27 行）：C R²=**0.9999452367** / corr 0.9999876414 / MAE
  0.0004293364 / mre **0.0025988312**；B R²=0.9114962992 / mre
  0.0916795650。
- per-H（C）：H1 R² 0.9998531159（B≡C 结构性平局）；H4 R² 0.9998802541、
  mre 0.0030133082 vs B mre 0.0758277891；H8 R² 0.9999297443、mre
  0.0023441441 vs B mre 0.1967718646。
- H>1 paired：**18/18/0**（win fraction 1.0），mean improvement
  **+0.133621**（D0 G1a 为 +0.16861）；per-seed 均值 +0.2062701656 /
  +0.0791880674 / +0.1154050690（3/3 正）。
- 机制（2042）：H=1 恒等 gap = 0.0；stream-first gain_curvature
  **+0.1717813477** / gain_clip **+0.0339615257**；official-cell
  +0.172470 / +0.033800（双口径均报告、永不混用）。对照 D0（stream-first
  +0.0574/+0.0147）：D1 曲率增益更大，登记为观察。
- 分布偏移核验：D1 100% 截断 126 tokens（D0 均值 114.9、73%）；13-gram
  shingle Jaccard 均值 1.576e-7；D1 820 行精确重复（0.058%）。
- 不确定性：层次 cluster bootstrap（seed→stage→cell，20000 resamples，
  rng 20260916）：observed +0.133621、mean +0.133125、median +0.121668、
  95% PI [**+0.067160, +0.251205**]、重采样差值为正比例 = **1.0**
  （**非 p 值**；3 seeds ⇒ 探索性）。
- 判定：**STRONG_DATA_REPLICATION**（冻结四级条件全过：C pooled R²
  0.9999452367≥0.90；H>1 18/18≥15/18；H=8 paired + R² 0.99993 + corr>0；
  3/3 seed 正；阈值结果前写入、未改）。
- Dynamics（仅诊断）：训练 loss 3.6333(D0)→3.5575(D1)；训练 clip 率
  1.0→0.9314；future 首步 clip 1.0→0.8611；**HVP norm mean 0.2837→
  1.9911（~7×）**——观察非定律、从不用于过滤。
- 边界：D0 vs D1 为 matched descriptive comparison（身份配对），**非**
  样本配对因果数据集实验；机制复现限当前 optimizer/clip 配置；
  mixed-backend HVP caveat 原样；chi_FD 注册有限 ε、C 零脉冲切线。
- 完整性：smoke（2042/EARLY/step 26，B9 指定）18/18（H=1 vs FD
  1.83e-3，含 train-loader deterministic replay 与 dataset/pool/
  partition/state-metadata 哈希核对）；theta 36/36 E2 流、rho0-noop
  36/36；replay 36/36 逐位精确；单测 31/31；无 OOM/retry/RESOURCE_LIMIT。
  审计包 256 members（bank/、theta q、raw D1 语料仅留哈希），SHA256
  `9553ffa5a4783e273487dc7b3df7cc6b1143cedec4e13ab5bc3d7c88491157a7`。

## D.3 G1c 首轮关键冻结事实（BLOCKED 历史记录，保留不改；溯源：`outputs/generalization_g1c_external_1b_v01/`）

> 注：本节首轮 HF-screening 结论按 §D.4 复核修正解读——它证明 HF
> 路径为 weights-oriented publishing，不证明 OLMo standard checkpoint
> 无 training state。本节记录保留为历史过程。

- **B2 核验范围（2026-09-16，HF API 逐 revision 文件列表）**：官方
  AllenAI OLMo ~1B 全系列 7 repo——OLMo-1B main + 9 抽样 revision（共
  336 branch，step20000-tokens84B…step738020-tokens3095B）+ OLMo-1B-hf +
  OLMo-1B-0724-hf + OLMo-1B-0724-954000steps-unsharded + OLMo-2-0425-1B
  main + 3 抽样（268 branch）+ OLMoE-1B-7B-0924（MoE 架构排除）+
  OLMo-7B 抽样 1（7B 超资源范围）；olmo-checkpoints.org = HTTP 404；
  旧 v0.2.4 + 当前 README 均描述 weights-only 发布。**state 文件
  （optim*/train*/scheduler*/rng*）= 0**，14 个 revision 文件列表 +
  8-spread 抽样双重确认。
- **OLMo-1B 候选事实（B1 登记，供任何未来重试）**：Apache-2.0；
  ~1.24B params；d_model=2048、n_layers=16、n_heads=16、vocab=50280、
  rope、swiglu、weight_tying；AdamW lr=4e-4、wd=0.1、betas=(0.9,0.95)、
  cosine_with_warmup（t_warmup=10485760000、t_max=3e12、alpha_f=0.1）、
  max_grad_norm=1.0（warmup ×5.0）；amp_bf16、flash_attention、
  global_batch=512、save_interval=1000；原则上满足 B1 全部 5 条理由，
  但不满足决定性要求（unsharded 含 model+optimizer+train state）。
- **执行事实**：大文件零下载、GPU 零运行；B3/B6/B10/B11/B14 全部未
  触发；H1 数值 gate（B12）与正式矩阵无值可报（结果文件仅表头 +
  "# BLOCKED" 注释行）。
- **硬件差异登记**：实测 2× NVIDIA RTX 5090 32GB（GPU0 共享余量
  ~15.8GB；GPU1 外部任务从未触碰），与用户假设的单卡 A100-40GB 不符；
  本轮因阻塞而未成为约束，但任何重试的 B6 资源 gate 须按实测硬件
  重新评估。
- **制品与审计**：协议/预注册 + §18 阻塞记录（G1C_PROTOCOL.md）、
  EXTERNAL_MODEL_PROVENANCE.md、external_checkpoint_manifest.json、
  RESOURCE_FEASIBILITY.md、future_stream_manifest.json
  （NOT_CONSTRUCTED_BLOCKED；Dolma 为计划来源）、smoke_report.json
  （NOT_RUN_BLOCKED）、results/ 五件套（零数据行）、
  报告/CHANGE_LOG/COMMANDS/ENVIRONMENT/SOURCE_AND_INPUT_HASHES/
  SHA256SUMS（21 文件）/screening_evidence 四件。审计包
  `g1c_external_1b_audit_package.tar.gz`（32,184 B），SHA256
  `c78dbbfb7b32b2b5c94b8287728e9aa47e2db3abaeb823323f7fd6f1eb0854c9`
  （self-check + read-back PASS）。

## D.4 G1c 直链复核关键冻结事实（AVAILABILITY_RECHECK_PASSED；溯源：`outputs/generalization_g1c_checkpoint_recheck_v01/`）

- **任务**：FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01——只读
  HTTP 文件级复核；无 GPU、无科学 response、每大文件 ≤1 MB range read。
- **核验逻辑修正**：官方 docs/Checkpoints.md——unsharded checkpoint =
  config.yaml + model.safetensors + optim.safetensors + train.pt，旧
  release 用 .pt；directory URL 不能直接访问但目录内文件公开可访问。
  "目录 404 ≠ checkpoint 不存在"。
- **冻结的 4 代表 checkpoint**（官方
  `configs/official-0425/OLMo-2-0425-1B.csv`，stage1 run peteish1，
  step 0…1,907,359；按 step 选取，先于探测）：step0（EARLIEST）、
  step380000（19.92%）、step950000（49.81%）、step1530000（80.22%）。
- **结果 4/4 四件齐全**：config.yaml（163–164KB）+ train.pt（14–27KB）
  完整读取 200；safetensors 404（旧 release 情形）→ 官方回退
  model.pt（5,939,703,038 B）+ optim.pt（~11.88GB ≈ 2× model，
  AdamW m+v fp32 一致）全部 HTTP 200，前 1 MB 实际读取验证
  （PK\x03\x04 zip magic、ETag 逐步不同、redirect=0 无替代 host）。
- **判定**：**FULL_STANDARD_CHECKPOINT_AVAILABLE** → 上一轮
  EXTERNAL_CHECKPOINT_UNAVAILABLE 修正为 **AVAILABILITY_RECHECK_PASSED
  / READY_FOR_1B_FEASIBILITY**（核验路径修正，非科学改判）。
- **资源登记**：单 checkpoint 磁盘 ~17.8GB（fp32 .pt）；实测硬件
  2× RTX 5090 32GB 须重新探测余量；native context 4096（stage1
  YAML），B8 评估须以此为准。
- **已验证存在（4/4）**：OLMo standard unsharded checkpoint 的
  config.yaml + model.pt + optim.pt + train.pt 四件齐全。
- **定位说明**：G1c 属 **external architecture validation**（外部
  架构/tokenizer/语料/optimizer 联合改变，confounded by design），
  **不是 5M/26M/100M 的 scale-only continuation**（scale-only 阶梯 =
  G1a/G1b）。科学实验未执行，禁止 "OLMo-1B response validated" 表述。
- **下一步**：唯一任务 **FR_G1C_1B_FEASIBILITY_GATE**（G1c 1B
  feasibility gate，B3/B6 起），**须用户授权，不自动启动**。

## E. G3 禁改清单（继承 G1a/G1b 的 scale-only 原则）

1. 数据集 **D0**、tokenizer、seq_len=128、batch/accum 不变。
2. optimizer 族不变：AdamW，betas/eps/wd 不变；lr schedule 规则不变。
3. **grad clip = 1.0 不变**（clip 干预是 G3，不是 G1b）。
4. 因果响应协议不变：CRN 三分支、ρ=0.25、pulse 位置 post-unscale/
   pre-clip、第一步 pulse；rho0-noop 检查；smoke gate 先行。
5. 模型阶梯与指标定义不变：FD/B/BH_fixed/C（Model A 排除、BH_old 副
   产物排除统计）；`_metric`/`relative_error` 从冻结 a0c 模块 import。
6. H 官方网格 **{1,4,8}**；B/C/FD 名称与语义不变。
7. 状态网格沿用归一化进度规则（3 stage × 3 seed = 9 态 × 4 streams）。
8. 阈值在任何 G1b 结果打开前冻结并写入预注册协议（同 G1a 流程）。

## F. 顺序与依赖

**~~G1b（~100M 规模复制）~~ ✅ 已完成（2026-09-16 冻结）→ ~~G3
（clip=1.0 vs no-clip 干预）~~ ✅ 已完成（2026-09-16 冻结，
MECHANISM_CONFIRMED）→ ~~G2（26M 数据分布 D0 vs D1）~~ ✅ 已完成
（2026-09-16 冻结，STRONG_DATA_REPLICATION）→ **G1c（~1B 压力测试）：
首轮 BLOCKED / EXTERNAL_CHECKPOINT_UNAVAILABLE（§D.3）→ 同日直链复核修正
AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY（§D.4，官方
OLMo-2-0425-1B standard checkpoint 四件齐全）——当前下一步 = G1c 1B
feasibility gate，须用户授权，不自动启动** → G4（A0G-lite，可选）**。
 rationale：先把"规模"这个变量单独走完（5M→26M→100M 三点已闭环），
再用 G3 直接检验 clip regime shift 观察（已完成），然后换数据分布
（已完成），最后才做 1B 压力测试；不允许一次改两个变量。

## G. DO NOT REDO / 红线

1. **不重做 A0A–A0G**（含 A0H 已完成轮次）——全部冻结，禁重跑、禁改判。
2. **不在 26M 加 seed 或加状态**——G1a 设计就是 3×3，加样本是新实验。
3. **不因 100M 结果不理想而回改 26M 协议或阈值**——G1a 已冻结。
4. **永不把 G1a/G1b 称为 1B 证据**；26M 只证明 26M，100M 只证明 ~100M。
5. **永不把 A0H 伴随算法线与 Paper-1 机制泛化线混写**——两条线独立。
6. 禁止声称"机制已普适到 LLM"、"clip gain 变化原因已确定"、"1B 已验证"；G1b 完成后原"100M 一定成功"禁令转为"100M 结果不能外推 1B"。
7. 冻结制品（CSV/summary/checkpoint/runner/旧 verdict）一律只读；新实验
   开新目录；GPU1 外部任务永不可杀。
