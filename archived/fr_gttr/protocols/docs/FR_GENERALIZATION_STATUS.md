# FR 机制泛化阶段总状态（FR_GENERALIZATION_STATUS v1）

> 建立：2026-09-16（G1a 结果冻结后的文档轮）。本文档只做状态登记与路线规划，
> **不含任何进行中的实验**；本节所有"下一步"均为计划，启动任何一项须用户
> 另行授权。所有数字均可溯源到列出的冻结制品（SHA256 见各目录
> SHA256SUMS / 审计包）；写作时已从制品本身重新核读，非转抄聊天记录。
> 恢复上下文优先读本文件 + `docs/FR_GENERALIZATION_HANDOFF.md` +
> `docs/FR_PROJECT_STATUS.md` §24/§24.1。

## 1. 科学目的（泛化阶段定位）

Paper 1 的核心命题——**真实 AdamW 轨迹上，含 optimizer state、未来曲率
（HVP）与 clipping JVP 的完整状态切线 C 是参数扰动因果响应的高精度预测器，
且其优势来自可分解的机制阶梯**——已在 5M anchor 上建立（A0 系列）。

**泛化阶段（G1/G2/G3 线）的任务是检验该机制的成立边界**：模型规模
（G1a 26M 已完成 → G1b ~100M → G1c ~1B）、数据分布（G2：26M 上 D0 vs D1）、
clipping 工况（G3：clip=1.0 vs no-clip 干预）。**不是**回到 anchor 加 seed
或加状态——anchor 侧已由 A0F2 56 态普查封闭（见 §2）。

## 2. Anchor 5M 证据（全部冻结，禁重跑）

- **对象**：v07 confirmatory bank，8 seeds（2030–2037）× 7 stages（steps
  46–82）= 56 态，MiniMind 4,983,168 unique params，AdamW（0.9/0.999，
  eps 1e-8，wd 0.01）+ clip 1.0 + bf16，replay 逐位精确。
- **机制三段的 anchor 证据**：
  - **optimizer state（B vs A）**：A0C/A0F2——B（optimizer-only 切线）系统性
    优于 A（scalar/momentum 代理），A mre 4.36 vs B 0.128（A0F2 全普查）。
  - **curvature + clipping（B → BH_fixed → C）**：A0E2——H=1 恒等 gap=0
    （24/24 流）；H>1 pooled：B 0.205255 → BH_fixed 0.136628（raw curvature
    正贡献 +0.068627）→ C 0.0218656（clip-JVP +0.114762）；gain 加总恒等
    max dev 8.9e-16。
  - **跨状态稳健性**：A0F2 全 56 态普查（224 行）——C pooled R²=**0.99781**、
    corr 0.99900、MAE 7.86e-4、mre 0.005872；C beat B 168/224（56 平 = H=1
    构造零对照，0 负），mean +0.121789，7/7 stage 正（最弱 +0.080713）；
    A0F 随机 32 态为其严格子集且共享行逐位复现。
  - **任务相关性**：A0G——H=4 上 C 的响应排序与真实训练决策效用相关
    （Level-1 Pearson 0.905；Level-3 mean Gain C 1.54e-4 > B 1.31e-4；
    H=8 C vs B bootstrap P=0.995）；墙钟成本优势不成立（shared-tangent
    speedup <1），A0H 线进一步给出伴随降成本的有限正点估计。
- **anchor 侧 caveat（泛化实验原样继承）**：mixed-backend HVP（math-SDPA
  reverse-over-reverse 代理）；chi_FD 为注册有限 ε 中央差分、C 为零脉冲
  局部光滑切线（A0D 残差主要为有限 ε 伪影）。

## 3. G1a 26M 规模泛化（COMPLETE / STRONG_SCALE_REPLICATION，冻结）

制品：`outputs/generalization_g1a_26m_v01/`（审计包 254KB，SHA256
`7b36669c0e1b323bc44a9f52ad0355c1e85cdd22003d3b811be53378323f92b0`；
10 个关键文件 SHA256 与 SHA256SUMS Section 1 全部 MATCH，归档时重算核实）。

- **scale-only**：4,983,168 → 26,681,728 unique params（**5.354×**；
  hidden 256→512、layers 4→7，其余一切与 v07 anchor 相同）；预注册估值
  26,682,048 在开任何结果前按实测更正（bookkeeping，模型尺寸不变）。
- **设计**：seeds 2042/2043/2044；EARLY=26（20.31%）/MIDDLE=64（50%）/
  LATE=102（79.69%），训练在 102 终止；9 态 × 4 future streams = 36 条
  baseline 流，36 份 manifest 全部先于任何 E2 分支写入；H={1,4,8}
  （27 正式单元）；机制子实验 seed 2042；smoke gate（2042/64）14/14；
  replay 36/36 逐位精确；rho0-noop 36/36。
- **主结果（27 正式单元，pooled）**：
  - C：R²=**0.99958**、corr=0.99985、MAE=0.001055、mre 0.00792
    （median 0.00232）。
  - B：R²=0.83204、corr=0.97300、MAE=0.021645、mre 0.12033。
  - H>1 paired（18 行）：**C 18 胜 / 0 平 / 0 负**，mean +0.16861、
    median +0.09549；per-seed +0.0732 / +0.1112 / +0.3214（3/3 正）；
    per-stage EARLY +0.1493 / MIDDLE +0.2966 / LATE +0.0599（3/3 正）。
  - H=8：C mre 0.00799、R²=0.99931 vs B mre 0.23664、R²=0.63582。
- **机制子实验（seed 2042，B → BH_fixed → C）**：H=1 恒等 max gap=0；
  pooled（3 stage × H>1）gain_curvature=**+0.05738**、
  gain_clip=**+0.01472**；6/6 单元两项 gain 均正（未预设逐单元为正）。
- **判定**：**STRONG_SCALE_REPLICATION**（冻结四级条件全过：C pooled
  R²≥0.90；H>1 胜率 ≥80%；H=8 paired + R²≥0.50 + corr>0；≥2/3 seed 正）。
- **解释边界（冻结，引用时必须保留）**：见
  `docs/FR_PROJECT_STATUS.md` §24.1——仅证明 26M；mixed-backend HVP caveat
  原样；chi_FD 注册有限 ε、C 零脉冲局部切线；clip-gain 缩小是观察非定律；
  18/18 非未来规模理论要求。

## 3b. G1b ~100M 规模泛化（COMPLETE / STRONG_SCALE_REPLICATION，冻结）

制品：`outputs/generalization_g1b_100m_v01/`（审计包 250 members，SHA256
`6fe9b0f99aac2f4c759cd2f07d606931b89a631fa65faa3720e32ade9f7a5f85`；
bank/ 9 checkpoint 与 theta q 大文件仅留哈希不入包）。

- **scale-only 第二次复制**：4,983,168 → 26,681,728（G1a）→
  **100,784,000** unique params（**20.225×** anchor、3.777× G1a）；
  hidden=768、layers=13、heads=12、kv=6、head_dim=64、
  intermediate=2432=ceil(768·π/64)·64（冻结默认公式，非手挑）；静态 CPU
  计数先行、结果前冻结、未尝试其它尺寸；state_dict elements=105,699,200。
- **与 G1a 完全一致**：dataset D0 / tokenizer / vocab 6400 / seq 128 /
  objective / AdamW(0.9/0.999/1e-8/0.01) / clip 1.0 / BF16 / CRN / q 语义 /
  ε=0.25σ_q 语义 / post-unscale/pre-clip pulse / FD/B/BH_fixed/C 定义 /
  mixed-backend HVP caveat；train seeds 2042/2043/2044（paired-scale
  design）；EARLY=26/MIDDLE=64/LATE=102；128 planned steps、1024
  tokens/step；训练在 102 终止。
- **主结果（27 正式单元，pooled）**：
  - C：R²=**0.99949**、corr=0.99994、MAE=0.001468、mre 0.008760
    （median 0.005079）。
  - B：R²=0.92725、corr=0.99467、MAE=0.015859、mre 0.078779。
  - H>1 paired（18 行）：**C 18 胜 / 0 平 / 0 负**，mean +0.10503、
    median +0.08037；per-seed +0.10932 / +0.09568 / +0.11008（3/3 正）；
    per-stage EARLY +0.14787 / MIDDLE +0.07972 / LATE +0.08749（3/3 正）。
  - H=8：C mre 0.008729、R²=0.99917 vs B mre 0.165960、R²=0.81076。
- **机制子实验（seed 2042，B → BH_fixed → C）**：H=1 恒等 max gap=0
  （精确）；pooled（3 stage × H>1）gain_curvature=**+0.07325**、
  gain_clip=**+0.03618**（带符号、不预设）。
- **判定**：**STRONG_SCALE_REPLICATION**（冻结四级条件全过：C pooled
  R² 0.99949≥0.90；H>1 胜率 100%≥80%；H=8 paired + R² 0.99917 + corr>0；
  3/3 seed 正）。
- **clipping regime 三点并列**（G1a 登记的观察在第三点延续；**不拟合
  经验定律、原因未定**，仅作 G3 前置观察）：

  | 规模 | future 第一步 clip | future 全步 clip | training clip | preclip norm mean | gain_curvature | gain_clip |
  |---|---:|---:|---:|---:|---:|---:|
  | 5M anchor | 0.25 | n/a | n/a | 2.154 | +0.06863 | +0.11476 |
  | 26M G1a | 1.0 | 0.98958 | 1.0 | 1.368 | +0.05738 | +0.01472 |
  | 100M G1b | 1.0 | 1.0 | 1.0 | 1.510 | +0.07325 | +0.03618 |

- **三规模机制比较表**（5M 行为 A0F2 56 态普查口径 H={1,2,4,8}，26M/100M
  为 3 seed × 3 stage 口径 H={1,4,8}；为 scale trend 比较，**非严格
  paired test**）：

  | 规模 | C pooled R² | C mre | B mre | H8 C mre | H8 B mre | mean paired improvement |
  |---|---:|---:|---:|---:|---:|---:|
  | 5M anchor | 0.9978128053 | 0.005872113467 | 0.1276611331 | 0.003987692355 | 0.3033932649 | +0.1217890196 |
  | 26M G1a | 0.9995784934 | 0.007923837804 | 0.1203291912 | 0.007993832810 | 0.2366420377 | +0.1686080300 |
  | 100M G1b | 0.9994888337 | 0.008760484109 | 0.0787785519 | 0.008728685661 | 0.1659595431 | +0.1050271017 |

- **当前结论（冻结表述）**：**在相同 MiniMind 实现、数据、优化器和响应
  协议下，完整切线 C 在约5M→26.7M→100.8M 的约20倍参数规模范围内保持
  R²>0.997，且在当前所有测试的 H>1 state-horizon 单元中均优于 B。**
  同时登记：**B 的 absolute/relative error 随当前三个规模点有所降低**
  （B mre 0.1277→0.1203→0.0788；H8 B mre 0.3034→0.2366→0.1660），
  **但 C 的额外优势仍稳定存在**（C pooled R² 均 ≥0.9977、H>1 paired
  全胜、mean paired improvement 三点均大幅为正 +0.122/+0.169/+0.105）。
  curvature gain 三点：+0.0686 / +0.0574 / +0.0732；clip gain 三点：
  +0.1148 / +0.0147 / +0.0362（仅登记为观察，禁止拟合经验律或声称因果
  原因已确定）。

- **解释边界（冻结）**：仅证明 ~100M，不能替代 A0F2、不能自动外推 1B；
  mixed-backend HVP caveat 原样；chi_FD 注册有限 ε、C 零脉冲切线；18/18
  非未来规模理论要求；clip-regime 表为三规模观察非定律、原因未定。

## 3c. G3 clipping 干预（COMPLETE / MECHANISM_CONFIRMED，冻结）

**科学问题**：A0E2 识别的 clipping-Jacobian 机制是否有可干预的因果证据。
26M 固定配置（与 G1a 逐字段一致），clip=1.0 对照：chi_fd/chi_B/chi_C
三 seed 直读冻结 G1a `results/per_stream.json`、seed 2042 的 BH_fixed
直读冻结 G1a `mechanism/*.json`；seed 2043/2044 的 BH_fixed 不在任何
冻结 G1a 制品中，由本实验 clipref 用冻结 a0e2 runner 对只读 G1a bank
新算补全（**G3 clip-reference extension**：未重训 checkpoint、未改
冻结字节，2042 同路径重算 cross-check = 0.0；详见 G3_PROTOCOL.md §16），
no-clip 条件真正重训（训练循环从不调用 `clip_grad_norm_`，
逐步记录 `clip_grad_norm_called=false`；冻结响应原语经
CLIP_DISABLED_MAX_NORM=1e30 数学恒等路径）。paired identity：沿用 G1a
E1/E2 stream seed 生成规则，未来 batch/RNG 跨条件逐位一致。

**G3a operator control（冻结 G1a bank，seed 2042 × 3 stage × 4 streams，
H={1,4,8}）**：C_identity（未来步 clip-JVP 换 identity，其余不变）≡
BH_fixed，36/36 单元**位级一致**（max abs gap = 0.0 ≤ 1e-12）；未修改
a0e2 重跑 vs G1a 冻结 mechanism JSON cross-check max gap = 0.0。按 B3
定义，G3a 只归因 operator，不作因果 regime 证据。

**G3b no-clip 训练**：seeds 2042/2043/2044 × EARLY/MIDDLE/LATE = 9 态；
clip_grad_norm_ 调用计数 = 0（三 seed 全 0）；counterfactual would-clip
率（preclip>1）= 1.000/1.000/0.980——干预落在高信息区（clip 本近恒激活）。
preclip norm mean 1.354（G1a clip=1.0 为 1.391）。

**no-clip 正式矩阵（27 单元，9 态 × 4 streams × H={1,4,8}，FD/B/BH_fixed/C
全算）**：BH_fixed ≡ C **逐位崩溃**（27 单元 max abs gap = 0.0）——理论
退化成立。pooled：B R²=0.93821/mre 0.11162；BH_fixed=C R²=**0.99964** /
mre 0.006124（median 0.002261）——no-clip 下 C 对 FD 仍保持高精度预测
（不声称 C==FD，有限 ε 非线性仍在）。

**机制增益对照（pooled H>1，18 官方单元/条件）**：

| 条件 | gain_curvature | gain_clip |
|---|---:|---:|
| clip=1.0（G1a 冻结 + clipref 2043/2044） | +0.07463 | +0.09398 |
| no-clip（G3b） | +0.15824 | **+0.00000** |

**Interaction**：Delta_clip = +0.09398；层次 cluster bootstrap
（seed>stage>stream，20000 次）mean +0.08563、95% PI [+0.00863,
+0.29384]、P(Δ>0)=0.99985。分组：H4 +0.06741 / H8 +0.12055；seed 2042
+0.01586 / 2043 +0.01785 / 2044 +0.24823；stage EARLY +0.00047 / MIDDLE
+0.25669 / LATE +0.02478。**异质性为真实数据属性**（clip 条件下 seed
2044 MIDDLE 的 BH_fixed 误差 0.54–0.78 而 C 仅 0.02–0.04），登记为观察。

**判定（冻结阈值，结果前写入，未改）**：**MECHANISM_CONFIRMED**——
cond1 operator control ✓（gap 0.0 ≤ 1e-12）；cond2 no-clip pooled
|gain_clip| = 0 ≤ 0.2×clip ✓；cond3 no-clip BH/C 各自 R²≥0.5 且 mre≤0.5
且 max rel gap vs FD = 0 ≤ 1e-6 ✓；cond4 Δ>0 ✓。

**解释边界（冻结）**：① G3a 是 operator control 非因果 regime 证据，
因果声明 rests on G3b；② **curvature 增益在 no-clip 下不消失**（+0.158
vs clip +0.075，反而更大）——clipping 不是唯一反馈；③ no-clip 不意味
C==FD；④ per-seed/stage 异质性登记为观察，不拟合定律；⑤ 干预仅证明
26M clip→response 因果链，不外推其它规模/optimizer。

**工程**：smoke（2042/EARLY）16/16（含 no-clip BH≡C 理论退化位级检验、
H1 B=BH=C、H1 vs FD 8.8e-4）；replay 36/36 逐位精确；rho0-noop 36/36；
clipref cross-check（2042 重算 vs 冻结）= 0.0；无 OOM、无 retry、无
RESOURCE_LIMIT；GPU0 共享、未杀外部任务。

制品 `outputs/generalization_g3_clipping_intervention_v01/`（G3_PROTOCOL、
config、四 manifest、smoke_report、operator_control/、bank/、theta/、
clip_reference/、no_clip_regime/{per_stream,per_state_horizon,
mechanism_decomposition,streams/}、results/{summary.json,
clipping_interaction.csv}、REPORT、CHANGE_LOG、COMMANDS.log、
ENVIRONMENT.json、HASHES、SHA256SUMS）；审计包 252 members（bank/ 与
theta q 大文件仅留哈希），SHA256 `3c2c6fbf…ff3b`。代码
`research/src/fr/g3_clipping_intervention_v01.py` + runner + 单测 18/18。

## 3d. G2 数据分布泛化（COMPLETE / STRONG_DATA_REPLICATION，冻结）

**科学问题（B0）**：模型规模、架构、tokenizer、训练目标、AdamW、clipping、
token 预算、因果响应协议全部固定，仅预训练语料 D0→D1 时，完整切线 C
是否仍准确预测 FD 响应、并在 H>1 系统性优于 optimizer-only 切线 B。

**数据（冻结 provenance）**：D0 = 冻结 G1a 实际训练池
`outputs/e3_confirmatory/v07/e3c_subset.jsonl`（4096 条，SHA256
`1e657af9…b0f4`；只读直取，永不重跑）。D1 = MiniMind 官方
`pretrain_hq.jsonl`，HF `jingyaogong/minimind_dataset` 删除前官方
revision `3a9bbe611244811e45b96661a84047f112c8ed01`（2026-01-17 上传、
2026-03-26 自 HEAD 删除），1,413,103 条，SHA256 `82b19375…18b`（= LFS
oid）。D1 pool = 源语料全量排列（seed 60202601）前 4096 索引升序
（SHA256 `5d4c5e6b…a882`）；partition（seed 60202603）probe/eval/train
= 32/128/3936（SHA256 `63fed388…f050`）。schema 双侧均为 `{"text": str}`
→ 无 adapter；同一 tokenizer/管线/PretrainDataset（B4）。

**分布偏移核验（matched descriptive，B14）**：D1 全量 100% 截断在 126
tokens（D0 均值 114.9、73% 截断）、平均字符长 464 vs 266；13-gram
shingle Jaccard 均值 1.576e-7（max 0.003546，语料几乎不相交）；D0
精确重复行 0、D1 820 行（0.058%）。非零重叠/重复 ≠ 数据集被更换。

**主结果（27 行 pooled，口径 = 冻结 G1a `cell_relative_errors` /
`aggregate_metrics`，无第三口径）**：B R²=0.9114962992/corr
0.9918458572/MAE 0.0178247738/mre 0.0916795650；C R²=**0.9999452367** /
corr 0.9999876414/MAE 0.0004293364/mre **0.0025988312**。per-H（C）：
H1 R² 0.9998531159（B≡C 结构性平局）；H4 R² 0.9998802541、mre 0.0030133082
vs B mre 0.0758277891；H8 R² 0.9999297443、mre 0.0023441441 vs B mre
0.1967718646。H>1 paired **18/18/0**（win fraction 1.0），mean
improvement **+0.133621**（D0 G1a 为 +0.16861）；per-seed 均值
+0.2062701656/+0.0791880674/+0.1154050690（3/3 正）。

**机制子集（seed 2042，双口径均报告、永不混用）**：H=1 恒等 gap = 0.0；
stream-first gain_curvature **+0.1717813477** / gain_clip
**+0.0339615257**；official-cell +0.172470 / +0.033800。对照 D0 G1a
（stream-first）：curvature +0.0574 / clip +0.0147——D1 上曲率增益反而
更大，登记为观察。

**不确定性（B15）**：层次 cluster bootstrap（seed→stage→cell，20000
resamples，rng 20260916）：observed +0.133621、mean +0.133125、median
+0.121668、95% PI [**+0.067160, +0.251205**]、重采样中差值为正的
比例 = **1.0**。该比例**不是** p 值/后验/显著性概率；3 seeds ⇒ 探索性。

**判定（冻结阈值，结果前写入，未改）**：**STRONG_DATA_REPLICATION**——
cond1 C pooled R² 0.9999452367 ≥ 0.90 ✓；cond2 H>1 18/18 wins ≥ 15/18 ✓；
cond3 H=8 mean err_C 0.002344 < mean err_B 0.196772、R²_C_h8
0.9999297443 ≥ 0.50、corr > 0 ✓；cond4 3/3 seed 均值 > 0 ✓。

**Dynamics（仅诊断，从不用于过滤）**：训练 loss 3.6333(D0) vs 3.5575(D1)；
训练 preclip norm 1.3906 vs 1.2115；训练 clip 率 1.0 vs 0.9314；future
首步 clip 率 1.0 vs 0.8611；future grad norm 1.3684 vs 1.0934；**HVP
norm mean 0.2837 vs 1.9911（~7×）**；gain_curvature stream-first 0.0574
vs 0.1718；gain_clip 0.0147 vs 0.0340。全部登记为观察。

**解释边界（冻结）**：① D0 vs D1 为 matched descriptive comparison（按
seed/stage/H 身份配对），**不是**样本配对的因果数据集实验；② 机制
复现限于"当前 optimizer/clip 配置 + 大幅语料分布偏移"这一工况，不外推
其它 optimizer/目标函数；③ mixed-backend HVP caveat 原样；④ chi_FD
注册有限 ε、C 零脉冲切线（A0C 同 caveat）；⑤ 18/18 与 3 seeds 均为
探索性量级。

**工程**：smoke（B9：seed 2042/EARLY/step 26）18/18（H=1 vs FD
1.83e-3，含 train-loader deterministic replay 与 dataset/pool/partition/
state-metadata 哈希核对）；theta 36/36 E2 流、rho0-noop 36/36；tangents
36/36 baseline replay 位级精确；单测 31/31；无 OOM、无 retry、无
RESOURCE_LIMIT；GPU0 共享、未杀外部任务、未触碰 GPU1。执行期一次审计
重入缺陷（第二次 audit 把旧 bundle 自包含）修复并复跑，记 CHANGE_LOG §3。

制品 `outputs/generalization_g2_data_distribution_v01/`（G2_PROTOCOL、
DATASET_PROVENANCE、dataset_manifest、config、run/training/checkpoint/
future_stream 四 manifest、smoke_report、data/d1_pool.jsonl、
partition_g2_d1.json、results/{summary.json, per_stream.csv,
per_state_horizon.csv, mechanism_decomposition.csv,
d0_vs_d1_comparison.csv, bootstrap.json}、REPORT、CHANGE_LOG、
COMMANDS.log、ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES、SHA256SUMS）；
审计包 256 members（bank/、theta q、raw D1 语料大文件仅留哈希），SHA256
`9553ffa5…157a7`。代码
`research/src/fr/g2_data_distribution_generalization_v01.py` + runner +
单测 31/31。

## 3e. 第一篇泛化覆盖地图（2026-09-16，G2 冻结后登记）

**已完成轴（全部冻结）**：

| # | 泛化轴 | 覆盖 | 证据 | 状态 |
|---|---|---|---|---|
| 1 | within-configuration | 8 seeds × 7 stages = 56 态普查 | A0F2（v0.7 bank） | COMPLETE |
| 2 | scale（controlled ladder） | 5M → 26.7M → 100.8M（scale-only，单一变量） | G1a / G1b | STRONG_SCALE_REPLICATION ×2 |
| 3 | mechanism intervention | clipping on/off（真正 no-clip 重训干预） | G3 | MECHANISM_CONFIRMED |
| 4 | data distribution | D0 → D1（cross-corpus 分布大幅偏移） | G2 | STRONG_DATA_REPLICATION |
| 5 | task relevance | 响应的任务层决策效用 + 成本 | A0G | 效用成立 / 墙钟优势不成立 |

**剩余主要未覆盖**：external architecture；billion-scale checkpoint；
broader multi-corpus coverage；optimizer-family generalization。

**G1c External ~1B Stress Test（2026-09-16 执行）**：结果 =
**EXTERNAL_CHECKPOINT_UNAVAILABLE**（详见 §3f）——官方 AllenAI OLMo ~1B
系列无公开 optimizer state，科学实验在任何结果查看前停止；外部
architecture / billion-scale checkpoint 两轴**未覆盖**。

**禁止表述**：不得写"已证明任意数据集泛化"、"多数据集普适"、
"任何语言/域都成立"；不得把 5M/26M/100M/OLMo-1B 画成一条纯 scaling
curve 或拟合 error vs parameter count 定律（controlled ladder 与
external stress 必须分开展示）。

## 3f. G1c External ~1B Stress Test（BLOCKED → 复核修正：AVAILABILITY_RECHECK_PASSED，冻结）

**科学定位（B0）**：非 pure scale test——外部 1B 同时改变 architecture /
tokenizer / corpus / optimizer 超参 / training schedule；唯一允许的主
结论是"核心有限时间状态空间响应结构（FD/B/C）能否在一个公开的、约 1B、
具有真实 optimizer state 的外部 LLM 训练系统上继续成立"。

**B2 核验（任何科学结果查看之前）**：HF API 逐 revision 文件列表检查
官方 AllenAI OLMo ~1B 全系列 7 repo（OLMo-1B main + 9 抽样 revision /
336 branch；OLMo-1B-hf；OLMo-1B-0724-hf；OLMo-1B-0724-954000steps-
unsharded；OLMo-2-0425-1B main + 3 抽样 / 268 branch；OLMoE-1B-7B-0924
（MoE 架构排除）；OLMo-7B 抽样 1（7B 超资源范围））；olmo-checkpoints.org
= HTTP 404。**state 文件（optim*/train*/scheduler*/rng*）= 0**。OLMo-1B
原则上满足 B1 候选 5 理由（1.24B、autoregressive Transformer、AdamW
betas(0.9,0.95) wd0.1 lr4e-4、max_grad_norm=1.0、336 个公开 step
checkpoint），但不满足决定性要求（unsharded 含 model+optimizer+train
state）。按 B2 不得补造、同系列无可切换对象 →
**EXTERNAL_CHECKPOINT_UNAVAILABLE，停止科学实验**。这是制品可用性声明，
**不是**机制的科学负结果。

**执行事实**：大文件零下载；GPU 零运行（B3/B6/B10/B11/B14 未触发）；
实测硬件 2× RTX 5090 32GB（GPU0 共享余量 ~15.8GB），与 A100-40GB 假设
不符已登记。制品 `outputs/generalization_g1c_external_1b_v01/`（协议 /
provenance / 核验 manifest / 阻塞态 future_stream_manifest 与
smoke_report / results 五件套零数据行 / 报告 / CHANGE_LOG /
ENVIRONMENT / SHA256SUMS / screening_evidence）；审计包 SHA256
`c78dbbfb…854c9`（21 文件 + 自校验 + 读回验证）。

**影响**：受控 scale ladder、G3、G2、A0G 结论不受影响；外部 ~1B 轴
未覆盖。更换外部家族（须公开 optimizer state checkpoint）须用户新
授权与新预注册，不自动执行。


**2026-09-16 复核修正（FR_G1C_OLMO_DIRECT_CHECKPOINT_AVAILABILITY_RECHECK_V01）**：
按官方 docs/Checkpoints.md 修正核验逻辑（directory URL 404 ≠ checkpoint
不存在；旧 release 用 .pt 扩展名），从官方
`configs/official-0425/OLMo-2-0425-1B.csv` 按 step 冻结 4 个代表
checkpoint（step0/380000/950000/1530000），逐文件 HTTP 探测：**4/4 的
config.yaml + train.pt 全读 200；model.pt（5.94GB）+ optim.pt（11.88GB）
200 且前 1MB 实际读取验证**（PK zip magic、ETag 逐步不同、redirect=0、
无替代 host）→ 判定 **FULL_STANDARD_CHECKPOINT_AVAILABLE**。上一轮
EXTERNAL_CHECKPOINT_UNAVAILABLE 修正为 **AVAILABILITY_RECHECK_PASSED /
READY_FOR_1B_FEASIBILITY**（核验路径修正，非科学改判；HF screening
保留为历史记录——它只证明 HF 路径为 weights-oriented publishing）。
G1c 可以恢复继续；下一步 = **FR_G1C_1B_FEASIBILITY_GATE**（G1c 1B
feasibility gate），**须用户授权、不自动启动**。**定位说明**：G1c 属
**external architecture validation**（外部架构/tokenizer/语料/optimizer
联合改变，confounded by design），**不是 5M/26M/100M 的 scale-only
continuation**（scale-only 阶梯 = G1a/G1b）。科学实验未执行，禁止
"OLMo-1B response validated"表述。制品 `outputs/generalization_g1c_checkpoint_recheck_v01/`。

## 4. 当前科学解释（冻结结论 + 登记的观察）

**冻结结论（G2 增补，2026-09-16）**：完整状态切线机制在**大幅预训练
语料分布偏移**（D0→D1：100% vs 73% 截断率、shingle Jaccard ~1.6e-7、
HVP norm ~7×）下仍稳定成立——C pooled R² = 0.9999452367、mre
0.0025988312，H>1 paired 18/18/0，三 seed 均值全正，四级冻结条件全过，
判定 **STRONG_DATA_REPLICATION**。与 G1a（同配置、D0）对照：C 的预测
效度不降反略升（R² 0.99958→0.99994、mre 0.00792→0.00260），mean
paired improvement 同量级（+0.169→+0.134）。机制三段结构（optimizer
state → curvature → clipping）在 D1 上同样可分解且增益为正（curvature
+0.17178 / clip +0.03396）。

**冻结结论**：G1a 支持"**完整状态切线机制在 5.354× 参数规模扩展后仍稳定
成立**"——在 26M、clip=1.0、本协议下，C⊃B⊃FD 的预测排序、机制三段
（optimizer state → curvature → clipping）的可分解增益、H=1 局部恒等性
全部复现，且跨 3 seed × 3 stage 无反例。

**冻结结论（G3 增补，2026-09-16）**：clip-Jacobian 机制的因果性在 26M
干预下获支持——no-clip 训练使 gain_clip 从 +0.094 精确崩溃到 0.000
（BH_fixed≡C 位级一致），clip 与 no-clip 的 paired interaction
Delta_clip=+0.094（bootstrap P>0 = 0.9999）。curvature 段是独立机制
（no-clip 下 +0.158 不消失）。C⊃B 的因果链（G1a/G1b 只证相关性）在
clip 工况上升级为干预证据；curvature 工况的因果干预未做（不是 G3 问题）。

**登记的观察（重要，但只是观察）**：**clip 工况发生 regime shift**——
5M anchor future 第一步 clip 激活率 0.25（preclip norm mean 2.154）→
26M 训练期 clip 恒激活、future 第一步 36/36 全 clip（preclip norm mean
1.368）。尽管如此：(a) C 仍保持高精度（pooled mre 0.00792）；
(b) curvature 段增益存活（+0.05738 vs anchor +0.06863，量级相当）；
(c) clip 段增益方向一致但明显变小（+0.01472 vs anchor +0.11476）。

该三件套登记为**未来规模/clip 研究的重要观察**：它提示 clip-JVP 段的
相对贡献可能随规模与 clip 工况变化，**但成因未定**——不得把"clip 近恒
激活 ⇒ clip gain 缩小"写成已确认机制定律（26M 数据只有 3 seed × 3 stage，
且未做 clip 干预；该因果链是 G3 的动机，不是 G1a 的结论）。

**明确不写**："机制已普适到 LLM"（26M≠LLM 规模）；"100M 一定会成功"
（G1b 未做）；"clip gain 缩小原因已确定"（观察非定律）；"1B 已验证"
（G1c 未做）。

## 5. 剩余路线（顺序与门禁；仅计划，均未启动）

| 顺序 | 实验 | 问题 | 规模/配置要点 | 前置条件 |
|---|---|---|---|---|
| ~~1~~ **G1b** ✅ | ~100M 规模复制 | hidden=768/layers=13/heads=12/kv=6（100,784,000 unique，20.225×） | **COMPLETE / STRONG_SCALE_REPLICATION（2026-09-16 冻结）** |
| ~~1~~ **G3** ✅ | clipping 干预：clip=1.0 vs no-clip | 26M 固定；paired with G1a；真正 no-clip 重训 | **COMPLETE / MECHANISM_CONFIRMED（2026-09-16 冻结，§3c）** |
| 2 | **G2** | 26M 数据分布泛化 D0 vs D1 | 同 26M 模型/协议，换第二数据集 | **COMPLETE / STRONG_DATA_REPLICATION（2026-09-16 冻结，§3d）** |
| 3 | **G1c** | ~1B 外部压力测试 | OLMo-2-0425-1B standard checkpoint（外部架构/语料/optimizer，confounded by design） | **制品复核通过：AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY（2026-09-16 复核，§3f）——官方 CSV 4 代表 checkpoint 的 config.yaml+model.pt+optim.pt+train.pt 四件齐全；下一步唯一任务 = FR_G1C_1B_FEASIBILITY_GATE（须用户授权）** |
| 4 | **G4** | A0G-lite（26M 任务效用抽查） | 可选；低优先级 | 用户决定 |

**2026-09-16 更新（复核后）：G1c 制品复核通过——AVAILABILITY_RECHECK_PASSED / READY_FOR_1B_FEASIBILITY（§3f：官方 OLMo-2-0425-1B standard checkpoint 4 代表 checkpoint 的 model+optimizer+trainer+config 四件齐全可读；上一轮 HF-screening 阻塞判定修正为核验路径修正，非科学改判）；G1c 科学实验未执行（feasibility gate 须用户授权，不自动启动）；G2 已完成并冻结（§3d，STRONG_DATA_REPLICATION）；剩余 G4 未启动。此前：G3 已完成并冻结（§3c，MECHANISM_CONFIRMED）。**
