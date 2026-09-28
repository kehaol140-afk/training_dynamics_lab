# G1b — ~100M 模型规模泛化验证协议（FR_G1B_100M_SCALE_GENERALIZATION_V01）

预注册冻结时间：2026-09-16T15:35Z（任何正式结果打开之前；训练、theta、
smoke 全部尚未运行）。本文件与 `research/configs/g1b_100m_v01.json` 内容
一致；冲突时以本文件 + config 联合为准。判定阈值在本文件 §10 冻结，
不得根据结果修改。

## 0. 与已完成实验的关系（禁重做声明）

G1b 是第二次规模复制：4.98M（v07 anchor，A0C/A0E2/A0F2/A0G 已冻结）→
26.68M（G1a，STRONG_SCALE_REPLICATION 已冻结）→ **100,784,000**（本轮）。
不重做 A0A–A0G、A0F2、G1a、A0H/A″；不修改任何 5M/26M 冻结制品
（CSV/summary/checkpoint/q/verdict）。G1b 使用全新版本化目录
`outputs/generalization_g1b_100m_v01/`。

## 1. 科学问题

唯一主问题：当模型由 4.98M → 26.68M → ~100M 继续扩大，而 dataset、
tokenizer、training objective、AdamW、gradient clipping、response
definition、CRN protocol、B/C/FD definitions 保持一致时：

1. 完整切线 C 是否仍高精度预测真实有限差分响应；
2. C 是否仍在 H>1 系统性优于 optimizer-only B；
3. B → BH_fixed → C 机制阶梯是否仍保留；
4. 是否不发生随 horizon 增长的系统性崩塌。

本轮不是算法效率实验，不是 wall-clock benchmark。

## 2. Scale-only 配置（静态计数先行，结果前冻结）

静态实例化冻结 MiniMindConfig（CPU，未训练、未开任何响应结果）实测：

| 项 | anchor (v07) | G1a (26M) | **G1b (100M)** |
|---|---|---|---|
| hidden_size | 256 | 512 | **768** |
| num_hidden_layers | 4 | 7 | **13** |
| heads / kv (GQA) | 8 / 4 | 8 / 4 | **12 / 6**（head_dim=64、q:kv=2:1 保持） |
| intermediate_size | 832 | 1664 | **2432**（同一默认公式 ceil(h·π/64)·64，非手挑） |
| unique params | 4,983,168 | 26,681,728 | **100,784,000**（20.225× anchor；3.777× G1a） |
| state_dict elements | — | 29,958,528 | **100,784,000**（tied embeddings） |
| vocab / seq len | 6400 / 128 | 不变 | 不变 |
| RoPE / RMSNorm / SiLU | 冻结实现 | 不变 | 不变 |
| 代码世代 | minimind model_minimind.py + trainer_utils.py | 同一份冻结代码 | **同一份冻结代码，不改一行** |

候选结构 hidden=768/layers=13/heads=12/kv=6 在冻结实现下合法且实测落在
95M–110M 窗口 ⇒ 按 mandate **直接冻结，未尝试任何其它尺寸**；
模型尺寸在任何结果打开后不得回改。

## 3. 与 G1a 完全一致的项

dataset D0（e3c_subset.jsonl，sha256 1e657af9…，只读）；tokenizer/
vocab=6400；seq_len=128；RoPE/RMSNorm/SiLU；objective；LR schedule 语义
（5e-4 cosine，get_lr，256-micro 轴）；AdamW(betas .9/.999, eps 1e-8,
wd 0.01)；clip=1.0；BF16 autocast 语义（GradScaler 禁用）；CRN
future-stream protocol；q construction 语义（32 probe 单 batch，全局 L2
归一）；ε=0.25·σ_q（16 E1 流，std ddof=1）；post-unscale/pre-clip 单步
pulse；FD/B/BH_fixed/C 定义；mixed-backend HVP caveat（math-SDPA
reverse-over-reverse）；Model A 不进主比较。

## 4. 训练 schedule（与 G1a 相同，隔离规模变量）

- 复用 G1a 训练 seed：**2042、2043、2044**（paired-scale design；
  train_batch_seed = train_seed + 1704 同规则）。不得根据训练结果换 seed。
- planned_total_optimizer_steps = 128（256 micro）；global batch 8 seq =
  1024 tokens/step；总 token 预算与 G1a 完全相同。
- 状态（归一化进度，与 G1a 同）：**EARLY=26（20.3125%，26624 tokens）、
  MIDDLE=64（50%，65536 tokens）、LATE=102（79.6875%，104448 tokens）**；
  训练在 102 终止。
- 每步 metadata 记录：optimizer step、fraction、tokens seen、LR、
  training loss、preclip grad norm、clip status、param norm。
- 资源调整规则（仅当 OOM）：允许 device microbatch 下降（4→2）+
  accumulation 增加（2→4），**global batch / tokens-per-step 必须保持
  8 seq / 1024 tokens**；该调整须在重跑前写入 config 与 CHANGE_LOG。
  不得改 seq length 或 global batch。

## 5. 正式矩阵与 stream

3 seeds × 3 stages = **9 态**；每态 **4 条预注册 future stream** =
36 baseline 流；H={1,4,8} ⇒ **27 个正式 state-horizon 单元**。
stream seed 派生规则与 v07/G1a 完全相同（two_level_seed + salt 404），
使用 G1b 专属 base（E1=408202602，E2=409202602）以保证跨实验流身份
不冲突、规则可比。**顺序规则**：全部 36 条 manifest 先生成并写入
future_stream_manifest.json，才允许执行任何 E2 响应分支。

## 6. smoke gate（正式实验前；seed 2042 / EARLY step 26）

验证：checkpoint 完整恢复；optimizer m/v 存在且 shape 正确；q 与 named
params 严格对齐；CRN +/− replay 一致；rho0-noop 一致；H=1 analytic vs
FD 合理；B/C shadow baseline 一致；HVP 可执行；**GPU peak memory 记录**；
**math/native HVP gap 记录**；无意外 optimizer step；全部 finite。
实现 bug 可修复并重 smoke（CHANGE_LOG 记录）；科学定义问题 ⇒
CONTRACT_INCOMPATIBLE 停止。OOM 救济仅限 §4 资源调整；若 100M full
tangent 仍不可运行 ⇒ RESOURCE_LIMIT + 完整诊断，**禁止偷偷缩成 64M
冒充 G1b**。

## 7. 主科学比较与机制子实验

- 9 态全部：FD、B、C（定义完全继承 G1a）；输出 chi 与 absolute /
  relative / signed error。
- 机制子实验：第一个冻结 seed **2042** 的 EARLY/MIDDLE/LATE，B、
  BH_fixed、C、FD，H={1,4,8}；H=1 必须 B=BH_fixed=C（注册容差）；
  gain_curvature = err_B − err_BH_fixed、gain_clip = err_BH_fixed − err_C，
  带符号报告，不预设符号。

## 8. 预注册判定（冻结，结果前写入，不得根据结果更改）

分析网格 = 27 行（H=1 行进入 pooled R²/MAE，不计入 paired 胜负）。

**STRONG_SCALE_REPLICATION** 当且仅当同时满足：
1. C pooled R²（27 行）≥ 0.90；
2. 18 个 H>1 行中 ≥80%（≥15/18）err_C < err_B（tie：|err_C−err_B| ≤ 1e-12）；
3. H=8（9 行）：paired mean err_C < mean err_B，且 C 的 H=8 R² ≥ 0.50 且 corr > 0；
4. ≥2/3 个 seed 在其 H>1 行（每 seed 6 行）上 mean(err_B − err_C) > 0。

**PARTIAL_SCALE_REPLICATION**：H>1 pooled mean rel err 满足 C < B，
但 STRONG 未全满足（含 seed/stage 明显异质）。
**NO_SCALE_REPLICATION**：C 未总体优于 B，或 C 对 FD 失去有效预测能力。
**RESOURCE_LIMIT**：100M full tangent 在可用硬件上不可执行（允许救济
用尽并完整记录）。不要求复制 anchor 的 168/168。

## 9. 统计输出（不只一个 pooled 指标）

A pooled 27 行（R²、corr、MAE、mean+median rel err）；B 按 H；
C 按 seed；D 按 stage；E H>1 paired（wins/ties/losses、mean/median
paired improvement）；F 机制 gain；G **5M/26M/100M scale comparison**
（C pooled R²/MAE/mean+median rel err、B mean rel err、H8 B/C err、
mean paired improvement、clip activation rate、grad/preclip norm、
gain_curvature/gain_clip）。5M 取自 v07 A0C/A0E2 冻结记录与 v07 theta
mapping；26M 取自 G1a 冻结 summary/CSV；三者训练状态并非同一模型，
只称 **scale trend / scale comparison**，不称严格 paired test
（除非具体量确有 paired identity）。

## 10. Clipping regime 重点诊断

明确记录并三规模并列：training clip rate、future first-step clip rate、
**future all-step clip rate**、preclip norm mean/median/分布、
gain_curvature、gain_clip 随 scale 的变化。这是 G1a 登记的 regime-shift
观察（5M future 第一步 clip 0.25 → 26M 1.0）在第三点的延续。**禁止**
用三个 scale 点拟合经验定律；**禁止**声称 clip gain 变化原因已确定；
这些只是 G3 clipping intervention 的前置观察。

## 11. 资源规则

共享 GPU 可用（不杀外部任务）；记录 GPU 型号/显存/峰值显存/共享情况/
运行时间（仅工程记录）。长时间运行可 nohup/tmux/后台，须写完整 log 和
PID 记录；不得因聊天超时中止科学任务；不设无限资源等待。

## 12. 不做的事

G2 数据分布、G3 no-clip、1B、A0G-lite、A0H/A″、新 optimizer、
beta sweep、epsilon sweep、scalar FDR、重做历史实验、修改第一篇报告
正文、修改任何冻结制品。本轮只完成 ~100M scale replication。

## 13. 输出

`outputs/generalization_g1b_100m_v01/`：G1B_PROTOCOL.md、
experiment_config.json、training_manifest.json、checkpoint_manifest.json、
future_stream_manifest.json、smoke_report.json、results/summary.json、
results/per_stream.csv、results/per_state_horizon.csv、
results/mechanism_decomposition.csv、
results/scale_comparison_5m_26m_100m.csv、
G1B_100M_SCALE_GENERALIZATION_REPORT.md、CHANGE_LOG.md、COMMANDS.log、
ENVIRONMENT.json、SOURCE_AND_INPUT_HASHES.txt、SHA256SUMS.txt、
审计包 g1b_100m_audit_package.tar.gz（不含 checkpoint / 大权重 /
optimizer bank / 大型 q tensor；含 source/config/protocol/summary/CSV/
reports/manifests/hash 记录/tests）。
