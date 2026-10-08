# T2A_PROTOCOL_v1 — 单状态真实尺度校准协议（已执行授权：本轮）

日期：2026-09-29。授权来源：`references/ROUND2_OPENCODE_TASK_CN.md`（用户已批准）。
冻结配置：`configs/t2a_scale_v1.json`。本文件只描述本轮 T2a，不改完整 T2 草案。

## 1. 问题与边界

只回答两件事：(a) 该真实终态的自然梯度变化（跨 batch 的 H 波动）处于什么相对尺度；
(b) 在同一真实 H 上，FP64 / FP32（无 BF16）/ 原生 BF16 三条有限 NS 路径的偏差、
有限幅度响应与可分辨性如何。零模型更新、零验证集访问、零父轨迹。

## 2. 锚点与矩阵

- 锚点：`m1_muon_lr2.0_wd0.02_p0.5_s2030/final_state.pt`（SHA `4f68ce24…`，
  mte_state_v1，cursor step 2048，56 个 fp32 EMA momentum buffer，四路 RNG 齐全；
  末步 group lr=0.001 与 reference 0.01×cosine 末值 0.1 一致）。这是**指定诊断锚点**，
  不是充分调优基线（其 mult4 外推未解析的记录保留）。
- 六矩阵（人类第 2、第 7 层 = 零基索引 1、6）：layers[1]/layers[6] 各
  q_proj(512×512)、k_proj(256×512)、down_proj(512×1664)；合计 2,490,368 元素（代码实算校验）。

## 3. 数据与组批

- 生产流接续：p=0.5、seed 2030 的确定性 manifest 继续生成到第 2058 步，取后 10 步
  （`build_step_manifest(a,b,0.5,2058,2030)[2048:]`）。窗口索引自 131,072 起
  （2048×64），A/B 各 640 个。前 5 校准 / 后 5 检查的 window 集合互斥（已验证）。
- manifest 全量封存：`outputs/t2a_scale_v1/batch_manifest.json`（steps SHA
  `1e8b29c7…`）。生产 BOS/DOC_SEP 打包、mask、bf16 autocast 前向全部沿用。
- 前 5/后 5 不是独立种子或最终测试集；只作参考点冻结与检验。

## 4. 采集语义

每 batch：清梯度 → 4 micro-batch（每 micro 32 窗口、mean CE、(loss/4).backward）
→ 在**全部有梯度 unique 参数**（Muon 56 + aux 34）上 `clip_grad_norm_(1.0)` →
保存剪裁前全局范数、clip 系数、有效 targets 计数、六矩阵**剪裁后** G（fp32）。
预期每 batch 32,768 slots / 32,640 有效 targets；实测不符则先定位差异。
train loss 只作采集诊断。模型参数/缓冲、锚点文件在采集前后做内容 hash 比对；
不调用任何 `optimizer.step()`，不推进 scheduler，不写回旧实验。

## 5. H 与参考点

每矩阵：M_* = checkpoint momentum。每 batch 独立（不跨 batch 递推）：
`M_i = lerp(M_*, G_i, 1−β)`，`H_i = lerp(G_i, M_i, β)`（β=0.95，fp32 生产次序；
数学校验 `H_i = β²M_*+(1−β²)G_i`）。H_ref = 前 5 个 H 的 FP64 均值再一次转 FP32；
其 hash 与数值规则写入 `calibration_lock.json` 后才计算后 5 的标签。

## 6. 三路径与诊断

同一捕获 FP32 H_in clone 给三条路径：FP64（安装语义有限 NS）、FP32（无 BF16，
TF32 关闭）、原生（安装 `_zeropower_via_newtonschulz`，内部 BF16）。D_raw 不含
LR/WD/形状因子；s(r,c)=sqrt(max(1,A/B)) 另行登记（六矩阵均不触发转置分支）。

主观察：10 个自然 H_i 的三路径输出 + 相对偏移 ||H_i−H_ref||/||H_ref||。
幅度诊断：自然方向 E_i=H_i−H_ref，α∈{1e-3,1e-2,5e-2,0.25,1.0}×(±)，含零参考；
α=+1 强制用保存的原始 H_i。每点记录 L（FP64 切线 JVP）、delta_native、r_real、
r_numeric，核对向量恒等式 `delta_native−L = r_real+r_numeric`；相对残差分母
max(‖delta_native‖,‖L‖)，零分母记 ZERO_BOTH 并保留原始值。另存 BF16 改变元素比例、
零响应标志、余弦（零向量→null）。H_ref、batch1、batch6 的自然 H 重复映射一次核对
确定性。不拟合死区掩码；不把 T0 的 4e-3/0.2% 当通用阈值。

## 7. 预算

单 GPU（cuda:1，探测空闲）墙钟 ≤30 分钟（含梯度采集、NS、JVP、重复映射）；
新输出 ≤1 GB；≤10 个唯一 batch；失败尝试记账；模型更新数必须为 0。

## 8. 交付物

`outputs/t2a_scale_v1/`：manifest、采集原始 G/M/H_ref/H_i/D_i（自然点全量，
诊断点标量表 + 第一层 q_proj batch1/batch6 固定样例）、`map_errors.csv`、
`response_errors.csv`、`summary.json`、`calibration_lock.json`、`cost_ledger.json`、
运行日志；`docs/T2A_REPORT.md`；回归测试并入 `tests/test_t2a_scale.py`。
