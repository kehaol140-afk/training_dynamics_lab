# CHANGE_LOG — FR_G1B_100M_SCALE_GENERALIZATION_V01

本文件记录 G1b 执行期间的全部偏离/事件。G1b 按冻结协议一次通过，无协议偏离、
无 OOM、无 retry、无 RESOURCE_LIMIT。

## 事件

1. **scale-only 配置冻结**（结果前）：静态 CPU 实例化实测 unique params =
   100,784,000（hidden=768, layers=13, heads=12, kv=6, head_dim=64,
   intermediate=2432 = ceil(768·π/64)·64 冻结默认公式），落在 mandate 窗口
   95M–110M，按 mandate 立即冻结，未尝试任何其它尺寸；state_dict elements
   = 105,699,200（tied embeddings）。记录于 `research/configs/g1b_100m_v01.json`
   与 `G1B_PROTOCOL.md` §2。
2. **训练一次通过**：3 seeds（2042/2043/2044）× 3 stages（EARLY=26 /
   MIDDLE=64 / LATE=102），每 seed 峰值显存 ≈ 2.56 GB；无 OOM；未启用
   device microbatch adjustment（batch 4 × accum 2 按注册值运行）。
3. **theta 一次通过**：9 态 q 构造 + 16 E1 流 ε 标定 + 36 条 E2 manifest
   全部先于任何 E2 分支写入并冻结（`future_stream_manifest.json`）；
   三分支 CRN / rho0-noop / paired LR path 全部一致；3 seed theta_mapping
   technical_pass=True。
4. **smoke gate PASS**（seed 2042 / EARLY step 26，mandate 指定态）：13/13
   检查通过；H=1 analytic vs FD rel err = 0.0183；math/native HVP gap max =
   3.06e-3；GPU peak memory = 11,032 MB；无意外 optimizer step。
5. **tangents 一次通过**：36/36 baseline replay 与 shadow 分支哈希逐位一致；
   native clip 张量 L2 gap max = 1.94e-07。
6. **mechanism 一次通过**：seed 2042 × 3 stage × 4 stream，H=1 恒等
   max|BH_fixed−B| = 0（精确 0）。
7. **analyze / verdict**：C pooled R²=0.99949、H>1 paired 18/18/0、
   3/3 seed 正、H=8 C R²=0.99917/corr>0 ⇒ **STRONG_SCALE_REPLICATION**。
8. **audit**：250 文件入包、24 个大文件（bank/ 9 checkpoint + theta q 张量）
   仅留哈希不入包；审计包 SHA256 =
   `6fe9b0f99aac2f4c759cd2f07d606931b89a631fa65faa3720e32ade9f7a5f85`。

## 协议偏离

无。训练 schedule、response 定义、CRN、q 语义、ε 规则、判定阈值均与
`G1B_PROTOCOL.md`（2026-09-16T15:35Z 冻结）一致。

## 已修复实现问题（执行中发现并修复，非协议偏离）

- 初版 runner analyze 段在 26M 行内联了一段错误的 H8 mean rel err 计算
  （引用不存在的 summary 字段）；修复为从冻结 G1a `per_state_horizon.csv`
  直接读取（结果数值不变，该内联表达式恒为 None）。修复发生在任何
  analyze 运行之前。
