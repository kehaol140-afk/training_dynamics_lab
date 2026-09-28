# GTTR_EVIDENCE_UPDATE — FR_C1_LATE_COMPLETION_V01

## 1. 已确立的数学恒等式（精确）

- 三分解 e = e_bank + e_local 与 ‖e‖² 恒等（含 signed_cross = 2⟨e_bank, e_local⟩）；
  late 恒等残差 ≤ 2.8e-20（fp64 舍入）。
- w_k 恒等式与 stop-v 配对结构（C1 已确立，本轮 late 复核不冲突）。
- 聚合门 = 固定 anchor 集合上的 ΣSSE/ΣTSS 比率（非逐点平均）（C0X erratum 后冻结）。

## 2. 已测经验支持（本轮新增）

- R/A/t48、C/A/t48 双通道 target 全部 resolved（signal ratio 1.94–2.07）；
  E_K32 0.222–0.248，halves ≤0.336。
- COMBINED6：field 3 distinct t（R/C PASS）；shadow 2 distinct t {16,48}（R/C PASS，
  带纪律性保留，见 REPORT §执行摘要-4）。
- stop-v：Z audit ratio 1.22–1.24、cosine 0.87；Q dev ratio 1.29–1.30（与 C1 同向）。
- clip activity：dev 18/32 全 active；audit R 出现 3/2/1 次 crossings（16 probes 上），
  C 无 crossings。

## 3. 尚不分辨（数据不足或未达预注册阈值）

- shadow t0 target（旧事实，late 不回溯）。
- 旧 shadow 每 probe 残差 SE（NOT_SAVED）；旧 stop-v 每 probe norm/cosine。
- C cell 的 Q_D/Z_A 等 q+ 投影 |mean| ≤ 2·SE。
- 半库聚合门的 g3 历史缺口（C0 DATA_MISSING 维持）。

## 4. 被特定实验削弱的假设

- "E 随 t 增长"（field late E 未升高，0.236/0.238，与早期同量级或更低）。
- "shadow 整体不可分辨"（t16/t48 均可分辨；但 t0 仍不可分辨，故削弱而非否定）。

## 5. 未检验（NOT_TESTED，维持）

未来 H16/H64 提前预测；adaptive 训练价值；最终训练改进；能耗收益。
不开发 controller/optimizer；不外推禁语（见 REPORT §纪律声明）。
