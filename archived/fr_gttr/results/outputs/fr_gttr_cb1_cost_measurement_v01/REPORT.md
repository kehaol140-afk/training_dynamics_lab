# GTTR-CB1 正式报告（CB0 离线 + CB1 确定性重放）

协议：FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md。旧 16 步判定保持原样；新 4 步读数仅为 REPLAY_RESOLUTION_DIAGNOSTIC；未运行持续策略实验。

## A. 旧成本恒等式与旧 hit 复核（36 条）

- alive 加权和 vs direct 成本：**最大残差 0.0**（36/36 路径严格相等；censor=0）
- 重建 OLD_GRID16_LAG16 与 R2B 记录 hit：**36/36 一致**
- 旧检测 tie 事实复现：FULL−LR_ONLY 成本差 TIE **11/12**（非 tie 1 例：2030/82 st202 FULL 早 16 步）

## B. 固定 18 路径：旧 T16 vs 新 T4lag16（确认间隔均 16）

|state/stream/arm|T16(旧)|T4lag16(新)|diff(步)|
|---|---|---|---|
|2030/46/st00/BASE_K2|96|84|-12|
|2030/46/st00/GTTR_LR_ONLY|96|84|-12|
|2030/46/st00/GTTR_FULL|96|84|-12|
|2030/82/st02/BASE_K2|96|84|-12|
|2030/82/st02/GTTR_LR_ONLY|96|84|-12|
|2030/82/st02/GTTR_FULL|80|80|0|
|2031/46/st04/BASE_K2|80|72|-8|
|2031/46/st04/GTTR_LR_ONLY|80|72|-8|
|2031/46/st04/GTTR_FULL|80|72|-8|
|2031/82/st06/BASE_K2|80|72|-8|
|2031/82/st06/GTTR_LR_ONLY|80|72|-8|
|2031/82/st06/GTTR_FULL|80|72|-8|
|2032/46/st08/BASE_K2|112|108|-4|
|2032/46/st08/GTTR_LR_ONLY|112|108|-4|
|2032/46/st08/GTTR_FULL|112|112|0|
|2032/82/st10/BASE_K2|112|108|-4|
|2032/82/st10/GTTR_LR_ONLY|112|108|-4|
|2032/82/st10/GTTR_FULL|112|108|-4|

新网格在 18 条中改变 16 条的读数（更早确认 16 条）；2 条仍为相同检测时点。两协议确认 lag 均为 16 updates。

## C. 三臂配对成本差（slots；分 seed/start/stream，全部 12 对）

|state/stream|FULL−LR|标签|FULL−BASE|标签|LR−BASE|标签|
|---|---|---|---|---|---|---|
|2030/46/st00|0|TIE|0|TIE|0|TIE|
|2030/46/st01|0|TIE|0|TIE|0|TIE|
|2030/82/st02|-16384|FULL_BETTER|-16384|FULL_BETTER|0|TIE|
|2030/82/st03|0|TIE|0|TIE|0|TIE|
|2031/46/st04|0|TIE|0|TIE|0|TIE|
|2031/46/st05|0|TIE|0|TIE|0|TIE|
|2031/82/st06|0|TIE|0|TIE|0|TIE|
|2031/82/st07|0|TIE|0|TIE|0|TIE|
|2032/46/st08|0|TIE|0|TIE|0|TIE|
|2032/46/st09|0|TIE|0|TIE|0|TIE|
|2032/82/st10|0|TIE|0|TIE|0|TIE|
|2032/82/st11|0|TIE|0|TIE|0|TIE|

FULL−LR：负（FULL 省）1，正 0，tie 11；均值 -1365 slots。选定流（200/202/204/206/208/210）子集单独见 paired_cost_effect.csv。

## D. gap/rho 最低阶闭合（past-only；无拟合）

### 旧 16 步网格版（36 条 × landmark16/32）

- 可用 72/72；abstain 0（set()）；FORECAST_BEYOND_BUDGET 0
- 绝对误差：中位 16384 slots，均值 12743；相对误差中位 0.20
- rho 范围 [2.616e-06,6.474e-06] nats/token/slot（路径区间净进展率，非条件漂移）
### 新 4 步网格版（18 条重放，T4lag16 对照）

- 可用 36；abstain 0
- 绝对误差中位 12288 slots；相对误差中位 0.21
- 全部输入、未截断预测与逐行误差在 progress_cost_diagnostic.csv（旧）与本节（新，写入 KEY_DATA）

## E. 结论分层

- **代数一致（实现层）**：alive 加权和=截尾检测成本（36/36，残差0）；OLD_GRID16 重建与旧规则逐条一致；GRID4_LAG16 非“连续两次 4 步”（CPU 反例锁定）。
- **执行层一致（REPLAY_IDENTITY）**：18/18 路径从 h1 full-state 恢复重放，h16 与 T16 两处重叠复核全部位级一致；评测量不改变 theta/m/v/RNG。
- **经验支持（范围内）**：一次 R1 解首步改动在固定首 stream 上的配对成本差（FULL−LR：1 对省 16384 slots，其余 tie）；新网格改变 16/18 条读数（均为更早确认）。
- **未分辨/不可用**：gap/rho 最低阶闭合的误差（相对中位 0.20），不支持把 rho 当作可靠剩余成本预测子；rho 非正路径 abstain；末端 8 步网格 tie 不证明精确首达时间相等。
- **不允许**：据此设计持续训练规则、把边界解当全局最优、将重放分辨率收益回写 R2B 成功、或宣称 tokens-to-target 已改善。

## F. 成本与数据角色

- 本轮重放研究成本：updates 1710/4800；forward 47196/125000；backward 3420/9600；monitor 整库 eval 342/900；q/HVP/probe=0；V_ctrl/V_final=0；失败 0
- 原轨迹训练 token 轴不变（T16 计 elapsed slots）；原 R2B 实验成本（updates 9479 等）单列于 R2B 账本，本轮不混计。
- dense_monitor 行来源：OLD_REUSED（0/4/16 倍数旧同轴值）/REPLAY_NEW（8,12,20,... 新 4096 行 sum/count）/REPLAY_OVERLAP（h16、T16 复核，位级一致）；无 V_final 调用。

