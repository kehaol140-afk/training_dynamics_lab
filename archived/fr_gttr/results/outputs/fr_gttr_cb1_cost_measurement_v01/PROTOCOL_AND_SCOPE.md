# CB1 协议与范围

本目录实现 FR_GTTR_CB1_Cost_Measurement_and_Progress_Link_v1.md：CB0 纯离线（36 条 R2B 旧轨迹的成本恒等式与past-only 进展诊断）＋CB1 固定首 stream 18 条的确定性重放（4 步观测网格、16 步确认间隔，REPLAY_RESOLUTION_DIAGNOSTIC）。

- 不增加优化器/预测器；研究一次 R1 已求解首步改动后共同 AdamW 基线下的成本读出。
- 训练 token 轴（轨迹 elapsed slots）与重放研究成本分开记账。
- rho<=0 -> NO_POSITIVE_PAST_PROGRESS；预测不截断；FORECAST_BEYOND_BUDGET 保留。
- 旧 16 步判定保持不变；新读数不回写 R2B 成功结论。
