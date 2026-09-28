# 关键数据摘要

1. h16 不可比的两项 CONFIRMED_DIFFERENCE：schedule Z16/Z256（首步 eta 差
   11.5%）与终点目标（随机线性 vs V_design 4096 行 sum/count）。
2. D3（q 缺 /count）与 D4（Z 归一范围）已做最小修复（CPU-only）；
   26+9 组回归全过。
3. 生产内核 FP64 合成精度 5.6e-17（九组逐步无分歧）。
4. b 坐标 h256 修正后 CALIBRATED（|SENS−FD005|=0.0484 ≤ tol=0.0723）。
5. a 坐标 h256 FD 对自身不一致（FD01=−0.005, FD005=+0.141），与 SENS
   修复无关。
6. 下一 GPU 步骤：修复版完整 256 步校准重跑。
