# GTTR-F0 实现一致性检查（IC v1）正式报告

协议：GTTR_F0_Implementation_Consistency_Package_v1/START_HERE.md。
本轮 0 GPU、0 真实训练更新、0 V_design/V_confirm/V_final 网络调用。

## 状态字段

    FORMULA_REFERENCE = PASS（附件参考原样重跑 26/26 断言全过）
    PRODUCTION_KERNEL = PASS_WITHIN_TEST_SCOPE（9组 FP64 合成，最大差 5.6e-17）
    LOSS_AND_READOUT = DEFECT_FOUND（D3 已修 CPU-only；D4 已修 CPU-only）
    HORIZON_PROVENANCE = DIFFERENT_OBJECT（旧 scan h16 与 CALFIX h16 不可比）
    KNOWN_DEFECTS = FIXED_CPU_ONLY（D3、D4 各一处最小修复）
    FULL_TRANSFORMER_256_SENS = NOT_REVALIDATED
    GRID_SOLVER_AUDIT_CONFIRM = NOT_RUN

## A. 旧 h16 vs 新 h16：DIFFERENT_OBJECT（两项 CONFIRMED_DIFFERENCE）

| 差异项 | 旧 scan h16 | CALFIX h16 |
|---|---|---|
| **schedule_total（Z 归一）** | Z 对 **16** 项归一（build_w(16)） | Z 对 **256** 项归一 |
| **首步 eta**（phi=(0.1,−0.2)） | 0.000818 | 0.000912（差 11.5%） |
| **终点目标** | 固定随机线性 dot(randn,theta) | V_design 4096 行 sum/count |
| target_count | 4（参数数） | 475330（全库有效 token） |

结论：旧 scan h16 的 SENS≈FD≈−22 是 **Z16 归一+线性目标** 上的一致（同一
对象的 SENS/FD 互恰——这正是它通过的原因），但 **不是 T256 规则的 h256
目标导数**。D4 修复消除了 schedule 差异；D3 修复消除了 q 的 count 倍缩放。
两项修复后，旧 scan 与 CALFIX 的 h16 数值都需以修复版重跑才能比较。

不称"16步后进入混沌"——h16 对象本身不可比，不能作为动力学结论。

## B. 真实生产函数路径、SHA 与是否实际调用

| 组件 | module.__file__ | SHA | 被调用 |
|---|---|---|---|
| f0_terminal_rule_v01.py | research/src/fr/ | ae4de2d3… | ✓ 适配器 9 组逐步调用 |
| f0_terminal_rule_run_v01.py | research/scripts/ | fc3aa7cf…(前)→be71534…(后) | 修改对象（D3/D4） |

关键数值（tau=0.25, h=256, 逐步一致无分歧）：
- max_diff theta = 2.78e-17
- max_diff S_theta = 2.78e-17  
- max_diff S_m = 1.56e-17
- grad_max_abs_err = 0.0
- 首次分歧 = 无

## C. 独立参考与生产路径的分项误差

九组（tau×h）：全部通过。最大逐分项误差见 production_reference_compare.csv。

| tau | h | theta | S_theta | S_m | S_v | grad | 首分歧 |
|---|---|---|---|---|---|---|---|
| 0.02 | 256 | 0 | 1.7e-18 | 1.1e-19 | 1.1e-22 | 0 | 无 |
| 0.25 | 256 | 2.8e-17 | 2.8e-17 | 1.6e-17 | 1.1e-19 | 0 | 无 |
| 100 | 256 | 1.4e-17 | 5.6e-17 | 4.2e-17 | 1.7e-18 | 6.9e-18 | 无 |

规则导数：autograd jacobian 误差 6.5e-19；sum eta 恒等式误差 0。

## D. HVP 与目标 q 的覆盖范围

| 检查 | 状态 |
|---|---|
| HVP 数学（解析 Hessian · S） | EXECUTED_CPU（九组适配器） |
| HVP 真实模型（4.98M Transformer） | NOT_RUN_GPU |
| 目标 V 值公式 sum/count | EXECUTED_CPU（D3 fixture） |
| 目标 q 公式 grad(sum)/count | EXECUTED_CPU（D3 fixture 修复前后） |
| q/S 同 theta_h 同时点 | STATIC_CHECKED |
| eval RNG 隔离 | STATIC_CHECKED |

## E. D3 修复前后关键数值

修复前：pattern q 与 grad(mean) 比值 = count（D3 fixture：count=10）。
修复后公式：q = grad(sum)/max(cnt,1)（1 行改动）。

**修正后的 h256 门（CALFIX-1 值 / count=475330）：**

| 坐标 | SENS 修正 | FD01 | FD005 | \|FD01−FD005\| | tol_fd | \|SENS−FD005\| | tol | 判定 |
|---|---|---|---|---|---|---|---|---|
| a | −0.010646 | −0.005446 | +0.140704 | 0.1462 | 0.0286 | 0.1514 | 0.0286 | **FD_INCONSISTENT** |
| b | **−0.310780** | −0.345367 | −0.359208 | 0.0138 ✓ | 0.0723 | **0.0484 ✓** | 0.0723 | **CALIBRATED** |

b 坐标 h256 修正后通过原门（含独立 FD 对一致性）。a 坐标 FD 对自身不
一致（0.146 >> 0.029），与 SENS 无关——有限差分在该尺度/方向噪声主导。

## F. 下一次真正需要 GPU 检验的唯一剩余问题

以修复版源码（D3+D4），在原 state/stream/规则下重跑一次完整 256 步
SENS+FD 校准。预期 b 坐标 CALIBRATED（已有离线算术支持）；a 坐标 FD
对一致性受宏观差分噪声限制，可能维持 FD_INCONSISTENT（非 SENS 错误）。

## 成本

本轮 0 GPU、0 真实训练更新、0 V_design/V_confirm/V_final 调用。
CPU 合成：附件参考（~5s）+ 适配器九组（~3s）+ D3/D4 fixture（<1s）+
26 项回归（~1s）。墙钟仅记录。
