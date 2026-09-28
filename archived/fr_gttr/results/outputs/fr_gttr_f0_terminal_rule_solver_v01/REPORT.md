# GTTR-F0 正式报告（BLOCKED_AT_CALIBRATION；未运行 solver/audit/confirm）

协议：FR_GTTR_F0_Full_Horizon_Terminal_Rule_Solver_v1.md。本轮在完整 256 步
终点导数校准处停止：FULL_HORIZON_GRADIENT_UNRESOLVED。未读取 V_confirm_F0
（0 次调用，保持封存）；未冻结五臂 policy；未运行 NATIVE/REF32 audit。

## 1. 实现与验证状态（支持）

- 规则数学（w/psi/Z/eta 及 a,b 导数，含 Z 导数）：CPU 15 项测试全过
  （cpu_tests.json）；phi=0 精确恢复 w（首 7.323586e-4=2×R1 eta@gm94，
  尾 1e-4）；sum eta=exp(a)·sum w 至 1e-9。
- 全状态前向灵敏度实现：**在 T=2/4/8/16 上与纯数值 FD 一致到 0.1%**
  （horizon_scan.csv；含 clip active、moments、P 反馈、wd、bias correction；
  CPU fp64/fp32 对照完整未截断 autograd 精确到 1e-7/2e-7）。
- GPU smoke PASS：REF32±SENS 位级一致（含 RNG）；NATIVE 规则 wrapper
  phi=0 与原 k2 位级一致。恒用 create_graph=True 统一算术（该选择本身
  会改变梯度最后一位舍入，已统一避免混用）。
- 灵敏度递推使用 float64（fp32 在 1/sqrt(v)^3 条件数下灾难消零，属于
  精度实现选择，不改值轨迹；值轨迹位级不变已由 smoke 验证）。

## 2. 校准数值（失败；terminal_gradient_check.csv）

固定 FIT seed2030、phi=(0,0)、REF32 256 步 + SENS；8 条 ±eps 全程 rollout。

| 坐标 | SENS | FD(eps=.01) | FD(eps=.005) | tol | 判定 |
|---|---|---|---|---|---|
| a | -5060.515 | -0.005446 | +0.140704 | 1012.1 | UNRESOLVED |
| b | -147722.888 | -0.345367 | -0.359208 | 29544.6 | UNRESOLVED |

eps_V=1e-7（同终点重复评价位级一致）；abs_floor=5e-4；clip active 184/256 步。

诊断证据（计入成本，含 3 次未通过 smoke 尝试的失败计费）：
- SENS 实现正确：T=2 纯数值 FD=-1.0459 vs SENS=-1.0451；T=4/8/16 比值
  0.999/0.999/1.000（horizon_scan.csv）。
- T=32 数值 FD 符号翻转（+27.99）：训练程序对 phi 的敏感在 16<T<32 进入
  混沌区。到 T=256 局部导数量级 ~1e2-1e5（SENS），而有限差分测得的是
  宏观有效斜率（b 坐标两尺度 -0.345/-0.359 一致但与局部导数差 4.5e5 倍；
  a 坐标两尺度自身不一致，FD 噪声主导）。
- 结论：实现层无缺陷；T=256 全时域梯度在该真实程序上不可用作下降信息。
  这不是可以通过调 eps/精度/T 修复的问题（协议禁止也不应尝试）。

## 3. 未运行阶段（BLOCKED）

- 九点 grid、FULL256/SCALE256/SHORT64 求解器：未运行（依赖终点导数）。
- 五臂 policy 封存、NATIVE 30 条、REF32 6 条、V_design 过程读数：未运行。
- V_confirm_F0：0 次调用（文件已提取并 sha 封存，未读取）。
- SHORT64 Z 归一化按 T=256 全规则切片（诊断中发现的实现修正，未来轮生效）。

## 4. 结论分层

- 支持：规则数学、灵敏度实现（短时域精确）、smoke 全部位级检查。
- 失败：T=256 终点导数校准（FULL_HORIZON_GRADIENT_UNRESOLVED）——
  实现正确但目标量本身混沌不可用。
- 未分辨：a 坐标 FD 两尺度不一致（宏观噪声）；chaos 起始精确时点未定位
  （在 16 与 32 之间，未继续诊断轮）。
- 未测试：DEPLOYABLE_OPTIMIZER_BENEFIT、SECOND_ORDER_EXTRA_VALUE、
  V_FINAL_CONFIRMED、一切 solver/audit/confirm 比较。

## 5. 成本（全部失败/诊断/smoke 计费）

| 计数 | 实际 | 上限 |
|---|---|---|
| trajectory_update_evals | 4864 | 34000 |
| network forward | 13824 | 110000 |
| backward/VJP microcalls | 12776 | 90000 |
| train-HVP directions | 1140 | 7000 |
| design full-set evals | 32（含 6 带 gradient） | 250（grad 24） |
| confirm evals / V_final | 0 / 0 | 40 / 0 |

失败记录：3 次 smoke 会话（2 次实现修正 + 1 次计数判定修正）；1 次校准
失败；2 轮导数诊断（小 T autograd 对照——其中 functional_call 对照版自身
有误，最终以纯数值 FD 仲裁；T 扫描）。wall 仅 OBSERVATIONAL_SHARED_GPU。

## 6. 对后续的含义（不自动执行）

固定预算终点 NLL 的可微求解路线在当前真实 GTTR 语义下被本轮证据否定于
校准门：全时域超参数导数虽可精确实现，但其值被混沌放大主导。后续若继续
F0 主线，应转向非导数搜索（九点 grid 类）或缩短有效时域分解；该决定留给
研究者，本轮不执行、不烧确认集。

## 7. 追加更正（源码/CPU 核验轮；原文以上保留）

见 CALIBRATION_PROVENANCE_AND_CORRECTION.md：确认两处实现缺陷（D1
active-clip 灵敏度 JVP 使用逐块局部点积、D2 init_sensitivity 浅拷贝别名），
已最小修订并通过 11+15 项 CPU 测试；两次校准 SENS 值均按无效处理；
"T=32 混沌起点"与"实现层无缺陷"两处结论撤回/更正；T=256 校准状态改为
UNVERIFIED_PENDING_FIXED_RERUN（本轮未重跑 GPU）。
