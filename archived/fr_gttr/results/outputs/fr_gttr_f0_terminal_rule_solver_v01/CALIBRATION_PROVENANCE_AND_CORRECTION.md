# 校准运行台账、精度路径与更正（源码/CPU 核验轮追加；原文件未改动）

日期：2026-09-25（核验轮）。本文件为追加说明，REPORT.md 原文保留。

## 1. 两次校准运行的核对（不混用）

| 项 | CAL-1 | CAL-2 |
|---|---|---|
| SENS a | −5060.515069535734（仅存 REPORT.md §2） | +5620.036362938654（当前 CSV） |
| SENS b | −147722.88773644276 | −158605.63405911648 |
| clip JVP（sens 分支） | a0c.clip_jvp ×2 次调用，**全局 dot 正确** | inline fp64 重写，**逐块局部 dot 缺陷** |
| S 递推精度 | fp32 | fp64 |
| 源码 hash | NOT_RECOVERABLE（未 commit 即被改写；行为经本轮 trace 复核） | 16eb8538…（修复前 fr 模块） |
| FD 值 | 与 CAL-2 完全一致（value 路径两次运行未变） | CSV 现值 |

注：任何制品中都不存在 "−5620.6"（REPORT 引 CAL-1 的 −5060.515；CSV 存
CAL-2 的 +5620.036）。两次 SENS 差 ~10%：至少其一有误，可能两者都误
（CAL-1：fp32 递推在 T=256 的精度未验证；CAL-2：active-clip 段 clip JVP
缺陷）。**两次校准的 SENS 值均按无效处理；FD 值（纯 value rollout）有效。**

## 2. 本轮确认的两处实现缺陷与最小修订（仅 CPU 验证，未重跑 GPU）

D1（影响生产 active-clip 段）：f0_adamw_sens_step 灵敏度分支的 clip JVP
径向项使用了逐参数块局部点积 <g_p, Hs_p>，正确公式要求全局
dot_global=Σ_p<g_p,Hs_p> 以同一标量作用于所有块。双块反例
g=(3,4),Hs=(4,-3),tau=1,eps=1e-6：正确 dc=(0.8,−0.6)（手算/autograd jvp/
FD 三路一致）；缺陷版=(0.512,−0.216)。修订：将 dot_global 提升到参数循环
外按方向计算（fr 模块已改，修复后 hash ae4de2d3…）。
D2（潜伏，注入式使用触发）：init_sensitivity 以 dict(z) 浅拷贝构造
S_theta/S_m/S_v，三组共享内层 dict，任何注入赋值泄漏到其余两组。生产
rollout 全零初始化且首步后整体替换，未受影响；本轮诊断脚本与测试受其
污染（解释了核验中 prod≠手算 的假象）。修订：fresh() 深构造三组独立 dict。

修复后验证：tests/test_f0_clipjvp_audit_v01.py 11 项全过（含生产 dc 反例、
完整一步 active/inactive 对 FD 参考、sv=0 两分支、mask 一致性、无别名）；
原 15 项套件无回归。

## 3. 结论更正（对 REPORT.md）

- "SENS 实现正确（T=2/4/8/16 与 FD 一致 0.1%）"：该证据只覆盖
  **clip inactive** 的步（T≤16 时范数低于阈值）；active-clip 段在缺陷修订
  前从未与 FD 对照过。此句降级为"仅 inactive 段验证"。
- "T=32 定位混沌起始"：撤回。T=32 无 SENS 值；仅实测到**数值 FD 在 T=32
  翻号**（宏观斜率非单调），不能据此声称混沌起点或"SENS 被混沌放大"。
- "FULL_HORIZON_GRADIENT_UNRESOLVED——实现层无缺陷"：更正为
  **CONFIRMED_IMPLEMENTATION_DEFECT（D1）**；校准失败的部分原因可能是
  缺陷而非（或不止）混沌。T=256 校准状态改为
  UNVERIFIED_PENDING_FIXED_RERUN；是否仍 UNRESOLVED 须以修复后重跑为准
  （本轮不执行）。

## 4. 精度路径表（按环节；不得称 fp32 图的 HVP 为完整 fp64 HVP）

| 环节 | dtype/backend（CAL-1 / CAL-2 / 修复后） |
|---|---|
| value 态 theta/m/v/c | fp32 GPU（三者相同） |
| value clip 系数 | a0c：fp64 范数标量 × fp32 张量（三者相同） |
| 训练梯度图 | fp32，恒 create_graph=True（两次校准均已统一） |
| HVP dot 累加 | fp64（sum dtype=float64；三者相同） |
| HVP 双反传内部 | **fp32 图/kernel**；CAL-2/修复后仅将输出 .to(float64) 存储——是"fp32 图内计算的 HVP、fp64 存储"，不是完整 fp64 HVP |
| S 递推 | CAL-1 fp32；CAL-2/修复后 fp64 |
| sens clip JVP | CAL-1 a0c（fp64 内积、fp32 张量，全局 dot 正确）；CAL-2 inline fp64（局部 dot 缺陷）；修复后 inline fp64 全局 dot |
| 终点 q·S | CAL-1 fp32 乘积后转 fp64 求和；CAL-2/修复后双方先升 fp64 再乘加 |
| eta/psi/Z/d_eta | fp64 numpy 标量（应用时转 python float 乘 fp32 张量） |

## 5. 可恢复的校准终点值（phi=(0,0)，T=256，REF32，stream 202609250400）

见 calibration_endpoints_recoverable.csv：J(0,0) 与 8 个 ±eps 端点值
**均未持久化**（当时仅落盘 FD 导数）；可恢复的只有 4 个对称差：
a±.01 差 −1.08921e-4、a±.005 差 +1.40704e-3、b±.01 差 −6.90734e-3、
b±.005 差 −7.18415e-3。仅解释已测有限尺度：b 的宏观下降斜率在两尺度
上同号且量级一致；a 的两尺度差分不同号（有限差分噪声量级 ~1e-3/0.01）。
不据此选择新规则、不读确认集。
