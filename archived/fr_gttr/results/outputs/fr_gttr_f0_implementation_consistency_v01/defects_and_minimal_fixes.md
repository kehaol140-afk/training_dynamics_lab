# F0-IC v1：确认缺陷与最小修复

## D3（IC 本轮确认）：eval_full_library 的 q 缺 /count 归一

- defect_id: D3
- file: research/scripts/f0_terminal_rule_run_v01.py : eval_full_library
- expected_formula: V = sum_j nll_sum_j / sum_j valid_count_j；
  q = grad V = grad(sum) / count
- failing_input: 两 chunk 有效 token 4/6，count=10
- before_output: pattern q 与 grad(mean) 比值 = 10.0（恰为 count）
- independent_expected: 比值应为 1.0
- minimal_diff: q = {n: p.grad...clone() / max(cnt,1) for ...}（1 行）
- after_output: 本轮 CPU fixture 已验证公式正确（修复行含 /max(cnt,1)）
- 影响：CAL-1/CAL-2/CALFIX-1 的全部 SENS 值被 count=475330 放大；
  b 坐标 h256 修正后 |SENS−FD005|=0.0484 ≤ tol=0.0723 -> CALIBRATED。
  a 坐标 FD 对自身不一致（0.146 > 0.029）不受此修复影响。
- test_command: python f0_ic_adapter_v01.py --defects
- source_hash_before: fc3aa7cf…(旧 runner)
- source_hash_after: 修复后 runner（含 /max(cnt,1)）

## D4（IC 本轮确认）：ref32/native rollout 用 build_w(T) 改变 Z 归一范围

- defect_id: D4
- file: research/scripts/f0_terminal_rule_run_v01.py : ref32_rollout L287 / native_rollout L453
- expected_formula: schedule 恒 T256：Z(b)=sum_{t=0..255}；rollout 截取 eta[:T]
- failing_input: phi=(0.1,-0.2), T=16
- before_output: eta16_first=0.000818 ≠ eta256_first=0.000912（差 11.5%）
- independent_expected: eta[:16]==eta256[:16]（差=0）
- minimal_diff: build_w(T_MAIN)/build_psi(T_MAIN) + eta[:T] 截取（2 处各 2 行）
- after_output: 修复后 26/26 CPU 测试无回归；适配器九组通过
- 影响：旧 scan h16 的 SENS/FD 使用了不同 LR 序列（Z16≠Z256），
  其数值与 CALFIX 的 h16 不可比（HORIZON_PROVENANCE=DIFFERENT_OBJECT）。
- test_command: python f0_ic_adapter_v01.py --defects
- source_hash_before: fc3aa7cf…
- source_hash_after: 同 D3 修复文件

## 不构成新缺陷的项

- init_sensitivity 别名（前轮 D2）：生产主循环零初始化不受影响；
  本轮适配器使用修复版，九组一致。
- 跨块 clip JVP（前轮 D1）：修复版全局 dot 在九组 FP64 中验证到 5.6e-17。
