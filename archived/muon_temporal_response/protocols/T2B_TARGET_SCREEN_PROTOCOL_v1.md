# T2B_TARGET_SCREEN_PROTOCOL_v1 — 目标价值前提实验（已授权执行）

日期：2026-09-29。授权：`references/ROUND3_TARGET_SCREEN_OPENCODE_CN.md`（用户批准）。
冻结配置：`configs/t2b_target_screen_v1.json`；数据/排列 manifest：
`outputs/t2b_target_screen_v1/data_and_perms_manifest.json`（SHA `31cf25f1…`）。

## 1. 设计一句话

同一 Muon 锚点（step2048 终态）上，两个 10-batch 窗口 × 8 个预定 shuffle 排列：
先在 θ* 冻结全参数梯度做 16 条完整非线性虚拟回放（10 步排列 + 128 步公共收尾），
以 Q_select 的虚拟终点 NLL（S_fin）事前锁定主候选 pi_F，再跑 16 条真实分支
（每步重算梯度），用 Q_check 在 step10/138 检查收益是否真实存在并保留到收尾后。

## 2. 关键口径（全部冻结于 config）

- **常量 LR 平台**：两组 LR=0.001（checkpoint 末步原值 = 峰值 0.01 × cosine 末值 0.1；
  不使用 0.01 峰值）、WD=0.02、Muon β=0.95/Nesterov/NS5 原系数、AdamW aux (0.9,0.95)/1e-8
  均从 checkpoint param_groups 读取。无 scheduler 推进。
- **数据块**：W0=[2048:2058)（与 T2a manifest 逐成员一致，已验证）、W1=[2058:2068)、
  C=[2068:2196)；两两互斥、每 batch 64A+64B。源切片索引只是样本身份，不是 LR 时钟。
- **排列**：PCG64(883000+100b+k).permutation(10)；8×2 全唯一（无碰撞，未触发 +10000 规则）；
  pi0 = 各窗 k=0。
- **Q 目标**：旧 dev 池按 conversation_id（翻译线程组）划分
  `sha256('T2b_Q_v1|'+gid)[0]%2`；select(A 801/B 941 docs) 与 check(A 1012/B 1299 docs)
  组级互斥（已验证）；每域 512 窗×255 targets=130,560 tokens；
  V=0.5·NLL_A+0.5·NLL_B（域等权）。confirm 全程封存。
  旧 dev 曾参与 M1/M2 —— 本轮结果只能称"选择持出的开发检查"，不是从未使用的 final。
- **回放语义**：θ* 处求全部 148 个 batch 的生产梯度（4 micro 累积 + 全参 global clip 1.0，
  post-clip G）；16 条虚拟轨迹完整递推 Muon+AdamW（含 WD、bias correction、step 计数），
  排列内不重置任何状态；C 尾流式共享同一批冻结 G；G 不二次 clip、无别名污染。
- **分数与锁定**：S_fin(π)=V_Qselect(θ̃₁₃₈^π)（主）、S_lin(π)=q_select·(θ̃₁₃₈^π−θ̃₁₃⁸^π0)
  （副对照）；pi_F=argmin S_fin（并列 ≤1e-6 取最小 k）；`selection_lock.json` 先于任何
  Q_check 访问写入。
- **真实分支**：16 条（2 窗 × 8 排列）各 138 步，每步当前参数重新求梯度、全参 clip、
  双 optimizer 实际 step；+1 条 W0/pi0 重复性检查（非独立样本）；
  step10 只诊断。NaN/Inf 记 recipe_divergence，不与 technical failure 混算。
- **筛查**：delta_screen=0.001 nats/有效target（投入决策用，非论文标准）；
  delta_num=max(1e-8, 10×e_repeat)。
  判定标签：TARGET_SELECTION_SIGNAL / EFFECT_WITHOUT_RELIABLE_SELECTION /
  NO_USEFUL_SIGNAL_THIS_SETTING / BLOCKED。

## 3. 预算

3 GPU-hours（含全部梯度、回放、评价、失败尝试）；CPU 30 min；输出 ≤8 GiB。
虚拟更新 2208；实际更新 ≤2346（16×138+138 重复）=76,573,440 有效训练 targets；
冻结梯度 batch 148。slots 与有效 targets 分列（32,768 vs 32,640/update）。

## 4. 交付

`outputs/t2b_target_screen_v1/`：shadow_scores.csv、selection_lock.json、
actual_objectives.csv（逐域 sum/count/NLL/V @step10/138）、prefix_tail_diagnostics.csv、
target_gain_summary.json、cost_ledger.json、run logs、`REPORT_CN.md`。
回归测试 `tests/test_t2b_screen.py`。
