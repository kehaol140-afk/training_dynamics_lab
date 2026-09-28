# KEY DATA FOR REVIEW — fr_phase_b_gttr_action_scoring_v03（2026-09-23）

## 一句话结论
预测封存前的预注册 FD 门在 **S3** 触发 `LOCAL_OPERATOR_NOT_CALIBRATED`
（fd_rel=0.7061>0.20），按 v3 §3.2 停止 GPU 主流程；预测未生成、
main 未启动；S0–S2 同门通过（0.0011/0.0660/0.0206）。

## 门结果（第一方向 half-check，δ=1e-3·max(‖θ0‖,1)）

| state | seed | n0 | rank | fd_rel | fd_cos | 结果 |
|---|---|---|---|---|---|---|
| S0 | 2030 | 46 | 4 | 0.00107 | 0.99999948 | PASS |
| S1 | 2030 | 82 | 4 | 0.06601 | — | PASS |
| S2 | 2031 | 46 | 4 | 0.02062 | — | PASS |
| S3 | 2031 | 82 | — | **0.70614** | **0.72824** | **FAIL→STOP** |
| S4 | 2032 | 46 | — | — | — | 未执行 |
| S5 | 2032 | 82 | — | — | — | 未执行 |

S0 完整 smoke 指标：‖g0‖=0.7659，‖q0‖=2520.58，‖s_ref0‖=0.1995，
orth_max=1.1e-8，labels=FULL_Q4。

## 预算（含 smoke 与门失败；cap 全过）

updates 16/1280；总 fwd 570/14500；总 bwd 162/2920；model 130/360；
FD 22/50；monitor eval 3/84；control eval 3/90；虚拟算术 1536 步；
SIM 5120 slots；峰值显存 0.82 GB；V_final=0；失败计数=1（门停止）。

## 关键文件
- 门诊断图：`plot_gate_fd_rel.png`
- 构建记录：`model_construction_reconstructed.csv`
- 状态汇总：`round_status.json`；账本：`counters.json`
- 测试：`research/tests/test_phase_b_gttr_v03.py`（31 项 PASS，
  含 amendment 计时器隔离：elapsed=5401 门不变）

## 评审要点
1. 门失败是 S3 局部性质（同代码路径 S0–S2 通过 1e-3–7e-2 量级）。
2. 协议纪律：未调 ε、未剔除状态、未部分封存、未进 main。
3. B–F 科学读数因此为 NOT_TESTED（前置条件未满足），非负结果证据。
