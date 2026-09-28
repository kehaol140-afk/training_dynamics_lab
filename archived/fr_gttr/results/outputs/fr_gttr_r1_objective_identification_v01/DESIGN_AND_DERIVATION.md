# GTTR-R1 设计与推导记录（DESIGN_AND_DERIVATION）

协议：FR_GTTR_R1_Model_To_Rule_Identification_v1.md（2026-09-23）。本文件只
记录公式与本地接口映射，不扩展宏观理论。

## 1. 控制切片坐标（协议 §1）

在同一恢复状态 z=(theta, m_old, v_old, n0) 与当前训练样本 omega 下，按冻结
配方计算累计梯度、全局 clip、raw moments、bias correction（[FR-NATIVE-ADAMW]）：

    c = clip(g(omega))                  # 2 microbatch 累计，tau=1.0
    m_plus = beta1*m_old + (1-beta1)*c
    v_plus = beta2*v_old + (1-beta2)*c^2
    mbar = m_plus/(1-beta1**t),  t = n0+1（true step）
    vbar = v_plus/(1-beta2**t)
    P = 1/(sqrt(vbar)+eps)              # 逐 probe 计算，不冻结随机 P

参考学习率 eta = k * schedule_eta(gm)，gm = 2*n0+2（本步最后一个 micro 的
gm，与 Phase A/D1 的 per-micro 写入语义一致）；k∈{1,2}。相同样本的 k1/k2
共用相同 m_plus/v_plus/P（本配方中它们不依赖本次 LR）。

    d0 = -eta*(P*mbar + wd*theta)       # native 完整位移（含衰减）
    d1 =  eta*P*(mbar - c)
    theta_plus = theta + d0
    theta(a) = theta_plus + x*d0 + y*d1
    eta_new = eta*(1+x), alpha_new = y/(1+x),  x > -1

精确重参数化（T1 CPU 测试对照直接读出式）：
    theta_new = (1-eta_new*wd)*theta
                - eta_new*P*((1-alpha_new)*mbar + alpha_new*c)

## 2. 目标系数（协议 §2）

V_ctrl = V_control_ref64：outputs/fr_decision_utility_v01/validation_split.json
的 v_score 64 行；FP32、autocast off、eval mode、batch 8、全库 sum/count。
q/H 与全部端点求值统一使用 MATH SDPA 后端（二阶可微；协议允许的已知数学
后端例外），记录 base loss/q/dtype/backend。

    b_i = D_i^T grad V_ctrl(theta_plus)      （全库 sum 梯度 / N_total）
    K_ij = D_i^T Hessian V_ctrl(theta_plus) D_j

实现：8 chunk 前向累计 S=sum(ce)；一次 create_graph 一阶梯度；dot0/dot1
（方向 detach，fp64 dot）；两次二阶 VJP 得 H d0、H d1；所有系数除以全库
有效分母 N_total（不能无权平均 batch means）。K12_direct/K21_direct 均保存，
相对非对称 <=1e-4 才用 (K+K^T)/2 求解；原值照报。

## 3. 受限求解（协议 §2.1）

m(a)=b^T a+0.5 a^T K a；主可行域 x∈[-r,r], y∈[0,r]，r_primary=.05（主），
r_half=.025（固定尺度诊断）。矩形域二次函数完整候选枚举：4 角点、原点、
四边内部极小点、可行内部驻点（cond<1e14 才尝试）；按 (m, y, |x|, x) 确定性
选优；保留负特征值，不 PSD 裁剪，不加 jitter；无法数值求解 -> UNRESOLVED。
Schur 消元仅在 K 正定且 K11、K_eff 非退化时输出 b_eff/K_eff/y_star/x_star。
y=0 的 KKT 依据：K 半正定（容差 1e-12 相对）且 b2+K21*x_LR>=0。

## 4. 状态/probe/预算映射（协议 §3、§7）

- 6 state：seeds 2030-2032 × start 46/82（v07 bank）；真实 n0 从 checkpoint 读。
- 每 state 16 个 effective-batch probes（FIT 8 / AUDIT 8，各 2 micro）；
  SIM seeds 202609230100..105（按 (seed,start) 排序；冲突整块 +1000，结果前冻结）。
  采样：derive_stream_seed(sim_seed, 404) -> make_permutation(3936) -> batch 4
  （[FR-BATCH-STREAM] 冻结 train pool 内，不碰 probe/eval）。
- k1/k2 共享 16 probes；FIT 全部封存（prediction_seal.json 覆盖
  per_probe_coefficients.csv / model_coefficients.csv / restricted_solutions.csv）
  后才读 AUDIT；AUDIT 不回写系数（T8）。
- AUDIT 坐标：两半径网格 x∈{-r,0,r}×y∈{0,r/2,r} 合并 16 唯一坐标 + 两半径
  各自 a_LR/a_2D（坐标去重仅复用）；native 差严格 0（同参数确定性求值）。
- V_check256：monitor 前 256 行（ID 4096..4351），batch 32，PhaseA 语义
  （bf16 autocast + logits.float()），仅 primary 的 native/a_LR/a_2D。
- 预算（协议 §7）：train fw/bw 192；FIT q/H fw 768、一阶 bw 768、VJP 方向
  192（micro 口径 1536）；AUDIT ctrl fw<=15360+base 1536；check fw 2304；
  全部计入 resource_log.jsonl 硬上限。

## 5. 分析口径（协议 §5）

- bootstrap：effective-batch probe 为单位，1000 次，PCG64 seed 2026092301；
  FIT 与 AUDIT 独立重抽；k1/k2 与动作保持配对（同 state 共用索引抽）。
- 模型误差：固定网格（16 坐标，不含必为 0 的 native）SSE/TSS；
  max|J_audit|<1e-9 -> LOW_SIGNAL（诊断阈值，非过关门）。
- 判定标签：MODEL_RECOMMENDS_NO_READOUT_CHANGE / PREDICTED_BENEFIT_NOT_
  CONFIRMED / TARGET_TRANSFER_NOT_ESTABLISHED / UNRESOLVED_*；
  汇总按 seed/start/k 分层（6x2 条件不是 12 个独立 seeds）。
