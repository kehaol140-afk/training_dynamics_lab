# Codex 只读科学代码审计提示词（v0.6）

你正在审计一个 Transformer / AdamW 涨落—响应实验框架。

## 重要限制

只审计，不修改任何文件。
不要优化代码。
不要自动修复。
不要运行正式 GPU 实验。
输出一份 `CODEX_AUDIT_V06.md`。

## 科学协议

完整训练状态：

Z_t = {theta, AdamW m/v/step, LR path, global step, data position, Python RNG,
       NumPy RNG, Torch CPU RNG, Torch CUDA RNG}

自然涨落：
s_r = <q_t, g_t^(r)>

观测斜率：
B_theta(t,H) = Cov(delta s, X_theta,H) / Var(delta s)

因果响应：
chi_theta(t,H) =
    E[X_theta,H(+epsilon)] - E[X_theta,H(-epsilon)]
    -------------------------------------------------
                     2 epsilon

epsilon = 0.25 * sigma_q

同一 random stream 内 baseline/+/-：
- 相同初始 Z_t
- 相同 future batch manifest
- 相同 RNG
- 相同 learning-rate path
- 唯一差异是首个 optimizer step 的 grad +/- epsilon*q

扰动位置必须是：

backward / accumulation
-> scaler.unscale_(optimizer)
-> grad +/- epsilon*q
-> clip_grad_norm_
-> native AdamW step
-> zero_grad

## 请逐项审计

### A. MiniMind 原生训练语义
检查：
- get_lr 在每个 micro-batch forward 前写入 optimizer；
- accumulation；
- AMP；
- unscale；
- clipping；
- scaler.step；
- zero_grad。

### B. full-state replay
检查：
- 模型原 dtype；
- optimizer state；
- scaler；
- Python/NumPy/Torch CPU/CUDA RNG；
- restore 顺序；
- 是否存在未保存但会影响 paired branch 的状态。

### C. probe/eval 数据隔离
确认 probe/eval 从 state trajectory 的第 0 步起不进入训练 pool。

### D. q_t
确认每个 state 独立构造 q_t；
q_t 来源是 held-out train-probe；
归一化后 norm≈1。

### E. E1
确认 s_r 读取的是 optimizer boundary 上、unscale/clip 之前的 accumulated raw gradient，
并且 E1 不施加人工 perturbation。

### F. E2/E3 paired intervention
确认 baseline/+/-：
- state 相同；
- batch 相同；
- RNG 相同；
- LR 相同；
- 只有 +/-epsilon*q 不同；
- perturb 仅第一个 optimizer step 一次。

### G. random stream 与 training seed
确认：
- state 内 random streams 只是局部重复；
- 2026/2027/2028/2029 是独立训练 seed；
- 跨 seed 的 future stream seed 不复用。

### H. 统计
确认：
- 不能把 28 state 简单宣称为 28 个独立训练实验；
- 每 seed 独立映射；
- within-seed detrending；
- first difference；
- leave-one-seed-out；
- LOSO 比较 step-only 与 step+B；
- 没有为了得到正结论硬编码 scientific threshold。

## 输出格式

对 A-H 每项给：

PASS / PARTIAL / FAIL

并列：
- 文件路径
- 行号
- 证据
- 风险

最后给：
`READY_FOR_CROSS_SEED_SCIENTIFIC_INTERPRETATION = YES/NO`

不要评价实验结果好不好，只评价实现是否符合协议。
