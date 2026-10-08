# F4A.1 — Matched 5M Coverage 预注册

- 冻结日期：2026-09-30，先于任何新 5M 计算。recipe 依据：`docs/F4A1_5M_RECONSTRUCTION_AUDIT.md`（RECIPE_RECONSTRUCTABLE）。
- 目标：补齐 5M 与 26M/100M matched 的 progress 点 **step26（p=.203）/ step102（p=.797）**（step64 已有原 bank），消除 F4-A 的覆盖偏差（erratum 修正 2）。本轮**只训 5M**，不训 26M/100M。

## 1. 冻结的 seeds 与保存点

- **全部 8 个原 confirmation seeds：2030–2037**（5M 训练成本极低，~秒级/step，预算合理）。
- 保存点：2030（canary seed）保存 {26, 64, 102}（64 作 reproduction control）；2031–2037 保存 {26, 102}。
- 新 states 位置：`outputs/f4a1_matched_coverage/new_states/v07_5m/seed_<s>/state_step_<t>.pt`，用 legacy `save_full_training_state` 原格式 + F4A1 metadata。

## 2. 冻结的训练实现

逐行复刻 `e3c_prepare_state_bank_v07.py` 的训练循环（只读 import legacy `fr` 包与 `minimind_base`）：`setup_seed(seed)` → `init_model(from_weight="none")` → bf16 `make_amp` → `AdamW(lr=5e-4)`（默认超参）→ `batches_from_train_pool(train_indices, 4, seed+1704, 204)` → 逐 micro `set_native_micro_lr(get_lr, micro, 256, 5e-4)` → autocast 前向 `(loss+aux)/2` → `scaler.scale().backward()` → 每 2 micro：unscale→clip(1.0)→step→update→zero。**不因结果调整任何项**。

## 3. Canary（先于全部正式生成）

seed 2030 重训至 102。**PASS 判据（分布等价，全部满足）**，与原 8 seed bank 在**最近 matched step（64）**的经验包络比较：

1. canary step64 的 `R_curv_32` ∈ [min, max]（原 8 seed step64 实测范围，取自 F4-A metrics）；
2. canary step64 的 `lambda_SO_32` ∈ [min, max]；
3. canary step64 的首步 primal loss ∈ [min, max]（原 8 seed step64 h=1 loss）；
4. canary step64 的 `fraction_sqrtv_lt_10eps` ∈ [min, max]；
5. 5 项中允许至多 1 项越界且越界幅度 < 1 个原家族 IQR。

（R_curv/λ_SO 需对 canary states 先跑 F4-A 单元度量——度量管线即 §4。）任一硬失败 → **RECONSTRUCTION_FAILED**，停止新增 5M states，负结果保存，F4A.5 照常进入但标 coverage-limited。

## 4. 新 states 的 F4-A 度量

完全复用 `scripts/f4a_phase_map.py unit`（冻结定义：B/C/SO 三臂 H≤32 + cheap indicators；无 Arnoldi/ablation）。future stream 冻结规则：`stream_id = 4074 + i`，i 为下列冻结顺序的序号（2030:26, 2030:102, 2031:26, 2031:102, …, 2037:26, 2037:102），另 canary 2030:64 用 `4074+16`。度量输出并入 `outputs/f4a_phase_map/raw/`（与新 states 同名规则），汇总进 matched 表。

## 5. Matched-scale 表（B6）

`outputs/f4a1_matched_coverage/matched_phase_map.csv`（scale × seed × p ∈ {.203,.5,.797} 的全指标行）与 `matched_summary.csv`（scale × p 的中位/极差）。正式回答（描述性 + 冻结的 Spearman 判据）：

- same p 下 R_curv_32 是否随 N 增大（对 {5M,26M,100M} 三点中位数的 Spearman 符号）；
- same p 下 λ_SO_32 同上；
- early（p=.203）scale effect 是否成立；
- late（p=.797）是否三 scale 全部 curvature-neutral（R_curv_32 < 3）。

## 6. Verdict（冻结标签集）

`MATCHED_SCALE_TREND_CONFIRMED / MATCHED_PHASE_ONLY / NONMONOTONIC_SCALE_STRUCTURE / HIGH_SEED_HETEROGENEITY / RECONSTRUCTION_FAILED / INSUFFICIENT_MATCHED_COVERAGE / UNRESOLVED`。操作化：scale trend confirmed := p=.203 与 p=.5 两点的中位 R_curv_32 对 logN 的 Spearman = +1 且 λ_SO_32 同向；NONMONOTONIC := 任一 matched p 点中位数非单调；HIGH_SEED_HETEROGENEITY := 同 (scale,p) 跨 seed 极差 > 1 dex。

## 7. 红线

不修改 legacy（只读 import）；不污染原 bank 目录；gate FAIL 即停、负结果如实保存；不 push；新 5M 数据仅用于本 matched map 与后续 F4A.5/F4B（在 coverage 判定允许范围内）。
