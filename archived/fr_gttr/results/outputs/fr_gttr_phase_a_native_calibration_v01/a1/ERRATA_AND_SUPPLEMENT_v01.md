# Phase A1 勘误与补充（ERRATA_AND_SUPPLEMENT_v01）

日期：2026-09-23。范围：仅 0-GPU 原始 CSV/JSON 复算；不重跑 native、不扩展审计、不改旧文件。
旧值出处：REPORT.md / target_calibration.csv（原 analyze 输出）；新值出处标注于每节。
陈述分类：[事实] = 直接复算；[恒等式] = 数学可证；[诊断] = 描述性观测。

---

## 一.1 SCHEDULE 修正（权威源 = step_metrics.csv 全精度 lr 列；schedule.csv 为稀疏探针）

**旧标签（REPORT.md §1 表）**：start=82 组 "h=0 已在 floor"。**错误。**

**新事实 [事实]**：

| 组 | gm0（horizon 前最后 micro 索引） | 首步消费 LR（gm=2·(n0+1)） | first_floor_preupdate_h | floor 处 optimizer true_step | floor 处 micro pair (global-micro) |
|---|---|---|---|---|---|
| start=46 | 92 | 3.661793e-4 (gm=94) | **h=82** | 128 | (255, 256) |
| start=82 | 164 | 1.738375e-4 (gm=166) | **h=46** | 128 | (255, 256) |

- start=82 在 h=0 的 horizon 前 LR = 1.788e-4（gm=164），**不是 floor**；沿 cosine 下降
  至 h=46 首次消费 5e-5。两组在同一绝对 schedule 点（true_step=128，gm=256 = cosine
  终点）到达 floor，horizon 偏移不同（46 组多走 36 步下降段）。
- floor 处 lr = 5.00000000000000024e-5（公式在 gm=256 的 float 值）；h=83 / h=47 起
  尾部常数与该值逐位相同。
- **判定：正文标签错误，非实际配方不同。** raw 全精度 LR 序列与冻结 eta 逐位一致
  （1536/1536 步，max dev = 0.0），0 回弹事件。raw schedule **未违反**已声明规则。
- 方法备注：schedule.csv 每 cell 仅 6 个探针点（h=0,1,32,128,255,256），其
  `%.10e` 打印精度引入 ≤3.3e-15 表观舍入差，且不足以定位 floor 边界；边界判定
  必须用 step_metrics.csv。曾从稀疏 CSV 误读 first_floor=h=128/64，已纠正。

## 一.2 TOKENS 修正（step_metrics.csv 逐步求和，非均值反推）

**旧标签（REPORT.md §1/§2）**："每 cell 训练累计 valid loss tokens = 262,144"。**错误。**
262,144 = 256×1024 **input slots**。

| cell | input_slots | nonpad_tokens | valid_loss_tokens | successful_updates |
|---|---|---|---|---|
| 2030/46 | 262,144 | 240,309 | 238,261 | 256/256 |
| 2030/82 | 262,144 | 241,546 | 239,498 | 256/256 |
| 2031/46 | 262,144 | 238,819 | 236,771 | 256/256 |
| 2031/82 | 262,144 | 240,160 | 238,112 | 256/256 |
| 2032/46 | 262,144 | 240,711 | 238,663 | 256/256 |
| 2032/82 | 262,144 | 241,599 | 239,551 | 256/256 |
| **TOTAL** | **1,572,864** | **1,443,144** | **1,430,856** | **1536/1536** |

口径说明 [事实]：mean valid/step = 1,430,856/1536 = 931.55 为平均每步 valid loss
token 数，与 262,144 slots/步是两种口径，不矛盾；旧文把 slots 总数标成了 valid 总数。

## 一.3 TARGET 余量重算（monitor_row_stats.csv 逐行全精度；ell = 组内 3 seeds h128 中位数）

- ell(46) = 7.020043501537879；ell(82) = 6.979538126804536（与旧值一致）。
- 逐 cell V0−ell、ell−V256 新值 vs 旧 target_calibration.csv：**全部一致，|diff| ≤ 2e-15**。
  旧表无计算错误。
- 明确区分（旧文易混量，这里冻结口径）：ell−V256 用**组 ell**，不是本 cell V128。对照 [事实]：

| start | seed | own V128−V256 | ell−V256（正确口径） | 本 cell V128 与 ell |
|---|---|---|---|---|
| 46 | 2030 | 0.111868 | 0.111868 | ==（该 seed 即中位数） |
| 46 | 2031 | 0.121156 | 0.128347 | != |
| 46 | 2032 | 0.117617 | **0.069927** | !=（差最大） |
| 82 | 2030 | 0.110037 | 0.110037 | == |
| 82 | 2031 | 0.120693 | 0.124425 | != |
| 82 | 2032 | 0.111995 | **0.061268** | !=（差最大） |

噪声余量判据（V0−ell > 2SE0 且 ell−V256 > 2SE256）6/6 不变。

## 一.4 HIT 全精度序列（frozen `<=` 规则 + 连续两次确认）

逐 cell 9 点全精度 monitor mean（h=0/32/…/256）：

| seed/start | ell | last_above_h | first_below_h | second_confirmation_h | 与旧 CSV 一致 |
|---|---|---|---|---|---|
| 2030/46 | 7.0200435015379 | 96 | 128 | 160 | ✓ |
| 2031/46 | 同上 | 96 | 128 | 160 | ✓ |
| 2032/46 | 同上 | 160 | 192 | 224 | ✓ |
| 2030/82 | 6.9795381268045 | 96 | 128 | 160 | ✓ |
| 2031/82 | 同上 | 96 | 128 | 160 | ✓ |
| 2032/82 | 同上 | 160 | 192 | 224 | ✓ |

- 全精度 NLL 序列见复算脚本输出存档（各点 13 位小数；例 2030/46：
  h96=7.0523594243873, h128=7.0200435015379（恰 == ell，`<=` 生效）, h160=6.9905581550754）。
- 冻结规则下 V128 == ell 的 cell（2030/46、2030/82）first_below_h=128。
- **首次穿越只被 bracket 在 (last_above, first_below] 内**，monitor 分辨率 32 步；
  second_confirmation 是确认点，**不是**精确首次穿越。monotone 序列下单调穿越一次。
- 全部 hit 保持 CALIBRATION_ONLY；final_calls = 0。

## 二.1 EMA 系数 [恒等式]

| 量 | β1=0.9 | β2=0.999 |
|---|---|---|
| step 半衰期 | 6.5788 steps | 692.8005 steps |
| input-token 半衰期（×1024 slots/步） | 6,736.7 | 709,427.8 |
| valid-token 半衰期（×931.55 valid/步，A1 实测均值） | 6,128.3 | 645,359.6 |
| β^64 | 1.179020e-3 | 0.937975 |
| β^128 | 1.389050e-6 | 0.879797 |
| β^256 | 1.929970e-12 | 0.774043 |

bias-corrected 旧状态权重 w_old = β^H(1−β^n0)/(1−β^(n0+H))：

| n0 | β1: H=1 / 4 / 16 / 64 / 128 | β2: H=1 / 4 / 16 / 64 / 128 / 256 |
|---|---|---|
| 46 | 0.899288 / 0.654318 / 0.184114 / 1.169770e-3 / 1.389047e-6 | 0.978230 / 0.918146 / 0.735968 / 0.404837 / 0.247679 / 0.133512 |
| 82 | 0.899984 / 0.656060 / 0.185275 / 1.178810e-3 / 1.389050e-6 | 0.987451 / 0.951556 / 0.829965 / 0.543616 / 0.365687 / 0.212488 |

## 二.2 分层统计 h128/h256（layer_diagnostics.csv，top-3 by r_total per cell，共 36 行）

全表：`a1/layer_stats_h128_h256_top3.json`（列：W_norm, s_norm, d_norm(=d_adaptive_norm),
dot(W,s)=b_rad·W_norm², b_rad, r_tangent, r_total, b_rad/r_total, r_tangent/r_total,
radial_tangent_ident_resid）。示例（2030/46）：

| h | param | W_norm | s_norm | d_norm | dot(W,s) | b_rad | r_tangent | r_total | b_rad/r_total |
|---|---|---|---|---|---|---|---|---|---|
| 128 | layers.0.self_attn.q_proj | 5.234767 | 3.949e-3 | 3.960e-3 | −1.120e-5 | −4.088e-7 | 7.545e-4 | 7.545e-4 | −0.001 |
| 128 | layers.2.self_attn.q_proj | 5.197535 | 3.854e-3 | 3.864e-3 | 4.956e-5 | 1.835e-6 | 7.414e-4 | 7.414e-4 | 0.002 |
| 256 | layers.3.self_attn.q_proj | 5.199816 | 4.622e-3 | 4.629e-3 | 2.175e-4 | 8.043e-6 | 8.889e-4 | 8.889e-4 | 0.009 |

[诊断] 共同形态：top 相对更新 |b_rad|/r_total ≤ ~1%（近纯切向）；
[恒等式] 残差 r_total²−(b_rad²+r_tangent²) ≤ 1e-13，radial/tangent 分解自洽。

## 二.3 paired SE 与逐 seed 差异分开（BLOCK_PROXY bootstrap SE vs cross-seed range）[事实]

| 组 | 量 | per-seed delta（2030/2031/2032） | per-seed SE | cross-seed range | 结论 |
|---|---|---|---|---|---|
| 46 | Δ0→256 | 0.3804 / 0.3895 / 0.3601 | 0.0025 / 0.0020 / 0.0020 | **0.0294** | seed 间差异 ≫ SE |
| 46 | Δ0→128 | 0.2685 / 0.2683 / 0.2424 | 0.0020 / 0.0016 / 0.0016 | **0.0261** | 同上 |
| 82 | Δ0→256 | 0.2745 / 0.2825 / 0.2732 | 0.0018 / 0.0015 / 0.0018 | **0.0092** | 同上 |
| 82 | Δ0→128 | 0.1645 / 0.1618 / 0.1612 | 0.0013 / 0.0009 / 0.0012 | **0.0033** | 同上 |

SE 只反映 V_monitor 库内条件采样不确定性；seed 间差异是主要变异源，两者不混用。

## 二.4 wall-time 分列（resource_log.jsonl mark 差分，原始记录；非 forward 比例推算）[事实]

| cell | train_steps | monitor_eval | snapshot | cell 合计 |
|---|---|---|---|---|
| 2030/46 | 61.4s | 74.8s | 1.0s | 137.1s |
| 2030/82 | 63.4s | 40.5s | 0.9s | 104.8s |
| 2031/46 | 61.8s | 39.6s | 0.8s | 102.3s |
| 2031/82 | 60.9s | 39.4s | 0.9s | 101.1s |
| 2032/46 | 61.7s | 39.2s | 0.8s | 101.7s |
| 2032/82 | 61.5s | 39.6s | 0.8s | 102.0s |
| **TOTAL** | **370.7s** | **273.0s** | **5.3s** | **649.0s（逐位）** |

- monitor_eval 分量含 per-cell bank-load/setup（首个 mark 差分吸收 setup；2030/46 的
  74.8s 含一次性进程 setup ~45s）；**eval 与 setup 的进一步细分 = NOT_SAVED**（mark
  粒度限制）。
- **诊断独立墙钟 = NOT_SAVED**（诊断嵌入 step marks，原始记录不可分离）。
- 交叉参照（smoke 实测，非推算）：monitor 3.28s/eval × 54 evals ≈ 177s ≈ 273.0 − 96s setup。

## 修订后的 PhaseA 判定

所有 PASS 判定不变（SCHEDULE / COVERAGE / NATIVE_PROGRESS_MEASURABLE /
TARGET_CALIBRATION / DIAGNOSTIC_AVAILABILITY）。本轮勘误性质：2 处报告**标签错误**
（start=82 "h0 已 floor"；"valid tokens = 262,144"），底层数据与全部定量结论
（monitor 均值、paired 差、ell、余量、hit、资源对账）复算后逐位成立。字段问题
不构成训练失效证据；raw schedule 与冻结规则无冲突。
