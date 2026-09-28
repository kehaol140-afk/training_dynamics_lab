# FR GTTR-D1 直接更新规则 pilot v1 —— 结果报告

日期：2026-09-23。执行端：Claude Code + Kimi K2.7。
协议：FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md。
性质：已知 checkpoint（v07 seed2030-2032 start46）上的开发 pilot；不是完整预训练/最终能力/墙钟/能耗结果。QH_RAW 为 QHAdamW 族（Ma & Yarats ICLR 2019, nu1=.9 nu2=1）读出对照，范数 grafting 有先例（Agarwal et al. 2020）；本轮不作首创声明。

## 判定

- **IMPLEMENTATION**: PASS
- **COVERAGE**: PASS
- **FINAL_BUDGET_NLL_EFFECT**: NO_OBSERVED_GAIN
- **DETECTED_TARGET_TOKEN_EFFECT**: NO_OBSERVED_GAIN
- **NORM_MATCHING_ABLATION**: RAW_BETTER_THAN_NORM
- **FIXED_LR_COMPARISON**: BASELINE_GRID_BOUNDARY

## 表 A：3 seeds × 5 arms 配置与覆盖

| seed | arm | kind | k | alpha | start(n0) | stream | LR首→尾 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 2030 | ADAMW_LR050 | native | 0.5 | 0.0 | 46 | 99000 | 2.50e-04→2.50e-05 | 256步完成 |
| 2030 | ADAMW_LR100 | native | 1.0 | 0.0 | 46 | 99000 | 5.00e-04→5.00e-05 | 256步完成 |
| 2030 | ADAMW_LR200 | native | 2.0 | 0.0 | 46 | 99000 | 1.00e-03→1.00e-04 | 256步完成 |
| 2030 | QH_RAW_A010 | QH读出 | 1.0 | 0.1 | 46 | 99000 | 5.00e-04→5.00e-05 | 256步完成 |
| 2030 | QH_NORM_A010 | QH读出 | 1.0 | 0.1 | 46 | 99000 | 5.00e-04→5.00e-05 | 256步完成 |
| 2031 | ADAMW_LR050 | native | 0.5 | 0.0 | 46 | 99001 | 2.50e-04→2.50e-05 | 256步完成 |
| 2031 | ADAMW_LR100 | native | 1.0 | 0.0 | 46 | 99001 | 5.00e-04→5.00e-05 | 256步完成 |
| 2031 | ADAMW_LR200 | native | 2.0 | 0.0 | 46 | 99001 | 1.00e-03→1.00e-04 | 256步完成 |
| 2031 | QH_RAW_A010 | QH读出 | 1.0 | 0.1 | 46 | 99001 | 5.00e-04→5.00e-05 | 256步完成 |
| 2031 | QH_NORM_A010 | QH读出 | 1.0 | 0.1 | 46 | 99001 | 5.00e-04→5.00e-05 | 256步完成 |
| 2032 | ADAMW_LR050 | native | 0.5 | 0.0 | 46 | 99002 | 2.50e-04→2.50e-05 | 256步完成 |
| 2032 | ADAMW_LR100 | native | 1.0 | 0.0 | 46 | 99002 | 5.00e-04→5.00e-05 | 256步完成 |
| 2032 | ADAMW_LR200 | native | 2.0 | 0.0 | 46 | 99002 | 1.00e-03→1.00e-04 | 256步完成 |
| 2032 | QH_RAW_A010 | QH读出 | 1.0 | 0.1 | 46 | 99002 | 5.00e-04→5.00e-05 | 256步完成 |
| 2032 | QH_NORM_A010 | QH读出 | 1.0 | 0.1 | 46 | 99002 | 5.00e-04→5.00e-05 | 256步完成 |

覆盖：15/15 轨迹 × 256 步；monitor h=0,16,…,256（h0 每 seed 共享一次评价）；schedule finite_cosine_then_hold_5e-5_v01；LR 倍率整段固定；无 re-warmup/无 step/moment reset/无新 cooldown。

## 表 B：h0/64/128/256 NLL（全库 sum/count/mean）与配对差

mean = sum(valid-target NLL)/count(valid targets)，4096 行全库聚合。

| seed | arm | h | sum | count | mean |
|---|---|---|---|---|---|
| 2030 | ADAMW_LR050 | 0 | 3464479.6 | 475330 | 7.28858 |
| 2030 | ADAMW_LR050 | 64 | 3400346.8 | 475330 | 7.15365 |
| 2030 | ADAMW_LR050 | 128 | 3382022.6 | 475330 | 7.11510 |
| 2030 | ADAMW_LR050 | 256 | 3349487.9 | 475330 | 7.04666 |
| 2030 | ADAMW_LR100 | 0 | 3464479.6 | 475330 | 7.28858 |
| 2030 | ADAMW_LR100 | 64 | 3366719.4 | 475330 | 7.08291 |
| 2030 | ADAMW_LR100 | 128 | 3336015.1 | 475330 | 7.01831 |
| 2030 | ADAMW_LR100 | 256 | 3281727.4 | 475330 | 6.90410 |
| 2030 | ADAMW_LR200 | 0 | 3464479.6 | 475330 | 7.28858 |
| 2030 | ADAMW_LR200 | 64 | 3323605.5 | 475330 | 6.99221 |
| 2030 | ADAMW_LR200 | 128 | 3274467.9 | 475330 | 6.88883 |
| 2030 | ADAMW_LR200 | 256 | 3185075.5 | 475330 | 6.70077 |
| 2030 | QH_RAW_A010 | 0 | 3464479.6 | 475330 | 7.28858 |
| 2030 | QH_RAW_A010 | 64 | 3362980.6 | 475330 | 7.07504 |
| 2030 | QH_RAW_A010 | 128 | 3332004.6 | 475330 | 7.00988 |
| 2030 | QH_RAW_A010 | 256 | 3277473.9 | 475330 | 6.89515 |
| 2030 | QH_NORM_A010 | 0 | 3464479.6 | 475330 | 7.28858 |
| 2030 | QH_NORM_A010 | 64 | 3368129.4 | 475330 | 7.08588 |
| 2030 | QH_NORM_A010 | 128 | 3338992.7 | 475330 | 7.02458 |
| 2030 | QH_NORM_A010 | 256 | 3288076.4 | 475330 | 6.91746 |
| 2031 | ADAMW_LR050 | 0 | 3460973.1 | 475330 | 7.28120 |
| 2031 | ADAMW_LR050 | 64 | 3401152.0 | 475330 | 7.15535 |
| 2031 | ADAMW_LR050 | 128 | 3380705.1 | 475330 | 7.11233 |
| 2031 | ADAMW_LR050 | 256 | 3348201.7 | 475330 | 7.04395 |
| 2031 | ADAMW_LR100 | 0 | 3460973.1 | 475330 | 7.28120 |
| 2031 | ADAMW_LR100 | 64 | 3369735.4 | 475330 | 7.08925 |
| 2031 | ADAMW_LR100 | 128 | 3335656.7 | 475330 | 7.01756 |
| 2031 | ADAMW_LR100 | 256 | 3277283.6 | 475330 | 6.89475 |
| 2031 | ADAMW_LR200 | 0 | 3460973.1 | 475330 | 7.28120 |
| 2031 | ADAMW_LR200 | 64 | 3330567.2 | 475330 | 7.00685 |
| 2031 | ADAMW_LR200 | 128 | 3272059.1 | 475330 | 6.88376 |
| 2031 | ADAMW_LR200 | 256 | 3179203.9 | 475330 | 6.68841 |
| 2031 | QH_RAW_A010 | 0 | 3460973.1 | 475330 | 7.28120 |
| 2031 | QH_RAW_A010 | 64 | 3359441.3 | 475330 | 7.06760 |
| 2031 | QH_RAW_A010 | 128 | 3324560.6 | 475330 | 6.99422 |
| 2031 | QH_RAW_A010 | 256 | 3265835.1 | 475330 | 6.87067 |
| 2031 | QH_NORM_A010 | 0 | 3460973.1 | 475330 | 7.28120 |
| 2031 | QH_NORM_A010 | 64 | 3364793.4 | 475330 | 7.07886 |
| 2031 | QH_NORM_A010 | 128 | 3331995.0 | 475330 | 7.00986 |
| 2031 | QH_NORM_A010 | 256 | 3277681.2 | 475330 | 6.89559 |
| 2032 | ADAMW_LR050 | 0 | 3474745.5 | 475330 | 7.31018 |
| 2032 | ADAMW_LR050 | 64 | 3420707.1 | 475330 | 7.19649 |
| 2032 | ADAMW_LR050 | 128 | 3401280.3 | 475330 | 7.15562 |
| 2032 | ADAMW_LR050 | 256 | 3369297.5 | 475330 | 7.08833 |
| 2032 | ADAMW_LR100 | 0 | 3474745.5 | 475330 | 7.31018 |
| 2032 | ADAMW_LR100 | 64 | 3392684.2 | 475330 | 7.13753 |
| 2032 | ADAMW_LR100 | 128 | 3361202.6 | 475330 | 7.07130 |
| 2032 | ADAMW_LR100 | 256 | 3305197.1 | 475330 | 6.95348 |
| 2032 | ADAMW_LR200 | 0 | 3474745.5 | 475330 | 7.31018 |
| 2032 | ADAMW_LR200 | 64 | 3354346.1 | 475330 | 7.05688 |
| 2032 | ADAMW_LR200 | 128 | 3299902.5 | 475330 | 6.94234 |
| 2032 | ADAMW_LR200 | 256 | 3213784.0 | 475330 | 6.76116 |
| 2032 | QH_RAW_A010 | 0 | 3474745.5 | 475330 | 7.31018 |
| 2032 | QH_RAW_A010 | 64 | 3384998.4 | 475330 | 7.12136 |
| 2032 | QH_RAW_A010 | 128 | 3351277.2 | 475330 | 7.05042 |
| 2032 | QH_RAW_A010 | 256 | 3297935.6 | 475330 | 6.93820 |
| 2032 | QH_NORM_A010 | 0 | 3474745.5 | 475330 | 7.31018 |
| 2032 | QH_NORM_A010 | 64 | 3392666.6 | 475330 | 7.13750 |
| 2032 | QH_NORM_A010 | 128 | 3361734.9 | 475330 | 7.07242 |
| 2032 | QH_NORM_A010 | 256 | 3311907.5 | 475330 | 6.96760 |

h256 配对差（BLOCK_PROXY：64 行连续 block，1000 reps，seed 77070001；delta = Σ(num_diff)/Σcount；数据级 SE/CI，不是训练 seed SE）：

| comparison | seed | Δmean@h256 | SE | 95%CI | p_boot |
|---|---|---|---|---|---|
| QH_NORM_A010_MINUS_ADAMW_LR100 | 2030 | +0.01336 | 0.00016 | [+0.01306, +0.01369] | 0.001 |
| QH_NORM_A010_MINUS_ADAMW_LR100 | 2031 | +0.00084 | 0.00028 | [+0.00028, +0.00137] | 0.004 |
| QH_NORM_A010_MINUS_ADAMW_LR100 | 2032 | +0.01412 | 0.00019 | [+0.01373, +0.01449] | 0.001 |
| QH_RAW_A010_MINUS_ADAMW_LR100 | 2030 | -0.00895 | 0.00012 | [-0.00920, -0.00870] | 1.000 |
| QH_RAW_A010_MINUS_ADAMW_LR100 | 2031 | -0.02409 | 0.00031 | [-0.02466, -0.02348] | 1.000 |
| QH_RAW_A010_MINUS_ADAMW_LR100 | 2032 | -0.01528 | 0.00033 | [-0.01593, -0.01463] | 1.000 |
| QH_NORM_A010_MINUS_QH_RAW_A010 | 2030 | +0.02231 | 0.00017 | [+0.02197, +0.02265] | 0.001 |
| QH_NORM_A010_MINUS_QH_RAW_A010 | 2031 | +0.02492 | 0.00020 | [+0.02453, +0.02533] | 0.001 |
| QH_NORM_A010_MINUS_QH_RAW_A010 | 2032 | +0.02939 | 0.00029 | [+0.02884, +0.02995] | 0.001 |
| QH_NORM_A010_MINUS_ADAMW_LR050 | 2030 | -0.12920 | 0.00101 | [-0.13127, -0.12722] | 1.000 |
| QH_NORM_A010_MINUS_ADAMW_LR050 | 2031 | -0.14836 | 0.00081 | [-0.14991, -0.14680] | 1.000 |
| QH_NORM_A010_MINUS_ADAMW_LR050 | 2032 | -0.12074 | 0.00070 | [-0.12204, -0.11929] | 1.000 |
| QH_NORM_A010_MINUS_ADAMW_LR200 | 2030 | +0.21669 | 0.00127 | [+0.21423, +0.21912] | 0.001 |
| QH_NORM_A010_MINUS_ADAMW_LR200 | 2031 | +0.20718 | 0.00118 | [+0.20500, +0.20957] | 0.001 |
| QH_NORM_A010_MINUS_ADAMW_LR200 | 2032 | +0.20643 | 0.00130 | [+0.20387, +0.20893] | 0.001 |
| ADAMW_LR050_MINUS_ADAMW_LR100 | 2030 | +0.14255 | 0.00112 | [+0.14039, +0.14477] | 0.001 |
| ADAMW_LR050_MINUS_ADAMW_LR100 | 2031 | +0.14920 | 0.00099 | [+0.14726, +0.15101] | 0.001 |
| ADAMW_LR050_MINUS_ADAMW_LR100 | 2032 | +0.13485 | 0.00080 | [+0.13330, +0.13641] | 0.001 |
| ADAMW_LR200_MINUS_ADAMW_LR100 | 2030 | -0.20334 | 0.00112 | [-0.20554, -0.20118] | 1.000 |
| ADAMW_LR200_MINUS_ADAMW_LR100 | 2031 | -0.20634 | 0.00111 | [-0.20846, -0.20415] | 1.000 |
| ADAMW_LR200_MINUS_ADAMW_LR100 | 2032 | -0.19232 | 0.00126 | [-0.19472, -0.18974] | 1.000 |

## 表 C：目标检测（离散 16 步监测，连续两次 ≤ ell）

ell_star = 7.020043501538（Phase A start46 全精度校准文件读取，未重新估计）。

| seed | arm | reached | h_detect | N_detect | last_above | 1st_in_pair | 2nd_conf | terminal NLL |
|---|---|---|---|---|---|---|---|---|
| 2030 | ADAMW_LR050 | NOT_REACHED | None | None | 256 | None | None | 7.04666 |
| 2030 | ADAMW_LR100 | Y | 144 | 147456 | 112 | 128 | 144 | 6.90410 |
| 2030 | ADAMW_LR200 | Y | 80 | 81920 | 48 | 64 | 80 | 6.70077 |
| 2030 | QH_RAW_A010 | Y | 144 | 147456 | 112 | 128 | 144 | 6.89515 |
| 2030 | QH_NORM_A010 | Y | 160 | 163840 | 128 | 144 | 160 | 6.91746 |
| 2031 | ADAMW_LR050 | NOT_REACHED | None | None | 256 | None | None | 7.04395 |
| 2031 | ADAMW_LR100 | Y | 144 | 147456 | 112 | 128 | 144 | 6.89475 |
| 2031 | ADAMW_LR200 | Y | 80 | 81920 | 48 | 64 | 80 | 6.68841 |
| 2031 | QH_RAW_A010 | Y | 128 | 131072 | 96 | 112 | 128 | 6.87067 |
| 2031 | QH_NORM_A010 | Y | 144 | 147456 | 112 | 128 | 144 | 6.89559 |
| 2032 | ADAMW_LR050 | NOT_REACHED | None | None | 256 | None | None | 7.08833 |
| 2032 | ADAMW_LR100 | Y | 208 | 212992 | 176 | 192 | 208 | 6.95348 |
| 2032 | ADAMW_LR200 | Y | 112 | 114688 | 80 | 96 | 112 | 6.76116 |
| 2032 | QH_RAW_A010 | Y | 192 | 196608 | 160 | 176 | 192 | 6.93820 |
| 2032 | QH_NORM_A010 | Y | 208 | 212992 | 176 | 192 | 208 | 6.96760 |

说明：h_detect 是离散监测下的确认时点，不是精确连续首次穿越；不假设相邻检查点间单调；本轮 16 步网格不同于旧 32 步网格，不拿旧 hit 声称提速。未达标/失败/检测间隔全部保留。

## 表 D：arm 级达标与固定预算表现

| arm | reached/3 | not_reached | aborted | h256 means (per seed) | avg |
|---|---|---|---|---|---|
| ADAMW_LR050 | 0 | 3 | 0 | 7.04666/7.04395/7.08833 | 7.05965 |
| ADAMW_LR100 | 3 | 0 | 0 | 6.90410/6.89475/6.95348 | 6.91745 |
| ADAMW_LR200 | 3 | 0 | 0 | 6.70077/6.68841/6.76116 | 6.71678 |
| QH_RAW_A010 | 3 | 0 | 0 | 6.89515/6.87067/6.93820 | 6.90134 |
| QH_NORM_A010 | 3 | 0 | 0 | 6.91746/6.89559/6.96760 | 6.92688 |

paired token ratio（双方达标的 seed 才配对；eligible pairs 单列，非全样本无偏平均）：
- QH_NORM_A010_vs_ADAMW_LR100: eligible=3/3, ratios: seed2030:1.111, seed2031:1.000, seed2032:1.000
- QH_NORM_A010_vs_ADAMW_LR050: eligible=0/3, ratios: 无配对
- QH_NORM_A010_vs_ADAMW_LR200: eligible=3/3, ratios: seed2030:2.000, seed2031:1.800, seed2032:1.857
- QH_NORM_A010_vs_QH_RAW_A010: eligible=3/3, ratios: seed2030:1.111, seed2031:1.125, seed2032:1.083

截尾统计 min(N_detect, 262144)（未达标按预算填，仅用于该截尾量，不是达标时间）：
- ADAMW_LR050: mean=262144 slots, values=[262144, 262144, 262144]
- ADAMW_LR100: mean=169301 slots, values=[147456, 147456, 212992]
- ADAMW_LR200: mean=92843 slots, values=[81920, 81920, 114688]
- QH_RAW_A010: mean=158379 slots, values=[147456, 131072, 196608]
- QH_NORM_A010: mean=174763 slots, values=[163840, 147456, 212992]

三种固定 LR 全列（粗网格，非充分调优）：ADAMW_LR050 / ADAMW_LR100 / ADAMW_LR200；每 seed 网格最优：{2030: 'ADAMW_LR200', 2031: 'ADAMW_LR200', 2032: 'ADAMW_LR200'}。边界最优一律标 BASELINE_GRID_BOUNDARY。

LOSO companion（描述性，小样本，不当独立确认）：
- held-out 2030: 他 seed 选 LR2.0 → h256 6.70077 vs native 6.90410
- held-out 2031: 他 seed 选 LR2.0 → h256 6.68841 vs native 6.89475
- held-out 2032: 他 seed 选 LR2.0 → h256 6.76116 vs native 6.95348

## 表 E：同状态读出诊断（h ∈ {1,16,64,128,256}，逐唯一参数张量）

- 范数匹配最大绝对残差 | ||r_N||−||r_A|| |：3.052e-05
- 范数匹配最大相对残差：1.659e-07
- 方向 cos(r_A, r_N) 最小值（NORM 臂）：0.819258
- 方向 cos(r_A, r_Q) 最小值（RAW 臂参考）：0.819258
- fallback 计数（NORM 臂全轨迹诊断点）：{'NONE': 690}
- RAW vs NORM 的 c^T(r_N−r_A) 平均：1.859e+00 （current-training proxy，不是 monitor 贡献）
- 每诊断点每参数临时张量 fp32 字节合计（7 份临时量）：2092984320（诊断点合计；逐步同量，无持久 state）
- 新增全维持久 state bytes：0
- 定义：cos(r_A, r_N) = dot/(||r_A||*||r_N||)，同状态参考；范数均为 Frobenius norm（sqrt sum sq），非 mean(norm)
- 警示：NORM 轨迹的 r_A 是本状态 native-formula 参考，不是另一条真实 native 轨迹的 r_A；禁止跨轨迹误称范数被严格控制相同。

## 表 F：actual / planned / cap

| 项目 | actual | planned | cap |
|---|---|---|---|
| 总真实 updates（含失败/重试） | 3904 | 3880 | 4096 |
| 训练 forward microcalls | 7808 | 7760 | （随总量） |
| 训练 backward microcalls | 7808 | 7760 | 8192 |
| monitor 整库 eval | 253 | 249 | 270 |
| monitor forward microcalls | 32000 | 31872 | 34560 |
| 全部 model forward | 39808 | 39632 | 45000 |
| 额外梯度/HVP/FD | 0/0/0 | 0/0/0 | 0/0/0 |
| 候选重放用于选择 | 0 | 0 | 0 |
| V_final 模型调用 | 0 | 0 | 0 |
| 训练 input slots（全部臂合计） | 3932160 | 3932160 | 按真实 batch |
| monitor input slots | 133693440 | 130547712 | 按实际计 |
| 失败 / 数值失败 | 0 / 0 | 0 | — |
| 墙钟 | OBSERVATIONAL_SHARED_GPU（见 resource_log.jsonl） | — | 无科学硬上限 |

within_caps = True；smoke 40 updates 已计入；失败/smoke/恢复全部计费。

## 机制迹象（有限解释，与算法收益分开）

- 同状态范数匹配在诊断点成立（残差见表 E）：它限制的是当前同状态数据步幅，不是总参数位移（还含 decay），更不是两条长期轨迹步长始终相等。
- current-batch 一阶对齐 c^T r 不是独立任务收益保证（CPU 测试 T5 反例）；本轮只把它作 current-training proxy 记录。
- 若 NORM 与 RAW 差异小，说明在本配方/尺度下范数匹配改变有限；若 RAW 同样或更好，不声称范数匹配是收益机制。
- 本轮 monitor 密度（16 步 × 全库）是研究测量成本，不代表部署频率。

## 失败 / 未达标 / 不确定

- 数值失败：0；其他失败：0。
- ADAMW_LR050: 未达标 3/3。
- ADAMW_LR100: 未达标 0/3。
- ADAMW_LR200: 未达标 0/3。
- QH_RAW_A010: 未达标 0/3。
- QH_NORM_A010: 未达标 0/3。
- n=3 seeds，逐 seed 结果为准；不做夸张总体显著性声明；bootstrap 是数据级 SE，不是训练 seed SE。
- 本轮 V_final=0：final 质量未独立确认；N_detect 是开发目标下的新流筛查证据。

## 摘要结论（协议 §10 五问）

1. 直接规则是否改善真实目标/检测训练 token：FINAL_BUDGET_NLL_EFFECT=NO_OBSERVED_GAIN，DETECTED_TARGET_TOKEN_EFFECT=NO_OBSERVED_GAIN。
2. 是否超过所测固定 LR 网格及 QH_RAW：FIXED_LR_COMPARISON=BASELINE_GRID_BOUNDARY，NORM_MATCHING_ABLATION=RAW_BETTER_THAN_NORM。
3. 改动/尺度控制/噪声代价的可见迹象：见表 E 与机制节。
4. 失败/未达标/不确定：数值失败 0；未达标情况见表 D。
5. 后续唯一建议见下（不自动执行）。

## 后续（唯一建议，不自动执行）

若本轮存在开发信号（SCREENING_SIGNAL/MIXED），下一轮唯一建议：在另一起点（start82 或新 bank）+ 新的独立训练 seeds/streams + 预先规定的有限 LR 匹配调参预算下复验，并事先规定 final 评价协议；若无收益，记录该具体规则与配置（alpha=0.10、QH 读出 + 逐张量范数匹配）的负结果并停止该候选，不堆叠补偿项。

## 产物清单

- outputs/fr_gttr_d1_direct_readout_v01/config.json
- outputs/fr_gttr_d1_direct_readout_v01/manifest.json
- outputs/fr_gttr_d1_direct_readout_v01/implementation_checks.json
- outputs/fr_gttr_d1_direct_readout_v01/smoke_checks.json
- outputs/fr_gttr_d1_direct_readout_v01/step_metrics.csv
- outputs/fr_gttr_d1_direct_readout_v01/monitor_metrics.csv
- outputs/fr_gttr_d1_direct_readout_v01/monitor_rows/*.npz
- outputs/fr_gttr_d1_direct_readout_v01/target_hits.csv
- outputs/fr_gttr_d1_direct_readout_v01/seed_method_results.csv
- outputs/fr_gttr_d1_direct_readout_v01/paired_comparisons.csv
- outputs/fr_gttr_d1_direct_readout_v01/readout_diagnostics.csv
- outputs/fr_gttr_d1_direct_readout_v01/resource_log.jsonl
- outputs/fr_gttr_d1_direct_readout_v01/SNAPSHOT_MANIFEST.json
- outputs/fr_gttr_d1_direct_readout_v01/SUMMARY.json
- outputs/fr_gttr_d1_direct_readout_v01/REPORT.md
- outputs/fr_gttr_d1_direct_readout_v01/KEY_DATA_FOR_REVIEW.md
- outputs/fr_gttr_d1_direct_readout_v01/figures/fig1_nll_vs_tokens.png
- outputs/fr_gttr_d1_direct_readout_v01/figures/fig2_h256_gain.png
- outputs/fr_gttr_d1_direct_readout_v01/figures/fig3_detection.png
- outputs/fr_gttr_d1_direct_readout_v01/figures/fig4_readout_diag.png
