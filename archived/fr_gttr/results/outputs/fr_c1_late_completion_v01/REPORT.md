# REPORT — FR_C1_LATE_COMPLETION_V01（C1 遗留 late anchors + 旧数据离线复核）

日期：2026-09-21。预算独立建账；旧 17,994/18,000 与旧 C1（45/605/304/1801）不动。
本轮合计 106/279/144/835
（updates/grads/HVPs/micros），within_caps=True。

## 执行摘要

1. **旧数据离线复核（O1）**：4 个旧 anchor 的逐 anchor/channel/K8/K16/K32/first16/last16
   SSE/TSS/E 由本仓库小参考计算独立重算，与已发布 aggregate_E.csv 完全一致
   （old_aggregate_recheck match 全 True）；resolved 标志与 probe_summary.csv 一致（0 不一致）。
   C0X erratum 维持：聚合门误用不追认为完整 PASS。
2. **两个 late anchor 执行完成**：R/A/t48、C/A/t48（native A 臂，更新前 local t=48，
   全重放 49 updates/cell，标量级恒等 diff=0.000e+00）。泄漏顺序、冻结 K=8/16/32、
   fixed halves、先存预测后读 audit、逐 probe U、fixed q+、stop-v 仅诊断，全部按协议。
3. **late 判定（scope 标注，非统一 headline）**：四个 anchor×channel 全部
   LATE_ANCHOR_RESOLVED_AND_IMPROVED（E_K32 0.222–0.248，halves ≤0.336，全部 resolved）。
4. **COMBINED6（固定 3-anchor 集合 A/t0+Bp/t16+A/t48）**：
   - field：R PASS（E=0.491，3 distinct t），C PASS（E=0.362，3 distinct t）。
   - shadow：R/C 数值门全过（E=0.665/0.571，halves ≤0.954），distinct-t=2（{16,48}）→ 判 PASS。
   - **两条纪律性保留**：(a) late 的通过不回溯改变 t0 的 NUMERICAL_UNRESOLVED 事实；
     (b) shadow 的 {16,48} 可分辨不是独立总体确认（同一 R/C 对、相邻 t、同批探针库）。

## 门追踪总表

| scope | cell | ch | status | E_K32 | E_first16 | E_last16 | n_t | distinct_t |
|---|---|---|---|---|---|---|---|---|
| OLD4 | R | field | PASS | 0.5049 | 0.6995 | 0.4968 | 2 | [0, 16] |
| OLD4 | R | shadow | NUMERICAL_UNRESOLVED | 0.6673 | 0.8725 | 0.6321 | 1 | [16] |
| OLD4 | C | field | PASS | 0.3982 | 0.3856 | 0.6936 | 2 | [0, 16] |
| OLD4 | C | shadow | NUMERICAL_UNRESOLVED | 0.5848 | 0.6035 | 0.9799 | 1 | [16] |
| LATE2 | R | field | NUMERICAL_UNRESOLVED | 0.2364 | 0.3195 | 0.3171 | 1 | [48] |
| LATE2 | R | shadow | NUMERICAL_UNRESOLVED | 0.2478 | 0.3139 | 0.3364 | 1 | [48] |
| LATE2 | C | field | NUMERICAL_UNRESOLVED | 0.2382 | 0.3277 | 0.3221 | 1 | [48] |
| LATE2 | C | shadow | NUMERICAL_UNRESOLVED | 0.2224 | 0.3097 | 0.2991 | 1 | [48] |
| COMBINED6 | R | field | PASS | 0.4914 | 0.6811 | 0.4865 | 3 | [0, 16, 48] |
| COMBINED6 | R | shadow | PASS | 0.6646 | 0.8690 | 0.6299 | 2 | [16, 48] |
| COMBINED6 | C | field | PASS | 0.3618 | 0.3711 | 0.6138 | 3 | [0, 16, 48] |
| COMBINED6 | C | shadow | PASS | 0.5706 | 0.5923 | 0.9535 | 2 | [16, 48] |

## 协议五问回答

### ① 数学/实现身份与经验支持分别是什么？

**数学恒等式（精确成立，与数据无关）**：e = p_D − y_A，e_bank = p_D − p_A，
e_local = p_A − y_A；‖e‖² = ‖e_bank‖² + ‖e_local‖² + 2⟨e_bank, e_local⟩。
late 两个 anchor 的恒等残差（max identity_residual over 4 anchor×channel）=
5.42101e-20（fp64 舍入量级，见
late_error_decomposition.csv）。
bank/local 分解以**带符号 cross 项**报告（不用 norm 百分比冒充可加贡献）。

**实现身份**：leakage order（预测 order_index 先于 audit）、冻结预测与实测
group_stats 逐位一致（diff=0.000e+00）、标量级状态恒等（v_pre/q_pre_norm/s_norm/eta_t）。

**经验支持（本轮新测）**：late E_K32 = 0.236/0.238（field R/C）、0.248/0.222（shadow R/C），
均 ≤0.8 且 halves ≤1；target 全部 resolved（signal ratio 1.94–2.07）。
COMBINED6 下 field 达 3 distinct t，shadow 达 2 distinct t（{16,48}）。

### ② field 的近似缺陷和跨库差在 late 点怎样变化？

- 分解占比（programmatic, 4 anchor×channel 行）：
  - R/t48 field: bank_SS/total_SS = 0.83731, local_SS/total_SS = 0.113462, |signed_cross|/total_SS = 0.049228
  - R/t48 shadow: bank_SS/total_SS = 0.762721, local_SS/total_SS = 0.16201, |signed_cross|/total_SS = 0.0752688
  - C/t48 field: bank_SS/total_SS = 0.972869, local_SS/total_SS = 0.0533084, |signed_cross|/total_SS = 0.026177
  - C/t48 shadow: bank_SS/total_SS = 0.850002, local_SS/total_SS = 0.0879068, |signed_cross|/total_SS = 0.0620908

- 跨库 bank 分量在 late 点仍主导（见上表），cross 项带符号、percent 级，完整报告（非 norm 百分比归因）。
- field 同 probe 配对残差（audit, mean±SE of ||p_j−y_j||）：
  - R: 0.00851219 ± 0.000192965
  - C: 0.010173 ± 0.000279215
- E 随 t 的比较（field, K32，R/C）：t0 = R 0.2799 / C 0.2191；t16 = R 0.5203 / C 0.4457；t48 = R 0.2364 / C 0.2382；无 E 随 t 增长的证据，t48（A0U 延迟获益区）E 保持低位。

### ③ shadow 分辨性是否与早期不同，原六点覆盖门能否复算并通过？

- 不同：t0 shadow target 不可分辨（ratio<1，NUMERICAL_UNRESOLVED）是**早期事实**；
  t16/t48 均可分辨（ratio≥1.94）。shadow 的可分辨性集中在 t≥16。
- 原六点覆盖门：432 HVP > 416 cap 的复算维持（C1 已披露），本轮不重修；
  覆盖以 OLD4+LATE2=6/6 聚合补足，并在 COMBINED6 固定集合下复算通过（见总表）。
- 旧 shadow 每 probe 残差 SE 缺失（NOT_SAVED），probe_summary K32 shadow 残差列为
  field 值伪影，已列入 old_data_missing.csv，不作 shadow 推断依据。

### ④ 动态 v 反馈有何已测效应，哪些读数缺统计支持？

已测（stop-v 诊断，未写回 optimizer）：
- Z audit norm ratio = 1.222（R）/ 1.236（C），mean cosine 0.875/ 0.867
  （冻结 P 的 Zstop 与 full-Z 差为配对正差，paired-diff SE 见表 C）；
- Q dev norm ratio = 1.303（R）/ 1.290（C）；
- 解释为：v 通道携带除 P 一阶路径外的增量信息，与 C1 一致。
- d_c_to_u 在 q+ 方向投影：R -0.0637838、
  C -0.0887293（负值=沿 q+ 的下降步）。

缺统计支持：q+ 投影的若干分量 |mean| ≤ 2·SE（如 C 的 Q_D/Z_A 投影），
按预注册规则不计显著读数（表 D pred_SE/tgt_SE 列）。stop-v 仅 norm 级诊断，
未对单参数分量做推断。

### ⑤ 哪些量仍缺失，哪些结论仍 NOT_TESTED？

缺失（O3，见 old_data_missing.csv）：C0 g3 半库聚合（DATA_MISSING，历史缺口）；
旧 shadow 每 probe 残差 SE；旧 stop-v 每 probe norm/cosine；A0T−v_eval 共轴；
旧记录每 group sumsq/centered_SS（本轮 late 已补存充分统计）。

NOT_TESTED（维持）：未来 H16/H64 提前预测；adaptive 训练价值；最终训练改进；
能耗收益。禁止外推：不得称"唯一原因已找到"、不得称"已解释 H64 反转"、
不得据本轮结果改 optimizer。

## 纪律声明

- 不混轴（V_diag/V_eval/A0T-v_score），不减常数对齐。
- first16 ≡ K16 前缀（同 probe ID，披露，非独立证据）。
- 无 t48 snapshot → 全重放 98 updates；标量级恒等 ≠ 完整 theta/m/v/RNG 逐位一致。
- 失败/smoke/复核全部入账；三笔账分开（17,994 | 45/605/304/1801 | 106/279/144/835）。
- 术语禁律遵守：本轮无禁词。

---

## ERRATA 追加（2026-09-21，0 GPU 字段核对；原表数值保留不改）

核对脚本 `research/scripts/c1_late_fieldcheck_v01.py`；全文见 `FIELD_CORRECTIONS.md`，
复算表 `fieldcheck_task{1..5}_*.csv` + `fieldcheck_summary.json`。

- **E1/E2**：Table A target/pred 的 norm 列 = norm(mean)（‖x̄‖），SE 列 = SE_vec；
  恒等 ‖x̄‖² + centered_SS/n = mean_sample_norm² 在 48/48 行精确成立（max diff 0.0）。
- **E3/E4**：resid_SE 实为 **SE_of_norm**（per-probe 库内残差范数列 SE，非 SE_vec），
  与 mean(norm)（R 8.512e-3 / C 1.017e-2）同列自洽；Jensen 间隙 +4.62e-3 / +6.77e-3 ≥0。
  target/pred 的 mean(norm)/SE_of_norm = NOT_SAVED（不补齐）。
- **E5/E6**：Table C full/stop 属 shadow；field 行撤下。ratio 拆为
  mean_of_probe_ratios（1.2222/1.2363，原报数值）与 ratio_of_mean_norms（1.0486/1.0432，新增）。
- **E7**：Table D "full_minus_stop" 实为 **stop_mean**；重建列
  full_mean / stop_mean / paired_diff_mean / paired_diff_SE（恒等差 ≤3.3e-20）。
- **E8**：SNR 拆为 signal_over_SE（5.82–6.22）与 ratio_to_3SE（1.94–2.07）；
  resolved 全部复核 True。
- **判定**：E/SSE/TSS、resolved、distinct-t、COMBINED6 十二行 E、门布尔、预算——**全部不变**；
  本次为标签/定义级勘误，无数值判定反转。NOT_TESTED 维持；停止条件维持。
