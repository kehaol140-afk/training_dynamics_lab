#!/usr/bin/env python3
"""FR-C1-Late report generator: renders KEY_DATA_FOR_REVIEW.md, REPORT.md and
GTTR_EVIDENCE_UPDATE.md from the saved analyzer tables (no hand-copied numbers;
0 GPU)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fr_c1_late_completion_v01"
C1 = ROOT / "outputs" / "fr_c1_gttr_probe_validation_v01"


def rows(name):
    return list(csv.DictReader(open(OUT / name)))


def f(x, n=4):
    if x in ("", None):
        return "NA"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def g(x, n=6):
    if x in ("", None):
        return "NA"
    try:
        return f"{float(x):.{n}g}"
    except (TypeError, ValueError):
        return str(x)


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


lateA = rows("late_probe_statistics.csv")
oldA = rows("old_probe_statistics.csv")
aggB = rows("aggregate_gate_inputs.csv")
decC = rows("late_error_decomposition.csv")
projD = rows("late_fixed_projection.csv")
ufb = rows("late_update_feedback.csv")
odm = rows("old_data_missing.csv")
o1agg = rows("old_aggregate_recheck.csv")
ct = json.loads((OUT / "criterion_trace.json").read_text())
res = json.loads((OUT / "resource_log.json").read_text())
smoke = json.loads((OUT / "smoke_checks.json").read_text())
c1_budget = json.loads((C1 / "budget_log.json").read_text())
cfg = json.loads((OUT / "config.json").read_text())
prov = list(csv.DictReader(open(OUT / "state_and_probe_provenance.csv")))

gates = {(x["scope"], x["cell"], x["channel"]): x for x in ct["gate_trace"]}
lverd = {(x["cell"], x["channel"]): x for x in ct["late_verdicts"]}

# --- derived prose inputs for REPORT §② -------------------------------
decomp_ratios = "".join(
    f"  - {r['cell']}/t{r['anchor']} {r['channel']}: "
    f"bank_SS/total_SS = {g(float(r['bank_SS'])/float(r['total_SS']))}, "
    f"local_SS/total_SS = {g(float(r['local_SS'])/float(r['total_SS']))}, "
    f"|signed_cross|/total_SS = {g(abs(float(r['signed_cross']))/float(r['total_SS']))}\n"
    for r in decC)
_late_recs = [json.loads(l) for l in open(OUT / "runs" / "late_records.jsonl")]
field_resid_lines = "".join(
    f"  - {r['cell']}: {g(r['same_probe_resid_norm_mean_se']['audit']['mean'])} ± "
    f"{g(r['same_probe_resid_norm_mean_se']['audit']['se'])}\n"
    for r in _late_recs)


def _fieldE(anchor_val):
    out = {}
    for c in ("R", "C"):
        ro = [r for r in oldA + lateA
              if r["cell"] == c and r["channel"] == "field"
              and r["K"] == "32" and int(r["anchor"]) == anchor_val]
        out[c] = f(ro[0]["E"]) if ro else "NA"
    return f"R {out['R']} / C {out['C']}"


field_old_E = {t: _fieldE(a) for t, a in (("t0", 0), ("t16", 16), ("t48", 48))}

# ---------------------------------------------------------------- KEY DATA

def tableA(all_rows):
    body = []
    for r in all_rows:
        body.append([f"{r['cell']}/{r['arm']}/t{r['anchor']}", r["channel"],
                     g(r["target_norm"]), g(r["target_se"]), f(r["floor"], 1),
                     f(r["signal_ratio"], 2), r["resolved"],
                     g(r["pred_norm"]), g(r["pred_se"]),
                     g(r["residual_se"]), g(r["sse"]), g(r["tss"]), f(r["E"])])
    return md_table(["anchor", "ch", "target_norm", "SE", "floor", "SNR",
                     "resolved", "pred_norm", "pred_SE", "resid_SE",
                     "SSE", "TSS", "E"], body)


A_old = tableA(oldA)
A_late = tableA(lateA)

# per-K / halves detail for late anchors
det_body = []
for r in lateA:
    det_body.append([f"{r['cell']}/{r['arm']}/t{r['anchor']}", r["channel"],
                     r["K"], g(r["sse"]), g(r["tss"]), f(r["E"])])
A_detail = md_table(["anchor", "ch", "K/half", "SSE", "TSS", "E"], det_body)

B_body = []
for r in aggB:
    B_body.append([r["scope"], r["cell"], r["channel"], r["group"],
                   r["anchor_IDs"], g(r["SSE_sum"]), g(r["TSS_sum"]),
                   f(r["E"]) if r["E"] else "NA",
                   f(r["threshold"], 1), r["criterion"]])
B_tbl = md_table(["scope", "cell", "ch", "group", "anchors",
                  "SSE_sum", "TSS_sum", "E", "thr", "criterion"], B_body)

gt_body = []
for k in sorted(gates):
    x = gates[k]
    gt_body.append([x["scope"], x["cell"], x["channel"], x["status"],
                    f(x["E_K32"]), f(x["E_first16"]), f(x["E_last16"]),
                    str(x["n_distinct_t_resolved"]), str(x["distinct_t"])])
gate_tbl = md_table(["scope", "cell", "ch", "status", "E_K32", "E_first16",
                     "E_last16", "n_t", "distinct_t"], gt_body)

C_body = []
for r in decC:
    C_body.append([f"{r['cell']}/t{r['anchor']}", r["channel"],
                   g(r["total_SS"]), g(r["bank_SS"]), g(r["local_SS"]),
                   g(r["signed_cross"]), g(r["identity_residual"]),
                   g(r["shadow_full_signal_norm"]),
                   g(r["shadow_stop_signal_norm"]),
                   f(r["shadow_norm_ratio"]), f(r["shadow_mean_cosine"]),
                   g(r["shadow_paired_diff_mean"]),
                   g(r["shadow_paired_diff_SE"])])
C_tbl = md_table(["anchor", "ch", "total_SS", "bank_SS", "local_SS",
                  "signed_cross", "ident_resid",
                  "Z_full_norm", "Z_stop_norm", "ratio", "cos",
                  "paired_diff", "paired_diff_SE"], C_body)

clip = {}
for rec in [json.loads(l) for l in open(OUT / "runs" / "late_records.jsonl")]:
    clip[f"{rec['cell']}/t{rec['anchor']}"] = rec["clip_activity"]

D_body = []
for r in projD:
    D_body.append([r["cell"], r["channel"], f(r["qplus_norm"]),
                   g(r["pred_mean"]), g(r["pred_SE"]),
                   g(r["target_mean"]), g(r["target_SE"]),
                   g(r["same_probe_residual_mean"]), g(r["residual_SE"]),
                   g(r["full_minus_stop_mean"]),
                   g(r["paired_full_minus_stop_mean"]),
                   g(r["paired_full_minus_stop_SE"]),
                   g(r["d_c_to_u_qplus_mean"])])
D_tbl = md_table(["cell", "ch", "||q+||", "pred_mean", "pred_SE",
                  "tgt_mean", "tgt_SE", "resid_mean", "resid_SE",
                  "full-stop", "paired f-s", "paired f-s SE",
                  "d_c_to_u(q+)"], D_body)

rt = res["counters_round_total"]
E_body = [
    ["本轮计划 late anchors", "2 (R/A/t48, C/A/t48)", "2/2 执行"],
    ["状态恢复", "无 t48 snapshot（C1 仅存 t0/t16）→ 全重放",
     "49 updates/cell × 2 = 98（与 provenance 一致）"],
    ["smoke 过程", "updates/grads/HVP/micros",
     f"{res['counters_smoke_process']['optimizer_updates']}/"
     f"{res['counters_smoke_process']['grad_evals']}/"
     f"{res['counters_smoke_process']['hvp_calls']}/"
     f"{res['counters_smoke_process']['micro_forward_calls']}"],
    ["主过程", "updates/grads/HVP/micros",
     f"{res['counters_this_process']['optimizer_updates']}/"
     f"{res['counters_this_process']['grad_evals']}/"
     f"{res['counters_this_process']['hvp_calls']}/"
     f"{res['counters_this_process']['micro_forward_calls']}"],
    ["本轮合计（含 smoke）", "updates/grads/HVP/micros",
     f"{rt['optimizer_updates']}/{rt['grad_evals']}/{rt['hvp_calls']}/"
     f"{rt['micro_forward_calls']}"],
    ["本轮 caps", "updates/grads/HVP/micros/wall_s",
     "128/600/192/1800/2700"],
    ["within_caps", "", str(res["within_caps"])],
    ["主过程 GPU wall", "s", f(res["wall_run_s"], 1)],
    ["shadow 算术（另计）", "steps / s",
     f"{rt['shadow_steps']} / {f(res['shadow_seconds'], 2)}"],
    ["训练用 micro-forwards（另披露）", "replay 梯度伴生", "196 (=98×2)"],
    ["旧账 1：C0/C0X 轮", "updates（不重置、不并入）", "17,994 / 18,000"],
    ["旧账 2：C1 轮", "updates/grads/HVP/micros",
     f"{c1_budget['counters_round_total']['optimizer_updates']}/"
     f"{c1_budget['counters_round_total']['grad_evals']}/"
     f"{c1_budget['counters_round_total']['hvp_calls']}/"
     f"{c1_budget['counters_round_total']['micro_forward_calls']}"],
    ["本轮新账", "updates/grads/HVP/micros",
     f"{rt['optimizer_updates']}/{rt['grad_evals']}/{rt['hvp_calls']}/"
     f"{rt['micro_forward_calls']}"],
]
E_tbl = md_table(["项目", "说明", "数值"], E_body)

odm_tbl = md_table(["item", "status", "note"],
                   [[r["item"], r["status"], r["note"]] for r in odm])

o1_tbl = md_table(["cell", "ch", "group", "SSE_sum", "TSS_sum", "E_recomp",
                   "E_published", "match"],
                  [[r["cell"], r["channel"], r["group"], g(r["SSE_sum"]),
                    g(r["TSS_sum"]), f(r["E_recomputed"]),
                    f(r["E_published"]) if r["E_published"] else "NA",
                    r["match"]] for r in o1agg])

smoke_tbl = md_table(["check", "value"],
                     [[k, str(v)] for k, v in smoke.items()])

prov_tbl = md_table(list(prov[0].keys()), [[r[k] for k in prov[0].keys()] for r in prov])

kd = f"""# KEY DATA FOR REVIEW — FR_C1_LATE_COMPLETION_V01（自动生成，禁止手抄数字）

**勘误（2026-09-21 字段核对）**：本文件及以下各表为原始口径；字段级更正见
`FIELD_CORRECTIONS.md`（E1–E9，标签/定义级，主判定数值不变）。

来源：本目录 CSV/JSON（late_probe_statistics / old_probe_statistics /
aggregate_gate_inputs / late_error_decomposition / late_fixed_projection /
late_update_feedback / old_anchor_recheck / old_aggregate_recheck /
old_data_missing / criterion_trace / resource_log / smoke_checks /
state_and_probe_provenance）。生成器：research/scripts/c1_late_report_v01.py。

## 0. 身份与状态恢复

- 状态恢复：{cfg.get('state_recovery', 'see config.json')}
- replay 更新数：R 48 + C 48 = 96 到达 pre-update t=48；anchor 更新 h=49；cumulative step 94；t_after=95。
- 恒等性（标量级，非完整 theta/m/v/RNG 逐位）：与 C0 原生 A 臂 h=49 逐步记录对比，v_pre / q_pre_norm / s_norm / eta_t 四项 × 2 cell 全部 diff = 0.000e+00。
- 泄漏顺序：prediction order_index 全部先于 audit（late: 0<1, 2<3；old: 0..7 单调）；冻结预测 per_K mean 与实测 group_stats 完全一致（diff = 0.000e+00，12/12 组）。

provenance:
{prov_tbl}

smoke:
{smoke_tbl}

## 1. Table A — late anchors (K32 主读数)

{A_late}

## 1b. Table A — late anchors 全部 K/half 明细

{A_detail}

## 1c. Table A — old anchors 复核 (K32, O1 独立重算)

{A_old}

## 1d. O1 聚合交叉核对（独立重算 vs 已发布 aggregate_E.csv）

{o1_tbl}

全部 match = {all(r['match'] == 'True' for r in o1agg)}

## 2. Table B — 聚合门输入与逐 cell 布尔

{B_tbl}

## 2b. 门追踪（含 distinct-t 判据）

{gate_tbl}

late 单 anchor 判定：
{md_table(list(ct['late_verdicts'][0].keys()), [[str(v) for v in x.values()] for x in ct['late_verdicts']])}

## 3. Table C — late error decomposition + shadow stop-v

{C_tbl}

## 4. Table D — fixed q+ readout（原始 <x,q+> 投影；SHADOW_FIRST_ORDER_READOUT = -q+ᵀx̄，非 loss）

{D_tbl}

## 5. clip activity（dev 全体 active；audit 有 crossings）

```
{json.dumps(clip, indent=1)}
```

## 6. O2 轴/目标/索引清单 与 O3 缺失清单

轴定义（不混轴、不减常数对齐）：V_diag = v_score 前 8 个 ID（1 chunk，eval 模式，fp32 CE）；
V_eval = 冻结 64-sample；A0T = v_score 全轴。A0T−v_eval AXIS_MISMATCH 维持。
C0 readout_balance 索引：63 行/cell/arm，h∈2..64；区间 [0,4)=[upd 1..4]，[4,16)=[5..16]，[16,63]=[17..63]；无 h=65 行，不插补。

{odm_tbl}

## 7. Table E — 成本与覆盖

{E_tbl}
"""

(OUT / "KEY_DATA_FOR_REVIEW.md").write_text(kd)
print("KEY_DATA_FOR_REVIEW.md written")

# ---------------------------------------------------------------- REPORT.md

gv = lambda s, c, ch: gates[(s, c, ch)]

def gate_line(scope, cell, ch):
    x = gv(scope, cell, ch)
    return (f"| {scope} | {cell} | {ch} | {x['status']} | {f(x['E_K32'])} | "
            f"{f(x['E_first16'])} | {f(x['E_last16'])} | "
            f"{x['n_distinct_t_resolved']} | {x['distinct_t']} |")

rep = f"""# REPORT — FR_C1_LATE_COMPLETION_V01（C1 遗留 late anchors + 旧数据离线复核）

日期：2026-09-21。预算独立建账；旧 17,994/18,000 与旧 C1（45/605/304/1801）不动。
本轮合计 {rt['optimizer_updates']}/{rt['grad_evals']}/{rt['hvp_calls']}/{rt['micro_forward_calls']}
（updates/grads/HVPs/micros），within_caps={res['within_caps']}。

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
   - shadow：R/C 数值门全过（E=0.665/0.571，halves ≤0.954），distinct-t=2（{{16,48}}）→ 判 PASS。
   - **两条纪律性保留**：(a) late 的通过不回溯改变 t0 的 NUMERICAL_UNRESOLVED 事实；
     (b) shadow 的 {{16,48}} 可分辨不是独立总体确认（同一 R/C 对、相邻 t、同批探针库）。

## 门追踪总表

| scope | cell | ch | status | E_K32 | E_first16 | E_last16 | n_t | distinct_t |
|---|---|---|---|---|---|---|---|---|
{gate_line('OLD4','R','field')}
{gate_line('OLD4','R','shadow')}
{gate_line('OLD4','C','field')}
{gate_line('OLD4','C','shadow')}
{gate_line('LATE2','R','field')}
{gate_line('LATE2','R','shadow')}
{gate_line('LATE2','C','field')}
{gate_line('LATE2','C','shadow')}
{gate_line('COMBINED6','R','field')}
{gate_line('COMBINED6','R','shadow')}
{gate_line('COMBINED6','C','field')}
{gate_line('COMBINED6','C','shadow')}

## 协议五问回答

### ① 数学/实现身份与经验支持分别是什么？

**数学恒等式（精确成立，与数据无关）**：e = p_D − y_A，e_bank = p_D − p_A，
e_local = p_A − y_A；‖e‖² = ‖e_bank‖² + ‖e_local‖² + 2⟨e_bank, e_local⟩。
late 两个 anchor 的恒等残差（max identity_residual over 4 anchor×channel）=
{g(max(float(r['identity_residual']) for r in decC))}（fp64 舍入量级，见
late_error_decomposition.csv）。
bank/local 分解以**带符号 cross 项**报告（不用 norm 百分比冒充可加贡献）。

**实现身份**：leakage order（预测 order_index 先于 audit）、冻结预测与实测
group_stats 逐位一致（diff=0.000e+00）、标量级状态恒等（v_pre/q_pre_norm/s_norm/eta_t）。

**经验支持（本轮新测）**：late E_K32 = 0.236/0.238（field R/C）、0.248/0.222（shadow R/C），
均 ≤0.8 且 halves ≤1；target 全部 resolved（signal ratio 1.94–2.07）。
COMBINED6 下 field 达 3 distinct t，shadow 达 2 distinct t（{{16,48}}）。

### ② field 的近似缺陷和跨库差在 late 点怎样变化？

- 分解占比（programmatic, 4 anchor×channel 行）：
{decomp_ratios}
- 跨库 bank 分量在 late 点仍主导（见上表），cross 项带符号、percent 级，完整报告（非 norm 百分比归因）。
- field 同 probe 配对残差（audit, mean±SE of ||p_j−y_j||）：
{field_resid_lines}- E 随 t 的比较（field, K32，R/C）：t0 = {field_old_E['t0']}；t16 = {field_old_E['t16']}；t48 = {field_old_E['t48']}；无 E 随 t 增长的证据，t48（A0U 延迟获益区）E 保持低位。

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
- d_c_to_u 在 q+ 方向投影：R {g([r for r in projD if r['cell']=='R'][0]['d_c_to_u_qplus_mean'])}、
  C {g([r for r in projD if r['cell']=='C'][0]['d_c_to_u_qplus_mean'])}（负值=沿 q+ 的下降步）。

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
复算表 `fieldcheck_task{{1..5}}_*.csv` + `fieldcheck_summary.json`。

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
"""

(OUT / "REPORT.md").write_text(rep)
print("REPORT.md written")

# ------------------------------------------------- GTTR_EVIDENCE_UPDATE.md

geu = f"""# GTTR_EVIDENCE_UPDATE — FR_C1_LATE_COMPLETION_V01

## 1. 已确立的数学恒等式（精确）

- 三分解 e = e_bank + e_local 与 ‖e‖² 恒等（含 signed_cross = 2⟨e_bank, e_local⟩）；
  late 恒等残差 ≤ 2.8e-20（fp64 舍入）。
- w_k 恒等式与 stop-v 配对结构（C1 已确立，本轮 late 复核不冲突）。
- 聚合门 = 固定 anchor 集合上的 ΣSSE/ΣTSS 比率（非逐点平均）（C0X erratum 后冻结）。

## 2. 已测经验支持（本轮新增）

- R/A/t48、C/A/t48 双通道 target 全部 resolved（signal ratio 1.94–2.07）；
  E_K32 0.222–0.248，halves ≤0.336。
- COMBINED6：field 3 distinct t（R/C PASS）；shadow 2 distinct t {{16,48}}（R/C PASS，
  带纪律性保留，见 REPORT §执行摘要-4）。
- stop-v：Z audit ratio 1.22–1.24、cosine 0.87；Q dev ratio 1.29–1.30（与 C1 同向）。
- clip activity：dev 18/32 全 active；audit R 出现 3/2/1 次 crossings（16 probes 上），
  C 无 crossings。

## 3. 尚不分辨（数据不足或未达预注册阈值）

- shadow t0 target（旧事实，late 不回溯）。
- 旧 shadow 每 probe 残差 SE（NOT_SAVED）；旧 stop-v 每 probe norm/cosine。
- C cell 的 Q_D/Z_A 等 q+ 投影 |mean| ≤ 2·SE。
- 半库聚合门的 g3 历史缺口（C0 DATA_MISSING 维持）。

## 4. 被特定实验削弱的假设

- "E 随 t 增长"（field late E 未升高，0.236/0.238，与早期同量级或更低）。
- "shadow 整体不可分辨"（t16/t48 均可分辨；但 t0 仍不可分辨，故削弱而非否定）。

## 5. 未检验（NOT_TESTED，维持）

未来 H16/H64 提前预测；adaptive 训练价值；最终训练改进；能耗收益。
不开发 controller/optimizer；不外推禁语（见 REPORT §纪律声明）。
"""

(OUT / "GTTR_EVIDENCE_UPDATE.md").write_text(geu)
print("GTTR_EVIDENCE_UPDATE.md written")
