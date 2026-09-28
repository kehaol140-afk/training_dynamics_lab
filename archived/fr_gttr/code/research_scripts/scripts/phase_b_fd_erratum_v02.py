#!/usr/bin/env python3
"""FR Phase B FD diagnostic v01 — 0-GPU 勘误与动作差补充 (v02).

仅读取 outputs/fr_phase_b_fd_diagnostic_v01/ 下已保存的 CSV 充分统计；
不 import torch、不触碰 GPU、不恢复 Phase B、不调阈值、不重选尺度。
产物：ERRATA_AND_ACTION_DIAG_v02.md + action_difference_diag.csv + d5_error_decomposition.csv
"""
import csv
import math
from pathlib import Path

OUT = Path("/home/b205/projects/fluctuation_response/outputs/fr_phase_b_fd_diagnostic_v01")


def rows(name):
    with open(OUT / name, newline="") as f:
        return list(csv.DictReader(f))


fd = rows("fd_scale_diagnostic.csv")
ad = rows("ad_reference_comparison.csv")
blk = rows("block_perturbation_summary.csv")

E05 = math.exp(0.05)
E05M1 = E05 - 1.0


def fdrow(sid, ho):
    for r in fd:
        if r["state_id"] == sid and not r.get("skipped") \
                and abs(float(r["h_over_delta0"]) - ho) < 1e-12:
            return r
    raise KeyError((sid, ho))


def adrows(sid):
    return [r for r in ad if r["state_id"] == sid and r["row_kind"] == "fd_vs_ad"]


def ckpts(sid):
    return [r for r in ad if r["state_id"] == sid
            and r["row_kind"] == "displacement_checkpoint"]


# =====================================================================
# 任务 1：errvec d = y(delta0) - y(delta0/2) 的 norm vs norm^2
# =====================================================================
t1 = {}
for sid in ("S3", "S1"):
    # 来源 A：block CSV 逐块 errvec_sq 求和（诊断代码直接保存的充分统计）
    sum_sq = sum(float(r["errvec_sq_at_delta0"]) for r in blk if r["state_id"] == sid)
    # 来源 B：门统计恒等式 fd_rel_adj(δ0,δ0/2) × max(‖y(δ0)‖,‖y(δ0/2)‖)
    # 注意：相邻对 (2h,h) 的 fd_rel/fd_cos 存在行 h 上 → (δ0,δ0/2) 对存在 h=0.5 行
    y1, y2 = fdrow(sid, 1.0), fdrow(sid, 0.5)
    n1, n2 = float(y1["y_norm"]), float(y2["y_norm"])
    rel = float(y2["fd_rel_adj"])
    d_norm_B = rel * max(n1, n2)
    # 派生（极化恒等式；交叉内积未直接保存 → 标 DERIVED）
    sq1, sq2 = float(y1["y_sq_norm"]), float(y2["y_sq_norm"])
    d_sq = sum_sq
    inner_derived = 0.5 * (sq1 + sq2 - d_sq)
    cos_derived = inner_derived / (n1 * n2)
    cos_saved = float(y2["fd_cos_adj"])
    t1[sid] = dict(
        sum_sq=sum_sq, d_norm=math.sqrt(sum_sq), d_norm_B=d_norm_B,
        sq1=sq1, sq2=sq2, n1=n1, n2=n2, rel=rel,
        inner_derived=inner_derived, cos_derived=cos_derived, cos_saved=cos_saved,
    )

# =====================================================================
# 任务 2：固定集合内各相邻尺度对与 0.20 门的对照（只做对照陈述，不改门）
# =====================================================================
pairs = [("δ0↔2δ0", 1.0), ("δ0/2↔δ0", 0.5), ("δ0/4↔δ0/2", 0.25), ("δ0/8↔δ0/4", 0.125)]
t2 = {}
for sid in ("S3", "S1"):
    t2[sid] = [(lab, float(fdrow(sid, ho)["fd_rel_adj"])) for lab, ho in pairs]

# =====================================================================
# 任务 3：D5 误差定义分解（全部从保存标量重构；cos 需向量/内积 → DATA_MISSING）
# rel_err 定义：‖ΔG_actual − ΔG_pred‖ / ‖ΔG_actual‖（分母 = 实际变化 norm）
# ΔG_pred = h_eff · y（y ∈ {FD@δ0, FD@2δ0, AD}；h_eff = c·‖s_ref0‖）
# =====================================================================
PRED = [("FD@δ0", 1.0), ("FD@2δ0", 2.0), ("AD", None)]  # None → y_ad_norm
t3 = []
for sid in ("S3", "S1"):
    yad = float(adrows(sid)[0]["y_ad_norm"])
    for ck in ckpts(sid):
        h_eff = float(ck["h_eff"])
        actual = float(ck["actual_dG_norm"])
        for pname, ho in PRED:
            ynorm = yad if ho is None else float(fdrow(sid, ho)["y_norm"])
            pred_norm = h_eff * ynorm
            relerr = float(ck[{"FD@δ0": "pred_fd_delta0_rel_err",
                               "FD@2δ0": "pred_fd_scale2_rel_err",
                               "AD": "pred_ad_rel_err"}[pname]])
            resid = relerr * actual
            t3.append(dict(state=sid, ck=ck["checkpoint"], h_eff=h_eff, actual=actual,
                           pred=pname, pred_norm=pred_norm, resid_norm=resid,
                           denom=actual, relerr=relerr, cos="DATA_MISSING"))

with open(OUT / "d5_error_decomposition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(t3[0].keys()))
    w.writeheader()
    w.writerows(t3)

# =====================================================================
# 任务 5：动作差 dG_action = G(θ+e^.05·s) − G(θ+s)
# 端点 G 向量与交叉内积未保存 → dG_action 的 norm/err/cos = DATA_MISSING
# pred_action = (e^.05−1)·‖s‖·y 的 norm 可由保存标量给出（纯标量，非择优）
# =====================================================================
t5 = []
for sid in ("S3", "S1"):
    s_norm = float(ckpts(sid)[0]["h_eff"])  # c=1 检查点的 h_eff = ‖s_ref0‖
    yad = float(adrows(sid)[0]["y_ad_norm"])
    for pname, ho in [("AD", None), ("FD@δ0", 1.0), ("FD@2δ0", 2.0)]:
        ynorm = yad if ho is None else float(fdrow(sid, ho)["y_norm"])
        t5.append(dict(
            state=sid, pred=pname,
            dalpha_norm=E05M1 * s_norm,
            pred_action_norm=E05M1 * s_norm * ynorm,
            dG_action_norm="DATA_MISSING", abs_err="DATA_MISSING",
            rel_err="DATA_MISSING", cos="DATA_MISSING",
            note="端点G向量与交叉内积未保存；不能由各自ΔG范数反推差向量范数"))
with open(OUT / "action_difference_diag.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(t5[0].keys()))
    w.writeheader()
    w.writerows(t5)

# =====================================================================
# 生成 ERRATA_AND_ACTION_DIAG_v02.md
# =====================================================================
L = []
A = L.append
A("# FD 局部算子诊断 v01 — 勘误与动作差补充 (v02, 0-GPU)")
A("")
A("日期：2026-09-23。输入：仅本目录已保存 CSV（fd_scale_diagnostic / ad_reference_comparison /")
A("block_perturbation_summary）。无 torch、无 GPU、无新数值路径、未恢复 Phase B、未调阈值、未重选尺度。")
A("性质：勘误（修正 v01 REPORT/KEY_DATA 中的表述）+ 动作差补充诊断。旧记录原样保留，本文件不覆盖任何旧文件。")
A("")
A("## 任务 1 — 表 E 勘误：10.04 / 5.85e-3 是 norm²，不是 norm")
A("")
A("对象：errvec d = y(δ0) − y(δ0/2)（S3/S1 首方向）。两条独立复算路径：")
A("")
A("| state | A: Σblock errvec_sq（直接保存的充分统计） | B: fd_rel_adj×max(‖y(δ0)‖,‖y(δ0/2)‖) | ‖d‖=√A |")
A("|---|---|---|---|")
for sid in ("S3", "S1"):
    d = t1[sid]
    A(f"| {sid} | {d['sum_sq']:.6f} | {d['d_norm_B']:.6f} | {d['d_norm']:.6f} |")
A("")
A("两来源一致。派生交叉内积（极化恒等式，交叉内积未直接保存，标 DERIVED）：")
A("")
A("| state | ‖y(δ0)‖² | ‖y(δ0/2)‖² | ⟨y(δ0),y(δ0/2)⟩ derived | cos derived | cos saved (CSV) |")
A("|---|---|---|---|---|---|")
for sid in ("S3", "S1"):
    d = t1[sid]
    A(f"| {sid} | {d['sq1']:.6f} | {d['sq2']:.6f} | {d['inner_derived']:.6f} "
      f"| {d['cos_derived']:.8f} | {d['cos_saved']:.8f} |")
A("")
sq_ratio = t1["S3"]["sum_sq"] / t1["S1"]["sum_sq"]
n_ratio = t1["S3"]["d_norm"] / t1["S1"]["d_norm"]
A(f"**勘误**：旧表述「差 3 个数量级」的对象是 **范数平方比** = {sq_ratio:.1f}（≈10^{math.log10(sq_ratio):.2f}）；")
A(f"**范数本身比** = {n_ratio:.1f}（≈10^{math.log10(n_ratio):.2f}），不到两个数量级。")
A("正确表述：‖d‖²(S3)/‖d‖²(S1)≈1.7e3；‖d‖ 比≈41。")
A("")
A("## 任务 2 — 尺度结论勘误")
A("")
A("固定集合内各相邻尺度对的 fd_rel 与 0.20 门对照（原门只查 (δ0,δ0/2) 这一对）：")
A("")
A("| 相邻对 | S3 fd_rel | S1 fd_rel |")
A("|---|---|---|")
for (lab, v3_), (lab2, v1_) in zip(t2["S3"], t2["S1"]):
    assert lab == lab2
    A(f"| {lab} | {v3_:.4f} | {v1_:.4f} |")
A("")
A("- **保留原失败**：v3 门在首方向查 (δ0,δ0/2)，S3=0.7061>0.20 → FAIL 原样保留，不追认 PASS。")
A("- **勘误旧表述**：「固定集合内没有任何尺度能通过 0.20 门」错误。(δ0/4↔δ0/8) 对 fd_rel=0.1396<0.20")
A("  （S1 全部四对均<0.20）。")
A("- **两件事分开**：① FD(h)→AD(h→0) 的收敛（S3 sym_rel_err 0.98→0.10，δ0/8 处仍未收敛；S1 至 0.002）")
A("  是**导数估计**问题；② 有限步线性模型在真实参考位移上是否有效（任务 3/5）是**预测**问题。")
A("  小尺度对过门只说明 ① 在改善，不构成 ② 的证据，也不回溯改变 v3 的门判定。")
A("")
A("## 任务 3 — D5 误差定义（明确化）")
A("")
A("对象：**固定 SIM、固定 θ0、参考一步位移** s_ref0（clip+AdamW 虚拟步，‖s_ref0‖ 见 h_eff 列），")
A("**不是已运行 MAIN 的真实轨迹**。定义：")
A("actual = G(θ0+c·s_ref0) − G(θ0)；pred = h_eff·y（线性模型，h_eff=c·‖s_ref0‖，y∈{FD@δ0, FD@2δ0, AD}）；")
A("relative_error = ‖actual−pred‖ / ‖actual‖（分母=实际变化范数，下表 denom 列）；")
A("cos(actual, pred)：需端点向量内积，**DATA_MISSING**（未保存）。")
A("")
A("| state | 检查点 | h_eff | ‖actual‖ | 预测 | ‖pred‖ | ‖residual‖ | denom | rel_err | cos |")
A("|---|---|---|---|---|---|---|---|---|---|")
for r in t3:
    A(f"| {r['state']} | {r['ck']} | {r['h_eff']:.4f} | {r['actual']:.4e} | {r['pred']} "
      f"| {r['pred_norm']:.4e} | {r['resid_norm']:.4e} | {r['denom']:.4e} "
      f"| {r['relerr']:.4f} | {r['cos']} |")
A("")
A("（‖pred‖=h_eff×‖y‖、‖residual‖=rel_err×‖actual‖ 均由保存标量重构；机器可读版：d5_error_decomposition.csv）")
A("")
A("## 任务 4 — FP32/FP64 「逐位一致」勘误（落实为真实字段）")
A("")
A("| 项 | FP32 路径 | FP64 路径 | 一致性 |")
A("|---|---|---|---|")
A("| path | ORIG_FP32（FD）/ ORIG_FP32_MATHREF（AD） | REF_FP64 | — |")
A("| dtype | float32 | float64 | — |")
A("| backend | PyTorchMemEffAttention f32 fmha_cutlassF + cutlass SIMT sgemm | cutlass tensorop d884 GEMM | 不同后端（REFERENCE_BACKEND_CHANGE） |")
A("| fd_rel 序列 | 0.7228/0.7061/0.3652/0.1396 | 0.7228/0.7061/0.3652/0.1396 | **4 位显示精度一致** |")
A("| fd_cos 序列 | 0.707582/0.728243/0.951932/0.990387 | 0.707581/0.728244/0.951931/0.990387 | 5 位一致，第 6 位有差 → **非逐位** |")
A("| ‖y_AD‖ | 6.067428e+00（保存 7 位） | 6.0674（console 仅 4 位） | 4 位一致 |")
A("| FP64 基线 | —（未测 FP32 对应项） | loss_rel=2.342e-08, g_cos=1.00000000, g_rel=7.217e-07 | — |")
A("| 向量 abs/rel 差 | — | — | **DATA_MISSING**（向量未保存） |")
A("")
A("**勘误**：旧表述「逐位一致/逐位相同」过强。落实为：fd_rel 在 4 位显示精度一致；fd_cos 第 6 位有差")
A("（0.707582 vs 0.707581 等），故**不得称逐位一致**；向量级绝对/相对差因向量未保存 = DATA_MISSING。")
A("")
A("## 任务 5 — 动作差补充诊断（dG_action）")
A("")
A("定义：dG_action = G(θ0+e^0.05·s_ref0) − G(θ0+s_ref0)；Δα=(e^0.05−1)‖s_ref0‖；")
A("pred_action = Δα·y（y 沿单位方向 v1，故 pred 范数 = Δα·‖y‖）。")
A("")
A("| state | 预测器 | Δα=‖Δ位移‖ | ‖pred_action‖ | ‖dG_action‖ | abs_err | rel_err | cos |")
A("|---|---|---|---|---|---|---|---|")
for r in t5:
    A(f"| {r['state']} | {r['pred']} | {r['dalpha_norm']:.4e} | {r['pred_action_norm']:.4e} "
      f"| {r['dG_action_norm']} | {r['abs_err']} | {r['rel_err']} | {r['cos']} |")
A("")
A("**DATA_MISSING 说明**：两个检查点的 ΔG 向量及其交叉内积未保存（仅存各自的范数")
A("0.5382/0.5473 (S3)、0.1307/0.1378 (S1)），按指令**不能从各自范数反推差向量范数**，")
A("故 dG_action 的 norm/abs_err/rel_err/cos 全部 DATA_MISSING。")
A("仅列原 FD 预测作并列参照（AD 与 FD@δ0/@2δ0 的 pred_action 范数上表），不择优、不改模型。")
A("机器可读版：action_difference_diag.csv。")
A("")
A("## 任务 6 — 结论限定（替换 v01 结论段）")
A("")
A("**可保留**：固定起点线性场（fixed-startpoint linear field）在 **S3 × 原SIM × 首方向** 的")
A("参考一步位移范围内**受限**（S3 AD 预测 rel_err=1.87/1.91；S1 同定义下 0.24/0.25）。")
A("")
A("**不得据此宣称**（v01 中过强或易被误读之处一并撤回）：")
A("- ~~非光滑/不可微~~ → 未做光滑性检验，只能称「强非线性/高曲率（fd_rel 尺度依赖显著）」；")
A("- 相变 /  regimes 切换 → 未检验，不宣称；")
A("- 「纯 checkpoint 根因」 → 未定位到任何单一机制，不宣称；")
A("- NLL 或 tokens-to-target 层面的失败 → 本诊断对象是一步位移响应 G，不涉及任何评测集表现，不宣称。")
A("")
A("**AD 参照的限定**：AD-HVP 只改进**导数核验**（区分 FD 收敛/未收敛），")
A("**不自动解决有限步预测**——即使导数精确（rel_err→0），线性外推在有限步上的有效性须由任务 3/5")
A("类位移检查单独评估。")
A("")
A("## 成本")
A("")
A("本轮 v02：0 前向 / 0 反向 / 0 AD-HVP / 0 updates / 0 V_* 调用；纯 CSV 解析。")

(OUT / "ERRATA_AND_ACTION_DIAG_v02.md").write_text("\n".join(L) + "\n")
print(f"[v02] wrote {OUT/'ERRATA_AND_ACTION_DIAG_v02.md'}")
print(f"[v02] wrote {OUT/'action_difference_diag.csv'} ({len(t5)} rows)")
print(f"[v02] wrote {OUT/'d5_error_decomposition.csv'} ({len(t3)} rows)")
for sid in ("S3", "S1"):
    d = t1[sid]
    print(f"[v02] {sid}: sum_sq={d['sum_sq']:.6f} norm={d['d_norm']:.6f} "
          f"normB={d['d_norm_B']:.6f} cos_derived={d['cos_derived']:.8f} "
          f"cos_saved={d['cos_saved']:.8f}")
