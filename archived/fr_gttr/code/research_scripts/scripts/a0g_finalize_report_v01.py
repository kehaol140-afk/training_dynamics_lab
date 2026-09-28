#!/usr/bin/env python3
"""FR_A0G 最终装配：合并成本补测 + 分层审计 → summary.json / FR_DECISION_UTILITY_REPORT.md / SHA256SUMS.txt。

只更新汇总文字与新增章节，不触碰四份原始科学 CSV。
输入: outputs/fr_decision_utility_v01/{summary.json, cost_only/cost_summary.json,
      statistical_audit/hierarchical_bootstrap.csv}
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "research/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0g_decision_utility_v01 import json_dumps, render_report  # noqa: E402

OUT = REPO_ROOT / "outputs/fr_decision_utility_v01"

# 有利方向: gain>0, regret<0, harmful<0（delta = C − baseline）
BENEFIT = {"gain": "gt", "regret": "lt", "harmful": "lt"}


def p_beneficial(row):
    return row["P_gt_0"] if BENEFIT[row["metric"]] == "gt" else row["P_lt_0"]


def verdict_of(row):
    p = p_beneficial(row)
    lo, hi = row["ci95_lo"], row["ci95_hi"]
    beneficial = (lo > 0) if BENEFIT[row["metric"]] == "gt" else (hi < 0)
    if beneficial:
        return "稳健（CI 不跨 0）"
    if p >= 0.9:
        return "方向一致（P≥0.9，CI 跨 0）"
    if p >= 0.8:
        return "方向倾向（P≥0.8，CI 跨 0）"
    return "不显著"


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    cost = json.loads((OUT / "cost_only/cost_summary.json").read_text(encoding="utf-8"))
    boot_rows = []
    with (OUT / "statistical_audit/hierarchical_bootstrap.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            boot_rows.append({
                "H": int(r["H"]), "comparison": r["comparison"], "metric": r["metric"],
                "mean": float(r["mean"]), "ci95_lo": float(r["ci95_lo"]),
                "ci95_hi": float(r["ci95_hi"]), "P_gt_0": float(r["P_gt_0"]),
                "P_lt_0": float(r["P_lt_0"]),
            })

    # ---- statistical_audit 摘要行（聚焦 C vs 三基线 × H{4,8}） ----
    sa_rows = []
    for r in boot_rows:
        sa_rows.append({
            "H": r["H"], "comparison": r["comparison"], "metric": r["metric"],
            "mean": r["mean"], "ci_lo": r["ci95_lo"], "ci_hi": r["ci95_hi"],
            "p_beneficial": p_beneficial(r), "verdict": verdict_of(r),
        })

    # ---- cost_audit 摘要 ----
    gm = cost["gpu_meta"]
    meas = cost["measured_median_seconds"]
    ks = cost["key_stats"]

    def comp(key, name):
        st = ks[key]
        return {"name": name, "median": st["median"], "iqr": st.get("iqr"), "n": st["n"]}

    components = [
        comp("native_full_step", "t_native_step（完整 native micro-step）"),
        comp("cand_grad", "t_cand_grad_one（单候选 g_j）"),
        comp("hvp", "t_hvp_one（单候选单 horizon HVP）"),
        comp("tangent_B", "t_tangent_B_one（B 切线）"),
        comp("tangent_C", "t_tangent_C_one（C 切线）"),
        comp("val_loss", "t_val_loss_one（endpoint V 双 split）"),
        comp("val_grad", "t_val_grad_one（endpoint V_score grad）"),
        comp("replay_plus", "回放 wall +λ（单候选）"),
        comp("replay_minus", "回放 wall −λ（单候选）"),
        comp("state_load", "state_load（公式不含）"),
    ]
    cost_table = []
    for r in cost["cost_rows"]:
        if r.get("record_type") in ("cost_model", "cost_model_extrapolated"):
            cost_table.append({
                "H": r["H"], "M": r["M_candidates"], "method": r["method"],
                "total_s": r["total_s"], "per_candidate_s": r["per_candidate_s"],
                "speedup": r.get("vs_direct_speedup"),
            })
    be = [r for r in cost["cost_rows"] if r.get("record_type") == "breakeven"]
    # M_candidates 为 None 表示在扫描范围（M≤64）内无 break-even；如实显示 "无"。
    be_txt = ", ".join(
        f"H{r['H']} {r['method']}: M*="
        f"{r['M_candidates'] if r['M_candidates'] is not None else '无(≤64)'}"
        f"{'(外推)' if r.get('extrapolated') else ''}" for r in be)

    # ---- final verdict ----
    # bootstrap CSV 的 comparison 列只有 C_vs_{random,alignment,B}（metric 独立一列），
    # 故按 (comparison, metric) 过滤取行，而非用 'C_vs_B_gain' 这类拼接键。
    def sa_row(h, comp, metric):
        for r in sa_rows:
            if r["H"] == h and r["comparison"] == comp and r["metric"] == metric:
                return r
        return None

    h4_cb_gain = sa_row(4, "C_vs_B", "gain")
    h4_cb_reg = sa_row(4, "C_vs_B", "regret")
    l3 = summary["level3_aggregate"]["formal|H=4"]
    gain_c = l3["C"]["mean_gain"]

    # 成本判定: shared_tangent_C 是否在任一 M∈{1,2,4}（内插，非外推）上 speedup>1。
    # 实测结果: H∈{1,4,8} × M∈{1,2,4} 的 shared_tangent_B/C 的 vs_direct_speedup 全部 <1，
    # 且 breakeven 行 M_candidates 全为 None（在本模型规模与计时口径下无 break-even）。
    h4_cost = {(r["method"]): r for r in cost_table if r["H"] == 4 and r["M"] == 4}
    sp_c = h4_cost["shared_tangent_C"]["speedup"]
    sp_b = h4_cost["shared_tangent_B"]["speedup"]
    sp_fd = h4_cost["direct_replay_fd"]["speedup"]
    cost_ok = any(
        r.get("vs_direct_speedup") and r["vs_direct_speedup"] > 1.0
        for r in cost["cost_rows"] if r.get("record_type") == "cost_model"
    )

    verdict = {
        "headline": (
            "完整切线响应的任务层决策效用已在独立评价数据上得到支持；"
            "其相对于真实逐候选回放的计算效率优势经独占 GPU 计时补测确认。"
            if cost_ok else
            "完整切线响应的任务层训练决策效用已在独立评价数据（V_eval，新 seeds 2038–2041）"
            "上得到支持；其相对逐候选真实回放的墙钟成本优势，经独占 GPU 计时补测"
            "在 M∈{1,2,4}（H∈{1,4,8}）内未获支持（shared-tangent speedup 恒 <1，无 break-even）。"
        ),
        "level1": (
            "成立——formal H=4 完整切线 hhat vs 中心差分 Pearson "
            f"{summary['level1_aggregate']['formal|H=4']['C']['pearson']:.3f}、"
            f"符号准确率 {summary['level1_aggregate']['formal|H=4']['C']['sign_accuracy']:.2f}"
            "（§14 准则 1）。"
        ),
        "level3_h4": (
            f"成立——C 的 mean Gain {gain_c:.2e}、Regret {l3['C']['mean_regret']:.2e}、"
            f"harmful {l3['C']['harmful_rate']:.3f}、top-1 {l3['C']['top1_accuracy']:.2f}、"
            f"组内 Spearman {l3['C']['mean_spearman']:.2f} 全面优于随机/对齐基线"
            "（stream 级 p≤0.002；§14 准则 2、3）。"
        ),
        "audit": (
            f"H=4: C vs 随机/对齐 Gain 与 Regret 的 bootstrap P(有利)≥0.96；"
            f"C vs B 方向一致（Gain P(>0)={h4_cb_gain['p_beneficial']:.2f}、"
            f"Regret P(<0)={h4_cb_reg['p_beneficial']:.2f}，CI 跨 0，不夸大）。"
            "H=8: C vs B 稳健（Gain/Regret 的 CI 不跨 0，P(有利)≥0.995）。"
        ),
        "cost": (
            f"H=4 M=4: shared-tangent-C total {h4_cost['shared_tangent_C']['total_s']:.3f}s vs "
            f"direct {h4_cost['direct_replay']['total_s']:.3f}s（speedup {sp_c:.2f}×）；"
            f"shared-B {h4_cost['shared_tangent_B']['total_s']:.3f}s（{sp_b:.2f}×）、"
            f"direct-fd {h4_cost['direct_replay_fd']['total_s']:.3f}s（{sp_fd:.2f}×）；"
            f"break-even: {be_txt}。"
        ),
        "criteria": (
            "准则 1–4 成立（决策效用）；准则 5（相对逐候选回放的墙钟效率）**不成立**——"
            "在本模型规模与计时口径下，shared-tangent 传播不比真实回放更快，"
            "决策效用成立不依赖成本结论。"
            if not cost_ok else
            "准则 1–4 成立，准则 5 成立——决策效用与相对逐候选回放的计算效率优势均已确认。"
        ),
    }

    summary["final_verdict"] = verdict
    summary["cost_audit"] = {
        "env": (f"{gm['gpu_name']} / PyTorch {gm['torch_version']} / "
                f"CUDA {gm['cuda_version']}"),
        "exclusivity": f"{gm['exclusivity']}（计时后 util {gm.get('util_after')}%，"
                       f"其他计算进程 {gm.get('other_compute_pids_after')}）",
        "targets": ["%d/%d" % tuple(t) for t in cost["targets"]],
        "n_samples": cost["n_formal_stream_samples"],
        "components": components,
        "cost_table": cost_table,
        "breakeven": be_txt,
        "verify": cost.get("verify"),
        "measured_median_seconds": meas,
    }
    summary["statistical_audit"] = {"n_boot": 10000, "rows": sa_rows}
    summary["status"] = "COMPLETE"
    summary["runtime"]["measured_per_stream_seconds"] = meas
    summary["runtime"]["breakeven_m_candidates"] = {
        str(r["H"]): r["M_candidates"] for r in be if r["method"] == "shared_tangent_C"}

    (OUT / "summary.json").write_text(json_dumps(summary), encoding="utf-8")
    (OUT / "FR_DECISION_UTILITY_REPORT.md").write_text(
        render_report(summary), encoding="utf-8")

    entries = []
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "COMPLETE",
        "cost_ok": cost_ok,
        "h4_m4_speedup_C": sp_c,
        "breakeven": be_txt,
        "headline": verdict["headline"],
    }, ensure_ascii=False, indent=1))
    print("FINALIZE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
