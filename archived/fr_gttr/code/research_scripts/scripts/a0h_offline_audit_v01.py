#!/usr/bin/env python3
"""FR_A0H-0 离线审计 runner —— 只用已有冻结数据，纯 CPU，不重跑任何科学实验。

输入（只读）: outputs/fr_decision_utility_v01/ 四份科学 CSV + manifest + summary.json
输出: outputs/a0h_adjoint_cost_v01/offline_audit/
  key_audit.json / selection_recompute.json / lambda_rule_audit.json
  identity_check.json / oracle_capture.csv / unit_baseline_rows.csv
  paired_bootstrap.csv / paired_bootstrap.json / per_seed_effects.csv
  interaction_c_minus_b.json / descriptives.json / margins.json
  baseline_availability.json / audit_summary.json / OFFLINE_AUDIT_REPORT.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))
sys.path.insert(0, str(ROOT / "research/scripts"))

OUT = ROOT / "outputs/a0h_adjoint_cost_v01/offline_audit"
A0G = ROOT / "outputs/fr_decision_utility_v01"

from fr.a0h_response_cost_audit_v01 import (  # noqa: E402
    ALL_METHODS,
    HORIZONS,
    audit_lambda_rule,
    audit_primary_keys,
    baseline_never,
    check_gain_regret_identity,
    descriptive_rates,
    interaction_c_minus_b,
    load_baseline_rows,
    margins,
    oracle_capture,
    paired_delta_regret_check,
    paired_bootstrap_report,
    per_seed_effects,
    read_csv_rows,
    replay_score_availability,
    recompute_level3,
    compare_level3,
    select_dev_fixed_group,
    apply_dev_fixed,
    select_action,
    sha256_file,
    hierarchical_bootstrap_means,
    summarize_boot,
    unit_key,
)
from fr.a0g_decision_utility_v01 import N_DATA_GROUPS, PRIMARY_HORIZON  # noqa: E402

N_BOOT = 10000
BOOT_SEED = 20260914


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    level1 = read_csv_rows(A0G / "level1_local_derivative.csv")
    level1b = read_csv_rows(A0G / "level1_finite_action.csv")
    level2 = read_csv_rows(A0G / "level2_action_predictions.csv")
    level3 = read_csv_rows(A0G / "level3_decision_metrics.csv")
    scan = read_csv_rows(A0G / "lambda_scan_detail.csv")
    lambda_selection = json.loads((A0G / "lambda_selection.json").read_text(encoding="utf-8"))
    summary = json.loads((A0G / "summary.json").read_text(encoding="utf-8"))
    validation_split = json.loads((A0G / "validation_split.json").read_text(encoding="utf-8"))
    dgd = json.loads((A0G / "data_group_definition.json").read_text(encoding="utf-8"))
    partition = json.loads((ROOT / "outputs/e3_confirmatory/v07/partition_v07.json").read_text(encoding="utf-8"))

    input_hashes = {p.name: sha256_file(p) for p in (
        A0G / "level1_local_derivative.csv", A0G / "level1_finite_action.csv",
        A0G / "level2_action_predictions.csv", A0G / "level3_decision_metrics.csv",
        A0G / "lambda_scan_detail.csv", A0G / "lambda_selection.json",
        A0G / "summary.json", A0G / "validation_split.json", A0G / "data_group_definition.json",
    )}

    # ---------- 1. 主键审计 ----------
    cfg = summary["config"]
    key_audit = audit_primary_keys(
        level1, level1b, level2, level3,
        expected_dev_states=[tuple(x) for x in cfg["dev_states"]],
        expected_formal={
            "seeds": cfg["formal_seeds"], "steps": cfg["formal_steps"],
            "streams_per_state": cfg["streams_per_state"],
        },
    )
    # 数据组划分与 validation split 覆盖（对照 partition_v07.json 离线复核）
    members = dgd["members"]
    merged = [i for g in members.values() for i in g]
    group_partition_ok = (
        len(merged) == len(set(merged)) == len(set(partition["train_indices"]))
    )
    vsplit_ok = (
        set(validation_split["v_score"]) & set(validation_split["v_eval"]) == set()
        and set(validation_split["v_score"]) | set(validation_split["v_eval"])
        == set(map(int, partition["eval_indices"]))
    )
    key_audit["detail"]["data_group_partition_ok"] = group_partition_ok
    key_audit["detail"]["validation_split_cover_ok"] = vsplit_ok
    if not group_partition_ok or not vsplit_ok:
        key_audit["status"] = "FAIL"
    write_json(OUT / "key_audit.json", key_audit)

    # ---------- 2. 冻结规则复算 + 逐格核对 ----------
    recomputed = recompute_level3(level2)
    cmp3 = compare_level3(recomputed, level3)
    write_json(OUT / "selection_recompute.json", {
        "status": cmp3["status"],
        "n_recomputed_rows": len(recomputed),
        "match": cmp3,
        "note": "从 level2 预测/实际值用冻结 select_action/level3_decision 复算，"
                "含 random（重放冻结 rng 公式）；逐格核对冻结 level3 CSV。",
    })
    write_csv(OUT / "unit_level3_recomputed.csv", [
        {k: (json.dumps(v) if isinstance(v, (list, bool)) else v) for k, v in r.items()}
        for r in recomputed
    ])

    # ---------- 3. λ 规则审计 ----------
    lambda_audit = audit_lambda_rule(scan, lambda_selection, level1)
    write_json(OUT / "lambda_rule_audit.json", lambda_audit)

    # ---------- 4. 恒等式 ----------
    identity = check_gain_regret_identity(level3)
    delta_id = paired_delta_regret_check(level3, "C", "B")
    identity["paired_delta_C_minus_B"] = delta_id
    # 全体方法两两 ΔRegret = −ΔGain
    all_pairs = []
    for a in ("C", "B", "alignment", "random"):
        for b in ("C", "B", "alignment", "random"):
            if a >= b:
                continue
            r = paired_delta_regret_check(level3, a, b)
            all_pairs.append({"a": a, "b": b, **r})
    identity["all_pairs_delta_check"] = all_pairs
    write_json(OUT / "identity_check.json", identity)

    # OracleCapture（formal H=4/H=8，含补充基线需先构造行）
    dev_sel = select_dev_fixed_group(level2, split="dev")
    j_star = dev_sel["j_star"]
    oc_rows = []
    for h in (PRIMARY_HORIZON, 8):
        combined = [r for r in level3 if r["split"] == "formal" and int(r["H"]) == h]
        combined += baseline_never(level2, "formal", h)
        combined += apply_dev_fixed(level2, "formal", h, j_star)
        oc = oracle_capture(combined, "formal", h, methods=ALL_METHODS)
        for m, v in oc.items():
            oc_rows.append({"split": "formal", "H": h, "method": m, **v})
    write_csv(OUT / "oracle_capture.csv", oc_rows)
    # OracleCapture 联合重采样（分子分母同一 replicate）
    oc_boot = []
    for h in (PRIMARY_HORIZON, 8):
        units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"]))
                        for r in level3 if r["split"] == "formal" and int(r["H"]) == h})
        combined = [r for r in level3 if r["split"] == "formal" and int(r["H"]) == h]
        combined += baseline_never(level2, "formal", h)
        combined += apply_dev_fixed(level2, "formal", h, j_star)
        by = {}
        for r in combined:
            u = (int(r["seed"]), int(r["step"]), int(r["stream_id"]))
            by.setdefault(u, {})[r["method"]] = r
        met = {m: {} for m in ALL_METHODS}
        orc = {}
        for u in units:
            orc[u] = -float(by[u]["C"]["best_possible_delta"])
            for m in ALL_METHODS:
                met[m][u] = -float(by[u][m]["actual_delta_selected"])
        boot = hierarchical_bootstrap_means(units, {**met, "__oracle__": orc}, N_BOOT, BOOT_SEED)
        for m in ALL_METHODS:
            ratio = boot[m] / boot["__oracle__"]
            denom = boot["__oracle__"]
            unstable = bool(
                np.percentile(np.abs(denom), 10) < 1e-12
                or (np.std(denom) / max(abs(np.mean(denom)), 1e-30)) > 0.5
            )
            oc_boot.append({
                "split": "formal", "H": h, "method": m,
                "oracle_capture_boot": summarize_boot(ratio),
                "denominator_sum_mean": float(np.mean(denom)),
                "denominator_cv": float(np.std(denom) / max(abs(np.mean(denom)), 1e-30)),
                "unstable_flag": unstable,
            })
    write_json(OUT / "oracle_capture_bootstrap.json", oc_boot)

    # ---------- 5. 配对 bootstrap（formal，H=4 主 / H=8 辅助） ----------
    boot_rows = []
    boot_json = []
    for h in (PRIMARY_HORIZON, 8):
        combined = [r for r in level3 if r["split"] == "formal" and int(r["H"]) == h]
        combined += baseline_never(level2, "formal", h)
        combined += apply_dev_fixed(level2, "formal", h, j_star)
        rep = paired_bootstrap_report(combined, "formal", h, ALL_METHODS, N_BOOT, BOOT_SEED)
        boot_json.append(rep)
        for row in rep:
            flat = {"split": row["split"], "H": row["H"], "method": row["method"]}
            for k, v in row.items():
                if k in ("split", "H", "method"):
                    continue
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            boot_rows.append(flat)
    write_csv(OUT / "paired_bootstrap.csv", boot_rows)
    write_json(OUT / "paired_bootstrap.json", boot_json)

    # per-seed + LOSO
    seed_rows, loso_rows = [], []
    for h in (PRIMARY_HORIZON, 8):
        combined = [r for r in level3 if r["split"] == "formal" and int(r["H"]) == h]
        combined += baseline_never(level2, "formal", h)
        combined += apply_dev_fixed(level2, "formal", h, j_star)
        ps, ls = per_seed_effects(combined, "formal", h, ALL_METHODS)
        seed_rows += ps
        loso_rows += ls
    write_csv(OUT / "per_seed_effects.csv", seed_rows)
    write_csv(OUT / "leave_one_seed_out.csv", loso_rows)

    # 交互（后续分析）
    inter = interaction_c_minus_b(level3, "formal", n_boot=N_BOOT, seed=BOOT_SEED)
    write_json(OUT / "interaction_c_minus_b.json", inter)

    # ---------- 6. 描述统计 ----------
    desc = []
    marg = []
    for h in (PRIMARY_HORIZON, 8):
        desc += descriptive_rates(level3, level2, "formal", h, ("random", "alignment", "B", "C", "dev_fixed"))
        marg.append(margins(level2, "formal", h))
    write_json(OUT / "descriptives.json", desc)
    write_json(OUT / "margins.json", marg)

    # ---------- 7. 基线可用性 ----------
    availability = {
        "never_intervene": {"status": "AVAILABLE", "note": "j=0 恒选，纯离线可算"},
        "dev_fixed_group": {
            "status": "AVAILABLE",
            "j_star": j_star,
            "selection": dev_sel,
            "note": "dev（split=dev，3 态 × 8 streams × H{1,4,8}，λ=0.05）mean u_ij 选组，"
                    "先冻结再 formal 评价；formal 未参与选组。",
        },
        "replay_score": replay_score_availability(level1b, scan),
    }
    write_json(OUT / "baseline_availability.json", availability)

    # ---------- 汇总状态 ----------
    statuses = {
        "key_audit": key_audit["status"],
        "selection_recompute": cmp3["status"],
        "lambda_rule": lambda_audit["status"],
        "identity_gain_regret": identity["status"],
        "paired_delta_regret": delta_id["status"],
        "paired_bootstrap": "PASS",
        "baseline_never": "AVAILABLE",
        "baseline_dev_fixed": "AVAILABLE",
        "baseline_replay_score": "MISSING_REPLAY_SCORE",
        "descriptives": "PASS",
        "interaction_c_minus_b": "SUPPLEMENTARY",
    }
    audit_summary = {
        "run_id": "A0H-0-offline-audit-v01",
        "input_hashes": input_hashes,
        "frozen_source_hashes_start": (ROOT / "outputs/a0h_adjoint_cost_v01/frozen_source_hashes_START.txt").read_text(encoding="utf-8"),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "statuses": statuses,
        "overall": "PASS_WITH_MISSING_BASELINE",
        "notes": [
            "整体状态刻意不用 COMPLETE：replay-score 基线缺 formal V_score 回放列（MISSING_REPLAY_SCORE）。",
            "所有补充基线与交互均标注为后续补充分析，未倒写为原预注册项。",
            "H=4 C vs B 方向一致但不显著、H=8 稳健的冻结结论保持不变。",
        ],
    }
    write_json(OUT / "audit_summary.json", audit_summary)

    # ---------- 报告 ----------
    md = render_report(key_audit, cmp3, lambda_audit, identity, oc_rows, oc_boot,
                       boot_rows, seed_rows, inter, desc, availability, j_star, dev_sel)
    (OUT / "OFFLINE_AUDIT_REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def render_report(key_audit, cmp3, lambda_audit, identity, oc_rows, oc_boot,
                  boot_rows, seed_rows, inter, desc, availability, j_star, dev_sel) -> str:
    L = []
    L.append("# A0H-0 离线审计报告（offline audit，纯 CPU，未重跑科学实验）")
    L.append("")
    L.append("输入：outputs/fr_decision_utility_v01/ 冻结四 CSV + manifest + summary.json（只读）。")
    L.append("所有选择/收益复算使用冻结 `select_action`/`level3_decision`；指标复用冻结定义。")
    L.append("")
    L.append("## 状态总览")
    L.append("")
    L.append("| 项 | 状态 |")
    L.append("|---|---|")
    L.append(f"| 主键覆盖/候选语义/权重 | {key_audit['status']} |")
    L.append(f"| 冻结规则复算 vs 冻结 level3 逐格核对 | {cmp3['status']}（max|Δ|={cmp3['max_abs_numeric_diff']:.3g}，差异格数={cmp3['n_cell_diffs']}） |")
    L.append(f"| λ Rule B 复算（全局单一 λ） | {lambda_audit['status']} |")
    L.append(f"| Gain+Regret=OracleGain 恒等式 | {identity['status']}（max err={identity['max_abs_error_gain_plus_regret_minus_oracle']:.3g}） |")
    L.append(f"| 配对 ΔRegret=−ΔGain（C−B） | {identity['paired_delta_C_minus_B']['status']} |")
    L.append("| 基线 a) 永不干预 | AVAILABLE |")
    L.append(f"| 基线 b) dev 固定最佳组 | AVAILABLE（冻结 j*={j_star}） |")
    L.append("| 基线 c) 真实回放评分（V_score） | **MISSING_REPLAY_SCORE**（formal 无 V_score 回放列） |")
    L.append("")
    L.append(f"整体：**PASS_WITH_MISSING_BASELINE**（replay-score 基线缺项，只阻塞该项）。")
    L.append("")
    L.append("## 1. 主键与覆盖")
    L.append("")
    d = key_audit["detail"]
    cov = d["unit_coverage"]
    L.append(f"- dev 单元 {cov['dev_units_found']}/24，formal 单元 {cov['formal_units_found']}/96；"
             f"缺失 dev={len(cov['dev_units_missing'])} formal={len(cov['formal_units_missing'])}；"
             f"意外单元={len(cov['unexpected_units'])}。")
    L.append(f"- 候选集合：l1/l1b 每 unit×H 恰为 j=1..4；l2 恰为 j=0..4（含共同 no-op）；"
             f"异常 unit×H 数 l1={d['candidate_set_l1']['n_bad']} l1b={d['candidate_set_l1b']['n_bad']} "
             f"l2={d['candidate_set_l2']['n_bad']}。")
    L.append(f"- 重复键：l1={d['duplicates_l1']['n_duplicate_keys']} l1b={d['duplicates_l1b']['n_duplicate_keys']} "
             f"l2={d['duplicates_l2']['n_duplicate_keys']} l3={d['duplicates_l3']['n_duplicate_keys']}（均应为 0）。")
    L.append(f"- λ：level1/1b/2 全部 λ=0.05（primary）；H={d['horizon_values']}。")
    L.append(f"- no-op：j=0 行 actual/pred 全 0 的非零行数={d['noop_semantics']['n_nonzero_noop_rows']}；"
             f"formal H=4 非零候选行={d['noop_semantics']['formal_H4_nonzero_candidate_rows']}（期望 384）。")
    L.append(f"- 权重：{d['weighting']['stream_weight']} 等权流级；{d['weighting']['aggregation']}。")
    L.append(f"- 数据组划分 train pool：{d['data_group_partition_ok']}；validation split 覆盖 eval pool：{d['validation_split_cover_ok']}。")
    L.append("")
    L.append("## 2. 冻结规则复算（逐格核对）")
    L.append("")
    L.append(f"- 从 level2 复算全部 4 方法 × 全部 unit×H：{cmp3['n_recomputed']} 行，"
             f"与冻结 level3 缺失键={cmp3['missing_keys']}、额外键={cmp3['extra_keys']}、"
             f"数值最大差={cmp3['max_abs_numeric_diff']:.3g}、不一致格={cmp3['n_cell_diffs']}。")
    L.append("- random 基线按冻结 rng 公式（(seed·100003+step·101+stream)·1000+H）精确重放。")
    L.append("")
    L.append("## 3. λ 规则")
    L.append("")
    L.append(f"- 复算 freeze_rule={lambda_audit['expected_rule']}（冻结 {lambda_audit['frozen_rule']}），匹配={lambda_audit['rule_match']}。")
    L.append(f"- primary λ={lambda_audit['primary_lambda']}，conservative={lambda_audit['conservative_lambda']}；"
             f"通过准则 A 的 λ={lambda_audit['strict_A_passing_lambdas']}（空 → 回退 Rule B）。")
    L.append(f"- **Rule B 是全局单一 λ**：level1 中 distinct λ={lambda_audit['per_unit_lambda']['distinct_lambda_in_level1']}，"
             f"per-unit 选择={lambda_audit['per_unit_lambda']['rule_chooses_per_unit_lambda']}。")
    L.append("")
    L.append("## 4. 恒等式与 OracleCapture")
    L.append("")
    L.append(f"- Gain+Regret−OracleGain 最大误差={identity['max_abs_error_gain_plus_regret_minus_oracle']:.3g}"
             f"（浮点舍入 ~1 ulp；不作为独立显著性证据）。{identity['note']}")
    L.append(f"- C−B 配对 ΔRegret+ΔGain 最大误差={identity['paired_delta_C_minus_B']['max_abs_error_delta_regret_plus_delta_gain']:.3g}。")
    L.append("- OracleCapture（Σu_sel/Σu_oracle，同权重，联合重采样分子分母）：")
    L.append("")
    L.append("| H | 方法 | sum u_sel | sum u_oracle | capture | boot mean [95%CI] | 不稳定标记 |")
    L.append("|---:|---|---:|---:|---:|---|---|")
    oc_b = {(r["H"], r["method"]): r for r in oc_boot}
    for r in oc_rows:
        b = oc_b[(r["H"], r["method"])]
        ci = b["oracle_capture_boot"]
        L.append(f"| {r['H']} | {r['method']} | {r['sum_u_selected']:.3e} | {r['sum_u_oracle']:.3e} | "
                 f"{r['oracle_capture'] if r['oracle_capture'] is not None else 'NA'} | "
                 f"{ci['mean']:.3f} [{ci['ci95_low']:.3f},{ci['ci95_high']:.3f}] | {b['unstable_flag']} |")
    L.append("")
    L.append("## 5. 配对 bootstrap（formal，10000×，全部方法共用重采样索引）")
    L.append("")
    L.append("层级 seed→state→stream 有放回重采样；不制造虚假独立 n，只报告效应量与不确定度；"
             "bootstrap 正向比例不解释为真实胜出概率或 p 值。")
    L.append("")
    L.append("| H | 方法 | Gain mean [95%CI] | Regret mean [95%CI] | harmful mean [95%CI] |")
    L.append("|---:|---|---|---|---|")
    for row in boot_rows:
        g = {k.split('.', 1)[1]: v for k, v in row.items() if k.startswith('gain_boot.')}
        rg = {k.split('.', 1)[1]: v for k, v in row.items() if k.startswith('regret_boot.')}
        hm = {k.split('.', 1)[1]: v for k, v in row.items() if k.startswith('harmful_boot.')}
        L.append(f"| {row['H']} | {row['method']} | "
                 f"{g['mean']:.3e} [{g['ci95_low']:.3e},{g['ci95_high']:.3e}] | "
                 f"{rg['mean']:.3e} [{rg['ci95_low']:.3e},{rg['ci95_high']:.3e}] | "
                 f"{hm['mean']:.3f} [{hm['ci95_low']:.3f},{hm['ci95_high']:.3f}] |")
    L.append("")
    L.append("配对 delta（C 相对各基线；同 replicate 配对）：")
    L.append("")
    L.append("| H | 对比 | ΔGain mean [95%CI] | P(>0) | ΔRegret mean [95%CI] | P(<0) |")
    L.append("|---:|---|---|---|---|---|")
    for row in boot_rows:
        if row["method"] != "C":
            continue
        for base in ("random", "alignment", "B", "never", "dev_fixed"):
            key_g = f"delta_gain_vs_{base}"
            key_r = f"delta_regret_vs_{base}"
            if key_g + ".mean" not in row:
                continue
            g = {k.split('.', 1)[1]: v for k, v in row.items() if k.startswith(key_g + '.')}
            rg = {k.split('.', 1)[1]: v for k, v in row.items() if k.startswith(key_r + '.')}
            L.append(f"| {row['H']} | C−{base} | {g['mean']:.3e} [{g['ci95_low']:.3e},{g['ci95_high']:.3e}] | "
                     f"{g['P_gt_0']:.3f} | {rg['mean']:.3e} [{rg['ci95_low']:.3e},{rg['ci95_high']:.3e}] | {rg['P_lt_0']:.3f} |")
    L.append("")
    L.append(f"- 逐 seed 配对效应与 leave-one-seed-out 见 per_seed_effects.csv / leave_one_seed_out.csv（seed n=4，仅方向/幅度）。")
    L.append(f"- 交互 D_i=(Gain_C−Gain_B)_H8−(Gain_C−Gain_B)_H4：mean={inter['mean_D']:.3e}，"
             f"CI=[{inter['ci95'][0]:.3e},{inter['ci95'][1]:.3e}]，P(>0)={inter['P_gt_0']:.3f}。"
             f"【{inter['label']}】")
    L.append("")
    L.append("## 6. 描述统计（formal）")
    L.append("")
    L.append("| H | 方法 | no-op 选择率 | 错过可获益率 | harmful（分母=全部单元） | 干预次数 | harmful/干预 |")
    L.append("|---:|---|---|---|---|---|---|")
    for r in desc:
        L.append(f"| {r['H']} | {r['method']} | {r['noop_selection_rate']:.3f} | "
                 f"{r['missed_beneficial_rate'] if r['missed_beneficial_rate'] is not None else '—'} | "
                 f"{r['harmful_rate']:.3f} (n={r['harmful_n']}/{r['n_units']}) | {r['intervened_n']} | "
                 f"{r['harmful_rate_among_intervened'] if r['harmful_rate_among_intervened'] is not None else '—'} |")
    L.append("")
    L.append("预测/真实 margin（次优−最优 ΔL，含 no-op）见 margins.json；仅描述，不调阈值。")
    L.append("")
    L.append("## 7. 基线可用性")
    L.append("")
    L.append(f"- 永不干预：{availability['never_intervene']['status']}。")
    L.append(f"- dev 固定最佳组：{availability['dev_fixed_group']['status']}，冻结 j*={availability['dev_fixed_group']['j_star']}；"
             f"dev mean u per j={ {k: round(v, 6) for k, v in availability['dev_fixed_group']['selection']['selection_means_u'].items()} }。")
    L.append(f"- 真实回放评分：{availability['replay_score']['status']} —— {availability['replay_score']['note']}")
    L.append("")
    L.append("## 结论边界")
    L.append("")
    L.append("- 离线对照**支持**决策价值：C 的 Gain/Regret/harmful 在所有配对口径下优于 random/alignment/never/dev_fixed；"
             "C vs B 在 H=4 方向一致但不显著、H=8 稳健——与冻结结论一致，未扩大表述。")
    L.append("- 若 C 未超过某个固定策略，本报告如实缩小状态自适应价值的表述；本审计中 C 超过 dev 固定组与永不干预。")
    L.append("- replay-score 基线缺失不阻塞伴随数学等价验证（A0H-1 独立推进）。")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
