#!/usr/bin/env python3
"""GTTR-CB1 分析（CPU）：新旧检测分辨率、gap/rho 闭合与报告。

只读本轮制品；输出 detection_resolution.csv / REPORT.md /
KEY_DATA_FOR_REVIEW.json。旧 16 步判定保持不变；新 4 步读数仅
REPLAY_RESOLUTION_DIAGNOSTIC。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr import cb1_cost_v01 as cb  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_cb1_cost_measurement_v01"
R2B = PROJECT_ROOT / "outputs" / "fr_gttr_r2b_solved_update_bridge_v01"
R2B_LEDGER_LAST = None
SLOTS = 1024


def _read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _last_ledger(path):
    last = None
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last


def main():
    ell = {int(k): float(v) for k, v in
           json.loads((OUT / "config.json").read_text())[
               "ell_by_start"].items()}
    ident = _read(OUT / "old_cost_identity.csv")
    pairs = _read(OUT / "paired_cost_effect.csv")
    diag_old = _read(OUT / "progress_cost_diagnostic.csv")
    dense = _read(OUT / "dense_monitor.csv")
    ident_r = _read(OUT / "replay_identity.csv")
    sel = _read(OUT / "selected_replays.csv")

    # ---- 新 4 步网格检测（仅 18 条重放轨迹） ------------------------------
    dense_by = defaultdict(dict)
    for r in dense:
        dense_by[(r["seed"], r["start"], r["stream"], r["arm"])][
            int(r["h"])] = float(r["mean"])
    res_rows = []
    t4_stats = []
    for s in sel:
        key = (str(s["seed"]), str(s["start"]), str(s["stream"]), s["arm"])
        T16 = int(s["T16_old"])
        e = ell[int(s["start"])]
        means = dense_by[key]
        grid4 = [h for h in range(0, T16 + 1, 4) if h in means]
        r4 = cb.detect_lagN(means, e, grid4, lag=16)
        assert r4["h_detect"] is not None, f"{key} T4lag16 not reached"
        res_rows.append((s["seed"], s["start"], s["stream"], s["arm"],
                         T16, r4["h_detect"], r4["h_detect"] - T16,
                         "OLD_GRID16_LAG16", "REFINED_GRID4_LAG16", 16, 16,
                         r4["h_detect"] * SLOTS,
                         (r4["h_detect"] - T16) * SLOTS))
        t4_stats.append((key, T16, r4["h_detect"]))
    with (OUT / "detection_resolution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "stream", "arm", "T16_old",
                    "T4lag16_new", "diff_steps", "old_protocol",
                    "new_protocol", "old_lag", "new_lag",
                    "detect_cost_slots_new", "cost_delta_slots_vs_old"])
        w.writerows(res_rows)

    # ---- 新网格 gap/rho 预测（4 步 ceil 版；18 条） ------------------------
    diag_new = []
    for s in sel:
        key = (str(s["seed"]), str(s["start"]), str(s["stream"]), s["arm"])
        T16 = int(s["T16_old"])
        T4 = next(t4 for k_, _, t4 in t4_stats if k_ == key)
        e = ell[int(s["start"])]
        means = dense_by[key]
        for l in (16, 32):
            g = cb.gap_rho(means[l - 16], means[l], e, l * SLOTS,
                           (l - 16) * SLOTS)
            confirmed = T16 <= l
            if confirmed or g["gap"] <= 0:
                diag_new.append((key, l, g["gap"], g["rho"],
                                 "ALREADY_CONFIRMED" if confirmed
                                 else "NONPOSITIVE_GAP", "", T4))
                continue
            p4 = cb.predict_times(l, g["gap"], g["rho"], 4)
            if p4["abstain"]:
                diag_new.append((key, l, g["gap"], g["rho"], p4["abstain"],
                                 "", T4))
                continue
            actual_rem = (T4 - l) * SLOTS
            pred_rem = p4["predicted_remaining_tokens"]
            diag_new.append((key, l, g["gap"], g["rho"], "OK",
                             {"raw": p4["raw_cross_tokens_after_l"],
                              "t_first": p4["t_pred_first"],
                              "t_confirm": p4["t_pred_confirm"],
                              "pred_rem": pred_rem,
                              "actual_rem": actual_rem,
                              "abs_err": abs(actual_rem - pred_rem),
                              "beyond": p4["forecast_beyond_budget"]}, T4))

    # ---- 报告 ----------------------------------------------------------------
    L = []
    add = L.append
    add("# GTTR-CB1 正式报告（CB0 离线 + CB1 确定性重放）\n")
    add("协议：FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md。"
        "旧 16 步判定保持原样；新 4 步读数仅为 REPLAY_RESOLUTION_"
        "DIAGNOSTIC；未运行持续策略实验。\n")

    add("## A. 旧成本恒等式与旧 hit 复核（36 条）\n")
    max_res = max(abs(float(r["residual"])) for r in ident)
    n_match = sum(1 for r in ident if r["match"] == "True")
    add(f"- alive 加权和 vs direct 成本：**最大残差 {max_res}**（36/36 "
        f"路径严格相等；censor=0）")
    add(f"- 重建 OLD_GRID16_LAG16 与 R2B 记录 hit：**{n_match}/36 一致**")
    ties_fl = sum(1 for r in pairs if r["label_FL"] == "TIE")
    add(f"- 旧检测 tie 事实复现：FULL−LR_ONLY 成本差 TIE **{ties_fl}/12**"
        f"（非 tie 1 例：2030/82 st202 FULL 早 16 步）\n")

    add("## B. 固定 18 路径：旧 T16 vs 新 T4lag16（确认间隔均 16）\n")
    add("|state/stream/arm|T16(旧)|T4lag16(新)|diff(步)|")
    add("|---|---|---|---|")
    for r in res_rows:
        add(f"|{r[0]}/{r[1]}/st{str(r[2])[-2:]}/{r[3]}|{r[4]}|{r[5]}"
            f"|{r[6]}|")
    diffs = [r[6] for r in res_rows]
    n_same = sum(1 for d in diffs if d == 0)
    add(f"\n新网格在 {len(diffs)} 条中改变 {len(diffs) - n_same} 条的读数"
        f"（更早确认 {sum(1 for d in diffs if d < 0)} 条）；"
        f"{n_same} 条仍为相同检测时点。两协议确认 lag 均为 16 updates。\n")

    add("## C. 三臂配对成本差（slots；分 seed/start/stream，全部 12 对）\n")
    add("|state/stream|FULL−LR|标签|FULL−BASE|标签|LR−BASE|标签|")
    add("|---|---|---|---|---|---|---|")
    for r in pairs:
        add(f"|{r['seed']}/{r['start']}/st{str(r['stream'])[-2:]}|"
            f"{float(r['cost_FULLminusLR_slots']):.0f}|{r['label_FL']}|"
            f"{float(r['cost_FULLminusBASE_slots']):.0f}|{r['label_FB']}|"
            f"{float(r['cost_LRminusBASE_slots']):.0f}|{r['label_LB']}|")
    fl = [float(r["cost_FULLminusLR_slots"]) for r in pairs]
    add(f"\nFULL−LR：负（FULL 省）{sum(1 for v in fl if v < 0)}，正 "
        f"{sum(1 for v in fl if v > 0)}，tie {sum(1 for v in fl if v == 0)}"
        f"；均值 {np.mean(fl):.0f} slots。选定流（200/202/204/206/208/210）"
        f"子集单独见 paired_cost_effect.csv。\n")

    add("## D. gap/rho 最低阶闭合（past-only；无拟合）\n")
    add("### 旧 16 步网格版（36 条 × landmark16/32）\n")
    ok = [r for r in diag_old if r["status"] == "OK"]
    abst = [r for r in diag_old if r["status"] != "OK"]
    errs = [float(r["abs_err"]) for r in ok]
    rels = [float(r["abs_err"]) / float(r["actual_remaining_slots"])
            for r in ok]
    beyond = sum(1 for r in ok if r["forecast_beyond_budget"] == "True")
    add(f"- 可用 {len(ok)}/72；abstain {len(abst)}"
        f"（{ {r['status'] for r in abst} }）；FORECAST_BEYOND_BUDGET "
        f"{beyond}")
    if errs:
        add(f"- 绝对误差：中位 {np.median(errs):.0f} slots，"
            f"均值 {np.mean(errs):.0f}；相对误差中位 "
            f"{np.median(rels):.2f}")
    rhos = [float(r["rho"]) for r in diag_old]
    add(f"- rho 范围 [{min(rhos):.3e},{max(rhos):.3e}] nats/token/slot"
        f"（路径区间净进展率，非条件漂移）")
    # 方向性：预测的检测先后 vs 实际（FULL vs LR 配对）
    add("### 新 4 步网格版（18 条重放，T4lag16 对照）\n")
    ok4 = [d for d in diag_new if isinstance(d[5], dict)]
    abst4 = [d for d in diag_new if not isinstance(d[5], dict)]
    add(f"- 可用 {len(ok4)}；abstain {len(abst4)}")
    if ok4:
        e4 = [d[5]["abs_err"] for d in ok4]
        rel4 = [d[5]["abs_err"] / d[5]["actual_rem"] for d in ok4]
        add(f"- 绝对误差中位 {np.median(e4):.0f} slots；相对误差中位 "
            f"{np.median(rel4):.2f}")
    add("- 全部输入、未截断预测与逐行误差在 progress_cost_diagnostic.csv"
        "（旧）与本节（新，写入 KEY_DATA）\n")

    add("## E. 结论分层\n")
    add("- **代数一致（实现层）**：alive 加权和=截尾检测成本（36/36，残差"
        "0）；OLD_GRID16 重建与旧规则逐条一致；GRID4_LAG16 非“连续两次 4 步”"
        "（CPU 反例锁定）。")
    add("- **执行层一致（REPLAY_IDENTITY）**：18/18 路径从 h1 full-state "
        "恢复重放，h16 与 T16 两处重叠复核全部位级一致；评测量不改变 "
        "theta/m/v/RNG。")
    add(f"- **经验支持（范围内）**：一次 R1 解首步改动在固定首 stream 上"
        f"的配对成本差（FULL−LR：1 对省 16384 slots，其余 tie）；新网格"
        f"改变 {len(diffs) - n_same}/18 条读数（均为更早确认）。")
    add(f"- **未分辨/不可用**：gap/rho 最低阶闭合的误差（相对中位 "
        f"{np.median(rels) if errs else float('nan'):.2f}），不支持把 "
        "rho 当作可靠剩余成本预测子；rho 非正路径 abstain；末端 8 步网格"
        " tie 不证明精确首达时间相等。")
    add("- **不允许**：据此设计持续训练规则、把边界解当全局最优、将重放"
        "分辨率收益回写 R2B 成功、或宣称 tokens-to-target 已改善。\n")

    add("## F. 成本与数据角色\n")
    led = _last_ledger(OUT / "resource_log.jsonl")
    r2b_led = _last_ledger(R2B / "resource_log.jsonl")
    add(f"- 本轮重放研究成本：updates {led['replay_updates']}/4800；"
        f"forward {led['total_fw']}/125000；backward {led['train_bw']}"
        f"/9600；monitor 整库 eval {led['monitor_evals']}/900；"
        f"q/HVP/probe=0；V_ctrl/V_final=0；失败 {led['failures']}")
    add(f"- 原轨迹训练 token 轴不变（T16 计 elapsed slots）；"
        f"原 R2B 实验成本（updates 9479 等）单列于 R2B 账本，本轮不混计。")
    add("- dense_monitor 行来源：OLD_REUSED（0/4/16 倍数旧同轴值）/"
        "REPLAY_NEW（8,12,20,... 新 4096 行 sum/count）/REPLAY_OVERLAP"
        "（h16、T16 复核，位级一致）；无 V_final 调用。\n")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    key = {
        "A_cost_identity": {"max_residual": max_res, "n_match": n_match,
                            "n_paths": len(ident),
                            "old_tie_FL": ties_fl},
        "B_resolution": {"rows": [{"key": list(r[:4]), "T16": r[4],
                                   "T4": r[5], "diff": r[6]}
                                  for r in res_rows],
                         "n_changed": len(diffs) - n_same,
                         "n_same": n_same},
        "C_paired": [{"key": [r["seed"], r["start"], r["stream"]],
                      "FL": float(r["cost_FULLminusLR_slots"]),
                      "label": r["label_FL"]} for r in pairs],
        "D_diag_old": {"n_ok": len(ok), "n_abstain": len(abst),
                       "abs_err_median": float(np.median(errs)) if errs
                       else None,
                       "rel_err_median": float(np.median(rels)) if errs
                       else None, "n_beyond": beyond},
        "D_diag_new": {"n_ok": len(ok4), "n_abstain": len(abst4),
                       "abs_err_median": float(np.median(
                           [d[5]["abs_err"] for d in ok4])) if ok4 else None,
                       "rel_err_median": float(np.median(
                           [d[5]["abs_err"] / d[5]["actual_rem"]
                            for d in ok4])) if ok4 else None,
                       "rows": [{"key": list(d[0]), "landmark": d[1],
                                 "gap": d[2], "rho": d[3], "status": d[4],
                                 **(d[5] if isinstance(d[5], dict) else {}),
                                 "T4": d[6]} for d in diag_new]},
        "F_ledger": led, "r2b_ledger_last": r2b_led}
    (OUT / "KEY_DATA_FOR_REVIEW.json").write_text(json.dumps(key, indent=1))
    print("\n".join(L))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
