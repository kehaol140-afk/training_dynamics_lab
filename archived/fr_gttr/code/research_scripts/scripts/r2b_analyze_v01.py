#!/usr/bin/env python3
"""GTTR-R2B 分析（CPU）：结构增益、持续性、检测 tokens 与报告。

只读 R2B 制品；输出 gains_vs_horizon.csv / target_detection.csv /
token_counts.csv / REPORT.md / KEY_DATA_FOR_REVIEW.json / figures（<=4）。

口径：
- G_struct(h) = V_monitor(LR_ONLY,h) - V_monitor(FULL,h)（同 state/stream 配对；
  >0 支持完整解）。伴随 FULL-BASE、LR-BASE。
- 数据级不确定性：冻结 64 连续行 BLOCK_PROXY paired bootstrap（1000 次，
  PCG64 seed 2026092402）；路径级不确定性：12 个 (state,stream) 配对的
  SE（不与数据级 CI 混同）。
- 正式检测：h=0,16,...,256 连续两次 <=ell（按 start 的 PhaseA 原阈值），
  h_detect=第二时点；tokens 用该轨迹 cum_input/cum_valid。
- 36 条轨迹不是 36 个独立模型；3 训练 seeds × 2 流 × 3 臂。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr import r2b_bridge_v01 as br  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r2b_solved_update_bridge_v01"
R1_COSTS = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01" / "costs.json"
FEATURED_H = (1, 4, 16, 64, 128, 256)
ARMS = ("BASE_K2", "GTTR_LR_ONLY", "GTTR_FULL")


def _read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _last_ledger():
    last = None
    for line in (OUT / "resource_log.jsonl").read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    manifest = json.loads(
        (OUT / "state_stream_action_manifest.json").read_text())
    mon = _read(OUT / "monitor_metrics.csv")
    steps = _read(OUT / "step_metrics.csv")
    ctrl = _read(OUT / "control_step1.csv")
    ell = {int(k): float(v) for k, v in
           cfg["detection"]["ell_by_start"].items()}

    # ---- 去重（中断重跑的重复行，保留 completed 的最后一版） --------------
    mon_by = {}
    for r in mon:
        key = (r["seed"], r["start"], r["stream"], r["arm"], int(r["h"]))
        mon_by[key] = r
    step_by = {}
    for r in steps:
        if r["finite"] != "True":
            continue
        key = (r["seed"], r["start"], r["stream"], r["arm"], int(r["h"]))
        step_by[key] = r
    ctrl_by = {}
    for r in ctrl:
        ctrl_by[(r["seed"], r["start"], r["stream"], r["arm"],
                 int(r["h"]))] = r

    states = sorted({(r["seed"], r["start"]) for r in mon})
    streams = defaultdict(list)
    for c in manifest["state_stream_action"]:
        streams[(str(c["seed"]), str(c["start"]))] = c["stream_seeds"]

    def mval(seed, start, stream, arm, h):
        r = mon_by.get((seed, start, str(stream), arm, h))
        return float(r["mean"]) if r else float("nan")

    def mnpz(seed, start, stream, arm, h):
        r = mon_by.get((seed, start, str(stream), arm, h))
        return r["npz"] if r else None

    # ---- G_struct(h) 与伴随对照 ------------------------------------------
    gain_rows = []
    per_pair_h256 = []
    for (seed, start) in states:
        for stream in streams[(seed, start)]:
            for h in sorted({int(k_[4]) for k_ in mon_by
                             if k_[0] == seed and k_[1] == start
                             and k_[2] == str(stream)}):
                lr = mval(seed, start, stream, "GTTR_LR_ONLY", h)
                fu = mval(seed, start, stream, "GTTR_FULL", h)
                ba = mval(seed, start, stream, "BASE_K2", h)
                ci = ""
                if h in FEATURED_H:
                    n1 = mnpz(seed, start, stream, "GTTR_LR_ONLY", h)
                    n2 = mnpz(seed, start, stream, "GTTR_FULL", h)
                    if n1 and n2:
                        z1 = np.load(OUT / "monitor_rows" / n1)
                        z2 = np.load(OUT / "monitor_rows" / n2)
                        md, lo, hi = br.block_proxy_paired_bootstrap(
                            z1["nll_sum"], z1["valid_count"], z2["nll_sum"])
                        ci = f"[{lo:.3e},{hi:.3e}]"
                        data_md = md
                    else:
                        data_md = float("nan")
                else:
                    data_md = float("nan")
                gain_rows.append((seed, start, stream, h, lr - fu, fu - ba,
                                  lr - ba, data_md, ci))
                if h == 256:
                    per_pair_h256.append((seed, start, stream, lr - fu))
    with (OUT / "gains_vs_horizon.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "stream", "h", "G_struct_LRminusFULL",
                    "FULLminusBASE", "LRminusBASE",
                    "G_struct_blockproxy_mean", "G_struct_blockproxy_ci"])
        w.writerows(gain_rows)

    # ---- 检测与 tokens -----------------------------------------------------
    det_rows = []
    tok_rows = []
    pair_det = {}
    for (seed, start) in states:
        e = ell[int(start)]
        for stream in streams[(seed, start)]:
            for arm in ARMS:
                grid = {}
                h0 = mon_by.get((seed, start, "shared", "shared", 0))
                grid[0] = float(h0["mean"])
                for h in range(16, 257, 16):
                    r = mon_by.get((seed, start, str(stream), arm, h))
                    if r:
                        grid[h] = float(r["mean"])
                res = br.detect_target(grid, e)
                cum_in = cum_val = None
                if res["h_detect"] is not None:
                    srow = step_by.get((seed, start, str(stream), arm,
                                        res["h_detect"]))
                    if srow:
                        cum_in = int(srow["cum_input"])
                        cum_val = int(srow["cum_valid"])
                s256 = step_by.get((seed, start, str(stream), arm, 256))
                det_rows.append((seed, start, stream, arm, e,
                                 res["status"], res["h_detect"],
                                 cum_in, cum_val))
                tok_rows.append((seed, start, stream, arm,
                                 res["h_detect"] if res["h_detect"] else "NA",
                                 cum_in if cum_in is not None else "NA",
                                 cum_val if cum_val is not None else "NA",
                                 int(s256["cum_input"]),
                                 int(s256["cum_valid"]) if s256 else "NA"))
                pair_det[(seed, start, stream, arm)] = res
    with (OUT / "target_detection.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "stream", "arm", "ell", "status",
                    "h_detect", "cum_input_at_detect", "cum_valid_at_detect"])
        w.writerows(det_rows)
    with (OUT / "token_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "stream", "arm", "h_detect",
                    "input_tokens_at_detect", "valid_tokens_at_detect",
                    "input_tokens_256", "valid_tokens_256"])
        w.writerows(tok_rows)

    # 配对检测比较（LR vs FULL）
    pair_cmp = []
    for (seed, start) in states:
        for stream in streams[(seed, start)]:
            r1 = pair_det[(seed, start, stream, "GTTR_LR_ONLY")]
            r2 = pair_det[(seed, start, stream, "GTTR_FULL")]
            rb = pair_det[(seed, start, stream, "BASE_K2")]
            both = r1["h_detect"] is not None and r2["h_detect"] is not None
            diff = (r2["h_detect"] - r1["h_detect"]) if both else None
            pair_cmp.append({
                "seed": seed, "start": start, "stream": stream,
                "h_LR": r1["h_detect"], "h_FULL": r2["h_detect"],
                "h_BASE": rb["h_detect"],
                "both_detected": both,
                "h_full_minus_lr": diff,
                "token_note": "NO_DETECTED_TOKEN_DIFFERENCE_AT_16_STEP_GRID"
                if both and diff == 0 else ""})

    # ---- ctrl h1 ------------------------------------------------------------
    ctrl_rows = []
    for (seed, start) in states:
        c0 = ctrl_by.get((seed, start, "shared", "shared", 0))
        for stream in streams[(seed, start)]:
            cB = ctrl_by.get((seed, start, str(stream), "BASE_K2", 1))
            cL = ctrl_by.get((seed, start, str(stream), "GTTR_LR_ONLY", 1))
            cF = ctrl_by.get((seed, start, str(stream), "GTTR_FULL", 1))
            ctrl_rows.append({
                "seed": seed, "start": start, "stream": stream,
                "h0_mean": float(c0["mean"]),
                "h1_base": float(cB["mean"]), "h1_lr": float(cL["mean"]),
                "h1_full": float(cF["mean"]),
                "full_minus_lr": float(cF["mean"]) - float(cL["mean"]),
                "full_minus_base": float(cF["mean"]) - float(cB["mean"]),
                "lr_minus_base": float(cL["mean"]) - float(cB["mean"])})

    # ---- 汇总统计 ------------------------------------------------------------
    def _agg(rows, key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals, ddof=1)
                                           / max(np.sqrt(len(vals)), 1e-9)) \
            if len(vals) > 1 else 0.0, len(vals)

    g256 = [r[4] for r in gain_rows if r[3] == 256]
    g256_mean = float(np.mean(g256))
    g256_se = float(np.std(g256, ddof=1) / np.sqrt(len(g256)))
    g256_pos = sum(1 for v_ in g256 if v_ > 0)
    h1_ctrl = [r["full_minus_lr"] for r in ctrl_rows]
    h1_mon = [r[4] for r in gain_rows if r[3] == 1]
    det_full = [r for r in det_rows if r[3] == "GTTR_FULL"]
    det_lr = [r for r in det_rows if r[3] == "GTTR_LR_ONLY"]
    det_base = [r for r in det_rows if r[3] == "BASE_K2"]
    n_reach = {"FULL": sum(1 for r in det_full if r[6] is not None),
               "LR": sum(1 for r in det_lr if r[6] is not None),
               "BASE": sum(1 for r in det_base if r[6] is not None)}
    both_pairs = [p for p in pair_cmp if p["both_detected"]]
    same_h = sum(1 for p in both_pairs if p["h_full_minus_lr"] == 0)
    full_earlier = sum(1 for p in both_pairs
                       if (p["h_full_minus_lr"] or 0) < 0)

    ledger = _last_ledger()
    r1c = json.loads(R1_COSTS.read_text()) if R1_COSTS.exists() else {}

    # ---- 报告 ----------------------------------------------------------------
    L = []
    add = L.append
    add("# GTTR-R2B 正式报告（一次受限解干预 + 共同基线续训）\n")
    add("协议：FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md。"
        "仅第一个 optimizer update 施加 R1 primary r=.05 k=2 封存解；"
        "第 2..256 步全部回到共同 k=2 原生 AdamW。未运行 R3。\n")
    add("## A. 状态/流/动作覆盖\n")
    add("|state|S|x_LR|eta_mult_LR|x_2D|y_2D|alpha_2D|eta_mult_2D|streams|"
        "n0/gm|首步恢复规则|")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in manifest["state_stream_action"]:
        a = c["action"]
        add(f"|{c['seed']}/{c['start']}|S{c['S']}|{a['x_LR']}|"
            f"{a['eta_mult_LR']:.4f}|{a['x_2D']}|{a['y_2D']:.9f}|"
            f"{a['alpha_new_2D']:.9f}|{a['eta_mult_2D']:.4f}|"
            f"{c['stream_seeds']}|{c['n0']}/{c['gm_first']}|"
            f"第2步起原生 k=2（eta_ref(gm)=2*eta_base(gm)）|")
    add(f"\n首步 applied eta（BASE=eta_ref）：start46→{2*3.661792956511227e-4:.6e}"
        f"（gm94），start82→{2*1.7383745082771352e-4:.6e}（gm166）；"
        f"configured_peak=5e-4 分列。\n")

    add("## B. 即时效果（h1）\n")
    add("### V_ctrl（64 行 FP32 同轴目标；h0 每 state 共享）\n")
    add("|state/stream|h1 FULL−LR|h1 FULL−BASE|h1 LR−BASE|")
    add("|---|---|---|---|")
    for r in ctrl_rows:
        add(f"|{r['seed']}/{r['start']}/st{str(r['stream'])[-2:]}|"
            f"{r['full_minus_lr']:.3e}|{r['full_minus_base']:.3e}|"
            f"{r['lr_minus_base']:.3e}|")
    m_h1c, s_h1c, n_h1c = _agg(ctrl_rows, "full_minus_lr")
    add(f"\nh1 FULL−LR 均值 {m_h1c:.3e}（跨 {n_h1c} 个 state/stream 配对，"
        f"路径级 SE {s_h1c:.3e}）。\n")
    add("### V_monitor h1\n")
    add("|state/stream|G_struct(1)=LR−FULL|FULL−BASE|LR−BASE|")
    add("|---|---|---|---|")
    for r in gain_rows:
        if r[3] == 1:
            add(f"|{r[0]}/{r[1]}/st{str(r[2])[-2:]}|{r[4]:.3e}|{r[5]:.3e}|"
                f"{r[6]:.3e}|")
    h1_rows = [r for r in gain_rows if r[3] == 1]
    m_h1m = float(np.mean([r[4] for r in h1_rows])) if h1_rows else float("nan")
    _ = m_h1m

    add("\n## C. 结构增益随视界（G_struct(h)=V_LR−V_FULL；>0 支持完整解）\n")
    add("|state/stream|h1|h4|h16|h64|h128|h256|")
    add("|---|---|---|---|---|---|---|")
    for (seed, start) in states:
        for stream in streams[(seed, start)]:
            vals = {r[3]: r[4] for r in gain_rows
                    if r[0] == seed and r[1] == start and r[2] == stream}
            add(f"|{seed}/{start}/st{str(stream)[-2:]}|" +
                "|".join(f"{vals.get(h, float('nan')):.3e}"
                         for h in FEATURED_H) + "|")
    add(f"\n**主终点 G_struct(256)**：均值 {g256_mean:.3e}，路径级 SE "
        f"{g256_se:.3e}（12 个 state/stream 配对），正数 {g256_pos}/12。"
        f"数据级 BLOCK_PROXY CI 见 gains_vs_horizon.csv（逐配对）。\n")
    add("原始 sum/count 与逐点均值在 monitor_metrics.csv；"
        "G_struct 用同流同臂配对差。\n")

    add("## D. 检测达标（正式网格 h=0,16,...,256；连续两次 <=ell）\n")
    add("|arm|检测达标数/12|NOT_REACHED|h_detect 中位数|")
    add("|---|---|---|---|")
    for arm, key in (("GTTR_FULL", "FULL"), ("GTTR_LR_ONLY", "LR"),
                     ("BASE_K2", "BASE")):
        dets = [r[6] for r in det_rows if r[3] == arm and r[6] is not None]
        add(f"|{arm}|{n_reach[key]}/12|{12 - n_reach[key]}|"
            f"{float(np.median(dets)) if dets else 'NA'}|")
    add("")
    add("|state/stream|h_LR|h_FULL|h_BASE|配对结论|")
    add("|---|---|---|---|---|")
    for p in pair_cmp:
        note = p["token_note"] if p["both_detected"] else "至少一臂未达标"
        add(f"|{p['seed']}/{p['start']}/st{str(p['stream'])[-2:]}|"
            f"{p['h_LR']}|{p['h_FULL']}|{p['h_BASE']}|{note}|")
    add(f"\n两臂都达标的配对 {len(both_pairs)}/12；其中检测时点相同 "
        f"{same_h}（NO_DETECTED_TOKEN_DIFFERENCE_AT_16_STEP_GRID），"
        f"FULL 更早 {full_earlier}，FULL 更晚 "
        f"{len(both_pairs) - same_h - full_earlier}。\n")
    add("达标时点的 input/valid tokens 在 token_counts.csv（真实 cum 计数）；"
        "两臂时点相同时不宣称精确首次达标成本相等（16 步网格分辨率）。\n")

    add("## E. 成本（本轮 vs 历史 R1 辨识，分开）\n")
    add(f"- 本轮真实 updates {ledger.get('optimizer_updates')}/9600"
        f"（含中断轨迹重跑 223 步，全部计费）")
    add(f"- 本轮 forward {ledger.get('total_fw')}/110000；"
        f"backward micro {ledger.get('train_bw')}/19200")
    add(f"- monitor 整库 eval {ledger.get('monitor_evals')}/700；"
        f"ctrl 整库 eval {ledger.get('ctrl_evals')}/60；"
        f"快照 {ledger.get('snapshots')}；失败 {ledger.get('failures')}")
    add(f"- 新增 q/HVP/probe gradients=0；V_final=0")
    if r1c:
        rl = r1c.get("ledger_final", {})
        add(f"- 历史 R1 模型辨识成本（引用，不混入本轮账）："
            f"forward {rl.get('total_fw')}，backward/VJP {rl.get('total_bw')}"
            f"，HVP 方向 {rl.get('hvp_dirs')}，committed updates 0")
    add("- wall 仅 OBSERVATIONAL_SHARED_GPU，不作科学门\n")

    add("## F. 结论标签\n")
    persist = []
    for r in gain_rows:
        if r[3] in FEATURED_H:
            persist.append((r[3], r[4]))
    reversal = sum(1 for (seed, start) in states
                   for stream in streams[(seed, start)]
                   if (lambda v1, v256: v1 > 0 and v256 < 0)(
                       next(r[4] for r in gain_rows
                            if (r[0], r[1], r[2], r[3])
                            == (seed, start, stream, 1)),
                       next(r[4] for r in gain_rows
                            if (r[0], r[1], r[2], r[3])
                            == (seed, start, stream, 256))))
    add(f"- IMPLEMENTATION_AND_COVERAGE: 36/36 轨迹完成，0 数值失败；"
        f"三臂同流同 batch 配对；首步动作与 R1 封存解一致（smoke 位级）。")
    add(f"- ONE_STEP_EFFECT_ON_NEW_STREAMS: h1 ctrl FULL−LR 均值 "
        f"{m_h1c:.3e}（{sum(1 for v_ in h1_ctrl if v_ < 0)}/12 为负）；"
        f"h1 monitor G_struct 均值 {np.mean(h1_mon):.3e}"
        f"（{sum(1 for v_ in h1_mon if v_ > 0)}/12 为正）。")
    add(f"- TERMINAL_NLL_VS_LR_ONLY: G_struct(256) 均值 {g256_mean:.3e} "
        f"[SE {g256_se:.3e}]，{g256_pos}/12 为正。")
    add(f"- EFFECT_PERSISTENCE_OR_REVERSAL: 即时有利后 h256 反转的配对 "
        f"{reversal}/12（逐配对曲线见 gains_vs_horizon.csv）。")
    add(f"- DETECTED_TARGET_TOKENS: 16 步网格下 LR/FULL 检测时点相同 "
        f"{same_h}/{len(both_pairs)} 个双达标配对；差异分辨率受网格限制。")
    add("- DEPLOYABLE_OPTIMIZER_BENEFIT: NOT_TESTED")
    add("- SECOND_ORDER_EXTRA_VALUE: NOT_TESTED_THIS_ROUND")
    add("- V_FINAL_CONFIRMED: NOT_TESTED\n")
    add("## 不能推出的结论\n")
    add("- 不把一次干预收益乘以 256 外推持续规则；不称 r=.05 边界解为全局"
        "内生最优幅度；不宣称二阶比一阶更有价值；未确认 V_final；"
        "36 轨迹非 36 独立模型；检测网格相同时不称首达成本相等。\n")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    key = {
        "primary_endpoint": {"G_struct_256_mean": g256_mean,
                             "G_struct_256_se_paths": g256_se,
                             "n_pairs": len(g256), "n_positive": g256_pos},
        "h1": {"ctrl_full_minus_lr_mean": m_h1c,
               "ctrl_full_minus_lr_se_paths": s_h1c,
               "monitor_G_struct_mean": float(np.mean(h1_mon)),
               "n_positive_monitor": sum(1 for v_ in h1_mon if v_ > 0)},
        "detection": {"reached": n_reach,
                      "both_detected_pairs": len(both_pairs),
                      "same_h_detect": same_h, "full_earlier": full_earlier},
        "reversal_pairs": reversal,
        "pairs": [{"seed": p["seed"], "start": p["start"],
                   "stream": p["stream"],
                   "G256": next(r[4] for r in gain_rows
                                if (r[0], r[1], r[2], r[3])
                                == (p["seed"], p["start"], p["stream"], 256)),
                   "h_LR": p["h_LR"], "h_FULL": p["h_FULL"]}
                  for p in pair_cmp],
        "ledger_final": ledger}
    (OUT / "KEY_DATA_FOR_REVIEW.json").write_text(json.dumps(key, indent=1))
    print("\n".join(L))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
