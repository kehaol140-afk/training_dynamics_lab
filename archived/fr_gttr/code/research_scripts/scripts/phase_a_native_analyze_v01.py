#!/usr/bin/env python3
"""Phase A native calibration —— 纯离线分析（不碰 GPU、不改任何训练状态）。

输入：a1/monitor_row_stats.csv、a1/step_metrics.csv、a1/layer_diagnostics.csv、
      a1/resource_log.jsonl、a1/config.json
输出：trajectory_metrics.csv（含 paired delta + block bootstrap SE）、
      target_calibration.csv、status.json、KEY_DATA_FOR_REVIEW.md

规则（结果前已冻结）：
- paired delta 用同一 cell 内相同 row 的 NLL 差；64 行连续 block proxy
  （无 doc ID，标 BLOCK_PROXY，不声称 doc 级独立）；1000 次 paired
  bootstrap，固定 seed 77070001；SE 只反映该参考库的条件采样差。
- 每个起点组 ell_candidate = 3 seeds h128 monitor mean 的中位数。
- INTERNAL_TARGET_BRACKETED：全部 3 seed V0>ell 且至少 2/3 V256<ell。
- 噪声余量：V0-ell>2SE0 且 ell-V256>2SE256（逐 seed）。
- 回看 hit：两个连续既定 monitor 检查点 <=ell，在第二点记 detected_hit；
  保存 last-above/first-below/second-confirmation；未命中 censored=true。
- 全部 hit CALIBRATION_ONLY；final_call_count 必须为 0。
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
BOOT_REPS = 1000
BOOT_SEED = 77070001
BLOCK_ROWS = 64  # BLOCK_PROXY：无 doc ID 时的固定行块
MONITOR_POINTS = (0, 32, 64, 96, 128, 160, 192, 224, 256)
SEEDS = (2030, 2031, 2032)
STARTS = (46, 82)


def load_monitor():
    """{(seed,start,h): {"sum": per-row np array, "count": np array}}

    CSV 混有 smoke（2030/46 的 h=0/4/8）与 main 的记录；同一 key 若有多批
    4096 行，保留最后一批（main，后写盘）。顺带核对 smoke 与 main 在
    h=0 的全库读数一致（同初态的独立复算）。"""
    data = defaultdict(dict)
    with (OUT / "monitor_row_stats.csv").open() as f:
        rd = csv.reader(f)
        header = next(rd)
        assert header == ["seed", "start", "h", "d0_row", "nll_sum", "valid_count"]
        acc = defaultdict(lambda: {"sum": [], "count": [], "row": []})
        order = []
        for seed, start, h, row, s, c in rd:
            key = (int(seed), int(start), int(h))
            if key not in acc:
                order.append(key)
            acc[key]["sum"].append(float(s))
            acc[key]["count"].append(int(c))
            acc[key]["row"].append(int(row))
    h0_check = {}
    for key in order:
        v = acc[key]
        n = len(v["sum"])
        assert n % 4096 == 0, f"{key}: {n} rows"
        nrep = n // 4096
        sums = [v["sum"][i * 4096:(i + 1) * 4096] for i in range(nrep)]
        if key == (2030, 46, 0) and nrep > 1:
            h0_check = {
                "smoke_vs_main_max_abs_sum_diff":
                    max(abs(a - b) for a, b in zip(sums[0], sums[-1])),
                "n_replicates": nrep}
        take = sums[-1]
        cnt = v["count"][-4096:]
        assert v["row"][-4096:] == list(range(4096, 8192)), "row order drift"
        data[key] = {"sum": np.asarray(take), "count": np.asarray(cnt)}
    if h0_check:
        assert h0_check["smoke_vs_main_max_abs_sum_diff"] < 1e-6, \
            f"smoke/main h0 mismatch: {h0_check}"
        data["_smoke_main_h0_check"] = h0_check
    return data


def block_bootstrap_se(diff_num: np.ndarray, count: np.ndarray) -> float:
    """paired delta 的 SE：64 行 block 重采样，delta=Σnum/Σcount。"""
    n_blocks = len(diff_num) // BLOCK_ROWS
    blocks_num = diff_num.reshape(n_blocks, BLOCK_ROWS)
    blocks_cnt = count.reshape(n_blocks, BLOCK_ROWS)
    rng = np.random.default_rng(BOOT_SEED)
    stats = np.empty(BOOT_REPS)
    for r in range(BOOT_REPS):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        num = blocks_num[idx].sum()
        den = blocks_cnt[idx].sum()
        stats[r] = num / den
    return float(stats.std(ddof=1))


def cell_mean(dat, seed, start, h):
    d = dat[(seed, start, h)]
    return float(d["sum"].sum() / d["count"].sum())


def main():
    dat = load_monitor()
    dat.pop("_smoke_main_h0_check", None)

    # ---- trajectory_metrics.csv ------------------------------------------
    rows = []
    for seed in SEEDS:
        for start in STARTS:
            means = {}
            for h in MONITOR_POINTS:
                d = dat[(seed, start, h)]
                s, c = float(d["sum"].sum()), int(d["count"].sum())
                means[h] = s / c
                rows.append({"seed": seed, "start": start, "h": h,
                             "sum": s, "count": c, "mean": means[h],
                             "delta_kind": "", "delta": "", "se": ""})
            for a, b, tag in ((0, 64, "0->64"), (0, 128, "0->128"),
                              (0, 256, "0->256"), (128, 256, "128->256")):
                da, db = dat[(seed, start, a)], dat[(seed, start, b)]
                num = da["sum"] - db["sum"]
                delta = float(num.sum() / da["count"].sum())
                se = block_bootstrap_se(num, da["count"])
                rows.append({"seed": seed, "start": start, "h": f"{a}-{b}",
                             "sum": "", "count": "", "mean": "",
                             "delta_kind": tag, "delta": delta, "se": se})
    with (OUT / "trajectory_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "start", "h", "sum", "count",
                                          "mean", "delta_kind", "delta", "se"])
        w.writeheader()
        w.writerows(rows)

    # ---- SE of a single eval mean（逐 cell、h）----------------------------
    def se_of_mean(seed, start, h):
        d = dat[(seed, start, h)]
        num = d["sum"].copy()
        return block_bootstrap_se(num, d["count"])

    # ---- target_calibration.csv -------------------------------------------
    tc_rows = []
    group_stats = {}
    for start in STARTS:
        m128 = [cell_mean(dat, s, start, 128) for s in SEEDS]
        ell = float(np.median(m128))
        stats = {"ell": ell, "seeds": {}}
        for seed in SEEDS:
            v0 = cell_mean(dat, seed, start, 0)
            v256 = cell_mean(dat, seed, start, 256)
            se0 = se_of_mean(seed, start, 0)
            se256 = se_of_mean(seed, start, 256)
            series = [cell_mean(dat, seed, start, h) for h in MONITOR_POINTS]
            below = [v <= ell for v in series]
            hit_idx = next((i for i in range(1, len(series))
                            if below[i - 1] and below[i]), None)
            last_above = max((i for i in range(len(series)) if not below[i]),
                             default=None)
            first_below = next((i for i in range(len(series)) if below[i]),
                               None)
            stats["seeds"][seed] = {
                "V0": v0, "V256": v256, "SE0": se0, "SE256": se256,
                "V0_minus_ell": v0 - ell, "ell_minus_V256": ell - v256,
                "noise_margin_V0": (v0 - ell) > 2 * se0,
                "noise_margin_V256": (ell - v256) > 2 * se256,
                "hit": hit_idx is not None,
                "hit_h": MONITOR_POINTS[hit_idx] if hit_idx is not None else None,
                "last_above_h": MONITOR_POINTS[last_above] if last_above is not None else None,
                "first_below_h": MONITOR_POINTS[first_below] if first_below is not None else None,
                "censored": hit_idx is None}
            tc_rows.append({
                "start": start, "ell_candidate": ell, "seed": seed,
                "V0": v0, "V256": v256, "SE0": se0, "SE256": se256,
                "V0_minus_ell": v0 - ell, "ell_minus_V256": ell - v256,
                "noise_margin_V0": stats["seeds"][seed]["noise_margin_V0"],
                "noise_margin_V256": stats["seeds"][seed]["noise_margin_V256"],
                "detected_hit": stats["seeds"][seed]["hit"],
                "hit_h": stats["seeds"][seed]["hit_h"],
                "last_above_h": stats["seeds"][seed]["last_above_h"],
                "first_below_h": stats["seeds"][seed]["first_below_h"],
                "censored": stats["seeds"][seed]["censored"],
                "hit_label": "CALIBRATION_ONLY",
                "final_call_count": 0})
        bracketed = all(stats["seeds"][s]["V0"] > ell for s in SEEDS) and \
            sum(stats["seeds"][s]["V256"] < ell for s in SEEDS) >= 2
        stats["bracketed"] = bool(bracketed)
        group_stats[start] = stats
    with (OUT / "target_calibration.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tc_rows[0].keys()) + ["group_bracketed"])
        w.writeheader()
        for r in tc_rows:
            r["group_bracketed"] = group_stats[r["start"]]["bracketed"]
            w.writerow(r)

    # ---- 分层诊断 top relative update（描述性，不用于选动作）--------------
    top = {}
    ld = defaultdict(list)
    with (OUT / "layer_diagnostics.csv").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("valid") and r["kind"] in ("attn_matrix", "mlp_matrix", "embedding"):
                ld[(r["cell_seed"], r["cell_start"], r["h"])].append(r)
    for (seed, start, h), lst in ld.items():
        lst.sort(key=lambda r: -r["r_total"])
        top[f"{seed}/{start}@{h}"] = [
            {"param": r["param"], "r_total": r["r_total"],
             "b_rad": r["b_rad"], "r_tangent": r["r_tangent"],
             "W_norm": r["W_norm"], "s_norm": r["s_norm"],
             "I_m_norm": r["I_m_norm"], "I_v_norm": r["I_v_norm"]}
            for r in lst[:3]]

    # ---- 资源分账 ----------------------------------------------------------
    res = [json.loads(l) for l in (OUT / "resource_log.jsonl").open()]
    # smoke 与 main 是两次进程，counter 各自从 0 起；分段取各段末行求和
    segments, cur = [], [res[0]]
    for r in res[1:]:
        if r["optimizer_updates"] < cur[-1]["optimizer_updates"]:
            segments.append(cur)
            cur = [r]
        else:
            cur.append(r)
    segments.append(cur)
    last = {k: sum(seg[-1][k] for seg in segments) for k in
            ("optimizer_updates", "train_fw", "monitor_fw", "monitor_evals",
             "failures", "snapshots")}
    wall = max(r["wall_s"] for r in res)
    resources = {
        "planned_updates": 1552, "cap_updates": 1664,
        "actual_updates": last["optimizer_updates"],
        "planned_monitor_fw": 7296, "cap_monitor_fw": 7800,
        "actual_monitor_fw": last["monitor_fw"],
        "planned_total_fw": 10400, "cap_total_fw": 12000,
        "actual_total_fw": last["train_fw"] + last["monitor_fw"],
        "actual_train_fw": last["train_fw"],
        "monitor_evals": last["monitor_evals"],
        "failures": last["failures"],
        "snapshots": last["snapshots"],
        "new_gradient_calls": 0, "new_hvp_calls": 0,
        "v_final_eval_calls": 0,
        "wall_s": round(wall, 1),
        "cap_wall_s": 5400}

    out = {"group_stats": {str(k): v for k, v in group_stats.items()},
           "top_layers": top, "resources": resources}
    (OUT / "analysis_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(resources, indent=1))
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main()
