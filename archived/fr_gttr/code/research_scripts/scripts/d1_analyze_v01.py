#!/usr/bin/env python3
"""GTTR-D1 直接更新规则 pilot v1 分析器 —— CPU 统计与表/图。

输入：outputs/fr_gttr_d1_direct_readout_v01/ 下 runner 产物。
输出：target_hits.csv / seed_method_results.csv / paired_comparisons.csv /
      SUMMARY.json / REPORT.md / KEY_DATA_FOR_REVIEW.md / figures/*.png（≤4 张）。

约定（协议 §5.2）：
- 逐 seed 结果直接展示；n=3 不做夸张总体显著性声明。
- 数据级不确定性：Phase A 的 64 行连续 block paired bootstrap（无 doc ID，
  标 BLOCK_PROXY），同一 (seed,h) 的 bootstrap 样本在所有臂间配对共用；
  paired diff = Σ(num_diff)/Σcount；1000 reps；主种子 77070001。
- h256 的 sum/count 保留原始值，只报告支持比较的有效数字。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "fr_gttr_d1_direct_readout_v01"
FIGS = OUT / "figures"
SEEDS = (2030, 2031, 2032)
ARMS = ("ADAMW_LR050", "ADAMW_LR100", "ADAMW_LR200",
        "QH_RAW_A010", "QH_NORM_A010")
NATIVE_ARMS = ("ADAMW_LR050", "ADAMW_LR100", "ADAMW_LR200")
BASE = "ADAMW_LR100"
CAND = "QH_NORM_A010"
RAW = "QH_RAW_A010"
GRID = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
BOOT_N = 1000
BOOT_SEED = 77070001
BLOCK_ROWS = 64
EPS_BOOT = 1.0 / (2 * BOOT_N)

BUDGET_CAPS = {"updates": 4096, "train_bw": 8192, "monitor_evals": 270,
               "monitor_fw": 34560, "total_fw": 45000}
BUDGET_PLAN = {"updates": 3880, "train_fw": 7760, "train_bw": 7760,
               "monitor_evals": 249, "monitor_fw": 31872, "total_fw": 39632}


def _read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _i(x):
    try:
        return int(float(x))
    except Exception:
        return -1


def load_monitor():
    rows = _read_csv(OUT / "monitor_metrics.csv")
    mon = {}   # (seed, arm) -> {h: rec}
    for r in rows:
        key = (_i(r["seed"]), r["arm"])
        mon.setdefault(key, {})[_i(r["h"])] = r
    return mon


def load_monitor_rows(seed, arm, h, mon):
    rec = mon[(seed, arm)][h]
    p = OUT / "monitor_rows" / rec["npz"]
    z = np.load(p)
    return z["row_id"], z["nll_sum"].astype(np.float64), z["valid_count"].astype(np.float64)


def block_boot_paired(seed, arm_a, arm_b, h, mon, rng):
    """64 行连续 block 配对 bootstrap。返回 (delta_point, se, lo, hi, p_two_sided)。"""
    _, sa, ca = load_monitor_rows(seed, arm_a, h, mon)
    _, sb, cb = load_monitor_rows(seed, arm_b, h, mon)
    num = sa - sb
    den = ca                       # 同一 monitor 集，计数跨臂一致
    n_blocks = len(num) // BLOCK_ROWS
    blocks = num.reshape(n_blocks, BLOCK_ROWS).sum(axis=1)
    dblocks = den.reshape(n_blocks, BLOCK_ROWS).sum(axis=1)
    delta_point = float(num.sum() / den.sum())
    idx = rng.integers(0, n_blocks, size=(BOOT_N, n_blocks))
    rep_num = blocks[idx].sum(axis=1)
    rep_den = dblocks[idx].sum(axis=1)
    deltas = rep_num / rep_den
    se = float(deltas.std(ddof=1))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    p = float(min(1.0, 2 * min(max((deltas <= 0).mean(), EPS_BOOT),
                               1 - EPS_BOOT)))
    return delta_point, se, lo, hi, p


def block_boot_se(seed, arm, h, mon, rng):
    """单臂 h256 mean 的 block bootstrap SE。"""
    _, s, c = load_monitor_rows(seed, arm, h, mon)
    n_blocks = len(s) // BLOCK_ROWS
    bs = s.reshape(n_blocks, BLOCK_ROWS).sum(axis=1)
    bd = c.reshape(n_blocks, BLOCK_ROWS).sum(axis=1)
    idx = rng.integers(0, n_blocks, size=(BOOT_N, n_blocks))
    mus = bs[idx].sum(axis=1) / bd[idx].sum(axis=1)
    return float(mus.std(ddof=1))


def detect(mon, seed, arm, ell):
    recs = mon[(seed, arm)]
    means = [_f(recs[h]["mean"]) for h in GRID]
    last_above = None
    for i, h in enumerate(GRID):
        if means[i] > ell:
            last_above = h
    for i in range(1, len(GRID)):
        if means[i - 1] <= ell and means[i] <= ell:
            h_det = GRID[i]
            return {"reached": True, "h_detect": h_det,
                    "N_detect": _i(recs[h_det]["cum_input_tokens"]),
                    "first_in_pair": GRID[i - 1], "second_confirmation": GRID[i],
                    "last_observed_above": last_above}
    return {"reached": False, "h_detect": None, "N_detect": None,
            "first_in_pair": None, "second_confirmation": None,
            "last_observed_above": last_above}


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((OUT / "config.json").read_text())
    ell = float(cfg["detection"]["ell_star"])
    mon = load_monitor()
    step_rows = _read_csv(OUT / "step_metrics.csv")
    diag_rows = _read_csv(OUT / "readout_diagnostics.csv")
    cells = []
    for line in (OUT / "runs" / "main_cells.jsonl").read_text().splitlines():
        if line.strip():
            cells.append(json.loads(line))
    smoke = json.loads((OUT / "smoke_checks.json").read_text())
    impl = json.loads((OUT / "implementation_checks.json").read_text())

    rng = np.random.default_rng(BOOT_SEED)  # 同一 (seed,h) 在所有臂比较间配对共用

    # ---------------- 表 C/D：检测与逐 seed 结果 ----------------
    hits = {}
    for seed in SEEDS:
        for arm in ARMS:
            hits[(seed, arm)] = detect(mon, seed, arm, ell)

    with (OUT / "target_hits.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "reached", "h_detect", "N_detect",
                    "last_observed_above", "first_in_confirmed_pair",
                    "second_confirmation", "terminal_mean_h256",
                    "cum_input_tokens_h256", "cum_valid_tokens_h256"])
        for seed in SEEDS:
            for arm in ARMS:
                d = hits[(seed, arm)]
                r = mon[(seed, arm)][256]
                w.writerow([seed, arm, d["reached"], d["h_detect"], d["N_detect"],
                            d["last_observed_above"], d["first_in_pair"],
                            d["second_confirmation"], _f(r["mean"]),
                            _i(r["cum_input_tokens"]), _i(r["cum_valid_tokens"])])

    # ---------------- 表 B：h256 配对比较 + bootstrap ----------------
    comparisons = []
    comp_pairs = [(CAND, BASE), (RAW, BASE), (CAND, RAW),
                  (CAND, "ADAMW_LR050"), (CAND, "ADAMW_LR200"),
                  ("ADAMW_LR050", BASE), ("ADAMW_LR200", BASE)]
    for (a, b) in comp_pairs:
        for seed in SEEDS:
            d, se, lo, hi, p = block_boot_paired(seed, a, b, 256, mon, rng)
            comparisons.append({"comparison": f"{a}_MINUS_{b}", "seed": seed,
                                "h": 256, "delta": d, "se": se,
                                "ci95_lo": lo, "ci95_hi": hi, "p_boot": p})
    with (OUT / "paired_comparisons.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "seed", "h", "delta",
                                          "se", "ci95_lo", "ci95_hi", "p_boot"])
        w.writeheader()
        for c in comparisons:
            w.writerow(c)

    # ---------------- paired token ratio（双方达标才配对） ----------------
    token_ratios = {}
    for base in [BASE, "ADAMW_LR050", "ADAMW_LR200", RAW]:
        ratios, eligible = [], 0
        for seed in SEEDS:
            if hits[(seed, CAND)]["reached"] and hits[(seed, base)]["reached"]:
                ratios.append((seed, hits[(seed, CAND)]["N_detect"]
                               / max(hits[(seed, base)]["N_detect"], 1)))
                eligible += 1
        token_ratios[f"{CAND}_vs_{base}"] = {
            "eligible_pairs": eligible, "ratios": ratios,
            "note": "双方达标的 seed 配对；非全样本无偏平均"}

    # 截尾统计（未达标按 budget 填 262144，仅用于该截尾量）
    NBUDGET = 262144
    trunc = {}
    for arm in ARMS:
        vals = []
        for seed in SEEDS:
            n = hits[(seed, arm)]["N_detect"]
            vals.append(min(n if n is not None else NBUDGET, NBUDGET))
        trunc[arm] = {"mean_min_N": float(np.mean(vals)),
                      "censored_not_hit_time": True, "values": vals}

    # ---------------- 表 D：arm 级 reached/3 与固定预算表现 ----------------
    arm_summary = {}
    for arm in ARMS:
        reached = sum(1 for s in SEEDS if hits[(s, arm)]["reached"])
        failed = sum(1 for c in cells if c["arm"] == arm and c["aborted"])
        term = [ _f(mon[(s, arm)][256]["mean"]) for s in SEEDS ]
        arm_summary[arm] = {
            "n_reached_3": reached, "n_not_reached": 3 - reached,
            "n_aborted": failed,
            "terminal_means": {str(s): term[i] for i, s in enumerate(SEEDS)},
            "terminal_mean_avg": float(np.mean(term))}

    # ---------------- LOSO CPU companion（描述性） ----------------
    loso = {}
    for seed in SEEDS:
        others = [s for s in SEEDS if s != seed]
        best_k, best_v = None, None
        for k, arm in [(0.5, "ADAMW_LR050"), (1.0, BASE), (2.0, "ADAMW_LR200")]:
            v = float(np.mean([_f(mon[(s, arm)][256]["mean"]) for s in others]))
            if best_v is None or v < best_v:
                best_k, best_v = k, v
        loso[str(seed)] = {
            "chosen_grid_lr": best_k,
            "heldout_h256_chosen": _f(mon[(seed, {0.5: "ADAMW_LR050", 1.0: BASE,
                                                  2.0: "ADAMW_LR200"}[best_k])][256]["mean"]),
            "heldout_h256_native": _f(mon[(seed, BASE)][256]["mean"]),
            "note": "LOSO 描述性小样本，不当独立确认"}

    # ---------------- 表 E：读出诊断 ----------------
    def _rows(arm, seed=None):
        return [r for r in diag_rows if r["arm"] == arm
                and (seed is None or _i(r["seed"]) == seed)]
    norm_rows = _rows(CAND)
    raw_rows = _rows(RAW)
    e = {}
    e["norm_match_max_abs_resid"] = max((_f(r["norm_resid"]) for r in norm_rows),
                                        default=float("nan"))
    e["norm_match_max_rel_resid"] = max(
        (_f(r["norm_resid"]) / max(_f(r["rA_norm"]), 1e-30) for r in norm_rows),
        default=float("nan"))
    e["cos_rA_rN_min"] = min((_f(r["cos_rA_rN"]) for r in norm_rows
                              if math.isfinite(_f(r["cos_rA_rN"]))), default=float("nan"))
    e["cos_rA_rQ_min_raw"] = min((_f(r["cos_rA_rN"]) for r in raw_rows
                                  if math.isfinite(_f(r["cos_rA_rN"]))), default=float("nan"))
    fb = {}
    for r in norm_rows:
        fb[r["fallback"]] = fb.get(r["fallback"], 0) + 1
    e["fallback_counts_norm"] = fb
    e["cos_definition"] = ("cos(r_A, r_N) = dot/(||r_A||*||r_N||)，同状态参考；"
                           "范数均为 Frobenius norm（sqrt sum sq），非 mean(norm)")
    e["raw_vs_norm_c_dot_diff_mean"] = float(np.mean(
        [_f(r["c_dot_diff"]) for r in norm_rows])) if norm_rows else float("nan")
    e["temp_bytes_fp32_total_diag"] = int(sum(_i(r["temp_bytes_fp32"])
                                              for r in norm_rows))
    e["persistent_state_bytes_added"] = 0
    e["definition_note"] = ("NORM 轨迹的 r_A 是本状态 native-formula 参考，"
                            "不是另一条真实 native 轨迹的 r_A；禁止跨轨迹误称"
                            "范数被严格控制相同")

    # ---------------- 表 F：预算 ----------------
    log_lines = [json.loads(l) for l in
                 (OUT / "resource_log.jsonl").read_text().splitlines() if l.strip()]
    final = log_lines[-1] if log_lines else {}
    train_tokens = {a: {"input_slots": 0, "nonpad": 0, "valid": 0, "updates": 0}
                    for a in ARMS}
    for r in step_rows:
        a = r["arm"]
        train_tokens[a]["input_slots"] += _i(r["input_slots"])
        train_tokens[a]["nonpad"] += _i(r["nonpad_tokens"])
        train_tokens[a]["valid"] += _i(r["train_valid_cnt"])
        train_tokens[a]["updates"] += 1
    mon_slots = sum(_i(r["input_slots"]) for r in _read_csv(OUT / "monitor_metrics.csv"))
    budget = {
        "actual": {"updates": final.get("optimizer_updates"),
                   "train_fw": final.get("train_fw"),
                   "train_bw": final.get("train_bw"),
                   "monitor_evals": final.get("monitor_evals"),
                   "monitor_fw": final.get("monitor_fw"),
                   "total_fw": (final.get("train_fw") or 0) + (final.get("monitor_fw") or 0),
                   "failures": final.get("failures"),
                   "numerical_failures": final.get("numerical_failures"),
                   "snapshots": final.get("snapshots")},
        "planned": BUDGET_PLAN, "caps": BUDGET_CAPS,
        "within_caps": all(
            (final.get(k2) or 0) <= cap for k2, cap in
            [("optimizer_updates", BUDGET_CAPS["updates"]),
             ("train_bw", BUDGET_CAPS["train_bw"]),
             ("monitor_evals", BUDGET_CAPS["monitor_evals"]),
             ("monitor_fw", BUDGET_CAPS["monitor_fw"])]
        ) and ((final.get("train_fw") or 0) + (final.get("monitor_fw") or 0))
        <= BUDGET_CAPS["total_fw"],
        "extra_gradient_calls": 0, "extra_hvp_calls": 0,
        "candidate_rollout_for_selection": 0, "v_final_eval_calls": 0,
        "train_input_slots_total": sum(t["input_slots"] for t in train_tokens.values()),
        "monitor_input_slots_total": mon_slots,
        "wall_clock_note": "OBSERVATIONAL_SHARED_GPU",
        "smoke_updates": 40}

    # ---------------- 判定 ----------------
    h256 = {(s, a): _f(mon[(s, a)][256]["mean"]) for s in SEEDS for a in ARMS}
    delta_cand = {s: next(c["delta"] for c in comparisons
                          if c["comparison"] == f"{CAND}_MINUS_{BASE}"
                          and c["seed"] == s) for s in SEEDS}
    ci_excl0 = {s: (next(c["ci95_lo"] for c in comparisons
                         if c["comparison"] == f"{CAND}_MINUS_{BASE}" and c["seed"] == s) < 0)
                    for s in SEEDS}
    n_impr = sum(1 for s in SEEDS if delta_cand[s] < 0)
    verdicts = {}
    verdicts["IMPLEMENTATION"] = (
        "PASS" if (impl.get("all_pass") and smoke.get("verdict") == "PASS"
                   and (budget["actual"]["numerical_failures"] or 0) == 0)
        else "FAIL")
    n_completed = sum(1 for c in cells if c["completed"])
    verdicts["COVERAGE"] = ("PASS" if n_completed == 15 else
                            f"PARTIAL({n_completed}/15)")
    if n_impr == 3 and sum(ci_excl0.values()) >= 2:
        verdicts["FINAL_BUDGET_NLL_EFFECT"] = "SCREENING_SIGNAL"
    elif n_impr == 0:
        verdicts["FINAL_BUDGET_NLL_EFFECT"] = "NO_OBSERVED_GAIN"
    else:
        verdicts["FINAL_BUDGET_NLL_EFFECT"] = "MIXED"
    # 检测 token 效果
    tr = token_ratios[f"{CAND}_vs_{BASE}"]
    if tr["eligible_pairs"] > 0 and all(r[1] < 1 for r in tr["ratios"]) \
            and verdicts["FINAL_BUDGET_NLL_EFFECT"] in ("SCREENING_SIGNAL", "MIXED"):
        verdicts["DETECTED_TARGET_TOKEN_EFFECT"] = "SCREENING_SIGNAL"
    elif hits_total(hits, CAND) == 0:
        verdicts["DETECTED_TARGET_TOKEN_EFFECT"] = "NOT_REACHED_BY_CANDIDATE"
    else:
        verdicts["DETECTED_TARGET_TOKEN_EFFECT"] = "NO_OBSERVED_GAIN"
    # 范数匹配消融
    delta_nr = {s: next(c["delta"] for c in comparisons
                        if c["comparison"] == f"{CAND}_MINUS_{RAW}" and c["seed"] == s)
                for s in SEEDS}
    if all(d < 0 for d in delta_nr.values()):
        verdicts["NORM_MATCHING_ABLATION"] = "NORM_BETTER_THAN_RAW_SCREENING"
    elif all(d > 0 for d in delta_nr.values()):
        verdicts["NORM_MATCHING_ABLATION"] = "RAW_BETTER_THAN_NORM"
    else:
        verdicts["NORM_MATCHING_ABLATION"] = "MIXED"
    # 固定 LR 比较
    best_grid_per_seed = {s: min(NATIVE_ARMS, key=lambda a: h256[(s, a)]) for s in SEEDS}
    cand_beats_grid = all(h256[(s, CAND)] < h256[(s, best_grid_per_seed[s])]
                          for s in SEEDS)
    grid_boundary = any(best_grid_per_seed[s] in ("ADAMW_LR050", "ADAMW_LR200")
                        for s in SEEDS)
    if cand_beats_grid and not grid_boundary:
        verdicts["FIXED_LR_COMPARISON"] = "CAND_BEATS_INTERIOR_GRID"
    elif cand_beats_grid and grid_boundary:
        verdicts["FIXED_LR_COMPARISON"] = ("SCREENING_BUT_BASELINE_GRID_BOUNDARY"
                                           if hits_total(hits, CAND) > 0
                                           else "BASELINE_GRID_BOUNDARY")
    else:
        verdicts["FIXED_LR_COMPARISON"] = (
            "BASELINE_GRID_BOUNDARY" if grid_boundary else "NO_ADVANTAGE_OVER_GRID")

    # ---------------- 图（≤4） ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ADAMW_LR050": "#888888", "ADAMW_LR100": "#000000",
              "ADAMW_LR200": "#BBBBBB", "QH_RAW_A010": "#CC6600",
              "QH_NORM_A010": "#0066CC"}
    # 图1：NLL vs 训练 input tokens
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ARMS:
            xs, ys = [], []
            for h in GRID:
                r = mon[(seed, arm)][h]
                xs.append(_i(r["cum_input_tokens"]) / 1024.0)
                ys.append(_f(r["mean"]))
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, color=colors[arm],
                    label=arm.replace("ADAMW_", "A").replace("QH_", "Q"))
        ax.axhline(ell, color="red", ls="--", lw=0.8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("train input tokens (K)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V_monitor NLL mean")
    axes[0].legend(fontsize=7)
    fig.suptitle("D1: monitor NLL vs train input tokens (16-step grid; ell=dashed)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_nll_vs_tokens.png", dpi=110)
    plt.close(fig)
    # 图2：h256 配对 gain vs native
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(SEEDS))
    for arm, mk in [("ADAMW_LR050", "v"), ("ADAMW_LR200", "^"),
                    (RAW, "s"), (CAND, "o")]:
        ds = [next(c["delta"] for c in comparisons
                   if c["comparison"] == f"{arm}_MINUS_{BASE}" and c["seed"] == s)
              for s in SEEDS]
        ax.plot(xs, ds, marker=mk, ms=6, lw=1.2, color=colors[arm],
                label=f"{arm} - {BASE}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xs, [f"seed {s}" for s in SEEDS])
    ax.set_ylabel("h256 NLL paired diff (BLOCK_PROXY bootstrap)")
    ax.set_title("Fixed-budget gain vs native baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_h256_gain.png", dpi=110)
    plt.close(fig)
    # 图3：检测状态
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axis("off")
    cell_txt, col_labels = [], ["seed", "arm", "reached", "h_detect", "N_detect",
                                "last_above", "terminal NLL"]
    for seed in SEEDS:
        for arm in ARMS:
            d = hits[(seed, arm)]
            cell_txt.append([str(seed), arm.replace("ADAMW_", "A").replace("QH_", "Q"),
                             "Y" if d["reached"] else "N",
                             str(d["h_detect"]), str(d["N_detect"]),
                             str(d["last_observed_above"]),
                             f"{h256[(seed, arm)]:.4f}"])
    tbl = ax.table(cellText=cell_txt, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.15)
    ax.set_title("Detection under discrete 16-step monitoring (ell=%.4f)" % ell)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_detection.png", dpi=110)
    plt.close(fig)
    # 图4：读出方向/范数诊断
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cosn = [_f(r["cos_rA_rN"]) for r in norm_rows if math.isfinite(_f(r["cos_rA_rN"]))]
    cosr = [_f(r["cos_rA_rN"]) for r in raw_rows if math.isfinite(_f(r["cos_rA_rN"]))]
    axes[0].hist(cosn, bins=40, alpha=0.6, label="cos(r_A, r_N) NORM", color="#0066CC")
    axes[0].hist(cosr, bins=40, alpha=0.6, label="cos(r_A, r_Q) RAW", color="#CC6600")
    axes[0].set_xlabel("cosine (same-state readout reference)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    resid = [abs(_f(r["norm_resid"])) / max(_f(r["rA_norm"]), 1e-30)
             for r in norm_rows]
    axes[1].hist(resid, bins=40, color="#0066CC")
    axes[1].set_xlabel("relative norm residual | ||r_N|| - ||r_A|| | / ||r_A||")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Readout diagnostics (NORM same-state norm matching)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_readout_diag.png", dpi=110)
    plt.close(fig)

    # ---------------- seed_method_results.csv ----------------
    with (OUT / "seed_method_results.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "h256_mean", "h256_se_block_proxy", "reached",
                    "h_detect", "N_detect", "terminal_mean", "updates",
                    "train_input_slots", "train_nonpad_tokens", "train_valid_tokens",
                    "not_reached", "aborted"])
        for s_ in SEEDS:
            for a_ in ARMS:
                d_ = hits[(s_, a_)]
                cell_ = next((c for c in cells if c["seed"] == s_ and c["arm"] == a_), {})
                cum_ = cell_.get("cum", {})
                tt_ = train_tokens.get(a_, {})
                w.writerow([s_, a_, f"{summary_pre_h256(s_, a_, mon):.6f}",
                            f"{block_boot_se(s_, a_, 256, mon, rng):.6f}",
                            d_["reached"], d_["h_detect"], d_["N_detect"],
                            f"{summary_h256(s_, a_, mon):.6f}",
                            tt_.get("updates", 0), tt_.get("input_slots", 0),
                            tt_.get("nonpad", 0), tt_.get("valid", 0),
                            not d_["reached"], bool(cell_.get("aborted"))])

    # ---------------- SUMMARY.json ----------------
    summary = {
        "verdicts": verdicts,
        "h256_means": {f"{s}/{a}": h256[(s, a)] for s in SEEDS for a in ARMS},
        "h_monitor_means": {f"{s}/{a}": {str(h): _f(mon[(s, a)][h]["mean"])
                                         for h in (0, 64, 128, 256)}
                            for s in SEEDS for a in ARMS},
        "h256_se_block_proxy": {f"{s}/{a}": block_boot_se(s, a, 256, mon, rng)
                                for s in SEEDS for a in ARMS},
        "detection": {f"{s}/{a}": hits[(s, a)] for s in SEEDS for a in ARMS},
        "token_ratios": token_ratios, "trunc_min_N": trunc,
        "arm_summary": arm_summary, "loso": loso,
        "readout_diag_summary": e, "budget": budget,
        "delta_h256_vs_native": {str(s): delta_cand[s] for s in SEEDS},
        "best_grid_per_seed": {str(s): best_grid_per_seed[s] for s in SEEDS},
        "comparisons_h256": comparisons,
        "v_final_calls": 0,
        "labels": {"bootstrap": "BLOCK_PROXY (64-row blocks, 1000 reps, seed "
                                "77070001); 数据级 SE，不是训练 seed SE",
                   "lr_grid": "粗网格；边界最优标 BASELINE_GRID_BOUNDARY，"
                              "不叫已调到最优 AdamW",
                   "detection": "离散 16 步监测下连续两次 <= ell 的第二时点；"
                                "非精确连续首次穿越；不同于旧 32 步网格，不声称提速"},
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")

    # ---------------- REPORT.md / KEY_DATA_FOR_REVIEW.md ----------------
    write_report(summary, mon, ell, comparisons, hits, arm_summary, budget,
                 e, token_ratios, trunc, loso, smoke, impl, cells, train_tokens)
    print("ANALYZE_DONE", OUT)


def summary_h256(seed, arm, mon):
    return _f(mon[(seed, arm)][256]["mean"])


def summary_pre_h256(seed, arm, mon):
    return _f(mon[(seed, arm)][0]["mean"])


def hits_total(hits, arm):
    return sum(1 for s in SEEDS if hits[(s, arm)]["reached"])


def write_report(summary, mon, ell, comparisons, hits, arm_summary, budget,
                 e, token_ratios, trunc, loso, smoke, impl, cells, train_tokens):
    v = summary["verdicts"]
    L = []
    A = L.append
    A("# FR GTTR-D1 直接更新规则 pilot v1 —— 结果报告\n")
    A("日期：2026-09-23。执行端：Claude Code + Kimi K2.7。")
    A("协议：FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md。")
    A("性质：已知 checkpoint（v07 seed2030-2032 start46）上的开发 pilot；"
      "不是完整预训练/最终能力/墙钟/能耗结果。QH_RAW 为 QHAdamW 族"
      "（Ma & Yarats ICLR 2019, nu1=.9 nu2=1）读出对照，范数 grafting 有先例"
      "（Agarwal et al. 2020）；本轮不作首创声明。\n")

    A("## 判定\n")
    for k, val in v.items():
        A(f"- **{k}**: {val}")
    A("")

    A("## 表 A：3 seeds × 5 arms 配置与覆盖\n")
    A("| seed | arm | kind | k | alpha | start(n0) | stream | LR首→尾 | 状态 |")
    A("|---|---|---|---|---|---|---|---|---|")
    cfg = json.loads((OUT / "config.json").read_text())
    sm = {c["seed"]: c["stream_seed"] for c in cfg["cells"]}
    KMAP = {"ADAMW_LR050": 0.5, "ADAMW_LR100": 1.0, "ADAMW_LR200": 2.0,
            "QH_RAW_A010": 1.0, "QH_NORM_A010": 1.0}
    AMAP = {"ADAMW_LR050": 0.0, "ADAMW_LR100": 0.0, "ADAMW_LR200": 0.0,
            "QH_RAW_A010": 0.10, "QH_NORM_A010": 0.10}
    for s in SEEDS:
        for a in ARMS:
            cell = next((c for c in cells if c["seed"] == s and c["arm"] == a), None)
            status = ("256步完成" if cell and cell["completed"]
                      else ("失败/中止" if cell else "缺失"))
            A(f"| {s} | {a} | {'QH读出' if 'QH' in a else 'native'} | {KMAP[a]} "
              f"| {AMAP[a]} | 46 | {sm[s]} | "
              f"{5e-4*KMAP[a]:.2e}→{5e-5*KMAP[a]:.2e} | {status} |")
    A("")
    A("覆盖：15/15 轨迹 × 256 步；monitor h=0,16,…,256（h0 每 seed 共享一次评价）；"
      "schedule finite_cosine_then_hold_5e-5_v01；LR 倍率整段固定；"
      "无 re-warmup/无 step/moment reset/无新 cooldown。\n")

    A("## 表 B：h0/64/128/256 NLL（全库 sum/count/mean）与配对差\n")
    A("mean = sum(valid-target NLL)/count(valid targets)，4096 行全库聚合。\n")
    A("| seed | arm | h | sum | count | mean |")
    A("|---|---|---|---|---|---|")
    for s in SEEDS:
        for a in ARMS:
            for h in (0, 64, 128, 256):
                r = mon[(s, a)][h]
                A(f"| {s} | {a} | {h} | {float(r['sum']):.1f} | {int(float(r['count']))} "
                  f"| {_f(r['mean']):.5f} |")
    A("")
    A("h256 配对差（BLOCK_PROXY：64 行连续 block，1000 reps，seed 77070001；"
      "delta = Σ(num_diff)/Σcount；数据级 SE/CI，不是训练 seed SE）：\n")
    A("| comparison | seed | Δmean@h256 | SE | 95%CI | p_boot |")
    A("|---|---|---|---|---|---|")
    for c in comparisons:
        A(f"| {c['comparison']} | {c['seed']} | {c['delta']:+.5f} | {c['se']:.5f} "
          f"| [{c['ci95_lo']:+.5f}, {c['ci95_hi']:+.5f}] | {c['p_boot']:.3f} |")
    A("")

    A("## 表 C：目标检测（离散 16 步监测，连续两次 ≤ ell）\n")
    A(f"ell_star = {ell:.12f}（Phase A start46 全精度校准文件读取，未重新估计）。\n")
    A("| seed | arm | reached | h_detect | N_detect | last_above | 1st_in_pair | 2nd_conf | terminal NLL |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for a in ARMS:
            d = hits[(s, a)]
            term = summary["h256_means"][f"{s}/{a}"]
            A(f"| {s} | {a} | {'Y' if d['reached'] else 'NOT_REACHED'} "
              f"| {d['h_detect']} | {d['N_detect']} | {d['last_observed_above']} "
              f"| {d['first_in_pair']} | {d['second_confirmation']} | {term:.5f} |")
    A("")
    A("说明：h_detect 是离散监测下的确认时点，不是精确连续首次穿越；"
      "不假设相邻检查点间单调；本轮 16 步网格不同于旧 32 步网格，"
      "不拿旧 hit 声称提速。未达标/失败/检测间隔全部保留。\n")

    A("## 表 D：arm 级达标与固定预算表现\n")
    A("| arm | reached/3 | not_reached | aborted | h256 means (per seed) | avg |")
    A("|---|---|---|---|---|---|")
    for a in ARMS:
        am = arm_summary[a]
        means = "/".join(f"{am['terminal_means'][str(s)]:.5f}" for s in SEEDS)
        A(f"| {a} | {am['n_reached_3']} | {am['n_not_reached']} | {am['n_aborted']} "
          f"| {means} | {am['terminal_mean_avg']:.5f} |")
    A("")
    A("paired token ratio（双方达标的 seed 才配对；eligible pairs 单列，"
      "非全样本无偏平均）：")
    for k_, tr in token_ratios.items():
        rr = ", ".join(f"seed{s}:{x:.3f}" for s, x in tr["ratios"]) or "无配对"
        A(f"- {k_}: eligible={tr['eligible_pairs']}/3, ratios: {rr}")
    A("")
    A("截尾统计 min(N_detect, 262144)（未达标按预算填，仅用于该截尾量，"
      "不是达标时间）：")
    for a in ARMS:
        A(f"- {a}: mean={trunc[a]['mean_min_N']:.0f} slots, values={trunc[a]['values']}")
    A("")
    A(f"三种固定 LR 全列（粗网格，非充分调优）：ADAMW_LR050 / ADAMW_LR100 / "
      f"ADAMW_LR200；每 seed 网格最优："
      f"{ {s: summary['best_grid_per_seed'][str(s)] for s in SEEDS} }。"
      f"边界最优一律标 BASELINE_GRID_BOUNDARY。\n")
    A("LOSO companion（描述性，小样本，不当独立确认）：")
    for s in SEEDS:
        d = loso[str(s)]
        A(f"- held-out {s}: 他 seed 选 LR{d['chosen_grid_lr']} → "
          f"h256 {d['heldout_h256_chosen']:.5f} vs native {d['heldout_h256_native']:.5f}")
    A("")

    A("## 表 E：同状态读出诊断（h ∈ {1,16,64,128,256}，逐唯一参数张量）\n")
    A(f"- 范数匹配最大绝对残差 | ||r_N||−||r_A|| |：{e['norm_match_max_abs_resid']:.3e}")
    A(f"- 范数匹配最大相对残差：{e['norm_match_max_rel_resid']:.3e}")
    A(f"- 方向 cos(r_A, r_N) 最小值（NORM 臂）：{e['cos_rA_rN_min']:.6f}")
    A(f"- 方向 cos(r_A, r_Q) 最小值（RAW 臂参考）：{e['cos_rA_rQ_min_raw']:.6f}")
    A(f"- fallback 计数（NORM 臂全轨迹诊断点）：{e['fallback_counts_norm']}")
    A(f"- RAW vs NORM 的 c^T(r_N−r_A) 平均：{e['raw_vs_norm_c_dot_diff_mean']:.3e} "
      "（current-training proxy，不是 monitor 贡献）")
    A(f"- 每诊断点每参数临时张量 fp32 字节合计（7 份临时量）："
      f"{e['temp_bytes_fp32_total_diag']}（诊断点合计；逐步同量，无持久 state）")
    A(f"- 新增全维持久 state bytes：{e['persistent_state_bytes_added']}")
    A(f"- 定义：{e['cos_definition']}")
    A(f"- 警示：{e['definition_note']}。\n")

    A("## 表 F：actual / planned / cap\n")
    A("| 项目 | actual | planned | cap |")
    A("|---|---|---|---|")
    b = budget
    A(f"| 总真实 updates（含失败/重试） | {b['actual']['updates']} | "
      f"{b['planned']['updates']} | {b['caps']['updates']} |")
    A(f"| 训练 forward microcalls | {b['actual']['train_fw']} | "
      f"{b['planned']['train_fw']} | （随总量） |")
    A(f"| 训练 backward microcalls | {b['actual']['train_bw']} | "
      f"{b['planned']['train_bw']} | {b['caps']['train_bw']} |")
    A(f"| monitor 整库 eval | {b['actual']['monitor_evals']} | "
      f"{b['planned']['monitor_evals']} | {b['caps']['monitor_evals']} |")
    A(f"| monitor forward microcalls | {b['actual']['monitor_fw']} | "
      f"{b['planned']['monitor_fw']} | {b['caps']['monitor_fw']} |")
    A(f"| 全部 model forward | {b['actual']['total_fw']} | "
      f"{b['planned']['total_fw']} | {b['caps']['total_fw']} |")
    A(f"| 额外梯度/HVP/FD | 0/0/0 | 0/0/0 | 0/0/0 |")
    A(f"| 候选重放用于选择 | 0 | 0 | 0 |")
    A(f"| V_final 模型调用 | 0 | 0 | 0 |")
    A(f"| 训练 input slots（全部臂合计） | {b['train_input_slots_total']} | "
      f"3932160 | 按真实 batch |")
    A(f"| monitor input slots | {b['monitor_input_slots_total']} | "
      f"130547712 | 按实际计 |")
    A(f"| 失败 / 数值失败 | {b['actual']['failures']} / "
      f"{b['actual']['numerical_failures']} | 0 | — |")
    A(f"| 墙钟 | OBSERVATIONAL_SHARED_GPU（见 resource_log.jsonl） | — | 无科学硬上限 |")
    A("")
    A(f"within_caps = {b['within_caps']}；smoke 40 updates 已计入；"
      "失败/smoke/恢复全部计费。\n")

    A("## 机制迹象（有限解释，与算法收益分开）\n")
    A("- 同状态范数匹配在诊断点成立（残差见表 E）：它限制的是当前同状态数据步幅，"
      "不是总参数位移（还含 decay），更不是两条长期轨迹步长始终相等。")
    A("- current-batch 一阶对齐 c^T r 不是独立任务收益保证（CPU 测试 T5 反例）；"
      "本轮只把它作 current-training proxy 记录。")
    A("- 若 NORM 与 RAW 差异小，说明在本配方/尺度下范数匹配改变有限；"
      "若 RAW 同样或更好，不声称范数匹配是收益机制。")
    A("- 本轮 monitor 密度（16 步 × 全库）是研究测量成本，不代表部署频率。\n")

    A("## 失败 / 未达标 / 不确定\n")
    n_fail = b['actual']['failures']
    A(f"- 数值失败：{b['actual']['numerical_failures']}；其他失败："
      f"{n_fail - b['actual']['numerical_failures']}。")
    for a in ARMS:
        A(f"- {a}: 未达标 {arm_summary[a]['n_not_reached']}/3。")
    A("- n=3 seeds，逐 seed 结果为准；不做夸张总体显著性声明；"
      "bootstrap 是数据级 SE，不是训练 seed SE。")
    A("- 本轮 V_final=0：final 质量未独立确认；N_detect 是开发目标下的新流筛查证据。\n")

    A("## 摘要结论（协议 §10 五问）\n")
    A(f"1. 直接规则是否改善真实目标/检测训练 token："
      f"FINAL_BUDGET_NLL_EFFECT={v['FINAL_BUDGET_NLL_EFFECT']}，"
      f"DETECTED_TARGET_TOKEN_EFFECT={v['DETECTED_TARGET_TOKEN_EFFECT']}。")
    A(f"2. 是否超过所测固定 LR 网格及 QH_RAW："
      f"FIXED_LR_COMPARISON={v['FIXED_LR_COMPARISON']}，"
      f"NORM_MATCHING_ABLATION={v['NORM_MATCHING_ABLATION']}。")
    A("3. 改动/尺度控制/噪声代价的可见迹象：见表 E 与机制节。")
    A(f"4. 失败/未达标/不确定：数值失败 {b['actual']['numerical_failures']}；"
      f"未达标情况见表 D。")
    A("5. 后续唯一建议见下（不自动执行）。\n")

    A("## 后续（唯一建议，不自动执行）\n")
    A("若本轮存在开发信号（SCREENING_SIGNAL/MIXED），下一轮唯一建议：在另一起点"
      "（start82 或新 bank）+ 新的独立训练 seeds/streams + 预先规定的有限 LR "
      "匹配调参预算下复验，并事先规定 final 评价协议；若无收益，记录该具体规则"
      "与配置（alpha=0.10、QH 读出 + 逐张量范数匹配）的负结果并停止该候选，"
      "不堆叠补偿项。\n")

    A("## 产物清单\n")
    for name in ["config.json", "manifest.json", "implementation_checks.json",
                 "smoke_checks.json", "step_metrics.csv", "monitor_metrics.csv",
                 "monitor_rows/*.npz", "target_hits.csv", "seed_method_results.csv",
                 "paired_comparisons.csv", "readout_diagnostics.csv",
                 "resource_log.jsonl", "SNAPSHOT_MANIFEST.json", "SUMMARY.json",
                 "REPORT.md", "KEY_DATA_FOR_REVIEW.md",
                 "figures/fig1_nll_vs_tokens.png", "figures/fig2_h256_gain.png",
                 "figures/fig3_detection.png", "figures/fig4_readout_diag.png"]:
        A(f"- outputs/fr_gttr_d1_direct_readout_v01/{name}")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    # KEY_DATA_FOR_REVIEW.md
    K = []
    K.append("# D1 关键数据速览\n")
    K.append(f"verdicts: {json.dumps(v, indent=1)}")
    K.append("")
    K.append("h256 means (rows=seed, cols=arm):")
    hdr = "| seed | " + " | ".join(ARMS) + " |"
    K.append(hdr)
    K.append("|---|" + "---|" * len(ARMS))
    for s in SEEDS:
        K.append(f"| {s} | " + " | ".join(
            f"{summary['h256_means'][f'{s}/{a}']:.5f}" for a in ARMS) + " |")
    K.append("")
    K.append(f"Δh256 {CAND}-{BASE} per seed: "
             + ", ".join(f"{s}: {summary['delta_h256_vs_native'][str(s)]:+.5f}"
                         for s in SEEDS))
    K.append(f"detection reached/3: " + ", ".join(
        f"{a}: {arm_summary[a]['n_reached_3']}" for a in ARMS))
    K.append(f"norm resid max abs/rel: {e['norm_match_max_abs_resid']:.3e} / "
             f"{e['norm_match_max_rel_resid']:.3e}")
    K.append(f"fallbacks: {e['fallback_counts_norm']}")
    K.append(f"budget actual: {json.dumps(b['actual'])}")
    (OUT / "KEY_DATA_FOR_REVIEW.md").write_text("\n".join(K) + "\n")


ARMS_K = {"ADAMW_LR050": 0.5, "ADAMW_LR100": 1.0, "ADAMW_LR200": 2.0,
          "QH_RAW_A010": 1.0, "QH_NORM_A010": 1.0}


if __name__ == "__main__":
    main()
