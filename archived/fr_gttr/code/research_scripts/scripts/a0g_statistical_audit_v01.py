#!/usr/bin/env python3
"""FR_A0G 分层统计审计（§七）——纯 CPU/离线，绝不重跑科学实验。

输入: outputs/fr_decision_utility_v01/level3_decision_metrics.csv（formal split，
seeds 2038–2041 × steps {46,64,82} × 8 streams，stream-level n=96）。

核对对象: H∈{4,8} × 指标 {Gain, Regret, harmful rate} × 对比
{C vs 随机(random), C vs 当前梯度对齐(alignment), C vs 仅优化器切线(B)}。

三种口径:
  1. stream-level（n=96）：配对差逐 stream，t 检验 + Wilcoxon 符号秩；
     仅作 conditional-stream inference。
  2. state-level（n=12）：每 state 8 streams 先平均，再配对；
  3. seed-level（n=4）：每 seed 3 states 再平均，n 过小只报告效应方向/幅度。

hierarchical bootstrap（≥10000 次）: 有放回重采样 seed → state within seed →
stream within state，输出 mean effect / 95% CI / P(effect>0) 与 P(effect<0)。
bootstrap 不制造虚假独立 n，只给效应量与不确定度。

效应方向约定: delta = C − baseline。
  Gain: delta>0 有利; Regret: delta<0 有利; harmful rate: delta<0 有利。
输出: statistical_audit/ 下 state_level_metrics.csv, seed_level_metrics.csv,
hierarchical_bootstrap.csv, STREAM_LEVEL_AUDIT(并入 md), STATISTICAL_AUDIT.md,
SHA256SUMS.txt。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "outputs/fr_decision_utility_v01"
AUD = OUT / "statistical_audit"
L3 = OUT / "level3_decision_metrics.csv"

H_LIST = (4, 8)
METRICS = ("gain", "regret", "harmful")
BASELINES = ("random", "alignment", "B")
N_BOOT = 10000
SEED = 20260914


# ---------------------------------------------------------------------------
# 读取与整形
# ---------------------------------------------------------------------------

def load():
    """返回 nested dict: data[H][method][(seed,step,stream)] = float metric row"""
    data = {h: {m: {} for m in ("C",) + tuple(BASELINES)} for h in H_LIST}
    seeds, steps = set(), set()
    with L3.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "formal":
                continue
            h = int(r["H"])
            if h not in H_LIST:
                continue
            m = r["method"]
            key = (int(r["seed"]), int(r["step"]), int(r["stream_id"]))
            seeds.add(int(r["seed"]))
            steps.add(int(r["step"]))
            for metric in METRICS:
                pass
            data[h][m][key] = {
                "gain": float(r["gain"]),
                "regret": float(r["regret"]),
                "harmful": 1.0 if r["harmful"].strip().lower() == "true" else 0.0,
            }
    return data, sorted(seeds), sorted(steps)


def paired_delta(data, h, metric, base, level):
    """level: 'stream'|'state'|'seed'。返回 (keys, delta 数组) 及每 key 的双方均值。"""
    c = data[h]["C"]
    b = data[h][base]

    def agg_key(k):
        seed, step, stream = k
        if level == "stream":
            return k
        if level == "state":
            return (seed, step)
        return (seed,)

    groups_c, groups_b = {}, {}
    for k, v in c.items():
        groups_c.setdefault(agg_key(k), []).append(v[metric])
    for k, v in b.items():
        groups_b.setdefault(agg_key(k), []).append(v[metric])
    keys = sorted(set(groups_c) & set(groups_b))
    dc = np.array([np.mean(groups_c[k]) for k in keys])
    db = np.array([np.mean(groups_b[k]) for k in keys])
    return keys, dc - db, dc, db


# ---------------------------------------------------------------------------
# 小样本推断
# ---------------------------------------------------------------------------

def t_test(delta):
    n = len(delta)
    if n < 2:
        return None, None
    se = delta.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return (float("inf") if delta.mean() > 0 else
                (float("-inf") if delta.mean() < 0 else 0.0)), 0.0 if delta.mean() != 0 else 1.0
    t = delta.mean() / se
    # 正态近似双侧 p（n 小仅作参考，主结论看 bootstrap）
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return t, p


def wilcoxon(delta):
    """Wilcoxon 符号秩（去零），正态近似带连续性修正。n<6 返回 None。"""
    d = delta[delta != 0]
    n = len(d)
    if n < 6:
        return None, None
    r = np.argsort(np.argsort(np.abs(d))) + 1  # 秩（无并列处理，连续量足够）
    w_plus = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu - 0.5 * np.sign(w_plus - mu)) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return z, p


# ---------------------------------------------------------------------------
# hierarchical bootstrap
# ---------------------------------------------------------------------------

def hier_bootstrap(data, h, metric, base, rng):
    """seed → state → stream 三层有放回。返回 effects 数组 (N_BOOT,)。"""
    c, b = data[h]["C"], data[h][base]
    seeds = sorted({k[0] for k in c})
    tree = {}
    for k, v in c.items():
        tree.setdefault(k[0], {}).setdefault(k[1], {})[k[2]] = v[metric]
    tree_b = {}
    for k, v in b.items():
        tree_b.setdefault(k[0], {}).setdefault(k[1], {})[k[2]] = v[metric]
    effects = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        acc_c, acc_b, n_pts = [], [], 0
        for s in pick_seeds:
            states = sorted(tree[s])
            pick_states = rng.choice(states, size=len(states), replace=True)
            for st in pick_states:
                streams = sorted(tree[s][st])
                pick_streams = rng.choice(streams, size=len(streams), replace=True)
                for sm in pick_streams:
                    acc_c.append(tree[s][st][sm])
                    acc_b.append(tree_b[s][st][sm])
                    n_pts += 1
        effects[i] = float(np.mean(acc_c)) - float(np.mean(acc_b))
    return effects


def summarize_boot(effects):
    lo, hi = np.percentile(effects, [2.5, 97.5])
    return {
        "mean": float(effects.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "P_gt_0": float((effects > 0).mean()),
        "P_lt_0": float((effects < 0).mean()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    AUD.mkdir(parents=True, exist_ok=True)
    data, seeds, steps = load()
    rng = np.random.default_rng(SEED)

    good_dir = {"gain": ">0", "regret": "<0", "harmful": "<0"}

    # ---- state-level / seed-level CSV ----
    state_rows, seed_rows = [], []
    stream_rows = []
    boot_rows = []
    md_blocks = []

    for h in H_LIST:
        md_blocks.append(f"### H={h}")
        header = ("| 对比 | 指标 | stream n=96 meanΔ | t | p(近似) | "
                  "Wilcoxon z | p | state n=12 meanΔ | t | seed n=4 meanΔ | "
                  "boot meanΔ [95%CI] | P(Δ>0) | P(Δ<0) | 有利方向 |")
        md_blocks.append(header)
        md_blocks.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
        for base in BASELINES:
            for metric in METRICS:
                _, d_stream, _, _ = paired_delta(data, h, metric, base, "stream")
                t_s, p_s = t_test(d_stream)
                z_s, pw_s = wilcoxon(d_stream)
                _, d_state, _, _ = paired_delta(data, h, metric, base, "state")
                t_st, p_st = t_test(d_state)
                _, d_seed, mc, mb = paired_delta(data, h, metric, base, "seed")
                fx = hier_bootstrap(data, h, metric, base, rng)
                bs = summarize_boot(fx)
                boot_rows.append({
                    "H": h, "comparison": f"C_vs_{base}", "metric": metric,
                    "n_boot": N_BOOT, **bs,
                    "beneficial_direction": good_dir[metric],
                })
                # 明细 CSV
                keys_s, ds, cs, bsmt = paired_delta(data, h, metric, base, "state")
                for k, dv, cv, bv in zip(keys_s, ds, cs, bsmt):
                    state_rows.append({
                        "H": h, "comparison": f"C_vs_{base}", "metric": metric,
                        "seed": k[0], "step": k[1],
                        "mean_C": cv, "mean_baseline": bv, "delta": dv,
                    })
                keys_g, dg, cg, bg = paired_delta(data, h, metric, base, "seed")
                for k, dv, cv, bv in zip(keys_g, dg, cg, bg):
                    seed_rows.append({
                        "H": h, "comparison": f"C_vs_{base}", "metric": metric,
                        "seed": k[0],
                        "mean_C": cv, "mean_baseline": bv, "delta": dv,
                    })
                keys_t, dt, ct, bt = paired_delta(data, h, metric, base, "stream")
                for k, dv, cv, bv in zip(keys_t, dt, ct, bt):
                    stream_rows.append({
                        "H": h, "comparison": f"C_vs_{base}", "metric": metric,
                        "seed": k[0], "step": k[1], "stream": k[2],
                        "C": cv, "baseline": bv, "delta": dv,
                    })
                fmt = lambda v: "—" if v is None else f"{v:.3e}"
                fmt2 = lambda v: "—" if v is None else f"{v:.4f}"
                md_blocks.append(
                    f"| C vs {base} | {metric} | {fmt(d_stream.mean())} | "
                    f"{fmt2(t_s)} | {fmt2(p_s)} | {fmt2(z_s)} | {fmt2(pw_s)} | "
                    f"{fmt(d_state.mean())} | {fmt2(t_st)} | {fmt(d_seed.mean())} | "
                    f"{fmt(bs['mean'])} [{fmt(bs['ci95_lo'])}, {fmt(bs['ci95_hi'])}] | "
                    f"{fmt2(bs['P_gt_0'])} | {fmt2(bs['P_lt_0'])} | {good_dir[metric]} |"
                )
        md_blocks.append("")

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(AUD / "state_level_metrics.csv", state_rows)
    write_csv(AUD / "seed_level_metrics.csv", seed_rows)
    write_csv(AUD / "hierarchical_bootstrap.csv", boot_rows)
    write_csv(AUD / "stream_level_pairs.csv", stream_rows)

    md = [
        "# STATISTICAL AUDIT — FR_DECISION_UTILITY_V01（分层统计审计）",
        "",
        "纯离线审计：只读 `level3_decision_metrics.csv`（formal，4 seeds × 3 states × "
        "8 streams），不重跑任何科学实验。",
        "",
        f"- 对比: 完整切线 C vs {', '.join(BASELINES)}",
        f"- 指标: Gain（delta>0 有利）、Regret（delta<0 有利）、harmful rate（delta<0 有利）",
        "- 效应定义: delta = C − baseline（配对，同一 state/stream/H）",
        f"- hierarchical bootstrap: {N_BOOT} 次，重采样层次 seed → state within seed → "
        "stream within state；不制造虚假独立 n，只报告效应量与不确定度",
        "- stream-level 推断仅作 conditional-stream inference；seed-level n=4 只看方向与幅度",
        "",
        "## 汇总表",
        "",
        *md_blocks,
        "## 说明",
        "",
        "- t/Wilcoxon 的 p 为近似值（正态近似），主结论以 bootstrap CI 与 P(Δ 符号) 为准；",
        "- `state_level_metrics.csv` / `seed_level_metrics.csv` / `stream_level_pairs.csv` "
        "为配对明细；`hierarchical_bootstrap.csv` 为 bootstrap 汇总；",
        "- 层级聚合均为算术平均（与主报告 mean Gain/Regret/harmful rate 口径一致）。",
        "",
    ]
    (AUD / "STATISTICAL_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    entries = []
    for path in sorted(AUD.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (AUD / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    # 关键数字给主报告用
    key = {}
    for row in boot_rows:
        key[f"H{row['H']}_{row['comparison']}_{row['metric']}"] = row
    print(json.dumps({
        "n_stream": len(stream_rows) // (len(H_LIST) * len(BASELINES) * len(METRICS)),
        "boot": {k: {"mean": v["mean"], "ci95": [v["ci95_lo"], v["ci95_hi"]],
                         "P_gt0": v["P_gt_0"]}
                 for k, v in key.items()},
    }, ensure_ascii=False, indent=1))
    print("AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
