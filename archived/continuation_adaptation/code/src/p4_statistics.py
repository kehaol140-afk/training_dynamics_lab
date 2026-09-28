"""p4_statistics — P4 冻结统计口径的纯函数实现（docs/P4_PROTOCOL.md §7）。

primary 比较：d(cell) = NLL_final(Cx) − NLL_final(A_best)（改善 < 0），
族 = {C1, C2}，Holm α=0.05。
- 同 seed 的 7 个 checkpoint 同轨迹且续训流重叠 → cell 间相关，
  primary 推断按 seed 聚类（8 簇）；
- primary 检验：seed 级均值 d̄_s 的 sign-flip 置换（单侧，改善方向），
  20000 次，重采样 RNG 种子 77080001，p = (count+1)/(reps+1)；
- primary CI：按 seed 聚类 bootstrap（10000 次，percentile）；
- naive cell-level bootstrap 仅描述性（标注反保守）。

本模块只做统计，不做任何训练/评测。
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

PERMUTATION_REPS = 20000
BOOTSTRAP_REPS = 10000
STATS_RNG_SEED = 77080001
FAMILY_ALPHA = 0.05
PRIMARY_ARMS = ["C1", "C2"]
SEED_CONSISTENCY_MIN = 6
CHECKPOINT_CONSISTENCY_MIN = 5


def paired_diffs(table: Dict[str, Dict[str, float]], arm: str,
                 a_best_key: str = "A_best") -> Dict[str, float]:
    """table: {cell_id: {arm_name: final_nll}} → {cell_id: d}。"""
    out = {}
    for cell_id, row in table.items():
        if arm not in row or a_best_key not in row:
            raise KeyError(f"cell {cell_id} 缺 {arm} 或 {a_best_key}")
        out[cell_id] = float(row[arm]) - float(row[a_best_key])
    return out


def _cell_meta_index(cells: Sequence[Dict]) -> Dict[str, Tuple[int, int]]:
    return {c["cell_id"]: (int(c["seed"]), int(c["state_optimizer_step"])) for c in cells}


def seed_cluster_means(d: Dict[str, float], cell_meta: Dict[str, Tuple[int, int]]) -> Dict[int, float]:
    by_seed: Dict[int, List[float]] = {}
    for cell_id, val in d.items():
        seed = cell_meta[cell_id][0]
        by_seed.setdefault(seed, []).append(float(val))
    return {s: sum(v) / len(v) for s, v in by_seed.items()}


def checkpoint_means(d: Dict[str, float], cell_meta: Dict[str, Tuple[int, int]]) -> Dict[int, float]:
    by_step: Dict[int, List[float]] = {}
    for cell_id, val in d.items():
        step = cell_meta[cell_id][1]
        by_step.setdefault(step, []).append(float(val))
    return {t: sum(v) / len(v) for t, v in by_step.items()}


def seed_cluster_permutation_test(d: Dict[str, float], cell_meta: Dict[str, Tuple[int, int]],
                                  *, reps: int = PERMUTATION_REPS,
                                  rng_seed: int = STATS_RNG_SEED) -> Dict:
    """单侧（改善方向 d<0）sign-flip 置换检验，翻转单位 = seed 簇。"""
    means = seed_cluster_means(d, cell_meta)
    values = [means[s] for s in sorted(means)]
    observed = sum(values) / len(values)

    rng = random.Random(rng_seed)
    count = 0
    for _ in range(reps):
        flipped = [v if rng.random() < 0.5 else -v for v in values]
        if sum(flipped) / len(flipped) <= observed:
            count += 1
    p = (count + 1) / (reps + 1)
    return {"observed_mean": observed, "seed_means": dict(sorted(means.items())),
            "p_value_one_sided": p, "reps": reps, "rng_seed": rng_seed}


def cluster_bootstrap_ci(d: Dict[str, float], cell_meta: Dict[str, Tuple[int, int]],
                         *, reps: int = BOOTSTRAP_REPS, rng_seed: int = STATS_RNG_SEED,
                         percentiles: Tuple[float, float] = (2.5, 97.5)) -> Dict:
    """按 seed 簇重采样（有放回）的均值配对差 percentile CI（primary CI）。"""
    seeds = sorted({meta[0] for meta in cell_meta.values()})
    by_seed = {s: [] for s in seeds}
    for cell_id, val in d.items():
        by_seed[cell_meta[cell_id][0]].append(float(val))
    rng = random.Random(rng_seed + 1)
    stats = []
    for _ in range(reps):
        sampled = [by_seed[rng.choice(seeds)] for _ in seeds]
        flat = [v for cluster in sampled for v in cluster]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    lo_idx = int(percentiles[0] / 100 * reps)
    hi_idx = int(percentiles[1] / 100 * reps) - 1
    return {"ci_low": stats[lo_idx], "ci_high": stats[hi_idx], "reps": reps,
            "percentiles": list(percentiles), "level": "primary_cluster_bootstrap_by_seed"}


def naive_bootstrap_ci(d: Dict[str, float], *, reps: int = BOOTSTRAP_REPS,
                       rng_seed: int = STATS_RNG_SEED) -> Dict:
    """cell 级 naive bootstrap（忽略 seed 内相关，反保守，仅描述性）。"""
    values = sorted(d.values())
    n = len(values)
    rng = random.Random(rng_seed + 2)
    stats = []
    for _ in range(reps):
        stats.append(sum(rng.choice(values) for _ in range(n)) / n)
    stats.sort()
    return {"ci_low": stats[int(0.025 * reps)], "ci_high": stats[int(0.975 * reps) - 1],
            "reps": reps, "level": "naive_cell_bootstrap_ANTI_CONSERVATIVE"}


def holm_correction(p_values: Dict[str, float], *, alpha: float = FAMILY_ALPHA) -> Dict[str, Dict]:
    """Holm 校正：按 p 升序逐个比较 p_(i) <= alpha/(m-i)。"""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    results, rejected_any_prior = {}, False
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        significant = (not rejected_any_prior) and p <= threshold
        if not significant:
            rejected_any_prior = True
        results[name] = {"p_value": p, "holm_threshold": threshold,
                         "significant": bool(significant)}
    return results


def evaluate_variant(d: Dict[str, float], cell_meta: Dict[str, Tuple[int, int]],
                     *, reps_perm: int = PERMUTATION_REPS, reps_boot: int = BOOTSTRAP_REPS,
                     rng_seed: int = STATS_RNG_SEED) -> Dict:
    """单个 C 变体的完整冻结统计（不含 Holm——Holm 在两变体间做）。"""
    perm = seed_cluster_permutation_test(d, cell_meta, reps=reps_perm, rng_seed=rng_seed)
    ci = cluster_bootstrap_ci(d, cell_meta, reps=reps_boot, rng_seed=rng_seed)
    naive = naive_bootstrap_ci(d, reps=reps_boot, rng_seed=rng_seed)
    seed_means = perm["seed_means"]
    ck_means = checkpoint_means(d, cell_meta)
    values = list(d.values())
    return {
        "aggregate_mean_diff": sum(values) / len(values),
        "aggregate_median_diff": sorted(values)[len(values) // 2],
        "std_diff": (sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1)) ** 0.5,
        "n_cells": len(values),
        "permutation": perm,
        "cluster_bootstrap_ci": ci,
        "naive_bootstrap_ci": naive,
        "seeds_improved": sum(1 for v in seed_means.values() if v < 0),
        "n_seeds": len(seed_means),
        "checkpoints_improved": sum(1 for v in ck_means.values() if v < 0),
        "n_checkpoints": len(ck_means),
        "per_seed_means": seed_means,
        "per_checkpoint_means": dict(sorted(ck_means.items())),
    }


def evaluate_gates(variant_stats: Dict[str, Dict], holm: Dict[str, Dict]) -> Dict:
    """冻结停止条件（§8）：继续条件 1–4 全满足（5 由扩张机制结构性满足）。

    variant_stats: {arm: evaluate_variant 输出}；holm: {arm: {...significant...}}。
    任一 C 变体通过全部 gate → continue_direction=True。
    """
    results = {}
    for arm, st in variant_stats.items():
        checks = {
            "1_aggregate_mean_diff_negative": st["aggregate_mean_diff"] < 0,
            "2_seed_consistency_ge_6of8": st["seeds_improved"] >= SEED_CONSISTENCY_MIN,
            "3_checkpoint_consistency_ge_5of7": st["checkpoints_improved"] >= CHECKPOINT_CONSISTENCY_MIN,
            "4_holm_significant": bool(holm[arm]["significant"]),
        }
        results[arm] = {"gate_checks": checks,
                        "all_gates_passed": all(checks.values()),
                        "holm_p": holm[arm]["p_value"]}
    continue_direction = any(r["all_gates_passed"] for r in results.values())
    return {"continue_direction": continue_direction, "per_variant": results,
            "stop_rule_note": "frozen §8: any listed stop condition triggers STOP; "
                              "gate 5 (grid robustness) is structurally satisfied by "
                              "the automatic single boundary expansion in stage 1a"}
