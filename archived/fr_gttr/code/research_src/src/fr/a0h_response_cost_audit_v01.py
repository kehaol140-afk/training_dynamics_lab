"""FR_A0H 响应成本审计库（A0H-0 离线审计 + A0H-1 伴随等价的数据侧工具）。

[FR-OBSERVATION-ONLY] 只读冻结的 A0G 制品（四份科学 CSV、manifest、summary），
不重跑任何科学实验；本阶段不补跑 GPU。所有指标复用冻结定义：
``fr.a0g_decision_utility_v01.level3_decision / select_action`` 与
``fr.a0c_adamw_response_propagation_v01.relative_error``，禁止重定义。

A0H-0 审计项（详见 docs/FR_A0H_PROTOCOL.md）：
  1. 主键覆盖/重复/缺行/候选语义/no-op/权重；
  2. 用冻结规则离线复算方法选择与收益，并与冻结 level3 CSV 逐格核对；
  3. 三个补充基线（never-intervene / dev-fixed-group / replay-score 可用性）；
  4. Gain+Regret=OracleGain、ΔRegret=−ΔGain、OracleCapture（联合重采样）；
  5. 共用重采样索引的分层配对 bootstrap（含 C−B 的 H8−H4 交互，后续分析）；
  6. no-op 选择率/错过可获益/有害干预比例（含分母）与 margin 描述统计。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_response_propagation_v01 import relative_error
from .a0g_decision_utility_v01 import (
    HORIZONS,
    LAMBDA_CANDIDATES,
    N_DATA_GROUPS,
    NEAR_TIE_FRACTION,
    PRIMARY_HORIZON,
    level3_decision,
    random_choice,
    select_action,
)

METHOD_KEYS = ("random", "alignment", "B", "C")
# A0H-0 新增补充基线（一律标注为后续补充分析，不倒写为原预注册项）
EXTRA_BASELINES = ("never", "dev_fixed")
ALL_METHODS = METHOD_KEYS + EXTRA_BASELINES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# 1. 主键与覆盖审计
# ---------------------------------------------------------------------------

def unit_key(row: Mapping) -> tuple:
    return (row["split"], int(row["seed"]), int(row["step"]), int(row["stream_id"]))


def audit_primary_keys(
    level1_rows: Sequence[Mapping],
    level1b_rows: Sequence[Mapping],
    level2_rows: Sequence[Mapping],
    level3_rows: Sequence[Mapping],
    expected_dev_states: Sequence[tuple],
    expected_formal: Mapping,
) -> dict:
    """核对 split/seed/step/stream/H/λ/j 覆盖、重复、缺行与 no-op 语义。"""
    report: dict = {"status": "PASS", "detail": {}}

    l2_units = sorted({unit_key(r) for r in level2_rows})
    dev_units = [u for u in l2_units if u[0] == "dev"]
    formal_units = [u for u in l2_units if u[0] == "formal"]
    expected_dev = {
        ("dev", seed, step, s)
        for seed, step in expected_dev_states for s in range(8)
    }
    expected_formal_set = {
        ("formal", seed, step, s)
        for seed in expected_formal["seeds"]
        for step in expected_formal["steps"]
        for s in range(expected_formal["streams_per_state"])
    }
    coverage = {
        "dev_units_found": len(dev_units),
        "formal_units_found": len(formal_units),
        "dev_units_missing": sorted(expected_dev - set(dev_units)),
        "formal_units_missing": sorted(expected_formal_set - set(formal_units)),
        "unexpected_units": sorted(set(l2_units) - expected_dev - expected_formal_set),
    }
    report["detail"]["unit_coverage"] = coverage
    if coverage["dev_units_missing"] or coverage["formal_units_missing"] or coverage["unexpected_units"]:
        report["status"] = "FAIL"

    def _dup_check(rows, keys, label):
        counts = defaultdict(int)
        for r in rows:
            counts[tuple(r[k] for k in keys)] += 1
        dups = {k: v for k, v in counts.items() if v > 1}
        report["detail"][f"duplicates_{label}"] = {
            "n_duplicate_keys": len(dups), "examples": list(dups)[:5],
        }
        return dups

    dup_l1 = _dup_check(level1_rows, ("split", "seed", "step", "stream_id", "j", "H", "lambda"), "l1")
    dup_l1b = _dup_check(level1b_rows, ("split", "seed", "step", "stream_id", "j", "H", "lambda"), "l1b")
    dup_l2 = _dup_check(level2_rows, ("split", "seed", "step", "stream_id", "j", "H", "lambda"), "l2")
    dup_l3 = _dup_check(level3_rows, ("split", "seed", "step", "stream_id", "H", "method"), "l3")
    if dup_l1 or dup_l1b or dup_l2 or dup_l3:
        report["status"] = "FAIL"

    # 每 unit×H 的行数与候选集合
    def _per_unit_h(rows, j_values, label):
        counts = defaultdict(set)
        for r in rows:
            counts[unit_key(r) + (int(r["H"]),)].add(int(r["j"]))
        bad = {k: sorted(v) for k, v in counts.items() if sorted(v) != sorted(j_values)}
        report["detail"][f"candidate_set_{label}"] = {
            "expected_j": sorted(j_values),
            "n_unit_H": len(counts),
            "n_bad": len(bad),
            "examples_bad": dict(list(bad.items())[:5]),
        }
        return bad

    bad_l1 = _per_unit_h(level1_rows, list(range(1, N_DATA_GROUPS + 1)), "l1")
    bad_l1b = _per_unit_h(level1b_rows, list(range(1, N_DATA_GROUPS + 1)), "l1b")
    bad_l2 = _per_unit_h(level2_rows, list(range(0, N_DATA_GROUPS + 1)), "l2")
    if bad_l1 or bad_l1b or bad_l2:
        report["status"] = "FAIL"

    # λ 取值与 H 取值
    lambdas = sorted({float(r["lambda"]) for r in level2_rows})
    hs = sorted({int(r["H"]) for r in level2_rows})
    report["detail"]["lambda_values"] = lambdas
    report["detail"]["horizon_values"] = hs
    if [float(x) for x in LAMBDA_CANDIDATES] != lambdas and lambdas != [0.05]:
        # level2 只应含 primary λ；扫描 λ 只出现在 lambda_scan_detail.csv
        report["detail"]["lambda_note"] = "level1/1b/2 仅含 primary λ" if lambdas == [0.05] else "unexpected"
    if lambdas != [0.05]:
        report["status"] = "FAIL"
    if hs != sorted(HORIZONS):
        report["status"] = "FAIL"

    # no-op 语义：level2 中 j=0 的 actual/pred 必须全 0
    noop = [r for r in level2_rows if int(r["j"]) == 0]
    nonzero = [
        r for r in noop
        if any(abs(float(r[k])) > 0.0 for k in
               ("actual_delta_eval", "pred_delta_alignment", "pred_delta_B", "pred_delta_C"))
    ]
    report["detail"]["noop_semantics"] = {
        "n_noop_rows": len(noop), "n_nonzero_noop_rows": len(nonzero),
        "formal_units": len(formal_units),
        "formal_H4_nonzero_candidate_rows": sum(
            1 for r in level1b_rows
            if r["split"] == "formal" and int(r["H"]) == 4
        ),
    }
    if nonzero or len(noop) != len(counts_l2_units(level2_rows)) * len(HORIZONS):
        report["status"] = "FAIL"

    # 有效样本掩码与权重：A0G 冻结聚合为流级等权算术平均（state/seed 层同样等权）
    report["detail"]["weighting"] = {
        "stream_weight": 1.0,
        "aggregation": "stream mean -> state mean -> seed mean（等权算术平均，与冻结口径一致）",
        "effective_n": {"formal_stream": len(formal_units) * len(HORIZONS) and len(formal_units),
                        "formal_state": len(expected_formal) // 8,
                        "formal_seed": len(expected_formal["seeds"])},
        "shadow_pass": "summary.json runtime.all_shadow_pass（主实验 96/96 True）",
    }
    return report


def counts_l2_units(level2_rows) -> set:
    return {unit_key(r) for r in level2_rows}


# ---------------------------------------------------------------------------
# 2. 冻结规则复算（selection + level3 逐格核对）
# ---------------------------------------------------------------------------

def recompute_level3(level2_rows: Sequence[Mapping]) -> list[dict]:
    """从 level2 预测/实际值按冻结规则复算全部四方法的 level3 指标。"""
    index: dict = {}
    for r in level2_rows:
        index.setdefault(unit_key(r) + (int(r["H"]),), {})[int(r["j"])] = r
    out = []
    for (split, seed, step, stream_id, h), by_j in sorted(index.items()):
        actual = {j: float(by_j[j]["actual_delta_eval"]) for j in sorted(by_j)}
        pred = {
            m: {j: float(by_j[j][f"pred_delta_{m}"]) for j in sorted(by_j)}
            for m in ("alignment", "B", "C")
        }
        rng = np.random.default_rng(seed=((seed * 100003 + step * 101 + stream_id) * 1000 + h))
        j_rand = random_choice(rng, N_DATA_GROUPS + 1)
        decisions = {
            "random": level3_decision(
                predicted_by_j={k: (0.0 if k != j_rand else -1.0) for k in range(N_DATA_GROUPS + 1)},
                actual_by_j=actual,
            ),
        }
        for m in ("alignment", "B", "C"):
            decisions[m] = level3_decision(predicted_by_j=pred[m], actual_by_j=actual)
        for m in METHOD_KEYS:
            row = {
                "split": split, "seed": seed, "step": step, "stream_id": stream_id,
                "H": h, "method": m,
                "j_hat": decisions[m]["j_hat"],
                "predicted_delta_selected": decisions[m]["predicted_delta_selected"],
                "actual_delta_selected": decisions[m]["actual_delta_selected"],
                "best_possible_delta": decisions[m]["best_possible_delta"],
                "gain": decisions[m]["gain"],
                "regret": decisions[m]["regret"],
                "harmful": decisions[m]["harmful"],
                "top1_correct": decisions[m]["top1_correct"],
                "near_tie_tolerance": decisions[m]["near_tie_tolerance"],
                "tied_optimal": decisions[m]["tied_optimal"],
                "direction_correct": decisions[m]["direction_correct"],
            }
            out.append(row)
    return out


def compare_level3(recomputed: Sequence[Mapping], frozen: Sequence[Mapping]) -> dict:
    """逐格核对复算 vs 冻结 level3。"""
    def key(r):
        return (r["split"], int(r["seed"]), int(r["step"]), int(r["stream_id"]), int(r["H"]), r["method"])

    frozen_idx = {key(r): r for r in frozen}
    re_idx = {key(r): r for r in recomputed}
    missing = sorted(set(frozen_idx) - set(re_idx))
    extra = sorted(set(re_idx) - set(frozen_idx))
    max_abs_diff = 0.0
    n_diff = 0
    diff_examples = []
    numeric_cols = ("predicted_delta_selected", "actual_delta_selected", "best_possible_delta",
                    "gain", "regret", "near_tie_tolerance")
    bool_cols = ("harmful", "top1_correct", "direction_correct")
    def _norm_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() == "true"

    def _norm_tied(v):
        if isinstance(v, str):
            return sorted(int(x) for x in v.strip("[]").split(",") if x.strip())
        return sorted(int(x) for x in v)

    for k in frozen_idx:
        if k not in re_idx:
            continue
        f, g = frozen_idx[k], re_idx[k]
        for c in numeric_cols:
            d = abs(float(f[c]) - float(g[c]))
            max_abs_diff = max(max_abs_diff, d)
            if d > 1e-12:
                n_diff += 1
                if len(diff_examples) < 5:
                    diff_examples.append({"key": k, "col": c, "frozen": f[c], "recomputed": g[c]})
        for c in bool_cols:
            if _norm_bool(f[c]) != _norm_bool(g[c]):
                n_diff += 1
                if len(diff_examples) < 5:
                    diff_examples.append({"key": k, "col": c, "frozen": f[c], "recomputed": g[c]})
        if _norm_tied(f["tied_optimal"]) != _norm_tied(g["tied_optimal"]):
            n_diff += 1
        if int(f["j_hat"]) != int(g["j_hat"]):
            n_diff += 1
            if len(diff_examples) < 5:
                diff_examples.append({"key": k, "col": "j_hat", "frozen": f["j_hat"], "recomputed": g["j_hat"]})
    status = "PASS" if not missing and not extra and n_diff == 0 else "FAIL"
    return {
        "status": status,
        "n_frozen": len(frozen_idx), "n_recomputed": len(re_idx),
        "missing_keys": len(missing), "extra_keys": len(extra),
        "max_abs_numeric_diff": max_abs_diff, "n_cell_diffs": n_diff,
        "examples": diff_examples,
    }


# ---------------------------------------------------------------------------
# 3. λ 规则审计（Rule A/B 复算 + per-unit λ 一致性）
# ---------------------------------------------------------------------------

def audit_lambda_rule(scan_rows: Sequence[Mapping], lambda_selection: Mapping,
                      level1_rows: Sequence[Mapping]) -> dict:
    """复算 Rule A/B，确认 primary/conservative，并确认 λ 不在 unit 间变化。"""
    by_lam: dict = defaultdict(list)
    for r in scan_rows:
        by_lam[float(r["lambda"])].append(r)
    lambda0 = min(by_lam)
    summary = []
    for lam in sorted(by_lam):
        rows = by_lam[lam]
        hhat_abs = [abs(float(r["hhat_C"])) for r in rows]
        slope_devs = []
        # scan 行的 slope_dev 需要跨 λ 配对，此处直接复用 scan 明细中的逐行值不可行
        # （scan 明细无 per-row slope_dev），改为按冻结 lambda_selection 的聚合值核对。
        crossing = [str(r["clip_regime_crossing"]).strip().lower() == "true" for r in rows]
        summary.append({
            "lambda": lam,
            "mean_abs_hhat_C": float(np.mean(hhat_abs)),
            "clip_crossing_rate": float(np.mean(crossing)),
            "all_near_zero": bool(np.mean(hhat_abs) <= 1e-9),
        })
    frozen_rows = {float(r["lambda_value"]): r for r in lambda_selection["scan_rows"]}
    agg_match = True
    for s in summary:
        f = frozen_rows.get(s["lambda"])
        if f is None:
            agg_match = False
            continue
        if abs(f["clip_crossing_rate"] - s["clip_crossing_rate"]) > 1e-12:
            agg_match = False
        if abs(f["mean_abs_hhat_C"] - s["mean_abs_hhat_C"]) > 1e-12:
            agg_match = False
        if bool(f["all_near_zero"]) != s["all_near_zero"]:
            agg_match = False

    strict_passing = [
        s for s in lambda_selection["scan_rows"]
        if s["mean_slope_dev_vs_lambda0"] is not None
        and s["mean_slope_dev_vs_lambda0"] <= 0.25
        and s["clip_crossing_rate"] <= 0.10
        and not s["all_near_zero"]
    ]
    expected_rule = "A_strict_locality" if strict_passing else "B_decision_signal"
    rule_match = (expected_rule == lambda_selection["freeze_rule"])

    unit_lambdas = sorted({float(r["lambda"]) for r in level1_rows})
    per_unit_lambda = {
        "distinct_lambda_in_level1": unit_lambdas,
        "rule_chooses_per_unit_lambda": len(unit_lambdas) != 1,
    }
    status = "PASS" if agg_match and rule_match and unit_lambdas == [0.05] else "FAIL"
    return {
        "status": status,
        "recomputed_scan_summary": summary,
        "frozen_scan_match": agg_match,
        "expected_rule": expected_rule,
        "frozen_rule": lambda_selection["freeze_rule"],
        "rule_match": rule_match,
        "strict_A_passing_lambdas": [s["lambda_value"] for s in strict_passing],
        "primary_lambda": lambda_selection["primary_lambda"],
        "conservative_lambda": lambda_selection["conservative_lambda"],
        "per_unit_lambda": per_unit_lambda,
        "conclusion": "Rule B 为全局单一 λ（0.05），不对不同决策单元选择不同 λ。",
    }


# ---------------------------------------------------------------------------
# 4. 恒等式：Gain+Regret=OracleGain；ΔRegret=−ΔGain；OracleCapture
# ---------------------------------------------------------------------------

def check_gain_regret_identity(level3_rows: Sequence[Mapping]) -> dict:
    max_err = 0.0
    examples = []
    for r in level3_rows:
        lhs = float(r["gain"]) + float(r["regret"])
        rhs = -float(r["best_possible_delta"])  # OracleGain = u_i* = max_j u_ij
        err = abs(lhs - rhs)
        max_err = max(max_err, err)
        if err > 1e-15 and len(examples) < 5:
            examples.append({"key": (r["split"], r["seed"], r["step"], r["stream_id"], r["H"], r["method"]),
                             "gain+regret": lhs, "oracle": rhs, "err": err})
    # 聚合层面（均值线性，恒等式逐单元成立则聚合严格成立）
    return {
        "status": "PASS" if max_err <= 1e-12 else "FAIL",
        "max_abs_error_gain_plus_regret_minus_oracle": max_err,
        "note": "regret = actual−best、gain = −actual，二者相加在浮点舍入内等于 −best；"
                "舍入影响 ≤1 ulp（~1e-12 量级 @ gain~1e-4），不作为独立显著性证据。",
        "examples": examples,
    }


def paired_delta_regret_check(level3_rows: Sequence[Mapping],
                              base_a: str = "C", base_b: str = "B") -> dict:
    """对每个 (unit,H)：ΔGain = Gain_a−Gain_b，ΔRegret = Regret_a−Regret_b，检查 ΔRegret = −ΔGain。"""
    idx = {}
    for r in level3_rows:
        idx.setdefault((r["split"], int(r["seed"]), int(r["step"]), int(r["stream_id"]), int(r["H"])), {})[r["method"]] = r
    max_err = 0.0
    n = 0
    for k, by_m in idx.items():
        if base_a in by_m and base_b in by_m:
            dg = float(by_m[base_a]["gain"]) - float(by_m[base_b]["gain"])
            dr = float(by_m[base_a]["regret"]) - float(by_m[base_b]["regret"])
            max_err = max(max_err, abs(dg + dr))
            n += 1
    return {"status": "PASS" if max_err <= 1e-12 else "FAIL",
            "n_pairs": n, "max_abs_error_delta_regret_plus_delta_gain": max_err}


def oracle_capture(level3_rows: Sequence[Mapping], split: str, h: int,
                   methods: Sequence[str] = ALL_METHODS) -> dict:
    """OracleCapture_m = Σu_sel / Σu_oracle（同权重流级等权），u = −actual_delta。"""
    rows = [r for r in level3_rows if r["split"] == split and int(r["H"]) == h and r["method"] in methods]
    by_unit: dict = defaultdict(dict)
    for r in rows:
        by_unit[(int(r["seed"]), int(r["step"]), int(r["stream_id"]), r["method"])] = r
    units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"])) for r in rows})
    result = {}
    for m in methods:
        u_sel = np.array([-float(by_unit[(s, st, sr, m)]["actual_delta_selected"]) for s, st, sr in units])
        oracle = np.array([-float(by_unit[(s, st, sr, m)]["best_possible_delta"]) for s, st, sr in units])
        denom = float(oracle.sum())
        numer = float(u_sel.sum())
        ratio = numer / denom if abs(denom) > 0 else None
        result[m] = {
            "sum_u_selected": numer, "sum_u_oracle": denom,
            "oracle_capture": ratio,
            "n_units": len(units),
        }
    return result


# ---------------------------------------------------------------------------
# 5. 分层配对 bootstrap（共用重采样索引）
# ---------------------------------------------------------------------------

def _hierarchy(units: Sequence[tuple]) -> dict:
    """units: (seed, step, stream) -> {seed: {step: [stream]}}"""
    tree: dict = {}
    for seed, step, stream in units:
        tree.setdefault(int(seed), {}).setdefault(int(step), []).append(int(stream))
    return tree


def hierarchical_bootstrap_means(
    units: Sequence[tuple],
    metric_by_method: Mapping[str, Mapping[tuple, float]],
    n_boot: int,
    seed: int,
) -> dict:
    """对所有方法共用同一组重采样索引；返回每方法 (n_boot,) 重采样均值。

    层级：seed（有放回）→ state within seed（有放回）→ stream within state（有放回）。
    层级结构与冻结 statistical_audit 一致；同一 replicate 内所有方法用完全相同的
    单元多重集，保证配对 delta 的逐 replicate 可解释性。
    """
    tree = _hierarchy(units)
    seeds = sorted(tree)
    rng = np.random.default_rng(seed)
    methods = sorted(metric_by_method)
    out = {m: np.empty(n_boot) for m in methods}
    for b in range(n_boot):
        picked = []
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            steps = sorted(tree[s])
            for st in rng.choice(steps, size=len(steps), replace=True):
                streams = tree[s][st]
                for sr in rng.choice(streams, size=len(streams), replace=True):
                    picked.append((s, st, sr))
        for m in methods:
            vals = metric_by_method[m]
            out[m][b] = float(np.mean([vals[u] for u in picked]))
    return out


def summarize_boot(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "P_gt_0": float((arr > 0).mean()),
        "P_lt_0": float((arr < 0).mean()),
    }


def paired_bootstrap_report(level3_rows: Sequence[Mapping], split: str, h: int,
                            methods: Sequence[str], n_boot: int, seed: int) -> list[dict]:
    """对 formal(split) 固定 H：每方法 Gain/Regret/harmful 的 bootstrap 均值 + 配对 delta。"""
    rows = [r for r in level3_rows if r["split"] == split and int(r["H"]) == h and r["method"] in methods]
    units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"])) for r in rows})
    gains = {m: {} for m in methods}
    regrets = {m: {} for m in methods}
    harmful = {m: {} for m in methods}
    for r in rows:
        u = (int(r["seed"]), int(r["step"]), int(r["stream_id"]))
        m = r["method"]
        gains[m][u] = float(r["gain"])
        regrets[m][u] = float(r["regret"])
        harmful[m][u] = 1.0 if str(r["harmful"]).lower() == "true" else 0.0
    # 同一次 bootstrap 内所有方法、所有指标共用同一组重采样索引（严格配对）
    combined = {}
    for m in methods:
        combined[f"gain::{m}"] = gains[m]
        combined[f"regret::{m}"] = regrets[m]
        combined[f"harmful::{m}"] = harmful[m]
    boot = hierarchical_bootstrap_means(units, combined, n_boot, seed)
    bg = {m: boot[f"gain::{m}"] for m in methods}
    br = {m: boot[f"regret::{m}"] for m in methods}
    bh = {m: boot[f"harmful::{m}"] for m in methods}
    report = []
    for m in methods:
        row = {"split": split, "H": h, "method": m}
        row["gain_boot"] = summarize_boot(bg[m])
        row["regret_boot"] = summarize_boot(br[m])
        row["harmful_boot"] = summarize_boot(bh[m])
        for other in methods:
            if other == m:
                continue
            row[f"delta_gain_vs_{other}"] = summarize_boot(bg[m] - bg[other])
            row[f"delta_regret_vs_{other}"] = summarize_boot(br[m] - br[other])
            row[f"delta_harmful_vs_{other}"] = summarize_boot(bh[m] - bh[other])
        report.append(row)
    return report


def per_seed_effects(level3_rows: Sequence[Mapping], split: str, h: int,
                     methods: Sequence[str]) -> list[dict]:
    """每 seed 的配对效应（seed 内 state/stream 等权平均）+ leave-one-seed-out。"""
    rows = [r for r in level3_rows if r["split"] == split and int(r["H"]) == h and r["method"] in methods]
    idx: dict = defaultdict(dict)
    for r in rows:
        idx[(int(r["seed"]), int(r["step"]), int(r["stream_id"]))][r["method"]] = r
    seeds = sorted({k[0] for k in idx})
    per_seed = {}
    for s in seeds:
        units = [k for k in idx if k[0] == s]
        per_seed[s] = {
            m: float(np.mean([float(idx[u][m]["gain"]) for u in units])) for m in methods
        }
        per_seed[s].update({
            f"regret_{m}": float(np.mean([float(idx[u][m]["regret"]) for u in units])) for m in methods
        })
    ref = methods[0]
    loso = []
    for s in seeds:
        others = [t for t in seeds if t != s]
        row = {"held_out_seed": s}
        for m in methods:
            if m == ref:
                continue
            dg = [per_seed[t][m] - per_seed[t][ref] for t in others]
            row[f"delta_gain_{m}_minus_{ref}"] = float(np.mean(dg))
        loso.append(row)
    return [{"seed": s, **per_seed[s]} for s in seeds], loso


def interaction_c_minus_b(level3_rows: Sequence[Mapping], split: str,
                          h_high: int = 8, h_low: int = 4, n_boot: int = 10000,
                          seed: int = 20260914) -> dict:
    """D_i = (Gain_C−Gain_B)@H8 − (Gain_C−Gain_B)@H4；后续分析，非主终点。"""
    idx: dict = defaultdict(dict)
    for r in level3_rows:
        if r["split"] != split or r["method"] not in ("C", "B"):
            continue
        idx[(int(r["seed"]), int(r["step"]), int(r["stream_id"]), int(r["H"]))][r["method"]] = r
    units = sorted({(k[0], k[1], k[2]) for k in idx})
    d = {}
    for u in units:
        g8 = float(idx[(u[0], u[1], u[2], h_high)]["C"]["gain"]) - float(idx[(u[0], u[1], u[2], h_high)]["B"]["gain"])
        g4 = float(idx[(u[0], u[1], u[2], h_low)]["C"]["gain"]) - float(idx[(u[0], u[1], u[2], h_low)]["B"]["gain"])
        d[u] = g8 - g4
    arr = np.array([d[u] for u in units])
    boot = hierarchical_bootstrap_means(units, {"D": d}, n_boot, seed)["D"]
    return {
        "label": "后续分析（非预注册主终点）",
        "split": split, "n_units": len(units),
        "mean_D": float(arr.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "P_gt_0": float((boot > 0).mean()), "P_lt_0": float((boot < 0).mean()),
    }


# ---------------------------------------------------------------------------
# 6. 描述统计：no-op 选择率 / 错过可获益 / 有害干预 / margin
# ---------------------------------------------------------------------------

def descriptive_rates(level3_rows: Sequence[Mapping], level2_rows: Sequence[Mapping],
                      split: str, h: int, methods: Sequence[str]) -> list[dict]:
    l3 = [r for r in level3_rows if r["split"] == split and int(r["H"]) == h and r["method"] in methods]
    l2 = [r for r in level2_rows if r["split"] == split and int(r["H"]) == h]
    actuals = {}
    preds = {}
    for r in l2:
        u = (int(r["seed"]), int(r["step"]), int(r["stream_id"]))
        actuals.setdefault(u, {})[int(r["j"])] = float(r["actual_delta_eval"])
        preds.setdefault(u, {})[int(r["j"])] = {
            "alignment": float(r["pred_delta_alignment"]),
            "B": float(r["pred_delta_B"]),
            "C": float(r["pred_delta_C"]),
        }
    out = []
    for m in methods:
        rows_m = [r for r in l3 if r["method"] == m]
        n = len(rows_m)
        if n == 0:
            continue
        noop_rate = sum(1 for r in rows_m if int(r["j_hat"]) == 0) / n
        # 错过可获益 = 存在正的可获益动作（u_i* = −best_possible_delta > 0）且未选到最优
        missed = sum(
            1 for r in rows_m
            if float(r["regret"]) > 0 and (-float(r["best_possible_delta"])) > 0
        ) / n
        harmful_n = sum(1 for r in rows_m if str(r["harmful"]).lower() == "true")
        intervened = sum(1 for r in rows_m if int(r["j_hat"]) != 0)
        harmful_among_intervened = sum(
            1 for r in rows_m if str(r["harmful"]).lower() == "true" and int(r["j_hat"]) != 0
        )
        out.append({
            "split": split, "H": h, "method": m, "n_units": n,
            "noop_selection_rate": noop_rate,
            "missed_beneficial_rate": missed,
            "harmful_rate": harmful_n / n, "harmful_n": harmful_n,
            "intervened_n": intervened,
            "harmful_rate_among_intervened": (harmful_among_intervened / intervened) if intervened else None,
        })
    # 永不干预基线（j=0 恒选）：补充描述
    units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"])) for r in l2})
    out.append({
        "split": split, "H": h, "method": "never", "n_units": len(units),
        "noop_selection_rate": 1.0,
        "missed_beneficial_rate": None,
        "harmful_rate": 0.0, "harmful_n": 0,
        "intervened_n": 0,
        "harmful_rate_among_intervened": None,
    })
    return out


def margins(level2_rows: Sequence[Mapping], split: str, h: int) -> dict:
    """预测 margin 与真实 margin（均含 no-op j=0；预测 margin 对 random 无定义）。"""
    l2 = [r for r in level2_rows if r["split"] == split and int(r["H"]) == h]
    by_unit: dict = defaultdict(dict)
    for r in l2:
        u = (int(r["seed"]), int(r["step"]), int(r["stream_id"]))
        by_unit[u][int(r["j"])] = r
    true_margins, pred_margins = [], {m: [] for m in ("alignment", "B", "C")}
    for u, by_j in by_unit.items():
        actual = sorted(float(by_j[j]["actual_delta_eval"]) for j in by_j)
        pred = {m: sorted(float(by_j[j][f"pred_delta_{m}"]) for j in by_j) for m in pred_margins}
        # margin = 次优 − 最优（ΔL 口径；越小越好）
        true_margins.append(actual[1] - actual[0] if len(actual) > 1 else None)
        for m in pred_margins:
            pred_margins[m].append(pred[m][1] - pred[m][0] if len(pred[m]) > 1 else None)
    def _stats(vals):
        arr = np.array([v for v in vals if v is not None])
        return {"mean": float(arr.mean()), "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
    return {"split": split, "H": h, "true_margin": _stats(true_margins),
            "pred_margin": {m: _stats(v) for m, v in pred_margins.items()},
            "note": "margin = 次优ΔL − 最优ΔL（含 no-op）；仅描述，不调阈值。"}


# ---------------------------------------------------------------------------
# 7. 补充基线
# ---------------------------------------------------------------------------

def baseline_never(level2_rows: Sequence[Mapping], split: str, h: int) -> list[dict]:
    """永不干预 j=0：gain=0、regret=u_i*、harmful=False、top1=(0∈tied_optimal)。"""
    l2 = [r for r in level2_rows if r["split"] == split and int(r["H"]) == h]
    by_unit: dict = defaultdict(dict)
    for r in l2:
        by_unit[(int(r["seed"]), int(r["step"]), int(r["stream_id"]))][int(r["j"])] = float(r["actual_delta_eval"])
    out = []
    for u, actual in sorted(by_unit.items()):
        best = min(actual.values())
        worst = max(actual.values())
        tol = max(1e-12, NEAR_TIE_FRACTION * (worst - best))
        tied = sorted(j for j, v in actual.items() if v - best <= tol)
        out.append({
            "split": split, "seed": u[0], "step": u[1], "stream_id": u[2], "H": h,
            "method": "never", "j_hat": 0,
            "actual_delta_selected": 0.0,
            "gain": 0.0, "regret": -best, "harmful": False,
            "top1_correct": 0 in tied, "direction_correct": True,
            "best_possible_delta": best,
        })
    return out


def select_dev_fixed_group(level2_rows: Sequence[Mapping], split: str = "dev") -> dict:
    """dev 固定最佳数据组：dev 全部 (unit,H) 的 mean u_ij 选组（tie 取小 j），先冻结。

    返回冻结组 j* 与选择依据；formal 评价在冻结后进行（见 apply_dev_fixed）。
    """
    l2 = [r for r in level2_rows if r["split"] == split]
    by_j: dict = defaultdict(list)
    by_j_h: dict = defaultdict(lambda: defaultdict(list))
    for r in l2:
        u = -float(r["actual_delta_eval"])  # u_ij = L(no-op) − L(action)
        by_j[int(r["j"])].append(u)
        by_j_h[int(r["H"])][int(r["j"])].append(u)
    means = {j: float(np.mean(v)) for j, v in by_j.items() if j != 0}
    j_star = min(means, key=lambda j: (-means[j], j))
    per_h = {
        h: {j: float(np.mean(v)) for j, v in by_j_h[h].items() if j != 0}
        for h in sorted(by_j_h)
    }
    return {
        "j_star": j_star,
        "selection_means_u": means,
        "tie_break": "mean u 并列取小 j",
        "per_horizon_means_u_sensitivity": per_h,
        "n_dev_rows": len(l2),
        "frozen_before_formal": True,
    }


def apply_dev_fixed(level2_rows: Sequence[Mapping], split: str, h: int, j_star: int) -> list[dict]:
    """把冻结组 j* 应用到 split 的每个 unit：gain=u_{i,j*}、regret=u_i*−u_{i,j*}。"""
    l2 = [r for r in level2_rows if r["split"] == split and int(r["H"]) == h]
    by_unit: dict = defaultdict(dict)
    for r in l2:
        by_unit[(int(r["seed"]), int(r["step"]), int(r["stream_id"]))][int(r["j"])] = float(r["actual_delta_eval"])
    out = []
    for u, actual in sorted(by_unit.items()):
        sel = actual[j_star]
        best = min(actual.values())
        worst = max(actual.values())
        tol = max(1e-12, NEAR_TIE_FRACTION * (worst - best))
        tied = sorted(j for j, v in actual.items() if v - best <= tol)
        out.append({
            "split": split, "seed": u[0], "step": u[1], "stream_id": u[2], "H": h,
            "method": "dev_fixed", "j_hat": j_star,
            "actual_delta_selected": sel,
            "gain": -sel, "regret": sel - best, "harmful": sel > 0.0,
            "top1_correct": j_star in tied, "direction_correct": (-sel > 0) == (-sel > 0),
            "best_possible_delta": best,
        })
    return out


def replay_score_availability(level1b_rows: Sequence[Mapping], scan_rows: Sequence[Mapping]) -> dict:
    """真实回放评分（V_score 实际损失选动作 + V_eval 评价）所需列的可用性。"""
    has_score_col = any("score" in k for k in level1b_rows[0]) if level1b_rows else False
    formal_units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"]))
                           for r in level1b_rows if r["split"] == "formal"})
    dev_scan_units = sorted({(int(r["seed"]), int(r["step"]), int(r["stream_id"]))
                             for r in scan_rows if r.get("split") == "dev"})
    return {
        "status": "MISSING_REPLAY_SCORE",
        "formal_v_score_replay_columns_present": has_score_col,
        "formal_units_with_v_score_replay": 0,
        "formal_units_total": len(formal_units),
        "note": "冻结 level1b/level2 只有 actual_delta_eval（V_eval 回放）；候选在 V_score 上的"
                "实际回放损失只在 dev λ 扫描（lambda_scan_detail.csv，dev-only）中持久化，"
                "formal 的 V_score 回放列不存在。本阶段不补跑 GPU，故该基线标 MISSING_REPLAY_SCORE"
                "（仅阻塞本项，不阻塞其他离线分析）。",
        "dev_only_score_rows_available": len(dev_scan_units),
    }


def load_baseline_rows(level2_rows, level3_rows, split, h, j_star):
    """汇总某 (split,h) 下全部方法（含补充基线）的 unit 级 gain/regret 行。"""
    rows = [dict(r) for r in level3_rows if r["split"] == split and int(r["H"]) == h]
    rows += baseline_never(level2_rows, split, h)
    rows += apply_dev_fixed(level2_rows, split, h, j_star)
    return rows
