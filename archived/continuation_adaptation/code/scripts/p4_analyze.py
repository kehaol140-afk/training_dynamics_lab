#!/usr/bin/env python3
"""p4_analyze — P4 Stage 3 单遍分析（docs/P4_PROTOCOL.md §7–§8 冻结口径）。

输入：Stage 2 的 sealed_evaluation_table.csv（只读）+ configs/p4_full.json（SHA 校验）。
输出：outputs/p4_full/stage3/analysis_report.json（含 primary 统计、Holm、
一致性门槛、停止条件裁决）。secondary 比较仅描述性。

按协议单遍执行：本脚本不重算 NLL、不做任何选择、不产生可反馈到 runs 的量。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p4_statistics import (  # noqa: E402
    FAMILY_ALPHA, PRIMARY_ARMS, cluster_bootstrap_ci, evaluate_gates,
    evaluate_variant, holm_correction, naive_bootstrap_ci, paired_diffs,
    seed_cluster_means,
)
from p4_runner import load_p4_config  # noqa: E402

DESCRIPTIVE_COMPARISONS = [("B", "A_best"), ("D", "A_best"), ("B", "D"), ("C1", "C2")]


def load_sealed_table(path: Path) -> dict:
    table, cell_meta = {}, {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell_id = row["cell_id"]
            cell_meta.setdefault(cell_id, (int(row["seed"]), int(row["checkpoint_step"])))
            table.setdefault(cell_id, {})[row["arm"]] = float(row["nll_mean"])
    return table, cell_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p4_full.json"))
    ap.add_argument("--sealed-table", default=str(ROOT / "outputs" / "p4_full" / "stage2" / "sealed_evaluation_table.csv"))
    ap.add_argument("--stage1-root", default=str(ROOT / "outputs" / "p4_full" / "stage1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "p4_full" / "stage3" / "analysis_report.json"))
    args = ap.parse_args()

    cfg = load_p4_config(Path(args.config))
    table, cell_meta = load_sealed_table(Path(args.sealed_table))
    cells = {c["cell_id"]: c for c in cfg["checkpoints"]["cells"]}
    if set(table) != set(cells):
        raise SystemExit(f"封存表 cell 集不完整: {len(table)}/56")

    # ---------- primary：C1/C2 vs A_best ----------
    variant_stats, p_values = {}, {}
    for arm in PRIMARY_ARMS:
        d = paired_diffs(table, arm)
        st = evaluate_variant(d, cell_meta)
        variant_stats[arm] = {"paired_diff": st}
        p_values[arm] = st["permutation"]["p_value_one_sided"]
    holm = holm_correction(p_values, alpha=FAMILY_ALPHA)
    gates = evaluate_gates({a: variant_stats[a]["paired_diff"] for a in PRIMARY_ARMS}, holm)

    # ---------- secondary（仅描述性） ----------
    secondary = {}
    for arm_x, arm_y in DESCRIPTIVE_COMPARISONS:
        d = {cid: table[cid][arm_x] - table[cid][arm_y] for cid in table}
        secondary[f"{arm_x}_minus_{arm_y}"] = {
            "aggregate_mean_diff": sum(d.values()) / len(d),
            "naive_bootstrap_ci": naive_bootstrap_ci(d),
            "per_seed_means": {str(k): v for k, v in
                               seed_cluster_means(d, cell_meta).items()},
            "descriptive_only": True,
        }

    # ---------- 边界占比（来自 stage1 台账，报告用） ----------
    n_unresolved = n_boundary_expanded = 0
    stage1 = Path(args.stage1_root)
    for cell_id in cells:
        ledger = stage1 / cell_id / "selection_ledger.json"
        if ledger.is_file():
            led = json.loads(ledger.read_text(encoding="utf-8"))
            n_unresolved += int(bool(led.get("boundary_unresolved")))
            n_boundary_expanded += int(any(r["multiplier"] in (0.25, 4.0)
                                           for r in led.get("runs", [])))

    report = {
        "config_id": cfg["config_id"],
        "primary_family": PRIMARY_ARMS,
        "family_alpha": FAMILY_ALPHA,
        "primary": variant_stats,
        "holm": holm,
        "gates": gates,
        "verdict": "CONTINUE" if gates["continue_direction"] else "STOP",
        "secondary_descriptive": secondary,
        "a_grid_boundary_accounting": {
            "cells_with_expansion": n_boundary_expanded,
            "cells_boundary_unresolved": n_unresolved,
            "note": "unresolved 占比高时，C>A 结论强度降级（见 P4_PROTOCOL_REVIEW §4.4）",
        },
        "stop_conditions_reference": "docs/P4_PROTOCOL.md §8",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage3] verdict={report['verdict']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
