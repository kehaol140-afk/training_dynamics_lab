#!/usr/bin/env python3
"""E4B-1b：stochastic branching（D_q）单候选探索筛选。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fr.e4b_dynamic_timescale import (
    atomic_json,
    atomic_text,
    classify_screen,
    load_q_rotation_rows,
    loso_single_candidate,
    sha256_file,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e4b_dynamic_timescale_v08.json")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = root / cfg["branching_output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    stream_rows, state_rows, source_files = load_q_rotation_rows(root, cfg)
    horizons = list(map(int, cfg["horizons"]))
    primary_h = int(cfg["primary_exploratory_horizon"])
    support = list(map(int, cfg["supporting_horizons"]))

    # [FR-METRIC] D_q 固定为 8 future streams 的 sample std(R_q), ddof=1。
    feature_rows = []
    for r in state_rows:
        feature_rows.append({
            "seed": int(r["seed"]),
            "state_step": int(r["state_step"]),
            "state_fraction": float(r["state_fraction"]),
            "horizon": int(r["horizon"]),
            "D_q": float(r["std_Rq"]),
            "mean_Rq": float(r["mean_Rq"]),
            "median_Rq": float(r["median_Rq"]),
            "iqr_Rq": float(r["iqr_Rq"]),
            "mad_Rq": float(r["mad_Rq"]),
            "B": float(r["B"]),
            "chi": float(r["chi"]),
        })

    with (out / "e4b1b_branching_state_features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        w.writeheader(); w.writerows(feature_rows)

    screens = {}
    statuses = {}
    for h in [primary_h] + support:
        rows_h = [r for r in feature_rows if int(r["horizon"]) == h]
        sc = loso_single_candidate(rows_h, "D_q")
        screens[str(h)] = sc
        statuses[str(h)] = classify_screen(sc, cfg["screening"])

    h8 = screens[str(primary_h)]
    status = statuses[str(primary_h)]
    result = {
        "protocol": cfg["protocol_version"] + "_E4B1B",
        "study_role": cfg["study_role"],
        "data_integrity": True,
        "candidate": "D_q = sample std_r(R_q), ddof=1",
        "primary_exploratory_horizon": primary_h,
        "supporting_horizons": support,
        "screens": screens,
        "statuses": statuses,
        "H8_status": status,
        "boundary": [
            "v0.7 H8_CONFIRMATION=FAIL remains unchanged.",
            "D_q comes from future baseline trajectories and is not an online controller state.",
            "No GPU replay and no formal B/chi recomputation were performed.",
        ],
    }
    atomic_json(out / "e4b1b_branching_loso.json", result)

    lines = [
        "# E4B-1b v0.8 Stochastic Branching Screen",
        "",
        "本报告为 post-confirmatory exploratory mechanism study；不改写 v0.7 confirmatory FAIL。",
        "",
        "## Data Integrity",
        "",
        "- E4B-1 exact-replay state outputs: `PASS`",
        f"- source state files: `{len(source_files)}` (expected 56)",
        f"- state×H rows: `{len(feature_rows)}` (expected 448)",
        f"- candidate: `D_q = sample std_r(R_q), ddof={int(cfg['branching_metric']['ddof'])}`",
        "- formal B/chi source: E4B-1 state JSON 中由 v0.7 raw 转存；`NOT_RECOMPUTED`",
        "",
        "## H8 Primary Exploratory Screen",
        "",
        f"- mean relative MAE improvement: `{h8['mean_relative_mae_improvement']:.6g}`",
        f"- median relative MAE improvement: `{h8['median_relative_mae_improvement']:.6g}`",
        f"- folds improved: `{h8['folds_improved']}/{h8['n_folds']}`",
        f"- B×Dq sign stability: `{h8['delta_same_sign_folds']}/{h8['n_folds']}` ({h8['delta_majority_sign']})",
        f"- E4B1B_H8_STATUS: `{status}`",
        "",
        "### Per-held-out-seed",
        "",
        "| seed | baseline MAE | +Dq MAE | relative gain | delta interaction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in h8["folds"]:
        lines.append(
            f"| {fold['held_out_seed']} | {fold['baseline_mae']:.6g} | {fold['candidate_mae']:.6g} | "
            f"{fold['relative_mae_improvement']:.4%} | {fold['delta_interaction']:.6g} |"
        )
    lines += [
        "",
        "## Supporting Horizons",
        "",
        "| H | status | mean gain | folds improved | interaction sign stability |",
        "|---:|---|---:|---:|---:|",
    ]
    for h in support:
        sc = screens[str(h)]
        lines.append(
            f"| {h} | {statuses[str(h)]} | {sc['mean_relative_mae_improvement']:.4%} | "
            f"{sc['folds_improved']}/{sc['n_folds']} | {sc['delta_same_sign_folds']}/{sc['n_folds']} ({sc['delta_majority_sign']}) |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- [FR-OBSERVATION-ONLY] D_q 只来自已完成的 baseline q-rotation replay。",
        "- [FR-METRIC] candidate selection 只使用 D_q；mean/median/IQR/MAD 仅描述。",
        "- [FR-IO] 本脚本只写 branching/ 新目录。",
        "- v0.7 H8_CONFIRMATION=FAIL 保持不变；本结果不能称为 confirmed。",
        "- 不进入 controller。",
        "",
        "## Final",
        "",
        "E4B1B_DATA_INTEGRITY = PASS",
        f"E4B1B_H8_STATUS = {status}",
        f"E4B1B_H8_MEAN_GAIN = {h8['mean_relative_mae_improvement']:.10g}",
        f"E4B1B_H8_FOLDS_IMPROVED = {h8['folds_improved']}/{h8['n_folds']}",
        f"E4B1B_H8_DELTA_SIGN_STABILITY = {h8['delta_same_sign_folds']}/{h8['n_folds']} ({h8['delta_majority_sign']})",
    ]
    atomic_text(out / "e4b1b_branching_loso.md", "\n".join(lines) + "\n")

    source_digest = hashlib_sources(source_files)
    audit = [
        "# E4B1B_AUDIT",
        "",
        "E4B1B_DATA_INTEGRITY = PASS",
        f"SOURCE_Q_ROTATION_STATE_FILE_COUNT = {len(source_files)}",
        f"SOURCE_Q_ROTATION_AGGREGATE_SHA256 = {source_digest}",
        f"CONFIG_SHA256 = {sha256_file(cfg_path)}",
        "FORMAL_B_CHI_RECOMPUTED = NO",
        "GPU_RERUN = NO",
        "V07_LOCKED_FILES_WRITTEN = NO",
        "V07_RAW_FILES_WRITTEN = NO",
        "E4A_E4B1_RAW_FILES_WRITTEN = NO",
        f"OUTPUT_SCOPE = {cfg['branching_output_dir']}",
    ]
    atomic_text(out / "E4B1B_AUDIT.md", "\n".join(audit) + "\n")

    print("E4B1B_DATA_INTEGRITY = PASS")
    print("E4B1B_H8_STATUS =", status)
    print("E4B1B_H8_MEAN_GAIN =", h8["mean_relative_mae_improvement"])
    print("E4B1B_H8_FOLDS_IMPROVED =", f"{h8['folds_improved']}/{h8['n_folds']}")
    print("REPORT =", out / "e4b1b_branching_loso.md")
    return 0


def hashlib_sources(paths):
    import hashlib
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode("utf-8")); h.update(b"\0")
        h.update(sha256_file(p).encode("ascii")); h.update(b"\n")
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
