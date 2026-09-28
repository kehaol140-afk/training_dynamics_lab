#!/usr/bin/env python3
"""E4B-1 q-rotation exploratory analysis。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fr.e4b_q_rotation import distribution_summary, loso_conditional_screen


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e4b_q_rotation_v08.json")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    out = root / cfg["output_dir"]
    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    states = list(map(int, cfg["state_steps"]))
    horizons = list(map(int, cfg["horizons"]))
    R = int(cfg["source_streams"])

    stream_rows = []
    state_rows = []
    for seed in seeds:
        for step in states:
            p = out / f"seed_{seed}" / f"state_step_{step:03d}.json"
            if not p.exists():
                raise SystemExit(f"缺少正式 E4B-1 state output: {p}")
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not obj.get("technical_pass") or len(obj.get("streams", [])) != R:
                raise SystemExit(f"E4B-1 state 未完整 PASS: {p}")
            by_h = {h: [] for h in horizons}
            for s in obj["streams"]:
                if not s.get("replay_exact", {}).get("pass"):
                    raise SystemExit(f"exact replay 未 PASS: seed={seed},state={step},stream={s.get('stream_id')}")
                for o in s["observations"]:
                    h = int(o["horizon"])
                    rq = float(o["R_q"])
                    by_h[h].append(rq)
                    stream_rows.append({
                        "seed": seed, "state_step": step, "state_fraction": float(obj["state_fraction"]),
                        "stream_id": int(s["stream_id"]), "horizon": h, "R_q": rq,
                        "q_t_sha256": o["q_t_sha256"], "q_future_sha256": o["q_future_sha256"],
                        "batch_manifest_sha256": s["batch_manifest_sha256"],
                    })
            for h in horizons:
                if len(by_h[h]) != R:
                    raise SystemExit(f"seed={seed},state={step},H={h}: Rq stream count != {R}")
                sm = distribution_summary(by_h[h])
                state_rows.append({
                    "seed": seed, "state_step": step, "state_fraction": float(obj["state_fraction"]),
                    "horizon": h, "mean_Rq": sm["mean"], "std_Rq": sm["std"],
                    "median_Rq": sm["median"], "q25_Rq": sm["q25"], "q75_Rq": sm["q75"],
                    "B": float(obj["B_theta"][str(h)]), "chi": float(obj["chi_theta"][str(h)]),
                })

    # CSV 输出
    with (out / "e4b1_q_rotation_streams.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stream_rows[0].keys())); w.writeheader(); w.writerows(stream_rows)
    with (out / "e4b1_q_rotation_states.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(state_rows[0].keys())); w.writeheader(); w.writerows(state_rows)

    horizon_summary = {}
    conditional = {}
    for h in horizons:
        sr = [r for r in stream_rows if int(r["horizon"]) == h]
        st = [r for r in state_rows if int(r["horizon"]) == h]
        horizon_summary[str(h)] = {
            "stream_level_Rq": distribution_summary([r["R_q"] for r in sr]),
            "state_mean_Rq": distribution_summary([r["mean_Rq"] for r in st]),
            "state_stream_dispersion": distribution_summary([r["std_Rq"] for r in st]),
        }
        conditional[str(h)] = loso_conditional_screen(st)

    h8 = conditional[str(int(cfg["primary_exploratory_horizon"]))]
    rule = cfg["conditional_screening"]
    if h8["mean_relative_mae_improvement"] <= float(rule["promising_mean_relative_mae_gain_gt"]):
        status = "UNSUPPORTED"
    elif (h8["folds_improved"] >= int(rule["promising_folds_improved_gte"]) and
          h8["delta_same_sign_folds"] >= int(rule["promising_delta_same_sign_folds_gte"])):
        status = "PROMISING"
    else:
        status = "WEAK"

    # 方向衰减只作描述，不强行假定单调。
    means = [horizon_summary[str(h)]["state_mean_Rq"]["mean"] for h in horizons]
    monotone_nonincreasing = all(means[i+1] <= means[i] + 1e-12 for i in range(len(means)-1))

    result = {
        "protocol": cfg["protocol_version"] + "_ANALYSIS",
        "study_role": cfg["study_role"],
        "data_integrity": True,
        "n_seeds": len(seeds), "states_per_seed": len(states), "streams_per_state": R,
        "horizons": horizons,
        "horizon_summary": horizon_summary,
        "conditional_loso": conditional,
        "primary_exploratory_horizon": int(cfg["primary_exploratory_horizon"]),
        "H8_conditional_status": status,
        "Rq_mean_monotone_nonincreasing": monotone_nonincreasing,
        "boundary": "R_q 使用未来 baseline trajectory，是机制诊断量，不是当前可直接在线使用的状态变量。",
    }
    atomic_text(out / "e4b1_q_rotation_analysis.json", json.dumps(result, ensure_ascii=False, indent=2))

    lines = [
        "# E4B-1 v0.8 q-Direction Rotation",
        "",
        "本报告为 exploratory mechanism study；不改写 v0.7 confirmatory FAIL。",
        "",
        "## Replay Integrity",
        "",
        f"- seeds: `{len(seeds)}`",
        f"- states/seed: `{len(states)}`",
        f"- streams/state: `{R}`",
        f"- total baseline trajectories: `{len(seeds)*len(states)*R}`",
        "- all formal state outputs exact replay: `PASS`",
        "",
        "## Horizon Curve",
        "",
        "| H | mean Rq | state-to-state std | mean stream dispersion | conditional LOSO gain | folds improved | delta sign stability |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in horizons:
        hs = horizon_summary[str(h)]
        cs = conditional[str(h)]
        lines.append(
            f"| {h} | {hs['state_mean_Rq']['mean']:.6f} | {hs['state_mean_Rq']['std']:.6f} | "
            f"{hs['state_stream_dispersion']['mean']:.6f} | {cs['mean_relative_mae_improvement']:.4%} | "
            f"{cs['folds_improved']}/{cs['n_folds']} | {cs['delta_same_sign_folds']}/{cs['n_folds']} ({cs['delta_majority_sign']}) |"
        )
    lines += [
        "",
        "## H8 Primary Exploratory Mechanism Screen",
        "",
        f"- mean relative MAE improvement: `{h8['mean_relative_mae_improvement']:.6g}`",
        f"- median relative MAE improvement: `{h8['median_relative_mae_improvement']:.6g}`",
        f"- folds improved: `{h8['folds_improved']}/{h8['n_folds']}`",
        f"- B×Rq interaction sign stability: `{h8['delta_same_sign_folds']}/{h8['n_folds']}` ({h8['delta_majority_sign']})",
        f"- E4B1_H8_CONDITIONAL_STATUS: `{status}`",
        "",
        "### H8 per-held-out-seed",
        "",
        "| seed | baseline MAE | +Rq MAE | relative gain | delta interaction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for f in h8["folds"]:
        lines.append(f"| {f['held_out_seed']} | {f['baseline_mae']:.6g} | {f['candidate_mae']:.6g} | {f['relative_mae_improvement']:.4%} | {f['delta_interaction']:.6g} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- [FR-OBSERVATION-ONLY] Rq 只由 baseline replay + fixed probe 测量。",
        "- [FR-REPLAY-CRITICAL] 每条 trajectory 的最终 model/optimizer hash 与 v0.7 baseline raw 一致。",
        "- Rq 是未来动态机制诊断量，不应直接当作可在线获取的 controller state。",
        "- H8 只作 exploratory mechanism screening；不能改写 v0.7 的 H8_CONFIRMATION=FAIL。",
        "",
        "## Final",
        "",
        "E4B1_REPLAY_INTEGRITY = PASS",
        "E4B1_Q_ROTATION_MEASURED = YES",
        f"E4B1_H8_CONDITIONAL_STATUS = {status}",
        f"E4B1_RQ_MONOTONE_DECAY = {'YES' if monotone_nonincreasing else 'NO'}",
    ]
    atomic_text(out / "e4b1_q_rotation_analysis.md", "\n".join(lines) + "\n")
    print("E4B1_ANALYSIS_PASS")
    print("E4B1_H8_CONDITIONAL_STATUS =", status)
    print("REPORT =", out / "e4b1_q_rotation_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
