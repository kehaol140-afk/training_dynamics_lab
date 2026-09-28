#!/usr/bin/env python3
"""E4A v0.8 Low-dimensional State Screening。

只读消费 v0.7 confirmation formal 结果，生成独立 outputs/e4a_lowdim_state/v08
探索性报告。不得修改 v0.7 locked files 或已有 raw data。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


PRIMARY_HORIZON = 8
SUPPORTING_HORIZONS = [4, 12]
E4A_HORIZONS = [4, 8, 12]
OUTPUT_REL = Path("outputs/e4a_lowdim_state/v08")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fmt(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(x) else f"{x:.6g}"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_value(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _assert_finite_row(row: Mapping, *, context: str) -> None:
    bad = [key for key, value in row.items() if not _finite_value(value)]
    if bad:
        raise RuntimeError(f"{context} 存在非有限数值字段：{bad}")


def _state_map(report: Mapping) -> Dict[int, Mapping]:
    return {int(row["step"]): row for row in report.get("states", [])}


def _theta_state_map(raw: Mapping) -> Dict[int, Mapping]:
    return {int(row["step"]): row for row in raw.get("states", [])}


def _manifest_refs(streams: Iterable[Mapping]) -> List[Dict]:
    refs = []
    for stream in streams:
        refs.append({
            "stream_id": int(stream["stream_id"]),
            "stream_seed": int(stream["stream_seed"]),
            "batch_seed": int(stream["batch_seed"]),
            "batch_manifest_hash": stream.get("batch_manifest_hash"),
            "batch_manifest_sha256": stream.get("batch_manifest_sha256"),
            "batch_manifest_path": stream.get("batch_manifest_path"),
            "first_preclip_grad_norm": float(stream["first_preclip_grad_norm"]),
        })
    return refs


def _e2_refs(streams: Iterable[Mapping]) -> List[Dict]:
    refs = []
    for stream in streams:
        refs.append({
            "stream_id": int(stream["stream_id"]),
            "stream_seed": int(stream["stream_seed"]),
            "batch_seed": int(stream["batch_seed"]),
            "batch_manifest_hash": stream.get("batch_manifest_hash"),
            "batch_manifest_sha256": stream.get("batch_manifest_sha256"),
            "baseline_first_preclip_grad_norm": float(stream["baseline"]["first_preclip_grad_norm"]),
            "plus_first_preclip_grad_norm": float(stream["plus"]["first_preclip_grad_norm"]),
            "minus_first_preclip_grad_norm": float(stream["minus"]["first_preclip_grad_norm"]),
            "baseline_first_clipped": bool(stream["baseline"]["first_clipped"]),
            "plus_first_clipped": bool(stream["plus"]["first_clipped"]),
            "minus_first_clipped": bool(stream["minus"]["first_clipped"]),
            "rho0_noop_pass": bool(stream.get("rho0_noop", {}).get("pass", False)),
        })
    return refs


def _read_v07_h8_status(out: Path) -> str:
    for name in ("e3c_locked_confirmation_recovery.json", "e3c_confirmatory_analysis_v07.json"):
        path = out / name
        if path.exists():
            data = _load_json(path)
            return str(data.get("H8_CONFIRMATION", "UNKNOWN"))
    return "UNKNOWN"


def _csv_rows(rows: List[Mapping]) -> List[Dict]:
    flat_keys = [
        "protocol",
        "exploratory_role",
        "train_seed",
        "state_step",
        "state_fraction",
        "lr",
        "preclip_norm",
        "clip_indicator",
        "clip_margin",
        "mu_q",
        "momentum_alignment",
        "update_alignment",
        "sigma_q",
        "mean_s_q",
        "NSR_q",
        "q_norm",
        "q_norm_from_tensor",
        "probe_grad_norm",
        "epsilon",
        "rho",
        "B_H4",
        "chi_H4",
        "B_H8",
        "chi_H8",
        "B_H12",
        "chi_H12",
        "baseline_clip_rate",
        "plus_clip_rate",
        "minus_clip_rate",
        "e1_stream_count",
        "e2_stream_count",
        "q_sha256",
        "q_path",
        "state_path",
        "state_model_hash",
        "state_optimizer_hash",
    ]
    return [{key: row.get(key) for key in flat_keys} for row in rows]


def collect_features(root: Path, cfg: Mapping, *, v07_out: Path):
    import torch

    from fr.direction import direction_sha256
    from fr.e4a_lowdim_state import adamw_lowdim_from_optimizer_state, e1_e2_observation_features

    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    states = list(map(int, cfg["e3_state_steps"]))
    expected_state_set = set(states)
    grad_clip = float(cfg["grad_clip"])

    feature_rows = []
    integrity = {
        "status": "PASS",
        "expected_state_count": len(seeds) * len(states),
        "actual_state_count": 0,
        "confirmation_seeds": seeds,
        "states": states,
        "q_hash_mismatches": [],
        "missing": [],
        "nonfinite": [],
        "B_chi_source": "v0.7 theta_mapping_raw.json",
        "B_chi_recomputed": False,
        "high_dimensional_tensors_saved_in_e4a_summary": False,
    }

    for seed in seeds:
        seed_dir = v07_out / f"seed_{seed}"
        theta_path = seed_dir / "theta_mapping_raw.json"
        state_report_path = seed_dir / "state_bank_report.json"
        if not theta_path.exists():
            integrity["missing"].append(str(theta_path))
            continue
        if not state_report_path.exists():
            integrity["missing"].append(str(state_report_path))
            continue
        theta_raw = _load_json(theta_path)
        state_report = _load_json(state_report_path)
        if not theta_raw.get("technical_pass") or not theta_raw.get("rho0_noop_pass"):
            raise RuntimeError(f"seed {seed} v0.7 theta technical/rho0 未 PASS")
        if not state_report.get("technical_pass"):
            raise RuntimeError(f"seed {seed} v0.7 state bank technical 未 PASS")
        if int(theta_raw.get("train_seed", -1)) != seed or int(state_report.get("train_seed", -1)) != seed:
            raise RuntimeError(f"seed {seed} train_seed 不匹配")

        state_by_step = _state_map(state_report)
        theta_by_step = _theta_state_map(theta_raw)
        if set(state_by_step) != expected_state_set:
            raise RuntimeError(f"seed {seed} state bank states={sorted(state_by_step)}，预期={states}")
        if set(theta_by_step) != expected_state_set:
            raise RuntimeError(f"seed {seed} theta states={sorted(theta_by_step)}，预期={states}")

        for step in states:
            state_meta = state_by_step[step]
            theta_state = theta_by_step[step]
            q_path = Path(theta_state["q_path"])
            state_path = Path(state_meta["path"])
            if not q_path.exists():
                integrity["missing"].append(str(q_path))
                continue
            if not state_path.exists():
                integrity["missing"].append(str(state_path))
                continue

            q_payload = torch.load(q_path, map_location="cpu", weights_only=False)
            current_q_sha = direction_sha256(q_payload["q"])
            expected_q_sha = str(theta_state["q_sha256"])
            if current_q_sha != expected_q_sha or str(q_payload.get("sha256")) != expected_q_sha:
                integrity["q_hash_mismatches"].append({
                    "train_seed": seed,
                    "state_step": step,
                    "expected_q_sha256": expected_q_sha,
                    "payload_q_sha256": q_payload.get("sha256"),
                    "current_q_sha256": current_q_sha,
                })
            q_metadata = dict(q_payload.get("metadata", {}))
            if int(q_metadata.get("train_seed", seed)) != seed or int(q_metadata.get("state_step", step)) != step:
                raise RuntimeError(f"q metadata 与 seed/state 不匹配：seed={seed}, step={step}")
            if "state_path" in q_metadata and Path(q_metadata["state_path"]) != state_path:
                raise RuntimeError(f"q metadata state_path 与 state bank 不一致：seed={seed}, step={step}")

            state_payload = torch.load(state_path, map_location="cpu", weights_only=False)
            opt_features = adamw_lowdim_from_optimizer_state(q_payload["q"], state_payload["optimizer"])
            stream_features = e1_e2_observation_features(theta_state, grad_clip=grad_clip)
            if not opt_features["finite"] or not stream_features["finite"]:
                integrity["nonfinite"].append({"train_seed": seed, "state_step": step})

            row = {
                "protocol": "E4A_LOWDIM_STATE_SCREEN_V08",
                "exploratory_role": "EXPLORATORY_MECHANISM_STUDY_DOES_NOT_CHANGE_V07",
                "source_protocol": theta_raw.get("protocol"),
                "train_seed": seed,
                "state_step": step,
                "step": step,
                "state_fraction": float(theta_state["state_fraction"]),
                "q_norm": float(theta_state["q_norm"]),
                "probe_grad_norm": float(theta_state["probe_grad_norm"]),
                "epsilon": float(theta_state["epsilon"]),
                "rho": float(theta_state["rho"]),
                "q_sha256": expected_q_sha,
                "q_payload_sha256": str(q_payload.get("sha256")),
                "q_current_sha256": current_q_sha,
                "q_path": str(q_path),
                "state_path": str(state_path),
                "state_model_hash": state_meta.get("model_hash"),
                "state_optimizer_hash": state_meta.get("optimizer_hash"),
                "theta_mapping_raw_path": str(theta_path),
                "state_bank_report_path": str(state_report_path),
                "e1_stream_audit_refs": _manifest_refs(theta_state.get("e1_streams", [])),
                "e2_paired_audit_refs": _e2_refs(theta_state.get("e2_streams", [])),
            }
            row.update({k: v for k, v in opt_features.items() if k != "finite"})
            row.update({k: v for k, v in stream_features.items() if k != "finite"})
            for h in E4A_HORIZONS:
                hk = str(int(h))
                row[f"B_H{h}"] = float(theta_state["B_theta"][hk])
                row[f"chi_H{h}"] = float(theta_state["chi_theta"][hk])
            _assert_finite_row(row, context=f"seed={seed}, step={step}")
            feature_rows.append(row)

            # [FR-OBSERVATION-ONLY] 不保留高维张量引用，释放本轮 state/q。
            del state_payload
            del q_payload
            gc.collect()

    integrity["actual_state_count"] = len(feature_rows)
    if (
        integrity["actual_state_count"] != integrity["expected_state_count"]
        or integrity["missing"]
        or integrity["q_hash_mismatches"]
        or integrity["nonfinite"]
    ):
        integrity["status"] = "FAIL"
    return feature_rows, integrity


def analyze_loso(features: List[Mapping]):
    from fr.e4a_lowdim_state import CANDIDATES, baseline_loso, rank_candidates, single_factor_loso

    by_horizon = {}
    for h in E4A_HORIZONS:
        candidate_results = {}
        for candidate in CANDIDATES:
            candidate_results[candidate] = single_factor_loso(features, candidate=candidate, horizon=h)
        by_horizon[str(h)] = {
            "baseline_loso": baseline_loso(features, horizon=h),
            "candidates": candidate_results,
        }

    ranking_h8 = rank_candidates(by_horizon[str(PRIMARY_HORIZON)]["candidates"])
    top = ranking_h8[0] if ranking_h8 else None
    if top is None or float(top["mean_relative_mae_improvement"]) <= 0.0:
        best_single_state = "NONE"
        best_status = "UNSUPPORTED"
    else:
        best_single_state = top["candidate"]
        best_status = top["exploratory_status"]
    return {
        "primary_exploratory_horizon": PRIMARY_HORIZON,
        "supporting_horizons": SUPPORTING_HORIZONS,
        "variable_ranking_rule": "H=8 LOSO mean relative MAE improvement only",
        "forbidden_selection_horizons": [1, 2, 16, 24, 32],
        "candidate_ranking_H8": ranking_h8,
        "by_horizon": by_horizon,
        "best_single_state": best_single_state,
        "best_status": best_status,
        "single_lowdim_state_clear_explanation": bool(best_status == "PROMISING"),
        "controller_suggestion_made": False,
        "multivariable_search_performed": False,
    }


def write_features(out_dir: Path, features: List[Mapping]) -> Dict[str, str]:
    json_path = out_dir / "e4a_state_features.json"
    csv_path = out_dir / "e4a_state_features.csv"
    json_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = _csv_rows(features)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    return {"json": str(json_path), "csv": str(csv_path)}


def write_loso_report(out_dir: Path, payload: Mapping) -> None:
    analysis = payload["analysis"]
    h8 = analysis["by_horizon"]["8"]
    lines = [
        "# E4A v0.8 Low-dimensional State Screening",
        "",
        "本报告是 exploratory mechanism study；不改写 v0.7 confirmatory 结果，不输出 confirmed。",
        "",
        "## Baseline H8 LOSO",
        "",
        "- model: `M0 = seed_FE + state_fraction + B`",
        f"- mean MAE: `{_fmt(h8['baseline_loso']['mean_mae_M0'])}`",
        f"- median MAE: `{_fmt(h8['baseline_loso']['median_mae_M0'])}`",
        "",
        "| held-out seed | M0 MAE |",
        "|---:|---:|",
    ]
    for fold in h8["baseline_loso"]["folds"]:
        lines.append(f"| {fold['held_out_seed']} | {_fmt(fold['mae_M0'])} |")

    lines += [
        "",
        "## H8 Candidate Ranking",
        "",
        "Ranking 只依据 H=8；H=4/H=12 仅作邻近 horizon 对照。",
        "",
        "| rank | candidate | status | mean gain | median gain | folds improved | delta sign stability | majority sign |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(analysis["candidate_ranking_H8"], start=1):
        lines.append(
            f"| {idx} | {row['candidate']} | {row['exploratory_status']} | "
            f"{_fmt(row['mean_relative_mae_improvement'])} | "
            f"{_fmt(row['median_relative_mae_improvement'])} | "
            f"{row['folds_improved']}/8 | {row['delta_sign_stability_count']}/8 | "
            f"{row['delta_majority_sign']} |"
        )

    lines += [
        "",
        "## Per-Fold H8 Gains",
        "",
    ]
    for row in analysis["candidate_ranking_H8"]:
        candidate = row["candidate"]
        result = h8["candidates"][candidate]
        lines += [
            f"### {candidate}",
            "",
            "| held-out seed | relative MAE improvement | delta interaction | delta sign |",
            "|---:|---:|---:|---|",
        ]
        for fold in result["folds"]:
            lines.append(
                f"| {fold['held_out_seed']} | {_fmt(fold['relative_mae_improvement'])} | "
                f"{_fmt(fold['delta_interaction'])} | {fold['delta_sign']} |"
            )
        lines.append("")

    lines += [
        "## H4/H8/H12 Contrast",
        "",
        "| candidate | H4 mean gain | H4 delta sign | H8 mean gain | H8 delta sign | H12 mean gain | H12 delta sign |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in analysis["candidate_ranking_H8"]:
        candidate = row["candidate"]
        h4_res = analysis["by_horizon"]["4"]["candidates"][candidate]
        h8_res = analysis["by_horizon"]["8"]["candidates"][candidate]
        h12_res = analysis["by_horizon"]["12"]["candidates"][candidate]
        lines.append(
            f"| {candidate} | {_fmt(h4_res['mean_relative_mae_improvement'])} | {h4_res['delta_majority_sign']} | "
            f"{_fmt(h8_res['mean_relative_mae_improvement'])} | {h8_res['delta_majority_sign']} | "
            f"{_fmt(h12_res['mean_relative_mae_improvement'])} | {h12_res['delta_majority_sign']} |"
        )

    lines += [
        "",
        "## Exploratory Boundary",
        "",
        f"- E4A_BEST_SINGLE_STATE: `{analysis['best_single_state']}`",
        f"- E4A_BEST_STATUS: `{analysis['best_status']}`",
        f"- 单一低维状态量明显解释 seed heterogeneity: `{analysis['single_lowdim_state_clear_explanation']}`",
        "- controller 建议: `NOT_PERFORMED`",
        "- 多变量组合搜索: `NOT_PERFORMED`",
    ]
    (out_dir / "e4a_single_factor_loso.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(out_dir: Path, payload: Mapping) -> None:
    integrity = payload["data_integrity"]
    lines = [
        "# E4A v0.8 Audit",
        "",
        f"- E4A_DATA_INTEGRITY: `{integrity['status']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- v0.7 H8_CONFIRMATION preserved: `{payload['v07_H8_CONFIRMATION']}`",
        f"- v0.7 prereg lock revision: `{payload['v07_prereg_lock_revision']}`",
        f"- v0.7 prereg lock status: `{payload['v07_prereg_lock_status']}`",
        f"- confirmation seeds: `{integrity['confirmation_seeds']}`",
        f"- states: `{integrity['states']}`",
        f"- state count: `{integrity['actual_state_count']}/{integrity['expected_state_count']}`",
        f"- q hash mismatches: `{len(integrity['q_hash_mismatches'])}`",
        f"- missing inputs: `{len(integrity['missing'])}`",
        f"- nonfinite rows: `{len(integrity['nonfinite'])}`",
        "- B/chi source: `v0.7 theta_mapping_raw.json`",
        "- B/chi recomputed: `False`",
        "- formal GPU mapping rerun: `False`",
        "- v0.7 locked files modified by E4A: `False`",
        "- existing raw data modified by E4A: `False`",
        "- high-dimensional q/m/v saved in E4A summaries: `False`",
        "- controller suggestion: `NOT_PERFORMED`",
        "- multivariable search: `NOT_PERFORMED`",
        "",
        "## Output Files",
    ]
    for key, value in payload["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "E4A_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/b205/projects/fluctuation_response")
    ap.add_argument("--config", default="research/configs/e3c_confirmatory_v07.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "research/src"))
    sys.path.insert(0, str(root / "research/scripts"))

    from scientific_preflight_v07 import assert_prereg_lock_valid, file_sha256

    cfg = _load_json(root / args.config)
    v07_out = root / cfg["output_dir"]
    lock = assert_prereg_lock_valid(root, v07_out)
    if int(lock.get("lock_revision", -1)) != 2:
        raise RuntimeError("E4A 要求 v0.7 prereg lock revision=2")
    v07_h8 = _read_v07_h8_status(v07_out)
    if v07_h8 != "FAIL":
        raise RuntimeError(f"E4A 要求保留 v0.7 H8_CONFIRMATION=FAIL，当前={v07_h8}")

    out_dir = root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    features, integrity = collect_features(root, cfg, v07_out=v07_out)
    if integrity["status"] != "PASS":
        raise RuntimeError(f"E4A data integrity FAIL: {json.dumps(integrity, ensure_ascii=False)}")

    feature_files = write_features(out_dir, features)
    loso = analyze_loso(features)
    payload = {
        "protocol": "E4A_LOWDIM_STATE_SCREEN_V08",
        "exploratory_role": "EXPLORATORY_MECHANISM_STUDY_DOES_NOT_CHANGE_V07",
        "generated_at_utc": _now_utc(),
        "project_root": str(root),
        "v07_output_dir": str(v07_out),
        "v07_H8_CONFIRMATION": v07_h8,
        "v07_prereg_lock_revision": int(lock.get("lock_revision")),
        "v07_prereg_lock_sha256": file_sha256(v07_out / "PREREG_LOCK.json"),
        "v07_prereg_lock_status": "MATCH",
        "data_integrity": integrity,
        "analysis": loso,
        "output_files": {
            "e4a_state_features_json": feature_files["json"],
            "e4a_state_features_csv": feature_files["csv"],
            "e4a_single_factor_loso_json": str(out_dir / "e4a_single_factor_loso.json"),
            "e4a_single_factor_loso_md": str(out_dir / "e4a_single_factor_loso.md"),
            "E4A_AUDIT": str(out_dir / "E4A_AUDIT.md"),
        },
    }
    (out_dir / "e4a_single_factor_loso.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_loso_report(out_dir, payload)
    write_audit(out_dir, payload)

    print(f"E4A_DATA_INTEGRITY = {integrity['status']}")
    print(f"E4A_BEST_SINGLE_STATE = {loso['best_single_state']}")
    print(f"E4A_BEST_STATUS = {loso['best_status']}")


if __name__ == "__main__":
    main()
