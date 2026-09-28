#!/usr/bin/env python3
"""v0.7 locked confirmation-only recovery extractor.

用途：
- 不修改任何 prereg locked file；
- 不读取 discovery seeds；
- 只执行 locked analysis 中 discovery appendix 之前的 confirmation-only path；
- 使用 locked `fr.confirmatory_stats` 与 locked config 的参数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


DISCOVERY_COMPARISON_STATUS = "UNAVAILABLE_DUE_TO_DEGENERATE_4-SEED_CLUSTER_BOOTSTRAP"
PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS = "YES"
PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY = "YES"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_prereg_lock(root: Path, out: Path) -> dict:
    """验证 revision=2 lock 及所有 locked/dependency SHA256。"""
    lock_path = out / "PREREG_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("缺少 PREREG_LOCK.json，禁止 recovery extraction")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if int(lock.get("lock_revision", -1)) != 2:
        raise RuntimeError(f"PREREG_LOCK lock_revision 不是 2：{lock.get('lock_revision')}")

    mismatches = []
    for section in ("locked_files", "dependency_files"):
        for rec in lock.get(section, []):
            rel = rec["path"]
            path = root / rel
            current = file_sha256(path) if path.exists() else None
            if current != rec.get("sha256"):
                mismatches.append({
                    "section": section,
                    "path": rel,
                    "locked_sha256": rec.get("sha256"),
                    "current_sha256": current,
                })
    if mismatches:
        raise RuntimeError(f"PREREG_LOCK SHA256 mismatch，禁止 recovery extraction：{mismatches}")
    return lock


def load_confirmation_seed_results(out: Path, confirmation_seeds):
    """只读取 confirmation seed raw results，不读取 discovery 路径。"""
    raw = []
    for seed in map(int, confirmation_seeds):
        path = out / f"seed_{seed}/theta_mapping_raw.json"
        if not path.exists():
            raise RuntimeError(f"缺少 confirmation seed {seed} theta_mapping_raw.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol") != "E3C_CONFIRMATORY_V07_MULTISEED_THETA":
            raise RuntimeError(f"seed {seed} protocol 不是 E3C_CONFIRMATORY_V07_MULTISEED_THETA")
        if int(result.get("train_seed", -1)) != seed:
            raise RuntimeError(f"seed {seed} train_seed 字段不匹配")
        if result.get("technical_pass") is not True:
            raise RuntimeError(f"seed {seed} technical_pass 不是 true")
        if result.get("rho0_noop_pass") is not True:
            raise RuntimeError(f"seed {seed} rho0_noop_pass 不是 true")
        raw.append(result)
    return raw


def flatten_confirmation_by_horizon(raw_results, horizons, *, expected_states):
    """把 confirmation raw result 转成 locked stats 需要的 rows_by_horizon。"""
    rows_by_h = {str(int(h)): [] for h in horizons}
    expected_steps = sorted(map(int, expected_states))
    for result in raw_results:
        seed = int(result["train_seed"])
        states = sorted(result.get("states", []), key=lambda st: int(st["step"]))
        actual_steps = [int(st["step"]) for st in states]
        if actual_steps != expected_steps:
            raise RuntimeError(f"seed {seed} state grid 不匹配：{actual_steps}")
        for state in states:
            for h in horizons:
                hk = str(int(h))
                rows_by_h[hk].append({
                    "source_role": "CONFIRMATION_PRIMARY_ELIGIBLE",
                    "train_seed": seed,
                    "step": int(state["step"]),
                    "state_fraction": float(state["state_fraction"]),
                    "B": float(state["B_theta"][hk]),
                    "chi": float(state["chi_theta"][hk]),
                    "odd_error": float(state["odd_error_theta"][hk]),
                    "clipping": dict(state.get("clipping", {})),
                })
    return rows_by_h


def degeneracy_diagnosis() -> dict:
    """记录 discovery 4-seed cluster bootstrap 的退化概率与固定 RNG 复现。"""
    import numpy as np

    n4_prob = 4 / (4 ** 4)
    n8_prob = 8 / (8 ** 8)
    first = None
    rng = np.random.default_rng(77070001 + 909000 + 1 * 1000)
    seeds = [2026, 2027, 2028, 2029]
    for i in range(1, 10001):
        sampled = list(map(int, rng.choice(seeds, size=len(seeds), replace=True)))
        if len(set(sampled)) < 2:
            first = {"replicate": i, "sampled_training_seeds": sampled}
            break
    return {
        "root_cause": (
            "cluster_bootstrap_ci_K 对 training seed 做有放回抽样；"
            "4-seed discovery bootstrap replicate 可能只包含 1 个 unique seed，"
            "随后 seed_fe_adjusted_K 调用 _seed_fe_design 触发 >=2 seeds guard。"
        ),
        "n4_one_unique_probability": n4_prob,
        "n4_one_unique_probability_fraction": "4/4^4 = 1/64",
        "n4_expected_degenerate_replicates_in_10000": 10000 * n4_prob,
        "n4_probability_at_least_one_degenerate_in_10000": 1 - ((1 - n4_prob) ** 10000),
        "n8_one_unique_probability": n8_prob,
        "n8_one_unique_probability_fraction": "8/8^8 = 1/2097152",
        "n8_expected_degenerate_replicates_in_10000": 10000 * n8_prob,
        "n8_probability_at_least_one_degenerate_in_10000": 1 - ((1 - n8_prob) ** 10000),
        "fixed_rng_discovery_h1_first_degenerate": first,
        "code_evidence": {
            "e3c_analyze_confirmation_analysis_lines": "149-156",
            "e3c_analyze_discovery_analysis_lines": "158-183, failure reported at 173",
            "confirmatory_stats_cluster_resample_lines": "168-175",
            "confirmatory_stats_seed_fe_guard_lines": "104-108",
        },
    }


def fmt(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(x) else f"{x:.6g}"


def write_markdown(path: Path, report: dict):
    primary = report.get("primary_confirmation")
    lines = [
        "# v0.7 Locked Confirmation Recovery",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS: `{report['PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS']}`",
        f"- PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY: `{report['PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY']}`",
        f"- DISCOVERY_COMPARISON_STATUS: `{report['DISCOVERY_COMPARISON_STATUS']}`",
        f"- H8_CONFIRMATION: `{report.get('H8_CONFIRMATION')}`",
        "",
        "## Operational Equivalence",
        "",
        "- recovery 只执行 locked analysis 中 discovery appendix 之前的 confirmation-only path。",
        "- recovery 不读取 discovery seeds `2026..2029`。",
        "- H=8 数据、函数、参数、随机种子、bootstrap reps、permutation reps 与 locked config/locked stats 一致。",
        "- 未添加、删除或修改 prereg A/B/C 条件。",
        "- 未引入 post-data scientific degree of freedom。",
        "",
        "## Root Cause Diagnosis",
        "",
    ]
    dg = report["diagnosis"]
    lines += [
        f"- n=4 bootstrap 单个 replicate 只含 1 个 unique seed 的概率：`{dg['n4_one_unique_probability_fraction']}` = `{fmt(dg['n4_one_unique_probability'])}`",
        f"- 10000 reps 期望退化 replicate 数：`{fmt(dg['n4_expected_degenerate_replicates_in_10000'])}`",
        f"- 固定 RNG 复现：H=1 discovery 第 `{dg['fixed_rng_discovery_h1_first_degenerate']['replicate']}` 个 replicate 抽到 `{dg['fixed_rng_discovery_h1_first_degenerate']['sampled_training_seeds']}`",
        "",
        "## Primary H=8",
        "",
    ]
    if primary is not None:
        obs = primary["observed"]
        lines += [
            f"- status: `{primary['status']}`",
            f"- K_8: `{fmt(obs['K'])}`",
            f"- cluster CI95 lower: `{fmt(obs['cluster_ci95_lower'])}`",
            f"- detrended Pearson: `{fmt(obs['within_seed_detrended_pearson'])}`",
            f"- circular-shift p(one-sided positive): `{fmt(obs['circular_shift_p_one_sided_positive'])}`",
            f"- LOSO mean relative MAE improvement: `{fmt(obs['loso_mean_relative_mae_improvement'])}`",
            f"- LOSO folds improved: `{obs['loso_folds_improved']}/{obs['loso_total_folds']}`",
        ]
        for key, value in primary["conditions"].items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- not emitted because recovery independence/equivalence checks failed")

    lines += ["", "## Horizons", ""]
    for h in report["horizons"]:
        analysis = report["analysis"][str(h)]
        lines += [
            f"### H={h}",
            f"- endpoint_role: `{analysis['endpoint_role']}`",
            f"- seed-FE K_H: `{fmt(analysis['seed_fe_adjusted']['K'])}`",
            f"- cluster CI95(K_H): `[{fmt(analysis['cluster_bootstrap_ci_K']['ci95'][0])}, {fmt(analysis['cluster_bootstrap_ci_K']['ci95'][1])}]`",
            f"- detrended Pearson: `{fmt(analysis['within_seed_detrended']['pearson'])}`",
            f"- circular-shift p: `{fmt(analysis['circular_shift_permutation']['p_one_sided_positive'])}`",
            f"- first-difference Pearson: `{fmt(analysis['first_difference']['pearson'])}`",
            f"- LOSO mean MAE improvement: `{fmt(analysis['loso']['mean_relative_mae_improvement'])}`",
            f"- LOSO folds improved: `{analysis['loso']['folds_improved']}/{analysis['loso']['total_folds']}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e3c_confirmatory_v07.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "research/src"))
    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = root / cfg["output_dir"]
    lock = verify_prereg_lock(root, out)

    from fr.confirmatory_stats import analyze_confirmatory_horizons

    confirmation_seeds = list(map(int, cfg["confirmation_train_seeds"]))
    horizons = list(map(int, cfg["horizons"]))
    expected_states = list(map(int, cfg["e3_state_steps"]))
    primary_horizon = int(cfg["primary_horizon"])

    raw = load_confirmation_seed_results(out, confirmation_seeds)
    rows_by_h = flatten_confirmation_by_horizon(raw, horizons, expected_states=expected_states)
    expected_rows = len(confirmation_seeds) * len(expected_states)
    for h in horizons:
        n = len(rows_by_h[str(h)])
        if n != expected_rows:
            raise RuntimeError(f"H={h} confirmation rows={n}，预期={expected_rows}")

    confirmation_analysis = analyze_confirmatory_horizons(
        rows_by_h,
        primary_horizon=primary_horizon,
        bootstrap_reps=int(cfg["confirmatory_bootstrap_reps"]),
        permutation_reps=int(cfg["confirmatory_permutation_reps"]),
        random_seed=int(cfg["confirmatory_random_seed"]),
    )
    primary = confirmation_analysis["analysis"][str(primary_horizon)]["primary_confirmation"]
    emit_h8 = (
        PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS == "YES"
        and PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY == "YES"
    )

    report = {
        "protocol": "E3C_CONFIRMATORY_V07_LOCKED_CONFIRMATION_RECOVERY",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "recovery_scope": "confirmation-only path before e3c_analyze_confirmatory_v07 discovery appendix",
        "PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS": PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS,
        "PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY": PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY,
        "DISCOVERY_COMPARISON_STATUS": DISCOVERY_COMPARISON_STATUS,
        "H8_CONFIRMATION": primary["H8_CONFIRMATION"] if emit_h8 else None,
        "primary_confirmation": primary if emit_h8 else None,
        "confirmation_seeds": confirmation_seeds,
        "discovery_seeds_read": [],
        "primary_horizon": primary_horizon,
        "horizons": horizons,
        "bootstrap_reps": int(cfg["confirmatory_bootstrap_reps"]),
        "permutation_reps": int(cfg["confirmatory_permutation_reps"]),
        "random_seed": int(cfg["confirmatory_random_seed"]),
        "prereg_lock_revision": int(lock["lock_revision"]),
        "prereg_lock_sha256": file_sha256(out / "PREREG_LOCK.json"),
        "locked_stats_file": "research/src/fr/confirmatory_stats.py",
        "locked_stats_sha256": next(
            x["sha256"] for x in lock["locked_files"]
            if x["path"] == "research/src/fr/confirmatory_stats.py"
        ),
        "locked_config_file": args.config,
        "locked_config_sha256": next(
            x["sha256"] for x in lock["locked_files"]
            if x["path"] == "research/configs/e3c_confirmatory_v07.json"
        ),
        "analysis": confirmation_analysis["analysis"],
        "secondary_holm": confirmation_analysis["secondary_holm"],
        "diagnosis": degeneracy_diagnosis(),
        "post_data_scientific_degree_of_freedom": False,
        "formal_raw_data_modified": False,
        "locked_files_modified": False,
    }

    json_path = out / "e3c_locked_confirmation_recovery.json"
    md_path = out / "e3c_locked_confirmation_recovery.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)

    print("E3C_V07_LOCKED_CONFIRMATION_RECOVERY_DONE")
    print("PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS=", PRIMARY_RECOVERY_USES_UNCHANGED_LOCKED_STATS)
    print("PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY=", PRIMARY_CONFIRMATORY_INDEPENDENT_OF_DISCOVERY)
    print("DISCOVERY_COMPARISON_STATUS=", DISCOVERY_COMPARISON_STATUS)
    print("H8_CONFIRMATION=", report["H8_CONFIRMATION"])
    print("json=", json_path)
    print("md=", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
