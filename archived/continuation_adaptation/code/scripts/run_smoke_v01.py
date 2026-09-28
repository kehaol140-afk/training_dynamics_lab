#!/usr/bin/env python3
"""run_smoke_v01 — P3 smoke：1 seed × 1 checkpoint × 臂 {A, A_repeat, B, C1, C2, D} × 64 步。

只报告实现状态 / 正确性 / 可复现性 / 异常；不解读性能（docs/EXPERIMENT_PROTOCOL.md §7）。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from continuation_runner import run_one_arm  # noqa: E402


def _read_loss_curve(run_dir: Path):
    return (run_dir / "loss_curve.csv").read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "smoke_v01.json"))
    ap.add_argument("--lab-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    lab_root = Path(args.lab_root).resolve()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(args.out_root) if args.out_root else (Path(__file__).resolve().parent.parent / cfg["outputs"]["report"]).parent
    runs_root = out_root / "runs"

    results, issues = {}, []
    for run_id in cfg["smoke_runs"]:
        run_dir = runs_root / run_id
        try:
            metrics = run_one_arm(cfg, lab_root, run_id, args.device, run_dir)
            results[run_id] = metrics
            print(f"[smoke] {run_id}: done, eval_final_mean={metrics['eval_final']['mean']:.6f}")
        except Exception:
            results[run_id] = None
            err = traceback.format_exc()
            issues.append({"run_id": run_id, "error": err})
            print(f"[smoke] {run_id}: FAILED\n{err}")

    report = {"config_id": cfg["config_id"], "device": args.device, "unexpected_issues": issues}

    completed = {rid: m for rid, m in results.items() if m is not None}
    report["checks"] = {}

    # 1. 全部完成
    report["checks"]["all_runs_complete"] = all(
        m is not None and m["n_optimizer_steps"] == cfg["budget_optimizer_steps"]
        for m in results.values()
    ) and len(results) == len(cfg["smoke_runs"])

    # 2. evaluation 可用
    report["checks"]["eval_works"] = all(
        m["eval_final"]["token_count"] > 0 for m in completed.values()
    )

    # 3. C0 起点 & batch 流（CRN 前提）跨 run 一致
    if completed:
        c0_hashes = {m["model_hash_at_c0"] for m in completed.values()}
        evals_c0 = {json.dumps(m["eval_at_c0"], sort_keys=True) for m in completed.values()}
        manifests = {m["batch_manifest_sha256"] for m in completed.values()}
        report["checks"]["c0_model_state_identical_across_runs"] = len(c0_hashes) == 1
        report["checks"]["eval_at_c0_identical_across_runs"] = len(evals_c0) == 1
        report["checks"]["batch_manifest_identical_across_runs"] = len(manifests) == 1

    # 4. 确定性：A vs A_repeat 逐位一致
    if results.get("A") and results.get("A_repeat"):
        a, ar = results["A"], results["A_repeat"]
        curve_equal = _read_loss_curve(runs_root / "A") == _read_loss_curve(runs_root / "A_repeat")
        report["checks"]["determinism_A_repeat"] = (
            a["model_hash_final"] == ar["model_hash_final"]
            and a["optimizer_hash_final"] == ar["optimizer_hash_final"]
            and curve_equal
        )

    # 5. 不同臂产生不同轨迹（原始事实，不解读）
    distinct_arms = ["A", "B", "C1", "C2", "D"]
    hashes = {arm: results[arm]["model_hash_final"] for arm in distinct_arms if results.get(arm)}
    report["checks"]["arms_produce_distinct_trajectories"] = (
        len(hashes) == len(distinct_arms) and len(set(hashes.values())) == len(distinct_arms)
    )
    report["final_model_hashes"] = hashes
    report["final_eval_mean_by_arm"] = {
        arm: results[arm]["eval_final"]["mean"] for arm in distinct_arms if results.get(arm)
    }

    # 6. 臂施加验证全部通过（runner 内部已断言，此处汇总）
    report["checks"]["arm_application_verified"] = all(
        m["arm_record"]["checks"]["betas_applied"] for m in completed.values()
    )

    report_path = out_root / "smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] report -> {report_path}")
    ok = all(bool(v) for v in report["checks"].values())
    print(f"[smoke] ALL_CHECKS_{'PASS' if ok and not issues else 'FAIL'}")
    return 0 if ok and not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
