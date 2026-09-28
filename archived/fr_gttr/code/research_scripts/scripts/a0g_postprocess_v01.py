#!/usr/bin/env python3
"""FR_A0G 决策效用实验的离线聚合/报告补齐（不重启 GPU 实验）。

背景
----
main phase 的 15 个 state × 8 stream 全部完成且原始行 CSV 已写盘
（level1_local_derivative / level1_finite_action / level2_action_predictions /
level3_decision_metrics），但进程在首个聚合调用 (_aggregate_level3)
处因解包 bug 崩溃：for (split, h), _ in ... 把 split 字符串逐字符解包。
计时器累计（成本核算所需）随进程丢失。

本脚本只做离线重算：
  - 读取已写盘的原始行 CSV（字节级不动，它们是冻结的实验记录）；
  - 重算 level1/level1b/level2/level3 聚合与逐 state Spearman；
  - 写 level1_per_state_spearman.csv / experiment_config.json /
    summary.json / FR_DECISION_UTILITY_REPORT.md / SHA256SUMS.txt。

成本核算（§12）
--------------
--measured-json 提供计时补测结果（各操作 per-unit GPU 秒）时，生成
cost_breakdown.csv、填充 runtime 成本字段并给 summary.status = COMPLETE；
否则 cost_rows 为空、status = COMPLETE_PENDING_COST_TIMING，报告成本章节
标注待补测（§14 准则 5 暂不判定）。

shadow 检查
-----------
逐检查明细随崩溃进程丢失；此处从 main_phase.log 的 STATE_DONE 行重建
粗粒度记录（dev 全 stream、formal stream 0 的 shadow 结果以 state 级
shadow_pass 汇总表示），并在 runtime 里注明来源。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "research/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import importlib.util

def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "a0g_runner", REPO_ROOT / "research/scripts/a0g_decision_utility_v01.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()

from fr.a0g_decision_utility_v01 import (  # noqa: E402
    HORIZONS,
    LAMBDA_CANDIDATES,
    N_DATA_GROUPS,
    NEAR_TIE_FRACTION,
    PRIMARY_HORIZON,
    _safe_spearman,
    assemble_summary,
    breakeven_m,
    cost_model,
    json_dumps,
    pooled_metric,
    render_report,
)
from fr.native_adamw_guard import optimizer_signature  # noqa: E402

CONDITIONAL_LOOKAHEAD_LIMITATION = RUNNER.CONDITIONAL_LOOKAHEAD_LIMITATION

# 进程启动时刻（2026-09-13 会话中 ps 记录：STIME 12:27），用于 elapsed 估算。
MAIN_PHASE_STARTED_EPOCH = time.mktime(time.strptime("2026-09-13 12:27:30", "%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------------------
# CSV 读取 + 类型还原（_process_state 在内存中写出的类型）
# ---------------------------------------------------------------------------

def _i(x):
    return int(x)


def _f(x):
    x = (x or "").strip()
    return float(x) if x not in ("", "None", "nan") else None


def _b(x):
    return (x or "").strip().lower() == "true"


_L1_INT = ("seed", "step", "stream_id", "j", "H")
_L1_FLOAT = (
    "lambda", "hhat_alignment", "hhat_B", "hhat_C", "h_FD",
    "abs_err_alignment", "abs_err_B", "abs_err_C",
    "rel_err_alignment", "rel_err_B", "rel_err_C",
    "g_j_gap_vs_tangent_pass",
)
_L1_BOOL = ("sign_match_alignment", "sign_match_B", "sign_match_C",
            "clip_regime_crossing")

_L1B_INT = ("seed", "step", "stream_id", "j", "H")
_L1B_FLOAT = (
    "lambda", "actual_delta_eval", "pred_delta_alignment", "pred_delta_B",
    "pred_delta_C", "baseline_val_eval", "action_val_eval",
    "nonlinear_residual_C",
)

_L2_INT = ("seed", "step", "stream_id", "j", "H")
_L2_FLOAT = ("lambda", "actual_delta_eval", "pred_delta_alignment",
             "pred_delta_B", "pred_delta_C")

_L3_INT = ("seed", "step", "stream_id", "H", "j_hat")
_L3_FLOAT = ("predicted_delta_selected", "actual_delta_selected",
             "best_possible_delta", "gain", "regret", "near_tie_tolerance")
_L3_BOOL = ("harmful", "top1_correct", "direction_correct")


def _load_csv(path: Path, int_fields, float_fields, bool_fields=()):
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if k in int_fields:
                    row[k] = _i(v)
                elif k in float_fields:
                    row[k] = _f(v)
                elif k in bool_fields:
                    row[k] = _b(v)
                else:
                    row[k] = v
            rows.append(row)
    return rows


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 聚合（与 runner main() 中的实现逐步一致）
# ---------------------------------------------------------------------------

def _l1_block(rows, pred_key):
    finite = [r for r in rows
              if r.get(pred_key) is not None and r.get("h_FD") is not None]
    if not finite:
        return {"n": 0}
    m = pooled_metric(finite, pred_key=pred_key, actual_key="h_FD")
    sign_key = pred_key.replace("hhat_", "sign_match_")
    abs_key = pred_key.replace("hhat_", "abs_err_")
    return {
        "n": int(m["n"]),
        "pearson": float(m["corr"]),
        "mae": float(m["MAE"]),
        "rmse": float(m["RMSE"]),
        "mean_relative_error": float(m["mean_relative_error"]),
        "sign_accuracy": RUNNER._mean(1.0 if r[sign_key] else 0.0 for r in finite),
        "mean_abs_err": RUNNER._mean(float(r[abs_key]) for r in finite),
    }


def aggregate(level1_rows, level1b_rows, level2_rows, level3_rows):
    level3_agg = RUNNER._aggregate_level3(level3_rows, level2_rows)

    level1_agg = {}
    for h in HORIZONS:
        for split in ("dev", "formal"):
            rows_h = [r for r in level1_rows
                      if int(r["H"]) == h and r["split"] == split]
            if not rows_h:
                continue
            level1_agg[f"{split}|H={h}"] = {
                "H": h, "_split": split,
                "alignment": _l1_block(rows_h, "hhat_alignment"),
                "B": _l1_block(rows_h, "hhat_B"),
                "C": _l1_block(rows_h, "hhat_C"),
            }

    level1_per_state = []
    state_groups = {}
    for r in level1_rows:
        state_groups.setdefault(
            (r["split"], r["seed"], r["step"], int(r["H"])), []
        ).append(r)
    for (split, seed, step, h), rows_g in sorted(state_groups.items()):
        rec = {"split": split, "seed": seed, "step": step, "H": h}
        for m_key, pk in (("alignment", "hhat_alignment"),
                          ("B", "hhat_B"),
                          ("C", "hhat_C")):
            vals = _safe_spearman([float(r[pk]) for r in rows_g],
                                  [float(r["h_FD"]) for r in rows_g])
            rec[f"spearman_{m_key}"] = vals
        level1_per_state.append(rec)

    level1b_agg = {}
    for h in HORIZONS:
        for split in ("dev", "formal"):
            rows_h = [r for r in level1b_rows
                      if int(r["H"]) == h and r["split"] == split]
            if not rows_h:
                continue
            block = {"H": h, "_split": split, "n": len(rows_h)}
            for m_key, pk in (("alignment", "pred_delta_alignment"),
                              ("B", "pred_delta_B"), ("C", "pred_delta_C")):
                actuals = [float(r["actual_delta_eval"]) for r in rows_h]
                preds = [float(r[pk]) for r in rows_h]
                block[m_key] = {
                    "mae": RUNNER._mean(abs(a - p) for a, p in zip(actuals, preds)),
                    "mean_actual": RUNNER._mean(actuals),
                    "mean_pred": RUNNER._mean(preds),
                    "sign_accuracy": RUNNER._mean(
                        (1 if (a > 0) == (p > 0) else 0) if (a != 0 and p != 0) else 0.5
                        for a, p in zip(actuals, preds)
                    ),
                    "mean_abs_nonlinear_residual": RUNNER._mean(
                        abs(float(r["actual_delta_eval"]) - float(r[pk])) for r in rows_h
                    ),
                }
            level1b_agg[f"{split}|H={h}"] = block

    return level3_agg, level1_agg, level1b_agg, level1_per_state


# ---------------------------------------------------------------------------
# config / runtime / shadow 重建
# ---------------------------------------------------------------------------

def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", "1", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        name = out.stdout.strip()
        return name or "NVIDIA GeForce RTX 5090"
    except Exception:
        return "NVIDIA GeForce RTX 5090"


def build_experiment_config(lam_primary, lam_conservative, cfg):
    import torch
    v07_out = REPO_ROOT / cfg["output_dir"]
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],
                              lr=float(cfg["learning_rate"]))
    try:
        sig = optimizer_signature(dummy)
    finally:
        del dummy
    return {
        "protocol": "FR_DECISION_UTILITY_V01",
        "dev_states": [list(s) for s in RUNNER.DEV_STATES],
        "formal_seeds": list(RUNNER.FORMAL_SEEDS),
        "formal_steps": list(RUNNER.FORMAL_STEPS),
        "streams_per_state": RUNNER.N_STREAMS_PER_STATE,
        "horizons": list(HORIZONS),
        "primary_horizon": PRIMARY_HORIZON,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_primary": lam_primary,
        "lambda_conservative": lam_conservative,
        "n_data_groups": N_DATA_GROUPS,
        "action": "g_action = g + sign*lambda*(g_j - g) at post-unscale/pre-clip, single optimizer step",
        "tangent_convention": "r_H is derivative per unit lambda; finite prediction = lambda*hhat",
        "validation_batch_size": RUNNER.EVAL_BATCH_SIZE,
        "seed_batch_rule": "Philox(stream_seed ^ (9000+j)) permutation over group members, 2 microbatches x 4",
        "near_tie_fraction": NEAR_TIE_FRACTION,
        "device": "cuda:0",
        "gpu_name": _gpu_name(),
        "optimizer_signature": sig,
        "input_files": {
            "v07_config": str(REPO_ROOT / "research/configs/e3c_confirmatory_v07.json"),
            "partition": str(v07_out / "partition_v07.json"),
            "data": str(v07_out / "e3c_subset.jsonl"),
            "confirmatory_bank_index": str(
                REPO_ROOT / "outputs/fr_decision_utility_v01/confirmatory_bank/bank_index.json"),
        },
        "postprocess_note": (
            "aggregates rebuilt offline from raw row CSVs after main-phase process "
            "crashed in _aggregate_level3 (unpack bug) post CSV write; raw CSVs untouched"
        ),
    }


def parse_shadow_log(log_path: Path):
    checks = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"STATE_DONE (\{.*\})", line)
            if m:
                d = ast.literal_eval(m.group(1))
                checks.append({
                    "state": d["state"],
                    "pass": bool(d["shadow_pass"]),
                    "source": "main_phase.log STATE_DONE (state-level summary; "
                              "per-check detail lost with crashed process)",
                })
    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(
        REPO_ROOT / "outputs/fr_decision_utility_v01"))
    ap.add_argument("--measured-json", default=None,
                    help="计时补测的 measured dict JSON（提供则生成成本核算）")
    ap.add_argument("--main-log", default="/tmp/a0g_logs/main_phase.log")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    level1_rows = _load_csv(out_dir / "level1_local_derivative.csv",
                            _L1_INT, _L1_FLOAT, _L1_BOOL)
    level1b_rows = _load_csv(out_dir / "level1_finite_action.csv",
                             _L1B_INT, _L1B_FLOAT)
    level2_rows = _load_csv(out_dir / "level2_action_predictions.csv",
                            _L2_INT, _L2_FLOAT)
    level3_rows = _load_csv(out_dir / "level3_decision_metrics.csv",
                            _L3_INT, _L3_FLOAT, _L3_BOOL)

    level3_agg, level1_agg, level1b_agg, level1_per_state = aggregate(
        level1_rows, level1b_rows, level2_rows, level3_rows)
    _write_csv(out_dir / "level1_per_state_spearman.csv", level1_per_state)

    lam_sel = RUNNER._read_json(out_dir / "lambda_selection.json")
    lam_primary = float(lam_sel["primary_lambda"])
    lam_cons_raw = lam_sel.get("conservative_lambda")
    lam_conservative = float(lam_cons_raw) if lam_cons_raw is not None else None

    cfg = RUNNER._read_json(REPO_ROOT / "research/configs/e3c_confirmatory_v07.json")
    experiment_config = build_experiment_config(lam_primary, lam_conservative, cfg)

    # ---- 成本（可选：--measured-json） ----
    measured = None
    if args.measured_json:
        measured = json.loads(Path(args.measured_json).read_text(encoding="utf-8"))
        cost_rows = cost_model(measured)
        cost_csv_rows = (
            [{"record_type": "measured_component", "component": k, "value_s": v}
             for k, v in measured.items()]
            + [{"record_type": "cost_model", **row} for row in cost_rows]
        )
        _write_csv(out_dir / "cost_breakdown.csv", cost_csv_rows)
        be = {h: breakeven_m(measured, h) for h in HORIZONS}
        status = "COMPLETE"
    else:
        cost_rows = []
        be = None
        status = "COMPLETE_PENDING_COST_TIMING"

    import torch
    shadow_checks = parse_shadow_log(Path(args.main_log))
    l3_mtime = (out_dir / "level3_decision_metrics.csv").stat().st_mtime
    runtime = {
        "elapsed_seconds_estimate": float(l3_mtime - MAIN_PHASE_STARTED_EPOCH),
        "elapsed_note": ("last raw CSV mtime minus recorded process start "
                         "(12:27:30 CST from ps); approximate"),
        "states_processed": 15,
        "streams_processed": 120,
        "measured_per_stream_seconds": measured,
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_overall": None,
        "peak_memory_note": "lost with crashed process (timers/peak in memory only)",
        "breakeven_m_candidates": (
            {str(h): be[h] for h in HORIZONS} if be else None),
        "shadow_protocol_checks": shadow_checks,
        "all_shadow_pass": bool(shadow_checks) and all(
            x["pass"] for x in shadow_checks),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    data_group_definition = RUNNER._read_json(out_dir / "data_group_definition.json")
    validation_split = RUNNER._read_json(out_dir / "validation_split.json")

    summary = assemble_summary(
        config=experiment_config,
        data_group_definition=data_group_definition,
        validation_split=validation_split,
        lambda_selection=lam_sel,
        level1_rows=level1_rows, level1b_rows=level1b_rows,
        level2_rows=level2_rows, level3_rows=level3_rows,
        level3_aggregate=level3_agg,
        cost_rows=cost_rows, runtime=runtime,
        conditional_lookahead_limitation=CONDITIONAL_LOOKAHEAD_LIMITATION,
    )
    summary["status"] = status
    summary["level1_aggregate"] = level1_agg
    summary["level1b_aggregate"] = level1b_agg
    summary["level1_per_state_spearman"] = level1_per_state
    (out_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")
    (out_dir / "FR_DECISION_UTILITY_REPORT.md").write_text(
        render_report(summary), encoding="utf-8")

    entries = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n",
                                            encoding="utf-8")

    # ---- 分析摘要（stdout，供 §14 判定） ----
    digest = {
        "status": status,
        "all_shadow_pass": runtime["all_shadow_pass"],
        "n_shadow_checks": len(shadow_checks),
        "formal_H4_level3": {
            m: level3_agg[f"formal|H={PRIMARY_HORIZON}"][m]
            for m in ("random", "alignment", "B", "C")
        },
        "formal_H1_level3": {
            m: level3_agg[f"formal|H=1"][m] for m in ("random", "alignment", "B", "C")
        },
        "formal_H8_level3": {
            m: level3_agg[f"formal|H=8"][m] for m in ("random", "alignment", "B", "C")
        },
        "formal_H4_level1_C": level1_agg["formal|H=4"]["C"],
        "formal_H4_level1_alignment": level1_agg["formal|H=4"]["alignment"],
        "formal_H4_level1b_C": level1b_agg["formal|H=4"]["C"],
        "dev_H4_level1_C": level1_agg["dev|H=4"]["C"],
    }
    print(json.dumps(digest, ensure_ascii=False, indent=2, default=str))
    print(f"POSTPROCESS_STATUS = {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
