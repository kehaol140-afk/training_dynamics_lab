#!/usr/bin/env python3
"""p4_stage1 — P4 Stage 1 编排器（grid-A → 选择 → B/C/D；docs/P4_PROTOCOL.md §9）。

用法（逐 cell 推进，可断点续跑——已存在且完整的 run 自动跳过）：
  python scripts/p4_stage1.py grid  --cell s2030_t064 [--device cuda:0]
  python scripts/p4_stage1.py bcd   --cell s2030_t064
  python scripts/p4_stage1.py status

grid 阶段严格按冻结规则：先跑 {0.5,1,2}，按 final V_monitor NLL 选择；
最优在边界且扩张值未跑 → 自动且仅一次扩张（0.25/4）→ 重选 → 写台账。
bcd 阶段读台账，用选中乘子跑 B/C1/C2/D。本脚本不接触 V_final。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p4_runner import load_p4_config, run_one_cell_arm  # noqa: E402
from p4_selection import INITIAL_GRID, select_best, write_ledger  # noqa: E402


def _cell_run_dir(stage_root: Path, cell_id: str, run_id: str) -> Path:
    return stage_root / cell_id / run_id


def _run_complete(run_dir: Path) -> bool:
    needed = ["metrics.json", "loss_curve.csv", "optimizer_state_statistics.csv", "final_state.pt"]
    return run_dir.is_dir() and all((run_dir / f).is_file() for f in needed)


def _run_or_skip(cfg, lab_root, device, cell_id, arm, mult, run_dir) -> dict:
    if _run_complete(run_dir):
        print(f"[stage1] skip existing {cell_id}/{run_dir.name}")
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = run_one_cell_arm(cfg, lab_root, cell_id, arm, mult, device, run_dir)
    print(f"[stage1] done {cell_id}/{metrics['run_id']} "
          f"final_monitor_nll={metrics['final_monitor_nll']:.6f}")
    return metrics


def cmd_grid(args, cfg, lab_root):
    stage_root = Path(args.stage_root)
    cell_id = args.cell
    metrics_by_mult, runs = {}, []

    # 冻结规则：先跑完初始网格 {0.5, 1.0, 2.0}，再选择（不得逐 run 提前选择）
    for mult in INITIAL_GRID:
        run_dir = _cell_run_dir(stage_root, cell_id, f"A_lr{mult:g}")
        m = _run_or_skip(cfg, lab_root, args.device, cell_id, "A", mult, run_dir)
        metrics_by_mult[mult] = m["final_monitor_nll"]
        runs.append({"multiplier": mult, "run_id": f"A_lr{mult:g}",
                     "final_monitor_nll": m["final_monitor_nll"]})

    sel = select_best(metrics_by_mult)
    # 边界扩张：自动且仅一次（0.5→0.25，2.0→4.0），补跑后重选
    if sel["needs_expansion"] is not None:
        mult = sel["needs_expansion"]
        run_dir = _cell_run_dir(stage_root, cell_id, f"A_lr{mult:g}")
        m = _run_or_skip(cfg, lab_root, args.device, cell_id, "A", mult, run_dir)
        metrics_by_mult[mult] = m["final_monitor_nll"]
        runs.append({"multiplier": mult, "run_id": f"A_lr{mult:g}",
                     "final_monitor_nll": m["final_monitor_nll"]})
        sel = select_best(metrics_by_mult)

    ledger = write_ledger(stage_root / cell_id / "selection_ledger.json", cell_id, runs, sel)
    print(f"[stage1] {cell_id} selected={sel['selected_multiplier']:g} "
          f"boundary_unresolved={sel['boundary_unresolved']}")
    return ledger


def cmd_bcd(args, cfg, lab_root):
    stage_root = Path(args.stage_root)
    cell_id = args.cell
    ledger_path = stage_root / cell_id / "selection_ledger.json"
    if not ledger_path.is_file():
        raise SystemExit(f"{cell_id}: 缺 selection_ledger（先跑 grid）")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    mult = float(ledger["selected_multiplier"])
    for arm in ("B", "C1", "C2", "D"):
        _run_or_skip(cfg, lab_root, args.device, cell_id, arm, mult,
                     _cell_run_dir(stage_root, cell_id, arm))
    print(f"[stage1] {cell_id} B/C1/C2/D done at multiplier {mult:g}")


def cmd_status(args, cfg):
    stage_root = Path(args.stage_root)
    cells = cfg["checkpoints"]["cells"]
    n_ledgers = n_bcd = 0
    for c in cells:
        cid = c["cell_id"]
        if (stage_root / cid / "selection_ledger.json").is_file():
            n_ledgers += 1
        if all(_run_complete(_cell_run_dir(stage_root, cid, a)) for a in ("B", "C1", "C2", "D")):
            n_bcd += 1
    print(f"[stage1] ledgers {n_ledgers}/56, cells_with_bcd {n_bcd}/56")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["grid", "bcd", "status"])
    ap.add_argument("--cell", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "p4_full.json"))
    ap.add_argument("--lab-root", default=str(ROOT.parent))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage-root", default=str(ROOT / "outputs" / "p4_full" / "stage1"))
    args = ap.parse_args()

    cfg = load_p4_config(Path(args.config))
    lab_root = Path(args.lab_root).resolve()

    if args.command == "status":
        cmd_status(args, cfg)
        return 0
    if not args.cell:
        raise SystemExit("--cell 必填（cell 清单见 configs/p4_full.json）")
    if args.command == "grid":
        cmd_grid(args, cfg, lab_root)
    else:
        cmd_bcd(args, cfg, lab_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
