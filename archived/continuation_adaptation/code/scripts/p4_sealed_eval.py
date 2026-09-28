#!/usr/bin/env python3
"""p4_sealed_eval — P4 Stage 2 封存评测（docs/P4_PROTOCOL.md §6）。

只允许在全部 Stage 1 runs + 全部选择完成之后运行：
1. 扫描 stage1，校验 56 cell 的台账与 runs 完整性（含 CRN/同 C0 断言）；
2. 先写 SEAL_LOCK.json（run 清单 + metrics/final_state SHA256 台账）；
3. 从 final_state.pt 一次性计算 V_final（partition eval pool，128 样本）NLL——
   评测对象每 cell 恰为 {A_best, B, C1, C2, D}；
4. 计算 update 向量范数与 cos(u_A_best, u_X)（解释性）；
5. 原子写出 sealed_evaluation_table.csv / sealed_evaluation.json（已存在即拒绝，
   不可重算覆盖）。

Stage 1 的任何代码路径都无法通过本脚本之外的方式访问 V_final。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p4_runner import bootstrap, load_p4_config, sha256_file  # noqa: E402

EVAL_ARMS = ["A_best", "B", "C1", "C2", "D"]


def _sha(path: Path) -> str:
    return sha256_file(path)


def _flatten_model(model_state: dict):
    import torch
    keys = sorted(model_state.keys())
    return torch.cat([model_state[k].double().reshape(-1) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p4_full.json"))
    ap.add_argument("--lab-root", default=str(ROOT.parent))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage-root", default=str(ROOT / "outputs" / "p4_full" / "stage1"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "p4_full" / "stage2"))
    args = ap.parse_args()

    import torch
    lab_root = Path(args.lab_root).resolve()
    bootstrap(lab_root)
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import init_model
    from vendored_fr.eval_observables import evaluate_token_losses
    from vendored_fr.minimind_v03_runtime import make_amp
    from vendored_fr.pilot_partition import load_partition

    cfg = load_p4_config(Path(args.config))
    stage_root, out_dir = Path(args.stage_root), Path(args.out_dir)
    table_csv = out_dir / "sealed_evaluation_table.csv"
    result_json = out_dir / "sealed_evaluation.json"
    if table_csv.exists() or result_json.exists():
        raise SystemExit("封存结果已存在，按协议不可覆盖（exactly once）")

    bank_dir = lab_root / cfg["lab_root_relative"]["bank_dir"]
    cells = {c["cell_id"]: c for c in cfg["checkpoints"]["cells"]}
    if len(cells) != 56:
        raise SystemExit("cell 数不为 56")

    # ---------- 1. 完整性校验 + 清单 ----------
    inventory, tables = [], []
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        ledger_path = stage_root / cell_id / "selection_ledger.json"
        if not ledger_path.is_file():
            raise SystemExit(f"{cell_id}: 缺 selection_ledger")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        selected = float(ledger["selected_multiplier"])
        a_best_run_id = f"A_lr{selected:g}"

        run_ids = [r["run_id"] for r in ledger["runs"]] + ["B", "C1", "C2", "D"]
        metrics = {}
        for rid in run_ids:
            run_dir = stage_root / cell_id / rid
            for f in ("metrics.json", "loss_curve.csv", "optimizer_state_statistics.csv", "final_state.pt"):
                if not (run_dir / f).is_file():
                    raise SystemExit(f"{cell_id}/{rid}: 缺 {f}")
            m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics[rid] = m
            inventory.append({
                "cell_id": cell_id, "run_id": rid,
                "metrics_sha256": _sha(run_dir / "metrics.json"),
                "final_state_sha256": _sha(run_dir / "final_state.pt"),
                "model_hash_final": m["model_hash_final"],
            })
        # CRN / 同 C0 断言
        manifests = {m["batch_manifest_sha256"] for m in metrics.values()}
        c0_hashes = {m["model_hash_at_c0"] for m in metrics.values()}
        if len(manifests) != 1 or len(c0_hashes) != 1:
            raise SystemExit(f"{cell_id}: CRN 前提破坏（manifest/C0 哈希不一致）")
        tables.append({"cell": cell, "ledger": ledger, "metrics": metrics,
                       "a_best_run_id": a_best_run_id})

    # ---------- 2. SEAL_LOCK（评测前写入） ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    seal_lock = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_id": cfg["config_id"],
        "config_sha256": _sha(Path(args.config)),
        "n_runs": len(inventory),
        "runs": inventory,
    }
    (out_dir / "SEAL_LOCK.json").write_text(
        json.dumps(seal_lock, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 3. 封存评测 ----------
    subset_path = bank_dir / cfg["checkpoints"]["corpus"]["subset_relpath"]
    partition = load_partition(bank_dir / cfg["checkpoints"]["corpus"]["partition_relpath"])
    if partition["sha256"] != cfg["checkpoints"]["partition_sha256"]:
        raise SystemExit("partition SHA 与冻结值不一致")
    eval_indices = [int(i) for i in partition["eval_indices"]]

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_hidden_layers=int(cfg["model"]["num_hidden_layers"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(mmcfg, from_weight="none",
                                  tokenizer_path=str(lab_root / "minimind_base" / "model"),
                                  device=args.device)
    ds = PretrainDataset(str(subset_path), tokenizer, max_length=int(cfg["model"]["max_seq_len"]))
    autocast_ctx, _, _ = make_amp(torch, args.device, cfg["train"]["dtype"])

    rows, c0_vec_cache = [], {}
    for entry in tables:
        cell, cell_id = entry["cell"], entry["cell"]["cell_id"]
        if cell_id not in c0_vec_cache:
            c0_state = torch.load(bank_dir / cell["state_relpath"], map_location="cpu", weights_only=False)
            c0_vec_cache[cell_id] = _flatten_model(c0_state["model"])
        u_vecs = {}
        for arm in EVAL_ARMS:
            rid = entry["a_best_run_id"] if arm == "A_best" else arm
            st = torch.load(stage_root / cell_id / rid / "final_state.pt",
                            map_location="cpu", weights_only=False)
            md = st["metadata"]
            if md.get("cell_id") != cell_id or md.get("run_id") != rid:
                raise SystemExit(f"{cell_id}/{rid}: final_state metadata 与 run 不匹配")
            model.load_state_dict(st["model"])
            nll = evaluate_token_losses(model, ds, eval_indices,
                                        int(cfg["evaluation"]["v_final"]["batch_size"]),
                                        args.device, autocast_ctx)
            u = _flatten_model(st["model"]) - c0_vec_cache[cell_id]
            u_vecs[arm] = u
            rows.append({
                "cell_id": cell_id, "seed": cell["seed"],
                "checkpoint_step": cell["state_optimizer_step"],
                "arm": arm, "run_id": rid,
                "lr_multiplier": entry["metrics"][rid]["lr_multiplier"],
                "token_count": nll["token_count"], "nll_mean": nll["mean"],
                "nll_p95": nll["p95"], "nll_p99": nll["p99"], "nll_cvar99": nll["cvar99"],
                "update_norm": float(u.norm()),
                "cos_vs_abest": None,
            })
        for arm in ("B", "C1", "C2", "D"):
            cos = float(torch.nn.functional.cosine_similarity(
                u_vecs["A_best"], u_vecs[arm], dim=0))
            rows[-len(EVAL_ARMS) + EVAL_ARMS.index(arm)]["cos_vs_abest"] = cos
        print(f"[stage2] {cell_id} sealed")

    # ---------- 4. 原子写出 ----------
    cols = list(rows[0].keys())
    tmp = table_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    tmp.replace(table_csv)

    result = {
        "config_id": cfg["config_id"], "config_sha256": seal_lock["config_sha256"],
        "seal_lock_sha256": _sha(out_dir / "SEAL_LOCK.json"),
        "n_rows": len(rows), "n_cells": len(tables),
        "eval_pool": cfg["evaluation"]["v_final"]["pool"],
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "table_csv": table_csv.name,
    }
    tmp = result_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(result_json)
    print(f"[stage2] sealed table -> {table_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
