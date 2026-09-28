#!/usr/bin/env python3
"""E4B-1 GPU replay 前的只读文件预检。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e4b_q_rotation_v08.json")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    src_cfg = json.loads((root / cfg["source_config"]).read_text(encoding="utf-8"))
    src_out = root / src_cfg["output_dir"]

    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    states = list(map(int, cfg["state_steps"]))
    R = int(cfg["source_streams"])
    errors = []
    counts = {"theta_raw": 0, "state_pt": 0, "q_pt": 0, "manifest": 0, "e2_records": 0}

    required_top = [
        src_out / "partition_v07.json",
        src_out / "e3c_subset.jsonl",
        src_out / "PREREG_LOCK.json",
    ]
    for p in required_top:
        if not p.exists():
            errors.append(f"missing: {p}")

    for seed in seeds:
        seed_dir = src_out / f"seed_{seed}"
        raw_path = seed_dir / "theta_mapping_raw.json"
        bank_path = seed_dir / "state_bank_report.json"
        if not raw_path.exists() or not bank_path.exists():
            errors.append(f"seed {seed}: missing theta raw/state bank report")
            continue
        counts["theta_raw"] += 1
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        raw_by_step = {int(x["step"]): x for x in raw.get("states", [])}
        bank_by_step = {int(x["step"]): x for x in bank.get("states", [])}
        for step in states:
            if step not in raw_by_step or step not in bank_by_step:
                errors.append(f"seed {seed} state {step}: raw/bank record missing")
                continue
            st = Path(bank_by_step[step]["path"])
            if not st.exists():
                # 常规服务器路径固定，若项目移动则检查标准相对位置。
                st = seed_dir / "states" / f"state_step_{step:03d}.pt"
            if st.exists(): counts["state_pt"] += 1
            else: errors.append(f"seed {seed} state {step}: state pt missing")
            q = Path(raw_by_step[step]["q_path"])
            if not q.exists(): q = seed_dir / "q" / f"q_state_step_{step:03d}.pt"
            if q.exists(): counts["q_pt"] += 1
            else: errors.append(f"seed {seed} state {step}: q pt missing")
            e2 = raw_by_step[step].get("e2_streams", [])
            if len(e2) != R:
                errors.append(f"seed {seed} state {step}: e2 stream count {len(e2)} != {R}")
            for rec in e2:
                counts["e2_records"] += 1
                mp = Path(rec["batch_manifest_path"])
                if not mp.exists():
                    mp = seed_dir / "e2_stream_manifests" / f"state_step_{step:03d}" / f"e2_stream_{int(rec['stream_id']):03d}_batches.json"
                if mp.exists(): counts["manifest"] += 1
                else: errors.append(f"seed {seed} state {step} stream {rec.get('stream_id')}: manifest missing")

    expected_states = len(seeds) * len(states)
    expected_streams = expected_states * R
    passed = (
        not errors and counts["theta_raw"] == len(seeds)
        and counts["state_pt"] == expected_states
        and counts["q_pt"] == expected_states
        and counts["manifest"] == expected_streams
        and counts["e2_records"] == expected_streams
    )
    print("E4B1_PREFLIGHT =", "PASS" if passed else "FAIL")
    print("COUNTS =", json.dumps(counts, ensure_ascii=False))
    if errors:
        print("ERRORS:")
        for e in errors[:50]: print(" -", e)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
