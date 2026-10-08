#!/usr/bin/env python3
"""F1 Stage 3: 1B optimizer-state audit (prereg section 4).

Per checkpoint (step in {380000, 950000, 1530000}) x frozen module group:
v floor stats (worst-over-tensors quantiles, F0 semantics), sqrt(v) threshold
fractions, exact zeros, m |.| quantiles, precision round-to-zero attribution
(fp32 storage -> bf16/fp16), c=0-anchor diagonal gain stats.

Requires: external/peteish1/step{N}_optim.pt (SHA256-verified).
Outputs: outputs/f1_1b_replication/state_floor_by_module.csv (+ logs/meta).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import opsens.f1_1b_replication_v01 as f1
from opsens.f1_1b_replication_v01 import MODULE_GROUPS

OUT = f1.OUT
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "state_audit_run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    torch.set_num_threads(min(32, torch.get_num_threads()))
    rows = []
    checkpoints_meta = {}
    steps_done = []
    for step in f1.STEPS:
        chk = f1.verify_checkpoint(step, "optim.pt")
        if not chk["sha_ok"]:
            log(f"step{step} optim.pt not ready ({chk['exists']=} {chk['size_ok']=}); skipped")
            continue
        t0 = time.time()
        payload = f1.load_optim_payload(step)
        named_state, meta = f1.optim_named_state(step, payload)
        g0 = meta["groups"][0]
        lr, beta1, beta2, eps = g0["lr"], g0["betas"][0], g0["betas"][1], g0["eps"]
        steps_stored = sorted(set(s["step"] for _, s in named_state))
        hp_unique = {(g["lr"], g["betas"], g["eps"]) for g in meta["groups"]}
        wd_map = {g["weight_decay"]: g["n_params"] for g in meta["groups"]}
        n_elements = sum(s["exp_avg_sq"].numel() for _, s in named_state)
        log(f"step{step}: params={len(named_state)} elements={n_elements} "
            f"lr={lr:.6g} betas=({beta1},{beta2}) eps={eps:g} stored_steps={steps_stored} "
            f"wd_map={wd_map}")
        checkpoints_meta[str(step)] = {
            "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps,
            "stored_step": steps_stored, "wd_map": {str(k): v for k, v in wd_map.items()},
            "n_elements": n_elements, "hyperparams_unique": len(hp_unique) == 1,
            "optim_sha256": chk["sha256"],
        }

        def new_acc():
            return {
                "n": 0, "ez": 0, "bf16": 0, "fp16": 0,
                "mz": 0, "mbf16": 0, "mfp16": 0,
                "thr": {m: 0 for m in f1.SQRTV_THRESHOLDS},
                "vq": [], "mq": [], "gq": [],
            }

        acc = {g: new_acc() for g in MODULE_GROUPS}
        for name, slot in named_state:
            v32 = slot["exp_avg_sq"]
            m32 = slot["exp_avg"]
            step_t = slot["step"]
            grp_idx = next(i for i, names in enumerate(meta["group_names"]) if name in names)
            lr_p = meta["groups"][grp_idx]["lr"]
            for gname, sl in f1.classify_groups(name):
                v_s = (v32[sl] if sl != slice(None) else v32)
                if v_s.numel() == 0:
                    continue
                m_s = (m32[sl] if sl != slice(None) else m32)
                v64 = v_s.detach().to(torch.float64)
                m64 = m_s.detach().to(torch.float64)
                a = acc[gname]
                fs = f1.floor_stats(v64, eps)
                ms = f1.signed_abs_stats(m64)
                gs = f1.gain_stats(f1.diagonal_gain(v_s, step_t, lr_p, beta1, beta2, eps,
                                                    compute_dtype=torch.float64))
                rt = f1.precision_roundtrip(v_s, m_s)
                a["n"] += fs["n_coords"]
                a["ez"] += fs["exact_zero_count"]
                a["bf16"] += rt["v_bf16_rt0"]
                a["fp16"] += rt["v_fp16_rt0"]
                a["mz"] += rt["m_math_zero"]
                a["mbf16"] += rt["m_bf16_rt0"]
                a["mfp16"] += rt["m_fp16_rt0"]
                for m in f1.SQRTV_THRESHOLDS:
                    a["thr"][m] += int(round(fs[f"sqrt_v_lt_{int(m)}eps_fraction"] * fs["n_coords"]))
                a["vq"].append((fs["min"], fs["p0_001"], fs["p0_01"], fs["p0_1"], fs["p5"],
                                fs["median"], fs["p99"], fs["p99_9"], fs["max"]))
                a["mq"].append((ms["m_abs_min"], ms["m_abs_p0_001"], ms["m_abs_p0_01"],
                                ms["m_abs_p0_1"], ms["m_abs_median"], ms["m_abs_p99"],
                                ms["m_abs_p99_9"], ms["m_abs_max"]))
                a["gq"].append((gs["gain_min"], gs["gain_p0_001"], gs["gain_median"],
                                gs["gain_p99_9"], gs["gain_max"]))
                del v64, m64
            del v32, m32
        del payload, named_state

        def wmin(entries, i):
            return min(e[i] for e in entries) if entries else float("nan")

        def wmax(entries, i):
            return max(e[i] for e in entries) if entries else float("nan")

        for gname in MODULE_GROUPS:
            a = acc[gname]
            if a["n"] == 0:
                continue
            row = {
                "step": step, "module": gname, "n_coords": a["n"],
                "v_exact_zero_count": a["ez"],
                "v_exact_zero_fraction": a["ez"] / a["n"],
                "v_min_worst": wmin(a["vq"], 0), "v_p0_001_worst": wmin(a["vq"], 1),
                "v_p0_01_worst": wmin(a["vq"], 2), "v_p0_1_worst": wmin(a["vq"], 3),
                "v_p5_worst": wmin(a["vq"], 4), "v_median_worst": wmin(a["vq"], 5),
                "v_max_max": wmax(a["vq"], 8),
                "v_bf16_rt0": a["bf16"], "v_fp16_rt0": a["fp16"],
                "m_abs_min_worst": wmin(a["mq"], 0), "m_abs_p0_001_worst": wmin(a["mq"], 1),
                "m_abs_median_worst": wmin(a["mq"], 4), "m_abs_max_max": wmax(a["mq"], 7),
                "m_zero": a["mz"], "m_bf16_rt0": a["mbf16"], "m_fp16_rt0": a["mfp16"],
                "gain_min": wmin(a["gq"], 0), "gain_p0_001_worst": wmin(a["gq"], 1),
                "gain_median_worst": wmin(a["gq"], 2), "gain_p99_9_max": wmax(a["gq"], 3),
                "gain_max": wmax(a["gq"], 4),
            }
            for m in f1.SQRTV_THRESHOLDS:
                row[f"sqrt_v_lt_{int(m)}eps_fraction"] = a["thr"][m] / a["n"]
            rows.append(row)
        log(f"step{step} aggregated ({time.time()-t0:.0f}s)")
        steps_done.append(step)

    if not rows:
        log("no checkpoints ready yet")
        return 1
    with (OUT / "state_floor_by_module.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "state_audit_meta.json").write_text(json.dumps(
        {"checkpoints": checkpoints_meta, "rows": len(rows),
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    log(f"written {len(rows)} rows for steps {steps_done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
