#!/usr/bin/env python3
"""F4B Stability-Aware Unrolling (frozen: docs/F4B_STABILITY_AWARE_UNROLLING_PREREG.md).

rollout <scale> <seed> <step> [--delta 0.05]
    Three CRN action rollouts (-d,0,+d) to H_eval=32; L = eval-set mean CE
    (fp32) at theta_{t+32}. Atomic per-cell JSON in outputs/f4b_sau/raw/.

analyze
    Loads F4A sensitivity profiles (g(H), lambda, q, h_eta) + rollout
    truths; simulates Fixed-H / BEST_FIXED_H_TRAIN / Oracle-H / SAU under
    LOSO-by-seed with frozen budget grid; emits all tables + verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs" / "f4b_sau"
OUTD = OUT  # delta-namespaced analysis output dir (anti-overwrite)
RAWF = OUT / "raw"
F4A_RAW = ROOT / "outputs" / "f4a_phase_map" / "raw"
F4A1_DIR = ROOT / "outputs" / "f4a1_matched_coverage" / "new_states"
H_GRID = (1, 2, 4, 8, 16, 32)
DELTA = 0.05
BUDGETS = [(8, 4, .10), (8, 4, .20), (16, 4, .10), (16, 8, .20),
           (32, 8, .10), (32, 8, .30), (64, 16, .20), (128, 16, .30)]
HMIN_GRID = (0.0, 0.005, 0.01, 0.02)
CELLS = ([(s, st) for s in (2042, 2043, 2044) for st in (26, 64, 102)]
         * 1)  # placeholder replaced below
SCALES = {"g1b_100m": [(s, st) for s in (2042, 2043, 2044)
                       for st in (26, 64, 102)],
          "g1a_26m": [(s, st) for s in (2042, 2043, 2044)
                      for st in (26, 64, 102)],
          "v07_5m": [(s, st) for s in range(2030, 2038)
                     for st in (26, 64, 102)]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cmd_rollout(scale, seed, step, delta, device="cuda:0") -> int:
    ud = RAWF / f"{scale}_{seed}_{step}_d{delta}"
    if (ud / "DONE").exists():
        log(f"rollout {scale} {seed}:{step} d={delta}: DONE exists")
        return 0
    ud.mkdir(parents=True, exist_ok=True)
    import torch
    import f3_curvature_mechanism as f3m
    import f4a_phase_map as f4a
    from f2_hypergradient import ACCUM, TOTAL_MICRO, BASE_LR
    from opsens import f1_mechanism_v01 as fm
    from opsens import f0_bank_audit_v01 as f0

    t0 = time.time()
    f3m.rig = f4a.F4ARig(device, scale)
    if scale == "v07_5m" and (seed, step) in {(s, t) for s in range(2030, 2038)
                                              for t in (26, 102)}:
        bp = F4A1_DIR / "v07_5m" / f"seed_{seed}" / f"state_step_{step:03d}.pt"
    else:
        bp = fm.bank_state_path(scale, seed, step)
    f3m.state_g = f0.load_bank_state(bp)
    f3m.micro0 = int(f3m.state_g["metadata"]["global_micro_step"])
    # frozen stream: reuse the F4A/F4A.1 stream id recorded in config.json
    cfg_path = F4A_RAW / scale / f"seed_{seed}" / f"step_{step}" / "config.json"
    stream_id = json.loads(cfg_path.read_text())["stream_id"]
    f3m.batches_g, stream_seed = f3m.rig.batches_for(stream_id)
    f3m.named, f3m.opt, f3m.model_cpu = f3m.rig.restore(f3m.state_g)
    orig_get_lr = f3m.rig.get_lr
    res = {"scale": scale, "seed": seed, "step": step, "delta": delta,
           "stream_id": stream_id, "stream_seed": stream_seed, "L": {}}
    for a in (-delta, 0.0, delta):
        f3m.named, f3m.opt, f3m.model_cpu = f3m.rig.restore(f3m.state_g)
        scale_f = math.exp(a)
        f3m.rig.get_lr = (lambda sf: (lambda m, t, b: orig_get_lr(m, t, b) * sf))(scale_f)
        for h in range(1, 33):
            f3m.primal_grads(h - 1, f3m.micro0)
            f3m.primal_advance()
        f3m.rig.get_lr = orig_get_lr
        with torch.no_grad():
            f3m.rig.sync_gpu(f3m.model_cpu, train=False)
            L = float(f3m.rig.eval_ce(f3m.model_cpu, autocast=False))
        res["L"][f"{a:+g}"] = L
        log(f"rollout {scale} {seed}:{step} a={a:+g} L32={L:.6f}")
    tmp = ud / "rollout.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, ud / "rollout.json")
    (ud / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(f"rollout {scale} {seed}:{step}: COMPLETE ({time.time()-t0:.0f}s)")
    return 0


# ---------------------------------------------------------------- analyze ----
def load_cells(delta=DELTA):
    cells = []
    for scale, pairs in SCALES.items():
        for seed, step in pairs:
            mp = F4A_RAW / scale / f"seed_{seed}" / f"step_{step}" / "metrics.json"
            rp = RAWF / f"{scale}_{seed}_{step}_d{delta}" / "rollout.json"
            if not (mp.exists() and rp.exists()):
                continue
            m = json.loads(mp.read_text())
            r = json.loads(rp.read_text())
            g = {H: m["G"][str(H)]["G_C_theta"] for H in H_GRID}
            heta = {H: m["G"][str(H)].get("h_eta") for H in H_GRID}
            rcurv = m["G"]["32"]["R_curv"]
            cells.append({
                "scale": scale, "seed": seed, "step": step,
                "progress": step / 128,
                "regime": ("dominant" if rcurv >= 10 else
                           "neutral" if rcurv < 3 else "middle"),
                "g": g, "h": heta,
                "Lm": float(r["L"][f"{-delta:+g}"]),
                "L0": float(r["L"][f"{0.0:+g}"]),
                "Lp": float(r["L"][f"{delta:+g}"]),
                "g32_raw": g[32], "R_curv_32": rcurv})
    return cells


def sau_horizon(c, bg, bq, blam):
    Hs = 1
    for H in H_GRID:
        ok = c["g"][H] <= bg and (H == 1 or
                                  c["g"][H] / (c["g"][H // 2] + 1e-30) <= bq) and \
             (math.log(max(c["g"][H], 1e-30)) / H) <= blam
        if ok:
            Hs = H
        else:
            break
    return Hs


def action_and_regret(c, H, hmin):
    h = c["h"].get(H)
    if h is None:
        h = 0.0
    if abs(h) < hmin:
        a = 0.0
    else:
        a = (-0.05 if h > 0 else 0.05)
    L = {-0.05: c["Lm"], 0.0: c["L0"], 0.05: c["Lp"]}
    Lmin = min(L.values())
    Lmax = max(L.values())
    reg = L[a] - Lmin
    return {"H": H, "a": a, "L": L[a], "regret": reg,
            "regret_norm": reg / (Lmax - Lmin + 1e-30),
            "harmful": L[a] > c["L0"]}


def cmd_analyze(delta=DELTA) -> int:
    global OUTD
    OUTD = OUT / ("delta_" + (f"{delta:g}".replace(".", "p")))
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "figures").mkdir(exist_ok=True)
    cells = load_cells(delta)
    log(f"analyze: {len(cells)} cells")
    seeds = sorted({c["seed"] for c in cells})
    fold_of = lambda c: c["seed"]

    def eval_method(method, budget=None, hmin=0.0, fixed_H=None, fold=None):
        rows = []
        for c in cells:
            if fold is not None and c["seed"] != fold:
                continue
            if method == "SAU":
                H = sau_horizon(c, *budget)
                rows.append(action_and_regret(c, H, hmin) | {"cell": c})
            elif method == "FIXED":
                rows.append(action_and_regret(c, fixed_H, hmin) | {"cell": c})
            elif method == "ORACLE":
                cands = [action_and_regret(c, H, 0.0) | {"cell": c}
                         for H in H_GRID]
                best = min(cands, key=lambda r: r["regret"])
                rows.append(best)
        return rows

    # per-fold hyperparameter selection (train side)
    fold_hp = {}
    for fold in seeds:
        train = [c for c in cells if c["seed"] != fold]
        best = None
        for budget in BUDGETS:
            for hmin in HMIN_GRID:
                rows = [action_and_regret(c, sau_horizon(c, *budget), hmin)
                        for c in train]
                mr = float(np.median([r["regret_norm"] for r in rows]))
                medH = float(np.median([sau_horizon(c, *budget) for c in train]))
                key = (mr, -medH, BUDGETS.index(budget), HMIN_GRID.index(hmin))
                if best is None or key < best[0]:
                    best = (key, budget, hmin)
        # BEST_FIXED_H on train (with the same hmin for fairness)
        bf = None
        for hmin in HMIN_GRID:
            for H in H_GRID:
                rows = [action_and_regret(c, H, hmin) for c in train]
                mr = float(np.mean([r["regret_norm"] for r in rows]))
                if bf is None or mr < bf[0]:
                    bf = (mr, H, hmin)
        fold_hp[fold] = {"budget": best[1], "hmin": best[2],
                         "best_fixed_H": bf[1], "best_fixed_hmin": bf[2]}
    # evaluate on held-out folds
    method_rows = {m: [] for m in ("SAU", "BEST_FIXED", "ORACLE",
                                   *(f"FIXED{H}" for H in H_GRID))}
    for fold in seeds:
        hp = fold_hp[fold]
        for r in eval_method("SAU", budget=hp["budget"], hmin=hp["hmin"],
                             fold=fold):
            r["fold"] = fold
            method_rows["SAU"].append(r)
        for r in eval_method("FIXED", fixed_H=hp["best_fixed_H"],
                             hmin=hp["best_fixed_hmin"], fold=fold):
            r["fold"] = fold
            method_rows["BEST_FIXED"].append(r)
        for r in eval_method("ORACLE", fold=fold):
            r["fold"] = fold
            method_rows["ORACLE"].append(r)
        for H in H_GRID:
            for r in eval_method("FIXED", fixed_H=H, hmin=0.0, fold=fold):
                r["fold"] = fold
                method_rows[f"FIXED{H}"].append(r)

    def stats(rows, key="regret_norm"):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "max": float(v.max()), "n": len(v)}

    per_method = {}
    for m, rows in method_rows.items():
        s = stats(rows)
        s["harmful_rate"] = float(np.mean([r["harmful"] for r in rows]))
        s["mean_H"] = float(np.mean([r["H"] for r in rows]))
        s["median_H"] = float(np.median([r["H"] for r in rows]))
        s["oracle_acc"] = float(np.mean([r["regret"] < 1e-12 for r in rows]))
        per_method[m] = s
    # per-subset
    def subset(m, regime=None, key_scale=None):
        rows = [r for r in method_rows[m]
                if (regime is None or r["cell"]["regime"] == regime)
                and (key_scale is None or r["cell"]["scale"] == key_scale)]
        return stats(rows) if rows else None
    # gate evaluation
    sau, bf = per_method["SAU"], per_method["BEST_FIXED"]
    dom_sau, dom_bf = subset("SAU", "dominant"), subset("BEST_FIXED", "dominant")
    neu_sau, neu_bf = subset("SAU", "neutral"), subset("BEST_FIXED", "neutral")
    gate = {
        "A_regret_median": sau["median"] < bf["median"],
        "B_harmful": sau["harmful_rate"] <= bf["harmful_rate"] + 1e-12,
        "C_dominant": (dom_sau and dom_bf and
                       dom_sau["median"] < dom_bf["median"]),
        "D_neutral": (neu_sau and neu_bf and
                      neu_sau["median"] <= 1.5 * neu_bf["median"]),
        "E_compute": sau["mean_H"] <= 32.0}
    primary_pass = all(gate.values())
    # practical gain: >=20% median improvement, majority of seeds same dir
    imp = (bf["median"] - sau["median"]) / max(bf["median"], 1e-30)
    seed_dirs = []
    for fold in seeds:
        s_ = np.median([r["regret_norm"] for r in method_rows["SAU"]
                        if r["fold"] == fold])
        b_ = np.median([r["regret_norm"] for r in method_rows["BEST_FIXED"]
                        if r["fold"] == fold])
        seed_dirs.append(s_ <= b_)
    practical = bool(imp >= 0.20 and sum(seed_dirs) > len(seeds) / 2)
    if not primary_pass:
        verdict = ("NO_METHOD_GAIN" if imp <= 0 else
                   "SAU_COMPUTE_QUALITY_TRADEOFF" if gate["E_compute"] and
                   gate["A_regret_median"] else "NO_METHOD_GAIN")
        if gate["B_harmful"] and gate["A_regret_median"]:
            verdict = "SAU_REDUCES_HARMFUL_DECISIONS"
    elif practical:
        verdict = "SAU_OUTPERFORMS_FIXED_H"
    else:
        verdict = "SAU_COMPUTE_QUALITY_TRADEOFF" if sau["mean_H"] < 32 else \
                  "SAU_OUTPERFORMS_FIXED_H"
    summary = {
        "study": "F4B_SAU", "n_cells": len(cells), "delta": delta,
        "verdict": verdict, "primary_gate": gate, "primary_pass": primary_pass,
        "practical_gain": practical, "median_improvement_frac": round(imp, 4),
        "per_method": per_method,
        "subsets": {"dominant": {"SAU": dom_sau, "BEST_FIXED": dom_bf},
                    "neutral": {"SAU": neu_sau, "BEST_FIXED": neu_bf}},
        "fold_hp": {str(k): {"budget": list(v["budget"]), "hmin": v["hmin"],
                             "best_fixed_H": v["best_fixed_H"],
                             "best_fixed_hmin": v["best_fixed_hmin"]}
                    for k, v in fold_hp.items()},
        "model_prescreen": "DISABLED_BY_PREREG"}
    (OUTD / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"verdict": verdict, "gate": gate,
                      "SAU_med": round(sau["median"], 4),
                      "BF_med": round(bf["median"], 4),
                      "SAU_H": sau["median_H"],
                      "imp": round(imp, 3)}, indent=1))
    # tables
    import csv
    with (OUTD / "per_method_regret.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "median_norm_regret", "mean", "max",
                    "harmful_rate", "median_H", "mean_H", "oracle_acc", "n"])
        for m, s in sorted(per_method.items()):
            w.writerow([m] + [round(s[k], 4) for k in
                              ("median", "mean", "max", "harmful_rate",
                               "median_H", "mean_H", "oracle_acc", "n")])
    with (OUTD / "per_seed_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "SAU_med", "BF_med", "SAU_harmful", "BF_harmful",
                    "SAU_medH"])
        for fold in seeds:
            s_ = [r for r in method_rows["SAU"] if r["fold"] == fold]
            b_ = [r for r in method_rows["BEST_FIXED"] if r["fold"] == fold]
            w.writerow([fold,
                        round(float(np.median([r["regret_norm"] for r in s_])), 4),
                        round(float(np.median([r["regret_norm"] for r in b_])), 4),
                        round(float(np.mean([r["harmful"] for r in s_])), 3),
                        round(float(np.mean([r["harmful"] for r in b_])), 3),
                        float(np.median([r["H"] for r in s_]))])
    with (OUTD / "selected_horizons.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "seed", "step", "regime", "g32", "H_safe",
                    "action", "L_chosen", "regret", "regret_norm"])
        for r in method_rows["SAU"]:
            c = r["cell"]
            w.writerow([c["scale"], c["seed"], c["step"], c["regime"],
                        f"{c['g32_raw']:.3g}", r["H"], r["a"], round(r["L"], 6),
                        round(r["regret"], 8), round(r["regret_norm"], 5)])
    with (OUTD / "action_outcomes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "seed", "step", "regime", "L_minus", "L_zero",
                    "L_plus", "range"])
        for c in cells:
            w.writerow([c["scale"], c["seed"], c["step"], c["regime"],
                        round(c["Lm"], 6), round(c["L0"], 6), round(c["Lp"], 6),
                        round(max(c["Lm"], c["L0"], c["Lp"]) -
                              min(c["Lm"], c["L0"], c["Lp"]), 8)])
    with (OUTD / "threshold_selection.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["held_out_seed", "budget(Bg,Bq,Blam)", "hmin",
                    "best_fixed_H", "best_fixed_hmin"])
        for k, v in fold_hp.items():
            w.writerow([k, v["budget"], v["hmin"], v["best_fixed_H"],
                        v["best_fixed_hmin"]])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rollout", "analyze"))
    ap.add_argument("--scale")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()
    if args.cmd == "analyze":
        return cmd_analyze(args.delta)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None
    return cmd_rollout(args.scale, args.seed, args.step, args.delta)


if __name__ == "__main__":
    sys.exit(main())
