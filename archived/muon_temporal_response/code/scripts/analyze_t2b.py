"""T2b stage 3: analysis and verdict (target_gain_summary.json)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "t2b_target_screen_v1"


def main():
    shadow = list(csv.DictReader(open(OUT / "shadow_scores.csv")))
    actual = [r for r in csv.DictReader(open(OUT / "actual_objectives.csv")) if r["repeat"] == "False"]
    rep_rows = [r for r in csv.DictReader(open(OUT / "actual_objectives.csv")) if r["repeat"] == "True"]
    lock = json.loads((OUT / "selection_lock.json").read_text())
    cfg = json.loads((ROOT / "configs" / "t2b_target_screen_v1.json").read_text())

    # repeatability
    e_repeat = 0.0
    if rep_rows:
        orig = [r for r in actual if r["window"] == rep_rows[0]["window"] and r["k"] == rep_rows[0]["k"]][0]
        e_repeat = abs(float(orig["step138_V"]) - float(rep_rows[0]["step138_V"]))
        e_repeat = max(e_repeat, abs(float(orig["step10_V"]) - float(rep_rows[0]["step10_V"])))
    delta_num = max(1e-8, 10 * e_repeat)
    delta_screen = cfg["screening"]["delta_screen_nats"]
    threshold = max(delta_screen, delta_num)

    def v(b, k, h):
        return float([r for r in actual if r["window"] == str(b) and r["k"] == str(k)][0][f"step{h}_V"])

    summary = {"thresholds": {"delta_screen": delta_screen, "e_repeat": e_repeat,
                              "delta_num": delta_num, "effective_threshold": threshold},
               "windows": {}, "verdict": None}

    for b in (0, 1):
        kF, kL, k0 = lock[f"W{b}"]["kF"], lock[f"W{b}"]["kL"], 0
        vs = {h: np.array([v(b, k, h) for k in range(8)]) for h in (10, 138)}
        s_fin = np.array([float(r["S_fin"]) for r in shadow if r["window"] == str(b)])
        s_lin = np.array([float(r["S_lin"]) for r in shadow if r["window"] == str(b)])
        G_F = {h: float(vs[h].mean() - vs[h][kF]) for h in (10, 138)}
        G_L = {h: float(vs[h].mean() - vs[h][kL]) for h in (10, 138)}
        G_pool = {h: float(vs[h].mean() - vs[h].min()) for h in (10, 138)}
        G_pi0 = {h: float(vs[h][k0] - vs[h][kF]) for h in (10, 138)}
        def rank_corr(a, bb):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(bb))
            return float(np.corrcoef(ra, rb)[0, 1])
        summary["windows"][f"W{b}"] = {
            "kF": kF, "kL": kL, "k0_selected_pi0": k0 == kF,
            "v_138_all": vs[138].tolist(), "v_10_all": vs[10].tolist(),
            "spread_138": float(vs[138].max() - vs[138].min()),
            "spread_10": float(vs[10].max() - vs[10].min()),
            "G_F": G_F, "G_L": G_L, "G_pool": G_pool, "G_pi0_minus_kF": G_pi0,
            "S_fin_vs_v138_spearman": rank_corr(-s_fin, -vs[138]),  # low S_fin should mean low v
            "S_lin_vs_v138_spearman": rank_corr(-s_lin, -vs[138]),
            "S_fin_vs_v10_spearman": rank_corr(-s_fin, -vs[10]),
            "abs_err_Sfin_vs_v138_for_kF": abs(float(s_fin[kF]) - vs[138][kF]),
        }

    w = summary["windows"]
    both_GF = all(w[f"W{b}"]["G_F"][138] >= threshold and
                  w[f"W{b}"]["G_F"][138] + w[f"W{b}"]["G_pi0_minus_kF"][138] >= -delta_num
                  for b in (0, 1))
    both_pool = all(w[f"W{b}"]["G_pool"][138] >= threshold for b in (0, 1))
    if both_GF:
        verdict = "TARGET_SELECTION_SIGNAL"
    elif both_pool:
        verdict = "EFFECT_WITHOUT_RELIABLE_SELECTION"
    else:
        verdict = "NO_USEFUL_SIGNAL_THIS_SETTING"
    summary["verdict"] = verdict
    summary["interpretation_notes"] = {
        "G_pool_is_post_hoc": "pool min comes from Q_check and is NOT a method or an upper bound claim",
        "one_checkpoint_two_windows": "not two independent training seeds; no seed-level significance",
        "step10_diagnostic_only": "prefix-end advantages that vanish after the common tail are memory-only",
    }
    json.dump(summary, open(OUT / "target_gain_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "interpretation_notes"}, indent=1))


if __name__ == "__main__":
    main()
