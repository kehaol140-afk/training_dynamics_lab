#!/usr/bin/env python3
"""F4B observability postmortem (frozen: docs/F4B_POSTMORTEM_OBSERVABILITY_PREREG.md).
Rebuilds per-(delta,cell,H) utility from canonical action outcomes + F4A h_H,
then computes the observability frontier (H_suff, Benefit_continue, LOSO
online models, offline diagnostic). No new rollouts."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
F4A = ROOT / "outputs" / "f4a_phase_map" / "raw"
F4B = ROOT / "outputs" / "f4b_sau"
OUT = ROOT / "outputs" / "f4b_postmortem"
OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
H_GRID = (1, 2, 4, 8, 16, 32)
H_OBS = (1, 2, 4, 8, 16)


def load(delta):
    cells = []
    for r in csv.DictReader((F4B / f"delta_{delta:g}".replace(".", "p") /
                             "action_outcomes.csv").open()):
        mp = F4A / r["scale"] / f"seed_{r['seed']}" / f"step_{r['step']}" / "metrics.json"
        m = json.loads(mp.read_text())
        heta = {H: m["G"][str(H)].get("h_eta") for H in H_GRID}
        g = {H: m["G"][str(H)]["G_C_theta"] for H in H_GRID}
        cells.append({
            "scale": r["scale"], "seed": int(r["seed"]), "step": int(r["step"]),
            "progress": int(r["step"]) / 128,
            "regime": r["regime"],
            "Lm": float(r["L_minus"]), "L0": float(r["L_zero"]),
            "Lp": float(r["L_plus"]), "h": heta, "g": g,
            "R_curv_32": m["G"]["32"]["R_curv"],
            "lambda_SO_32": m["G"]["32"]["lambda_SO"],
            "clip": m["clip"]["clip_coefficient_1"],
            "logA_m": math.log10(max(m["indicators"]["A_m"], 1e-30)),
            "logA_v": math.log10(max(m["indicators"]["A_v"], 1e-30)),
            "vfloor": m["indicators"]["fraction_sqrtv_lt_eps"],
            "K_lr": math.log10(max(m["K_lr"], 1e-30))})
    return cells


def decide(c, H, delta, hmin=0.0):
    h = c["h"].get(H) or 0.0
    a = 0.0 if abs(h) < hmin else (-delta if h > 0 else delta)
    L = {-delta: c["Lm"], 0.0: c["L0"], delta: c["Lp"]}
    reg = L[a] - min(L.values())
    rng = max(L.values()) - min(L.values()) + 1e-30
    return a, reg, reg / rng


def run(delta):
    cells = load(delta)
    ub_rows, suff_rows, ben_rows = [], [], []
    for c in cells:
        per_H = {}
        for H in H_GRID:
            a, reg, nreg = decide(c, H, delta)
            per_H[H] = {"a": a, "reg": reg, "nreg": nreg,
                        "harmful": ({-delta: c["Lm"], 0.0: c["L0"],
                                     delta: c["Lp"]}[a] > c["L0"])}
        r_or = min(v["nreg"] for v in per_H.values())
        for H in H_GRID:
            ub_rows.append({"delta": delta, "scale": c["scale"], "seed": c["seed"],
                            "step": c["step"], "regime": c["regime"], "H": H,
                            "action": per_H[H]["a"], "regret": round(per_H[H]["reg"], 8),
                            "nregret": round(per_H[H]["nreg"], 5),
                            "harmful": per_H[H]["harmful"],
                            "oracle_match": per_H[H]["nreg"] <= r_or + 1e-12})
        for eps in (0.0, 0.05, 0.10):
            Hs = next((H for H in H_GRID
                       if per_H[H]["nreg"] <= r_or + eps), None)
            suff_rows.append({"delta": delta, "scale": c["scale"], "seed": c["seed"],
                              "step": c["step"], "regime": c["regime"],
                              "eps_u": eps, "H_suff": Hs,
                              "r_oracle_nreg": round(r_or, 5)})
        for h in H_OBS:
            fut = min(per_H[H]["nreg"] for H in H_GRID if H > h)
            ben = per_H[h]["nreg"] - fut
            ben_rows.append({"delta": delta, "scale": c["scale"], "seed": c["seed"],
                             "step": c["step"], "regime": c["regime"], "h": h,
                             "benefit": round(ben, 5), "cost": round(max(0, -ben), 5),
                             "worthwhile": ben >= 0.05,
                             **online_feats(c, h)})
    return cells, ub_rows, suff_rows, ben_rows


def online_feats(c, h):
    g1, gh = c["g"][1], c["g"][h]
    hprev = c["h"].get(next((x for x in H_GRID if x < h), 1))
    hcur = c["h"][h]
    lg = math.log10(max(gh, 1e-30))
    lam = math.log(max(gh, 1e-30)) / h
    q = gh / (c["g"][next((x for x in H_GRID if x < h), 1)] + 1e-30) if h > 1 else 1.0
    return {"log_g": round(lg, 4), "lambda_hat": round(lam, 4),
            "q": round(q, 4),
            "abs_h": round(abs(hcur) if hcur is not None else 0.0, 4),
            "sign_h": int(np.sign(hcur) if hcur is not None else 0),
            "sign_change": int((hcur is not None and hprev is not None
                                and np.sign(hcur) != np.sign(hprev))),
            "dlog_g": round(lg - math.log10(max(g1, 1e-30)), 4),
            "d_h": round((hcur - hprev) if (hcur is not None and hprev is not None)
                         else 0.0, 4),
            "clip": round(c["clip"], 4), "progress": round(c["progress"], 3),
            "logA_m": round(c["logA_m"], 3), "logA_v": round(c["logA_v"], 3),
            "vfloor": c["vfloor"], "K_lr": round(c["K_lr"], 3),
            "s26": int(c["scale"] == "g1a_26m"), "s100": int(c["scale"] == "g1b_100m")}


FEATS = ["log_g", "lambda_hat", "q", "abs_h", "sign_h", "sign_change",
         "dlog_g", "d_h", "clip", "progress", "logA_m", "logA_v", "vfloor",
         "K_lr", "s26", "s100"]


def loso_auc(X, y, seeds):
    """y: 1-d numpy int target; seeds: fold labels aligned to y."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, \
        average_precision_score
    y = np.asarray(y)
    preds = np.full(len(y), np.nan)
    for s in sorted(set(seeds.tolist())):
        te = np.asarray(seeds == s)
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    if len(set(y[ok])) < 2:
        return None
    return {"auroc": float(roc_auc_score(y[ok], preds[ok])),
            "balacc": float(balanced_accuracy_score(
                y[ok], (preds[ok] >= .5).astype(int))),
            "pr_auc": float(average_precision_score(y[ok], preds[ok])),
            "n_pos": int(y[ok].sum()), "n": int(ok.sum())}


def main() -> int:
    import pandas as pd
    all_ub, all_suff, all_ben = [], [], []
    frontier = []
    for delta in (0.05, 0.02):
        cells, ub, suff, ben = run(delta)
        all_ub += ub; all_suff += suff; all_ben += ben
        dfb = pd.DataFrame(ben)
        # per-h online observability
        online = {}
        for h in H_OBS:
            sub = dfb[dfb["h"] == h]
            y = sub["worthwhile"].astype(int).values
            X = sub[FEATS].values.astype(float)
            res = loso_auc(X, y, sub["seed"].values)
            res and res.update({"h": h, "delta": delta})
            online[h] = res
        H_sep = next((h for h in H_OBS
                      if online[h] and online[h]["auroc"] >= 0.75
                      and online[h]["balacc"] >= 0.65
                      and online[h]["n_pos"] >= 5), None)
        ds = pd.DataFrame(suff)
        d05 = ds[(ds["eps_u"] == 0.05)]
        Hs_le2 = float((d05["H_suff"] <= 2).mean())
        med_Hs = float(d05["H_suff"].median())
        # offline diagnostic: predict harmful(H=32) and worthwhile(1)
        dfu = pd.DataFrame([r for r in ub if r["delta"] == delta and r["H"] == 32])
        dfb1 = dfb[dfb["h"] == 1]
        off = {}
        offX = np.column_stack([
            np.array([c["R_curv_32"] for c in cells]),
            np.array([c["lambda_SO_32"] for c in cells])])
        # log-transform R_curv
        offX = np.column_stack([np.log10(np.maximum(offX[:, 0], 1e-30)), offX[:, 1]])
        yh = dfu["harmful"].astype(int).values
        off["harmful32"] = loso_auc(offX, yh, dfu["seed"].values) if len(set(yh)) > 1 else None
        frontier.append({
            "delta": delta, "H_sep": H_sep if H_sep else "NOT_OBSERVED",
            "median_H_suff_eps05": med_Hs,
            "frac_H_suff_le_2": Hs_le2,
            "online_by_h": {str(h): online[h] for h in H_OBS},
            "offline_harmful32": off["harmful32"]})
    pd.DataFrame(all_ub).to_csv(OUT / "utility_by_horizon.csv", index=False)
    pd.DataFrame(all_suff).to_csv(OUT / "horizon_sufficiency.csv", index=False)
    pd.DataFrame(all_ben).to_csv(OUT / "continue_benefit.csv", index=False)
    pd.DataFrame(all_ben)[["delta", "scale", "seed", "step", "h", "worthwhile",
                           "benefit"]].to_csv(OUT / "online_observability.csv",
                                                         index=False)
    # offline table
    import pandas as pd
    off_rows = []
    for f in frontier:
        o = f["offline_harmful32"]
        off_rows.append({"delta": f["delta"], "target": "harmful_H32",
                         "auroc": o and round(o["auroc"], 3),
                         "n_pos": o and o["n_pos"]})
    pd.DataFrame(off_rows).to_csv(OUT / "offline_diagnostic.csv", index=False)
    pd.DataFrame(frontier).to_json(OUT / "observability_frontier.json",
                                   orient="records", indent=1)
    # verdict (prereg §6)
    f05, f02 = frontier
    def gap(fr):
        Hsep = fr["H_sep"]
        Hsep_v = None if Hsep == "NOT_OBSERVED" else Hsep
        cond_i = fr["frac_H_suff_le_2"] > 0.5
        cond_ii = (Hsep_v is None) or (Hsep_v > 2)
        return cond_i, cond_ii
    i5, ii5 = gap(f05)
    offline_strong = any(
        (f["offline_harmful32"] or {}).get("auroc", 0) >= 0.75 for f in frontier)
    if i5 and ii5 and offline_strong:
        verdict = "OBSERVABILITY_GAP_SUPPORTED"
    elif any(f["H_sep"] in (1, 2) for f in frontier):
        verdict = "ONLINE_RISK_EARLY_OBSERVABLE"
    elif i5:
        verdict = "UTILITY_ALREADY_SHORT_HORIZON"
    else:
        verdict = "NO_GENERAL_STRUCTURE"
    summary = {"study": "F4B_POSTMORTEM_OBSERVABILITY",
               "verdict": verdict, "frontier": frontier,
               "offline_strong": offline_strong}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"verdict": verdict,
                      "H_sep_05": f05["H_sep"], "H_sep_02": f02["H_sep"],
                      "medHs_05": f05["median_H_suff_eps05"],
                      "medHs_02": f02["median_H_suff_eps05"],
                      "fracHs<=2_05": f05["frac_H_suff_le_2"],
                      "offline": [(f["delta"], f["offline_harmful32"] and
                                   round(f["offline_harmful32"]["auroc"], 3),
                                   f["offline_harmful32"] and f["offline_harmful32"]["n_pos"])
                                  for f in frontier]}, indent=1))
    # figure: mean normalized regret by H per delta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dfu = pd.DataFrame(all_ub)
    fig, ax = plt.subplots(figsize=(6, 4))
    for delta, mk in ((0.05, "o-"), (0.02, "s--")):
        sub = dfu[dfu["delta"] == delta].groupby("H")["nregret"].mean()
        ax.plot(sub.index, sub.values, mk, label=f"delta={delta}")
    ax.set_xlabel("H"); ax.set_ylabel("mean normalized regret")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig1_mean_regret_by_H.png", dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
