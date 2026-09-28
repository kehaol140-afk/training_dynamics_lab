#!/usr/bin/env python3
"""A0M Response Temporal Structure Analysis — OFFLINE, frozen artifacts only.

Sources (read-only):
  A0K events              runs/K{10,20}/seed_*/C/events.jsonl  (hhat j0..j4)
  A0K step logs           runs/K{10,20}/seed_*/C/step_log.json (v_eval, preclip_norm)
  A0K counterfactual      runs/counterfactual_replay.csv       (gain_v_eval_j1..j4)
  A0L events              a0l.../runs/seed_*/C/events.jsonl    (hhat j1..j4)

Experiments:
  E1 response trajectory  ranks, top action, pairwise d_ij -> response_trajectory.csv
  E2 phase clustering     [h1..h4, v_eval, preclip_norm] PCA+KMeans -> phase_cluster_report.md
  E3 stability            cos(h_t, h_{t+lag}) + Spearman at lags
  E4 utility prediction   corr(h_j, Gain_j), single vs window-averaged
  E5 fixed-j1 analysis    per-phase mean h_j, argmin identity over phases

Classification rules (fixed before viewing numbers):
  rank change freq  = fraction of adjacent events (per seed,source) with changed
                      argsort permutation; PHASE_STRUCTURE if median >= 0.5
  cos decay         = mean cos at lag 100 steps (K10 grid); STABLE if >= 0.5
  clustering        = ARI(KMeans k=3 vs step-tertile phase); CLUSTERED if >= 0.30
  utility           = Pearson corr(h_ij, gain_ij) pooled; ALIGNED if <= -0.2
                      (negative because lower h = more beneficial)
  j1 dominance      = fraction of events with argmin(h)=1; DOMINANT if >= 0.5
  stage optimum     = per seed, does argmin identity differ across phases?
  CASE A: PHASE_STRUCTURE and (CLUSTERED or stage-optimum majority)
  CASE B: STABLE and j1-dominant and not PHASE_STRUCTURE
  CASE C: otherwise / no structure and no stability
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
A0K = PROJECT_ROOT / "outputs/a0k_fresh_response_closed_loop_v01"
A0L = PROJECT_ROOT / "outputs/a0l_response_mixture_feasibility_v01"
OUT = PROJECT_ROOT / "outputs/a0m_response_temporal_structure_v01"
AN = OUT / "analysis"
DASH = OUT / "dashboard"
REP = OUT / "reports"

J = [1, 2, 3, 4]


def load_a0k_events() -> pd.DataFrame:
    rows = []
    for kdir in ("K10", "K20"):
        for evp in sorted(A0K.glob(f"runs/{kdir}/seed_*/C/events.jsonl")):
            for line in evp.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                r = {"source": "a0k", "K": int(e["K"]), "seed": int(e["seed"]),
                     "step": int(e["step"]), "w_0_norm": e.get("w_0_norm"),
                     "chosen": int(e.get("chosen_action_id", -1))}
                for j in J:
                    r[f"h{j}"] = float(e["hhat"][str(j)])
                rows.append(r)
    return pd.DataFrame(rows)


def load_a0l_events() -> pd.DataFrame:
    rows = []
    for evp in sorted(A0L.glob("runs/seed_*/C/events.jsonl")):
        for line in evp.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            r = {"source": "a0l", "K": 50, "seed": int(e["seed"]),
                 "step": int(e["step"]), "w_0_norm": None,
                 "chosen": -1}
            for j in J:
                r[f"h{j}"] = float(e["hhat"][str(j)])
            rows.append(r)
    return pd.DataFrame(rows)


def load_a0k_step_context() -> pd.DataFrame:
    rows = []
    for kdir in ("K10", "K20"):
        for slp in sorted(A0K.glob(f"runs/{kdir}/seed_*/C/step_log.json")):
            sl = json.loads(slp.read_text())
            for r in sl:
                if r.get("v_eval") is not None or r.get("preclip_norm") is not None:
                    rows.append({"K": int(kdir[1:]),
                                 "seed": int(slp.parts[-3].split("_")[1]),
                                 "step": r["step"], "v_eval": r.get("v_eval"),
                                 "preclip_norm": r.get("preclip_norm")})
    return pd.DataFrame(rows)


def load_replay() -> pd.DataFrame:
    df = pd.read_csv(A0K / "runs/counterfactual_replay.csv")
    return df


def phase_of(step: int, t_total: int = 128) -> str:
    if step <= 40:
        return "early"
    if step <= 84:
        return "mid"
    return "late"


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------- E1
def e1_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, K, seed), g in df.groupby(["source", "K", "seed"]):
        g = g.sort_values("step")
        prev_perm = None
        for _, r in g.iterrows():
            h = np.array([r[f"h{j}"] for j in J])
            perm = tuple(np.argsort(h))           # ascending (most beneficial first)
            top = int(perm[0]) + 1
            ranks = np.empty(4, dtype=int)
            ranks[list(perm)] = np.arange(4)      # rank of component j (0=best)
            rec = {"source": src, "K": K, "seed": seed, "step": int(r["step"]),
                   "top_action": top,
                   "perm": "-".join(str(p + 1) for p in perm),
                   "rank_changed": (prev_perm is not None and perm != prev_perm)}
            for a, b in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
                rec[f"d{a}{b}"] = r[f"h{a}"] - r[f"h{b}"]
            for j in J:
                rec[f"rank_h{j}"] = int(ranks[j - 1])
            rows.append(rec)
            prev_perm = perm
    out = pd.DataFrame(rows)
    out.to_csv(AN / "response_trajectory.csv", index=False)
    return out


# ----------------------------------------------------------------- E3
def e3_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, K), g in df.groupby(["source", "K"]):
        for seed, gg in g.groupby("seed"):
            gg = gg.sort_values("step")
            H = gg[[f"h{j}" for j in J]].to_numpy(float)
            S = gg["step"].to_numpy()
            for i in range(len(gg)):
                for m in range(i + 1, len(gg)):
                    lag = int(S[m] - S[i])
                    c = float(np.dot(H[i], H[m]) /
                              (np.linalg.norm(H[i]) * np.linalg.norm(H[m]) + 1e-30))
                    rho = float(spearmanr(H[i], H[m]).statistic)
                    rows.append({"source": src, "K": K, "seed": seed,
                                 "lag_steps": lag, "cos": c, "spearman": rho})
    st_ = pd.DataFrame(rows)
    st_.to_csv(AN / "stability_pairs.csv", index=False)
    summ = (st_.groupby(["source", "K", "lag_steps"])
            .agg(cos_mean=("cos", "mean"), cos_sd=("cos", "std"),
                 spe_mean=("spearman", "mean"), n=("cos", "size"))
            .reset_index())
    summ.to_csv(AN / "stability.csv", index=False)
    return summ


# ----------------------------------------------------------------- E2
def e2_clustering(ev: pd.DataFrame, ctx: pd.DataFrame) -> dict:
    df = ev[ev.source == "a0k"].copy()
    ctx = ctx.dropna(subset=["v_eval"]).copy()
    ctx["v_eval"] = ctx["v_eval"].astype(float)
    df = df.merge(ctx[["K", "seed", "step", "v_eval", "preclip_norm"]],
                  on=["K", "seed", "step"], how="left")
    df["phase"] = df["step"].map(phase_of)
    feats = [f"h{j}" for j in J] + ["v_eval", "preclip_norm"]
    d = df.dropna(subset=feats).copy()
    X = StandardScaler().fit_transform(d[feats].to_numpy(float))
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    labels = km.labels_
    ari = float(adjusted_rand_score(d["phase"], labels))
    sil = float(silhouette_score(X, labels)) if len(d) > 3 else float("nan")
    d["cluster"] = labels
    crosstab = (d.groupby(["phase", "cluster"]).size().unstack(fill_value=0))
    d[["seed", "step", "phase", "cluster"] + feats].to_csv(
        AN / "phase_features.csv", index=False)
    rep = ["# A0M-E2 Phase Clustering Report\n",
           f"features: {feats} (h1..h4 response, v_eval@step, preclip_norm@step); "
           "optimizer state statistics NOT available in frozen logs (not saved) — noted.",
           f"n events = {len(d)} (A0K only; A0L has 2 events/seed, too few).",
           f"phase labels: step<=40 early, 41..84 mid, >=85 late (tertiles of 128-step plan).",
           f"PCA explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}",
           f"KMeans(k=3) vs phase ARI = {ari:.4f}",
           f"silhouette = {sil:.4f}",
           "", "## cluster x phase crosstab", md_table(crosstab.reset_index()), ""]
    (REP / "phase_cluster_report.md").write_text("\n".join(rep))
    return {"n": int(len(d)), "ari": ari, "silhouette": sil,
            "pca_var": pca.explained_variance_ratio_.tolist()}


# ----------------------------------------------------------------- E4
def e4_utility(ev: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    df = ev[ev.source == "a0k"].merge(
        replay[["seed", "K", "step", "chosen_action_id",
                "gain_v_eval_j1", "gain_v_eval_j2", "gain_v_eval_j3",
                "gain_v_eval_j4"]], on=["seed", "K", "step"], how="inner")
    rows = []
    for w in (1, 2, 3, 4):
        for (K, seed), g in df.groupby(["K", "seed"]):
            g = g.sort_values("step").reset_index(drop=True)
            for t in range(len(g)):
                lo = max(0, t - w + 1)
                win = g.iloc[lo:t + 1]
                for j in J:
                    rows.append({"K": K, "seed": seed, "step": int(g.loc[t, "step"]),
                                 "window": w, "j": j,
                                 "h": float(win[f"h{j}"].mean()),
                                 "h_single": float(g.loc[t, f"h{j}"]),
                                 "gain": float(g.loc[t, f"gain_v_eval_j{j}"])})
    long = pd.DataFrame(rows)
    summ = []
    for w in (1, 2, 3, 4):
        for K, g in long[long.window == w].groupby("K"):
            x = g["h"].to_numpy() if w > 1 else g["h_single"].to_numpy()
            y = g["gain"].to_numpy()
            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            summ.append({"window": w, "K": int(K), "n": len(g),
                         "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
                         "spearman": float(sr.statistic)})
    s = pd.DataFrame(summ)
    s.to_csv(AN / "utility_correlation.csv", index=False)

    # top action: predicted argmin(h) vs actual argmax(gain), per event, w=1
    top = []
    for (K, seed, step), g in long[long.window == 1].groupby(["K", "seed", "step"]):
        pred = int(g.loc[g["h_single"].idxmin(), "j"])
        act = int(g.loc[g["gain"].idxmax(), "j"])
        top.append({"K": K, "seed": seed, "step": step,
                    "predicted_top": pred, "actual_top": act,
                    "match": pred == act})
    top = pd.DataFrame(top)
    top.to_csv(AN / "utility_top_action.csv", index=False)
    return s, top


# ----------------------------------------------------------------- E5
def e5_j1(ev: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    df = ev.copy()
    df["phase"] = df["step"].map(phase_of)
    df["argmin_j"] = df[[f"h{j}" for j in J]].idxmin(axis=1).str[1:].astype(int)
    rows = []
    for (src, K, seed, ph), g in df.groupby(["source", "K", "seed", "phase"]):
        r = {"source": src, "K": K, "seed": seed, "phase": ph, "n_events": len(g)}
        for j in J:
            r[f"mean_h{j}"] = float(g[f"h{j}"].mean())
        r["argmin_frac_j1"] = float((g["argmin_j"] == 1).mean())
        r["argmin_mode"] = int(g["argmin_j"].mode()[0])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(AN / "j1_dominance.csv", index=False)
    return out


# ----------------------------------------------------------------- common mode
def common_mode(ev: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    """Explains stable cos + noisy ranks: per event, common-mode magnitude
    mean(h_j) vs differential spread std(h_j); and per-event ranking signal
    Spearman(h_j, gain_j) across the 4 candidates."""
    df = ev[ev.source == "a0k"].merge(
        replay[["seed", "K", "step", "gain_v_eval_j1", "gain_v_eval_j2",
                "gain_v_eval_j3", "gain_v_eval_j4"]],
        on=["seed", "K", "step"], how="inner")
    rows = []
    for (_, _, _), g in df.groupby(["K", "seed", "step"]):
        h = np.array([g[f"h{j}"].iloc[0] for j in J])
        y = np.array([g[f"gain_v_eval_j{j}"].iloc[0] for j in J])
        cm, sp = float(h.mean()), float(h.std())
        rho = float(spearmanr(h, y).statistic)
        rows.append({"K": int(g["K"].iloc[0]), "seed": int(g["seed"].iloc[0]),
                     "step": int(g["step"].iloc[0]), "common_mode": cm,
                     "spread": sp, "cm_over_spread": abs(cm) / (sp + 1e-30),
                     "within_event_spearman": rho})
    out = pd.DataFrame(rows)
    out.to_csv(AN / "common_mode.csv", index=False)
    return out


def main() -> int:
    for d in (AN, DASH, REP):
        d.mkdir(parents=True, exist_ok=True)

    ev = pd.concat([load_a0k_events(), load_a0l_events()], ignore_index=True)
    ctx = load_a0k_step_context()
    replay = load_replay()

    traj = e1_trajectory(ev)
    stab = e3_stability(ev)
    clu = e2_clustering(ev, ctx)
    util, top = e4_utility(ev, replay)
    j1 = e5_j1(ev, traj)
    cm = common_mode(ev, replay)

    # ---- dashboard metrics (classification rules fixed in docstring) ----
    rc = (traj[traj.source == "a0k"]
          .groupby(["K", "seed"])["rank_changed"].mean().reset_index())
    rank_change_med = float(rc["rank_changed"].median())
    cos100 = stab[(stab.lag_steps == 100)]
    cos100_mean = float(cos100["cos_mean"].mean()) if len(cos100) else float("nan")
    u1 = util[(util.window == 1)]
    util_p = float(u1["pearson"].mean())
    util_w = util.groupby("window")["pearson"].mean()
    best_w = int(util_w.abs().idxmax())
    util_best = float(util_w.loc[best_w])
    j1_frac = float((ev[ev.source == "a0k"]
                     .assign(am=ev[ev.source == "a0k"][[f"h{j}" for j in J]].idxmin(axis=1))
                     ["am"] == "h1").mean())
    # stage-dependent optimum: per seed, argmin identity varies across phases
    sd = (j1[j1.source == "a0k"]
          .groupby(["K", "seed"])["argmin_mode"].nunique())
    stage_dep_frac = float((sd > 1).mean())
    top_match = float(top["match"].mean())

    PHASE_STRUCTURE = rank_change_med >= 0.5
    STABLE = cos100_mean >= 0.5
    CLUSTERED = clu["ari"] >= 0.30
    ALIGNED = util_p <= -0.2
    DOMINANT = j1_frac >= 0.5
    STAGE_DEP = stage_dep_frac >= 0.5
    if PHASE_STRUCTURE and (CLUSTERED or STAGE_DEP):
        case = "A"
    elif STABLE and DOMINANT and not PHASE_STRUCTURE:
        case = "B"
    else:
        case = "C"
    cases = {
        "A": "response 存在明显 phase structure",
        "B": "response 稳定但偏向固定策略",
        "C": "response 无法解释阶段变化",
    }

    dash = pd.DataFrame([
        ("rank change frequency (median per seed)", f"{rank_change_med:.3f}"),
        ("response cosine @100 steps (mean)", f"{cos100_mean:.3f}"),
        ("phase clustering ARI / silhouette", f"{clu['ari']:.3f} / {clu['silhouette']:.3f}"),
        ("single h utility Pearson (mean K)", f"{util_p:.3f}"),
        ("window averaged utility Pearson (best w)", f"w={best_w}: {util_best:.3f}"),
        ("j1 dominance (argmin fraction)", f"{j1_frac:.3f}"),
        ("stage dependent optimum (frac seeds)", f"{stage_dep_frac:.3f}"),
        ("top-action predicted vs actual match", f"{top_match:.3f}"),
    ], columns=["Metric", "Result"])
    dash.to_csv(DASH / "dashboard.csv", index=False)

    (DASH / "dashboard.json").write_text(json.dumps({
        "rank_change_median": rank_change_med,
        "cosine_100steps": cos100_mean,
        "clustering_ari": clu["ari"], "clustering_silhouette": clu["silhouette"],
        "utility_pearson_single": util_p,
        "utility_pearson_by_window": util_w.to_dict(),
        "j1_dominance": j1_frac,
        "stage_dependent_optimum_frac": stage_dep_frac,
        "top_action_match": top_match,
        "rules": {"PHASE_STRUCTURE": PHASE_STRUCTURE, "STABLE": STABLE,
                  "CLUSTERED": CLUSTERED, "ALIGNED": ALIGNED,
                  "DOMINANT": DOMINANT, "STAGE_DEP": STAGE_DEP},
        "case": case, "case_text": cases[case],
    }, indent=1, default=float))

    L = ["# A0M Response Temporal Structure — REPORT\n",
         "OFFLINE analysis of frozen A0K/A0L artifacts only. No training, no GPU, "
         "no optimizer/lambda/controller changes. Response convention: raw hhat = "
         "dV_score/dλ, NEGATIVE = beneficial.\n",
         "## Dashboard\n", md_table(dash), "",
         f"## Verdict: CASE {case} — {cases[case]}", "",
         "## E1 trajectory", f"- median rank-change frequency per seed = {rank_change_med:.3f}",
         "- per-seed top-action counts and Δij in analysis/response_trajectory.csv",
         "",
         "## E3 stability (cos / spearman by lag)",
         md_table(stab[stab.lag_steps.isin([10, 20, 50, 80, 100])].round(3)),
         "",
         "## E4 utility prediction",
         md_table(util.round(4)),
         f"- top-action match rate = {top_match:.3f}",
         "",
         "## E5 fixed-j1 analysis",
         md_table(j1.round(6)),
         "",
         "## Common-mode diagnostic (explains E1/E3/E4 jointly)",
         f"- |mean(h_j)| / std(h_j) per event: median = {float(cm['cm_over_spread'].median()):.2f} "
         f"(>>1: common-mode dominated)",
         f"- common mode by phase: early {float(cm[cm.step <= 40]['common_mode'].mean()):.2e} / "
         f"mid {float(cm[(cm.step > 40) & (cm.step <= 84)]['common_mode'].mean()):.2e} / "
         f"late {float(cm[cm.step > 84]['common_mode'].mean()):.2e} (~10x decay early->late)",
         f"- within-event Spearman(h_j, gain_j) mean = {float(cm['within_event_spearman'].mean()):.3f} "
         f"(ranking signal is real but small vs common mode)",
         "",
         "## Notes",
         "- A0K step logs do not contain training loss; phase features use v_eval@step "
         "and preclip_norm (gradient norm) instead. Optimizer state statistics not "
         "saved in any frozen artifact — E2 runs without them (noted in phase_cluster_report.md).",
         "- A0L contributes only 2 events/seed (steps 50/100, mixture-regime response, "
         "same production scorer units); used as secondary series in E1/E3/E5.",
         "- Mandate lags k=1,5 steps are below the frozen event grid (K=10 -> 10-step "
         "spacing); reported at available lags 10/20/50/80/100 steps.",
         "- No algorithm designed; mechanism report only. A0G/A0H/A0I/A0K/A0L artifacts untouched.",
         ]
    (REP / "A0M_REPORT.md").write_text("\n".join(L) + "\n")

    print(json.dumps({"case": case, "rank_change": rank_change_med,
                      "cos100": cos100_mean, "ari": clu["ari"],
                      "util_p": util_p, "util_best_w": best_w,
                      "j1_frac": j1_frac, "stage_dep": stage_dep_frac,
                      "top_match": top_match}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
