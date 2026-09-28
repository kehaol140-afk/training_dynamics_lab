#!/usr/bin/env python3
"""GTTR-R1 分析（CPU）：bootstrap CI、模型误差、结构增益与判定标签。

输入：outputs/fr_gttr_r1_objective_identification_v01/ 下已封存的 FIT/解与
AUDIT/check 原子数据；输出 audit_model_errors.csv、incremental_gain.csv、
model_scope_and_rejections.csv、costs.json、REPORT.md、KEY_DATA_FOR_REVIEW.md、
figures/（<=4 张）。不改任何 FIT 文件（seal 只读复核）。

口径（协议 §5）：
- bootstrap 以 effective-batch probe 为单位，1000 次，PCG64 seed 2026092301；
  FIT 与 AUDIT 独立重抽；k1/k2 与动作配对（同 state 共用索引）。
- 固定网格（15 个非 native 网格点）SSE/TSS；max|J_audit|<1e-9 -> LOW_SIGNAL。
- 判定：MODEL_RECOMMENDS_NO_READOUT_CHANGE / PREDICTED_BENEFIT_NOT_CONFIRMED
  / TARGET_TRANSFER_NOT_ESTABLISHED / SUPPORTS_FURTHER_VALIDATION / UNRESOLVED。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import numpy as np  # noqa: E402

from fr import r1_objective_identification_v01 as r1m  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
FIT_CSV = OUT / "per_probe_coefficients.csv"
MODEL_CSV = OUT / "model_coefficients.csv"
SOL_CSV = OUT / "restricted_solutions.csv"
SEAL_JSON = OUT / "prediction_seal.json"
AUDIT_CSV = OUT / "audit_target_values.csv"
CHECK_CSV = OUT / "check_transfer.csv"

R_PRIMARY, R_HALF = 0.05, 0.025
BOOT_N, BOOT_SEED = 1000, 2026092301
LOW_SIGNAL_ABS = 1e-9
COLLINEAR_COS = 0.999


def _read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(v):
    return float(v)


def _ci(samples, lo=2.5, hi=97.5):
    if not samples:
        return (float("nan"), float("nan"))
    a = np.asarray(samples, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi)))


def _pct_key(x, y):
    return (round(float(x), 9), round(float(y), 9))


def main():
    ok, why = r1m.verify_seal(SEAL_JSON, [FIT_CSV, MODEL_CSV, SOL_CSV])
    assert ok, f"seal verify failed: {why}"
    config = json.loads((OUT / "config.json").read_text())

    fit_rows = _read(FIT_CSV)
    model_rows = _read(MODEL_CSV)
    sol_rows = _read(SOL_CSV)
    audit_rows = _read(AUDIT_CSV)
    check_rows = _read(CHECK_CSV)

    # ---- 组织数据 ------------------------------------------------------
    fit_by_cond = defaultdict(list)
    for r in fit_rows:
        fit_by_cond[(r["seed"], r["start"], r["k"])].append(
            {"b1": _f(r["b1"]), "b2": _f(r["b2"]), "k11": _f(r["k11"]),
             "k12_direct": _f(r["k12_direct"]), "k21_direct": _f(r["k21_direct"]),
             "k22": _f(r["k22"]), "d0_norm": _f(r["d0_norm"]),
             "d1_norm": _f(r["d1_norm"]), "d0_d1_dot": _f(r["d0_d1_dot"])})
    conds = sorted(fit_by_cond, key=lambda c: (int(c[0]), int(c[1]),
                                               float(c[2])))

    sol_by_cond = {}
    for r in sol_rows:
        sol_by_cond.setdefault((r["seed"], r["start"], r["k"], r["r"]),
                               {})[r["solution"]] = (
            _f(r["x"]), _f(r["y"]), _f(r["m"]), r["flags"])

    audit_by = defaultdict(dict)   # cond -> (probe -> {(x,y): delta})
    for r in audit_rows:
        key = (r["seed"], r["start"], r["k"])
        audit_by[key].setdefault(int(r["probe"]), {})[
            _pct_key(r["x"], r["y"])] = _f(r["delta_v"])

    check_by = defaultdict(dict)   # cond -> (probe -> {label: mean})
    for r in check_rows:
        key = (r["seed"], r["start"], r["k"])
        check_by[key].setdefault(int(r["probe"]), {})[r["label"]] = _f(r["mean"])

    # ---- bootstrap（FIT：k1/k2 配对，同 state 共用索引）------------------
    rng = np.random.default_rng(BOOT_SEED)
    boot = {}
    for cond in conds:
        probes = fit_by_cond[cond]
        n = len(probes)
        draws = {}
        for rep in range(BOOT_N):
            idx = rng.integers(0, n, n)
            sub = [probes[i] for i in idx]
            agg = r1m.mean_k_operators(sub)
            sym = r1m.symmetrized_K(agg["k12_direct"], agg["k21_direct"],
                                    agg["k11"], agg["k22"])
            b = [agg["b1"], agg["b2"]]
            for q in ("b1", "b2", "k11", "k12_direct", "k21_direct", "k22"):
                draws.setdefault(q, []).append(agg[q])
            if sym["ok"]:
                for rr in (R_PRIMARY, R_HALF):
                    lr = r1m.solve_lr_only(b, sym["K"], rr)
                    s2 = r1m.solve_rectangle(b, sym["K"], rr, rr)
                    draws.setdefault(f"x_LR{rr:g}", []).append(lr["x"])
                    draws.setdefault(f"x2D{rr:g}", []).append(s2["x"])
                    draws.setdefault(f"y2D{rr:g}", []).append(s2["y"])
                    draws.setdefault(f"gain_pred{rr:g}", []).append(
                        lr["m"] - s2["m"])
                sch = r1m.schur_elimination(b, sym["K"])
                if sch["valid"]:
                    draws.setdefault("b_eff", []).append(sch["b_eff"])
                    draws.setdefault("K_eff", []).append(sch["K_eff"])
                    draws.setdefault("y_star", []).append(sch["y_star"])
        boot[cond] = draws

    # ---- 模型误差表 ------------------------------------------------------
    err_rows = []
    err_summary = {}
    for cond in conds:
        agg = r1m.mean_k_operators(fit_by_cond[cond])
        sym = r1m.symmetrized_K(agg["k12_direct"], agg["k21_direct"],
                                agg["k11"], agg["k22"])
        b = [agg["b1"], agg["b2"]]
        probes_a = audit_by[cond]
        plist = sorted(probes_a)
        coords = sorted({c for p in plist for c in probes_a[p]})
        sse = 0.0
        jn = []
        for (x, y) in coords:
            vals = [probes_a[p][(x, y)] for p in plist
                    if (x, y) in probes_a[p]]
            if not vals:
                continue
            jm = float(np.mean(vals))
            se = r1m.paired_se(vals)
            jh = r1m.quad_value((x, y), b, sym["K"])
            err_rows.append((cond[0], cond[1], cond[2], f"{x:.9g},{y:.9g}",
                             x, y, jh, jm, se, abs(jh - jm)))
            jn.append(jm)
        # SSE/TSS：固定网格 = 非 native、非解标签的网格坐标
        # （解坐标不在“固定网格”内；native=0 恒等排除）
        sol_pts = set()
        for rr in (R_PRIMARY, R_HALF):
            for sname in ("LR", "2D"):
                if sname in sol_by_cond.get(cond + (f"{rr:g}",), {}):
                    sol_pts.add(_pct_key(
                        sol_by_cond[cond + (f"{rr:g}",)][sname][0],
                        sol_by_cond[cond + (f"{rr:g}",)][sname][1]))
        grid_vals = []
        for (x, y) in coords:
            if (x, y) == (0.0, 0.0) or (x, y) in sol_pts:
                continue
            # 两半径网格点：|x|,|y| ∈ {r, r/2, 0}
            if _is_grid_point(x, y):
                vals = [probes_a[p][(x, y)] for p in plist]
                grid_vals.append((r1m.quad_value((x, y), b, sym["K"]),
                                  float(np.mean(vals))))
        sse = sum((jh - jm) ** 2 for jh, jm in grid_vals)
        jm_all = [jm for _, jm in grid_vals]
        tss = sum((v - np.mean(jm_all)) ** 2 for v in jm_all) if jm_all else 0.0
        maxabs = max((abs(jh - jm) for jh, jm in grid_vals), default=0.0)
        rmse = math.sqrt(sse / len(grid_vals)) if grid_vals else float("nan")
        err_summary[cond] = {
            "sse": sse, "tss": tss, "maxabs_grid": maxabs, "rmse": rmse,
            "n_grid": len(grid_vals),
            "max_abs_J": max((abs(v) for v in jm_all), default=0.0),
            "low_signal": max((abs(v) for v in jm_all), default=0.0)
            < LOW_SIGNAL_ABS}

    with (OUT / "audit_model_errors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "coord", "x", "y", "J_hat",
                    "J_audit_mean", "J_audit_paired_se", "abs_error"])
        for r in err_rows:
            w.writerow(r)

    # ---- 结构增益（AUDIT 配对 bootstrap）--------------------------------
    gain_rows = []
    verdicts = {}
    for cond in conds:
        probes_a = audit_by[cond]
        plist = sorted(probes_a)
        agg = r1m.mean_k_operators(fit_by_cond[cond])
        sym = r1m.symmetrized_K(agg["k12_direct"], agg["k21_direct"],
                                agg["k11"], agg["k22"])
        b = [agg["b1"], agg["b2"]]
        mrow = next(m for m in model_rows
                    if (m["seed"], m["start"], m["k"]) == cond)
        cos_mean = _f(mrow["cos_mean"])
        collinear = cos_mean >= COLLINEAR_COS
        # check256（primary）
        chk = check_by.get(cond, {})
        chk_plist = sorted(chk)
        v = {}
        for rr in (R_PRIMARY, R_HALF):
            sLR = sol_by_cond[cond + (f"{rr:g}",)]["LR"]
            s2D = sol_by_cond[cond + (f"{rr:g}",)]["2D"]
            pLR = _pct_key(sLR[0], sLR[1])
            p2D = _pct_key(s2D[0], s2D[1])
            gl = [probes_a[p][pLR] - probes_a[p][p2D] for p in plist
                  if pLR in probes_a[p] and p2D in probes_a[p]]
            gm_ = float(np.mean(gl)) if gl else float("nan")
            gse = r1m.paired_se(gl) if gl else float("nan")
            # bootstrap CI（probe 重抽）
            bs = []
            arr = np.asarray(gl, dtype=np.float64)
            for rep in range(BOOT_N):
                idx = rng.integers(0, len(arr), len(arr))
                bs.append(float(arr[idx].mean()))
            lo, hi = _ci(bs)
            dLR = [probes_a[p][pLR] for p in plist if pLR in probes_a[p]]
            d2D = [probes_a[p][p2D] for p in plist if p2D in probes_a[p]]
            gain_pred = sLR[2] - s2D[2]
            eta_mult = 1.0 + s2D[0]
            alpha2 = s2D[1] / eta_mult
            y_star = _f(mrow["y_star"]) if mrow["y_star"] != "NA" else float("nan")
            no_readout = (s2D[1] == 0.0) or (p2D == pLR) or \
                (math.isfinite(y_star) and y_star == 0.0)
            v[rr] = dict(gain_pred=gain_pred, gain_audit=gm_, gain_se=gse,
                         ci=(lo, hi), x_LR=sLR[0], y_LR=0.0, x_2D=s2D[0],
                         y_2D=s2D[1], dLR=float(np.mean(dLR)) if dLR else float("nan"),
                         d2D=float(np.mean(d2D)) if d2D else float("nan"),
                         eta_mult=eta_mult, alpha=alpha2, no_readout=no_readout)
        # check256 迁移（primary）
        gchk = None
        if chk:
            gl_c = [chk[p]["aLR"] - chk[p]["a2D"] for p in chk_plist
                    if "aLR" in chk[p] and "a2D" in chk[p]]
            gchk = float(np.mean(gl_c)) if gl_c else float("nan")
        verdict = decide(v[R_PRIMARY], gchk, collinear,
                         err_summary[cond]["low_signal"], mrow)
        verdicts[cond] = verdict
        for rr in (R_PRIMARY, R_HALF):
            d = v[rr]
            gain_rows.append((cond[0], cond[1], cond[2], f"{rr:g}",
                              d["gain_pred"], d["gain_audit"], d["gain_se"],
                              d["ci"][0], d["ci"][1],
                              d["x_LR"], d["x_2D"], d["y_2D"],
                              d["eta_mult"], d["alpha"],
                              d["dLR"], d["d2D"],
                              gchk if rr == R_PRIMARY else "",
                              verdict["code"],
                              "COLLINEAR" if collinear else ""))
    with (OUT / "incremental_gain.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "r", "gain_pred", "gain_audit_mean",
                    "gain_audit_se", "gain_ci_lo", "gain_ci_hi", "x_LR",
                    "x_2D", "y_2D", "eta_mult_2D", "alpha_2D",
                    "delta_aLR_vs_native", "delta_a2D_vs_native",
                    "gain_check256_primary", "verdict", "collinear_flag"])
        for r in gain_rows:
            w.writerow(r)

    # ---- scope / rejections ---------------------------------------------
    with (OUT / "model_scope_and_rejections.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "r", "decision_code", "details"])
        for cond in conds:
            for rr in (R_PRIMARY, R_HALF):
                w.writerow((cond[0], cond[1], cond[2], f"{rr:g}",
                            verdicts[cond]["code"] if rr == R_PRIMARY
                            else "HALF_RADIUS_DIAGNOSTIC_ONLY",
                            verdicts[cond]["detail"] if rr == R_PRIMARY
                            else err_summary[cond]["rmse"]))

    # ---- costs.json -------------------------------------------------------
    costs = {"plan": r1m.plan_budget(), "caps": config["budget"]["caps"],
             "ledger_final": _last_ledger(),
             "n_conditions": len(conds),
             "wall_note": "OBSERVATIONAL_SHARED_GPU only"}
    (OUT / "costs.json").write_text(json.dumps(costs, indent=1) + "\n")

    # ---- figures ----------------------------------------------------------
    try:
        _figures(conds, model_rows, boot, gain_rows, err_rows)
    except Exception as exc:  # noqa
        (OUT / "figures_note.txt").write_text(f"figure generation failed: "
                                              f"{exc!r}\n")

    # ---- REPORT / KEY_DATA ------------------------------------------------
    report(conds, fit_by_cond, model_rows, sol_by_cond, boot, err_summary,
           gain_rows, verdicts, check_by, costs)
    print("ANALYZE_DONE")


def _is_grid_point(x, y):
    for r in (R_PRIMARY, R_HALF):
        if abs(x) <= r + 1e-12 and -1e-12 <= y <= r + 1e-12:
            xs = {round(-r, 9), 0.0, round(r, 9)}
            ys = {0.0, round(r / 2, 9), round(r, 9)}
            if round(x, 9) in xs and round(y, 9) in ys:
                return True
    return False


def _last_ledger():
    p = OUT / "resource_log.jsonl"
    last = None
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last


def decide(vp, gchk, collinear, low_signal, mrow):
    detail = (f"gain_pred={vp['gain_pred']:.3e};"
              f"gain_audit={vp['gain_audit']:.3e} "
              f"[{vp['ci'][0]:.3e},{vp['ci'][1]:.3e}]")
    if collinear:
        return {"code": "STRUCTURE_LR_INDISTINGUISHABLE", "detail": detail}
    if vp["no_readout"]:
        return {"code": "MODEL_RECOMMENDS_NO_READOUT_CHANGE", "detail": detail}
    if not math.isfinite(vp["gain_audit"]):
        return {"code": "UNRESOLVED_AUDIT_MISSING", "detail": detail}
    if vp["gain_pred"] > 0:
        if vp["ci"][1] <= 0 or vp["gain_audit"] <= 0:
            return {"code": "PREDICTED_BENEFIT_NOT_CONFIRMED", "detail": detail}
        if vp["ci"][0] > 0:
            if gchk is not None and math.isfinite(gchk) and gchk < 0:
                return {"code": "TARGET_TRANSFER_NOT_ESTABLISHED",
                        "detail": detail + f";gain_check256={gchk:.3e}"}
            return {"code": "SUPPORTS_FURTHER_VALIDATION", "detail": detail}
        return {"code": "UNRESOLVED_AUDIT_NOISE", "detail": detail}
    return {"code": "MODEL_RECOMMENDS_NO_READOUT_CHANGE", "detail": detail}


def _figures(conds, model_rows, boot, gain_rows, err_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    labs = [f"s{c[0]}/t{c[1]}/k{c[2]}" for c in conds]

    # fig1: b1/b2 与 K 特征值 CI
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(conds))
    for q, off, color in (("b1", -0.18, "tab:blue"), ("b2", 0.18, "tab:red")):
        mu = [float(next(m for m in model_rows
                         if (m["seed"], m["start"], m["k"]) == c)[q + "_mean"])
              for c in conds]
        lo = [_ci(boot[c][q])[0] for c in conds]
        hi = [_ci(boot[c][q])[1] for c in conds]
        ax.errorbar(xs + off, mu, yerr=[np.array(mu) - np.array(lo),
                                        np.array(hi) - np.array(mu)],
                    fmt="o", label=q, color=color, ms=4, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("FIT coefficients b1/b2 with bootstrap 95% CI (per condition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig1_coefficients.png", dpi=140)
    plt.close(fig)

    # fig2: 模型 vs 真实目标面
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    jh = [r[6] for r in err_rows]
    jm = [r[7] for r in err_rows]
    ax.scatter(jm, jh, s=6, alpha=0.5)
    lim = [min(jm + jh), max(jm + jh)]
    ax.plot(lim, lim, "r--", lw=0.8)
    ax.set_xlabel("J_audit (mean over AUDIT probes)")
    ax.set_ylabel("J_hat (quadratic model)")
    ax.set_title("Model vs real objective offsets")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_model_vs_audit.png", dpi=140)
    plt.close(fig)

    # fig3: gain_pred vs gain_audit（primary）
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    gp = [r[4] for r in gain_rows if r[3] == "0.05"]
    ga = [r[5] for r in gain_rows if r[3] == "0.05"]
    gerr = [[r[5] - r[7] for r in gain_rows if r[3] == "0.05"],
            [r[8] - r[5] for r in gain_rows if r[3] == "0.05"]]
    ax.errorbar(gp, ga, yerr=gerr, fmt="o", ms=4, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("gain_pred = m(a_LR)-m(a_2D)")
    ax.set_ylabel("gain_audit = V(a_LR)-V(a_2D)")
    ax.set_title("Structure gain, primary radius")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_gain_pred_vs_audit.png", dpi=140)
    plt.close(fig)

    # fig4: primary vs half gain_audit + y_2D
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    ga_p = {r[0] + r[1] + r[2]: r[5] for r in gain_rows if r[3] == "0.05"}
    ga_h = {r[0] + r[1] + r[2]: r[5] for r in gain_rows if r[3] == "0.025"}
    keys = sorted(ga_p)
    axes[0].scatter([ga_p[k_] for k_ in keys], [ga_h[k_] for k_ in keys])
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel("gain_audit primary r=.05")
    axes[0].set_ylabel("gain_audit half r=.025")
    y2d = {r[0] + r[1] + r[2]: r[11] for r in gain_rows if r[3] == "0.05"}
    axes[1].bar(range(len(keys)), [y2d[k_] for k_ in keys])
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(keys, rotation=60, fontsize=6)
    axes[1].set_ylabel("y_2D (alpha-ish readout)")
    axes[1].set_title("Optimal readout coordinate y*")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_radius_and_y2d.png", dpi=140)
    plt.close(fig)


def report(conds, fit_by_cond, model_rows, sol_by_cond, boot, err_summary,
           gain_rows, verdicts, check_by, costs):
    lines = []
    add = lines.append
    add("# GTTR-R1 正式报告（R0/R1；未启动 R2/R3）\n")
    add("协议：FR_GTTR_R1_Model_To_Rule_Identification_v1.md。"
        "本轮为离线辨识：0 committed updates，0 V_final 调用。\n")
    add("## A. 条件覆盖\n")
    add("|cond|state|k|eta_applied|FIT|AUDIT|dtype/backend|")
    add("|---|---|---|---|---|---|---|")
    cfg = json.loads((OUT / "config.json").read_text())
    for cond in conds:
        eta = cfg["schedule"]["eta_by_cell"][f"{cond[0]}/{cond[1]}"][
            f"eta_k{float(cond[2]):g}"]
        add(f"|{r1m.condition_label(*[int(cond[0]), int(cond[1])], float(cond[2]))}"
            f"|{cond[0]}/{cond[1]}|{cond[2]}|{eta:.6e}|8|8|"
            f"ctrl=FP32/MATH-SDPA; train=BF16; check=BF16+default|")
    add("\n## B. FIT 系数（均值，bootstrap 95% CI；K 特征值/条件数）\n")
    add("|cond|b1 [CI]|b2 [CI]|K11|K12(dir avg)|K22|eig_min|eig_max|cond|"
        "b_eff|K_eff|y_star|cos(d0,d1)|")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in conds:
        m = next(x for x in model_rows
                 if (x["seed"], x["start"], x["k"]) == cond)
        def cq(q):
            lo, hi = _ci(boot[cond][q])
            return f"{_f(m[q + '_mean']):.3e} [{lo:.2e},{hi:.2e}]"
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{cq('b1')}|{cq('b2')}"
            f"|{_f(m['k11_mean']):.3e}|"
            f"{0.5*(_f(m['k12_direct_mean'])+_f(m['k21_direct_mean'])):.3e}"
            f"|{_f(m['k22_mean']):.3e}|{_f(m['eig_min']):.3e}"
            f"|{_f(m['eig_max']):.3e}|{_f(m['cond_K']):.3g}"
            f"|{m['b_eff']}|{m['K_eff']}|{m['y_star']}|{_f(m['cos_mean']):.4f}|")
    add("\n## C. 受限解（两半径）\n")
    add("|cond|r|x_LR|x_2D|y_2D|eta_mult|alpha_new|gain_pred|边界/KKT|")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in gain_rows:
        sL = sol_by_cond[(row[0], row[1], row[2], row[3])]["LR"]
        add(f"|{row[0]}/{row[1]}/k{row[2]}|{row[3]}|{row[9]:.4f}|{row[10]:.4f}"
            f"|{row[11]:.4f}|{row[12]:.4f}|{row[13]:.4f}|{row[4]:.3e}"
            f"|{sL[3]}|")
    add("\n## D. AUDIT 真实目标差与模型误差\n")
    add("|cond|r|delta(a_LR)|delta(a_2D)|gain_audit [CI]|gain_pred|"
        "grid maxabs|grid RMSE|SSE/TSS|LOW_SIGNAL|")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in gain_rows:
        cond = (row[0], row[1], row[2])
        e = err_summary[cond]
        add(f"|{row[0]}/{row[1]}/k{row[2]}|{row[3]}|{row[14]:.3e}"
            f"|{row[15]:.3e}|{row[5]:.3e} [{row[7]:.2e},{row[8]:.2e}]"
            f"|{row[4]:.3e}|{e['maxabs_grid']:.2e}|{e['rmse']:.2e}"
            f"|{e['sse']:.3e}/{e['tss']:.3e}|{e['low_signal']}|")
    add("\n## E. V_check256 目标迁移（primary；独立目标，不与 ctrl 混算）\n")
    add("|cond|delta_check(a_LR)|delta_check(a_2D)|gain_check256|")
    add("|---|---|---|---|")
    for cond in conds:
        chk = check_by.get(cond, {})
        if not chk:
            add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|NA|NA|NA|")
            continue
        pl = sorted(chk)
        dLRc = [chk[p]["aLR"] - chk[p]["native"] for p in pl
                if "aLR" in chk[p]]
        d2Dc = [chk[p]["a2D"] - chk[p]["native"] for p in pl
                if "a2D" in chk[p]]
        gc = [chk[p]["aLR"] - chk[p]["a2D"] for p in pl
              if "aLR" in chk[p] and "a2D" in chk[p]]
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|"
            f"{np.mean(dLRc) if dLRc else float('nan'):.3e}|"
            f"{np.mean(d2Dc) if d2Dc else float('nan'):.3e}|"
            f"{np.mean(gc) if gc else float('nan'):.3e}|")
    add("\n## F. 判定（按条件；primary）\n")
    add("|cond|verdict|detail|")
    add("|---|---|---|")
    for cond in conds:
        v = verdicts[cond]
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{v['code']}|{v['detail']}|")
    strat = r1m.stratify([r1m.condition_label(int(c[0]), int(c[1]),
                                               float(c[2])) for c in conds])
    add(f"\n分层：{len(strat['seed'])} seeds × {len(strat['start'])} starts "
        f"× {len(strat['k'])} k = {len(conds)} 条件（非独立 seeds）。\n")
    add("## G. 成本\n")
    led = costs["ledger_final"] or {}
    add(f"- total forward（实际/上限）：{led.get('total_fw')}/"
        f"{costs['caps']['total_fw']}")
    add(f"- backward/VJP microcall：{led.get('total_bw')}/"
        f"{costs['caps']['total_bw_vjp_micro']}")
    add(f"- 目标 HVP 方向：{led.get('hvp_dirs')}/{costs['caps']['hvp_dirs']}")
    add(f"- committed optimizer updates：0；V_final 调用：0")
    add(f"- failures={led.get('failures', 0)} retries={led.get('retries', 0)}"
        f"（详见 resource_log.jsonl）")
    add("\n## 不能从本轮推出的结论\n")
    add("- 不声称 mu/历史偏差/噪声已分别辨识；不声称 tokens-to-target 改善；")
    add("- 不把 K 特征值归因到具体层；不因局部增益宣称 QH/grafting 首创；")
    add("- 不给旧 NORM/v3 追加成功结论；y=0/噪声大/成本高均为有效结果。\n")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")

    # KEY_DATA_FOR_REVIEW：机器可读关键数
    key = {"conditions": []}
    for cond in conds:
        m = next(x for x in model_rows
                 if (x["seed"], x["start"], x["k"]) == cond)
        grow = [r for r in gain_rows
                if (r[0], r[1], r[2]) == cond and r[3] == "0.05"]
        g5 = grow[0] if grow else None
        key["conditions"].append({
            "cond": list(cond),
            "b1": _f(m["b1_mean"]), "b2": _f(m["b2_mean"]),
            "k11": _f(m["k11_mean"]), "k22": _f(m["k22_mean"]),
            "k12": 0.5 * (_f(m["k12_direct_mean"]) + _f(m["k21_direct_mean"])),
            "eig": [_f(m["eig_min"]), _f(m["eig_max"])],
            "cond_K": _f(m["cond_K"]), "cos_d0d1": _f(m["cos_mean"]),
            "b_eff": m["b_eff"], "K_eff": m["K_eff"], "y_star": m["y_star"],
            "gain_pred_primary": g5[4] if g5 else None,
            "gain_audit_primary": g5[5] if g5 else None,
            "gain_audit_ci": [g5[7], g5[8]] if g5 else None,
            "y_2D_primary": g5[11] if g5 else None,
            "verdict": verdicts[cond]["code"],
            "model_rmse_grid": err_summary[cond]["rmse"],
        })
    (OUT / "KEY_DATA_FOR_REVIEW.json").write_text(json.dumps(key, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
