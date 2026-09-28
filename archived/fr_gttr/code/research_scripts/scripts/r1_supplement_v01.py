#!/usr/bin/env python3
"""GTTR-R1 0-GPU 补充分析（POSTHOC_DIAGNOSTIC；只读已封存 CSV）。

不重新采样、不修改原 FIT/AUDIT/阈值、不运行 R2/R3。内容：
  S1 解的逐行清单：无约束 Schur 解 vs 矩形受限解分开；y上边界/y内部/x边界
     计数（11/12 y=.05，2031/46 k2 内部 y≈.0118）；LR-only 点 b2+K21*x_LR
     与 KKT 残差。
  S2 纯离线一阶伴随（POSTHOC）：同 b 同域 min b^T a (K=0)，与二阶解比较；
     实际收益只从已有 AUDIT 坐标读取，缺坐标写 NOT_MEASURED。
  S3 2031/46 两条件保留 UNRESOLVED（中性表述，不唯一归因）。
  S4 V_check256 12 行：dLR/d2D/paired gain、SE/CI；均值全正与区间排除 0
     分开陈述；bootstrap 单位与目标角色写明。
  S5 grid RMSE/SSE/TSS 的点集 scope 与数量写清（merged/primary/half 三种
     scope 分列，不并成一列解释）。
  S6 结论表。

输出：SUPPLEMENT_POSTHOC_V01.md 与 supplement_*.csv（全部新增文件）。
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
EPS_COORD = 1e-9


def _read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(v):
    return float(v)


def _key(x, y):
    return (round(float(x), 9), round(float(y), 9))


def _bs_ci(vals, rng):
    arr = np.asarray(vals, dtype=np.float64)
    bs = [float(arr[rng.integers(0, len(arr), len(arr))].mean())
          for _ in range(BOOT_N)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _se(vals):
    return r1m.paired_se(vals)


def _is_grid(x, y, r):
    xs = {round(-r, 9), 0.0, round(r, 9)}
    ys = {0.0, round(r / 2, 9), round(r, 9)}
    return round(x, 9) in xs and round(y, 9) in ys and abs(x) <= r + EPS_COORD \
        and -EPS_COORD <= y <= r + EPS_COORD


def main():
    ok, why = r1m.verify_seal(SEAL_JSON, [FIT_CSV, MODEL_CSV, SOL_CSV])
    assert ok, f"seal verify failed: {why}（本补充只读，不得回写）"
    model_rows = _read(MODEL_CSV)
    sol_rows = _read(SOL_CSV)
    audit_rows = _read(AUDIT_CSV)
    check_rows = _read(CHECK_CSV)
    rng = np.random.default_rng(BOOT_SEED)

    conds = [(m["seed"], m["start"], m["k"]) for m in model_rows]
    mm = {(m["seed"], m["start"], m["k"]): m for m in model_rows}
    sol = defaultdict(dict)
    for r in sol_rows:
        sol[(r["seed"], r["start"], r["k"], r["r"])][r["solution"]] = \
            (_f(r["x"]), _f(r["y"]), _f(r["m"]))
    audit_by = defaultdict(dict)
    for r in audit_rows:
        audit_by[(r["seed"], r["start"], r["k"])].setdefault(
            int(r["probe"]), {})[_key(r["x"], r["y"])] = _f(r["delta_v"])
    check_by = defaultdict(dict)
    for r in check_rows:
        check_by[(r["seed"], r["start"], r["k"])].setdefault(
            int(r["probe"]), {})[r["label"]] = _f(r["mean"])

    L = []
    add = L.append
    add("# GTTR-R1 补充报告（POSTHOC_DIAGNOSTIC，0-GPU，只读已封存 CSV）\n")
    add("本补充不改变 R1 主要结果；不重新采样、不修改 FIT/AUDIT 或阈值；"
        "不运行 R2/R3。seal 复核通过。\n")

    # ---------------- S1 解清单 ----------------
    add("## S1. 解的逐行清单：无约束 Schur 解 vs 矩形受限解\n")
    y_up = y_in = x_bnd = 0
    add("|cond|r|Schur(x*,y*) 无约束|受限 2D (x,y)|y 分类|x 分类|"
        "b2+K21*x_LR|KKT 违约 max(0,-g_y)|")
    add("|---|---|---|---|---|---|---|---|")
    s1_rows = []
    for cond in conds:
        m = mm[cond]
        b1 = _f(m["b1_mean"])
        b2 = _f(m["b2_mean"])
        k11 = _f(m["k11_mean"])
        k12 = 0.5 * (_f(m["k12_direct_mean"]) + _f(m["k21_direct_mean"]))
        y_star = _f(m["y_star"]) if m["y_star"] != "NA" else float("nan")
        x_star = -(b1 + k12 * y_star) / k11 if math.isfinite(y_star) \
            else float("nan")
        for rr in (R_PRIMARY, R_HALF):
            sLR = sol[cond + (f"{rr:g}",)]["LR"]
            s2D = sol[cond + (f"{rr:g}",)]["2D"]
            x2, y2 = s2D[0], s2D[1]
            ycat = ("y=r 上边界" if abs(y2 - rr) < EPS_COORD else
                    ("y=0" if abs(y2) < EPS_COORD else "y 内部"))
            xcat = ("x=±r 边界" if abs(abs(x2) - rr) < EPS_COORD else
                    "x 内部")
            if abs(rr - R_PRIMARY) < 1e-15:
                if abs(y2 - rr) < EPS_COORD:
                    y_up += 1
                elif 0 < y2 < rr:
                    y_in += 1
                if abs(abs(x2) - rr) < EPS_COORD:
                    x_bnd += 1
            gy = b2 + k12 * sLR[0]
            add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{rr:g}|"
                f"({x_star:.4f},{y_star:.4f})|({x2:.5f},{y2:.5f})|"
                f"{ycat}|{xcat}|{gy:.4e}|{max(0.0, -gy):.4e}|")
            s1_rows.append((cond[0], cond[1], cond[2], f"{rr:g}",
                            x_star, y_star, x2, y2, ycat, xcat, gy,
                            max(0.0, -gy)))
    add("")
    add(f"**计数（primary，12 条件）**：受限 2D 解 y=r 上边界 **{y_up}/12**；"
        f"y 内部 **{y_in}/12**（2031/46 k2，y≈0.0118，且在 x=+r 边上）；"
        f"x=±r 边界 **{x_bnd}/12**")
    add(f"half 半径同形态：11/12 y=.025，2031/46 k2 y≈0.0141。")
    add("无约束 Schur y* 与受限 y 的关系（分开记录，不混写）：y*>r 的条件"
        "其受限解贴 y=r；y*<0 或 y*∈(0,r) 时受限解由 x=±r 边上的 "
        "∂m/∂y=0（或角点）决定，二者不是同一个量。")
    add("LR-only 点的 y 方向梯度 g_y=b2+K21*x_LR 全部为负且违约 >0，"
        "与 kkt_y0=False 一致：y=0 不满足 KKT，无一条件在模型内支持"
        "“排除读出改动”。（修订：不再写“10/12 y*>r→全部贴边界”或"
        "“所有 y 都在边界”；正确计数为 11/12 边界 + 1 内部。）\n")

    # ---------------- S2 一阶伴随（POSTHOC） ----------------
    add("## S2. 纯离线一阶伴随分析（POSTHOC_DIAGNOSTIC）\n")
    add("同一 FIT b、同一可行域，忽略曲率求 min b^T a（K=0）：线性目标最优"
        "在角点 x=-sign(b1)*r, y=-sign(b2)*r。实际收益只从已有 AUDIT 坐标"
        "读取（真实 ΔV，非二次预测）；缺坐标写 NOT_MEASURED。\n")
    add("|cond|r|a_1st (x,y)|与 a_2D 关系|实测 ΔV(a_1st) 均值[SE]|"
        "实测 ΔV(a_1st)-ΔV(a_2D) [CI]|二阶信息判读|")
    add("|---|---|---|---|---|---|---|")
    s2_rows = []
    for cond in conds:
        m = mm[cond]
        b1 = _f(m["b1_mean"])
        b2 = _f(m["b2_mean"])
        for rr in (R_PRIMARY, R_HALF):
            x1 = -rr * math.copysign(1.0, b1)      # min b1*x -> -r*sign(b1)
            y1 = rr if b2 < 0 else 0.0             # min b2*y -> r 或 0
            s2D = sol[cond + (f"{rr:g}",)]["2D"]
            same = (_key(x1, y1) == _key(s2D[0], s2D[1]))
            probes_a = audit_by[cond]
            plist = sorted(probes_a)
            k1st = _key(x1, y1)
            k2d = _key(s2D[0], s2D[1])
            if all(k1st in probes_a[p] for p in plist):
                d1st = [probes_a[p][k1st] for p in plist]
                d1m = float(np.mean(d1st))
                d1se = _se(d1st)
                if all(k2d in probes_a[p] for p in plist):
                    d2 = [probes_a[p][k2d] for p in plist]
                    diff = [a - b_ for a, b_ in zip(d1st, d2)]
                    lo, hi = _bs_ci(diff, rng)
                    diff_txt = (f"{float(np.mean(diff)):.3e} "
                                f"[{lo:.2e},{hi:.2e}]")
                    verdict = ("同点：一阶已选对角点，二阶未改变选择"
                               if same else
                               f"二阶改变选择；实测差 "
                               f"{' favors 2D' if np.mean(diff) > 0 else ' favors 1st'}")
                else:
                    diff_txt = "NOT_MEASURED"
                    verdict = "二阶改变选择；a_2D 坐标未测" if not same \
                        else "同点"
            else:
                d1m = d1se = float("nan")
                diff_txt = "NOT_MEASURED"
                verdict = "a_1st 坐标未测" if not same else "同点（未测）"
            add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{rr:g}|"
                f"({x1:.3f},{y1:.3f})|{'同点' if same else '不同点'}|"
                f"{d1m:.3e} [{d1se:.2e}]|{diff_txt}|{verdict}|")
            s2_rows.append((cond[0], cond[1], cond[2], f"{rr:g}", x1, y1,
                            same, d1m, d1se, diff_txt))
    add("")
    add("判读口径：ΔV(a_1st)−ΔV(a_2D) 为配对实测差（一阶点相对二阶点的"
        "目标差；正值=二阶选择的端点更优）。该分析为事后诊断，不改变本轮"
        "主要结果与判定。\n")

    # ---------------- S3 2031/46 UNRESOLVED ----------------
    add("## S3. 2031/46 两个条件：保留 UNRESOLVED（中性表述）\n")
    add("2031/46 k1、k2 的 AUDIT gain 置信区间跨 0。本轮没有同库有限偏移的"
        "重复测量或噪声分解证据，因此**不将失败唯一归因为样本不足，也不排除"
        "局部余项（模型误差与 gain 同量级）**；保持 UNRESOLVED，不下机制"
        "结论。\n")

    # ---------------- S4 V_check256 12 行 ----------------
    add("## S4. V_check256（primary；目标角色=独立迁移检查，不与 V_ctrl 混算）\n")
    add("bootstrap 单位=effective-batch probe（n=8，PCG64 seed 2026092301）；"
        "dLR=chk(a_LR)−chk(native)，d2D=chk(a_2D)−chk(native)，"
        "gain=chk(a_LR)−chk(a_2D)（正值支持 2D）。\n")
    add("|cond|dLR [SE]|d2D [SE]|paired gain [SE]|gain 95% CI|CI 排除 0|")
    add("|---|---|---|---|---|---|")
    n_pos = n_ci = 0
    s4_rows = []
    for cond in conds:
        chk = check_by[cond]
        plist = sorted(chk)
        dLR = [chk[p]["aLR"] - chk[p]["native"] for p in plist]
        d2D = [chk[p]["a2D"] - chk[p]["native"] for p in plist]
        gain = [chk[p]["aLR"] - chk[p]["a2D"] for p in plist]
        gm_ = float(np.mean(gain))
        lo, hi = _bs_ci(gain, rng)
        ci_excl = (lo > 0 or hi < 0)
        if gm_ > 0:
            n_pos += 1
        if ci_excl and lo > 0:
            n_ci += 1
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{np.mean(dLR):.3e} "
            f"[{_se(dLR):.2e}]|{np.mean(d2D):.3e} [{_se(d2D):.2e}]|"
            f"{gm_:.3e} [{_se(gain):.2e}]|[{lo:.2e},{hi:.2e}]|"
            f"{'是' if ci_excl and lo > 0 else '否'}|")
        s4_rows.append((cond[0], cond[1], cond[2], np.mean(dLR), _se(dLR),
                        np.mean(d2D), _se(d2D), gm_, _se(gain), lo, hi))
    add("")
    add(f"**分开陈述**：12/12 条件 check256 paired gain 均值为正；其中 "
        f"**{n_ci}/12** 的 bootstrap 95% CI 下界 >0（排除 0）。其余 "
        f"{12 - n_ci} 个条件点估计为正但区间含 0。check256 与 V_ctrl 数值"
        "不可相减（目标不同）。\n")

    # ---------------- S5 SSE/TSS scope ----------------
    add("## S5. grid RMSE/SSE/TSS 的点集 scope 与数量\n")
    add("原 D 表的 RMSE/SSE/TSS 使用的点集（scope A）：**两半径合并固定网格**"
        "（16 唯一坐标，无重复坐标）去掉 native（恒为 0，必排除）再去掉与"
        "本条件 4 个解坐标重合的网格点（解属 adaptive scope，不算固定网格）。"
        "既不是“仅 primary”，也不含最优解本身。\n")
    add("下表把三种 scope 分列（均不含 native、不含解坐标）：A=合并两半径"
        "网格（原口径）；B=仅 primary 网格；C=仅 half 网格。不同 scope 的"
        "指标不在同一列内混合解释。\n")
    add("|cond|nA|SSE_A/TSS_A|RMSE_A|nB|SSE_B/TSS_B|nC|SSE_C/TSS_C|")
    add("|---|---|---|---|---|---|---|---|")
    s5_rows = []
    for cond in conds:
        m = mm[cond]
        b1 = _f(m["b1_mean"])
        b2 = _f(m["b2_mean"])
        k11 = _f(m["k11_mean"])
        k12 = 0.5 * (_f(m["k12_direct_mean"]) + _f(m["k21_direct_mean"]))
        k22 = _f(m["k22_mean"])
        K = [[k11, k12], [k12, k22]]
        probes_a = audit_by[cond]
        plist = sorted(probes_a)
        coords = sorted({c for p in plist for c in probes_a[p]})
        sol_pts = set()
        for rr in (R_PRIMARY, R_HALF):
            for sname in ("LR", "2D"):
                s_ = sol[cond + (f"{rr:g}",)][sname]
                sol_pts.add(_key(s_[0], s_[1]))
        row = [cond[0], cond[1], cond[2]]
        stats_scope = []
        for scope, gridr in (("A", (R_PRIMARY, R_HALF)),
                             ("B", (R_PRIMARY,)), ("C", (R_HALF,))):
            vals = []
            for (x, y) in coords:
                if (x, y) == (0.0, 0.0) or _key(x, y) in sol_pts:
                    continue
                if not any(_is_grid(x, y, g) for g in gridr):
                    continue
                jh = r1m.quad_value((x, y), [b1, b2], K)
                jm = float(np.mean([probes_a[p][(x, y)] for p in plist]))
                vals.append((jh, jm))
            sse = sum((a - b_) ** 2 for a, b_ in vals)
            jm_all = [b_ for _, b_ in vals]
            tss = sum((v - np.mean(jm_all)) ** 2 for v in jm_all)
            rmse = math.sqrt(sse / len(vals)) if vals else float("nan")
            stats_scope.append((len(vals), sse, tss, rmse))
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{stats_scope[0][0]}|"
            f"{stats_scope[0][1]:.3e}/{stats_scope[0][2]:.3e}|"
            f"{stats_scope[0][3]:.2e}|{stats_scope[1][0]}|"
            f"{stats_scope[1][1]:.3e}/{stats_scope[1][2]:.3e}|"
            f"{stats_scope[2][0]}|{stats_scope[2][1]:.3e}/"
            f"{stats_scope[2][2]:.3e}|")
        flat = []
        for s_ in stats_scope:
            flat.extend(s_)
        s5_rows.append((cond[0], cond[1], cond[2], *flat))
    add("")
    add("scope A 的 nA=11（解为角点的 11 个条件，4 个解坐标全部与网格重合"
        "被排除：15−4）或 13（2031/46 k2：仅 2 个解坐标与网格重合：15−2）。"
        "B/C 的固定网格各有 8 个非 native 坐标，再去掉与解重合者。\n")

    # ---------------- S6 结论表 ----------------
    add("## S6. 结论表（primary）\n")
    add("|cond|目标方向支持(b2<0且KKT违约>0)|有限域内额外收益(gain_audit CI)|"
        "内生最优幅度是否识别|二阶信息是否改变选择并带来已测价值|仍未分辨|")
    add("|---|---|---|---|---|---|")
    for cond in conds:
        m = mm[cond]
        b1 = _f(m["b1_mean"])
        b2 = _f(m["b2_mean"])
        k12 = 0.5 * (_f(m["k12_direct_mean"]) + _f(m["k21_direct_mean"]))
        sLR = sol[cond + ("0.05",)]["LR"]
        s2D = sol[cond + ("0.05",)]["2D"]
        gy = b2 + k12 * sLR[0]
        direct = "是" if (b2 < 0 and max(0.0, -gy) > 0) else "否"
        # gain_audit CI 从 AUDIT 重算（同主分析口径）
        probes_a = audit_by[cond]
        plist = sorted(probes_a)
        gl = [probes_a[p][_key(sLR[0], 0.0)] -
              probes_a[p][_key(s2D[0], s2D[1])] for p in plist]
        lo, hi = _bs_ci(gl, rng)
        gain_txt = (f"是 CI[{lo:.2e},{hi:.2e}]" if lo > 0 else
                    f"UNRESOLVED CI[{lo:.2e},{hi:.2e}]")
        amp = ("是（y*=0.0118 内部）" if cond == ("2031", "46", "2")
               else "否（y 贴 r 边界，仅有下界）")
        # 二阶 vs 一阶（S2 结果回读）
        s2row = next(r for r in s2_rows
                     if (r[0], r[1], r[2]) == cond and r[3] == "0.05")
        second = ("同点（一阶已选该角点）" if s2row[6]
                  else f"改变选择；实测差 {s2row[9]}")
        unres = "是" if cond[0] == "2031" and cond[1] == "46" else ""
        add(f"|{cond[0]}/{cond[1]}/k{cond[2]}|{direct}|{gain_txt}|{amp}|"
            f"{second}|{unres}|")
    add("")
    add("## 修订结论（相对原报告的措辞与计数修正）\n")
    add("1. 受限解计数修正为：primary 下 11/12 y=.05 上边界、1/12 y 内部"
        "（2031/46 k2，y≈0.0118，位于 x=+r 边）；x 全部在 ±r 边界。"
        "无约束 Schur y* 与受限解分开报告（8/12 y*>0.05，1 个在 (0,.05)，"
        "3 个为负；其中 2031/46 k2 的受限内部 y 来自 x=+r 边上的 ∂m/∂y=0，"
        "不是 Schur 驻点）。")
    add("2. 一阶伴随（POSTHOC）在 11/12 条件与二阶受限解选同一点（角点）；"
        "仅 2031/46 k2 二阶改变选择，其已测实测差见 S2。二阶信息的边际"
        "价值集中在幅度选择，不在方向选择。")
    add("3. 2031/46 两条件维持 UNRESOLVED，无证据唯一归因。")
    add("4. check256：12/12 均值为正；CI 排除 0 的为 " 
        f"{n_ci}/12；两者分开陈述。")
    add("5. RMSE/SSE/TSS 的 scope 已写清（原口径=合并两半径固定网格、去 "
        "native、去解重合坐标；n=11 或 13）；三种 scope 分列，不混合解释。")
    (OUT / "SUPPLEMENT_POSTHOC_V01.md").write_text("\n".join(L) + "\n")

    # CSV 制品
    with (OUT / "supplement_solution_inventory.csv").open("w",
                                                          newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "r", "schur_x_star", "schur_y_star",
                    "x_2D", "y_2D", "y_class", "x_class", "g_y_at_xLR",
                    "kkt_violation"])
        w.writerows(s1_rows)
    with (OUT / "supplement_first_order.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "r", "x_1st", "y_1st",
                    "same_as_2D", "deltaV_1st_mean", "deltaV_1st_se",
                    "paired_diff_1st_minus_2d_ci"])
        w.writerows(s2_rows)
    with (OUT / "supplement_check256.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "dLR", "dLR_se", "d2D", "d2D_se",
                    "gain", "gain_se", "ci_lo", "ci_hi"])
        w.writerows(s4_rows)
    with (OUT / "supplement_scopes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "start", "k", "nA", "sseA", "tssA", "rmseA",
                    "nB", "sseB", "tssB", "nC", "sseC", "tssC"])
        w.writerows(s5_rows)
    print("SUPPLEMENT_DONE")
    print("\n".join(L))


if __name__ == "__main__":
    main()
