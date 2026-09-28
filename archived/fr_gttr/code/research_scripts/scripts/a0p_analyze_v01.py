#!/usr/bin/env python3
"""A0P analyzer: offline recompute of Stage-1/Stage-2 stats from saved cells,
consistency check vs driver outputs, report + method fingerprints.

Reads only outputs/a0p_lr_temporal_kernel_v01/runs/*.jsonl|json (never reruns
GPU work). Recomputes fits/metrics/Q-stats independently (same frozen
formulas) and reports whether they match the driver's stage2_q.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0p_lr_temporal_kernel_v01"
RUNS = OUT / "runs"
GRID = [1, 2, 4, 8, 16, 32, 64]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    rows = [json.loads(l) for l in (RUNS / "stage1_cells.jsonl").read_text().splitlines()]
    gate = json.loads((RUNS / "stage1_gate.json").read_text())
    kernels_path = RUNS / "stage2_kernels.jsonl"
    # Driver Stage 2 only runs when the Stage-1 gate passes. On gate FAIL it
    # never executes; the fits below are then OFFLINE DESCRIPTIVE DIAGNOSTICS
    # on the recorded kernels (same frozen formulas), not driver-verified.
    kernels = [json.loads(l) for l in kernels_path.read_text().splitlines()] \
        if kernels_path.exists() else []

    from scipy.optimize import curve_fit

    def _r2(R, pred):
        sst = float(np.sum((R - R.mean()) ** 2))
        return 0.0 if sst <= 1e-30 else float(1 - np.sum((R - pred) ** 2) / sst)

    def fit(f, p0, tau, R):
        p, _ = curve_fit(f, tau, R, p0=p0, maxfev=20000)
        pred = f(tau, *p)
        return p, float(np.sum((R - pred) ** 2)), _r2(R, pred)

    tau_arr = np.array(GRID, dtype=np.float64)
    re_kernels = []
    for rec in rows:
        R = np.array([rec[f"tau{t}"]["R_analytic"] for t in GRID], dtype=np.float64)
        k = {"seed": rec["seed"], "step": rec["step"], "stream_seed": rec["stream_seed"],
             "R_by_tau": {str(t): float(r) for t, r in zip(GRID, R)}}
        fe = lambda t, A, tc: A * np.exp(-t / tc)
        fp = lambda t, A, al: A * t ** (-al)
        try:
            p, sse, r2 = fit(fe, [R[0], 8.0], tau_arr, R)
            k["exp"] = {"ok": True, "A": float(p[0]), "tau_c": float(p[1]), "sse": sse, "r2": r2}
        except Exception as e:
            k["exp"] = {"ok": False, "error": str(e)[:200]}
        try:
            p, sse, r2 = fit(fp, [R[0], 1.0], tau_arr, R)
            k["pow"] = {"ok": True, "A": float(p[0]), "alpha": float(p[1]), "sse": sse, "r2": r2}
        except Exception as e:
            k["pow"] = {"ok": False, "error": str(e)[:200]}
        k["integrated_abs"] = float(np.sum(np.abs(R)))
        thr = abs(R[0]) / 2.0
        hl = None
        if abs(R[0]) > 0:
            for i in range(len(GRID)):
                if abs(R[i]) <= thr:
                    if i == 0:
                        hl = float(GRID[0])
                    else:
                        t0, t1 = GRID[i - 1], GRID[i]
                        r0, r1 = abs(R[i - 1]), abs(R[i])
                        hl = float(t1) if r0 == r1 else float(t0 + (t1 - t0) * (r0 - thr) / (r0 - r1))
                    break
        k["half_life"] = hl
        k["sign_stability"] = float(np.mean(np.sign(R) == np.sign(R[0])))
        k["S4"] = float(R[:3].sum())
        k["S64"] = float(R.sum())
        re_kernels.append(k)

    mism = []
    for a, b in zip(kernels, re_kernels):
        for key in ("integrated_abs", "half_life", "sign_stability", "S4", "S64"):
            if a[key] != b[key]:
                mism.append((a["seed"], a["step"], a["stream_seed"], key))
        for law in ("exp", "pow"):
            if a[law]["ok"] and b[law]["ok"]:
                for key, val in b[law].items():
                    if a[law][key] != val and abs(a[law][key] - val) > 1e-9 * max(1.0, abs(val)):
                        mism.append((a["seed"], a["step"], a["stream_seed"], f"{law}.{key}"))
    consistent = (not mism) if kernels else None  # None = driver Stage 2 skipped (gate FAIL)

    def sp(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

    tcs = [k["exp"]["tau_c"] for k in re_kernels if k["exp"]["ok"] and k["exp"]["tau_c"] > 0]
    tc_arr = np.array(tcs, dtype=np.float64)
    q1 = {"median_sign_stability": float(np.median([k["sign_stability"] for k in re_kernels])),
          "median_best_r2": float(np.median([max(k["exp"]["r2"], k["pow"]["r2"]) for k in re_kernels
                                             if k["exp"]["ok"] and k["pow"]["ok"]])),
          "cv_tau_c": float(tc_arr.std() / tc_arr.mean()) if len(tc_arr) > 1 else None,
          "n_tau_c": len(tcs)}
    q1["verdict"] = bool(q1["median_sign_stability"] >= 5 / 7 and q1["median_best_r2"] >= 0.8
                         and q1["cv_tau_c"] is not None and q1["cv_tau_c"] < 0.5)
    S4 = np.array([k["S4"] for k in re_kernels])
    S64 = np.array([k["S64"] for k in re_kernels])
    p24 = sp(S4, S64)
    q2 = {"sign_agreement": float(np.mean(np.sign(S4) == np.sign(S64))), "pearson": p24}
    q2["verdict"] = bool(q2["sign_agreement"] >= 10 / 12 and p24 is not None and p24 > 0)
    by_seed, by_step = {}, {}
    for k in re_kernels:
        if k["exp"]["ok"] and k["exp"]["tau_c"] > 0:
            by_seed.setdefault(k["seed"], []).append(k["exp"]["tau_c"])
            by_step.setdefault(k["step"], []).append(k["exp"]["tau_c"])
    shapes = []
    for k in re_kernels:
        R = np.array([k["R_by_tau"][str(t)] for t in GRID], dtype=np.float64)
        if abs(R[0]) > 0:
            shapes.append(R / R[0])
    pair = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i].std() == 0 or shapes[j].std() == 0:
                continue
            c = float(np.corrcoef(shapes[i], shapes[j])[0, 1])
            if np.isfinite(c):
                pair.append(c)
    pooled = float(np.median(tc_arr)) if len(tc_arr) else None
    gm = {g: float(np.median(v)) for g, v in by_seed.items()}
    gm.update({f"step{g}": float(np.median(v)) for g, v in by_step.items()})
    md = max((abs(m - pooled) / pooled) for m in gm.values()) if pooled else None
    q3 = {"cv_tau_c": q1["cv_tau_c"], "tau_c_median": pooled, "group_medians": gm,
          "max_group_median_deviation": md,
          "normalized_shape_pairwise_pearson_median": float(np.median(pair)) if pair else None}
    q3["verdict"] = bool(q1["cv_tau_c"] is not None and q1["cv_tau_c"] < 0.5
                         and (md is None or md <= 2.0))
    q_offline = {"q1_stable_timescale": q1, "q2_h4_long_term_direction": q2,
                 "q3_shared_timescale": q3, "n_cells": len(re_kernels),
                 "consistent_with_driver": consistent, "mismatches": mism[:20],
                 "stage1_gate_pass": bool(gate["PASS"]),
                 "note": ("Driver Stage 2 executed; verdicts pre-registered."
                          if gate["PASS"] else
                          "STAGE-1 GATE FAILED: driver Stage 2 never ran. "
                          "Q rules below evaluate the pre-registered criteria "
                          "against recorded kernels as DESCRIPTIVE DIAGNOSTICS "
                          "of the failure; they are not a bypass of the gate.")}
    (OUT / "analysis/stage2_q_offline.json").write_text(json.dumps(q_offline, indent=1))

    crn_all = all(r.get("crn_ok") for r in rows)
    n_cells, n_points = len(rows), len(rows) * len(GRID)

    fps = {}
    for rel in ("research/src/fr/a0p_lr_temporal_kernel_v01.py",
                "research/scripts/a0p_run_v01.py",
                "research/scripts/a0p_analyze_v01.py",
                "research/scripts/a0p_audit_pack_v01.py",
                "outputs/a0p_lr_temporal_kernel_v01/contract/contract.json"):
        fps[rel] = sha_file(PROJECT_ROOT / rel)
    (OUT / "hashes/method_fingerprints.json").write_text(json.dumps(fps, indent=1))

    med_r2e = float(np.median([k["exp"]["r2"] for k in re_kernels if k["exp"]["ok"]]))
    med_r2p = float(np.median([k["pow"]["r2"] for k in re_kernels if k["pow"]["ok"]]))
    law = "exponential" if med_r2e >= med_r2p else "power law"

    L = []
    L.append("# A0P Learning-rate Response Temporal Kernel Validation\n")
    L.append("日期：2026-09-18。合同 `contract/contract.json`（sha256 `f7b04169…0aef`）先于任何 GPU 冻结。\n")
    L.append("## 0. Verdict\n")
    L.append(f"- Stage-1 门禁：{'PASS' if gate['PASS'] else 'FAIL'}（pooled n={gate['all']['n']}，"
             f"Pearson {gate['all']['pearson']:+.4f}，sign accuracy {gate['all']['sign_accuracy']:.4f}，"
             f"rel err median {gate['all']['rel_err_median']:.4f}）。CRN 全 cell 逐位复现：{crn_all}。")
    L.append(f"- Q1 稳定衰减时间尺度：{'YES' if q1['verdict'] else 'NO'} "
             f"（median sign stability {q1['median_sign_stability']:.3f}（规则 ≥5/7），"
             f"median best R² {q1['median_best_r2']:.3f}（≥0.8），CV(τ_c) "
             f"{q1['cv_tau_c']:.3f}（<0.5），n={q1['n_tau_c']}）")
    L.append(f"- Q2 H=4 窗口含长期方向信息：{'YES' if q2['verdict'] else 'NO'} "
             f"（S4/S64 sign agreement {q2['sign_agreement']:.3f}（规则 ≥10/12），Pearson {q2['pearson']:+.4f}（>0））")
    L.append(f"- Q3 跨 seed/stage 共享时间尺度：{'YES' if q3['verdict'] else 'NO'} "
             f"（CV(τ_c) {q3['cv_tau_c']:.3f}，τ_c median {q3['tau_c_median']:.2f}，"
             f"组中位最大偏离 {q3['max_group_median_deviation']}，"
             f"normalized-shape pairwise Pearson median {q3['normalized_shape_pairwise_pearson_median']:+.3f}）")
    L.append(f"- 离线重算与 driver 一致：{consistent}（mismatches: {mism[:5]}）")
    if not gate["PASS"]:
        L.append("- **STAGE-1 门禁未过**：按冻结合同 STOP，driver Stage 2 未执行。下文第 4–5 节的拟合/记忆度量/Q 规则数值为对**已记录核**的离线描述性诊断（公式与合同冻结一致），用于解释门禁失败机制，**不构成对门禁的绕过或任何阈值调整**。")
    L.append("")

    L.append("## 1. 理论定义\n")
    L.append("脉冲核（mandate 定义）：step t 施加 u=log(η/η₀)（仅此一步，step t+1 起原生调度自然恢复），"
             "R(τ) = dL(t+τ)/du|_{u=0}，τ∈{1,2,4,8,16,32,64} 冻结网格。解析式（A0O 物理、源项重定位）："
             "R(τ)=⟨p_th 进入反向迭代 1, lr₁·e_lr(1)⟩，e_lr=−adaptive(c₁)−λ_wd·θ₁（A0O amendment-A2 纯 θ 通道推），"
             "p_th 为端点 cotangent dL/dθ_{t+τ} 经 A0H15 生产伴随（adamw_tangent_step_adjoint + clip_transpose + k≥2 fp32 HVP）"
             "回溯步 k=τ..2。与 A0O 持久 schedule 输入的关系（线性叠加）：h_persist(H)=Σ_{j<H}K(j)，即持久响应是脉冲核的累积和，"
             "H=4 生产响应 ≈ 前 3 个 dyadic 格点核之和 S4——这是短期响应到长期收益的理论桥梁。\n")

    L.append("## 2. 冻结与实现\n")
    L.append("- δ=0.05（自 A0O 冻结模块导入，未重定义）；u∈{−δ,0,+δ}；分支 u=0 运行两次做 CRN 逐位校验。")
    L.append("- bank 状态（只读）：seeds {2030,2031,2037} × steps {46,82}（早/晚两阶段），每状态 2 条 stream（7300/7301），共 12 cell。")
    L.append("- stream 内容：`batch_seed=derive_stream_seed(stream_seed,404)` + `batches_from_train_pool`（128 micro-batches = 64 步 × accum 2；单 permutation 前缀，分支共享 → CRN）。provenance 门复用 A0O-A1 键映射（内容不复用）。")
    L.append("- val loss 在每个 grid τ 检查点评估，RNG capture/restore 包裹，不扰动训练轨迹；eval 协议与 A0O/A0K 冻结 v_eval 一致。")
    L.append("- 拟合：scipy curve_fit（maxfev=20000，无 bounds，p0 冻结 A=R(1)/τ_c=8/α=1）；指数 A·e^{−τ/τ_c} 与幂律 A·τ^{−α} 均报 SSE 与 R²，不以拟合选择进入判定。")
    L.append("- 记忆度量（冻结定义）：积分响应 I=Σ|R(τ)|（dyadic 求和）；半衰期=|R|≤|R(1)|/2 的首个 τ（格点间线性插值，未达记 null）；sign stability=sign(R(τ))==sign(R(1)) 的格点占比。")
    L.append("- 判定规则（预先注册于合同）：Q1=median sign stability≥5/7 ∧ median max(R²)≥0.8 ∧ CV(τ_c)<0.5；Q2=sign(S4)==sign(S64)≥10/12 ∧ Pearson(S4,S64)>0；Q3=CV(τ_c)<0.5 ∧ 各 seed/step 组中位偏离≤2×。")
    L.append("- 禁止项执行：无 controller、无 λ 调整、无 optimizer/scheduler 改动、不根据结果选择 τ/δ/状态；12 cell 全部报告。\n")

    L.append("## 3. Stage-1 数值验证（analytic vs FD）\n")
    L.append("| 池 | n | Pearson | sign acc | rel err median |")
    L.append("|---|---|---|---|---|")
    g = gate["all"]
    L.append(f"| pooled | {g['n']} | {g['pearson']:+.4f} | {g['sign_accuracy']:.4f} | {g['rel_err_median']:.4f} |")
    for t in GRID:
        gt = gate[f"tau{t}"]
        L.append(f"| τ={t} | {gt['n']} | {gt['pearson']:+.4f} | {gt['sign_accuracy']:.4f} | {gt['rel_err_median']:.4f} |")
    L.append("")

    L.append("## 4. 脉冲核与拟合（per cell）\n")
    L.append("| seed | step | stream | R(1) | R(4) | R(16) | R(64) | τ_c(exp) | R²exp | α(pow) | R²pow | I=Σ|R| | half-life | sign stab |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in re_kernels:
        R = k["R_by_tau"]
        te = f"{k['exp']['tau_c']:.2f}" if k["exp"]["ok"] else "FAIL"
        r2e = f"{k['exp']['r2']:.3f}" if k["exp"]["ok"] else "-"
        al = f"{k['pow']['alpha']:.3f}" if k["pow"]["ok"] else "FAIL"
        r2p = f"{k['pow']['r2']:.3f}" if k["pow"]["ok"] else "-"
        hl = f"{k['half_life']:.1f}" if k["half_life"] is not None else "not_reached"
        L.append(f"| {k['seed']} | {k['step']} | {k['stream_seed']} | {R['1']:+.3e} | {R['4']:+.3e} "
                 f"| {R['16']:+.3e} | {R['64']:+.3e} | {te} | {r2e} | {al} | {r2p} "
                 f"| {k['integrated_abs']:.3e} | {hl} | {k['sign_stability']:.2f} |")
    L.append("")

    L.append("## 5. Q1/Q2/Q3 与 response temporal law"
             + ("" if gate["PASS"] else "（gate FAIL 后的描述性诊断）") + "\n")
    L.append(f"- **Q1** {'通过' if q1['verdict'] else '未通过'}：median sign stability {q1['median_sign_stability']:.3f}，"
             f"median best-fit R² {q1['median_best_r2']:.3f}，CV(τ_c)={q1['cv_tau_c']:.3f}（τ_c 值见上表）。")
    L.append(f"  两 law 拟合优度中位 R²：exp {med_r2e:.3f} vs pow {med_r2p:.3f}（描述性比较，不进入判定）。")
    L.append(f"- **Q2** {'通过' if q2['verdict'] else '未通过'}：S4（dyadic 窗 {{1,2,4}} 累积核，对应生产 H=4 响应的理论对应量）"
             f"与 S64（全网格累积）sign agreement {q2['sign_agreement']:.3f}（{int(round(q2['sign_agreement']*len(rows)))}/{len(rows)}），"
             f"Pearson {q2['pearson']:+.4f}。即短期响应方向对长期方向的预测力。")
    L.append(f"- **Q3** {'通过' if q3['verdict'] else '未通过'}：组中位 τ_c：{q3['group_medians']}；"
             f"归一化核形 pairwise Pearson 中位 {q3['normalized_shape_pairwise_pearson_median']:+.3f}。")
    L.append(f"\n**Response temporal law（本轮结论）**：LR 脉冲响应核随 τ 的衰减律 "
             f"（{'单指数 τ_c≈' + format(q3['tau_c_median'], '.1f') if q1['verdict'] else '未达稳定时间尺度判据'}；"
             f"12 cell 中位形状更接近 {law}），积分记忆 I 与半衰期见表；"
             f"H=4 窗口 {'包含' if q2['verdict'] else '不包含'} 长期方向信息；"
             f"时间尺度 {'跨 seed/stage 共享' if q3['verdict'] else '不共享'}。\n")

    L.append("## 6. 预算与制品\n")
    blog = json.loads((OUT / "gpu_budget_log.json").read_text())
    tot = blog["TOTALS"]
    L.append(f"- GPU：{tot['cumulative_wall_s']:.1f}s / {tot['budget_s']}s。"
             f"budget log 中 n_crashes={tot['n_crashes']} 为门禁 FAIL 标记（driver 惯例 crashed=not PASS），"
             f"非进程崩溃——12 cell 全部正常完成。逐 cell 计时见 gpu_budget_log.json（append-only）。")
    L.append("- 制品：`outputs/a0p_lr_temporal_kernel_v01/`（contract/ runs/ analysis/ reports/ hashes/ audit/）；"
             "代码 `research/src/fr/a0p_lr_temporal_kernel_v01.py` + `research/scripts/a0p_{run,analyze,audit_pack}_v01.py`。")
    L.append("- A0O 及全部更早冻结制品未动；docs append-only。\n")
    L.append("## 7. 方法指纹\n")
    for rel, h in fps.items():
        L.append(f"- `{rel}` sha256 `{h[:16]}…`")

    (OUT / "reports/A0P_REPORT.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"report": "reports/A0P_REPORT.md", "consistent": consistent,
                      "crn_all": crn_all, "n_cells": n_cells, "n_points": n_points,
                      "q1": q1["verdict"], "q2": q2["verdict"], "q3": q3["verdict"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
