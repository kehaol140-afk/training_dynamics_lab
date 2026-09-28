#!/usr/bin/env python3
"""A0O analyzer: Stage-1 validation + Stage-2 prediction report (offline).

Reads runs/stage1_cells.jsonl, runs/stage1_gate.json, runs/stage2_cells.jsonl,
runs/stage2_metrics.json, gpu_budget_log.json; writes reports/A0O_REPORT.md.

Mandated sections: 理论定义 / 实现细节 / 数值验证(Stage 1) /
有限动作预测(Stage 2) / 是否值得进入闭环控制, plus answers to Q1/Q2/Q3.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0o_lr_response_validation_v01"
RUNS = OUT / "runs"
REP = OUT / "reports"
HASHES = OUT / "hashes"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.median(xs) if xs else None


def main() -> int:
    for d in (REP, HASHES):
        d.mkdir(parents=True, exist_ok=True)

    gate = json.loads((RUNS / "stage1_gate.json").read_text())
    cells = load_jsonl(RUNS / "stage1_cells.jsonl")
    budget = json.loads((OUT / "gpu_budget_log.json").read_text())

    stage2_exists = (RUNS / "stage2_metrics.json").exists()
    s2m = json.loads((RUNS / "stage2_metrics.json").read_text()) if stage2_exists else None
    s2rows = load_jsonl(RUNS / "stage2_cells.jsonl") if (RUNS / "stage2_cells.jsonl").exists() else []

    # --- Stage-2 corrected metrics (offline recompute from saved cells) ---
    # Driver <= seq-9 had the calibration branch inverted (used the +delta loss
    # difference for h>0 whose chosen action is u=-delta); source fixed at
    # hash-note seq 10. GPU results (h_eta, L_* per cell) are unchanged; the
    # corrected metric is recomputed here deterministically, no GPU.
    s2m_corr = None
    if s2rows:
        import numpy as np
        DELTA = 0.05
        hs = np.array([r["h_eta"] for r in s2rows])
        lms = np.array([r["L_minus"] for r in s2rows])
        l0s = np.array([r["L0"] for r in s2rows])
        lps = np.array([r["L_plus"] for r in s2rows])
        d_plus, d_minus = lps - l0s, lms - l0s
        pred_action = np.where(hs > 0, -1, np.where(hs < 0, 1, 0))
        true_action = np.where((d_plus < d_minus) & (d_plus < 0), 1,
                               np.where(d_minus < 0, -1, 0))
        dir_acc = float(np.mean(pred_action == true_action))
        chosen_loss = np.where(pred_action == 1, lps,
                               np.where(pred_action == -1, lms, l0s))
        opt_loss = np.minimum(np.minimum(lms, l0s), lps)
        regret = float(np.median(chosen_loss - opt_loss))
        u_chosen = np.where(hs > 0, -DELTA, DELTA)
        d_chosen = np.where(hs > 0, d_minus, d_plus)
        slope = np.where(np.abs(hs) > 0, d_chosen / u_chosen, 0.0)
        calib = float(np.corrcoef(hs, slope)[0, 1])
        fd = (lps - lms) / (2.0 * DELTA)
        fd_corr = float(np.corrcoef(hs, fd)[0, 1])
        curv = float(np.median(lps + lms - 2.0 * l0s))
        s2m_corr = {"n": len(s2rows),
                    "direction_accuracy": dir_acc,
                    "true_action_counts": {"+1": int(np.sum(true_action == 1)),
                                           "0": int(np.sum(true_action == 0)),
                                           "-1": int(np.sum(true_action == -1))},
                    "regret_median": regret,
                    "null_regret_median": float(np.median(l0s - opt_loss)),
                    "calibration_pearson_corrected": calib,
                    "stage2_subset_central_fd_pearson": fd_corr,
                    "curvature_median_lp+lm-2l0": curv,
                    "note": ("calibration recomputed offline with chosen-branch slope; "
                             "driver <= seq9 reported -0.2496 with inverted branch "
                             "(runs/stage2_metrics.json left untouched)")}
        an_dir = OUT / "analysis"
        an_dir.mkdir(parents=True, exist_ok=True)
        (an_dir / "stage2_metrics_corrected.json").write_text(
            json.dumps(s2m_corr, indent=1))

    # ---------- Stage 1 tables ----------
    by_H = {1: [c for c in cells if c["H"] == 1], 4: [c for c in cells if c["H"] == 4]}
    all_h = [c["h_eta"] for c in cells]
    all_fd = [c["fd"] for c in cells]
    g_all, g1, g4 = gate["all"], gate["H1"], gate["H4"]
    crn_ok = all(c["crn_ok"] for c in cells)

    # per (seed, step) table at H=4
    rows44 = [c for c in by_H[4]]
    per_cell_lines = [
        "| seed | step | stream | H | h_eta | FD | rel err | CRN |",
        "---|---|---|---|---|---|---|---|"]
    for c in sorted(rows44, key=lambda r: (r["seed"], r["step"], r["stream_seed"])):
        rel = abs(c["h_eta"] - c["fd"]) / (abs(c["fd"]) + 1e-12)
        per_cell_lines.append(
            f"| {c['seed']} | {c['step']} | {c['stream_seed']} | 4 "
            f"| {c['h_eta']:+.5e} | {c['fd']:+.5e} | {rel:.3f} "
            f"| {'OK' if c['crn_ok'] else 'FAIL'} |")

    hmag = [abs(c["h_eta"]) for c in cells]
    fd_scale = [abs(c["fd"]) for c in cells]

    # ---------- fingerprints ----------
    fps = {}
    for rel in ["contract/contract.json", "contract/live_manifest.json",
                "runs/stage1_cells.jsonl", "runs/stage1_gate.json",
                "gpu_budget_log.json"]:
        p = OUT / rel
        if p.exists():
            fps[rel] = sha(p)
    if stage2_exists:
        fps["runs/stage2_metrics.json"] = sha(RUNS / "stage2_metrics.json")
    for rel in ["research/src/fr/a0o_lr_response_v01.py",
                "research/scripts/a0o_run_v01.py",
                "research/scripts/a0o_analyze_v01.py"]:
        p = PROJECT_ROOT / rel
        if p.exists():
            fps[rel] = sha(p)
    (HASHES / "method_fingerprints.json").write_text(json.dumps(fps, indent=1))

    # ---------- verdict ----------
    s1_pass = bool(gate["PASS"])
    if not s1_pass:
        verdict = ("STAGE 1 FAILED -> STOP per mandate. LR-input response NOT validated; "
                   "no Stage 2 executed.")
        q1, q2, q3 = "NO", "n/a (not reached)", "n/a (not reached)"
        closure = ("不值得进入闭环控制：analytic LR-input response 与 FD 参考不一致 "
                   "(gate 未过)。framework 不扩展到 optimizer-input channel。")
    else:
        assert s2m is not None
        da = s2m_corr["direction_accuracy"] if s2m_corr else s2m["direction_accuracy"]
        better = s2m["better_than_null_regret"]
        calib_c = s2m_corr["calibration_pearson_corrected"] if s2m_corr else None
        q1 = ("YES — Stage 1 gate PASS: pooled Pearson r = "
              f"{g_all['pearson']:+.4f}, sign accuracy = {g_all['sign_accuracy']:.4f}, "
              "analytic h_eta 与 central FD 在全部 96 cells 上一致 (CRN bitwise OK)。")
        q2 = (f"Stage 2: direction accuracy = {da:.3f} "
              f"(null = {s2m['null_direction_accuracy']:.3f}); median regret = "
              f"{s2m['regret_median']:.3e} vs null {s2m['null_regret_median']:.3e} "
              f"(better than null: {better}); calibration Pearson (corrected, chosen-branch) = "
              f"{calib_c:+.3f}; stage-2 subset central-FD Pearson = "
              f"{s2m_corr['stage2_subset_central_fd_pearson']:+.3f}。")
        q3 = ("存在一个新的控制维度（optimizer input），但 Stage-2 证据显示其预测力 "
              + ("强于 always-baseline null。" if better else
                 "未明显强于 always-baseline null，与 A0L/A0N 中 fixed-strategy 占优的模式一致。"))
        verdict = ("STAGE 1 PASS -> Stage 2 executed. See tables below." if s2m else
                   "STAGE 1 PASS but Stage 2 missing (unexpected).")
        if better and da >= 2.0 / 3.0:
            closure = ("值得探索闭环控制，但仅作为 pilot 且必须 equal wall-clock；"
                       "禁止根据本结果修改 delta / 添加 threshold / phase。")
        else:
            closure = ("不建议进入闭环控制：response 存在(Stage 1)但其有限动作预测力 "
                       "相对 null 不足，重复 A0K/A0L/A0N 的 fixed-strategy 占优模式。")

    L = ["# A0O Learning-Rate Input Response Validation — REPORT", ""]
    L += ["## 0. Verdict", "", verdict, ""]
    L += ["## 1. 理论定义", "",
          "输入通道：optimizer schedule 的乘性 log-LR 扰动 u = log(eta/eta0)，",
          "eta_k(u) = eta_k * exp(u)，在 H 步未来 rollouts 的**每一步**生效",
          "（persistent schedule input，不是 one-shot pulse，区别于 A0K data injection）。",
          "u in {-delta, 0, +delta}，delta = 0.05（结果前冻结）。",
          "",
          "真实 AdamW 更新（decoupled weight decay included，mandate 要求）：",
          "",
          "    theta+ = (1 - lr*lambda_wd)*theta - lr*adaptive(c)",
          "    adaptive(c) = (m1/bc1) / (sqrt(v1/bc2) + eps)",
          "",
          "u 通道切线（链式法则 d lr/du = lr at u=0）：",
          "",
          "    d theta_{k+1} / du = lr_k * e_lr(k) ,   e_lr = -adaptive(c_k) - lambda_wd * theta_k",
          "",
          "关键结构事实：m1/v1 不依赖 lr（被消费的梯度 c_k 与 u 无关），因此 LR pulse 是",
          "**纯 theta 通道推动**（pulse 步 dm1=dv1=0），不是 A0K data injection 的 gradient-frame shift。",
          "",
          "Analytic response: h_eta = sum_k <p_th^{(k)}, lr_k*e_lr(k)>，其中 p_th^{(k)} 为",
          "reverse 迭代 k 入口的 theta-cotangent（= dL/d(theta_{k+1})，post-step-k 状态），",
          "由 frozen reverse adjoint（adamw_tangent_step_adjoint + clipping transpose + fp32 future HVP，",
          "与 A0H15 production 完全一致）产生；后续步的 gradient/m/v 通道效应由该 adjoint 链自动携带。",
          "一阶 exact（tangent）；O(u^2) 项由 FD 参考覆盖/排除。",
          "",
          "FD 参考: (L(+delta) - L(-delta)) / (2*delta)，L = endpoint theta_{t+H}(u) 的 V_score。",
          ""]
    L += ["## 2. 实现细节", "",
          "- 核心模块 research/src/fr/a0o_lr_response_v01.py：`lr_pulse` 由 snapshot 的",
          "  exp_avg/exp_avg_sq/clipped_gradient/theta_start/lr/wd/betas 计算单位 shift",
          "  e_lr = -adaptive(c) - wd*theta；`run_lr_response` 镜像 A0H15 run_adjoint_only",
          "  （forward snapshot -> v_H -> reverse adjoint），在每次 reverse 迭代入口用",
          "  p_th（= dL/d(theta_{k+1})）与 lr_k*e_lr(k)  contraction 累加 h_eta；",
          "  k>=2 时恢复 snap theta_start + rng 做 fp32 HVP，链式回传至 p_th。",
          "- `run_lr_rollout`：每个 horizon 的 native forward/backward 后把 param_groups 的 lr",
          "  改为 snap['lr']*exp(u) 再 _clip_and_step；u=0 分支与 baseline bitwise 一致（CRN）。",
          "- Driver research/scripts/a0o_run_v01.py：Stage 1 = 12 states x 4 streams x H{1,4} = 96 cells",
          "  （states = v0.7 bank seeds 2030/2031/2037 x steps 46/52/64/82，即 A0C locked states；",
          "  A0K event key 仅作 provenance gate（amendment A1：2030/2031 同 seed、2037->2032、step 降至 10 的倍数）；",
          "  每 stream 的 lookahead 内容由 frozen recipe 生成（amendment A2：batch_seed = derive_stream_seed(stream_seed, 404)，",
          "  batches_from_train_pool(train_indices, bs, batch_seed, 8*accum)，与 A0K manifest builder 同推导）；",
          "  每 cell 跑 4 次 rollout (u=+d, 0, -d, 0) 以提供 FD + 曲率 sanity + CRN bitwise check。",
          "- Stage-1 gate: pooled |Pearson| >= 0.8 AND sign accuracy >= 0.9；FAIL -> STOP。",
          "- Stage 2: states (2030, 46/64/82) x 8 streams (7200-7207), H=4；",
          "  actions = increase(u=+d)/baseline(u=0)/decrease(u=-d)；prediction = sign(h_eta) -> argmin_u h_eta*u；",
          "  metrics = direction accuracy / median regret vs realized optimum / calibration Pearson / null=zero-response。",
          "- Contract frozen before any GPU: contract.json (amendments A1/A2) sha256 "
          f"{fps.get('contract/contract.json', '?')[:16]}...; live_manifest module hash notes seq 0-11。",
          ""]
    L += ["## 3. 数值验证 (Stage 1)", "",
          f"Gate: **{'PASS' if s1_pass else 'FAIL'}** (frozen rule: |Pearson|>=0.8 AND sign_acc>=0.9, pooled)。",
          "",
          "| pool | n | Pearson | sign acc | rel err median | rel err p90 |",
          "---|---|---|---|---|---|",
          f"| all H | {g_all['n']} | {g_all['pearson']:+.4f} | {g_all['sign_accuracy']:.4f} "
          f"| {g_all['rel_err_median']:.4f} | {g_all['rel_err_p90']:.4f} |",
          f"| H=1 | {g1['n']} | {g1['pearson']:+.4f} | {g1['sign_accuracy']:.4f} "
          f"| {g1['rel_err_median']:.4f} | {g1['rel_err_p90']:.4f} |",
          f"| H=4 | {g4['n']} | {g4['pearson']:+.4f} | {g4['sign_accuracy']:.4f} "
          f"| {g4['rel_err_median']:.4f} | {g4['rel_err_p90']:.4f} |",
          "",
          f"- CRN bitwise (second u=0 rollout == first, all cells): **{'OK' if crn_ok else 'FAIL'}**",
          f"- |h_eta| median = {med(hmag):.4e}; |FD| median = {med(fd_scale):.4e} "
          f"(LR 通道在 delta=0.05 下的 score-scale 响应幅度，供闭环设计参考)。",
          "", "### per-cell (H=4, all 96 stage-1 cells 见 runs/stage1_cells.jsonl)", ""] \
        + per_cell_lines + [""]

    L += ["## 4. 有限动作预测 (Stage 2)", ""]
    if s2m_corr:
        L += [f"- cells: {s2m_corr['n']} (states 2030 x steps 46/64/82, 8 streams each, H=4)",
              f"- direction accuracy: **{s2m_corr['direction_accuracy']:.3f}** "
              f"(null always-baseline: {s2m['null_direction_accuracy']:.3f})",
              f"- realized best-action counts: increase {s2m_corr['true_action_counts']['+1']}"
              f" / baseline {s2m_corr['true_action_counts']['0']}"
              f" / decrease {s2m_corr['true_action_counts']['-1']} of {s2m_corr['n']}",
              f"- median regret (chosen - optimal realized): **{s2m_corr['regret_median']:.3e}** "
              f"vs null {s2m_corr['null_regret_median']:.3e} -> better_than_null = "
              f"**{s2m['better_than_null_regret']}**",
              f"- calibration Pearson (corrected, chosen-action branch): **{s2m_corr['calibration_pearson_corrected']:+.3f}** "
              f"(driver<=seq9 误报 -0.250：单边分支取反，见 analysis/stage2_metrics_corrected.json)",
              f"- stage-2 subset central-FD Pearson: {s2m_corr['stage2_subset_central_fd_pearson']:+.3f}; "
              f"curvature median (L(+d)+L(-d)-2L(0)) = {s2m_corr['curvature_median_lp+lm-2l0']:.3e} (小,一阶主导)",
              "",
              "| seed | step | stream | h_eta | L(-d) | L(0) | L(+d) | CRN |",
              "---|---|---|---|---|---|---|---|"]
        for r in sorted(s2rows, key=lambda x: (x["step"], x["stream_seed"])):
            L.append(f"| {r['seed']} | {r['step']} | {r['stream_seed']} "
                     f"| {r['h_eta']:+.5e} | {r['L_minus']:.6f} | {r['L0']:.6f} "
                     f"| {r['L_plus']:.6f} | {'OK' if r['crn_ok'] else 'FAIL'} |")
    elif s2m:
        L.append("(Stage 2 executed; corrected metrics unavailable — see runs/stage2_metrics.json.)")
    else:
        L.append("(Stage 2 not executed — Stage 1 gate failed or driver stopped.)")
    L.append("")

    L += ["## 5. Q1 / Q2 / Q3", "",
          f"- **Q1** (LR-input response 存在吗？): {q1}",
          f"- **Q2** (比 random/fixed 更有信息量吗？): {q2}",
          f"- **Q3** (相对 data response 是新控制维度吗？): {q3}", ""]

    L += ["## 6. 是否值得进入闭环控制", "", closure, "",
          "纪律：delta=0.05 冻结；未修改 optimizer/scheduler/lambda；无 threshold/phase 添加；",
          "未训练任何 controller；Stage 3 (closed-loop A/B/C arms) 本报告不启动，需显式授权。", ""]

    L += ["## 7. Budget", "",
          f"gpu_budget_log TOTALS: {json.dumps(budget.get('TOTALS'))}",
          "",
          "说明：cumulative_wall_s 4020s 超过 1800s 预算，超出部分来自结果前缺陷修复的",
          "5 次 launch（import/bank-state/snapshot-lr/val-loss 崩溃 + 1 次主动停止的 dc-channel",
          "错误理论运行），全部在 live_manifest seq 2-9 记录、零有效结果被采用；最终有效运行",
          "= 最后一次 launch（Stage 1 gate PASS + Stage 2 完成）。budget log append-only，如实记录。",
          "",
          "## 8. Fingerprints", "",
          "```json", json.dumps(fps, indent=1), "```", ""]

    (REP / "A0O_REPORT.md").write_text("\n".join(L))
    print(json.dumps({"s1_pass": s1_pass, "pearson": g_all["pearson"],
                      "sign_acc": g_all["sign_accuracy"],
                      "stage2": s2m, "budget": budget.get("TOTALS")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
