#!/usr/bin/env python3
"""A0Q analyzer: offline recompute + consistency check + report + fingerprints."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/a0q_response_control_map_v01"
RUNS = OUT / "runs"
CHANNELS = ("lr", "clip", "wd")
SEEDS = (2030, 2031, 2037)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    rows = [json.loads(l) for l in (RUNS / "cells.jsonl").read_text().splitlines()]
    gates = json.loads((RUNS / "gates.json").read_text())
    ev = json.loads((RUNS / "evaluation.json").read_text())
    verdict = json.loads((RUNS / "verdict.json").read_text())

    # ---- offline recompute (frozen formulas) ----
    def gate_block(hs, fs):
        hs = np.asarray(hs, dtype=np.float64)
        fs = np.asarray(fs, dtype=np.float64)
        pearson = None
        if hs.std() > 0 and fs.std() > 0:
            pearson = float(np.corrcoef(hs, fs)[0, 1])
        return {"n": len(hs), "pearson": pearson,
                "sign_accuracy": float(np.mean(np.sign(hs) == np.sign(fs))),
                "rel_err_median": float(np.median(np.abs(hs - fs) / (np.abs(fs) + 1e-12)))}

    mism = []
    for ch in CHANNELS:
        g = gate_block([r[f"h_{ch}"] for r in rows], [r[f"fd_{ch}"] for r in rows])
        for key, val in g.items():
            if gates[ch][key] != val:
                mism.append((ch, key))
        hs = np.array([r[f"h_{ch}"] for r in rows], dtype=np.float64)
        if abs(float(np.median(np.abs(hs))) - ev[ch]["median_abs_h"]) > 1e-15:
            mism.append((ch, "median_abs_h"))
    consistent = not mism

    q_offline = {"consistent_with_driver": consistent, "mismatches": mism,
                 "gates": {ch: gates[ch]["PASS"] for ch in CHANNELS},
                 "verdict": verdict}
    (OUT / "analysis/q_offline.json").write_text(json.dumps(q_offline, indent=1))

    crn_all = all(r.get("crn_ok") for r in rows)
    n_clip_active = sum(sum(1 for p in r["per_step"] if p["clip_active"])
                        for r in rows)

    fps = {}
    for rel in ("research/src/fr/a0q_response_control_map_v01.py",
                "research/scripts/a0q_run_v01.py",
                "research/scripts/a0q_analyze_v01.py",
                "research/scripts/a0q_audit_pack_v01.py",
                "outputs/a0q_response_control_map_v01/contract/contract.json"):
        fps[rel] = sha_file(PROJECT_ROOT / rel)
    (OUT / "hashes/method_fingerprints.json").write_text(json.dumps(fps, indent=1))

    L = []
    L.append("# A0Q Response-Control Map Exploration\n")
    L.append("日期：2026-09-18。前置：STATUS §40（A0G–A0P 正负结果台账）已先于本轮写入。合同 sha256 `543851ca…71d2` 先于任何 GPU 冻结。\n")
    L.append("## 0. Verdict\n")
    for ch in CHANNELS:
        g = gates[ch]
        L.append(f"- **{ch}**：gate {'PASS' if g['PASS'] else 'FAIL'}（Pearson "
                 f"{g['pearson'] if g['pearson'] is not None else float('nan'):+.4f}，"
                 f"sign {g['sign_accuracy']:.3f}，rel err {g['rel_err_median']:.3f}）；"
                 f"median |h|={ev[ch]['median_abs_h']:.3e}"
                 f"（{ev[ch]['magnitude_vs_lr']:.2f}×LR），"
                 f"sign stability {ev[ch]['sign_stability']:.2f}，"
                 f"CV(|h|) {ev[ch]['cv_abs_h']:.2f}，{ev[ch]['cross_seed_sign_agreement']}")
    L.append(f"- 稳定通道（预注册规则：gate PASS ∧ sign stability≥2/3 ∧ 3/3 seed sign agreement，按 median |h| 排序）："
             f"{verdict['control_variable_candidates_in_rule_order'] or '无'}")
    L.append(f"- CRN 12/12：{crn_all}；clip 在全部 48 step 中激活 {n_clip_active} 次。离线重算一致：{consistent}\n")

    L.append("## 1. 理论与冻结\n")
    L.append("三通道均为持久乘性对数扰动（u=log(x/x₀)，δ=0.05 自冻结 a0o 导入，H=4 每步施加，与 A0O 验证的 FD 参考同语义）。解析源项：")
    L.append("- LR：A0O amendment-A2 纯 θ 通道推，h_lr=Σ⟨p_th,lr·e_lr⟩（冻结 lr_pulse import-only）。")
    L.append("- WD：纯 θ 通道（AdamW 中 wd 不触碰 m/v），h_wd=Σ⟨p_th,−lr·λ₀·θ_k⟩，λ₀=0.01（v0.2 bank 存储的原生值）。")
    L.append("- CLIP：dc 通道，clipped=g·min(1,τ/‖g‖)，∂clipped/∂τ=ĝ·1[preclip>τ₀]，h_clip=Στ₀⟨b_k,ĝ_k⟩·1[active]，b_k 为冻结 clip transpose 入口的 dc-cotangent，τ₀=1.0（cfg grad_clip 冻结断言）。")
    L.append("12 cell：seeds {2030,2031,2037} × steps {46,82} × streams {7400,7401}；stream 配方 derive_stream_seed(·,404)+batches_from_train_pool（8 micro-batches）；provenance 门复用 A0O-A1。禁止 controller/optimizer/λ 改动；不根据结果选择通道或状态。\n")

    L.append("## 2. Response matrix J（12 cell × 3 通道）\n")
    L.append("| seed | step | stream | h_lr | fd_lr | h_clip | fd_clip | h_wd | fd_wd |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['seed']} | {r['step']} | {r['stream_seed']} | {r['h_lr']:+.4e} | {r['fd_lr']:+.4e} "
                 f"| {r['h_clip']:+.4e} | {r['fd_clip']:+.4e} | {r['h_wd']:+.4e} | {r['fd_wd']:+.4e} |")
    L.append("")

    L.append("## 3. 逐通道评价（预注册四指标）\n")
    L.append("| 通道 | median |h| | ×LR magnitude | sign stability | seed sign agreement | CV(|h|) | FD Pearson | FD sign |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ch in CHANNELS:
        g = gates[ch]
        L.append(f"| {ch} | {ev[ch]['median_abs_h']:.3e} | {ev[ch]['magnitude_vs_lr']:.2f} "
                 f"| {ev[ch]['sign_stability']:.2f} | {ev[ch]['cross_seed_sign_agreement']} "
                 f"| {ev[ch]['cv_abs_h']:.2f} | {g['pearson']:+.3f} | {g['sign_accuracy']:.3f} |")
    L.append("")
    L.append("per-seed median h：")
    for ch in CHANNELS:
        ps = ev[ch]["per_seed_median"]
        L.append(f"- {ch}: " + "，".join(f"{s}: {v['median']:+.3e}({int(v['sign']):+d})"
                                          for s, v in ps.items()))
    L.append("")

    L.append("## 4. 结论：哪个通道稳定、哪个适合作为未来控制变量\n")
    ranked = verdict["control_variable_candidates_in_rule_order"]
    L.append(f"- 预注册规则下进入候选（稳定且跨 seed 符号一致）的通道（按 |h| 排序）：**{', '.join(ranked) if ranked else '无'}**。")
    for ch in CHANNELS:
        if not gates[ch]["PASS"]:
            reason = []
            if gates[ch]["pearson"] is None or abs(gates[ch]["pearson"]) < 0.8:
                reason.append(f"FD 一致性不足（Pearson {gates[ch]['pearson']}）")
            if gates[ch]["sign_accuracy"] < 0.9:
                reason.append(f"符号不稳（{gates[ch]['sign_accuracy']:.2f}）")
            L.append(f"- {ch} 未过 gate：" + "；".join(reason))
    if not ranked:
        L.append("- **结论：三通道无一满足全部预注册稳定性判据。** 按 mandate 如实报告：LR 通道虽有最大 |h|（A0O 已验证），")
        L.append("  但稳定控制变量的目标是 physical input channel——若 clip/wd 的响应幅度接近 LR 且 gate 通过，则它们与数据/loss 的几何耦合更直接（非 scheduler 标定通道）。")
    else:
        L.append(f"- **推荐未来控制变量（预注册规则内）：{ranked[0]}**。判定依据 = gate PASS + 跨 seed 符号一致 + 相对幅度。")
    L.append("- 与 A0K/A0L/A0N 负结果联合限定：任何通道进入闭环前必须证明相对 fixed 策略的 equal-wall-clock 净收益；A0P 限定长 τ 一阶外推不可靠（本轮 H=4 窗口内评估不受影响）。\n")

    L.append("## 5. 预算与制品\n")
    blog = json.loads((OUT / "gpu_budget_log.json").read_text())
    tot = blog["TOTALS"]
    L.append(f"- GPU：{tot['cumulative_wall_s']:.1f}s / {tot['budget_s']}s（budget log n_crashes 为 gate 标记非进程崩溃）。逐 cell 计时见 gpu_budget_log.json（append-only）。")
    L.append("- 制品：`outputs/a0q_response_control_map_v01/`（contract/ runs/ analysis/ reports/ hashes/ audit/）；代码 `research/src/fr/a0q_response_control_map_v01.py` + `research/scripts/a0q_{run,analyze,audit_pack}_v01.py`。")
    L.append("- A0O/A0P 及全部更早冻结制品未动；docs append-only。\n")
    L.append("## 6. 方法指纹\n")
    for rel, h in fps.items():
        L.append(f"- `{rel}` sha256 `{h[:16]}…`")

    (OUT / "reports/A0Q_REPORT.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"report": "reports/A0Q_REPORT.md", "consistent": consistent,
                      "crn_all": crn_all, "n_cells": len(rows),
                      "gates": {ch: gates[ch]["PASS"] for ch in CHANNELS},
                      "ranked": ranked}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
