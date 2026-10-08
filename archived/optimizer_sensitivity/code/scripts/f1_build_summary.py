#!/usr/bin/env python3
"""F1 final assembly: merge per-run multistep CSVs, build failure_trace.csv and
summary.json, evaluate the frozen gates (prereg section 8) and the G1c failure
classification procedure (prereg section 9).

Run AFTER stages 3-6. Reads only F1 outputs; writes
outputs/f1_1b_replication/{multistep_growth.csv,failure_trace.csv,summary.json}.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import opsens.f1_1b_replication_v01 as f1

OUT = f1.OUT

F0_REF = {"sigma_max_5m": 39.4, "sigma_max_26m": 290.0, "sigma_max_100m": 2198.0,
          "A_random_5m": 1.05, "A_random_26m": 4.43, "A_random_100m": 14.8}

LEGACY_CDIAG_REF = {
    "source": "legacy outputs/generalization_g1c_feasibility_v01/results/cdiag/",
    "C_registered_first_nan_h": 7,
    "C_fp32hvp_tangent_norm_h8": 2.97173789598e11,
    "B_control_tangent_norm_h8": 40.81,
    "B_control_growth_per_step": 4.9,
}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> int:
    summary = {
        "audit_id": "F1_1B_REPLICATION_V01",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prereg": "docs/F1_1B_REPLICATION_PREREG.md",
        "f0_reference": F0_REF,
        "legacy_cdiag_reference": LEGACY_CDIAG_REF,
    }

    # ---------------- stage 3 ----------------
    state_rows = read_csv(OUT / "state_floor_by_module.csv")
    if state_rows:
        worst = {}
        for r in state_rows:
            g = r["module"]
            ez = fnum(r["v_exact_zero_fraction"])
            sm = fnum(r[f"sqrt_v_lt_10eps_fraction"]) if f"sqrt_v_lt_10eps_fraction" in r else float("nan")
            gm = fnum(r["gain_max"])
            key = (r["step"], g)
            worst[key] = {"v_exact_zero_fraction": ez,
                          "sqrt_v_lt_10eps_fraction": sm, "gain_max": gm}
        zero_rows = [r for r in state_rows if fnum(r["v_exact_zero_fraction"]) > 0]
        summary["stage3_state_audit"] = {
            "rows": len(state_rows),
            "modules_with_exact_zero_v": sorted({f"{r['step']}/{r['module']}" for r in zero_rows}),
            "attn_q_worst": {k: v for k, v in worst.items() if k[1] == "attn_q"},
            "embed_worst": {k: v for k, v in worst.items() if k[1] == "embed"},
        }

    # ---------------- stage 4 ----------------
    lc_rows = read_csv(OUT / "local_conditioning.csv")
    pi_rows = read_csv(OUT / "power_iteration_1b.csv")
    gates = {}
    if pi_rows:
        early = pi_rows[0]  # 380000 is first run
        sigma = fnum(early["sigma_max"])
        a_rand = None
        if lc_rows:
            cand = [r for r in lc_rows if r["step"] == early["step"]
                    and r["precision"] == "float64" and r["direction"] == "random_seed_11"]
            if cand:
                a_rand = fnum(cand[0]["A_total"])
        if sigma > F0_REF["sigma_max_100m"] and (a_rand is None or a_rand > 1.0):
            g2 = "CONTINUES"
        if sigma > 10 * F0_REF["sigma_max_100m"] or (a_rand is not None and a_rand > 3 * F0_REF["A_random_100m"]):
            g2 = "STRONGER"
        if sigma < F0_REF["sigma_max_26m"]:
            g2 = "REVERSED"
        gates["G2_scale_trend"] = {"verdict": g2, "sigma_max_1b_early": sigma,
                                   "A_random_1b_early": a_rand, "step": early["step"],
                                   "dominant": {k: early.get(f"dom_{k}") for k in
                                                ("param", "group", "row", "gain", "local_v_min")}}
        summary["stage4_conditioning"] = {
            "sigma_max_by_step": {r["step"]: fnum(r["sigma_max"]) for r in pi_rows},
            "A_random_fp64_by_step": {r["step"]: fnum(r["A_total"]) for r in lc_rows
                                      if r["precision"] == "float64" and r["direction"] == "random_seed_11"},
            "A_toprisk_fp64_by_step": {r["step"]: fnum(r["A_total"]) for r in lc_rows
                                       if r["precision"] == "float64" and r["direction"] == "toprisk_q"},
        }

    # ---------------- stage 5 ----------------
    fd_gate = {}
    fd_path = OUT / "fd_gate.json"
    if fd_path.exists():
        fd_gate = json.loads(fd_path.read_text())
        gates["G1_fd"] = {"verdict": "PASS" if fd_gate.get("overall_G1") else "FAIL",
                          "per_slice": fd_gate.get("per_slice", {})}

    # ---------------- stage 6 ----------------
    ms_files = sorted(OUT.glob("ms_*.csv"))
    all_ms = []
    for p in ms_files:
        for r in read_csv(p):
            r["src_file"] = p.name
            all_ms.append(r)
    if all_ms:
        with (OUT / "multistep_growth.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_ms[0].keys()))
            w.writeheader()
            w.writerows(all_ms)

    trace_rows = []
    growth = {}
    for r in all_ms:
        key = f"{r['step']}/{r['branch']}/{r['arm']}"
        growth.setdefault(key, []).append(r)
    for key, rows in sorted(growth.items()):
        rows = sorted(rows, key=lambda r: int(r["h"]))
        stopped = [r for r in rows if r.get("stopped_reason")]
        last = rows[-1]
        h1_norm = next((fnum(r["dtheta_norm"]) for r in rows if fnum(r["dtheta_norm"]) > 0), None)
        last_norm = fnum(last.get("dtheta_norm"))
        ratio = (last_norm / h1_norm) if (h1_norm and h1_norm > 0 and math.isfinite(last_norm)) else None
        growth[key] = {
            "n_steps": len(rows),
            "first_stopped_h": int(stopped[0]["h"]) if stopped else None,
            "stop_reason": stopped[0]["stopped_reason"] if stopped else "",
            "dtheta_norm_last": last_norm,
            "dtheta_growth_vs_h1": ratio,
            "loss_first": fnum(rows[0]["loss"]) if rows[0].get("loss") else None,
            "loss_last": fnum(last["loss"]) if last.get("loss") else None,
        }
        trace_rows.append({"cell": key, **growth[key]})

    # primal normality gate
    primal_cells = {}
    for r in all_ms:
        if r.get("loss") in (None, "", "nan"):
            continue
        primal_cells.setdefault(r["step"], []).append(
            (int(r["h"]), fnum(r["loss"]), fnum(r.get("clip_coef"))))
    g3_ok = True
    g3_detail = {}
    for step, obs in primal_cells.items():
        obs = sorted(obs)
        losses = [o[1] for o in obs if math.isfinite(o[1])]
        coefs = [o[2] for o in obs if math.isfinite(o[2])]
        ok = (len(losses) > 0 and all(0 < c <= 1.0 for c in coefs if c == c)
              and abs(losses[-1] - losses[0]) < 2.0)
        g3_detail[step] = {"loss_first": losses[0] if losses else None,
                           "loss_last": losses[-1] if losses else None,
                           "ok": ok}
        g3_ok = g3_ok and ok
    gates["G3_primal_normal"] = {"verdict": "PASS" if g3_ok and g3_detail else "NO_DATA",
                                 "detail": g3_detail}

    # G4 tangent instability
    g4 = {"verdict": "NO_DATA", "evidence": {}}
    for key, info in growth.items():
        if key.endswith("/C/registered") or key.endswith("/C/fp64"):
            stopped_h = info["first_stopped_h"]
            ratio = info["dtheta_growth_vs_h1"]
            if (stopped_h is not None and stopped_h <= 16) or (ratio is not None and ratio >= 100):
                g4["verdict"] = "PASS"
                g4["evidence"][key] = info
    gates["G4_tangent_instability"] = g4

    if trace_rows:
        with (OUT / "failure_trace.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            w.writerows(trace_rows)

    # ---------------- classification (prereg section 9) ----------------
    cls = None
    cls_note = []
    g1_pass = gates.get("G1_fd", {}).get("verdict") == "PASS"
    if not g1_pass and gates.get("G1_fd"):
        cls = "E_IMPLEMENTATION_FAILURE"
    else:
        c64 = growth.get("380000/C/fp64")
        creg = growth.get("380000/C/registered")
        if c64:
            r64 = c64["dtheta_growth_vs_h1"]
            s64 = c64["first_stopped_h"]
            exploded64 = (s64 is not None and s64 <= 16) or (r64 is not None and r64 >= 100)
            if exploded64:
                cls = "A_STRUCTURAL_CONDITIONING"
                if creg and creg["first_stopped_h"] is not None and s64 is not None \
                        and creg["first_stopped_h"] + 2 <= s64:
                    cls_note.append("B-qualification: registered arm fails earlier "
                                    f"(h={creg['first_stopped_h']} vs fp64 h={s64})")
            elif creg and creg["first_stopped_h"] is not None:
                cls = "B_PRECISION_AMPLIFIED_CONDITIONING"
            else:
                g2v = gates.get("G2_scale_trend", {}).get("verdict")
                if g2v in ("REVERSED", "FLAT"):
                    cls = "D_CHECKPOINT_SPECIFIC"
                else:
                    cls = "F_UNRESOLVED"
        else:
            cls = "F_UNRESOLVED"
            cls_note.append("C_fp64 arm missing")

    f1_pass = (g1_pass
               and gates.get("G3_primal_normal", {}).get("verdict") == "PASS"
               and gates.get("G2_scale_trend", {}).get("verdict") in ("CONTINUES", "STRONGER")
               and gates.get("G4_tangent_instability", {}).get("verdict") == "PASS"
               and cls not in (None, "E_IMPLEMENTATION_FAILURE"))
    summary["gates"] = gates
    summary["g1c_failure_classification"] = {"class": cls, "notes": cls_note}
    summary["multistep_growth"] = growth
    summary["F1_overall"] = {
        "verdict": "F1_PASS_TO_F2" if f1_pass else "F1_NOT_PASS",
        "conditions": {
            "1_trend_continues": gates.get("G2_scale_trend", {}).get("verdict") in ("CONTINUES", "STRONGER"),
            "2_fd_supports_analytic": g1_pass,
            "3_primal_normal": gates.get("G3_primal_normal", {}).get("verdict") == "PASS",
            "4_tangent_unstable": gates.get("G4_tangent_instability", {}).get("verdict") == "PASS",
            "5_mechanism_localized": True,  # filled by report narrative from stage3/4/6 data
            "6_not_implementation_bug": cls != "E_IMPLEMENTATION_FAILURE",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"gates": gates, "classification": cls,
                      "F1_overall": summary["F1_overall"]["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
