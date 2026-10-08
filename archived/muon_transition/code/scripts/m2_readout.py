"""M2 readout per docs/M2_PROTOCOL_ADDENDUM.md §3-§4 (frozen).

Reads result.json for the full M2 grid: M1 p=0.5 cells (reused) + M2 local cells
(24 base + 4 freeze-justified extensions + any conditional extensions launched per §1.2).

Usage: python scripts/m2_readout.py [--check-extensions]

- Default: full readout written to outputs/m2/M2_SUMMARY.md + m2_summary.json.
- --check-extensions: do NOT write summary; print JSON list of conditional extension cells
  still needed at p in {0.25, 0.75} (base best at mult 2, no mult-4 result present).
Selection: per (arm,p) min dev V_Q (tie-break lower lr, then lower wd); p*_arm = argmin_p
(tie-break p=0.5, then lower p). Diagnostic fields are never read here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m2"

ARMS = ("adamw", "muon")
PS = (0.25, 0.5, 0.75)
WD_GRID = (0.01, 0.02)
BASE_MULTS = (0.5, 1.0, 2.0)
EXT_MULT = 4.0
REF_LR = {"adamw": 0.0003, "muon": 0.005}


def find_result(run_id):
    for stage in ("m2", "m1"):
        p = ROOT / "outputs" / stage / run_id / "result.json"
        if p.exists():
            return p
    return None


def cell_key(arm, p, mult, wd):
    return f"m2_{arm}_lr{mult}_wd{wd}_p{p}_s2030" if p != 0.5 else \
           f"m2_{arm}_lr{mult}_wd{wd}_p{p}_s2030"


def m1_run_id(arm, mult, wd):
    return f"m1_{arm}_lr{mult}_wd{wd}_p0.5_s2030"


def load_cell(arm, p, mult, wd):
    """Return dict with V_Q etc., or None if not run yet."""
    if p == 0.5 and mult in BASE_MULTS:
        rid = m1_run_id(arm, mult, wd)
    else:
        rid = cell_key(arm, p, mult, wd)
    rp = find_result(rid)
    if rp is None:
        return None
    r = json.loads(rp.read_text())
    fd = r["final_dev"]
    return {
        "run_id": rid, "arm": arm, "p": p, "mult": mult, "wd": wd,
        "lr": r["lr"], "v_q": fd["v_q"], "nll_A": fd["nll_A"], "nll_B": fd["nll_B"],
        "ok": bool(r["loss_decreased"] and not r["any_nan_loss"] and not r["any_nonfinite_gradnorm"]),
    }


def best_at(arm, p):
    """Best cell at (arm,p): min V_Q, tie-break lower mult then lower wd."""
    cells = [c for c in ALL_CELLS if c["arm"] == arm and c["p"] == p]
    cells = [c for c in cells if c["ok"]]
    if not cells:
        return None
    return min(cells, key=lambda c: (c["v_q"], c["mult"], c["wd"]))


def check_extensions():
    needed = []
    for arm in ARMS:
        for p in (0.25, 0.75):
            base = [load_cell(arm, p, m, wd) for m in BASE_MULTS for wd in WD_GRID]
            base = [c for c in base if c is not None and c["ok"]]
            ext = [load_cell(arm, p, EXT_MULT, wd) for wd in WD_GRID]
            ext_done = any(c is not None and c["ok"] for c in ext)
            if not base or ext_done:
                continue
            b = min(base, key=lambda c: (c["v_q"], c["mult"], c["wd"]))
            if b["mult"] == EXT_MULT / 2:  # mult 2 is base boundary -> one extension allowed
                for wd in WD_GRID:
                    if load_cell(arm, p, EXT_MULT, wd) is None:
                        needed.append({
                            "run_id": cell_key(arm, p, EXT_MULT, wd), "stage": "M2",
                            "arm": arm, "p": p, "seed": 2030, "lr_multiplier": EXT_MULT,
                            "lr": REF_LR[arm] * EXT_MULT, "reference_lr": REF_LR[arm],
                            "weight_decay": wd, "extension": True,
                            "justification": f"M2 addendum §1.2: base-grid best at mult=2.0 for {arm} at p={p}",
                        })
    print(json.dumps(needed, indent=1))
    return needed


ALL_CELLS = []  # filled in main


def main():
    global ALL_CELLS
    if "--check-extensions" in sys.argv:
        check_extensions()
        return

    # Load full grid: base cells at all p, extensions where present.
    for arm in ARMS:
        for p in PS:
            for mult in BASE_MULTS:
                for wd in WD_GRID:
                    c = load_cell(arm, p, mult, wd)
                    if c:
                        ALL_CELLS.append(c)
            for wd in WD_GRID:
                c = load_cell(arm, p, EXT_MULT, wd)
                if c:
                    ALL_CELLS.append(c)

    # ---- tables ----
    lines = ["# M2 summary — optimizer × mixture grid (seed 2030, B0=2^26, dev V_Q selection)",
             "", "## 1. Experiment table (all cells)", "",
             "| run_id | arm | p | mult | lr | wd | V_Q | NLL_A | NLL_B | ok |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(ALL_CELLS, key=lambda c: (c["arm"], c["p"], c["mult"], c["wd"])):
        lines.append(f"| {c['run_id']} | {c['arm']} | {c['p']:g} | {c['mult']:g} | {c['lr']:g} | "
                     f"{c['wd']:g} | {c['v_q']:.4f} | {c['nll_A']:.4f} | {c['nll_B']:.4f} | {c['ok']} |")

    lines += ["", "## 2. Optimizer × mixture performance (best cell per arm × p)", "",
              "| arm | p | best run | mult | wd | V_Q | NLL_A | NLL_B |", "|---|---|---|---|---|---|---|---|"]
    best_by_ap = {}
    for arm in ARMS:
        for p in PS:
            b = best_at(arm, p)
            best_by_ap[(arm, p)] = b
            if b:
                lines.append(f"| {arm} | {p:g} | {b['run_id']} | {b['mult']:g} | {b['wd']:g} | "
                             f"{b['v_q']:.4f} | {b['nll_A']:.4f} | {b['nll_B']:.4f} |")
            else:
                lines.append(f"| {arm} | {p:g} | MISSING | | | | | |")

    # ---- LR boundary resolution status (addendum §1.3) ----
    lines += ["", "## 2b. LR search resolution per (arm, p) (addendum §1.3)", "",
              "| arm | p | best mult | status |", "|---|---|---|---|"]
    lr_status = {}
    for arm in ARMS:
        for p in PS:
            b = best_by_ap[(arm, p)]
            if b is None:
                continue
            if b["mult"] == EXT_MULT:
                st = "LR_RANGE_UNRESOLVED"
            elif b["mult"] == max(BASE_MULTS) and load_cell(arm, p, EXT_MULT, WD_GRID[0]) is None:
                st = "EXTENSION_NOT_JUSTIFIED_BEST_AT_BOUNDARY"
            else:
                st = "RESOLVED"
            lr_status[(arm, p)] = st
            lines.append(f"| {arm} | {p:g} | {b['mult']:g} | {st} |")

    # ---- p* selection (tie-break p=0.5, then lower p) ----
    def p_star(arm):
        cands = [(p, best_by_ap[(arm, p)]) for p in PS]
        cands = [(p, b) for p, b in cands if b is not None]
        if not cands:
            return None, None
        order = {0.5: 0, 0.25: 1, 0.75: 2}
        p, b = min(cands, key=lambda pb: (pb[1]["v_q"], order[pb[0]]))
        return p, b

    p_adamw, b_adamw = p_star("adamw")
    p_muon, b_muon = p_star("muon")

    lines += ["", "## 3. Selected mixtures", "",
              f"- p*_AdamW = **{p_adamw}**  (V_Q = {b_adamw['v_q']:.4f})" if p_adamw else "- p*_AdamW = MISSING",
              f"- p*_Muon  = **{p_muon}**  (V_Q = {b_muon['v_q']:.4f})" if p_muon else "- p*_Muon  = MISSING"]

    # ---- migration cost ----
    R = None
    Rw = None
    if p_adamw is not None and p_muon is not None:
        m_at_a = best_by_ap[("muon", p_adamw)]
        m_at_m = best_by_ap[("muon", p_muon)]
        a_at_m = best_by_ap[("adamw", p_muon)]
        a_at_a = best_by_ap[("adamw", p_adamw)]
        if all(x is not None for x in (m_at_a, m_at_m, a_at_m, a_at_a)):
            R = m_at_a["v_q"] - m_at_m["v_q"]
            Rw = a_at_m["v_q"] - a_at_a["v_q"]
    lines += ["", "## 4. Migration cost", "",
              f"- R = Muon(p*_AdamW) − Muon(p*_Muon) = **{R:.4f}**" if R is not None else "- R = n/a",
              f"- R^AdamW (symmetric diagnostic) = AdamW(p*_Muon) − AdamW(p*_AdamW) = **{Rw:.4f}**" if Rw is not None else "- R^AdamW = n/a"]

    # ---- stop rule (addendum §4) ----
    verdict = "PENDING"
    detail = ""
    if p_adamw is not None and p_muon is not None:
        spread = {}
        for arm in ARMS:
            spreads = {}
            for p in PS:
                cells = [c for c in ALL_CELLS if c["arm"] == arm and c["p"] == p and c["ok"]]
                if cells:
                    vs = [c["v_q"] for c in cells]
                    spreads[p] = max(vs) - min(vs)
            spread[arm] = spreads
        proxy = {arm: max(spread[arm].values()) for arm in ARMS}
        adv_muon = abs(best_by_ap[("muon", p_muon)]["v_q"] - best_by_ap[("muon", p_adamw)]["v_q"])
        adv_adamw = abs(best_by_ap[("adamw", p_adamw)]["v_q"] - best_by_ap[("adamw", p_muon)]["v_q"])
        same_p = p_adamw == p_muon
        within = (adv_muon <= proxy["muon"]) or (adv_adamw <= proxy["adamw"])
        if same_p or within:
            verdict = "NO_SIGNAL_STOP_LINE"
            detail = (f"p* identical={same_p}; adv_muon={adv_muon:.4f} vs proxy_muon={proxy['muon']:.4f}; "
                      f"adv_adamw={adv_adamw:.4f} vs proxy_adamw={proxy['adamw']:.4f}")
        else:
            verdict = "SIGNAL_AWAIT_M3"
            detail = (f"adv_muon={adv_muon:.4f} > proxy_muon={proxy['muon']:.4f}; "
                      f"adv_adamw={adv_adamw:.4f} > proxy_adamw={proxy['adamw']:.4f}")

    lines += ["", "## 5. Stop rule verdict (addendum §4)", "",
              f"- verdict: **{verdict}**", f"- detail: {detail}",
              "- M3 recommendation: " + ("await M3 confirmation (3 new seeds, no retuning)" if verdict == "SIGNAL_AWAIT_M3"
                                          else "do not enter M3; negative result is terminal" if verdict == "NO_SIGNAL_STOP_LINE"
                                          else "grid incomplete")]

    (OUT / "M2_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (OUT / "m2_summary.json").write_text(json.dumps({
        "cells": ALL_CELLS,
        "p_star_adamw": p_adamw, "p_star_muon": p_muon,
        "R_muon": R, "R_adamw_symmetric": Rw,
        "lr_resolution": {f"{a}|{p}": s for (a, p), s in lr_status.items()},
        "verdict": verdict, "detail": detail,
    }, indent=2))
    print("\n".join(lines[-12:]))
    print(f"\nwrote {OUT/'M2_SUMMARY.md'} and m2_summary.json")


if __name__ == "__main__":
    main()
