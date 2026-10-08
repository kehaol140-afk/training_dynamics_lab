#!/usr/bin/env python3
"""R0/R1 analysis for the F3 current-environment compatibility study.

Implements exactly the frozen metrics/gates of
docs/F3_REPRODUCTION_GATE_AMENDMENT.md §4-§6. Reads the 9 compat raw files,
writes repeatability.csv, phenomenology_gate.csv, and (only if PASS)
frozen_reference/ + CURRENT_ENV_REFERENCE_SHA256.txt + baseline_summary.json.

Run only after all 9 compat repeats complete:
    python scripts/f3_compatibility_analysis.py
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "outputs" / "f3_compatibility"
RAW = COMPAT / "raw"
FROZEN = COMPAT / "frozen_reference"

CELLS = [("2042", "26"), ("2043", "26"), ("2042", "64")]
EARLY = [("2042", "26"), ("2043", "26")]
MIDDLE = [("2042", "64")]
REPS = (1, 2, 3)
H_OUT = (1, 2, 4, 8, 16, 32)

# frozen tolerances (amendment §4/§5 — do not edit post hoc)
TOL_PRIMAL_SPREAD = 1e-6
TOL_GROWTH_MAXMIN = 1.10
P1_RATIO = 5.0
P2_RATIO = 5.0
P3_H1_REL = 1e-12
P3_H2_LO, P3_H2_HI = 0.9, 1.1


def load(seed, step, rep):
    p = RAW / f"compat_{seed}_{step}_rep{rep}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.load(p.open())


def norms_by_h(data, arm):
    return {r["h"]: r["arms"][arm]["dtheta_u_norm"] for r in data["steps"]}


def growth(norms, rep_data, H):
    return norms[H] / max(norms[1], 1e-30)


def main():
    data = {(s, st): {r: load(s, st, r) for r in REPS}
            for s, st in CELLS}

    rows_rep = []       # repeatability.csv rows
    r0 = {"A": True, "B": True, "C": True, "D": True}

    # ---------- per-cell repeatability metrics ------------------------------
    for (s, st) in CELLS:
        reps = data[(s, st)]
        losses = {r: [st_["loss"] for st_ in reps[r]["steps"]] for r in REPS}
        max_spread = 0.0
        for h in range(32):
            vals = [losses[r][h] for r in REPS]
            spread = (max(vals) - min(vals)) / max(abs(statistics.mean(vals)), 1e-30)
            max_spread = max(max_spread, spread)
        r0["A"] &= max_spread <= TOL_PRIMAL_SPREAD
        row = {"cell": f"{s}:{st}", "metric": "primal_loss_relative_spread_max",
               "value": max_spread, "tol": TOL_PRIMAL_SPREAD,
               "pass": max_spread <= TOL_PRIMAL_SPREAD}
        rows_rep.append(row)

        for arm in ("fp64/B", "fp64/C"):
            norms = {r: norms_by_h(reps[r], arm) for r in REPS}
            for H in (16, 32):
                g = {r: growth(norms[r], reps[r], H) for r in REPS}
                ratio = max(g.values()) / max(min(g.values()), 1e-30)
                ok = ratio <= TOL_GROWTH_MAXMIN
                r0["B"] &= ok
                rows_rep.append({"cell": f"{s}:{st}",
                                 "metric": f"{'C' if arm.endswith('C') else 'B'}_growth_ratio_maxmin_H{H}",
                                 "value": ratio, "tol": TOL_GROWTH_MAXMIN, "pass": ok})

        # branch ordering per repeat (gate C uses 2042:26 only, but record all)
        nb = {r: norms_by_h(reps[r], "fp64/B") for r in REPS}
        nc = {r: norms_by_h(reps[r], "fp64/C") for r in REPS}
        for r in REPS:
            ok = (nc[r][16] > nb[r][16]) and (nc[r][32] > nb[r][32])
            rows_rep.append({"cell": f"{s}:{st}", "metric": f"branch_order_rep{r}",
                             "value": int(ok), "tol": "C>B at H16,H32", "pass": ok})
        if (s, st) == (2042, 26):
            r0["C"] &= all((nc[r][16] > nb[r][16]) and (nc[r][32] > nb[r][32])
                           for r in REPS)
        # stash for R1
        data[(s, st)]["nb"], data[(s, st)]["nc"] = nb, nc

    # ---------- EARLY/MIDDLE ordering (gate D) ------------------------------
    for (s, st) in EARLY:
        mins_early = min(data[(s, st)]["nc"][r][16] for r in REPS)
        maxs_mid = max(data[("2042", "64")]["nc"][r][16] for r in REPS)
        ok = mins_early > maxs_mid
        r0["D"] &= ok
        rows_rep.append({"cell": f"{s}:{st} vs 2042:64",
                         "metric": "early_middle_order_minCearly16>maxCmiddle16",
                         "value": mins_early / max(maxs_mid, 1e-30),
                         "tol": ">1 strict", "pass": ok})

    r0_pass = all(r0.values())

    # ---------- R1 phenomenology gates --------------------------------------
    pg = []
    r1 = {"P1": True, "P2": True, "P3": True}
    for (s, st) in EARLY:
        nb, nc = data[(s, st)]["nb"], data[(s, st)]["nc"]
        for r in REPS:
            for H in (16, 32):
                ratio = (nc[r][H] / max(nb[r][H], 1e-30))
                ok = ratio >= P1_RATIO
                r1["P1"] &= ok
                pg.append({"gate": f"P1_{s}:{st}_rep{r}_H{H}", "value": ratio,
                           "threshold": f">={P1_RATIO}", "pass": ok})
    early16 = [data[(s, st)]["nc"][r][16] for (s, st) in EARLY for r in REPS]
    mid16 = [data[("2042", "64")]["nc"][r][16] for r in REPS]
    p2v = statistics.median(early16) / max(statistics.median(mid16), 1e-30)
    r1["P2"] &= p2v >= P2_RATIO
    pg.append({"gate": "P2_medianCearly16_over_medianCmiddle16", "value": p2v,
               "threshold": f">={P2_RATIO}", "pass": p2v >= P2_RATIO})
    for (s, st) in CELLS:
        nb, nc = data[(s, st)]["nb"], data[(s, st)]["nc"]
        for r in REPS:
            h1rel = abs(nc[r][1] - nb[r][1]) / max(nb[r][1], 1e-30)
            ok1 = h1rel <= P3_H1_REL
            h2ratio = nc[r][2] / max(nb[r][2], 1e-30)
            ok2 = P3_H2_LO <= h2ratio <= P3_H2_HI
            r1["P3"] &= ok1 and ok2
            pg.append({"gate": f"P3_{s}:{st}_rep{r}_H1rel", "value": h1rel,
                       "threshold": f"<={P3_H1_REL}", "pass": ok1})
            pg.append({"gate": f"P3_{s}:{st}_rep{r}_H2ratio", "value": h2ratio,
                       "threshold": f"[{P3_H2_LO},{P3_H2_HI}]", "pass": ok2})
    r1_pass = all(r1.values())

    if not r0_pass:
        verdict = "CURRENT_ENV_NOT_REPEATABLE"
    elif not r1_pass:
        verdict = "F2_PHENOMENOLOGY_NOT_REPRODUCED"
    elif r0_pass and r1_pass:
        verdict = "CURRENT_ENV_REFERENCE_ESTABLISHED"
    else:
        verdict = "UNRESOLVED"

    # ---------- write CSVs ---------------------------------------------------
    COMPAT.mkdir(parents=True, exist_ok=True)
    with (COMPAT / "repeatability.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "metric", "value", "tol", "pass"])
        w.writeheader()
        w.writerows(rows_rep)
    with (COMPAT / "phenomenology_gate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gate", "value", "threshold", "pass"])
        w.writeheader()
        w.writerows(pg)

    summary = {
        "verdict": verdict,
        "r0_pass": r0_pass, "r0_gates": r0,
        "r1_pass": r1_pass, "r1_gates": r1,
        "frozen": {"TOL_PRIMAL_SPREAD": TOL_PRIMAL_SPREAD,
                   "TOL_GROWTH_MAXMIN": TOL_GROWTH_MAXMIN,
                   "P1_RATIO": P1_RATIO, "P2_RATIO": P2_RATIO,
                   "P3_H1_REL": P3_H1_REL, "P3_H2": [P3_H2_LO, P3_H2_HI]},
        "per_cell": {},
    }
    for (s, st) in CELLS:
        nb, nc = data[(s, st)]["nb"], data[(s, st)]["nc"]
        summary["per_cell"][f"{s}:{st}"] = {
            "G_B": {str(H): [nb[r][H] for r in REPS] for H in H_OUT},
            "G_C": {str(H): [nc[r][H] for r in REPS] for H in H_OUT},
            "C_over_B_H16": [nc[r][16] / max(nb[r][16], 1e-30) for r in REPS],
            "C_over_B_H32": [nc[r][32] / max(nb[r][32], 1e-30) for r in REPS],
        }

    # ---------- frozen reference (only on PASS) ------------------------------
    if verdict == "CURRENT_ENV_REFERENCE_ESTABLISHED":
        FROZEN.mkdir(parents=True, exist_ok=True)
        tangent_rows = []
        for (s, st) in CELLS:
            d1 = data[(s, st)][1]
            for r_ in d1["steps"]:
                for arm in ("fp64/B", "fp64/C"):
                    tangent_rows.append({"cell": f"{s}:{st}", "arm": arm,
                                         "h": r_["h"],
                                         "dtheta_u_norm": r_["arms"][arm]["dtheta_u_norm"]})
        baseline = {"reference": "repeat 1 of 3 per cell (frozen choice, amendment §6)",
                    "manifest": {f"{s}:{st}": data[(s, st)][1]["manifest_sha256"]
                                 for s, st in CELLS},
                    "losses": {f"{s}:{st}": [st_["loss"] for st_ in data[(s, st)][1]["steps"]]
                               for s, st in CELLS},
                    "tangent": tangent_rows}
        (FROZEN / "baseline_summary.json").write_text(json.dumps(baseline, indent=1))

        with (FROZEN / "primal_traces.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "rep", "h", "loss", "lr", "clip_coef", "grad_norm"])
            for (s, st) in CELLS:
                for r in REPS:
                    for st_ in data[(s, st)][r]["steps"]:
                        w.writerow([f"{s}:{st}", r, st_["h"], st_["loss"], st_["lr"],
                                    st_["clip_coef"], st_["grad_norm"]])
        for arm, fname in (("fp64/B", "B_tangent_traces.csv"),
                           ("fp64/C", "C_tangent_traces.csv")):
            with (FROZEN / fname).open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["cell", "rep", "h", "dtheta_u_norm", "dm_u_norm",
                            "dv_u_norm", "dz_u_norm", "hvp_norm"])
                for (s, st) in CELLS:
                    for r in REPS:
                        for st_ in data[(s, st)][r]["steps"]:
                            a = st_["arms"][arm]
                            w.writerow([f"{s}:{st}", r, st_["h"], a["dtheta_u_norm"],
                                        a["dm_u_norm"], a["dv_u_norm"], a["dz_u_norm"],
                                        a["hvp_norm"]])
        with (FROZEN / "module_shares.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "rep", "h", "arm", "group", "share"])
            for (s, st) in CELLS:
                for r in REPS:
                    for st_ in data[(s, st)][r]["steps"]:
                        if "shares" not in st_["arms"]["fp64/C"]:
                            continue
                        for arm in ("fp64/B", "fp64/C"):
                            sh = st_["arms"][arm].get("shares")
                            if not sh:
                                continue
                            for g, v in sh.items():
                                w.writerow([f"{s}:{st}", r, st_["h"], arm, g, v])
        with (FROZEN / "hvp_norms.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "rep", "h", "hvp_norm_B", "hvp_norm_C"])
            for (s, st) in CELLS:
                for r in REPS:
                    for st_ in data[(s, st)][r]["steps"]:
                        w.writerow([f"{s}:{st}", r, st_["h"],
                                    st_["arms"]["fp64/B"]["hvp_norm"],
                                    st_["arms"]["fp64/C"]["hvp_norm"]])
        (FROZEN / "repeatability.csv").write_text(
            (COMPAT / "repeatability.csv").read_text())

        # canonical bundle hash
        h = hashlib.sha256()
        for p in sorted(FROZEN.iterdir()):
            if p.is_file():
                h.update(p.name.encode())
                h.update(hashlib.sha256(p.read_bytes()).digest())
        (COMPAT / "CURRENT_ENV_REFERENCE_SHA256.txt").write_text(
            h.hexdigest() + "\n")
        summary["frozen_reference_sha256"] = h.hexdigest()

    (COMPAT / "baseline_summary.json").write_text(json.dumps(
        {"verdict": verdict,
         "r0": {"pass": r0_pass, "gates": r0},
         "r1": {"pass": r1_pass, "gates": r1},
         "per_cell_growth": summary["per_cell"],
         "frozen_tolerances": summary["frozen"],
         "frozen_reference_sha256": summary.get("frozen_reference_sha256")}, indent=1))

    print(json.dumps({"verdict": verdict, "r0": r0, "r1": r1}, indent=1))
    return 0 if verdict == "CURRENT_ENV_REFERENCE_ESTABLISHED" else 1


if __name__ == "__main__":
    sys.exit(main())
