#!/usr/bin/env python3
"""F2.5 analysis: build frozen outputs from raw per-cell JSONs.

Implements exactly docs/F2_5_TRUST_REGION_PREREG.md §2 metrics:
  Delta_H(+/-delta), central FD D_H(delta), R_H(delta) (eps_metric=1e-7),
  usable := R<=0.2 AND sign(h*delta)==sign(Delta) on BOTH arms,
  resolvable := |Delta(+-delta)|>eps_abs (both arms) AND sign stability to the
    next larger grid delta (per arm; largest delta self-compares -> auto-pass),
  delta*(H) := largest delta with a contiguous usable prefix over resolvable
    grid points (opsens.f25_trust_region_v01.trust_radius),
  FD convergence: adjacent relative drift of D_H over resolvable deltas,
    FD_NONCONVERGENT if max drift > 0.2; Richardson h+c*delta fit on the
    smallest three resolvable deltas (report only, no gate).

Verdict procedure: prereg §5 (first hit wins), computed on fp64/C (primary)
with fp64/B recorded as control.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "f2_5_trust_region"

DELTA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1)
H_READOUTS = (1, 2, 4, 8, 16, 32)
EPS_METRIC = 1e-7
EPS_ABS = 1e-6
R_THR = 0.2
CELLS = ((2042, 26, "EARLY"), (2043, 26, "EARLY"), (2042, 64, "MIDDLE"))
INTERVENTIONS = ("none", "mask_den_0.01%", "mask_den_0.1%", "vfloor_eps", "vfloor_10eps")
VFLOOR_IVS = ("mask_den_0.01%", "mask_den_0.1%", "vfloor_eps", "vfloor_10eps")

sys.path.insert(0, str(ROOT / "src"))
from opsens.f25_trust_region_v01 import trust_radius  # noqa: E402


def load_cell(seed, step):
    p = OUT / f"raw_{seed}_{step}.json"
    with p.open() as fh:
        return json.load(fh)


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def cell_fd_table(raw):
    """{(H): {delta: row}} from raw fd rows."""
    tab = {}
    for r in raw["fd"]:
        tab.setdefault(r["H"], {})[r["delta"]] = r
    return tab


def resolvable_flags(tab_h):
    """Frozen RESOLVABLE rule for one (cell,H): ascending delta grid."""
    flags = {}
    n = len(DELTA_GRID)
    for i, d in enumerate(DELTA_GRID):
        r = tab_h[d]
        dp, dm = r["Delta_plus"], r["Delta_minus"]
        ok = (abs(dp) > EPS_ABS) and (abs(dm) > EPS_ABS)
        if ok:
            dnext = DELTA_GRID[i + 1] if i + 1 < n else d
            rn = tab_h[dnext]
            ok = (sign(dp) == sign(rn["Delta_plus"])) and \
                 (sign(dm) == sign(rn["Delta_minus"]))
        flags[d] = ok
    return flags


def usable_flags(tab_h, h_tan):
    """Frozen usable rule on BOTH arms. Returns (usable, R_plus, R_minus)."""
    u, rp, rm = {}, {}, {}
    for d in DELTA_GRID:
        r = tab_h[d]
        pred_p, pred_m = h_tan * d, h_tan * (-d)
        Rp = abs(r["Delta_plus"] - pred_p) / (abs(pred_p) + EPS_METRIC)
        Rm = abs(r["Delta_minus"] - pred_m) / (abs(pred_m) + EPS_METRIC)
        rp[d], rm[d] = Rp, Rm
        u[d] = (Rp <= R_THR) and (Rm <= R_THR) and \
               (sign(pred_p) == sign(r["Delta_plus"])) and \
               (sign(pred_m) == sign(r["Delta_minus"]))
    return u, rp, rm


def fd_convergence(tab_h, resol):
    """Return (rows, h_FD, nonconvergent, richardson_h)."""
    rows = []
    ds = [d for d in DELTA_GRID if resol[d]]
    hfd = None
    nonconv = False
    rich = None
    prev = None
    for d in DELTA_GRID:
        D = tab_h[d]["D_H"]
        adj = None
        if resol[d] and prev is not None:
            adj = abs(D - prev) / (abs(prev) + 1e-300)
            if adj > 0.2:
                nonconv = True
        if resol[d]:
            prev = D
        rows.append({"delta": d, "D_H": D, "resolvable": resol[d],
                     "adj_rel_diff": adj})
    if ds:
        hfd = tab_h[ds[0]]["D_H"]
    if len(ds) >= 3:
        # Richardson view: D(delta) = h + c*delta on smallest three resolvable
        xs = ds[:3]
        ys = [tab_h[d]["D_H"] for d in xs]
        xbar, ybar = sum(xs) / 3.0, sum(ys) / 3.0
        sxx = sum((x - xbar) ** 2 for x in xs)
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        c = sxy / sxx if sxx > 0 else 0.0
        rich = ybar - c * xbar
    return rows, hfd, nonconv, rich


def main():
    raws = {(s, st): load_cell(s, st) for s, st, _ in CELLS}

    # ---- tangent tables ----
    def tan_lookup(raw):
        """{(arm_key): {h: row}}"""
        tab = {}
        for r in raw["tangent"]:
            if "h_tangent" in r:
                tab.setdefault(r["arm"], {})[r["h"]] = r
        return tab

    trust_rows = []
    fd_rows = []
    interv_rows = []
    cond_rows = []
    detail = {}

    for seed, step, phase in CELLS:
        raw = raws[(seed, step)]
        tab = cell_fd_table(raw)
        tans = tan_lookup(raw)
        cell_id = f"{seed}:{step}"
        detail[cell_id] = {"forward_match": raw.get("forward_bitwise_match"),
                           "max_dloss": raw.get("max_baseline_loss_abs_diff")}

        per_h = {}
        for H in H_READOUTS:
            tab_h = tab[H]
            resol = resolvable_flags(tab_h)
            fd_conv_rows, h_fd, nonconv, rich = fd_convergence(tab_h, resol)
            for r in fd_conv_rows:
                fd_rows.append({"cell": cell_id, "phase": phase, "H": H, **r})
            per_h[H] = {"resol": resol, "h_FD": h_fd,
                        "fd_nonconvergent": nonconv, "richardson_h": rich}

            for iv in INTERVENTIONS:
                for br in ("C", "B"):
                    key = f"{iv}|{br}"
                    trow = tans.get(key, {}).get(H)
                    if trow is None:
                        trust_rows.append({"cell": cell_id, "phase": phase,
                                           "intervention": iv, "branch": br, "H": H,
                                           "h_tangent": None, "delta_star": None,
                                           "n_resolvable": sum(resol.values()),
                                           "n_usable": 0, "status": "ARM_STOPPED"})
                        continue
                    h_tan = trow["h_tangent"]
                    u, rp, rm = usable_flags(tab_h, h_tan)
                    dstar = trust_radius(DELTA_GRID,
                                         [u[d] for d in DELTA_GRID],
                                         [resol[d] for d in DELTA_GRID])
                    trust_rows.append({"cell": cell_id, "phase": phase,
                                       "intervention": iv, "branch": br, "H": H,
                                       "h_tangent": h_tan, "delta_star": dstar,
                                       "n_resolvable": sum(resol.values()),
                                       "n_usable": sum(u.values()),
                                       "status": "OK"})
                    detail[cell_id].setdefault("R", {}).setdefault(key, {})[H] = {
                        "R_plus": rp, "R_minus": rm, "usable": u,
                        "resolvable": resol}

            # intervention summary rows (C branch primary)
            nrmC = {iv: tans.get(f"{iv}|C", {}).get(H, {}).get("dtheta_u_norm")
                    for iv in INTERVENTIONS}
            hC = {iv: tans.get(f"{iv}|C", {}).get(H, {}).get("h_tangent")
                  for iv in INTERVENTIONS}
            hB = {iv: tans.get(f"{iv}|B", {}).get(H, {}).get("h_tangent")
                  for iv in INTERVENTIONS}
            dstarC = {iv: next((r["delta_star"] for r in trust_rows
                                if r["cell"] == cell_id and r["intervention"] == iv
                                and r["branch"] == "C" and r["H"] == H), None)
                      for iv in INTERVENTIONS}
            nrm1 = tans.get("none|C", {}).get(1, {}).get("dtheta_u_norm")
            growth = (tans.get("none|C", {}).get(H, {}).get("dtheta_u_norm") / nrm1
                      if nrm1 else None)
            interv_rows.append({"cell": cell_id, "phase": phase, "H": H,
                                "growth_none_C": growth, **{f"h_{iv}": hC[iv] for iv in INTERVENTIONS},
                                **{f"hB_{iv}": hB[iv] for iv in INTERVENTIONS},
                                **{f"dstar_{iv}": dstarC[iv] for iv in INTERVENTIONS},
                                **{f"nrm_{iv}": nrmC[iv] for iv in INTERVENTIONS}})

        # conditioning inputs (F0/F1 artifacts, read-only)
        cond = load_conditioning(seed, step)
        for H in H_READOUTS:
            dstar_map = {iv: next((r["delta_star"] for r in trust_rows
                                   if r["cell"] == cell_id and r["intervention"] == iv
                                   and r["branch"] == "C" and r["H"] == H), None)
                         for iv in INTERVENTIONS}
            cond_rows.append({"cell": cell_id, "phase": phase, "H": H,
                              "A_v": cond["A_v"], "sigma_max": cond["sigma_max"],
                              "attn_q_share": cond["attn_q_share"],
                              "floor_frac": cond["floor_frac"],
                              "growth_none_C": interv_rows[-(6 - H_READOUTS.index(H))]["growth_none_C"],
                              "fd_nonconvergent": per_h[H]["fd_nonconvergent"],
                              "h_FD": per_h[H]["h_FD"],
                              **{f"dstar_{iv}": dstar_map[iv] for iv in INTERVENTIONS}})

    write_csv(OUT / "trust_radius.csv", trust_rows)
    write_csv(OUT / "fd_convergence.csv", fd_rows)
    write_csv(OUT / "backward_intervention.csv", interv_rows)
    write_csv(OUT / "conditioning_vs_radius.csv", cond_rows)

    verdict, vdetail = decide(trust_rows, fd_rows, raws)
    summary = {
        "audit_id": "F2_5_TRUST_REGION_V01",
        "generated": time_now(),
        "prereg": "docs/F2_5_TRUST_REGION_PREREG.md",
        "cells": {f"{s}:{st}": {"phase": ph,
                                "stream_seed": raws[(s, st)]["stream_seed"],
                                "manifest_sha256": raws[(s, st)]["manifest_sha256"],
                                "forward_match": raws[(s, st)].get("forward_bitwise_match"),
                                "max_baseline_loss_abs_diff": raws[(s, st)].get("max_baseline_loss_abs_diff")}
                  for s, st, ph in CELLS},
        "constants": {"delta_grid": DELTA_GRID, "H_readouts": H_READOUTS,
                      "eps_metric": EPS_METRIC, "eps_abs": EPS_ABS,
                      "R_threshold": R_THR},
        "detail": detail,
        "verdict_detail": vdetail,
        "verdict": verdict,
        "artifacts": ["trust_radius.csv", "fd_convergence.csv",
                      "backward_intervention.csv", "conditioning_vs_radius.csv"],
    }
    with (OUT / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"verdict: {verdict}")
    return 0


def load_conditioning(seed, step):
    """Read-only pulls from frozen F0/F1 artifacts (prereg §4)."""
    import statistics
    f1 = json.loads((ROOT / "outputs" / "f1_mechanism" / "summary.json").read_text())
    A_v = f1["F1E_precision_amplification"]["fp64|v"][f"{seed}|{step}"]
    # sigma_max: F0 power_iteration only covers seed 2042 -> G-channel diagonal
    # max from per_parameter_group.csv (prereg-allowed fallback), else power_iteration.
    sig = None
    pi = list(csv.DictReader((ROOT / "outputs" / "f0_existing_bank_audit" /
                              "power_iteration.csv").open()))
    for r in pi:
        if r["scale"] == "g1b_100m" and int(r["seed"]) == seed and int(r["step"]) == step:
            sig = float(r["sigma_max"])
    if sig is None:
        pp = list(csv.DictReader((ROOT / "outputs" / "f0_existing_bank_audit" /
                                  "per_parameter_group.csv").open()))
        sig = max(float(r["diag_gain_max"]) for r in pp
                  if r["scale"] == "g1b_100m" and int(r["seed"]) == seed
                  and int(r["step"]) == step)
        sig = -sig  # mark provenance: negative = diag-max proxy
    ml = list(csv.DictReader((ROOT / "outputs" / "f1_mechanism" /
                              "module_localization.csv").open()))
    aq = next(float(r["G2_share_attn_q"]) for r in ml
              if r["scale"] == "g1b_100m" and int(r["seed"]) == seed
              and int(r["step"]) == step)
    pp = list(csv.DictReader((ROOT / "outputs" / "f0_existing_bank_audit" /
                              "per_parameter_group.csv").open()))
    num = den = 0.0
    for r in pp:
        if r["scale"] == "g1b_100m" and int(r["seed"]) == seed and int(r["step"]) == step:
            n = int(r["v_n_coords"])
            num += float(r["v_sqrt_v_lt_10eps_fraction"]) * n
            den += n
    return {"A_v": A_v, "sigma_max": sig, "attn_q_share": aq,
            "floor_frac": num / den}


def time_now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path.name}: {len(rows)} rows")


def decide(trust_rows, fd_rows, raws):
    """Frozen verdict procedure (prereg §5), first hit wins. Primary branch C."""
    d = {}

    def dstar(cell, iv, H, br="C"):
        for r in trust_rows:
            if r["cell"] == cell and r["intervention"] == iv and r["H"] == H and r["branch"] == br:
                return r["delta_star"]
        return None

    early = ["2042:26", "2043:26"]
    middle = "2042:64"
    cells = early + [middle]

    # collapse pattern per cell (baseline fp64/C)
    collapse = {}
    for c in cells:
        d1, d16 = dstar(c, "none", 1), dstar(c, "none", 16)
        collapse[c] = (d16 is not None and d1) and \
                      ((d16 == 0.0) or (d1 and d16 <= d1 / 10.0))
    d["collapse_pattern"] = collapse
    d["delta_star_none"] = {c: {H: dstar(c, "none", H) for H in H_READOUTS}
                            for c in cells}

    # rule 1: FD_RESOLUTION_LIMITED
    n_unres = 0
    for c in early:
        for H in (16, 32):
            n_res = next(r["n_resolvable"] for r in trust_rows
                         if r["cell"] == c and r["intervention"] == "none"
                         and r["H"] == H and r["branch"] == "C")
            if n_res == 0:
                n_unres += 1
    d["unresolvable_early_Hge16"] = n_unres
    if n_unres >= 2:
        return "FD_RESOLUTION_LIMITED", d

    # re-expansion: v-floor intervention re-opens >=10x baseline at collapse H
    reexp = {}
    for c in early:
        if not collapse[c]:
            reexp[c] = False
            continue
        hs = [H for H in H_READOUTS
              if (dstar(c, "none", H) == 0.0) or
                 (dstar(c, "none", 1) and dstar(c, "none", H) is not None
                  and dstar(c, "none", H) <= dstar(c, "none", 1) / 10.0)]
        got = False
        for H in hs:
            base = dstar(c, "none", H)
            for iv in VFLOOR_IVS:
                ivd = dstar(c, iv, H)
                if ivd is None:
                    continue
                if base == 0.0:
                    if ivd >= 1e-3:  # re-opened to a meaningful radius
                        got = True
                elif ivd >= 10.0 * base:
                    got = True
            if got:
                break
        reexp[c] = got
    d["reexpansion"] = reexp
    d["reexp_H_basis"] = "collapse H set per EARLY cell (all H meeting the collapse inequality)"

    collapse_holds = all(collapse[c] for c in early)
    collapse_holds_lenient = any(collapse[c] for c in early)
    d["collapse_reading_strict_both_early"] = collapse_holds
    d["collapse_reading_lenient_any_early"] = collapse_holds_lenient

    # middle contrast: prereg rule 2 requires MIDDLE no collapse OR much weaker
    # (delta*_MIDDLE(16) > delta*_EARLY(16)).
    d_middle = dstar(middle, "none", 16)
    d_early16 = [dstar(c, "none", 16) for c in early]
    middle_no_collapse = not collapse[middle]
    middle_much_weaker = (d_middle is not None) and all(
        (e is not None and d_middle > e) for e in d_early16)
    middle_ok = middle_no_collapse or middle_much_weaker
    d["middle_delta_star_16"] = d_middle
    d["middle_no_collapse"] = middle_no_collapse
    d["middle_much_weaker"] = middle_much_weaker
    d["middle_contrast_ok"] = middle_ok

    if collapse_holds and any(reexp.values()) and middle_ok:
        return "TRUST_REGION_COLLAPSE_CAUSALLY_LINKED", d
    if collapse_holds_lenient and not collapse_holds:
        # strict reading failed; fall through to correlated/others on strict
        pass
    if collapse_holds and not any(reexp.values()):
        return "TRUST_REGION_COLLAPSE_CORRELATED_ONLY", d

    # rule 4: TANGENT_CORRECT_BUT_FINITE_RESPONSE_NONLINEAR
    # need h_H^FD: smallest-resolvable-delta D_H and |h-hFD|/|h| <= 0.2 while
    # some R_H(delta) > 0.2 at delta >= 1e-2 (baseline C arm).
    fd_tab = {}
    for r in fd_rows:
        fd_tab.setdefault((r["cell"], r["H"]), []).append(r)
    tan_corr = False
    for c in cells:
        for H in H_READOUTS:
            ht = next((r["h_tangent"] for r in trust_rows
                       if r["cell"] == c and r["intervention"] == "none"
                       and r["H"] == H and r["branch"] == "C"), None)
            if ht is None:
                continue
            rows = sorted(fd_tab.get((c, H), []), key=lambda r: r["delta"])
            resol = [r for r in rows if r["resolvable"]]
            if not resol:
                continue
            hfd = resol[0]["D_H"]
            cond = (abs(ht - hfd) / (abs(ht) + 1e-300) <= 0.2)
            big_r = False
            det = raws[tuple(int(x) for x in c.split(":"))]
            for r in rows:
                if r["delta"] >= 1e-2:
                    D = r["D_H"]
                    # R vs FD-based prediction equivalent: check usable failure
                    pred = ht * r["delta"]
                    Rp = abs(r["Delta_plus"] - pred) / (abs(pred) + EPS_METRIC)
                    if Rp > R_THR:
                        big_r = True
            if cond and big_r:
                tan_corr = True
    d["tangent_correct_but_nonlinear_evidence"] = tan_corr
    if tan_corr:
        return "TANGENT_CORRECT_BUT_FINITE_RESPONSE_NONLINEAR", d

    # rule 5: NO_TRUST_REGION_COLLAPSE
    n_ok = 0
    for c in cells:
        vals = [dstar(c, "none", H) for H in H_READOUTS]
        if all(v is not None and v >= 2e-2 for v in vals):
            n_ok += 1
    d["cells_all_H_delta_star_ge_2e-2"] = n_ok
    if n_ok >= 2:
        return "NO_TRUST_REGION_COLLAPSE", d

    return "UNRESOLVED", d


if __name__ == "__main__":
    sys.exit(main())
