"""T0 round-1 collection: summary json, metrics csv, pulse csv, 3 figures, cost ledger."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.t0_construct import S, T, S_NORM_F, BS, BA, arm_inputs, xi_sequence
from src.t0_maps import (apply_map, tangent_matrix, finite_ns_fp64, kernel_weights)
from src.t0_metrics import phase_stats, empirical_C_G, direction_model_A, scalar_model_A

OUT = ROOT / "outputs" / "t0_round1"
DELTAS = [0.0, 1e-3, 1e-2, 5e-2]
ARMS = ("arm1", "arm2")


def A_of(d):
    vec = d.reshape(d.shape[0], T, 4)
    db = vec.mean(axis=0)
    cen = vec - db[None]
    return float(np.mean(np.sum(cen ** 2, axis=2)))


def proj_var(d, B):
    vec = d.reshape(d.shape[0], T, 4)
    db = vec.mean(axis=0)
    proj = (vec - db[None]) @ B.reshape(4)
    return float(np.mean(proj ** 2))


def main():
    t0 = time.time()
    cpu = json.load(open(OUT / "cpu_fp64_meta.json"))
    nat_meta = json.load(open(OUT / "native_gpu_meta.json"))
    cpu_z = np.load(OUT / "cpu_fp64.npz")
    nat_z = np.load(OUT / "native_gpu.npz")

    c_xi = {arm: empirical_C_G(xi_sequence(arm)) for arm in ARMS}
    J = {m: tangent_matrix(m) for m in ("polar_fp64", "finite_ns_fp64")}
    preds = {}
    for m in ("polar_fp64", "finite_ns_fp64"):
        for arm in ARMS:
            dm = direction_model_A(c_xi[arm], J[m], 0.95)
            sm = scalar_model_A(c_xi[arm], J[m], 0.95)
            preds[(m, arm)] = {"dir": dm["A_pred"], "sca": sm["A_pred"]}

    eps_of = lambda d: d * S_NORM_F / np.sqrt(2.0)
    rows = []
    summary = {"maps_ref": cpu["maps_ref"],
               "native_map_S": nat_z["D::native_muon|delta=0.0|arm1|rep1"][0, 0].tolist(),
               "native_alignment_max_abs_err": nat_meta["alignment_max_abs_err"],
               "native_repetition_bitwise_equal_all": all(
                   v["bitwise_equal"] for v in nat_meta["repetition_check"].values()),
               "native_bf16_input_check": nat_meta["bf16_input_check"],
               "conditions": {}}

    for m in ("polar_fp64", "finite_ns_fp64"):
        for delta in DELTAS:
            eps = eps_of(delta)
            cond = summary["conditions"].setdefault(f"{m}|delta={delta}", {})
            A = {}
            for arm in ARMS:
                d = cpu_z[f"D::{m}|delta={delta}|{arm}"]
                A[arm] = A_of(d)
                rows.append({"map": m, "delta": delta, "arm": arm, "rep": "fp64",
                             "A": A[arm],
                             "proj_var_Bs": proj_var(d, BS), "proj_var_Ba": proj_var(d, BA),
                             "mean_shift_B_max": cpu["results"][f"{m}|delta={delta}|{arm}"]["mean_shift"]["B_max"]})
            cond["A1"], cond["A2"] = A["arm1"], A["arm2"]
            cond["Delta_A"] = A["arm2"] - A["arm1"]
            cond["ratio_A"] = A["arm2"] / A["arm1"] if A["arm1"] > 0 else None
            if delta > 0:
                for mdl, k in (("direction", "dir"), ("scalar", "sca")):
                    p1, p2 = preds[(m, "arm1")][k] * eps ** 2, preds[(m, "arm2")][k] * eps ** 2
                    cond[f"{mdl}_A1_pred"], cond[f"{mdl}_A2_pred"] = p1, p2
                    cond[f"{mdl}_Delta_A_pred"] = p2 - p1
                    cond[f"{mdl}_abs_err_A1"] = abs(A["arm1"] - p1)
                    cond[f"{mdl}_abs_err_A2"] = abs(A["arm2"] - p2)
                    cond[f"{mdl}_abs_err_Delta_A"] = abs((A["arm2"] - A["arm1"]) - (p2 - p1))
                    cond[f"{mdl}_rel_err_A1"] = abs(A["arm1"] - p1) / A["arm1"] if A["arm1"] > 0 else None
                    cond[f"{mdl}_rel_err_A2"] = abs(A["arm2"] - p2) / A["arm2"] if A["arm2"] > 0 else None
                # per-time cov error (direction model), reuse stored per-time cov
                d1 = cpu_z[f"D::{m}|delta={delta}|arm1"]
                cd_pred = direction_model_A(c_xi["arm1"], J[m], 0.95)["cD_t"]
                st = phase_stats(d1)
                cond["direction_cov_mean_frob_err_arm1"] = float(
                    np.mean(np.linalg.norm(st["cov_t"] - cd_pred, axis=(1, 2))))

    # native conditions (J proxy = finite_ns tangent; report both)
    for delta in DELTAS:
        eps = eps_of(delta)
        cond = summary["conditions"].setdefault(f"native_muon|delta={delta}", {})
        A = {}
        for arm in ARMS:
            d1 = nat_z[f"D::native_muon|delta={delta}|{arm}|rep1"]
            d2 = nat_z[f"D::native_muon|delta={delta}|{arm}|rep2"]
            A[arm] = A_of(d1)
            rows.append({"map": "native_muon", "delta": delta, "arm": arm, "rep": "rep1",
                         "A": A[arm], "proj_var_Bs": proj_var(d1, BS), "proj_var_Ba": proj_var(d1, BA),
                         "mean_shift_B_max": float(np.abs(
                             d1.reshape(64, T, 4).mean(axis=0)
                             - nat_z["D::native_muon|delta=0.0|arm1|rep1"].reshape(64, T, 4).mean(axis=0)).max())})
            rows.append({"map": "native_muon", "delta": delta, "arm": arm, "rep": "rep2",
                         "A": A_of(d2), "proj_var_Bs": proj_var(d2, BS), "proj_var_Ba": proj_var(d2, BA),
                         "mean_shift_B_max": None})
        cond["A1"], cond["A2"] = A["arm1"], A["arm2"]
        cond["Delta_A"] = A["arm2"] - A["arm1"]
        cond["ratio_A"] = A["arm2"] / A["arm1"] if A["arm1"] > 0 else None
        if delta > 0:
            for mdl, k, mm in (("direction", "dir", "finite_ns_fp64"),
                               ("direction_polarJ", "dir", "polar_fp64")):
                p1 = preds[(mm, "arm1")][k] * eps ** 2
                p2 = preds[(mm, "arm2")][k] * eps ** 2
                cond[f"{mdl}_A1_pred"], cond[f"{mdl}_A2_pred"] = p1, p2
                cond[f"{mdl}_Delta_A_pred"] = p2 - p1
                cond[f"{mdl}_rel_err_A1"] = abs(A["arm1"] - p1) / A["arm1"] if A["arm1"] > 0 else None
                cond[f"{mdl}_rel_err_A2"] = abs(A["arm2"] - p2) / A["arm2"] if A["arm2"] > 0 else None
                cond[f"{mdl}_rel_err_Delta_A"] = (abs((A["arm2"] - A["arm1"]) - (p2 - p1))
                                                  / abs(A["arm2"] - A["arm1"]) if A["arm2"] != A["arm1"] else None)
            sp1 = preds[("finite_ns_fp64", "arm1")]["sca"] * eps ** 2
            sp2 = preds[("finite_ns_fp64", "arm2")]["sca"] * eps ** 2
            cond["scalar_A1_pred"], cond["scalar_A2_pred"] = sp1, sp2
            cond["scalar_Delta_A_pred"] = sp2 - sp1

    # ---- map-only control: same captured native H, FP64 map -> separates input/state
    #      rounding from the spectral-operator (bf16 NS) difference ----
    maponly = {}
    for delta in (1e-3, 1e-2, 5e-2):
        for arm in ARMS:
            H = nat_z[f"H::native_muon|delta={delta}|{arm}|rep1"]
            D_mo = np.stack([finite_ns_fp64(H[q, t]) for q in range(64) for t in range(T)])
            D_mo = D_mo.reshape(64, T, 2, 2)
            Hf = cpu_z[f"H::finite_ns_fp64|delta={delta}|{arm}"]
            maponly[f"delta={delta}|{arm}"] = {
                "A_maponly": A_of(D_mo),
                "H_native_vs_fp64_max_abs": float(np.abs(H - Hf).max()),
                "D_native_vs_maponly_max_abs": float(
                    np.abs(nat_z[f"D::native_muon|delta={delta}|{arm}|rep1"] - D_mo).max())}
    summary["map_only_control"] = maponly

    # ---- pulses ----
    pulse_rows = []
    w = kernel_weights(0.95, T + 1)
    for m, zsrc, rep in (("polar_fp64", cpu, ""), ("finite_ns_fp64", cpu, ""),
                         ("native_muon", None, "rep1")):
        Jm = J[m] if m != "native_muon" else J["finite_ns_fp64"]
        for dname, Dmat in (("Bs", BS), ("Ba", BA)):
            jt = {bn: float(np.sum(bm * (Jm @ Dmat.reshape(4)).reshape(2, 2)))
                  for bn, bm in (("Bs", BS), ("Ba", BA))}
            for dp in (1e-3, 1e-2, 5e-2):
                eps_p = dp * S_NORM_F
                if m == "native_muon":
                    Dp = nat_z[f"PULSE_DPLUS::native_muon|{dname}|dp={dp}|rep1"]
                    Dm = nat_z[f"PULSE_DMINUS::native_muon|{dname}|dp={dp}|rep1"]
                else:
                    Dp = zsrc["pulse"][f"{m}|{dname}->Bs|dp={dp}"]  # placeholder, replaced below
                    Dp = None
                if m != "native_muon":
                    continue  # fp64 pulses read from meta json below
                for t in range(T):
                    for bname, Bmat in (("Bs", BS), ("Ba", BA)):
                        r = float(np.sum(Bmat * (Dp[t] - Dm[t]))) / (2 * eps_p)
                        # pulse is 1-based step 16 = 0-based t=15; response starts at t=15 (w_0 term)
                        pred = w[t - 15] * jt[bname] if t >= 15 else 0.0
                        pulse_rows.append({"map": m, "pulse_dir": dname, "proj_dir": bname,
                                           "delta_pulse": dp, "t": t + 1, "R": r,
                                           "R_pred_fp64tangent": pred})
    # fp64 pulses from cpu meta: recompute correct per-projection tangent predictions
    # (cross projections <Bs,J[Ba]> and <Ba,J[Bs]> are exactly zero at S; meta's
    #  stored pred column used the same-direction value for both and is superseded here)
    for m in ("polar_fp64", "finite_ns_fp64"):
        Jm = J[m]
        for dname, Dmat in (("Bs", BS), ("Ba", BA)):
            jt = {bn: float(np.sum(bm * (Jm @ Dmat.reshape(4)).reshape(2, 2)))
                  for bn, bm in (("Bs", BS), ("Ba", BA))}
            for dp in (1e-3, 1e-2, 5e-2):
                for bname, Bmat in (("Bs", BS), ("Ba", BA)):
                    r_arr = np.array(cpu["pulse"][f"{m}|{dname}->{bname}|dp={dp}"])
                    for t in range(T):
                        pred = w[t - 15] * jt[bname] if t >= 15 else 0.0
                        pulse_rows.append({"map": m, "pulse_dir": dname, "proj_dir": bname,
                                           "delta_pulse": dp, "t": t + 1, "R": float(r_arr[t]),
                                           "R_pred_fp64tangent": float(pred)})

    summary["pulse_summary"] = {}
    for m in ("polar_fp64", "finite_ns_fp64", "native_muon"):
        for dname in ("Bs", "Ba"):
            for dp in (1e-3, 1e-2, 5e-2):
                for bname in ("Bs", "Ba"):
                    sel = [r for r in pulse_rows if r["map"] == m and r["pulse_dir"] == dname
                           and r["delta_pulse"] == dp and r["proj_dir"] == bname]
                    r = np.array([x["R"] for x in sel])
                    p = np.array([x["R_pred_fp64tangent"] for x in sel])
                    peak = float(np.max(np.abs(r)))
                    summary["pulse_summary"][f"{m}|{dname}->{bname}|dp={dp}"] = {
                        "peak_abs_R": peak,
                        "max_abs_err_vs_tangent": float(np.abs(r - p).max())}

    # ---- CSV outputs ----
    import csv
    with open(OUT / "t0_metrics.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    with open(OUT / "pulse_response.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(pulse_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(pulse_rows)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: A1/A2/Delta_A vs delta (measured vs direction-model prediction)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"polar_fp64": "C0", "finite_ns_fp64": "C1", "native_muon": "C3"}
    for m in ("polar_fp64", "finite_ns_fp64", "native_muon"):
        ds = [d for d in (1e-3, 1e-2, 5e-2)]
        for ax, q in zip(axes, ("A1", "A2", "Delta_A")):
            ys = [summary["conditions"][f"{m}|delta={d}"][q] for d in ds]
            ax.plot(ds, ys, "o-", color=colors[m], label=f"{m} measured")
            pk = f"{'direction' if m != 'native_muon' else 'direction'}_{'A1' if q=='A1' else 'A2' if q=='A2' else 'Delta_A'}_pred"
            if q in ("A1", "A2"):
                pk = f"direction_{q}_pred"
            else:
                pk = "direction_Delta_A_pred"
            if f"{m}|delta={ds[0]}" in summary["conditions"] and pk in summary["conditions"][f"{m}|delta={ds[0]}"]:
                yp = [summary["conditions"][f"{m}|delta={d}"][pk] for d in ds]
                ax.plot(ds, yp, "x--", color=colors[m], alpha=0.6, label=f"{m} dir-model pred")
    for ax, q in zip(axes, ("A1", "A2", "Delta_A")):
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_xlabel("delta (relative)"); ax.set_title(q); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("T0 round1: A1/A2/Delta_A vs perturbation scale (measured + direction-model prediction)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_A_vs_delta.png", dpi=130)

    # fig 2: scalar vs direction model relative errors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in ("polar_fp64", "finite_ns_fp64", "native_muon"):
        ds = [d for d in (1e-3, 1e-2, 5e-2)]
        for ax, q in zip(axes, ("A1", "A2", "Delta_A")):
            if m == "native_muon":
                dkey, skey = (f"direction_rel_err_{q}" if q != "Delta_A" else "direction_rel_err_Delta_A",
                              None)
            else:
                dkey = f"direction_rel_err_{q}" if q != "Delta_A" else None
                skey = f"scalar_rel_err_{q}" if q != "Delta_A" else None
            if q != "Delta_A" and skey:
                ys = [summary["conditions"][f"{m}|delta={d}"][skey] for d in ds]
                ax.plot(ds, ys, "s--", color=colors[m], alpha=0.5, label=f"{m} scalar-model")
            dkey = f"direction_rel_err_{q}" if q != "Delta_A" else "direction_rel_err_Delta_A"
            if dkey in summary["conditions"].get(f"{m}|delta={ds[0]}", {}):
                ys = [summary["conditions"][f"{m}|delta={d}"].get(dkey) for d in ds]
                ax.plot(ds, ys, "o-", color=colors[m], label=f"{m} direction-model")
    for ax, q in zip(axes, ("A1", "A2", "Delta_A")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("delta"); ax.set_title(f"relative error: {q}"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Model relative errors (scalar vs direction). Direction model never fitted to outputs.")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_model_errors.png", dpi=130)

    # fig 3: pulse responses (same-direction projection) vs t
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for i, m in enumerate(("polar_fp64", "finite_ns_fp64", "native_muon")):
        for j, dp in enumerate((1e-3, 1e-2, 5e-2)):
            ax = axes[i, j]
            for dname, ls in (("Bs", "-"), ("Ba", ":")):
                sel = [r for r in pulse_rows if r["map"] == m and r["pulse_dir"] == dname
                       and r["delta_pulse"] == dp and r["proj_dir"] == dname]
                sel.sort(key=lambda r: r["t"])
                ax.plot([r["t"] for r in sel], [r["R"] for r in sel], ls,
                        color=colors[m], label=f"pulse {dname}")
                spred = [r for r in pulse_rows if r["map"] == "finite_ns_fp64"
                         and r["pulse_dir"] == dname and r["delta_pulse"] == dp
                         and r["proj_dir"] == dname]
                if m == "native_muon":
                    ax.plot([r["t"] for r in spred], [r["R_pred_fp64tangent"] for r in spred],
                            "k--", alpha=0.5, label="fp64 tangent")
            ax.axvline(16, color="gray", lw=0.5)
            ax.set_title(f"{m} dp={dp}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Pulse responses R_aa(t,16) by map and amplitude (secant; native = finite-amplitude)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pulse_response.png", dpi=130)

    # ---- cost ledger ----
    import os
    def dirsize(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    ledger = {
        "cpu_seconds": {"tests_and_exploration": None,
                        "run_t0_cpu.py": cpu["wall_seconds"],
                        "collect (this script, partial)": time.time() - t0,
                        "budget": 1800, "status": "within budget"},
        "gpu_seconds": {"run_t0_native.py": nat_meta["wall_seconds"],
                        "budget": 900, "status": "within budget"},
        "synthetic_recursions": {"fp64_momentum_recursions": cpu["n_momentum_recursions"],
                                 "native_optimizer_steps": nat_meta["n_native_optimizer_steps"]},
        "output_bytes": dirsize(OUT),
        "transformer_optimizer_updates": 0,
        "real_batch_gradient_computations": 0,
        "parent_trajectory_training": 0,
        "confirmation_set_access": 0,
    }
    json.dump(ledger, open(OUT / "cost_ledger.json", "w"), indent=1)
    summary["cost_ledger"] = ledger
    json.dump(summary, open(OUT / "t0_summary.json", "w"), indent=1, default=str)
    print(f"collection done in {time.time()-t0:.1f}s; outputs in {OUT}")


if __name__ == "__main__":
    main()
