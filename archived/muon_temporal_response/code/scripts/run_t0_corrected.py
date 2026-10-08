"""Round-1 T0 corrected post-processing (erratum stage A).

Recomputes all statistics from the ORIGINAL round-1 arrays (cpu_fp64.npz / native_gpu.npz)
with the corrected conventions:
  - population covariance denominator n_phase (not n_phase-1)
  - absolute prediction covariances include epsilon_perturb**2 (unit-coefficient fields `_over_eps2`)
  - B2 = mean_t ||mean_phase(D)-D0||^2, total_mse = A + B2
  - per-arm (A1,A2) errors separate from Delta_A error and sign
  - BF16 quantization checked on the captured H that actually entered NS (not G)
  - same-H numeric-error decomposition A_native = A_ref + A_q + 2*cross + mean_q
  - pulse absolute AND relative errors per channel (input dir -> projection dir)
No GPU re-run; original outputs are never overwritten.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.t0_construct import S, T, S_NORM_F, BS, BA, xi_sequence
from src.t0_maps import finite_ns_fp64, tangent_matrix, kernel_weights
from src.t0_metrics import (phase_stats, bias_decomposition,
                            same_H_error_decomposition, empirical_C_G,
                            direction_model_A, scalar_model_A)

OUT = ROOT / "outputs" / "t0_round1_corrected"
DELTAS = [0.0, 1e-3, 1e-2, 5e-2]
ARMS = ("arm1", "arm2")
MAPS_FP64 = ("polar_fp64", "finite_ns_fp64")


def apply_fp64(map_name, h):
    if map_name == "finite_ns_fp64":
        return finite_ns_fp64(h)
    u, _, vt = np.linalg.svd(h)
    return u @ vt


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "outputs" / "t0_round1" / "cpu_fp64.npz")
    zn = np.load(ROOT / "outputs" / "t0_round1" / "native_gpu.npz")

    eps_of = lambda d: d * S_NORM_F / np.sqrt(2.0)
    c_xi = {arm: empirical_C_G(xi_sequence(arm)) for arm in ARMS}
    J = {m: tangent_matrix(m) for m in MAPS_FP64}

    rows, summary = [], {"conditions": {}, "conventions": {
        "cov_denominator": "n_phase (population)",
        "absolute_pred_rule": "unit-coefficient prediction (fields *_over_eps2) times epsilon_perturb**2",
        "B2_def": "mean_t ||mean_phase(D_t) - D0||_F^2, D0 = same-map zero-perturbation baseline",
        "identity": "total_mse == A + B2 (exact)",
        "source_arrays": ["outputs/t0_round1/cpu_fp64.npz", "outputs/t0_round1/native_gpu.npz"],
        "note": "original outputs untouched; see docs/ROUND1_ERRATUM.md"}}

    # baseline D0 per map/path (delta=0 arrays, phase- and time-invariant up to float dust)
    D0 = {"finite_ns_fp64": z["D::finite_ns_fp64|delta=0.0|arm1"].reshape(64, T, 4).mean(axis=0).mean(axis=0),
          "polar_fp64": np.eye(2).reshape(4),
          "native_muon": zn["D::native_muon|delta=0.0|arm1|rep1"].reshape(64, T, 4).mean(axis=0).mean(axis=0)}

    for map_name in MAPS_FP64:
        for delta in DELTAS:
            eps = eps_of(delta)
            cond = summary["conditions"].setdefault(f"{map_name}|delta={delta}", {})
            A = {}
            for arm in ARMS:
                d = z[f"D::{map_name}|delta={delta}|{arm}"]
                st = phase_stats(d)
                bd = bias_decomposition(d, D0[map_name].reshape(2, 2))
                A[arm] = st["A"]
                assert abs(st["A"] - bd["A"]) < 1e-18
                rows.append({"map": map_name, "delta": delta, "arm": arm,
                             "A_population": st["A"], "B2": bd["B2"],
                             "total_mse": bd["total_mse"], "B1_mean_norm": bd["B1_mean_norm"],
                             "B2_over_A": bd["B2"] / st["A"] if st["A"] > 0 else None})
            cond.update({"A1": A["arm1"], "A2": A["arm2"],
                         "Delta_A": A["arm2"] - A["arm1"],
                         "ratio_A": A["arm2"] / A["arm1"] if A["arm1"] > 0 else None,
                         "B2_arm1": rows[-2]["B2"], "B2_arm2": rows[-1]["B2"]})
            if delta > 0:
                for mdl, key in (("direction", "dir"), ("scalar", "sca")):
                    for arm in ARMS:
                        coef = (direction_model_A(c_xi[arm], J[map_name], 0.95) if key == "dir"
                                else scalar_model_A(c_xi[arm], J[map_name], 0.95))
                        # absolute predicted A and per-time absolute cov
                        abs_a = coef["A_pred"] * eps ** 2
                        cD_abs = coef["cD_t"] * eps ** 2
                        trace_pred = float(np.mean(np.trace(cD_abs, axis1=1, axis2=2)))
                        assert abs(abs_a - trace_pred) < 1e-18 * max(1.0, abs(abs_a)) + 1e-20
                        cond[f"{mdl}_{arm}_A_pred_absolute"] = abs_a
                        cond[f"{mdl}_{arm}_A_pred_coef_over_eps2"] = coef["A_pred"]
                        if A[arm] > 0:
                            cond[f"{mdl}_{arm}_rel_err"] = abs(A[arm] - abs_a) / A[arm]
                        # corrected per-time covariance error (population, absolute)
                        st_arm = phase_stats(z[f"D::{map_name}|delta={delta}|{arm}"])
                        cond[f"{mdl}_{arm}_cov_mean_frob_abs_err"] = float(
                            np.mean(np.linalg.norm(st_arm["cov_t"] - cD_abs, axis=(1, 2))))
                dmeas = A["arm2"] - A["arm1"]
                for mdl in ("direction", "scalar"):
                    dp = (cond[f"{mdl}_arm2_A_pred_absolute"] - cond[f"{mdl}_arm1_A_pred_absolute"])
                    cond[f"{mdl}_Delta_A_pred_absolute"] = dp
                    if dmeas != 0:
                        cond[f"{mdl}_Delta_A_rel_err"] = abs(dmeas - dp) / abs(dmeas)
                        cond[f"{mdl}_Delta_A_sign_match"] = bool(np.sign(dp) == np.sign(dmeas))
                cond["Delta_A_sign_arm2_gt_arm1"] = bool(dmeas > 0)

    # native conditions
    for delta in DELTAS:
        eps = eps_of(delta)
        cond = summary["conditions"].setdefault(f"native_muon|delta={delta}", {})
        A = {}
        for arm in ARMS:
            d = zn[f"D::native_muon|delta={delta}|{arm}|rep1"]
            st = phase_stats(d)
            bd = bias_decomposition(d, D0["native_muon"].reshape(2, 2))
            A[arm] = st["A"]
            rows.append({"map": "native_muon", "delta": delta, "arm": arm,
                         "A_population": st["A"], "B2": bd["B2"], "total_mse": bd["total_mse"],
                         "B1_mean_norm": bd["B1_mean_norm"],
                         "B2_over_A": bd["B2"] / st["A"] if st["A"] > 0 else None})
        cond.update({"A1": A["arm1"], "A2": A["arm2"], "Delta_A": A["arm2"] - A["arm1"],
                     "ratio_A": A["arm2"] / A["arm1"] if A["arm1"] > 0 else None,
                     "B2_arm1": rows[-2]["B2"], "B2_arm2": rows[-1]["B2"]})
        if delta > 0:
            for arm in ARMS:
                coef = direction_model_A(c_xi[arm], J["finite_ns_fp64"], 0.95)
                cond[f"direction_{arm}_A_pred_absolute"] = coef["A_pred"] * eps ** 2
                cond[f"direction_{arm}_A_pred_coef_over_eps2"] = coef["A_pred"]
                if A[arm] > 0:
                    cond[f"direction_{arm}_rel_err"] = abs(A[arm] - coef["A_pred"] * eps ** 2) / A[arm]
            dmeas = A["arm2"] - A["arm1"]
            dp = cond["direction_arm2_A_pred_absolute"] - cond["direction_arm1_A_pred_absolute"]
            cond["direction_Delta_A_pred_absolute"] = dp
            if dmeas != 0:
                cond["direction_Delta_A_rel_err"] = abs(dmeas - dp) / abs(dmeas)
                cond["direction_Delta_A_sign_match"] = bool(np.sign(dp) == np.sign(dmeas))

    # ---- same-H numeric-error decomposition (native vs FP64 map on captured H) ----
    decomp = {}
    for delta in (1e-3, 1e-2, 5e-2):
        for arm in ARMS:
            H = zn[f"H::native_muon|delta={delta}|{arm}|rep1"]
            Db = zn[f"D::native_muon|delta={delta}|{arm}|rep1"]
            Dr = np.stack([finite_ns_fp64(H[q, t]) for q in range(64) for t in range(T)]).reshape(64, T, 2, 2)
            dec = same_H_error_decomposition(Dr, Db)
            decomp[f"delta={delta}|{arm}"] = dec
            np.save(OUT / f"D_maponly64_{str(delta).replace('.','_')}_{arm}.npy", Dr)
    summary["same_H_decomposition"] = decomp

    # ---- BF16 quantization check on the REAL H that entered NS (not G) ----
    hq = {}
    for delta in (1e-3, 1e-2, 5e-2):
        for arm in ARMS:
            H = torch.tensor(zn[f"H::native_muon|delta={delta}|{arm}|rep1"], dtype=torch.float32)
            H0 = torch.tensor(zn["H::native_muon|delta=0.0|arm1|rep1"], dtype=torch.float32)
            hb, h0b = H.bfloat16(), H0.bfloat16()
            frac = float((hb != h0b).float().mean())
            # per-entry distinct diagonal levels of the (0,0)/(1,1) entries across (phase,time)
            d00 = sorted(set(float(x) for x in hb[..., 0, 0].flatten().tolist()))
            d11 = sorted(set(float(x) for x in hb[..., 1, 1].flatten().tolist()))
            hq[f"delta={delta}|{arm}"] = {
                "bf16_changed_fraction_vs_delta0": frac,
                "H00_bf16_distinct_levels": d00,
                "H11_bf16_distinct_levels": d11,
                "note": "checked on captured H (the tensor actually cast inside NS), supersedes the "
                        "round-1 G-based bf16_quantization_check"}
    summary["h_bf16_quantization"] = hq

    # ---- pulse errors: absolute AND relative per channel ----
    rows_pulse = []
    w = kernel_weights(0.95, T + 1)
    for m_, src_z, rep in (("polar_fp64", z, ""), ("finite_ns_fp64", z, ""),
                           ("native_muon", zn, "|rep1")):
        Jm = J["finite_ns_fp64"] if m_ == "native_muon" else J[m_]
        for dname, Dmat in (("Bs", BS), ("Ba", BA)):
            jt = {bn: float(np.sum(bm * (Jm @ Dmat.reshape(4)).reshape(2, 2)))
                  for bn, bm in (("Bs", BS), ("Ba", BA))}
            for dp in (1e-3, 1e-2, 5e-2):
                eps_p = dp * S_NORM_F
                if m_ == "native_muon":
                    Dp = zn[f"PULSE_DPLUS::native_muon|{dname}|dp={dp}|rep1"]
                    Dm_ = zn[f"PULSE_DMINUS::native_muon|{dname}|dp={dp}|rep1"]
                else:
                    Dp = np.stack([apply_fp64(m_, h) for h in _pulse_H(z, m_, dname, dp, +1.0)])
                    Dm_ = np.stack([apply_fp64(m_, h) for h in _pulse_H(z, m_, dname, dp, -1.0)])
                for bname, Bmat in (("Bs", BS), ("Ba", BA)):
                    r = np.array([float(np.sum(Bmat * (Dp[t] - Dm_[t]))) / (2 * eps_p) for t in range(T)])
                    pred = np.array([w[t - 15] * jt[bname] if t >= 15 else 0.0 for t in range(T)])
                    peak = float(np.abs(r).max())
                    abserr = float(np.abs(r - pred).max())
                    relerr = abserr / peak if peak > 0 else None
                    rows_pulse.append({"map": m_, "pulse_dir": dname, "proj_dir": bname,
                                       "delta_pulse": dp, "peak_abs_R": peak,
                                       "max_abs_err_vs_tangent": abserr,
                                       "rel_err_vs_peak": relerr,
                                       "zero_response": peak == 0.0})
    summary["pulse_corrected"] = {f"{m_}|{pd_}->{pj}|dp={dp_}":
                                  {k: v for k, v in r.items() if k not in ("map", "pulse_dir", "proj_dir", "delta_pulse")}
                                  for r in rows_pulse
                                  for m_, pd_, pj, dp_ in [(r["map"], r["pulse_dir"], r["proj_dir"], r["delta_pulse"])]}

    with open(OUT / "corrected_metrics.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    with open(OUT / "pulse_corrected.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_pulse[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows_pulse)
    summary["wall_seconds"] = time.time() - t0
    json.dump(summary, open(OUT / "corrected_summary.json", "w"), indent=1, default=str)
    print(f"corrected done in {summary['wall_seconds']:.1f}s -> {OUT}")


def _pulse_H(z, map_name, dname, dp, sign):
    """Rebuild the fp64 pulse-step H sequence (used only to recompute FP64 pulse outputs)."""
    from src.t0_maps import nesterov_H_direct
    eps_p = dp * S_NORM_F
    Dmat = BS if dname == "Bs" else BA
    g = np.repeat(S[None], T, axis=0)
    g[15] = S + sign * eps_p * Dmat
    return nesterov_H_direct(g, S)


if __name__ == "__main__":
    main()
