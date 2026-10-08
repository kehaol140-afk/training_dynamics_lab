"""T0 round-1 CPU runner: FP64 main matrix (polar, finite_ns), predictions, pulses.

Outputs outputs/t0_round1/cpu_fp64.npz + cpu_fp64_meta.json. Counts synthetic recursions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.t0_construct import S, T, S_NORM_F, arm_inputs, xi_sequence, direction_lag_covariance
from src.t0_maps import apply_map, tangent_matrix, nesterov_H_direct, kernel_weights
from src.t0_metrics import (phase_stats, empirical_C_G, direction_model_A,
                            scalar_model_A, mean_shift_stats)

DELTAS = [0.0, 1e-3, 1e-2, 5e-2]
MAPS = ["polar_fp64", "finite_ns_fp64"]


def main():
    t0 = time.time()
    out_dir = ROOT / "outputs" / "t0_round1"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_recur = 0  # synthetic momentum recursions counter
    results = {}
    arrays = {}
    diag = {"maps_ref": {m: apply_map(m, S).tolist() for m in MAPS},
            "J": {m: tangent_matrix(m).tolist() for m in MAPS}}

    # phase-enumerated direction process covariances (independent of delta: symbols fixed)
    c_xi = {arm: empirical_C_G(xi_sequence(arm)) for arm in ("arm1", "arm2")}
    # empirical check: C_G from phase stats at a small delta should match c_xi*eps^2

    for map_name in MAPS:
        J = tangent_matrix(map_name)
        map_ref = apply_map(map_name, S)
        for delta in DELTAS:
            eps = delta * S_NORM_F / np.sqrt(2.0)
            for arm in ("arm1", "arm2"):
                D = np.empty((64, T, 2, 2))
                H = np.empty((64, T, 2, 2))
                for q in range(64):
                    g = arm_inputs(arm, q, eps)
                    h = nesterov_H_direct(g, S)
                    n_recur += T
                    H[q] = h
                    for t in range(T):
                        D[q, t] = apply_map(map_name, h[t])
                key = f"{map_name}|delta={delta}|{arm}"
                results[key] = {"A": phase_stats(D)["A"],
                                "H_stats": phase_stats(H),
                                "mean_shift": mean_shift_stats(D, map_ref)}
                arrays[f"D::{key}"] = D
                arrays[f"H::{key}"] = H
            # arm-level aggregates
            a1 = results[f"{map_name}|delta={delta}|arm1"]["A"]
            a2 = results[f"{map_name}|delta={delta}|arm2"]["A"]
            results[f"{map_name}|delta={delta}|__agg__"] = {
                "A1": a1, "A2": a2, "Delta_A": a2 - a1,
                "ratio_A": (a2 / a1) if a1 > 0 else None}
            # predictions (same J for both arms; only input cov differs)
            if delta > 0:
                for arm in ("arm1", "arm2"):
                    dm = direction_model_A(c_xi[arm], J, beta=0.95)
                    sm = scalar_model_A(c_xi[arm], J, beta=0.95)
                    results[f"{map_name}|delta={delta}|{arm}"]["pred"] = {
                        "direction_A_pred_over_eps2": dm["A_pred"],
                        "scalar_A_pred_over_eps2": sm["A_pred"]}
                d1 = results[f"{map_name}|delta={delta}|arm1"]["pred"]["direction_A_pred_over_eps2"]
                d2 = results[f"{map_name}|delta={delta}|arm2"]["pred"]["direction_A_pred_over_eps2"]
                s1 = results[f"{map_name}|delta={delta}|arm1"]["pred"]["scalar_A_pred_over_eps2"]
                s2 = results[f"{map_name}|delta={delta}|arm2"]["pred"]["scalar_A_pred_over_eps2"]
                results[f"{map_name}|delta={delta}|__agg__"]["pred"] = {
                    "direction_A1_over_eps2": d1, "direction_A2_over_eps2": d2,
                    "direction_Delta_A_over_eps2": d2 - d1,
                    "scalar_A1_over_eps2": s1, "scalar_A2_over_eps2": s2,
                    "scalar_Delta_A_over_eps2": s2 - s1}

    # ---- pulses (baseline G=S; pulse at step 16 along Bs/Ba; 3 amplitudes) ----
    pulse = {}
    PULSE_DELTAS = [1e-3, 1e-2, 5e-2]
    from src.t0_construct import BS, BA
    pulse_step = 16
    for map_name in MAPS:
        J = tangent_matrix(map_name)
        w = kernel_weights(0.95, T + 1)
        for dname, Dmat in (("Bs", BS), ("Ba", BA)):
            for dp in PULSE_DELTAS:
                eps_p = dp * S_NORM_F
                Dp, Dm = np.empty((T, 2, 2)), np.empty((T, 2, 2))
                for sign, out in ((+1.0, Dp), (-1.0, Dm)):
                    g = np.repeat(S[None], T, axis=0)
                    g[pulse_step - 1] = S + sign * eps_p * Dmat
                    h = nesterov_H_direct(g, S)
                    n_recur += T
                    for t in range(T):
                        out[t] = apply_map(map_name, h[t])
                for bname, Bmat in (("Bs", BS), ("Ba", BA)):
                    r = np.array([float(np.sum(Bmat * (Dp[t] - Dm[t]))) / (2 * eps_p)
                                  for t in range(T)])
                    pulse[f"{map_name}|{dname}->{bname}|dp={dp}"] = r
                    arrays[f"PULSE::{map_name}|{dname}->{bname}|dp={dp}"] = r
                # full matrix responses for the same-direction projections
                arrays[f"PULSE_DPLUS::{map_name}|{dname}|dp={dp}"] = Dp
                arrays[f"PULSE_DMINUS::{map_name}|{dname}|dp={dp}"] = Dm
                # tangent prediction: R_b(t) = w[t-16] * <Bb, J[Dmat]> for t>=16
                jt = {bn: float(np.sum(bm * (J @ Dmat.reshape(4)).reshape(2, 2)))
                      for bn, bm in (("Bs", BS), ("Ba", BA))}
                pred = np.zeros(T)
                for t in range(pulse_step, T + 1):
                    pred[t - 1] = w[t - pulse_step] * jt[dname]
                pulse[f"{map_name}|{dname}->PRED|dp={dp}"] = pred

    np.savez_compressed(out_dir / "cpu_fp64.npz", **arrays)
    meta = {"config": "configs/t0_round1.json",
            "n_momentum_recursions": n_recur,
            "wall_seconds": time.time() - t0,
            "results": results,
            "pulse": {k: v.tolist() for k, v in pulse.items()},
            **diag}
    (out_dir / "cpu_fp64_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"CPU FP64 done in {meta['wall_seconds']:.1f}s, recursions={n_recur}")


if __name__ == "__main__":
    main()
