"""T0 round-1 native runner: installed torch.optim.Muon on one 2x2 fp32 param, CUDA.

Production path: p.grad fp32 -> buf lerp fp32 -> H = grad.lerp(buf, m) fp32 ->
_zeropower_via_newtonschulz(H) internally bf16 -> D captured pre-lr (shape factor 1 for 2x2).
Alignment check: full Muon.step() with lr=1, wd=0 must give delta_param == -D.
Main matrix run twice (repetition for numerical reproducibility, NOT two scientific samples).
Counts wall/GPU seconds against the 15-minute GPU budget.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.t0_construct import S, T, S_NORM_F, arm_inputs, BS, BA

from torch.optim import Muon
from torch.optim._muon import _zeropower_via_newtonschulz

COEFFS = (3.4445, -4.775, 2.0315)
EPS_NORM = 1e-7
NS_STEPS = 5
MOM = 0.95
DELTAS = [0.0, 1e-3, 1e-2, 5e-2]
PULSE_DELTAS = [1e-3, 1e-2, 5e-2]
S_T = torch.tensor(S, dtype=torch.float32, device="cuda")


def make_opt(param):
    return Muon([param], lr=1.0, weight_decay=0.0, momentum=MOM, nesterov=True,
                ns_steps=NS_STEPS, ns_coefficients=COEFFS, eps=EPS_NORM, adjust_lr_fn=None)


def reset_state(opt, param):
    opt.state[param] = {"momentum_buffer": S_T.clone()}


def run_arm(arm: str, phase: int, eps: float):
    """One (arm, phase) trajectory through the native optimizer. Returns H, D (T,2,2) fp32."""
    param = torch.zeros(2, 2, dtype=torch.float32, device="cuda")
    opt = make_opt(param)
    reset_state(opt, param)
    g_seq = arm_inputs(arm, phase, eps)  # fp64 host
    Hs, Ds = [], []
    for t in range(T):
        g = torch.tensor(g_seq[t], dtype=torch.float32, device="cuda")
        param.grad = g
        opt.step()
        buf = opt.state[param]["momentum_buffer"]
        h = g.lerp(buf, MOM)  # installed production line, fp32
        d = _zeropower_via_newtonschulz(h, COEFFS, NS_STEPS, EPS_NORM)
        Hs.append(h.double().cpu().numpy())
        Ds.append(d.float().double().cpu().numpy())
    return np.stack(Hs), np.stack(Ds)


def run_pulse(direction: np.ndarray, eps_p: float, sign: float):
    param = torch.zeros(2, 2, dtype=torch.float32, device="cuda")
    opt = make_opt(param)
    reset_state(opt, param)
    Hs, Ds = [], []
    dir_t = torch.tensor(direction, dtype=torch.float32, device="cuda")
    for t in range(T):
        g_t = S_T.clone()
        if t == 15:  # step 16, 1-based
            g_t = S_T + sign * eps_p * dir_t
        param.grad = g_t
        opt.step()
        buf = opt.state[param]["momentum_buffer"]
        h = g_t.lerp(buf, MOM)
        d = _zeropower_via_newtonschulz(h, COEFFS, NS_STEPS, EPS_NORM)
        Hs.append(h.double().cpu().numpy())
        Ds.append(d.float().double().cpu().numpy())
    return np.stack(Hs), np.stack(Ds)


def alignment_check():
    """Full step vs captured D: delta_param == -D (lr=1, wd=0, shape factor 1)."""
    rng = np.random.default_rng(11)
    max_err = 0.0
    for trial in range(8):
        param = torch.tensor(rng.standard_normal((2, 2)), dtype=torch.float32, device="cuda")
        opt = make_opt(param)
        if trial % 2 == 0:
            reset_state(opt, param)
        g = torch.tensor(rng.standard_normal((2, 2)), dtype=torch.float32, device="cuda")
        param.grad = g
        before = param.detach().clone()
        opt.step()
        buf = opt.state[param]["momentum_buffer"]
        h = g.lerp(buf, MOM)
        d = _zeropower_via_newtonschulz(h, COEFFS, NS_STEPS, EPS_NORM)
        delta = (param.detach() - before).double()
        max_err = max(max_err, float((delta + d.double()).abs().max()))
    return max_err


def bf16_quantization_check():
    """Does the bf16 NS input preserve the perturbation? Fraction of distinct elements."""
    info = {}
    for delta in (1e-3, 1e-2, 5e-2):
        eps = delta * S_NORM_F / np.sqrt(2.0)
        g1 = arm_inputs("arm1", 0, eps)
        base = arm_inputs("arm1", 0, 0.0)
        h_diff = 0
        tot = 0
        for t in range(T):
            hb = torch.tensor(base[t], dtype=torch.float32, device="cuda")
            hg = torch.tensor(g1[t], dtype=torch.float32, device="cuda")
            tot += 4
            h_diff += int((hb.bfloat16() != hg.bfloat16()).sum().item())
        info[f"delta={delta}"] = {"bf16_distinct_fraction": h_diff / tot}
    return info


def main():
    t_start = time.time()
    torch.manual_seed(0)
    out_dir = ROOT / "outputs" / "t0_round1"
    arrays, results = {}, {}

    align_err = alignment_check()
    results["alignment_max_abs_err"] = align_err
    print(f"alignment |delta_param + D|max = {align_err:.3e}", flush=True)

    results["bf16_input_check"] = bf16_quantization_check()
    print("bf16 input check:", results["bf16_input_check"], flush=True)

    n_steps_total = 0
    # main matrix, two repetitions
    for rep in (1, 2):
        for delta in DELTAS:
            eps = delta * S_NORM_F / np.sqrt(2.0)
            for arm in ("arm1", "arm2"):
                D = np.empty((64, T, 2, 2))
                H = np.empty((64, T, 2, 2))
                for q in range(64):
                    h, d = run_arm(arm, q, eps)
                    H[q], D[q] = h, d
                    n_steps_total += T
                arrays[f"D::native_muon|delta={delta}|{arm}|rep{rep}"] = D
                arrays[f"H::native_muon|delta={delta}|{arm}|rep{rep}"] = H
                print(f"rep{rep} delta={delta} {arm} done ({time.time()-t_start:.0f}s)", flush=True)

    # pulses: one pass + repetition
    for rep in (1, 2):
        for dname, Dmat in (("Bs", BS), ("Ba", BA)):
            for dp in PULSE_DELTAS:
                eps_p = dp * S_NORM_F
                _, Dp = run_pulse(Dmat, eps_p, +1.0)
                _, Dm = run_pulse(Dmat, eps_p, -1.0)
                n_steps_total += 2 * T
                arrays[f"PULSE_DPLUS::native_muon|{dname}|dp={dp}|rep{rep}"] = Dp
                arrays[f"PULSE_DMINUS::native_muon|{dname}|dp={dp}|rep{rep}"] = Dm

    np.savez_compressed(out_dir / "native_gpu.npz", **arrays)

    # repetition consistency
    rep_check = {}
    for delta in DELTAS:
        for arm in ("arm1", "arm2"):
            d1 = arrays[f"D::native_muon|delta={delta}|{arm}|rep1"]
            d2 = arrays[f"D::native_muon|delta={delta}|{arm}|rep2"]
            rep_check[f"delta={delta}|{arm}"] = {
                "max_abs_diff": float(np.abs(d1 - d2).max()),
                "bitwise_equal": bool(np.array_equal(d1, d2))}
    for dname in ("Bs", "Ba"):
        for dp in PULSE_DELTAS:
            p1 = arrays[f"PULSE_DPLUS::native_muon|{dname}|dp={dp}|rep1"]
            p2 = arrays[f"PULSE_DPLUS::native_muon|{dname}|dp={dp}|rep2"]
            rep_check[f"pulse|{dname}|dp={dp}"] = {
                "max_abs_diff": float(np.abs(p1 - p2).max()),
                "bitwise_equal": bool(np.array_equal(p1, p2))}
    results["repetition_check"] = rep_check

    results.update({
        "n_native_optimizer_steps": n_steps_total,
        "device": torch.cuda.get_device_name(0),
        "wall_seconds": time.time() - t_start,
    })
    (out_dir / "native_gpu_meta.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"native done in {results['wall_seconds']:.1f}s, steps={n_steps_total}", flush=True)


if __name__ == "__main__":
    main()
