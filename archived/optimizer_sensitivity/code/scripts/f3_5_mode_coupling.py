#!/usr/bin/env python3
"""F3.5 unstable-mode coupling runner (frozen: docs/F3_5_UNSTABLE_MODE_COUPLING_PREREG.md).

Subcommands (each atomic + resumable via complete raw JSONs):
  traj <cell>      -- replay + forced-alpha1 & SO-q0 tangent propagation with
                      bitwise regression anchors vs committed F3 raws; snapshot
                      dz (t in {0,1,3,7,15,32}) and LR sources inh_t to runtime/
  adjoint <cell>   -- PHASE E gate at offset 0 (3 fixed probe pairs)
  eig <cell> <metric> -- PHASES F/G/H/I/J: per-offset right+left Arnoldi
                      (top-8), unstable subspaces, E_left/E_right/S_exc,
                      subspace rotation + propagated overlap, offset-0 causal
                      mode-projection probe. metric in {RAW, DIML}
  postprocess      -- raw JSONs -> 9 frozen CSVs + summary.json + verdict

Run with CUDA_VISIBLE_DEVICES set. Long jobs belong in tmux (driver script).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LAB = ROOT.parent
MM = LAB / "minimind_base"
if str(MM) not in sys.path:
    sys.path.insert(0, str(MM))

import torch  # noqa: E402

import f3_curvature_mechanism as f3m  # noqa: E402  (proven F3 pipeline)
from f2_hypergradient import (ACCUM, MAX_NORM, BASE_LR, TOTAL_MICRO,  # noqa: E402
                              CELL_STREAM, Rig)
from opsens import f0_bank_audit_v01 as f0  # noqa: E402
from opsens import f1_mechanism_v01 as fm  # noqa: E402
from opsens import f3_curvature_mechanism_v01 as f3  # noqa: E402
from opsens import f3_5_mode_coupling_v01 as f35  # noqa: E402
from opsens.a0c_adamw_response_propagation_v01 import clip_jvp  # noqa: E402
from opsens.batch_manifest import manifest_sha256  # noqa: E402

OUT = ROOT / "outputs" / "f3_5_mode_coupling"
RAW = OUT / "raw"
RUNTIME = OUT / "runtime"
SNAP_TS = (1, 3, 7, 15, 32)
FROZEN = ROOT / "outputs" / "f3_compatibility" / "frozen_reference" / "baseline_summary.json"
F3RAW = ROOT / "outputs" / "f3_curvature_mechanism" / "raw"
for d in (OUT, RAW, RUNTIME):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


try:
    _LIBC = ctypes.CDLL("libc.so.6")

    def malloc_trim() -> None:
        _LIBC.malloc_trim(0)
except Exception:  # pragma: no cover
    def malloc_trim() -> None:
        pass


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2 ** 20
    except Exception:
        return -1.0
    return -1.0


def phase_gc(tag: str) -> None:
    """Release glibc arena free-lists back to the OS (torch CPU workloads
    fragment mid-size blocks across threads; without this the eig job's RSS
    climbs ~85GB above live tensors -> kernel OOM)."""
    gc.collect()
    malloc_trim()
    log(f"mem[{tag}] rss={rss_gb():.1f}GB")


def save_raw(name: str, payload) -> None:
    p = RAW / name
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, p)


def load_raw(name: str):
    p = RAW / name
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def state_norms(dz):
    """Bit-identical to f3m.state_norms' (dtheta, dz) pair (dz norm computed
    as sqrt(n_th^2+n_m^2+n_v^2) so the F3 raw-anchor regression is bitwise)."""
    n_th = math.sqrt(sum(float(t.double().square().sum()) for t in dz["th"].values()))
    n_m = math.sqrt(sum(float(t.double().square().sum()) for t in dz["m"].values()))
    n_v = math.sqrt(sum(float(t.double().square().sum()) for t in dz["v"].values()))
    return n_th, math.sqrt(n_th ** 2 + n_m ** 2 + n_v ** 2)


def clone_state(dz):
    return {ch: {n: t.clone() for n, t in dz[ch].items()} for ch in ("th", "m", "v")}


# ---------------------------------------------------------------- setup ------
def setup_cell(cell: str, device: str):
    seed, step = (int(x) for x in cell.split(":"))
    f3m.rig = Rig(device)
    f3m.batches_g, stream_seed = f3m.rig.batches_for(CELL_STREAM[(seed, step)])
    f3m.state_g = f0.load_bank_state(fm.bank_state_path("g1b_100m", seed, step))
    f3m.micro0 = int(f3m.state_g["metadata"]["global_micro_step"])
    f3m.named, f3m.opt, f3m.model_cpu = f3m.rig.restore(f3m.state_g)
    msha = manifest_sha256(f3m.batches_g)
    log(f"f3.5 cell {seed}:{step} stream_seed={stream_seed} manifest={msha[:16]} "
        f"micro0={f3m.micro0}")
    return seed, step


def cell_dir(seed, step) -> Path:
    d = RUNTIME / f"{seed}_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_scales(seed, step):
    p = cell_dir(seed, step) / "scales.json"
    if p.exists():
        d = json.load(p.open())
        return {ch: {n: float(v) for n, v in d[ch].items()}
                for ch in ("th", "m", "v")}
    sc = f35.state_scales(f3m.named, f3m.opt)
    ser = {ch: {n: float(v) for n, v in sc[ch].items()} for ch in ("th", "m", "v")}
    with p.open("w") as fh:
        json.dump(ser, fh)
    return sc


# ------------------------------------------------- operator builder (F3-faithful)
def build_operator(off: int):
    """EXACT copy of f3_curvature_mechanism.run_rq4's per-offset construction
    (validated by the bitwise rho anchor regression in cmd_eig)."""
    named_o, opt_o, model_o = f3m.rig.restore(f3m.state_g)
    micro0 = f3m.micro0
    for h in range(1, off + 1):
        lr = f3m.rig.get_lr(micro0 + (h - 1) * ACCUM, TOTAL_MICRO, BASE_LR)
        opt_o.param_groups[0]["lr"] = lr
        grads, _ = f3m.rig.compute_grads(model_o, f3m.batches_g, h - 1)
        for n, p in named_o:
            p.grad = grads[n].clone()
        torch.nn.utils.clip_grad_norm_([p for _, p in named_o], MAX_NORM)
        opt_o.step()
        opt_o.zero_grad(set_to_none=True)
    lr = f3m.rig.get_lr(micro0 + off * ACCUM, TOTAL_MICRO, BASE_LR)
    opt_o.param_groups[0]["lr"] = lr
    grads, loss = f3m.rig.compute_grads(model_o, f3m.batches_g, off)
    for n, p in named_o:
        p.grad = grads[n].clone()
    torch.nn.utils.clip_grad_norm_([p for _, p in named_o], MAX_NORM)
    clipped = {n: p.grad.detach().clone() for n, p in named_o}
    cj = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=grads,
                  max_norm=MAX_NORM)
    coef, active, k = f3.clip_linear_coefficients(cj, MAX_NORM)

    def hvp_fn(x, _off=off, _model=model_o):
        out = f3m.rig.hvp(_model, f3m.batches_g, _off,
                          {n: t.to(torch.float32) for n, t in x.items()}, fp32=True)
        return {n: t.double() for n, t in out.items()}

    op = f3.StepOperator(named_o, opt_o, clipped, coef=coef, active=active,
                         k=k, hvp_fn=hvp_fn, enabled=True)
    return op, named_o, opt_o, model_o, float(loss)


# ---------------------------------------------------------------- traj -------
def cmd_traj(seed: int, step: int) -> int:
    cell = f"{seed}:{step}"
    out_name = f"traj_{seed}_{step}.json"
    if load_raw(out_name) is not None:
        log(f"traj {cell}: already complete")
        return 0
    cd = cell_dir(seed, step)
    ref_losses = json.load(FROZEN.open())["losses"][cell]
    forced = {ch: {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named}
              for ch in ("th", "m", "v")}
    impulse = None
    losses = []
    traj_rows = []
    for h in range(1, f3m.H_MAX + 1):
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = f3m.primal_grads(h - 1, f3m.micro0)
        dc_f, hvpn_f = f3m.dc_for(forced["th"], grads, 1.0, h)
        pend_f = f3.hom_step(f3m.named, f3m.opt, clipped, dc_f, forced)
        if impulse is not None:
            dc_i, _ = f3m.dc_for(impulse["th"], grads, 1.0, h)
            pend_i = f3.hom_step(f3m.named, f3m.opt, clipped, dc_i, impulse)
        theta_next = f3m.primal_advance()
        if h == 1:
            q0 = f3.lr_source_q0(f3m.named, theta_prev, theta_next)
            impulse = clone_state(q0)
            zero_dc = {n: torch.zeros_like(p, dtype=torch.float64)
                       for n, p in f3m.named}
            pend_i = f3.hom_step(f3m.named, f3m.opt, clipped, zero_dc, impulse)
            torch.save(q0, cd / "q0.pt")
        inh = {n: (theta_next[n].double() - theta_prev[n].double())
               for n, _ in f3m.named}
        forced = {"th": {n: pend_f["th"][n] + inh[n] for n, _ in f3m.named},
                  "m": pend_f["m"], "v": pend_f["v"]}
        impulse = pend_i
        losses.append(loss)
        torch.save({"th": inh}, cd / f"inh_t{h}.pt")
        if h in SNAP_TS:
            torch.save(forced, cd / f"forced_t{h}.pt")
            torch.save(impulse, cd / f"impulse_t{h}.pt")
        nf, zf = state_norms(forced)
        ni, zi = state_norms(impulse)
        traj_rows.append({"h": h, "loss": loss, "forced_dtheta": nf,
                          "impulse_dz": zi, "sec": round(time.time() - t0, 1)})
        log(f"traj {cell} h={h} loss={loss:.5f} |dth|F={nf:.4g} "
            f"|dz|I={zi:.4g} ({time.time() - t0:.0f}s)")
    # ---- hard gates (prereg §8) ----
    assert losses == list(ref_losses), f"traj {cell}: loss drift vs frozen ref"
    rq1 = json.load((F3RAW / f"rq1_{seed}_{step}.json").open())
    rq2 = json.load((F3RAW / f"rq2_{seed}_{step}.json").open())
    ref_f = rq1["steps"][-1]["arms"]["1.0"]["dtheta_norm"]
    ref_i = rq2["steps"][-1]["arms"]["a1.0|q0_lrsource"]["dz_norm"]
    nf, _ = state_norms(forced)
    _, zi = state_norms(impulse)
    assert nf == ref_f, f"traj {cell}: forced final {nf!r} != F3 rq1 {ref_f!r}"
    assert zi == ref_i, f"traj {cell}: impulse final {zi!r} != F3 rq2 {ref_i!r}"
    load_scales(seed, step)  # freeze scales file for eig jobs
    save_raw(out_name, {
        "cell": cell, "snaps": list(SNAP_TS),
        "regression": {"forced_final_dtheta": nf, "rq1_ref": ref_f,
                       "impulse_final_dz": zi, "rq2_ref": ref_i,
                       "losses_bitwise_vs_frozen": True},
        "steps": traj_rows})
    log(f"traj {cell}: COMPLETE (bitwise regression PASS)")
    return 0


# ------------------------------------------------------------- adjoint -------
def cmd_adjoint(seed: int, step: int) -> int:
    cell = f"{seed}:{step}"
    out_name = f"adjoint_{seed}_{step}.json"
    if load_raw(out_name) is not None:
        log(f"adjoint {cell}: already complete")
        return 0
    op, named_o, _, _, loss = build_operator(0)
    rows, ok = f35.adjoint_gate(op, named_o)
    save_raw(out_name, {"cell": cell, "offset": 0, "rows": rows,
                        "gate_pass": bool(ok), "tol": f35.F35_ADJOINT_TOL})
    discs = " ".join(f"{r['disc']:.2e}" for r in rows)
    log(f"adjoint {cell}: gate_pass={ok} discs=[{discs}]")
    return 0


# ---------------------------------------------------------------- eig --------
def _propagate_homog(arms, h_from: int, h_to: int):
    """Propagate homogeneous (source-off, alpha=1) tangent states through the
    TRUE J sequence along the real trajectory. Resets the shared primal to the
    bank state, fast-forwards (cheap) to h_from, then replays steps
    h_from+1..h_to with one hom_step per arm per step (dc from each arm's own
    dth; NO source term) -- identical matvec sequence to StepOperator.jvp."""
    f3m.named, f3m.opt, f3m.model_cpu = f3m.rig.restore(f3m.state_g)
    if h_from > 0:
        f3m.fast_forward_primal(h_from, f3m.micro0)
    for h in range(h_from + 1, h_to + 1):
        _, _, grads, clipped, _, _ = f3m.primal_grads(h - 1, f3m.micro0)
        pend = []
        for a in arms:
            dc, _ = f3m.dc_for(a["th"], grads, 1.0, h)
            pend.append(f3.hom_step(f3m.named, f3m.opt, clipped, dc, a))
        f3m.primal_advance()
        arms = pend
    return arms


def _share_top(shares):
    return max(shares, key=shares.get) if shares else ""


def _pairs_rows(pairs, named):
    rows = []
    for i, p in enumerate(pairs):
        sh = f35.pair_module_shares(p, named, fm.modulewise_shares)
        rows.append({"idx": i, "kind": p["kind"],
                     "lam_re": p["lam"].real, "lam_im": p["lam"].imag,
                     "abs_lam": abs(p["lam"]), "resid": p["resid"],
                     "unstable": bool(abs(p["lam"]) > f35.F35_UNSTABLE_MARGIN),
                     "top_module": _share_top(sh), "shares": sh})
    return rows


def _match_lams(rp_lams, lp_lams):
    """Nearest-eigenvalue matching of unstable right/left representatives
    (conjugate-consistent: both lists hold the Im>0 representative)."""
    m = []
    for lam in rp_lams:
        best, bd = None, None
        for q in lp_lams:
            d = abs(lam - q)
            if bd is None or d < bd:
                best, bd = q, d
        m.append({"lam_right": [lam.real, lam.imag],
                  "lam_left": ([best.real, best.imag] if best else None),
                  "delta": bd})
    return m


def cmd_eig(seed: int, step: int, metric: str) -> int:
    cell = f"{seed}:{step}"
    out_name = f"eig_{seed}_{step}_{metric}.json"
    done = load_raw(out_name)
    if done is not None and done.get("complete"):
        log(f"eig {cell} {metric}: already complete")
        return 0
    cd = cell_dir(seed, step)
    scales = load_scales(seed, step)
    out = {"cell": cell, "metric": metric, "offsets": {}, "complete": False}
    f3rho = json.load((F3RAW / f"rq4_{seed}_{step}.json").open())["offsets"]

    def scaled(z):
        return z if metric == "RAW" else f35.scale_state(z, scales)

    def ucache_path(off):
        return cd / f"ucache_{metric}_{off}.pt"

    # Memory-bounded schedule (2026-09-29 OOM postmortem: simultaneous
    # basis(39GB) + U_prev(24) + U(24) + W(24) peaked at 123GB -> kernel OOM).
    # The unstable RIGHT basis is offloaded to disk per offset and reloaded
    # only when the Arnoldi basis is no longer resident; peak now ~77GB.
    prev_ucache, off_prev = None, None
    for o in f35.F35_OFFSETS:
        t0 = time.time()
        op_raw, named_o, opt_o, model_o, loss_o = build_operator(o)
        f3m.named, f3m.opt, f3m.model_cpu = named_o, opt_o, model_o
        op = op_raw if metric == "RAW" else f35.ScaledOperator(op_raw, scales)
        # ---- right eigenspace (streaming; peak = basis + one pair + U) ----
        H, basis, m_eff = f35.arnoldi_basis(op.jvp, named_o)
        rho_all = float(np.abs(np.linalg.eig(H[:m_eff, :m_eff].numpy())[0]).max())
        right_rows, U_unst, r_lams = f35.extract_pairs_streaming(
            H, basis, named_o, fm.modulewise_shares)
        del basis
        phase_gc(f"{cell}-{metric}-o{o}-right")
        dU = len(U_unst)
        # ---- snapshots at t=o: E_right now (U alive) ----
        zs = {}
        fp = cd / f"forced_t{o}.pt"
        if o >= 1 and fp.exists():
            zs["forced"] = torch.load(fp, weights_only=False)
        if o == 0:
            zs["impulse"] = torch.load(cd / "q0.pt", weights_only=False)
        elif (cd / f"impulse_t{o}.pt").exists():
            zs["impulse"] = torch.load(cd / f"impulse_t{o}.pt", weights_only=False)
        e_right = {k: f35.alignment(U_unst, scaled(z)) for k, z in zs.items()}
        # offload this offset's unstable RIGHT basis; free from RAM
        tmpu = ucache_path(o).with_suffix(".tmp")
        torch.save(U_unst, tmpu)
        os.replace(tmpu, ucache_path(o))
        del U_unst
        # ---- left eigenspace (peak = basis + W) ----
        Hl, basis_l, m_eff_l = f35.arnoldi_basis(op.vjp, named_o)
        left_rows, W_unst, l_lams = f35.extract_pairs_streaming(
            Hl, basis_l, named_o, fm.modulewise_shares)
        del basis_l
        phase_gc(f"{cell}-{metric}-o{o}-left")
        rec = {"loss": loss_o, "m_eff": m_eff, "rho_all": rho_all,
               "right_pairs": right_rows, "left_pairs": left_rows,
               "d_unst_right": dU, "d_unst_left": len(W_unst),
               "match": _match_lams(r_lams, l_lams)}
        if metric == "RAW":
            ref = float(f3rho[str(o)]["alpha1.0"]["rho"])
            rec["rho_anchor_f3"] = ref
            rec["rho_anchor_match"] = bool(rho_all == ref)
            assert rho_all == ref, (
                f"rho anchor FAIL {cell} off={o}: {rho_all!r} vs F3 {ref!r}")
        else:
            # DIML numerical-validity flag (ERRATUM): similarity preserves the
            # spectrum, but fp64 Krylov arithmetic without balancing degrades
            # when per-tensor scale ratios are extreme (v-floor ~1e-15).
            ref = float(f3rho[str(o)]["alpha1.0"]["rho"])
            rel_dev = abs(rho_all - ref) / max(ref, 1e-300)
            rec["rho_ref_f3"] = ref
            rec["diml_rel_dev"] = rel_dev
            rec["diml_num_unstable"] = bool(rel_dev > 0.05)
        # ---- alignments (t = o): E_left with W alive ----
        for key, z in zs.items():
            rec[key] = {"E_left": f35.alignment(W_unst, scaled(z)),
                        "E_right": e_right.get(key)}
        # ---- S_exc over the offset's source range (prereg §6 mapping) ----
        if o == 0:
            t_lo, t_hi = 1, 1
        elif o < 15:
            t_lo, t_hi = o + 1, 2 * o + 1
        else:
            t_lo, t_hi = 16, f3m.H_MAX
        sexc = []
        for t in range(t_lo, t_hi + 1):
            p = cd / f"inh_t{t}.pt"
            if not p.exists():
                continue
            s = torch.load(p, weights_only=False)  # {"th": inh} only
            s = {"th": s["th"],
                 "m": {n: torch.zeros_like(tt) for n, tt in s["th"].items()},
                 "v": {n: torch.zeros_like(tt) for n, tt in s["th"].items()}}
            s = scaled(s)
            sexc.append({"t": t, "S_exc": f35.alignment(W_unst, s)})
        rec["S_exc"] = sexc
        # ---- offset-0 extras: random probes + causal projection probe ----
        if o == 0:
            probes = {name: f3.make_probe(f3m.named, name, ps)
                      for name, ps in f3.F3_PROBE_SEEDS.items()}
            rq2 = json.load((F3RAW / f"rq2_{seed}_{step}.json").open())
            g32 = {}
            for name in list(probes) + ["q0_lrsource"]:
                arm = rq2["steps"][-1]["arms"].get(f"a1.0|{name}")
                g32[name] = arm["dz_norm"] if arm else None
            rp = []
            for name, q in probes.items():
                rp.append({"direction": name,
                           "E_left_init": f35.alignment(W_unst, scaled(q)),
                           "G32_f3": g32.get(name)})
            rp.append({"direction": "q0_lrsource",
                       "E_left_init": f35.alignment(W_unst, scaled(zs["impulse"])),
                       "G32_f3": g32.get("q0_lrsource")})
            rec["random_probes"] = {"rows": rp, "G32_f3": g32}

            # -------- PHASE J: causal mode-projection probe --------
            probe = {"metric": metric, "kappa": None, "status": "NOT_ESTIMATED",
                     "d_u": 0}
            if rec.get("diml_num_unstable"):
                probe["status"] = "NUMERICALLY_UNSTABLE_DIML"
            elif dU > 0 and dU == len(W_unst):
                U0 = torch.load(ucache_path(o), weights_only=False)
                kappa, P = f35.biorthogonal_projector(U0, W_unst)
                del U0
                probe["kappa"] = (float(kappa) if math.isfinite(kappa)
                                  else None)
                if P is not None and kappa <= f35.F35_KAPPA_MAX:
                    q0 = torch.load(cd / "q0.pt", weights_only=False)
                    q = scaled(q0)
                    qu, qr = f35.split_projection(P, q)
                    del P
                    if metric == "DIML":  # propagate in RAW physics space
                        qu = f35.scale_state(qu, scales, inverse=True)
                        qr = f35.scale_state(qr, scales, inverse=True)
                    arms = [clone_state(q0), clone_state(qu), clone_state(qr)]
                    names = ["full", "unstable", "removed"]
                    init = [f35.state_norm(a) for a in arms]
                    arms = _propagate_homog(arms, 0, f3m.H_MAX)
                    rows = []
                    for nm, a, i0 in zip(names, arms, init):
                        fin = f35.state_norm(a)
                        rows.append({"arm": nm, "init_norm": i0,
                                     "final_norm": fin,
                                     "G32": fin / max(i0, 1e-30)})
                    gf, gu, gr = (rows[0]["G32"], rows[1]["G32"], rows[2]["G32"])
                    phase_gc(f"{cell}-{metric}-probe")
                    probe.update({
                        "status": "OK", "d_u": dU, "rows": rows,
                        "MODE_COMPONENT_NECESSARY": bool(gr <= gf / 10.0),
                        "MODE_COMPONENT_SUFFICIENT": bool(gu >= 0.5 * gf)})
            rec["probe"] = probe
        del W_unst
        phase_gc(f"{cell}-{metric}-o{o}-align")
        # ---- interval comparisons with previous offset (PHASE I) ----
        if prev_ucache is not None and prev_ucache.exists() and dU > 0:
            U_prev = torch.load(prev_ucache, weights_only=False)
            U_now = torch.load(ucache_path(o), weights_only=False)
            if len(U_prev) > 0:
                s, th_max, th_mean = f35.principal_angles(U_prev, U_now)
                vp = {"t": off_prev,
                      "O_sub": f35.mean_singulars(U_prev, U_now),
                      "theta_max": th_max, "theta_mean": th_mean,
                      "d_prev": len(U_prev), "d_now": len(U_now)}
                vecs = [clone_state(v) for v in U_prev]
                del U_prev
                vecs = _propagate_homog(vecs, off_prev, o)
                # frozen O_prop (prereg §7): overlap(Phi U_t, U_t') -- each
                # propagated vector NORMALIZED to unit norm before projection,
                # so singular values lie in [0,1]. The un-normalized Gram is
                # kept as O_prop_gain (exploratory: subspace-mediated gain,
                # >1 under amplification).
                norms = [f35.state_norm(v) for v in vecs]
                vecs_n = [f35.state_scale(v, 1.0 / max(n, 1e-300))
                          for v, n in zip(vecs, norms)]
                vp["O_prop"] = f35.mean_singulars(vecs_n, U_now)
                vp["O_prop_gain"] = f35.mean_singulars(vecs, U_now)
                vp["prop_norms"] = norms
                del vecs, vecs_n
                rec["vs_prev"] = vp
                phase_gc(f"{cell}-{metric}-o{o}-interval")
            else:
                del U_prev
            del U_now
        out["offsets"][str(o)] = rec
        save_raw(out_name, out)  # incremental atomic; completion flag below
        log(f"eig {cell} {metric} off={o} rho={rho_all:.6g} dU={dU} "
            f"dW={rec['d_unst_left']} ({time.time() - t0:.0f}s)")
        prev_ucache, off_prev = ucache_path(o), o
        del op, op_raw
    for o in f35.F35_OFFSETS:
        ucache_path(o).unlink(missing_ok=True)
    out["complete"] = True
    save_raw(out_name, out)
    log(f"eig {cell} {metric}: COMPLETE")
    return 0


# ------------------------------------------------------------ postprocess ----
def _spearman(es, gs):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(es, gs).statistic)
    except Exception:
        re_ = np.argsort(np.argsort(es))
        rg_ = np.argsort(np.argsort(gs))
        return float(np.corrcoef(re_, rg_)[0, 1])


def cmd_postprocess() -> int:
    import csv
    cells = (("2042", "26"), ("2043", "26"), ("2042", "64"))
    cs = lambda s, t: f"{s}:{t}"  # noqa: E731
    adj = {cs(*c): load_raw(f"adjoint_{c[0]}_{c[1]}.json") for c in cells}
    traj = {cs(*c): load_raw(f"traj_{c[0]}_{c[1]}.json") for c in cells}
    eig = {(cs(*c), m): load_raw(f"eig_{c[0]}_{c[1]}_{m}.json")
           for c in cells for m in f35.F35_METRICS}

    with (OUT / "adjoint_consistency.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "seed_u", "seed_v", "lhs", "rhs", "disc", "pass",
                    "gate_pass", "tol"])
        for cell, d in adj.items():
            for r in d["rows"]:
                w.writerow([cell, r["seed_u"], r["seed_v"], r["lhs"], r["rhs"],
                            r["disc"], r["pass"], d["gate_pass"], d["tol"]])
    gate_ok = all(d["gate_pass"] for d in adj.values() if d)

    with (OUT / "eigenpairs_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "offset", "side", "idx", "kind", "lam_re",
                    "lam_im", "abs_lam", "resid", "unstable", "top_module",
                    "d_unst", "rho_all", "rho_anchor_f3", "rho_anchor_match"])
        for (cell, metric), d in eig.items():
            for off, rec in d["offsets"].items():
                for side, key in (("right", "right_pairs"), ("left", "left_pairs")):
                    for r in rec[key]:
                        w.writerow([cell, metric, off, side, r["idx"], r["kind"],
                                    r["lam_re"], r["lam_im"], r["abs_lam"],
                                    r["resid"], r["unstable"], r["top_module"],
                                    rec[f"d_unst_{side}"], rec["rho_all"],
                                    rec.get("rho_anchor_f3", ""),
                                    rec.get("rho_anchor_match", "")])

    align_sum = {}
    with (OUT / "tangent_unstable_alignment.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "traj", "t", "E_left", "E_right",
                    "num_flag"])
        for (cell, metric), d in eig.items():
            for off, rec in d["offsets"].items():
                numf = "DIML_UNSTABLE" if rec.get("diml_num_unstable") else ""
                for tn in ("forced", "impulse"):
                    if tn in rec:
                        r = rec[tn]
                        w.writerow([cell, metric, tn, off,
                                    "" if r["E_left"] is None else r["E_left"],
                                    "" if r["E_right"] is None else r["E_right"],
                                    numf])
                        if not numf:
                            align_sum[(cell, metric, tn, int(off))] = r["E_left"]

    src_sum = {}
    with (OUT / "source_unstable_alignment.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "t", "offset_used", "S_exc", "num_flag"])
        for (cell, metric), d in eig.items():
            for off, rec in d["offsets"].items():
                numf = "DIML_UNSTABLE" if rec.get("diml_num_unstable") else ""
                for r in rec["S_exc"]:
                    w.writerow([cell, metric, r["t"], off,
                                "" if r["S_exc"] is None else r["S_exc"], numf])
                    if not numf:
                        src_sum[(cell, metric, r["t"])] = r["S_exc"]

    with (OUT / "subspace_rotation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "t_prev", "t", "O_sub", "theta_max",
                    "theta_mean", "d_prev", "d_now"])
        for (cell, metric), d in eig.items():
            for off, rec in d["offsets"].items():
                if "vs_prev" in rec:
                    r = rec["vs_prev"]
                    w.writerow([cell, metric, r["t"], off, r["O_sub"],
                                r["theta_max"], r["theta_mean"], r["d_prev"],
                                r["d_now"]])

    with (OUT / "propagated_overlap.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "t_prev", "t", "O_prop"])
        for (cell, metric), d in eig.items():
            for off, rec in d["offsets"].items():
                if "vs_prev" in rec and rec["vs_prev"].get("O_prop") is not None:
                    r = rec["vs_prev"]
                    w.writerow([cell, metric, r["t"], off, r["O_prop"]])

    with (OUT / "mode_projection_probe.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "status", "kappa", "d_u", "arm",
                    "init_norm", "final_norm", "G32", "NECESSARY", "SUFFICIENT"])
        probe_sum = {}
        for (cell, metric), d in eig.items():
            p = d["offsets"].get("0", {}).get("probe") or {}
            for r in (p.get("rows") or []):
                w.writerow([cell, metric, p.get("status"), p.get("kappa"),
                            p.get("d_u"), r["arm"], r["init_norm"],
                            r["final_norm"], r["G32"],
                            p.get("MODE_COMPONENT_NECESSARY", ""),
                            p.get("MODE_COMPONENT_SUFFICIENT", "")])
            probe_sum[(cell, metric)] = p

    with (OUT / "random_probe_relation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "metric", "direction", "E_left_init", "G32_f3",
                    "rank_E", "rank_G", "spearman_cell"])
        for (cell, metric), d in eig.items():
            rows = d["offsets"].get("0", {}).get("random_probes", {}).get("rows", [])
            vals = [(r["E_left_init"], r["G32_f3"]) for r in rows
                    if r["E_left_init"] is not None and r["G32_f3"] is not None]
            rho_s = None
            if len(vals) >= 4:
                es = [v[0] for v in vals]
                gs = [math.log10(max(v[1], 1e-30)) for v in vals]
                rho_s = _spearman(es, gs)
            order_e = sorted(range(len(vals)), key=lambda i: -vals[i][0])
            order_g = sorted(range(len(vals)), key=lambda i: -vals[i][1])
            rank_e = {i: k for k, i in enumerate(order_e)}
            rank_g = {i: k for k, i in enumerate(order_g)}
            for i, r in enumerate(rows):
                w.writerow([cell, metric, r["direction"], r["E_left_init"],
                            r["G32_f3"], rank_e.get(i, ""), rank_g.get(i, ""),
                            "" if rho_s is None else rho_s])

    # ---------------- verdict (prereg §11) ----------------
    EARLY = ("2042:26", "2043:26")
    MID = "2042:64"

    def e_summary(cell, metric, tn):
        vals = {t: v for (c, m, t2, t), v in align_sum.items()
                if c == cell and m == metric and t2 == tn and v is not None}
        return f35.traj_summary(vals)

    def s_summary(cell, metric):
        vals = {t: v for (c, m, t), v in src_sum.items()
                if c == cell and m == metric and v is not None}
        return f35.traj_summary(vals)

    def op_summary(cell, metric):
        vals = []
        for (c, m), d in eig.items():
            if c != cell or m != metric:
                continue
            for rec in d["offsets"].values():
                if rec.get("diml_num_unstable"):
                    continue
                if "vs_prev" in rec and rec["vs_prev"].get("O_prop") is not None:
                    vals.append(rec["vs_prev"]["O_prop"])
        return float(np.mean(vals)) if vals else None

    def direction_support(metric, stat="median"):
        for tn in ("forced", "impulse"):
            se = [e_summary(c, metric, tn)[stat] for c in EARLY]
            sm = e_summary(MID, metric, tn)[stat]
            if sm and all(s is not None and s > f35.F35_EXCITE_RATIO * sm
                          for s in se):
                return tn
        ss = [s_summary(c, metric)[stat] for c in EARLY]
        sm = s_summary(MID, metric)[stat]
        if sm and all(s is not None and s > f35.F35_EXCITE_RATIO * sm for s in ss):
            return "source"
        return None

    def persistence_support(metric):
        se = [op_summary(c, metric) for c in EARLY]
        sm = op_summary(MID, metric)
        if sm is None or sm <= 0 or any(s is None for s in se):
            return False
        return all(s > f35.F35_PERSIST_RATIO * sm for s in se)

    excite = {m: direction_support(m) for m in f35.F35_METRICS}
    pers = {m: persistence_support(m) for m in f35.F35_METRICS}

    robust_rows = []
    for tn in ("forced", "impulse"):
        for stat in ("median", "auc"):
            robust_rows.append(
                [f"E_left_{tn}_{stat}"] +
                [e_summary(c, m, tn)[stat] for m in f35.F35_METRICS
                 for c in EARLY + (MID,)])
    for stat in ("median", "auc"):
        robust_rows.append(
            [f"S_exc_{stat}"] +
            [s_summary(c, m)[stat] for m in f35.F35_METRICS
             for c in EARLY + (MID,)])
    robust_rows.append(
        ["O_prop_mean"] +
        [op_summary(c, m) for m in f35.F35_METRICS for c in EARLY + (MID,)])

    def agree(row):
        v = row[1:]
        seps = []
        for m0 in (0, 3):
            e1, e2, mm = v[m0], v[m0 + 1], v[m0 + 2]
            if None in (e1, e2, mm):
                seps.append(None)
            else:
                seps.append(min(e1, e2) / (mm if mm != 0 else 1e-30))
        if any(s is None for s in seps):
            return "INCOMPLETE"
        return "AGREE" if ((seps[0] > 1) == (seps[1] > 1)) else "DISAGREE"

    with (OUT / "metric_robustness.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "RAW_2042:26", "RAW_2043:26", "RAW_2042:64",
                    "DIML_2042:26", "DIML_2043:26", "DIML_2042:64",
                    "qualitative"])
        for row in robust_rows:
            w.writerow(row + [agree(row)])

    metric_contradict = any(agree(r) == "DISAGREE" for r in robust_rows)
    if not gate_ok:
        label = "LEFT_OPERATOR_GATE_FAIL"
    elif metric_contradict:
        label = "METRIC_DEPENDENT"
    else:
        both_e = all(excite[m] is not None for m in f35.F35_METRICS)
        both_p = all(pers[m] for m in f35.F35_METRICS)
        if both_e and both_p:
            label = "EXCITATION_AND_PERSISTENCE"
        elif both_e:
            label = "PHASE_SPECIFIC_MODE_EXCITATION"
        elif both_p:
            label = "UNSTABLE_SUBSPACE_PERSISTENCE"
        else:
            no_sep = all(excite[m] is None for m in f35.F35_METRICS)
            label = "MODE_COUPLING_NOT_SUPPORTED" if no_sep else "LOCAL_SPECTRUM_ONLY"

    summary = {"study": "F3_5_UNSTABLE_MODE_COUPLING",
               "verdict": {"label": label, "gate_pass": gate_ok,
                           "excitation_support": excite,
                           "persistence_support": pers,
                           "metric_contradict": metric_contradict},
               "cells": {}}
    for c in cells:
        cell = cs(*c)
        summary["cells"][cell] = {
            "traj_regression": (traj[cell]["regression"] if traj[cell] else None),
            "adjoint_gate": (adj[cell]["gate_pass"] if adj[cell] else None),
            "E_left": {m: {tn: e_summary(cell, m, tn)
                           for tn in ("forced", "impulse")}
                       for m in f35.F35_METRICS},
            "S_exc": {m: s_summary(cell, m) for m in f35.F35_METRICS},
            "O_prop_mean": {m: op_summary(cell, m) for m in f35.F35_METRICS},
            "probe": {m: probe_sum.get((cell, m), {}).get("status")
                      for m in f35.F35_METRICS}}
    with (OUT / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"verdict": label, "excitation": excite,
                      "persistence": pers, "gate_pass": gate_ok}, indent=1))
    return 0


# ------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("traj", "adjoint", "eig", "postprocess"))
    ap.add_argument("--cell", default=None, help="seed:step")
    ap.add_argument("--metric", default=None, choices=("RAW", "DIML"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.cmd != "postprocess":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, \
            "set CUDA_VISIBLE_DEVICES"
    torch.set_num_threads(min(32, torch.get_num_threads()))
    torch.manual_seed(0)
    if args.cmd == "postprocess":
        return cmd_postprocess()
    seed, step = (int(x) for x in args.cell.split(":"))
    assert (seed, step) in ((2042, 26), (2043, 26), (2042, 64))
    setup_cell(args.cell, args.device)
    if args.cmd == "traj":
        return cmd_traj(seed, step)
    if args.cmd == "adjoint":
        return cmd_adjoint(seed, step)
    return cmd_eig(seed, step, args.metric)


if __name__ == "__main__":
    sys.exit(main())
