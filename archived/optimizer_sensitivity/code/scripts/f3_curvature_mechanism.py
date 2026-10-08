#!/usr/bin/env python3
"""F3 curvature-feedback mechanism runner (frozen: docs/F3_CURVATURE_MECHANISM_PREREG.md).

Modes:
  gate : reproduction gates G1-G5 vs F2 artifacts (per cell, no resume)
  rq1  : alpha sweep (7 alphas x 32 steps, forced, shared primal)
  rq2  : impulse / source-off (alpha in {0,1} x 5 probes + q0_lrsource)
  rq3  : module receiver ablation (14 forced arms alpha=1 + 14 SO| twins),
         streamed in bounded RQ3_CHUNK chunks (memory-safe execution; frozen
         math unchanged). --arms restricts to a subset of forced arms (their
         SO twins are included automatically); completed arms are never
         recomputed (resume unit = arm).
  rq4  : per-offset spectrum (power iteration + Arnoldi, alpha in {0,1})

Resume: per-step metrics raw JSON + arm-state sidecar checkpoint every
CHECKPOINT_EVERY steps; rerunning a partial cell fast-forwards the (cheap)
primal and resumes the tangent from the last checkpoint. Never overwrites
completed raw artifacts beyond the checkpoint truncation point.
Run with CUDA_VISIBLE_DEVICES set. Usage:
  python scripts/f3_curvature_mechanism.py --rq rq1 --cell 2042:26
  python scripts/f3_curvature_mechanism.py --rq rq3 --cell 2042:26 \
      --arms FULL,NO_ATTN_Q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LAB = ROOT.parent
MM = LAB / "minimind_base"
if str(MM) not in sys.path:
    sys.path.insert(0, str(MM))

import torch

import opsens.f0_bank_audit_v01 as f0
import opsens.f1_mechanism_v01 as fm
import opsens.f3_curvature_mechanism_v01 as f3
from f2_hypergradient import (ACCUM, MAX_NORM, CLIP_DENOM_EPS, BASE_LR,
                              TOTAL_MICRO, H_MAX, CELL_STREAM,
                              STOP_TANGENT_NORM, Rig)
from opsens.a0c_adamw_response_propagation_v01 import clip_jvp
from opsens.batch_manifest import manifest_sha256

OUT = ROOT / "outputs" / "f3_curvature_mechanism"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
F2OUT = ROOT / "outputs" / "f2_hypergradient"
F3LOG = OUT / "f3_run.log"
COMPAT = ROOT / "outputs" / "f3_compatibility"
COMPAT_RAW = COMPAT / "raw"


def log(msg: str) -> None:
    import time as _t
    line = f"[{_t.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with F3LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def compat_log(msg: str) -> None:
    """Separate provenance log for compatibility-baseline runs (amendment §4)."""
    import time as _t
    line = f"[{_t.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    COMPAT.mkdir(parents=True, exist_ok=True)
    with (COMPAT / "compat_run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

CELLS = ((2042, 26), (2043, 26), (2042, 64))
RQ3_ARMS = ("FULL", "NO_ATTN_Q", "NO_ATTN_K", "NO_ATTN_V", "NO_ATTN_O",
            "NO_MLP", "NO_EMBED", "NO_NORM",
            "ONLY_ATTN_Q", "ONLY_ATTN_K", "ONLY_MLP",
            "NO_MLP_UP", "NO_MLP_GATE", "NO_MLP_DOWN")
CHECKPOINT_EVERY = 4
# Execution-level constants (NOT science): rq3 streams arms in bounded
# chunks instead of materialising all 28 fp64 arm states at once
# (~136 GB RSS -> swap thrash; root cause of the four dead 2026-09-28
# launches). Arms are independent given the shared deterministic primal,
# so chunking is bit-identical to single-pass semantics.
RQ3_CHUNK = 4
RQ3_PRIORITY = ("FULL", "NO_ATTN_Q", "ONLY_ATTN_Q", "NO_MLP", "ONLY_MLP")
# Exploratory per-arm A/C decomposition (same numbers f3.curvature_decomposition
# would produce on the masked arm trajectory; adds one dc=0 hom_step per step).
RQ3_DECOMP_ARMS = ("FULL", "NO_ATTN_Q", "ONLY_ATTN_Q", "NO_MLP", "ONLY_MLP")

rig = named = opt = model_cpu = None


def load_raw(name: str):
    p = RAW / name
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def save_raw(name: str, payload):
    p = RAW / name
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, p)


def state_path(name: str):
    return RAW / f"{name}.state.pt"


def save_state(name: str, arms_state: dict, done: int):
    torch.save({"done": done, "arms": arms_state}, state_path(name))


def load_state(name: str):
    p = state_path(name)
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def clone_state(dz):
    return {ch: {n: t.clone() for n, t in dz[ch].items()} for ch in ("th", "m", "v")}


def state_norms(dz):
    n_th = math.sqrt(sum(float(t.double().square().sum()) for t in dz["th"].values()))
    n_m = math.sqrt(sum(float(t.double().square().sum()) for t in dz["m"].values()))
    n_v = math.sqrt(sum(float(t.double().square().sum()) for t in dz["v"].values()))
    return n_th, n_m, n_v, math.sqrt(n_th ** 2 + n_m ** 2 + n_v ** 2)


def shares_row(named, dz):
    mods = fm.modulewise_shares(named, dz["th"])
    top = max(mods, key=mods.get) if mods else ""
    return top, mods


def primal_grads(h0, micro0):
    """Phase 1 at PRE-step theta: snapshot theta_prev, compute grads + clip.
    Deliberately does NOT apply the optimizer step: the curvature HVP and the
    vendored tangent step must see pre-step theta / pre-step optimizer state
    (exact F2 loop order: grads -> clip -> HVP -> clip_jvp -> tangent -> step).
    """
    theta_prev = {n: p.detach().clone() for n, p in named}
    lr = rig.get_lr(micro0 + h0 * ACCUM, TOTAL_MICRO, BASE_LR)
    opt.param_groups[0]["lr"] = lr
    grads, loss = rig.compute_grads(model_cpu, batches_g, h0)
    for n, p in named:
        p.grad = grads[n].clone()
    total = torch.nn.utils.clip_grad_norm_([p for _, p in named], MAX_NORM)
    coef = min(1.0, MAX_NORM / (float(total) + CLIP_DENOM_EPS))
    clipped = {n: p.grad.detach().clone() for n, p in named}
    return theta_prev, lr, grads, clipped, loss, {
        "preclip": float(total), "coef": coef, "lr": lr}


def primal_advance():
    """Phase 2: apply the pending optimizer step; snapshot POST-step theta."""
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {n: p.detach().clone() for n, p in named}


def dc_for(dth_fp64, grads, alpha, h, micro0_lossinfo=None):
    """dc for one arm at step h (h>=2 uses HVP of this arm's dtheta)."""
    if alpha == 0.0 or h == 1:
        return ({n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}, 0.0)
    src = rig.hvp(model_cpu, batches_g, h - 1,
                  {n: dth_fp64[n].to(torch.float32) for n, _ in named}, fp32=True)
    hvp_norm = math.sqrt(sum(float(t.double().square().sum()) for t in src.values()))
    cj = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=src,
                  max_norm=MAX_NORM)
    return ({n: t.double() for n, t in cj["clip_jvp"].items()}, hvp_norm)


def fast_forward_primal(done, micro0):
    """Re-run primal steps 1..done (cheap, deterministic) to position the
    primal at step done+1. Returns nothing; mutates named/opt/model_cpu."""
    for h in range(1, done + 1):
        primal_grads(h - 1, micro0)
        primal_advance()


def resume_plan(rq, seed, step):
    """Returns (raw_dict, done, arms_state_or_None). If partial raw exists but
    no matching sidecar, truncates raw rows to the sidecar checkpoint."""
    raw = load_raw(f"{rq}_{seed}_{step}.json") or {"steps": [], "meta": {}}
    st = load_state(f"{rq}_{seed}_{step}")
    if st is None:
        if raw["steps"]:
            log(f"{rq} {seed}:{step}: partial raw without sidecar -> restarting cell")
            raw = {"steps": [], "meta": {}}
        return raw, 0, None
    done = st["done"]
    if done > len(raw["steps"]):
        log(f"{rq} {seed}:{step}: sidecar ahead of raw -> restarting cell")
        raw = {"steps": [], "meta": {}}
        return raw, 0, None
    if done < len(raw["steps"]):
        raw["steps"] = raw["steps"][:done]
    return raw, done, st["arms"]


# ---------------------------------------------------------------- gate ------
def run_gate(seed, step, micro0, ref_raw=None, out_name=None):
    """G1/G2 reproduction gate. ref_raw defaults to the frozen archived F2 raw;
    the optional override exists ONLY for diagnostics (e.g. comparing against
    a same-day F2 rerun to separate code errors from environment drift) and
    never changes the frozen gate definition or tolerances."""
    cell = f"{seed}:{step}"
    ref_path = Path(ref_raw) if ref_raw else (F2OUT / f"raw_{seed}_{step}.json")
    ref = json.load(ref_path.open())
    ref_norms = {}
    for r in ref["tangent"]:
        if r.get("cell", cell) == cell:
            ref_norms[(r["arm"], r["h"])] = r["dtheta_u_norm"]
    arms = {}
    for label, alpha in (("B", 0.0), ("C", 1.0)):
        arms[label] = {"dth": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "dm": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "dv": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "alpha": alpha}
    mask_full = f3.module_mask(named, "FULL")
    got = {"B": {}, "C": {}}
    losses = []
    for h in range(1, H_MAX + 1):
        theta_prev, lr, grads, clipped, loss, info = primal_grads(h - 1, micro0)
        losses.append(loss)
        pending = {}
        for label, a in arms.items():
            dc, _ = dc_for(a["dth"], grads, a["alpha"], h)
            dc = {n: a["alpha"] * t for n, t in f3.apply_mask(dc, mask_full).items()}
            pending[label] = f3.hom_step(named, opt, clipped, dc,
                                         {"th": a["dth"], "m": a["dm"], "v": a["dv"]})
        theta_next = primal_advance()
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}
        for label, a in arms.items():
            out = pending[label]
            a["dth"] = {n: out["th"][n] + inh[n] for n, _ in named}
            a["dm"] = out["m"]
            a["dv"] = out["v"]
            got[label][h] = math.sqrt(sum(float(t.double().square().sum())
                                          for t in a["dth"].values()))
        log(f"gate {cell} h={h} loss={loss:.6f} |dth|B={got['B'][h]:.6g} |dth|C={got['C'][h]:.6g}")
    rep = {"cell": cell, "G1": {}, "G2": {}, "losses": losses, "pass": True}
    worst = 0.0
    for h in range(1, H_MAX + 1):
        d1 = abs(got["B"][h] - ref_norms[("fp64/B", h)]) / max(ref_norms[("fp64/B", h)], 1e-30)
        d2 = abs(got["C"][h] - ref_norms[("fp64/C", h)]) / max(ref_norms[("fp64/C", h)], 1e-30)
        rep["G1"][h] = d1
        rep["G2"][h] = d2
        worst = max(worst, d1, d2)
        if d1 > 1e-12 or d2 > 1e-12:
            rep["pass"] = False
    # G4: primal losses vs reference losses. Default reference: F2 run log
    # (%.5f parse of archived run). If an explicit ref_raw is given and
    # carries full-precision per-cell "losses" (current-environment frozen
    # reference, amendment §6), those are the G4 reference — the archived
    # log is environment-stale and must not gate current-environment runs.
    ref_losses = ref.get("losses") if ref_raw else None
    if isinstance(ref_losses, dict) and cell in ref_losses and len(ref_losses[cell]) >= H_MAX:
        f2_losses = {h + 1: v for h, v in enumerate(ref_losses[cell])}
    else:
        f2_losses = {}
        current = None
        for line in (F2OUT / "f2_run.log").read_text().splitlines():
            if "cell" in line and "stream_seed=" in line:
                current = line.split("cell", 1)[1].split("stream_seed=")[0].strip()
            elif current == cell and "h=" in line and "loss=" in line:
                try:
                    hh = int(line.split("h=", 1)[1].split("/")[0])
                    f2_losses[hh] = float(line.split("loss=", 1)[1].split()[0])
                except Exception:
                    continue
    g4 = None
    if len(f2_losses) >= H_MAX:
        g4 = max(abs(losses[h - 1] - f2_losses[h]) for h in range(1, H_MAX + 1))
        if g4 > 2e-5:
            rep["pass"] = False
    rep["G4_max_loss_absdiff"] = g4
    save_raw(out_name or f"gate_{seed}_{step}.json", rep)
    log(f"gate {cell}: pass={rep['pass']} worst_rel={worst:.3g} G4={g4}")
    return 0 if rep["pass"] else 1


# ------------------------------------------------------------- compat -------
def run_compat(seed, step, micro0, rep, msha=""):
    """Compatibility baseline (amendment docs/F3_REPRODUCTION_GATE_AMENDMENT.md
    §4): F2-faithful fp64/B + fp64/C with full traces, one independent-process
    repeat per invocation. Loop order is byte-identical to the validated gate
    (primal_grads -> per-arm dc/hom_step at PRE-step state -> primal_advance
    -> add s_h)."""
    cell = f"{seed}:{step}"
    arms = {}
    for label, alpha in (("fp64/B", 0.0), ("fp64/C", 1.0)):
        arms[label] = {"th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                       "alpha": alpha}
    mask_full = f3.module_mask(named, "FULL")
    vg_cache = {}
    steps = []
    for h in range(1, H_MAX + 1):
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = primal_grads(h - 1, micro0)
        pending = {}
        hvp_by_label = {}
        for label, a in arms.items():
            dc, hvp_norm = dc_for(a["th"], grads, a["alpha"], h)
            hvp_by_label[label] = hvp_norm
            dc = {n: a["alpha"] * t for n, t in f3.apply_mask(dc, mask_full).items()}
            pending[label] = f3.hom_step(
                named, opt, clipped, dc, {"th": a["th"], "m": a["m"], "v": a["v"]})
        theta_next = primal_advance()
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}
        gnorm = math.sqrt(sum(float(t.float().square().sum()) for t in grads.values()))
        row = {"h": h, "loss": loss, "lr": lr, "clip_coef": info["coef"],
               "grad_norm": gnorm, "arms": {}}
        for label, a in arms.items():
            out = pending[label]
            a["th"] = {n: out["th"][n] + inh[n] for n, _ in named}
            a["m"] = out["m"]
            a["v"] = out["v"]
            n_th, n_m, n_v, n_z = state_norms(a)
            rec = {"dtheta_u_norm": n_th, "dm_u_norm": n_m, "dv_u_norm": n_v,
                   "dz_u_norm": n_z, "hvp_norm": hvp_by_label[label]}
            if h in f3.F3_H_READOUTS:
                if h not in vg_cache:
                    vg_cache[h] = rig.val_grad(model_cpu)
                vg = vg_cache[h]
                rec["h_eta"] = sum(float((vg[n].double() * a["th"][n].double()).sum())
                                   for n, _ in named)
                top, mods = shares_row(named, a)
                rec["top_module"] = top
                rec["shares"] = mods
            row["arms"][label] = rec
        steps.append(row)
        compat_log(f"compat {cell} rep={rep} h={h} loss={loss:.6f} "
                   f"|dth|B={row['arms']['fp64/B']['dtheta_u_norm']:.6g} "
                   f"|dth|C={row['arms']['fp64/C']['dtheta_u_norm']:.6g} "
                   f"({time.time() - t0:.0f}s)")
    COMPAT_RAW.mkdir(parents=True, exist_ok=True)
    payload = {"cell": cell, "rep": rep, "micro0": micro0,
               "manifest_sha256": msha, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "steps": steps}
    with (COMPAT_RAW / f"compat_{seed}_{step}_rep{rep}.json").open("w") as fh:
        json.dump(payload, fh, indent=1)
    compat_log(f"compat {cell} rep={rep} complete -> raw/compat_{seed}_{step}_rep{rep}.json")
    return 0


# ------------------------------------------------------------------ rq1 -----
def run_rq1(seed, step, micro0):
    cell = f"{seed}:{step}"
    raw, done, arm_state = resume_plan("rq1", seed, step)
    arms = {}
    for alpha in f3.F3_ALPHA_GRID:
        key = str(alpha)
        if arm_state is not None and key in arm_state:
            st = arm_state[key]
            arms[alpha] = {ch: st[ch] for ch in ("th", "m", "v")}
            arms[alpha]["stopped"] = st["stopped"]
        else:
            arms[alpha] = {
                "th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                "stopped": False}
    mask_full = f3.module_mask(named, "FULL")
    vg_cache = {}
    if done:
        fast_forward_primal(done, micro0)
        log(f"rq1 {cell}: resumed at h={done+1}")
    for h in range(1, H_MAX + 1):
        if h <= done:
            continue
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = primal_grads(h - 1, micro0)
        gnorm = math.sqrt(sum(float(t.float().square().sum()) for t in grads.values()))
        row = {"h": h, "loss": loss, "clip_coef": info["coef"], "grad_norm": gnorm,
               "arms": {}}
        dc_all = {}
        pending = {}
        for alpha, a in arms.items():
            if a["stopped"]:
                continue
            dc, hvp_norm = dc_for(a["th"], grads, alpha, h)
            dc = {n: alpha * t for n, t in f3.apply_mask(dc, mask_full).items()}
            dc_all[alpha] = (dc, hvp_norm)
            pending[alpha] = f3.hom_step(
                named, opt, clipped, dc, {"th": a["th"], "m": a["m"], "v": a["v"]})
        dec_row = {}
        for alpha in (0.0, 1.0):
            a = arms[alpha]
            if a["stopped"]:
                continue
            dz = {"th": a["th"], "m": a["m"], "v": a["v"]}
            dec = f3.curvature_decomposition(named, opt, clipped, dc_all[alpha][0], dz)
            dec_row[str(alpha)] = dec
        row["decomposition"] = dec_row
        theta_next = primal_advance()
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}
        for alpha, a in arms.items():
            if a["stopped"]:
                continue
            out = pending[alpha]
            a["th"] = {n: out["th"][n] + inh[n] for n, _ in named}
            a["m"] = out["m"]
            a["v"] = out["v"]
            n_th, n_m, n_v, n_z = state_norms(a)
            if not all(math.isfinite(x) for x in (n_z,)) or n_z > STOP_TANGENT_NORM:
                a["stopped"] = True
            top, mods = shares_row(named, a)
            rec = {"dtheta_norm": n_th, "dm_norm": n_m, "dv_norm": n_v,
                   "dz_norm": n_z, "hvp_norm": dc_all[alpha][1], "top_module": top,
                   "shares": mods, "stopped": a["stopped"]}
            if h in f3.F3_H_READOUTS and not a["stopped"]:
                if h not in vg_cache:
                    vg_cache[h] = rig.val_grad(model_cpu)
                vg = vg_cache[h]
                rec["h_eta"] = sum(float((vg[n].double() * a["th"][n].double()).sum())
                                   for n, _ in named)
            row["arms"][str(alpha)] = rec
        raw["steps"].append(row)
        save_raw(f"rq1_{seed}_{step}.json", raw)
        if h % CHECKPOINT_EVERY == 0 or h == H_MAX:
            save_state(f"rq1_{seed}_{step}", {
                str(al): {**{ch: arms[al][ch] for ch in ("th", "m", "v")},
                          "stopped": arms[al]["stopped"]}
                for al in arms}, h)
        log(f"rq1 {cell} h={h} loss={loss:.5f} coef={info['coef']:.4f} "
            f"|dth|a0={row['arms']['0.0']['dtheta_norm']:.4g} "
            f"|dth|a1={row['arms']['1.0']['dtheta_norm']:.4g} ({time.time()-t0:.0f}s)")
    return 0


# ------------------------------------------------------------------ rq2 -----
def run_rq2(seed, step, micro0):
    cell = f"{seed}:{step}"
    raw, done, arm_state = resume_plan("rq2", seed, step)
    probes = {name: f3.make_probe(named, name, pseed)
              for name, pseed in f3.F3_PROBE_SEEDS.items()}
    arms = {}
    keys = [f"a{a}|{p}" for a in (0.0, 1.0) for p in list(probes)] + \
           ["a1.0|q0_lrsource", "a0.0|q0_lrsource"]
    if arm_state is not None:
        for key in keys:
            if key in arm_state:
                st = arm_state[key]
                arms[key] = {"dz": {ch: st[ch] for ch in ("th", "m", "v")},
                             "alpha": 1.0 if key.startswith("a1.") else 0.0,
                             "stopped": st["stopped"]}
    for key in keys:
        if key not in arms:
            if "q0" in key:
                continue  # created at h==1 below
            pname = key.split("|")[1]
            arms[key] = {"dz": clone_state(probes[pname]),
                         "alpha": 1.0 if key.startswith("a1.") else 0.0,
                         "stopped": False}
    if done:
        fast_forward_primal(done, micro0)
        log(f"rq2 {cell}: resumed at h={done+1}")
    for h in range(1, H_MAX + 1):
        if h <= done:
            continue
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = primal_grads(h - 1, micro0)
        pending = {}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            dz = a["dz"]
            dc, _ = dc_for(dz["th"], grads, a["alpha"], h)
            pending[key] = f3.hom_step(named, opt, clipped, dc, dz)  # source-off
        theta_next = primal_advance()
        if h == 1:
            q0 = f3.lr_source_q0(named, theta_prev, theta_next)
            for a in (0.0, 1.0):
                key = f"a{a}|q0_lrsource"
                if key not in arms:
                    arms[key] = {"dz": clone_state(q0), "alpha": a, "stopped": False}
            raw.setdefault("probes", {})["q0_lrsource"] = \
                "theta-only unit tangent along (theta_1 - theta_0)"
            # J_1 on a theta-only probe is state-independent ((1-lr*wd) scaling
            # of the theta channel; dm=dv=0), so evaluating it here (after the
            # primal advance) is bit-identical to pre-step state. dc==0 at h=1.
            zero_dc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}
            for a in (0.0, 1.0):
                key = f"a{a}|q0_lrsource"
                pending[key] = f3.hom_step(named, opt, clipped, zero_dc, arms[key]["dz"])
        row = {"h": h, "loss": loss, "clip_coef": info["coef"], "arms": {}}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            a["dz"] = pending[key]
            out = a["dz"]
            n_th, n_m, n_v, n_z = state_norms(out)
            if not all(math.isfinite(x) for x in (n_z,)) or n_z > STOP_TANGENT_NORM:
                a["stopped"] = True
            row["arms"][key] = {"dz_norm": n_z, "dtheta_norm": n_th,
                                "dm_norm": n_m, "dv_norm": n_v,
                                "stopped": a["stopped"]}
        raw["steps"].append(row)
        save_raw(f"rq2_{seed}_{step}.json", raw)
        if h % CHECKPOINT_EVERY == 0 or h == H_MAX:
            save_state(f"rq2_{seed}_{step}", {
                key: {**{ch: a["dz"][ch] for ch in ("th", "m", "v")},
                      "stopped": a["stopped"]} for key, a in arms.items()}, h)
        s = row["arms"].get("a1.0|q0_lrsource", {}).get("dz_norm", float("nan"))
        s0 = row["arms"].get("a0.0|q0_lrsource", {}).get("dz_norm", float("nan"))
        log(f"rq2 {cell} h={h} loss={loss:.5f} |dz|q0_a1={s:.4g} |dz|q0_a0={s0:.4g} "
            f"({time.time()-t0:.0f}s)")
    return 0


# ------------------------------------------------------------------ rq3 -----
def _rq3_key_order(only_arms=None):
    """Flat execution order for pending rq3 keys: priority arms first
    (interim-report enablement), remaining arms in frozen RQ3_ARMS order,
    then the SO| twins in the same order. Chunked by the caller."""
    sel = set(only_arms) if only_arms else None

    def rank(arm):
        p = RQ3_PRIORITY.index(arm) if arm in RQ3_PRIORITY else len(RQ3_PRIORITY)
        return (p, RQ3_ARMS.index(arm))

    base = [a for a in RQ3_ARMS if sel is None or a in sel]
    base.sort(key=rank)
    return list(base) + [f"SO|{a}" for a in base]


def _rq3_arm_init(key, so, resume_state=None):
    """Build one arm dict; fresh zeros unless a sidecar state is given."""
    mask_arm = key[3:] if so else key
    if resume_state is not None:
        a = {"mask": f3.module_mask(named, mask_arm),
             "stopped": resume_state["stopped"], "so": so}
        if so:
            a["dz"] = resume_state["dz"]
        else:
            for ch in ("th", "m", "v"):
                a[ch] = resume_state[ch]
        return a
    if so:
        return {"dz": None, "mask": f3.module_mask(named, mask_arm),
                "stopped": False, "so": True}
    return {"th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
            "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
            "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
            "mask": f3.module_mask(named, mask_arm), "stopped": False, "so": False}


def _rq3_decomposition(out_a, out_ad):
    """Inline of f3.curvature_decomposition for an already-computed masked-dc
    pass (identical numbers; avoids recomputing the (A+C) hom_step)."""
    c = f3.state_add(out_ad, out_a, sa=1.0, sb=-1.0)
    n_adc, n_a, n_c = f3.state_norm(out_ad), f3.state_norm(out_a), f3.state_norm(c)
    inner = f3.state_inner(out_a, c)
    cos = inner / max(n_a * n_c, f3.F3_EPS_METRIC)
    return {"norm_AplusC": n_adc, "norm_A": n_a, "norm_C": n_c,
            "inner_A_C": inner, "cos": cos}


def _rq3_run_chunk(seed, step, micro0, cell, raw, chunk_keys, resume=None):
    """Propagate one bounded chunk of rq3 arms over the full 32-step shared
    primal rollout. Fresh rig.restore per chunk; per-step raw save (atomic);
    sidecar checkpoint every CHECKPOINT_EVERY steps holding only this
    chunk's arm states. Prereg §5 semantics:
      forced arm  -> dc = mask ⊙ clip_jvp(g_h, HVP_h(dθ_arm)), s_h added;
      SO|arm twin -> SAME masked dc for every h>=2, q0 init at h==1, s_h==0.
    """
    global named, opt, model_cpu
    t_wall0 = time.time()
    t_load0 = time.time()
    named, opt, model_cpu = rig.restore(state_g)
    load_s = time.time() - t_load0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    done = 0
    arms = {}
    if resume is not None:
        _, done, st_arms = resume
        for key in chunk_keys:
            arms[key] = _rq3_arm_init(key, key.startswith("SO|"), st_arms[key])
        fast_forward_primal(done, micro0)
        log(f"rq3 {cell}: chunk [{'+'.join(chunk_keys)}] resumed at h={done + 1}")
    else:
        for key in chunk_keys:
            arms[key] = _rq3_arm_init(key, key.startswith("SO|"))
    vg_cache = {}
    t_hvp = t_hom = t_prim = 0.0
    for h in range(1, H_MAX + 1):
        if h <= done:
            continue
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = primal_grads(h - 1, micro0)
        t_prim += time.time() - t0
        while len(raw["steps"]) < h:
            raw["steps"].append({"h": len(raw["steps"]) + 1, "arms": {}})
        row = raw["steps"][h - 1]
        if "loss" in row:
            if row["loss"] != loss:
                raise RuntimeError(
                    f"rq3 {cell}: primal loss drift at h={h} "
                    f"({loss!r} vs recorded {row['loss']!r}) -- environment "
                    f"no longer bitwise-matches the recorded rollout")
        else:
            row["loss"] = loss
            row["clip_coef"] = info["coef"]
        pending = {}
        hvp_by_arm = {}
        dec_by = {}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            if a["so"] and h == 1:
                continue  # initialized right after the primal advance below
            t1 = time.time()
            dc, hvp_norm = dc_for(a["dz"]["th"] if a["so"] else a["th"],
                                  grads, 1.0, h)
            dc = f3.apply_mask(dc, a["mask"])
            t_hvp += time.time() - t1
            if not a["so"]:
                hvp_by_arm[key] = hvp_norm
            t1 = time.time()
            dz_in = a["dz"] if a["so"] else {"th": a["th"], "m": a["m"], "v": a["v"]}
            pending[key] = f3.hom_step(named, opt, clipped, dc, dz_in)
            if (not a["so"] and key in RQ3_DECOMP_ARMS and h >= 2):
                zero = {n: torch.zeros_like(t) for n, t in clipped.items()}
                out_a = f3.hom_step(named, opt, clipped, zero, dz_in)
                dec_by[key] = _rq3_decomposition(out_a, pending[key])
            t_hom += time.time() - t1
        theta_next = primal_advance()
        if h == 1:
            # Source-off twins start from the frozen q0 probe. J_1 on a
            # theta-only probe is state-independent ((1-lr*wd) scaling of
            # the theta channel; dm=dv=0), so evaluating it here (after the
            # primal advance) is bit-identical to pre-step state; dc==0 at
            # h==1. (Same reasoning as rq2's q0 arms.)
            q0 = f3.lr_source_q0(named, theta_prev, theta_next)
            zero_dc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}
            for key, a in arms.items():
                if a["so"] and a["dz"] is None:
                    a["dz"] = clone_state(q0)
                    pending[key] = f3.hom_step(named, opt, clipped, zero_dc,
                                               a["dz"])
            raw.setdefault("probes", {})["q0_lrsource"] = \
                "theta-only unit tangent along (theta_1 - theta_0)"
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            if a["so"]:
                if key not in pending:
                    continue
                a["dz"] = pending[key]
                n_th, n_m, n_v, n_z = state_norms(a["dz"])
                if not all(math.isfinite(x) for x in (n_z,)) or n_z > STOP_TANGENT_NORM:
                    a["stopped"] = True
                row["arms"][key] = {"dz_norm": n_z, "dtheta_norm": n_th,
                                    "dm_norm": n_m, "dv_norm": n_v,
                                    "stopped": a["stopped"]}
                continue
            out = pending[key]
            a["th"] = {n: out["th"][n] + inh[n] for n, _ in named}
            a["m"] = out["m"]
            a["v"] = out["v"]
            n_th, n_m, n_v, n_z = state_norms(a)
            if not all(math.isfinite(x) for x in (n_z,)) or n_z > STOP_TANGENT_NORM:
                a["stopped"] = True
            top, mods = shares_row(named, a)
            rec = {"dtheta_norm": n_th, "dm_norm": n_m, "dv_norm": n_v,
                   "dz_norm": n_z, "hvp_norm": hvp_by_arm.get(key),
                   "top_module": top, "shares": mods, "stopped": a["stopped"]}
            if key in dec_by:
                rec["decomposition"] = dec_by[key]
            if h in f3.F3_H_READOUTS and not a["stopped"]:
                if h not in vg_cache:
                    vg_cache[h] = rig.val_grad(model_cpu)
                vg = vg_cache[h]
                rec["h_eta"] = sum(float((vg[n].double() * a["th"][n].double()).sum())
                                   for n, _ in named)
            row["arms"][key] = rec
        save_raw(f"rq3_{seed}_{step}.json", raw)
        if h % CHECKPOINT_EVERY == 0 or h == H_MAX:
            save_state(f"rq3_{seed}_{step}", {
                key: ({"dz": a["dz"], "stopped": a["stopped"]} if a["so"]
                      else {**{ch: a[ch] for ch in ("th", "m", "v")},
                            "stopped": a["stopped"]})
                for key, a in arms.items()}, h)
        full = row["arms"].get("FULL", {}).get("dtheta_norm", float("nan"))
        noq = row["arms"].get("NO_ATTN_Q", {}).get("dtheta_norm", float("nan"))
        sof = row["arms"].get("SO|FULL", {}).get("dz_norm", float("nan"))
        log(f"rq3 {cell} h={h} loss={loss:.5f} [{'+'.join(chunk_keys)}] "
            f"|dth|FULL={full:.4g} |dth|NOQ={noq:.4g} |dz|SO_FULL={sof:.4g} "
            f"({time.time() - t0:.0f}s)")
    stats = {"keys": list(chunk_keys), "wall_s": round(time.time() - t_wall0, 1),
             "restore_s": round(load_s, 1), "primal_s": round(t_prim, 1),
             "hvp_s": round(t_hvp, 1), "tangent_s": round(t_hom, 1),
             "peak_cpu_rss_gb": round(resource.getrusage(
                 resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
    if torch.cuda.is_available():
        try:
            stats["peak_gpu_mem_gb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        except Exception:
            pass
    raw["meta"].setdefault("chunks", []).append(stats)
    return stats


def run_rq3(seed, step, micro0, only_arms=None):
    """Module causal ablation (forced, alpha=1, masked dc) PLUS the frozen
    source-off masked impulse arms (prereg §5: G_impulse(H=16,32) at alpha=1
    with mask varying, same-table flow). Source-off twins are keyed SO|<arm>,
    initialized at h==1 from the frozen q0 probe and propagated with the SAME
    masked dc at every h>=2 but WITHOUT the LR source (s_h == 0).

    Execution is streamed in RQ3_CHUNK-arm chunks (resume unit = arm, recorded
    in raw.meta.completed_arms). The 2026-09-28 version materialised all 28
    fp64 arm states at once (~136 GB RSS -> swap thrash) and never propagated
    the SO twins beyond h=1 (its summary log line would KeyError at h=2);
    both were execution defects, fixed here against the frozen prereg text.
    Bit-identity with single-pass semantics holds because arms are
    independent given the shared deterministic primal (compat R0: bitwise
    repeatable across processes)."""
    cell = f"{seed}:{step}"
    raw = load_raw(f"rq3_{seed}_{step}.json") or {"steps": [], "meta": {}}
    raw.setdefault("meta", {}).setdefault("completed_arms", [])
    completed = set(raw["meta"]["completed_arms"])
    all_keys = list(RQ3_ARMS) + [f"SO|{a}" for a in RQ3_ARMS]
    todo = [k for k in all_keys if k not in completed]
    if only_arms is not None:
        allow = set(only_arms) | {f"SO|{a}" for a in only_arms}
        todo = [k for k in todo if k in allow]
    if not todo:
        log(f"rq3 {cell}: all requested arms already complete "
            f"({len(completed)} total)")
        _rq3_finalize(seed, step)
        return 0
    ordered = [k for k in _rq3_key_order(
        [a for a in only_arms] if only_arms is not None else None)
        if k in set(todo)]
    # An interrupted chunk (sidecar with done < H_MAX) resumes first, as its
    # own chunk, exactly from its checkpointed arm states.
    st = load_state(f"rq3_{seed}_{step}")
    resume = None
    if st is not None and 0 < st["done"] < H_MAX:
        sk = list(st["arms"].keys())
        if sk and all(k in set(todo) for k in sk):
            resume = (sk, st["done"], st["arms"])
            ordered = sk + [k for k in ordered if k not in set(sk)]
            log(f"rq3 {cell}: resuming interrupted chunk {sk} at h={st['done'] + 1}")
    chunks = [ordered[i:i + RQ3_CHUNK] for i in range(0, len(ordered), RQ3_CHUNK)]
    for chunk_keys in chunks:
        r = resume if (resume is not None and chunk_keys == resume[0]) else None
        _rq3_run_chunk(seed, step, micro0, cell, raw, chunk_keys, resume=r)
        completed |= set(chunk_keys)
        raw["meta"]["completed_arms"] = sorted(completed)
        save_raw(f"rq3_{seed}_{step}.json", raw)
    _rq3_finalize(seed, step)
    return 0


def _rq3_finalize(seed, step):
    """Validate full 28-arm schema at h=32; drop the resume sidecar only
    when the cell is complete (sidecar is a runtime artifact, not a result)."""
    cell = f"{seed}:{step}"
    raw = load_raw(f"rq3_{seed}_{step}.json") or {}
    want = set(RQ3_ARMS) | {f"SO|{a}" for a in RQ3_ARMS}
    steps = raw.get("steps", [])
    ok = (len(steps) == H_MAX and set(steps[-1].get("arms", {})) >= want)
    if ok:
        sp = state_path(f"rq3_{seed}_{step}")
        if sp.exists():
            sp.unlink()
        log(f"rq3 {cell}: ALL ARMS COMPLETE ({len(want)} arms, sidecar released)")
    else:
        have = set(steps[-1].get("arms", {})) if steps else set()
        log(f"rq3 {cell}: partial ({len(have & want)}/{len(want)} arms at h=32)")
    return ok


# ------------------------------------------------------------------ rq4 -----
def run_rq4(seed, step, micro0):
    cell = f"{seed}:{step}"
    raw = load_raw(f"rq4_{seed}_{step}.json") or {"offsets": {}, "meta": {}}
    for off in f3.F3_OFFSETS:
        if str(off) in raw["offsets"]:
            continue
        named_o, opt_o, model_o = rig.restore(state_g)
        for h in range(1, off + 1):
            lr = rig.get_lr(micro0 + (h - 1) * ACCUM, TOTAL_MICRO, BASE_LR)
            opt_o.param_groups[0]["lr"] = lr
            grads, loss = rig.compute_grads(model_o, batches_g, h - 1)
            for n, p in named_o:
                p.grad = grads[n].clone()
            torch.nn.utils.clip_grad_norm_([p for _, p in named_o], MAX_NORM)
            opt_o.step()
            opt_o.zero_grad(set_to_none=True)
        lr = rig.get_lr(micro0 + off * ACCUM, TOTAL_MICRO, BASE_LR)
        opt_o.param_groups[0]["lr"] = lr
        grads, loss = rig.compute_grads(model_o, batches_g, off)
        for n, p in named_o:
            p.grad = grads[n].clone()
        torch.nn.utils.clip_grad_norm_([p for _, p in named_o], MAX_NORM)
        clipped = {n: p.grad.detach().clone() for n, p in named_o}
        cj = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=grads,
                      max_norm=MAX_NORM)
        coef, active, k = f3.clip_linear_coefficients(cj, MAX_NORM)

        def hvp_fn(x, _off=off, _model=model_o):
            out = rig.hvp(_model, batches_g, _off,
                          {n: t.to(torch.float32) for n, t in x.items()}, fp32=True)
            return {n: t.double() for n, t in out.items()}

        rec = {"clip_active": active, "clip_coef": coef, "loss": loss}
        for alpha in (0.0, 1.0):
            t0 = time.time()
            op = f3.StepOperator(named_o, opt_o, clipped, coef=coef, active=active,
                                 k=k, hvp_fn=hvp_fn, enabled=(alpha == 1.0))
            sigma, iters = f3.power_iteration_sigma_max(op, named_o)
            rho, reliable, resid, m_eff = f3.arnoldi_rho(op, named_o)
            rec[f"alpha{alpha}"] = {
                "sigma_max": sigma, "pi_iters": iters,
                "rho": (rho if reliable else None),
                "rho_reliable": reliable, "rho_residual": resid, "arnoldi_m": m_eff,
                "nonnormal_proxy": (sigma / max(rho, 1e-30)) if reliable else None}
            log(f"rq4 {cell} off={off} alpha={alpha} sigma={sigma:.6g} "
                f"rho={'NA' if not reliable else f'{rho:.6g}'} resid={resid:.3g} "
                f"({time.time()-t0:.0f}s)")
        raw["offsets"][str(off)] = rec
        save_raw(f"rq4_{seed}_{step}.json", raw)
    return 0


# ------------------------------------------------------------------- main ---
batches_g = state_g = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", required=True,
                    choices=("gate", "compat", "rq1", "rq2", "rq3", "rq4"))
    ap.add_argument("--cell", required=True, help="seed:step e.g. 2042:26")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rep", type=int, default=1,
                    help="compat mode: independent-repeat index (amendment §4)")
    ap.add_argument("--ref-raw", default=None,
                    help="DIAGNOSTIC ONLY: alternate reference raw JSON for gate "
                         "(frozen default remains the archived F2 raw)")
    ap.add_argument("--out-name", default=None,
                    help="DIAGNOSTIC ONLY: alternate raw output file name")
    ap.add_argument("--arms", default=None,
                    help="rq3 only: comma-separated subset of forced arms "
                         "(SO twins included automatically); completed arms "
                         "are never recomputed")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "set CUDA_VISIBLE_DEVICES"
    torch.set_num_threads(min(32, torch.get_num_threads()))
    torch.manual_seed(0)
    seed, step = (int(x) for x in args.cell.split(":"))
    assert (seed, step) in CELLS, f"cell must be one of {CELLS}"

    global rig, named, opt, model_cpu, batches_g, state_g
    rig = Rig(args.device)
    batches_g, stream_seed = rig.batches_for(CELL_STREAM[(seed, step)])
    msha = manifest_sha256(batches_g)
    state_g = f0.load_bank_state(fm.bank_state_path("g1b_100m", seed, step))
    micro0 = int(state_g["metadata"]["global_micro_step"])
    named, opt, model_cpu = rig.restore(state_g)
    log(f"f3 {args.rq} cell {seed}:{step} stream_seed={stream_seed} "
        f"manifest={msha[:16]} micro0={micro0}")

    if args.rq == "gate":
        return run_gate(seed, step, micro0, ref_raw=args.ref_raw,
                        out_name=args.out_name)
    if args.rq == "compat":
        return run_compat(seed, step, micro0, rep=args.rep, msha=msha)
    if args.rq == "rq1":
        return run_rq1(seed, step, micro0)
    if args.rq == "rq2":
        return run_rq2(seed, step, micro0)
    if args.rq == "rq3":
        only = ([a.strip() for a in args.arms.split(",")]
                if args.arms else None)
        return run_rq3(seed, step, micro0, only_arms=only)
    if args.rq == "rq4":
        return run_rq4(seed, step, micro0)
    return 2


if __name__ == "__main__":
    sys.exit(main())
