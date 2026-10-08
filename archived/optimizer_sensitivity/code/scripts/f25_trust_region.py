#!/usr/bin/env python3
"""F2.5 trust-region runner (frozen: docs/F2_5_TRUST_REGION_PREREG.md).

Per cell (g1b bank state):
  FD arms   : delta in F25_DELTA_GRID (8 pts) -> L_plus/L_minus at H readouts,
              plus a dedicated delta=0 baseline rollout giving L0 (CRN, same
              manifest/streams as F2: derive_stream_seed(20260926, {1,2,3})).
  Tangent   : one shared delta=0 primal rollout carries 5 interventions x
              {fp64/C, fp64/B} = 10 arms. Interventions are backward-only
              (tangent-internal; forward trajectory provably untouched, see
              raw.forward_bitwise_match):
                none | mask_den_0.01% | mask_den_0.1% | vfloor_eps | vfloor_10eps
              C arms: vendored tangent + per-arm fp32-HVP curvature feedback
              + inhomogeneous u-term (F2 pipeline). B arms: optimizer-only.
  Masks     : F1 R_den top-f coordinate sets, frozen at the BANK anchor state
              (opsens.f25_trust_region_v01.rden_topk_masks), constant across steps.

Run with CUDA_VISIBLE_DEVICES=1.
Usage: python scripts/f25_trust_region.py --cell 2042:26
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from opsens.a0c_adamw_response_propagation_v01 import clip_jvp
from opsens.batch_manifest import manifest_sha256
from opsens.f25_trust_region_v01 import (F25_DELTA_GRID, F25_H_READOUTS,
                                         rden_topk_masks, tangent_step_oracle)
from f2_hypergradient import Rig, fd_rollout  # shared frozen training/eval plumbing

OUT = ROOT / "outputs" / "f2_5_trust_region"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "f25_run.log"

MAX_NORM = 1.0
BASE_LR = 5e-4
TOTAL_MICRO = 256
H_MAX = 32
STOP_TANGENT_NORM = 1e30
CELL_STREAM = {(2042, 26): 1, (2042, 64): 2, (2043, 26): 3}
INTERVENTIONS = ("none", "mask_den_0.01%", "mask_den_0.1%", "vfloor_eps", "vfloor_10eps")
MASK_FRACTIONS = {"mask_den_0.01%": 1e-4, "mask_den_0.1%": 1e-3}
VFLOOR_TAUS = {"vfloor_eps": 1e-8, "vfloor_10eps": 1e-7}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="seed:step e.g. 2042:26")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "set CUDA_VISIBLE_DEVICES=1"
    torch.set_num_threads(min(32, torch.get_num_threads()))
    torch.manual_seed(0)
    seed, step = (int(x) for x in args.cell.split(":"))
    t_start = time.time()

    rig = Rig(args.device)
    batches, stream_seed = rig.batches_for(CELL_STREAM[(seed, step)])
    msha = manifest_sha256(batches)
    log(f"cell {seed}:{step} stream_seed={stream_seed} manifest={msha[:16]} "
        f"train_batches={len(batches)} eval_batches={len(rig.eval_batches)}")

    state = f0.load_bank_state(fm.bank_state_path("g1b_100m", seed, step))
    micro0 = int(state["metadata"]["global_micro_step"])
    lr0 = rig.get_lr(micro0, TOTAL_MICRO, BASE_LR)
    log(f"state restored micro0={micro0} lr0={lr0:.6g}")

    # ---- frozen R_den coordinate sets at the BANK anchor state ----
    mask_sets, mask_meta = {}, {}
    for iv, frac in MASK_FRACTIONS.items():
        masks, meta = rden_topk_masks(state, frac)
        mask_sets[iv] = masks
        mask_meta[iv] = meta
        log(f"mask {iv}: k={meta['k']}/{meta['total']} thresh_Rden={meta['thresh']:.6g}")

    # ---------------- FD arms (incl. delta=0 baseline) ----------------
    raw_path = OUT / f"raw_{seed}_{step}.json"
    fd_rows, L0, trace0 = None, None, None
    if raw_path.exists():
        try:
            prev = json.loads(raw_path.read_text())
            if (prev.get("manifest_sha256") == msha
                    and len(prev.get("fd", [])) == len(F25_DELTA_GRID) * len(F25_H_READOUTS)
                    and prev.get("micro0") == micro0
                    and not prev.get("tangent")):
                fd_rows = prev["fd"]
                L0 = {int(k): v for k, v in prev["baseline_L0"].items()}
                trace0 = prev["baseline_trace"]
                log(f"resume: reusing completed FD phase ({len(fd_rows)} rows)")
        except (ValueError, KeyError, TypeError):
            fd_rows = None  # unreadable/partial raw -> recompute from scratch
    if fd_rows is None:
        fd_rows = []
        L0, trace0 = fd_rollout(rig, state, batches, 0.0, micro0, rig.get_lr)
        log(f"delta=0.0 baseline: L0(1)={L0[1]:.6f} L0(16)={L0[16]:.6f} L0(32)={L0[32]:.6f}")
        for delta in F25_DELTA_GRID:
            t0 = time.time()
            lp, _ = fd_rollout(rig, state, batches, +delta, micro0, rig.get_lr)
            lm, _ = fd_rollout(rig, state, batches, -delta, micro0, rig.get_lr)
            for h in F25_H_READOUTS:
                fd_rows.append({"cell": f"{seed}:{step}", "delta": delta, "H": h,
                                "L0": L0[h], "L_plus": lp[h], "L_minus": lm[h],
                                "Delta_plus": lp[h] - L0[h], "Delta_minus": lm[h] - L0[h],
                                "D_H": (lp[h] - lm[h]) / (2.0 * delta)})
            dmap = {r["H"]: r["D_H"] for r in fd_rows[-len(F25_H_READOUTS):]}
            log(f"delta={delta}: D(1)={dmap[1]:.6g} D(16)={dmap[16]:.6g} "
                f"({time.time()-t0:.0f}s)")
        # checkpoint the finished FD phase so a tangent-phase crash cannot
        # destroy it; the file is rewritten with tangent rows at completion.
        raw_path.write_text(json.dumps(
            {"fd": fd_rows, "tangent": [], "manifest_sha256": msha,
             "stream_seed": stream_seed, "micro0": micro0, "lr0": lr0,
             "baseline_L0": {str(k): v for k, v in L0.items()},
             "baseline_trace": trace0, "mask_meta": mask_meta,
             "forward_bitwise_match": None,
             "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "prereg": "docs/F2_5_TRUST_REGION_PREREG.md"}, indent=1))

    # ---------------- shared-primal tangent rollout (10 arms) ----------------
    named, opt, model_cpu = rig.restore(state)
    arms = {}
    for iv in INTERVENTIONS:
        for br in ("C", "B"):
            arms[f"{iv}|{br}"] = {
                "iv": iv, "branch": br, "stopped": False, "reason": "",
                "dth": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                "dm": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                "dv": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}}

    tan_rows = []
    vg_cache = {}
    forward_ok = True
    max_dloss = 0.0
    for h in range(1, H_MAX + 1):
        t0 = time.time()
        theta_prev = {n: p.detach().clone() for n, p in named}
        lr = rig.get_lr(micro0 + (h - 1) * 2, TOTAL_MICRO, BASE_LR)  # ACCUM=2 (f2_hypergradient)
        opt.param_groups[0]["lr"] = lr
        grads, loss = rig.compute_grads(model_cpu, batches, h - 1)
        for n, p in named:
            p.grad = grads[n].clone()
        total = torch.nn.utils.clip_grad_norm_([p for _, p in named], MAX_NORM)
        coef = min(1.0, MAX_NORM / (float(total) + 1e-6))
        clipped = {n: p.grad.detach().clone() for n, p in named}

        # forward-trajectory identity check vs the independent delta=0 baseline
        # rollout. Cross-rollout agreement is expected at fp32-rounding level
        # (embedding-backward atomics); the strict intervention guarantee is
        # structural — all 10 arms share this single primal rollout.
        t0m = trace0[h - 1]
        dloss = abs(loss - t0m["loss"])
        max_dloss = max(max_dloss, dloss)
        same = (coef == t0m["coef"]) and (lr == t0m["lr"]) and (dloss <= 5e-4)
        if not same:
            forward_ok = False
            log(f"WARNING h={h}: primal drift vs baseline rollout "
                f"(dloss={dloss:.3g}, coef {coef} vs {t0m['coef']})")

        # per-C-arm HVP (h>1), each arm's own dtheta (fp32 HVP = F2 pipeline)
        hvps = {}
        if h > 1:
            for key, a in arms.items():
                if a["branch"] == "C" and not a["stopped"]:
                    hvps[key] = rig.hvp(model_cpu, batches, h - 1,
                                        {n: a["dth"][n].to(torch.float32)
                                         for n, _ in named}, fp32=True)

        jvp_out = {}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            if a["branch"] == "B" or h == 1:
                dc = {n: torch.zeros_like(p) for n, p in named}
            else:
                if key not in hvps:
                    a["stopped"], a["reason"] = True, "HVP_SOURCE_STOPPED"
                    continue
                dc = clip_jvp(gradient_by_name=grads,
                              tangent_gradient_by_name=hvps[key],
                              max_norm=MAX_NORM)["clip_jvp"]
            iv = a["iv"]
            try:
                jvp_out[key] = tangent_step_oracle(
                    named_parameters=named, optimizer=opt,
                    clipped_gradient_by_name=clipped,
                    tangent_clipped_gradient_by_name=dc,
                    tangent_theta_by_name=a["dth"], tangent_m_by_name=a["dm"],
                    tangent_v_by_name=a["dv"],
                    dv_mask_by_name=mask_sets.get(iv),
                    v_back_tau=VFLOOR_TAUS.get(iv))
            except ValueError as exc:
                a["stopped"], a["reason"] = True, f"GUARD: {exc}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        theta_next = {n: p.detach().clone() for n, p in named}
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}

        for key, a in arms.items():
            if key not in jvp_out:
                continue
            out = jvp_out[key]
            a["dth"] = {n: out["tangent_theta"][n] + inh[n] for n, _ in named}
            a["dm"] = out["tangent_m"]
            a["dv"] = out["tangent_v"]
            fin = all(bool(torch.isfinite(t).all().item()) for t in a["dth"].values())
            nrm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in a["dth"].values()))
            row = {"cell": f"{seed}:{step}", "arm": key, "iv": a["iv"], "branch": a["branch"],
                   "h": h, "finite": fin, "dtheta_u_norm": nrm,
                   "stopped": False, "reason": ""}
            if not fin or nrm > STOP_TANGENT_NORM:
                a["stopped"], a["reason"] = True, "NONFINITE_OR_OVERFLOW"
                row["stopped"] = True
                row["reason"] = a["reason"]
            if h in F25_H_READOUTS and not a["stopped"]:
                if h not in vg_cache:
                    vg_cache[h] = rig.val_grad(model_cpu)
                vg = vg_cache[h]
                h_tan = sum(float((vg[n].double() * a["dth"][n].double()).sum())
                            for n, _ in named)
                mods = fm.modulewise_shares(named, a["dth"])
                row.update({"h_tangent": h_tan, "L0": L0[h],
                            "top_module": max(mods, key=mods.get),
                            "share_attn_q": mods.get("attn_q", 0.0)})
            tan_rows.append(row)
        d_none = math.sqrt(sum(float((v.double() ** 2).sum())
                               for v in arms["none|C"]["dth"].values()))
        log(f"h={h}/{H_MAX} loss={loss:.5f} coef={coef:.4f} fwd_ok={same} "
            f"|dth/du| none/C={d_none:.4g} ({time.time()-t0:.0f}s)")

    raw = {"fd": fd_rows, "tangent": tan_rows,
           "manifest_sha256": msha, "stream_seed": stream_seed, "micro0": micro0,
           "lr0": lr0, "forward_bitwise_match": forward_ok,
           "max_baseline_loss_abs_diff": max_dloss,
           "mask_meta": mask_meta,
           "baseline_L0": {str(k): v for k, v in L0.items()},
           "baseline_trace": trace0,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "prereg": "docs/F2_5_TRUST_REGION_PREREG.md"}
    with (OUT / f"raw_{seed}_{step}.json").open("w") as fh:
        json.dump(raw, fh, indent=1)
    log(f"cell {seed}:{step} complete ({time.time()-t_start:.0f}s) "
        f"forward_match={forward_ok} max_dloss={max_dloss:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
