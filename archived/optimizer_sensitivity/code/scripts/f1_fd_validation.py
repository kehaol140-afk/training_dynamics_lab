#!/usr/bin/env python3
"""F1 Stage 5: finite-difference validation of the analytic one-step Jacobian
on the real 1B checkpoint state (prereg section 6).

Anchors: A0 c=0 (matches the stage-4 exact diagonal); A1 c=exp_avg (activates
m/dsqrt cross terms). Slices (frozen):
  (a) wte_zero_v_rows   — up to 64 wte rows with most exact-zero v coords
  (b) wte_low_v_nonzero — 64 smallest-denominator v>0 coords in wte (q support)
  (c) attn_q_low_denom  — 64 smallest-denominator v>0 coords within attn_q slices
  (d) mlp_gate_sample   — 4096 random coords in blocks.0.ff_proj gate half
  (e) full_model_random — random_seed_11 full support (fp64 manual + fp32 native,
                          A0 only)
eps grids (frozen): fp64 {1e-2..1e-12}, fp32 {1e-2..1e-6}.
G1 gates evaluated per prereg section 8 and written to fd_gate.json.

Requires: external/peteish1/step{N}_{model,optim}.pt (SHA256-verified).
Outputs: outputs/f1_1b_replication/fd_validation.csv, fd_gate.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import opsens.f1_1b_replication_v01 as f1

OUT = f1.OUT
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "fd_validation_run.log"
WTE = "transformer.wte.weight"

FD64 = [10.0 ** -e for e in range(2, 13)]          # 1e-2 .. 1e-12
FD32 = [10.0 ** -e for e in range(2, 8)]           # 1e-2 .. 1e-6 (F0 floor)
ZERO_V_SLICE_REL_TOL = 0.05
NONZERO_SLICE_REL_TOL = 1e-6
COS_TOL = 0.999999
FP32_REL_TOL = 1e-3


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def manual_adamw_update(theta, m, v, step_t, c, lr, b1, b2, eps, wd):
    """Forward AdamW math (torch semantics), input dtypes."""
    t_after = step_t + 1.0
    m1 = b1 * m + (1.0 - b1) * c
    v1 = b2 * v + (1.0 - b2) * c.square()
    mhat = m1 / (1.0 - b1 ** t_after)
    vhat = v1 / (1.0 - b2 ** t_after)
    denom = vhat.sqrt() + eps
    return theta * (1.0 - lr * wd) - lr * mhat / denom


def native_adamw_step(theta, m, v, step_t, c, lr, b1, b2, eps, wd):
    p = torch.nn.Parameter(theta.detach().to(torch.float32).clone())
    opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    opt.state[p] = {"step": torch.tensor(float(step_t), dtype=torch.float32),
                    "exp_avg": m.detach().to(torch.float32).clone(),
                    "exp_avg_sq": v.detach().to(torch.float32).clone()}
    p.grad = c.detach().to(torch.float32).clone()
    opt.step()
    return p.detach()


def analytic_jvp_vendored(theta, m, v, step_t, c0, q, lr, b1, b2, eps, wd):
    """Frozen tangent math (vendored a0c adamw_tangent_step, single homogeneous slice)."""
    from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step

    p = torch.nn.Parameter(theta.detach().to(torch.float64).clone())

    class _O:
        pass

    opt = _O()
    opt.param_groups = [{"params": [p], "lr": lr, "betas": (b1, b2), "eps": eps,
                         "weight_decay": wd, "amsgrad": False, "maximize": False,
                         "differentiable": False}]
    opt.state = {p: {"step": torch.tensor(float(step_t)),
                     "exp_avg": m.detach().to(torch.float64).clone(),
                     "exp_avg_sq": v.detach().to(torch.float64).clone()}}
    out = adamw_tangent_step(
        named_parameters=[("slice", p)], optimizer=opt,
        clipped_gradient_by_name={"slice": c0.detach().to(torch.float64).clone()},
        tangent_clipped_gradient_by_name={"slice": q.detach().to(torch.float64).clone()},
        tangent_theta_by_name={"slice": torch.zeros_like(p)},
        tangent_m_by_name={"slice": torch.zeros_like(p)},
        tangent_v_by_name={"slice": torch.zeros_like(p)},
    )
    return out["tangent_theta"]["slice"]


def compare(fd, an):
    fd64 = fd.double().flatten()
    an64 = an.double().flatten()
    fn = float(fd64.norm())
    an_ = float(an64.norm())
    dot = float((fd64 * an64).sum()) if fd64.shape == an64.shape else float("nan")
    cos = dot / (fn * an_) if fn > 0 and an_ > 0 else float("nan")
    rel = abs(fn - an_) / an_ if an_ > 0 else (0.0 if fn == 0.0 else float("inf"))
    return rel, cos, fn, an_


def build_slices(named_state, meta, theta_state, gen):
    """Frozen slice construction. theta_state: name -> fp32 theta tensor."""
    g0 = meta["groups"][0]
    lr, b1, b2, eps = g0["lr"], g0["betas"][0], g0["betas"][1], g0["eps"]
    slots = dict(named_state)
    slices = []

    # (a) wte zero-v rows
    slot = slots[WTE]
    v = slot["exp_avg_sq"]
    zero_counts = (v.to(torch.float64) == 0).sum(dim=1)
    cand_rows = torch.nonzero(zero_counts > 0).reshape(-1)
    if cand_rows.numel():
        order = cand_rows[torch.argsort(-zero_counts[cand_rows], stable=True)][:64]
        ids = order.tolist()
        t = torch.zeros_like(v, dtype=torch.float64)
        mm = torch.zeros_like(v, dtype=torch.float64)
        vv = torch.zeros_like(v, dtype=torch.float64)
        t[ids] = theta_state[WTE][ids].to(torch.float64)
        mm[ids] = slot["exp_avg"][ids].to(torch.float64)
        vv[ids] = slot["exp_avg_sq"][ids].to(torch.float64)
        q = torch.zeros_like(vv)
        q[ids] = torch.randn((len(ids), v.shape[1]), generator=gen, dtype=torch.float64)
        q = q / q.norm()
        slices.append({"name": "wte_zero_v_rows", "theta": t, "m": mm, "v": vv, "q": q,
                       "n_coords": int(v[ids].numel()), "zero_v_slice": True,
                       "lr": lr, "wd": meta["groups"][1]["weight_decay"], "step_t": slot["step"]})
        del t, mm, vv, q
    del zero_counts, cand_rows

    # denominators for (b)/(c)
    def denom_fp64(v32, step_t):
        return torch.sqrt(f1.vhat1(v32.to(torch.float64), step_t, b2)) + eps

    # (b) wte low-v nonzero coords
    d = denom_fp64(v, slot["step"])
    mask = v > 0
    d = torch.where(mask, d, torch.full_like(d, float("inf")))
    k = 64
    vals, flat = torch.topk(d.reshape(-1), k, largest=False)
    rows_b = sorted(set((flat // v.shape[1]).tolist()))
    t = torch.zeros_like(v, dtype=torch.float64)
    mm = torch.zeros_like(v, dtype=torch.float64)
    vv = torch.zeros_like(v, dtype=torch.float64)
    t[rows_b] = theta_state[WTE][rows_b].to(torch.float64)
    mm[rows_b] = slot["exp_avg"][rows_b].to(torch.float64)
    vv[rows_b] = slot["exp_avg_sq"][rows_b].to(torch.float64)
    q = torch.zeros_like(vv).reshape(-1)
    q[flat] = torch.randn(k, generator=gen, dtype=torch.float64)
    q = (q / q.norm()).reshape(vv.shape)
    slices.append({"name": "wte_low_v_nonzero", "theta": t, "m": mm, "v": vv, "q": q,
                   "n_coords": k, "zero_v_slice": False,
                   "lr": lr, "wd": meta["groups"][1]["weight_decay"], "step_t": slot["step"]})
    del t, mm, vv, q, d

    # (c) attn_q low-denominator coords across layers
    cands = []
    for name, s in named_state:
        if ".att_proj.weight" not in name:
            continue
        vq = s["exp_avg_sq"][:f1.ATT_D]
        dq = denom_fp64(vq, s["step"])
        dq = torch.where(vq > 0, dq, torch.full_like(dq, float("inf")))
        vals, flat = torch.topk(dq.reshape(-1), 64, largest=False)
        cands.append((name, vals, flat))
    allv = torch.cat([c[1] for c in cands])
    order = torch.argsort(allv)[:64]
    by_name = {}
    for pos in order.tolist():
        off = 0
        for name, vals, flat in cands:
            if off <= pos < off + vals.numel():
                by_name.setdefault(name, []).append(int(flat[pos - off]))
                break
            off += vals.numel()
    if by_name:
        first_name = sorted(by_name)[0]
        s = slots[first_name]
        t = torch.zeros_like(s["exp_avg_sq"], dtype=torch.float64)
        # frozen: use the first sorted involved tensor; q restricted to its
        # selected coords; state = that full tensor.
        ids = by_name[first_name]
        vv = torch.zeros_like(s["exp_avg_sq"], dtype=torch.float64)
        mm = torch.zeros_like(s["exp_avg_sq"], dtype=torch.float64)
        t[:] = theta_state[first_name].to(torch.float64)
        mm[:] = s["exp_avg"].to(torch.float64)
        vv[:] = s["exp_avg_sq"].to(torch.float64)
        q = torch.zeros_like(vv).reshape(-1)
        qf = torch.tensor([i for i in ids], dtype=torch.long)
        q[qf] = torch.randn(qf.numel(), generator=gen, dtype=torch.float64)
        q = (q / q.norm()).reshape(vv.shape)
        slices.append({"name": "attn_q_low_denom", "theta": t, "m": mm, "v": vv, "q": q,
                       "n_coords": len(ids), "zero_v_slice": False,
                       "lr": lr, "wd": meta["groups"][0]["weight_decay"],
                       "step_t": s["step"]})
        del t, mm, vv, q

    # (d) mlp_gate sample
    name_d = "transformer.blocks.0.ff_proj.weight"
    s = slots[name_d]
    vg = s["exp_avg_sq"][:f1.MLP_H]
    t = torch.zeros_like(vg, dtype=torch.float64)
    mm = torch.zeros_like(vg, dtype=torch.float64)
    vv = torch.zeros_like(vg, dtype=torch.float64)
    t[:] = theta_state[name_d][:f1.MLP_H].to(torch.float64)
    mm[:] = s["exp_avg"][:f1.MLP_H].to(torch.float64)
    vv[:] = s["exp_avg_sq"][:f1.MLP_H].to(torch.float64)
    q = torch.zeros_like(vv).reshape(-1)
    sel = torch.randperm(vv.numel(), generator=gen)[:4096]
    q[sel] = torch.randn(sel.numel(), generator=gen, dtype=torch.float64)
    q = (q / q.norm()).reshape(vv.shape)
    slices.append({"name": "mlp_gate_sample", "theta": t, "m": mm, "v": vv, "q": q,
                   "n_coords": int(sel.numel()), "zero_v_slice": False,
                   "lr": lr, "wd": meta["groups"][0]["weight_decay"], "step_t": s["step"]})
    del t, mm, vv, q
    return slices, (lr, b1, b2, eps)


def full_model_fd(named_state, meta, theta_state, gen):
    """Slice (e): streamed fp64 manual + fp32 native FD vs streamed diagonal."""
    lr, b1, b2, eps = (meta["groups"][0]["lr"], meta["groups"][0]["betas"][0],
                       meta["groups"][0]["betas"][1], meta["groups"][0]["eps"])
    results = []

    for eps_fd in FD64:
        gen2 = torch.Generator().manual_seed(20260926)
        fn2 = 0.0
        dot = 0.0
        cos_num = 0.0
        for name, slot in named_state:
            q = torch.randn(slot["exp_avg_sq"].shape, generator=gen2, dtype=torch.float64)
            wd = next(g["weight_decay"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            lr_p = next(g["lr"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            th = theta_state[name].to(torch.float64)
            up = manual_adamw_update(th, slot["exp_avg"].to(torch.float64),
                                     slot["exp_avg_sq"].to(torch.float64), slot["step"],
                                     eps_fd * q, lr_p, b1, b2, eps, wd)
            um = manual_adamw_update(th, slot["exp_avg"].to(torch.float64),
                                     slot["exp_avg_sq"].to(torch.float64), slot["step"],
                                     -eps_fd * q, lr_p, b1, b2, eps, wd)
            fd = (up - um) / (2.0 * eps_fd)
            g_dt = f1.diagonal_gain(slot["exp_avg_sq"], slot["step"], lr_p, b1, b2, eps, compute_dtype=torch.float64)
            an = -g_dt * q
            fn2 += float((fd * fd).sum())
            dot += float((fd * an).sum())
            del q, up, um, fd, an, g_dt, th
        gen3 = torch.Generator().manual_seed(20260926)
        # recompute analytic norms with same gen stream
        qn2 = 0.0
        an_norm2 = 0.0
        for name, slot in named_state:
            q = torch.randn(slot["exp_avg_sq"].shape, generator=gen3, dtype=torch.float64)
            lr_p = next(g["lr"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            g_dt = f1.diagonal_gain(slot["exp_avg_sq"], slot["step"], lr_p, b1, b2, eps, compute_dtype=torch.float64)
            jv = -g_dt * q
            qn2 += float((q * q).sum())
            an_norm2 += float((jv * jv).sum())
            del q, jv, g_dt
        fn = math.sqrt(fn2)
        an_ = math.sqrt(an_norm2)
        cos = dot / (fn * an_) if fn > 0 and an_ > 0 else float("nan")
        rel = abs(fn - an_) / an_
        results.append({"eps": eps_fd, "precision": "float64_manual_math",
                        "rel_err": rel, "cosine": cos, "fd_norm": fn, "analytic_norm": an_})
        log(f"  full_model fp64 eps={eps_fd:.0e} rel={rel:.3g} cos={cos:.8f}")
    for eps_fd in FD32:
        gen2 = torch.Generator().manual_seed(20260926)
        fn2 = 0.0
        dot = 0.0
        for name, slot in named_state:
            q = torch.randn(slot["exp_avg_sq"].shape, generator=gen2, dtype=torch.float64)
            wd = next(g["weight_decay"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            lr_p = next(g["lr"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            up = native_adamw_step(theta_state[name], slot["exp_avg"], slot["exp_avg_sq"],
                                   slot["step"], (eps_fd * q).to(torch.float32), lr_p, b1, b2, eps, wd)
            um = native_adamw_step(theta_state[name], slot["exp_avg"], slot["exp_avg_sq"],
                                   slot["step"], (-eps_fd * q).to(torch.float32), lr_p, b1, b2, eps, wd)
            fd = (up.double() - um.double()) / (2.0 * eps_fd)
            g_dt = f1.diagonal_gain(slot["exp_avg_sq"], slot["step"], lr_p, b1, b2, eps, compute_dtype=torch.float64)
            an = -g_dt * q
            fn2 += float((fd * fd).sum())
            dot += float((fd * an).sum())
            del q, up, um, fd, an, g_dt
        gen3 = torch.Generator().manual_seed(20260926)
        qn2 = 0.0
        an_norm2 = 0.0
        for name, slot in named_state:
            q = torch.randn(slot["exp_avg_sq"].shape, generator=gen3, dtype=torch.float64)
            lr_p = next(g["lr"] for g, names in zip(meta["groups"], meta["group_names"]) if name in names)
            g_dt = f1.diagonal_gain(slot["exp_avg_sq"], slot["step"], lr_p, b1, b2, eps, compute_dtype=torch.float64)
            jv = -g_dt * q
            qn2 += float((q * q).sum())
            an_norm2 += float((jv * jv).sum())
            del q, jv, g_dt
        fn = math.sqrt(fn2)
        an_ = math.sqrt(an_norm2)
        cos = dot / (fn * an_) if fn > 0 and an_ > 0 else float("nan")
        rel = abs(fn - an_) / an_
        results.append({"eps": eps_fd, "precision": "float32_native_adamw",
                        "rel_err": rel, "cosine": cos, "fd_norm": fn, "analytic_norm": an_})
        log(f"  full_model fp32 eps={eps_fd:.0e} rel={rel:.3g} cos={cos:.8f}")
    return results


def main() -> int:
    torch.set_num_threads(min(32, torch.get_num_threads()))
    rows = []
    gate = {"per_slice": {}, "steps_done": []}
    for step in f1.STEPS:
        ok_o = f1.verify_checkpoint(step, "optim.pt")
        ok_m = f1.verify_checkpoint(step, "model.pt")
        if not (ok_o["sha_ok"] and ok_m["sha_ok"]):
            log(f"step{step} files not ready (optim {ok_o['sha_ok']}, model {ok_m['sha_ok']}); skipped")
            continue
        t0 = time.time()
        payload = f1.load_optim_payload(step)
        named_state, meta = f1.optim_named_state(step, payload)
        model_state = f1.load_model_state(str(f1.ckpt_path(step, "model.pt")))
        theta_state = {k: v for k, v in model_state.items()}
        gen = torch.Generator().manual_seed(20260926)
        slices, (lr, b1, b2, eps) = build_slices(named_state, meta, theta_state, gen)
        log(f"step{step} slices: {[s['name'] for s in slices]}")

        for s in slices:
            for anchor_name, c0 in (("c0_zero", torch.zeros_like(s["v"])),
                                    ("c0_mhat", s["m"].clone())):
                an = analytic_jvp_vendored(s["theta"], s["m"], s["v"], s["step_t"],
                                           c0, s["q"], s["lr"], b1, b2, eps, s["wd"])
                for eps_fd in FD64:
                    up = manual_adamw_update(s["theta"], s["m"], s["v"], s["step_t"],
                                             c0 + eps_fd * s["q"], s["lr"], b1, b2, eps, s["wd"])
                    um = manual_adamw_update(s["theta"], s["m"], s["v"], s["step_t"],
                                             c0 - eps_fd * s["q"], s["lr"], b1, b2, eps, s["wd"])
                    fd = (up - um) / (2.0 * eps_fd)
                    rel, cos, fn, an_ = compare(fd, an)
                    rows.append({"step": step, "slice": s["name"], "anchor": anchor_name,
                                 "precision": "float64_manual_math", "eps": eps_fd,
                                 "rel_err": rel, "cosine": cos,
                                 "fd_norm": fn, "analytic_norm": an_,
                                 "fd_finite": bool(torch.isfinite(fd).all())})
                for eps_fd in FD32:
                    up = native_adamw_step(s["theta"], s["m"], s["v"], s["step_t"],
                                           (c0 + eps_fd * s["q"]).to(torch.float32),
                                           s["lr"], b1, b2, eps, s["wd"])
                    um = native_adamw_step(s["theta"], s["m"], s["v"], s["step_t"],
                                           (c0 - eps_fd * s["q"]).to(torch.float32),
                                           s["lr"], b1, b2, eps, s["wd"])
                    fd = (up.double() - um.double()) / (2.0 * eps_fd)
                    rel, cos, fn, an_ = compare(fd, an)
                    rows.append({"step": step, "slice": s["name"], "anchor": anchor_name,
                                 "precision": "float32_native_adamw", "eps": eps_fd,
                                 "rel_err": rel, "cosine": cos,
                                 "fd_norm": fn, "analytic_norm": an_,
                                 "fd_finite": bool(torch.isfinite(fd).all())})
                log(f"step{step} slice {s['name']} {anchor_name} done")
            del s["theta"], s["m"], s["v"], s["q"]

        # slice (e)
        e_rows = full_model_fd(named_state, meta, theta_state, gen)
        for r in e_rows:
            rows.append({"step": step, "slice": "full_model_random", "anchor": "c0_zero",
                         **r, "fd_finite": math.isfinite(r["fd_norm"])})
        del theta_state, model_state, payload, named_state

        # gates for this step
        step_rows = [r for r in rows if r["step"] == step]
        for slice_name in sorted({r["slice"] for r in step_rows}):
            is_zero_v = slice_name == "wte_zero_v_rows"
            sub = [r for r in step_rows if r["slice"] == slice_name and r["anchor"] == "c0_zero"]
            f64 = [r for r in sub if r["precision"] == "float64_manual_math"]
            f32 = [r for r in sub if r["precision"] == "float32_native_adamw"]
            if is_zero_v:
                seq = sorted(f64, key=lambda r: r["eps"], reverse=True)  # 1e-8 -> 1e-12
                rels = [r["rel_err"] for r in seq]
                argmin = rels.index(min(rels))
                mono_prefix = all(rels[i] >= rels[i + 1] * 0.5 for i in range(argmin))
                best = min(rels)
                ok64 = mono_prefix and best <= ZERO_V_SLICE_REL_TOL
            else:
                ok64 = any(r["rel_err"] <= NONZERO_SLICE_REL_TOL and r["cosine"] >= COS_TOL
                           for r in f64)
            ok32 = any(r["rel_err"] <= FP32_REL_TOL for r in f32)
            gate["per_slice"][f"{step}/{slice_name}"] = {
                "gate64_pass": ok64, "gate32_pass": ok32,
                "best_rel64": min(r["rel_err"] for r in f64),
                "best_rel32": min(r["rel_err"] for r in f32),
            }
        gate["steps_done"].append(step)
        log(f"step{step} FD complete ({time.time()-t0:.0f}s); gates: "
            f"{ {k: v['gate64_pass'] for k, v in gate['per_slice'].items() if k.startswith(str(step))} }")

    if rows:
        with (OUT / "fd_validation.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        gate["overall_G1"] = all(v["gate64_pass"] for v in gate["per_slice"].values()) if gate["per_slice"] else None
        (OUT / "fd_gate.json").write_text(json.dumps(gate, indent=2))
        log(f"written {len(rows)} FD rows; overall_G1={gate['overall_G1']}")
        return 0
    log("no checkpoints ready")
    return 1


if __name__ == "__main__":
    sys.exit(main())
