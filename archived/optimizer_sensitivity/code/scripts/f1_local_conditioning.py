#!/usr/bin/env python3
"""F1 Stage 4: 1-step Jacobian conditioning at the c=0 anchor (prereg section 5).

Exact per-coordinate diagonal G_i (central derivative; prereg section 0) from
the real optim.pt state; streamed direction amplifications A(q) for the frozen
direction classes in fp64/fp32/bf16 linearization arithmetic; module-wise
components via prereg row-slices; sigma_max(J0) by power iteration on a cached
fp32 diagonal (exact for diagonal maps; 8 iterations / 1e-12 convergence) plus
dominant-coordinate localization (argmax cross-check); implementation
cross-check of the streamed diagonal against the vendored frozen tangent on
three probe tensors (rel err must be <= 1e-10).

Requires: external/peteish1/step{N}_optim.pt (SHA256-verified).
Outputs: outputs/f1_1b_replication/local_conditioning.csv,
         outputs/f1_1b_replication/power_iteration_1b.csv
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
from opsens.f1_1b_replication_v01 import MODULE_GROUPS

OUT = f1.OUT
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "local_conditioning_run.log"
WTE = "transformer.wte.weight"
PROBE_TENSORS = ("transformer.wte.weight",
                 "transformer.blocks.0.att_proj.weight",
                 "transformer.blocks.0.ff_proj.weight")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


class DiagonalField:
    def __init__(self, named_state, meta):
        self.named_state = named_state
        g0 = meta["groups"][0]
        self.beta1, self.beta2, self.eps = g0["betas"][0], g0["betas"][1], g0["eps"]
        self.lr_of = {}
        self.step_of = {}
        for g, names in zip(meta["groups"], meta["group_names"]):
            for n in names:
                self.lr_of[n] = g["lr"]
        for n, s in named_state:
            self.step_of[n] = s["step"]

    def gain(self, name, v_s, dtype):
        return f1.diagonal_gain(v_s, self.step_of[name], self.lr_of[name],
                                self.beta1, self.beta2, self.eps,
                                compute_dtype=dtype)


def evaluate_direction(field, named_state, direction, dtype, gen,
                       toprisk_coords=None, embed_rows=None):
    """Streamed A(q) + module components for one direction in one dtype."""
    q_sq = 0.0
    out_sq = {g: 0.0 for g in MODULE_GROUPS}
    for name, slot in named_state:
        v32, m32 = slot["exp_avg_sq"], slot["exp_avg"]
        for gname, sl in f1.classify_groups(name):
            v_s = (v32[sl] if sl != slice(None) else v32)
            if v_s.numel() == 0:
                continue
            g_dt = field.gain(name, v_s, dtype)
            q = None
            if direction.startswith("random:"):
                q = torch.randn(v_s.shape, generator=gen, dtype=torch.float64)
            elif direction == "grad_aligned":
                src = (m32[sl] if sl != slice(None) else m32)
                q = src.to(torch.float64).clone()
            elif direction == "toprisk_q":
                if gname == "attn_q":
                    ids = toprisk_coords.get(name)
                    if ids:
                        q = torch.zeros(v_s.numel(), dtype=torch.float64)
                        flat = torch.tensor(sorted(ids), dtype=torch.long)
                        vals = torch.randn(flat.numel(), generator=gen, dtype=torch.float64)
                        q[flat] = vals
                        q = q.reshape(v_s.shape)
                    else:
                        q = torch.zeros(v_s.shape, dtype=torch.float64)
            elif direction == "mlp_only":
                if gname.startswith("mlp_"):
                    q = torch.randn(v_s.shape, generator=gen, dtype=torch.float64)
                else:
                    q = torch.zeros(v_s.shape, dtype=torch.float64)
            elif direction in ("embed_tail", "embed_head_rows", "embed_zero_v_rows"):
                q = torch.zeros(v_s.shape, dtype=torch.float64)
                if name == WTE:
                    rows = embed_rows.get(direction, [])
                    if len(rows):
                        r = torch.tensor(rows, dtype=torch.long)
                        q[r] = torch.randn((r.numel(), q.shape[1]), generator=gen,
                                           dtype=torch.float64)
            else:
                raise ValueError(direction)
            if q is None:
                q = torch.zeros(v_s.shape, dtype=torch.float64)
            q_dt = q.to(dtype)
            prod = (g_dt * q_dt).to(torch.float64)
            q_sq += float((q_dt.to(torch.float64) ** 2).sum().item())
            out_sq[gname] += float((prod * prod).sum().item())
            del q, q_dt, prod, g_dt
    qn = math.sqrt(q_sq)
    a_total = math.sqrt(sum(out_sq.values())) / qn if qn > 0 else float("nan")
    comps = {g: math.sqrt(v) / qn for g, v in out_sq.items()}
    return a_total, comps, qn


def find_toprisk(field, named_state, k_frac=0.001):
    """Global k smallest denominators within attn_q slices (frozen rule,
    INCLUDING any zero-v coords, whose denominator is exactly eps). Pass 1
    counts coords; pass 2 takes per-tensor topk == global k, then merges."""
    total_q = 0
    for name, slot in named_state:
        for gname, _ in f1.classify_groups(name):
            if gname == "attn_q":
                total_q += slot["exp_avg_sq"][:f1.ATT_D].numel()
    k = max(1, int(k_frac * total_q))
    cands = []
    for name, slot in named_state:
        for gname, _ in f1.classify_groups(name):
            if gname != "attn_q":
                continue
            v_s = slot["exp_avg_sq"][:f1.ATT_D]
            vh = f1.vhat1(v_s.to(torch.float64), slot["step"], field.beta2)
            denom = torch.sqrt(vh) + field.eps
            kt = min(k, denom.numel())
            vals, idx = torch.topk(denom.reshape(-1), kt, largest=False)
            cands.append((name, vals, idx))
            del denom, vh
    allv = torch.cat([c[1] for c in cands])
    order = torch.argsort(allv)[:k]
    bounds = []
    off = 0
    for name, vals, idx in cands:
        bounds.append((off, off + vals.numel(), name, idx))
        off += vals.numel()
    by_name = {}
    sel = allv[order]
    for pos in order.tolist():
        for lo, hi, name, idx in bounds:
            if lo <= pos < hi:
                by_name.setdefault(name, []).append(int(idx[pos - lo]))
                break
    stats = {"k": k, "total_q_coords": total_q,
             "denom_min": float(sel.min()), "denom_max_selected": float(sel.max())}
    return by_name, stats


def select_embed_rows(named_state):
    slot = next(s for n, s in named_state if n == WTE)
    v = slot["exp_avg_sq"].to(torch.float64)
    row_mean = v.mean(dim=1)
    k = min(1024, v.shape[0])
    tail_vals, tail_ids = torch.topk(row_mean, k, largest=False)
    _, head_ids = torch.topk(row_mean, k, largest=True)
    zero_rows = torch.nonzero((v == 0).any(dim=1)).reshape(-1)
    out = {"embed_tail": tail_ids.tolist(),
           "embed_head_rows": head_ids.tolist(),
           "embed_zero_v_rows": zero_rows.tolist()}
    stats = {"n_tail": k, "n_zero_rows": int(zero_rows.numel()),
             "tail_row_mean_max": float(tail_vals.max()),
             "zero_row_ids_preview": zero_rows[:16].tolist()}
    del v, row_mean
    return out, stats


def sigma_max(field, named_state, iters=8, tol=1e-12):
    """Power iteration on the cached fp32 diagonal (5.9G); returns
    (sigma, iterations_used, dominant info, argmax crosscheck)."""
    diag = {}
    gmax = -1.0
    dom = None
    for name, slot in named_state:
        v32 = slot["exp_avg_sq"]
        parts = []
        for gname, sl in f1.classify_groups(name):
            v_s = (v32[sl] if sl != slice(None) else v32)
            if v_s.numel() == 0:
                continue
            g64 = field.gain(name, v_s, torch.float64)
            lm = float(g64.max().item())
            if lm > gmax:
                flat = int(g64.argmax().item())
                row_off = sl.start if (sl != slice(None) and sl.start is not None) else 0
                row = row_off + (flat // v_s.shape[1] if v_s.dim() == 2 else flat)
                full = slot["exp_avg_sq"]
                v_row = full[row] if full.dim() == 2 else full.reshape(1)
                m_row = slot["exp_avg"][row] if slot["exp_avg"].dim() == 2 else slot["exp_avg"].reshape(1)
                vh_row = f1.vhat1(v_row.to(torch.float64), slot["step"], field.beta2)
                gmax = lm
                dom = {"param": name, "group": gname, "row": int(row),
                       "gain": lm, "local_v_min": float(v_row.min().item()),
                       "local_denom_min": float((torch.sqrt(vh_row) + field.eps).min().item()),
                       "local_m_abs_max": float(m_row.abs().max().item())}
            parts.append(g64.to(torch.float32).reshape(-1))
            del g64
        diag[name] = torch.cat(parts) if len(parts) > 1 else parts[0]
        del parts
    # power iteration
    gen = torch.Generator().manual_seed(20260926)
    v_vec = {n: torch.randn(d.numel(), generator=gen, dtype=torch.float32) for n, d in diag.items()}
    nrm = math.sqrt(sum(float((x * x).sum().item()) for x in v_vec.values()))
    v_vec = {n: x / nrm for n, x in v_vec.items()}
    sigma_prev = None
    used = 0
    for it in range(1, iters + 1):
        used = it
        w_vec = {n: diag[n] * v_vec[n] for n in v_vec}
        nrm = math.sqrt(sum(float((x * x).sum().item()) for x in w_vec.values()))
        sigma = nrm
        v_vec = {n: x / nrm for n, x in w_vec.items()}
        if sigma_prev is not None and abs(sigma - sigma_prev) / max(sigma, 1e-300) < tol:
            break
        sigma_prev = sigma
    pi_dom_name = max(v_vec, key=lambda n: float(v_vec[n].abs().max().item()))
    return sigma, used, dom, {"pi_dominant_param": pi_dom_name,
                              "pi_consistent_with_argmax": bool(
                                  abs(float(v_vec[pi_dom_name].abs().max().item()) - 1.0) < 1e-3)}


def crosscheck_vendored(field, named_state, meta, gen):
    from opsens.g1c_olmo2_tangent import g1c_adamw_tangent_step
    results = {}
    for name in PROBE_TENSORS:
        slot = next(s for n, s in named_state if n == name)
        v = slot["exp_avg_sq"]
        q64 = torch.randn(v.shape, generator=gen, dtype=torch.float64)
        q64 = q64 / q64.norm()
        g_dt = field.gain(name, v, torch.float64)
        expected = -g_dt * q64
        p = torch.nn.Parameter(torch.zeros(v.shape, dtype=torch.float64))
        grp_idx = next(i for i, names in enumerate(meta["group_names"]) if name in names)
        g = meta["groups"][grp_idx]

        class _O:
            pass

        opt = _O()
        opt.param_groups = [{"params": [p], "lr": g["lr"], "betas": g["betas"],
                             "eps": g["eps"], "weight_decay": g["weight_decay"],
                             "amsgrad": False, "maximize": False, "differentiable": False}]
        opt.state = {p: {"step": torch.tensor(float(slot["step"])),
                         "exp_avg": slot["exp_avg"].to(torch.float64),
                         "exp_avg_sq": v.to(torch.float64)}}
        out = g1c_adamw_tangent_step(
            named_parameters=[(name, p)], optimizer=opt,
            clipped_gradient_by_name={name: torch.zeros_like(p)},
            tangent_clipped_gradient_by_name={name: q64},
            tangent_theta_by_name={name: torch.zeros_like(p)},
            tangent_m_by_name={name: torch.zeros_like(p)},
            tangent_v_by_name={name: torch.zeros_like(p)},
        )
        got = out["tangent_theta"][name]
        rel = float((got - expected).norm()) / float(expected.norm())
        results[name] = rel
        del p, opt, got, expected, g_dt, q64
    return results


def main() -> int:
    torch.set_num_threads(min(32, torch.get_num_threads()))
    rows = []
    pi_rows = []
    for step in f1.STEPS:
        chk = f1.verify_checkpoint(step, "optim.pt")
        if not chk["sha_ok"]:
            log(f"step{step} optim.pt not ready; skipped")
            continue
        t0 = time.time()
        payload = f1.load_optim_payload(step)
        named_state, meta = f1.optim_named_state(step, payload)
        field = DiagonalField(named_state, meta)
        gen = torch.Generator().manual_seed(20260926)
        cc = crosscheck_vendored(field, named_state, meta, gen)
        log(f"step{step} vendored-tangent crosscheck rel_errs: {cc}")
        if max(cc.values()) > 1e-10:
            log(f"step{step} CROSSCHECK FAILED; skipping this checkpoint")
            (OUT / "crosscheck_fail.json").write_text(json.dumps(
                {"step": step, "rels": cc}, indent=2))
            del payload, named_state
            continue

        toprisk_coords, toprisk_stats = find_toprisk(field, named_state)
        embed_rows, embed_stats = select_embed_rows(named_state)
        log(f"step{step} toprisk {toprisk_stats}; embed {embed_stats}")

        base = {"step": step, "lr": meta["groups"][0]["lr"]}
        directions = [("random:11", "random_seed_11"), ("random:22", "random_seed_22"),
                      ("grad_aligned", "grad_aligned_mhat"), ("toprisk_q", "toprisk_q"),
                      ("mlp_only", "mlp_only"), ("embed_tail", "embed_tail"),
                      ("embed_head_rows", "embed_head_rows"),
                      ("embed_zero_v_rows", "embed_zero_v_rows")]
        for dtype_name, dtype in DTYPES.items():
            for dkey, dname in directions:
                gen = torch.Generator().manual_seed(20260926)
                a_total, comps, qn = evaluate_direction(
                    field, named_state, dkey, dtype, gen,
                    toprisk_coords=toprisk_coords, embed_rows=embed_rows)
                row = {**base, "precision": dtype_name, "direction": dname,
                       "q_norm": qn, "A_total": a_total,
                       "all_finite": math.isfinite(a_total)}
                for g in MODULE_GROUPS:
                    row[f"A_{g}"] = comps.get(g, 0.0)
                rows.append(row)
                log(f"step{step} {dtype_name}/{dname}: A={a_total:.6g}")
            del dtype
        sigma, used, dom, pi_extra = sigma_max(field, named_state)
        pi_rows.append({**base, "sigma_max": sigma, "iterations": used,
                        **{f"dom_{k}": v for k, v in dom.items()},
                        **pi_extra,
                        "toprisk_k": toprisk_stats["k"],
                        "toprisk_denom_min": toprisk_stats["denom_min"],
                        "embed_n_zero_rows": embed_stats["n_zero_rows"]})
        log(f"step{step} sigma_max={sigma:.6g} dom={dom['param']}/{dom['group']} "
            f"row={dom['row']} ({time.time()-t0:.0f}s)")
        del payload, named_state

    if rows:
        with (OUT / "local_conditioning.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"written {len(rows)} conditioning rows")
    if pi_rows:
        with (OUT / "power_iteration_1b.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pi_rows[0].keys()))
            w.writeheader()
            w.writerows(pi_rows)
        log(f"written {len(pi_rows)} power-iteration rows")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
