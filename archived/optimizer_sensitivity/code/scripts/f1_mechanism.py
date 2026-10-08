#!/usr/bin/env python3
"""F1 MECHANISM CLOSURE runner (frozen scope: configs/f1_mechanism_v01.json).

Parts (CSVs under outputs/f1_mechanism/):
  A  state_channel_amplification.csv — blockwise theta/m/v channel JVP
     amplification at the c=0 anchor, 21 states, fp64, module decomposition.
  B  vfloor_intervention.csv — counterfactual v_tau = max(v, tau^2) on frozen
     states; amplification + sigma_max(tau) + forward one-step difference.
  C  risk_knockout.csv (+ risk_knockout_top64_meta.csv) — R_den / R_der risk
     masking of the gradient channel, top-f in {0.01%,0.1%,1%,5%}; kill-curve,
     f90, top-64 coordinate metadata.
  D  module_localization.csv (+ _stability.csv) — gradient-channel energy share
     per module per state + Kendall-tau stability across scale/seed/phase.
  E  precision_decomposition.csv — fp64/fp32/bf16 sensitivity arithmetic on
     representative 100M states + per-channel FD validation (theta/m additive,
     v log-multiplicative).

All bank access READ-ONLY; forward AdamW rule untouched (manual math follows
torch semantics and is cross-checked against the vendored frozen tangent).
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

import opsens.f0_bank_audit_v01 as f0
import opsens.f1_mechanism_v01 as fm

OUT = fm.OUT
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "f1_mechanism_run.log"

STATES_A = (
    [("v07_5m", s, t) for s in (2030, 2031, 2037) for t in (46, 64, 82)]
    + [("g1a_26m", s, t) for s in (2042, 2043, 2044) for t in (26, 64, 102)]
    + [("g1b_100m", s, t) for s in (2042, 2043, 2044) for t in (26, 64, 102)]
)
STATES_BC = (
    [("v07_5m", 2030, 46), ("v07_5m", 2031, 64), ("v07_5m", 2037, 82),
     ("g1a_26m", 2042, 26), ("g1a_26m", 2042, 64), ("g1a_26m", 2042, 102),
     ("g1b_100m", 2042, 26), ("g1b_100m", 2042, 64), ("g1b_100m", 2042, 102)]
)
STATES_E = [("g1b_100m", 2042, 26), ("g1b_100m", 2042, 64), ("g1b_100m", 2043, 26)]
MODULES = ("embed_tied", "attn_q", "attn_k", "attn_v", "attn_o", "attn_qk_norm",
           "rmsnorm", "mlp_gate", "mlp_up", "mlp_down", "others")
EPS64 = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
EPS32 = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_csv(name: str, rows: list):
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with (OUT / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, restval="")
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {name} ({len(rows)} rows)")


def load(scale, seed, step):
    return f0.load_bank_state(fm.bank_state_path(scale, seed, step))


def out_norms(out):
    return (float(out["tangent_theta_norm"]), float(out["tangent_m_norm"]),
            float(out["tangent_v_norm"]))


def a_full_of(out):
    return math.sqrt(sum(x * x for x in out_norms(out)))


def part_a() -> list:
    rows = []
    for scale, seed, step in STATES_A:
        t0 = time.time()
        state = load(scale, seed, step)
        named, opt = fm.fake_optimizer(state, torch.float64)
        mhat_dir = fm.unit_mhat_direction(named, opt.state)
        for rseed in (11, 22):
            dirs = fm.channel_directions(named, fm.GEN_SEED * 10 + rseed)
            for ch in ("theta", "m", "v"):
                out = fm.run_tangent_state_channels(
                    named, opt,
                    q_theta=dirs["theta"] if ch == "theta" else None,
                    q_m=dirs["m"] if ch == "m" else None,
                    q_v=dirs["v"] if ch == "v" else None)
                shares = fm.modulewise_shares(named, out["tangent_theta"])
                row = {
                    "scale": scale, "seed": seed, "step": step,
                    "direction_random_seed": rseed, "channel": ch,
                    "A_full": a_full_of(out), "A_theta_only": out["tangent_theta_norm"],
                    "dm_out_norm": out["tangent_m_norm"], "dv_out_norm": out["tangent_v_norm"],
                }
                row.update({f"share_theta_out_{m}": shares.get(m, 0.0) for m in MODULES})
                rows.append(row)
        out = fm.run_tangent_state_channels(named, opt, q_m=mhat_dir)
        shares = fm.modulewise_shares(named, out["tangent_theta"])
        row = {
            "scale": scale, "seed": seed, "step": step, "direction_random_seed": -1,
            "channel": "m_mhat_aligned", "A_full": a_full_of(out),
            "A_theta_only": out["tangent_theta_norm"],
            "dm_out_norm": out["tangent_m_norm"], "dv_out_norm": out["tangent_v_norm"],
        }
        row.update({f"share_theta_out_{m}": shares.get(m, 0.0) for m in MODULES})
        rows.append(row)
        del state, named, opt
        log(f"A: {scale}/{seed}/{step} done ({time.time()-t0:.0f}s)")
    return rows


def _sigma_max_G(state, v_override=None):
    hp = f0.group_hyperparams(state)
    best = (-1.0, None, -1)
    for name, idx, slot in f0.named_optimizer_entries(state):
        v = slot["exp_avg_sq"] if v_override is None else v_override[name]
        g = fm.gradient_channel_diagonal(v, float(slot["step"]), hp["lr"],
                                       hp["beta1"], hp["beta2"], hp["eps"])
        mx = float(g.max().item())
        if mx > best[0]:
            best = (mx, name, int(g.argmax().item()))
    return best


def part_b() -> list:
    rows = []
    for scale, seed, step in STATES_BC:
        t0 = time.time()
        state = load(scale, seed, step)
        named, opt = fm.fake_optimizer(state, torch.float64)
        hp = f0.group_hyperparams(state)
        dirs = fm.channel_directions(named, fm.GEN_SEED * 10 + 11)
        mhat_dir = fm.unit_mhat_direction(named, opt.state)
        theta = {n: state["model"][n].detach().to(torch.float64) for n, _ in named}
        entries = {n: (idx, slot) for n, idx, slot in f0.named_optimizer_entries(state)}
        for tau in fm.TAU_GRID:
            t_tau = time.time()
            if tau == 0.0:
                v_over = None
                named_t, opt_t = named, opt
                n_changed = 0
            else:
                v_over = {n: torch.clamp(entries[n][1]["exp_avg_sq"], min=tau * tau)
                          for n, _ in named}
                named_t, opt_t = fm.fake_optimizer(state, torch.float64, v_override=v_over)
                n_changed = sum(int((v_over[n] != entries[n][1]["exp_avg_sq"]).sum().item())
                                for n, _ in named)
            sm, sm_name, sm_flat = _sigma_max_G(state, v_override=v_over)
            tau_rows = []
            for ch in ("m", "v", "m_mhat_aligned"):
                if ch == "m_mhat_aligned":
                    out = fm.run_tangent_state_channels(named_t, opt_t, q_m=mhat_dir)
                else:
                    out = fm.run_tangent_state_channels(
                        named_t, opt_t,
                        q_m=dirs["m"] if ch == "m" else None,
                        q_v=dirs["v"] if ch == "v" else None)
                tau_rows.append({
                    "scale": scale, "seed": seed, "step": step, "tau": tau,
                    "channel": ch, "A_full_tau": a_full_of(out),
                    "A_theta_only_tau": out["tangent_theta_norm"],
                    "sigma_max_tau": sm, "sigma_max_param": sm_name,
                    "n_v_changed_coords": n_changed,
                })
            fd_tot = fd_ref = 0.0
            fd_mod = {}
            for n, p in named:
                idx, slot = entries[n]
                u0 = fm.anchor_update(slot["exp_avg_sq"], slot["exp_avg"], float(slot["step"]),
                                      hp["lr"], hp["beta1"], hp["beta2"], hp["eps"],
                                      hp["weight_decay"], theta[n])
                vv = slot["exp_avg_sq"] if v_over is None else v_over[n]
                u1 = fm.anchor_update(vv, slot["exp_avg"], float(slot["step"]),
                                      hp["lr"], hp["beta1"], hp["beta2"], hp["eps"],
                                      hp["weight_decay"], theta[n])
                d2 = float(torch.sum(((u1 - u0).double() ** 2)).item())
                fd_mod[f0.classify_group(n)] = fd_mod.get(f0.classify_group(n), 0.0) + d2
                fd_tot += d2
                fd_ref += float(torch.sum((u0.double() ** 2)).item())
                del u0, u1
            fwd_rel = math.sqrt(fd_tot) / fd_ref if fd_ref > 0 else 0.0
            for r in tau_rows:
                r["fwd_delta_norm"] = math.sqrt(fd_tot)
                r["fwd_update_norm"] = math.sqrt(fd_ref)
                r["fwd_rel_change"] = fwd_rel
                top_mod = max(fd_mod, key=fd_mod.get)
                r["fwd_top_module"] = top_mod
            rows.extend(tau_rows)
            if v_over is not None:
                del v_over, named_t, opt_t
            log(f"B: {scale}/{seed}/{step} tau={tau:g} ({time.time()-t_tau:.0f}s) "
                f"sigma_max={sm:.4g} fwd_rel={fwd_rel:.3g}")
        del state, named, opt, theta
    return rows


def part_c() -> tuple:
    rows = []
    meta_rows = []
    for scale, seed, step in STATES_BC:
        t0 = time.time()
        state = load(scale, seed, step)
        named, _ = fm.fake_optimizer(state, torch.float64)
        hp = f0.group_hyperparams(state)
        vals_den, vals_der, G_flat, q_flat = [], [], [], []
        param_names, param_offsets = [], [0]
        gen = torch.Generator().manual_seed(fm.GEN_SEED * 10 + 11)
        slot_of = {}
        for name, idx, slot in f0.named_optimizer_entries(state):
            step_t = float(slot["step"])
            v = slot["exp_avg_sq"]
            m = slot["exp_avg"]
            g = fm.gradient_channel_diagonal(v, step_t, hp["lr"], hp["beta1"],
                                         hp["beta2"], hp["eps"])
            dver = fm.v_channel_diagonal(m, v, step_t, hp["lr"], hp["beta1"],
                                         hp["beta2"], hp["eps"])
            bc1 = 1.0 - hp["beta1"] ** (step_t + 1.0)
            r_den = g / (hp["lr"] * (1.0 - hp["beta1"]) / bc1)   # = 1/(sqrt(v1_hat)+eps)
            vals_den.append(r_den.flatten())
            vals_der.append(dver.abs().flatten())
            G_flat.append(g.flatten())
            qd = torch.randn(v.shape, generator=gen, dtype=torch.float64)
            q_flat.append(qd.flatten())
            param_names.append(name)
            param_offsets.append(param_offsets[-1] + v.numel())
            slot_of[name] = slot
        R_den = torch.cat(vals_den)
        R_der = torch.cat(vals_der)
        G_all = torch.cat(G_flat)
        q_all = torch.cat(q_flat)
        q_all = q_all / q_all.norm()
        contrib = (G_all * q_all) ** 2
        total_contrib = float(contrib.sum().item())
        n_coords = G_all.numel()
        sigma_orig = float(G_all.max().item())
        del vals_den, vals_der, G_flat, q_flat

        def locate(pos):
            import bisect
            i = bisect.bisect_right(param_offsets, pos) - 1
            return param_names[i], pos - param_offsets[i]

        order_den = torch.argsort(R_den, descending=True)
        order_der = torch.argsort(R_der, descending=True)
        base = {"scale": scale, "seed": seed, "step": step, "n_coords": n_coords,
                "sigma_max_original": sigma_orig}
        for risk_name, order, Rv in (("R_den", order_den, R_den), ("R_der", order_der, R_der)):
            Gs = G_all[order]
            cs = contrib[order]
            cum = torch.cumsum(cs, 0) / total_contrib
            for f in fm.KNOCKOUT_FRACTIONS:
                k = max(1, int(f * n_coords))
                sigma_masked = float(Gs[k:].max().item()) if k < n_coords else 0.0
                rows.append({**base, "risk": risk_name, "kind": "knockout",
                             "fraction": f, "k_coords": k,
                             "sigma_max_masked": sigma_masked,
                             "sigma_reduction": 1.0 - sigma_masked / sigma_orig,
                             "contrib_share_masked": float(cum[k - 1].item()),
                             "A_random_masked": math.sqrt(float(cs[k:].sum().item())),
                             "A_random_original": math.sqrt(total_contrib)})
            for f in fm.KILLCURVE_FRACTIONS:
                k = max(1, int(f * n_coords))
                rows.append({**base, "risk": risk_name, "kind": "killcurve",
                             "fraction": f, "k_coords": k,
                             "sigma_max_masked": float(Gs[k:].max().item()) if k < n_coords else 0.0,
                             "contrib_share_masked": float(cum[k - 1].item()),
                             "A_random_masked": math.sqrt(float(cs[k:].sum().item())),
                             "A_random_original": math.sqrt(total_contrib)})
            f90_idx = int(torch.searchsorted(cum, torch.tensor(0.9)).item()) + 1
            rows.append({**base, "risk": risk_name, "kind": "f90",
                         "fraction": f90_idx / n_coords, "k_coords": f90_idx,
                         "contrib_share_masked": 0.9})
            for pos in order[:64].tolist():
                name, flat = locate(pos)
                slot = slot_of[name]
                meta_rows.append({
                    "scale": scale, "seed": seed, "step": step, "risk": risk_name,
                    "param": name, "module": f0.classify_group(name),
                    "flat_idx": flat,
                    "v": float(slot["exp_avg_sq"].flatten()[flat].item()),
                    "m": float(slot["exp_avg"].flatten()[flat].item()),
                    "m_hat": float(slot["exp_avg"].flatten()[flat].item()),
                    "gradient": "NOT_STORED",
                    "G": float(G_all[pos].item()),
                    "R_value": float(Rv[pos].item()),
                })
        del state, named, R_den, R_der, G_all, q_all, contrib, order_den, order_der
        log(f"C: {scale}/{seed}/{step} done ({time.time()-t0:.0f}s)")
    return rows, meta_rows


def part_d(a_rows: list) -> tuple:
    rows = []
    states = sorted({(r["scale"], int(r["seed"]), int(r["step"])) for r in a_rows})
    shares = {}
    for scale, seed, step in states:
        state = load(scale, seed, step)
        hp = f0.group_hyperparams(state)
        acc = {}
        total = 0.0
        for name, idx, slot in f0.named_optimizer_entries(state):
            g = fm.gradient_channel_diagonal(slot["exp_avg_sq"], float(slot["step"]),
                                       hp["lr"], hp["beta1"], hp["beta2"],
                                       hp["eps"])
            s = float(torch.sum((g.double() ** 2).sum()).item())
            acc[f0.classify_group(name)] = acc.get(f0.classify_group(name), 0.0) + s
            total += s
        shares[(scale, seed, step)] = {k: v / total for k, v in acc.items()}
        del state
    mods = [m for m in MODULES if any(m in s for s in shares.values())]
    for (scale, seed, step), sh in sorted(shares.items()):
        row = {"scale": scale, "seed": seed, "step": step}
        row.update({f"G2_share_{m}": sh.get(m, 0.0) for m in mods})
        row["top1_module"] = max(sh, key=sh.get)
        rows.append(row)
    stab = []

    def add_pair(sa, sb, label):
        v1, v2 = shares.get(sa), shares.get(sb)
        if not (v1 and v2):
            return
        x = [v1.get(m, 0.0) for m in mods]
        y = [v2.get(m, 0.0) for m in mods]
        stab.append({"pair": label, "kendall_tau": fm.kendall_tau(x, y),
                     "top1_agree": max(v1, key=v1.get) == max(v2, key=v2.get)})

    for seed in (2042, 2043, 2044):
        for step in (26, 64, 102):
            add_pair(("g1a_26m", seed, step), ("g1b_100m", seed, step),
                     f"26m|100m:{seed}:{step}")
    for scale in ("v07_5m", "g1a_26m", "g1b_100m"):
        for seed in sorted({s for (sc, s, _) in shares if sc == scale}):
            for s1, s2 in ((26, 64), (26, 102), (64, 102)) if scale != "v07_5m" \
                    else ((46, 64), (46, 82), (64, 82)):
                add_pair((scale, seed, s1), (scale, seed, s2), f"{scale}:{seed}:{s1}|{s2}")
    return rows, stab


def part_e() -> list:
    rows = []
    for scale, seed, step in STATES_E:
        t0 = time.time()
        state = load(scale, seed, step)
        hp = f0.group_hyperparams(state)
        entries = {n: (i, s) for n, i, s in f0.named_optimizer_entries(state)}
        theta = {n: state["model"][n].detach().to(torch.float64) for n in entries}
        for dtype_name, dtype in (("fp64", torch.float64), ("fp32", torch.float32),
                                  ("bf16", torch.bfloat16)):
            named, opt = fm.fake_optimizer(state, dtype)
            dirs = fm.channel_directions(named, fm.GEN_SEED * 10 + 11)
            def cast_dir(d):
                return {k: v.to(dtype) for k, v in d.items()}
            for ch in ("theta", "m", "v"):
                out = fm.run_tangent_state_channels(
                    named, opt,
                    q_theta=cast_dir(dirs["theta"]) if ch == "theta" else None,
                    q_m=cast_dir(dirs["m"]) if ch == "m" else None,
                    q_v=cast_dir(dirs["v"]) if ch == "v" else None)
                fin = all(bool(torch.isfinite(t).all().item())
                          for t in out["tangent_theta"].values())
                rows.append({"scale": scale, "seed": seed, "step": step,
                             "precision": dtype_name, "channel": ch, "kind": "amplification",
                             "A_full": a_full_of(out), "A_theta_only": out["tangent_theta_norm"],
                             "all_finite": fin})
            del named, opt
        probe_names = [n for n in entries if n in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.embed_tokens.weight")]
        for name in probe_names:
            idx, slot = entries[name]
            m64 = slot["exp_avg"].to(torch.float64)
            v64 = slot["exp_avg_sq"].to(torch.float64)
            th64 = theta[name]
            step_t = float(slot["step"])
            gen = torch.Generator().manual_seed(fm.GEN_SEED)
            q = torch.randn(v64.shape, generator=gen, dtype=torch.float64)
            q = q / q.norm()
            bc1 = 1.0 - hp["beta1"] ** (step_t + 1.0)
            g = fm.gradient_channel_diagonal(v64, step_t, hp["lr"], hp["beta1"],
                                         hp["beta2"], hp["eps"])
            dv = fm.v_channel_diagonal(m64, v64, step_t, hp["lr"], hp["beta1"],
                                       hp["beta2"], hp["eps"])

            def upd(theta_v, m_v, v_v):
                mhat = hp["beta1"] * m_v / bc1
                vhat = hp["beta2"] * v_v / (1.0 - hp["beta2"] ** (step_t + 1.0))
                return theta_v * (1 - hp["lr"] * hp["weight_decay"]) \
                    - hp["lr"] * mhat / (vhat.sqrt() + hp["eps"])

            def cmp_row(prec, ch, eps, fd, an):
                fn, an_ = float(fd.norm()), float(an.norm())
                rel = abs(fn - an_) / an_ if an_ > 0 else (0.0 if fn == 0 else float("inf"))
                cos = float((fd * an).sum() / (fn * an_)) if fn > 0 and an_ > 0 else float("nan")
                return {"scale": scale, "seed": seed, "step": step, "precision": prec,
                        "channel": ch, "param": name, "eps": eps, "kind": "fd",
                        "rel_err": rel, "cosine": cos}

            an_th = (1 - hp["lr"] * hp["weight_decay"]) * q
            an_m = -fm.m_channel_diagonal(v64, step_t, hp["lr"], hp["beta1"],
                                      hp["beta2"], hp["eps"]) * q
            an_v = dv * v64 * q
            for prec, eps_grid, castf in (("fp64_manual", EPS64, lambda x: x.double()),
                                          ("fp32_manual", EPS32, lambda x: x.float())):
                th_c, m_c, v_c, q_c = castf(th64), castf(m64), castf(v64), castf(q)
                an_c = {"theta": castf(an_th), "m": castf(an_m), "v_log": castf(an_v)}
                for eps in eps_grid:
                    fd = (upd(th_c + eps * q_c, m_c, v_c) - upd(th_c - eps * q_c, m_c, v_c)) / (2 * eps)
                    rows.append(cmp_row(prec, "theta", eps, fd.double(), an_c["theta"].double()))
                    fd = (upd(th_c, m_c + eps * q_c, v_c) - upd(th_c, m_c - eps * q_c, v_c)) / (2 * eps)
                    rows.append(cmp_row(prec, "m", eps, fd.double(), an_c["m"].double()))
                    fd = (upd(th_c, m_c, v_c * torch.exp(eps * q_c))
                          - upd(th_c, m_c, v_c * torch.exp(-eps * q_c))) / (2 * eps)
                    rows.append(cmp_row(prec, "v_log", eps, fd.double(), an_c["v_log"].double()))
            del m64, v64, th64, g, dv, q
        del state, theta
        log(f"E: {scale}/{seed}/{step} done ({time.time()-t0:.0f}s)")
    return rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDE")
    args = ap.parse_args()
    torch.set_num_threads(min(32, torch.get_num_threads()))
    t_start = time.time()
    log(f"F1 MECHANISM CLOSURE start (parts={args.parts}; frozen: configs/f1_mechanism_v01.json)")
    a_rows = []
    if "A" in args.parts:
        a_rows = part_a()
        write_csv("state_channel_amplification.csv", a_rows)
    if "B" in args.parts:
        write_csv("vfloor_intervention.csv", part_b())
    if "C" in args.parts:
        c_rows, c_meta = part_c()
        write_csv("risk_knockout.csv", c_rows)
        write_csv("risk_knockout_top64_meta.csv", c_meta)
    if "D" in args.parts:
        if not a_rows:
            import csv as _csv
            with (OUT / "state_channel_amplification.csv").open(newline="") as fh:
                a_rows = list(_csv.DictReader(fh))
        d_rows, d_stab = part_d(a_rows)
        write_csv("module_localization.csv", d_rows)
        write_csv("module_localization_stability.csv", d_stab)
    if "E" in args.parts:
        write_csv("precision_decomposition.csv", part_e())
    log(f"F1 mechanism runner complete ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
