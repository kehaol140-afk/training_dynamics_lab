#!/usr/bin/env python3
"""F0 Phase 4: local sensitivity conditioning of the one-step AdamW map at
frozen bank checkpoints (READ-ONLY; no trajectory advance, no model forward).

Map under test (frozen vendored adamw_tangent_step at the c=0 anchor):
  gradient channel q -> delta theta' with per-coordinate exact diagonal
  G_i = lr*(1-beta1)/((1-beta1^(t+1))*(sqrt(vhat1_i)+eps)), vhat1=beta2*v/bc2.
Also reports the m-channel (adjoint-relevant) gain beta1*G_i and the benign
theta-channel factor (1-lr*wd).

Direction classes (frozen in configs/f0_bank_audit_v01.json):
  random x3, grad-aligned (m-hat), embed_tail_only, embed_head_only, mlp_only.

Precision variants: float64 reference, float32, bfloat16 (arithmetic of the
linearization; forward AdamW rule untouched).

Power iteration on the gradient channel (exact diagonal at this anchor) gives
sigma_max and identifies the dominant coordinate (parameter/token/group).

Outputs: outputs/f0_existing_bank_audit/sensitivity_conditioning.csv,
         power_iteration.csv, sensitivity_summary.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import opsens.f0_bank_audit_v01 as f0
from opsens.f0_bank_audit_v01 import classify_group

CONFIG = json.loads((ROOT / "configs/f0_bank_audit_v01.json").read_text())
OUT = ROOT / "outputs/f0_existing_bank_audit"
OUT.mkdir(parents=True, exist_ok=True)

DTYPES = {"float64_reference": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def build_directions(named, opt_state, embed_name, tail_ids, head_ids, gen):
    """Yield (name, unit-direction dict) lazily to keep peak memory low."""
    def rand_unit(mask_fn):
        q = {}
        sq = 0.0
        for name, p in named:
            t = mask_fn(name, p)
            if t is None:
                q[name] = torch.zeros_like(p, dtype=torch.float64)
                continue
            sq += float(torch.sum(t * t).item())
            q[name] = t
        n = math.sqrt(sq)
        return {k: v / n for k, v in q.items()}

    def full(name, p):
        return torch.randn(p.shape, generator=gen, dtype=torch.float64)

    for s in (11, 22, 33):
        yield f"random_seed_{s}", rand_unit(full)

    m = opt_state
    qs = {}
    sq = 0.0
    for name, p in named:
        t = m[p]["exp_avg"].detach().to(torch.float64)
        sq += float(torch.sum(t * t).item())
        qs[name] = t
    n = math.sqrt(sq)
    yield "grad_aligned_mhat", {k: v / n for k, v in qs.items()}

    def embed_rows(ids):
        def fn(name, p):
            if name != embed_name:
                return None
            t = torch.zeros(p.shape, dtype=torch.float64)
            if ids is not None and ids.numel():
                t[ids] = torch.randn((ids.numel(), p.shape[1]), generator=gen, dtype=torch.float64)
            return t
        return fn

    yield "embed_tail_only", rand_unit(embed_rows(tail_ids))
    yield "embed_head_only", rand_unit(embed_rows(head_ids))

    def mlp_mask(name, p):
        if name.startswith("model.layers.") and ".mlp." in name:
            return torch.randn(p.shape, generator=gen, dtype=torch.float64)
        return None

    yield "mlp_only", rand_unit(mlp_mask)


def main() -> None:
    t0 = time.time()
    freq = f0.token_frequency_bins()
    row_bins = f0.embedding_row_bins(freq)
    bin_of = {}
    for b in ("head", "mid", "tail", "never"):
        for i in freq[b]:
            bin_of[int(i)] = b

    rows = []
    pi_rows = []
    state_reports = []

    for scale, cells in CONFIG["sensitivity_states"].items():
        spec = CONFIG["banks"][scale]
        for seed, step in cells:
            p = Path(spec["root"]) / spec["state_glob"].format(seed=seed, step=step)
            state = f0.load_bank_state(p)
            hp = f0.group_hyperparams(state)
            embed_name = "model.embed_tokens.weight"
            step_t = None
            tail_ids = head_ids = None
            for name, idx, slot in f0.named_optimizer_entries(state):
                if step_t is None:
                    step_t = float(slot["step"])
                if name == embed_name:
                    tail_ids = row_bins["tail"]
                    head_ids = row_bins["head"]
            base = {"scale": scale, "seed": seed, "step": step, "lr": hp["lr"], "eps": hp["eps"],
                    "optimizer_step_t": step_t}

            for dtype_name, dtype in DTYPES.items():
                named, opt = f0.fake_optimizer_from_bank(state, dtype)
                gen = torch.Generator().manual_seed(20260926)
                for dname, q in build_directions(named, opt.state, embed_name, tail_ids, head_ids, gen):
                    qd = {k: v.to(dtype) for k, v in q.items()}
                    del q
                    qnorm = f0.total_norm(qd)
                    out = f0.run_tangent_at_null_gradient(named, opt, qd)
                    tan = out["tangent_theta"]
                    groups = f0.norm_by_group(named, tan)
                    finite = all(torch.isfinite(t).all().item() for t in tan.values())
                    rows.append({
                        **base, "precision": dtype_name, "direction": dname,
                        "q_norm": qnorm,
                        "A_total": out["tangent_theta_norm"] / qnorm if qnorm > 0 else float("nan"),
                        "A_embed": groups.get("embed_tied", 0.0) / qnorm,
                        "A_mlp_gate": groups.get("mlp_gate", 0.0) / qnorm,
                        "A_mlp_up": groups.get("mlp_up", 0.0) / qnorm,
                        "A_mlp_down": groups.get("mlp_down", 0.0) / qnorm,
                        "A_attn": (groups.get("attn_q", 0) + groups.get("attn_k", 0) + groups.get("attn_v", 0) + groups.get("attn_o", 0)) / qnorm,
                        "A_rmsnorm_qknorm": (groups.get("rmsnorm", 0) + groups.get("attn_qk_norm", 0)) / qnorm,
                        "all_finite": finite,
                    })
                    del tan, qd, out, groups

                if dtype_name == "float64_reference":
                    vk = {name: torch.randn(pp.shape, generator=gen, dtype=torch.float64) for name, pp in named}
                    nrm = f0.total_norm(vk)
                    vk = {k: (x / nrm).to(dtype) for k, x in vk.items()}
                    sigma_prev = None
                    it = 0
                    for it in range(1, 9):
                        out = f0.run_tangent_at_null_gradient(named, opt, vk)
                        tan = out["tangent_theta"]
                        nrm = f0.total_norm(tan)
                        vk = {k: (x / nrm).to(dtype) for k, x in tan.items()}
                        if sigma_prev is not None and abs(nrm - sigma_prev) / max(nrm, 1e-300) < 1e-12:
                            break
                        sigma_prev = nrm
                    argmax = {k: (x.to(torch.float64).abs().max().item(), int(x.to(torch.float64).abs().argmax().item())) for k, x in vk.items()}
                    dom_name = max(argmax, key=lambda k: argmax[k][0])
                    dom_val, dom_flat = argmax[dom_name]
                    tok_id = tok_count = tok_bin = None
                    if dom_name == embed_name:
                        tok_id = int(dom_flat // vk[dom_name].shape[1])
                        tok_count = int(freq["counts"][tok_id].item())
                        tok_bin = bin_of[tok_id]
                    pi_rows.append({
                        **base, "sigma_max": sigma_prev, "iterations": it,
                        "dominant_param": dom_name, "dominant_group": classify_group(dom_name),
                        "dominant_coord_flat": dom_flat, "dominant_token_id": tok_id,
                        "dominant_token_count": tok_count, "dominant_token_bin": tok_bin,
                    })
                del named, opt
            state_reports.append({**base, "done_s": round(time.time() - t0, 1)})
            print(f"[{scale}] {seed}/{step} done ({time.time()-t0:.0f}s)", flush=True)
            del state

    with open(OUT / "sensitivity_conditioning.csv", "w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "power_iteration.csv", "w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(pi_rows[0].keys()))
        w.writeheader()
        w.writerows(pi_rows)
    summary = {
        "audit_id": "F0_PHASE4_SENSITIVITY_V01",
        "anchor": "c=0 (null next-step gradient); gradient channel q -> theta' exact diagonal at anchor",
        "m_channel_note": "dm-channel gain = beta1 * G_i (same denominators); theta-channel factor = 1-lr*wd (benign)",
        "not_estimated": CONFIG["not_estimated_this_round"],
        "states": state_reports,
        "n_rows": len(rows),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {len(rows)} rows, wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
