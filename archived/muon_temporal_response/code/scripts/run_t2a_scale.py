"""T2a single-state real-scale calibration (round 2, stage B).

Executes configs/t2a_scale_v1.json exactly:
  1. restore anchor model, run 10 production-semantic gradient batches (no step()),
  2. per-batch independent H from the shared checkpoint momentum M_*,
  3. H_ref from first-5 (fp64 mean -> fp32), write calibration_lock.json,
  4. three-path NS on the same captured fp32 H (fp64 / fp32 no-bf16 / native bf16),
  5. natural outputs + alpha-grid finite-amplitude diagnostics with JVP identity,
  6. summary / map_errors.csv / response_errors.csv / cost_ledger.json.

Zero model updates; dev/eval/confirm untouched; anchor file read-only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition import config as mte_config
from muon_transition.data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
from muon_transition.model import build_model, unique_named_parameters
from muon_transition.mixture import MixtureStream
from src.t0_metrics import decompose_pair

OUT = ROOT / "outputs" / "t2a_scale_v1"
# respect CUDA_VISIBLE_DEVICES remapping: physical free GPU is selected by the launcher
DEVICE = "cuda:0"
BETA = 0.95
NS_A, NS_B, NS_C = 3.4445, -4.775, 2.0315
NS_STEPS, EPS_NORM = 5, 1e-7
MATRICES = [
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.6.self_attn.q_proj.weight",
    "model.layers.6.self_attn.k_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
]
ALPHAS = [0.001, 0.01, 0.05, 0.25, 1.0]

from torch.optim._muon import _zeropower_via_newtonschulz


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ns_torch(h: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Installed-semantics NS in a chosen dtype (no bf16 cast)."""
    x = h.to(dtype).clone()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    n = x.norm().clamp(min=EPS_NORM)
    x = x / n
    for _ in range(NS_STEPS):
        a = x @ x.T
        bu = torch.addmm(a, a, a, beta=NS_B, alpha=NS_C)
        x = torch.addmm(x, bu, x, beta=NS_A)
    if transposed:
        x = x.T
    return x


def jvp_ns(h_ref: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """FP64 directional tangent of the installed-semantics NS at h_ref along e (E2)."""
    h = h_ref.double()
    e = e.double()
    transposed = h.shape[0] > h.shape[1]
    if transposed:
        h, e = h.T, e.T
    n = h.norm()
    x = h / n
    dx = e / n - h * (h * e).sum() / n ** 3
    for _ in range(NS_STEPS):
        a = x @ x.T
        bu = NS_B * a + NS_C * (a @ a)
        da = dx @ x.T + x @ dx.T
        dbu = NS_B * da + NS_C * (da @ a + a @ da)
        dx = NS_A * dx + dbu @ x + bu @ dx
        x = NS_A * x + bu @ x
    if transposed:
        dx = dx.T
    return dx


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    t_start = time.time()
    gpu_t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT / "run_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    cfg = json.loads((ROOT / "configs" / "t2a_scale_v1.json").read_text())
    manifest_blob = json.loads((OUT / "batch_manifest.json").read_text())
    steps10 = manifest_blob["steps"]

    # ---- anchor verification ----
    anchor_path = LAB / "muon_transition_experiment" / "outputs" / "m1" / cfg["anchor"]["run_id"] / "final_state.pt"
    sha = sha256_file(anchor_path)
    assert sha == cfg["anchor"]["final_state_sha256"], f"anchor sha mismatch {sha}"
    payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    assert payload["cursor"]["step"] == 2048
    log(f"anchor verified sha={sha[:16]}...")

    # ---- model restore ----
    model = build_model().to(DEVICE)
    model.load_state_dict(payload["model_state"])
    model.train()  # production loop mode; dropout=0.0 so equivalent to eval for grads
    named = dict(unique_named_parameters(model))
    assert sum(p.numel() for _, p in unique_named_parameters(model)) == 30025216

    # checkpoint momentum M_* for the six matrices
    # optimizer state keys are int indices into the Muon group's param list, whose order
    # equals the muon-route order from route_parameters (verified pre-run).
    from muon_transition.routing import route_parameters
    _, _, route_table = route_parameters(model)
    muon_names_ordered = [t["name"] for t in route_table if t["route"] == "muon"]
    assert len(muon_names_ordered) == 56
    muon_state = payload["optimizer_states"][0]["state"]
    M_star = {}
    for idx, nm in enumerate(muon_names_ordered):
        if nm in MATRICES:
            M_star[nm] = muon_state[idx]["momentum_buffer"].clone()
    assert sorted(M_star) == sorted(MATRICES) and all(M_star[n] is not None for n in MATRICES)
    log("momentum M_* restored for 6 matrices")

    # integrity: parameter hash before collection
    phash_before = {n: tensor_sha(p) for n, p in unique_named_parameters(model)}
    rng_before = torch.get_rng_state()

    # ---- data ----
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    stream = MixtureStream({"steps": steps10}, a_train, b_train)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    all_params = [p for _, p in unique_named_parameters(model)]

    G = {n: [] for n in MATRICES}
    batch_info = []
    for bi in range(10):
        for p in all_params:
            p.grad = None
        micro_losses, n_targets = [], 0
        for mb in stream.micro_batches(bi):
            batch = mb.to(DEVICE)
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                targets = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1), reduction="mean")
            (loss / mte_config.GRAD_ACCUM_STEPS).backward()
            micro_losses.append(float(loss.detach()))
            n_targets += int(targets.numel())
        preclip = torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.grad is not None], 1.0)
        coef = float(min(1.0, 1.0 / preclip)) if preclip > 0 else 1.0
        for n in MATRICES:
            G[n].append(named[n].grad.detach().clone())
        batch_info.append({"batch": bi + 1, "preclip_global_norm": float(preclip),
                           "clip_coef": coef, "clipped": bool(preclip > 1.0),
                           "valid_targets": n_targets, "train_loss_mean_micro": float(np.mean(micro_losses))})
        log(f"batch {bi+1}: preclip_norm={float(preclip):.4f} clip_coef={coef:.6f} "
            f"targets={n_targets} loss={np.mean(micro_losses):.4f} "
            f"({time.time()-t_start:.0f}s)")

    phash_after = {n: tensor_sha(p) for n, p in unique_named_parameters(model)}
    assert phash_before == phash_after, "model parameters changed during collection!"
    assert torch.equal(rng_before, torch.get_rng_state()), "torch RNG advanced"
    log("model params + torch RNG unchanged after collection (hash equal)")

    # ---- H construction (per-batch independent, production lerp order, fp32) ----
    H = {n: [] for n in MATRICES}
    for n in MATRICES:
        m_star = M_star[n].to(DEVICE)
        for bi in range(10):
            g = G[n][bi].to(DEVICE)
            m_i = m_star.lerp(g, 1.0 - BETA)     # beta*M_* + (1-beta)*G
            h_i = g.lerp(m_i, BETA)              # (1-beta)*G + beta*M_i
            # fp64 math check
            chk = BETA ** 2 * m_star.double() + (1 - BETA ** 2) * g.double()
            assert float((h_i.double() - chk).abs().max()) < 1e-6, "H math check failed"
            H[n].append(h_i)
        G[n] = torch.stack(G[n]).cpu()  # save post-clip G fp32

    # ---- H_ref from first 5 (fp64 mean -> single fp32 cast), lock ----
    H_ref = {}
    for n in MATRICES:
        stack5 = torch.stack([H[n][i] for i in range(5)]).double()
        H_ref[n] = stack5.mean(dim=0).float()

    lock = {"written_after": "first-5 calibration outputs computed below; check-half labels untouched",
            "H_ref_sha256": {n: tensor_sha(H_ref[n]) for n in MATRICES},
            "rule": "H_ref = fp64 mean of first-5 H_i cast once to fp32; "
                    "alpha grid {0.001,0.01,0.05,0.25,1.0} x {+,-} on natural direction E_i = H_i - H_ref; "
                    "alpha=+1 forced to saved original H_i; rel-residual denominator max(||delta_native||,||L||)",
            "code_files_sha256": {}, "natural_rel_offsets_first5": {}}
    for f in ("scripts/run_t2a_scale.py", "src/t0_metrics.py"):
        lock["code_files_sha256"][f] = sha256_file(ROOT / f)
    for n in MATRICES:
        offs = [float((H[n][i] - H_ref[n]).norm() / H_ref[n].norm().clamp(min=1e-30))
                for i in range(5)]
        lock["natural_rel_offsets_first5"][n] = offs
    json.dump(lock, open(OUT / "calibration_lock.json", "w"), indent=1)
    log("calibration_lock.json written")

    # ---- three paths on natural points ----
    tf32_state = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 diagnostics must not use TF32

    def three_paths(h32: torch.Tensor):
        d64 = ns_torch(h32, torch.float64)
        d32 = ns_torch(h32, torch.float32)
        dnat = _zeropower_via_newtonschulz(h32, (NS_A, NS_B, NS_C), NS_STEPS, EPS_NORM)
        return d64, d32, dnat.float()

    D_ref = {n: three_paths(H_ref[n]) for n in MATRICES}
    D_nat = {n: [] for n in MATRICES}   # per batch: (d64, d32, dnat)
    for n in MATRICES:
        for bi in range(10):
            D_nat[n].append(three_paths(H[n][bi]))
        H[n] = torch.stack([h.cpu() for h in H[n]])
        H_ref[n] = H_ref[n].cpu()

    # determinism repetition: H_ref, batch1, batch6
    rep = {}
    for tag, hsrc in (("H_ref", {n: H_ref[n].to(DEVICE) for n in MATRICES}),
                      ("batch1", {n: H[n][0].to(DEVICE) for n in MATRICES}),
                      ("batch6", {n: H[n][5].to(DEVICE) for n in MATRICES})):
        rep[tag] = {}
        for n in MATRICES:
            d64, d32, dnat = three_paths(hsrc[n])
            if tag == "H_ref":
                base = D_ref[n]
            elif tag == "batch1":
                base = D_nat[n][0]
            else:
                base = D_nat[n][5]
            rep[tag][n] = {"d64_bitwise": bool(torch.equal(d64, base[0])),
                           "d32_bitwise": bool(torch.equal(d32, base[1])),
                           "dnat_bitwise": bool(torch.equal(dnat, base[2]))}
    log("repetition check done: " + json.dumps(
        {t: all(v["d64_bitwise"] and v["d32_bitwise"] and v["dnat_bitwise"] for v in d.values())
         for t, d in rep.items()}))

    # ---- alpha-grid diagnostics ----
    map_rows, resp_rows = [], []
    sample_l1q = {}  # layer-1 q_proj batch1/batch6 full arrays for review bundle
    for n in MATRICES:
        h_ref_d = H_ref[n].to(DEVICE)
        is_sample = n == MATRICES[0]
        for bi in range(10):
            h_i_orig = H[n][bi].to(DEVICE)
            e32 = h_i_orig - h_ref_d  # natural direction in fp32 (captured difference)
            natural_rel = float(e32.norm() / h_ref_d.norm().clamp(min=1e-30))
            points = [(0.0, None)] + [(a, s) for a in ALPHAS for s in (+1.0, -1.0)]
            for alpha, sign in points:
                if alpha == 0.0:
                    h_in = h_ref_d.clone()
                elif alpha == 1.0 and sign == +1.0:
                    h_in = h_i_orig  # forced saved original, no reconstruct rounding
                else:
                    h_in = (h_ref_d + sign * alpha * e32).float()
                actual_rel = float((h_in - h_ref_d).norm() / h_ref_d.norm().clamp(min=1e-30))
                d64_in, d32_in, dnat_in = three_paths(h_in)
                e_dir = h_in.double() - h_ref_d.double()
                L = jvp_ns(h_ref_d, e_dir)
                d64_ref, d32_ref, dnat_ref = D_ref[n]
                delta_native = dnat_in.float() - dnat_ref.float()
                r_real = d64_in - d64_ref - L
                r_numeric = (dnat_in.float().double() - d64_in) - (dnat_ref.float().double() - d64_ref)
                resid = delta_native.double() - L
                ident_err = float((resid - (r_real + r_numeric)).abs().max())
                nd, nl = float(delta_native.norm()), float(L.norm())
                denom = max(nd, nl)
                cos = (float(torch.sum(delta_native.double() * L) / (nd * nl))
                       if nd > 0 and nl > 0 else None)
                zero_both = denom == 0.0
                rel_resid = float(resid.norm() / denom) if not zero_both else None
                bf16_changed = float((h_in.bfloat16() != h_ref_d.bfloat16()).float().mean())
                zero_response = bool(torch.equal(dnat_in, dnat_ref))
                row = {"matrix": n, "batch": bi + 1, "half": "cal" if bi < 5 else "check",
                       "alpha": alpha, "sign": ("+" if sign is None or sign > 0 else "-"),
                       "actual_rel_perturb": actual_rel,
                       "natural_rel_offset": natural_rel,
                       "norm_delta_native": nd, "norm_L": nl,
                       "norm_r_real": float(r_real.norm()), "norm_r_numeric": float(r_numeric.norm()),
                       "rel_residual": rel_resid, "zero_both": zero_both,
                       "cos_delta_L": cos, "bf16_changed_frac": bf16_changed,
                       "native_zero_response": zero_response,
                       "identity_max_abs_err": ident_err,
                       "norm_phi32_minus_phi64_in": float((d32_in.double() - d64_in).norm())}
                resp_rows.append(row)
                if is_sample and bi in (0, 5):
                    key = f"b{bi+1}_a{alpha}{'p' if (sign is None or sign>0) else 'm'}"
                    sample_l1q[key] = {"H_in": h_in.cpu(), "L": L.cpu(),
                                       "delta_native": delta_native.cpu(),
                                       "d64": d64_in.cpu(), "d32": d32_in.cpu(),
                                       "dnat": dnat_in.cpu()}
        # map-level natural errors per matrix (three paths, population /10)
        d64s = torch.stack([d[0] for d in D_nat[n]]).cpu().numpy()
        d32s = torch.stack([d[1] for d in D_nat[n]]).cpu().numpy()
        dnats = torch.stack([d[2] for d in D_nat[n]]).cpu().numpy()
        for half, sl in (("cal", slice(0, 5)), ("check", slice(5, 10))):
            dec64_32 = decompose_pair(d64s[sl], d32s[sl])
            dec64_nat = decompose_pair(d64s[sl], dnats[sl])
            h_arr = H[n].numpy()
            mean_drift = float(np.linalg.norm(h_arr[sl].mean(axis=0) - H_ref[n].numpy()) /
                               np.linalg.norm(H_ref[n].numpy()))
            for path, dec in (("fp32_vs_fp64", dec64_32), ("native_vs_fp64", dec64_nat)):
                map_rows.append({"matrix": n, "half": half, "comparison": path,
                                 "A_ref": dec["A_ref"], "A_q": dec["A_q"], "cross2": dec["cross2"],
                                 "A_second": dec["A_native"], "mean_q": dec["mean_q"],
                                 "identity_ok": dec["identity_ok"],
                                 "input_mean_drift_rel": mean_drift})
            for path, arr in (("fp64", d64s), ("fp32", d32s), ("native", dnats)):
                mean_out = arr[sl].mean(axis=0)
                ref_out = (D_ref[n][0].cpu().numpy() if path == "fp64" else
                           D_ref[n][1].cpu().numpy() if path == "fp32" else
                           D_ref[n][2].cpu().numpy())
                map_rows.append({"matrix": n, "half": half, "comparison": f"output_stats::{path}",
                                 "A_ref": float(np.mean(np.sum((arr[sl] - arr[sl].mean(axis=0)[None]) ** 2, axis=(1, 2)))),
                                 "A_q": None, "cross2": None, "A_second": None, "mean_q": None,
                                 "identity_ok": None,
                                 "input_mean_drift_rel": mean_drift,
                                 "output_mean_shift_sq": float(np.sum((mean_out - ref_out) ** 2)),
                                 "output_mean_shift_rel": float(np.linalg.norm(mean_out - ref_out) /
                                                                 np.linalg.norm(ref_out))})

    torch.backends.cuda.matmul.allow_tf32 = tf32_state
    gpu_seconds = time.time() - gpu_t0

    # ---- save raw arrays ----
    raw = {"G_post_clip": {n: G[n] for n in MATRICES},
           "M_star": {n: M_star[n].cpu() for n in MATRICES},
           "H_ref": {n: H_ref[n].cpu() for n in MATRICES},
           "H_i": {n: H[n] for n in MATRICES},
           "D_ref": {n: tuple(d.cpu() for d in D_ref[n]) for n in MATRICES},
           "D_i": {n: [tuple(d.cpu() for d in tup) for tup in D_nat[n]] for n in MATRICES},
           "batch_info": batch_info, "repetition": rep}
    torch.save(raw, OUT / "raw_natural.pt")
    torch.save(sample_l1q, OUT / "raw_sample_l1q.pt")

    import csv
    map_fields = ["matrix", "half", "comparison", "A_ref", "A_q", "cross2", "A_second",
                  "mean_q", "identity_ok", "input_mean_drift_rel",
                  "output_mean_shift_sq", "output_mean_shift_rel"]
    with open(OUT / "map_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=map_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(map_rows)
    with open(OUT / "response_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resp_rows[0].keys()))
        w.writeheader(); w.writerows(resp_rows)

    summary = {"anchor": cfg["anchor"]["run_id"], "anchor_sha256": sha,
               "batch_info": batch_info,
               "natural_rel_offsets": {n: [float((H[n][i] - H_ref[n]).norm() / H_ref[n].norm()) for i in range(10)]
                                       for n in MATRICES},
               "repetition_all_bitwise": {t: all(v["d64_bitwise"] and v["d32_bitwise"] and v["dnat_bitwise"]
                                                 for v in d.values()) for t, d in rep.items()},
               "matrix_elements_actual": {n: int(np.prod(H[n].shape[1:])) for n in MATRICES},
               "identity_max_abs_err_over_all_points": max(r["identity_max_abs_err"] for r in resp_rows),
               "n_response_points": len(resp_rows),
               "model_updates": 0, "eval_access": 0,
               "gpu_seconds": gpu_seconds, "wall_seconds": time.time() - t_start}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)

    import os
    def dirsize(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    ledger = {"gpu_wall_seconds": gpu_seconds, "budget_gpu_seconds": 1800,
              "batches_used": 10, "micro_backward_count": 40,
              "model_updates": 0, "eval_or_confirm_access": 0,
              "output_bytes_t2a": dirsize(OUT),
              "cpu_postprocess_seconds": 0.0}
    json.dump(ledger, open(OUT / "cost_ledger.json", "w"), indent=1)
    log(f"T2a done: gpu={gpu_seconds:.1f}s wall={summary['wall_seconds']:.1f}s "
        f"outputs={dirsize(OUT)/1e6:.1f}MB points={len(resp_rows)}")


if __name__ == "__main__":
    main()
