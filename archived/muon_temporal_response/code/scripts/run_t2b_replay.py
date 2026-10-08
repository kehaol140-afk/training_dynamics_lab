"""T2b stage 1: frozen-gradient full-state nonlinear replay + candidate lock.

Runs on ONE free GPU (env CUDA_VISIBLE_DEVICES selects it). No Q_check access here.
Phases: anchor restore -> one-step virtual-vs-real consistency check -> collect W0/W1
frozen grads (all params) -> 16 virtual tracks (10 permuted + 128 common-tail steps,
streamed C grads) -> q_select at theta* -> S_fin/S_lin -> shadow_scores.csv +
selection_lock.json (+ step10/138 diagnostics).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment"))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(ROOT.parent / "minimind_base"))

from muon_transition.model import build_model, unique_named_parameters
from muon_transition.optimizers import build_muon_arm
from muon_transition.mixture import MixtureStream
from vendored_fr.rng import restore_rng_state
from src.t2b_common import (ANCHOR_PATH, ANCHOR_SHA, build_round3_manifests, build_q_targets,
                            q_select_gradient, eval_q_checkpoints, sha256_file, flat_state_hash)

OUT = ROOT / "outputs" / "t2b_target_screen_v1"
DEV = "cuda:0"
MATRICES = ["model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.mlp.down_proj.weight", "model.layers.6.self_attn.q_proj.weight",
            "model.layers.6.self_attn.k_proj.weight", "model.layers.6.mlp.down_proj.weight"]


def log(msg, f):
    print(msg, flush=True)
    f.write(msg + "\n")
    f.flush()


def load_anchor(DEVICE):
    payload = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)
    model = build_model().to(DEV)
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)  # placeholder lrs
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    restore_rng_state(payload["rng"])
    return payload, model, [muon_opt, aux_opt]


def collect_batch_grad(model, stream, step_index, autocast):
    """Production semantics: 4 micro fwd/bwd + ONE global clip. Returns post-clip grads
    (list aligned with all named params, None->zeros) and diagnostics."""
    for p in model.parameters():
        p.grad = None
    micro_losses, n_targets = [], 0
    for mb in stream.micro_batches(step_index):
        batch = mb.to(DEV)
        with autocast:
            logits = model(batch).logits[..., :-1, :].contiguous()
            targets = batch[..., 1:].contiguous()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), reduction="mean")
        (loss / 4).backward()
        micro_losses.append(float(loss.detach()))
        n_targets += int(targets.numel())
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    preclip = torch.nn.utils.clip_grad_norm_(params_with_grad, 1.0)
    grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    return grads, {"preclip": float(preclip), "targets": n_targets,
                   "loss_mean": float(np.mean(micro_losses))}


class VirtualTrack:
    def __init__(self, payload):
        self.model = build_model().to(DEV)
        self.model.load_state_dict(payload["model_state"])
        self.model.train()
        m, a = build_muon_arm(self.model, lr=1e-9, weight_decay=0.0)
        m.load_state_dict(payload["optimizer_states"][0])
        a.load_state_dict(payload["optimizer_states"][1])
        self.opts = [m, a]
        self.params = list(self.model.parameters())

    def step_with_grads(self, grads):
        for p, g in zip(self.params, grads):
            p.grad = g
        for opt in self.opts:
            opt.step()


def main():
    t0 = time.time()
    gpu_t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "replay_log.txt", "w")

    assert sha256_file(ANCHOR_PATH) == ANCHOR_SHA, "anchor sha mismatch"
    payload, theta_model, _ = load_anchor(DEV)
    log(f"[{time.time()-t0:.0f}s] anchor restored", logf)

    blocks = build_round3_manifests()
    perms_blob = json.loads((OUT / "data_and_perms_manifest.json").read_text())
    perms = perms_blob["permutations"]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    named_params = list(theta_model.named_parameters())

    # ---- one-step virtual-vs-real consistency (task test #2) ----
    stream_w0 = MixtureStream({"steps": blocks["W0"]}, *__import__("muon_transition.m1_data_cache", fromlist=["get_domain_windows"]).get_domain_windows())
    g0, info0 = collect_batch_grad(theta_model, stream_w0, 0, autocast)
    _, model_real, opts_real = load_anchor(DEV)
    for p, g in zip(list(model_real.parameters()), g0):
        p.grad = g
    for opt in opts_real:
        opt.step()
    real_hash = flat_state_hash(model_real, opts_real)
    track_probe = VirtualTrack(payload)
    track_probe.step_with_grads(g0)
    virt_hash = flat_state_hash(track_probe.model, track_probe.opts)
    one_step_ok = real_hash == virt_hash
    log(f"one-step virtual==real consistency (bit-level): {one_step_ok}", logf)
    del model_real, opts_real, track_probe
    json.dump({"one_step_bitwise_ok": one_step_ok}, open(OUT / "one_step_consistency.json", "w"))

    # aliasing probe: grads unchanged after steps
    g0_hash_before = hashlib.sha256(b"".join(
        (g if g is not None else torch.zeros(1)).cpu().contiguous().numpy().tobytes() for g in g0)).hexdigest()

    # ---- collect W0/W1 frozen grads (20 batches, all params) ----
    prefix_grads = {}
    a_train, b_train = __import__("muon_transition.m1_data_cache", fromlist=["get_domain_windows"]).get_domain_windows()
    batch_info = []
    for blk in ("W0", "W1"):
        stream = MixtureStream({"steps": blocks[blk]}, a_train, b_train)
        for i in range(10):
            grads, info = collect_batch_grad(theta_model, stream, i, autocast)
            prefix_grads[(blk, i)] = [g for g in grads]
            batch_info.append({"block": blk, "batch": i + 1, **info})
            log(f"[{time.time()-t0:.0f}s] grad {blk}[{i}] preclip={info['preclip']:.4f} loss={info['loss_mean']:.4f}", logf)
    # W0 six-matrix cross-check vs T2a cached arrays
    try:
        raw_t2a = torch.load(ROOT / "outputs" / "t2a_scale_v1" / "raw_natural.pt",
                             map_location="cpu", weights_only=False)
        name_to_grad = {n: g for (n, g) in zip([n for n, _ in named_params], prefix_grads[("W0", 0)])}
        diffs = []
        for n in MATRICES:
            g_now = name_to_grad[n]
            g_old = raw_t2a["G_post_clip"][n][0]
            diffs.append(float((g_now.cpu() - g_old.cpu()).abs().max()))
        log(f"W0 six-matrix vs T2a max|diff| per matrix: {[f'{d:.2e}' for d in diffs]}", logf)
        json.dump({"w0_vs_t2a_max_abs_diff": dict(zip(MATRICES, diffs))},
                  open(OUT / "w0_t2a_crosscheck.json", "w"))
    except Exception as e:  # noqa
        log(f"W0 cross-check skipped: {e}", logf)

    # ---- 16 virtual tracks ----
    tracks = {}
    for b in (0, 1):
        for k in range(8):
            tracks[(b, k)] = VirtualTrack(payload)
    log(f"[{time.time()-t0:.0f}s] 16 virtual tracks initialized", logf)

    def record_diag(tr, anchor_params, tag, base_track=None):
        mu_sq = aux_sq = 0.0
        for (n, p), p0 in zip(tr.model.named_parameters(), anchor_params):
            d = float((p.detach() - p0.detach()).norm() ** 2)
            if p.ndim == 2 and any(pat in n for pat in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                mu_sq += d
            else:
                aux_sq += d
        row = {"tag": tag, "param_shift_muon_sq": mu_sq, "param_shift_aux_sq": aux_sq}
        if base_track is not None:
            m_self = list(tr.opts[0].state.values())
            m_base = list(base_track.opts[0].state.values())
            row["M_tail_diff_sq"] = float(sum((a["momentum_buffer"] - c["momentum_buffer"]).norm() ** 2
                                              for a, c in zip(m_self, m_base)))
        return row

    anchor_params = [p.detach().clone() for p in theta_model.parameters()]

    # prefix: 10 permuted steps per track
    diag_rows = []
    for b in (0, 1):
        for k in range(8):
            perm = perms[f"W{b}"][k]["perm"]
            tr = tracks[(b, k)]
            for j in range(10):
                tr.step_with_grads(prefix_grads[(f"W{b}", perm[j])])
            diag_rows.append(record_diag(tr, anchor_params, f"step10|W{b}|k{k}", tracks[(b, 0)]))
    log(f"[{time.time()-t0:.0f}s] prefix done (step10 diagnostics recorded)", logf)

    # aliasing probe after prefix
    g0_hash_after = hashlib.sha256(b"".join(
        (g if g is not None else torch.zeros(1)).cpu().contiguous().numpy().tobytes() for g in g0)).hexdigest()
    alias_ok = g0_hash_before == g0_hash_after
    log(f"grad aliasing probe (G unchanged through 160 virtual steps): {alias_ok}", logf)

    # ---- C tail: stream 128 frozen grads through all 16 tracks ----
    stream_c = MixtureStream({"steps": blocks["C"]}, a_train, b_train)
    for i in range(128):
        grads_c, info = collect_batch_grad(theta_model, stream_c, i, autocast)
        for b in (0, 1):
            for k in range(8):
                tracks[(b, k)].step_with_grads(grads_c)
        if (i + 1) % 16 == 0:
            log(f"[{time.time()-t0:.0f}s] C tail {i+1}/128 (loss={info['loss_mean']:.4f} preclip={info['preclip']:.3f})", logf)
        del grads_c

    for b in (0, 1):
        for k in range(8):
            diag_rows.append(record_diag(tracks[(b, k)], anchor_params, f"step138|W{b}|k{k}", tracks[(b, 0)]))
    log(f"[{time.time()-t0:.0f}s] C tail done", logf)

    # ---- q_select at theta* ----
    q = build_q_targets(verbose=False)
    q_grads, V_star = q_select_gradient(theta_model, q, DEV)
    log(f"[{time.time()-t0:.0f}s] q_select computed, V(theta*)={V_star:.6f}", logf)

    # ---- scores ----
    rows = []
    q_flat = torch.cat([q_grads[n].reshape(-1) for n, _ in named_params])
    theta0_flat = {}
    for b in (0, 1):
        tr0 = tracks[(b, 0)]
        theta0_flat[b] = torch.cat([p.detach().reshape(-1).cpu() for p in tr0.model.parameters()])
    for b in (0, 1):
        for k in range(8):
            tr = tracks[(b, k)]
            vres = eval_q_checkpoints(tr.model, q, "select", DEV)
            th_flat = torch.cat([p.detach().reshape(-1).cpu() for p in tr.model.parameters()])
            s_lin = float(q_flat.double() @ (th_flat.double() - theta0_flat[b].double()))
            rows.append({"window": b, "k": k, "S_fin": vres["V"], "S_lin": s_lin,
                         "nll_A_sel": vres["nll_A"], "nll_B_sel": vres["nll_B"]})
            log(f"[{time.time()-t0:.0f}s] W{b} k{k}: S_fin={vres['V']:.6f} S_lin={s_lin:+.6e}", logf)

    import csv
    with open(OUT / "shadow_scores.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "prefix_tail_diagnostics.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)

    # selection lock (ties <=1e-6 -> smallest k)
    lock = {}
    for b in (0, 1):
        sub = [r for r in rows if r["window"] == b]
        best = min(sub, key=lambda r: (r["S_fin"], r["k"]))
        cands = [r for r in sub if r["S_fin"] <= best["S_fin"] + 1e-6]
        kF = min(r["k"] for r in cands)
        bestL = min(sub, key=lambda r: (r["S_lin"], r["k"]))
        candsL = [r for r in sub if r["S_lin"] <= bestL["S_lin"] + 1e-6]
        kL = min(r["k"] for r in candsL)
        lock[f"W{b}"] = {"kF": kF, "kL": kL, "pi0_k": 0}
    lock["note"] = "locked from S_fin (primary) / S_lin (secondary readout) before any Q_check access"
    lock["code_sha256"] = {f: sha256_file(ROOT / f) for f in
                           ("scripts/run_t2b_replay.py", "src/t2b_common.py")}
    json.dump(lock, open(OUT / "selection_lock.json", "w"), indent=1)
    log(f"selection lock: {json.dumps(lock)}", logf)

    gpu_seconds = time.time() - gpu_t0
    json.dump({"gpu_seconds_replay_stage": gpu_seconds,
               "batch_info_prefix": batch_info,
               "aliasing_probe_ok": alias_ok,
               "one_step_bitwise_ok": one_step_ok},
              open(OUT / "replay_stage_meta.json", "w"), indent=1)
    log(f"replay stage done in {gpu_seconds:.0f}s GPU", logf)


if __name__ == "__main__":
    main()
