#!/usr/bin/env python3
"""F2 hypergradient pilot runner (frozen: docs/F2_HYPERGRADIENT_PILOT_PREREG.md).

Per cell (g1b bank state): CRN rollouts over the frozen future-batch manifest.
  FD arms: delta in {0.02,0.05,0.1,0.25,0.5} -> h_FD(delta,H) via L_val(+-delta).
  Baseline (delta=0) rollout carries the u=log(eta) forward tangent in arms
    fp64/B, fp32/B, fp64/C, fp32/C (fp32 HVP) and frozen/C (bf16-autocast HVP);
    per-step order (frozen, mirrors legacy): grads -> clip(no step) -> HVP ->
    clip_jvp -> vendored tangent (reads PRE-step optimizer state) -> opt.step()
    -> add inhomogeneous u-term (theta_next - theta_prev).
  h_tangent(H) = <grad_theta L_val(theta_H), dtheta_H/du>; L_val readouts at
  H in {1,2,4,8,16,32} (fp32 no-autocast forward, frozen eval set of 128).

Run with CUDA_VISIBLE_DEVICES=1.
Usage: python scripts/f2_hypergradient.py --cell 2042:26
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LAB = ROOT.parent
MM = LAB / "minimind_base"
if str(MM) not in sys.path:
    sys.path.insert(0, str(MM))

import torch

import opsens.f0_bank_audit_v01 as f0
import opsens.f1_mechanism_v01 as fm
from opsens.a0c_adamw_response_propagation_v01 import adamw_tangent_step, clip_jvp
from opsens.batch_manifest import make_permutation, manifest_sha256
from opsens.minimind_v03_runtime import stack_batch
from opsens.stream_rng import derive_stream_seed

OUT = ROOT / "outputs" / "f2_hypergradient"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "f2_run.log"

D0 = LAB / "legacy_fluctuation_response/outputs/e3_confirmatory/v07/e3c_subset.jsonl"
PARTITION = LAB / "legacy_fluctuation_response/outputs/e3_confirmatory/v07/partition_v07.json"
BATCH_SIZE = 4
ACCUM = 2
MAX_SEQ = 128
MAX_NORM = 1.0
CLIP_DENOM_EPS = 1e-6
BASE_LR = 5e-4
TOTAL_MICRO = 256
H_READOUTS = (1, 2, 4, 8, 16, 32)
H_MAX = 32
DELTA_GRID = (0.02, 0.05, 0.1, 0.25, 0.5)
DELTA_PRIMARY = 0.05
CELL_STREAM = {(2042, 26): 1, (2042, 64): 2, (2043, 26): 3}
STOP_TANGENT_NORM = 1e30


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class Rig:
    """Shared model/ds/batches/eval plumbing for one cell."""

    def __init__(self, device: str):
        from model.model_minimind import MiniMindConfig
        from trainer.trainer_utils import get_lr, init_model
        from dataset.lm_dataset import PretrainDataset

        mmcfg = MiniMindConfig(hidden_size=768, num_hidden_layers=13, max_seq_len=MAX_SEQ,
                               num_attention_heads=12, num_key_value_heads=6, use_moe=False)
        model, tokenizer = init_model(mmcfg, from_weight="none",
                                      tokenizer_path=str(MM / "model"), device="cpu")
        self.template = model
        self.tokenizer = tokenizer
        self.get_lr = get_lr
        self.ds = PretrainDataset(str(D0), tokenizer, max_length=MAX_SEQ)
        self.device = device
        self.model_gpu = copy.deepcopy(model).to(device)
        part = json.loads(PARTITION.read_text())
        self.train_idx = part["train_indices"]
        self.eval_idx = part["eval_indices"]
        self.eval_batches = [self.eval_idx[i:i + 16] for i in range(0, len(self.eval_idx), 16)]

    def batches_for(self, stream_id: int):
        stream_seed = derive_stream_seed(20260926, stream_id)
        perm = make_permutation(len(self.train_idx), stream_seed)
        return [[self.train_idx[i] for i in perm[k:k + BATCH_SIZE]]
                for k in range(0, len(perm), BATCH_SIZE)], stream_seed

    def restore(self, state):
        model_cpu = copy.deepcopy(self.template)
        model_cpu.load_state_dict({k: v.detach().clone() for k, v in state["model"].items()})
        named = list(model_cpu.named_parameters())
        g = state["optimizer"]["param_groups"][0]
        opt = torch.optim.AdamW([p for _, p in named], lr=float(g["lr"]),
                                betas=tuple(float(b) for b in g["betas"]),
                                eps=float(g["eps"]), weight_decay=float(g["weight_decay"]))
        for (name, p), (ename, idx, slot) in zip(named, f0.named_optimizer_entries(state)):
            assert ename == name, f"bank order mismatch {ename} vs {name}"
            opt.state[p] = {"step": slot["step"].clone(),
                            "exp_avg": slot["exp_avg"].clone(),
                            "exp_avg_sq": slot["exp_avg_sq"].clone()}
        return named, opt, model_cpu

    def sync_gpu(self, model_cpu, train=True):
        sd = model_cpu.state_dict()
        with torch.no_grad():
            for n, p in self.model_gpu.named_parameters():
                p.data = sd[n].to(self.device)
        self.model_gpu.train(train)

    def compute_grads(self, model_cpu, batches, h0):
        self.sync_gpu(model_cpu, train=True)
        self.model_gpu.zero_grad()
        loss_total = 0.0
        for mb in range(ACCUM):
            idx = batches[h0 * ACCUM + mb]
            x, y = stack_batch(self.ds, idx, self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model_gpu(x).logits
                loss = torch.nn.functional.cross_entropy(
                    out.view(-1, out.size(-1)).float(), y.view(-1))
            (loss / ACCUM).backward()
            loss_total += float(loss.detach().float().cpu()) / ACCUM
        grads = {n: p.grad.detach().to("cpu", torch.float32).clone()
                 for n, p in self.model_gpu.named_parameters()}
        self.model_gpu.zero_grad()
        return grads, loss_total

    def eval_ce(self, model_cpu, autocast=False):
        self.sync_gpu(model_cpu, train=False)
        with torch.no_grad():
            total, count = 0.0, 0
            for idx in self.eval_batches:
                x, y = stack_batch(self.ds, idx, self.device)
                if autocast:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = self.model_gpu(x).logits
                    logits = out.float()
                else:
                    logits = self.model_gpu(x).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
                total += float(loss.detach().cpu())
                count += y.numel()
        return total / count

    def val_grad(self, model_cpu):
        self.sync_gpu(model_cpu, train=True)
        for p in self.model_gpu.parameters():
            p.grad = None
        nb = len(self.eval_batches)
        for idx in self.eval_batches:
            x, y = stack_batch(self.ds, idx, self.device)
            logits = self.model_gpu(x).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")
            (loss / nb).backward()
        g = {n: p.grad.detach().to("cpu", torch.float32).clone()
             for n, p in self.model_gpu.named_parameters()}
        self.model_gpu.zero_grad()
        return g

    def hvp(self, model_cpu, batches, h0, dtheta, fp32=True):
        self.sync_gpu(model_cpu, train=True)
        params = dict(self.model_gpu.named_parameters())
        dth_gpu = {n: v.to(self.device, torch.float32) for n, v in dtheta.items()}
        acc = None
        # math-SDPA required for double backward (legacy registered HVP semantics)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for mb in range(ACCUM):
                idx = batches[h0 * ACCUM + mb]
                x, y = stack_batch(self.ds, idx, self.device)
                self.model_gpu.zero_grad()
                if fp32:
                    logits = self.model_gpu(x).logits
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model_gpu(x).logits
                    logits = logits.float()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))
                g = torch.autograd.grad(loss, list(params.values()), create_graph=True)
                dot = None
                for (n, _), gi in zip(params.items(), g):
                    term = (gi.float() * dth_gpu[n]).sum()
                    dot = term if dot is None else dot + term
                hv = torch.autograd.grad(dot, list(params.values()), retain_graph=False)
                part = {n: hi.detach().float().cpu() for n, hi in zip(params.keys(), hv)}
                if acc is None:
                    acc = part
                else:
                    for n in acc:
                        acc[n] += part[n]
                self.model_gpu.zero_grad()
                del g, dot, hv, part
        for n in acc:
            acc[n] /= ACCUM
        return acc


def fd_rollout(rig: Rig, state, batches, delta, micro0, get_lr):
    named, opt, model_cpu = rig.restore(state)
    losses = {}
    trace = []
    for h in range(1, H_MAX + 1):
        lr = get_lr(micro0 + (h - 1) * ACCUM, TOTAL_MICRO, BASE_LR) * math.exp(delta)
        opt.param_groups[0]["lr"] = lr
        grads, loss = rig.compute_grads(model_cpu, batches, h - 1)
        for n, p in named:
            p.grad = grads[n].clone()
        total = torch.nn.utils.clip_grad_norm_([p for _, p in named], MAX_NORM)
        coef = min(1.0, MAX_NORM / (float(total) + CLIP_DENOM_EPS))
        opt.step()
        opt.zero_grad(set_to_none=True)
        trace.append({"h": h, "loss": loss, "preclip": float(total), "coef": coef, "lr": lr})
        if h in H_READOUTS:
            losses[h] = rig.eval_ce(model_cpu)
    return losses, trace


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
    log(f"state restored micro0={micro0} lr0={rig.get_lr(micro0, TOTAL_MICRO, BASE_LR):.6g}")

    # ---------------- FD arms ----------------
    fd_rows = []
    hfd = {}
    for delta in DELTA_GRID:
        t0 = time.time()
        lp, _ = fd_rollout(rig, state, batches, +delta, micro0, rig.get_lr)
        lm, _ = fd_rollout(rig, state, batches, -delta, micro0, rig.get_lr)
        for h in H_READOUTS:
            h_fd = (lp[h] - lm[h]) / (2.0 * delta)
            hfd[(delta, h)] = h_fd
            fd_rows.append({"cell": f"{seed}:{step}", "delta": delta, "H": h,
                            "L_plus": lp[h], "L_minus": lm[h], "h_FD": h_fd})
        log(f"delta={delta}: h_FD(1)={hfd[(delta, 1)]:.6g} h_FD(16)={hfd[(delta, 16)]:.6g} "
            f"h_FD(32)={hfd[(delta, 32)]:.6g} ({time.time()-t0:.0f}s)")

    # ---------------- baseline + tangent arms ----------------
    named, opt, model_cpu = rig.restore(state)
    arms = {}
    for an, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        for br in ("B", "C"):
            arms[f"{an}/{br}"] = {
                "dtype": dt, "stopped": False, "reason": "",
                "dth": {n: torch.zeros_like(p, dtype=dt) for n, p in named},
                "dm": {n: torch.zeros_like(p, dtype=dt) for n, p in named},
                "dv": {n: torch.zeros_like(p, dtype=dt) for n, p in named}}
    arms["frozen/C"] = {
        "dtype": torch.float32, "stopped": False, "reason": "",
        "dth": {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named},
        "dm": {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named},
        "dv": {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named}}

    tan_rows = []
    vg_cache = {}
    for h in range(1, H_MAX + 1):
        t0 = time.time()
        theta_prev = {n: p.detach().clone() for n, p in named}
        lr = rig.get_lr(micro0 + (h - 1) * ACCUM, TOTAL_MICRO, BASE_LR)
        opt.param_groups[0]["lr"] = lr
        grads, loss = rig.compute_grads(model_cpu, batches, h - 1)
        for n, p in named:
            p.grad = grads[n].clone()
        total = torch.nn.utils.clip_grad_norm_([p for _, p in named], MAX_NORM)
        coef = min(1.0, MAX_NORM / (float(total) + CLIP_DENOM_EPS))
        clipped = {n: p.grad.detach().clone() for n, p in named}
        # p.grad stays set so opt.step() below advances the primal

        # HVP on PRE-update dtheta at PRE-step theta (C branches, h>1)
        hvp_fp32 = hvp_bf16 = None
        if h > 1:
            if not (arms["fp64/C"]["stopped"] and arms["fp32/C"]["stopped"]):
                hvp_fp32 = rig.hvp(model_cpu, batches, h - 1,
                                   {n: arms["fp64/C"]["dth"][n].to(torch.float32)
                                    for n, _ in named}, fp32=True)
            if not arms["frozen/C"]["stopped"]:
                hvp_bf16 = rig.hvp(model_cpu, batches, h - 1,
                                   {n: arms["frozen/C"]["dth"][n].to(torch.float32)
                                    for n, _ in named}, fp32=False)

        # per-arm dc (tangent of the clip map; zero for B and for h==1)
        def dc_for(key):
            if h == 1 or key.endswith("/B"):
                return {n: torch.zeros_like(p) for n, p in named}
            src = hvp_bf16 if key.startswith("frozen") else hvp_fp32
            if src is None:
                return None
            clip = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=src,
                            max_norm=MAX_NORM)
            return clip["clip_jvp"]

        # vendored tangent JVP with PRE-step optimizer state, then primal step
        jvp_out = {}
        for key, a in arms.items():
            if a["stopped"]:
                continue
            dc = dc_for(key)
            if dc is None:
                a["stopped"], a["reason"] = True, "HVP_SOURCE_STOPPED"
                continue
            try:
                out = adamw_tangent_step(
                    named_parameters=named, optimizer=opt,
                    clipped_gradient_by_name=clipped,
                    tangent_clipped_gradient_by_name=dc,
                    tangent_theta_by_name={n: a["dth"][n].to(a["dtype"]) for n, _ in named},
                    tangent_m_by_name={n: a["dm"][n].to(a["dtype"]) for n, _ in named},
                    tangent_v_by_name={n: a["dv"][n].to(a["dtype"]) for n, _ in named},
                )
                jvp_out[key] = out
            except ValueError as exc:
                a["stopped"], a["reason"] = True, f"GUARD: {exc}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        theta_next = {n: p.detach().clone() for n, p in named}
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}

        for key, a in arms.items():
            if key not in jvp_out:
                continue
            dt = a["dtype"]
            new_th = {n: jvp_out[key]["tangent_theta"][n] + inh[n].to(dt) for n, _ in named}
            a["dth"] = new_th
            a["dm"] = jvp_out[key]["tangent_m"]
            a["dv"] = jvp_out[key]["tangent_v"]
            fin = all(bool(torch.isfinite(t).all().item()) for t in new_th.values())
            nrm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in new_th.values()))
            row = {"cell": f"{seed}:{step}", "arm": key, "h": h, "finite": fin,
                   "dtheta_u_norm": nrm, "stopped": False, "reason": ""}
            if not fin or nrm > STOP_TANGENT_NORM:
                a["stopped"], a["reason"] = True, "NONFINITE_OR_OVERFLOW"
                row["stopped"] = True
                row["reason"] = a["reason"]
            if h in H_READOUTS and not a["stopped"]:
                if h not in vg_cache:
                    vg_cache[h] = rig.val_grad(model_cpu)
                vg = vg_cache[h]
                h_tan = sum(float((vg[n].double() * a["dth"][n].double()).sum())
                            for n, _ in named)
                mods = fm.modulewise_shares(named, a["dth"])
                row.update({"h_tangent": h_tan, "h_FD_primary": hfd[(DELTA_PRIMARY, h)],
                            "top_module": max(mods, key=mods.get)})
            tan_rows.append(row)
        dth64 = math.sqrt(sum(float((v.double() ** 2).sum())
                              for v in arms["fp64/C"]["dth"].values()))
        log(f"h={h}/{H_MAX} loss={loss:.5f} coef={coef:.4f} |dth/du|fp64/C={dth64:.4g} "
            f"({time.time()-t0:.0f}s)")

    with (OUT / f"raw_{seed}_{step}.json").open("w") as fh:
        json.dump({"fd": fd_rows, "tangent": tan_rows, "manifest_sha256": msha,
                   "stream_seed": stream_seed, "micro0": micro0,
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)
    log(f"cell {seed}:{step} complete ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
