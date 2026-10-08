#!/usr/bin/env python3
"""F1 Stage 6: multi-step tangent growth on the real 1B primal trajectory
(prereg section 7).

Branches/arms (frozen):
  B (optimizer-only tangent): fp64 / fp32 / bf16 accumulation, H<=32,
    checkpoints 380000/950000/1530000, stream1, impulse dc=q_stream1 at h=1.
  C (HVP feedback): 'registered' (bf16 model + fp16 dtheta HVP, fp32 tangent)
    and 'fp64' (fp32 model + fp32 dtheta HVP, fp64 tangent), H<=16, step380000.

Primal is IDENTICAL across arms and runs: bf16-autocast GPU grads on the
selected device, fp32 CPU native AdamW, global clip 1.0 (torch semantics),
frozen stream1 tokens (cyclic extension beyond H=16, prereg section 7).

Per-step order (frozen, mirrors legacy tangent_pass): grads -> primal clip
(without stepping) -> clip_jvp per arm -> vendored per-tensor tangent step
reading PRE-step optimizer state -> primal optimizer.step().

Arm dtype handling (frozen): fp64 arm upcasts grads to fp64 before clip_jvp
(clip arithmetic fp64); fp32/bf16 arms use fp32 grads (clip arithmetic fp32,
vendored clip_jvp accumulates norms in fp64 internally either way); tangent
state always kept in the arm dtype.

Stop rule (frozen): non-finite tangent value or ||dtheta||>1e30 stops the arm;
non-finite primal aborts the run. Partial rows are always flushed.

Run with CUDA_VISIBLE_DEVICES=1 (GPU0 is externally occupied).
Writes outputs/f1_1b_replication/ms_{step}_{branch}_{arm}.csv.

Usage:
  python scripts/f1_multistep.py --step 380000 --branch B --arm fp64 --H 32
  python scripts/f1_multistep.py --step 380000 --branch C --arm registered --H 16
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import opsens.f1_1b_replication_v01 as f1
from opsens.g1c_olmo2_architecture import build_from_config, chunked_mean_ce
from opsens.g1c_olmo2_checkpoint import attach_model_state, build_optimizer, load_model_state
from opsens.a0c_adamw_response_propagation_v01 import clip_jvp
from opsens.g1c_olmo2_tangent import g1c_adamw_tangent_step

OUT = f1.OUT
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "multistep_run.log"

SEQ_LEN = 512
SEQS_PER_MB = 8
MB_PER_STEP = 4
MAX_NORM = 1.0
CLIP_DENOM_EPS = 1e-6
STOP_TANGENT_NORM = 1e30

ARM_DTYPES = {"fp64": torch.float64, "fp32": torch.float32, "bf16": torch.bfloat16}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_tokens(H: int) -> np.ndarray:
    raw = (f1.G1C_ROOT / "streams/stream1_tokens.bin").read_bytes()
    arr = np.frombuffer(raw, dtype="<i4")
    need = H * MB_PER_STEP * SEQ_LEN * SEQS_PER_MB
    if need <= arr.size:
        return arr[:need].copy()
    reps = int(math.ceil(need / arr.size))
    tiled = np.tile(arr, reps)[:need]
    log(f"stream cyclic extension: {arr.size} -> {need} tokens (x{reps})")
    return tiled.copy()


def microbatch_tokens(tokens: np.ndarray, h0: int, mb: int):
    offset = (h0 * MB_PER_STEP + mb) * SEQ_LEN * SEQS_PER_MB
    windows = tokens[offset:offset + SEQ_LEN * SEQS_PER_MB].reshape(SEQS_PER_MB, SEQ_LEN)
    ids = torch.from_numpy(windows.astype(np.int64))
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def nullctx():
    return _NullCtx()


class GpuSession:
    def __init__(self, model_cfg: dict, device: str):
        self.device = device
        self.model = build_from_config(model_cfg).to(self.device)
        self.model.eval()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    @property
    def named_params(self):
        return list(self.model.named_parameters())

    def upload(self, state, dtype):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data = state[name].to(device=self.device, dtype=dtype).contiguous()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    def forward_loss(self, ids, targets, create_graph=False, autocast=True):
        ids = ids.to(self.device)
        targets = targets.to(self.device)
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        ctx = (torch.autocast(dev, dtype=torch.bfloat16) if autocast else nullctx())
        with ctx:
            hidden = self.model.forward_hidden(ids, checkpoint_blocks=not create_graph)
            loss = chunked_mean_ce(hidden, self.model.transformer.ff_out.weight,
                                    targets, chunk=1024,
                                    recompute_logits=not create_graph)
        return loss

    def grads_cpu(self):
        out = {n: p.grad.detach().to("cpu", torch.float32).contiguous()
               for n, p in self.model.named_parameters()}
        self.zero_grad()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        return out


def compute_grads(gpu: GpuSession, tokens, h0: int):
    gpu.zero_grad()
    loss_total = 0.0
    for mb in range(MB_PER_STEP):
        ids, targets = microbatch_tokens(tokens, h0, mb)
        loss = gpu.forward_loss(ids, targets, autocast=True)
        (loss / MB_PER_STEP).backward()
        loss_total += float(loss.detach().float().cpu()) / MB_PER_STEP
    return gpu.grads_cpu(), loss_total


def hvp_pass(gpu: GpuSession, tokens, h0: int, dtheta, dtype, chunk_seqs: int = 1):
    """Legacy-registered batch-chunked double backward (frozen semantics)."""
    dtheta_gpu = {n: v.to(gpu.device, dtype) for n, v in dtheta.items()}
    params = dict(gpu.named_params)
    hvp_cpu = None
    n_chunks = 0
    for mb in range(MB_PER_STEP):
        ids, targets = microbatch_tokens(tokens, h0, mb)
        for start in range(0, ids.shape[0], chunk_seqs):
            if str(gpu.device).startswith("cuda"):
                free_b, _ = torch.cuda.mem_get_info(gpu.device)
                if free_b < 3 * 2**30:
                    torch.cuda.empty_cache()
            ids_c = ids[start:start + chunk_seqs].contiguous()
            tg_c = targets[start:start + chunk_seqs].contiguous()
            gpu.zero_grad()
            loss = gpu.forward_loss(ids_c, tg_c, create_graph=True, autocast=False)
            g = torch.autograd.grad(loss, list(params.values()), create_graph=True)
            dot = None
            for (n, _), gi in zip(params.items(), g):
                term = (gi.float() * dtheta_gpu[n].float()).sum()
                dot = term if dot is None else dot + term
            hvp = torch.autograd.grad(dot, list(params.values()), retain_graph=False)
            part = {n: hi.detach().float().cpu() for n, hi in zip(params.keys(), hvp)}
            if hvp_cpu is None:
                hvp_cpu = part
            else:
                for n in hvp_cpu:
                    hvp_cpu[n] += part[n]
            n_chunks += 1
            gpu.zero_grad()
            del g, dot, hvp, part, loss, ids_c, tg_c
            if str(gpu.device).startswith("cuda"):
                torch.cuda.empty_cache()
    for n in hvp_cpu:
        hvp_cpu[n] /= float(n_chunks)
    del dtheta_gpu
    if str(gpu.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return hvp_cpu


class CpuState:
    def __init__(self, step: int):
        t0 = time.time()
        self.step = step
        self.model_cfg = f1.load_model_cfg(step)
        self.model = build_from_config(self.model_cfg)
        state = load_model_state(str(f1.ckpt_path(step, "model.pt")))
        self.attach_report = attach_model_state(self.model, state)
        del state
        self.optimizer, self.optim_report = build_optimizer(
            str(f1.ckpt_path(step, "optim.pt")), self.model)
        self.named = list(self.model.named_parameters())
        self.group_of = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.group_of[p] = group
        log(f"CpuState step={step} restored ({time.time()-t0:.0f}s) "
            f"elements={self.attach_report['n_elements']}")

    def state_dict_fp32(self):
        return {n: t for n, t in self.model.state_dict().items()}

    def clip_no_step(self, raw_grads):
        """Set .grad, apply torch global clip, return (clipped, norm, coef);
        optimizer.step() is NOT executed here (frozen per-step order)."""
        for n, p in self.named:
            p.grad = raw_grads[n].clone()
        total = torch.nn.utils.clip_grad_norm_([p for _, p in self.named], MAX_NORM)
        norm = float(total)
        coef = min(1.0, MAX_NORM / (norm + CLIP_DENOM_EPS))
        clipped = {n: p.grad.detach().clone() for n, p in self.named}
        return clipped, norm, coef

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


def tangent_step_arm(cpu: CpuState, c_by_name, dc_by_name, dtheta, dm, dv, dtype):
    """Per-tensor vendored tangent step reading the PRE-step optimizer state,
    upcast per tensor into the arm dtype. c = clipped real gradient,
    dc = clip JVP of the tangent gradient."""
    opt_state = cpu.optimizer.state
    out_th, out_m, out_v = {}, {}, {}
    for name, p in cpu.named:
        st = opt_state[p]
        g = cpu.group_of[p]

        class _O:
            pass

        fake = _O()
        fake_param = torch.nn.Parameter(torch.zeros(p.shape, dtype=dtype))
        fake.param_groups = [{"params": [fake_param], "lr": float(g["lr"]),
                              "betas": (float(g["betas"][0]), float(g["betas"][1])),
                              "eps": float(g["eps"]),
                              "weight_decay": float(g["weight_decay"]),
                              "amsgrad": False, "maximize": False, "differentiable": False}]
        fake.state = {fake_param: {"step": torch.tensor(float(st["step"])),
                                   "exp_avg": st["exp_avg"].to(dtype),
                                   "exp_avg_sq": st["exp_avg_sq"].to(dtype)}}
        res = g1c_adamw_tangent_step(
            named_parameters=[(name, fake_param)], optimizer=fake,
            clipped_gradient_by_name={name: c_by_name[name].to(dtype)},
            tangent_clipped_gradient_by_name={name: dc_by_name[name].to(dtype)},
            tangent_theta_by_name={name: dtheta[name]},
            tangent_m_by_name={name: dm[name]},
            tangent_v_by_name={name: dv[name]},
        )
        out_th[name] = res["tangent_theta"][name]
        out_m[name] = res["tangent_m"][name]
        out_v[name] = res["tangent_v"][name]
        del fake, fake_param, res
    return out_th, out_m, out_v


def norm_stats(tensor_by_name):
    total = 0.0
    for t in tensor_by_name.values():
        x = t.detach().to(torch.float64)
        total += float(torch.sum(x * x).item())
    return math.sqrt(total)


def all_finite(tensors) -> bool:
    for t in tensors.values():
        if not bool(torch.isfinite(t).all().item()):
            return False
    return True


def build_row(cpu, step, arm, branch, h, loss, grad_norm, coef, active,
              dtheta, dm, dv, hvp_norm, finite, stopped_reason=""):
    th_norm = norm_stats(dtheta)
    dm_norm = norm_stats(dm)
    dv_norm = norm_stats(dv)
    mods = f1.modulewise_norms(cpu.named, dtheta)
    best = None
    opt_state = cpu.optimizer.state
    for name, p in cpu.named:
        t = dtheta[name].detach().to(torch.float64)
        flat = int(t.abs().argmax().item())
        val = float(t.flatten()[flat].item())
        if best is None or val > best["max_abs"]:
            st = opt_state[p]
            best = {"max_abs": val, "param": name, "row": int(
                flat // t.shape[1] if t.dim() == 2 else flat),
                "local_v": float(st["exp_avg_sq"].flatten()[flat].item()),
                "local_m_abs": abs(float(st["exp_avg"].flatten()[flat].item()))}
    row = {
        "step": step, "arm": arm, "branch": branch, "h": h,
        "loss": loss, "preclip_grad_norm": grad_norm,
        "clip_coef": coef, "clip_active": bool(active),
        "dtheta_norm": th_norm, "dm_norm": dm_norm, "dv_norm": dv_norm,
        "total_tangent_norm": math.sqrt(th_norm ** 2 + dm_norm ** 2 + dv_norm ** 2),
        "hvp_norm": hvp_norm, "jvp_norm": th_norm,
        "max_abs_dtheta": best["max_abs"], "max_param": best["param"],
        "max_row": best["row"], "local_v": best["local_v"],
        "local_m_abs": best["local_m_abs"],
        "finite": finite, "stopped_reason": stopped_reason,
    }
    row.update({f"mod_{k}": v for k, v in mods.items()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=list(f1.STEPS))
    ap.add_argument("--branch", choices=["B", "C"], required=True)
    ap.add_argument("--arm", required=True,
                    help="B: comma-separated subset of fp64,fp32,bf16; C: registered|fp64")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        log("REFUSING to run without CUDA_VISIBLE_DEVICES (set =1; GPU0 externally occupied)")
        return 1
    torch.set_num_threads(min(32, torch.get_num_threads()))
    torch.manual_seed(0)

    arm_names = [a.strip() for a in args.arm.split(",")]
    arms = {}
    for a in arm_names:
        if args.branch == "B":
            assert a in ARM_DTYPES, "B arms: fp64|fp32|bf16"
            arms[a] = {"dtype": ARM_DTYPES[a], "hvp_dtype": None,
                       "hvp_model_dtype": None}
        else:
            assert len(arm_names) == 1 and a in ("registered", "fp64"), \
                "C arms run one per process: registered|fp64"
            arms[a] = {"dtype": torch.float32 if a == "registered" else torch.float64,
                       "hvp_dtype": torch.float16 if a == "registered" else torch.float32,
                       "hvp_model_dtype": torch.bfloat16 if a == "registered" else torch.float32}

    tag = f"{args.step}_{args.branch}_{args.arm.replace(',', '+')}"
    log(f"=== run {tag}: H={args.H} arms={list(arms)} device={args.device} ===")

    ok_o = f1.verify_checkpoint(args.step, "optim.pt")
    ok_m = f1.verify_checkpoint(args.step, "model.pt")
    if not (ok_o["sha_ok"] and ok_m["sha_ok"]):
        log(f"checkpoints not verified: optim={ok_o['sha_ok']} model={ok_m['sha_ok']}; abort")
        return 1

    tokens = load_tokens(args.H)
    cpu = CpuState(args.step)
    gpu = GpuSession(cpu.model_cfg, args.device)
    q = f1.regenerate_q_stream1(cpu.model)
    q_sha = f1.direction_tensor_sha256(q)
    if q_sha != f1.Q_STREAM1_TENSOR_SHA256:
        log(f"q_stream1 SHA MISMATCH: {q_sha} != {f1.Q_STREAM1_TENSOR_SHA256}; abort")
        return 1
    log(f"q_stream1 regenerated + SHA verified ({q_sha[:16]}...)")

    for a, cfg in arms.items():
        dt = cfg["dtype"]
        cfg["dtheta"] = {n: torch.zeros_like(p, dtype=dt) for n, p in cpu.named}
        cfg["dm"] = {n: torch.zeros_like(p, dtype=dt) for n, p in cpu.named}
        cfg["dv"] = {n: torch.zeros_like(p, dtype=dt) for n, p in cpu.named}
        cfg["stopped"] = False
        cfg["reason"] = ""
        cfg["csv"] = OUT / f"ms_{args.step}_{args.branch}_{a}.csv"
        cfg["fieldnames"] = None
        cfg["pending"] = []

    def flush_arm(a):
        cfg = arms[a]
        if not cfg["pending"]:
            return
        exists = cfg["csv"].exists() and cfg["csv"].stat().st_size > 0
        with cfg["csv"].open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cfg["fieldnames"], extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerows(cfg["pending"])
        cfg["pending"].clear()

    primal_ok = True
    for h in range(1, args.H + 1):
        t0 = time.time()
        gpu.upload(cpu.state_dict_fp32(), torch.float32)
        grads, loss = compute_grads(gpu, tokens, h - 1)
        if not math.isfinite(loss) or not all_finite(grads):
            log(f"h={h}: PRIMAL NON-FINITE (loss={loss}); aborting")
            for a, cfg in arms.items():
                row = {"step": args.step, "arm": a, "branch": args.branch, "h": h,
                       "loss": loss, "finite": False, "stopped_reason": "PRIMAL_NONFINITE"}
                if cfg["fieldnames"] is None:
                    cfg["fieldnames"] = list(row.keys())
                cfg["pending"].append(row)
                flush_arm(a)
            return 2

        # primal clip (no step yet: tangent must read pre-step state)
        clipped, norm, coef = cpu.clip_no_step(grads)
        hvp_norms = {}

        for a, cfg in arms.items():
            if cfg["stopped"]:
                continue
            dtype = cfg["dtype"]
            hvp_norm = 0.0
            try:
                if h == 1:
                    tg = {n: q[n].to(dtype) for n, _ in cpu.named}
                elif args.branch == "B":
                    tg = {n: torch.zeros_like(p, dtype=dtype) for n, p in cpu.named}
                else:
                    gpu.upload(cpu.state_dict_fp32(), cfg["hvp_model_dtype"])
                    t_hvp = time.time()
                    tg = hvp_pass(gpu, tokens, h - 1, cfg["dtheta"], cfg["hvp_dtype"])
                    hvp_norm = norm_stats(tg)
                    gpu.upload(cpu.state_dict_fp32(), torch.float32)
                    log(f"  h={h} [{a}] HVP {time.time()-t_hvp:.0f}s norm={hvp_norm:.4g}")
                clip_grads = grads if dtype != torch.float64 else \
                    {n: g.to(torch.float64) for n, g in grads.items()}
                clip = clip_jvp(gradient_by_name=clip_grads,
                                tangent_gradient_by_name=tg, max_norm=MAX_NORM)
                out_th, out_m, out_v = tangent_step_arm(
                    cpu, clipped, clip["clip_jvp"], cfg["dtheta"], cfg["dm"], cfg["dv"], dtype)
                cfg["dtheta"], cfg["dm"], cfg["dv"] = out_th, out_m, out_v
                fin = all_finite(out_th) and all_finite(out_m) and all_finite(out_v)
                th_n = norm_stats(out_th)
                if not fin:
                    cfg["stopped"], cfg["reason"] = True, "NONFINITE_TANGENT"
                elif th_n > STOP_TANGENT_NORM:
                    cfg["stopped"], cfg["reason"] = True, "TANGENT_NORM_OVERFLOW"
                row = build_row(cpu, args.step, a, args.branch, h, loss, norm, coef,
                                coef < 1.0, out_th, out_m, out_v, hvp_norm, fin, cfg["reason"])
                hvp_norms[a] = hvp_norm
            except ValueError as exc:
                cfg["stopped"], cfg["reason"] = True, f"GUARD: {exc}"
                row = {"step": args.step, "arm": a, "branch": args.branch, "h": h,
                       "loss": loss, "preclip_grad_norm": norm, "clip_coef": coef,
                       "finite": False, "stopped_reason": cfg["reason"],
                       "hvp_norm": hvp_norm}
                log(f"h={h}: arm {a} stopped ({cfg['reason']})")
            if cfg["fieldnames"] is None:
                cfg["fieldnames"] = list(row.keys())
            cfg["pending"].append(row)
            flush_arm(a)

        cpu.step()
        line = (f"h={h}/{args.H} loss={loss:.5f} |g|={norm:.3f} coef={coef:.5f} "
                + " ".join(f"|{a}|={norm_stats(arms[a]['dtheta']) if not arms[a]['stopped'] else float('nan'):.4g}"
                           for a in arms) + f" ({time.time()-t0:.0f}s)")
        log(line)
        del grads, clipped
        if all(cfg["stopped"] for cfg in arms.values()):
            break

    for a, cfg in arms.items():
        log(f"=== arm {a} finished (stopped={cfg['stopped']} reason={cfg['reason']}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
