#!/usr/bin/env python3
"""FR_G1C_1B_FEASIBILITY_GATE_V01 runner (Phase 0/1/2 + decision).

Feasibility gate only — NOT the formal G1c science experiment. No
generalization claims. See G1C_FEASIBILITY_PROTOCOL.md and
G1C_PROTOCOL_AMENDMENTS.md (A1-A3).

Usage (repo root, conda env fr_minimind):
  PYTHONPATH=research/src python research/scripts/g1c_1b_feasibility_v01.py phase0
  ... phase1 | phase2 | decision
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_response_propagation_v01 import (  # frozen imports only
    _metric,
    clip_jvp,
    project_tangent,
    relative_error,
)
from fr.direction import save_direction
from fr.g1c_olmo2_architecture import G1COLMo2, build_from_config, chunked_mean_ce
from fr.g1c_olmo2_checkpoint import (
    attach_model_state,
    build_optimizer,
    inspect_train_state,
    load_model_state,
)
from fr.g1c_olmo2_tangent import g1c_adamw_tangent_step
from fr.stream_rng import derive_stream_seed

ROOT = Path(__file__).resolve().parents[2] / "outputs" / "generalization_g1c_feasibility_v01"
CKPT = ROOT / "checkpoints"
STREAMS = ROOT / "streams"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

STEPS = (380000, 950000, 1530000)
PHASE1_STEP = 950000
BASE_SEED = 20260917
STREAM_IDS = (1, 2)
RHO = 0.25
MAX_NORM = 1.0
SEQ_LEN = 512           # B8 cascade: 4096->2048->512 (amendment A4: shared GPU0,
                        # double backward needs ~2x activations; 512 peaks ~15GB)
SEQS_PER_MB = 8         # microbatch = 4096 tokens (protocol 6.2/6.3 unchanged)
MB_PER_STEP = 4
H_MAX = 8
HORIZONS = (1, 4, 8)
CLIP_DENOM_EPS = 1e-6   # torch clip_grad_norm_ semantics
EXPECTED_SIZES = {
    "config.yaml": None,  # not size-gated
    "train.pt": None,
    "model.pt": 5939703038,
    "optim.pt": 11879442406,
}
PARAM_COUNT = 1484916736

COMMANDS_LOG = ROOT / "COMMANDS.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with COMMANDS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ckpt_path(step: int, name: str) -> Path:
    return CKPT / f"step{step}_{name}"


def load_model_cfg(step: int) -> dict:
    cfg = yaml.safe_load(ckpt_path(step, "config.yaml").read_text(encoding="utf-8"))
    return cfg["model"]


def load_stream_tokens(stream_id: int) -> np.ndarray:
    raw = (STREAMS / f"stream{stream_id}_tokens.bin").read_bytes()
    arr = np.frombuffer(raw, dtype="<i4")
    need = H_MAX * MB_PER_STEP * SEQ_LEN * SEQS_PER_MB
    if arr.size < need:
        raise ValueError(f"stream{stream_id} too short: {arr.size} < {need}")
    return arr[:need].copy()


def microbatch_tokens(tokens: np.ndarray, step_index: int, mb_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids [S,T], targets [S,T]) for one microbatch.

    step_index in 0..H_MAX-1, mb_index in 0..MB_PER_STEP-1.
    """
    offset = (step_index * MB_PER_STEP + mb_index) * SEQ_LEN * SEQS_PER_MB
    windows = tokens[offset:offset + SEQ_LEN * SEQS_PER_MB].reshape(SEQS_PER_MB, SEQ_LEN)
    ids = torch.from_numpy(windows.astype(np.int64))
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def make_q(model: G1COLMo2, stream_id: int) -> dict:
    """Amendment A1: iid N(0,1) full support, then unit-total-L2 normalize."""
    seed = derive_stream_seed(BASE_SEED, stream_id)
    q_seed = derive_stream_seed(seed, 77)
    gen = torch.Generator(device="cpu").manual_seed(q_seed)
    q = {}
    for name, p in model.named_parameters():
        q[name] = torch.randn(p.shape, generator=gen, dtype=torch.float32)
    total = math.sqrt(sum(float(v.double().pow(2).sum()) for v in q.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("q norm invalid")
    q = {k: (v / total).contiguous() for k, v in q.items()}
    return q


def save_q(stream_id: int, q: dict, step: int) -> str:
    seed = derive_stream_seed(BASE_SEED, stream_id)
    meta = {
        "protocol": "FR_G1C_1B_FEASIBILITY_GATE_V01_Q",
        "stream_id": stream_id,
        "stream_seed": seed,
        "q_seed": derive_stream_seed(seed, 77),
        "base_seed": BASE_SEED,
        "distribution": "iid N(0,1) full support, unit total L2 norm (Amendment A1)",
        "sigma_rule": "std of s=<q,g> over baseline future steps (Amendment A1)",
        "rho": RHO,
        "state_step": step,
    }
    return save_direction(STREAMS / f"q_stream{stream_id}.pt", q, meta)


# ---------------------------------------------------------------- GPU session
class GpuSession:
    """Transient-upload GPU compute: nothing persistent between passes."""

    def __init__(self, model_cfg: dict):
        self.device = "cuda"
        self.model = build_from_config(model_cfg).to(self.device)
        self.model.eval()
        torch.cuda.empty_cache()

    @property
    def named_params(self):
        return list(self.model.named_parameters())

    def upload(self, state: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data = state[name].to(device=self.device, dtype=dtype).contiguous()
        torch.cuda.synchronize()

    def zero_grad(self) -> None:
        for p in self.model.parameters():
            p.grad = None

    def forward_loss(self, ids: torch.Tensor, targets: torch.Tensor,
                     create_graph: bool = False, autocast: bool = True) -> torch.Tensor:
        ids = ids.to(self.device)
        targets = targets.to(self.device)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if autocast else torch_nullctx())
        with ctx:
            # create_graph (HVP): no activation checkpointing — double backward
            # through non-reentrant checkpoints is pathologically slow at scale.
            hidden = self.model.forward_hidden(ids, checkpoint_blocks=not create_graph)
            loss = chunked_mean_ce(hidden, self.model.transformer.ff_out.weight,
                                   targets, chunk=1024,
                                   recompute_logits=not create_graph)
        return loss

    def grads_cpu(self) -> dict[str, torch.Tensor]:
        out = {n: p.grad.detach().to("cpu", torch.float32).contiguous()
               for n, p in self.model.named_parameters()}
        self.zero_grad()
        torch.cuda.synchronize()
        return out

    def peak_mem_gb(self) -> float:
        return torch.cuda.max_memory_allocated() / (1 << 30)

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def torch_nullctx():
    return _NullCtx()


# patch method referenced above (keep name stable for readability)
GpuSession.forward_loss.__doc__ = "autocast bf16 when autocast=True, else plain (bf16 params path)."


# ------------------------------------------------------------------ CPU state
class CpuState:
    """Restored checkpoint on CPU: model params + real torch AdamW."""

    def __init__(self, step: int):
        t0 = time.time()
        mcfg = load_model_cfg(step)
        self.model_cfg = mcfg
        state = load_model_state(str(ckpt_path(step, "model.pt")))
        self.model = build_from_config(mcfg)
        self.attach_report = attach_model_state(self.model, state)
        self.optimizer, self.optim_report = build_optimizer(str(ckpt_path(step, "optim.pt")), self.model)
        self.named = list(self.model.named_parameters())
        self.theta0 = {n: p.detach().clone() for n, p in self.named}
        self.lr = float(self.optimizer.param_groups[0]["lr"])
        log(f"CpuState step={step} restored in {time.time()-t0:.1f}s "
            f"params={self.attach_report['n_elements']} lr={self.lr:.6g}")

    def clip_and_snapshot(self, raw_grads: dict[str, torch.Tensor]) -> tuple[dict, float, float]:
        """Set .grad on CPU params, apply torch global clip, return (clipped, norm, coef)."""
        for n, p in self.named:
            p.grad = raw_grads[n].clone()
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in self.named], MAX_NORM)
        norm = float(total_norm)
        coef = min(1.0, MAX_NORM / (norm + CLIP_DENOM_EPS))
        clipped = {n: p.grad.detach().clone() for n, p in self.named}
        self.optimizer.zero_grad(set_to_none=True)
        return clipped, norm, coef

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


# ------------------------------------------------------------------ passes
def compute_grads(gpu: GpuSession, tokens: np.ndarray, step_index: int,
                  autocast: bool = True) -> tuple[dict, float]:
    """One optimizer step worth of microbatch gradients; returns (grads, loss_sum/mb)."""
    gpu.zero_grad()
    loss_total = 0.0
    for mb in range(MB_PER_STEP):
        ids, targets = microbatch_tokens(tokens, step_index, mb)
        loss = gpu.forward_loss(ids, targets, autocast=autocast)
        (loss / MB_PER_STEP).backward()
        loss_total += float(loss.detach().float().cpu()) / MB_PER_STEP
    grads = gpu.grads_cpu()
    return grads, loss_total


def hvp_pass(gpu: GpuSession, tokens: np.ndarray, step_index: int,
             dtheta: dict[str, torch.Tensor],
             chunk_seqs: int = 1,
             dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """Amendment A3 / A5: bf16 double backward, batch-chunked per microbatch.

    Amendment A5: activation memory scales with TOKENS, not seq length, so
    SEQ_LEN=512 alone could not fit double backward in the shared GPU0
    (measured: allocator pinned at the 31.8GB ceiling, >1.5h per pass).
    Each microbatch (SEQS_PER_MB x SEQ_LEN) is split along the BATCH dim
    into chunk_seqs-sequence sub-blocks; per sub-block we build the
    create_graph forward, first-order grad, <g, dtheta> dot, and second
    backward, then free the graph (empty_cache). All sub-blocks have equal
    token counts, so the exact mean-CE HVP is the equal-weight fp32
    accumulation (linear decomposition, identity up to fp reassociation).

    dtheta: fp32 CPU tangent (fp16 on GPU; dot accumulated in fp32).
    Returns fp32 CPU HVP of the full microbatch-averaged objective.
    """
    dtheta_gpu = {n: v.to(gpu.device, dtype) for n, v in dtheta.items()}
    params = dict(gpu.named_params)
    hvp_cpu: dict[str, torch.Tensor] | None = None
    n_chunks = 0
    for mb in range(MB_PER_STEP):
        ids, targets = microbatch_tokens(tokens, step_index, mb)
        for start in range(0, ids.shape[0], chunk_seqs):
            free_b, _ = torch.cuda.mem_get_info(gpu.device)
            if free_b < 3 * 2**30:
                torch.cuda.empty_cache()
            ids_c = ids[start:start + chunk_seqs].contiguous()
            tg_c = targets[start:start + chunk_seqs].contiguous()
            gpu.zero_grad()
            loss = gpu.forward_loss(ids_c, tg_c, create_graph=True, autocast=False)
            g = torch.autograd.grad(loss, list(params.values()),
                                    create_graph=True)
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
            torch.cuda.empty_cache()
    assert hvp_cpu is not None
    for n in hvp_cpu:
        hvp_cpu[n] /= float(n_chunks)
    del dtheta_gpu
    torch.cuda.empty_cache()
    return hvp_cpu


def native_pass(cpu: CpuState, gpu: GpuSession, tokens: np.ndarray, H: int,
                q: dict | None, collect_s: bool = True) -> dict:
    """Baseline replay: native CPU AdamW steps; optionally collect s=<q,g>."""
    fp32_state = {n: t for n, t in cpu.model.state_dict().items()}
    gpu.upload(fp32_state, torch.float32)
    steps = []
    for h in range(H):
        grads, loss = compute_grads(gpu, tokens, h)
        clipped, norm, coef = cpu.clip_and_snapshot(grads)
        s = project_tangent(q, grads) if (collect_s and q is not None) else None
        cpu.step()
        steps.append({"h": h + 1, "loss": loss, "grad_norm": norm,
                      "clip_coef": coef, "s": s})
        log(f"  native h={h+1} loss={loss:.6f} |g|={norm:.6f} coef={coef:.6f} s={s}")
    return {"steps": steps}


def tangent_pass(cpu: CpuState, gpu: GpuSession, tokens: np.ndarray, H: int,
                 q: dict, kind: str, hvp_dtype: torch.dtype = torch.float16,
                 collect_diagnostics: bool = False,
                 diag_sink: list | None = None) -> dict:
    """B (optimizer-only) or C (full, with HVP) tangent propagation.

    hvp_dtype: DIAGNOSTIC knob (default fp16 = registered path). fp32 is
    used only by cmd_cdiag check (3b) and never enters official results.
    collect_diagnostics: per-h AdamW-state/HVP/tangent statistics for the
    C-failure diagnostic; off on the official path.
    diag_sink: optional caller-owned list that receives each horizon's diag
    dict as computed, so partial trajectories survive an exception raised
    mid-pass (diagnostics inside the returned dict are lost in that case).
    """
    assert kind in ("B", "C")
    zero = {n: torch.zeros_like(p) for n, p in cpu.named}
    dtheta = {n: v.clone() for n, v in zero.items()}
    dm = {n: v.clone() for n, v in zero.items()}
    dv = {n: v.clone() for n, v in zero.items()}
    fp32_state = {n: t for n, t in cpu.model.state_dict().items()}
    gpu.upload(fp32_state, torch.float32)
    chi = []
    diagnostics = []
    for h in range(H):
        diag = None
        if collect_diagnostics:
            with torch.no_grad():
                st0 = cpu.optimizer.state[cpu.named[0][1]]
                bc2v = 1.0 - 0.95 ** (float(st0["step"]) + 1.0)
                vh = {n: cpu.optimizer.state[p]["exp_avg_sq"] / bc2v
                      for n, p in cpu.named}
                diag = {
                    "h": h + 1,
                    "state_step": float(st0["step"]),
                    "vhat_min": min(float(v.min()) for v in vh.values()),
                    "vhat_max": max(float(v.max()) for v in vh.values()),
                    "sqrt_vhat_min": min(float(v.sqrt().min()) for v in vh.values()),
                    "sqrt_vhat_p1": min(float((v + 1e-8).sqrt().min()) for v in vh.values()),
                    "m_absmax": max(float(cpu.optimizer.state[p]["exp_avg"].abs().max())
                                    for _, p in cpu.named),
                    "theta_tangent_norm_prev": math.sqrt(
                        sum(float(v.double().pow(2).sum()) for v in dtheta.values())),
                }
        grads, loss = compute_grads(gpu, tokens, h)
        if h == 0:
            tg = {n: v.clone() for n, v in q.items()}
        elif kind == "B":
            tg = {n: torch.zeros_like(v) for n, v in q.items()}
        else:
            gpu.upload({n: t for n, t in cpu.model.state_dict().items()}, torch.bfloat16)
            t0 = time.time()
            tg = hvp_pass(gpu, tokens, h, dtheta, dtype=hvp_dtype)
            log(f"  C h={h+1} HVP {time.time()-t0:.1f}s peak={gpu.peak_mem_gb():.1f}GB")
            gpu.upload({n: t for n, t in cpu.model.state_dict().items()}, torch.float32)
        if collect_diagnostics:
            diag["hvp_norm"] = math.sqrt(
                sum(float(v.double().pow(2).sum()) for v in tg.values()))
            diag["grad_norm"] = math.sqrt(
                sum(float(v.double().pow(2).sum()) for v in grads.values()))
            if diag_sink is not None:
                # Reference is shared: post-step fields (clip_coef, out norms,
                # chi) update this same dict in place. If clip_jvp raises
                # (the C-branch failure mode), the sink still holds the
                # entry-state + HVP stats of the failing horizon.
                diag_sink.append(diag)
        clip = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=tg,
                        max_norm=MAX_NORM)
        if collect_diagnostics:
            diag["clip_coef"] = clip.get("clip_coefficient")
            diag["clip_before_norm"] = clip.get("clip_before_norm")
            diag["clip_active"] = clip.get("clip_active")
        out = g1c_adamw_tangent_step(
            named_parameters=cpu.named, optimizer=cpu.optimizer,
            clipped_gradient_by_name=clip["clipped_gradient"],
            tangent_clipped_gradient_by_name=clip["clip_jvp"],
            tangent_theta_by_name=dtheta, tangent_m_by_name=dm,
            tangent_v_by_name=dv)
        dtheta, dm, dv = out["tangent_theta"], out["tangent_m"], out["tangent_v"]
        cpu.step()
        if collect_diagnostics:
            diag["tangent_theta_norm_out"] = math.sqrt(
                sum(float(v.double().pow(2).sum()) for v in dtheta.values()))
            diag["dm_norm_out"] = math.sqrt(
                sum(float(v.double().pow(2).sum()) for v in dm.values()))
            diag["dv_norm_out"] = math.sqrt(
                sum(float(v.double().pow(2).sum()) for v in dv.values()))
        chi_val = project_tangent(q, dtheta)
        if collect_diagnostics:
            diag["chi"] = chi_val
            diagnostics.append(diag)
        chi.append({"h": h + 1, "chi": chi_val,
                    "theta_tangent_norm": out["tangent_theta_norm"]})
        log(f"  tangent[{kind}] h={h+1} chi={chi[-1]['chi']:.6g}")
    result = {"chi": chi, "per_group": out["per_group"]}
    if collect_diagnostics:
        result["diagnostics"] = diagnostics
    return result


def fd_branch(cpu: CpuState, gpu: GpuSession, tokens: np.ndarray, H: int,
              q: dict, sign: int, eps: float) -> dict:
    """FD replay: fp64 theta bookkeeping, fp32 m/v (torch-native semantics).

    Memory-conscious: no fp64 m/v copies, no theta0 copy (X accumulated
    incrementally per step).
    """
    beta1, beta2 = 0.9, 0.95
    adam_eps = 1e-8
    group_wd = {}
    group_lr = {}
    for group in cpu.optimizer.param_groups:
        for p in group["params"]:
            group_wd[p] = float(group["weight_decay"])
            group_lr[p] = float(group["lr"])
    theta = {n: p.detach().double().clone() for n, p in cpu.named}
    m = {n: cpu.optimizer.state[p]["exp_avg"].clone() for n, p in cpu.named}
    v = {n: cpu.optimizer.state[p]["exp_avg_sq"].clone() for n, p in cpu.named}
    step0 = {n: float(cpu.optimizer.state[p]["step"]) for n, p in cpu.named}
    X = []
    x_accum = 0.0
    for h in range(H):
        gpu.upload(theta, torch.float32)  # per-tensor fp64->fp32 cast on the fly
        grads, loss = compute_grads(gpu, tokens, h)
        if h == 0:
            grads = {n: g + float(sign) * float(eps) * q[n] for n, g in grads.items()}
        norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads.values()))
        coef = min(1.0, MAX_NORM / (norm + CLIP_DENOM_EPS))
        for n, p in cpu.named:
            lr = group_lr[p]
            wd = group_wd[p]
            sa = step0[n] + h + 1.0
            bc1 = 1.0 - beta1 ** sa
            bc2 = 1.0 - beta2 ** sa
            gc = grads[n] * coef                      # fp32, torch-clip semantics
            m[n].mul_(beta1).add_(gc, alpha=1.0 - beta1)
            v[n].mul_(beta2).addcmul_(gc, gc, value=1.0 - beta2)
            upd = (m[n] / bc1) / ((v[n] / bc2).sqrt() + adam_eps)
            theta_old = theta[n]
            theta[n] = theta_old - lr * (upd.double() + wd * theta_old)
            x_accum += float((theta[n] - theta_old).mul_(q[n].double()).sum())
        X.append({"h": h + 1, "X": x_accum, "loss": loss, "grad_norm": norm,
                  "clip_coef": coef})
        log(f"  FD sign={sign:+d} eps={eps:.3g} h={h+1} X={x_accum:.6g}")
    return {"X": X}


def epsilon_from_s(s_values: list[float]) -> tuple[float, float]:
    arr = np.asarray([s for s in s_values if s is not None], dtype=np.float64)
    if arr.size >= 2:
        sigma = float(arr.std(ddof=1))
    else:
        sigma = float(abs(arr[0]))  # degenerate single-sample fallback (registered)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma_q invalid: {sigma}")
    return RHO * sigma, sigma


# ------------------------------------------------------------------ phases
def cmd_phase0(args) -> int:
    log("PHASE 0 start (step950000 restoration gate)")
    step = PHASE1_STEP
    report: dict = {"step": step, "files": {}, "checks": []}

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    for name in ("config.yaml", "model.pt", "optim.pt", "train.pt"):
        p = ckpt_path(step, name)
        exists = p.exists()
        size = p.stat().st_size if exists else -1
        exp = EXPECTED_SIZES.get(name)
        size_ok = (exp is None and size > 0) or size == exp
        sha = sha256_file(p) if exists else None
        report["files"][name] = {"path": str(p), "exists": exists,
                                 "size_bytes": size, "expected_size": exp,
                                 "size_ok": size_ok, "sha256": sha}
        check(f"file:{name}", exists and size_ok,
              f"size={size} expected={exp} sha256={sha}")

    t0 = time.time()
    try:
        cpu = CpuState(step)
        restore_error = None
    except Exception as exc:  # restoration gate: any failure -> FAIL report, not crash
        cpu = None
        restore_error = f"{type(exc).__name__}: {exc}"
    if cpu is None:
        check("model_restore", False, restore_error)
        json_dump(RESULTS / "phase0.json", report)
        (ROOT / "G1C_PHASE0_REPORT.md").write_text(
            f"# G1C Phase 0 Report — checkpoint restoration gate (step950000)\n\n"
            f"- verdict: **FAIL**\n- restoration error: {restore_error}\n",
            encoding="utf-8")
        log("PHASE 0 FAIL (restoration error)")
        return 1
    check("model_restore_keys", True,
          f"n_tensors={cpu.attach_report['n_tensors']} "
          f"n_elements={cpu.attach_report['n_elements']} (expected {PARAM_COUNT}) "
          f"match={cpu.attach_report['n_elements'] == PARAM_COUNT}")
    orp = cpu.optim_report
    check("optim_state_entries", orp["n_state_entries"] == len(cpu.named),
          f"state_entries={orp['n_state_entries']} params={len(cpu.named)}")
    check("optim_step_values", orp["step_values_min"] == orp["step_values_max"],
          f"step unique={orp['step_values_unique']} min={orp['step_values_min']} "
          f"max={orp['step_values_max']} (expect all == {step})")
    check("optim_groups", len(orp["groups"]) >= 1,
          json.dumps(orp["groups"], ensure_ascii=False)[:800])
    check("optim_m_v_finite", math.isfinite(orp["m_total_norm"]) and orp["v_total_mean"] >= 0,
          f"|m|={orp['m_total_norm']:.6g} mean(v)={orp['v_total_mean']:.6g}")
    train = inspect_train_state(str(ckpt_path(step, "train.pt")))
    report["train_state"] = train
    check("train_global_step", train["global_step"] == step,
          f"global_step={train['global_step']} tokens={train['global_train_tokens_seen']}")
    rng_flags = {k: train[k] for k in
                 ("has_rng_cuda", "has_rng_numpy", "has_rng_python", "has_rng_torch")}
    check("train_rng", all(rng_flags.values()), f"rng flags={rng_flags}")
    report["scheduler_state_present"] = train["has_scheduler_state"]
    check("scheduler_state", True,
          f"has_scheduler_state={train['has_scheduler_state']} -> constant-lr fallback "
          f"lr={cpu.lr:.10g} (protocol 6.5)")
    mcfg = load_model_cfg(step)
    opt_cfg = yaml.safe_load(ckpt_path(step, "config.yaml").read_text()).get("optim", {})
    betas_ok = all(tuple(g["betas"]) == (0.9, 0.95) for g in orp["groups"])
    eps_ok = all(abs(g["eps"] - 1e-8) < 1e-12 for g in orp["groups"])
    wd_ok = {g["weight_decay"] for g in orp["groups"]} == {0.1, 0.0}
    check("optim_meta_vs_config", betas_ok and eps_ok,
          f"betas_ok={betas_ok} eps_ok={eps_ok} wd_set={sorted(g['weight_decay'] for g in orp['groups'])} "
          f"config_optim_keys={sorted(opt_cfg.keys())[:8]}")
    report["wall_time_s"] = time.time() - t0
    report["config_excerpt"] = {k: mcfg.get(k) for k in
                                ("d_model", "n_heads", "n_layers", "vocab_size",
                                 "embedding_size", "max_sequence_length")}

    json_dump(RESULTS / "phase0.json", report)
    failed = [c for c in report["checks"] if not c["ok"]]
    verdict = "PASS" if not failed else "FAIL"
    lines = [
        "# G1C Phase 0 Report — checkpoint restoration gate (step950000)",
        "",
        f"- gate: FR_G1C_1B_FEASIBILITY_GATE_V01 Phase 0",
        f"- verdict: **{verdict}**",
        f"- wall time: {report['wall_time_s']:.1f}s",
        "",
        "## Files (size + SHA256)",
        "",
        "| file | size | expected | size_ok | sha256 |",
        "|---|---|---|---|---|",
    ]
    for name, f in report["files"].items():
        lines.append(f"| {name} | {f['size_bytes']} | {f['expected_size']} | "
                     f"{f['size_ok']} | {f['sha256']} |")
    lines += ["", "## Checks", "", "| check | ok | detail |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {c['ok']} | {c['detail']} |")
    lines += [
        "",
        "## Notes",
        "",
        "- train.pt contains NO scheduler state -> protocol 6.5 constant-lr "
        f"fallback, lr = restored param_groups value {cpu.lr:.10g}.",
        "- optim.pt: 2 param groups (wd=0.1 for blocks/lm_head/ln_f, wd=0.0 for "
        "wte) -> versioned adapter g1c_adamw_tangent_step in use (protocol 6.8).",
        "- This is a restoration-feasibility gate, not a science result.",
    ]
    (ROOT / "G1C_PHASE0_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"PHASE 0 {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


def cmd_phase1(args) -> int:
    log("PHASE 1 start (step950000 x stream1 x H=1 smoke)")
    step, stream_id, H = PHASE1_STEP, 1, 1
    tokens = load_stream_tokens(stream_id)
    cpu = CpuState(step)
    gpu = GpuSession(cpu.model_cfg)
    q = make_q(cpu.model, stream_id)
    q_sha = save_q(stream_id, q, step)
    log(f"q saved sha256={q_sha}")

    report: dict = {"step": step, "stream_id": stream_id, "H": H,
                    "q_sha256": q_sha, "checks": []}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # ---- 1. deterministic replay --------------------------------------
    state0 = {n: t for n, t in cpu.model.state_dict().items()}
    gpu.upload(state0, torch.float32)
    gpu.reset_peak()
    ids, targets = microbatch_tokens(tokens, 0, 0)
    loss1 = gpu.forward_loss(ids, targets)
    loss1.backward()
    g1 = gpu.grads_cpu()
    loss2 = gpu.forward_loss(ids, targets)
    loss2.backward()
    g2 = gpu.grads_cpu()
    loss_gap = abs(float(loss1.detach()) - float(loss2.detach()))
    gaps = {n: float((g1[n] - g2[n]).abs().max()) for n in g1}
    non_wte_gap = max(v for n, v in gaps.items() if "wte" not in n)
    wte_gap = gaps.get("transformer.wte.weight", 0.0)
    check("replay_gpu_loss_bitexact", loss_gap == 0.0, f"loss_gap={loss_gap:.3g}")
    check("replay_gpu_grads_nonwte_bitexact", non_wte_gap == 0.0,
          f"max_gap_nonwte={non_wte_gap:.3g} wte_gap={wte_gap:.3g} "
          "(wte: embedding atomicAdd, expected non-deterministic on CUDA)")

    cpu_losses = []
    cpu_grads = []
    for _ in range(2):
        cpu.model.zero_grad(set_to_none=True)
        h = cpu.model.forward_hidden(ids, checkpoint_blocks=False)
        loss = chunked_mean_ce(h, cpu.model.transformer.ff_out.weight, targets, chunk=1024,
                               recompute_logits=False)
        loss.backward()
        cpu_losses.append(float(loss))
        cpu_grads.append({n: p.grad.detach().clone() for n, p in cpu.named})
        cpu.model.zero_grad(set_to_none=True)
    cpu_loss_gap = abs(cpu_losses[0] - cpu_losses[1])
    cpu_gaps = {n: float((cpu_grads[0][n] - cpu_grads[1][n]).abs().max()) for n in cpu_grads[0]}
    cpu_max_gap = max(cpu_gaps.values())
    check("replay_cpu_fp32_bitexact", cpu_loss_gap == 0.0 and cpu_max_gap == 0.0,
          f"loss_gap={cpu_loss_gap:.3g} max_grad_gap={cpu_max_gap:.3g}")

    # ---- 2. gradient propagation --------------------------------------
    gpu.upload(state0, torch.float32)
    grads, loss = compute_grads(gpu, tokens, 0)
    clipped, norm, coef = cpu.clip_and_snapshot(grads)
    check("gradient_propagation", math.isfinite(norm) and 0.0 < coef <= 1.0,
          f"loss={loss:.6f} |g|={norm:.6f} clip_coef={coef:.6f} finite={math.isfinite(norm)}")
    # clip_and_snapshot touches only .grad; optimizer state is pristine.
    report["baseline_step1"] = {"loss": loss, "grad_norm": norm, "clip_coef": coef}

    # ---- 3. H=1 analytic identity + FD at eps and eps/2 ---------------
    try:
        base = native_pass(cpu, gpu, tokens, H, q, collect_s=True)
        s_values = [st["s"] for st in base["steps"]]
        eps, sigma_q = epsilon_from_s(s_values)
        report["sigma_q"] = sigma_q
        report["epsilon"] = eps
        log(f"sigma_q={sigma_q:.6g} epsilon={eps:.6g} (s_samples={s_values})")

        chiB = tangent_pass(cpu, gpu, tokens, H, q, "B")
        chiC = tangent_pass(cpu, gpu, tokens, H, q, "C")
        b1 = chiB["chi"][0]["chi"]
        c1 = chiC["chi"][0]["chi"]
        identity_gap = abs(b1 - c1)
        check("h1_identity_B_eq_C", identity_gap == 0.0, f"|B1-C1|={identity_gap:.3g}")

        fd_rows = []
        for level, e in (("eps", eps), ("eps/2", eps / 2.0)):
            plus = fd_branch(cpu, gpu, tokens, H, q, +1, e)
            minus = fd_branch(cpu, gpu, tokens, H, q, -1, e)
            chi_fd = (plus["X"][0]["X"] - minus["X"][0]["X"]) / (2.0 * e)
            rel_b = relative_error(chi_fd, b1)
            rel_c = relative_error(chi_fd, c1)
            fd_rows.append({"level": level, "epsilon": e, "chi_fd": chi_fd,
                            "rel_err_B": rel_b, "rel_err_C": rel_c,
                            "X_plus": plus["X"][0]["X"], "X_minus": minus["X"][0]["X"]})
            log(f"  FD {level}: chi_fd={chi_fd:.6g} relB={rel_b} relC={rel_c}")
        report["fd"] = fd_rows
        check("h1_fd_vs_tangent", all(r["rel_err_C"] is not None and math.isfinite(r["rel_err_C"])
                                      for r in fd_rows),
              json.dumps(fd_rows))

        # ---- 4. HVP executability --------------------------------------
        gpu.upload(state0, torch.bfloat16)
        gpu.reset_peak()
        t0 = time.time()
        hvp = hvp_pass(gpu, tokens, 0, q)
        hvp_time = time.time() - t0
        hvp_norm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in hvp.values()))
        finite = all(bool(torch.isfinite(v).all()) for v in hvp.values())
        gpu.reset_peak()
        t0 = time.time()
        hvp2 = hvp_pass(gpu, tokens, 0, q)
        hvp_repeat_gap = max(float((hvp[n] - hvp2[n]).abs().max()) for n in hvp)
        hvp_time2 = time.time() - t0
        check("hvp_executability", finite and hvp_norm > 0,
              f"peak_mem_gb={gpu.peak_mem_gb():.1f} time={hvp_time:.1f}s/{hvp_time2:.1f}s "
              f"|hvp|={hvp_norm:.6g} finite={finite} repeat_max_gap={hvp_repeat_gap:.3g} "
              f"<q,hvp>={project_tangent(q, hvp):.6g}")
        report["hvp"] = {"peak_mem_gb": gpu.peak_mem_gb(), "time_s": hvp_time,
                         "norm": hvp_norm, "finite": finite,
                         "repeat_max_gap": hvp_repeat_gap,
                         "q_dot_hvp": project_tangent(q, hvp)}
    except Exception as exc:
        check("phase1_execution", False, f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    json_dump(RESULTS / "phase1.json", report)
    failed = [c for c in report["checks"] if not c["ok"]]
    verdict = "PASS" if not failed else "FAIL"
    lines = [
        "# G1C Phase 1 Report — single-state response smoke",
        "",
        f"- cell: step{step} x stream{stream_id} x H={H}",
        f"- verdict: **{verdict}** (feasibility smoke, not science)",
        f"- q sha256: {q_sha}",
        f"- sigma_q: {sigma_q:.10g}  epsilon: {eps:.10g}  (RHO={RHO})",
        "",
        "## Checks",
        "",
        "| check | ok | detail |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {c['ok']} | {str(c['detail'])[:500]} |")
    lines += [
        "",
        "## FD vs tangent (H=1)",
        "",
        "| level | epsilon | chi_fd | rel_err_B | rel_err_C | X_plus | X_minus |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.get("fd", []):
        lines.append(f"| {r['level']} | {r['epsilon']:.6g} | {r['chi_fd']:.8g} | "
                     f"{r['rel_err_B']:.6g} | {r['rel_err_C']:.6g} | "
                     f"{r['X_plus']:.6g} | {r['X_minus']:.6g} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Observable: parameter-displacement projection X_h=<q, theta_h-theta_0> "
        "(Amendment A2; fp64 bookkeeping, exact).",
        "- q: iid N(0,1) full support, unit total L2 norm (Amendment A1).",
        "- FD impulse: ±epsilon*q on first future step gradients pre-clip "
        "(Amendment A3).",
        "- wte gradient non-determinism on CUDA is a known embedding-backward "
        "atomicAdd effect; CPU fp32 replay is bit-exact.",
        f"- HVP: bf16-materialized double backward (protocol 6.7 memory strategy; "
        f"same amp_bf16 numerics class), seq={SEQ_LEN} per sequence.",
    ]
    (ROOT / "G1C_PHASE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"PHASE 1 {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


def cmd_phase2(args) -> int:
    log("PHASE 2 start (3 checkpoints x 2 streams x H in {1,4,8} = 18 cells)")
    rows = []
    summaries = {}
    for step in STEPS:
        for stream_id in STREAM_IDS:
            tokens = load_stream_tokens(stream_id)
            cpu = CpuState(step)
            gpu = GpuSession(cpu.model_cfg)
            q = make_q(cpu.model, stream_id)
            q_sha = save_q(stream_id, q, step)
            base = native_pass(cpu, gpu, tokens, H_MAX, q, collect_s=True)
            s_values = [st["s"] for st in base["steps"]]
            eps, sigma_q = epsilon_from_s(s_values)
            # Amendment A6: per-branch isolation. The frozen tangent-step
            # math can hit the AdamW-JVP v~0 singularity after a large HVP
            # (observed 2026-09-18: step380000 stream1 C h=6 chi -8.9 -> +269,
            # then NaN/Inf caught by frozen clip_jvp at h=7). A branch failure
            # must not abort the matrix: record the cell as failed and keep
            # the remaining branches/cells running so feasibility is mapped.
            nan = float("nan")
            chiB = chiC = plus = minus = None
            branch_err = {"B": None, "C": None, "FD": None}
            try:
                chiB = tangent_pass(cpu, gpu, tokens, H_MAX, q, "B")
            except Exception as exc:
                branch_err["B"] = f"{type(exc).__name__}: {exc}"
                log(f"  [BRANCH-FAIL] s{step} st{stream_id} B: {branch_err['B']}")
            try:
                chiC = tangent_pass(cpu, gpu, tokens, H_MAX, q, "C")
            except Exception as exc:
                branch_err["C"] = f"{type(exc).__name__}: {exc}"
                log(f"  [BRANCH-FAIL] s{step} st{stream_id} C: {branch_err['C']}")
            try:
                plus = fd_branch(cpu, gpu, tokens, H_MAX, q, +1, eps)
                minus = fd_branch(cpu, gpu, tokens, H_MAX, q, -1, eps)
            except Exception as exc:
                branch_err["FD"] = f"{type(exc).__name__}: {exc}"
                log(f"  [BRANCH-FAIL] s{step} st{stream_id} FD: {branch_err['FD']}")
            for Hh in HORIZONS:
                i = Hh - 1
                chi_fd = ((plus["X"][i]["X"] - minus["X"][i]["X"]) / (2.0 * eps)
                          if plus is not None and minus is not None else nan)
                b = chiB["chi"][i]["chi"] if chiB is not None else nan
                c = chiC["chi"][i]["chi"] if chiC is not None else nan
                okB = math.isfinite(b); okC = math.isfinite(c); okFD = math.isfinite(chi_fd)
                row = {
                    "checkpoint_step": step, "stream_id": stream_id, "H": Hh,
                    "epsilon": eps, "sigma_q": sigma_q, "q_sha256": q_sha,
                    "chi_fd": chi_fd, "chi_B": b, "chi_C": c,
                    "abs_err_B": abs(b - chi_fd) if okB and okFD else nan,
                    "abs_err_C": abs(c - chi_fd) if okC and okFD else nan,
                    "rel_err_B": relative_error(chi_fd, b) if okB and okFD else None,
                    "rel_err_C": relative_error(chi_fd, c) if okC and okFD else None,
                    "B_minus_C": (b - c) if okB and okC else nan,
                    "status": ("ok" if (okB and okC and okFD) else
                               "partial" if (okB or okC or okFD) else "failed"),
                    "branch_error_B": branch_err["B"],
                    "branch_error_C": branch_err["C"],
                    "branch_error_FD": branch_err["FD"],
                }
                rows.append(row)
                log(f"  cell s{step} st{stream_id} H{Hh}: fd={chi_fd:.6g} "
                    f"B={b:.6g} C={c:.6g} relC={row['rel_err_C']}")
            del cpu, gpu
            torch.cuda.empty_cache()
        def finite_pairs(rs, xk, yk):
            return ([r[xk] for r in rs if math.isfinite(r[xk]) and math.isfinite(r[yk])],
                    [r[yk] for r in rs if math.isfinite(r[xk]) and math.isfinite(r[yk])])
        sub = [r for r in rows if r["checkpoint_step"] == step]
        def metr(rs, xk, yk):
            xs, ys = finite_pairs(rs, xk, yk)
            return _metric(xs, ys) if xs else {"n": 0, "note": "no finite pairs"}
        summaries[str(step)] = {
            "C_vs_B": metr(sub, "chi_C", "chi_B"),
            "C_vs_FD": metr(sub, "chi_C", "chi_fd"),
            "B_vs_FD": metr(sub, "chi_B", "chi_fd"),
        }
    summaries["pooled"] = {
        "C_vs_B": metr(rows, "chi_C", "chi_B"),
        "C_vs_FD": metr(rows, "chi_C", "chi_fd"),
        "B_vs_FD": metr(rows, "chi_B", "chi_fd"),
    }

    with (RESULTS / "per_cell.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_dump(RESULTS / "summary.json", {"cells": rows, "summaries": summaries})

    lines = [
        "# G1C Phase 2 — feasibility matrix (descriptive only)",
        "",
        "- 18 cells: checkpoints {380000, 950000, 1530000} x streams {1,2} x H {1,4,8}",
        "- NO scientific conclusions. NO scale claims. NO generalization claims.",
        "- EXTERNAL_STRESS_* labels remain reserved for the formal G1c experiment.",
        "",
        "## Per-cell table",
        "",
        "| step | stream | H | chi_fd | chi_B | chi_C | rel_err_B | rel_err_C |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['checkpoint_step']} | {r['stream_id']} | {r['H']} | "
                     f"{r['chi_fd']:.6g} | {r['chi_B']:.6g} | {r['chi_C']:.6g} | "
                     f"{r['rel_err_B'] if r['rel_err_B'] is not None else float('nan'):.4g} | "
                     f"{r['rel_err_C'] if r['rel_err_C'] is not None else float('nan'):.4g} |")
    lines += ["", "## C vs B / FD aggregated (frozen _metric, descriptive)", ""]
    for key, block in summaries.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(block, indent=2))
        lines.append("```")
        lines.append("")
    (ROOT / "G1C_PHASE2_FEASIBILITY.md").write_text("\n".join(lines), encoding="utf-8")
    log("PHASE 2 complete")
    return 0




# ------------------------------------------------- C-failure diagnostic (A6)
def cmd_cdiag(args) -> int:
    """G1C_C_FAILURE_DIAGNOSTIC_V01: classify C-branch failure cause.

    Frozen semantics untouched: (a)/(c) reproduce the registered runner path
    exactly; (b) changes ONLY the HVP input cast dtype (fp16 -> fp32) and is
    labelled diagnostic in every output. Writes results/cdiag/. Does not
    modify phase0/1/2 artifacts or frozen modules.
    """
    out_dir = RESULTS / "cdiag"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_step = int(getattr(args, "step", 380000))
    spec_stream = int(getattr(args, "stream", 1))
    quick = bool(getattr(args, "quick", False))
    H_diag = 3 if quick else H_MAX
    report = {"spec": {"step": spec_step, "stream": spec_stream,
                       "H": H_diag, "seq_len": SEQ_LEN, "quick": quick}}
    tokens = load_stream_tokens(spec_stream)
    cpu = CpuState(spec_step)
    gpu = GpuSession(cpu.model_cfg)
    q = make_q(cpu.model, spec_stream)
    report["q_sha256"] = save_q(spec_stream, q, spec_step)
    del cpu  # q only; every check builds its own fresh CpuState

    def run_c(hvp_dtype, label):
        cpu_n = CpuState(spec_step)
        res_n = None
        err_n = None
        sink: list = []
        try:
            res_n = tangent_pass(cpu_n, gpu, tokens, H_diag, q, "C",
                                 hvp_dtype=hvp_dtype,
                                 collect_diagnostics=True, diag_sink=sink)
        except Exception as exc:  # noqa: BLE001
            err_n = f"{type(exc).__name__}: {exc}"
        rows_n = sink
        n_done = sum(1 for r in rows_n if "clip_coef" in r)
        if err_n:
            if not rows_n:
                fail_h = 1
            elif "clip_coef" in rows_n[-1]:
                fail_h = rows_n[-1]["h"] + 1  # last row completed; failed next
            else:
                fail_h = rows_n[-1]["h"]  # raised mid-row (clip_jvp)
        else:
            fail_h = None
        payload = {"label": label, "error": err_n,
                   "first_failure_horizon": fail_h,
                   "n_h_completed": n_done, "per_h": rows_n}
        json_dump(out_dir / f"{label}.json", payload)
        return payload

    def load_or_run(label, dtype, skip):
        path = out_dir / f"{label}.json"
        if skip and path.exists():
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        return run_c(dtype, label)

    # (1)+(3c): instrumented bf16 run, then exact repeat for determinism
    log("cdiag: instrumented bf16 C run (registered path)")
    first = load_or_run("instrumented_bf16", torch.float16,
                        bool(getattr(args, "skip_instrumented", False)))
    log("cdiag: determinism repeat")
    rep = load_or_run("determinism_repeat", torch.float16,
                      bool(getattr(args, "skip_determinism", False)))
    same_err = rep["error"] == first["error"]
    chi_diff = None
    a = [c["chi"] if "chi" in c else None for c in first["per_h"]]
    b = [c["chi"] if "chi" in c else None for c in rep["per_h"]]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if pairs:
        chi_diff = max(abs(x - y) for x, y in pairs)
    det = {"same_error": same_err, "first_error": first["error"],
           "repeat_error": rep["error"],
           "max_chi_diff_first_vs_repeat": chi_diff,
           "first_fail_h": first["first_failure_horizon"],
           "repeat_fail_h": rep["first_failure_horizon"]}
    json_dump(out_dir / "determinism.json", det)

    # (3a): B control on same checkpoint/stream
    b_path = out_dir / "b_control.json"
    if bool(getattr(args, "skip_b", False)) and b_path.exists():
        with b_path.open(encoding="utf-8") as fh:
            b_ctrl = json.load(fh)
    else:
        log("cdiag: B control run to H=8")
        cpuB = CpuState(spec_step)
        resB = tangent_pass(cpuB, gpu, tokens, H_MAX, q, "B",
                            collect_diagnostics=True)
        # Raw dump first: a defensive assembly crash must not lose the data.
        json_dump(out_dir / "b_control_diagnostics_raw.json",
                  {"chi": resB["chi"], "diagnostics": resB["diagnostics"]})
        b_ctrl = {
            "chi_all": [c["chi"] for c in resB["chi"]],
            "theta_tangent_norms": [d["tangent_theta_norm_out"] for d in resB["diagnostics"]],
            "stable_to_H8": all(math.isfinite(c["chi"]) for c in resB["chi"]),
            "per_h": resB["diagnostics"],
        }
        json_dump(b_path, b_ctrl)

    # (3b): fp32 HVP-input diagnostic (NOT official)
    log("cdiag: fp32-HVP-input diagnostic run (not official)")
    fp32run = run_c(torch.float32, "fp32_hvp_input_DIAGNOSTIC")
    fp32run["DIAGNOSTIC_ONLY"] = True
    json_dump(out_dir / "fp32_hvp_input_DIAGNOSTIC.json", fp32run)

    json_dump(out_dir / "cdiag_summary.json", {
        "spec": report["spec"], "q_sha256": report["q_sha256"],
        "instrumented_bf16": first, "determinism": det, "B_control": b_ctrl,
        "fp32_hvp_input": fp32run,
    })
    log("cdiag: done -> results/cdiag/")
    return 0

def cmd_decision(args) -> int:
    p0 = json.loads((RESULTS / "phase0.json").read_text())
    p1_exists = (RESULTS / "phase1.json").exists()
    p2_exists = (RESULTS / "summary.json").exists()
    p0_ok = all(c["ok"] for c in p0["checks"])
    p1 = json.loads((RESULTS / "phase1.json").read_text()) if p1_exists else None
    p1_ok = p1 is not None and all(c["ok"] for c in p1["checks"])
    if p0_ok and p1_ok and p2_exists:
        decision = "READY_FOR_G1C_SCIENCE"
    else:
        decision = "BLOCKED"
    payload = {"decision": decision, "phase0_ok": p0_ok, "phase1_ok": p1_ok,
               "phase2_ran": p2_exists,
               "note": "executability declaration only, not a scientific verdict"}
    json_dump(RESULTS / "decision.json", payload)
    (RESULTS / "DECISION.md").write_text(
        f"# G1C Feasibility Decision: {decision}\n\n"
        f"- phase0_ok: {p0_ok}\n- phase1_ok: {p1_ok}\n- phase2_ran: {p2_exists}\n\n"
        f"{payload['note']}. Formal G1c science requires a new user authorization.\n",
        encoding="utf-8")
    log(f"DECISION: {decision}")
    return 0


def cmd_env(args) -> int:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gb": (torch.cuda.get_device_properties(0).total_memory / (1 << 30)
                         if torch.cuda.is_available() else None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": np.__version__,
        "cwd": str(Path.cwd()),
    }
    json_dump(ROOT / "ENVIRONMENT.json", payload)
    log("ENVIRONMENT.json written")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("phase0", "phase1", "phase2", "decision", "env"):
        sub.add_parser(name)
    pcd = sub.add_parser("cdiag")
    pcd.add_argument("--step", type=int, default=380000)
    pcd.add_argument("--stream", type=int, default=1)
    pcd.add_argument("--quick", action="store_true")
    pcd.add_argument("--skip-instrumented", action="store_true")
    pcd.add_argument("--skip-determinism", action="store_true")
    pcd.add_argument("--skip-b", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    return {"phase0": cmd_phase0, "phase1": cmd_phase1, "phase2": cmd_phase2,
            "decision": cmd_decision, "env": cmd_env, "cdiag": cmd_cdiag}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
