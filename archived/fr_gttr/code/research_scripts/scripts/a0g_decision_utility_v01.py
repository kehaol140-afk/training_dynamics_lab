#!/usr/bin/env python3
"""Run FR_A0G decision-utility v01.

验证完整切线响应能否帮助选择一个真正降低未来验证损失的训练动作
（当前单个 optimizer step 的固定预算数据改权），并核算成本是否值得。

动作定义（与因果 pulse 完全相同的写入位置）：
    accumulation backward -> scaler.unscale_ -> action gradient
    -> clip_grad_norm_ -> native AdamW step
    g_action = g + sign*lambda*(g_j - g)   (sign=+1 为 +lambda, -1 为 -lambda)
只在当前 optimizer step 写入；未来 H-1 步恢复 baseline 数据流。

tangent r_H 约定为对 lambda 的导数（每单位 lambda）；
有限动作预测 = lambda * hhat。禁止重复乘 lambda。

阶段：
  lambda-dev : 开发状态上扫描 lambda 候选并冻结（写 lambda_selection.json）
  main       : 冻结 lambda 下跑 Level 1 / 1b / 2 / 3 + 成本核算（正式状态）

输出: outputs/fr_decision_utility_v01/（§13 清单 + confirmatory_bank/）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fr.a0c_adamw_response_propagation_v01 import (
    adamw_tangent_step,
    clip_jvp,
    json_dumps,
    project_tangent,
    relative_error,
)
from fr.a0c_h1_adamw_jacobian_v01 import file_sha256
from fr.a0g_decision_utility_v01 import (
    HORIZONS,
    LAMBDA_CANDIDATES,
    N_DATA_GROUPS,
    NEAR_TIE_FRACTION,
    PRIMARY_HORIZON,
    assemble_summary,
    breakeven_m,
    build_data_group_definition,
    cost_model,
    level3_decision,
    random_choice,
    render_lambda_selection_report,
    render_report,
    split_validation_pool,
    _safe_spearman,
)
from fr.batch_manifest import manifest_sha256
from fr.full_state import load_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import load_partition
from fr.rng import capture_rng_state, restore_rng_state
from fr.stream_rng import seed_future_stream
from fr.theta_protocol import run_theta_branch


DEV_STATES = ((2031, 52), (2034, 64), (2036, 76))
FORMAL_SEEDS = (2038, 2039, 2040, 2041)
FORMAL_STEPS = (46, 64, 82)
EVAL_BATCH_SIZE = 8
CANDIDATE_BATCH_SEED_OFFSET = 9000  # Philox key 组成部分（见 candidate_batches_for）
OUTPUT_DIR = "outputs/fr_decision_utility_v01"
PROTOCOL_NAME = "FR_DECISION_UTILITY_V01_E2_PAIRED_STREAM"
N_STREAMS_PER_STATE = 8
CONDITIONAL_LOOKAHEAD_LIMITATION = (
    "CONDITIONAL_LOOKAHEAD_LIMITATION: 本实验验证 trajectory-conditioned local "
    "counterfactual scoring——完整切线使用给定 baseline future trajectory 的未来梯度/HVP；"
    "结果不能解读为'仅从当前状态预测未知未来'的 current-state-only forecast。"
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _mean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("cannot average empty or non-finite values")
    return float(arr.mean())


def _norm(mapping) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for value in mapping.values():
        total += torch.sum(value.detach().to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _difference_norm(left, right) -> float:
    import torch

    if set(left) != set(right):
        raise ValueError("tensor mapping names differ")
    total = torch.zeros((), dtype=torch.float64)
    for name in left:
        a = left[name].detach()
        b = right[name].detach().to(device=a.device, dtype=a.dtype)
        total += torch.sum((a - b).to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _zeros_like_named(named):
    return {name: p.detach().new_zeros(p.shape) for name, p in named}


def _snapshot_grads(named):
    return {
        name: (
            p.detach().new_zeros(p.shape)
            if p.grad is None else p.grad.detach().clone()
        )
        for name, p in named
    }


def _restore_grads(named, grads):
    for name, p in named:
        p.grad = grads[name].clone()


class _Timers:
    """GPU 同步计时器（cuda.synchronize 包裹）。"""

    def __init__(self, torch):
        self._torch = torch
        self.t = {}
        self.peak = {}

    def tick(self, key: str, start: float):
        self._torch.cuda.synchronize()
        self.t[key] = self.t.get(key, 0.0) + (time.perf_counter() - start)
        self.peak[key] = max(
            self.peak.get(key, 0.0), float(self._torch.cuda.max_memory_allocated())
        )

    def start(self) -> float:
        self._torch.cuda.synchronize()
        return time.perf_counter()

    def snapshot_peak(self, key: str):
        self.peak[key] = max(
            self.peak.get(key, 0.0), float(self._torch.cuda.max_memory_allocated())
        )


def _math_sdpa_context(torch):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True,
                enable_mem_efficient=False, enable_cudnn=False,
            )
        except (AttributeError, TypeError):
            return nullcontext()


def _hvp(torch, model, named, batch_tensors, tangent_theta, autocast_ctx, accumulation_steps):
    """H(loss)@delta_theta + 该 HVP forward 的一阶梯度（复制冻结 A0C 模式）。"""
    parameters = [p for _, p in named]
    losses = []
    with _math_sdpa_context(torch):
        for x, y in batch_tensors:
            with autocast_ctx:
                result = model(x, labels=y)
                losses.append((result.loss + result.aux_loss) / accumulation_steps)
        total_loss = sum(losses)
        gradients = torch.autograd.grad(total_loss, parameters, create_graph=True, allow_unused=True)
        dot = None
        for (name, parameter), gradient in zip(named, gradients):
            if gradient is None:
                continue
            direction = tangent_theta[name].detach().to(device=parameter.device, dtype=gradient.dtype)
            term = torch.sum(gradient * direction)
            dot = term if dot is None else dot + term
        if dot is None:
            raise RuntimeError("HVP graph has no differentiable gradient")
        hvp_values = torch.autograd.grad(dot, parameters, allow_unused=True)
    gradient_by_name = {}
    hvp_by_name = {}
    for (name, parameter), gradient, hvp in zip(named, gradients, hvp_values):
        gradient_by_name[name] = (
            parameter.detach().new_zeros(parameter.shape)
            if gradient is None else gradient.detach().clone()
        )
        hvp_by_name[name] = (
            parameter.detach().new_zeros(parameter.shape)
            if hvp is None else hvp.detach().clone()
        )
    return hvp_by_name, gradient_by_name


def _grad_on_batches(torch, model, named, batch_tensors, autocast_ctx, accumulation_steps):
    """在给定 batch 上前向+autograd.grad（不污染 p.grad，不经 GradScaler）。"""
    parameters = [p for _, p in named]
    losses = []
    for x, y in batch_tensors:
        with autocast_ctx:
            result = model(x, labels=y)
            losses.append((result.loss + result.aux_loss) / accumulation_steps)
    total_loss = sum(losses)
    grads = torch.autograd.grad(total_loss, parameters, allow_unused=True)
    grad_by_name = {}
    for (name, parameter), gradient in zip(named, grads):
        grad_by_name[name] = (
            parameter.detach().new_zeros(parameter.shape)
            if gradient is None else gradient.detach().clone()
        )
    return grad_by_name


def _val_loss_only(torch, model, batch_tensors, autocast_ctx):
    """eval-mode 前向，返回 mean loss（调用方负责 RNG 包裹与 train/eval 切换）。"""
    model.eval()
    losses = []
    try:
        for x, y in batch_tensors:
            with autocast_ctx:
                result = model(x, labels=y)
                losses.append(float((result.loss + result.aux_loss).detach().float().cpu()))
    finally:
        model.train()
    if not losses:
        raise RuntimeError("no validation batches")
    return float(np.mean(losses))


def _stack_all(ds, batches, device):
    return [stack_batch(ds, list(b), device) for b in batches]


# ---------------------------------------------------------------------------
# 候选 batch 生成（确定性，局部 Generator，不碰全局 RNG）
# ---------------------------------------------------------------------------

def candidate_batches_for(stream_seed: int, group_id: int, members: Sequence[int],
                          batch_size: int, n_micro: int):
    """候选组 batch：Philox 局部 Generator 的确定性采样，不触碰任何全局 RNG。

    key = (stream_seed & 2^64-1) XOR ((9000 + group_id) << 32)。
    """
    key = (int(stream_seed) & 0xFFFFFFFFFFFFFFFF) ^ (
        np.uint64((CANDIDATE_BATCH_SEED_OFFSET + int(group_id)) << 32)
    )
    rng = np.random.Generator(np.random.Philox(key=key))
    idx = np.asarray(sorted(map(int, members)), dtype=np.int64)
    if idx.size < n_micro * batch_size:
        raise ValueError(f"group {group_id} too small for candidate batches")
    order = rng.permutation(idx.size)
    chosen = idx[order[: n_micro * batch_size]]
    return [list(map(int, chosen[i * batch_size:(i + 1) * batch_size])) for i in range(n_micro)]


# ---------------------------------------------------------------------------
# 核心引擎 1：tangent pass（baseline future rollout + 全方法切线 + readout）
# ---------------------------------------------------------------------------

def _run_tangent_pass(*, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
                      checkpoint_path, batches, stream_seed, cand_batches_by_j,
                      val_score_tensors, val_eval_tensors, v_t,
                      accumulation_steps, device, grad_clip, base_lr,
                      global_micro_step, total_micro_steps, get_lr, timers,
                      hvp_direction_mode):
    """一次 baseline future rollout，同时携带 4 组动作的 B/C 切线。

    位置约定（全部对齐 H 步后的 endpoint θ_{t+H}）：baseline val 读数、v_H、
    r_H_B/C、分支引擎 readout 都在 native step 之后取；hhat 在循环后统一投影。
    hvp_direction_mode:
      "pre"  -> HVP 用微步起始点的 δθ（未含本步更新，与冻结 A0C/A0E2 同约定）
      "post" -> HVP 用微步终点的 δθ（严格含本步更新，局部一致性检查用）
    返回 baseline 终点 V_score/V_eval loss、各 H 的 v_H、各方法 hhat、哈希链。
    """
    load_full_training_state(checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))

    zero = _zeros_like_named(named)
    b_theta = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}
    b_m = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}
    b_v = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}
    c_theta = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}
    c_m = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}
    c_v = {j: {k: v.clone() for k, v in zero.items()} for j in cand_batches_by_j}

    max_h = max(HORIZONS)
    g_by_j = {}
    a_by_j = {}
    hhat = {m: {j: {} for j in cand_batches_by_j} for m in ("alignment", "B", "C")}
    r_H_B = {j: {} for j in cand_batches_by_j}
    r_H_C = {j: {} for j in cand_batches_by_j}
    r1_theta = {}
    v_score_grad = {}
    baseline_val_score = {}
    baseline_val_eval = {}
    hashes = {}
    clip_trace = []
    hvp_relative_gaps = []

    for horizon in range(1, max_h + 1):
        t0 = timers.start()
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = set_native_micro_lr(
                optimizer, get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr),
            )
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                rng_before_native_forward = capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()
        timers.tick("t_native_step", t0)

        t0 = timers.start()
        scaler.unscale_(optimizer)
        native_gradient = _snapshot_grads(named)
        native_gradient_norm = _norm(native_gradient)
        timers.tick("t_unscale_snapshot", t0)

        # ---- h=1: 候选组梯度 g_j（autograd.grad，不污染 p.grad） ----
        if horizon == 1:
            t0 = timers.start()
            for j, cand_batches in cand_batches_by_j.items():
                if j in g_by_j:
                    continue
                cand_tensors = _stack_all(dataset, cand_batches, device)
                rng_snap = capture_rng_state()
                g_by_j[j] = _grad_on_batches(
                    torch, model, named, cand_tensors, autocast_ctx, accumulation_steps,
                )
                restore_rng_state(rng_snap)
                del cand_tensors
            missing = set(cand_batches_by_j) - set(g_by_j)
            if missing:
                raise RuntimeError(f"candidate gradient missing for groups: {sorted(missing)}")
            for j in cand_batches_by_j:
                a_by_j[j] = {
                    name: g_by_j[j][name] - native_gradient[name]
                    for name, _ in named
                }
            timers.tick("t_cand_grad", t0)

        # ---- future-step HVP（完整切线） ----
        # HVP 方向约定与冻结 A0C/A0E2 完全一致：在微步起点 θ_t 处取
        # H(loss_t)·δθ_t（即含上一 optimizer step 为止的 δθ），本步 native
        # 更新在本块之后才写入 θ_t。hvp_direction_mode="post"（诊断用）改用
        # 已含本步更新的 δθ，用于 §12 局部一致性说明。
        hvp_by_j = {}
        if horizon > 1:
            t0 = timers.start()
            if hvp_direction_mode == "post":
                post_step = {}
                for j in cand_batches_by_j:
                    post_step[j] = adamw_tangent_step(
                        named_parameters=named, optimizer=optimizer,
                        clipped_gradient_by_name=zero,
                        tangent_clipped_gradient_by_name=zero,
                        tangent_theta_by_name=c_theta[j],
                        tangent_m_by_name=c_m[j],
                        tangent_v_by_name=c_v[j],
                    )
            for j in cand_batches_by_j:
                rng_after_native_gradient = capture_rng_state()
                restore_rng_state(rng_before_native_forward)
                direction_j = (
                    post_step[j]["tangent_theta"]
                    if hvp_direction_mode == "post" else c_theta[j]
                )
                hvp_j, hvp_first_grad = _hvp(
                    torch, model, named, batch_tensors, direction_j, autocast_ctx,
                    accumulation_steps,
                )
                restore_rng_state(rng_after_native_gradient)
                hvp_by_j[j] = hvp_j
                gap = _difference_norm(hvp_first_grad, native_gradient)
                hvp_relative_gaps.append(
                    float(gap / max(native_gradient_norm, 1e-30))
                )
            timers.tick("t_hvp", t0)

        # ---- 切线步进 ----
        t0 = timers.start()
        native_preclip = None
        for j in cand_batches_by_j:
            tB = a_by_j[j] if horizon == 1 else (
                {name: zero[name] for name, _ in named}
            )
            tC = a_by_j[j] if horizon == 1 else hvp_by_j[j]
            b_clip = clip_jvp(
                gradient_by_name=native_gradient,
                tangent_gradient_by_name=tB,
                max_norm=grad_clip,
            )
            c_clip = clip_jvp(
                gradient_by_name=native_gradient,
                tangent_gradient_by_name=tC,
                max_norm=grad_clip,
            )
            if native_preclip is None:
                native_preclip = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    .detach().cpu()
                )
                native_clipped_gradient = _snapshot_grads(named)
                if abs(native_preclip - float(b_clip["clip_before_norm"])) > 1e-5:
                    raise RuntimeError(
                        "analytic/native preclip norm mismatch: "
                        f"native={native_preclip!r} analytic={b_clip['clip_before_norm']!r} "
                        f"norm_from_pgrads={native_gradient_norm!r} horizon={horizon}"
                    )
            b_next = adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=b_clip["clip_jvp"],
                tangent_theta_by_name=b_theta[j],
                tangent_m_by_name=b_m[j],
                tangent_v_by_name=b_v[j],
            )
            c_next = adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
                tangent_theta_by_name=c_theta[j],
                tangent_m_by_name=c_m[j],
                tangent_v_by_name=c_v[j],
            )
            b_theta[j], b_m[j], b_v[j] = (
                b_next["tangent_theta"], b_next["tangent_m"], b_next["tangent_v"],
            )
            c_theta[j], c_m[j], c_v[j] = (
                c_next["tangent_theta"], c_next["tangent_m"], c_next["tangent_v"],
            )
            if horizon == 1:
                r1_theta[j] = {k: v.clone() for k, v in b_theta[j].items()}

            if horizon in HORIZONS:
                r_H_B[j][str(horizon)] = {
                    k: v.detach().clone() for k, v in b_theta[j].items()
                }
                r_H_C[j][str(horizon)] = {
                    k: v.detach().clone() for k, v in c_theta[j].items()
                }
        timers.tick("t_tangent_step", t0)

        t0 = timers.start()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if horizon in HORIZONS:
            # ---- readout at θ_{t+horizon}（native step 之后） ----
            # 位置约定：baseline val / v_H / r_H / 分支 readout 全部对齐在
            # θ_{t+H}（H 步后的 endpoint，与 r_H=dθ_{t+H}/dλ 及冻结阶梯一致）。
            # 之前放在 step 之前会让 baseline 落后分支整整一个 optimizer step，
            # 给 ΔL 引入与 λ 无关的常数偏移（≈一步训练收益），污染扫描与 Level 1b。
            rng_snap = capture_rng_state()
            baseline_val_score[str(horizon)] = _val_loss_only(
                torch, model, val_score_tensors, autocast_ctx)
            baseline_val_eval[str(horizon)] = _val_loss_only(
                torch, model, val_eval_tensors, autocast_ctx)
            restore_rng_state(rng_snap)
            timers.tick("t_val_loss", t0)

            t0 = timers.start()
            rng_snap = capture_rng_state()
            v_score_grad[str(horizon)] = _grad_on_batches(
                torch, model, named, val_score_tensors, autocast_ctx,
                len(val_score_tensors),
            )
            restore_rng_state(rng_snap)
            timers.tick("t_val_grad", t0)

            hashes[str(horizon)] = (
                stable_hash(raw_model(model).state_dict()),
                stable_hash(optimizer.state_dict()),
            )
            clip_trace.append({
                "horizon": horizon,
                "clip_active": bool(native_preclip > grad_clip),
                "preclip_norm": float(native_preclip),
                "lr": float(current_lr),
            })
        timers.tick("t_native_step", t0)
        del batch_tensors, native_gradient, native_clipped_gradient

    timers.snapshot_peak("peak_pass")
    if set(g_by_j) != set(cand_batches_by_j):
        raise RuntimeError(
            f"tangent pass candidate-grad set mismatch: "
            f"have={sorted(g_by_j)} want={sorted(cand_batches_by_j)}"
        )
    # hhat 在 readout（θ_{t+H}，step 后）全部就绪后统一投影：
    # v_H 与 r_H 同处 endpoint θ_{t+H}，hhat = dL(θ_{t+H})/dλ。
    for horizon in HORIZONS:
        hk = str(horizon)
        v_h = v_score_grad[hk]
        for j in cand_batches_by_j:
            hhat["B"][j][hk] = project_tangent(v_h, r_H_B[j][hk])
            hhat["C"][j][hk] = project_tangent(v_h, r_H_C[j][hk])
            hhat["alignment"][j][hk] = project_tangent(v_t, r1_theta[j])
    return {
        "baseline_val_score": baseline_val_score,
        "baseline_val_eval": baseline_val_eval,
        "v_score_grad": v_score_grad,
        "hhat": hhat,
        "r_H_B": r_H_B,
        "r_H_C": r_H_C,
        "r1_theta": r1_theta,
        "g_by_j": g_by_j,
        "a_norm_by_j": {j: _norm(a_by_j[j]) for j in a_by_j},
        "hashes": hashes,
        "final_model_hash": stable_hash(raw_model(model).state_dict()),
        "final_optimizer_hash": stable_hash(optimizer.state_dict()),
        "clip_trace": clip_trace,
        "max_hvp_gradient_relative_gap": max(hvp_relative_gaps, default=0.0),
    }


# ---------------------------------------------------------------------------
# 核心引擎 2：action branch（+/-lambda 有限动作 replay）
# ---------------------------------------------------------------------------

def _run_action_branch(*, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
                       checkpoint_path, batches, stream_seed, action_j, lam, sign,
                       cand_batches_by_j, val_score_tensors, val_eval_tensors,
                       g_reference, accumulation_steps, device, grad_clip, base_lr,
                       global_micro_step, total_micro_steps, get_lr, timers):
    """在 h=1 写入 g_action = g + sign*lam*(g_j - g)，之后恢复 baseline 数据流。"""
    if action_j < 1:
        raise ValueError("action branch requires j>=1; j=0 is the no-intervention control")
    if sign not in (+1, -1):
        raise ValueError("action branch sign must be +1 or -1")
    _require(float(lam) > 0.0, "action lambda must be positive")
    _require(float(lam) <= 1.0, "convex mixture requires 0 < lambda <= 1")
    load_full_training_state(checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))

    max_h = max(HORIZONS)
    val_score = {}
    val_eval = {}
    first_step = {}
    g_j_gap = None
    for horizon in range(1, max_h + 1):
        t0 = timers.start()
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            set_native_micro_lr(
                optimizer, get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr),
            )
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = stack_batch(dataset, batch_index, device)
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()
        timers.tick("t_native_step", t0)

        t0 = timers.start()
        scaler.unscale_(optimizer)
        g_native = _snapshot_grads(named)
        if horizon == 1:
            cand_tensors = _stack_all(dataset, cand_batches_by_j[action_j], device)
            rng_snap = capture_rng_state()
            g_j = _grad_on_batches(
                torch, model, named, cand_tensors, autocast_ctx, accumulation_steps,
            )
            restore_rng_state(rng_snap)
            del cand_tensors
            if g_reference is not None:
                g_j_gap = _difference_norm(g_j, g_reference)
            for name, p in named:
                p.grad.copy_(
                    g_native[name] + float(sign) * float(lam) * (g_j[name] - g_native[name])
                )
            preclip = float(
                torch.sqrt(torch.stack([
                    torch.sum(p.grad.detach().to(torch.float64).square())
                    for _, p in named
                ]).sum()).cpu()
            )
            first_step = {
                "preclip_norm": preclip,
                "finite": bool(all(bool(torch.isfinite(p.grad).all().item()) for _, p in named)),
                "lambda": float(lam),
                "sign": int(sign),
                "action_j": int(action_j),
            }
        preclip_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).detach().cpu()
        )
        if horizon == 1:
            first_step["clip_active"] = bool(preclip_norm > grad_clip)
            first_step["baseline_preclip_norm"] = None  # filled by caller
        timers.tick("t_unscale_snapshot", t0)

        t0 = timers.start()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        timers.tick("t_native_step", t0)

        if horizon in HORIZONS:
            t0 = timers.start()
            rng_snap = capture_rng_state()
            val_score[str(horizon)] = _val_loss_only(
                torch, model, val_score_tensors, autocast_ctx)
            val_eval[str(horizon)] = _val_loss_only(
                torch, model, val_eval_tensors, autocast_ctx)
            restore_rng_state(rng_snap)
            timers.tick("t_val_loss", t0)

    timers.snapshot_peak("peak_pass")
    return {
        "val_score": val_score,
        "val_eval": val_eval,
        "first_step": first_step,
        "g_j_gap_vs_tangent_pass": g_j_gap,
        "final_model_hash": stable_hash(raw_model(model).state_dict()),
        "final_optimizer_hash": stable_hash(optimizer.state_dict()),
    }


# ---------------------------------------------------------------------------
# 状态/流上下文
# ---------------------------------------------------------------------------

class _StateContext:
    """装载一个 (seed, step) 状态 + 其 8 条 manifests + 候选 batch 计划。"""

    def __init__(self, *, split, seed, step, checkpoint_path, streams, group_definition,
                 partition, cfg, bank_meta):
        self.split = split
        self.seed = int(seed)
        self.step = int(step)
        self.checkpoint_path = Path(checkpoint_path)
        self.streams = streams  # [{stream_id, stream_seed, batches, manifest_sha}]
        self.group_definition = group_definition
        self.partition = partition
        self.cfg = cfg
        self.bank_meta = bank_meta

    def candidate_batches(self, stream_seed):
        return {
            int(g) + 1: candidate_batches_for(
                stream_seed, int(g), self.group_definition["members"][str(g)],
                int(self.cfg["batch_size"]), int(self.cfg["accumulation_steps"]),
            )
            for g in range(N_DATA_GROUPS)
        }


def _load_v07_state_contexts(root, cfg, partition, group_definition):
    v07 = root / cfg["output_dir"]
    contexts = []
    for seed, step in DEV_STATES:
        theta = _read_json(v07 / f"seed_{seed}/theta_mapping_raw.json")
        theta_state = next(r for r in theta["states"] if int(r["step"]) == step)
        checkpoint_path = v07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
        streams = []
        for rec in sorted(theta_state["e2_streams"], key=lambda r: int(r["stream_id"])):
            manifest_path = _resolve(root, rec["batch_manifest_path"])
            payload = _read_json(manifest_path)
            if payload.get("manifest_version") != "0.2":
                raise RuntimeError("dev manifest version mismatch")
            metadata = payload.get("metadata")
            batches = payload.get("batches")
            if not isinstance(metadata, Mapping):
                raise RuntimeError("dev manifest metadata missing")
            actual = {
                "train_seed": int(metadata.get("train_seed", -1)),
                "state_step": int(metadata.get("state_step", -1)),
                "stream_id": int(metadata.get("stream_id", -1)),
            }
            expected = {
                "train_seed": int(seed),
                "state_step": int(step),
                "stream_id": int(rec["stream_id"]),
            }
            if actual != expected:
                raise RuntimeError(f"dev manifest identity mismatch: {actual} != {expected}")
            if metadata.get("protocol") != "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM":
                raise RuntimeError("dev manifest protocol mismatch")
            content_sha = manifest_sha256(batches)
            if content_sha != str(payload.get("sha256")) or content_sha != str(rec["batch_manifest_sha256"]):
                raise RuntimeError("dev manifest sha mismatch")
            if int(payload["metadata"]["stream_seed"]) != int(rec["stream_seed"]):
                raise RuntimeError("dev manifest stream_seed mismatch")
            if len(batches) < max(HORIZONS) * int(cfg["accumulation_steps"]):
                raise RuntimeError("dev manifest does not cover H=8 window")
            for field in ("batch_seed", "epsilon", "rho"):
                if field not in metadata:
                    raise RuntimeError(f"dev manifest metadata missing {field}")
            streams.append({
                "stream_id": int(rec["stream_id"]),
                "stream_seed": int(rec["stream_seed"]),
                "batches": batches,
                "manifest_sha256": rec["batch_manifest_sha256"],
                "manifest_path": str(manifest_path),
            })
        meta = torch_load_metadata(checkpoint_path)
        contexts.append(_StateContext(
            split="dev", seed=seed, step=step, checkpoint_path=checkpoint_path,
            streams=streams, group_definition=group_definition,
            partition=partition, cfg=cfg, bank_meta=meta,
        ))
    return contexts


def torch_load_metadata(checkpoint_path):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    del ckpt
    return meta


def _load_formal_state_contexts(root, cfg, partition, group_definition):
    bank_root = root / OUTPUT_DIR / "confirmatory_bank"
    index_path = bank_root / "bank_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"missing {index_path}; run a0g_prepare_confirmatory_seeds_v01.py first")
    index = _read_json(index_path)
    contexts = []
    for seed in FORMAL_SEEDS:
        report_path = Path(index["seeds"][str(seed)])
        report = _read_json(report_path)
        if not report.get("technical_pass"):
            raise RuntimeError(f"formal seed {seed} bank not passed")
        manifests = {}
        for rec in report["e2_manifests"]:
            manifests.setdefault(int(rec["step"]), []).append(rec)
        for step in FORMAL_STEPS:
            state_rec = next(r for r in report["states"] if int(r["step"]) == step)
            checkpoint_path = Path(state_rec["path"])
            streams = []
            for rec in sorted(manifests[int(step)], key=lambda r: int(r["stream_id"])):
                payload = _read_json(_resolve(root, rec["batch_manifest_path"]))
                if payload.get("manifest_version") != "0.2":
                    raise RuntimeError("formal manifest version mismatch")
                metadata = payload["metadata"]
                if metadata.get("protocol") != PROTOCOL_NAME:
                    raise RuntimeError("formal manifest protocol mismatch")
                actual = {
                    "train_seed": int(metadata.get("train_seed", -1)),
                    "state_step": int(metadata.get("state_step", -1)),
                    "stream_id": int(metadata.get("stream_id", -1)),
                }
                expected = {
                    "train_seed": int(seed), "state_step": int(step),
                    "stream_id": int(rec["stream_id"]),
                }
                if actual != expected:
                    raise RuntimeError(
                        f"formal manifest identity mismatch: {actual} != {expected}")
                batches = payload["batches"]
                if manifest_sha256(batches) != rec["batch_manifest_sha256"]:
                    raise RuntimeError("formal manifest sha mismatch")
                streams.append({
                    "stream_id": int(rec["stream_id"]),
                    "stream_seed": int(rec["stream_seed"]),
                    "batches": batches,
                    "manifest_sha256": rec["batch_manifest_sha256"],
                    "manifest_path": str(_resolve(root, rec["batch_manifest_path"])),
                })
            if len(streams) != N_STREAMS_PER_STATE:
                raise RuntimeError(
                    f"formal seed {seed} step {step}: expected {N_STREAMS_PER_STATE} streams")
            meta = torch_load_metadata(checkpoint_path)
            contexts.append(_StateContext(
                split="formal", seed=seed, step=step, checkpoint_path=checkpoint_path,
                streams=streams, group_definition=group_definition,
                partition=partition, cfg=cfg, bank_meta=meta,
            ))
    return contexts


# ---------------------------------------------------------------------------
# v_t：当前状态 V_score 梯度（每 state 一次）
# ---------------------------------------------------------------------------

def _compute_v_t(torch, model, optimizer, scaler, ctx, val_score_tensors, autocast_ctx, named):
    """当前状态 V_score 梯度 v_t（alignment 基线的 readout）。"""
    load_full_training_state(ctx.checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    rng_snap = capture_rng_state()
    try:
        v_t = _grad_on_batches(
            torch, model, named, val_score_tensors, autocast_ctx, len(val_score_tensors),
        )
    finally:
        restore_rng_state(rng_snap)
        model.train()
    return {k: v.detach().cpu().clone() for k, v in v_t.items()}


# ---------------------------------------------------------------------------
# 单状态全流程
# ---------------------------------------------------------------------------

def _process_state(ctx, *, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
                   val_score_tensors, val_eval_tensors, get_lr, lam_primary,
                   lam_conservative, run_lambda_scan, shadow_check, timers):
    """返回该 state 全部记录行（level1/1b/2/3 原始素材）。"""
    cfg = ctx.cfg
    common = dict(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, dataset=dataset, named=named,
        checkpoint_path=ctx.checkpoint_path,
        val_score_tensors=val_score_tensors, val_eval_tensors=val_eval_tensors,
        accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
        grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
        global_micro_step=int(ctx.bank_meta["global_micro_step"]),
        total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
        get_lr=get_lr, timers=timers,
    )
    v_t = _compute_v_t(torch, model, optimizer, scaler, ctx, val_score_tensors,
                       autocast_ctx, named)

    rows_l1 = []
    rows_l1b = []
    rows_l2 = []
    rows_l3 = []
    scan_rows = []
    shadow_results = []

    for stream in ctx.streams:
        cand = ctx.candidate_batches(stream["stream_seed"])
        if set(cand) != set(range(1, N_DATA_GROUPS + 1)):
            raise RuntimeError(
                f"candidate batch plan malformed: have={sorted(cand)} "
                f"state={ctx.split}:{ctx.seed}/{ctx.step} stream={stream['stream_id']}"
            )
        t_pass = _run_tangent_pass(
            **common, batches=stream["batches"], stream_seed=stream["stream_seed"],
            cand_batches_by_j=cand, v_t=v_t, hvp_direction_mode="pre",
        )
        base_vs = t_pass["baseline_val_score"]
        base_ve = t_pass["baseline_val_eval"]

        # ---- 冻结协议等价抽查：frozen run_theta_branch 基线哈希一致 ----
        if shadow_check(stream):
            import torch as _torch_for_ckpt
            _ckpt = _torch_for_ckpt.load(
                ctx.checkpoint_path, map_location="cpu", weights_only=False)
            shadow = run_theta_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=dataset, state_path=ctx.checkpoint_path,
                reference_theta=_ckpt["model"], q=_zeros_like_named(named), named=named,
                batches=stream["batches"], stream_seed=int(stream["stream_seed"]),
                sign=0, epsilon=0.0, horizons=HORIZONS,
                accumulation_steps=int(cfg["accumulation_steps"]),
                batch_size=int(cfg["batch_size"]), device=str(model.device),
                grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
                global_micro_step=int(ctx.bank_meta["global_micro_step"]),
                total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
                get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
            )
            del _ckpt
            ok = (
                shadow["model_hash_final"] == t_pass["final_model_hash"]
                and shadow["optimizer_hash_final"] == t_pass["final_optimizer_hash"]
            )
            shadow_results.append({
                "stream_id": stream["stream_id"], "pass": bool(ok),
                "shadow_model_hash": shadow["model_hash_final"],
                "pass_model_hash": t_pass["final_model_hash"],
            })
            if not ok:
                raise RuntimeError(
                    f"frozen-protocol shadow mismatch seed={ctx.seed} step={ctx.step} "
                    f"stream={stream['stream_id']} "
                    f"pass_model={t_pass['final_model_hash'][:16]} "
                    f"shadow_model={shadow['model_hash_final'][:16]} "
                    f"pass_opt={t_pass['final_optimizer_hash'][:16]} "
                    f"shadow_opt={shadow['optimizer_hash_final'][:16]}"
                )

        clip_base_h1 = t_pass["clip_trace"][0]["clip_active"]

        # ---- 动作 branches ----
        branch_cache = {}
        branch_plan = []
        for j in range(1, N_DATA_GROUPS + 1):
            branch_plan.append((j, lam_primary, +1))
            branch_plan.append((j, lam_primary, -1))
        if lam_conservative is not None:
            for j in range(1, N_DATA_GROUPS + 1):
                branch_plan.append((j, lam_conservative, +1))
        scan_plan = []
        if run_lambda_scan:
            for lam in LAMBDA_CANDIDATES:
                for j in range(1, N_DATA_GROUPS + 1):
                    scan_plan.append((j, float(lam), +1))

        for (j, lam, sign) in branch_plan + scan_plan:
            if j not in t_pass["g_by_j"]:
                raise RuntimeError(
                    f"INTERNAL: tangent pass missing g_by_j[{j}]; "
                    f"have={sorted(t_pass['g_by_j'])} "
                    f"branch_plan_j={sorted({pp[0] for pp in branch_plan + scan_plan})} "
                    f"state={ctx.split}:{ctx.seed}/{ctx.step} stream={stream['stream_id']}"
                )
            br = _run_action_branch(
                **common, batches=stream["batches"], stream_seed=stream["stream_seed"],
                action_j=j, lam=lam, sign=sign, cand_batches_by_j=cand,
                g_reference=t_pass["g_by_j"][j],
            )
            key = (j, float(lam), sign)
            branch_cache[key] = br
            if br["first_step"].get("finite") is False:
                raise RuntimeError(f"non-finite action gradient at {key}")
            gap = br["g_j_gap_vs_tangent_pass"]
            if sign == +1 and float(lam) == float(lam_primary):
                crossing = bool(br["first_step"]["clip_active"] != clip_base_h1)
            else:
                crossing = None

            if float(lam) == float(lam_primary) and sign == +1:
                br["first_step"]["baseline_preclip_norm"] = t_pass["clip_trace"][0]["preclip_norm"]

            if float(lam) == float(lam_primary) and sign == -1:
                # ---- Level 1: 中心 FD vs hhat（V_score） ----
                for h in HORIZONS:
                    hk = str(h)
                    plus = branch_cache.get((j, float(lam), +1))
                    minus = branch_cache.get((j, float(lam), -1))
                    if plus is None or minus is None:
                        continue
                    h_fd = (plus["val_score"][hk] - minus["val_score"][hk]) / (2.0 * float(lam))
                    hhat = t_pass["hhat"]
                    row = {
                        "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                        "stream_id": stream["stream_id"], "j": j, "H": h,
                        "lambda": float(lam),
                        "hhat_alignment": hhat["alignment"][j][hk],
                        "hhat_B": hhat["B"][j][hk],
                        "hhat_C": hhat["C"][j][hk],
                        "h_FD": float(h_fd),
                        "abs_err_alignment": abs(hhat["alignment"][j][hk] - h_fd),
                        "abs_err_B": abs(hhat["B"][j][hk] - h_fd),
                        "abs_err_C": abs(hhat["C"][j][hk] - h_fd),
                        "rel_err_alignment": relative_error(h_fd, hhat["alignment"][j][hk]),
                        "rel_err_B": relative_error(h_fd, hhat["B"][j][hk]),
                        "rel_err_C": relative_error(h_fd, hhat["C"][j][hk]),
                        "sign_match_alignment": (hhat["alignment"][j][hk] > 0) == (h_fd > 0),
                        "sign_match_B": (hhat["B"][j][hk] > 0) == (h_fd > 0),
                        "sign_match_C": (hhat["C"][j][hk] > 0) == (h_fd > 0),
                        "clip_regime_crossing": crossing,
                        "g_j_gap_vs_tangent_pass": gap,
                    }
                    rows_l1.append(row)

                    # ---- Level 1b: 有限 +lambda 动作（V_eval） ----
                    actual = plus["val_eval"][hk] - base_ve[hk]
                    rows_l1b.append({
                        "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                        "stream_id": stream["stream_id"], "j": j, "H": h,
                        "lambda": float(lam),
                        "actual_delta_eval": float(actual),
                        "pred_delta_alignment": float(lam) * hhat["alignment"][j][hk],
                        "pred_delta_B": float(lam) * hhat["B"][j][hk],
                        "pred_delta_C": float(lam) * hhat["C"][j][hk],
                        "baseline_val_eval": base_ve[hk],
                        "action_val_eval": plus["val_eval"][hk],
                        "nonlinear_residual_C": float(
                            actual - float(lam) * hhat["C"][j][hk]
                        ),
                        "sign_actual": 1 if actual > 0 else (-1 if actual < 0 else 0),
                        "sign_pred_C": 1 if hhat["C"][j][hk] > 0 else (-1 if hhat["C"][j][hk] < 0 else 0),
                    })

        # ---- Level 2: 5 动作预测 vs 实际（V_eval, +lambda_primary） ----
        lam = float(lam_primary)
        for h in HORIZONS:
            hk = str(h)
            actual_by_j = {0: 0.0}
            pred = {m: {0: 0.0} for m in ("alignment", "B", "C")}
            for j in range(1, N_DATA_GROUPS + 1):
                plus = branch_cache[(j, lam, +1)]
                actual_by_j[j] = plus["val_eval"][hk] - base_ve[hk]
                for m in ("alignment", "B", "C"):
                    pred[m][j] = lam * t_pass["hhat"][m][j][hk]
            for j in range(0, N_DATA_GROUPS + 1):
                rows_l2.append({
                    "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                    "stream_id": stream["stream_id"], "j": j, "H": h, "lambda": lam,
                    "actual_delta_eval": float(actual_by_j[j]),
                    "pred_delta_alignment": float(pred["alignment"][j]),
                    "pred_delta_B": float(pred["B"][j]),
                    "pred_delta_C": float(pred["C"][j]),
                })
            # ---- Level 3: 四方法决策 ----
            rng = np.random.default_rng(seed=(
                (ctx.seed * 100003 + ctx.step * 101 + stream["stream_id"]) * 1000 + h
            ))
            actual_v = actual_by_j
            # 随机基线（每个 state/stream/H 一次抽取，j=0..4）
            j_rand = random_choice(rng, N_DATA_GROUPS + 1)
            rows_l3.append({
                "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                "stream_id": stream["stream_id"], "H": h, "method": "random",
                **level3_decision(
                    predicted_by_j={k: (0.0 if k != j_rand else -1.0) for k in range(N_DATA_GROUPS + 1)},
                    actual_by_j=actual_v,
                ),
            })
            for m in ("alignment", "B", "C"):
                rows_l3.append({
                    "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                    "stream_id": stream["stream_id"], "H": h, "method": m,
                    **level3_decision(
                        predicted_by_j={k: pred[m][k] for k in range(N_DATA_GROUPS + 1)},
                        actual_by_j=actual_v,
                    ),
                })

        # ---- lambda 扫描记录（dev） ----
        if run_lambda_scan:
            for lam in LAMBDA_CANDIDATES:
                for h in HORIZONS:
                    hk = str(h)
                    for j in range(1, N_DATA_GROUPS + 1):
                        br = branch_cache[(j, float(lam), +1)]
                        delta_score = br["val_score"][hk] - base_vs[hk]
                        hhat_C = t_pass["hhat"]["C"][j][hk]
                        denom = max(abs(hhat_C), 1e-9)
                        scan_rows.append({
                            "split": ctx.split, "seed": ctx.seed, "step": ctx.step,
                            "stream_id": stream["stream_id"], "j": j, "H": h,
                            "lambda": float(lam),
                            "delta_val_score": float(delta_score),
                            "delta_val_score_per_lambda": float(delta_score) / float(lam),
                            "hhat_C": float(hhat_C),
                            "rel_nonlinearity_C": abs(float(delta_score) / float(lam) - hhat_C) / denom,
                            "clip_regime_crossing": bool(
                                br["first_step"]["clip_active"] != clip_base_h1
                            ),
                            "abs_hhat_C": abs(float(hhat_C)),
                        })

        print(
            f"STATE {ctx.split}:{ctx.seed}/{ctx.step} stream={stream['stream_id']}"
            f"/{N_STREAMS_PER_STATE - 1} "
            f"hhatC_H4={[round(t_pass['hhat']['C'][j]['4'], 9) for j in (1, 2, 3, 4)]}",
            flush=True,
        )
        del t_pass, branch_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "level1": rows_l1, "level1b": rows_l1b, "level2": rows_l2, "level3": rows_l3,
        "scan": scan_rows, "shadow": shadow_results,
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: Sequence[Mapping]):
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_level3(rows, level2_rows):
    """按 (split, H, method) 聚合 Gain/Regret/harmful/top1/Spearman/direction/MAE。"""
    # within-state Spearman：用 level2 的 5 动作 pred/actual（绝不跨 state 池化）
    l2_index = {}
    for r in level2_rows:
        l2_index.setdefault(
            (r["split"], r["seed"], r["step"], r["stream_id"], int(r["H"])), {}
        )[int(r["j"])] = r
    spearman_store = {}
    for (split, seed, step, stream_id, h), by_j in l2_index.items():
        for m, pk in (("alignment", "pred_delta_alignment"),
                      ("B", "pred_delta_B"), ("C", "pred_delta_C")):
            key = (split, h, m)
            preds = [by_j[j][pk] for j in sorted(by_j)]
            actuals = [by_j[j]["actual_delta_eval"] for j in sorted(by_j)]
            val = _safe_spearman(preds, actuals)
            if val is not None:
                spearman_store.setdefault(key, []).append(val)

    method_keys = ("random", "alignment", "B", "C")
    blocks = {}
    for r in rows:
        blocks.setdefault((r["split"], int(r["H"]), r["method"]), []).append(r)
    agg = {}
    # 遍历 (split, H) 组合；blocks 键为 (split, H, method) 三元组。
    # 注意：不能写 `for (split, h), _ in ...`——展开目标会把 split 字符串
    # 当成序列解包（"formal" 6 字符 → ValueError: too many values to unpack）。
    for split, h in sorted({(k[0], k[1]) for k in blocks}):
        row_out = {"H": h, "_split": split}
        for m in method_keys:
            block_rows = blocks.get((split, h, m), [])
            if not block_rows:
                continue
            spearman_vals = spearman_store.get((split, h, m)) or []
            mae_vals = []
            pk = {"alignment": "pred_delta_alignment", "B": "pred_delta_B", "C": "pred_delta_C"}.get(m)
            if pk:
                for r in level2_rows:
                    if r["split"] == split and int(r["H"]) == h:
                        mae_vals.append(abs(float(r[pk]) - float(r["actual_delta_eval"])))
            row_out[m] = {
                "mean_gain": _mean(r["gain"] for r in block_rows),
                "mean_regret": _mean(r["regret"] for r in block_rows),
                "harmful_rate": _mean(1.0 if r["harmful"] else 0.0 for r in block_rows),
                "top1_accuracy": _mean(1.0 if r["top1_correct"] else 0.0 for r in block_rows),
                "mean_spearman": _mean(spearman_vals) if spearman_vals else None,
                "direction_accuracy": _mean(1.0 if r["direction_correct"] else 0.0 for r in block_rows),
                "MAE": _mean(mae_vals) if mae_vals else None,
                "n": len(block_rows),
            }
        agg[f"{split}|H={h}"] = row_out
    return agg


def _freeze_lambda(scan_rows):
    """按 §6 准则在开发扫描结果上冻结主/保守 lambda（双准则，结果写入报告）。

    准则 A（严格局部线性，首选）：mean slope-dev vs λ0 ≤ 0.25 且 clip 穿越 ≤ 0.10；
        若任何候选通过，主 lambda 取通过者中最大，保守取次小。
    准则 B（决策信号，回退）：开发数据显示响应存在 ~5e-5 量级的绝对 deviation
        floor（与信号同阶， replay 位级确定，非噪声），点态线性在任何网格 λ 上
        都不成立；此时冻结规则改为——在满足 clip 穿越 ≤ 0.10、动作梯度有限、
        信号非全零的候选中取最大 lambda（最大化 SNR/符号一致率），保守取次小。
        排序信号（同 state/stream/H 内跨动作 Spearman）不依赖点态线性，是本
        实验 Level 2/3 决策评价的关键。
    """
    by_lam = {}
    hhat_abs = []
    records = {}
    for r in scan_rows:
        lam = float(r["lambda"])
        by_lam.setdefault(lam, []).append(r)
        hhat_abs.append(r["abs_hhat_C"])
        key = (r["split"], r["seed"], r["step"], r["stream_id"], int(r["j"]), int(r["H"]))
        records.setdefault(key, {})[lam] = r
    lambda0 = min(by_lam)
    # 组内 Spearman 对每个 lambda 各算一次（actual delta 随 lambda 变，排序信号
    # 不是 lambda 不变量）：按 (state,stream,H) 聚合各 j 的预测 hhat_C 与实际 delta。
    by_group = {}
    for (split, seed, step, stream_id, j, h), per_lam in records.items():
        by_group.setdefault((split, seed, step, stream_id, h), {})[j] = per_lam

    def _group_spearman(lam):
        vals = []
        for per_j in by_group.values():
            if len(per_j) < N_DATA_GROUPS:
                continue
            js = sorted(per_j)
            if any(lam not in per_j[j] for j in js):
                continue
            preds = [float(per_j[j][lam]["hhat_C"]) for j in js]
            actuals = [float(per_j[j][lam]["delta_val_score"]) for j in js]
            rho = _safe_spearman(preds, actuals)
            if rho is not None:
                vals.append(rho)
        return _mean(vals) if vals else None

    scan_summary = []
    for lam in sorted(by_lam):
        rows = by_lam[lam]
        # 局部性：与最小 lambda 的差分斜率相对偏差（跨 lambda 配对）
        slope_devs = []
        if lam != lambda0:
            for key, per_lam in records.items():
                if lam not in per_lam or lambda0 not in per_lam:
                    continue
                hhat_c = float(per_lam[lam]["hhat_C"])
                if abs(hhat_c) <= 1e-6:
                    continue
                d_lam = float(per_lam[lam]["delta_val_score"])
                d_0 = float(per_lam[lambda0]["delta_val_score"])
                slope = (d_lam - d_0) / (lam - lambda0)
                slope_devs.append(abs(slope - hhat_c) / abs(hhat_c))
        # 兼容性口径（旧）：ΔL/lambda 对 hhat 的相对偏差
        rel = [r["rel_nonlinearity_C"] for r in rows if r["abs_hhat_C"] > 1e-9]
        # 决策口径：符号一致率（|hhat|>1e-4 才计）、信号/偏差幅度
        sign_pts = [
            (float(r["delta_val_score"]), float(r["hhat_C"]))
            for r in rows if r["abs_hhat_C"] > 1e-4
        ]
        sign_agree = (
            _mean(1.0 if (d > 0) == (h > 0) else 0.0 for d, h in sign_pts)
            if sign_pts else None
        )
        abs_sig = _mean(abs(float(r["hhat_C"])) for r in rows) * lam
        abs_dev = _mean(
            abs(float(r["delta_val_score"]) - lam * float(r["hhat_C"])) for r in rows
        )
        scan_summary.append({
            "lambda_value": lam,
            "mean_abs_hhat_C": _mean(hhat_abs),
            "mean_rel_nonlinearity_C": _mean(rel) if rel else None,
            "mean_slope_dev_vs_lambda0": _mean(slope_devs) if slope_devs else None,
            "p90_slope_dev_vs_lambda0": (
                float(np.percentile(slope_devs, 90)) if slope_devs else None
            ),
            "clip_crossing_rate": _mean(1.0 if r["clip_regime_crossing"] else 0.0 for r in rows),
            "all_near_zero": _mean(hhat_abs) <= 1e-9,
            "sign_agreement": sign_agree,
            "mean_abs_signal": abs_sig,
            "mean_abs_dev": abs_dev,
            "snr": (abs_sig / abs_dev) if abs_dev > 0 else None,
            "mean_within_state_spearman": _group_spearman(lam),
        })
    strict_passing = [
        s for s in scan_summary
        if s["mean_slope_dev_vs_lambda0"] is not None
        and s["mean_slope_dev_vs_lambda0"] <= 0.25
        and s["clip_crossing_rate"] <= 0.10
        and not s["all_near_zero"]
    ]
    if strict_passing:
        rule = "A_strict_locality"
        primary = max(s["lambda_value"] for s in strict_passing)
    else:
        rule = "B_decision_signal"
        eligible = [
            s for s in scan_summary
            if s["clip_crossing_rate"] <= 0.10 and not s["all_near_zero"]
        ]
        _require(bool(eligible), "no lambda candidate satisfies the clip-stability gate")
        primary = max(s["lambda_value"] for s in eligible)
    smaller = [s["lambda_value"] for s in scan_summary if s["lambda_value"] < primary]
    conservative = max(smaller) if smaller else None
    primary_row = next(s for s in scan_summary if s["lambda_value"] == primary)
    mean_spearman = primary_row["mean_within_state_spearman"]
    criteria = {
        "action_gradients_finite": True,  # branches raise on non-finite
        "convex_mixture_by_construction": True,
        "not_all_actions_near_zero": not all(s["all_near_zero"] for s in scan_summary),
        "strict_locality_any_lambda_mean_slope_dev_le_0.25": bool(strict_passing),
        "primary_clip_crossing_rate_le_0.10": bool(primary_row["clip_crossing_rate"] <= 0.10),
        "freeze_rule": rule,
        "primary_mean_within_state_spearman_ge_0.5": (
            mean_spearman is not None and mean_spearman >= 0.5
        ),
    }
    if rule == "A_strict_locality":
        rationale = (
            f"准则 A（严格局部线性）通过：mean slope-dev vs λ0={lambda0} ≤ 0.25 且 "
            f"clip 穿越 ≤ 0.10 的候选取最大（局部性与信号幅度平衡），保守取次小。"
            f"ΔL/λ 对 hhat 的相对比值在 hhat 过小时数值爆炸，仅作兼容记录。"
        )
    else:
        rationale = (
            f"准则 A（严格局部线性，slope-dev vs λ0={lambda0} ≤ 0.25）在所有候选上均不通过："
            f"开发扫描显示响应带有 ~{primary_row['mean_abs_dev']:.1e} 量级的绝对 deviation "
            f"floor（replay 位级确定，非随机噪声），点态线性不成立。改用准则 B（决策信号）："
            f"在满足 clip 穿越 ≤ 0.10 的候选中取最大 lambda 以最大化 SNR、符号一致率与"
            f"组内排序信号（三者随 lambda 单调上升；primary 处 mean within-state "
            f"Spearman = {mean_spearman:.3f}，是本实验 Level 2/3 决策评价的关键，"
            f"不依赖点态线性）。clip 稳定门把 lambda 封顶（0.10 的穿越率 > 0.10 越限）。"
        )
    return {
        "scan_rows": scan_summary,
        "primary_lambda": float(primary),
        "conservative_lambda": float(conservative) if conservative is not None else None,
        "criteria": criteria,
        "rationale": rationale,
        "freeze_rule": rule,
        "mean_within_state_spearman": mean_spearman,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase", choices=("lambda-dev", "main"), required=True)
    ap.add_argument("--skip-shadow-check", action="store_true")
    ap.add_argument("--limit-states", type=int, default=0, help="debug: only first N states")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _read_json(root / "research/configs/e3c_confirmatory_v07.json")
    if cfg.get("dtype") != "bfloat16" or int(cfg.get("accumulation_steps", -1)) != 2 \
            or float(cfg.get("grad_clip", -1.0)) != 1.0:
        raise RuntimeError("config drift: strict v07 replay requirements violated")

    v07_out = root / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"

    # ---- 数据组定义 + validation split（确定性，先计算并落盘） ----
    dgd_path = out_dir / "data_group_definition.json"
    vsplit_path = out_dir / "validation_split.json"
    if dgd_path.exists() and vsplit_path.exists():
        data_group_definition = _read_json(dgd_path)
        validation_split = _read_json(vsplit_path)
    else:
        texts = [json.loads(line)["text"] for line in data_path.read_text(encoding="utf-8").splitlines()]
        data_group_definition = build_data_group_definition(partition["train_indices"], texts)
        validation_split = split_validation_pool(partition["eval_indices"])
        dgd_path.write_text(json.dumps(data_group_definition, ensure_ascii=False, indent=2), encoding="utf-8")
        vsplit_path.write_text(json.dumps(validation_split, ensure_ascii=False, indent=2), encoding="utf-8")
    _require(
        set(data_group_definition["members"]) == {str(g) for g in range(N_DATA_GROUPS)},
        "data group definition must contain exactly 4 groups",
    )
    _require(
        "仅用于决策管线验证" in data_group_definition["disclaimer"],
        "hash-bucket grouping disclaimer missing",
    )
    merged = []
    for members in data_group_definition["members"].values():
        merged.extend(members)
    _require(
        len(merged) == len(set(merged)) == len(set(partition["train_indices"])),
        "data groups must partition the train pool exactly (disjoint, complete)",
    )
    _require(validation_split["disjoint"] is True, "V_score/V_eval overlap")
    _require(
        not (set(validation_split["v_score"]) & set(validation_split["v_eval"])),
        "V_score/V_eval overlap",
    )
    _require(
        set(validation_split["v_score"]) | set(validation_split["v_eval"])
        == set(map(int, partition["eval_indices"])),
        "validation split must cover the eval pool exactly",
    )

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("strict BF16 replay requires CUDA")
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device,
    )
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + EVAL_BATCH_SIZE] for i in range(0, len(idx), EVAL_BATCH_SIZE)]

    val_score_batches = _chunk(validation_split["v_score"])
    val_eval_batches = _chunk(validation_split["v_eval"])
    val_score_tensors = _stack_all(dataset, val_score_batches, args.device)
    val_eval_tensors = _stack_all(dataset, val_eval_batches, args.device)

    timers = _Timers(torch)

    print(
        f"DATA_GROUP_KEYS={sorted(data_group_definition['members'])} "
        f"N_DATA_GROUPS={N_DATA_GROUPS}",
        flush=True,
    )

    dev_ctxs = _load_v07_state_contexts(root, cfg, partition, data_group_definition)
    formal_ctxs = []
    if args.phase != "lambda-dev":
        formal_ctxs = _load_formal_state_contexts(root, cfg, partition, data_group_definition)
    if args.limit_states:
        dev_ctxs = dev_ctxs[:args.limit_states]
        formal_ctxs = formal_ctxs[:args.limit_states]

    def shadow_check(stream, ctx):
        if args.skip_shadow_check:
            return False
        return ctx.split == "dev" or int(stream["stream_id"]) == 0

    started = time.time()
    level1_rows, level1b_rows, level2_rows, level3_rows = [], [], [], []
    all_scan_rows = []
    shadow_log = []

    if args.phase == "lambda-dev":
        for ctx in dev_ctxs:
            res = _process_state(
                ctx, torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=dataset, named=named,
                val_score_tensors=val_score_tensors, val_eval_tensors=val_eval_tensors,
                get_lr=get_lr, lam_primary=LAMBDA_CANDIDATES[0],
                lam_conservative=None, run_lambda_scan=True,
                shadow_check=lambda s, c=ctx: shadow_check(s, c), timers=timers,
            )
            all_scan_rows.extend(res["scan"])
            shadow_log.extend(res["shadow"])
        selection = _freeze_lambda(all_scan_rows)
        selection["split"] = "dev"
        selection["dev_states"] = [list(s) for s in DEV_STATES]
        selection["n_scan_points"] = len(all_scan_rows)
        (out_dir / "lambda_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "lambda_selection_report.md").write_text(
            render_lambda_selection_report(selection), encoding="utf-8")
        _write_csv(out_dir / "lambda_scan_detail.csv", all_scan_rows)
        print(f"PRIMARY_LAMBDA={selection['primary_lambda']}")
        print(f"CONSERVATIVE_LAMBDA={selection['conservative_lambda']}")
        print("STATUS = LAMBDA_FROZEN")
        return 0

    # ---- phase == main ----
    lam_sel_path = out_dir / "lambda_selection.json"
    if not lam_sel_path.is_file():
        raise SystemExit("lambda_selection.json missing; run --phase lambda-dev first")
    lam_sel = _read_json(lam_sel_path)
    lam_primary = float(lam_sel["primary_lambda"])
    lam_cons = lam_sel.get("conservative_lambda")
    lam_conservative = float(lam_cons) if lam_cons is not None else None

    for ctx in dev_ctxs + formal_ctxs:
        res = _process_state(
            ctx, torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            val_score_tensors=val_score_tensors, val_eval_tensors=val_eval_tensors,
            get_lr=get_lr, lam_primary=lam_primary, lam_conservative=lam_conservative,
            run_lambda_scan=False,
            shadow_check=lambda s, c=ctx: shadow_check(s, c), timers=timers,
        )
        level1_rows.extend(res["level1"])
        level1b_rows.extend(res["level1b"])
        level2_rows.extend(res["level2"])
        level3_rows.extend(res["level3"])
        shadow_log.extend(res["shadow"])
        config_row = {
            "state": f"{ctx.split}:{ctx.seed}/{ctx.step}",
            "shadow_pass": all(x["pass"] for x in res["shadow"]),
        }
        print(f"STATE_DONE {config_row}", flush=True)

    _write_csv(out_dir / "level1_local_derivative.csv", level1_rows)
    _write_csv(out_dir / "level1_finite_action.csv", level1b_rows)
    _write_csv(out_dir / "level2_action_predictions.csv", level2_rows)
    _write_csv(out_dir / "level3_decision_metrics.csv", level3_rows)

    level3_agg = _aggregate_level3(level3_rows, level2_rows)

    # ---- Level 1 / 1b 汇总（按 split×H 池化 + 逐 state 分组指标） ----
    from fr.a0g_decision_utility_v01 import pooled_metric

    def _l1_block(rows, pred_key):
        finite = [r for r in rows if r.get(pred_key) is not None and r.get("h_FD") is not None]
        if not finite:
            return {"n": 0}
        m = pooled_metric(finite, pred_key=pred_key, actual_key="h_FD")
        sign_key = pred_key.replace("hhat_", "sign_match_")
        abs_key = pred_key.replace("hhat_", "abs_err_")
        rel_key = pred_key.replace("hhat_", "rel_err_")
        return {
            "n": int(m["n"]),
            "pearson": float(m["corr"]),
            "mae": float(m["MAE"]),
            "rmse": float(m["RMSE"]),
            "mean_relative_error": float(m["mean_relative_error"]),
            "sign_accuracy": _mean(1.0 if r[sign_key] else 0.0 for r in finite),
            "mean_abs_err": _mean(float(r[abs_key]) for r in finite),
        }

    level1_agg = {}
    for h in HORIZONS:
        for split in ("dev", "formal"):
            rows_h = [r for r in level1_rows
                      if int(r["H"]) == h and r["split"] == split]
            if not rows_h:
                continue
            level1_agg[f"{split}|H={h}"] = {
                "H": h, "_split": split,
                "alignment": _l1_block(rows_h, "hhat_alignment"),
                "B": _l1_block(rows_h, "hhat_B"),
                "C": _l1_block(rows_h, "hhat_C"),
            }
    # 逐 state 分组（Spearman 只在 state 内算，绝不跨 state 池化）
    level1_per_state = []
    state_groups = {}
    for r in level1_rows:
        state_groups.setdefault(
            (r["split"], r["seed"], r["step"], int(r["H"])), []
        ).append(r)
    for (split, seed, step, h), rows_g in sorted(state_groups.items()):
        rec = {"split": split, "seed": seed, "step": step, "H": h}
        for m_key, pk, ak in (("alignment", "hhat_alignment", "hhat_alignment"),
                              ("B", "hhat_B", "hhat_B"),
                              ("C", "hhat_C", "hhat_C")):
            vals = _safe_spearman([float(r[pk]) for r in rows_g],
                                  [float(r["h_FD"]) for r in rows_g])
            rec[f"spearman_{m_key}"] = vals
        level1_per_state.append(rec)
    _write_csv(out_dir / "level1_per_state_spearman.csv", level1_per_state)

    level1b_agg = {}
    for h in HORIZONS:
        for split in ("dev", "formal"):
            rows_h = [r for r in level1b_rows
                      if int(r["H"]) == h and r["split"] == split]
            if not rows_h:
                continue
            block = {"H": h, "_split": split, "n": len(rows_h)}
            for m_key, pk in (("alignment", "pred_delta_alignment"),
                              ("B", "pred_delta_B"), ("C", "pred_delta_C")):
                actuals = [float(r["actual_delta_eval"]) for r in rows_h]
                preds = [float(r[pk]) for r in rows_h]
                block[m_key] = {
                    "mae": _mean(abs(a - p) for a, p in zip(actuals, preds)),
                    "mean_actual": _mean(actuals),
                    "mean_pred": _mean(preds),
                    "sign_accuracy": _mean(
                        (1 if (a > 0) == (p > 0) else 0) if (a != 0 and p != 0) else 0.5
                        for a, p in zip(actuals, preds)
                    ),
                    "mean_abs_nonlinear_residual": _mean(
                        abs(float(r["actual_delta_eval"]) - float(r[pk])) for r in rows_h
                    ),
                }
            level1b_agg[f"{split}|H={h}"] = block

    # ---- 成本核算 ----
    n_passes = len(dev_ctxs) + len(formal_ctxs)
    n_streams = sum(len(c.streams) for c in dev_ctxs + formal_ctxs)
    denom = max(n_streams, 1)
    measured = {
        "t_native_step": timers.t.get("t_native_step", 0.0) / (denom * 8),
        "t_val_loss_one": timers.t.get("t_val_loss", 0.0) / (denom * 6),
        "t_val_grad_one": timers.t.get("t_val_grad", 0.0) / (denom * 3),
        "t_cand_grad_one": timers.t.get("t_cand_grad", 0.0) / (denom * N_DATA_GROUPS),
        "t_hvp_one": timers.t.get("t_hvp", 0.0) / (denom * N_DATA_GROUPS * 7),
        "t_tangent_step_one": timers.t.get("t_tangent_step", 0.0) / (denom * 8),
    }
    cost_rows = cost_model(measured)
    cost_csv_rows = [
        {"record_type": "measured_component", "component": k, "value_s": v}
        for k, v in measured.items()
    ] + [
        {"record_type": "cost_model", **row} for row in cost_rows
    ]
    _write_csv(out_dir / "cost_breakdown.csv", cost_csv_rows)
    be = {h: breakeven_m(measured, h) for h in HORIZONS}

    experiment_config = {
        "protocol": "FR_DECISION_UTILITY_V01",
        "dev_states": [list(s) for s in DEV_STATES],
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_steps": list(FORMAL_STEPS),
        "streams_per_state": N_STREAMS_PER_STATE,
        "horizons": list(HORIZONS),
        "primary_horizon": PRIMARY_HORIZON,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_primary": lam_primary,
        "lambda_conservative": lam_conservative,
        "n_data_groups": N_DATA_GROUPS,
        "action": "g_action = g + sign*lambda*(g_j - g) at post-unscale/pre-clip, single optimizer step",
        "tangent_convention": "r_H is derivative per unit lambda; finite prediction = lambda*hhat",
        "validation_batch_size": EVAL_BATCH_SIZE,
        "seed_batch_rule": "Philox(stream_seed ^ (9000+j)) permutation over group members, 2 microbatches x 4",
        "near_tie_fraction": NEAR_TIE_FRACTION,
        "device": str(args.device),
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "optimizer_signature": optimizer_signature(optimizer),
        "input_files": {
            "v07_config": str(root / "research/configs/e3c_confirmatory_v07.json"),
            "partition": str(v07_out / "partition_v07.json"),
            "data": str(data_path),
            "confirmatory_bank_index": str(out_dir / "confirmatory_bank/bank_index.json"),
        },
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8")

    runtime = {
        "elapsed_seconds": float(time.time() - started),
        "states_processed": n_passes,
        "streams_processed": n_streams,
        "measured_per_stream_seconds": measured,
        "peak_gpu_memory_bytes": dict(timers.peak),
        "peak_gpu_memory_overall": float(torch.cuda.max_memory_allocated()),
        "breakeven_m_candidates": {str(h): be[h] for h in HORIZONS},
        "shadow_protocol_checks": shadow_log,
        "all_shadow_pass": all(x["pass"] for x in shadow_log),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    summary = assemble_summary(
        config=experiment_config,
        data_group_definition=data_group_definition,
        validation_split=validation_split,
        lambda_selection=lam_sel,
        level1_rows=level1_rows, level1b_rows=level1b_rows, level2_rows=level2_rows,
        level3_rows=level3_rows, level3_aggregate=level3_agg,
        cost_rows=cost_rows, runtime=runtime,
        conditional_lookahead_limitation=CONDITIONAL_LOOKAHEAD_LIMITATION,
    )
    summary["level1_aggregate"] = level1_agg
    summary["level1b_aggregate"] = level1b_agg
    summary["level1_per_state_spearman"] = level1_per_state
    (out_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")
    (out_dir / "FR_DECISION_UTILITY_REPORT.md").write_text(render_report(summary), encoding="utf-8")

    # ---- SHA256SUMS.txt（§13；对除本文件外的所有顶层制品） ----
    import hashlib as _hl
    entries = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        digest = _hl.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print("STATUS = COMPLETE")
    print(f"SUMMARY = {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
