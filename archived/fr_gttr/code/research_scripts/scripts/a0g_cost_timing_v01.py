#!/usr/bin/env python3
"""FR_A0G cost-only benchmark（§12）——3 个代表 formal state × 8 streams 独占 GPU 计时。

科学结果零影响：本模块只做计时，不写任何科学 CSV；切线 pass 的数值输出仅用于
与已冻结的 level1_local_derivative.csv 做一致性抽查（验证计时副本未漂移）。

计时方法（相对旧设计的修正）：
  1. 每个计算组成使用独立计时键（native_fwd_bwd / native_unscale / native_clip /
     cand_grad / hvp / tangent_B / tangent_C / native_step / val_loss / val_grad /
     hhat_projection / v_t_compute / state_load / replay_plus / replay_minus）；
  2. 每键记录 total / count / mean / std / median / IQR，分母一律用实际 tick 数；
  3. 每次计时区间前后 torch.cuda.synchronize() 再读 time.perf_counter()；
  4. 分支引擎（真实回放）与切线 pass 使用分离的 timers，杜绝共用键污染；
  5. 每 state 正式计时前先 warm-up（1 条 stream，样本单独标记丢弃）。

独占 GPU 门：正式计时前轮询 nvidia-smi，要求目标 GPU util 连续低且不存在
其他计算进程；不杀任何外部任务，无限等待直至满足（--max-wait-s 上限）。

分别计时的组成（对应任务书 §四）：
  1 基准未来轨迹回放      -> native_fwd_bwd + native_unscale + native_clip + native_step（逐 horizon 成对）
  2 候选数据梯度 g_j      -> cand_grad（逐 j）
  3 validation readout    -> val_loss / val_grad（逐 H）+ v_t_compute（每 state 一次）
  4 仅优化器切线传播 B    -> tangent_B（逐 (h,j)）
  5 Hessian-向量积        -> hvp（逐 (h,j)）
  6 完整切线传播 C        -> tangent_C（逐 (h,j)）
  7 单候选 +lambda 回放   -> replay_plus（wall − state_load = net）
  8 单候选 ±lambda 中心差分 -> replay_minus + replay_plus（两次独立回放）

成本公式（h=horizon, M=候选数；t_* 为实测中位数）：
  direct(h,M)      = h·t_native + t_vloss + M·(h·t_native + t_vloss)
  direct_fd(h,M)   = h·t_native + t_vloss + M·2·(h·t_native + t_vloss)   # ±λ 两次回放
  shared_B(h,M)    = h·t_native + t_vloss + t_vgrad + M·(t_cand + h·t_tstep_B)
  shared_C(h,M)    = h·t_native + t_vloss + t_vgrad + M·(t_cand + (h−1)·t_hvp + h·t_tstep_C)
  breakeven M*     = min{M : shared_X(h,M) < direct(h,M)}
M>4 的 breakeven 由组件成本外推，明确标注"基于实测组件的成本模型外推"。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "research/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import importlib.util

def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "a0g_runner", REPO_ROOT / "research/scripts/a0g_decision_utility_v01.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()

HORIZONS = RUNNER.HORIZONS
N_DATA_GROUPS = RUNNER.N_DATA_GROUPS
PRIMARY_HORIZON = 4
DEFAULT_STATES = ((2038, 46), (2039, 64), (2041, 82))


# ---------------------------------------------------------------------------
# 计时器：每键独立样本列表，前后 cuda synchronize
# ---------------------------------------------------------------------------

class SampleTimers:
    def __init__(self, torch_):
        self._torch = torch_
        self.samples = {}
        self.peak = {}

    def start(self):
        self._torch.cuda.synchronize()
        return time.perf_counter()

    def tick(self, key: str, t0: float):
        self._torch.cuda.synchronize()
        self.samples.setdefault(key, []).append(time.perf_counter() - t0)

    def snapshot_peak(self, key: str):
        # 与 runner 的 Timers.snapshot_peak 等价：记录显存峰值（与计时无关，分支/tangent
        # pass 会调用；计时副本只需提供同签名方法，峰值可顺带保留供核查）。
        self.peak[key] = max(
            self.peak.get(key, 0.0),
            float(self._torch.cuda.max_memory_allocated()),
        )


# ---------------------------------------------------------------------------
# 计时版切线 pass：对 runner._run_tangent_pass 的忠实副本，仅改计时键结构。
# 数学操作序列与原 pass 完全一致（值级一致性由 VERIFY 对冻结 CSV 抽查保证）。
# ---------------------------------------------------------------------------

def _timing_tangent_pass(*, torch, model, optimizer, scaler, autocast_ctx, dataset,
                         named, checkpoint_path, batches, stream_seed,
                         cand_batches_by_j, val_score_tensors, val_eval_tensors, v_t,
                         accumulation_steps, device, grad_clip, base_lr,
                         global_micro_step, total_micro_steps, get_lr, timers):
    R = RUNNER
    R.load_full_training_state(checkpoint_path, model, optimizer, scaler,
                               restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    R.seed_future_stream(int(stream_seed))

    zero = R._zeros_like_named(named)
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
    clip_trace = []

    for horizon in range(1, max_h + 1):
        t0 = timers.start()
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = R.set_native_micro_lr(
                optimizer, get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr),
            )
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = R.stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                rng_before_native_forward = R.capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()
        timers.tick(f"native_fwd_bwd@h{horizon}", t0)

        t0 = timers.start()
        scaler.unscale_(optimizer)
        native_gradient = R._snapshot_grads(named)
        native_gradient_norm = R._norm(native_gradient)
        timers.tick(f"native_unscale@h{horizon}", t0)

        if horizon == 1:
            for j, cand_batches in cand_batches_by_j.items():
                t0 = timers.start()
                cand_tensors = R._stack_all(dataset, cand_batches, device)
                rng_snap = R.capture_rng_state()
                g_by_j[j] = R._grad_on_batches(
                    torch, model, named, cand_tensors, autocast_ctx,
                    accumulation_steps,
                )
                a_by_j[j] = {
                    name: g_by_j[j][name] - native_gradient[name]
                    for name, _ in named
                }
                R.restore_rng_state(rng_snap)
                del cand_tensors
                timers.tick(f"cand_grad@h{horizon}", t0)
            missing = set(cand_batches_by_j) - set(g_by_j)
            if missing:
                raise RuntimeError(f"candidate gradient missing for groups: {sorted(missing)}")

        hvp_by_j = {}
        if horizon > 1:
            for j in cand_batches_by_j:
                t0 = timers.start()
                rng_after_native_gradient = R.capture_rng_state()
                R.restore_rng_state(rng_before_native_forward)
                hvp_j, hvp_first_grad = R._hvp(
                    torch, model, named, batch_tensors, c_theta[j], autocast_ctx,
                    accumulation_steps,
                )
                R.restore_rng_state(rng_after_native_gradient)
                hvp_by_j[j] = hvp_j
                gap = R._difference_norm(hvp_first_grad, native_gradient)
                timers.tick(f"hvp@h{horizon}", t0)

        t0 = timers.start()
        native_preclip = None
        native_clipped_gradient = None
        for j in cand_batches_by_j:
            if native_preclip is None:
                tc0 = timers.start()
                native_preclip = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(grad_clip)).detach().cpu()
                )
                native_clipped_gradient = R._snapshot_grads(named)
                timers.tick(f"native_clip@h{horizon}", tc0)
            tb0 = timers.start()
            tB = a_by_j[j] if horizon == 1 else {name: zero[name] for name, _ in named}
            b_clip = R.clip_jvp(
                gradient_by_name=native_gradient,
                tangent_gradient_by_name=tB,
                max_norm=grad_clip,
            )
            if abs(native_preclip - float(b_clip["clip_before_norm"])) > 1e-5:
                raise RuntimeError(
                    "analytic/native preclip norm mismatch: "
                    f"native={native_preclip!r} analytic={b_clip['clip_before_norm']!r} "
                    f"horizon={horizon}"
                )
            b_next = R.adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=b_clip["clip_jvp"],
                tangent_theta_by_name=b_theta[j],
                tangent_m_by_name=b_m[j],
                tangent_v_by_name=b_v[j],
            )
            timers.tick(f"tangent_B@h{horizon}", tb0)
            b_theta[j], b_m[j], b_v[j] = (
                b_next["tangent_theta"], b_next["tangent_m"], b_next["tangent_v"],
            )
            if horizon == 1:
                r1_theta[j] = {k: v.clone() for k, v in b_theta[j].items()}

            tc0 = timers.start()
            tC = a_by_j[j] if horizon == 1 else hvp_by_j[j]
            c_clip = R.clip_jvp(
                gradient_by_name=native_gradient,
                tangent_gradient_by_name=tC,
                max_norm=grad_clip,
            )
            c_next = R.adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
                tangent_theta_by_name=c_theta[j],
                tangent_m_by_name=c_m[j],
                tangent_v_by_name=c_v[j],
            )
            timers.tick(f"tangent_C@h{horizon}", tc0)
            c_theta[j], c_m[j], c_v[j] = (
                c_next["tangent_theta"], c_next["tangent_m"], c_next["tangent_v"],
            )

            if horizon in HORIZONS:
                r_H_B[j][str(horizon)] = {
                    k: v.detach().clone() for k, v in b_theta[j].items()
                }
                r_H_C[j][str(horizon)] = {
                    k: v.detach().clone() for k, v in c_theta[j].items()
                }
        timers.tick(f"tangent_loop_wall@h{horizon}", t0)

        t0 = timers.start()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        timers.tick(f"native_step@h{horizon}", t0)

        if horizon in HORIZONS:
            t0 = timers.start()
            rng_snap = R.capture_rng_state()
            baseline_val_score[str(horizon)] = R._val_loss_only(
                torch, model, val_score_tensors, autocast_ctx)
            baseline_val_eval[str(horizon)] = R._val_loss_only(
                torch, model, val_eval_tensors, autocast_ctx)
            R.restore_rng_state(rng_snap)
            timers.tick(f"val_loss@h{horizon}", t0)

            t0 = timers.start()
            rng_snap = R.capture_rng_state()
            v_score_grad[str(horizon)] = R._grad_on_batches(
                torch, model, named, val_score_tensors, autocast_ctx,
                len(val_score_tensors),
            )
            R.restore_rng_state(rng_snap)
            timers.tick(f"val_grad@h{horizon}", t0)

            clip_trace.append({
                "horizon": horizon,
                "clip_active": bool(native_preclip > grad_clip),
                "preclip_norm": float(native_preclip),
                "lr": float(current_lr),
            })
        del batch_tensors, native_gradient, native_clipped_gradient

    t0 = timers.start()
    for horizon in HORIZONS:
        hk = str(horizon)
        v_h = v_score_grad[hk]
        for j in cand_batches_by_j:
            hhat["B"][j][hk] = R.project_tangent(v_h, r_H_B[j][hk])
            hhat["C"][j][hk] = R.project_tangent(v_h, r_H_C[j][hk])
            hhat["alignment"][j][hk] = R.project_tangent(v_t, r1_theta[j])
    timers.tick("hhat_projection@all", t0)

    return {
        "baseline_val_score": baseline_val_score,
        "baseline_val_eval": baseline_val_eval,
        "v_score_grad": v_score_grad,
        "hhat": hhat,
        "g_by_j": g_by_j,
        "clip_trace": clip_trace,
    }


def _nvidia(args_):
    try:
        out = subprocess.run(args_, capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def gpu_state(device_index: int):
    util = _nvidia([
        "nvidia-smi", f"-i", str(device_index),
        "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
    ])
    apps = _nvidia([
        "nvidia-smi", f"-i", str(device_index),
        "--query-compute-apps=pid", "--format=csv,noheader",
    ])
    pids = [int(x) for x in apps.split() if x.strip().isdigit()]
    try:
        u, mem = [int(v) for v in util.split(",")]
    except Exception:
        u, mem = -1, -1
    return u, mem, pids


def wait_for_exclusive(device_index: int, max_wait_s: int, log):
    our = os.getpid()
    t_end = time.time() + max_wait_s
    n_poll = 0
    while True:
        n_poll += 1
        utils, others = [], []
        for _ in range(3):
            u, _mem, pids = gpu_state(device_index)
            utils.append(u)
            others.extend(p for p in pids if p != our)
            time.sleep(2)
        med_util = statistics.median(utils)
        if med_util < 10 and not others:
            log(f"GATE_PASS util_median={med_util:.0f}% other_compute_pids=0")
            return True
        if time.time() > t_end:
            log(f"GATE_TIMEOUT after {max_wait_s}s util_median={med_util:.0f}% "
                f"others={sorted(set(others))}")
            return False
        if n_poll % 15 == 1:
            log(f"WAIT_EXCLUSIVE util_median={med_util:.0f}% "
                f"other_compute_pids={sorted(set(others))} "
                f"elapsed={int(time.time() - (t_end - max_wait_s))}s")


# ---------------------------------------------------------------------------
# 统计 + 成本模型
# ---------------------------------------------------------------------------

def key_stats(samples):
    s = sorted(samples)
    n = len(s)
    q = statistics.quantiles(s, n=4) if n >= 4 else [None, None, None]
    return {
        "n": n,
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if n >= 2 else 0.0,
        "median": statistics.median(s),
        "q1": q[0], "q3": q[2],
        "iqr": (q[2] - q[0]) if n >= 4 else None,
        "min": s[0], "max": s[-1],
    }


def cost_tables(measured):
    """返回 (cost_rows, breakeven)；M∈{1,2,4} 实测内插，M∈{8,16,32} 外推。"""
    t_native = measured["t_native_step"]
    t_vloss = measured["t_val_loss_one"]
    t_vgrad = measured["t_val_grad_one"]
    t_cand = measured["t_cand_grad_one"]
    t_hvp = measured["t_hvp_one"]
    t_tB = measured["t_tangent_B_one"]
    t_tC = measured["t_tangent_C_one"]

    def direct(h, m):
        return h * t_native + t_vloss + m * (h * t_native + t_vloss)

    def direct_fd(h, m):
        return h * t_native + t_vloss + m * 2 * (h * t_native + t_vloss)

    def shared_b(h, m):
        return h * t_native + t_vloss + t_vgrad + m * (t_cand + h * t_tB)

    def shared_c(h, m):
        return h * t_native + t_vloss + t_vgrad + m * (
            t_cand + max(h - 1, 0) * t_hvp + h * t_tC)

    rows = []
    for h in HORIZONS:
        for m in (1, 2, 4):
            d, dfd = direct(h, m), direct_fd(h, m)
            sb, sc = shared_b(h, m), shared_c(h, m)
            for method, val in (("direct_replay", d), ("direct_replay_fd", dfd),
                                ("shared_tangent_B", sb), ("shared_tangent_C", sc)):
                base = d
                rows.append({
                    "record_type": "cost_model", "H": h, "M_candidates": m,
                    "method": method, "total_s": val,
                    "per_candidate_s": val / m,
                    "vs_direct_speedup": base / val if val > 0 else None,
                    "extrapolated": False,
                })
        def be(fn):
            for m in range(1, 65):
                if fn(h, m) < direct(h, m):
                    return m
            return None
        for method, fn in (("shared_tangent_B", shared_b), ("shared_tangent_C", shared_c)):
            rows.append({
                "record_type": "breakeven", "H": h,
                "M_candidates": be(fn), "method": method,
                "extrapolated": be(fn) is not None and be(fn) > 4,
            })
    for h in HORIZONS:
        for m in (8, 16, 32):
            d = direct(h, m)
            for method, val in (("shared_tangent_B", shared_b(h, m)),
                                ("shared_tangent_C", shared_c(h, m))):
                rows.append({
                    "record_type": "cost_model_extrapolated", "H": h,
                    "M_candidates": m, "method": method, "total_s": val,
                    "per_candidate_s": val / m,
                    "vs_direct_speedup": d / val if val > 0 else None,
                    "extrapolated": True,
                })
    return rows


# ---------------------------------------------------------------------------
# COST_ONLY_BENCHMARK.md 渲染
# ---------------------------------------------------------------------------

def _f(x, nd=6):
    if x is None:
        return "—"
    return f"{float(x):.{nd}g}"


def _render_md(s: dict) -> str:
    gm = s["gpu_meta"]
    meas = s["measured_median_seconds"]
    ks = s["key_stats"]
    lines = [
        "# COST ONLY BENCHMARK — FR_DECISION_UTILITY_V01（§12 成本核算）",
        "",
        "本基准只测 GPU 时间，不产出或修改任何科学结果。计时副本的数值输出经与冻结的",
        "`level1_local_derivative.csv` 抽查一致（见文末 VERIFY）。",
        "",
        "## 环境",
        "",
        f"- GPU: {gm['gpu_name']}（物理序号 {gm['physical_gpu_index']}，device {gm['device_arg']}）",
        f"- PyTorch {gm['torch_version']} / CUDA {gm['cuda_version']}",
        f"- 独占性: {gm['exclusivity']}（正式计时前无其他计算进程，util 中位数 <10%）",
        f"- 计时后 util {gm.get('util_after')}% / 显存 {gm.get('memory_used_mib_after')} MiB / "
        f"其他计算进程 {gm.get('other_compute_pids_after')}",
        f"- 目标状态: {['%d/%d' % tuple(t) for t in s['targets']]}，每状态 8 streams"
        f"（n={s['n_formal_stream_samples']} stream 样本，另有每状态 1 次 warm-up 弃用）",
        f"- 总墙钟: {gm['wall_seconds']:.0f}s；primary lambda = {s['lambda_primary']}",
        "",
        "## 计时方法",
        "",
        "- 每计算组成独立计时键；每区间前后 `torch.cuda.synchronize()` 后读 `time.perf_counter()`；",
        "- 每键记录 total/count/mean/std/median/IQR，**分母一律为实际 tick 数**；",
        "- 分支引擎（真实回放）与切线 pass 使用分离 timers，无共用键污染；",
        "- 主报告口径：**median [IQR]**，同时保留 mean±std（cost_summary.json）；",
        "- 每 horizon 完整 native step = fwd_bwd + unscale + clip + step 四键之和（逐 horizon 成对）。",
        "",
        "## 实测组件（median [IQR] 秒）",
        "",
        "| 组件 | median | IQR | mean±std | n |",
        "|---|---:|---:|---:|---:|",
    ]
    name_map = {
        "native_full_step": "t_native_step（完整 native micro-step）",
        "cand_grad": "t_cand_grad_one（单候选 g_j）",
        "hvp": "t_hvp_one（单候选单 horizon HVP）",
        "tangent_B": "t_tangent_B_one（单候选单 horizon B 切线）",
        "tangent_C": "t_tangent_C_one（单候选单 horizon C 切线）",
        "val_loss": "t_val_loss_one（endpoint V 双 split loss）",
        "val_grad": "t_val_grad_one（endpoint V_score grad）",
        "v_t_compute": "t_v_t（对齐基线 v_t，每 state 一次）",
        "hhat_projection": "hhat 投影（每 stream 一次）",
        "replay_plus": "回放 wall +λ（单候选）",
        "replay_minus": "回放 wall −λ（单候选）",
        "state_load": "state_load（checkpoint 装载，成本公式不含）",
        "native_fwd_bwd": "[diag] fwd_bwd",
        "native_unscale": "[diag] unscale",
        "native_clip": "[diag] clip_grad_norm_",
        "native_step": "[diag] scaler.step",
    }
    for key in ["native_full_step", "cand_grad", "hvp", "tangent_B", "tangent_C",
                "val_loss", "val_grad", "v_t_compute", "hhat_projection",
                "replay_plus", "replay_minus", "state_load"]:
        st = ks.get(key)
        if not st:
            continue
        lines.append(
            f"| {name_map.get(key, key)} | {_f(st['median'])} | "
            f"{_f(st.get('iqr'))} | {_f(st['mean'])}±{_f(st.get('std'))} | {st['n']} |"
        )
    lines.extend([
        "",
        "回放净成本（公式用）: t_replay_net = wall − state_load",
        f"- +λ: {_f(meas['t_replay_plus_net'])}s；−λ: {_f(meas['t_replay_minus_net'])}s",
        "",
        "## 成本公式",
        "",
        "```",
        "direct(h,M)    = h·t_native + t_vloss + M·(h·t_native + t_vloss)",
        "direct_fd(h,M) = h·t_native + t_vloss + M·2·(h·t_native + t_vloss)   # ±λ 两次回放",
        "shared_B(h,M)  = h·t_native + t_vloss + t_vgrad + M·(t_cand + h·t_tstep_B)",
        "shared_C(h,M)  = h·t_native + t_vloss + t_vgrad + M·(t_cand + (h−1)·t_hvp + h·t_tstep_C)",
        "```",
        "",
        "注：shared_B 无 HVP（B 切线在 horizon>1 用零输入方向）；shared_C 的 HVP 按候选×horizon 计费。",
        "若同一管线同时评估 B 与 C，HVP 计算实际被两者共享（此处按 C 全价、B 零价归属，",
        "对 B 略有高估、对 C 全计，是保守口径）。t_tangent_*_one 均含 clip_jvp+adamw_tangent_step。",
        "",
        "## 成本表（M∈{1,2,4} 实测组件内插）",
        "",
    ])
    for h in HORIZONS:
        lines.extend([
            f"### H={h}",
            "",
            "| M | 方法 | total s | s/候选 | vs direct speedup |",
            "|---:|---|---:|---:|---:|",
        ])
        for m in (1, 2, 4):
            rows = [r for r in s["cost_rows"]
                    if r.get("record_type") == "cost_model"
                    and int(r["H"]) == h and int(r["M_candidates"]) == m]
            for r in rows:
                lines.append(
                    f"| {m} | {r['method']} | {_f(r['total_s'])} | "
                    f"{_f(r['per_candidate_s'])} | {_f(r.get('vs_direct_speedup'))} |"
                )
            lines.append("")
        be_rows = [r for r in s["cost_rows"]
                   if r.get("record_type") == "breakeven" and int(r["H"]) == h]
        txt = ", ".join(
            f"{r['method']}: M*={r['M_candidates']}"
            f"{'（外推）' if r.get('extrapolated') else ''}" for r in be_rows)
        lines.extend([f"break-even（shared < direct 的最小 M）: {txt}", ""])
    lines.extend([
        "### M∈{8,16,32}（基于实测组件的成本模型外推，非真实运行）",
        "",
        "| H | M | 方法 | total s | s/候选 | vs direct speedup |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for r in s["cost_rows"]:
        if r.get("record_type") == "cost_model_extrapolated":
            lines.append(
                f"| {r['H']} | {r['M_candidates']} | {r['method']} | "
                f"{_f(r['total_s'])} | {_f(r['per_candidate_s'])} | "
                f"{_f(r.get('vs_direct_speedup'))} |"
            )
    v = s.get("verify") or {}
    lines.extend([
        "",
        "## VERIFY（计时副本 vs 冻结科学 CSV）",
        "",
        f"- state {v.get('state')} stream {v.get('stream')}: "
        f"max |Δhhat| = {_f(v.get('max_abs_diff_vs_frozen_level1'))} "
        f"（tol {v.get('tolerance')}）→ {'PASS' if v.get('pass') else 'FAIL'}",
        "",
        "## 引用",
        "",
        "组件明细: cost_timing_raw.csv；结构化结果: cost_summary.json；",
        "成本表: cost_breakdown.csv。",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", default=str(REPO_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-index", type=int, default=1,
                    help="物理 GPU 序号（nvidia-smi 视角）")
    ap.add_argument("--states", default=",".join(f"{s}:{t}" for s, t in DEFAULT_STATES))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-wait-s", type=int, default=43200)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--skip-branch-timing", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs/fr_decision_utility_v01/cost_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "cost_timing_run.log"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    targets = []
    for tok in args.states.split(","):
        s, t = tok.split(":")
        targets.append((int(s), int(t)))
    log(f"COST_TIMING_START targets={targets} device=GPU{args.device_index}({args.device})")

    cfg = RUNNER._read_json(root / "research/configs/e3c_confirmatory_v07.json")
    if cfg.get("dtype") != "bfloat16" or int(cfg.get("accumulation_steps", -1)) != 2 \
            or float(cfg.get("grad_clip", -1.0)) != 1.0:
        raise RuntimeError("config drift: strict v07 replay requirements violated")
    lam_sel = RUNNER._read_json(root / "outputs/fr_decision_utility_v01/lambda_selection.json")
    lam_primary = float(lam_sel["primary_lambda"])

    if not args.skip_gate:
        if not wait_for_exclusive(args.device_index, args.max_wait_s, log):
            log("COST_TIMING_ABORT no exclusive GPU")
            return 3
    else:
        log("GATE_SKIPPED")

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu_meta = {
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_arg": args.device,
        "physical_gpu_index": args.device_index,
        "exclusivity": "gate_passed" if not args.skip_gate else "skipped",
        "started_epoch": time.time(),
    }
    log(f"GPU {gpu_meta['gpu_name']} | torch {gpu_meta['torch_version']} "
        f"| cuda {gpu_meta['cuda_version']}")

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"),
        device=args.device,
    )
    v07_out = root / cfg["output_dir"]
    data_path = v07_out / "e3c_subset.jsonl"
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = RUNNER.make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    RUNNER.assert_native_adamw(optimizer)
    named = [(n, p) for n, p in RUNNER.raw_model(model).named_parameters() if p.requires_grad]

    partition = RUNNER.load_partition(v07_out / "partition_v07.json")
    data_group_definition = RUNNER._read_json(
        root / "outputs/fr_decision_utility_v01/data_group_definition.json")
    validation_split = RUNNER._read_json(
        root / "outputs/fr_decision_utility_v01/validation_split.json")

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + RUNNER.EVAL_BATCH_SIZE] for i in range(0, len(idx), RUNNER.EVAL_BATCH_SIZE)]

    val_score_tensors = RUNNER._stack_all(dataset, _chunk(validation_split["v_score"]), args.device)
    val_eval_tensors = RUNNER._stack_all(dataset, _chunk(validation_split["v_eval"]), args.device)

    formal_ctxs = RUNNER._load_formal_state_contexts(root, cfg, partition, data_group_definition)
    ctx_by_key = {(c.seed, c.step): c for c in formal_ctxs}
    missing = [t for t in targets if t not in ctx_by_key]
    if missing:
        raise RuntimeError(f"target states not in formal bank: {missing}")

    raw_rows = []
    verify_report = {}

    def drain(timers, state_tag, stream_tag, phase):
        for key, samples in timers.samples.items():
            for i, s in enumerate(samples):
                raw_rows.append({
                    "state": state_tag, "stream": stream_tag, "phase": phase,
                    "key": key, "tick_index": i, "seconds": float(s),
                })
        timers.samples.clear()

    for (seed, step) in targets:
        ctx = ctx_by_key[(seed, step)]
        state_tag = f"formal:{seed}/{step}"
        t0 = timers_vt = SampleTimers(torch)
        tv0 = timers_vt.start()
        v_t = RUNNER._compute_v_t(torch, model, optimizer, scaler, ctx,
                                  val_score_tensors, autocast_ctx, named)
        timers_vt.tick("v_t_compute@state", tv0)
        drain(timers_vt, state_tag, "-", "formal")

        common = dict(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            checkpoint_path=ctx.checkpoint_path,
            val_score_tensors=val_score_tensors, val_eval_tensors=val_eval_tensors,
            accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=get_lr,
        )

        # ---- warm-up：stream 0，样本标记 warmup ----
        wstream = ctx.streams[0]
        wcand = ctx.candidate_batches(wstream["stream_seed"])
        wtimers = SampleTimers(torch)
        _timing_tangent_pass(
            **common, timers=wtimers, batches=wstream["batches"],
            stream_seed=wstream["stream_seed"], cand_batches_by_j=wcand, v_t=v_t,
        )
        drain(wtimers, state_tag, str(wstream["stream_id"]), "warmup")
        log(f"{state_tag} warmup done")

        for stream in ctx.streams:
            cand = ctx.candidate_batches(stream["stream_seed"])
            timers = SampleTimers(torch)
            t_pass = _timing_tangent_pass(
                **common, timers=timers, batches=stream["batches"],
                stream_seed=stream["stream_seed"], cand_batches_by_j=cand, v_t=v_t,
            )
            drain(timers, state_tag, str(stream["stream_id"]), "formal")

            if (seed, step) == targets[0] and int(stream["stream_id"]) == 0:
                frozen = {}
                with (root / "outputs/fr_decision_utility_v01/level1_local_derivative.csv").open() as fh:
                    for row in csv.DictReader(fh):
                        if row["split"] == "formal" and int(row["seed"]) == seed \
                                and int(row["step"]) == step \
                                and int(row["stream_id"]) == 0:
                            frozen[(int(row["j"]), int(row["H"]))] = row
                max_diff = 0.0
                for j in cand:
                    for h in HORIZONS:
                        hk = str(h)
                        fr = frozen[(j, h)]
                        for mkey, ckey in (("B", "hhat_B"), ("C", "hhat_C"),
                                           ("alignment", "hhat_alignment")):
                            d = abs(float(fr[ckey]) - float(t_pass["hhat"][mkey][j][hk]))
                            max_diff = max(max_diff, d)
                verify_report = {
                    "state": state_tag, "stream": 0,
                    "max_abs_diff_vs_frozen_level1": max_diff,
                    "tolerance": 1e-9, "pass": bool(max_diff <= 1e-9),
                }
                log(f"VERIFY hhat-vs-frozen max|diff|={max_diff:.3e} "
                    f"pass={verify_report['pass']}")

            if not args.skip_branch_timing:
                for sign, key in ((-1, "replay_minus"), (+1, "replay_plus")):
                    tt = SampleTimers(torch)
                    b0 = tt.start()
                    RUNNER._run_action_branch(
                        **common, timers=SampleTimers(torch),
                        batches=stream["batches"], stream_seed=stream["stream_seed"],
                        action_j=1, lam=lam_primary, sign=sign,
                        cand_batches_by_j=cand, g_reference=t_pass["g_by_j"][1],
                    )
                    tt.tick(f"{key}@all", b0)
                    drain(tt, state_tag, str(stream["stream_id"]), "formal")
            log(f"{state_tag} stream={stream['stream_id']} timed")

    # ---- state_load 计时（每 state 一次，供回放净成本扣除） ----
    load_samples = []
    for (seed, step) in targets:
        ctx = ctx_by_key[(seed, step)]
        lt = SampleTimers(torch)
        l0 = lt.start()
        RUNNER.load_full_training_state(ctx.checkpoint_path, model, optimizer,
                                        scaler, restore_rng=False)
        lt.tick("state_load@state", l0)
        load_samples.extend(lt.samples["state_load@state"])
    for i, s in enumerate(load_samples):
        raw_rows.append({
            "state": "all", "stream": "-", "phase": "formal",
            "key": "state_load@state", "tick_index": i, "seconds": float(s),
        })

    # ---- 汇总 ----
    formal_rows = [r for r in raw_rows if r["phase"] == "formal"]
    by_key = {}
    for r in formal_rows:
        base = r["key"].split("@")[0]
        by_key.setdefault(base, []).append(r["seconds"])

    stats = {k: key_stats(v) for k, v in sorted(by_key.items())}

    def med(base):
        return stats[base]["median"]

    # 每个 horizon 的完整 native step（fwd_bwd + unscale + clip + step，逐 horizon 成对）
    per_step_samples = []
    pair_index = {}
    for r in formal_rows:
        base, htag = r["key"].split("@")
        pair_index.setdefault((r["state"], r["stream"], htag), {})[base] = r["seconds"]
    for (st, sm, htag), d in pair_index.items():
        if htag.startswith("h"):
            per_step_samples.append(
                d.get("native_fwd_bwd", 0.0) + d.get("native_unscale", 0.0)
                + d.get("native_clip", 0.0) + d.get("native_step", 0.0))
    native_step_stats = key_stats(per_step_samples)
    stats["native_full_step"] = native_step_stats

    t_load = med("state_load")
    measured = {
        "t_native_step": native_step_stats["median"],
        "t_val_loss_one": med("val_loss"),
        "t_val_grad_one": med("val_grad"),
        "t_cand_grad_one": med("cand_grad"),
        "t_hvp_one": med("hvp"),
        "t_tangent_B_one": med("tangent_B"),
        "t_tangent_C_one": med("tangent_C"),
        "t_v_t_compute": med("v_t_compute"),
        "t_replay_plus_wall": med("replay_plus"),
        "t_replay_minus_wall": med("replay_minus"),
        "t_replay_plus_net": med("replay_plus") - t_load,
        "t_replay_minus_net": med("replay_minus") - t_load,
        "t_state_load": t_load,
    }
    cost_rows = cost_tables(measured)

    gpu_meta["finished_epoch"] = time.time()
    gpu_meta["wall_seconds"] = gpu_meta["finished_epoch"] - gpu_meta["started_epoch"]
    u, mem, pids = gpu_state(args.device_index)
    gpu_meta["util_after"] = u
    gpu_meta["memory_used_mib_after"] = mem
    gpu_meta["other_compute_pids_after"] = [p for p in pids if p != os.getpid()]
    # 注意：gpu_meta 在下方 summary 字典（L888 一带）中以 "gpu_meta" 键统一入册，
    # 此处不得提前引用 summary（其尚未定义，会触发 UnboundLocalError）。

    with (out_dir / "cost_timing_raw.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state", "stream", "phase", "key", "tick_index", "seconds"])
        w.writeheader()
        w.writerows(raw_rows)

    with (out_dir / "cost_breakdown.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "record_type", "H", "M_candidates", "method", "total_s",
            "per_candidate_s", "vs_direct_speedup", "extrapolated"])
        w.writeheader()
        for row in cost_rows:
            w.writerow({k: row.get(k) for k in w.fieldnames})

    summary = {
        "gpu_meta": gpu_meta,
        "targets": [list(t) for t in targets],
        "lambda_primary": lam_primary,
        "streams_per_state": RUNNER.N_STREAMS_PER_STATE,
        "n_formal_stream_samples": len({(r["state"], r["stream"]) for r in formal_rows}),
        "verify": verify_report,
        "key_stats": stats,
        "measured_median_seconds": measured,
        "cost_rows": cost_rows,
        "convention": {
            "t_native_step": "median full native micro-step (fwd_bwd+unscale+clip+step) per horizon",
            "t_tangent_B_one": "per (candidate,horizon) optimizer-only tangent (clip_jvp+adamw_tangent_step, B 切线)",
            "t_tangent_C_one": "per (candidate,horizon) full tangent (clip_jvp+adamw_tangent_step, C 切线, 不含 HVP)",
            "t_hvp_one": "per (candidate,horizon>1) HVP",
            "t_replay_*_net": "wall minus state_load（成本公式不含加载，与成本模型约定一致）",
            "formulas": {
                "direct(h,M)": "h*t_native + t_vloss + M*(h*t_native + t_vloss)",
                "direct_fd(h,M)": "h*t_native + t_vloss + M*2*(h*t_native + t_vloss)",
                "shared_B(h,M)": "h*t_native + t_vloss + t_vgrad + M*(t_cand + h*t_tangent_B)",
                "shared_C(h,M)": "h*t_native + t_vloss + t_vgrad + M*(t_cand + (h-1)*t_hvp + h*t_tangent_C)",
            },
        },
    }
    (out_dir / "cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "COST_ONLY_BENCHMARK.md").write_text(_render_md(summary), encoding="utf-8")

    entries = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    log(f"COST_TIMING_DONE measured={json.dumps(measured)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
